from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from db.postgres import get_session, Chunk, Document
from db.qdrant import client
from config import QDRANT_COLLECTION, EMBEDDING_MODEL

embedder = SentenceTransformer(EMBEDDING_MODEL)


def index_chunks(state: dict) -> dict:
    session = get_session()
    chunks = state["chunks"]

    texts = (
        [c["enriched_text"] for c in chunks]
        if "enriched_text" in chunks[0]
        else [c["text"] for c in chunks]
    )
    vectors = embedder.encode(texts, normalize_embeddings=True)

    # Insert Chunk rows first and flush (not commit) to get their real
    # Postgres ids without ending the transaction. The Qdrant point id is
    # then set to that SAME id, stringified — not a random uuid. This is
    # required for hybrid.py's RRF fusion: it matches chunks across the
    # vector and BM25 result lists by id, so both sides must use the same
    # id space for the same chunk, or fusion silently never deduplicates.
    db_chunks = []
    for chunk in chunks:
        db_chunk = Chunk(
            document_id=state["document_id"],
            page=chunk.get("page"),
            text=chunk["enriched_text"] if "enriched_text" in chunk  else chunk["text"],
            qdrant_point_id="",  # filled in below once the real id exists
        )
        session.add(db_chunk)
        db_chunks.append(db_chunk)
    session.flush()  # assigns .id to each Chunk without committing yet

    points = []
    for db_chunk, vector in zip(db_chunks, vectors):
        point_id = db_chunk.id
        db_chunk.qdrant_point_id = str(db_chunk.id)

        points.append(PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload={
                "content": db_chunk.text,
                "page_number": db_chunk.page,
                "document_id": state["document_id"],
            },
        ))

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)

    session.query(Document).filter_by(id=state["document_id"]).update({"status": "ready"})
    session.commit()
    session.close()

    return {**state, "indexed": len(points)}
