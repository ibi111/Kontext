from sentence_transformers import SentenceTransformer
from qdrant_client.models import ScoredPoint

from db.qdrant import client
from config import QDRANT_COLLECTION, EMBEDDING_MODEL

embedder = SentenceTransformer(EMBEDDING_MODEL)


def vector_search(query: str, top_k: int = 20) -> list[ScoredPoint]:
    """
    Returns raw Qdrant ScoredPoint objects, NOT converted dicts.
    hybrid.py's reciprocal_rank_fusion reads .id, .payload, .score directly
    off these objects and builds its own normalized dict internally — it
    does not accept a pre-shaped dict.
    """
    vector = embedder.encode(query, normalize_embeddings=True).tolist()
    response = client.query_points(
        collection_name=QDRANT_COLLECTION, query=vector, limit=top_k
    )
    return response.points
