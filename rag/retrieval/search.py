from rag.retrieval.bm25 import bm25_index
from rag.retrieval.hybrid import reciprocal_rank_fusion
from rag.retrieval.reranker import reranker  # a Reranker() instance, not a bare function
from rag.retrieval.vectorStore import vector_search
from db.postgres import get_session, Chunk
from config import RETRIEVAL_TOP_K, RERANK_TOP_K


def _ensure_bm25_built() -> None:
    """
    bm25_index is a singleton, built once from Postgres, then reused
    across all queries — not rebuilt per call. Lazy-build on first use.

    "id" here MUST match the Qdrant point id (both are str(Chunk.id)) —
    see indexer.py — so reciprocal_rank_fusion can recognize the same
    chunk coming back from both retrieval methods and actually fuse them,
    instead of treating every vector hit and every BM25 hit as distinct.
    """
    if bm25_index.index is None:
        session = get_session()
        rows = session.query(Chunk).all()
        session.close()

        chunks = [
            {
                "id": str(row.id),
                "content": row.text,
                "document_id": row.document_id,
                "page_number": row.page,
            }
            for row in rows
        ]
        bm25_index.build(chunks)


def search_knowledge_base(query: str) -> list[dict]:
    """
    Hybrid search (vector + BM25) -> RRF fusion -> cross-encoder rerank.
    Returns chunk dicts using hybrid.py's field names: "content" and
    "page_number" (not vectorStore.py's "text"/"page" — those only exist
    on the raw ScoredPoint payload before fusion normalizes everything).
    """
    _ensure_bm25_built()

    vector_results = vector_search(query, top_k=RETRIEVAL_TOP_K)       # raw ScoredPoint list
    bm25_results = bm25_index.search(query, top_k=RETRIEVAL_TOP_K)     # chunk dicts + bm25_score

    fused = reciprocal_rank_fusion(vector_results, bm25_results, top_k=RETRIEVAL_TOP_K)
    return reranker.rerank(query, fused, top_k=RERANK_TOP_K)
