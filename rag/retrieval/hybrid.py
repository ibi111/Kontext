from typing import List, Dict
from qdrant_client.models import ScoredPoint


def reciprocal_rank_fusion(
    vector_results: List[ScoredPoint],
    bm25_results: List[Dict],
    top_k: int = 20,
    k: int = 60           # RRF constant
) -> List[Dict]:
    """
    Combine vector search and BM25 results using
    Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank)) across all result lists.

    k=60 is the standard value from the original RRF paper.
    Higher k reduces the impact of top-ranked documents.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    # Score vector results
    for rank, result in enumerate(vector_results):
        chunk_id = str(result.id)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        chunk_map[chunk_id] = {
            "id": chunk_id,
            "content": result.payload.get("content", ""),
            "document_id": result.payload.get("document_id", ""),
            "document_name": result.payload.get("document_name", ""),
            "page_number": result.payload.get("page_number"),
            "section": result.payload.get("section", ""),
            "metadata": result.payload.get("metadata", {}),
            "vector_score": result.score
        }

    # Score BM25 results
    for rank, result in enumerate(bm25_results):
        chunk_id = result["id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = {
                "id": chunk_id,
                "content": result.get("content", ""),
                "document_id": result.get("document_id", ""),
                "document_name": result.get("document_name", ""),
                "page_number": result.get("page_number"),
                "section": result.get("section", ""),
                "metadata": result.get("metadata", {}),
                "vector_score": 0.0
            }
        chunk_map[chunk_id]["bm25_score"] = result.get("bm25_score", 0.0)

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    results = []
    for chunk_id in sorted_ids:
        chunk = chunk_map[chunk_id]
        chunk["rrf_score"] = rrf_scores[chunk_id]
        results.append(chunk)

    return results