from rag.retrieval.embeddings import embedding_model
from cache.sementicCache import check_cache
from config import CACHE_ENABLED


def cache_check(state: dict) -> dict:
    if not CACHE_ENABLED:
        return {**state, "cache_hit": False}

    cached_answer = check_cache(state["query"], embedding_model.embed_query)
    if cached_answer is not None:
        return {**state, "cache_hit": True, "final_answer": cached_answer}

    return {**state, "cache_hit": False}
