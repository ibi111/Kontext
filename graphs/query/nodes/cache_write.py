from rag.retrieval.embeddings import embedding_model
from cache.sementicCache import write_cache
from config import CACHE_ENABLED


def cache_write(state: dict) -> dict:
    if CACHE_ENABLED and not state.get("cache_hit"):
        write_cache(state["query"], state["final_answer"], embedding_model.embed_query)
    return state
