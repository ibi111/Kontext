"""
Semantic cache sitting in front of the query agent. Checked right after the
input guardrail, before the agent runs — a hit means zero LLM calls and zero
retrieval for that turn.

Approach: small-scale by design. Entries are stored as JSON in Redis; an
index set tracks which keys exist; cosine similarity against the query is
computed in Python at lookup time (O(n) over cached entries). Fine at the
scale of a demo corpus (dozens to low hundreds of distinct questions) — not
how you'd do this at real production scale, where you'd reach for a vector
index (e.g. Redis's own vector search module) instead of a linear scan.
"""

import json
import uuid
from typing import Callable

import numpy as np
import redis

from config import REDIS_HOST, REDIS_PORT, CACHE_SIMILARITY_THRESHOLD, CACHE_TTL_SECONDS

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

INDEX_KEY = "semantic_cache:index"


def _cosine(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-8))


def check_cache(query: str, embed_fn: Callable[[str], list[float]]) -> str | None:
    """
    Returns a cached answer if a semantically similar past query exists
    above CACHE_SIMILARITY_THRESHOLD, otherwise None.

    embed_fn is passed in rather than imported directly to avoid a circular
    import with the embedding model used in the ingestion indexer — both
    should use the same model in practice (config.EMBEDDING_MODEL).
    """
    query_vec = embed_fn(query)
    entry_ids = r.smembers(INDEX_KEY)

    for entry_id in entry_ids:
        raw = r.get(f"semantic_cache:{entry_id}")
        if raw is None:
            # Entry expired (TTL) but the index set wasn't cleaned up yet.
            r.srem(INDEX_KEY, entry_id)
            continue
        entry = json.loads(raw)
        if _cosine(query_vec, entry["embedding"]) >= CACHE_SIMILARITY_THRESHOLD:
            return entry["answer"]

    return None


def write_cache(query: str, answer: str, embed_fn: Callable[[str], list[float]]) -> None:
    entry_id = str(uuid.uuid4())
    entry = {"query": query, "embedding": embed_fn(query), "answer": answer}
    r.setex(f"semantic_cache:{entry_id}", CACHE_TTL_SECONDS, json.dumps(entry))
    r.sadd(INDEX_KEY, entry_id)