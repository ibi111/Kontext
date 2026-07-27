from sentence_transformers import CrossEncoder
from typing import List, Dict
import torch
import logging

logger = logging.getLogger(__name__)


RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"   # multilingual reranker


class Reranker:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading reranker {RERANKER_MODEL} on {device}...")
        self.model = CrossEncoder(
            RERANKER_MODEL,
            device=device,
            max_length=512
        )

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank chunks using cross-encoder.
        Returns top_k chunks sorted by reranker score.
        """
        if not chunks:
            return []

        # Build (query, passage) pairs
        pairs = [(query, chunk["content"]) for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Add reranker score to each chunk
        for chunk, score in zip(chunks, scores):
            chunk["reranker_score"] = float(score)

        # Sort by reranker score descending
        chunks.sort(key=lambda x: x["reranker_score"], reverse=True)

        return chunks[:top_k]


# Singleton
reranker = Reranker()