from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
import json
import re
import logging

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Lowercases, removes punctuation, splits on whitespace.
    Works for both German and English.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


class BM25Index:
    """
    In-memory BM25 index.
    Rebuilt on each server start from PostgreSQL chunk data.
    Appropriate for portfolio scale.
    """

    def __init__(self):
        self.index: Optional[BM25Okapi] = None
        self.chunks: List[Dict] = []   # parallel list to index

    def build(self, chunks: List[Dict]):
        """
        Build index from list of chunk dicts.
        Each dict must have: id, content, document_id, metadata
        """
        if not chunks:
            print("No chunks to index.")
            return

        self.chunks = chunks
        tokenized = [tokenize(c["content"]) for c in chunks]
        self.index = BM25Okapi(tokenized)
        logger.info(f"BM25 index built.")


    def search(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Search BM25 index.
        Returns top_k chunks with bm25_score added.
        """
        if self.index is None or not self.chunks:
            return []

        tokens = tokenize(query)
        scores = self.index.get_scores(tokens)

        # Pair each chunk with its score
        scored = [
            {**chunk, "bm25_score": float(score)}
            for chunk, score in zip(self.chunks, scores)
        ]

        # Sort by score descending, return top_k
        scored.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored[:top_k]

    def add_chunk(self, chunk: Dict):
        """
        Add a single chunk and rebuild index.
        Simple approach — fine at portfolio scale.
        """
        self.chunks.append(chunk)
        tokenized = [tokenize(c["content"]) for c in self.chunks]
        self.index = BM25Okapi(tokenized)

    @property
    def size(self) -> int:
        return len(self.chunks)


# Singleton
bm25_index = BM25Index()