from sentence_transformers import SentenceTransformer
from typing import List, Union
import torch

import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-m3"

class EmbeddingModel:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {MODEL_NAME} on {device}...")
        self.model = SentenceTransformer(MODEL_NAME, device=device)
        logger.info(f"Finished loading {MODEL_NAME} on {device}...")

    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        Embed a single string or list of strings.
        Always returns List[List[float]].
        """
        if isinstance(text, str):
            text = [text]
        embeddings = self.model.encode(
            text,
            normalize_embeddings=True,   # cosine similarity ready
            batch_size=32,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Single query embedding — returns flat list."""
        return self.embed(query)[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed documents."""
        return self.embed(texts)


# Singleton — load once at startup
embedding_model = EmbeddingModel()

