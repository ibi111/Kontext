from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_DIM

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def init_collection(vector_size: int = EMBEDDING_DIM) -> None:
    """
    Idempotent: safe to call on every startup. Creates the collection only
    if it doesn't already exist.
    """
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )