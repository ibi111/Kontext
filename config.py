import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- LLM (generation) — OpenRouter, OpenAI-compatible API ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Pick current free IDs from https://openrouter.ai/models (filter: Price = Free)
# right before a demo run — the free lineup rotates without notice.
CHAT_MODEL_FAST = os.getenv("CHAT_MODEL_FAST", "openai/gpt-oss-20b:free")
CHAT_MODEL_SMART = os.getenv("CHAT_MODEL_SMART", "openai/gpt-oss-120b:free")
CHAT_MODEL_FALLBACK = os.getenv("CHAT_MODEL_FALLBACK", "openrouter/free")

# --- Embeddings — local, free, no API key needed ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # bge-large-en-v1.5 output size

# --- Web search tool (MCP) ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- Postgres ---
POSTGRES_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

# --- Qdrant ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

# --- Redis (semantic cache) ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# protects against burning that budget on repeated/similar questions
# during testing and demos.
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(7 * 24 * 60 * 60)))  # 7 days

# --- Langfuse (observability) ---
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --- Pipeline tuning ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 5
MAX_REFLECTION_RETRIES = 2

# Skip enrichment on short chunks while testing to conserve OpenRouter
ENRICH_MIN_CHUNK_CHARS = int(os.getenv("ENRICH_MIN_CHUNK_CHARS", "0"))

# --- Paths ---
DATA_DIR = ROOT / "corpus"
UPLOADS_DIR = DATA_DIR / "uploads"
DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)