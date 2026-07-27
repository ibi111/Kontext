"""
Docling's HybridChunker, tokenization-aware and structure-respecting.
Tables are serialized as markdown pipe-tables (via the custom serializer
below) instead of flattened text, so a financial table survives chunking
as a readable table rather than a wall of unstructured numbers.
"""

from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer

from config import EMBEDDING_MODEL

# Explicit wrapper, not a bare model-name string — passing a string hits
# HybridChunker's deprecated init path and its max_tokens defaulting isn't
# reliable. 512 matches bge-large-en-v1.5's actual max sequence length.
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
    max_tokens=512,
)


class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(),
        )


chunker = HybridChunker(
    tokenizer=tokenizer,
    serializer_provider=MarkdownTableSerializerProvider(),
)

# Docling emits a bare placeholder for images it doesn't extract text from.
# A chunk that's just this placeholder (or near-empty after stripping) has
# zero retrieval value — filter it out before it ever reaches enrichment
# or indexing, rather than spending an LLM call and an embedding on noise.
JUNK_CHUNK_MIN_CHARS = 20
IMAGE_PLACEHOLDER = "<!-- image -->"


def _is_junk(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) < JUNK_CHUNK_MIN_CHARS or stripped == IMAGE_PLACEHOLDER


def chunk_pages(state: dict) -> dict:
    doc = state["docling_document"]
    raw_chunks = list(chunker.chunk(doc))

    chunks = []
    dropped = 0
    for c in raw_chunks:
        if _is_junk(c.text):
            dropped += 1
            continue
        page = None
        if c.meta.doc_items and c.meta.doc_items[0].prov:
            page = c.meta.doc_items[0].prov[0].page_no
        chunks.append({"page": page, "text": c.text})

    return {**state, "chunks": chunks, "chunks_dropped_as_junk": dropped}