from typing import TypedDict, Any, Optional
from langgraph.graph import StateGraph, END
from config import ENRICH_CHUNKS

from .nodes.validator import validate_file
from .nodes.parser import parse_document
from .nodes.chunker import chunk_pages
from .nodes.enricher import enrich_chunks
from .nodes.indexer import index_chunks


class IngestionState(TypedDict, total=False):
    file_path: str
    document_id: int
    validated: bool
    docling_document: Any          # DoclingDocument, not JSON-serializable —
                                    # fine for a single in-memory invoke(),
                                    # would need handling if a checkpointer
                                    # (e.g. Postgres-backed persistence) is
                                    # added later.
    doc_markdown_preview: str
    chunks: list
    chunks_dropped_as_junk: int
    indexed: int


def build_ingestion_graph():
    g = StateGraph(IngestionState)

    g.add_node("validate", validate_file)
    g.add_node("parse", parse_document)
    g.add_node("chunk", chunk_pages)

    if ENRICH_CHUNKS:
        g.add_node("enrich", enrich_chunks)
    g.add_node("index", index_chunks)

    g.set_entry_point("validate")
    g.add_edge("validate", "parse")
    g.add_edge("parse", "chunk")
    if ENRICH_CHUNKS:
        g.add_edge("chunk", "enrich")
        g.add_edge("enrich", "index")
    else:
        g.add_edge("chunk", "index")
    g.add_edge("index", END)

    return g.compile()


ingestion_graph = build_ingestion_graph()