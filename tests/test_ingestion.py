"""
Tests validate -> parse -> chunk -> enrich on ONE real PDF from your corpus,
with no Qdrant/Postgres/Redis dependency — isolates whether Docling and the
OpenRouter LLM call actually work before wiring the full graph + Docker.

Usage:
    uv run python scripts/test_ingestion.py path/to/some.pdf

If no path is given, it picks the first PDF it finds under ./corpus/.
"""

import sys
import glob
from pathlib import Path

from graphs.ingestion.nodes.validator import validate_file
from graphs.ingestion.nodes.parser import parse_document
from graphs.ingestion.nodes.chunker import chunk_pages
from graphs.ingestion.nodes.enricher import enrich_chunks

# Anchor to the project root (parent of this file's directory), not the
# current working directory. IDE run configs (e.g. PyCharm) often default
# cwd to the script's own folder, which would otherwise make a relative
# "corpus/**/*.pdf" glob silently look in the wrong place.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_test_pdf() -> str:
    matches = glob.glob(str(PROJECT_ROOT / "corpus" / "**" / "*.pdf"), recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"No PDF found under {PROJECT_ROOT / 'corpus'}. Pass a path explicitly: "
            "python scripts/test_ingestion.py path/to/file.pdf"
        )
    return matches[0]


def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else find_test_pdf()
    print(f"Testing against: {file_path}\n")

    state = {"file_path": file_path}

    print("[1/4] validate_file ...")
    state = validate_file(state)
    print("  OK — validated:", state["validated"])

    print("\n[2/4] parse_document (Docling) — this is the slow one, be patient ...")
    state = parse_document(state)
    print("  OK — got DoclingDocument")
    print("  markdown preview (first 300 chars):")
    print(" ", state["doc_markdown_preview"][:300].replace("\n", " "))

    print("\n[3/4] chunk_pages (HybridChunker) ...")
    state = chunk_pages(state)
    print(f"  OK — {len(state['chunks'])} chunks produced")
    print("  first chunk (page, first 200 chars):")
    first = state["chunks"][0]
    print(f"    page={first['page']}")
    print(f"    text={first['text'][:200].replace(chr(10), ' ')}")

    print("\n[4/4] enrich_chunks (OpenRouter call) — testing on first 3 chunks only ...")
    # Trim to 3 chunks so this test doesn't burn your daily free-tier request cap.
    state["chunks"] = state["chunks"][:3]
    state = enrich_chunks(state)
    for i, c in enumerate(state["chunks"]):
        print(f"\n  chunk {i} (page {c['page']}):")
        print(f"    context : {c['context']}")
        print(f"    original: {c['text'][:150].replace(chr(10), ' ')}")

    print("\nAll four steps ran without errors.")
    print("Next: check the printed context lines above by eye — are they")
    print("actually useful descriptions, or generic filler? That's the")
    print("real test of whether enrichment is worth its cost.")


if __name__ == "__main__":
    main()