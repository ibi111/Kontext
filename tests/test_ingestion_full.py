
import sys
import glob
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.postgres import get_session, Document, Chunk
from db.qdrant import client
from config import QDRANT_COLLECTION
from graphs.ingestion.graph import ingestion_graph


def find_test_pdf() -> str:
    project_root = Path(__file__).resolve().parent.parent
    matches = glob.glob(str(project_root / "corpus" / "**" / "*.pdf"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No PDF found under {project_root / 'corpus'}.")
    return matches[0]


def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else find_test_pdf()
    filename = os.path.basename(file_path)
    print(f"Testing full ingestion graph against: {file_path}\n")

    # Create the Document row the graph's indexer step will write chunks
    # against — in the real API, api/main.py does this before invoking the
    # graph (see the /ingest endpoint sketch from the implementation guide).
    session = get_session()
    doc = Document(filename=filename, status="processing")
    session.add(doc)
    session.commit()
    document_id = doc.id
    session.close()
    print(f"Created document row: id={document_id}, status=processing\n")

    print("Running ingestion_graph.invoke(...) — this runs parsing (slow, "
          "real Docling model inference) and enrichment (real OpenRouter "
          "calls, one per chunk) for the WHOLE document, not just a few "
          "chunks like the earlier standalone test. Expect this to take a "
          "while and to use real request budget.\n")

    result = ingestion_graph.invoke({"file_path": file_path, "document_id": document_id})

    print("Graph finished. Result summary:")
    print(f"  chunks produced      : {len(result.get('chunks', []))}")
    print(f"  chunks dropped (junk): {result.get('chunks_dropped_as_junk')}")
    print(f"  chunks indexed       : {result.get('indexed')}")
    parse_errors = result.get("parse_errors", [])
    if parse_errors:
        print(f"  WARNING: {len(parse_errors)} page-level parse error(s) — "
              f"content from those pages is likely missing:")
        for err in parse_errors:
            print(f"    - {err}")
    else:
        print("  parse errors         : none")

    # Verify against the stores directly, not just the graph's own return
    # value — this is the actual proof the write path worked end to end.
    session = get_session()
    db_chunk_count = session.query(Chunk).filter_by(document_id=document_id).count()
    db_doc_status = session.query(Document).filter_by(id=document_id).first().status
    session.close()

    qdrant_count = client.count(
        collection_name=QDRANT_COLLECTION,
        count_filter={"must": [{"key": "document_id", "match": {"value": document_id}}]},
    ).count

    print(f"\nVerification against live stores:")
    print(f"  Postgres chunk rows for this document: {db_chunk_count}")
    print(f"  Postgres document status              : {db_doc_status}")
    print(f"  Qdrant points for this document        : {qdrant_count}")

    assert db_chunk_count == result.get("indexed"), "Postgres chunk count doesn't match indexer's own count"
    assert db_doc_status == "ready", "document status was not flipped to ready"
    assert qdrant_count == result.get("indexed"), "Qdrant point count doesn't match indexer's own count"

    print("\nFull ingestion pipeline verified end to end.")


if __name__ == "__main__":
    main()