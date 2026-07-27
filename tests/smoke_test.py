"""
Full smoke test: writes and reads back real data in all three stores.
Run this after scripts/init_db.py — that only creates schema, this
confirms the write path actually works end to end.

Usage:
    python scripts/smoke_test.py
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.postgres import (
    get_session, create_chatroom, add_message, get_messages, Chunk, Document,
)
from db.qdrant import client, init_collection
from cache.sementicCache import check_cache, write_cache
from config import QDRANT_COLLECTION, EMBEDDING_DIM


def test_postgres_chat():
    print("[1/4] Postgres — chatroom + message round-trip...")
    session_id = f"smoke-test-{uuid.uuid4()}"
    chatroom_id = create_chatroom(session_id, title="Smoke test chat")
    add_message(chatroom_id, role="user", content="Hello, does this work?")
    add_message(
        chatroom_id, role="tool", tool_name="search_documents",
        tool_args={"query": "test"}, tool_result={"chunks": ["dummy result"]},
    )
    add_message(chatroom_id, role="assistant", content="Yes, it works.", model_used="test-model")

    messages = get_messages(chatroom_id)
    assert len(messages) == 3, f"expected 3 messages, got {len(messages)}"
    assert messages[1]["tool_name"] == "search_documents"
    print(f"  OK — chatroom {chatroom_id}, {len(messages)} messages round-tripped correctly")


def test_postgres_documents():
    print("\n[2/4] Postgres — document + chunk round-trip...")
    session = get_session()
    doc = Document(filename="smoke_test.pdf", status="ready")
    session.add(doc)
    session.commit()

    chunk = Chunk(
        document_id=doc.id, page=1, section="test",
        text="This is a smoke test chunk.", qdrant_point_id=str(uuid.uuid4()),
    )
    session.add(chunk)
    session.commit()

    fetched = session.query(Chunk).filter_by(document_id=doc.id).first()
    assert fetched.text == "This is a smoke test chunk."
    session.close()
    print(f"  OK — document {doc.id}, chunk {fetched.id} round-tripped correctly")


def test_qdrant():
    print("\n[3/4] Qdrant — vector upsert + search round-trip...")
    init_collection(vector_size=EMBEDDING_DIM)

    point_id = str(uuid.uuid4())
    dummy_vector = [0.1] * EMBEDDING_DIM
    client.upsert(QDRANT_COLLECTION, points=[{
        "id": point_id, "vector": dummy_vector,
        "payload": {"text": "smoke test point", "page": 1},
    }])

    # .search() is deprecated in current qdrant-client — .query_points() is
    # the current API. Note the different return shape: query_points()
    # returns a response object with a .points list, not a bare list.
    response = client.query_points(
        collection_name=QDRANT_COLLECTION, query=dummy_vector, limit=1
    )
    results = response.points
    assert len(results) == 1
    assert results[0].payload["text"] == "smoke test point"
    print(f"  OK — point {point_id} upserted and retrieved via query_points")


def test_redis_cache():
    print("\n[4/4] Redis — semantic cache write + read round-trip...")

    def dummy_embed(text: str) -> list[float]:
        return [0.5] * 16

    query = f"smoke test query {uuid.uuid4()}"
    write_cache(query, "smoke test answer", dummy_embed)
    result = check_cache(query, dummy_embed)
    assert result == "smoke test answer", f"expected cache hit, got {result}"
    print("  OK — cache write and semantic lookup both work")


def main():
    test_postgres_chat()
    test_postgres_documents()
    test_qdrant()
    test_redis_cache()
    print("\nAll four stores verified end to end. Safe to move on to the indexer node.")


if __name__ == "__main__":
    main()