"""
Initializes the schema in both stores. Safe to re-run — both init
functions are idempotent (SQLAlchemy's create_all skips existing tables,
init_collection checks before creating).

Usage:
    python scripts/init_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.postgres import init_db, engine
from db.qdrant import init_collection, client
from config import QDRANT_COLLECTION




def main():
    print("Initializing Postgres schema...")
    init_db()
    tables = engine.dialect.get_table_names(engine.connect())
    print(f"  OK — tables present: {tables}")

    print("\nInitializing Qdrant collection...")
    init_collection()
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"  OK — collection '{QDRANT_COLLECTION}' ready "
          f"(vector size={info.config.params.vectors.size}, "
          f"points={info.points_count})")

    print("\nSchema initialized in both stores.")


if __name__ == "__main__":
    main()