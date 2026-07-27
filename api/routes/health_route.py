
from fastapi import APIRouter
from db.postgres import get_session
from db.qdrant import client as qdrant_client
from cache.sementicCache import r as redis_client
from config import QDRANT_COLLECTION
from sqlalchemy import text

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health():
    """Check all four backing services."""
    status = {}

    # Postgres
    try:
        session = get_session()
        session.execute(text("SELECT 1"))
        status["postgres"] = "ok"
        session.close()
    except Exception as e:
        session.close()
        status["postgres"] = f"error: {e}"

    # Qdrant
    try:
        info = qdrant_client.get_collection(QDRANT_COLLECTION)
        status["qdrant"] = f"ok — {info.points_count} points"
    except Exception as e:
        status["qdrant"] = f"error: {e}"

    # Redis
    try:
        redis_client.ping()
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = f"error: {e}"

    overall = "ok" if all("ok" in v for v in status.values()) else "degraded"
    return {"status": overall, "services": status}
