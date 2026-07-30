import uvicorn

from api.logging_config import *

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.postgres import init_db
from db.qdrant import init_collection
from api.routes.query_route import router as query_router
from api.routes.ingest_route import router as ingest_router
from api.routes.chatrooms_route import router as chatrooms_router
from api.routes.health_route import router as health_router




@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Postgres schema...")
    init_db()

    logger.info("Initializing Qdrant collection...")
    init_collection()

    logger.info("App ready.")
    yield

    logger.info("Flushing Langfuse traces...")
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception as e:
        logger.warning(f"Langfuse flush failed: {e}")

    logger.info("Shutdown complete.")


app = FastAPI(
    title="Kontext",
    description="Agentic document intelligence — hybrid RAG with multi-turn memory.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(chatrooms_router)
app.include_router(health_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

