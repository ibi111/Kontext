from langfuse.langchain import CallbackHandler
from db.postgres import get_session


def get_db():
    """Yield a SQLAlchemy session, close it after the request."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def get_langfuse_handler() -> CallbackHandler:
    """
    Fresh CallbackHandler per request so each request gets its own
    trace in Langfuse rather than all requests sharing one trace.
    Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
    from environment automatically.
    """
    return CallbackHandler()
