"""
Load MCP tools for the query agent via langchain-mcp-adapters.

Prefers Tavily's remote MCP when TAVILY_API_KEY is set; falls back to the
local stdio server (kontext_mcp/web_search_server.py). Returns [] if
unavailable so the app still starts without web search.

"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from config import ROOT, TAVILY_API_KEY

logger = logging.getLogger(__name__)

_SERVER_PATH = Path(__file__).resolve().parent / "web_search_server.py"
_cached_tools: list[Any] | None = None


def _server_config() -> dict[str, dict[str, Any]]:
    key = (TAVILY_API_KEY or "").strip()
    if key:
        return {
            "tavily": {
                "transport": "streamable_http",
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={key}",
            }
        }
    return {
        "web_search": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(_SERVER_PATH)],
            "env": {
                **{k: v for k, v in os.environ.items() if v is not None},
                "TAVILY_API_KEY": key,
            },
            "cwd": str(ROOT),
        }
    }


def _run_async(factory):
    """
    Run ``factory()`` which must return a coroutine.
    Coroutine is always created on the thread that owns the event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    import concurrent.futures

    def _in_thread():
        return asyncio.run(factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_in_thread).result(timeout=120)


def _wrap_for_sync(tool: Any) -> Any:
    """
    MCP StructuredTools are async-only (ainvoke). LangGraph ToolNode and
    scripts often call .invoke() — wrap with a sync func that bridges to async.
    """
    from langchain_core.tools import StructuredTool

    async def _acall(**kwargs: Any) -> Any:
        return await tool.ainvoke(kwargs)

    def _call(**kwargs: Any) -> Any:
        return _run_async(lambda: tool.ainvoke(kwargs))

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.args_schema,
        func=_call,
        coroutine=_acall,
        # MCP tools use 'content_and_artifact'; ainvoke returns just the
        # content (list of blocks) without a tool_call_id, which is valid
        # ToolMessage content. Keep the wrapper at plain 'content'.
        response_format="content",
        metadata=getattr(tool, "metadata", None) or {},
    )


async def _load_tools_async() -> list[Any]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    config = _server_config()
    client = MultiServerMCPClient(config)
    tools = await client.get_tools()
    wrapped = [_wrap_for_sync(t) for t in tools]
    logger.info("Loaded %d MCP tool(s): %s", len(wrapped), [t.name for t in wrapped])
    return wrapped


def get_mcp_tools() -> list[Any]:
    """Sync, cached load of MCP tools. Safe to call at import time."""
    global _cached_tools
    if _cached_tools is not None:
        return _cached_tools

    if not (TAVILY_API_KEY or "").strip():
        logger.warning("TAVILY_API_KEY unset — MCP web tools not loaded")
        _cached_tools = []
        return _cached_tools

    try:
        _cached_tools = _run_async(_load_tools_async)
    except Exception:
        logger.exception("Failed to load MCP tools — continuing without them")
        _cached_tools = []

    return _cached_tools
