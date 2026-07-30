
from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("kontext-web-search")

TAVILY_URL = "https://api.tavily.com/search"
MAX_RESULTS = 5


def _search_tavily(query: str, max_results: int = MAX_RESULTS) -> list[dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to the environment or .env."
        )

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "include_answer": False,
        "search_depth": "basic",
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(TAVILY_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return data.get("results") or []


def _format_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No web results found."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip() or "(no title)"
        url = (r.get("url") or "").strip() or "(no url)"
        snippet = (r.get("content") or r.get("snippet") or "").strip()
        if len(snippet) > 400:
            snippet = snippet[:400].rstrip() + "…"
        lines.append(f"{i}. {title}\n   URL: {url}\n   {snippet}")
    return "\n\n".join(lines)


@mcp.tool()
def web_search(query: str) -> str:
    """
    Search the public web for current or external information
    (news, official pages, facts not in the internal document corpus).
    Prefer internal document search when the question is about ingested
    PDFs (BMW/Mercedes reports, EU AI Act, VDA, supplier docs).
    """
    query = (query or "").strip()
    if not query:
        return "Error: empty query."

    try:
        results = _search_tavily(query)
    except httpx.HTTPStatusError as e:
        return f"Web search failed (HTTP {e.response.status_code}): {e.response.text[:300]}"
    except Exception as e:
        return f"Web search failed: {e}"

    return _format_results(results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
