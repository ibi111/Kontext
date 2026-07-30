"""
Smoke-load MCP tools and optionally run one Tavily search.

Usage:
    uv run python tests/test_mcp_tools.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TAVILY_API_KEY
from kontext_mcp.client import get_mcp_tools


def main():
    if not (TAVILY_API_KEY or "").strip():
        print("SKIP — TAVILY_API_KEY not set in .env")
        sys.exit(0)

    tools = get_mcp_tools()
    print(f"Loaded {len(tools)} MCP tool(s):")
    for t in tools:
        print(f"  - {t.name}: {(t.description or '')[:80]}")

    if not tools:
        print("FAIL — no tools loaded")
        sys.exit(1)

    search = next((t for t in tools if "search" in t.name.lower()), tools[0])
    print(f"\nInvoking {search.name} ...")
    result = search.invoke({"query": "EU AI Act latest news"})
    preview = result if isinstance(result, str) else str(result)
    print(preview[:800])
    print("\nOK")


if __name__ == "__main__":
    main()
