from langchain_core.tools import tool
from rag.retrieval.search import search_knowledge_base
from kontext_mcp.client import get_mcp_tools


@tool
def search_documents(query: str) -> str:
    """
    Search the internal document knowledge base (BMW/Mercedes annual
    reports, EU AI Act, VDA guidelines, supplier compliance documents) for
    relevant passages. Use this whenever the question needs a specific
    fact, figure, table value, or quote from the ingested documents. Do
    not use it for general knowledge you already know, or for questions
    unrelated to the documents.
    """
    results = search_knowledge_base(query)
    if not results:
        return "No relevant passages found."

    formatted = [
        f"(page {r.get('page_number', 'unknown')}) {r['content']}"
        for r in results
    ]
    return "\n\n---\n\n".join(formatted)


AVAILABLE_TOOLS = [search_documents, *get_mcp_tools()]
