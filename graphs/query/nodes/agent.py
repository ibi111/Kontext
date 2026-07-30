from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, CHAT_MODEL_FAST, CHAT_MODEL_SMART,
)
from .tools import AVAILABLE_TOOLS

SYSTEM_PROMPT = """You are a document intelligence assistant for enterprise
technical and corporate documents (BMW/Mercedes annual reports, EU AI Act,
VDA guidelines, supplier compliance documents).

Tools:
- `search_documents` — internal ingested PDFs. Prefer this first for any
  question the corpus could answer (regulation, automotive, compliance,
  figures from reports). Search even if you think you already know.
- Tavily web tools (`tavily_search`, `tavily_extract`, and related) — live
  web. Use for current events, external facts, or when internal search is
  empty / insufficient. Cite URLs when you use web results.

If the first internal search is weak, reformulate once or use web search
before giving up. For pure arithmetic or topics with no corpus or web need,
answer directly.

Always cite page numbers for document passages. If you still cannot find
the answer, say so plainly rather than guessing.
"""




def _make_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        max_tokens=600,
    ).bind_tools(AVAILABLE_TOOLS)


# One bound-tools LLM per tier, built once at import time rather than per
# call — bind_tools() attaches the tool schema, no need to redo that work
# on every request.
llm_fast = _make_llm(CHAT_MODEL_FAST)
llm_smart = _make_llm(CHAT_MODEL_SMART)


def classify_complexity(query: str) -> str:
    """
    Route between the fast and smart free-tier models. With both tiers at
    $0 cost on OpenRouter, this is a latency/reliability choice, not a
    cost-saving one: the smaller model responds faster and is less likely
    to hit rate limits, so default to it unless the query looks like it
    needs deeper reasoning.
    """
    long_query = len(query.split()) > 15
    needs_reasoning = any(w in query.lower() for w in ["compare", "why", "explain", "analyze"])
    return CHAT_MODEL_SMART if (long_query or needs_reasoning) else CHAT_MODEL_FAST


def agent_node(state: dict) -> dict:
    existing = state.get("messages", [])
    new_messages = []

    if not existing:
        # First turn — seed system + user message. The state's "messages"
        # key uses LangGraph's add_messages reducer, so whatever list we
        # return here gets appended to (not replacing) what's already in
        # state, which is correctly empty on this first call.
        new_messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=state["query"])]
        existing = new_messages

    model_choice = classify_complexity(state["query"])
    llm = llm_smart if model_choice == CHAT_MODEL_SMART else llm_fast
    response = llm.invoke(existing)

    return {
        "messages": new_messages + [response],
        "model_used": model_choice,
        "steps": state.get("steps", 0) + 1,
    }
