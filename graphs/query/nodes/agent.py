from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, CHAT_MODEL_FAST, CHAT_MODEL_SMART,
)
from .tools import AVAILABLE_TOOLS

SYSTEM_PROMPT = """You are a document intelligence assistant for enterprise
technical and corporate documents (BMW/Mercedes annual reports, EU AI Act,
VDA guidelines, supplier compliance documents).

You have a `search_documents` tool. Default to using it. Search FIRST for
any question that touches a topic your document corpus could plausibly
cover — including questions you feel confident you already know the
general answer to. Your training knowledge may be generic, outdated, or
worded differently than the exact ingested document, and this system's
entire purpose is to ground answers in that specific ingested text, not
in what you already know. Only skip the search tool for questions that are
clearly unrelated to the corpus entirely (e.g. basic arithmetic, or topics
with no plausible connection to enterprise documents, AI regulation, or
automotive standards).

If your first search doesn't return enough to answer confidently, call it
again with a reformulated query before giving up.

Always cite the page number when your answer comes from a searched
passage — an answer without a page citation should be rare, and only
happen when the question genuinely has nothing to do with the documents.
If you searched and still cannot find the answer, say so plainly rather
than guessing or falling back on general knowledge."""


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
