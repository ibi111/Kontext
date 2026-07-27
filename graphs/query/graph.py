"""
Graph flow:
    input_guardrail
        → END (if blocked)
        → cache_check
            → cost_tracker (cache hit — skip everything)
            → seed_conversation
                → agent
                    → tools → agent (loop, max MAX_AGENT_STEPS)
                    → finalize
                        → output_guardrail
                            → cache_write
                                → cost_tracker
                                    → END
"""

from typing import Annotated, TypedDict, Optional

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from db.postgres import get_or_create_chatroom, history_to_chat_messages, add_message
from .nodes.agent import agent_node, SYSTEM_PROMPT
from .nodes.cache_check import cache_check
from .nodes.cache_write import cache_write
from .nodes.cost_tracker import track_cost
from .nodes.input_guardrail import input_guardrail
from .nodes.output_guardrail import output_guardrail
from .nodes.tools import AVAILABLE_TOOLS

MAX_AGENT_STEPS = 4


class QueryState(TypedDict):
    query: str
    session_id: str
    chatroom_id: Optional[int]
    blocked: bool
    cache_hit: bool
    messages: Annotated[list, add_messages]
    steps: int
    final_answer: str
    model_used: str
    usage: dict


tool_node = ToolNode(AVAILABLE_TOOLS)




def route_after_guardrail(state: dict):
    return END if state.get("blocked") else "cache_check"


def route_after_cache_check(state: dict):
    return "cost_tracker" if state.get("cache_hit") else "seed_conversation"


def route_after_agent(state: dict):
    last = state["messages"][-1]
    has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if has_tool_calls and state["steps"] < MAX_AGENT_STEPS:
        return "tools"
    return "finalize"


def seed_conversation(state: dict) -> dict:
    """
    Runs ONCE per API call, before the agent loop starts.

    Responsibilities:
    1. Resolve (or create) the chatroom for this session.
    2. Load prior user/assistant turns from Postgres as LangChain messages.
    3. Append the new HumanMessage for this turn.
    4. Prepend the SystemMessage.

    By doing this here instead of inside agent_node we avoid the
    "first call ever" vs "first call in this loop iteration" ambiguity
    that caused the old code to re-seed the system prompt on every loop.
    """
    session_id = state.get("session_id", "anonymous")
    chatroom_id = get_or_create_chatroom(session_id, state.get("chatroom_id"))

    # Persist the incoming user message now, before the agent runs,
    # so it's in the DB even if the agent crashes mid-turn.
    add_message(chatroom_id, role="user", content=state["query"])

    # Rebuild prior turns as LangChain messages (user/assistant text only)
    prior = history_to_chat_messages(chatroom_id)

    # Convert to LangChain message objects, excluding the just-added user msg
    # (history_to_chat_messages returns all rows including the one we just
    # inserted, so drop the last user message to avoid duplication).
    lc_messages = []
    for m in prior[:-1]:          # exclude the message we just inserted
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    # Build the full message list: system + history + current user turn
    seeded = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + lc_messages
        + [HumanMessage(content=state["query"])]
    )

    return {
        **state,
        "chatroom_id": chatroom_id,
        "messages": seeded,
    }


def finalize(state: dict) -> dict:
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None
    )
    answer = last_ai.content if last_ai and last_ai.content else "I don't know."

    # Persist assistant answer to the chatroom
    if state.get("chatroom_id"):
        add_message(
            state["chatroom_id"],
            role="assistant",
            content=answer,
            model_used=state.get("model_used"),
        )

    return {**state, "final_answer": answer}


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_query_graph():
    g = StateGraph(QueryState)

    g.add_node("input_guardrail", input_guardrail)
    g.add_node("cache_check", cache_check)
    g.add_node("seed_conversation", seed_conversation)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("finalize", finalize)
    g.add_node("output_guardrail", output_guardrail)
    g.add_node("cache_write", cache_write)
    g.add_node("cost_tracker", track_cost)

    g.set_entry_point("input_guardrail")

    g.add_conditional_edges(
        "input_guardrail", route_after_guardrail,
        {"cache_check": "cache_check", END: END}
    )
    g.add_conditional_edges(
        "cache_check", route_after_cache_check,
        {"cost_tracker": "cost_tracker", "seed_conversation": "seed_conversation"}
    )
    g.add_edge("seed_conversation", "agent")
    g.add_conditional_edges(
        "agent", route_after_agent,
        {"tools": "tools", "finalize": "finalize"}
    )
    g.add_edge("tools", "agent")
    g.add_edge("finalize", "output_guardrail")
    g.add_edge("output_guardrail", "cache_write")
    g.add_edge("cache_write", "cost_tracker")
    g.add_edge("cost_tracker", END)

    return g.compile()


query_graph = build_query_graph()