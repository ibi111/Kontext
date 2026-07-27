from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

from .nodes.input_guardrail import input_guardrail
from .nodes.cache_check import cache_check
from .nodes.agent import agent_node
from .nodes.tools import AVAILABLE_TOOLS
from .nodes.output_guardrail import output_guardrail
from .nodes.cache_write import cache_write
from .nodes.cost_tracker import track_cost

MAX_AGENT_STEPS = 4  # hard cap: at most 2 search rounds before forcing a final answer


class QueryState(TypedDict):
    query: str
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
    # On a hit, skip the entire agent loop and go straight to logging —
    # no LLM calls, no retrieval, nothing to redact again (it was already
    # redacted the first time this answer was cached).
    return "cost_tracker" if state.get("cache_hit") else "agent"


def route_after_agent(state: dict):
    last = state["messages"][-1]
    has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if has_tool_calls and state["steps"] < MAX_AGENT_STEPS:
        return "tools"
    return "finalize"


def finalize(state: dict) -> dict:
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    return {**state, "final_answer": last_ai.content or "I don't know."}


def build_query_graph():
    g = StateGraph(QueryState)
    g.add_node("input_guardrail", input_guardrail)
    g.add_node("cache_check", cache_check)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("finalize", finalize)
    g.add_node("output_guardrail", output_guardrail)
    g.add_node("cache_write", cache_write)
    g.add_node("cost_tracker", track_cost)

    g.set_entry_point("input_guardrail")
    g.add_conditional_edges("input_guardrail", route_after_guardrail,
                             {"cache_check": "cache_check", END: END})
    g.add_conditional_edges("cache_check", route_after_cache_check,
                             {"cost_tracker": "cost_tracker", "agent": "agent"})
    g.add_conditional_edges("agent", route_after_agent,
                             {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", "output_guardrail")
    g.add_edge("output_guardrail", "cache_write")
    g.add_edge("cache_write", "cost_tracker")
    g.add_edge("cost_tracker", END)
    return g.compile()


query_graph = build_query_graph()
