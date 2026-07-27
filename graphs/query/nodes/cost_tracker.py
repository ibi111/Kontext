"""
Logs real per-turn usage: request count (what actually matters against
OpenRouter's free-tier daily cap) and token totals (for when you migrate
off free models and dollar cost becomes real).

Uses response.usage_metadata directly on each AIMessage rather than
LangChain's get_openai_callback() — that callback is OpenAI-pricing-table
specific and won't recognize OpenRouter's free model IDs. usage_metadata
is framework-agnostic and populated by ChatOpenAI regardless of provider.
"""

import logging
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def track_cost(state: dict) -> dict:
    ai_messages = [m for m in state.get("messages", []) if isinstance(m, AIMessage)]
    tool_messages_count = sum(
        1 for m in state.get("messages", []) if getattr(m, "type", None) == "tool"
    )

    # Each AIMessage in the loop = one real OpenRouter API call consumed
    # against the free-tier daily cap, regardless of how many tokens it used.
    request_count = len(ai_messages)

    input_tokens = 0
    output_tokens = 0
    for m in ai_messages:
        usage = getattr(m, "usage_metadata", None)
        if usage:
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)

    logger.info(
        f"[cost] model={state.get('model_used')} "
        f"agent_steps={state.get('steps', 0)} "
        f"api_requests={request_count} "
        f"tool_calls={tool_messages_count} "
        f"input_tokens={input_tokens} "
        f"output_tokens={output_tokens}"
    )

    return {
        **state,
        "usage": {
            "request_count": request_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }