"""
Single place all LLM calls go through — used by enricher.py (plain text
call) and the query agent (tool-calling call).
"""

import logging
from openai import OpenAI, RateLimitError, APIStatusError
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log,
)
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, CHAT_MODEL_FALLBACK

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


@retry(
    retry=retry_if_exception_type((RateLimitError, APIStatusError)),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _call(model: str, messages: list, max_tokens: int, tools: list | None):
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
    if tools:
        kwargs["tools"] = tools
    return client.chat.completions.create(**kwargs)


def chat(model: str, messages: list, tools: list | None = None, max_tokens: int = 500):
    """
    Returns the raw message object (has .content and .tool_calls), with
    .finish_reason attached so callers can detect truncation — reasoning
    models (e.g. openai/gpt-oss-*:free) can burn their whole token budget
    on hidden reasoning and cut off final content mid-sentence; checking
    finish_reason == "length" is how you catch that instead of silently
    accepting a truncated response.
    """
    try:
        resp = _call(model, messages, max_tokens, tools)
    except Exception as e:
        logger.warning(f"Model {model} failed after retries ({e}); "
                        f"falling back to {CHAT_MODEL_FALLBACK}")
        resp = _call(CHAT_MODEL_FALLBACK, messages, max_tokens, tools)

    message = resp.choices[0].message
    message.finish_reason = resp.choices[0].finish_reason
    return message


def call_llm(model: str, messages: list, max_tokens: int = 500) -> str:
    """Simple text-only call (no tools) — used by the ingestion enricher."""
    return chat(model, messages, tools=None, max_tokens=max_tokens).content