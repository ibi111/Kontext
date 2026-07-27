from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    query: str
    session_id: str
    chatroom_id: Optional[int] = None   # None = start new chatroom


class QueryResponse(BaseModel):
    final_answer: str
    chatroom_id: int
    model_used: Optional[str] = None
    cache_hit: bool = False
    blocked: bool = False
    usage: Optional[dict] = None


class StreamChunk(BaseModel):
    """
    Each line in the SSE stream is one of these, JSON-encoded.

    type:
      "token"      - one LLM output token (delta)
      "tool_start" - agent is about to call a tool
      "tool_end"   - tool returned a result
      "done"       - stream finished, full answer in `content`
      "error"      - something went wrong
      "cache_hit"  - answered from cache, no LLM calls made
      "blocked"    - input guardrail fired
    """
    type: str                        # token | tool_start | tool_end | done | error | cache_hit | blocked
    content: Optional[str] = None    # token text, tool name, final answer, error message
    chatroom_id: Optional[int] = None
    meta: Optional[dict] = None      # tool args, usage stats, etc.
