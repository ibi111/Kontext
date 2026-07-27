
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, AIMessage, ToolMessage

from api.deps import get_langfuse_handler
from api.schemas.query_schemas import QueryRequest, StreamChunk
from graphs.query.graph import query_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["query"])


def _chunk_json(chunk: StreamChunk) -> str:
    """Format one StreamChunk as an SSE data line."""
    return f"data: {chunk.model_dump_json()}\n\n"


async def _stream(
    request: QueryRequest,
    langfuse_handler,
) -> AsyncGenerator[str, None]:

    initial_state = {
        "query": request.query,
        "session_id": request.session_id,
        "chatroom_id": request.chatroom_id,
        "blocked": False,
        "cache_hit": False,
        "messages": [],
        "steps": 0,
        "final_answer": "",
        "model_used": "",
        "usage": {},
    }

    config = {
        "callbacks": [langfuse_handler],
        "run_name": "query_graph",
    }

    chatroom_id = request.chatroom_id  # updated once seed_conversation runs

    async for stream_mode, data in query_graph.astream(
        initial_state,
        config=config,
        stream_mode=["updates", "messages"],
    ):
        # ── "messages" mode: token-by-token
        if stream_mode == "messages":
            msg, metadata = data
            node = metadata.get("langgraph_node", "")

            # Only stream tokens from the agent node, not tool nodes
            if node == "agent" and isinstance(msg, AIMessageChunk):
                if msg.content:
                    yield _chunk_json(StreamChunk(
                        type="token",
                        content=msg.content,
                    ))

        # ── "updates" mode: node-level state
        elif stream_mode == "updates":
            # data is a dict keyed by node name whose value is the state
            # update that node returned.

            for node_name, node_update in data.items():

                # seed_conversation resolved the chatroom — capture it
                if node_name == "seed_conversation":
                    chatroom_id = node_update.get("chatroom_id", chatroom_id)

                # input guardrail fired
                elif node_name == "input_guardrail" and node_update.get("blocked"):
                    yield _chunk_json(StreamChunk(
                        type="blocked",
                        content=node_update.get("final_answer", "Request blocked."),
                        chatroom_id=chatroom_id,
                    ))

                # cache hit
                elif node_name == "cache_check" and node_update.get("cache_hit"):
                    yield _chunk_json(StreamChunk(
                        type="cache_hit",
                        content=node_update.get("final_answer"),
                        chatroom_id=chatroom_id,
                    ))

                # tool node executed — emit tool_start and tool_end
                elif node_name == "tools":
                    messages = node_update.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, ToolMessage):
                            yield _chunk_json(StreamChunk(
                                type="tool_end",
                                content=msg.name if hasattr(msg, "name") else "tool",
                                meta={
                                    "tool_call_id": msg.tool_call_id,
                                    "result_preview": (msg.content or "")[:200],
                                },
                                chatroom_id=chatroom_id,
                            ))

                # agent decided to call a tool — emit tool_start before
                # the tool node actually runs
                elif node_name == "agent":
                    messages = node_update.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield _chunk_json(StreamChunk(
                                    type="tool_start",
                                    content=tc["name"],
                                    meta={"args": tc.get("args", {})},
                                    chatroom_id=chatroom_id,
                                ))

                # cost_tracker ran — stream is ending
                elif node_name == "cost_tracker":
                    usage = node_update.get("usage", {})
                    final = node_update.get("final_answer", "")

                    # If final_answer wasn't set yet grab it from finalize's update
                    if not final:
                        final = data.get("finalize", {}).get("final_answer", "")

                    yield _chunk_json(StreamChunk(
                        type="done",
                        content=final or node_update.get("final_answer", ""),
                        chatroom_id=chatroom_id,
                        meta={
                            "usage": usage,
                            "model_used": node_update.get("model_used"),
                            "cache_hit": node_update.get("cache_hit", False),
                        },
                    ))


@router.post("")
async def chat(
    request: QueryRequest,
    langfuse_handler=Depends(get_langfuse_handler),
):
    """
    Streaming chat endpoint.

    Returns an SSE stream of JSON-encoded StreamChunk objects.
    Each chunk has a `type` field:
      - "token"      one LLM output token
      - "tool_start" agent is calling a tool
      - "tool_end"   tool returned
      - "done"       final answer + usage stats
      - "cache_hit"  answered from cache
      - "blocked"    input guardrail fired
      - "error"      unexpected error

    The stream ends after the "done" (or "blocked"/"cache_hit") event.
    """
    return StreamingResponse(
        _stream(request, langfuse_handler),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # disable nginx buffering
        },
    )
