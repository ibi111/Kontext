import json
from rag.retrieval.search import search_knowledge_base


def execute_tools(state: dict) -> dict:
    messages = state["messages"]

    for tool_call in state["pending_tool_calls"]:
        if tool_call.function.name == "search_documents":
            args = json.loads(tool_call.function.arguments)
            results = search_knowledge_base(args["query"])
            result_text = "\n\n---\n\n".join(results) if results else "No relevant passages found."
        else:
            result_text = f"Unknown tool: {tool_call.function.name}"

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_text,
        })

    return {**state, "messages": messages, "pending_tool_calls": []}
