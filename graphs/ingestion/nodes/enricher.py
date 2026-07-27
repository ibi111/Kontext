import logging

from config import CHAT_MODEL_FAST, ENRICH_MIN_CHUNK_CHARS
from rag.generation.llm import chat

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """<document>{doc_summary}</document>
Here is a chunk from this document:
<chunk>{chunk_text}</chunk>
Write a 1-2 sentence description situating this chunk within the overall
document, to improve search retrieval. Answer only with the description."""


ENRICH_MAX_TOKENS = 400
ENRICH_RETRY_MAX_TOKENS = 700  # used only if the first attempt truncates


def _enrich_one(chunk_text: str, doc_summary: str, max_tokens: int):
    message = chat(
        model=CHAT_MODEL_FAST,
        messages=[{
            "role": "user",
            "content": CONTEXT_PROMPT.format(doc_summary=doc_summary, chunk_text=chunk_text),
        }],
        max_tokens=max_tokens,
    )
    return message.content, message.finish_reason


def enrich_chunks(state: dict) -> dict:
    doc_summary = state.get("doc_summary", state.get("doc_markdown_preview", ""))

    for chunk in state["chunks"]:
        if len(chunk["text"]) < ENRICH_MIN_CHUNK_CHARS:
            chunk["context"] = ""
            chunk["enriched_text"] = chunk["text"]
            continue

        context, finish_reason = _enrich_one(chunk["text"], doc_summary, ENRICH_MAX_TOKENS)

        if finish_reason == "length":
            # Truncated mid-sentence — retry once with more headroom rather
            # than silently keeping a cut-off description.
            logger.warning(
                f"Enrichment truncated (page={chunk.get('page')}); retrying with "
                f"max_tokens={ENRICH_RETRY_MAX_TOKENS}."
            )
            context, finish_reason = _enrich_one(
                chunk["text"], doc_summary, ENRICH_RETRY_MAX_TOKENS
            )
            if finish_reason == "length":
                logger.warning(
                    f"Enrichment still truncated after retry (page={chunk.get('page')}); "
                    f"falling back to unenriched text."
                )
                context = None

        if context is None:
            # Reasoning model burned its whole budget without producing
            # final content, even after retrying. Don't silently stringify
            # None into the chunk text — fall back to raw, unenriched text.
            if finish_reason != "length":
                logger.warning(
                    f"Enrichment returned no content (page={chunk.get('page')}); "
                    f"falling back to unenriched text."
                )
            chunk["context"] = ""
            chunk["enriched_text"] = chunk["text"]
            continue

        chunk["context"] = context
        chunk["enriched_text"] = f"{context}\n\n{chunk['text']}"

    return state