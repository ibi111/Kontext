# Kontext: Agentic Document Intelligence

A RAG system for enterprise document intelligence, built on a real corpus of BMW/Mercedes annual reports, the EU AI Act, VDA guidelines, and supplier compliance documents.

## Architecture

Two independent LangGraph pipelines sharing two data stores.

**Ingestion** (`POST /ingest`):
`validate -> parse (Docling) -> chunk (HybridChunker) -> enrich (LLM) -> index (Qdrant + Postgres)`

**Query** (`POST /query`, `POST /query/stream`):
`input guardrail -> semantic cache check -> seed conversation history -> agent (decides: search or answer) <-> search tool -> finalize -> output guardrail -> cache write -> cost tracking`

The agent decides per question whether to search, and can re-search with a reformulated query (max 2 rounds) before answering.

## Stack

- Orchestration: LangGraph, LangChain tool calling
- LLM: OpenRouter free tier (gpt-oss-20b / gpt-oss-120b), routed by query complexity
- Parsing: Docling (table structure recognition, HybridChunker)
- Retrieval: Qdrant (vectors) + BM25 (keyword), fused with RRF, reranked with bge-reranker-v2-m3
- Embeddings: BAAI/bge-m3, local, no API cost
- Storage: Postgres (chatrooms, messages, tool calls, chunk metadata), Qdrant (vectors), Redis (cache)
- API: FastAPI, SSE streaming for tokens and tool calls

## Key decisions

- MCP used only for genuine agent tool choice (web search). Deterministic steps are direct code.
- Semantic cache is core, not optional: OpenRouter free tier caps at 20 req/min / 50 req/day.
- Postgres and Qdrant share chunk IDs, required for RRF to match chunks across both retrievers.
- Single shared knowledge base and cache, no per-user isolation (not needed at this scale).
- Guardrails mapped to OWASP LLM Top 10 (input.py: LLM01, output.py: LLM06).

## Setup

```bash
uv sync
cp .env.example .env

docker-compose --env-file .env -f docker/docker-compose.yml up -d
python scripts/init_db.py
python scripts/smoke_test.py

python scripts/download_documents.py
python scripts/test_ingestion_full.py "corpus/some_document.pdf"
python scripts/test_query.py "a question about that document"

uvicorn api.main:app --reload --port 8010
```

## Structure

```
graphs/ingestion/    ingestion pipeline
graphs/query/        query agent loop
rag/retrieval/        bm25, hybrid fusion, reranker, embeddings, vector store
rag/generation/        OpenRouter LLM client
db/                    Postgres + Qdrant clients
cache/                 Redis semantic cache
guardrails/            input/output security logic
api/                   FastAPI app
scripts/               test and init scripts
corpus/                ingested documents
```

## Status

Ingestion and query pipelines built and tested end to end, including multi-turn memory, tool-call streaming, Langfuse observability, and semantic caching. Not yet built: Tavily web search, RAGAS evaluation, test suite.
