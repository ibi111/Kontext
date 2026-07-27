import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from rag.retrieval.embeddings import embedding_model
from rag.retrieval.bm25 import bm25_index



# Dummy chunks to test BM25
test_chunks = [
    {"id": "1", "content": "The EU AI Act regulates high-risk AI systems.", "document_id": "doc1", "document_name": "EU AI Act", "page_number": 1, "section": "Article 1", "metadata": {}},
    {"id": "2", "content": "Providers of AI systems must ensure transparency.", "document_id": "doc1", "document_name": "EU AI Act", "page_number": 2, "section": "Article 13", "metadata": {}},
    {"id": "3", "content": "High-risk AI systems require conformity assessments.", "document_id": "doc1", "document_name": "EU AI Act", "page_number": 3, "section": "Article 9", "metadata": {}},
]

# Build BM25 index
bm25_index.build(test_chunks)

# Test BM25
query = "What are high-risk AI system requirements?"
bm25_results = bm25_index.search(query, top_k=3)
print(f"BM25 results: {len(bm25_results)}")

# Test embedding
query_vec = embedding_model.embed_query(query)
print(f"Query vector dim: {len(query_vec)}")

print("All retrieval components working.")