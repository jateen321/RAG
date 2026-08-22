"""
Retriever — Searches ChromaDB for relevant chunks given a query.

Embeds the user's question using Gemini and finds the most similar chunks.
"""

import chromadb
from rich.console import Console

from config import (
    EMBEDDING_MODEL, TOP_K,
    CHROMA_DB_PATH, COLLECTION_NAME,
)
from llm_client import get_client

console = Console()

# Initialize Gemini client (backend chosen in config: Developer API or Vertex)
_client = get_client()


def retrieve(query: str, top_k: int = None) -> list[dict]:
    """
    Retrieve the most relevant chunks for a given query.

    Args:
        query: User's question (Hindi or English).
        top_k: Number of results to return (default: from config).

    Returns:
        List of dicts with keys: 'text', 'page', 'source', 'distance',
        'source_type', 'document_id', 'chunk_index', 'extraction_method',
        'content_hash'
        Sorted by relevance (most relevant first).
    """
    if top_k is None:
        top_k = TOP_K

    # Step 1: Embed the query
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
    )
    query_embedding = result.embeddings[0].values

    # Step 2: Search ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        console.print("[red]❌ No indexed documents found![/red]")
        console.print("   Run: [bold]python app.py index <pdf_file>[/bold] first.")
        return []

    if collection.count() == 0:
        console.print("[red]❌ Database is empty. Index a PDF first.[/red]")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Step 3: Format results.
    # 'page' and 'source' are kept as the caller-facing names even though the
    # stored metadata now uses 'page_number' and 'source_name' — this module is
    # the translation layer, so rag_engine/evaluate/api need no changes. The
    # remaining metadata fields are passed through for diagnostics.
    retrieved = []
    for i in range(len(results["ids"][0])):
        md = results["metadatas"][0][i] or {}
        retrieved.append({
            "text": results["documents"][0][i],
            "page": md.get("page_number", "?"),
            "source": md.get("source_name", "unknown"),
            "distance": results["distances"][0][i],
            # Passed through so callers can diagnose *why* a chunk was returned:
            "source_type": md.get("source_type", "unknown"),
            "document_id": md.get("document_id", "unknown"),
            "chunk_index": md.get("chunk_index"),
            "extraction_method": md.get("extraction_method", "unknown"),
            "content_hash": md.get("content_hash", ""),
        })

    return retrieved
