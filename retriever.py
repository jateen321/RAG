"""
Retriever — Searches ChromaDB for relevant chunks given a query.

Embeds the user's question using Gemini and finds the most similar chunks.
"""

from google import genai
import chromadb
from rich.console import Console

from config import (
    GEMINI_API_KEY, EMBEDDING_MODEL, TOP_K,
    CHROMA_DB_PATH, COLLECTION_NAME,
)

console = Console()

# Initialize Gemini client (new SDK)
_client = genai.Client(api_key=GEMINI_API_KEY)


def retrieve(query: str, top_k: int = None) -> list[dict]:
    """
    Retrieve the most relevant chunks for a given query.

    Args:
        query: User's question (Hindi or English).
        top_k: Number of results to return (default: from config).

    Returns:
        List of dicts with keys: 'text', 'page', 'source', 'distance'
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

    # Step 3: Format results
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i].get("page", "?"),
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "distance": results["distances"][0][i],
        })

    return retrieved
