"""Use the Gemini Developer API SDK client for embeddings."""

from llm_client import get_client


def get_embedding_client():
    """Return the shared Gemini Developer API client used by the indexer."""
    return get_client()
