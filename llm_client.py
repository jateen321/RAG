"""Central factory for the Gemini Developer API client."""

from google import genai

from config import GEMINI_API_KEY


def get_client() -> genai.Client:
    """Build a Gemini Developer API client using the configured API key."""
    return genai.Client(api_key=GEMINI_API_KEY)
