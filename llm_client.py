"""Central factory for the Gemini client.

Returns a ``google-genai`` Client for the backend named in ``LLM_BACKEND``:

* ``"vertex"``    — Agent Platform / Vertex in *express mode*: API-key auth
  (``VERTEX_API_KEY``), no ADC / project setup required.
* ``"developer"`` — Gemini Developer API, API-key auth (``GEMINI_API_KEY``).

Centralizing construction here means ``indexer``, ``retriever`` and
``rag_engine`` all share one path, so switching backends is a one-line change
in ``.env`` (``LLM_BACKEND``) instead of edits scattered across three files.
"""

from google import genai

from config import LLM_BACKEND, GEMINI_API_KEY, VERTEX_API_KEY


def get_client() -> genai.Client:
    """Build the Gemini client for the backend selected in config."""
    if LLM_BACKEND == "vertex":
        # Agent Platform express mode — API key only. `enterprise=True` is the
        # current name for what older google-genai called `vertexai=True`
        # (both still work in 2.x). Passing project/location alongside api_key
        # is rejected by the SDK, so express mode carries no location.
        return genai.Client(enterprise=True, api_key=VERTEX_API_KEY)
    # Gemini Developer API — simple API-key auth (free tier).
    return genai.Client(api_key=GEMINI_API_KEY)
