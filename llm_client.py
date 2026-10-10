"""Central factory for the Gemini client.

Returns a ``google-genai`` Client for the backend named in ``LLM_BACKEND``:

* ``"vertex"``    — Agent Platform / Vertex in *express mode*: API-key auth
  (``VERTEX_API_KEY``), no ADC / project setup required.
* ``"developer"`` — Gemini Developer API, API-key auth (``GEMINI_API_KEY``).

Centralizing construction here means ``indexer``, ``retriever`` and
``rag_engine`` all share one path, so switching backends is a one-line change
in ``.env`` (``LLM_BACKEND``) instead of edits scattered across three files.
"""

import os
import threading
import time
from collections import deque

from google import genai

from config import LLM_BACKEND, GEMINI_API_KEY, VERTEX_API_KEY

_LLM_METHODS = {"generate_content", "generate_content_stream"}
_request_starts: deque[float] = deque()
_request_lock = threading.Lock()


def _wait_for_llm_slot() -> None:
    """Enforce an optional process-wide rolling-window generation RPM cap."""
    configured_limit = os.getenv("LLM_REQUESTS_PER_MINUTE", "").strip()
    if not configured_limit:
        return
    try:
        limit = int(configured_limit)
    except ValueError as exc:
        raise RuntimeError("LLM_REQUESTS_PER_MINUTE must be a positive integer.") from exc
    if limit < 1:
        raise RuntimeError("LLM_REQUESTS_PER_MINUTE must be a positive integer.")

    with _request_lock:
        while True:
            now = time.monotonic()
            while _request_starts and now - _request_starts[0] >= 60:
                _request_starts.popleft()
            if len(_request_starts) < limit:
                _request_starts.append(now)
                return
            time.sleep(max(0.001, 60 - (now - _request_starts[0])))


class _RateLimitedModels:
    """SDK models facade that paces every content-generation request."""

    def __init__(self, models) -> None:
        self._models = models

    def __getattr__(self, name):
        method = getattr(self._models, name)
        if name not in _LLM_METHODS:
            return method

        def call(*args, **kwargs):
            _wait_for_llm_slot()
            return method(*args, **kwargs)

        return call


class _RateLimitedClient:
    """Preserve the SDK client interface while wrapping its models methods."""

    def __init__(self, client) -> None:
        self._client = client
        self.models = _RateLimitedModels(client.models)

    def __getattr__(self, name):
        return getattr(self._client, name)


def get_client() -> genai.Client:
    """Build the Gemini client for the backend selected in config."""
    if LLM_BACKEND == "vertex":
        # Agent Platform express mode — API key only. `enterprise=True` is the
        # current name for what older google-genai called `vertexai=True`
        # (both still work in 2.x). Passing project/location alongside api_key
        # is rejected by the SDK, so express mode carries no location.
        client = genai.Client(enterprise=True, api_key=VERTEX_API_KEY)
        return _RateLimitedClient(client)
    # Gemini Developer API — simple API-key auth (free tier).
    client = genai.Client(api_key=GEMINI_API_KEY)
    return _RateLimitedClient(client)
