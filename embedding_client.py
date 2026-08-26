"""Embedding-only client with Vertex regional round-robin and failover.

Generation continues to use :func:`llm_client.get_client`.  Keeping this as a
separate facade prevents embedding quota routing from changing where answer
generation is sent.
"""

from dataclasses import dataclass
import threading
from types import SimpleNamespace

import requests

from config import (
    EMBEDDING_REGION_ROTATION_ENABLED,
    LLM_BACKEND,
    VERTEX_API_KEY,
    VERTEX_EMBEDDING_PROJECT_ID,
    VERTEX_EMBEDDING_REGIONS,
    VERTEX_EMBEDDING_TIMEOUT_S,
)
from llm_client import get_client


_RETRYABLE_STATUSES = {404, 408, 429, 500, 502, 503, 504}
_thread_state = threading.local()


@dataclass
class _Embedding:
    values: list[float]


@dataclass
class _EmbeddingResponse:
    embeddings: list[_Embedding]


class RegionalEmbeddingError(RuntimeError):
    """Sanitized regional request failure compatible with indexer retry logic."""

    def __init__(self, message: str, status_code: int | None, details=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = status_code
        self.details = details


def _http_session() -> requests.Session:
    """Give each worker a connection-pooled Session without sharing it across threads."""
    session = getattr(_thread_state, "session", None)
    if session is None:
        session = requests.Session()
        _thread_state.session = session
    return session


class RotatingEmbeddingClient:
    """Expose the SDK's ``client.models.embed_content`` shape.

    Only when ``LLM_BACKEND=vertex`` and the model is
    ``gemini-embedding-001``, each logical call begins in the next configured
    region. A retryable regional failure immediately falls through to the next
    region. Developer mode always retains the Developer API SDK path, even if a
    Vertex API key is also present in the environment.
    """

    def __init__(
        self,
        sdk_client,
        *,
        project_id: str,
        api_key: str | None,
        regions: tuple[str, ...],
        enabled: bool,
        timeout_s: float,
    ):
        self.models = self
        self._sdk_client = sdk_client
        self._project_id = project_id
        self._api_key = api_key
        self._regions = regions
        self._enabled = enabled
        self._timeout_s = timeout_s
        self._next_index = 0
        self._index_lock = threading.Lock()

    def _region_order(self) -> tuple[str, ...]:
        with self._index_lock:
            start = self._next_index
            self._next_index = (self._next_index + 1) % len(self._regions)
        return self._regions[start:] + self._regions[:start]

    def embed_content(self, *, model: str, contents):
        rotate = (
            self._enabled
            and LLM_BACKEND == "vertex"
            and model == "gemini-embedding-001"
        )
        if not rotate:
            return self._sdk_client.models.embed_content(
                model=model,
                contents=contents,
            )

        texts = [contents] if isinstance(contents, str) else list(contents)
        if not texts:
            return _EmbeddingResponse(embeddings=[])
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Regional embedding contents must be text strings.")

        payload = {"instances": [{"content": text} for text in texts]}
        last_error = None

        for region in self._region_order():
            host = f"{region}-aiplatform.googleapis.com"
            url = (
                f"https://{host}/v1/projects/{self._project_id}/locations/{region}"
                f"/publishers/google/models/{model}:predict"
            )
            try:
                response = _http_session().post(
                    url,
                    params={"key": self._api_key},
                    json=payload,
                    timeout=self._timeout_s,
                )
            except requests.RequestException as exc:
                # Never include the Requests exception text: it may contain the
                # fully rendered URL and therefore the API key query parameter.
                last_error = RegionalEmbeddingError(
                    f"Vertex embedding network failure in {region} "
                    f"({type(exc).__name__}).",
                    None,
                )
                continue

            try:
                body = response.json()
            except ValueError:
                body = {}

            if response.status_code == 200:
                predictions = body.get("predictions") or []
                values = [
                    item.get("embeddings", {}).get("values", [])
                    for item in predictions
                ]
                if len(values) != len(texts) or any(not vector for vector in values):
                    raise RegionalEmbeddingError(
                        f"Vertex returned {len(values)} embeddings for "
                        f"{len(texts)} texts in {region}.",
                        502,
                        body,
                    )
                return _EmbeddingResponse(
                    embeddings=[_Embedding(values=vector) for vector in values]
                )

            message = (
                body.get("error", {}).get("message")
                if isinstance(body, dict)
                else None
            )
            last_error = RegionalEmbeddingError(
                f"Vertex embedding failed in {region}: HTTP {response.status_code}"
                + (f" ({message[:160]})" if message else ""),
                response.status_code,
                body,
            )
            if response.status_code not in _RETRYABLE_STATUSES:
                raise last_error

        if last_error is not None:
            raise last_error
        raise RegionalEmbeddingError("No Vertex embedding regions configured.", None)


def get_embedding_client() -> RotatingEmbeddingClient:
    """Build the embedding facade while preserving the selected SDK backend."""
    return RotatingEmbeddingClient(
        get_client(),
        project_id=VERTEX_EMBEDDING_PROJECT_ID,
        api_key=VERTEX_API_KEY,
        regions=VERTEX_EMBEDDING_REGIONS,
        enabled=EMBEDDING_REGION_ROTATION_ENABLED,
        timeout_s=VERTEX_EMBEDDING_TIMEOUT_S,
    )
