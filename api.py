"""FastAPI interface for the Gyaan Sarthi RAG pipeline."""

import hashlib
import re
import os
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Callable
from uuid import UUID, uuid4

from fastapi import (
    Depends, FastAPI, File, Form, HTTPException, Query, Request, Response,
    UploadFile, status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from config import (
    FIREBASE_PROJECT_ID,
    CONVERSATION_DB_PATH,
    DATA_DIR,
    GENERATED_IMAGE_DIR,
    INDEX_FOLDER_ROOTS,
    LEGACY_ADMIN_UID,
    SHARED_CORPUS_OWNER_ID,
    SESSION_COOKIE_MAX_AGE_S,
    SESSION_COOKIE_NAME,
    SESSION_COOKIE_SAMESITE,
    SESSION_COOKIE_SECURE,
)
from auth import (
    AuthenticatedUser,
    create_session_cookie,
    get_current_user,
    get_optional_user,
    require_admin,
    verify_session_cookie,
)
from conversation_store import (
    attach_generated_image,
    conversation_exists,
    delete_conversation,
    get_conversation,
    get_generated_image_metadata,
    get_history_before_exchange,
    get_recent_history,
    list_conversations,
    record_exchange,
    replace_exchange_and_truncate,
    replace_latest_exchange_with_web_answer,
)
from document_ingester import (
    SOURCE_TYPES,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    extract_document as _extract_document,
    extract_text_document as _extract_text_document,
    index_folder,
    resolve_allowed_folder,
)
from rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    get_rate_limiter,
)


app = FastAPI(
    title="Gyaan Sarthi API",
    description="Index PDF, TXT, Markdown, and YouTube sources, then ask grounded questions.",
    version="1.3.0",
)

_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("RAG_ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_PROMPT_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_PROMPT_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_CONTENT_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
}

_STATE_CHANGING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_INLINE_CITATION = re.compile(r"\s*⟦[^⟦⟧\n]+⟧")


def _redact_guest_citations(answer: str) -> str:
    """Remove model citation markers before returning an anonymous answer."""
    return _INLINE_CITATION.sub("", answer).strip()


def _corpus_owner_id(user: AuthenticatedUser | None) -> str:
    """Select the corpus visible to a request.

    Guests and administrators use the shared library. Every other verified
    user receives a private corpus keyed by their Firebase UID.
    """
    if user is None or user.is_admin:
        return SHARED_CORPUS_OWNER_ID
    return user.uid


@app.middleware("http")
async def enforce_trusted_origin(request: Request, call_next):
    """Reject cross-site writes so a session cookie alone cannot authorize them.

    ``SameSite=Lax`` already blocks forged writes for same-site deployments, but
    the README documents ``SESSION_COOKIE_SAMESITE=none`` for cross-site
    domains. There the cookie would ride along on a forged POST, so every
    state-changing method is checked here rather than route by route.
    """
    if request.method in _STATE_CHANGING_METHODS:
        try:
            _require_trusted_origin(request)
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
    expose_headers=["Retry-After"],
)


def _answer_limits(use_web: bool, generate_image: bool) -> tuple[tuple[str, ...], tuple[str, ...]]:
    rates = ["ask"]
    concurrency = ["interactive"]
    if use_web:
        rates.append("web")
        concurrency.append("web")
    if generate_image:
        rates.append("image")
        concurrency.append("image")
    return tuple(rates), tuple(concurrency)


def _rate_limited(
    *,
    rates: tuple[str, ...] = (),
    concurrency: tuple[str, ...] = (),
    select: Callable[[dict], tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
):
    """Apply distributed admission without changing FastAPI route signatures."""
    def decorate(endpoint):
        endpoint_signature = signature(endpoint)

        @wraps(endpoint)
        async def wrapped(*args, **kwargs):
            arguments = endpoint_signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            selected_rates, selected_concurrency = (
                select(arguments.arguments) if select else (rates, concurrency)
            )
            user = arguments.arguments["user"]
            if user is None:
                http_request = arguments.arguments.get("http_request")
                client_host = (
                    http_request.client.host
                    if isinstance(http_request, Request) and http_request.client
                    else "unknown"
                )
                identity = "guest:" + hashlib.sha256(
                    client_host.encode("utf-8")
                ).hexdigest()
            else:
                identity = user.uid
            try:
                limiter = get_rate_limiter()
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Request admission is not configured. Please try again later.",
                ) from exc
            try:
                async with limiter.admit(
                    identity,
                    rates=selected_rates,
                    concurrency=selected_concurrency,
                ):
                    return await endpoint(*args, **kwargs)
            except RateLimitExceeded as exc:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=exc.detail,
                    headers={"Retry-After": str(exc.retry_after)},
                ) from exc
            except RateLimitUnavailable as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(exc),
                ) from exc

        return wrapped
    return decorate


def _ask_request_limits(arguments: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    payload = arguments["payload"]
    if arguments["user"] is None:
        return _answer_limits(False, False)
    return _answer_limits(payload.use_web, payload.generate_image)


def _image_ask_limits(arguments: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return _answer_limits(arguments["use_web"], arguments["generate_image"])


@app.on_event("startup")
async def migrate_legacy_tenant_data() -> None:
    """Make pre-authentication vector rows part of the shared corpus."""
    from indexer import assign_legacy_documents

    await run_in_threadpool(assign_legacy_documents, SHARED_CORPUS_OWNER_ID)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)
    use_web: bool = False
    generate_image: bool = False


class SessionRequest(BaseModel):
    id_token: str = Field(min_length=20, max_length=10000)


class EditExchangeRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class IndexRequest(BaseModel):
    filename: str = Field(
        min_length=1,
        description="PDF, TXT, or Markdown filename inside the project's data directory.",
    )


class YouTubeIndexRequest(BaseModel):
    url: str = Field(
        min_length=1,
        max_length=2000,
        description="YouTube video or playlist URL.",
    )


class FolderIndexRequest(BaseModel):
    folder_path: str = Field(
        min_length=1,
        max_length=4096,
        description="Server-local folder under one of INDEX_FOLDER_ROOTS.",
    )
    recursive: bool = Field(
        default=True,
        description="Include supported documents in nested folders.",
    )


def _detected_prompt_image_type(data: bytes) -> str | None:
    """Recognize the supported formats from their file signatures."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _generated_image_path(image_id: str) -> Path:
    """Resolve only canonical UUID filenames inside the generated-image root."""
    try:
        canonical_id = str(UUID(image_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Generated image not found.") from exc
    if canonical_id != image_id:
        raise HTTPException(status_code=404, detail="Generated image not found.")
    return Path(GENERATED_IMAGE_DIR).resolve() / canonical_id


def _save_generated_image(image_id: str, data: bytes) -> Path:
    directory = Path(GENERATED_IMAGE_DIR).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / image_id
    temporary = directory / f".{image_id}.tmp"
    temporary.write_bytes(data)
    temporary.replace(destination)
    return destination


def _tenant_data_root(owner_id: str) -> Path:
    """Use an opaque directory so Firebase UIDs never become path segments."""
    import hashlib

    # A server without Firebase configuration cannot authenticate real traffic;
    # retaining the historical root keeps dependency-overridden unit tests and
    # local maintenance tools backward compatible.
    if not FIREBASE_PROJECT_ID:
        return Path(DATA_DIR).resolve()
    tenant = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()
    return (Path(DATA_DIR).resolve() / "users" / tenant).resolve()


def _resolve_data_document(filename: str, owner_id: str | None = None) -> Path:
    """Resolve a supported document while preventing traversal outside ``data/``."""
    data_root = _tenant_data_root(owner_id) if owner_id else Path(DATA_DIR).resolve()
    relative_path = Path(filename)
    # Older folder-indexing runs stored paths relative to the repository root,
    # including the leading ``data/`` directory. Accept both that legacy form
    # and the canonical path relative to DATA_DIR.
    if relative_path.parts[:1] == ("data",):
        relative_path = Path(*relative_path.parts[1:])
    candidate = (data_root / relative_path).resolve()
    if data_root != candidate.parent and data_root not in candidate.parents:
        raise ValueError("The document must be located inside the data directory.")
    if candidate.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError("Only PDF, TXT, and Markdown files can be indexed.")
    if not candidate.is_file() and owner_id in {
        LEGACY_ADMIN_UID,
        SHARED_CORPUS_OWNER_ID,
    }:
        legacy_root = Path(DATA_DIR).resolve()
        legacy_candidate = (legacy_root / relative_path).resolve()
        if (
            legacy_candidate.parent == legacy_root
            or legacy_root in legacy_candidate.parents
        ) and legacy_candidate.is_file():
            candidate = legacy_candidate
    if not candidate.is_file():
        raise ValueError(f"Document not found: {filename}")
    return candidate


@app.get("/")
def root() -> dict:
    return {
        "name": "Gyaan Sarthi API",
        "docs": "/docs",
        "health": "/health",
    }


def _require_trusted_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        raise HTTPException(status_code=403, detail="Untrusted request origin.")


@app.post("/auth/session")
async def session_login(
    payload: SessionRequest,
    request: Request,
    response: Response,
) -> dict:
    """Exchange a recently issued Firebase ID token for an HttpOnly session."""
    _require_trusted_origin(request)
    try:
        session_cookie = await run_in_threadpool(
            create_session_cookie, payload.id_token, SESSION_COOKIE_MAX_AGE_S
        )
        user = await run_in_threadpool(verify_session_cookie, session_cookie)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_cookie,
        max_age=SESSION_COOKIE_MAX_AGE_S,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        samesite=SESSION_COOKIE_SAMESITE,
        path="/",
    )
    return {"uid": user.uid, "email": user.email, "is_admin": user.is_admin}


@app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def session_logout(request: Request, response: Response) -> None:
    _require_trusted_origin(request)
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        secure=SESSION_COOKIE_SECURE,
        httponly=True,
        samesite=SESSION_COOKIE_SAMESITE,
    )


@app.get("/auth/me")
async def session_identity(
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    return {"uid": user.uid, "email": user.email, "is_admin": user.is_admin}


@app.get("/health")
def health(
    user: AuthenticatedUser | None = Depends(get_optional_user),
) -> dict:
    """Expose only liveness to guests; source metadata requires a session."""
    if user is None:
        return {"status": "ok"}

    from indexer import get_stats

    return {"status": "ok", **get_stats(owner_id=_corpus_owner_id(user))}


@app.get("/documents/{source_path:path}", response_class=FileResponse)
def open_document(
    source_path: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> FileResponse:
    """Open a user's own source or an administrator-managed shared source."""
    try:
        document_path = _resolve_data_document(source_path, _corpus_owner_id(user))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc

    media_types = {
        ".pdf": "application/pdf",
        ".txt": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
    }
    return FileResponse(document_path, media_type=media_types[document_path.suffix.lower()])


@app.get("/passages/resolve-legacy")
async def resolve_legacy_passage(
    source: str = Query(min_length=1, max_length=4096),
    page: int = Query(ge=0),
    preview: str = Query(min_length=1, max_length=500),
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """Resolve citations saved before source responses included chunk IDs."""
    from indexer import get_chunk_by_citation

    passage = await run_in_threadpool(
        get_chunk_by_citation, source, page, preview, _corpus_owner_id(user)
    )
    if passage is None:
        raise HTTPException(status_code=404, detail="Exact cited passage not found.")
    return passage


@app.get("/passages/{chunk_id}")
async def passage_detail(
    chunk_id: str,
    source: str = Query(min_length=1, max_length=4096),
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """Return one cited passage, scoped to its source to prevent id confusion."""
    from indexer import get_chunk

    passage = await run_in_threadpool(
        get_chunk, chunk_id, source, _corpus_owner_id(user)
    )
    if passage is None:
        raise HTTPException(status_code=404, detail="Passage not found.")
    return passage


@app.post("/ask")
@_rate_limited(select=_ask_request_limits)
async def ask_question(
    payload: AskRequest,
    http_request: Request,
    user: AuthenticatedUser | None = Depends(get_optional_user),
) -> dict:
    if user is None and (
        payload.conversation_id or payload.use_web or payload.generate_image
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to save conversations, search the web, or generate images.",
        )
    return await _answer_and_record(
        payload.question,
        payload.conversation_id,
        use_web=payload.use_web,
        generate_image_requested=payload.generate_image,
        conversation_owner_id=user.uid if user else None,
        corpus_owner_id=_corpus_owner_id(user),
        include_sources=user is not None,
    )


@app.post("/ask/image")
@_rate_limited(select=_image_ask_limits)
async def ask_question_with_image(
    question: str = Form(min_length=1, max_length=2000),
    image: UploadFile = File(...),
    conversation_id: str | None = Form(default=None, min_length=1, max_length=64),
    use_web: bool = Form(default=False),
    generate_image: bool = Form(default=False),
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """Answer one multimodal prompt without persisting the attached image."""
    if image.content_type not in ALLOWED_PROMPT_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Choose a PNG, JPEG, or WebP image.",
        )
    image_data = await image.read(MAX_PROMPT_IMAGE_BYTES + 1)
    if len(image_data) > MAX_PROMPT_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Choose an image no larger than 10 MB.",
        )
    if not image_data:
        raise HTTPException(status_code=400, detail="The attached image is empty.")
    if _detected_prompt_image_type(image_data) != image.content_type:
        raise HTTPException(
            status_code=400,
            detail="The image contents do not match its file type.",
        )
    return await _answer_and_record(
        question,
        conversation_id,
        image_data=image_data,
        image_mime_type=image.content_type,
        use_web=use_web,
        generate_image_requested=generate_image,
        conversation_owner_id=user.uid,
        corpus_owner_id=_corpus_owner_id(user),
        include_sources=True,
    )


async def _answer_and_record(
    raw_question: str,
    conversation_id: str | None,
    *,
    image_data: bytes | None = None,
    image_mime_type: str | None = None,
    use_web: bool = False,
    generate_image_requested: bool = False,
    conversation_owner_id: str | None,
    corpus_owner_id: str,
    include_sources: bool,
) -> dict:
    from rag_engine import (
        ask_with_sources,
        generate_image,
        image_prompt_for_answer,
        search_web,
    )

    if conversation_id and not conversation_owner_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to continue a saved conversation.",
        )
    if conversation_id and not await run_in_threadpool(
        conversation_exists,
        CONVERSATION_DB_PATH,
        conversation_id,
        conversation_owner_id,
    ):
        raise HTTPException(status_code=404, detail="Conversation not found.")

    try:
        question = raw_question.strip()
        image_kwargs = {}
        if image_data is not None:
            image_kwargs = {
                "image_data": image_data,
                "image_mime_type": image_mime_type,
            }
        history = []
        if conversation_id:
            from conversation_memory import MAX_HISTORY_EXCHANGES

            history = await run_in_threadpool(
                get_recent_history, CONVERSATION_DB_PATH,
                conversation_id, MAX_HISTORY_EXCHANGES,
                conversation_owner_id,
            )
        if use_web:
            result = await run_in_threadpool(
                search_web,
                question,
                chat_history=history,
                **image_kwargs,
            )
        else:
            document_kwargs = {
                "prepare_image_prompt": True,
            } if generate_image_requested else {}
            document_kwargs["owner_id"] = corpus_owner_id
            result = await run_in_threadpool(
                ask_with_sources,
                question,
                chat_history=history,
                **image_kwargs,
                **document_kwargs,
            )
        image_result = None
        image_prompt = result.pop("image_prompt", "")
        if generate_image_requested:
            if use_web:
                image_prompt = image_prompt_for_answer(question, result["answer"])
            if not image_prompt:
                raise RuntimeError("No grounded image prompt was prepared.")
            image_result = await run_in_threadpool(generate_image, image_prompt)
        exchange_id = str(uuid4())
        if conversation_owner_id:
            conversation_id = await run_in_threadpool(
                record_exchange,
                CONVERSATION_DB_PATH,
                conversation_id,
                question,
                result["answer"],
                result.get("sources", []),
                result.get("timings", {}).get("total_s"),
                exchange_id,
                result.get("answer_basis", "documents"),
                result.get("web_search_available", False),
                generate_image_requested,
                image_prompt,
                conversation_owner_id,
            )
        response = {
            **result,
            "conversation_id": conversation_id,
            "exchange_id": exchange_id,
            "web_search_available": False,
            "image_generation_available": False,
        }
        if not include_sources:
            # The model still uses the shared corpus to ground the answer, but
            # anonymous callers must not receive source names or passage text.
            response["answer"] = _redact_guest_citations(response["answer"])
            response.pop("sources", None)
        if image_result is not None:
            image_id = str(uuid4())
            destination = await run_in_threadpool(
                _save_generated_image, image_id, image_result["image_data"]
            )
            attached = await run_in_threadpool(
                attach_generated_image,
                CONVERSATION_DB_PATH,
                conversation_id,
                exchange_id,
                image_id,
                image_result["image_mime_type"],
                conversation_owner_id,
            )
            if not attached:
                destination.unlink(missing_ok=True)
                raise RuntimeError("The generated image could not be attached to the answer.")
            response.update({
                "generated_image_id": image_id,
                "generated_image_url": f"/generated-images/{image_id}",
                "generated_image_mime_type": image_result["image_mime_type"],
            })
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/conversations")
async def conversation_history(
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    conversations = await run_in_threadpool(
        list_conversations, CONVERSATION_DB_PATH, 50, user.uid
    )
    return {"conversations": conversations}


@app.get("/conversations/{conversation_id}")
async def conversation_detail(
    conversation_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    conversation = await run_in_threadpool(
        get_conversation, CONVERSATION_DB_PATH, conversation_id, user.uid
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@app.put("/conversations/{conversation_id}/exchanges/{exchange_id}")
@_rate_limited(rates=("ask",), concurrency=("interactive",))
async def edit_exchange(
    conversation_id: str,
    exchange_id: str,
    request: EditExchangeRequest,
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """Edit one prompt, regenerate it, and discard later branch exchanges."""
    from conversation_memory import MAX_HISTORY_EXCHANGES
    from rag_engine import ask_with_sources

    question = request.question.strip()
    history = await run_in_threadpool(
        get_history_before_exchange,
        CONVERSATION_DB_PATH,
        conversation_id,
        exchange_id,
        MAX_HISTORY_EXCHANGES,
        user.uid,
    )
    if history is None:
        raise HTTPException(status_code=404, detail="Exchange not found.")
    try:
        result = await run_in_threadpool(
            ask_with_sources,
            question,
            chat_history=history,
            owner_id=_corpus_owner_id(user),
        )
        replaced = await run_in_threadpool(
            replace_exchange_and_truncate,
            CONVERSATION_DB_PATH,
            conversation_id,
            exchange_id,
            question,
            result["answer"],
            result.get("sources", []),
            result.get("timings", {}).get("total_s"),
            result.get("answer_basis", "documents"),
            result.get("web_search_available", False),
            result.get("image_generation_available", False),
            result.get("image_prompt", ""),
            user.uid,
        )
        if not replaced:
            raise HTTPException(status_code=404, detail="Exchange not found.")
        return {
            **result,
            "conversation_id": conversation_id,
            "exchange_id": exchange_id,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/conversations/{conversation_id}/exchanges/{exchange_id}/search-web")
@_rate_limited(rates=("web",), concurrency=("interactive", "web"))
async def search_web_for_exchange(
    conversation_id: str,
    exchange_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """Run Google Search only after an eligible document answer offers it."""
    from conversation_memory import MAX_HISTORY_EXCHANGES
    from rag_engine import search_web

    conversation = await run_in_threadpool(
        get_conversation, CONVERSATION_DB_PATH, conversation_id, user.uid
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    exchange = next(
        (item for item in conversation["exchanges"] if item["id"] == exchange_id),
        None,
    )
    if exchange is None:
        raise HTTPException(status_code=404, detail="Exchange not found.")
    if not exchange["web_search_available"]:
        raise HTTPException(
            status_code=409,
            detail="Web search is not available for this answer.",
        )
    if conversation["exchanges"][-1]["id"] != exchange_id:
        raise HTTPException(
            status_code=409,
            detail="Only the latest eligible answer can search the web.",
        )
    history = await run_in_threadpool(
        get_history_before_exchange,
        CONVERSATION_DB_PATH,
        conversation_id,
        exchange_id,
        MAX_HISTORY_EXCHANGES,
        user.uid,
    )
    try:
        result = await run_in_threadpool(
            search_web, exchange["question"], chat_history=history or [],
        )
        replaced = await run_in_threadpool(
            replace_latest_exchange_with_web_answer,
            CONVERSATION_DB_PATH,
            conversation_id,
            exchange_id,
            result["answer"],
            result.get("sources", []),
            result.get("timings", {}).get("total_s"),
            user.uid,
        )
        if not replaced:
            raise HTTPException(
                status_code=409,
                detail="Only the latest eligible answer can search the web.",
            )
        return {
            **result,
            "conversation_id": conversation_id,
            "exchange_id": exchange_id,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/conversations/{conversation_id}/exchanges/{exchange_id}/generate-image")
@_rate_limited(rates=("image",), concurrency=("interactive", "image"))
async def generate_image_for_exchange(
    conversation_id: str,
    exchange_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """Generate one image only after the structured answer offers the action."""
    from rag_engine import generate_image

    conversation = await run_in_threadpool(
        get_conversation, CONVERSATION_DB_PATH, conversation_id, user.uid
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    exchange = next(
        (item for item in conversation["exchanges"] if item["id"] == exchange_id),
        None,
    )
    if exchange is None:
        raise HTTPException(status_code=404, detail="Exchange not found.")
    if not exchange["image_generation_available"] or not exchange["image_prompt"]:
        raise HTTPException(
            status_code=409,
            detail="Image generation is not available for this answer.",
        )

    try:
        result = await run_in_threadpool(generate_image, exchange["image_prompt"])
        image_id = str(uuid4())
        destination = await run_in_threadpool(
            _save_generated_image, image_id, result["image_data"]
        )
        attached = await run_in_threadpool(
            attach_generated_image,
            CONVERSATION_DB_PATH,
            conversation_id,
            exchange_id,
            image_id,
            result["image_mime_type"],
            user.uid,
        )
        if not attached:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=409,
                detail="An image has already been generated for this answer.",
            )
        return {
            "conversation_id": conversation_id,
            "exchange_id": exchange_id,
            "generated_image_id": image_id,
            "generated_image_url": f"/generated-images/{image_id}",
            "generated_image_mime_type": result["image_mime_type"],
            "timings": result.get("timings", {}),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/generated-images/{image_id}", response_class=FileResponse)
async def generated_image(
    image_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> FileResponse:
    path = _generated_image_path(image_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Generated image not found.")
    metadata = await run_in_threadpool(
        get_generated_image_metadata, CONVERSATION_DB_PATH, image_id, user.uid
    )
    conversation_mime = (metadata or {}).get("generated_image_mime_type")
    if conversation_mime not in ALLOWED_PROMPT_IMAGE_TYPES:
        raise HTTPException(status_code=404, detail="Generated image not found.")
    return FileResponse(path, media_type=conversation_mime)


@app.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_conversation(
    conversation_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> None:
    conversation = await run_in_threadpool(
        get_conversation, CONVERSATION_DB_PATH, conversation_id, user.uid
    )
    deleted = await run_in_threadpool(
        delete_conversation, CONVERSATION_DB_PATH, conversation_id, user.uid
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    for exchange in (conversation or {}).get("exchanges", []):
        image_id = exchange.get("generated_image_id")
        if image_id:
            _generated_image_path(image_id).unlink(missing_ok=True)


@app.post("/index")
@_rate_limited(rates=("ingest",), concurrency=("ingest",))
async def index_file(
    request: IndexRequest,
    user: AuthenticatedUser = Depends(require_admin),
) -> dict:
    from indexer import index_document, is_document_indexed

    try:
        document_path = _resolve_data_document(
            request.filename, SHARED_CORPUS_OWNER_ID
        )
        tenant_root = _tenant_data_root(SHARED_CORPUS_OWNER_ID)
        source_name = (
            document_path.relative_to(tenant_root).as_posix()
            if tenant_root in document_path.parents
            else document_path.relative_to(Path(DATA_DIR).resolve()).as_posix()
        )
        if await run_in_threadpool(
            is_document_indexed,
            source_name,
            file_path=document_path,
            owner_id=SHARED_CORPUS_OWNER_ID,
        ):
            return {"source": source_name, "pages_with_text": 0,
                    "chunks_indexed": 0, "deduplicated": True}
        pages = await run_in_threadpool(_extract_document, document_path)
        chunks = await run_in_threadpool(
            index_document,
            pages,
            source_name,
            SOURCE_TYPES[document_path.suffix.lower()],
            file_path=document_path,
            owner_id=SHARED_CORPUS_OWNER_ID,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "source": source_name,
        "pages_with_text": len(pages),
        "chunks_indexed": chunks,
        "deduplicated": chunks == 0 and bool(pages),
    }


@app.post("/upload", status_code=status.HTTP_201_CREATED)
@_rate_limited(rates=("ingest",), concurrency=("ingest",))
async def upload_document(
    file: UploadFile = File(...),
    relative_path: str | None = Form(default=None),
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """Save and index a shared or private document, reusing local copies."""
    original_name = Path(file.filename or "").name
    if not original_name or original_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="A document filename is required.")
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, and Markdown files can be uploaded.",
        )
    if file.content_type not in ALLOWED_CONTENT_TYPES[extension]:
        raise HTTPException(
            status_code=400,
            detail=f"The uploaded file type does not match its {extension} extension.",
        )

    corpus_owner_id = _corpus_owner_id(user)
    data_root = _tenant_data_root(corpus_owner_id)
    data_root.mkdir(parents=True, exist_ok=True)

    path_parts = [original_name]
    if relative_path:
        if "\\" in relative_path:
            raise HTTPException(status_code=400, detail="Invalid document path.")
        path_parts = relative_path.split("/")
        if (
            any(part in {"", ".", ".."} for part in path_parts)
            or path_parts[-1] != original_name
        ):
            raise HTTPException(status_code=400, detail="Invalid document path.")

    source_name = "/".join(path_parts)
    destination = data_root.joinpath(*path_parts).resolve()
    if destination.parent != data_root and data_root not in destination.parents:
        raise HTTPException(status_code=400, detail="Invalid document filename.")

    from indexer import index_document, is_document_indexed

    total = 0
    created_by_request = False
    used_existing_file = destination.exists()
    try:
        if not used_existing_file:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                created_by_request = True
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="Document is larger than the 500 MB upload limit.",
                        )
                    output.write(chunk)

        if await run_in_threadpool(
            is_document_indexed,
            source_name,
            file_path=destination,
            owner_id=corpus_owner_id,
        ):
            raise HTTPException(
                status_code=409,
                detail=f"'{source_name}' is already indexed in the library.",
            )

        pages = await run_in_threadpool(_extract_document, destination)
        if not pages:
            raise HTTPException(
                status_code=422,
                detail="No readable text could be extracted from this document.",
            )
        chunks = await run_in_threadpool(
            index_document, pages, source_name, SOURCE_TYPES[extension],
            file_path=destination,
            owner_id=corpus_owner_id,
        )
    except ValueError as exc:
        if created_by_request:
            destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        if created_by_request:
            destination.unlink(missing_ok=True)
        raise
    except RuntimeError as exc:
        if created_by_request:
            destination.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        if created_by_request:
            destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    return {
        "source": source_name,
        "pages_with_text": len(pages),
        "chunks_indexed": chunks,
        "deduplicated": chunks == 0,
        "used_existing_file": used_existing_file,
    }


@app.post("/index/youtube")
@_rate_limited(rates=("ingest",), concurrency=("ingest",))
async def index_youtube(
    request: YouTubeIndexRequest,
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    from youtube_ingester import ingest_youtube

    try:
        return await run_in_threadpool(
            ingest_youtube, request.url, _corpus_owner_id(user)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/index/folder")
@_rate_limited(rates=("ingest",), concurrency=("ingest",))
async def index_local_folder(
    request: FolderIndexRequest,
    user: AuthenticatedUser = Depends(require_admin),
) -> dict:
    """Index an allowlisted server-local folder without blocking the event loop."""
    try:
        folder_path = resolve_allowed_folder(
            request.folder_path, INDEX_FOLDER_ROOTS
        )
        return await run_in_threadpool(
            index_folder,
            folder_path,
            request.recursive,
            owner_id=SHARED_CORPUS_OWNER_ID,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
