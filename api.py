"""FastAPI interface for the Sarthi AI RAG pipeline."""

import os
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import (
    CONVERSATION_DB_PATH,
    DATA_DIR,
    GENERATED_IMAGE_DIR,
    INDEX_FOLDER_ROOTS,
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


app = FastAPI(
    title="Sarthi AI API",
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)
    use_web: bool = False
    generate_image: bool = False


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


def _resolve_data_document(filename: str) -> Path:
    """Resolve a supported document while preventing traversal outside ``data/``."""
    data_root = Path(DATA_DIR).resolve()
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
    if not candidate.is_file():
        raise ValueError(f"Document not found: {filename}")
    return candidate


@app.get("/")
def root() -> dict:
    return {
        "name": "Sarthi AI API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    from indexer import get_stats

    return {"status": "ok", **get_stats()}


@app.get("/documents/{source_path:path}", response_class=FileResponse)
def open_document(source_path: str) -> FileResponse:
    """Open an indexed local document while keeping access inside ``data/``."""
    try:
        document_path = _resolve_data_document(source_path)
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
) -> dict:
    """Resolve citations saved before source responses included chunk IDs."""
    from indexer import get_chunk_by_citation

    passage = await run_in_threadpool(
        get_chunk_by_citation, source, page, preview
    )
    if passage is None:
        raise HTTPException(status_code=404, detail="Exact cited passage not found.")
    return passage


@app.get("/passages/{chunk_id}")
async def passage_detail(
    chunk_id: str,
    source: str = Query(min_length=1, max_length=4096),
) -> dict:
    """Return one cited passage, scoped to its source to prevent id confusion."""
    from indexer import get_chunk

    passage = await run_in_threadpool(get_chunk, chunk_id, source)
    if passage is None:
        raise HTTPException(status_code=404, detail="Passage not found.")
    return passage


@app.post("/ask")
async def ask_question(request: AskRequest) -> dict:
    return await _answer_and_record(
        request.question,
        request.conversation_id,
        use_web=request.use_web,
        generate_image_requested=request.generate_image,
    )


@app.post("/ask/image")
async def ask_question_with_image(
    question: str = Form(min_length=1, max_length=2000),
    image: UploadFile = File(...),
    conversation_id: str | None = Form(default=None, min_length=1, max_length=64),
    use_web: bool = Form(default=False),
    generate_image: bool = Form(default=False),
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
    )


async def _answer_and_record(
    raw_question: str,
    conversation_id: str | None,
    *,
    image_data: bytes | None = None,
    image_mime_type: str | None = None,
    use_web: bool = False,
    generate_image_requested: bool = False,
) -> dict:
    from rag_engine import (
        ask_with_sources,
        generate_image,
        image_prompt_for_answer,
        search_web,
    )

    if conversation_id and not await run_in_threadpool(
        conversation_exists, CONVERSATION_DB_PATH, conversation_id
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
            )
        if use_web:
            result = await run_in_threadpool(
                search_web,
                question,
                chat_history=history,
                **image_kwargs,
            )
        else:
            result = await run_in_threadpool(
                ask_with_sources,
                question,
                chat_history=history,
                **image_kwargs,
            )
        image_result = None
        image_prompt = ""
        if generate_image_requested:
            image_prompt = image_prompt_for_answer(question, result["answer"])
            image_result = await run_in_threadpool(generate_image, image_prompt)
        exchange_id = str(uuid4())
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
        )
        response = {
            **result,
            "conversation_id": conversation_id,
            "exchange_id": exchange_id,
            "web_search_available": False,
            "image_generation_available": False,
        }
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
async def conversation_history() -> dict:
    conversations = await run_in_threadpool(
        list_conversations, CONVERSATION_DB_PATH
    )
    return {"conversations": conversations}


@app.get("/conversations/{conversation_id}")
async def conversation_detail(conversation_id: str) -> dict:
    conversation = await run_in_threadpool(
        get_conversation, CONVERSATION_DB_PATH, conversation_id
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@app.put("/conversations/{conversation_id}/exchanges/{exchange_id}")
async def edit_exchange(
    conversation_id: str,
    exchange_id: str,
    request: EditExchangeRequest,
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
    )
    if history is None:
        raise HTTPException(status_code=404, detail="Exchange not found.")
    try:
        result = await run_in_threadpool(
            ask_with_sources, question, chat_history=history,
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
async def search_web_for_exchange(conversation_id: str, exchange_id: str) -> dict:
    """Run Google Search only after an eligible document answer offers it."""
    from conversation_memory import MAX_HISTORY_EXCHANGES
    from rag_engine import search_web

    conversation = await run_in_threadpool(
        get_conversation, CONVERSATION_DB_PATH, conversation_id
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
async def generate_image_for_exchange(conversation_id: str, exchange_id: str) -> dict:
    """Generate one image only after the structured answer offers the action."""
    from rag_engine import generate_image

    conversation = await run_in_threadpool(
        get_conversation, CONVERSATION_DB_PATH, conversation_id
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
async def generated_image(image_id: str) -> FileResponse:
    path = _generated_image_path(image_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Generated image not found.")
    metadata = await run_in_threadpool(
        get_generated_image_metadata, CONVERSATION_DB_PATH, image_id
    )
    conversation_mime = (metadata or {}).get("generated_image_mime_type")
    if conversation_mime not in ALLOWED_PROMPT_IMAGE_TYPES:
        raise HTTPException(status_code=404, detail="Generated image not found.")
    return FileResponse(path, media_type=conversation_mime)


@app.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_conversation(conversation_id: str) -> None:
    conversation = await run_in_threadpool(
        get_conversation, CONVERSATION_DB_PATH, conversation_id
    )
    deleted = await run_in_threadpool(
        delete_conversation, CONVERSATION_DB_PATH, conversation_id
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    for exchange in (conversation or {}).get("exchanges", []):
        image_id = exchange.get("generated_image_id")
        if image_id:
            _generated_image_path(image_id).unlink(missing_ok=True)


@app.post("/index")
async def index_file(request: IndexRequest) -> dict:
    from indexer import index_document, is_document_indexed

    try:
        document_path = _resolve_data_document(request.filename)
        source_name = document_path.relative_to(Path(DATA_DIR).resolve()).as_posix()
        if await run_in_threadpool(
            is_document_indexed, source_name, file_path=document_path
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
async def upload_document(
    file: UploadFile = File(...),
    relative_path: str | None = Form(default=None),
) -> dict:
    """Save and index one document, reusing an unindexed local copy if present."""
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

    data_root = Path(DATA_DIR).resolve()
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
            is_document_indexed, source_name, file_path=destination
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
async def index_youtube(request: YouTubeIndexRequest) -> dict:
    from youtube_ingester import ingest_youtube

    try:
        return await run_in_threadpool(ingest_youtube, request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/index/folder")
async def index_local_folder(request: FolderIndexRequest) -> dict:
    """Index an allowlisted server-local folder without blocking the event loop."""
    try:
        folder_path = resolve_allowed_folder(
            request.folder_path, INDEX_FOLDER_ROOTS
        )
        return await run_in_threadpool(
            index_folder, folder_path, request.recursive
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
