"""FastAPI interface for the Hindi Textbook RAG pipeline."""

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import CONVERSATION_DB_PATH, DATA_DIR, INDEX_FOLDER_ROOTS
from conversation_store import (
    conversation_exists,
    delete_conversation,
    get_conversation,
    list_conversations,
    record_exchange,
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
    title="Hindi Textbook RAG API",
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
ALLOWED_CONTENT_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)


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


def _resolve_data_document(filename: str) -> Path:
    """Resolve a supported document while preventing traversal outside ``data/``."""
    data_root = Path(DATA_DIR).resolve()
    candidate = (data_root / filename).resolve()
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
        "name": "Hindi Textbook RAG API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    from indexer import get_stats

    return {"status": "ok", **get_stats()}


@app.post("/ask")
async def ask_question(request: AskRequest) -> dict:
    from rag_engine import ask_with_sources

    if request.conversation_id and not await run_in_threadpool(
        conversation_exists, CONVERSATION_DB_PATH, request.conversation_id
    ):
        raise HTTPException(status_code=404, detail="Conversation not found.")

    try:
        question = request.question.strip()
        result = await run_in_threadpool(ask_with_sources, question)
        conversation_id = await run_in_threadpool(
            record_exchange,
            CONVERSATION_DB_PATH,
            request.conversation_id,
            question,
            result["answer"],
            result.get("sources", []),
            result.get("timings", {}).get("total_s"),
        )
        return {**result, "conversation_id": conversation_id}
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


@app.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_conversation(conversation_id: str) -> None:
    deleted = await run_in_threadpool(
        delete_conversation, CONVERSATION_DB_PATH, conversation_id
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")


@app.post("/index")
async def index_file(request: IndexRequest) -> dict:
    from indexer import index_document

    try:
        document_path = _resolve_data_document(request.filename)
        pages = await run_in_threadpool(_extract_document, document_path)
        chunks = await run_in_threadpool(
            index_document,
            pages,
            document_path.name,
            SOURCE_TYPES[document_path.suffix.lower()],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "source": document_path.name,
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

    if await run_in_threadpool(is_document_indexed, source_name):
        raise HTTPException(
            status_code=409,
            detail=f"'{source_name}' is already indexed in the library.",
        )

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

        pages = await run_in_threadpool(_extract_document, destination)
        if not pages:
            raise HTTPException(
                status_code=422,
                detail="No readable text could be extracted from this document.",
            )
        chunks = await run_in_threadpool(
            index_document, pages, source_name, SOURCE_TYPES[extension]
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
