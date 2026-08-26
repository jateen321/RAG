"""FastAPI interface for the Hindi Textbook RAG pipeline."""

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import DATA_DIR


app = FastAPI(
    title="Hindi Textbook RAG API",
    description="Index PDF, TXT, Markdown, and YouTube sources, then ask grounded questions.",
    version="1.2.0",
)

_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("RAG_ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}
ALLOWED_CONTENT_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
}
SOURCE_TYPES = {".pdf": "pdf", ".txt": "text", ".md": "markdown"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
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


def _extract_text_document(document_path: Path) -> list[dict]:
    """Read a UTF-8 text document as page-like sections for the shared indexer."""
    try:
        text = document_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{document_path.name} must use UTF-8 text encoding."
        ) from exc
    return [{"page": 1, "text": text, "method": "text"}] if text.strip() else []


def _extract_document(document_path: Path) -> list[dict]:
    if document_path.suffix.lower() == ".pdf":
        from ocr_engine import extract_text_from_pdf

        return extract_text_from_pdf(str(document_path))
    return _extract_text_document(document_path)


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

    try:
        return await run_in_threadpool(ask_with_sources, request.question.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Save and index one supported document without traversal or replacement."""
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
    destination = (data_root / original_name).resolve()
    if destination.parent != data_root:
        raise HTTPException(status_code=400, detail="Invalid document filename.")
    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail=f"A file named '{original_name}' already exists in the library.",
        )

    total = 0
    created_by_request = False
    try:
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

        from indexer import index_document
        pages = await run_in_threadpool(_extract_document, destination)
        if not pages:
            raise HTTPException(
                status_code=422,
                detail="No readable text could be extracted from this document.",
            )
        chunks = await run_in_threadpool(
            index_document, pages, destination.name, SOURCE_TYPES[extension]
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
        "source": destination.name,
        "pages_with_text": len(pages),
        "chunks_indexed": chunks,
        "deduplicated": chunks == 0,
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
