"""FastAPI interface for the Hindi Textbook RAG pipeline."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from config import DATA_DIR


app = FastAPI(
    title="Hindi Textbook RAG API",
    description="Index Hindi/English PDFs and ask grounded questions.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class IndexRequest(BaseModel):
    filename: str = Field(
        min_length=1,
        description="PDF filename located inside the project's data directory.",
    )


def _resolve_data_pdf(filename: str) -> Path:
    """Resolve a PDF path while preventing traversal outside ``data/``."""
    data_root = Path(DATA_DIR).resolve()
    candidate = (data_root / filename).resolve()
    if data_root != candidate.parent and data_root not in candidate.parents:
        raise ValueError("The PDF must be located inside the data directory.")
    if candidate.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be indexed.")
    if not candidate.is_file():
        raise ValueError(f"PDF not found: {filename}")
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

    try:
        return await run_in_threadpool(ask_with_sources, request.question.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/index")
async def index_pdf(request: IndexRequest) -> dict:
    from indexer import index_document
    from ocr_engine import extract_text_from_pdf

    try:
        pdf_path = _resolve_data_pdf(request.filename)
        pages = await run_in_threadpool(extract_text_from_pdf, str(pdf_path))
        chunks = await run_in_threadpool(index_document, pages, pdf_path.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "source": pdf_path.name,
        "pages_with_text": len(pages),
        "chunks_indexed": chunks,
        "deduplicated": chunks == 0 and bool(pages),
    }
