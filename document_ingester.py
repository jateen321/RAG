"""Shared ingestion for local PDF, TXT, Markdown files, and folders."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


SUPPORTED_DOCUMENT_EXTENSIONS = frozenset({".pdf", ".txt", ".md"})
SOURCE_TYPES = {".pdf": "pdf", ".txt": "text", ".md": "markdown"}


def extract_text_document(document_path: Path) -> list[dict]:
    """Read one UTF-8 text document as a page-like unit for the indexer."""
    try:
        text = document_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{document_path.name} must use UTF-8 text encoding."
        ) from exc
    return [{"page": 1, "text": text, "method": "text"}] if text.strip() else []


def extract_document(document_path: Path) -> list[dict]:
    """Extract a supported document using PDF routing or direct UTF-8 reading."""
    extension = document_path.suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError("Only PDF, TXT, and Markdown files can be indexed.")
    if extension == ".pdf":
        from ocr_engine import extract_text_from_pdf

        return extract_text_from_pdf(str(document_path))
    return extract_text_document(document_path)


def discover_documents(folder_path: Path, recursive: bool = True) -> list[Path]:
    """Return supported regular files in stable relative-path order."""
    root = folder_path.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Folder not found: {folder_path}")

    candidates = root.rglob("*") if recursive else root.glob("*")
    documents = [
        path
        for path in candidates
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    ]
    return sorted(documents, key=lambda path: path.relative_to(root).as_posix().casefold())


def resolve_allowed_folder(folder_path: str | Path, allowed_roots: list[str]) -> Path:
    """Resolve an API folder path and require it to be under an allowlisted root."""
    candidate = Path(folder_path).expanduser().resolve()
    if not candidate.is_dir():
        raise ValueError(f"Folder not found: {folder_path}")

    roots = [Path(root).expanduser().resolve() for root in allowed_roots]
    if not any(candidate == root or root in candidate.parents for root in roots):
        raise ValueError(
            "Folder is outside INDEX_FOLDER_ROOTS. Add its parent directory "
            "to the API allowlist."
        )
    return candidate


def index_folder(
    folder_path: str | Path,
    recursive: bool = True,
    *,
    extract: Callable[[Path], list[dict]] = extract_document,
) -> dict:
    """Index every supported file in a folder and isolate per-file failures."""
    from indexer import index_document

    root = Path(folder_path).expanduser().resolve()
    documents = discover_documents(root, recursive=recursive)
    results = []
    chunks_indexed = 0

    for document_path in documents:
        relative_path = document_path.relative_to(root).as_posix()
        source_name = f"{root.name}/{relative_path}"
        try:
            pages = extract(document_path)
            if not pages:
                results.append({
                    "source": source_name,
                    "status": "skipped",
                    "reason": "No readable text found.",
                    "chunks_indexed": 0,
                })
                continue

            chunk_count = index_document(
                pages,
                source_name,
                SOURCE_TYPES[document_path.suffix.lower()],
                document_key=str(document_path),
                source_metadata={
                    "folder_root": str(root),
                    "relative_path": relative_path,
                    "file_extension": document_path.suffix.lower(),
                },
            )
            chunks_indexed += chunk_count
            results.append({
                "source": source_name,
                "status": "indexed",
                "pages_with_text": len(pages),
                "chunks_indexed": chunk_count,
            })
        except Exception as exc:
            results.append({
                "source": source_name,
                "status": "failed",
                "reason": str(exc),
                "chunks_indexed": 0,
            })

    return {
        "folder_path": str(root),
        "recursive": recursive,
        "files_found": len(documents),
        "files_indexed": sum(r["status"] == "indexed" for r in results),
        "files_skipped": sum(r["status"] == "skipped" for r in results),
        "files_failed": sum(r["status"] == "failed" for r in results),
        "chunks_indexed": chunks_indexed,
        "results": results,
    }
