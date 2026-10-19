"""Shared ingestion for local PDF, TXT, Markdown files, and folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from config import OCR_CACHE_DIR, OCR_CACHE_ENABLED


SUPPORTED_DOCUMENT_EXTENSIONS = frozenset({".pdf", ".txt", ".md"})
SOURCE_TYPES = {".pdf": "pdf", ".txt": "text", ".md": "markdown"}

# Bump when the cached structure changes, so old entries are ignored rather
# than silently misread.
_CACHE_VERSION = 1


def _cache_fingerprint(document_path: Path) -> dict:
    """Everything that, if changed, invalidates a cached OCR result.

    Both the SOURCE (size + mtime) and the SETTINGS that produced the text
    (DPI, backend) are recorded. Caching on filename alone would happily serve
    Tesseract output for a document you have since re-run under Vision.
    """
    from config import OCR_BACKEND, PDF_DPI
    from indexer import file_sha256

    stat = document_path.stat()
    return {
        "version": _CACHE_VERSION,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "pdf_dpi": PDF_DPI,
        "ocr_backend": OCR_BACKEND,
        "file_sha256": file_sha256(document_path),
    }


def _cache_path(document_path: Path) -> Path:
    """Where a document's cached OCR lives.

    Keep the legacy filename-based cache location. Its fingerprint verifies
    file bytes and extraction settings before reuse; Chroma identity is now
    content-based and independent of this cache filename.
    """
    from indexer import _document_id

    extension = document_path.suffix.lower()
    doc_id = _document_id(document_path.name, SOURCE_TYPES.get(extension, "pdf"))
    return Path(OCR_CACHE_DIR) / f"{doc_id}.json"


def load_cached_pages(document_path: Path) -> list[dict] | None:
    """Return cached OCR pages for this document, or None if unusable."""
    if not OCR_CACHE_ENABLED:
        return None
    path = _cache_path(document_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None                      # a corrupt cache must never block work
    if payload.get("fingerprint") != _cache_fingerprint(document_path):
        return None                      # source or OCR settings changed
    pages = payload.get("pages")
    return pages if isinstance(pages, list) and pages else None


def save_cached_pages(document_path: Path, pages: list[dict]) -> None:
    """Persist OCR pages. Best-effort: a cache failure must not fail ingestion."""
    if not OCR_CACHE_ENABLED or not pages:
        return
    path = _cache_path(document_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_name": document_path.name,
            "fingerprint": _cache_fingerprint(document_path),
            "pages": pages,
        }
        # Write to a temp file and rename, so an interrupted write cannot leave
        # a half-written cache that later reads as valid JSON.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


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
    """Extract a supported document using PDF routing or direct UTF-8 reading.

    PDF results are cached to disk. OCR is the slow, BILLED half of ingestion
    and embedding is the half that trips on quota; without a cache between them
    a single 429 discards every page already read for that document.

    Only PDFs are cached -- re-reading a .txt is free, so caching it would add
    staleness risk for no gain.
    """
    extension = document_path.suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError("Only PDF, TXT, and Markdown files can be indexed.")
    if extension != ".pdf":
        return extract_text_document(document_path)

    cached = load_cached_pages(document_path)
    if cached is not None:
        from rich.console import Console

        Console().print(
            f"   [green]💾 OCR cache hit[/green] — {len(cached)} pages, no OCR calls."
        )
        return cached

    from ocr_engine import extract_text_from_pdf

    pages = extract_text_from_pdf(str(document_path))
    save_cached_pages(document_path, pages)
    return pages


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


def indexed_file_names() -> set[str]:
    """Legacy basename inventory, retained for display/caller compatibility.

    Do not use this as a duplicate guard: different documents can have the same
    basename. Ingestion now checks file content and complete extracted text.
    """
    from indexer import _get_collection

    try:
        metadatas = _get_collection().get(include=["metadatas"])["metadatas"] or []
    except Exception:
        # An unreadable collection must not stop ingestion.
        return set()
    # A document is only "indexed" if it is COMPLETE. index_chunks writes each
    # batch as it is embedded, so a run killed by quota leaves a partial
    # document behind -- and counting that as indexed would skip it forever,
    # leaving a book permanently half-searchable. Chunks carry `chunk_total`,
    # so completeness is checkable; chunks written before that field existed
    # have no total and are trusted as complete, since the old all-or-nothing
    # path could not leave a partial document in the first place.
    stored_counts: dict[str, int] = {}
    expected: dict[str, int] = {}
    names: dict[str, str] = {}
    for md in metadatas:
        if not md or not md.get("source_name"):
            continue
        doc_id = str(md.get("document_id", md["source_name"]))
        stored_counts[doc_id] = stored_counts.get(doc_id, 0) + 1
        names[doc_id] = str(md["source_name"]).rsplit("/", 1)[-1].casefold()
        total = md.get("chunk_total")
        if isinstance(total, int) and total > 0:
            expected[doc_id] = max(expected.get(doc_id, 0), total)

    return {
        name
        for doc_id, name in names.items()
        if stored_counts[doc_id] >= expected.get(doc_id, 0)
    }


def index_folder(
    folder_path: str | Path,
    recursive: bool = True,
    *,
    extract: Callable[[Path], list[dict]] = extract_document,
    force: bool = False,
    owner_id: str | None = None,
) -> dict:
    """Index every supported file in a folder and isolate per-file failures.

    Already-indexed files are skipped unless `force` is set. index_document()
    upserts and clears stale positions, so re-indexing is CORRECT either way --
    but re-extracting a 902-page scan costs ~14 minutes and real Vision spend,
    so doing it by default makes the command needlessly expensive to re-run.
    """
    from indexer import index_document, is_document_indexed

    root = Path(folder_path).expanduser().resolve()
    documents = discover_documents(root, recursive=recursive)

    # Smallest first. discover_documents() keeps its documented path ordering
    # for callers that depend on it; the size ordering is an operational
    # concern that belongs here: a misconfiguration then surfaces on a 1-page
    # file in seconds rather than after 14 minutes on the largest scan.
    documents = sorted(documents, key=lambda path: path.stat().st_size)

    results = []
    chunks_indexed = 0

    for document_path in documents:
        relative_path = document_path.relative_to(root).as_posix()
        source_name = f"{root.name}/{relative_path}"

        try:
            if not force and is_document_indexed(
                source_name, file_path=document_path, owner_id=owner_id,
            ):
                results.append({
                    "source": source_name,
                    "status": "skipped",
                    "reason": "Identical file content is already indexed.",
                    "chunks_indexed": 0,
                })
                continue
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
                file_path=document_path,
                source_metadata={
                    "folder_root": str(root),
                    "relative_path": relative_path,
                    "file_extension": document_path.suffix.lower(),
                },
                owner_id=owner_id,
            )
            chunks_indexed += chunk_count
            results.append({
                "source": source_name,
                "status": "indexed" if chunk_count else "skipped",
                **({"reason": "No new chunks; content already indexed or too short."}
                   if not chunk_count else {}),
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
