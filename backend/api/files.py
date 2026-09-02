"""Validated file and tenant-path helpers used by API routes."""

import hashlib
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException

from config import (
    DATA_DIR,
    FIREBASE_PROJECT_ID,
    GENERATED_IMAGE_DIR,
    LEGACY_ADMIN_UID,
    SHARED_CORPUS_OWNER_ID,
)
from document_ingester import SUPPORTED_DOCUMENT_EXTENSIONS


def detected_prompt_image_type(data: bytes) -> str | None:
    """Recognize supported image formats from their file signatures."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def generated_image_path(image_id: str, image_root: str | Path = GENERATED_IMAGE_DIR) -> Path:
    """Resolve only canonical UUID filenames inside the generated-image root."""
    try:
        canonical_id = str(UUID(image_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Generated image not found.") from exc
    if canonical_id != image_id:
        raise HTTPException(status_code=404, detail="Generated image not found.")
    return Path(image_root).resolve() / canonical_id


def save_generated_image(
    image_id: str,
    data: bytes,
    image_root: str | Path = GENERATED_IMAGE_DIR,
) -> Path:
    directory = Path(image_root).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / image_id
    temporary = directory / f".{image_id}.tmp"
    temporary.write_bytes(data)
    temporary.replace(destination)
    return destination


def tenant_data_root(owner_id: str, data_root: str | Path = DATA_DIR) -> Path:
    """Use an opaque directory so Firebase UIDs never become path segments."""
    if not FIREBASE_PROJECT_ID:
        return Path(data_root).resolve()
    tenant = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()
    return (Path(data_root).resolve() / "users" / tenant).resolve()


def resolve_data_document(
    filename: str,
    owner_id: str | None = None,
    data_root: str | Path = DATA_DIR,
) -> Path:
    """Resolve a supported document while preventing traversal outside ``data/``."""
    root = tenant_data_root(owner_id, data_root) if owner_id else Path(data_root).resolve()
    relative_path = Path(filename)
    if relative_path.parts[:1] == ("data",):
        relative_path = Path(*relative_path.parts[1:])
    candidate = (root / relative_path).resolve()
    if root != candidate.parent and root not in candidate.parents:
        raise ValueError("The document must be located inside the data directory.")
    if candidate.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError("Only PDF, TXT, and Markdown files can be indexed.")
    if not candidate.is_file() and owner_id in {LEGACY_ADMIN_UID, SHARED_CORPUS_OWNER_ID}:
        legacy_root = Path(data_root).resolve()
        legacy_candidate = (legacy_root / relative_path).resolve()
        if (
            legacy_candidate.parent == legacy_root or legacy_root in legacy_candidate.parents
        ) and legacy_candidate.is_file():
            candidate = legacy_candidate
    if not candidate.is_file():
        raise ValueError(f"Document not found: {filename}")
    return candidate
