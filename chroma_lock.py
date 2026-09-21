"""Cross-process coordination for the host-mounted Chroma database."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import pickle
import sqlite3
from typing import Iterator

import fcntl

from config import CHROMA_DB_PATH


class ChromaStoreError(RuntimeError):
    """The persisted Chroma store is unsafe to open for reads or writes."""


def validate_chroma_store(
    db_path: str | Path = CHROMA_DB_PATH,
    collection_name: str = "hindi_textbook",
) -> None:
    """Fail closed before Rust opens an invalid persisted HNSW segment.

    Chroma's Rust loader can segfault instead of raising when a persisted
    segment's ``index_metadata.pickle`` has ``dimensionality=None``.  Validate
    the small SQLite metadata and segment pickle first so callers get a safe,
    actionable exception and no write can trigger a destructive rebuild.
    Empty/new stores have no vector segment and are allowed through.
    """
    root = Path(db_path)
    sqlite_path = root / "chroma.sqlite3"
    if not sqlite_path.exists():
        return

    try:
        with sqlite3.connect(sqlite_path) as connection:
            integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
            if integrity != "ok":
                raise ChromaStoreError(
                    f"Chroma SQLite quick_check failed: {integrity}. "
                    "Restore a known-good chroma_db backup before indexing."
                )
            segments = connection.execute(
                """
                SELECT s.id, s.scope, c.dimension
                FROM segments AS s
                JOIN collections AS c ON c.id = s.collection
                WHERE c.name = ? AND s.scope = 'VECTOR'
                """,
                (collection_name,),
            ).fetchall()
            metadata_segment_ids = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT s.id
                    FROM segments AS s
                    JOIN collections AS c ON c.id = s.collection
                    WHERE c.name = ? AND s.scope = 'METADATA'
                    """,
                    (collection_name,),
                ).fetchall()
            ]
            rows = 0
            if metadata_segment_ids:
                placeholders = ",".join("?" for _ in metadata_segment_ids)
                rows = connection.execute(
                    f"SELECT COUNT(*) FROM embeddings WHERE segment_id IN ({placeholders})",
                    metadata_segment_ids,
                ).fetchone()[0]
            for segment_id, _scope, dimension in segments:
                if not rows:
                    continue

                segment_root = root / str(segment_id)
                metadata_path = segment_root / "index_metadata.pickle"
                if not metadata_path.exists():
                    raise ChromaStoreError(
                        f"Chroma vector segment {segment_id} has {rows} rows but "
                        "no index_metadata.pickle. Restore the segment backup; "
                        "do not run indexing against this store."
                    )
                try:
                    metadata = pickle.loads(metadata_path.read_bytes())
                except Exception as exc:  # noqa: BLE001 - actionable wrapper
                    raise ChromaStoreError(
                        f"Chroma vector metadata is unreadable: {metadata_path}. "
                        "Restore the segment backup before indexing."
                    ) from exc

                actual_dimension = metadata.get("dimensionality")
                if not isinstance(actual_dimension, int) or actual_dimension <= 0:
                    raise ChromaStoreError(
                        f"Chroma vector segment {segment_id} has invalid "
                        f"dimensionality={actual_dimension!r}. Restore a known-good "
                        "segment or rebuild the collection before indexing."
                    )
                if dimension is not None and actual_dimension != dimension:
                    raise ChromaStoreError(
                        f"Chroma dimension mismatch: SQLite={dimension}, "
                        f"HNSW={actual_dimension}. Restore or rebuild the collection."
                    )
    except ChromaStoreError:
        raise
    except sqlite3.Error as exc:
        raise ChromaStoreError(
            f"Unable to validate Chroma SQLite metadata: {exc}. "
            "Restore a known-good chroma_db backup before indexing."
        ) from exc


@contextmanager
def chroma_db_lock() -> Iterator[None]:
    """Serialize Chroma operations across the API and indexing containers."""
    lock_path = Path(CHROMA_DB_PATH) / ".access.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class LockedCollection:
    """Proxy that locks each Chroma collection operation."""

    def __init__(self, collection):
        self._collection = collection

    def __getattr__(self, name):
        attribute = getattr(self._collection, name)
        if not callable(attribute):
            return attribute

        @wraps(attribute)
        def call(*args, **kwargs):
            with chroma_db_lock():
                return attribute(*args, **kwargs)

        return call
