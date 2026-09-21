"""Cross-process coordination for the host-mounted Chroma database."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Iterator

import fcntl

from config import CHROMA_DB_PATH


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
