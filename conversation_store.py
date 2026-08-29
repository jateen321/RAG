"""Durable SQLite storage for local conversation history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    _ensure_schema(connection)
    return connection


@contextmanager
def _database(database_path: str | Path):
    connection = _connect(database_path)
    try:
        yield connection
    finally:
        connection.close()


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS exchanges (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources_json TEXT NOT NULL DEFAULT '[]',
            total_seconds REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_exchanges_conversation_created
        ON exchanges(conversation_id, created_at)
        """
    )
    connection.execute("PRAGMA optimize")
    connection.commit()


def _title_from_question(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= 64 else f"{title[:61].rstrip()}…"


def conversation_exists(database_path: str | Path, conversation_id: str) -> bool:
    with _database(database_path) as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return row is not None


def record_exchange(
    database_path: str | Path,
    conversation_id: str | None,
    question: str,
    answer: str,
    sources: list[dict],
    total_seconds: float | None,
) -> str:
    timestamp = _now()
    resolved_id = conversation_id or str(uuid4())
    with _database(database_path) as connection:
        if conversation_id is None:
            connection.execute(
                """
                INSERT INTO conversations(id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (resolved_id, _title_from_question(question), timestamp, timestamp),
            )
        connection.execute(
            """
            INSERT INTO exchanges(
                id, conversation_id, question, answer, sources_json,
                total_seconds, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                resolved_id,
                question,
                answer,
                json.dumps(sources, ensure_ascii=False),
                total_seconds,
                timestamp,
            ),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (timestamp, resolved_id),
        )
        connection.commit()
    return resolved_id


def list_conversations(database_path: str | Path, limit: int = 50) -> list[dict]:
    with _database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(e.id) AS exchange_count
            FROM conversations AS c
            LEFT JOIN exchanges AS e ON e.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(database_path: str | Path, conversation_id: str) -> dict | None:
    with _database(database_path) as connection:
        conversation = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return None
        rows = connection.execute(
            """
            SELECT id, question, answer, sources_json, total_seconds, created_at
            FROM exchanges
            WHERE conversation_id = ?
            ORDER BY created_at, rowid
            """,
            (conversation_id,),
        ).fetchall()

    exchanges = []
    for row in rows:
        exchange = dict(row)
        exchange["sources"] = json.loads(exchange.pop("sources_json"))
        exchanges.append(exchange)
    return {**dict(conversation), "exchanges": exchanges}


def get_recent_history(
    database_path: str | Path, conversation_id: str, limit: int = 12,
) -> list[dict]:
    """Fetch recent turns for this conversation only, oldest first.

    Bound the database read as well as the eventual model prompt. rowid breaks
    ties because timestamps have only second-level precision.
    """
    if limit < 1:
        return []
    with _database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT question, answer FROM exchanges
            WHERE conversation_id = ?
            ORDER BY created_at DESC, rowid DESC LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
    return [
        {"role": role, "parts": [{"text": row[field]}]}
        for row in reversed(rows)
        for role, field in (("user", "question"), ("model", "answer"))
    ]


def delete_conversation(database_path: str | Path, conversation_id: str) -> bool:
    with _database(database_path) as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
        connection.commit()
    return cursor.rowcount > 0
