"""Durable SQLite storage for local conversation history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from citation_sources import align_answer_sources
from config import LEGACY_ADMIN_UID


LOCAL_OWNER_ID = "local-user"


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
            owner_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conversation_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(conversations)")
    }
    if "owner_id" not in conversation_columns:
        connection.execute("ALTER TABLE conversations ADD COLUMN owner_id TEXT")
    if LEGACY_ADMIN_UID:
        connection.execute(
            "UPDATE conversations SET owner_id = ? WHERE owner_id IS NULL OR owner_id = ''",
            (LEGACY_ADMIN_UID,),
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
            answer_basis TEXT NOT NULL DEFAULT 'documents',
            web_search_available INTEGER NOT NULL DEFAULT 0,
            image_generation_available INTEGER NOT NULL DEFAULT 0,
            image_prompt TEXT NOT NULL DEFAULT '',
            generated_image_id TEXT,
            generated_image_mime_type TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                ON DELETE CASCADE
        )
        """
    )
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(exchanges)")
    }
    if "answer_basis" not in columns:
        connection.execute(
            "ALTER TABLE exchanges ADD COLUMN answer_basis TEXT NOT NULL DEFAULT 'documents'"
        )
    if "web_search_available" not in columns:
        connection.execute(
            "ALTER TABLE exchanges ADD COLUMN web_search_available INTEGER NOT NULL DEFAULT 0"
        )
    if "image_generation_available" not in columns:
        connection.execute(
            "ALTER TABLE exchanges ADD COLUMN image_generation_available INTEGER NOT NULL DEFAULT 0"
        )
    if "image_prompt" not in columns:
        connection.execute(
            "ALTER TABLE exchanges ADD COLUMN image_prompt TEXT NOT NULL DEFAULT ''"
        )
    if "generated_image_id" not in columns:
        connection.execute("ALTER TABLE exchanges ADD COLUMN generated_image_id TEXT")
    if "generated_image_mime_type" not in columns:
        connection.execute(
            "ALTER TABLE exchanges ADD COLUMN generated_image_mime_type TEXT"
        )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_exchanges_conversation_created
        ON exchanges(conversation_id, created_at)
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_owner_updated "
        "ON conversations(owner_id, updated_at)"
    )
    connection.execute("PRAGMA optimize")
    connection.commit()


def _title_from_question(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= 64 else f"{title[:61].rstrip()}…"


def conversation_exists(
    database_path: str | Path,
    conversation_id: str,
    owner_id: str | None = None,
) -> bool:
    with _database(database_path) as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND (? IS NULL OR owner_id = ?)",
            (conversation_id, owner_id, owner_id),
        ).fetchone()
    return row is not None


def record_exchange(
    database_path: str | Path,
    conversation_id: str | None,
    question: str,
    answer: str,
    sources: list[dict],
    total_seconds: float | None,
    exchange_id: str | None = None,
    answer_basis: str = "documents",
    web_search_available: bool = False,
    image_generation_available: bool = False,
    image_prompt: str = "",
    owner_id: str | None = None,
) -> str:
    timestamp = _now()
    resolved_id = conversation_id or str(uuid4())
    resolved_owner = owner_id or LEGACY_ADMIN_UID or LOCAL_OWNER_ID
    with _database(database_path) as connection:
        if conversation_id is None:
            connection.execute(
                """
                INSERT INTO conversations(id, owner_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    resolved_id, resolved_owner, _title_from_question(question),
                    timestamp, timestamp,
                ),
            )
        elif owner_id is not None and not connection.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND owner_id = ?",
            (conversation_id, owner_id),
        ).fetchone():
            raise ValueError("Conversation not found.")
        connection.execute(
            """
            INSERT INTO exchanges(
                id, conversation_id, question, answer, sources_json,
                total_seconds, created_at, answer_basis, web_search_available,
                image_generation_available, image_prompt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exchange_id or str(uuid4()),
                resolved_id,
                question,
                answer,
                json.dumps(sources, ensure_ascii=False),
                total_seconds,
                timestamp,
                answer_basis,
                int(web_search_available),
                int(image_generation_available),
                image_prompt,
            ),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (timestamp, resolved_id),
        )
        connection.commit()
    return resolved_id


def get_history_before_exchange(
    database_path: str | Path,
    conversation_id: str,
    exchange_id: str,
    limit: int = 12,
    owner_id: str | None = None,
) -> list[dict] | None:
    """Return bounded history before one exchange, or ``None`` if it is absent."""
    with _database(database_path) as connection:
        target = connection.execute(
            """
            SELECT e.rowid FROM exchanges e
            JOIN conversations c ON c.id = e.conversation_id
            WHERE e.id = ? AND e.conversation_id = ?
              AND (? IS NULL OR c.owner_id = ?)
            """,
            (exchange_id, conversation_id, owner_id, owner_id),
        ).fetchone()
        if target is None:
            return None
        rows = connection.execute(
            """
            SELECT question, answer, sources_json FROM exchanges
            WHERE conversation_id = ? AND rowid < ?
            ORDER BY rowid DESC LIMIT ?
            """,
            (conversation_id, target["rowid"], max(limit, 0)),
        ).fetchall()
    history = []
    for row in reversed(rows):
        history.append({"role": "user", "parts": [{"text": row["question"]}]})
        model = {"role": "model", "parts": [{"text": row["answer"]}]}
        sources = json.loads(row["sources_json"])
        if sources:
            model["sources"] = sources
        history.append(model)
    return history


def replace_exchange_and_truncate(
    database_path: str | Path,
    conversation_id: str,
    exchange_id: str,
    question: str,
    answer: str,
    sources: list[dict],
    total_seconds: float | None,
    answer_basis: str = "documents",
    web_search_available: bool = False,
    image_generation_available: bool = False,
    image_prompt: str = "",
    owner_id: str | None = None,
) -> bool:
    """Replace an exchange and delete later turns from the abandoned branch."""
    timestamp = _now()
    with _database(database_path) as connection:
        target = connection.execute(
            """
            SELECT e.rowid FROM exchanges e
            JOIN conversations c ON c.id = e.conversation_id
            WHERE e.id = ? AND e.conversation_id = ?
              AND (? IS NULL OR c.owner_id = ?)
            """,
            (exchange_id, conversation_id, owner_id, owner_id),
        ).fetchone()
        if target is None:
            return False
        earlier = connection.execute(
            """
            SELECT 1 FROM exchanges
            WHERE conversation_id = ? AND rowid < ? LIMIT 1
            """,
            (conversation_id, target["rowid"]),
        ).fetchone()
        connection.execute(
            "DELETE FROM exchanges WHERE conversation_id = ? AND rowid >= ?",
            (conversation_id, target["rowid"]),
        )
        connection.execute(
            """
            INSERT INTO exchanges(
                id, conversation_id, question, answer, sources_json,
                total_seconds, created_at, answer_basis, web_search_available,
                image_generation_available, image_prompt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exchange_id,
                conversation_id,
                question,
                answer,
                json.dumps(sources, ensure_ascii=False),
                total_seconds,
                timestamp,
                answer_basis,
                int(web_search_available),
                int(image_generation_available),
                image_prompt,
            ),
        )
        if earlier is None:
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (_title_from_question(question), timestamp, conversation_id),
            )
        else:
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, conversation_id),
            )
        connection.commit()
    return True


def replace_latest_exchange_with_web_answer(
    database_path: str | Path,
    conversation_id: str,
    exchange_id: str,
    answer: str,
    sources: list[dict],
    total_seconds: float | None,
    owner_id: str | None = None,
) -> bool:
    """Replace only the latest exchange when it still offers web fallback."""
    timestamp = _now()
    with _database(database_path) as connection:
        latest = connection.execute(
            """
            SELECT e.id, e.web_search_available FROM exchanges e
            JOIN conversations c ON c.id = e.conversation_id
            WHERE e.conversation_id = ? AND (? IS NULL OR c.owner_id = ?)
            ORDER BY e.rowid DESC LIMIT 1
            """,
            (conversation_id, owner_id, owner_id),
        ).fetchone()
        if (
            latest is None
            or latest["id"] != exchange_id
            or not latest["web_search_available"]
        ):
            return False
        connection.execute(
            """
            UPDATE exchanges
            SET answer = ?, sources_json = ?, total_seconds = ?,
                answer_basis = 'web', web_search_available = 0,
                image_generation_available = 0, image_prompt = '',
                generated_image_id = NULL, generated_image_mime_type = NULL
            WHERE id = ? AND conversation_id = ?
            """,
            (
                answer,
                json.dumps(sources, ensure_ascii=False),
                total_seconds,
                exchange_id,
                conversation_id,
            ),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (timestamp, conversation_id),
        )
        connection.commit()
    return True


def attach_generated_image(
    database_path: str | Path,
    conversation_id: str,
    exchange_id: str,
    image_id: str,
    mime_type: str,
    owner_id: str | None = None,
) -> bool:
    """Attach one image only while the exchange remains eligible and empty."""
    timestamp = _now()
    with _database(database_path) as connection:
        updated = connection.execute(
            """
            UPDATE exchanges
            SET generated_image_id = ?, generated_image_mime_type = ?,
                image_generation_available = 0
            WHERE id = ? AND conversation_id = ?
              AND image_generation_available = 1
              AND generated_image_id IS NULL
              AND (? IS NULL OR EXISTS (
                  SELECT 1 FROM conversations c
                  WHERE c.id = exchanges.conversation_id AND c.owner_id = ?
              ))
            """,
            (image_id, mime_type, exchange_id, conversation_id, owner_id, owner_id),
        )
        if not updated.rowcount:
            return False
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (timestamp, conversation_id),
        )
        connection.commit()
    return True


def get_generated_image_metadata(
    database_path: str | Path, image_id: str, owner_id: str | None = None,
) -> dict | None:
    with _database(database_path) as connection:
        row = connection.execute(
            """
            SELECT e.generated_image_id, e.generated_image_mime_type
            FROM exchanges e
            JOIN conversations c ON c.id = e.conversation_id
            WHERE e.generated_image_id = ? AND (? IS NULL OR c.owner_id = ?)
            LIMIT 1
            """,
            (image_id, owner_id, owner_id),
        ).fetchone()
    return dict(row) if row else None


def list_conversations(
    database_path: str | Path,
    limit: int = 50,
    owner_id: str | None = None,
) -> list[dict]:
    with _database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(e.id) AS exchange_count
            FROM conversations AS c
            LEFT JOIN exchanges AS e ON e.conversation_id = c.id
            WHERE (? IS NULL OR c.owner_id = ?)
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (owner_id, owner_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(
    database_path: str | Path,
    conversation_id: str,
    owner_id: str | None = None,
) -> dict | None:
    with _database(database_path) as connection:
        conversation = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations WHERE id = ? AND (? IS NULL OR owner_id = ?)
            """,
            (conversation_id, owner_id, owner_id),
        ).fetchone()
        if conversation is None:
            return None
        rows = connection.execute(
            """
            SELECT id, question, answer, sources_json, total_seconds, created_at,
                   answer_basis, web_search_available,
                   image_generation_available, image_prompt,
                   generated_image_id, generated_image_mime_type
            FROM exchanges
            WHERE conversation_id = ?
            ORDER BY created_at, rowid
            """,
            (conversation_id,),
        ).fetchall()

    exchanges = []
    historical_sources = []
    for row in rows:
        exchange = dict(row)
        stored_sources = json.loads(exchange.pop("sources_json"))
        exchange["sources"] = align_answer_sources(
            exchange["answer"], stored_sources, historical_sources,
        )
        exchange["web_search_available"] = bool(exchange["web_search_available"])
        exchange["image_generation_available"] = bool(
            exchange["image_generation_available"]
        )
        exchanges.append(exchange)
        historical_sources.extend(exchange["sources"])
    return {**dict(conversation), "exchanges": exchanges}


def get_recent_history(
    database_path: str | Path, conversation_id: str, limit: int = 12,
    owner_id: str | None = None,
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
            SELECT e.question, e.answer, e.sources_json FROM exchanges e
            JOIN conversations c ON c.id = e.conversation_id
            WHERE e.conversation_id = ? AND (? IS NULL OR c.owner_id = ?)
            ORDER BY e.created_at DESC, e.rowid DESC LIMIT ?
            """,
            (conversation_id, owner_id, owner_id, limit),
        ).fetchall()
    history = []
    for row in reversed(rows):
        history.append({"role": "user", "parts": [{"text": row["question"]}]})
        model = {"role": "model", "parts": [{"text": row["answer"]}]}
        sources = json.loads(row["sources_json"])
        if sources:
            model["sources"] = sources
        history.append(model)
    return history


def delete_conversation(
    database_path: str | Path,
    conversation_id: str,
    owner_id: str | None = None,
) -> bool:
    with _database(database_path) as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ? AND (? IS NULL OR owner_id = ?)",
            (conversation_id, owner_id, owner_id),
        )
        connection.commit()
    return cursor.rowcount > 0
