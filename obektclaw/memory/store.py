"""SQLite + FTS5 backed storage. Single-process, file-local, no external deps."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at REAL,
    gateway TEXT,
    user_key TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    ts REAL NOT NULL,
    role TEXT NOT NULL,           -- user / assistant / tool / system
    content TEXT NOT NULL,
    tool_name TEXT,
    meta_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, ts);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    role UNINDEXED,
    tool_name UNINDEXED,
    content='messages',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, role, tool_name)
        VALUES (new.id, new.content, new.role, new.tool_name);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, role, tool_name)
        VALUES('delete', old.id, old.content, old.role, old.tool_name);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, role, tool_name)
        VALUES('delete', old.id, old.content, old.role, old.tool_name);
    INSERT INTO messages_fts(rowid, content, role, tool_name)
        VALUES (new.id, new.content, new.role, new.tool_name);
END;

-- Persistent semantic memory: durable facts about the user, project, environment.
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    confidence REAL NOT NULL DEFAULT 0.7,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(category, key)
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    key, value, category UNINDEXED,
    content='facts', content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, key, value, category)
        VALUES (new.id, new.key, new.value, new.category);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, key, value, category)
        VALUES('delete', old.id, old.key, old.value, old.category);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, key, value, category)
        VALUES('delete', old.id, old.key, old.value, old.category);
    INSERT INTO facts_fts(rowid, key, value, category)
        VALUES (new.id, new.key, new.value, new.category);
END;

-- 12-layer user model (Honcho-style).
CREATE TABLE IF NOT EXISTS user_traits (
    layer TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    evidence TEXT,
    updated_at REAL NOT NULL
);

-- Skill index. Skill bodies live as markdown files on disk; this table mirrors
-- the metadata so we can run FTS5 search across the corpus.
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    body TEXT NOT NULL,
    use_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
    name, description, body,
    content='skills', content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS skills_ai AFTER INSERT ON skills BEGIN
    INSERT INTO skills_fts(rowid, name, description, body)
        VALUES (new.rowid, new.name, new.description, new.body);
END;
CREATE TRIGGER IF NOT EXISTS skills_ad AFTER DELETE ON skills BEGIN
    INSERT INTO skills_fts(skills_fts, rowid, name, description, body)
        VALUES('delete', old.rowid, old.name, old.description, old.body);
END;
CREATE TRIGGER IF NOT EXISTS skills_au AFTER UPDATE ON skills BEGIN
    INSERT INTO skills_fts(skills_fts, rowid, name, description, body)
        VALUES('delete', old.rowid, old.name, old.description, old.body);
    INSERT INTO skills_fts(rowid, name, description, body)
        VALUES (new.rowid, new.name, new.description, new.body);
END;
"""


def _fts_query(q: str) -> str:
    """Sanitize a free-text query for the FTS5 MATCH operator."""
    cleaned = []
    for tok in q.replace('"', " ").split():
        tok = "".join(ch for ch in tok if ch.isalnum() or ch in "_-")
        if tok:
            cleaned.append(tok + "*")
    return " OR ".join(cleaned) if cleaned else '""'


class Store:
    """Thread-safe SQLite wrapper. Holds one connection guarded by a lock."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    # ----- low-level helpers -----
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, [tuple(r) for r in rows])

    def fetchall(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return cur.fetchall()

    def fetchone(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return cur.fetchone()

    # ----- session lifecycle -----
    def open_session(self, gateway: str, user_key: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (started_at, gateway, user_key) VALUES (?,?,?)",
                (time.time(), gateway, user_key),
            )
            return int(cur.lastrowid)

    def close_session(self, session_id: int) -> None:
        self.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (time.time(), session_id),
        )

    # ----- messages -----
    def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        tool_name: str | None = None,
        meta: dict | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO messages (session_id, ts, role, content, tool_name, meta_json)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    session_id,
                    time.time(),
                    role,
                    content,
                    tool_name,
                    json.dumps(meta) if meta else None,
                ),
            )
            return int(cur.lastrowid)

    def fts_messages(self, query: str, limit: int = 20) -> list[sqlite3.Row]:
        return self.fetchall(
            """
            SELECT m.id, m.session_id, m.ts, m.role, m.tool_name, m.content
            FROM messages_fts f
            JOIN messages m ON m.id = f.rowid
            WHERE messages_fts MATCH ?
            ORDER BY m.ts DESC
            LIMIT ?
            """,
            (_fts_query(query), limit),
        )

    def recent_messages(self, session_id: int, limit: int = 50) -> list[sqlite3.Row]:
        return self.fetchall(
            """
            SELECT id, ts, role, tool_name, content, meta_json
            FROM messages WHERE session_id = ?
            ORDER BY ts ASC LIMIT ?
            """,
            (session_id, limit),
        )

    def fts_facts(self, query: str, limit: int = 20) -> list[sqlite3.Row]:
        return self.fetchall(
            """
            SELECT f.id, f.key, f.value, f.category, f.confidence
            FROM facts_fts ff
            JOIN facts f ON f.id = ff.rowid
            WHERE facts_fts MATCH ?
            LIMIT ?
            """,
            (_fts_query(query), limit),
        )

    def fts_skills(self, query: str, limit: int = 10) -> list[sqlite3.Row]:
        return self.fetchall(
            """
            SELECT s.name, s.description, s.use_count, s.success_count
            FROM skills_fts sf
            JOIN skills s ON s.rowid = sf.rowid
            WHERE skills_fts MATCH ?
            LIMIT ?
            """,
            (_fts_query(query), limit),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
