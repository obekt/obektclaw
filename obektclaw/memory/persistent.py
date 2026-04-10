"""Layer 2 — Persistent semantic memory: durable facts about user/project/env.

These are key/value pairs grouped by category. They survive across sessions
and get retrieved on demand via FTS5 (for free-text recall) or directly by
category (when bootstrapping a new session prompt).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .store import Store


@dataclass
class Fact:
    id: int
    category: str
    key: str
    value: str
    confidence: float

    def render(self) -> str:
        return f"- ({self.category}) {self.key}: {self.value}"


CATEGORIES = ("user", "project", "env", "preference", "general")


class PersistentMemory:
    def __init__(self, store: Store):
        self.store = store

    def upsert(self, key: str, value: str, *, category: str = "general", confidence: float = 0.8) -> None:
        if category not in CATEGORIES:
            category = "general"
        now = time.time()
        existing = self.store.fetchone(
            "SELECT id FROM facts WHERE category = ? AND key = ?", (category, key)
        )
        if existing is None:
            self.store.execute(
                """
                INSERT INTO facts (key, value, category, confidence, created_at, updated_at)
                VALUES (?,?,?,?,?,?)
                """,
                (key, value, category, confidence, now, now),
            )
        else:
            self.store.execute(
                """
                UPDATE facts SET value = ?, confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (value, confidence, now, existing["id"]),
            )

    def delete(self, category: str, key: str) -> None:
        self.store.execute(
            "DELETE FROM facts WHERE category = ? AND key = ?", (category, key)
        )

    def list_category(self, category: str, limit: int = 50) -> list[Fact]:
        rows = self.store.fetchall(
            """
            SELECT id, category, key, value, confidence
            FROM facts WHERE category = ?
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
            """,
            (category, limit),
        )
        return [Fact(r["id"], r["category"], r["key"], r["value"], r["confidence"]) for r in rows]

    def search(self, query: str, limit: int = 12) -> list[Fact]:
        rows = self.store.fts_facts(query, limit=limit)
        return [Fact(r["id"], r["category"], r["key"], r["value"], r["confidence"]) for r in rows]

    def all_top(self, per_category: int = 8) -> list[Fact]:
        out: list[Fact] = []
        for cat in CATEGORIES:
            out.extend(self.list_category(cat, limit=per_category))
        return out
