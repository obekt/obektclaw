"""Layer 1 — Session memory: episodic record of what happened.

Stores every message, tool call, and tool result. Loads on demand via FTS5
rather than dumping the full transcript into the prompt.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .store import Store


@dataclass
class MessageRecord:
    id: int
    ts: float
    role: str
    tool_name: str | None
    content: str

    def render(self) -> str:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.ts))
        tag = self.tool_name or self.role
        snippet = self.content if len(self.content) < 400 else self.content[:380] + "..."
        return f"[{when}] {tag}: {snippet}"


class SessionMemory:
    def __init__(self, store: Store, session_id: int):
        self.store = store
        self.session_id = session_id

    def add(self, role: str, content: str, *, tool_name: str | None = None, meta: dict | None = None) -> int:
        return self.store.add_message(self.session_id, role, content, tool_name=tool_name, meta=meta)

    def recent(self, limit: int = 50) -> list[MessageRecord]:
        rows = self.store.recent_messages(self.session_id, limit=limit)
        return [
            MessageRecord(
                id=r["id"], ts=r["ts"], role=r["role"], tool_name=r["tool_name"], content=r["content"]
            )
            for r in rows
        ]

    def search_history(self, query: str, limit: int = 8) -> list[MessageRecord]:
        """FTS5 search across the *entire* message archive (all sessions)."""
        rows = self.store.fts_messages(query, limit=limit)
        return [
            MessageRecord(
                id=r["id"], ts=r["ts"], role=r["role"], tool_name=r["tool_name"], content=r["content"]
            )
            for r in rows
        ]
