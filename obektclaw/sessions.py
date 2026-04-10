"""Session management: list, inspect, export, and resume conversations.

Sessions are stored in SQLite via the Store. Each session has a gateway,
user_key, timestamps, and a sequence of messages. This module provides
read-only operations on that data for browsing, exporting, and resuming.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .memory.store import Store


@dataclass
class SessionSummary:
    id: int
    started_at: float
    ended_at: float | None
    gateway: str
    user_key: str
    message_count: int
    preview: str

    @property
    def started_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.started_at))

    @property
    def ended_str(self) -> str | None:
        if self.ended_at is None:
            return None
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.ended_at))

    @property
    def duration_str(self) -> str:
        end = self.ended_at or time.time()
        secs = end - self.started_at
        if secs < 60:
            return f"{int(secs)}s"
        if secs < 3600:
            return f"{int(secs // 60)}m"
        return f"{secs / 3600:.1f}h"


@dataclass
class SessionMessage:
    id: int
    ts: float
    role: str
    content: str
    tool_name: str | None
    meta_json: str | None

    @property
    def ts_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts))


def list_sessions(store: Store, *, limit: int = 20, gateway: str | None = None) -> list[SessionSummary]:
    """List recent sessions with message counts and a preview of the first user message."""
    where = ""
    params: list = []
    if gateway:
        where = "WHERE s.gateway = ?"
        params.append(gateway)

    rows = store.fetchall(
        f"""
        SELECT
            s.id,
            s.started_at,
            s.ended_at,
            s.gateway,
            s.user_key,
            COUNT(m.id) AS message_count,
            (
                SELECT content FROM messages
                WHERE session_id = s.id AND role = 'user'
                ORDER BY ts ASC LIMIT 1
            ) AS first_user_msg
        FROM sessions s
        LEFT JOIN messages m ON m.session_id = s.id
        {where}
        GROUP BY s.id
        ORDER BY s.started_at DESC
        LIMIT ?
        """,
        (*params, limit),
    )
    out: list[SessionSummary] = []
    for r in rows:
        preview = r["first_user_msg"] or ""
        if len(preview) > 80:
            preview = preview[:77] + "..."
        out.append(SessionSummary(
            id=r["id"],
            started_at=r["started_at"],
            ended_at=r["ended_at"],
            gateway=r["gateway"] or "",
            user_key=r["user_key"] or "",
            message_count=r["message_count"],
            preview=preview,
        ))
    return out


def get_session_info(store: Store, session_id: int) -> SessionSummary | None:
    """Get detailed info about a single session."""
    row = store.fetchone(
        """
        SELECT
            s.id,
            s.started_at,
            s.ended_at,
            s.gateway,
            s.user_key,
            COUNT(m.id) AS message_count,
            (
                SELECT content FROM messages
                WHERE session_id = s.id AND role = 'user'
                ORDER BY ts ASC LIMIT 1
            ) AS first_user_msg
        FROM sessions s
        LEFT JOIN messages m ON m.session_id = s.id
        WHERE s.id = ?
        GROUP BY s.id
        """,
        (session_id,),
    )
    if row is None:
        return None
    preview = row["first_user_msg"] or ""
    if len(preview) > 80:
        preview = preview[:77] + "..."
    return SessionSummary(
        id=row["id"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        gateway=row["gateway"] or "",
        user_key=row["user_key"] or "",
        message_count=row["message_count"],
        preview=preview,
    )


def get_session_messages(store: Store, session_id: int) -> list[SessionMessage]:
    """Get all messages for a session, ordered by timestamp."""
    rows = store.fetchall(
        """
        SELECT id, ts, role, content, tool_name, meta_json
        FROM messages
        WHERE session_id = ?
        ORDER BY ts ASC
        """,
        (session_id,),
    )
    return [
        SessionMessage(
            id=r["id"],
            ts=r["ts"],
            role=r["role"],
            content=r["content"],
            tool_name=r["tool_name"],
            meta_json=r["meta_json"],
        )
        for r in rows
    ]


def export_session_markdown(store: Store, session_id: int) -> str | None:
    """Export a session as a readable markdown document.

    Returns None if the session doesn't exist.
    """
    info = get_session_info(store, session_id)
    if info is None:
        return None

    messages = get_session_messages(store, session_id)

    lines: list[str] = []
    lines.append(f"# Session {info.id}")
    lines.append("")
    lines.append(f"- **Started:** {info.started_str}")
    if info.ended_str:
        lines.append(f"- **Ended:** {info.ended_str}")
    lines.append(f"- **Duration:** {info.duration_str}")
    lines.append(f"- **Gateway:** {info.gateway}")
    lines.append(f"- **Messages:** {info.message_count}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for msg in messages:
        if msg.role == "system":
            lines.append(f"> *[system {msg.ts_str}]* {msg.content}")
            lines.append("")
        elif msg.role == "user":
            lines.append(f"**User** ({msg.ts_str}):")
            lines.append("")
            lines.append(msg.content)
            lines.append("")
        elif msg.role == "assistant":
            lines.append(f"**Assistant** ({msg.ts_str}):")
            lines.append("")
            lines.append(msg.content)
            lines.append("")
        elif msg.role == "tool":
            tool = msg.tool_name or "unknown"
            # Show tool output compactly
            content = msg.content
            if len(content) > 500:
                content = content[:480] + "\n... (truncated)"
            lines.append(f"<details><summary>Tool: {tool} ({msg.ts_str})</summary>")
            lines.append("")
            lines.append("```")
            lines.append(content)
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines)


def export_session_json(store: Store, session_id: int) -> dict | None:
    """Export a session as a JSON-serializable dict.

    Returns None if the session doesn't exist.
    """
    info = get_session_info(store, session_id)
    if info is None:
        return None

    messages = get_session_messages(store, session_id)

    return {
        "session_id": info.id,
        "started_at": info.started_at,
        "started_at_str": info.started_str,
        "ended_at": info.ended_at,
        "ended_at_str": info.ended_str,
        "duration": info.duration_str,
        "gateway": info.gateway,
        "user_key": info.user_key,
        "message_count": info.message_count,
        "messages": [
            {
                "id": m.id,
                "ts": m.ts,
                "ts_str": m.ts_str,
                "role": m.role,
                "content": m.content,
                "tool_name": m.tool_name,
                "meta": json.loads(m.meta_json) if m.meta_json else None,
            }
            for m in messages
        ],
    }
