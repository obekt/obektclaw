"""Tests for obektclaw/sessions.py — session management, export, and resume."""
import json
import tempfile
import time
from pathlib import Path

import pytest

from obektclaw.memory.store import Store
from obektclaw.sessions import (
    SessionSummary,
    SessionMessage,
    list_sessions,
    get_session_info,
    get_session_messages,
    export_session_markdown,
    export_session_json,
)


@pytest.fixture
def store() -> Store:
    """Create a temporary SQLite DB for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        s = Store(db_path)
        yield s
        s.close()


def _populate(store: Store) -> tuple[int, int]:
    """Create two sessions with messages. Returns (session1_id, session2_id)."""
    s1 = store.open_session("cli", "user-a")
    store.add_message(s1, "user", "Hello, how are you?")
    store.add_message(s1, "assistant", "I'm great, thanks!")
    store.add_message(s1, "user", "Can you list my files?")
    store.add_message(s1, "tool", "file1.py\nfile2.py", tool_name="bash")
    store.add_message(s1, "assistant", "Here are your files: file1.py and file2.py")
    store.close_session(s1)

    s2 = store.open_session("telegram", "tg:12345")
    store.add_message(s2, "user", "What's the weather like?")
    store.add_message(s2, "assistant", "I can't check weather, but I can help with code!")
    # leave s2 open (no close_session)

    return s1, s2


# ── list_sessions ──────────────────────────────────────────────────────────


class TestListSessions:
    def test_empty_store(self, store: Store):
        result = list_sessions(store)
        assert result == []

    def test_returns_sessions_newest_first(self, store: Store):
        s1, s2 = _populate(store)
        result = list_sessions(store)
        assert len(result) == 2
        assert result[0].id == s2  # newest first
        assert result[1].id == s1

    def test_includes_message_count(self, store: Store):
        s1, s2 = _populate(store)
        result = list_sessions(store)
        # s2 has 2 messages, s1 has 5
        by_id = {s.id: s for s in result}
        assert by_id[s1].message_count == 5
        assert by_id[s2].message_count == 2

    def test_preview_is_first_user_message(self, store: Store):
        s1, s2 = _populate(store)
        result = list_sessions(store)
        by_id = {s.id: s for s in result}
        assert by_id[s1].preview == "Hello, how are you?"
        assert by_id[s2].preview == "What's the weather like?"

    def test_preview_truncated_at_80_chars(self, store: Store):
        sid = store.open_session("cli", "user")
        store.add_message(sid, "user", "x" * 200)
        result = list_sessions(store)
        assert len(result[0].preview) <= 80
        assert result[0].preview.endswith("...")

    def test_limit(self, store: Store):
        for i in range(10):
            sid = store.open_session("cli", f"user-{i}")
            store.add_message(sid, "user", f"Session {i}")
        result = list_sessions(store, limit=3)
        assert len(result) == 3

    def test_gateway_filter(self, store: Store):
        _populate(store)
        cli_only = list_sessions(store, gateway="cli")
        assert len(cli_only) == 1
        assert cli_only[0].gateway == "cli"

        tg_only = list_sessions(store, gateway="telegram")
        assert len(tg_only) == 1
        assert tg_only[0].gateway == "telegram"

    def test_gateway_and_user_key(self, store: Store):
        s1, s2 = _populate(store)
        result = list_sessions(store)
        by_id = {s.id: s for s in result}
        assert by_id[s1].gateway == "cli"
        assert by_id[s1].user_key == "user-a"
        assert by_id[s2].gateway == "telegram"
        assert by_id[s2].user_key == "tg:12345"


# ── get_session_info ───────────────────────────────────────────────────────


class TestGetSessionInfo:
    def test_existing_session(self, store: Store):
        s1, _ = _populate(store)
        info = get_session_info(store, s1)
        assert info is not None
        assert info.id == s1
        assert info.message_count == 5
        assert info.gateway == "cli"

    def test_nonexistent_session(self, store: Store):
        assert get_session_info(store, 9999) is None

    def test_ended_at_set_for_closed(self, store: Store):
        s1, s2 = _populate(store)
        info1 = get_session_info(store, s1)
        info2 = get_session_info(store, s2)
        assert info1.ended_at is not None
        assert info2.ended_at is None


# ── get_session_messages ───────────────────────────────────────────────────


class TestGetSessionMessages:
    def test_returns_messages_in_order(self, store: Store):
        s1, _ = _populate(store)
        messages = get_session_messages(store, s1)
        assert len(messages) == 5
        assert messages[0].role == "user"
        assert messages[0].content == "Hello, how are you?"
        assert messages[1].role == "assistant"

    def test_tool_messages_have_tool_name(self, store: Store):
        s1, _ = _populate(store)
        messages = get_session_messages(store, s1)
        tool_msgs = [m for m in messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_name == "bash"

    def test_empty_session(self, store: Store):
        sid = store.open_session("cli", "user")
        messages = get_session_messages(store, sid)
        assert messages == []

    def test_nonexistent_session_returns_empty(self, store: Store):
        messages = get_session_messages(store, 9999)
        assert messages == []


# ── SessionSummary properties ──────────────────────────────────────────────


class TestSessionSummaryProperties:
    def test_started_str(self):
        s = SessionSummary(
            id=1, started_at=1700000000.0, ended_at=None,
            gateway="cli", user_key="user", message_count=0, preview="",
        )
        assert s.started_str  # should not crash
        assert "-" in s.started_str  # date format

    def test_ended_str_none(self):
        s = SessionSummary(
            id=1, started_at=1700000000.0, ended_at=None,
            gateway="cli", user_key="user", message_count=0, preview="",
        )
        assert s.ended_str is None

    def test_ended_str_set(self):
        s = SessionSummary(
            id=1, started_at=1700000000.0, ended_at=1700003600.0,
            gateway="cli", user_key="user", message_count=0, preview="",
        )
        assert s.ended_str is not None

    def test_duration_str_seconds(self):
        now = time.time()
        s = SessionSummary(
            id=1, started_at=now - 30, ended_at=now,
            gateway="cli", user_key="user", message_count=0, preview="",
        )
        assert s.duration_str == "30s"

    def test_duration_str_minutes(self):
        now = time.time()
        s = SessionSummary(
            id=1, started_at=now - 300, ended_at=now,
            gateway="cli", user_key="user", message_count=0, preview="",
        )
        assert s.duration_str == "5m"

    def test_duration_str_hours(self):
        now = time.time()
        s = SessionSummary(
            id=1, started_at=now - 7200, ended_at=now,
            gateway="cli", user_key="user", message_count=0, preview="",
        )
        assert s.duration_str == "2.0h"


# ── SessionMessage properties ─────────────────────────────────────────────


class TestSessionMessageProperties:
    def test_ts_str(self):
        m = SessionMessage(
            id=1, ts=1700000000.0, role="user", content="hi",
            tool_name=None, meta_json=None,
        )
        assert m.ts_str  # should not crash
        assert ":" in m.ts_str  # time format


# ── export_session_markdown ────────────────────────────────────────────────


class TestExportMarkdown:
    def test_nonexistent_session(self, store: Store):
        assert export_session_markdown(store, 9999) is None

    def test_basic_structure(self, store: Store):
        s1, _ = _populate(store)
        md = export_session_markdown(store, s1)
        assert md is not None
        assert md.startswith("# Session")
        assert "Started:" in md
        assert "Duration:" in md
        assert "Gateway:" in md
        assert "Messages:" in md

    def test_contains_user_messages(self, store: Store):
        s1, _ = _populate(store)
        md = export_session_markdown(store, s1)
        assert "Hello, how are you?" in md
        assert "**User**" in md

    def test_contains_assistant_messages(self, store: Store):
        s1, _ = _populate(store)
        md = export_session_markdown(store, s1)
        assert "I'm great, thanks!" in md
        assert "**Assistant**" in md

    def test_tool_messages_in_details_tag(self, store: Store):
        s1, _ = _populate(store)
        md = export_session_markdown(store, s1)
        assert "<details>" in md
        assert "Tool: bash" in md
        assert "file1.py" in md

    def test_tool_output_truncated(self, store: Store):
        sid = store.open_session("cli", "user")
        store.add_message(sid, "user", "run something")
        store.add_message(sid, "tool", "x" * 1000, tool_name="bash")
        md = export_session_markdown(store, sid)
        assert "truncated" in md


# ── export_session_json ────────────────────────────────────────────────────


class TestExportJSON:
    def test_nonexistent_session(self, store: Store):
        assert export_session_json(store, 9999) is None

    def test_basic_structure(self, store: Store):
        s1, _ = _populate(store)
        data = export_session_json(store, s1)
        assert data is not None
        assert data["session_id"] == s1
        assert data["gateway"] == "cli"
        assert data["message_count"] == 5
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) == 5

    def test_message_fields(self, store: Store):
        s1, _ = _populate(store)
        data = export_session_json(store, s1)
        msg = data["messages"][0]
        assert "id" in msg
        assert "ts" in msg
        assert "ts_str" in msg
        assert "role" in msg
        assert "content" in msg
        assert msg["role"] == "user"
        assert msg["content"] == "Hello, how are you?"

    def test_tool_message_has_tool_name(self, store: Store):
        s1, _ = _populate(store)
        data = export_session_json(store, s1)
        tool_msgs = [m for m in data["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_name"] == "bash"

    def test_json_serializable(self, store: Store):
        s1, _ = _populate(store)
        data = export_session_json(store, s1)
        # Should not raise
        serialized = json.dumps(data)
        assert isinstance(serialized, str)
        # Round-trip
        parsed = json.loads(serialized)
        assert parsed["session_id"] == s1

    def test_meta_parsed(self, store: Store):
        sid = store.open_session("cli", "user")
        store.add_message(sid, "user", "run it")
        store.add_message(
            sid, "tool", "output",
            tool_name="bash", meta={"args": "ls -la", "is_error": False},
        )
        data = export_session_json(store, sid)
        tool_msg = data["messages"][1]
        assert tool_msg["meta"] == {"args": "ls -la", "is_error": False}

    def test_timestamps_present(self, store: Store):
        s1, _ = _populate(store)
        data = export_session_json(store, s1)
        assert "started_at" in data
        assert "started_at_str" in data
        assert isinstance(data["started_at"], float)
        assert isinstance(data["started_at_str"], str)


# ── Agent resume ───────────────────────────────────────────────────────────


class TestAgentResume:
    """Test that Agent can resume an existing session."""

    def test_agent_uses_existing_session_id(self, store: Store):
        from obektclaw.agent import Agent
        from obektclaw.config import Config
        from obektclaw.skills import SkillManager

        # Create a session and add messages
        sid = store.open_session("cli", "user")
        store.add_message(sid, "user", "Hello from the past")
        store.add_message(sid, "assistant", "Hi there!")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config = Config(
                home=tmppath,
                db_path=tmppath / "test.db",
                skills_dir=tmppath / "skills",
                bundled_skills_dir=tmppath / "bundled",
                logs_dir=tmppath / "logs",
                llm_base_url="http://localhost:1234/v1",
                llm_api_key="test-key",
                llm_model="test-model",
                llm_fast_model="test-model",
                tg_token="",
                tg_allowed_chat_ids=(),
                bash_timeout=10,
                workdir=tmppath,
            )
            (tmppath / "skills").mkdir()
            (tmppath / "bundled").mkdir()
            skills = SkillManager(store, tmppath / "skills", tmppath / "bundled")

            agent = Agent(
                config=config, store=store, skills=skills,
                gateway="cli", user_key="cli-local",
                session_id=sid,
                run_learning_loop=False,
                load_mcp=False,
            )

            # Agent should be using the existing session
            assert agent.session_id == sid
            assert agent._resumed is True

            # Should see existing messages in session history
            recent = agent.session.recent(limit=10)
            assert len(recent) == 2
            assert recent[0].content == "Hello from the past"

            agent.close()

    def test_agent_new_session_not_resumed(self, store: Store):
        from obektclaw.agent import Agent
        from obektclaw.config import Config
        from obektclaw.skills import SkillManager

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            config = Config(
                home=tmppath,
                db_path=tmppath / "test.db",
                skills_dir=tmppath / "skills",
                bundled_skills_dir=tmppath / "bundled",
                logs_dir=tmppath / "logs",
                llm_base_url="http://localhost:1234/v1",
                llm_api_key="test-key",
                llm_model="test-model",
                llm_fast_model="test-model",
                tg_token="",
                tg_allowed_chat_ids=(),
                bash_timeout=10,
                workdir=tmppath,
            )
            (tmppath / "skills").mkdir()
            (tmppath / "bundled").mkdir()
            skills = SkillManager(store, tmppath / "skills", tmppath / "bundled")

            agent = Agent(
                config=config, store=store, skills=skills,
                gateway="cli", user_key="cli-local",
                run_learning_loop=False,
                load_mcp=False,
            )

            assert agent._resumed is False
            agent.close()
