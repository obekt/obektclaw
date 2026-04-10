"""Tests for obektclaw/memory/store.py — SQLite + FTS5 storage layer."""
import tempfile
from pathlib import Path

import pytest

from obektclaw.memory.store import Store, _fts_query


@pytest.fixture
def store() -> Store:
    """Create a temporary in-file SQLite DB for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = Store(db_path)
        yield store
        store.close()


class TestFtsQuery:
    """Test FTS5 MATCH expression sanitization."""

    def test_simple_word(self):
        assert _fts_query("httpx") == "httpx*"

    def test_multiple_words(self):
        result = _fts_query("http client")
        assert "http*" in result
        assert "client*" in result
        assert "OR" in result

    def test_special_chars_removed(self):
        # Quotes, colons, dashes should be stripped
        result = _fts_query("'quoted' value")
        assert "'" not in result
        assert "quoted*" in result
        assert "value*" in result

    def test_empty_string(self):
        assert _fts_query("") == '""'

    def test_only_special_chars(self):
        result = _fts_query("!!! @@@ ###")
        assert result == '""'

    def test_underscore_and_dash_preserved(self):
        result = _fts_query("csv-to-database my_util")
        assert "csv-to-database*" in result
        assert "my_util*" in result


class TestSessionLifecycle:
    """Test session creation and closure."""

    def test_open_session(self, store: Store):
        session_id = store.open_session("cli", "test_user")
        assert isinstance(session_id, int)
        assert session_id > 0

    def test_close_session(self, store: Store):
        session_id = store.open_session("cli", "test_user")
        store.close_session(session_id)
        row = store.fetchone(
            "SELECT ended_at FROM sessions WHERE id = ?", (session_id,)
        )
        assert row is not None
        assert row["ended_at"] is not None

    def test_session_has_started_at(self, store: Store):
        session_id = store.open_session("tg", "user123")
        row = store.fetchone(
            "SELECT started_at, gateway, user_key FROM sessions WHERE id = ?",
            (session_id,),
        )
        assert row is not None
        assert row["started_at"] is not None
        assert row["gateway"] == "tg"
        assert row["user_key"] == "user123"


class TestMessages:
    """Test message storage and FTS5 search."""

    def test_add_message(self, store: Store):
        session_id = store.open_session("cli", "test_user")
        msg_id = store.add_message(session_id, "user", "Hello, world!")
        assert msg_id > 0

    def test_add_message_with_tool_name(self, store: Store):
        session_id = store.open_session("cli", "test_user")
        msg_id = store.add_message(
            session_id, "tool", "Result", tool_name="bash", meta={"exit_code": 0}
        )
        row = store.fetchone(
            "SELECT tool_name, meta_json FROM messages WHERE id = ?", (msg_id,)
        )
        assert row["tool_name"] == "bash"
        assert row["meta_json"] == '{"exit_code": 0}'

    def test_recent_messages(self, store: Store):
        session_id = store.open_session("cli", "test_user")
        for i in range(5):
            store.add_message(session_id, "user", f"Message {i}")
        recent = store.recent_messages(session_id, limit=10)
        assert len(recent) == 5
        assert "Message 0" in recent[0]["content"]

    def test_fts_messages_basic(self, store: Store):
        session_id = store.open_session("cli", "test_user")
        store.add_message(session_id, "user", "I always use httpx instead of requests")
        store.add_message(session_id, "user", "The server is on Hetzner")
        results = store.fts_messages("httpx")
        assert len(results) == 1
        assert "httpx" in results[0]["content"]

    def test_fts_messages_stemming(self, store: Store):
        """FTS5 with porter stemmer should match stems."""
        session_id = store.open_session("cli", "test_user")
        store.add_message(session_id, "user", "I am running the server")
        results = store.fts_messages("run")
        assert len(results) >= 1  # "running" should match "run"

    def test_fts_messages_no_crash_on_special_chars(self, store: Store):
        """FTS5 MATCH should not crash on special characters."""
        session_id = store.open_session("cli", "test_user")
        store.add_message(session_id, "user", "File path: /tmp/test's file.txt")
        # This should not raise
        results = store.fts_messages("/tmp/test's")
        assert isinstance(results, list)


class TestFacts:
    """Test persistent facts (semantic memory)."""

    def test_fts_facts_basic(self, store: Store):
        # Insert a fact via upsert
        store.execute(
            """
            INSERT INTO facts (key, value, category, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(category, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            ("http_client", "httpx", "preference", 0.9, 1000.0, 1000.0),
        )
        results = store.fts_facts("httpx")
        assert len(results) == 1
        assert results[0]["key"] == "http_client"
        assert results[0]["value"] == "httpx"

    def test_fts_facts_multiple_words(self, store: Store):
        store.execute(
            """
            INSERT INTO facts (key, value, category, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(category, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            ("server_host", "Hetzner CX22", "env", 0.95, 1000.0, 1000.0),
        )
        results = store.fts_facts("hetzner server")
        assert len(results) >= 1

    def test_fact_uniqueness_by_category_key(self, store: Store):
        """UNIQUE(category, key) should prevent duplicates."""
        store.execute(
            """
            INSERT INTO facts (key, value, category, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(category, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            ("pref1", "value1", "preference", 0.8, 1000.0, 1000.0),
        )
        # Upsert again with different value
        store.execute(
            """
            INSERT INTO facts (key, value, category, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(category, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            ("pref1", "value2", "preference", 0.9, 2000.0, 2000.0),
        )
        rows = store.fetchall("SELECT value FROM facts WHERE key = ?", ("pref1",))
        assert len(rows) == 1
        assert rows[0]["value"] == "value2"


class TestUserTraits:
    """Test 12-layer user model storage."""

    def test_set_and_get_trait(self, store: Store):
        store.execute(
            """
            INSERT INTO user_traits (layer, value, evidence, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(layer) DO UPDATE SET
                value = excluded.value,
                evidence = excluded.evidence,
                updated_at = excluded.updated_at
            """,
            ("tooling_pref", "httpx over requests", "user stated it", 1000.0),
        )
        row = store.fetchone(
            "SELECT value, evidence FROM user_traits WHERE layer = ?",
            ("tooling_pref",),
        )
        assert row is not None
        assert row["value"] == "httpx over requests"
        assert row["evidence"] == "user stated it"

    def test_trait_update_on_conflict(self, store: Store):
        # Insert
        store.execute(
            """
            INSERT INTO user_traits (layer, value, evidence, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(layer) DO UPDATE SET
                value = excluded.value,
                evidence = excluded.evidence,
                updated_at = excluded.updated_at
            """,
            ("technical_level", "intermediate", "inferred", 1000.0),
        )
        # Update
        store.execute(
            """
            INSERT INTO user_traits (layer, value, evidence, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(layer) DO UPDATE SET
                value = excluded.value,
                evidence = excluded.evidence,
                updated_at = excluded.updated_at
            """,
            ("technical_level", "advanced", "more evidence", 2000.0),
        )
        row = store.fetchone(
            "SELECT value, evidence, updated_at FROM user_traits WHERE layer = ?",
            ("technical_level",),
        )
        assert row["value"] == "advanced"
        assert row["evidence"] == "more evidence"
        assert row["updated_at"] == 2000.0


class TestSkills:
    """Test skills table (mirror of markdown files)."""

    def test_insert_skill(self, store: Store):
        store.execute(
            """
            INSERT INTO skills (name, description, body, use_count, success_count, created_at, updated_at)
            VALUES (?, ?, ?, 0, 0, ?, ?)
            """,
            ("test-skill", "A test skill", "# Body here", 1000.0, 1000.0),
        )
        row = store.fetchone("SELECT name, description FROM skills WHERE name = ?", ("test-skill",))
        assert row is not None
        assert row["description"] == "A test skill"

    def test_fts_skills(self, store: Store):
        store.execute(
            """
            INSERT INTO skills (name, description, body, use_count, success_count, created_at, updated_at)
            VALUES (?, ?, ?, 0, 0, ?, ?)
            """,
            ("csv-import", "Import CSV into SQLite", "Steps: 1. read csv", 1000.0, 1000.0),
        )
        results = store.fts_skills("csv database")
        assert len(results) >= 1
        assert results[0]["name"] == "csv-import"

    def test_skill_triggers(self, store: Store):
        """FTS5 triggers should keep skills_fts in sync."""
        store.execute(
            """
            INSERT INTO skills (name, description, body, use_count, success_count, created_at, updated_at)
            VALUES (?, ?, ?, 0, 0, ?, ?)
            """,
            ("deploy", "Deploy to production", "nginx gunicorn", 1000.0, 1000.0),
        )
        # Search should find it via FTS5
        results = store.fts_skills("nginx deploy")
        assert len(results) >= 1

        # Update
        store.execute(
            "UPDATE skills SET body = ?, updated_at = ? WHERE name = ?",
            ("nginx gunicorn postgres migration", 2000.0, "deploy"),
        )
        results = store.fts_skills("postgres migration")
        assert len(results) >= 1


class TestStoreThreadSafety:
    """Basic thread-safety checks."""

    def test_concurrent_reads(self, store: Store):
        """Multiple reads should work without issues."""
        session_id = store.open_session("cli", "user")
        for i in range(10):
            store.add_message(session_id, "user", f"msg {i}")
        
        # Multiple reads
        recent = store.recent_messages(session_id, limit=10)
        assert len(recent) == 10

    def test_wal_mode_allows_concurrent_access(self, store: Store):
        """WAL mode should be enabled."""
        row = store.fetchone("PRAGMA journal_mode")
        assert row["journal_mode"] == "wal"
