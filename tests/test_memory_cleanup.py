"""Tests for the memory cleanup CLI command."""
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock

import pytest

from obektclaw.config import Config
from obektclaw.memory.store import Store
from obektclaw.memory import PersistentMemory
from obektclaw.__main__ import main


class FakeCleanupLLM:
    """Fake LLM that returns specific cleanup responses."""

    def __init__(self, keys_to_delete: Optional[list[str]] = None):
        self.keys_to_delete = keys_to_delete or []
        self.call_count = 0

    def chat(
        self,
        messages,
        *,
        tools=None,
        fast=False,
        temperature=0.4,
        max_tokens=2048,
    ):
        self.call_count += 1
        from obektclaw.llm import LLMResponse
        return LLMResponse(content="OK", tool_calls=[], raw=None)

    def chat_simple(self, system: str, user: str, *, fast=True, temperature=0.3) -> str:
        return "OK"

    def chat_json(
        self, system: str, user: str, *, fast=True
    ) -> Optional[Dict[str, Any]]:
        self.call_count += 1
        return self.keys_to_delete


@pytest.fixture
def cleanup_env():
    """Create a test environment with some stale facts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        db_path = tmp / "test.db"
        skills_dir = tmp / "skills"
        bundled_dir = tmp / "bundled"
        logs_dir = tmp / "logs"

        skills_dir.mkdir()
        bundled_dir.mkdir()
        logs_dir.mkdir()

        store = Store(db_path)
        pm = PersistentMemory(store)

        # Create some facts: some good, some ephemeral
        pm.upsert("preferred_http_client", "httpx", category="preference")  # Good
        pm.upsert("server_provider", "Hetzner CX22", category="env")  # Good
        pm.upsert("csv_file_path", "/tmp/x.csv", category="general")  # Ephemeral
        pm.upsert("file_count", "7 files in directory", category="general")  # Ephemeral

        yield tmp, db_path, store, pm
        store.close()


class TestMemoryCleanup:
    """Test the memory cleanup command."""

    @patch("obektclaw.__main__.CONFIG")
    @patch("obektclaw.llm.LLMClient")
    def test_cleanup_deletes_ephemeral_facts(
        self, mock_llm_cls, mock_config, cleanup_env
    ):
        """Cleanup should delete ephemeral facts identified by LLM."""
        tmp, db_path, store, pm = cleanup_env

        # Setup mock config
        mock_config.home = tmp
        mock_config.db_path = db_path
        mock_config.skills_dir = tmp / "skills"
        mock_config.bundled_skills_dir = tmp / "bundled"
        mock_config.llm_base_url = "http://fake"
        mock_config.llm_api_key = "fake"
        mock_config.llm_model = "fake"
        mock_config.llm_fast_model = "fake"

        # Setup fake LLM to identify ephemeral facts
        fake_llm = FakeCleanupLLM(keys_to_delete=["csv_file_path", "file_count"])
        mock_llm_cls.return_value = fake_llm

        # Run cleanup command
        result = main(["memory", "cleanup"])

        assert result == 0

        # Verify ephemeral facts deleted, good facts kept
        all_facts = []
        for cat in ("user", "project", "env", "preference", "general"):
            all_facts.extend(pm.list_category(cat, limit=200))

        fact_keys = {f.key for f in all_facts}
        assert "preferred_http_client" in fact_keys  # Good fact kept
        assert "server_provider" in fact_keys  # Good fact kept
        assert "csv_file_path" not in fact_keys  # Ephemeral deleted
        assert "file_count" not in fact_keys  # Ephemeral deleted

    @patch("obektclaw.__main__.CONFIG")
    @patch("obektclaw.llm.LLMClient")
    def test_cleanup_no_facts_to_delete(
        self, mock_llm_cls, mock_config, cleanup_env, capsys
    ):
        """Cleanup should report when no facts need deletion."""
        tmp, db_path, store, pm = cleanup_env

        mock_config.home = tmp
        mock_config.db_path = db_path
        mock_config.skills_dir = tmp / "skills"
        mock_config.bundled_skills_dir = tmp / "bundled"
        mock_config.llm_base_url = "http://fake"
        mock_config.llm_api_key = "fake"
        mock_config.llm_model = "fake"
        mock_config.llm_fast_model = "fake"

        # LLM returns empty list (nothing to delete)
        fake_llm = FakeCleanupLLM(keys_to_delete=[])
        mock_llm_cls.return_value = fake_llm

        result = main(["memory", "cleanup"])
        assert result == 0

        out, err = capsys.readouterr()
        assert "No facts identified for deletion" in out

    @patch("obektclaw.__main__.CONFIG")
    @patch("obektclaw.llm.LLMClient")
    def test_cleanup_empty_database(
        self, mock_llm_cls, mock_config, capsys
    ):
        """Cleanup should handle empty database gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "test.db"
            skills_dir = tmp / "skills"
            bundled_dir = tmp / "bundled"
            logs_dir = tmp / "logs"

            skills_dir.mkdir()
            bundled_dir.mkdir()
            logs_dir.mkdir()

            mock_config.home = tmp
            mock_config.db_path = db_path
            mock_config.skills_dir = skills_dir
            mock_config.bundled_skills_dir = bundled_dir
            mock_config.llm_base_url = "http://fake"
            mock_config.llm_api_key = "fake"
            mock_config.llm_model = "fake"
            mock_config.llm_fast_model = "fake"

            result = main(["memory", "cleanup"])
            assert result == 0

            out, err = capsys.readouterr()
            assert "(no facts to clean up)" in out

    @patch("obektclaw.__main__.CONFIG")
    @patch("obektclaw.llm.LLMClient")
    def test_cleanup_llm_returns_none(
        self, mock_llm_cls, mock_config, cleanup_env, capsys
    ):
        """Cleanup should handle LLM returning None."""
        tmp, db_path, store, pm = cleanup_env

        mock_config.home = tmp
        mock_config.db_path = db_path
        mock_config.skills_dir = tmp / "skills"
        mock_config.bundled_skills_dir = tmp / "bundled"
        mock_config.llm_base_url = "http://fake"
        mock_config.llm_api_key = "fake"
        mock_config.llm_model = "fake"
        mock_config.llm_fast_model = "fake"

        # LLM returns None (malformed response)
        fake_llm = FakeCleanupLLM(keys_to_delete=None)
        fake_llm.keys_to_delete = None  # Override to return None
        mock_llm_cls.return_value = fake_llm

        result = main(["memory", "cleanup"])
        assert result == 1  # Error exit code

        out, err = capsys.readouterr()
        assert "LLM did not return a list" in out

    @patch("obektclaw.__main__.CONFIG")
    @patch("obektclaw.llm.LLMClient")
    def test_cleanup_llm_returns_non_list(
        self, mock_llm_cls, mock_config, cleanup_env, capsys
    ):
        """Cleanup should handle LLM returning non-list JSON."""
        tmp, db_path, store, pm = cleanup_env

        mock_config.home = tmp
        mock_config.db_path = db_path
        mock_config.skills_dir = tmp / "skills"
        mock_config.bundled_skills_dir = tmp / "bundled"
        mock_config.llm_base_url = "http://fake"
        mock_config.llm_api_key = "fake"
        mock_config.llm_model = "fake"
        mock_config.llm_fast_model = "fake"

        # LLM returns a dict instead of list
        class DictLLM(FakeCleanupLLM):
            def chat_json(self, system, user, *, fast=True):
                return {"error": "wrong format"}

        mock_llm_cls.return_value = DictLLM()

        result = main(["memory", "cleanup"])
        assert result == 1

        out, err = capsys.readouterr()
        assert "LLM did not return a list" in out


class TestMemoryStatus:
    """Test the memory status command."""

    @patch("obektclaw.__main__.CONFIG")
    def test_memory_status(self, mock_config, capsys):
        """Memory status should show database stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "test.db"
            skills_dir = tmp / "skills"
            bundled_dir = tmp / "bundled"
            logs_dir = tmp / "logs"

            skills_dir.mkdir()
            bundled_dir.mkdir()
            logs_dir.mkdir()

            mock_config.home = tmp
            mock_config.db_path = db_path
            mock_config.skills_dir = skills_dir
            mock_config.bundled_skills_dir = bundled_dir

            # Create some data
            store = Store(db_path)
            session_id = store.open_session("test", "test_user")
            store.add_message(session_id, "user", "Hello")
            store.add_message(session_id, "assistant", "Hi")
            from obektclaw.memory import PersistentMemory
            pm = PersistentMemory(store)
            pm.upsert("test_fact", "test value", category="general")
            store.close()

            result = main(["memory", "status"])
            assert result == 0

            out, err = capsys.readouterr()
            assert "Sessions:" in out
            assert "Facts:" in out
            assert "Messages:" in out
            assert "FTS5" in out