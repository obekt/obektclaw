"""Tests for the memory cleanup CLI command.

Updated for VectorMemory-based cleanup.
"""

import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock

import pytest

from obektclaw.config import Config
from obektclaw.memory.store import Store
from obektclaw.memory import VectorMemory
from obektclaw.__main__ import main


class FakeCleanupLLM:
    """Fake LLM that returns specific cleanup responses."""

    def __init__(self, ids_to_delete: Optional[list[str]] = None):
        self.ids_to_delete = ids_to_delete or []
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
        return self.ids_to_delete


@pytest.fixture
def cleanup_env():
    """Create a test environment with some facts in VectorMemory."""
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
        # VectorMemory uses CONFIG.chroma_path internally - we need to patch CONFIG
        # For this fixture, we just return the store and create mock VectorMemory later
        yield tmp, db_path, store
        store.close()


class TestMemoryCleanup:
    """Test the memory cleanup command."""

    @patch("obektclaw.__main__.CONFIG")
    @patch("obektclaw.memory.VectorMemory")
    @patch("obektclaw.llm.LLMClient")
    def test_cleanup_deletes_ephemeral_facts(
        self, mock_llm_cls, mock_vm_cls, mock_config, cleanup_env
    ):
        """Cleanup should delete ephemeral facts identified by LLM."""
        tmp, db_path, store = cleanup_env

        # Setup mock config
        mock_config.home = tmp
        mock_config.db_path = db_path
        mock_config.skills_dir = tmp / "skills"
        mock_config.bundled_skills_dir = tmp / "bundled"
        mock_config.llm_base_url = "http://fake"
        mock_config.llm_api_key = "fake"
        mock_config.llm_model = "fake"
        mock_config.llm_fast_model = "fake"

        # Mock VectorMemory to return facts
        mock_vm_instance = MagicMock()
        mock_vm_cls.return_value = mock_vm_instance
        mock_vm_instance.get_recent_facts.return_value = [
            {"id": "fact_good_001", "content": "preferred HTTP client is httpx"},
            {"id": "fact_good_002", "content": "server hosted on Hetzner CX22"},
            {"id": "fact_ephemeral_001", "content": "csv_file_path: /tmp/x.csv"},
            {"id": "fact_ephemeral_002", "content": "file_count: 7 files"},
        ]

        # Track deleted facts
        deleted_ids = []
        mock_vm_instance.delete_fact.side_effect = lambda fid: deleted_ids.append(fid)

        # Setup fake LLM to identify ephemeral facts by ID
        fake_llm = FakeCleanupLLM(
            ids_to_delete=["fact_ephemeral_001", "fact_ephemeral_002"]
        )
        mock_llm_cls.return_value = fake_llm

        # Run cleanup command
        result = main(["memory", "cleanup"])

        assert result == 0

        # Verify ephemeral facts were marked for deletion
        assert "fact_ephemeral_001" in deleted_ids
        assert "fact_ephemeral_002" in deleted_ids
        # Good facts should NOT be deleted
        assert "fact_good_001" not in deleted_ids
        assert "fact_good_002" not in deleted_ids

    @patch("obektclaw.__main__.CONFIG")
    @patch("obektclaw.memory.VectorMemory")
    @patch("obektclaw.llm.LLMClient")
    def test_cleanup_no_facts_to_delete(
        self, mock_llm_cls, mock_vm_cls, mock_config, cleanup_env, capsys
    ):
        """Cleanup should report when no facts need deletion."""
        tmp, db_path, store = cleanup_env

        mock_config.home = tmp
        mock_config.db_path = db_path
        mock_config.skills_dir = tmp / "skills"
        mock_config.bundled_skills_dir = tmp / "bundled"
        mock_config.llm_base_url = "http://fake"
        mock_config.llm_api_key = "fake"
        mock_config.llm_model = "fake"
        mock_config.llm_fast_model = "fake"

        mock_vm_instance = MagicMock()
        mock_vm_cls.return_value = mock_vm_instance
        mock_vm_instance.get_recent_facts.return_value = [
            {"id": "fact_001", "content": "test"},
        ]

        # LLM returns empty list (nothing to delete)
        fake_llm = FakeCleanupLLM(ids_to_delete=[])
        mock_llm_cls.return_value = fake_llm

        result = main(["memory", "cleanup"])
        assert result == 0

        out, err = capsys.readouterr()
        assert "No facts identified for deletion" in out

    @patch("obektclaw.__main__.CONFIG")
    @patch("obektclaw.memory.VectorMemory")
    def test_cleanup_empty_database(self, mock_vm_cls, mock_config, capsys):
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

            mock_vm_instance = MagicMock()
            mock_vm_cls.return_value = mock_vm_instance
            mock_vm_instance.get_recent_facts.return_value = []

            result = main(["memory", "cleanup"])
            assert result == 0

            out, err = capsys.readouterr()
            assert "(no facts to clean up)" in out

    @patch("obektclaw.__main__.CONFIG")
    @patch("obektclaw.memory.VectorMemory")
    @patch("obektclaw.llm.LLMClient")
    def test_cleanup_llm_returns_none(
        self, mock_llm_cls, mock_vm_cls, mock_config, cleanup_env, capsys
    ):
        """Cleanup should handle LLM returning None."""
        tmp, db_path, store = cleanup_env

        mock_config.home = tmp
        mock_config.db_path = db_path
        mock_config.skills_dir = tmp / "skills"
        mock_config.bundled_skills_dir = tmp / "bundled"
        mock_config.llm_base_url = "http://fake"
        mock_config.llm_api_key = "fake"
        mock_config.llm_model = "fake"
        mock_config.llm_fast_model = "fake"

        mock_vm_instance = MagicMock()
        mock_vm_cls.return_value = mock_vm_instance
        mock_vm_instance.get_recent_facts.return_value = [
            {"id": "fact_001", "content": "test"},
        ]

        # LLM returns None (malformed response)
        fake_llm = FakeCleanupLLM(ids_to_delete=None)
        fake_llm.ids_to_delete = None  # Override to return None
        mock_llm_cls.return_value = fake_llm

        result = main(["memory", "cleanup"])
        assert result == 1  # Error exit code

        out, err = capsys.readouterr()
        assert "LLM did not return a list" in out

    @patch("obektclaw.__main__.CONFIG")
    @patch("obektclaw.memory.VectorMemory")
    @patch("obektclaw.llm.LLMClient")
    def test_cleanup_llm_returns_non_list(
        self, mock_llm_cls, mock_vm_cls, mock_config, cleanup_env, capsys
    ):
        """Cleanup should handle LLM returning non-list JSON."""
        tmp, db_path, store = cleanup_env

        mock_config.home = tmp
        mock_config.db_path = db_path
        mock_config.skills_dir = tmp / "skills"
        mock_config.bundled_skills_dir = tmp / "bundled"
        mock_config.llm_base_url = "http://fake"
        mock_config.llm_api_key = "fake"
        mock_config.llm_model = "fake"
        mock_config.llm_fast_model = "fake"

        mock_vm_instance = MagicMock()
        mock_vm_cls.return_value = mock_vm_instance
        mock_vm_instance.get_recent_facts.return_value = [
            {"id": "fact_001", "content": "test"},
        ]

        # LLM returns a dict instead of list
        class DictLLM(FakeCleanupLLM):
            def chat_json(self, system, user, *, fast=True):
                return {"error": "wrong format"}

        mock_llm_cls.return_value = DictLLM()

        result = main(["memory", "cleanup"])
        assert result == 1

        out, err = capsys.readouterr()
        assert "LLM did not return a list" in out

    @patch("obektclaw.__main__.CONFIG")
    @patch("obektclaw.memory.VectorMemory")
    @patch("obektclaw.llm.LLMClient")
    def test_cleanup_llm_returns_non_list(
        self, mock_llm_cls, mock_vm_cls, mock_config, cleanup_env, capsys
    ):
        """Cleanup should handle LLM returning non-list JSON."""
        tmp, db_path, store = cleanup_env

        mock_config.home = tmp
        mock_config.db_path = db_path
        mock_config.skills_dir = tmp / "skills"
        mock_config.bundled_skills_dir = tmp / "bundled"
        mock_config.llm_base_url = "http://fake"
        mock_config.llm_api_key = "fake"
        mock_config.llm_model = "fake"
        mock_config.llm_fast_model = "fake"

        mock_vm_instance = MagicMock()
        mock_vm_cls.return_value = mock_vm_instance
        mock_vm_instance.get_recent_facts.return_value = [
            {"id": "fact_001", "content": "test"},
        ]

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
            store.close()

            result = main(["memory", "status"])
            assert result == 0

            out, err = capsys.readouterr()
            assert "Sessions:" in out
            assert "Messages:" in out
            assert "FTS5" in out
