"""Tests for obektclaw/learning.py — Learning Loop retrospection.

These tests verify that the Learning Loop correctly:
1. Parses LLM JSON responses
2. Persists facts to PersistentMemory
3. Updates user model layers
4. Creates and improves skills
"""
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pytest

from obektclaw.agent import Agent, Turn
from obektclaw.config import Config
from obektclaw.learning import LearningLoop
from obektclaw.llm import LLMClient, LLMResponse
from obektclaw.memory.store import Store
from obektclaw.skills import SkillManager


class RecordingFakeLLM:
    """Fake LLM that records chat_json calls and returns a preset JSON."""

    def __init__(self, retro_response: Optional[Dict[str, Any]] = None):
        self.retro_response = retro_response
        self.chat_json_calls: List[Tuple[str, str]] = []
        self.chat_calls: List[Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]] = []

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        fast: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self.chat_calls.append((messages, tools))
        # Return a simple response with no tool calls
        return LLMResponse(content="I understand.", tool_calls=[], raw=None)

    def chat_json(
        self, system: str, user: str, *, fast: bool = True
    ) -> Optional[Dict[str, Any]]:
        self.chat_json_calls.append((system, user))
        return self.retro_response


@pytest.fixture
def learning_env():
    """Create agent + learning loop with recording fake LLM."""
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
        skills = SkillManager(store, skills_dir, bundled_dir)

        config = Config(
            home=tmp,
            db_path=db_path,
            skills_dir=skills_dir,
            bundled_skills_dir=bundled_dir,
            logs_dir=logs_dir,
            llm_base_url="http://fake",
            llm_api_key="fake",
            llm_model="fake",
            llm_fast_model="fake",
            tg_token="",
            tg_allowed_chat_ids=(),
            bash_timeout=30,
            workdir=tmp,
        )

        fake_llm = RecordingFakeLLM()

        agent = Agent(
            config=config,
            store=store,
            skills=skills,
            llm=fake_llm,
            gateway="test",
            user_key="test_user",
            run_learning_loop=False,  # We'll call it manually
        )

        loop = LearningLoop(agent)

        yield agent, loop, fake_llm, store, skills
        agent.close()


class TestLearningLoopBasic:
    """Basic Learning Loop tests."""

    def test_skips_trivial_input(self, learning_env):
        """Learning loop should skip inputs < 12 chars with no tool steps."""
        agent, loop, fake_llm, _, _ = learning_env
        
        turn = Turn(user_text="Hi", assistant_text="Hello!", tool_steps=0)
        loop.run(turn)
        
        assert len(fake_llm.chat_json_calls) == 0

    def test_runs_on_substantial_input(self, learning_env):
        """Learning loop should run on substantial input."""
        agent, loop, fake_llm, _, _ = learning_env
        fake_llm.retro_response = {"notes": "test"}
        
        turn = Turn(
            user_text="Tell me about httpx vs requests",
            assistant_text="httpx is async-capable.",
            tool_steps=0,
        )
        loop.run(turn)
        
        assert len(fake_llm.chat_json_calls) == 1

    def test_runs_when_tools_used(self, learning_env):
        """Learning loop should run when tool steps were used."""
        agent, loop, fake_llm, _, _ = learning_env
        fake_llm.retro_response = {"notes": "test"}
        
        # Short text but with tool steps
        turn = Turn(user_text="ls", assistant_text="file.txt", tool_steps=1)
        loop.run(turn)
        
        assert len(fake_llm.chat_json_calls) == 1


class TestFactPersistence:
    """Test that facts from retro are persisted."""

    def test_saves_single_fact(self, learning_env):
        agent, loop, fake_llm, store, _ = learning_env
        
        fake_llm.retro_response = {
            "facts": [
                {
                    "category": "preference",
                    "key": "http_client",
                    "value": "prefers httpx over requests",
                    "confidence": 0.9,
                }
            ],
            "user_model_updates": [],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "User has a strong preference.",
        }
        
        turn = Turn(
            user_text="I always use httpx",
            assistant_text="Noted.",
            tool_steps=0,
        )
        loop.run(turn)
        
        # Verify fact was saved
        rows = store.fetchall(
            "SELECT key, value, category, confidence FROM facts WHERE key = ?",
            ("http_client",)
        )
        assert len(rows) == 1
        assert rows[0]["value"] == "prefers httpx over requests"
        assert rows[0]["category"] == "preference"
        assert rows[0]["confidence"] == 0.9

    def test_saves_multiple_facts(self, learning_env):
        agent, loop, fake_llm, store, _ = learning_env
        
        fake_llm.retro_response = {
            "facts": [
                {
                    "category": "preference",
                    "key": "http_client",
                    "value": "httpx",
                    "confidence": 0.9,
                },
                {
                    "category": "env",
                    "key": "server_host",
                    "value": "Hetzner CX22",
                    "confidence": 0.95,
                },
            ],
            "user_model_updates": [],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "Multiple facts extracted.",
        }
        
        turn = Turn(
            user_text="My server is on Hetzner and I use httpx",
            assistant_text="Got it.",
            tool_steps=0,
        )
        loop.run(turn)
        
        rows = store.fetchall("SELECT key, value FROM facts ORDER BY key")
        assert len(rows) == 2
        fact_dict = {r["key"]: r["value"] for r in rows}
        assert fact_dict["http_client"] == "httpx"
        assert fact_dict["server_host"] == "Hetzner CX22"

    def test_ignores_malformed_fact(self, learning_env):
        """Malformed facts should be silently skipped."""
        agent, loop, fake_llm, store, _ = learning_env
        
        fake_llm.retro_response = {
            "facts": [
                {"key": "valid", "value": "test", "category": "general"},
                {"missing_value": True},  # Missing required fields
                None,  # Null entry
            ],
            "user_model_updates": [],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "test",
        }
        
        turn = Turn(user_text="test", assistant_text="test", tool_steps=1)
        loop.run(turn)
        
        # Only the valid fact should be saved
        rows = store.fetchall("SELECT key FROM facts")
        keys = [r["key"] for r in rows]
        assert "valid" in keys
        assert len(keys) == 1


class TestUserModelUpdates:
    """Test that user model updates are applied."""

    def test_updates_single_layer(self, learning_env):
        agent, loop, fake_llm, store, _ = learning_env
        
        fake_llm.retro_response = {
            "facts": [],
            "user_model_updates": [
                {
                    "layer": "tooling_pref",
                    "value": "httpx over requests",
                    "evidence": "User stated preference",
                }
            ],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "Updated tooling preference.",
        }
        
        turn = Turn(
            user_text="I prefer httpx",
            assistant_text="OK",
            tool_steps=0,
        )
        loop.run(turn)
        
        row = store.fetchone(
            "SELECT value, evidence FROM user_traits WHERE layer = ?",
            ("tooling_pref",)
        )
        assert row is not None
        assert row["value"] == "httpx over requests"
        assert row["evidence"] == "User stated preference"

    def test_updates_multiple_layers(self, learning_env):
        agent, loop, fake_llm, store, _ = learning_env
        
        fake_llm.retro_response = {
            "facts": [],
            "user_model_updates": [
                {
                    "layer": "tooling_pref",
                    "value": "httpx",
                    "evidence": "stated",
                },
                {
                    "layer": "technical_level",
                    "value": "intermediate",
                    "evidence": "inferred from questions",
                },
            ],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "Multiple updates.",
        }
        
        turn = Turn(user_text="test", assistant_text="test", tool_steps=1)
        loop.run(turn)
        
        rows = store.fetchall("SELECT layer, value FROM user_traits")
        assert len(rows) == 2
        layers = {r["layer"]: r["value"] for r in rows}
        assert layers["tooling_pref"] == "httpx"
        assert layers["technical_level"] == "intermediate"

    def test_ignores_invalid_layer(self, learning_env):
        """Invalid layer names should be silently skipped."""
        agent, loop, fake_llm, store, _ = learning_env
        
        fake_llm.retro_response = {
            "facts": [],
            "user_model_updates": [
                {
                    "layer": "invalid_layer_xyz",
                    "value": "test",
                    "evidence": "test",
                },
                {
                    "layer": "tooling_pref",
                    "value": "valid",
                    "evidence": "test",
                },
            ],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "test",
        }
        
        turn = Turn(user_text="test", assistant_text="test", tool_steps=1)
        loop.run(turn)
        
        rows = store.fetchall("SELECT layer FROM user_traits")
        layers = [r["layer"] for r in rows]
        assert "invalid_layer_xyz" not in layers
        assert "tooling_pref" in layers


class TestSkillCreation:
    """Test that new skills are created from retro."""

    def test_creates_new_skill(self, learning_env):
        agent, loop, fake_llm, store, skills = learning_env
        
        fake_llm.retro_response = {
            "facts": [],
            "user_model_updates": [],
            "new_skill": {
                "name": "async-http-client",
                "description": "Use httpx for async HTTP calls",
                "body": "# Steps\n1. Import httpx\n2. Use AsyncClient",
            },
            "skill_improvement": None,
            "notes": "Created new skill.",
        }
        
        turn = Turn(
            user_text="How do I make async HTTP calls?",
            assistant_text="Use httpx.AsyncClient",
            tool_steps=0,
        )
        loop.run(turn)
        
        skill = skills.get("async-http-client")
        assert skill is not None
        assert "httpx" in skill.description
        assert "AsyncClient" in skill.body

    def test_ignores_skill_with_missing_name(self, learning_env):
        """Skills without names should be skipped."""
        agent, loop, fake_llm, store, skills = learning_env

        fake_llm.retro_response = {
            "facts": [],
            "user_model_updates": [],
            "new_skill": {
                "name": None,  # No name
                "description": "test",
                "body": "test",
            },
            "skill_improvement": None,
            "notes": "test",
        }

        turn = Turn(user_text="test", assistant_text="test", tool_steps=1)
        loop.run(turn)

        # No new skills should be created
        all_skills = skills.list_all()
        assert len(all_skills) == 0


class TestSkillImprovement:
    """Test that skills are improved via append."""

    def test_appends_to_existing_skill(self, learning_env):
        agent, loop, fake_llm, store, skills = learning_env

        # Create initial skill
        skills.create("deploy", "Deploy to prod", "# Original\nOriginal content")
        
        fake_llm.retro_response = {
            "facts": [],
            "user_model_updates": [],
            "new_skill": None,
            "skill_improvement": {
                "name": "deploy",
                "append": "# Gotcha\nDon't forget to restart the service.",
            },
            "skill_improvement_alt": None,
            "notes": "Improved deploy skill.",
        }
        
        turn = Turn(
            user_text="Deployed successfully",
            assistant_text="Great!",
            tool_steps=0,
        )
        loop.run(turn)
        
        improved = skills.get("deploy")
        assert improved is not None
        assert "Original content" in improved.body
        assert "Don't forget to restart" in improved.body

    def test_ignores_improvement_for_missing_skill(self, learning_env):
        """Improvements for non-existent skills should be skipped."""
        agent, loop, fake_llm, _, skills = learning_env
        
        initial_count = len(skills.list_all())
        
        fake_llm.retro_response = {
            "facts": [],
            "user_model_updates": [],
            "new_skill": None,
            "skill_improvement": {
                "name": "nonexistent",
                "append": "test",
            },
            "notes": "test",
        }
        
        turn = Turn(user_text="test", assistant_text="test", tool_steps=1)
        loop.run(turn)
        
        assert len(skills.list_all()) == initial_count


class TestLearningLoopNotes:
    """Test that notes are added to session memory."""

    def test_notes_added_as_system_message(self, learning_env):
        agent, loop, fake_llm, _, _ = learning_env
        
        fake_llm.retro_response = {
            "facts": [],
            "user_model_updates": [],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "User revealed a strong preference for async patterns.",
        }
        
        turn = Turn(
            user_text="I love async code",
            assistant_text="Async is great.",
            tool_steps=0,
        )
        loop.run(turn)
        
        recent = agent.session.recent(limit=10)
        system_msgs = [m for m in recent if m.role == "system"]
        assert len(system_msgs) >= 1
        assert "learning loop" in system_msgs[0].content
        assert "async patterns" in system_msgs[0].content


class TestEmptyRetro:
    """Test behavior when retro returns empty/null."""

    def test_none_response_handled(self, learning_env):
        """None response should be handled gracefully."""
        agent, loop, fake_llm, _, _ = learning_env
        fake_llm.retro_response = None
        
        turn = Turn(user_text="test", assistant_text="test", tool_steps=1)
        # Should not raise
        loop.run(turn)

    def test_empty_arrays_handled(self, learning_env):
        """Empty arrays should result in no changes."""
        agent, loop, fake_llm, store, skills = learning_env

        fake_llm.retro_response = {
            "facts": [],
            "user_model_updates": [],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "Nothing to learn.",
        }

        initial_facts = store.fetchall("SELECT COUNT(*) as c FROM facts")[0]["c"]

        turn = Turn(user_text="test", assistant_text="test", tool_steps=1)
        loop.run(turn)

        final_facts = store.fetchall("SELECT COUNT(*) as c FROM facts")[0]["c"]
        assert final_facts == initial_facts
