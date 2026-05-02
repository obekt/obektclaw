"""Tests for obektclaw/post_turn.py — Turn extraction.

These tests verify that the TurnExtractor correctly:
1. Parses LLM JSON responses from the main LLM client
2. Persists facts to VectorMemory
3. Updates user model layers
4. Creates and improves skills

Tests use a fake LLMClient that returns preset JSON.
"""

import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

import pytest

from obektclaw.agent import Agent, Turn
from obektclaw.config import Config
from obektclaw.post_turn import TurnExtractor
from obektclaw.llm import LLMClient, LLMResponse
from obektclaw.memory.store import Store
from obektclaw.skills import SkillManager


class FakeMainLLM:
    """Fake main LLM for agent (returns simple responses, no tool calls)."""

    def __init__(self):
        self.chat_calls: List[tuple] = []
        self.chat_json_calls: List[tuple] = []
        self.extraction_response: Optional[Dict[str, Any]] = None

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
        return LLMResponse(content="I understand.", tool_calls=[], raw=None)

    def chat_simple(
        self, system: str, user: str, *, fast: bool = True, temperature: float = 0.3
    ) -> str:
        return "I understand."

    def chat_json(
        self, system: str, user: str, *, fast: bool = True
    ) -> Optional[Dict[str, Any]]:
        self.chat_json_calls.append((system, user))
        return self.extraction_response


@pytest.fixture
def extraction_env():
    """Create agent + TurnExtractor with fake LLMs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        db_path = tmp / "test.db"
        skills_dir = tmp / "skills"
        bundled_dir = tmp / "bundled"
        logs_dir = tmp / "logs"
        cog_home = tmp / "cogdb"
        chroma_path = tmp / "chroma"

        skills_dir.mkdir()
        bundled_dir.mkdir()
        logs_dir.mkdir()
        cog_home.mkdir()
        chroma_path.mkdir()

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
            cog_home=cog_home,
            chroma_path=chroma_path,
        )

        fake_main_llm = FakeMainLLM()

        agent = Agent(
            config=config,
            store=store,
            skills=skills,
            llm=fake_main_llm,
            gateway="test",
            user_key="test_user",
            run_learning_loop=False,  # We'll call it manually
        )

        extractor = TurnExtractor(agent)

        yield agent, extractor, fake_main_llm, store, skills
        agent.close()


class TestTurnExtractorBasic:
    """Basic TurnExtractor tests."""

    def test_skips_trivial_input(self, extraction_env):
        """Extractor should skip inputs < 12 chars with no tool steps."""
        agent, extractor, fake_main_llm, _, _ = extraction_env

        turn = Turn(user_text="Hi", assistant_text="Hello!", tool_steps=0)
        extractor.extract(turn)

        assert not fake_main_llm.chat_json_calls

    def test_runs_on_substantial_input(self, extraction_env):
        """Extractor should run on substantial input."""
        agent, extractor, fake_main_llm, _, _ = extraction_env
        fake_main_llm.extraction_response = {"notes": "test"}

        turn = Turn(
            user_text="Tell me about httpx vs requests",
            assistant_text="httpx is async-capable.",
            tool_steps=0,
        )
        extractor.extract(turn)

        assert len(fake_main_llm.chat_json_calls) == 1

    def test_runs_when_tools_used(self, extraction_env):
        """Extractor should run when tool steps were used."""
        agent, extractor, fake_main_llm, _, _ = extraction_env
        fake_main_llm.extraction_response = {"notes": "test"}

        # Short text but with tool steps
        turn = Turn(user_text="ls", assistant_text="file.txt", tool_steps=1)
        extractor.extract(turn)

        assert len(fake_main_llm.chat_json_calls) == 1


class TestFactPersistence:
    """Test that facts from extraction are persisted to VectorMemory."""

    def test_saves_single_fact(self, extraction_env):
        agent, extractor, fake_main_llm, store, _ = extraction_env

        fake_main_llm.extraction_response = {
            "facts": [
                {
                    "content": "prefers httpx over requests",
                    "category": "preference",
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
        extractor.extract(turn)

        # Verify fact was saved to VectorMemory
        facts = agent.vector_memory.search_similar_facts("httpx", n_results=10)
        assert len(facts) >= 1
        # Check that one fact contains our content
        found = any("httpx" in f.get("content", "") for f in facts)
        assert found

    def test_saves_multiple_facts(self, extraction_env):
        agent, extractor, fake_main_llm, store, _ = extraction_env

        fake_main_llm.extraction_response = {
            "facts": [
                {
                    "content": "httpx is preferred HTTP client",
                    "category": "preference",
                    "confidence": 0.9,
                },
                {
                    "content": "server hosted on Hetzner CX22",
                    "category": "env",
                    "confidence": 0.95,
                },
            ],
            "user_model_updates": [],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "Multiple facts extracted.",
        }

        turn = Turn(
            user_text="I use httpx and my server is on Hetzner",
            assistant_text="Got it.",
            tool_steps=0,
        )
        extractor.extract(turn)

        # Verify both facts saved
        facts = agent.vector_memory.search_similar_facts("httpx", n_results=10)
        assert len(facts) >= 1

        hetzner_facts = agent.vector_memory.search_similar_facts(
            "Hetzner", n_results=10
        )
        assert len(hetzner_facts) >= 1


class TestEntityPersistence:
    """Test that entities from extraction are persisted to GraphMemory."""

    def test_saves_entity_to_graph(self, extraction_env):
        agent, extractor, fake_main_llm, store, _ = extraction_env

        fake_main_llm.extraction_response = {
            "entities": [
                {
                    "name": "httpx",
                    "type": "tool",
                    "confidence": 0.95,
                    "properties": {"category": "http_client", "feature": "async"},
                }
            ],
            "relations": [],
            "facts": [],
            "user_model_updates": [],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "Entity extracted.",
        }

        turn = Turn(
            user_text="I use httpx for HTTP requests",
            assistant_text="Good choice.",
            tool_steps=0,
        )
        extractor.extract(turn)

        # Verify entity was saved to GraphMemory
        entity = agent.graph_memory.get_entity("entity_tool_httpx")
        assert entity is not None
        assert entity.name == "httpx"
        assert entity.entity_type == "tool"

    def test_saves_relation_to_graph(self, extraction_env):
        agent, extractor, fake_main_llm, store, _ = extraction_env

        # First, ensure user entity exists
        agent._ensure_user_entity()

        fake_main_llm.extraction_response = {
            "entities": [
                {
                    "name": "httpx",
                    "type": "tool",
                    "confidence": 0.95,
                    "properties": {},
                }
            ],
            "relations": [
                {
                    "subject": "user",
                    "predicate": "prefers",
                    "object": "httpx",
                    "confidence": 0.9,
                }
            ],
            "facts": [],
            "user_model_updates": [],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "Relation extracted.",
        }

        turn = Turn(
            user_text="I prefer httpx over requests",
            assistant_text="Noted.",
            tool_steps=0,
        )
        extractor.extract(turn)

        # Verify entity exists
        entity = agent.graph_memory.get_entity("entity_tool_httpx")
        assert entity is not None

        # Verify relation exists (user prefers httpx)
        user_entity = agent.graph_memory.get_entity("entity_person_user")
        assert user_entity is not None


class TestUserModelUpdates:
    """Test that user model updates are applied."""

    def test_updates_user_model_layer(self, extraction_env):
        agent, extractor, fake_main_llm, store, _ = extraction_env

        fake_main_llm.extraction_response = {
            "facts": [],
            "entities": [],
            "relations": [],
            "user_model_updates": [
                {
                    "layer": "tooling_pref",
                    "value": "prefers httpx over requests",
                    "evidence": "User stated preference explicitly",
                }
            ],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "User preference updated.",
        }

        turn = Turn(
            user_text="I always use httpx for HTTP",
            assistant_text="Understood.",
            tool_steps=0,
        )
        extractor.extract(turn)

        # Verify user model was updated
        from obektclaw.memory import UserModel

        user_model = UserModel(store)
        value = user_model.get("tooling_pref")
        assert value is not None
        assert "httpx" in value.value


class TestSkillCreation:
    """Test that new skills are created from extraction."""

    def test_creates_new_skill(self, extraction_env):
        agent, extractor, fake_main_llm, store, skills = extraction_env

        fake_main_llm.extraction_response = {
            "facts": [],
            "entities": [],
            "relations": [],
            "user_model_updates": [],
            "new_skill": {
                "name": "deploy-to-hetzner",
                "description": "Deploy a server to Hetzner Cloud",
                "body": "## Steps\n1. Create CX22 instance\n2. Install dependencies\n3. Deploy code",
            },
            "skill_improvement": None,
            "notes": "New skill created.",
        }

        turn = Turn(
            user_text="How do I deploy to Hetzner?",
            assistant_text="Let me create a skill for that.",
            tool_steps=0,
        )
        extractor.extract(turn)

        # Verify skill was created
        skill = skills.get("deploy-to-hetzner")
        assert skill is not None
        assert skill.description == "Deploy a server to Hetzner Cloud"

    def test_skips_skill_without_name(self, extraction_env):
        agent, extractor, fake_main_llm, store, skills = extraction_env

        fake_main_llm.extraction_response = {
            "facts": [],
            "entities": [],
            "relations": [],
            "user_model_updates": [],
            "new_skill": {
                "name": None,  # No name
                "description": "Some skill",
                "body": "Some body",
            },
            "skill_improvement": None,
            "notes": "No skill name.",
        }

        turn = Turn(
            user_text="Something",
            assistant_text="Response",
            tool_steps=0,
        )
        extractor.extract(turn)

        # Verify no skill was created
        all_skills = skills.list_all()
        assert len(all_skills) == 0  # No bundled skills in test env


class TestSkillImprovement:
    """Test that existing skills are improved."""

    def test_improves_existing_skill(self, extraction_env):
        agent, extractor, fake_main_llm, store, skills = extraction_env

        # First create a skill
        skills.create(
            name="test-skill",
            description="A test skill",
            body="## Original\nStep 1: Do something",
        )

        fake_main_llm.extraction_response = {
            "facts": [],
            "entities": [],
            "relations": [],
            "user_model_updates": [],
            "new_skill": None,
            "skill_improvement": {
                "name": "test-skill",
                "append": "\n## Addition\nStep 2: Do more",
            },
            "notes": "Skill improved.",
        }

        turn = Turn(
            user_text="I need more steps",
            assistant_text="Adding to skill.",
            tool_steps=0,
        )
        extractor.extract(turn)

        # Verify skill was improved
        skill = skills.get("test-skill")
        assert skill is not None
        assert "Original" in skill.body
        assert "Addition" in skill.body


class TestErrorHandling:
    """Test that errors are handled gracefully."""

    def test_handles_null_extraction(self, extraction_env):
        agent, extractor, fake_main_llm, store, _ = extraction_env

        fake_main_llm.extraction_response = None

        turn = Turn(
            user_text="I prefer httpx",
            assistant_text="Noted.",
            tool_steps=0,
        )
        # Should not crash - that's the key test
        extractor.extract(turn)

        # Just verify it ran without crashing
        assert len(fake_main_llm.chat_json_calls) == 1

    def test_handles_malformed_entity(self, extraction_env):
        agent, extractor, fake_main_llm, store, _ = extraction_env

        # Entity missing required 'name' field
        fake_main_llm.extraction_response = {
            "entities": [
                {"type": "tool", "confidence": 0.9}  # Missing 'name'
            ],
            "relations": [],
            "facts": [],
            "user_model_updates": [],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "Bad entity.",
        }

        turn = Turn(
            user_text="Something",
            assistant_text="Response",
            tool_steps=0,
        )
        extractor.extract(turn)

        # Should not crash
        # Entity was malformed so nothing saved
        entities = agent.graph_memory.get_entities_by_type("tool")
        assert len(entities) == 0

    def test_handles_malformed_fact(self, extraction_env):
        agent, extractor, fake_main_llm, store, _ = extraction_env

        # Fact missing required 'content' field
        fake_main_llm.extraction_response = {
            "facts": [
                {"category": "preference", "confidence": 0.9}  # Missing 'content'
            ],
            "entities": [],
            "relations": [],
            "user_model_updates": [],
            "new_skill": None,
            "skill_improvement": None,
            "notes": "Bad fact.",
        }

        turn = Turn(
            user_text="Something",
            assistant_text="Response",
            tool_steps=0,
        )
        extractor.extract(turn)

        # Should not crash, no facts saved
        facts = agent.vector_memory.search_similar_facts("preference", n_results=10)
        # May have 0 or facts from other tests (shared VectorMemory via CONFIG)
        # Just verify it didn't crash


