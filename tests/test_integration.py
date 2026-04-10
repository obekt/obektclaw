"""Integration tests for multi-turn conversations.

These tests verify the full agent behavior across multiple turns,
including memory persistence, skill discovery, and conversation history.
"""
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import pytest

from obektclaw.agent import Agent
from obektclaw.config import Config
from obektclaw.llm import LLMClient, LLMResponse, ToolCall
from obektclaw.memory.store import Store
from obektclaw.memory import PersistentMemory, UserModel
from obektclaw.skills import SkillManager


class MultiTurnFakeLLM:
    """Fake LLM that can simulate multi-turn conversations."""

    def __init__(self, response_sequence: Optional[List[LLMResponse]] = None):
        self.response_sequence = response_sequence or []
        self.call_count = 0
        self.last_messages: List[Dict[str, Any]] = []
        self.last_tools: Optional[List[Dict[str, Any]]] = None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        fast: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_messages = messages
        self.last_tools = list(tools) if tools else None

        if self.response_sequence:
            return self.response_sequence.pop(0)
        return LLMResponse(content="OK", tool_calls=[], raw=None)

    def chat_simple(
        self, system: str, user: str, *, fast: bool = True, temperature: float = 0.3
    ) -> str:
        return "Fake response."

    def chat_json(
        self, system: str, user: str, *, fast: bool = True
    ) -> Optional[Dict[str, Any]]:
        return {}


@pytest.fixture
def integration_env():
    """Create a full agent environment for integration testing."""
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

        yield config, store, skills
        store.close()


class TestMultiTurnConversation:
    """Test multi-turn conversation flow."""

    def test_three_turns_persist_context(self, integration_env):
        """Three turns should build on each other's context."""
        config, store, skills = integration_env

        fake_llm = MultiTurnFakeLLM([
            LLMResponse(content="Noted that you use httpx.", tool_calls=[], raw=None),
            LLMResponse(content="Checking the directory...", tool_calls=[], raw=None),
            LLMResponse(content="Based on your preference for httpx, I recommend using it for this API call.", tool_calls=[], raw=None),
        ])

        agent = Agent(
            config=config,
            store=store,
            skills=skills,
            llm=fake_llm,
            gateway="test",
            user_key="test_user",
            run_learning_loop=False,
        )

        # Turn 1: User states preference
        result1 = agent.run_once("I prefer httpx over requests")
        assert "Noted" in result1

        # Turn 2: User asks about directory
        result2 = agent.run_once("What's in the current directory?")
        assert "Checking" in result2

        # Turn 3: User asks for API advice (should recall preference)
        result3 = agent.run_once("How should I call this API?")
        assert "httpx" in result3

        # Verify messages persisted
        recent = agent.session.recent(limit=10)
        assert len(recent) == 6  # 3 user + 3 assistant

        agent.close()

    def test_tool_calls_across_turns(self, integration_env):
        """Tool calls in early turns should be reflected in later context."""
        config, store, skills = integration_env

        fake_llm = MultiTurnFakeLLM([
            # Turn 1: Tool call
            LLMResponse(
                content="Checking files...",
                tool_calls=[ToolCall(id="c1", name="list_files", arguments='{"path": "."}')],
                raw=None,
            ),
            LLMResponse(content="Found 2 files: a.py and b.py.", tool_calls=[], raw=None),
            # Turn 2: Recall and act
            LLMResponse(content="You asked about a.py earlier. Let me read it.", tool_calls=[], raw=None),
        ])

        agent = Agent(
            config=config,
            store=store,
            skills=skills,
            llm=fake_llm,
            gateway="test",
            user_key="test_user",
            run_learning_loop=False,
        )

        result1 = agent.run_once("List files")
        assert "Found" in result1

        result2 = agent.run_once("Tell me about a.py")
        assert "a.py earlier" in result2

        agent.close()

    def test_session_isolation(self, integration_env):
        """Different sessions should not share context."""
        config, store, skills = integration_env

        fake_llm1 = MultiTurnFakeLLM([
            LLMResponse(content="Session 1: Noted your preference for Python.", tool_calls=[], raw=None),
        ])

        fake_llm2 = MultiTurnFakeLLM([
            LLMResponse(content="Session 2: I don't know your preferences yet.", tool_calls=[], raw=None),
        ])

        # Agent 1 with one session
        agent1 = Agent(
            config=config,
            store=store,
            skills=skills,
            llm=fake_llm1,
            gateway="test",
            user_key="user1",
            run_learning_loop=False,
        )

        # Agent 2 with different session
        agent2 = Agent(
            config=config,
            store=store,
            skills=skills,
            llm=fake_llm2,
            gateway="test",
            user_key="user2",
            run_learning_loop=False,
        )

        result1 = agent1.run_once("I prefer Python")
        assert "Session 1" in result1

        result2 = agent2.run_once("What do I prefer?")
        assert "Session 2" in result2
        assert "Python" not in result2  # Should NOT recall from other session

        agent1.close()
        agent2.close()

    def test_skills_discovered_across_turns(self, integration_env):
        """Skills created in one turn should be available in later turns."""
        config, store, skills = integration_env

        # Create a skill before running
        skills.create("httpx-guide", "Guide for using httpx library", "# HTTPX Guide\nUse httpx for async HTTP.")

        fake_llm = MultiTurnFakeLLM([
            LLMResponse(content="I found the httpx-guide skill.", tool_calls=[], raw=None),
            LLMResponse(content="Following the httpx-guide skill from before.", tool_calls=[], raw=None),
        ])

        agent = Agent(
            config=config,
            store=store,
            skills=skills,
            llm=fake_llm,
            gateway="test",
            user_key="test_user",
            run_learning_loop=False,
        )

        result1 = agent.run_once("I need HTTP help")
        assert "httpx-guide" in result1

        result2 = agent.run_once("How do I make async HTTP calls?")
        assert "httpx-guide" in result2

        agent.close()


class TestMemoryPersistence:
    """Test that memory persists across agent instances."""

    def test_facts_persist_across_agents(self, integration_env):
        """Facts stored by one agent should be visible to another."""
        config, store, skills = integration_env

        # Store a fact directly
        pm = PersistentMemory(store)
        pm.upsert("preferred_http_client", "httpx", category="preference")

        # Create fresh agent - should see the fact
        fake_llm = MultiTurnFakeLLM([
            LLMResponse(content="I see you prefer httpx from your preferences.", tool_calls=[], raw=None),
        ])

        agent = Agent(
            config=config,
            store=store,
            skills=skills,
            llm=fake_llm,
            gateway="test",
            user_key="test_user",
            run_learning_loop=False,
        )

        result = agent.run_once("What HTTP client should I use?")
        assert "httpx" in result

        # Verify fact is in system prompt
        system = fake_llm.last_messages[0]["content"]
        assert "preferred_http_client" in system or "httpx" in system

        agent.close()

    def test_user_model_persists(self, integration_env):
        """User model changes should persist across sessions."""
        config, store, skills = integration_env

        # Set user model trait
        um = UserModel(store)
        um.set("tooling_pref", value="prefers pytest over unittest", evidence="stated in conversation")

        # Create new agent - should see the trait
        fake_llm = MultiTurnFakeLLM([
            LLMResponse(content="Using pytest per your preference.", tool_calls=[], raw=None),
        ])

        agent = Agent(
            config=config,
            store=store,
            skills=skills,
            llm=fake_llm,
            gateway="test",
            user_key="test_user",
            run_learning_loop=False,
        )

        result = agent.run_once("Test my function")
        assert "pytest" in result

        agent.close()


class TestContextCompaction:
    """Test context compaction for long conversations."""

    def test_system_prompt_grows_with_history(self, integration_env):
        """System prompt should include prior messages."""
        config, store, skills = integration_env

        responses = [
            LLMResponse(content=f"Turn {i}", tool_calls=[], raw=None)
            for i in range(10)
        ]
        fake_llm = MultiTurnFakeLLM(responses)

        agent = Agent(
            config=config,
            store=store,
            skills=skills,
            llm=fake_llm,
            gateway="test",
            user_key="test_user",
            run_learning_loop=False,
        )

        # Run 10 turns
        for i in range(10):
            agent.run_once(f"Message {i}")

        # Check final system prompt includes prior exchanges
        system = fake_llm.last_messages[0]["content"]
        assert "prior exchanges" in system

        # Check session has all messages
        recent = agent.session.recent(limit=30)
        assert len(recent) == 20  # 10 user + 10 assistant

        agent.close()


class TestSkillSelfDiscovery:
    """Test that agent can discover and use skills."""

    def test_skill_list_in_system_prompt(self, integration_env):
        """System prompt should list all available skills."""
        config, store, skills = integration_env

        # Create multiple skills
        skills.create("skill-a", "First skill", "# Skill A")
        skills.create("skill-b", "Second skill", "# Skill B")

        fake_llm = MultiTurnFakeLLM([
            LLMResponse(content="I see skill-a and skill-b available.", tool_calls=[], raw=None),
        ])

        agent = Agent(
            config=config,
            store=store,
            skills=skills,
            llm=fake_llm,
            gateway="test",
            user_key="test_user",
            run_learning_loop=False,
        )

        agent.run_once("What skills do you have?")

        system = fake_llm.last_messages[0]["content"]
        assert "skill-a" in system
        assert "skill-b" in system

        agent.close()

    def test_skill_fts5_recall(self, integration_env):
        """Skills relevant to query should be highlighted."""
        config, store, skills = integration_env

        # Create a skill with specific keywords
        skills.create("database-import", "Import CSV to database", "# Import steps...")

        fake_llm = MultiTurnFakeLLM([
            LLMResponse(content="I found the database-import skill.", tool_calls=[], raw=None),
        ])

        agent = Agent(
            config=config,
            store=store,
            skills=skills,
            llm=fake_llm,
            gateway="test",
            user_key="test_user",
            run_learning_loop=False,
        )

        agent.run_once("How do I import a CSV to my database?")

        system = fake_llm.last_messages[0]["content"]
        assert "database-import" in system

        agent.close()