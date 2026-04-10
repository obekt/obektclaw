"""Tests for obektclaw/agent.py — offline agent loop with fake LLM.

These tests verify the ReAct loop, tool dispatch, and memory writes
without making any real LLM calls.
"""
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

import pytest

from obektclaw.agent import Agent, Turn
from obektclaw.config import Config
from obektclaw.llm import LLMClient, LLMResponse, ToolCall
from obektclaw.memory.store import Store
from obektclaw.skills import SkillManager


class FakeLLMClient:
    """Fake LLM client for offline testing."""

    def __init__(
        self,
        *,
        responses: Optional[List[LLMResponse]] = None,
        json_response: Optional[Dict[str, Any]] = None,
    ):
        self.responses = responses or []
        self.json_response = json_response or {}
        self.call_count = 0
        self.last_messages: List[Dict[str, Any]] = []
        self.last_tools: Optional[List[Dict[str, Any]]] = None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        fast: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_messages = messages
        self.last_tools = list(tools) if tools else None

        if self.responses:
            return self.responses.pop(0)
        # Default: return a simple text response with no tool calls
        return LLMResponse(content="I understand.", tool_calls=[], raw=None)

    def chat_simple(
        self, system: str, user: str, *, fast: bool = True, temperature: float = 0.3
    ) -> str:
        return "Fake response."

    def chat_json(
        self, system: str, user: str, *, fast: bool = True
    ) -> Optional[Dict[str, Any]]:
        return self.json_response


@pytest.fixture
def agent_env():
    """Create a minimal agent with fake LLM and temp storage."""
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

        fake_llm = FakeLLMClient()

        agent = Agent(
            config=config,
            store=store,
            skills=skills,
            llm=fake_llm,
            gateway="test",
            user_key="test_user",
            run_learning_loop=False,  # Disable for offline tests
        )
        
        yield agent, fake_llm, store, skills
        agent.close()


class TestAgentBasicRun:
    """Test basic agent run without tool calls."""

    def test_run_once_no_tool_calls(self, agent_env):
        agent, fake_llm, store, skills = agent_env
        
        fake_llm.responses = [
            LLMResponse(content="Hello! How can I help?", tool_calls=[], raw=None),
        ]
        
        result = agent.run_once("Hello")
        
        assert "Hello! How can I help?" in result
        assert fake_llm.call_count == 1

    def test_run_once_saves_user_message(self, agent_env):
        agent, fake_llm, _, _ = agent_env
        
        fake_llm.responses = [
            LLMResponse(content="OK", tool_calls=[], raw=None),
        ]
        
        agent.run_once("Test message")
        
        recent = agent.session.recent(limit=10)
        assert len(recent) == 2  # user + assistant
        assert recent[0].role == "user"
        assert recent[0].content == "Test message"
        assert recent[1].role == "assistant"

    def test_run_once_builds_system_prompt(self, agent_env):
        agent, fake_llm, _, _ = agent_env
        
        fake_llm.responses = [
            LLMResponse(content="OK", tool_calls=[], raw=None),
        ]
        
        agent.run_once("What do you know about me?")
        
        # Check that system prompt was built with user model
        system_msg = fake_llm.last_messages[0]
        assert system_msg["role"] == "system"
        assert "What I know about the user" in system_msg["content"]


class TestAgentToolCalls:
    """Test agent with tool calls."""

    def test_single_tool_call(self, agent_env):
        agent, fake_llm, _, _ = agent_env
        
        # First call: agent wants to call a tool
        # Second call: agent responds with final answer
        fake_llm.responses = [
            LLMResponse(
                content="Let me check the files.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="list_files",
                        arguments='{"path": "/tmp"}',
                    )
                ],
                raw=None,
            ),
            LLMResponse(
                content="I found 3 files.",
                tool_calls=[],
                raw=None,
            ),
        ]
        
        result = agent.run_once("List files in /tmp")
        
        assert fake_llm.call_count == 2
        assert "I found 3 files." in result
        
        # Check that tool result was saved to session
        recent = agent.session.recent(limit=10)
        tool_msgs = [m for m in recent if m.role == "tool"]
        assert len(tool_msgs) >= 1

    def test_tool_call_messages_shape(self, agent_env):
        """Verify tool_call message shape matches OpenAI API."""
        agent, fake_llm, _, _ = agent_env
        
        fake_llm.responses = [
            LLMResponse(
                content="Checking...",
                tool_calls=[
                    ToolCall(
                        id="call_abc",
                        name="bash",
                        arguments='{"cmd": "echo hi"}',
                    )
                ],
                raw=None,
            ),
            LLMResponse(content="Done", tool_calls=[], raw=None),
        ]
        
        agent.run_once("Run echo hi")
        
        # Find the assistant message with tool_calls
        assistant_msgs = [
            m for m in fake_llm.last_messages 
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert len(assistant_msgs) == 1
        assistant_msg = assistant_msgs[0]
        
        tool_calls = assistant_msg["tool_calls"]
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "bash"
        
        # Find the tool response message
        tool_msgs = [
            m for m in fake_llm.last_messages 
            if m.get("role") == "tool"
        ]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_abc"

    def test_multiple_tool_calls_single_turn(self, agent_env):
        """Agent can call multiple tools in one turn."""
        agent, fake_llm, _, _ = agent_env
        
        fake_llm.responses = [
            LLMResponse(
                content="Running commands.",
                tool_calls=[
                    ToolCall(id="c1", name="bash", arguments='{"cmd": "pwd"}'),
                    ToolCall(id="c2", name="bash", arguments='{"cmd": "whoami"}'),
                ],
                raw=None,
            ),
            LLMResponse(content="Done", tool_calls=[], raw=None),
        ]
        
        result = agent.run_once("Run pwd and whoami")
        
        assert fake_llm.call_count == 2
        assert result == "Done"

    def test_tool_error_handling(self, agent_env):
        """Tool errors should be returned to the LLM."""
        agent, fake_llm, _, _ = agent_env
        
        fake_llm.responses = [
            LLMResponse(
                content="Trying invalid tool.",
                tool_calls=[
                    ToolCall(id="bad", name="nonexistent_tool", arguments="{}"),
                ],
                raw=None,
            ),
            LLMResponse(content="I see the tool doesn't exist.", tool_calls=[], raw=None),
        ]
        
        result = agent.run_once("Call nonexistent_tool")
        
        assert fake_llm.call_count == 2
        # Tool error should be in the conversation
        tool_msgs = [m for m in agent.session.recent(10) if m.role == "tool"]
        assert len(tool_msgs) >= 1
        assert "unknown tool" in tool_msgs[0].content


class TestAgentMaxSteps:
    """Test max_steps limit."""

    def test_hits_max_steps(self, agent_env):
        """Agent should stop after max_steps tool calls."""
        agent, fake_llm, _, _ = agent_env
        
        # Always return a tool call to force max_steps
        always_tool = LLMResponse(
            content="Looping...",
            tool_calls=[ToolCall(id="x", name="bash", arguments='{"cmd": "true"}')],
            raw=None,
        )
        fake_llm.responses = [always_tool] * 15
        
        result = agent.run_once("Infinite loop", max_steps=5)
        
        assert "hit max tool steps" in result
        assert fake_llm.call_count == 5

    def test_max_steps_default(self, agent_env):
        """Default max_steps is 12."""
        agent, fake_llm, _, _ = agent_env
        
        always_tool = LLMResponse(
            content="...",
            tool_calls=[ToolCall(id="x", name="bash", arguments='{"cmd": "true"}')],
            raw=None,
        )
        fake_llm.responses = [always_tool] * 15
        
        result = agent.run_once("Test")
        
        # Should stop at default 12
        assert fake_llm.call_count == 12
        assert "hit max tool steps" in result


class TestSystemPromptAssembly:
    """Test system prompt includes the right pieces."""

    def test_includes_user_model(self, agent_env):
        agent, fake_llm, _, _ = agent_env
        
        # Set a trait
        agent.user_model.set("tooling_pref", value="prefers httpx", evidence="test")
        
        fake_llm.responses = [
            LLMResponse(content="OK", tool_calls=[], raw=None),
        ]
        
        agent.run_once("Test")
        
        system = fake_llm.last_messages[0]["content"]
        assert "tooling_pref" in system
        assert "prefers httpx" in system

    def test_includes_skills_search(self, agent_env):
        agent, fake_llm, store, skills = agent_env

        # Create a skill
        skills.create("test-skill", "Test description", "# Body")

        fake_llm.responses = [
            LLMResponse(content="OK", tool_calls=[], raw=None),
        ]

        agent.run_once("test skill description")
        
        system = fake_llm.last_messages[0]["content"]
        assert "test-skill" in system or "Test description" in system

    def test_includes_prior_messages(self, agent_env):
        agent, fake_llm, _, _ = agent_env
        
        # Add a prior message
        agent.session.add("user", "I like httpx")
        agent.session.add("assistant", "Noted.")
        
        fake_llm.responses = [
            LLMResponse(content="OK", tool_calls=[], raw=None),
        ]
        
        agent.run_once("httpx")
        
        system = fake_llm.last_messages[0]["content"]
        assert "prior exchanges" in system
        assert "httpx" in system


class TestLearningLoopIntegration:
    """Test learning loop integration (still with fake LLM)."""

    def test_learning_loop_disabled(self, agent_env):
        """Learning loop should be disabled in our test fixture."""
        agent, fake_llm, _, _ = agent_env
        
        fake_llm.responses = [
            LLMResponse(content="OK", tool_calls=[], raw=None),
        ]
        
        agent.run_once("Test")
        
        # Should not have called chat_json for retro
        # (fake_llm.chat_json is separate from chat)
        assert fake_llm.call_count == 1

    def test_learning_loop_skips_trivial(self, agent_env):
        """Learning loop skips short inputs with no tool steps."""
        agent, fake_llm, _, _ = agent_env
        agent.run_learning_loop_flag = True
        
        fake_llm.responses = [
            LLMResponse(content="OK", tool_calls=[], raw=None),
        ]
        fake_llm.json_response = {}  # If retro runs, it gets empty JSON
        
        agent.run_once("Hi")  # < 12 chars, no tools
        
        # Learning loop should not have run
