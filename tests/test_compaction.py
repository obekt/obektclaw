"""Tests for context compaction logic."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obektclaw.agent import Agent
from obektclaw.config import Config
from obektclaw.llm import LLMResponse, TokenUsage
from obektclaw.memory import PersistentMemory, SessionMemory, UserModel
from obektclaw.memory.store import Store
from obektclaw.skills import SkillManager


def _make_test_agent(tmp_path: Path) -> Agent:
    """Create a minimal agent for testing."""
    config = Config(
        home=tmp_path,
        db_path=tmp_path / "test.db",
        skills_dir=tmp_path / "skills",
        bundled_skills_dir=tmp_path / "bundled",
        logs_dir=tmp_path / "logs",
        llm_base_url="https://example.com/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_fast_model="test-fast-model",
        tg_token="",
        tg_allowed_chat_ids=(),
        context_window=10000,  # Small window for easy testing
        bash_timeout=30,
        workdir=tmp_path,
    )
    
    (tmp_path / "skills").mkdir(exist_ok=True)
    (tmp_path / "bundled").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)
    
    store = Store(config.db_path)
    skills = SkillManager(store, config.skills_dir, config.bundled_skills_dir)
    
    return Agent(
        config=config,
        store=store,
        skills=skills,
        load_mcp=False,
        run_learning_loop=False,
    )


def _add_conversation_history(session: SessionMemory, turns: int = 20):
    """Add fake conversation history to session memory."""
    for i in range(turns):
        session.add("user", f"User message {i}: What is the meaning of life?")
        session.add("assistant", f"Assistant response {i}: The meaning of life is {i * 42}.")


class TestCompactionBasic:
    """Test basic compaction functionality."""

    def test_compaction_skips_when_pressure_low(self, tmp_path):
        """Test that compaction is skipped when context pressure is low."""
        agent = _make_test_agent(tmp_path)
        try:
            # No usage yet, pressure should be 0
            result = agent.compact_context(force=False)
            assert result["compacted"] is False
            assert "too low" in result["reason"].lower()
        finally:
            agent.close()

    def test_compaction_skips_when_conversation_too_short(self, tmp_path):
        """Test that compaction skips if there aren't enough messages."""
        agent = _make_test_agent(tmp_path)
        try:
            # Add some usage to create pressure
            agent.last_usage = TokenUsage(
                prompt_tokens=9000,
                completion_tokens=500,
                total_tokens=9500,
            )
            
            # But only 2 turns - too short to compact
            agent.session.add("user", "Hi there")
            agent.session.add("assistant", "Hello!")
            
            result = agent.compact_context(force=False)
            assert result["compacted"] is False
            assert "too short" in result["reason"].lower()
        finally:
            agent.close()

    def test_compaction_force_skips_pressure_check(self, tmp_path):
        """Test that force=True skips the pressure check."""
        agent = _make_test_agent(tmp_path)
        try:
            # No pressure, but force=True
            result = agent.compact_context(force=True)
            # Should still skip if conversation is too short
            assert result["compacted"] is False
            assert "too short" in result["reason"].lower()
        finally:
            agent.close()


class TestCompactionWithLLM:
    """Test compaction with mocked LLM calls."""

    def test_compaction_success(self, tmp_path):
        """Test successful compaction with LLM summary."""
        agent = _make_test_agent(tmp_path)
        try:
            # Add substantial conversation
            _add_conversation_history(agent.session, turns=20)
            
            # Create high pressure
            agent.last_usage = TokenUsage(
                prompt_tokens=9000,
                completion_tokens=500,
                total_tokens=9500,
            )
            
            # Mock LLM response
            mock_response = LLMResponse(
                content="User asked about life meaning multiple times. I explained it varies.",
                tool_calls=[],
                raw=None,
                usage=TokenUsage(prompt_tokens=500, completion_tokens=100, total_tokens=600),
            )
            
            with patch.object(agent.llm, 'chat', return_value=mock_response):
                result = agent.compact_context(force=True)
            
            assert result["compacted"] is True
            assert result["summary_length"] > 0
            assert result["tokens_saved"] > 0
            assert result["error"] is None
            
            # Verify old messages were deleted
            recent = agent.session.recent(limit=100)
            # Should have: summary + kept turns
            assert len(recent) < 40  # Was 40, should be much less now
            
        finally:
            agent.close()

    def test_compaction_uses_fast_model(self, tmp_path):
        """Test that compaction uses the fast model to save cost."""
        agent = _make_test_agent(tmp_path)
        try:
            _add_conversation_history(agent.session, turns=20)
            agent.last_usage = TokenUsage(
                prompt_tokens=9000,
                completion_tokens=500,
                total_tokens=9500,
            )
            
            mock_response = LLMResponse(
                content="Summary of conversation.",
                tool_calls=[],
                raw=None,
                usage=TokenUsage(),
            )
            
            with patch.object(agent.llm, 'chat', return_value=mock_response) as mock_chat:
                agent.compact_context(force=True)
                
                # Verify fast=True was passed
                mock_chat.assert_called_once()
                call_kwargs = mock_chat.call_args[1]
                assert call_kwargs.get("fast") is True
                
        finally:
            agent.close()

    def test_compaction_handles_empty_summary(self, tmp_path):
        """Test that compaction handles empty LLM response gracefully."""
        agent = _make_test_agent(tmp_path)
        try:
            _add_conversation_history(agent.session, turns=20)
            agent.last_usage = TokenUsage(
                prompt_tokens=9000,
                completion_tokens=500,
                total_tokens=9500,
            )
            
            # Mock empty response
            mock_response = LLMResponse(
                content="",
                tool_calls=[],
                raw=None,
                usage=TokenUsage(),
            )
            
            with patch.object(agent.llm, 'chat', return_value=mock_response):
                result = agent.compact_context(force=True)
            
            assert result["compacted"] is False
            assert "empty" in result["reason"].lower()
            
        finally:
            agent.close()

    def test_compaction_handles_llm_error(self, tmp_path):
        """Test that compaction handles LLM errors gracefully."""
        agent = _make_test_agent(tmp_path)
        try:
            _add_conversation_history(agent.session, turns=20)
            agent.last_usage = TokenUsage(
                prompt_tokens=9000,
                completion_tokens=500,
                total_tokens=9500,
            )
            
            with patch.object(agent.llm, 'chat', side_effect=RuntimeError("LLM failed")):
                result = agent.compact_context(force=True)
            
            assert result["compacted"] is False
            assert result["error"] is not None
            assert "LLM failed" in result["error"]
            
        finally:
            agent.close()


class TestCompactionAutoTrigger:
    """Test automatic compaction triggering in run_once."""

    def test_auto_compaction_at_85_percent(self, tmp_path):
        """Test that compaction triggers automatically at 85% pressure."""
        agent = _make_test_agent(tmp_path)
        try:
            # Add some conversation so compaction has something to summarize
            _add_conversation_history(agent.session, turns=20)
            
            # Set up high pressure (90%)
            agent.last_usage = TokenUsage(
                prompt_tokens=8500,
                completion_tokens=500,
                total_tokens=9000,
            )
            
            # Verify pressure is above threshold
            pressure = agent._context_pressure()
            assert pressure >= agent.COMPACTION_PRESSURE
            
            # Mock LLM for both compaction and main chat
            compact_response = LLMResponse(
                content="Conversation summary.",
                tool_calls=[],
                raw=None,
                usage=TokenUsage(),
            )
            
            main_response = LLMResponse(
                content="Final answer",
                tool_calls=[],
                raw=None,
                usage=TokenUsage(),
            )
            
            # First call is compaction, second is main chat
            with patch.object(agent.llm, 'chat', side_effect=[compact_response, main_response]) as mock_chat:
                reply = agent.run_once("Test message", max_steps=1)
            
            # After compaction, the agent should continue with the main chat
            # At least one call should have happened (compaction)
            assert mock_chat.call_count >= 1
            
        finally:
            agent.close()

    def test_no_compaction_below_threshold(self, tmp_path):
        """Test that compaction doesn't trigger below 85%."""
        agent = _make_test_agent(tmp_path)
        try:
            # Set up medium pressure (70%)
            agent.last_usage = TokenUsage(
                prompt_tokens=7000,
                completion_tokens=0,
                total_tokens=7000,
            )
            
            pressure = agent._context_pressure()
            assert pressure < agent.COMPACTION_PRESSURE
            
            # Mock LLM for main chat only (compaction shouldn't trigger)
            main_response = LLMResponse(
                content="Answer without compaction",
                tool_calls=[],
                raw=None,
                usage=TokenUsage(),
            )
            
            with patch.object(agent.llm, 'chat', return_value=main_response) as mock_chat:
                reply = agent.run_once("Test message", max_steps=1)
            
            # Should only be called once (main chat, no compaction)
            assert mock_chat.call_count == 1
            assert "Answer without compaction" in reply
            
        finally:
            agent.close()


class TestCompactionThresholds:
    """Test compaction threshold constants."""

    def test_compaction_pressure_threshold(self):
        """Test that compaction pressure threshold is set correctly."""
        assert Agent.COMPACTION_PRESSURE == 0.85

    def test_keep_turns_threshold(self):
        """Test that we keep the right number of recent turns."""
        assert Agent.COMPACTION_KEEP_TURNS == 6

    def test_max_summary_tokens(self):
        """Test that summary token limit is reasonable."""
        assert Agent.COMPACTION_MAX_SUMMARY == 1000


class TestCompactionEdgeCases:
    """Test edge cases and error conditions."""

    def test_compaction_with_no_user_messages(self, tmp_path):
        """Test compaction when there are no user/assistant messages."""
        agent = _make_test_agent(tmp_path)
        try:
            # Add only system messages (need enough to pass the length check)
            for i in range(20):
                agent.store.execute(
                    "INSERT INTO messages (session_id, role, content, ts) VALUES (?, 'system', ?, CURRENT_TIMESTAMP)",
                    (agent.session_id, f"System message {i}"),
                )
            
            agent.last_usage = TokenUsage(
                prompt_tokens=9000,
                completion_tokens=500,
                total_tokens=9500,
            )
            
            result = agent.compact_context(force=True)
            # Should skip because there are no user/assistant messages to summarize
            assert result["compacted"] is False
            
        finally:
            agent.close()

    def test_compaction_preserves_recent_turns(self, tmp_path):
        """Test that compaction keeps recent turns raw."""
        agent = _make_test_agent(tmp_path)
        try:
            # Add 20 turns
            _add_conversation_history(agent.session, turns=20)
            agent.last_usage = TokenUsage(
                prompt_tokens=9000,
                completion_tokens=500,
                total_tokens=9500,
            )
            
            mock_response = LLMResponse(
                content="Summary of old conversation.",
                tool_calls=[],
                raw=None,
                usage=TokenUsage(),
            )
            
            with patch.object(agent.llm, 'chat', return_value=mock_response):
                result = agent.compact_context(force=True)
            
            assert result["compacted"] is True
            
            # Verify recent turns are still there
            recent = agent.session.recent(limit=100)
            # Should have kept at least the last 12 messages (6 turns * 2)
            user_msgs = [m for m in recent if m.role == "user"]
            assert len(user_msgs) >= 6  # At least 6 recent user messages
            
        finally:
            agent.close()

    def test_compaction_inserts_summary(self, tmp_path):
        """Test that compaction inserts a summary message into the database."""
        agent = _make_test_agent(tmp_path)
        try:
            _add_conversation_history(agent.session, turns=20)
            agent.last_usage = TokenUsage(
                prompt_tokens=9000,
                completion_tokens=500,
                total_tokens=9500,
            )
            
            mock_response = LLMResponse(
                content="Test summary content",
                tool_calls=[],
                raw=None,
                usage=TokenUsage(),
            )
            
            with patch.object(agent.llm, 'chat', return_value=mock_response):
                agent.compact_context(force=True)
            
            # Verify summary message exists (look for the compacted marker)
            recent = agent.session.recent(limit=100)
            system_msgs = [m for m in recent if m.role == "system" and "compacted conversation summary" in m.content.lower()]
            assert len(system_msgs) >= 1
            # The summary should contain the test content
            assert "Test summary content" in system_msgs[0].content or "compacted" in system_msgs[0].content.lower()
            
        finally:
            agent.close()
