import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from obektclaw.agent import Agent
from obektclaw.config import Config
from obektclaw.memory.store import Store
from obektclaw.skills.manager import SkillManager
import json

@pytest.fixture
def mock_config(tmp_path):
    return Config(
        home=tmp_path,
        db_path=tmp_path / "obektclaw.db",
        skills_dir=tmp_path / "skills",
        bundled_skills_dir=tmp_path / "bundled",
        logs_dir=tmp_path / "logs",
        llm_base_url="mock",
        llm_api_key="mock",
        llm_model="mock",
        llm_fast_model="mock",
        tg_token="mock",
        tg_allowed_chat_ids=(),
        bash_timeout=30,
        workdir=tmp_path,
    )

def test_agent_mcp_load_success_and_close(mock_config, monkeypatch, tmp_path):
    mcp_config = mock_config.home / "mcp.json"
    mcp_config.write_text(json.dumps({"mcpServers": {"test": {"command": "echo", "args": ["hello"]}}}))
    
    class MockServer:
        def stop(self):
            pass
    
    def mock_attach(*args, **kwargs):
        return [MockServer()]
    
    monkeypatch.setattr("obektclaw.mcp.attach_mcp_servers", mock_attach)
    
    store = Store(mock_config.db_path)
    skills = SkillManager(store, mock_config.skills_dir, mock_config.bundled_skills_dir)
    agent = Agent(config=mock_config, store=store, skills=skills, load_mcp=True)
    
    # Check if servers were loaded
    assert len(agent._mcp_servers) == 1
    
    # Close it to trigger line 196-199
    agent.close()

def test_agent_mcp_load_exception(mock_config, monkeypatch, tmp_path):
    mcp_config = mock_config.home / "mcp.json"
    mcp_config.write_text('{"mcpServers": {"test": {"command": "echo", "args": ["hello"]}}}')
    
    def mock_attach(*args, **kwargs):
        raise ValueError("attach failed")
    
    monkeypatch.setattr("obektclaw.mcp.attach_mcp_servers", mock_attach)
    
    store = Store(mock_config.db_path)
    skills = SkillManager(store, mock_config.skills_dir, mock_config.bundled_skills_dir)
    # Shouldn't raise
    agent = Agent(config=mock_config, store=store, skills=skills, load_mcp=True)
    assert agent._mcp_servers is None
    msgs = agent.session.recent()
    assert any("MCP failed to load: attach failed" in m.content for m in msgs if m.role == "system")

def test_agent_mcp_close_exception(mock_config, monkeypatch, tmp_path):
    class MockServer:
        def stop(self):
            raise ValueError("stop failed")
            
    store = Store(mock_config.db_path)
    skills = SkillManager(store, mock_config.skills_dir, mock_config.bundled_skills_dir)
    agent = Agent(config=mock_config, store=store, skills=skills, load_mcp=False)
    agent._mcp_servers = [MockServer()]
    
    # Shouldn't raise
    agent.close()

def test_agent_run_once_max_steps(mock_config, monkeypatch):
    store = Store(mock_config.db_path)
    skills = SkillManager(store, mock_config.skills_dir, mock_config.bundled_skills_dir)
    agent = Agent(config=mock_config, store=store, skills=skills, load_mcp=False)
    
    class MockLLM:
        def chat(self, *args, **kwargs):
            from obektclaw.llm import LLMResponse, ToolCall
            # Always return a tool call to prevent break and hit else block
            return LLMResponse(content="hi", tool_calls=[ToolCall("id", "test_tool", "{}")], raw="{}")
            
    agent.llm = MockLLM()
    agent.registry.call = MagicMock(return_value=MagicMock(content="tool result", is_error=False))
    
    res = agent.run_once("hello", max_steps=2)
    assert "(stopped: hit max tool steps)" in res

def test_agent_learning_loop_exception(mock_config, monkeypatch):
    store = Store(mock_config.db_path)
    skills = SkillManager(store, mock_config.skills_dir, mock_config.bundled_skills_dir)
    agent = Agent(config=mock_config, store=store, skills=skills, load_mcp=False)
    agent.run_learning_loop_flag = True
    
    class MockLLM:
        def chat(self, *args, **kwargs):
            from obektclaw.llm import LLMResponse
            return LLMResponse(content="hi", tool_calls=[], raw=None)
            
    agent.llm = MockLLM()
    
    def mock_run(*args, **kwargs):
        raise ValueError("learning loop failed")
        
    import obektclaw.learning
    monkeypatch.setattr(obektclaw.learning.LearningLoop, "run", mock_run)
    
    # Wait for thread to finish
    import threading
    orig_start = threading.Thread.start
    def sync_start(self):
        self.run()
    monkeypatch.setattr(threading.Thread, "start", sync_start)
    
    # Shouldn't raise
    agent.run_once("hello")
    # Check if system message was added
    msgs = agent.session.recent()
    assert any("learning loop failed" in m.content for m in msgs if m.role == "system")

def test_agent_build_messages_tool_role(mock_config):
    store = Store(mock_config.db_path)
    skills = SkillManager(store, mock_config.skills_dir, mock_config.bundled_skills_dir)
    agent = Agent(config=mock_config, store=store, skills=skills, load_mcp=False)
    
    # Add a tool message
    agent.session.add("tool", "tool result", tool_name="test")
    # Add an assistant message to have some valid ones
    agent.session.add("assistant", "hello")
    
    msgs = agent._build_messages("my prompt")
    
    # Ensure tool message is skipped
    roles = [m["role"] for m in msgs]
    assert "tool" not in roles

def test_agent_compose_system_prompt_skills_limit(mock_config, monkeypatch):
    store = Store(mock_config.db_path)
    skills = SkillManager(store, mock_config.skills_dir, mock_config.bundled_skills_dir)
    agent = Agent(config=mock_config, store=store, skills=skills, load_mcp=False)
    
    class MockSkill:
        def __init__(self, name):
            self.name = name
        def render_brief(self):
            return f"- {self.name}"
            
    def mock_list_all():
        return [MockSkill(f"skill_{i}") for i in range(35)]
        
    agent.skills.list_all = mock_list_all
    
    sys_prompt = agent._compose_system_prompt("hello")
    assert "... and 5 more" in sys_prompt

def test_agent_compose_system_prompt_facts(mock_config):
    store = Store(mock_config.db_path)
    skills = SkillManager(store, mock_config.skills_dir, mock_config.bundled_skills_dir)
    agent = Agent(config=mock_config, store=store, skills=skills, load_mcp=False)
    
    agent.persistent.upsert("my_fact", "my_val", category="general")
    sys_prompt = agent._compose_system_prompt("hello")
    assert "Persistent facts" in sys_prompt
    assert "my_fact: my_val" in sys_prompt
