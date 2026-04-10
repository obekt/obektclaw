import sys
import pytest
from unittest.mock import MagicMock, patch
from obektclaw.__main__ import main, _open
from obektclaw.config import Config

def test_open():
    with patch("obektclaw.__main__.Store") as mock_store, patch("obektclaw.__main__.SkillManager") as mock_skills:
        s, sk = _open()
        assert s is not None
        assert sk is not None

def test_main_help(capsys):
    assert main([]) == 0
    out, _ = capsys.readouterr()
    assert "usage:" in out

    assert main(["-h"]) == 0
    assert main(["--help"]) == 0
    assert main(["help"]) == 0

def test_main_unknown(capsys):
    assert main(["unknown"]) == 1
    out, _ = capsys.readouterr()
    assert "unknown command: unknown" in out

def test_main_setup(capsys, tmp_path):
    mcp_config = tmp_path / "mcp.json"
    
    # Test without mcp or tg
    dummy_config = Config(
        home=tmp_path,
        db_path=tmp_path / "db.sqlite",
        skills_dir=tmp_path / "skills",
        logs_dir=tmp_path / "logs",
        tg_token="",
        tg_allowed_chat_ids=None,
        llm_base_url="url",
        llm_api_key="key",
        llm_model="model",
        llm_fast_model="fast_model",
        bundled_skills_dir=tmp_path / "bundled",
        bash_timeout=10,
        workdir=tmp_path
    )
    
    with patch("obektclaw.__main__.CONFIG", dummy_config):
        assert main(["setup"]) == 0
        out, _ = capsys.readouterr()
        assert "obektclaw Setup" in out
        assert "○ MCP servers: not configured" in out
        assert "○ Telegram: not configured" in out
        
    # Test with mcp and tg
    mcp_config.touch()
    dummy_config2 = Config(
        home=tmp_path,
        db_path=tmp_path / "db.sqlite",
        skills_dir=tmp_path / "skills",
        logs_dir=tmp_path / "logs",
        tg_token="token",
        tg_allowed_chat_ids=None,
        llm_base_url="url",
        llm_api_key="key",
        llm_model="model",
        llm_fast_model="fast_model",
        bundled_skills_dir=tmp_path / "bundled",
        bash_timeout=10,
        workdir=tmp_path
    )
    with patch("obektclaw.__main__.CONFIG", dummy_config2):
        assert main(["setup"]) == 0
        out, _ = capsys.readouterr()
        assert "✓ MCP servers: configured" in out
        assert "✓ Telegram: configured" in out

@patch("obektclaw.__main__._open")
def test_main_skill(mock_open, capsys):
    mock_store = MagicMock()
    mock_skills = MagicMock()
    mock_open.return_value = (mock_store, mock_skills)
    
    assert main(["skill"]) == 1
    out, _ = capsys.readouterr()
    assert "usage: skill list | skill show <name>" in out
    
    mock_sk = MagicMock()
    mock_sk.render_brief.return_value = "Skill Brief"
    mock_skills.list_all.return_value = [mock_sk]
    
    assert main(["skill", "list"]) == 0
    out, _ = capsys.readouterr()
    assert "Skill Brief" in out
    
    assert main(["skill", "show"]) == 1
    
    mock_skills.get.return_value = None
    assert main(["skill", "show", "unknown"]) == 1
    
    mock_sk.render.return_value = "Skill Render"
    mock_skills.get.return_value = mock_sk
    assert main(["skill", "show", "known"]) == 0
    out, _ = capsys.readouterr()
    assert "Skill Render" in out
    
    assert main(["skill", "unknown"]) == 1

@patch("obektclaw.__main__._open")
@patch("obektclaw.memory.UserModel")
def test_main_traits(mock_user_model, mock_open, capsys):
    mock_store = MagicMock()
    mock_skills = MagicMock()
    mock_open.return_value = (mock_store, mock_skills)
    
    mock_user_model.return_value.render_for_prompt.return_value = "Traits Prompt"
    assert main(["traits"]) == 0
    out, _ = capsys.readouterr()
    assert "Traits Prompt" in out

@patch("obektclaw.__main__._open")
@patch("obektclaw.memory.PersistentMemory")
@patch("obektclaw.llm.LLMClient")
def test_main_memory(mock_llm_cls, mock_pm_cls, mock_open, capsys, tmp_path):
    mock_store = MagicMock()
    mock_skills = MagicMock()
    mock_open.return_value = (mock_store, mock_skills)
    
    dummy_config = Config(
        home=tmp_path,
        db_path=tmp_path / "db.sqlite",
        skills_dir=tmp_path / "skills",
        logs_dir=tmp_path / "logs",
        tg_token="token",
        tg_allowed_chat_ids=None,
        llm_base_url="url",
        llm_api_key="key",
        llm_model="model",
        llm_fast_model="fast_model",
        bundled_skills_dir=tmp_path / "bundled",
        bash_timeout=10,
        workdir=tmp_path
    )
    with patch("obektclaw.__main__.CONFIG", dummy_config):
        assert main(["memory"]) == 1
        out, _ = capsys.readouterr()
        assert "usage:" in out
        
        assert main(["memory", "unknown"]) == 1
        
        # recent
        mock_store.fetchone.return_value = None
        assert main(["memory", "recent"]) == 0
        out, _ = capsys.readouterr()
        assert "(no sessions)" in out
        
        mock_store.fetchone.return_value = {"id": 1}
        mock_store.recent_messages.return_value = [{"role": "user", "content": "hello"}]
        assert main(["memory", "recent"]) == 0
        out, _ = capsys.readouterr()
        assert "[user] hello" in out
        
        # search
        assert main(["memory", "search"]) == 1
        mock_pm = MagicMock()
        mock_pm_cls.return_value = mock_pm
        mock_fact = MagicMock()
        mock_fact.render.return_value = "Fact 1"
        mock_pm.search.return_value = [mock_fact]
        mock_store.fts_messages.return_value = [{"role": "user", "content": "msg 1"}]
        assert main(["memory", "search", "query"]) == 0
        out, _ = capsys.readouterr()
        assert "Fact 1" in out
        assert "[user] msg 1" in out

        # cleanup
        mock_pm.list_category.return_value = []
        assert main(["memory", "cleanup"]) == 0
        out, _ = capsys.readouterr()
        assert "(no facts to clean up)" in out
        
        mock_fact.key = "f1"
        mock_pm.list_category.side_effect = lambda cat, limit: [mock_fact] if cat == "user" else []
        
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        
        mock_llm.chat_json.return_value = None
        assert main(["memory", "cleanup"]) == 1
        
        mock_llm.chat_json.return_value = []
        assert main(["memory", "cleanup"]) == 0
        
        mock_llm.chat_json.return_value = ["f1"]
        mock_pm.list_category.side_effect = lambda cat, limit: [mock_fact] if cat == "user" else []
        assert main(["memory", "cleanup"]) == 0
        out, _ = capsys.readouterr()
        assert "Deleted: (user) f1" in out

@patch("obektclaw.__main__._open")
def test_main_memory_status(mock_open, capsys):
    mock_store = MagicMock()
    mock_skills = MagicMock()
    mock_open.return_value = (mock_store, mock_skills)
    
    mock_store.fetchone.side_effect = [
        {"c": 1}, # sessions
        {"c": 2}, # facts
        {"c": 3}, # traits
        {"c": 4}, # messages
        {"c": 5}, # skills
        {"ts": "2023-01-01"}, # last_session
        {"journal_mode": "wal"} # wal
    ]
    
    mock_store.fts_messages.side_effect = Exception("fts msg error")
    mock_store.fts_facts.side_effect = Exception("fts facts error")
    mock_store.fts_skills.side_effect = Exception("fts skills error")
    
    assert main(["memory", "status"]) == 0
    out, _ = capsys.readouterr()
    assert "Memory System Status" in out
    assert "Sessions:    1" in out
    assert "Facts:       2" in out
    assert "Traits:      3" in out
    assert "Messages:    4" in out
    assert "Skills:      5" in out
    assert "✗ FTS5 messages index: fts msg error" in out
    assert "✗ FTS5 facts index: fts facts error" in out
    assert "✗ FTS5 skills index: fts skills error" in out
    assert "Last session:  2023-01-01" in out
    assert "Journal mode: wal" in out
    assert "Memory system is healthy!" in out
    
    # Test FTS5 success
    mock_store.fetchone.side_effect = [
        {"c": -1}, # sessions (to trigger warning)
        {"c": -1}, # facts
        {"c": -1}, # traits
        {"c": -1}, # messages
        {"c": -1}, # skills
        None, # last_session
        {"journal_mode": "delete"} # wal
    ]
    mock_store.fts_messages.side_effect = None
    mock_store.fts_facts.side_effect = None
    mock_store.fts_skills.side_effect = None
    
    assert main(["memory", "status"]) == 0
    out, _ = capsys.readouterr()
    assert "✓ FTS5 messages index: OK" in out
    assert "✓ FTS5 facts index: OK" in out
    assert "✓ FTS5 skills index: OK" in out
    assert "Warning: Check errors above" in out

@patch("obektclaw.gateways.cli.run")
def test_main_chat(mock_run):
    mock_run.return_value = 0
    assert main(["chat"]) == 0

@patch("obektclaw.gateways.telegram.run")
def test_main_tg(mock_run):
    mock_run.return_value = 0
    assert main(["tg"]) == 0

def test_main_dunder():
    # Test if __name__ == "__main__" block
    with patch("obektclaw.__main__.main") as mock_main:
        mock_main.return_value = 0
        
        with patch("sys.argv", ["__main__.py", "help"]):
            with pytest.raises(SystemExit) as e:
                import runpy
                runpy.run_module("obektclaw.__main__", run_name="__main__")
            assert e.value.code == 0
