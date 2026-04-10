import sys
import builtins
from unittest.mock import MagicMock, patch
import pytest
from obektclaw.gateways.cli import run, _first_run_welcome, _show_setup, _check_config


def test_check_config_missing_key():
    with patch("obektclaw.gateways.cli.CONFIG") as mock_config:
        mock_config.llm_api_key = "your-api-key-here"
        assert _check_config() is False


def test_check_config_valid_key():
    with patch("obektclaw.gateways.cli.CONFIG") as mock_config:
        mock_config.llm_api_key = "sk-real-key-abc123"
        assert _check_config() is True


def test_run_exits_on_bad_config():
    with patch("obektclaw.gateways.cli._check_config", return_value=False), \
         patch("obektclaw.gateways.cli._setup_wizard", return_value=None):
        assert run() == 1


def test_first_run_welcome(capsys):
    _first_run_welcome()
    out, err = capsys.readouterr()
    assert "Welcome to obektclaw" in out


def test_show_setup(capsys, tmp_path):
    mock_config = MagicMock()
    mock_config.home = tmp_path
    mock_config.db_path = tmp_path / "db.sqlite"
    mock_config.skills_dir = tmp_path / "skills"
    mock_config.logs_dir = tmp_path / "logs"
    mock_config.tg_token = "some_token"
    mock_config.llm_model = "test-model"
    mock_config.llm_base_url = "https://test.api/v1"

    mcp_config = tmp_path / "mcp.json"

    # Test 1: MCP config exists, tg_token exists
    mcp_config.touch()
    _show_setup(mock_config)
    out, err = capsys.readouterr()
    assert "MCP servers configured" in out
    assert "Telegram bot configured" in out

    # Test 2: MCP config missing, tg_token missing
    mcp_config.unlink()
    mock_config.tg_token = ""
    _show_setup(mock_config)
    out, err = capsys.readouterr()
    assert "MCP servers: not configured" in out
    assert "Telegram bot: not configured" in out


def _mock_session(inputs):
    """Create a mock PromptSession whose .prompt() returns inputs in sequence."""
    session = MagicMock()
    side_effects = []
    for inp in inputs:
        if isinstance(inp, Exception):
            side_effects.append(inp)
        else:
            side_effects.append(inp)
    session.prompt.side_effect = side_effects
    return session


@patch("obektclaw.gateways.cli._check_config", return_value=True)
@patch("obektclaw.gateways.cli.Store")
@patch("obektclaw.gateways.cli.SkillManager")
@patch("obektclaw.gateways.cli.Agent")
@patch("obektclaw.gateways.cli._make_session")
def test_run_first_run(mock_make_session, mock_agent_cls, mock_skill_mgr, mock_store_cls, _check, capsys):
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_store.fetchone.return_value = {"c": 0}  # is_first_run = True

    mock_make_session.return_value = _mock_session(["/exit"])

    run()

    out, err = capsys.readouterr()
    assert "Welcome to obektclaw" in out


@patch("obektclaw.gateways.cli._check_config", return_value=True)
@patch("obektclaw.gateways.cli.Store")
@patch("obektclaw.gateways.cli.SkillManager")
@patch("obektclaw.gateways.cli.Agent")
@patch("obektclaw.gateways.cli._make_session")
def test_run_commands(mock_make_session, mock_agent_cls, mock_skill_mgr, mock_store_cls, _check, capsys):
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_store.fetchone.return_value = {"c": 1}  # is_first_run = False

    mock_skills = MagicMock()
    mock_skill_mgr.return_value = mock_skills

    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent

    mock_make_session.return_value = _mock_session([
        "",             # Empty line
        "/help",
        "/skills",      # No skills
        "/memory",      # No query
        "/memory foo",  # Results
        "/memory bar",  # No results
        "/traits",      # No traits
        "/traits",      # With traits
        "/setup",
        "hello",        # Normal message
        "error",        # Error message
        EOFError()      # EOF
    ])

    # Setup mocks for commands
    mock_skills.list_all.side_effect = [
        [],  # First call /skills
    ]

    mock_agent.persistent.search.side_effect = [
        [MagicMock(render=lambda: "Memory result 1")],  # /memory foo
        []  # /memory bar
    ]

    mock_agent.user_model.all.side_effect = [
        [],  # First call /traits
        [MagicMock(layer="test", value="test_value")]  # Second call /traits
    ]

    mock_agent.run_once.side_effect = [
        "reply text",  # "hello"
        Exception("test error")  # "error"
    ]

    run()

    out, err = capsys.readouterr()
    assert "obektclaw" in out
    assert "self-improving" in out  # help text
    assert "No skills yet" in out
    assert "/memory <query>" in out
    assert "Memory result 1" in out
    assert "No memories" in out
    assert "No user model yet" in out
    assert "test" in out and "test_value" in out
    assert "Setup" in out
    assert "reply text" in out

    assert "test error" in err


@patch("obektclaw.gateways.cli._check_config", return_value=True)
@patch("obektclaw.gateways.cli.Store")
@patch("obektclaw.gateways.cli.SkillManager")
@patch("obektclaw.gateways.cli.Agent")
@patch("obektclaw.gateways.cli._make_session")
def test_run_skills_with_results(mock_make_session, mock_agent_cls, mock_skill_mgr, mock_store_cls, _check, capsys):
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_store.fetchone.return_value = {"c": 1}  # is_first_run = False

    mock_skills = MagicMock()
    mock_skill_mgr.return_value = mock_skills

    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent

    mock_make_session.return_value = _mock_session([
        "/skills",
        "/quit"
    ])

    mock_sk = MagicMock()
    mock_sk.name = "skill-1"
    mock_sk.description = "does a thing"
    mock_skills.list_all.return_value = [mock_sk]

    run()
    out, err = capsys.readouterr()
    assert "skill-1" in out
    assert "does a thing" in out
