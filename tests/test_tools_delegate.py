import pytest
from unittest.mock import MagicMock, patch
from obektclaw.tools.registry import ToolContext, ToolRegistry
from obektclaw.tools.delegate import delegate, _get_registry_no_delegate, register, _REGISTRY_NO_DELEGATE

@pytest.fixture
def mock_ctx():
    ctx = MagicMock(spec=ToolContext)
    ctx.config = MagicMock()
    ctx.session = MagicMock()
    ctx.session.store = MagicMock()
    ctx.session.session_id = "test-session"
    ctx.skills = MagicMock()
    ctx.llm = MagicMock()
    return ctx

def test_delegate_missing_task(mock_ctx):
    res = delegate({}, mock_ctx)
    assert res.is_error
    assert "missing 'task'" in res.content

@patch("obektclaw.agent.Agent")
@patch("obektclaw.tools.delegate._get_registry_no_delegate")
def test_delegate_success(mock_get_reg, mock_agent_class, mock_ctx):
    mock_agent_inst = MagicMock()
    mock_agent_class.return_value = mock_agent_inst
    mock_agent_inst.run_once.return_value = "task done"

    res = delegate({"task": "do it", "max_steps": "5"}, mock_ctx)

    assert not res.is_error
    assert "sub-agent answer:\ntask done" in res.content

    mock_agent_class.assert_called_once_with(
        config=mock_ctx.config,
        store=mock_ctx.session.store,
        skills=mock_ctx.skills,
        registry=mock_get_reg.return_value,
        llm=mock_ctx.llm,
        gateway="delegate",
        user_key="sub-of-test-session",
        run_learning_loop=False,
    )
    mock_agent_inst.run_once.assert_called_once_with("do it", max_steps=5)
    mock_agent_inst.close.assert_called_once()

@patch("obektclaw.tools.registry.build_default_registry")
def test_get_registry_no_delegate(mock_build):
    # Reset cache before test
    import obektclaw.tools.delegate as delegate_mod
    delegate_mod._REGISTRY_NO_DELEGATE = None

    mock_reg = MagicMock()
    mock_reg._tools = {"delegate": MagicMock(), "other": MagicMock()}
    mock_build.return_value = mock_reg

    reg = _get_registry_no_delegate()
    assert "delegate" not in reg._tools
    assert "other" in reg._tools


@patch("obektclaw.tools.registry.build_default_registry")
def test_get_registry_no_delegate_no_delegate(mock_build):
    # Reset cache before test
    import obektclaw.tools.delegate as delegate_mod
    delegate_mod._REGISTRY_NO_DELEGATE = None

    mock_reg = MagicMock()
    mock_reg._tools = {"other": MagicMock()}
    mock_build.return_value = mock_reg

    reg = _get_registry_no_delegate()
    assert "delegate" not in reg._tools
    assert "other" in reg._tools


@patch("obektclaw.tools.registry.build_default_registry")
def test_get_registry_caches(mock_build):
    # Reset cache before test
    import obektclaw.tools.delegate as delegate_mod
    delegate_mod._REGISTRY_NO_DELEGATE = None

    mock_reg = MagicMock()
    mock_reg._tools = {"other": MagicMock()}
    mock_build.return_value = mock_reg

    # Call twice
    reg1 = _get_registry_no_delegate()
    reg2 = _get_registry_no_delegate()

    # Should only build once
    assert mock_build.call_count == 1
    assert reg1 is reg2  # Same object returned

def test_register():
    reg = MagicMock(spec=ToolRegistry)
    register(reg)
    assert reg.register.call_count == 1