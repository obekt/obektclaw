import pytest
from unittest.mock import MagicMock
from obektclaw.tools.registry import ToolContext, ToolRegistry
from obektclaw.tools.memory_tools import (
    memory_search,
    memory_set_fact,
    memory_forget_fact,
    user_model_set,
    register
)

@pytest.fixture
def mock_ctx():
    ctx = MagicMock(spec=ToolContext)
    ctx.session = MagicMock()
    ctx.persistent = MagicMock()
    ctx.user_model = MagicMock()
    return ctx

def test_memory_search_missing_query(mock_ctx):
    res = memory_search({}, mock_ctx)
    assert res.is_error
    assert "missing 'query'" in res.content

def test_memory_search_success(mock_ctx):
    msg1 = MagicMock()
    msg1.render.return_value = "session msg 1"
    mock_ctx.session.search_history.return_value = [msg1]
    
    fact1 = MagicMock()
    fact1.render.return_value = "persistent fact 1"
    mock_ctx.persistent.search.return_value = [fact1]
    
    res = memory_search({"query": "test"}, mock_ctx)
    assert not res.is_error
    assert "session memory hits" in res.content
    assert "- session msg 1" in res.content
    assert "persistent facts" in res.content
    assert "persistent fact 1" in res.content
    mock_ctx.session.search_history.assert_called_once_with("test", limit=8)
    mock_ctx.persistent.search.assert_called_once_with("test", limit=8)

def test_memory_set_fact_missing_args(mock_ctx):
    res = memory_set_fact({"key": "foo"}, mock_ctx)
    assert res.is_error
    assert "need 'key' and 'value'" in res.content

def test_memory_set_fact_success(mock_ctx):
    res = memory_set_fact({"key": "foo", "value": "bar", "category": "user", "confidence": "0.9"}, mock_ctx)
    assert not res.is_error
    assert "saved fact user/foo" in res.content
    mock_ctx.persistent.upsert.assert_called_once_with("foo", "bar", category="user", confidence=0.9)

def test_memory_forget_fact_missing_key(mock_ctx):
    res = memory_forget_fact({}, mock_ctx)
    assert res.is_error
    assert "missing 'key'" in res.content

def test_memory_forget_fact_success(mock_ctx):
    res = memory_forget_fact({"key": "foo", "category": "project"}, mock_ctx)
    assert not res.is_error
    assert "forgot project/foo" in res.content
    mock_ctx.persistent.delete.assert_called_once_with("project", "foo")

def test_user_model_set_missing_args(mock_ctx):
    res = user_model_set({"layer": "tech"}, mock_ctx)
    assert res.is_error
    assert "need 'layer' and 'value'" in res.content

def test_user_model_set_success(mock_ctx):
    res = user_model_set({"layer": "tech", "value": "high", "evidence": "wrote code"}, mock_ctx)
    assert not res.is_error
    assert "updated user_model.tech" in res.content
    mock_ctx.user_model.set.assert_called_once_with("tech", "high", evidence="wrote code")

def test_register():
    reg = MagicMock(spec=ToolRegistry)
    register(reg)
    assert reg.register.call_count == 4