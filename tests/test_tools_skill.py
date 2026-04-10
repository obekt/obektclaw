import pytest
from unittest.mock import MagicMock
from obektclaw.tools.registry import ToolContext, ToolRegistry
from obektclaw.tools.skill_tools import (
    skill_search,
    skill_load,
    skill_create,
    skill_improve,
    register
)

@pytest.fixture
def mock_ctx():
    ctx = MagicMock(spec=ToolContext)
    ctx.skills = MagicMock()
    return ctx

def test_skill_search_missing_query(mock_ctx):
    res = skill_search({}, mock_ctx)
    assert res.is_error
    assert "missing 'query'" in res.content

def test_skill_search_no_hits(mock_ctx):
    mock_ctx.skills.search.return_value = []
    res = skill_search({"query": "foo"}, mock_ctx)
    assert not res.is_error
    assert "(no matching skills)" in res.content

def test_skill_search_success(mock_ctx):
    s1 = MagicMock()
    s1.render_brief.return_value = "skill1 brief"
    mock_ctx.skills.search.return_value = [s1]
    res = skill_search({"query": "foo"}, mock_ctx)
    assert not res.is_error
    assert "skill1 brief" in res.content
    mock_ctx.skills.search.assert_called_once_with("foo", limit=5)

def test_skill_load_missing_name(mock_ctx):
    res = skill_load({}, mock_ctx)
    assert res.is_error
    assert "missing 'name'" in res.content

def test_skill_load_not_found(mock_ctx):
    mock_ctx.skills.get.return_value = None
    res = skill_load({"name": "foo"}, mock_ctx)
    assert res.is_error
    assert "no skill named foo" in res.content

def test_skill_load_success(mock_ctx):
    sk = MagicMock()
    sk.render.return_value = "skill content"
    mock_ctx.skills.get.return_value = sk
    res = skill_load({"name": "foo"}, mock_ctx)
    assert not res.is_error
    assert "skill content" in res.content
    mock_ctx.skills.get.assert_called_once_with("foo")

def test_skill_create_missing_args(mock_ctx):
    res = skill_create({"name": "foo", "description": "bar"}, mock_ctx)
    assert res.is_error
    assert "need 'name', 'description', and 'body'" in res.content

def test_skill_create_success(mock_ctx):
    sk = MagicMock()
    sk.name = "foo"
    mock_ctx.skills.create.return_value = sk
    res = skill_create({"name": "foo", "description": "bar", "body": "baz"}, mock_ctx)
    assert not res.is_error
    assert "created skill: foo" in res.content
    mock_ctx.skills.create.assert_called_once_with("foo", "bar", "baz")

def test_skill_improve_missing_name(mock_ctx):
    res = skill_improve({}, mock_ctx)
    assert res.is_error
    assert "missing 'name'" in res.content

def test_skill_improve_not_found(mock_ctx):
    mock_ctx.skills.improve.return_value = None
    res = skill_improve({"name": "foo"}, mock_ctx)
    assert res.is_error
    assert "no skill named foo" in res.content

def test_skill_improve_success(mock_ctx):
    sk = MagicMock()
    sk.name = "foo"
    mock_ctx.skills.improve.return_value = sk
    res = skill_improve({"name": "foo", "description": "new", "body": "new body", "append": "app"}, mock_ctx)
    assert not res.is_error
    assert "updated skill: foo" in res.content
    mock_ctx.skills.improve.assert_called_once_with("foo", new_description="new", new_body="new body", append="app")

def test_register():
    reg = MagicMock(spec=ToolRegistry)
    register(reg)
    assert reg.register.call_count == 4