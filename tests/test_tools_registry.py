import json
import pytest
from unittest.mock import MagicMock

from obektclaw.tools.registry import (
    ToolContext, ToolResult, Tool, ToolRegistry, build_default_registry
)

def test_tool_context():
    ctx = ToolContext(
        config=MagicMock(),
        session=MagicMock(),
        persistent=MagicMock(),
        user_model=MagicMock(),
        skills=MagicMock(),
        llm=MagicMock()
    )
    assert ctx.extras == {}

def test_tool_result():
    res = ToolResult(content="hello")
    assert res.content == "hello"
    assert not res.is_error
    assert res.meta is None

def test_tool():
    def dummy_fn(args, ctx):
        return ToolResult("ok")
    t = Tool(name="my_tool", description="desc", schema={"type": "object"}, fn=dummy_fn)
    assert t.to_openai() == {
        "type": "function",
        "function": {
            "name": "my_tool",
            "description": "desc",
            "parameters": {"type": "object"}
        }
    }

def test_registry():
    reg = ToolRegistry()
    
    def dummy_fn(args, ctx):
        return ToolResult(f"args: {args}")
    
    t1 = Tool(name="t1", description="", schema={}, fn=dummy_fn)
    t2 = Tool(name="t2", description="", schema={}, fn=dummy_fn, auto=False)
    
    reg.register(t1)
    reg.register(t2)
    
    assert reg.get("t1") == t1
    assert reg.get("t3") is None
    
    assert len(reg.all()) == 2
    assert len(reg.auto()) == 1
    
    tools_auto = reg.to_openai_tools(only_auto=True)
    assert len(tools_auto) == 1
    assert tools_auto[0]["function"]["name"] == "t1"
    
    tools_all = reg.to_openai_tools(only_auto=False)
    assert len(tools_all) == 2

def test_registry_call():
    reg = ToolRegistry()
    
    def fail_fn(args, ctx):
        raise ValueError("crash")
    
    def echo_fn(args, ctx):
        return ToolResult(f"args: {args}")
    
    reg.register(Tool(name="echo", description="", schema={}, fn=echo_fn))
    reg.register(Tool(name="fail", description="", schema={}, fn=fail_fn))
    
    ctx = MagicMock()
    
    # Unknown tool
    res = reg.call("unknown", {}, ctx)
    assert res.is_error
    assert "unknown tool" in res.content
    
    # Valid call with dict args
    res = reg.call("echo", {"a": 1}, ctx)
    assert not res.is_error
    assert "args: {'a': 1}" in res.content
    
    # Valid call with string args JSON
    res = reg.call("echo", '{"a": 1}', ctx)
    assert not res.is_error
    assert "args: {'a': 1}" in res.content
    
    # Valid call with empty string args JSON
    res = reg.call("echo", '  ', ctx)
    assert not res.is_error
    assert "args: {}" in res.content

    # Valid call with None args
    res = reg.call("echo", None, ctx)
    assert not res.is_error
    assert "args: {}" in res.content
    
    # Invalid JSON
    res = reg.call("echo", '{"a": 1', ctx)
    assert res.is_error
    assert "invalid JSON" in res.content
    
    # Crash
    res = reg.call("fail", {}, ctx)
    assert res.is_error
    assert "tool fail crashed" in res.content

def test_build_default_registry():
    reg = build_default_registry()
    assert isinstance(reg, ToolRegistry)
    assert reg.get("read_file") is not None
    assert reg.get("bash") is not None
    assert reg.get("web_fetch") is not None
