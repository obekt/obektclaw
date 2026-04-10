import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from obektclaw.mcp import MCPServerSpec, MCPServer, load_mcp_config, attach_mcp_servers
from obektclaw.tools.registry import ToolRegistry

@pytest.fixture
def mock_subprocess():
    with patch("obektclaw.mcp.subprocess.Popen") as mock_popen:
        yield mock_popen

def test_mcp_server_spec():
    spec = MCPServerSpec(name="test", command=["echo"], env={"A": "B"})
    assert spec.name == "test"
    assert spec.command == ["echo"]
    assert spec.env == {"A": "B"}

def test_mcp_server_start(mock_subprocess):
    mock_proc = MagicMock()
    mock_subprocess.return_value = mock_proc
    
    # Mock readline for initialize response
    mock_proc.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}) + "\n"
    ]
    
    spec = MCPServerSpec(name="test", command=["test_cmd"])
    server = MCPServer(spec)
    
    server.start()
    
    assert server._initialized is True
    mock_subprocess.assert_called_once()
    assert mock_proc.stdin.write.call_count == 2 # initialize and notifications/initialized
    
    # Starting again should do nothing
    server.start()
    assert mock_subprocess.call_count == 1

def test_mcp_server_start_notify_exception(mock_subprocess):
    mock_proc = MagicMock()
    mock_subprocess.return_value = mock_proc
    
    def side_effect(req_json):
        if "notifications/initialized" in req_json:
            raise RuntimeError("Notify failed")
    
    def rpc_read():
        return json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}) + "\n"

    mock_proc.stdout.readline.side_effect = [rpc_read()]
    
    # Setup mock to raise on notify
    mock_proc.stdin.write.side_effect = side_effect
    
    spec = MCPServerSpec(name="test", command=["test_cmd"])
    server = MCPServer(spec)
    
    # Should not raise exception
    server.start()
    assert server._initialized is True

def test_mcp_server_stop(mock_subprocess):
    mock_proc = MagicMock()
    mock_subprocess.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n"
    ]
    
    server = MCPServer(MCPServerSpec(name="test", command=["test_cmd"]))
    server.start()
    
    server.stop()
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=3)
    assert server._proc is None
    
    # Stopping again should do nothing
    server.stop()
    assert mock_proc.terminate.call_count == 1

def test_mcp_server_stop_fallback_kill(mock_subprocess):
    mock_proc = MagicMock()
    mock_subprocess.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n"
    ]
    
    server = MCPServer(MCPServerSpec(name="test", command=["test_cmd"]))
    server.start()
    
    mock_proc.terminate.side_effect = Exception("Terminate failed")
    
    server.stop()
    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()
    assert server._proc is None

def test_mcp_server_stop_fallback_kill_exception(mock_subprocess):
    mock_proc = MagicMock()
    mock_subprocess.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n"
    ]
    
    server = MCPServer(MCPServerSpec(name="test", command=["test_cmd"]))
    server.start()
    
    mock_proc.terminate.side_effect = Exception("Terminate failed")
    mock_proc.kill.side_effect = Exception("Kill failed")
    
    # Should not raise
    server.stop()
    assert server._proc is None

def test_mcp_rpc_not_started():
    server = MCPServer(MCPServerSpec(name="test", command=["test_cmd"]))
    with pytest.raises(RuntimeError, match="MCP server not started"):
        server._rpc("test", {})

def test_mcp_notify_not_started():
    server = MCPServer(MCPServerSpec(name="test", command=["test_cmd"]))
    # Should not raise
    server._notify("test", {})

def test_mcp_rpc_readline_eof(mock_subprocess):
    mock_proc = MagicMock()
    mock_subprocess.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n", # for initialize
        "" # EOF
    ]
    
    server = MCPServer(MCPServerSpec(name="test", command=["test_cmd"]))
    server.start()
    
    with pytest.raises(RuntimeError, match="closed unexpectedly"):
        server._rpc("test_method", {})

def test_mcp_rpc_skip_decode_error(mock_subprocess):
    mock_proc = MagicMock()
    mock_subprocess.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n", # for initialize
        "not json\n",
        json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"success": True}}) + "\n"
    ]
    
    server = MCPServer(MCPServerSpec(name="test", command=["test_cmd"]))
    server.start()
    
    res = server._rpc("test_method", {})
    assert res == {"success": True}

def test_mcp_rpc_skip_stray_notifications(mock_subprocess):
    mock_proc = MagicMock()
    mock_subprocess.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n", # for initialize
        json.dumps({"jsonrpc": "2.0", "method": "notification"}) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": 999, "result": {}}) + "\n", # wrong id
        json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"success": True}}) + "\n"
    ]
    
    server = MCPServer(MCPServerSpec(name="test", command=["test_cmd"]))
    server.start()
    
    res = server._rpc("test_method", {})
    assert res == {"success": True}

def test_mcp_rpc_error(mock_subprocess):
    mock_proc = MagicMock()
    mock_subprocess.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n", # for initialize
        json.dumps({"jsonrpc": "2.0", "id": 2, "error": "Something went wrong"}) + "\n"
    ]
    
    server = MCPServer(MCPServerSpec(name="test", command=["test_cmd"]))
    server.start()
    
    with pytest.raises(RuntimeError, match="MCP error: Something went wrong"):
        server._rpc("test_method", {})

def test_mcp_list_tools(mock_subprocess):
    mock_proc = MagicMock()
    mock_subprocess.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n", # for initialize
        json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "tool1"}]}}) + "\n"
    ]
    
    server = MCPServer(MCPServerSpec(name="test", command=["test_cmd"]))
    server.start()
    
    tools = server.list_tools()
    assert tools == [{"name": "tool1"}]

def test_mcp_call_tool(mock_subprocess):
    mock_proc = MagicMock()
    mock_subprocess.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n", # for initialize
        json.dumps({"jsonrpc": "2.0", "id": 2, "result": {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "data": "b64"}
            ],
            "isError": False
        }}) + "\n"
    ]
    
    server = MCPServer(MCPServerSpec(name="test", command=["test_cmd"]))
    server.start()
    
    res = server.call_tool("tool1", {})
    assert "Hello" in res
    assert '{"type": "image", "data": "b64"}' in res

def test_mcp_call_tool_error(mock_subprocess):
    mock_proc = MagicMock()
    mock_subprocess.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n", # for initialize
        json.dumps({"jsonrpc": "2.0", "id": 2, "result": {
            "content": [{"type": "text", "text": "Failed!"}],
            "isError": True
        }}) + "\n"
    ]
    
    server = MCPServer(MCPServerSpec(name="test", command=["test_cmd"]))
    server.start()
    
    res = server.call_tool("tool1", {})
    assert "MCP tool error:" in res
    assert "Failed!" in res

def test_load_mcp_config(tmp_path):
    conf_path = tmp_path / "mcp_servers.json"
    
    # not exists
    assert load_mcp_config(conf_path) == []
    
    conf_path.write_text(json.dumps({
        "mcpServers": {
            "srv1": {"command": "echo", "args": ["1"]},
            "srv2": {"command": ["echo", "2"], "env": {"A": "B"}}
        }
    }))
    
    specs = load_mcp_config(conf_path)
    assert len(specs) == 2
    assert specs[0].name == "srv1"
    assert specs[0].command == ["echo", "1"]
    assert specs[1].name == "srv2"
    assert specs[1].command == ["echo", "2"]
    assert specs[1].env == {"A": "B"}
    
    # None config
    conf_path.write_text("{}")
    assert load_mcp_config(conf_path) == []

def test_attach_mcp_servers(mock_subprocess):
    registry = ToolRegistry()
    specs = [MCPServerSpec(name="test1", command=["cmd1"]), MCPServerSpec(name="test2", command=["cmd2"])]
    
    mock_proc = MagicMock()
    mock_subprocess.return_value = mock_proc
    
    # We will be called twice for start, once for each spec
    # spec1: init, list_tools
    # spec2: init (fail)
    
    call_idx = 0
    def readline_side_effect():
        nonlocal call_idx
        responses = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n", # srv1 init
            json.dumps({"jsonrpc": "2.0", "id": 2, "result": {            # srv1 tools/list
                "tools": [
                    {"name": "t1", "description": "desc1"},
                    {} # empty name
                ]
            }}) + "\n",
            "" # srv2 fails to read init
        ]
        if call_idx < len(responses):
            res = responses[call_idx]
            call_idx += 1
            return res
        return ""
        
    mock_proc.stdout.readline.side_effect = readline_side_effect
    
    servers = attach_mcp_servers(registry, specs)
    
    assert len(servers) == 1
    
    tools = registry.all()
    assert len(tools) == 1
    assert tools[0].name == "mcp__test1__t1"
    assert tools[0].description == "desc1"
    
    # Test tool invocation
    mock_proc.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "tool output"}]}}) + "\n"
    ]
    res = tools[0].fn({}, MagicMock())
    assert res.content == "tool output"
    
    # Test tool invocation exception
    mock_proc.stdout.readline.side_effect = [
        "not json",
        ""
    ]
    res = tools[0].fn({}, MagicMock())
    assert res.is_error is True
    assert "mcp call failed" in res.content
