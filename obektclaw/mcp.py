"""Minimal MCP (Model Context Protocol) stdio client.

Speaks JSON-RPC 2.0 over a child process's stdin/stdout. Just enough to:
  - launch an MCP server (`mcp_servers.json`-style config)
  - call `initialize` and `tools/list`
  - register each remote tool into our local ToolRegistry
  - relay `tools/call` requests when the agent invokes them

We keep this dependency-free on purpose. A full MCP client would handle
resources, prompts, sampling, etc.; this only covers tools — which is what
the Hermes orange book emphasizes ("connect everything").
"""
from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .tools.registry import Tool, ToolContext, ToolRegistry, ToolResult


@dataclass
class MCPServerSpec:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)


class MCPServer:
    def __init__(self, spec: MCPServerSpec):
        self.spec = spec
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next_id = 1
        self._initialized = False

    def start(self) -> None:
        if self._proc is not None:
            return
        import os
        env = os.environ.copy()
        env.update(self.spec.env)
        self._proc = subprocess.Popen(
            self.spec.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._rpc("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "obektclaw", "version": "0.1"},
        })
        try:
            self._notify("notifications/initialized", {})
        except Exception:
            pass
        self._initialized = True

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=3)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    # ----- JSON-RPC plumbing -----
    def _rpc(self, method: str, params: dict) -> dict:
        if self._proc is None:
            raise RuntimeError("MCP server not started")
        with self._lock:
            req_id = self._next_id
            self._next_id += 1
            req = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
            assert self._proc.stdin is not None and self._proc.stdout is not None
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"MCP server {self.spec.name} closed unexpectedly")
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") != req_id:
                    continue  # skip stray notifications
                if "error" in msg:
                    raise RuntimeError(f"MCP error: {msg['error']}")
                return msg.get("result", {})

    def _notify(self, method: str, params: dict) -> None:
        if self._proc is None:
            return
        with self._lock:
            req = {"jsonrpc": "2.0", "method": method, "params": params}
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()

    # ----- public API -----
    def list_tools(self) -> list[dict]:
        result = self._rpc("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        # mcp tools/call result has a "content" array of blocks
        chunks: list[str] = []
        for block in result.get("content", []):
            if block.get("type") == "text":
                chunks.append(block.get("text", ""))
            else:
                chunks.append(json.dumps(block))
        if result.get("isError"):
            return "MCP tool error:\n" + "\n".join(chunks)
        return "\n".join(chunks)


def load_mcp_config(path: Path) -> list[MCPServerSpec]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out: list[MCPServerSpec] = []
    for name, entry in (data.get("mcpServers") or {}).items():
        cmd = entry.get("command")
        args = entry.get("args", [])
        if isinstance(cmd, str):
            cmd = [cmd]
        out.append(MCPServerSpec(name=name, command=list(cmd or []) + list(args), env=entry.get("env", {})))
    return out


def attach_mcp_servers(registry: ToolRegistry, specs: list[MCPServerSpec]) -> list[MCPServer]:
    """Start each server and register its tools into our registry."""
    servers: list[MCPServer] = []
    for spec in specs:
        srv = MCPServer(spec)
        try:
            srv.start()
            tools = srv.list_tools()
        except Exception as e:  # noqa: BLE001
            print(f"[mcp] failed to start {spec.name}: {e}")
            continue
        servers.append(srv)
        for t in tools:
            tname = t.get("name")
            if not tname:
                continue
            full_name = f"mcp__{spec.name}__{tname}"
            schema = t.get("inputSchema") or {"type": "object", "properties": {}}
            description = t.get("description", "") or f"MCP tool {tname} from {spec.name}"

            def _make(srv=srv, tname=tname):
                def _call(args: dict, ctx: ToolContext) -> ToolResult:
                    try:
                        text = srv.call_tool(tname, args or {})
                    except Exception as e:  # noqa: BLE001
                        return ToolResult(f"mcp call failed: {e}", is_error=True)
                    return ToolResult(text)
                return _call

            registry.register(Tool(
                name=full_name,
                description=description,
                schema=schema,
                fn=_make(),
                category="mcp",
            ))
    return servers
