"""Execution tools: bash + python sandbox."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from .registry import Tool, ToolContext, ToolRegistry, ToolResult


MAX_OUTPUT = 16_000


def _truncate(s: str) -> str:
    if len(s) <= MAX_OUTPUT:
        return s
    return s[:MAX_OUTPUT] + f"\n... (truncated, total {len(s)} chars)"


def bash(args: dict, ctx: ToolContext) -> ToolResult:
    cmd = args.get("command")
    if not cmd:
        return ToolResult("missing 'command'", is_error=True)
    timeout = int(args.get("timeout", ctx.config.bash_timeout))
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(ctx.config.workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(f"command timed out after {timeout}s", is_error=True)
    body = ""
    if proc.stdout:
        body += "STDOUT:\n" + proc.stdout
    if proc.stderr:
        if body:
            body += "\n"
        body += "STDERR:\n" + proc.stderr
    body += f"\nEXIT: {proc.returncode}"
    return ToolResult(_truncate(body), is_error=proc.returncode != 0)


def exec_python(args: dict, ctx: ToolContext) -> ToolResult:
    code = args.get("code")
    if not code:
        return ToolResult("missing 'code'", is_error=True)
    timeout = int(args.get("timeout", ctx.config.bash_timeout))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        path = fh.name
    try:
        try:
            proc = subprocess.run(
                [sys.executable, path],
                cwd=str(ctx.config.workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(f"python execution timed out after {timeout}s", is_error=True)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    body = ""
    if proc.stdout:
        body += "STDOUT:\n" + proc.stdout
    if proc.stderr:
        if body:
            body += "\n"
        body += "STDERR:\n" + proc.stderr
    body += f"\nEXIT: {proc.returncode}"
    return ToolResult(_truncate(body), is_error=proc.returncode != 0)


def register(reg: ToolRegistry) -> None:
    reg.register(Tool(
        name="bash",
        description="Run a shell command in the agent's working directory. Returns combined stdout/stderr and exit code.",
        schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
        fn=bash,
        category="exec",
    ))
    reg.register(Tool(
        name="exec_python",
        description="Run a Python 3 script as a subprocess. Use for ad-hoc analysis, file munging, or computations.",
        schema={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["code"],
        },
        fn=exec_python,
        category="exec",
    ))
