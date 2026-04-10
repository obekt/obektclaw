"""Filesystem tools: read, write, list, glob, grep."""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from .registry import Tool, ToolContext, ToolRegistry, ToolResult


MAX_READ_BYTES = 200_000


def _resolve(ctx: ToolContext, path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ctx.config.workdir / p
    return p


def read_file(args: dict, ctx: ToolContext) -> ToolResult:
    path = args.get("path")
    if not path:
        return ToolResult("missing 'path'", is_error=True)
    p = _resolve(ctx, path)
    if not p.exists():
        return ToolResult(f"no such file: {p}", is_error=True)
    if p.is_dir():
        return ToolResult(f"is a directory: {p}", is_error=True)
    try:
        data = p.read_bytes()
    except OSError as e:
        return ToolResult(f"read error: {e}", is_error=True)
    if len(data) > MAX_READ_BYTES:
        return ToolResult(
            f"file too large ({len(data)} bytes); read at most {MAX_READ_BYTES}",
            is_error=True,
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")
    return ToolResult(text)


def write_file(args: dict, ctx: ToolContext) -> ToolResult:
    path = args.get("path")
    content = args.get("content")
    if not path or content is None:
        return ToolResult("need 'path' and 'content'", is_error=True)
    text = str(content)
    p = _resolve(ctx, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(text)
    except OSError as e:
        return ToolResult(f"write error: {e}", is_error=True)
    return ToolResult(f"wrote {len(text)} chars to {p}")


def list_files(args: dict, ctx: ToolContext) -> ToolResult:
    path = args.get("path", ".")
    pattern = args.get("pattern", "*")
    p = _resolve(ctx, path)
    if not p.exists() or not p.is_dir():
        return ToolResult(f"not a directory: {p}", is_error=True)
    out = []
    for entry in sorted(p.iterdir()):
        if not fnmatch.fnmatch(entry.name, pattern):
            continue
        kind = "d" if entry.is_dir() else "f"
        try:
            size = entry.stat().st_size if entry.is_file() else "-"
        except OSError:
            size = "?"
        out.append(f"{kind} {size:>8} {entry.name}")
    return ToolResult("\n".join(out) or "(empty)")


def grep(args: dict, ctx: ToolContext) -> ToolResult:
    pattern = args.get("pattern")
    path = args.get("path", ".")
    glob_pat = args.get("glob", "*")
    if not pattern:
        return ToolResult("missing 'pattern'", is_error=True)
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return ToolResult(f"bad regex: {e}", is_error=True)
    root = _resolve(ctx, path)
    if not root.exists():
        return ToolResult(f"no such path: {root}", is_error=True)
    matches: list[str] = []
    targets: list[Path] = [root] if root.is_file() else []
    if root.is_dir():
        for dirpath, _, filenames in os.walk(root):
            # skip dotfolders for sanity
            if any(seg.startswith(".") and seg not in (".", "..") for seg in Path(dirpath).relative_to(root).parts):
                continue
            for fn in filenames:
                if fnmatch.fnmatch(fn, glob_pat):
                    targets.append(Path(dirpath) / fn)
    for target in targets[:1000]:
        try:
            with target.open("r", encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if rx.search(line):
                        rel = target.relative_to(root) if target != root else target.name
                        matches.append(f"{rel}:{i}: {line.rstrip()}")
                        if len(matches) >= 200:
                            break
        except OSError:
            continue
        if len(matches) >= 200:
            matches.append("(truncated)")
            break
    return ToolResult("\n".join(matches) or "(no matches)")


def register(reg: ToolRegistry) -> None:
    reg.register(Tool(
        name="read_file",
        description="Read a UTF-8 text file from disk. Path may be absolute or relative to the agent's workdir.",
        schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        fn=read_file,
        category="fs",
    ))
    reg.register(Tool(
        name="write_file",
        description="Write a string to a file, creating parent directories as needed. Overwrites existing files.",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        fn=write_file,
        category="fs",
    ))
    reg.register(Tool(
        name="list_files",
        description="List entries in a directory. Optional fnmatch-style 'pattern'.",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "pattern": {"type": "string"},
            },
        },
        fn=list_files,
        category="fs",
    ))
    reg.register(Tool(
        name="grep",
        description="Recursively search file contents for a regex. Returns at most 200 matching lines.",
        schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string", "description": "fnmatch glob filtering filenames, e.g. '*.py'"},
            },
            "required": ["pattern"],
        },
        fn=grep,
        category="fs",
    ))
