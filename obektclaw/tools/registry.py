"""Tool registry. Each tool exposes a JSON schema (for the LLM) and a Python
callable. Tool functions receive a ToolContext for access to memory/skills/etc.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from ..memory import PersistentMemory, SessionMemory, UserModel
    from ..skills import SkillManager
    from ..llm import LLMClient


@dataclass
class ToolContext:
    config: "Config"
    session: "SessionMemory"
    persistent: "PersistentMemory"
    user_model: "UserModel"
    skills: "SkillManager"
    llm: "LLMClient"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    meta: dict | None = None


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[[dict, ToolContext], ToolResult]
    category: str = "general"
    auto: bool = True  # if False, only loaded on demand

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def auto(self) -> list[Tool]:
        return [t for t in self._tools.values() if t.auto]

    def to_openai_tools(self, only_auto: bool = True) -> list[dict]:
        tools = self.auto() if only_auto else self.all()
        return [t.to_openai() for t in tools]

    def call(self, name: str, args_json: str | dict, ctx: ToolContext) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(f"unknown tool: {name}", is_error=True)
        if isinstance(args_json, str):
            try:
                args = json.loads(args_json) if args_json.strip() else {}
            except json.JSONDecodeError as e:
                return ToolResult(f"invalid JSON args: {e}", is_error=True)
        else:
            args = args_json or {}
        try:
            return tool.fn(args, ctx)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc(limit=4)
            return ToolResult(f"tool {name} crashed: {e}\n{tb}", is_error=True)


def build_default_registry() -> ToolRegistry:
    """Assemble all built-in tools into a single registry."""
    from . import fs, execution as execmod, web, memory_tools, skill_tools, delegate

    reg = ToolRegistry()
    fs.register(reg)
    execmod.register(reg)
    web.register(reg)
    memory_tools.register(reg)
    skill_tools.register(reg)
    delegate.register(reg)
    return reg
