"""Sub-agent delegation. Spawns a transient agent in the same process with its
own scratch session, runs it to completion, and returns the final answer.

Hermes-style: lets the parent stay focused while the child handles a tightly
scoped subtask without polluting the parent's context.
"""
from __future__ import annotations

from .registry import Tool, ToolContext, ToolRegistry, ToolResult


def delegate(args: dict, ctx: ToolContext) -> ToolResult:
    task = args.get("task")
    if not task:
        return ToolResult("missing 'task'", is_error=True)
    max_steps = int(args.get("max_steps", 8))

    # Lazy import to avoid a hard cycle (agent imports tools).
    from ..agent import Agent

    sub = Agent(
        config=ctx.config,
        store=ctx.session.store,
        skills=ctx.skills,
        registry=_clone_registry_minus_delegate(),
        llm=ctx.llm,
        gateway="delegate",
        user_key=f"sub-of-{ctx.session.session_id}",
        run_learning_loop=False,
    )
    answer = sub.run_once(task, max_steps=max_steps)
    sub.close()
    return ToolResult(f"sub-agent answer:\n{answer}")


def _clone_registry_minus_delegate():
    from .registry import build_default_registry
    reg = build_default_registry()
    # Block recursive delegation to avoid runaway fan-out.
    if "delegate" in reg._tools:  # noqa: SLF001
        del reg._tools["delegate"]
    return reg


def register(reg: ToolRegistry) -> None:
    reg.register(Tool(
        name="delegate",
        description="Hand off a tightly scoped subtask to a fresh sub-agent. Returns the sub-agent's final answer. Use for things like 'summarize this file', 'find all functions matching X', 'try this approach in isolation' — anything where you want to protect the main conversation from clutter.",
        schema={
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "max_steps": {"type": "integer"},
            },
            "required": ["task"],
        },
        fn=delegate,
        category="orchestration",
    ))
