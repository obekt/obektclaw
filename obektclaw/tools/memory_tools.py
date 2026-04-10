"""Tools the agent calls to read and write its own memory.

The Learning Loop also writes memory after each task — but the agent can write
during a task too, when the user explicitly tells it to remember something
("remember: I always use httpx").
"""
from __future__ import annotations

from .registry import Tool, ToolContext, ToolRegistry, ToolResult


def memory_search(args: dict, ctx: ToolContext) -> ToolResult:
    query = args.get("query", "").strip()
    if not query:
        return ToolResult("missing 'query'", is_error=True)
    msgs = ctx.session.search_history(query, limit=8)
    facts = ctx.persistent.search(query, limit=8)
    out = ["# session memory hits"]
    for m in msgs:
        out.append("- " + m.render())
    out.append("\n# persistent facts")
    for f in facts:
        out.append(f.render())
    return ToolResult("\n".join(out))


def memory_set_fact(args: dict, ctx: ToolContext) -> ToolResult:
    key = args.get("key")
    value = args.get("value")
    category = args.get("category", "general")
    confidence = float(args.get("confidence", 0.85))
    if not key or value is None:
        return ToolResult("need 'key' and 'value'", is_error=True)
    ctx.persistent.upsert(key, str(value), category=category, confidence=confidence)
    return ToolResult(f"saved fact {category}/{key}")


def memory_forget_fact(args: dict, ctx: ToolContext) -> ToolResult:
    key = args.get("key")
    category = args.get("category", "general")
    if not key:
        return ToolResult("missing 'key'", is_error=True)
    ctx.persistent.delete(category, key)
    return ToolResult(f"forgot {category}/{key}")


def user_model_set(args: dict, ctx: ToolContext) -> ToolResult:
    layer = args.get("layer")
    value = args.get("value")
    evidence = args.get("evidence")
    if not layer or not value:
        return ToolResult("need 'layer' and 'value'", is_error=True)
    ctx.user_model.set(layer, value, evidence=evidence)
    return ToolResult(f"updated user_model.{layer}")


def register(reg: ToolRegistry) -> None:
    reg.register(Tool(
        name="memory_search",
        description="Full-text search across past sessions and persistent facts. Use this to recall context from prior conversations.",
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        fn=memory_search,
        category="memory",
    ))
    reg.register(Tool(
        name="memory_set_fact",
        description="Save a durable fact about the user, project, or environment. Use sparingly — only for facts you expect to still be true next week.",
        schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
                "category": {"type": "string", "enum": ["user", "project", "env", "preference", "general"]},
                "confidence": {"type": "number"},
            },
            "required": ["key", "value"],
        },
        fn=memory_set_fact,
        category="memory",
    ))
    reg.register(Tool(
        name="memory_forget_fact",
        description="Delete a previously saved fact (e.g. when the user says it was wrong).",
        schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "category": {"type": "string"},
            },
            "required": ["key"],
        },
        fn=memory_forget_fact,
        category="memory",
    ))
    reg.register(Tool(
        name="user_model_set",
        description="Update one of the 12 user-model layers (technical_level, primary_goals, work_rhythm, comm_style, code_style, tooling_pref, domain_focus, emotional_pattern, trust_boundary, contradictions, knowledge_gaps, long_term_themes).",
        schema={
            "type": "object",
            "properties": {
                "layer": {"type": "string"},
                "value": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["layer", "value"],
        },
        fn=user_model_set,
        category="memory",
    ))
