"""Tools the agent uses to read, search, create, and improve its own skills."""
from __future__ import annotations

from .registry import Tool, ToolContext, ToolRegistry, ToolResult


def skill_search(args: dict, ctx: ToolContext) -> ToolResult:
    query = args.get("query", "").strip()
    if not query:
        return ToolResult("missing 'query'", is_error=True)
    hits = ctx.skills.search(query, limit=5)
    if not hits:
        return ToolResult("(no matching skills)")
    return ToolResult("\n".join(s.render_brief() for s in hits))


def skill_load(args: dict, ctx: ToolContext) -> ToolResult:
    name = args.get("name")
    if not name:
        return ToolResult("missing 'name'", is_error=True)
    sk = ctx.skills.get(name)
    if sk is None:
        return ToolResult(f"no skill named {name}", is_error=True)
    return ToolResult(sk.render())


def skill_create(args: dict, ctx: ToolContext) -> ToolResult:
    name = args.get("name")
    description = args.get("description")
    body = args.get("body")
    if not name or not description or not body:
        return ToolResult("need 'name', 'description', and 'body'", is_error=True)
    sk = ctx.skills.create(name, description, body)
    return ToolResult(f"created skill: {sk.name}")


def skill_improve(args: dict, ctx: ToolContext) -> ToolResult:
    name = args.get("name")
    if not name:
        return ToolResult("missing 'name'", is_error=True)
    sk = ctx.skills.improve(
        name,
        new_description=args.get("description"),
        new_body=args.get("body"),
        append=args.get("append"),
    )
    if sk is None:
        return ToolResult(f"no skill named {name}", is_error=True)
    return ToolResult(f"updated skill: {sk.name}")


def register(reg: ToolRegistry) -> None:
    reg.register(Tool(
        name="skill_search",
        description="Search the skill library for relevant procedural knowledge. Use this BEFORE attempting a non-trivial task to see if you've solved something similar before.",
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        fn=skill_search,
        category="skill",
    ))
    reg.register(Tool(
        name="skill_load",
        description="Load the full body of a skill by name.",
        schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        fn=skill_load,
        category="skill",
    ))
    reg.register(Tool(
        name="skill_create",
        description="Create a new skill (procedural memory) for a kind of task you expect to encounter again. Body should be markdown with concrete steps and gotchas.",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["name", "description", "body"],
        },
        fn=skill_create,
        category="skill",
    ))
    reg.register(Tool(
        name="skill_improve",
        description="Edit an existing skill — replace the body, replace the description, or append a new note.",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "body": {"type": "string"},
                "append": {"type": "string"},
            },
            "required": ["name"],
        },
        fn=skill_improve,
        category="skill",
    ))
