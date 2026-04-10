# AI Developer Notes

This document is for AI coding agents (Claude, Gemini, Cursor, etc.) continuing work on obektclaw.

## Project Identity

- **Name:** obektclaw
- **Concept:** Self-improving AI agent based on Nous Research Hermes Agent
- **Size:** ~2,900 lines of Python
- **Tests:** 300+ tests (all offline with fake LLM)

## Quick Context

obektclaw implements a complete agent harness that improves itself after every turn:

1. **User sends message** → Agent processes with ReAct loop
2. **Agent responds** → May use tools (file ops, bash, web fetch, memory, skills)
3. **Learning Loop runs** → Fast LLM extracts facts, user model updates, skills
4. **Memory persists** → SQLite + FTS5, survives restarts
5. **Skills evolve** → Markdown files auto-create and improve

## Architecture (One Paragraph)

Synchronous ReAct loop (`obektclaw/agent.py`) builds system prompt from user model (12 layers) + persistent facts + all skills (capped @ 30) + FTS5-recalled relevant skills + FTS5-recalled message history. Calls LLM with 16 built-in tools. Executes tool calls, feeds results back, loops until done. Learning Loop (`obektclaw/learning.py`) runs post-turn retrospection via fast LLM, emits structured JSON, applies updates to memory/skills, logs to JSONL.

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `obektclaw/agent.py` | ReAct loop, prompt assembly | ~250 |
| `obektclaw/learning.py` | Learning Loop, retro prompt | ~185 |
| `obektclaw/memory/store.py` | SQLite + FTS5 wrapper | ~320 |
| `obektclaw/memory/session.py` | Layer 1: conversation history | ~50 |
| `obektclaw/memory/persistent.py` | Layer 2: semantic facts | ~80 |
| `obektclaw/memory/user_model.py` | Layer 3: 12-layer identity | ~70 |
| `obektclaw/skills/manager.py` | Markdown skill system | ~250 |
| `obektclaw/tools/registry.py` | Tool registration, execution | ~100 |
| `obektclaw/tools/*.py` | Built-in tools (fs, exec, web, etc.) | ~500 |
| `obektclaw/mcp.py` | MCP stdio client | ~180 |
| `obektclaw/gateways/cli.py` | CLI REPL | ~220 |
| `obektclaw/gateways/telegram.py` | Telegram bot | ~150 |
| `tests/test_*.py` | Test suite | ~1,200 |

## Design Principles

1. **Skill files on disk = source of truth** (not SQLite)
2. **Memory stays local** (no telemetry, no phone home)
3. **One SQLite connection, one file** (no vector DB dependency)
4. **OpenAI-shaped LLM client** (vendor agnostic)
5. **Tools as functions** (minimal abstraction)
6. **Learning Loop is fire-and-forget** (never break user turn)
7. **Synchronous by default** (simpler reasoning)

## Common Tasks

### Add a Tool

1. Create `obektclaw/tools/mytool.py`
2. Write function:
   ```python
   def my_fn(args: dict, ctx: ToolContext) -> ToolResult:
       # args is parsed JSON from LLM
       # ctx has config, session, persistent, user_model, skills, llm
       return ToolResult("result text")
   ```
3. Register in `obektclaw/tools/registry.py::build_default_registry()`

### Add a Gateway

1. Create `obektclaw/gateways/mygateway.py`
2. Instantiate `Agent` per user/session
3. Call `agent.run_once(user_input)` for each message

### Modify Learning Loop

Edit `obektclaw/learning.py::RETRO_SYSTEM` prompt. Test with:
```python
from obektclaw.learning import LearningLoop, RETRO_SYSTEM
print(RETRO_SYSTEM)  # Verify prompt changes
```

### Change User Model Layers

Edit `obektclaw/memory/user_model.py::LAYERS` tuple. Update `RETRO_SYSTEM` in `learning.py` to match.

## Testing

```bash
# Run all tests
python -m pytest

# Run specific test
python -m pytest tests/test_agent_offline.py -v

# Add test
# Create tests/test_myfeature.py with FakeLLMClient pattern
```

## Known Issues (from AGENTS.md §10) — All Resolved

1. **Learning Loop over-eager** — ✅ Fixed: exclusion examples in `RETRO_SYSTEM`
2. **User model misclassification** — ✅ Fixed: layer descriptions in retro prompt
3. **No MCP auto-load** — ✅ Fixed in `agent.py::Agent.__init__`
4. **Skills not always in context** — ✅ Fixed: lists all skills in system prompt
5. **No memory expiration** — ✅ Fixed: `python -m obektclaw memory cleanup`
6. **`tools/exec.py` shadows stdlib** — ✅ Fixed: renamed to `tools/execution.py`
7. **No tests** — ✅ Fixed: 300+ tests
8. **Tools run with full privileges** — By design (personal VPS deployment)
9. **Telegram no graceful shutdown** — ✅ Fixed: sentinel-aware worker, 2s timeout
10. **System prompt never says "be brief"** — ✅ Fixed: "Be concise" instruction added
11. **No retro inspection** — ✅ Fixed: JSONL logging to `logs/`
12. **Delegate builds new registry each time** — ✅ Fixed: module-level cache

## Configuration

All via `.env` or environment variables:

```bash
OBEKTCLAW_HOME=~/.obektclaw          # Data directory
OBEKTCLAW_LLM_BASE_URL=...        # LLM endpoint
OBEKTCLAW_LLM_API_KEY=...         # API key
OBEKTCLAW_LLM_MODEL=...           # Main model
OBEKTCLAW_LLM_FAST_MODEL=...      # Learning Loop model
OBEKTCLAW_TG_TOKEN=...            # Telegram bot token
```

## Storage Layout

```
~/.obektclaw/
├── obektclaw.db          # SQLite (sessions, messages, facts, traits, skills)
├── skills/            # Markdown files (source of truth)
├── logs/              # Learning Loop JSONL
└── mcp.json           # Optional MCP config
```

## MCP Integration

Config format (Claude-Desktop compatible):
```json
{
  "mcpServers": {
    "fs": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

Auto-loaded in `Agent.__init__`. Tools registered as `mcp__fs__read_file`.

## FTS5 Query Sanitization

```python
def _fts_query(q: str) -> str:
    cleaned = []
    for tok in q.replace('"', " ").split():
        tok = "".join(ch for ch in tok if ch.isalnum() or ch in "_-")
        if tok:
            cleaned.append(tok + "*")
    return " OR ".join(cleaned) if cleaned else '""'
```

Prevents `OperationalError: malformed MATCH expression` on user input like `test's file: /tmp`.

## Skill Frontmatter Parsing

```python
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
```

Fallback: filename = name, first line = description.

## Learning Loop Retro JSON

```json
{
  "facts": [{"category": "...", "key": "...", "value": "...", "confidence": 0.9}],
  "user_model_updates": [{"layer": "...", "value": "...", "evidence": "..."}],
  "new_skill": {"name": "...", "description": "...", "body": "..."} | null,
  "skill_improvement": {"name": "...", "append": "..."} | null,
  "notes": "..."
}
```

## Release Checklist

Before committing:
1. No `.env` files (only `.env.example`)
2. No API keys in code
3. No user data (`*.db`, `logs/`, `*.jsonl`)
4. All tests pass
5. Docs reference "obektclaw" not "obektclaw-mini"
6. `.gitignore` is comprehensive

See `RELEASE.md` for full checklist.

## When Stuck

1. Read `AGENTS.md` — Full architecture and known issues
2. Read `docs/ARCHITECTURE.md` — System design
3. Read `docs/NOVELTY.md` — Why this approach
4. Run `python -m obektclaw memory status` — Check memory health
5. Check `~/.obektclaw/logs/learning-*.jsonl` — Learning Loop debug

## First Task for New AI Agent

1. Read this file end-to-end
2. Read `AGENTS.md` §1-10 (architecture + known issues)
3. Run `python -m pytest` — Verify tests pass
4. Pick a TODO from `docs/NOVELTY.md` §"Future Research Directions"
5. Implement + test + document

---

**Remember:** The harness is what matters. Keep it minimal, keep it readable, keep it self-improving.
