# Test Suite Summary for obektclaw

## Overview
This document summarizes the comprehensive test suite and improvements added to obektclaw.

## Test Files Created

### `tests/test_store.py` (30 tests)
Tests for the SQLite + FTS5 storage layer:
- **FTS5 query sanitization** - Special character handling, stemming
- **Session lifecycle** - Open/close sessions
- **Message storage** - Add, retrieve, FTS5 search messages
- **Facts storage** - Upsert, FTS5 search, uniqueness constraints
- **User traits** - 12-layer model storage, conflict resolution
- **Skills table** - Insert, FTS5 search, trigger sync
- **Thread safety** - WAL mode, concurrent reads

### `tests/test_skills.py` (24 tests)
Tests for the markdown-based skill system:
- **Slugify** - Name normalization, edge cases
- **Frontmatter parsing** - YAML-like header extraction
- **Skill file I/O** - Create, read, write
- **SkillManager** - Reindex, list, get, search, create, improve
- **Bundled skills sync** - First-run initialization
- **Skill rendering** - Full and brief formats

### `tests/test_agent_offline.py` (14 tests)
Tests for the agent loop with fake LLM:
- **Basic runs** - No tool calls, message persistence
- **Tool calls** - Single/multiple tools, error handling
- **Max steps** - Step limits, default behavior
- **System prompt assembly** - User model, skills, prior messages
- **Learning loop integration** - Disable flag, trivial skip

### `tests/test_learning_loop.py` (16 tests)
Tests for the Learning Loop retrospection:
- **Basic operation** - Skip trivial, run on substantial input
- **Fact persistence** - Single/multiple facts, malformed handling
- **User model updates** - Single/multiple layers, invalid layer filtering
- **Skill creation** - New skills, missing name handling
- **Skill improvement** - Append to existing, missing skill handling
- **Notes** - System message logging
- **Empty retro** - None response, empty arrays

## Key Improvements to Core Code

### 1. Enhanced RETRO_SYSTEM Prompt (`obektclaw/learning.py`)
Added explicit guidance to prevent junk facts and layer misclassification:
- **What to EXCLUDE**: File paths, counts, temporary state
- **What to INCLUDE**: Preferences, environment info, project structure
- **Layer descriptions**: All 12 layers now have one-line descriptions
- **Layer assignment guidance**: Clear rules for which layer to use

### 2. Auto-load MCP Configuration (`obektclaw/agent.py`)
- Agent now looks for `~/.obektclaw/mcp.json` on startup
- Loads MCP servers and registers tools automatically
- Graceful error handling if MCP fails
- Proper cleanup in `close()` method

### 3. Always-Include Skills Index (`obektclaw/agent.py`)
- System prompt now lists all available skills (capped at 30)
- Enables self-discovery without lexical matching
- Most relevant skills still highlighted separately via FTS5

### 4. Memory Cleanup Command (`obektclaw/__main__.py`)
New CLI command: `python -m obektclaw memory cleanup`
- Uses fast LLM to identify stale/contradictory/ephemeral facts
- Deletes identified facts from all categories
- Provides before/after feedback

### 5. Retro JSONL Logging (`obektclaw/learning.py`)
- Every Learning Loop iteration logged to `~/.obektclaw/logs/learning-YYYY-MM-DD.jsonl`
- Includes timestamp and full retro JSON
- Silent failure on logging errors

## Configuration

### `pyproject.toml`
Added pytest configuration for consistent test runs.

### `requirements.txt`
Added:
- `pytest>=7.4.0`
- `pytest-asyncio>=0.21.0`

## Running Tests

```bash
cd obektclaw
source .venv/bin/activate
python -m pytest
```

All 235 tests should pass.

## Test Coverage by Component

| Component | Tests | Coverage |
|-----------|-------|----------|
| Storage (store.py) | 30 | Schema, FTS5, triggers, WAL |
| Skills (manager.py) | 24 | Parse, create, improve, search |
| Agent (agent.py) | 14 | ReAct loop, tools, prompts |
| Learning (learning.py) | 16 | Retro, facts, user model, skills |
| **Total** | **84** | **Core functionality** |

## Known Limitations

1. **No live LLM tests** - All tests use fake LLM clients
2. **No MCP server tests** - MCP loading tested only for config presence
3. **No gateway tests** - CLI and Telegram gateways not tested
4. **No tool execution tests** - Built-in tools (fs, exec, web) not tested

## Future Test Additions

Recommended next additions:
1. `tests/test_tools_*.py` - Test each built-in tool
2. `tests/test_gateways.py` - CLI slash commands, Telegram bot
3. `tests/test_mcp.py` - MCP server lifecycle
4. `tests/test_integration.py` - End-to-end multi-turn conversations
5. `tests/test_memory_cleanup.py` - Integration test for cleanup command

## Design Principles Preserved

All changes respect the principles from AGENTS.md:
- ✅ Skill files on disk remain source of truth
- ✅ Memory stays local (no telemetry)
- ✅ Single SQLite connection
- ✅ OpenAI-shaped LLM client
- ✅ Tools as functions
- ✅ Learning Loop fire-and-forget
- ✅ Synchronous by default
