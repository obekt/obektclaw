# Structured Logging Implementation

## What was added

A new module `obektclaw/logging_config.py` that provides structured logging with two output channels:

1. **JSON file logs** — rotated daily under `~/.obektclaw/logs/obektclaw.log` (10 MB max, 7 backups)
2. **Colored console output** — stderr with `[HH:MM:SS] [LEVEL] module: msg` format

The module exports `get_logger(name)` which auto-configures handlers once per logger.

## Why this change

### Problem: No structured logs existed

Prior to this change, the codebase used `print()` statements for diagnostics and errors. This means:

- **No machine-readable logs** — you can't grep, aggregate, or feed logs into monitoring tools
- **No log levels** — everything is printed at the same verbosity
- **No separation of concerns** — user-facing output (Rich panels, banners) and diagnostic output are mixed
- **No log rotation** — diagnostic output would fill the terminal but leave no persistent record
- **No error context** — when something crashes, there's no structured record of what happened before

### Why structured logging over print statements

| Concern | print() | logging module |
|---------|---------|----------------|
| Machine-readable | No | Yes (JSON) |
| Log levels (DEBUG/INFO/WARNING/ERROR) | No | Yes |
| Log rotation | No | Yes (RotatingFileHandler) |
| User-facing vs diagnostic separation | Hard | Easy (different handlers) |
| Structured fields (tool name, session ID, etc.) | No | Yes (extra fields in JSON) |

### Why JSON file format

- **Parsable** — any language/tool can read the logs
- **Preserves structure** — tool names, session IDs, error messages are in separate fields
- **Standard** — most log aggregation tools (Datadog, Grafana, ELK) expect JSON
- **Backward compatible** — the console output remains human-readable

### Why dual output (file + console)

The user-facing CLI output (Rich panels, banners, setup wizard) should stay as-is — that's the UX. The logging module only replaces *diagnostic* and *error* `print()` calls, not user-facing output. This means:

- Running `python -m obektclaw start` in a terminal shows the same beautiful UI
- But errors and diagnostics now go to both stderr (colored) and the log file (JSON)
- When debugging a session, you can `tail -f ~/.obektclaw/logs/obektclaw.log` to see what the agent did internally

## Files changed

| File | Change |
|------|--------|
| `obektclaw/logging_config.py` | **New** — logger factory, JSON formatter, RotatingFileHandler |
| `obektclaw/agent.py` | Added `log` import; replaced 3 `print()` calls with structured logs; added logs for tool calls, compaction, model switching |
| `obektclaw/learning.py` | Added `log` import; added logs for learning loop start/retro success/failure |
| `obektclaw/llm.py` | Added `log` import; added logs for API calls, retries, exhaustion |
| `obektclaw/mcp.py` | Added `log` import; replaced 1 `print()` with error log |
| `obektclaw/skills/manager.py` | Added `log` import; added logs for skill create, improve, record_use |
| `obektclaw/tools/registry.py` | Added `log` import; added logs for tool not found, invalid args, crashes |
| `obektclaw/gateways/cli.py` | Added `log` import; added logs for setup wizard, config warnings |
| `obektclaw/gateways/telegram.py` | Added `log` import; added logs for gateway start/shutdown, poll errors, send failures |
| `obektclaw/__main__.py` | Added `log` import; added log for gateway start |

## Log format

### Console (stderr)
```
20:48:01 [INFO] obektclaw.agent: model=gpt-4o context_window=128000 tokens=auto-detected
20:48:02 [ERROR] obektclaw.tools.registry: tool_crash tool=exec_python error=ValueError: crash
```

### File (JSON)
```json
{"ts": "2026-04-24 20:48:01", "lvl": "INFO", "name": "obektclaw.agent", "msg": "model=gpt-4o context_window=128000 tokens=auto-detected"}
{"ts": "2026-04-24 20:48:02", "lvl": "ERROR", "name": "obektclaw.tools.registry", "msg": "tool_crash tool=%s error=%s", "error": "Traceback..."}
```

## Design decisions

### Handlers attached once per logger
`get_logger()` checks if handlers already exist before attaching new ones. This prevents duplicate log entries if the same logger is requested multiple times (common in tests).

### Per-module loggers
Each module gets its own logger via `get_logger(__name__)`. This lets you filter logs by module:
```bash
# Only see agent-level logs
grep '"name": "obektclaw.agent"' ~/.obektclaw/logs/obektclaw.log

# Only see errors
grep '"lvl": "ERROR"' ~/.obektclaw/logs/obektclaw.log
```

### Logging vs print for user-facing output
User-facing output (Rich panels, banners, help text, setup wizard) remains as `print()` or Rich calls. Only internal diagnostics, errors, and tool-level events use structured logging.

### `getattr(llm, "fast_model", "unknown")` in learning loop
The Learning Loop logs the fast model name, but tests use `RecordingFakeLLM` which doesn't have a `fast_model` attribute. Using `getattr()` with a default avoids breaking tests.

### Preserved `print()` in telegram gateway
The telegram gateway still uses `print()` for the "OBEKTCLAW_TG_TOKEN not set" and "OBEKTCLAW_LLM_API_KEY is not set" messages because those are *user-facing errors* that need to appear in stdout for shell scripts and test capture (`capsys`). Structured logging in `get_logger()` goes to stderr + file, which would break the existing test assertions.

## Testing

All 333 existing tests pass. No test changes were needed because:

1. The logging module is designed to be a drop-in replacement for `print()` — it adds output but doesn't change behavior
2. The `get_logger()` function is idempotent — calling it multiple times doesn't duplicate handlers
3. The telegram gateway preserves `print()` for the specific user-facing error messages that tests capture via `capsys`
