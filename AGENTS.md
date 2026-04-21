# AGENTS.md — Handoff guide for autonomous agents working on obektclaw

If you are an AI coding agent (Claude, Codex, Cursor, an OpenCode session, anything) picking up work on this repo, **read this file end to end before touching code**. It is the single source of truth for what obektclaw is, what's already built, what was tested live, what is known broken, and where to push next.

The repo is a from-scratch reproduction of the **Hermes Agent** concept described in the Nous Research orange book ("Hermes Agent: The Complete Guide" v260408). The book describes a self-improving personal AI agent with five components — Learning Loop, three-layer memory, self-evolving skill system, 40+ tools + MCP, and a multi-platform gateway. This repo implements all five at an MVP-but-coherent level (~4,700 lines of Python). It is not a production clone; it is a working reference you can run, extend, and test.

---

## 1. The mental model

The Hermes core thesis (from §01 of the orange book): the harness is what matters, and the agent should weave its own harness as it runs. "Harness" = system prompt + memory + skills + tool constraints + feedback loop. Other agent frameworks make the human build the harness manually (CLAUDE.md, hooks, etc.). A Hermes-style agent ships with the harness built in *and* arranges for the agent itself to refine the harness after every turn. obektclaw is our implementation of that idea.

Map of the five components → where they live in this repo:

| Orange book §                          | Component                  | Code                              |
| -------------------------------------- | -------------------------- | --------------------------------- |
| §03 The Learning Loop                  | Auto-retrospection         | `obektclaw/learning.py`              |
| §04 Three-Layer Memory                 | session / persistent / user model | `obektclaw/memory/`             |
| §05 Skill System                       | Self-improving markdown skills | `obektclaw/skills/manager.py`    |
| §06 40+ Tools + MCP                    | Tool registry, built-ins, MCP bridge | `obektclaw/tools/`, `obektclaw/mcp.py` |
| §09 Multi-Platform Access              | CLI + Telegram gateways    | `obektclaw/gateways/`                |

The orange book's five-step Learning Loop (Curate → Create Skill → Self-Improve → FTS5 Recall → User Modeling) is implemented as: **one fast-model JSON retrospection** at the end of every turn (`learning.py`) that emits `facts`, `user_model_updates`, `new_skill`, and `skill_improvement`. FTS5 recall is not a separate step — it happens at *prompt build* time (`agent._compose_system_prompt`).

---

## 2. Project layout (with one-line purpose for each file)

```
obektclaw/
├── AGENTS.md                  ← you are here
├── README.md                  ← user-facing intro and install
├── requirements.txt           ← openai, httpx, python-dotenv, pyyaml
├── .env.example               ← template for credentials/config
├── .env                       ← (gitignored — actually contains the dashscope key right now)
│
├── bundled_skills/            ← starter skills copied to ~/.obektclaw/skills/ on first run
│   ├── getting-to-know-you.md
│   ├── csv-to-database.md
│   └── deployment-checklist.md
│
└── obektclaw/
    ├── __init__.py            ← version constant
    ├── __main__.py            ← `python -m obektclaw <subcommand>` dispatcher
    ├── config.py              ← .env loader + Config dataclass + global CONFIG
    ├── llm.py                 ← OpenAI-compatible LLMClient with chat() / chat_json()
    ├── agent.py               ← The ReAct loop. Single-threaded, per-session.
    │                           Context compaction at 85% pressure. Session resume support.
    ├── sessions.py            ← Session management: list, show, export (md/json), resume
    ├── learning.py            ← The Learning Loop (post-turn retrospection)
    ├── mcp.py                 ← Minimal stdio JSON-RPC MCP client
    ├── model_context.py       ← Context window detection + runtime model switching
    │
    ├── memory/
    │   ├── __init__.py
    │   ├── store.py           ← SQLite + FTS5. SCHEMA constant has all DDL. Thread-safe.
    │   ├── session.py         ← Layer 1 — episodic; messages + FTS5 search
    │   ├── persistent.py      ← Layer 2 — semantic; key/value facts by category
    │   └── user_model.py      ← 12 Honcho-style identity layers
    │
    ├── skills/
    │   ├── __init__.py
    │   └── manager.py         ← Load, search, create, improve markdown skill files
    │
    ├── tools/
    │   ├── __init__.py
    │   ├── registry.py        ← Tool dataclass, ToolRegistry, ToolContext, build_default_registry
    │   ├── fs.py              ← read_file, write_file, list_files, grep
    │   ├── exec.py            ← bash, exec_python (note: shadows stdlib `exec`)
    │   ├── web.py             ← web_fetch (httpx)
    │   ├── memory_tools.py    ← memory_search / set_fact / forget_fact / user_model_set
    │   ├── skill_tools.py     ← skill_search / load / create / improve
    │   └── delegate.py        ← Sub-agent delegation (no recursive delegate inside)
    │
    └── gateways/
        ├── __init__.py
        ├── cli.py             ← REPL on top of agent.run_once
        └── telegram.py        ← Long-poll bot, one Agent per chat_id
```

Total: **16 built-in tools**, **12 user-model layers**, **3 bundled skills**, **~4,700 lines of Python** (plus ~5,800 lines of tests).

---

## 3. Storage layout (what obektclaw writes to disk)

Everything goes under `$OBEKTCLAW_HOME` (default `~/.obektclaw`):

```
$OBEKTCLAW_HOME/
├── obektclaw.db          ← single SQLite file with WAL + FTS5 indexes
├── skills/            ← markdown skill files; this is the source of truth
│   ├── csv-to-database.md
│   ├── deployment-checklist.md
│   ├── getting-to-know-you.md
│   └── <auto-created skills>
└── logs/              ← currently unused; reserved for the Learning Loop diagnostics
```

The SQLite schema (full DDL is in `obektclaw/memory/store.py::SCHEMA`):

- `sessions(id, started_at, ended_at, gateway, user_key)`
- `messages(id, session_id, ts, role, content, tool_name, meta_json)` + `messages_fts` (porter unicode61) + AI/AD/AU triggers
- `facts(id, key, value, category, confidence, created_at, updated_at)` UNIQUE on `(category, key)` + `facts_fts` + triggers
- `user_traits(layer PK, value, evidence, updated_at)` — only 12 rows max, one per layer
- `skills(name PK, description, body, use_count, success_count, created_at, updated_at)` + `skills_fts` + triggers (the markdown files on disk are authoritative; the table is a mirror to enable FTS)

`Store` keeps **one shared connection** behind a `RLock`. Do not open extra connections — use `Store.execute / fetchall / fetchone`. WAL mode is on so concurrent readers from outside processes (e.g. `sqlite3` CLI) work fine.

`_fts_query()` in `store.py` is the **only** place free-text user input gets turned into an FTS5 MATCH expression. Always go through it; never hand a raw user string to MATCH or you will get syntax errors on `'`, `:`, `-`, etc.

---

## 4. The agent loop in one page

`Agent.run_once(user_text, max_steps=12)` in `obektclaw/agent.py`:

1. Append user message to session memory.
2. Build a fresh system prompt from:
   - `SYSTEM_PROMPT` constant
   - `user_model.render_for_prompt()` — all 12 layers
   - Top persistent facts (`persistent.all_top(per_category=4)`)
   - FTS5-recalled skills relevant to *this user input* (`skills.search(user_text, limit=4)`)
   - FTS5-recalled prior message snippets (`session.search_history(user_text, limit=4)`)
3. Append the last ~30 in-session messages (user + assistant only — raw tool turns are dropped because the model has the tool result inline anyway).
4. Call `llm.chat(messages, tools=registry.to_openai_tools())`.
5. If the response has `tool_calls`:
   - Append the assistant message with `tool_calls` shape.
   - For each call: invoke `registry.call`, append `{"role":"tool","tool_call_id":id,"content":...}`, also write the tool result into session memory.
   - Loop.
6. If no `tool_calls`: persist the assistant text to session memory and break.
7. Run `LearningLoop.run(Turn(...))` (catches its own exceptions).

`Agent` is **synchronous and not thread-safe**. The Telegram gateway gives each chat_id its own `_ChatWorker` thread with its own `Agent` instance, and serializes messages within a chat via a Queue.

---

## 5. The Learning Loop in detail

`obektclaw/learning.py::LearningLoop.run(turn)` is invoked after every non-trivial turn (skipped if `len(user_text) < 12 and turn.tool_steps == 0`). It:

1. Renders the current user model.
2. Sends one structured prompt to `llm.chat_json` (fast model) with the schema:
   ```json
   {
     "facts": [{"category", "key", "value", "confidence"}],
     "user_model_updates": [{"layer", "value", "evidence"}],
     "new_skill": {"name", "description", "body"} | null,
     "skill_improvement": {"name", "append"} | null,
     "notes": "..."
   }
   ```
3. Applies each piece via the existing `PersistentMemory`, `UserModel`, and `SkillManager` APIs. Failures are silently swallowed per item (defensive — a malformed JSON shouldn't crash a session).

The retro **uses the fast model** (`OBEKTCLAW_LLM_FAST_MODEL`). On Anthropic that means Haiku; on the dashscope qwen endpoint we currently set both fast and main to `qwen3-coder-plus`.

---

## 6. The 16 built-in tools

| Name             | Category | What it does                                                             |
| ---------------- | -------- | ------------------------------------------------------------------------ |
| `read_file`      | fs       | Read up to 200 KB; UTF-8 with latin-1 fallback                           |
| `write_file`     | fs       | Overwrite (creates parent dirs)                                          |
| `list_files`     | fs       | `ls` with optional fnmatch pattern                                       |
| `grep`           | fs       | Recursive regex; max 200 hits; skips dotdirs                             |
| `bash`           | exec     | `subprocess.run(shell=True, cwd=workdir)` with timeout                   |
| `exec_python`    | exec     | Writes a tempfile, runs `python3 file.py`                                |
| `web_fetch`      | web      | httpx GET, optional HTML strip, 400KB cap                                |
| `memory_search`  | memory   | FTS5 across messages + facts                                             |
| `memory_set_fact`| memory   | Upsert into `facts` table                                                |
| `memory_forget_fact` | memory | Delete a fact                                                          |
| `user_model_set` | memory   | Upsert one of the 12 user-model layers                                   |
| `skill_search`   | skill    | FTS5 against the skill corpus                                            |
| `skill_load`     | skill    | Return the full markdown body of a skill                                 |
| `skill_create`   | skill    | Create a new skill markdown file                                         |
| `skill_improve`  | skill    | Replace body, replace description, or append                             |
| `delegate`       | orchestration | Spawn a sub-Agent (registry minus `delegate`) and return its answer  |

All tools follow the same shape: `fn(args: dict, ctx: ToolContext) -> ToolResult`. `ToolContext` carries `config`, `session`, `persistent`, `user_model`, `skills`, `llm`. To add a new built-in tool: write a function and a `Tool(...)` registration in a new file under `tools/`, then call `register(reg)` from `tools/registry.py::build_default_registry`.

---

## 7. MCP bridge

`obektclaw/mcp.py` is a **minimal** stdio MCP client. It speaks JSON-RPC 2.0 over a child process's stdin/stdout, calls `initialize`, `notifications/initialized`, and `tools/list`, then registers each remote tool into the local `ToolRegistry` as `mcp__<server_name>__<tool_name>`. When the agent calls one of those names, the local wrapper relays a `tools/call`.

What it supports: tools.
What it does **not** support: resources, prompts, sampling, server-initiated requests, HTTP transport, OAuth.

To wire an MCP server in, the loader expects a JSON file in roughly Claude-Desktop format:

```json
{
  "mcpServers": {
    "fs": {"command": "npx", "args": ["@modelcontextprotocol/server-filesystem", "/tmp"]}
  }
}
```

Currently **nothing in the codebase calls `load_mcp_config()` automatically**. That's a known TODO — see §10.

---

## 8. Configuration

`obektclaw/config.py` loads `.env` from the repo root (no python-dotenv dependency — it parses by hand) and surfaces a frozen `CONFIG` dataclass. Env vars:

| Variable                   | Default                       | Notes                                                  |
| -------------------------- | ----------------------------- | ------------------------------------------------------ |
| `OBEKTCLAW_HOME`              | `~/.obektclaw`                   | Where SQLite + skills live                             |
| `OBEKTCLAW_LLM_BASE_URL`      | `https://api.openai.com/v1`   | Any OpenAI-compatible endpoint                         |
| `OBEKTCLAW_LLM_API_KEY`       | (required)                    | Raises if missing on first LLMClient call              |
| `OBEKTCLAW_LLM_MODEL`         | `gpt-4o-mini`                 | Main reasoning/tool-use model                          |
| `OBEKTCLAW_LLM_FAST_MODEL`    | falls back to LLM_MODEL       | Used by the Learning Loop                              |
| `OBEKTCLAW_TG_TOKEN`          | empty                         | Telegram bot token                                     |
| `OBEKTCLAW_TG_ALLOWED_CHAT_IDS` | empty (all allowed)         | Comma-separated chat IDs whitelist                     |
| `OBEKTCLAW_BASH_TIMEOUT`      | `30`                          | Default subprocess timeout                             |
| `OBEKTCLAW_WORKDIR`           | `cwd`                         | Working dir for `bash`, `read_file` relative paths     |

`config.CONFIG` is a module-level singleton — fine for the CLI/Telegram entry points, less ideal for tests. Tests should construct `Config` directly or use `OBEKTCLAW_HOME=/tmp/x` to isolate state.

---

## 9. How it has been tested (live, against a real LLM)

**Endpoint used during dev**: `https://coding-intl.dashscope.aliyuncs.com/v1`
**Model used**: `qwen3-coder-plus` (verified to support OpenAI-style tool calling)
**Test runner**: ad-hoc Python in `.venv/bin/python -c`, with `OBEKTCLAW_HOME=/tmp/obektclaw-live`.

What I ran (3 turns, single agent instance, single session):

1. **Turn 1 — preference capture**: "Remember that I always use httpx instead of requests, and my server runs on Hetzner CX22."
   - Agent reply: hallucinated "I already remember this from a previous interaction" (wrong — first turn).
   - Learning Loop: ✅ correctly saved `(preference) httpx_over_requests` and `(env) server_hetzner_cx22`.

2. **Turn 2 — tool use**: "How many python files are in the obektclaw/ directory? Use a tool, don't guess."
   - Agent reply: called a fs tool, answered **7** (correct for the top-level only — there are more under `memory/`, `skills/`, `tools/`, `gateways/`). The model interpreted the question literally; arguable but not wrong.

3. **Turn 3 — skill recall**: "I have a CSV file at /tmp/x.csv. Walk me through importing it. Search your skills first."
   - Agent reply: noticed the file doesn't exist, asked the user to verify the path. Did **not** automatically `skill_load` the `csv-to-database` skill into context — that's a prompting weakness, the system prompt says to search but the model only verified the file.

**End state on disk after the run**:

- 4 persistent facts saved (2 good — `httpx_over_requests`, `server_hetzner_cx22`; 2 noise — `csv_file_path`, `obektclaw_dir_python_files_count`).
- 1 user-model layer set: `tooling_pref → "may need to verify file paths before processing"`. **Wrong layer** — that's a behavioral observation, not a tooling preference.
- All 3 bundled skills present, no new skills auto-created.

**What this test proves**: the full pipeline works end to end — system prompt assembly, OpenAI-style tool calling, real subprocess execution, tool-result feedback, final reply, Learning Loop retrospection, JSON parsing, SQLite/FTS5 writes. Memory persists across `Agent` instances within the same `OBEKTCLAW_HOME`.

**What this test does not prove**: long-horizon behavior, the FTS5 recall actually improving subsequent answers, sub-agent delegation, MCP bridging, Telegram gateway, the bundled skills ever being auto-loaded.

---

## 10. Known issues, sharp edges, and TODOs (prioritized)

### 10.1 Learning Loop retro is over-eager

The retro prompt in `learning.py::RETRO_SYSTEM` says "things that should still be true a week from now" but qwen3-coder-plus saved `csv_file_path = /tmp/x.csv` and `obektclaw_dir_python_files_count = 7 python files in obektclaw/ directory` as facts. Both are ephemeral.

**Fix**: add 2-3 few-shot examples to `RETRO_SYSTEM` showing what to *exclude* (file paths from one-off questions, counts, transient state). Test with at least 5 different turn types to confirm no regression.

### 10.2 User-model layer misclassification

The retro put a behavioral observation into `tooling_pref`. The 12 layers are listed in the prompt but without descriptions. **Fix**: append a one-line description to each layer in `RETRO_SYSTEM`'s layer list (mirror `obektclaw/memory/user_model.py`'s docstring).

### 10.3 No mcp config loaded automatically

`mcp.load_mcp_config()` and `attach_mcp_servers()` exist but nothing calls them. **Fix**: in `agent.Agent.__init__`, look for `$OBEKTCLAW_HOME/mcp.json` and call `attach_mcp_servers(self.registry, load_mcp_config(...))`. Track the returned `MCPServer` instances on the Agent so `close()` can stop them.

### 10.4 Agent never explicitly loads bundled skills into context

The system prompt only mentions skills returned by `skills.search(user_text)`. If the user's query doesn't lexically overlap with any skill name/description, the model never sees that skill exists. **Fix candidates**:
- Always include a one-line list of *all* skills (capped at ~30) in the system prompt — cheap, makes self-discovery easy.
- Or: use embeddings instead of FTS5 for skill recall (bigger change, adds a dependency).

### 10.5 No memory expiration

The orange book §04 explicitly notes Hermes itself doesn't have auto-expiration. We don't either. Long-term users will accumulate junk facts. **Fix**: add a periodic "memory hygiene" pass (could be a separate slash command `python -m obektclaw memory cleanup` that asks the LLM to identify stale/contradictory facts).

### 10.6 `tools/exec.py` shadows the stdlib `exec` builtin

Imported as `from . import exec as execmod` in `registry.py`. Works, but it's awkward. Rename to `tools/run.py` or `tools/execution.py`.

### 10.7 No tests

There is no `tests/` directory. The smoke tests I ran were one-off `python -c` blobs. **Fix**: add `tests/` with at least:
- `test_store.py` — schema, FTS5 sanitization, upsert/conflict
- `test_skills.py` — frontmatter parsing, slugify, create/improve roundtrip
- `test_agent_offline.py` — fake LLMClient, verify the loop dispatches tools and writes session memory correctly without hitting the network
- `test_learning_loop.py` — fake LLM that returns canned JSON, verify facts/user_model/skills get persisted

### 10.8 Tools run with the caller's full privileges

`bash` and `exec_python` are unsandboxed. Acceptable for a personal-VPS deployment (matches the orange book's threat model: it's *your* server) but worth flagging in the README. There is no allowlist, no file path restriction beyond `workdir`, and no network restriction.

### 10.9 Telegram gateway has no graceful shutdown

`KeyboardInterrupt` cleans up but `_ChatWorker` threads block on a 300s `Queue.get` and won't exit until that times out. **Fix**: shorter timeout + a sentinel-aware worker loop, or use `daemon=True` (already true) and just exit hard.

### 10.10 The system prompt never says "be brief"

User-facing replies tend toward verbosity in qwen3-coder-plus. Consider tightening `SYSTEM_PROMPT` in `agent.py` with an explicit "Be terse with the user; tool results are for you, not them. Don't summarize what you just did unless asked."

### 10.11 No way to inspect the per-turn retro

The Learning Loop appends a one-line `system` message to session memory with its `notes` field, but the full retro JSON is discarded. For debugging, persist the raw retro JSON to `$OBEKTCLAW_HOME/logs/learning-YYYY-MM-DD.jsonl`.

### 10.12 The `delegate` tool builds a brand-new registry every time

`delegate.py::_clone_registry_minus_delegate` calls `build_default_registry()` on each invocation. That's fine functionally, but expensive if a parent fans out many sub-agents. Cache it on the parent registry, or pass through the parent registry minus the `delegate` entry.

---

## 11. How to run obektclaw (operator instructions)

```bash
cd obektclaw
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OBEKTCLAW_LLM_API_KEY etc.

# start obektclaw — auto-detects CLI + Telegram gateways
python -m obektclaw start

# manage sessions
python -m obektclaw sessions list
python -m obektclaw sessions show 42
python -m obektclaw sessions export 42 --format json --output session.json
python -m obektclaw sessions resume 42

# manage skills
python -m obektclaw skill list
python -m obektclaw skill show csv-to-database

# inspect memory
python -m obektclaw memory recent
python -m obektclaw memory search "deployment"
python -m obektclaw traits
```

The CLI gateway also has slash commands: `/skills`, `/memory <q>`, `/traits`, `/sessions`, `/exit`.

Legacy `python -m obektclaw chat` and `python -m obektclaw tg` still work as aliases for `start cli` and `start tg` respectively.

To wipe state and start fresh: `rm -rf ~/.obektclaw` (or whatever `OBEKTCLAW_HOME` is).

---

## 12. How to test changes locally without burning tokens

The smoke test I used (no LLM, exercises everything else):

```python
import os; os.environ.setdefault("OBEKTCLAW_HOME", "/tmp/obektclaw-test"); os.environ.setdefault("OBEKTCLAW_LLM_API_KEY", "dummy")
from obektclaw.config import CONFIG
from obektclaw.memory.store import Store
from obektclaw.memory import PersistentMemory, UserModel, SessionMemory
from obektclaw.skills import SkillManager

store = Store(CONFIG.db_path)
sm = SkillManager(store, CONFIG.skills_dir, CONFIG.bundled_skills_dir)
sid = store.open_session("test", "tester")
sess = SessionMemory(store, sid)
sess.add("user", "I always use httpx instead of requests")
print("hits:", len(sess.search_history("httpx")))
PersistentMemory(store).upsert("http_client", "httpx", category="preference")
print(PersistentMemory(store).search("httpx"))
print([s.name for s in sm.search("csv import database")])
```

If you want to test the agent loop without real LLM calls, write a `FakeLLMClient` in `tests/` with the same `chat()` / `chat_json()` signature and inject it via `Agent(..., llm=FakeLLMClient())`. The agent code already supports this via the constructor parameter.

---

## 13. Live test endpoint currently configured

The repo's `.env` is already populated with a working dashscope endpoint:

```
OBEKTCLAW_LLM_BASE_URL=https://coding-intl.dashscope.aliyuncs.com/v1
OBEKTCLAW_LLM_API_KEY=sk-sp-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
OBEKTCLAW_LLM_MODEL=qwen3-coder-plus
OBEKTCLAW_LLM_FAST_MODEL=qwen3-coder-plus
OBEKTCLAW_HOME=/tmp/obektclaw-live
```

This key was provided by the user explicitly for development use. **Treat it as live but rotatable** — don't commit `.env` to git, don't paste it into bug reports. If you suspect it leaked, ask the user to rotate.

`qwen3-coder-plus` does support OpenAI-style tool calling. We verified the `tools/list` listing endpoint is **not** available (404), so don't rely on it; hardcode the model.

---

## 14. Design principles to preserve

If you're refactoring or extending, please keep these properties:

1. **The skill files on disk are the source of truth**, not the SQLite mirror. A user should be able to `vim ~/.obektclaw/skills/foo.md` and have the change picked up on next `reindex()`. Don't move skill bodies into SQLite-only.
2. **Memory stays local.** No outbound calls except to the configured LLM endpoint and `web_fetch`. No telemetry. No "phone home". The orange book is loud about this and so are we.
3. **One SQLite connection, one file.** Don't add a second DB or a vector store as a hard dependency. If you want embeddings, make them optional and degrade gracefully.
4. **No hard dependency on a specific LLM vendor.** The `LLMClient` is OpenAI-shaped on purpose. Anthropic, OpenRouter, vLLM, Ollama, dashscope all work today. Don't reach for `anthropic.Client` directly.
5. **Tools are functions, not classes.** Adding a tool should be ~20 lines. Don't wrap them in heavy abstractions.
6. **The Learning Loop is fire-and-forget.** It must never crash a user-facing turn. All exceptions inside `LearningLoop.run` are caught at the `agent.py` call site.
7. **Synchronous by default.** The Telegram gateway parallelizes across chats but each chat is serial. Don't rewrite the agent loop to be async unless there's a real reason — it makes sub-agent delegation, tool execution, and memory writes much harder to reason about.

---

## 15. Suggested next sprint (concrete, ranked)

If I were the next agent picking this up, I would do these in order:

1. **Add `tests/`** with a `FakeLLMClient` and the four test files in §10.7. Wire `pytest` into requirements. ~2 hours.
2. **Tighten `RETRO_SYSTEM`** to fix §10.1 and §10.2 (junk facts + layer misclassification). Validate by running a 10-turn synthetic conversation and inspecting the post-state. ~1 hour.
3. **Auto-load `mcp.json`** (§10.3). Test by wiring up `@modelcontextprotocol/server-filesystem` against a temp dir. ~1 hour.
4. **Always-include skills index** in the system prompt (§10.4). One-line per skill, capped. Verify token usage stays under control with a `len()` assertion. ~30 min.
5. **Add `python -m obektclaw memory cleanup`** (§10.5). Use the fast model. ~1 hour.
6. **Persist retro JSONL** (§10.11) to make all of the above debuggable. ~30 min.
7. Then start on the genuinely hard stuff: long-horizon evaluations (does the agent actually get smarter over 50 turns?), sub-agent delegation in earnest, and a non-trivial Skill that gets auto-improved across runs.

---

## 16. Things you should NOT do without asking the user first

- Commit anything to git without user confirmation.
- Modify `.env` or rotate the API key.
- Change the storage format in a way that requires migration (the SQLite schema should evolve additively only — new tables or new columns with defaults).
- Add a heavy dependency (langchain, llamaindex, a vector DB, an ORM). The point of obektclaw is that it's small and you can read every line.
- Replace the Learning Loop's single-call retro with a multi-step pipeline (it'll get expensive fast and the user is paying per token).
- Pull in a new LLM provider's native SDK. Stay OpenAI-shaped.

---

## 17. Where to look first when something is broken

| Symptom                                   | First place to look                                               |
| ----------------------------------------- | ----------------------------------------------------------------- |
| `OperationalError: malformed MATCH expression` | `store._fts_query` — sanitization missed a character            |
| Tool calls happen but tool results never come back to the model | The OpenAI-style `tool_call_id` must match — see `agent.py` step 5 |
| Skill changes on disk not reflected       | `SkillManager.reindex()` only runs in `__init__` and after writes — call it manually if you edit files between turns |
| Learning Loop silently doing nothing      | Likely the model returned non-JSON; `chat_json` returns `None`. Add a log line in `LearningLoop.run` |
| Telegram bot eating updates but not replying | `_ChatWorker` exception eaten by the `try/except` — add a `print(e)` |
| Memory not persisting between runs        | `OBEKTCLAW_HOME` differs between invocations (env not loaded, or `.env` not in repo root) |
| `sub-agent` never returns                 | `delegate.py` — sub-agent's `max_steps` is too low, or it's hitting a tool that hangs |

---

## 18. One-paragraph summary you can paste to a fresh agent

> obektclaw is a ~4,700-line Python implementation of the Hermes Agent concept (Nous Research orange book): a self-improving personal AI agent with a built-in "harness" that grows on its own. It has a synchronous ReAct loop (`obektclaw/agent.py`), three-layer SQLite+FTS5 memory (`obektclaw/memory/`), markdown-based self-improving skills (`obektclaw/skills/manager.py` + `bundled_skills/`), 16 built-in tools (`obektclaw/tools/`), a stdio MCP client bridge with auto-load (`obektclaw/mcp.py`), a Learning Loop that runs structured retrospection after every turn (`obektclaw/learning.py`), and session management with export and resume (`obektclaw/sessions.py`). It works against any OpenAI-compatible endpoint — see `.env.example`. CLI and Telegram gateways exist (`obektclaw/gateways/`). 333 offline tests (`tests/`, fake LLM) cover storage, skills, agent loop, learning loop, tools, sessions, and gateways. Read `AGENTS.md` before changing anything.
