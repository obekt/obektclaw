# obektclaw Architecture

## Overview

obektclaw is a minimal, self-improving AI agent implementation based on the [Nous Research Hermes Agent](https://github.com/NousResearch/Hermes) concept. It implements a complete agent harness with memory, skills, tools, and a learning loop in ~4,700 lines of Python.

## Core Thesis

**The harness is what matters.** Traditional agent frameworks require humans to manually construct the agent's "harness" (system prompt + memory + skills + tools). The Hermes thesis says the agent itself should weave that harness — ship with it built-in and refine it after every turn. obektclaw implements that idea.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Gateway Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    CLI      │  │  Telegram   │  │  (Future: Web,…)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Core (ReAct Loop)                  │
│  • System prompt assembly                                   │
│  • LLM chat with tool calling                               │
│  • Tool execution & result feedback                         │
│  • Learning Loop invocation                                 │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                      Memory Layer                           │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────────────┐  │
│  │Session │  │ Vector │  │ Graph  │  │   Hybrid         │  │
│  │SQLite  │  │ChromaDB│  │ CogDB  │  │   Retriever      │  │
│  │+ FTS5  │  │+Embed  │  │+Triple │  │   (Ranking)      │  │
│  └────────┘  └────────┘  └────────┘  └──────────────────┘  │
│       │            │            │                │          │
│       └────────────┴────────────┘                │          │
│                    │                             │          │
│              MemorySync                          │          │
│         (CogDB ↔ ChromaDB)                       │          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│    Skills    │   │    Tools     │   │   User Model │
│  ┌────────┐  │   │  ┌────────┐  │   │  ┌────────┐  │
│  │Markdown│  │   │  │  FS    │  │   │  │ 12 Lay │  │
│  │ files  │  │   │  │  Exec  │  │   │  │ SQLite │  │
│  └────────┘  │   │  │  Web   │  │   │  └────────┘  │
│              │   │  │  Memory│  │   │              │
│  • FTS5     │   │  │  Skill │  │   │              │
│  • Vector   │   │  │Delegate│  │   │              │
│  • Create   │   │  └────────┘  │   │              │
│  • Improve  │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Learning Loop (Post-turn)                  │
│  • Fast-model JSON retrospection                            │
│  • Extract facts → Persistent Memory                        │
│  • Update user model → 12-layer profile                     │
│  • Create/improve skills → Markdown files                   │
│  • Log retro → JSONL for debugging                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              External Services (Optional)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  LLM API    │  │   MCP       │  │  (Future: RAG,…)    │ │
│  │ (OpenAI-*)  │  │  Servers    │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Agent Core (`obektclaw/agent.py`)

**Purpose:** Synchronous ReAct loop that orchestrates perception, reasoning, and action.

**Key responsibilities:**
- Build system prompt from user model + facts + skills + relevant history
- Call LLM with tool definitions (OpenAI-style tool calling)
- Execute tool calls and feed results back to LLM
- Loop until LLM produces final answer (no tool calls)
- Invoke Learning Loop after each turn

**Design decisions:**
- **Synchronous:** Simpler reasoning about state; gateways parallelize across users
- **Single connection per session:** Each chat gets its own Agent instance
- **Max steps limit:** Prevents infinite tool loops (default: 12)

### 2. Automatic Memory System (`obektclaw/memory/`)

Memory is **fully automatic** — the agent never calls memory tools. Context is injected transparently during system prompt assembly.

#### Session Memory (`session.py`)
- **Purpose:** Episodic record of current conversation
- **Storage:** SQLite `messages` table + FTS5 index
- **Access:** `recent(limit)` for prompt building, `search_history(query)` for recall
- **Lifetime:** Tied to session; persists across restarts

#### Vector Memory (`vector_memory.py`)
- **Purpose:** Semantic search across facts, conversations, skills, entities
- **Storage:** ChromaDB with local embeddings (`all-MiniLM-L6-v2`, 384-dim, ~80MB)
- **Collections:** `facts`, `memories`, `skills`, `entities`
- **Access:** Similarity search with metadata filtering
- **Lifetime:** Indefinite

#### Graph Memory (`graph_memory.py`)
- **Purpose:** Entity and relationship storage as a knowledge graph
- **Storage:** CogDB triple-store (`cog.torque.Graph`)
- **Entities:** tool, concept, environment, project, person, workflow
- **Relations:** prefers, uses, dislikes, depends_on, related_to, owns, works_on, deployed_on
- **Access:** Graph traversal, BFS for connected entities
- **Lifetime:** Indefinite

#### Hybrid Retriever (`hybrid_retriever.py`)
- **Purpose:** Automatic context assembly for system prompt injection
- **Workflow:**
  1. Vector search for facts and skills (ChromaDB)
  2. Graph traversal for entities connected to retrieved facts (CogDB)
  3. Graph query for user preferences/dislikes
  4. Multi-factor ranking (`ranking.py`)
  5. Select top items within 2000-token budget
- **Agent awareness:** Zero — agent never calls this directly

#### Memory Sync (`memory_sync.py`)
- **Purpose:** Keep entity IDs consistent across CogDB and ChromaDB
- **Syncs:** Entities from graph → vector store; links facts to entities

#### Ranking Algorithm (`ranking.py`)
- **Purpose:** Score retrieved items by multiple factors
- **Factors (100 points max):**
  1. Semantic similarity — 0-40 points
  2. Confidence score — 0-20 points
  3. Recency boost — 0-15 points
  4. Entity connection strength — 0-15 points
  5. Category priority — 0-10 points
- **Selection:** Greedy with diversity penalty to avoid same-source overload

#### User Model (`user_model.py`)
- **Purpose:** 12-layer identity profile (inspired by Honcho)
- **Layers:** technical_level, primary_goals, work_rhythm, comm_style, code_style, tooling_pref, domain_focus, emotional_pattern, trust_boundary, contradictions, knowledge_gaps, long_term_themes
- **Storage:** SQLite `user_traits` table (12 rows max)

**Design decisions:**
- **No agent awareness:** Memory retrieval is transparent; agent never calls memory tools
- **Local embeddings:** `sentence-transformers` with persistent cache; no API calls
- **Graph + vector together:** Graph gives relationships; vector gives semantic similarity
- **Fire-and-forget extraction:** `post_turn.py` runs after every turn, catches all exceptions

### 3. Skill System (`obektclaw/skills/`)

**Purpose:** Markdown-based, self-improving procedural knowledge.

**Storage:**
- **Source of truth:** `~/.obektclaw/skills/*.md` files
- **Mirror:** SQLite `skills` table with FTS5 for search

**Skill file format:**
```markdown
---
name: csv-to-database
description: Clean a CSV file and import it into SQLite
---
# Steps
1. Read the CSV with pandas
2. Clean column names...
```

**Operations:**
- `skill_search(query)` — FTS5 across name/description/body
- `skill_load(name)` — Get full skill for context
- `skill_create(name, desc, body)` — Auto-created by Learning Loop
- `skill_improve(name, append=...)` — Incremental refinement

**Design decisions:**
- **Files on disk:** Users can `vim` skills; git-trackable
- **FTS5 recall:** Lexical + stemming; no embeddings dependency
- **Auto-create:** Learning Loop generates skills from successful patterns

### 4. Tool Registry (`obektclaw/tools/`)

**Built-in tools (16):**

| Category | Tools |
|----------|-------|
| Filesystem | `read_file`, `write_file`, `list_files`, `grep` |
| Execution | `bash`, `exec_python` |
| Web | `web_fetch` |
| Memory | `memory_search`, `memory_set_fact`, `memory_forget_fact`, `user_model_set` |
| Skills | `skill_search`, `skill_load`, `skill_create`, `skill_improve` |
| Orchestration | `delegate` |

**Tool signature:**
```python
def tool_fn(args: dict, ctx: ToolContext) -> ToolResult:
    ...
```

**Design decisions:**
- **Functions, not classes:** Minimal abstraction
- **ToolContext:** Single object carries all dependencies
- **Error handling:** Tool crashes return error text, don't break loop

### 5. Learning Loop / Turn Extraction (`obektclaw/post_turn.py`)

**Purpose:** Post-turn retrospection that makes the agent self-improving.

**Process:**
1. Skip if exchange trivial (<12 chars, no tools)
2. Render current user model
3. Call fast LLM with structured JSON prompt
4. Apply updates:
   - `facts[]` → `PersistentMemory.upsert()`
   - `user_model_updates[]` → `UserModel.set()`
   - `new_skill` → `SkillManager.create()`
   - `skill_improvement` → `SkillManager.improve()`
5. Log retro JSON to `~/.obektclaw/logs/learning-YYYY-MM-DD.jsonl`

**Retro prompt improvements:**
- Explicit "what to EXCLUDE" section (no file paths, counts, transient state)
- Layer descriptions for all 12 user model layers
- Layer assignment guidance (e.g., "tooling_pref is for tools only")

**Design decisions:**
- **Single fast-model call:** Cheap enough for every turn
- **Fire-and-forget:** Exceptions caught; never break user turn
- **JSONL logs:** Debuggable; can replay retrospection

### 6. MCP Bridge (`obektclaw/mcp.py`)

**Purpose:** Connect external tools via Model Context Protocol.

**Implementation:**
- Minimal stdio JSON-RPC 2.0 client
- Calls `initialize`, `notifications/initialized`, `tools/list`
- Registers remote tools as `mcp__<server>__<tool>`
- Relays `tools/call` on invocation

**Auto-load:**
- `Agent.__init__` looks for `~/.obektclaw/mcp.json`
- Claude-Desktop format:
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

**Design decisions:**
- **Tools only:** No resources, prompts, sampling (scope creep)
- **Stdio only:** No HTTP/SSE (complexity)
- **Auto-start:** Config file presence triggers load

### 7. Gateways (`obektclaw/gateways/`)

#### CLI (`cli.py`)
- REPL with slash commands
- First-run welcome message
- Commands: `/help`, `/skills`, `/memory`, `/traits`, `/sessions`, `/setup`, `/exit`
- Session resume: `python -m obektclaw sessions resume <id>`

#### Telegram (`telegram.py`)
- Long-polling bot
- One Agent per `chat_id`
- Thread-per-chat with message queue

**Design decisions:**
- **Stateless gateways:** Agent owns all state
- **Serialization:** Telegram queues per chat to avoid races

## Data Flow (Single Turn)

```
User input
    │
    ▼
┌───────────────────────────────┐
│ 1. Add to session memory      │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ 2. Build system prompt        │
│    • User model (12 layers)   │
│    • HybridRetriever context  │
│      (vector + graph + rank)  │
│    • All skills (capped @ 30) │
│    • Relevant skills (FTS5)   │
│    • Relevant history (FTS5)  │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ 3. LLM chat with tools        │
│    • OpenAI-style tool_call   │
└───────────────────────────────┘
    │
    ├───── No tool_calls ───────┐
    │                           │
    ▼                           │
┌───────────────────────────────┐
│ 4. Execute tool calls         │
│    • registry.call()          │
│    • Append tool results      │
│    • Loop to step 3           │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│ 5. Turn Extraction            │
│    • Extract entities         │
│    • Extract relations        │
│    • Extract facts            │
│    • Update user model        │
│    • Create/improve skills    │
│    • Log to JSONL             │
└───────────────────────────────┘
    │
    ▼
Assistant reply to user
```

## Storage Layout

```
~/.obektclaw/
├── obektclaw.db          # SQLite (WAL mode)
│   ├── sessions       # Conversation sessions
│   ├── messages       # + FTS5 index
│   ├── facts          # Persistent facts (legacy mirror)
│   ├── facts_fts      # FTS5 index
│   ├── user_traits    # 12-layer model
│   └── skills         # Metadata mirror
├── chroma/            # ChromaDB vector store
├── cog-home/          # CogDB graph store
├── models/            # Cached embedding model (~80MB)
│   └── sentence-transformers/
├── skills/            # Markdown files (source of truth)
│   ├── csv-to-database.md
│   ├── deployment-checklist.md
│   └── getting-to-know-you.md
├── logs/              # Extraction JSONL logs
│   └── extraction-2026-04-09.jsonl
└── mcp.json           # Optional MCP config
```

## Configuration

All via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OBEKTCLAW_HOME` | `~/.obektclaw` | Data directory |
| `OBEKTCLAW_LLM_BASE_URL` | OpenAI | LLM endpoint |
| `OBEKTCLAW_LLM_API_KEY` | (required) | API key |
| `OBEKTCLAW_LLM_MODEL` | `gpt-4o-mini` | Main model |
| `OBEKTCLAW_LLM_FAST_MODEL` | same | Learning Loop model |
| `OBEKTCLAW_TG_TOKEN` | (empty) | Telegram bot token |
| `OBEKTCLAW_BASH_TIMEOUT` | `30` | Bash timeout (seconds) |

## Security Model

- **Personal VPS deployment:** Agent runs with user's privileges
- **No sandbox:** `bash` and `exec_python` are unrestricted
- **No telemetry:** No outbound calls except configured LLM + `web_fetch`
- **Secrets in `.env`:** Gitignored; user manages

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| LLM chat (main) | 500ms–5s | Network + model size |
| LLM chat (fast) | 200ms–2s | Learning Loop |
| FTS5 search | <10ms | SQLite index |
| Vector search | <50ms | ChromaDB + local embed |
| Graph traversal | <20ms | CogDB triple-store |
| Tool execution | varies | Bash/Python/network |
| Learning Loop | +500ms–2s | Post-turn overhead |

## Extensibility

### Adding a Tool
1. Create `obektclaw/tools/mytool.py`
2. Write function `def my_fn(args, ctx) -> ToolResult`
3. Register in `build_default_registry()`

### Adding a Gateway
1. Create `obektclaw/gateways/mygateway.py`
2. Instantiate `Agent` per user/session
3. Call `agent.run_once(user_input)`

### Adding Memory Stores
- Not recommended without understanding the sync pipeline
- Would require updates to `MemorySync`, `HybridRetriever`, and `TurnExtractor`

## Known Limitations

1. **No memory expiration:** Facts accumulate indefinitely
2. **Single-threaded agent:** Per-session serialization
3. **No sandbox:** Tools run with full privileges
4. **No multi-agent:** `delegate` is sequential sub-agent

## Testing Strategy

- **Unit tests:** `tests/` (606 tests, all offline)
- **Fake LLM:** `FakeLLMClient` for deterministic tests
- **Temp storage:** `OBEKTCLAW_HOME=/tmp/...` for isolation
- **No live tests:** Token cost; use smoke test script

## Design Principles

1. **Skill files on disk = source of truth**
2. **Memory stays local** (no phone home)
3. **One SQLite connection, one file**
4. **OpenAI-shaped LLM client** (vendor agnostic)
5. **Tools as functions** (minimal abstraction)
6. **Learning Loop is fire-and-forget** (never break turns)
7. **Synchronous by default** (simpler reasoning)

## Future Work

- [x] Session management (list, show, export, resume)
- [x] Context compaction at 85% pressure
- [x] Memory cleanup (auto-expiry + contradiction detection)
- [x] Embeddings-based recall (ChromaDB + sentence-transformers)
- [x] Graph memory (CogDB entity/relationship store)
- [x] Hybrid retrieval (vector + graph + ranking)
- [ ] Multi-agent orchestration (parallel delegate)
- [ ] Multi-agent orchestration (parallel delegate)
- [ ] HTTP MCP transport
- [ ] Sandboxed tool execution (opt-in)
- [ ] Long-horizon evaluation (does agent improve over 50 turns?)
