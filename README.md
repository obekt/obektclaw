# obektclaw

> **The AI agent you can read, own, and that actually gets smarter.**
>
> 2,900 lines. No containers. No vendor lock-in. No telemetry. Just a self-improving agent that fits in your head.

obektclaw is a minimal, complete implementation of the [Nous Research Hermes Agent](https://github.com/NousResearch/Hermes) concept — except it actually learns from every conversation. It features:

- 🧠 **Three-layer memory** (session + persistent + 12-layer user model)
- 📚 **Self-improving skills** (markdown files that auto-create and improve)
- 🛠️ **16 built-in tools** + MCP bridge for external tools
- 🔄 **Learning Loop** (retrospects after every turn, learns from experience)
- 💬 **CLI + Telegram gateways** (chat locally or via Telegram)

## Quick Start

```bash
git clone <repo-url> obektclaw
cd obektclaw
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Edit .env with your LLM credentials
cp .env.example .env
# Edit .env...

# Start chatting
python -m obektclaw chat
```

On first run, you'll see a welcome message with examples. Type `/help` for full documentation.

## Why obektclaw over OpenClaw / NanoClaw / others?

| | obektclaw | OpenClaw | NanoClaw |
|---|---|---|---|
| **Codebase** | ~2,900 lines of Python. Read it in an afternoon. | 50k+ lines. Good luck auditing that. | Containers + Anthropic SDK. You're renting, not owning. |
| **Memory** | 3-layer: session + persistent facts + 12-trait user model | Basic conversation history | Session-only, no user modeling |
| **Self-improvement** | Learning Loop runs *every turn* — extracts facts, updates your model, creates & improves skills automatically | Manual skill creation | No skill system |
| **Skills** | Plain markdown on disk. `vim` them. `git` them. The agent rewrites them as it learns. | Plugin system (code-heavy) | N/A |
| **Dependencies** | `pip install` — 4 runtime deps. No Docker, no Node, no Rust toolchain. | Heavy dependency tree | Requires Docker containers |
| **Privacy** | Zero telemetry. Your LLM, your data, your machine. Period. | Telemetry opt-out | Cloud-dependent features |
| **Bring your LLM** | Any OpenAI-compatible endpoint — OpenRouter, Ollama, local models, whatever | Anthropic-first | Anthropic-only (Agents SDK) |
| **You can actually fork it** | Small enough to understand, modify, and own. This is *your* agent. | Fork it and maintain 50k lines? | Fork a container orchestrator? |

> **The thesis:** Most agent frameworks give you a black box that talks to an API.
> obektclaw gives you a white box that *rewrites itself* — and it fits in your head.

---

## What Makes It Special

### Self-Improving Skills

Skills are **markdown files on disk** that auto-create when the agent discovers reusable patterns:

```markdown
---
name: csv-to-database
description: Clean a CSV file and import it into SQLite
---
# Steps
1. Read the CSV with pandas
2. Clean column names...
```

You can edit them with `vim`, track them with git, and the agent improves them after each use.

### Three-Layer Memory

1. **Session memory** — Conversation history (FTS5 searchable)
2. **Persistent facts** — Long-term knowledge about you
3. **User model** — 12-layer profile (technical level, goals, preferences, etc.)

### Learning Loop

After every turn, a fast LLM call extracts:
- New facts to remember
- User model updates
- New skills to create
- Existing skills to improve

All applied immediately. Logged to JSONL for debugging.

### MCP Bridge

Connect external tools via [Model Context Protocol](https://modelcontextprotocol.io/):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

Tools auto-register as `mcp__filesystem__read_file`, etc.

## Commands

### In Chat

| Command | Description |
|---------|-------------|
| `/help` | Show detailed help |
| `/skills` | List known skills |
| `/memory <q>` | Search persistent memory |
| `/traits` | Show your user model |
| `/setup` | Configuration wizard |
| `/exit` | Quit |

### CLI

```bash
python -m obektclaw chat          # Interactive REPL
python -m obektclaw setup         # Setup wizard
python -m obektclaw tg            # Telegram bot
python -m obektclaw skill list    # List skills
python -m obektclaw memory status # Memory health check
python -m obektclaw --help        # All commands
```

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** — Getting started guide
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — System architecture
- **[docs/NOVELTY.md](docs/NOVELTY.md)** — Why this is novel
- **[AGENTS.md](AGENTS.md)** — AI developer handoff guide
- **[TEST_SUMMARY.md](TEST_SUMMARY.md)** — Test suite overview

## Configuration

All via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OBEKTCLAW_HOME` | `~/.obektclaw` | Data directory |
| `OBEKTCLAW_LLM_BASE_URL` | OpenAI | LLM endpoint |
| `OBEKTCLAW_LLM_API_KEY` | (required) | API key |
| `OBEKTCLAW_LLM_MODEL` | `gpt-4o-mini` | Main model |
| `OBEKTCLAW_LLM_FAST_MODEL` | same | Learning Loop model |
| `OBEKTCLAW_CONTEXT_WINDOW` | auto | Context window size (e.g. `200000` for Claude) |
| `OBEKTCLAW_TG_TOKEN` | (empty) | Telegram bot token |

See `.env.example` for full configuration.

## Testing

```bash
pip install -r requirements.txt  # includes pytest
python -m pytest
```

235 tests covering storage, skills, agent loop, and learning loop. All offline (fake LLM).

## Security Model

- **Personal deployment** — Runs with your privileges on your server
- **No sandbox** — `bash` and `exec_python` are unrestricted (by design)
- **No telemetry** — No outbound calls except configured LLM + `web_fetch`
- **Secrets in `.env`** — Gitignored; you manage

## Project Structure

```
obektclaw/
├── obektclaw/              # Core package (yes, folder is still "obektclaw")
│   ├── agent.py         # ReAct loop
│   ├── learning.py      # Learning Loop
│   ├── memory/          # 3-layer memory
│   ├── skills/          # Markdown skill system
│   ├── tools/           # 16 built-in tools
│   ├── mcp.py           # MCP bridge
│   └── gateways/        # CLI + Telegram
├── bundled_skills/      # Starter skills
├── tests/               # 235 tests
├── docs/                # Architecture + novelty docs
├── QUICKSTART.md        # Getting started
└── README.md            # This file
```

## Why "obektclaw"?

This is a personal implementation of the Hermes Agent concept, customized and extended. The name reflects ownership while honoring the original thesis.

## License

MIT — see LICENSE file.

## Contributing

1. Read [AGENTS.md](AGENTS.md) for architecture and design principles
2. Run tests: `python -m pytest`
3. Keep it minimal — no heavy dependencies
4. Skills stay on disk (not DB-only)
5. Memory stays local (no phone home)

## Future Work

- [ ] Memory cleanup (auto-expiry + contradiction detection)
- [ ] Embeddings-based recall (optional, degrades to FTS5)
- [ ] Multi-agent orchestration (parallel delegate)
- [ ] HTTP MCP transport
- [ ] Sandboxed tool execution (opt-in)
- [ ] Long-horizon evaluation (does agent improve over 50+ turns?)

---

**The harness is what matters — and the agent should weave its own harness as it runs.**

If obektclaw clicks for you, star the repo and share it. The best agent framework is the one you can actually understand.
