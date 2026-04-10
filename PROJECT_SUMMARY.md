# obektclaw — Project Summary

## What We Built

**obektclaw** is a self-improving AI agent that weaves its own harness. It's a minimal (~4,700 lines), complete implementation of the Nous Research Hermes Agent concept with:

- 🧠 Three-layer memory (session + persistent + 12-layer user model)
- 📚 Self-improving skills (markdown files that auto-create and improve)
- 🛠️ 16 built-in tools + MCP bridge
- 🔄 Learning Loop (retrospects after every turn)
- 💬 CLI + Telegram gateways
- 📋 Session management (list, show, export, resume)
- ✅ 326 tests (all offline)

## Files Created/Updated

### Documentation (Public)
| File | Purpose |
|------|---------|
| `README.md` | Public-facing project overview |
| `QUICKSTART.md` | Getting started guide |
| `docs/ARCHITECTURE.md` | System architecture |
| `docs/NOVELTY.md` | Why this is novel |
| `RELEASE.md` | Pre-commit checklist |
| `AI_DEV_NOTES.md` | AI developer handoff |
| `TEST_SUMMARY.md` | Test suite overview |

### Configuration (Git-Safe)
| File | Purpose |
|------|---------|
| `.gitignore` | Excludes secrets, DBs, logs |
| `.env.example` | Template with placeholders |
| `pyproject.toml` | pytest configuration |

### Code (Renamed)
| Old | New |
|-----|-----|
| `obektclaw/` | `obektclaw/` |
| All imports updated | `from obektclaw.*` |
| CLI: `python -m obektclaw` | `python -m obektclaw` |

### Internal Docs (Private)
| File | Purpose |
|------|---------|
| `AGENTS.md` | AI developer handoff |

## Current State

```
obektclaw/
├── obektclaw/           # Core package
│   ├── __init__.py      # v0.1.0
│   ├── __main__.py      # CLI dispatcher
│   ├── agent.py         # ReAct loop + session resume
│   ├── sessions.py      # Session management, export, resume
│   ├── learning.py      # Learning Loop
│   ├── llm.py           # OpenAI-compatible client
│   ├── config.py        # Configuration
│   ├── mcp.py           # MCP bridge
│   ├── memory/          # 3-layer memory
│   ├── skills/          # Markdown skills
│   ├── tools/           # 16 built-in tools
│   └── gateways/        # CLI + Telegram
├── tests/               # 326 tests
├── bundled_skills/      # 3 starter skills
├── docs/                # Architecture + novelty
├── .gitignore           # Secrets excluded
├── .env.example         # Config template
├── README.md            # Public docs
├── QUICKSTART.md        # Getting started
└── pyproject.toml       # pytest config
```

## Test Results

```
326 passed in 0.65s
```

All tests pass. No live LLM calls (fake LLM clients).

## Security Audit

✅ `.env` is gitignored
✅ `.env.example` has placeholders (no real keys)
✅ No API keys in code
✅ No user data in repo (DBs, logs, JSONL)
✅ `~/.obektclaw/` is outside repo

## How to Use

### For Users
```bash
git clone <repo>
cd obektclaw
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your LLM credentials
python -m obektclaw chat
```

### For Developers
```bash
# Read docs
cat README.md
cat docs/ARCHITECTURE.md
cat AI_DEV_NOTES.md

# Run tests
python -m pytest

# Add a tool
# See AI_DEV_NOTES.md "Common Tasks"
```

### For AI Agents
```bash
# Read this first
cat AI_DEV_NOTES.md

# Then
cat AGENTS.md
cat docs/ARCHITECTURE.md

# Run tests to verify
python -m pytest
```

## Novel Contributions

1. **Markdown skills on disk** — Editable with vim, git-trackable
2. **FTS5-only recall** — No embeddings dependency
3. **12-layer user model** — Forced abstraction
4. **Fire-and-forget Learning Loop** — Every turn, cheap
5. **MCP auto-load** — Plug-and-play tools
6. **Self-documenting UX** — No external docs needed

## Next Steps

### Immediate (Before Public Release)
1. ✅ Rename to obektclaw
2. ✅ Create .gitignore
3. ✅ Create .env.example
4. ✅ Create public README
5. ✅ Create RELEASE.md checklist
6. ✅ Audit for secrets
7. ✅ Update all imports
8. ⏳ Run `./RELEASE.md` checklist
9. ⏳ Update AGENTS.md to reference obektclaw

### Future Development
1. Memory cleanup (auto-expiry)
2. Embeddings-based recall (optional)
3. Multi-agent orchestration
4. HTTP MCP transport
5. Sandboxed execution (opt-in)
6. Long-horizon evaluation

## Git Commands (When Ready)

```bash
# Initial commit
git add .
git commit -m "Initial release: obektclaw v0.1.0

A minimal, self-improving AI agent (~2,900 lines).

Features:
- Three-layer memory
- Self-improving markdown skills
- 16 built-in tools + MCP bridge
- Learning Loop
- CLI + Telegram gateways
- 235 tests

Based on Nous Research Hermes Agent concept."

# Push
git remote add origin <repo-url>
git push -u origin main
```

## Credits

- **Concept:** Nous Research Hermes Agent (orange book)
- **Implementation:** obektclaw team
- **Tests:** 84 offline tests with fake LLM clients
- **Design principles:** Minimal, readable, self-improving

---

**The harness is what matters — and the agent should weave its own harness as it runs.**
