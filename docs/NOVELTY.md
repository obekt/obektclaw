# Novelty Statement: Why obektclaw Matters

## The Problem

Current AI agent frameworks (LangChain, AutoGen, CrewAI, etc.) treat the **agent architecture as static infrastructure**. Humans must manually configure:
- System prompts
- Memory schemas
- Tool definitions
- Skill libraries
- Feedback mechanisms

This creates **fragile, high-maintenance agents** that don't improve with use and require expert knowledge to extend.

## The Hermes Thesis

The Nous Research "orange book" (Hermes Agent: The Complete Guide) proposes a different approach:

> **The harness is what matters — and the agent should weave its own harness as it runs.**

"Harness" = system prompt + memory + skills + tool constraints + feedback loop.

A Hermes-style agent ships with the harness built-in and **arranges for the agent itself to refine the harness after every turn**.

## What We Built

obektclaw is a **complete, minimal reproduction** of the Hermes Agent concept in ~2,900 lines of Python. It implements all five core components:

| Component | Implementation | Lines |
|-----------|----------------|-------|
| Learning Loop | `obektclaw/learning.py` | ~150 |
| Three-Layer Memory | `obektclaw/memory/` | ~400 |
| Self-Evolving Skills | `obektclaw/skills/` | ~250 |
| 16 Built-in Tools + MCP | `obektclaw/tools/`, `obektclaw/mcp.py` | ~600 |
| Multi-Platform Gateway | `obektclaw/gateways/` | ~300 |

Plus: agent core, config, LLM client, test suite (235 tests).

## Novel Contributions

### 1. Markdown-Based Self-Improving Skills

**Prior art:** Skills/prompts are code or database records.

**Our approach:** Skills are **markdown files on disk** that:
- Can be edited with `vim` (no DB tooling)
- Are git-trackable (version control, diff, review)
- Auto-create when Learning Loop discovers patterns
- Self-improve via append/replace after each use

**Why it matters:** Users can **read, edit, and audit** their agent's skills. No black box.

### 2. FTS5-Only Recall (No Embeddings)

**Prior art:** Vector databases, embeddings, semantic search.

**Our approach:** SQLite FTS5 with porter stemming + query sanitization.

**Why it matters:**
- **Zero new dependencies** (no FAISS, no Pinecone, no torch)
- **Deterministic** (no embedding drift, no ANN approximation)
- **Fast enough** (<10ms queries)
- **Degrades gracefully** (can add embeddings later as optional layer)

### 3. 12-Layer User Model (Honcho-Inspired)

**Prior art:** Flat key-value memory or unstructured "notes about user".

**Our approach:** Exactly 12 layers with explicit semantics:
1. `technical_level`
2. `primary_goals`
3. `work_rhythm`
4. `comm_style`
5. `code_style`
6. `tooling_pref`
7. `domain_focus`
8. `emotional_pattern`
9. `trust_boundary`
10. `contradictions`
11. `knowledge_gaps`
12. `long_term_themes`

**Why it matters:** Forces **abstraction over accumulation**. Learning Loop must categorize inferences, not just dump facts.

### 4. Fire-and-Forget Learning Loop

**Prior art:** Separate training phases, human-in-the-loop feedback, expensive RLHF.

**Our approach:** After **every turn**, a fast LLM call emits structured JSON:
```json
{
  "facts": [...],
  "user_model_updates": [...],
  "new_skill": {...} | null,
  "skill_improvement": {...} | null,
  "notes": "..."
}
```

Applied immediately. Exceptions caught. Never breaks user turn.

**Why it matters:** **Continuous improvement** without human intervention. Cheap enough to run every turn (~500ms with fast model).

### 5. MCP Auto-Load

**Prior art:** Manual tool registration, hardcoded tool sets.

**Our approach:** `Agent.__init__` checks for `~/.obektclaw/mcp.json` and auto-registers external tools as `mcp__<server>__<tool>`.

**Why it matters:** **Plug-and-play tool ecosystem**. Connect filesystem, database, API servers without code changes.

### 6. Self-Documenting UX

**Prior art:** CLI tools with `--help` that list commands but not examples.

**Our approach:**
- First-run welcome with **concrete examples**
- `/help` shows **what Obektclaw can do**, not just commands
- `/setup` is a **guided wizard**, not a config file
- `/traits` explains **what was learned**, not raw data
- `/memory status` shows **health check**, not just counts

**Why it matters:** **No docs to read**. Users discover capabilities through interaction.

## Technical Innovations

### FTS5 Query Sanitization

```python
def _fts_query(q: str) -> str:
    cleaned = []
    for tok in q.replace('"', " ").split():
        tok = "".join(ch for ch in tok if ch.isalnum() or ch in "_-")
        if tok:
            cleaned.append(tok + "*")
    return " OR ".join(cleaned) if cleaned else '""'
```

**Problem:** User input like `"test's file: /tmp"` crashes FTS5 MATCH.

**Solution:** Strip special chars, preserve `_` and `-`, add wildcards, OR-join.

### Skill Frontmatter Parsing

```python
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
```

**Problem:** YAML libs are heavy dependencies; skills should be simple.

**Solution:** Regex-based frontmatter + fallback to filename/first-line.

### Learning Loop Retro Prompt Engineering

Added explicit guidance to prevent junk facts:

```
## What to EXCLUDE from facts (ephemeral / one-off):
- File paths from one-off questions (e.g., "csv_file_path: /tmp/x.csv")
- Counts or statistics that will change (e.g., "python_files_count: 7")
- Temporary state (e.g., "server_is_running", "current_directory")
```

**Problem:** LLM saves transient details as facts (observed in live testing).

**Solution:** Few-shot-style exclusion examples in system prompt.

## Empirical Validation

### Live Test (3 turns, qwen3-coder-plus)

**Turn 1 — Preference capture:**
> User: "Remember that I always use httpx instead of requests, and my server runs on Hetzner CX22."

**Result:**
- ✓ Saved `(preference) httpx_over_requests`
- ✓ Saved `(env) server_hetzner_cx22`
- ✗ Hallucinated "I already remember this" (first turn bug)

**Turn 2 — Tool use:**
> User: "How many python files are in the obektclaw/ directory? Use a tool."

**Result:**
- ✓ Called `list_files`, answered correctly (7 files)
- ✗ Saved ephemeral fact `obektclaw_dir_python_files_count` (junk)

**Turn 3 — Skill recall:**
> User: "I have a CSV file at /tmp/x.csv. Walk me through importing it."

**Result:**
- ✓ Noticed file doesn't exist, asked to verify
- ✗ Didn't auto-load `csv-to-database` skill (prompt weakness)
- ✓ Saved `csv_file_path` (junk — one-off detail)

**End state:**
- 4 facts (2 good, 2 junk)
- 1 user model layer (misclassified)
- 3 bundled skills (no new skills created)

**Lessons learned:**
1. Learning Loop needs exclusion examples (implemented)
2. User model layers need descriptions (implemented)
3. Skills should always be listed in system prompt (implemented)

## Comparison to Alternatives

| Feature | obektclaw | LangChain | AutoGen | CrewAI |
|---------|-------------|-----------|---------|--------|
| Self-improving skills | ✓ (markdown) | ✗ | ✗ | ✗ |
| 3-layer memory | ✓ | △ (custom) | ✗ | ✗ |
| Learning Loop | ✓ (every turn) | ✗ | ✗ | ✗ |
| MCP support | ✓ | ✗ | ✗ | ✗ |
| FTS5 recall | ✓ | ✗ (vectors) | ✗ | ✗ |
| Lines of code | ~2,900 | ~100k+ | ~50k+ | ~20k+ |
| Dependencies | 4 | 50+ | 30+ | 20+ |
| Editable skills | ✓ (vim) | ✗ | ✗ | ✗ |
| Git-trackable memory | △ (skills) | ✗ | ✗ | ✗ |

## Why This is Novel

1. **Complete implementation of the Hermes thesis** in minimal code
2. **Markdown skills on disk** (not DB, not code)
3. **FTS5-only recall** (no embeddings dependency)
4. **12-layer user model** (forced abstraction)
5. **Fire-and-forget Learning Loop** (every turn, cheap)
6. **MCP auto-load** (plug-and-play tools)
7. **Self-documenting UX** (no external docs needed)

## How Anyone Can Use It

### For Developers

```bash
git clone <repo> obektclaw
cd obektclaw
pip install -r requirements.txt
python -m obektclaw chat
```

### For Extending

- Add tool: 20 lines in `obektclaw/tools/`
- Add gateway: 100 lines in `obektclaw/gateways/`
- Customize skills: `vim ~/.obektclaw/skills/`

### For Deploying

- Personal VPS (your privileges, your data)
- Any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, Dashscope)
- Optional: Telegram bot, MCP servers

## Future Research Directions

1. **Long-horizon evaluation:** Does agent actually improve over 50+ turns?
2. **Embeddings vs FTS5:** When does semantic recall beat lexical?
3. **Multi-agent delegation:** Fan-out sub-agents safely?
4. **Memory hygiene:** Auto-expiry + contradiction detection
5. **Sandboxed execution:** Opt-in security for untrusted users

## Conclusion

obektclaw proves that a **self-improving agent harness** can be built in ~2,900 lines without heavy dependencies. The novelty is not in individual components (FTS5, markdown, ReAct loop are all known) but in their **integration into a coherent, minimal system** that embodies the Hermes thesis: **the agent weaves its own harness**.
