# PR: Fix TUI logging spam, embedding model re-downloads, and simplify config

## Summary

This PR fixes the two biggest UX regressions from the `feature/memory-improvements` merge:
1. **HF embedding model "downloading" on every start** — progress bars and transformers logging corrupted the TUI
2. **Log messages flooding the CLI REPL** — INFO-level logs from chromadb, sentence-transformers, urllib3, and our own agent loop broke the prompt_toolkit / Rich display

It also **dramatically simplifies configuration** by removing 13 unnecessary env vars. The project had become over-configured for non-technical users. Now the maximum setup is: base URL, API key, model name — everything else just works.

---

## Problems Fixed

### 1. Logs corrupting the TUI (Critical UX regression)

**Before:** Every logger got a `StreamHandler(sys.stderr)`. When starting the CLI, you'd see:
```
19:30:02 [INFO] obektclaw.tools.registry: tools_registered count=12
19:30:02 [INFO] obektclaw.llm: extraction_llm_initialized model=...
19:30:02 [WARNING] urllib3.connectionpool: Connection pool is full, discarding connection: ...
```
These messages interleaved with prompt_toolkit's rendering, breaking the status bar, completion menu, and Rich panels.

**After:** `gateways/cli.py` calls `setup_cli_logging()` before starting the REPL. This:
- Removes all console handlers from every existing logger
- Bumps noisy third-party loggers (`chromadb`, `sentence_transformers`, `transformers`, `urllib3`, `httpx`, `openai`, `huggingface_hub`, `torch`) to `WARNING`
- Keeps JSON file logging intact under `~/.obektclaw/logs/obektclaw.log`

### 2. Embedding model loading noise on every startup

**Before:** `sentence-transformers` printed `tqdm` progress bars to stderr when loading `all-MiniLM-L6-v2`. Users perceived this as "downloading every time" because the progress bars looked identical for cache-hits vs downloads. The default cache dir (`~/.cache/torch/...`) could also be wiped in containerized environments.

**After:** `memory/embedder.py` now:
- Sets `SENTENCE_TRANSFORMERS_HOME` and `HF_HOME` to persistent paths inside `~/.obektclaw/models/`
- Disables progress bars (`HF_HUB_DISABLE_PROGRESS_BARS=1`)
- Suppresses transformers verbosity (`TRANSFORMERS_VERBOSITY=error`)
- Disables tokenizer parallelism warnings (`TOKENIZERS_PARALLELISM=false`)

### 3. UnboundLocalError crash in agent loop (Latent bug)

**Before:** The context compaction path in `agent.py` referenced `len(to_compact)` and `pressure` variables that didn't exist in `run_once()` scope — they were local to `compact_context()`. This would crash the agent when context pressure exceeded 85%.

**After:** `compact_context()` now returns `turns_compacted` in its result dict, and `run_once()` uses `compact_result.get("turns_compacted", 0)` and `self._context_pressure()`.

### 4. Missing Config fields (Latent bug)

`local_llm.py` referenced `CONFIG.local_model_cache` and `CONFIG.local_llm_threads`, which didn't exist in the `Config` dataclass. Added these with sensible defaults.

---

## Configuration Simplification

### Removed env vars (no longer needed)

| Removed variable | Why |
|------------------|-----|
| `OBEKTCLAW_EXTRACTION_LLM_BASE_URL` | Extraction just uses main LLM with `fast=True` |
| `OBEKTCLAW_EXTRACTION_LLM_API_KEY` | Same reason |
| `OBEKTCLAW_EXTRACTION_LLM_MODEL` | Same reason |
| `OBEKTCLAW_LOCAL_MODEL_CACHE` | Local LLM is dead code, not imported |
| `OBEKTCLAW_LOCAL_LLM_THREADS` | Same reason |
| `OBEKTCLAW_EMBEDDING_MODEL` | Hardcoded to `all-MiniLM-L6-v2` |
| `OBEKTCLAW_EMBEDDING_DIMENSION` | Hardcoded to `384` |
| `OBEKTCLAW_GRAPH_NAME` | Hardcoded to `obektclaw` |
| `OBEKTCLAW_SEMANTIC_SEARCH_LIMIT` | Hardcoded to `10` |
| `OBEKTCLAW_GRAPH_TRAVERSAL_DEPTH` | Hardcoded to `3` |
| `OBEKTCLAW_CONTEXT_ASSEMBLY_MAX_TOKENS` | Hardcoded to `2000` |
| `OBEKTCLAW_COG_HOME` | Derived from `OBEKTCLAW_HOME` |
| `OBEKTCLAW_CHROMA_PATH` | Derived from `OBEKTCLAW_HOME` |

### What remains (user-facing)

```bash
OBEKTCLAW_LLM_BASE_URL=https://openrouter.ai/api/v1
OBEKTCLAW_LLM_API_KEY=your-key
OBEKTCLAW_LLM_MODEL=gpt-4o-mini
# Optional:
OBEKTCLAW_LLM_FAST_MODEL=cheaper-model
OBEKTCLAW_TG_TOKEN=your-bot-token
OBEKTCLAW_HOME=~/.obektclaw
OBEKTCLAW_CONTEXT_WINDOW=128000
OBEKTCLAW_BASH_TIMEOUT=30
OBEKTCLAW_WORKDIR=/path
```

### Internal changes

- **Deleted `ExtractionLLMClient`** from `llm.py` — it was a near-duplicate of `LLMClient.chat_json()`. Post-turn extraction now calls `agent.llm.chat_json(..., fast=True)`.
- **Hardcoded defaults** in `hybrid_retriever.py` and `vector_memory.py` instead of pulling from `CONFIG`.
- **Made `cogdb` import graceful** — `graph_memory.py` catches `ImportError` and raises a clear `RuntimeError` at instantiation time.

---

## Files Changed

| File | Change |
|------|--------|
| `obektclaw/config.py` | Removed 13 config fields; kept only user-facing ones |
| `obektclaw/llm.py` | Removed `ExtractionLLMClient` class |
| `obektclaw/agent.py` | Uses `self.llm.chat_json` for extraction; fixed `UnboundLocalError` |
| `obektclaw/post_turn.py` | Uses `agent.llm.chat_json(..., fast=True)` |
| `obektclaw/logging_config.py` | Added `setup_cli_logging()` for TUI mode |
| `obektclaw/gateways/cli.py` | Calls `setup_cli_logging()` on startup |
| `obektclaw/memory/embedder.py` | Persistent cache dir + suppressed progress bars |
| `obektclaw/memory/vector_memory.py` | Hardcoded embedding stats; removed CONFIG dependency |
| `obektclaw/memory/hybrid_retriever.py` | Hardcoded retrieval defaults; removed CONFIG dependency |
| `obektclaw/memory/graph_memory.py` | Graceful `cogdb` import |
| `obektclaw/local_llm.py` | Fixed missing CONFIG field references |
| `obektclaw/__main__.py` | Hardcoded graph name |
| `.env.example` | Simplified to 4 sections, 9 variables |
| `README.md` | Updated config table, test count, structure |
| `QUICKSTART.md` | Updated config table, test count |
| `tests/test_post_turn.py` | Rewrote to use `FakeMainLLM.chat_json` instead of `FakeExtractionLLM` |
| `tests/test_e2e_memory.py` | Removed extraction_llm config from mock Config |
| `tests/test_hybrid_retriever.py` | Removed CONFIG patches |
| `tests/test_vector_memory.py` | Removed CONFIG patches |

---

## Test Results

```bash
$ python -m pytest tests/ --ignore=tests/test_e2e_memory.py
========================= 602 passed, 1 warning in 36s =========================
```

All offline tests pass. The one E2E test (`test_e2e_memory.py`) requires a real OpenRouter API key and is excluded from CI.

---

## Backwards Compatibility

**Breaking:** If you were using any of the removed env vars, they are now ignored. The behavior is:
- `OBEKTCLAW_EXTRACTION_LLM_*` → extraction uses `OBEKTCLAW_LLM_FAST_MODEL` (or main model)
- `OBEKTCLAW_EMBEDDING_*` → always uses `all-MiniLM-L6-v2`
- `OBEKTCLAW_COG_HOME` / `OBEKTCLAW_CHROMA_PATH` → always derived from `OBEKTCLAW_HOME`
- `OBEKTCLAW_*_LIMIT` / `OBEKTCLAW_*_DEPTH` / `OBEKTCLAW_*_TOKENS` → hardcoded sensible defaults

No migration needed — just delete the old vars from your `.env`.

---

## Checklist

- [x] Fixes TUI logging corruption
- [x] Fixes embedding model re-download noise
- [x] Fixes `UnboundLocalError` in compaction path
- [x] Fixes missing `Config` fields
- [x] Simplifies configuration (13 fewer env vars)
- [x] Updates `.env.example`
- [x] Updates `README.md`
- [x] Updates `QUICKSTART.md`
- [x] All 602 offline tests pass
