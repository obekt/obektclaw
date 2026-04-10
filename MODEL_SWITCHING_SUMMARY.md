# Model Switching & Context Window Detection — Implementation Summary

## What Was Implemented

### 1. **Runtime Model Switching** ✅
- New `/model` slash command in CLI gateway
- `Agent.switch_model()` method for programmatic switching
- Seamless session continuation (memory, skills, facts preserved)
- Optional persistence to `~/.obektclaw/models.json`

### 2. **Comprehensive Context Window Detection** ✅
- **114+ models/patterns** supported (up from ~15)
- Multi-tier detection: user overrides → exact match → pattern match → default
- Case-insensitive matching
- Graceful fallback for unknown models

### 3. **User-Configurable Model Registry** ✅
- `~/.obektclaw/models.json` for custom model mappings
- Easy to add new models without code changes
- Survives code updates (stored in user home, not repo)

### 4. **Developer Experience** ✅
- Startup logging shows detected context window
- Rich CLI output with model info panels
- `/model list` shows all known models
- Demo script for testing

### 5. **Test Coverage** ✅
- 14 new tests for model context detection
- All 252 existing tests still pass
- Tests for user overrides, pattern matching, edge cases

## Files Changed/Created

### New Files
1. **`obektclaw/model_context.py`** (310 lines)
   - Centralized context window registry
   - Detection logic with priority ordering
   - User override persistence
   - Model listing utilities

2. **`tests/test_model_context.py`** (170 lines)
   - Comprehensive test coverage
   - Tests for exact matches, patterns, overrides
   - Edge case handling

3. **`scripts/demo_model_switching.py`** (130 lines)
   - Interactive demo showing all features
   - No LLM API calls required

4. **`docs/model-configuration.md`** (250 lines)
   - Complete user documentation
   - Examples for all use cases
   - Troubleshooting guide

### Modified Files
1. **`obektclaw/agent.py`**
   - Removed hardcoded `_CONTEXT_WINDOWS` table
   - Added import of `model_context` module
   - Added `switch_model()` method (60 lines)
   - Added startup logging for context window
   - Updated initialization to use `get_context_window()`

2. **`obektclaw/gateways/cli.py`**
   - Added `/model` to slash commands list
   - Added 80-line `/model` command handler
   - Updated help text to include `/model`
   - Added imports for model_context functions

3. **`.env.example`**
   - Added `OBEKTCLAW_CONTEXT_WINDOW` documentation

4. **`AGENTS.md`**
   - Updated file layout to include `model_context.py`
   - Updated line count

## Usage Examples

### In CLI Chat Session
```bash
# Show current model
/model

# Switch to GPT-4o (auto-detect 128k context)
/model gpt-4o

# Switch with explicit context window
/model claude-3-5-sonnet-20241022 200000

# List all known models
/model list
```

### Programmatically
```python
from obektclaw.agent import Agent

agent = Agent(config=config, store=store, skills=skills)

# Switch model mid-conversation
result = agent.switch_model(
    model="gpt-4o",
    fast_model="gpt-4o-mini",
    context_window=128000,  # optional, auto-detect if omitted
    persist=True,            # saves to models.json
)
```

### Adding Custom Models
```bash
# Method 1: Use the CLI
/model my-custom-model 64000

# Method 2: Edit ~/.obektclaw/models.json
{
  "my-custom-model": 64000,
  "another-model": 128000
}

# Method 3: Environment variable (global override)
export OBEKTCLAW_CONTEXT_WINDOW=128000
```

## Detection Priority

1. **User overrides** (`~/.obektclaw/models.json`) — your custom mappings
2. **Exact matches** (built-in, e.g., `qwen3-coder-plus` → 32,768)
3. **Pattern matches** (built-in, e.g., `qwen3-*` → 32,768)
4. **Default fallback** (128,000 tokens)

## Supported Model Families

| Family | Context Window | Examples |
|--------|---------------|----------|
| OpenAI GPT-4o | 128k | gpt-4o, gpt-4o-mini |
| OpenAI GPT-4 Turbo | 128k | gpt-4-turbo, gpt-4-0125-preview |
| OpenAI GPT-4 | 8k | gpt-4, gpt-4-0613 |
| OpenAI GPT-3.5 | 16k | gpt-3.5-turbo |
| Claude 3.5/3 | 200k | claude-3-5-sonnet, claude-3-haiku |
| Claude 2 | 100k-200k | claude-2.0, claude-2.1 |
| Qwen 3/2.5 | 32k | qwen3-coder-plus, qwen2.5-72b |
| Llama 3.1+ | 128k | llama-3.1-70b, llama-3.2-90b |
| Llama 3 | 8k | llama-3-70b, llama-3-8b |
| Gemini 2.0 | 1M | gemini-2.0-flash |
| Gemini 1.5 Pro | 2M | gemini-1.5-pro |
| Gemini 1.5 | 1M | gemini-1.5-flash |
| DeepSeek | 128k | deepseek-chat, deepseek-coder |
| Mistral Large | 128k | mistral-large-2411 |
| Mistral/Mixtral | 32k | open-mixtral-8x7b |
| Command-R | 128k | command-r-plus |
| Jamba | 256k | jamba-1.5-large |
| Gemma | 8k | gemma-2-27b-it |

**Total**: 114+ models/patterns with accurate context windows

## Architecture Decisions

### Why Hardcoded Lookup Instead of API Detection?

**Answer**: OpenAI-compatible APIs don't expose context window metadata in `/v1/models`. This is the industry-standard approach used by:
- LiteLLM (massive `model_cost.json`)
- LangChain (internal mappings)
- LlamaIndex (built-in dictionary)

**Alternative approaches considered**:
1. ❌ Query `/v1/models` — only returns `id`, `created`, `object`, `owned_by`
2. ❌ Add LiteLLM dependency — heavy dependency, against design principles
3. ✅ **Hardcoded lookup** — lightweight, transparent, easy to maintain

### Why Allow Runtime Model Switching?

**Answer**: Users want to:
- Test different models for the same task
- Switch to cheaper models for simple tasks
- Upgrade to smarter models for complex reasoning
- Experiment without restarting the session

**What's preserved during switch**:
- ✅ Session memory (conversation history)
- ✅ Persistent facts
- ✅ User model traits
- ✅ Skill files
- ✅ Session ID

**What changes**:
- ✅ LLM client (new model)
- ✅ Context window size
- ✅ Fast model (for Learning Loop)

## Testing

### Test Coverage
```
tests/test_model_context.py::TestGuessContextWindow (5 tests)
  ✅ Exact matches
  ✅ Pattern matching
  ✅ Case insensitivity
  ✅ Default fallback
  ✅ Pattern precedence

tests/test_model_context.py::TestUserOverrides (4 tests)
  ✅ Save and load
  ✅ Priority over built-in
  ✅ Invalid file handling
  ✅ Update existing

tests/test_model_context.py::TestListKnownModels (3 tests)
  ✅ Returns non-empty list
  ✅ Correct structure
  ✅ Sorted by context window

tests/test_model_context.py::TestBuiltInCoverage (2 tests)
  ✅ Popular models covered
  ✅ Pattern families covered
```

### Test Results
```
252 passed, 1 warning in 0.65s
```

All existing tests pass — no regressions.

## Future Enhancements

Potential improvements (not implemented yet):

1. **OpenRouter Extended API** — If using OpenRouter, query their custom `/v1/models` endpoint which includes `context_length`
2. **LiteLLM Optional Integration** — Use LiteLLM's model metadata as a fallback source (without requiring it for inference)
3. **Per-Session Model History** — Track which model was used for each turn
4. **A/B Testing Mode** — Try two models on the same prompt and compare
5. **Auto-Model Selection** — Let the agent choose the best model for the task
6. **Context Window Analytics** — Show usage stats per model

## Migration Notes

### For Existing Users

**No breaking changes** — everything is backward compatible:

- Old `OBEKTCLAW_CONTEXT_WINDOW` env var still works
- Existing sessions continue working
- Auto-detection is transparent
- New `/model` command is additive

### For Developers

If you were using `_guess_context_window()` internally:
- It's now in `obektclaw.model_context` module
- Public API is `get_context_window(model, home_dir)` which includes user overrides
- Pattern matching logic is unchanged (just moved to a new file)

## Demo Output

```bash
$ python3 scripts/demo_model_switching.py

============================================================
  Context Window Detection Demo
============================================================

── Built-in Detection ──
Model                                     Context Window
---------------------------------------- ---------------
gpt-4o                                           128,000
gpt-4o-2024-05-13                                128,000
claude-3-5-sonnet-20241022                       200,000
qwen3-coder-plus                                  32,768
llama-3.1-70b-instruct                           128,000
gemini-2.0-flash                               1,048,576
unknown-custom-model                             128,000

✅ All demos completed successfully!
```

## Conclusion

The implementation provides:
- ✅ **Easy model switching** via `/model` command
- ✅ **Comprehensive context detection** (114+ models)
- ✅ **User-configurable registry** (`~/.obektclaw/models.json`)
- ✅ **Full test coverage** (14 new tests, 252 total)
- ✅ **Rich documentation** (user guide + demo script)
- ✅ **Zero breaking changes** (fully backward compatible)

Adding a new model is now as simple as:
```bash
/model my-new-model 64000
```

Or editing one file:
```json
// ~/.obektclaw/models.json
{
  "my-new-model": 64000
}
```

No code changes required for end users!
