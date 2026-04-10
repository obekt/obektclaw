# Model Configuration & Context Management

## Overview

obektclaw supports **runtime model switching**, **automatic context window detection**, and **intelligent conversation compaction**. You can change models mid-conversation without losing your session, memory, or skills, and the agent automatically manages context pressure.

## Changing Models During a Chat Session

### CLI Slash Command

While chatting, use the `/model` command:

```bash
# Show current model and detected context window
/model

# Switch to a different model (auto-detect context window)
/model gpt-4o

# Switch with explicit context window override
/model gpt-4o 128000

# List all known models with their context windows
/model list
```

### Programmatic API

```python
from obektclaw.agent import Agent

agent = Agent(config=config, store=store, skills=skills)

# Switch model
result = agent.switch_model(
    model="gpt-4o",
    fast_model="gpt-4o-mini",  # optional, defaults to main model
    context_window=128000,      # optional, 0 = auto-detect
    persist=True,               # optional, saves to models.json
)

print(result)
# {
#     "model": "gpt-4o",
#     "fast_model": "gpt-4o-mini",
#     "context_window": 128000,
#     "was_overridden": False
# }
```

## Context Window Detection

### How It Works

Since OpenAI-compatible APIs don't expose context window metadata, obektclaw uses a **multi-tier detection system** (industry-standard approach used by LiteLLM, LangChain, etc.):

1. **User overrides** (`~/.obektclaw/models.json`) — your custom mappings
2. **Built-in exact matches** — precise model names (e.g., `qwen3-coder-plus`)
3. **Built-in pattern matching** — substring patterns (e.g., `qwen3-*` → 32k)
4. **Default fallback** — 128,000 tokens

### Adding a New Model

#### Option 1: Use the `/model` Command (Easiest)

```bash
# Switch to a new model with explicit context window
/model my-custom-model 64000

# This automatically saves it to ~/.obektclaw/models.json
```

#### Option 2: Edit `models.json` Manually

Create or edit `~/.obektclaw/models.json`:

```json
{
  "my-custom-model": 64000,
  "another-model": 128000
}
```

#### Option 3: Add to the Built-in Registry

Edit `obektclaw/model_context.py`:

```python
# For exact model names
CONTEXT_WINDOW_EXACT["qwen3-coder-plus"] = 32_768

# For pattern matching (checked in order, first match wins)
CONTEXT_WINDOW_PATTERNS = [
    ("qwen3", 32_768),  # matches qwen3-*, qwen3.5-*, etc.
    # ... more patterns
]
```

### Environment Variable Override

Set `OBEKTCLAW_CONTEXT_WINDOW` to force a specific context window for all models:

```bash
export OBEKTCLAW_CONTEXT_WINDOW=128000
```

This takes priority over auto-detection but **not** over user-defined overrides in `models.json`.

## Supported Models (Partial List)

### OpenAI
- `gpt-4o`, `gpt-4o-mini` — 128k
- `gpt-4-turbo` — 128k
- `gpt-4` — 8k
- `gpt-3.5-turbo` — 16k

### Anthropic Claude
- Claude 3.5 Sonnet — 200k
- Claude 3 (Opus, Sonnet, Haiku) — 200k
- Claude 2.1 — 200k
- Claude 2.0 — 100k

### Qwen / DashScope
- `qwen3-*` — 32k
- `qwen2.5-*` — 32k
- `qwen-plus`, `qwen-turbo`, `qwen-max` — 32k

### Meta Llama
- `llama-3.1-*`, `llama-3.2-*`, `llama-3.3-*` — 128k
- `llama-3-*` — 8k

### Google Gemini
- `gemini-2.0-*` — 1M
- `gemini-1.5-pro` — 2M
- `gemini-1.5-*` — 1M

### DeepSeek
- `deepseek-chat`, `deepseek-coder`, `deepseek-v3`, `deepseek-r1` — 128k

### Others
- Mistral/Mixtral — 32k-128k
- Cohere Command-R — 128k
- AI21 Jamba — 256k
- Google Gemma — 8k

**Full list**: Use `/model list` in the CLI to see all 100+ supported models.

## Model Switching Behavior

When you switch models:

1. ✅ **New LLM client is created** with the new model name
2. ✅ **Context window is updated** (auto-detected or overridden)
3. ✅ **Session memory is preserved** — conversation continues seamlessly
4. ✅ **Persistent memory and skills unchanged**
5. ✅ **System message logged** — "Model switched from X → Y"
6. ✅ **Optional persistence** — saves to `models.json` for future sessions

### What Doesn't Change

- Current conversation history
- Persistent facts
- User model traits
- Skill files
- Session ID

### What Does Change

- The model used for reasoning and tool calls
- The model used for the Learning Loop (fast model)
- The context window size (affects truncation behavior)

## Troubleshooting

### "Invalid context window" Error

Make sure you're passing a valid integer:

```bash
# ✅ Correct
/model gpt-4o 128000

# ❌ Wrong (no comma in numbers)
/model gpt-4o 128,000
```

### Model Not Recognized

If your model isn't in the built-in registry:

1. Use `/model my-model 32000` to set it explicitly
2. Or add it to `~/.obektclaw/models.json`
3. Or submit a PR to add it to `model_context.py`

### Context Window Too Small/Large

Override it explicitly:

```bash
# Force a specific context window
/model qwen3-coder-plus 64000

# Or set globally in .env
echo "OBEKTCLAW_CONTEXT_WINDOW=64000" >> ~/.obektclaw/.env
```

## Testing Your Configuration

```python
# Quick test without starting a full session
from obektclaw.model_context import get_context_window, list_known_models

# Check detected context window
window = get_context_window("qwen3-coder-plus")
print(f"Context window: {window:,} tokens")

# List all known models
models = list_known_models()
for m in models[:10]:
    print(f"{m['name']}: {m['context_window']:,} ({m['source']})")
```

## Architecture

```
model_context.py
├── CONTEXT_WINDOW_EXACT      # Precise model names → size
├── CONTEXT_WINDOW_PATTERNS   # Substring patterns → size (ordered)
├── DEFAULT_CONTEXT_WINDOW    # Fallback (128k)
├── guess_context_window()    # Built-in detection logic
├── get_context_window()      # Public API with user override support
├── load_user_model_overrides()  # Load ~/.obektclaw/models.json
├── save_user_model_override()   # Save to ~/.obektclaw/models.json
└── list_known_models()          # Return sorted list of all known models
```

**Priority order**: User overrides → Exact match → Pattern match → Default

## Future Enhancements

- [ ] Auto-detect context window from `/v1/models` endpoint (OpenRouter-style extensions)
- [ ] Optional LiteLLM integration for live model metadata
- [ ] Per-session vs persistent model switching
- [ ] Context window usage analytics and warnings

## Context Compaction

### What is Compaction?

**Context compaction** is when the LLM summarizes older conversation history instead of just dropping it. This preserves important context (goals, decisions, user preferences) while freeing up context window space.

**Compaction vs Truncation:**
- **Truncation** (old approach): Mechanically drops oldest messages when context gets full → causes "amnesia"
- **Compaction** (new approach): Uses LLM to summarize old messages → preserves semantic meaning

### Automatic Compaction

obektclaw automatically compacts conversation when context pressure reaches **85%**:

1. LLM (fast model, to save cost) summarizes older conversation turns
2. Summary preserves: goals, decisions, user preferences, key context
3. Old messages are deleted, replaced with compact summary
4. Recent ~6 turns kept raw for immediate context continuity
5. Agent continues with fresh context space

**This is transparent to the user** — you'll see a brief "compacting context..." status message.

### Manual Compaction

You can also trigger compaction anytime:

```bash
/compact
```

This is useful when:
- You want to save tokens before a long task
- You're switching topics and want a clean slate
- You want to verify the agent "remembers" key points

### Compaction Behavior

**What's preserved:**
- ✅ User goals and requests
- ✅ Key decisions and reasoning
- ✅ User preferences (also saved to persistent memory via Learning Loop)
- ✅ Important file/environment context
- ✅ Recent ~6 turns (raw, for continuity)

**What's summarized:**
- Older conversation turns (beyond last 6)
- Resolved debugging steps
- Repetitive exchanges

**What's omitted:**
- Pleasantries ("hi", "thanks", etc.)
- Redundant clarifications
- Tool output details (facts extracted to memory)

### Compaction Settings

You can adjust compaction behavior by modifying the Agent class constants:

```python
from obektclaw.agent import Agent

# Change auto-compaction threshold (default: 85%)
Agent.COMPACTION_PRESSURE = 0.80  # Compact earlier

# Change how many recent turns to keep raw (default: 6)
Agent.COMPACTION_KEEP_TURNS = 8  # Keep more context

# Change max summary size (default: 1000 tokens)
Agent.COMPACTION_MAX_SUMMARY = 1500  # Allow longer summaries
```

### Compaction Cost

- **Token cost**: ~500-1000 tokens per compaction (uses fast model)
- **Time cost**: ~1-3 seconds for summarization
- **Savings**: Typically saves 2000-5000 tokens by compressing old messages
- **Net benefit**: Positive — more room for new conversation while preserving context

### When Compaction Helps

**Long conversations** (50+ turns): Without compaction, truncation would lose early context. Compaction preserves it as a summary.

**Complex multi-step tasks**: When working on a project over many turns, compaction keeps the overall goals visible even as individual turns get old.

**Model switching**: When you switch to a different model, compaction gives the new model a clean summary instead of raw history.

### Troubleshooting Compaction

**"Conversation too short to compact"**
- Normal for early conversation — compaction needs enough history to summarize
- Keep chatting, it'll trigger automatically at 85% pressure

**"Compaction failed: LLM error"**
- The LLM call to generate the summary failed
- Agent falls back to truncation if pressure > 75%
- Check your LLM connection and try `/compact` again

**Summary seems to miss important details**
- The Learning Loop should have extracted key facts to persistent memory
- Use `/memory` to verify important facts were saved
- You can always manually re-explain context if needed
