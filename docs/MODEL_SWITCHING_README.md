# Model Switching — Quick Start

## Change Models Mid-Conversation

While chatting with obektclaw:

```bash
# See current model
/model

# Switch to a different model
/model gpt-4o

# Switch with custom context window
/model my-custom-model 64000

# List all 114+ supported models
/model list
```

## Add Your Own Models

### Easy Way (Recommended)
Just use the `/model` command with a context window:
```bash
/model my-fine-tuned-llm 45000
```
This automatically saves to `~/.obektclaw/models.json`.

### Manual Way
Edit `~/.obektclaw/models.json`:
```json
{
  "my-model-1": 32000,
  "my-model-2": 128000
}
```

### Global Override
Set in `~/.obektclaw/.env`:
```bash
OBEKTCLAW_CONTEXT_WINDOW=128000
```

## Auto-Detection

obektclaw automatically detects context windows for **114+ models** including:
- OpenAI (GPT-4o, GPT-4, GPT-3.5)
- Anthropic Claude (3.5, 3, 2)
- Qwen (3, 2.5, Plus, Turbo, Max)
- Meta Llama (3.1, 3.2, 3.3)
- Google Gemini (2.0, 1.5)
- DeepSeek (V3, R1, Coder)
- Mistral/Mixtral
- And many more...

Use `/model list` to see the full roster.

## Programmatic API

```python
from obektclaw.agent import Agent

agent = Agent(config=config, store=store, skills=skills)

# Switch model mid-conversation
agent.switch_model(
    model="gpt-4o",
    fast_model="gpt-4o-mini",  # optional
    context_window=128000,      # optional, auto-detect if omitted
    persist=True,               # saves to models.json
)
```

## What's Preserved?

When you switch models:
- ✅ Conversation history
- ✅ Persistent memories
- ✅ User model traits
- ✅ Skill files
- ✅ Session ID

What changes:
- ✅ The LLM used for reasoning
- ✅ The context window size
- ✅ The fast model (Learning Loop)

## Documentation

- **Full guide**: [docs/model-configuration.md](model-configuration.md)
- **Implementation details**: [MODEL_SWITCHING_SUMMARY.md](../MODEL_SWITCHING_SUMMARY.md)
- **Demo**: Run `python3 scripts/demo_model_switching.py`

## Adding New Models to the Built-in Registry

If you want to contribute model mappings to the codebase:

1. Edit `obektclaw/model_context.py`
2. Add exact matches to `CONTEXT_WINDOW_EXACT`
3. Add patterns to `CONTEXT_WINDOW_PATTERNS` (more specific first)
4. Add tests to `tests/test_model_context.py`
5. Submit a PR

For personal/custom models, use `~/.obektclaw/models.json` instead (no code changes needed).
