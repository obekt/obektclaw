"""Model context window registry and detection utilities.

Centralized mapping of model names/patterns to context window sizes.
This is the industry-standard approach (used by LiteLLM, LangChain, etc.)
since OpenAI-compatible APIs don't expose context metadata.

To add a new model:
  1. Add an exact match to CONTEXT_WINDOW_EXACT if the model name is precise
  2. Add a pattern match to CONTEXT_WINDOW_PATTERNS for fuzzy matching
  3. Or create a ~/.obektclaw/models.json file for user-specific models

Context windows can also be overridden via OBEKTCLAW_CONTEXT_WINDOW env var.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ── Exact model name → context window ─────────────────────────────────────
# For models with precise names (e.g. "gpt-4o-2024-05-13", "qwen3-coder-plus")
CONTEXT_WINDOW_EXACT: dict[str, int] = {
    # OpenAI GPT-4o variants
    "gpt-4o": 128_000,
    "gpt-4o-2024-05-13": 128_000,
    "gpt-4o-2024-08-06": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o-mini-2024-07-18": 128_000,
    
    # OpenAI GPT-4 Turbo
    "gpt-4-turbo": 128_000,
    "gpt-4-turbo-2024-04-09": 128_000,
    "gpt-4-turbo-preview": 128_000,
    "gpt-4-0125-preview": 128_000,
    "gpt-4-1106-preview": 128_000,
    
    # OpenAI GPT-4 (legacy)
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-4-0613": 8_192,
    "gpt-4-32k-0613": 32_768,
    
    # OpenAI GPT-3.5
    "gpt-3.5-turbo": 16_385,
    "gpt-3.5-turbo-0125": 16_385,
    "gpt-3.5-turbo-1106": 16_385,
    "gpt-3.5-turbo-instruct": 4_096,
    
    # Anthropic Claude 3
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-sonnet-20240620": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    
    # Anthropic Claude 2
    "claude-2.1": 200_000,
    "claude-2.0": 100_000,
    
    # Anthropic Claude (generic)
    "claude-instant-1.2": 100_000,
    
    # Qwen / DashScope
    "qwen3-coder-plus": 32_768,
    "qwen3-235b-a22b": 32_768,
    "qwen3-72b": 32_768,
    "qwen3-32b": 32_768,
    "qwen3-14b": 32_768,
    "qwen3-7b": 32_768,
    "qwen-plus": 32_768,
    "qwen-turbo": 32_768,
    "qwen-max": 32_768,
    "qwen2.5-72b-instruct": 32_768,
    "qwen2.5-coder-32b-instruct": 32_768,
    
    # DeepSeek
    "deepseek-chat": 128_000,
    "deepseek-coder": 128_000,
    "deepseek-v3": 128_000,
    "deepseek-r1": 128_000,
    
    # Meta Llama
    "llama-3.3-70b-instruct": 128_000,
    "llama-3.1-405b-instruct": 128_000,
    "llama-3.1-70b-instruct": 128_000,
    "llama-3.1-8b-instruct": 128_000,
    "llama-3.2-90b-vision-instruct": 128_000,
    "llama-3.2-11b-vision-instruct": 128_000,
    "llama-3-70b-instruct": 8_192,
    "llama-3-8b-instruct": 8_192,
    
    # Mistral
    "mistral-large-2411": 128_000,
    "mistral-large-2407": 128_000,
    "mistral-small-2402": 32_000,
    "mistral-medium": 32_000,
    "mistral-open-mixtral-8x22b": 64_000,
    "open-mixtral-8x7b": 32_000,
    "open-mistral-nemo": 128_000,
    
    # Google Gemini
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.0-flash-lite": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-flash-8b": 1_048_576,
    "gemini-pro": 32_760,
    
    # Google Gemma
    "gemma-2-27b-it": 8_192,
    "gemma-2-9b-it": 8_192,
    "gemma-7b-it": 8_192,
    
    # Cohere
    "command-r-plus": 128_000,
    "command-r": 128_000,
    "command-r7b": 128_000,
    "command": 4_096,
    
    # AI21
    "jamba-1.5-large": 256_000,
    "jamba-1.5-mini": 256_000,
    "jamba-instruct": 256_000,
}


# ── Pattern matching (substring → context window) ─────────────────────────
# Checked in order; first match wins. More specific patterns should come first.
CONTEXT_WINDOW_PATTERNS: list[tuple[str, int]] = [
    # OpenAI
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4-32k", 32_768),
    ("gpt-4", 8_192),
    ("gpt-3.5-turbo-instruct", 4_096),
    ("gpt-3.5", 16_385),
    
    # Anthropic
    ("claude-3.5-sonnet", 200_000),
    ("claude-3-5-sonnet", 200_000),
    ("claude-3", 200_000),
    ("claude-2.1", 200_000),
    ("claude-2", 100_000),
    ("claude-instant", 100_000),
    ("claude", 200_000),
    
    # Qwen
    ("qwen3", 32_768),
    ("qwen2.5", 32_768),
    ("qwen2", 32_768),
    ("qwen-plus", 32_768),
    ("qwen-turbo", 32_768),
    ("qwen-max", 32_768),
    ("qwen", 32_000),
    
    # DeepSeek
    ("deepseek", 128_000),
    
    # Llama
    ("llama-3.3", 128_000),
    ("llama-3.2", 128_000),
    ("llama-3.1", 128_000),
    ("llama3", 8_192),
    ("llama-3", 8_192),
    
    # Mistral / Mixtral
    ("mistral-large", 128_000),
    ("mistral-small", 32_000),
    ("mistral-nemo", 128_000),
    ("mistral", 32_000),
    ("mixtral", 32_000),
    
    # Gemini
    ("gemini-2.0", 1_048_576),
    ("gemini-1.5-pro", 2_097_152),
    ("gemini-1.5", 1_048_576),
    ("gemini-pro", 32_760),
    ("gemini", 32_760),
    
    # Gemma
    ("gemma-2", 8_192),
    ("gemma", 8_192),
    
    # Cohere
    ("command-r", 128_000),
    ("command", 4_096),
    
    # AI21
    ("jamba", 256_000),
]

# Default fallback if no model matches
DEFAULT_CONTEXT_WINDOW = 128_000


def guess_context_window(model: str) -> int:
    """Guess the context window size for a model name.
    
    Strategy:
      1. Check exact match first (case-insensitive)
      2. Check pattern matches (substring search, case-insensitive)
      3. Return DEFAULT_CONTEXT_WINDOW
    
    Args:
        model: Model name string (e.g. "gpt-4o-2024-05-13", "qwen3-coder-plus")
    
    Returns:
        Context window size in tokens.
    """
    m = model.lower().strip()
    
    # 1. Exact match
    if m in CONTEXT_WINDOW_EXACT:
        return CONTEXT_WINDOW_EXACT[m]
    
    # 2. Pattern match
    for pattern, size in CONTEXT_WINDOW_PATTERNS:
        if pattern.lower() in m:
            return size
    
    # 3. Default
    return DEFAULT_CONTEXT_WINDOW


def list_known_models() -> list[dict[str, Any]]:
    """Return a sorted list of all known models with their context windows.
    
    Returns:
        List of dicts with keys: name, context_window, source ('exact' or 'pattern')
    """
    models = []
    
    # Add exact models
    for name, size in CONTEXT_WINDOW_EXACT.items():
        models.append({"name": name, "context_window": size, "source": "exact"})
    
    # Add pattern models (deduplicate by pattern)
    seen_patterns = set()
    for pattern, size in CONTEXT_WINDOW_PATTERNS:
        if pattern not in seen_patterns:
            models.append({"name": f"*{pattern}*", "context_window": size, "source": "pattern"})
            seen_patterns.add(pattern)
    
    # Sort by context window (largest first), then by name
    models.sort(key=lambda m: (-m["context_window"], m["name"]))
    return models


def load_user_model_overrides(home_dir: Path) -> dict[str, int]:
    """Load user-defined model context windows from ~/.obektclaw/models.json.
    
    Expected format:
    {
        "my-custom-model": 32768,
        "another-model": 128000
    }
    
    Args:
        home_dir: Path to OBEKTCLAW_HOME directory.
    
    Returns:
        Dict mapping model names to context window sizes.
    """
    models_file = home_dir / "models.json"
    if not models_file.exists():
        return {}
    
    try:
        with open(models_file) as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            return {}
        
        # Validate all values are ints
        return {
            str(k).lower(): int(v)
            for k, v in data.items()
            if isinstance(v, (int, float))
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def save_user_model_override(home_dir: Path, model: str, context_window: int) -> None:
    """Save or update a user-defined model context window.
    
    Args:
        home_dir: Path to OBEKTCLAW_HOME directory.
        model: Model name.
        context_window: Context window size in tokens.
    """
    models_file = home_dir / "models.json"
    
    # Load existing
    overrides = {}
    if models_file.exists():
        try:
            with open(models_file) as f:
                overrides = json.load(f)
        except (json.JSONDecodeError, ValueError):
            overrides = {}
    
    if not isinstance(overrides, dict):
        overrides = {}
    
    # Update
    overrides[model.lower()] = context_window
    
    # Save
    models_file.parent.mkdir(parents=True, exist_ok=True)
    with open(models_file, "w") as f:
        json.dump(overrides, f, indent=2)
        f.write("\n")


def get_context_window(model: str, home_dir: Path | None = None) -> int:
    """Get context window for a model, checking user overrides first.
    
    Priority:
      1. User-defined overrides in ~/.obektclaw/models.json
      2. Built-in exact match
      3. Built-in pattern match
      4. DEFAULT_CONTEXT_WINDOW
    
    Args:
        model: Model name string.
        home_dir: Path to OBEKTCLAW_HOME (optional).
    
    Returns:
        Context window size in tokens.
    """
    m = model.lower().strip()
    
    # 1. Check user overrides
    if home_dir:
        user_overrides = load_user_model_overrides(home_dir)
        if m in user_overrides:
            return user_overrides[m]
    
    # 2-4. Use built-in lookup
    return guess_context_window(model)
