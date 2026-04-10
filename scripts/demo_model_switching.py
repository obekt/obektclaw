#!/usr/bin/env python3
"""Demo: Model switching and context window detection.

Run this to see the new model configuration features in action.
No LLM API calls required — everything is local.
"""
import os
import sys
import tempfile
from pathlib import Path

# Add parent dir to path so we can import obektclaw
sys.path.insert(0, str(Path(__file__).parent.parent))

from obektclaw.model_context import (
    get_context_window,
    guess_context_window,
    list_known_models,
    save_user_model_override,
    load_user_model_overrides,
)


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_section(title: str):
    print(f"\n── {title} ──")


def demo_detection():
    """Show context window detection in action."""
    print_header("Context Window Detection Demo")
    
    test_models = [
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "claude-3-5-sonnet-20241022",
        "qwen3-coder-plus",
        "qwen3-235b-a22b",
        "llama-3.1-70b-instruct",
        "gemini-2.0-flash",
        "deepseek-chat",
        "unknown-custom-model",
    ]
    
    print_section("Built-in Detection")
    print(f"{'Model':<40} {'Context Window':>15}")
    print(f"{'-' * 40} {'-' * 15}")
    
    for model in test_models:
        window = guess_context_window(model)
        print(f"{model:<40} {window:>15,}")
    
    print(f"\nDefault fallback for unknown models: 128,000 tokens")


def demo_user_overrides():
    """Show how user overrides work."""
    print_header("User Override Demo")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        home = Path(tmpdir)
        
        print_section("Step 1: Save custom model mappings")
        custom_models = {
            "my-fine-tuned-model": 45000,
            "internal-llm-v2": 64000,
        }
        
        for model, window in custom_models.items():
            save_user_model_override(home, model, window)
            print(f"  ✓ Saved: {model} → {window:,} tokens")
        
        print_section("Step 2: Load them back")
        overrides = load_user_model_overrides(home)
        for model, window in overrides.items():
            print(f"  ✓ Loaded: {model} → {window:,} tokens")
        
        print_section("Step 3: Priority check (user override > built-in)")
        # Override a known model
        save_user_model_override(home, "gpt-4o", 256000)
        
        builtin_window = guess_context_window("gpt-4o")
        user_window = get_context_window("gpt-4o", home)
        
        print(f"  Built-in detection:      {builtin_window:,} tokens")
        print(f"  With user override:      {user_window:,} tokens")
        print(f"  ✅ User override wins!")


def demo_list_models():
    """Show the full model list."""
    print_header("Known Models")
    
    models = list_known_models()
    
    print(f"Total: {len(models)} models/patterns\n")
    
    # Group by context window size
    by_size = {}
    for m in models:
        size = m["context_window"]
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(m["name"])
    
    # Print from largest to smallest
    for size in sorted(by_size.keys(), reverse=True):
        print(f"\n{size:>10,} tokens:")
        for name in sorted(by_size[size]):
            print(f"  • {name}")


def demo_summary():
    """Print usage summary."""
    print_header("How to Use")
    
    print("In the CLI (while chatting):")
    print("  /model                  — Show current model")
    print("  /model gpt-4o           — Switch model (auto-detect)")
    print("  /model my-model 64000   — Switch with custom context")
    print("  /model list             — Show all known models")
    print()
    print("Programmatically:")
    print("  agent.switch_model('gpt-4o', context_window=128000)")
    print()
    print("Configuration:")
    print("  Edit ~/.obektclaw/models.json:")
    print('  {"my-model": 64000, "another-model": 128000}')
    print()
    print("Environment variable:")
    print("  export OBEKTCLAW_CONTEXT_WINDOW=128000")


if __name__ == "__main__":
    demo_detection()
    demo_user_overrides()
    demo_list_models()
    demo_summary()
    
    print("\n" + "=" * 60)
    print("  ✅ All demos completed successfully!")
    print("=" * 60 + "\n")
