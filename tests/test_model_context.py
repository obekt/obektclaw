"""Tests for model context window detection and switching."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from obektclaw.model_context import (
    CONTEXT_WINDOW_EXACT,
    CONTEXT_WINDOW_PATTERNS,
    DEFAULT_CONTEXT_WINDOW,
    get_context_window,
    guess_context_window,
    list_known_models,
    load_user_model_overrides,
    save_user_model_override,
)


class TestGuessContextWindow:
    """Test the built-in context window detection."""

    def test_exact_matches(self):
        """Test exact model name matches."""
        assert guess_context_window("gpt-4o") == 128_000
        assert guess_context_window("gpt-4o-2024-05-13") == 128_000
        assert guess_context_window("gpt-4") == 8_192
        assert guess_context_window("qwen3-coder-plus") == 32_768
        assert guess_context_window("claude-3-5-sonnet-20241022") == 200_000

    def test_pattern_matches(self):
        """Test fuzzy pattern matching."""
        # Qwen variants
        assert guess_context_window("qwen3-235b-a22b") == 32_768
        assert guess_context_window("qwen2.5-72b-instruct") == 32_768
        assert guess_context_window("qwen-plus") == 32_768
        
        # Llama variants
        assert guess_context_window("llama-3.1-70b-instruct") == 128_000
        assert guess_context_window("llama-3.2-90b-vision-instruct") == 128_000
        assert guess_context_window("llama3-70b") == 8_192
        
        # Claude variants
        assert guess_context_window("claude-3-haiku") == 200_000
        assert guess_context_window("claude-2.1") == 200_000
        
        # DeepSeek
        assert guess_context_window("deepseek-chat") == 128_000
        assert guess_context_window("deepseek-coder") == 128_000
        
        # Gemini
        assert guess_context_window("gemini-2.0-flash") == 1_048_576
        assert guess_context_window("gemini-1.5-pro") == 2_097_152

    def test_case_insensitive(self):
        """Test that matching is case-insensitive."""
        assert guess_context_window("GPT-4O") == 128_000
        assert guess_context_window("Qwen3-Coder-Plus") == 32_768
        assert guess_context_window("CLAUDE-3-SONNET") == 200_000

    def test_default_fallback(self):
        """Test unknown models get default context window."""
        assert guess_context_window("unknown-model-xyz") == DEFAULT_CONTEXT_WINDOW

    def test_specific_patterns_take_precedence(self):
        """Test that more specific patterns match before generic ones."""
        # gpt-4-turbo should match before gpt-4
        assert guess_context_window("gpt-4-turbo-preview") == 128_000
        
        # gemini-1.5-pro should match before gemini-1.5
        assert guess_context_window("gemini-1.5-pro-latest") == 2_097_152


class TestUserOverrides:
    """Test user-defined model context windows."""

    def test_save_and_load_override(self):
        """Test saving and loading user overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            
            # Save an override
            save_user_model_override(home, "my-custom-model", 64_000)
            
            # Check file was created
            models_file = home / "models.json"
            assert models_file.exists()
            
            # Check content
            with open(models_file) as f:
                data = json.load(f)
            assert data["my-custom-model"] == 64_000
            
            # Load it back
            overrides = load_user_model_overrides(home)
            assert overrides["my-custom-model"] == 64_000

    def test_get_context_window_with_user_override(self):
        """Test that user overrides take priority."""
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            
            # User overrides a known model
            save_user_model_override(home, "gpt-4o", 256_000)
            
            # Should use user override, not built-in
            assert get_context_window("gpt-4o", home) == 256_000
            
            # Without home_dir, should use built-in
            assert get_context_window("gpt-4o") == 128_000

    def test_invalid_models_file(self):
        """Test graceful handling of corrupted models.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            models_file = home / "models.json"
            
            # Write invalid JSON
            models_file.write_text("not valid json{{{")
            
            # Should return empty dict, not crash
            overrides = load_user_model_overrides(home)
            assert overrides == {}

    def test_update_existing_override(self):
        """Test updating an existing override preserves other models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            
            save_user_model_override(home, "model-a", 32_000)
            save_user_model_override(home, "model-b", 64_000)
            
            overrides = load_user_model_overrides(home)
            assert overrides["model-a"] == 32_000
            assert overrides["model-b"] == 64_000
            
            # Update model-a
            save_user_model_override(home, "model-a", 128_000)
            
            overrides = load_user_model_overrides(home)
            assert overrides["model-a"] == 128_000
            assert overrides["model-b"] == 64_000  # Still there


class TestListKnownModels:
    """Test the list_known_models function."""

    def test_returns_non_empty_list(self):
        """Test that we have at least some known models."""
        models = list_known_models()
        assert len(models) > 50  # Should have lots of models

    def test_model_structure(self):
        """Test that each model dict has the right keys."""
        models = list_known_models()
        for m in models:
            assert "name" in m
            assert "context_window" in m
            assert "source" in m
            assert m["source"] in ("exact", "pattern")
            assert isinstance(m["context_window"], int)
            assert m["context_window"] > 0

    def test_sorted_by_context_window(self):
        """Test that models are sorted by context window (largest first)."""
        models = list_known_models()
        windows = [m["context_window"] for m in models]
        # Should be roughly sorted (some ties are ok)
        for i in range(len(windows) - 1):
            assert windows[i] >= windows[i + 1]


class TestBuiltInCoverage:
    """Test that the built-in coverage is comprehensive."""

    def test_exact_has_popular_models(self):
        """Test that popular production models are in exact matches."""
        popular_models = [
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
            "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307",
            "qwen3-coder-plus", "qwen2.5-72b-instruct",
            "deepseek-chat", "deepseek-coder",
            "llama-3.1-70b-instruct",
            "gemini-1.5-pro", "gemini-2.0-flash",
        ]
        for model in popular_models:
            assert model in CONTEXT_WINDOW_EXACT, f"Missing: {model}"

    def test_patterns_cover_major_families(self):
        """Test that pattern matching covers major model families."""
        test_cases = [
            ("qwen3-something", 32_768),
            ("llama-3.1-something", 128_000),
            ("claude-3-something", 200_000),
            ("gemini-1.5-something", 1_048_576),
            ("deepseek-something", 128_000),
        ]
        for model, expected in test_cases:
            assert guess_context_window(model) == expected, f"Failed for {model}"
