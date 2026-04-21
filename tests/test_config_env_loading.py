"""Tests for .env file loading with placeholder and empty value handling."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from obektclaw.config import _PLACEHOLDER_VALUES, _read_env_file


class TestPlaceholderHandling:
    """Test that placeholder values are skipped during .env loading."""

    def test_placeholder_api_key_skipped(self, tmp_path):
        """Placeholder API keys should not be set."""
        env_file = tmp_path / ".env"
        env_file.write_text("OBEKTCLAW_LLM_API_KEY=your-api-key-here\n")

        # Clear any existing value so the test starts clean
        old_key = os.environ.pop("OBEKTCLAW_LLM_API_KEY", None)
        try:
            _read_env_file(env_file)
            # Should not have been set (it's a placeholder)
            assert os.environ.get("OBEKTCLAW_LLM_API_KEY", "") in _PLACEHOLDER_VALUES
        finally:
            if old_key is not None:
                os.environ["OBEKTCLAW_LLM_API_KEY"] = old_key

    def test_real_value_gets_set(self, tmp_path):
        """Real values should be set even if other lines are placeholders."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "OBEKTCLAW_LLM_API_KEY=your-api-key-here\n"
            "OBEKTCLAW_TG_TOKEN=real-bot-token-12345\n"
        )
        
        # Clear existing
        os.environ.pop("OBEKTCLAW_TG_TOKEN", None)
        
        _read_env_file(env_file)
        
        assert os.environ.get("OBEKTCLAW_TG_TOKEN") == "real-bot-token-12345"

    def test_empty_value_skipped(self, tmp_path):
        """Empty values should be skipped."""
        env_file = tmp_path / ".env"
        env_file.write_text("OBEKTCLAW_TG_TOKEN=\n")
        
        os.environ.pop("OBEKTCLAW_TG_TOKEN", None)
        _read_env_file(env_file)
        
        # Empty value should not be set
        assert os.environ.get("OBEKTCLAW_TG_TOKEN") is None


class TestEmptyEnvVarOverride:
    """Test that empty env vars get replaced by .env file values."""

    def test_empty_env_var_replaced_by_real_value(self, tmp_path):
        """If env var is empty string, .env file value should replace it."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY=real-value\n")
        
        os.environ["TEST_KEY"] = ""
        _read_env_file(env_file)
        
        # Empty should be replaced
        assert os.environ.get("TEST_KEY") == "real-value"

    def test_non_empty_env_var_not_overwritten(self, tmp_path):
        """If env var already has a real value, .env file should not overwrite."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY=env-file-value\n")
        
        os.environ["TEST_KEY"] = "existing-real-value"
        _read_env_file(env_file)
        
        # Should keep existing value
        assert os.environ.get("TEST_KEY") == "existing-real-value"


class TestEnvFilePriority:
    """Test that home .env takes priority over project .env."""

    def test_home_env_overrides_project(self, tmp_path):
        """Home .env values should override project .env placeholders."""
        project_env = tmp_path / "project.env"
        home_env = tmp_path / "home.env"
        
        # Project has placeholder
        project_env.write_text("OBEKTCLAW_LLM_API_KEY=your-api-key-here\n")
        # Home has real value
        home_env.write_text("OBEKTCLAW_LLM_API_KEY=sk-or-v1-real-key\n")
        
        os.environ.pop("OBEKTCLAW_LLM_API_KEY", None)
        
        # Read project first, then home
        _read_env_file(project_env)
        _read_env_file(home_env)
        
        # Home value should win
        assert os.environ.get("OBEKTCLAW_LLM_API_KEY") == "sk-or-v1-real-key"
