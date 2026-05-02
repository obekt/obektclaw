"""Structured logging for obektclaw.

Provides:
- JSON file logs (rotated daily) under $OBEKTCLAW_HOME/logs/obektclaw.log
- Colored console output for CLI mode
- A `get_logger(name)` helper that returns a properly configured logger

Usage:
    from ..logging_config import get_logger
    log = get_logger(__name__)
    log.info("context_window=%d model=%s", 128000, "gpt-4o")
    log.error("MCP failed", exc_info=True)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


# Global flag: console output can be disabled for TUI modes (e.g. CLI REPL)
_console_enabled = True


def _get_home() -> Path:
    return Path(os.environ.get("OBEKTCLAW_HOME") or Path.home() / ".obektclaw").expanduser()


_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


class _JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        # Ensure message is serializable
        msg = record.msg
        if isinstance(msg, dict):
            payload = {**msg, "ts": self.formatTime(record), "lvl": record.levelname, "name": record.name}
            return json.dumps(payload, ensure_ascii=False)
        if isinstance(msg, str):
            payload = {
                "ts": self.formatTime(record),
                "lvl": record.levelname,
                "name": record.name,
                "msg": msg,
            }
            # Attach extra fields if present
            for key in ("event", "session_id", "tool", "model", "error", "gateway", "duration_ms", "tokens"):
                val = getattr(record, key, None)
                if val is not None:
                    payload[key] = val
            if record.exc_info and record.exc_info[0] is not None:
                payload["error"] = self.formatException(record.exc_info)
            return json.dumps(payload, ensure_ascii=False)
        # Fallback for any other msg type (int, Exception, etc.)
        fallback = {
            "ts": self.formatTime(record),
            "lvl": record.levelname,
            "name": record.name,
            "msg": str(msg),
        }
        if record.exc_info and record.exc_info[0] is not None:
            fallback["error"] = self.formatException(record.exc_info)
        return json.dumps(fallback, ensure_ascii=False)


def _setup_file_handler(log_dir: Path) -> logging.FileHandler:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "obektclaw.log"
    handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=7,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_JSONFormatter(datefmt=_DATE_FMT))
    return handler


def _setup_console_handler() -> logging.StreamHandler | None:
    if not _console_enabled:
        return None
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    # Simple colored format for console
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    return handler


def get_logger(name: str) -> logging.Logger:
    """Get a logger for *name*, configured with file + console handlers.

    Handlers are attached once per process. Subsequent calls return the same
    logger with handlers already attached.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    home = _get_home()
    log_dir = home / "logs"

    logger.addHandler(_setup_file_handler(log_dir))
    console_handler = _setup_console_handler()
    if console_handler is not None:
        logger.addHandler(console_handler)

    return logger


def setup_logging(level: str = "INFO") -> None:
    """Attach a console handler to the root logger at *level*.

    Call this early (e.g. from __main__.py) if you want all nested loggers
    to emit to stderr at a given level. The per-module handlers handle
    individual module-level logging; this is a safety net.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        console_handler = _setup_console_handler()
        if console_handler is not None:
            root.addHandler(console_handler)


def setup_cli_logging() -> None:
    """Suppress console logging for interactive TUI mode.

    Removes all StreamHandlers writing to stdout/stderr from the root logger
    and every existing logger, then bumps noisy third-party loggers to
    WARNING so they don't corrupt the prompt_toolkit / Rich display.
    File logging continues unaffected.
    """
    global _console_enabled
    _console_enabled = False

    # Remove existing console handlers from root
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
            root.removeHandler(handler)

    # Remove console handlers from all existing loggers
    for logger_name in list(logging.root.manager.loggerDict):
        logger = logging.getLogger(logger_name)
        for handler in list(logger.handlers):
            if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
                logger.removeHandler(handler)

    # Suppress noisy third-party libraries
    for noisy in (
        "chromadb",
        "chromadb.telemetry",
        "sentence_transformers",
        "transformers",
        "transformers.tokenization_utils",
        "urllib3",
        "httpx",
        "openai",
        "huggingface_hub",
        "torch",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
