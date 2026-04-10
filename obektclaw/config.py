"""Runtime configuration. Reads .env if present, falls back to environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_env_file(env_path: Path) -> None:
    """Read a .env file and set vars that aren't already in os.environ."""
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _load_dotenv() -> None:
    # 1. User home config (~/.obektclaw/.env) — primary location
    home = Path(os.environ.get("OBEKTCLAW_HOME") or Path.home() / ".obektclaw").expanduser()
    _read_env_file(home / ".env")
    # 2. Project-local .env — dev override (lower priority, won't overwrite)
    project_env = Path(__file__).resolve().parent.parent / ".env"
    _read_env_file(project_env)


_load_dotenv()


@dataclass(frozen=True)
class Config:
    home: Path
    db_path: Path
    skills_dir: Path
    bundled_skills_dir: Path
    logs_dir: Path

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_fast_model: str

    tg_token: str
    tg_allowed_chat_ids: tuple[int, ...]

    bash_timeout: int
    workdir: Path
    context_window: int = 0  # 0 = auto-detect from model name


def _int_list(raw: str) -> tuple[int, ...]:
    out: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            out.append(int(piece))
        except ValueError:
            pass
    return tuple(out)


def load_config() -> Config:
    home = Path(os.environ.get("OBEKTCLAW_HOME") or Path.home() / ".obektclaw").expanduser()
    home.mkdir(parents=True, exist_ok=True)

    skills_dir = home / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = home / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    bundled = Path(__file__).resolve().parent.parent / "bundled_skills"

    workdir_env = os.environ.get("OBEKTCLAW_WORKDIR", "").strip()
    workdir = Path(workdir_env).expanduser() if workdir_env else Path.cwd()

    return Config(
        home=home,
        db_path=home / "obektclaw.db",
        skills_dir=skills_dir,
        bundled_skills_dir=bundled,
        logs_dir=logs_dir,
        llm_base_url=os.environ.get("OBEKTCLAW_LLM_BASE_URL", "https://api.openai.com/v1"),
        llm_api_key=os.environ.get("OBEKTCLAW_LLM_API_KEY", ""),
        llm_model=os.environ.get("OBEKTCLAW_LLM_MODEL", "gpt-4o-mini"),
        llm_fast_model=os.environ.get("OBEKTCLAW_LLM_FAST_MODEL", os.environ.get("OBEKTCLAW_LLM_MODEL", "gpt-4o-mini")),
        tg_token=os.environ.get("OBEKTCLAW_TG_TOKEN", ""),
        tg_allowed_chat_ids=_int_list(os.environ.get("OBEKTCLAW_TG_ALLOWED_CHAT_IDS", "")),
        context_window=int(os.environ.get("OBEKTCLAW_CONTEXT_WINDOW", "0")),
        bash_timeout=int(os.environ.get("OBEKTCLAW_BASH_TIMEOUT", "30")),
        workdir=workdir,
    )


CONFIG = load_config()
