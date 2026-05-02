"""Runtime configuration. Reads .env if present, falls back to environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_PLACEHOLDER_VALUES = {"your-api-key-here", "sk-xxx", "sk-your-key", ""}


def _read_env_file(env_path: Path) -> None:
    """Read a .env file and set vars that aren't already set to a meaningful value."""
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        # Skip placeholder values — they're not meaningful config
        if v in _PLACEHOLDER_VALUES:
            continue
        # Only skip if already set to a non-empty value
        existing = os.environ.get(k, "")
        if existing:
            continue
        os.environ[k] = v


def _load_dotenv() -> None:
    # 1. Project-local .env first — lower priority (dev defaults)
    project_env = Path(__file__).resolve().parent.parent / ".env"
    _read_env_file(project_env)
    # 2. User home config (~/.obektclaw/.env) — higher priority (user's real config)
    home = Path(
        os.environ.get("OBEKTCLAW_HOME") or Path.home() / ".obektclaw"
    ).expanduser()
    _read_env_file(home / ".env")


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

    # Memory system config (CogDB + ChromaDB + Local LLM)
    # All have defaults computed in load_config()
    cog_home: Path = Path.home() / ".obektclaw" / "cog-home"
    chroma_path: Path = Path.home() / ".obektclaw" / "chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    graph_name: str = "obektclaw"

    # Hybrid retrieval config
    semantic_search_limit: int = 10
    graph_traversal_depth: int = 3
    context_assembly_max_tokens: int = 2000

    # Context window - 0 = auto-detect from model name
    context_window: int = 0

    # Extraction LLM config (entity/relationship extraction for Learning Loop)
    # Falls back to main LLM config if not specified
    extraction_llm_base_url: str | None = None  # Falls back to llm_base_url
    extraction_llm_api_key: str | None = None  # Falls back to llm_api_key
    extraction_llm_model: str | None = None  # Falls back to llm_fast_model or llm_model


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
    home = Path(
        os.environ.get("OBEKTCLAW_HOME") or Path.home() / ".obektclaw"
    ).expanduser()
    home.mkdir(parents=True, exist_ok=True)

    skills_dir = home / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = home / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    bundled = Path(__file__).resolve().parent.parent / "bundled_skills"

    workdir_env = os.environ.get("OBEKTCLAW_WORKDIR", "").strip()
    workdir = Path(workdir_env).expanduser() if workdir_env else Path.cwd()

    # Memory system paths
    cog_home = Path(
        os.environ.get("OBEKTCLAW_COG_HOME") or home / "cog-home"
    ).expanduser()
    cog_home.mkdir(parents=True, exist_ok=True)

    chroma_path = Path(
        os.environ.get("OBEKTCLAW_CHROMA_PATH") or home / "chroma"
    ).expanduser()
    chroma_path.mkdir(parents=True, exist_ok=True)

    # Read main LLM config first (extraction LLM falls back to these)
    llm_base_url = os.environ.get("OBEKTCLAW_LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key = os.environ.get("OBEKTCLAW_LLM_API_KEY", "")
    llm_model = os.environ.get("OBEKTCLAW_LLM_MODEL", "gpt-4o-mini")
    llm_fast_model = os.environ.get(
        "OBEKTCLAW_LLM_FAST_MODEL",
        llm_model,  # Fall back to main model
    )

    # Extraction LLM config (falls back to main LLM if not specified)
    extraction_base_url = os.environ.get("OBEKTCLAW_EXTRACTION_LLM_BASE_URL")
    extraction_api_key = os.environ.get("OBEKTCLAW_EXTRACTION_LLM_API_KEY")
    extraction_model = os.environ.get("OBEKTCLAW_EXTRACTION_LLM_MODEL")

    return Config(
        home=home,
        db_path=home / "obektclaw.db",
        skills_dir=skills_dir,
        bundled_skills_dir=bundled,
        logs_dir=logs_dir,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_fast_model=llm_fast_model,
        tg_token=os.environ.get("OBEKTCLAW_TG_TOKEN", ""),
        tg_allowed_chat_ids=_int_list(
            os.environ.get("OBEKTCLAW_TG_ALLOWED_CHAT_IDS", "")
        ),
        context_window=int(os.environ.get("OBEKTCLAW_CONTEXT_WINDOW", "0")),
        bash_timeout=int(os.environ.get("OBEKTCLAW_BASH_TIMEOUT", "30")),
        workdir=workdir,
        # Memory system
        cog_home=cog_home,
        chroma_path=chroma_path,
        embedding_model=os.environ.get("OBEKTCLAW_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        embedding_dimension=int(os.environ.get("OBEKTCLAW_EMBEDDING_DIMENSION", "384")),
        graph_name=os.environ.get("OBEKTCLAW_GRAPH_NAME", "obektclaw"),
        semantic_search_limit=int(
            os.environ.get("OBEKTCLAW_SEMANTIC_SEARCH_LIMIT", "10")
        ),
        graph_traversal_depth=int(
            os.environ.get("OBEKTCLAW_GRAPH_TRAVERSAL_DEPTH", "3")
        ),
        context_assembly_max_tokens=int(
            os.environ.get("OBEKTCLAW_CONTEXT_ASSEMBLY_MAX_TOKENS", "2000")
        ),
        # Extraction LLM (falls back to main LLM if not specified)
        extraction_llm_base_url=extraction_base_url,
        extraction_llm_api_key=extraction_api_key,
        extraction_llm_model=extraction_model,
    )


CONFIG = load_config()
