"""Interactive CLI gateway. A modern REPL on top of Agent.run_once.

Uses prompt_toolkit for interactive slash-command completion, styled
prompts, history persistence, and a status toolbar.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from ..agent import Agent
from ..config import CONFIG, load_config
from ..memory.store import Store
from ..skills import SkillManager


# ── Slash commands ──────────────────────────────────────────────────────────

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help",       "Show help and available commands"),
    ("/skills",     "List known skills"),
    ("/memory",     "Search persistent memory  (/memory <query>)"),
    ("/traits",     "Show your user model"),
    ("/setup",      "Guided setup (Telegram, MCP, etc.)"),
    ("/exit",       "Quit the session"),
]


class SlashCompleter(Completer):
    """Show an interactive selection menu when the user types '/'."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return
        # Match against the typed prefix (e.g. "/sk" → "/skills")
        for cmd, desc in SLASH_COMMANDS:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=HTML(f"<b>{cmd}</b>"),
                    display_meta=desc,
                )


# ── Styles ──────────────────────────────────────────────────────────────────

PROMPT_STYLE = Style.from_dict({
    # Prompt arrow
    "prompt":       "#00d7ff bold",
    # Completion menu
    "completion-menu":                "bg:#1e1e2e #cdd6f4",
    "completion-menu.completion":     "bg:#1e1e2e #cdd6f4",
    "completion-menu.completion.current": "bg:#45475a #cdd6f4 bold",
    "completion-menu.meta":           "bg:#1e1e2e #6c7086 italic",
    "completion-menu.meta.current":   "bg:#45475a #a6adc8 italic",
    # Bottom toolbar
    "bottom-toolbar":       "bg:#1e1e2e #6c7086",
    "bottom-toolbar.text":  "#6c7086",
})

PROMPT_MESSAGE = [("class:prompt", "❯ ")]


# ── Context size formatting ─────────────────────────────────────────────────

def _format_tokens(n: int) -> str:
    """Format token count compactly: 1234 → '1.2k', 128000 → '128k'."""
    if n < 1_000:
        return str(n)
    if n < 10_000:
        return f"{n / 1_000:.1f}k"
    return f"{n // 1_000}k"


def _make_toolbar(agent_ref: list):
    """Return a toolbar callable that reads live token usage from the agent."""
    def _bottom_toolbar():
        parts = [
            " <b>/</b> commands  ",
            "<style bg='#313244'> ctrl-c </style> cancel  ",
            "<style bg='#313244'> ctrl-d </style> exit",
        ]
        agent = agent_ref[0] if agent_ref else None
        if agent and agent.last_usage:
            u = agent.last_usage
            ctx_window = agent.context_window
            used = u.prompt_tokens + u.completion_tokens
            pct = min(int(used / ctx_window * 100), 100) if ctx_window else 0
            # Color: green < 50%, yellow 50-80%, red > 80%
            if pct < 50:
                color = "#a6e3a1"
            elif pct < 80:
                color = "#f9e2af"
            else:
                color = "#f38ba8"
            parts.append(
                f"  <style bg='#313244' fg='{color}'> "
                f"{_format_tokens(used)}/{_format_tokens(ctx_window)} ({pct}%) </style>"
            )
        return HTML("".join(parts))
    return _bottom_toolbar


# ── Session factory ─────────────────────────────────────────────────────────

def _make_session(agent_ref: list) -> PromptSession:
    history_path = CONFIG.home / "history.txt"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(
        message=PROMPT_MESSAGE,
        style=PROMPT_STYLE,
        completer=SlashCompleter(),
        complete_while_typing=True,
        history=FileHistory(str(history_path)),
        bottom_toolbar=_make_toolbar(agent_ref),
        mouse_support=False,
    )


# ── Banners / help ──────────────────────────────────────────────────────────

BANNER = """\
\033[38;5;75m╭───────────────────────────────────────────╮\033[0m
\033[38;5;75m│\033[0m         \033[1mobektclaw\033[0m agent                  \033[38;5;75m│\033[0m
\033[38;5;75m│\033[0m  self-improving AI · memory · skills    \033[38;5;75m│\033[0m
\033[38;5;75m╰───────────────────────────────────────────╯\033[0m

  Type a message and hit enter.  \033[2mType / for commands.\033[0m
"""

HELP_TEXT = """\
\033[1mobektclaw\033[0m — self-improving AI agent

\033[38;5;75m  Memory\033[0m      session history · persistent facts · user model
\033[38;5;75m  Tools\033[0m       files · bash · python · web · memory · skills · sub-agents
\033[38;5;75m  Skills\033[0m      auto-created markdown guides that improve after each use

\033[1mCommands\033[0m
  /help          this help
  /skills        list skills
  /memory <q>    search memory
  /traits        show user model
  /setup         configure integrations
  /exit          quit

\033[2mTip: type / to see interactive command picker.\033[0m
"""

SETUP_TEXT = """
╔═══════════════════════════════════════════════════════════╗
║                    obektclaw Setup                        ║
╚═══════════════════════════════════════════════════════════╝

Current configuration:
  OBEKTCLAW_HOME: {obektclaw_home}
  Database: {db_path}
  Skills: {skills_dir}
  Logs: {logs_dir}

"""


def _first_run_welcome():
    """Show a welcome message on first run."""
    print(f"""\
\033[38;5;75m╭───────────────────────────────────────────╮\033[0m
\033[38;5;75m│\033[0m    \033[1mWelcome to obektclaw\033[0m                  \033[38;5;75m│\033[0m
\033[38;5;75m╰───────────────────────────────────────────╯\033[0m

  I'm a self-improving AI agent with memory & skills.

  \033[38;5;75m1.\033[0m \033[1mRemember\033[0m   "I always use httpx over requests"
  \033[38;5;75m2.\033[0m \033[1mExecute\033[0m    "List all Python files here"
  \033[38;5;75m3.\033[0m \033[1mLearn\033[0m      I create skills from patterns I discover
  \033[38;5;75m4.\033[0m \033[1mConnect\033[0m    /setup to add Telegram or MCP servers

  \033[2mType / for commands · /help for docs · /exit to quit\033[0m
""")


def _show_setup(config=None):
    """Show current configuration and guided options."""
    cfg = config or CONFIG
    print(SETUP_TEXT.format(
        obektclaw_home=cfg.home,
        db_path=cfg.db_path,
        skills_dir=cfg.skills_dir,
        logs_dir=cfg.logs_dir,
    ))

    # LLM
    print(f"  LLM: {cfg.llm_model} via {cfg.llm_base_url}")
    print(f"  Config: {_env_file()}")
    print()

    # Check if MCP config exists
    mcp_config = cfg.home / "mcp.json"
    if mcp_config.exists():
        print("✓ MCP servers configured")
    else:
        print("○ MCP servers: not configured")
        print("  Create {} to connect external tools".format(mcp_config))

    # Check Telegram
    if cfg.tg_token:
        print("✓ Telegram bot configured")
        print("  Run: python -m obektclaw tg")
    else:
        print("○ Telegram bot: not configured")
        print("  To enable Telegram chat:")
        print("  1. Message @BotFather on Telegram")
        print("  2. Create a new bot with /newbot")
        print("  3. Copy the token")
        print(f"  4. Add to {_env_file()}: OBEKTCLAW_TG_TOKEN=your_token")

    print()


_PLACEHOLDER_KEYS = {"your-api-key-here", "sk-xxx", "sk-your-key", ""}

def _env_file() -> Path:
    """The .env lives alongside data in OBEKTCLAW_HOME (~/.obektclaw/.env)."""
    return CONFIG.home / ".env"

_PROVIDERS = [
    ("OpenRouter", "https://openrouter.ai/api/v1", "anthropic/claude-haiku-4.5"),
    ("OpenAI", "https://api.openai.com/v1", "gpt-4o-mini"),
    ("Ollama (local)", "http://localhost:11434/v1", "llama3"),
    ("Other", "", ""),
]


def _prompt(label: str, default: str = "") -> str:
    """Prompt with an optional default shown in brackets."""
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    return val or default


def _test_llm(base_url: str, api_key: str, model: str) -> str | None:
    """Try a minimal LLM call. Returns None on success, error string on failure."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        if resp.choices and resp.choices[0].message.content:
            return None
        return "LLM returned an empty response"
    except Exception as e:
        return str(e)


def _write_env(base_url: str, api_key: str, model: str) -> None:
    """Write or update .env with the LLM settings."""
    lines: list[str] = []
    keys_written: set[str] = set()
    settings = {
        "OBEKTCLAW_LLM_BASE_URL": base_url,
        "OBEKTCLAW_LLM_API_KEY": api_key,
        "OBEKTCLAW_LLM_MODEL": model,
        "OBEKTCLAW_LLM_FAST_MODEL": model,
    }

    if _env_file().exists():
        for raw in _env_file().read_text().splitlines():
            stripped = raw.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k = stripped.partition("=")[0].strip()
                if k in settings:
                    lines.append(f"{k}={settings[k]}")
                    keys_written.add(k)
                    continue
            lines.append(raw)

    for k, v in settings.items():
        if k not in keys_written:
            lines.append(f"{k}={v}")

    _env_file().write_text("\n".join(lines) + "\n")


def _setup_wizard() -> "Config | None":
    """Interactive setup. Returns a new Config on success, None on abort."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║              obektclaw — First Time Setup                  ║
╚═══════════════════════════════════════════════════════════╝

Let's connect to an LLM provider. Pick one:
""")
    for i, (name, url, _model) in enumerate(_PROVIDERS, 1):
        if url:
            print(f"  {i}. {name}  ({url})")
        else:
            print(f"  {i}. {name}")
    print()

    choice = _prompt("Provider [1-4]", "1")
    if not choice:
        return None

    try:
        idx = int(choice) - 1
        if not 0 <= idx < len(_PROVIDERS):
            idx = 0
    except ValueError:
        idx = 0

    _name, default_url, default_model = _PROVIDERS[idx]

    if idx == len(_PROVIDERS) - 1:
        # "Other" — ask for everything
        base_url = _prompt("Base URL (e.g. https://api.example.com/v1)")
        if not base_url:
            return None
        default_model = ""
    else:
        base_url = default_url

    print()
    api_key = _prompt("API key")
    if not api_key:
        print("\n  No key provided. Aborting setup.")
        return None

    print()
    model = _prompt("Model", default_model)
    if not model:
        model = default_model

    # Test the connection
    print(f"\n  Testing connection to {_name}...", end=" ", flush=True)
    err = _test_llm(base_url, api_key, model)
    if err:
        print("FAILED")
        # Show a concise error
        if "401" in err or "auth" in err.lower():
            print(f"  Invalid API key.")
        elif "connection" in err.lower() or "refused" in err.lower():
            print(f"  Can't reach {base_url}")
        elif "model" in err.lower() and ("not found" in err.lower() or "404" in err):
            print(f"  Model '{model}' not found at this provider.")
        else:
            print(f"  {err[:200]}")
        print()
        retry = _prompt("Save anyway? (y/n)", "n")
        if retry.lower() != "y":
            return None

    else:
        print("OK!")

    # Write to .env and force into os.environ
    _write_env(base_url, api_key, model)
    os.environ["OBEKTCLAW_LLM_BASE_URL"] = base_url
    os.environ["OBEKTCLAW_LLM_API_KEY"] = api_key
    os.environ["OBEKTCLAW_LLM_MODEL"] = model
    os.environ["OBEKTCLAW_LLM_FAST_MODEL"] = model

    print(f"\n  Saved to {_env_file()}")
    print(f"  Provider: {_name}")
    print(f"  Model:    {model}")
    print()

    return load_config()


def _check_config() -> bool:
    """Returns True if the API key looks real."""
    key = CONFIG.llm_api_key.strip()
    return bool(key) and key not in _PLACEHOLDER_KEYS


def run() -> int:
    config = CONFIG
    if not _check_config():
        result = _setup_wizard()
        if result is None:
            return 1
        config = result

    store = Store(config.db_path)
    skills = SkillManager(store, config.skills_dir, config.bundled_skills_dir)

    # Check if this is the first run (no sessions yet)
    row = store.fetchone("SELECT COUNT(*) as c FROM sessions")
    is_first_run = row["c"] == 0

    agent = Agent(config=config, store=store, skills=skills, gateway="cli", user_key="cli-local")
    agent_ref = [agent]  # mutable ref so toolbar can read live usage

    if is_first_run:
        _first_run_welcome()
    else:
        print(BANNER)

    session = _make_session(agent_ref)

    try:
        while True:
            try:
                line = session.prompt().strip()
            except KeyboardInterrupt:
                continue          # ctrl-c clears current input
            except EOFError:
                break             # ctrl-d exits
            if not line:
                continue
            if line in ("/exit", "/quit"):
                break
            if line == "/help":
                print(HELP_TEXT)
                continue
            if line == "/skills":
                all_skills = skills.list_all()
                if not all_skills:
                    print("  \033[2mNo skills yet. They auto-create when I learn patterns.\033[0m")
                else:
                    print(f"\n  \033[1mSkills\033[0m ({len(all_skills)})\n")
                    for sk in all_skills:
                        print(f"  \033[38;5;75m●\033[0m {sk.name}  \033[2m{sk.description}\033[0m")
                    print()
                continue
            if line.startswith("/memory"):
                q = line[len("/memory"):].strip()
                if not q:
                    print("  \033[2musage: /memory <query>\033[0m")
                    continue
                results = list(agent.persistent.search(q))
                if not results:
                    print(f"  \033[2mNo memories found for '{q}'\033[0m")
                else:
                    print(f"\n  \033[1mMemories\033[0m ({len(results)})\n")
                    for f in results:
                        print(f"  \033[38;5;75m●\033[0m {f.render()}")
                    print()
                continue
            if line == "/traits":
                traits = agent.user_model.all()
                if not traits:
                    print("  \033[2mNo user model yet. It builds as we talk.\033[0m")
                else:
                    print(f"\n  \033[1mUser model\033[0m ({len(traits)} traits)\n")
                    for t in traits:
                        print(f"  \033[38;5;75m●\033[0m \033[1m{t.layer}\033[0m  {t.value}")
                    print()
                continue
            if line == "/setup":
                _show_setup(config)
                continue

            # ── Agent turn ──────────────────────────────────────────────
            try:
                reply = agent.run_once(line)
            except Exception as e:  # noqa: BLE001
                err = str(e)
                if "401" in err or "auth" in err.lower() or "api key" in err.lower():
                    print(f"\033[31m  error:\033[0m Invalid API key. Check OBEKTCLAW_LLM_API_KEY in {_env_file()}", file=sys.stderr)
                elif "connection" in err.lower() or "refused" in err.lower():
                    print(f"\033[31m  error:\033[0m Can't reach LLM at {config.llm_base_url}", file=sys.stderr)
                elif "429" in err or "rate" in err.lower():
                    print("\033[31m  error:\033[0m Rate limited. Wait a moment and try again.", file=sys.stderr)
                elif "model" in err.lower() and ("not found" in err.lower() or "404" in err):
                    print(f"\033[31m  error:\033[0m Model '{config.llm_model}' not found.", file=sys.stderr)
                else:
                    print(f"\033[31m  error:\033[0m {e}", file=sys.stderr)
                continue
            print(f"\n\033[38;5;75m  obektclaw\033[0m  {reply}\n")
            # Warn when context is getting tight
            try:
                pressure = agent._context_pressure()
            except (AttributeError, TypeError):
                pressure = 0.0
            if isinstance(pressure, (int, float)) and pressure > 0.8:
                print(
                    "\033[33m  warning:\033[0m context window is "
                    f"{int(pressure * 100)}% full — "
                    "consider starting a new session.\033[0m\n"
                )
    finally:
        agent.close()
        store.close()
    return 0

