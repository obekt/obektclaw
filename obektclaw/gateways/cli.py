"""Interactive CLI gateway. A modern REPL on top of Agent.run_once.

Uses prompt_toolkit for interactive slash-command completion, styled
prompts, history persistence, and a status toolbar.
Enhanced with Rich for beautiful terminal output.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PromptStyle
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich import box

from ..agent import Agent
from ..config import CONFIG, load_config
from ..memory.store import Store
from ..model_context import get_context_window, list_known_models, guess_context_window
from ..skills import SkillManager

# Initialize Rich console
console = Console()


# ── Slash commands ──────────────────────────────────────────────────────────

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help",       "Show help and available commands"),
    ("/skills",     "List known skills"),
    ("/memory",     "Search persistent memory  (/memory <query>)"),
    ("/traits",     "Show your user model"),
    ("/model",      "Show or change the current LLM model"),
    ("/compact",    "Compact conversation history to save context"),
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

PROMPT_STYLE = PromptStyle.from_dict({
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

def show_banner():
    """Show a rich styled banner."""
    title = Text("obektclaw", style="bold cyan")
    subtitle = Text("self-improving AI agent · memory · skills", style="dim")
    
    panel = Panel(
        Text.assemble(title, "\n", subtitle),
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)
    console.print()
    console.print("Type a message and hit enter. ", end="")
    console.print("Type / for commands.", style="dim italic")
    console.print()

def show_help():
    """Show rich formatted help."""
    console.print(Panel("obektclaw — self-improving AI agent", style="bold cyan", padding=(0, 1)))
    console.print()
    
    # Feature list
    features = [
        ("[cyan]Memory[/cyan]", "session history · persistent facts · user model"),
        ("[cyan]Tools[/cyan]", "files · bash · python · web · memory · skills · sub-agents"),
        ("[cyan]Skills[/cyan]", "auto-created markdown guides that improve after each use"),
    ]
    for title, desc in features:
        console.print(f"  {title:12s} {desc}")
    
    console.print()
    console.print(Rule("Commands", style="cyan"))
    
    commands = [
        ("/help", "this help"),
        ("/skills", "list skills"),
        ("/memory <q>", "search memory"),
        ("/traits", "show user model"),
        ("/model", "show/change LLM model"),
        ("/compact", "compact conversation history"),
        ("/setup", "configure integrations"),
        ("/exit", "quit"),
    ]
    
    table = Table.grid(padding=(0, 2))
    table.add_column("Command", style="bold cyan")
    table.add_column("Description", style="dim")
    
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    
    console.print(table)
    console.print()
    console.print("Tip: type / to see interactive command picker.", style="italic dim")
    console.print()

def show_setup(config=None):
    """Show current configuration and guided options."""
    cfg = config or CONFIG
    
    console.print(Panel("obektclaw Setup", style="bold cyan", padding=(0, 1)))
    console.print()
    
    # Current configuration
    console.print("[bold]Current configuration:[/bold]")
    config_items = [
        ("OBEKTCLAW_HOME", str(cfg.home)),
        ("Database", str(cfg.db_path)),
        ("Skills", str(cfg.skills_dir)),
        ("Logs", str(cfg.logs_dir)),
    ]
    
    table = Table.grid(padding=(0, 2))
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="dim")
    
    for setting, value in config_items:
        table.add_row(setting, value)
    
    console.print(table)
    console.print()
    
    # LLM config
    console.print(f"  [cyan]LLM:[/cyan] {cfg.llm_model} via {cfg.llm_base_url}")
    console.print(f"  [cyan]Config:[/cyan] {_env_file()}")
    console.print()
    
    # Check if MCP config exists
    mcp_config = cfg.home / "mcp.json"
    if mcp_config.exists():
        console.print("✓ [green]MCP servers configured[/green]")
    else:
        console.print("○ [dim]MCP servers: not configured[/dim]")
        console.print(f"  Create [cyan]{mcp_config}[/cyan] to connect external tools")
    
    # Check Telegram
    if cfg.tg_token:
        console.print("✓ [green]Telegram bot configured[/green]")
        console.print("  Run: [cyan]python -m obektclaw tg[/cyan]")
    else:
        console.print("○ [dim]Telegram bot: not configured[/dim]")
        console.print("  To enable Telegram chat:")
        console.print("  1. Message @BotFather on Telegram")
        console.print("  2. Create a new bot with /newbot")
        console.print("  3. Copy the token")
        console.print(f"  4. Add to [cyan]{_env_file()}[/cyan]: OBEKTCLAW_TG_TOKEN=your_token")
    
    console.print()


def _first_run_welcome():
    """Show a welcome message on first run."""
    title = Text("Welcome to obektclaw", style="bold cyan")
    panel = Panel(
        title,
        box=box.DOUBLE,
        border_style="cyan",
        padding=(0, 2),
    )
    console.print(panel)
    console.print()
    console.print("I'm a self-improving AI agent with memory & skills.")
    console.print()
    
    steps = [
        ("1.", "Remember", '"I always use httpx over requests"'),
        ("2.", "Execute", '"List all Python files here"'),
        ("3.", "Learn", "I create skills from patterns I discover"),
        ("4.", "Connect", "/setup to add Telegram or MCP servers"),
    ]
    
    for num, action, example in steps:
        console.print(f"  [cyan]{num}[/cyan] [bold]{action}[/bold]  {example}")
    
    console.print()
    console.print("Type / for commands · /help for docs · /exit to quit", style="dim italic")
    console.print()

def _env_file() -> Path:
    """The .env lives alongside data in OBEKTCLAW_HOME (~/.obektclaw/.env)."""
    return CONFIG.home / ".env"

_PLACEHOLDER_KEYS = {"your-api-key-here", "sk-xxx", "sk-your-key", ""}

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
    console.print(Panel("obektclaw — First Time Setup", style="bold cyan", padding=(0, 1)))
    console.print()
    console.print("Let's connect to an LLM provider. Pick one:")
    console.print()
    
    for i, (name, url, _model) in enumerate(_PROVIDERS, 1):
        if url:
            console.print(f"  [cyan]{i}.[/cyan] {name}  [dim]({url})[/dim]")
        else:
            console.print(f"  [cyan]{i}.[/cyan] {name}")
    console.print()

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

    console.print()
    api_key = _prompt("API key")
    if not api_key:
        console.print("\n[cyan]No key provided. Aborting setup.[/cyan]")
        return None

    console.print()
    model = _prompt("Model", default_model)
    if not model:
        model = default_model

    # Test the connection
    console.print(f"\nTesting connection to {_name}...", end="")
    err = _test_llm(base_url, api_key, model)
    if err:
        console.print(" [red]FAILED[/red]")
        # Show a concise error
        if "401" in err or "auth" in err.lower():
            console.print("  [red]Invalid API key.[/red]")
        elif "connection" in err.lower() or "refused" in err.lower():
            console.print(f"  [red]Can't reach {base_url}[/red]")
        elif "model" in err.lower() and ("not found" in err.lower() or "404" in err):
            console.print(f"  [red]Model '{model}' not found at this provider.[/red]")
        else:
            console.print(f"  [red]{err[:200]}[/red]")
        console.print()
        retry = _prompt("Save anyway? (y/n)", "n")
        if retry.lower() != "y":
            return None

    else:
        console.print(" [green]OK![/green]")

    # Write to .env and force into os.environ
    _write_env(base_url, api_key, model)
    os.environ["OBEKTCLAW_LLM_BASE_URL"] = base_url
    os.environ["OBEKTCLAW_LLM_API_KEY"] = api_key
    os.environ["OBEKTCLAW_LLM_MODEL"] = model
    os.environ["OBEKTCLAW_LLM_FAST_MODEL"] = model

    console.print(f"\n[cyan]Saved to[/cyan] {_env_file()}")
    console.print(f"  [cyan]Provider:[/cyan] {_name}")
    console.print(f"  [cyan]Model:[/cyan]    {model}")
    console.print()

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
        show_banner()

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
                show_help()
                continue
            if line == "/skills":
                all_skills = skills.list_all()
                if not all_skills:
                    console.print("  [dim]No skills yet. They auto-create when I learn patterns.[/dim]")
                else:
                    console.print()
                    table = Table(
                        title=f"Skills ({len(all_skills)})",
                        box=box.ROUNDED,
                        border_style="cyan",
                    )
                    table.add_column("Name", style="bold cyan")
                    table.add_column("Description", style="dim")
                    table.add_column("Uses", justify="right", style="green")
                    
                    for sk in all_skills:
                        table.add_row(
                            sk.name,
                            sk.description or "",
                            str(sk.use_count),
                        )
                    console.print(table)
                    console.print()
                continue
            if line.startswith("/memory"):
                q = line[len("/memory"):].strip()
                if not q:
                    console.print("  [dim]usage: /memory <query>[/dim]")
                    continue
                results = list(agent.persistent.search(q))
                if not results:
                    console.print(f"  [dim]No memories found for '{q}'[/dim]")
                else:
                    console.print()
                    console.print(f"[bold cyan]Memories[/bold cyan] ({len(results)})")
                    console.print(Rule(style="cyan"))
                    for f in results:
                        console.print(f"  [cyan]●[/cyan] {f.render()}")
                    console.print()
                continue
            if line == "/traits":
                traits = agent.user_model.all()
                if not traits:
                    console.print("  [dim]No user model yet. It builds as we talk.[/dim]")
                else:
                    console.print()
                    console.print(f"[bold cyan]User model[/bold cyan] ({len(traits)} traits)")
                    console.print(Rule(style="cyan"))
                    
                    table = Table(box=box.SIMPLE, padding=(0, 2))
                    table.add_column("Layer", style="bold cyan")
                    table.add_column("Value", style="white")
                    
                    for t in traits:
                        table.add_row(t.layer, t.value)
                    console.print(table)
                    console.print()
                continue
            if line == "/setup":
                show_setup(config)
                continue
            if line.startswith("/model"):
                args = line[len("/model"):].strip()
                if not args:
                    # Show current model
                    detected = get_context_window(config.llm_model, config.home)
                    console.print()
                    console.print(Panel(
                        f"[cyan]Current Model:[/cyan] {config.llm_model}\n"
                        f"[cyan]Fast Model:[/cyan] {config.llm_fast_model}\n"
                        f"[cyan]Context Window:[/cyan] {agent.context_window:,} tokens "
                        f"(detected: {detected:,})",
                        title="[cyan]Model Info[/cyan]",
                        border_style="cyan",
                        box=box.ROUNDED,
                    ))
                    console.print()
                    console.print("  [dim]Usage: /model <name> [context_window][/dim]")
                    console.print("  [dim]Example: /model gpt-4o 128000[/dim]")
                    console.print()
                elif args in ("list", "ls", "-l"):
                    # Show known models
                    models = list_known_models()
                    console.print()
                    console.print(f"[bold cyan]Known Models ({len(models)})[/bold cyan]")
                    console.print(Rule(style="cyan"))

                    table = Table(box=box.SIMPLE, padding=(0, 1))
                    table.add_column("Model", style="bold cyan")
                    table.add_column("Context", justify="right", style="green")
                    table.add_column("Source", style="dim")

                    for m in models:
                        table.add_row(
                            m["name"],
                            f"{m['context_window']:,}",
                            m["source"],
                        )
                    console.print(table)
                    console.print()
                else:
                    # Switch model
                    parts = args.split()
                    new_model = parts[0]
                    new_context = None
                    
                    if len(parts) > 1:
                        try:
                            new_context = int(parts[1])
                        except ValueError:
                            console.print(f"  [red]Invalid context window: {parts[1]}[/red]")
                            continue
                    
                    try:
                        result = agent.switch_model(
                            model=new_model,
                            context_window=new_context,
                        )
                        console.print()
                        console.print(Panel(
                            f"[green]✓ Model switched successfully[/green]\n\n"
                            f"[cyan]Model:[/cyan] {result['model']}\n"
                            f"[cyan]Fast Model:[/cyan] {result['fast_model']}\n"
                            f"[cyan]Context Window:[/cyan] {result['context_window']:,} tokens"
                            + (f"\n[dim](saved to models.json)[/dim]" if result['was_overridden'] else ""),
                            title="[cyan]Model Switch[/cyan]",
                            border_style="green",
                            box=box.ROUNDED,
                        ))
                        console.print()
                    except Exception as e:
                        console.print(Panel(
                            f"Failed to switch model: {e}",
                            title="[red]Error[/red]",
                            border_style="red",
                            box=box.ROUNDED,
                        ))
                        console.print()
                continue
            if line == "/compact":
                # Force compaction regardless of pressure
                console.print()
                with console.status("[cyan]Compacting context...[/cyan]", spinner="dots"):
                    result = agent.compact_context(force=True)
                
                if result["compacted"]:
                    console.print(Panel(
                        f"[green]✓ Context compacted successfully[/green]\n\n"
                        f"[cyan]Summary:[/cyan] {result['summary_length']} words\n"
                        f"[cyan]Tokens saved:[/cyan] ~{result['tokens_saved']:,}",
                        title="[cyan]Compaction[/cyan]",
                        border_style="green",
                        box=box.ROUNDED,
                    ))
                else:
                    console.print(Panel(
                        f"[yellow]Compaction skipped[/yellow]\n\n"
                        f"[dim]Reason: {result['reason']}[/dim]",
                        title="[cyan]Compaction[/cyan]",
                        border_style="yellow",
                        box=box.ROUNDED,
                    ))
                console.print()
                continue

            # ── Agent turn ──────────────────────────────────────────────
            # Use the agent's status callback to show a dynamic spinner
            status_obj = None
            def _on_status(msg: str):
                nonlocal status_obj
                if msg:
                    if status_obj is None:
                        status_obj = console.status(f"[cyan]{msg}[/cyan]", spinner="dots")
                        status_obj.start()
                    else:
                        status_obj.update(f"[cyan]{msg}[/cyan]")
                elif status_obj is not None:
                    status_obj.stop()
                    status_obj = None

            try:
                reply = agent.run_once(line, status_fn=_on_status)
            except Exception as e:  # noqa: BLE001
                if status_obj:
                    status_obj.stop()
                    status_obj = None
                err = str(e)
                if "401" in err or "auth" in err.lower() or "api key" in err.lower():
                    console.print(Panel(
                        f"Invalid API key. Check OBEKTCLAW_LLM_API_KEY in {_env_file()}",
                        title="[red]Error[/red]",
                        border_style="red",
                        box=box.ROUNDED,
                    ))
                elif "connection" in err.lower() or "refused" in err.lower():
                    console.print(Panel(
                        f"Can't reach LLM at {config.llm_base_url}",
                        title="[red]Error[/red]",
                        border_style="red",
                        box=box.ROUNDED,
                    ))
                elif "429" in err or "rate" in err.lower():
                    console.print(Panel(
                        "Rate limited. Wait a moment and try again.",
                        title="[red]Error[/red]",
                        border_style="red",
                        box=box.ROUNDED,
                    ))
                elif "model" in err.lower() and ("not found" in err.lower() or "404" in err):
                    console.print(Panel(
                        f"Model '{config.llm_model}' not found.",
                        title="[red]Error[/red]",
                        border_style="red",
                        box=box.ROUNDED,
                    ))
                else:
                    console.print(Panel(
                        str(e),
                        title="[red]Error[/red]",
                        border_style="red",
                        box=box.ROUNDED,
                    ))
                continue
            
            # Display agent response with rich markdown rendering
            console.print()
            response_panel = Panel(
                Markdown(reply),
                title="[cyan]obektclaw[/cyan]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
            console.print(response_panel)
            console.print()
            
            # Warn when context is getting tight
            try:
                pressure = agent._context_pressure()
            except (AttributeError, TypeError):
                pressure = 0.0
            if isinstance(pressure, (int, float)) and pressure > 0.8:
                console.print(Panel(
                    f"Context window is {int(pressure * 100)}% full — consider starting a new session.",
                    title="[yellow]Warning[/yellow]",
                    border_style="yellow",
                    box=box.ROUNDED,
                ))
                console.print()
    finally:
        agent.close()
        store.close()
    return 0

