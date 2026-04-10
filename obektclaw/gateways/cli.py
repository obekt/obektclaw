"""Interactive CLI gateway. A modern REPL on top of Agent.run_once.

Uses prompt_toolkit for interactive slash-command completion, styled
prompts, history persistence, multi-line input, file completion, and
a status toolbar.
Enhanced with Rich for beautiful terminal output with syntax highlighting.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter, merge_completers
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit.filters import Condition
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.live import Live
from rich import box, print as rprint

from ..agent import Agent
from ..config import CONFIG, load_config
from ..memory.store import Store
from ..model_context import get_context_window, list_known_models, guess_context_window
from ..skills import SkillManager

# Initialize Rich console with custom theme support
console = Console()

# ── Theme definitions ────────────────────────────────────────────────────────

THEMES = {
    "catppuccin": {
        "name": "Catppuccin",
        "primary": "#00d7ff",      # cyan
        "secondary": "#a6e3a1",    # green
        "warning": "#f9e2af",      # yellow
        "error": "#f38ba8",        # red
        "dim": "#6c7086",
        "bg": "#1e1e2e",
        "bg_alt": "#313244",
        "panel_border": "cyan",
        "prompt_arrow": "#00d7ff bold",
    },
    "dracula": {
        "name": "Dracula",
        "primary": "#bd93f9",      # purple
        "secondary": "#50fa7b",    # green
        "warning": "#f1fa8c",      # yellow
        "error": "#ff5555",        # red
        "dim": "#6272a4",
        "bg": "#282a36",
        "bg_alt": "#44475a",
        "panel_border": "purple",
        "prompt_arrow": "#bd93f9 bold",
    },
    "monokai": {
        "name": "Monokai",
        "primary": "#a6e22e",      # green
        "secondary": "#66d9ef",    # blue
        "warning": "#e6db74",      # yellow
        "error": "#f92672",        # red
        "dim": "#75715e",
        "bg": "#272822",
        "bg_alt": "#3e3d32",
        "panel_border": "green",
        "prompt_arrow": "#a6e22e bold",
    },
    "nord": {
        "name": "Nord",
        "primary": "#88c0d0",      # cyan
        "secondary": "#a3be8c",    # green
        "warning": "#ebcb8b",      # yellow
        "error": "#bf616a",        # red
        "dim": "#4c566a",
        "bg": "#2e3440",
        "bg_alt": "#3b4252",
        "panel_border": "cyan",
        "prompt_arrow": "#88c0d0 bold",
    },
    "gruvbox": {
        "name": "Gruvbox",
        "primary": "#fe8019",      # orange
        "secondary": "#b8bb26",    # green
        "warning": "#fabd2f",      # yellow
        "error": "#fb4934",        # red
        "dim": "#928374",
        "bg": "#282828",
        "bg_alt": "#3c3836",
        "panel_border": "#fe8019",  # orange (Rich doesn't accept "orange" as named color)
        "prompt_arrow": "#fe8019 bold",
    },
}

# Current theme (can be changed via /theme command)
_current_theme = "catppuccin"


def get_theme() -> dict:
    """Get the current theme configuration."""
    return THEMES.get(_current_theme, THEMES["catppuccin"])


def set_theme(name: str) -> bool:
    """Set the current theme. Returns True if successful."""
    if name in THEMES:
        _current_theme = name
        return True
    return False


# ── Slash commands with icons ────────────────────────────────────────────────

SLASH_COMMANDS: list[tuple[str, str, str]] = [
    ("/help",       "📚  Show help and available commands", "help"),
    ("/skills",     "🎯  List known skills", "skills"),
    ("/memory",     "🧠  Search persistent memory  (/memory <query>)", "memory"),
    ("/traits",     "👤  Show your user model", "traits"),
    ("/model",      "🤖  Show or change the current LLM model", "model"),
    ("/compact",    "🗜️  Compact conversation history to save context", "compact"),
    ("/sessions",   "📜  Browse recent sessions", "sessions"),
    ("/setup",      "⚙️  Guided setup (Telegram, MCP, etc.)", "setup"),
    ("/theme",      "🎨  Change color theme", "theme"),
    ("/clear",      "🧹  Clear the screen", "clear"),
    ("/exit",       "👋  Quit the session", "exit"),
]


class SlashCompleter(Completer):
    """Show an interactive selection menu when the user types '/'."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return
        # Match against the typed prefix (e.g. "/sk" → "/skills")
        for cmd, desc, _icon in SLASH_COMMANDS:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=HTML(f"<b>{cmd}</b>"),
                    display_meta=desc,
                )


class SmartCompleter(Completer):
    """Smart completer: slash commands when input starts with '/', otherwise file paths."""

    def __init__(self):
        self._slash_completer = SlashCompleter()
        self._path_completer = PathCompleter(
            get_paths=lambda: [str(CONFIG.workdir)],
            file_filter=lambda name: not name.startswith("."),
        )

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        # Only show slash commands when typing '/'
        if text.startswith("/"):
            return self._slash_completer.get_completions(document, complete_event)
        # Otherwise show file path completion (for typing file paths in messages)
        return self._path_completer.get_completions(document, complete_event)


# ── Dynamic styles based on theme ────────────────────────────────────────────

def _make_prompt_style() -> PromptStyle:
    """Create prompt style based on current theme."""
    theme = get_theme()
    return PromptStyle.from_dict({
        # Prompt arrow
        "prompt":       theme["prompt_arrow"],
        # Completion menu
        "completion-menu":                f"bg:{theme['bg']} #cdd6f4",
        "completion-menu.completion":     f"bg:{theme['bg']} #cdd6f4",
        "completion-menu.completion.current": f"bg:{theme['bg_alt']} #cdd6f4 bold",
        "completion-menu.meta":           f"bg:{theme['bg']} {theme['dim']} italic",
        "completion-menu.meta.current":   f"bg:{theme['bg_alt']} #a6adc8 italic",
        # Bottom toolbar
        "bottom-toolbar":       f"bg:{theme['bg']} {theme['dim']}",
        "bottom-toolbar.text":  theme["dim"],
    })


def _make_prompt_message() -> list:
    """Create prompt message with themed arrow."""
    theme = get_theme()
    return [("class:prompt", "❯ ")]


# ── Key bindings ─────────────────────────────────────────────────────────────

def _make_key_bindings() -> KeyBindings:
    """Create custom key bindings for the REPL."""
    kb = KeyBindings()

    # Ctrl-L: Clear screen
    @kb.add("c-l")
    def _(event):
        event.app.renderer.clear()

    # Ctrl-U: Clear input line
    @kb.add("c-u")
    def _(event):
        event.current_buffer.text = ""

    # Meta-Enter (Alt+Enter or Esc+Enter): Submit with newline support
    @kb.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    # Ctrl-R: Search history (built-in, but we can add custom handling)
    # This is already handled by prompt_toolkit's default behavior

    # Ctrl-D: Exit (handled by prompt_toolkit EOFError)

    return kb


# Legacy static styles (for backwards compatibility)
PROMPT_STYLE = _make_prompt_style()
PROMPT_MESSAGE = _make_prompt_message()


# ── Context size formatting ─────────────────────────────────────────────────

def _format_tokens(n: int) -> str:
    """Format token count compactly: 1234 → '1.2k', 128000 → '128k'."""
    if n < 1_000:
        return str(n)
    if n < 10_000:
        return f"{n / 1_000:.1f}k"
    return f"{n // 1_000}k"


def _make_toolbar(agent_ref: list):
    """Return a toolbar callable with icons and theme colors."""
    def _bottom_toolbar():
        theme = get_theme()
        parts = [
            f" <b>/</b> 📚 commands  ",
            f"<style bg='{theme['bg_alt']}'> ⌃C </style> cancel  ",
            f"<style bg='{theme['bg_alt']}'> ⌃D </style> exit  ",
            f"<style bg='{theme['bg_alt']}'> ⌃L </style> clear  ",
            f"<style bg='{theme['bg_alt']}'> ⌃R </style> history",
        ]
        agent = agent_ref[0] if agent_ref else None
        if agent and agent.last_usage:
            u = agent.last_usage
            ctx_window = agent.context_window
            used = u.prompt_tokens + u.completion_tokens
            pct = min(int(used / ctx_window * 100), 100) if ctx_window else 0
            # Color based on pressure: green < 50%, yellow 50-80%, red > 80%
            if pct < 50:
                color = theme["secondary"]
                icon = "✅"
            elif pct < 80:
                color = theme["warning"]
                icon = "⚠️"
            else:
                color = theme["error"]
                icon = "🔥"
            parts.append(
                f"  <style bg='{theme['bg_alt']}' fg='{color}'> "
                f"{icon} {_format_tokens(used)}/{_format_tokens(ctx_window)} ({pct}%) </style>"
            )
        return HTML("".join(parts))
    return _bottom_toolbar


# ── Session factory with enhanced features ───────────────────────────────────

def _make_session(agent_ref: list) -> PromptSession:
    """Create a PromptSession with multi-line input, completion, and keybindings."""
    history_path = CONFIG.home / "history.txt"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(
        message=_make_prompt_message(),
        style=_make_prompt_style(),
        completer=SmartCompleter(),
        complete_while_typing=True,
        history=FileHistory(str(history_path)),
        bottom_toolbar=_make_toolbar(agent_ref),
        mouse_support=False,
        key_bindings=_make_key_bindings(),
        multiline=False,  # Enter submits immediately; Esc+Enter adds newline
        prompt_continuation="  ",  # Indentation for continuation lines
    )


# ── Syntax highlighting for code blocks ───────────────────────────────────────

class SyntaxMarkdown(Markdown):
    """Custom Markdown renderer with syntax highlighting for code blocks."""

    def __init__(self, text: str, code_theme: str = "monokai", *args, **kwargs):
        self._code_theme = code_theme
        super().__init__(text, *args, **kwargs)

    def render_code_block(self, tokens, idx, options, env):
        """Render code blocks with syntax highlighting."""
        token = tokens[idx]
        code = token.content
        lang = token.info.strip() if token.info else ""

        if lang:
            try:
                syntax = Syntax(
                    code,
                    lang,
                    theme=self._code_theme,
                    line_numbers=False,
                    word_wrap=False,
                )
                return syntax
            except Exception:
                # Fallback to plain code block if syntax highlighting fails
                pass

        # Default fallback
        return super().render_code_block(tokens, idx, options, env)


def render_response(reply: str, stream: bool = False) -> None:
    """Render the agent's response with syntax highlighting and optional streaming."""
    theme = get_theme()

    # Parse and highlight code blocks
    # First extract code blocks to apply syntax highlighting
    import re
    # Use finditer to get positions, avoiding the split/replace bug
    code_block_matches = list(re.finditer(r'```(\w*)\n(.*?)```', reply, re.DOTALL))

    if code_block_matches:
        # Build a rich display with syntax-highlighted code
        console.print()
        parts = []
        last_end = 0

        for match in code_block_matches:
            lang = match.group(1)
            code = match.group(2)
            
            # Text before this code block
            text_before = reply[last_end:match.start()]
            if text_before.strip():
                parts.append(Markdown(text_before.strip()))

            # Syntax highlight the code
            try:
                syntax_theme = "monokai"  # Good contrast for dark terminals
                syntax = Syntax(
                    code.strip(),
                    lang if lang else "text",
                    theme=syntax_theme,
                    line_numbers=True,
                )
                parts.append(syntax)
            except Exception:
                parts.append(Panel(code.strip(), style="dim", box=box.SIMPLE))

            last_end = match.end()

        # Text after the last code block
        text_after = reply[last_end:]
        if text_after.strip():
            parts.append(Markdown(text_after.strip()))

        # Print all parts
        for part in parts:
            console.print(part)

    else:
        # No code blocks, just render as markdown
        if stream:
            _stream_text(reply)
        else:
            md = Markdown(reply)
            response_panel = Panel(
                md,
                title=f"[{theme['primary']}]🤖 obektclaw[/]",
                border_style=theme["panel_border"],
                box=box.ROUNDED,
                padding=(1, 2),
            )
            console.print()
            console.print(response_panel)


def _stream_text(text: str, delay: float = 0.02) -> None:
    """Stream text with a typing animation effect."""
    theme = get_theme()
    buffer = ""

    def generate():
        for char in text:
            buffer += char
            yield Text(buffer, style=theme["primary"])

    with Live(generate(), console=console, refresh_per_second=10) as live:
        for char in text:
            buffer += char
            live.update(Text(buffer))
            time.sleep(delay)


# ── Progress indicators ──────────────────────────────────────────────────────

def show_progress(description: str, total: int = None) -> Progress:
    """Create a themed progress bar for long operations."""
    theme = get_theme()
    return Progress(
        SpinnerColumn(style=theme["primary"]),
        TextColumn(f"[{theme['primary']}]{description}"),
        BarColumn(style=theme["primary"], complete_style=theme["secondary"]),
        TimeElapsedColumn(),
        console=console,
    )


# ── Banners / help ──────────────────────────────────────────────────────────

def show_banner():
    """Show a rich styled banner with themed colors."""
    theme = get_theme()
    title = Text("🤖 obektclaw", style=f"bold {theme['primary']}")
    subtitle = Text("self-improving AI agent · memory · skills · tools", style="dim")
    version = Text("v1.0.1", style=theme["dim"])

    panel = Panel(
        Text.assemble(title, "\n", subtitle, "\n", version),
        box=box.DOUBLE,
        border_style=theme["panel_border"],
        padding=(1, 2),
    )
    console.print(panel)
    console.print()
    console.print("  💬 Type a message and hit enter.", end="")
    console.print("  📚 Type / for commands.", style="dim italic")
    console.print("  ⌨️  Press ⌃R to search history, ⌃L to clear screen.", style="dim italic")
    console.print()


def show_help():
    """Show rich formatted help with icons and themed colors."""
    theme = get_theme()
    console.print(Panel(
        "🤖 obektclaw — self-improving AI agent",
        style=f"bold {theme['primary']}",
        padding=(0, 1),
    ))
    console.print()

    # Feature list with icons
    features = [
        (f"[{theme['primary']}]🧠 Memory[/]", "session history · persistent facts · user model"),
        (f"[{theme['primary']}]🔧 Tools[/]", "files · bash · python · web · memory · skills · sub-agents"),
        (f"[{theme['primary']}]🎯 Skills[/]", "auto-created markdown guides that improve after each use"),
        (f"[{theme['primary']}]🎨 Themes[/]", "5 color themes: catppuccin, dracula, monokai, nord, gruvbox"),
    ]
    for title, desc in features:
        console.print(f"  {title:20s} {desc}")

    console.print()
    console.print(Rule("📚 Commands", style=theme["primary"]))

    table = Table.grid(padding=(0, 2))
    table.add_column("Command", style=f"bold {theme['primary']}")
    table.add_column("Description", style="dim")

    for cmd, desc, _icon in SLASH_COMMANDS:
        table.add_row(cmd, desc)

    console.print(table)
    console.print()
    console.print("  💡 Tip: type / to see interactive command picker.", style="italic dim")
    console.print("  ⌨️  Keybindings: ⌃L clear · ⌃R history · ⌃U undo · ⌃D exit", style="italic dim")
    console.print("  📝 Multi-line: Press Esc+Enter to add newlines", style="italic dim")
    console.print()

def show_setup(config=None):
    """Show current configuration and guided options with themed styling."""
    cfg = config or CONFIG
    theme = get_theme()

    console.print(Panel("⚙️  obektclaw Setup", style=f"bold {theme['primary']}", padding=(0, 1)))
    console.print()

    # Current configuration
    console.print(f"[bold]Current configuration:[/bold]")
    config_items = [
        ("🏠 OBEKTCLAW_HOME", str(cfg.home)),
        ("📦 Database", str(cfg.db_path)),
        ("🎯 Skills", str(cfg.skills_dir)),
        ("📝 Logs", str(cfg.logs_dir)),
    ]

    table = Table.grid(padding=(0, 2))
    table.add_column("Setting", style=theme["primary"])
    table.add_column("Value", style="dim")

    for setting, value in config_items:
        table.add_row(setting, value)

    console.print(table)
    console.print()

    # LLM config
    console.print(f"  [{theme['primary']}]🤖 LLM:[/] {cfg.llm_model} via {cfg.llm_base_url}")
    console.print(f"  [{theme['primary']}]📄 Config:[/] {_env_file()}")
    console.print()

    # Current theme
    console.print(f"  [{theme['primary']}]🎨 Theme:[/] {THEMES[_current_theme]['name']}")
    console.print(f"    [dim]Change with: /theme <name>[/dim]")
    console.print()

    # Check if MCP config exists
    mcp_config = cfg.home / "mcp.json"
    if mcp_config.exists():
        console.print("  ✅ [green]MCP servers configured[/green]")
    else:
        console.print("  ○ [dim]MCP servers: not configured[/dim]")
        console.print(f"    Create [{theme['primary']}]{mcp_config}[/] to connect external tools")

    # Check Telegram
    if cfg.tg_token:
        console.print("  ✅ [green]Telegram bot configured[/green]")
        console.print(f"    Run: [{theme['primary']}]python -m obektclaw tg[/]")
    else:
        console.print("  ○ [dim]Telegram bot: not configured[/dim]")
        console.print("    To enable Telegram chat:")
        console.print("    1. Message @BotFather on Telegram")
        console.print("    2. Create a new bot with /newbot")
        console.print("    3. Copy the token")
        console.print(f"    4. Add to [{theme['primary']}]{_env_file()}[/]: OBEKTCLAW_TG_TOKEN=your_token")

    console.print()


def _first_run_welcome():
    """Show a welcome message on first run with enhanced visuals."""
    theme = get_theme()
    title = Text("👋 Welcome to obektclaw", style=f"bold {theme['primary']}")
    panel = Panel(
        title,
        box=box.DOUBLE,
        border_style=theme["panel_border"],
        padding=(0, 2),
    )
    console.print(panel)
    console.print()
    console.print("  🤖 I'm a self-improving AI agent with memory & skills.")
    console.print()

    steps = [
        ("1️⃣", "Remember", '"I always use httpx over requests"', "🧠"),
        ("2️⃣", "Execute", '"List all Python files here"', "🔧"),
        ("3️⃣", "Learn", "I create skills from patterns I discover", "🎯"),
        ("4️⃣", "Connect", "/setup to add Telegram or MCP servers", "⚙️"),
        ("5️⃣", "Customize", "/theme to change colors", "🎨"),
    ]

    for num, action, example, icon in steps:
        console.print(f"  [{theme['primary']}]{icon} {num}[/] [bold]{action}[/bold]  {example}")

    console.print()
    console.print("  📚 Type / for commands · 📖 /help for docs · 🚪 /exit to quit", style="dim italic")
    console.print("  ⌨️  ⌃L clear · ⌃R history · ⌃U undo · ⌃D exit", style="dim italic")
    console.print()


def show_theme_help():
    """Show available themes and current selection."""
    theme = get_theme()
    console.print()
    console.print(Panel(
        "🎨 Color Themes",
        style=f"bold {theme['primary']}",
        padding=(0, 1),
    ))
    console.print()

    table = Table(box=box.ROUNDED, border_style=theme["panel_border"])
    table.add_column("Theme", style=f"bold {theme['primary']}")
    table.add_column("Primary", style="bold")
    table.add_column("Description", style="dim")

    descriptions = {
        "catppuccin": "Pastel colors, soft and cozy",
        "dracula": "Dark purple, vibrant accents",
        "monokai": "Classic dark, green-primary",
        "nord": "Arctic, bluish-cold tones",
        "gruvbox": "Warm retro, orange-primary",
    }

    for name, t in THEMES.items():
        marker = "✓" if name == _current_theme else " "
        table.add_row(
            f"{marker} {name}",
            f"[{t['primary']}]████[/]",
            descriptions.get(name, ""),
        )

    console.print(table)
    console.print()
    console.print(f"  [{theme['primary']}]Current:[/] {THEMES[_current_theme]['name']}")
    console.print(f"  [dim]Change with: /theme <name>[/dim]")
    console.print()


def clear_screen():
    """Clear the terminal screen."""
    console.clear()
    show_banner()

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


def _repl(agent: Agent, store: Store, skills: SkillManager, config) -> int:
    """Shared REPL loop used by both run() and run_with_session()."""
    agent_ref = [agent]
    session = _make_session(agent_ref)
    theme = get_theme()

    try:
        while True:
            try:
                # Multi-line input: lines ending with \ continue
                line = session.prompt().strip()
            except KeyboardInterrupt:
                continue          # ctrl-c clears current input
            except EOFError:
                break             # ctrl-d exits
            if not line:
                continue
            if line in ("/exit", "/quit"):
                console.print(f"  [{theme['dim']}]👋 Goodbye![/]")
                break
            if line == "/help":
                show_help()
                continue
            if line == "/clear":
                clear_screen()
                continue
            if line.startswith("/theme"):
                args = line[len("/theme"):].strip()
                if not args:
                    show_theme_help()
                elif args in THEMES:
                    global _current_theme
                    _current_theme = args
                    theme = get_theme()
                    # Rebuild session with new theme
                    session = _make_session(agent_ref)
                    console.print()
                    console.print(Panel(
                        f"[{theme['secondary']}]✓ Theme changed to {THEMES[args]['name']}[/]",
                        title=f"[{theme['primary']}]🎨 Theme[/]",
                        border_style=theme["panel_border"],
                        box=box.ROUNDED,
                    ))
                    console.print()
                else:
                    console.print(f"  [{theme['error']}]Unknown theme: {args}[/]")
                    console.print(f"  [{theme['dim']}]Available: {', '.join(THEMES.keys())}[/]")
                continue
            if line == "/skills":
                all_skills = skills.list_all()
                if not all_skills:
                    console.print("  🎯 [dim]No skills yet. They auto-create when I learn patterns.[/dim]")
                else:
                    console.print()
                    table = Table(
                        title=f"🎯 Skills ({len(all_skills)})",
                        box=box.ROUNDED,
                        border_style=theme["panel_border"],
                    )
                    table.add_column("Name", style=f"bold {theme['primary']}")
                    table.add_column("Description", style="dim")
                    table.add_column("Uses", justify="right", style=theme["secondary"])

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
                    console.print("  🧠 [dim]usage: /memory <query>[/dim]")
                    continue
                results = list(agent.persistent.search(q))
                if not results:
                    console.print(f"  🧠 [dim]No memories found for '{q}'[/dim]")
                else:
                    console.print()
                    console.print(f"[bold {theme['primary']}]🧠 Memories[/] ({len(results)})")
                    console.print(Rule(style=theme["primary"]))
                    for f in results:
                        console.print(f"  [{theme['primary']}]●[/] {f.render()}")
                    console.print()
                continue
            if line == "/traits":
                traits = agent.user_model.all()
                if not traits:
                    console.print("  👤 [dim]No user model yet. It builds as we talk.[/dim]")
                else:
                    console.print()
                    console.print(f"[bold {theme['primary']}]👤 User model[/] ({len(traits)} traits)")
                    console.print(Rule(style=theme["primary"]))

                    table = Table(box=box.SIMPLE, padding=(0, 2))
                    table.add_column("Layer", style=f"bold {theme['primary']}")
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
                        f"[{theme['primary']}]🤖 Current Model:[/] {config.llm_model}\n"
                        f"[{theme['primary']}]⚡ Fast Model:[/] {config.llm_fast_model}\n"
                        f"[{theme['primary']}]📊 Context Window:[/] {agent.context_window:,} tokens "
                        f"(detected: {detected:,})",
                        title=f"[{theme['primary']}]🤖 Model Info[/]",
                        border_style=theme["panel_border"],
                        box=box.ROUNDED,
                    ))
                    console.print()
                    console.print(f"  [{theme['dim']}]Usage: /model <name> [context_window][/]")
                    console.print(f"  [{theme['dim']}]Example: /model gpt-4o 128000[/]")
                    console.print()
                elif args in ("list", "ls", "-l"):
                    # Show known models
                    models = list_known_models()
                    console.print()
                    console.print(f"[bold {theme['primary']}]🤖 Known Models[/] ({len(models)})")
                    console.print(Rule(style=theme["primary"]))

                    table = Table(box=box.SIMPLE, padding=(0, 1))
                    table.add_column("Model", style=f"bold {theme['primary']}")
                    table.add_column("Context", justify="right", style=theme["secondary"])
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
                            console.print(f"  [{theme['error']}]Invalid context window: {parts[1]}[/]")
                            continue

                    try:
                        result = agent.switch_model(
                            model=new_model,
                            context_window=new_context,
                        )
                        console.print()
                        console.print(Panel(
                            f"[{theme['secondary']}]✓ Model switched successfully[/]\n\n"
                            f"[{theme['primary']}]Model:[/] {result['model']}\n"
                            f"[{theme['primary']}]Fast Model:[/] {result['fast_model']}\n"
                            f"[{theme['primary']}]Context Window:[/] {result['context_window']:,} tokens"
                            + (f"\n[dim](saved to models.json)[/dim]" if result['was_overridden'] else ""),
                            title=f"[{theme['primary']}]🤖 Model Switch[/]",
                            border_style=theme["secondary"],
                            box=box.ROUNDED,
                        ))
                        console.print()
                    except Exception as e:
                        console.print(Panel(
                            f"Failed to switch model: {e}",
                            title=f"[{theme['error']}]Error[/]",
                            border_style=theme["error"],
                            box=box.ROUNDED,
                        ))
                        console.print()
                continue
            if line == "/compact":
                # Force compaction regardless of pressure
                console.print()
                with console.status(f"[{theme['primary']}]🗜️ Compacting context...[/]", spinner="dots"):
                    result = agent.compact_context(force=True)

                if result["compacted"]:
                    console.print(Panel(
                        f"[{theme['secondary']}]✓ Context compacted successfully[/]\n\n"
                        f"[{theme['primary']}]Summary:[/] {result['summary_length']} words\n"
                        f"[{theme['primary']}]Tokens saved:[/] ~{result['tokens_saved']:,}",
                        title=f"[{theme['primary']}]🗜️ Compaction[/]",
                        border_style=theme["secondary"],
                        box=box.ROUNDED,
                    ))
                else:
                    console.print(Panel(
                        f"[{theme['warning']}]Compaction skipped[/]\n\n"
                        f"[dim]Reason: {result['reason']}[/dim]",
                        title=f"[{theme['primary']}]🗜️ Compaction[/]",
                        border_style=theme["warning"],
                        box=box.ROUNDED,
                    ))
                console.print()
                continue
            if line == "/sessions":
                _show_sessions(store)
                continue

            # ── Agent turn ──────────────────────────────────────────────
            # Use the agent's status callback to show a dynamic spinner with theme
            status_obj = None
            def _on_status(msg: str):
                nonlocal status_obj
                if msg:
                    if status_obj is None:
                        status_obj = console.status(f"[{theme['primary']}]⏳ {msg}[/]", spinner="dots")
                        status_obj.start()
                    else:
                        status_obj.update(f"[{theme['primary']}]⏳ {msg}[/]")
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
                theme = get_theme()  # Refresh theme for error display
                if "401" in err or "auth" in err.lower() or "api key" in err.lower():
                    console.print(Panel(
                        f"🔑 Invalid API key. Check OBEKTCLAW_LLM_API_KEY in {_env_file()}",
                        title=f"[{theme['error']}]Error[/]",
                        border_style=theme["error"],
                        box=box.ROUNDED,
                    ))
                elif "connection" in err.lower() or "refused" in err.lower():
                    console.print(Panel(
                        f"🔌 Can't reach LLM at {config.llm_base_url}",
                        title=f"[{theme['error']}]Error[/]",
                        border_style=theme["error"],
                        box=box.ROUNDED,
                    ))
                elif "429" in err or "rate" in err.lower():
                    console.print(Panel(
                        "⏱️ Rate limited. Wait a moment and try again.",
                        title=f"[{theme['error']}]Error[/]",
                        border_style=theme["error"],
                        box=box.ROUNDED,
                    ))
                elif "model" in err.lower() and ("not found" in err.lower() or "404" in err):
                    console.print(Panel(
                        f"🤖 Model '{config.llm_model}' not found.",
                        title=f"[{theme['error']}]Error[/]",
                        border_style=theme["error"],
                        box=box.ROUNDED,
                    ))
                else:
                    console.print(Panel(
                        str(e),
                        title=f"[{theme['error']}]Error[/]",
                        border_style=theme["error"],
                        box=box.ROUNDED,
                    ))
                continue

            # Display agent response with syntax highlighting
            render_response(reply)

            # Warn when context is getting tight
            try:
                pressure = agent._context_pressure()
            except (AttributeError, TypeError):
                pressure = 0.0
            theme = get_theme()
            if isinstance(pressure, (int, float)) and pressure > 0.8:
                console.print(Panel(
                    f"🔥 Context window is {int(pressure * 100)}% full — consider starting a new session.",
                    title=f"[{theme['warning']}]⚠️ Warning[/]",
                    border_style=theme["warning"],
                    box=box.ROUNDED,
                ))
                console.print()
    finally:
        agent.close()
        store.close()
    return 0


def _show_sessions(store: Store) -> None:
    """Show recent sessions in the interactive CLI with themed styling."""
    from ..sessions import list_sessions

    theme = get_theme()
    sessions = list_sessions(store, limit=15)
    if not sessions:
        console.print("  📜 [dim]No past sessions.[/dim]")
        return

    console.print()
    table = Table(
        title="📜 Recent Sessions",
        box=box.ROUNDED,
        border_style=theme["panel_border"],
    )
    table.add_column("ID", style=f"bold {theme['primary']}", justify="right")
    table.add_column("Started", style="white")
    table.add_column("Duration", justify="right", style="dim")
    table.add_column("GW", style="dim")
    table.add_column("Msgs", justify="right", style=theme["secondary"])
    table.add_column("Preview", style="dim", max_width=40)

    for s in sessions:
        table.add_row(
            str(s.id),
            s.started_str,
            s.duration_str,
            s.gateway,
            str(s.message_count),
            s.preview,
        )
    console.print(table)
    console.print()
    console.print(
        f"  [{theme['dim']}]Resume: [{theme['primary']}]python -m obektclaw sessions resume <id>[/]  "
        f"Export: [{theme['primary']}]python -m obektclaw sessions export <id>[/][/]"
    )
    console.print()


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

    if is_first_run:
        _first_run_welcome()
    else:
        show_banner()

    return _repl(agent, store, skills, config)


def run_with_session(session_id: int, info) -> int:
    """Resume an old session in CLI mode with themed styling."""
    config = CONFIG
    if not _check_config():
        result = _setup_wizard()
        if result is None:
            return 1
        config = result

    store = Store(config.db_path)
    skills = SkillManager(store, config.skills_dir, config.bundled_skills_dir)
    theme = get_theme()

    agent = Agent(
        config=config, store=store, skills=skills,
        gateway="cli", user_key="cli-local",
        session_id=session_id,
    )

    console.print(Panel(
        f"📜 Resuming session #{info.id}\n"
        f"[{theme['primary']}]Started:[/] {info.started_str}  "
        f"[{theme['primary']}]Messages:[/] {info.message_count}  "
        f"[{theme['primary']}]Gateway:[/] {info.gateway}\n"
        f"[dim]{info.preview}[/dim]",
        title=f"[{theme['primary']}]Session Resume[/]",
        border_style=theme["secondary"],
        box=box.ROUNDED,
    ))
    console.print()

    return _repl(agent, store, skills, config)

