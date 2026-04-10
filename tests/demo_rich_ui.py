#!/usr/bin/env python3
"""Quick visual demo of the new Rich-styled CLI."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obektclaw.gateways.cli import (
    console,
    show_banner,
    show_help,
    show_setup,
    _first_run_welcome,
)
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

console = Console()

def demo():
    """Show off all the visual improvements."""
    console.clear()
    
    # Banner
    console.print("[bold cyan]1. Main Banner[/bold cyan]")
    console.print(Rule(style="cyan"))
    show_banner()
    console.print()
    
    # First run welcome
    console.print("[bold cyan]2. First Run Welcome[/bold cyan]")
    console.print(Rule(style="cyan"))
    _first_run_welcome()
    console.print()
    
    # Help
    console.print("[bold cyan]3. Help Screen[/bold cyan]")
    console.print(Rule(style="cyan"))
    show_help()
    console.print()
    
    # Sample agent response
    console.print("[bold cyan]4. Agent Response Example[/bold cyan]")
    console.print(Rule(style="cyan"))
    sample_response = """
Here's how to use `httpx` for async HTTP requests:

```python
import httpx

async def fetch_data(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

**Key benefits:**
- Async support
- Type hints
- Modern API

> Always prefer httpx over requests for new projects.
"""
    panel = Panel(
        Markdown(sample_response),
        title="[cyan]obektclaw[/cyan]",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)
    console.print()
    
    # Error example
    console.print("[bold cyan]5. Error Styling[/bold cyan]")
    console.print(Rule(style="cyan"))
    error_panel = Panel(
        "Invalid API key. Check OBEKTCLAW_LLM_API_KEY in ~/.obektclaw/.env",
        title="[red]Error[/red]",
        border_style="red",
    )
    console.print(error_panel)
    console.print()
    
    # Warning
    console.print("[bold cyan]6. Warning Styling[/bold cyan]")
    console.print(Rule(style="cyan"))
    warning_panel = Panel(
        "Context window is 85% full — consider starting a new session.",
        title="[yellow]Warning[/yellow]",
        border_style="yellow",
    )
    console.print(warning_panel)
    console.print()
    
    console.print("[bold green]✓ Demo complete![/bold green]")
    console.print()

if __name__ == "__main__":
    demo()
