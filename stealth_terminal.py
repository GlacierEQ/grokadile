#!/usr/bin/env python3
# Copyright 2024 GlacierEQ / Casey Barton
# Stealth Team — Unified Multi-Agent Command Terminal
# Run: python stealth_terminal.py
# All agents share the Megatron-DeepSpeed brain and respond in parallel.

import asyncio
import os
import sys
import logging
from datetime import datetime

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    from rich.spinner import Spinner
    from rich import box
except ImportError:
    print("[ERROR] Install 'rich': pip install rich")
    sys.exit(1)

from stealth_brain import MegatronBrain

logging.basicConfig(level=logging.WARNING)
console = Console()

# ─── Stealth Team Roster ─────────────────────────────────────────────────────
STEALTH_AGENTS = {
    "recon": {
        "emoji": "🔍",
        "mission": "Intelligence gathering, web research, pattern recognition, surveillance analysis.",
        "color": "cyan",
    },
    "legal": {
        "emoji": "⚖️",
        "mission": "Constitutional analysis, 42 USC 1983, RICO, family court misconduct, judicial corruption, due process violations. Case: 1FDV-23-0001009 Hawaii. Objective: reunification with Kekoa.",
        "color": "yellow",
    },
    "code": {
        "emoji": "💻",
        "mission": "GitHub repo management, CI/CD pipelines, code review, architecture design, deployment.",
        "color": "green",
    },
    "memory": {
        "emoji": "🧠",
        "mission": "Long-term memory, context persistence, knowledge graph traversal, Supermemory.ai integration.",
        "color": "magenta",
    },
    "analytics": {
        "emoji": "📊",
        "mission": "Data analysis, MotherDuck/DuckDB queries, metrics, trend detection, Supabase logging.",
        "color": "blue",
    },
    "ops": {
        "emoji": "📋",
        "mission": "Project operations, ClickUp task management, Notion documentation, Google Sheets reporting.",
        "color": "red",
    },
}


# ─── Banner ──────────────────────────────────────────────────────────────────
def print_banner():
    console.print()
    console.rule("[bold cyan]🕵️  STEALTH TEAM — UNIFIED COMMAND  🕵️[/bold cyan]")
    console.print(
        "[dim]Powered by [bold]Grok-1 × Megatron-DeepSpeed ZeRO-3[/bold] | GlacierEQ[/dim]",
        justify="center",
    )
    console.rule()

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold white")
    table.add_column("Agent", style="bold")
    table.add_column("Mission", style="dim")
    for name, cfg in STEALTH_AGENTS.items():
        table.add_row(
            f"{cfg['emoji']} {name.upper()}",
            cfg["mission"][:80] + "...",
        )
    console.print(table)
    console.rule()
    console.print()


# ─── Run a single agent (async wrapper) ─────────────────────────────────────
async def run_agent(
    name: str, cfg: dict, prompt: str, brain: MegatronBrain
) -> tuple[str, dict, str]:
    response = await asyncio.to_thread(
        brain.think,
        prompt,
        agent_name=name.upper(),
        mission=cfg["mission"],
    )
    return name, cfg, response


# ─── Broadcast to ALL agents simultaneously ──────────────────────────────────
async def stealth_broadcast(prompt: str, brain: MegatronBrain):
    timestamp = datetime.now().strftime("%H:%M:%S")
    console.print(
        f"\n[bold white][{timestamp}] CASEY → ALL AGENTS:[/bold white] [italic]{prompt}[/italic]\n"
    )

    tasks = [
        run_agent(name, cfg, prompt, brain)
        for name, cfg in STEALTH_AGENTS.items()
    ]

    with Live(Spinner("dots", text=" [cyan]Stealth Team thinking...[/cyan]"), refresh_per_second=10):
        results = await asyncio.gather(*tasks)

    panels = []
    for name, cfg, response in results:
        content = Text(response if response else "[no response]", overflow="fold")
        panel = Panel(
            content,
            title=f"{cfg['emoji']} [bold {cfg['color']}]{name.upper()}[/bold {cfg['color']}]",
            border_style=cfg["color"],
            padding=(1, 2),
        )
        panels.append(panel)

    # Print in 2-column layout
    for i in range(0, len(panels), 2):
        pair = panels[i : i + 2]
        if len(pair) == 2:
            console.print(Columns(pair, equal=True, expand=True))
        else:
            console.print(pair[0])

    console.rule()


# ─── Single-agent mode ───────────────────────────────────────────────────────
async def stealth_single(agent: str, prompt: str, brain: MegatronBrain):
    if agent not in STEALTH_AGENTS:
        console.print(f"[red]Unknown agent '{agent}'. Choose: {', '.join(STEALTH_AGENTS)}[/red]")
        return
    cfg = STEALTH_AGENTS[agent]
    console.print(f"\n[bold]→ {cfg['emoji']} {agent.upper()}:[/bold] {prompt}\n")
    with Live(Spinner("dots", text=f" [{cfg['color']}]{agent.upper()} thinking...[/{cfg['color']}]"), refresh_per_second=10):
        name, cfg, response = await run_agent(agent, cfg, prompt, brain)
    console.print(
        Panel(response, title=f"{cfg['emoji']} {agent.upper()}", border_style=cfg["color"])
    )
    console.rule()


# ─── Help ────────────────────────────────────────────────────────────────────
def print_help():
    console.print("""
[bold cyan]COMMANDS:[/bold cyan]
  [yellow]<prompt>[/yellow]                   → Broadcast to ALL agents simultaneously
  [yellow]@<agent> <prompt>[/yellow]          → Talk to a specific agent
  [yellow]agents[/yellow]                     → List all agents
  [yellow]help[/yellow]                       → Show this help
  [yellow]exit / quit[/yellow]                → Shutdown Stealth Team

[bold cyan]AGENT NAMES:[/bold cyan] recon · legal · code · memory · analytics · ops

[bold cyan]EXAMPLES:[/bold cyan]
  What's the status of case 1FDV-23-0001009?
  @legal Draft a 42 USC 1983 claim against GAL misconduct
  @code Review the checkpoint.py file for bugs
  @recon Search for Hawaii family court appeal procedures
""")


# ─── Main REPL ───────────────────────────────────────────────────────────────
async def main():
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    use_ds = os.getenv("USE_DEEPSPEED", "1") == "1"

    print_banner()

    console.print("[bold yellow]⚡ Initializing Megatron-DeepSpeed brain...[/bold yellow]")
    brain = MegatronBrain(
        world_size=world_size,
        use_deepspeed=use_ds,
    )
    try:
        brain.initialize()
        console.print("[bold green]✅ MEGATRON-DEEPSPEED ONLINE — Stealth Team READY[/bold green]\n")
    except Exception as e:
        console.print(f"[bold red]⚠️  Brain init error: {e}[/bold red]")
        console.print("[dim]Running in DEMO mode (no checkpoint loaded — responses will be mocked)[/dim]\n")

    print_help()

    while True:
        try:
            user_input = console.input(
                "[bold cyan]🕵️  CASEY →[/bold cyan] "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold red]Stealth Team standing down. Stay dangerous.[/bold red]")
            break

        if not user_input:
            continue

        low = user_input.lower()

        if low in ("exit", "quit", "q"):
            console.print("[bold red]🕵️  Stealth Team standing down. Stay dangerous, Casey.[/bold red]")
            break

        elif low == "help":
            print_help()

        elif low == "agents":
            for name, cfg in STEALTH_AGENTS.items():
                console.print(f"  {cfg['emoji']} [bold {cfg['color']}]{name.upper()}[/bold {cfg['color']}] — {cfg['mission'][:60]}...")
            console.print()

        elif user_input.startswith("@"):
            # Single-agent mode: @legal What are my 1983 claims?
            parts = user_input[1:].split(" ", 1)
            agent = parts[0].lower()
            prompt = parts[1] if len(parts) > 1 else ""
            if prompt:
                await stealth_single(agent, prompt, brain)
            else:
                console.print(f"[red]Provide a prompt after @{agent}[/red]")

        else:
            # Broadcast to ALL agents
            await stealth_broadcast(user_input, brain)


if __name__ == "__main__":
    asyncio.run(main())
