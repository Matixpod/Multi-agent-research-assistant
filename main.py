#!/usr/bin/env python3
# main.py
"""
Multi-Agent Research Assistant — CLI entry point.

Usage:
    python main.py --query "What is quantum computing?"
    python main.py --query "Latest advances in fusion energy" --verbose --max-iterations 5
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config.settings import get_settings
from src.graph.state import create_initial_state
from src.graph.workflow import create_research_graph

console = Console()

# Load .env before anything else
load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant — AI-powered research with local LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py --query "What is quantum computing?"\n'
            '  python main.py --query "Climate change effects" --verbose\n'
            '  python main.py --query "AI in healthcare" --max-iterations 5\n'
        ),
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="The research question to investigate.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose output with detailed agent logs.",
    )
    parser.add_argument(
        "--max-iterations", "-m",
        type=int,
        default=None,
        help="Maximum supervisor loop iterations (default: from settings).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path for the Markdown report (auto-generated if omitted).",
    )
    return parser.parse_args()


def save_report(report: str, query: str, output_path: str | None = None) -> Path:
    """Save the final report to a Markdown file.

    Args:
        report: The Markdown report content.
        query: The original research query (used for filename).
        output_path: Optional explicit output path.

    Returns:
        The path to the saved report file.
    """
    if output_path:
        path = Path(output_path)
    else:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in query)[:50].strip()
        safe_name = safe_name.replace(" ", "_").lower()
        path = reports_dir / f"report_{safe_name}_{timestamp}.md"

    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    """Run the multi-agent research pipeline."""
    args = parse_args()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    console.print(
        Panel.fit(
            "[bold cyan]🔬 Multi-Agent Research Assistant[/bold cyan]\n"
            "[dim]Powered by LangGraph + Ollama (local LLMs)[/dim]",
            border_style="cyan",
        )
    )
    console.print(f"\n[bold]Query:[/bold] {args.query}\n")

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    settings = get_settings()
    max_iter = args.max_iterations or settings.max_research_loops

    console.print(f"[dim]Supervisor model: {settings.supervisor_model}[/dim]")
    console.print(f"[dim]Agent model:      {settings.agent_model}[/dim]")
    console.print(f"[dim]Ollama URL:       {settings.ollama_base_url}[/dim]")
    console.print(f"[dim]Max iterations:   {max_iter}[/dim]\n")

    # ------------------------------------------------------------------
    # Build graph & initial state
    # ------------------------------------------------------------------
    try:
        graph = create_research_graph()
    except Exception as exc:
        console.print(f"[bold red]✗ Failed to build graph:[/bold red] {exc}")
        sys.exit(1)

    initial_state = create_initial_state(query=args.query, max_iterations=max_iter)

    # ------------------------------------------------------------------
    # Run the pipeline
    # ------------------------------------------------------------------
    console.print(
        Panel("[bold]Starting research pipeline...[/bold]", border_style="blue")
    )

    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        console.print(f"\n[bold red]✗ Pipeline failed:[/bold red] {exc}")
        if args.verbose:
            console.print_exception()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    report = final_state.get("final_report", "")
    errors = final_state.get("errors", [])

    if errors:
        console.print("\n[bold yellow]⚠ Errors encountered during research:[/bold yellow]")
        for err in errors:
            console.print(f"  • {err}")

    if report:
        console.print("\n")
        console.print(
            Panel("[bold green]📄 Final Report[/bold green]", border_style="green")
        )
        console.print(Markdown(report))

        # Save to file
        report_path = save_report(report, args.query, args.output)
        console.print(f"\n[bold green]✓ Report saved to:[/bold green] {report_path}")
    else:
        console.print("\n[bold red]✗ No report was generated.[/bold red]")
        if args.verbose:
            console.print("[dim]Final state:[/dim]")
            for key, value in final_state.items():
                if key != "messages":
                    console.print(f"  {key}: {value}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    console.print(
        Panel.fit(
            f"[bold]Iterations used:[/bold] {final_state.get('iteration', 0)}/{max_iter}\n"
            f"[bold]Claims found:[/bold] {len(final_state.get('key_claims', []))}\n"
            f"[bold]Verified:[/bold] {len(final_state.get('verified_claims', []))}\n"
            f"[bold]Disputed:[/bold] {len(final_state.get('disputed_claims', []))}",
            title="[bold cyan]Summary[/bold cyan]",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    main()
