# src/agents/synthesizer.py
"""
Synthesizer Agent — generates a Markdown research report from verified data.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from src.config.settings import get_settings
from src.prompts.templates import SYNTHESIZER_SYSTEM_PROMPT

console = Console()


def synthesizer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Generate a polished Markdown report from the research and verification data.

    Args:
        state: Current ``ResearchState`` dictionary.

    Returns:
        Partial state update with ``final_report`` and ``messages``.
    """
    settings = get_settings()
    llm = settings.get_agent_llm()

    research_notes: str = state.get("research_notes", "")
    verified_claims: list[str] = state.get("verified_claims", [])
    disputed_claims: list[str] = state.get("disputed_claims", [])
    verification_results: list[dict] = state.get("verification_results", [])
    query: str = state.get("query", "")

    console.print("\n[bold magenta]📝 Synthesizer Agent[/bold magenta] starting...")

    # ------------------------------------------------------------------
    # Build context for the LLM
    # ------------------------------------------------------------------
    verified_str = "\n".join(f"  ✅ {c}" for c in verified_claims) or "  (none)"
    disputed_str = "\n".join(f"  ❌ {c}" for c in disputed_claims) or "  (none)"

    verification_detail = ""
    for vr in verification_results:
        claim = vr.get("claim", "N/A")
        verdict = vr.get("verdict", "unknown")
        evidence = vr.get("evidence", "N/A")
        confidence = vr.get("confidence", "N/A")
        icon = {"confirmed": "✅", "disputed": "❌"}.get(verdict, "⚠️")
        verification_detail += (
            f"  {icon} [{confidence}] {claim}\n"
            f"     Evidence: {evidence}\n\n"
        )

    user_prompt = (
        f"Original research query: {query}\n\n"
        f"Research notes:\n{research_notes}\n\n"
        f"Confirmed claims:\n{verified_str}\n\n"
        f"Disputed claims:\n{disputed_str}\n\n"
        f"Detailed verification results:\n{verification_detail}\n\n"
        "Write a comprehensive, professional Markdown research report covering "
        "all findings. Include verification status icons next to each fact."
    )

    # ------------------------------------------------------------------
    # Call the LLM
    # ------------------------------------------------------------------
    try:
        response = llm.invoke([
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        final_report = response.content.strip()
        console.print(
            f"[bold green]✓[/bold green] Report generated "
            f"([cyan]{len(final_report)}[/cyan] characters)"
        )

    except Exception as exc:
        error_msg = f"Synthesizer LLM call failed: {exc}"
        console.print(f"[bold red]✗ {error_msg}[/bold red]")
        final_report = (
            f"# Research Report — Error\n\n"
            f"Report generation failed: {exc}\n\n"
            f"## Raw Research Notes\n\n{research_notes}"
        )
        return {
            "final_report": final_report,
            "errors": [error_msg],
            "current_agent": "synthesizer",
            "messages": [HumanMessage(content=f"Synthesizer error: {error_msg}")],
        }

    return {
        "final_report": final_report,
        "current_agent": "synthesizer",
        "messages": [
            HumanMessage(content="Synthesizer completed. Final report is ready.")
        ],
    }
