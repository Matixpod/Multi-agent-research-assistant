# src/agents/translator.py
"""Translator Agent — translates the final report to Polish."""

from __future__ import annotations
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from src.config.settings import get_settings
from src.prompts.templates import TRANSLATOR_SYSTEM_PROMPT

console = Console()


def translator_node(state: dict[str, Any]) -> dict[str, Any]:
    """Translate the final report to Polish."""
    settings = get_settings()
    llm = settings.get_agent_llm()

    final_report: str = state.get("final_report", "")

    console.print("\n[bold green]🌍 Translator Agent[/bold green] starting...")

    # Obsługa pustego raportu
    if not final_report:
        console.print("[dim]   No report to translate — skipping.[/dim]")
        return {
            "translated_report": "",
            "current_agent": "translator",
            "messages": [HumanMessage(content="Translator: no report.")],
        }

    # Wywołanie LLM
    try:
        response = llm.invoke([
            SystemMessage(content=TRANSLATOR_SYSTEM_PROMPT),
            HumanMessage(content=f"Przetłumacz:\n\n{final_report}"),
        ])
        translated_report = response.content.strip()
        console.print(f"[bold green]✓[/bold green] Translated ({len(translated_report)} chars)")

    except Exception as exc:
        console.print(f"[bold red]✗ Translation failed: {exc}[/bold red]")
        return {
            "translated_report": "",
            "errors": [f"Translator error: {exc}"],
            "current_agent": "translator",
            "messages": [HumanMessage(content=f"Translator error: {exc}")],
        }

    return {
        "translated_report": translated_report,
        "current_agent": "translator",
        "messages": [HumanMessage(content="Translation completed.")],
    }