# src/agents/supervisor.py
"""
Supervisor Agent — orchestrates the research pipeline by deciding which agent acts next.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from src.config.settings import get_settings
from src.prompts.templates import SUPERVISOR_SYSTEM_PROMPT

console = Console()

VALID_AGENTS = {"researcher", "verifier", "synthesizer", "translator", "FINISH"}


def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
    """Decide which agent should act next based on the current pipeline state.

    Uses the supervisor LLM (qwen2.5) to reason about the state and return
    a routing decision.

    Args:
        state: Current ``ResearchState`` dictionary.

    Returns:
        Partial state update with ``next_agent``, ``iteration``, and
        optionally ``is_complete``.
    """
    settings = get_settings()
    llm = settings.get_supervisor_llm()

    iteration: int = state.get("iteration", 0)
    max_iterations: int = state.get("max_iterations", settings.max_research_loops)

    console.print(
        f"\n[bold cyan]🧠 Supervisor[/bold cyan] — iteration "
        f"[cyan]{iteration + 1}[/cyan]/[cyan]{max_iterations}[/cyan]"
    )

    # ------------------------------------------------------------------
    # Enforce iteration cap
    # ------------------------------------------------------------------
    if iteration >= max_iterations:
        console.print("[bold yellow]⚠ Max iterations reached — finishing.[/bold yellow]")
        return {
            "next_agent": "FINISH",
            "iteration": iteration,
            "is_complete": True,
            "current_agent": "supervisor",
            "messages": [
                HumanMessage(content="Supervisor: max iterations reached. Finishing.")
            ],
        }

    # ------------------------------------------------------------------
    # Build a state summary for the LLM
    # ------------------------------------------------------------------
    has_research = bool(state.get("research_notes"))
    has_claims = bool(state.get("key_claims"))
    has_verification = bool(state.get("verification_results"))
    has_report = bool(state.get("final_report"))
    has_translation = bool(state.get("translated_report"))
    
    state_summary = (
        f"Iteration: {iteration + 1}/{max_iterations}\n"
        f"Research done: {'yes' if has_research else 'no'}\n"
        f"Key claims extracted: {'yes' if has_claims else 'no'} "
        f"({len(state.get('key_claims', []))} claims)\n"
        f"Verification done: {'yes' if has_verification else 'no'} "
        f"(confirmed={len(state.get('verified_claims', []))}, "
        f"disputed={len(state.get('disputed_claims', []))})\n"
        f"Report generated: {'yes' if has_report else 'no'}\n"
        f"Translation done: {'yes' if has_translation else 'no'}\n" 
        f"Errors: {state.get('errors', [])}\n"
    )

    user_prompt = (
        f"User query: {state.get('query', '')}\n\n"
        f"Current state:\n{state_summary}\n"
        "Decide which agent should work next. Respond with JSON only."
    )

    # ------------------------------------------------------------------
    # Call the supervisor LLM
    # ------------------------------------------------------------------
    try:
        response = llm.invoke([
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        response_text = response.content.strip()

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from code-blocks
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
            else:
                # Fallback heuristic
                parsed = _fallback_routing(state)

        next_agent = parsed.get("next_agent", "FINISH")
        reasoning = parsed.get("reasoning", "")

        # Validate
        if next_agent not in VALID_AGENTS:
            console.print(
                f"[yellow]⚠ Invalid agent '{next_agent}' — falling back to heuristic[/yellow]"
            )
            fallback = _fallback_routing(state)
            next_agent = fallback["next_agent"]
            reasoning = fallback["reasoning"]

    except Exception as exc:
        error_msg = f"Supervisor LLM call failed: {exc}"
        console.print(f"[bold red]✗ {error_msg}[/bold red]")
        fallback = _fallback_routing(state)
        next_agent = fallback["next_agent"]
        reasoning = f"LLM error fallback — {fallback['reasoning']}"

    is_complete = next_agent == "FINISH"

    console.print(f"   Decision: [bold]{next_agent}[/bold]")
    console.print(f"   Reasoning: [dim]{reasoning}[/dim]")

    return {
        "next_agent": next_agent,
        "iteration": iteration + 1,
        "is_complete": is_complete,
        "current_agent": "supervisor",
        "messages": [
            HumanMessage(
                content=f"Supervisor decided: {next_agent}. Reason: {reasoning}"
            )
        ],
    }


def _fallback_routing(state: dict[str, Any]) -> dict[str, str]:
    """Deterministic fallback routing when the LLM fails to produce valid JSON.

    Follows the canonical flow: researcher → verifier → synthesizer → FINISH.
    """
    if not state.get("research_notes"):
        return {"next_agent": "researcher", "reasoning": "No research yet — starting research."}
    if not state.get("verification_results"):
        return {"next_agent": "verifier", "reasoning": "Research done, need verification."}
    if not state.get("final_report"):
        return {"next_agent": "synthesizer", "reasoning": "Verification done, need report."}
    if not state.get("translated_report"):
        return {"next_agent": "translator", "reasoning": "Need translation."}
    return {"next_agent": "FINISH", "reasoning": "All steps completed."}
