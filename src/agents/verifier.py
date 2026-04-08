# src/agents/verifier.py
"""
Verifier Agent — cross-references claims against multiple sources for fact-checking.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from src.config.settings import get_settings
from src.prompts.templates import VERIFIER_SYSTEM_PROMPT
from src.tools.search import search_web

console = Console()


def verifier_node(state: dict[str, Any]) -> dict[str, Any]:
    """Execute the verifier step: fact-check each claim against sources.

    Args:
        state: Current ``ResearchState`` dictionary.

    Returns:
        Partial state update with ``verification_results``,
        ``verified_claims``, ``disputed_claims``, and ``messages``.
    """
    settings = get_settings()
    llm = settings.get_agent_llm_json()

    key_claims: list[str] = state.get("key_claims", [])
    search_results: list[dict] = state.get("search_results", [])

    console.print("\n[bold yellow]🔎 Verifier Agent[/bold yellow] starting...")
    console.print(f"   Claims to verify: [cyan]{len(key_claims)}[/cyan]")

    if not key_claims:
        console.print("[dim]   No claims to verify — skipping.[/dim]")
        return {
            "verification_results": [],
            "verified_claims": [],
            "disputed_claims": [],
            "current_agent": "verifier",
            "messages": [HumanMessage(content="Verifier: no claims to verify.")],
        }

    # ------------------------------------------------------------------
    # 1. Optional: additional search for cross-referencing
    # ------------------------------------------------------------------
    extra_context = ""
    try:
        verification_query = f"fact check: {' '.join(key_claims[:3])}"
        extra_results = search_web(verification_query, max_results=3)
        extra_context = "\n\n".join(
            f"Source: {r.get('url', 'N/A')}\n{r.get('content', '')}"
            for r in extra_results
        )
    except Exception as exc:
        console.print(f"[dim]   Additional search failed (non-fatal): {exc}[/dim]")

    # ------------------------------------------------------------------
    # 2. Build context for the LLM
    # ------------------------------------------------------------------
    original_context = "\n\n".join(
        f"Source: {r.get('url', 'N/A')}\n{r.get('content', '')}"
        for r in search_results
    )

    claims_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(key_claims))

    user_prompt = (
        f"Claims to verify:\n{claims_list}\n\n"
        f"Original sources:\n{original_context}\n\n"
        f"Additional cross-reference sources:\n{extra_context}\n\n"
        "For each claim, determine if it is confirmed, disputed, or unverified. "
        "Provide evidence and confidence level. Respond with valid JSON."
    )

    # ------------------------------------------------------------------
    # 3. Call the LLM
    # ------------------------------------------------------------------
    try:
        response = llm.invoke([
            SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        response_text = response.content.strip()

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
            else:
                # Fallback: treat all claims as unverified
                parsed = {
                    "verification_results": [
                        {"claim": c, "verdict": "unverified", "evidence": "N/A", "confidence": "low"}
                        for c in key_claims
                    ],
                    "verified_claims": [],
                    "disputed_claims": [],
                }

        verification_results = parsed.get("verification_results", [])
        verified_claims = parsed.get("verified_claims", [])
        disputed_claims = parsed.get("disputed_claims", [])

        console.print(
            f"[bold green]✓[/bold green] Verified: [green]{len(verified_claims)}[/green] | "
            f"Disputed: [red]{len(disputed_claims)}[/red] | "
            f"Total checked: [cyan]{len(verification_results)}[/cyan]"
        )

    except Exception as exc:
        error_msg = f"Verifier LLM call failed: {exc}"
        console.print(f"[bold red]✗ {error_msg}[/bold red]")
        return {
            "errors": [error_msg],
            "current_agent": "verifier",
            "messages": [HumanMessage(content=f"Verifier error: {error_msg}")],
        }

    return {
        "verification_results": verification_results,
        "verified_claims": verified_claims,
        "disputed_claims": disputed_claims,
        "current_agent": "verifier",
        "messages": [
            HumanMessage(
                content=f"Verifier completed. {len(verified_claims)} confirmed, "
                        f"{len(disputed_claims)} disputed out of {len(key_claims)} claims."
            )
        ],
    }
