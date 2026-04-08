# src/agents/researcher.py
"""
Research Agent — searches the web using Tavily and extracts key claims.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from src.config.settings import get_settings
from src.prompts.templates import RESEARCHER_SYSTEM_PROMPT
from src.tools.search import search_web

console = Console()


def researcher_node(state: dict[str, Any]) -> dict[str, Any]:
    """Execute the researcher step: search the web and extract key claims.

    Args:
        state: Current ``ResearchState`` dictionary.

    Returns:
        Partial state update with ``search_results``, ``research_notes``,
        ``key_claims``, ``current_agent``, and ``messages``.
    """
    settings = get_settings()
    llm = settings.get_agent_llm_json()

    query: str = state.get("query", "")
    existing_notes: str = state.get("research_notes", "")

    console.print("\n[bold blue]🔍 Researcher Agent[/bold blue] starting...")
    console.print(f"   Query: [italic]{query}[/italic]")

    # ------------------------------------------------------------------
    # 1. Search the web
    # ------------------------------------------------------------------
    try:
        search_results = search_web(query, max_results=settings.max_search_results)
    except Exception as exc:
        error_msg = f"Search failed: {exc}"
        console.print(f"[bold red]✗ {error_msg}[/bold red]")
        return {
            "errors": [error_msg],
            "current_agent": "researcher",
            "messages": [HumanMessage(content=f"Researcher error: {error_msg}")],
        }

    # ------------------------------------------------------------------
    # 2. Build context for the LLM (truncated to avoid exceeding context window)
    # ------------------------------------------------------------------
    MAX_CHARS_PER_RESULT = 1500
    MAX_TOTAL_CONTEXT = 6000

    truncated_results = []
    for r in search_results:
        content = r.get("content", "")[:MAX_CHARS_PER_RESULT]
        truncated_results.append(f"Source: {r.get('url', 'N/A')}\n{content}")

    search_context = "\n\n".join(truncated_results)[:MAX_TOTAL_CONTEXT]

    user_prompt = (
        f"Research query: {query}\n\n"
        f"Existing notes:\n{existing_notes[:2000]}\n\n"
        f"New search results:\n{search_context}\n\n"
        "Analyse the search results. Extract key claims and write structured "
        "research notes. You MUST respond with valid JSON and nothing else."
    )

    # ------------------------------------------------------------------
    # 3. Call the LLM
    # ------------------------------------------------------------------
    try:
        response = llm.invoke([
            SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        response_text = response.content.strip() if response.content else ""

        # Handle empty LLM response
        if not response_text:
            console.print("[yellow]⚠ LLM returned an empty response — using raw search data[/yellow]")
            # Build notes directly from search results as a fallback
            fallback_notes = "\n".join(
                f"- {r.get('content', '')[:300]}  (Source: {r.get('url', 'N/A')})"
                for r in search_results
            )
            parsed = {
                "research_notes": f"## Raw Search Results\n\n{fallback_notes}",
                "key_claims": [r.get("content", "")[:150] for r in search_results if r.get("content")],
            }
        else:
            # Try to parse JSON from the response
            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError:
                # Attempt to extract JSON from markdown code-blocks
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    parsed = json.loads(json_str)
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                    parsed = json.loads(json_str)
                else:
                    # Use the raw text as research notes
                    parsed = {
                        "research_notes": response_text,
                        "key_claims": [],
                    }

        research_notes = parsed.get("research_notes", response_text)
        key_claims = parsed.get("key_claims", [])

        console.print(f"[bold green]✓[/bold green] Extracted [cyan]{len(key_claims)}[/cyan] claims")

    except Exception as exc:
        error_msg = f"Researcher LLM call failed: {exc}"
        console.print(f"[bold red]✗ {error_msg}[/bold red]")
        return {
            "errors": [error_msg],
            "current_agent": "researcher",
            "messages": [HumanMessage(content=f"Researcher error: {error_msg}")],
        }

    return {
        "search_results": search_results,
        "research_notes": research_notes,
        "key_claims": key_claims,
        "current_agent": "researcher",
        "messages": [
            HumanMessage(
                content=f"Researcher completed. Found {len(search_results)} sources "
                        f"and extracted {len(key_claims)} key claims."
            )
        ],
    }
