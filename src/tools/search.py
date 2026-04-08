# src/tools/search.py
"""
Tavily web-search wrapper — provides both a LangChain tool and a standalone function.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults
from rich.console import Console

from src.config.settings import get_settings

console = Console()

# Ensure the env var is set so any code that reads os.environ directly finds it
_settings = get_settings()
if _settings.tavily_api_key and not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = _settings.tavily_api_key


def create_search_tool(max_results: int = 5) -> TavilySearchResults:
    """Create a Tavily search tool for use inside LangChain agents.

    Args:
        max_results: Maximum number of search results to return.

    Returns:
        A configured ``TavilySearchResults`` tool instance.
    """
    return TavilySearchResults(max_results=max_results)


def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Perform a standalone web search using Tavily.

    This function can be called outside of an agent chain — useful for
    direct search within agent node functions.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        A list of result dicts, each containing ``url`` and ``content`` keys.
        Returns an empty list on failure.
    """
    try:
        tool = create_search_tool(max_results=max_results)
        raw_results = tool.invoke({"query": query})

        # Handle different Tavily response formats
        if isinstance(raw_results, str):
            import json
            raw_results = json.loads(raw_results)

        # New Tavily format may wrap results in a dict
        if isinstance(raw_results, dict) and "results" in raw_results:
            results = raw_results["results"]
        elif isinstance(raw_results, list):
            results = raw_results
        else:
            results = []

        # Filter only valid dicts with 'content' key
        results = [r for r in results if isinstance(r, dict) and "content" in r]

        console.print(
            f"[bold green]✓[/bold green] Search returned "
            f"[cyan]{len(results)}[/cyan] results for: [italic]{query}[/italic]"
        )
        return results

    except Exception as exc:
        console.print(f"[bold red]✗ Search failed:[/bold red] {exc}")
        return []
