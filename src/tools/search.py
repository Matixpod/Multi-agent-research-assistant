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
        results = tool.invoke({"query": query})

        # TavilySearchResults may return a list of dicts or a JSON string
        if isinstance(results, str):
            import json
            results = json.loads(results)

        console.print(
            f"[bold green]✓[/bold green] Search returned "
            f"[cyan]{len(results)}[/cyan] results for: [italic]{query}[/italic]"
        )
        return results

    except Exception as exc:
        console.print(f"[bold red]✗ Search failed:[/bold red] {exc}")
        return []
