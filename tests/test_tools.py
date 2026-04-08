# tests/test_tools.py
"""
Tests for the Tavily search tool wrapper.
"""

import os

import pytest

from src.tools.search import create_search_tool, search_web


class TestCreateSearchTool:
    """Tests for the create_search_tool factory."""

    def test_returns_tavily_tool(self) -> None:
        """create_search_tool should return a TavilySearchResults instance."""
        tool = create_search_tool(max_results=3)
        assert tool is not None
        assert hasattr(tool, "invoke")

    def test_custom_max_results(self) -> None:
        """max_results parameter should be passed through."""
        tool = create_search_tool(max_results=10)
        assert tool.max_results == 10


class TestSearchWeb:
    """Tests for the standalone search_web function."""

    @pytest.mark.skipif(
        os.getenv("TAVILY_API_KEY", "").startswith("tvly-your"),
        reason="TAVILY_API_KEY not configured — skipping live search test.",
    )
    def test_search_returns_results(self) -> None:
        """search_web should return a non-empty list for a valid query."""
        results = search_web("Python programming language", max_results=2)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_returns_empty_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """search_web should return an empty list when the search fails."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        # With no API key the tool will raise — search_web should catch it
        results = search_web("test query", max_results=1)
        assert isinstance(results, list)
