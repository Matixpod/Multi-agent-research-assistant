# tests/test_graph.py
"""
Tests for the LangGraph workflow and state definitions.
"""

import pytest

from src.graph.state import ResearchState, create_initial_state
from src.graph.workflow import create_research_graph, route_from_supervisor


class TestResearchState:
    """Tests for the ResearchState creation."""

    def test_create_initial_state_defaults(self) -> None:
        """create_initial_state should populate all required fields."""
        state = create_initial_state("test query")

        assert state["query"] == "test query"
        assert state["messages"] == []
        assert state["search_results"] == []
        assert state["research_notes"] == ""
        assert state["key_claims"] == []
        assert state["verification_results"] == []
        assert state["verified_claims"] == []
        assert state["disputed_claims"] == []
        assert state["final_report"] == ""
        assert state["current_agent"] == "supervisor"
        assert state["next_agent"] == ""
        assert state["iteration"] == 0
        assert state["max_iterations"] == 3
        assert state["is_complete"] is False
        assert state["errors"] == []

    def test_create_initial_state_custom_max_iterations(self) -> None:
        """create_initial_state should accept a custom max_iterations value."""
        state = create_initial_state("q", max_iterations=10)
        assert state["max_iterations"] == 10


class TestRouteFromSupervisor:
    """Tests for the conditional routing function."""

    def test_routes_to_researcher(self) -> None:
        """Should return 'researcher' when next_agent is set."""
        state = {"next_agent": "researcher"}
        assert route_from_supervisor(state) == "researcher"

    def test_routes_to_verifier(self) -> None:
        state = {"next_agent": "verifier"}
        assert route_from_supervisor(state) == "verifier"

    def test_routes_to_synthesizer(self) -> None:
        state = {"next_agent": "synthesizer"}
        assert route_from_supervisor(state) == "synthesizer"

    def test_routes_to_finish(self) -> None:
        state = {"next_agent": "FINISH"}
        assert route_from_supervisor(state) == "FINISH"

    def test_unknown_agent_routes_to_finish(self) -> None:
        """Unknown agent names should default to FINISH."""
        state = {"next_agent": "unknown_agent"}
        assert route_from_supervisor(state) == "FINISH"

    def test_missing_next_agent_routes_to_finish(self) -> None:
        """Missing next_agent key should default to FINISH."""
        state = {}
        assert route_from_supervisor(state) == "FINISH"


class TestGraphCompilation:
    """Tests for graph compilation."""

    def test_graph_compiles(self) -> None:
        """The research graph should compile without errors."""
        graph = create_research_graph()
        assert graph is not None

    def test_graph_has_invoke(self) -> None:
        """The compiled graph should expose an invoke method."""
        graph = create_research_graph()
        assert hasattr(graph, "invoke")
