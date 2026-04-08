# tests/test_agents.py
"""
Smoke tests for agent node functions.

These tests require Ollama running with the configured models.
They are skipped by default in CI environments.
"""

import os

import pytest

from src.graph.state import create_initial_state

# Skip all tests in this module if Ollama is not reachable
pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_OLLAMA_TESTS", "true").lower() == "true",
    reason="Ollama tests are skipped by default. Set SKIP_OLLAMA_TESTS=false to run.",
)


class TestSupervisorNode:
    """Smoke tests for the supervisor agent."""

    def test_supervisor_returns_next_agent(self) -> None:
        """Supervisor should return a valid next_agent decision."""
        from src.agents.supervisor import supervisor_node

        state = create_initial_state("What is quantum computing?")
        result = supervisor_node(state)

        assert "next_agent" in result
        assert result["next_agent"] in {"researcher", "verifier", "synthesizer", "FINISH"}


class TestResearcherNode:
    """Smoke tests for the researcher agent."""

    @pytest.mark.skipif(
        os.getenv("TAVILY_API_KEY", "").startswith("tvly-your"),
        reason="TAVILY_API_KEY not configured.",
    )
    def test_researcher_returns_notes(self) -> None:
        """Researcher should return research_notes and key_claims."""
        from src.agents.researcher import researcher_node

        state = create_initial_state("What is Python programming language?")
        result = researcher_node(state)

        assert "research_notes" in result or "errors" in result


class TestVerifierNode:
    """Smoke tests for the verifier agent."""

    def test_verifier_handles_no_claims(self) -> None:
        """Verifier should handle an empty claims list gracefully."""
        from src.agents.verifier import verifier_node

        state = create_initial_state("test query")
        result = verifier_node(state)

        assert result["verified_claims"] == []
        assert result["disputed_claims"] == []


class TestSynthesizerNode:
    """Smoke tests for the synthesizer agent."""

    def test_synthesizer_returns_report(self) -> None:
        """Synthesizer should return a final_report string."""
        from src.agents.synthesizer import synthesizer_node

        state = create_initial_state("test query")
        state["research_notes"] = "Python is a programming language."
        state["verified_claims"] = ["Python was created by Guido van Rossum"]
        state["disputed_claims"] = []
        state["verification_results"] = [
            {
                "claim": "Python was created by Guido van Rossum",
                "verdict": "confirmed",
                "evidence": "Multiple sources confirm this.",
                "confidence": "high",
            }
        ]
        result = synthesizer_node(state)

        assert "final_report" in result
        assert len(result["final_report"]) > 0
