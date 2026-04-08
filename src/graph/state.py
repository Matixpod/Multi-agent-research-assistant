# src/graph/state.py
"""
Graph state definition — typed dictionary shared across all agent nodes.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """Shared state flowing through the LangGraph research pipeline.

    Every agent node reads from and writes to this state. The supervisor
    uses it to decide which agent should act next.
    """

    # User input
    query: str

    # Communication history
    messages: Annotated[list[BaseMessage], add_messages]

    # Research data
    search_results: list[dict]
    research_notes: str
    key_claims: list[str]

    # Verification data
    verification_results: list[dict]  # [{claim, verdict, evidence}]
    verified_claims: list[str]
    disputed_claims: list[str]

    # Final output
    final_report: str

    # Orchestration
    current_agent: str
    next_agent: str
    iteration: int
    max_iterations: int
    is_complete: bool

    # Error tracking
    errors: list[str]


def create_initial_state(query: str, max_iterations: int = 3) -> ResearchState:
    """Create a blank initial state for a new research query.

    Args:
        query: The user's research question.
        max_iterations: Maximum supervisor loop iterations.

    Returns:
        A fully initialised ``ResearchState``.
    """
    return ResearchState(
        query=query,
        messages=[],
        search_results=[],
        research_notes="",
        key_claims=[],
        verification_results=[],
        verified_claims=[],
        disputed_claims=[],
        final_report="",
        current_agent="supervisor",
        next_agent="",
        iteration=0,
        max_iterations=max_iterations,
        is_complete=False,
        errors=[],
    )
