# src/graph/workflow.py
"""
LangGraph workflow — builds and compiles the cyclic research graph.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from rich.console import Console

from src.agents.researcher import researcher_node
from src.agents.supervisor import supervisor_node
from src.agents.synthesizer import synthesizer_node
from src.agents.verifier import verifier_node
from src.agents.translator import translator_node
from src.graph.state import ResearchState

console = Console()


def route_from_supervisor(state: dict[str, Any]) -> str:
    """Read the supervisor's routing decision from the state.

    Args:
        state: Current ``ResearchState`` dictionary.

    Returns:
        The name of the next node (``"researcher"``, ``"verifier"``,
        ``"synthesizer"``, or ``"FINISH"``).
    """
    next_agent: str = state.get("next_agent", "FINISH")
    if next_agent not in {"researcher", "verifier", "synthesizer", "translator", "FINISH"}:
        console.print(f"[yellow]⚠ Unknown next_agent '{next_agent}' — finishing.[/yellow]")
        return "FINISH"
    return next_agent


def create_research_graph() -> Any:
    """Build and compile the multi-agent research StateGraph.

    Graph topology::

        START ──▶ supervisor ──┬──▶ researcher  ──▶ supervisor
                               ├──▶ verifier    ──▶ supervisor
                               ├──▶ synthesizer ──▶ supervisor
                               └──▶ FINISH (END)

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready to be invoked.
    """
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("translator", translator_node)
    # Entry edge
    workflow.add_edge(START, "supervisor")

    # Conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "verifier": "verifier",
            "synthesizer": "synthesizer",
            "translator": "translator",
            "FINISH": END,
        },
    )

    # After each agent, return to supervisor
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("verifier", "supervisor")
    workflow.add_edge("synthesizer", "supervisor")
    workflow.add_edge("translator", "supervisor")
    console.print("[bold green]✓[/bold green] Research graph compiled successfully.")
    return workflow.compile()
