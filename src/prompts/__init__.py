# src/prompts/__init__.py
"""Prompt templates for all agents."""

from src.prompts.templates import (
    RESEARCHER_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
)

__all__ = [
    "SUPERVISOR_SYSTEM_PROMPT",
    "RESEARCHER_SYSTEM_PROMPT",
    "VERIFIER_SYSTEM_PROMPT",
    "SYNTHESIZER_SYSTEM_PROMPT",
]
