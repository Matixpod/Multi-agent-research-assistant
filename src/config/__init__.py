# src/config/__init__.py
"""Application configuration."""

from src.config.settings import Settings, get_settings

__all__ = [
    "Settings",
    "get_settings",
]
