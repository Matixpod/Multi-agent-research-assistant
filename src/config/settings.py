# src/config/settings.py
"""
Application settings — loads configuration from .env using pydantic-settings.
"""

from functools import lru_cache

from langchain_ollama import ChatOllama
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Tavily Search (only external API cost)
    tavily_api_key: str = "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"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Models
    supervisor_model: str = "qwen2.5:7b-instruct"
    agent_model: str = "mistral:7b-instruct"

    # Research parameters
    max_research_loops: int = 6
    max_search_results: int = 10
    model_temperature: float = 0.1

    # LangSmith (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "multi-agent-research-assistant"

    # ------------------------------------------------------------------
    # LLM factory methods
    # ------------------------------------------------------------------

    def get_supervisor_llm(self) -> ChatOllama:
        """Return a ChatOllama instance configured for the supervisor agent."""
        return ChatOllama(
            model=self.supervisor_model,
            base_url=self.ollama_base_url,
            temperature=self.model_temperature,
        )

    def get_agent_llm(self) -> ChatOllama:
        """Return a ChatOllama instance configured for sub-agents (plain text output)."""
        return ChatOllama(
            model=self.agent_model,
            base_url=self.ollama_base_url,
            temperature=self.model_temperature,
            num_ctx=8192,
            num_predict=2048,
        )

    def get_agent_llm_json(self) -> ChatOllama:
        """Return a ChatOllama instance with JSON mode for structured output."""
        return ChatOllama(
            model=self.agent_model,
            base_url=self.ollama_base_url,
            temperature=self.model_temperature,
            num_ctx=8192,
            num_predict=2048,
            format="json",
        )


@lru_cache
def get_settings() -> Settings:
    """Cached singleton for application settings."""
    return Settings()
