"""Central configuration: environment, paths, and the LLM factory.

Supports two providers, both routed through CrewAI's LiteLLM layer:
  - Anthropic (Claude)   model strings like "anthropic/claude-sonnet-4-5"
  - Groq (open models)   model strings like "groq/llama-3.3-70b-versatile"

The provider is inferred from the model string's prefix, and the matching API
key env var is checked. Everything that touches the outside world is resolved
here so the rest of the code stays clean.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Where trained models, reports and charts are written.
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts")).resolve()
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# provider -> (env var holding the key, sensible default model)
PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic/claude-sonnet-4-5"),
    "openai": ("OPENAI_API_KEY", "openai/gpt-4o-mini"),
    "groq": ("GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
}

# Default model. Override with the MODEL env var or in the UI.
DEFAULT_MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-4-5")


def provider_for(model: str) -> str:
    """Infer the provider from a LiteLLM model string ('groq/...' -> 'groq')."""
    prefix = (model or "").split("/", 1)[0].lower()
    return prefix if prefix in PROVIDERS else "anthropic"


def key_env_for(model: str) -> str:
    return PROVIDERS[provider_for(model)][0]


def get_llm(model: str | None = None, temperature: float = 0.2):
    """Return a CrewAI LLM for the given model, validating the right API key.

    Raises a clear error if the matching key is missing, so the failure happens
    here (configuration) instead of deep inside a crew run.
    """
    from crewai import LLM

    model = model or DEFAULT_MODEL
    key_env = key_env_for(model)
    if not os.getenv(key_env):
        raise RuntimeError(
            f"{key_env} is not set for model '{model}'. Add it to your "
            "environment or .env file, or run in dry-run mode (deterministic "
            "analysis only, no agents)."
        )
    return LLM(model=model, temperature=temperature)


def has_api_key(model: str | None = None) -> bool:
    return bool(os.getenv(key_env_for(model or DEFAULT_MODEL)))
