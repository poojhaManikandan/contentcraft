"""Select the local or hosted AI provider from environment or Streamlit secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ai_engine.hosted_model import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    HostedModel,
    normalize_base_url,
)
from ai_engine.local_model import (
    DEFAULT_HOST,
    DEFAULT_MODEL as LOCAL_DEFAULT_MODEL,
    LocalModel,
    normalize_host,
)
from ai_engine.model_client import TextModel

VALID_MODES = ("hosted", "local", "auto")


def _read_setting(name: str, default: str = "") -> str:
    """Read a setting from the environment first, then Streamlit secrets."""
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()

    try:
        import streamlit as st

        secret = st.secrets.get(name)
        if secret is not None and str(secret).strip():
            return str(secret).strip()
    except Exception:
        pass
    return default


def _hosted_api_key() -> str:
    """Return the configured hosted API key without exposing it to callers."""
    return (
        _read_setting("AI_API_KEY")
        or _read_setting("GEMINI_API_KEY")
        or _read_setting("GOOGLE_API_KEY")
    )


def infer_provider_label(base_url: str) -> str:
    """Use a friendly provider name based on the API endpoint."""
    url = base_url.lower()
    if "groq.com" in url:
        return "Groq"
    if "openrouter.ai" in url:
        return "OpenRouter"
    if "generativelanguage.googleapis.com" in url or "googleapis.com" in url:
        return "Google Gemini"
    if "openai.com" in url:
        return "OpenAI"
    if "mistral.ai" in url:
        return "Mistral"
    if "together.xyz" in url:
        return "Together AI"
    return "Hosted AI API"


@dataclass(frozen=True)
class ProviderConfig:
    """Non-secret description of the configured model provider."""

    mode: str
    label: str
    model: str
    endpoint: str
    hosted: bool
    key_configured: bool


def get_provider_config() -> ProviderConfig:
    """Return the active provider configuration without exposing the API key."""
    mode = _read_setting("AI_MODE", "hosted").lower()
    if mode not in VALID_MODES:
        mode = "hosted"

    api_key = _hosted_api_key()
    hosted = mode == "hosted" or (mode == "auto" and bool(api_key))

    if hosted:
        endpoint = normalize_base_url(_read_setting("AI_BASE_URL", DEFAULT_BASE_URL))
        model = _read_setting("AI_MODEL", DEFAULT_MODEL)
        label = _read_setting("AI_PROVIDER", infer_provider_label(endpoint))
        return ProviderConfig(
            mode="hosted",
            label=label,
            model=model,
            endpoint=endpoint,
            hosted=True,
            key_configured=bool(api_key),
        )

    model = _read_setting("OLLAMA_MODEL", LOCAL_DEFAULT_MODEL)
    endpoint = normalize_host(_read_setting("OLLAMA_HOST", DEFAULT_HOST))
    return ProviderConfig(
        mode="local",
        label="Local Ollama",
        model=model,
        endpoint=endpoint,
        hosted=False,
        key_configured=True,
    )


def create_model(config: Optional[ProviderConfig] = None) -> TextModel:
    """Create the configured model client."""
    config = config or get_provider_config()
    if config.hosted:
        return HostedModel(
            api_key=_hosted_api_key(),
            base_url=config.endpoint,
            model=config.model,
            provider=config.label,
        )
    return LocalModel(model=config.model, host=config.endpoint)


def model_description(config: Optional[ProviderConfig] = None) -> str:
    """Return a short label for captions and result metadata."""
    config = config or get_provider_config()
    return f"{config.model} via {config.label}"
