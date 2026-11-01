"""Client for an OpenAI-compatible hosted AI API."""

from __future__ import annotations

from typing import Optional

import requests

from ai_engine.errors import ModelError
from ai_engine.local_model import ModelStatus

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_PROVIDER = "Google Gemini"


def normalize_base_url(url: str) -> str:
    """Return a clean API base URL without a trailing slash."""
    value = (url or DEFAULT_BASE_URL).strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = "https://" + value
    return value


def _response_detail(response: requests.Response) -> str:
    """Extract a short, safe error message without exposing credentials."""
    try:
        data = response.json()
    except ValueError:
        return (response.text or f"HTTP {response.status_code}")[:300]

    error = data.get("error")
    if isinstance(error, dict):
        detail = error.get("message") or error.get("code") or str(error)
    elif error:
        detail = str(error)
    else:
        detail = data.get("message") or str(data)
    return str(detail)[:300]


def _message_text(content: object) -> str:
    """Read the text from either a string or a list of content parts."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return ""


class HostedModel:
    """Small wrapper around the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        provider: str = DEFAULT_PROVIDER,
        timeout: int = 120,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = normalize_base_url(base_url)
        self.model = (model or DEFAULT_MODEL).strip()
        self.provider = (provider or DEFAULT_PROVIDER).strip()
        self.host = self.base_url
        self.timeout = timeout

    def check(self) -> ModelStatus:
        """Report whether the hosted provider is configured."""
        if not self.api_key:
            return ModelStatus(
                connected=False,
                model_available=False,
                host=self.base_url,
                model=self.model,
                message=(
                    "GEMINI_API_KEY or AI_API_KEY is not configured. Add the hosted "
                    "API key to Streamlit secrets or the environment, then refresh "
                    "the status."
                ),
            )
        return ModelStatus(
            connected=True,
            model_available=True,
            host=self.base_url,
            model=self.model,
            available_models=[self.model],
            message=(
                f"{self.provider} is configured with {self.model}. "
                "The key is checked when the first request is sent."
            ),
        )

    def is_ready(self) -> bool:
        """Return True when an API key is configured."""
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        seed: Optional[int] = None,
    ) -> str:
        """Generate text with the hosted chat completions endpoint."""
        if not prompt or not prompt.strip():
            raise ModelError("The prompt is empty. Please provide some input text.")
        if not self.api_key:
            raise ModelError(
                "The hosted AI API key is missing. Set GEMINI_API_KEY or "
                "AI_API_KEY in Streamlit secrets or the environment."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": max(0.0, min(float(temperature), 2.0)),
            "top_p": max(0.0, min(float(top_p), 1.0)),
            "max_tokens": max(32, int(max_tokens)),
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = int(seed)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "ContentCraft",
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=(5, self.timeout),
            )
        except requests.exceptions.Timeout as exc:
            raise ModelError(
                f"The hosted AI request timed out after {self.timeout} seconds. "
                "Try a shorter input or a shorter target length."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ModelError(f"Could not contact the hosted AI provider: {exc}") from exc

        if response.status_code == 401 or response.status_code == 403:
            raise ModelError(
                "The hosted AI provider rejected the API key. Check AI_API_KEY "
                "and restart the app."
            )
        if response.status_code == 404:
            raise ModelError(
                f"Model '{self.model}' was not found at {self.base_url}. "
                "Check AI_MODEL and AI_BASE_URL."
            )
        if response.status_code == 429:
            raise ModelError(
                "The hosted AI provider rate limit was reached. Wait a moment and try again."
            )
        if response.status_code >= 400:
            raise ModelError(
                f"{self.provider} returned an error: {_response_detail(response)}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ModelError(
                f"{self.provider} returned an invalid response "
                f"(HTTP {response.status_code})."
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise ModelError(
                f"{self.provider} returned no choices. Check the model name and try again."
            )

        message = choices[0].get("message") or {}
        text = _message_text(message.get("content"))
        if not text:
            raise ModelError(
                "The hosted model returned an empty response. Try again with a shorter prompt."
            )
        return text
