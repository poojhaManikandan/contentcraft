"""Reusable client for an Ollama server used in local development.

Production deployments use the hosted provider in ``hosted_model.py``. This
client remains available when ``AI_MODE=local`` is selected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import requests

from ai_engine.errors import ModelError

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

LocalModelError = ModelError


@dataclass
class ModelStatus:
    """Result of checking the local Ollama installation."""

    connected: bool
    model_available: bool
    host: str
    model: str
    available_models: list[str] = field(default_factory=list)
    message: str = ""


def normalize_host(host: str) -> str:
    """Add a scheme when the user supplies a bare host:port value."""
    host = (host or DEFAULT_HOST).strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


def _same_model(name: str, target: str) -> bool:
    """Compare model names while treating ':latest' as optional."""
    clean = lambda value: value.strip().lower().removesuffix(":latest")
    return clean(name) == clean(target)


def setup_steps(model: str = DEFAULT_MODEL) -> list[str]:
    """Short, honest setup instructions shown in the UI and error messages."""
    return [
        "Install Ollama from https://ollama.com/download and start it.",
        f"Download the model once: ollama pull {model}",
        "Keep Ollama running (ollama serve) while you use ContentCraft.",
        "Click Refresh status in the sidebar after setup.",
    ]


class LocalModel:
    """Small wrapper around the Ollama HTTP API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: int = 300,
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> None:
        self.model = model.strip() or DEFAULT_MODEL
        self.host = normalize_host(host)
        self.timeout = timeout
        self.num_ctx = num_ctx

    # ------------------------------------------------------------------
    # Connection and model checks
    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """Return the names of models installed in Ollama."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=(3, 5))
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError as exc:
            raise LocalModelError(
                f"Ollama is not reachable at {self.host}. Start Ollama and try again."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LocalModelError(
                f"Ollama at {self.host} did not respond in time."
            ) from exc
        except (requests.RequestException, ValueError) as exc:
            raise LocalModelError(
                f"Could not read the model list from Ollama at {self.host}."
            ) from exc

        models = data.get("models", [])
        names = []
        for item in models:
            name = item.get("name") or item.get("model")
            if name:
                names.append(name)
        return sorted(set(names))

    def check(self) -> ModelStatus:
        """Check connectivity and whether the configured model is installed."""
        try:
            names = self.list_models()
        except LocalModelError as exc:
            return ModelStatus(
                connected=False,
                model_available=False,
                host=self.host,
                model=self.model,
                message=str(exc),
            )

        available = any(_same_model(name, self.model) for name in names)
        if available:
            message = f"Ollama is running and {self.model} is ready."
        else:
            message = (
                f"Ollama is running, but {self.model} is not installed. "
                f"Run: ollama pull {self.model}"
            )
        return ModelStatus(
            connected=True,
            model_available=available,
            host=self.host,
            model=self.model,
            available_models=names,
            message=message,
        )

    def is_ready(self) -> bool:
        """Return True only when Ollama and the configured model are usable."""
        status = self.check()
        return status.connected and status.model_available

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        seed: Optional[int] = None,
    ) -> str:
        """Generate text with the local model and return the cleaned response."""
        if not prompt or not prompt.strip():
            raise LocalModelError("The prompt is empty. Please provide some input text.")

        status = self.check()
        if not status.connected:
            raise LocalModelError(status.message)
        if not status.model_available:
            raise LocalModelError(status.message)

        options = {
            "temperature": max(0.0, min(float(temperature), 2.0)),
            "top_p": max(0.0, min(float(top_p), 1.0)),
            "num_predict": max(32, int(max_tokens)),
            "num_ctx": self.num_ctx,
        }
        if seed is not None:
            options["seed"] = int(seed)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": options,
        }
        if system:
            payload["system"] = system

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=(5, self.timeout),
            )
        except requests.exceptions.ConnectionError as exc:
            raise LocalModelError(
                f"Lost connection to Ollama at {self.host}. Is it still running?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LocalModelError(
                f"Generation timed out after {self.timeout} seconds. "
                "Try a shorter input or a smaller target length."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LocalModelError(f"Could not contact Ollama: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LocalModelError(
                f"Ollama returned an invalid response (HTTP {response.status_code})."
            ) from exc

        if response.status_code == 404:
            raise LocalModelError(
                f"Model '{self.model}' was not found. Run: ollama pull {self.model}"
            )
        if response.status_code >= 400 or "error" in data:
            detail = data.get("error") or response.text[:300]
            raise LocalModelError(f"Ollama error: {detail}")

        text = (data.get("response") or "").strip()
        if not text:
            raise LocalModelError(
                "The model returned an empty response. Try again with a shorter prompt."
            )
        return text


def get_default_model() -> str:
    """Return the model configured for this installation."""
    return DEFAULT_MODEL


def get_default_host() -> str:
    """Return the Ollama host configured for this installation."""
    return normalize_host(DEFAULT_HOST)
