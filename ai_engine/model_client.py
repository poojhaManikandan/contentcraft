"""Common interface implemented by local and hosted model clients."""

from __future__ import annotations

from typing import Optional, Protocol

from ai_engine.local_model import ModelStatus


class TextModel(Protocol):
    """Minimal model interface used by the content modules."""

    model: str
    host: str

    def check(self) -> ModelStatus:
        """Return the current availability status."""

    def is_ready(self) -> bool:
        """Return True when the model can accept a request."""

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        seed: Optional[int] = None,
    ) -> str:
        """Generate text for the supplied prompt."""
