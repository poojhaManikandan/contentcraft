"""Shared exceptions for local and hosted model providers."""

from __future__ import annotations


class ModelError(RuntimeError):
    """Raised when an AI model is unavailable or a generation request fails."""


LocalModelError = ModelError
