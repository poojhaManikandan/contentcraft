"""Rewrite text in a selected style while preserving its meaning."""

from __future__ import annotations

from typing import Optional

from ai_engine import prompts
from ai_engine.model_client import TextModel
from ai_engine.provider import create_model
from utils.text_cleaner import clean_text, strip_code_fences, strip_leading_label, word_count

REWRITE_STYLES = ("Professional", "Academic", "Simple English", "Formal", "Concise")

MAX_REWRITE_CHARS = 8000


def rewrite_text(
    text: str,
    style: str = "Professional",
    model: Optional[TextModel] = None,
) -> str:
    """Rewrite the supplied text using the selected style."""
    cleaned = clean_text(text)
    if word_count(cleaned) < 5:
        raise ValueError("Please provide at least 5 words to rewrite.")
    if len(cleaned) > MAX_REWRITE_CHARS:
        raise ValueError(
            f"The text is too long for rewriting ({len(cleaned)} characters). "
            f"Please keep it under {MAX_REWRITE_CHARS} characters."
        )
    if style not in REWRITE_STYLES:
        raise ValueError(f"Unsupported rewrite style: {style}")

    model = model or create_model()
    prompt = prompts.rewrite_prompt(cleaned, style)
    max_tokens = min(4096, (word_count(cleaned) * 2) + 250)

    result = model.generate(
        prompt,
        system=prompts.SYSTEM_WRITER,
        temperature=0.45,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    return strip_leading_label(strip_code_fences(result), ["Rewritten text:", "Rewritten version:"])
