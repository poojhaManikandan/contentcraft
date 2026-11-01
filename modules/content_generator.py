"""Content generation module powered by the local Ollama model."""

from __future__ import annotations

from typing import Optional

from ai_engine import prompts
from ai_engine.model_client import TextModel
from ai_engine.provider import create_model
from utils.text_cleaner import strip_code_fences

CONTENT_TYPES = (
    "Blog Post",
    "Article",
    "Essay",
    "Email",
    "Report",
    "Social Media Post",
    "Study Notes",
)

TONES = ("Professional", "Academic", "Formal", "Casual", "Creative")

MIN_WORDS = 100
MAX_WORDS = 1000


def validate_request(
    topic: str,
    content_type: str,
    tone: str,
    word_count: int,
) -> str:
    """Validate user input and return the cleaned topic."""
    cleaned_topic = (topic or "").strip()
    if len(cleaned_topic) < 3:
        raise ValueError("Please enter a topic with at least 3 characters.")
    if len(cleaned_topic) > 300:
        raise ValueError("The topic is too long. Please keep it under 300 characters.")
    if content_type not in CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {content_type}")
    if tone not in TONES:
        raise ValueError(f"Unsupported writing tone: {tone}")
    if not isinstance(word_count, int) or not (MIN_WORDS <= word_count <= MAX_WORDS):
        raise ValueError(f"Word count must be between {MIN_WORDS} and {MAX_WORDS}.")
    return cleaned_topic


def generate_content(
    topic: str,
    content_type: str,
    tone: str,
    word_count: int,
    extra_instructions: str = "",
    variation: int = 0,
    model: Optional[TextModel] = None,
) -> str:
    """Generate a piece of content with the configured model.

    ``variation`` is increased by the Regenerate button to ask for a fresh
    version instead of returning the same draft again.
    """
    cleaned_topic = validate_request(topic, content_type, tone, word_count)
    model = model or create_model()

    prompt = prompts.content_generation_prompt(
        cleaned_topic, content_type, tone, word_count, extra_instructions
    )
    if variation > 0:
        prompt += (
            "\n- This is a regeneration request. Use a different opening, structure "
            "and examples from the previous draft while keeping the same topic.\n"
        )

    temperature = min(0.4 + (0.04 * variation), 0.6)
    max_tokens = min(4096, (word_count * 2) + 350)

    content = model.generate(
        prompt,
        system=prompts.SYSTEM_WRITER,
        temperature=temperature,
        top_p=0.92,
        max_tokens=max_tokens,
    )
    return strip_code_fences(content)
