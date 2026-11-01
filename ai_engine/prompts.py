"""Prompt templates used by the local language model.

Keeping prompts in one place makes the AI behaviour easy to review and tune.
All prompts ask the model to return only the final content so that the UI can
display the response directly.
"""

from __future__ import annotations

SYSTEM_WRITER = (
    "You are ContentCraft, a careful and helpful writing assistant. "
    "Follow the user's instructions exactly. Return only the requested content, "
    "without prefaces, explanations, or notes about being an AI."
)

CONTENT_TYPE_STRUCTURES = {
    "Blog Post": (
        "a catchy title, a short engaging introduction, 3-5 body sections with "
        "subheadings, and a concluding paragraph"
    ),
    "Article": (
        "a title, an introduction, 3-4 informative sections, and a conclusion"
    ),
    "Essay": (
        "a title, an introduction containing a clear thesis, 3-4 body paragraphs, "
        "and a conclusion"
    ),
    "Email": (
        "a subject line, a greeting, concise body paragraphs, and a professional closing"
    ),
    "Report": (
        "a title, an objective, key findings or discussion, and a conclusion with recommendations"
    ),
    "Social Media Post": (
        "a strong opening hook, 2-3 short paragraphs, a closing call to action, "
        "and 3-5 relevant hashtags"
    ),
    "Study Notes": (
        "a title, a short overview, organised bullet points under clear headings, "
        "and key takeaways"
    ),
}

# Percentage of the original text each summary length should keep, with
# sensible word bounds so very short or very long inputs still behave well.
LENGTH_TARGET_PERCENT = {"Short": 0.15, "Medium": 0.30, "Detailed": 0.45}
LENGTH_TARGET_BOUNDS = {"Short": (35, 140), "Medium": (50, 320), "Detailed": (80, 700)}


def summary_target_words(original_words: int, length: str) -> int:
    """Return an explicit target word count for a summary."""
    percent = LENGTH_TARGET_PERCENT.get(length, LENGTH_TARGET_PERCENT["Medium"])
    low, high = LENGTH_TARGET_BOUNDS.get(length, LENGTH_TARGET_BOUNDS["Medium"])
    target = max(low, min(high, int(round(max(original_words, 1) * percent))))
    # Never ask for a summary longer than the original text.
    return max(20, min(target, max(original_words, 20)))


def content_generation_prompt(
    topic: str,
    content_type: str,
    tone: str,
    word_count: int,
    extra_instructions: str = "",
) -> str:
    """Build the prompt for the content generator."""
    structure = CONTENT_TYPE_STRUCTURES.get(content_type, "a clear title and well-organised sections")
    low, high = int(word_count * 0.9), int(word_count * 1.1)
    intro_words = max(20, round(word_count * 0.12))
    conclusion_words = max(20, round(word_count * 0.15))
    section_count = 2 if word_count < 220 else 3 if word_count < 450 else 4
    body_words = max(40, round((word_count - intro_words - conclusion_words) / section_count))

    instructions = (
        f'Write a {content_type.lower()} about: "{topic.strip()}"\n\n'
        "Requirements:\n"
        f"- Tone: {tone}\n"
        f"- Target length: about {word_count} words (acceptable range {low}-{high} words).\n"
        f"- Length plan: about {intro_words} words for the opening, {section_count} main "
        f"section(s) of about {body_words} words each, and about {conclusion_words} words "
        "for the conclusion or closing.\n"
        f"- Structure: {structure}\n"
        "- Follow the length plan and do not add extra sections.\n"
        "- Use clear headings where they improve readability.\n"
        "- Produce original, accurate and well-organised content.\n"
        "- Do not invent statistics, citations, links or sources.\n"
        "- Start directly with the content. Do not describe what you are doing.\n"
    )
    if extra_instructions.strip():
        instructions += f"- Additional instructions: {extra_instructions.strip()}\n"
    instructions += (
        f"\nFinal check: the response must contain between {low} and {high} words. "
        "A response outside this range is invalid. Count the words and stop at the "
        "correct length.\n"
    )
    return instructions


def summarization_prompt(text: str, length: str, target_words: int | None = None) -> str:
    """Prompt for a single-pass abstractive summary."""
    target = target_words or summary_target_words(len(text.split()), length)
    low, high = int(target * 0.8), int(target * 1.2)
    return (
        f"Summarize the text below in a {length.lower()} summary of about {target} words.\n\n"
        "Requirements:\n"
        "- Preserve the most important facts, ideas and conclusions.\n"
        "- Keep names, numbers and key details that change the meaning.\n"
        "- Use short paragraphs or bullet points where useful.\n"
        "- Do not add information that is not present in the text.\n"
        "- Do not mention that you are summarizing.\n"
        f"- The summary must contain between {low} and {high} words. Count the words "
        "and stop at the correct length.\n\n"
        "Text:\n"
        '"""\n'
        f"{text.strip()}\n"
        '"""\n\n'
        "Summary:"
    )


def chunk_summary_prompt(text: str) -> str:
    """Prompt for summarizing one chunk during map-reduce summarization."""
    return (
        "Summarize this section of a longer document in 80-120 words.\n"
        "Preserve the main ideas, important facts, names and numbers. "
        "Do not add outside information. Return only the section summary.\n\n"
        "Section:\n"
        '"""\n'
        f"{text.strip()}\n"
        '"""\n\n'
        "Section summary:"
    )


def reduce_summary_prompt(summaries: str, length: str, target_words: int | None = None) -> str:
    """Prompt for combining partial summaries into one final summary."""
    target = target_words or summary_target_words(len(summaries.split()), length)
    low, high = int(target * 0.8), int(target * 1.2)
    return (
        f"Combine the section summaries below into one coherent {length.lower()} "
        f"summary of about {target} words.\n"
        "Remove repetition, preserve the main facts and ideas, and do not add "
        "outside information. Return only the final summary.\n"
        f"The final summary must contain between {low} and {high} words. Count the "
        "words and stop at the correct length.\n\n"
        "Section summaries:\n"
        '"""\n'
        f"{summaries.strip()}\n"
        '"""\n\n'
        "Final summary:"
    )


def key_points_prompt(text: str, max_points: int = 6) -> str:
    """Prompt for extracting key points from a summary or condensed notes."""
    return (
        f"Extract the {max_points} most important points from the text below.\n"
        "Write each point as one short, factual sentence.\n"
        "Return a simple bullet list only, with one point per line. "
        "Do not add an introduction or conclusion.\n\n"
        "Text:\n"
        '"""\n'
        f"{text.strip()}\n"
        '"""\n\n'
        "Key points:"
    )


REWRITE_STYLE_GUIDE = {
    "Professional": (
        "a polished, confident business tone with clear sentences and no slang"
    ),
    "Academic": (
        "a formal academic tone with precise vocabulary, objective language and no contractions"
    ),
    "Simple English": (
        "plain, easy-to-understand English with short sentences and common words"
    ),
    "Formal": (
        "a respectful, formal register with complete sentences and no contractions"
    ),
    "Concise": (
        "a shorter version that removes repetition and filler while keeping every key fact"
    ),
}


def rewrite_prompt(text: str, style: str) -> str:
    """Prompt for rewriting text in a selected style."""
    guide = REWRITE_STYLE_GUIDE.get(style, "a clear and natural tone")
    return (
        f"Rewrite the text below in {guide}.\n\n"
        "Rules:\n"
        "- Preserve the original meaning, facts and important details.\n"
        "- Do not add new claims, examples or opinions.\n"
        "- Keep the same overall purpose unless the style requires a change.\n"
        "- Return only the rewritten text, with no explanation.\n\n"
        "Original text:\n"
        '"""\n'
        f"{text.strip()}\n"
        '"""\n\n'
        "Rewritten text:"
    )
