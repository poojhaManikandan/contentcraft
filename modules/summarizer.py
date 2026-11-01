"""Extractive (TF-IDF) and abstractive (local LLM) summarization."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ai_engine import prompts
from ai_engine.errors import ModelError
from ai_engine.model_client import TextModel
from ai_engine.provider import create_model
from utils.text_cleaner import (
    chunk_text,
    clean_text,
    get_stopwords,
    parse_bullet_points,
    split_sentences,
    strip_leading_label,
    truncate_text,
    word_count,
)

SUMMARY_LENGTHS = ("Short", "Medium", "Detailed")
METHODS = ("Extractive", "Abstractive")

ProgressCallback = Callable[[float, str], None]

_RATIOS = {"Short": 0.18, "Medium": 0.30, "Detailed": 0.45}
_MIN_SENTENCES = {"Short": 2, "Medium": 3, "Detailed": 4}
_MAX_SENTENCES = {"Short": 5, "Medium": 9, "Detailed": 14}
_FINAL_TOKENS = {"Short": 380, "Medium": 720, "Detailed": 1150}


def _notify(callback: Optional[ProgressCallback], fraction: float, message: str) -> None:
    """Send progress updates without letting UI errors break generation."""
    if callback is None:
        return
    try:
        callback(max(0.0, min(1.0, fraction)), message)
    except Exception:
        pass


def _validate(text: str, length: str) -> str:
    cleaned = clean_text(text)
    if word_count(cleaned) < 10:
        raise ValueError("Please provide at least 10 words to summarize.")
    if length not in SUMMARY_LENGTHS:
        raise ValueError(f"Unknown summary length: {length}")
    return cleaned


def _score_sentences(sentences: list[str]) -> np.ndarray:
    """Score each sentence with TF-IDF; fall back to length when vocabulary is empty."""
    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(get_stopwords()),
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        matrix = vectorizer.fit_transform(sentences)
        scores = np.asarray(matrix.sum(axis=1)).ravel()
    except ValueError:
        scores = np.array([word_count(sentence) for sentence in sentences], dtype=float)

    if len(scores) > 0:
        # A small position bonus helps the opening sentence act as a lead.
        scores[0] *= 1.15
        if len(scores) > 1:
            scores[-1] *= 1.05
    return scores


def _valid_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in split_sentences(text) if word_count(s) >= 4]
    if len(sentences) < 2:
        sentences = [s.strip() for s in split_sentences(text) if s.strip()]
    return sentences


def extractive_summary(text: str, length: str = "Medium") -> str:
    """Select the highest-scoring sentences using TF-IDF.

    Extractive summaries keep the original wording, so they run fully offline
    and never invent new information.
    """
    cleaned = _validate(text, length)
    sentences = _valid_sentences(cleaned)
    if not sentences:
        return cleaned
    if len(sentences) <= 2:
        return " ".join(sentences)

    target = int(round(len(sentences) * _RATIOS[length]))
    target = max(_MIN_SENTENCES[length], target)
    target = min(_MAX_SENTENCES[length], target, len(sentences))
    if target >= len(sentences):
        return " ".join(sentences)

    scores = _score_sentences(sentences)
    best = np.argsort(scores)[::-1][:target]
    ordered = sorted(int(index) for index in best)
    return " ".join(sentences[index] for index in ordered)


def important_sentences(text: str, count: int = 6) -> list[str]:
    """Return the most information-rich sentences (used for extractive key points)."""
    cleaned = clean_text(text)
    sentences = _valid_sentences(cleaned)
    if not sentences:
        return []
    count = max(1, min(count, len(sentences)))
    scores = _score_sentences(sentences)
    best = np.argsort(scores)[::-1][:count]
    ordered = sorted(int(index) for index in best)
    return [sentences[index] for index in ordered]


def _clean_model_text(text: str, labels: list[str]) -> str:
    return strip_leading_label(text, labels)


def _summarize_chunks(
    chunks: list[str],
    model: TextModel,
    callback: Optional[ProgressCallback],
    base: float,
    span: float,
) -> list[str]:
    """Summarize each chunk (the 'map' phase of map-reduce)."""
    partials: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        summary = model.generate(
            prompts.chunk_summary_prompt(chunk),
            system=prompts.SYSTEM_WRITER,
            temperature=0.3,
            top_p=0.9,
            max_tokens=300,
        )
        partials.append(_clean_model_text(summary, ["Section summary:", "Summary:"]))
        _notify(
            callback,
            base + span * (index / len(chunks)),
            f"Summarizing section {index} of {len(chunks)}...",
        )
    return partials


def _map_reduce(
    text: str,
    length: str,
    model: TextModel,
    callback: Optional[ProgressCallback],
) -> tuple[str, str, int]:
    """Summarize a long text by summarizing chunks and combining the results."""
    chunks = chunk_text(text, max_chars=3500)
    if not chunks:
        raise ValueError("The document does not contain readable text.")
    total_chunks = len(chunks)
    target_words = prompts.summary_target_words(word_count(text), length)

    # A short input can be summarized directly, without a map-reduce round trip.
    if total_chunks == 1:
        _notify(callback, 0.1, "Analyzing the text...")
        summary = model.generate(
            prompts.summarization_prompt(chunks[0], length, target_words),
            system=prompts.SYSTEM_WRITER,
            temperature=0.3,
            top_p=0.9,
            max_tokens=_FINAL_TOKENS[length],
        )
        _notify(callback, 0.9, "Summary ready.")
        return _clean_model_text(summary, ["Final summary:", "Summary:"]), chunks[0], 1

    _notify(callback, 0.05, f"Summarizing {total_chunks} sections...")
    partials = _summarize_chunks(chunks, model, callback, base=0.05, span=0.60)
    combined = "\n\n".join(partials)

    # If the combined notes are still long, reduce them again before the final call.
    for round_number in range(1, 4):
        if len(combined) <= 6500:
            break
        sub_chunks = chunk_text(combined, max_chars=3500)
        if len(sub_chunks) <= 1:
            break
        partials = _summarize_chunks(
            sub_chunks,
            model,
            callback,
            base=0.65,
            span=0.05 * round_number,
        )
        new_combined = "\n\n".join(partials)
        if len(new_combined) >= len(combined):
            break
        combined = new_combined

    _notify(callback, 0.82, "Combining section summaries...")
    final = model.generate(
        prompts.reduce_summary_prompt(combined, length, target_words),
        system=prompts.SYSTEM_WRITER,
        temperature=0.3,
        top_p=0.9,
        max_tokens=_FINAL_TOKENS[length],
    )
    _notify(callback, 0.92, "Final summary ready.")
    return _clean_model_text(final, ["Final summary:", "Summary:"]), combined, total_chunks


def abstractive_summary(
    text: str,
    length: str = "Medium",
    model: Optional[TextModel] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """Generate a concise summary in the model's own words."""
    cleaned = _validate(text, length)
    model = model or create_model()
    summary, _, _ = _map_reduce(cleaned, length, model, progress_callback)
    return summary


def abstractive_document_summary(
    text: str,
    length: str = "Medium",
    model: Optional[TextModel] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> dict:
    """Summarize a document and extract important points with the local model."""
    cleaned = _validate(text, length)
    model = model or create_model()
    summary, condensed, chunk_count = _map_reduce(cleaned, length, model, progress_callback)

    points: list[str] = []
    points_error: Optional[str] = None
    _notify(progress_callback, 0.95, "Extracting important points...")
    try:
        raw_points = model.generate(
            prompts.key_points_prompt(truncate_text(condensed, 6000), max_points=6),
            system=prompts.SYSTEM_WRITER,
            temperature=0.2,
            top_p=0.9,
            max_tokens=350,
        )
        points = parse_bullet_points(raw_points, max_points=6)
    except ModelError as exc:
        # The summary is still useful, so keep it and report the partial failure.
        points_error = str(exc)

    _notify(progress_callback, 1.0, "Document summary complete.")
    return {
        "summary": summary,
        "key_points": points,
        "key_points_error": points_error,
        "chunk_count": chunk_count,
    }


def summarize_text(
    text: str,
    method: str = "Abstractive",
    length: str = "Medium",
    model: Optional[TextModel] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> dict:
    """Convenience wrapper used by the UI and tests."""
    if method not in METHODS:
        raise ValueError(f"Unknown summarization method: {method}")
    cleaned = _validate(text, length)

    if method == "Extractive":
        _notify(progress_callback, 0.5, "Scoring sentences with TF-IDF...")
        summary = extractive_summary(cleaned, length)
        _notify(progress_callback, 1.0, "Summary ready.")
        return {"summary": summary, "method": method, "length": length, "chunk_count": 0}

    if model is None:
        model = create_model()
    summary, _, chunk_count = _map_reduce(cleaned, length, model, progress_callback)
    return {"summary": summary, "method": method, "length": length, "chunk_count": chunk_count}
