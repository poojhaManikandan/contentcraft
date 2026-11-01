"""Text cleaning, sentence splitting and chunking helpers.

NLTK is used for sentence tokenization and stopwords. If the required NLTK
data is not available (for example, on a machine with no internet during the
first run), simple rule-based fallbacks keep the application usable.
"""

from __future__ import annotations

import re
from typing import Iterable

import nltk

# Remember attempts so we do not try to download the same data repeatedly.
_NLTK_STATE = {
    "sentences": False,
    "sentences_failed": False,
    "stopwords": False,
    "stopwords_failed": False,
}

_FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves",
}


def _ensure_sentence_tokenizer() -> bool:
    """Make sure NLTK's sentence tokenizer data is present."""
    if _NLTK_STATE["sentences_failed"]:
        return False
    if _NLTK_STATE["sentences"]:
        return True
    try:
        nltk.data.find("tokenizers/punkt_tab")
        _NLTK_STATE["sentences"] = True
        return True
    except LookupError:
        pass
    try:
        # punkt_tab is required by NLTK 3.9+, punkt by older versions.
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.data.find("tokenizers/punkt_tab")
        _NLTK_STATE["sentences"] = True
    except (LookupError, OSError):
        # Offline or download blocked: the regex fallback below is used.
        _NLTK_STATE["sentences_failed"] = True
    return _NLTK_STATE["sentences"]


def get_stopwords() -> set[str]:
    """Return English stopwords, with a built-in fallback list."""
    if _NLTK_STATE["stopwords_failed"]:
        return set(_FALLBACK_STOPWORDS)
    if _NLTK_STATE["stopwords"]:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            nltk.data.find("corpora/stopwords")
        except (LookupError, OSError):
            _NLTK_STATE["stopwords_failed"] = True
            return set(_FALLBACK_STOPWORDS)
    try:
        from nltk.corpus import stopwords

        words = set(stopwords.words("english"))
        _NLTK_STATE["stopwords"] = True
        return words
    except LookupError:
        _NLTK_STATE["stopwords_failed"] = True
        return set(_FALLBACK_STOPWORDS)


def clean_text(text: str) -> str:
    """Normalise line endings and collapse excessive whitespace."""
    if text is None:
        return ""
    text = str(text).replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    """Count words using a Unicode-friendly pattern."""
    return len(re.findall(r"\b[\w'-]+\b", text or "", flags=re.UNICODE))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, falling back to a regex when NLTK is unavailable."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if _ensure_sentence_tokenizer():
        try:
            sentences = nltk.sent_tokenize(cleaned)
            return [s.strip() for s in sentences if s.strip()]
        except (LookupError, ValueError):
            pass

    # Fallback: split after ., ! or ? followed by whitespace and a capital letter.
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split a very long sentence/word block into pieces under max_chars."""
    pieces: list[str] = []
    current = ""
    for token in text.split():
        if len(token) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(token[i : i + max_chars] for i in range(0, len(token), max_chars))
            continue
        candidate = f"{current} {token}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = token
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, max_chars: int = 3500) -> list[str]:
    """Break a long document into paragraph/sentence-aware chunks.

    Chunks are kept below ``max_chars`` so that each one fits comfortably in
    the local model's context window.
    """
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    cleaned = clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    # Build small units first: paragraphs, then sentences for long paragraphs.
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", cleaned):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        for sentence in split_sentences(paragraph):
            if len(sentence) <= max_chars:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, max_chars))

    # Pack units into chunks without cutting sentences.
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text at a word boundary and add an ellipsis when needed."""
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[: max(0, max_chars - 1)]
    if " " in cut[max_chars // 2 :]:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "…"


def compression_ratio(original_words: int, summary_words: int) -> float:
    """Return the percentage of the original text kept by the summary."""
    if original_words <= 0:
        return 0.0
    return round((summary_words / original_words) * 100, 1)


def join_items(items: Iterable[str], separator: str = " ") -> str:
    """Join non-empty strings, useful for formatting summaries."""
    return separator.join(item.strip() for item in items if item and item.strip())


def strip_code_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence if the model added one."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```") and cleaned.endswith("```") and len(cleaned) > 6:
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return cleaned


def strip_leading_label(text: str, labels: Iterable[str]) -> str:
    """Remove a leading prompt label such as 'Summary:' from model output."""
    cleaned = strip_code_fences(text)
    for label in labels:
        if cleaned.lower().startswith(label.lower()):
            cleaned = cleaned[len(label) :].lstrip(" :\n\t")
            break
    return cleaned.strip()


def parse_bullet_points(text: str, max_points: int = 10) -> list[str]:
    """Parse bullet/list style model output into clean point strings."""
    points: list[str] = []
    for line in strip_code_fences(text).splitlines():
        line = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s*", "", line).strip()
        if not line:
            continue
        if line.lower().rstrip(":") in {"key points", "important points"}:
            continue
        if len(line) > 3:
            points.append(line)
        if len(points) >= max_points:
            break
    return points
