"""TF-IDF keyword and important-sentence extraction (no LLM required)."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.text_cleaner import clean_text, get_stopwords, split_sentences, word_count


def _fallback_keywords(text: str, top_n: int) -> dict:
    """Simple frequency fallback if TF-IDF cannot build a vocabulary."""
    stopwords = get_stopwords()
    words = [
        word.lower()
        for word in re.findall(r"(?u)\b[^\W\d_][\w'-]{2,}\b", text)
        if word.lower() not in stopwords
    ]
    counts = Counter(words)
    top = counts.most_common(top_n)
    maximum = top[0][1] if top else 1
    sentences = split_sentences(text)
    return {
        "keywords": [
            {
                "keyword": word,
                "score": float(count),
                "normalized_score": round(count / maximum, 3),
            }
            for word, count in top
        ],
        "sentences": [
            {"sentence": sentence, "score": 1.0 if index == 0 else 0.5}
            for index, sentence in enumerate(sentences[:5])
        ],
        "word_count": word_count(text),
        "sentence_count": len(sentences),
    }


def extract_keywords(text: str, top_n: int = 12) -> dict:
    """Return top TF-IDF keywords and the most important sentences.

    Keywords are scored across sentences, so a term that is important in
    several parts of the text ranks higher than a one-off mention.
    """
    cleaned = clean_text(text)
    if word_count(cleaned) < 15:
        raise ValueError("Please provide at least 15 words for keyword extraction.")
    if not 3 <= top_n <= 50:
        raise ValueError("Top keywords must be between 3 and 50.")

    sentences = [s.strip() for s in split_sentences(cleaned) if word_count(s) >= 4]
    if not sentences:
        sentences = [cleaned]

    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(get_stopwords()),
            ngram_range=(1, 1),
            sublinear_tf=True,
            max_features=3000,
            token_pattern=r"(?u)\b[^\W\d_][\w'-]{2,}\b",
        )
        matrix = vectorizer.fit_transform(sentences)
        terms = vectorizer.get_feature_names_out()
        keyword_scores = np.asarray(matrix.sum(axis=0)).ravel()
    except ValueError:
        return _fallback_keywords(cleaned, top_n)

    if keyword_scores.size == 0:
        return _fallback_keywords(cleaned, top_n)

    order = np.argsort(keyword_scores)[::-1]
    maximum = float(keyword_scores[order[0]]) or 1.0

    keywords = []
    for index in order[:top_n]:
        score = float(keyword_scores[index])
        keywords.append(
            {
                "keyword": str(terms[index]),
                "score": round(score, 4),
                "normalized_score": round(score / maximum, 3),
            }
        )

    sentence_scores = np.asarray(matrix.sum(axis=1)).ravel()
    sentence_max = float(sentence_scores.max()) or 1.0
    best = np.argsort(sentence_scores)[::-1][: min(5, len(sentences))]
    ordered = sorted(int(index) for index in best)

    return {
        "keywords": keywords,
        "sentences": [
            {
                "sentence": sentences[index],
                "score": round(float(sentence_scores[index]) / sentence_max, 3),
            }
            for index in ordered
        ],
        "word_count": word_count(cleaned),
        "sentence_count": len(sentences),
    }


def keywords_as_text(result: dict) -> str:
    """Format keyword results for TXT/DOCX/PDF exports."""
    lines = ["Top Keywords", "------------"]
    for rank, item in enumerate(result.get("keywords", []), start=1):
        lines.append(f"{rank}. {item['keyword']}  (score: {item['score']})")
    lines.extend(["", "Important Sentences", "-------------------"])
    for item in result.get("sentences", []):
        lines.append(f"- {item['sentence']}  (score: {item['score']})")
    return "\n".join(lines)
