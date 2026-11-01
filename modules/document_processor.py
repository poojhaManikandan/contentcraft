"""Extract text from TXT, DOCX and PDF files, then summarize documents."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Optional

import pymupdf
from docx import Document

from ai_engine.model_client import TextModel
from modules.summarizer import (
    METHODS,
    SUMMARY_LENGTHS,
    abstractive_document_summary,
    extractive_summary,
    important_sentences,
)
from utils.text_cleaner import chunk_text, clean_text, word_count

ALLOWED_EXTENSIONS = (".pdf", ".docx", ".txt")
MAX_FILE_BYTES = 25 * 1024 * 1024  # Keep in sync with .streamlit/config.toml

ProgressCallback = Callable[[float, str], None]


class DocumentProcessingError(ValueError):
    """Raised when an uploaded file cannot be read or contains no text."""


def _notify(callback: Optional[ProgressCallback], fraction: float, message: str) -> None:
    if callback is None:
        return
    try:
        callback(max(0.0, min(1.0, fraction)), message)
    except Exception:
        pass


def _extract_pdf(data: bytes) -> str:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentProcessingError("The PDF file could not be opened. It may be damaged.") from exc

    try:
        if document.needs_pass:
            raise DocumentProcessingError("This PDF is password-protected. Remove the password and upload it again.")
        pages = [page.get_text("text") for page in document]
        text = "\n\n".join(pages)
    finally:
        document.close()

    if not word_count(text):
        raise DocumentProcessingError(
            "No readable text was found in this PDF. Scanned or image-only PDFs need OCR, "
            "which is not part of this project."
        )
    return text


def _extract_docx(data: bytes) -> str:
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentProcessingError(
            "The DOCX file could not be opened. Make sure it is a valid .docx file."
        ) from exc

    parts = [paragraph.text for paragraph in document.paragraphs]
    # Include table text because reports often place key information there.
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _extract_txt(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DocumentProcessingError("The TXT file could not be decoded as text.")

    if "\x00" in text:
        raise DocumentProcessingError("The file does not appear to be a plain text file.")
    return text


def extract_document(filename: str, data: bytes) -> str:
    """Validate a file and return its cleaned text."""
    name = (filename or "").strip()
    extension = Path(name).suffix.lower()

    if not name:
        raise DocumentProcessingError("The uploaded file has no name.")
    if extension not in ALLOWED_EXTENSIONS:
        supported = ", ".join(ALLOWED_EXTENSIONS)
        raise DocumentProcessingError(f"Unsupported file type '{extension or 'unknown'}'. Supported types: {supported}.")
    if not data:
        raise DocumentProcessingError("The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise DocumentProcessingError("The file is larger than 25 MB. Please upload a smaller document.")

    if extension == ".pdf":
        text = _extract_pdf(data)
    elif extension == ".docx":
        text = _extract_docx(data)
    else:
        text = _extract_txt(data)

    cleaned = clean_text(text)
    if word_count(cleaned) < 3:
        raise DocumentProcessingError("The document contains no readable text.")
    return cleaned


def process_document(
    filename: str,
    data: bytes,
    length: str = "Medium",
    method: str = "Abstractive",
    model: Optional[TextModel] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> dict:
    """Extract text and create a summary plus important points.

    Abstractive mode uses chunked map-reduce summarization for long documents.
    Extractive mode uses local TF-IDF scoring and needs no model.
    """
    if length not in SUMMARY_LENGTHS:
        raise ValueError(f"Unknown summary length: {length}")
    if method not in METHODS:
        raise ValueError(f"Unknown summarization method: {method}")

    _notify(progress_callback, 0.03, "Extracting text from the document...")
    text = extract_document(filename, data)
    total_words = word_count(text)
    total_chunks = len(chunk_text(text, max_chars=3500))
    _notify(progress_callback, 0.08, f"Extracted {total_words} words. Preparing summary...")

    points_error = None
    if method == "Extractive":
        _notify(progress_callback, 0.4, "Scoring sentences with TF-IDF...")
        summary = extractive_summary(text, length)
        key_points = important_sentences(text, count=6)
        chunk_count = total_chunks
        _notify(progress_callback, 1.0, "Document summary complete.")
    else:
        result = abstractive_document_summary(
            text,
            length=length,
            model=model,
            progress_callback=progress_callback,
        )
        summary = result["summary"]
        key_points = result["key_points"]
        points_error = result.get("key_points_error")
        chunk_count = result["chunk_count"]

    return {
        "filename": Path(filename).name,
        "word_count": total_words,
        "char_count": len(text),
        "summary": summary,
        "key_points": key_points,
        "key_points_error": points_error,
        "chunk_count": chunk_count,
        "method": method,
        "length": length,
        "summary_word_count": word_count(summary),
    }
