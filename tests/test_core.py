"""Unit tests for ContentCraft's non-UI modules."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import pymupdf
from docx import Document

from ai_engine.local_model import LocalModel, LocalModelError
from modules.content_generator import generate_content
from modules.document_processor import (
    DocumentProcessingError,
    extract_document,
    process_document,
)
from modules.keyword_extractor import extract_keywords, keywords_as_text
from modules.summarizer import (
    abstractive_document_summary,
    abstractive_summary,
    extractive_summary,
    important_sentences,
    summarize_text,
)
from modules.text_rewriter import rewrite_text
from utils.export_manager import build_docx, build_pdf, build_txt, export_filename
from utils.text_cleaner import (
    chunk_text,
    clean_text,
    compression_ratio,
    parse_bullet_points,
    split_sentences,
    word_count,
)

SAMPLE_TEXT = (
    "Artificial intelligence is changing how colleges teach and how students learn. "
    "Teachers use intelligent tutoring systems to give learners immediate feedback. "
    "Students use summarization tools to review long readings before examinations. "
    "However, AI tools also raise concerns about accuracy, privacy and academic honesty. "
    "Colleges therefore need clear policies about when AI assistance is allowed. "
    "They should also teach students how to check facts and cite sources properly. "
    "A balanced approach combines useful technology with human judgement. "
    "Teachers remain responsible for designing meaningful assessments. "
    "Students remain responsible for understanding and explaining their own work. "
    "Used carefully, AI can support learning without replacing the teacher. "
)

LONG_TEXT = "\n\n".join(
    [
        (
            "Renewable energy projects can strengthen rural economies in several ways. "
            "Solar installations create construction jobs and long-term maintenance work. "
            "Wind farms provide lease income to landowners who host turbines. "
        ),
        (
            "Community solar programs let households buy shares without installing panels. "
            "Local ownership keeps more of the revenue inside the region. "
            "Training programs help workers move into technical roles. "
        ),
        (
            "Grid upgrades are often needed before new projects can connect. "
            "Storage systems improve reliability when sunshine and wind are variable. "
            "Planning should include environmental and land-use reviews. "
        ),
        (
            "Small businesses can benefit from lower and more predictable energy costs. "
            "Farmers can combine crop production with solar grazing. "
            "Schools can use savings to fund other priorities. "
        ),
    ]
) * 8


class FakeModel:
    """Duck-typed stand-in for LocalModel used to test prompt wiring."""

    model = "fake-model"

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        if "Extract the" in prompt and "key points" not in prompt.lower():
            return "- AI supports personalized learning.\n- Policies must protect privacy.\n- Teachers remain essential."
        if "Extract the" in prompt:
            return "- Local ownership keeps revenue in the region.\n- Training creates technical jobs.\n- Grid upgrades improve reliability."
        if "Summarize this section" in prompt:
            return "This section explains how renewable projects create local jobs, income and energy savings."
        if "Combine the section summaries" in prompt:
            return "Renewable energy can strengthen rural economies through jobs, local ownership and grid investment."
        if "Summarize the text below" in prompt:
            return "AI can improve college learning, but colleges must manage accuracy, privacy and academic honesty."
        if "Rewrite the text below" in prompt:
            return "This is a professionally rewritten version of the supplied passage."
        return "A newly generated piece of content about the requested topic."


class TextCleanerTests(unittest.TestCase):
    def test_clean_and_count(self) -> None:
        text = "  Hello   world!\r\n\r\n\r\nThis is a test.  "
        self.assertEqual(clean_text(text), "Hello world!\n\nThis is a test.")
        self.assertEqual(word_count("One two three, four."), 4)

    def test_sentence_splitting(self) -> None:
        sentences = split_sentences("First sentence. Second sentence! Third question?")
        self.assertEqual(len(sentences), 3)

    def test_chunking_respects_limit(self) -> None:
        chunks = chunk_text(LONG_TEXT, max_chars=500)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))
        self.assertEqual("".join(chunks).replace("\n\n", "").replace(" ", ""),
                         clean_text(LONG_TEXT).replace("\n\n", "").replace(" ", ""))

    def test_short_text_is_single_chunk(self) -> None:
        self.assertEqual(chunk_text("A short document.", max_chars=500), ["A short document."])

    def test_bullet_parser(self) -> None:
        points = parse_bullet_points("Key points:\n- First point\n2. Second point\n* Third", 5)
        self.assertEqual(points, ["First point", "Second point", "Third"])

    def test_compression_ratio(self) -> None:
        self.assertEqual(compression_ratio(200, 50), 25.0)
        self.assertEqual(compression_ratio(0, 10), 0.0)


class SummarizerTests(unittest.TestCase):
    def test_extractive_summary_is_shorter(self) -> None:
        summary = extractive_summary(SAMPLE_TEXT, "Short")
        self.assertTrue(summary)
        self.assertLess(word_count(summary), word_count(SAMPLE_TEXT))

    def test_extractive_requires_enough_words(self) -> None:
        with self.assertRaises(ValueError):
            extractive_summary("Too short to summarize.", "Short")

    def test_important_sentences(self) -> None:
        sentences = important_sentences(SAMPLE_TEXT, count=3)
        self.assertEqual(len(sentences), 3)
        for sentence in sentences:
            self.assertIn(sentence, SAMPLE_TEXT)

    def test_abstractive_single_chunk(self) -> None:
        fake = FakeModel()
        summary = abstractive_summary(SAMPLE_TEXT, "Medium", model=fake)
        self.assertIn("AI can improve college learning", summary)
        self.assertEqual(len(fake.prompts), 1)

    def test_abstractive_long_text_uses_chunks(self) -> None:
        fake = FakeModel()
        summary = abstractive_summary(LONG_TEXT, "Short", model=fake)
        self.assertTrue(summary)
        self.assertGreater(len(fake.prompts), 1)

    def test_document_summary_with_key_points(self) -> None:
        fake = FakeModel()
        result = abstractive_document_summary(LONG_TEXT, "Medium", model=fake)
        self.assertTrue(result["summary"])
        self.assertGreaterEqual(len(result["key_points"]), 2)
        self.assertIsNone(result["key_points_error"])

    def test_summarize_text_dispatch(self) -> None:
        result = summarize_text(SAMPLE_TEXT, "Extractive", "Medium")
        self.assertEqual(result["method"], "Extractive")
        self.assertTrue(result["summary"])
        self.assertEqual(result["chunk_count"], 0)


class KeywordTests(unittest.TestCase):
    def test_extract_keywords(self) -> None:
        result = extract_keywords(SAMPLE_TEXT, top_n=10)
        self.assertGreater(len(result["keywords"]), 0)
        self.assertLessEqual(len(result["keywords"]), 10)
        scores = [item["score"] for item in result["keywords"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0 <= item["normalized_score"] <= 1 for item in result["keywords"]))
        self.assertGreater(len(result["sentences"]), 0)
        self.assertIn("Top Keywords", keywords_as_text(result))

    def test_keyword_validation(self) -> None:
        with self.assertRaises(ValueError):
            extract_keywords("Too short.", top_n=10)


class GenerationAndRewriteTests(unittest.TestCase):
    def test_content_generation_prompt_and_validation(self) -> None:
        fake = FakeModel()
        content = generate_content(
            "Renewable energy in villages",
            "Blog Post",
            "Academic",
            300,
            extra_instructions="Include examples",
            variation=1,
            model=fake,
        )
        self.assertTrue(content)
        self.assertIn("Renewable energy in villages", fake.prompts[0])
        self.assertIn("Academic", fake.prompts[0])
        self.assertIn("regeneration", fake.prompts[0])
        self.assertIn("between 270 and 330 words", fake.prompts[0])
        self.assertGreaterEqual(fake.kwargs[0]["temperature"], 0.4)
        self.assertLessEqual(fake.kwargs[0]["temperature"], 0.6)

    def test_content_validation(self) -> None:
        with self.assertRaises(ValueError):
            generate_content("", "Blog Post", "Formal", 300, model=FakeModel())
        with self.assertRaises(ValueError):
            generate_content("Valid topic", "Unknown", "Formal", 300, model=FakeModel())

    def test_rewrite(self) -> None:
        fake = FakeModel()
        rewritten = rewrite_text(SAMPLE_TEXT, "Concise", model=fake)
        self.assertIn("professionally rewritten", rewritten)
        self.assertIn("shorter version", fake.prompts[0])
        with self.assertRaises(ValueError):
            rewrite_text("short", "Formal", model=fake)
        with self.assertRaises(ValueError):
            rewrite_text(SAMPLE_TEXT, "Unknown", model=fake)


class ExportTests(unittest.TestCase):
    def test_txt_export(self) -> None:
        data = build_txt("Test title", "Body paragraph.", {"Tone": "Formal"})
        text = data.decode("utf-8")
        self.assertIn("Test title", text)
        self.assertIn("Body paragraph.", text)
        self.assertIn("Tone: Formal", text)

    def test_docx_export(self) -> None:
        data = build_docx("Test title", "# Heading\n\n- First item\n\nA paragraph.", {"Tone": "Formal"})
        self.assertTrue(data.startswith(b"PK"))
        document = Document(io.BytesIO(data))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        self.assertIn("Test title", text)
        self.assertIn("First item", text)

    def test_pdf_export(self) -> None:
        data = build_pdf("Test title", "# Heading\n\nA paragraph with enough words to export.", {"Tone": "Formal"})
        self.assertTrue(data.startswith(b"%PDF"))
        document = pymupdf.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text("text") for page in document)
        document.close()
        self.assertIn("Test title", text)
        self.assertIn("A paragraph", text)

    def test_export_filename(self) -> None:
        self.assertTrue(export_filename("My Test Title", "txt").startswith("my-test-title-"))
        self.assertTrue(export_filename("", "pdf").startswith("contentcraft-"))


def _make_docx_bytes() -> bytes:
    document = Document()
    document.add_heading("Document title", level=1)
    document.add_paragraph(SAMPLE_TEXT)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _make_pdf_bytes() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Document title")
    y = 100
    for line in SAMPLE_TEXT.split(". "):
        page.insert_text((72, y), line[:80])
        y += 16
    data = document.tobytes()
    document.close()
    return data


class DocumentTests(unittest.TestCase):
    def test_txt_extraction(self) -> None:
        text = extract_document("notes.txt", SAMPLE_TEXT.encode("utf-8"))
        self.assertIn("Artificial intelligence", text)

    def test_docx_extraction(self) -> None:
        text = extract_document("notes.docx", _make_docx_bytes())
        self.assertIn("Document title", text)
        self.assertIn("Artificial intelligence", text)

    def test_pdf_extraction(self) -> None:
        text = extract_document("notes.pdf", _make_pdf_bytes())
        self.assertIn("Document title", text)

    def test_invalid_and_empty_files(self) -> None:
        with self.assertRaises(DocumentProcessingError):
            extract_document("notes.exe", b"data")
        with self.assertRaises(DocumentProcessingError):
            extract_document("notes.txt", b"")
        with self.assertRaises(DocumentProcessingError):
            extract_document("notes.docx", b"not a docx file")
        with self.assertRaises(DocumentProcessingError):
            extract_document("notes.txt", b"   \n  ")

    def test_process_document_extractive(self) -> None:
        result = process_document("notes.txt", SAMPLE_TEXT.encode("utf-8"), "Short", "Extractive")
        self.assertEqual(result["filename"], "notes.txt")
        self.assertGreater(result["word_count"], 20)
        self.assertTrue(result["summary"])
        self.assertGreater(len(result["key_points"]), 0)

    def test_process_document_abstractive_with_fake_model(self) -> None:
        fake = FakeModel()
        result = process_document(
            "notes.txt", LONG_TEXT.encode("utf-8"), "Short", "Abstractive", model=fake
        )
        self.assertTrue(result["summary"])
        self.assertGreaterEqual(len(result["key_points"]), 1)
        self.assertGreater(result["chunk_count"], 1)


class LocalModelTests(unittest.TestCase):
    def test_offline_status_is_honest(self) -> None:
        model = LocalModel(host="http://127.0.0.1:1", timeout=1)
        status = model.check()
        self.assertFalse(status.connected)
        self.assertFalse(status.model_available)
        self.assertIn("not reachable", status.message)
        with self.assertRaises(LocalModelError):
            model.generate("Hello")

    def test_generate_with_mocked_ollama(self) -> None:
        class Response:
            def __init__(self, data, status_code=200):
                self._data = data
                self.status_code = status_code
                self.text = ""

            def json(self):
                return self._data

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError("HTTP error")

        with patch("ai_engine.local_model.requests.get") as get, patch(
            "ai_engine.local_model.requests.post"
        ) as post:
            get.return_value = Response({"models": [{"name": "qwen2.5:3b"}]})
            post.return_value = Response({"response": "  Hello from Ollama.  "})
            model = LocalModel(host="http://example.test")
            status = model.check()
            self.assertTrue(status.connected)
            self.assertTrue(status.model_available)
            answer = model.generate("Write something", max_tokens=100)
            self.assertEqual(answer, "Hello from Ollama.")
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["model"], "qwen2.5:3b")
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["options"]["num_predict"], 100)


if __name__ == "__main__":
    unittest.main()
