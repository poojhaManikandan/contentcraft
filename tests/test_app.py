"""Streamlit UI smoke tests using the official AppTest harness."""

from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest

PAGES = (
    "Dashboard",
    "Content Generator",
    "Text Summarizer",
    "Document Summarizer",
    "Text Rewriter",
    "Keyword Extractor",
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
    "Used carefully, AI can support learning without replacing the teacher."
)


class AppSmokeTests(unittest.TestCase):
    def test_every_page_loads(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)

        for page in PAGES:
            app.sidebar.radio[0].set_value(page).run()
            self.assertEqual(len(app.exception), 0, f"Page failed: {page}")

    def test_hosted_ui_hides_provider_details(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=60).run()
        rendered = " ".join(str(element.value) for element in app.markdown)
        rendered += " " + " ".join(str(element.value) for element in app.caption)
        lowered = rendered.lower()

        self.assertNotIn("gemini", lowered)
        self.assertNotIn("google", lowered)
        self.assertNotIn("generativelanguage", lowered)

    def test_extractive_summary_ui_flow(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=60).run()
        app.sidebar.radio[0].set_value("Text Summarizer").run()
        app.text_area[0].set_value(SAMPLE_TEXT).run()

        next(button for button in app.button if button.label == "Summarize text").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        result = app.session_state["summary_result"]
        self.assertTrue(result["summary"])
        self.assertLess(result["summary_words"], result["original_words"])

    def test_keyword_ui_flow(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=60).run()
        app.sidebar.radio[0].set_value("Keyword Extractor").run()
        app.text_area[0].set_value(SAMPLE_TEXT).run()

        next(button for button in app.button if button.label == "Extract keywords").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        result = app.session_state["keyword_result"]["result"]
        self.assertGreater(len(result["keywords"]), 0)
        self.assertGreater(len(result["sentences"]), 0)


if __name__ == "__main__":
    unittest.main()
