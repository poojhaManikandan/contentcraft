"""Tests for the hosted OpenAI-compatible provider and provider selection."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ai_engine.errors import ModelError
from ai_engine.hosted_model import HostedModel
from ai_engine.local_model import LocalModel
from ai_engine.provider import create_model, get_provider_config


class HostedResponse:
    def __init__(self, data: dict, status_code: int = 200, text: str = "") -> None:
        self._data = data
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return self._data


class HostedModelTests(unittest.TestCase):
    def test_missing_key_is_reported_honestly(self) -> None:
        model = HostedModel(api_key="")
        status = model.check()
        self.assertFalse(status.connected)
        self.assertFalse(status.model_available)
        self.assertIn("AI_API_KEY", status.message)
        with self.assertRaises(ModelError):
            model.generate("Write a short paragraph.")

    def test_generate_uses_chat_completions(self) -> None:
        response = HostedResponse(
            {"choices": [{"message": {"content": "  Hello from the hosted model.  "}}]}
        )
        with patch("ai_engine.hosted_model.requests.post", return_value=response) as post:
            model = HostedModel(
                api_key="test-key",
                base_url="https://api.groq.com/openai/v1",
                model="test-model",
                provider="Test Provider",
            )
            text = model.generate(
                "Write something",
                system="You are a writer.",
                temperature=0.4,
                max_tokens=250,
            )

        self.assertEqual(text, "Hello from the hosted model.")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"]["model"], "test-model")
        self.assertEqual(kwargs["json"]["messages"][0]["role"], "system")
        self.assertEqual(kwargs["json"]["messages"][1]["content"], "Write something")
        self.assertEqual(kwargs["json"]["max_tokens"], 250)
        self.assertFalse(kwargs["json"]["stream"])

    def test_gemini_model_404_uses_current_fallback(self) -> None:
        missing = HostedResponse({"error": {"message": "model not found"}}, status_code=404)
        success = HostedResponse(
            {"choices": [{"message": {"content": "Fallback response."}}]}
        )
        with patch(
            "ai_engine.hosted_model.requests.post", side_effect=[missing, success]
        ) as post:
            model = HostedModel(api_key="test-key", model="gemini-2.5-flash")
            text = model.generate("Hello")

        self.assertEqual(text, "Fallback response.")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[1].kwargs["json"]["model"], "gemini-3.8-flash")

    def test_unauthorized_response_is_clear(self) -> None:
        response = HostedResponse(
            {"error": {"message": "invalid key"}},
            status_code=401,
            text="invalid key",
        )
        with patch("ai_engine.hosted_model.requests.post", return_value=response):
            model = HostedModel(api_key="bad-key")
            with self.assertRaisesRegex(ModelError, "rejected the API key"):
                model.generate("Hello")

    def test_missing_choices_is_rejected(self) -> None:
        response = HostedResponse({"choices": []})
        with patch("ai_engine.hosted_model.requests.post", return_value=response):
            model = HostedModel(api_key="test-key")
            with self.assertRaisesRegex(ModelError, "no choices"):
                model.generate("Hello")


class ProviderSelectionTests(unittest.TestCase):
    def test_hosted_mode_reads_configuration(self) -> None:
        env = {
            "AI_MODE": "hosted",
            "AI_API_KEY": "test-key",
            "AI_BASE_URL": "https://api.groq.com/openai/v1",
            "AI_MODEL": "test-model",
        }
        with patch.dict(os.environ, env, clear=True):
            config = get_provider_config()
            model = create_model(config)

        self.assertTrue(config.hosted)
        self.assertTrue(config.key_configured)
        self.assertEqual(config.label, "Groq")
        self.assertEqual(config.model, "test-model")
        self.assertIsInstance(model, HostedModel)

    def test_gemini_defaults_and_key_fallback(self) -> None:
        env = {"AI_MODE": "hosted", "GEMINI_API_KEY": "gemini-test-key"}
        with patch.dict(os.environ, env, clear=True):
            config = get_provider_config()
            model = create_model(config)

        self.assertTrue(config.hosted)
        self.assertTrue(config.key_configured)
        self.assertEqual(config.label, "Google Gemini")
        self.assertEqual(config.model, "gemini-3.8-flash")
        self.assertEqual(
            config.endpoint,
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
        self.assertIsInstance(model, HostedModel)
        self.assertEqual(model.api_key, "gemini-test-key")

    def test_local_mode_still_selects_ollama(self) -> None:
        env = {
            "AI_MODE": "local",
            "OLLAMA_HOST": "http://localhost:11434",
            "OLLAMA_MODEL": "qwen2.5:3b",
        }
        with patch.dict(os.environ, env, clear=True):
            config = get_provider_config()
            model = create_model(config)

        self.assertFalse(config.hosted)
        self.assertEqual(config.label, "Local Ollama")
        self.assertIsInstance(model, LocalModel)


if __name__ == "__main__":
    unittest.main()
