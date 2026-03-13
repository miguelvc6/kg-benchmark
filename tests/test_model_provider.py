import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from guardian.model_provider import OllamaChatProvider, OpenAIChatProvider, create_model_provider


class OpenAIChatProviderTests(unittest.TestCase):
    def test_loads_api_settings_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / ".env").write_text(
                "OPENAI_API_KEY=test-key\nOPENAI_MODEL=test-model\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                previous_cwd = Path.cwd()
                try:
                    os.chdir(root)
                    provider = OpenAIChatProvider()
                finally:
                    os.chdir(previous_cwd)

            self.assertEqual(provider.api_key, "test-key")
            self.assertEqual(provider.model, "test-model")
            self.assertEqual(provider.base_url, "https://api.openai.com/v1")

    def test_factory_allows_model_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / ".env").write_text(
                "MODEL_PROVIDER=openai\nOPENAI_API_KEY=test-key\nOPENAI_MODEL=env-model\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                previous_cwd = Path.cwd()
                try:
                    os.chdir(root)
                    provider = create_model_provider("override-model")
                finally:
                    os.chdir(previous_cwd)

            self.assertIsInstance(provider, OpenAIChatProvider)
            self.assertEqual(provider.model, "override-model")

    def test_strips_common_api_key_wrappers(self) -> None:
        with patch.dict(os.environ, {"OPENAI_MODEL": "test-model"}, clear=True):
            provider = OpenAIChatProvider(api_key="Bearer OPENAI_API_KEY=test-key")

        self.assertEqual(provider.api_key, "test-key")

    def test_omits_temperature_for_gpt5_models(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": "{\"case_id\": \"c1\"}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }

        with patch("guardian.model_provider.requests.post", return_value=response) as post:
            provider = OpenAIChatProvider(api_key="test-key", model="gpt-5-mini-2025-08-07")
            raw, parsed, usage = provider.generate(
                prompt="{}",
                system_prompt="Return JSON only.",
                response_format={"type": "json_object"},
                metadata={"case_id": "c1"},
            )

        self.assertEqual(parsed, {"case_id": "c1"})
        self.assertEqual(usage["provider"], "openai")
        self.assertEqual(raw["usage"]["total_tokens"], 18)
        request_payload = json.loads(post.call_args.kwargs["data"].decode("utf-8"))
        self.assertNotIn("temperature", request_payload)
        self.assertEqual(request_payload["tool_choice"], "none")

    def test_rejects_tool_call_responses(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "search"}}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }

        with patch("guardian.model_provider.requests.post", return_value=response):
            provider = OpenAIChatProvider(api_key="test-key", model="gpt-5-mini-2025-08-07")
            with self.assertRaisesRegex(RuntimeError, "disables tool use"):
                provider.generate(
                    prompt="{}",
                    system_prompt="Return JSON only.",
                    response_format={"type": "json_object"},
                    metadata={"case_id": "c1"},
                )

    def test_fails_before_http_when_payload_is_not_strict_json(self) -> None:
        provider = OpenAIChatProvider(api_key="test-key", model="gpt-5-mini-2025-08-07")

        with self.assertRaisesRegex(RuntimeError, "case_id='c1'"):
            provider.generate(
                prompt="{}",
                system_prompt="Return JSON only.",
                response_format={"type": "json_object", "bad": float("nan")},
                metadata={"case_id": "c1", "task_type": "proposal"},
            )


class OllamaChatProviderTests(unittest.TestCase):
    def test_loads_model_settings_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / ".env").write_text(
                "MODEL_PROVIDER=ollama\nOLLAMA_MODEL=llama3.2\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                previous_cwd = Path.cwd()
                try:
                    os.chdir(root)
                    provider = create_model_provider()
                finally:
                    os.chdir(previous_cwd)

            self.assertIsInstance(provider, OllamaChatProvider)
            self.assertEqual(provider.model, "llama3.2")
            self.assertEqual(provider.base_url, "http://localhost:11434/api")

    def test_maps_ollama_chat_response(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "{\"case_id\": \"c1\"}"},
            "prompt_eval_count": 11,
            "eval_count": 7,
        }

        with patch("guardian.model_provider.requests.post", return_value=response) as post:
            provider = OllamaChatProvider(model="llama3.2")
            raw, parsed, usage = provider.generate(
                prompt="{}",
                system_prompt="Return JSON only.",
                response_format={"type": "json_object"},
                metadata={"case_id": "c1"},
            )

        self.assertEqual(parsed, {"case_id": "c1"})
        self.assertEqual(usage["provider"], "ollama")
        self.assertEqual(usage["prompt_tokens"], 11)
        self.assertEqual(usage["completion_tokens"], 7)
        self.assertEqual(usage["total_tokens"], 18)
        self.assertEqual(raw["model"], "llama3.2")
        self.assertEqual(post.call_args.kwargs["json"]["format"], "json")


if __name__ == "__main__":
    unittest.main()
