import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from guardian.model_provider import (
    OllamaChatProvider,
    OpenAIChatProvider,
    StaticResponseProvider,
    _format_batch_progress,
    create_model_provider,
)


class OpenAIChatProviderTests(unittest.TestCase):
    def test_formats_batch_progress_with_eta(self) -> None:
        formatted = _format_batch_progress(
            {"total": 10, "completed": 4, "failed": 1},
            elapsed_seconds=20.0,
        )

        self.assertIn("progress=5/10", formatted)
        self.assertIn("eta=20s", formatted)
        self.assertIn("'completed': 4", formatted)

    def test_execute_batch_status_callback_includes_eta(self) -> None:
        upload_response = MagicMock()
        upload_response.raise_for_status.return_value = None
        upload_response.json.return_value = {"id": "file-input-1"}

        create_response = MagicMock()
        create_response.raise_for_status.return_value = None
        create_response.json.return_value = {
            "id": "batch-1",
            "status": "validating",
            "request_counts": {"total": 10, "completed": 0, "failed": 0},
        }

        status_response_1 = MagicMock()
        status_response_1.raise_for_status.return_value = None
        status_response_1.json.return_value = {
            "id": "batch-1",
            "status": "in_progress",
            "request_counts": {"total": 10, "completed": 4, "failed": 1},
        }

        status_response_2 = MagicMock()
        status_response_2.raise_for_status.return_value = None
        status_response_2.json.return_value = {
            "id": "batch-1",
            "status": "completed",
            "request_counts": {"total": 10, "completed": 9, "failed": 1},
            "output_file_id": "file-output-1",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            batch_input_path = Path(tmp_dir) / "batch_input.jsonl"
            batch_input_path.write_text("{}", encoding="utf-8")
            status_messages: list[str] = []
            provider = OpenAIChatProvider(api_key="test-key", model="test-model")

            with (
                patch(
                    "guardian.model_provider.requests.post",
                    side_effect=[upload_response, create_response],
                ),
                patch(
                    "guardian.model_provider.requests.get",
                    side_effect=[status_response_1, status_response_2],
                ),
                patch(
                    "guardian.model_provider.time.perf_counter",
                    side_effect=[100.0, 120.0, 140.0],
                ),
                patch.object(
                    provider,
                    "_download_file",
                    return_value=output_dir / "openai_batch_output.jsonl",
                ),
            ):
                provider.execute_batch(
                    batch_input_path,
                    request_manifest_path=Path(tmp_dir) / "batch_request_manifest.jsonl",
                    output_dir=output_dir,
                    completion_window="24h",
                    poll_interval_seconds=0.0,
                    status_callback=status_messages.append,
                )

        self.assertTrue(any("eta=20s" in message for message in status_messages))

    def test_loads_api_settings_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / ".env").write_text(
                "OPENAI_API_KEY=test-key\nOPENAI_MODEL=test-model\nOPENAI_REASONING_EFFORT=low\n",
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
            self.assertEqual(provider.reasoning_effort, "low")

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
        self.assertNotIn("tool_choice", request_payload)

    def test_includes_reasoning_effort_in_generate_payload_when_configured(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": "{\"case_id\": \"c1\"}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }

        with patch("guardian.model_provider.requests.post", return_value=response) as post:
            provider = OpenAIChatProvider(
                api_key="test-key",
                model="gpt-5.4",
                reasoning_effort="low",
            )
            provider.generate(
                prompt="{}",
                system_prompt="Return JSON only.",
                response_format={"type": "json_object"},
                metadata={"case_id": "c1"},
            )

        request_payload = json.loads(post.call_args.kwargs["data"].decode("utf-8"))
        self.assertEqual(request_payload["reasoning"], {"effort": "low"})

    def test_includes_reasoning_effort_in_batch_payload_when_configured(self) -> None:
        provider = OpenAIChatProvider(
            api_key="test-key",
            model="gpt-5.4",
            reasoning_effort="low",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "batch.jsonl"
            with output_path.open("w", encoding="utf-8") as handle:
                provider.write_batch_request(
                    handle,
                    custom_id="case-1",
                    prompt="{}",
                    system_prompt="Return JSON only.",
                    response_format={"type": "json_object"},
                    metadata={"case_id": "c1"},
                )

            batch_record = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(batch_record["body"]["reasoning"], {"effort": "low"})

    def test_rejects_invalid_reasoning_effort(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "OPENAI_REASONING_EFFORT"):
            OpenAIChatProvider(
                api_key="test-key",
                model="gpt-5.4",
                reasoning_effort="fast",
            )

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
            with self.assertRaisesRegex(RuntimeError, "does not configure tools"):
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

    def test_applies_ollama_keep_alive_and_context_length(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "{\"case_id\": \"c1\"}"},
            "prompt_eval_count": 11,
            "eval_count": 7,
        }

        with patch.dict(
            os.environ,
            {
                "OLLAMA_MODEL": "qwen3:8b",
                "OLLAMA_KEEP_ALIVE": "30m",
                "OLLAMA_CONTEXT_LENGTH": "4096",
            },
            clear=True,
        ):
            with patch("guardian.model_provider.requests.post", return_value=response) as post:
                provider = OllamaChatProvider()
                provider.generate(
                    prompt="{}",
                    system_prompt="Return JSON only.",
                    response_format={"type": "json_object"},
                    metadata={"case_id": "c1"},
                )

        request_payload = post.call_args.kwargs["json"]
        self.assertEqual(request_payload["keep_alive"], "30m")
        self.assertEqual(request_payload["options"], {"num_ctx": 4096})


class StaticResponseProviderBatchTests(unittest.TestCase):
    def test_execute_batch_emits_status_callbacks(self) -> None:
        provider = StaticResponseProvider(lambda metadata: {"case_id": metadata["case_id"]})
        status_messages: list[str] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            batch_input_path = root / "batch_input.jsonl"
            request_manifest_path = root / "batch_request_manifest.jsonl"
            output_dir = root / "outputs"
            batch_input_path.write_text(
                json.dumps({"custom_id": "rf_000000000", "body": {}}) + "\n",
                encoding="utf-8",
            )
            request_manifest_path.write_text(
                json.dumps(
                    {
                        "custom_id": "rf_000000000",
                        "metadata": {"case_id": "case-1", "model": "static-model"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            provider.execute_batch(
                batch_input_path,
                request_manifest_path=request_manifest_path,
                output_dir=output_dir,
                completion_window="24h",
                poll_interval_seconds=0.0,
                status_callback=status_messages.append,
            )

        self.assertGreaterEqual(len(status_messages), 2)
        self.assertIn("Processing static batch input file", status_messages[0])
        self.assertIn("Static batch completed with 1/1 successful requests.", status_messages[-1])


if __name__ == "__main__":
    unittest.main()
