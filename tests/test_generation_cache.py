import json
import tempfile
import unittest
from pathlib import Path

from guardian.generation_cache import (
    CachedModelProvider,
    GenerationCache,
    build_generation_request_spec,
    generation_request_key,
)
from guardian.model_provider import StaticResponseProvider
from lib.utils import iter_jsonl


class GenerationCacheTests(unittest.TestCase):
    def test_request_key_binds_model_digest_prompt_and_settings(self) -> None:
        common = {
            "provider": "ollama",
            "model": "qwen3:30b",
            "model_digest": "sha256:model-v1",
            "inference_settings": {"temperature": 0},
            "prompt": "case",
            "system_prompt": "system",
            "response_format": {"type": "json_object"},
        }
        first = generation_request_key(build_generation_request_spec(**common))
        changed = generation_request_key(
            build_generation_request_spec(**{**common, "model_digest": "sha256:model-v2"})
        )
        self.assertNotEqual(first, changed)

    def test_sync_provider_reuses_exact_generation(self) -> None:
        calls = 0

        def resolver(metadata: dict) -> dict:
            nonlocal calls
            calls += 1
            return {"case_id": metadata["case_id"], "answer": "cached"}

        with tempfile.TemporaryDirectory() as temporary:
            cache_path = Path(temporary) / "generations.sqlite"
            provider = CachedModelProvider(
                StaticResponseProvider(resolver),
                cache_path=cache_path,
                model_digest="sha256:static-v1",
                inference_settings={"temperature": 0},
            )
            first = provider.generate(
                "prompt", "system", {"type": "json_object"}, {"case_id": "c1"}
            )
            second = provider.generate(
                "prompt", "system", {"type": "json_object"}, {"case_id": "c1"}
            )

            self.assertEqual(calls, 1)
            self.assertEqual(first[1], second[1])
            self.assertTrue(second[2]["generation_cache_hit"])
            self.assertEqual(second[2]["total_tokens"], 0)
            self.assertEqual(provider.stats, {"hits": 1, "misses": 1, "stores": 1})

    def test_batch_provider_can_execute_entirely_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "generations.sqlite"
            metadata = {"case_id": "c1", "model": "static-model"}
            first_delegate = StaticResponseProvider(
                lambda request: {"case_id": request["case_id"], "answer": "cached"}
            )
            first = CachedModelProvider(
                first_delegate,
                cache_path=cache_path,
                model_digest="sha256:static-v1",
                inference_settings={},
            )
            first_input = root / "first_input.jsonl"
            first_manifest = root / "first_manifest.jsonl"
            with first_input.open("w", encoding="utf-8") as handle:
                first.write_batch_request(
                    handle,
                    custom_id="r1",
                    prompt="prompt",
                    system_prompt="system",
                    response_format={"type": "json_object"},
                    metadata=metadata,
                )
            first_manifest.write_text(
                json.dumps({"custom_id": "r1", "metadata": metadata}) + "\n",
                encoding="utf-8",
            )
            first_result = first.execute_batch(
                first_input,
                request_manifest_path=first_manifest,
                output_dir=root / "first_output",
                completion_window="24h",
                poll_interval_seconds=0,
            )
            first_row = next(iter_jsonl(first_result.output_path))
            _, parsed, _, error = first.parse_batch_result(first_row, metadata)
            self.assertIsNone(error)
            self.assertEqual(parsed["answer"], "cached")

            def should_not_run(_: dict) -> dict:
                raise AssertionError("provider was called for a cached batch request")

            second = CachedModelProvider(
                StaticResponseProvider(should_not_run),
                cache_path=cache_path,
                model_digest="sha256:static-v1",
                inference_settings={},
            )
            second_input = root / "second_input.jsonl"
            second_manifest = root / "second_manifest.jsonl"
            with second_input.open("w", encoding="utf-8") as handle:
                second.write_batch_request(
                    handle,
                    custom_id="r2",
                    prompt="prompt",
                    system_prompt="system",
                    response_format={"type": "json_object"},
                    metadata=metadata,
                )
            second_manifest.write_text(
                json.dumps({"custom_id": "r2", "metadata": metadata}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(second_input.read_text(encoding="utf-8"), "")
            second_result = second.execute_batch(
                second_input,
                request_manifest_path=second_manifest,
                output_dir=root / "second_output",
                completion_window="24h",
                poll_interval_seconds=0,
            )
            second_row = next(iter_jsonl(second_result.output_path))
            _, parsed, usage, error = second.parse_batch_result(second_row, metadata)
            self.assertIsNone(error)
            self.assertEqual(parsed["answer"], "cached")
            self.assertTrue(usage["generation_cache_hit"])
            self.assertEqual(second_result.batch["provider_request_count"], 0)

    def test_cache_is_append_only_for_existing_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = GenerationCache(Path(temporary) / "cache.sqlite")
            specification = build_generation_request_spec(
                provider="ollama",
                model="m",
                model_digest="sha256:d",
                inference_settings={},
                prompt="p",
                system_prompt="s",
                response_format={},
            )
            self.assertTrue(
                cache.put(specification, raw_response={"v": 1}, parsed_payload={"v": 1}, usage={})
            )
            self.assertFalse(
                cache.put(specification, raw_response={"v": 2}, parsed_payload={"v": 2}, usage={})
            )
            self.assertEqual(cache.get(specification)["raw_response"], {"v": 1})


if __name__ == "__main__":
    unittest.main()
