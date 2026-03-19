import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from guardian.model_provider import BatchExecutionResult, StaticResponseProvider
from guardian.prompts import get_prompt_template
from guardian.reasoning import (
    _disable_generation_progress,
    _collect_selected_records_in_order,
    build_prompt_bundle,
    build_track_diagnosis_prompt_bundle,
    run_reasoning_floor,
)


def _extract_input_case(prompt: str) -> dict[str, Any]:
    marker = "Input case:\n"
    _, payload = prompt.split(marker, 1)
    return json.loads(payload)


class CostedStaticOpenAIProvider(StaticResponseProvider):
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        raw, parsed, usage = super().generate(prompt, system_prompt, response_format, metadata)
        usage["estimated_cost_usd"] = 1.0
        usage["input_cost_per_1m_tokens_usd"] = 2.0
        usage["output_cost_per_1m_tokens_usd"] = 4.0
        return raw, parsed, usage

    def parse_batch_result(
        self,
        result_record: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any], str | None]:
        raw, parsed, usage, error = super().parse_batch_result(result_record, metadata)
        usage["estimated_cost_usd"] = 1.0
        usage["input_cost_per_1m_tokens_usd"] = 2.0
        usage["output_cost_per_1m_tokens_usd"] = 4.0
        return raw, parsed, usage, error


class ReasoningFloorHelperTests(unittest.TestCase):
    def test_batch_generation_progress_is_disabled(self) -> None:
        self.assertTrue(_disable_generation_progress(execution_mode="batch", total_requests=1))
        self.assertFalse(_disable_generation_progress(execution_mode="sync", total_requests=1))
        self.assertTrue(_disable_generation_progress(execution_mode="parallel", total_requests=0))


class RetryableBatchFailureOpenAIProvider(CostedStaticOpenAIProvider):
    def _iter_jsonl_rows(self, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _usage_payload(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt_tokens": metadata.get("prompt_tokens", 0),
            "completion_tokens": metadata.get("completion_tokens", 0),
            "total_tokens": metadata.get("total_tokens", 0),
            "cached_tokens": metadata.get("cached_tokens"),
            "estimated_cost_usd": 1.0,
            "input_cost_per_1m_tokens_usd": 2.0,
            "output_cost_per_1m_tokens_usd": 4.0,
            "model": metadata.get("model", self.model),
            "provider": self.provider_name,
            "request_metadata": metadata,
        }

    def execute_batch(
        self,
        batch_input_path: Path,
        *,
        request_manifest_path: Path,
        output_dir: Path,
        completion_window: str,
        poll_interval_seconds: float,
        status_callback: Callable[[str], None] | None = None,
    ) -> BatchExecutionResult:
        del completion_window, poll_interval_seconds
        if not hasattr(self, "_failed_batch_custom_ids"):
            self._failed_batch_custom_ids: set[str] = set()
        output_dir.mkdir(parents=True, exist_ok=True)
        if status_callback is not None:
            status_callback(f"Processing retry-test batch input file {batch_input_path.name}.")

        manifest_by_custom_id = {
            row["custom_id"]: row
            for row in self._iter_jsonl_rows(request_manifest_path)
            if isinstance(row.get("custom_id"), str)
        }
        output_path = output_dir / "static_batch_output.jsonl"
        error_path = output_dir / "static_batch_errors.jsonl"
        completed = 0
        total = 0
        has_errors = False

        with open(output_path, "w", encoding="utf-8") as output_fh, open(error_path, "w", encoding="utf-8") as error_fh:
            for request_row in self._iter_jsonl_rows(batch_input_path):
                total += 1
                custom_id = request_row.get("custom_id")
                metadata = dict((manifest_by_custom_id.get(custom_id) or {}).get("metadata") or {})
                if (
                    isinstance(custom_id, str)
                    and metadata.get("task_type") == "proposal"
                    and custom_id not in self._failed_batch_custom_ids
                ):
                    self._failed_batch_custom_ids.add(custom_id)
                    has_errors = True
                    error_fh.write(
                        json.dumps(
                            {
                                "custom_id": custom_id,
                                "response": {
                                    "status_code": 500,
                                    "body": {
                                        "error": {
                                            "message": "The server had an error while processing your request. Sorry about that!"
                                        }
                                    },
                                },
                                "error": None,
                            }
                        )
                        + "\n"
                    )
                    continue

                payload = self.resolver(dict(metadata))
                response_body = {
                    "id": f"chatcmpl-{custom_id}",
                    "object": "chat.completion",
                    "model": metadata.get("model", self.model),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": json.dumps(payload)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": metadata.get("prompt_tokens", 0),
                        "completion_tokens": metadata.get("completion_tokens", 0),
                        "total_tokens": metadata.get("total_tokens", 0),
                    },
                }
                output_fh.write(
                    json.dumps(
                        {
                            "custom_id": custom_id,
                            "response": {"status_code": 200, "body": response_body},
                            "error": None,
                        }
                    )
                    + "\n"
                )
                completed += 1

        if not has_errors:
            error_path.unlink()
            error_path_result = None
        else:
            error_path_result = error_path

        batch_payload = {
            "id": "retry-test-batch",
            "status": "completed",
            "request_counts": {"total": total, "completed": completed, "failed": total - completed},
        }
        (output_dir / "static_batch_job.json").write_text(json.dumps(batch_payload, indent=2), encoding="utf-8")
        if status_callback is not None:
            status_callback(f"Retry-test batch completed with {completed}/{total} successful requests.")
        return BatchExecutionResult(batch=batch_payload, output_path=output_path, error_path=error_path_result)

    def parse_batch_result(
        self,
        result_record: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any], str | None]:
        response_block = result_record.get("response")
        response_body = response_block.get("body") if isinstance(response_block, dict) else None
        status_code = response_block.get("status_code") if isinstance(response_block, dict) else None
        if isinstance(status_code, int) and status_code >= 400:
            body_error = ((response_body.get("error") or {}).get("message")) if isinstance(response_body, dict) else None
            return (
                response_body if isinstance(response_body, dict) else result_record,
                None,
                self._usage_payload(metadata),
                body_error or f"Batch request failed with status {status_code}.",
            )
        return super().parse_batch_result(result_record, metadata)


class CountingStaticResponseProvider(StaticResponseProvider):
    def __init__(self, resolver: Callable[[dict[str, Any]], dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(resolver, **kwargs)
        self.generate_call_count = 0
        self.batch_execute_call_count = 0

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        self.generate_call_count += 1
        return super().generate(prompt, system_prompt, response_format, metadata)

    def execute_batch(
        self,
        batch_input_path: Path,
        *,
        request_manifest_path: Path,
        output_dir: Path,
        completion_window: str,
        poll_interval_seconds: float,
        status_callback: Callable[[str], None] | None = None,
    ) -> BatchExecutionResult:
        self.batch_execute_call_count += 1
        return super().execute_batch(
            batch_input_path,
            request_manifest_path=request_manifest_path,
            output_dir=output_dir,
            completion_window=completion_window,
            poll_interval_seconds=poll_interval_seconds,
            status_callback=status_callback,
        )


class FailingAfterNGenerateCallsProvider(CountingStaticResponseProvider):
    def __init__(
        self,
        resolver: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        fail_after: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(resolver, **kwargs)
        self.fail_after = fail_after

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        if self.generate_call_count >= self.fail_after:
            raise RuntimeError("intentional test failure")
        return super().generate(prompt, system_prompt, response_format, metadata)


class ConfiguredStaticOpenAIProvider(StaticResponseProvider):
    def __init__(self, resolver: Callable[[dict[str, Any]], dict[str, Any]], *, model: str, reasoning_effort: str | None) -> None:
        super().__init__(resolver, provider_name="openai", model=model)
        self.reasoning_effort = reasoning_effort


class ReasoningFloorTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _make_stub_fixture(self) -> tuple[Path, Path, Path, Path, Callable[[dict[str, Any]], dict[str, Any]]]:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        root = Path(tmp_dir.name)
        classified_path = root / "classified.jsonl"
        world_state_path = root / "world_state.json"
        selection_manifest_path = root / "selection.json"

        classified_rows = [
            {
                "id": "repair_case",
                "qid": "Q1",
                "property": "P31",
                "track": "A_BOX",
                "labels_en": {},
                "violation_context": {"value": ["Q2"]},
                "repair_target": {"action": "UPDATE", "old_value": ["Q2"], "new_value": ["Q5"]},
                "persistence_check": {},
                "popularity": {"score": 0.3},
                "classification": {"class": "TypeB", "subtype": "LOCAL_NEIGHBOR_IDS"},
            },
            {
                "id": "reform_case",
                "qid": "Q2",
                "property": "P31",
                "track": "T_BOX",
                "labels_en": {},
                "violation_context": {},
                "repair_target": {
                    "constraint_delta": {
                        "changed_constraint_types": ["Q21510859"],
                        "signature_after": [
                            {
                                "constraint_qid": "Q21510859",
                                "snaktype": "VALUE",
                                "rank": "normal",
                                "qualifiers": [{"property_id": "P2305", "values": ["Q5", "Q43229"]}],
                            }
                        ],
                    }
                },
                "persistence_check": {},
                "popularity": {"score": 0.9},
                "classification": {"class": "T_BOX", "subtype": "RELAXATION_SET_EXPANSION"},
            },
        ]
        self._write_jsonl(classified_path, classified_rows)
        with open(world_state_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "repair_case": {
                        "L1_ego_node": {
                            "qid": "Q1",
                            "label": "Entity",
                            "description": "desc",
                            "properties": {"P31": ["Q2"]},
                        },
                        "L2_labels": {},
                        "L3_neighborhood": {"outgoing_edges": []},
                        "L4_constraints": {"constraints": []},
                    },
                    "reform_case": {
                        "L1_ego_node": {
                            "qid": "Q2",
                            "label": "Entity",
                            "description": "desc",
                            "properties": {},
                        },
                        "L2_labels": {},
                        "L3_neighborhood": {"outgoing_edges": []},
                        "L4_constraints": {"constraints": []},
                        "constraint_change_context": {},
                    },
                },
                fh,
            )
        selection_manifest_path.write_text(
            json.dumps({"selected_case_ids": ["reform_case"]}),
            encoding="utf-8",
        )

        def resolver(metadata: dict[str, Any]) -> dict[str, Any]:
            if metadata["task_type"] == "track_diagnosis":
                return {
                    "case_id": metadata["case_id"],
                    "predicted_track": "T_BOX" if metadata["case_id"] == "reform_case" else "A_BOX",
                    "confidence": "high",
                }
            if metadata["case_id"] == "reform_case":
                return {
                    "case_id": "reform_case",
                    "target": {"pid": "P31", "constraint_type_qid": "Q21510859"},
                    "proposal": {
                        "action": "RELAXATION_SET_EXPANSION",
                        "signature_after": [
                            {
                                "constraint_qid": "Q21510859",
                                "snaktype": "VALUE",
                                "rank": "normal",
                                "qualifiers": [{"property_id": "P2305", "values": ["Q5", "Q43229"]}],
                            }
                        ],
                    },
                }
            return {
                "case_id": "repair_case",
                "target": {"qid": "Q1", "pid": "P31"},
                "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}],
            }

        return root, classified_path, world_state_path, selection_manifest_path, resolver

    def test_reasoning_floor_stub_run(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
        )
        self.assertIn("paper_summary", summary)
        self.assertIn("run_info", summary)
        self.assertIn("usage", summary)
        self.assertEqual(summary["counts"]["cases"], 1)
        self.assertEqual(summary["counts"]["track_diagnosis_exact_match"], 1)
        self.assertEqual(summary["run_info"]["model"], "stub-model")
        self.assertEqual(summary["run_info"]["execution_mode"], "sync")
        self.assertEqual(summary["usage"]["prompt_tokens"], 0)
        self.assertEqual(summary["usage"]["completion_tokens"], 0)
        self.assertIn("stub_model", summary["run_info"]["output_dir"])
        self.assertEqual(summary["inputs"]["selection_manifest"], str(selection_manifest_path))
        self.assertEqual(summary["run_info"]["evaluation"]["classified_record_strategy"], "memory_cache")
        self.assertIsNone(summary["run_info"]["evaluation"]["filtered_classified_path"])
        self.assertEqual(summary["parse_errors"]["proposal_parse_error_count"], 0)

    def test_reasoning_floor_streams_filtered_classified_subset_for_large_eval(self) -> None:
        root, classified_path, world_state_path, _selection_manifest_path, resolver = self._make_stub_fixture()
        with patch("guardian.reasoning.EVALUATION_IN_MEMORY_CASE_THRESHOLD", 1):
            summary = run_reasoning_floor(
                classified_path=classified_path,
                world_state_path=world_state_path,
                output_dir=root / "outputs",
                provider=StaticResponseProvider(resolver, model="stub-model"),
                ablation_bundles=["minimal_case"],
            )

        run_dir = Path(summary["run_info"]["output_dir"])
        filtered_path = run_dir / "selected_classified_records.jsonl"
        evaluation_summary = json.loads((run_dir / "minimal_case" / "evaluation_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["run_info"]["evaluation"]["classified_record_strategy"], "filtered_subset_stream")
        self.assertEqual(summary["run_info"]["evaluation"]["filtered_classified_path"], str(filtered_path))
        self.assertEqual(summary["run_info"]["evaluation"]["filtered_record_count"], 2)
        self.assertTrue(filtered_path.exists())
        filtered_rows = [json.loads(line) for line in filtered_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual([row["id"] for row in filtered_rows], ["repair_case", "reform_case"])
        self.assertEqual(evaluation_summary["inputs"]["classified_benchmark"], str(classified_path))

    def test_reasoning_floor_batch_stub_run(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            execution_mode="batch",
            batch_poll_interval_seconds=0.0,
        )
        run_dir = Path(summary["run_info"]["output_dir"])
        manifest_rows = [
            json.loads(line)
            for line in (run_dir / "run_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(summary["counts"]["cases"], 1)
        self.assertEqual(summary["counts"]["track_diagnosis_exact_match"], 1)
        self.assertEqual(summary["run_info"]["execution_mode"], "batch")
        self.assertEqual(summary["run_info"]["batch"]["mode"], "one_stage")
        self.assertEqual(summary["run_info"]["batch"]["overall_status"], "completed")
        self.assertEqual(summary["run_info"]["batch"]["phases"]["combined"]["status"], "completed")
        self.assertTrue((run_dir / "batch_input.jsonl").exists())
        self.assertTrue((run_dir / "batch_request_manifest.jsonl").exists())
        self.assertTrue((run_dir / "static_batch_output.jsonl").exists())
        self.assertTrue(all(row.get("custom_id") for row in manifest_rows))
        self.assertEqual(summary["usage"]["prompt_tokens"], 0)
        self.assertEqual(summary["usage"]["completion_tokens"], 0)

    def test_reasoning_floor_parallel_stub_run(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            execution_mode="parallel",
            parallel_workers=2,
        )

        self.assertEqual(summary["counts"]["cases"], 1)
        self.assertEqual(summary["counts"]["track_diagnosis_exact_match"], 1)
        self.assertEqual(summary["run_info"]["execution_mode"], "parallel")
        self.assertEqual(summary["run_info"]["parallel"]["workers"], 1)
        self.assertEqual(summary["run_info"]["parallel"]["source"], "argument")
        self.assertEqual(summary["usage"]["prompt_tokens"], 0)
        self.assertEqual(summary["usage"]["completion_tokens"], 0)

    def test_reasoning_floor_defaults_to_batch_for_openai_provider(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, provider_name="openai", model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            batch_poll_interval_seconds=0.0,
        )
        run_dir = Path(summary["run_info"]["output_dir"])
        self.assertEqual(summary["run_info"]["execution_mode"], "batch")
        self.assertTrue((run_dir / "batch_input.jsonl").exists())
        self.assertTrue((run_dir / "batch_request_manifest.jsonl").exists())

    def test_reasoning_floor_batch_applies_openai_cost_discount(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=CostedStaticOpenAIProvider(resolver, provider_name="openai", model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            execution_mode="batch",
            batch_poll_interval_seconds=0.0,
        )
        self.assertTrue(summary["run_info"]["batch_mode_used"])
        self.assertTrue(summary["usage"]["batch_pricing_applied"])
        self.assertEqual(summary["usage"]["cost_estimation_mode"], "openai_batch_discount_applied")
        self.assertEqual(summary["usage"]["cost_estimation_multiplier"], 0.5)
        self.assertEqual(summary["usage"]["estimated_cost_usd"], 1.0)

    def test_reasoning_floor_batch_retries_retryable_openai_errors_synchronously(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=RetryableBatchFailureOpenAIProvider(resolver, provider_name="openai", model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            execution_mode="batch",
            batch_poll_interval_seconds=0.0,
        )
        run_dir = Path(summary["run_info"]["output_dir"])
        manifest_rows = [
            json.loads(line)
            for line in (run_dir / "run_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        proposal_row = next(row for row in manifest_rows if row.get("task_type") == "proposal")

        self.assertEqual(summary["run_info"]["batch"]["sync_retry_fallback"]["eligible"], 1)
        self.assertEqual(summary["run_info"]["batch"]["sync_retry_fallback"]["attempted"], 1)
        self.assertEqual(summary["run_info"]["batch"]["sync_retry_fallback"]["succeeded"], 1)
        self.assertEqual(summary["run_info"]["batch"]["sync_retry_fallback"]["failed"], 0)
        self.assertEqual(summary["run_info"]["batch"]["phases"]["combined"]["sync_retry_fallback"]["eligible"], 1)
        self.assertFalse(any(row.get("parse_status") == "request_error" for row in manifest_rows))
        self.assertEqual(proposal_row["parse_status"], "normalized")
        self.assertEqual(proposal_row["recovery"]["type"], "sync_retry_after_batch_error")
        self.assertEqual(proposal_row["recovery"]["phase"], "combined")
        self.assertEqual(proposal_row["recovery"]["batch_status_code"], 500)
        self.assertTrue(proposal_row["recovery"]["attempted"])
        self.assertTrue(proposal_row["recovery"]["succeeded"])
        self.assertEqual(summary["request_errors"]["proposal_request_error_count"], 0)
        self.assertEqual(summary["overall_metrics"]["proposal_request_error_count"], 0)
        self.assertEqual(summary["usage"]["estimated_cost_usd"], 1.5)
        self.assertTrue(summary["usage"]["batch_pricing_applied"])
        self.assertEqual(summary["usage"]["cost_estimation_mode"], "mixed")
        self.assertEqual(
            summary["usage"]["per_call_cost_estimation_modes"],
            ["openai_batch_discount_applied", "provider_default"],
        )
        self.assertEqual(summary["usage"]["per_call_cost_estimation_multipliers"], [0.5, 1.0])
        self.assertIsNone(summary["usage"]["cost_estimation_multiplier"])

    def test_reasoning_floor_resume_sync_run_only_executes_missing_cases(self) -> None:
        root, classified_path, world_state_path, _selection_manifest_path, resolver = self._make_stub_fixture()
        failing_provider = FailingAfterNGenerateCallsProvider(resolver, fail_after=2, model="stub-model")

        with self.assertRaisesRegex(RuntimeError, "intentional test failure"):
            run_reasoning_floor(
                classified_path=classified_path,
                world_state_path=world_state_path,
                output_dir=root / "outputs",
                provider=failing_provider,
                ablation_bundles=["minimal_case"],
            )

        run_dir = next((root / "outputs").iterdir())
        initial_manifest_rows = self._read_jsonl(run_dir / "run_manifest.jsonl")
        resumed_provider = CountingStaticResponseProvider(resolver, model="stub-model")

        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "ignored",
            resume_run_dir=run_dir,
            provider=resumed_provider,
            ablation_bundles=["minimal_case"],
        )

        manifest_rows = self._read_jsonl(run_dir / "run_manifest.jsonl")
        self.assertEqual(len(initial_manifest_rows), 2)
        self.assertEqual(len(manifest_rows), 4)
        self.assertEqual(resumed_provider.generate_call_count, 2)
        self.assertEqual(summary["counts"]["cases"], 2)
        self.assertTrue(summary["run_info"]["resume"]["enabled"])
        self.assertEqual(summary["run_info"]["resume"]["existing_manifest_rows"], 2)
        self.assertEqual(summary["run_info"]["output_dir"], str(run_dir))

    def test_reasoning_floor_records_openai_reasoning_effort_in_run_config(self) -> None:
        root, classified_path, world_state_path, _selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=ConfiguredStaticOpenAIProvider(resolver, model="stub-model", reasoning_effort="low"),
            ablation_bundles=["minimal_case"],
            execution_mode="sync",
        )

        run_dir = Path(summary["run_info"]["output_dir"])
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(run_config["openai_reasoning_effort"], "low")
        report = json.loads((run_dir / "reasoning_floor_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["run_info"]["openai_reasoning_effort"], "low")
        self.assertEqual(report["run_info"]["openai_reasoning_effort"], "low")
        self.assertEqual(report["inputs"]["openai_reasoning_effort"], "low")

    def test_reasoning_floor_resume_rejects_changed_openai_reasoning_effort(self) -> None:
        root, classified_path, world_state_path, _selection_manifest_path, resolver = self._make_stub_fixture()
        initial_summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=ConfiguredStaticOpenAIProvider(resolver, model="stub-model", reasoning_effort="low"),
            ablation_bundles=["minimal_case"],
            execution_mode="sync",
        )

        run_dir = Path(initial_summary["run_info"]["output_dir"])
        with self.assertRaisesRegex(ValueError, "openai_reasoning_effort"):
            run_reasoning_floor(
                classified_path=classified_path,
                world_state_path=world_state_path,
                output_dir=root / "ignored",
                resume_run_dir=run_dir,
                provider=ConfiguredStaticOpenAIProvider(resolver, model="stub-model", reasoning_effort="high"),
                ablation_bundles=["minimal_case"],
                execution_mode="sync",
            )

    def test_reasoning_floor_resume_diagnosis_routed_batch_only_submits_missing_proposals(self) -> None:
        root, classified_path, world_state_path, _selection_manifest_path, resolver = self._make_stub_fixture()
        initial_summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            execution_mode="batch",
            batch_poll_interval_seconds=0.0,
            proposal_track_mode="diagnosis_routed",
        )

        run_dir = Path(initial_summary["run_info"]["output_dir"])
        manifest_path = run_dir / "run_manifest.jsonl"
        raw_path = run_dir / "raw_model_responses.jsonl"
        t_box_path = run_dir / "minimal_case" / "t_box_proposals.jsonl"
        self._write_jsonl(
            manifest_path,
            [
                row
                for row in self._read_jsonl(manifest_path)
                if not (row["case_id"] == "reform_case" and row["task_type"] == "proposal")
            ],
        )
        self._write_jsonl(
            raw_path,
            [
                row
                for row in self._read_jsonl(raw_path)
                if not (row["case_id"] == "reform_case" and row["task_type"] == "proposal")
            ],
        )
        self._write_jsonl(
            t_box_path,
            [row for row in self._read_jsonl(t_box_path) if row["case_id"] != "reform_case"],
        )

        resumed_provider = CountingStaticResponseProvider(resolver, model="stub-model")
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "ignored",
            resume_run_dir=run_dir,
            provider=resumed_provider,
            ablation_bundles=["minimal_case"],
            execution_mode="batch",
            batch_poll_interval_seconds=0.0,
            proposal_track_mode="diagnosis_routed",
        )

        proposal_manifest_rows = self._read_jsonl(run_dir / "proposal_batch_request_manifest.jsonl")
        manifest_rows = self._read_jsonl(manifest_path)
        self.assertEqual(resumed_provider.batch_execute_call_count, 1)
        self.assertEqual(len(proposal_manifest_rows), 1)
        self.assertEqual(proposal_manifest_rows[0]["metadata"]["case_id"], "reform_case")
        self.assertEqual(summary["counts"]["cases"], 2)
        self.assertTrue(summary["run_info"]["resume"]["enabled"])
        self.assertEqual(summary["run_info"]["resume"]["existing_manifest_rows"], 3)
        self.assertEqual(len(manifest_rows), 4)

    def test_prompt_bundles_use_named_templates(self) -> None:
        record = {
            "id": "repair_case",
            "qid": "Q1",
            "property": "P31",
            "track": "A_BOX",
            "classification": {"class": "TypeB"},
            "labels_en": {},
            "violation_context": {"value": ["Q2"]},
            "persistence_check": {},
        }
        world_state_entry = {"L4_constraints": {"constraints": []}}

        proposal_bundle = build_prompt_bundle(record, world_state_entry, "logic_only")
        diagnosis_bundle = build_track_diagnosis_prompt_bundle(record, world_state_entry, "logic_only")

        self.assertEqual(proposal_bundle.prompt_name, "reasoning_floor_a_box_zero_shot")
        self.assertEqual(
            proposal_bundle.system_prompt,
            get_prompt_template("reasoning_floor_a_box_zero_shot").system_prompt,
        )
        self.assertEqual(diagnosis_bundle.prompt_name, "reasoning_floor_track_diagnosis_zero_shot")
        self.assertEqual(
            diagnosis_bundle.system_prompt,
            get_prompt_template("reasoning_floor_track_diagnosis_zero_shot").system_prompt,
        )
        self.assertEqual(proposal_bundle.response_format, {"type": "json_object"})
        self.assertIn('"logic_context"', proposal_bundle.prompt)
        self.assertNotIn('"classification"', proposal_bundle.prompt)
        self.assertNotIn('"persistence_check"', proposal_bundle.prompt)
        self.assertNotIn('"repair_target"', proposal_bundle.prompt)
        self.assertNotIn('"track":', proposal_bundle.prompt)

    def test_t_box_prompt_bundle_prunes_local_graph_context(self) -> None:
        record = {
            "id": "reform_case",
            "qid": "Q2",
            "property": "P31",
            "track": "T_BOX",
            "classification": {"class": "T_BOX"},
            "labels_en": {},
            "violation_context": {
                "report_violation_type": "one-of",
                "report_violation_type_normalized": "one-of",
                "value": ["Q5"],
            },
            "persistence_check": {"truth_tokens": ["leak"]},
        }
        world_state_entry = {
            "L1_ego_node": {
                "qid": "Q2",
                "label": "Entity",
                "description": "desc",
                "properties": {"P31": ["Q5"], "P279": ["Q35120"]},
            },
            "L2_labels": {
                "entities": {
                    "Q2": {"label": "Entity"},
                    "Q5": {"label": "human"},
                    "Q35120": {"label": "entity"},
                    "Q999": {"label": "unrelated"},
                }
            },
            "L3_neighborhood": {
                "outgoing_edges": [
                    {"pid": "P279", "target": "Q35120"},
                    {"pid": "P31", "target": "Q5"},
                    {"pid": "P999", "target": "Q999"},
                ]
            },
            "L4_constraints": {
                "constraints": [
                    {
                        "constraint_type": {"qid": "Q21510859"},
                        "qualifiers": [{"property_id": "P2305", "values": ["Q5"]}],
                    },
                    {
                        "constraint_type": {"qid": "Q21503250"},
                        "qualifiers": [{"property_id": "P2305", "values": ["Q999"]}],
                    },
                ]
            },
        }

        proposal_bundle = build_prompt_bundle(record, world_state_entry, "local_graph")
        diagnosis_bundle = build_track_diagnosis_prompt_bundle(record, world_state_entry, "local_graph")

        self.assertIn('"local_context"', proposal_bundle.prompt)
        self.assertIn('"L4_constraints"', proposal_bundle.prompt)
        self.assertIn('"L2_labels"', proposal_bundle.prompt)
        self.assertIn('"L3_neighborhood"', proposal_bundle.prompt)
        self.assertIn('"P31"', proposal_bundle.prompt)
        self.assertNotIn('"P999"', proposal_bundle.prompt)
        self.assertNotIn('"Q999"', proposal_bundle.prompt)
        self.assertNotIn('"persistence_check"', proposal_bundle.prompt)
        self.assertIn('"L2_labels"', diagnosis_bundle.prompt)

    def test_a_box_local_graph_prompt_uses_pre_repair_target_state(self) -> None:
        record = {
            "id": "repair_case",
            "qid": "Q1",
            "property": "P31",
            "track": "A_BOX",
            "classification": {"class": "TypeC", "subtype": "EXTERNAL"},
            "labels_en": {},
            "violation_context": {
                "value": ["Q_OLD"],
                "value_labels_en": ["Old value"],
                "value_descriptions_en": ["Old description"],
            },
            "repair_target": {
                "kind": "A_BOX",
                "action": "UPDATE",
                "old_value": ["Q_OLD"],
                "old_value_labels_en": ["Old value"],
                "old_value_descriptions_en": ["Old description"],
                "new_value": ["Q_NEW"],
            },
        }
        world_state_entry = {
            "L1_ego_node": {
                "qid": "Q1",
                "label": "Entity",
                "description": "desc",
                "sitelinks_count": 1,
                "properties": {"P31": ["Q_NEW"]},
            },
            "L2_labels": {
                "entities": {
                    "Q1": {"label": "Entity"},
                    "P31": {"label": "instance of"},
                    "Q_NEW": {"label": "New value", "description": "Current value"},
                }
            },
            "L3_neighborhood": {
                "outgoing_edges": [
                    {
                        "property_id": "P31",
                        "target_qid": "Q_NEW",
                        "target_label": "New value",
                        "target_description": "Current value",
                    }
                ]
            },
            "L4_constraints": {"constraints": []},
        }

        proposal_payload = _extract_input_case(build_prompt_bundle(record, world_state_entry, "local_graph").prompt)
        diagnosis_payload = _extract_input_case(
            build_track_diagnosis_prompt_bundle(record, world_state_entry, "local_graph").prompt
        )

        for payload in (proposal_payload, diagnosis_payload):
            local_context = payload["local_context"]
            self.assertEqual(local_context["L1_ego_node"]["properties"], {"P31": ["Q_OLD"]})
            self.assertEqual(local_context["L3_neighborhood"]["outgoing_edges"], [])
            self.assertEqual(local_context["L2_labels"]["entities"]["Q_OLD"]["label"], "Old value")
            self.assertEqual(local_context["L2_labels"]["entities"]["Q_OLD"]["description"], "Old description")
            self.assertNotIn("Q_NEW", local_context["L2_labels"]["entities"])

    def test_logic_only_prompt_uses_pre_repair_target_values_for_constraint_pruning(self) -> None:
        record = {
            "id": "repair_case",
            "qid": "Q1",
            "property": "P31",
            "track": "A_BOX",
            "classification": {"class": "TypeC", "subtype": "EXTERNAL"},
            "labels_en": {},
            "violation_context": {"value": ["Q_OLD"]},
            "repair_target": {
                "kind": "A_BOX",
                "action": "UPDATE",
                "old_value": ["Q_OLD"],
                "new_value": ["Q_NEW"],
            },
        }
        world_state_entry = {
            "L4_constraints": {
                "constraints": [
                    {
                        "constraint_type": {"qid": "Q_KEEP"},
                        "qualifiers": [{"property_id": "P2305", "values": ["Q_OLD"]}],
                    },
                    {
                        "constraint_type": {"qid": "Q_DROP"},
                        "qualifiers": [{"property_id": "P2305", "values": ["Q_NEW"]}],
                    },
                ]
            }
        }

        proposal_payload = _extract_input_case(build_prompt_bundle(record, world_state_entry, "logic_only").prompt)
        diagnosis_payload = _extract_input_case(
            build_track_diagnosis_prompt_bundle(record, world_state_entry, "logic_only").prompt
        )

        for payload in (proposal_payload, diagnosis_payload):
            kept_qids = [
                constraint["constraint_type"]["qid"]
                for constraint in payload["logic_context"]["constraints"]
            ]
            self.assertEqual(kept_qids, ["Q_KEEP"])

    def test_t_box_local_graph_prompt_uses_violation_value_fallback_for_target_state(self) -> None:
        record = {
            "id": "reform_case",
            "qid": "Q2",
            "property": "P31",
            "track": "T_BOX",
            "classification": {"class": "T_BOX"},
            "labels_en": {},
            "violation_context": {
                "value": ["Q_HIST"],
                "value_labels_en": ["Historical value"],
                "value_descriptions_en": ["Historical description"],
            },
            "repair_target": {
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q21510859"],
                    "signature_before": [],
                    "signature_after": [],
                },
            },
        }
        world_state_entry = {
            "L1_ego_node": {
                "qid": "Q2",
                "label": "Entity",
                "description": "desc",
                "sitelinks_count": 1,
                "properties": {"P31": ["Q_CURR"]},
            },
            "L2_labels": {
                "entities": {
                    "Q2": {"label": "Entity"},
                    "P31": {"label": "instance of"},
                    "Q_CURR": {"label": "Current value"},
                }
            },
            "L3_neighborhood": {
                "outgoing_edges": [
                    {
                        "property_id": "P31",
                        "target_qid": "Q_CURR",
                        "target_label": "Current value",
                        "target_description": "Current description",
                    }
                ]
            },
            "L4_constraints": {
                "constraints": [
                    {
                        "constraint_type": {"qid": "Q21510859"},
                        "qualifiers": [{"property_id": "P2305", "values": ["Q_HIST"]}],
                    }
                ]
            },
        }

        proposal_payload = _extract_input_case(build_prompt_bundle(record, world_state_entry, "local_graph").prompt)
        diagnosis_payload = _extract_input_case(
            build_track_diagnosis_prompt_bundle(record, world_state_entry, "local_graph").prompt
        )

        for payload in (proposal_payload, diagnosis_payload):
            local_context = payload["local_context"]
            self.assertEqual(local_context["L1_ego_node"]["properties"], {"P31": ["Q_HIST"]})
            self.assertEqual(local_context["L3_neighborhood"]["outgoing_edges"], [])
            self.assertEqual(local_context["L2_labels"]["entities"]["Q_HIST"]["label"], "Historical value")
            self.assertNotIn("Q_CURR", local_context["L2_labels"]["entities"])

    def test_local_graph_prompt_omits_target_property_when_no_pre_repair_source(self) -> None:
        record = {
            "id": "repair_case",
            "qid": "Q1",
            "property": "P31",
            "track": "A_BOX",
            "classification": {"class": "TypeC", "subtype": "EXTERNAL"},
            "labels_en": {},
            "violation_context": {"report_violation_type": "Value type"},
            "repair_target": {"kind": "A_BOX", "action": "UPDATE", "new_value": ["Q_NEW"]},
        }
        world_state_entry = {
            "L1_ego_node": {
                "qid": "Q1",
                "label": "Entity",
                "description": "desc",
                "sitelinks_count": 1,
                "properties": {"P31": ["Q_NEW"]},
            },
            "L2_labels": {
                "entities": {
                    "Q1": {"label": "Entity"},
                    "P31": {"label": "instance of"},
                    "Q_NEW": {"label": "Current value"},
                }
            },
            "L3_neighborhood": {
                "outgoing_edges": [
                    {
                        "property_id": "P31",
                        "target_qid": "Q_NEW",
                        "target_label": "Current value",
                        "target_description": "Current description",
                    }
                ]
            },
            "L4_constraints": {"constraints": []},
        }

        proposal_payload = _extract_input_case(build_prompt_bundle(record, world_state_entry, "local_graph").prompt)

        local_context = proposal_payload["local_context"]
        self.assertNotIn("properties", local_context["L1_ego_node"])
        self.assertEqual(local_context["L3_neighborhood"]["outgoing_edges"], [])
        self.assertNotIn("Q_NEW", local_context["L2_labels"]["entities"])

    def test_t_box_prompt_template_avoids_specific_anchor_example(self) -> None:
        template = get_prompt_template("reasoning_floor_t_box_zero_shot")
        self.assertNotIn("Q21510859", template.user_prompt_template)
        self.assertNotIn("Q43229", template.user_prompt_template)
        self.assertNotIn("Q_CONSTRAINT_", template.user_prompt_template)
        self.assertNotIn("Q_ITEM_", template.user_prompt_template)
        self.assertIn("constraint-family QIDs from the supplied constraint context", template.user_prompt_template)
        self.assertIn("Do not copy violating entity or type QIDs into constraint_type_qid", template.user_prompt_template)

    def test_track_diagnosis_prompt_template_includes_schema_vs_claim_guidance(self) -> None:
        template = get_prompt_template("reasoning_floor_track_diagnosis_zero_shot")
        self.assertIn("Allowed-entity-types, property-scope, one-of, range", template.user_prompt_template)
        self.assertIn("If a property currently allows only certain entity types", template.user_prompt_template)
        self.assertIn("predict AMBIGUOUS", template.user_prompt_template)

    def test_cli_defaults_skip_minimal_case(self) -> None:
        module_path = Path(__file__).resolve().parents[1] / "src" / "reasoning_floor.py"
        spec = importlib.util.spec_from_file_location("reasoning_floor_cli_under_test", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.DEFAULT_ABLATION_BUNDLES, ("logic_only", "local_graph"))

    def test_reasoning_floor_preserves_manifest_order_before_max_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            world_state_path = root / "world_state.json"
            selection_manifest_path = root / "selection.json"

            classified_rows = [
                {
                    "id": "case_a",
                    "qid": "Q1",
                    "property": "P31",
                    "track": "A_BOX",
                    "labels_en": {},
                    "violation_context": {"value": ["Q2"]},
                    "repair_target": {"action": "UPDATE", "old_value": ["Q2"], "new_value": ["Q3"]},
                    "classification": {"class": "TypeA", "subtype": "DIRECT_VALUE"},
                },
                {
                    "id": "case_b",
                    "qid": "Q2",
                    "property": "P31",
                    "track": "A_BOX",
                    "labels_en": {},
                    "violation_context": {"value": ["Q4"]},
                    "repair_target": {"action": "UPDATE", "old_value": ["Q4"], "new_value": ["Q5"]},
                    "classification": {"class": "TypeA", "subtype": "DIRECT_VALUE"},
                },
                {
                    "id": "case_c",
                    "qid": "Q3",
                    "property": "P31",
                    "track": "A_BOX",
                    "labels_en": {},
                    "violation_context": {"value": ["Q6"]},
                    "repair_target": {"action": "UPDATE", "old_value": ["Q6"], "new_value": ["Q7"]},
                    "classification": {"class": "TypeA", "subtype": "DIRECT_VALUE"},
                },
            ]
            self._write_jsonl(classified_path, classified_rows)
            world_state_path.write_text(
                json.dumps(
                    {
                        row["id"]: {
                            "L1_ego_node": {"qid": row["qid"], "properties": {"P31": row["violation_context"]["value"]}},
                            "L4_constraints": {"constraints": []},
                        }
                        for row in classified_rows
                    }
                ),
                encoding="utf-8",
            )
            selection_manifest_path.write_text(
                json.dumps({"selected_case_ids": ["case_c", "case_a", "case_b"]}),
                encoding="utf-8",
            )

            def resolver(metadata: dict[str, Any]) -> dict[str, Any]:
                if metadata["task_type"] == "track_diagnosis":
                    return {"case_id": metadata["case_id"], "predicted_track": "A_BOX", "confidence": "high"}
                return {
                    "case_id": metadata["case_id"],
                    "target": {"qid": {"case_a": "Q1", "case_b": "Q2", "case_c": "Q3"}[metadata["case_id"]], "pid": "P31"},
                    "ops": [{"op": "SET", "pid": "P31", "value": {"case_a": "Q3", "case_b": "Q5", "case_c": "Q7"}[metadata["case_id"]]}],
                }

            summary = run_reasoning_floor(
                classified_path=classified_path,
                world_state_path=world_state_path,
                output_dir=root / "outputs",
                provider=StaticResponseProvider(resolver, model="stub-model"),
                ablation_bundles=["minimal_case"],
                selection_manifest_path=selection_manifest_path,
                max_cases=2,
            )

            run_dir = Path(summary["run_info"]["output_dir"])
            manifest_rows = [
                json.loads(line)
                for line in (run_dir / "run_manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            proposal_case_ids = [row["case_id"] for row in manifest_rows if row["task_type"] == "proposal"]
            self.assertEqual(proposal_case_ids, ["case_c", "case_a"])

    def test_collect_selected_records_limits_manifest_scan_before_reading_tail_records(self) -> None:
        rows = [
            {"id": "case_a", "track": "A_BOX"},
            {"id": "case_c", "track": "A_BOX"},
            {"id": "tail_1", "track": "A_BOX"},
            {"id": "tail_2", "track": "A_BOX"},
            {"id": "case_b", "track": "A_BOX"},
        ]
        yielded_ids: list[str] = []

        def fake_iter_jsonl(_path: str | Path) -> Any:
            for row in rows:
                yielded_ids.append(row["id"])
                yield row

        with patch("guardian.reasoning.iter_jsonl", side_effect=fake_iter_jsonl):
            selected = _collect_selected_records_in_order(
                "ignored.jsonl",
                case_ids=["case_c", "case_a", "case_b"],
                max_cases=2,
            )

        self.assertEqual([row["id"] for row in selected], ["case_c", "case_a"])
        self.assertEqual(yielded_ids, ["case_a", "case_c"])

    def test_reasoning_floor_diagnosis_routed_skips_ambiguous_proposals(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, _resolver = self._make_stub_fixture()

        def resolver(metadata: dict[str, Any]) -> dict[str, Any]:
            if metadata["task_type"] == "track_diagnosis":
                return {
                    "case_id": metadata["case_id"],
                    "predicted_track": "AMBIGUOUS" if metadata["case_id"] == "reform_case" else "A_BOX",
                    "confidence": "medium",
                }
            return {
                "case_id": metadata["case_id"],
                "target": {"qid": "Q1", "pid": "P31"},
                "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}],
            }

        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            proposal_track_mode="diagnosis_routed",
        )

        run_dir = Path(summary["run_info"]["output_dir"])
        manifest_rows = [
            json.loads(line)
            for line in (run_dir / "run_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        proposal_row = next(row for row in manifest_rows if row["case_id"] == "reform_case" and row["task_type"] == "proposal")
        self.assertEqual(summary["run_info"]["proposal_track_mode"], "diagnosis_routed")
        self.assertEqual(proposal_row["parse_status"], "skipped_ambiguous_track")
        self.assertEqual(proposal_row["proposal_track_used"], "AMBIGUOUS")
        t_box_rows = [
            json.loads(line)
            for line in (run_dir / "minimal_case" / "t_box_proposals.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(t_box_rows, [])

    def test_reasoning_floor_diagnosis_routed_parallel_stub_run(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            execution_mode="parallel",
            parallel_workers=2,
            proposal_track_mode="diagnosis_routed",
        )

        self.assertEqual(summary["counts"]["cases"], 1)
        self.assertEqual(summary["run_info"]["execution_mode"], "parallel")
        self.assertEqual(summary["run_info"]["proposal_track_mode"], "diagnosis_routed")
        self.assertEqual(summary["counts"]["track_diagnosis_exact_match"], 1)

    def test_reasoning_floor_diagnosis_routed_batch_uses_two_stage_artifacts(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, resolver = self._make_stub_fixture()
        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
            execution_mode="batch",
            batch_poll_interval_seconds=0.0,
            proposal_track_mode="diagnosis_routed",
        )

        run_dir = Path(summary["run_info"]["output_dir"])
        self.assertEqual(summary["run_info"]["batch"]["mode"], "two_stage")
        self.assertEqual(summary["run_info"]["batch"]["overall_status"], "completed")
        self.assertIn("diagnosis", summary["run_info"]["batch"]["phases"])
        self.assertIn("proposal", summary["run_info"]["batch"]["phases"])
        self.assertTrue((run_dir / "diagnosis_batch_input.jsonl").exists())
        self.assertTrue((run_dir / "proposal_batch_input.jsonl").exists())
        self.assertTrue((run_dir / "diagnosis_static_batch_output.jsonl").exists())
        self.assertTrue((run_dir / "proposal_static_batch_output.jsonl").exists())

    def test_reasoning_floor_normalizes_non_list_provenance_outputs(self) -> None:
        root, classified_path, world_state_path, _selection_manifest_path, _resolver = self._make_stub_fixture()

        def resolver(metadata: dict[str, Any]) -> dict[str, Any]:
            if metadata["task_type"] == "track_diagnosis":
                return {
                    "case_id": metadata["case_id"],
                    "predicted_track": "T_BOX" if metadata["case_id"] == "reform_case" else "A_BOX",
                    "confidence": "high",
                }
            if metadata["case_id"] == "reform_case":
                return {
                    "case_id": "reform_case",
                    "target": {"pid": "P31", "constraint_type_qid": "Q21510859"},
                    "proposal": {
                        "action": "RELAXATION_SET_EXPANSION",
                        "signature_after": [
                            {
                                "constraint_qid": "Q21510859",
                                "snaktype": "VALUE",
                                "rank": "normal",
                                "qualifiers": [{"property_id": "P2305", "values": ["Q5", "Q43229"]}],
                            }
                        ],
                    },
                    "provenance": {"node_id": "Q21510859", "snippet": "constraint"},
                }
            return {
                "case_id": "repair_case",
                "target": {"qid": "Q1", "pid": "P31"},
                "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}],
                "provenance": "historical statement",
            }

        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
        )

        run_dir = Path(summary["run_info"]["output_dir"])
        a_box_rows = [
            json.loads(line)
            for line in (run_dir / "minimal_case" / "a_box_proposals.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        t_box_rows = [
            json.loads(line)
            for line in (run_dir / "minimal_case" / "t_box_proposals.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(summary["parse_errors"]["proposal_parse_error_count"], 0)
        self.assertEqual(a_box_rows[0]["provenance"], [{"kind": "OTHER", "snippet": "historical statement"}])
        self.assertEqual(t_box_rows[0]["provenance"], [{"kind": "KG", "node_id": "Q21510859", "snippet": "constraint"}])

    def test_reasoning_floor_rejects_invalid_t_box_constraint_family_qids(self) -> None:
        root, classified_path, world_state_path, selection_manifest_path, _resolver = self._make_stub_fixture()

        def resolver(metadata: dict[str, Any]) -> dict[str, Any]:
            if metadata["task_type"] == "track_diagnosis":
                return {
                    "case_id": metadata["case_id"],
                    "predicted_track": "T_BOX" if metadata["case_id"] == "reform_case" else "A_BOX",
                    "confidence": "high",
                }
            if metadata["case_id"] == "reform_case":
                return {
                    "case_id": "reform_case",
                    "target": {"pid": "P31", "constraint_type_qid": "Q11122"},
                    "proposal": {
                        "action": "RELAXATION_SET_EXPANSION",
                        "signature_after": [
                            {
                                "constraint_qid": "Q11122",
                                "snaktype": "VALUE",
                                "rank": "normal",
                                "qualifiers": [{"property_id": "P2305", "values": ["Q5", "Q43229"]}],
                            }
                        ],
                    },
                }
            return {
                "case_id": "repair_case",
                "target": {"qid": "Q1", "pid": "P31"},
                "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}],
            }

        summary = run_reasoning_floor(
            classified_path=classified_path,
            world_state_path=world_state_path,
            output_dir=root / "outputs",
            provider=StaticResponseProvider(resolver, model="stub-model"),
            ablation_bundles=["minimal_case"],
            selection_manifest_path=selection_manifest_path,
        )

        run_dir = Path(summary["run_info"]["output_dir"])
        manifest_rows = [
            json.loads(line)
            for line in (run_dir / "run_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        proposal_row = next(row for row in manifest_rows if row["case_id"] == "reform_case" and row["task_type"] == "proposal")
        t_box_rows = [
            json.loads(line)
            for line in (run_dir / "minimal_case" / "t_box_proposals.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(summary["parse_errors"]["proposal_parse_error_count"], 1)
        self.assertEqual(proposal_row["parse_status"], "parse_error")
        self.assertEqual(proposal_row["parser_error"], "invalid constraint_type_qid for T-box proposal")
        self.assertEqual(t_box_rows, [])


if __name__ == "__main__":
    unittest.main()
