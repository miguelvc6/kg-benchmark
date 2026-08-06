from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from guardian.generation_cache import GenerationCache
from kg_benchmark.matrix.workflow import (
    MatrixWorkflowError,
    _models_differ_only_by_azure_deployment,
    dry_run_matrix,
    execute_matrix,
    matrix_status,
    plan_matrix,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, relative: str) -> dict[str, Any]:
    result: dict[str, Any] = {"path": relative, "bytes": path.stat().st_size, "sha256": _sha(path)}
    if path.suffix == ".jsonl":
        result["records"] = len(path.read_text(encoding="utf-8").splitlines())
    return result


def _population(name: str, case_ids: list[str]) -> dict[str, Any]:
    return {
        "manifest_type": "evaluation_population",
        "manifest_version": 1,
        "name": name,
        "quotas": {"IC-L": len(case_ids), "IC-G": 0, "IC-E-elim": 0, "TBOX": 0},
        "case_count": len(case_ids),
        "selected_case_ids": case_ids,
        "selected_group_keys": [f"ABOX|Q{index + 1}|P31" for index in range(len(case_ids))],
        "support_groups_excluded": 3,
        "parent": None,
    }


class ExperimentMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        self.models_path = self.root / "paper" / "models.json"
        self.protocol_path = self.root / "paper" / "protocol.json"
        self.output_root = self.root / "runs" / "matrices"
        self.cache_path = self.root / "runs" / "generation-cache.sqlite"
        self._build_fixture()

    def _build_fixture(self) -> None:
        eval_rows = [
            {
                "id": "eval_a",
                "qid": "Q1",
                "property": "P31",
                "track": "A_BOX",
                "labels_en": {},
                "violation_context": {"value": ["Q2"]},
                "repair_target": {"action": "UPDATE", "old_value": ["Q2"], "new_value": ["Q3"]},
                "classification": {"class": "TypeA", "subtype": "DIRECT_VALUE"},
            },
            {
                "id": "eval_b",
                "qid": "Q4",
                "property": "P31",
                "track": "A_BOX",
                "labels_en": {},
                "violation_context": {"value": ["Q5"]},
                "repair_target": {"action": "UPDATE", "old_value": ["Q5"], "new_value": ["Q6"]},
                "classification": {"class": "TypeB", "subtype": "LOCAL_NEIGHBOR_IDS"},
            },
        ]
        support_rows = [
            {
                "id": "support_a",
                "qid": "Q10",
                "property": "P31",
                "track": "A_BOX",
                "labels_en": {},
                "violation_context": {"value": ["Q11"]},
                "repair_target": {"action": "UPDATE", "old_value": ["Q11"], "new_value": ["Q12"]},
                "classification": {"class": "TypeA", "subtype": "DIRECT_VALUE"},
            },
            {
                "id": "support_t",
                "qid": "Q20",
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
                                "qualifiers": [{"property_id": "P2305", "values": ["Q21"]}],
                            }
                        ],
                    }
                },
                "classification": {"class": "T_BOX", "subtype": "RELAXATION_SET_EXPANSION"},
            },
        ]
        cases_path = self.dataset / "cases.jsonl"
        world_path = self.dataset / "source" / "world-state.jsonl"
        support_path = self.dataset / "selections" / "support-bank.json"
        _write_jsonl(cases_path, [*eval_rows, *support_rows])
        _write_jsonl(
            world_path,
            [
                {
                    "id": row["id"],
                    "world_state": {
                        "L1_ego_node": {"qid": row["qid"], "properties": {"P31": []}},
                        "L2_labels": {},
                        "L3_neighborhood": {"outgoing_edges": []},
                        "L4_constraints": {"constraints": []},
                    },
                }
                for row in [*eval_rows, *support_rows]
            ],
        )
        _write_json(
            support_path,
            {
                "manifest_type": "few_shot_support_bank",
                "manifest_version": 1,
                "capacity_per_locus": 16,
                "support_sets": {
                    "a_box_repair": [
                        {"case_id": "support_a", "group_key": "ABOX|Q10|P31", "role": "A_BOX", "visible_example_id": "example_000001"}
                    ],
                    "t_box_repair": [
                        {"case_id": "support_t", "group_key": "TBOX|P31|r1", "role": "T_BOX", "visible_example_id": "example_000002"}
                    ],
                    "track_diagnosis": [
                        {"case_id": "support_a", "group_key": "ABOX|Q10|P31", "role": "A_BOX", "visible_example_id": "example_000003"}
                    ],
                },
            },
        )
        main_path = self.dataset / "selections" / "main-2.json"
        api_path = self.dataset / "selections" / "api-2.json"
        _write_json(main_path, _population("main-2", ["eval_a", "eval_b"]))
        _write_json(api_path, _population("api-2", ["eval_a", "eval_b"]))
        protocol = {
            "manifest_type": "paper_protocol",
            "tasks": {
                "repair_proposal": {"routing": "oracle"},
                "track_diagnosis": {"routes_proposals": False},
            },
            "conditions": {
                "prompt_regimes": ["zero_shot", "static_few_shot"],
                "context_bundles": ["logic_only", "local_graph"],
            },
            "few_shot": {
                "default_example_counts": {"a_box_repair": 1, "t_box_repair": 1, "track_diagnosis": 1}
            },
        }
        models = {
            "manifest_type": "paper_model_matrix",
            "manifest_version": 1,
            "status": "frozen",
            "request_dimensions": {
                "tasks": ["repair_proposal", "track_diagnosis"],
                "prompt_regimes": ["zero_shot", "static_few_shot"],
                "context_bundles": ["logic_only", "local_graph"],
            },
            "generation_identity": [
                "case_payload_sha256",
                "task",
                "rendered_prompt_sha256",
                "context_sha256",
                "model_revision",
                "inference_parameters_sha256",
            ],
            "models": [
                {
                    "model_id": "ollama_test",
                    "provider": "ollama",
                    "model": "test:1b",
                    "model_revision": "a" * 64,
                    "revision_status": "resolved",
                    "revision_resolution": "test",
                    "population": "main-2",
                    "execution_mode": "sync",
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 13,
                    "context_length": 4096,
                    "max_output_tokens": 512,
                    "ollama_think": "enabled",
                    "max_transport_retries": 2,
                    "tools_disabled": True,
                    "expected_calls": 16,
                },
                {
                    "model_id": "azure_test",
                    "provider": "azure",
                    "model": "gpt-test",
                    "deployment": "gpt-test-deployment",
                    "model_revision": "azure-snapshot-1",
                    "revision_status": "resolved",
                    "revision_resolution": "test",
                    "population": "api-2",
                    "execution_mode": "sync",
                    "reasoning_effort": "high",
                    "max_output_tokens": 512,
                    "max_transport_retries": 2,
                    "tools_disabled": True,
                    "expected_calls": 16,
                },
            ],
        }
        _write_json(self.protocol_path, protocol)
        _write_json(self.models_path, models)
        lock_path = self.dataset / "methodology" / "methodology.lock.json"
        _write_json(
            lock_path,
            {
                "manifest_type": "paper_methodology_lock",
                "freeze_scope_sha256": "f" * 64,
                "source_git_revision": "1" * 40,
                "files": {
                    "paper/models.json": _sha(self.models_path),
                    "paper/protocol.json": _sha(self.protocol_path),
                },
            },
        )
        artifacts = {
            "cases": _artifact(cases_path, "cases.jsonl"),
            "world_state": _artifact(world_path, "source/world-state.jsonl"),
            "support_bank": _artifact(support_path, "selections/support-bank.json"),
        }
        _write_json(
            self.dataset / "manifest.json",
            {
                "manifest_type": "kg_benchmark_dataset",
                "manifest_version": 2,
                "dataset_id": "wikidata-repair-eval-paper",
                "status": "final",
                "protocol": {"path": "methodology/protocol.json", "sha256": _sha(self.protocol_path)},
                "methodology": {
                    "lock": {"path": "methodology/methodology.lock.json", "sha256": _sha(lock_path)},
                    "freeze_scope_sha256": "f" * 64,
                    "source_git_revision": "1" * 40,
                },
                "artifacts": artifacts,
            },
        )

    def _plan(self, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
        result = plan_matrix(
            dataset_dir=self.dataset,
            models_path=self.models_path,
            protocol_path=self.protocol_path,
            output_root=self.output_root,
            schema_root=ROOT / "schemas",
            **kwargs,
        )
        return Path(result["matrix_dir"]), result["matrix"]

    def test_plan_and_dry_run_materialize_full_cross_product_without_calls(self) -> None:
        matrix_dir, matrix = self._plan()
        self.assertEqual(matrix["workload"]["logical_cells"], 16)
        self.assertEqual(matrix["workload"]["execution_groups"], 8)
        self.assertEqual(matrix["workload"]["expected_requests"], 32)
        report = dry_run_matrix(
            matrix_dir=matrix_dir,
            generation_cache_path=self.cache_path,
            schema_root=ROOT / "schemas",
        )
        self.assertTrue(report["no_provider_calls"])
        self.assertEqual(report["unique_new_requests"], 32)
        self.assertEqual(report["missing_revision_models"], [])

        cache = GenerationCache(self.cache_path)
        requests = [json.loads(line) for line in (matrix_dir / "requests.jsonl").read_text().splitlines()]
        for row in requests:
            cache.put(row["request_spec"], raw_response={}, parsed_payload={}, usage={})
        cached = dry_run_matrix(
            matrix_dir=matrix_dir,
            generation_cache_path=self.cache_path,
            schema_root=ROOT / "schemas",
        )
        self.assertEqual(cached["unique_cache_hits"], 32)
        self.assertEqual(cached["unique_new_requests"], 0)

    def test_nested_extension_only_schedules_generation_keys_absent_from_cache(self) -> None:
        parent = self.root / "parent.json"
        extension = self.root / "extension.json"
        _write_json(parent, _population("parent-1", ["eval_a"]))
        _write_json(extension, _population("extension-2", ["eval_a", "eval_b"]))
        matrix_dir, _ = self._plan(
            model_ids=["ollama_test"],
            population_paths=[parent, extension],
        )
        rows = [json.loads(line) for line in (matrix_dir / "requests.jsonl").read_text().splitlines()]
        cache = GenerationCache(self.cache_path)
        parent_rows = [row for row in rows if row["population"] == "parent-1"]
        for row in parent_rows:
            cache.put(row["request_spec"], raw_response={}, parsed_payload={}, usage={})
        report = dry_run_matrix(
            matrix_dir=matrix_dir,
            generation_cache_path=self.cache_path,
            schema_root=ROOT / "schemas",
        )
        self.assertEqual(report["expected_request_memberships"], 24)
        self.assertEqual(report["unique_resolved_requests"], 16)
        self.assertEqual(report["unique_cache_hits"], 8)
        self.assertEqual(report["unique_new_requests"], 8)
        existing = cache.existing_keys({row["request_key"] for row in rows})
        new_keys = {row["request_key"] for row in rows if row["request_key"] not in existing}
        self.assertTrue(new_keys.isdisjoint(existing))
        self.assertEqual({row["case_id"] for row in rows if row["request_key"] in new_keys}, {"eval_b"})

    def test_execute_enforces_provider_policy_and_resumes_per_group(self) -> None:
        matrix_dir, _ = self._plan()
        requests = [json.loads(line) for line in (matrix_dir / "requests.jsonl").read_text().splitlines()]
        calls: list[dict[str, Any]] = []

        def fake_executor(**kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            group_dir = Path(kwargs["resume_run_dir"])
            population = json.loads(Path(kwargs["selection_manifest_path"]).read_text())
            population_name = population["name"]
            model_name = kwargs["model_name"]
            context = kwargs["ablation_bundles"][0]
            regime = kwargs["prompt_regime"]
            selected = [
                row
                for row in requests
                if row["population"] == population_name
                and row["model"] == model_name
                and row["context_bundle"] == context
                and row["prompt_regime"] == regime
            ]
            cache = GenerationCache(Path(kwargs["generation_cache_path"]))
            for row in selected:
                cache.put(row["request_spec"], raw_response={}, parsed_payload={}, usage={})
            run_rows = [
                {
                    "case_id": row["case_id"],
                    "task_type": "proposal" if row["task"] == "repair_proposal" else "track_diagnosis",
                    "ablation_bundle": context,
                    "parse_status": "normalized",
                }
                for row in selected
            ]
            _write_json(group_dir / "run_config.json", {"model": model_name, "regime": regime})
            _write_jsonl(group_dir / "run_manifest.jsonl", run_rows)
            _write_json(group_dir / "reasoning_floor_summary.json", {"requests": len(run_rows)})
            return {"ok": True}

        result = execute_matrix(
            matrix_dir=matrix_dir,
            generation_cache_path=self.cache_path,
            repo_root=self.root,
            schema_root=ROOT / "schemas",
            methodology_check=lambda _: {"frozen": True},
            run_executor=fake_executor,
        )
        self.assertTrue(result["complete"])
        self.assertEqual(result["executed_groups"], 8)
        self.assertEqual(len(calls), 8)
        ollama = next(call for call in calls if call["model_endpoint"] == "ollama")
        azure = next(call for call in calls if call["model_endpoint"] == "azure")
        self.assertEqual(
            (ollama["execution_mode"], ollama["temperature"], ollama["seed"], ollama["ollama_think"]),
            ("sync", 0, 13, True),
        )
        self.assertEqual(
            (
                azure["execution_mode"],
                azure["reasoning_effort"],
                azure["max_retries"],
            ),
            ("sync", "high", 2),
        )

        cell_path = next((matrix_dir / "cells").glob("*.json"))
        cell_path.unlink()
        calls.clear()
        resumed = execute_matrix(
            matrix_dir=matrix_dir,
            generation_cache_path=self.cache_path,
            repo_root=self.root,
            schema_root=ROOT / "schemas",
            methodology_check=lambda _: {"frozen": True},
            run_executor=fake_executor,
        )
        self.assertTrue(resumed["complete"])
        self.assertEqual(resumed["executed_groups"], 1)
        self.assertEqual(len(calls), 1)

    def test_dataset_binding_allows_only_azure_deployment_name_change(self) -> None:
        frozen_bytes = self.models_path.read_bytes()
        lock = json.loads(
            (self.dataset / "methodology" / "methodology.lock.json").read_text(encoding="utf-8")
        )
        models = json.loads(frozen_bytes)
        azure = next(model for model in models["models"] if model["provider"] == "azure")
        azure["deployment"] = "renamed-azure-deployment"
        _write_json(self.models_path, models)
        git_show = MagicMock(stdout=frozen_bytes)

        with patch("kg_benchmark.matrix.workflow.subprocess.run", return_value=git_show):
            self.assertTrue(_models_differ_only_by_azure_deployment(lock, self.models_path))
            self._plan()

        azure["model_revision"] = "different-snapshot"
        _write_json(self.models_path, models)
        with patch("kg_benchmark.matrix.workflow.subprocess.run", return_value=git_show):
            self.assertFalse(_models_differ_only_by_azure_deployment(lock, self.models_path))
            with self.assertRaisesRegex(
                MatrixWorkflowError, "Model configuration does not match"
            ):
                self._plan()

    def test_unresolved_revision_is_visible_in_dry_run_and_blocks_execution(self) -> None:
        models = json.loads(self.models_path.read_text())
        models["models"] = [models["models"][0]]
        models["models"][0]["model_revision"] = None
        models["models"][0]["revision_status"] = "unresolved"
        _write_json(self.models_path, models)
        lock_path = self.dataset / "methodology" / "methodology.lock.json"
        lock = json.loads(lock_path.read_text())
        lock["files"]["paper/models.json"] = _sha(self.models_path)
        _write_json(lock_path, lock)
        manifest = json.loads((self.dataset / "manifest.json").read_text())
        manifest["methodology"]["lock"]["sha256"] = _sha(lock_path)
        _write_json(self.dataset / "manifest.json", manifest)
        matrix_dir, _ = self._plan()
        report = dry_run_matrix(
            matrix_dir=matrix_dir,
            generation_cache_path=self.cache_path,
            schema_root=ROOT / "schemas",
        )
        self.assertEqual(report["missing_revision_models"], ["ollama_test"])
        with self.assertRaisesRegex(MatrixWorkflowError, "unresolved model revisions"):
            execute_matrix(
                matrix_dir=matrix_dir,
                generation_cache_path=self.cache_path,
                repo_root=self.root,
                schema_root=ROOT / "schemas",
                methodology_check=lambda _: {"frozen": True},
                run_executor=lambda **_: {},
            )

    def test_status_detects_mutated_execution_artifact(self) -> None:
        matrix_dir, _ = self._plan(model_ids=["ollama_test"])
        requests = [json.loads(line) for line in (matrix_dir / "requests.jsonl").read_text().splitlines()]

        def fake_executor(**kwargs: Any) -> dict[str, Any]:
            group_dir = Path(kwargs["resume_run_dir"])
            population = json.loads(Path(kwargs["selection_manifest_path"]).read_text())
            selected = [
                row
                for row in requests
                if row["population"] == population["name"]
                and row["context_bundle"] == kwargs["ablation_bundles"][0]
                and row["prompt_regime"] == kwargs["prompt_regime"]
            ]
            cache = GenerationCache(self.cache_path)
            for row in selected:
                cache.put(row["request_spec"], raw_response={}, parsed_payload={}, usage={})
            _write_json(group_dir / "run_config.json", {})
            _write_jsonl(
                group_dir / "run_manifest.jsonl",
                [
                    {
                        "case_id": row["case_id"],
                        "task_type": "proposal" if row["task"] == "repair_proposal" else "track_diagnosis",
                        "ablation_bundle": row["context_bundle"],
                        "parse_status": "normalized",
                    }
                    for row in selected
                ],
            )
            _write_json(group_dir / "reasoning_floor_summary.json", {})
            return {}

        execute_matrix(
            matrix_dir=matrix_dir,
            generation_cache_path=self.cache_path,
            repo_root=self.root,
            schema_root=ROOT / "schemas",
            methodology_check=lambda _: {},
            run_executor=fake_executor,
        )
        run_manifest = next((matrix_dir / "executions").glob("*/run_manifest.jsonl"))
        run_manifest.write_text("", encoding="utf-8")
        status = matrix_status(
            matrix_dir=matrix_dir,
            generation_cache_path=self.cache_path,
            schema_root=ROOT / "schemas",
        )
        self.assertFalse(status["complete"])
        self.assertGreater(status["incomplete_cells"], 0)


if __name__ == "__main__":
    unittest.main()
