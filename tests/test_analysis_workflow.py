from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

from kg_benchmark.analysis.workflow import (
    AnalysisWorkflowError,
    _analysis_scope_and_roles,
    build_paper_results,
    replay_matrix_evaluations,
    verify_paper_results,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, *, relative_to: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve().relative_to(relative_to.resolve())) if path.is_relative_to(relative_to) else str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }
    if path.suffix == ".jsonl":
        result["records"] = len(path.read_text(encoding="utf-8").splitlines())
    return result


def _fingerprint(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


class AnalysisWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.matrix_dir = self.root / "runs" / "matrices" / ("matrix_" + "a" * 20)
        self.results = self.root / "results"
        self.analysis_config = self.root / "analysis.json"
        self.cache = self.root / "cache.sqlite"
        self._build_fixture()

    def _build_fixture(self) -> None:
        matrix_id = "matrix_" + "a" * 20
        cases = self.root / "dataset" / "cases.jsonl"
        world = self.root / "dataset" / "source" / "world-state.jsonl"
        population = self.root / "dataset" / "selections" / "main-2.json"
        models = self.root / "paper" / "models.json"
        protocol = self.root / "paper" / "protocol.json"
        _write_jsonl(
            cases,
            [
                {
                    "id": "case_a",
                    "qid": "Q1",
                    "property": "P31",
                    "track": "A_BOX",
                    "classification": {"class": "TypeA", "subtype": "DIRECT_VALUE"},
                },
                {
                    "id": "case_b",
                    "qid": "Q2",
                    "property": "P31",
                    "track": "A_BOX",
                    "classification": {"class": "TypeA", "subtype": "DIRECT_VALUE"},
                },
                {
                    "id": "case_t",
                    "qid": "Q3",
                    "property": "P279",
                    "track": "T_BOX",
                    "repair_target": {"property_revision_id": 99},
                    "classification": {"class": "T_BOX", "subtype": "SCHEMA_UPDATE"},
                },
            ],
        )
        _write_jsonl(
            world,
            [
                {"id": "case_a", "world_state": {}},
                {"id": "case_b", "world_state": {}},
                {"id": "case_t", "world_state": {}},
            ],
        )
        _write_json(
            population,
            {
                "manifest_type": "evaluation_population",
                "manifest_version": 1,
                "name": "main-2",
                "quotas": {"IC-L": 2, "IC-G": 0, "IC-E-elim": 0, "TBOX": 1},
                "case_count": 3,
                "selected_case_ids": ["case_a", "case_b", "case_t"],
                "selected_group_keys": ["ABOX|Q1|P31", "ABOX|Q2|P31", "TBOX|P279|99"],
                "support_groups_excluded": 0,
                "parent": None,
            },
        )
        model = {
            "model_id": "ollama_test",
            "provider": "ollama",
            "model": "test:1b",
            "model_revision": "d" * 64,
            "revision_status": "resolved",
            "revision_resolution": "test",
            "population": "main-2",
            "execution_mode": "sync",
            "temperature": 0,
            "top_p": 1,
            "seed": 13,
            "context_length": 4096,
            "max_output_tokens": 512,
            "ollama_think": "disabled",
            "max_transport_retries": 2,
            "tools_disabled": True,
            "expected_calls": 24,
        }
        _write_json(
            models,
            {
                "manifest_type": "paper_model_matrix",
                "manifest_version": 1,
                "status": "frozen",
                "request_dimensions": {
                    "tasks": ["repair_proposal", "track_diagnosis"],
                    "prompt_regimes": ["zero_shot", "static_few_shot"],
                    "context_bundles": ["logic_only", "local_graph"],
                },
                "generation_identity": ["a", "b", "c", "d", "e", "f"],
                "models": [model],
            },
        )
        _write_json(protocol, {"protocol": "synthetic"})
        _write_json(
            self.analysis_config,
            {
                "manifest_type": "paper_analysis_configuration",
                "manifest_version": 1,
                "status": "frozen",
                "seed": 13,
                "confirmatory_population": "main-2",
                "calibration_population": "api-1",
                "pool_models": False,
                "primary_contrasts": [
                    {
                        "contrast_id": "context_within_zero_shot",
                        "left": {"prompt_regime": "zero_shot", "context_bundle": "logic_only"},
                        "right": {"prompt_regime": "zero_shot", "context_bundle": "local_graph"},
                    },
                    {
                        "contrast_id": "context_within_few_shot",
                        "left": {"prompt_regime": "static_few_shot", "context_bundle": "logic_only"},
                        "right": {"prompt_regime": "static_few_shot", "context_bundle": "local_graph"},
                    },
                    {
                        "contrast_id": "few_shot_within_logic_only",
                        "left": {"prompt_regime": "zero_shot", "context_bundle": "logic_only"},
                        "right": {"prompt_regime": "static_few_shot", "context_bundle": "logic_only"},
                    },
                    {
                        "contrast_id": "few_shot_within_local_graph",
                        "left": {"prompt_regime": "zero_shot", "context_bundle": "local_graph"},
                        "right": {"prompt_regime": "static_few_shot", "context_bundle": "local_graph"},
                    },
                ],
                "inference": {
                    "paired_binary_test": "exact_mcnemar",
                    "multiplicity_correction": "holm",
                    "bootstrap": {
                        "method": "percentile_cluster_bootstrap",
                        "samples": 5000,
                        "confidence_level": 0.95,
                        "seed": 13,
                    }
                },
            },
        )

        cells: list[dict[str, Any]] = []
        requests: list[dict[str, Any]] = []
        condition_outcomes = {
            ("zero_shot", "logic_only"): [False, False],
            ("zero_shot", "local_graph"): [True, False],
            ("static_few_shot", "logic_only"): [True, True],
            ("static_few_shot", "local_graph"): [True, False],
        }
        for condition_index, ((regime, context), outcomes) in enumerate(condition_outcomes.items(), 1):
            group_id = f"group_{condition_index:020x}"
            for task_index, task in enumerate(("repair_proposal", "track_diagnosis"), 1):
                cell_id = f"cell_{condition_index * 10 + task_index:020x}"
                cells.append(
                    {
                        "cell_id": cell_id,
                        "execution_group_id": group_id,
                        "model_id": "ollama_test",
                        "population": "main-2",
                        "task": task,
                        "prompt_regime": regime,
                        "context_bundle": context,
                        "expected_requests": 3,
                        "manifest_path": f"cells/{cell_id}.json",
                    }
                )
                for case_id in ("case_a", "case_b", "case_t"):
                    requests.append(
                        {
                            "matrix_id": matrix_id,
                            "cell_id": cell_id,
                            "execution_group_id": group_id,
                            "model_id": "ollama_test",
                            "provider": "ollama",
                            "model": "test:1b",
                            "population": "main-2",
                            "case_id": case_id,
                            "case_payload_sha256": "1" * 64,
                            "task": task,
                            "prompt_regime": regime,
                            "rendered_prompt_sha256": "2" * 64,
                            "context_bundle": context,
                            "context_sha256": "3" * 64,
                            "model_revision": "d" * 64,
                            "inference_parameters_sha256": "4" * 64,
                            "request_key": "5" * 64,
                            "request_spec": {},
                        }
                    )
            self._write_evaluation(group_id, regime, context, outcomes)
        requests_path = self.matrix_dir / "requests.jsonl"
        _write_jsonl(requests_path, requests)
        matrix = {
            "manifest_type": "paper_experiment_matrix",
            "manifest_version": 1,
            "matrix_id": matrix_id,
            "dataset": {
                "cases": _artifact(cases, relative_to=self.matrix_dir),
                "world_state": _artifact(world, relative_to=self.matrix_dir),
            },
            "model_configuration": _artifact(models, relative_to=self.matrix_dir),
            "protocol": _artifact(protocol, relative_to=self.matrix_dir),
            "dimensions": {
                "tasks": ["repair_proposal", "track_diagnosis"],
                "prompt_regimes": ["zero_shot", "static_few_shot"],
                "context_bundles": ["logic_only", "local_graph"],
            },
            "populations": {
                "main-2": {"artifact": _artifact(population, relative_to=self.matrix_dir), "case_count": 3}
            },
            "models": [model],
            "cells": cells,
            "requests": _artifact(requests_path, relative_to=self.matrix_dir),
            "workload": {
                "expected_requests": 24,
                "logical_cells": 8,
                "execution_groups": 4,
                "by_model": {"ollama_test": 24},
            },
            "validation": {
                "population_not_in_generation_identity": True,
                "tools_disabled": True,
                "provider_policies_valid": True,
            },
        }
        _write_json(self.matrix_dir / "matrix.json", matrix)

    def _write_evaluation(self, group_id: str, regime: str, context: str, outcomes: list[bool]) -> None:
        group = self.matrix_dir / "executions" / group_id
        evaluation = group / "evaluations" / "metrics_v1"
        bundle = evaluation / context
        _write_json(group / "run_config.json", {"regime": regime, "context": context})
        _write_jsonl(group / "run_manifest.jsonl", [])
        traces = []
        diagnosis = []
        for case_id, outcome in zip(("case_a", "case_b"), outcomes, strict=True):
            traces.append(
                {
                    "case_id": case_id,
                    "accepted": outcome,
                    "metrics": {
                        "functional_success": float(outcome),
                        "a_box_exact_action_match": float(outcome),
                        "a_box_exact_value_match": float(outcome),
                        "a_box_regression_pass": float(outcome),
                        "auditability_complete": float(outcome),
                        "provenance_supported": float(outcome),
                    },
                }
            )
            diagnosis.append(
                {
                    "case_id": case_id,
                    "historical_track": "A_BOX",
                    "predicted_track": "A_BOX" if outcome else "T_BOX",
                    "exact_track_match": outcome,
                    "parse_status": "normalized",
                }
            )
        tbox_outcome = outcomes[0]
        tbox_trace = {
            "case_id": "case_t",
            "parsed": tbox_outcome,
            "metrics": {
                "schema_decision_match": tbox_outcome,
                "taxonomy_code_exact_match": tbox_outcome,
                "repair_op_exact_match": tbox_outcome,
                "qualifier_property_match": tbox_outcome,
                "evidence_level_exact_match": tbox_outcome,
                "gold_has_value_delta": True,
            },
            "metric_detail": {
                "value_tp": 1 if tbox_outcome else 0,
                "value_pred": 1 if tbox_outcome else 0,
                "value_gold": 1,
            },
        }
        diagnosis.append(
            {
                "case_id": "case_t",
                "historical_track": "T_BOX",
                "predicted_track": "T_BOX" if tbox_outcome else "A_BOX",
                "exact_track_match": tbox_outcome,
                "parse_status": "normalized" if tbox_outcome else "parse_error",
            }
        )
        _write_jsonl(bundle / "evaluation_traces.jsonl", traces)
        _write_json(bundle / "evaluation_summary.json", {})
        _write_jsonl(bundle / "tbox_taxonomy_patch_evaluation_traces.jsonl", [tbox_trace])
        _write_json(bundle / "tbox_taxonomy_patch_evaluation_summary.json", {})
        _write_jsonl(bundle / "diagnosis_evaluation_traces.jsonl", diagnosis)
        _write_json(bundle / "diagnosis_evaluation_summary.json", {})
        _write_json(evaluation / "evaluation_summary.json", {})
        outputs = {
            "traces": _fingerprint(bundle / "evaluation_traces.jsonl"),
            "summary": _fingerprint(bundle / "evaluation_summary.json"),
            "tbox_taxonomy_patch_traces": _fingerprint(bundle / "tbox_taxonomy_patch_evaluation_traces.jsonl"),
            "tbox_taxonomy_patch_summary": _fingerprint(bundle / "tbox_taxonomy_patch_evaluation_summary.json"),
            "diagnosis_traces": _fingerprint(bundle / "diagnosis_evaluation_traces.jsonl"),
            "diagnosis_summary": _fingerprint(bundle / "diagnosis_evaluation_summary.json"),
        }
        _write_json(
            evaluation / "evaluation_manifest.json",
            {
                "manifest_type": "evaluation_replay",
                "manifest_version": 1,
                "evaluation_id": "metrics_v1",
                "created_at_utc": "2026-01-01T00:00:00Z",
                "source_run_dir": str(group.resolve()),
                "provider_calls": 0,
                "selected_case_count": 3,
                "ablation_bundles": [context],
                "metric_families": ["a_box_repair_v1", "tbox_taxonomy_patch_v1", "track_diagnosis_v1"],
                "combined_repair_success_score": False,
                "source_artifacts": {
                    "run_config": _fingerprint(group / "run_config.json"),
                    "run_manifest": _fingerprint(group / "run_manifest.jsonl"),
                },
                "evaluation_code": {},
                "outputs": {"combined_summary": _fingerprint(evaluation / "evaluation_summary.json"), context: outputs},
            },
        )

    @staticmethod
    def _complete(**_: Any) -> dict[str, Any]:
        return {"complete": True}

    def test_replay_verifies_and_reuses_every_provider_free_evaluation(self) -> None:
        calls = []
        report = replay_matrix_evaluations(
            matrix_dir=self.matrix_dir,
            evaluation_id="metrics_v1",
            generation_cache_path=self.cache,
            schema_root=ROOT / "schemas",
            rescorer=lambda **kwargs: calls.append(kwargs),
            completeness_check=self._complete,
        )
        self.assertEqual(calls, [])
        self.assertEqual(report["provider_calls"], 0)
        self.assertEqual(report["reused_evaluations"], 4)
        self.assertTrue(report["complete_evaluation_coverage"])

    def test_replay_rejects_a_mutated_combined_summary(self) -> None:
        summary = (
            self.matrix_dir
            / "executions"
            / "group_00000000000000000001"
            / "evaluations"
            / "metrics_v1"
            / "evaluation_summary.json"
        )
        _write_json(summary, {"mutated": True})

        with self.assertRaisesRegex(AnalysisWorkflowError, "fingerprint mismatch"):
            replay_matrix_evaluations(
                matrix_dir=self.matrix_dir,
                evaluation_id="metrics_v1",
                generation_cache_path=self.cache,
                schema_root=ROOT / "schemas",
                completeness_check=self._complete,
            )

    def test_builds_separated_reproducible_result_package(self) -> None:
        result = build_paper_results(
            matrix_dir=self.matrix_dir,
            evaluation_id="metrics_v1",
            generation_cache_path=self.cache,
            analysis_config_path=self.analysis_config,
            output_root=self.results,
            schema_root=ROOT / "schemas",
            completeness_check=self._complete,
        )
        result_dir = Path(result["result_dir"])
        manifest_bytes = (result_dir / "manifest.json").read_bytes()
        summary = json.loads((result_dir / "summary.json").read_text())
        estimates = [json.loads(line) for line in (result_dir / "estimates.jsonl").read_text().splitlines()]
        contrasts = [json.loads(line) for line in (result_dir / "contrasts.jsonl").read_text().splitlines()]
        self.assertEqual(summary["scope"], "confirmatory_and_calibration")
        self.assertEqual(summary["roles"], ["confirmatory"])
        self.assertEqual(summary["models"], ["ollama_test"])
        self.assertTrue(all(row["role"] == "confirmatory" for row in estimates))
        self.assertEqual(
            {row["task"] for row in estimates},
            {"a_box_repair", "t_box_repair", "track_diagnosis"},
        )
        self.assertIn("tbox_patch_schema_decision_match_rate", {row["endpoint"] for row in estimates})
        self.assertEqual({count for count in Counter(row["family_id"] for row in contrasts).values()}, {4})
        self.assertTrue((result_dir / "tables" / "confirmatory.md").is_file())
        self.assertTrue(verify_paper_results(result_dir=result_dir, schema_root=ROOT / "schemas")["valid"])

        repeated = build_paper_results(
            matrix_dir=self.matrix_dir,
            evaluation_id="metrics_v1",
            generation_cache_path=self.cache,
            analysis_config_path=self.analysis_config,
            output_root=self.results,
            schema_root=ROOT / "schemas",
            completeness_check=self._complete,
        )
        self.assertEqual(Path(repeated["result_dir"]), result_dir)
        self.assertEqual((result_dir / "manifest.json").read_bytes(), manifest_bytes)

        independent = build_paper_results(
            matrix_dir=self.matrix_dir,
            evaluation_id="metrics_v1",
            generation_cache_path=self.cache,
            analysis_config_path=self.analysis_config,
            output_root=self.root / "results-copy",
            schema_root=ROOT / "schemas",
            completeness_check=self._complete,
        )
        independent_dir = Path(independent["result_dir"])
        for relative in (
            "manifest.json",
            "summary.json",
            "estimates.jsonl",
            "contrasts.jsonl",
            "diagnosis.jsonl",
            "tables/confirmatory.md",
        ):
            self.assertEqual((result_dir / relative).read_bytes(), (independent_dir / relative).read_bytes())

    def test_result_verification_detects_mutated_compact_output(self) -> None:
        result = build_paper_results(
            matrix_dir=self.matrix_dir,
            evaluation_id="metrics_v1",
            generation_cache_path=self.cache,
            analysis_config_path=self.analysis_config,
            output_root=self.results,
            schema_root=ROOT / "schemas",
            completeness_check=self._complete,
        )
        result_dir = Path(result["result_dir"])
        (result_dir / "estimates.jsonl").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(AnalysisWorkflowError, "fingerprint mismatch"):
            verify_paper_results(result_dir=result_dir, schema_root=ROOT / "schemas")

    def test_extension_requires_hash_bound_per_stratum_parent_prefix(self) -> None:
        config = {"confirmatory_population": "main", "calibration_population": "api"}
        metadata = {
            "a": {"stratum": "IC-L"},
            "b": {"stratum": "IC-L"},
        }
        parent = {"name": "main", "selected_case_ids": ["a"], "__artifact_sha256": "1" * 64}
        extension = {
            "name": "expanded",
            "selected_case_ids": ["a", "b"],
            "parent": {"name": "main", "nesting_proven": True, "relationship": "nested_extension_of_parent"},
            "provenance": {"parent_population": {"sha256": "1" * 64}},
        }
        scope, roles = _analysis_scope_and_roles(
            config=config,
            populations={"main": parent, "expanded": extension},
            metadata=metadata,
        )
        self.assertEqual(scope, "extension")
        self.assertIsNone(roles["main"])
        self.assertEqual(roles["expanded"], "extension")
        extension["selected_case_ids"] = ["b", "a"]
        with self.assertRaisesRegex(AnalysisWorkflowError, "parent prefix"):
            _analysis_scope_and_roles(
                config=config,
                populations={"main": parent, "expanded": extension},
                metadata=metadata,
            )

    def test_base_populations_have_distinct_reporting_roles(self) -> None:
        scope, roles = _analysis_scope_and_roles(
            config={"confirmatory_population": "main", "calibration_population": "api"},
            populations={"main": {}, "api": {}},
            metadata={},
        )

        self.assertEqual(scope, "confirmatory_and_calibration")
        self.assertEqual(roles, {"main": "confirmatory", "api": "azure_calibration"})


if __name__ == "__main__":
    unittest.main()
