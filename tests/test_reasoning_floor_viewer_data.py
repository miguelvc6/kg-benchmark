import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from guardian.evaluator import evaluate_benchmark, write_json, write_jsonl
from guardian.reasoning_floor_viewer_data import build_case_prompt_debug, load_bundle_debug_data


class ReasoningFloorViewerDataTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def _raw_response_record(self, case_id: str, bundle: str, task_type: str, content: str) -> dict:
        return {
            "run_id": "run_001",
            "case_id": case_id,
            "ablation_bundle": bundle,
            "task_type": task_type,
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "content": content,
                        }
                    }
                ]
            },
            "parsed_payload": json.loads(content),
        }

    def _build_fixture(self, *, with_evaluation_artifacts: bool) -> tuple[Path, Path, Path, Path]:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        root = Path(tmp_dir.name)
        classified_path = root / "data" / "04_classified_benchmark.jsonl"
        world_state_path = root / "data" / "03_world_state.json"
        reports_root = root / "reports" / "reasoning_floor"
        run_dir = reports_root / "run_001_openai_stub_model"
        bundle_dir = run_dir / "minimal_case"

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
            {
                "id": "broken_case",
                "qid": "Q3",
                "property": "P31",
                "track": "A_BOX",
                "labels_en": {},
                "violation_context": {"value": ["Q7"]},
                "repair_target": {"action": "UPDATE", "old_value": ["Q7"], "new_value": ["Q8"]},
                "persistence_check": {},
                "popularity": {"score": 0.5},
                "classification": {"class": "TypeA", "subtype": "DIRECT_VALUE"},
            },
        ]
        self._write_jsonl(classified_path, classified_rows)

        self._write_json(
            world_state_path,
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
                    "L1_ego_node": {"qid": "Q2", "label": "Entity", "description": "desc", "properties": {}},
                    "L2_labels": {},
                    "L3_neighborhood": {"outgoing_edges": []},
                    "L4_constraints": {"constraints": []},
                    "constraint_change_context": {},
                },
                "broken_case": {
                    "L1_ego_node": {
                        "qid": "Q3",
                        "label": "Broken",
                        "description": "desc",
                        "properties": {"P31": ["Q7"]},
                    },
                    "L2_labels": {},
                    "L3_neighborhood": {"outgoing_edges": []},
                    "L4_constraints": {"constraints": []},
                },
            },
        )

        diagnosis_rows = [
            {"case_id": "repair_case", "predicted_track": "A_BOX", "confidence": "high", "canonical_hash": "hash-a"},
            {"case_id": "reform_case", "predicted_track": "T_BOX", "confidence": "high", "canonical_hash": "hash-b"},
            {"case_id": "broken_case", "predicted_track": "A_BOX", "confidence": "medium", "canonical_hash": "hash-c"},
        ]
        a_box_rows = [
            {
                "case_id": "repair_case",
                "target": {"qid": "Q1", "pid": "P31"},
                "ops": [{"op": "SET", "pid": "P31", "value": "Q5", "rank": "normal"}],
                "canonical_hash": "proposal-a",
            }
        ]
        t_box_rows = [
            {
                "case_id": "reform_case",
                "target": {"pid": "P31", "constraint_type_qid": "Q21510859"},
                "proposal": {
                    "action": "RELAXATION_SET_EXPANSION",
                    "signature_after": [
                        {
                            "constraint_qid": "Q21510859",
                            "snaktype": "VALUE",
                            "rank": "normal",
                            "qualifiers": [{"property_id": "P2305", "values": ["Q43229", "Q5"]}],
                        }
                    ],
                },
                "canonical_hash": "proposal-b",
            }
        ]
        self._write_jsonl(bundle_dir / "track_diagnoses.jsonl", diagnosis_rows)
        self._write_jsonl(bundle_dir / "a_box_proposals.jsonl", a_box_rows)
        self._write_jsonl(bundle_dir / "t_box_proposals.jsonl", t_box_rows)

        manifest_rows = [
            {
                "run_id": "run_001",
                "case_id": "repair_case",
                "ablation_bundle": "minimal_case",
                "track": "A_BOX",
                "task_type": "track_diagnosis",
                "provider": "openai",
                "model": "stub-model",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "elapsed_seconds": 1.0},
                "parse_status": "normalized",
            },
            {
                "run_id": "run_001",
                "case_id": "repair_case",
                "ablation_bundle": "minimal_case",
                "track": "A_BOX",
                "task_type": "proposal",
                "provider": "openai",
                "model": "stub-model",
                "usage": {"prompt_tokens": 11, "completion_tokens": 6, "total_tokens": 17, "elapsed_seconds": 2.0},
                "parse_status": "normalized",
            },
            {
                "run_id": "run_001",
                "case_id": "reform_case",
                "ablation_bundle": "minimal_case",
                "track": "T_BOX",
                "task_type": "track_diagnosis",
                "provider": "openai",
                "model": "stub-model",
                "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19, "elapsed_seconds": 1.5},
                "parse_status": "normalized",
            },
            {
                "run_id": "run_001",
                "case_id": "reform_case",
                "ablation_bundle": "minimal_case",
                "track": "T_BOX",
                "task_type": "proposal",
                "provider": "openai",
                "model": "stub-model",
                "usage": {"prompt_tokens": 13, "completion_tokens": 8, "total_tokens": 21, "elapsed_seconds": 2.5},
                "parse_status": "normalized",
            },
            {
                "run_id": "run_001",
                "case_id": "broken_case",
                "ablation_bundle": "minimal_case",
                "track": "A_BOX",
                "task_type": "track_diagnosis",
                "provider": "openai",
                "model": "stub-model",
                "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14, "elapsed_seconds": 1.1},
                "parse_status": "normalized",
            },
            {
                "run_id": "run_001",
                "case_id": "broken_case",
                "ablation_bundle": "minimal_case",
                "track": "A_BOX",
                "task_type": "proposal",
                "provider": "openai",
                "model": "stub-model",
                "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25, "elapsed_seconds": 3.0},
                "parse_status": "parse_error",
                "parser_error": "case_id must be a non-empty string.",
            },
        ]
        raw_rows = [
            self._raw_response_record(
                "repair_case",
                "minimal_case",
                "track_diagnosis",
                '{"case_id":"repair_case","predicted_track":"A_BOX","confidence":"high"}',
            ),
            self._raw_response_record(
                "repair_case",
                "minimal_case",
                "proposal",
                '{"case_id":"repair_case","target":{"qid":"Q1","pid":"P31"},"ops":[{"op":"SET","pid":"P31","value":"Q5"}]}',
            ),
            self._raw_response_record(
                "reform_case",
                "minimal_case",
                "track_diagnosis",
                '{"case_id":"reform_case","predicted_track":"T_BOX","confidence":"high"}',
            ),
            self._raw_response_record(
                "reform_case",
                "minimal_case",
                "proposal",
                '{"case_id":"reform_case","target":{"pid":"P31","constraint_type_qid":"Q21510859"},"proposal":{"action":"RELAXATION_SET_EXPANSION","signature_after":[{"constraint_qid":"Q21510859","snaktype":"VALUE","rank":"normal","qualifiers":[{"property_id":"P2305","values":["Q5","Q43229"]}]}]}}',
            ),
            self._raw_response_record(
                "broken_case",
                "minimal_case",
                "track_diagnosis",
                '{"case_id":"broken_case","predicted_track":"A_BOX","confidence":"medium"}',
            ),
            self._raw_response_record(
                "broken_case",
                "minimal_case",
                "proposal",
                '{"id":"broken_case","ops":[]}',
            ),
        ]
        batch_request_manifest_rows = [
            {
                "custom_id": "req_diag_repair",
                "metadata": {
                    "run_id": "run_001",
                    "case_id": "repair_case",
                    "ablation_bundle": "minimal_case",
                    "task_type": "track_diagnosis",
                },
            },
            {
                "custom_id": "req_prop_repair",
                "metadata": {
                    "run_id": "run_001",
                    "case_id": "repair_case",
                    "ablation_bundle": "minimal_case",
                    "task_type": "proposal",
                },
            },
        ]
        batch_input_rows = [
            {
                "custom_id": "req_diag_repair",
                "body": {
                    "model": "stub-model",
                    "messages": [
                        {"role": "system", "content": "diagnosis system prompt"},
                        {"role": "user", "content": '{"id":"repair_case","mode":"diagnosis"}'},
                    ],
                    "response_format": {"type": "json_object"},
                },
            },
            {
                "custom_id": "req_prop_repair",
                "body": {
                    "model": "stub-model",
                    "messages": [
                        {"role": "system", "content": "proposal system prompt"},
                        {"role": "user", "content": '{"id":"repair_case","mode":"proposal"}'},
                    ],
                    "response_format": {"type": "json_object"},
                },
            },
        ]
        self._write_jsonl(run_dir / "run_manifest.jsonl", manifest_rows)
        self._write_jsonl(run_dir / "raw_model_responses.jsonl", raw_rows)
        self._write_jsonl(run_dir / "batch_request_manifest.jsonl", batch_request_manifest_rows)
        self._write_jsonl(run_dir / "batch_input.jsonl", batch_input_rows)
        self._write_json(
            run_dir / "reasoning_floor_summary.json",
            {
                "run_info": {
                    "run_id": "run_001",
                    "provider": "openai",
                    "model": "stub-model",
                    "output_dir": str(run_dir),
                    "execution_mode": "batch",
                    "batch_mode_used": True,
                    "proposal_track_mode": "diagnosis_routed",
                    "batch": {
                        "mode": "two_stage",
                        "overall_status": "completed",
                        "elapsed_seconds": 3.5,
                        "phases": {
                            "diagnosis": {"status": "completed", "request_counts": {"total": 3}},
                            "proposal": {"status": "completed", "request_counts": {"total": 2}},
                        },
                    },
                }
            },
        )

        if with_evaluation_artifacts:
            traces, summary = evaluate_benchmark(
                classified_path=classified_path,
                world_state_path=world_state_path,
                a_box_proposals_path=bundle_dir / "a_box_proposals.jsonl",
                t_box_proposals_path=bundle_dir / "t_box_proposals.jsonl",
                track_diagnoses_path=bundle_dir / "track_diagnoses.jsonl",
                run_manifest_path=run_dir / "run_manifest.jsonl",
                ablation_bundle="minimal_case",
                case_ids=["repair_case", "reform_case", "broken_case"],
                collect_traces=True,
            )
            write_jsonl(bundle_dir / "evaluation_traces.jsonl", traces)
            write_json(bundle_dir / "evaluation_summary.json", summary)

        return root, reports_root, run_dir, bundle_dir

    def test_load_bundle_debug_data_computes_live_evaluation_and_joins_cases(self) -> None:
        root, reports_root, run_dir, _ = self._build_fixture(with_evaluation_artifacts=False)

        bundle_data = load_bundle_debug_data(
            reports_root=reports_root,
            run_dir=run_dir,
            bundle_name="minimal_case",
            classified_benchmark=root / "data" / "04_classified_benchmark.jsonl",
            world_state=root / "data" / "03_world_state.json",
        )

        self.assertEqual(bundle_data.traces_source, "live")
        self.assertEqual(bundle_data.summary_source, "live")
        self.assertEqual(bundle_data.bundle_summary["counts"]["cases"], 3)
        self.assertEqual(len(bundle_data.case_rows), 3)

        repair_case = next(row for row in bundle_data.case_rows if row.case_id == "repair_case")
        reform_case = next(row for row in bundle_data.case_rows if row.case_id == "reform_case")
        broken_case = next(row for row in bundle_data.case_rows if row.case_id == "broken_case")

        self.assertEqual(repair_case.proposal_type, "A_BOX")
        self.assertIsNotNone(repair_case.proposal_normalized)
        self.assertEqual(reform_case.proposal_type, "T_BOX")
        self.assertIsNotNone(reform_case.proposal_normalized)
        self.assertIsNone(broken_case.proposal_normalized)
        self.assertEqual(broken_case.proposal_parse_status, "parse_error")
        self.assertEqual(
            broken_case.proposal_manifest["parser_error"],
            "case_id must be a non-empty string.",
        )
        self.assertIsNotNone(broken_case.proposal_raw)
        self.assertIsNotNone(broken_case.diagnosis_normalized)

    def test_load_bundle_debug_data_prefers_existing_evaluation_artifacts(self) -> None:
        root, reports_root, run_dir, _ = self._build_fixture(with_evaluation_artifacts=True)

        bundle_data = load_bundle_debug_data(
            reports_root=reports_root,
            run_dir=run_dir,
            bundle_name="minimal_case",
            classified_benchmark=root / "data" / "04_classified_benchmark.jsonl",
            world_state=root / "data" / "03_world_state.json",
        )

        self.assertEqual(bundle_data.traces_source, "artifact")
        self.assertEqual(bundle_data.summary_source, "artifact")
        self.assertEqual(bundle_data.bundle_summary["counts"]["cases"], 3)
        self.assertEqual(bundle_data.usage_summary["call_count"], 6)
        self.assertEqual(bundle_data.run_info["proposal_track_mode"], "diagnosis_routed")
        self.assertEqual(bundle_data.run_info["batch"]["mode"], "two_stage")

    def test_build_case_prompt_debug_reconstructs_case_inputs(self) -> None:
        root, reports_root, run_dir, _ = self._build_fixture(with_evaluation_artifacts=False)

        bundle_data = load_bundle_debug_data(
            reports_root=reports_root,
            run_dir=run_dir,
            bundle_name="minimal_case",
            classified_benchmark=root / "data" / "04_classified_benchmark.jsonl",
            world_state=root / "data" / "03_world_state.json",
        )
        prompt_debug = build_case_prompt_debug(bundle_data, "repair_case")

        self.assertIsNone(prompt_debug.error)
        self.assertEqual(prompt_debug.proposal_prompt.prompt_name, "reasoning_floor_a_box_zero_shot")
        self.assertEqual(prompt_debug.diagnosis_prompt.prompt_name, "reasoning_floor_track_diagnosis_zero_shot")
        self.assertIn('"id": "repair_case"', prompt_debug.proposal_prompt.prompt)

    def test_build_case_prompt_debug_uses_routed_proposal_track_when_present(self) -> None:
        root, reports_root, run_dir, _ = self._build_fixture(with_evaluation_artifacts=False)
        manifest_path = run_dir / "run_manifest.jsonl"
        manifest_rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in manifest_rows:
            if row["case_id"] == "repair_case" and row["task_type"] == "proposal":
                row["proposal_track_used"] = "T_BOX"
        self._write_jsonl(manifest_path, manifest_rows)

        bundle_data = load_bundle_debug_data(
            reports_root=reports_root,
            run_dir=run_dir,
            bundle_name="minimal_case",
            classified_benchmark=root / "data" / "04_classified_benchmark.jsonl",
            world_state=root / "data" / "03_world_state.json",
        )
        prompt_debug = build_case_prompt_debug(bundle_data, "repair_case")

        self.assertIsNone(prompt_debug.error)
        self.assertEqual(prompt_debug.proposal_prompt.prompt_name, "reasoning_floor_t_box_zero_shot")

    def test_load_bundle_debug_data_can_skip_eager_case_record_loading(self) -> None:
        root, reports_root, run_dir, _ = self._build_fixture(with_evaluation_artifacts=True)

        with mock.patch("guardian.reasoning_floor_viewer_data._load_case_records") as mocked_loader:
            mocked_loader.side_effect = AssertionError("_load_case_records should not be called")
            bundle_data = load_bundle_debug_data(
                reports_root=reports_root,
                run_dir=run_dir,
                bundle_name="minimal_case",
                classified_benchmark=root / "data" / "04_classified_benchmark.jsonl",
                world_state=root / "data" / "03_world_state.json",
                include_case_records=False,
            )

        repair_case = next(row for row in bundle_data.case_rows if row.case_id == "repair_case")
        self.assertIsNone(repair_case.record)
        self.assertEqual(repair_case.diagnosis_request["body"]["messages"][0]["content"], "diagnosis system prompt")
        self.assertEqual(repair_case.proposal_request["body"]["messages"][0]["content"], "proposal system prompt")


if __name__ == "__main__":
    unittest.main()
