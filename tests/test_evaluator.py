import json
import tempfile
import unittest
from pathlib import Path

from guardian.evaluator import evaluate_benchmark, summarize_trace_iterable


class EvaluatorTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def test_invalid_python_regex_in_constraint_does_not_crash_regression_check(self) -> None:
        from guardian.evaluator import evaluate_a_box_case
        from guardian.patch_parser import normalize_proposal

        record = {
            "id": "repair_regex",
            "qid": "Q1",
            "property": "P1",
            "track": "A_BOX",
            "repair_target": {"action": "UPDATE", "old_value": ["old"], "new_value": ["new"]},
            "classification": {"class": "TypeA", "subtype": "FORMAT_NORMALIZATION"},
        }
        world_state = {
            "L1_ego_node": {"properties": {"P1": ["new"]}},
            "L4_constraints": {
                "constraints": [
                    {
                        "constraint_type": {"qid": "Q21502404"},
                        "qualifiers": [{"property_id": "P1793", "values": [r"^\p{Lu}+$"]}],
                    }
                ]
            },
        }
        proposal = normalize_proposal(
            {
                "case_id": "repair_regex",
                "target": {"qid": "Q1", "pid": "P1"},
                "ops": [{"op": "SET", "pid": "P1", "value": "new"}],
                "rationale": "Normalize format.",
                "provenance": [{"kind": "KG", "node_id": "P1"}],
                "uncertainty": {"confidence": 0.8},
            }
        )

        trace = evaluate_a_box_case(record, world_state, proposal, {}, {}, "mid", None)

        self.assertTrue(trace["proposal_executable"])
        self.assertEqual(trace["details"]["supported_violations_before"], 1)
        self.assertEqual(trace["details"]["supported_violations_after"], 1)

    def test_a_box_and_t_box_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            world_state_path = root / "world_state.json"
            a_box_path = root / "a_box.jsonl"
            t_box_path = root / "t_box.jsonl"
            track_path = root / "track.jsonl"
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
                    "popularity": {"score": 0.5},
                    "context_ref": {"world_state_id": "repair_case"},
                    "classification": {"class": "TypeB", "subtype": "LOCAL_NEIGHBOR_IDS"},
                    "build": {}
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
                                    "qualifiers": [{"property_id": "P2305", "values": ["Q5", "Q43229"]}]
                                }
                            ]
                        }
                    },
                    "persistence_check": {},
                    "popularity": {"score": 0.8},
                    "context_ref": {"world_state_id": "reform_case"},
                    "classification": {"class": "T_BOX", "subtype": "RELAXATION_SET_EXPANSION"},
                    "build": {}
                }
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
                                "properties": {"P31": ["Q2"]}
                            },
                            "L2_labels": {},
                            "L3_neighborhood": {"outgoing_edges": []},
                            "L4_constraints": {"constraints": []}
                        },
                        "reform_case": {
                            "L1_ego_node": {
                                "qid": "Q2",
                                "label": "Entity",
                                "description": "desc",
                                "properties": {}
                            },
                            "L2_labels": {},
                            "L3_neighborhood": {"outgoing_edges": []},
                            "L4_constraints": {"constraints": []},
                            "constraint_change_context": {}
                        }
                    },
                    fh,
                )
            self._write_jsonl(
                a_box_path,
                [
                    {
                        "case_id": "repair_case",
                        "target": {"qid": "Q1", "pid": "P31"},
                        "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}],
                        "rationale": "Replace the invalid target value with the historical repair value.",
                        "provenance": [{"kind": "KG", "node_id": "Q5", "snippet": "historical value"}],
                        "uncertainty": {"confidence": 0.1, "notes": "Historical target is explicit."},
                    }
                ],
            )
            self._write_jsonl(
                t_box_path,
                [
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
                                    "qualifiers": [{"property_id": "P2305", "values": ["Q5", "Q43229"]}]
                                }
                            ]
                        },
                        "rationale": "Expand the allowed set to match the historical repair.",
                        "provenance": [
                            {"kind": "KG", "node_id": "Q21510859", "snippet": "historical constraint family"}
                        ],
                        "uncertainty": {
                            "confidence": 0.15,
                            "notes": "Signature order is deterministic after normalization.",
                        },
                    }
                ],
            )
            self._write_jsonl(
                track_path,
                [
                    {"case_id": "repair_case", "predicted_track": "A_BOX"},
                    {"case_id": "reform_case", "predicted_track": "T_BOX"},
                ],
            )
            selection_manifest_path.write_text(
                json.dumps({"selected_case_ids": ["reform_case"]}),
                encoding="utf-8",
            )

            traces, summary = evaluate_benchmark(
                classified_path=classified_path,
                world_state_path=world_state_path,
                a_box_proposals_path=a_box_path,
                t_box_proposals_path=t_box_path,
                track_diagnoses_path=track_path,
                selection_manifest_path=selection_manifest_path,
            )

            self.assertEqual(len(traces), 1)
            self.assertEqual(summary["counts"]["cases"], 1)
            self.assertEqual(summary["counts"]["accepted"], 1)
            self.assertEqual(summary["counts"]["track_diagnosis_exact_match"], 1)
            self.assertEqual(summary["inputs"]["selection_manifest"], str(selection_manifest_path))

    def test_evaluate_benchmark_invokes_progress_callback_per_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            world_state_path = root / "world_state.json"
            a_box_path = root / "a_box.jsonl"
            track_path = root / "track.jsonl"
            progress_case_ids: list[str] = []

            self._write_jsonl(
                classified_path,
                [
                    {
                        "id": "repair_case_1",
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
                        "id": "repair_case_2",
                        "qid": "Q2",
                        "property": "P31",
                        "track": "A_BOX",
                        "labels_en": {},
                        "violation_context": {"value": ["Q3"]},
                        "repair_target": {"action": "UPDATE", "old_value": ["Q3"], "new_value": ["Q6"]},
                        "persistence_check": {},
                        "popularity": {"score": 0.6},
                        "classification": {"class": "TypeB", "subtype": "LOCAL_NEIGHBOR_IDS"},
                    },
                ],
            )
            world_state_path.write_text(
                json.dumps(
                    {
                        "repair_case_1": {
                            "L1_ego_node": {"properties": {"P31": ["Q2"]}},
                            "L4_constraints": {"constraints": []},
                        },
                        "repair_case_2": {
                            "L1_ego_node": {"properties": {"P31": ["Q3"]}},
                            "L4_constraints": {"constraints": []},
                        },
                    }
                ),
                encoding="utf-8",
            )
            self._write_jsonl(
                a_box_path,
                [
                    {
                        "case_id": "repair_case_1",
                        "target": {"qid": "Q1", "pid": "P31"},
                        "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}],
                        "rationale": "Historical repair value.",
                        "provenance": [{"kind": "KG", "node_id": "Q5"}],
                        "uncertainty": {"confidence": 0.1},
                    },
                    {
                        "case_id": "repair_case_2",
                        "target": {"qid": "Q2", "pid": "P31"},
                        "ops": [{"op": "SET", "pid": "P31", "value": "Q6"}],
                        "rationale": "Historical repair value.",
                        "provenance": [{"kind": "KG", "node_id": "Q6"}],
                        "uncertainty": {"confidence": 0.1},
                    },
                ],
            )
            self._write_jsonl(
                track_path,
                [
                    {"case_id": "repair_case_1", "predicted_track": "A_BOX", "confidence": "high"},
                    {"case_id": "repair_case_2", "predicted_track": "A_BOX", "confidence": "high"},
                ],
            )

            traces, summary = evaluate_benchmark(
                classified_path=classified_path,
                world_state_path=world_state_path,
                a_box_proposals_path=a_box_path,
                track_diagnoses_path=track_path,
                progress_callback=lambda trace: progress_case_ids.append(trace["case_id"]),
            )

            self.assertEqual(summary["counts"]["cases"], 2)
            self.assertEqual([trace["case_id"] for trace in traces], ["repair_case_1", "repair_case_2"])
            self.assertEqual(progress_case_ids, ["repair_case_1", "repair_case_2"])

    def test_evaluate_benchmark_accepts_preloaded_classified_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            world_state_path = root / "world_state.json"
            a_box_path = root / "a_box.jsonl"
            track_path = root / "track.jsonl"

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
                }
            ]
            self._write_jsonl(classified_path, classified_rows)
            world_state_path.write_text(
                json.dumps(
                    {
                        "repair_case": {
                            "L1_ego_node": {"properties": {"P31": ["Q2"]}},
                            "L4_constraints": {"constraints": []},
                        }
                    }
                ),
                encoding="utf-8",
            )
            self._write_jsonl(
                a_box_path,
                [
                    {
                        "case_id": "repair_case",
                        "target": {"qid": "Q1", "pid": "P31"},
                        "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}],
                        "rationale": "Historical repair value.",
                        "provenance": [{"kind": "KG", "node_id": "Q5"}],
                        "uncertainty": {"confidence": 0.1},
                    }
                ],
            )
            self._write_jsonl(
                track_path,
                [
                    {"case_id": "repair_case", "predicted_track": "A_BOX", "confidence": "high"},
                ],
            )

            traces, summary = evaluate_benchmark(
                classified_path=root / "unused_filtered_subset.jsonl",
                classified_records=classified_rows,
                classified_input_path=classified_path,
                world_state_path=world_state_path,
                a_box_proposals_path=a_box_path,
                track_diagnoses_path=track_path,
            )

            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0]["case_id"], "repair_case")
            self.assertEqual(summary["counts"]["cases"], 1)
            self.assertEqual(summary["inputs"]["classified_benchmark"], str(classified_path))

    def test_t_box_semantic_only_match_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            world_state_path = root / "world_state.json"
            t_box_path = root / "t_box.jsonl"
            track_path = root / "track.jsonl"
            manifest_path = root / "manifest.jsonl"

            self._write_jsonl(
                classified_path,
                [
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
                        "popularity": {"score": 0.8},
                        "classification": {"class": "T_BOX", "subtype": "RELAXATION_SET_EXPANSION"},
                    }
                ],
            )
            world_state_path.write_text(
                json.dumps({"reform_case": {"L4_constraints": {"constraints": []}}}),
                encoding="utf-8",
            )
            self._write_jsonl(
                t_box_path,
                [
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
                                    "qualifiers": [{"property_id": "P2305", "values": ["Q5"]}],
                                }
                            ],
                        },
                        "rationale": "Use the same reform family as the historical edit.",
                        "provenance": [{"kind": "KG", "node_id": "Q21510859"}],
                        "uncertainty": {"confidence": 0.4},
                    }
                ],
            )
            self._write_jsonl(track_path, [{"case_id": "reform_case", "predicted_track": "T_BOX"}])
            self._write_jsonl(
                manifest_path,
                [
                    {
                        "case_id": "reform_case",
                        "ablation_bundle": "minimal_case",
                        "task_type": "proposal",
                        "parse_status": "normalized",
                        "usage": {"total_tokens": 10},
                    },
                    {
                        "case_id": "reform_case",
                        "ablation_bundle": "minimal_case",
                        "task_type": "track_diagnosis",
                        "parse_status": "normalized",
                        "usage": {"total_tokens": 5},
                    },
                ],
            )

            traces, summary = evaluate_benchmark(
                classified_path=classified_path,
                world_state_path=world_state_path,
                t_box_proposals_path=t_box_path,
                track_diagnoses_path=track_path,
                run_manifest_path=manifest_path,
                ablation_bundle="minimal_case",
            )

            self.assertEqual(len(traces), 1)
            trace = traces[0]
            self.assertFalse(trace["accepted"])
            self.assertEqual(trace["metrics"]["functional_success"], 0.0)
            self.assertEqual(trace["metrics"]["semantic_success"], 1.0)
            self.assertEqual(trace["metrics"]["semantic_family_success"], 1.0)
            self.assertEqual(trace["comparison"]["exact_action_match"], True)
            self.assertEqual(trace["comparison"]["literal_action_match"], True)
            self.assertEqual(trace["comparison"]["exact_signature_match"], False)
            self.assertEqual(trace["comparison"]["semantic_family_match"], True)
            self.assertEqual(trace["comparison"]["changed_constraint_type_hit"], True)
            self.assertTrue(trace["semantic_success"])
            self.assertEqual(summary["overall_metrics"]["semantic_success_rate"], 1.0)
            self.assertEqual(summary["overall_metrics"]["semantic_family_success_rate"], 1.0)
            self.assertEqual(summary["overall_metrics"]["accepted_rate"], 0.0)
            self.assertEqual(summary["overall_metrics"]["metric_applicability"]["semantic_success"], 1)
            self.assertEqual(summary["overall_metrics"]["metric_applicability"]["semantic_family_success"], 1)
            self.assertEqual(summary["overall_metrics"]["metric_applicability"]["signature_after_jaccard"], 1)

    def test_t_box_exact_signature_without_exact_action_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            world_state_path = root / "world_state.json"
            t_box_path = root / "t_box.jsonl"

            self._write_jsonl(
                classified_path,
                [
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
                        "popularity": {"score": 0.8},
                        "classification": {"class": "T_BOX", "subtype": "RELAXATION_SET_EXPANSION"},
                    }
                ],
            )
            world_state_path.write_text(
                json.dumps(
                    {
                        "reform_case": {
                            "L1_ego_node": {"properties": {"P31": ["Q5"]}},
                            "L4_constraints": {"constraints": []},
                        }
                    }
                ),
                encoding="utf-8",
            )
            self._write_jsonl(
                t_box_path,
                [
                    {
                        "case_id": "reform_case",
                        "target": {"pid": "P31", "constraint_type_qid": "Q21510859"},
                        "proposal": {
                            "action": "SCHEMA_UPDATE",
                            "signature_after": [
                                {
                                    "constraint_qid": "Q21510859",
                                    "snaktype": "VALUE",
                                    "rank": "normal",
                                    "qualifiers": [{"property_id": "P2305", "values": ["Q5", "Q43229"]}],
                                }
                            ],
                        },
                        "rationale": "Exact signature but the wrong reform action.",
                        "provenance": [{"kind": "KG", "node_id": "Q21510859"}],
                        "uncertainty": {"confidence": 0.3},
                    }
                ],
            )

            traces, _summary = evaluate_benchmark(
                classified_path=classified_path,
                world_state_path=world_state_path,
                t_box_proposals_path=t_box_path,
            )

            trace = traces[0]
            self.assertFalse(trace["accepted"])
            self.assertEqual(trace["comparison"]["exact_action_match"], False)
            self.assertEqual(trace["comparison"]["exact_signature_match"], True)
            self.assertEqual(trace["comparison"]["semantic_family_match"], False)
            self.assertEqual(trace["metrics"]["functional_success"], 0.0)
            self.assertEqual(trace["metrics"]["exact_historical_agreement"], 0.0)
            self.assertEqual(trace["metrics"]["semantic_success"], 0.0)
            self.assertEqual(trace["metrics"]["semantic_family_success"], 0.0)

    def test_t_box_schema_update_can_score_family_level_semantic_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            world_state_path = root / "world_state.json"
            t_box_path = root / "t_box.jsonl"

            self._write_jsonl(
                classified_path,
                [
                    {
                        "id": "reform_case",
                        "qid": "Q2",
                        "property": "P31",
                        "track": "T_BOX",
                        "labels_en": {},
                        "violation_context": {"report_violation_type": "One of"},
                        "repair_target": {
                            "constraint_delta": {
                                "changed_constraint_types": ["Q21510859"],
                                "signature_before": [
                                    {
                                        "constraint_qid": "Q21510859",
                                        "snaktype": "VALUE",
                                        "rank": "normal",
                                        "qualifiers": [{"property_id": "P2305", "values": ["Q5"]}],
                                    }
                                ],
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
                        "popularity": {"score": 0.8},
                        "classification": {"class": "T_BOX", "subtype": "SCHEMA_UPDATE"},
                    }
                ],
            )
            world_state_path.write_text(
                json.dumps(
                    {
                        "reform_case": {
                            "L1_ego_node": {"properties": {"P31": ["Q5"]}},
                            "L4_constraints": {"constraints": []},
                        }
                    }
                ),
                encoding="utf-8",
            )
            self._write_jsonl(
                t_box_path,
                [
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
                                    "qualifiers": [{"property_id": "P2305", "values": ["Q5", "Q43229"]}],
                                }
                            ],
                        },
                        "rationale": "Use the narrower set-expansion family that matches the historical delta.",
                        "provenance": [{"kind": "KG", "node_id": "Q21510859"}],
                        "uncertainty": {"confidence": 0.3},
                    }
                ],
            )

            traces, summary = evaluate_benchmark(
                classified_path=classified_path,
                world_state_path=world_state_path,
                t_box_proposals_path=t_box_path,
            )

            trace = traces[0]
            self.assertFalse(trace["accepted"])
            self.assertEqual(trace["comparison"]["exact_action_match"], False)
            self.assertEqual(trace["comparison"]["semantic_family_match"], True)
            self.assertEqual(trace["details"]["historical_semantic_family"], "set_relaxation")
            self.assertEqual(trace["details"]["proposal_action_family"], "set_relaxation")
            self.assertEqual(trace["metrics"]["exact_historical_agreement"], 0.0)
            self.assertEqual(trace["metrics"]["semantic_success"], 1.0)
            self.assertEqual(trace["metrics"]["semantic_family_success"], 1.0)
            self.assertEqual(summary["overall_metrics"]["semantic_success_rate"], 1.0)
            self.assertEqual(summary["overall_metrics"]["accepted_rate"], 0.0)

    def test_t_box_semantic_success_requires_historical_target_constraint_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            world_state_path = root / "world_state.json"
            t_box_path = root / "t_box.jsonl"

            self._write_jsonl(
                classified_path,
                [
                    {
                        "id": "reform_case",
                        "qid": "Q2",
                        "property": "P31",
                        "track": "T_BOX",
                        "labels_en": {},
                        "violation_context": {"report_violation_type": "One of"},
                        "repair_target": {
                            "constraint_delta": {
                                "changed_constraint_types": ["Q21510859", "Q52004125"],
                                "signature_before": [
                                    {
                                        "constraint_qid": "Q21510859",
                                        "snaktype": "VALUE",
                                        "rank": "normal",
                                        "qualifiers": [{"property_id": "P2305", "values": ["Q5"]}],
                                    },
                                    {
                                        "constraint_qid": "Q52004125",
                                        "snaktype": "VALUE",
                                        "rank": "normal",
                                        "qualifiers": [{"property_id": "P2305", "values": ["Q29934200"]}],
                                    },
                                ],
                                "signature_after": [
                                    {
                                        "constraint_qid": "Q21510859",
                                        "snaktype": "VALUE",
                                        "rank": "normal",
                                        "qualifiers": [{"property_id": "P2305", "values": ["Q5", "Q43229"]}],
                                    },
                                    {
                                        "constraint_qid": "Q52004125",
                                        "snaktype": "VALUE",
                                        "rank": "normal",
                                        "qualifiers": [{"property_id": "P2305", "values": ["Q29934200"]}],
                                    },
                                ],
                            }
                        },
                        "persistence_check": {},
                        "popularity": {"score": 0.8},
                        "classification": {"class": "T_BOX", "subtype": "SCHEMA_UPDATE"},
                    }
                ],
            )
            world_state_path.write_text(
                json.dumps(
                    {
                        "reform_case": {
                            "L1_ego_node": {"properties": {"P31": ["Q5"]}},
                            "L4_constraints": {"constraints": []},
                        }
                    }
                ),
                encoding="utf-8",
            )
            self._write_jsonl(
                t_box_path,
                [
                    {
                        "case_id": "reform_case",
                        "target": {"pid": "P31", "constraint_type_qid": "Q52004125"},
                        "proposal": {
                            "action": "SCHEMA_UPDATE",
                            "signature_after": [
                                {
                                    "constraint_qid": "Q52004125",
                                    "snaktype": "VALUE",
                                    "rank": "normal",
                                    "qualifiers": [{"property_id": "P2305", "values": ["Q29934200"]}],
                                }
                            ],
                        },
                        "rationale": "Edits a different changed constraint family than the historical target.",
                        "provenance": [{"kind": "KG", "node_id": "Q52004125"}],
                        "uncertainty": {"confidence": 0.3},
                    }
                ],
            )

            traces, summary = evaluate_benchmark(
                classified_path=classified_path,
                world_state_path=world_state_path,
                t_box_proposals_path=t_box_path,
            )

            trace = traces[0]
            self.assertTrue(trace["comparison"]["changed_constraint_type_hit"])
            self.assertFalse(trace["comparison"]["target_constraint_match"])
            self.assertFalse(trace["comparison"]["semantic_family_match"])
            self.assertEqual(trace["details"]["historical_target_constraint_qid"], "Q21510859")
            self.assertEqual(trace["metrics"]["semantic_success"], 0.0)
            self.assertEqual(trace["metrics"]["semantic_family_success"], 0.0)
            self.assertEqual(summary["overall_metrics"]["semantic_success_rate"], 0.0)

    def test_t_box_semantic_success_when_target_constraint_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            world_state_path = root / "world_state.json"
            t_box_path = root / "t_box.jsonl"

            self._write_jsonl(
                classified_path,
                [
                    {
                        "id": "reform_case",
                        "qid": "Q2",
                        "property": "P31",
                        "track": "T_BOX",
                        "labels_en": {},
                        "violation_context": {},
                        "repair_target": {
                            "constraint_delta": {
                                "changed_constraint_types": [],
                                "hash_before": "before",
                                "hash_after": "after",
                            }
                        },
                        "persistence_check": {},
                        "popularity": {"score": 0.8},
                        "classification": {"class": "T_BOX", "subtype": "RELAXATION_SET_EXPANSION"},
                    }
                ],
            )
            world_state_path.write_text(
                json.dumps(
                    {
                        "reform_case": {
                            "L1_ego_node": {"properties": {"P31": ["Q5"]}},
                            "L4_constraints": {"constraints": []},
                        }
                    }
                ),
                encoding="utf-8",
            )
            self._write_jsonl(
                t_box_path,
                [
                    {
                        "case_id": "reform_case",
                        "target": {"pid": "P31", "constraint_type_qid": "Q21510859"},
                        "proposal": {
                            "action": "RELAXATION_SET_EXPANSION",
                            "signature_after": [],
                        },
                        "rationale": "The exact target constraint family is not available, but the reform family matches.",
                        "provenance": [{"kind": "KG", "node_id": "Q21510859"}],
                        "uncertainty": {"confidence": 0.3},
                    }
                ],
            )

            traces, summary = evaluate_benchmark(
                classified_path=classified_path,
                world_state_path=world_state_path,
                t_box_proposals_path=t_box_path,
            )

            trace = traces[0]
            self.assertFalse(trace["accepted"])
            self.assertEqual(trace["comparison"]["changed_constraint_type_hit"], None)
            self.assertEqual(trace["comparison"]["target_constraint_match"], None)
            self.assertEqual(trace["comparison"]["semantic_family_match"], True)
            self.assertEqual(trace["metrics"]["semantic_success"], 1.0)
            self.assertEqual(trace["metrics"]["semantic_family_success"], 1.0)
            self.assertEqual(trace["metrics"]["exact_signature_match"], 0.0)
            self.assertEqual(trace["metrics"]["changed_constraint_type_hit"], None)
            self.assertEqual(trace["metrics"]["signature_after_jaccard"], None)
            self.assertEqual(trace["metrics"]["t_box_target_constraint_match"], None)
            self.assertEqual(summary["overall_metrics"]["semantic_success_rate"], 1.0)
            self.assertEqual(summary["overall_metrics"]["exact_signature_match_rate"], 0.0)
            self.assertEqual(summary["overall_metrics"]["changed_constraint_type_hit_rate"], None)
            self.assertEqual(summary["overall_metrics"]["signature_after_jaccard_mean"], None)
            self.assertEqual(summary["overall_metrics"]["t_box_target_constraint_match_rate"], None)

    def test_summary_tracks_t_box_proxy_metric_applicability(self) -> None:
        summary = summarize_trace_iterable(
            [
                {
                    "case_id": "reform_case",
                    "accepted": False,
                    "proposal_present": True,
                    "proposal_executable": True,
                    "parse_status": "normalized",
                    "classification_class": "T_BOX",
                    "classification_subtype": "RELAXATION_SET_EXPANSION",
                    "track": "T_BOX",
                    "ablation_bundle": "minimal_case",
                    "popularity_bucket": "mid",
                    "metrics": {
                        "functional_success": 0.0,
                        "exact_historical_agreement": 0.0,
                        "semantic_success": 1.0,
                        "semantic_family_success": 1.0,
                        "information_preservation": None,
                        "provenance_completeness": 1.0,
                        "auditability_complete": 1.0,
                        "token_usage": {"total_tokens": 30},
                        "conversion_rate": 0.0,
                        "tokens_to_fix": None,
                        "exact_action_match": 1.0,
                        "exact_signature_match": 0.0,
                        "changed_constraint_type_hit": 1.0,
                        "signature_after_jaccard": 0.5,
                        "t_box_target_constraint_match": 1.0,
                        "proposal_admits_current_values": 1.0,
                    },
                    "details": {"proposal_admits_current_values": True},
                    "track_diagnosis": {"present": True, "exact_track_match": True, "ambiguous_prediction": False},
                }
            ],
            {"classified_benchmark": "stub"},
        )

        self.assertEqual(summary["counts"]["exact_action_match_applicable"], 1)
        self.assertEqual(summary["counts"]["conversion_rate_applicable"], 1)
        self.assertEqual(summary["counts"]["signature_after_jaccard_applicable"], 1)
        self.assertEqual(summary["counts"]["semantic_family_success_applicable"], 1)
        self.assertEqual(summary["counts"]["t_box_target_constraint_match_applicable"], 1)
        self.assertEqual(summary["overall_metrics"]["auditability_complete_rate"], 1.0)
        self.assertEqual(summary["overall_metrics"]["conversion_rate"], 0.0)
        self.assertEqual(summary["overall_metrics"]["exact_action_match_rate"], 1.0)
        self.assertEqual(summary["overall_metrics"]["semantic_family_success_rate"], 1.0)
        self.assertEqual(summary["overall_metrics"]["signature_after_jaccard_mean"], 0.5)
        self.assertEqual(summary["overall_metrics"]["t_box_target_constraint_match_rate"], 1.0)
        self.assertEqual(summary["overall_metrics"]["proposal_admits_current_values_rate"], 1.0)
        self.assertEqual(summary["overall_metrics"]["metric_applicability"]["proposal_admits_current_values"], 1)
        self.assertEqual(summary["overall_metrics"]["metric_applicability"]["semantic_family_success"], 1)
        self.assertEqual(summary["overall_metrics"]["metric_applicability"]["t_box_target_constraint_match"], 1)

    def test_summary_tracks_a_box_proxy_metric_applicability(self) -> None:
        summary = summarize_trace_iterable(
            [
                {
                    "case_id": "repair_case",
                    "accepted": False,
                    "proposal_present": True,
                    "proposal_executable": True,
                    "parse_status": "normalized",
                    "classification_class": "TypeB",
                    "classification_subtype": "LOCAL_NEIGHBOR_IDS",
                    "track": "A_BOX",
                    "ablation_bundle": "minimal_case",
                    "popularity_bucket": "mid",
                    "metrics": {
                        "functional_success": 1.0,
                        "exact_historical_agreement": 0.0,
                        "semantic_success": None,
                        "information_preservation": 1.0,
                        "provenance_completeness": 1.0,
                        "auditability_complete": 1.0,
                        "token_usage": {"total_tokens": 18},
                        "conversion_rate": 0.0,
                        "tokens_to_fix": 18.0,
                        "a_box_exact_action_match": 1.0,
                        "a_box_exact_value_match": 0.0,
                        "a_box_regression_pass": 1.0,
                    },
                    "details": {},
                    "track_diagnosis": {"present": True, "exact_track_match": True, "ambiguous_prediction": False},
                }
            ],
            {"classified_benchmark": "stub"},
        )

        self.assertEqual(summary["counts"]["a_box_exact_action_match_applicable"], 1)
        self.assertEqual(summary["counts"]["a_box_exact_value_match_applicable"], 1)
        self.assertEqual(summary["counts"]["a_box_regression_pass_applicable"], 1)
        self.assertEqual(summary["overall_metrics"]["a_box_exact_action_match_rate"], 1.0)
        self.assertEqual(summary["overall_metrics"]["a_box_exact_value_match_rate"], 0.0)
        self.assertEqual(summary["overall_metrics"]["a_box_regression_pass_rate"], 1.0)
        self.assertEqual(summary["overall_metrics"]["metric_applicability"]["a_box_exact_action_match"], 1)
        self.assertEqual(summary["overall_metrics"]["metric_applicability"]["a_box_exact_value_match"], 1)
        self.assertEqual(summary["overall_metrics"]["metric_applicability"]["a_box_regression_pass"], 1)

    def test_summary_exposes_parse_error_counts(self) -> None:
        summary = summarize_trace_iterable(
            [
                {
                    "case_id": "repair_case",
                    "accepted": False,
                    "proposal_present": False,
                    "proposal_executable": False,
                    "parse_status": "parse_error",
                    "classification_class": "TypeC",
                    "classification_subtype": "EXTERNAL",
                    "track": "A_BOX",
                    "ablation_bundle": "minimal_case",
                    "popularity_bucket": "mid",
                    "metrics": {
                        "functional_success": 0.0,
                        "exact_historical_agreement": 0.0,
                        "semantic_success": None,
                        "information_preservation": 0.0,
                        "provenance_completeness": 0.0,
                        "auditability_complete": 0.0,
                        "token_usage": {"total_tokens": 30},
                        "conversion_rate": 0.0,
                        "tokens_to_fix": None,
                    },
                    "details": {"parser_error": "provenance must be a list."},
                    "track_diagnosis": {"present": True, "exact_track_match": True, "ambiguous_prediction": False},
                }
            ],
            {"classified_benchmark": "stub"},
        )

        self.assertEqual(summary["counts"]["proposal_parse_error"], 1)
        self.assertEqual(summary["parse_errors"]["proposal_parse_error_count"], 1)
        self.assertEqual(summary["parse_errors"]["by_message"]["provenance must be a list."], 1)
        self.assertEqual(summary["overall_metrics"]["proposal_parse_error_count"], 1)
        self.assertEqual(summary["overall_metrics"]["proposal_parse_error_rate"], 1.0)

    def test_summary_exposes_request_error_counts(self) -> None:
        summary = summarize_trace_iterable(
            [
                {
                    "case_id": "repair_case",
                    "accepted": False,
                    "proposal_present": False,
                    "proposal_executable": False,
                    "parse_status": "request_error",
                    "classification_class": "TypeC",
                    "classification_subtype": "EXTERNAL",
                    "track": "A_BOX",
                    "ablation_bundle": "minimal_case",
                    "popularity_bucket": "mid",
                    "metrics": {
                        "functional_success": 0.0,
                        "exact_historical_agreement": 0.0,
                        "semantic_success": None,
                        "information_preservation": 0.0,
                        "provenance_completeness": 0.0,
                        "auditability_complete": 0.0,
                        "token_usage": {"total_tokens": 30},
                        "conversion_rate": None,
                        "tokens_to_fix": None,
                    },
                    "details": {"provider_error": "OpenAI batch request failed (500)."},
                    "track_diagnosis": {
                        "present": False,
                        "exact_track_match": False,
                        "ambiguous_prediction": False,
                        "parse_status": "request_error",
                        "provider_error": "OpenAI diagnosis request failed (500).",
                    },
                }
            ],
            {"classified_benchmark": "stub"},
        )

        self.assertEqual(summary["counts"]["proposal_request_error"], 1)
        self.assertEqual(summary["counts"]["track_diagnosis_request_error"], 1)
        self.assertEqual(summary["request_errors"]["proposal_request_error_count"], 1)
        self.assertEqual(summary["request_errors"]["track_diagnosis_request_error_count"], 1)
        self.assertEqual(summary["overall_metrics"]["proposal_request_error_count"], 1)
        self.assertEqual(summary["overall_metrics"]["proposal_request_error_rate"], 1.0)
        self.assertEqual(summary["overall_metrics"]["track_diagnosis_request_error_count"], 1)
        self.assertEqual(summary["overall_metrics"]["track_diagnosis_request_error_rate"], 1.0)

    def test_a_box_acceptance_requires_auditability_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            world_state_path = root / "world_state.json"
            a_box_path = root / "a_box.jsonl"

            self._write_jsonl(
                classified_path,
                [
                    {
                        "id": "repair_case",
                        "qid": "Q1",
                        "property": "P31",
                        "track": "A_BOX",
                        "labels_en": {},
                        "violation_context": {"value": ["Q2"]},
                        "repair_target": {"action": "UPDATE", "old_value": ["Q2"], "new_value": ["Q5"]},
                        "persistence_check": {},
                        "popularity": {"score": 0.5},
                        "classification": {"class": "TypeB", "subtype": "LOCAL_NEIGHBOR_IDS"},
                    }
                ],
            )
            world_state_path.write_text(
                json.dumps(
                    {
                        "repair_case": {
                            "L1_ego_node": {"properties": {"P31": ["Q2"]}},
                            "L4_constraints": {"constraints": []},
                        }
                    }
                ),
                encoding="utf-8",
            )
            self._write_jsonl(
                a_box_path,
                [
                    {
                        "case_id": "repair_case",
                        "target": {"qid": "Q1", "pid": "P31"},
                        "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}],
                    }
                ],
            )

            traces, _summary = evaluate_benchmark(
                classified_path=classified_path,
                world_state_path=world_state_path,
                a_box_proposals_path=a_box_path,
            )

            trace = traces[0]
            self.assertFalse(trace["accepted"])
            self.assertEqual(trace["metrics"]["auditability_complete"], 0.0)
            self.assertEqual(trace["details"]["rationale_present"], False)
            self.assertEqual(trace["details"]["provenance_present"], False)
            self.assertEqual(trace["details"]["uncertainty_present"], False)

    def test_tokens_to_fix_sums_diagnosis_and_proposal_tokens_for_accepted_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            world_state_path = root / "world_state.json"
            a_box_path = root / "a_box.jsonl"
            track_path = root / "track.jsonl"
            manifest_path = root / "manifest.jsonl"

            self._write_jsonl(
                classified_path,
                [
                    {
                        "id": "repair_case",
                        "qid": "Q1",
                        "property": "P31",
                        "track": "A_BOX",
                        "labels_en": {},
                        "violation_context": {"value": ["Q2"]},
                        "repair_target": {"action": "UPDATE", "old_value": ["Q2"], "new_value": ["Q5"]},
                        "persistence_check": {},
                        "popularity": {"score": 0.5},
                        "classification": {"class": "TypeB", "subtype": "LOCAL_NEIGHBOR_IDS"},
                    }
                ],
            )
            world_state_path.write_text(
                json.dumps(
                    {
                        "repair_case": {
                            "L1_ego_node": {"properties": {"P31": ["Q2"]}},
                            "L4_constraints": {"constraints": []},
                        }
                    }
                ),
                encoding="utf-8",
            )
            self._write_jsonl(
                a_box_path,
                [
                    {
                        "case_id": "repair_case",
                        "target": {"qid": "Q1", "pid": "P31"},
                        "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}],
                        "rationale": "Historical repair value.",
                        "provenance": [{"kind": "KG", "node_id": "Q5"}],
                        "uncertainty": {"confidence": 0.1},
                    }
                ],
            )
            self._write_jsonl(track_path, [{"case_id": "repair_case", "predicted_track": "A_BOX"}])
            self._write_jsonl(
                manifest_path,
                [
                    {
                        "case_id": "repair_case",
                        "ablation_bundle": "minimal_case",
                        "task_type": "proposal",
                        "parse_status": "normalized",
                        "usage": {"total_tokens": 17},
                    },
                    {
                        "case_id": "repair_case",
                        "ablation_bundle": "minimal_case",
                        "task_type": "track_diagnosis",
                        "parse_status": "normalized",
                        "usage": {"total_tokens": 5},
                    },
                ],
            )

            traces, summary = evaluate_benchmark(
                classified_path=classified_path,
                world_state_path=world_state_path,
                a_box_proposals_path=a_box_path,
                track_diagnoses_path=track_path,
                run_manifest_path=manifest_path,
                ablation_bundle="minimal_case",
            )

            trace = traces[0]
            self.assertEqual(trace["metrics"]["conversion_rate"], 1.0)
            self.assertEqual(trace["metrics"]["tokens_to_fix"], 22)
            self.assertEqual(summary["overall_metrics"]["conversion_rate"], 1.0)
            self.assertEqual(summary["overall_metrics"]["tokens_to_fix_mean"], 22.0)


if __name__ == "__main__":
    unittest.main()
