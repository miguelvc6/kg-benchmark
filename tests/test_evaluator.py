import json
import tempfile
import unittest
from pathlib import Path

from guardian.evaluator import evaluate_benchmark


class EvaluatorTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

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
                        "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}]
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
                        }
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
                    },
                    {
                        "case_id": "repair_case_2",
                        "target": {"qid": "Q2", "pid": "P31"},
                        "ops": [{"op": "SET", "pid": "P31", "value": "Q6"}],
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


if __name__ == "__main__":
    unittest.main()
