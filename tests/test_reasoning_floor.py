import json
import tempfile
import unittest
from pathlib import Path

from guardian.model_provider import StaticResponseProvider
from guardian.reasoning import run_reasoning_floor


class ReasoningFloorTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def test_reasoning_floor_stub_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            world_state_path = root / "world_state.json"

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
                    "classification": {"class": "TypeB", "subtype": "LOCAL_NEIGHBOR_IDS"}
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
                    "popularity": {"score": 0.9},
                    "classification": {"class": "T_BOX", "subtype": "RELAXATION_SET_EXPANSION"}
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

            def resolver(metadata: dict) -> dict:
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
                                    "qualifiers": [{"property_id": "P2305", "values": ["Q5", "Q43229"]}]
                                }
                            ]
                        }
                    }
                return {
                    "case_id": "repair_case",
                    "target": {"qid": "Q1", "pid": "P31"},
                    "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}]
                }

            summary = run_reasoning_floor(
                classified_path=classified_path,
                world_state_path=world_state_path,
                output_dir=root / "outputs",
                provider=StaticResponseProvider(resolver),
                ablation_bundles=["minimal_case"],
            )
            self.assertIn("paper_summary", summary)
            self.assertEqual(summary["counts"]["cases"], 2)


if __name__ == "__main__":
    unittest.main()
