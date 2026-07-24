import json
import logging
import tempfile
import unittest
from pathlib import Path

from classifier import WorldStateStore
from guardian.reasoning import (
    _contains_hidden_support_key,
    _few_shot_examples_from_bank,
    _load_support_bank_for_runner,
)


class FewShotRuntimeTests(unittest.TestCase):
    def test_new_support_bank_renders_neutral_leakage_checked_example(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank_path = root / "support-bank.json"
            bank_path.write_text(
                json.dumps(
                    {
                        "support_sets": {
                            "a_box_repair": [
                                {
                                    "case_id": "raw-support-case",
                                    "group_key": "ABOX|Q9|P31",
                                    "role": "ic_l_rule",
                                    "visible_example_id": "example_000001",
                                }
                            ],
                            "t_box_repair": [],
                            "track_diagnosis": [],
                        }
                    }
                ),
                encoding="utf-8",
            )
            manifest, case_ids = _load_support_bank_for_runner(bank_path)
            self.assertEqual(case_ids, {"raw-support-case"})
            support = {
                "id": "raw-support-case",
                "qid": "Q9",
                "property": "P31",
                "track": "A_BOX",
                "classification": {"class": "TypeA", "subtype": "TARGET_REQUIRED_CLAIM"},
                "repair_target": {"action": "UPDATE", "old_value": ["Q2"], "new_value": ["Q5"]},
                "violation_context": {"value": ["Q2"]},
            }
            evaluation = {
                "id": "evaluation-case",
                "qid": "Q1",
                "property": "P31",
                "track": "A_BOX",
                "classification": {"class": "TypeA"},
                "repair_target": {"action": "UPDATE", "old_value": ["Q2"], "new_value": ["Q3"]},
                "violation_context": {"value": ["Q2"]},
            }
            world_state_path = root / "world-state.jsonl"
            world_state_path.write_text(
                json.dumps(
                    {
                        "id": "raw-support-case",
                        "world_state": {
                            "L1_ego_node": {"qid": "Q9", "properties": {"P31": ["Q2"]}},
                            "L2_labels": {
                                "entities": {
                                    "Q2": {
                                        "label": "classification",
                                        "description": "visible domain vocabulary",
                                    }
                                }
                            },
                            "L3_neighborhood": {"outgoing_edges": []},
                            "L4_constraints": {"constraints": []},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            store = WorldStateStore(world_state_path, logging.getLogger(__name__))
            store.open()
            try:
                examples = _few_shot_examples_from_bank(
                    support_manifest=manifest,
                    eval_record=evaluation,
                    task="a_box_repair",
                    records_by_id={"raw-support-case": support},
                    world_store=store,
                    context_bundle="local_graph",
                    example_count=1,
                )
            finally:
                store.close()
            encoded = json.dumps(examples)
            self.assertEqual(examples[0]["visible_case_id"], "example_000001")
            self.assertNotIn("raw-support-case", encoded)
            self.assertIn('"classification"', encoded)
            self.assertNotIn('"repair_target"', encoded)

    def test_hidden_support_guard_checks_keys_not_visible_values(self) -> None:
        self.assertFalse(_contains_hidden_support_key({"label": "classification"}))
        self.assertFalse(_contains_hidden_support_key({"description": "repair_target"}))
        self.assertTrue(_contains_hidden_support_key({"nested": {"classification": {"class": "TypeA"}}}))
        self.assertTrue(_contains_hidden_support_key([{"repair_target": {"action": "UPDATE"}}]))


if __name__ == "__main__":
    unittest.main()
