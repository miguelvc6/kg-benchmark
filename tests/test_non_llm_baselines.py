import unittest

from guardian.evaluator import evaluate_a_box_case
from lib.non_llm_baselines import (
    constraint_only_typea_proposal,
    do_nothing_proposal,
    local_lookup_oracle_proposal,
    track_metrics,
    track_prediction_rows,
)


def _world_state(*, properties=None, constraints=None, description="") -> dict:
    return {
        "L1_ego_node": {
            "qid": "Q1",
            "label": "Focus",
            "description": description,
            "properties": properties or {},
        },
        "L2_labels": {"entities": {}},
        "L3_neighborhood": {"outgoing_edges": []},
        "L4_constraints": {"constraints": constraints or []},
    }


class NonLlmBaselineTests(unittest.TestCase):
    def test_constant_track_predictions_and_macro_f1(self) -> None:
        records = [
            {"id": "a1", "track": "A_BOX"},
            {"id": "a2", "track": "A_BOX"},
            {"id": "t1", "track": "T_BOX"},
        ]

        rows = track_prediction_rows(records, "majority_track")
        predictions = {row["case_id"]: row["predicted_track"] for row in rows}
        metrics = track_metrics(records, predictions)

        self.assertEqual({row["predicted_track"] for row in rows}, {"A_BOX"})
        self.assertEqual(metrics["confusion_matrix"]["A_BOX"]["A_BOX"], 2)
        self.assertEqual(metrics["confusion_matrix"]["T_BOX"]["A_BOX"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["t_box_miss_rate"], 1.0)

    def test_constraint_only_format_normalization_proposal(self) -> None:
        record = {
            "id": "repair_format",
            "qid": "Q1",
            "property": "P1",
            "track": "A_BOX",
            "repair_target": {"kind": "A_BOX", "action": "UPDATE", "old_value": ["ABC/"], "new_value": ["ABC"]},
            "classification": {
                "class": "TypeA",
                "subtype": "FORMAT_NORMALIZATION",
                "classification_rule_subfamily": "strip_trailing_slash",
                "diagnostics": {
                    "value_change_summary": {
                        "old_values": ["ABC/"],
                        "new_values": ["ABC"],
                        "old_unique": ["ABC/"],
                        "new_unique": ["ABC"],
                        "retained_unique_values": [],
                        "removed_unique_values": ["ABC/"],
                        "added_unique_values": ["ABC"],
                        "semantic_action": "REPLACE_1_TO_1",
                    }
                },
            },
        }

        proposal = constraint_only_typea_proposal(record, _world_state(properties={"P1": ["ABC/"]}))

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["ops"], [{"op": "SET", "pid": "P1", "value": "ABC", "rank": "normal"}])
        trace = evaluate_a_box_case(
            record,
            _world_state(properties={"P1": ["ABC"]}),
            _proposal_obj(proposal),
            {},
            {},
            "mid",
            None,
        )
        self.assertTrue(trace["accepted"])

    def test_constraint_only_set_membership_rejection_proposal(self) -> None:
        record = {
            "id": "repair_set",
            "qid": "Q1",
            "property": "P2",
            "track": "A_BOX",
            "violation_context": {"report_violation_type_normalized": "One of"},
            "repair_target": {
                "kind": "A_BOX",
                "action": "UPDATE",
                "old_value": ["Q2", "Q3"],
                "new_value": ["Q3"],
            },
            "classification": {
                "class": "TypeA",
                "subtype": "SET_MEMBERSHIP_REJECTION",
                "classification_rule_subfamily": "one_of",
                "diagnostics": {
                    "value_change_summary": {
                        "old_values": ["Q2", "Q3"],
                        "new_values": ["Q3"],
                        "old_unique": ["Q2", "Q3"],
                        "new_unique": ["Q3"],
                        "retained_unique_values": ["Q3"],
                        "removed_unique_values": ["Q2"],
                        "added_unique_values": [],
                        "semantic_action": "DELETE_SUBSET",
                    }
                },
            },
        }
        world_state = _world_state(
            properties={"P2": ["Q3"]},
            constraints=[
                {
                    "constraint_type": {"qid": "Q21510859"},
                    "qualifiers": [{"property_id": "P2305", "values": ["Q3"]}],
                }
            ],
        )

        proposal = constraint_only_typea_proposal(record, world_state)

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["ops"], [{"op": "SET", "pid": "P2", "value": "Q3", "rank": "normal"}])

    def test_local_lookup_derives_p8726_value(self) -> None:
        record = {
            "id": "repair_local",
            "qid": "Q1",
            "property": "P8726",
            "track": "A_BOX",
            "repair_target": {
                "kind": "A_BOX",
                "action": "CREATE",
                "old_value": ["MISSING"],
                "new_value": ["2007/si/483/made"],
            },
            "classification": {
                "class": "TypeB",
                "subtype": "LOCAL_TEXT_DERIVED",
                "diagnostics": {
                    "value_change_summary": {
                        "old_values": [],
                        "new_values": ["2007/si/483/made"],
                        "old_unique": [],
                        "new_unique": ["2007/si/483/made"],
                        "retained_unique_values": [],
                        "removed_unique_values": [],
                        "added_unique_values": ["2007/si/483/made"],
                        "semantic_action": "CREATE_FROM_MISSING",
                    }
                },
            },
        }

        proposal = local_lookup_oracle_proposal(
            record,
            _world_state(description="S.I. No. 483/2007 made under the relevant act."),
        )

        self.assertIsNotNone(proposal)
        self.assertEqual(
            proposal["ops"],
            [{"op": "SET", "pid": "P8726", "value": "2007/si/483/made", "rank": "normal"}],
        )

    def test_do_nothing_uses_pre_repair_target_state(self) -> None:
        record = {
            "id": "repair_noop",
            "qid": "Q1",
            "property": "P3",
            "track": "A_BOX",
            "repair_target": {"kind": "A_BOX", "action": "UPDATE", "old_value": ["Q2"], "new_value": ["Q3"]},
        }

        proposal = do_nothing_proposal(record)

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["ops"], [{"op": "SET", "pid": "P3", "value": "Q2", "rank": "normal"}])


def _proposal_obj(payload: dict):
    from guardian.patch_parser import normalize_proposal

    return normalize_proposal(payload)


if __name__ == "__main__":
    unittest.main()
