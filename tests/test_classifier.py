import unittest

from classifier import classify_one


def _base_repair(**overrides):
    repair = {
        "id": "case-1",
        "qid": "Q1",
        "property": "P1",
        "track": "A_BOX",
        "repair_target": {"kind": "A_BOX", "action": "UPDATE", "old_value": "Q1", "new_value": "Q99"},
        "violation_context": {"report_violation_type_normalized": "Type"},
    }
    repair.update(overrides)
    return repair


def _world_state(**overrides):
    world_state = {
        "L1_ego_node": {
            "qid": "Q1",
            "label": "Focus",
            "description": "Focus description",
            "properties": {},
        },
        "L2_labels": {"entities": {}},
        "L3_neighborhood": {"outgoing_edges": []},
        "L4_constraints": {"constraints": []},
    }
    world_state.update(overrides)
    return world_state


class ClassifierPhaseBTests(unittest.TestCase):
    def test_post_repair_target_property_edge_does_not_create_local_match(self):
        repair = _base_repair()
        world_state = _world_state(
            L3_neighborhood={"outgoing_edges": [{"property_id": "P1", "target_qid": "Q99"}]},
        )

        classification, _, _ = classify_one(repair, world_state)

        self.assertNotEqual(classification["class"], "TypeB")

    def test_non_target_l1_property_qid_can_produce_type_b(self):
        repair = _base_repair()
        world_state = _world_state(
            L1_ego_node={
                "qid": "Q1",
                "label": "Focus",
                "description": "Focus description",
                "properties": {"P2": ["Q99"], "P1": ["Q99"]},
            },
        )

        classification, _, _ = classify_one(repair, world_state)

        self.assertEqual(classification["class"], "TypeB")
        self.assertEqual(classification["subtype"], "LOCAL_FOCUS_NON_TARGET_PROPERTY")

    def test_l2_label_does_not_create_qid_match_without_local_reference(self):
        repair = _base_repair()
        world_state = _world_state(
            L2_labels={"entities": {"Q99": {"label": "Target", "description": "Only in L2"}}},
        )

        classification, _, _ = classify_one(repair, world_state)

        self.assertNotEqual(classification["class"], "TypeB")

    def test_short_literal_does_not_substring_match(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "CA", "new_value": "US"},
        )
        world_state = _world_state(
            L1_ego_node={
                "qid": "Q1",
                "label": "Focus",
                "description": "music venue",
                "properties": {"P2": ["MUSIC"]},
            },
        )

        classification, _, _ = classify_one(repair, world_state)

        self.assertNotEqual(classification["class"], "TypeB")

    def test_numeric_range_boundary_uses_p2313_p2312(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "9", "new_value": "10"},
        )
        world_state = _world_state(
            L4_constraints={
                "constraints": [
                    {
                        "constraint_type": {"qid": "Q21510860", "label": "range constraint"},
                        "qualifiers": [{"property_id": "P2313", "values": ["10"]}],
                    }
                ]
            },
        )

        classification, _, _ = classify_one(repair, world_state)

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "LOGICAL")

    def test_date_range_boundary_uses_p2310_p2311(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "1999-12-31", "new_value": "2000-01-01"},
        )
        world_state = _world_state(
            L4_constraints={
                "constraints": [
                    {
                        "constraint_type": {"qid": "Q21510860", "label": "range constraint"},
                        "qualifiers": [{"property_id": "P2310", "values": ["2000-01-01"]}],
                    }
                ]
            },
        )

        classification, _, _ = classify_one(repair, world_state)

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "LOGICAL")

    def test_format_update_type_a_only_for_simple_normalization(self):
        simple = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": " ABC ", "new_value": "ABC"},
            violation_context={"report_violation_type_normalized": "Format"},
        )
        non_deterministic = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "ABC", "new_value": "XYZ"},
            violation_context={"report_violation_type_normalized": "Format"},
        )
        world_state = _world_state(
            L4_constraints={
                "constraints": [
                    {
                        "constraint_type": {"qid": "Q21502404", "label": "format constraint"},
                        "qualifiers": [],
                    }
                ]
            },
        )

        simple_classification, _, _ = classify_one(simple, world_state)
        nondet_classification, _, _ = classify_one(non_deterministic, world_state)

        self.assertEqual(simple_classification["class"], "TypeA")
        self.assertNotEqual(nondet_classification["class"], "TypeA")

    def test_missing_truth_routes_to_unknown_missing_truth(self):
        repair = _base_repair(repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "Q1"})

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeC")
        self.assertEqual(classification["subtype"], "UNKNOWN_MISSING_TRUTH")

    def test_current_value_truth_fallback_is_quarantined(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "Q1"},
            persistence_check={"current_value_2026": "Q99"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeC")
        self.assertEqual(classification["subtype"], "UNKNOWN_CURRENT_VALUE_FALLBACK")

    def test_missing_world_state_routes_to_unknown_missing_world_state(self):
        repair = _base_repair()

        classification, _, _ = classify_one(repair, None)

        self.assertEqual(classification["class"], "TypeC")
        self.assertEqual(classification["subtype"], "UNKNOWN_MISSING_WORLD_STATE")

    def test_value_type_tbox_expansion_uses_p2308(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q21510865"],
                    "signature_before": [
                        {
                            "constraint_qid": "Q21510865",
                            "qualifiers": [{"property_id": "P2308", "values": ["Q5"]}],
                        }
                    ],
                    "signature_after": [
                        {
                            "constraint_qid": "Q21510865",
                            "qualifiers": [{"property_id": "P2308", "values": ["Q5", "Q6"]}],
                        }
                    ],
                },
            },
            violation_context={"report_violation_type_normalized": "Value type"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "T_BOX")
        self.assertEqual(classification["subtype"], "RELAXATION_SET_EXPANSION")

    def test_delete_under_single_value_conflict_is_not_high_confidence_rejection(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "DELETE", "old_value": "Q1"},
            violation_context={"report_violation_type_normalized": "Single value"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["subtype"], "DELETE_AMBIGUOUS")
        self.assertEqual(classification["confidence"], "low")


if __name__ == "__main__":
    unittest.main()
