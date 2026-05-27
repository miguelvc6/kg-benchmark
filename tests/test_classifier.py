import unittest

from classifier import classify_one, local_context_buckets, match_truth_locally


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
    def test_prerepair_literal_not_reported_as_generic_focus_text(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "PHABRA/", "new_value": "PHABRA"}
        )
        world_state = _world_state()

        classification, _, _ = classify_one(repair, world_state)
        trace = classification["decision_trace"]
        local_step = next(step for step in trace if step.get("step") == "local_availability")
        evidence = local_step.get("evidence") or {}
        sources = {match.get("source") for match in evidence.get("matches", []) if isinstance(match, dict)}

        self.assertNotIn("FOCUS_TEXT", sources)

    def test_prerepair_qid_uses_explicit_source(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["Q2"], "new_value": ["Q2", "Q3"]}
        )
        world_state = _world_state()

        buckets, _ = local_context_buckets(repair, world_state)
        _, evidence = match_truth_locally(["Q2"], buckets)
        sources = {match.get("source") for match in evidence.get("matches", []) if isinstance(match, dict)}

        self.assertIn("FOCUS_PREREPAIR_TARGET_PROPERTY_QID", sources)

    def test_retained_old_value_only_in_prerepair_property_is_not_type_b(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["Q2", "Q3"], "new_value": ["Q3"]},
            violation_context={"report_violation_type_normalized": "Unique value"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertNotEqual(classification["class"], "TypeB")
        self.assertEqual(classification["subtype"], "UNKNOWN_SELECTION_AMBIGUOUS")

    def test_independent_focus_label_can_produce_local_text_confirmed(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "ABC", "new_value": "Correct Name"},
            violation_context={"report_violation_type_normalized": "Item"},
        )
        world_state = _world_state(
            L1_ego_node={"qid": "Q1", "label": "Correct Name", "description": "", "properties": {}}
        )

        classification, _, _ = classify_one(repair, world_state)

        self.assertEqual(classification["class"], "TypeB")
        self.assertEqual(classification["subtype"], "LOCAL_TEXT_CONFIRMED")

    def test_added_focus_qid_can_produce_local_focus_qid(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "MISSING", "new_value": "Q1"},
            violation_context={"report_violation_type_normalized": "Item"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeB")
        self.assertEqual(classification["subtype"], "LOCAL_FOCUS_QID")

    def test_multiplicity_increase_same_unique_is_unknown_artifact(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["x"], "new_value": ["x", "x"]}
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["subtype"], "UNKNOWN_MULTIPLICITY_ARTIFACT")
        self.assertNotEqual(classification["class"], "TypeB")

    def test_multiplicity_decrease_same_unique_is_type_a(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["x", "x"], "new_value": ["x"]}
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "MULTIPLICITY_NORMALIZATION")

    def test_add_superset_uses_added_value_not_retained_value(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["Q2"], "new_value": ["Q2", "Q3"]},
        )
        world_state = _world_state(
            L1_ego_node={"qid": "Q1", "label": "Focus", "description": "", "properties": {"P2": ["Q2"]}},
        )

        classification, _, _ = classify_one(repair, world_state)

        self.assertNotEqual(classification["class"], "TypeB")
        target_tokens = classification["diagnostics"]["classification_target_tokens"]
        self.assertEqual(target_tokens["tokens"], ["Q3"])

    def test_trailing_slash_format_normalization(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "PHABRA/", "new_value": "PHABRA"},
            violation_context={"report_violation_type_normalized": "Format"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "FORMAT_NORMALIZATION")

    def test_schembl_format_normalization(self):
        repair = _base_repair(
            repair_target={
                "kind": "A_BOX",
                "action": "UPDATE",
                "old_value": "SCHEMBL119427",
                "new_value": "119427",
            },
            violation_context={"report_violation_type_normalized": "Format"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "FORMAT_NORMALIZATION")

    def test_format_value_pruning_uses_removed_value(self):
        repair = _base_repair(
            repair_target={
                "kind": "A_BOX",
                "action": "UPDATE",
                "old_value": ["ps00551", "120617218"],
                "new_value": ["120617218"],
            },
            violation_context={"report_violation_type_normalized": "Format", "value": ["ps00551"]},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "FORMAT_VALUE_PRUNING")

    def test_delete_format_invalid_requires_format_report(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "DELETE", "old_value": "bad"},
            violation_context={"report_violation_type_normalized": "Format"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["subtype"], "REJECTION_FORMAT_INVALID")

    def test_unique_value_delete_not_format_just_because_format_constraint_exists(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "DELETE", "old_value": "bad"},
            violation_context={"report_violation_type_normalized": "Unique value"},
        )
        world_state = _world_state(
            L4_constraints={"constraints": [{"constraint_type": {"qid": "Q21502404", "label": "format constraint"}}]}
        )

        classification, _, _ = classify_one(repair, world_state)

        self.assertNotEqual(classification["subtype"], "REJECTION_FORMAT_INVALID")

    def test_self_link_subset_delete_is_type_a(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["Q1", "Q2"], "new_value": ["Q2"]},
            violation_context={"report_violation_type_normalized": "Self link"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "SELF_LINK_REJECTION")

    def test_self_link_delete_to_missing_is_type_a(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "DELETE", "old_value": "Q1"},
            violation_context={"report_violation_type_normalized": "Self link"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "SELF_LINK_REJECTION")

    def test_no_unqualified_typec_external(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "ABC", "new_value": "XYZ"},
            violation_context={"report_violation_type_normalized": "Item"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeC")
        self.assertNotEqual(classification["subtype"], "EXTERNAL")

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
