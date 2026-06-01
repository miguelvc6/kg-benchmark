import unittest

from classifier import classify_one, local_context_buckets, match_truth_locally
from lib.benchmark_selection import derive_case_metadata
from lib.manual_audit import audit_truth_token_kind
from lib.repair_state import comparable_atom, derive_value_change_summary, normalize_value_list


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


def _format_world_state(regex: str):
    return _world_state(
        L4_constraints={
            "constraints": [
                {
                    "constraint_type": {"qid": "Q21502404", "label": "format constraint"},
                    "qualifiers": [{"property_id": "P1793", "values": [{"raw": regex}]}],
                }
            ]
        }
    )


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

    def test_single_value_report_create_multiple_values_is_bad_target_diagnostic(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "MISSING", "new_value": ["Q2", "Q3"]},
            violation_context={"report_violation_type_normalized": "Single value"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        metadata = derive_case_metadata({**repair, "classification": classification}, tier="core")

        self.assertEqual(classification["class"], "TypeC")
        self.assertEqual(classification["subtype"], "UNKNOWN_BAD_TARGET_OR_CONTEXT")
        self.assertEqual(classification["confidence"], "low")
        self.assertFalse(metadata["main_score"])
        branch = next(step for step in classification["decision_trace"] if step.get("step") == "branch")
        self.assertEqual(branch["result"], "single_value_report_multiple_new_values")

    def test_single_value_subset_reduction_does_not_trigger_multiple_new_guard(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["Q2", "Q3"], "new_value": ["Q2"]},
            violation_context={"report_violation_type_normalized": "Single value"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertNotEqual(classification["subtype"], "UNKNOWN_BAD_TARGET_OR_CONTEXT")

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

    def test_added_focus_qid_without_rule_is_domain_reasoning_diagnostic(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "MISSING", "new_value": "Q1"},
            violation_context={"report_violation_type_normalized": "Item"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeC")
        self.assertEqual(classification["subtype"], "UNKNOWN_FOCUS_QID_DOMAIN_REASONING")
        self.assertEqual(classification["confidence"], "low")

    def test_target_required_focus_qid_still_type_a(self):
        repair = _base_repair(
            property="P5236",
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "MISSING", "new_value": "Q1"},
            violation_context={"report_violation_type_normalized": "Target required claim P|5236"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "TARGET_REQUIRED_CLAIM")

    def test_p8726_local_text_derived_from_statutory_instrument_label(self):
        repair = _base_repair(
            property="P8726",
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "MISSING", "new_value": "2007/si/483/made"},
            violation_context={"report_violation_type_normalized": "Item"},
        )
        world_state = _world_state(
            L1_ego_node={
                "qid": "Q1",
                "label": "Irish Statutory Instrument S.I. No. 483/2007",
                "description": "",
                "properties": {},
            }
        )

        classification, _, _ = classify_one(repair, world_state)

        self.assertEqual(classification["class"], "TypeB")
        self.assertEqual(classification["subtype"], "LOCAL_TEXT_DERIVED")
        detail = next(step for step in classification["decision_trace"] if step.get("step") == "local_text_derived")["detail"]
        self.assertEqual(detail["derivation_rule"], "p8726_statutory_instrument_id")
        self.assertTrue(detail["independent_of_target_property"])

    def test_p8726_without_matching_local_text_is_not_local_text_derived(self):
        repair = _base_repair(
            property="P8726",
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "MISSING", "new_value": "2007/si/483/made"},
            violation_context={"report_violation_type_normalized": "Item"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertNotEqual(classification["subtype"], "LOCAL_TEXT_DERIVED")

    def test_p8726_prerepair_target_literal_only_is_not_local_text_derived(self):
        repair = _base_repair(
            property="P8726",
            repair_target={
                "kind": "A_BOX",
                "action": "UPDATE",
                "old_value": "S.I. No. 483/2007",
                "new_value": "2007/si/483/made",
            },
            violation_context={"report_violation_type_normalized": "Item"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertNotEqual(classification["subtype"], "LOCAL_TEXT_DERIVED")

    def test_classification_rule_metadata_fields_exist(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "PHABRA/", "new_value": "PHABRA"},
            violation_context={"report_violation_type_normalized": "Format"},
        )

        classification, _, _ = classify_one(repair, _format_world_state(r"^[A-Z]+$"))

        self.assertEqual(classification["classification_rule_family"], "format")
        self.assertIn("classification_rule_subfamily", classification)
        self.assertEqual(classification["decision_constraint_type_qid"], "Q21502404")
        self.assertIn("decision_constraint_source", classification)

    def test_self_link_rule_metadata_does_not_claim_symmetric_constraint(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["Q1", "Q2"], "new_value": ["Q2"]},
            violation_context={"report_violation_type_normalized": "Self link"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["subtype"], "SELF_LINK_REJECTION")
        self.assertEqual(classification["classification_rule_family"], "self_link_report")
        self.assertIsNone(classification["decision_constraint_type_qid"])

    def test_multiplicity_increase_same_unique_is_unknown_artifact(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["x"], "new_value": ["x", "x"]}
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["subtype"], "UNKNOWN_MULTIPLICITY_ARTIFACT")
        self.assertNotEqual(classification["class"], "TypeB")

    def test_multiplicity_decrease_same_unique_is_type_a(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["x", "x"], "new_value": ["x"]},
            violation_context={"report_violation_type_normalized": "Single value"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "MULTIPLICITY_NORMALIZATION")

    def test_multiplicity_decrease_unique_value_is_type_a(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["x", "x"], "new_value": ["x"]},
            violation_context={"report_violation_type_normalized": "Unique value"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        metadata = derive_case_metadata({**repair, "classification": classification}, tier="core")

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "MULTIPLICITY_NORMALIZATION")
        self.assertTrue(metadata["main_score"])

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

    def test_format_value_pruning_requires_retained_value_to_pass_regex(self):
        repair = _base_repair(
            repair_target={
                "kind": "A_BOX",
                "action": "UPDATE",
                "old_value": ["bad", "also-bad"],
                "new_value": ["also-bad"],
            },
            violation_context={"report_violation_type_normalized": "Format"},
        )

        classification, _, _ = classify_one(repair, _format_world_state(r"\d+"))

        self.assertEqual(classification["class"], "TypeC")
        self.assertEqual(classification["subtype"], "UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED")

    def test_p8748_style_retained_numeric_id_passes_format_regex(self):
        repair = _base_repair(
            property="P8748",
            repair_target={
                "kind": "A_BOX",
                "action": "UPDATE",
                "old_value": ["ps00551", "120617218"],
                "new_value": ["120617218"],
            },
            violation_context={"report_violation_type_normalized": "Format"},
        )

        classification, _, _ = classify_one(repair, _format_world_state(r"\d+"))

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "FORMAT_VALUE_PRUNING")
        detail = next(step for step in classification["decision_trace"] if step.get("step") == "rule_deterministic")[
            "detail"
        ]
        self.assertIs(detail["retained_pass_regex"], True)

    def test_mixed_schembl_format_normalization_is_type_a(self):
        repair = _base_repair(
            repair_target={
                "kind": "A_BOX",
                "action": "UPDATE",
                "old_value": ["SCHEMBL117208", "7582475"],
                "new_value": ["117208", "7582475"],
            },
            violation_context={"report_violation_type_normalized": "Format"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "FORMAT_NORMALIZATION")
        self.assertNotEqual(classification["subtype"], "EXTERNAL_BY_ELIMINATION")
        target = classification["diagnostics"]["classification_target_tokens"]
        self.assertEqual(target["old_changed_value"], "SCHEMBL117208")
        self.assertEqual(target["new_changed_value"], "117208")

    def test_mixed_schembl_normalization_under_unique_value_is_type_a(self):
        repair = _base_repair(
            property="P2877",
            repair_target={
                "kind": "A_BOX",
                "action": "UPDATE",
                "old_value": ["40723", "SCHEMBL3175"],
                "new_value": ["40723", "3175"],
            },
            violation_context={"report_violation_type_normalized": "Unique value"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "FORMAT_NORMALIZATION")
        self.assertEqual(classification["confidence"], "medium")

    def test_trailing_slash_direction_uses_regex_when_required(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "OENDRU", "new_value": "OENDRU/"},
            violation_context={"report_violation_type_normalized": "Format"},
        )

        classification, _, _ = classify_one(repair, _format_world_state(r"[A-Z]+/"))

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "FORMAT_NORMALIZATION")

    def test_format_update_to_regex_invalid_target_is_diagnostic(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "OENDRU", "new_value": "OENDRU/"},
            violation_context={"report_violation_type_normalized": "Format"},
        )

        classification, _, _ = classify_one(repair, _format_world_state(r"[A-Z]+"))

        self.assertEqual(classification["class"], "TypeC")
        self.assertEqual(classification["subtype"], "UNKNOWN_BAD_TARGET_OR_CONTEXT")

    def test_target_required_claim_focus_qid_is_type_a_not_type_b(self):
        repair = _base_repair(
            qid="Q1187903",
            property="P5236",
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": "MISSING", "new_value": "Q1187903"},
            violation_context={"report_violation_type_normalized": "Target required claim P|5236"},
        )

        classification, _, _ = classify_one(repair, _world_state(L1_ego_node={"qid": "Q1187903", "label": "", "description": "", "properties": {}}))

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "TARGET_REQUIRED_CLAIM")

    def test_self_link_add_focus_qid_is_diagnostic_bad_target(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["Q2"], "new_value": ["Q2", "Q1"]},
            violation_context={"report_violation_type_normalized": "Self link"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeC")
        self.assertEqual(classification["subtype"], "UNKNOWN_BAD_TARGET_OR_CONTEXT")

    def test_category_prefix_normalization_precedes_local_text(self):
        repair = _base_repair(
            repair_target={
                "kind": "A_BOX",
                "action": "UPDATE",
                "old_value": "Category:Blueground",
                "new_value": "Blueground",
            },
            violation_context={"report_violation_type_normalized": "Format"},
        )
        world_state = _world_state(L1_ego_node={"qid": "Q1", "label": "Blueground", "description": "", "properties": {}})

        classification, _, _ = classify_one(repair, world_state)

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "FORMAT_NORMALIZATION")

    def test_url_slug_normalization_precedes_local_text(self):
        repair = _base_repair(
            repair_target={
                "kind": "A_BOX",
                "action": "UPDATE",
                "old_value": "https://www.linkedin.com/company/vacasa/",
                "new_value": "vacasa",
            },
            violation_context={"report_violation_type_normalized": "Format"},
        )
        world_state = _world_state(L1_ego_node={"qid": "Q1", "label": "vacasa", "description": "", "properties": {}})

        classification, _, _ = classify_one(repair, world_state)

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "FORMAT_NORMALIZATION")

    def test_multiplicity_decrease_none_of_report_is_diagnostic(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["x", "x"], "new_value": ["x"]},
            violation_context={"report_violation_type_normalized": "None of"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeC")
        self.assertEqual(classification["subtype"], "UNKNOWN_MULTIPLICITY_ARTIFACT")

    def test_multiplicity_decrease_item_report_is_diagnostic(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["x", "x"], "new_value": ["x"]},
            violation_context={"report_violation_type_normalized": "Item P|170"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeC")
        self.assertEqual(classification["subtype"], "UNKNOWN_MULTIPLICITY_ARTIFACT")

    def test_multiplicity_decrease_label_report_is_diagnostic(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["x", "x"], "new_value": ["x"]},
            violation_context={"report_violation_type_normalized": "Label in mul language"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "TypeC")
        self.assertEqual(classification["subtype"], "UNKNOWN_MULTIPLICITY_ARTIFACT")

    def test_set_membership_rejection_when_removed_value_ruled_out(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["Q2", "Q3"], "new_value": ["Q3"]},
            violation_context={"report_violation_type_normalized": "None of"},
        )
        world_state = _world_state(
            L4_constraints={
                "constraints": [
                    {
                        "constraint_type": {"qid": "Q52558054", "label": "none of constraint"},
                        "qualifiers": [{"property_id": "P2305", "values": [{"qid": "Q2"}]}],
                    }
                ]
            }
        )

        classification, _, _ = classify_one(repair, world_state)
        metadata = derive_case_metadata({**repair, "classification": classification}, tier="core")

        self.assertEqual(classification["class"], "TypeA")
        self.assertEqual(classification["subtype"], "SET_MEMBERSHIP_REJECTION")
        self.assertTrue(metadata["main_score"])

    def test_ambiguous_set_membership_subset_is_not_type_a(self):
        repair = _base_repair(
            repair_target={"kind": "A_BOX", "action": "UPDATE", "old_value": ["Q2", "Q3"], "new_value": ["Q3"]},
            violation_context={"report_violation_type_normalized": "One of"},
        )
        world_state = _world_state(
            L4_constraints={
                "constraints": [
                    {
                        "constraint_type": {"qid": "Q21510859", "label": "one of constraint"},
                        "qualifiers": [{"property_id": "P2305", "values": [{"qid": "Q2"}, {"qid": "Q3"}]}],
                    }
                ]
            }
        )

        classification, _, _ = classify_one(repair, world_state)

        self.assertNotEqual(classification["subtype"], "SET_MEMBERSHIP_REJECTION")

    def test_phone_number_literal_is_not_date_token_kind(self):
        record = {
            "classification": {
                "class": "TypeC",
                "subtype": "EXTERNAL_BY_ELIMINATION",
                "confidence": "medium",
                "diagnostics": {"truth_tokens": ["+886-5-2284567"]},
            }
        }

        self.assertEqual(audit_truth_token_kind(record), "literal")
        self.assertEqual(derive_case_metadata({**_base_repair(id="phone"), **record}, tier="core")["truth_token_kind"], "literal")

    def test_wikidata_date_literal_is_date_token_kind(self):
        record = {
            "classification": {
                "class": "TypeC",
                "subtype": "EXTERNAL_BY_ELIMINATION",
                "confidence": "medium",
                "diagnostics": {"truth_tokens": ["+2020-01-01T00:00:00Z"]},
            }
        }

        self.assertEqual(audit_truth_token_kind(record), "date")
        self.assertEqual(derive_case_metadata({**_base_repair(id="date"), **record}, tier="core")["truth_token_kind"], "date")

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

    def test_tbox_qualifier_only_change_counts_changed_constraint_family(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": [],
                    "signature_before": [
                        {"constraint_qid": "Q19474404", "qualifiers": [{"property_id": "P4155", "values": ["Q1"]}]}
                    ],
                    "signature_after": [
                        {"constraint_qid": "Q19474404", "qualifiers": [{"property_id": "P4155", "values": ["Q1", "Q2"]}]}
                    ],
                },
            },
            violation_context={"report_violation_type_normalized": "Single value"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertIn("Q19474404", causality["changed_constraint_qids_all"])
        self.assertIn("Q19474404", causality["changed_constraint_qids_from_qualifier_changes"])

    def test_tbox_unmapped_violation_is_unknown_causality(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q21502404"],
                    "signature_before": [{"constraint_qid": "Q21502404", "qualifiers": []}],
                    "signature_after": [{"constraint_qid": "Q21502404", "qualifiers": [{"property_id": "P1793", "values": ["[0-9]+"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "Unmapped custom report"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertEqual(classification["class"], "T_BOX")
        self.assertEqual(classification["subtype"], "UNKNOWN_TBOX_CAUSALITY")
        self.assertEqual(classification["confidence"], "low")

    def test_tbox_family_mismatch_does_not_become_directional(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q19474404"],
                    "signature_before": [{"constraint_qid": "Q19474404", "qualifiers": []}],
                    "signature_after": [{"constraint_qid": "Q19474404", "qualifiers": [{"property_id": "P4155", "values": ["P1"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "Inverse"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertIn(classification["subtype"], {"UNKNOWN_TBOX_CAUSALITY", "COINCIDENTAL_SCHEMA_CHANGE"})

    def test_tbox_value_type_report_selects_value_type_constraint(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q19474404", "Q21510865"],
                    "signature_before": [
                        {"constraint_qid": "Q19474404", "qualifiers": []},
                        {"constraint_qid": "Q21510865", "qualifiers": [{"property_id": "P2308", "values": ["Q5"]}]},
                    ],
                    "signature_after": [
                        {"constraint_qid": "Q19474404", "qualifiers": [{"property_id": "P4155", "values": ["P1"]}]},
                        {"constraint_qid": "Q21510865", "qualifiers": [{"property_id": "P2308", "values": ["Q5", "Q42"]}]},
                    ],
                },
            },
            violation_context={"report_violation_type_normalized": "Value type Q|42"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertEqual(causality["target_constraint_qid"], "Q21510865")
        self.assertEqual(classification["subtype"], "RELAXATION_SET_EXPANSION")

    def test_tbox_forbidden_set_polarity(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q52558054"],
                    "signature_before": [{"constraint_qid": "Q52558054", "qualifiers": [{"property_id": "P2305", "values": ["Q1"]}]}],
                    "signature_after": [{"constraint_qid": "Q52558054", "qualifiers": [{"property_id": "P2305", "values": ["Q1", "Q2"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "None of"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertEqual(classification["subtype"], "RESTRICTION_SET_CONTRACTION")
        self.assertEqual(causality["set_semantics"], "forbidden")
        self.assertEqual(causality["polarity"], "restriction")

    def test_tbox_required_set_polarity(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q21510856"],
                    "signature_before": [{"constraint_qid": "Q21510856", "qualifiers": []}],
                    "signature_after": [{"constraint_qid": "Q21510856", "qualifiers": [{"property_id": "P2306", "values": ["P580"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "Mandatory Qualifiers"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertEqual(classification["subtype"], "RESTRICTION_SET_CONTRACTION")
        self.assertEqual(causality["set_semantics"], "required")

    def test_tbox_label_language_status_only_change_is_ignored_for_direction(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q108139345"],
                    "signature_before": [{"constraint_qid": "Q108139345", "qualifiers": [{"property_id": "P2316", "values": ["Q1"]}]}],
                    "signature_after": [{"constraint_qid": "Q108139345", "qualifiers": [{"property_id": "P2316", "values": ["Q2"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "Label in en-gb language"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertIn("P2316", causality["ignored_changed_qualifier_properties"])
        self.assertEqual(causality["semantic_added_values"], [])
        self.assertNotIn(classification["subtype"], {"RELAXATION_SET_EXPANSION", "RESTRICTION_SET_CONTRACTION"})

    def test_tbox_required_statement_filters_status_from_semantic_values(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q21503247"],
                    "signature_before": [{"constraint_qid": "Q21503247", "qualifiers": []}],
                    "signature_after": [
                        {
                            "constraint_qid": "Q21503247",
                            "qualifiers": [
                                {"property_id": "P2306", "values": ["P170"]},
                                {"property_id": "P2316", "values": ["Q21502408"]},
                            ],
                        }
                    ],
                },
            },
            violation_context={"report_violation_type_normalized": "Item P|170"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertIn("P2306", causality["semantic_changed_qualifier_properties"])
        self.assertIn("P2316", causality["ignored_changed_qualifier_properties"])
        self.assertEqual(causality["semantic_added_values"], ["P170"])
        self.assertIn("Q21502408", causality["ignored_added_values"])
        self.assertEqual(causality["compatible_property_overlap_with_report_pids"], ["P170"])

    def test_tbox_format_filters_regex_as_semantic_and_status_as_ignored(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q21502404"],
                    "signature_before": [{"constraint_qid": "Q21502404", "qualifiers": [{"property_id": "P1793", "values": ["[A-Z]+"]}]}],
                    "signature_after": [
                        {
                            "constraint_qid": "Q21502404",
                            "qualifiers": [
                                {"property_id": "P1793", "values": ["[0-9]+"]},
                                {"property_id": "P2316", "values": ["Q21502408"]},
                            ],
                        }
                    ],
                },
            },
            violation_context={"report_violation_type_normalized": "Format"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertEqual(classification["subtype"], "SCHEMA_UPDATE")
        self.assertIn("P1793", causality["semantic_changed_qualifier_properties"])
        self.assertIn("P2316", causality["ignored_changed_qualifier_properties"])
        self.assertIn("[0-9]+", causality["semantic_added_values"])

    def test_tbox_value_type_filters_class_relation_semantic_values(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q21510865"],
                    "signature_before": [{"constraint_qid": "Q21510865", "qualifiers": []}],
                    "signature_after": [{"constraint_qid": "Q21510865", "qualifiers": [{"property_id": "P2308", "values": ["Q5", "Q42"]}, {"property_id": "P2316", "values": ["Q1"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "Value type Q|42"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertIn("P2308", causality["semantic_changed_qualifier_properties"])
        self.assertIn("P2316", causality["ignored_changed_qualifier_properties"])
        self.assertEqual(causality["compatible_value_overlap_with_report_qids"], ["Q42"])

    def test_tbox_concrete_value_type_without_overlap_is_not_directional(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q21510865"],
                    "signature_before": [{"constraint_qid": "Q21510865", "qualifiers": []}],
                    "signature_after": [{"constraint_qid": "Q21510865", "qualifiers": [{"property_id": "P2308", "values": ["Q6"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "Value type Q|42"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertEqual(classification["subtype"], "SCHEMA_UPDATE")
        self.assertEqual(classification["confidence"], "medium")
        self.assertTrue(causality["value_specific_without_overlap"])
        self.assertEqual(causality["causality_match_level"], "exact_constraint_family_only_no_compatible_overlap")
        self.assertIsNone(causality["directional_subtype_precise"])
        self.assertEqual(classification["analysis_slice_precise"], "main_tbox_schema_update")
        self.assertEqual(causality["analysis_slice_precise"], "main_tbox_schema_update")
        self.assertEqual(causality["potential_directional_subtype_precise"], "RELAXATION_ALLOWED_SET_EXPANSION")

    def test_tbox_related_allowed_entity_target_selected_for_subject_type_report(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q52004125"],
                    "signature_before": [{"constraint_qid": "Q52004125", "qualifiers": [{"property_id": "P2308", "values": ["Q5"]}]}],
                    "signature_after": [{"constraint_qid": "Q52004125", "qualifiers": [{"property_id": "P2308", "values": ["Q5", "Q42"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "Type Q|42"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertEqual(causality["mapped_report_constraint_qid"], "Q21503250")
        self.assertEqual(causality["target_constraint_qid"], "Q52004125")
        self.assertEqual(causality["target_constraint_selection_reason"], "changed_related_constraint_has_report_overlap")
        self.assertTrue(causality["target_constraint_is_related_family"])

    def test_tbox_related_allowed_entity_without_overlap_has_no_target(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q52004125"],
                    "signature_before": [{"constraint_qid": "Q52004125", "qualifiers": [{"property_id": "P2308", "values": ["Q5"]}]}],
                    "signature_after": [{"constraint_qid": "Q52004125", "qualifiers": [{"property_id": "P2308", "values": ["Q6"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "Type Q|42"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertIn(classification["subtype"], {"UNKNOWN_TBOX_CAUSALITY", "COINCIDENTAL_SCHEMA_CHANGE"})

    def test_tbox_mapped_item_required_changed_label_language_has_no_target(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q108139345"],
                    "signature_before": [{"constraint_qid": "Q108139345", "qualifiers": []}],
                    "signature_after": [{"constraint_qid": "Q108139345", "qualifiers": [{"property_id": "P424", "values": ["en"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "Item P|170"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertIn(classification["subtype"], {"UNKNOWN_TBOX_CAUSALITY", "COINCIDENTAL_SCHEMA_CHANGE"})

    def test_tbox_analysis_slice_precise_exists_for_directional_cases(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q21510859"],
                    "signature_before": [{"constraint_qid": "Q21510859", "qualifiers": [{"property_id": "P2305", "values": ["Q1"]}]}],
                    "signature_after": [{"constraint_qid": "Q21510859", "qualifiers": [{"property_id": "P2305", "values": ["Q1", "Q2"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "One of"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertEqual(causality["directional_subtype_precise"], "RELAXATION_ALLOWED_SET_EXPANSION")
        self.assertEqual(classification["analysis_slice_precise"], "main_tbox_relaxation_allowed_set_expansion")
        self.assertEqual(causality["analysis_slice_precise"], "main_tbox_relaxation_allowed_set_expansion")

    def test_tbox_symmetric_mapping_uses_symmetric_constraint_not_self_link(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q21510862"],
                    "signature_before": [{"constraint_qid": "Q21510862", "qualifiers": []}],
                    "signature_after": [{"constraint_qid": "Q21510862", "qualifiers": [{"property_id": "P2306", "values": ["P26"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "Symmetric"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertEqual(causality["mapped_violation_constraint_qid"], "Q21510862")
        self.assertEqual(causality["mapped_violation_constraint_label"], "symmetric constraint")
        self.assertEqual(causality["mapped_violation_family"], "symmetric")

    def test_tbox_label_language_mapping_uses_qid_and_language_overlap(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q108139345"],
                    "signature_before": [{"constraint_qid": "Q108139345", "qualifiers": []}],
                    "signature_after": [{"constraint_qid": "Q108139345", "qualifiers": [{"property_id": "P424", "values": ["cy"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "Label in cy language"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertEqual(causality["mapped_violation_constraint_qid"], "Q108139345")
        self.assertEqual(causality["mapped_violation_family"], "label_in_language")
        self.assertEqual(causality["language_overlap_with_report_langs"], ["cy"])

    def test_tbox_incompatible_language_overlap_does_not_support_format_causality(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q108139345"],
                    "signature_before": [{"constraint_qid": "Q108139345", "qualifiers": []}],
                    "signature_after": [{"constraint_qid": "Q108139345", "qualifiers": [{"property_id": "P424", "values": ["cy"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "Format", "message": "format violation in cy"},
        )

        classification, _, _ = classify_one(repair, _world_state())

        self.assertIn(classification["subtype"], {"UNKNOWN_TBOX_CAUSALITY", "COINCIDENTAL_SCHEMA_CHANGE"})

    def test_tbox_candidate_selection_uses_secondary_exact_family_candidate(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q21510865"],
                    "signature_before": [{"constraint_qid": "Q21510865", "qualifiers": [{"property_id": "P2308", "values": ["Q5"]}]}],
                    "signature_after": [{"constraint_qid": "Q21510865", "qualifiers": [{"property_id": "P2308", "values": ["Q5", "Q42"]}]}],
                },
            },
            violation_context={
                "report_violation_type_normalized": "Unmapped custom report",
                "report_violation_type": "Value type Q|42",
            },
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertEqual(causality["selected_violation_name"], "Value type Q|42")
        self.assertEqual(causality["target_constraint_qid"], "Q21510865")
        self.assertEqual(classification["subtype"], "RELAXATION_SET_EXPANSION")

    def test_tbox_directional_metadata_includes_precise_polarity(self):
        repair = _base_repair(
            track="T_BOX",
            repair_target={
                "kind": "T_BOX",
                "constraint_delta": {
                    "changed_constraint_types": ["Q52558054"],
                    "signature_before": [{"constraint_qid": "Q52558054", "qualifiers": [{"property_id": "P2305", "values": ["Q1"]}]}],
                    "signature_after": [{"constraint_qid": "Q52558054", "qualifiers": [{"property_id": "P2305", "values": ["Q1", "Q2"]}]}],
                },
            },
            violation_context={"report_violation_type_normalized": "None of"},
        )

        classification, _, _ = classify_one(repair, _world_state())
        causality = next(step for step in classification["decision_trace"] if step.get("step") == "tbox_causality")

        self.assertEqual(causality["set_operation"], "expansion")
        self.assertEqual(causality["directional_subtype_precise"], "RESTRICTION_FORBIDDEN_SET_EXPANSION")

    def test_repair_state_normalizes_structured_time_amount_and_text_values(self):
        self.assertEqual(comparable_atom({"time": "+1983-00-00T00:00:00Z"}), "+1983-00-00T00:00:00Z")
        self.assertEqual(comparable_atom({"amount": "+42"}), "+42")
        self.assertEqual(comparable_atom({"text": "literal"}), "literal")
        self.assertEqual(normalize_value_list([{"time": "+1983-00-00T00:00:00Z"}, "MISSING"]), ["+1983-00-00T00:00:00Z"])

    def test_repair_state_value_change_summary_actions(self):
        summary = derive_value_change_summary(
            {"repair_target": {"kind": "A_BOX", "action": "UPDATE", "old_value": ["x", "x"], "new_value": ["x"]}}
        )
        self.assertEqual(summary.semantic_action, "MULTIPLICITY_DECREASE_SAME_UNIQUE")
        summary = derive_value_change_summary(
            {"repair_target": {"kind": "A_BOX", "action": "UPDATE", "old_value": ["a", "b"], "new_value": ["b"]}}
        )
        self.assertEqual(summary.semantic_action, "DELETE_SUBSET")

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
