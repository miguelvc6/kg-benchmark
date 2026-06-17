import unittest

from guardian.tbox_taxonomy_patch_evaluator import NEW_TBOX_PATCH_METRICS, evaluate_tbox_taxonomy_patch_predictions


OPERATION_CODE_PAIRS = [
    ("CONSTRAINT_REMOVE", "C_MINUS"),
    ("CONSTRAINT_DEPRECATE", "C_D"),
    ("CONSTRAINT_ADD", "C_PLUS"),
    ("CONSTRAINT_TYPE_REPLACE", "C_REPLACE"),
    ("CONSTRAINT_QUALIFIER_ADD", "CQ_PLUS"),
    ("CONSTRAINT_QUALIFIER_REMOVE", "CQ_MINUS"),
    ("CONSTRAINT_QUALIFIER_REPLACE", "CQ_REPLACE"),
    ("CLASS_HIERARCHY_ADD", "SUBCLASS_PLUS"),
    ("EXCEPTION_ADD", "E_PLUS"),
    ("OTHER_TBOX_UPDATE", "OTHER"),
]


def repair(repair_op: str = "CONSTRAINT_QUALIFIER_ADD", taxonomy_code: str = "CQ_PLUS", **overrides):
    payload = {
        "repair_op": repair_op,
        "taxonomy_code": taxonomy_code,
        "constraint_type_qid": "Q21510859",
        "qualifier_property_id": "P2305",
        "added_values": ["Q5"],
        "removed_values": [],
        "old_value": None,
        "new_value": "Q5",
        "rank_after": "normal",
        "snaktype_after": "VALUE",
        "evidence_level": "VALUE_DELTA_VISIBLE",
    }
    payload.update(overrides)
    return payload


def patch(case_id: str, *, decision: str = "CAUSAL_SCHEMA_REPAIR", repairs=None, qid: str = "Q21510859"):
    return {
        "case_id": case_id,
        "schema_decision": decision,
        "target": {"pid": "P31", "constraint_type_qid": qid},
        "repairs": [repair(constraint_type_qid=qid)] if repairs is None else repairs,
        "rationale": "Visible historical delta.",
        "provenance": [{"kind": "KG", "node_id": "P31", "snippet": "constraint delta"}],
        "uncertainty": {"confidence": 0.75, "notes": "direct evidence"},
    }


class TBoxTaxonomyPatchEvaluatorTests(unittest.TestCase):
    def metric(self, summary: dict, name: str, subset: str = "all_core") -> dict:
        return summary["subsets"][subset]["metrics"][name]

    def test_scores_synthetic_example_for_every_operation(self) -> None:
        gold = []
        predictions = []
        annotations = {}
        for idx, (repair_op, taxonomy_code) in enumerate(OPERATION_CODE_PAIRS):
            case_id = f"case-{idx}"
            entry = repair(repair_op, taxonomy_code)
            if repair_op in {"CLASS_HIERARCHY_ADD", "OTHER_TBOX_UPDATE"}:
                entry.update(
                    {
                        "qualifier_property_id": None,
                        "added_values": [],
                        "removed_values": [],
                        "old_value": None,
                        "new_value": None,
                        "evidence_level": "OPERATION_VISIBLE",
                    }
                )
            if repair_op == "CONSTRAINT_DEPRECATE":
                entry.update(
                    {
                        "added_values": [],
                        "new_value": None,
                        "rank_after": "deprecated",
                        "evidence_level": "OPERATION_VISIBLE",
                    }
                )
            gold.append(patch(case_id, repairs=[entry]))
            predictions.append(patch(case_id, repairs=[dict(entry)]))
            annotations[case_id] = {"main_score": idx % 2 == 0, "diagnostic_only": idx % 2 == 1}

        summary = evaluate_tbox_taxonomy_patch_predictions(
            gold_rows=gold,
            prediction_rows=predictions,
            case_annotations=annotations,
        )
        self.assertEqual(summary["metric_family"], "tbox_taxonomy_patch_v1")
        self.assertEqual(summary["strict_signature_metrics_role"], "diagnostic_only")
        self.assertEqual(summary["subsets"]["all_core"]["count"], len(OPERATION_CODE_PAIRS))
        self.assertEqual(summary["subsets"]["main_score"]["count"], 5)
        self.assertEqual(summary["subsets"]["diagnostic"]["count"], 5)
        self.assertEqual(self.metric(summary, "tbox_patch_parse_rate")["rate"], 1.0)
        self.assertEqual(self.metric(summary, "tbox_patch_taxonomy_level_success")["rate"], 1.0)

    def test_every_new_metric_reports_denominator_metadata(self) -> None:
        summary = evaluate_tbox_taxonomy_patch_predictions(
            gold_rows=[patch("case-1")],
            prediction_rows=[patch("case-1")],
        )
        metrics = summary["subsets"]["all_core"]["metrics"]
        self.assertEqual(set(NEW_TBOX_PATCH_METRICS), set(metrics))
        for name in NEW_TBOX_PATCH_METRICS:
            with self.subTest(metric=name):
                self.assertIn("numerator", metrics[name])
                self.assertIn("applicable_denominator", metrics[name])
                self.assertIn("total_tbox_rows", metrics[name])
                self.assertIn("applicability_coverage", metrics[name])
                self.assertIn("rate", metrics[name])

    def test_multi_edit_gold_and_prediction_matching(self) -> None:
        gold_repairs = [
            repair("CONSTRAINT_QUALIFIER_ADD", "CQ_PLUS", added_values=["Q5"], new_value="Q5"),
            repair(
                "CONSTRAINT_QUALIFIER_REMOVE",
                "CQ_MINUS",
                added_values=[],
                removed_values=["Q43229"],
                old_value="Q43229",
                new_value=None,
            ),
        ]
        prediction_repairs = [dict(gold_repairs[1]), dict(gold_repairs[0])]
        summary = evaluate_tbox_taxonomy_patch_predictions(
            gold_rows=[patch("case-1", repairs=gold_repairs)],
            prediction_rows=[patch("case-1", repairs=prediction_repairs)],
        )
        self.assertEqual(self.metric(summary, "tbox_patch_repair_op_exact_match_rate")["rate"], 1.0)
        self.assertEqual(self.metric(summary, "tbox_patch_repair_op_f1")["rate"], 1.0)
        self.assertEqual(self.metric(summary, "tbox_patch_value_delta_success")["rate"], 1.0)

    def test_empty_repair_no_causal_schema_repair(self) -> None:
        gold = patch("case-1", decision="NO_CAUSAL_SCHEMA_REPAIR", repairs=[])
        prediction = patch("case-1", decision="NO_CAUSAL_SCHEMA_REPAIR", repairs=[])
        summary = evaluate_tbox_taxonomy_patch_predictions(gold_rows=[gold], prediction_rows=[prediction])
        self.assertEqual(self.metric(summary, "tbox_patch_no_causal_schema_repair_match_rate")["rate"], 1.0)
        self.assertEqual(self.metric(summary, "tbox_patch_taxonomy_level_success")["rate"], 1.0)

    def test_empty_repair_unclear_schema_evidence(self) -> None:
        gold = patch("case-1", decision="UNCLEAR_SCHEMA_EVIDENCE", repairs=[])
        prediction = patch("case-1", decision="UNCLEAR_SCHEMA_EVIDENCE", repairs=[])
        summary = evaluate_tbox_taxonomy_patch_predictions(gold_rows=[gold], prediction_rows=[prediction])
        self.assertEqual(self.metric(summary, "tbox_patch_unclear_schema_evidence_match_rate")["rate"], 1.0)
        self.assertEqual(self.metric(summary, "tbox_patch_taxonomy_level_success")["rate"], 1.0)

    def test_value_delta_claimed_when_gold_absent_is_reported(self) -> None:
        gold = patch(
            "case-1",
            repairs=[
                repair(
                    "OTHER_TBOX_UPDATE",
                    "OTHER",
                    qualifier_property_id=None,
                    added_values=[],
                    removed_values=[],
                    old_value=None,
                    new_value=None,
                    evidence_level="FAMILY_ONLY",
                )
            ],
        )
        prediction = patch("case-1")
        summary = evaluate_tbox_taxonomy_patch_predictions(gold_rows=[gold], prediction_rows=[prediction])
        self.assertEqual(self.metric(summary, "tbox_patch_value_delta_claimed_when_gold_absent_rate")["rate"], 1.0)

    def test_missing_prediction_counts_parse_error(self) -> None:
        summary = evaluate_tbox_taxonomy_patch_predictions(gold_rows=[patch("case-1")], prediction_rows=[])
        self.assertEqual(self.metric(summary, "tbox_patch_parse_rate")["rate"], 0.0)
        self.assertEqual(self.metric(summary, "tbox_patch_parse_error_rate")["rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
