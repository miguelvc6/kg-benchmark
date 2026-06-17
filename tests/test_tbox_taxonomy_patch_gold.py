import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from lib.tbox_taxonomy_patch_gold import (
    CoverageError,
    extract_selected_tbox_gold,
    gold_patch_for_record,
    normalize_signature_families,
    summarize_patches,
)


class TBoxTaxonomyPatchGoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with (repo_root / "schemas" / "tbox_taxonomy_patch_proposal.schema.json").open(encoding="utf-8") as handle:
            cls.validator = Draft202012Validator(json.load(handle))

    def assertSchemaValid(self, payload: dict) -> None:
        errors = sorted(self.validator.iter_errors(payload), key=lambda error: list(error.path))
        if errors:
            self.fail("; ".join(error.message for error in errors))

    def _record(self, *, subtype: str = "SCHEMA_UPDATE", summary: dict | None = None) -> dict:
        return {
            "id": "reform_Q1_P31_123",
            "property": "P31",
            "track": "T_BOX",
            "qid": "Q1",
            "repair_target": {
                "kind": "T_BOX",
                "property_revision_id": 123,
                "constraint_delta": {"changed_constraint_types": ["Q21510859"]},
            },
            "classification": {
                "class": "T_BOX",
                "subtype": subtype,
                "confidence": "medium",
                "decision_constraint_type_qid": "Q21510859",
                "diagnostics": {"tbox_diff_summary": summary or {}},
            },
            "violation_context": {"report_violation_type": "Allowed values"},
        }

    def test_normalize_signature_families(self) -> None:
        normalized = normalize_signature_families(
            [
                {
                    "constraint_qid": "q21510859",
                    "snaktype": "value",
                    "rank": "normal",
                    "qualifiers": [{"property_id": "p2305", "values": ["q5", "Q43229", "abc"]}],
                }
            ]
        )
        self.assertEqual(
            normalized,
            {
                "Q21510859": {
                    "snaktypes": ["VALUE"],
                    "ranks": ["normal"],
                    "qualifiers": {"P2305": ["Q43229", "Q5", "abc"]},
                }
            },
        )

    def test_diagnostics_no_causal_schema_repair_has_empty_repairs(self) -> None:
        patch = gold_patch_for_record(self._record(subtype="COINCIDENTAL_SCHEMA_CHANGE"))
        self.assertEqual(patch["schema_decision"], "NO_CAUSAL_SCHEMA_REPAIR")
        self.assertEqual(patch["repairs"], [])
        self.assertSchemaValid(patch)

    def test_diagnostics_single_qualifier_add_has_value_delta(self) -> None:
        patch = gold_patch_for_record(
            self._record(
                subtype="RELAXATION_SET_EXPANSION",
                summary={
                    "target_constraint_qid": "Q21510859",
                    "semantic_changed_qualifier_properties": ["P2305"],
                    "semantic_added_values": ["Q5"],
                    "semantic_removed_values": [],
                },
            )
        )
        self.assertEqual(patch["repairs"][0]["repair_op"], "CONSTRAINT_QUALIFIER_ADD")
        self.assertEqual(patch["repairs"][0]["taxonomy_code"], "CQ_PLUS")
        self.assertEqual(patch["repairs"][0]["added_values"], ["Q5"])
        self.assertEqual(patch["repairs"][0]["evidence_level"], "VALUE_DELTA_VISIBLE")
        self.assertSchemaValid(patch)

    def test_diagnostics_single_qualifier_remove_has_value_delta(self) -> None:
        patch = gold_patch_for_record(
            self._record(
                subtype="RESTRICTION_SET_CONTRACTION",
                summary={
                    "target_constraint_qid": "Q21510859",
                    "semantic_changed_qualifier_properties": ["P2305"],
                    "semantic_added_values": [],
                    "semantic_removed_values": ["Q5"],
                },
            )
        )
        self.assertEqual(patch["repairs"][0]["repair_op"], "CONSTRAINT_QUALIFIER_REMOVE")
        self.assertEqual(patch["repairs"][0]["removed_values"], ["Q5"])
        self.assertSchemaValid(patch)

    def test_diagnostics_single_qualifier_replace_has_value_delta(self) -> None:
        patch = gold_patch_for_record(
            self._record(
                summary={
                    "target_constraint_qid": "Q21510859",
                    "semantic_changed_qualifier_properties": ["P2305"],
                    "semantic_added_values": ["Q5"],
                    "semantic_removed_values": ["Q43229"],
                },
            )
        )
        self.assertEqual(patch["repairs"][0]["repair_op"], "CONSTRAINT_QUALIFIER_REPLACE")
        self.assertEqual(patch["repairs"][0]["added_values"], ["Q5"])
        self.assertEqual(patch["repairs"][0]["removed_values"], ["Q43229"])
        self.assertSchemaValid(patch)

    def test_diagnostics_multi_qualifier_delta_is_operation_visible(self) -> None:
        patch = gold_patch_for_record(
            self._record(
                summary={
                    "target_constraint_qid": "Q21510864",
                    "semantic_changed_qualifier_properties": ["P2305", "P2306"],
                    "semantic_added_values": [],
                    "semantic_removed_values": ["P361", "Q137041397"],
                },
            ),
            annotation={"decision_constraint_type_qid": "Q21510864"},
        )
        self.assertEqual([repair["repair_op"] for repair in patch["repairs"]], ["CONSTRAINT_QUALIFIER_REMOVE"] * 2)
        self.assertEqual([repair["evidence_level"] for repair in patch["repairs"]], ["OPERATION_VISIBLE"] * 2)
        self.assertEqual([repair["removed_values"] for repair in patch["repairs"]], [[], []])
        self.assertSchemaValid(patch)

    def test_diagnostics_causal_without_qualifier_delta_is_other_family_only(self) -> None:
        patch = gold_patch_for_record(self._record(summary={"target_constraint_qid": "Q21502410"}))
        self.assertEqual(patch["repairs"][0]["repair_op"], "OTHER_TBOX_UPDATE")
        self.assertEqual(patch["repairs"][0]["taxonomy_code"], "OTHER")
        self.assertEqual(patch["repairs"][0]["evidence_level"], "FAMILY_ONLY")
        self.assertSchemaValid(patch)

    def test_full_signature_delta_extracts_family_add_remove_and_replace(self) -> None:
        record = self._record()
        record["repair_target"]["constraint_delta"] = {
            "signature_before": [
                {"constraint_qid": "Q21510859", "snaktype": "VALUE", "rank": "normal", "qualifiers": []}
            ],
            "signature_after": [
                {"constraint_qid": "Q21502410", "snaktype": "VALUE", "rank": "normal", "qualifiers": []}
            ],
            "changed_constraint_types": ["Q21510859", "Q21502410"],
        }
        patch = gold_patch_for_record(record, annotation={"decision_constraint_type_qid": "Q21502410"})
        self.assertEqual(patch["repairs"][0]["repair_op"], "CONSTRAINT_TYPE_REPLACE")
        self.assertEqual(patch["repairs"][0]["old_value"], "Q21510859")
        self.assertEqual(patch["repairs"][0]["new_value"], "Q21502410")
        self.assertSchemaValid(patch)

    def test_full_signature_delta_extracts_deprecate(self) -> None:
        record = self._record()
        record["repair_target"]["constraint_delta"] = {
            "signature_before": [
                {"constraint_qid": "Q21510859", "snaktype": "VALUE", "rank": "normal", "qualifiers": []}
            ],
            "signature_after": [
                {"constraint_qid": "Q21510859", "snaktype": "VALUE", "rank": "deprecated", "qualifiers": []}
            ],
        }
        patch = gold_patch_for_record(record)
        self.assertEqual(patch["repairs"][0]["repair_op"], "CONSTRAINT_DEPRECATE")
        self.assertEqual(patch["repairs"][0]["rank_after"], "deprecated")
        self.assertSchemaValid(patch)

    def test_summary_contains_required_distributions(self) -> None:
        patches = [
            gold_patch_for_record(
                self._record(
                    subtype="RELAXATION_SET_EXPANSION",
                    summary={
                        "target_constraint_qid": "Q21510859",
                        "semantic_changed_qualifier_properties": ["P2305"],
                        "semantic_added_values": ["Q5"],
                    },
                )
            )
        ]
        summary = summarize_patches(
            patches,
            selected_records=1,
            selected_tbox_records=1,
            unsupported_case_ids=[],
        )
        self.assertEqual(summary["unsupported_count"], 0)
        self.assertEqual(summary["by_repair_op"], {"CONSTRAINT_QUALIFIER_ADD": 1})
        self.assertIn("class_hierarchy_delta_supported", summary)
        self.assertFalse(summary["exception_delta_supported"])

    def test_require_coverage_fails_when_selected_record_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            classified = tmp / "classified.jsonl"
            manifest = tmp / "manifest.json"
            classified.write_text("", encoding="utf-8")
            manifest.write_text(json.dumps({"selected_case_ids": ["missing-case"]}), encoding="utf-8")
            with self.assertRaises(CoverageError) as ctx:
                extract_selected_tbox_gold(
                    classified_benchmark=classified,
                    selection_manifest=manifest,
                    require_coverage=True,
                )
            self.assertEqual(ctx.exception.summary["unsupported_case_ids"], ["missing-case"])


if __name__ == "__main__":
    unittest.main()
