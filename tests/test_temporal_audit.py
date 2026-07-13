import json
import tempfile
import unittest
from pathlib import Path

from temporal_audit import audit_rendered_prompts, forbidden_claims


class TemporalAuditTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_forbidden_claims_cover_target_truth_and_hidden_metadata(self) -> None:
        claims = forbidden_claims(
            {
                "id": "repair_Q1_123456",
                "track": "A_BOX",
                "repair_target": {
                    "author": "HiddenEditor",
                    "new_value": ["Q99"],
                    "property_revision_id": 123456,
                },
                "persistence_check": {"current_value_2026": ["Q100"]},
                "classification": {"class": "TypeB", "subtype": "LOCAL_TEXT_CONFIRMED"},
            }
        )
        observed = {(claim["field"], claim["token"]) for claim in claims}
        self.assertIn(("repair_target.new_value", "Q99"), observed)
        self.assertIn(("persistence_check.current_value_2026", "Q100"), observed)
        self.assertIn(("repair_target.author", "HiddenEditor"), observed)

    def test_target_required_focus_qid_is_expected_rule_visibility(self) -> None:
        claims = forbidden_claims(
            {
                "id": "repair_Q9_123456",
                "qid": "Q9",
                "track": "A_BOX",
                "violation_context": {"value": ["MISSING"]},
                "repair_target": {"old_value": ["MISSING"], "new_value": ["Q9"]},
                "classification": {"class": "TypeA", "subtype": "TARGET_REQUIRED_CLAIM"},
            }
        )
        target_claim = next(claim for claim in claims if claim["field"] == "repair_target.new_value")
        self.assertEqual(target_claim["severity"], "expected_rule_derived")

    def test_audit_fails_high_risk_leak_and_emits_stratified_manual_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            self._write_jsonl(
                benchmark,
                [
                    {
                        "id": "repair_case_123456",
                        "track": "A_BOX",
                        "repair_target": {"new_value": ["Q999"], "author": "HiddenEditor"},
                        "persistence_check": {"current_value_2026": ["Q999"]},
                        "classification": {"class": "TypeB", "subtype": "LOCAL_TEXT_CONFIRMED"},
                    },
                    {
                        "id": "reform_case_654321",
                        "track": "T_BOX",
                        "repair_target": {"property_revision_id": 654321},
                        "classification": {"class": "T_BOX", "subtype": "SCHEMA_UPDATE"},
                    },
                ],
            )
            self._write_jsonl(
                prompts,
                [
                    {
                        "matrix_id": "m1",
                        "case_id": "repair_case_123456",
                        "task": "a_box_repair",
                        "context_bundle": "local_graph",
                        "historical_track": "A_BOX",
                        "system_prompt": "Use visible evidence only.",
                        "user_prompt": "The answer is Q999.",
                    },
                    {
                        "matrix_id": "m2",
                        "case_id": "reform_case_654321",
                        "task": "t_box_repair",
                        "context_bundle": "logic_only",
                        "historical_track": "T_BOX",
                        "system_prompt": "Use visible evidence only.",
                        "user_prompt": "No hidden revision is shown.",
                    },
                ],
            )

            report = audit_rendered_prompts(
                rendered_prompts_path=prompts,
                classified_benchmark_path=benchmark,
                sample_size=2,
                seed=7,
            )

            self.assertFalse(report["passed_automated_gate"])
            self.assertEqual(report["counts"]["high_risk_hits"], 2)
            self.assertEqual(len(report["manual_review_sample"]), 2)
            self.assertTrue(
                all(row["review_status"] == "pending_human_review" for row in report["manual_review_sample"])
            )

    def test_audit_passes_sanitized_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            self._write_jsonl(
                benchmark,
                [
                    {
                        "id": "repair_case_123456",
                        "track": "A_BOX",
                        "repair_target": {"new_value": ["Q999"]},
                        "classification": {"class": "TypeB", "subtype": "LOCAL_TEXT_CONFIRMED"},
                    }
                ],
            )
            self._write_jsonl(
                prompts,
                [
                    {
                        "matrix_id": "m1",
                        "case_id": "repair_case_123456",
                        "task": "a_box_repair",
                        "context_bundle": "minimal_case",
                        "historical_track": "A_BOX",
                        "system_prompt": "Use visible evidence only.",
                        "user_prompt": "Neutral case_000001 has an invalid value.",
                    }
                ],
            )

            report = audit_rendered_prompts(
                rendered_prompts_path=prompts,
                classified_benchmark_path=benchmark,
                sample_size=1,
            )

            self.assertTrue(report["passed_automated_gate"])
            self.assertEqual(report["counts"]["high_risk_hits"], 0)

    def test_target_token_does_not_match_inside_pre_repair_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            self._write_jsonl(
                benchmark,
                [
                    {
                        "id": "repair_case_123456",
                        "track": "A_BOX",
                        "violation_context": {"value": ["SCHEMBL54432"]},
                        "repair_target": {"old_value": ["SCHEMBL54432"], "new_value": ["54432"]},
                        "classification": {"class": "TypeA", "subtype": "FORMAT_NORMALIZATION"},
                    }
                ],
            )
            self._write_jsonl(
                prompts,
                [
                    {
                        "matrix_id": "m1",
                        "case_id": "repair_case_123456",
                        "task": "a_box_repair",
                        "context_bundle": "logic_only",
                        "historical_track": "A_BOX",
                        "system_prompt": "Use visible evidence only.",
                        "user_prompt": "The invalid old value is SCHEMBL54432.",
                    }
                ],
            )

            report = audit_rendered_prompts(
                rendered_prompts_path=prompts,
                classified_benchmark_path=benchmark,
                sample_size=1,
            )

            self.assertEqual(report["counts"]["high_risk_hits"], 0)


if __name__ == "__main__":
    unittest.main()
