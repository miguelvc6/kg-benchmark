import json
import tempfile
import unittest
from pathlib import Path

from temporal_audit import audit_rendered_prompts, forbidden_claims, mutation_sensitivity_checks


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

    def test_target_required_focus_label_is_expected_rule_visibility(self) -> None:
        claims = forbidden_claims(
            {
                "id": "repair_Q9_123456",
                "qid": "Q9",
                "track": "A_BOX",
                "repair_target": {
                    "old_value": ["MISSING"],
                    "new_value": ["Q9"],
                    "new_value_labels_en": ["Focus label"],
                },
                "classification": {"class": "TypeA", "subtype": "TARGET_REQUIRED_CLAIM"},
            }
        )
        label_claim = next(claim for claim in claims if claim["field"] == "repair_target.new_value_labels_en")
        self.assertEqual(label_claim["severity"], "expected_rule_derived")

    def test_retained_target_label_in_mixed_update_is_expected_historical(self) -> None:
        claims = forbidden_claims(
            {
                "id": "repair_Q9_123456",
                "track": "A_BOX",
                "violation_context": {"value_labels_en": ["Retained label"]},
                "repair_target": {
                    "old_value": ["Q1", "Q2"],
                    "old_value_labels_en": ["Removed label", "Retained label"],
                    "new_value": ["Q2", "Q3"],
                    "new_value_labels_en": ["Retained label", "Added label"],
                },
                "classification": {"class": "TypeA", "subtype": "SET_MEMBERSHIP_REJECTION"},
            }
        )
        retained = next(
            claim
            for claim in claims
            if claim["field"] == "repair_target.new_value_labels_en" and claim["token"] == "Retained label"
        )
        added = next(
            claim
            for claim in claims
            if claim["field"] == "repair_target.new_value_labels_en" and claim["token"] == "Added label"
        )
        self.assertEqual(retained["severity"], "expected_historical")
        self.assertEqual(added["severity"], "high")

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
            self.assertTrue(report["passed_case_exclusion_gate"])
            self.assertEqual(report["counts"]["high_risk_hits"], 2)
            self.assertEqual(len(report["manual_review_sample"]), 2)
            self.assertTrue(all(row["review_status"] == "pending_ai_review" for row in report["manual_review_sample"]))

    def test_mutation_sensitivity_checks_cover_normalized_and_boundary_cases(self) -> None:
        self.assertTrue(all(mutation_sensitivity_checks().values()))

    def test_encoded_target_value_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            self._write_jsonl(benchmark, [{
                "id": "repair_encoded", "track": "A_BOX", "repair_target": {"new_value": ["Q999"]},
                "classification": {"class": "TypeB", "subtype": "LOCAL_TEXT_CONFIRMED"},
            }])
            self._write_jsonl(prompts, [{
                "matrix_id": "encoded", "case_id": "repair_encoded", "task": "a_box_repair",
                "context_bundle": "local_graph", "historical_track": "A_BOX", "system_prompt": "neutral",
                "user_prompt": "payload=UTk5OQ==",
            }])
            report = audit_rendered_prompts(
                rendered_prompts_path=prompts, classified_benchmark_path=benchmark, sample_size=1
            )
            self.assertFalse(report["passed_automated_gate"])
            self.assertEqual(report["hits"][0]["match_mode"], "encoded_value")

    def test_tbox_current_value_embedded_in_url_is_high_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            self._write_jsonl(benchmark, [{
                "id": "reform_url", "track": "T_BOX",
                "repair_target": {"property_revision_id": 123456},
                "persistence_check": {"current_value_2026": ["32013L0012"]},
                "classification": {"class": "T_BOX", "subtype": "SCHEMA_UPDATE"},
            }])
            self._write_jsonl(prompts, [{
                "matrix_id": "url", "case_id": "reform_url", "task": "t_box_repair",
                "context_bundle": "local_graph", "historical_track": "T_BOX", "system_prompt": "neutral",
                "user_prompt": "source=https://example.test/?uri=CELEX:32013L0012",
            }])
            report = audit_rendered_prompts(
                rendered_prompts_path=prompts, classified_benchmark_path=benchmark, sample_size=1
            )
            self.assertFalse(report["passed_automated_gate"])
            self.assertEqual(report["counts"]["high_risk_hits"], 1)

    def test_future_version_alias_is_high_but_shared_cross_qid_label_is_historical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            self._write_jsonl(benchmark, [
                {
                    "id": "repair_alias", "track": "A_BOX",
                    "violation_context": {"value": ["Q1"], "value_labels_en": ["Replacement office"]},
                    "repair_target": {
                        "old_value": ["Q1"], "old_value_labels_en": ["Replacement office"],
                        "new_value": ["Q2"], "new_value_labels_en": ["Replacement office"],
                    },
                    "classification": {"class": "TypeB", "subtype": "LOCAL_TEXT_CONFIRMED"},
                },
                {
                    "id": "repair_version", "track": "A_BOX",
                    "repair_target": {"new_value": ["PocGH01_00229100.1"]},
                    "classification": {"class": "TypeB", "subtype": "LOCAL_TEXT_CONFIRMED"},
                },
            ])
            self._write_jsonl(prompts, [
                {
                    "matrix_id": "alias", "case_id": "repair_alias", "task": "a_box_repair",
                    "context_bundle": "local_graph", "historical_track": "A_BOX", "system_prompt": "neutral",
                    "user_prompt": "Visible Q1 is labeled Replacement office.",
                },
                {
                    "matrix_id": "version", "case_id": "repair_version", "task": "a_box_repair",
                    "context_bundle": "local_graph", "historical_track": "A_BOX", "system_prompt": "neutral",
                    "user_prompt": "encoded by PocGH01_00229100",
                },
            ])
            report = audit_rendered_prompts(
                rendered_prompts_path=prompts, classified_benchmark_path=benchmark, sample_size=2
            )
            self.assertFalse(report["passed_automated_gate"])
            modes = {hit["match_mode"] for hit in report["hits"] if hit["severity"] == "high"}
            self.assertIn("semantic_alias", modes)
            self.assertEqual(report["counts"]["high_risk_hits"], 1)
            alias_hit = next(hit for hit in report["hits"] if hit["case_id"] == "repair_alias")
            self.assertEqual(alias_hit["severity"], "expected_historical")

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

    def test_format_normalization_only_explains_occurrence_inside_historical_literal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            self._write_jsonl(benchmark, [{
                "id": "repair_format", "track": "A_BOX",
                "violation_context": {"value": ["uk.bl.ethos.490072"]},
                "repair_target": {
                    "old_value": ["uk.bl.ethos.490072"],
                    "new_value": ["490072"],
                },
                "classification": {"class": "TypeA", "subtype": "FORMAT_NORMALIZATION"},
            }])
            self._write_jsonl(prompts, [{
                "matrix_id": "format", "case_id": "repair_format", "task": "a_box_repair",
                "context_bundle": "logic_only", "historical_track": "A_BOX", "system_prompt": "neutral",
                "user_prompt": "Historical value uk.bl.ethos.490072; leaked result 490072.",
            }])

            report = audit_rendered_prompts(
                rendered_prompts_path=prompts, classified_benchmark_path=benchmark, sample_size=1
            )

            target_hits = [hit for hit in report["hits"] if hit["field"] == "repair_target.new_value"]
            self.assertEqual(
                [hit["severity"] for hit in target_hits],
                ["expected_rule_derived", "high"],
            )
            self.assertEqual(report["counts"]["high_risk_hits"], 1)

    def test_json_escaped_historical_description_covers_nested_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            historical = 'actor in a production; use "voice actor" for voice roles'
            self._write_jsonl(benchmark, [{
                "id": "reform_escaped", "track": "T_BOX",
                "labels_en": {"property": {"description": historical}},
                "persistence_check": {"current_value_2026_descriptions_en": ["actor"]},
                "classification": {"class": "T_BOX", "subtype": "SCHEMA_UPDATE"},
            }])
            self._write_jsonl(prompts, [{
                "matrix_id": "escaped", "case_id": "reform_escaped", "task": "t_box_repair",
                "context_bundle": "logic_only", "historical_track": "T_BOX", "system_prompt": "neutral",
                "user_prompt": "Input case:\n" + json.dumps({
                    "labels_en": {"property": {"description": historical}},
                }, indent=2),
            }])

            report = audit_rendered_prompts(
                rendered_prompts_path=prompts, classified_benchmark_path=benchmark, sample_size=1
            )

            self.assertTrue(report["passed_automated_gate"])
            self.assertEqual(report["counts"]["high_risk_hits"], 0)
            self.assertEqual(report["counts"]["expected_historical_hits"], 2)

    def test_short_author_word_inside_visible_phrase_is_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            self._write_jsonl(benchmark, [{
                "id": "repair_author_collision", "track": "A_BOX",
                "repair_target": {"author": "Trade"},
                "classification": {"class": "TypeA", "subtype": "DELETE_AMBIGUOUS"},
            }])
            self._write_jsonl(prompts, [{
                "matrix_id": "author", "case_id": "repair_author_collision", "task": "a_box_repair",
                "context_bundle": "local_graph", "historical_track": "A_BOX", "system_prompt": "neutral",
                "user_prompt": "Input case:\n" + json.dumps({
                    "local_context": {"label": "World Trade Center"},
                }, indent=2),
            }])

            report = audit_rendered_prompts(
                rendered_prompts_path=prompts, classified_benchmark_path=benchmark, sample_size=1
            )

            self.assertTrue(report["passed_automated_gate"])
            self.assertEqual(report["counts"]["high_risk_hits"], 0)
            self.assertEqual(report["hits"][0]["severity"], "diagnostic")

    def test_missing_sentinel_in_prompt_contract_is_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            self._write_jsonl(benchmark, [{
                "id": "repair_delete", "track": "A_BOX",
                "repair_target": {"old_value": ["Q1"], "new_value": ["MISSING"]},
                "classification": {"class": "TypeA", "subtype": "DELETE_AMBIGUOUS"},
            }])
            self._write_jsonl(prompts, [{
                "matrix_id": "missing", "case_id": "repair_delete", "task": "a_box_repair",
                "context_bundle": "logic_only", "historical_track": "A_BOX",
                "system_prompt": "Use visible evidence only.",
                "user_prompt": "MISSING is the benchmark sentinel for an absent claim value.",
            }])

            report = audit_rendered_prompts(
                rendered_prompts_path=prompts, classified_benchmark_path=benchmark, sample_size=1
            )

            self.assertTrue(report["passed_automated_gate"])
            self.assertEqual(report["counts"]["high_risk_hits"], 0)
            self.assertEqual(report["hits"][0]["severity"], "diagnostic")
            self.assertEqual(report["hits"][0]["classification_reason"], "prompt_contract_vocabulary")

    def test_type_b_aligned_label_uses_recorded_local_value_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            self._write_jsonl(benchmark, [{
                "id": "repair_local_label", "track": "A_BOX",
                "repair_target": {
                    "old_value": ["Q1"], "new_value": ["Q999"],
                    "new_value_labels_en": ["Farhan Rana Rajpoot"],
                },
                "classification": {
                    "class": "TypeB", "subtype": "LOCAL_FOCUS_NON_TARGET_PROPERTY",
                    "decision_trace": [{
                        "step": "local_availability", "result": True,
                        "evidence": {"matches": [{
                            "token": "Q999", "source": "FOCUS_NON_TARGET_PROPERTY",
                            "independent_of_target_property": True,
                        }]},
                    }],
                },
            }])
            self._write_jsonl(prompts, [{
                "matrix_id": "local-label", "case_id": "repair_local_label", "task": "a_box_repair",
                "context_bundle": "local_graph", "historical_track": "A_BOX", "system_prompt": "neutral",
                "user_prompt": "Input case:\n" + json.dumps({
                    "local_context": {
                        "description": "film directed by Farhan Rana Rajpoot",
                        "properties": {"P57": ["Q999"]},
                    },
                }, indent=2),
            }])

            report = audit_rendered_prompts(
                rendered_prompts_path=prompts, classified_benchmark_path=benchmark, sample_size=1
            )

            severities = {
                hit["field"]: hit["severity"]
                for hit in report["hits"]
                if hit["field"].startswith("repair_target.new_value")
            }
            self.assertTrue(report["passed_automated_gate"])
            self.assertEqual(severities["repair_target.new_value"], "expected_local_evidence")
            self.assertEqual(
                severities["repair_target.new_value_labels_en"], "expected_local_evidence"
            )

    def test_focus_qid_current_value_overlap_is_expected_rule_derived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            self._write_jsonl(benchmark, [{
                "id": "repair_focus", "qid": "Q9", "track": "A_BOX",
                "repair_target": {"old_value": ["MISSING"], "new_value": ["Q9"]},
                "persistence_check": {"current_value_2026": ["Q9"]},
                "classification": {"class": "TypeA", "subtype": "TARGET_REQUIRED_CLAIM"},
            }])
            self._write_jsonl(prompts, [{
                "matrix_id": "focus", "case_id": "repair_focus", "task": "a_box_repair",
                "context_bundle": "logic_only", "historical_track": "A_BOX", "system_prompt": "neutral",
                "user_prompt": "Input case:\n{\"qid\": \"Q9\"}",
            }])

            report = audit_rendered_prompts(
                rendered_prompts_path=prompts, classified_benchmark_path=benchmark, sample_size=1
            )

            self.assertTrue(report["passed_automated_gate"])
            self.assertGreaterEqual(report["counts"]["expected_rule_derived_hits"], 2)
            self.assertEqual(report["counts"]["high_risk_hits"], 0)

    def test_type_b_local_evidence_is_expected_only_in_visible_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            record = {
                "id": "repair_local", "track": "A_BOX",
                "repair_target": {"old_value": ["Q1"], "new_value": ["Q999"]},
                "classification": {
                    "class": "TypeB", "subtype": "LOCAL_NEIGHBOR_IDS",
                    "decision_trace": [{
                        "step": "local_availability", "result": True,
                        "evidence": {"matches": [{
                            "token": "Q999", "kind": "id_exact", "source": "NEIGHBOR_ID",
                            "independent_of_target_property": True,
                        }]},
                    }],
                },
            }
            self._write_jsonl(benchmark, [record])
            self._write_jsonl(prompts, [
                {
                    "matrix_id": "local", "case_id": "repair_local", "task": "a_box_repair",
                    "context_bundle": "local_graph", "historical_track": "A_BOX", "system_prompt": "neutral",
                    "user_prompt": "Input case:\n{\"local_context\": {\"neighbor\": \"Q999\"}}",
                },
                {
                    "matrix_id": "logic", "case_id": "repair_local", "task": "a_box_repair",
                    "context_bundle": "logic_only", "historical_track": "A_BOX", "system_prompt": "neutral",
                    "user_prompt": "Input case:\n{\"logic_context\": {\"unexpected\": \"Q999\"}}",
                },
            ])

            report = audit_rendered_prompts(
                rendered_prompts_path=prompts, classified_benchmark_path=benchmark, sample_size=1
            )

            severities = {
                hit["matrix_id"]: hit["severity"]
                for hit in report["hits"]
                if hit["field"] == "repair_target.new_value"
            }
            self.assertEqual(severities["local"], "expected_local_evidence")
            self.assertEqual(severities["logic"], "high")

    def test_future_only_alias_and_hidden_revision_author_metadata_remain_high(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = root / "stage4.jsonl"
            prompts = root / "prompts.jsonl"
            self._write_jsonl(benchmark, [{
                "id": "repair_metadata", "track": "A_BOX",
                "repair_target": {
                    "old_value": ["Q1"], "new_value": ["Q2"],
                    "new_value_aliases_en": [["Future-only alias"]],
                    "author": "HiddenEditor", "revision_id": 987654,
                    "constraint_delta": {"signature_after": [{"constraint_qid": "Q777777"}]},
                },
                "classification": {"class": "TypeC", "subtype": "EXTERNAL_BY_ELIMINATION"},
            }])
            self._write_jsonl(prompts, [{
                "matrix_id": "metadata", "case_id": "repair_metadata", "task": "a_box_repair",
                "context_bundle": "local_graph", "historical_track": "A_BOX", "system_prompt": "neutral",
                "user_prompt": "Future-only alias by HiddenEditor at revision 987654 with Q777777.",
            }])

            report = audit_rendered_prompts(
                rendered_prompts_path=prompts, classified_benchmark_path=benchmark, sample_size=1
            )

            high_fields = {hit["field"] for hit in report["hits"] if hit["severity"] == "high"}
            self.assertIn("repair_target.new_value_aliases_en", high_fields)
            self.assertIn("repair_target.author", high_fields)
            self.assertIn("repair_target.revision_id", high_fields)
            self.assertIn("repair_target.constraint_delta.signature_after", high_fields)
            self.assertEqual(report["report_version"], 5)
            self.assertEqual(report["excluded_case_ids"], ["repair_metadata"])
            for severity in (
                "high", "expected_historical", "expected_rule_derived", "expected_local_evidence", "diagnostic"
            ):
                self.assertIn(severity, report["raw_hits_by_severity"])
                self.assertIn(severity, report["unique_cases_by_severity"])


if __name__ == "__main__":
    unittest.main()
