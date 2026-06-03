import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import validate

from lib.benchmark_selection import derive_case_metadata
from lib.manual_audit import (
    AUDIT_FIELDNAMES,
    AuditBuildOptions,
    apply_audit_policy,
    audit_annotation_completion,
    audit_annotation_schema,
    build_audit_sample,
    summarize_annotations,
    write_audit_csv,
    write_audit_policy_markdown,
)

CASE_CARD_SPEC = importlib.util.spec_from_file_location(
    "generate_manual_audit_case_cards",
    Path(__file__).resolve().parents[1] / "scripts" / "generate_manual_audit_case_cards.py",
)
case_card_module = importlib.util.module_from_spec(CASE_CARD_SPEC)
assert CASE_CARD_SPEC and CASE_CARD_SPEC.loader
CASE_CARD_SPEC.loader.exec_module(case_card_module)


def _record(
    case_id: str,
    cls: str,
    subtype: str,
    *,
    qid: str = "Q1",
    property_id: str = "P1",
    revision_id: str = "r1",
    truth_tokens: list[str] | None = None,
    bucket: str = "mid",
    local_ids_count: int = 10,
) -> dict:
    track = "T_BOX" if cls == "T_BOX" else "A_BOX"
    branch = {
        "TypeA": "delete_refined",
        "TypeB": "local_match",
        "TypeC": "external_by_elimination",
        "T_BOX": "missing",
    }.get(cls, "missing")
    record = {
        "id": case_id,
        "qid": qid,
        "property": property_id,
        "track": track,
        "classification": {
            "class": cls,
            "subtype": subtype,
            "confidence": "medium" if cls == "TypeC" else "high",
            "decision_trace": [
                {
                    "step": "local_availability",
                    "result": cls == "TypeB",
                    "evidence": {
                        "matched": cls == "TypeB",
                        "matches": [
                            {"token": "Q9", "kind": "id_exact", "source": "FOCUS_TEXT"}
                        ]
                        if cls == "TypeB"
                        else [],
                        "sources_used": ["FOCUS_TEXT"] if cls == "TypeB" else [],
                        "local_ids_count": local_ids_count,
                    },
                },
                {"step": "branch", "result": branch},
            ],
            "constraint_types": [{"qid": "Q21503250"}],
            "classification_rule_family": "local_evidence" if cls == "TypeB" else "test",
            "classification_rule_subfamily": subtype.lower(),
            "decision_constraint_type_qid": "Q21503250",
            "decision_constraint_type_label": "type constraint",
            "decision_constraint_source": "test_fixture",
            "diagnostics": {
                "truth_source": "repair_target.new_value" if truth_tokens else "none_expected",
                "truth_tokens": truth_tokens or [],
            },
        },
        "popularity": {"bucket": bucket, "score": 0.5},
    }
    if track == "T_BOX":
        record["repair_target"] = {"kind": "T_BOX", "property_revision_id": revision_id}
        record["classification"]["analysis_slice_precise"] = "main_tbox_relaxation_allowed_set_expansion"
        record["classification"]["decision_trace"].append(
            {
                "step": "tbox_causality",
                "result": subtype,
                "selected_violation_name": "One of",
                "candidate_violation_names": ["One of"],
                "mapped_report_constraint_qid": "Q21510859",
                "mapped_report_constraint_label": "one-of constraint",
                "mapped_report_family": "one_of",
                "target_constraint_qid": "Q21510859",
                "target_constraint_label": "one-of constraint",
                "target_constraint_selection_reason": "mapped_violation_constraint_changed",
                "target_constraint_selection_confidence": "high",
                "target_constraint_is_changed": True,
                "target_constraint_is_related_family": False,
                "compatible_value_overlap_with_report_qids": ["Q1"],
                "compatible_property_overlap_with_report_pids": [],
                "compatible_language_overlap_with_report_langs": [],
                "compatible_scope_overlap_with_report_values": [],
                "incompatible_overlap_ignored": {},
                "value_specific_without_overlap": False,
                "compatible_overlap_used": True,
                "compatible_overlap_reason": "one_of_compatible_report_argument_overlap",
                "semantic_changed_qualifier_properties": ["P2305"],
                "ignored_changed_qualifier_properties": ["P2316"],
                "semantic_added_values": ["Q1"],
                "semantic_removed_values": [],
                "ignored_added_values": ["Q2"],
                "ignored_removed_values": [],
                "qualifier_filter_reason": "family_relevant_qualifiers_kept_metadata_or_irrelevant_qualifiers_ignored",
                "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
                "set_semantics": "allowed",
                "set_operation": "expansion",
                "polarity": "relaxation",
                "polarity_basis": "allowed set gained values",
                "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
            }
        )
    return record


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _schema_update_tbox_record(case_id: str = "schema") -> dict:
    record = _record(case_id, "T_BOX", "SCHEMA_UPDATE", revision_id="schema-r1")
    record["classification"]["analysis_slice_precise"] = "main_tbox_schema_update"
    step = record["classification"]["decision_trace"][-1]
    step.update(
        {
            "result": "SCHEMA_UPDATE",
            "directional_subtype_precise": None,
            "analysis_slice_precise": "main_tbox_schema_update",
            "polarity": "unknown",
            "polarity_basis": "not active because final T-box subtype is non-directional",
            "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
            "potential_set_semantics": "allowed",
            "potential_set_operation": "expansion",
            "potential_polarity": "relaxation",
            "potential_polarity_basis": "allowed set gained values",
            "potential_directional_subtype_basis": "allowed set expansion",
        }
    )
    return record


def _write_manifest(path: Path, rows: list[dict]) -> None:
    annotations = {}
    ids = []
    for row in rows:
        metadata = derive_case_metadata(row, tier="core")
        if metadata is None:
            continue
        ids.append(row["id"])
        annotations[row["id"]] = metadata
    path.write_text(
        json.dumps({"selected_case_ids": ids, "case_annotations": annotations}),
        encoding="utf-8",
    )


class ManualAuditTests(unittest.TestCase):
    def _build(
        self,
        root: Path,
        rows: list[dict],
        quotas: dict[str, int],
        *,
        core_rows: list[dict] | None = None,
        dev_rows: list[dict] | None = None,
        tbox_cap: int = 5,
        abox_cap: int = 3,
    ) -> tuple[list[dict], dict]:
        classified = root / "classified.jsonl"
        core = root / "core.json"
        dev = root / "dev.json"
        _write_jsonl(classified, rows)
        _write_manifest(core, core_rows if core_rows is not None else rows)
        _write_manifest(dev, dev_rows or [])
        return build_audit_sample(
            AuditBuildOptions(
                classified_benchmark=classified,
                core_manifest=core,
                dev_manifest=dev,
                quotas=quotas,
                progress_every=0,
                tbox_cap_per_revision=tbox_cap,
                abox_cap_per_group=abox_cap,
            )
        )

    def test_audit_sample_emits_all_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [_record("c1", "TypeA", "REJECTION_FORMAT_INVALID")]
            sample, _ = self._build(Path(tmp_dir), rows, {"TypeA_REJECTION_FORMAT_INVALID": 1})

            self.assertEqual(set(AUDIT_FIELDNAMES), set(sample[0]))

    def test_audit_quotas_underfill_with_explicit_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [_record("c1", "TypeA", "REJECTION_FORMAT_INVALID")]
            sample, metadata = self._build(Path(tmp_dir), rows, {"TypeA_REJECTION_FORMAT_INVALID": 2})

            self.assertEqual(len(sample), 1)
            self.assertEqual(metadata["underfilled_quotas"][0]["selection_stratum"], "TypeA_REJECTION_FORMAT_INVALID")
            self.assertTrue(metadata["warnings"])

    def test_set_membership_rejection_audit_stratum_and_underfill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [_record("c1", "TypeA", "SET_MEMBERSHIP_REJECTION")]
            sample, metadata = self._build(Path(tmp_dir), rows, {"TypeA_SET_MEMBERSHIP_REJECTION": 2})

            self.assertEqual(len(sample), 1)
            self.assertEqual(sample[0]["selection_stratum"], "TypeA_SET_MEMBERSHIP_REJECTION")
            self.assertEqual(metadata["underfilled_quotas"][0]["selection_stratum"], "TypeA_SET_MEMBERSHIP_REJECTION")

    def test_local_text_derived_audit_stratum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [_record("c1", "TypeB", "LOCAL_TEXT_DERIVED", truth_tokens=["2007/si/483/made"])]
            sample, _ = self._build(Path(tmp_dir), rows, {"TypeB_LOCAL_TEXT_DERIVED": 1})

            self.assertEqual(sample[0]["selection_stratum"], "TypeB_LOCAL_TEXT_DERIVED")
            self.assertEqual(sample[0]["classification_rule_family"], "local_evidence")

    def test_tbox_unknown_causality_audit_stratum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [_record("t1", "T_BOX", "UNKNOWN_TBOX_CAUSALITY", revision_id="r1")]
            sample, _ = self._build(Path(tmp_dir), rows, {"TBOX_UNKNOWN_TBOX_CAUSALITY": 1})

            self.assertEqual(sample[0]["selection_stratum"], "TBOX_UNKNOWN_TBOX_CAUSALITY")

    def test_tbox_case_card_compact_diff_summary_uses_diagnostics(self) -> None:
        record = _record("t1", "T_BOX", "RELAXATION_SET_EXPANSION", revision_id="r1")
        record["classification"]["diagnostics"]["tbox_diff_summary"] = {
            "lean_stage4_pruned_full_signatures": True,
            "target_constraint_qid": "Q21510865",
            "changed_constraint_qids_all": ["Q21510865"],
            "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
        }

        summary = case_card_module.tbox_compact_diff_summary(record)

        self.assertTrue(summary["lean_stage4_pruned_full_signatures"])
        self.assertEqual(summary["target_constraint_qid"], "Q21510865")
        self.assertEqual(summary["directional_subtype_precise"], "RELAXATION_ALLOWED_SET_EXPANSION")

    def test_audit_row_includes_new_tbox_metadata_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [_record("t1", "T_BOX", "RELAXATION_SET_EXPANSION", revision_id="r1")]
            sample, _ = self._build(Path(tmp_dir), rows, {"TBOX_DIRECTIONAL_RELAXATION_OR_RESTRICTION": 1})

            self.assertIn("analysis_slice_precise", sample[0])
            self.assertEqual(sample[0]["directional_subtype_precise"], "RELAXATION_ALLOWED_SET_EXPANSION")
            self.assertEqual(sample[0]["semantic_changed_qualifier_properties"], "[\"P2305\"]")
            self.assertEqual(sample[0]["ignored_changed_qualifier_properties"], "[\"P2316\"]")
            self.assertEqual(sample[0]["compatible_overlap_used"], "true")

    def test_schema_update_audit_row_does_not_expose_active_precise_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [_schema_update_tbox_record("t-schema")]
            sample, _ = self._build(Path(tmp_dir), rows, {"TBOX_SCHEMA_UPDATE": 1})

            self.assertEqual(sample[0]["analysis_slice_precise"], "main_tbox_schema_update")
            self.assertEqual(sample[0]["directional_subtype_precise"], "")
            self.assertEqual(sample[0]["potential_directional_subtype_precise"], "RELAXATION_ALLOWED_SET_EXPANSION")

    def test_schema_update_case_card_keeps_potential_direction_separate(self) -> None:
        record = _schema_update_tbox_record("t-schema")

        causality = case_card_module.tbox_causality_summary(record)
        compact = case_card_module.tbox_compact_diff_summary(record)

        self.assertIsNone(causality["directional_subtype_precise"])
        self.assertEqual(causality["analysis_slice_precise"], "main_tbox_schema_update")
        self.assertEqual(causality["potential_directional_subtype_precise"], "RELAXATION_ALLOWED_SET_EXPANSION")
        self.assertIsNone(compact["directional_subtype_precise"])
        self.assertEqual(compact["analysis_slice_precise"], "main_tbox_schema_update")

    def test_max_tbox_revision_cap_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _record(f"t{i}", "T_BOX", "SCHEMA_UPDATE", property_id="P1", revision_id="r1")
                for i in range(8)
            ]
            sample, metadata = self._build(Path(tmp_dir), rows, {"TBOX_SCHEMA_UPDATE": 8}, tbox_cap=2)

            self.assertEqual(len(sample), 2)
            self.assertLessEqual(metadata["counts"]["max_tbox_per_revision"], 2)

    def test_max_abox_group_cap_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _record(f"a{i}", "TypeB", "LOCAL_TEXT_CONFIRMED", qid="Q1", property_id="P1", truth_tokens=["label"])
                for i in range(8)
            ]
            sample, metadata = self._build(Path(tmp_dir), rows, {"TypeB_LOCAL_TEXT_CONFIRMED": 8}, abox_cap=2)

            self.assertEqual(len(sample), 2)
            self.assertLessEqual(metadata["counts"]["max_abox_per_group"], 2)

    def test_dev_overlap_is_avoided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dev_case = _record("dev", "TypeA", "REJECTION_FORMAT_INVALID", qid="Q1")
            core_case = _record("core", "TypeA", "REJECTION_FORMAT_INVALID", qid="Q2")
            sample, metadata = self._build(
                root,
                [dev_case, core_case],
                {"TypeA_REJECTION_FORMAT_INVALID": 1},
                dev_rows=[dev_case],
            )

            self.assertEqual(sample[0]["case_id"], "core")
            self.assertEqual(metadata["counts"]["dev_overlap"], 0)

    def test_unknown_typec_quota_backfills_when_no_unknown_cases_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _record(f"c{i}", "TypeC", "EXTERNAL_BY_ELIMINATION", qid=f"Q{i}", truth_tokens=[f"Q{i}"], local_ids_count=i)
                for i in range(5)
            ]
            sample, metadata = self._build(Path(tmp_dir), rows, {"TypeC_UNKNOWN_OR_SPARSE_DIAGNOSTIC": 3})

            self.assertEqual(len(sample), 3)
            self.assertEqual({row["selection_stratum"] for row in sample}, {"TypeC_UNKNOWN_OR_SPARSE_DIAGNOSTIC"})
            self.assertTrue(any("backfilled" in warning for warning in metadata["warnings"]))

    def test_annotation_schema_validates_allowed_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [_record("c1", "TypeA", "REJECTION_FORMAT_INVALID")]
            sample, metadata = self._build(Path(tmp_dir), rows, {"TypeA_REJECTION_FORMAT_INVALID": 1})
            schema = audit_annotation_schema(metadata)
            row = dict(sample[0])
            row["typea_judgment"] = "clean_rule_or_format"
            row["tbox_judgment"] = "unknown_causality"
            row["core_recommendation"] = "main"

            validate(instance=row, schema=schema)
            bad = dict(row)
            bad["typea_judgment"] = "not_allowed"
            with self.assertRaises(Exception):
                validate(instance=bad, schema=schema)

    def test_summary_script_computes_metrics_on_annotated_csv(self) -> None:
        rows = [
            {
                **{field: "" for field in AUDIT_FIELDNAMES},
                "case_id": "typec",
                "class": "TypeC",
                "subtype": "EXTERNAL_BY_ELIMINATION",
                "selection_stratum": "TypeC_EXTERNAL_BY_ELIMINATION_QID_TRUTH",
                "main_score": "True",
                "typec_judgment": "external_confirmed",
                "core_recommendation": "main",
            },
            {
                **{field: "" for field in AUDIT_FIELDNAMES},
                "case_id": "typeb",
                "class": "TypeB",
                "subtype": "LOCAL_TEXT_CONFIRMED",
                "selection_stratum": "TypeB_LOCAL_TEXT_CONFIRMED",
                "main_score": "True",
                "typeb_judgment": "local_confirmed",
                "core_recommendation": "diagnostic",
            },
            {
                **{field: "" for field in AUDIT_FIELDNAMES},
                "case_id": "typea",
                "class": "TypeA",
                "subtype": "REJECTION_FORMAT_INVALID",
                "selection_stratum": "TypeA_REJECTION_FORMAT_INVALID",
                "main_score": "True",
                "typea_judgment": "overclaimed",
                "core_recommendation": "exclude",
            },
        ]

        summary = summarize_annotations(rows)

        self.assertEqual(summary["TypeC_confirmed_external_rate"]["rate"], 1.0)
        self.assertEqual(summary["TypeB_local_precision"]["rate"], 1.0)
        self.assertEqual(summary["TypeA_overclaim_rate"]["rate"], 1.0)
        self.assertEqual(summary["diagnostic_or_exclude_rate"]["rate"], 2 / 3)

    def test_label_precision_uses_stratum_specific_good_values(self) -> None:
        def row(case_id: str, cls: str, subtype: str, stratum: str, field: str, judgment: str) -> dict[str, str]:
            values = {name: "" for name in AUDIT_FIELDNAMES}
            values.update(
                {
                    "case_id": case_id,
                    "class": cls,
                    "subtype": subtype,
                    "selection_stratum": stratum,
                    field: judgment,
                }
            )
            return values

        rows = [
            row(
                "typec-bad-target",
                "TypeC",
                "UNKNOWN_BAD_TARGET_OR_CONTEXT",
                "TypeC_UNKNOWN_BAD_TARGET_OR_CONTEXT",
                "typec_judgment",
                "bad_target",
            ),
            row(
                "typec-elim",
                "TypeC",
                "EXTERNAL_BY_ELIMINATION",
                "TypeC_EXTERNAL_BY_ELIMINATION_QID_TRUTH",
                "typec_judgment",
                "external_by_elimination_ok",
            ),
            row(
                "tbox-unknown",
                "T_BOX",
                "UNKNOWN_TBOX_CAUSALITY",
                "TBOX_UNKNOWN_TBOX_CAUSALITY",
                "tbox_judgment",
                "unknown_causality",
            ),
            row(
                "typeb-selection",
                "TypeB",
                "LOCAL_SELECTION_CONFIRMED",
                "TypeB_LOCAL_SELECTION_CONFIRMED",
                "typeb_judgment",
                "local_confirmed",
            ),
        ]

        by_stratum = summarize_annotations(rows)["label_precision_by_stratum"]

        self.assertEqual(by_stratum["TypeC_UNKNOWN_BAD_TARGET_OR_CONTEXT"]["rate"], 1.0)
        self.assertEqual(by_stratum["TypeC_EXTERNAL_BY_ELIMINATION_QID_TRUTH"]["rate"], 1.0)
        self.assertEqual(by_stratum["TBOX_UNKNOWN_TBOX_CAUSALITY"]["rate"], 1.0)
        self.assertEqual(by_stratum["TypeB_LOCAL_SELECTION_CONFIRMED"]["rate"], 1.0)

    def test_summary_handles_unannotated_rows_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "audit.csv"
            row = {field: "" for field in AUDIT_FIELDNAMES}
            row.update({"case_id": "c1", "class": "TypeC", "subtype": "UNKNOWN_MISSING_TRUTH"})
            write_audit_csv([row], path)
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            summary = summarize_annotations(rows)

            self.assertEqual(summary["row_count"], 1)
            self.assertEqual(summary["unannotated_row_count"], 1)
            self.assertIsNone(summary["TypeC_unknown_or_incomplete_rate"]["rate"])

    def test_audit_policy_blocks_incomplete_annotations(self) -> None:
        row = {field: "" for field in AUDIT_FIELDNAMES}
        row.update({"case_id": "c1", "class": "TypeC", "subtype": "EXTERNAL_BY_ELIMINATION"})

        completion = audit_annotation_completion([row])
        policy = apply_audit_policy([row], require_complete=True)

        self.assertFalse(completion["ready_for_audit_policy"])
        self.assertEqual(policy["status"], "blocked_incomplete_annotations")
        self.assertEqual(policy["completion"]["unannotated_row_count"], 1)
        self.assertIn("missing", policy["recommendation_counts"])

    def test_audit_policy_collects_completed_recommendations(self) -> None:
        base = {field: "" for field in AUDIT_FIELDNAMES}
        annotated = {
            "repair_locus_correct": "yes",
            "historical_target_well_defined": "yes",
            "target_visible_locally": "no",
            "extractor_missed_local_evidence": "not_applicable",
            "external_evidence_required": "yes",
            "typec_judgment": "external_by_elimination_ok",
            "typea_judgment": "not_typea",
            "typeb_judgment": "not_typeb",
            "tbox_judgment": "not_tbox",
        }
        rows = [
            {**base, **annotated, "case_id": "main", "class": "TypeC", "core_recommendation": "main"},
            {**base, **annotated, "case_id": "diag", "class": "TypeC", "core_recommendation": "diagnostic"},
            {**base, **annotated, "case_id": "exclude", "class": "TypeC", "core_recommendation": "exclude"},
        ]

        policy = apply_audit_policy(rows, require_complete=True)

        self.assertEqual(policy["status"], "ready")
        self.assertEqual(policy["main_case_ids"], ["main"])
        self.assertEqual(policy["diagnostic_case_ids"], ["diag"])
        self.assertEqual(policy["exclude_case_ids"], ["exclude"])
        self.assertEqual(policy["recommendation_counts"], {"diagnostic": 1, "exclude": 1, "main": 1})

    def test_audit_policy_markdown_writes_blocked_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            row = {field: "" for field in AUDIT_FIELDNAMES}
            row.update({"case_id": "c1", "class": "TypeC", "subtype": "EXTERNAL_BY_ELIMINATION"})
            policy = apply_audit_policy([row])
            out = Path(tmp_dir) / "policy.md"

            write_audit_policy_markdown(policy, out)

            text = out.read_text(encoding="utf-8")
            self.assertIn("blocked_incomplete_annotations", text)
            self.assertIn("Blocking Condition", text)


if __name__ == "__main__":
    unittest.main()
