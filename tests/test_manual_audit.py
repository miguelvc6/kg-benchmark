import csv
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import validate

from lib.benchmark_selection import derive_case_metadata
from lib.manual_audit import (
    AUDIT_FIELDNAMES,
    AuditBuildOptions,
    audit_annotation_schema,
    build_audit_sample,
    summarize_annotations,
    write_audit_csv,
)


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
            "diagnostics": {
                "truth_source": "repair_target.new_value" if truth_tokens else "none_expected",
                "truth_tokens": truth_tokens or [],
            },
        },
        "popularity": {"bucket": bucket, "score": 0.5},
    }
    if track == "T_BOX":
        record["repair_target"] = {"kind": "T_BOX", "property_revision_id": revision_id}
    return record


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


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


if __name__ == "__main__":
    unittest.main()
