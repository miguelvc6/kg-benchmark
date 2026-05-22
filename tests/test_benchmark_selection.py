import json
import tempfile
import unittest
from pathlib import Path

from lib.benchmark_selection import (
    SelectionOptions,
    build_tier_manifest,
    derive_case_metadata,
    load_selection_manifest,
    resolve_case_id_filter,
)


def _record(
    case_id: str,
    cls: str,
    subtype: str,
    *,
    confidence: str = "high",
    qid: str | None = None,
    property_id: str = "P1",
    revision_id: str | None = None,
    bucket: str = "mid",
) -> dict:
    track = "T_BOX" if cls == "T_BOX" else "A_BOX"
    record = {
        "id": case_id,
        "track": track,
        "property": property_id,
        "classification": {
            "class": cls,
            "subtype": subtype,
            "confidence": confidence,
            "diagnostics": {"truth_source": "synthetic", "truth_tokens": ["Q1"]},
            "constraint_types": [{"qid": "Q21503250"}],
        },
        "popularity": {"bucket": bucket, "score": 0.5},
    }
    if track == "T_BOX":
        record["repair_target"] = {"kind": "T_BOX"}
        if revision_id is not None:
            record["repair_target"]["property_revision_id"] = revision_id
    else:
        if qid is not None:
            record["qid"] = qid
    return record


class BenchmarkSelectionTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def _manifest(
        self,
        classified_path: Path,
        *,
        tier: str = "core",
        quotas: dict[str, int],
        seed: int = 13,
        selected_case_order: str = "sorted",
        exclude_manifest: Path | None = None,
        tbox_cap_core: int = 10,
        tbox_cap_dev: int = 3,
        abox_cap_core: int = 3,
        abox_cap_dev: int = 2,
    ) -> dict:
        target = sum(quotas.values())
        return build_tier_manifest(
            SelectionOptions(
                classified_benchmark=classified_path,
                tier=tier,
                seed=seed,
                core_size=target,
                dev_size=target,
                tbox_cap_core=tbox_cap_core,
                tbox_cap_dev=tbox_cap_dev,
                abox_cap_core=abox_cap_core,
                abox_cap_dev=abox_cap_dev,
                selected_case_order=selected_case_order,
                progress_every=0,
                exclude_manifest=exclude_manifest,
                quotas=quotas,
            )
        )

    def test_same_seed_produces_identical_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "classified.jsonl"
            rows = [_record(f"case_{i}", "TypeB", "LOCAL_TEXT", qid=f"Q{i}") for i in range(10)]
            self._write_jsonl(path, rows)

            first = self._manifest(path, quotas={"TypeB_LOCAL_TEXT": 5}, selected_case_order="shuffled")
            second = self._manifest(path, quotas={"TypeB_LOCAL_TEXT": 5}, selected_case_order="shuffled")

            self.assertEqual(first["selected_case_ids"], second["selected_case_ids"])
            self.assertEqual(first["case_annotations"], second["case_annotations"])

    def test_different_seed_changes_order_but_keeps_quota_validity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "classified.jsonl"
            rows = [_record(f"case_{i}", "TypeB", "LOCAL_TEXT", qid=f"Q{i}") for i in range(20)]
            self._write_jsonl(path, rows)

            first = self._manifest(path, quotas={"TypeB_LOCAL_TEXT": 6}, seed=13, selected_case_order="shuffled")
            second = self._manifest(path, quotas={"TypeB_LOCAL_TEXT": 6}, seed=17, selected_case_order="shuffled")

            self.assertNotEqual(first["selected_case_ids"], second["selected_case_ids"])
            self.assertEqual(first["counts"]["selected"], 6)
            self.assertEqual(second["counts"]["selected"], 6)
            self.assertTrue(first["validation"]["hard_validation_passed"])
            self.assertTrue(second["validation"]["hard_validation_passed"])

    def test_dev_and_core_case_ids_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            path = root / "classified.jsonl"
            rows = [
                _record(f"tbox_{i}", "T_BOX", "RELAXATION_SET_EXPANSION", property_id=f"P{i}", revision_id=f"r{i}")
                for i in range(8)
            ]
            self._write_jsonl(path, rows)

            dev = self._manifest(path, tier="dev", quotas={"DEV_TBOX_RELAXATION_SET_EXPANSION": 2})
            dev_path = root / "dev.json"
            dev_path.write_text(json.dumps(dev), encoding="utf-8")
            core = self._manifest(path, quotas={"TBOX_RELAXATION_SET_EXPANSION": 4}, exclude_manifest=dev_path)

            self.assertFalse(set(dev["selected_case_ids"]) & set(core["selected_case_ids"]))
            self.assertEqual(core["validation"]["dev_core_case_overlap"], 0)

    def test_dev_and_core_tbox_revision_keys_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            path = root / "classified.jsonl"
            rows = [
                _record(f"tbox_{i}", "T_BOX", "SCHEMA_UPDATE", property_id=f"P{i}", revision_id=f"r{i}")
                for i in range(8)
            ]
            self._write_jsonl(path, rows)

            dev = self._manifest(path, tier="dev", quotas={"DEV_TBOX_SCHEMA_UPDATE": 2})
            dev_path = root / "dev.json"
            dev_path.write_text(json.dumps(dev), encoding="utf-8")
            core = self._manifest(path, quotas={"TBOX_SCHEMA_UPDATE": 4}, exclude_manifest=dev_path)

            dev_keys = {ann["tbox_revision_key"] for ann in dev["case_annotations"].values()}
            core_keys = {ann["tbox_revision_key"] for ann in core["case_annotations"].values()}
            self.assertFalse(dev_keys & core_keys)
            self.assertEqual(core["validation"]["dev_core_tbox_revision_overlap"], 0)

    def test_core_tbox_cap_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "classified.jsonl"
            rows = [
                _record(f"tbox_{i}", "T_BOX", "RELAXATION_SET_EXPANSION", property_id="P1", revision_id="r1")
                for i in range(8)
            ]
            self._write_jsonl(path, rows)

            manifest = self._manifest(path, quotas={"TBOX_RELAXATION_SET_EXPANSION": 5}, tbox_cap_core=2)

            self.assertLessEqual(manifest["validation"]["max_tbox_per_revision"], 2)
            self.assertEqual(manifest["counts"]["selected"], 2)

    def test_dev_tbox_cap_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "classified.jsonl"
            rows = [
                _record(f"tbox_{i}", "T_BOX", "SCHEMA_UPDATE", property_id="P1", revision_id="r1")
                for i in range(8)
            ]
            self._write_jsonl(path, rows)

            manifest = self._manifest(
                path,
                tier="dev",
                quotas={"DEV_TBOX_SCHEMA_UPDATE": 4},
                tbox_cap_dev=1,
            )

            self.assertLessEqual(manifest["validation"]["max_tbox_per_revision"], 1)
            self.assertEqual(manifest["counts"]["selected"], 1)

    def test_diagnostic_only_subtypes_are_excluded_from_main_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "classified.jsonl"
            rows = [
                _record("ambiguous_delete", "TypeA", "DELETE_AMBIGUOUS", qid="Q1"),
                _record("coincidental", "T_BOX", "COINCIDENTAL_SCHEMA_CHANGE", property_id="P1", revision_id="r1"),
            ]
            self._write_jsonl(path, rows)

            manifest = self._manifest(
                path,
                quotas={"TypeA_DELETE_AMBIGUOUS": 1, "TBOX_COINCIDENTAL_SCHEMA_CHANGE": 1},
            )

            self.assertEqual(manifest["main_score_case_ids"], [])
            self.assertEqual(set(manifest["diagnostic_case_ids"]), {"ambiguous_delete", "coincidental"})

    def test_low_confidence_and_unknown_typec_are_excluded_from_main_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "classified.jsonl"
            rows = [
                _record("external_low", "TypeC", "EXTERNAL_BY_ELIMINATION", confidence="low", qid="Q1"),
                _record("unknown", "TypeC", "UNKNOWN_INCOMPLETE_LOCAL_CONTEXT", confidence="high", qid="Q2"),
            ]
            self._write_jsonl(path, rows)

            manifest = self._manifest(
                path,
                quotas={"TypeC_EXTERNAL_BY_ELIMINATION": 1, "TypeC_UNKNOWN_INCOMPLETE_LOCAL_CONTEXT": 1},
            )

            self.assertEqual(manifest["main_score_case_ids"], [])
            self.assertEqual(set(manifest["diagnostic_case_ids"]), {"external_low", "unknown"})

    def test_rare_subtype_underfill_records_underfilled_quota_and_backfills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "classified.jsonl"
            rows = [_record("rare", "TypeA", "REJECTION_RULE_INVALID", qid="Q0")]
            rows.extend(_record(f"fmt_{i}", "TypeA", "REJECTION_FORMAT_INVALID", qid=f"Q{i}") for i in range(1, 5))
            self._write_jsonl(path, rows)

            manifest = self._manifest(
                path,
                quotas={"TypeA_REJECTION_RULE_INVALID": 2, "TypeA_REJECTION_FORMAT_INVALID": 2},
            )

            self.assertEqual(manifest["counts"]["selected"], 4)
            self.assertEqual(manifest["underfilled_quotas"][0]["selection_stratum"], "TypeA_REJECTION_RULE_INVALID")
            self.assertIn("quota_backfill_for", manifest["case_annotations"]["fmt_3"])

    def test_every_selected_id_has_case_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "classified.jsonl"
            rows = [_record(f"case_{i}", "TypeB", "LOCAL_TEXT", qid=f"Q{i}") for i in range(5)]
            self._write_jsonl(path, rows)

            manifest = self._manifest(path, quotas={"TypeB_LOCAL_TEXT": 5})

            self.assertEqual(set(manifest["selected_case_ids"]), set(manifest["case_annotations"]))

    def test_manifest_counts_equal_selected_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "classified.jsonl"
            rows = [_record(f"case_{i}", "TypeB", "LOCAL_TEXT", qid=f"Q{i}") for i in range(4)]
            self._write_jsonl(path, rows)

            manifest = self._manifest(path, quotas={"TypeB_LOCAL_TEXT": 4})

            self.assertEqual(manifest["counts"]["selected"], len(manifest["selected_case_ids"]))
            self.assertEqual(
                manifest["counts"]["main_score"] + manifest["counts"]["diagnostic"],
                len(manifest["selected_case_ids"]),
            )

    def test_group_key_fallback_marks_weak_group_key(self) -> None:
        tbox = _record("tbox_missing_revision", "T_BOX", "SCHEMA_UPDATE", property_id="P1", revision_id=None)
        abox = _record("abox_missing_qid", "TypeB", "LOCAL_TEXT", qid=None, property_id="P2")

        self.assertTrue(derive_case_metadata(tbox)["weak_group_key"])
        self.assertTrue(derive_case_metadata(abox)["weak_group_key"])

    def test_resolve_case_id_filter_intersects_manifest_and_explicit_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "selection.json"
            manifest_path.write_text(json.dumps({"selected_case_ids": ["case_a", "case_b", "case_c"]}), encoding="utf-8")

            manifest = load_selection_manifest(manifest_path)
            self.assertEqual(manifest["selected_case_ids"], ["case_a", "case_b", "case_c"])

            resolved = resolve_case_id_filter(case_ids=["case_b", "case_x"], selection_manifest_path=manifest_path)
            self.assertEqual(resolved, ["case_b"])

    def test_resolve_case_id_filter_preserves_manifest_relative_order_when_intersecting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "selection.json"
            manifest_path.write_text(json.dumps({"selected_case_ids": ["case_c", "case_a", "case_b"]}), encoding="utf-8")

            resolved = resolve_case_id_filter(case_ids=["case_b", "case_c"], selection_manifest_path=manifest_path)
            self.assertEqual(resolved, ["case_c", "case_b"])


if __name__ == "__main__":
    unittest.main()
