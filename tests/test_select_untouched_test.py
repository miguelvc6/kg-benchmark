import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from select_untouched_test import (
    build_reserve_manifest,
    build_untouched_test_manifest,
    finalize_reserve_manifest,
    validate_untouched_test_manifest,
    write_allocation_manifests,
)


def _record(case_id: str, qid: str, property_id: str) -> dict:
    return {
        "id": case_id,
        "qid": qid,
        "property": property_id,
        "track": "A_BOX",
        "repair_target": {"kind": "A_BOX"},
        "classification": {
            "class": "TypeB",
            "subtype": "LOCAL_TEXT_CONFIRMED",
            "confidence": "high",
            "diagnostics": {"truth_tokens": ["Q9"], "truth_source": "repair_target"},
        },
        "popularity": {"score": 0.5},
    }


class UntouchedTestSelectionTests(unittest.TestCase):
    def _build(self, root: Path) -> tuple[dict, Path, Path]:
        classified = root / "classified.jsonl"
        rows = [
            _record("old", "Q1", "P1"),
            _record("old_group_peer", "Q1", "P1"),
            _record("new_a", "Q2", "P2"),
            _record("new_b", "Q2", "P2"),
        ]
        classified.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        exclusion = root / "exclude.json"
        # Deliberately omit case_annotations: exclusions must be reconstructed from Stage 4.
        exclusion.write_text(json.dumps({"selected_case_ids": ["old"]}), encoding="utf-8")
        protocol = root / "protocol.json"
        protocol.write_text("{}", encoding="utf-8")
        verification = {
            "passed": True,
            "manifest": {
                "protocol_id": "protocol_v1",
                "protocol_phase": "allocation",
                "release": {"release_kind": "dataset"},
                "expected_population": {"selected_count": 2, "main_score_count": 2},
            },
        }
        with patch("select_untouched_test.verify_protocol_manifest", return_value=verification):
            manifest = build_untouched_test_manifest(
                classified_path=classified,
                exclude_manifests=[exclusion],
                stratum_targets={"TypeB_LOCAL_TEXT_CONFIRMED": 2},
                protocol_manifest_path=protocol,
                protocol_root=root,
                property_holdout=True,
            )
        return manifest, classified, exclusion

    def test_exclusions_are_reconstructed_and_complete_groups_are_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest, _, _ = self._build(root)
            self.assertEqual(set(manifest["selected_case_ids"]), {"new_a", "new_b"})
            self.assertTrue(manifest["validation"]["passed"])
            self.assertEqual(manifest["validation"]["overlap_counts"], {"cases": 0, "groups": 0, "properties": 0})

    def test_private_output_is_restricted_and_public_commitment_has_no_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest, _, _ = self._build(root)
            private = root / "private" / "allocation.json"
            public = root / "public.json"
            public_manifest = write_allocation_manifests(
                manifest, private_output=private, public_output=public
            )
            self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)
            public_text = public.read_text(encoding="utf-8")
            self.assertNotIn("new_a", public_text)
            self.assertNotIn("selected_case_ids", public_text)
            self.assertEqual(public_manifest["counts"]["selected"], 2)

    def test_independent_validator_detects_annotation_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest, classified, exclusion = self._build(root)
            manifest["case_annotations"]["new_a"]["class"] = "TypeA"
            validation = validate_untouched_test_manifest(
                manifest,
                classified_path=classified,
                exclude_manifests=[exclusion],
            )
            self.assertFalse(validation["checks"]["annotations_recomputed"])
            self.assertFalse(validation["passed"])

    def test_allocation_requires_targets_matching_protocol_population(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified = root / "classified.jsonl"
            classified.write_text(json.dumps(_record("new", "Q2", "P2")) + "\n", encoding="utf-8")
            exclusion = root / "exclude.json"
            exclusion.write_text(json.dumps({"selected_case_ids": []}), encoding="utf-8")
            protocol = root / "protocol.json"
            protocol.write_text("{}", encoding="utf-8")
            verification = {
                "passed": True,
                "manifest": {
                    "protocol_id": "protocol_v1",
                    "protocol_phase": "allocation",
                    "release": {"release_kind": "dataset"},
                    "expected_population": {"selected_count": 2, "main_score_count": 2},
                },
            }
            with patch("select_untouched_test.verify_protocol_manifest", return_value=verification):
                with self.assertRaisesRegex(ValueError, "sum to the protocol"):
                    build_untouched_test_manifest(
                        classified_path=classified,
                        exclude_manifests=[exclusion],
                        stratum_targets={"TypeB_LOCAL_TEXT_CONFIRMED": 1},
                        protocol_manifest_path=protocol,
                        protocol_root=root,
                    )

    def test_allocation_requires_main_score_count_matching_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified = root / "classified.jsonl"
            classified.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in [_record("new_a", "Q2", "P2"), _record("new_b", "Q2", "P2")]
                ),
                encoding="utf-8",
            )
            exclusion = root / "exclude.json"
            exclusion.write_text(json.dumps({"selected_case_ids": []}), encoding="utf-8")
            protocol = root / "protocol.json"
            protocol.write_text("{}", encoding="utf-8")
            verification = {
                "passed": True,
                "manifest": {
                    "protocol_id": "protocol_v1",
                    "protocol_phase": "allocation",
                    "release": {"release_kind": "dataset"},
                    "expected_population": {"selected_count": 2, "main_score_count": 1},
                },
            }
            with patch("select_untouched_test.verify_protocol_manifest", return_value=verification):
                with self.assertRaisesRegex(ValueError, "main-score count"):
                    build_untouched_test_manifest(
                        classified_path=classified,
                        exclude_manifests=[exclusion],
                        stratum_targets={"TypeB_LOCAL_TEXT_CONFIRMED": 2},
                        protocol_manifest_path=protocol,
                        protocol_root=root,
                    )

    def test_confirmatory_reserve_finalization_and_nested_api_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified = root / "classified.jsonl"
            rows = []
            specifications = [
                ("TypeA", "SELF_LINK_REJECTION", 276),
                ("TypeB", "LOCAL_TEXT_CONFIRMED", 450),
                ("TypeC", "EXTERNAL_BY_ELIMINATION", 354),
                ("T_BOX", "RELAXATION_SET_EXPANSION", 156),
                ("T_BOX", "RESTRICTION_SET_CONTRACTION", 60),
                ("T_BOX", "SCHEMA_UPDATE", 144),
            ]
            index = 0
            for cls, subtype, count in specifications:
                for _ in range(count):
                    index += 1
                    if cls == "T_BOX":
                        row = {
                            "id": f"case_{index}", "qid": f"Q{index}", "property": f"P{index}",
                            "track": "T_BOX", "repair_target": {"kind": "T_BOX", "property_revision_id": index},
                            "classification": {"class": cls, "subtype": subtype, "confidence": "high"},
                        }
                    else:
                        row = {
                            "id": f"case_{index}", "qid": f"Q{index}", "property": f"P{index}",
                            "track": "A_BOX", "repair_target": {"kind": "A_BOX"},
                            "classification": {"class": cls, "subtype": subtype, "confidence": "high"},
                        }
                    rows.append(row)
            classified.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            dispositions = root / "dispositions.jsonl"
            dispositions.write_text(
                "".join(json.dumps({"case_id": row["id"], "disposition": "include"}) + "\n" for row in rows),
                encoding="utf-8",
            )
            dataset_audit = root / "dataset_audit.json"
            dataset_audit.write_text(json.dumps({
                "report_type": "automated_consistency_audit", "report_version": 2,
                "lineage_validation": {"passed": True},
            }), encoding="utf-8")
            temporal = root / "temporal.json"
            temporal.write_text(json.dumps({
                "report_type": "temporal_prompt_leakage_audit", "report_version": 2,
                "passed_automated_gate": True, "hits": [],
            }), encoding="utf-8")
            snapshot = root / "snapshot.json"
            snapshot.write_text(json.dumps({"manifest_version": 2, "snapshot_id": "post-freeze-test"}), encoding="utf-8")
            reserve = build_reserve_manifest(
                classified_path=classified, dispositions_path=dispositions, dataset_audit_path=dataset_audit,
                temporal_audit_path=temporal, exclude_manifests=[], snapshot_manifest_path=snapshot,
            )
            self.assertEqual(reserve["counts"]["selected"], 1440)
            reserve_path = root / "reserve.json"
            reserve_path.write_text(json.dumps(reserve, sort_keys=True), encoding="utf-8")
            final = finalize_reserve_manifest(reserve_manifest_path=reserve_path, temporal_audit_path=temporal)
            self.assertEqual(final["counts"]["selected"], 1200)
            self.assertEqual(final["counts"]["api_subset"], 600)
            self.assertEqual(final["counts"]["by_stratum"], {"TBOX": 300, "TypeA": 230, "TypeB": 375, "TypeC": 295})
            self.assertTrue(set(final["api_subset_case_ids"]).issubset(final["selected_case_ids"]))
            self.assertEqual(
                final,
                finalize_reserve_manifest(reserve_manifest_path=reserve_path, temporal_audit_path=temporal),
            )


if __name__ == "__main__":
    unittest.main()
