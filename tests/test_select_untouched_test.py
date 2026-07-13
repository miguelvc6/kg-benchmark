import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from select_untouched_test import (
    build_untouched_test_manifest,
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


if __name__ == "__main__":
    unittest.main()
