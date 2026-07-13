import json
import tempfile
import unittest
from pathlib import Path

from select_untouched_test import build_untouched_test_manifest


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
    def test_excludes_prior_properties_and_selects_complete_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified = root / "classified.jsonl"
            rows = [
                _record("old", "Q1", "P1"),
                _record("new_a", "Q2", "P2"),
                _record("new_b", "Q2", "P2"),
            ]
            classified.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            exclusion = root / "exclude.json"
            exclusion.write_text(
                json.dumps(
                    {
                        "selected_case_ids": ["old"],
                        "case_annotations": {"old": {"group_key": "ABOX::Q1::P1", "property": "P1"}},
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_untouched_test_manifest(
                classified_path=classified,
                exclude_manifests=[exclusion],
                target_size=2,
                property_holdout=True,
            )

            self.assertEqual(set(manifest["selected_case_ids"]), {"new_a", "new_b"})
            self.assertEqual(manifest["validation"]["excluded_case_overlap"], 0)
            self.assertEqual(manifest["validation"]["property_holdout_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
