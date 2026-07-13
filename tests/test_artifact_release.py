import json
import tempfile
import unittest
from pathlib import Path

from artifact_release import build_release_manifest, validate_release_inputs


class ArtifactReleaseTests(unittest.TestCase):
    def _record(self) -> dict:
        return {
            "id": "repair_case",
            "qid": "Q1",
            "property": "P1",
            "track": "A_BOX",
            "information_type": "TBD",
            "labels_en": {
                "qid": {"label": "entity", "description": None},
                "property": {"label": "property", "description": None},
            },
            "violation_context": {
                "report_page_title": "report",
                "report_fix_date": "2026-01-01",
                "report_revision_old": 1,
                "report_revision_new": 2,
                "report_violation_type_raw": "type",
                "report_violation_type_normalized": "type",
                "report_violation_type_qids": [],
                "value": "Q2",
            },
            "repair_target": {
                "kind": "A_BOX",
                "author": "editor",
                "action": "UPDATE",
                "old_value": ["Q2"],
                "new_value": ["Q3"],
            },
            "persistence_check": {"status": "passed", "current_value_2026": ["Q3"]},
            "popularity": {
                "score": 0.5,
                "components": {
                    "pageviews_365d": 1,
                    "out_degree": 1,
                    "sitelinks_count": 1,
                    "pageviews_norm": 0.5,
                    "degree_norm": 0.5,
                    "sitelinks_norm": 0.5,
                },
            },
            "context_ref": {"world_state_id": "repair_case", "world_state_path": "world.json"},
            "classification": {
                "class": "TypeB",
                "subtype": "LOCAL_TEXT_CONFIRMED",
                "confidence": "high",
                "decision_trace": [{"step": "classify", "result": True}],
                "rationale": "Local evidence.",
                "constraint_types": [],
            },
            "build": {"version": "test"},
        }

    def test_build_release_manifest_validates_and_hashes_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stage4 = root / "stage4.jsonl"
            stage4.write_text(json.dumps(self._record()) + "\n", encoding="utf-8")
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "selected_case_ids": ["repair_case"],
                        "main_score_case_ids": ["repair_case"],
                        "diagnostic_case_ids": [],
                        "policy": {
                            "tbox_cap_per_property_revision": 1,
                            "abox_cap_per_qid_property": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            schema = Path(__file__).resolve().parents[1] / "schemas" / "04_classified_benchmark.schema.json"

            manifest = build_release_manifest(
                stage4_path=stage4,
                schema_path=schema,
                selection_manifest_path=selection,
            )

            self.assertTrue(manifest["validation"]["passed"])
            self.assertTrue(all(len(file["sha256"]) == 64 for file in manifest["files"]))

    def test_release_validation_requires_main_diagnostic_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stage4 = root / "stage4.jsonl"
            stage4.write_text(json.dumps(self._record()) + "\n", encoding="utf-8")
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "selected_case_ids": ["repair_case"],
                        "policy": {
                            "tbox_cap_per_property_revision": 1,
                            "abox_cap_per_qid_property": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            schema = Path(__file__).resolve().parents[1] / "schemas" / "04_classified_benchmark.schema.json"

            validation = validate_release_inputs(
                stage4_path=stage4,
                schema_path=schema,
                selection_manifest_path=selection,
            )

            self.assertFalse(validation["checks"]["subset_partition_present"])
            self.assertFalse(validation["passed"])


if __name__ == "__main__":
    unittest.main()
