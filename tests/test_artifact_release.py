import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from artifact_release import build_release_manifest, sha256_file, validate_release_inputs, verify_release_manifest


class ArtifactReleaseTests(unittest.TestCase):
    def _record(self, case_id: str = "repair_case") -> dict:
        return {
            "id": case_id,
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
            "context_ref": {"world_state_id": case_id, "world_state_path": "world.json"},
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

    def _world_entry(self) -> dict:
        return {
            "L1_ego_node": {"qid": "Q1", "label": "entity", "description": None, "properties": {}},
            "L2_labels": {},
            "L3_neighborhood": {"outgoing_edges": []},
            "L4_constraints": {},
        }

    def _fixture(self, root: Path) -> dict[str, Path]:
        stage2 = root / "stage2.json"
        stage2.write_text(json.dumps([self._record()]), encoding="utf-8")
        world = root / "world.json"
        world.write_text(json.dumps({"repair_case": self._world_entry()}), encoding="utf-8")
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
        schema = root / "stage4.schema.json"
        source_schema = Path(__file__).resolve().parents[1] / "schemas" / "04_classified_benchmark.schema.json"
        schema.write_bytes(source_schema.read_bytes())
        snapshot = root / "snapshot.json"
        snapshot.write_text(
            json.dumps(
                {
                    "manifest_type": "kg_benchmark_snapshot",
                    "manifest_version": 1,
                    "snapshot_id": "synthetic-2026-01-01",
                    "created_at_utc": "2026-01-01T00:00:00Z",
                    "context_policy": "later_frozen_context_with_historical_target_reconstruction",
                    "repair_time_complete": False,
                    "sources": [
                        {
                            "name": "synthetic",
                            "source_id": "fixture-v1",
                            "retrieved_at_utc": "2026-01-01T00:00:00Z",
                            "immutable": True,
                        }
                    ],
                    "artifacts": {
                        "stage2_repairs_sha256": sha256_file(stage2),
                        "world_state_sha256": sha256_file(world),
                        "classified_benchmark_sha256": sha256_file(stage4),
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "stage2": stage2,
            "world": world,
            "stage4": stage4,
            "selection": selection,
            "schema": schema,
            "snapshot": snapshot,
        }

    def test_build_and_verify_evaluation_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = self._fixture(root)
            manifest = build_release_manifest(
                release_root=root,
                stage2_path=paths["stage2"],
                world_state_path=paths["world"],
                stage4_path=paths["stage4"],
                schema_path=paths["schema"],
                snapshot_manifest_path=paths["snapshot"],
                selection_manifest_path=paths["selection"],
                release_kind="evaluation",
            )
            self.assertTrue(manifest["validation"]["passed"])
            self.assertTrue(all(not Path(item["path"]).is_absolute() for item in manifest["files"]))
            self.assertEqual(manifest["snapshot_id"], "synthetic-2026-01-01")
            manifest_path = root / "release.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            verification = verify_release_manifest(manifest_path, release_root=root)
            self.assertTrue(verification["passed"])

    def test_unselected_invalid_stage4_record_fails_full_release_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = self._fixture(root)
            with paths["stage4"].open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "invalid_unselected"}) + "\n")
            validation = validate_release_inputs(
                stage2_path=paths["stage2"],
                world_state_path=paths["world"],
                stage4_path=paths["stage4"],
                schema_path=paths["schema"],
                snapshot_manifest_path=paths["snapshot"],
                selection_manifest_path=paths["selection"],
                release_kind="evaluation",
            )
            self.assertFalse(validation["checks"]["full_stage4_schema_valid"])
            self.assertFalse(validation["checks"]["stage2_stage4_ids_match"])
            self.assertFalse(validation["passed"])

    def test_verifier_detects_file_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = self._fixture(root)
            manifest = build_release_manifest(
                release_root=root,
                stage2_path=paths["stage2"],
                world_state_path=paths["world"],
                stage4_path=paths["stage4"],
                schema_path=paths["schema"],
                snapshot_manifest_path=paths["snapshot"],
                selection_manifest_path=paths["selection"],
                release_kind="evaluation",
            )
            manifest_path = root / "release.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            paths["world"].write_text("{}", encoding="utf-8")
            verification = verify_release_manifest(manifest_path, release_root=root)
            self.assertFalse(verification["checks"]["all_files_match"])
            self.assertFalse(verification["passed"])

    def test_confirmatory_release_requires_clean_git(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = self._fixture(root)
            with patch("artifact_release._git_state", return_value={"commit": "abc", "dirty": True}):
                with self.assertRaisesRegex(ValueError, "clean Git commit"):
                    build_release_manifest(
                        release_root=root,
                        stage2_path=paths["stage2"],
                        world_state_path=paths["world"],
                        stage4_path=paths["stage4"],
                        schema_path=paths["schema"],
                        snapshot_manifest_path=paths["snapshot"],
                        selection_manifest_path=paths["selection"],
                        release_kind="evaluation",
                        release_status="confirmatory",
                    )

    def test_snapshot_manifest_must_bind_all_released_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = self._fixture(root)
            snapshot = json.loads(paths["snapshot"].read_text(encoding="utf-8"))
            snapshot["artifacts"]["world_state_sha256"] = "0" * 64
            paths["snapshot"].write_text(json.dumps(snapshot), encoding="utf-8")

            validation = validate_release_inputs(
                stage2_path=paths["stage2"],
                world_state_path=paths["world"],
                stage4_path=paths["stage4"],
                schema_path=paths["schema"],
                snapshot_manifest_path=paths["snapshot"],
                selection_manifest_path=paths["selection"],
                release_kind="evaluation",
            )

            self.assertTrue(validation["checks"]["snapshot_manifest_valid"])
            self.assertFalse(validation["checks"]["snapshot_artifact_hashes_match"])
            self.assertFalse(validation["passed"])

    def test_confirmatory_verifier_rejects_nonexistent_git_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = self._fixture(root)
            manifest = build_release_manifest(
                release_root=root,
                stage2_path=paths["stage2"],
                world_state_path=paths["world"],
                stage4_path=paths["stage4"],
                schema_path=paths["schema"],
                snapshot_manifest_path=paths["snapshot"],
                release_kind="evaluation",
                selection_manifest_path=paths["selection"],
            )
            manifest["status"] = "confirmatory"
            manifest["code"] = {"commit": "not-a-real-commit", "dirty": False}
            manifest_path = root / "release.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with patch("artifact_release.git_commit_exists", return_value=False):
                verification = verify_release_manifest(manifest_path, release_root=root)

            self.assertFalse(verification["checks"]["confirmatory_code_verified"])
            self.assertFalse(verification["passed"])


if __name__ == "__main__":
    unittest.main()
