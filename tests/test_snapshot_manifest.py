import json
import tempfile
import unittest
from pathlib import Path

from snapshot_manifest import build_snapshot_manifest, verify_snapshot_manifest


class SnapshotManifestTests(unittest.TestCase):
    def test_build_verify_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stage2 = root / "stage2.json"
            world = root / "world.json"
            stage4 = root / "stage4.jsonl"
            stage2.write_text("[]\n", encoding="utf-8")
            world.write_text("{}\n", encoding="utf-8")
            stage4.write_text("", encoding="utf-8")
            manifest = build_snapshot_manifest(
                snapshot_id="synthetic-v1",
                stage2_path=stage2,
                world_state_path=world,
                stage4_path=stage4,
                sources=[
                    {
                        "name": "synthetic",
                        "source_id": "fixture-v1",
                        "retrieved_at_utc": "2026-01-01T00:00:00Z",
                        "immutable": True,
                    }
                ],
                limitations=["Synthetic test data only."],
            )
            manifest_path = root / "snapshot.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            verified = verify_snapshot_manifest(
                manifest_path,
                stage2_path=stage2,
                world_state_path=world,
                stage4_path=stage4,
            )
            self.assertTrue(verified["passed"])

            stage4.write_text("tampered\n", encoding="utf-8")
            tampered = verify_snapshot_manifest(
                manifest_path,
                stage2_path=stage2,
                world_state_path=world,
                stage4_path=stage4,
            )
            self.assertFalse(tampered["checks"]["artifact_hashes_match"])


if __name__ == "__main__":
    unittest.main()
