import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from artifact_acquisition import ManifestError, fetch_artifacts, load_manifest, validate_manifest, verify_artifacts


class ArtifactAcquisitionTest(unittest.TestCase):
    def _manifest(self, source: Path, *, digest: str | None = None) -> dict:
        content = source.read_bytes()
        return {
            "manifest_type": "kg_benchmark_distribution",
            "manifest_version": 1,
            "status": "draft",
            "release": {
                "benchmark_version": "test",
                "repository_url": "https://example.org/repository",
                "source_commit": None,
                "release_manifest_url": None,
                "release_manifest_sha256": None,
            },
            "source_snapshots": [
                {
                    "source_id": "fixture",
                    "description": "test fixture",
                    "status": "resolved",
                    "snapshot_at_utc": "2026-01-01T00:00:00Z",
                    "source_url": source.as_uri(),
                    "upstream_license_id": "CC0-1.0",
                    "notes": "",
                }
            ],
            "artifacts": [
                {
                    "artifact_id": "fixture",
                    "role": "test fixture",
                    "path": "data/fixture.bin",
                    "required": True,
                    "status": "published",
                    "urls": [source.as_uri()],
                    "sha256": digest or hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                    "media_type": "application/octet-stream",
                    "license_id": "CC0-1.0",
                    "source_snapshot_ids": ["fixture"],
                    "notes": "",
                }
            ],
        }

    def test_repository_template_validates_against_schema_and_runtime(self) -> None:
        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "schemas/artifact_distribution.schema.json").read_text())
        manifest_path = root / "release/artifact_distribution.template.json"
        manifest = json.loads(manifest_path.read_text())
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER).validate(manifest)
        self.assertEqual(load_manifest(manifest_path), manifest)

    def test_unresolved_required_artifact_fails_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = load_manifest(root / "release/artifact_distribution.template.json")
        results = verify_artifacts(manifest, root=root)
        self.assertTrue(results)
        self.assertTrue(all(result.status == "error" for result in results))
        allowed = verify_artifacts(manifest, root=root, allow_unresolved=True)
        self.assertTrue(all(result.status == "skipped" for result in allowed))

    def test_fetches_and_verifies_before_atomic_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.bin"
            source.write_bytes(b"immutable fixture\n")
            manifest = self._manifest(source)
            results = fetch_artifacts(manifest, root=root)
            self.assertEqual(results[0].status, "fetched")
            self.assertEqual((root / "data/fixture.bin").read_bytes(), source.read_bytes())
            self.assertEqual(verify_artifacts(manifest, root=root)[0].status, "verified")

    def test_checksum_failure_leaves_no_target_or_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.bin"
            source.write_bytes(b"fixture")
            manifest = self._manifest(source, digest="0" * 64)
            results = fetch_artifacts(manifest, root=root)
            self.assertEqual(results[0].status, "error")
            self.assertFalse((root / "data/fixture.bin").exists())
            self.assertEqual(list((root / "data").glob("*.part")), [])

    def test_rejects_unsafe_and_duplicate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.bin"
            source.write_bytes(b"fixture")
            manifest = self._manifest(source)
            manifest["artifacts"][0]["path"] = "../outside.bin"
            with self.assertRaises(ManifestError):
                validate_manifest(manifest)

            manifest = self._manifest(source)
            duplicate = dict(manifest["artifacts"][0])
            duplicate["artifact_id"] = "duplicate"
            manifest["artifacts"].append(duplicate)
            with self.assertRaises(ManifestError):
                validate_manifest(manifest)

    def test_rejects_unknown_source_reference_and_unlicensed_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.bin"
            source.write_bytes(b"fixture")
            manifest = self._manifest(source)
            manifest["artifacts"][0]["source_snapshot_ids"] = ["missing"]
            with self.assertRaises(ManifestError):
                validate_manifest(manifest)

            manifest = self._manifest(source)
            manifest["artifacts"][0]["license_id"] = None
            with self.assertRaises(ManifestError):
                validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
