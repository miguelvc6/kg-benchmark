import json
import tempfile
import unittest
from pathlib import Path

from kg_benchmark.dataset.release import CANONICAL_FILES, fetch_dataset, promote_dataset, sha256_file, verify_dataset


class SingleDatasetReleaseTests(unittest.TestCase):
    def _write_work(self, root: Path) -> tuple[Path, Path, Path]:
        work = root / "work"
        for role, relative in CANONICAL_FILES.items():
            path = work / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".jsonl":
                path.write_text(json.dumps({"id": role}) + "\n", encoding="utf-8")
            else:
                path.write_text(json.dumps({"role": role}) + "\n", encoding="utf-8")
        (work / "cache").mkdir()
        (work / "cache" / "derived.sqlite").write_bytes(b"not canonical")
        protocol = root / "protocol.json"
        protocol.write_text(json.dumps({"protocol": "test"}) + "\n", encoding="utf-8")
        provenance = root / "source-provenance.json"
        provenance.write_text(json.dumps({"source": "synthetic"}) + "\n", encoding="utf-8")
        return work, protocol, provenance

    def test_promotion_is_atomic_and_copies_only_canonical_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, protocol, provenance = self._write_work(root)
            dataset = root / "dataset"
            manifest = promote_dataset(
                work_dir=work,
                dataset_dir=dataset,
                protocol_path=protocol,
                source_provenance_path=provenance,
            )
            self.assertEqual(manifest["status"], "final")
            self.assertTrue(verify_dataset(dataset)["valid"])
            self.assertFalse((dataset / "cache").exists())
            with self.assertRaises(FileExistsError):
                promote_dataset(
                    work_dir=work,
                    dataset_dir=dataset,
                    protocol_path=protocol,
                    source_provenance_path=provenance,
                )

    def test_promotion_rejects_missing_artifact_without_partial_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, protocol, provenance = self._write_work(root)
            (work / CANONICAL_FILES["cases"]).unlink()
            dataset = root / "dataset"
            with self.assertRaisesRegex(ValueError, "missing or empty"):
                promote_dataset(
                    work_dir=work,
                    dataset_dir=dataset,
                    protocol_path=protocol,
                    source_provenance_path=provenance,
                )
            self.assertFalse(dataset.exists())

    def test_promotion_requires_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, protocol, _ = self._write_work(root)
            with self.assertRaisesRegex(ValueError, "source-provenance"):
                promote_dataset(work_dir=work, dataset_dir=root / "dataset", protocol_path=protocol)

    def test_fetch_recreates_and_verifies_the_canonical_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, protocol, provenance = self._write_work(root)
            published = root / "published"
            promote_dataset(
                work_dir=work,
                dataset_dir=published,
                protocol_path=protocol,
                source_provenance_path=provenance,
            )
            fetched = root / "fetched"
            result = fetch_dataset(
                manifest_url=(published / "manifest.json").as_uri(),
                manifest_sha256=sha256_file(published / "manifest.json"),
                dataset_dir=fetched,
            )
            self.assertTrue(result["valid"])
            self.assertEqual((published / "manifest.json").read_bytes(), (fetched / "manifest.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
