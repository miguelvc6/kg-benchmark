import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kg_benchmark.dataset.gates import SCHEMA_ARTIFACTS, WORK_ARTIFACTS
from kg_benchmark.dataset.release import (
    canonicalize_case_context_references,
    fetch_dataset,
    promote_dataset,
    sha256_file,
    verify_dataset,
)
from kg_benchmark.methodology import MethodologyError

ROOT = Path(__file__).resolve().parents[1]


class SingleDatasetReleaseTests(unittest.TestCase):
    def test_classified_context_references_are_release_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cases = Path(temporary) / "cases.jsonl"
            rows = [
                {
                    "id": "case-1",
                    "context_ref": {
                        "world_state_id": "case-1",
                        "world_state_path": "/construction-machine/work/acquisition/03_world_state.json",
                    },
                },
                {
                    "id": "case-2",
                    "context_ref": {
                        "world_state_id": "case-2",
                        "world_state_path": "C:\\construction\\03_world_state.json",
                    },
                },
            ]
            cases.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            self.assertEqual(canonicalize_case_context_references(cases), 2)
            rewritten = [json.loads(line) for line in cases.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                {row["context_ref"]["world_state_path"] for row in rewritten},
                {"source/world-state.jsonl"},
            )

    def _write_work(self, root: Path) -> tuple[Path, Path, Path, dict]:
        work = root / "work"
        for role, relative in WORK_ARTIFACTS.items():
            path = work / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".jsonl":
                path.write_text(json.dumps({"id": role}) + "\n", encoding="utf-8")
            else:
                path.write_text(json.dumps({"role": role}) + "\n", encoding="utf-8")
        (work / "cache").mkdir()
        (work / "cache" / "derived.sqlite").write_bytes(b"not canonical")

        paper = root / "paper"
        paper.mkdir()
        protocol = paper / "protocol.json"
        protocol.write_text(
            json.dumps({"protocol_id": "synthetic-protocol", "status": "frozen"}) + "\n",
            encoding="utf-8",
        )
        lock = {
            "manifest_type": "paper_methodology_lock",
            "manifest_version": 1,
            "protocol_id": "synthetic-protocol",
            "source_git_revision": "b" * 40,
            "freeze_scope_sha256": "a" * 64,
            "files": {"paper/protocol.json": sha256_file(protocol)},
        }
        (paper / "methodology.lock.json").write_text(json.dumps(lock) + "\n", encoding="utf-8")
        schemas = root / "schemas"
        schemas.mkdir()
        for filename in SCHEMA_ARTIFACTS.values():
            shutil.copyfile(ROOT / "schemas" / filename, schemas / filename)
        methodology = {
            "files": {"paper/protocol.json": sha256_file(protocol)},
            "freeze_scope_sha256": "a" * 64,
            "lock": {
                "path": "paper/methodology.lock.json",
                "source_git_revision": "b" * 40,
            },
        }
        return work, protocol, work / "source-provenance.json", methodology

    @staticmethod
    def _semantics() -> dict:
        return {"valid": True, "records": {}, "selection": {}, "external_sources_checked": False}

    def test_promotion_is_atomic_and_copies_only_canonical_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, protocol, provenance, methodology = self._write_work(root)
            dataset = root / "dataset"
            with patch("kg_benchmark.dataset.release.validate_dataset_semantics", return_value=self._semantics()):
                manifest = promote_dataset(
                    work_dir=work,
                    dataset_dir=dataset,
                    protocol_path=protocol,
                    source_provenance_path=provenance,
                    repo_root=root,
                    methodology_check=lambda _: methodology,
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
                    repo_root=root,
                    methodology_check=lambda _: methodology,
                )

    def test_promotion_rejects_missing_artifact_without_partial_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, protocol, provenance, methodology = self._write_work(root)
            (work / WORK_ARTIFACTS["cases"]).unlink()
            dataset = root / "dataset"
            with self.assertRaisesRegex(ValueError, "missing or empty"):
                promote_dataset(
                    work_dir=work,
                    dataset_dir=dataset,
                    protocol_path=protocol,
                    source_provenance_path=provenance,
                    repo_root=root,
                    methodology_check=lambda _: methodology,
                )
            self.assertFalse(dataset.exists())

    def test_promotion_requires_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, protocol, _, methodology = self._write_work(root)
            with self.assertRaisesRegex(ValueError, "source-provenance"):
                promote_dataset(
                    work_dir=work,
                    dataset_dir=root / "dataset",
                    protocol_path=protocol,
                    repo_root=root,
                    methodology_check=lambda _: methodology,
                )

    def test_promotion_requires_a_valid_frozen_methodology(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, protocol, provenance, _ = self._write_work(root)

            def reject(_: Path) -> dict:
                raise MethodologyError("synthetic methodology lock failure")

            with self.assertRaisesRegex(MethodologyError, "lock failure"):
                promote_dataset(
                    work_dir=work,
                    dataset_dir=root / "dataset",
                    protocol_path=protocol,
                    source_provenance_path=provenance,
                    repo_root=root,
                    methodology_check=reject,
                )
            self.assertFalse((root / "dataset").exists())

    def test_promotion_rejects_a_protocol_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, protocol, provenance, methodology = self._write_work(root)
            protocol.write_text(json.dumps({"protocol_id": "changed", "status": "frozen"}) + "\n")
            with self.assertRaisesRegex(ValueError, "does not bind the promoted protocol hash"):
                promote_dataset(
                    work_dir=work,
                    dataset_dir=root / "dataset",
                    protocol_path=protocol,
                    source_provenance_path=provenance,
                    repo_root=root,
                    methodology_check=lambda _: methodology,
                )
            self.assertFalse((root / "dataset").exists())

    def test_fetch_recreates_and_verifies_the_canonical_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, protocol, provenance, methodology = self._write_work(root)
            published = root / "published"
            with patch("kg_benchmark.dataset.release.validate_dataset_semantics", return_value=self._semantics()):
                promote_dataset(
                    work_dir=work,
                    dataset_dir=published,
                    protocol_path=protocol,
                    source_provenance_path=provenance,
                    repo_root=root,
                    methodology_check=lambda _: methodology,
                )
                fetched = root / "fetched"
                result = fetch_dataset(
                    manifest_url=(published / "manifest.json").as_uri(),
                    manifest_sha256=sha256_file(published / "manifest.json"),
                    dataset_dir=fetched,
                )
            self.assertTrue(result["valid"])
            self.assertEqual((published / "manifest.json").read_bytes(), (fetched / "manifest.json").read_bytes())
            self.assertEqual(
                hashlib.sha256((fetched / "manifest.json").read_bytes()).hexdigest(),
                result["manifest_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
