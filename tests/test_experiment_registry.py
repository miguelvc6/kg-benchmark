import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from artifact_release import sha256_file
from experiment_registry import REQUIRED_FINGERPRINTS, register_experiment


class ExperimentRegistryTests(unittest.TestCase):
    def _confirmatory_fixture(self, root: Path) -> tuple[Path, Path, dict]:
        artifacts = {}
        for name in REQUIRED_FINGERPRINTS:
            path = root / f"{name}.json"
            path.write_text(json.dumps({"name": name}), encoding="utf-8")
            artifacts[name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        release = root / "release.json"
        release.write_text(
            json.dumps(
                {
                    "files": [
                        {"role": name, "sha256": artifacts[name]["sha256"]}
                        for name in ("classified_benchmark", "world_state", "selection_manifest")
                    ]
                }
            ),
            encoding="utf-8",
        )
        protocol = root / "protocol.json"
        protocol.write_text("{}", encoding="utf-8")
        protocol_manifest = {
            "protocol_id": "protocol_v1",
            "protocol_phase": "execution",
            "status": "frozen",
            "code": {"commit": "abc", "dirty": False},
            "release": {"path": "release.json", "release_kind": "evaluation"},
            "models": [{"name": "model", "digest": "digest-123"}],
            "conditions": ["logic_only", "local_graph"],
            "expected_population": {"selected_count": 2, "main_score_count": 2},
            "files": [
                {"sha256": value["sha256"]} for value in artifacts.values()
            ],
        }
        verification = {
            "passed": True,
            "manifest": protocol_manifest,
            "release_verification": {"validation": {"counts": {"selected": 2}}},
        }
        summary = root / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "run_info": {
                        "run_id": "run_a",
                        "model_digest": "digest-123",
                        "code": {"commit": "abc", "dirty": False},
                        "artifact_fingerprints": artifacts,
                    },
                    "paper_subsets": {"all_selected": {"count": 2}, "main_score": {"count": 2}},
                    "by_ablation_bundle": {"logic_only": {}, "local_graph": {}},
                }
            ),
            encoding="utf-8",
        )
        return summary, protocol, verification

    def test_exploratory_run_is_never_paper_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps({"run_info": {"run_id": "run_a"}}), encoding="utf-8")
            entry = register_experiment(
                registry_path=root / "registry.json",
                run_summary_path=summary_path,
                status="exploratory",
                schema_path=Path(__file__).resolve().parents[1] / "schemas" / "experiment_registry.schema.json",
            )
            self.assertFalse(entry["paper_eligible"])
            self.assertIn("status_is_not_confirmatory", entry["ineligibility_reasons"])

    def test_verified_confirmatory_run_is_paper_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary, protocol, verification = self._confirmatory_fixture(root)
            with patch("experiment_registry.verify_protocol_manifest", return_value=verification):
                entry = register_experiment(
                    registry_path=root / "registry.json",
                    run_summary_path=summary,
                    status="confirmatory",
                    protocol_manifest_path=protocol,
                    protocol_root=root,
                    schema_path=Path(__file__).resolve().parents[1]
                    / "schemas"
                    / "experiment_registry.schema.json",
                )
            self.assertTrue(entry["paper_eligible"])
            self.assertEqual(entry["ineligibility_reasons"], [])

    def test_fabricated_fingerprint_cannot_be_registered_confirmatory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary, protocol, verification = self._confirmatory_fixture(root)
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["run_info"]["artifact_fingerprints"]["classified_benchmark"]["sha256"] = "0" * 64
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with patch("experiment_registry.verify_protocol_manifest", return_value=verification):
                with self.assertRaisesRegex(ValueError, "classified_benchmark_fingerprint_unverified"):
                    register_experiment(
                        registry_path=root / "registry.json",
                        run_summary_path=summary,
                        status="confirmatory",
                        protocol_manifest_path=protocol,
                        protocol_root=root,
                        schema_path=Path(__file__).resolve().parents[1]
                        / "schemas"
                        / "experiment_registry.schema.json",
                    )

    def test_nonexistent_protocol_is_rejected_for_confirmatory_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = root / "summary.json"
            summary.write_text(json.dumps({"run_info": {"run_id": "fake"}}), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                register_experiment(
                    registry_path=root / "registry.json",
                    run_summary_path=summary,
                    status="confirmatory",
                    protocol_manifest_path=root / "missing.json",
                    protocol_root=root,
                )


if __name__ == "__main__":
    unittest.main()
