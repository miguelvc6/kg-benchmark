import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from protocol_freeze import build_protocol_manifest, verify_protocol_manifest


class ProtocolFreezeTests(unittest.TestCase):
    def test_methodology_freeze_precedes_and_does_not_reference_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prompt = root / "prompt.py"
            prompt.write_text("PROMPT = 'frozen'\n", encoding="utf-8")
            schema = root / "schema.json"
            schema.write_text("{}", encoding="utf-8")
            analysis = root / "analysis.md"
            analysis.write_text("Primary effect.\n", encoding="utf-8")
            with patch("protocol_freeze._git_state", return_value={"commit": "abc", "dirty": False}):
                manifest = build_protocol_manifest(
                    protocol_root=root,
                    protocol_id="methodology_v1",
                    protocol_phase="methodology",
                    release_manifest_path=None,
                    models=[{"name": "model", "digest": None}],
                    conditions=["logic_only", "local_graph"],
                    prompt_files=[prompt],
                    schema_files=[schema],
                    analysis_plan_path=analysis,
                    expected_selected_count=1200,
                    expected_main_score_count=1200,
                    status="frozen",
                )
            self.assertIsNone(manifest["release"])
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch("protocol_freeze.git_commit_exists", return_value=True):
                result = verify_protocol_manifest(protocol_path, protocol_root=root)
            self.assertTrue(result["passed"])

    def test_frozen_protocol_binds_release_model_conditions_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            release = root / "release.json"
            release.write_text(
                json.dumps(
                    {
                        "release_kind": "evaluation",
                        "status": "confirmatory",
                        "code": {"commit": "abc", "dirty": False},
                    }
                ),
                encoding="utf-8",
            )
            prompt = root / "prompt.py"
            prompt.write_text("PROMPT = 'frozen'\n", encoding="utf-8")
            schema = root / "proposal.schema.json"
            schema.write_text("{}", encoding="utf-8")
            analysis = root / "analysis.md"
            analysis.write_text("Primary effect: local - logic.\n", encoding="utf-8")
            with (
                patch("protocol_freeze.verify_release_manifest", return_value={"passed": True}),
                patch("protocol_freeze._git_state", return_value={"commit": "abc", "dirty": False}),
            ):
                manifest = build_protocol_manifest(
                    protocol_root=root,
                    protocol_id="protocol_v1",
                    protocol_phase="execution",
                    release_manifest_path=release,
                    models=[{"name": "model", "digest": "digest-123"}],
                    conditions=["logic_only", "local_graph"],
                    prompt_files=[prompt],
                    schema_files=[schema],
                    analysis_plan_path=analysis,
                    expected_selected_count=10,
                    expected_main_score_count=8,
                    status="frozen",
                )
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(manifest), encoding="utf-8")
            with (
                patch("protocol_freeze.verify_release_manifest", return_value={"passed": True}),
                patch("protocol_freeze.git_commit_exists", return_value=True),
            ):
                result = verify_protocol_manifest(protocol_path, protocol_root=root)
            self.assertTrue(result["passed"])
            prompt.write_text("PROMPT = 'changed'\n", encoding="utf-8")
            with (
                patch("protocol_freeze.verify_release_manifest", return_value={"passed": True}),
                patch("protocol_freeze.git_commit_exists", return_value=True),
            ):
                tampered = verify_protocol_manifest(protocol_path, protocol_root=root)
            self.assertFalse(tampered["checks"]["all_protocol_files_match"])
            self.assertFalse(tampered["passed"])

    def test_protocol_phase_requires_matching_release_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            release = root / "release.json"
            release.write_text(
                json.dumps(
                    {
                        "release_kind": "evaluation",
                        "status": "confirmatory",
                        "code": {"commit": "abc", "dirty": False},
                    }
                ),
                encoding="utf-8",
            )
            prompt = root / "prompt.py"
            prompt.write_text("PROMPT = 'frozen'\n", encoding="utf-8")
            schema = root / "schema.json"
            schema.write_text("{}", encoding="utf-8")
            analysis = root / "analysis.md"
            analysis.write_text("Primary effect.\n", encoding="utf-8")
            with (
                patch("protocol_freeze.verify_release_manifest", return_value={"passed": True}),
                patch("protocol_freeze._git_state", return_value={"commit": "def", "dirty": False}),
            ):
                with self.assertRaisesRegex(ValueError, "Allocation protocols require a dataset release"):
                    build_protocol_manifest(
                        protocol_root=root,
                        protocol_id="allocation_v1",
                        protocol_phase="allocation",
                        release_manifest_path=release,
                        models=[{"name": "model", "digest": "digest-123"}],
                        conditions=["logic_only"],
                        prompt_files=[prompt],
                        schema_files=[schema],
                        analysis_plan_path=analysis,
                        expected_selected_count=10,
                        expected_main_score_count=8,
                        status="frozen",
                    )


if __name__ == "__main__":
    unittest.main()
