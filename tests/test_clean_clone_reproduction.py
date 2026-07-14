from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CleanCloneReproductionTests(unittest.TestCase):
    def test_tracked_files_only_clone_builds_and_runs_installed_cli(self) -> None:
        uv = shutil.which("uv")
        self.assertIsNotNone(uv, "The clean-clone smoke test requires the project's uv runner.")
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        source_is_clean = not subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clone = root / "clone"
            if source_is_clean:
                subprocess.run(
                    ["git", "clone", "--quiet", "--no-local", str(ROOT), str(clone)],
                    check=True,
                )
            else:
                clone.mkdir()
                for raw_path in tracked:
                    if not raw_path:
                        continue
                    relative = Path(os.fsdecode(raw_path))
                    source = ROOT / relative
                    if not source.is_file():
                        continue
                    destination = clone / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)

            for excluded in (".env", ".venv", ".venv-wsl", "data", "data_post_freeze", "work", "runs"):
                self.assertFalse((clone / excluded).exists(), excluded)
            if not source_is_clean:
                subprocess.run(["git", "init", "-q"], cwd=clone, check=True)
                subprocess.run(["git", "config", "user.name", "Reproduction Smoke"], cwd=clone, check=True)
                subprocess.run(["git", "config", "user.email", "smoke@example.invalid"], cwd=clone, check=True)
                subprocess.run(["git", "add", "."], cwd=clone, check=True)
                subprocess.run(
                    ["git", "commit", "-q", "-m", "clean clone fixture"],
                    cwd=clone,
                    check=True,
                )

            distribution = root / "dist"
            subprocess.run(
                [str(uv), "build", "--out-dir", str(distribution)],
                cwd=clone,
                check=True,
                capture_output=True,
                text=True,
            )
            wheels = list(distribution.glob("*.whl"))
            source_distributions = list(distribution.glob("*.tar.gz"))
            self.assertEqual((len(wheels), len(source_distributions)), (1, 1))

            installed = root / "installed"
            subprocess.run(
                [str(uv), "pip", "install", "--target", str(installed), "--no-deps", str(wheels[0])],
                check=True,
                capture_output=True,
                text=True,
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(installed)
            import_check = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from guardian.prompts import get_prompt_template; "
                        "from kg_benchmark.cli import main; "
                        "assert get_prompt_template('reasoning_floor_a_box_zero_shot').system_prompt; "
                        "raise SystemExit(main(['--help']))"
                    ),
                ],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("kg-benchmark", import_check.stdout)

            source_environment = dict(os.environ)
            source_environment["PYTHONPATH"] = os.pathsep.join((str(clone / "src"), str(clone)))
            methodology = subprocess.run(
                [sys.executable, "-m", "kg_benchmark.cli", "methodology", "check", "--repo-root", str(clone)],
                cwd=clone,
                env=source_environment,
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(methodology.stdout)
            self.assertTrue(report["valid"])
            self.assertEqual(report["errors"], [])


if __name__ == "__main__":
    unittest.main()
