from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from kg_benchmark.cli import main as cli_main
from kg_benchmark.methodology import (
    MethodologyError,
    check_methodology,
    create_methodology_lock,
    hash_freeze_scope,
    load_methodology_bundle,
    require_frozen_methodology,
    validate_methodology_bundle,
)

ROOT = Path(__file__).resolve().parents[1]


class MethodologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = load_methodology_bundle(ROOT)

    def test_repository_freeze_candidate_is_valid_and_workloads_are_exact(self) -> None:
        report = check_methodology(ROOT)
        self.assertTrue(report["valid"], report["errors"])
        self.assertFalse(report["freeze_ready"])
        self.assertTrue(all("__pycache__" not in path for path in report["files"]))
        self.assertEqual(
            report["workloads"],
            {
                "ollama_qwen3_30b": 9600,
                "ollama_llama3_3_70b": 9600,
                "ollama_gpt_oss_120b": 9600,
                "azure_gpt_5_6_sol_high": 4800,
            },
        )
        self.assertEqual(
            report["unresolved_model_revisions"],
            ["ollama_qwen3_30b", "ollama_llama3_3_70b", "azure_gpt_5_6_sol_high"],
        )

    def test_repository_methodology_bundle_matches_strict_schema(self) -> None:
        schema = json.loads((ROOT / "schemas/methodology.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(self.bundle)
        mutated = copy.deepcopy(self.bundle)
        mutated["protocol"]["unexpected"] = True
        self.assertTrue(list(Draft202012Validator(schema).iter_errors(mutated)))

    def test_cross_file_mutations_are_rejected(self) -> None:
        mutations = {
            "model": lambda value: value["models"]["models"][0].__setitem__("tools_disabled", False),
            "quota": lambda value: value["selection_policy"]["populations"]["main-1200"].__setitem__(
                "IC-L", 231
            ),
            "metric": lambda value: value["analysis"]["inference"]["bootstrap"].__setitem__("samples", 4999),
            "audit_rule": lambda value: value["protocol"]["audit"]["construct_review"].__setitem__(
                "sample_size", 449
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                mutated = copy.deepcopy(self.bundle)
                mutate(mutated)
                errors, _ = validate_methodology_bundle(mutated)
                self.assertTrue(errors)

    def test_freeze_scope_hash_changes_for_prompt_schema_and_audit_code_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "paper/prompt.txt": "prompt-v1\n",
                "schemas/response.schema.json": "{}\n",
                "src/audit.py": "RULE = 1\n",
            }
            for relative, content in paths.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            protocol = {
                "freeze_policy": {
                    "scope": {
                        "files": ["paper/prompt.txt", "src/audit.py"],
                        "directories": ["schemas"],
                    }
                }
            }
            _, baseline = hash_freeze_scope(root, protocol)
            for relative in paths:
                with self.subTest(relative=relative):
                    path = root / relative
                    original = path.read_text(encoding="utf-8")
                    path.write_text(original + "mutation\n", encoding="utf-8")
                    _, mutated = hash_freeze_scope(root, protocol)
                    self.assertNotEqual(mutated, baseline)
                    path.write_text(original, encoding="utf-8")

    def test_resolved_ollama_revision_must_be_full_digest(self) -> None:
        mutated = copy.deepcopy(self.bundle)
        qwen = mutated["models"]["models"][0]
        qwen["revision_status"] = "resolved"
        qwen["model_revision"] = "short-tag"
        errors, _ = validate_methodology_bundle(mutated)
        self.assertTrue(any("full lowercase SHA-256" in error for error in errors))

    def test_candidate_cannot_create_lock_or_run_frozen_operations(self) -> None:
        with self.assertRaises(MethodologyError):
            create_methodology_lock(ROOT)
        with self.assertRaises(MethodologyError):
            require_frozen_methodology(ROOT)
        with self.assertRaises(SystemExit):
            cli_main(["acquire"])
        with self.assertRaises(SystemExit):
            cli_main(["run"])

    def test_public_methodology_check_accepts_candidate_but_freeze_ready_mode_fails(self) -> None:
        self.assertEqual(cli_main(["methodology", "check", "--repo-root", str(ROOT)]), 0)
        self.assertEqual(
            cli_main(["methodology", "check", "--repo-root", str(ROOT), "--require-freeze-ready"]),
            1,
        )


if __name__ == "__main__":
    unittest.main()
