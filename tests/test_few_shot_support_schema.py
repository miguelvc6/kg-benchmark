from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


def _support_manifest() -> dict:
    return {
        "manifest_type": "few_shot_support_set",
        "manifest_version": "static_support_v1",
        "created_at_utc": "2026-06-18T00:00:00Z",
        "source_manifest": "reports/benchmark_selection/dev_prompt_v1_seed_13.json",
        "blocked_manifest": "reports/benchmark_selection/core_v1_seed_13.json",
        "selection_policy": "static_diverse",
        "support_sets": {
            "a_box_repair": [
                {
                    "raw_case_id": "repair_dev_000001",
                    "visible_example_id": "example_a_000001",
                    "role": "a_box_clean_rule",
                    "task_schema": "a_box_v4_spec_only",
                    "notes": "internal only",
                }
            ],
            "t_box_repair": [
                {
                    "raw_case_id": "reform_dev_000001",
                    "visible_example_id": "example_t_000001",
                    "role": "tbox_taxonomy_cq_plus",
                    "task_schema": "tbox_taxonomy_patch_v1",
                    "gold_version": "tbox_taxonomy_patch_gold_dev_v1",
                    "notes": "internal only",
                }
            ],
            "track_diagnosis": [
                {
                    "raw_case_id": "repair_dev_000002",
                    "visible_example_id": "example_d_000001",
                    "role": "diagnosis_a_box_or_t_box",
                    "task_schema": "track_diagnosis_v1",
                    "notes": "internal only",
                }
            ],
        },
        "blocked_overlaps": {
            "core_case_overlap": 0,
            "core_qid_overlap": 0,
            "core_tbox_revision_overlap": 0,
        },
    }


class FewShotSupportSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with (repo_root / "schemas" / "few_shot_support_set.schema.json").open(encoding="utf-8") as handle:
            cls.schema = json.load(handle)
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_generated_support_manifest_validates(self) -> None:
        manifest = _support_manifest()

        self.validator.validate(manifest)
        self.assertEqual(set(manifest["support_sets"]), {"a_box_repair", "t_box_repair", "track_diagnosis"})

    def test_visible_example_ids_are_neutral_not_raw_case_ids(self) -> None:
        manifest = _support_manifest()
        for examples in manifest["support_sets"].values():
            for example in examples:
                visible_id = example["visible_example_id"]
                self.assertTrue(visible_id.startswith("example_"))
                self.assertNotIn("repair_", visible_id)
                self.assertNotIn("reform_", visible_id)

    def test_schema_rejects_raw_case_id_as_visible_example_id(self) -> None:
        manifest = copy.deepcopy(_support_manifest())
        manifest["support_sets"]["a_box_repair"][0]["visible_example_id"] = "repair_dev_000001"

        errors = list(self.validator.iter_errors(manifest))

        self.assertTrue(errors)
        self.assertIn("does not match", errors[0].message)

    def test_schema_separates_task_schemas_by_support_set(self) -> None:
        manifest = copy.deepcopy(_support_manifest())
        manifest["support_sets"]["t_box_repair"][0]["task_schema"] = "a_box_v4_spec_only"

        errors = list(self.validator.iter_errors(manifest))

        self.assertTrue(errors)
        self.assertIn("tbox_taxonomy_patch_v1", errors[0].message)


if __name__ == "__main__":
    unittest.main()
