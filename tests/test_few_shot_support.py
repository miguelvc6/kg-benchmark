from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from select_few_shot_support import build_support_manifest


def _abox_record(case_id: str, qid: str, pid: str, subtype: str) -> dict:
    return {
        "id": case_id,
        "track": "A_BOX",
        "qid": qid,
        "property": pid,
        "labels_en": {"qid": "Example item", "property": "Example property"},
        "violation_context": {"value": "Q1", "report_violation_type_qids": ["Q21503250"]},
        "repair_target": {"kind": "A_BOX", "action": "UPDATE", "old_value": ["Q1"], "new_value": ["Q5"]},
        "classification": {
            "class": "TypeB",
            "subtype": subtype,
            "confidence": "high",
            "constraint_types": [{"qid": "Q21503250"}],
            "diagnostics": {"truth_source": "repair_target.new_value", "truth_tokens": ["Q5"]},
        },
    }


def _tbox_record(case_id: str, qid: str, pid: str, revision: str) -> dict:
    signature = [
        {
            "constraint_qid": "Q21510859",
            "snaktype": "VALUE",
            "rank": "normal",
            "qualifiers": [{"property_id": "P2305", "values": ["Q5"]}],
        }
    ]
    return {
        "id": case_id,
        "track": "T_BOX",
        "qid": qid,
        "property": pid,
        "labels_en": {"qid": "Example item", "property": "Example property"},
        "violation_context": {"value": "Q1", "report_violation_type_qids": ["Q21510859"]},
        "repair_target": {
            "kind": "T_BOX",
            "property_revision_id": revision,
            "constraint_delta": {
                "changed_constraint_types": ["Q21510859"],
                "signature_after": signature,
            },
        },
        "classification": {
            "class": "T_BOX",
            "subtype": "RELAXATION_SET_EXPANSION",
            "confidence": "high",
            "constraint_types": [{"qid": "Q21510859"}],
            "diagnostics": {"truth_source": "constraint_delta", "truth_tokens": ["Q5"]},
        },
    }


def _manifest(case_ids: list[str], records: list[dict]) -> dict:
    annotations = {}
    by_id = {record["id"]: record for record in records}
    for case_id in case_ids:
        record = by_id[case_id]
        annotations[case_id] = {
            "selection_stratum": f"DEV_{record['track']}",
            "group_key": f"group:{case_id}",
            "tbox_revision_key": record.get("repair_target", {}).get("property_revision_id"),
            "main_score": True,
            "diagnostic_only": False,
        }
    return {"selected_case_ids": case_ids, "case_annotations": annotations}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class FewShotSupportTests(unittest.TestCase):
    def test_generator_excludes_core_case_and_core_tbox_revision_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            records = [
                _abox_record("core_abox", "Q1", "P1", "LOCAL_TEXT_CONFIRMED"),
                _abox_record("usable_abox", "Q2", "P2", "LOCAL_TEXT_CONFIRMED"),
                _tbox_record("core_tbox", "Q3", "P3", "rev-blocked"),
                _tbox_record("same_revision_tbox", "Q4", "P4", "rev-blocked"),
                _tbox_record("usable_tbox", "Q5", "P5", "rev-usable"),
            ]
            classified = root / "classified.jsonl"
            classified.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            dev_manifest = root / "dev.json"
            _write_json(
                dev_manifest,
                _manifest(["core_abox", "usable_abox", "same_revision_tbox", "usable_tbox"], records),
            )
            core_manifest = root / "core.json"
            _write_json(core_manifest, _manifest(["core_abox", "core_tbox"], records))

            manifest, report = build_support_manifest(
                classified_benchmark=classified,
                dev_manifest_path=dev_manifest,
                core_manifest_path=core_manifest,
                seed=13,
            )

            selected_ids = {
                example["raw_case_id"]
                for examples in manifest["support_sets"].values()
                for example in examples
            }
            self.assertNotIn("core_abox", selected_ids)
            self.assertNotIn("same_revision_tbox", selected_ids)
            self.assertIn("usable_abox", selected_ids)
            self.assertIn("usable_tbox", selected_ids)
            self.assertEqual(manifest["blocked_overlaps"]["core_case_overlap"], 0)
            self.assertEqual(manifest["blocked_overlaps"]["core_tbox_revision_overlap"], 0)
            self.assertEqual(report["overlaps"]["core_case_overlap"], 0)
            self.assertEqual(report["overlaps"]["core_tbox_revision_overlap"], 0)

    def test_support_schema_rejects_mixed_task_schema_examples(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with (repo_root / "schemas" / "few_shot_support_set.schema.json").open(encoding="utf-8") as handle:
            schema = json.load(handle)
        validator = Draft202012Validator(schema)
        manifest = {
            "manifest_type": "few_shot_support_set",
            "manifest_version": "static_support_v1",
            "created_at_utc": "2026-06-18T00:00:00Z",
            "source_manifest": "dev.json",
            "blocked_manifest": "core.json",
            "selection_policy": "static_diverse",
            "support_sets": {
                "a_box_repair": [
                    {
                        "raw_case_id": "repair_dev_000001",
                        "visible_example_id": "example_a_000001",
                        "role": "a_box_local_evidence",
                        "task_schema": "tbox_taxonomy_patch_v1",
                        "notes": "wrong schema for set",
                    }
                ],
                "t_box_repair": [],
                "track_diagnosis": [],
            },
            "blocked_overlaps": {
                "core_case_overlap": 0,
                "core_qid_overlap": 0,
                "core_tbox_revision_overlap": 0,
            },
        }

        errors = list(validator.iter_errors(manifest))

        self.assertTrue(errors)
        self.assertIn("a_box_v4_spec_only", errors[0].message)


if __name__ == "__main__":
    unittest.main()
