import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from artifact_lineage import (
    canonical_record_sha256,
    stage2_projection,
    validate_lineage,
    verify_bound_lineage_manifest,
)


def _stage2(case_id: str = "repair_Q1_2") -> dict:
    return {
        "id": case_id,
        "qid": "Q1",
        "property": "P1",
        "track": "A_BOX",
        "information_type": "TBD",
        "violation_context": {
            "report_fix_date": "2026-01-01T00:00:00",
            "report_revision_old": 1,
            "report_revision_new": 2,
        },
        "repair_target": {"kind": "A_BOX", "action": "UPDATE", "old_value": ["x"], "new_value": ["y"]},
        "persistence_check": {"status": "passed"},
        "qid_label_en": "ignored by lean Stage 4",
    }


class ArtifactLineageTests(unittest.TestCase):
    def _files(self, root: Path, rows: list[dict]) -> dict[str, Path]:
        paths = {name: root / name for name in ("s0.json", "s1.json", "s2.json", "s2.jsonl", "s3.json", "s4.jsonl")}
        paths["s0.json"].write_text(json.dumps({"Q1": {"score": 1.0}}), encoding="utf-8")
        paths["s1.json"].write_text(
            json.dumps([{"qid": "Q1", "property_id": "P1", "fix_date": "2026-01-01T00:00:00", "report_revision_old": 1, "report_revision_new": 2}]),
            encoding="utf-8",
        )
        paths["s2.json"].write_text(json.dumps(rows), encoding="utf-8")
        paths["s2.jsonl"].write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        paths["s3.json"].write_text(json.dumps({row["id"]: {} for row in rows}), encoding="utf-8")
        paths["s4.jsonl"].write_text("".join(json.dumps(stage2_projection(row)) + "\n" for row in rows), encoding="utf-8")
        return paths

    def test_canonical_hash_ignores_object_key_order(self) -> None:
        self.assertEqual(canonical_record_sha256({"b": 2, "a": 1}), canonical_record_sha256({"a": 1, "b": 2}))
        self.assertEqual(canonical_record_sha256({"score": Decimal("0.5")}), canonical_record_sha256({"score": 0.5}))

    def test_complete_lineage_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            p = self._files(Path(temporary), [_stage2()])
            result = validate_lineage(
                stage0_path=p["s0.json"], stage1_path=p["s1.json"], stage2_json_path=p["s2.json"],
                stage2_jsonl_path=p["s2.jsonl"], stage3_path=p["s3.json"], stage4_path=p["s4.jsonl"]
            )
            self.assertTrue(result["validation"]["passed"])
            manifest_path = Path(temporary) / "lineage.json"
            manifest_path.write_text(json.dumps(result), encoding="utf-8")
            bound = verify_bound_lineage_manifest(
                manifest_path, stage2_path=p["s2.json"], stage3_path=p["s3.json"], stage4_path=p["s4.jsonl"]
            )
            self.assertTrue(bound["passed"])

    def test_representation_mutation_and_projection_mutation_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            p = self._files(Path(temporary), [_stage2()])
            mutated = _stage2()
            mutated["repair_target"]["new_value"] = ["z"]
            p["s2.json"].write_text(json.dumps([mutated]), encoding="utf-8")
            stage4 = stage2_projection(_stage2())
            stage4["property"] = "P2"
            p["s4.jsonl"].write_text(json.dumps(stage4) + "\n", encoding="utf-8")
            result = validate_lineage(
                stage0_path=p["s0.json"], stage1_path=p["s1.json"], stage2_json_path=p["s2.json"],
                stage2_jsonl_path=p["s2.jsonl"], stage3_path=p["s3.json"], stage4_path=p["s4.jsonl"]
            )
            self.assertFalse(result["validation"]["passed"])
            self.assertFalse(result["validation"]["stage2_representation_equivalence"]["passed"])
            self.assertFalse(result["validation"]["stage234_identity_and_projection"]["passed"])

    def test_concatenated_jsonl_values_are_counted_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            p = self._files(Path(temporary), [_stage2()])
            extra = _stage2("repair_Q1_3")
            p["s2.jsonl"].write_text(
                json.dumps(_stage2()) + json.dumps(extra) + "\n", encoding="utf-8"
            )
            result = validate_lineage(
                stage0_path=p["s0.json"], stage1_path=p["s1.json"],
                stage2_json_path=p["s2.json"], stage2_jsonl_path=p["s2.jsonl"],
                stage3_path=p["s3.json"], stage4_path=p["s4.jsonl"]
            )
            representation = result["validation"]["stage2_representation_equivalence"]
            self.assertEqual(result["artifacts"]["stage2_jsonl"]["record_count"], 2)
            self.assertEqual(representation["counts"]["jsonl_physical_lines"], 1)
            self.assertFalse(representation["checks"]["jsonl_one_record_per_line"])
            self.assertFalse(representation["passed"])


if __name__ == "__main__":
    unittest.main()
