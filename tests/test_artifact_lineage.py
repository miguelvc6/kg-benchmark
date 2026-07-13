import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from artifact_lineage import (
    canonical_record_sha256,
    refresh_lineage_provenance,
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
        "popularity": {"score": 1.0},
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
            precursor_bound = verify_bound_lineage_manifest(
                manifest_path, stage2_path=p["s2.jsonl"], stage3_path=p["s3.json"], stage4_path=p["s4.jsonl"]
            )
            self.assertFalse(precursor_bound["checks"]["stage2_is_declared_authoritative_artifact"])
            self.assertFalse(precursor_bound["passed"])

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

    def test_stage0_popularity_payload_mismatch_fails_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            p = self._files(Path(temporary), [_stage2()])
            p["s0.json"].write_text(json.dumps({"Q1": {"score": 2.0}}), encoding="utf-8")
            result = validate_lineage(
                stage0_path=p["s0.json"], stage1_path=p["s1.json"],
                stage2_json_path=p["s2.json"], stage2_jsonl_path=p["s2.jsonl"],
                stage3_path=p["s3.json"], stage4_path=p["s4.jsonl"],
            )
            provenance = result["validation"]["stage0_stage1_provenance"]
            self.assertFalse(provenance["checks"]["stage2_popularity_payloads_equal"])
            self.assertEqual(provenance["popularity_payload_mismatch_case_ids"], ["repair_Q1_2"])
            self.assertFalse(result["validation"]["passed"])

    def test_provenance_refresh_reuses_only_hash_bound_passing_subchecks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            p = self._files(Path(temporary), [_stage2()])
            prior = validate_lineage(
                stage0_path=p["s0.json"], stage1_path=p["s1.json"],
                stage2_json_path=p["s2.json"], stage2_jsonl_path=p["s2.jsonl"],
                stage3_path=p["s3.json"], stage4_path=p["s4.jsonl"],
            )
            prior_path = Path(temporary) / "prior.json"
            prior_path.write_text(json.dumps(prior), encoding="utf-8")
            refreshed = refresh_lineage_provenance(
                prior_manifest_path=prior_path,
                stage0_path=p["s0.json"], stage1_path=p["s1.json"], stage2_json_path=p["s2.json"],
            )
            self.assertTrue(refreshed["validation"]["passed"])
            self.assertTrue(refreshed["validation_reuse"]["all_artifact_hashes_reverified"])
            p["s4.jsonl"].write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stage4 changed"):
                refresh_lineage_provenance(
                    prior_manifest_path=prior_path,
                    stage0_path=p["s0.json"], stage1_path=p["s1.json"],
                    stage2_json_path=p["s2.json"],
                )

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

    def test_declared_filtered_enriched_successor_passes_without_rewriting_precursor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authoritative = _stage2()
            authoritative["popularity"] = {"score": 1.0}
            p = self._files(root, [authoritative])
            precursor_rows = [_stage2(), _stage2("repair_Q1_extra")]
            for row in precursor_rows:
                row.pop("popularity")
            p["s2.jsonl"].write_text(
                "".join(json.dumps(row) + "\n" for row in precursor_rows), encoding="utf-8"
            )
            policy = {
                "mode": "ordered_filtered_enriched_successor",
                "authoritative_artifact": "stage2_json",
                "precursor_artifact": "stage2_jsonl",
                "allowed_authoritative_only_fields": ["popularity"],
                "allow_precursor_only_records": True,
                "allow_precursor_jsonl_multi_value_lines": False,
                "expected_artifact_sha256": {
                    "stage2_json": canonical_record_sha256([]),
                    "stage2_jsonl": canonical_record_sha256([]),
                },
            }
            from artifact_release import sha256_file

            policy["expected_artifact_sha256"] = {
                "stage2_json": sha256_file(p["s2.json"]),
                "stage2_jsonl": sha256_file(p["s2.jsonl"]),
            }
            result = validate_lineage(
                stage0_path=p["s0.json"], stage1_path=p["s1.json"],
                stage2_json_path=p["s2.json"], stage2_jsonl_path=p["s2.jsonl"],
                stage3_path=p["s3.json"], stage4_path=p["s4.jsonl"],
                source_provenance=[{"status": "restored", "note": "test fixture"}],
                reconciliation_policy=policy,
            )
            relationship = result["validation"]["stage2_relationship"]
            self.assertTrue(result["validation"]["passed"])
            self.assertTrue(relationship["passed"])
            self.assertEqual(relationship["counts"]["precursor_only"], 1)
            self.assertFalse(result["validation"]["stage2_representation_equivalence"]["passed"])

    def test_reconciliation_rejects_mutation_reordering_and_unbound_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _stage2("repair_Q1_first")
            second = _stage2("repair_Q1_second")
            first["popularity"] = {"score": 1.0}
            second["popularity"] = {"score": 1.0}
            p = self._files(root, [second, first])
            mutated = _stage2("repair_Q1_second")
            mutated["repair_target"]["new_value"] = ["unexpected"]
            mutated.pop("popularity")
            p["s2.jsonl"].write_text(
                json.dumps(first | {"popularity": None}) + "\n" + json.dumps(mutated) + "\n",
                encoding="utf-8",
            )
            policy = {
                "mode": "ordered_filtered_enriched_successor",
                "authoritative_artifact": "stage2_json",
                "precursor_artifact": "stage2_jsonl",
                "allowed_authoritative_only_fields": ["popularity"],
                "allow_precursor_only_records": True,
                "allow_precursor_jsonl_multi_value_lines": False,
                "expected_artifact_sha256": {"stage2_json": "0" * 64, "stage2_jsonl": "0" * 64},
            }
            result = validate_lineage(
                stage0_path=p["s0.json"], stage1_path=p["s1.json"],
                stage2_json_path=p["s2.json"], stage2_jsonl_path=p["s2.jsonl"],
                stage3_path=p["s3.json"], stage4_path=p["s4.jsonl"],
                source_provenance=[{"status": "restored"}], reconciliation_policy=policy,
            )
            checks = result["validation"]["stage2_relationship"]["checks"]
            self.assertFalse(checks["declared_artifact_hashes_match"])
            self.assertFalse(checks["authoritative_order_is_precursor_subsequence"])
            self.assertFalse(checks["shared_records_equal_after_declared_enrichment_removed"])
            self.assertFalse(checks["declared_enrichment_absent_from_precursor"])
            self.assertFalse(result["validation"]["passed"])


if __name__ == "__main__":
    unittest.main()
