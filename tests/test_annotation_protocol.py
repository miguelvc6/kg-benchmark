import csv
import json
import stat
import tempfile
import unittest
from pathlib import Path

from annotation_protocol import (
    TASK_ALLOWED_VALUES,
    adjudicate_reviews,
    build_annotation_assignments,
    merge_completed_reviews,
)


class AnnotationProtocolTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        audit = root / "audit.csv"
        with audit.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["case_id"])
            writer.writeheader()
            writer.writerows([{"case_id": "case_a"}, {"case_id": "case_t"}])
        records = [
            {
                "id": "case_a",
                "qid": "Q1",
                "property": "P1",
                "track": "A_BOX",
                "repair_target": {
                    "kind": "A_BOX",
                    "author": "secret-editor",
                    "action": "UPDATE",
                    "old_value": ["Q2"],
                    "new_value": ["Q3"],
                },
                "classification": {
                    "class": "TypeB",
                    "subtype": "LOCAL_TEXT_CONFIRMED",
                    "confidence": "high",
                },
                "popularity": {"score": 0.5},
                "build": {"classifier_version": "secret"},
            },
            {
                "id": "case_t",
                "qid": "P2",
                "property": "P2",
                "track": "T_BOX",
                "repair_target": {
                    "kind": "T_BOX",
                    "author": "secret-editor",
                    "property_revision_id": 2,
                    "property_revision_prev": 1,
                    "constraint_delta": {"changed_constraint_types": ["Q1"]},
                },
                "classification": {"class": "T_BOX", "subtype": "SCHEMA_UPDATE", "confidence": "high"},
                "popularity": {"score": 0.5},
            },
        ]
        classified = root / "classified.jsonl"
        classified.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        world = root / "world.json"
        world.write_text(
            json.dumps(
                {
                    "case_a": {"L1_ego_node": {"qid": "Q1"}},
                    "case_t": {"L1_ego_node": {"qid": "P2"}},
                }
            ),
            encoding="utf-8",
        )
        return audit, classified, world

    def _complete_assignments(self, assignment_dir: Path) -> list[Path]:
        outputs = []
        for path in sorted(assignment_dir.glob("*.csv")):
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0]) if rows else []
            for row in rows:
                task = row["task"]
                for field, allowed in TASK_ALLOWED_VALUES[task].items():
                    row[field] = allowed[0]
                if (
                    task == "locus"
                    and row["blinded_case_id"] == "audit_000001"
                    and row["annotator_id"] == "reviewer_b"
                ):
                    row["predicted_locus"] = "T_BOX"
                row["annotation_timestamp_utc"] = "2026-07-13T00:00:00Z"
            output = path.with_name(f"completed_{path.name}")
            with output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            outputs.append(output)
        return outputs

    def test_build_merge_and_adjudicate_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audit, classified, world = self._fixture(root)
            output = root / "public_assignments"
            private_map = root / "private" / "case_map.json"
            manifest = build_annotation_assignments(
                audit_csv=audit,
                classified_path=classified,
                world_state_path=world,
                output_dir=output,
                private_map_path=private_map,
                annotators=["reviewer_a", "reviewer_b", "reviewer_c"],
            )
            self.assertEqual(manifest["review_count"], 10)
            self.assertEqual(stat.S_IMODE(private_map.stat().st_mode), 0o600)
            locus_card = json.loads(
                (output / "evidence_cards" / "locus" / "audit_000001.json").read_text(encoding="utf-8")
            )
            evidence_card = json.loads(
                (output / "evidence_cards" / "evidence_sufficiency" / "audit_000001.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("repair_target", locus_card["benchmark_evidence"])
            self.assertNotIn("kind", evidence_card["benchmark_evidence"]["repair_target"])
            self.assertNotIn("author", evidence_card["benchmark_evidence"]["repair_target"])
            self.assertNotIn("case_a", json.dumps(locus_card))

            completed = self._complete_assignments(output / "assignments")
            review_output = root / "review_output"
            report = merge_completed_reviews(
                assignment_manifest_path=output / "protocol_manifest.json",
                completed_review_paths=completed,
                private_map_path=private_map,
                output_dir=review_output,
                bootstrap_seed=7,
            )
            self.assertEqual(report["review_count"], 10)
            self.assertEqual(report["disagreement_count"], 1)
            self.assertIn("cohen_kappa", report["metrics"]["locus"]["predicted_locus"])

            disagreements = review_output / "disagreements.csv"
            with disagreements.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0])
            rows[0].update(
                {
                    "adjudicated_value": "A_BOX",
                    "adjudicator_id": "adjudicator_x",
                    "rationale": "Independent evidence supports entity repair.",
                    "adjudication_timestamp_utc": "2026-07-14T00:00:00Z",
                }
            )
            with disagreements.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            private_final = root / "private" / "final.jsonl"
            public_final = root / "adjudicated_summary.json"
            final = adjudicate_reviews(
                merged_reviews_path=review_output / "merged_reviews.jsonl",
                completed_disagreements_path=disagreements,
                private_map_path=private_map,
                agreement_report_path=review_output / "agreement_report.json",
                private_output=private_final,
                public_output=public_final,
            )
            self.assertTrue(final["adjudication_signoff"]["completed"])
            self.assertEqual(final["adjudication_signoff"]["adjudicator_ids"], ["adjudicator_x"])
            self.assertEqual(len(final["artifact_signature"]), 64)
            self.assertNotIn("case_a", public_final.read_text(encoding="utf-8"))
            self.assertEqual(stat.S_IMODE(private_final.stat().st_mode), 0o600)

    def test_private_map_must_be_outside_assignment_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audit, classified, world = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "outside"):
                build_annotation_assignments(
                    audit_csv=audit,
                    classified_path=classified,
                    world_state_path=world,
                    output_dir=root / "out",
                    private_map_path=root / "out" / "private.json",
                    annotators=["a", "b"],
                )


if __name__ == "__main__":
    unittest.main()
