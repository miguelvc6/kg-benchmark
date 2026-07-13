import csv
import json
import tempfile
import unittest
from pathlib import Path

from annotation_protocol import build_blinded_assignments


class AnnotationProtocolTests(unittest.TestCase):
    def test_assigns_two_reviewers_and_removes_classifier_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audit = root / "audit.csv"
            with audit.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case_id", "class", "subtype"])
                writer.writeheader()
                writer.writerow({"case_id": "case_a", "class": "TypeC", "subtype": "EXTERNAL"})
            classified = root / "classified.jsonl"
            classified.write_text(
                json.dumps(
                    {
                        "id": "case_a",
                        "qid": "Q1",
                        "property": "P1",
                        "track": "A_BOX",
                        "repair_target": {"kind": "A_BOX"},
                        "classification": {"class": "TypeC", "subtype": "EXTERNAL"},
                        "build": {"classifier_version": "secret"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            world = root / "world.json"
            world.write_text(json.dumps({"case_a": {"L1_ego_node": {"qid": "Q1"}}}), encoding="utf-8")

            manifest = build_blinded_assignments(
                audit_csv=audit,
                classified_path=classified,
                world_state_path=world,
                output_dir=root / "out",
                annotators=["reviewer_a", "reviewer_b", "reviewer_c"],
            )

            self.assertEqual(manifest["review_count"], 2)
            self.assertEqual(sum(manifest["assignment_counts"].values()), 2)
            card = json.loads((root / "out" / "evidence_cards" / "audit_000001.json").read_text())
            self.assertNotIn("classification", card["benchmark_evidence"])
            self.assertNotIn("track", card["benchmark_evidence"])
            self.assertNotIn("case_a", json.dumps(card))


if __name__ == "__main__":
    unittest.main()
