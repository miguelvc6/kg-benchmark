import json
import tempfile
import unittest
from pathlib import Path

from lib.benchmark_selection import build_selection_manifest, load_selection_manifest, resolve_case_id_filter


class BenchmarkSelectionTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def test_build_selection_manifest_caps_t_box_per_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            classified_path = root / "classified.jsonl"
            self._write_jsonl(
                classified_path,
                [
                    {
                        "id": "a_box_case",
                        "track": "A_BOX",
                    },
                    {
                        "id": "tbox_a",
                        "track": "T_BOX",
                        "repair_target": {"property_revision_id": 101},
                    },
                    {
                        "id": "tbox_b",
                        "track": "T_BOX",
                        "repair_target": {"property_revision_id": 101},
                    },
                    {
                        "id": "tbox_c",
                        "track": "T_BOX",
                        "repair_target": {"property_revision_id": 202},
                    },
                ],
            )

            manifest = build_selection_manifest(classified_path, tbox_cap_per_update=1, seed=13)

            self.assertEqual(manifest["counts"]["selected_a_box_cases"], 1)
            self.assertEqual(manifest["counts"]["selected_t_box_cases"], 2)
            self.assertEqual(manifest["counts"]["selected_cases"], 3)
            self.assertEqual(manifest["counts"]["distinct_t_box_updates"], 2)
            self.assertIn("a_box_case", manifest["selected_case_ids"])

            selected_101 = {
                case_id
                for case_id in manifest["selected_case_ids"]
                if case_id in {"tbox_a", "tbox_b"}
            }
            self.assertEqual(len(selected_101), 1)
            self.assertIn("tbox_c", manifest["selected_case_ids"])

    def test_resolve_case_id_filter_intersects_manifest_and_explicit_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "selection.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "selected_case_ids": ["case_a", "case_b", "case_c"],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_selection_manifest(manifest_path)
            self.assertEqual(manifest["selected_case_ids"], ["case_a", "case_b", "case_c"])

            resolved = resolve_case_id_filter(
                case_ids=["case_b", "case_x"],
                selection_manifest_path=manifest_path,
            )
            self.assertEqual(resolved, ["case_b"])


if __name__ == "__main__":
    unittest.main()
