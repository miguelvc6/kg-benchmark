import unittest

from splitter import build_split_manifest, derive_split_group


class SplitterGroupKeyTests(unittest.TestCase):
    def test_tbox_group_key_uses_revision_fallback_and_marks_weak_when_missing(self) -> None:
        strong = {
            "id": "strong_tbox",
            "track": "T_BOX",
            "property": "P123",
            "repair_target": {"kind": "T_BOX", "property_revision_new": "456"},
        }
        weak = {
            "id": "weak_tbox",
            "track": "T_BOX",
            "property": "P123",
            "repair_target": {"kind": "T_BOX"},
        }

        self.assertEqual(derive_split_group(strong)["group_key"], "TBOX::P123::456")
        self.assertFalse(derive_split_group(strong)["weak_group_key"])
        self.assertEqual(derive_split_group(weak)["group_key"], "TBOX::P123::weak_tbox")
        self.assertTrue(derive_split_group(weak)["weak_group_key"])

    def test_abox_group_key_marks_weak_when_qid_or_property_is_missing(self) -> None:
        strong = {"id": "strong_abox", "track": "A_BOX", "qid": "Q1", "property": "P1"}
        weak = {"id": "weak_abox", "track": "A_BOX", "property": "P1"}

        self.assertEqual(derive_split_group(strong)["group_key"], "ABOX::Q1::P1")
        self.assertFalse(derive_split_group(strong)["weak_group_key"])
        self.assertEqual(derive_split_group(weak)["group_key"], "ABOX::weak_abox")
        self.assertTrue(derive_split_group(weak)["weak_group_key"])

    def test_complete_groups_are_never_split(self) -> None:
        records = []
        for group_index in range(15):
            for member_index in range(2):
                records.append(
                    {
                        "id": f"case_{group_index}_{member_index}",
                        "track": "A_BOX",
                        "qid": f"Q{group_index}",
                        "property": "P1",
                        "classification": {"class": "TypeA" if group_index % 2 else "TypeB"},
                        "popularity": {"score": group_index / 14},
                    }
                )

        manifest = build_split_manifest(records, max_delta=1.0)

        split_by_id = {
            case_id: split_name
            for split_name, case_ids in manifest["splits"].items()
            for case_id in case_ids
        }
        for group_index in range(15):
            self.assertEqual(
                split_by_id[f"case_{group_index}_0"],
                split_by_id[f"case_{group_index}_1"],
            )
        self.assertEqual(manifest["validation"]["cross_split_group_count"], 0)
        self.assertEqual(manifest["counts"]["groups"], 15)
        self.assertEqual(
            manifest["counts"]["popularity_buckets"],
            {"tail": 10, "mid": 10, "head": 10},
        )

    def test_group_assignment_is_deterministic(self) -> None:
        records = [
            {
                "id": f"case_{index}",
                "track": "A_BOX",
                "qid": f"Q{index}",
                "property": "P1",
                "classification": {"class": "TypeA"},
                "popularity": {"score": index / 9},
            }
            for index in range(10)
        ]

        first = build_split_manifest(records, seed=41, max_delta=1.0)
        second = build_split_manifest(reversed(records), seed=41, max_delta=1.0)

        self.assertEqual(first["splits"], second["splits"])


if __name__ == "__main__":
    unittest.main()
