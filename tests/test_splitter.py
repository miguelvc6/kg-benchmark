import unittest

from splitter import derive_split_group


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


if __name__ == "__main__":
    unittest.main()
