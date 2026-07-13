import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from guardian.tbox_taxonomy_patch_run import (
    prepare_tbox_taxonomy_gold,
    taxonomy_gold_eligibility,
)


class TBoxTaxonomyPatchRunTests(unittest.TestCase):
    def test_prepare_requires_extractable_gold_for_every_selected_tbox_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = Path(tmp) / "classified.jsonl"
            benchmark.write_text(
                json.dumps(
                    {
                        "id": "unsupported",
                        "property": "P31",
                        "track": "T_BOX",
                        "repair_target": {"kind": "T_BOX"},
                        "classification": {
                            "class": "T_BOX",
                            "subtype": "SCHEMA_UPDATE",
                            "confidence": "high",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mechanically supported taxonomy gold"):
                prepare_tbox_taxonomy_gold(
                    classified_path=benchmark,
                    selected_case_ids=["unsupported"],
                    require_complete=True,
                )

    def test_schema_supported_but_unmined_operations_are_ineligible(self) -> None:
        patch_payload = {
            "case_id": "case-1",
            "schema_decision": "CAUSAL_SCHEMA_REPAIR",
            "target": {"pid": "P31", "constraint_type_qid": "Q21510859"},
            "repairs": [{"repair_op": "CLASS_HIERARCHY_ADD"}],
        }
        with patch(
            "guardian.tbox_taxonomy_patch_run.gold_patch_for_record",
            return_value=patch_payload,
        ):
            gold, reason = taxonomy_gold_eligibility({"id": "case-1"})
        self.assertIsNone(gold)
        self.assertEqual(reason, "unsupported_taxonomy_operations:CLASS_HIERARCHY_ADD")


if __name__ == "__main__":
    unittest.main()
