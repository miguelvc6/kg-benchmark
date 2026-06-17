import unittest
from pathlib import Path

from guardian.tbox_taxonomy_patch_parser import PatchValidationError, load_schema, normalize_tbox_taxonomy_patch


OPERATION_CODE_PAIRS = [
    ("CONSTRAINT_REMOVE", "C_MINUS"),
    ("CONSTRAINT_DEPRECATE", "C_D"),
    ("CONSTRAINT_ADD", "C_PLUS"),
    ("CONSTRAINT_TYPE_REPLACE", "C_REPLACE"),
    ("CONSTRAINT_QUALIFIER_ADD", "CQ_PLUS"),
    ("CONSTRAINT_QUALIFIER_REMOVE", "CQ_MINUS"),
    ("CONSTRAINT_QUALIFIER_REPLACE", "CQ_REPLACE"),
    ("CLASS_HIERARCHY_ADD", "SUBCLASS_PLUS"),
    ("EXCEPTION_ADD", "E_PLUS"),
    ("OTHER_TBOX_UPDATE", "OTHER"),
]


class TBoxTaxonomyPatchParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cls.schema = load_schema(repo_root / "schemas" / "tbox_taxonomy_patch_proposal.schema.json")

    def _repair(self, repair_op: str = "CONSTRAINT_QUALIFIER_ADD", taxonomy_code: str = "CQ_PLUS") -> dict:
        return {
            "repair_op": repair_op,
            "taxonomy_code": taxonomy_code,
            "constraint_type_qid": "q21510859",
            "qualifier_property_id": "p2305",
            "added_values": ["q5"],
            "removed_values": [],
            "old_value": None,
            "new_value": "q5",
            "rank_after": "normal",
            "snaktype_after": "value",
            "evidence_level": "value_delta_visible",
        }

    def _patch(self) -> dict:
        return {
            "case_id": "case-1",
            "schema_decision": "causal_schema_repair",
            "target": {"pid": "p31", "constraint_type_qid": "q21510859"},
            "repairs": [self._repair()],
            "rationale": "Visible historical delta.",
            "provenance": [{"kind": "kg", "node_id": "p31", "snippet": "constraint delta"}],
            "uncertainty": {"confidence": 0.75, "notes": "direct evidence"},
        }

    def test_accepts_valid_example_for_every_operation(self) -> None:
        for repair_op, taxonomy_code in OPERATION_CODE_PAIRS:
            with self.subTest(repair_op=repair_op):
                payload = self._patch()
                payload["repairs"] = [self._repair(repair_op, taxonomy_code)]
                if repair_op == "CONSTRAINT_DEPRECATE":
                    payload["repairs"][0]["rank_after"] = "deprecated"
                    payload["repairs"][0]["added_values"] = []
                    payload["repairs"][0]["new_value"] = None
                    payload["repairs"][0]["evidence_level"] = "operation_visible"
                if repair_op in {"CONSTRAINT_REMOVE", "CONSTRAINT_QUALIFIER_REMOVE"}:
                    payload["repairs"][0]["added_values"] = []
                    payload["repairs"][0]["removed_values"] = ["q43229"]
                    payload["repairs"][0]["old_value"] = "q43229"
                    payload["repairs"][0]["new_value"] = None
                if repair_op == "CONSTRAINT_QUALIFIER_REPLACE":
                    payload["repairs"][0]["removed_values"] = ["q43229"]
                    payload["repairs"][0]["old_value"] = "q43229"
                if repair_op in {"CLASS_HIERARCHY_ADD", "OTHER_TBOX_UPDATE"}:
                    payload["repairs"][0]["qualifier_property_id"] = None
                    payload["repairs"][0]["added_values"] = []
                    payload["repairs"][0]["removed_values"] = []
                    payload["repairs"][0]["old_value"] = None
                    payload["repairs"][0]["new_value"] = None
                    payload["repairs"][0]["evidence_level"] = "operation_visible"
                normalized = normalize_tbox_taxonomy_patch(payload, schema=self.schema)
                self.assertEqual(normalized.repairs[0].repair_op, repair_op)
                self.assertEqual(normalized.repairs[0].taxonomy_code, taxonomy_code)
                self.assertEqual(normalized.target.pid, "P31")
                self.assertEqual(normalized.target.constraint_type_qid, "Q21510859")

    def test_rejects_placeholder_identifiers(self) -> None:
        for path, value in [
            (("target", "pid"), "P..."),
            (("target", "constraint_type_qid"), "Q..."),
            (("repairs", 0, "constraint_type_qid"), "Q0"),
            (("repairs", 0, "qualifier_property_id"), "P0"),
            (("repairs", 0, "added_values", 0), "none"),
        ]:
            with self.subTest(path=path):
                payload = self._patch()
                cursor = payload
                for part in path[:-1]:
                    cursor = cursor[part]
                cursor[path[-1]] = value
                with self.assertRaises(PatchValidationError):
                    normalize_tbox_taxonomy_patch(payload, schema=self.schema)

    def test_rejects_invalid_enum_values(self) -> None:
        payload = self._patch()
        payload["schema_decision"] = "SCHEMA_REPAIR"
        with self.assertRaises(PatchValidationError):
            normalize_tbox_taxonomy_patch(payload, schema=self.schema)

        payload = self._patch()
        payload["repairs"][0]["repair_op"] = "UPDATE_CONSTRAINT"
        with self.assertRaises(PatchValidationError):
            normalize_tbox_taxonomy_patch(payload, schema=self.schema)

    def test_rejects_operation_code_mismatch(self) -> None:
        payload = self._patch()
        payload["repairs"][0]["taxonomy_code"] = "C_MINUS"
        with self.assertRaises(PatchValidationError):
            normalize_tbox_taxonomy_patch(payload, schema=self.schema)

    def test_rejects_causal_schema_repair_with_empty_repairs(self) -> None:
        payload = self._patch()
        payload["repairs"] = []
        with self.assertRaises(PatchValidationError):
            normalize_tbox_taxonomy_patch(payload, schema=self.schema)

    def test_accepts_no_causal_schema_repair_with_empty_repairs(self) -> None:
        payload = self._patch()
        payload["schema_decision"] = "NO_CAUSAL_SCHEMA_REPAIR"
        payload["repairs"] = []
        normalized = normalize_tbox_taxonomy_patch(payload, schema=self.schema)
        self.assertEqual(normalized.repairs, [])

    def test_accepts_unclear_schema_evidence_with_empty_repairs(self) -> None:
        payload = self._patch()
        payload["schema_decision"] = "UNCLEAR_SCHEMA_EVIDENCE"
        payload["repairs"] = []
        normalized = normalize_tbox_taxonomy_patch(payload, schema=self.schema)
        self.assertEqual(normalized.repairs, [])

    def test_rejects_constraint_qid_outside_allowed_set(self) -> None:
        payload = self._patch()
        with self.assertRaises(PatchValidationError):
            normalize_tbox_taxonomy_patch(payload, schema=self.schema, constraint_type_qids={"Q21502410"})

    def test_canonicalization_is_deterministic(self) -> None:
        payload_a = self._patch()
        payload_a["repairs"] = [
            self._repair("CONSTRAINT_QUALIFIER_REMOVE", "CQ_MINUS"),
            self._repair("CONSTRAINT_QUALIFIER_ADD", "CQ_PLUS"),
        ]
        payload_a["repairs"][0]["added_values"] = []
        payload_a["repairs"][0]["removed_values"] = ["q43229", "q5"]
        payload_a["repairs"][0]["old_value"] = "q43229"
        payload_a["repairs"][0]["new_value"] = None
        payload_a["repairs"][1]["added_values"] = ["q5", "q43229"]

        payload_b = self._patch()
        payload_b["repairs"] = [payload_a["repairs"][1], payload_a["repairs"][0]]

        normalized_a = normalize_tbox_taxonomy_patch(payload_a, schema=self.schema)
        normalized_b = normalize_tbox_taxonomy_patch(payload_b, schema=self.schema)
        self.assertEqual(normalized_a.canonical_hash, normalized_b.canonical_hash)
        self.assertEqual([repair.repair_op for repair in normalized_a.repairs], ["CONSTRAINT_QUALIFIER_ADD", "CONSTRAINT_QUALIFIER_REMOVE"])
        self.assertEqual(normalized_a.repairs[0].added_values, ["Q43229", "Q5"])

    def test_hidden_metadata_is_not_preserved(self) -> None:
        payload = self._patch()
        payload["metadata"] = {"answer": "hidden"}
        normalized = normalize_tbox_taxonomy_patch(payload, schema=self.schema)
        self.assertNotIn("metadata", normalized.to_dict())


if __name__ == "__main__":
    unittest.main()
