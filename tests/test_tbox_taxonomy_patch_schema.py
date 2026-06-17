import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


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


class TBoxTaxonomyPatchSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        schema_path = repo_root / "schemas" / "tbox_taxonomy_patch_proposal.schema.json"
        with schema_path.open(encoding="utf-8") as handle:
            cls.schema = json.load(handle)
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def _valid_repair(self, repair_op: str = "CONSTRAINT_QUALIFIER_ADD", taxonomy_code: str = "CQ_PLUS") -> dict:
        return {
            "repair_op": repair_op,
            "taxonomy_code": taxonomy_code,
            "constraint_type_qid": "Q21510859",
            "qualifier_property_id": "P2305",
            "added_values": ["Q5"],
            "removed_values": [],
            "old_value": None,
            "new_value": "Q5",
            "rank_after": "normal",
            "snaktype_after": "VALUE",
            "evidence_level": "VALUE_DELTA_VISIBLE",
        }

    def _valid_patch(self) -> dict:
        return {
            "case_id": "case-1",
            "schema_decision": "CAUSAL_SCHEMA_REPAIR",
            "target": {"pid": "P31", "constraint_type_qid": "Q21510859"},
            "repairs": [self._valid_repair()],
            "rationale": "The historical constraint delta added Q5 to the allowed values.",
            "provenance": [{"kind": "KG", "node_id": "P31", "snippet": "constraint delta"}],
            "uncertainty": {"confidence": 0.9, "notes": "Directly visible value delta."},
        }

    def assertValid(self, payload: dict) -> None:
        errors = sorted(self.validator.iter_errors(payload), key=lambda error: list(error.path))
        if errors:
            self.fail("; ".join(error.message for error in errors))

    def assertInvalid(self, payload: dict) -> None:
        self.assertNotEqual([], list(self.validator.iter_errors(payload)))

    def test_valid_example_for_every_operation_code_pair(self) -> None:
        for repair_op, taxonomy_code in OPERATION_CODE_PAIRS:
            with self.subTest(repair_op=repair_op):
                payload = self._valid_patch()
                payload["repairs"] = [self._valid_repair(repair_op, taxonomy_code)]
                if repair_op in {"CONSTRAINT_REMOVE", "CONSTRAINT_DEPRECATE", "CONSTRAINT_TYPE_REPLACE"}:
                    payload["repairs"][0]["removed_values"] = ["Q43229"]
                    payload["repairs"][0]["old_value"] = "Q43229"
                if repair_op == "CONSTRAINT_DEPRECATE":
                    payload["repairs"][0]["rank_after"] = "deprecated"
                    payload["repairs"][0]["added_values"] = []
                    payload["repairs"][0]["new_value"] = None
                    payload["repairs"][0]["evidence_level"] = "OPERATION_VISIBLE"
                if repair_op == "CONSTRAINT_QUALIFIER_REMOVE":
                    payload["repairs"][0]["added_values"] = []
                    payload["repairs"][0]["removed_values"] = ["Q43229"]
                    payload["repairs"][0]["old_value"] = "Q43229"
                    payload["repairs"][0]["new_value"] = None
                if repair_op == "CONSTRAINT_QUALIFIER_REPLACE":
                    payload["repairs"][0]["removed_values"] = ["Q43229"]
                    payload["repairs"][0]["old_value"] = "Q43229"
                if repair_op in {"CLASS_HIERARCHY_ADD", "OTHER_TBOX_UPDATE"}:
                    payload["repairs"][0]["qualifier_property_id"] = None
                    payload["repairs"][0]["added_values"] = []
                    payload["repairs"][0]["removed_values"] = []
                    payload["repairs"][0]["old_value"] = None
                    payload["repairs"][0]["new_value"] = None
                    payload["repairs"][0]["evidence_level"] = "OPERATION_VISIBLE"
                self.assertValid(payload)

    def test_rejects_placeholder_identifiers(self) -> None:
        for path, value in [
            (("target", "pid"), "P..."),
            (("target", "constraint_type_qid"), "Q..."),
            (("repairs", 0, "constraint_type_qid"), "Q0"),
            (("repairs", 0, "qualifier_property_id"), "P0"),
            (("repairs", 0, "added_values", 0), "none"),
            (("repairs", 0, "removed_values", 0), ""),
        ]:
            with self.subTest(path=path):
                payload = self._valid_patch()
                cursor = payload
                for part in path[:-1]:
                    cursor = cursor[part]
                if isinstance(cursor, list) and not cursor:
                    cursor.append("Q5")
                cursor[path[-1]] = value
                self.assertInvalid(payload)

    def test_no_causal_schema_repair_allows_empty_repairs(self) -> None:
        payload = self._valid_patch()
        payload["schema_decision"] = "NO_CAUSAL_SCHEMA_REPAIR"
        payload["repairs"] = []
        self.assertValid(payload)

    def test_unclear_schema_evidence_allows_empty_repairs(self) -> None:
        payload = self._valid_patch()
        payload["schema_decision"] = "UNCLEAR_SCHEMA_EVIDENCE"
        payload["repairs"] = []
        self.assertValid(payload)

    def test_causal_schema_repair_rejects_empty_repairs(self) -> None:
        payload = self._valid_patch()
        payload["repairs"] = []
        self.assertInvalid(payload)

    def test_rejects_operation_code_mismatch(self) -> None:
        payload = self._valid_patch()
        payload["repairs"][0]["taxonomy_code"] = "C_MINUS"
        self.assertInvalid(payload)

    def test_rejects_missing_value_delta_arrays(self) -> None:
        payload = self._valid_patch()
        del payload["repairs"][0]["added_values"]
        self.assertInvalid(payload)

    def test_rejects_null_value_delta_arrays(self) -> None:
        payload = self._valid_patch()
        payload["repairs"][0]["removed_values"] = None
        self.assertInvalid(payload)

    def test_rejects_unknown_enum_values(self) -> None:
        payload = self._valid_patch()
        payload["schema_decision"] = "SCHEMA_REPAIR"
        self.assertInvalid(payload)

        payload = self._valid_patch()
        payload["repairs"][0]["repair_op"] = "UPDATE_CONSTRAINT"
        self.assertInvalid(payload)

    def test_literal_and_numeric_values_are_allowed(self) -> None:
        payload = self._valid_patch()
        payload["repairs"][0]["added_values"] = ["en", 123, 4.5, True]
        payload["repairs"][0]["new_value"] = "en"
        self.assertValid(payload)

    def test_validates_deepcopy_without_mutation_dependency(self) -> None:
        payload = self._valid_patch()
        copied = copy.deepcopy(payload)
        self.assertValid(copied)


if __name__ == "__main__":
    unittest.main()
