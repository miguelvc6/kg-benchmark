import math
import unittest
from pathlib import Path

from guardian.patch_parser import PatchValidationError, load_schema, normalize_proposal
from guardian.patch_parser import canonicalize


class PatchParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cls.schema_path = repo_root / "schemas" / "verified_repair_proposal.schema.json"
        cls.schema = load_schema(str(cls.schema_path))

    def _base_proposal(self) -> dict:
        return {
            "case_id": "case-1",
            "target": {"qid": "Q42", "pid": "P31"},
            "ops": [{"op": "SET", "pid": "P31", "value": "Q5"}],
        }

    def test_valid_set_with_qid(self) -> None:
        proposal = self._base_proposal()
        proposal["ops"][0]["value"] = "q42"
        normalized = normalize_proposal(proposal, schema=self.schema)
        self.assertEqual(normalized.ops[0].value, "Q42")
        self.assertEqual(normalized.ops[0].op, "SET")
        self.assertEqual(len(normalized.canonical_hash), 64)

    def test_valid_set_with_date(self) -> None:
        proposal = self._base_proposal()
        proposal["ops"][0]["value"] = "2024-02-29"
        normalized = normalize_proposal(proposal, schema=self.schema)
        self.assertEqual(normalized.ops[0].value, "2024-02-29")

    def test_add_then_remove(self) -> None:
        proposal = self._base_proposal()
        proposal["ops"] = [
            {"op": "add", "pid": "p31", "value": "Q5"},
            {"op": "remove", "pid": "P31", "value": "Q5"},
        ]
        normalized = normalize_proposal(proposal, schema=self.schema)
        self.assertEqual([op.op for op in normalized.ops], ["ADD", "REMOVE"])

    def test_remove_without_value_becomes_delete_all(self) -> None:
        proposal = self._base_proposal()
        proposal["ops"] = [{"op": "remove", "pid": "P31"}]
        normalized = normalize_proposal(proposal, schema=self.schema)
        self.assertEqual(normalized.ops[0].op, "DELETE_ALL")
        self.assertIsNone(normalized.ops[0].value)

    def test_invalid_qid(self) -> None:
        proposal = self._base_proposal()
        proposal["target"]["qid"] = "Q0"
        with self.assertRaises(PatchValidationError) as ctx:
            normalize_proposal(proposal, schema=self.schema)
        self.assertEqual(ctx.exception.code, "INVALID_ID")

    def test_invalid_pid(self) -> None:
        proposal = self._base_proposal()
        proposal["target"]["pid"] = "P0"
        with self.assertRaises(PatchValidationError) as ctx:
            normalize_proposal(proposal, schema=self.schema)
        self.assertEqual(ctx.exception.code, "INVALID_ID")

    def test_invalid_date(self) -> None:
        proposal = self._base_proposal()
        proposal["ops"][0]["value"] = "2024-02-30"
        with self.assertRaises(PatchValidationError) as ctx:
            normalize_proposal(proposal, schema=self.schema)
        self.assertEqual(ctx.exception.code, "INVALID_VALUE")

    def test_empty_ops(self) -> None:
        proposal = self._base_proposal()
        proposal["ops"] = []
        with self.assertRaises(PatchValidationError) as ctx:
            normalize_proposal(proposal, schema=self.schema)
        self.assertEqual(ctx.exception.code, "SCHEMA_VIOLATION")

    def test_too_many_ops(self) -> None:
        proposal = self._base_proposal()
        proposal["ops"] = [{"op": "SET", "pid": "P31", "value": "Q5"} for _ in range(51)]
        with self.assertRaises(PatchValidationError) as ctx:
            normalize_proposal(proposal, schema=self.schema)
        self.assertEqual(ctx.exception.code, "SCHEMA_VIOLATION")

    def test_invalid_json_string(self) -> None:
        raw = "{"
        with self.assertRaises(PatchValidationError) as ctx:
            normalize_proposal(raw, schema=self.schema)
        self.assertEqual(ctx.exception.code, "INVALID_JSON")

    def test_nan_value_rejected(self) -> None:
        proposal = self._base_proposal()
        proposal["ops"][0]["value"] = math.nan
        with self.assertRaises(PatchValidationError) as ctx:
            normalize_proposal(proposal, schema=self.schema)
        self.assertEqual(ctx.exception.code, "INVALID_VALUE")

    def test_canonicalize_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            canonicalize({"value": math.nan})


if __name__ == "__main__":
    unittest.main()
