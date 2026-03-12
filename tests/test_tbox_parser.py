import unittest
from pathlib import Path

from guardian.common import PatchValidationError
from guardian.tbox_parser import canonicalize, load_schema, normalize_proposal, normalize_signature_after


class TBoxParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cls.schema_path = repo_root / "schemas" / "tbox_reform_proposal.schema.json"
        cls.schema = load_schema(str(cls.schema_path))

    def _base_proposal(self) -> dict:
        return {
            "case_id": "reform-1",
            "target": {"pid": "P31", "constraint_type_qid": "Q21510859"},
            "proposal": {
                "action": "RELAXATION_SET_EXPANSION",
                "signature_after": [
                    {
                        "constraint_qid": "Q21510859",
                        "snaktype": "value",
                        "rank": "normal",
                        "qualifiers": [{"property_id": "P2305", "values": ["q5", "Q43229"]}],
                    }
                ],
            },
        }

    def test_valid_reform_proposal(self) -> None:
        normalized = normalize_proposal(self._base_proposal(), schema=self.schema)
        self.assertEqual(normalized.target.pid, "P31")
        self.assertEqual(normalized.target.constraint_type_qid, "Q21510859")
        self.assertEqual(normalized.proposal.signature_after[0]["qualifiers"][0]["values"], ["Q43229", "Q5"])
        self.assertEqual(len(normalized.canonical_hash), 64)

    def test_invalid_action(self) -> None:
        proposal = self._base_proposal()
        proposal["proposal"]["action"] = "WRONG"
        with self.assertRaises(PatchValidationError) as ctx:
            normalize_proposal(proposal, schema=self.schema)
        self.assertEqual(ctx.exception.code, "SCHEMA_VIOLATION")

    def test_invalid_constraint_qid(self) -> None:
        proposal = self._base_proposal()
        proposal["target"]["constraint_type_qid"] = "Q0"
        with self.assertRaises(PatchValidationError) as ctx:
            normalize_proposal(proposal, schema=self.schema)
        self.assertEqual(ctx.exception.code, "INVALID_ID")

    def test_signature_requires_list(self) -> None:
        with self.assertRaises(PatchValidationError):
            normalize_signature_after({"bad": "shape"})

    def test_canonicalize_is_deterministic(self) -> None:
        left = canonicalize({"b": 1, "a": [2, 1]})
        right = canonicalize({"a": [2, 1], "b": 1})
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()

