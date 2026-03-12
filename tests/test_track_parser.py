import unittest
from pathlib import Path

from guardian.common import PatchValidationError
from guardian.track_parser import canonicalize, load_schema, normalize_diagnosis


class TrackParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cls.schema_path = repo_root / "schemas" / "track_diagnosis.schema.json"
        cls.schema = load_schema(str(cls.schema_path))

    def test_valid_track_diagnosis(self) -> None:
        normalized = normalize_diagnosis(
            {
                "case_id": "case-1",
                "predicted_track": "a_box",
                "confidence": "high",
                "rationale": "Looks like instance repair."
            },
            schema=self.schema,
        )
        self.assertEqual(normalized.predicted_track, "A_BOX")
        self.assertEqual(len(normalized.canonical_hash), 64)

    def test_invalid_track_diagnosis(self) -> None:
        with self.assertRaises(PatchValidationError) as ctx:
            normalize_diagnosis({"case_id": "case-1", "predicted_track": "WRONG"}, schema=self.schema)
        self.assertEqual(ctx.exception.code, "SCHEMA_VIOLATION")

    def test_canonicalize_deterministic(self) -> None:
        left = canonicalize({"b": 1, "a": 2})
        right = canonicalize({"a": 2, "b": 1})
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
