import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


class JsonSchemaTests(unittest.TestCase):
    def test_all_repository_schemas_are_valid_draft_2020_12(self) -> None:
        schemas_dir = Path(__file__).resolve().parents[1] / "schemas"
        schema_paths = sorted(schemas_dir.glob("*.schema.json"))
        self.assertTrue(schema_paths)
        for path in schema_paths:
            with self.subTest(schema=path.name):
                schema = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(schema.get("$schema"), "https://json-schema.org/draft/2020-12/schema")
                self.assertTrue(str(schema.get("$id", "")).startswith("https://kg-benchmark.local/schemas/"))
                Draft202012Validator.check_schema(schema)


if __name__ == "__main__":
    unittest.main()
