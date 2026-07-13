import json
import tempfile
import unittest
from pathlib import Path

from experiment_registry import register_experiment


class ExperimentRegistryTests(unittest.TestCase):
    def test_exploratory_run_is_never_paper_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "run_info": {"run_id": "run_a", "code": {"commit": "abc", "dirty": False}},
                        "paper_subsets": {"main_score": {"count": 1}},
                    }
                ),
                encoding="utf-8",
            )
            registry = root / "registry.json"
            schema = Path(__file__).resolve().parents[1] / "schemas" / "experiment_registry.schema.json"

            entry = register_experiment(
                registry_path=registry,
                run_summary_path=summary_path,
                status="exploratory",
                protocol_id="protocol_v1",
                schema_path=schema,
            )

            self.assertFalse(entry["paper_eligible"])
            self.assertIn("status_is_not_confirmatory", entry["ineligibility_reasons"])


if __name__ == "__main__":
    unittest.main()
