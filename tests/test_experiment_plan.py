import copy
import unittest
from pathlib import Path

from experiment_plan import build_execution_plan, load_execution_matrix, validate_execution_matrix


class ExperimentPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = load_execution_matrix(
            Path(__file__).resolve().parents[1] / "experiments" / "paper_execution_models_v1.json"
        )

    def test_requested_workloads_and_azure_policy(self) -> None:
        plan = build_execution_plan(self.matrix)
        counts = {run["model_id"]: run["expected_request_count"] for run in plan["runs"]}
        self.assertEqual(counts["ollama_qwen3_30b"], 2400)
        self.assertEqual(counts["ollama_llama3_3_70b"], 2400)
        self.assertEqual(counts["ollama_gpt_oss_120b"], 2400)
        self.assertEqual(counts["azure_gpt_5_6_sol_high"], 1200)
        self.assertEqual(plan["tbox_task_version"], "tbox_taxonomy_patch_v1")
        self.assertFalse(plan["reporting_policy"]["combined_abox_tbox_score"])
        self.assertEqual(plan["prompt_configuration"], "experiments/paper_prompt_profile_v1.json")
        azure = next(run for run in plan["runs"] if run["model_id"] == "azure_gpt_5_6_sol_high")
        self.assertIn("high", azure["argv"])
        self.assertIn("tbox_taxonomy_patch_v1", azure["argv"])
        self.assertIn("--no-batch-sync-retry-fallback", azure["argv"])
        self.assertIn("--prompt-profile", azure["argv"])
        self.assertIn("--max-output-tokens", azure["argv"])
        qwen = next(run for run in plan["runs"] if run["model_id"] == "ollama_qwen3_30b")
        self.assertEqual(qwen["argv"][qwen["argv"].index("--ollama-think") + 1], "enabled")
        self.assertEqual(qwen["argv"][qwen["argv"].index("--temperature") + 1], "0")

    def test_added_model_inherits_population_without_code_changes(self) -> None:
        extended = copy.deepcopy(self.matrix)
        model = copy.deepcopy(extended["models"][0])
        model["model_id"] = "ollama_future_model"
        model["model"] = "future:1b"
        extended["models"].append(model)
        plan = build_execution_plan(extended)
        self.assertEqual(len(plan["runs"]), 5)
        self.assertEqual(plan["runs"][-1]["expected_request_count"], 2400)

    def test_confirmatory_validation_rejects_unresolved_revisions(self) -> None:
        with self.assertRaisesRegex(ValueError, "status=frozen"):
            validate_execution_matrix(self.matrix, require_frozen=True)

    def test_methodology_gate_does_not_require_selection_hashes_or_model_digests(self) -> None:
        methodology = copy.deepcopy(self.matrix)
        methodology["status"] = "methodology_frozen"
        validate_execution_matrix(methodology, require_methodology_frozen=True)
        with self.assertRaisesRegex(ValueError, "status=frozen"):
            validate_execution_matrix(methodology, require_frozen=True)


if __name__ == "__main__":
    unittest.main()
