import unittest

from analyze_results import analyze_comparison_suite, paired_cluster_analysis


def _binary_traces() -> list[dict]:
    traces = []
    for case_id, group, baseline, treatment, weight in (
        ("a", "g1", False, True, 2.0),
        ("b", "g1", False, True, 1.0),
        ("c", "g2", True, True, 3.0),
    ):
        for bundle, accepted in (("logic_only", baseline), ("local_graph", treatment)):
            traces.append(
                {
                    "case_id": case_id,
                    "ablation_bundle": bundle,
                    "evaluation_subset": "main_score",
                    "selection_group_key": group,
                    "accepted": accepted,
                    "population_weight": weight,
                    "metrics": {"information_preservation": 0.2 if bundle == "logic_only" else 0.6},
                }
            )
    return traces


class AnalyzeResultsTests(unittest.TestCase):
    def test_binary_analysis_reports_all_estimands_intervals_and_pairing(self) -> None:
        report = paired_cluster_analysis(
            _binary_traces(),
            baseline_bundle="logic_only",
            treatment_bundle="local_graph",
            comparison_id="local_minus_logic",
            hypothesis_role="primary",
            expected_case_ids=["a", "b", "c"],
            bootstrap_samples=100,
            seed=7,
            weight_field="population_weight",
        )
        self.assertEqual(report["population_accounting"]["paired_case_count"], 3)
        self.assertEqual(report["cluster_count"], 2)
        self.assertAlmostEqual(report["case_micro"]["difference"], 2 / 3)
        self.assertAlmostEqual(report["event_macro"]["difference"], 0.5)
        self.assertEqual(len(report["event_macro"]["confidence_interval_95"]), 2)
        self.assertTrue(report["population_weighted"]["computed"])
        self.assertIsNotNone(report["population_weighted"]["confidence_interval_95"][0])
        self.assertEqual(report["paired_binary_test"]["test"], "two_sided_exact_mcnemar")

    def test_continuous_metric_does_not_emit_binary_test(self) -> None:
        report = paired_cluster_analysis(
            _binary_traces(),
            baseline_bundle="logic_only",
            treatment_bundle="local_graph",
            comparison_id="preservation",
            hypothesis_role="secondary",
            metric="information_preservation",
            metric_type="continuous",
            bootstrap_samples=50,
        )
        self.assertAlmostEqual(report["case_micro"]["difference"], 0.4)
        self.assertIsNone(report["paired_binary_test"])

    def test_incomplete_population_fails_by_default_and_is_reported_when_allowed(self) -> None:
        traces = [
            row
            for row in _binary_traces()
            if not (row["case_id"] == "c" and row["ablation_bundle"] == "local_graph")
        ]
        with self.assertRaisesRegex(ValueError, "Incomplete paired population"):
            paired_cluster_analysis(
                traces,
                baseline_bundle="logic_only",
                treatment_bundle="local_graph",
                comparison_id="incomplete",
                hypothesis_role="exploratory",
                expected_case_ids=["a", "b", "c"],
                bootstrap_samples=20,
            )
        report = paired_cluster_analysis(
            traces,
            baseline_bundle="logic_only",
            treatment_bundle="local_graph",
            comparison_id="incomplete",
            hypothesis_role="exploratory",
            expected_case_ids=["a", "b", "c"],
            allow_incomplete_pairs=True,
            bootstrap_samples=20,
        )
        self.assertEqual(report["population_accounting"]["missing_treatment_case_ids"], ["c"])
        self.assertTrue(report["population_accounting"]["incomplete_pair_units"])

    def test_suite_applies_holm_adjustment(self) -> None:
        common = {
            "baseline_bundle": "logic_only",
            "treatment_bundle": "local_graph",
            "hypothesis_role": "exploratory",
            "bootstrap_samples": 20,
        }
        suite = analyze_comparison_suite(
            _binary_traces(),
            comparisons=[
                {**common, "comparison_id": "accepted", "metric": "accepted", "metric_type": "binary"},
                {**common, "comparison_id": "accepted_copy", "metric": "accepted", "metric_type": "binary"},
            ],
            family_id="exploratory_family",
            multiplicity_policy="holm",
        )
        self.assertEqual(suite["multiplicity_policy"], "holm")
        for report in suite["comparisons"]:
            adjusted = report["multiplicity"]["adjusted_p_value"]
            raw = report["paired_binary_test"]["raw_p_value"]
            self.assertGreaterEqual(adjusted, raw)


if __name__ == "__main__":
    unittest.main()
