import unittest

from analyze_results import paired_cluster_analysis


class AnalyzeResultsTests(unittest.TestCase):
    def test_paired_cluster_analysis_uses_main_subset_and_event_groups(self) -> None:
        traces = []
        for case_id, group, baseline, treatment in (
            ("a", "g1", False, True),
            ("b", "g1", False, True),
            ("c", "g2", True, True),
        ):
            for bundle, accepted in (("logic_only", baseline), ("local_graph", treatment)):
                traces.append(
                    {
                        "case_id": case_id,
                        "ablation_bundle": bundle,
                        "evaluation_subset": "main_score",
                        "selection_group_key": group,
                        "accepted": accepted,
                        "metrics": {},
                    }
                )
        traces.extend(
            {
                "case_id": "diagnostic",
                "ablation_bundle": bundle,
                "evaluation_subset": "diagnostic",
                "selection_group_key": "g3",
                "accepted": True,
                "metrics": {},
            }
            for bundle in ("logic_only", "local_graph")
        )

        report = paired_cluster_analysis(
            traces,
            baseline_bundle="logic_only",
            treatment_bundle="local_graph",
            bootstrap_samples=100,
            seed=7,
        )

        self.assertEqual(report["paired_case_count"], 3)
        self.assertEqual(report["cluster_count"], 2)
        self.assertAlmostEqual(report["case_micro"]["difference"], 2 / 3)
        self.assertAlmostEqual(report["event_macro"]["difference"], 0.5)
        self.assertFalse(report["population_weighted"]["computed"])


if __name__ == "__main__":
    unittest.main()
