import unittest

from kg_benchmark.analysis.statistics import (
    AnalysisStatisticsError,
    cluster_estimate,
    exact_mcnemar,
    holm_adjust,
    paired_cluster_contrast,
)


class AnalysisStatisticsTests(unittest.TestCase):
    def test_cluster_estimate_reports_distinct_micro_and_macro(self) -> None:
        rows = [
            {"case_id": "a", "cluster_key": "g1", "value": 1.0},
            {"case_id": "b", "cluster_key": "g1", "value": 1.0},
            {"case_id": "c", "cluster_key": "g1", "value": 0.0},
            {"case_id": "d", "cluster_key": "g2", "value": 0.0},
        ]
        first = cluster_estimate(rows, samples=100, seed=13)
        second = cluster_estimate(reversed(rows), samples=100, seed=13)
        self.assertEqual(first, second)
        self.assertEqual(first["case_micro"]["estimate"], 0.5)
        self.assertAlmostEqual(first["event_cluster_macro"]["estimate"], 1 / 3)
        self.assertEqual(first["cluster_count"], 2)

    def test_exact_mcnemar_uses_two_sided_binomial_tail(self) -> None:
        result = exact_mcnemar(
            [True] + [False] * 9,
            [False] + [True] * 9,
        )
        self.assertEqual(result["left_correct_right_incorrect"], 1)
        self.assertEqual(result["left_incorrect_right_correct"], 9)
        self.assertAlmostEqual(result["p_value_exact_two_sided"], 22 / 1024)

    def test_paired_contrast_blocks_missing_pairs(self) -> None:
        with self.assertRaisesRegex(AnalysisStatisticsError, "coverage differs"):
            paired_cluster_contrast(
                [{"case_id": "a", "cluster_key": "g1", "value": True}],
                [{"case_id": "b", "cluster_key": "g2", "value": True}],
                samples=10,
            )

    def test_holm_adjustment_is_step_down_monotone(self) -> None:
        adjusted = holm_adjust(
            [
                {"id": "a", "p_value": 0.01},
                {"id": "b", "p_value": 0.04},
                {"id": "c", "p_value": 0.03},
                {"id": "d", "p_value": 0.2},
            ]
        )
        self.assertEqual([row["id"] for row in adjusted], ["a", "b", "c", "d"])
        self.assertEqual(
            [round(row["holm_adjusted_p_value"], 6) for row in adjusted],
            [0.04, 0.09, 0.09, 0.2],
        )


if __name__ == "__main__":
    unittest.main()
