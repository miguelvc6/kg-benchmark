from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


class AnalysisStatisticsError(ValueError):
    """Raised when a predeclared statistical estimate cannot be computed."""


def _observations(rows: Iterable[dict[str, Any]]) -> list[tuple[str, str, float]]:
    values: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for row in rows:
        case_id = row.get("case_id")
        cluster = row.get("cluster_key")
        value = row.get("value")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise AnalysisStatisticsError("Every estimate requires unique, non-empty case IDs.")
        if not isinstance(cluster, str) or not cluster:
            raise AnalysisStatisticsError(f"Observation {case_id} has no event-cluster key.")
        if isinstance(value, bool):
            numeric = float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            numeric = float(value)
        else:
            raise AnalysisStatisticsError(f"Observation {case_id} has a non-finite numeric value.")
        seen.add(case_id)
        values.append((case_id, cluster, numeric))
    if not values:
        raise AnalysisStatisticsError("Cannot estimate an empty observation set.")
    values.sort()
    return values


def _cluster_summaries(values: list[tuple[str, str, float]]) -> list[tuple[str, float, int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for _, cluster, value in values:
        grouped[cluster].append(value)
    return [
        (cluster, math.fsum(grouped[cluster]), len(grouped[cluster]))
        for cluster in sorted(grouped)
    ]


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise AnalysisStatisticsError("Cannot take a percentile of an empty sample.")
    if not 0 <= probability <= 1:
        raise AnalysisStatisticsError("Percentile probability must be in [0, 1].")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def cluster_estimate(
    rows: Iterable[dict[str, Any]],
    *,
    samples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 13,
) -> dict[str, Any]:
    """Return case-micro and event-macro means with a cluster-bootstrap percentile interval."""
    if not isinstance(samples, int) or samples < 1:
        raise AnalysisStatisticsError("Bootstrap samples must be a positive integer.")
    if not 0 < confidence_level < 1:
        raise AnalysisStatisticsError("Confidence level must be strictly between zero and one.")
    values = _observations(rows)
    clusters = _cluster_summaries(values)
    case_micro = math.fsum(value for _, _, value in values) / len(values)
    event_macro = math.fsum(total / count for _, total, count in clusters) / len(clusters)
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency failure is environment-specific
        raise AnalysisStatisticsError(
            "Paper bootstrap analysis requires NumPy; install the project with --extra analysis."
        ) from exc
    cluster_count = len(clusters)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, cluster_count, size=(samples, cluster_count), dtype=np.int32)
    totals = np.asarray([total for _, total, _ in clusters], dtype=np.float64)
    counts = np.asarray([count for _, _, count in clusters], dtype=np.float64)
    micro_samples = np.sort(totals[indexes].sum(axis=1) / counts[indexes].sum(axis=1)).tolist()
    macro_samples = np.sort((totals / counts)[indexes].mean(axis=1)).tolist()
    alpha = (1 - confidence_level) / 2
    return {
        "case_count": len(values),
        "cluster_count": cluster_count,
        "case_micro": {
            "estimate": case_micro,
            "ci_lower": _percentile(micro_samples, alpha),
            "ci_upper": _percentile(micro_samples, 1 - alpha),
        },
        "event_cluster_macro": {
            "estimate": event_macro,
            "ci_lower": _percentile(macro_samples, alpha),
            "ci_upper": _percentile(macro_samples, 1 - alpha),
        },
        "bootstrap": {
            "method": "percentile_cluster_bootstrap",
            "samples": samples,
            "confidence_level": confidence_level,
            "seed": seed,
        },
    }


def exact_mcnemar(left: Iterable[bool], right: Iterable[bool]) -> dict[str, Any]:
    """Compute the two-sided exact conditional McNemar test for paired binary outcomes."""
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values) or not left_values:
        raise AnalysisStatisticsError("Exact McNemar requires non-empty, equally sized paired outcomes.")
    if any(not isinstance(value, bool) for value in [*left_values, *right_values]):
        raise AnalysisStatisticsError("Exact McNemar outcomes must be boolean.")
    left_only = sum(l_value and not r_value for l_value, r_value in zip(left_values, right_values, strict=True))
    right_only = sum(not l_value and r_value for l_value, r_value in zip(left_values, right_values, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
        p_value = min(1.0, 2 * tail / (2**discordant))
    return {
        "left_correct_right_incorrect": left_only,
        "left_incorrect_right_correct": right_only,
        "discordant_pairs": discordant,
        "p_value_exact_two_sided": p_value,
    }


def paired_cluster_contrast(
    left_rows: Iterable[dict[str, Any]],
    right_rows: Iterable[dict[str, Any]],
    *,
    samples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 13,
) -> dict[str, Any]:
    """Estimate right-minus-left effects with cluster-bootstrap intervals and exact McNemar."""
    left = {case_id: (cluster, value) for case_id, cluster, value in _observations(left_rows)}
    right = {case_id: (cluster, value) for case_id, cluster, value in _observations(right_rows)}
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise AnalysisStatisticsError(
            f"Paired contrast case coverage differs: missing_left={missing_left[:5]} missing_right={missing_right[:5]}"
        )
    differences: list[dict[str, Any]] = []
    left_binary: list[bool] = []
    right_binary: list[bool] = []
    for case_id in sorted(left):
        left_cluster, left_value = left[case_id]
        right_cluster, right_value = right[case_id]
        if left_cluster != right_cluster:
            raise AnalysisStatisticsError(f"Paired case {case_id} changed event-cluster key.")
        if left_value not in {0.0, 1.0} or right_value not in {0.0, 1.0}:
            raise AnalysisStatisticsError("Exact McNemar contrasts require binary endpoint rows.")
        differences.append(
            {"case_id": case_id, "cluster_key": left_cluster, "value": right_value - left_value}
        )
        left_binary.append(bool(left_value))
        right_binary.append(bool(right_value))
    estimate = cluster_estimate(
        differences,
        samples=samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        "pair_count": estimate["case_count"],
        "cluster_count": estimate["cluster_count"],
        "effect_direction": "right_minus_left",
        "case_micro_difference": estimate["case_micro"],
        "event_cluster_macro_difference": estimate["event_cluster_macro"],
        "bootstrap": estimate["bootstrap"],
        "mcnemar": exact_mcnemar(left_binary, right_binary),
    }


def holm_adjust(rows: Iterable[dict[str, Any]], *, alpha: float = 0.05) -> list[dict[str, Any]]:
    """Apply Holm's step-down adjustment while preserving input row order."""
    values = [dict(row) for row in rows]
    if not values:
        raise AnalysisStatisticsError("Holm adjustment requires at least one p-value.")
    if not 0 < alpha < 1:
        raise AnalysisStatisticsError("Holm alpha must be strictly between zero and one.")
    for row in values:
        p_value = row.get("p_value")
        if not isinstance(p_value, (int, float)) or not math.isfinite(float(p_value)) or not 0 <= p_value <= 1:
            raise AnalysisStatisticsError("Holm p-values must be finite values in [0, 1].")
    ranked = sorted(enumerate(values), key=lambda item: (float(item[1]["p_value"]), item[0]))
    running = 0.0
    adjusted: dict[int, float] = {}
    family_size = len(ranked)
    for rank, (original_index, row) in enumerate(ranked):
        candidate = min(1.0, (family_size - rank) * float(row["p_value"]))
        running = max(running, candidate)
        adjusted[original_index] = running
    for index, row in enumerate(values):
        row["holm_adjusted_p_value"] = adjusted[index]
        row["holm_reject_at_0_05"] = adjusted[index] <= alpha
    return values
