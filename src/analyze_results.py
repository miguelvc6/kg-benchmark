#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from lib.utils import iter_jsonl


def _metric_value(trace: dict[str, Any], metric: str) -> float | None:
    if metric == "accepted":
        return 1.0 if trace.get("accepted") else 0.0
    value = trace.get("metrics", {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _exact_paired_binary_p_value(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if discordant == 0:
        return None
    smaller = min(left_only, right_only)
    lower_tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def paired_cluster_analysis(
    traces: Iterable[dict[str, Any]],
    *,
    baseline_bundle: str,
    treatment_bundle: str,
    metric: str = "accepted",
    evaluation_subset: str = "main_score",
    bootstrap_samples: int = 5000,
    seed: int = 13,
    weight_field: str | None = None,
) -> dict[str, Any]:
    by_case: dict[str, dict[str, tuple[float, dict[str, Any]]]] = defaultdict(dict)
    for trace in traces:
        if trace.get("evaluation_subset", "all_selected") != evaluation_subset:
            continue
        bundle = trace.get("ablation_bundle")
        if bundle not in {baseline_bundle, treatment_bundle}:
            continue
        case_id = trace.get("case_id")
        value = _metric_value(trace, metric)
        if not isinstance(case_id, str) or value is None:
            continue
        if bundle in by_case[case_id]:
            raise ValueError(f"Duplicate trace for case_id={case_id}, bundle={bundle}")
        by_case[case_id][bundle] = (value, trace)

    paired: list[tuple[str, float, float, str, float | None]] = []
    for case_id, bundle_rows in by_case.items():
        if baseline_bundle not in bundle_rows or treatment_bundle not in bundle_rows:
            continue
        baseline, baseline_trace = bundle_rows[baseline_bundle]
        treatment, treatment_trace = bundle_rows[treatment_bundle]
        cluster = (
            treatment_trace.get("selection_group_key")
            or treatment_trace.get("tbox_revision_key")
            or baseline_trace.get("selection_group_key")
            or case_id
        )
        raw_weight = treatment_trace.get(weight_field) if weight_field else None
        weight = float(raw_weight) if isinstance(raw_weight, (int, float)) and raw_weight > 0 else None
        paired.append((case_id, baseline, treatment, str(cluster), weight))
    if not paired:
        raise ValueError("No paired traces were available for the requested comparison.")

    clusters: dict[str, list[tuple[str, float, float, str, float | None]]] = defaultdict(list)
    for row in paired:
        clusters[row[3]].append(row)
    cluster_keys = sorted(clusters)
    rng = random.Random(seed)
    bootstrap_differences: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_keys = [rng.choice(cluster_keys) for _ in cluster_keys]
        sampled_rows = [row for key in sampled_keys for row in clusters[key]]
        bootstrap_differences.append(
            sum(row[2] - row[1] for row in sampled_rows) / len(sampled_rows)
        )

    baseline_mean = sum(row[1] for row in paired) / len(paired)
    treatment_mean = sum(row[2] for row in paired) / len(paired)
    cluster_effects = [
        sum(row[2] - row[1] for row in rows) / len(rows)
        for rows in clusters.values()
    ]
    baseline_only = sum(1 for _, left, right, _, _ in paired if left == 1.0 and right == 0.0)
    treatment_only = sum(1 for _, left, right, _, _ in paired if left == 0.0 and right == 1.0)
    weights_available = weight_field is not None and all(row[4] is not None for row in paired)
    weighted_effect = None
    if weights_available:
        total_weight = sum(row[4] or 0.0 for row in paired)
        weighted_effect = sum((row[4] or 0.0) * (row[2] - row[1]) for row in paired) / total_weight

    return {
        "analysis_type": "paired_cluster_bootstrap_v1",
        "metric": metric,
        "evaluation_subset": evaluation_subset,
        "baseline_bundle": baseline_bundle,
        "treatment_bundle": treatment_bundle,
        "paired_case_count": len(paired),
        "cluster_count": len(clusters),
        "case_micro": {
            "baseline_mean": baseline_mean,
            "treatment_mean": treatment_mean,
            "difference": treatment_mean - baseline_mean,
            "confidence_interval_95": [
                _percentile(bootstrap_differences, 0.025),
                _percentile(bootstrap_differences, 0.975),
            ],
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        },
        "event_macro": {
            "difference": sum(cluster_effects) / len(cluster_effects),
            "cluster_definition": "selection_group_key, then tbox_revision_key, then case_id",
        },
        "paired_binary_test": {
            "baseline_only_successes": baseline_only,
            "treatment_only_successes": treatment_only,
            "two_sided_exact_p_value": _exact_paired_binary_p_value(baseline_only, treatment_only),
        },
        "population_weighted": {
            "weight_field": weight_field,
            "difference": weighted_effect,
            "computed": weights_available,
            "reason": None if weights_available else "A positive weight is required for every paired case.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute paired cluster-aware benchmark effects.")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--baseline-bundle", required=True)
    parser.add_argument("--treatment-bundle", required=True)
    parser.add_argument("--metric", default="accepted")
    parser.add_argument("--evaluation-subset", default="main_score")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--weight-field", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = paired_cluster_analysis(
        iter_jsonl(args.traces),
        baseline_bundle=args.baseline_bundle,
        treatment_bundle=args.treatment_bundle,
        metric=args.metric,
        evaluation_subset=args.evaluation_subset,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        weight_field=args.weight_field,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
