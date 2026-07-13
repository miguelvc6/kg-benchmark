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

METRIC_TYPES = {"binary", "continuous"}
HYPOTHESIS_ROLES = {"primary", "secondary", "exploratory"}


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


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[key] = running
    return adjusted


def paired_cluster_analysis(
    traces: Iterable[dict[str, Any]],
    *,
    baseline_bundle: str,
    treatment_bundle: str,
    comparison_id: str,
    hypothesis_role: str,
    metric: str = "accepted",
    metric_type: str = "binary",
    evaluation_subset: str = "main_score",
    expected_case_ids: Iterable[str] | None = None,
    allow_incomplete_pairs: bool = False,
    bootstrap_samples: int = 5000,
    seed: int = 13,
    weight_field: str | None = None,
    replicate_field: str | None = None,
) -> dict[str, Any]:
    if metric_type not in METRIC_TYPES:
        raise ValueError(f"metric_type must be one of {sorted(METRIC_TYPES)}")
    if hypothesis_role not in HYPOTHESIS_ROLES:
        raise ValueError(f"hypothesis_role must be one of {sorted(HYPOTHESIS_ROLES)}")
    expected = set(expected_case_ids or [])
    by_unit: dict[tuple[str, str], dict[str, tuple[float, dict[str, Any]]]] = defaultdict(dict)
    observed_by_bundle: dict[str, set[str]] = {baseline_bundle: set(), treatment_bundle: set()}
    missing_metric_by_bundle: dict[str, set[str]] = {baseline_bundle: set(), treatment_bundle: set()}
    for trace in traces:
        if trace.get("evaluation_subset", "all_selected") != evaluation_subset:
            continue
        bundle = trace.get("ablation_bundle")
        if bundle not in {baseline_bundle, treatment_bundle}:
            continue
        case_id = trace.get("case_id")
        if not isinstance(case_id, str):
            continue
        observed_by_bundle[bundle].add(case_id)
        value = _metric_value(trace, metric)
        if value is None:
            missing_metric_by_bundle[bundle].add(case_id)
            continue
        if metric_type == "binary" and value not in {0.0, 1.0}:
            raise ValueError(f"Binary metric {metric!r} has non-binary value {value} for case {case_id}.")
        replicate = str(trace.get(replicate_field, "missing")) if replicate_field else "single"
        key = (case_id, replicate)
        if bundle in by_unit[key]:
            raise ValueError(f"Duplicate trace for case_id={case_id}, replicate={replicate}, bundle={bundle}")
        by_unit[key][bundle] = (value, trace)

    baseline_ids = observed_by_bundle[baseline_bundle]
    treatment_ids = observed_by_bundle[treatment_bundle]
    expected_ids = expected or (baseline_ids | treatment_ids)
    missing_baseline = expected_ids - baseline_ids
    missing_treatment = expected_ids - treatment_ids
    population_mismatch = baseline_ids != treatment_ids or bool(missing_baseline or missing_treatment)
    if population_mismatch and not allow_incomplete_pairs:
        raise ValueError(
            "Incomplete paired population: "
            f"missing_baseline={len(missing_baseline)} missing_treatment={len(missing_treatment)} "
            f"baseline_only={len(baseline_ids - treatment_ids)} treatment_only={len(treatment_ids - baseline_ids)}"
        )

    paired: list[dict[str, Any]] = []
    incomplete_units: list[str] = []
    for (case_id, replicate), bundle_rows in sorted(by_unit.items()):
        if baseline_bundle not in bundle_rows or treatment_bundle not in bundle_rows:
            incomplete_units.append(f"{case_id}::{replicate}")
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
        paired.append(
            {
                "case_id": case_id,
                "replicate": replicate,
                "baseline": baseline,
                "treatment": treatment,
                "difference": treatment - baseline,
                "cluster": str(cluster),
                "weight": weight,
            }
        )
    if incomplete_units and not allow_incomplete_pairs:
        raise ValueError(f"Incomplete condition pairs for {len(incomplete_units)} case/replicate units.")
    if not paired:
        raise ValueError("No paired traces were available for the requested comparison.")

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        clusters[row["cluster"]].append(row)
    cluster_keys = sorted(clusters)
    weights_available = weight_field is not None and all(row["weight"] is not None for row in paired)
    rng = random.Random(seed)
    micro_bootstrap: list[float] = []
    macro_bootstrap: list[float] = []
    weighted_bootstrap: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_keys = [rng.choice(cluster_keys) for _ in cluster_keys]
        sampled_clusters = [clusters[key] for key in sampled_keys]
        sampled_rows = [row for rows in sampled_clusters for row in rows]
        micro_bootstrap.append(sum(row["difference"] for row in sampled_rows) / len(sampled_rows))
        macro_bootstrap.append(
            sum(sum(row["difference"] for row in rows) / len(rows) for rows in sampled_clusters)
            / len(sampled_clusters)
        )
        if weights_available:
            total_weight = sum(row["weight"] for row in sampled_rows)
            weighted_bootstrap.append(
                sum(row["weight"] * row["difference"] for row in sampled_rows) / total_weight
            )

    baseline_mean = sum(row["baseline"] for row in paired) / len(paired)
    treatment_mean = sum(row["treatment"] for row in paired) / len(paired)
    cluster_effects = [sum(row["difference"] for row in rows) / len(rows) for rows in clusters.values()]
    weighted_effect = None
    if weights_available:
        total_weight = sum(row["weight"] for row in paired)
        weighted_effect = sum(row["weight"] * row["difference"] for row in paired) / total_weight
    replicate_effects = {
        replicate: sum(row["difference"] for row in paired if row["replicate"] == replicate)
        / sum(row["replicate"] == replicate for row in paired)
        for replicate in sorted({row["replicate"] for row in paired})
    }

    binary_test = None
    if metric_type == "binary":
        baseline_only = sum(
            1 for row in paired if row["baseline"] == 1.0 and row["treatment"] == 0.0
        )
        treatment_only = sum(
            1 for row in paired if row["baseline"] == 0.0 and row["treatment"] == 1.0
        )
        binary_test = {
            "test": "two_sided_exact_mcnemar",
            "baseline_only_successes": baseline_only,
            "treatment_only_successes": treatment_only,
            "raw_p_value": _exact_paired_binary_p_value(baseline_only, treatment_only),
        }

    micro_effect = treatment_mean - baseline_mean
    macro_effect = sum(cluster_effects) / len(cluster_effects)
    micro_ci = [_percentile(micro_bootstrap, 0.025), _percentile(micro_bootstrap, 0.975)]
    macro_ci = [_percentile(macro_bootstrap, 0.025), _percentile(macro_bootstrap, 0.975)]
    weighted_ci = (
        [_percentile(weighted_bootstrap, 0.025), _percentile(weighted_bootstrap, 0.975)]
        if weights_available
        else [None, None]
    )
    return {
        "analysis_type": "paired_cluster_bootstrap_v2",
        "comparison_id": comparison_id,
        "hypothesis_role": hypothesis_role,
        "metric": metric,
        "metric_type": metric_type,
        "evaluation_subset": evaluation_subset,
        "baseline_bundle": baseline_bundle,
        "treatment_bundle": treatment_bundle,
        "population_accounting": {
            "expected_case_count": len(expected_ids),
            "baseline_observed_case_count": len(baseline_ids),
            "treatment_observed_case_count": len(treatment_ids),
            "paired_case_count": len({row["case_id"] for row in paired}),
            "paired_unit_count": len(paired),
            "missing_baseline_case_ids": sorted(missing_baseline),
            "missing_treatment_case_ids": sorted(missing_treatment),
            "baseline_only_case_ids": sorted(baseline_ids - treatment_ids),
            "treatment_only_case_ids": sorted(treatment_ids - baseline_ids),
            "missing_metric_case_ids": {
                baseline_bundle: sorted(missing_metric_by_bundle[baseline_bundle]),
                treatment_bundle: sorted(missing_metric_by_bundle[treatment_bundle]),
            },
            "incomplete_pair_units": incomplete_units,
            "incomplete_pairs_allowed": allow_incomplete_pairs,
        },
        "cluster_count": len(clusters),
        "replicate_field": replicate_field,
        "replicate_effects": replicate_effects,
        "case_micro": {
            "baseline_mean": baseline_mean,
            "treatment_mean": treatment_mean,
            "difference": micro_effect,
            "confidence_interval_95": micro_ci,
        },
        "event_macro": {
            "difference": macro_effect,
            "confidence_interval_95": macro_ci,
            "cluster_definition": "selection_group_key, then tbox_revision_key, then case_id",
        },
        "paired_binary_test": binary_test,
        "population_weighted": {
            "weight_field": weight_field,
            "difference": weighted_effect,
            "confidence_interval_95": weighted_ci,
            "computed": weights_available,
            "reason": None if weights_available else "A positive weight is required for every paired unit.",
        },
        "bootstrap": {"samples": bootstrap_samples, "seed": seed, "resampling_unit": "event_cluster"},
        "standardized_effect_table": [
            {"estimand": "case_micro", "effect": micro_effect, "ci_low": micro_ci[0], "ci_high": micro_ci[1]},
            {"estimand": "event_macro", "effect": macro_effect, "ci_low": macro_ci[0], "ci_high": macro_ci[1]},
            {
                "estimand": "population_weighted",
                "effect": weighted_effect,
                "ci_low": weighted_ci[0],
                "ci_high": weighted_ci[1],
            },
        ],
        "multiplicity": {"family_id": None, "policy": None, "adjusted_p_value": None},
    }


def analyze_comparison_suite(
    traces: Iterable[dict[str, Any]],
    *,
    comparisons: list[dict[str, Any]],
    family_id: str,
    multiplicity_policy: str = "holm",
) -> dict[str, Any]:
    if multiplicity_policy not in {"holm", "none"}:
        raise ValueError("multiplicity_policy must be 'holm' or 'none'.")
    trace_rows = list(traces)
    reports = [paired_cluster_analysis(trace_rows, **comparison) for comparison in comparisons]
    raw_p_values = {
        report["comparison_id"]: report["paired_binary_test"]["raw_p_value"]
        for report in reports
        if report["paired_binary_test"] is not None
        and report["paired_binary_test"]["raw_p_value"] is not None
    }
    adjusted = _holm_adjust(raw_p_values) if multiplicity_policy == "holm" else raw_p_values
    for report in reports:
        report["multiplicity"] = {
            "family_id": family_id,
            "policy": multiplicity_policy,
            "family_test_count": len(raw_p_values),
            "adjusted_p_value": adjusted.get(report["comparison_id"]),
        }
    return {
        "analysis_type": "preregistered_comparison_suite_v1",
        "family_id": family_id,
        "multiplicity_policy": multiplicity_policy,
        "comparisons": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute paired, population-checked cluster-aware effects.")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--suite-spec", help="JSON specification for a multiplicity-adjusted comparison family.")
    parser.add_argument("--baseline-bundle")
    parser.add_argument("--treatment-bundle")
    parser.add_argument("--comparison-id")
    parser.add_argument("--hypothesis-role", choices=sorted(HYPOTHESIS_ROLES))
    parser.add_argument("--metric", default="accepted")
    parser.add_argument("--metric-type", choices=sorted(METRIC_TYPES), default="binary")
    parser.add_argument("--evaluation-subset", default="main_score")
    parser.add_argument("--expected-case-ids", help="Text file with one expected case ID per line.")
    parser.add_argument("--allow-incomplete-pairs", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--weight-field")
    parser.add_argument("--replicate-field")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    traces = list(iter_jsonl(args.traces))
    if args.suite_spec:
        spec = json.loads(Path(args.suite_spec).read_text(encoding="utf-8"))
        report = analyze_comparison_suite(
            traces,
            comparisons=spec["comparisons"],
            family_id=spec["family_id"],
            multiplicity_policy=spec.get("multiplicity_policy", "holm"),
        )
    else:
        if not all((args.baseline_bundle, args.treatment_bundle, args.comparison_id, args.hypothesis_role)):
            parser.error("Single comparisons require bundles, comparison ID, and hypothesis role.")
        expected = None
        if args.expected_case_ids:
            expected = [line.strip() for line in Path(args.expected_case_ids).read_text().splitlines() if line.strip()]
        report = paired_cluster_analysis(
            traces,
            baseline_bundle=args.baseline_bundle,
            treatment_bundle=args.treatment_bundle,
            comparison_id=args.comparison_id,
            hypothesis_role=args.hypothesis_role,
            metric=args.metric,
            metric_type=args.metric_type,
            evaluation_subset=args.evaluation_subset,
            expected_case_ids=expected,
            allow_incomplete_pairs=args.allow_incomplete_pairs,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            weight_field=args.weight_field,
            replicate_field=args.replicate_field,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
