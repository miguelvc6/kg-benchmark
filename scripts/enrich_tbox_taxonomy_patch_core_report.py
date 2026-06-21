"""Enrich the frozen full-core T-box taxonomy-patch report.

This script derives report-only diagnostics from existing gold, prediction, and
evaluation-summary artifacts. It does not render prompts, call a model, or
change gold extraction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

MISSING = "__MISSING__"
EXTRA = "__EXTRA__"
NO_REPAIR = "__NO_REPAIR__"
NO_QUALIFIER = "__NO_QUALIFIER__"

HEADLINE_GROUP_METRICS = {
    "family_success": "family_level_success",
    "schema_decision_match": "schema_decision_match",
    "taxonomy_code_match": "taxonomy_code_exact_match",
    "value_delta_false_positive": "value_delta_claimed_when_gold_absent",
    "value_delta_under_specification": "family_only_when_value_delta_gold_present",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repair_values(row: dict[str, Any], field: str) -> list[str]:
    values = [str(repair.get(field)) for repair in row.get("repairs", []) if repair.get(field) is not None]
    if values:
        return values
    if field == "qualifier_property_id":
        return [NO_QUALIFIER]
    return [NO_REPAIR]


def _schema_values(row: dict[str, Any]) -> list[str]:
    return [str(row.get("schema_decision") or MISSING)]


def _values_for_matrix(row: dict[str, Any], matrix_name: str) -> list[str]:
    if matrix_name == "schema_decision":
        return _schema_values(row)
    if matrix_name == "taxonomy_code":
        return _repair_values(row, "taxonomy_code")
    if matrix_name == "repair_operation":
        return _repair_values(row, "repair_op")
    if matrix_name == "qualifier_property":
        return _repair_values(row, "qualifier_property_id")
    raise ValueError(f"unknown confusion matrix: {matrix_name}")


def _aligned_multiset_confusion(gold_values: list[str], pred_values: list[str]) -> Counter[tuple[str, str]]:
    """Align multi-label gold/prediction values without cross-product inflation."""
    gold = Counter(gold_values)
    pred = Counter(pred_values)
    pairs: Counter[tuple[str, str]] = Counter()

    for label in sorted(gold.keys() & pred.keys()):
        overlap = min(gold[label], pred[label])
        if overlap:
            pairs[(label, label)] += overlap
            gold[label] -= overlap
            pred[label] -= overlap

    remaining_gold = [label for label, count in sorted(gold.items()) for _ in range(count) if count > 0]
    remaining_pred = [label for label, count in sorted(pred.items()) for _ in range(count) if count > 0]

    for gold_label, pred_label in zip(remaining_gold, remaining_pred):
        pairs[(gold_label, pred_label)] += 1

    if len(remaining_gold) > len(remaining_pred):
        for gold_label in remaining_gold[len(remaining_pred) :]:
            pairs[(gold_label, MISSING)] += 1
    elif len(remaining_pred) > len(remaining_gold):
        for pred_label in remaining_pred[len(remaining_gold) :]:
            pairs[(EXTRA, pred_label)] += 1

    return pairs


def build_confusion_matrices(
    gold_by_case: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    matrices: dict[str, dict[str, Any]] = {}
    for matrix_name in ("schema_decision", "taxonomy_code", "repair_operation", "qualifier_property"):
        pair_counts: Counter[tuple[str, str]] = Counter()
        row_total = 0
        for prediction in predictions:
            case_id = prediction["case_id"]
            gold_row = gold_by_case[case_id]
            pair_counts.update(
                _aligned_multiset_confusion(
                    _values_for_matrix(gold_row, matrix_name),
                    _values_for_matrix(prediction, matrix_name),
                )
            )
            row_total += 1

        labels = sorted({gold for gold, _ in pair_counts} | {pred for _, pred in pair_counts})
        rows = [
            {
                "gold": gold_label,
                "predicted": pred_label,
                "count": count,
            }
            for (gold_label, pred_label), count in sorted(pair_counts.items())
        ]
        matrices[matrix_name] = {
            "alignment": (
                "exact-overlap-first multiset alignment; unmatched gold labels use __MISSING__, "
                "unmatched predicted labels use __EXTRA__"
            ),
            "labels": labels,
            "rows": rows,
            "total_aligned_items": sum(pair_counts.values()),
            "total_cases": row_total,
        }
    return matrices


def build_out_of_current_gold_operation_fp_rates(
    gold_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_operation_counts = Counter(
        repair.get("repair_op") for row in gold_rows for repair in row.get("repairs", []) if repair.get("repair_op")
    )
    prediction_counts = Counter(
        repair.get("repair_op")
        for row in predictions
        for repair in row.get("repairs", [])
        if repair.get("repair_op")
    )
    current_gold_ops = set(gold_operation_counts)
    total_predicted = sum(prediction_counts.values())
    rows = []
    for operation, predicted_count in sorted(prediction_counts.items()):
        out_of_current_gold = operation not in current_gold_ops
        false_positive_count = predicted_count if out_of_current_gold else 0
        rows.append(
            {
                "operation": operation,
                "predicted_count": predicted_count,
                "gold_count": gold_operation_counts.get(operation, 0),
                "out_of_current_gold_operation": out_of_current_gold,
                "false_positive_count": false_positive_count,
                "false_positive_rate_among_predicted_repairs": (
                    false_positive_count / total_predicted if total_predicted else None
                ),
            }
        )
    return {
        "definition": (
            "Operations absent from the current gold operation set are counted as out-of-current-gold "
            "false positives."
        ),
        "current_gold_operation_set": sorted(current_gold_ops),
        "gold_operation_counts": dict(sorted(gold_operation_counts.items())),
        "total_predicted_repairs": total_predicted,
        "rows": rows,
        "overall_out_of_current_gold_false_positive_rate": (
            sum(row["false_positive_count"] for row in rows) / total_predicted if total_predicted else None
        ),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(tp: int, pred: int, gold: int) -> float | None:
    denominator = pred + gold
    return 2 * tp / denominator if denominator else None


def _property_from_row(row: dict[str, Any], annotation: dict[str, Any]) -> str:
    pid = row.get("target", {}).get("pid")
    if pid:
        return str(pid)
    group_key = str(annotation.get("group_key", ""))
    parts = group_key.split("::")
    return parts[1] if len(parts) >= 2 else "__UNKNOWN_PROPERTY__"


def _summarize_group(traces: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for display_name, trace_metric_name in HEADLINE_GROUP_METRICS.items():
        applicable = [trace["metrics"].get(trace_metric_name) for trace in traces]
        applicable = [value for value in applicable if value is not None]
        numerator = sum(1 for value in applicable if value is True)
        denominator = len(applicable)
        metrics[display_name] = {
            "numerator": numerator,
            "denominator": denominator,
            "rate": _rate(numerator, denominator),
        }

    detail_totals = Counter()
    gold_value_rows = 0
    for trace in traces:
        detail_totals.update(trace.get("metric_detail", {}))
        if trace["metrics"].get("gold_has_value_delta") is True:
            gold_value_rows += 1

    value_gold = int(detail_totals.get("value_gold", 0))
    value_pred = int(detail_totals.get("value_pred", 0))
    value_tp = int(detail_totals.get("value_tp", 0))
    metrics["value_delta_f1_when_applicable"] = {
        "applicable_gold_rows": gold_value_rows,
        "gold_value_delta_items": value_gold,
        "predicted_value_delta_items": value_pred,
        "true_positive_value_delta_items": value_tp,
        "rate": _f1(value_tp, value_pred, value_gold) if value_gold else None,
    }
    return {"row_count": len(traces), "metrics": metrics}


def _macro_from_groups(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted({metric for group in groups.values() for metric in group["metrics"]})
    macro_metrics: dict[str, Any] = {}
    for metric_name in metric_names:
        rates = [
            group["metrics"][metric_name]["rate"]
            for group in groups.values()
            if group["metrics"][metric_name]["rate"] is not None
        ]
        macro_metrics[metric_name] = {
            "macro_average": mean(rates) if rates else None,
            "groups_with_applicable_rate": len(rates),
            "total_groups": len(groups),
        }
    return {
        "group_count": len(groups),
        "row_count": sum(group["row_count"] for group in groups.values()),
        "metrics": macro_metrics,
        "groups": groups,
    }


def build_macro_averages(
    traces: list[dict[str, Any]],
    gold_by_case: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_property: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_revision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        case_id = trace["case_id"]
        annotation = annotations.get(case_id, {})
        property_key = _property_from_row(gold_by_case[case_id], annotation)
        revision_key = str(annotation.get("tbox_revision_key") or annotation.get("group_key") or "__UNKNOWN_REVISION__")
        by_property[property_key].append(trace)
        by_revision[revision_key].append(trace)

    return {
        "by_property": _macro_from_groups({key: _summarize_group(value) for key, value in sorted(by_property.items())}),
        "by_tbox_revision": _macro_from_groups(
            {key: _summarize_group(value) for key, value in sorted(by_revision.items())}
        ),
    }


def build_subset_audit(
    gold_rows: list[dict[str, Any]],
    evaluation_summary: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    taxonomy_case_ids = {row["case_id"] for row in gold_rows}
    manifest_main_score_case_ids = set(manifest.get("main_score_case_ids", []))
    manifest_diagnostic_case_ids = set(manifest.get("diagnostic_case_ids", []))
    trace_main_score_case_ids = {
        trace["case_id"]
        for trace in evaluation_summary.get("traces", [])
        if trace.get("subset_flags", {}).get("main_score") is True
    }
    trace_diagnostic_case_ids = {
        trace["case_id"]
        for trace in evaluation_summary.get("traces", [])
        if trace.get("subset_flags", {}).get("diagnostic") is True
    }
    taxonomy_main_score_intersection = taxonomy_case_ids & manifest_main_score_case_ids
    taxonomy_diagnostic_intersection = taxonomy_case_ids & manifest_diagnostic_case_ids
    return {
        "manifest_main_score_case_ids_count": len(manifest_main_score_case_ids),
        "manifest_diagnostic_case_ids_count": len(manifest_diagnostic_case_ids),
        "taxonomy_gold_case_ids_count": len(taxonomy_case_ids),
        "taxonomy_main_score_intersection_count": len(taxonomy_main_score_intersection),
        "taxonomy_diagnostic_intersection_count": len(taxonomy_diagnostic_intersection),
        "evaluator_main_score_subset_count": len(trace_main_score_case_ids),
        "evaluator_diagnostic_subset_count": len(trace_diagnostic_case_ids),
        "evaluator_main_score_equals_manifest_main_score_case_ids": (
            trace_main_score_case_ids == manifest_main_score_case_ids
        ),
        "evaluator_main_score_equals_taxonomy_manifest_intersection": (
            trace_main_score_case_ids == taxonomy_main_score_intersection
        ),
        "evaluator_diagnostic_equals_taxonomy_manifest_intersection": (
            trace_diagnostic_case_ids == taxonomy_diagnostic_intersection
        ),
        "report_subset_labels": {
            "main_score": "taxonomy_main_score",
            "diagnostic": "taxonomy_diagnostic",
        },
        "interpretation": (
            "The taxonomy evaluator subset named main_score is the intersection of T-box taxonomy gold rows "
            "with the core manifest main_score_case_ids, not the full manifest-level main_score_case_ids list."
        ),
    }


def value_delta_display_metrics(subsets: dict[str, Any]) -> dict[str, Any]:
    display = {}
    for subset_name, subset in sorted(subsets.items()):
        metrics = subset.get("metrics", {})
        f1_metric = metrics.get("tbox_patch_value_delta_f1_when_applicable", {})
        display[subset_name] = {
            "value_delta_f1_display_rate": (
                None if f1_metric.get("applicability_coverage") == 0 else f1_metric.get("rate")
            ),
            "value_delta_f1_display": (
                "n/a" if f1_metric.get("applicability_coverage") == 0 else f'{f1_metric.get("rate", 0):.3f}'
            ),
            "value_delta_f1_reason": (
                "n/a because gold value-delta applicability is zero"
                if f1_metric.get("applicability_coverage") == 0
                else "computed on applicable gold value-delta rows"
            ),
            "value_delta_false_positive_rate": metrics.get(
                "tbox_patch_value_delta_claimed_when_gold_absent_rate", {}
            ).get("rate"),
            "value_delta_under_specification_rate": metrics.get(
                "tbox_patch_family_only_when_value_delta_gold_present_rate", {}
            ).get("rate"),
        }
    return display


def enrich_report(args: argparse.Namespace) -> None:
    report = load_json(args.report_json)
    manifest = load_json(args.core_manifest)
    gold_rows = load_jsonl(args.gold_jsonl)
    gold_by_case = {row["case_id"]: row for row in gold_rows}
    annotations = manifest.get("case_annotations", {})

    evaluation_by_matrix_id = {
        path.parent.name: load_json(path)
        for path in args.run_dir.glob("matrices/*/tbox_taxonomy_patch_evaluation_summary.json")
    }
    predictions_by_matrix_id = {
        path.parent.name: load_jsonl(path)
        for path in args.run_dir.glob("matrices/*/t_box_taxonomy_patch_proposals.jsonl")
    }

    if not evaluation_by_matrix_id or not predictions_by_matrix_id:
        raise FileNotFoundError(f"no taxonomy-patch matrices found under {args.run_dir}")

    first_summary = next(iter(evaluation_by_matrix_id.values()))
    report["subset_audit"] = build_subset_audit(gold_rows, first_summary, manifest)

    for matrix in report["matrices"]:
        matrix_id = matrix["matrix_id"]
        evaluation_summary = evaluation_by_matrix_id[matrix_id]
        predictions = predictions_by_matrix_id[matrix_id]
        matrix["confusion_matrices"] = build_confusion_matrices(gold_by_case, predictions)
        matrix["out_of_current_gold_operation_false_positive_rates"] = build_out_of_current_gold_operation_fp_rates(
            gold_rows, predictions
        )
        matrix["macro_averages"] = build_macro_averages(evaluation_summary["traces"], gold_by_case, annotations)
        matrix["value_delta_display_metrics"] = value_delta_display_metrics(evaluation_summary["subsets"])
        matrix["report_subset_labels"] = report["subset_audit"]["report_subset_labels"]

    report["report_enrichment"] = {
        "script": "scripts/enrich_tbox_taxonomy_patch_core_report.py",
        "inputs": {
            "report_json": str(args.report_json),
            "run_dir": str(args.run_dir),
            "gold_jsonl": str(args.gold_jsonl),
            "core_manifest": str(args.core_manifest),
        },
        "model_inference": "not_run",
        "prompt_changes": "none",
        "gold_extraction_changes": "none",
    }
    write_json(args.report_json, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/analysis/tbox_taxonomy_patch_core_results.json"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("reports/prompt_dev/evaluation_prompt_dev_v5_tbox_taxonomy_patch_core_tbox_all_zero_shot"),
    )
    parser.add_argument(
        "--gold-jsonl",
        type=Path,
        default=Path("reports/gold/tbox_taxonomy_patch_gold_core_v1.jsonl"),
    )
    parser.add_argument(
        "--core-manifest",
        type=Path,
        default=Path("reports/benchmark_selection/core_v1_seed_13.json"),
    )
    return parser.parse_args()


def main() -> None:
    enrich_report(parse_args())


if __name__ == "__main__":
    main()
