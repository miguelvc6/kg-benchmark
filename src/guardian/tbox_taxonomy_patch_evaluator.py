from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import PatchValidationError
from .tbox_taxonomy_patch_parser import NormalizedTBoxTaxonomyPatch, normalize_tbox_taxonomy_patch


NEW_TBOX_PATCH_METRICS = [
    "tbox_patch_parse_rate",
    "tbox_patch_contract_valid_rate",
    "tbox_patch_parse_error_rate",
    "tbox_patch_target_pid_match_rate",
    "tbox_patch_primary_constraint_family_match_rate",
    "tbox_patch_any_changed_family_hit_rate",
    "tbox_patch_constraint_family_precision",
    "tbox_patch_constraint_family_recall",
    "tbox_patch_constraint_family_f1",
    "tbox_patch_schema_decision_match_rate",
    "tbox_patch_no_causal_schema_repair_match_rate",
    "tbox_patch_unclear_schema_evidence_match_rate",
    "tbox_patch_repair_op_exact_match_rate",
    "tbox_patch_taxonomy_code_exact_match_rate",
    "tbox_patch_repair_op_precision",
    "tbox_patch_repair_op_recall",
    "tbox_patch_repair_op_f1",
    "tbox_patch_qualifier_property_match_rate",
    "tbox_patch_added_values_precision",
    "tbox_patch_added_values_recall",
    "tbox_patch_added_values_f1",
    "tbox_patch_removed_values_precision",
    "tbox_patch_removed_values_recall",
    "tbox_patch_removed_values_f1",
    "tbox_patch_value_delta_f1_when_applicable",
    "tbox_patch_evidence_level_exact_match_rate",
    "tbox_patch_value_delta_claimed_when_gold_absent_rate",
    "tbox_patch_family_only_when_value_delta_gold_present_rate",
    "tbox_patch_family_level_success",
    "tbox_patch_decision_level_success",
    "tbox_patch_taxonomy_level_success",
    "tbox_patch_value_delta_success",
]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def evaluate_tbox_taxonomy_patch_predictions(
    *,
    gold_rows: Iterable[dict[str, Any]],
    prediction_rows: Iterable[dict[str, Any]],
    case_annotations: dict[str, dict[str, Any]] | None = None,
    constraint_type_qids: Iterable[str] | None = None,
) -> dict[str, Any]:
    case_annotations = case_annotations or {}
    gold_by_case: dict[str, NormalizedTBoxTaxonomyPatch] = {}
    for row in gold_rows:
        normalized = normalize_tbox_taxonomy_patch(row, constraint_type_qids=constraint_type_qids)
        gold_by_case[normalized.case_id] = normalized

    prediction_by_case: dict[str, dict[str, Any]] = {}
    for row in prediction_rows:
        case_id = row.get("case_id") if isinstance(row, dict) else None
        if isinstance(case_id, str) and case_id.strip():
            prediction_by_case[case_id.strip()] = row

    traces = [
        _evaluate_case(
            case_id=case_id,
            gold=gold,
            raw_prediction=prediction_by_case.get(case_id),
            constraint_type_qids=constraint_type_qids,
            annotation=case_annotations.get(case_id, {}),
        )
        for case_id, gold in sorted(gold_by_case.items())
    ]
    return {
        "metric_family": "tbox_taxonomy_patch_v1",
        "strict_signature_metrics_role": "diagnostic_only",
        "total_tbox_rows": len(traces),
        "traces": traces,
        "subsets": {
            "all_core": _summarize_subset(traces),
            "main_score": _summarize_subset([trace for trace in traces if trace["subset_flags"]["main_score"]]),
            "diagnostic": _summarize_subset([trace for trace in traces if trace["subset_flags"]["diagnostic"]]),
        },
    }


def evaluate_tbox_taxonomy_patch_files(
    *,
    gold_jsonl: str | Path,
    predictions_jsonl: str | Path,
    case_annotations: dict[str, dict[str, Any]] | None = None,
    constraint_type_qids: Iterable[str] | None = None,
) -> dict[str, Any]:
    return evaluate_tbox_taxonomy_patch_predictions(
        gold_rows=load_jsonl(gold_jsonl),
        prediction_rows=load_jsonl(predictions_jsonl),
        case_annotations=case_annotations,
        constraint_type_qids=constraint_type_qids,
    )


def _evaluate_case(
    *,
    case_id: str,
    gold: NormalizedTBoxTaxonomyPatch,
    raw_prediction: dict[str, Any] | None,
    constraint_type_qids: Iterable[str] | None,
    annotation: dict[str, Any],
) -> dict[str, Any]:
    parse_error = None
    prediction: NormalizedTBoxTaxonomyPatch | None = None
    if raw_prediction is None:
        parse_error = "missing_prediction"
    else:
        try:
            prediction = normalize_tbox_taxonomy_patch(raw_prediction, constraint_type_qids=constraint_type_qids)
        except PatchValidationError as exc:
            parse_error = exc.code

    row = _row_metrics(gold, prediction)
    return {
        "case_id": case_id,
        "parsed": prediction is not None,
        "contract_valid": prediction is not None,
        "parse_error": parse_error,
        "gold_schema_decision": gold.schema_decision,
        "prediction_schema_decision": prediction.schema_decision if prediction else None,
        "metrics": row,
        "metric_detail": _metric_detail(gold, prediction),
        "subset_flags": {
            "main_score": bool(annotation.get("main_score", True) and not annotation.get("diagnostic_only", False)),
            "diagnostic": bool(annotation.get("diagnostic_only", False) or annotation.get("main_score") is False),
        },
    }


def _row_metrics(
    gold: NormalizedTBoxTaxonomyPatch,
    prediction: NormalizedTBoxTaxonomyPatch | None,
) -> dict[str, Any]:
    if prediction is None:
        return {
            "target_pid_match": None,
            "primary_constraint_family_match": None,
            "any_changed_family_hit": None,
            "schema_decision_match": None,
            "no_causal_schema_repair_match": None,
            "unclear_schema_evidence_match": None,
            "repair_op_exact_match": None,
            "taxonomy_code_exact_match": None,
            "qualifier_property_match": None,
            "evidence_level_exact_match": None,
            "value_delta_claimed_when_gold_absent": None,
            "family_only_when_value_delta_gold_present": None,
            "family_level_success": None,
            "decision_level_success": None,
            "taxonomy_level_success": None,
            "value_delta_success": None,
            "gold_has_value_delta": _has_value_delta(gold),
        }

    gold_families = _constraint_family_counter(gold)
    pred_families = _constraint_family_counter(prediction)
    family_overlap = gold_families & pred_families
    target_pid_match = prediction.target.pid == gold.target.pid
    primary_family_match = prediction.target.constraint_type_qid == gold.target.constraint_type_qid
    any_changed_family_hit = bool(family_overlap)
    schema_decision_match = prediction.schema_decision == gold.schema_decision
    gold_ops = _repair_counter(gold, "repair_op")
    pred_ops = _repair_counter(prediction, "repair_op")
    gold_codes = _repair_counter(gold, "taxonomy_code")
    pred_codes = _repair_counter(prediction, "taxonomy_code")
    op_overlap = gold_ops & pred_ops
    repair_op_exact = gold_ops == pred_ops
    taxonomy_code_exact = gold_codes == pred_codes
    gold_has_value_delta = _has_value_delta(gold)
    pred_has_value_delta = _has_value_delta(prediction)
    family_level_success = bool(target_pid_match and any_changed_family_hit)
    decision_level_success = bool(family_level_success and schema_decision_match)
    if not gold_ops and not pred_ops:
        taxonomy_overlap_success = True
    else:
        taxonomy_overlap_success = bool(op_overlap)
    taxonomy_level_success = bool(decision_level_success and taxonomy_overlap_success)
    value_delta_success = bool(taxonomy_level_success and _value_delta_success(gold, prediction))

    return {
        "target_pid_match": target_pid_match,
        "primary_constraint_family_match": primary_family_match,
        "any_changed_family_hit": any_changed_family_hit,
        "schema_decision_match": schema_decision_match,
        "no_causal_schema_repair_match": (
            schema_decision_match if gold.schema_decision == "NO_CAUSAL_SCHEMA_REPAIR" else None
        ),
        "unclear_schema_evidence_match": (
            schema_decision_match if gold.schema_decision == "UNCLEAR_SCHEMA_EVIDENCE" else None
        ),
        "repair_op_exact_match": repair_op_exact,
        "taxonomy_code_exact_match": taxonomy_code_exact,
        "qualifier_property_match": _qualifier_properties(gold) == _qualifier_properties(prediction)
        if gold_has_value_delta
        else None,
        "evidence_level_exact_match": _repair_counter(gold, "evidence_level") == _repair_counter(prediction, "evidence_level"),
        "value_delta_claimed_when_gold_absent": (pred_has_value_delta if not gold_has_value_delta else None),
        "family_only_when_value_delta_gold_present": (
            all(repair.evidence_level == "FAMILY_ONLY" for repair in prediction.repairs) if gold_has_value_delta else None
        ),
        "family_level_success": family_level_success,
        "decision_level_success": decision_level_success,
        "taxonomy_level_success": taxonomy_level_success,
        "value_delta_success": value_delta_success,
        "gold_has_value_delta": gold_has_value_delta,
    }


def _summarize_subset(traces: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(traces)
    rows = [trace["metrics"] for trace in traces]
    parsed = sum(1 for trace in traces if trace["parsed"])
    parse_errors = total - parsed
    metric_payload = {
        "tbox_patch_parse_rate": _metric(parsed, total, total),
        "tbox_patch_contract_valid_rate": _metric(parsed, total, total),
        "tbox_patch_parse_error_rate": _metric(parse_errors, total, total),
    }

    bool_metric_map = {
        "tbox_patch_target_pid_match_rate": "target_pid_match",
        "tbox_patch_primary_constraint_family_match_rate": "primary_constraint_family_match",
        "tbox_patch_any_changed_family_hit_rate": "any_changed_family_hit",
        "tbox_patch_schema_decision_match_rate": "schema_decision_match",
        "tbox_patch_no_causal_schema_repair_match_rate": "no_causal_schema_repair_match",
        "tbox_patch_unclear_schema_evidence_match_rate": "unclear_schema_evidence_match",
        "tbox_patch_repair_op_exact_match_rate": "repair_op_exact_match",
        "tbox_patch_taxonomy_code_exact_match_rate": "taxonomy_code_exact_match",
        "tbox_patch_qualifier_property_match_rate": "qualifier_property_match",
        "tbox_patch_evidence_level_exact_match_rate": "evidence_level_exact_match",
        "tbox_patch_value_delta_claimed_when_gold_absent_rate": "value_delta_claimed_when_gold_absent",
        "tbox_patch_family_only_when_value_delta_gold_present_rate": "family_only_when_value_delta_gold_present",
        "tbox_patch_family_level_success": "family_level_success",
        "tbox_patch_decision_level_success": "decision_level_success",
        "tbox_patch_taxonomy_level_success": "taxonomy_level_success",
        "tbox_patch_value_delta_success": "value_delta_success",
    }
    for metric_name, row_key in bool_metric_map.items():
        metric_payload[metric_name] = _bool_row_metric(rows, row_key, total)

    counts = _micro_counts(traces)
    metric_payload["tbox_patch_constraint_family_precision"] = _metric(
        counts["family_tp"], counts["family_pred"], total
    )
    metric_payload["tbox_patch_constraint_family_recall"] = _metric(counts["family_tp"], counts["family_gold"], total)
    metric_payload["tbox_patch_constraint_family_f1"] = _f1_metric(
        counts["family_tp"], counts["family_pred"], counts["family_gold"], total
    )
    metric_payload["tbox_patch_repair_op_precision"] = _metric(counts["op_tp"], counts["op_pred"], total)
    metric_payload["tbox_patch_repair_op_recall"] = _metric(counts["op_tp"], counts["op_gold"], total)
    metric_payload["tbox_patch_repair_op_f1"] = _f1_metric(counts["op_tp"], counts["op_pred"], counts["op_gold"], total)
    metric_payload["tbox_patch_added_values_precision"] = _metric(
        counts["added_tp"], counts["added_pred"], total
    )
    metric_payload["tbox_patch_added_values_recall"] = _metric(counts["added_tp"], counts["added_gold"], total)
    metric_payload["tbox_patch_added_values_f1"] = _f1_metric(
        counts["added_tp"], counts["added_pred"], counts["added_gold"], total
    )
    metric_payload["tbox_patch_removed_values_precision"] = _metric(
        counts["removed_tp"], counts["removed_pred"], total
    )
    metric_payload["tbox_patch_removed_values_recall"] = _metric(counts["removed_tp"], counts["removed_gold"], total)
    metric_payload["tbox_patch_removed_values_f1"] = _f1_metric(
        counts["removed_tp"], counts["removed_pred"], counts["removed_gold"], total
    )
    metric_payload["tbox_patch_value_delta_f1_when_applicable"] = _f1_metric(
        counts["value_tp"], counts["value_pred"], counts["value_gold"], total
    )
    return {
        "count": total,
        "metrics": metric_payload,
    }


def _micro_counts(traces: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for trace in traces:
        detail = trace.get("metric_detail", {})
        for key, value in detail.items():
            counts[key] += value
    return counts


def _metric(numerator: int | float, denominator: int, total_tbox_rows: int, *, rate: float | None = None) -> dict[str, Any]:
    computed_rate = rate if rate is not None else (float(numerator) / denominator if denominator else None)
    return {
        "numerator": numerator,
        "applicable_denominator": denominator,
        "total_tbox_rows": total_tbox_rows,
        "applicability_coverage": denominator / total_tbox_rows if total_tbox_rows else None,
        "rate": computed_rate,
    }


def _f1_metric(tp: int, predicted: int, gold: int, total_tbox_rows: int) -> dict[str, Any]:
    precision = tp / predicted if predicted else None
    recall = tp / gold if gold else None
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return _metric(tp, max(predicted + gold, 0), total_tbox_rows, rate=f1)


def _bool_row_metric(rows: list[dict[str, Any]], key: str, total_tbox_rows: int) -> dict[str, Any]:
    applicable = [row[key] for row in rows if row.get(key) is not None]
    numerator = sum(1 for value in applicable if bool(value))
    return _metric(numerator, len(applicable), total_tbox_rows)


def _constraint_family_counter(patch: NormalizedTBoxTaxonomyPatch) -> Counter[str]:
    values = [patch.target.constraint_type_qid]
    values.extend(repair.constraint_type_qid for repair in patch.repairs)
    return Counter(values)


def _repair_counter(patch: NormalizedTBoxTaxonomyPatch, field_name: str) -> Counter[str]:
    return Counter(getattr(repair, field_name) for repair in patch.repairs)


def _qualifier_properties(patch: NormalizedTBoxTaxonomyPatch) -> Counter[str]:
    return Counter(repair.qualifier_property_id for repair in patch.repairs if repair.qualifier_property_id is not None)


def _has_value_delta(patch: NormalizedTBoxTaxonomyPatch) -> bool:
    return any(
        repair.evidence_level == "VALUE_DELTA_VISIBLE" and (repair.added_values or repair.removed_values)
        for repair in patch.repairs
    )


def _value_delta_success(gold: NormalizedTBoxTaxonomyPatch, prediction: NormalizedTBoxTaxonomyPatch) -> bool:
    if not _has_value_delta(gold):
        return True
    return (
        _qualifier_properties(gold) == _qualifier_properties(prediction)
        and _value_counter(gold, "added_values") == _value_counter(prediction, "added_values")
        and _value_counter(gold, "removed_values") == _value_counter(prediction, "removed_values")
    )


def _value_counter(patch: NormalizedTBoxTaxonomyPatch, field_name: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for repair in patch.repairs:
        for value in getattr(repair, field_name):
            counter[json.dumps(value, sort_keys=True)] += 1
    return counter


def _counter_overlap_size(left: Counter[Any], right: Counter[Any]) -> int:
    return sum((left & right).values())


def _metric_detail(gold: NormalizedTBoxTaxonomyPatch, prediction: NormalizedTBoxTaxonomyPatch | None) -> dict[str, int]:
    if prediction is None:
        return {}
    gold_families = _constraint_family_counter(gold)
    pred_families = _constraint_family_counter(prediction)
    gold_ops = _repair_counter(gold, "repair_op")
    pred_ops = _repair_counter(prediction, "repair_op")
    gold_added = _value_counter(gold, "added_values")
    pred_added = _value_counter(prediction, "added_values")
    gold_removed = _value_counter(gold, "removed_values")
    pred_removed = _value_counter(prediction, "removed_values")
    gold_value = gold_added + gold_removed
    pred_value = pred_added + pred_removed
    return {
        "family_tp": _counter_overlap_size(gold_families, pred_families),
        "family_gold": sum(gold_families.values()),
        "family_pred": sum(pred_families.values()),
        "op_tp": _counter_overlap_size(gold_ops, pred_ops),
        "op_gold": sum(gold_ops.values()),
        "op_pred": sum(pred_ops.values()),
        "added_tp": _counter_overlap_size(gold_added, pred_added),
        "added_gold": sum(gold_added.values()),
        "added_pred": sum(pred_added.values()),
        "removed_tp": _counter_overlap_size(gold_removed, pred_removed),
        "removed_gold": sum(gold_removed.values()),
        "removed_pred": sum(pred_removed.values()),
        "value_tp": _counter_overlap_size(gold_value, pred_value),
        "value_gold": sum(gold_value.values()),
        "value_pred": sum(pred_value.values()),
    }
