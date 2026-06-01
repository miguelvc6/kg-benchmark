#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CASE_ID_RE = re.compile(r'"id"\s*:\s*"([^"]+)"')

DEFAULT_STRATUM_ORDER = [
    "TypeB_LOCAL_TEXT_CONFIRMED",
    "TypeB_LOCAL_TEXT_DERIVED",
    "TypeB_LOCAL_SELECTION_CONFIRMED",
    "TypeB_LOCAL_FOCUS_QID",
    "TypeC_EXTERNAL_BY_ELIMINATION_QID_TRUTH",
    "TypeC_EXTERNAL_BY_ELIMINATION_LITERAL_TRUTH",
    "TypeC_UNKNOWN_SELECTION_AMBIGUOUS",
    "TypeC_UNKNOWN_MULTIPLICITY_ARTIFACT",
    "TypeC_UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED",
    "TypeC_UNKNOWN_BAD_TARGET_OR_CONTEXT",
    "TypeC_UNKNOWN_FOCUS_QID_DOMAIN_REASONING",
    "TypeC_UNKNOWN_OR_SPARSE_DIAGNOSTIC",
    "TypeA_DELETE_AMBIGUOUS",
    "TypeA_TARGET_REQUIRED_CLAIM",
    "TypeA_FORMAT_NORMALIZATION",
    "TypeA_FORMAT_VALUE_PRUNING",
    "TypeA_REJECTION_FORMAT_INVALID",
    "TypeA_SELF_LINK_REJECTION",
    "TypeA_SET_MEMBERSHIP_REJECTION",
    "TypeA_MULTIPLICITY_NORMALIZATION",
    "TBOX_SCHEMA_UPDATE",
    "TBOX_COINCIDENTAL_SCHEMA_CHANGE",
    "TBOX_DIRECTIONAL_RELAXATION_OR_RESTRICTION",
    "TBOX_UNKNOWN_TBOX_CAUSALITY",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate human-reviewable Phase D manual-audit case cards by stratum."
    )
    parser.add_argument("--audit-csv", default="reports/manual_audit/audit_phase_d_v1_seed_13.csv")
    parser.add_argument("--classified-benchmark", default="data/04_classified_benchmark.jsonl")
    parser.add_argument("--output-dir", default="reports/manual_audit/case_cards_by_stratum")
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument(
        "--max-list-items",
        type=int,
        default=24,
        help="Maximum values to display from long lists before adding an omitted-count marker.",
    )
    return parser.parse_args()


def short(value: Any, *, max_chars: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + f"... [truncated {len(text) - max_chars + 20} chars]"


def md(value: Any) -> str:
    return short(value).replace("|", "\\|")


def compact(value: Any, *, max_items: int = 24, max_string: int = 500) -> Any:
    if isinstance(value, dict):
        return {str(k): compact(v, max_items=max_items, max_string=max_string) for k, v in value.items()}
    if isinstance(value, list):
        result = [compact(v, max_items=max_items, max_string=max_string) for v in value[:max_items]]
        if len(value) > max_items:
            result.append(f"... omitted {len(value) - max_items} items")
        return result
    if isinstance(value, str) and len(value) > max_string:
        return value[: max_string - 32] + f"... [truncated {len(value) - max_string + 32} chars]"
    return value


def json_block(value: Any, *, max_items: int = 24) -> str:
    if value in (None, "", [], {}):
        return "_empty_"
    return "```json\n" + json.dumps(compact(value, max_items=max_items), ensure_ascii=False, indent=2, sort_keys=True) + "\n```"


def trace(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("classification", {}).get("decision_trace")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def trace_step(record: dict[str, Any], name: str) -> dict[str, Any] | None:
    for item in trace(record):
        if item.get("step") == name:
            return item
    return None


def listify(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def scalar_values(value: Any) -> list[str]:
    result: list[str] = []
    for item in listify(value):
        if isinstance(item, dict):
            for key in ("id", "qid", "value", "text", "amount", "time"):
                if key in item:
                    result.append(str(item[key]))
                    break
            else:
                result.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        else:
            result.append(str(item))
    return result


def value_change_summary(record: dict[str, Any]) -> dict[str, Any]:
    diagnostics = record.get("classification", {}).get("diagnostics")
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("value_change_summary"), dict):
        return diagnostics["value_change_summary"]
    rt = record.get("repair_target") if isinstance(record.get("repair_target"), dict) else {}
    old_values = scalar_values(rt.get("old_value"))
    new_values = scalar_values(rt.get("new_value"))
    deleted_values = scalar_values(rt.get("deleted_value"))
    old_counter = Counter(old_values)
    new_counter = Counter(new_values)
    return {
        "action": rt.get("action"),
        "kind": rt.get("kind"),
        "old_value": old_values,
        "new_value": new_values,
        "deleted_value": deleted_values,
        "old_count": len(old_values),
        "new_count": len(new_values),
        "old_unique": sorted(old_counter),
        "new_unique": sorted(new_counter),
        "added_unique_values": sorted(set(new_counter) - set(old_counter)),
        "removed_unique_values": sorted(set(old_counter) - set(new_counter)),
        "value_multiplicity_changes": {
            value: {"old": old_counter.get(value, 0), "new": new_counter.get(value, 0)}
            for value in sorted(set(old_counter) | set(new_counter))
            if old_counter.get(value, 0) != new_counter.get(value, 0)
        },
        "normalized_unique_values_unchanged": bool(old_values or new_values)
        and set(old_counter) == set(new_counter),
        "exact_value_lists_unchanged": old_values == new_values,
    }


def repair_target_details(record: dict[str, Any]) -> dict[str, Any]:
    rt = record.get("repair_target") if isinstance(record.get("repair_target"), dict) else {}
    skip = {
        "constraint_delta",
        "signature_before_raw",
        "signature_after_raw",
        "old_constraints",
        "new_constraints",
        "signature_before",
        "signature_after",
    }
    details = {key: value for key, value in rt.items() if key not in skip}
    if rt.get("kind") == "A_BOX":
        details["value_change_summary"] = value_change_summary(record)
    return details


def classification_target_summary(record: dict[str, Any]) -> dict[str, Any]:
    diagnostics = record.get("classification", {}).get("diagnostics")
    target = {}
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("classification_target_tokens"), dict):
        target = dict(diagnostics["classification_target_tokens"])
    summary = value_change_summary(record)
    return {
        "semantic_action": summary.get("semantic_action"),
        "added_unique_values": summary.get("added_unique_values"),
        "removed_unique_values": summary.get("removed_unique_values"),
        "retained_unique_values": summary.get("retained_unique_values"),
        "removed_target_tokens": summary.get("removed_unique_values"),
        "retained_support_tokens": summary.get("retained_unique_values"),
        "classification_target_tokens": target.get("tokens", []),
        "classification_target_role": target.get("role", ""),
        "classification_target_reason": target.get("reason", ""),
        "old_changed_value": target.get("old_changed_value"),
        "new_changed_value": target.get("new_changed_value"),
    }


def classifier_rule_summary(record: dict[str, Any]) -> dict[str, Any]:
    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    return {
        "constraint_family": classification.get("constraint_family"),
        "classification_rule_family": classification.get("classification_rule_family"),
        "classification_rule_subfamily": classification.get("classification_rule_subfamily"),
        "decision_constraint_type_qid": classification.get("decision_constraint_type_qid"),
        "decision_constraint_type_label": classification.get("decision_constraint_type_label"),
        "decision_constraint_source": classification.get("decision_constraint_source"),
    }


def tbox_causality_summary(record: dict[str, Any]) -> dict[str, Any]:
    step = trace_step(record, "tbox_causality") or trace_step(record, "causality_filter") or {}
    keys = [
        "selected_violation_name",
        "candidate_violation_names",
        "candidate_violation_mappings_preview",
        "mapped_report_constraint_qid",
        "mapped_report_constraint_label",
        "mapped_report_family",
        "mapped_violation_constraint_qid",
        "mapped_violation_constraint_label",
        "mapped_violation_family",
        "mapped_violation_confidence",
        "changed_constraint_qids_from_entries",
        "changed_constraint_qids_from_qualifier_changes",
        "changed_constraint_qids_all",
        "target_constraint_qid",
        "target_constraint_label",
        "target_constraint_selection_reason",
        "target_constraint_selection_confidence",
        "target_constraint_is_changed",
        "target_constraint_is_related_family",
        "value_overlap_with_report_qids",
        "property_overlap_with_report_pids",
        "language_overlap_with_report_langs",
        "scope_overlap_with_report_values",
        "compatible_value_overlap_with_report_qids",
        "compatible_property_overlap_with_report_pids",
        "compatible_language_overlap_with_report_langs",
        "compatible_scope_overlap_with_report_values",
        "incompatible_overlap_ignored",
        "value_specific_without_overlap",
        "compatible_overlap_used",
        "compatible_overlap_reason",
        "causality_match_level",
        "causality_match_reason",
        "semantic_changed_qualifier_properties",
        "ignored_changed_qualifier_properties",
        "semantic_added_values",
        "semantic_removed_values",
        "ignored_added_values",
        "ignored_removed_values",
        "semantic_added_value_count",
        "semantic_removed_value_count",
        "ignored_value_count",
        "qualifier_filter_reason",
        "set_semantics",
        "set_operation",
        "polarity",
        "polarity_basis",
        "directional_subtype_basis",
        "directional_subtype_precise",
        "analysis_slice_precise",
    ]
    return {key: step.get(key) for key in keys if key in step}


def tbox_compact_diff_summary(record: dict[str, Any]) -> dict[str, Any]:
    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    diagnostics = classification.get("diagnostics") if isinstance(classification.get("diagnostics"), dict) else {}
    summary = diagnostics.get("tbox_diff_summary") if isinstance(diagnostics.get("tbox_diff_summary"), dict) else {}
    if summary:
        return summary
    step = trace_step(record, "tbox_causality") or {}
    return {
        "lean_stage4_pruned_full_signatures": True,
        "source": "classification.decision_trace.tbox_causality",
        "selected_violation_name": step.get("selected_violation_name"),
        "target_constraint_qid": step.get("target_constraint_qid"),
        "target_constraint_label": step.get("target_constraint_label"),
        "target_constraint_is_changed": step.get("target_constraint_is_changed"),
        "target_constraint_is_related_family": step.get("target_constraint_is_related_family"),
        "mapped_report_constraint_qid": step.get("mapped_report_constraint_qid"),
        "mapped_report_constraint_label": step.get("mapped_report_constraint_label"),
        "mapped_report_family": step.get("mapped_report_family"),
        "changed_constraint_qids_from_entries": step.get("changed_constraint_qids_from_entries"),
        "changed_constraint_qids_from_qualifier_changes": step.get("changed_constraint_qids_from_qualifier_changes"),
        "changed_constraint_qids_all": step.get("changed_constraint_qids_all"),
        "changed_qualifier_properties": step.get("changed_qualifier_properties"),
        "added_values": step.get("added_values"),
        "removed_values": step.get("removed_values"),
        "added_value_count": step.get("added_value_count"),
        "removed_value_count": step.get("removed_value_count"),
        "set_semantics": step.get("set_semantics"),
        "set_operation": step.get("set_operation"),
        "polarity": step.get("polarity"),
        "polarity_basis": step.get("polarity_basis"),
        "directional_subtype_precise": step.get("directional_subtype_precise"),
        "analysis_slice_precise": step.get("analysis_slice_precise"),
        "semantic_changed_qualifier_properties": step.get("semantic_changed_qualifier_properties"),
        "ignored_changed_qualifier_properties": step.get("ignored_changed_qualifier_properties"),
        "semantic_added_values": step.get("semantic_added_values"),
        "semantic_removed_values": step.get("semantic_removed_values"),
        "ignored_added_values": step.get("ignored_added_values"),
        "ignored_removed_values": step.get("ignored_removed_values"),
        "ignored_value_count": step.get("ignored_value_count"),
        "qualifier_filter_reason": step.get("qualifier_filter_reason"),
        "compatible_overlap_used": step.get("compatible_overlap_used"),
        "compatible_overlap_reason": step.get("compatible_overlap_reason"),
        "incompatible_overlap_ignored": step.get("incompatible_overlap_ignored"),
        "value_specific_without_overlap": step.get("value_specific_without_overlap"),
    }


def local_evidence_summary(record: dict[str, Any]) -> dict[str, Any]:
    step = trace_step(record, "local_availability") or {}
    evidence = step.get("evidence") if isinstance(step.get("evidence"), dict) else {}
    synthetic = step.get("synthetic") if isinstance(step.get("synthetic"), dict) else {}
    diagnostics = record.get("classification", {}).get("diagnostics")
    truth_tokens = diagnostics.get("truth_tokens") if isinstance(diagnostics, dict) else []
    match_tokens = [
        str(match.get("token"))
        for match in evidence.get("matches", [])
        if isinstance(match, dict) and match.get("token") is not None
    ]
    summary = value_change_summary(record)
    retained = set(map(str, summary.get("retained_unique_values") or []))
    retained_matches = [
        match
        for match in evidence.get("matches", [])
        if isinstance(match, dict) and str(match.get("token")) in retained
    ]
    return {
        "local_availability_result": step.get("result"),
        "truth_tokens": truth_tokens,
        "matched": evidence.get("matched"),
        "needed": evidence.get("needed"),
        "found": evidence.get("found"),
        "matches": evidence.get("matches"),
        "sources_used": evidence.get("sources_used"),
        "used_literal_substring": evidence.get("used_literal_substring"),
        "local_ids_count": evidence.get("local_ids_count"),
        "truth_tokens_in_recorded_matches": sorted(set(map(str, truth_tokens)) & set(match_tokens)),
        "local_support_for_retained_value": retained_matches,
        "synthetic_pre_repair": synthetic,
    }


def violation_summary(record: dict[str, Any]) -> dict[str, Any]:
    vc = record.get("violation_context") if isinstance(record.get("violation_context"), dict) else {}
    preferred = [
        "report_violation_type_qids",
        "violation_name",
        "constraint_qid",
        "constraint_type_qid",
        "property",
        "qid",
        "value",
        "message",
        "report",
        "constraint_scope",
    ]
    out = {key: vc.get(key) for key in preferred if key in vc}
    for key, value in vc.items():
        if key not in out and len(out) < 18:
            out[key] = value
    return out


def labels_summary(record: dict[str, Any]) -> dict[str, Any]:
    labels = record.get("labels_en") if isinstance(record.get("labels_en"), dict) else {}
    return {key: labels.get(key) for key in sorted(labels)[:40]}


def qualifier_map(constraint: dict[str, Any]) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for qualifier in constraint.get("qualifiers", []):
        if not isinstance(qualifier, dict):
            continue
        property_id = str(qualifier.get("property_id") or "UNKNOWN")
        for value in qualifier.get("values", []):
            result[property_id][str(value)] += 1
    return result


def constraint_identity(constraint: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(constraint.get("constraint_qid") or ""),
        str(constraint.get("snaktype") or ""),
        str(constraint.get("rank") or ""),
        len(constraint.get("qualifiers", []) if isinstance(constraint.get("qualifiers"), list) else []),
    )


def tbox_constraint_diff(record: dict[str, Any]) -> dict[str, Any]:
    rt = record.get("repair_target") if isinstance(record.get("repair_target"), dict) else {}
    delta = rt.get("constraint_delta") if isinstance(rt.get("constraint_delta"), dict) else {}
    before = delta.get("signature_before") if isinstance(delta.get("signature_before"), list) else []
    after = delta.get("signature_after") if isinstance(delta.get("signature_after"), list) else []
    before_json = Counter(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in before)
    after_json = Counter(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in after)

    added_entries = []
    removed_entries = []
    for payload, count in sorted((after_json - before_json).items()):
        item = json.loads(payload)
        item["_count"] = count
        added_entries.append(item)
    for payload, count in sorted((before_json - after_json).items()):
        item = json.loads(payload)
        item["_count"] = count
        removed_entries.append(item)

    before_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    after_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in before:
        if isinstance(item, dict):
            before_by_qid[str(item.get("constraint_qid"))].append(item)
    for item in after:
        if isinstance(item, dict):
            after_by_qid[str(item.get("constraint_qid"))].append(item)

    qualifier_value_changes = []
    for qid in sorted(set(before_by_qid) | set(after_by_qid)):
        before_items = before_by_qid.get(qid, [])
        after_items = after_by_qid.get(qid, [])
        for idx in range(min(len(before_items), len(after_items))):
            old_qualifiers = qualifier_map(before_items[idx])
            new_qualifiers = qualifier_map(after_items[idx])
            for prop in sorted(set(old_qualifiers) | set(new_qualifiers)):
                added = sorted((new_qualifiers[prop] - old_qualifiers[prop]).elements())
                removed = sorted((old_qualifiers[prop] - new_qualifiers[prop]).elements())
                if added or removed:
                    qualifier_value_changes.append(
                        {
                            "constraint_qid": qid,
                            "same_qid_index": idx,
                            "qualifier_property": prop,
                            "added_values": added,
                            "removed_values": removed,
                        }
                    )

    return {
        "property_revision_prev": rt.get("property_revision_prev"),
        "property_revision_id": rt.get("property_revision_id"),
        "author": rt.get("author"),
        "hash_before": delta.get("hash_before"),
        "hash_after": delta.get("hash_after"),
        "changed_constraint_types": delta.get("changed_constraint_types"),
        "before_constraint_count": len(before),
        "after_constraint_count": len(after),
        "added_constraint_entries": added_entries,
        "removed_constraint_entries": removed_entries,
        "qualifier_value_changes": qualifier_value_changes,
        "rule_summaries_en": delta.get("rule_summaries_en"),
        "constraints_readable_en": delta.get("constraints_readable_en"),
    }


def annotation_focus(row: dict[str, str]) -> list[str]:
    cls = row["class"]
    subtype = row["subtype"]
    if cls == "TypeC":
        return [
            "Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.",
            "Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.",
            "If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.",
        ]
    if cls == "TypeB":
        return [
            "Decide whether the recorded local match actually supports the historical target.",
            "For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.",
            "For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.",
            "For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.",
            "For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.",
        ]
    if cls == "TypeA" and subtype == "DELETE_AMBIGUOUS":
        return [
            "Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.",
            "This stratum should normally remain diagnostic unless audit shows a clean split.",
        ]
    if cls == "TypeA" and subtype == "TARGET_REQUIRED_CLAIM":
        return [
            "Confirm that the report is a target-required-claim violation and that the required target is the focus QID.",
            "This should be rule-deterministic, not local-evidence TypeB.",
        ]
    if cls == "TypeA" and subtype == "SET_MEMBERSHIP_REJECTION":
        return [
            "Check whether the removed value is directly ruled out by a set-membership constraint.",
            "If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.",
        ]
    if cls == "TypeA":
        return [
            "Check whether the rule or format alone justifies the repair.",
            "If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.",
        ]
    if cls == "T_BOX":
        return [
            "Check whether the changed constraint family matches the reported violation family.",
            "Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.",
            "For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.",
        ]
    return ["Fill the class-specific judgment and core_recommendation."]


def load_audit_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_stage4_records(path: Path, case_ids: set[str], *, progress_every: int) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = CASE_ID_RE.search(line)
            if not match:
                continue
            case_id = match.group(1)
            if case_id not in case_ids:
                continue
            records[case_id] = json.loads(line)
            if len(records) == len(case_ids):
                break
            if progress_every > 0 and line_number % progress_every == 0:
                print(f"[progress] scanned={line_number:,} found={len(records):,}/{len(case_ids):,}")
    missing = sorted(case_ids - set(records))
    if missing:
        raise ValueError(f"Missing {len(missing)} audit records in Stage 4: {missing[:5]}")
    return records


def write_stratum_file(
    *,
    path: Path,
    stratum: str,
    rows: list[dict[str, str]],
    records: dict[str, dict[str, Any]],
    max_list_items: int,
) -> None:
    lines = [
        f"# {stratum}",
        "",
        f"Cases: {len(rows)}",
        "",
        "Use this file for evidence review. Enter final annotations in the CSV copy, not here.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        record = records[row["case_id"]]
        classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
        lines.extend(
            [
                f"## {index:03d}. `{row['case_id']}`",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| qid | {md(row.get('qid'))} |",
                f"| property | {md(row.get('property'))} |",
                f"| track | {md(row.get('track'))} |",
                f"| class / subtype / confidence | {md(row.get('class'))} / {md(row.get('subtype'))} / {md(row.get('confidence'))} |",
                f"| main_score / diagnostic_only | {md(row.get('main_score'))} / {md(row.get('diagnostic_only'))} |",
                f"| analysis_slice | {md(row.get('analysis_slice'))} |",
                f"| popularity_bucket | {md(row.get('popularity_bucket'))} |",
                f"| constraint_family | {md(row.get('constraint_family'))} |",
                f"| classification_rule_family | {md(row.get('classification_rule_family'))} |",
                f"| classification_rule_subfamily | {md(row.get('classification_rule_subfamily'))} |",
                f"| decision_constraint_type | {md(row.get('decision_constraint_type_qid'))} {md(row.get('decision_constraint_type_label'))} |",
                f"| group_key | {md(row.get('group_key'))} |",
                f"| tbox_revision_key | {md(row.get('tbox_revision_key'))} |",
                "",
                "### Annotation Focus",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in annotation_focus(row))
        lines.extend(
            [
                "",
                "### Classifier Summary",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| truth_source | {md(row.get('truth_source'))} |",
                f"| truth_token_kind | {md(row.get('truth_token_kind'))} |",
                f"| truth_tokens_preview | {md(row.get('truth_tokens_preview'))} |",
                f"| classification_target_tokens | {md(json.dumps(classification_target_summary(record).get('classification_target_tokens'), ensure_ascii=False))} |",
                f"| classification_target_reason | {md(classification_target_summary(record).get('classification_target_reason'))} |",
                f"| decision_branch | {md(row.get('decision_branch'))} |",
                f"| rationale | {md(classification.get('rationale'))} |",
                f"| local_match_kind | {md(row.get('local_match_kind'))} |",
                f"| local_match_source | {md(row.get('local_match_source'))} |",
                "",
                "### What Changed",
                "",
                "#### Delta Summary",
                "",
                json_block(classification_target_summary(record), max_items=max_list_items),
                "",
                "#### Classifier Rule Metadata",
                "",
                json_block(classifier_rule_summary(record), max_items=max_list_items),
                "",
                "#### Repair Target",
                "",
                json_block(repair_target_details(record), max_items=max_list_items),
                "",
                "### Violation Context",
                "",
                json_block(violation_summary(record), max_items=max_list_items),
                "",
                "### Local Evidence",
                "",
                json_block(local_evidence_summary(record), max_items=max_list_items),
                "",
                "### Labels / Human-Readable Context",
                "",
                json_block(labels_summary(record), max_items=max_list_items),
                "",
                "### Constraint Types",
                "",
                json_block(classification.get("constraint_types"), max_items=max_list_items),
                "",
            ]
        )
        if row.get("track") == "T_BOX":
            lines.extend(
                [
                    "### T-box Causality",
                    "",
                    "_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._",
                    "",
                    json_block(tbox_causality_summary(record), max_items=max_list_items),
                    "",
                    "### T-box Compact Diff Summary",
                    "",
                    json_block(tbox_compact_diff_summary(record), max_items=max_list_items),
                    "",
                    "### T-box Constraint Diff",
                    "",
                    "_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._",
                    "",
                    json_block(tbox_constraint_diff(record), max_items=max_list_items),
                    "",
                ]
            )
        lines.extend(
            [
                "### Decision Trace",
                "",
                json_block(trace(record), max_items=max_list_items),
                "",
                "---",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    audit_csv = Path(args.audit_csv)
    benchmark = Path(args.classified_benchmark)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.md"):
        old_file.unlink()

    rows = load_audit_rows(audit_csv)
    records = load_stage4_records(
        benchmark,
        {row["case_id"] for row in rows},
        progress_every=args.progress_every,
    )
    order_index = {stratum: index for index, stratum in enumerate(DEFAULT_STRATUM_ORDER)}
    rows.sort(key=lambda row: (order_index.get(row["selection_stratum"], 999), row["case_id"]))

    rows_by_stratum: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_stratum[row["selection_stratum"]].append(row)

    written: list[tuple[int, str, int, Path]] = []
    for order, stratum in enumerate(DEFAULT_STRATUM_ORDER, start=1):
        stratum_rows = rows_by_stratum.get(stratum, [])
        if not stratum_rows:
            continue
        path = output_dir / f"{order:02d}_{stratum}.md"
        write_stratum_file(
            path=path,
            stratum=stratum,
            rows=stratum_rows,
            records=records,
            max_list_items=args.max_list_items,
        )
        written.append((order, stratum, len(stratum_rows), path))

    index_lines = [
        "# Phase D Manual Audit Case Cards by Stratum",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
        "",
        f"Audit CSV: `{audit_csv}`",
        f"Stage 4 benchmark: `{benchmark}`",
        "",
        "Open one stratum file at a time. Enter final annotations in the CSV copy, not in these Markdown files.",
        "",
        "For T-box cards, prefer the compact diff summary and causality block. Lean Stage 4 may prune full constraint signatures; public directional subtypes are coarse, while `directional_subtype_precise` carries polarity-specific semantics.",
        "",
        "| Order | Stratum | Cases | File | Size |",
        "|---:|---|---:|---|---:|",
    ]
    for order, stratum, count, path in written:
        index_lines.append(f"| {order} | `{stratum}` | {count} | [{path.name}]({path.name}) | {path.stat().st_size:,} bytes |")
    readme = output_dir / "README.md"
    readme.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {len(written)} stratum files to {output_dir}")
    print(f"[done] wrote {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
