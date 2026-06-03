from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from classifier import WorldStateStore
from guardian.common import PatchValidationError
from guardian.evaluator import evaluate_benchmark
from guardian.patch_parser import normalize_proposal as normalize_a_box_proposal
from guardian.track_parser import normalize_diagnosis
from lib.benchmark_selection import load_selection_manifest
from lib.repair_state import comparable_atom, derive_value_change_summary, normalize_value_list, pre_repair_target_state
from lib.utils import iter_jsonl, normalize_text

TRACK_LABELS = ("A_BOX", "T_BOX", "AMBIGUOUS")
TRACK_BASELINES = ("majority_track", "always_a_box", "always_t_box", "always_ambiguous")


@dataclass(frozen=True)
class PhaseEOptions:
    classified_benchmark: Path
    world_state: Path
    selection_manifest: Path
    output_dir: Path
    tier: str = "core"
    progress_every: int = 1000


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def selected_records(classified_path: str | Path, selected_case_ids: Iterable[str]) -> list[dict[str, Any]]:
    selected = list(selected_case_ids)
    selected_set = set(selected)
    by_id: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(classified_path):
        if not isinstance(record, dict):
            continue
        case_id = record.get("id")
        if isinstance(case_id, str) and case_id in selected_set:
            by_id[case_id] = record
            if len(by_id) == len(selected_set):
                break
    missing = [case_id for case_id in selected if case_id not in by_id]
    if missing:
        raise ValueError(f"Selected case ids are missing from classified benchmark: {missing[:5]}")
    return [by_id[case_id] for case_id in selected]


def _classification(record: dict[str, Any]) -> dict[str, Any]:
    classification = record.get("classification")
    return classification if isinstance(classification, dict) else {}


def _diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _classification(record).get("diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else {}


def _value_change(record: dict[str, Any]) -> dict[str, Any]:
    existing = _diagnostics(record).get("value_change_summary")
    if isinstance(existing, dict):
        return existing
    return derive_value_change_summary(record).as_dict()


def _old_values(record: dict[str, Any]) -> list[str]:
    summary = _value_change(record)
    old_values = summary.get("old_values")
    if isinstance(old_values, list):
        return [comparable_atom(value) for value in old_values]
    return normalize_value_list(record.get("repair_target", {}).get("old_value"))


def _new_values(record: dict[str, Any]) -> list[str]:
    summary = _value_change(record)
    new_values = summary.get("new_values")
    if isinstance(new_values, list):
        return [comparable_atom(value) for value in new_values]
    repair_target = record.get("repair_target", {})
    return normalize_value_list(repair_target.get("new_value") or repair_target.get("value"))


def _first_unique(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _proposal_ops(pid: str, final_values: list[str]) -> list[dict[str, Any]]:
    if not final_values:
        return [{"op": "DELETE_ALL", "pid": pid}]
    if len(final_values) == 1:
        return [{"op": "SET", "pid": pid, "value": final_values[0]}]
    return [{"op": "DELETE_ALL", "pid": pid}] + [{"op": "ADD", "pid": pid, "value": value} for value in final_values]


def _baseline_provenance(
    record: dict[str, Any],
    baseline_name: str,
    node_id: str | None = None,
) -> list[dict[str, Any]]:
    node = node_id or record.get("property") or record.get("qid") or "Q0"
    return [
        {
            "kind": "KG",
            "node_id": str(node),
            "snippet": f"{baseline_name} deterministic benchmark-local evidence",
        }
    ]


def _a_box_proposal(
    record: dict[str, Any],
    final_values: list[str],
    *,
    baseline_name: str,
    rationale: str,
    confidence: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    case_id = record.get("id")
    qid = record.get("qid")
    pid = record.get("property")
    if not all(isinstance(value, str) and value for value in (case_id, qid, pid)):
        return None
    ops = _proposal_ops(pid, final_values)
    if len(ops) > 50:
        return None
    payload = {
        "case_id": case_id,
        "target": {"qid": qid, "pid": pid},
        "ops": ops,
        "rationale": rationale,
        "provenance": _baseline_provenance(record, baseline_name),
        "uncertainty": {"confidence": confidence, "notes": f"Generated by {baseline_name}."},
        "metadata": {"baseline": baseline_name, **(metadata or {})},
    }
    try:
        return normalize_a_box_proposal(payload).to_dict()
    except PatchValidationError:
        return None


def _collapse_internal_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _normalize_format_value(value: str, rule_subfamily: str | None) -> str | None:
    text = str(value)
    rule = normalize_text(rule_subfamily or "")
    if rule == "strip_whitespace":
        return text.strip()
    if rule == "collapse_whitespace":
        return _collapse_internal_whitespace(text)
    if rule == "strip_trailing_punctuation":
        return text.strip().rstrip(".,;:")
    if rule == "strip_trailing_slash":
        return text.strip().rstrip("/\\")
    if rule == "append_trailing_slash":
        stripped = text.strip()
        return stripped if stripped.endswith("/") else f"{stripped}/"
    if rule == "strip_category_prefix" and text.strip().startswith("Category:"):
        return text.strip()[len("Category:") :]
    if rule == "strip_schembl_prefix" and re.fullmatch(r"SCHEMBL\d+", text.strip(), flags=re.IGNORECASE):
        return re.sub(r"(?i)^SCHEMBL", "", text.strip())
    if rule == "strip_alpha_prefix" and re.fullmatch(r"[A-Za-z]+[0-9]+", text.strip()):
        stripped = re.sub(r"^[A-Za-z]+", "", text.strip())
        return stripped if len(stripped) >= 3 else None
    if rule == "extract_url_slug":
        match = re.match(r"^https?://[^/]+/(?:[^?#]*/)?([^/?#]+)/?[/]?(?:[?#].*)?$", text.strip(), flags=re.I)
        return match.group(1) if match else None
    if rule == "normalize_case":
        return None
    candidate = text.strip().rstrip(".,;:/\\")
    if candidate != text:
        return candidate
    if re.fullmatch(r"SCHEMBL\d+", text.strip(), flags=re.IGNORECASE):
        return re.sub(r"(?i)^SCHEMBL", "", text.strip())
    return None


def _constraint_qid(constraint: dict[str, Any]) -> str | None:
    ctype = constraint.get("constraint_type")
    if isinstance(ctype, dict) and isinstance(ctype.get("qid"), str):
        return ctype["qid"]
    if isinstance(constraint.get("constraint_qid"), str):
        return constraint["constraint_qid"]
    return None


def _constraint_qualifier_values(constraint: dict[str, Any], property_id: str) -> list[str]:
    values: list[str] = []
    qualifiers = constraint.get("qualifiers")
    if not isinstance(qualifiers, list):
        return values
    for qualifier in qualifiers:
        if not isinstance(qualifier, dict) or qualifier.get("property_id") != property_id:
            continue
        q_values = qualifier.get("values")
        if isinstance(q_values, list):
            values.extend(comparable_atom(value) for value in q_values)
    return values


def _constraints(world_state_entry: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(world_state_entry, dict):
        return []
    constraints = world_state_entry.get("L4_constraints", {}).get("constraints", [])
    if not isinstance(constraints, list):
        return []
    return [constraint for constraint in constraints if isinstance(constraint, dict)]


def _regexes(world_state_entry: dict[str, Any] | None) -> list[str]:
    patterns: list[str] = []
    for constraint in _constraints(world_state_entry):
        if _constraint_qid(constraint) == "Q21502404":
            patterns.extend(_constraint_qualifier_values(constraint, "P1793"))
    return patterns


def _passes_any_regex(value: str, patterns: list[str]) -> bool | None:
    if not patterns:
        return None
    for pattern in patterns:
        try:
            if re.fullmatch(pattern, str(value)):
                return True
        except re.error:
            continue
    return False


def _allowed_or_forbidden_values(world_state_entry: dict[str, Any] | None) -> set[str]:
    values: set[str] = set()
    for constraint in _constraints(world_state_entry):
        if _constraint_qid(constraint) in {"Q21510859", "Q21502402"}:
            values.update(_constraint_qualifier_values(constraint, "P2305"))
    return values


def _numeric(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _range_repair_values(record: dict[str, Any], world_state_entry: dict[str, Any] | None) -> list[str] | None:
    old_values = _old_values(record)
    if len(old_values) != 1:
        return None
    old_value = old_values[0]
    old_numeric = _numeric(old_value)
    for constraint in _constraints(world_state_entry):
        if _constraint_qid(constraint) != "Q21510860":
            continue
        minimums = _constraint_qualifier_values(constraint, "P2313") + _constraint_qualifier_values(constraint, "P2310")
        maximums = _constraint_qualifier_values(constraint, "P2312") + _constraint_qualifier_values(constraint, "P2311")
        numeric_mins = [value for value in (_numeric(item) for item in minimums) if value is not None]
        numeric_maxs = [value for value in (_numeric(item) for item in maximums) if value is not None]
        if old_numeric is not None:
            if numeric_mins and old_numeric < min(numeric_mins):
                return [str(min(numeric_mins)).rstrip("0").rstrip(".")]
            if numeric_maxs and old_numeric > max(numeric_maxs):
                return [str(max(numeric_maxs)).rstrip("0").rstrip(".")]
        for boundary in minimums + maximums:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", boundary):
                return [boundary]
    return None


def constraint_only_typea_proposal(
    record: dict[str, Any],
    world_state_entry: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if record.get("track") != "A_BOX":
        return None
    classification = _classification(record)
    if classification.get("class") != "TypeA":
        return None
    subtype = classification.get("subtype")
    if not isinstance(subtype, str):
        return None

    pid = record.get("property")
    if not isinstance(pid, str) or not pid:
        return None
    old_values = _old_values(record)
    rule_subfamily = classification.get("classification_rule_subfamily")

    final_values: list[str] | None = None
    reason = ""
    confidence = 0.85

    if subtype == "FORMAT_NORMALIZATION":
        summary = _value_change(record)
        retained = [comparable_atom(value) for value in summary.get("retained_unique_values", []) if value is not None]
        removed = [comparable_atom(value) for value in summary.get("removed_unique_values", []) if value is not None]
        changed_old = summary.get("old_changed_value")
        old_candidates = [comparable_atom(changed_old)] if changed_old is not None else removed or old_values
        normalized: list[str] = []
        for old_value in old_candidates:
            new_value = _normalize_format_value(old_value, rule_subfamily if isinstance(rule_subfamily, str) else None)
            if new_value is None:
                return None
            normalized.append(new_value)
        final_values = _first_unique(retained + normalized)
        reason = "Apply a deterministic format normalization rule to the changed value."
    elif subtype in {"REJECTION_FORMAT_INVALID", "FORMAT_VALUE_PRUNING"}:
        patterns = _regexes(world_state_entry)
        if patterns:
            final_values = [value for value in old_values if _passes_any_regex(value, patterns) is not False]
        elif record.get("repair_target", {}).get("action") == "DELETE":
            final_values = []
        if final_values is None or final_values == old_values:
            return None
        reason = "Remove only values that fail the available format rule."
        confidence = 0.75
    elif subtype == "SET_MEMBERSHIP_REJECTION":
        values = _allowed_or_forbidden_values(world_state_entry)
        if not values:
            return None
        report_type = normalize_text(record.get("violation_context", {}).get("report_violation_type_normalized") or "")
        if report_type == "none of":
            final_values = [value for value in old_values if value not in values]
        else:
            final_values = [value for value in old_values if value in values]
            if not final_values and len(values) == 1:
                final_values = sorted(values)
        if final_values == old_values:
            return None
        reason = "Keep values permitted by the set-membership constraint."
        confidence = 0.8
    elif subtype == "SELF_LINK_REJECTION":
        focus_qid = record.get("qid")
        final_values = [value for value in old_values if value != focus_qid]
        if final_values == old_values:
            return None
        reason = "Remove the focus entity from a self-link violation."
    elif subtype == "TARGET_REQUIRED_CLAIM":
        focus_qid = record.get("qid")
        if not isinstance(focus_qid, str) or not focus_qid:
            return None
        final_values = [focus_qid]
        reason = "A target-required-claim report deterministically requires the focus entity."
        confidence = 0.75
    elif subtype == "MULTIPLICITY_NORMALIZATION":
        final_values = _first_unique(old_values)
        if final_values == old_values:
            return None
        reason = "Collapse duplicate values while preserving the unique value set."
        confidence = 0.8
    elif subtype in {"REJECTION_RULE_INVALID", "LOGICAL"}:
        if record.get("repair_target", {}).get("action") != "DELETE":
            range_values = _range_repair_values(record, world_state_entry)
            if range_values is None:
                return None
            final_values = range_values
            reason = "Set the target to a deterministic range boundary."
        else:
            final_values = []
            reason = "Delete a value identified as invalid by the supported rule family."
        confidence = 0.7

    if final_values is None:
        return None
    return _a_box_proposal(
        record,
        final_values,
        baseline_name="constraint_only_typea",
        rationale=reason,
        confidence=confidence,
        metadata={"classification_subtype": subtype, "rule_subfamily": rule_subfamily},
    )


def _local_text_fields(record: dict[str, Any], world_state_entry: dict[str, Any] | None) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    if not isinstance(world_state_entry, dict):
        return fields
    l1 = world_state_entry.get("L1_ego_node")
    if isinstance(l1, dict):
        for key in ("label", "description"):
            value = l1.get(key)
            if isinstance(value, str) and value.strip():
                fields.append((f"focus_{key}", value))
        aliases = l1.get("aliases")
        if isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    fields.append(("focus_alias", alias))
        properties = l1.get("properties")
        target_pid = record.get("property")
        if isinstance(properties, dict):
            for pid, values in properties.items():
                if pid == target_pid:
                    continue
                for value in normalize_value_list(values):
                    fields.append((f"focus_property:{pid}", value))
    edges = world_state_entry.get("L3_neighborhood", {}).get("outgoing_edges")
    if isinstance(edges, list):
        for edge in edges:
            if not isinstance(edge, dict) or edge.get("property_id") == record.get("property"):
                continue
            for key in ("target_qid", "target_label", "target_description"):
                value = edge.get(key)
                if isinstance(value, str) and value.strip():
                    fields.append((f"neighbor_{key}", value))
    labels = world_state_entry.get("L2_labels", {}).get("entities")
    if isinstance(labels, dict):
        for entity_id, entry in labels.items():
            if not isinstance(entry, dict):
                continue
            for key in ("label", "description", "label_en", "description_en"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    fields.append((f"l2:{entity_id}:{key}", value))
    return fields


def _token_visible_locally(token: str, fields: list[tuple[str, str]]) -> bool:
    token_text = str(token).strip()
    if not token_text:
        return False
    token_norm = normalize_text(token_text)
    for _, raw_value in fields:
        value = str(raw_value).strip()
        if token_text == value:
            return True
        value_norm = normalize_text(value)
        if value_norm == token_norm:
            return True
        if len(token_norm) >= 4 and re.search(rf"(?<![0-9a-z]){re.escape(token_norm)}(?![0-9a-z])", value_norm):
            return True
    return False


def _p8726_derived_value(record: dict[str, Any], world_state_entry: dict[str, Any] | None) -> str | None:
    if record.get("property") != "P8726":
        return None
    fields = _local_text_fields(record, world_state_entry)
    pattern = re.compile(
        r"(?:\bS\.?\s*I\.?\s*(?:No\.?)?\s*(?P<number>\d+)\s*/\s*(?P<year>\d{4})\b|"
        r"\bStatutory\s+Instrument\b.{0,80}\b(?P<number2>\d+)\s*/\s*(?P<year2>\d{4})\b)",
        flags=re.IGNORECASE,
    )
    for _, text in fields:
        match = pattern.search(text)
        if not match:
            continue
        number = match.group("number") or match.group("number2")
        year = match.group("year") or match.group("year2")
        if number and year:
            return f"{year}/si/{int(number)}/made"
    return None


def local_lookup_oracle_proposal(
    record: dict[str, Any],
    world_state_entry: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if record.get("track") != "A_BOX":
        return None
    classification = _classification(record)
    if classification.get("class") != "TypeB":
        return None
    subtype = classification.get("subtype")
    if not isinstance(subtype, str) or not subtype.startswith("LOCAL_"):
        return None

    summary = _value_change(record)
    semantic_action = summary.get("semantic_action")
    fields = _local_text_fields(record, world_state_entry)

    if subtype == "LOCAL_TEXT_DERIVED":
        derived = _p8726_derived_value(record, world_state_entry)
        if derived is None:
            return None
        final_values = [derived]
    elif subtype == "LOCAL_SELECTION_CONFIRMED" or semantic_action == "DELETE_SUBSET":
        final_values = [
            comparable_atom(value) for value in summary.get("retained_unique_values", []) if value is not None
        ]
        if not final_values or not all(_token_visible_locally(value, fields) for value in final_values):
            return None
    else:
        final_values = _new_values(record)
        if not final_values or not all(_token_visible_locally(value, fields) for value in final_values):
            return None

    return _a_box_proposal(
        record,
        final_values,
        baseline_name="local_lookup_oracle",
        rationale="Use target values found or deterministically derived from local graph context.",
        confidence=0.9,
        metadata={"classification_subtype": subtype, "semantic_action": semantic_action},
    )


def do_nothing_proposal(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("track") != "A_BOX":
        return None
    state = pre_repair_target_state(record)
    return _a_box_proposal(
        record,
        list(state.values),
        baseline_name="do_nothing_pre_repair",
        rationale="Leave the reconstructed pre-repair target-property state unchanged.",
        confidence=0.95,
        metadata={"pre_repair_source": state.source},
    )


def track_prediction_rows(records: list[dict[str, Any]], baseline_name: str) -> list[dict[str, Any]]:
    if baseline_name not in TRACK_BASELINES:
        raise ValueError(f"Unsupported track baseline: {baseline_name}")
    track_counts = Counter(record.get("track") for record in records)
    majority_track = "A_BOX"
    if track_counts:
        majority_track = str(max(("A_BOX", "T_BOX"), key=lambda track: (track_counts.get(track, 0), track)))
    fixed = {
        "majority_track": majority_track,
        "always_a_box": "A_BOX",
        "always_t_box": "T_BOX",
        "always_ambiguous": "AMBIGUOUS",
    }[baseline_name]
    rows = []
    for record in records:
        rows.append(
            normalize_diagnosis(
                {
                    "case_id": record["id"],
                    "predicted_track": fixed,
                    "confidence": "high",
                    "rationale": f"{baseline_name} deterministic baseline.",
                }
            ).to_dict()
        )
    return rows


def track_metrics(records: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    confusion: dict[str, dict[str, int]] = {
        actual: {predicted: 0 for predicted in TRACK_LABELS} for actual in TRACK_LABELS
    }
    for record in records:
        actual = record.get("track") if record.get("track") in TRACK_LABELS else "AMBIGUOUS"
        predicted = predictions.get(record["id"], "AMBIGUOUS")
        if predicted not in TRACK_LABELS:
            predicted = "AMBIGUOUS"
        confusion[str(actual)][predicted] += 1

    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    total = len(records)
    correct = 0
    for label in TRACK_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in TRACK_LABELS if actual != label)
        fn = sum(confusion[label][predicted] for predicted in TRACK_LABELS if predicted != label)
        correct += tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }
    tbox_total = sum(confusion["T_BOX"].values())
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "confusion_matrix": confusion,
        "per_label": per_label,
        "a_box_overuse_rate": sum(confusion[actual]["A_BOX"] for actual in TRACK_LABELS) / total if total else 0.0,
        "t_box_miss_rate": (
            (tbox_total - confusion["T_BOX"]["T_BOX"]) / tbox_total if tbox_total else 0.0
        ),
    }


def _coverage(
    records: list[dict[str, Any]],
    proposal_case_ids: set[str],
    *,
    target_class: str | None = None,
) -> dict[str, Any]:
    eligible = [
        record
        for record in records
        if record.get("track") == "A_BOX"
        and (target_class is None or _classification(record).get("class") == target_class)
    ]
    by_subtype: dict[str, dict[str, int]] = defaultdict(lambda: {"eligible": 0, "covered": 0})
    for record in eligible:
        subtype = str(_classification(record).get("subtype"))
        by_subtype[subtype]["eligible"] += 1
        if record["id"] in proposal_case_ids:
            by_subtype[subtype]["covered"] += 1
    return {
        "eligible_cases": len(eligible),
        "proposal_cases": len(proposal_case_ids),
        "coverage": len(proposal_case_ids) / len(eligible) if eligible else 0.0,
        "by_subtype": {
            subtype: {
                **counts,
                "coverage": counts["covered"] / counts["eligible"] if counts["eligible"] else 0.0,
            }
            for subtype, counts in sorted(by_subtype.items())
        },
    }


def _evaluate(
    *,
    records: list[dict[str, Any]],
    options: PhaseEOptions,
    baseline_dir: Path,
    a_box_proposals_path: Path | None = None,
    t_box_proposals_path: Path | None = None,
    track_diagnoses_path: Path | None = None,
    run_manifest_path: Path | None = None,
    ablation_bundle: str,
) -> dict[str, Any]:
    _, summary = evaluate_benchmark(
        classified_path=options.classified_benchmark,
        world_state_path=options.world_state,
        a_box_proposals_path=a_box_proposals_path,
        t_box_proposals_path=t_box_proposals_path,
        track_diagnoses_path=track_diagnoses_path,
        run_manifest_path=run_manifest_path,
        selection_manifest_path=options.selection_manifest,
        out_traces_path=baseline_dir / "evaluation_traces.jsonl",
        out_summary_path=baseline_dir / "evaluation_summary.json",
        classified_records=records,
        classified_input_path=options.classified_benchmark,
        collect_traces=False,
        ablation_bundle=ablation_bundle,
    )
    return summary


def _run_manifest_parse_errors(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                "case_id": record["id"],
                "task_type": "proposal",
                "ablation_bundle": "invalid_empty",
                "parse_status": "parse_error",
                "parser_error": "invalid_empty_baseline_no_normalized_proposal",
                "provider": "non_llm",
                "model": "invalid_empty",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )
    return rows


def run_phase_e(options: PhaseEOptions) -> dict[str, Any]:
    manifest = load_selection_manifest(options.selection_manifest)
    records = selected_records(options.classified_benchmark, manifest["selected_case_ids"])
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "manifest_type": "phase_e_non_llm_baseline_run",
        "manifest_version": "phase_e_v1",
        "created_at_utc": utc_now(),
        "inputs": {
            "classified_benchmark": str(options.classified_benchmark),
            "world_state": str(options.world_state),
            "selection_manifest": str(options.selection_manifest),
        },
        "tier": options.tier,
        "selected_cases": len(records),
        "selected_track_counts": dict(Counter(record.get("track") for record in records)),
    }
    write_json(output_dir / "run_config.json", config)

    summaries: dict[str, Any] = {}

    for baseline_name in TRACK_BASELINES:
        baseline_dir = output_dir / baseline_name
        rows = track_prediction_rows(records, baseline_name)
        track_path = baseline_dir / "track_diagnoses.jsonl"
        write_jsonl(track_path, rows)
        predictions = {row["case_id"]: row["predicted_track"] for row in rows}
        evaluation = _evaluate(
            records=records,
            options=options,
            baseline_dir=baseline_dir,
            track_diagnoses_path=track_path,
            ablation_bundle=baseline_name,
        )
        metrics = track_metrics(records, predictions)
        summary = {
            "baseline": baseline_name,
            "artifact_paths": {
                "track_diagnoses": str(track_path),
                "evaluation_traces": str(baseline_dir / "evaluation_traces.jsonl"),
                "evaluation_summary": str(baseline_dir / "evaluation_summary.json"),
            },
            "track_metrics": metrics,
            "evaluation": evaluation,
        }
        write_json(baseline_dir / "baseline_summary.json", summary)
        summaries[baseline_name] = summary

    world_store = WorldStateStore(options.world_state, __import__("logging").getLogger("non_llm_baselines"))
    world_store.open()
    try:
        proposal_builders = {
            "constraint_only_typea": lambda record, world: constraint_only_typea_proposal(record, world),
            "local_lookup_oracle": lambda record, world: local_lookup_oracle_proposal(record, world),
        }
        for baseline_name, builder in proposal_builders.items():
            baseline_dir = output_dir / baseline_name
            proposals: list[dict[str, Any]] = []
            for idx, record in enumerate(records, start=1):
                proposal = builder(record, world_store.get(record["id"]))
                if proposal is not None:
                    proposals.append(proposal)
                if options.progress_every and idx % options.progress_every == 0:
                    print(f"[phase-e] {baseline_name}: scanned {idx:,}/{len(records):,} cases")
            proposal_path = baseline_dir / "a_box_proposals.jsonl"
            write_jsonl(proposal_path, proposals)
            proposal_ids = {proposal["case_id"] for proposal in proposals}
            evaluation = _evaluate(
                records=records,
                options=options,
                baseline_dir=baseline_dir,
                a_box_proposals_path=proposal_path,
                ablation_bundle=baseline_name,
            )
            target_class = "TypeA" if baseline_name == "constraint_only_typea" else "TypeB"
            coverage = _coverage(records, proposal_ids, target_class=target_class)
            summary = {
                "baseline": baseline_name,
                "artifact_paths": {
                    "a_box_proposals": str(proposal_path),
                    "evaluation_traces": str(baseline_dir / "evaluation_traces.jsonl"),
                    "evaluation_summary": str(baseline_dir / "evaluation_summary.json"),
                },
                "coverage": coverage,
                "evaluation": evaluation,
            }
            write_json(baseline_dir / "baseline_summary.json", summary)
            summaries[baseline_name] = summary

        do_nothing_dir = output_dir / "do_nothing_pre_repair"
        do_nothing_proposals = [proposal for record in records if (proposal := do_nothing_proposal(record)) is not None]
        do_nothing_path = do_nothing_dir / "a_box_proposals.jsonl"
        write_jsonl(do_nothing_path, do_nothing_proposals)
        evaluation = _evaluate(
            records=records,
            options=options,
            baseline_dir=do_nothing_dir,
            a_box_proposals_path=do_nothing_path,
            ablation_bundle="do_nothing_pre_repair",
        )
        summary = {
            "baseline": "do_nothing_pre_repair",
            "artifact_paths": {
                "a_box_proposals": str(do_nothing_path),
                "evaluation_traces": str(do_nothing_dir / "evaluation_traces.jsonl"),
                "evaluation_summary": str(do_nothing_dir / "evaluation_summary.json"),
            },
            "coverage": _coverage(records, {proposal["case_id"] for proposal in do_nothing_proposals}),
            "evaluation": evaluation,
        }
        write_json(do_nothing_dir / "baseline_summary.json", summary)
        summaries["do_nothing_pre_repair"] = summary
    finally:
        world_store.close()

    invalid_dir = output_dir / "invalid_empty"
    raw_invalid_path = invalid_dir / "raw_invalid_proposals.jsonl"
    write_jsonl(
        raw_invalid_path,
        (
            {"case_id": record["id"], "proposal": {}, "error": "intentionally invalid empty proposal"}
            for record in records
        ),
    )
    run_manifest_path = invalid_dir / "run_manifest.jsonl"
    write_jsonl(run_manifest_path, _run_manifest_parse_errors(records))
    evaluation = _evaluate(
        records=records,
        options=options,
        baseline_dir=invalid_dir,
        run_manifest_path=run_manifest_path,
        ablation_bundle="invalid_empty",
    )
    summary = {
        "baseline": "invalid_empty",
        "artifact_paths": {
            "raw_invalid_proposals": str(raw_invalid_path),
            "run_manifest": str(run_manifest_path),
            "evaluation_traces": str(invalid_dir / "evaluation_traces.jsonl"),
            "evaluation_summary": str(invalid_dir / "evaluation_summary.json"),
        },
        "evaluation": evaluation,
        "sanity_check": {
            "accepted_rate_should_be_near_zero": True,
            "accepted_rate": evaluation.get("overall_metrics", {}).get("accepted_rate"),
            "proposal_parse_error_rate": evaluation.get("parse_errors", {}).get("proposal_parse_error_rate"),
        },
    }
    write_json(invalid_dir / "baseline_summary.json", summary)
    summaries["invalid_empty"] = summary

    aggregate = {
        **config,
        "baselines": {
            name: {
                "artifact_paths": summary.get("artifact_paths", {}),
                "track_metrics": summary.get("track_metrics"),
                "coverage": summary.get("coverage"),
                "accepted_rate": summary.get("evaluation", {}).get("overall_metrics", {}).get("accepted_rate"),
                "exact_historical_agreement_rate": summary.get("evaluation", {})
                .get("overall_metrics", {})
                .get("exact_historical_agreement_rate"),
                "track_diagnosis_accuracy": summary.get("evaluation", {})
                .get("overall_metrics", {})
                .get("track_diagnosis_accuracy"),
            }
            for name, summary in sorted(summaries.items())
        },
    }
    write_json(output_dir / "baseline_summary.json", aggregate)
    write_phase_e_markdown(output_dir / "baseline_summary.md", aggregate)
    return aggregate


def write_phase_e_markdown(path: str | Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "# Phase E Non-LLM Baseline Summary",
        "",
        f"- Created: `{aggregate['created_at_utc']}`",
        f"- Selection manifest: `{aggregate['inputs']['selection_manifest']}`",
        f"- Selected cases: `{aggregate['selected_cases']}`",
        "",
        "| Baseline | Accepted rate | Exact agreement | Track accuracy | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in aggregate["baselines"].items():
        coverage = summary.get("coverage", {})
        coverage_value = coverage.get("coverage") if isinstance(coverage, dict) else None
        lines.append(
            "| {name} | {accepted} | {exact} | {track} | {coverage} |".format(
                name=name,
                accepted=_fmt(summary.get("accepted_rate")),
                exact=_fmt(summary.get("exact_historical_agreement_rate")),
                track=_fmt(summary.get("track_diagnosis_accuracy")),
                coverage=_fmt(coverage_value),
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{value:.4f}"


__all__ = [
    "PhaseEOptions",
    "TRACK_BASELINES",
    "constraint_only_typea_proposal",
    "do_nothing_proposal",
    "local_lookup_oracle_proposal",
    "run_phase_e",
    "selected_records",
    "track_metrics",
    "track_prediction_rows",
]
