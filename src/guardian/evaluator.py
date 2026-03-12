from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from classifier import WorldStateStore
from lib.utils import iter_jsonl, normalize_text
from guardian.patch_parser import normalize_proposal as normalize_a_box_proposal
from guardian.track_parser import normalize_diagnosis as normalize_track_diagnosis
from guardian.tbox_parser import normalize_proposal as normalize_t_box_proposal
from guardian.tbox_parser import normalize_signature_after

FORMAT_QIDS = {"Q21502404"}
ONE_OF_QIDS = {"Q21510859", "Q21502402"}
RANGE_QIDS = {"Q21510860"}


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_json_default)


def _normalized_case_ids(case_ids: Optional[Iterable[str]]) -> Optional[set[str]]:
    if not case_ids:
        return None
    return {case_id for case_id in case_ids if isinstance(case_id, str) and case_id}


def _comparable_atom(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("qid", "id", "raw", "value"):
            if key in value:
                return _comparable_atom(value[key])
    return str(value)


def _normalize_value_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    normalized = [_comparable_atom(item) for item in items if item not in (None, "MISSING")]
    return normalized


def _derive_popularity_buckets(records: list[dict[str, Any]]) -> dict[str, str]:
    scored: list[tuple[float, str]] = []
    missing: list[str] = []
    for record in records:
        rid = record["id"]
        popularity = record.get("popularity")
        score = popularity.get("score") if isinstance(popularity, dict) else None
        if isinstance(score, (int, float)):
            scored.append((float(score), rid))
        else:
            missing.append(rid)
    scored.sort(key=lambda item: (item[0], item[1]))
    total = len(scored)
    tail_cut = int(total * 0.2)
    head_cut = int(total * 0.2)
    buckets: dict[str, str] = {}
    for idx, (_, rid) in enumerate(scored):
        if idx < tail_cut:
            buckets[rid] = "tail"
        elif idx >= total - head_cut:
            buckets[rid] = "head"
        else:
            buckets[rid] = "mid"
    for rid in missing:
        buckets[rid] = "unknown"
    return buckets


def _load_records(classified_path: str | Path, case_ids: Optional[set[str]] = None) -> list[dict[str, Any]]:
    records = []
    for record in iter_jsonl(classified_path):
        if not isinstance(record, dict):
            continue
        rid = record.get("id")
        if not isinstance(rid, str) or not rid:
            continue
        if case_ids is not None and rid not in case_ids:
            continue
        records.append(record)
    return records


def _load_a_box_proposals(path: str | Path | None) -> dict[str, Any]:
    proposals = {}
    if not path:
        return proposals
    proposal_path = Path(path)
    if not proposal_path.exists():
        return proposals
    for record in iter_jsonl(proposal_path):
        normalized = normalize_a_box_proposal(record)
        proposals[normalized.case_id] = normalized
    return proposals


def _load_t_box_proposals(path: str | Path | None) -> dict[str, Any]:
    proposals = {}
    if not path:
        return proposals
    proposal_path = Path(path)
    if not proposal_path.exists():
        return proposals
    for record in iter_jsonl(proposal_path):
        normalized = normalize_t_box_proposal(record)
        proposals[normalized.case_id] = normalized
    return proposals


def _load_run_manifest(path: str | Path | None) -> dict[tuple[str, Optional[str]], dict[str, Any]]:
    return _load_run_manifest_by_task(path)


def _load_run_manifest_by_task(
    path: str | Path | None,
) -> dict[tuple[str, Optional[str], str], dict[str, Any]]:
    manifest = {}
    if not path:
        return manifest
    manifest_path = Path(path)
    if not manifest_path.exists():
        return manifest
    for record in iter_jsonl(manifest_path):
        if not isinstance(record, dict):
            continue
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            continue
        bundle = record.get("ablation_bundle")
        bundle_key = bundle if isinstance(bundle, str) and bundle else None
        task_type = record.get("task_type")
        task_key = task_type if isinstance(task_type, str) and task_type else "proposal"
        manifest[(case_id, bundle_key, task_key)] = record
    return manifest


def _manifest_record(
    manifest: dict[tuple[str, Optional[str], str], dict[str, Any]],
    case_id: str,
    ablation_bundle: Optional[str],
    task_type: str = "proposal",
) -> dict[str, Any]:
    return (
        manifest.get((case_id, ablation_bundle, task_type))
        or manifest.get((case_id, None, task_type))
        or manifest.get((case_id, ablation_bundle, "proposal"))
        or manifest.get((case_id, None, "proposal"))
        or {}
    )


def _load_track_diagnoses(path: str | Path | None) -> dict[str, Any]:
    diagnoses = {}
    if not path:
        return diagnoses
    diagnosis_path = Path(path)
    if not diagnosis_path.exists():
        return diagnoses
    for record in iter_jsonl(diagnosis_path):
        normalized = normalize_track_diagnosis(record)
        diagnoses[normalized.case_id] = normalized
    return diagnoses


def _constraint_type_qids(world_state_entry: dict[str, Any]) -> list[str]:
    constraints = world_state_entry.get("L4_constraints", {}).get("constraints", [])
    qids = []
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        ctype = constraint.get("constraint_type", {})
        qid = ctype.get("qid") if isinstance(ctype, dict) else None
        if isinstance(qid, str) and qid not in qids:
            qids.append(qid)
    return qids


def _qualifier_values(constraint: dict[str, Any], property_id: str) -> list[str]:
    values: list[str] = []
    for qualifier in constraint.get("qualifiers", []):
        if not isinstance(qualifier, dict):
            continue
        if qualifier.get("property_id") != property_id:
            continue
        for value in qualifier.get("values", []):
            if isinstance(value, dict):
                values.append(_comparable_atom(value))
            else:
                values.append(_comparable_atom(value))
    return values


def _parse_numeric(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _supported_constraint_violations(
    properties: dict[str, list[str]],
    property_id: str,
    constraints: list[dict[str, Any]],
) -> int:
    target_values = properties.get(property_id, [])
    violations = 0
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        ctype = constraint.get("constraint_type", {})
        constraint_qid = ctype.get("qid") if isinstance(ctype, dict) else None
        if constraint_qid in FORMAT_QIDS:
            patterns = _qualifier_values(constraint, "P1793")
            if not patterns:
                continue
            import re

            matched = True
            for value in target_values:
                if not any(re.fullmatch(pattern, value) for pattern in patterns):
                    matched = False
                    break
            if not matched:
                violations += 1
            continue
        if constraint_qid in ONE_OF_QIDS:
            allowed = set(_qualifier_values(constraint, "P2305"))
            if allowed and any(value not in allowed for value in target_values):
                violations += 1
            continue
        if constraint_qid in RANGE_QIDS:
            mins = [_parse_numeric(value) for value in _qualifier_values(constraint, "P2310")]
            maxs = [_parse_numeric(value) for value in _qualifier_values(constraint, "P2311")]
            mins = [value for value in mins if value is not None]
            maxs = [value for value in maxs if value is not None]
            if not mins and not maxs:
                continue
            range_invalid = False
            for raw_value in target_values:
                numeric = _parse_numeric(raw_value)
                if numeric is None:
                    continue
                if mins and numeric < min(mins):
                    range_invalid = True
                if maxs and numeric > max(maxs):
                    range_invalid = True
            if range_invalid:
                violations += 1
    return violations


def _reconstruct_pre_repair_properties(record: dict[str, Any], world_state_entry: dict[str, Any]) -> dict[str, list[str]]:
    properties = copy.deepcopy(world_state_entry.get("L1_ego_node", {}).get("properties", {}))
    if not isinstance(properties, dict):
        properties = {}
    target_pid = record.get("property")
    repair_target = record.get("repair_target", {})
    old_value = repair_target.get("old_value") if isinstance(repair_target, dict) else None
    if old_value is None:
        old_value = record.get("violation_context", {}).get("value")
    old_values = _normalize_value_list(old_value)
    if old_values:
        properties[target_pid] = old_values
    else:
        properties.pop(target_pid, None)
    return properties


def _apply_ops(properties: dict[str, list[str]], ops: list[Any]) -> tuple[dict[str, list[str]], bool, Optional[str]]:
    state = copy.deepcopy(properties)
    try:
        for op in ops:
            pid = op.pid
            current = list(state.get(pid, []))
            if op.op == "SET":
                state[pid] = [_comparable_atom(op.value)]
            elif op.op == "ADD":
                value = _comparable_atom(op.value)
                if value not in current:
                    current.append(value)
                state[pid] = current
            elif op.op == "REMOVE":
                value = _comparable_atom(op.value)
                state[pid] = [item for item in current if item != value]
                if not state[pid]:
                    state.pop(pid, None)
            elif op.op == "DELETE_ALL":
                state.pop(pid, None)
            else:
                return properties, False, f"unsupported_op:{op.op}"
        return state, True, None
    except Exception as exc:
        return properties, False, f"execution_error:{exc}"


def _expected_target_values(record: dict[str, Any]) -> list[str]:
    repair_target = record.get("repair_target", {})
    if not isinstance(repair_target, dict):
        return []
    if repair_target.get("action") == "DELETE":
        return []
    return _normalize_value_list(repair_target.get("new_value") or repair_target.get("value"))


def _derived_action(before_values: list[str], after_values: list[str]) -> str:
    if before_values and not after_values:
        return "DELETE"
    if not before_values and after_values:
        return "CREATE"
    if before_values != after_values:
        return "UPDATE"
    return "NOOP"


def _token_usage(manifest_record: dict[str, Any]) -> dict[str, Optional[int]]:
    usage = manifest_record.get("usage")
    if not isinstance(usage, dict):
        metadata = manifest_record.get("metadata")
        usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    if not isinstance(usage, dict):
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _provenance_completeness(payload: Any) -> float:
    if not isinstance(payload, list) or not payload:
        return 0.0
    complete = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind == "WEB" and item.get("url"):
            complete += 1
        elif kind == "KG" and item.get("node_id"):
            complete += 1
        elif kind == "HISTORY" and item.get("revision_id"):
            complete += 1
        elif kind not in {"WEB", "KG", "HISTORY"}:
            complete += 1
    return complete / len(payload)


def evaluate_a_box_case(
    record: dict[str, Any],
    world_state_entry: Optional[dict[str, Any]],
    proposal: Any,
    manifest_record: dict[str, Any],
    popularity_bucket: str,
    ablation_bundle: Optional[str],
) -> dict[str, Any]:
    proposal_missing = proposal is None
    target_pid = record.get("property")
    target_qid = record.get("qid")
    valid = proposal is not None
    executable = bool(valid and proposal.target.qid == target_qid and proposal.target.pid == target_pid)
    parse_status = manifest_record.get("parse_status") or ("missing" if proposal_missing else "normalized")

    exact_action_match = False
    exact_value_match = False
    regression_pass = False
    supported_violations_before = None
    supported_violations_after = None
    execution_error = None
    info_preservation = 0.0

    if executable and isinstance(world_state_entry, dict):
        before_properties = _reconstruct_pre_repair_properties(record, world_state_entry)
        after_properties, executable, execution_error = _apply_ops(before_properties, proposal.ops)
        before_target = before_properties.get(target_pid, [])
        after_target = after_properties.get(target_pid, [])
        expected_target = _expected_target_values(record)
        derived_action = _derived_action(before_target, after_target)
        historical_action = record.get("repair_target", {}).get("action")
        exact_action_match = derived_action == historical_action
        exact_value_match = after_target == expected_target
        constraints = world_state_entry.get("L4_constraints", {}).get("constraints", [])
        if isinstance(constraints, list):
            supported_violations_before = _supported_constraint_violations(before_properties, target_pid, constraints)
            supported_violations_after = _supported_constraint_violations(after_properties, target_pid, constraints)
            regression_pass = supported_violations_after <= supported_violations_before
        else:
            regression_pass = True

        if exact_action_match and exact_value_match:
            info_preservation = 1.0
        elif historical_action != "DELETE" and not after_target:
            info_preservation = -0.5
    functional_success = bool(executable and exact_value_match)
    accepted = bool(functional_success and regression_pass)
    provenance = proposal.provenance if proposal is not None else []
    metrics = {
        "functional_success": 1.0 if functional_success else 0.0,
        "exact_historical_agreement": 1.0 if exact_action_match and exact_value_match else 0.0,
        "information_preservation": info_preservation,
        "provenance_completeness": _provenance_completeness(provenance),
        "token_usage": _token_usage(manifest_record),
        "conversion_rate": None,
        "tokens_to_fix": None,
    }
    return {
        "case_id": record["id"],
        "track": record.get("track"),
        "classification_class": record.get("classification", {}).get("class"),
        "classification_subtype": record.get("classification", {}).get("subtype"),
        "ablation_bundle": ablation_bundle,
        "popularity_bucket": popularity_bucket,
        "proposal_type": "A_BOX",
        "proposal_present": not proposal_missing,
        "proposal_valid": valid,
        "proposal_executable": executable,
        "parse_status": parse_status,
        "accepted": accepted,
        "comparison": {
            "exact_action_match": exact_action_match,
            "exact_value_match": exact_value_match,
            "semantic_reform_match": None,
            "exact_reform_match": None,
        },
        "metrics": metrics,
        "details": {
            "execution_error": execution_error,
            "supported_violations_before": supported_violations_before,
            "supported_violations_after": supported_violations_after,
            "historical_action": record.get("repair_target", {}).get("action"),
            "expected_target_values": _expected_target_values(record),
        },
    }


def evaluate_t_box_case(
    record: dict[str, Any],
    world_state_entry: Optional[dict[str, Any]],
    proposal: Any,
    manifest_record: dict[str, Any],
    popularity_bucket: str,
    ablation_bundle: Optional[str],
) -> dict[str, Any]:
    del world_state_entry
    proposal_missing = proposal is None
    valid = proposal is not None
    target_pid = record.get("property")
    executable = False
    exact_reform_match = False
    semantic_reform_match = False
    if proposal is not None:
        historical_signature = record.get("repair_target", {}).get("constraint_delta", {}).get("signature_after", [])
        normalized_historical_signature = normalize_signature_after(historical_signature)
        changed_constraint_types = {
            constraint_qid
            for constraint_qid in record.get("repair_target", {})
            .get("constraint_delta", {})
            .get("changed_constraint_types", [])
            if isinstance(constraint_qid, str)
        }
        if not changed_constraint_types:
            changed_constraint_types = {
                entry.get("constraint_qid")
                for entry in normalized_historical_signature
                if isinstance(entry, dict) and isinstance(entry.get("constraint_qid"), str)
            }
        executable = proposal.target.pid == target_pid and proposal.target.constraint_type_qid in changed_constraint_types
        exact_reform_match = proposal.proposal.signature_after == normalized_historical_signature
        semantic_reform_match = proposal.proposal.action == record.get("classification", {}).get("subtype")
    accepted = bool(valid and executable and (exact_reform_match or semantic_reform_match))
    metrics = {
        "functional_success": 1.0 if exact_reform_match else 0.0,
        "exact_historical_agreement": 1.0 if exact_reform_match else 0.0,
        "information_preservation": None,
        "provenance_completeness": _provenance_completeness(proposal.provenance if proposal else []),
        "token_usage": _token_usage(manifest_record),
        "conversion_rate": None,
        "tokens_to_fix": None,
    }
    return {
        "case_id": record["id"],
        "track": record.get("track"),
        "classification_class": record.get("classification", {}).get("class"),
        "classification_subtype": record.get("classification", {}).get("subtype"),
        "ablation_bundle": ablation_bundle,
        "popularity_bucket": popularity_bucket,
        "proposal_type": "T_BOX",
        "proposal_present": not proposal_missing,
        "proposal_valid": valid,
        "proposal_executable": executable,
        "parse_status": manifest_record.get("parse_status") or ("missing" if proposal_missing else "normalized"),
        "accepted": accepted,
        "comparison": {
            "exact_action_match": None,
            "exact_value_match": None,
            "semantic_reform_match": semantic_reform_match,
            "exact_reform_match": exact_reform_match,
        },
        "metrics": metrics,
        "details": {},
    }


def evaluate_track_diagnosis(
    record: dict[str, Any],
    diagnosis: Any,
    manifest_record: dict[str, Any],
) -> dict[str, Any]:
    diagnosis_missing = diagnosis is None
    historical_track = record.get("track")
    predicted_track = diagnosis.predicted_track if diagnosis is not None else None
    exact_track_match = bool(predicted_track == historical_track)
    ambiguous_prediction = bool(predicted_track == "AMBIGUOUS")
    confidence = diagnosis.confidence if diagnosis is not None else None
    rationale = diagnosis.rationale if diagnosis is not None else None
    return {
        "valid": diagnosis is not None,
        "present": not diagnosis_missing,
        "parse_status": manifest_record.get("parse_status") or ("missing" if diagnosis_missing else "normalized"),
        "historical_track": historical_track,
        "predicted_track": predicted_track,
        "exact_track_match": exact_track_match,
        "ambiguous_prediction": ambiguous_prediction,
        "confidence": confidence,
        "rationale": rationale,
        "token_usage": _token_usage(manifest_record),
    }


def summarize_traces(
    traces: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    groups = {
        "by_class": defaultdict(list),
        "by_subtype": defaultdict(list),
        "by_track": defaultdict(list),
        "by_ablation_bundle": defaultdict(list),
        "by_popularity_bucket": defaultdict(list),
    }
    counts = Counter()
    for trace in traces:
        counts["cases"] += 1
        if trace.get("accepted"):
            counts["accepted"] += 1
        if trace.get("proposal_present"):
            counts["proposal_present"] += 1
        if trace.get("proposal_executable"):
            counts["proposal_executable"] += 1
        if trace.get("metrics", {}).get("functional_success") == 1.0:
            counts["functional_success"] += 1
        diagnosis = trace.get("track_diagnosis")
        if isinstance(diagnosis, dict):
            if diagnosis.get("present"):
                counts["track_diagnosis_present"] += 1
            if diagnosis.get("exact_track_match"):
                counts["track_diagnosis_exact_match"] += 1
            if diagnosis.get("ambiguous_prediction"):
                counts["track_diagnosis_ambiguous"] += 1
        groups["by_class"][trace.get("classification_class")].append(trace)
        groups["by_subtype"][trace.get("classification_subtype")].append(trace)
        groups["by_track"][trace.get("track")].append(trace)
        groups["by_ablation_bundle"][trace.get("ablation_bundle")].append(trace)
        groups["by_popularity_bucket"][trace.get("popularity_bucket")].append(trace)

    def aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"count": 0}
        def avg(field: str) -> Optional[float]:
            values = [trace["metrics"].get(field) for trace in group if trace["metrics"].get(field) is not None]
            if not values:
                return None
            return sum(values) / len(values)

        token_totals = [trace["metrics"]["token_usage"].get("total_tokens") for trace in group]
        token_totals = [value for value in token_totals if isinstance(value, int)]
        diagnosis_group = [trace.get("track_diagnosis") for trace in group if isinstance(trace.get("track_diagnosis"), dict)]
        return {
            "count": len(group),
            "accepted_rate": sum(1 for trace in group if trace.get("accepted")) / len(group),
            "functional_success_rate": avg("functional_success"),
            "exact_historical_agreement_rate": avg("exact_historical_agreement"),
            "information_preservation_mean": avg("information_preservation"),
            "provenance_completeness_mean": avg("provenance_completeness"),
            "token_usage_total_mean": (sum(token_totals) / len(token_totals)) if token_totals else None,
            "track_diagnosis_accuracy": (
                sum(1 for diagnosis in diagnosis_group if diagnosis.get("exact_track_match")) / len(diagnosis_group)
                if diagnosis_group
                else None
            ),
            "track_diagnosis_present_rate": (
                sum(1 for diagnosis in diagnosis_group if diagnosis.get("present")) / len(group)
                if group
                else None
            ),
        }

    summary = {
        "build_utc": _utc_now(),
        "inputs": inputs,
        "counts": dict(counts),
        "overall_metrics": aggregate(traces),
        "by_class": {str(key): aggregate(value) for key, value in groups["by_class"].items()},
        "by_subtype": {str(key): aggregate(value) for key, value in groups["by_subtype"].items()},
        "by_track": {str(key): aggregate(value) for key, value in groups["by_track"].items()},
        "by_ablation_bundle": {str(key): aggregate(value) for key, value in groups["by_ablation_bundle"].items()},
        "by_popularity_bucket": {str(key): aggregate(value) for key, value in groups["by_popularity_bucket"].items()},
    }
    return summary


def evaluate_benchmark(
    *,
    classified_path: str | Path,
    world_state_path: str | Path,
    a_box_proposals_path: str | Path | None = None,
    t_box_proposals_path: str | Path | None = None,
    track_diagnoses_path: str | Path | None = None,
    run_manifest_path: str | Path | None = None,
    ablation_bundle: Optional[str] = None,
    case_ids: Optional[Iterable[str]] = None,
    out_traces_path: str | Path | None = None,
    out_summary_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_case_ids = _normalized_case_ids(case_ids)
    records = _load_records(classified_path, selected_case_ids)
    popularity_buckets = _derive_popularity_buckets(records)
    a_box_proposals = _load_a_box_proposals(a_box_proposals_path)
    t_box_proposals = _load_t_box_proposals(t_box_proposals_path)
    track_diagnoses = _load_track_diagnoses(track_diagnoses_path)
    run_manifest = _load_run_manifest(run_manifest_path)

    traces: list[dict[str, Any]] = []
    world_state_store = WorldStateStore(Path(world_state_path), __import__("logging").getLogger("evaluator"))
    world_state_store.open()
    try:
        for record in records:
            case_id = record["id"]
            manifest_record = _manifest_record(run_manifest, case_id, ablation_bundle, "proposal")
            diagnosis_manifest_record = _manifest_record(run_manifest, case_id, ablation_bundle, "track_diagnosis")
            world_state_entry = world_state_store.get(case_id)
            popularity_bucket = popularity_buckets.get(case_id, "unknown")
            if record.get("track") == "T_BOX":
                trace = evaluate_t_box_case(
                    record,
                    world_state_entry,
                    t_box_proposals.get(case_id),
                    manifest_record,
                    popularity_bucket,
                    ablation_bundle,
                )
            else:
                trace = evaluate_a_box_case(
                    record,
                    world_state_entry,
                    a_box_proposals.get(case_id),
                    manifest_record,
                    popularity_bucket,
                    ablation_bundle,
                )
            trace["track_diagnosis"] = evaluate_track_diagnosis(
                record,
                track_diagnoses.get(case_id),
                diagnosis_manifest_record,
            )
            traces.append(trace)
    finally:
        world_state_store.close()

    inputs = {
        "classified_benchmark": str(classified_path),
        "world_state": str(world_state_path),
        "a_box_proposals": str(a_box_proposals_path) if a_box_proposals_path else None,
        "t_box_proposals": str(t_box_proposals_path) if t_box_proposals_path else None,
        "track_diagnoses": str(track_diagnoses_path) if track_diagnoses_path else None,
        "run_manifest": str(run_manifest_path) if run_manifest_path else None,
        "ablation_bundle": ablation_bundle,
    }
    summary = summarize_traces(traces, inputs)

    if out_traces_path:
        write_jsonl(out_traces_path, traces)
    if out_summary_path:
        write_json(out_summary_path, summary)

    return traces, summary
