from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from classifier import WorldStateStore
from lib.benchmark_selection import resolve_case_id_filter
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


def _iter_records(classified_path: str | Path, case_ids: Optional[set[str]] = None) -> Iterable[dict[str, Any]]:
    for record in iter_jsonl(classified_path):
        if not isinstance(record, dict):
            continue
        rid = record.get("id")
        if not isinstance(rid, str) or not rid:
            continue
        if case_ids is not None and rid not in case_ids:
            continue
        yield record


def _load_records(classified_path: str | Path, case_ids: Optional[set[str]] = None) -> list[dict[str, Any]]:
    return list(_iter_records(classified_path, case_ids))


def _derive_popularity_buckets_from_path(
    classified_path: str | Path,
    case_ids: Optional[set[str]] = None,
) -> dict[str, str]:
    scored: list[tuple[float, str]] = []
    missing: list[str] = []
    for record in _iter_records(classified_path, case_ids):
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


def _proposal_parse_status(manifest_record: dict[str, Any], proposal_missing: bool) -> str:
    return manifest_record.get("parse_status") or ("missing" if proposal_missing else "normalized")


def _proposal_error_details(manifest_record: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    parser_error = manifest_record.get("parser_error")
    if isinstance(parser_error, str) and parser_error.strip():
        details["parser_error"] = parser_error.strip()
    provider_error = manifest_record.get("provider_error")
    if isinstance(provider_error, str) and provider_error.strip():
        details["provider_error"] = provider_error.strip()
    return details


def _world_state_target_values(
    world_state_entry: Optional[dict[str, Any]],
    property_id: str | None,
) -> list[str]:
    if not isinstance(world_state_entry, dict) or not isinstance(property_id, str) or not property_id:
        return []
    l1_node = world_state_entry.get("L1_ego_node")
    if not isinstance(l1_node, dict):
        return []
    properties = l1_node.get("properties")
    if not isinstance(properties, dict):
        return []
    return _normalize_value_list(properties.get(property_id))


def _signature_after_jaccard(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    left_set = {json.dumps(entry, sort_keys=True, ensure_ascii=False) for entry in left if isinstance(entry, dict)}
    right_set = {json.dumps(entry, sort_keys=True, ensure_ascii=False) for entry in right if isinstance(entry, dict)}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _proposal_signature_entries_for_constraint(
    signature_after: list[dict[str, Any]],
    constraint_qid: str | None,
) -> list[dict[str, Any]]:
    if not isinstance(constraint_qid, str) or not constraint_qid:
        return [entry for entry in signature_after if isinstance(entry, dict)]
    matching_entries = [
        entry
        for entry in signature_after
        if isinstance(entry, dict) and entry.get("constraint_qid") == constraint_qid
    ]
    if matching_entries:
        return matching_entries
    return [entry for entry in signature_after if isinstance(entry, dict)]


def _proposal_admits_current_values(
    *,
    proposal: Any,
    current_values: list[str],
) -> Optional[bool]:
    if proposal is None:
        return None
    constraint_qid = getattr(proposal.target, "constraint_type_qid", None)
    signature_after = getattr(proposal.proposal, "signature_after", None)
    if not isinstance(signature_after, list):
        return None
    relevant_entries = _proposal_signature_entries_for_constraint(signature_after, constraint_qid)
    if constraint_qid in ONE_OF_QIDS:
        allowed_values = {
            value
            for entry in relevant_entries
            for value in _qualifier_values(entry, "P2305")
        }
        if not allowed_values:
            return None
        return all(value in allowed_values for value in current_values)
    if constraint_qid in RANGE_QIDS:
        mins = [
            numeric
            for entry in relevant_entries
            for numeric in (_parse_numeric(value) for value in _qualifier_values(entry, "P2310"))
            if numeric is not None
        ]
        maxs = [
            numeric
            for entry in relevant_entries
            for numeric in (_parse_numeric(value) for value in _qualifier_values(entry, "P2311"))
            if numeric is not None
        ]
        if not mins and not maxs:
            return None
        for raw_value in current_values:
            numeric_value = _parse_numeric(raw_value)
            if numeric_value is None:
                return False
            if mins and numeric_value < min(mins):
                return False
            if maxs and numeric_value > max(maxs):
                return False
        return True
    if constraint_qid in FORMAT_QIDS:
        patterns = [
            pattern
            for entry in relevant_entries
            for pattern in _qualifier_values(entry, "P1793")
        ]
        if not patterns:
            return None
        import re

        return all(any(re.fullmatch(pattern, value) for pattern in patterns) for value in current_values)
    return None


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
    parse_status = _proposal_parse_status(manifest_record, proposal_missing)

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
        "semantic_success": None,
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
        "semantic_success": None,
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
            **_proposal_error_details(manifest_record),
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
    proposal_missing = proposal is None
    valid = proposal is not None
    target_pid = record.get("property")
    executable = False
    exact_action_match = False
    exact_signature_match = False
    changed_constraint_type_hit = False
    exact_reform_match = False
    semantic_reform_match = False
    semantic_success = False
    signature_after_jaccard = 0.0
    proposal_admits_current_values = None
    normalized_historical_signature: list[dict[str, Any]] = []
    changed_constraint_types: set[str] = set()
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
        changed_constraint_type_hit = proposal.target.constraint_type_qid in changed_constraint_types
        executable = proposal.target.pid == target_pid and changed_constraint_type_hit
        exact_action_match = proposal.proposal.action == record.get("classification", {}).get("subtype")
        exact_signature_match = proposal.proposal.signature_after == normalized_historical_signature
        exact_reform_match = exact_action_match and exact_signature_match
        semantic_reform_match = proposal.proposal.action == record.get("classification", {}).get("subtype")
        semantic_success = bool(executable and semantic_reform_match)
        signature_after_jaccard = _signature_after_jaccard(
            proposal.proposal.signature_after,
            normalized_historical_signature,
        )
        proposal_admits_current_values = _proposal_admits_current_values(
            proposal=proposal,
            current_values=_world_state_target_values(world_state_entry, target_pid),
        )
    accepted = bool(valid and executable and exact_reform_match)
    metrics = {
        "functional_success": 1.0 if accepted else 0.0,
        "exact_historical_agreement": 1.0 if exact_reform_match else 0.0,
        "semantic_success": 1.0 if semantic_success else 0.0,
        "information_preservation": None,
        "provenance_completeness": _provenance_completeness(proposal.provenance if proposal else []),
        "token_usage": _token_usage(manifest_record),
        "conversion_rate": None,
        "tokens_to_fix": None,
        "exact_action_match": 1.0 if exact_action_match else 0.0,
        "exact_signature_match": 1.0 if exact_signature_match else 0.0,
        "changed_constraint_type_hit": 1.0 if changed_constraint_type_hit else 0.0,
        "signature_after_jaccard": signature_after_jaccard,
        "proposal_admits_current_values": (
            None if proposal_admits_current_values is None else (1.0 if proposal_admits_current_values else 0.0)
        ),
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
        "parse_status": _proposal_parse_status(manifest_record, proposal_missing),
        "semantic_success": semantic_success,
        "accepted": accepted,
        "comparison": {
            "exact_action_match": exact_action_match,
            "exact_value_match": None,
            "semantic_reform_match": semantic_reform_match,
            "exact_reform_match": exact_reform_match,
            "exact_signature_match": exact_signature_match,
            "changed_constraint_type_hit": changed_constraint_type_hit,
        },
        "metrics": metrics,
        "details": {
            "proposal_admits_current_values": proposal_admits_current_values,
            **_proposal_error_details(manifest_record),
        },
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
    return summarize_trace_iterable(traces, inputs)


def summarize_trace_iterable(
    traces: Iterable[dict[str, Any]],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    class GroupAccumulator:
        def __init__(self) -> None:
            self.count = 0
            self.accepted = 0
            self.metric_sums: dict[str, float] = defaultdict(float)
            self.metric_counts: dict[str, int] = defaultdict(int)
            self.token_total_sum = 0
            self.token_total_count = 0
            self.diagnosis_count = 0
            self.diagnosis_exact = 0
            self.diagnosis_present = 0

        def add(self, trace: dict[str, Any]) -> None:
            self.count += 1
            if trace.get("accepted"):
                self.accepted += 1
            metrics = trace.get("metrics", {})
            for field in (
                "functional_success",
                "exact_historical_agreement",
                "semantic_success",
                "information_preservation",
                "provenance_completeness",
                "exact_action_match",
                "exact_signature_match",
                "changed_constraint_type_hit",
                "signature_after_jaccard",
                "proposal_admits_current_values",
            ):
                value = metrics.get(field)
                if isinstance(value, (int, float)):
                    self.metric_sums[field] += float(value)
                    self.metric_counts[field] += 1
            if trace.get("parse_status") == "parse_error":
                self.metric_sums["proposal_parse_error_count"] += 1.0
            parser_error = trace.get("details", {}).get("parser_error")
            if isinstance(parser_error, str) and parser_error:
                self.metric_counts[f"parser_error::{parser_error}"] += 1
            total_tokens = metrics.get("token_usage", {}).get("total_tokens")
            if isinstance(total_tokens, int):
                self.token_total_sum += total_tokens
                self.token_total_count += 1
            diagnosis = trace.get("track_diagnosis")
            if isinstance(diagnosis, dict):
                self.diagnosis_count += 1
                if diagnosis.get("exact_track_match"):
                    self.diagnosis_exact += 1
                if diagnosis.get("present"):
                    self.diagnosis_present += 1

        def as_dict(self) -> dict[str, Any]:
            if self.count == 0:
                return {"count": 0}

            def avg(field: str) -> Optional[float]:
                count = self.metric_counts.get(field, 0)
                if count == 0:
                    return None
                return self.metric_sums[field] / count

            return {
                "count": self.count,
                "accepted_rate": self.accepted / self.count,
                "functional_success_rate": avg("functional_success"),
                "exact_historical_agreement_rate": avg("exact_historical_agreement"),
                "semantic_success_rate": avg("semantic_success"),
                "information_preservation_mean": avg("information_preservation"),
                "provenance_completeness_mean": avg("provenance_completeness"),
                "exact_action_match_rate": avg("exact_action_match"),
                "exact_signature_match_rate": avg("exact_signature_match"),
                "changed_constraint_type_hit_rate": avg("changed_constraint_type_hit"),
                "signature_after_jaccard_mean": avg("signature_after_jaccard"),
                "proposal_admits_current_values_rate": avg("proposal_admits_current_values"),
                "metric_applicability": {
                    "semantic_success": self.metric_counts.get("semantic_success", 0),
                    "information_preservation": self.metric_counts.get("information_preservation", 0),
                    "exact_action_match": self.metric_counts.get("exact_action_match", 0),
                    "exact_signature_match": self.metric_counts.get("exact_signature_match", 0),
                    "changed_constraint_type_hit": self.metric_counts.get("changed_constraint_type_hit", 0),
                    "signature_after_jaccard": self.metric_counts.get("signature_after_jaccard", 0),
                    "proposal_admits_current_values": self.metric_counts.get("proposal_admits_current_values", 0),
                },
                "proposal_parse_error_count": int(self.metric_sums.get("proposal_parse_error_count", 0.0)),
                "proposal_parse_error_rate": self.metric_sums.get("proposal_parse_error_count", 0.0) / self.count,
                "proposal_parse_errors_by_message": {
                    key.removeprefix("parser_error::"): value
                    for key, value in sorted(self.metric_counts.items())
                    if key.startswith("parser_error::")
                },
                "token_usage_total_mean": (
                    self.token_total_sum / self.token_total_count if self.token_total_count else None
                ),
                "track_diagnosis_accuracy": (
                    self.diagnosis_exact / self.diagnosis_count if self.diagnosis_count else None
                ),
                "track_diagnosis_present_rate": self.diagnosis_present / self.count if self.count else None,
            }

    groups = {
        "by_class": defaultdict(GroupAccumulator),
        "by_subtype": defaultdict(GroupAccumulator),
        "by_track": defaultdict(GroupAccumulator),
        "by_ablation_bundle": defaultdict(GroupAccumulator),
        "by_popularity_bucket": defaultdict(GroupAccumulator),
    }
    counts = Counter()
    parse_errors = Counter()
    overall = GroupAccumulator()
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
        metrics = trace.get("metrics", {})
        if metrics.get("semantic_success") == 1.0:
            counts["semantic_success"] += 1
        for metric_name in (
            "semantic_success",
            "information_preservation",
            "exact_action_match",
            "exact_signature_match",
            "changed_constraint_type_hit",
            "signature_after_jaccard",
            "proposal_admits_current_values",
        ):
            if isinstance(metrics.get(metric_name), (int, float)):
                counts[f"{metric_name}_applicable"] += 1
        if trace.get("parse_status") == "parse_error":
            counts["proposal_parse_error"] += 1
            parser_error = trace.get("details", {}).get("parser_error")
            if isinstance(parser_error, str) and parser_error:
                parse_errors[parser_error] += 1
        diagnosis = trace.get("track_diagnosis")
        if isinstance(diagnosis, dict):
            if diagnosis.get("present"):
                counts["track_diagnosis_present"] += 1
            if diagnosis.get("exact_track_match"):
                counts["track_diagnosis_exact_match"] += 1
            if diagnosis.get("ambiguous_prediction"):
                counts["track_diagnosis_ambiguous"] += 1
        overall.add(trace)
        groups["by_class"][trace.get("classification_class")].add(trace)
        groups["by_subtype"][trace.get("classification_subtype")].add(trace)
        groups["by_track"][trace.get("track")].add(trace)
        groups["by_ablation_bundle"][trace.get("ablation_bundle")].add(trace)
        groups["by_popularity_bucket"][trace.get("popularity_bucket")].add(trace)

    summary = {
        "build_utc": _utc_now(),
        "inputs": inputs,
        "counts": dict(counts),
        "overall_metrics": overall.as_dict(),
        "parse_errors": {
            "proposal_parse_error_count": counts.get("proposal_parse_error", 0),
            "proposal_parse_error_rate": (
                counts.get("proposal_parse_error", 0) / counts["cases"] if counts.get("cases", 0) else 0.0
            ),
            "by_message": dict(parse_errors),
        },
        "by_class": {str(key): value.as_dict() for key, value in groups["by_class"].items()},
        "by_subtype": {str(key): value.as_dict() for key, value in groups["by_subtype"].items()},
        "by_track": {str(key): value.as_dict() for key, value in groups["by_track"].items()},
        "by_ablation_bundle": {str(key): value.as_dict() for key, value in groups["by_ablation_bundle"].items()},
        "by_popularity_bucket": {str(key): value.as_dict() for key, value in groups["by_popularity_bucket"].items()},
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
    selection_manifest_path: str | Path | None = None,
    out_traces_path: str | Path | None = None,
    out_summary_path: str | Path | None = None,
    collect_traces: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    classified_records: Optional[Iterable[dict[str, Any]]] = None,
    classified_input_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_case_ids = resolve_case_id_filter(
        case_ids=case_ids,
        selection_manifest_path=selection_manifest_path,
    )
    selected_case_id_set = set(selected_case_ids) if selected_case_ids is not None else None
    if classified_records is not None:
        records = [
            record
            for record in classified_records
            if isinstance(record, dict)
            and isinstance(record.get("id"), str)
            and (selected_case_id_set is None or record["id"] in selected_case_id_set)
        ]
        popularity_buckets = _derive_popularity_buckets(records)
        record_iterable: Iterable[dict[str, Any]] = records
    else:
        popularity_buckets = _derive_popularity_buckets_from_path(classified_path, selected_case_id_set)
        record_iterable = _iter_records(classified_path, selected_case_id_set)
    a_box_proposals = _load_a_box_proposals(a_box_proposals_path)
    t_box_proposals = _load_t_box_proposals(t_box_proposals_path)
    track_diagnoses = _load_track_diagnoses(track_diagnoses_path)
    run_manifest = _load_run_manifest(run_manifest_path)

    traces: list[dict[str, Any]] = []
    trace_writer = None
    world_state_store = WorldStateStore(Path(world_state_path), __import__("logging").getLogger("evaluator"))
    world_state_store.open()
    try:
        if out_traces_path:
            trace_path = Path(out_traces_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_writer = open(trace_path, "w", encoding="utf-8")
        for record in record_iterable:
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
            if progress_callback is not None:
                progress_callback(trace)
            if collect_traces:
                traces.append(trace)
            if trace_writer is not None:
                trace_writer.write(json.dumps(trace, ensure_ascii=False, default=_json_default) + "\n")
    finally:
        if trace_writer is not None:
            trace_writer.close()
        world_state_store.close()

    inputs = {
        "classified_benchmark": str(classified_input_path or classified_path),
        "world_state": str(world_state_path),
        "a_box_proposals": str(a_box_proposals_path) if a_box_proposals_path else None,
        "t_box_proposals": str(t_box_proposals_path) if t_box_proposals_path else None,
        "track_diagnoses": str(track_diagnoses_path) if track_diagnoses_path else None,
        "run_manifest": str(run_manifest_path) if run_manifest_path else None,
        "ablation_bundle": ablation_bundle,
        "selection_manifest": str(selection_manifest_path) if selection_manifest_path else None,
    }
    summary = summarize_trace_iterable(iter_jsonl(out_traces_path), inputs) if out_traces_path else summarize_traces(traces, inputs)
    if out_summary_path:
        write_json(out_summary_path, summary)

    return traces, summary
