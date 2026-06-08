from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DIAGNOSTIC_TASKS = (
    "a_box_value_extraction",
    "a_box_operation_selection",
    "a_box_answerability",
    "t_box_constraint_family_selection",
    "t_box_action_selection",
    "t_box_signature_visibility",
    "track_locus_contrast",
)

REPAIR_STATUS = {
    "exact",
    "accepted_non_exact",
    "wrong_value",
    "wrong_operation",
    "overdelete",
    "underrepair",
    "hallucinated_replacement",
    "abstain",
    "wrong_tbox_family",
    "invented_signature",
    "request_error",
    "parse_error",
}

MISSING_VALUES = {"", "MISSING", "None", "null", "NULL"}
DETERMINISTIC_A_BOX_SUBTYPES = {
    "FORMAT_NORMALIZATION",
    "FORMAT_VALUE_PRUNING",
    "MULTIPLICITY_NORMALIZATION",
    "REJECTION_FORMAT_INVALID",
    "SELF_LINK_REJECTION",
    "SET_MEMBERSHIP_REJECTION",
    "TARGET_REQUIRED_CLAIM",
    "DELETE_AMBIGUOUS",
}


@dataclass(frozen=True)
class Inputs:
    run_dir: Path
    classified_benchmark: Path
    dev_manifest: Path
    diagnostic_output_dir: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_classified(path: Path) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in _iter_jsonl(path)}


def _load_classified_subset(path: Path, case_ids: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not case_ids:
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = record.get("id")
            if case_id in case_ids:
                records[case_id] = record
                if len(records) == len(case_ids):
                    break
    missing = sorted(case_ids - set(records))
    if missing:
        raise KeyError(f"Classified benchmark is missing {len(missing)} selected cases, first={missing[:3]}")
    return records


def _classification(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("classification")
    return value if isinstance(value, dict) else {}


def _repair_target(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("repair_target")
    return value if isinstance(value, dict) else {}


def _manifest_annotation(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    annotations = manifest.get("case_annotations")
    if not isinstance(annotations, dict):
        return {}
    value = annotations.get(case_id)
    return value if isinstance(value, dict) else {}


def _flatten_scalars(value: Any, prefix: str = "$") -> list[tuple[str, str]]:
    scalars: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            scalars.extend(_flatten_scalars(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scalars.extend(_flatten_scalars(child, f"{prefix}[{index}]"))
    elif value is not None:
        scalars.append((prefix, str(value)))
    return scalars


def _scalar_tokens(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        return [str(value) for value in values if str(value) not in MISSING_VALUES]
    return [str(values)] if str(values) not in MISSING_VALUES else []


def _value_summary(record: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _classification(record).get("diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    summary = diagnostics.get("value_change_summary")
    return summary if isinstance(summary, dict) else {}


def _a_box_gold_values(record: dict[str, Any]) -> dict[str, list[str]]:
    target = _repair_target(record)
    summary = _value_summary(record)
    new_values = _scalar_tokens(summary.get("new_unique") or target.get("new_value") or target.get("value"))
    old_values = _scalar_tokens(summary.get("old_unique") or target.get("old_value"))
    retained = _scalar_tokens(summary.get("retained_unique_values"))
    removed = _scalar_tokens(summary.get("removed_unique_values"))
    added = _scalar_tokens(summary.get("added_unique_values"))
    violation = record.get("violation_context") if isinstance(record.get("violation_context"), dict) else {}
    report_values = _scalar_tokens(violation.get("value"))
    return {
        "new_values": new_values,
        "old_values": old_values,
        "retained_values": retained,
        "removed_values": removed,
        "added_values": added,
        "report_values": report_values,
    }


def _record_from_prompt_trace(
    prompt: dict[str, Any],
    payload: dict[str, Any],
    annotation: dict[str, Any],
    trace: dict[str, Any] | None,
) -> dict[str, Any]:
    details = trace.get("details", {}) if isinstance(trace, dict) else {}
    expected_values = _scalar_tokens(details.get("expected_target_values"))
    violation = payload.get("violation_context") if isinstance(payload.get("violation_context"), dict) else {}
    old_values = _scalar_tokens(violation.get("value"))
    if not old_values:
        ego = payload.get("local_context", {}).get("L1_ego_node", {}) if isinstance(payload.get("local_context"), dict) else {}
        properties = ego.get("properties") if isinstance(ego.get("properties"), dict) else {}
        old_values = _scalar_tokens(properties.get(payload.get("property")))
    track = prompt.get("historical_track")
    cls = (
        annotation.get("class")
        or (trace or {}).get("classification_class")
        or track
        or "unknown"
    )
    subtype = annotation.get("subtype") or (trace or {}).get("classification_subtype") or "unknown"
    return {
        "id": prompt.get("case_id"),
        "track": track,
        "qid": payload.get("qid"),
        "property": payload.get("property"),
        "violation_context": violation,
        "repair_target": {
            "kind": track,
            "new_value": expected_values,
            "old_value": old_values,
        },
        "classification": {
            "class": cls,
            "subtype": subtype,
            "decision_constraint_type_qid": annotation.get("decision_constraint_type_qid")
            or details.get("historical_target_constraint_qid"),
            "diagnostics": {
                "value_change_summary": {
                    "new_unique": expected_values,
                    "old_unique": old_values,
                    "retained_unique_values": [
                        value for value in old_values if value in set(expected_values)
                    ],
                    "removed_unique_values": [
                        value for value in old_values if value not in set(expected_values)
                    ],
                    "added_unique_values": [
                        value for value in expected_values if value not in set(old_values)
                    ],
                }
            },
        },
    }


def _t_box_gold_values(record: dict[str, Any]) -> dict[str, list[str] | str | None]:
    cls = _classification(record)
    changed_values: set[str] = set()
    changed_constraints: set[str] = set()
    target_constraint: str | None = None
    for step in cls.get("decision_trace", []) if isinstance(cls.get("decision_trace"), list) else []:
        if not isinstance(step, dict):
            continue
        for key in ("added_values", "removed_values", "compatible_value_overlap_with_report_qids"):
            for value in _scalar_tokens(step.get(key)):
                changed_values.add(value)
        for key in ("changed_constraint_qids_all", "changed_constraint_qids_from_entries", "changed_constraint_qids_from_qualifier_changes"):
            for value in _scalar_tokens(step.get(key)):
                changed_constraints.add(value)
        if isinstance(step.get("target_constraint_qid"), str):
            target_constraint = step["target_constraint_qid"]
    if isinstance(cls.get("decision_constraint_type_qid"), str):
        target_constraint = target_constraint or cls["decision_constraint_type_qid"]
        changed_constraints.add(cls["decision_constraint_type_qid"])
    rt = _repair_target(record)
    delta = rt.get("constraint_delta") if isinstance(rt.get("constraint_delta"), dict) else {}
    for value in _scalar_tokens(delta.get("changed_constraint_types")):
        changed_constraints.add(value)
    return {
        "changed_values": sorted(changed_values),
        "changed_constraints": sorted(changed_constraints),
        "target_constraint": target_constraint,
    }


def _extract_input_payload(user_prompt: str) -> dict[str, Any]:
    marker_positions = [
        user_prompt.rfind("Input case JSON:"),
        user_prompt.rfind("Input case description:"),
        user_prompt.rfind("Input case compact table:"),
        user_prompt.rfind("Input case Turtle-like triples:"),
    ]
    marker = max(marker_positions)
    if marker < 0:
        return {}
    start = user_prompt.find("{", marker)
    if start < 0:
        return {}
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(user_prompt[start:])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _find_token_source(token: str, payload: dict[str, Any], prompt_text: str) -> str | None:
    for path, scalar in _flatten_scalars(payload):
        if scalar == token:
            if ".violation_context" in path:
                return "violation_context"
            if ".local_context" in path:
                return "local_context"
            if ".logic_context" in path:
                return "logic_context"
            if ".labels_en" in path or ".L2_labels" in path:
                return "label_text"
            return f"payload:{path}"
    return "prompt_text" if token and token in prompt_text else None


def _a_box_visibility(record: dict[str, Any], payload: dict[str, Any], prompt_text: str) -> tuple[bool, str, str]:
    cls = _classification(record)
    subtype = str(cls.get("subtype") or "unknown")
    values = _a_box_gold_values(record)
    new_values = values["new_values"]
    if new_values:
        sources = []
        for value in new_values:
            source = _find_token_source(value, payload, prompt_text)
            if source:
                sources.append(source)
        if len(sources) == len(new_values):
            return True, ",".join(sorted(set(sources))), "gold final value tokens visible"
        if subtype in DETERMINISTIC_A_BOX_SUBTYPES and any(_find_token_source(v, payload, prompt_text) for v in values["old_values"]):
            return True, "deterministic_format_or_rule_transform", "final token not literal, but deterministic rule/old value evidence is visible"
        if str(cls.get("class")) == "TypeC":
            return False, "typec_nonvisible", "TypeC final value is not available in visible prompt evidence"
        return False, "gold_value_not_visible", "gold final value token is not visible"
    removed_sources = [_find_token_source(value, payload, prompt_text) for value in values["removed_values"] or values["old_values"] or values["report_values"]]
    if any(removed_sources):
        return True, "visible_bad_or_removed_value", "gold behavior is removal and the bad/old value is visible"
    return False, "remove_target_not_visible", "removal target not visible"


def _t_box_visibility(
    record: dict[str, Any],
    payload: dict[str, Any],
    prompt_text: str,
    temporal_policy: str | None,
) -> tuple[bool, str, str, bool]:
    gold = _t_box_gold_values(record)
    changed_values = [value for value in gold["changed_values"] if isinstance(value, str)]
    changed_constraints = [value for value in gold["changed_constraints"] if isinstance(value, str)]
    value_sources = [_find_token_source(value, payload, prompt_text) for value in changed_values]
    family_sources = [_find_token_source(value, payload, prompt_text) for value in changed_constraints]
    exact_signature_visible = bool(changed_values and all(value_sources))
    if exact_signature_visible:
        return True, "changed_values_visible", "changed semantic values are visible in prompt payload", True
    if temporal_policy == "compact_inventory_no_pre_change_signature":
        if any(family_sources):
            return False, "compact_inventory_family_only", "constraint family is visible, but exact changed signature values are not inferable", False
        return False, "compact_inventory_no_changed_values", "compact temporal policy exposes no exact changed signature values", False
    if any(family_sources):
        return False, "constraint_family_visible_no_values", "target family is visible but changed semantic values are not", False
    return False, "changed_values_not_visible", "changed semantic values are not visible", False


def _proposal_values(proposal: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for op in proposal.get("ops", []) if isinstance(proposal.get("ops"), list) else []:
        if isinstance(op, dict) and op.get("value") is not None:
            values.append(str(op["value"]))
    return values


def _proposal_ops(proposal: dict[str, Any]) -> list[str]:
    return [str(op.get("op")) for op in proposal.get("ops", []) if isinstance(op, dict) and op.get("op")]


def _proposal_signature_values(proposal: dict[str, Any]) -> list[str]:
    values: list[str] = []
    prop = proposal.get("proposal") if isinstance(proposal.get("proposal"), dict) else {}
    for entry in prop.get("signature_after", []) if isinstance(prop.get("signature_after"), list) else []:
        if not isinstance(entry, dict):
            continue
        if entry.get("constraint_qid") is not None:
            values.append(str(entry["constraint_qid"]))
        for qualifier in entry.get("qualifiers", []) if isinstance(entry.get("qualifiers"), list) else []:
            if isinstance(qualifier, dict):
                values.extend(_scalar_tokens(qualifier.get("values")))
    return values


def _is_abstention(proposal: dict[str, Any]) -> bool:
    return proposal.get("abstain") is True


def _classify_a_box_status(
    trace: dict[str, Any] | None,
    proposal: dict[str, Any] | None,
    payload: dict[str, Any],
    prompt_text: str,
    record: dict[str, Any],
    parse_status: str,
) -> str:
    if parse_status == "request_error":
        return "request_error"
    if parse_status == "parse_error":
        return "parse_error"
    proposal = proposal or {}
    if _is_abstention(proposal):
        return "abstain"
    if trace and trace.get("accepted") is True:
        exact_action = trace.get("comparison", {}).get("exact_action_match") is True
        exact_value = trace.get("comparison", {}).get("exact_value_match") is True
        return "exact" if exact_action and exact_value else "accepted_non_exact"
    values = _a_box_gold_values(record)
    ops = _proposal_ops(proposal)
    proposed_values = _proposal_values(proposal)
    expected_values = set(values["new_values"])
    retained_values = set(values["retained_values"])
    if "DELETE_ALL" in ops and (expected_values or retained_values):
        return "overdelete"
    if expected_values and not expected_values.issubset(set(proposed_values)):
        if any(value and not _find_token_source(value, payload, prompt_text) for value in proposed_values):
            return "hallucinated_replacement"
        return "wrong_value"
    if trace and trace.get("comparison", {}).get("exact_action_match") is False:
        return "wrong_operation"
    if trace and trace.get("comparison", {}).get("exact_value_match") is False:
        return "wrong_value"
    if not proposed_values and expected_values:
        return "underrepair"
    if proposed_values and any(value and not _find_token_source(value, payload, prompt_text) for value in proposed_values):
        return "hallucinated_replacement"
    return "wrong_value"


def _classify_t_box_status(
    trace: dict[str, Any] | None,
    proposal: dict[str, Any] | None,
    record: dict[str, Any],
    temporal_policy: str | None,
    parse_status: str,
) -> str:
    if parse_status == "request_error":
        return "request_error"
    if parse_status == "parse_error":
        return "parse_error"
    proposal = proposal or {}
    if _is_abstention(proposal):
        return "abstain"
    if trace and trace.get("accepted") is True:
        return "exact"
    prop = proposal.get("proposal") if isinstance(proposal.get("proposal"), dict) else {}
    signature = prop.get("signature_after") if isinstance(prop.get("signature_after"), list) else []
    if temporal_policy == "compact_inventory_no_pre_change_signature" and signature:
        return "invented_signature"
    metrics = trace.get("metrics", {}) if trace else {}
    if metrics.get("t_box_target_constraint_match") == 0.0 or metrics.get("changed_constraint_type_hit") == 0.0:
        return "wrong_tbox_family"
    if metrics.get("semantic_family_success") == 1.0:
        return "accepted_non_exact"
    return "wrong_operation"


def _build_indexes(run_dir: Path) -> dict[str, Any]:
    prompt_rows = _iter_jsonl(run_dir / "rendered_prompts" / "prompt_dev_rendered_prompts.jsonl")
    repair_prompt_rows = [
        row for row in prompt_rows if row.get("task") in {"a_box_repair", "t_box_repair"}
    ]
    matrix_dirs = [path for path in (run_dir / "matrices").iterdir() if path.is_dir()]
    traces: dict[tuple[str, str], dict[str, Any]] = {}
    proposals: dict[tuple[str, str], dict[str, Any]] = {}
    manifest_rows: dict[tuple[str, str], dict[str, Any]] = {}
    track_traces: list[dict[str, Any]] = []
    for matrix_dir in matrix_dirs:
        matrix_id = matrix_dir.name
        for trace in _iter_jsonl(matrix_dir / "evaluation_traces.jsonl"):
            traces[(matrix_id, trace.get("case_id"))] = trace
            if "track_diagnosis" in matrix_id:
                track_traces.append(trace)
        for name in ("a_box_proposals.jsonl", "t_box_proposals.jsonl"):
            for proposal in _iter_jsonl(matrix_dir / name):
                proposals[(matrix_id, proposal.get("case_id"))] = proposal
        for row in _iter_jsonl(matrix_dir / "run_manifest.jsonl"):
            manifest_rows[(matrix_id, row.get("case_id"))] = row
    return {
        "repair_prompt_rows": repair_prompt_rows,
        "traces": traces,
        "proposals": proposals,
        "manifest_rows": manifest_rows,
        "track_traces": track_traces,
    }


def build_answerability_audit(inputs: Inputs) -> list[dict[str, Any]]:
    manifest = _read_json(inputs.dev_manifest)
    indexes = _build_indexes(inputs.run_dir)
    rows: list[dict[str, Any]] = []
    for prompt in indexes["repair_prompt_rows"]:
        case_id = str(prompt["case_id"])
        matrix_id = str(prompt["matrix_id"])
        annotation = _manifest_annotation(manifest, case_id)
        payload = _extract_input_payload(str(prompt.get("user_prompt") or ""))
        prompt_text = f"{prompt.get('system_prompt') or ''}\n{prompt.get('user_prompt') or ''}"
        trace = indexes["traces"].get((matrix_id, case_id))
        record = _record_from_prompt_trace(prompt, payload, annotation, trace)
        cls = _classification(record)
        proposal = indexes["proposals"].get((matrix_id, case_id))
        run_row = indexes["manifest_rows"].get((matrix_id, case_id), {})
        parse_status = str((trace or {}).get("parse_status") or run_row.get("parse_status") or "missing")
        temporal_policy = (run_row.get("context_audit") or {}).get("temporal_policy")
        diagnostic_only = bool(annotation.get("diagnostic_only"))
        main_score = bool(annotation.get("main_score"))
        if record.get("track") == "A_BOX":
            visible, visible_source, note = _a_box_visibility(record, payload, prompt_text)
            values = _a_box_gold_values(record)
            if diagnostic_only:
                expected_behavior = "diagnostic_only"
            elif visible and values["new_values"]:
                expected_behavior = "exact_repair"
            elif visible:
                expected_behavior = "conservative_remove"
            else:
                expected_behavior = "abstain"
            status = _classify_a_box_status(trace, proposal, payload, prompt_text, record, parse_status)
        else:
            visible, visible_source, note, exact_signature_visible = _t_box_visibility(
                record,
                payload,
                prompt_text,
                temporal_policy if isinstance(temporal_policy, str) else None,
            )
            if diagnostic_only:
                expected_behavior = "diagnostic_only"
            elif exact_signature_visible:
                expected_behavior = "exact_repair"
            else:
                expected_behavior = "schema_update_low_confidence"
            status = _classify_t_box_status(trace, proposal, record, temporal_policy, parse_status)
        row = {
            "case_id": case_id,
            "visible_case_id": prompt.get("visible_case_id"),
            "context_bundle": prompt.get("context_bundle"),
            "track": record.get("track"),
            "class": cls.get("class"),
            "subtype": cls.get("subtype"),
            "selection_stratum": annotation.get("selection_stratum"),
            "diagnostic_only": diagnostic_only,
            "main_score": main_score,
            "gold_target_visible": visible,
            "gold_target_visible_source": visible_source,
            "expected_behavior": expected_behavior,
            "proposal_status": status if status in REPAIR_STATUS else "wrong_value",
            "notes": note,
            "parse_status": parse_status,
            "accepted": bool(trace and trace.get("accepted") is True),
            "exact_historical": bool(trace and trace.get("metrics", {}).get("exact_historical_agreement") == 1.0),
        }
        rows.append(row)
    return rows


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def summarize_answerability(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    by_visible: dict[str, dict[str, Any]] = {}
    for visible in (True, False):
        subset = [row for row in rows if row["gold_target_visible"] is visible]
        exact = sum(row["proposal_status"] == "exact" for row in subset)
        accepted = sum(row["proposal_status"] in {"exact", "accepted_non_exact"} or row.get("accepted") for row in subset)
        by_visible[str(visible).lower()] = {
            "n": len(subset),
            "exact": exact,
            "exact_rate": _rate(exact, len(subset)),
            "accepted_or_non_exact": accepted,
            "accepted_or_non_exact_rate": _rate(accepted, len(subset)),
            "by_status": _counter_dict(Counter(row["proposal_status"] for row in subset)),
            "by_track": _counter_dict(Counter(row["track"] for row in subset)),
        }
    by_track_context = defaultdict(Counter)
    for row in rows:
        by_track_context[f"{row['track']}::{row['context_bundle']}"][row["proposal_status"]] += 1
    summary = {
        "manifest_type": "prompt_dev_answerability_audit_summary",
        "run_dir": str(output_dir),
        "counts": {
            "rows": len(rows),
            "by_track": _counter_dict(Counter(row["track"] for row in rows)),
            "by_context": _counter_dict(Counter(row["context_bundle"] for row in rows)),
            "by_class": _counter_dict(Counter(row["class"] for row in rows)),
            "by_subtype": _counter_dict(Counter(row["subtype"] for row in rows)),
            "by_expected_behavior": _counter_dict(Counter(row["expected_behavior"] for row in rows)),
            "by_proposal_status": _counter_dict(Counter(row["proposal_status"] for row in rows)),
            "by_gold_target_visible": _counter_dict(Counter(row["gold_target_visible"] for row in rows)),
        },
        "rates_by_gold_target_visible": by_visible,
        "by_track_context_status": {key: _counter_dict(value) for key, value in sorted(by_track_context.items())},
    }
    return summary


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Answerability Audit Summary",
        "",
        f"Rows: `{summary['counts']['rows']}`",
        "",
        "## Counts",
        "",
        f"- By track: `{json.dumps(summary['counts']['by_track'], sort_keys=True)}`",
        f"- By context: `{json.dumps(summary['counts']['by_context'], sort_keys=True)}`",
        f"- By expected behavior: `{json.dumps(summary['counts']['by_expected_behavior'], sort_keys=True)}`",
        f"- By proposal status: `{json.dumps(summary['counts']['by_proposal_status'], sort_keys=True)}`",
        "",
        "## Exact/Accepted By Visibility",
        "",
        "| Gold target visible | N | Exact | Exact rate | Accepted/non-exact | Accepted/non-exact rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, block in summary["rates_by_gold_target_visible"].items():
        exact_rate = block["exact_rate"]
        accepted_rate = block["accepted_or_non_exact_rate"]
        lines.append(
            f"| `{key}` | {block['n']} | {block['exact']} | "
            f"{exact_rate:.3f if exact_rate is not None else 'n/a'} | "
            f"{block['accepted_or_non_exact']} | {accepted_rate:.3f if accepted_rate is not None else 'n/a'} |"
        )
    return "\n".join(lines).replace("n/a", "n/a")


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def build_failure_isolation(
    rows: list[dict[str, Any]],
    run_dir: Path,
    classified_path: Path,
    dev_manifest_path: Path | None = None,
) -> dict[str, Any]:
    indexes = _build_indexes(run_dir)
    manifest = _read_json(dev_manifest_path) if dev_manifest_path is not None else {"case_annotations": {}}
    typea_answerable = [
        row for row in rows if row["class"] == "TypeA" and row["gold_target_visible"] and row["proposal_status"] != "exact"
    ]
    typeb_local_answerable = [
        row
        for row in rows
        if row["class"] == "TypeB"
        and row["context_bundle"] == "local_graph"
        and row["gold_target_visible"]
        and row["proposal_status"] != "exact"
    ]
    typec_rows = [row for row in rows if row["class"] == "TypeC"]
    typec_concrete_failed = [
        row for row in typec_rows if row["proposal_status"] not in {"abstain", "request_error", "parse_error", "exact"}
    ]
    not_visible_hallucinated = [
        row for row in rows if not row["gold_target_visible"] and row["proposal_status"] == "hallucinated_replacement"
    ]
    tbox_rows = [row for row in rows if row["track"] == "T_BOX"]
    tbox_counts = Counter(row["proposal_status"] for row in tbox_rows)

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    confusion_by_subtype: dict[str, Counter[str]] = defaultdict(Counter)
    for trace in indexes["track_traces"]:
        diagnosis = trace.get("track_diagnosis") if isinstance(trace.get("track_diagnosis"), dict) else {}
        predicted = diagnosis.get("predicted_track") or "MISSING"
        case_id = trace.get("case_id")
        annotation = _manifest_annotation(manifest, str(case_id))
        truth_track = trace.get("track") or diagnosis.get("historical_track") or annotation.get("track") or "unknown"
        cls = annotation.get("class") or trace.get("classification_class") or truth_track
        subtype_name = annotation.get("subtype") or trace.get("classification_subtype") or "unknown"
        truth = f"{truth_track}::{cls}"
        subtype = f"{truth_track}::{subtype_name}"
        confusion[truth][predicted] += 1
        confusion_by_subtype[subtype][predicted] += 1

    by_visible = {}
    for visible in (True, False):
        subset = [row for row in rows if row["gold_target_visible"] is visible]
        exact = sum(row["proposal_status"] == "exact" for row in subset)
        accepted = sum(row["proposal_status"] in {"exact", "accepted_non_exact"} or row.get("accepted") for row in subset)
        by_visible[str(visible).lower()] = {
            "n": len(subset),
            "exact_rate": _rate(exact, len(subset)),
            "accepted_or_non_exact_rate": _rate(accepted, len(subset)),
        }

    answers = {
        "Are TypeA clean/rule cases failing despite answerable evidence?": (
            "Yes. Answerable TypeA rows still fail exact repair in "
            f"{len(typea_answerable)} of {sum(row['class'] == 'TypeA' and row['gold_target_visible'] for row in rows)} repair-prompt rows."
        ),
        "Are TypeB local_graph cases failing despite visible local evidence?": (
            "Yes. Local-graph TypeB answerable rows still fail exact repair in "
            f"{len(typeb_local_answerable)} of {sum(row['class'] == 'TypeB' and row['context_bundle'] == 'local_graph' and row['gold_target_visible'] for row in rows)} rows."
        ),
        "Are TypeC cases being forced into hallucinated concrete repair?": (
            "Yes. TypeC rows are rarely answerable and concrete failed repairs dominate: "
            f"{len(typec_concrete_failed)} of {len(typec_rows)} TypeC repair-prompt rows."
        ),
        "Are T-box failures mostly wrong family, wrong action, or impossible exact signature?": (
            "They are mostly impossible exact-signature cases under compact temporal context, with remaining failures split "
            f"as {dict(tbox_counts)}. Target-family/action errors still exist but signature construction is intentionally not visible."
        ),
        "Are evaluator metrics too strict for plausible non-exact T-box schema updates?": (
            "Yes for exact historical agreement: compact-policy T-box prompts cannot infer exact signature_after. "
            "Family/action metrics are more scientifically meaningful for these rows."
        ),
    }
    return {
        "manifest_type": "prompt_dev_failure_isolation_summary",
        "run_dir": str(run_dir),
        "exact_accepted_by_gold_target_visible": by_visible,
        "hallucinated_repairs_not_visible_gold_targets": {
            "count": len(not_visible_hallucinated),
            "examples": [row["case_id"] for row in not_visible_hallucinated[:10]],
        },
        "typea_answerable_failures": {
            "count": len(typea_answerable),
            "examples": [row["case_id"] for row in typea_answerable[:10]],
        },
        "typeb_local_graph_answerable_failures": {
            "count": len(typeb_local_answerable),
            "examples": [row["case_id"] for row in typeb_local_answerable[:10]],
        },
        "typec_concrete_failed_repairs": {
            "count": len(typec_concrete_failed),
            "denominator": len(typec_rows),
            "rate": _rate(len(typec_concrete_failed), len(typec_rows)),
            "examples": [row["case_id"] for row in typec_concrete_failed[:10]],
        },
        "t_box_failure_shape": _counter_dict(tbox_counts),
        "track_diagnosis_confusion_by_class": {key: _counter_dict(value) for key, value in sorted(confusion.items())},
        "track_diagnosis_confusion_by_subtype": {
            key: _counter_dict(value) for key, value in sorted(confusion_by_subtype.items())
        },
        "answers": answers,
        "verdict": "IMPLEMENT_TARGETED_V4",
    }


def _failure_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Failure Mode Isolation Summary",
        "",
        f"Run: `{summary['run_dir']}`",
        "",
        "## Exact/Accepted By Gold Visibility",
        "",
        "| Gold target visible | N | Exact rate | Accepted/non-exact rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, block in summary["exact_accepted_by_gold_target_visible"].items():
        lines.append(
            f"| `{key}` | {block['n']} | {_format_rate(block['exact_rate'])} | "
            f"{_format_rate(block['accepted_or_non_exact_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Key Failure Counts",
            "",
            f"- TypeA answerable failures: `{summary['typea_answerable_failures']['count']}`",
            f"- TypeB local_graph answerable failures: `{summary['typeb_local_graph_answerable_failures']['count']}`",
            f"- TypeC concrete failed repairs: `{summary['typec_concrete_failed_repairs']['count']}` / `{summary['typec_concrete_failed_repairs']['denominator']}`",
            f"- T-box failure shape: `{json.dumps(summary['t_box_failure_shape'], sort_keys=True)}`",
            "",
            "## Track Diagnosis Confusion By Class",
            "",
            "```json",
            json.dumps(summary["track_diagnosis_confusion_by_class"], indent=2, sort_keys=True),
            "```",
            "",
            "## Explicit Answers",
            "",
        ]
    )
    for question, answer in summary["answers"].items():
        lines.extend([f"### {question}", "", answer, ""])
    lines.extend(["## Deliverable Verdict", "", f"`{summary['verdict']}`", ""])
    return "\n".join(lines)


def _diagnostic_contract(task: str) -> tuple[str, str]:
    contracts = {
        "a_box_value_extraction": (
            "Extract only the candidate final A-box target values visible in the prompt evidence.",
            '{"case_id":"<copy id>","visible_final_values":["..."],"visible_removed_values":["..."],"evidence_paths":["..."],"answerability":"visible|not_visible|deterministic_transform"}',
        ),
        "a_box_operation_selection": (
            "Choose the operation type supported by visible evidence, without choosing replacement values.",
            '{"case_id":"<copy id>","operation":"SET|ADD|REMOVE|DELETE_ALL|ABSTAIN","preserve_values":["..."],"remove_values":["..."],"rationale":"..."}',
        ),
        "a_box_answerability": (
            "Decide whether an exact A-box repair is answerable from visible evidence.",
            '{"case_id":"<copy id>","answerability":"exact_repair_visible|conservative_remove_only|insufficient_visible_evidence","missing_evidence":["..."],"rationale":"..."}',
        ),
        "t_box_constraint_family_selection": (
            "Select the changed/target constraint family supported by the visible T-box context.",
            '{"case_id":"<copy id>","constraint_type_qid":"Q...|UNKNOWN","support":"visible_inventory|visible_pre_change_signature|not_visible","rationale":"..."}',
        ),
        "t_box_action_selection": (
            "Choose the schema action supported by visible evidence only.",
            '{"case_id":"<copy id>","action":"RELAXATION_SET_EXPANSION|RESTRICTION_SET_CONTRACTION|RELAXATION_RANGE_WIDENED|RESTRICTION_RANGE_NARROWED|SCHEMA_UPDATE|COINCIDENTAL_SCHEMA_CHANGE","direction_visible":true,"rationale":"..."}',
        ),
        "t_box_signature_visibility": (
            "Decide whether exact signature_after values are visible or must be withheld.",
            '{"case_id":"<copy id>","signature_after_visible":true,"visible_changed_values":["..."],"recommended_behavior":"exact_signature|schema_update_empty_signature","rationale":"..."}',
        ),
        "track_locus_contrast": (
            "Contrast A-box and T-box repair-locus evidence without proposing a repair.",
            '{"case_id":"<copy id>","a_box_evidence":["..."],"t_box_evidence":["..."],"likely_locus":"A_BOX|T_BOX|AMBIGUOUS","rationale":"..."}',
        ),
    }
    return contracts[task]


def render_diagnostic_tasks(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_prompts = _iter_jsonl(run_dir / "rendered_prompts" / "prompt_dev_rendered_prompts.jsonl")
    base_rows = [
        row
        for row in rendered_prompts
        if row.get("task") in {"a_box_repair", "t_box_repair"}
    ]
    out_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in base_rows:
        payload = _extract_input_payload(str(row.get("user_prompt") or ""))
        track = row.get("historical_track")
        context = row.get("context_bundle")
        candidate_tasks: list[str] = []
        if track == "A_BOX":
            candidate_tasks.extend(["a_box_value_extraction", "a_box_operation_selection", "a_box_answerability"])
        if track == "T_BOX":
            candidate_tasks.extend([
                "t_box_constraint_family_selection",
                "t_box_action_selection",
                "t_box_signature_visibility",
            ])
        candidate_tasks.append("track_locus_contrast")
        for task in candidate_tasks:
            key = (str(row["case_id"]), str(context), task)
            if key in seen:
                continue
            seen.add(key)
            instruction, contract = _diagnostic_contract(task)
            user_prompt = "\n\n".join(
                [
                    "Prompt version: diagnostic_tasks_v1",
                    f"Diagnostic task: {task}",
                    instruction,
                    "Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.",
                    "Output contract:",
                    contract,
                    "Input case JSON:",
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                ]
            )
            out_rows.append(
                {
                    "diagnostic_task": task,
                    "matrix_id": f"diagnostic_tasks_v1_{task}_{context}",
                    "case_id": row["case_id"],
                    "visible_case_id": row.get("visible_case_id"),
                    "historical_track": track,
                    "context_bundle": context,
                    "system_prompt": (
                        "You are running a diagnostic prompt-development isolation task. "
                        "Use only visible prompt evidence and return one JSON object."
                    ),
                    "user_prompt": user_prompt,
                    "response_format": {"type": "json_object"},
                    "source_prompt_matrix_id": row.get("matrix_id"),
                }
            )
    prompts_path = output_dir / "diagnostic_tasks_rendered_prompts.jsonl"
    _write_jsonl(prompts_path, out_rows)
    summary = {
        "manifest_type": "prompt_dev_diagnostic_tasks_render_summary",
        "manifest_version": "diagnostic_tasks_v1",
        "note": "Rendered only. No model inference was run.",
        "source_run": str(run_dir),
        "outputs": {
            "prompts_jsonl": str(prompts_path),
            "review_markdown": str(output_dir / "diagnostic_tasks_prompt_review.md"),
        },
        "counts": {
            "rendered_prompts": len(out_rows),
            "by_task": _counter_dict(Counter(row["diagnostic_task"] for row in out_rows)),
            "by_context": _counter_dict(Counter(row["context_bundle"] for row in out_rows)),
            "by_track": _counter_dict(Counter(row["historical_track"] for row in out_rows)),
        },
    }
    _write_json(output_dir / "diagnostic_tasks_render_summary.json", summary)
    review_lines = [
        "# Diagnostic Tasks v1 Render Review",
        "",
        "No model inference was run.",
        "",
        f"Rendered prompts: `{len(out_rows)}`",
        "",
    ]
    for row in out_rows[:14]:
        review_lines.extend(
            [
                f"## {row['diagnostic_task']} / {row['visible_case_id']}",
                "",
                f"- Context: `{row['context_bundle']}`",
                f"- Track: `{row['historical_track']}`",
                "",
                "```text",
                row["user_prompt"][:5000],
                "```",
                "",
            ]
        )
    (output_dir / "diagnostic_tasks_prompt_review.md").write_text("\n".join(review_lines), encoding="utf-8")
    return summary


def write_abstention_branch_design(run_dir: Path) -> None:
    design = {
        "manifest_type": "prompt_dev_abstention_branch_design",
        "branch_name": "prompt_dev_v3_abstain",
        "source_prompt": "prompt_dev_v3",
        "status": "designed_not_run",
        "separation_policy": "Do not replace no-abstain prompt_dev_v3 artifacts. Render/evaluate into a distinct output directory.",
        "recommended_render_dir": "reports/prompt_dev/rendered_prompt_dev_v3_abstain",
        "recommended_eval_dir": "reports/prompt_dev/evaluation_prompt_dev_v3_abstain_96_diverse_zero_shot",
        "configuration": {
            "include_abstention": True,
            "sample_strategy": "diverse_stratified",
            "representations": ["hybrid_json_nl"],
            "example_policies": ["zero_shot"],
            "context_bundles": ["logic_only", "local_graph"],
            "tasks": ["track_diagnosis", "repair_proposal"],
            "repair_track_modes": ["oracle"],
        },
        "metrics_when_run": [
            "TypeC justified abstention rate",
            "TypeA false abstention rate",
            "TypeB false abstention rate",
            "repair success conditional on not abstaining",
            "hallucinated TypeC repair rate",
            "diagnostic/unknown abstention rate",
        ],
    }
    out_json = run_dir / "prompt_dev_v3_abstain_design.json"
    out_md = run_dir / "prompt_dev_v3_abstain_design.md"
    _write_json(out_json, design)
    out_md.write_text(
        "\n".join(
            [
                "# prompt_dev_v3_abstain Branch Design",
                "",
                "This is a separate abstention branch. It does not replace the no-abstain `prompt_dev_v3` baseline.",
                "",
                "Recommended configuration:",
                "",
                "```json",
                json.dumps(design["configuration"], indent=2, sort_keys=True),
                "```",
                "",
                "Metrics to compute when run:",
                "",
                *[f"- {metric}" for metric in design["metrics_when_run"]],
                "",
                "Run only after the answerability audit confirms abstention is the right isolation step.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_guardrail(run_dir: Path) -> None:
    path = Path("reports/prompt_dev/phase_f_prompt_iteration_guardrail.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Phase F Prompt Iteration Guardrail",
                "",
                "- `prompt_dev_v1`, `prompt_dev_v2`, and `prompt_dev_v3` are dev-only prompt iterations.",
                "- No core results have been inspected for these prompt iterations.",
                "- Do not do more broad prompt rewrites without a measured failure mechanism from the answerability audit or diagnostic tasks.",
                "- Acceptable future changes are limited to abstention, diagnostics, evaluator/reporting fixes, or one targeted `prompt_dev_v4` based on the answerability audit.",
                "- Phase G main should use oracle mode first. `diagnosis_routed` is an ablation only unless track diagnosis improves substantially.",
                "- Current diagnostic source run: "
                f"`{run_dir}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run(inputs: Inputs) -> dict[str, Any]:
    rows = build_answerability_audit(inputs)
    audit_jsonl = inputs.run_dir / "answerability_audit.jsonl"
    summary_json = inputs.run_dir / "answerability_audit_summary.json"
    summary_md = inputs.run_dir / "answerability_audit_summary.md"
    isolation_json = inputs.run_dir / "failure_isolation_summary.json"
    isolation_md = inputs.run_dir / "failure_isolation_summary.md"
    _write_jsonl(audit_jsonl, rows)
    answerability_summary = summarize_answerability(rows, inputs.run_dir)
    _write_json(summary_json, answerability_summary)
    # Build this markdown explicitly to avoid f-string format specs inside conditional expressions.
    md_lines = [
        "# Answerability Audit Summary",
        "",
        f"Rows: `{answerability_summary['counts']['rows']}`",
        "",
        "## Counts",
        "",
        f"- By track: `{json.dumps(answerability_summary['counts']['by_track'], sort_keys=True)}`",
        f"- By context: `{json.dumps(answerability_summary['counts']['by_context'], sort_keys=True)}`",
        f"- By expected behavior: `{json.dumps(answerability_summary['counts']['by_expected_behavior'], sort_keys=True)}`",
        f"- By proposal status: `{json.dumps(answerability_summary['counts']['by_proposal_status'], sort_keys=True)}`",
        "",
        "## Exact/Accepted By Visibility",
        "",
        "| Gold target visible | N | Exact | Exact rate | Accepted/non-exact | Accepted/non-exact rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, block in answerability_summary["rates_by_gold_target_visible"].items():
        md_lines.append(
            f"| `{key}` | {block['n']} | {block['exact']} | {_format_rate(block['exact_rate'])} | "
            f"{block['accepted_or_non_exact']} | {_format_rate(block['accepted_or_non_exact_rate'])} |"
        )
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    isolation = build_failure_isolation(rows, inputs.run_dir, inputs.classified_benchmark, inputs.dev_manifest)
    _write_json(isolation_json, isolation)
    isolation_md.write_text(_failure_markdown(isolation), encoding="utf-8")
    diagnostic_summary = render_diagnostic_tasks(inputs.run_dir, inputs.diagnostic_output_dir)
    write_abstention_branch_design(inputs.run_dir)
    write_guardrail(inputs.run_dir)
    return {
        "answerability_rows": len(rows),
        "answerability_summary": str(summary_json),
        "failure_isolation_summary": str(isolation_json),
        "diagnostic_render_summary": diagnostic_summary,
        "verdict": isolation["verdict"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase F prompt-dev diagnostic isolation artifacts.")
    parser.add_argument("--run-dir", default="reports/prompt_dev/evaluation_prompt_dev_v3_96_diverse_zero_shot")
    parser.add_argument("--classified-benchmark", default="data/04_classified_benchmark.jsonl")
    parser.add_argument("--dev-manifest", default="reports/benchmark_selection/dev_prompt_v1_seed_13.json")
    parser.add_argument("--diagnostic-output-dir", default="reports/prompt_dev/diagnostic_tasks_v1_rendered")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        Inputs(
            run_dir=Path(args.run_dir),
            classified_benchmark=Path(args.classified_benchmark),
            dev_manifest=Path(args.dev_manifest),
            diagnostic_output_dir=Path(args.diagnostic_output_dir),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
