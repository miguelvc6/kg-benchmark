from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from guardian.common import PatchValidationError, normalize_pid, normalize_qid


REPAIR_OP_TO_CODE = {
    "CONSTRAINT_REMOVE": "C_MINUS",
    "CONSTRAINT_DEPRECATE": "C_D",
    "CONSTRAINT_ADD": "C_PLUS",
    "CONSTRAINT_TYPE_REPLACE": "C_REPLACE",
    "CONSTRAINT_QUALIFIER_ADD": "CQ_PLUS",
    "CONSTRAINT_QUALIFIER_REMOVE": "CQ_MINUS",
    "CONSTRAINT_QUALIFIER_REPLACE": "CQ_REPLACE",
    "CLASS_HIERARCHY_ADD": "SUBCLASS_PLUS",
    "EXCEPTION_ADD": "E_PLUS",
    "OTHER_TBOX_UPDATE": "OTHER",
}

CAUSAL_SUBTYPES = {
    "RELAXATION_SET_EXPANSION",
    "RESTRICTION_SET_CONTRACTION",
    "RELAXATION_RANGE_WIDENED",
    "RESTRICTION_RANGE_NARROWED",
    "SCHEMA_UPDATE",
}
NO_CAUSAL_SUBTYPES = {"COINCIDENTAL_SCHEMA_CHANGE"}

SUMMARY_REQUIRED_KEYS = [
    "selected_records",
    "selected_tbox_records",
    "gold_extracted",
    "unsupported_count",
    "unsupported_case_ids",
    "by_schema_decision",
    "by_repair_op",
    "by_taxonomy_code",
    "by_constraint_type_qid",
    "by_qualifier_property_id",
    "by_evidence_level",
    "value_delta_available_count",
    "empty_repairs_count",
    "class_hierarchy_delta_supported",
    "exception_delta_supported",
]


@dataclass
class ExtractionResult:
    patches: list[dict[str, Any]]
    summary: dict[str, Any]
    unsupported_case_ids: list[str] = field(default_factory=list)


def load_selection_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("selected_case_ids"), list):
        raise ValueError(f"Selection manifest {path} must contain selected_case_ids.")
    return manifest


def selected_case_ids(manifest: dict[str, Any]) -> list[str]:
    return [str(case_id) for case_id in manifest.get("selected_case_ids", [])]


def iter_selected_records(classified_benchmark: str | Path, selected_ids: Iterable[str]) -> Iterable[dict[str, Any]]:
    remaining = set(selected_ids)
    with Path(classified_benchmark).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not remaining:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {classified_benchmark}:{line_number}") from exc
            case_id = record.get("id")
            if isinstance(case_id, str) and case_id in remaining:
                remaining.remove(case_id)
                yield record


def is_tbox_record(record: dict[str, Any]) -> bool:
    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    return record.get("track") == "T_BOX" or classification.get("class") == "T_BOX"


def extract_selected_tbox_gold(
    *,
    classified_benchmark: str | Path,
    selection_manifest: str | Path,
    require_coverage: bool = False,
) -> ExtractionResult:
    manifest = load_selection_manifest(selection_manifest)
    case_ids = selected_case_ids(manifest)
    annotations = manifest.get("case_annotations") if isinstance(manifest.get("case_annotations"), dict) else {}
    patches: list[dict[str, Any]] = []
    unsupported_case_ids: list[str] = []
    seen_selected: set[str] = set()
    selected_tbox_records = 0

    for record in iter_selected_records(classified_benchmark, case_ids):
        case_id = record.get("id")
        if isinstance(case_id, str):
            seen_selected.add(case_id)
        if not is_tbox_record(record):
            continue
        selected_tbox_records += 1
        annotation = annotations.get(case_id, {}) if isinstance(case_id, str) else {}
        patch = gold_patch_for_record(record, annotation=annotation)
        if patch is None:
            if isinstance(case_id, str):
                unsupported_case_ids.append(case_id)
            continue
        patches.append(patch)

    missing_selected = sorted(set(case_ids) - seen_selected)
    unsupported_case_ids.extend(missing_selected)
    summary = summarize_patches(
        patches,
        selected_records=len(case_ids),
        selected_tbox_records=selected_tbox_records,
        unsupported_case_ids=sorted(set(unsupported_case_ids)),
    )
    if require_coverage and summary["unsupported_count"] > 0:
        raise CoverageError("T-box taxonomy patch gold coverage is incomplete.", summary=summary)
    return ExtractionResult(patches=patches, summary=summary, unsupported_case_ids=summary["unsupported_case_ids"])


class CoverageError(RuntimeError):
    def __init__(self, message: str, *, summary: dict[str, Any]) -> None:
        super().__init__(message)
        self.summary = summary


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gold_patch_for_record(record: dict[str, Any], *, annotation: dict[str, Any] | None = None) -> dict[str, Any] | None:
    annotation = annotation or {}
    case_id = _string(record.get("id"))
    pid = _normalize_pid_or_none(record.get("property"))
    if case_id is None or pid is None:
        return None

    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    subtype = _string(classification.get("subtype"))
    schema_decision = _schema_decision_for_subtype(subtype)
    constraint_delta = _constraint_delta(record)
    target_qid = _select_target_constraint_qid(record, annotation=annotation)
    if target_qid is None:
        return None

    repairs: list[dict[str, Any]]
    full_signature_repairs = _repairs_from_signature_delta(constraint_delta, target_qid=target_qid)
    if full_signature_repairs is not None:
        repairs = full_signature_repairs if schema_decision == "CAUSAL_SCHEMA_REPAIR" else []
    else:
        repairs = _repairs_from_diagnostics(record, target_qid=target_qid, schema_decision=schema_decision)

    if schema_decision == "CAUSAL_SCHEMA_REPAIR" and not repairs:
        repairs = [
            _repair_entry(
                "OTHER_TBOX_UPDATE",
                target_qid,
                qualifier_property_id=None,
                evidence_level="FAMILY_ONLY",
            )
        ]

    return {
        "case_id": case_id,
        "schema_decision": schema_decision,
        "target": {"pid": pid, "constraint_type_qid": target_qid},
        "repairs": repairs,
        "rationale": _rationale(record, schema_decision=schema_decision, repairs=repairs),
        "provenance": _provenance(record, pid=pid, target_qid=target_qid),
        "uncertainty": _uncertainty(classification, schema_decision=schema_decision),
    }


def summarize_patches(
    patches: list[dict[str, Any]],
    *,
    selected_records: int,
    selected_tbox_records: int,
    unsupported_case_ids: list[str],
) -> dict[str, Any]:
    by_schema_decision: Counter[str] = Counter()
    by_repair_op: Counter[str] = Counter()
    by_taxonomy_code: Counter[str] = Counter()
    by_constraint_type_qid: Counter[str] = Counter()
    by_qualifier_property_id: Counter[str] = Counter()
    by_evidence_level: Counter[str] = Counter()
    value_delta_available_count = 0
    empty_repairs_count = 0

    for patch in patches:
        by_schema_decision.update([patch["schema_decision"]])
        by_constraint_type_qid.update([patch["target"]["constraint_type_qid"]])
        repairs = patch.get("repairs") if isinstance(patch.get("repairs"), list) else []
        if not repairs:
            empty_repairs_count += 1
        for repair in repairs:
            by_repair_op.update([repair["repair_op"]])
            by_taxonomy_code.update([repair["taxonomy_code"]])
            by_constraint_type_qid.update([repair["constraint_type_qid"]])
            qualifier_property_id = repair.get("qualifier_property_id")
            if isinstance(qualifier_property_id, str):
                by_qualifier_property_id.update([qualifier_property_id])
            by_evidence_level.update([repair["evidence_level"]])
            if repair["evidence_level"] == "VALUE_DELTA_VISIBLE":
                value_delta_available_count += 1

    summary = {
        "selected_records": selected_records,
        "selected_tbox_records": selected_tbox_records,
        "gold_extracted": len(patches),
        "unsupported_count": len(unsupported_case_ids),
        "unsupported_case_ids": unsupported_case_ids,
        "by_schema_decision": dict(sorted(by_schema_decision.items())),
        "by_repair_op": dict(sorted(by_repair_op.items())),
        "by_taxonomy_code": dict(sorted(by_taxonomy_code.items())),
        "by_constraint_type_qid": dict(sorted(by_constraint_type_qid.items())),
        "by_qualifier_property_id": dict(sorted(by_qualifier_property_id.items())),
        "by_evidence_level": dict(sorted(by_evidence_level.items())),
        "value_delta_available_count": value_delta_available_count,
        "empty_repairs_count": empty_repairs_count,
        "class_hierarchy_delta_supported": False,
        "exception_delta_supported": False,
    }
    missing_keys = set(SUMMARY_REQUIRED_KEYS) - set(summary)
    if missing_keys:
        raise AssertionError(f"Gold summary missing keys: {sorted(missing_keys)}")
    return summary


def _schema_decision_for_subtype(subtype: str | None) -> str:
    if subtype in CAUSAL_SUBTYPES:
        return "CAUSAL_SCHEMA_REPAIR"
    if subtype in NO_CAUSAL_SUBTYPES:
        return "NO_CAUSAL_SCHEMA_REPAIR"
    return "UNCLEAR_SCHEMA_EVIDENCE"


def _constraint_delta(record: dict[str, Any]) -> dict[str, Any]:
    repair_target = record.get("repair_target") if isinstance(record.get("repair_target"), dict) else {}
    delta = repair_target.get("constraint_delta") if isinstance(repair_target.get("constraint_delta"), dict) else {}
    return delta


def _select_target_constraint_qid(record: dict[str, Any], *, annotation: dict[str, Any]) -> str | None:
    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    diagnostics = classification.get("diagnostics") if isinstance(classification.get("diagnostics"), dict) else {}
    summary = diagnostics.get("tbox_diff_summary") if isinstance(diagnostics.get("tbox_diff_summary"), dict) else {}
    delta = _constraint_delta(record)
    candidates: list[Any] = [
        summary.get("target_constraint_qid"),
        classification.get("decision_constraint_type_qid"),
        annotation.get("decision_constraint_type_qid"),
        annotation.get("constraint_family"),
    ]
    candidates.extend(_ensure_list(summary.get("changed_constraint_qids_all")))
    candidates.extend(_ensure_list(delta.get("changed_constraint_types")))
    for candidate in candidates:
        qid = _normalize_qid_or_none(candidate)
        if qid is not None:
            return qid
    return None


def _repairs_from_signature_delta(delta: dict[str, Any], *, target_qid: str) -> list[dict[str, Any]] | None:
    before_raw = delta.get("signature_before") or delta.get("old_constraints")
    after_raw = delta.get("signature_after") or delta.get("new_constraints")
    if not isinstance(before_raw, list) or not isinstance(after_raw, list):
        return None
    before = normalize_signature_families(before_raw)
    after = normalize_signature_families(after_raw)
    before_qids = set(before)
    after_qids = set(after)
    removed_qids = sorted(before_qids - after_qids)
    added_qids = sorted(after_qids - before_qids)
    repairs: list[dict[str, Any]] = []

    if len(removed_qids) == 1 and len(added_qids) == 1:
        repairs.append(
            _repair_entry(
                "CONSTRAINT_TYPE_REPLACE",
                added_qids[0],
                old_value=removed_qids[0],
                new_value=added_qids[0],
                evidence_level="OPERATION_VISIBLE",
            )
        )
    else:
        for qid in removed_qids:
            repairs.append(_repair_entry("CONSTRAINT_REMOVE", qid, evidence_level="OPERATION_VISIBLE"))
        for qid in added_qids:
            repairs.append(_repair_entry("CONSTRAINT_ADD", qid, evidence_level="OPERATION_VISIBLE"))

    for qid in sorted(before_qids & after_qids):
        before_family = before[qid]
        after_family = after[qid]
        repairs.extend(_qualifier_repairs(qid, before_family["qualifiers"], after_family["qualifiers"]))
        before_ranks = set(before_family["ranks"])
        after_ranks = set(after_family["ranks"])
        if "deprecated" in after_ranks and "deprecated" not in before_ranks:
            repairs.append(_repair_entry("CONSTRAINT_DEPRECATE", qid, rank_after="deprecated", evidence_level="OPERATION_VISIBLE"))
        elif before_ranks != after_ranks or set(before_family["snaktypes"]) != set(after_family["snaktypes"]):
            repairs.append(
                _repair_entry(
                    "OTHER_TBOX_UPDATE",
                    qid,
                    rank_after=_single_or_none(after_ranks),
                    snaktype_after=_single_or_none(set(after_family["snaktypes"])),
                    evidence_level="OPERATION_VISIBLE",
                )
            )

    if not repairs and target_qid in before_qids | after_qids and before != after:
        repairs.append(_repair_entry("OTHER_TBOX_UPDATE", target_qid, evidence_level="FAMILY_ONLY"))
    return _canonical_repairs(repairs)


def normalize_signature_families(signature: list[Any]) -> dict[str, dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    for entry in signature:
        if not isinstance(entry, dict):
            continue
        qid = _normalize_qid_or_none(entry.get("constraint_qid") or entry.get("qid"))
        if qid is None:
            continue
        family = families.setdefault(qid, {"snaktypes": set(), "ranks": set(), "qualifiers": {}})
        snaktype = _string(entry.get("snaktype"))
        if snaktype:
            family["snaktypes"].add(snaktype.upper())
        rank = _string(entry.get("rank"))
        if rank:
            family["ranks"].add(rank.strip().lower())
        qualifiers = entry.get("qualifiers")
        if isinstance(qualifiers, dict):
            qualifiers = [{"property_id": key, "values": value} for key, value in qualifiers.items()]
        for qualifier in _ensure_list(qualifiers):
            if not isinstance(qualifier, dict):
                continue
            pid = _normalize_pid_or_none(qualifier.get("property_id") or qualifier.get("pid"))
            if pid is None:
                continue
            values = qualifier.get("values")
            if values is None and "value" in qualifier:
                values = [qualifier.get("value")]
            normalized_values = family["qualifiers"].setdefault(pid, set())
            for value in _ensure_list(values):
                normalized = _normalize_value(value)
                if normalized is not None:
                    normalized_values.add(normalized)
    return {
        qid: {
            "snaktypes": sorted(data["snaktypes"]),
            "ranks": sorted(data["ranks"]),
            "qualifiers": {pid: sorted(values, key=_value_sort_key) for pid, values in sorted(data["qualifiers"].items())},
        }
        for qid, data in sorted(families.items())
    }


def _qualifier_repairs(
    qid: str,
    before_qualifiers: dict[str, list[Any]],
    after_qualifiers: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    for pid in sorted(set(before_qualifiers) | set(after_qualifiers)):
        before_values = set(before_qualifiers.get(pid, []))
        after_values = set(after_qualifiers.get(pid, []))
        added = sorted(after_values - before_values, key=_value_sort_key)
        removed = sorted(before_values - after_values, key=_value_sort_key)
        if added and removed:
            repairs.append(
                _repair_entry(
                    "CONSTRAINT_QUALIFIER_REPLACE",
                    qid,
                    qualifier_property_id=pid,
                    added_values=added,
                    removed_values=removed,
                    old_value=removed[0],
                    new_value=added[0],
                    evidence_level="VALUE_DELTA_VISIBLE",
                )
            )
        elif added:
            repairs.append(
                _repair_entry(
                    "CONSTRAINT_QUALIFIER_ADD",
                    qid,
                    qualifier_property_id=pid,
                    added_values=added,
                    new_value=added[0],
                    evidence_level="VALUE_DELTA_VISIBLE",
                )
            )
        elif removed:
            repairs.append(
                _repair_entry(
                    "CONSTRAINT_QUALIFIER_REMOVE",
                    qid,
                    qualifier_property_id=pid,
                    removed_values=removed,
                    old_value=removed[0],
                    evidence_level="VALUE_DELTA_VISIBLE",
                )
            )
    return repairs


def _repairs_from_diagnostics(
    record: dict[str, Any],
    *,
    target_qid: str,
    schema_decision: str,
) -> list[dict[str, Any]]:
    if schema_decision != "CAUSAL_SCHEMA_REPAIR":
        return []
    summary = _tbox_diff_summary(record)
    qualifier_properties = [_normalize_pid_or_none(value) for value in _summary_list(summary, "semantic_changed_qualifier_properties", "changed_qualifier_properties")]
    qualifier_properties = sorted({pid for pid in qualifier_properties if pid is not None})
    added_values = [_normalize_value(value) for value in _summary_list(summary, "semantic_added_values", "added_values")]
    removed_values = [_normalize_value(value) for value in _summary_list(summary, "semantic_removed_values", "removed_values")]
    added_values = sorted({value for value in added_values if value is not None}, key=_value_sort_key)
    removed_values = sorted({value for value in removed_values if value is not None}, key=_value_sort_key)

    if not qualifier_properties:
        return [_repair_entry("OTHER_TBOX_UPDATE", target_qid, evidence_level="FAMILY_ONLY")]

    repairs: list[dict[str, Any]] = []
    value_delta_is_unambiguous = len(qualifier_properties) == 1
    evidence_level = "VALUE_DELTA_VISIBLE" if value_delta_is_unambiguous and (added_values or removed_values) else "OPERATION_VISIBLE"
    visible_added = added_values if value_delta_is_unambiguous else []
    visible_removed = removed_values if value_delta_is_unambiguous else []
    if added_values and removed_values:
        repair_op = "CONSTRAINT_QUALIFIER_REPLACE"
    elif added_values:
        repair_op = "CONSTRAINT_QUALIFIER_ADD"
    elif removed_values:
        repair_op = "CONSTRAINT_QUALIFIER_REMOVE"
    else:
        repair_op = "OTHER_TBOX_UPDATE"
        evidence_level = "OPERATION_VISIBLE"

    for qualifier_pid in qualifier_properties:
        repairs.append(
            _repair_entry(
                repair_op,
                target_qid,
                qualifier_property_id=qualifier_pid if repair_op != "OTHER_TBOX_UPDATE" else qualifier_pid,
                added_values=visible_added,
                removed_values=visible_removed,
                old_value=visible_removed[0] if visible_removed else None,
                new_value=visible_added[0] if visible_added else None,
                evidence_level=evidence_level,
            )
        )
    return _canonical_repairs(repairs)


def _repair_entry(
    repair_op: str,
    constraint_type_qid: str,
    *,
    qualifier_property_id: str | None = None,
    added_values: list[Any] | None = None,
    removed_values: list[Any] | None = None,
    old_value: Any = None,
    new_value: Any = None,
    rank_after: str | None = None,
    snaktype_after: str | None = None,
    evidence_level: str,
) -> dict[str, Any]:
    return {
        "repair_op": repair_op,
        "taxonomy_code": REPAIR_OP_TO_CODE[repair_op],
        "constraint_type_qid": constraint_type_qid,
        "qualifier_property_id": qualifier_property_id,
        "added_values": added_values or [],
        "removed_values": removed_values or [],
        "old_value": old_value,
        "new_value": new_value,
        "rank_after": rank_after,
        "snaktype_after": snaktype_after,
        "evidence_level": evidence_level,
    }


def _canonical_repairs(repairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        repairs,
        key=lambda repair: (
            repair["repair_op"],
            repair["taxonomy_code"],
            repair["constraint_type_qid"],
            repair["qualifier_property_id"] or "",
            json.dumps(repair["added_values"], sort_keys=True),
            json.dumps(repair["removed_values"], sort_keys=True),
        ),
    )


def _tbox_diff_summary(record: dict[str, Any]) -> dict[str, Any]:
    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    diagnostics = classification.get("diagnostics") if isinstance(classification.get("diagnostics"), dict) else {}
    summary = diagnostics.get("tbox_diff_summary")
    if isinstance(summary, dict):
        return summary
    for step in reversed(_ensure_list(classification.get("decision_trace"))):
        if isinstance(step, dict) and step.get("step") == "tbox_causality":
            return step
    return {}


def _summary_list(summary: dict[str, Any], preferred_key: str, fallback_key: str) -> list[Any]:
    preferred = summary.get(preferred_key)
    if isinstance(preferred, list):
        return preferred
    fallback = summary.get(fallback_key)
    return fallback if isinstance(fallback, list) else []


def _rationale(record: dict[str, Any], *, schema_decision: str, repairs: list[dict[str, Any]]) -> str:
    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    existing = _string(classification.get("rationale"))
    if schema_decision == "NO_CAUSAL_SCHEMA_REPAIR":
        return existing or "The selected T-box record is classified as a coincidental schema change."
    if schema_decision == "UNCLEAR_SCHEMA_EVIDENCE":
        return existing or "The selected T-box record has unclear schema causality."
    if repairs:
        ops = ", ".join(sorted({repair["repair_op"] for repair in repairs}))
        return existing or f"The historical T-box diagnostics support these schema repair operations: {ops}."
    return existing or "The historical T-box diagnostics support a causal schema repair."


def _provenance(record: dict[str, Any], *, pid: str, target_qid: str) -> list[dict[str, Any]]:
    repair_target = record.get("repair_target") if isinstance(record.get("repair_target"), dict) else {}
    revision_id = repair_target.get("property_revision_id")
    snippet = f"property {pid}; target constraint {target_qid}"
    if isinstance(revision_id, int):
        snippet = f"{snippet}; property revision {revision_id}"
    return [{"kind": "KG", "node_id": pid, "snippet": snippet}]


def _uncertainty(classification: dict[str, Any], *, schema_decision: str) -> dict[str, Any]:
    confidence = classification.get("confidence")
    score = {"high": 0.75, "medium": 0.5, "low": 0.25}.get(str(confidence).lower(), 0.5)
    if schema_decision == "UNCLEAR_SCHEMA_EVIDENCE":
        score = min(score, 0.25)
    return {"confidence": score, "notes": f"classification confidence: {confidence or 'unknown'}"}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            return None
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        upper = text.upper()
        if upper.startswith("Q") and upper[1:].isdigit():
            return _normalize_qid_or_none(upper)
        if upper.startswith("P") and upper[1:].isdigit():
            return _normalize_pid_or_none(upper)
        return text
    if isinstance(value, dict):
        for key in ("qid", "id", "value", "raw"):
            if key in value:
                return _normalize_value(value[key])
    return None


def _normalize_qid_or_none(value: Any) -> str | None:
    try:
        return normalize_qid(value)
    except PatchValidationError:
        return None


def _normalize_pid_or_none(value: Any) -> str | None:
    try:
        return normalize_pid(value)
    except PatchValidationError:
        return None


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _value_sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def _single_or_none(values: set[str]) -> str | None:
    return next(iter(values)) if len(values) == 1 else None
