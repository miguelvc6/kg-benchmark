from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .common import PatchValidationError, canonical_hash, canonicalize, load_schema, normalize_pid, normalize_qid


SCHEMA_DECISIONS = {
    "CAUSAL_SCHEMA_REPAIR",
    "NO_CAUSAL_SCHEMA_REPAIR",
    "UNCLEAR_SCHEMA_EVIDENCE",
}
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
TAXONOMY_CODES = set(REPAIR_OP_TO_CODE.values())
EVIDENCE_LEVELS = {"FAMILY_ONLY", "OPERATION_VISIBLE", "VALUE_DELTA_VISIBLE"}
RANK_VALUES = {"normal", "preferred", "deprecated"}
SNAKTYPE_VALUES = {"VALUE", "SOMEVALUE", "NOVALUE"}
PROVENANCE_KINDS = {"KG", "OTHER"}
PLACEHOLDER_STRINGS = {"", "none", "null", "q...", "p...", "q0", "p0"}


@dataclass(frozen=True)
class TaxonomyPatchTarget:
    pid: str
    constraint_type_qid: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {"pid": self.pid, "constraint_type_qid": self.constraint_type_qid}


@dataclass(frozen=True)
class TaxonomyPatchRepair:
    repair_op: str
    taxonomy_code: str
    constraint_type_qid: str
    qualifier_property_id: str | None = None
    added_values: list[Any] = field(default_factory=list)
    removed_values: list[Any] = field(default_factory=list)
    old_value: Any = None
    new_value: Any = None
    rank_after: str | None = None
    snaktype_after: str | None = None
    evidence_level: str = "FAMILY_ONLY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_op": self.repair_op,
            "taxonomy_code": self.taxonomy_code,
            "constraint_type_qid": self.constraint_type_qid,
            "qualifier_property_id": self.qualifier_property_id,
            "added_values": self.added_values,
            "removed_values": self.removed_values,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "rank_after": self.rank_after,
            "snaktype_after": self.snaktype_after,
            "evidence_level": self.evidence_level,
        }


@dataclass(frozen=True)
class NormalizedTBoxTaxonomyPatch:
    case_id: str
    schema_decision: str
    target: TaxonomyPatchTarget
    repairs: list[TaxonomyPatchRepair]
    rationale: str
    provenance: list[dict[str, Any]]
    uncertainty: dict[str, Any]
    canonical_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "case_id": self.case_id,
            "schema_decision": self.schema_decision,
            "target": self.target.to_dict(),
            "repairs": [repair.to_dict() for repair in self.repairs],
            "rationale": self.rationale,
            "provenance": self.provenance,
            "uncertainty": self.uncertainty,
        }
        if self.canonical_hash:
            payload["canonical_hash"] = self.canonical_hash
        return payload

    def schema_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "schema_decision": self.schema_decision,
            "target": self.target.to_dict(),
            "repairs": [repair.to_dict() for repair in self.repairs],
            "rationale": self.rationale,
            "provenance": self.provenance,
            "uncertainty": self.uncertainty,
        }


def normalize_tbox_taxonomy_patch(
    raw: Any,
    schema: Any = None,
    *,
    constraint_type_qids: Iterable[str] | None = None,
) -> NormalizedTBoxTaxonomyPatch:
    del schema
    payload = _load_payload(raw)
    allowed_constraint_qids = _normalize_allowed_qids(constraint_type_qids)
    case_id = _require_text(payload.get("case_id"), "case_id")
    schema_decision = _normalize_enum(payload.get("schema_decision"), SCHEMA_DECISIONS, "schema_decision")

    target_payload = payload.get("target")
    if not isinstance(target_payload, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "target must be an object.")
    target_qid = _normalize_target_constraint_qid(
        target_payload.get("constraint_type_qid"),
        schema_decision=schema_decision,
        allowed_qids=allowed_constraint_qids,
    )
    target = TaxonomyPatchTarget(pid=normalize_pid(target_payload.get("pid")), constraint_type_qid=target_qid)

    repairs_payload = payload.get("repairs")
    if not isinstance(repairs_payload, list):
        raise PatchValidationError("SCHEMA_VIOLATION", "repairs must be a list.")
    if schema_decision == "CAUSAL_SCHEMA_REPAIR" and not repairs_payload:
        raise PatchValidationError("SCHEMA_VIOLATION", "CAUSAL_SCHEMA_REPAIR requires at least one repair.")
    if schema_decision != "CAUSAL_SCHEMA_REPAIR" and repairs_payload is None:
        repairs_payload = []
    repairs = [_normalize_repair(repair, allowed_constraint_qids) for repair in repairs_payload]
    repairs.sort(key=_repair_sort_key)

    rationale = _require_text(payload.get("rationale"), "rationale")
    provenance = _normalize_provenance(payload.get("provenance"))
    if not provenance:
        raise PatchValidationError("SCHEMA_VIOLATION", "provenance must contain at least one entry.")
    uncertainty = _normalize_uncertainty(payload.get("uncertainty"))

    canonical_payload = {
        "case_id": case_id,
        "schema_decision": schema_decision,
        "target": target.to_dict(),
        "repairs": [repair.to_dict() for repair in repairs],
        "rationale": rationale,
        "provenance": provenance,
        "uncertainty": uncertainty,
    }
    return NormalizedTBoxTaxonomyPatch(
        case_id=case_id,
        schema_decision=schema_decision,
        target=target,
        repairs=repairs,
        rationale=rationale,
        provenance=provenance,
        uncertainty=uncertainty,
        canonical_hash=canonical_hash(canonical_payload),
    )


def _load_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PatchValidationError("INVALID_JSON", "T-box taxonomy patch is not valid JSON.") from exc
    if not isinstance(raw, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "T-box taxonomy patch must be a JSON object.")
    return dict(raw)


def _normalize_allowed_qids(constraint_type_qids: Iterable[str] | None) -> set[str] | None:
    if constraint_type_qids is None:
        return None
    return {normalize_qid(value) for value in constraint_type_qids}


def _normalize_target_constraint_qid(
    value: Any,
    *,
    schema_decision: str,
    allowed_qids: set[str] | None,
) -> str | None:
    if value is None:
        if schema_decision == "UNCLEAR_SCHEMA_EVIDENCE":
            return None
        raise PatchValidationError(
            "SCHEMA_VIOLATION",
            "target.constraint_type_qid may be null only for UNCLEAR_SCHEMA_EVIDENCE.",
        )
    qid = normalize_qid(value)
    _validate_constraint_qid(qid, allowed_qids, "target.constraint_type_qid")
    return qid


def _validate_constraint_qid(qid: str | None, allowed_qids: set[str] | None, field_name: str) -> None:
    if qid is None:
        return
    if allowed_qids is not None and qid not in allowed_qids:
        raise PatchValidationError("SCHEMA_VIOLATION", f"{field_name} is not in the allowed constraint-family set.")


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise PatchValidationError("SCHEMA_VIOLATION", f"{field_name} must be a string.")
    text = value.strip()
    if text.lower() in PLACEHOLDER_STRINGS:
        raise PatchValidationError("INVALID_VALUE", f"{field_name} cannot be empty or a placeholder.")
    return text


def _normalize_enum(value: Any, allowed: set[str], field_name: str) -> str:
    if not isinstance(value, str):
        raise PatchValidationError("SCHEMA_VIOLATION", f"{field_name} must be a string.")
    normalized = value.strip().upper()
    if normalized not in allowed:
        raise PatchValidationError("SCHEMA_VIOLATION", f"Unknown {field_name}: {value!r}")
    return normalized


def _normalize_repair(payload: Any, allowed_constraint_qids: set[str] | None) -> TaxonomyPatchRepair:
    if not isinstance(payload, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "Each repair must be an object.")
    repair_op = _normalize_enum(payload.get("repair_op"), set(REPAIR_OP_TO_CODE), "repair_op")
    taxonomy_code = _normalize_enum(payload.get("taxonomy_code"), TAXONOMY_CODES, "taxonomy_code")
    expected_code = REPAIR_OP_TO_CODE[repair_op]
    if taxonomy_code != expected_code:
        raise PatchValidationError(
            "SCHEMA_VIOLATION",
            f"taxonomy_code {taxonomy_code} does not match repair_op {repair_op}.",
        )
    qid = normalize_qid(payload.get("constraint_type_qid"))
    _validate_constraint_qid(qid, allowed_constraint_qids, "repairs[].constraint_type_qid")
    qualifier_property_id = payload.get("qualifier_property_id")
    if qualifier_property_id is not None:
        qualifier_property_id = normalize_pid(qualifier_property_id)
    added_values = _normalize_value_list(payload.get("added_values"), "added_values")
    removed_values = _normalize_value_list(payload.get("removed_values"), "removed_values")
    old_value = _normalize_nullable_value(payload.get("old_value"), "old_value")
    new_value = _normalize_nullable_value(payload.get("new_value"), "new_value")
    rank_after = _normalize_nullable_rank(payload.get("rank_after"))
    snaktype_after = _normalize_nullable_snaktype(payload.get("snaktype_after"))
    evidence_level = _normalize_enum(payload.get("evidence_level"), EVIDENCE_LEVELS, "evidence_level")
    return TaxonomyPatchRepair(
        repair_op=repair_op,
        taxonomy_code=taxonomy_code,
        constraint_type_qid=qid,
        qualifier_property_id=qualifier_property_id,
        added_values=sorted(set(added_values), key=_value_sort_key),
        removed_values=sorted(set(removed_values), key=_value_sort_key),
        old_value=old_value,
        new_value=new_value,
        rank_after=rank_after,
        snaktype_after=snaktype_after,
        evidence_level=evidence_level,
    )


def _normalize_value_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise PatchValidationError("SCHEMA_VIOLATION", f"{field_name} must be a list.")
    return [_normalize_value(item, field_name) for item in value]


def _normalize_nullable_value(value: Any, field_name: str) -> Any:
    if value is None:
        return None
    return _normalize_value(value, field_name)


def _normalize_value(value: Any, field_name: str) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PatchValidationError("INVALID_VALUE", f"{field_name} contains a non-finite number.")
        return value
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered in PLACEHOLDER_STRINGS:
            raise PatchValidationError("INVALID_VALUE", f"{field_name} contains a placeholder value.")
        upper = text.upper()
        if upper.startswith("Q") and upper[1:].isdigit():
            return normalize_qid(upper)
        if upper.startswith("P") and upper[1:].isdigit():
            return normalize_pid(upper)
        if upper in {"Q...", "P...", "Q0", "P0"}:
            raise PatchValidationError("INVALID_VALUE", f"{field_name} contains a placeholder identifier.")
        return text
    if isinstance(value, dict):
        for key in ("qid", "pid", "id", "value", "raw"):
            if key in value:
                return _normalize_value(value[key], field_name)
    raise PatchValidationError("INVALID_VALUE", f"{field_name} contains an unsupported value.")


def _normalize_nullable_rank(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PatchValidationError("SCHEMA_VIOLATION", "rank_after must be a string or null.")
    normalized = value.strip().lower()
    if normalized not in RANK_VALUES:
        raise PatchValidationError("SCHEMA_VIOLATION", f"Unknown rank_after: {value!r}")
    return normalized


def _normalize_nullable_snaktype(value: Any) -> str | None:
    if value is None:
        return None
    return _normalize_enum(value, SNAKTYPE_VALUES, "snaktype_after")


def _normalize_provenance(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PatchValidationError("SCHEMA_VIOLATION", "provenance must be a list.")
    provenance: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise PatchValidationError("SCHEMA_VIOLATION", "Each provenance entry must be an object.")
        kind = _normalize_enum(item.get("kind"), PROVENANCE_KINDS, "provenance.kind")
        entry: dict[str, Any] = {"kind": kind}
        node_id = item.get("node_id")
        if node_id is not None:
            if not isinstance(node_id, str):
                raise PatchValidationError("INVALID_ID", "provenance.node_id must be a QID, PID, or null.")
            upper = node_id.strip().upper()
            if upper.startswith("Q"):
                entry["node_id"] = normalize_qid(upper)
            elif upper.startswith("P"):
                entry["node_id"] = normalize_pid(upper)
            else:
                raise PatchValidationError("INVALID_ID", "provenance.node_id must be a QID, PID, or null.")
        snippet = item.get("snippet")
        if snippet is not None:
            entry["snippet"] = _require_text(snippet, "provenance.snippet")
        provenance.append(entry)
    return provenance


def _normalize_uncertainty(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "uncertainty must be an object.")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise PatchValidationError("SCHEMA_VIOLATION", "uncertainty.confidence must be numeric.")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise PatchValidationError("SCHEMA_VIOLATION", "uncertainty.confidence must be between 0.0 and 1.0.")
    notes = value.get("notes", "")
    if notes is None:
        notes = ""
    if not isinstance(notes, str):
        raise PatchValidationError("SCHEMA_VIOLATION", "uncertainty.notes must be a string.")
    return {"confidence": confidence, "notes": notes.strip()}


def _repair_sort_key(repair: TaxonomyPatchRepair) -> tuple[Any, ...]:
    payload = repair.to_dict()
    return (
        repair.repair_op,
        repair.taxonomy_code,
        repair.constraint_type_qid,
        repair.qualifier_property_id or "",
        canonicalize(payload["added_values"]),
        canonicalize(payload["removed_values"]),
        canonicalize(payload["old_value"]),
        canonicalize(payload["new_value"]),
        repair.rank_after or "",
        repair.snaktype_after or "",
        repair.evidence_level,
    )


def _value_sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, str(value))


__all__ = [
    "NormalizedTBoxTaxonomyPatch",
    "TaxonomyPatchRepair",
    "TaxonomyPatchTarget",
    "PatchValidationError",
    "load_schema",
    "normalize_tbox_taxonomy_patch",
]
