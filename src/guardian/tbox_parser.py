from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .common import PatchValidationError, canonical_hash, canonicalize, load_schema, normalize_pid, normalize_qid

SUPPORTED_ACTIONS = {
    "RELAXATION_RANGE_WIDENED",
    "RESTRICTION_RANGE_NARROWED",
    "RELAXATION_SET_EXPANSION",
    "RESTRICTION_SET_CONTRACTION",
    "SCHEMA_UPDATE",
    "COINCIDENTAL_SCHEMA_CHANGE",
}


@dataclass
class ReformTarget:
    pid: str
    constraint_type_qid: str

    def to_dict(self) -> dict[str, str]:
        return {"pid": self.pid, "constraint_type_qid": self.constraint_type_qid}


@dataclass
class ReformProposal:
    action: str
    signature_after: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "signature_after": self.signature_after}


@dataclass
class NormalizedReformProposal:
    case_id: str
    target: ReformTarget
    proposal: ReformProposal
    rationale: Optional[str] = None
    provenance: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    canonical_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "target": self.target.to_dict(),
            "proposal": self.proposal.to_dict(),
            "canonical_hash": self.canonical_hash,
        }
        if self.rationale is not None:
            payload["rationale"] = self.rationale
        if self.provenance:
            payload["provenance"] = self.provenance
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def _load_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PatchValidationError("INVALID_JSON", "Proposal is not valid JSON.") from exc
    if not isinstance(raw, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "T-box proposal must be a JSON object.")
    return dict(raw)


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PatchValidationError("SCHEMA_VIOLATION", f"{field_name} must be a non-empty string.")
    return value.strip()


def _normalize_signature_value(value: Any) -> str:
    if isinstance(value, bool):
        raise PatchValidationError("INVALID_VALUE", "Boolean values are not supported in signature values.")
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:
            raise PatchValidationError("INVALID_VALUE", "Non-finite numeric value is invalid.")
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise PatchValidationError("INVALID_VALUE", "Signature values cannot be empty.")
        upper = text.upper()
        if upper.startswith("Q") and upper[1:].isdigit():
            return normalize_qid(upper)
        if upper.startswith("P") and upper[1:].isdigit():
            return normalize_pid(upper)
        return text
    if isinstance(value, dict):
        if isinstance(value.get("qid"), str):
            return normalize_qid(value["qid"])
        if isinstance(value.get("id"), str):
            identifier = value["id"].strip().upper()
            if identifier.startswith("Q"):
                return normalize_qid(identifier)
            if identifier.startswith("P"):
                return normalize_pid(identifier)
        if "raw" in value:
            return _normalize_signature_value(value["raw"])
        if "value" in value:
            return _normalize_signature_value(value["value"])
    raise PatchValidationError("INVALID_VALUE", "Unsupported signature value.")


def normalize_signature_after(signature_after: Any) -> list[dict[str, Any]]:
    if not isinstance(signature_after, list):
        raise PatchValidationError("SCHEMA_VIOLATION", "proposal.signature_after must be a list.")
    normalized: list[dict[str, Any]] = []
    for entry in signature_after:
        if not isinstance(entry, dict):
            raise PatchValidationError("SCHEMA_VIOLATION", "Each signature entry must be an object.")
        constraint_qid = normalize_qid(entry.get("constraint_qid"))
        snaktype = _require_non_empty_string(entry.get("snaktype"), "snaktype").upper()
        rank = entry.get("rank")
        if rank is not None:
            rank = _require_non_empty_string(rank, "rank")
        qualifiers_payload = entry.get("qualifiers")
        if not isinstance(qualifiers_payload, list):
            raise PatchValidationError("SCHEMA_VIOLATION", "qualifiers must be a list.")
        qualifiers = []
        for qualifier in qualifiers_payload:
            if not isinstance(qualifier, dict):
                raise PatchValidationError("SCHEMA_VIOLATION", "Each qualifier must be an object.")
            property_id = normalize_pid(qualifier.get("property_id"))
            values_payload = qualifier.get("values")
            if not isinstance(values_payload, list):
                raise PatchValidationError("SCHEMA_VIOLATION", "qualifier values must be a list.")
            values = sorted({_normalize_signature_value(value) for value in values_payload})
            qualifiers.append({"property_id": property_id, "values": values})
        qualifiers.sort(key=lambda item: (item["property_id"], canonicalize(item["values"])))
        normalized.append(
            {
                "constraint_qid": constraint_qid,
                "snaktype": snaktype,
                "rank": rank,
                "qualifiers": qualifiers,
            }
        )
    normalized.sort(
        key=lambda entry: (
            entry["constraint_qid"],
            entry["snaktype"],
            entry["rank"] or "",
            canonicalize(entry["qualifiers"]),
        )
    )
    return normalized


def _normalize_provenance(items: Any) -> list[dict[str, Any]]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise PatchValidationError("SCHEMA_VIOLATION", "provenance must be a list.")
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise PatchValidationError("SCHEMA_VIOLATION", "Each provenance entry must be an object.")
        payload = dict(item)
        kind = payload.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise PatchValidationError("SCHEMA_VIOLATION", "Each provenance entry requires a non-empty kind.")
        payload["kind"] = kind.strip().upper()
        normalized.append(payload)
    return normalized


def _normalize_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "metadata must be an object.")
    return dict(metadata)


def normalize_proposal(raw: Any, schema: Any = None) -> NormalizedReformProposal:
    del schema
    payload = _load_payload(raw)
    case_id = _require_non_empty_string(payload.get("case_id"), "case_id")
    target_payload = payload.get("target")
    if not isinstance(target_payload, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "target must be an object.")
    target = ReformTarget(
        pid=normalize_pid(target_payload.get("pid")),
        constraint_type_qid=normalize_qid(target_payload.get("constraint_type_qid")),
    )

    proposal_payload = payload.get("proposal")
    if not isinstance(proposal_payload, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "proposal must be an object.")
    action = _require_non_empty_string(proposal_payload.get("action"), "proposal.action").upper()
    if action not in SUPPORTED_ACTIONS:
        raise PatchValidationError("SCHEMA_VIOLATION", f"Unsupported reform action: {action!r}")
    signature_after = normalize_signature_after(proposal_payload.get("signature_after"))

    rationale = payload.get("rationale")
    if rationale is not None:
        rationale = _require_non_empty_string(rationale, "rationale")
    provenance = _normalize_provenance(payload.get("provenance"))
    metadata = _normalize_metadata(payload.get("metadata"))

    base_payload = {
        "case_id": case_id,
        "target": target.to_dict(),
        "proposal": {
            "action": action,
            "signature_after": signature_after,
        },
    }
    if rationale is not None:
        base_payload["rationale"] = rationale
    if provenance:
        base_payload["provenance"] = provenance
    if metadata:
        base_payload["metadata"] = metadata

    return NormalizedReformProposal(
        case_id=case_id,
        target=target,
        proposal=ReformProposal(action=action, signature_after=signature_after),
        rationale=rationale,
        provenance=provenance,
        metadata=metadata,
        canonical_hash=canonical_hash(base_payload),
    )


__all__ = [
    "PatchValidationError",
    "NormalizedReformProposal",
    "ReformProposal",
    "ReformTarget",
    "SUPPORTED_ACTIONS",
    "canonicalize",
    "load_schema",
    "normalize_proposal",
    "normalize_signature_after",
]

