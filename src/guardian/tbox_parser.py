from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .common import (
    PatchValidationError,
    canonical_hash,
    canonicalize,
    load_schema,
    normalize_pid,
    normalize_provenance_payload,
    normalize_qid,
    normalize_uncertainty_payload,
)

SUPPORTED_ACTIONS = {
    "RELAXATION_RANGE_WIDENED",
    "RESTRICTION_RANGE_NARROWED",
    "RELAXATION_SET_EXPANSION",
    "RESTRICTION_SET_CONTRACTION",
    "SCHEMA_UPDATE",
    "COINCIDENTAL_SCHEMA_CHANGE",
}
KNOWN_CONSTRAINT_TYPE_QIDS = {
    "Q19474404",
    "Q21502402",
    "Q21502404",
    "Q21502410",
    "Q21503250",
    "Q21510857",
    "Q21510859",
    "Q21510860",
    "Q21510861",
    "Q21510865",
    "Q52004125",
    "Q53869507",
}
CASE_ID_SUFFIX_RE = re.compile(r"(?i)(?:_proposal|_patch|_draft|_fix|_v\d+)+$")


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
    uncertainty: dict[str, Any] | None = None
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
        if self.uncertainty is not None:
            payload["uncertainty"] = self.uncertainty
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


def _first_present(container: Any, *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current = container
        found = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                found = False
                break
            current = current[key]
        if found and current is not None:
            return current
    return None


def _canonical_case_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith(("repair_", "reform_")):
        text = CASE_ID_SUFFIX_RE.sub("", text)
    return text or None


def _extract_case_id(payload: dict[str, Any]) -> str | None:
    for key in ("case_id", "id", "reform_id", "proposal_id"):
        case_id = _canonical_case_id(payload.get(key))
        if case_id:
            return case_id
    return None


def _extract_pid(payload: dict[str, Any]) -> Any:
    pid = _first_present(
        payload,
        ("target", "pid"),
        ("target_property", "pid"),
        ("property", "pid"),
        ("context", "property"),
    )
    if pid is not None:
        return pid
    for candidate in (_first_present(payload, ("property",)), _first_present(payload, ("target", "property"))):
        if isinstance(candidate, str):
            return candidate
    return None


def _iter_constraint_candidates(payload: dict[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    for path in (
        ("target", "constraint_type_qid"),
        ("target_constraint_qid",),
        ("context", "target_constraint_qid"),
        ("proposal", "target_constraint_qid"),
    ):
        value = _first_present(payload, path)
        if value is not None:
            candidates.append(value)
    for list_path in (("proposed_changes",), ("changes",), ("recommended_changes",), ("proposals",)):
        value = _first_present(payload, list_path)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    candidates.extend(
                        [
                            item.get("constraint_qid"),
                            item.get("target_constraint_qid"),
                            _first_present(item, ("change", "target_constraint_qid")),
                        ]
                    )
    return [candidate for candidate in candidates if candidate is not None]


def _extract_constraint_type_qid(payload: dict[str, Any]) -> Any:
    for candidate in _iter_constraint_candidates(payload):
        if isinstance(candidate, str):
            return candidate
    return None


def _extract_action(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if normalized in SUPPORTED_ACTIONS:
        return normalized
    for action in SUPPORTED_ACTIONS:
        if action in normalized:
            return action
    return None


def _placeholder_signature_after(constraint_qid: Any) -> list[dict[str, Any]] | None:
    if not isinstance(constraint_qid, str):
        return None
    return [
        {
            "constraint_qid": constraint_qid,
            "snaktype": "VALUE",
            "qualifiers": [],
        }
    ]


def _coerce_payload_shape(payload: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(payload)
    case_id = _extract_case_id(payload)
    pid = _extract_pid(payload)
    constraint_type_qid = _extract_constraint_type_qid(payload)

    if case_id and not coerced.get("case_id"):
        coerced["case_id"] = case_id

    target = coerced.get("target")
    if not isinstance(target, dict):
        target = {}
    else:
        target = dict(target)
    if target.get("pid") is None and pid is not None:
        target["pid"] = pid
    if target.get("constraint_type_qid") is None and constraint_type_qid is not None:
        target["constraint_type_qid"] = constraint_type_qid
    if target:
        coerced["target"] = target

    proposal = coerced.get("proposal")
    if not isinstance(proposal, dict):
        proposal = {}
    else:
        proposal = dict(proposal)
    if proposal.get("action") is None:
        for candidate in (
            _first_present(payload, ("proposal", "action")),
            _first_present(payload, ("reform_type",)),
            _first_present(payload, ("proposal_type",)),
            _first_present(payload, ("type",)),
        ):
            action = _extract_action(candidate)
            if action:
                proposal["action"] = action
                break
    if proposal.get("signature_after") is None:
        signature_after = _first_present(payload, ("proposal", "signature_after")) or _first_present(
            payload,
            ("signature_after",),
        )
        if signature_after is None:
            signature_after = _placeholder_signature_after(target.get("constraint_type_qid"))
        if signature_after is not None:
            proposal["signature_after"] = signature_after
    if proposal:
        coerced["proposal"] = proposal

    rationale = _first_present(payload, ("rationale",), ("summary",), ("motivation",), ("short_summary",))
    if rationale is not None and "rationale" not in coerced:
        coerced["rationale"] = rationale
    provenance = _first_present(payload, ("provenance",), ("references",))
    if provenance is not None and "provenance" not in coerced:
        coerced["provenance"] = provenance
    uncertainty = _first_present(
        payload,
        ("uncertainty",),
        ("proposal", "uncertainty"),
        ("confidence",),
        ("metadata", "uncertainty"),
        ("metadata", "confidence"),
    )
    if uncertainty is not None and "uncertainty" not in coerced:
        coerced["uncertainty"] = uncertainty
    metadata = _first_present(payload, ("metadata",), ("context",), ("diagnostics",))
    if metadata is not None and "metadata" not in coerced:
        coerced["metadata"] = metadata
    return coerced


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


def _normalize_provenance(items: Any) -> list[dict[str, Any]]:
    return normalize_provenance_payload(items)


def _normalize_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "metadata must be an object.")
    return dict(metadata)


def _normalize_constraint_type_qids(constraint_type_qids: Iterable[str] | None) -> set[str] | None:
    if constraint_type_qids is None:
        return None
    normalized: set[str] = set()
    for value in constraint_type_qids:
        normalized.add(normalize_qid(value))
    return normalized


def _validate_constraint_family_qid(
    qid: str,
    *,
    allowed_qids: set[str] | None,
    error_message: str,
) -> None:
    if allowed_qids is None:
        return
    if qid not in allowed_qids:
        raise PatchValidationError("SCHEMA_VIOLATION", error_message)


def normalize_signature_after(
    signature_after: Any,
    *,
    constraint_type_qids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    allowed_constraint_qids = _normalize_constraint_type_qids(constraint_type_qids)
    if not isinstance(signature_after, list):
        raise PatchValidationError("SCHEMA_VIOLATION", "proposal.signature_after must be a list.")
    normalized: list[dict[str, Any]] = []
    for entry in signature_after:
        if not isinstance(entry, dict):
            raise PatchValidationError("SCHEMA_VIOLATION", "Each signature entry must be an object.")
        constraint_qid = normalize_qid(entry.get("constraint_qid"))
        _validate_constraint_family_qid(
            constraint_qid,
            allowed_qids=allowed_constraint_qids,
            error_message="invalid signature constraint_qid for T-box proposal",
        )
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


def normalize_proposal(
    raw: Any,
    schema: Any = None,
    *,
    constraint_type_qids: Iterable[str] | None = None,
) -> NormalizedReformProposal:
    del schema
    payload = _coerce_payload_shape(_load_payload(raw))
    allowed_constraint_qids = _normalize_constraint_type_qids(constraint_type_qids)
    case_id = _require_non_empty_string(payload.get("case_id"), "case_id")
    target_payload = payload.get("target")
    if not isinstance(target_payload, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "target must be an object.")
    target_constraint_type_qid = normalize_qid(target_payload.get("constraint_type_qid"))
    _validate_constraint_family_qid(
        target_constraint_type_qid,
        allowed_qids=allowed_constraint_qids,
        error_message="invalid constraint_type_qid for T-box proposal",
    )
    target = ReformTarget(
        pid=normalize_pid(target_payload.get("pid")),
        constraint_type_qid=target_constraint_type_qid,
    )

    proposal_payload = payload.get("proposal")
    if not isinstance(proposal_payload, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "proposal must be an object.")
    action = _require_non_empty_string(proposal_payload.get("action"), "proposal.action").upper()
    if action not in SUPPORTED_ACTIONS:
        raise PatchValidationError("SCHEMA_VIOLATION", f"Unsupported reform action: {action!r}")
    signature_after = normalize_signature_after(
        proposal_payload.get("signature_after"),
        constraint_type_qids=allowed_constraint_qids,
    )

    rationale = payload.get("rationale")
    if rationale is not None:
        rationale = _require_non_empty_string(rationale, "rationale")
    provenance = _normalize_provenance(payload.get("provenance"))
    uncertainty = normalize_uncertainty_payload(payload.get("uncertainty"))
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
    if uncertainty is not None:
        base_payload["uncertainty"] = uncertainty
    if metadata:
        base_payload["metadata"] = metadata

    return NormalizedReformProposal(
        case_id=case_id,
        target=target,
        proposal=ReformProposal(action=action, signature_after=signature_after),
        rationale=rationale,
        provenance=provenance,
        uncertainty=uncertainty,
        metadata=metadata,
        canonical_hash=canonical_hash(base_payload),
    )


__all__ = [
    "PatchValidationError",
    "NormalizedReformProposal",
    "KNOWN_CONSTRAINT_TYPE_QIDS",
    "ReformProposal",
    "ReformTarget",
    "SUPPORTED_ACTIONS",
    "canonicalize",
    "load_schema",
    "normalize_proposal",
    "normalize_signature_after",
]

