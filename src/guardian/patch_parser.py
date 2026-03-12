from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from .common import PatchValidationError, canonical_hash, canonicalize, load_schema, normalize_pid, normalize_qid

ALLOWED_OPS = {"SET", "ADD", "REMOVE", "DELETE_ALL"}
ALLOWED_RANKS = {"normal", "preferred", "deprecated"}
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class ProposalTarget:
    qid: str
    pid: str

    def to_dict(self) -> dict[str, str]:
        return {"qid": self.qid, "pid": self.pid}


@dataclass
class ProposalOp:
    op: str
    pid: str
    value: Optional[Any] = None
    rank: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"op": self.op, "pid": self.pid}
        if self.value is not None:
            payload["value"] = self.value
        if self.rank is not None:
            payload["rank"] = self.rank
        return payload


@dataclass
class NormalizedProposal:
    case_id: str
    target: ProposalTarget
    ops: list[ProposalOp]
    rationale: Optional[str] = None
    provenance: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    canonical_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "target": self.target.to_dict(),
            "ops": [op.to_dict() for op in self.ops],
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
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PatchValidationError("INVALID_JSON", "Proposal is not valid JSON.") from exc
        raw = loaded
    if not isinstance(raw, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "Proposal must be a JSON object.")
    return dict(raw)


def _normalize_case_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PatchValidationError("SCHEMA_VIOLATION", "case_id must be a non-empty string.")
    return value.strip()


def _normalize_rank(value: Any) -> str:
    if value is None:
        return "normal"
    if not isinstance(value, str):
        raise PatchValidationError("SCHEMA_VIOLATION", "rank must be a string.")
    normalized = value.strip().lower()
    if normalized not in ALLOWED_RANKS:
        raise PatchValidationError("SCHEMA_VIOLATION", f"Unsupported rank: {value!r}")
    return normalized


def _normalize_value(value: Any) -> Any:
    if isinstance(value, bool):
        raise PatchValidationError("INVALID_VALUE", "Boolean values are not supported repair values.")
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise PatchValidationError("INVALID_VALUE", "Non-finite numeric value is invalid.")
        return value
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise PatchValidationError("INVALID_VALUE", "Repair value must be a string or finite number.")

    text = value.strip()
    if not text:
        raise PatchValidationError("INVALID_VALUE", "Repair value cannot be empty.")

    upper = text.upper()
    if upper.startswith("Q") and upper[1:].isdigit():
        return normalize_qid(upper)

    try:
        date.fromisoformat(text)
        return text
    except ValueError:
        if ISO_DATE_RE.fullmatch(text):
            raise PatchValidationError("INVALID_VALUE", f"Invalid ISO date value: {text!r}")
        return text


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
    payload = dict(metadata)
    token_usage = payload.get("token_usage")
    if token_usage is not None:
        if not isinstance(token_usage, dict):
            raise PatchValidationError("SCHEMA_VIOLATION", "metadata.token_usage must be an object.")
        normalized_usage = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = token_usage.get(key)
            if value is None:
                continue
            if not isinstance(value, int) or value < 0:
                raise PatchValidationError("SCHEMA_VIOLATION", f"{key} must be a non-negative integer.")
            normalized_usage[key] = value
        payload["token_usage"] = normalized_usage
    return payload


def _normalize_ops(items: Any) -> list[ProposalOp]:
    if not isinstance(items, list):
        raise PatchValidationError("SCHEMA_VIOLATION", "ops must be a list.")
    if not items:
        raise PatchValidationError("SCHEMA_VIOLATION", "ops must contain at least one operation.")
    if len(items) > 50:
        raise PatchValidationError("SCHEMA_VIOLATION", "ops exceeds the maximum supported length.")
    normalized: list[ProposalOp] = []
    for item in items:
        if not isinstance(item, dict):
            raise PatchValidationError("SCHEMA_VIOLATION", "Each op must be an object.")
        op_raw = item.get("op")
        if not isinstance(op_raw, str) or not op_raw.strip():
            raise PatchValidationError("SCHEMA_VIOLATION", "Each op requires a non-empty op field.")
        op = op_raw.strip().upper()
        if op not in ALLOWED_OPS and op != "REMOVE":
            raise PatchValidationError("SCHEMA_VIOLATION", f"Unsupported op: {op_raw!r}")
        pid = normalize_pid(item.get("pid"))
        value_present = "value" in item
        value = item.get("value")
        rank: Optional[str] = None

        if op == "REMOVE" and not value_present:
            normalized.append(ProposalOp(op="DELETE_ALL", pid=pid, value=None, rank=None))
            continue
        if op == "DELETE_ALL":
            normalized.append(ProposalOp(op="DELETE_ALL", pid=pid, value=None, rank=None))
            continue
        if not value_present:
            raise PatchValidationError("SCHEMA_VIOLATION", f"{op} requires a value.")

        normalized_value = _normalize_value(value)
        if op in {"SET", "ADD"}:
            rank = _normalize_rank(item.get("rank"))
        normalized.append(ProposalOp(op=op, pid=pid, value=normalized_value, rank=rank))
    return normalized


def normalize_proposal(raw: Any, schema: Any = None) -> NormalizedProposal:
    del schema  # The on-disk schema remains the public contract; validation is implemented manually.
    payload = _load_payload(raw)
    case_id = _normalize_case_id(payload.get("case_id"))
    target_payload = payload.get("target")
    if not isinstance(target_payload, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "target must be an object.")
    target = ProposalTarget(
        qid=normalize_qid(target_payload.get("qid")),
        pid=normalize_pid(target_payload.get("pid")),
    )
    ops = _normalize_ops(payload.get("ops"))
    rationale = payload.get("rationale")
    if rationale is not None:
        if not isinstance(rationale, str) or not rationale.strip():
            raise PatchValidationError("SCHEMA_VIOLATION", "rationale must be a non-empty string when present.")
        rationale = rationale.strip()
    provenance = _normalize_provenance(payload.get("provenance"))
    metadata = _normalize_metadata(payload.get("metadata"))

    base_payload = {
        "case_id": case_id,
        "target": target.to_dict(),
        "ops": [op.to_dict() for op in ops],
    }
    if rationale is not None:
        base_payload["rationale"] = rationale
    if provenance:
        base_payload["provenance"] = provenance
    if metadata:
        base_payload["metadata"] = metadata

    return NormalizedProposal(
        case_id=case_id,
        target=target,
        ops=ops,
        rationale=rationale,
        provenance=provenance,
        metadata=metadata,
        canonical_hash=canonical_hash(base_payload),
    )


__all__ = [
    "PatchValidationError",
    "ProposalOp",
    "ProposalTarget",
    "NormalizedProposal",
    "canonicalize",
    "load_schema",
    "normalize_proposal",
]
