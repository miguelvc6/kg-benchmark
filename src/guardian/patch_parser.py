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
CASE_ID_SUFFIX_RE = re.compile(r"(?i)(?:_proposal|_patch|_draft|_fix|_v\d+)+$")


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
    for key in ("case_id", "id", "repair_id", "proposal_id"):
        case_id = _canonical_case_id(payload.get(key))
        if case_id:
            return case_id
    return None


def _extract_qid(payload: dict[str, Any]) -> Any:
    return _first_present(
        payload,
        ("target", "qid"),
        ("qid",),
        ("entity", "qid"),
        ("scope", "qid"),
        ("apply_change", "target", "qid"),
    )


def _extract_pid(payload: dict[str, Any]) -> Any:
    pid = _first_present(
        payload,
        ("target", "pid"),
        ("property", "pid"),
        ("apply_change", "target", "property"),
    )
    if pid is not None:
        return pid
    for candidate in (
        _first_present(payload, ("property",)),
        _first_present(payload, ("target", "property")),
    ):
        if isinstance(candidate, str):
            return candidate
    return None


def _extract_atom(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("qid", "value", "value_qid", "id", "raw"):
            if key in value:
                return value[key]
    return value


def _extract_value_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        values: list[Any] = []
        for item in value:
            atom = _extract_atom(item)
            if atom is not None:
                values.append(atom)
        return values
    atom = _extract_atom(value)
    return [] if atom is None else [atom]


def _final_values_from_payload(payload: dict[str, Any], pid: str) -> list[Any] | None:
    for candidate in (
        _first_present(payload, ("proposal", "proposed_values")),
        _first_present(payload, ("proposed_values",)),
        _first_present(payload, ("proposal", "keep_values")),
        _first_present(payload, ("proposal", "values_to_keep")),
        _first_present(payload, ("persistence_check", "current_value_2026")),
        _first_present(payload, ("meta", "persistence_check", "current_value_2026")),
        _first_present(payload, ("current_state", "persistence_check", "current_value_2026")),
        _first_present(payload, ("expected_result_after_apply", pid)),
        _first_present(payload, ("proposal", "new_values")),
        _first_present(payload, ("new_value",)),
        _first_present(payload, ("proposed_value",)),
        _first_present(payload, ("apply_change", "add_value")),
    ):
        values = _extract_value_list(candidate)
        if values:
            return values
    if isinstance(_first_present(payload, ("expected_result_after_apply", pid)), list):
        return []
    return None


def _ops_from_patch_edits(payload: dict[str, Any], pid: str) -> list[dict[str, Any]]:
    edits = _first_present(payload, ("patch", "edits"))
    if not isinstance(edits, list):
        return []
    ops: list[dict[str, Any]] = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        edit_pid = edit.get("property") if isinstance(edit.get("property"), str) else pid
        action = str(edit.get("type") or edit.get("action") or "").strip().lower()
        value = _extract_atom(edit.get("value"))
        if action in {"remove_claim", "remove_claim_value", "remove_statement"} and value is not None:
            ops.append({"op": "REMOVE", "pid": edit_pid, "value": value})
        elif action in {"add_statement", "add_claim"} and value is not None:
            ops.append({"op": "ADD", "pid": edit_pid, "value": value})
    return ops


def _coerce_payload_shape(payload: dict[str, Any]) -> dict[str, Any]:
    if "case_id" in payload and "target" in payload and "ops" in payload:
        return payload

    case_id = _extract_case_id(payload)
    qid = _extract_qid(payload)
    pid = _extract_pid(payload)

    coerced = dict(payload)
    if case_id and not coerced.get("case_id"):
        coerced["case_id"] = case_id

    target_payload = coerced.get("target")
    if not isinstance(target_payload, dict):
        target_payload = {}
    else:
        target_payload = dict(target_payload)
    if target_payload.get("qid") is None and qid is not None:
        target_payload["qid"] = qid
    if target_payload.get("pid") is None and pid is not None:
        target_payload["pid"] = pid
    if target_payload:
        coerced["target"] = target_payload

    if "ops" not in coerced and isinstance(pid, str):
        final_values = _final_values_from_payload(payload, pid)
        if final_values is not None:
            if not final_values:
                coerced["ops"] = [{"op": "DELETE_ALL", "pid": pid}]
            elif len(final_values) == 1:
                coerced["ops"] = [{"op": "SET", "pid": pid, "value": final_values[0]}]
            else:
                coerced["ops"] = [{"op": "DELETE_ALL", "pid": pid}] + [
                    {"op": "ADD", "pid": pid, "value": value} for value in final_values
                ]
        else:
            remove_values = (
                _extract_value_list(_first_present(payload, ("remove_values",)))
                or _extract_value_list(_first_present(payload, ("proposal", "remove_values")))
                or _extract_value_list(_first_present(payload, ("proposal", "values_to_remove")))
            )
            add_values = (
                _extract_value_list(_first_present(payload, ("add_values",)))
                or _extract_value_list(_first_present(payload, ("proposal", "add_values")))
            )
            ops = [{"op": "REMOVE", "pid": pid, "value": value} for value in remove_values] + [
                {"op": "ADD", "pid": pid, "value": value} for value in add_values
            ]
            if not ops:
                ops = _ops_from_patch_edits(payload, pid)
            action = str(_first_present(payload, ("action",), ("proposal", "action"), ("operation",)) or "").strip().upper()
            if not ops and action in {"DELETE", "DELETE_ALL", "REMOVE_ALL"}:
                ops = [{"op": "DELETE_ALL", "pid": pid}]
            if ops:
                coerced["ops"] = ops

    rationale = _first_present(payload, ("rationale",), ("justification",), ("summary",), ("proposal", "summary"))
    if rationale is not None and "rationale" not in coerced:
        coerced["rationale"] = rationale
    provenance = _first_present(payload, ("provenance",), ("references",), ("proposal", "references"))
    if provenance is not None and "provenance" not in coerced:
        coerced["provenance"] = provenance
    metadata = _first_present(payload, ("metadata",), ("meta",), ("evidence",))
    if metadata is not None and "metadata" not in coerced:
        coerced["metadata"] = metadata
    return coerced


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
    payload = _coerce_payload_shape(_load_payload(raw))
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
