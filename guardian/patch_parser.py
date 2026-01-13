from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import jsonschema


QID_RE = re.compile(r"^Q[1-9][0-9]*$")
PID_RE = re.compile(r"^P[1-9][0-9]*$")
QID_RE_CASE_INSENSITIVE = re.compile(r"^Q[1-9][0-9]*$", re.IGNORECASE)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SUPPORTED_OPS = {"SET", "ADD", "REMOVE", "DELETE_ALL"}


class PatchValidationError(Exception):
    def __init__(self, code: str, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True)
class NormalizedOp:
    op: str
    pid: str
    value: Optional[Any] = None
    rank: Optional[str] = None


@dataclass(frozen=True)
class NormalizedRepairProposal:
    case_id: str
    target: dict[str, str]
    ops: list[NormalizedOp]
    rationale: Optional[str]
    provenance: list[dict[str, Any]]
    metadata: Optional[dict[str, Any]]
    canonical_hash: str


def load_schema(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_against_schema(obj: dict[str, Any], schema: dict[str, Any]) -> None:
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(obj), key=lambda e: len(list(e.absolute_path)))
    if errors:
        error = errors[0]
        details = {
            "path": list(error.absolute_path),
            "schema_path": list(error.absolute_schema_path),
            "message": error.message,
            "instance": error.instance,
        }
        raise PatchValidationError("SCHEMA_VIOLATION", "Schema validation failed.", details)


def canonicalize(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _default_schema_path() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "schemas", "verified_repair_proposal.schema.json")


def _parse_input(obj: Any) -> dict[str, Any]:
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
        except json.JSONDecodeError as exc:
            details = {"message": exc.msg, "line": exc.lineno, "column": exc.colno}
            raise PatchValidationError("INVALID_JSON", "Input is not valid JSON.", details)
        obj = parsed
    if not isinstance(obj, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "Top-level proposal must be an object.", {"value": obj})
    return obj


def _minimal_normalize(obj: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(obj)
    target = normalized.get("target")
    if isinstance(target, dict):
        qid = target.get("qid")
        if isinstance(qid, str):
            target["qid"] = qid.strip().upper()
        pid = target.get("pid")
        if isinstance(pid, str):
            target["pid"] = pid.strip().upper()

    ops = normalized.get("ops")
    if isinstance(ops, list):
        for op in ops:
            if not isinstance(op, dict):
                continue
            op_name = op.get("op")
            if isinstance(op_name, str):
                op["op"] = op_name.strip().upper()
            pid = op.get("pid")
            if isinstance(pid, str):
                op["pid"] = pid.strip().upper()
            value = op.get("value")
            if isinstance(value, str) and QID_RE_CASE_INSENSITIVE.match(value.strip()):
                op["value"] = value.strip().upper()
    return normalized


def _validate_ids(obj: dict[str, Any]) -> None:
    target = obj.get("target")
    if isinstance(target, dict):
        qid = target.get("qid")
        pid = target.get("pid")
        if isinstance(qid, str) and not QID_RE.match(qid):
            raise PatchValidationError("INVALID_ID", "Invalid QID.", {"field": "target.qid", "value": qid})
        if isinstance(pid, str) and not PID_RE.match(pid):
            raise PatchValidationError("INVALID_ID", "Invalid PID.", {"field": "target.pid", "value": pid})

    ops = obj.get("ops")
    if isinstance(ops, list):
        for idx, op in enumerate(ops):
            if not isinstance(op, dict):
                continue
            pid = op.get("pid")
            if isinstance(pid, str) and not PID_RE.match(pid):
                raise PatchValidationError(
                    "INVALID_ID",
                    "Invalid PID.",
                    {"field": f"ops[{idx}].pid", "value": pid},
                )


def _normalize_value(value: Any) -> Any:
    if isinstance(value, bool):
        raise PatchValidationError("INVALID_VALUE", "Boolean values are not supported.", {"value": value})
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise PatchValidationError("INVALID_VALUE", "Empty string values are not allowed.", {"value": value})
        if QID_RE_CASE_INSENSITIVE.match(stripped):
            normalized = stripped.upper()
            if not QID_RE.match(normalized):
                raise PatchValidationError("INVALID_VALUE", "Invalid QID value.", {"value": value})
            return normalized
        if DATE_RE.match(stripped):
            try:
                dt.date.fromisoformat(stripped)
            except ValueError as exc:
                raise PatchValidationError("INVALID_VALUE", "Invalid ISO date.", {"value": value, "error": str(exc)})
            return stripped
        return stripped
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    raise PatchValidationError("INVALID_VALUE", "Unsupported value type.", {"value": value})


def _normalize_ops(ops: list[dict[str, Any]]) -> list[NormalizedOp]:
    if not ops:
        raise PatchValidationError("SCHEMA_VIOLATION", "At least one op is required.", {"value": ops})
    if len(ops) > 50:
        raise PatchValidationError("SCHEMA_VIOLATION", "Too many ops.", {"count": len(ops)})
    normalized_ops: list[NormalizedOp] = []
    for idx, op in enumerate(ops):
        if not isinstance(op, dict):
            raise PatchValidationError("SCHEMA_VIOLATION", "Op must be an object.", {"index": idx, "value": op})
        op_name = op.get("op")
        if not isinstance(op_name, str):
            raise PatchValidationError("SCHEMA_VIOLATION", "Op name must be a string.", {"index": idx})
        op_name = op_name.upper()
        if op_name not in SUPPORTED_OPS:
            raise PatchValidationError("UNSUPPORTED_OP", "Unsupported op type.", {"index": idx, "op": op_name})

        pid = op.get("pid")
        if not isinstance(pid, str) or not PID_RE.match(pid):
            raise PatchValidationError("INVALID_ID", "Invalid PID.", {"index": idx, "value": pid})

        # Normalize REMOVE without value to DELETE_ALL.
        if op_name == "REMOVE" and "value" not in op:
            op_name = "DELETE_ALL"

        value = None
        if op_name in {"SET", "ADD", "REMOVE"}:
            if "value" not in op:
                raise PatchValidationError("SCHEMA_VIOLATION", "Missing value.", {"index": idx})
            value = _normalize_value(op.get("value"))
        rank = op.get("rank")
        normalized_ops.append(NormalizedOp(op=op_name, pid=pid, value=value, rank=rank))
    return normalized_ops


def _proposal_to_dict(proposal: NormalizedRepairProposal, include_hash: bool = True) -> dict[str, Any]:
    ops: list[dict[str, Any]] = []
    for op in proposal.ops:
        op_dict: dict[str, Any] = {"op": op.op, "pid": op.pid}
        if op.value is not None:
            op_dict["value"] = op.value
        if op.rank is not None:
            op_dict["rank"] = op.rank
        ops.append(op_dict)

    payload: dict[str, Any] = {
        "case_id": proposal.case_id,
        "target": proposal.target,
        "ops": ops,
        "provenance": proposal.provenance,
    }
    if proposal.rationale is not None:
        payload["rationale"] = proposal.rationale
    if proposal.metadata is not None:
        payload["metadata"] = proposal.metadata
    if include_hash:
        payload["canonical_hash"] = proposal.canonical_hash
    return payload


def normalize_proposal(obj: Any, schema: Optional[dict[str, Any]] = None) -> NormalizedRepairProposal:
    parsed = _parse_input(obj)
    minimal = _minimal_normalize(parsed)
    _validate_ids(minimal)
    schema_obj = schema or load_schema(_default_schema_path())
    validate_against_schema(minimal, schema_obj)

    target = minimal["target"]
    ops = _normalize_ops(minimal["ops"])
    rationale = minimal.get("rationale")
    provenance = minimal.get("provenance") or []
    metadata = minimal.get("metadata")

    proposal = NormalizedRepairProposal(
        case_id=minimal["case_id"],
        target={"qid": target["qid"], "pid": target["pid"]},
        ops=ops,
        rationale=rationale,
        provenance=provenance,
        metadata=metadata,
        canonical_hash="",
    )

    canonical_payload = _proposal_to_dict(proposal, include_hash=False)
    canonical_json = canonicalize(canonical_payload)
    canonical_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return dataclasses.replace(proposal, canonical_hash=canonical_hash)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Normalize and validate repair proposals.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input JSON proposal file.")
    parser.add_argument("--out", dest="output_path", required=True, help="Output normalized JSON file.")
    parser.add_argument("--schema", dest="schema_path", default=_default_schema_path(), help="Schema path.")
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    schema = load_schema(args.schema_path)
    proposal = normalize_proposal(raw, schema=schema)
    normalized_dict = _proposal_to_dict(proposal, include_hash=True)

    with open(args.output_path, "w", encoding="utf-8") as handle:
        handle.write(canonicalize(normalized_dict))
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
