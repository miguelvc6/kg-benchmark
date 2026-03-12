from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class PatchValidationError(ValueError):
    """Structured validation failure exposed by proposal parsers."""

    def __init__(self, code: str, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def load_schema(path: str | Path) -> dict[str, Any]:
    schema_path = Path(path)
    with open(schema_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise PatchValidationError("INVALID_SCHEMA", f"Schema at {schema_path} must decode to an object.")
    return payload


def _normalize_for_json(payload: Any) -> Any:
    if is_dataclass(payload):
        return _normalize_for_json(asdict(payload))
    if isinstance(payload, dict):
        return {str(key): _normalize_for_json(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_normalize_for_json(item) for item in payload]
    if isinstance(payload, tuple):
        return [_normalize_for_json(item) for item in payload]
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise ValueError("Non-finite numeric value cannot be canonicalized.")
        return payload
    return payload


def canonicalize(payload: Any) -> str:
    normalized = _normalize_for_json(payload)
    return json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonicalize(payload).encode("utf-8")).hexdigest()


def normalize_qid(value: Any) -> str:
    if not isinstance(value, str):
        raise PatchValidationError("INVALID_ID", "QID must be a string.")
    normalized = value.strip().upper()
    if not normalized.startswith("Q") or normalized == "Q":
        raise PatchValidationError("INVALID_ID", f"Invalid QID: {value!r}")
    if not normalized[1:].isdigit() or int(normalized[1:]) <= 0:
        raise PatchValidationError("INVALID_ID", f"Invalid QID: {value!r}")
    return normalized


def normalize_pid(value: Any) -> str:
    if not isinstance(value, str):
        raise PatchValidationError("INVALID_ID", "PID must be a string.")
    normalized = value.strip().upper()
    if not normalized.startswith("P") or normalized == "P":
        raise PatchValidationError("INVALID_ID", f"Invalid PID: {value!r}")
    if not normalized[1:].isdigit() or int(normalized[1:]) <= 0:
        raise PatchValidationError("INVALID_ID", f"Invalid PID: {value!r}")
    return normalized

