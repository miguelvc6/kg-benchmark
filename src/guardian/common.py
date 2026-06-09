from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


CONFIDENCE_LABELS = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
}


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


def _normalize_provenance_kind(payload: dict[str, Any]) -> str:
    kind = payload.get("kind")
    if isinstance(kind, str) and kind.strip():
        return kind.strip().upper()
    url = payload.get("url")
    if isinstance(url, str) and url.strip():
        return "WEB"
    node_id = payload.get("node_id")
    if isinstance(node_id, str) and node_id.strip():
        return "KG"
    revision_id = payload.get("revision_id")
    if isinstance(revision_id, int) and revision_id > 0:
        return "HISTORY"
    source = payload.get("source")
    if isinstance(source, str):
        candidate = source.strip().upper()
        if candidate.startswith(("Q", "P")) and len(candidate) > 1 and candidate[1:].isdigit():
            return "KG"
    return "OTHER"


def _normalize_provenance_entry(item: Any) -> dict[str, Any] | None:
    if item is None:
        return None
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        return {"kind": "OTHER", "snippet": text}
    if isinstance(item, (int, float)) and not isinstance(item, bool):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return {"kind": "OTHER", "snippet": str(item)}
    if not isinstance(item, dict):
        return None

    payload = dict(item)
    normalized: dict[str, Any] = {"kind": _normalize_provenance_kind(payload)}

    url = payload.get("url")
    if isinstance(url, str) and url.strip():
        normalized["url"] = url.strip()

    node_id = payload.get("node_id")
    if node_id is None and isinstance(payload.get("source"), str):
        source = payload["source"].strip().upper()
        if source.startswith(("Q", "P")) and len(source) > 1 and source[1:].isdigit():
            node_id = source
    if isinstance(node_id, str) and node_id.strip():
        normalized["node_id"] = node_id.strip().upper()

    revision_id = payload.get("revision_id")
    if isinstance(revision_id, int) and revision_id > 0:
        normalized["revision_id"] = revision_id

    snippet_candidates = (
        payload.get("snippet"),
        payload.get("summary"),
        payload.get("description"),
        payload.get("evidence"),
        payload.get("source"),
    )
    for candidate in snippet_candidates:
        if isinstance(candidate, str) and candidate.strip():
            normalized["snippet"] = candidate.strip()
            break

    confidence = payload.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        if math.isfinite(float(confidence)) and 0.0 <= float(confidence) <= 1.0:
            normalized["confidence"] = float(confidence)
    elif isinstance(confidence, str):
        try:
            parsed = float(confidence.strip())
        except ValueError:
            parsed = None
        if isinstance(parsed, float) and math.isfinite(parsed) and 0.0 <= parsed <= 1.0:
            normalized["confidence"] = parsed

    return normalized


def normalize_provenance_payload(items: Any) -> list[dict[str, Any]]:
    if items is None:
        return []
    if not isinstance(items, list):
        items = [items]

    normalized: list[dict[str, Any]] = []
    for item in items:
        entry = _normalize_provenance_entry(item)
        if entry is not None:
            normalized.append(entry)
    return normalized


def _normalize_confidence_score(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise PatchValidationError("SCHEMA_VIOLATION", f"{field_name} must be a numeric confidence score.")
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if not text:
            raise PatchValidationError("SCHEMA_VIOLATION", f"{field_name} must be a non-empty confidence score.")
        if text in CONFIDENCE_LABELS:
            return CONFIDENCE_LABELS[text]
        try:
            score = float(text)
        except ValueError as exc:
            raise PatchValidationError("SCHEMA_VIOLATION", f"{field_name} must be a numeric confidence score.") from exc
    else:
        raise PatchValidationError("SCHEMA_VIOLATION", f"{field_name} must be a numeric confidence score.")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise PatchValidationError("SCHEMA_VIOLATION", f"{field_name} must be between 0.0 and 1.0.")
    return score


def normalize_uncertainty_payload(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, (str, int, float)) and not isinstance(payload, bool):
        return {"confidence": _normalize_confidence_score(payload, "uncertainty")}
    if not isinstance(payload, dict):
        raise PatchValidationError(
            "SCHEMA_VIOLATION",
            "uncertainty must be an object or numeric confidence score.",
        )

    confidence_raw = payload.get("confidence")
    if confidence_raw is None:
        confidence_raw = payload.get("score")
    if confidence_raw is None:
        confidence_raw = payload.get("probability")
    if confidence_raw is None:
        raise PatchValidationError("SCHEMA_VIOLATION", "uncertainty.confidence is required when uncertainty is present.")

    normalized = {"confidence": _normalize_confidence_score(confidence_raw, "uncertainty.confidence")}
    notes = payload.get("notes")
    if notes is None:
        notes = payload.get("summary")
    if notes is None:
        notes = payload.get("rationale")
    if notes is not None:
        if not isinstance(notes, str):
            raise PatchValidationError("SCHEMA_VIOLATION", "uncertainty.notes must be a non-empty string when present.")
        stripped_notes = notes.strip()
        if stripped_notes:
            normalized["notes"] = stripped_notes
    return normalized

