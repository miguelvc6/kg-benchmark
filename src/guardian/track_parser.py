from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from .common import PatchValidationError, canonical_hash, canonicalize

SUPPORTED_TRACKS = {"A_BOX", "T_BOX", "AMBIGUOUS"}


@dataclass
class NormalizedTrackDiagnosis:
    case_id: str
    predicted_track: str
    confidence: Optional[str] = None
    rationale: Optional[str] = None
    canonical_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "case_id": self.case_id,
            "predicted_track": self.predicted_track,
            "canonical_hash": self.canonical_hash,
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.rationale is not None:
            payload["rationale"] = self.rationale
        return payload


def load_schema(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise PatchValidationError("INVALID_SCHEMA", "Track diagnosis schema must decode to an object.")
    return payload


def normalize_diagnosis(raw: Any, schema: Any = None) -> NormalizedTrackDiagnosis:
    del schema
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PatchValidationError("INVALID_JSON", "Track diagnosis is not valid JSON.") from exc
    if not isinstance(raw, dict):
        raise PatchValidationError("SCHEMA_VIOLATION", "Track diagnosis must be a JSON object.")

    case_id = raw.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise PatchValidationError("SCHEMA_VIOLATION", "case_id must be a non-empty string.")
    predicted_track = raw.get("predicted_track")
    if not isinstance(predicted_track, str) or not predicted_track.strip():
        raise PatchValidationError("SCHEMA_VIOLATION", "predicted_track must be a non-empty string.")
    predicted_track = predicted_track.strip().upper()
    if predicted_track not in SUPPORTED_TRACKS:
        raise PatchValidationError("SCHEMA_VIOLATION", f"Unsupported predicted track: {predicted_track!r}")
    confidence = raw.get("confidence")
    if confidence is not None:
        if not isinstance(confidence, str) or not confidence.strip():
            raise PatchValidationError("SCHEMA_VIOLATION", "confidence must be a non-empty string when present.")
        confidence = confidence.strip().lower()
    rationale = raw.get("rationale")
    if rationale is not None:
        if not isinstance(rationale, str) or not rationale.strip():
            raise PatchValidationError("SCHEMA_VIOLATION", "rationale must be a non-empty string when present.")
        rationale = rationale.strip()

    payload = {"case_id": case_id.strip(), "predicted_track": predicted_track}
    if confidence is not None:
        payload["confidence"] = confidence
    if rationale is not None:
        payload["rationale"] = rationale
    return NormalizedTrackDiagnosis(
        case_id=payload["case_id"],
        predicted_track=predicted_track,
        confidence=confidence,
        rationale=rationale,
        canonical_hash=canonical_hash(payload),
    )


__all__ = [
    "PatchValidationError",
    "NormalizedTrackDiagnosis",
    "SUPPORTED_TRACKS",
    "canonicalize",
    "load_schema",
    "normalize_diagnosis",
]

