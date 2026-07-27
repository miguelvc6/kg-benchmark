"""Field-level leakage audit for rendered benchmark prompts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, quote_plus, unquote, unquote_plus

from kg_benchmark.dataset.release import sha256_file
from lib.repair_state import derive_value_change_summary, normalize_value_list
from lib.utils import iter_jsonl

TEMPORAL_REPORT_VERSION = 5
HEARTBEAT_SECONDS = 60.0
SEVERITIES = (
    "high",
    "expected_historical",
    "expected_rule_derived",
    "expected_local_evidence",
    "diagnostic",
)

COMMON_HIGH_RISK_FIELDS = {
    "case_id",
    "repair_target.author",
    "repair_target.revision_id",
    "repair_target.property_revision_id",
    "repair_target.property_revision_prev",
    "repair_target.property_revision_new",
    "repair_target.constraint_delta.revision_id",
    "repair_target.constraint_delta.property_revision_id",
    "repair_target.constraint_delta.property_revision_prev",
    "repair_target.constraint_delta.property_revision_new",
    "repair_target.constraint_delta.hash_after",
    "repair_target.constraint_delta.signature_after",
    "repair_target.constraint_delta.new_constraints",
    "repair_target.constraint_delta.added_constraint_types",
    "persistence_check.current_value_2026",
    "persistence_check.current_value_2026_aliases_en",
    "persistence_check.current_value_2026_descriptions_en",
    "persistence_check.current_value_2026_labels_en",
    "violation_context.value_current_2026",
    "violation_context.value_current_2026_aliases_en",
    "violation_context.value_current_2026_descriptions_en",
    "violation_context.value_current_2026_labels_en",
}
A_BOX_TARGET_FIELDS = {
    "repair_target.new_value",
    "repair_target.new_value_aliases_en",
    "repair_target.new_value_descriptions_en",
    "repair_target.new_value_labels_en",
    "repair_target.value",
    "repair_target.value_aliases_en",
    "repair_target.value_descriptions_en",
    "repair_target.value_labels_en",
    "persistence_check.current_value_2026",
    "persistence_check.current_value_2026_aliases_en",
    "persistence_check.current_value_2026_descriptions_en",
    "persistence_check.current_value_2026_labels_en",
    "violation_context.value_current_2026",
    "violation_context.value_current_2026_aliases_en",
    "violation_context.value_current_2026_descriptions_en",
    "violation_context.value_current_2026_labels_en",
}

HISTORICAL_VISIBLE_FIELDS = (
    "qid",
    "property",
    "labels_en.qid.label",
    "labels_en.qid.description",
    "labels_en.property.label",
    "labels_en.property.description",
    "repair_target.old_value",
    "repair_target.old_value_aliases_en",
    "repair_target.old_value_labels_en",
    "repair_target.old_value_descriptions_en",
    "violation_context.report_violation_type",
    "violation_context.report_violation_type_normalized",
    "violation_context.report_violation_type_raw",
    "violation_context.report_violation_type_qids",
    "violation_context.report_page_title",
    "violation_context.value",
    "violation_context.value_aliases_en",
    "violation_context.value_labels_en",
    "violation_context.value_descriptions_en",
)

DIAGNOSTIC_FIELDS = {
    "classification.class",
    "classification.subtype",
    "information_type",
    "track",
}


def _progress(message: str) -> None:
    print(f"[temporal-audit] {message}", file=sys.stderr, flush=True)


class _Heartbeat:
    def __init__(self, phase: str, *, total: int | None = None) -> None:
        self.phase = phase
        self.total = total
        self.started = time.monotonic()
        self.last = self.started

    def update(self, completed: int) -> None:
        now = time.monotonic()
        if now - self.last < HEARTBEAT_SECONDS:
            return
        elapsed = now - self.started
        rate = completed / elapsed if elapsed else 0.0
        total = f"/{self.total}" if self.total is not None else ""
        _progress(
            f"heartbeat: phase={self.phase} completed={completed}{total} "
            f"elapsed={elapsed:.0f}s rate={rate:.1f}/s"
        )
        self.last = now


def _scalars(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from _scalars(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _scalars(nested)
    elif value is not None:
        yield str(value)


def _eligible_token(value: str) -> bool:
    token = value.strip()
    if not token:
        return False
    if len(token) >= 4:
        return True
    return len(token) >= 2 and token[0] in {"P", "Q"} and token[1:].isdigit()


def _field(record: dict[str, Any], dotted_path: str) -> Any:
    value: Any = record
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _decoded_surfaces(text: str) -> list[tuple[str, str, str]]:
    surfaces = [("exact", "original", text)]
    candidates = [
        ("html_unescape", html.unescape(text)),
        ("url_unquote", unquote(text)),
        ("url_unquote_plus", unquote_plus(text)),
    ]
    for coordinate_space, candidate in candidates:
        if candidate != text and all(candidate != value for _, _, value in surfaces):
            surfaces.append(("decoded", coordinate_space, candidate))
    return surfaces


def _literal_spans(text: str, token: str, *, boundaries: bool = True) -> list[tuple[int, int]]:
    if not token:
        return []
    if token and token[0].isalnum() and token[-1].isalnum():
        pattern = (
            rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])"
            if boundaries
            else re.escape(token)
        )
        return [(match.start(), match.end()) for match in re.finditer(pattern, text)]
    return [(match.start(), match.end()) for match in re.finditer(re.escape(token), text)]


def _occurrence_literal(text: str, token: str) -> int:
    return len(_literal_spans(text, token))


def _match_rows(
    analyzed_text: str,
    spans: Iterable[tuple[int, int]],
    *,
    match_mode: str,
    coordinate_space: str,
) -> list[dict[str, Any]]:
    return [
        {
            "start": start,
            "end": end,
            "matched_text": analyzed_text[start:end],
            "match_mode": match_mode,
            "coordinate_space": coordinate_space,
            "_analyzed_text": analyzed_text,
        }
        for start, end in spans
    ]


def _occurrences(text: str, token: str) -> list[dict[str, Any]]:
    for mode, coordinate_space, surface in _decoded_surfaces(text):
        spans = _literal_spans(surface, token)
        if spans:
            return _match_rows(
                surface,
                spans,
                match_mode=mode,
                coordinate_space=coordinate_space,
            )
    # Long identifiers embedded in URLs, paths, query values, or prefixed
    # literals are distinctive enough to scan without the short-token false
    # positives guarded by _literal_spans.
    if len(token) >= 8:
        spans = _literal_spans(text, token, boundaries=False)
        if spans:
            return _match_rows(
                text,
                spans,
                match_mode="embedded_token",
                coordinate_space="original",
            )
    normalized_text = _normalized_text(html.unescape(unquote_plus(text)))
    normalized_token = _normalized_text(html.unescape(unquote_plus(token)))
    if normalized_token:
        spans = _literal_spans(normalized_text, normalized_token)
        if spans:
            return _match_rows(
                normalized_text,
                spans,
                match_mode="semantic_normalized",
                coordinate_space="semantic_normalized",
            )
    version_alias = re.fullmatch(r"(.{8,}?)[._-]\d+", token)
    if version_alias:
        base = version_alias.group(1)
        spans = _literal_spans(text, base) or _literal_spans(text, base, boundaries=False)
        if spans:
            return _match_rows(
                text,
                spans,
                match_mode="semantic_alias",
                coordinate_space="original",
            )
    encoded_variants = {
        quote(token, safe=""),
        quote_plus(token, safe=""),
        token.encode("utf-8").hex(),
        base64.b64encode(token.encode("utf-8")).decode("ascii"),
        base64.urlsafe_b64encode(token.encode("utf-8")).decode("ascii").rstrip("="),
    }
    for encoded in sorted(encoded_variants):
        if not encoded or encoded == token:
            continue
        spans = _literal_spans(text, encoded, boundaries=False)
        if spans:
            return _match_rows(
                text,
                spans,
                match_mode="encoded_value",
                coordinate_space="original",
            )
    return []


def _occurrence(text: str, token: str) -> tuple[int, str | None]:
    matches = _occurrences(text, token)
    return (len(matches), str(matches[0]["match_mode"])) if matches else (0, None)


def _occurrence_count(text: str, token: str) -> int:
    return _occurrence(text, token)[0]


def mutation_sensitivity_checks() -> dict[str, bool]:
    """Self-check the exact, normalized, label, and boundary behavior used by the gate."""
    checks = {
        "exact_identifier": _occurrence_count("hidden Q999 value", "Q999") == 1,
        "normalized_label": _occurrence_count("NEW   TARGET LABEL", "New target label") == 1,
        "serialized_identifier": _occurrence_count('{"value":"Q999"}', "Q999") == 1,
        "substring_boundary": _occurrence_count("SCHEMBL54432", "54432") == 0,
        "url_encoded_label": _occurrence("target=Hidden%20Value", "Hidden Value")[1] == "decoded",
        "base64_value": _occurrence("payload=UTk5OQ==", "Q999")[1] == "encoded_value",
        "unicode_semantic_alias": _occurrence("ＣＡＦÉ", "café")[1] == "semantic_normalized",
        "long_embedded_identifier": _occurrence("url:CELEX:32013L0012", "32013L0012")[1]
        in {"exact", "embedded_token"},
        "version_suffix_alias": _occurrence("encoded by PocGH01_00229100", "PocGH01_00229100.1")[1]
        == "semantic_alias",
    }
    format_record = {
        "track": "A_BOX",
        "repair_target": {"old_value": ["uk.bl.ethos.490072"], "new_value": ["490072"]},
        "violation_context": {"value": ["uk.bl.ethos.490072"]},
        "classification": {"class": "TypeA", "subtype": "FORMAT_NORMALIZATION"},
    }
    format_claim = {"field": "repair_target.new_value", "token": "490072", "severity": "high"}
    format_occurrences = _occurrences("old uk.bl.ethos.490072; exposed 490072", "490072")
    format_severities = [
        _classify_occurrence(format_record, {"context_bundle": "logic_only"}, format_claim, occurrence)[0]
        for occurrence in format_occurrences
    ]
    checks["historical_span_coverage"] = format_severities[:1] == ["expected_rule_derived"]
    checks["uncovered_occurrence_blocks"] = format_severities[1:] == ["high"]
    return checks


def _expected_rule_derived_visibility(record: dict[str, Any], field: str, token: str) -> bool:
    if field not in A_BOX_TARGET_FIELDS:
        return False
    classification = record.get("classification")
    classification = classification if isinstance(classification, dict) else {}
    if classification.get("class") != "TypeA":
        return False
    subtype = classification.get("subtype")
    if subtype == "TARGET_REQUIRED_CLAIM":
        expected = {str(record.get("qid") or "")}
        expected.update(_scalars(_field(record, "labels_en.qid.label")))
        expected.update(_scalars(_field(record, "labels_en.qid.description")))
        expected.update(_scalars(_field(record, "repair_target.new_value_labels_en")))
        expected.update(_scalars(_field(record, "repair_target.new_value_descriptions_en")))
        return token in expected
    return False


def _historical_sources(record: dict[str, Any]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field in HISTORICAL_VISIBLE_FIELDS:
        for raw_token in _scalars(_field(record, field)):
            token = raw_token.strip()
            key = (field, token)
            if token and key not in seen:
                seen.add(key)
                sources.append({"field": field, "token": token})
    return sources


def _equivalent(left: str, right: str) -> bool:
    return _normalized_text(left) == _normalized_text(right)


def _token_is_historical(token: str, sources: list[dict[str, str]]) -> bool:
    return any(_equivalent(token, source["token"]) for source in sources)


def _aligned_values_for_field(record: dict[str, Any], field: str) -> list[str]:
    if field.startswith("repair_target.new_value"):
        return normalize_value_list(_field(record, "repair_target.new_value"))
    if field.startswith("repair_target.value"):
        return normalize_value_list(_field(record, "repair_target.value"))
    if field.startswith("persistence_check.current_value_2026"):
        return normalize_value_list(_field(record, "persistence_check.current_value_2026"))
    if field.startswith("violation_context.value_current_2026"):
        return normalize_value_list(_field(record, "violation_context.value_current_2026"))
    return []


def _is_value_field(field: str) -> bool:
    return field.endswith(("new_value", ".value", "current_value_2026", "value_current_2026"))


def _aligned_with_retained_value(record: dict[str, Any], field: str, token: str) -> bool:
    if field not in A_BOX_TARGET_FIELDS:
        return False
    retained = set(derive_value_change_summary(record).retained_unique_values)
    if not retained:
        return False
    if _is_value_field(field):
        return token in retained
    raw_items = _field(record, field)
    items = raw_items if isinstance(raw_items, list) else [raw_items]
    values = _aligned_values_for_field(record, field)
    for index, item in enumerate(items):
        if index >= len(values) or values[index] not in retained:
            continue
        if any(_equivalent(token, scalar.strip()) for scalar in _scalars(item) if scalar.strip()):
            return True
    return False


def forbidden_claims(record: dict[str, Any]) -> list[dict[str, str]]:
    """Return auditable claims with value-delta-aware initial severities."""
    historical_sources = _historical_sources(record)
    high_risk_fields = set(COMMON_HIGH_RISK_FIELDS)
    if record.get("track") == "A_BOX":
        high_risk_fields.update(A_BOX_TARGET_FIELDS)
    fields = sorted(high_risk_fields | DIAGNOSTIC_FIELDS)
    claims: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field in fields:
        value = record.get("id") if field == "case_id" else _field(record, field)
        for token in _scalars(value):
            token = token.strip()
            key = (field, token)
            if _eligible_token(token) and key not in seen:
                seen.add(key)
                severity = "high" if field in high_risk_fields else "diagnostic"
                if _expected_rule_derived_visibility(record, field, token):
                    severity = "expected_rule_derived"
                elif _token_is_historical(token, historical_sources) or _aligned_with_retained_value(
                    record, field, token
                ):
                    severity = "expected_historical"
                claims.append(
                    {
                        "field": field,
                        "token": token,
                        "severity": severity,
                    }
                )
    return claims


def _input_payload_range(text: str) -> tuple[int, int] | None:
    marker = "Input case:\n"
    marker_start = text.rfind(marker)
    if marker_start < 0:
        return None
    start = marker_start + len(marker)
    while start < len(text) and text[start].isspace():
        start += 1
    try:
        _, length = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return start, start + length


def _source_token_for_coordinate(token: str, coordinate_space: str) -> str:
    if coordinate_space == "semantic_normalized":
        return _normalized_text(html.unescape(unquote_plus(token)))
    if coordinate_space == "html_unescape":
        return html.unescape(token)
    if coordinate_space == "url_unquote":
        return unquote(token)
    if coordinate_space == "url_unquote_plus":
        return unquote_plus(token)
    return token


def _source_token_surfaces(token: str, coordinate_space: str) -> list[str]:
    """Return decoded and JSON-rendered forms of a visible scalar."""
    candidates = [
        token,
        json.dumps(token, ensure_ascii=False)[1:-1],
        json.dumps(token, ensure_ascii=True)[1:-1],
    ]
    surfaces: list[str] = []
    for candidate in candidates:
        surface = _source_token_for_coordinate(candidate, coordinate_space)
        if surface and surface not in surfaces:
            surfaces.append(surface)
    return surfaces


def _covering_source(
    occurrence: dict[str, Any], sources: Iterable[dict[str, str]]
) -> dict[str, Any] | None:
    text = str(occurrence["_analyzed_text"])
    payload_range = _input_payload_range(text)
    occurrence_start = int(occurrence["start"])
    occurrence_end = int(occurrence["end"])
    if payload_range is not None and not (
        payload_range[0] <= occurrence_start and occurrence_end <= payload_range[1]
    ):
        return None
    for source in sources:
        for source_token in _source_token_surfaces(
            source["token"], str(occurrence["coordinate_space"])
        ):
            for start, end in _literal_spans(text, source_token, boundaries=False):
                if payload_range is not None and not (payload_range[0] <= start and end <= payload_range[1]):
                    continue
                if start <= occurrence_start and occurrence_end <= end:
                    return {
                        "field": source["field"],
                        "token": source["token"],
                        "span": {"start": start, "end": end},
                    }
    return None


def _input_payload(text: str) -> dict[str, Any] | None:
    payload_range = _input_payload_range(text)
    if payload_range is None:
        return None
    try:
        payload = json.loads(text[payload_range[0] : payload_range[1]])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _path_scalars(value: Any, path: str) -> Iterable[dict[str, str]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            yield from _path_scalars(nested, nested_path)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _path_scalars(nested, f"{path}[{index}]")
    elif value is not None:
        yield {"field": f"prompt_input.{path}", "token": str(value)}


def _visible_input_sources(text: str, *, local_only: bool = False) -> list[dict[str, str]]:
    payload = _input_payload(text)
    if payload is None:
        return []
    if local_only:
        local = payload.get("local_context")
        return list(_path_scalars(local, "local_context")) if local is not None else []
    return list(_path_scalars(payload, ""))


def _classification(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("classification")
    return value if isinstance(value, dict) else {}


def _format_rule_sources(record: dict[str, Any], token: str) -> list[dict[str, str]]:
    classification = _classification(record)
    if classification.get("class") != "TypeA" or classification.get("subtype") != "FORMAT_NORMALIZATION":
        return []
    return [
        source
        for source in _historical_sources(record)
        if not _equivalent(source["token"], token)
        and _normalized_text(token) in _normalized_text(source["token"])
    ]


def _recorded_rule_sources(record: dict[str, Any], token: str) -> list[dict[str, str]]:
    classification = _classification(record)
    if classification.get("class") != "TypeA":
        return []
    trace = classification.get("decision_trace") or classification.get("trace")
    if not isinstance(trace, list):
        return []
    sources: list[dict[str, str]] = []
    for step in trace:
        if not isinstance(step, dict) or step.get("step") != "rule_deterministic" or step.get("result") is not True:
            continue
        detail = step.get("detail")
        if not isinstance(detail, dict):
            continue
        allowed = detail.get("allowed_value")
        if allowed is not None and _equivalent(str(allowed), token):
            sources.append({"field": "classification.decision_trace.rule_input", "token": str(allowed)})
    return sources


_LOCAL_GRAPH_SOURCES = {
    "FOCUS_NON_TARGET_PROPERTY",
    "FOCUS_NON_TARGET_PROPERTY_TEXT",
    "NEIGHBOR_ID",
    "NEIGHBOR_LABEL",
    "NEIGHBOR_DESCRIPTION",
    "NEIGHBOR_PROPERTY_TEXT",
}
_ALWAYS_VISIBLE_LOCAL_SOURCES = {"FOCUS_QID", "FOCUS_LABEL", "FOCUS_DESCRIPTION"}


def _claim_aligned_values(record: dict[str, Any], field: str, token: str) -> set[str]:
    values = _aligned_values_for_field(record, field)
    if not values:
        return set()
    if _is_value_field(field):
        return {value for value in values if _equivalent(value, token)}
    raw_items = _field(record, field)
    items = raw_items if isinstance(raw_items, list) else [raw_items]
    return {
        values[index]
        for index, item in enumerate(items)
        if index < len(values)
        and any(_equivalent(token, scalar.strip()) for scalar in _scalars(item) if scalar.strip())
    }


def _recorded_local_evidence(
    record: dict[str, Any], field: str, token: str, context_bundle: str
) -> set[str]:
    classification = _classification(record)
    if classification.get("class") != "TypeB":
        return set()
    trace = classification.get("decision_trace") or classification.get("trace")
    if not isinstance(trace, list):
        return set()
    aligned_values = _claim_aligned_values(record, field, token)
    if not aligned_values:
        return set()
    source_names: set[str] = set()
    for step in trace:
        if not isinstance(step, dict) or step.get("step") != "local_availability" or step.get("result") is not True:
            continue
        evidence = step.get("evidence")
        matches = evidence.get("matches") if isinstance(evidence, dict) else None
        if not isinstance(matches, list):
            continue
        for match in matches:
            if not isinstance(match, dict) or match.get("independent_of_target_property") is not True:
                continue
            source_name = str(match.get("source") or "")
            if source_name in _LOCAL_GRAPH_SOURCES and context_bundle != "local_graph":
                continue
            if source_name not in _LOCAL_GRAPH_SOURCES | _ALWAYS_VISIBLE_LOCAL_SOURCES:
                continue
            match_token = str(match.get("token") or match.get("actual_target") or "")
            if match_token and any(_equivalent(match_token, value) for value in aligned_values):
                source_names.add(source_name)
    return source_names


_PROMPT_CONTRACT_VOCABULARY = {
    "family",
    "historical",
    "missing",
    "novalue",
    "null",
    "somevalue",
    "title",
    "type",
    "value",
}


def _diagnostic_occurrence_source(
    claim: dict[str, str], occurrence: dict[str, Any]
) -> tuple[str, dict[str, Any] | None] | None:
    token = _normalized_text(claim["token"])
    original_text = str(occurrence.get("_original_text") or occurrence["_analyzed_text"])
    if token in _PROMPT_CONTRACT_VOCABULARY:
        return "prompt_contract_vocabulary", None

    # A short lexical metadata value can coincide with one word inside a
    # longer visible phrase (for example editor ``Trade`` in ``World Trade
    # Center``).  The longer scalar, rather than the hidden metadata field, is
    # the source of that occurrence.  Distinct identifiers and full-scalar
    # matches remain blocking.
    if claim["field"] == "repair_target.author" and token.isalpha() and len(token) <= 8:
        for source in _visible_input_sources(original_text):
            source_token = _normalized_text(source["token"])
            if source_token == token:
                continue
            if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", source_token):
                return "short_word_within_visible_phrase", source
    return None


def _classify_occurrence(
    record: dict[str, Any], row: dict[str, Any], claim: dict[str, str], occurrence: dict[str, Any]
) -> tuple[str, str, dict[str, Any] | None]:
    if claim["severity"] == "diagnostic":
        return "diagnostic", "class_or_track_vocabulary", None
    if claim["severity"] == "expected_rule_derived":
        return "expected_rule_derived", "deterministic_target_required_identity", None
    if claim["severity"] == "expected_historical":
        return "expected_historical", "retained_or_historically_visible_value", None

    historical = _covering_source(occurrence, _historical_sources(record))
    format_source = _covering_source(occurrence, _format_rule_sources(record, claim["token"]))
    if format_source is not None:
        return "expected_rule_derived", "format_normalization_within_historical_literal", format_source
    if historical is not None:
        return "expected_historical", "occurrence_covered_by_historical_source_span", historical

    recorded_rule = _covering_source(occurrence, _recorded_rule_sources(record, claim["token"]))
    if recorded_rule is not None:
        return "expected_rule_derived", "recorded_rule_input_visible_in_prompt", recorded_rule

    context_bundle = str(row.get("context_bundle") or "")
    evidence_sources = _recorded_local_evidence(record, claim["field"], claim["token"], context_bundle)
    if evidence_sources:
        original_text = str(occurrence.get("_original_text") or occurrence["_analyzed_text"])
        local = _covering_source(
            occurrence,
            _visible_input_sources(
                original_text,
                local_only=evidence_sources.isdisjoint(_ALWAYS_VISIBLE_LOCAL_SOURCES),
            ),
        )
        if local is not None:
            return "expected_local_evidence", "independent_local_evidence_visible_in_bundle", local

    diagnostic = _diagnostic_occurrence_source(claim, occurrence)
    if diagnostic is not None:
        reason, source = diagnostic
        return "diagnostic", reason, source
    return "high", "unexplained_future_only_or_hidden_value", None


def _sample_rows(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0:
        return []
    # The protocol specifies a case sample, not a prompt-row sample. Select one
    # stable prompt variant per case before balancing across task/context/track.
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case_id = row.get("case_id")
        if isinstance(case_id, str) and case_id:
            rows_by_case[case_id].append(row)
    representatives: list[dict[str, Any]] = []
    for case_id, case_rows in rows_by_case.items():
        case_rows.sort(
            key=lambda row: hashlib.sha256(
                (
                    f"{seed}|temporal-case-row|{case_id}|{row.get('task')}|"
                    f"{row.get('context_bundle')}|{row.get('matrix_id')}"
                ).encode()
            ).hexdigest()
        )
        representatives.append(case_rows[0])
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in representatives:
        key = (
            str(row.get("task") or "unknown"),
            str(row.get("context_bundle") or "unknown"),
            str(row.get("historical_track") or "unknown"),
        )
        strata[key].append(row)
    for key, values in strata.items():
        values.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}|{key}|{row.get('matrix_id')}|{row.get('case_id')}".encode()
            ).hexdigest()
        )
    chosen: list[dict[str, Any]] = []
    keys = sorted(strata)
    while len(chosen) < min(sample_size, len(representatives)):
        progressed = False
        for key in keys:
            if strata[key] and len(chosen) < sample_size:
                chosen.append(strata[key].pop(0))
                progressed = True
        if not progressed:
            break
    return chosen


def audit_rendered_prompts(
    *,
    rendered_prompts_path: str | Path,
    classified_benchmark_path: str | Path,
    sample_size: int = 50,
    seed: int = 13,
) -> dict[str, Any]:
    prompt_path = Path(rendered_prompts_path)
    benchmark_path = Path(classified_benchmark_path)
    _progress(f"phase start: load rendered prompts path={prompt_path}")
    prompt_rows: list[dict[str, Any]] = []
    load_heartbeat = _Heartbeat("load_prompts")
    for row_number, row in enumerate(iter_jsonl(prompt_path), start=1):
        prompt_rows.append(row)
        load_heartbeat.update(row_number)
    _progress(f"phase complete: load rendered prompts rows={len(prompt_rows)}")
    case_ids = {str(row.get("case_id")) for row in prompt_rows if row.get("case_id")}
    records: dict[str, dict[str, Any]] = {}
    benchmark_digest = hashlib.sha256()
    _progress(f"phase start: bind classified records requested_cases={len(case_ids)} path={benchmark_path}")
    benchmark_heartbeat = _Heartbeat("bind_classified_records")
    with benchmark_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            benchmark_digest.update(raw_line)
            benchmark_heartbeat.update(line_number)
            if len(records) == len(case_ids) or not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {benchmark_path}:{line_number}") from exc
            case_id = record.get("id") if isinstance(record, dict) else None
            if case_id in case_ids:
                records[str(case_id)] = record
    _progress(f"phase complete: bind classified records matched_cases={len(records)}")

    missing_case_ids = sorted(case_ids - records.keys())
    hits: list[dict[str, Any]] = []
    scanned_claims = 0
    claim_cache: dict[str, list[dict[str, str]]] = {}
    _progress(f"phase start: scan prompt surfaces rows={len(prompt_rows)}")
    scan_heartbeat = _Heartbeat("scan_prompt_surfaces", total=len(prompt_rows))
    for row_index, row in enumerate(prompt_rows, start=1):
        scan_heartbeat.update(row_index)
        case_id = str(row.get("case_id") or "")
        record = records.get(case_id)
        if record is None:
            continue
        surfaces = {
            "system_prompt": str(row.get("system_prompt") or ""),
            "user_prompt": str(row.get("user_prompt") or ""),
        }
        claims = claim_cache.get(case_id)
        if claims is None:
            claims = forbidden_claims(record)
            claim_cache[case_id] = claims
        scanned_claims += len(claims)
        for claim in claims:
            for surface, text in surfaces.items():
                for occurrence in _occurrences(text, claim["token"]):
                    occurrence["_original_text"] = text
                    severity, classification_reason, covered_by = _classify_occurrence(
                        record, row, claim, occurrence
                    )
                    hit = {
                        "row": row_index,
                        "matrix_id": row.get("matrix_id"),
                        "case_id": case_id,
                        "task": row.get("task"),
                        "context_bundle": row.get("context_bundle"),
                        "surface": surface,
                        "field": claim["field"],
                        "token": claim["token"],
                        "severity": severity,
                        "classification_reason": classification_reason,
                        "occurrences": 1,
                        "match_mode": occurrence["match_mode"],
                        "coordinate_space": occurrence["coordinate_space"],
                        "span": {"start": occurrence["start"], "end": occurrence["end"]},
                        "matched_text": occurrence["matched_text"],
                    }
                    if covered_by is not None:
                        hit["covered_by"] = covered_by
                    hits.append(
                        hit
                    )
    _progress(f"phase complete: scan prompt surfaces raw_hits={len(hits)}")

    hit_counts = Counter(hit["severity"] for hit in hits)
    hits_by_field = Counter(hit["field"] for hit in hits)
    cases_by_severity = {
        severity: len({hit["case_id"] for hit in hits if hit["severity"] == severity})
        for severity in SEVERITIES
    }
    sample = []
    hit_rows = {hit["row"] for hit in hits}
    row_numbers = {id(row): index for index, row in enumerate(prompt_rows, start=1)}
    for row in _sample_rows(prompt_rows, sample_size, seed):
        row_number = row_numbers[id(row)]
        sample.append(
            {
                "row": row_number,
                "matrix_id": row.get("matrix_id"),
                "case_id": row.get("case_id"),
                "task": row.get("task"),
                "context_bundle": row.get("context_bundle"),
                "historical_track": row.get("historical_track"),
                "prompt_sha256": hashlib.sha256(
                    (str(row.get("system_prompt") or "") + "\n" + str(row.get("user_prompt") or "")).encode()
                ).hexdigest(),
                "automated_hit": row_number in hit_rows,
                "review_status": "pending_ai_review",
            }
        )

    mutation_checks = mutation_sensitivity_checks()
    high_case_ids = sorted({hit["case_id"] for hit in hits if hit["severity"] == "high"})
    case_exclusion_gate_passed = not missing_case_ids and all(mutation_checks.values())
    report = {
        "report_type": "temporal_prompt_leakage_audit",
        "report_version": TEMPORAL_REPORT_VERSION,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "rendered_prompts": str(prompt_path),
            "rendered_prompts_sha256": sha256_file(prompt_path),
            "classified_benchmark": str(benchmark_path),
            "classified_benchmark_sha256": benchmark_digest.hexdigest(),
        },
        "scope": {
            "automated_check": "source/span-aware exact, embedded, decoded, encoded, and semantic-normalized occurrence classification in model-visible prompt text",
            "ai_error_discovery_required": True,
            "ai_review_is_ground_truth": False,
            "temporal_claim": "later frozen context with historical target-property reconstruction",
            "blocking_severity": "high",
            "blocking_effect": "permanent_case_exclusion",
            "severity_categories": list(SEVERITIES),
        },
        "counts": {
            "prompt_rows": len(prompt_rows),
            "case_ids": len(case_ids),
            "matched_case_ids": len(records),
            "missing_case_ids": len(missing_case_ids),
            "forbidden_claims_scanned": scanned_claims,
            "raw_hits": len(hits),
            "unique_hit_cases": len({hit["case_id"] for hit in hits}),
            "high_risk_hits": hit_counts["high"],
            "high_unique_cases": cases_by_severity["high"],
            "expected_historical_hits": hit_counts["expected_historical"],
            "expected_historical_unique_cases": cases_by_severity["expected_historical"],
            "diagnostic_hits": hit_counts["diagnostic"],
            "diagnostic_unique_cases": cases_by_severity["diagnostic"],
            "expected_rule_derived_hits": hit_counts["expected_rule_derived"],
            "expected_rule_derived_unique_cases": cases_by_severity["expected_rule_derived"],
            "expected_local_evidence_hits": hit_counts["expected_local_evidence"],
            "expected_local_evidence_unique_cases": cases_by_severity["expected_local_evidence"],
            "manual_sample": len(sample),
        },
        "passed_automated_gate": not missing_case_ids
        and hit_counts["high"] == 0
        and all(mutation_checks.values()),
        "passed_case_exclusion_gate": case_exclusion_gate_passed,
        "excluded_case_ids": high_case_ids,
        "mutation_sensitivity_checks": mutation_checks,
        "missing_case_ids": missing_case_ids,
        "raw_hits_by_severity": {severity: hit_counts[severity] for severity in SEVERITIES},
        "unique_cases_by_severity": cases_by_severity,
        "hits_by_field": dict(sorted(hits_by_field.items())),
        "hits": hits,
        "manual_review_sample": sample,
    }
    _progress(
        "audit complete: "
        f"prompts={len(prompt_rows)} cases={len(records)} high_hits={hit_counts['high']} "
        f"passed={report['passed_automated_gate']}"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit rendered prompts for hidden temporal target leakage.")
    parser.add_argument("--rendered-prompts", required=True)
    parser.add_argument("--classified-benchmark", required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_rendered_prompts(
        rendered_prompts_path=args.rendered_prompts,
        classified_benchmark_path=args.classified_benchmark,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], sort_keys=True))
    return 0 if report["passed_automated_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
