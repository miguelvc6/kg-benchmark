"""Field-level leakage audit for rendered benchmark prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from artifact_release import sha256_file
from lib.utils import iter_jsonl

COMMON_HIGH_RISK_FIELDS = {
    "case_id",
    "repair_target.author",
    "repair_target.property_revision_id",
    "repair_target.property_revision_prev",
    "repair_target.constraint_delta.hash_after",
    "repair_target.constraint_delta.signature_after",
}
A_BOX_TARGET_FIELDS = {
    "repair_target.new_value",
    "repair_target.value",
    "persistence_check.current_value_2026",
    "violation_context.value_current_2026",
}


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


def _occurrence_count(text: str, token: str) -> int:
    if token and token[0].isalnum() and token[-1].isalnum():
        pattern = rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])"
        return len(re.findall(pattern, text))
    return text.count(token)


def _expected_rule_derived_visibility(record: dict[str, Any], field: str, token: str) -> bool:
    if field not in A_BOX_TARGET_FIELDS:
        return False
    classification = record.get("classification")
    classification = classification if isinstance(classification, dict) else {}
    if classification.get("class") != "TypeA":
        return False
    subtype = classification.get("subtype")
    if subtype == "TARGET_REQUIRED_CLAIM" and token == record.get("qid"):
        return True
    if subtype == "FORMAT_NORMALIZATION":
        old_values = list(_scalars(_field(record, "repair_target.old_value")))
        return any(token != old and token in old for old in old_values)
    return False


def forbidden_claims(record: dict[str, Any]) -> list[dict[str, str]]:
    """Return hidden values worth scanning without treating generic prompt vocabulary as truth."""
    pre_repair_tokens = {
        token.strip()
        for value in (
            _field(record, "repair_target.old_value"),
            _field(record, "violation_context.value"),
        )
        for token in _scalars(value)
        if token.strip()
    }
    post_repair_value_fields = {
        "repair_target.new_value",
        "repair_target.value",
        "persistence_check.current_value_2026",
        "violation_context.value_current_2026",
    }
    high_risk_fields = set(COMMON_HIGH_RISK_FIELDS)
    if record.get("track") == "A_BOX":
        high_risk_fields.update(A_BOX_TARGET_FIELDS)
    fields = sorted(
        high_risk_fields
        | {
            "classification.class",
            "classification.subtype",
            "information_type",
            "track",
        }
    )
    claims: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field in fields:
        value = record.get("id") if field == "case_id" else _field(record, field)
        for token in _scalars(value):
            token = token.strip()
            if field in post_repair_value_fields and token in pre_repair_tokens:
                continue
            key = (field, token)
            if _eligible_token(token) and key not in seen:
                seen.add(key)
                severity = "high" if field in high_risk_fields else "diagnostic"
                if _expected_rule_derived_visibility(record, field, token):
                    severity = "expected_rule_derived"
                claims.append(
                    {
                        "field": field,
                        "token": token,
                        "severity": severity,
                    }
                )
    return claims


def _sample_rows(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0:
        return []
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
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
    while len(chosen) < min(sample_size, len(rows)):
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
    prompt_rows = list(iter_jsonl(prompt_path))
    case_ids = {str(row.get("case_id")) for row in prompt_rows if row.get("case_id")}
    records: dict[str, dict[str, Any]] = {}
    benchmark_digest = hashlib.sha256()
    with benchmark_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            benchmark_digest.update(raw_line)
            if len(records) == len(case_ids) or not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {benchmark_path}:{line_number}") from exc
            case_id = record.get("id") if isinstance(record, dict) else None
            if case_id in case_ids:
                records[str(case_id)] = record

    missing_case_ids = sorted(case_ids - records.keys())
    hits: list[dict[str, Any]] = []
    scanned_claims = 0
    for row_index, row in enumerate(prompt_rows, start=1):
        case_id = str(row.get("case_id") or "")
        record = records.get(case_id)
        if record is None:
            continue
        surfaces = {
            "system_prompt": str(row.get("system_prompt") or ""),
            "user_prompt": str(row.get("user_prompt") or ""),
        }
        claims = forbidden_claims(record)
        scanned_claims += len(claims)
        for claim in claims:
            for surface, text in surfaces.items():
                count = _occurrence_count(text, claim["token"])
                if count:
                    hits.append(
                        {
                            "row": row_index,
                            "matrix_id": row.get("matrix_id"),
                            "case_id": case_id,
                            "task": row.get("task"),
                            "context_bundle": row.get("context_bundle"),
                            "surface": surface,
                            "field": claim["field"],
                            "token": claim["token"],
                            "severity": claim["severity"],
                            "occurrences": count,
                        }
                    )

    hit_counts = Counter(hit["severity"] for hit in hits)
    hits_by_field = Counter(hit["field"] for hit in hits)
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
                "review_status": "pending_human_review",
            }
        )

    return {
        "report_type": "temporal_prompt_leakage_audit",
        "report_version": 1,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "rendered_prompts": str(prompt_path),
            "rendered_prompts_sha256": sha256_file(prompt_path),
            "classified_benchmark": str(benchmark_path),
            "classified_benchmark_sha256": benchmark_digest.hexdigest(),
        },
        "scope": {
            "automated_check": "exact hidden-field token occurrence in model-visible prompt text",
            "manual_review_required": True,
            "temporal_claim": "later frozen context with historical target-property reconstruction",
        },
        "counts": {
            "prompt_rows": len(prompt_rows),
            "case_ids": len(case_ids),
            "matched_case_ids": len(records),
            "missing_case_ids": len(missing_case_ids),
            "forbidden_claims_scanned": scanned_claims,
            "high_risk_hits": hit_counts["high"],
            "diagnostic_hits": hit_counts["diagnostic"],
            "expected_rule_derived_hits": hit_counts["expected_rule_derived"],
            "manual_sample": len(sample),
        },
        "passed_automated_gate": not missing_case_ids and hit_counts["high"] == 0,
        "missing_case_ids": missing_case_ids,
        "hits_by_field": dict(sorted(hits_by_field.items())),
        "hits": hits,
        "manual_review_sample": sample,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit rendered prompts for hidden temporal target leakage.")
    parser.add_argument("--rendered-prompts", required=True)
    parser.add_argument("--classified-benchmark", required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-hits", action="store_true")
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
    return 0 if report["passed_automated_gate"] or args.allow_hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
