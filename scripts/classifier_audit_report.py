#!/usr/bin/env python3
"""Summarize and compare Stage 4 classifier outputs for Phase B audits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

JSONL_DECODER = json.JSONDecoder(strict=False)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = JSONL_DECODER.decode(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL record {line_number} in {path}: {exc}") from exc
            if isinstance(payload, dict):
                yield payload


def _branch(classification: dict[str, Any]) -> str:
    for step in classification.get("decision_trace", []):
        if isinstance(step, dict) and step.get("step") == "branch":
            return str(step.get("result"))
    return "missing"


def _truth_source(classification: dict[str, Any]) -> str:
    diagnostics = classification.get("diagnostics")
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("truth_source"), str):
        return diagnostics["truth_source"]
    return "missing"


def _case_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    classification = record.get("classification")
    if not isinstance(classification, dict):
        classification = {}
    return (
        str(classification.get("class")),
        str(classification.get("subtype")),
        str(classification.get("confidence")),
        str(classification.get("local_subtype")),
    )


def _new_summary_state() -> dict[str, Any]:
    return {
        "counts": Counter(),
        "typec_subtypes": Counter(),
        "typea_subtypes": Counter(),
        "typec_low_confidence": 0,
        "current_value_truth_fallback": 0,
        "missing_world_state": 0,
        "missing_truth": 0,
        "typea_deletes": 0,
        "typea_format_repairs": 0,
    }


def _accumulate_summary(state: dict[str, Any], record: dict[str, Any]) -> None:
    counts: Counter[str] = state["counts"]
    typec_subtypes: Counter[str] = state["typec_subtypes"]
    typea_subtypes: Counter[str] = state["typea_subtypes"]

    classification = record.get("classification")
    if not isinstance(classification, dict):
        classification = {}
    cls = classification.get("class")
    subtype = classification.get("subtype")
    confidence = classification.get("confidence")
    truth_source = _truth_source(classification)
    branch = _branch(classification)
    local_subtype = classification.get("local_subtype")
    track = record.get("track")

    counts["records"] += 1
    counts[f"class:{cls}"] += 1
    counts[f"subtype:{subtype}"] += 1
    counts[f"confidence:{confidence}"] += 1
    counts[f"truth_source:{truth_source}"] += 1
    counts[f"decision_branch:{branch}"] += 1
    counts[f"track:{track}"] += 1
    if isinstance(local_subtype, str):
        counts[f"local_subtype:{local_subtype}"] += 1

    if cls == "TypeC":
        typec_subtypes[str(subtype)] += 1
        if confidence == "low":
            state["typec_low_confidence"] += 1
    if cls == "TypeA":
        typea_subtypes[str(subtype)] += 1
    if truth_source in {
        "persistence_check.current_value_2026",
        "violation_context.value_current_2026",
        "persistence_check.current_value_2025",
        "violation_context.value_current_2025",
    }:
        state["current_value_truth_fallback"] += 1
    if subtype == "UNKNOWN_MISSING_WORLD_STATE":
        state["missing_world_state"] += 1
    if subtype == "UNKNOWN_MISSING_TRUTH":
        state["missing_truth"] += 1
    repair_target = record.get("repair_target")
    if isinstance(repair_target, dict) and repair_target.get("action") == "DELETE" and cls == "TypeA":
        state["typea_deletes"] += 1
    if cls == "TypeA":
        trace_text = json.dumps(classification.get("decision_trace", []), sort_keys=True)
        if subtype in {"LOGICAL", "FORMAT", "REJECTION_FORMAT_INVALID"} and "FORMAT" in trace_text.upper():
            state["typea_format_repairs"] += 1


def _finish_summary(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    counts: Counter[str] = state["counts"]
    typec_subtypes: Counter[str] = state["typec_subtypes"]
    typea_subtypes: Counter[str] = state["typea_subtypes"]
    return {
        "source": str(path),
        "counts": dict(sorted(counts.items())),
        "metrics": {
            "typec_total": sum(typec_subtypes.values()),
            "typec_subtypes": dict(sorted(typec_subtypes.items())),
            "typec_low_confidence": state["typec_low_confidence"],
            "current_value_truth_fallback": state["current_value_truth_fallback"],
            "missing_world_state": state["missing_world_state"],
            "missing_truth": state["missing_truth"],
            "typea_deletes": state["typea_deletes"],
            "typea_format_repairs": state["typea_format_repairs"],
            "typea_subtypes": dict(sorted(typea_subtypes.items())),
        },
    }


def summarize(path: Path) -> dict[str, Any]:
    state = _new_summary_state()
    for record in _iter_jsonl(path):
        _accumulate_summary(state, record)
    return _finish_summary(path, state)


def compare(old_path: Path, new_path: Path, *, example_limit: int = 5) -> dict[str, Any]:
    old_by_id: dict[str, tuple[str, str, str, str]] = {}
    for record in _iter_jsonl(old_path):
        rid = record.get("id")
        if isinstance(rid, str):
            old_by_id[rid] = _case_key(record)

    matrix: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    matched = 0
    missing_old = 0
    for record in _iter_jsonl(new_path):
        rid = record.get("id")
        if not isinstance(rid, str) or rid not in old_by_id:
            missing_old += 1
            continue
        old_key = old_by_id[rid]
        new_key = _case_key(record)
        transition = f"{old_key[0]}/{old_key[1]} -> {new_key[0]}/{new_key[1]}"
        matrix[transition] += 1
        matched += 1
        if len(examples[transition]) < example_limit:
            examples[transition].append(rid)

    return {
        "old_source": str(old_path),
        "new_source": str(new_path),
        "matched_records": matched,
        "new_records_missing_old": missing_old,
        "transition_matrix": dict(sorted(matrix.items())),
        "examples": dict(sorted(examples.items())),
    }


def compare_with_summaries(old_path: Path, new_path: Path, *, example_limit: int = 5) -> dict[str, Any]:
    old_by_id: dict[str, tuple[str, str, str, str]] = {}
    old_summary = _new_summary_state()
    for record in _iter_jsonl(old_path):
        _accumulate_summary(old_summary, record)
        rid = record.get("id")
        if isinstance(rid, str):
            old_by_id[rid] = _case_key(record)

    new_summary = _new_summary_state()
    matrix: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    matched = 0
    missing_old = 0
    for record in _iter_jsonl(new_path):
        _accumulate_summary(new_summary, record)
        rid = record.get("id")
        if not isinstance(rid, str) or rid not in old_by_id:
            missing_old += 1
            continue
        old_key = old_by_id[rid]
        new_key = _case_key(record)
        transition = f"{old_key[0]}/{old_key[1]} -> {new_key[0]}/{new_key[1]}"
        matrix[transition] += 1
        matched += 1
        if len(examples[transition]) < example_limit:
            examples[transition].append(rid)

    return {
        "baseline": _finish_summary(old_path, old_summary),
        "redesigned": _finish_summary(new_path, new_summary),
        "comparison": {
            "old_source": str(old_path),
            "new_source": str(new_path),
            "matched_records": matched,
            "new_records_missing_old": missing_old,
            "transition_matrix": dict(sorted(matrix.items())),
            "examples": dict(sorted(examples.items())),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Classified benchmark JSONL to summarize.")
    parser.add_argument("--compare-to", type=Path, help="New classified benchmark JSONL to compare against --input.")
    parser.add_argument("--out", type=Path, required=True, help="Report JSON output path.")
    parser.add_argument("--example-limit", type=int, default=5)
    args = parser.parse_args()

    if args.compare_to:
        report = compare_with_summaries(args.input, args.compare_to, example_limit=args.example_limit)
    else:
        report = {"baseline": summarize(args.input)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
