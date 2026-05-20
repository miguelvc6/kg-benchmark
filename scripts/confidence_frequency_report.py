#!/usr/bin/env python3
"""Report classifier confidence frequencies from a Stage 4 benchmark JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "data/04_classified_benchmark.jsonl"
DEFAULT_OUTPUT = "reports/confidence_frequency_report.json"
MISSING = "__missing__"
TRACK_RE = re.compile(r'"track"\s*:\s*"([^"]+)"')
CLASS_RE = re.compile(r'"class"\s*:\s*"([^"]+)"')
SUBTYPE_RE = re.compile(r'"subtype"\s*:\s*"([^"]+)"')
CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*"([^"]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a classified benchmark JSONL artifact and count classification "
            "confidence values overall and by class, track, subtype, and combined strata."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the Stage 4 classified benchmark JSONL artifact.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to write the JSON confidence-frequency report.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print a progress update every N input records. Set to 0 to disable.",
    )
    return parser.parse_args()


def _clean_key(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value is None:
        return MISSING
    return str(value)


def _match_value(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        return MISSING
    return _clean_key(match.group(1))


def _classification_slice(line: str) -> str:
    start = line.find('"classification"')
    if start < 0:
        return ""
    brace = line.find("{", start)
    if brace < 0:
        return ""
    # The fields we need appear at the start of the classification object.
    # Limiting the slice avoids scanning large decision traces or embedded context.
    return line[brace + 1 : brace + 2001]


def _extract_fields(line: str) -> tuple[str, str, str, str]:
    classification_text = _classification_slice(line)
    return (
        _match_value(CONFIDENCE_RE, classification_text),
        _match_value(CLASS_RE, classification_text),
        _match_value(SUBTYPE_RE, classification_text),
        _match_value(TRACK_RE, line[:2000]),
    )


def _counter_payload(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    values = {}
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        values[key] = {
            "count": count,
            "fraction": (count / total) if total else 0.0,
        }
    return {
        "total": total,
        "values": values,
    }


def _nested_payload(strata: dict[str, Counter[str]]) -> dict[str, Any]:
    return {
        key: _counter_payload(counter)
        for key, counter in sorted(strata.items(), key=lambda item: item[0])
    }


def build_report(input_path: str | Path, progress_every: int = 100000) -> dict[str, Any]:
    input_path = Path(input_path)
    overall: Counter[str] = Counter()
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    by_track: dict[str, Counter[str]] = defaultdict(Counter)
    by_subtype: dict[str, Counter[str]] = defaultdict(Counter)
    by_class_and_track: dict[str, Counter[str]] = defaultdict(Counter)
    by_class_and_subtype: dict[str, Counter[str]] = defaultdict(Counter)
    missing_classification_lines = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if progress_every > 0 and line_number % progress_every == 0:
                print(f"[progress] records={line_number:,}")

            confidence, cls, subtype, track = _extract_fields(line)
            if confidence == MISSING and cls == MISSING and subtype == MISSING:
                missing_classification_lines += 1

            overall[confidence] += 1
            by_class[cls][confidence] += 1
            by_track[track][confidence] += 1
            by_subtype[subtype][confidence] += 1
            by_class_and_track[f"{cls}|{track}"][confidence] += 1
            by_class_and_subtype[f"{cls}|{subtype}"][confidence] += 1

    return {
        "report_type": "classification_confidence_frequency",
        "report_version": 1,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input": str(input_path),
        "missing_classification_lines": missing_classification_lines,
        "overall": _counter_payload(overall),
        "strata": {
            "by_class": _nested_payload(by_class),
            "by_track": _nested_payload(by_track),
            "by_subtype": _nested_payload(by_subtype),
            "by_class_and_track": _nested_payload(by_class_and_track),
            "by_class_and_subtype": _nested_payload(by_class_and_subtype),
        },
    }


def main() -> int:
    args = parse_args()
    report = build_report(args.input, progress_every=args.progress_every)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {output_path}")
    print(f"[done] records={report['overall']['total']:,}")
    print(f"[done] missing_classification_lines={report['missing_classification_lines']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
