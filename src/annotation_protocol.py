#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from classifier import WorldStateStore
from lib.manual_audit import HUMAN_ALLOWED_VALUES
from lib.utils import iter_jsonl

ASSIGNMENT_FIELDS = [
    "blinded_case_id",
    "evidence_card",
    *HUMAN_ALLOWED_VALUES,
    "notes",
    "annotator_id",
    "annotation_timestamp_utc",
]


def _stable_rank(seed: int, case_id: str, annotator: str) -> str:
    return hashlib.sha256(f"{seed}|{case_id}|{annotator}".encode()).hexdigest()


def _redact_case_id(value: Any, raw_case_id: str, blinded_case_id: str) -> Any:
    if isinstance(value, str):
        return value.replace(raw_case_id, blinded_case_id)
    if isinstance(value, list):
        return [_redact_case_id(item, raw_case_id, blinded_case_id) for item in value]
    if isinstance(value, dict):
        return {
            _redact_case_id(key, raw_case_id, blinded_case_id): _redact_case_id(item, raw_case_id, blinded_case_id)
            for key, item in value.items()
        }
    return value


def _load_audit_case_ids(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    case_ids = [row.get("case_id", "").strip() for row in rows]
    if any(not case_id for case_id in case_ids):
        raise ValueError("Every audit row must contain case_id.")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Audit input contains duplicate case_id values.")
    return case_ids


def build_blinded_assignments(
    *,
    audit_csv: str | Path,
    classified_path: str | Path,
    world_state_path: str | Path,
    output_dir: str | Path,
    annotators: list[str],
    seed: int = 13,
) -> dict[str, Any]:
    normalized_annotators = list(dict.fromkeys(value.strip() for value in annotators if value.strip()))
    if len(normalized_annotators) < 2:
        raise ValueError("At least two distinct annotators are required.")
    case_ids = _load_audit_case_ids(audit_csv)
    case_id_set = set(case_ids)
    records = {
        record["id"]: record
        for record in iter_jsonl(classified_path)
        if isinstance(record, dict) and record.get("id") in case_id_set
    }
    missing = sorted(case_id_set - set(records))
    if missing:
        raise ValueError(f"Classified benchmark is missing {len(missing)} audit cases.")

    output = Path(output_dir)
    cards_dir = output / "evidence_cards"
    assignments_dir = output / "assignments"
    cards_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir.mkdir(parents=True, exist_ok=True)
    assignment_rows: dict[str, list[dict[str, str]]] = {name: [] for name in normalized_annotators}
    private_map: dict[str, str] = {}
    assignment_counts: Counter[str] = Counter()

    world_store = WorldStateStore(Path(world_state_path), __import__("logging").getLogger("annotation_protocol"))
    world_store.open()
    try:
        for index, case_id in enumerate(case_ids, start=1):
            blind_id = f"audit_{index:06d}"
            private_map[blind_id] = case_id
            record = records[case_id]
            evidence = {
                key: value
                for key, value in record.items()
                if key not in {"id", "classification", "build", "track", "information_type"}
            }
            card = {
                "blinded_case_id": blind_id,
                "benchmark_evidence": _redact_case_id(evidence, case_id, blind_id),
                "frozen_world_state": _redact_case_id(world_store.get(case_id), case_id, blind_id),
                "blinding": {
                    "removed_fields": ["id", "classification", "build", "track", "information_type"],
                    "historical_repair_visible": True,
                },
            }
            card_path = cards_dir / f"{blind_id}.json"
            card_path.write_text(json.dumps(card, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
            reviewers = sorted(
                normalized_annotators,
                key=lambda annotator: (assignment_counts[annotator], _stable_rank(seed, case_id, annotator), annotator),
            )[:2]
            for reviewer in reviewers:
                row = {field: "" for field in ASSIGNMENT_FIELDS}
                row.update(
                    {
                        "blinded_case_id": blind_id,
                        "evidence_card": str(card_path.resolve()),
                        "annotator_id": reviewer,
                    }
                )
                assignment_rows[reviewer].append(row)
                assignment_counts[reviewer] += 1
    finally:
        world_store.close()

    for annotator, rows in assignment_rows.items():
        with (assignments_dir / f"{annotator}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ASSIGNMENT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    private_map_path = output / "private_case_map.json"
    private_map_path.write_text(json.dumps(private_map, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "manifest_type": "blinded_double_annotation_assignment",
        "manifest_version": 1,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "seed": seed,
        "case_count": len(case_ids),
        "review_count": len(case_ids) * 2,
        "reviews_per_case": 2,
        "annotators": normalized_annotators,
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "blinded": True,
        "private_case_map": str(private_map_path.resolve()),
        "adjudication_required_on_disagreement": True,
        "allowed_values": HUMAN_ALLOWED_VALUES,
    }
    (output / "protocol_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build blinded double-annotation assignments.")
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--classified-benchmark", default="data/04_classified_benchmark.jsonl")
    parser.add_argument("--world-state", default="data/03_world_state.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annotators", required=True, help="Comma-separated stable annotator IDs.")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    build_blinded_assignments(
        audit_csv=args.audit_csv,
        classified_path=args.classified_benchmark,
        world_state_path=args.world_state,
        output_dir=args.output_dir,
        annotators=args.annotators.split(","),
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
