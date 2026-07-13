#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from artifact_release import sha256_file
from classifier import WorldStateStore
from lib.benchmark_selection import derive_case_metadata
from lib.utils import iter_jsonl

TASK_ALLOWED_VALUES: dict[str, dict[str, list[str]]] = {
    "locus": {
        "predicted_locus": ["A_BOX", "T_BOX", "AMBIGUOUS"],
        "locus_confidence": ["high", "medium", "low"],
    },
    "evidence_sufficiency": {
        "historical_target_well_defined": ["yes", "no", "unclear"],
        "target_visible_locally": ["yes", "no", "partial", "unclear"],
        "external_evidence_required": ["yes", "no", "maybe", "unclear", "not_applicable"],
        "core_recommendation": ["main", "diagnostic", "exclude", "needs_discussion"],
    },
    "tbox_validity": {
        "causal_linkage": ["causal", "plausible", "coincidental", "unclear"],
        "historical_repair_semantically_valid": ["yes", "no", "partial", "unclear"],
        "alternative_valid_repairs": ["none_known", "one_or_more", "unclear"],
    },
}
BASE_FIELDS = [
    "task",
    "blinded_case_id",
    "evidence_card",
    "annotator_id",
    *sorted({field for fields in TASK_ALLOWED_VALUES.values() for field in fields}),
    "alternative_repair_description",
    "notes",
    "annotation_timestamp_utc",
]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_rank(seed: int, case_id: str, annotator: str, task: str) -> str:
    return hashlib.sha256(f"{seed}|{case_id}|{annotator}|{task}".encode()).hexdigest()


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


def _shuffled_mapping(value: dict[str, Any], *, seed: int, key: str) -> dict[str, Any]:
    items = list(value.items())
    random.Random(f"{seed}|{key}").shuffle(items)
    return dict(items)


def _load_audit_case_ids(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    case_ids = [row.get("case_id", "").strip() for row in rows]
    if any(not case_id for case_id in case_ids):
        raise ValueError("Every audit row must contain case_id.")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Audit input contains duplicate case_id values.")
    return case_ids


def _task_evidence(record: dict[str, Any], task: str) -> tuple[dict[str, Any], list[str]]:
    removed = {"id", "classification", "build", "track", "information_type"}
    evidence = {key: value for key, value in record.items() if key not in removed}
    if task == "locus":
        evidence.pop("repair_target", None)
        removed.add("repair_target")
    else:
        repair_target = evidence.get("repair_target")
        if isinstance(repair_target, dict):
            evidence["repair_target"] = {
                key: value for key, value in repair_target.items() if key not in {"kind", "author"}
            }
            removed.update({"repair_target.kind", "repair_target.author"})
    return evidence, sorted(removed)


def build_annotation_assignments(
    *,
    audit_csv: str | Path,
    classified_path: str | Path,
    world_state_path: str | Path,
    output_dir: str | Path,
    private_map_path: str | Path,
    annotators: list[str],
    tasks: Iterable[str] = ("locus", "evidence_sufficiency", "tbox_validity"),
    seed: int = 13,
) -> dict[str, Any]:
    normalized_annotators = list(dict.fromkeys(value.strip() for value in annotators if value.strip()))
    if len(normalized_annotators) < 2:
        raise ValueError("At least two distinct annotators are required.")
    normalized_tasks = list(dict.fromkeys(tasks))
    if not normalized_tasks or any(task not in TASK_ALLOWED_VALUES for task in normalized_tasks):
        raise ValueError(f"Tasks must be selected from {sorted(TASK_ALLOWED_VALUES)}.")
    output = Path(output_dir).resolve()
    private_path = Path(private_map_path).resolve()
    if private_path == output or output in private_path.parents:
        raise ValueError("The private case map must be stored outside the assignment output directory.")

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

    cards_dir = output / "evidence_cards"
    assignments_dir = output / "assignments"
    cards_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir.mkdir(parents=True, exist_ok=True)
    assignment_rows: dict[str, list[dict[str, str]]] = {name: [] for name in normalized_annotators}
    private_map: dict[str, dict[str, Any]] = {}
    assignment_counts: Counter[str] = Counter()
    expected_assignments: list[dict[str, str]] = []

    with WorldStateStore(Path(world_state_path), __import__("logging").getLogger("annotation_protocol")) as world_store:
        for index, case_id in enumerate(case_ids, start=1):
            blind_id = f"audit_{index:06d}"
            record = records[case_id]
            metadata = derive_case_metadata(record, tier="core") or {}
            private_map[blind_id] = {
                "case_id": case_id,
                "group_key": metadata.get("group_key") or case_id,
                "selection_stratum": metadata.get("selection_stratum") or "unknown",
                "historical_track": record.get("track"),
            }
            applicable_tasks = [
                task
                for task in normalized_tasks
                if task != "tbox_validity" or record.get("track") == "T_BOX"
            ]
            for task in applicable_tasks:
                evidence, removed = _task_evidence(record, task)
                card = {
                    "task": task,
                    "blinded_case_id": blind_id,
                    "benchmark_evidence": _redact_case_id(
                        _shuffled_mapping(evidence, seed=seed, key=f"{case_id}|{task}"), case_id, blind_id
                    ),
                    "frozen_world_state": _redact_case_id(world_store.get(case_id), case_id, blind_id),
                    "blinding": {
                        "removed_fields": removed,
                        "historical_target_visible": task != "locus",
                        "task_interpretation": (
                            "blind_locus_inference"
                            if task == "locus"
                            else "target_conditioned_evidence_review"
                            if task == "evidence_sufficiency"
                            else "property_revision_tbox_validity_review"
                        ),
                    },
                    "response_contract": TASK_ALLOWED_VALUES[task],
                }
                task_cards = cards_dir / task
                task_cards.mkdir(parents=True, exist_ok=True)
                card_path = task_cards / f"{blind_id}.json"
                card_path.write_text(json.dumps(card, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
                reviewers = sorted(
                    normalized_annotators,
                    key=lambda annotator: (
                        assignment_counts[annotator],
                        _stable_rank(seed, case_id, annotator, task),
                        annotator,
                    ),
                )[:2]
                for reviewer in reviewers:
                    row = {field: "" for field in BASE_FIELDS}
                    row.update(
                        {
                            "task": task,
                            "blinded_case_id": blind_id,
                            "evidence_card": f"../evidence_cards/{task}/{blind_id}.json",
                            "annotator_id": reviewer,
                        }
                    )
                    assignment_rows[reviewer].append(row)
                    assignment_counts[reviewer] += 1
                    expected_assignments.append(
                        {"task": task, "blinded_case_id": blind_id, "annotator_id": reviewer}
                    )

    for annotator, rows in assignment_rows.items():
        with (assignments_dir / f"{annotator}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BASE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(json.dumps(private_map, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.chmod(private_path, 0o600)
    manifest = {
        "manifest_type": "independent_double_annotation_assignment",
        "manifest_version": 2,
        "created_at_utc": _utc_now(),
        "seed": seed,
        "case_count": len(case_ids),
        "review_count": len(expected_assignments),
        "reviews_per_case_task": 2,
        "annotators": normalized_annotators,
        "tasks": normalized_tasks,
        "task_contracts": TASK_ALLOWED_VALUES,
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "expected_assignments": expected_assignments,
        "blinded": True,
        "private_case_map_sha256": sha256_file(private_path),
        "private_case_map_location_recorded": False,
        "adjudication_required_on_disagreement": True,
    }
    (output / "protocol_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({value for pair in pairs for value in pair})
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum((left_counts[label] / len(pairs)) * (right_counts[label] / len(pairs)) for label in labels)
    return None if math.isclose(expected, 1.0) else (observed - expected) / (1.0 - expected)


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _cluster_agreement_interval(
    rows: list[dict[str, Any]], *, field: str, seed: int, samples: int = 2000
) -> list[float | None]:
    by_group: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_group[row["group_key"]].append(row["responses"][0][field] == row["responses"][1][field])
    keys = sorted(by_group)
    if not keys:
        return [None, None]
    rng = random.Random(f"{seed}|{field}")
    estimates = []
    for _ in range(samples):
        sampled = [value for key in (rng.choice(keys) for _ in keys) for value in by_group[key]]
        estimates.append(sum(sampled) / len(sampled))
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def merge_completed_reviews(
    *,
    assignment_manifest_path: str | Path,
    completed_review_paths: Iterable[str | Path],
    private_map_path: str | Path,
    output_dir: str | Path,
    bootstrap_seed: int = 13,
) -> dict[str, Any]:
    assignment = json.loads(Path(assignment_manifest_path).read_text(encoding="utf-8"))
    private_path = Path(private_map_path)
    if sha256_file(private_path) != assignment.get("private_case_map_sha256"):
        raise ValueError("Private case map hash does not match the assignment manifest.")
    private_map = json.loads(private_path.read_text(encoding="utf-8"))
    expected = {
        (row["task"], row["blinded_case_id"], row["annotator_id"])
        for row in assignment.get("expected_assignments", [])
    }
    received: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in completed_review_paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("task", ""), row.get("blinded_case_id", ""), row.get("annotator_id", ""))
                if key not in expected:
                    raise ValueError(f"Unexpected completed review assignment: {key}")
                if key in received:
                    raise ValueError(f"Duplicate completed review assignment: {key}")
                contract = TASK_ALLOWED_VALUES[key[0]]
                for field, allowed in contract.items():
                    if row.get(field) not in allowed:
                        raise ValueError(f"Invalid or missing {field} for assignment {key}: {row.get(field)!r}")
                if key[0] == "tbox_validity" and row.get("alternative_valid_repairs") == "one_or_more":
                    if not row.get("alternative_repair_description", "").strip():
                        raise ValueError(f"Alternative repair description is required for assignment {key}.")
                received[key] = row
    missing = sorted(expected - set(received))
    if missing:
        raise ValueError(f"Missing {len(missing)} completed reviews; first={missing[0]}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for (task, blind_id, _), row in received.items():
        grouped[(task, blind_id)].append(row)
    merged_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, str]] = []
    metrics: dict[str, dict[str, Any]] = defaultdict(dict)
    for (task, blind_id), rows in sorted(grouped.items()):
        if len(rows) != 2 or len({row["annotator_id"] for row in rows}) != 2:
            raise ValueError(f"Each case/task requires two distinct reviewers: {(task, blind_id)}")
        rows.sort(key=lambda row: row["annotator_id"])
        responses = [
            {
                **{field: row[field] for field in TASK_ALLOWED_VALUES[task]},
                "alternative_repair_description": row.get("alternative_repair_description", ""),
                "notes": row.get("notes", ""),
                "annotator_id": row["annotator_id"],
                "annotation_timestamp_utc": row.get("annotation_timestamp_utc", ""),
            }
            for row in rows
        ]
        metadata = private_map[blind_id]
        merged = {
            "task": task,
            "blinded_case_id": blind_id,
            "group_key": metadata["group_key"],
            "selection_stratum": metadata["selection_stratum"],
            "responses": responses,
        }
        merged_rows.append(merged)
        for field in TASK_ALLOWED_VALUES[task]:
            if responses[0][field] != responses[1][field]:
                disagreement_rows.append(
                    {
                        "task": task,
                        "blinded_case_id": blind_id,
                        "field": field,
                        "reviewer_a": responses[0]["annotator_id"],
                        "value_a": responses[0][field],
                        "reviewer_b": responses[1]["annotator_id"],
                        "value_b": responses[1][field],
                        "adjudicated_value": "",
                        "adjudicator_id": "",
                        "rationale": "",
                        "adjudication_timestamp_utc": "",
                    }
                )

    for task, contract in TASK_ALLOWED_VALUES.items():
        task_rows = [row for row in merged_rows if row["task"] == task]
        for field in contract:
            pairs = [(row["responses"][0][field], row["responses"][1][field]) for row in task_rows]
            metrics[task][field] = {
                "n": len(pairs),
                "percent_agreement": sum(left == right for left, right in pairs) / len(pairs) if pairs else None,
                "cohen_kappa": _cohen_kappa(pairs),
                "cluster_bootstrap_agreement_ci_95": _cluster_agreement_interval(
                    task_rows, field=field, seed=bootstrap_seed
                ),
            }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged_path = output / "merged_reviews.jsonl"
    merged_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in merged_rows), encoding="utf-8"
    )
    disagreements_path = output / "disagreements.csv"
    disagreement_fields = [
        "task",
        "blinded_case_id",
        "field",
        "reviewer_a",
        "value_a",
        "reviewer_b",
        "value_b",
        "adjudicated_value",
        "adjudicator_id",
        "rationale",
        "adjudication_timestamp_utc",
    ]
    with disagreements_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=disagreement_fields)
        writer.writeheader()
        writer.writerows(disagreement_rows)
    report = {
        "manifest_type": "independent_annotation_agreement",
        "manifest_version": 1,
        "created_at_utc": _utc_now(),
        "case_task_count": len(merged_rows),
        "review_count": len(received),
        "disagreement_count": len(disagreement_rows),
        "metrics": metrics,
        "merged_reviews_sha256": sha256_file(merged_path),
        "disagreements_sha256": sha256_file(disagreements_path),
    }
    (output / "agreement_report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return report


def adjudicate_reviews(
    *,
    merged_reviews_path: str | Path,
    completed_disagreements_path: str | Path,
    private_map_path: str | Path,
    agreement_report_path: str | Path,
    private_output: str | Path,
    public_output: str | Path,
) -> dict[str, Any]:
    merged = [json.loads(line) for line in Path(merged_reviews_path).read_text(encoding="utf-8").splitlines() if line]
    private_map = json.loads(Path(private_map_path).read_text(encoding="utf-8"))
    decisions: dict[tuple[str, str, str], dict[str, str]] = {}
    with Path(completed_disagreements_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["task"], row["blinded_case_id"], row["field"])
            allowed = TASK_ALLOWED_VALUES.get(row["task"], {}).get(row["field"], [])
            if row.get("adjudicated_value") not in allowed:
                raise ValueError(f"Invalid adjudicated value for {key}: {row.get('adjudicated_value')!r}")
            if not row.get("adjudicator_id") or not row.get("rationale") or not row.get("adjudication_timestamp_utc"):
                raise ValueError(f"Adjudication sign-off is incomplete for {key}.")
            if row["adjudicator_id"] in {row.get("reviewer_a"), row.get("reviewer_b")}:
                raise ValueError(f"Adjudicator must be independent of both reviewers for {key}.")
            decisions[key] = row

    final_rows = []
    adjudicators: set[str] = set()
    distributions: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in merged:
        task = row["task"]
        blind_id = row["blinded_case_id"]
        labels = {}
        for field in TASK_ALLOWED_VALUES[task]:
            values = [response[field] for response in row["responses"]]
            if values[0] == values[1]:
                labels[field] = values[0]
            else:
                key = (task, blind_id, field)
                if key not in decisions:
                    raise ValueError(f"Missing adjudication for {key}")
                labels[field] = decisions[key]["adjudicated_value"]
                adjudicators.add(decisions[key]["adjudicator_id"])
            distributions[task][field][labels[field]] += 1
        final_rows.append(
            {
                "case_id": private_map[blind_id]["case_id"],
                "task": task,
                "selection_stratum": private_map[blind_id]["selection_stratum"],
                "group_key": private_map[blind_id]["group_key"],
                "final_labels": labels,
                "reviewer_ids": [response["annotator_id"] for response in row["responses"]],
            }
        )
    private_path = Path(private_output)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in final_rows), encoding="utf-8"
    )
    os.chmod(private_path, 0o600)
    public = {
        "manifest_type": "independent_annotation_adjudicated_summary",
        "manifest_version": 1,
        "created_at_utc": _utc_now(),
        "case_task_count": len(final_rows),
        "label_distributions": {
            task: {field: dict(sorted(counts.items())) for field, counts in fields.items()}
            for task, fields in distributions.items()
        },
        "adjudication_signoff": {
            "adjudicator_ids": sorted(adjudicators),
            "completed": True,
            "completed_at_utc": _utc_now(),
        },
        "agreement_report_sha256": sha256_file(agreement_report_path),
        "private_final_labels_sha256": sha256_file(private_path),
    }
    public["artifact_signature"] = hashlib.sha256(
        json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    public_path = Path(public_output)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(json.dumps(public, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return public


def main() -> int:
    parser = argparse.ArgumentParser(description="Build, merge, or adjudicate independent annotation tasks.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--audit-csv", required=True)
    build.add_argument("--classified-benchmark", default="data/04_classified_benchmark.jsonl")
    build.add_argument("--world-state", default="data/03_world_state.json")
    build.add_argument("--output-dir", required=True)
    build.add_argument("--private-map", required=True)
    build.add_argument("--annotators", required=True)
    build.add_argument("--tasks", default=",".join(TASK_ALLOWED_VALUES))
    build.add_argument("--seed", type=int, default=13)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--assignment-manifest", required=True)
    merge.add_argument("--completed-review", action="append", required=True)
    merge.add_argument("--private-map", required=True)
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--bootstrap-seed", type=int, default=13)
    adjudicate = subparsers.add_parser("adjudicate")
    adjudicate.add_argument("--merged-reviews", required=True)
    adjudicate.add_argument("--completed-disagreements", required=True)
    adjudicate.add_argument("--private-map", required=True)
    adjudicate.add_argument("--agreement-report", required=True)
    adjudicate.add_argument("--private-output", required=True)
    adjudicate.add_argument("--public-output", required=True)
    args = parser.parse_args()
    if args.command == "build":
        build_annotation_assignments(
            audit_csv=args.audit_csv,
            classified_path=args.classified_benchmark,
            world_state_path=args.world_state,
            output_dir=args.output_dir,
            private_map_path=args.private_map,
            annotators=args.annotators.split(","),
            tasks=args.tasks.split(","),
            seed=args.seed,
        )
    elif args.command == "merge":
        merge_completed_reviews(
            assignment_manifest_path=args.assignment_manifest,
            completed_review_paths=args.completed_review,
            private_map_path=args.private_map,
            output_dir=args.output_dir,
            bootstrap_seed=args.bootstrap_seed,
        )
    else:
        adjudicate_reviews(
            merged_reviews_path=args.merged_reviews,
            completed_disagreements_path=args.completed_disagreements,
            private_map_path=args.private_map,
            agreement_report_path=args.agreement_report,
            private_output=args.private_output,
            public_output=args.public_output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
