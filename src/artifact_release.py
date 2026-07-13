#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from lib.benchmark_selection import group_key_for_record, load_selection_manifest
from lib.utils import iter_jsonl


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty}


def validate_release_inputs(
    *,
    stage4_path: str | Path,
    schema_path: str | Path,
    selection_manifest_path: str | Path,
) -> dict[str, Any]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    selection = load_selection_manifest(selection_manifest_path)
    selected_ids = set(selection["selected_case_ids"])
    subset_partition_present = isinstance(selection.get("main_score_case_ids"), list) and isinstance(
        selection.get("diagnostic_case_ids"), list
    )
    policy = selection.get("policy") if isinstance(selection.get("policy"), dict) else {}
    tbox_cap = policy.get("tbox_cap_per_property_revision")
    abox_cap = policy.get("abox_cap_per_qid_property")

    observed_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    schema_error_count = 0
    schema_error_examples: list[dict[str, Any]] = []
    tbox_counts: Counter[str] = Counter()
    abox_counts: Counter[str] = Counter()
    selected_group_keys: set[str] = set()
    selected_tbox_keys: set[str] = set()
    for record in iter_jsonl(stage4_path):
        case_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(case_id, str) or case_id not in selected_ids:
            continue
        if case_id in observed_ids:
            duplicate_ids.add(case_id)
        observed_ids.add(case_id)
        for error in validator.iter_errors(record):
            schema_error_count += 1
            if len(schema_error_examples) < 20:
                schema_error_examples.append(
                    {
                        "case_id": case_id,
                        "path": list(error.path),
                        "message": error.message,
                    }
                )
        group_key, tbox_key, _ = group_key_for_record(record)
        selected_group_keys.add(group_key)
        if isinstance(tbox_key, str):
            selected_tbox_keys.add(tbox_key)
            tbox_counts[tbox_key] += 1
        else:
            abox_counts[group_key] += 1

    missing_ids = sorted(selected_ids - observed_ids)
    exclude_case_overlap: set[str] = set()
    exclude_group_overlap: set[str] = set()
    exclude_tbox_overlap: set[str] = set()
    exclude_path_value = selection.get("inputs", {}).get("exclude_manifest")
    if isinstance(exclude_path_value, str) and exclude_path_value:
        exclude_path = Path(exclude_path_value)
        if not exclude_path.is_absolute():
            exclude_path = Path.cwd() / exclude_path
        if not exclude_path.is_file():
            raise ValueError(f"Selection exclusion manifest does not exist: {exclude_path}")
        excluded = load_selection_manifest(exclude_path)
        exclude_case_overlap = selected_ids & set(excluded["selected_case_ids"])
        excluded_annotations = excluded.get("case_annotations", {})
        if isinstance(excluded_annotations, dict):
            excluded_groups = {
                annotation.get("group_key")
                for annotation in excluded_annotations.values()
                if isinstance(annotation, dict) and isinstance(annotation.get("group_key"), str)
            }
            excluded_tbox = {
                annotation.get("tbox_revision_key")
                for annotation in excluded_annotations.values()
                if isinstance(annotation, dict) and isinstance(annotation.get("tbox_revision_key"), str)
            }
            exclude_group_overlap = selected_group_keys & excluded_groups
            exclude_tbox_overlap = selected_tbox_keys & excluded_tbox

    checks = {
        "subset_partition_present": subset_partition_present,
        "selected_ids_present": not missing_ids,
        "selected_ids_unique_in_stage4": not duplicate_ids,
        "selected_records_schema_valid": schema_error_count == 0,
        "tbox_cap_valid": isinstance(tbox_cap, int) and max(tbox_counts.values(), default=0) <= tbox_cap,
        "abox_cap_valid": isinstance(abox_cap, int) and max(abox_counts.values(), default=0) <= abox_cap,
        "exclude_case_isolation": not exclude_case_overlap,
        "exclude_group_isolation": not exclude_group_overlap,
        "exclude_tbox_isolation": not exclude_tbox_overlap,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "selected": len(selected_ids),
            "observed": len(observed_ids),
            "schema_errors": schema_error_count,
            "max_tbox_per_revision": max(tbox_counts.values(), default=0),
            "max_abox_per_qid_property": max(abox_counts.values(), default=0),
            "exclude_case_overlap": len(exclude_case_overlap),
            "exclude_group_overlap": len(exclude_group_overlap),
            "exclude_tbox_overlap": len(exclude_tbox_overlap),
        },
        "missing_case_ids": missing_ids[:100],
        "duplicate_case_ids": sorted(duplicate_ids)[:100],
        "schema_error_examples": schema_error_examples,
    }


def build_release_manifest(
    *,
    stage4_path: str | Path,
    schema_path: str | Path,
    selection_manifest_path: str | Path,
    artifacts: Iterable[str | Path] = (),
) -> dict[str, Any]:
    validation = validate_release_inputs(
        stage4_path=stage4_path,
        schema_path=schema_path,
        selection_manifest_path=selection_manifest_path,
    )
    if not validation["passed"]:
        raise ValueError(f"Release validation failed: {json.dumps(validation['checks'], sort_keys=True)}")
    paths = [Path(stage4_path), Path(schema_path), Path(selection_manifest_path), *(Path(p) for p in artifacts)]
    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    files = []
    for path in unique_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "manifest_type": "kg_benchmark_release_candidate",
        "manifest_version": 1,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "candidate",
        "code": _git_state(),
        "validation": validation,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and fingerprint a benchmark release candidate.")
    parser.add_argument("--stage4", default="data/04_classified_benchmark.jsonl")
    parser.add_argument("--schema", default="schemas/04_classified_benchmark.schema.json")
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build_release_manifest(
        stage4_path=args.stage4,
        schema_path=args.schema,
        selection_manifest_path=args.selection_manifest,
        artifacts=args.artifact,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
