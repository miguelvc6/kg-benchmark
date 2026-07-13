#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from artifact_release import sha256_file
from lib.benchmark_selection import derive_case_metadata, group_key_for_record, load_selection_manifest
from lib.utils import iter_jsonl


def _rank(seed: int, group_key: str) -> str:
    return hashlib.sha256(f"{seed}|untouched_test_v1|{group_key}".encode()).hexdigest()


def _property_from_group_key(group_key: str) -> str | None:
    parts = group_key.split("::")
    if len(parts) >= 3 and parts[0] in {"ABOX", "TBOX"} and parts[2 if parts[0] == "ABOX" else 1].startswith("P"):
        return parts[2 if parts[0] == "ABOX" else 1]
    return None


def _exclusions(paths: Iterable[str | Path]) -> tuple[set[str], set[str], set[str]]:
    case_ids: set[str] = set()
    group_keys: set[str] = set()
    properties: set[str] = set()
    for path in paths:
        manifest = load_selection_manifest(path)
        case_ids.update(manifest["selected_case_ids"])
        annotations = manifest.get("case_annotations", {})
        if not isinstance(annotations, dict):
            continue
        for annotation in annotations.values():
            if not isinstance(annotation, dict):
                continue
            group_key = annotation.get("group_key")
            if isinstance(group_key, str):
                group_keys.add(group_key)
                property_id = _property_from_group_key(group_key)
                if property_id:
                    properties.add(property_id)
            property_id = annotation.get("property")
            if isinstance(property_id, str):
                properties.add(property_id)
    return case_ids, group_keys, properties


def build_untouched_test_manifest(
    *,
    classified_path: str | Path,
    exclude_manifests: Iterable[str | Path],
    target_size: int,
    seed: int = 13,
    property_holdout: bool = False,
) -> dict[str, Any]:
    if target_size < 1:
        raise ValueError("target_size must be positive.")
    excluded_ids, excluded_groups, excluded_properties = _exclusions(exclude_manifests)
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record in iter_jsonl(classified_path):
        case_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(case_id, str) or case_id in excluded_ids:
            continue
        group_key, _, weak = group_key_for_record(record)
        if weak or group_key in excluded_groups:
            continue
        property_id = record.get("property")
        if property_holdout and property_id in excluded_properties:
            continue
        metadata = derive_case_metadata(record, tier="core")
        if metadata is None:
            continue
        metadata = dict(metadata)
        metadata["property"] = property_id
        groups[group_key].append((record, metadata))

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group_key in sorted(groups, key=lambda key: (_rank(seed, key), key)):
        rows = sorted(groups[group_key], key=lambda item: item[0]["id"])
        if len(selected) + len(rows) > target_size:
            continue
        selected.extend(rows)
        if len(selected) == target_size:
            break

    annotations = {record["id"]: metadata for record, metadata in selected}
    selected_ids = [record["id"] for record, _ in selected]
    main_ids = [case_id for case_id in selected_ids if annotations[case_id].get("main_score")]
    diagnostic_ids = [case_id for case_id in selected_ids if annotations[case_id].get("diagnostic_only")]
    return {
        "manifest_type": "untouched_test_candidate",
        "manifest_version": 1,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "SEALED_DO_NOT_INSPECT_BEFORE_PROTOCOL_FREEZE",
        "scientific_warning": (
            "Code-level exclusion is not sufficient for untouchedness. Build this manifest only from a post-freeze "
            "snapshot and restrict access until models, prompts, metrics, and analysis are preregistered."
        ),
        "seed": seed,
        "inputs": {
            "classified_benchmark": str(Path(classified_path)),
            "classified_benchmark_sha256": sha256_file(classified_path),
            "exclude_manifests": [str(Path(path)) for path in exclude_manifests],
        },
        "policy": {
            "target_size": target_size,
            "assignment_unit": "complete A-box qid-property or T-box property-revision group",
            "weak_groups_excluded": True,
            "property_holdout": property_holdout,
            "stable_ordering": "sha256(seed|untouched_test_v1|group_key)",
        },
        "selected_case_ids": selected_ids,
        "main_score_case_ids": main_ids,
        "diagnostic_case_ids": diagnostic_ids,
        "case_annotations": annotations,
        "validation": {
            "target_size_met": len(selected_ids) == target_size,
            "selected_count": len(selected_ids),
            "excluded_case_overlap": len(set(selected_ids) & excluded_ids),
            "excluded_group_overlap": len(set(groups) & excluded_groups),
            "property_holdout_overlap": len(
                {annotation.get("property") for annotation in annotations.values()} & excluded_properties
            )
            if property_holdout
            else 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a sealed, group-isolated untouched-test candidate.")
    parser.add_argument("--classified-benchmark", required=True)
    parser.add_argument("--exclude-manifest", action="append", required=True)
    parser.add_argument("--target-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--property-holdout", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build_untouched_test_manifest(
        classified_path=args.classified_benchmark,
        exclude_manifests=args.exclude_manifest,
        target_size=args.target_size,
        seed=args.seed,
        property_holdout=args.property_holdout,
    )
    if not manifest["validation"]["target_size_met"]:
        raise ValueError(
            f"Untouched-test selection underfilled: {manifest['validation']['selected_count']} / {args.target_size}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
