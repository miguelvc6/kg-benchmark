#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from artifact_release import sha256_file
from lib.benchmark_selection import derive_case_metadata, group_key_for_record, load_selection_manifest
from lib.utils import iter_jsonl
from protocol_freeze import verify_protocol_manifest


def _rank(seed: int, group_key: str) -> str:
    return hashlib.sha256(f"{seed}|untouched_test_v2|{group_key}".encode()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_excluded_ids(paths: Iterable[str | Path]) -> tuple[set[str], list[dict[str, Any]]]:
    case_ids: set[str] = set()
    fingerprints: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        manifest = load_selection_manifest(path)
        case_ids.update(manifest["selected_case_ids"])
        fingerprints.append(
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return case_ids, fingerprints


def _aggregate_composition(annotations: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    fields = ("track", "class", "subtype", "selection_stratum", "popularity_bucket", "confidence")
    return {
        field: dict(sorted(Counter(str(value.get(field, "unknown")) for value in annotations.values()).items()))
        for field in fields
    }


def _select_groups(
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    *,
    stratum_targets: dict[str, int],
    seed: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    counts: Counter[str] = Counter()
    for group_key in sorted(groups, key=lambda key: (_rank(seed, key), key)):
        rows = sorted(groups[group_key], key=lambda item: item[0]["id"])
        profile = Counter(str(metadata["selection_stratum"]) for _, metadata in rows)
        if any(counts[stratum] + count > stratum_targets.get(stratum, 0) for stratum, count in profile.items()):
            continue
        selected.extend(rows)
        counts.update(profile)
        if all(counts[stratum] == target for stratum, target in stratum_targets.items()):
            break
    if dict(counts) != {key: value for key, value in stratum_targets.items() if value}:
        raise ValueError(
            "Untouched-test stratum targets could not be filled with complete groups: "
            f"selected={dict(counts)} targets={stratum_targets}"
        )
    return selected


def validate_untouched_test_manifest(
    manifest: dict[str, Any],
    *,
    classified_path: str | Path,
    exclude_manifests: Iterable[str | Path],
) -> dict[str, Any]:
    """Independently reconstruct allocation groups and exclusions from Stage 4."""
    selected_ids = set(manifest.get("selected_case_ids", []))
    excluded_ids, _ = _load_excluded_ids(exclude_manifests)
    selected_records: dict[str, dict[str, Any]] = {}
    excluded_groups: set[str] = set()
    excluded_properties: set[str] = set()
    eligible_group_members: dict[str, set[str]] = defaultdict(set)
    selected_groups: set[str] = set()
    selected_properties: set[str] = set()
    property_holdout = bool(manifest.get("policy", {}).get("property_holdout"))

    records: list[dict[str, Any]] = []
    for record in iter_jsonl(classified_path):
        case_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(case_id, str):
            continue
        records.append(record)
        if case_id in excluded_ids:
            group_key, _, _ = group_key_for_record(record)
            excluded_groups.add(group_key)
            if isinstance(record.get("property"), str):
                excluded_properties.add(record["property"])

    for record in records:
        case_id = record["id"]
        group_key, _, weak = group_key_for_record(record)
        property_id = record.get("property")
        metadata = derive_case_metadata(record, tier="core")
        if (
            case_id in excluded_ids
            or group_key in excluded_groups
            or weak
            or metadata is None
            or (property_holdout and property_id in excluded_properties)
        ):
            continue
        eligible_group_members[group_key].add(case_id)
        if case_id in selected_ids:
            selected_records[case_id] = record
            selected_groups.add(group_key)
            if isinstance(property_id, str):
                selected_properties.add(property_id)

    selected_group_members = {
        case_id
        for group_key in selected_groups
        for case_id in eligible_group_members.get(group_key, set())
    }
    annotations = manifest.get("case_annotations", {})
    recomputed_annotations = {
        case_id: {**(derive_case_metadata(record, tier="core") or {}), "property": record.get("property")}
        for case_id, record in selected_records.items()
    }
    expected_targets = manifest.get("policy", {}).get("stratum_targets", {})
    actual_targets = Counter(
        str(annotation.get("selection_stratum")) for annotation in recomputed_annotations.values()
    )
    checks = {
        "source_hash_matches": sha256_file(classified_path)
        == manifest.get("inputs", {}).get("classified_benchmark_sha256"),
        "selected_ids_present": set(selected_records) == selected_ids,
        "excluded_case_isolation": not (selected_ids & excluded_ids),
        "excluded_group_isolation": not (selected_groups & excluded_groups),
        "property_holdout_isolation": not property_holdout or not (selected_properties & excluded_properties),
        "complete_groups_selected": selected_group_members == selected_ids,
        "annotations_recomputed": annotations == recomputed_annotations,
        "stratum_targets_met": dict(actual_targets) == expected_targets,
        "selected_ids_unique": len(manifest.get("selected_case_ids", [])) == len(selected_ids),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "selected": len(selected_ids),
            "selected_groups": len(selected_groups),
            "excluded_cases": len(excluded_ids),
            "excluded_groups": len(excluded_groups),
            "excluded_properties": len(excluded_properties),
        },
        "overlap_counts": {
            "cases": len(selected_ids & excluded_ids),
            "groups": len(selected_groups & excluded_groups),
            "properties": len(selected_properties & excluded_properties) if property_holdout else 0,
        },
        "missing_selected_ids": sorted(selected_ids - set(selected_records))[:100],
        "incomplete_group_case_ids": sorted(selected_group_members ^ selected_ids)[:100],
    }


def build_untouched_test_manifest(
    *,
    classified_path: str | Path,
    exclude_manifests: Iterable[str | Path],
    stratum_targets: dict[str, int],
    protocol_manifest_path: str | Path,
    protocol_root: str | Path,
    seed: int = 13,
    property_holdout: bool = False,
) -> dict[str, Any]:
    normalized_targets = {
        str(key): int(value) for key, value in stratum_targets.items() if isinstance(value, int) and value > 0
    }
    if not normalized_targets or normalized_targets != stratum_targets:
        raise ValueError("Every stratum target must be a positive integer.")
    protocol_verification = verify_protocol_manifest(protocol_manifest_path, protocol_root=protocol_root)
    if not protocol_verification["passed"]:
        raise ValueError("Untouched allocation requires a verified frozen protocol.")
    protocol = protocol_verification["manifest"]
    target_size = sum(normalized_targets.values())
    if protocol.get("expected_population", {}).get("selected_count") != target_size:
        raise ValueError("Stratum targets must sum to the protocol's expected selected count.")

    exclusion_paths = [Path(path) for path in exclude_manifests]
    excluded_ids, exclusion_fingerprints = _load_excluded_ids(exclusion_paths)
    records: list[dict[str, Any]] = []
    excluded_groups: set[str] = set()
    excluded_properties: set[str] = set()
    for record in iter_jsonl(classified_path):
        case_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(case_id, str):
            continue
        records.append(record)
        if case_id in excluded_ids:
            group_key, _, _ = group_key_for_record(record)
            excluded_groups.add(group_key)
            if isinstance(record.get("property"), str):
                excluded_properties.add(record["property"])

    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record in records:
        case_id = record["id"]
        group_key, _, weak = group_key_for_record(record)
        property_id = record.get("property")
        if case_id in excluded_ids or group_key in excluded_groups or weak:
            continue
        if property_holdout and property_id in excluded_properties:
            continue
        metadata = derive_case_metadata(record, tier="core")
        if metadata is None or metadata["selection_stratum"] not in normalized_targets:
            continue
        groups[group_key].append((record, {**metadata, "property": property_id}))

    selected = _select_groups(groups, stratum_targets=normalized_targets, seed=seed)
    annotations = {record["id"]: metadata for record, metadata in selected}
    selected_ids = [record["id"] for record, _ in selected]
    main_ids = [case_id for case_id in selected_ids if annotations[case_id].get("main_score")]
    diagnostic_ids = [case_id for case_id in selected_ids if annotations[case_id].get("diagnostic_only")]
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest = {
        "manifest_type": "untouched_test_private_allocation",
        "manifest_version": 2,
        "created_at_utc": created_at,
        "status": "PRIVATE_ALLOCATION_DO_NOT_DISTRIBUTE",
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "manifest_sha256": sha256_file(protocol_manifest_path),
        },
        "seed": seed,
        "inputs": {
            "classified_benchmark": str(Path(classified_path)),
            "classified_benchmark_sha256": sha256_file(classified_path),
            "exclude_manifests": exclusion_fingerprints,
        },
        "policy": {
            "target_size": target_size,
            "stratum_targets": normalized_targets,
            "assignment_unit": "complete A-box qid-property or T-box property-revision group",
            "weak_groups_excluded": True,
            "property_holdout": property_holdout,
            "stable_ordering": "sha256(seed|untouched_test_v2|group_key)",
        },
        "selected_case_ids": selected_ids,
        "main_score_case_ids": main_ids,
        "diagnostic_case_ids": diagnostic_ids,
        "case_annotations": annotations,
        "composition": _aggregate_composition(annotations),
    }
    manifest["validation"] = validate_untouched_test_manifest(
        manifest,
        classified_path=classified_path,
        exclude_manifests=exclusion_paths,
    )
    if not manifest["validation"]["passed"]:
        raise ValueError(f"Independent untouched-test validation failed: {manifest['validation']['checks']}")
    return manifest


def write_allocation_manifests(
    manifest: dict[str, Any], *, private_output: str | Path, public_output: str | Path
) -> dict[str, Any]:
    private_path = Path(private_output)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.chmod(private_path, 0o600)
    allocation_commitment = _canonical_hash(
        {
            "seed": manifest["seed"],
            "source": manifest["inputs"]["classified_benchmark_sha256"],
            "selected_case_ids": manifest["selected_case_ids"],
        }
    )
    public = {
        "manifest_type": "untouched_test_public_commitment",
        "manifest_version": 2,
        "created_at_utc": manifest["created_at_utc"],
        "status": "ALLOCATED_PRIVATE_IDS_WITHHELD",
        "protocol_id": manifest["protocol"]["protocol_id"],
        "private_manifest_sha256": sha256_file(private_path),
        "allocation_commitment": allocation_commitment,
        "policy": {
            "target_size": manifest["policy"]["target_size"],
            "stratum_targets": manifest["policy"]["stratum_targets"],
            "assignment_unit": manifest["policy"]["assignment_unit"],
            "property_holdout": manifest["policy"]["property_holdout"],
        },
        "counts": {
            "selected": len(manifest["selected_case_ids"]),
            "main_score": len(manifest["main_score_case_ids"]),
            "diagnostic": len(manifest["diagnostic_case_ids"]),
        },
        "composition": manifest["composition"],
        "validation": manifest["validation"],
    }
    public_path = Path(public_output)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(json.dumps(public, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return public


def main() -> int:
    parser = argparse.ArgumentParser(description="Allocate a protocol-bound private untouched test.")
    parser.add_argument("--classified-benchmark", required=True)
    parser.add_argument("--exclude-manifest", action="append", required=True)
    parser.add_argument("--stratum-targets", required=True, help="JSON file mapping selection strata to counts.")
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--protocol-root", default=".")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--property-holdout", action="store_true")
    parser.add_argument("--private-output", required=True)
    parser.add_argument("--public-output", required=True)
    args = parser.parse_args()
    targets = json.loads(Path(args.stratum_targets).read_text(encoding="utf-8"))
    manifest = build_untouched_test_manifest(
        classified_path=args.classified_benchmark,
        exclude_manifests=args.exclude_manifest,
        stratum_targets=targets,
        protocol_manifest_path=args.protocol_manifest,
        protocol_root=args.protocol_root,
        seed=args.seed,
        property_holdout=args.property_holdout,
    )
    write_allocation_manifests(
        manifest,
        private_output=args.private_output,
        public_output=args.public_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
