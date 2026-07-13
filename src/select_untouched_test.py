#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from artifact_release import sha256_file
from guardian.tbox_taxonomy_patch_run import (
    UNSUPPORTED_CONFIRMATORY_REPAIR_OPS,
    taxonomy_gold_eligibility,
)
from lib.benchmark_selection import derive_case_metadata, group_key_for_record, load_selection_manifest
from lib.utils import iter_jsonl
from protocol_freeze import verify_protocol_manifest

CONFIRMATORY_TARGETS = {"TypeA": 230, "TypeB": 375, "TypeC": 295, "TBOX": 300}
RESERVE_TARGETS = {"TypeA": 276, "TypeB": 450, "TypeC": 354, "TBOX": 360}
TBOX_FINAL_TARGETS = {
    "RELAXATION_SET_EXPANSION": 130,
    "RESTRICTION_SET_CONTRACTION": 50,
    "SCHEMA_UPDATE": 120,
}
TBOX_RESERVE_TARGETS = {
    "RELAXATION_SET_EXPANSION": 156,
    "RESTRICTION_SET_CONTRACTION": 60,
    "SCHEMA_UPDATE": 144,
}


def _rank(seed: int, group_key: str) -> str:
    return hashlib.sha256(f"{seed}|untouched_test_v2|{group_key}".encode()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _broad_stratum(record: dict[str, Any]) -> str | None:
    classification = record.get("classification")
    classification = classification if isinstance(classification, dict) else {}
    value = classification.get("class")
    if value in {"TypeA", "TypeB", "TypeC"}:
        return str(value)
    if value == "T_BOX" or record.get("track") == "T_BOX":
        return "TBOX"
    return None


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_complete_dispositions(
    path: str | Path, *, expected_ids: set[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    source = Path(path)
    dispositions: dict[str, str] = {}
    allowed = {"include", "diagnostic", "exclude", "exclude_pending_rerender", "unsupported", "disagreement", "malformed"}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("case_id") if isinstance(row, dict) else None
            disposition = row.get("disposition") if isinstance(row, dict) else None
            if not isinstance(case_id, str) or not case_id or disposition not in allowed:
                raise ValueError(f"Invalid final disposition at {source}:{line_number}.")
            if case_id in dispositions:
                raise ValueError(f"Duplicate final disposition for {case_id}.")
            dispositions[case_id] = disposition
    if set(dispositions) != expected_ids:
        raise ValueError(
            "Confirmatory selection requires complete unique disposition coverage: "
            f"missing={len(expected_ids - set(dispositions))}, unexpected={len(set(dispositions) - expected_ids)}"
        )
    return dispositions, {"path": str(source), "size_bytes": source.stat().st_size, "sha256": sha256_file(source)}


def _audit_gate(path: str | Path, *, report_type: str, minimum_version: int = 2) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path)
    payload = _read_json(source)
    if not isinstance(payload, dict) or payload.get("report_type") != report_type:
        raise ValueError(f"Expected {report_type} at {source}.")
    if int(payload.get("report_version", 0)) < minimum_version:
        raise ValueError(f"Confirmatory selection requires {report_type} v{minimum_version} or later.")
    return payload, {"path": str(source), "size_bytes": source.stat().st_size, "sha256": sha256_file(source)}


def _largest_remainder(total: int, weights: dict[str, int]) -> dict[str, int]:
    denominator = sum(weights.values())
    exact = {key: total * value / denominator for key, value in weights.items()}
    result = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(weights, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def _excluded_groups(classified_path: str | Path, exclude_manifests: Iterable[str | Path]) -> tuple[set[str], list[dict[str, Any]]]:
    excluded_ids, fingerprints = _load_excluded_ids(exclude_manifests)
    groups: set[str] = set()
    for record in iter_jsonl(classified_path):
        if record.get("id") in excluded_ids:
            groups.add(group_key_for_record(record)[0])
    return groups, fingerprints


def _ranked_eligible_groups(
    *,
    records: Iterable[dict[str, Any]],
    dispositions: dict[str, str],
    excluded_groups: set[str],
    seed: int,
    audit_hashes: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    by_group: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = record["id"]
        group_key, _, weak = group_key_for_record(record)
        metadata = derive_case_metadata(record, tier="core")
        broad = _broad_stratum(record)
        if (
            dispositions.get(case_id) != "include"
            or group_key in excluded_groups
            or weak
            or broad is None
            or metadata is None
            or not metadata.get("main_score")
        ):
            continue
        taxonomy_gold = None
        if broad == "TBOX":
            taxonomy_gold, _taxonomy_exclusion_reason = taxonomy_gold_eligibility(
                record,
                annotation=metadata,
            )
            if taxonomy_gold is None:
                continue
        annotation = {
            **metadata,
            "case_id": case_id,
            "group_key": group_key,
            "broad_stratum": broad,
            "tbox_subtype": record.get("classification", {}).get("subtype") if broad == "TBOX" else None,
        }
        if taxonomy_gold is not None:
            annotation["tbox_taxonomy_gold"] = {
                "schema_decision": taxonomy_gold["schema_decision"],
                "constraint_type_qid": taxonomy_gold["target"]["constraint_type_qid"],
                "repair_ops": [repair["repair_op"] for repair in taxonomy_gold["repairs"]],
                "taxonomy_codes": [repair["taxonomy_code"] for repair in taxonomy_gold["repairs"]],
                "evidence_levels": [repair["evidence_level"] for repair in taxonomy_gold["repairs"]],
            }
        eligibility_payload = {
            "case_id": case_id,
            "group_key": group_key,
            "disposition": "include",
            **audit_hashes,
        }
        if taxonomy_gold is not None:
            eligibility_payload["tbox_taxonomy_gold"] = taxonomy_gold
        annotation["eligibility_sha256"] = _canonical_hash(eligibility_payload)
        current = by_group.get(group_key)
        if current is None or (_rank(seed, case_id), case_id) < (
            _rank(seed, current["case_id"]), current["case_id"]
        ):
            by_group[group_key] = annotation
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group_key, winner in by_group.items():
        winner["ranking_sha256"] = _rank(seed, group_key)
        selected[winner["broad_stratum"]].append(winner)
    for values in selected.values():
        values.sort(key=lambda row: (row["ranking_sha256"], row["group_key"], row["case_id"]))
    return selected


def _choose_tbox(values: list[dict[str, Any]], targets: dict[str, int], cap: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for subtype, target in targets.items():
        matching = [row for row in values if row.get("tbox_subtype") == subtype][:target]
        chosen.extend(matching)
        used.update(row["case_id"] for row in matching)
    if len(chosen) < cap:
        remaining = [row for row in values if row["case_id"] not in used]
        chosen.extend(remaining[: cap - len(chosen)])
    return sorted(chosen, key=lambda row: (row["ranking_sha256"], row["case_id"]))


def build_reserve_manifest(
    *,
    classified_path: str | Path,
    dispositions_path: str | Path,
    dataset_audit_path: str | Path,
    temporal_audit_path: str | Path,
    exclude_manifests: Iterable[str | Path],
    snapshot_manifest_path: str | Path,
    seed: int = 13,
) -> dict[str, Any]:
    if seed != 13:
        raise ValueError("Confirmatory reserve ranking seed is frozen at 13.")
    record_ids: set[str] = set()
    record_count = 0
    for row in iter_jsonl(classified_path):
        record_count += 1
        if isinstance(row.get("id"), str):
            record_ids.add(row["id"])
    if len(record_ids) != record_count:
        raise ValueError("Stage 4 must contain complete unique case IDs.")
    dispositions, disposition_input = _load_complete_dispositions(dispositions_path, expected_ids=record_ids)
    dataset_audit, dataset_input = _audit_gate(dataset_audit_path, report_type="automated_consistency_audit")
    temporal_audit, temporal_input = _audit_gate(temporal_audit_path, report_type="temporal_prompt_leakage_audit")
    if not dataset_audit.get("lineage_validation", {}).get("passed"):
        raise ValueError("Reserve allocation requires a passing complete lineage validation.")
    if not temporal_audit.get("passed_automated_gate"):
        raise ValueError("Reserve allocation requires a passing whole-reserve temporal audit.")
    excluded_groups, exclusion_inputs = _excluded_groups(classified_path, exclude_manifests)
    snapshot_path = Path(snapshot_manifest_path)
    snapshot = _read_json(snapshot_path)
    if not isinstance(snapshot, dict) or int(snapshot.get("manifest_version", 0)) < 2:
        raise ValueError("Confirmatory reserve requires a v2 post-freeze snapshot manifest.")
    audit_hashes = {"dataset_audit": dataset_input["sha256"], "temporal_audit": temporal_input["sha256"]}
    eligible = _ranked_eligible_groups(
        records=iter_jsonl(classified_path),
        dispositions=dispositions,
        excluded_groups=excluded_groups,
        seed=seed,
        audit_hashes=audit_hashes,
    )
    selected: list[dict[str, Any]] = []
    for stratum in ("TypeA", "TypeB", "TypeC"):
        selected.extend(eligible[stratum][: RESERVE_TARGETS[stratum]])
    selected.extend(_choose_tbox(eligible["TBOX"], TBOX_RESERVE_TARGETS, RESERVE_TARGETS["TBOX"]))
    selected.sort(key=lambda row: (row["ranking_sha256"], row["case_id"]))
    composition = Counter(row["broad_stratum"] for row in selected)
    manifest = {
        "manifest_type": "confirmatory_reserve_private",
        "manifest_version": 3,
        "seed": seed,
        "snapshot_id": snapshot.get("snapshot_id"),
        "inputs": {
            "classified_benchmark": {"path": str(classified_path), "sha256": sha256_file(classified_path)},
            "dispositions": disposition_input,
            "dataset_audit": dataset_input,
            "temporal_audit": temporal_input,
            "snapshot_manifest": {"path": str(snapshot_path), "sha256": sha256_file(snapshot_path)},
            "exclusions": exclusion_inputs,
        },
        "policy": {
            "reserve_targets": RESERVE_TARGETS,
            "tbox_reserve_targets": TBOX_RESERVE_TARGETS,
            "selection_eligible_disposition": "include",
            "one_case_per_event_group": True,
            "property_holdout": False,
            "ranking": "sha256(seed|untouched_test_v2|group_key)",
            "tbox_task_version": "tbox_taxonomy_patch_v1",
            "tbox_gold_requirement": "complete_mechanically_supported_gold",
            "unsupported_tbox_repair_ops": sorted(UNSUPPORTED_CONFIRMATORY_REPAIR_OPS),
        },
        "selected_case_ids": [row["case_id"] for row in selected],
        "case_annotations": {row["case_id"]: {key: value for key, value in row.items() if key != "case_id"} for row in selected},
        "counts": {"selected": len(selected), "by_stratum": dict(sorted(composition.items()))},
        "eligible_group_counts": {key: len(value) for key, value in sorted(eligible.items())},
    }
    manifest["reserve_commitment"] = _canonical_hash(
        {"seed": seed, "snapshot_id": manifest["snapshot_id"], "selected_case_ids": manifest["selected_case_ids"]}
    )
    return manifest


def _prompt_failure_ids(temporal: dict[str, Any], prompt_review: dict[str, Any] | None) -> set[str]:
    failures = {
        str(hit["case_id"])
        for hit in temporal.get("hits", [])
        if isinstance(hit, dict) and hit.get("severity") == "high" and hit.get("case_id")
    }
    if prompt_review:
        failures.update(str(value) for value in prompt_review.get("failed_case_ids", []) if value)
    return failures


def _api_subset(rows: list[dict[str, Any]], seed: int) -> list[str]:
    counts = Counter(row["broad_stratum"] for row in rows)
    targets = _largest_remainder(600, dict(counts))
    chosen: list[str] = []
    for stratum, target in targets.items():
        values = sorted(
            (row for row in rows if row["broad_stratum"] == stratum),
            key=lambda row: (_rank(seed, "api|" + row["group_key"]), row["case_id"]),
        )
        chosen.extend(row["case_id"] for row in values[:target])
    return sorted(chosen, key=lambda case_id: (_rank(seed, "api-case|" + case_id), case_id))


def finalize_reserve_manifest(
    *,
    reserve_manifest_path: str | Path,
    temporal_audit_path: str | Path,
    prompt_review_path: str | Path | None = None,
    seed: int = 13,
) -> dict[str, Any]:
    if seed != 13:
        raise ValueError("Confirmatory finalization seed is frozen at 13.")
    reserve_path = Path(reserve_manifest_path)
    reserve = _read_json(reserve_path)
    if not isinstance(reserve, dict) or reserve.get("manifest_type") != "confirmatory_reserve_private":
        raise ValueError("Finalize requires a confirmatory reserve manifest.")
    if reserve.get("seed") != seed:
        raise ValueError("Finalize seed must match the reserve seed.")
    temporal_path = Path(temporal_audit_path)
    temporal = _read_json(temporal_path)
    if not isinstance(temporal, dict) or temporal.get("report_type") != "temporal_prompt_leakage_audit":
        raise ValueError("Finalize requires a temporal prompt audit.")
    review = _read_json(prompt_review_path) if prompt_review_path else None
    failures = _prompt_failure_ids(temporal, review if isinstance(review, dict) else None)
    annotations = reserve.get("case_annotations", {})
    rows = [
        {"case_id": case_id, **annotations[case_id]}
        for case_id in reserve.get("selected_case_ids", [])
        if case_id in annotations and case_id not in failures
    ]
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[row["broad_stratum"]].append(row)
    tbox = _choose_tbox(by_stratum["TBOX"], TBOX_FINAL_TARGETS, CONFIRMATORY_TARGETS["TBOX"])
    tbox_count = min(len(tbox), CONFIRMATORY_TARGETS["TBOX"])
    deficit = CONFIRMATORY_TARGETS["TBOX"] - tbox_count
    transfer = _largest_remainder(deficit, {key: CONFIRMATORY_TARGETS[key] for key in ("TypeA", "TypeB", "TypeC")})
    targets = {
        key: CONFIRMATORY_TARGETS[key] + transfer[key] for key in ("TypeA", "TypeB", "TypeC")
    }
    targets["TBOX"] = tbox_count
    selected = tbox[:tbox_count]
    for stratum in ("TypeA", "TypeB", "TypeC"):
        selected.extend(by_stratum[stratum][: targets[stratum]])
    if len(selected) != 1200 or any(Counter(row["broad_stratum"] for row in selected)[key] != value for key, value in targets.items()):
        raise ValueError(
            "Fewer than 1,200 independent include-disposition reserve cases survive prompt gates; quality rules are not relaxed."
        )
    group_keys = [row["group_key"] for row in selected]
    if len(group_keys) != len(set(group_keys)):
        raise ValueError("Final selection repeats an event group.")
    selected.sort(key=lambda row: (row["ranking_sha256"], row["case_id"]))
    api_ids = _api_subset(selected, seed)
    if len(api_ids) != 600 or not set(api_ids).issubset({row["case_id"] for row in selected}):
        raise ValueError("Nested API subset construction failed.")
    manifest = {
        "manifest_type": "confirmatory_final_private",
        "manifest_version": 3,
        "seed": seed,
        "snapshot_id": reserve.get("snapshot_id"),
        "inputs": {
            "reserve_manifest": {"path": str(reserve_path), "sha256": sha256_file(reserve_path)},
            "temporal_audit": {"path": str(temporal_path), "sha256": sha256_file(temporal_path)},
            "prompt_review": ({"path": str(prompt_review_path), "sha256": sha256_file(prompt_review_path)} if prompt_review_path else None),
        },
        "policy": {
            "target_size": 1200,
            "api_subset_size": 600,
            "targets": targets,
            "tbox_preferred_targets": TBOX_FINAL_TARGETS,
            "tbox_deficit_transfer": transfer,
            "selection_eligible_disposition": "include",
            "one_case_per_event_group": True,
            "property_holdout": False,
            "tbox_task_version": "tbox_taxonomy_patch_v1",
            "tbox_gold_requirement": "complete_mechanically_supported_gold",
            "unsupported_tbox_repair_ops": sorted(UNSUPPORTED_CONFIRMATORY_REPAIR_OPS),
        },
        "selected_case_ids": [row["case_id"] for row in selected],
        "api_subset_case_ids": api_ids,
        "case_annotations": {row["case_id"]: {key: value for key, value in row.items() if key != "case_id"} for row in selected},
        "counts": {
            "selected": 1200,
            "api_subset": 600,
            "by_stratum": dict(sorted(Counter(row["broad_stratum"] for row in selected).items())),
            "prompt_failures_removed": len(failures),
        },
    }
    manifest["selection_commitment"] = _canonical_hash(
        {"seed": seed, "snapshot_id": manifest["snapshot_id"], "selected_case_ids": manifest["selected_case_ids"], "api_subset_case_ids": api_ids}
    )
    return manifest


def write_confirmatory_manifests(manifest: dict[str, Any], *, private_output: str | Path, public_output: str | Path) -> dict[str, Any]:
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "confirmatory_selection.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(manifest)
    private_path = Path(private_output)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(private_path, 0o600)
    public = {
        "manifest_type": manifest["manifest_type"].replace("_private", "_public_commitment"),
        "manifest_version": manifest["manifest_version"],
        "snapshot_id": manifest["snapshot_id"],
        "seed": manifest["seed"],
        "policy": manifest["policy"],
        "counts": manifest["counts"],
        "private_manifest_sha256": sha256_file(private_path),
        "commitment": manifest.get("selection_commitment", manifest.get("reserve_commitment")),
    }
    public_path = Path(public_output)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(json.dumps(public, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return public


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
    if protocol.get("protocol_phase") != "allocation" or protocol.get("release", {}).get("release_kind") != "dataset":
        raise ValueError("Untouched allocation requires an allocation protocol bound to a dataset release.")
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
    expected_main = protocol.get("expected_population", {}).get("main_score_count")
    if expected_main != len(main_ids):
        raise ValueError(
            f"Allocated main-score count {len(main_ids)} does not match protocol expectation {expected_main}."
        )
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


def _legacy_main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)
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


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"reserve", "finalize"}:
        return _legacy_main()
    parser = argparse.ArgumentParser(description="Build and finalize a quality-gated confirmatory selection.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    reserve = subparsers.add_parser("reserve")
    reserve.add_argument("--classified-benchmark", required=True)
    reserve.add_argument("--dispositions", required=True)
    reserve.add_argument("--dataset-audit", required=True)
    reserve.add_argument("--temporal-audit", required=True)
    reserve.add_argument("--exclude-manifest", action="append", default=[])
    reserve.add_argument("--snapshot-manifest", required=True)
    reserve.add_argument("--seed", type=int, default=13)
    reserve.add_argument("--private-output", required=True)
    reserve.add_argument("--public-output", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--reserve-manifest", required=True)
    finalize.add_argument("--temporal-audit", required=True)
    finalize.add_argument("--prompt-review")
    finalize.add_argument("--seed", type=int, default=13)
    finalize.add_argument("--private-output", required=True)
    finalize.add_argument("--public-output", required=True)
    args = parser.parse_args()
    if args.command == "reserve":
        manifest = build_reserve_manifest(
            classified_path=args.classified_benchmark,
            dispositions_path=args.dispositions,
            dataset_audit_path=args.dataset_audit,
            temporal_audit_path=args.temporal_audit,
            exclude_manifests=args.exclude_manifest,
            snapshot_manifest_path=args.snapshot_manifest,
            seed=args.seed,
        )
    else:
        manifest = finalize_reserve_manifest(
            reserve_manifest_path=args.reserve_manifest,
            temporal_audit_path=args.temporal_audit,
            prompt_review_path=args.prompt_review,
            seed=args.seed,
        )
    write_confirmatory_manifests(
        manifest, private_output=args.private_output, public_output=args.public_output
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
