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

import ijson
from jsonschema import Draft202012Validator

from lib.benchmark_selection import group_key_for_record, load_selection_manifest
from lib.utils import iter_jsonl, iter_repairs
from lib.world_state import validate_world_state_entry

RELEASE_KINDS = {"dataset", "evaluation"}
RELEASE_STATUSES = {"candidate", "confirmatory"}


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


def _portable_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Release file must be inside release root {root}: {resolved}") from exc


def _selection_exclusions(
    selection: dict[str, Any], selection_path: Path
) -> tuple[set[str], set[str], set[str]]:
    exclude_value = selection.get("inputs", {}).get("exclude_manifest")
    if not isinstance(exclude_value, str) or not exclude_value:
        return set(), set(), set()
    exclude_path = Path(exclude_value)
    if not exclude_path.is_absolute():
        exclude_path = selection_path.parent / exclude_path
        if not exclude_path.is_file():
            exclude_path = Path.cwd() / Path(exclude_value)
    if not exclude_path.is_file():
        raise ValueError(f"Selection exclusion manifest does not exist: {exclude_path}")
    excluded = load_selection_manifest(exclude_path)
    annotations = excluded.get("case_annotations", {})
    if not isinstance(annotations, dict):
        annotations = {}
    group_keys = {
        value.get("group_key")
        for value in annotations.values()
        if isinstance(value, dict) and isinstance(value.get("group_key"), str)
    }
    tbox_keys = {
        value.get("tbox_revision_key")
        for value in annotations.values()
        if isinstance(value, dict) and isinstance(value.get("tbox_revision_key"), str)
    }
    return set(excluded["selected_case_ids"]), group_keys, tbox_keys


def validate_release_inputs(
    *,
    stage2_path: str | Path,
    world_state_path: str | Path,
    stage4_path: str | Path,
    schema_path: str | Path,
    release_kind: str,
    selection_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate every released record and all Stage 2/3/4 identity links."""
    if release_kind not in RELEASE_KINDS:
        raise ValueError(f"Unsupported release kind: {release_kind}")
    if release_kind == "evaluation" and selection_manifest_path is None:
        raise ValueError("Evaluation releases require a selection manifest.")

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    selection_path = Path(selection_manifest_path) if selection_manifest_path is not None else None
    selection = load_selection_manifest(selection_path) if selection_path is not None else None
    selected_ids = set(selection["selected_case_ids"]) if selection else set()
    subset_partition_present = selection is None or (
        isinstance(selection.get("main_score_case_ids"), list)
        and isinstance(selection.get("diagnostic_case_ids"), list)
    )
    policy = selection.get("policy", {}) if selection else {}
    tbox_cap = policy.get("tbox_cap_per_property_revision")
    abox_cap = policy.get("abox_cap_per_qid_property")

    stage2_ids: set[str] = set()
    stage2_duplicates: set[str] = set()
    stage2_missing_id_count = 0
    for record in iter_repairs(stage2_path):
        case_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(case_id, str) or not case_id:
            stage2_missing_id_count += 1
        elif case_id in stage2_ids:
            stage2_duplicates.add(case_id)
        else:
            stage2_ids.add(case_id)

    stage4_ids: set[str] = set()
    stage4_duplicates: set[str] = set()
    schema_error_count = 0
    schema_invalid_record_count = 0
    schema_error_examples: list[dict[str, Any]] = []
    context_ref_mismatches: list[str] = []
    observed_selected_ids: set[str] = set()
    selected_tbox_counts: Counter[str] = Counter()
    selected_abox_counts: Counter[str] = Counter()
    selected_group_keys: set[str] = set()
    selected_tbox_keys: set[str] = set()
    for line_number, record in enumerate(iter_jsonl(stage4_path), start=1):
        case_id = record.get("id") if isinstance(record, dict) else None
        errors = list(validator.iter_errors(record))
        if errors:
            schema_invalid_record_count += 1
            schema_error_count += len(errors)
            for error in errors[: max(0, 20 - len(schema_error_examples))]:
                schema_error_examples.append(
                    {
                        "line": line_number,
                        "case_id": case_id,
                        "path": list(error.path),
                        "message": error.message,
                    }
                )
        if not isinstance(case_id, str) or not case_id:
            continue
        if case_id in stage4_ids:
            stage4_duplicates.add(case_id)
        stage4_ids.add(case_id)
        context_id = record.get("context_ref", {}).get("world_state_id")
        if context_id != case_id and len(context_ref_mismatches) < 100:
            context_ref_mismatches.append(case_id)
        if selection is None or case_id not in selected_ids:
            continue
        observed_selected_ids.add(case_id)
        group_key, tbox_key, _ = group_key_for_record(record)
        selected_group_keys.add(group_key)
        if isinstance(tbox_key, str):
            selected_tbox_keys.add(tbox_key)
            selected_tbox_counts[tbox_key] += 1
        else:
            selected_abox_counts[group_key] += 1

    world_state_ids: set[str] = set()
    world_state_duplicates: set[str] = set()
    world_state_error_count = 0
    world_state_error_examples: list[dict[str, Any]] = []
    with Path(world_state_path).open("rb") as handle:
        for entry_id, entry in ijson.kvitems(handle, ""):
            if entry_id in world_state_ids:
                world_state_duplicates.add(entry_id)
            world_state_ids.add(entry_id)
            try:
                validate_world_state_entry(entry_id, entry)
            except ValueError as exc:
                world_state_error_count += 1
                if len(world_state_error_examples) < 20:
                    world_state_error_examples.append({"case_id": entry_id, "message": str(exc)})

    missing_selected_ids = selected_ids - observed_selected_ids
    excluded_case_ids: set[str] = set()
    excluded_group_keys: set[str] = set()
    excluded_tbox_keys: set[str] = set()
    if selection is not None and selection_path is not None:
        excluded_case_ids, excluded_group_keys, excluded_tbox_keys = _selection_exclusions(
            selection, selection_path
        )

    checks = {
        "stage2_nonempty": bool(stage2_ids),
        "stage2_ids_complete": stage2_missing_id_count == 0,
        "stage2_ids_unique": not stage2_duplicates,
        "stage4_nonempty": bool(stage4_ids),
        "stage4_ids_unique": not stage4_duplicates,
        "full_stage4_schema_valid": schema_error_count == 0,
        "world_state_contract_valid": world_state_error_count == 0,
        "world_state_ids_unique": not world_state_duplicates,
        "stage2_stage4_ids_match": stage2_ids == stage4_ids,
        "stage2_world_state_ids_match": stage2_ids == world_state_ids,
        "stage4_context_refs_match": not context_ref_mismatches,
        "subset_partition_present": subset_partition_present,
        "selected_ids_present": selection is None or not missing_selected_ids,
        "tbox_cap_valid": selection is None
        or (isinstance(tbox_cap, int) and max(selected_tbox_counts.values(), default=0) <= tbox_cap),
        "abox_cap_valid": selection is None
        or (isinstance(abox_cap, int) and max(selected_abox_counts.values(), default=0) <= abox_cap),
        "exclude_case_isolation": not (selected_ids & excluded_case_ids),
        "exclude_group_isolation": not (selected_group_keys & excluded_group_keys),
        "exclude_tbox_isolation": not (selected_tbox_keys & excluded_tbox_keys),
    }
    return {
        "passed": all(checks.values()),
        "release_kind": release_kind,
        "checks": checks,
        "counts": {
            "stage2_records": len(stage2_ids),
            "stage4_records": len(stage4_ids),
            "world_state_records": len(world_state_ids),
            "selected": len(selected_ids),
            "selected_observed": len(observed_selected_ids),
            "stage4_schema_invalid_records": schema_invalid_record_count,
            "stage4_schema_errors": schema_error_count,
            "world_state_errors": world_state_error_count,
            "max_tbox_per_revision": max(selected_tbox_counts.values(), default=0),
            "max_abox_per_qid_property": max(selected_abox_counts.values(), default=0),
        },
        "differences": {
            "stage2_only_vs_stage4": sorted(stage2_ids - stage4_ids)[:100],
            "stage4_only_vs_stage2": sorted(stage4_ids - stage2_ids)[:100],
            "stage2_only_vs_world_state": sorted(stage2_ids - world_state_ids)[:100],
            "world_state_only_vs_stage2": sorted(world_state_ids - stage2_ids)[:100],
            "missing_selected_case_ids": sorted(missing_selected_ids)[:100],
            "context_ref_mismatches": context_ref_mismatches,
        },
        "schema_error_examples": schema_error_examples,
        "world_state_error_examples": world_state_error_examples,
    }


def _file_entry(role: str, path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "path": _portable_path(path, root),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_release_manifest(
    *,
    release_root: str | Path,
    stage2_path: str | Path,
    world_state_path: str | Path,
    stage4_path: str | Path,
    schema_path: str | Path,
    release_kind: str,
    selection_manifest_path: str | Path | None = None,
    release_status: str = "candidate",
    artifacts: Iterable[str | Path] = (),
) -> dict[str, Any]:
    if release_status not in RELEASE_STATUSES:
        raise ValueError(f"Unsupported release status: {release_status}")
    validation = validate_release_inputs(
        stage2_path=stage2_path,
        world_state_path=world_state_path,
        stage4_path=stage4_path,
        schema_path=schema_path,
        release_kind=release_kind,
        selection_manifest_path=selection_manifest_path,
    )
    if not validation["passed"]:
        raise ValueError(f"Release validation failed: {json.dumps(validation['checks'], sort_keys=True)}")
    code = _git_state()
    if release_status == "confirmatory" and (not code.get("commit") or code.get("dirty") is not False):
        raise ValueError("Confirmatory releases require a clean Git commit.")

    root = Path(release_root).resolve()
    role_paths: list[tuple[str, Path]] = [
        ("stage2_repairs", Path(stage2_path)),
        ("world_state", Path(world_state_path)),
        ("classified_benchmark", Path(stage4_path)),
        ("classified_benchmark_schema", Path(schema_path)),
    ]
    if selection_manifest_path is not None:
        role_paths.append(("selection_manifest", Path(selection_manifest_path)))
    role_paths.extend((f"extra_{index:03d}", Path(path)) for index, path in enumerate(artifacts, start=1))
    files = [_file_entry(role, path, root) for role, path in role_paths]
    manifest = {
        "manifest_type": "kg_benchmark_release",
        "manifest_version": 2,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "release_kind": release_kind,
        "status": release_status,
        "code": code,
        "validation": validation,
        "files": files,
    }
    schema_file = Path(__file__).resolve().parents[1] / "schemas" / "release_manifest.schema.json"
    Draft202012Validator(json.loads(schema_file.read_text(encoding="utf-8"))).validate(manifest)
    return manifest


def verify_release_manifest(
    manifest_path: str | Path,
    *,
    release_root: str | Path,
    rerun_validation: bool = True,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    schema_file = Path(__file__).resolve().parents[1] / "schemas" / "release_manifest.schema.json"
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    schema_errors = list(Draft202012Validator(schema).iter_errors(manifest))
    root = Path(release_root).resolve()
    file_checks: dict[str, dict[str, Any]] = {}
    role_paths: dict[str, Path] = {}
    for entry in manifest.get("files", []):
        role = entry.get("role")
        relative = entry.get("path")
        if not isinstance(role, str) or not isinstance(relative, str):
            continue
        path = (root / relative).resolve()
        inside_root = path == root or root in path.parents
        exists = inside_root and path.is_file()
        size_matches = exists and path.stat().st_size == entry.get("size_bytes")
        hash_matches = bool(size_matches and sha256_file(path) == entry.get("sha256"))
        file_checks[role] = {
            "inside_root": inside_root,
            "exists": exists,
            "size_matches": size_matches,
            "sha256_matches": hash_matches,
        }
        if inside_root:
            role_paths[role] = path

    required_roles = {
        "stage2_repairs",
        "world_state",
        "classified_benchmark",
        "classified_benchmark_schema",
    }
    if manifest.get("release_kind") == "evaluation":
        required_roles.add("selection_manifest")
    roles_present = required_roles.issubset(role_paths)
    files_pass = roles_present and all(all(check.values()) for check in file_checks.values())
    validation: dict[str, Any] | None = None
    if rerun_validation and files_pass:
        validation = validate_release_inputs(
            stage2_path=role_paths["stage2_repairs"],
            world_state_path=role_paths["world_state"],
            stage4_path=role_paths["classified_benchmark"],
            schema_path=role_paths["classified_benchmark_schema"],
            release_kind=manifest["release_kind"],
            selection_manifest_path=role_paths.get("selection_manifest"),
        )
    checks = {
        "manifest_schema_valid": not schema_errors,
        "required_roles_present": roles_present,
        "all_files_match": files_pass,
        "validation_recomputed": not rerun_validation or bool(validation and validation["passed"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "files": file_checks,
        "validation": validation,
        "schema_errors": [error.message for error in schema_errors[:20]],
        "manifest_sha256": sha256_file(manifest_file),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify an immutable benchmark release manifest.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--release-root", default=".")
    build.add_argument("--release-kind", choices=sorted(RELEASE_KINDS), required=True)
    build.add_argument("--release-status", choices=sorted(RELEASE_STATUSES), default="candidate")
    build.add_argument("--stage2", required=True)
    build.add_argument("--world-state", required=True)
    build.add_argument("--stage4", required=True)
    build.add_argument("--schema", default="schemas/04_classified_benchmark.schema.json")
    build.add_argument("--selection-manifest")
    build.add_argument("--artifact", action="append", default=[])
    build.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--release-root", default=".")
    verify.add_argument("--hashes-only", action="store_true")
    args = parser.parse_args()

    if args.command == "verify":
        result = verify_release_manifest(
            args.manifest,
            release_root=args.release_root,
            rerun_validation=not args.hashes_only,
        )
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["passed"] else 1

    manifest = build_release_manifest(
        release_root=args.release_root,
        stage2_path=args.stage2,
        world_state_path=args.world_state,
        stage4_path=args.stage4,
        schema_path=args.schema,
        release_kind=args.release_kind,
        selection_manifest_path=args.selection_manifest,
        release_status=args.release_status,
        artifacts=args.artifact,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
