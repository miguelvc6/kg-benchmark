#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from artifact_release import _git_state, sha256_file, verify_release_manifest

PROTOCOL_STATUSES = {"draft", "frozen"}


def _relative_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Protocol file must be inside protocol root {root}: {resolved}") from exc


def _fingerprint(role: str, path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "path": _relative_path(path, root),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_protocol_manifest(
    *,
    protocol_root: str | Path,
    protocol_id: str,
    release_manifest_path: str | Path,
    models: Iterable[dict[str, str]],
    conditions: Iterable[str],
    prompt_files: Iterable[str | Path],
    schema_files: Iterable[str | Path],
    analysis_plan_path: str | Path,
    expected_selected_count: int,
    expected_main_score_count: int,
    status: str = "draft",
) -> dict[str, Any]:
    if status not in PROTOCOL_STATUSES:
        raise ValueError(f"Unsupported protocol status: {status}")
    normalized_models = [dict(model) for model in models]
    if not normalized_models or any(not model.get("name") or not model.get("digest") for model in normalized_models):
        raise ValueError("At least one model with a stable name and digest is required.")
    normalized_conditions = list(dict.fromkeys(value for value in conditions if value))
    if not normalized_conditions:
        raise ValueError("At least one preregistered condition is required.")
    if expected_selected_count < 1 or not 0 < expected_main_score_count <= expected_selected_count:
        raise ValueError("Expected population counts are invalid.")

    root = Path(protocol_root).resolve()
    release_path = Path(release_manifest_path)
    release_verification = verify_release_manifest(release_path, release_root=root)
    if not release_verification["passed"]:
        raise ValueError("Protocol cannot reference an invalid release manifest.")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    code = _git_state()
    if status == "frozen":
        if release.get("status") != "confirmatory":
            raise ValueError("Frozen protocols require a confirmatory release.")
        if not code.get("commit") or code.get("dirty") is not False:
            raise ValueError("Frozen protocols require a clean Git commit.")
        if release.get("code", {}).get("commit") != code["commit"]:
            raise ValueError("Protocol and release must reference the same Git commit.")

    files = [_fingerprint("release_manifest", release_path, root)]
    files.extend(
        _fingerprint(f"prompt_{index:03d}", Path(path), root)
        for index, path in enumerate(prompt_files, start=1)
    )
    files.extend(
        _fingerprint(f"schema_{index:03d}", Path(path), root)
        for index, path in enumerate(schema_files, start=1)
    )
    files.append(_fingerprint("analysis_plan", Path(analysis_plan_path), root))
    manifest = {
        "manifest_type": "research_protocol_freeze",
        "manifest_version": 1,
        "protocol_id": protocol_id,
        "status": status,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "code": code,
        "release": {
            "path": _relative_path(release_path, root),
            "sha256": sha256_file(release_path),
            "release_kind": release["release_kind"],
            "release_status": release["status"],
        },
        "models": normalized_models,
        "conditions": normalized_conditions,
        "expected_population": {
            "selected_count": expected_selected_count,
            "main_score_count": expected_main_score_count,
        },
        "files": files,
    }
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "research_protocol.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(manifest)
    return manifest


def verify_protocol_manifest(
    manifest_path: str | Path,
    *,
    protocol_root: str | Path,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "research_protocol.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_errors = list(Draft202012Validator(schema).iter_errors(manifest))
    root = Path(protocol_root).resolve()
    file_checks: dict[str, dict[str, bool]] = {}
    for entry in manifest.get("files", []):
        role = entry.get("role")
        relative = entry.get("path")
        if not isinstance(role, str) or not isinstance(relative, str):
            continue
        candidate = (root / relative).resolve()
        inside_root = candidate == root or root in candidate.parents
        exists = inside_root and candidate.is_file()
        size_matches = exists and candidate.stat().st_size == entry.get("size_bytes")
        hash_matches = bool(size_matches and sha256_file(candidate) == entry.get("sha256"))
        file_checks[role] = {
            "inside_root": inside_root,
            "exists": exists,
            "size_matches": size_matches,
            "sha256_matches": hash_matches,
        }
    release_relative = manifest.get("release", {}).get("path")
    release_path = (root / release_relative).resolve() if isinstance(release_relative, str) else root / "missing"
    release_hash_matches = release_path.is_file() and sha256_file(release_path) == manifest.get("release", {}).get(
        "sha256"
    )
    release_verification = (
        verify_release_manifest(release_path, release_root=root) if release_hash_matches else {"passed": False}
    )
    release = json.loads(release_path.read_text(encoding="utf-8")) if release_hash_matches else {}
    checks = {
        "manifest_schema_valid": not schema_errors,
        "status_frozen": manifest.get("status") == "frozen",
        "code_clean": manifest.get("code", {}).get("dirty") is False,
        "release_hash_matches": bool(release_hash_matches),
        "release_verified": bool(release_verification.get("passed")),
        "release_confirmatory": release.get("status") == "confirmatory",
        "code_matches_release": manifest.get("code", {}).get("commit") == release.get("code", {}).get("commit"),
        "all_protocol_files_match": bool(file_checks) and all(
            all(item.values()) for item in file_checks.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "files": file_checks,
        "release_verification": release_verification,
        "schema_errors": [error.message for error in schema_errors[:20]],
        "manifest": manifest,
        "manifest_sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify a frozen research protocol.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--protocol-root", default=".")
    build.add_argument("--protocol-id", required=True)
    build.add_argument("--release-manifest", required=True)
    build.add_argument("--model", action="append", required=True, help="NAME=DIGEST; repeat for each model.")
    build.add_argument("--condition", action="append", required=True)
    build.add_argument("--prompt", action="append", required=True)
    build.add_argument("--schema", action="append", required=True)
    build.add_argument("--analysis-plan", required=True)
    build.add_argument("--expected-selected-count", required=True, type=int)
    build.add_argument("--expected-main-score-count", required=True, type=int)
    build.add_argument("--status", choices=sorted(PROTOCOL_STATUSES), default="draft")
    build.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--protocol-root", default=".")
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_protocol_manifest(args.manifest, protocol_root=args.protocol_root)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["passed"] else 1
    models = []
    for value in args.model:
        if "=" not in value:
            raise ValueError("Each --model must use NAME=DIGEST.")
        name, digest = value.split("=", 1)
        models.append({"name": name, "digest": digest})
    manifest = build_protocol_manifest(
        protocol_root=args.protocol_root,
        protocol_id=args.protocol_id,
        release_manifest_path=args.release_manifest,
        models=models,
        conditions=args.condition,
        prompt_files=args.prompt,
        schema_files=args.schema,
        analysis_plan_path=args.analysis_plan,
        expected_selected_count=args.expected_selected_count,
        expected_main_score_count=args.expected_main_score_count,
        status=args.status,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
