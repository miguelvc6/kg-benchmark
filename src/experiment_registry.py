#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from artifact_release import sha256_file
from protocol_freeze import verify_protocol_manifest

REQUIRED_FINGERPRINTS = (
    "classified_benchmark",
    "world_state",
    "selection_manifest",
    "a_box_schema",
    "t_box_schema",
    "track_diagnosis_schema",
    "prompt_definitions",
)


def _resolve_fingerprint_path(value: str, summary_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    summary_relative = (summary_path.parent / path).resolve()
    return summary_relative if summary_relative.is_file() else (Path.cwd() / path).resolve()


def _verify_run_fingerprints(
    summary: dict[str, Any], summary_path: Path, protocol_files: set[str], release_files: dict[str, str]
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    evidence: dict[str, Any] = {}
    run_info = summary.get("run_info") if isinstance(summary.get("run_info"), dict) else {}
    fingerprints = run_info.get("artifact_fingerprints")
    fingerprints = fingerprints if isinstance(fingerprints, dict) else {}
    release_role_by_name = {
        "classified_benchmark": "classified_benchmark",
        "world_state": "world_state",
        "selection_manifest": "selection_manifest",
    }
    for name in REQUIRED_FINGERPRINTS:
        fingerprint = fingerprints.get(name)
        check = {"present": isinstance(fingerprint, dict)}
        if not isinstance(fingerprint, dict):
            reasons.append(f"{name}_fingerprint_missing")
            evidence[name] = check
            continue
        path_value = fingerprint.get("path")
        expected_hash = fingerprint.get("sha256")
        expected_size = fingerprint.get("size_bytes")
        path = _resolve_fingerprint_path(path_value, summary_path) if isinstance(path_value, str) else None
        check["exists"] = bool(path and path.is_file())
        check["size_matches"] = bool(
            path and path.is_file() and isinstance(expected_size, int) and path.stat().st_size == expected_size
        )
        check["hash_matches"] = bool(
            check["size_matches"] and isinstance(expected_hash, str) and sha256_file(path) == expected_hash
        )
        release_role = release_role_by_name.get(name)
        check["matches_release"] = (
            release_role is None and isinstance(expected_hash, str) and expected_hash in protocol_files
        ) or (release_role is not None and release_files.get(release_role) == expected_hash)
        if not all(check.values()):
            reasons.append(f"{name}_fingerprint_unverified")
        evidence[name] = check
    return reasons, evidence


def _eligibility(
    summary: dict[str, Any],
    summary_path: Path,
    status: str,
    protocol_verification: dict[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    evidence: dict[str, Any] = {}
    run_info = summary.get("run_info") if isinstance(summary.get("run_info"), dict) else {}
    if status != "confirmatory":
        reasons.append("status_is_not_confirmatory")
    if not protocol_verification or not protocol_verification.get("passed"):
        reasons.append("frozen_protocol_not_verified")
        return reasons, evidence

    protocol = protocol_verification["manifest"]
    if protocol.get("protocol_phase") != "execution":
        reasons.append("protocol_is_not_execution_phase")
    if protocol.get("release", {}).get("release_kind") != "evaluation":
        reasons.append("protocol_release_is_not_evaluation_release")
    release_manifest = protocol_verification.get("release_verification", {}).get("validation")
    release_manifest_file_entries = (
        json.loads(
            (
                Path(protocol_verification["protocol_root"])
                / protocol["release"]["path"]
            ).read_text(encoding="utf-8")
        ).get("files", [])
        if protocol_verification.get("protocol_root")
        else []
    )
    release_files = {
        item["role"]: item["sha256"]
        for item in release_manifest_file_entries
        if isinstance(item, dict) and isinstance(item.get("role"), str) and isinstance(item.get("sha256"), str)
    }
    protocol_hashes = {
        item["sha256"]
        for item in protocol.get("files", [])
        if isinstance(item, dict) and isinstance(item.get("sha256"), str)
    }

    code = run_info.get("code") if isinstance(run_info.get("code"), dict) else {}
    if code.get("commit") != protocol.get("code", {}).get("commit"):
        reasons.append("git_commit_does_not_match_protocol")
    if code.get("dirty") is not False:
        reasons.append("git_worktree_not_clean")
    digest = run_info.get("model_digest")
    model_digests = {model.get("digest") for model in protocol.get("models", []) if isinstance(model, dict)}
    if not digest or digest not in model_digests:
        reasons.append("model_digest_not_preregistered")

    fingerprint_reasons, fingerprint_evidence = _verify_run_fingerprints(
        summary, summary_path, protocol_hashes, release_files
    )
    reasons.extend(fingerprint_reasons)
    evidence["artifact_fingerprints"] = fingerprint_evidence

    subsets = summary.get("paper_subsets") if isinstance(summary.get("paper_subsets"), dict) else {}
    main_score = subsets.get("main_score") if isinstance(subsets.get("main_score"), dict) else {}
    all_selected = subsets.get("all_selected") if isinstance(subsets.get("all_selected"), dict) else {}
    expected = protocol.get("expected_population", {})
    if main_score.get("count") != expected.get("main_score_count"):
        reasons.append("main_score_count_does_not_match_protocol")
    if all_selected.get("count") != expected.get("selected_count"):
        reasons.append("selected_count_does_not_match_protocol")
    release_selected = (
        release_manifest.get("counts", {}).get("selected") if isinstance(release_manifest, dict) else None
    )
    if release_selected != expected.get("selected_count"):
        reasons.append("release_selected_count_does_not_match_protocol")

    observed_conditions = set(summary.get("by_ablation_bundle", {}))
    registered_conditions = set(protocol.get("conditions", []))
    if not observed_conditions or not observed_conditions.issubset(registered_conditions):
        reasons.append("run_conditions_not_preregistered")
    evidence["population"] = {
        "main_score_count": main_score.get("count"),
        "selected_count": all_selected.get("count"),
        "expected": expected,
    }
    evidence["conditions"] = {
        "observed": sorted(observed_conditions),
        "preregistered": sorted(registered_conditions),
    }
    return list(dict.fromkeys(reasons)), evidence


def _relative_to_registry(path: Path, registry_path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=registry_path.parent.resolve())).as_posix()


def register_experiment(
    *,
    registry_path: str | Path,
    run_summary_path: str | Path,
    status: str,
    protocol_manifest_path: str | Path | None = None,
    protocol_root: str | Path = ".",
    notes: str = "",
    supersedes: list[str] | None = None,
    schema_path: str | Path = "schemas/experiment_registry.schema.json",
) -> dict[str, Any]:
    if status not in {"exploratory", "confirmatory", "superseded"}:
        raise ValueError("Unsupported experiment status.")
    summary_path = Path(run_summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_info = summary.get("run_info") if isinstance(summary.get("run_info"), dict) else {}
    experiment_id = str(run_info.get("run_id") or summary.get("run_id") or summary_path.parent.name)
    registry_file = Path(registry_path)
    registry = (
        json.loads(registry_file.read_text(encoding="utf-8"))
        if registry_file.exists()
        else {"manifest_type": "experiment_registry", "manifest_version": 2, "entries": []}
    )
    if registry.get("manifest_version") == 1 and not registry.get("entries"):
        registry["manifest_version"] = 2
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Experiment registry entries must be a list.")
    if any(entry.get("experiment_id") == experiment_id for entry in entries if isinstance(entry, dict)):
        raise ValueError(f"Experiment already registered: {experiment_id}")

    protocol_verification = None
    protocol_path = Path(protocol_manifest_path).resolve() if protocol_manifest_path else None
    if protocol_path is not None:
        protocol_verification = verify_protocol_manifest(protocol_path, protocol_root=protocol_root)
        protocol_verification["protocol_root"] = str(Path(protocol_root).resolve())
    reasons, evidence = _eligibility(summary, summary_path, status, protocol_verification)
    if status == "confirmatory" and reasons:
        raise ValueError(f"Confirmatory registration failed: {', '.join(reasons)}")
    entry = {
        "experiment_id": experiment_id,
        "registered_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "protocol_id": (
            protocol_verification.get("manifest", {}).get("protocol_id") if protocol_verification else None
        ),
        "protocol_manifest": _relative_to_registry(protocol_path, registry_file) if protocol_path else None,
        "protocol_manifest_sha256": sha256_file(protocol_path) if protocol_path else None,
        "run_summary": _relative_to_registry(summary_path, registry_file),
        "run_summary_sha256": sha256_file(summary_path),
        "paper_eligible": status == "confirmatory" and not reasons,
        "ineligibility_reasons": reasons,
        "verification_evidence": evidence,
        "supersedes": list(dict.fromkeys(supersedes or [])),
        "notes": notes,
    }
    entries.append(entry)
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(registry)
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(json.dumps(registry, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Register and cryptographically verify an experiment.")
    parser.add_argument("--registry", default="experiments/registry.json")
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--status", choices=("exploratory", "confirmatory", "superseded"), required=True)
    parser.add_argument("--protocol-manifest")
    parser.add_argument("--protocol-root", default=".")
    parser.add_argument("--notes", default="")
    parser.add_argument("--supersedes", default="")
    args = parser.parse_args()
    register_experiment(
        registry_path=args.registry,
        run_summary_path=args.run_summary,
        status=args.status,
        protocol_manifest_path=args.protocol_manifest,
        protocol_root=args.protocol_root,
        notes=args.notes,
        supersedes=[value.strip() for value in args.supersedes.split(",") if value.strip()],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
