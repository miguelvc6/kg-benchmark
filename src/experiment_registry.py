#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from artifact_release import sha256_file


def _eligibility(summary: dict[str, Any], status: str) -> list[str]:
    reasons: list[str] = []
    run_info = summary.get("run_info") if isinstance(summary.get("run_info"), dict) else {}
    if status != "confirmatory":
        reasons.append("status_is_not_confirmatory")
    if not run_info.get("model_digest"):
        reasons.append("model_digest_missing")
    code = run_info.get("code") if isinstance(run_info.get("code"), dict) else {}
    if not code.get("commit"):
        reasons.append("git_commit_missing")
    if code.get("dirty") is not False:
        reasons.append("git_worktree_not_clean")
    fingerprints = (
        run_info.get("artifact_fingerprints") if isinstance(run_info.get("artifact_fingerprints"), dict) else {}
    )
    for name in ("classified_benchmark", "world_state", "selection_manifest"):
        fingerprint = fingerprints.get(name)
        if not isinstance(fingerprint, dict) or not fingerprint.get("sha256"):
            reasons.append(f"{name}_fingerprint_missing")
    main_score = summary.get("paper_subsets", {}).get("main_score", {})
    if not isinstance(main_score, dict) or not isinstance(main_score.get("count"), int) or main_score["count"] < 1:
        reasons.append("main_score_subset_missing_or_empty")
    return reasons


def register_experiment(
    *,
    registry_path: str | Path,
    run_summary_path: str | Path,
    status: str,
    protocol_id: str,
    notes: str = "",
    supersedes: list[str] | None = None,
    schema_path: str | Path = "schemas/experiment_registry.schema.json",
) -> dict[str, Any]:
    if status not in {"exploratory", "confirmatory", "superseded"}:
        raise ValueError("Unsupported experiment status.")
    summary_path = Path(run_summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_info = summary.get("run_info") if isinstance(summary.get("run_info"), dict) else {}
    experiment_id = str(run_info.get("run_id") or summary_path.parent.name)
    registry_file = Path(registry_path)
    registry = (
        json.loads(registry_file.read_text(encoding="utf-8"))
        if registry_file.exists()
        else {"manifest_type": "experiment_registry", "manifest_version": 1, "entries": []}
    )
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Experiment registry entries must be a list.")
    if any(entry.get("experiment_id") == experiment_id for entry in entries if isinstance(entry, dict)):
        raise ValueError(f"Experiment already registered: {experiment_id}")
    reasons = _eligibility(summary, status)
    entry = {
        "experiment_id": experiment_id,
        "registered_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": status,
        "protocol_id": protocol_id,
        "run_summary": str(summary_path),
        "run_summary_sha256": sha256_file(summary_path),
        "paper_eligible": not reasons,
        "ineligibility_reasons": reasons,
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
    parser = argparse.ArgumentParser(description="Register an experiment and derive paper eligibility.")
    parser.add_argument("--registry", default="experiments/registry.json")
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--status", choices=("exploratory", "confirmatory", "superseded"), required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--supersedes", default="")
    args = parser.parse_args()
    register_experiment(
        registry_path=args.registry,
        run_summary_path=args.run_summary,
        status=args.status,
        protocol_id=args.protocol_id,
        notes=args.notes,
        supersedes=[value.strip() for value in args.supersedes.split(",") if value.strip()],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
