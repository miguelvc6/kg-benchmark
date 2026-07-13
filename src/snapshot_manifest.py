"""Build and verify snapshot identity manifests for benchmark releases."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from artifact_release import sha256_file

CONTEXT_POLICY = "later_frozen_context_with_historical_target_reconstruction"


def _schema() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "schemas" / "snapshot_manifest.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_snapshot_manifest(
    *,
    snapshot_id: str,
    stage2_path: str | Path,
    world_state_path: str | Path,
    stage4_path: str | Path,
    sources: Iterable[dict[str, Any]],
    limitations: Iterable[str] = (),
) -> dict[str, Any]:
    normalized_sources = [dict(source) for source in sources]
    if not snapshot_id.strip():
        raise ValueError("snapshot_id is required.")
    manifest = {
        "manifest_type": "kg_benchmark_snapshot",
        "manifest_version": 1,
        "snapshot_id": snapshot_id.strip(),
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "context_policy": CONTEXT_POLICY,
        "repair_time_complete": False,
        "sources": normalized_sources,
        "artifacts": {
            "stage2_repairs_sha256": sha256_file(stage2_path),
            "world_state_sha256": sha256_file(world_state_path),
            "classified_benchmark_sha256": sha256_file(stage4_path),
        },
        "limitations": list(dict.fromkeys(value.strip() for value in limitations if value.strip())),
    }
    Draft202012Validator(_schema()).validate(manifest)
    return manifest


def verify_snapshot_manifest(
    manifest_path: str | Path,
    *,
    stage2_path: str | Path,
    world_state_path: str | Path,
    stage4_path: str | Path,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema_errors = list(Draft202012Validator(_schema()).iter_errors(manifest))
    expected = manifest.get("artifacts", {})
    observed = {
        "stage2_repairs_sha256": sha256_file(stage2_path),
        "world_state_sha256": sha256_file(world_state_path),
        "classified_benchmark_sha256": sha256_file(stage4_path),
    }
    checks = {
        "manifest_schema_valid": not schema_errors,
        "artifact_hashes_match": expected == observed,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "snapshot_id": manifest.get("snapshot_id"),
        "expected_artifacts": expected,
        "observed_artifacts": observed,
        "schema_errors": [error.message for error in schema_errors[:20]],
        "manifest_sha256": sha256_file(path),
    }


def _source(value: str) -> dict[str, Any]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise ValueError("Each --source must use NAME=SOURCE_ID=RETRIEVED_AT_UTC.")
    return {"name": parts[0].strip(), "source_id": parts[1].strip(), "retrieved_at_utc": parts[2].strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify a benchmark snapshot manifest.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--snapshot-id", required=True)
    build.add_argument("--stage2", required=True)
    build.add_argument("--world-state", required=True)
    build.add_argument("--stage4", required=True)
    build.add_argument("--source", action="append", required=True)
    build.add_argument("--limitation", action="append", default=[])
    build.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--stage2", required=True)
    verify.add_argument("--world-state", required=True)
    verify.add_argument("--stage4", required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_snapshot_manifest(
            args.manifest,
            stage2_path=args.stage2,
            world_state_path=args.world_state,
            stage4_path=args.stage4,
        )
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["passed"] else 1
    manifest = build_snapshot_manifest(
        snapshot_id=args.snapshot_id,
        stage2_path=args.stage2,
        world_state_path=args.world_state,
        stage4_path=args.stage4,
        sources=[_source(value) for value in args.source],
        limitations=args.limitation,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
