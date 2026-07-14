#!/usr/bin/env python3
"""Replay evaluation from immutable generation artifacts without provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from guardian.evaluator import evaluate_benchmark, evaluate_track_diagnosis_bundle
from guardian.tbox_taxonomy_patch_run import (
    evaluate_tbox_taxonomy_patch_bundle,
    prepare_tbox_taxonomy_gold,
)

EVALUATION_REPLAY_MANIFEST_VERSION = 1
EVALUATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _optional_fingerprint(path: Path) -> dict[str, Any] | None:
    return _fingerprint(path) if path.is_file() else None


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


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _resolve_input_path(override: str | Path | None, recorded: Any, *, field: str) -> Path:
    candidate = Path(override if override is not None else str(recorded or "")).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Cannot replay evaluation because {field} is unavailable at {candidate}. "
            f"Provide an explicit --{field.replace('_', '-')} override."
        )
    return candidate


def rescore_run(
    *,
    run_dir: str | Path,
    evaluation_id: str,
    classified_path: str | Path | None = None,
    world_state_path: str | Path | None = None,
    selection_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    if not EVALUATION_ID_PATTERN.fullmatch(evaluation_id):
        raise ValueError(
            "evaluation_id must start with an alphanumeric character and contain only "
            "letters, digits, dots, underscores, and hyphens."
        )
    source_run_dir = Path(run_dir).resolve()
    run_config_path = source_run_dir / "run_config.json"
    run_config = _read_object(run_config_path)
    taxonomy_mode = run_config.get("tbox_task_version") == "tbox_taxonomy_patch_v1"
    if not taxonomy_mode:
        raise ValueError("Only taxonomy-patch paper runs are supported by the active evaluator.")

    output_dir = source_run_dir / "evaluations" / evaluation_id
    if output_dir.exists():
        raise FileExistsError(
            f"Evaluation {evaluation_id!r} already exists at {output_dir}; choose a new version id."
        )
    benchmark = _resolve_input_path(
        classified_path, run_config.get("classified_benchmark"), field="classified_benchmark"
    )
    world_state = _resolve_input_path(
        world_state_path, run_config.get("world_state"), field="world_state"
    )
    recorded_selection = run_config.get("selection_manifest")
    selection = None
    if selection_manifest_path is not None or recorded_selection:
        selection = _resolve_input_path(
            selection_manifest_path, recorded_selection, field="selection_manifest"
        )

    bundles = run_config.get("ablation_bundles")
    if not isinstance(bundles, list) or not bundles or not all(isinstance(item, str) for item in bundles):
        raise ValueError("run_config.json does not contain a valid ablation_bundles list.")
    selected_case_ids = run_config.get("selected_case_ids")
    if not isinstance(selected_case_ids, list) or not all(
        isinstance(case_id, str) and case_id for case_id in selected_case_ids
    ):
        raise ValueError("run_config.json does not contain valid selected_case_ids.")
    run_manifest_path = source_run_dir / "run_manifest.jsonl"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(run_manifest_path)

    taxonomy_gold = None
    standard_evaluation_case_ids = selected_case_ids
    if taxonomy_mode:
        taxonomy_gold = prepare_tbox_taxonomy_gold(
            classified_path=benchmark,
            selected_case_ids=selected_case_ids,
            selection_manifest_path=selection,
            require_complete=True,
        )
        tbox_case_ids = set(taxonomy_gold.tbox_case_ids)
        standard_evaluation_case_ids = [
            case_id for case_id in selected_case_ids if case_id not in tbox_case_ids
        ]

    output_dir.mkdir(parents=True)
    bundle_summaries: dict[str, Any] = {}
    source_artifacts: dict[str, Any] = {
        "run_config": _fingerprint(run_config_path),
        "run_manifest": _fingerprint(run_manifest_path),
        "classified_benchmark": _fingerprint(benchmark),
        "world_state": _fingerprint(world_state),
        "selection_manifest": _fingerprint(selection) if selection is not None else None,
        "bundles": {},
    }
    for bundle in bundles:
        source_bundle_dir = source_run_dir / bundle
        a_box_path = source_bundle_dir / "a_box_proposals.jsonl"
        t_box_path = source_bundle_dir / "t_box_proposals.jsonl"
        diagnoses_path = source_bundle_dir / "track_diagnoses.jsonl"
        taxonomy_path = source_bundle_dir / "t_box_taxonomy_patch_proposals.jsonl"
        source_artifacts["bundles"][bundle] = {
            "a_box_proposals": _optional_fingerprint(a_box_path),
            "t_box_proposals": _optional_fingerprint(t_box_path),
            "t_box_taxonomy_patch_proposals": _optional_fingerprint(taxonomy_path),
            "track_diagnoses": _optional_fingerprint(diagnoses_path),
        }
        bundle_output = output_dir / bundle
        bundle_output.mkdir()
        _traces, summary = evaluate_benchmark(
            classified_path=benchmark,
            world_state_path=world_state,
            a_box_proposals_path=a_box_path if a_box_path.is_file() else None,
            t_box_proposals_path=t_box_path if t_box_path.is_file() else None,
            track_diagnoses_path=diagnoses_path if diagnoses_path.is_file() else None,
            run_manifest_path=run_manifest_path,
            ablation_bundle=bundle,
            case_ids=standard_evaluation_case_ids or None,
            selection_manifest_path=selection if standard_evaluation_case_ids else None,
            out_traces_path=bundle_output / "evaluation_traces.jsonl",
            out_summary_path=bundle_output / "evaluation_summary.json",
            collect_traces=False,
            classified_records=None if standard_evaluation_case_ids else [],
            classified_input_path=benchmark,
        )
        diagnosis_summary = evaluate_track_diagnosis_bundle(
            classified_path=benchmark,
            track_diagnoses_path=diagnoses_path if diagnoses_path.is_file() else None,
            run_manifest_path=run_manifest_path,
            ablation_bundle=bundle,
            case_ids=selected_case_ids,
            out_traces_path=bundle_output / "diagnosis_evaluation_traces.jsonl",
            out_summary_path=bundle_output / "diagnosis_evaluation_summary.json",
        )
        if taxonomy_gold is not None:
            taxonomy_summary = evaluate_tbox_taxonomy_patch_bundle(
                prepared_gold=taxonomy_gold,
                predictions_path=taxonomy_path,
                out_traces_path=bundle_output / "tbox_taxonomy_patch_evaluation_traces.jsonl",
                out_summary_path=bundle_output / "tbox_taxonomy_patch_evaluation_summary.json",
            )
            bundle_summaries[bundle] = {
                "a_box": summary,
                "tbox_taxonomy_patch": taxonomy_summary,
                "track_diagnosis": diagnosis_summary,
                "combined_repair_success_score": None,
            }
        else:
            bundle_summaries[bundle] = {"a_box": summary, "track_diagnosis": diagnosis_summary}

    evaluator_path = Path(__file__).resolve().parent / "guardian" / "evaluator.py"
    taxonomy_evaluator_path = (
        Path(__file__).resolve().parent / "guardian" / "tbox_taxonomy_patch_evaluator.py"
    )
    taxonomy_gold_path = Path(__file__).resolve().parent / "lib" / "tbox_taxonomy_patch_gold.py"
    taxonomy_run_path = (
        Path(__file__).resolve().parent / "guardian" / "tbox_taxonomy_patch_run.py"
    )
    replay_path = Path(__file__).resolve()
    manifest = {
        "manifest_type": "evaluation_replay",
        "manifest_version": EVALUATION_REPLAY_MANIFEST_VERSION,
        "evaluation_id": evaluation_id,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_run_dir": str(source_run_dir),
        "provider_calls": 0,
        "selected_case_count": len(selected_case_ids),
        "ablation_bundles": bundles,
        "metric_families": ["a_box_repair_v1", "tbox_taxonomy_patch_v1", "track_diagnosis_v1"],
        "combined_repair_success_score": False,
        "source_artifacts": source_artifacts,
        "evaluation_code": {
            "git": _git_state(),
            "evaluator": _fingerprint(evaluator_path),
            "tbox_taxonomy_patch_evaluator": (
                _fingerprint(taxonomy_evaluator_path) if taxonomy_mode else None
            ),
            "tbox_taxonomy_patch_gold_extractor": (
                _fingerprint(taxonomy_gold_path) if taxonomy_mode else None
            ),
            "tbox_taxonomy_patch_run": _fingerprint(taxonomy_run_path) if taxonomy_mode else None,
            "replay_driver": _fingerprint(replay_path),
        },
        "outputs": {},
    }
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(bundle_summaries, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    manifest["outputs"]["combined_summary"] = _fingerprint(summary_path)
    for bundle in bundles:
        bundle_output = output_dir / bundle
        manifest["outputs"][bundle] = {
            "traces": _fingerprint(bundle_output / "evaluation_traces.jsonl"),
            "summary": _fingerprint(bundle_output / "evaluation_summary.json"),
            "diagnosis_traces": _fingerprint(bundle_output / "diagnosis_evaluation_traces.jsonl"),
            "diagnosis_summary": _fingerprint(bundle_output / "diagnosis_evaluation_summary.json"),
            "tbox_taxonomy_patch_traces": (
                _fingerprint(bundle_output / "tbox_taxonomy_patch_evaluation_traces.jsonl")
                if taxonomy_mode
                else None
            ),
            "tbox_taxonomy_patch_summary": (
                _fingerprint(bundle_output / "tbox_taxonomy_patch_evaluation_summary.json")
                if taxonomy_mode
                else None
            ),
        }
    manifest_path = output_dir / "evaluation_manifest.json"
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "evaluation-replay.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(manifest))
    if errors:
        raise ValueError(f"Evaluation replay manifest fails its schema: {errors[0].message}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-evaluate an existing generation run without issuing model requests."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--classified-benchmark", default=None)
    parser.add_argument("--world-state", default=None)
    parser.add_argument("--selection-manifest", default=None)
    args = parser.parse_args()
    manifest = rescore_run(
        run_dir=args.run_dir,
        evaluation_id=args.evaluation_id,
        classified_path=args.classified_benchmark,
        world_state_path=args.world_state,
        selection_manifest_path=args.selection_manifest,
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
