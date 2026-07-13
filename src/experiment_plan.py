#!/usr/bin/env python3
"""Validate and materialize the extensible paper execution matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from paper_prompt_profile import REPO_ROOT, load_prompt_profile, sha256_file


def load_execution_matrix(path: str | Path) -> dict[str, Any]:
    matrix_path = Path(path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "execution_matrix.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(matrix)
    prompt_configuration = matrix.get("prompt_configuration")
    if prompt_configuration:
        prompt_path = (REPO_ROOT / prompt_configuration).resolve()
        load_prompt_profile(prompt_path)
        actual_prompt_sha = sha256_file(prompt_path)
        if matrix.get("prompt_configuration_sha256") != actual_prompt_sha:
            raise ValueError(
                "Execution matrix prompt_configuration_sha256 does not match the prompt profile: "
                f"expected {matrix.get('prompt_configuration_sha256')}, computed {actual_prompt_sha}."
            )
    validate_execution_matrix(matrix)
    return matrix


def validate_execution_matrix(
    matrix: dict[str, Any],
    *,
    require_methodology_frozen: bool = False,
    require_frozen: bool = False,
) -> None:
    population_ids = [entry["population_id"] for entry in matrix["populations"]]
    if len(population_ids) != len(set(population_ids)):
        raise ValueError("population_id values must be unique.")
    model_ids = [entry["model_id"] for entry in matrix["models"]]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("model_id values must be unique.")
    known_populations = set(population_ids)
    unknown = sorted(
        {entry["population_id"] for entry in matrix["models"]} - known_populations
    )
    if unknown:
        raise ValueError(f"Models reference unknown populations: {', '.join(unknown)}.")
    for model in matrix["models"]:
        if model["provider"] == "azure":
            if model["execution_mode"] != "batch":
                raise ValueError("Azure reference models must use batch execution.")
            if model["batch_sync_retry_fallback"]:
                raise ValueError("Azure reference models must disable synchronous retry fallback.")
            if any(
                model[field] is not None
                for field in ("context_length", "temperature", "top_p", "seed", "ollama_think")
            ):
                raise ValueError("Azure inference must omit Ollama-only controls.")
        if model["provider"] == "ollama":
            if model["reasoning_effort"] is not None:
                raise ValueError("Ollama models use ollama_think, not OpenAI reasoning_effort.")
            if any(model[field] is None for field in ("context_length", "temperature", "top_p", "seed", "ollama_think")):
                raise ValueError("Ollama paper models require explicit context, sampling, seed, and thinking controls.")
    if matrix["tbox_task_version"] != "tbox_taxonomy_patch_v1":
        raise ValueError("Paper execution requires the tbox_taxonomy_patch_v1 T-box task.")
    if matrix["reporting_policy"]["combined_abox_tbox_score"]:
        raise ValueError("A-box and T-box metric families must not be collapsed into one score.")
    methodology_required = (
        require_methodology_frozen
        or require_frozen
        or matrix["status"] in {"methodology_frozen", "frozen"}
    )
    if methodology_required:
        if matrix["status"] not in {"methodology_frozen", "frozen"}:
            if require_frozen:
                raise ValueError("Confirmatory planning requires matrix status=frozen.")
            raise ValueError("Post-freeze acquisition requires matrix status=methodology_frozen or frozen.")
        if not matrix["prompt_configuration"]:
            raise ValueError("A methodology-frozen matrix requires a prompt_configuration reference.")
        if not matrix["prompt_configuration_sha256"]:
            raise ValueError("A methodology-frozen matrix requires a prompt_configuration_sha256.")
    frozen_required = require_frozen or matrix["status"] == "frozen"
    if not frozen_required:
        return
    missing_revisions = [model["model_id"] for model in matrix["models"] if not model["model_digest"]]
    missing_selections = [
        population["population_id"]
        for population in matrix["populations"]
        if not population["selection_sha256"]
    ]
    if matrix["status"] != "frozen":
        raise ValueError("Confirmatory planning requires matrix status=frozen.")
    if not matrix["prompt_configuration"]:
        raise ValueError("A frozen matrix requires a prompt_configuration reference.")
    if not matrix["prompt_configuration_sha256"]:
        raise ValueError("A frozen matrix requires a prompt_configuration_sha256.")
    if missing_revisions:
        raise ValueError(f"Frozen matrix models lack immutable revisions: {', '.join(missing_revisions)}.")
    if missing_selections:
        raise ValueError(
            f"Frozen matrix populations lack selection hashes: {', '.join(missing_selections)}."
        )


def build_execution_plan(matrix: dict[str, Any]) -> dict[str, Any]:
    validate_execution_matrix(matrix)
    populations = {entry["population_id"]: entry for entry in matrix["populations"]}
    runs: list[dict[str, Any]] = []
    for model in matrix["models"]:
        if not model["enabled"]:
            continue
        population = populations[model["population_id"]]
        requests_per_case = len(population["ablation_bundles"])
        if population["proposal_track_mode"] == "diagnosis_routed" or population["oracle_diagnosis_mode"] == "run":
            requests_per_case *= 2
        cache_path = str(Path(matrix["generation_cache_root"]) / f"{model['model_id']}.sqlite")
        model_digest = model["model_digest"] or f"<REQUIRED_MODEL_DIGEST:{model['model_id']}>"
        argv = [
            "kg-reasoning-floor",
            "--model-endpoint",
            model["provider"],
            "--model",
            model["model"],
            "--prompt-profile",
            matrix["prompt_configuration"],
            "--tbox-task-version",
            matrix["tbox_task_version"],
            "--model-digest",
            model_digest,
            "--generation-cache",
            cache_path,
            "--selection-manifest",
            population["selection_manifest"],
            "--ablation-bundles",
            ",".join(population["ablation_bundles"]),
            "--proposal-track-mode",
            population["proposal_track_mode"],
            "--oracle-diagnosis-mode",
            population["oracle_diagnosis_mode"],
            "--execution-mode",
            model["execution_mode"],
            "--max-output-tokens",
            str(model["max_output_tokens"]),
            "--max-retries",
            str(model["transport_max_retries"]),
        ]
        if model["parallel_workers"] is not None:
            argv.extend(["--parallel-workers", str(model["parallel_workers"])])
        if model["reasoning_effort"] is not None:
            argv.extend(["--reasoning-effort", model["reasoning_effort"]])
        if model["provider"] == "ollama":
            argv.extend(
                [
                    "--context-length",
                    str(model["context_length"]),
                    "--temperature",
                    str(model["temperature"]),
                    "--top-p",
                    str(model["top_p"]),
                    "--seed",
                    str(model["seed"]),
                    "--ollama-think",
                    model["ollama_think"],
                ]
            )
        if not model["batch_sync_retry_fallback"]:
            argv.append("--no-batch-sync-retry-fallback")
        runs.append(
            {
                "model_id": model["model_id"],
                "population_id": population["population_id"],
                "expected_case_count": population["expected_case_count"],
                "requests_per_case": requests_per_case,
                "expected_request_count": population["expected_case_count"] * requests_per_case,
                "generation_cache": cache_path,
                "argv": argv,
            }
        )
    return {
        "matrix_id": matrix["matrix_id"],
        "status": matrix["status"],
        "prompt_configuration": matrix["prompt_configuration"],
        "prompt_configuration_sha256": matrix["prompt_configuration_sha256"],
        "tbox_task_version": matrix["tbox_task_version"],
        "reporting_policy": matrix["reporting_policy"],
        "runs": runs,
        "expected_request_count": sum(run["expected_request_count"] for run in runs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or plan paper model execution.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--matrix", default="experiments/paper_execution_models_v1.json")
    validate_parser.add_argument("--require-methodology-frozen", action="store_true")
    validate_parser.add_argument("--require-frozen", action="store_true")
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--matrix", default="experiments/paper_execution_models_v1.json")
    plan_parser.add_argument("--output", default=None)
    args = parser.parse_args()
    matrix = load_execution_matrix(args.matrix)
    if args.command == "validate":
        validate_execution_matrix(
            matrix,
            require_methodology_frozen=args.require_methodology_frozen,
            require_frozen=args.require_frozen,
        )
        print(json.dumps({"valid": True, "matrix_id": matrix["matrix_id"], "status": matrix["status"]}))
        return 0
    plan = build_execution_plan(matrix)
    rendered = json.dumps(plan, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
