from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from jsonschema import Draft202012Validator

from guardian.generation_cache import (
    GenerationCache,
    build_generation_request_spec,
    generation_request_key,
)
from guardian.reasoning import (
    _few_shot_examples_from_bank,
    _load_support_bank_for_runner,
    _prepend_few_shot_examples,
    build_prompt_bundle,
    build_track_diagnosis_prompt_bundle,
    prompt_visible_case_id,
    run_reasoning_floor,
)
from kg_benchmark.methodology import require_frozen_methodology

MATRIX_FILENAME = "matrix.json"
REQUESTS_FILENAME = "requests.jsonl"
TASK_TO_MANIFEST = {"repair_proposal": "proposal", "track_diagnosis": "track_diagnosis"}


class MatrixWorkflowError(ValueError):
    """Raised when a paper execution matrix is invalid or incomplete."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _artifact(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise MatrixWorkflowError(f"Required artifact is missing: {path}")
    recorded_path = os.path.relpath(path.resolve(), relative_to.resolve()) if relative_to else str(path.resolve())
    result: dict[str, Any] = {
        "path": recorded_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix == ".jsonl":
        result["records"] = _jsonl_count(path)
    return result


def _resolve(matrix_dir: Path, recorded_path: str) -> Path:
    path = Path(recorded_path)
    return path if path.is_absolute() else (matrix_dir / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixWorkflowError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatrixWorkflowError(f"JSON artifact must contain an object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise MatrixWorkflowError(f"Cannot read JSONL artifact {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MatrixWorkflowError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise MatrixWorkflowError(f"Expected an object at {path}:{line_number}")
            yield value


@lru_cache(maxsize=None)
def _validator(schema_path: Path) -> Draft202012Validator:
    return Draft202012Validator(_load_json(schema_path))


def _validate(value: Any, schema_path: Path, *, label: str) -> None:
    errors = sorted(_validator(schema_path).iter_errors(value), key=lambda error: list(error.path))
    if errors:
        location = ".".join(str(part) for part in errors[0].path)
        suffix = f" at {location}" if location else ""
        raise MatrixWorkflowError(f"{label} fails {schema_path.name}{suffix}: {errors[0].message}")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True).encode())
            handle.write(b"\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                count += 1
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return count


def _policy_validated_models(payload: dict[str, Any], schema_root: Path) -> list[dict[str, Any]]:
    _validate(payload, schema_root / "model-matrix.schema.json", label="model configuration")
    models = payload["models"]
    identifiers = [model["model_id"] for model in models]
    if len(identifiers) != len(set(identifiers)):
        raise MatrixWorkflowError("Model identifiers must be unique.")
    for model in models:
        model_id = model["model_id"]
        resolved = model["revision_status"] == "resolved"
        if resolved != (isinstance(model.get("model_revision"), str) and bool(model["model_revision"].strip())):
            raise MatrixWorkflowError(f"{model_id} has inconsistent revision_status and model_revision.")
        if model.get("tools_disabled") is not True:
            raise MatrixWorkflowError(f"{model_id} must disable tools.")
        if model["provider"] == "ollama":
            if model["execution_mode"] != "sync":
                raise MatrixWorkflowError(f"{model_id} must use synchronous Ollama execution.")
            required = ("temperature", "top_p", "seed", "context_length", "ollama_think")
            if any(key not in model for key in required):
                raise MatrixWorkflowError(f"{model_id} is missing a frozen Ollama inference parameter.")
        else:
            if not model.get("reasoning_effort") or not model.get("deployment"):
                raise MatrixWorkflowError(f"{model_id} must freeze Azure deployment and reasoning effort.")
            if model["execution_mode"] == "batch" and model.get("batch_sync_retry_fallback") is not False:
                raise MatrixWorkflowError(f"{model_id} must explicitly disable Azure batch-to-sync fallback.")
    return models


def _think(value: Any) -> bool | str | None:
    if value == "enabled":
        return True
    if value == "disabled":
        return False
    return value


def _inference_settings(model: dict[str, Any]) -> dict[str, Any]:
    """Mirror guardian.reasoning._resolved_inference_settings exactly."""
    return {
        "context_length": model.get("context_length"),
        "max_output_tokens": model.get("max_output_tokens"),
        "temperature": model.get("temperature"),
        "top_p": model.get("top_p"),
        "seed": model.get("seed"),
        "think": _think(model.get("ollama_think")),
        "max_retries": model.get("max_transport_retries"),
        "reasoning_effort": model.get("reasoning_effort") if model["provider"] == "azure" else None,
        "tools_disabled": True,
    }


def _dataset_artifacts(dataset_dir: Path, matrix_dir: Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("manifest_type") != "kg_benchmark_dataset" or manifest.get("status") != "final":
        raise MatrixWorkflowError("Experiment planning requires the promoted final dataset manifest.")
    artifact_rows = manifest.get("artifacts")
    if not isinstance(artifact_rows, dict):
        raise MatrixWorkflowError("Dataset manifest has no artifacts object.")
    roles = {
        "manifest": manifest_path,
        "cases": dataset_dir / str(artifact_rows.get("cases", {}).get("path", "cases.jsonl")),
        "world_state": dataset_dir / str(artifact_rows.get("world_state", {}).get("path", "source/world-state.jsonl")),
        "support_bank": dataset_dir / str(
            artifact_rows.get("support_bank", {}).get("path", "selections/support-bank.json")
        ),
        "methodology_lock": dataset_dir / str(manifest.get("methodology", {}).get("lock", {}).get("path", "")),
    }
    recorded = {role: _artifact(path, relative_to=matrix_dir) for role, path in roles.items()}
    for role, path in roles.items():
        published = artifact_rows.get(role)
        if role == "manifest":
            continue
        if role == "methodology_lock":
            published = manifest.get("methodology", {}).get("lock")
        if not isinstance(published, dict) or published.get("sha256") != recorded[role]["sha256"]:
            raise MatrixWorkflowError(f"Dataset {role} does not match its final manifest.")
    return recorded, roles, manifest


def _check_freeze_bindings(
    *,
    dataset_manifest: dict[str, Any],
    methodology_lock_path: Path,
    models_path: Path,
    protocol_path: Path,
) -> None:
    lock = _load_json(methodology_lock_path)
    files = lock.get("files")
    if not isinstance(files, dict):
        raise MatrixWorkflowError("Dataset methodology lock has no file hash mapping.")
    expected_models = files.get("paper/models.json")
    expected_protocol = files.get("paper/protocol.json")
    if expected_models != _sha256_file(models_path):
        raise MatrixWorkflowError("Model configuration does not match the dataset methodology freeze.")
    if expected_protocol != _sha256_file(protocol_path):
        raise MatrixWorkflowError("Protocol does not match the dataset methodology freeze.")
    if dataset_manifest.get("protocol", {}).get("sha256") != expected_protocol:
        raise MatrixWorkflowError("Dataset protocol and methodology-lock protocol hashes disagree.")


def _population(path: Path, schema_root: Path) -> dict[str, Any]:
    value = _load_json(path)
    _validate(value, schema_root / "selection-manifest.schema.json", label=f"population {path}")
    case_ids = value["selected_case_ids"]
    if value["case_count"] != len(case_ids) or len(case_ids) != len(set(case_ids)):
        raise MatrixWorkflowError(f"Population {path} has inconsistent or duplicate selected case IDs.")
    return value


def _load_rows_by_id(path: Path, required: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        case_id = row.get("id")
        if isinstance(case_id, str) and case_id in required:
            if case_id in result:
                raise MatrixWorkflowError(f"Duplicate case id in {path}: {case_id}")
            result[case_id] = row
    missing = sorted(required - set(result))
    if missing:
        raise MatrixWorkflowError(f"Cases referenced by a population/support bank are missing: {missing[:5]}")
    return result


def _load_world(path: Path, required: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        case_id = row.get("id")
        payload = row.get("world_state") if isinstance(row.get("world_state"), dict) else row.get("context")
        if isinstance(case_id, str) and case_id in required and isinstance(payload, dict):
            result[case_id] = payload
    missing = sorted(required - set(result))
    if missing:
        raise MatrixWorkflowError(f"World-state records are missing for cases: {missing[:5]}")
    return result


class _MemoryWorldStore:
    def __init__(self, values: dict[str, dict[str, Any]]):
        self.values = values

    def get(self, case_id: str) -> dict[str, Any] | None:
        return self.values.get(case_id)


def _model_selection(
    models: list[dict[str, Any]], model_ids: Iterable[str] | None
) -> list[dict[str, Any]]:
    requested = list(model_ids or [])
    if not requested:
        return models
    unknown = sorted(set(requested) - {model["model_id"] for model in models})
    if unknown:
        raise MatrixWorkflowError(f"Unknown model identifiers: {unknown}")
    requested_set = set(requested)
    return [model for model in models if model["model_id"] in requested_set]


def _id(prefix: str, value: Any) -> str:
    return f"{prefix}_{_sha256_value(value)[:20]}"


def _render_bundle(
    *,
    task: str,
    record: dict[str, Any],
    world: dict[str, Any] | None,
    context: str,
    regime: str,
    support_manifest: dict[str, Any] | None,
    records_by_id: dict[str, dict[str, Any]],
    world_store: _MemoryWorldStore,
    example_counts: dict[str, int],
) -> Any:
    visible_id = prompt_visible_case_id(str(record["id"]))
    if task == "track_diagnosis":
        bundle = build_track_diagnosis_prompt_bundle(record, world, context, visible_case_id=visible_id)
        support_task = "track_diagnosis"
    else:
        proposal_track = str(record.get("track") or "A_BOX")
        bundle = build_prompt_bundle(
            record,
            world,
            context,
            proposal_track=proposal_track,
            visible_case_id=visible_id,
        )
        support_task = "t_box_repair" if proposal_track == "T_BOX" else "a_box_repair"
    if regime == "static_few_shot":
        if support_manifest is None:
            raise MatrixWorkflowError("Static few-shot planning requires the support bank.")
        examples = _few_shot_examples_from_bank(
            support_manifest=support_manifest,
            eval_record=record,
            task=support_task,
            records_by_id=records_by_id,
            world_store=world_store,  # type: ignore[arg-type]
            context_bundle=context,
            example_count=example_counts[support_task],
        )
        bundle = _prepend_few_shot_examples(bundle, examples)
    return bundle


def plan_matrix(
    *,
    dataset_dir: Path,
    models_path: Path,
    protocol_path: Path,
    output_root: Path,
    population_paths: Iterable[Path] | None = None,
    model_ids: Iterable[str] | None = None,
    schema_root: Path = Path("schemas"),
) -> dict[str, Any]:
    """Materialize all logical cells and exact generation requests without provider calls."""
    dataset_dir = dataset_dir.resolve()
    models_path = models_path.resolve()
    protocol_path = protocol_path.resolve()
    schema_root = schema_root.resolve()
    output_root = output_root.resolve()

    model_payload = _load_json(models_path)
    models = _model_selection(_policy_validated_models(model_payload, schema_root), model_ids)
    protocol = _load_json(protocol_path)
    dimensions = model_payload["request_dimensions"]
    if dimensions["prompt_regimes"] != protocol.get("conditions", {}).get("prompt_regimes"):
        raise MatrixWorkflowError("Model and protocol prompt-regime dimensions disagree.")
    if dimensions["context_bundles"] != protocol.get("conditions", {}).get("context_bundles"):
        raise MatrixWorkflowError("Model and protocol context dimensions disagree.")
    if dimensions["tasks"] != list(protocol.get("tasks", {})):
        raise MatrixWorkflowError("Model and protocol task dimensions disagree.")

    provisional_dir = output_root / ".planning"
    dataset_artifacts, dataset_paths, dataset_manifest = _dataset_artifacts(dataset_dir, provisional_dir)
    _check_freeze_bindings(
        dataset_manifest=dataset_manifest,
        methodology_lock_path=dataset_paths["methodology_lock"],
        models_path=models_path,
        protocol_path=protocol_path,
    )

    population_values: dict[str, dict[str, Any]] = {}
    population_files: dict[str, Path] = {}
    associations: dict[str, list[str]] = {}
    explicit_paths = list(population_paths or [])
    if explicit_paths:
        for raw_path in explicit_paths:
            path = raw_path.resolve()
            value = _population(path, schema_root)
            name = value["name"]
            if name in population_values:
                raise MatrixWorkflowError(f"Duplicate population name: {name}")
            population_values[name] = value
            population_files[name] = path
        for model in models:
            associations[model["model_id"]] = list(population_values)
    else:
        for model in models:
            name = model["population"]
            path = dataset_dir / "selections" / f"{name}.json"
            if name not in population_values:
                population_values[name] = _population(path, schema_root)
                population_files[name] = path
            associations[model["model_id"]] = [name]

    for model in models:
        if not explicit_paths:
            expected = (
                population_values[model["population"]]["case_count"]
                * len(dimensions["tasks"])
                * len(dimensions["prompt_regimes"])
                * len(dimensions["context_bundles"])
            )
            if model["expected_calls"] != expected:
                raise MatrixWorkflowError(
                    f"{model['model_id']} expected_calls={model['expected_calls']} but its frozen matrix requires {expected}."
                )

    support_raw, support_ids = _load_support_bank_for_runner(dataset_paths["support_bank"])
    selected_ids = {
        case_id
        for value in population_values.values()
        for case_id in value["selected_case_ids"]
    }
    all_required = selected_ids | support_ids
    records = _load_rows_by_id(dataset_paths["cases"], all_required)
    world_values = _load_world(dataset_paths["world_state"], all_required)
    world_store = _MemoryWorldStore(world_values)
    example_counts = protocol.get("few_shot", {}).get("default_example_counts")
    if not isinstance(example_counts, dict) or any(not isinstance(value, int) for value in example_counts.values()):
        raise MatrixWorkflowError("Protocol has invalid few-shot example counts.")

    basis = {
        "dataset": {key: value["sha256"] for key, value in dataset_artifacts.items()},
        "models_sha256": _sha256_file(models_path),
        "protocol_sha256": _sha256_file(protocol_path),
        "dimensions": dimensions,
        "models": models,
        "associations": associations,
        "populations": {
            name: {"sha256": _sha256_file(population_files[name]), "case_ids": value["selected_case_ids"]}
            for name, value in population_values.items()
        },
    }
    matrix_id = _id("matrix", basis)
    matrix_dir = output_root / matrix_id
    dataset_artifacts, dataset_paths, dataset_manifest = _dataset_artifacts(dataset_dir, matrix_dir)
    model_artifact = _artifact(models_path, relative_to=matrix_dir)
    protocol_artifact = _artifact(protocol_path, relative_to=matrix_dir)
    population_records = {
        name: {
            "artifact": _artifact(population_files[name], relative_to=matrix_dir),
            "case_count": value["case_count"],
        }
        for name, value in population_values.items()
    }

    cells: list[dict[str, Any]] = []
    cell_lookup: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for model in models:
        for population_name in associations[model["model_id"]]:
            for regime in dimensions["prompt_regimes"]:
                for context in dimensions["context_bundles"]:
                    group_dims = [model["model_id"], population_name, regime, context]
                    group_id = _id("group", [matrix_id, *group_dims])
                    for task in dimensions["tasks"]:
                        cell_dims = [*group_dims, task]
                        cell_id = _id("cell", [matrix_id, *cell_dims])
                        cell = {
                            "cell_id": cell_id,
                            "execution_group_id": group_id,
                            "model_id": model["model_id"],
                            "population": population_name,
                            "task": task,
                            "prompt_regime": regime,
                            "context_bundle": context,
                            "expected_requests": population_values[population_name]["case_count"],
                            "manifest_path": f"cells/{cell_id}.json",
                        }
                        cells.append(cell)
                        cell_lookup[(model["model_id"], population_name, task, regime, context)] = cell

    request_rows: list[dict[str, Any]] = []
    for model in models:
        inference = _inference_settings(model)
        inference_hash = _sha256_value(inference)
        request_model = model.get("deployment") if model["provider"] == "azure" else model["model"]
        for population_name in associations[model["model_id"]]:
            for task in dimensions["tasks"]:
                for regime in dimensions["prompt_regimes"]:
                    for context in dimensions["context_bundles"]:
                        cell = cell_lookup[(model["model_id"], population_name, task, regime, context)]
                        for case_id in population_values[population_name]["selected_case_ids"]:
                            record = records[case_id]
                            bundle = _render_bundle(
                                task=task,
                                record=record,
                                world=world_values.get(case_id),
                                context=context,
                                regime=regime,
                                support_manifest=support_raw,
                                records_by_id=records,
                                world_store=world_store,
                                example_counts=example_counts,
                            )
                            revision = model.get("model_revision")
                            spec = None
                            key = None
                            if isinstance(revision, str) and revision:
                                spec = build_generation_request_spec(
                                    provider=model["provider"],
                                    model=request_model,
                                    model_digest=revision,
                                    inference_settings=inference,
                                    prompt=bundle.prompt,
                                    system_prompt=bundle.system_prompt,
                                    response_format=bundle.response_format,
                                )
                                key = generation_request_key(spec)
                            row = {
                                "matrix_id": matrix_id,
                                "cell_id": cell["cell_id"],
                                "execution_group_id": cell["execution_group_id"],
                                "model_id": model["model_id"],
                                "provider": model["provider"],
                                "model": request_model,
                                "population": population_name,
                                "case_id": case_id,
                                "case_payload_sha256": _sha256_value(record),
                                "task": task,
                                "prompt_regime": regime,
                                "rendered_prompt_sha256": _sha256_value(
                                    {
                                        "system_prompt": bundle.system_prompt,
                                        "prompt": bundle.prompt,
                                        "response_format": bundle.response_format,
                                    }
                                ),
                                "context_bundle": context,
                                "context_sha256": _sha256_value(bundle.context_audit),
                                "model_revision": revision,
                                "inference_parameters_sha256": inference_hash,
                                "request_key": key,
                                "request_spec": spec,
                            }
                            _validate(row, schema_root / "matrix-request.schema.json", label="matrix request")
                            request_rows.append(row)

    matrix_dir.mkdir(parents=True, exist_ok=True)
    requests_path = matrix_dir / REQUESTS_FILENAME
    prospective_bytes = b"".join(_canonical_bytes(row) for row in request_rows)
    if requests_path.exists() and requests_path.read_bytes() != prospective_bytes:
        raise MatrixWorkflowError(f"Existing request plan differs for stable matrix id {matrix_id}.")
    if not requests_path.exists():
        _write_jsonl_atomic(requests_path, request_rows)
    request_artifact = _artifact(requests_path, relative_to=matrix_dir)
    workload_by_model = Counter(row["model_id"] for row in request_rows)
    manifest = {
        "manifest_type": "paper_experiment_matrix",
        "manifest_version": 1,
        "matrix_id": matrix_id,
        "dataset": dataset_artifacts,
        "model_configuration": model_artifact,
        "protocol": protocol_artifact,
        "dimensions": dimensions,
        "populations": population_records,
        "models": models,
        "cells": cells,
        "requests": request_artifact,
        "workload": {
            "expected_requests": len(request_rows),
            "logical_cells": len(cells),
            "execution_groups": len({cell["execution_group_id"] for cell in cells}),
            "by_model": dict(workload_by_model),
        },
        "validation": {
            "population_not_in_generation_identity": True,
            "tools_disabled": True,
            "provider_policies_valid": True,
        },
    }
    _validate(manifest, schema_root / "experiment-matrix.schema.json", label="experiment matrix")
    matrix_path = matrix_dir / MATRIX_FILENAME
    prospective_manifest = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True).encode() + b"\n"
    if matrix_path.exists() and matrix_path.read_bytes() != prospective_manifest:
        raise MatrixWorkflowError(f"Existing matrix manifest differs for stable matrix id {matrix_id}.")
    if not matrix_path.exists():
        _write_json_atomic(matrix_path, manifest)
    return {"matrix_dir": str(matrix_dir), "matrix": manifest}


def _load_matrix(matrix_dir: Path, schema_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    matrix_dir = matrix_dir.resolve()
    matrix_path = matrix_dir / MATRIX_FILENAME
    manifest = _load_json(matrix_path)
    _validate(manifest, schema_root / "experiment-matrix.schema.json", label="experiment matrix")
    requests_path = _resolve(matrix_dir, manifest["requests"]["path"])
    if _sha256_file(requests_path) != manifest["requests"]["sha256"]:
        raise MatrixWorkflowError("Matrix request-plan hash mismatch.")
    if requests_path.stat().st_size != manifest["requests"]["bytes"]:
        raise MatrixWorkflowError("Matrix request-plan size mismatch.")
    rows = list(_iter_jsonl(requests_path))
    if len(rows) != manifest["workload"]["expected_requests"]:
        raise MatrixWorkflowError("Matrix request-plan count mismatch.")
    request_schema = schema_root / "matrix-request.schema.json"
    for row in rows:
        _validate(row, request_schema, label="matrix request")
        if row["matrix_id"] != manifest["matrix_id"]:
            raise MatrixWorkflowError("Request row is bound to another matrix.")
    bound_inputs = [
        *manifest["dataset"].values(),
        manifest["model_configuration"],
        manifest["protocol"],
        *(population["artifact"] for population in manifest["populations"].values()),
    ]
    for record in bound_inputs:
        path = _resolve(matrix_dir, record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or _sha256_file(path) != record["sha256"]:
            raise MatrixWorkflowError(f"Matrix input artifact changed: {record['path']}")
    return manifest, rows, matrix_path


def dry_run_matrix(
    *, matrix_dir: Path, generation_cache_path: Path, schema_root: Path = Path("schemas")
) -> dict[str, Any]:
    """Inspect cache coverage and revision blockers without constructing a provider."""
    manifest, rows, _ = _load_matrix(matrix_dir, schema_root.resolve())
    resolved_rows = [row for row in rows if isinstance(row.get("request_key"), str)]
    existing = GenerationCache(generation_cache_path).existing_keys({row["request_key"] for row in resolved_rows})
    missing_models = sorted({row["model_id"] for row in rows if row.get("request_key") is None})
    by_model: dict[str, dict[str, int]] = {}
    for model in manifest["models"]:
        model_rows = [row for row in rows if row["model_id"] == model["model_id"]]
        by_model[model["model_id"]] = {
            "expected_memberships": len(model_rows),
            "cache_hit_memberships": sum(row.get("request_key") in existing for row in model_rows),
            "new_request_memberships": sum(
                isinstance(row.get("request_key"), str) and row["request_key"] not in existing for row in model_rows
            ),
            "missing_revision_memberships": sum(row.get("request_key") is None for row in model_rows),
        }
    unique_keys = {row["request_key"] for row in resolved_rows}
    return {
        "matrix_id": manifest["matrix_id"],
        "no_provider_calls": True,
        "logical_cells": len(manifest["cells"]),
        "execution_groups": manifest["workload"]["execution_groups"],
        "expected_request_memberships": len(rows),
        "unique_resolved_requests": len(unique_keys),
        "cache_hit_memberships": sum(row["request_key"] in existing for row in resolved_rows),
        "new_request_memberships": sum(row["request_key"] not in existing for row in resolved_rows),
        "unique_cache_hits": len(unique_keys & existing),
        "unique_new_requests": len(unique_keys - existing),
        "missing_revision_models": missing_models,
        "missing_revision_memberships": sum(row.get("request_key") is None for row in rows),
        "by_model": by_model,
    }


def _run_rows(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path)) if path.is_file() else []


def _cell_observation(
    *,
    cell: dict[str, Any],
    request_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    cached_keys: set[str],
) -> dict[str, Any]:
    expected_ids = {row["case_id"] for row in request_rows if row["cell_id"] == cell["cell_id"]}
    manifest_task = TASK_TO_MANIFEST[cell["task"]]
    matching = {
        row.get("case_id"): row
        for row in run_rows
        if row.get("task_type") == manifest_task and row.get("ablation_bundle") == cell["context_bundle"]
    }
    failed_ids = {
        case_id for case_id, row in matching.items() if case_id in expected_ids and row.get("parse_status") == "request_error"
    }
    completed_ids = {
        case_id for case_id, row in matching.items() if case_id in expected_ids and row.get("parse_status") != "request_error"
    }
    missing_ids = expected_ids - completed_ids - failed_ids
    cell_keys = {
        row["request_key"]
        for row in request_rows
        if row["cell_id"] == cell["cell_id"] and isinstance(row.get("request_key"), str)
    }
    cache_present = len(cell_keys & cached_keys)
    counts = {
        "expected": len(expected_ids),
        "completed": len(completed_ids),
        "failed": len(failed_ids),
        "missing": len(missing_ids),
    }
    cache = {
        "present": cache_present,
        "missing": len(cell_keys) - cache_present,
        "all_present": cache_present == len(cell_keys) == len(expected_ids),
    }
    status = "complete" if counts["completed"] == counts["expected"] and cache["all_present"] else "incomplete"
    if failed_ids:
        status = "failed"
    return {"status": status, "counts": counts, "cache_validation": cache}


def _make_cell_manifest(
    *,
    matrix_dir: Path,
    matrix: dict[str, Any],
    matrix_path: Path,
    cell: dict[str, Any],
    requests_path: Path,
    request_rows: list[dict[str, Any]],
    cached_keys: set[str],
) -> dict[str, Any]:
    group_dir = matrix_dir / "executions" / cell["execution_group_id"]
    paths = {
        "run_config": group_dir / "run_config.json",
        "run_manifest": group_dir / "run_manifest.jsonl",
        "summary": group_dir / "reasoning_floor_summary.json",
    }
    for role, path in paths.items():
        if not path.is_file():
            raise MatrixWorkflowError(f"Execution group lacks {role}: {path}")
    observation = _cell_observation(
        cell=cell,
        request_rows=request_rows,
        run_rows=_run_rows(paths["run_manifest"]),
        cached_keys=cached_keys,
    )
    return {
        "manifest_type": "experiment_matrix_cell",
        "manifest_version": 1,
        "matrix_id": matrix["matrix_id"],
        "matrix_sha256": _sha256_file(matrix_path),
        "cell_id": cell["cell_id"],
        "execution_group_id": cell["execution_group_id"],
        "dimensions": {
            "model_id": cell["model_id"],
            "population": cell["population"],
            "task": cell["task"],
            "prompt_regime": cell["prompt_regime"],
            "context_bundle": cell["context_bundle"],
        },
        "status": observation["status"],
        "counts": observation["counts"],
        "request_plan": _artifact(requests_path, relative_to=matrix_dir),
        "artifacts": {role: _artifact(path, relative_to=matrix_dir) for role, path in paths.items()},
        "cache_validation": observation["cache_validation"],
    }


def matrix_status(
    *, matrix_dir: Path, generation_cache_path: Path, schema_root: Path = Path("schemas")
) -> dict[str, Any]:
    matrix_dir = matrix_dir.resolve()
    schema_root = schema_root.resolve()
    matrix, request_rows, matrix_path = _load_matrix(matrix_dir, schema_root)
    resolved_keys = {row["request_key"] for row in request_rows if isinstance(row.get("request_key"), str)}
    cached_keys = GenerationCache(generation_cache_path).existing_keys(resolved_keys)
    cell_results: list[dict[str, Any]] = []
    for cell in matrix["cells"]:
        manifest_path = _resolve(matrix_dir, cell["manifest_path"])
        if not manifest_path.is_file():
            cell_results.append({"cell_id": cell["cell_id"], "status": "incomplete", "reason": "manifest_missing"})
            continue
        try:
            stored = _load_json(manifest_path)
            _validate(stored, schema_root / "matrix-cell.schema.json", label="matrix cell")
            if stored["matrix_sha256"] != _sha256_file(matrix_path) or stored["cell_id"] != cell["cell_id"]:
                raise MatrixWorkflowError("Cell manifest binding mismatch.")
            for record in [stored["request_plan"], *stored["artifacts"].values()]:
                path = _resolve(matrix_dir, record["path"])
                if path.stat().st_size != record["bytes"] or _sha256_file(path) != record["sha256"]:
                    raise MatrixWorkflowError(f"Cell artifact changed: {record['path']}")
            observed = _cell_observation(
                cell=cell,
                request_rows=request_rows,
                run_rows=_run_rows(_resolve(matrix_dir, stored["artifacts"]["run_manifest"]["path"])),
                cached_keys=cached_keys,
            )
            if any(stored[field] != observed[field] for field in ("status", "counts", "cache_validation")):
                raise MatrixWorkflowError("Stored cell counts do not match execution artifacts.")
            cell_results.append({"cell_id": cell["cell_id"], "status": observed["status"]})
        except (OSError, MatrixWorkflowError) as exc:
            cell_results.append({"cell_id": cell["cell_id"], "status": "incomplete", "reason": str(exc)})
    counts = Counter(result["status"] for result in cell_results)
    return {
        "matrix_id": matrix["matrix_id"],
        "complete": counts["complete"] == len(matrix["cells"]),
        "logical_cells": len(matrix["cells"]),
        "complete_cells": counts["complete"],
        "failed_cells": counts["failed"],
        "incomplete_cells": counts["incomplete"],
        "cells": cell_results,
    }


def execute_matrix(
    *,
    matrix_dir: Path,
    generation_cache_path: Path,
    repo_root: Path = Path("."),
    model_ids: Iterable[str] | None = None,
    schema_root: Path = Path("schemas"),
    methodology_check: Callable[[Path], Any] = require_frozen_methodology,
    run_executor: Callable[..., dict[str, Any]] = run_reasoning_floor,
) -> dict[str, Any]:
    """Run missing physical groups, then seal and independently verify logical cells."""
    methodology_check(repo_root.resolve())
    matrix_dir = matrix_dir.resolve()
    schema_root = schema_root.resolve()
    matrix, request_rows, matrix_path = _load_matrix(matrix_dir, schema_root)
    missing_revisions = sorted({row["model_id"] for row in request_rows if row.get("request_key") is None})
    if missing_revisions:
        raise MatrixWorkflowError(f"Execution is blocked by unresolved model revisions: {missing_revisions}")
    selected_models = _model_selection(matrix["models"], model_ids)
    selected_ids = {model["model_id"] for model in selected_models}
    model_by_id = {model["model_id"]: model for model in selected_models}
    selected_cells = [cell for cell in matrix["cells"] if cell["model_id"] in selected_ids]
    requests_path = _resolve(matrix_dir, matrix["requests"]["path"])
    protocol_path = _resolve(matrix_dir, matrix["protocol"]["path"])
    cases_path = _resolve(matrix_dir, matrix["dataset"]["cases"]["path"])
    world_path = _resolve(matrix_dir, matrix["dataset"]["world_state"]["path"])
    support_path = _resolve(matrix_dir, matrix["dataset"]["support_bank"]["path"])
    protocol = _load_json(protocol_path)
    counts = protocol["few_shot"]["default_example_counts"]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in selected_cells:
        grouped[cell["execution_group_id"]].append(cell)
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            matrix["populations"][item[1][0]["population"]]["case_count"],
            next(index for index, model in enumerate(matrix["models"]) if model["model_id"] == item[1][0]["model_id"]),
            item[1][0]["prompt_regime"],
            item[1][0]["context_bundle"],
            item[0],
        ),
    )
    executed_groups = 0
    skipped_groups = 0
    for group_id, cells in ordered_groups:
        group_complete = True
        for cell in cells:
            path = _resolve(matrix_dir, cell["manifest_path"])
            if not path.is_file():
                group_complete = False
                break
        if group_complete:
            partial = matrix_status(
                matrix_dir=matrix_dir,
                generation_cache_path=generation_cache_path,
                schema_root=schema_root,
            )
            states = {row["cell_id"]: row["status"] for row in partial["cells"]}
            group_complete = all(states.get(cell["cell_id"]) == "complete" for cell in cells)
        if group_complete:
            skipped_groups += 1
            continue
        exemplar = cells[0]
        model = model_by_id[exemplar["model_id"]]
        population_path = _resolve(matrix_dir, matrix["populations"][exemplar["population"]]["artifact"]["path"])
        group_dir = matrix_dir / "executions" / group_id
        run_executor(
            classified_path=cases_path,
            world_state_path=world_path,
            output_dir=group_dir.parent,
            resume_run_dir=group_dir,
            model_name=model.get("deployment") if model["provider"] == "azure" else model["model"],
            model_endpoint=model["provider"],
            reasoning_effort=model.get("reasoning_effort"),
            context_length=model.get("context_length"),
            max_output_tokens=model.get("max_output_tokens"),
            temperature=model.get("temperature"),
            top_p=model.get("top_p"),
            seed=model.get("seed"),
            ollama_think=_think(model.get("ollama_think")),
            max_retries=model["max_transport_retries"],
            protocol_path=protocol_path,
            model_digest=model["model_revision"],
            generation_cache_path=generation_cache_path,
            execution_mode=model["execution_mode"],
            batch_completion_window=model.get("batch_completion_window", "24h"),
            batch_poll_interval_seconds=model.get("batch_poll_interval_seconds", 60),
            batch_sync_retry_fallback=model.get("batch_sync_retry_fallback", False),
            proposal_track_mode="oracle",
            oracle_diagnosis_mode="run",
            prompt_regime=exemplar["prompt_regime"],
            support_bank_path=support_path if exemplar["prompt_regime"] == "static_few_shot" else None,
            a_box_example_count=counts["a_box_repair"],
            t_box_example_count=counts["t_box_repair"],
            diagnosis_example_count=counts["track_diagnosis"],
            selection_manifest_path=population_path,
            ablation_bundles=[exemplar["context_bundle"]],
        )
        executed_groups += 1
        group_keys = {
            row["request_key"]
            for row in request_rows
            if row["execution_group_id"] == group_id and isinstance(row.get("request_key"), str)
        }
        cached = GenerationCache(generation_cache_path).existing_keys(group_keys)
        for cell in cells:
            cell_manifest = _make_cell_manifest(
                matrix_dir=matrix_dir,
                matrix=matrix,
                matrix_path=matrix_path,
                cell=cell,
                requests_path=requests_path,
                request_rows=request_rows,
                cached_keys=cached,
            )
            _validate(cell_manifest, schema_root / "matrix-cell.schema.json", label="matrix cell")
            _write_json_atomic(_resolve(matrix_dir, cell["manifest_path"]), cell_manifest)

    status = matrix_status(
        matrix_dir=matrix_dir,
        generation_cache_path=generation_cache_path,
        schema_root=schema_root,
    )
    selected_states = [row for row in status["cells"] if any(row["cell_id"] == cell["cell_id"] for cell in selected_cells)]
    if not all(row["status"] == "complete" for row in selected_states):
        raise MatrixWorkflowError("Selected execution cells remain incomplete after execution.")
    return {**status, "executed_groups": executed_groups, "skipped_groups": skipped_groups}
