from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

CONFIG_PATHS = {
    "protocol": "paper/protocol.json",
    "models": "paper/models.json",
    "selection_policy": "paper/selection-policy.json",
    "analysis": "paper/analysis.json",
}
METHODOLOGY_SCHEMA_PATH = "schemas/methodology.schema.json"
METHODOLOGY_LOCK_PATH = "paper/methodology.lock.json"
REQUIRED_MODEL_CONFIGURATION = {
    "ollama_qwen3_30b": {
        "provider": "ollama",
        "model": "qwen3:30b",
        "population": "main-1200",
        "execution_mode": "sync",
        "ollama_think": "enabled",
        "max_transport_retries": 2,
        "tools_disabled": True,
    },
    "ollama_llama3_3_70b": {
        "provider": "ollama",
        "model": "llama3.3:70b",
        "population": "main-1200",
        "execution_mode": "sync",
        "ollama_think": "disabled",
        "max_transport_retries": 2,
        "tools_disabled": True,
    },
    "ollama_gpt_oss_120b": {
        "provider": "ollama",
        "model": "gpt-oss:120b",
        "population": "main-1200",
        "execution_mode": "sync",
        "ollama_think": "high",
        "max_transport_retries": 2,
        "tools_disabled": True,
    },
    "azure_gpt_5_6_sol_high": {
        "provider": "azure",
        "model": "gpt-5.6-sol",
        "population": "azure-600",
        "execution_mode": "sync",
        "reasoning_effort": "high",
        "max_transport_retries": 2,
        "tools_disabled": True,
    },
}
EXPECTED_GENERATION_IDENTITY = {
    "case_payload_sha256",
    "task",
    "rendered_prompt_sha256",
    "context_sha256",
    "model_revision",
    "inference_parameters_sha256",
}
EXPECTED_CONTRASTS = {
    "context_within_zero_shot",
    "context_within_few_shot",
    "few_shot_within_logic_only",
    "few_shot_within_local_graph",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MethodologyError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MethodologyError(f"Required methodology file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MethodologyError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MethodologyError(f"Methodology file must contain a JSON object: {path}")
    return value


def load_methodology_bundle(repo_root: Path) -> dict[str, dict[str, Any]]:
    root = repo_root.resolve()
    return {name: _read_json(root / relative_path) for name, relative_path in CONFIG_PATHS.items()}


def _schema_errors(repo_root: Path, bundle: dict[str, Any]) -> list[str]:
    schema_path = repo_root / METHODOLOGY_SCHEMA_PATH
    try:
        schema = _read_json(schema_path)
    except MethodologyError as exc:
        return [str(exc)]
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        return [f"Invalid methodology schema: {exc}"]
    errors = sorted(Draft202012Validator(schema).iter_errors(bundle), key=lambda item: list(item.absolute_path))
    rendered: list[str] = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "methodology"
        rendered.append(f"{location}: {error.message}")
    return rendered


def _duplicates(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _population_size(population: Any) -> int | None:
    if not isinstance(population, dict):
        return None
    values = list(population.values())
    if not values or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        return None
    return sum(values)


def validate_methodology_bundle(bundle: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    workloads: dict[str, int] = {}
    protocol = bundle.get("protocol", {})
    models = bundle.get("models", {})
    selection = bundle.get("selection_policy", {})
    analysis = bundle.get("analysis", {})

    statuses = {
        "protocol": protocol.get("status"),
        "models": models.get("status"),
        "selection_policy": selection.get("status"),
        "analysis": analysis.get("status"),
    }
    if len(set(statuses.values())) != 1:
        errors.append(f"Methodology component statuses disagree: {statuses}")
    if statuses["protocol"] not in {"freeze_candidate", "frozen"}:
        errors.append("Methodology status must be freeze_candidate or frozen.")

    seed = protocol.get("seed")
    for name, value in (("selection_policy", selection.get("seed")), ("analysis", analysis.get("seed"))):
        if value != seed:
            errors.append(f"{name} seed {value!r} does not match protocol seed {seed!r}.")

    expected_references = {
        "selection_policy": CONFIG_PATHS["selection_policy"],
        "model_configuration": CONFIG_PATHS["models"],
        "analysis_configuration": CONFIG_PATHS["analysis"],
        "analysis_plan": "paper/analysis-plan.md",
    }
    for field, expected in expected_references.items():
        if protocol.get(field) != expected:
            errors.append(f"protocol.{field} must be {expected!r}.")

    tasks = protocol.get("tasks", {})
    conditions = protocol.get("conditions", {})
    dimensions = models.get("request_dimensions", {})
    if set(dimensions.get("tasks", [])) != set(tasks):
        errors.append("Model task dimensions do not match protocol tasks.")
    if dimensions.get("prompt_regimes") != conditions.get("prompt_regimes"):
        errors.append("Model prompt-regime dimensions do not match the protocol.")
    if dimensions.get("context_bundles") != conditions.get("context_bundles"):
        errors.append("Model context-bundle dimensions do not match the protocol.")
    if tasks.get("repair_proposal", {}).get("routing") != "oracle":
        errors.append("Confirmatory repair proposals must use oracle routing.")
    if tasks.get("track_diagnosis", {}).get("routes_proposals") is not False:
        errors.append("Track diagnosis must not route confirmatory proposals.")
    if conditions.get("external_retrieval") is not False or conditions.get("tools_disabled") is not True:
        errors.append("Confirmatory conditions must disable retrieval and tools.")
    if conditions.get("semantic_retries") != 0:
        errors.append("Confirmatory conditions must use zero semantic retries.")

    failure_policy = protocol.get("failure_policy", {})
    expected_failure_policy = {
        "transport_retries": "exact_request_only",
        "semantic_retries": 0,
        "parse_or_schema_failure": "score_incorrect",
        "missing_confirmatory_pair": "block_analysis",
        "incomplete_matrix": "block_release",
    }
    for field, expected in expected_failure_policy.items():
        if failure_policy.get(field) != expected:
            errors.append(f"protocol.failure_policy.{field} must be {expected!r}.")

    few_shot = protocol.get("few_shot", {})
    if few_shot.get("support_bank_capacity") != {"A_BOX": 16, "T_BOX": 16}:
        errors.append("Few-shot support-bank capacity must be 16 per locus.")
    if few_shot.get("default_example_counts") != {
        "a_box_repair": 4,
        "t_box_repair": 4,
        "track_diagnosis": 2,
    }:
        errors.append("Few-shot default example counts must be A-box=4, T-box=4, diagnosis=2.")
    if few_shot.get("selection") != "deterministic_prefix":
        errors.append("Few-shot examples must use deterministic support-bank prefixes.")

    audit = protocol.get("audit", {})
    if audit.get("construct_review", {}).get("sample_size") != 450:
        errors.append("The label-hidden construct review must contain 450 cases.")
    if audit.get("temporal_review", {}).get("sample_size") != 50:
        errors.append("The temporal prompt review must contain 50 cases.")
    if audit.get("selection_eligible_disposition") != "include":
        errors.append("Only final disposition include may be selection eligible.")
    if audit.get("unresolved_systemic_findings_allowed") is not False:
        errors.append("Unresolved systemic findings must block release.")

    populations = selection.get("populations", {})
    expected_population_sizes = {"reserve-1440": 1440, "main-1200": 1200, "azure-600": 600}
    for name, expected_size in expected_population_sizes.items():
        actual_size = _population_size(populations.get(name))
        if actual_size != expected_size:
            errors.append(f"Population {name} must contain {expected_size} requested cases, not {actual_size!r}.")
    reserve = populations.get("reserve-1440", {})
    main = populations.get("main-1200", {})
    api = populations.get("azure-600", {})
    for stratum in ("IC-L", "IC-G", "IC-E-elim", "TBOX"):
        if isinstance(reserve.get(stratum), int) and isinstance(main.get(stratum), int):
            if reserve[stratum] < main[stratum]:
                errors.append(f"Reserve quota for {stratum} is smaller than the main quota.")
        if isinstance(main.get(stratum), int) and isinstance(api.get(stratum), int):
            if api[stratum] > main[stratum]:
                errors.append(f"Azure quota for {stratum} is not nested within the main quota.")
    tbox_target = selection.get("tbox_main_target", {})
    tbox_target_total = sum(
        value for key, value in tbox_target.items() if key != "allocation_role" and isinstance(value, int)
    )
    if tbox_target_total != main.get("TBOX"):
        errors.append("T-box main allocation targets must sum to the main TBOX quota.")
    prompt_gate = selection.get("reserve_prompt_gate", {})
    if prompt_gate.get("temporal_review_sample_size") != audit.get("temporal_review", {}).get("sample_size"):
        errors.append("Reserve and audit temporal-review sample sizes disagree.")
    if prompt_gate.get("finalize_only_after_all_failures_removed") is not True:
        errors.append("Selection must remove all prompt failures before finalization.")
    if selection.get("expansion", {}).get("tbox_underfill_policy") != "transfer_to_abox_by_largest_remainder":
        errors.append("T-box underfill must use declared largest-remainder transfer to A-box.")

    analysis_contrast_ids = [row.get("contrast_id") for row in analysis.get("primary_contrasts", []) if isinstance(row, dict)]
    if set(analysis_contrast_ids) != EXPECTED_CONTRASTS or _duplicates(analysis_contrast_ids):
        errors.append("Analysis must define the four unique predeclared factorial contrasts.")
    inference = analysis.get("inference", {})
    bootstrap = inference.get("bootstrap", {})
    if bootstrap != {
        "method": "percentile_cluster_bootstrap",
        "samples": 5000,
        "confidence_level": 0.95,
        "seed": 13,
    }:
        errors.append("Analysis bootstrap must be the predeclared 5,000-sample seed-13 95% percentile cluster bootstrap.")
    if inference.get("paired_binary_test") != "exact_mcnemar":
        errors.append("Paired binary comparisons must use exact McNemar tests.")
    if inference.get("multiplicity_correction") != "holm":
        errors.append("Primary contrast p-values must use Holm correction.")
    if analysis.get("pool_models") is not False:
        errors.append("Confirmatory analysis must report models separately rather than pooling them.")
    if analysis.get("failure_handling", {}).get("missing_confirmatory_pair") != "block_analysis":
        errors.append("Missing confirmatory pairs must block analysis.")
    if analysis.get("azure_calibration", {}).get("role") != "paired_calibration_reference":
        errors.append("Azure must remain a paired calibration reference.")

    generation_identity = models.get("generation_identity", [])
    if set(generation_identity) != EXPECTED_GENERATION_IDENTITY or _duplicates(generation_identity):
        errors.append("Generation identity fields are incomplete or duplicated.")
    model_rows = models.get("models", [])
    model_ids = [row.get("model_id") for row in model_rows if isinstance(row, dict)]
    duplicate_model_ids = _duplicates(model_ids)
    if duplicate_model_ids:
        errors.append(f"Duplicate model_id values: {duplicate_model_ids}")
    models_by_id = {row.get("model_id"): row for row in model_rows if isinstance(row, dict)}
    missing_required_models = sorted(set(REQUIRED_MODEL_CONFIGURATION) - set(models_by_id))
    if missing_required_models:
        errors.append(f"Required paper models are missing: {missing_required_models}")
    for model_id, expected_fields in REQUIRED_MODEL_CONFIGURATION.items():
        row = models_by_id.get(model_id)
        if not isinstance(row, dict):
            continue
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                errors.append(f"Model {model_id} field {field} must be {expected!r}.")
    task_count = len(dimensions.get("tasks", []))
    regime_count = len(dimensions.get("prompt_regimes", []))
    context_count = len(dimensions.get("context_bundles", []))
    for row in model_rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("model_id") or "<missing>")
        population_name = row.get("population")
        population_size = _population_size(populations.get(population_name))
        if population_size is None:
            errors.append(f"Model {model_id} references unknown or invalid population {population_name!r}.")
            continue
        expected_calls = population_size * task_count * regime_count * context_count
        workloads[model_id] = expected_calls
        if row.get("expected_calls") != expected_calls:
            errors.append(
                f"Model {model_id} expected_calls must be {expected_calls}, not {row.get('expected_calls')!r}."
            )
        revision = row.get("model_revision")
        revision_status = row.get("revision_status")
        if revision_status == "resolved" and row.get("provider") == "ollama":
            if not isinstance(revision, str) or not SHA256_RE.fullmatch(revision):
                errors.append(f"Resolved Ollama model {model_id} must have a full lowercase SHA-256 revision.")
        if revision_status == "resolved" and row.get("provider") == "azure":
            if not isinstance(revision, str) or not revision.strip() or revision.lower() in {"unknown", "latest"}:
                errors.append(f"Resolved Azure model {model_id} must have an immutable deployment or snapshot revision.")
        if revision_status == "unresolved" and revision is not None:
            errors.append(f"Unresolved model {model_id} must use null model_revision.")

    return errors, workloads


def _scope_paths(repo_root: Path, protocol: dict[str, Any]) -> list[Path]:
    scope = protocol.get("freeze_policy", {}).get("scope", {})
    relative_paths: set[str] = set()
    for raw_path in scope.get("files", []):
        if isinstance(raw_path, str):
            relative_paths.add(raw_path)
    for raw_directory in scope.get("directories", []):
        if not isinstance(raw_directory, str):
            continue
        directory = (repo_root / raw_directory).resolve()
        try:
            directory.relative_to(repo_root)
        except ValueError as exc:
            raise MethodologyError(f"Freeze-scope directory escapes repository: {raw_directory}") from exc
        if not directory.is_dir():
            raise MethodologyError(f"Freeze-scope directory is missing: {raw_directory}")
        for path in directory.rglob("*"):
            relative = path.relative_to(repo_root)
            if (
                path.is_file()
                and "__pycache__" not in relative.parts
                and not any(part.endswith(".egg-info") for part in relative.parts)
                and path.suffix not in {".pyc", ".pyo"}
            ):
                relative_paths.add(relative.as_posix())
    paths: list[Path] = []
    for relative_path in sorted(relative_paths):
        path = (repo_root / relative_path).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise MethodologyError(f"Freeze-scope file escapes repository: {relative_path}") from exc
        if not path.is_file():
            raise MethodologyError(f"Freeze-scope file is missing: {relative_path}")
        paths.append(path)
    if not paths:
        raise MethodologyError("Freeze scope is empty.")
    return paths


def hash_freeze_scope(repo_root: Path, protocol: dict[str, Any]) -> tuple[dict[str, str], str]:
    file_hashes: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for path in _scope_paths(repo_root.resolve(), protocol):
        relative = path.relative_to(repo_root.resolve()).as_posix()
        digest = sha256_file(path)
        file_hashes[relative] = digest
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
    return file_hashes, aggregate.hexdigest()


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_lines = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MethodologyError(f"Could not inspect Git state: {exc}") from exc
    return {"revision": revision, "clean": not status_lines, "changed_paths": status_lines}


def _lock_validation(
    repo_root: Path,
    protocol: dict[str, Any],
    file_hashes: dict[str, str],
    scope_hash: str,
) -> tuple[bool, list[str], dict[str, Any] | None]:
    lock_path = repo_root / METHODOLOGY_LOCK_PATH
    if not lock_path.is_file():
        return False, [f"Frozen methodology lock is missing: {METHODOLOGY_LOCK_PATH}"], None
    try:
        lock = _read_json(lock_path)
    except MethodologyError as exc:
        return False, [str(exc)], None
    errors: list[str] = []
    if lock.get("manifest_type") != "paper_methodology_lock" or lock.get("manifest_version") != 1:
        errors.append("Methodology lock has an unsupported type or version.")
    if lock.get("protocol_id") != protocol.get("protocol_id"):
        errors.append("Methodology lock protocol_id does not match the protocol.")
    if lock.get("freeze_scope_sha256") != scope_hash:
        errors.append("Methodology freeze-scope hash differs from the lock.")
    if lock.get("files") != file_hashes:
        errors.append("Methodology freeze-scope file hashes differ from the lock.")
    revision = lock.get("source_git_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        errors.append("Methodology lock source_git_revision is not a full Git commit hash.")
    else:
        try:
            subprocess.run(
                ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
                cwd=repo_root,
                check=True,
                capture_output=True,
            )
            scoped_paths = sorted(file_hashes)
            scoped_diff = subprocess.run(
                ["git", "diff", "--quiet", revision, "--", *scoped_paths],
                cwd=repo_root,
                check=False,
            )
            if scoped_diff.returncode != 0:
                errors.append("Freeze-scoped files differ from the lock's source Git revision.")
        except (OSError, subprocess.CalledProcessError):
            errors.append("Methodology lock source_git_revision is not available in this repository.")
    return not errors, errors, lock


def check_methodology(repo_root: Path | str = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    errors: list[str] = []
    try:
        bundle = load_methodology_bundle(root)
    except MethodologyError as exc:
        return {
            "manifest_type": "methodology_check",
            "manifest_version": 1,
            "valid": False,
            "freeze_ready": False,
            "errors": [str(exc)],
            "blockers": ["Methodology files could not be loaded."],
        }
    errors.extend(_schema_errors(root, bundle))
    cross_errors, workloads = validate_methodology_bundle(bundle)
    errors.extend(cross_errors)
    try:
        file_hashes, scope_hash = hash_freeze_scope(root, bundle["protocol"])
    except MethodologyError as exc:
        errors.append(str(exc))
        file_hashes, scope_hash = {}, ""
    try:
        git = _git_state(root)
    except MethodologyError as exc:
        errors.append(str(exc))
        git = {"revision": None, "clean": False, "changed_paths": []}

    statuses = {name: config.get("status") for name, config in bundle.items()}
    unresolved_models = [
        row.get("model_id")
        for row in bundle["models"].get("models", [])
        if isinstance(row, dict) and row.get("revision_status") != "resolved"
    ]
    lock_valid, lock_errors, lock = _lock_validation(root, bundle["protocol"], file_hashes, scope_hash)
    if lock is not None and not lock_valid:
        errors.extend(lock_errors)
    blockers: list[str] = []
    if any(status != "frozen" for status in statuses.values()):
        blockers.append("All methodology components must have status frozen.")
    if unresolved_models:
        blockers.append(f"Unresolved model revisions: {', '.join(str(item) for item in unresolved_models)}")
    if not git.get("clean"):
        blockers.append("Git worktree must be clean for final freeze.")
    if not lock_valid:
        blockers.extend(lock_errors)
    valid = not errors
    freeze_ready = valid and not blockers
    return {
        "manifest_type": "methodology_check",
        "manifest_version": 1,
        "protocol_id": bundle["protocol"].get("protocol_id"),
        "component_status": statuses,
        "valid": valid,
        "freeze_ready": freeze_ready,
        "errors": errors,
        "blockers": blockers,
        "unresolved_model_revisions": unresolved_models,
        "workloads": workloads,
        "freeze_scope_sha256": scope_hash,
        "files": file_hashes,
        "git": git,
        "lock": {
            "path": METHODOLOGY_LOCK_PATH,
            "present": lock is not None,
            "valid": lock_valid,
            "source_git_revision": lock.get("source_git_revision") if isinstance(lock, dict) else None,
        },
    }


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def create_methodology_lock(repo_root: Path | str = ".", output_path: Path | str | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    report = check_methodology(root)
    allowed_blocker = f"Frozen methodology lock is missing: {METHODOLOGY_LOCK_PATH}"
    non_lock_blockers = [blocker for blocker in report.get("blockers", []) if blocker != allowed_blocker]
    if report.get("errors") or non_lock_blockers:
        details = [*report.get("errors", []), *non_lock_blockers]
        raise MethodologyError("Methodology cannot be frozen: " + "; ".join(details))
    destination = Path(output_path) if output_path is not None else root / METHODOLOGY_LOCK_PATH
    if not destination.is_absolute():
        destination = root / destination
    if destination.exists():
        raise FileExistsError(f"Methodology lock already exists: {destination}")
    lock = {
        "manifest_type": "paper_methodology_lock",
        "manifest_version": 1,
        "protocol_id": report["protocol_id"],
        "source_git_revision": report["git"]["revision"],
        "freeze_scope_sha256": report["freeze_scope_sha256"],
        "files": report["files"],
    }
    _write_json_atomic(destination, lock)
    return lock


def require_frozen_methodology(repo_root: Path | str = ".") -> dict[str, Any]:
    report = check_methodology(repo_root)
    if not report.get("freeze_ready"):
        details = [*report.get("errors", []), *report.get("blockers", [])]
        raise MethodologyError("A valid frozen methodology lock is required: " + "; ".join(details))
    return report
