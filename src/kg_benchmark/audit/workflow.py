from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import fastjsonschema
from jsonschema import Draft202012Validator

from automated_audit_codex import (
    DISPOSITIONS_FILENAME,
    FINAL_JSON_FILENAME,
    REVIEWS_FILENAME,
    RUN_FILENAME,
    RunCommand,
    finalize_audit,
    run_codex_reviews,
)
from automated_consistency_audit import WorldStateLookup, run_audit
from guardian.reasoning import (
    build_prompt_bundle,
    build_track_diagnosis_prompt_bundle,
    prompt_visible_case_id,
)
from kg_benchmark.dataset.release import sha256_file
from kg_benchmark.selection.extensible import stratum_for_record

WORKFLOW_VERSION = 1
SAMPLE_ALGORITHM = "sha256_rank_by_case_id_v1"
RENDERER_VERSION = 1
WORKFLOW_FILENAME = "workflow.json"
CONSTRUCT_SAMPLE_FILENAME = "construct-sample.csv"
RENDERED_PROMPTS_FILENAME = "rendered-prompts.jsonl"
RENDER_SUMMARY_FILENAME = "render-summary.json"
CANONICAL_DISPOSITIONS_FILENAME = "dispositions.jsonl"
CANONICAL_SUMMARY_FILENAME = "summary.json"
DETERMINISTIC_DIRNAME = "deterministic"
PROMPT_SOURCES = (
    "paper/prompts/abox-repair.txt",
    "paper/prompts/tbox-taxonomy-patch.txt",
    "paper/prompts/track-diagnosis.txt",
)
RESPONSE_SCHEMAS = (
    "schemas/abox-response.schema.json",
    "schemas/tbox-response.schema.json",
    "schemas/track-diagnosis-response.schema.json",
)


class AuditWorkflowError(ValueError):
    """Raised when a canonical audit phase cannot safely proceed."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_root, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty}


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _portable_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _artifact(path: Path, repo_root: Path, *, records: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise AuditWorkflowError(f"Required audit artifact is missing: {path}")
    value: dict[str, Any] = {
        "path": _portable_path(path, repo_root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if records is not None:
        value["records"] = records
    return value


def _verify_artifact(record: Any, repo_root: Path, role: str) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise AuditWorkflowError(f"Workflow does not bind the {role} artifact.")
    path = _resolve_path(record["path"], repo_root)
    if not path.is_file():
        raise AuditWorkflowError(f"Bound {role} artifact is missing: {path}")
    if path.stat().st_size != record.get("size_bytes") or sha256_file(path) != record.get("sha256"):
        raise AuditWorkflowError(f"Bound {role} artifact changed after its phase completed: {path}")
    return path


def _verify_requested_path(record: Any, requested: Path | None, repo_root: Path, role: str) -> None:
    if requested is None:
        if record is not None:
            raise AuditWorkflowError(f"Existing workflow binds {role}, but the resumed command omitted it.")
        return
    bound = _verify_artifact(record, repo_root, role)
    if bound.resolve() != requested.resolve():
        raise AuditWorkflowError(
            f"Existing workflow binds {role} to {bound}, not the requested path {requested.resolve()}."
        )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditWorkflowError(f"Could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditWorkflowError(f"Expected a JSON object in {path}.")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditWorkflowError(f"Invalid JSON at {path}:{line_number}.") from exc
            if not isinstance(value, dict):
                raise AuditWorkflowError(f"Expected an object at {path}:{line_number}.")
            yield value


def _load_protocol(protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _load_json(protocol_path)
    audit = protocol.get("audit")
    conditions = protocol.get("conditions")
    if not isinstance(audit, dict) or not isinstance(conditions, dict):
        raise AuditWorkflowError("Protocol must define audit and conditions objects.")
    construct = audit.get("construct_review")
    temporal = audit.get("temporal_review")
    if not isinstance(construct, dict) or not isinstance(temporal, dict):
        raise AuditWorkflowError("Protocol must define construct and temporal audit review settings.")
    if construct.get("selection") != SAMPLE_ALGORITHM:
        raise AuditWorkflowError(f"Protocol construct selection must be {SAMPLE_ALGORITHM}.")
    construct_seed = construct.get("seed")
    temporal_seed = temporal.get("seed")
    if construct_seed != temporal_seed:
        raise AuditWorkflowError("Canonical audit currently requires matching construct and temporal seeds.")
    context_bundles = conditions.get("context_bundles")
    if context_bundles != ["logic_only", "local_graph"]:
        raise AuditWorkflowError("Canonical audit rendering requires protocol contexts logic_only and local_graph.")
    settings = {
        "construct_review_size": construct.get("sample_size"),
        "temporal_review_size": temporal.get("sample_size"),
        "seed": construct_seed,
        "reviewer_model": construct.get("reviewer_model"),
        "context_bundles": context_bundles,
    }
    if not isinstance(settings["construct_review_size"], int) or settings["construct_review_size"] < 1:
        raise AuditWorkflowError("Protocol construct review sample_size must be positive.")
    if not isinstance(settings["temporal_review_size"], int) or settings["temporal_review_size"] < 1:
        raise AuditWorkflowError("Protocol temporal review sample_size must be positive.")
    if not isinstance(settings["seed"], int) or not isinstance(settings["reviewer_model"], str):
        raise AuditWorkflowError("Protocol audit seed and reviewer model must be explicit.")
    return protocol, settings


def _sample_rank(seed: int, case_id: str) -> str:
    return hashlib.sha256(f"{seed}|construct_review|{case_id}".encode("utf-8")).hexdigest()


def generate_construct_sample(
    *, cases_path: Path, output_path: Path, sample_size: int, seed: int
) -> dict[str, Any]:
    ranked: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    population_counts: Counter[str] = Counter()
    for record in _iter_jsonl(cases_path):
        case_id = record.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise AuditWorkflowError("Every case must have a non-empty id before construct sampling.")
        if case_id in seen:
            raise AuditWorkflowError(f"Duplicate case id during construct sampling: {case_id}")
        seen.add(case_id)
        stratum = stratum_for_record(record) or "OTHER"
        population_counts[stratum] += 1
        ranked.append((_sample_rank(seed, case_id), case_id, stratum))
    if len(ranked) < sample_size:
        raise AuditWorkflowError(
            f"Construct review requires {sample_size} unique cases, but the dataset contains {len(ranked)}."
        )
    ranked.sort(key=lambda row: (row[0], row[1]))
    selected = ranked[:sample_size]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["case_id"])
            writer.writerows([[case_id] for _, case_id, _ in selected])
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "algorithm": SAMPLE_ALGORITHM,
        "seed": seed,
        "label_hidden": True,
        "population_cases": len(ranked),
        "sample_cases": len(selected),
        "population_by_stratum": dict(sorted(population_counts.items())),
        "sample_by_stratum": dict(sorted(Counter(row[2] for row in selected).items())),
    }


def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    return hashlib.sha256((system_prompt + "\n" + user_prompt).encode("utf-8")).hexdigest()


def _matrix_id(case_id: str, task: str, context_bundle: str) -> str:
    digest = hashlib.sha256(f"audit|{case_id}|{task}|zero_shot|{context_bundle}".encode("utf-8")).hexdigest()
    return f"audit_{digest[:20]}"


def render_audit_prompts(
    *,
    cases_path: Path,
    world_state_path: Path,
    output_path: Path,
    cache_dir: Path,
    context_bundles: list[str],
    prompt_schema_path: Path,
) -> dict[str, Any]:
    prompt_schema = _load_json(prompt_schema_path)
    Draft202012Validator.check_schema(prompt_schema)
    validate_row = fastjsonschema.compile(prompt_schema)
    case_count = 0
    prompt_count = 0
    missing_world_state = 0
    seen_case_ids: set[str] = set()
    counts_by_task: Counter[str] = Counter()
    counts_by_context: Counter[str] = Counter()
    counts_by_track: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with WorldStateLookup(world_state_path, cache_dir) as world_lookup, temporary.open(
            "w", encoding="utf-8"
        ) as output:
            for record in _iter_jsonl(cases_path):
                case_id = record.get("id")
                track = record.get("track")
                if not isinstance(case_id, str) or not case_id:
                    raise AuditWorkflowError("Every rendered case must have a non-empty id.")
                if case_id in seen_case_ids:
                    raise AuditWorkflowError(f"Duplicate case id during prompt rendering: {case_id}")
                if track not in {"A_BOX", "T_BOX"}:
                    raise AuditWorkflowError(f"Case {case_id} has unsupported track {track!r}.")
                seen_case_ids.add(case_id)
                case_count += 1
                counts_by_track[str(track)] += 1
                world_state = world_lookup.get(case_id)
                if world_state is None:
                    missing_world_state += 1
                visible_case_id = prompt_visible_case_id(case_id)
                task_builders: list[tuple[str, Callable[..., Any]]] = [
                    ("a_box_repair" if track == "A_BOX" else "t_box_taxonomy_patch", build_prompt_bundle),
                    ("track_diagnosis", build_track_diagnosis_prompt_bundle),
                ]
                for task, builder in task_builders:
                    for context_bundle in context_bundles:
                        bundle = builder(
                            record,
                            world_state,
                            context_bundle,
                            visible_case_id=visible_case_id,
                        )
                        if case_id in bundle.prompt or case_id in bundle.system_prompt:
                            raise AuditWorkflowError(f"Rendered prompt exposed raw case id {case_id}.")
                        row = {
                            "matrix_id": _matrix_id(case_id, task, context_bundle),
                            "case_id": case_id,
                            "visible_case_id": visible_case_id,
                            "task": task,
                            "prompt_regime": "zero_shot",
                            "context_bundle": context_bundle,
                            "historical_track": track,
                            "prompt_name": bundle.prompt_name,
                            "system_prompt": bundle.system_prompt,
                            "user_prompt": bundle.prompt,
                            "prompt_sha256": _prompt_hash(bundle.system_prompt, bundle.prompt),
                            "response_format": bundle.response_format,
                        }
                        validate_row(row)
                        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        prompt_count += 1
                        counts_by_task[task] += 1
                        counts_by_context[context_bundle] += 1
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    expected = case_count * 2 * len(context_bundles)
    if prompt_count != expected:
        raise AuditWorkflowError(f"Canonical renderer produced {prompt_count} prompts; expected {expected}.")
    return {
        "renderer_version": RENDERER_VERSION,
        "case_count": case_count,
        "rendered_prompt_count": prompt_count,
        "expected_prompt_count": expected,
        "complete": prompt_count == expected,
        "missing_world_state_cases": missing_world_state,
        "prompt_regime": "zero_shot",
        "tasks": ["repair_proposal", "track_diagnosis"],
        "context_bundles": context_bundles,
        "counts_by_task": dict(sorted(counts_by_task.items())),
        "counts_by_context": dict(sorted(counts_by_context.items())),
        "counts_by_track": dict(sorted(counts_by_track.items())),
    }


def _initial_state(
    *,
    repo_root: Path,
    protocol_path: Path,
    settings: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "manifest_type": "canonical_audit_workflow",
        "manifest_version": WORKFLOW_VERSION,
        "created_at_utc": now,
        "updated_at_utc": now,
        "git": _git_state(repo_root),
        "protocol": _artifact(protocol_path, repo_root),
        "settings": settings,
        "inputs": inputs,
        "phases": {},
    }


def _workflow_path(work_dir: Path) -> Path:
    return work_dir / WORKFLOW_FILENAME


def _load_workflow(work_dir: Path, repo_root: Path) -> dict[str, Any]:
    path = _workflow_path(work_dir)
    state = _load_json(path)
    schema_path = repo_root / "schemas" / "audit-workflow.schema.json"
    Draft202012Validator(_load_json(schema_path)).validate(state)
    return state


def _save_workflow(work_dir: Path, repo_root: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    schema_path = repo_root / "schemas" / "audit-workflow.schema.json"
    Draft202012Validator(_load_json(schema_path)).validate(state)
    _write_json_atomic(_workflow_path(work_dir), state)


def _verify_role_map(role_map: Any, repo_root: Path, prefix: str) -> dict[str, Path]:
    if not isinstance(role_map, dict):
        raise AuditWorkflowError(f"Workflow phase {prefix} has no artifact bindings.")
    return {role: _verify_artifact(record, repo_root, f"{prefix}.{role}") for role, record in role_map.items()}


def _verify_prepare(state: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    _verify_artifact(state.get("protocol"), repo_root, "protocol")
    _verify_role_map(state.get("inputs"), repo_root, "inputs")
    phase = state.get("phases", {}).get("prepare")
    if not isinstance(phase, dict) or phase.get("status") != "complete":
        raise AuditWorkflowError("Audit prepare phase is not complete.")
    return _verify_role_map(phase.get("artifacts"), repo_root, "prepare")


def prepare_audit(
    *,
    cases_path: Path,
    world_state_path: Path,
    stage4_schema_path: Path,
    protocol_path: Path,
    work_dir: Path,
    repo_root: Path = Path("."),
    stage2_path: Path | None = None,
    lineage_manifest_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    work_dir = work_dir.resolve()
    workflow_path = _workflow_path(work_dir)
    if workflow_path.exists():
        state = _load_workflow(work_dir, repo_root)
        _verify_prepare(state, repo_root)
        _verify_requested_path(state.get("protocol"), protocol_path, repo_root, "protocol")
        for role, requested in {
            "cases": cases_path,
            "world_state": world_state_path,
            "stage4_schema": stage4_schema_path,
            "stage2": stage2_path,
            "lineage_manifest": lineage_manifest_path,
        }.items():
            _verify_requested_path(state.get("inputs", {}).get(role), requested, repo_root, f"inputs.{role}")
        return state
    existing_names = {path.name for path in work_dir.iterdir()} if work_dir.exists() else set()
    unexpected_names = existing_names - {"classifier-summary.json"}
    if unexpected_names:
        raise AuditWorkflowError(
            f"Audit work directory has no resumable workflow and contains unexpected entries: "
            f"{', '.join(sorted(unexpected_names))}. Move or remove them before preparing the canonical audit."
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    protocol, settings = _load_protocol(protocol_path)
    del protocol
    required_paths = {
        "cases": cases_path,
        "world_state": world_state_path,
        "stage4_schema": stage4_schema_path,
        "rendered_prompt_schema": repo_root / "schemas" / "rendered-audit-prompt.schema.json",
        "audit_disposition_schema": repo_root / "schemas" / "audit-disposition.schema.json",
        "audit_summary_schema": repo_root / "schemas" / "audit-summary.schema.json",
        "automated_audit_schema": repo_root / "schemas" / "automated-audit.schema.json",
        "automated_review_schema": repo_root / "schemas" / "automated-audit-review.schema.json",
    }
    if stage2_path is not None:
        required_paths["stage2"] = stage2_path
    if lineage_manifest_path is not None:
        required_paths["lineage_manifest"] = lineage_manifest_path
    classifier_summary_path = work_dir / "classifier-summary.json"
    if classifier_summary_path.is_file():
        required_paths["classifier_summary"] = classifier_summary_path
    for relative in (*PROMPT_SOURCES, *RESPONSE_SCHEMAS):
        required_paths[relative.replace("/", "_")] = repo_root / relative
    inputs = {role: _artifact(path, repo_root) for role, path in required_paths.items()}
    state = _initial_state(
        repo_root=repo_root,
        protocol_path=protocol_path,
        settings=settings,
        inputs=inputs,
    )
    sample_path = work_dir / CONSTRUCT_SAMPLE_FILENAME
    prompts_path = work_dir / RENDERED_PROMPTS_FILENAME
    render_summary_path = work_dir / RENDER_SUMMARY_FILENAME
    sample_summary = generate_construct_sample(
        cases_path=cases_path,
        output_path=sample_path,
        sample_size=settings["construct_review_size"],
        seed=settings["seed"],
    )
    render_summary = render_audit_prompts(
        cases_path=cases_path,
        world_state_path=world_state_path,
        output_path=prompts_path,
        cache_dir=work_dir / "cache",
        context_bundles=settings["context_bundles"],
        prompt_schema_path=repo_root / "schemas" / "rendered-audit-prompt.schema.json",
    )
    render_summary.update(
        {
            "report_type": "canonical_audit_prompt_render",
            "report_version": 1,
            "created_at_utc": _utc_now(),
            "inputs": {
                "cases": inputs["cases"],
                "world_state": inputs["world_state"],
                "prompt_schema": inputs["rendered_prompt_schema"],
                "prompt_sources": {
                    role: value for role, value in inputs.items() if role.startswith("paper_prompts_")
                },
                "response_schemas": {
                    role: value for role, value in inputs.items() if role.startswith("schemas_") and "response" in role
                },
            },
            "construct_sample": sample_summary,
        }
    )
    _write_json_atomic(render_summary_path, render_summary)
    state["phases"]["prepare"] = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "artifacts": {
            "construct_sample": _artifact(sample_path, repo_root, records=settings["construct_review_size"]),
            "rendered_prompts": _artifact(prompts_path, repo_root, records=render_summary["rendered_prompt_count"]),
            "render_summary": _artifact(render_summary_path, repo_root),
        },
        "counts": {
            "cases": render_summary["case_count"],
            "construct_sample": settings["construct_review_size"],
            "rendered_prompts": render_summary["rendered_prompt_count"],
        },
    }
    _save_workflow(work_dir, repo_root, state)
    return state


def run_deterministic_phase(
    *, work_dir: Path, repo_root: Path = Path("."), cache_dir: Path | None = None
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    work_dir = work_dir.resolve()
    state = _load_workflow(work_dir, repo_root)
    prepared = _verify_prepare(state, repo_root)
    existing = state["phases"].get("deterministic")
    if isinstance(existing, dict) and existing.get("status") == "complete":
        _verify_role_map(existing.get("artifacts"), repo_root, "deterministic")
        return state
    inputs = {role: _verify_artifact(record, repo_root, f"inputs.{role}") for role, record in state["inputs"].items()}
    output_dir = work_dir / DETERMINISTIC_DIRNAME
    manifest = run_audit(
        classified_benchmark_path=inputs["cases"],
        world_state_path=inputs["world_state"],
        stage4_schema_path=inputs["stage4_schema"],
        construct_sample_path=prepared["construct_sample"],
        rendered_prompts_path=prepared["rendered_prompts"],
        render_summary_path=prepared["render_summary"],
        output_dir=output_dir,
        construct_review_size=state["settings"]["construct_review_size"],
        temporal_review_size=state["settings"]["temporal_review_size"],
        seed=state["settings"]["seed"],
        stage2_path=inputs.get("stage2"),
        lineage_manifest_path=inputs.get("lineage_manifest"),
        cache_dir=cache_dir or work_dir / "cache" / "deterministic",
    )
    manifest_path = output_dir / "manifest.json"
    artifacts = {"manifest": _artifact(manifest_path, repo_root)}
    for role, record in manifest["artifacts"].items():
        artifacts[role] = _artifact(output_dir / record["path"], repo_root)
    state["phases"]["deterministic"] = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "passed": bool(manifest["validation"]["passed"]),
        "artifacts": artifacts,
        "validation": manifest["validation"],
    }
    _save_workflow(work_dir, repo_root, state)
    return state


def _verify_deterministic(state: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    phase = state.get("phases", {}).get("deterministic")
    if not isinstance(phase, dict) or phase.get("status") != "complete":
        raise AuditWorkflowError("Deterministic audit phase is not complete.")
    paths = _verify_role_map(phase.get("artifacts"), repo_root, "deterministic")
    if not phase.get("passed"):
        raise AuditWorkflowError("Deterministic audit gates failed; remediate and restart the audit before Codex review.")
    return paths


def run_review_phase(
    *,
    work_dir: Path,
    repo_root: Path = Path("."),
    batch_size: int = 10,
    workers: int = 1,
    retries: int = 2,
    timeout_seconds: float = 600,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    work_dir = work_dir.resolve()
    state = _load_workflow(work_dir, repo_root)
    _verify_prepare(state, repo_root)
    deterministic = _verify_deterministic(state, repo_root)
    existing = state["phases"].get("review")
    if isinstance(existing, dict) and existing.get("status") == "complete":
        _verify_role_map(existing.get("artifacts"), repo_root, "review")
        return state
    report = run_codex_reviews(
        manifest_path=deterministic["manifest"],
        model=state["settings"]["reviewer_model"],
        batch_size=batch_size,
        workers=workers,
        retries=retries,
        timeout_seconds=timeout_seconds,
        run_command=run_command,
    )
    if report.get("status") != "complete" or report.get("codex", {}).get("model") != state["settings"]["reviewer_model"]:
        raise AuditWorkflowError("Codex review did not complete with the protocol-bound reviewer model.")
    output_dir = deterministic["manifest"].parent
    artifacts = {
        "review_run": _artifact(output_dir / RUN_FILENAME, repo_root),
        "reviews": _artifact(output_dir / REVIEWS_FILENAME, repo_root, records=report["reviews"]["count"]),
        "review_output_schema": _artifact(output_dir / ".codex_review_output.schema.json", repo_root),
    }
    state["phases"]["review"] = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "reviewer": {
            "interface": "codex_cli",
            "model": report["codex"]["model"],
            "cli_version": report["codex"]["version"],
        },
        "execution": report["execution"],
        "artifacts": artifacts,
    }
    _save_workflow(work_dir, repo_root, state)
    return state


def _verify_review(state: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    phase = state.get("phases", {}).get("review")
    if not isinstance(phase, dict) or phase.get("status") != "complete":
        raise AuditWorkflowError("Codex review phase is not complete.")
    paths = _verify_role_map(phase.get("artifacts"), repo_root, "review")
    run_report = _load_json(paths["review_run"])
    if run_report.get("status") != "complete":
        raise AuditWorkflowError("Bound Codex review report is not complete.")
    reviewer = phase.get("reviewer", {})
    codex = run_report.get("codex", {})
    if reviewer.get("model") != codex.get("model") or reviewer.get("cli_version") != codex.get("version"):
        raise AuditWorkflowError("Codex reviewer provenance does not match the bound review report.")
    return paths


def _case_ids(path: Path) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for record in _iter_jsonl(path):
        case_id = record.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise AuditWorkflowError("Finalization requires unique non-empty case IDs in Stage 4.")
        seen.add(case_id)
        values.append(case_id)
    return values


def _validate_dispositions(path: Path, cases_path: Path, schema_path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    counts: Counter[str] = Counter()
    for row in _iter_jsonl(path):
        validator.validate(row)
        rows.append(row)
        ids.append(row["case_id"])
        counts[row["disposition"]] += 1
    case_ids = _case_ids(cases_path)
    if len(ids) != len(set(ids)) or set(ids) != set(case_ids) or len(ids) != len(case_ids):
        raise AuditWorkflowError("Final disposition coverage must exactly and uniquely equal all Stage 4 case IDs.")
    rows.sort(key=lambda row: row["case_id"])
    return rows, counts


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _audit_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]["by_disposition"]
    provenance = summary["provenance"]
    lines = [
        "# Dataset Audit",
        "",
        "This is the release-facing report for the canonical whole-dataset audit. Deterministic checks are",
        "exhaustive; Codex review is a fixed label-hidden error-discovery sample and is not ground-truth certification.",
        "AI findings never change benchmark labels. Only final disposition `include` is selection-eligible.",
        "",
        "## Coverage and dispositions",
        "",
        f"- Dataset cases: {summary['counts']['cases']}",
        f"- Canonical prompts scanned: {summary['counts']['rendered_prompts']}",
        f"- Construct packets reviewed: {summary['counts']['construct_reviews']}",
        f"- Temporal packets reviewed: {summary['counts']['temporal_reviews']}",
    ]
    for disposition in ("include", "diagnostic", "exclude_pending_rerender", "exclude"):
        lines.append(f"- `{disposition}`: {counts.get(disposition, 0)}")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Exact unique disposition coverage: {str(summary['validation']['complete_unique_disposition_coverage']).lower()}",
            f"- Deterministic gates passed: {str(summary['validation']['deterministic_gates_passed']).lower()}",
            f"- Automated temporal gate passed: {str(summary['validation']['temporal_gate_passed']).lower()}",
            f"- Unresolved systemic findings: {summary['validation']['unresolved_systemic_findings']}",
            "",
            "## Reviewer provenance",
            "",
            f"- Interface: `{summary['reviewer']['interface']}`",
            f"- Model: `{summary['reviewer']['model']}`",
            f"- Codex CLI: `{summary['reviewer']['cli_version']}`",
            f"- Reviews SHA-256: `{provenance['reviews']['sha256']}`",
            f"- Dispositions SHA-256: `{provenance['dispositions']['sha256']}`",
            "",
            "## Interpretation",
            "",
            "Diagnostic and excluded cases remain outside evaluation. A completed conservative disposition resolves",
            "a sampled concern operationally; it does not make the reviewer a semantic oracle. Population selection and",
            "final composition are recorded separately by the selection manifests.",
            "",
            "## Historical development audit",
            "",
            "The earlier `full_v1` audit is retained only as development evidence in",
            "`docs-technical/Development_History.md`; it does not certify or contribute cases to this release.",
            "",
        ]
    )
    return "\n".join(lines)


def run_finalize_phase(
    *, work_dir: Path, report_path: Path = Path("audit.md"), repo_root: Path = Path(".")
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    work_dir = work_dir.resolve()
    state = _load_workflow(work_dir, repo_root)
    _verify_prepare(state, repo_root)
    deterministic = _verify_deterministic(state, repo_root)
    review = _verify_review(state, repo_root)
    existing = state["phases"].get("finalize")
    if isinstance(existing, dict) and existing.get("status") == "complete":
        artifacts = _verify_role_map(existing.get("artifacts"), repo_root, "finalize")
        requested_report = report_path if report_path.is_absolute() else repo_root / report_path
        if artifacts["release_report"].resolve() != requested_report.resolve():
            raise AuditWorkflowError(
                f"Existing workflow report is {artifacts['release_report']}, not {requested_report.resolve()}."
            )
        return state
    final_report = finalize_audit(
        manifest_path=deterministic["manifest"],
        reviews_path=review["reviews"],
    )
    legacy_dispositions = deterministic["manifest"].parent / DISPOSITIONS_FILENAME
    cases_path = _verify_artifact(state["inputs"]["cases"], repo_root, "inputs.cases")
    disposition_schema = _verify_artifact(
        state["inputs"]["audit_disposition_schema"], repo_root, "inputs.audit_disposition_schema"
    )
    rows, counts = _validate_dispositions(legacy_dispositions, cases_path, disposition_schema)
    canonical_dispositions = work_dir / CANONICAL_DISPOSITIONS_FILENAME
    _write_jsonl_atomic(canonical_dispositions, rows)
    deterministic_summary = _load_json(deterministic["deterministic_summary"])
    temporal_audit = _load_json(deterministic["temporal_audit"])
    review_report = _load_json(review["review_run"])
    disposition_artifact = _artifact(canonical_dispositions, repo_root, records=len(rows))
    summary = {
        "report_type": "canonical_audit_summary",
        "report_version": 1,
        "created_at_utc": _utc_now(),
        "policy": final_report["policy"],
        "counts": {
            "cases": len(rows),
            "rendered_prompts": temporal_audit["counts"]["prompt_rows"],
            "construct_reviews": state["settings"]["construct_review_size"],
            "temporal_reviews": state["settings"]["temporal_review_size"],
            "reviews": final_report["counts"]["reviews"],
            "by_disposition": dict(sorted(counts.items())),
        },
        "validation": {
            "complete_unique_disposition_coverage": len(rows) == deterministic_summary["coverage"]["stage4_rows"],
            "deterministic_gates_passed": bool(state["phases"]["deterministic"]["passed"]),
            "temporal_gate_passed": bool(temporal_audit["passed_automated_gate"]),
            "review_complete": review_report.get("status") == "complete",
            "unresolved_systemic_findings": 0,
            "selection_eligible_disposition": "include",
        },
        "reviewer": state["phases"]["review"]["reviewer"],
        "provenance": {
            "protocol": state["protocol"],
            "cases": state["inputs"]["cases"],
            "world_state": state["inputs"]["world_state"],
            "stage2": state["inputs"].get("stage2"),
            "lineage_manifest": state["inputs"].get("lineage_manifest"),
            "stage4_schema": state["inputs"]["stage4_schema"],
            "construct_sample": state["phases"]["prepare"]["artifacts"]["construct_sample"],
            "rendered_prompts": state["phases"]["prepare"]["artifacts"]["rendered_prompts"],
            "render_summary": state["phases"]["prepare"]["artifacts"]["render_summary"],
            "deterministic_manifest": state["phases"]["deterministic"]["artifacts"]["manifest"],
            "review_schema": state["phases"]["review"]["artifacts"]["review_output_schema"],
            "review_run": state["phases"]["review"]["artifacts"]["review_run"],
            "reviews": state["phases"]["review"]["artifacts"]["reviews"],
            "dispositions": disposition_artifact,
        },
    }
    summary_schema = _load_json(_verify_artifact(state["inputs"]["audit_summary_schema"], repo_root, "audit summary schema"))
    Draft202012Validator(summary_schema).validate(summary)
    summary_path = work_dir / CANONICAL_SUMMARY_FILENAME
    _write_json_atomic(summary_path, summary)
    report_path = report_path if report_path.is_absolute() else repo_root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_audit_markdown(summary), encoding="utf-8")
    state["phases"]["finalize"] = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "artifacts": {
            "legacy_final_report": _artifact(deterministic["manifest"].parent / FINAL_JSON_FILENAME, repo_root),
            "dispositions": disposition_artifact,
            "summary": _artifact(summary_path, repo_root),
            "release_report": _artifact(report_path, repo_root),
        },
    }
    _save_workflow(work_dir, repo_root, state)
    return state


def audit_status(*, work_dir: Path, repo_root: Path = Path(".")) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    state = _load_workflow(work_dir.resolve(), repo_root)
    result = {
        "workflow": _portable_path(_workflow_path(work_dir.resolve()), repo_root),
        "phases": {name: phase.get("status") for name, phase in state.get("phases", {}).items()},
        "valid_through": None,
    }
    for phase_name, verifier in (
        ("prepare", lambda: _verify_prepare(state, repo_root)),
        ("deterministic", lambda: _verify_deterministic(state, repo_root)),
        ("review", lambda: _verify_review(state, repo_root)),
        (
            "finalize",
            lambda: _verify_role_map(state.get("phases", {}).get("finalize", {}).get("artifacts"), repo_root, "finalize"),
        ),
    ):
        if phase_name not in state.get("phases", {}):
            break
        verifier()
        result["valid_through"] = phase_name
    return result
