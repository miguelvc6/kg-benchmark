from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import fastjsonschema
from jsonschema import Draft202012Validator

from automated_audit_codex import REVIEWS_FILENAME, RUN_FILENAME, RunCommand, run_codex_reviews
from automated_consistency_audit import _temporal_packets
from classifier import WorldStateStore
from guardian.reasoning import (
    _few_shot_examples_from_bank,
    _load_records_by_case_id,
    _load_support_bank_for_runner,
    _prepend_few_shot_examples,
    build_prompt_bundle,
    build_track_diagnosis_prompt_bundle,
    prompt_visible_case_id,
)
from kg_benchmark.audit.workflow import (
    _artifact,
    _iter_jsonl,
    _load_json,
    _portable_path,
    _verify_artifact,
    _write_json_atomic,
    _write_jsonl_atomic,
)
from kg_benchmark.dataset.release import sha256_file
from kg_benchmark.selection.extensible import (
    STRATA,
    _dispositions,
    _effective_quotas,
    _support_group_keys,
    _weighted_tbox_order,
    build_eligibility_order,
    build_support_bank,
    group_key_for_record,
    materialize_population,
    reserve_quotas,
    stratum_for_record,
    tbox_composition,
)
from temporal_audit import audit_rendered_prompts

WORKFLOW_FILENAME = "selection-workflow.json"
RANKING_FILENAME = "ranking.json"
ELIGIBILITY_FILENAME = "eligibility-order.jsonl"
SUPPORT_FILENAME = "support-bank.json"
RESERVE_FILENAME = "reserve.json"
PROMPTS_FILENAME = "reserve-prompts.jsonl"
RENDER_SUMMARY_FILENAME = "reserve-render-summary.json"
RENDER_FAILURES_FILENAME = "reserve-render-failures.jsonl"
PRE_REVIEW_AUDIT_FILENAME = "reserve-prompt-audit.pre-review.json"
TEMPORAL_PACKETS_FILENAME = "temporal-review-packets.jsonl"
EMPTY_CONSTRUCT_PACKETS_FILENAME = "empty-construct-review-packets.jsonl"
PRIVATE_MAP_FILENAME = "private-temporal-review-map.json"
REVIEW_MANIFEST_FILENAME = "review-manifest.json"
PROMPT_AUDIT_FILENAME = "prompt-audit.json"
PER_CASE_FILENAME = "per-case-eligibility.jsonl"
CLEAN_ORDER_FILENAME = "prompt-clean-eligibility-order.jsonl"
REPLACEMENTS_FILENAME = "replacements.jsonl"
MAIN_FILENAME = "main-1200.json"
AZURE_FILENAME = "azure-600.json"


class SelectionWorkflowError(ValueError):
    """Raised when reserve or population selection cannot satisfy the frozen policy."""


def _workflow_path(output_dir: Path) -> Path:
    return output_dir / WORKFLOW_FILENAME


def _load_state(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    state = _load_json(_workflow_path(output_dir))
    schema = _load_json(repo_root / "schemas" / "selection-workflow.schema.json")
    Draft202012Validator(schema).validate(state)
    return state


def _save_state(output_dir: Path, repo_root: Path, state: dict[str, Any]) -> None:
    schema = _load_json(repo_root / "schemas" / "selection-workflow.schema.json")
    Draft202012Validator(schema).validate(state)
    _write_json_atomic(_workflow_path(output_dir), state)


def _verify_map(records: Any, repo_root: Path, prefix: str) -> dict[str, Path]:
    if not isinstance(records, dict):
        raise SelectionWorkflowError(f"Selection workflow has no {prefix} artifact bindings.")
    return {role: _verify_artifact(record, repo_root, f"{prefix}.{role}") for role, record in records.items()}


def _load_exclusions(path: Path, schema_path: Path) -> set[str]:
    value = _load_json(path)
    schema = _load_json(schema_path)
    Draft202012Validator(schema).validate(value)
    return set(value["group_keys"])


def _require_matching_bound_inputs(state: dict[str, Any], current: dict[str, Path]) -> None:
    bound = state.get("inputs")
    if not isinstance(bound, dict):
        raise SelectionWorkflowError("Selection workflow has no bound inputs.")
    for role, path in current.items():
        record = bound.get(role)
        if not isinstance(record, dict):
            raise SelectionWorkflowError(f"Selection workflow does not bind input role {role}.")
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            raise SelectionWorkflowError(
                f"Existing selection workflow was prepared from a different {role} artifact."
            )


def _policy_settings(policy: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    if policy.get("seed") != 13 or policy.get("ranking") != "sha256":
        raise SelectionWorkflowError("Canonical selection requires SHA-256 ranking with seed 13.")
    populations = policy.get("populations")
    gate = policy.get("reserve_prompt_gate")
    tbox_target = policy.get("tbox_main_target")
    support = policy.get("support_bank")
    conditions = protocol.get("conditions")
    few_shot = protocol.get("few_shot")
    audit = protocol.get("audit")
    if not all(isinstance(value, dict) for value in (populations, gate, tbox_target, support, conditions, few_shot, audit)):
        raise SelectionWorkflowError("Selection policy or protocol is missing required configuration objects.")
    if gate.get("temporal_review_sample_size") != 50:
        raise SelectionWorkflowError("The reserve temporal review sample must contain exactly 50 cases.")
    if conditions.get("prompt_regimes") != ["zero_shot", "static_few_shot"]:
        raise SelectionWorkflowError("Reserve rendering requires the two registered paper prompt regimes.")
    if conditions.get("context_bundles") != ["logic_only", "local_graph"]:
        raise SelectionWorkflowError("Reserve rendering requires logic_only and local_graph.")
    reviewer = audit.get("construct_review")
    if not isinstance(reviewer, dict) or not isinstance(reviewer.get("reviewer_model"), str):
        raise SelectionWorkflowError("Protocol does not bind the Codex reviewer model.")
    targets = {
        name: int(tbox_target[name])
        for name in ("relaxation_expansions", "restriction_contractions", "schema_updates")
    }
    return {
        "seed": 13,
        "reserve_quotas": dict(populations["reserve-1440"]),
        "main_quotas": dict(populations["main-1200"]),
        "azure_quotas": dict(populations["azure-600"]),
        "tbox_targets": targets,
        "support_capacity": int(support["capacity_per_locus"]),
        "temporal_review_size": 50,
        "prompt_regimes": list(conditions["prompt_regimes"]),
        "context_bundles": list(conditions["context_bundles"]),
        "few_shot_counts": dict(few_shot["default_example_counts"]),
        "reviewer_model": reviewer["reviewer_model"],
    }


def _validate_audit_inputs(cases: Path, dispositions: Path, summary_path: Path) -> dict[str, Any]:
    summary = _load_json(summary_path)
    validation = summary.get("validation")
    provenance = summary.get("provenance")
    if not isinstance(validation, dict) or not isinstance(provenance, dict):
        raise SelectionWorkflowError("Canonical audit summary lacks validation or provenance.")
    required = {
        "complete_unique_disposition_coverage": True,
        "deterministic_gates_passed": True,
        "temporal_gate_passed": True,
        "review_complete": True,
        "unresolved_systemic_findings": 0,
        "selection_eligible_disposition": "include",
    }
    for key, expected in required.items():
        if validation.get(key) != expected:
            raise SelectionWorkflowError(f"Audit summary does not satisfy {key}={expected!r}.")
    bound_cases = provenance.get("cases")
    bound_dispositions = provenance.get("dispositions")
    if not isinstance(bound_cases, dict) or bound_cases.get("sha256") != sha256_file(cases):
        raise SelectionWorkflowError("Audit summary does not bind the selected dataset cases.")
    if not isinstance(bound_dispositions, dict) or bound_dispositions.get("sha256") != sha256_file(dispositions):
        raise SelectionWorkflowError("Audit summary does not bind the supplied dispositions.")
    return summary


def _reserve_rows(
    eligibility_rows: list[dict[str, Any]], support_bank: dict[str, Any], quotas: dict[str, int]
) -> list[dict[str, Any]]:
    blocked = _support_group_keys(support_bank)
    rows: list[dict[str, Any]] = []
    for stratum in STRATA:
        candidates = [
            row for row in eligibility_rows if row["stratum"] == stratum and row["group_key"] not in blocked
        ]
        rows.extend(candidates[: quotas[stratum]])
    return rows


def _matrix_id(case_id: str, task: str, regime: str, context: str) -> str:
    payload = f"reserve|{case_id}|{task}|{regime}|{context}"
    return "reserve_" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def _prompt_sha256(system_prompt: str, user_prompt: str) -> str:
    return hashlib.sha256((system_prompt + "\n" + user_prompt).encode()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_eligibility_digest(row: dict[str, Any]) -> None:
    digest = row.get("eligibility_sha256")
    basis = {key: value for key, value in row.items() if key != "eligibility_sha256"}
    if digest != _canonical_sha256(basis):
        raise SelectionWorkflowError(f"Per-case eligibility digest mismatch for {row.get('case_id')}.")


def _render_reserve_prompts(
    *,
    cases_path: Path,
    world_state_path: Path,
    reserve_rows: list[dict[str, Any]],
    support_bank_path: Path,
    settings: dict[str, Any],
    output_path: Path,
    failures_path: Path,
    prompt_schema_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    reserve_ids = {row["case_id"] for row in reserve_rows}
    records = {record["id"]: record for record in _iter_jsonl(cases_path) if record.get("id") in reserve_ids}
    if set(records) != reserve_ids:
        raise SelectionWorkflowError("Reserve cases are missing from the dataset.")
    adapted_support, support_ids = _load_support_bank_for_runner(support_bank_path)
    support_records = _load_records_by_case_id(cases_path, support_ids)
    schema = _load_json(prompt_schema_path)
    validate_prompt = fastjsonschema.compile(schema)
    prompt_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counts_by_regime: Counter[str] = Counter()
    counts_by_task: Counter[str] = Counter()
    counts_by_context: Counter[str] = Counter()
    logger = logging.getLogger("kg_benchmark.selection.prompt_render")
    with WorldStateStore(world_state_path, logger) as world_store:
        for reserve_row in reserve_rows:
            case_id = reserve_row["case_id"]
            record = records[case_id]
            track = record.get("track")
            world_state = world_store.get(case_id)
            visible_case_id = prompt_visible_case_id(case_id)
            task_specs = (
                (
                    "a_box_repair" if track == "A_BOX" else "t_box_taxonomy_patch",
                    "a_box_repair" if track == "A_BOX" else "t_box_repair",
                    build_prompt_bundle,
                ),
                ("track_diagnosis", "track_diagnosis", build_track_diagnosis_prompt_bundle),
            )
            for task, support_task, builder in task_specs:
                for regime in settings["prompt_regimes"]:
                    for context in settings["context_bundles"]:
                        try:
                            bundle = builder(record, world_state, context, visible_case_id=visible_case_id)
                            if regime == "static_few_shot":
                                examples = _few_shot_examples_from_bank(
                                    support_manifest=adapted_support,
                                    eval_record=record,
                                    task=support_task,
                                    records_by_id=support_records,
                                    world_store=world_store,
                                    context_bundle=context,
                                    example_count=int(settings["few_shot_counts"][support_task]),
                                )
                                bundle = _prepend_few_shot_examples(bundle, examples)
                            if case_id in bundle.prompt or case_id in bundle.system_prompt:
                                raise ValueError("raw case id exposed in model-visible text")
                            row = {
                                "matrix_id": _matrix_id(case_id, task, regime, context),
                                "case_id": case_id,
                                "visible_case_id": visible_case_id,
                                "task": task,
                                "prompt_regime": regime,
                                "context_bundle": context,
                                "historical_track": track,
                                "prompt_name": bundle.prompt_name,
                                "system_prompt": bundle.system_prompt,
                                "user_prompt": bundle.prompt,
                                "prompt_sha256": _prompt_sha256(bundle.system_prompt, bundle.prompt),
                                "response_format": bundle.response_format,
                            }
                            validate_prompt(row)
                            prompt_rows.append(row)
                            counts_by_regime[regime] += 1
                            counts_by_task[task] += 1
                            counts_by_context[context] += 1
                        except Exception as exc:
                            failures.append(
                                {
                                    "case_id": case_id,
                                    "task": task,
                                    "prompt_regime": regime,
                                    "context_bundle": context,
                                    "error_type": exc.__class__.__name__,
                                    "error": str(exc),
                                }
                            )
    _write_jsonl_atomic(output_path, prompt_rows)
    _write_jsonl_atomic(failures_path, failures)
    expected = len(reserve_rows) * 2 * len(settings["prompt_regimes"]) * len(settings["context_bundles"])
    return (
        {
            "report_type": "reserve_prompt_render_summary",
            "report_version": 1,
            "reserve_cases": len(reserve_rows),
            "expected_prompts": expected,
            "rendered_prompts": len(prompt_rows),
            "render_failures": len(failures),
            "cases_with_render_failures": len({row["case_id"] for row in failures}),
            "counts_by_regime": dict(sorted(counts_by_regime.items())),
            "counts_by_task": dict(sorted(counts_by_task.items())),
            "counts_by_context": dict(sorted(counts_by_context.items())),
        },
        records,
    )


def _input_artifacts(
    *,
    repo_root: Path,
    cases: Path,
    world_state: Path,
    dispositions: Path,
    audit_summary: Path,
    exclusions: Path,
    protocol: Path,
    policy: Path,
) -> dict[str, Any]:
    paths = {
        "dataset": cases,
        "world_state": world_state,
        "audit_dispositions": dispositions,
        "audit": audit_summary,
        "exclusions": exclusions,
        "protocol": protocol,
        "selection_policy": policy,
        "selection_manifest_schema": repo_root / "schemas" / "selection-manifest.schema.json",
        "reserve_manifest_schema": repo_root / "schemas" / "reserve-manifest.schema.json",
        "prompt_schema": repo_root / "schemas" / "rendered-audit-prompt.schema.json",
        "per_case_schema": repo_root / "schemas" / "per-case-eligibility.schema.json",
        "prompt_audit_schema": repo_root / "schemas" / "selection-prompt-audit.schema.json",
        "exclusion_schema": repo_root / "schemas" / "group-exclusions.schema.json",
    }
    return {role: _artifact(path, repo_root) for role, path in paths.items()}


def prepare_reserve(
    *,
    cases_path: Path,
    world_state_path: Path,
    dispositions_path: Path,
    audit_summary_path: Path,
    exclusions_path: Path,
    protocol_path: Path,
    policy_path: Path,
    output_dir: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if _workflow_path(output_dir).exists():
        state = _load_state(output_dir, repo_root)
        _require_matching_bound_inputs(
            state,
            {
                "dataset": cases_path,
                "world_state": world_state_path,
                "audit_dispositions": dispositions_path,
                "audit": audit_summary_path,
                "exclusions": exclusions_path,
                "protocol": protocol_path,
                "selection_policy": policy_path,
            },
        )
        verify_reserve(state=state, repo_root=repo_root)
        return state
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SelectionWorkflowError(f"Selection output directory is non-empty without a workflow: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_audit_inputs(cases_path, dispositions_path, audit_summary_path)
    policy = _load_json(policy_path)
    protocol = _load_json(protocol_path)
    settings = _policy_settings(policy, protocol)
    exclusions = _load_exclusions(exclusions_path, repo_root / "schemas" / "group-exclusions.schema.json")
    eligibility_rows, records_by_id = build_eligibility_order(
        cases_path=cases_path,
        dispositions_path=dispositions_path,
        seed=settings["seed"],
        excluded_group_keys=exclusions,
        tbox_targets=settings["tbox_targets"],
    )
    support_bank = build_support_bank(
        eligibility_rows=eligibility_rows,
        records_by_id=records_by_id,
        capacity_per_locus=settings["support_capacity"],
    )
    effective_reserve = reserve_quotas(
        requested=settings["reserve_quotas"],
        eligibility_rows=eligibility_rows,
        support_bank=support_bank,
    )
    reserve_rows = _reserve_rows(eligibility_rows, support_bank, effective_reserve)
    inputs = _input_artifacts(
        repo_root=repo_root,
        cases=cases_path,
        world_state=world_state_path,
        dispositions=dispositions_path,
        audit_summary=audit_summary_path,
        exclusions=exclusions_path,
        protocol=protocol_path,
        policy=policy_path,
    )
    ranking = {
        "manifest_type": "selection_ranking",
        "manifest_version": 1,
        "algorithm": "sha256",
        "seed": settings["seed"],
        "group_units": policy["sampling_unit"],
        "tbox_weighted_prefix_target": settings["tbox_targets"],
        "tie_break": "case_id",
    }
    ranking_path = output_dir / RANKING_FILENAME
    eligibility_path = output_dir / ELIGIBILITY_FILENAME
    support_path = output_dir / SUPPORT_FILENAME
    reserve_path = output_dir / RESERVE_FILENAME
    _write_json_atomic(ranking_path, ranking)
    _write_jsonl_atomic(eligibility_path, eligibility_rows)
    _write_json_atomic(support_path, support_bank)
    reserve = materialize_population(
        name="reserve-1440",
        quotas=effective_reserve,
        eligibility_rows=eligibility_rows,
        support_bank=support_bank,
    )
    reserve.update(
        {
            "manifest_type": "selection_reserve",
            "manifest_version": 1,
            "requested_quotas": settings["reserve_quotas"],
            "tbox_composition": tbox_composition(eligibility_rows, reserve["selected_case_ids"]),
        }
    )
    reserve.pop("parent", None)
    Draft202012Validator(_load_json(repo_root / "schemas" / "reserve-manifest.schema.json")).validate(reserve)
    _write_json_atomic(reserve_path, reserve)
    prompts_path = output_dir / PROMPTS_FILENAME
    failures_path = output_dir / RENDER_FAILURES_FILENAME
    render_summary, records = _render_reserve_prompts(
        cases_path=cases_path,
        world_state_path=world_state_path,
        reserve_rows=reserve_rows,
        support_bank_path=support_path,
        settings=settings,
        output_path=prompts_path,
        failures_path=failures_path,
        prompt_schema_path=repo_root / "schemas" / "rendered-audit-prompt.schema.json",
    )
    render_summary_path = output_dir / RENDER_SUMMARY_FILENAME
    _write_json_atomic(render_summary_path, render_summary)
    temporal = audit_rendered_prompts(
        rendered_prompts_path=prompts_path,
        classified_benchmark_path=cases_path,
        sample_size=settings["temporal_review_size"],
        seed=settings["seed"],
    )
    if temporal["counts"]["manual_sample"] != settings["temporal_review_size"]:
        raise SelectionWorkflowError(
            f"Reserve temporal review requires {settings['temporal_review_size']} distinct rendered cases."
        )
    if temporal["missing_case_ids"]:
        raise SelectionWorkflowError("Deterministic reserve prompt scanning could not resolve every rendered case.")
    if not all(temporal["mutation_sensitivity_checks"].values()):
        raise SelectionWorkflowError("Deterministic reserve prompt scanner failed its mutation-sensitivity checks.")
    pre_review = {
        "report_type": "reserve_prompt_audit_pre_review",
        "report_version": 1,
        "render_summary": render_summary,
        "temporal_audit": temporal,
        "deterministic_failed_case_ids": sorted(
            {row["case_id"] for row in _iter_jsonl(failures_path)}
            | {hit["case_id"] for hit in temporal["hits"] if hit.get("severity") == "high"}
        ),
    }
    pre_review_path = output_dir / PRE_REVIEW_AUDIT_FILENAME
    _write_json_atomic(pre_review_path, pre_review)
    packets, private_map = _temporal_packets(temporal, prompts_path, records)
    packets_path = output_dir / TEMPORAL_PACKETS_FILENAME
    empty_construct_path = output_dir / EMPTY_CONSTRUCT_PACKETS_FILENAME
    private_map_path = output_dir / PRIVATE_MAP_FILENAME
    _write_jsonl_atomic(packets_path, packets)
    _write_jsonl_atomic(empty_construct_path, [])
    _write_json_atomic(private_map_path, private_map)
    review_dir = output_dir / "review"
    review_manifest_path = output_dir / REVIEW_MANIFEST_FILENAME
    _write_json_atomic(
        review_manifest_path,
        {
            "manifest_type": "reserve_temporal_review",
            "manifest_version": 1,
            "output_dir": str(review_dir.resolve()),
            "model": settings["reviewer_model"],
        },
    )
    artifacts = {
        "ranking": _artifact(ranking_path, repo_root),
        "eligibility_order": _artifact(eligibility_path, repo_root, records=len(eligibility_rows)),
        "support_bank": _artifact(support_path, repo_root),
        "reserve": _artifact(reserve_path, repo_root),
        "rendered_prompts": _artifact(prompts_path, repo_root, records=render_summary["rendered_prompts"]),
        "render_summary": _artifact(render_summary_path, repo_root),
        "render_failures": _artifact(failures_path, repo_root, records=render_summary["render_failures"]),
        "pre_review_prompt_audit": _artifact(pre_review_path, repo_root),
        "temporal_review_packets": _artifact(packets_path, repo_root, records=len(packets)),
        "empty_construct_packets": _artifact(empty_construct_path, repo_root, records=0),
        "private_review_map": _artifact(private_map_path, repo_root),
        "review_manifest": _artifact(review_manifest_path, repo_root),
    }
    state = {
        "manifest_type": "selection_workflow",
        "manifest_version": 1,
        "settings": settings,
        "inputs": inputs,
        "phases": {
            "reserve": {
                "status": "complete",
                "artifacts": artifacts,
                "counts": {
                    "eligible_groups": len(eligibility_rows),
                    "reserve_cases": len(reserve_rows),
                    "rendered_prompts": render_summary["rendered_prompts"],
                    "temporal_review_cases": len(packets),
                },
            }
        },
    }
    _save_state(output_dir, repo_root, state)
    return state


def verify_reserve(*, state: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    _verify_map(state.get("inputs"), repo_root, "inputs")
    phase = state.get("phases", {}).get("reserve")
    if not isinstance(phase, dict) or phase.get("status") != "complete":
        raise SelectionWorkflowError("Reserve phase is not complete.")
    return _verify_map(phase.get("artifacts"), repo_root, "reserve")


def review_reserve(
    *,
    output_dir: Path,
    repo_root: Path = Path("."),
    batch_size: int = 10,
    workers: int = 1,
    retries: int = 2,
    timeout_seconds: float = 600,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    state = _load_state(output_dir, repo_root)
    reserve = verify_reserve(state=state, repo_root=repo_root)
    existing = state["phases"].get("review")
    if isinstance(existing, dict) and existing.get("status") == "complete":
        verify_review(state=state, repo_root=repo_root)
        return state
    report = run_codex_reviews(
        manifest_path=reserve["review_manifest"],
        construct_packets_path=reserve["empty_construct_packets"],
        temporal_packets_path=reserve["temporal_review_packets"],
        model=state["settings"]["reviewer_model"],
        batch_size=batch_size,
        workers=workers,
        retries=retries,
        timeout_seconds=timeout_seconds,
        run_command=run_command,
    )
    review_dir = output_dir / "review"
    if report.get("status") != "complete" or report.get("reviews", {}).get("count") != 50:
        raise SelectionWorkflowError("Reserve temporal review is incomplete.")
    state["phases"]["review"] = {
        "status": "complete",
        "reviewer": {
            "interface": "codex_cli",
            "model": report["codex"]["model"],
            "cli_version": report["codex"]["version"],
        },
        "artifacts": {
            "review_run": _artifact(review_dir / RUN_FILENAME, repo_root),
            "reviews": _artifact(review_dir / REVIEWS_FILENAME, repo_root, records=50),
            "review_schema": _artifact(review_dir / ".codex_review_output.schema.json", repo_root),
        },
    }
    _save_state(output_dir, repo_root, state)
    return state


def verify_review(*, state: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    phase = state.get("phases", {}).get("review")
    if not isinstance(phase, dict) or phase.get("status") != "complete":
        raise SelectionWorkflowError("Reserve temporal review phase is not complete.")
    paths = _verify_map(phase.get("artifacts"), repo_root, "review")
    report = _load_json(paths["review_run"])
    if report.get("status") != "complete" or report.get("reviews", {}).get("count") != 50:
        raise SelectionWorkflowError("Bound reserve temporal review is incomplete.")
    reviews = list(_iter_jsonl(paths["reviews"]))
    private_map = _load_json(_verify_artifact(
        state["phases"]["reserve"]["artifacts"]["private_review_map"],
        repo_root,
        "reserve.private_review_map",
    ))
    reviewed_ids = [row.get("case_id") for row in reviews]
    if len(reviews) != 50 or len(set(reviewed_ids)) != 50 or set(reviewed_ids) != set(private_map):
        raise SelectionWorkflowError("Bound reserve temporal review does not cover the fixed 50 blinded packets.")
    return paths


def _review_failures(reviews_path: Path, private_map_path: Path) -> dict[str, list[str]]:
    private_map = _load_json(private_map_path)
    failures: dict[str, list[str]] = defaultdict(list)
    for review in _iter_jsonl(reviews_path):
        raw_case_id = private_map.get(review.get("case_id"))
        if not isinstance(raw_case_id, str):
            raise SelectionWorkflowError("Temporal review references an unknown blinded case.")
        if review.get("verdict") != "pass":
            failures[raw_case_id].append(f"codex_temporal_{review.get('verdict')}")
    return failures


def _per_case_rows(
    *,
    cases_path: Path,
    dispositions_path: Path,
    eligibility_rows: list[dict[str, Any]],
    support_bank: dict[str, Any],
    excluded_groups: set[str],
    reserve_ids: set[str],
    failure_reasons: dict[str, list[str]],
) -> list[dict[str, Any]]:
    dispositions = _dispositions(dispositions_path)
    representative_ids = {row["case_id"] for row in eligibility_rows}
    support_groups = _support_group_keys(support_bank)
    rows: list[dict[str, Any]] = []
    for record in _iter_jsonl(cases_path):
        case_id = record["id"]
        disposition = dispositions[case_id]
        reasons: list[str] = []
        try:
            group_key = group_key_for_record(record)
        except ValueError:
            group_key = None
            reasons.append("invalid_group_key")
        stratum = stratum_for_record(record)
        representative = case_id in representative_ids
        in_reserve = case_id in reserve_ids
        prompt_clean = in_reserve and not failure_reasons.get(case_id)
        if disposition != "include":
            reasons.append("disposition_not_include")
        if group_key in excluded_groups:
            reasons.append("previously_used_group")
        if disposition == "include" and group_key is not None and group_key not in excluded_groups and not representative:
            reasons.append("duplicate_group_nonrepresentative")
        if group_key in support_groups:
            reasons.append("few_shot_support_group")
        if in_reserve and not prompt_clean:
            reasons.extend(failure_reasons.get(case_id, ["prompt_not_clean"]))
        if representative and group_key not in support_groups and not in_reserve:
            reasons.append("outside_audited_reserve")
        eligible = (
            disposition == "include"
            and representative
            and group_key not in excluded_groups
            and group_key not in support_groups
            and in_reserve
            and prompt_clean
        )
        row = {
            "case_id": case_id,
            "disposition": disposition,
            "group_key": group_key,
            "stratum": stratum,
            "independent_group_representative": representative,
            "in_reserve": in_reserve,
            "prompt_clean": prompt_clean if in_reserve else None,
            "selection_eligible": eligible,
            "reasons": sorted(set(reasons)),
        }
        row["eligibility_sha256"] = _canonical_sha256(row)
        rows.append(row)
    rows.sort(key=lambda row: row["case_id"])
    return rows


def _population_v2(
    *,
    population: dict[str, Any],
    requested_quotas: dict[str, int],
    ordering_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    eligibility_by_case: dict[str, dict[str, Any]],
    excluded_groups: set[str],
    support_groups: set[str],
    parent: dict[str, Any] | None = None,
    parent_case_ids: set[str] | None = None,
    parent_relationship: str | None = None,
) -> dict[str, Any]:
    selected_ids = population["selected_case_ids"]
    try:
        eligibility = [eligibility_by_case[case_id] for case_id in selected_ids]
    except KeyError as exc:
        raise SelectionWorkflowError(f"Population references a case without an eligibility record: {exc.args[0]}") from exc
    selected_groups = set(population["selected_group_keys"])
    if parent is None:
        nested_subset_proven = True
    elif parent_relationship == "subset_of_parent":
        nested_subset_proven = (
            bool(parent.get("nesting_proven"))
            and parent_case_ids is not None
            and set(selected_ids).issubset(parent_case_ids)
        )
    elif parent_relationship == "nested_extension_of_parent":
        nested_subset_proven = (
            bool(parent.get("nesting_proven"))
            and parent_case_ids is not None
            and parent_case_ids.issubset(selected_ids)
        )
    else:
        nested_subset_proven = False
    population["manifest_version"] = 2
    population["requested_quotas"] = requested_quotas
    population["tbox_composition"] = tbox_composition(ordering_rows, population["selected_case_ids"])
    population["case_eligibility_sha256"] = {
        row["case_id"]: row["eligibility_sha256"] for row in eligibility
    }
    population["provenance"] = provenance
    population["validation"] = {
        "prompt_clean_only": all(row["prompt_clean"] is True for row in eligibility),
        "include_disposition_only": all(
            row["disposition"] == "include" and row["selection_eligible"] is True for row in eligibility
        ),
        "independent_groups": len(population["selected_group_keys"]) == len(set(population["selected_group_keys"])),
        "prior_groups_excluded": selected_groups.isdisjoint(excluded_groups),
        "support_groups_excluded": selected_groups.isdisjoint(support_groups),
        "nested_subset_proven": nested_subset_proven,
    }
    if parent is not None:
        population["parent"] = {**parent, "relationship": parent_relationship}
    return population


def _replacements(
    reserve_rows: list[dict[str, Any]], clean_ids: set[str], quotas: dict[str, int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stratum in STRATA:
        candidates = [row for row in reserve_rows if row["stratum"] == stratum]
        initial = candidates[: quotas[stratum]]
        final = [row for row in candidates if row["case_id"] in clean_ids][: quotas[stratum]]
        initial_ids = {row["case_id"] for row in initial}
        failures = [row for row in initial if row["case_id"] not in clean_ids]
        additions = [row for row in final if row["case_id"] not in initial_ids]
        for failed, replacement in zip(failures, additions, strict=True):
            rows.append(
                {
                    "stratum": stratum,
                    "failed_case_id": failed["case_id"],
                    "failed_group_key": failed["group_key"],
                    "replacement_case_id": replacement["case_id"],
                    "replacement_group_key": replacement["group_key"],
                }
            )
    return rows


def _prompt_clean_order(
    reserve_rows: list[dict[str, Any]],
    failure_reasons: dict[str, list[str]],
    tbox_targets: dict[str, int],
) -> list[dict[str, Any]]:
    """Preserve frozen ranks while restoring the declared T-box prefix after prompt QA."""
    clean = [row for row in reserve_rows if not failure_reasons.get(row["case_id"])]
    abox = [row for row in clean if row["stratum"] != "TBOX"]
    tbox = _weighted_tbox_order(
        [row for row in clean if row["stratum"] == "TBOX"],
        tbox_targets,
    )
    rows = [*abox, *tbox]
    for stratum in STRATA:
        for position, row in enumerate((row for row in rows if row["stratum"] == stratum), 1):
            row["prompt_clean_stratum_position"] = position
    return rows


def finalize_selection(
    *, output_dir: Path, repo_root: Path = Path(".")
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    state = _load_state(output_dir, repo_root)
    reserve_artifacts = verify_reserve(state=state, repo_root=repo_root)
    review_artifacts = verify_review(state=state, repo_root=repo_root)
    existing = state["phases"].get("finalize")
    if isinstance(existing, dict) and existing.get("status") == "complete":
        verify_finalization(state=state, repo_root=repo_root)
        return state
    inputs = _verify_map(state["inputs"], repo_root, "inputs")
    eligibility_rows = list(_iter_jsonl(reserve_artifacts["eligibility_order"]))
    support_bank = _load_json(reserve_artifacts["support_bank"])
    reserve = _load_json(reserve_artifacts["reserve"])
    reserve_ids = set(reserve["selected_case_ids"])
    reserve_rows = [row for row in eligibility_rows if row["case_id"] in reserve_ids]
    failure_reasons: dict[str, list[str]] = defaultdict(list)
    for failure in _iter_jsonl(reserve_artifacts["render_failures"]):
        failure_reasons[failure["case_id"]].append("prompt_render_failure")
    pre_review = _load_json(reserve_artifacts["pre_review_prompt_audit"])
    for hit in pre_review["temporal_audit"]["hits"]:
        if hit.get("severity") == "high":
            failure_reasons[hit["case_id"]].append("deterministic_temporal_leakage")
    for case_id, reasons in _review_failures(
        review_artifacts["reviews"], reserve_artifacts["private_review_map"]
    ).items():
        failure_reasons[case_id].extend(reasons)
    clean_rows = _prompt_clean_order(
        reserve_rows,
        failure_reasons,
        state["settings"]["tbox_targets"],
    )
    clean_ids = {row["case_id"] for row in clean_rows}
    effective_main = _effective_quotas(
        requested=state["settings"]["main_quotas"],
        eligibility_rows=clean_rows,
        support_bank=support_bank,
    )
    main = materialize_population(
        name="main-1200",
        quotas=effective_main,
        eligibility_rows=clean_rows,
        support_bank=support_bank,
    )
    if main["case_count"] != sum(state["settings"]["main_quotas"].values()):
        raise SelectionWorkflowError("Final main population does not contain exactly 1,200 cases.")
    main_ids = set(main["selected_case_ids"])
    main_order = [row for row in clean_rows if row["case_id"] in main_ids]
    effective_api = _effective_quotas(
        requested=state["settings"]["azure_quotas"],
        eligibility_rows=main_order,
        support_bank=support_bank,
    )
    azure = materialize_population(
        name="azure-600",
        quotas=effective_api,
        eligibility_rows=main_order,
        support_bank=support_bank,
    )
    if azure["case_count"] != sum(state["settings"]["azure_quotas"].values()):
        raise SelectionWorkflowError("Final Azure population does not contain exactly 600 cases.")
    replacements = _replacements(reserve_rows, clean_ids, effective_main)
    replacements_path = output_dir / REPLACEMENTS_FILENAME
    _write_jsonl_atomic(replacements_path, replacements)
    excluded_groups = _load_exclusions(inputs["exclusions"], inputs["exclusion_schema"])
    per_case_rows = _per_case_rows(
        cases_path=inputs["dataset"],
        dispositions_path=inputs["audit_dispositions"],
        eligibility_rows=eligibility_rows,
        support_bank=support_bank,
        excluded_groups=excluded_groups,
        reserve_ids=reserve_ids,
        failure_reasons=failure_reasons,
    )
    per_case_schema = _load_json(inputs["per_case_schema"])
    validator = Draft202012Validator(per_case_schema)
    for row in per_case_rows:
        validator.validate(row)
        _verify_eligibility_digest(row)
    per_case_path = output_dir / PER_CASE_FILENAME
    clean_order_path = output_dir / CLEAN_ORDER_FILENAME
    _write_jsonl_atomic(per_case_path, per_case_rows)
    _write_jsonl_atomic(clean_order_path, clean_rows)
    eligibility_by_case = {row["case_id"]: row for row in per_case_rows}
    support_groups = _support_group_keys(support_bank)
    prompt_audit = {
        "report_type": "selection_prompt_audit",
        "report_version": 1,
        "seed": state["settings"]["seed"],
        "reserve_cases": len(reserve_ids),
        "rendered_prompts": pre_review["render_summary"]["rendered_prompts"],
        "deterministically_scanned_prompts": pre_review["temporal_audit"]["counts"]["prompt_rows"],
        "temporal_review_cases": 50,
        "failed_reserve_cases": len(failure_reasons),
        "prompt_clean_reserve_cases": len(clean_rows),
        "replacement_count": len(replacements),
        "failed_case_ids": sorted(failure_reasons),
        "failure_reasons": {key: sorted(set(value)) for key, value in sorted(failure_reasons.items())},
        "validation": {
            "all_reserve_prompts_attempted": (
                pre_review["render_summary"]["rendered_prompts"]
                + pre_review["render_summary"]["render_failures"]
                == pre_review["render_summary"]["expected_prompts"]
            ),
            "all_rendered_prompts_scanned": (
                pre_review["render_summary"]["rendered_prompts"]
                == pre_review["temporal_audit"]["counts"]["prompt_rows"]
            ),
            "fixed_temporal_review_complete": True,
            "all_main_cases_prompt_clean": set(main["selected_case_ids"]).issubset(clean_ids),
            "all_azure_cases_prompt_clean": set(azure["selected_case_ids"]).issubset(clean_ids),
        },
        "reviewer": state["phases"]["review"]["reviewer"],
    }
    prompt_audit_schema = _load_json(inputs["prompt_audit_schema"])
    Draft202012Validator(prompt_audit_schema).validate(prompt_audit)
    prompt_audit_path = output_dir / PROMPT_AUDIT_FILENAME
    _write_json_atomic(prompt_audit_path, prompt_audit)
    provenance = {
        "dataset": state["inputs"]["dataset"],
        "audit": state["inputs"]["audit"],
        "audit_dispositions": state["inputs"]["audit_dispositions"],
        "exclusions": state["inputs"]["exclusions"],
        "ranking": state["phases"]["reserve"]["artifacts"]["ranking"],
        "support_bank": state["phases"]["reserve"]["artifacts"]["support_bank"],
        "eligibility_order": state["phases"]["reserve"]["artifacts"]["eligibility_order"],
        "prompt_audit": _artifact(prompt_audit_path, repo_root),
        "per_case_eligibility": _artifact(per_case_path, repo_root, records=len(per_case_rows)),
    }
    main = _population_v2(
        population=main,
        requested_quotas=state["settings"]["main_quotas"],
        ordering_rows=clean_rows,
        provenance=provenance,
        eligibility_by_case=eligibility_by_case,
        excluded_groups=excluded_groups,
        support_groups=support_groups,
    )
    azure = _population_v2(
        population=azure,
        requested_quotas=state["settings"]["azure_quotas"],
        ordering_rows=main_order,
        provenance=provenance,
        eligibility_by_case=eligibility_by_case,
        excluded_groups=excluded_groups,
        support_groups=support_groups,
        parent={"name": "main-1200", "nesting_proven": set(azure["selected_case_ids"]).issubset(main_ids)},
        parent_case_ids=main_ids,
        parent_relationship="subset_of_parent",
    )
    schema = _load_json(inputs["selection_manifest_schema"])
    Draft202012Validator(schema).validate(main)
    Draft202012Validator(schema).validate(azure)
    main_path = output_dir / MAIN_FILENAME
    azure_path = output_dir / AZURE_FILENAME
    _write_json_atomic(main_path, main)
    _write_json_atomic(azure_path, azure)
    state["phases"]["finalize"] = {
        "status": "complete",
        "artifacts": {
            "prompt_audit": _artifact(prompt_audit_path, repo_root),
            "per_case_eligibility": _artifact(per_case_path, repo_root, records=len(per_case_rows)),
            "prompt_clean_eligibility_order": _artifact(clean_order_path, repo_root, records=len(clean_rows)),
            "replacements": _artifact(replacements_path, repo_root, records=len(replacements)),
            "main": _artifact(main_path, repo_root),
            "azure": _artifact(azure_path, repo_root),
        },
        "counts": {
            "prompt_clean_cases": len(clean_rows),
            "main_cases": main["case_count"],
            "azure_cases": azure["case_count"],
            "replacements": len(replacements),
        },
    }
    _save_state(output_dir, repo_root, state)
    verify_finalization(state=state, repo_root=repo_root)
    return state


def verify_finalization(*, state: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    phase = state.get("phases", {}).get("finalize")
    if not isinstance(phase, dict) or phase.get("status") != "complete":
        raise SelectionWorkflowError("Selection finalization phase is not complete.")
    paths = _verify_map(phase.get("artifacts"), repo_root, "finalize")
    inputs = _verify_map(state.get("inputs"), repo_root, "inputs")
    reserve = verify_reserve(state=state, repo_root=repo_root)

    per_case_schema = _load_json(inputs["per_case_schema"])
    per_case_validator = Draft202012Validator(per_case_schema)
    per_case_by_id: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(paths["per_case_eligibility"]):
        per_case_validator.validate(row)
        _verify_eligibility_digest(row)
        case_id = row["case_id"]
        if case_id in per_case_by_id:
            raise SelectionWorkflowError(f"Duplicate per-case eligibility record for {case_id}.")
        per_case_by_id[case_id] = row

    clean_rows = list(_iter_jsonl(paths["prompt_clean_eligibility_order"]))
    clean_ids = [row["case_id"] for row in clean_rows]
    if len(clean_ids) != len(set(clean_ids)):
        raise SelectionWorkflowError("Prompt-clean eligibility ordering contains duplicate cases.")
    expected_clean = {
        case_id for case_id, row in per_case_by_id.items() if row.get("selection_eligible") is True
    }
    if set(clean_ids) != expected_clean:
        raise SelectionWorkflowError("Prompt-clean ordering and per-case eligibility decisions disagree.")

    prompt_audit = _load_json(paths["prompt_audit"])
    Draft202012Validator(_load_json(inputs["prompt_audit_schema"])).validate(prompt_audit)
    population_schema = _load_json(inputs["selection_manifest_schema"])
    populations = {name: _load_json(paths[name]) for name in ("main", "azure")}
    expected_provenance = {
        "dataset": state["inputs"]["dataset"],
        "audit": state["inputs"]["audit"],
        "audit_dispositions": state["inputs"]["audit_dispositions"],
        "exclusions": state["inputs"]["exclusions"],
        "ranking": state["phases"]["reserve"]["artifacts"]["ranking"],
        "support_bank": state["phases"]["reserve"]["artifacts"]["support_bank"],
        "eligibility_order": state["phases"]["reserve"]["artifacts"]["eligibility_order"],
        "prompt_audit": phase["artifacts"]["prompt_audit"],
        "per_case_eligibility": phase["artifacts"]["per_case_eligibility"],
    }
    for name, population in populations.items():
        Draft202012Validator(population_schema).validate(population)
        selected_ids = population["selected_case_ids"]
        if population["case_count"] != len(selected_ids):
            raise SelectionWorkflowError(f"{name} population case count disagrees with its selected IDs.")
        if set(population["case_eligibility_sha256"]) != set(selected_ids):
            raise SelectionWorkflowError(f"{name} population does not bind every selected eligibility decision.")
        for case_id, digest in population["case_eligibility_sha256"].items():
            if case_id not in per_case_by_id or digest != per_case_by_id[case_id]["eligibility_sha256"]:
                raise SelectionWorkflowError(f"{name} population has a mismatched eligibility digest for {case_id}.")
        if population["provenance"] != expected_provenance:
            raise SelectionWorkflowError(f"{name} population provenance does not match the selection workflow.")
        if not set(selected_ids).issubset(clean_ids):
            raise SelectionWorkflowError(f"{name} population contains a case outside the prompt-clean order.")
    if not set(populations["azure"]["selected_case_ids"]).issubset(populations["main"]["selected_case_ids"]):
        raise SelectionWorkflowError("Azure population is not nested within the main population.")
    if reserve["reserve"].stat().st_size == 0:
        raise SelectionWorkflowError("Bound reserve manifest is empty.")
    return paths


def expand_population(
    *,
    output_dir: Path,
    parent_path: Path,
    name: str,
    requested_quotas: dict[str, int],
    destination: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    state = _load_state(output_dir, repo_root)
    verify_reserve(state=state, repo_root=repo_root)
    verify_review(state=state, repo_root=repo_root)
    final = state.get("phases", {}).get("finalize")
    if not isinstance(final, dict) or final.get("status") != "complete":
        raise SelectionWorkflowError("Selection must be finalized before materializing an extension.")
    final_artifacts = verify_finalization(state=state, repo_root=repo_root)
    if destination.exists():
        raise FileExistsError(f"Population manifest already exists: {destination}")
    clean_rows = list(_iter_jsonl(final_artifacts["prompt_clean_eligibility_order"]))
    per_case_rows = list(_iter_jsonl(final_artifacts["per_case_eligibility"]))
    eligibility_by_case = {row["case_id"]: row for row in per_case_rows}
    support_bank = _load_json(_verify_artifact(state["phases"]["reserve"]["artifacts"]["support_bank"], repo_root, "support bank"))
    inputs = _verify_map(state["inputs"], repo_root, "inputs")
    excluded_groups = _load_exclusions(inputs["exclusions"], inputs["exclusion_schema"])
    parent = _load_json(parent_path)
    effective = _effective_quotas(
        requested=requested_quotas,
        eligibility_rows=clean_rows,
        support_bank=support_bank,
    )
    population = materialize_population(
        name=name,
        quotas=effective,
        eligibility_rows=clean_rows,
        support_bank=support_bank,
        parent_manifest=parent,
    )
    main = _load_json(final_artifacts["main"])
    provenance = dict(main["provenance"])
    provenance["parent_population"] = _artifact(parent_path, repo_root)
    population = _population_v2(
        population=population,
        requested_quotas=requested_quotas,
        ordering_rows=clean_rows,
        provenance=provenance,
        eligibility_by_case=eligibility_by_case,
        excluded_groups=excluded_groups,
        support_groups=_support_group_keys(support_bank),
        parent=population["parent"],
        parent_case_ids=set(parent["selected_case_ids"]),
        parent_relationship="nested_extension_of_parent",
    )
    schema = _load_json(repo_root / "schemas" / "selection-manifest.schema.json")
    Draft202012Validator(schema).validate(population)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(destination, population)
    return population


def selection_status(*, output_dir: Path, repo_root: Path = Path(".")) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    state = _load_state(output_dir.resolve(), repo_root)
    valid_through: str | None = None
    verify_reserve(state=state, repo_root=repo_root)
    valid_through = "reserve"
    if "review" in state.get("phases", {}):
        verify_review(state=state, repo_root=repo_root)
        valid_through = "review"
    if "finalize" in state.get("phases", {}):
        verify_finalization(state=state, repo_root=repo_root)
        valid_through = "finalize"
    return {
        "workflow": _portable_path(_workflow_path(output_dir.resolve()), repo_root),
        "valid_through": valid_through,
        "phases": {name: phase.get("status") for name, phase in state.get("phases", {}).items()},
    }
