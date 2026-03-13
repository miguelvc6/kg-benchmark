from __future__ import annotations

import json
import re
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from tqdm import tqdm

from classifier import WorldStateStore
from guardian.evaluator import evaluate_benchmark, summarize_trace_iterable, write_json
from guardian.model_provider import BatchModelProvider, ModelProvider, create_model_provider
from guardian.patch_parser import load_schema as load_a_box_schema
from guardian.patch_parser import normalize_proposal as normalize_a_box_proposal
from guardian.prompts import get_prompt_template
from guardian.tbox_parser import load_schema as load_t_box_schema
from guardian.tbox_parser import normalize_proposal as normalize_t_box_proposal
from guardian.track_parser import load_schema as load_track_schema
from guardian.track_parser import normalize_diagnosis
from lib.benchmark_selection import resolve_case_id_filter
from lib.utils import iter_jsonl


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


ABLATION_BUNDLES = ("minimal_case", "logic_only", "local_graph")


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered)
    return slug.strip("_") or "unknown"


def _usage_block(usage: dict[str, Any], elapsed_seconds: float | None) -> dict[str, Any]:
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": usage.get("cached_tokens"),
        "estimated_cost_usd": usage.get("estimated_cost_usd"),
        "input_cost_per_1m_tokens_usd": usage.get("input_cost_per_1m_tokens_usd"),
        "output_cost_per_1m_tokens_usd": usage.get("output_cost_per_1m_tokens_usd"),
        "elapsed_seconds": round(elapsed_seconds, 6) if isinstance(elapsed_seconds, (int, float)) else None,
    }


def _aggregate_run_usage(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cached_tokens = 0
    estimated_cost_usd = 0.0
    elapsed_seconds = 0.0
    has_prompt = False
    has_completion = False
    has_total = False
    has_cached = False
    has_cost = False
    has_elapsed = False
    for record in manifest:
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
        value = usage.get("prompt_tokens")
        if isinstance(value, int):
            prompt_tokens += value
            has_prompt = True
        value = usage.get("completion_tokens")
        if isinstance(value, int):
            completion_tokens += value
            has_completion = True
        value = usage.get("total_tokens")
        if isinstance(value, int):
            total_tokens += value
            has_total = True
        value = usage.get("cached_tokens")
        if isinstance(value, int):
            cached_tokens += value
            has_cached = True
        value = usage.get("estimated_cost_usd")
        if isinstance(value, (int, float)):
            estimated_cost_usd += float(value)
            has_cost = True
        value = usage.get("elapsed_seconds")
        if isinstance(value, (int, float)):
            elapsed_seconds += float(value)
            has_elapsed = True
    return {
        "prompt_tokens": prompt_tokens if has_prompt else None,
        "completion_tokens": completion_tokens if has_completion else None,
        "total_tokens": total_tokens if has_total else None,
        "cached_tokens": cached_tokens if has_cached else None,
        "estimated_cost_usd": round(estimated_cost_usd, 10) if has_cost else None,
        "generation_elapsed_seconds": round(elapsed_seconds, 6) if has_elapsed else None,
    }


def _format_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.4f}"


@dataclass
class PromptBundle:
    ablation_bundle: str
    prompt_name: str
    prompt: str
    system_prompt: str
    response_format: dict[str, Any]


def _base_case_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "qid": record.get("qid"),
        "property": record.get("property"),
        "track": record.get("track"),
        "classification": record.get("classification"),
        "labels_en": record.get("labels_en"),
        "violation_context": record.get("violation_context"),
        "persistence_check": record.get("persistence_check"),
    }


def build_prompt_bundle(record: dict[str, Any], world_state_entry: Optional[dict[str, Any]], bundle: str) -> PromptBundle:
    if bundle not in ABLATION_BUNDLES:
        raise ValueError(f"Unsupported ablation bundle: {bundle}")
    case_payload = _base_case_payload(record)
    if bundle == "logic_only" and isinstance(world_state_entry, dict):
        case_payload["logic_context"] = world_state_entry.get("L4_constraints")
    elif bundle == "local_graph" and isinstance(world_state_entry, dict):
        case_payload["local_context"] = {
            "L1_ego_node": world_state_entry.get("L1_ego_node"),
            "L2_labels": world_state_entry.get("L2_labels"),
            "L3_neighborhood": world_state_entry.get("L3_neighborhood"),
            "L4_constraints": world_state_entry.get("L4_constraints"),
        }

    prompt_template_name = (
        "reasoning_floor_t_box_zero_shot" if record.get("track") == "T_BOX" else "reasoning_floor_a_box_zero_shot"
    )
    prompt_template = get_prompt_template(prompt_template_name)
    return PromptBundle(
        ablation_bundle=bundle,
        prompt_name=prompt_template.name,
        prompt=prompt_template.render(case_payload),
        system_prompt=prompt_template.system_prompt,
        response_format=prompt_template.response_format_copy(),
    )


def build_track_diagnosis_prompt_bundle(
    record: dict[str, Any],
    world_state_entry: Optional[dict[str, Any]],
    bundle: str,
) -> PromptBundle:
    case_payload = _base_case_payload(record)
    if bundle == "logic_only" and isinstance(world_state_entry, dict):
        case_payload["logic_context"] = world_state_entry.get("L4_constraints")
    elif bundle == "local_graph" and isinstance(world_state_entry, dict):
        case_payload["local_context"] = {
            "L1_ego_node": world_state_entry.get("L1_ego_node"),
            "L2_labels": world_state_entry.get("L2_labels"),
            "L3_neighborhood": world_state_entry.get("L3_neighborhood"),
            "L4_constraints": world_state_entry.get("L4_constraints"),
        }
    prompt_template = get_prompt_template("reasoning_floor_track_diagnosis_zero_shot")
    return PromptBundle(
        ablation_bundle=bundle,
        prompt_name=prompt_template.name,
        prompt=prompt_template.render(case_payload),
        system_prompt=prompt_template.system_prompt,
        response_format=prompt_template.response_format_copy(),
    )


def _iter_selected_records(
    classified_path: str | Path,
    *,
    case_ids: Optional[Iterable[str]] = None,
    tracks: Optional[Iterable[str]] = None,
    max_cases: Optional[int] = None,
) -> Iterable[dict[str, Any]]:
    case_set = {case_id for case_id in case_ids if case_id} if case_ids else None
    track_set = {track for track in tracks if track} if tracks else None
    emitted = 0
    limit = max(0, max_cases) if max_cases is not None else None
    for record in iter_jsonl(classified_path):
        if not isinstance(record, dict):
            continue
        case_id = record.get("id")
        if not isinstance(case_id, str) or not case_id:
            continue
        if case_set is not None and case_id not in case_set:
            continue
        if track_set is not None and record.get("track") not in track_set:
            continue
        yield record
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def _selected_case_ids(
    classified_path: str | Path,
    *,
    case_ids: Optional[Iterable[str]] = None,
    tracks: Optional[Iterable[str]] = None,
    max_cases: Optional[int] = None,
) -> list[str]:
    return [
        record["id"]
        for record in _iter_selected_records(
            classified_path,
            case_ids=case_ids,
            tracks=tracks,
            max_cases=max_cases,
        )
    ]


def _append_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _iter_bundle_traces(output_dir: Path, bundle_list: list[str]) -> Iterable[dict[str, Any]]:
    for bundle in bundle_list:
        traces_path = output_dir / bundle / "evaluation_traces.jsonl"
        for trace in iter_jsonl(traces_path):
            if isinstance(trace, dict):
                yield trace


def _failure_taxonomy_from_traces(traces: Iterable[dict[str, Any]]) -> dict[str, float]:
    total = 0
    invalid = 0
    non_executable = 0
    diagnosis_errors = 0
    for trace in traces:
        total += 1
        if not trace.get("proposal_valid"):
            invalid += 1
        if not trace.get("proposal_executable"):
            non_executable += 1
        if not (trace.get("track_diagnosis") or {}).get("exact_track_match"):
            diagnosis_errors += 1
    if total == 0:
        return {
            "missing_or_invalid_proposal_rate": 0.0,
            "non_executable_rate": 0.0,
            "track_diagnosis_error_rate": 0.0,
        }
    return {
        "missing_or_invalid_proposal_rate": invalid / total,
        "non_executable_rate": non_executable / total,
        "track_diagnosis_error_rate": diagnosis_errors / total,
    }


def _proposal_output_name(track: str) -> str:
    return "t_box_proposals.jsonl" if track == "T_BOX" else "a_box_proposals.jsonl"


def _request_metadata(
    *,
    run_id: str,
    case_id: str,
    bundle: str,
    prompt_name: str,
    track: str | None,
    task_type: str,
    model: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "case_id": case_id,
        "ablation_bundle": bundle,
        "prompt_name": prompt_name,
        "track": track,
        "task_type": task_type,
        "model": model,
    }


def _empty_usage_payload(provider_name: str, model: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cached_tokens": None,
        "estimated_cost_usd": None,
        "input_cost_per_1m_tokens_usd": None,
        "output_cost_per_1m_tokens_usd": None,
        "model": model,
        "provider": provider_name,
        "request_metadata": metadata,
    }


def _record_request_result(
    *,
    request_info: dict[str, Any],
    raw_response: Any,
    parsed_payload: Any,
    usage: dict[str, Any],
    raw_log_fh: Any,
    manifest_fh: Any,
    a_box_fh: Any,
    t_box_fh: Any,
    track_fh: Any,
    elapsed_seconds: float | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    manifest_record = {
        "run_id": request_info.get("run_id"),
        "case_id": request_info.get("case_id"),
        "ablation_bundle": request_info.get("ablation_bundle"),
        "prompt_name": request_info.get("prompt_name"),
        "track": request_info.get("track"),
        "task_type": request_info.get("task_type"),
        "provider": usage.get("provider"),
        "model": usage.get("model"),
        "usage": _usage_block(usage, elapsed_seconds),
        "timestamp_utc": _utc_now(),
    }
    custom_id = request_info.get("custom_id")
    raw_record = {
        "run_id": request_info.get("run_id"),
        "case_id": request_info.get("case_id"),
        "ablation_bundle": request_info.get("ablation_bundle"),
        "prompt_name": request_info.get("prompt_name"),
        "track": request_info.get("track"),
        "task_type": request_info.get("task_type"),
        "raw_response": raw_response,
        "parsed_payload": parsed_payload,
    }
    if isinstance(custom_id, str):
        raw_record["custom_id"] = custom_id
        manifest_record["custom_id"] = custom_id
    if error_message:
        raw_record["error"] = error_message
    _append_jsonl_record(raw_log_fh, raw_record)

    if error_message:
        manifest_record["parse_status"] = "request_error"
        manifest_record["provider_error"] = error_message
        _append_jsonl_record(manifest_fh, manifest_record)
        return manifest_record

    try:
        if request_info.get("task_type") == "track_diagnosis":
            normalized_diagnosis = normalize_diagnosis(parsed_payload)
            _append_jsonl_record(track_fh, normalized_diagnosis.to_dict())
            manifest_record["parse_status"] = "normalized"
            manifest_record["canonical_hash"] = normalized_diagnosis.canonical_hash
        elif request_info.get("track") == "T_BOX":
            normalized = normalize_t_box_proposal(parsed_payload)
            _append_jsonl_record(t_box_fh, normalized.to_dict())
            manifest_record["parse_status"] = "normalized"
            manifest_record["canonical_hash"] = normalized.canonical_hash
        else:
            normalized = normalize_a_box_proposal(parsed_payload)
            _append_jsonl_record(a_box_fh, normalized.to_dict())
            manifest_record["parse_status"] = "normalized"
            manifest_record["canonical_hash"] = normalized.canonical_hash
    except Exception as exc:
        manifest_record["parse_status"] = "parse_error"
        manifest_record["parser_error"] = str(exc)

    _append_jsonl_record(manifest_fh, manifest_record)
    return manifest_record


def _update_cost_tracking(
    usage_records: Iterable[dict[str, Any]],
    current_estimated_cost_usd: float,
    has_cost_data: bool,
) -> tuple[float, bool]:
    updated_cost = current_estimated_cost_usd
    updated_has_cost = has_cost_data
    for usage_record in usage_records:
        estimated_cost = usage_record.get("estimated_cost_usd")
        if isinstance(estimated_cost, (int, float)):
            updated_cost += float(estimated_cost)
            updated_has_cost = True
    return updated_cost, updated_has_cost


def run_reasoning_floor(
    *,
    classified_path: str | Path,
    world_state_path: str | Path,
    output_dir: str | Path,
    provider: Optional[ModelProvider] = None,
    model_name: str | None = None,
    ablation_bundles: Iterable[str] = ABLATION_BUNDLES,
    case_ids: Optional[Iterable[str]] = None,
    selection_manifest_path: str | Path | None = None,
    tracks: Optional[Iterable[str]] = None,
    max_cases: Optional[int] = None,
    execution_mode: str | None = None,
    batch_completion_window: str = "24h",
    batch_poll_interval_seconds: float = 60.0,
) -> dict[str, Any]:
    run_started_utc = _utc_now()
    run_started_at = time.perf_counter()
    if provider is None:
        provider = create_model_provider(model_name)
    selected_model = getattr(provider, "model", None) or model_name or "unknown-model"
    selected_provider = getattr(provider, "provider_name", None) or provider.__class__.__name__.replace("ChatProvider", "").lower()

    normalized_execution_mode = (execution_mode or "").strip().lower()
    if not normalized_execution_mode:
        normalized_execution_mode = "batch" if selected_provider == "openai" else "sync"
    if normalized_execution_mode not in {"sync", "batch"}:
        raise ValueError(f"Unsupported execution mode: {execution_mode!r}")
    if normalized_execution_mode == "batch" and not isinstance(provider, BatchModelProvider):
        raise RuntimeError(
            f"Execution mode 'batch' is not supported by provider {provider.__class__.__name__}."
        )

    resolved_case_ids = resolve_case_id_filter(
        case_ids=case_ids,
        selection_manifest_path=selection_manifest_path,
    )
    selected_case_ids = _selected_case_ids(
        classified_path,
        case_ids=resolved_case_ids,
        tracks=tracks,
        max_cases=max_cases,
    )
    bundle_list = [bundle for bundle in ablation_bundles if bundle in ABLATION_BUNDLES]
    if not bundle_list:
        raise ValueError("At least one supported ablation bundle is required.")
    total_instances = len(selected_case_ids) * len(bundle_list)
    total_requests = total_instances * 2 if normalized_execution_mode == "batch" else total_instances
    refresh_every = max(1, min(1000, (total_requests + 9) // 10)) if total_requests else 1

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_dir_name = f"{run_id}_{_slugify(selected_provider)}_{_slugify(selected_model)}"
    out_dir = Path(output_dir) / run_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = out_dir / "raw_model_responses.jsonl"
    manifest_path = out_dir / "run_manifest.jsonl"

    world_state_store = WorldStateStore(Path(world_state_path), __import__("logging").getLogger("reasoning_floor"))
    world_state_store.open()
    progress = tqdm(
        total=total_requests,
        desc="reasoning-floor",
        unit="request" if normalized_execution_mode == "batch" else "case",
        disable=total_requests == 0,
    )
    pending_progress = 0
    completed_work_units = 0
    current_estimated_cost_usd = 0.0
    has_cost_data = False
    batch_summary: dict[str, Any] | None = None
    try:
        usage_manifest: list[dict[str, Any]] = []

        a_box_schema = load_a_box_schema(Path("schemas") / "verified_repair_proposal.schema.json")
        t_box_schema = load_t_box_schema(Path("schemas") / "tbox_reform_proposal.schema.json")
        track_schema = load_track_schema(Path("schemas") / "track_diagnosis.schema.json")
        del a_box_schema, t_box_schema, track_schema

        for bundle in bundle_list:
            bundle_dir = out_dir / bundle
            bundle_dir.mkdir(parents=True, exist_ok=True)
            (bundle_dir / "a_box_proposals.jsonl").touch()
            (bundle_dir / "t_box_proposals.jsonl").touch()
            (bundle_dir / "track_diagnoses.jsonl").touch()

        with open(raw_log_path, "w", encoding="utf-8") as raw_log_fh, open(
            manifest_path, "w", encoding="utf-8"
        ) as manifest_fh:
            if normalized_execution_mode == "sync":
                for bundle in bundle_list:
                    bundle_dir = out_dir / bundle
                    a_box_path = bundle_dir / "a_box_proposals.jsonl"
                    t_box_path = bundle_dir / "t_box_proposals.jsonl"
                    track_path = bundle_dir / "track_diagnoses.jsonl"

                    with open(a_box_path, "w", encoding="utf-8") as a_box_fh, open(
                        t_box_path, "w", encoding="utf-8"
                    ) as t_box_fh, open(track_path, "w", encoding="utf-8") as track_fh:
                        for record in _iter_selected_records(
                            classified_path,
                            case_ids=resolved_case_ids,
                            tracks=tracks,
                            max_cases=max_cases,
                        ):
                            case_id = record["id"]
                            world_state_entry = world_state_store.get(case_id)
                            diagnosis_bundle = build_track_diagnosis_prompt_bundle(record, world_state_entry, bundle)
                            diagnosis_metadata = _request_metadata(
                                run_id=run_id,
                                case_id=case_id,
                                bundle=bundle,
                                prompt_name=diagnosis_bundle.prompt_name,
                                track=record.get("track"),
                                task_type="track_diagnosis",
                                model=getattr(provider, "model", "unknown-model"),
                            )
                            diagnosis_started_at = time.perf_counter()
                            diagnosis_raw_response, diagnosis_payload, diagnosis_usage = provider.generate(
                                diagnosis_bundle.prompt,
                                diagnosis_bundle.system_prompt,
                                diagnosis_bundle.response_format,
                                diagnosis_metadata,
                            )
                            diagnosis_elapsed_seconds = time.perf_counter() - diagnosis_started_at
                            diagnosis_manifest_record = _record_request_result(
                                request_info=diagnosis_metadata,
                                raw_response=diagnosis_raw_response,
                                parsed_payload=diagnosis_payload,
                                usage=diagnosis_usage,
                                raw_log_fh=raw_log_fh,
                                manifest_fh=manifest_fh,
                                a_box_fh=a_box_fh,
                                t_box_fh=t_box_fh,
                                track_fh=track_fh,
                                elapsed_seconds=diagnosis_elapsed_seconds,
                            )
                            usage_manifest.append(diagnosis_manifest_record)

                            prompt_bundle = build_prompt_bundle(record, world_state_entry, bundle)
                            proposal_metadata = _request_metadata(
                                run_id=run_id,
                                case_id=case_id,
                                bundle=bundle,
                                prompt_name=prompt_bundle.prompt_name,
                                track=record.get("track"),
                                task_type="proposal",
                                model=getattr(provider, "model", "unknown-model"),
                            )
                            proposal_started_at = time.perf_counter()
                            raw_response, parsed_payload, usage = provider.generate(
                                prompt_bundle.prompt,
                                prompt_bundle.system_prompt,
                                prompt_bundle.response_format,
                                proposal_metadata,
                            )
                            proposal_elapsed_seconds = time.perf_counter() - proposal_started_at
                            proposal_manifest_record = _record_request_result(
                                request_info=proposal_metadata,
                                raw_response=raw_response,
                                parsed_payload=parsed_payload,
                                usage=usage,
                                raw_log_fh=raw_log_fh,
                                manifest_fh=manifest_fh,
                                a_box_fh=a_box_fh,
                                t_box_fh=t_box_fh,
                                track_fh=track_fh,
                                elapsed_seconds=proposal_elapsed_seconds,
                            )
                            usage_manifest.append(proposal_manifest_record)

                            current_estimated_cost_usd, has_cost_data = _update_cost_tracking(
                                (diagnosis_usage, usage),
                                current_estimated_cost_usd,
                                has_cost_data,
                            )
                            completed_work_units += 1
                            pending_progress += 1
                            if pending_progress >= refresh_every or completed_work_units == total_requests:
                                estimated_total_cost = None
                                if has_cost_data and completed_work_units > 0:
                                    estimated_total_cost = (
                                        current_estimated_cost_usd / completed_work_units
                                    ) * total_requests
                                progress.update(pending_progress)
                                progress.set_postfix(
                                    {
                                        "current_cost": _format_cost(
                                            current_estimated_cost_usd if has_cost_data else None
                                        ),
                                        "est_total_cost": _format_cost(estimated_total_cost),
                                    },
                                    refresh=True,
                                )
                                pending_progress = 0
            else:
                assert isinstance(provider, BatchModelProvider)
                batch_input_path = out_dir / "batch_input.jsonl"
                request_manifest_path = out_dir / "batch_request_manifest.jsonl"
                request_counter = 0
                with open(batch_input_path, "w", encoding="utf-8") as batch_input_fh, open(
                    request_manifest_path, "w", encoding="utf-8"
                ) as request_manifest_fh:
                    for bundle in bundle_list:
                        for record in _iter_selected_records(
                            classified_path,
                            case_ids=resolved_case_ids,
                            tracks=tracks,
                            max_cases=max_cases,
                        ):
                            case_id = record["id"]
                            world_state_entry = world_state_store.get(case_id)

                            diagnosis_bundle = build_track_diagnosis_prompt_bundle(record, world_state_entry, bundle)
                            diagnosis_metadata = _request_metadata(
                                run_id=run_id,
                                case_id=case_id,
                                bundle=bundle,
                                prompt_name=diagnosis_bundle.prompt_name,
                                track=record.get("track"),
                                task_type="track_diagnosis",
                                model=getattr(provider, "model", "unknown-model"),
                            )
                            diagnosis_custom_id = f"rf_{request_counter:09d}"
                            request_counter += 1
                            provider.write_batch_request(
                                batch_input_fh,
                                custom_id=diagnosis_custom_id,
                                prompt=diagnosis_bundle.prompt,
                                system_prompt=diagnosis_bundle.system_prompt,
                                response_format=diagnosis_bundle.response_format,
                                metadata=diagnosis_metadata,
                            )
                            _append_jsonl_record(
                                request_manifest_fh,
                                {
                                    "custom_id": diagnosis_custom_id,
                                    "metadata": diagnosis_metadata,
                                },
                            )

                            prompt_bundle = build_prompt_bundle(record, world_state_entry, bundle)
                            proposal_metadata = _request_metadata(
                                run_id=run_id,
                                case_id=case_id,
                                bundle=bundle,
                                prompt_name=prompt_bundle.prompt_name,
                                track=record.get("track"),
                                task_type="proposal",
                                model=getattr(provider, "model", "unknown-model"),
                            )
                            proposal_custom_id = f"rf_{request_counter:09d}"
                            request_counter += 1
                            provider.write_batch_request(
                                batch_input_fh,
                                custom_id=proposal_custom_id,
                                prompt=prompt_bundle.prompt,
                                system_prompt=prompt_bundle.system_prompt,
                                response_format=prompt_bundle.response_format,
                                metadata=proposal_metadata,
                            )
                            _append_jsonl_record(
                                request_manifest_fh,
                                {
                                    "custom_id": proposal_custom_id,
                                    "metadata": proposal_metadata,
                                },
                            )

                if request_counter == 0:
                    batch_execution = None
                    batch_summary = {
                        "id": None,
                        "status": "skipped_empty",
                        "completion_window": batch_completion_window,
                        "poll_interval_seconds": batch_poll_interval_seconds,
                        "elapsed_seconds": 0.0,
                        "request_counts": {"total": 0, "completed": 0, "failed": 0},
                        "output_path": None,
                        "error_path": None,
                        "input_path": str(batch_input_path),
                        "request_manifest_path": str(request_manifest_path),
                    }
                else:
                    batch_started_at = time.perf_counter()
                    batch_execution = provider.execute_batch(
                        batch_input_path,
                        request_manifest_path=request_manifest_path,
                        output_dir=out_dir,
                        completion_window=batch_completion_window,
                        poll_interval_seconds=batch_poll_interval_seconds,
                    )
                    batch_elapsed_seconds = time.perf_counter() - batch_started_at
                    batch_summary = {
                        "id": batch_execution.batch.get("id"),
                        "status": batch_execution.batch.get("status"),
                        "completion_window": batch_completion_window,
                        "poll_interval_seconds": batch_poll_interval_seconds,
                        "elapsed_seconds": round(batch_elapsed_seconds, 6),
                        "request_counts": batch_execution.batch.get("request_counts"),
                        "output_path": str(batch_execution.output_path) if batch_execution.output_path else None,
                        "error_path": str(batch_execution.error_path) if batch_execution.error_path else None,
                        "input_path": str(batch_input_path),
                        "request_manifest_path": str(request_manifest_path),
                    }

                request_map: dict[str, dict[str, Any]] = {}
                for row in iter_jsonl(request_manifest_path):
                    if not isinstance(row, dict):
                        continue
                    custom_id = row.get("custom_id")
                    metadata = row.get("metadata")
                    if isinstance(custom_id, str) and isinstance(metadata, dict):
                        request_map[custom_id] = {"custom_id": custom_id, **metadata}

                seen_custom_ids: set[str] = set()
                with ExitStack() as stack:
                    bundle_handles = {}
                    for bundle in bundle_list:
                        bundle_dir = out_dir / bundle
                        bundle_handles[bundle] = {
                            "a_box": stack.enter_context(
                                open(bundle_dir / "a_box_proposals.jsonl", "w", encoding="utf-8")
                            ),
                            "t_box": stack.enter_context(
                                open(bundle_dir / "t_box_proposals.jsonl", "w", encoding="utf-8")
                            ),
                            "track": stack.enter_context(
                                open(bundle_dir / "track_diagnoses.jsonl", "w", encoding="utf-8")
                            ),
                        }

                    result_paths = ()
                    if batch_execution is not None:
                        result_paths = (batch_execution.output_path, batch_execution.error_path)
                    for result_path in result_paths:
                        if result_path is None:
                            continue
                        for result_row in iter_jsonl(result_path):
                            if not isinstance(result_row, dict):
                                continue
                            custom_id = result_row.get("custom_id")
                            if not isinstance(custom_id, str) or custom_id in seen_custom_ids:
                                continue
                            request_info = request_map.get(custom_id)
                            if not isinstance(request_info, dict):
                                continue
                            raw_response, parsed_payload, usage, error_message = provider.parse_batch_result(
                                result_row,
                                request_info,
                            )
                            handles = bundle_handles[request_info["ablation_bundle"]]
                            manifest_record = _record_request_result(
                                request_info=request_info,
                                raw_response=raw_response,
                                parsed_payload=parsed_payload,
                                usage=usage,
                                raw_log_fh=raw_log_fh,
                                manifest_fh=manifest_fh,
                                a_box_fh=handles["a_box"],
                                t_box_fh=handles["t_box"],
                                track_fh=handles["track"],
                                elapsed_seconds=None,
                                error_message=error_message,
                            )
                            usage_manifest.append(manifest_record)
                            current_estimated_cost_usd, has_cost_data = _update_cost_tracking(
                                (usage,),
                                current_estimated_cost_usd,
                                has_cost_data,
                            )
                            seen_custom_ids.add(custom_id)
                            completed_work_units += 1
                            pending_progress += 1
                            if pending_progress >= refresh_every or completed_work_units == total_requests:
                                estimated_total_cost = None
                                if has_cost_data and completed_work_units > 0:
                                    estimated_total_cost = (
                                        current_estimated_cost_usd / completed_work_units
                                    ) * total_requests
                                progress.update(pending_progress)
                                progress.set_postfix(
                                    {
                                        "current_cost": _format_cost(
                                            current_estimated_cost_usd if has_cost_data else None
                                        ),
                                        "est_total_cost": _format_cost(estimated_total_cost),
                                    },
                                    refresh=True,
                                )
                                pending_progress = 0

                    missing_custom_ids = sorted(set(request_map) - seen_custom_ids)
                    for custom_id in missing_custom_ids:
                        request_info = request_map[custom_id]
                        handles = bundle_handles[request_info["ablation_bundle"]]
                        usage = _empty_usage_payload(selected_provider, selected_model, request_info)
                        manifest_record = _record_request_result(
                            request_info=request_info,
                            raw_response=None,
                            parsed_payload=None,
                            usage=usage,
                            raw_log_fh=raw_log_fh,
                            manifest_fh=manifest_fh,
                            a_box_fh=handles["a_box"],
                            t_box_fh=handles["t_box"],
                            track_fh=handles["track"],
                            elapsed_seconds=None,
                            error_message="Batch result was missing from both output and error files.",
                        )
                        usage_manifest.append(manifest_record)
                        completed_work_units += 1
                        pending_progress += 1
                        if pending_progress >= refresh_every or completed_work_units == total_requests:
                            progress.update(pending_progress)
                            pending_progress = 0

        for bundle in bundle_list:
            bundle_dir = out_dir / bundle
            evaluate_benchmark(
                classified_path=classified_path,
                world_state_path=world_state_path,
                a_box_proposals_path=bundle_dir / "a_box_proposals.jsonl",
                t_box_proposals_path=bundle_dir / "t_box_proposals.jsonl",
                track_diagnoses_path=bundle_dir / "track_diagnoses.jsonl",
                run_manifest_path=manifest_path,
                ablation_bundle=bundle,
                case_ids=selected_case_ids,
                selection_manifest_path=selection_manifest_path,
                out_traces_path=bundle_dir / "evaluation_traces.jsonl",
                out_summary_path=bundle_dir / "evaluation_summary.json",
                collect_traces=False,
            )

        summary = summarize_trace_iterable(
            _iter_bundle_traces(out_dir, bundle_list),
            {
                "classified_benchmark": str(classified_path),
                "world_state": str(world_state_path),
                "run_id": run_id,
                "ablation_bundles": bundle_list,
                "provider": selected_provider,
                "model": selected_model,
                "output_dir": str(out_dir),
                "selection_manifest": str(selection_manifest_path) if selection_manifest_path else None,
            },
        )
        failure_taxonomy = _failure_taxonomy_from_traces(_iter_bundle_traces(out_dir, bundle_list))
        run_elapsed_seconds = time.perf_counter() - run_started_at
        run_usage = _aggregate_run_usage(usage_manifest)
        summary["run_info"] = {
            "run_id": run_id,
            "provider": selected_provider,
            "model": selected_model,
            "output_dir": str(out_dir),
            "started_at_utc": run_started_utc,
            "elapsed_seconds": round(run_elapsed_seconds, 6),
            "generation_elapsed_seconds": run_usage["generation_elapsed_seconds"],
            "execution_mode": normalized_execution_mode,
        }
        if batch_summary is not None:
            summary["run_info"]["batch"] = batch_summary
        summary["usage"] = {
            "prompt_tokens": run_usage["prompt_tokens"],
            "completion_tokens": run_usage["completion_tokens"],
            "total_tokens": run_usage["total_tokens"],
            "cached_tokens": run_usage["cached_tokens"],
            "estimated_cost_usd": run_usage["estimated_cost_usd"],
        }
        summary["paper_summary"] = {
            "overall_success_by_class": summary.get("by_class"),
            "success_by_ablation_bundle": summary.get("by_ablation_bundle"),
            "success_by_track": summary.get("by_track"),
            "success_by_popularity_bucket": summary.get("by_popularity_bucket"),
            "track_diagnosis_by_class": {
                key: value.get("track_diagnosis_accuracy") for key, value in summary.get("by_class", {}).items()
            },
            "failure_taxonomy": failure_taxonomy,
        }
        write_json(out_dir / "reasoning_floor_summary.json", summary)
        return summary
    finally:
        if pending_progress:
            progress.update(pending_progress)
        progress.close()
        world_state_store.close()
