from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from tqdm import tqdm

from classifier import WorldStateStore
from lib.utils import iter_jsonl
from guardian.evaluator import evaluate_benchmark, summarize_traces, write_json, write_jsonl
from guardian.model_provider import ModelProvider, create_model_provider
from guardian.patch_parser import load_schema as load_a_box_schema
from guardian.patch_parser import normalize_proposal as normalize_a_box_proposal
from guardian.track_parser import load_schema as load_track_schema
from guardian.track_parser import normalize_diagnosis
from guardian.tbox_parser import load_schema as load_t_box_schema
from guardian.tbox_parser import normalize_proposal as normalize_t_box_proposal


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


ABLATION_BUNDLES = ("minimal_case", "logic_only", "local_graph")


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered)
    return slug.strip("_") or "unknown"


def _usage_block(usage: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "estimated_cost_usd": usage.get("estimated_cost_usd"),
        "input_cost_per_1m_tokens_usd": usage.get("input_cost_per_1m_tokens_usd"),
        "output_cost_per_1m_tokens_usd": usage.get("output_cost_per_1m_tokens_usd"),
        "elapsed_seconds": round(elapsed_seconds, 6),
    }


def _aggregate_run_usage(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    estimated_cost_usd = 0.0
    elapsed_seconds = 0.0
    has_prompt = False
    has_completion = False
    has_total = False
    has_cost = False
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
        value = usage.get("estimated_cost_usd")
        if isinstance(value, (int, float)):
            estimated_cost_usd += float(value)
            has_cost = True
        value = usage.get("elapsed_seconds")
        if isinstance(value, (int, float)):
            elapsed_seconds += float(value)
    return {
        "prompt_tokens": prompt_tokens if has_prompt else None,
        "completion_tokens": completion_tokens if has_completion else None,
        "total_tokens": total_tokens if has_total else None,
        "estimated_cost_usd": round(estimated_cost_usd, 10) if has_cost else None,
        "generation_elapsed_seconds": round(elapsed_seconds, 6),
    }


def _format_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.4f}"


@dataclass
class PromptBundle:
    ablation_bundle: str
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

    if record.get("track") == "T_BOX":
        system_prompt = (
            "You are producing a zero-shot T-box reform proposal for a benchmark case. "
            "Return JSON only. Do not use tools or external retrieval."
        )
    else:
        system_prompt = (
            "You are producing a zero-shot A-box repair proposal for a benchmark case. "
            "Return JSON only. Do not use tools or external retrieval."
        )

    prompt = json.dumps(case_payload, ensure_ascii=False, indent=2, sort_keys=True)
    response_format = {"type": "json_object"}
    return PromptBundle(ablation_bundle=bundle, prompt=prompt, system_prompt=system_prompt, response_format=response_format)


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
    system_prompt = (
        "You are performing a zero-shot benchmark diagnosis task. "
        "Decide whether the historical case should be treated as A_BOX, T_BOX, or AMBIGUOUS. "
        "Return JSON only with case_id, predicted_track, optional confidence, and optional rationale. "
        "Do not use tools or external retrieval."
    )
    prompt = json.dumps(case_payload, ensure_ascii=False, indent=2, sort_keys=True)
    return PromptBundle(ablation_bundle=bundle, prompt=prompt, system_prompt=system_prompt, response_format={"type": "json_object"})


def _load_records(classified_path: str | Path) -> list[dict[str, Any]]:
    records = []
    for record in iter_jsonl(classified_path):
        if isinstance(record, dict) and isinstance(record.get("id"), str):
            records.append(record)
    return records


def _select_records(
    records: list[dict[str, Any]],
    *,
    case_ids: Optional[Iterable[str]] = None,
    tracks: Optional[Iterable[str]] = None,
    max_cases: Optional[int] = None,
) -> list[dict[str, Any]]:
    selected = records
    if case_ids:
        case_set = {case_id for case_id in case_ids if case_id}
        selected = [record for record in selected if record.get("id") in case_set]
    if tracks:
        track_set = {track for track in tracks if track}
        selected = [record for record in selected if record.get("track") in track_set]
    if max_cases is not None:
        selected = selected[: max(0, max_cases)]
    return selected


def _proposal_output_name(track: str) -> str:
    return "t_box_proposals.jsonl" if track == "T_BOX" else "a_box_proposals.jsonl"


def run_reasoning_floor(
    *,
    classified_path: str | Path,
    world_state_path: str | Path,
    output_dir: str | Path,
    provider: Optional[ModelProvider] = None,
    model_name: str | None = None,
    ablation_bundles: Iterable[str] = ABLATION_BUNDLES,
    case_ids: Optional[Iterable[str]] = None,
    tracks: Optional[Iterable[str]] = None,
    max_cases: Optional[int] = None,
) -> dict[str, Any]:
    run_started_utc = _utc_now()
    run_started_at = time.perf_counter()
    if provider is None:
        provider = create_model_provider(model_name)
    selected_model = getattr(provider, "model", None) or model_name or "unknown-model"
    selected_provider = getattr(provider, "provider_name", None) or provider.__class__.__name__.replace("ChatProvider", "").lower()

    records = _select_records(
        _load_records(classified_path),
        case_ids=case_ids,
        tracks=tracks,
        max_cases=max_cases,
    )
    bundle_list = [bundle for bundle in ablation_bundles if bundle in ABLATION_BUNDLES]
    if not bundle_list:
        raise ValueError("At least one supported ablation bundle is required.")
    total_instances = len(records) * len(bundle_list)
    refresh_every = max(1, min(1000, (total_instances + 9) // 10)) if total_instances else 1

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_dir_name = f"{run_id}_{_slugify(selected_provider)}_{_slugify(selected_model)}"
    out_dir = Path(output_dir) / run_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = out_dir / "raw_model_responses.jsonl"
    manifest_path = out_dir / "run_manifest.jsonl"

    world_state_store = WorldStateStore(Path(world_state_path), __import__("logging").getLogger("reasoning_floor"))
    world_state_store.open()
    progress = tqdm(
        total=total_instances,
        desc="reasoning-floor",
        unit="case",
        disable=total_instances == 0,
    )
    pending_progress = 0
    completed_instances = 0
    current_estimated_cost_usd = 0.0
    has_cost_data = False
    try:
        raw_logs: list[dict[str, Any]] = []
        manifest: list[dict[str, Any]] = []
        combined_traces: list[dict[str, Any]] = []

        a_box_schema = load_a_box_schema(Path("schemas") / "verified_repair_proposal.schema.json")
        t_box_schema = load_t_box_schema(Path("schemas") / "tbox_reform_proposal.schema.json")
        track_schema = load_track_schema(Path("schemas") / "track_diagnosis.schema.json")
        del a_box_schema, t_box_schema, track_schema

        for bundle in bundle_list:
            bundle_dir = out_dir / bundle
            bundle_dir.mkdir(parents=True, exist_ok=True)
            a_box_proposals: list[dict[str, Any]] = []
            t_box_proposals: list[dict[str, Any]] = []
            track_diagnoses: list[dict[str, Any]] = []

            for record in records:
                case_id = record["id"]
                world_state_entry = world_state_store.get(case_id)
                diagnosis_bundle = build_track_diagnosis_prompt_bundle(record, world_state_entry, bundle)
                diagnosis_metadata = {
                    "run_id": run_id,
                    "case_id": case_id,
                    "ablation_bundle": bundle,
                    "track": record.get("track"),
                    "task_type": "track_diagnosis",
                    "model": getattr(provider, "model", "unknown-model"),
                }
                diagnosis_started_at = time.perf_counter()
                diagnosis_raw_response, diagnosis_payload, diagnosis_usage = provider.generate(
                    diagnosis_bundle.prompt,
                    diagnosis_bundle.system_prompt,
                    diagnosis_bundle.response_format,
                    diagnosis_metadata,
                )
                diagnosis_elapsed_seconds = time.perf_counter() - diagnosis_started_at
                diagnosis_manifest_record = {
                    "run_id": run_id,
                    "case_id": case_id,
                    "ablation_bundle": bundle,
                    "track": record.get("track"),
                    "task_type": "track_diagnosis",
                    "provider": diagnosis_usage.get("provider"),
                    "model": diagnosis_usage.get("model"),
                    "usage": _usage_block(diagnosis_usage, diagnosis_elapsed_seconds),
                    "timestamp_utc": _utc_now(),
                }
                raw_logs.append(
                    {
                        "run_id": run_id,
                        "case_id": case_id,
                        "ablation_bundle": bundle,
                        "track": record.get("track"),
                        "task_type": "track_diagnosis",
                        "raw_response": diagnosis_raw_response,
                        "parsed_payload": diagnosis_payload,
                    }
                )
                try:
                    normalized_diagnosis = normalize_diagnosis(diagnosis_payload)
                    track_diagnoses.append(normalized_diagnosis.to_dict())
                    diagnosis_manifest_record["parse_status"] = "normalized"
                    diagnosis_manifest_record["canonical_hash"] = normalized_diagnosis.canonical_hash
                except Exception as exc:
                    diagnosis_manifest_record["parse_status"] = "parse_error"
                    diagnosis_manifest_record["parser_error"] = str(exc)
                manifest.append(diagnosis_manifest_record)

                prompt_bundle = build_prompt_bundle(record, world_state_entry, bundle)
                metadata = {
                    "run_id": run_id,
                    "case_id": case_id,
                    "ablation_bundle": bundle,
                    "track": record.get("track"),
                    "task_type": "proposal",
                    "model": getattr(provider, "model", "unknown-model"),
                }
                proposal_started_at = time.perf_counter()
                raw_response, parsed_payload, usage = provider.generate(
                    prompt_bundle.prompt,
                    prompt_bundle.system_prompt,
                    prompt_bundle.response_format,
                    metadata,
                )
                proposal_elapsed_seconds = time.perf_counter() - proposal_started_at
                manifest_record = {
                    "run_id": run_id,
                    "case_id": case_id,
                    "ablation_bundle": bundle,
                    "track": record.get("track"),
                    "task_type": "proposal",
                    "provider": usage.get("provider"),
                    "model": usage.get("model"),
                    "usage": _usage_block(usage, proposal_elapsed_seconds),
                    "timestamp_utc": _utc_now(),
                }
                raw_logs.append(
                    {
                        "run_id": run_id,
                        "case_id": case_id,
                        "ablation_bundle": bundle,
                        "track": record.get("track"),
                        "task_type": "proposal",
                        "raw_response": raw_response,
                        "parsed_payload": parsed_payload,
                    }
                )
                try:
                    if record.get("track") == "T_BOX":
                        normalized = normalize_t_box_proposal(parsed_payload)
                        t_box_proposals.append(normalized.to_dict())
                    else:
                        normalized = normalize_a_box_proposal(parsed_payload)
                        a_box_proposals.append(normalized.to_dict())
                    manifest_record["parse_status"] = "normalized"
                    manifest_record["canonical_hash"] = normalized.canonical_hash
                except Exception as exc:
                    manifest_record["parse_status"] = "parse_error"
                    manifest_record["parser_error"] = str(exc)
                manifest.append(manifest_record)

                for usage_record in (diagnosis_usage, usage):
                    estimated_cost = usage_record.get("estimated_cost_usd")
                    if isinstance(estimated_cost, (int, float)):
                        current_estimated_cost_usd += float(estimated_cost)
                        has_cost_data = True
                completed_instances += 1
                pending_progress += 1
                if pending_progress >= refresh_every or completed_instances == total_instances:
                    estimated_total_cost = None
                    if has_cost_data and completed_instances > 0:
                        estimated_total_cost = (current_estimated_cost_usd / completed_instances) * total_instances
                    progress.update(pending_progress)
                    progress.set_postfix(
                        {
                            "current_cost": _format_cost(current_estimated_cost_usd if has_cost_data else None),
                            "est_total_cost": _format_cost(estimated_total_cost),
                        },
                        refresh=True,
                    )
                    pending_progress = 0

            write_jsonl(raw_log_path, raw_logs)
            write_jsonl(manifest_path, manifest)
            a_box_path = bundle_dir / "a_box_proposals.jsonl"
            t_box_path = bundle_dir / "t_box_proposals.jsonl"
            track_path = bundle_dir / "track_diagnoses.jsonl"
            traces_path = bundle_dir / "evaluation_traces.jsonl"
            summary_path = bundle_dir / "evaluation_summary.json"
            write_jsonl(a_box_path, a_box_proposals)
            write_jsonl(t_box_path, t_box_proposals)
            write_jsonl(track_path, track_diagnoses)

            traces, _ = evaluate_benchmark(
                classified_path=classified_path,
                world_state_path=world_state_path,
                a_box_proposals_path=a_box_path,
                t_box_proposals_path=t_box_path,
                track_diagnoses_path=track_path,
                run_manifest_path=manifest_path,
                ablation_bundle=bundle,
                case_ids=[record["id"] for record in records],
                out_traces_path=traces_path,
                out_summary_path=summary_path,
            )
            combined_traces.extend(traces)

        write_jsonl(raw_log_path, raw_logs)
        write_jsonl(manifest_path, manifest)
        summary = summarize_traces(
            combined_traces,
            {
                "classified_benchmark": str(classified_path),
                "world_state": str(world_state_path),
                "run_id": run_id,
                "ablation_bundles": bundle_list,
                "provider": selected_provider,
                "model": selected_model,
                "output_dir": str(out_dir),
            },
        )
        run_elapsed_seconds = time.perf_counter() - run_started_at
        run_usage = _aggregate_run_usage(manifest)
        summary["run_info"] = {
            "run_id": run_id,
            "provider": selected_provider,
            "model": selected_model,
            "output_dir": str(out_dir),
            "started_at_utc": run_started_utc,
            "elapsed_seconds": round(run_elapsed_seconds, 6),
            "generation_elapsed_seconds": run_usage["generation_elapsed_seconds"],
        }
        summary["usage"] = {
            "prompt_tokens": run_usage["prompt_tokens"],
            "completion_tokens": run_usage["completion_tokens"],
            "total_tokens": run_usage["total_tokens"],
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
            "failure_taxonomy": {
                "missing_or_invalid_proposal_rate": (
                    0.0
                    if not combined_traces
                    else sum(1 for trace in combined_traces if not trace.get("proposal_valid")) / len(combined_traces)
                ),
                "non_executable_rate": (
                    0.0
                    if not combined_traces
                    else sum(1 for trace in combined_traces if not trace.get("proposal_executable")) / len(combined_traces)
                ),
                "track_diagnosis_error_rate": (
                    0.0
                    if not combined_traces
                    else sum(
                        1
                        for trace in combined_traces
                        if not (trace.get("track_diagnosis") or {}).get("exact_track_match")
                    )
                    / len(combined_traces)
                ),
            },
        }
        write_json(out_dir / "reasoning_floor_summary.json", summary)
        return summary
    finally:
        if pending_progress:
            progress.update(pending_progress)
        progress.close()
        world_state_store.close()
