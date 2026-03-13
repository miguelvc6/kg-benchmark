from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from classifier import WorldStateStore
from guardian.evaluator import evaluate_benchmark, summarize_trace_iterable
from guardian.reasoning import (
    ABLATION_BUNDLES,
    PromptBundle,
    build_prompt_bundle,
    build_track_diagnosis_prompt_bundle,
)
from lib.utils import iter_jsonl, read_json

DEFAULT_CLASSIFIED_BENCHMARK = Path("data/04_classified_benchmark.jsonl")
DEFAULT_WORLD_STATE = Path("data/03_world_state.json")

LOG = logging.getLogger("reasoning_floor_viewer")


@dataclass
class CaseDebugRecord:
    case_id: str
    historical_track: Optional[str]
    proposal_type: Optional[str]
    record: Optional[dict[str, Any]]
    proposal_manifest: Optional[dict[str, Any]]
    diagnosis_manifest: Optional[dict[str, Any]]
    proposal_raw: Optional[dict[str, Any]]
    diagnosis_raw: Optional[dict[str, Any]]
    proposal_normalized: Optional[dict[str, Any]]
    diagnosis_normalized: Optional[dict[str, Any]]
    trace: Optional[dict[str, Any]]

    @property
    def proposal_parse_status(self) -> str:
        if isinstance(self.proposal_manifest, dict):
            value = self.proposal_manifest.get("parse_status")
            if isinstance(value, str) and value:
                return value
        return "missing"

    @property
    def diagnosis_parse_status(self) -> str:
        if isinstance(self.diagnosis_manifest, dict):
            value = self.diagnosis_manifest.get("parse_status")
            if isinstance(value, str) and value:
                return value
        return "missing"

    @property
    def accepted(self) -> Optional[bool]:
        if isinstance(self.trace, dict):
            value = self.trace.get("accepted")
            if isinstance(value, bool):
                return value
        return None


@dataclass
class BundleDebugData:
    reports_root: Path
    run_dir: Path
    bundle_name: str
    bundle_names: list[str]
    run_summary: Optional[dict[str, Any]]
    run_info: dict[str, Any]
    bundle_summary: Optional[dict[str, Any]]
    summary_source: str
    traces: list[dict[str, Any]]
    traces_source: str
    case_rows: list[CaseDebugRecord]
    bundle_manifest_rows: list[dict[str, Any]]
    usage_summary: dict[str, Any]
    parse_status_counts: list[dict[str, Any]]
    input_paths: dict[str, Optional[str]]
    input_sources: dict[str, str]


@dataclass
class CasePromptDebug:
    case_id: str
    bundle_name: str
    world_state_entry: Optional[dict[str, Any]]
    proposal_prompt: Optional[PromptBundle]
    diagnosis_prompt: Optional[PromptBundle]
    error: Optional[str] = None


def discover_run_directories(reports_root: str | Path) -> list[Path]:
    root = Path(reports_root)
    if not root.exists() or not root.is_dir():
        return []
    return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True)


def list_run_bundles(run_dir: str | Path) -> list[str]:
    path = Path(run_dir)
    if not path.exists() or not path.is_dir():
        return []
    bundles = [child.name for child in path.iterdir() if child.is_dir()]
    ordered = [name for name in ABLATION_BUNDLES if name in bundles]
    extras = sorted(name for name in bundles if name not in set(ordered))
    return ordered + extras


def load_bundle_debug_data(
    *,
    reports_root: str | Path,
    run_dir: str | Path,
    bundle_name: str,
    classified_benchmark: str | Path | None = None,
    world_state: str | Path | None = None,
) -> BundleDebugData:
    reports_root = Path(reports_root)
    run_dir = Path(run_dir)
    bundle_dir = run_dir / bundle_name
    bundle_names = list_run_bundles(run_dir)
    if bundle_name not in bundle_names:
        raise ValueError(f"Unknown ablation bundle {bundle_name!r} for run {run_dir}")

    run_summary = _load_optional_json(run_dir / "reasoning_floor_summary.json")
    input_paths, input_sources = _resolve_input_paths(
        run_summary,
        classified_benchmark=classified_benchmark,
        world_state=world_state,
    )
    manifest_path = run_dir / "run_manifest.jsonl"
    raw_path = run_dir / "raw_model_responses.jsonl"
    manifest_rows = _iter_jsonl_if_exists(manifest_path)
    raw_rows = _iter_jsonl_if_exists(raw_path)
    bundle_manifest_rows = [
        row for row in manifest_rows if (row.get("ablation_bundle") or None) == bundle_name
    ]
    bundle_raw_rows = [row for row in raw_rows if (row.get("ablation_bundle") or None) == bundle_name]

    diagnosis_rows = _iter_jsonl_if_exists(bundle_dir / "track_diagnoses.jsonl")
    a_box_rows = _iter_jsonl_if_exists(bundle_dir / "a_box_proposals.jsonl")
    t_box_rows = _iter_jsonl_if_exists(bundle_dir / "t_box_proposals.jsonl")

    evaluation_summary_path = bundle_dir / "evaluation_summary.json"
    traces, traces_source, bundle_summary, summary_source = _load_or_compute_evaluation(
        bundle_name=bundle_name,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        input_paths=input_paths,
        case_ids=_collect_case_ids(
            bundle_manifest_rows=bundle_manifest_rows,
            bundle_raw_rows=bundle_raw_rows,
            diagnosis_rows=diagnosis_rows,
            a_box_rows=a_box_rows,
            t_box_rows=t_box_rows,
        ),
    )
    if traces_source == "artifact" and bundle_summary is None:
        bundle_summary = summarize_trace_iterable(traces, _summary_inputs(run_dir, bundle_name, input_paths))
        summary_source = "derived_from_traces"
    elif traces_source == "artifact" and evaluation_summary_path.exists():
        bundle_summary = _load_optional_json(evaluation_summary_path)
        summary_source = "artifact"
    elif traces_source == "live" and bundle_summary is not None:
        summary_source = "live"

    if traces_source == "unavailable" and evaluation_summary_path.exists():
        bundle_summary = _load_optional_json(evaluation_summary_path)
        summary_source = "artifact"

    record_map = _load_case_records(
        input_paths.get("classified_benchmark"),
        _case_ids_from_summary_or_rows(traces, bundle_manifest_rows),
    )
    trace_map = {row["case_id"]: row for row in traces if isinstance(row, dict) and isinstance(row.get("case_id"), str)}
    diagnosis_map = {row["case_id"]: row for row in diagnosis_rows if isinstance(row.get("case_id"), str)}
    a_box_map = {row["case_id"]: row for row in a_box_rows if isinstance(row.get("case_id"), str)}
    t_box_map = {row["case_id"]: row for row in t_box_rows if isinstance(row.get("case_id"), str)}
    manifest_map = {
        (row.get("case_id"), row.get("task_type") or "proposal"): row
        for row in bundle_manifest_rows
        if isinstance(row.get("case_id"), str)
    }
    raw_map = {
        (row.get("case_id"), row.get("task_type") or "proposal"): row
        for row in bundle_raw_rows
        if isinstance(row.get("case_id"), str)
    }

    case_ids = sorted(
        {
            *record_map.keys(),
            *trace_map.keys(),
            *diagnosis_map.keys(),
            *a_box_map.keys(),
            *t_box_map.keys(),
            *(case_id for case_id, _ in manifest_map.keys() if isinstance(case_id, str)),
            *(case_id for case_id, _ in raw_map.keys() if isinstance(case_id, str)),
        }
    )
    case_rows = []
    for case_id in case_ids:
        record = record_map.get(case_id)
        proposal_manifest = manifest_map.get((case_id, "proposal"))
        diagnosis_manifest = manifest_map.get((case_id, "track_diagnosis"))
        trace = trace_map.get(case_id)
        historical_track = _historical_track(record, proposal_manifest, diagnosis_manifest, trace)
        proposal_type = _proposal_type(historical_track, trace)
        proposal_normalized = t_box_map.get(case_id) if proposal_type == "T_BOX" else a_box_map.get(case_id)
        case_rows.append(
            CaseDebugRecord(
                case_id=case_id,
                historical_track=historical_track,
                proposal_type=proposal_type,
                record=record,
                proposal_manifest=proposal_manifest,
                diagnosis_manifest=diagnosis_manifest,
                proposal_raw=raw_map.get((case_id, "proposal")),
                diagnosis_raw=raw_map.get((case_id, "track_diagnosis")),
                proposal_normalized=proposal_normalized,
                diagnosis_normalized=diagnosis_map.get(case_id),
                trace=trace,
            )
        )

    return BundleDebugData(
        reports_root=reports_root,
        run_dir=run_dir,
        bundle_name=bundle_name,
        bundle_names=bundle_names,
        run_summary=run_summary,
        run_info=_build_run_info(run_dir, run_summary, bundle_manifest_rows),
        bundle_summary=bundle_summary,
        summary_source=summary_source,
        traces=traces,
        traces_source=traces_source,
        case_rows=case_rows,
        bundle_manifest_rows=bundle_manifest_rows,
        usage_summary=_summarize_usage(bundle_manifest_rows),
        parse_status_counts=_parse_status_counts(bundle_manifest_rows),
        input_paths=input_paths,
        input_sources=input_sources,
    )


def build_case_prompt_debug(bundle_data: BundleDebugData, case_id: str) -> CasePromptDebug:
    case_row = next((row for row in bundle_data.case_rows if row.case_id == case_id), None)
    if case_row is None:
        return CasePromptDebug(
            case_id=case_id,
            bundle_name=bundle_data.bundle_name,
            world_state_entry=None,
            proposal_prompt=None,
            diagnosis_prompt=None,
            error=f"Unknown case_id {case_id!r}.",
        )
    if not isinstance(case_row.record, dict):
        return CasePromptDebug(
            case_id=case_id,
            bundle_name=bundle_data.bundle_name,
            world_state_entry=None,
            proposal_prompt=None,
            diagnosis_prompt=None,
            error="Classified benchmark record is unavailable, so prompts cannot be reconstructed.",
        )

    world_state_path = bundle_data.input_paths.get("world_state")
    world_state_entry = None
    if world_state_path and Path(world_state_path).exists() and bundle_data.bundle_name != "minimal_case":
        with WorldStateStore(Path(world_state_path), LOG) as store:
            world_state_entry = store.get(case_id)

    try:
        proposal_prompt = build_prompt_bundle(case_row.record, world_state_entry, bundle_data.bundle_name)
        diagnosis_prompt = build_track_diagnosis_prompt_bundle(
            case_row.record,
            world_state_entry,
            bundle_data.bundle_name,
        )
    except Exception as exc:
        return CasePromptDebug(
            case_id=case_id,
            bundle_name=bundle_data.bundle_name,
            world_state_entry=world_state_entry,
            proposal_prompt=None,
            diagnosis_prompt=None,
            error=str(exc),
        )
    return CasePromptDebug(
        case_id=case_id,
        bundle_name=bundle_data.bundle_name,
        world_state_entry=world_state_entry,
        proposal_prompt=proposal_prompt,
        diagnosis_prompt=diagnosis_prompt,
    )


def extract_response_content(raw_entry: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(raw_entry, dict):
        return None
    raw_response = raw_entry.get("raw_response")
    if not isinstance(raw_response, dict):
        return None
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    return None


def _load_or_compute_evaluation(
    *,
    bundle_name: str,
    bundle_dir: Path,
    manifest_path: Path,
    input_paths: dict[str, Optional[str]],
    case_ids: list[str],
) -> tuple[list[dict[str, Any]], str, Optional[dict[str, Any]], str]:
    traces_path = bundle_dir / "evaluation_traces.jsonl"
    summary_path = bundle_dir / "evaluation_summary.json"
    if traces_path.exists():
        traces = _iter_jsonl_if_exists(traces_path)
        summary = _load_optional_json(summary_path)
        return traces, "artifact", summary, "artifact" if summary is not None else "derived_from_traces"

    classified_path = input_paths.get("classified_benchmark")
    world_state_path = input_paths.get("world_state")
    if not classified_path or not world_state_path:
        return [], "unavailable", None, "unavailable"
    if not Path(classified_path).exists() or not Path(world_state_path).exists():
        return [], "unavailable", None, "unavailable"

    traces, summary = evaluate_benchmark(
        classified_path=classified_path,
        world_state_path=world_state_path,
        a_box_proposals_path=bundle_dir / "a_box_proposals.jsonl",
        t_box_proposals_path=bundle_dir / "t_box_proposals.jsonl",
        track_diagnoses_path=bundle_dir / "track_diagnoses.jsonl",
        run_manifest_path=manifest_path,
        ablation_bundle=bundle_name,
        case_ids=case_ids,
        collect_traces=True,
    )
    return traces, "live", summary, "live"


def _summary_inputs(run_dir: Path, bundle_name: str, input_paths: dict[str, Optional[str]]) -> dict[str, Any]:
    return {
        "classified_benchmark": input_paths.get("classified_benchmark"),
        "world_state": input_paths.get("world_state"),
        "a_box_proposals": str(run_dir / bundle_name / "a_box_proposals.jsonl"),
        "t_box_proposals": str(run_dir / bundle_name / "t_box_proposals.jsonl"),
        "track_diagnoses": str(run_dir / bundle_name / "track_diagnoses.jsonl"),
        "run_manifest": str(run_dir / "run_manifest.jsonl"),
        "ablation_bundle": bundle_name,
    }


def _resolve_input_paths(
    run_summary: Optional[dict[str, Any]],
    *,
    classified_benchmark: str | Path | None,
    world_state: str | Path | None,
) -> tuple[dict[str, Optional[str]], dict[str, str]]:
    summary_inputs = (run_summary or {}).get("inputs") if isinstance(run_summary, dict) else {}
    input_paths: dict[str, Optional[str]] = {}
    input_sources: dict[str, str] = {}
    for key, override, default in (
        ("classified_benchmark", classified_benchmark, DEFAULT_CLASSIFIED_BENCHMARK),
        ("world_state", world_state, DEFAULT_WORLD_STATE),
    ):
        if override:
            input_paths[key] = str(Path(override))
            input_sources[key] = "override"
            continue
        summary_value = summary_inputs.get(key) if isinstance(summary_inputs, dict) else None
        if isinstance(summary_value, str) and summary_value:
            input_paths[key] = summary_value
            input_sources[key] = "run_summary"
            continue
        default_path = Path(default)
        if default_path.exists():
            input_paths[key] = str(default_path)
            input_sources[key] = "default"
        else:
            input_paths[key] = None
            input_sources[key] = "missing"
    return input_paths, input_sources


def _build_run_info(
    run_dir: Path,
    run_summary: Optional[dict[str, Any]],
    bundle_manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_info = (run_summary or {}).get("run_info")
    if isinstance(summary_info, dict):
        return dict(summary_info)
    first_manifest = bundle_manifest_rows[0] if bundle_manifest_rows else {}
    return {
        "run_id": first_manifest.get("run_id"),
        "provider": first_manifest.get("provider"),
        "model": first_manifest.get("model"),
        "output_dir": str(run_dir),
    }


def _collect_case_ids(
    *,
    bundle_manifest_rows: list[dict[str, Any]],
    bundle_raw_rows: list[dict[str, Any]],
    diagnosis_rows: list[dict[str, Any]],
    a_box_rows: list[dict[str, Any]],
    t_box_rows: list[dict[str, Any]],
) -> list[str]:
    case_ids = {
        row.get("case_id")
        for row in bundle_manifest_rows + bundle_raw_rows + diagnosis_rows + a_box_rows + t_box_rows
        if isinstance(row.get("case_id"), str)
    }
    return sorted(case_ids)


def _case_ids_from_summary_or_rows(
    traces: list[dict[str, Any]],
    bundle_manifest_rows: list[dict[str, Any]],
) -> list[str]:
    case_ids = {
        row.get("case_id")
        for row in traces + bundle_manifest_rows
        if isinstance(row.get("case_id"), str)
    }
    return sorted(case_ids)


def _load_case_records(
    classified_path: Optional[str],
    case_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not classified_path or not case_ids:
        return {}
    path = Path(classified_path)
    if not path.exists():
        return {}
    allowed = set(case_ids)
    records = {}
    for row in iter_jsonl(path):
        if not isinstance(row, dict):
            continue
        case_id = row.get("id")
        if not isinstance(case_id, str) or case_id not in allowed:
            continue
        records[case_id] = row
    return records


def _historical_track(
    record: Optional[dict[str, Any]],
    proposal_manifest: Optional[dict[str, Any]],
    diagnosis_manifest: Optional[dict[str, Any]],
    trace: Optional[dict[str, Any]],
) -> Optional[str]:
    for source in (record, proposal_manifest, diagnosis_manifest, trace):
        if not isinstance(source, dict):
            continue
        value = source.get("track")
        if isinstance(value, str) and value:
            return value
    return None


def _proposal_type(historical_track: Optional[str], trace: Optional[dict[str, Any]]) -> Optional[str]:
    if historical_track in {"A_BOX", "T_BOX"}:
        return historical_track
    if isinstance(trace, dict):
        value = trace.get("proposal_type")
        if isinstance(value, str) and value:
            return value
    return None


def _parse_status_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter()
    for row in rows:
        task_type = row.get("task_type") if isinstance(row.get("task_type"), str) else "proposal"
        parse_status = row.get("parse_status") if isinstance(row.get("parse_status"), str) else "missing"
        counts[(task_type, parse_status)] += 1
    return [
        {"task_type": task_type, "parse_status": parse_status, "count": count}
        for (task_type, parse_status), count in sorted(counts.items())
    ]


def _summarize_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "elapsed_seconds": 0.0,
        "call_count": 0,
        "token_call_count": 0,
    }
    has_prompt = False
    has_completion = False
    has_total = False
    has_cost = False
    has_elapsed = False
    batch_pricing_flags: set[bool] = set()
    cost_modes: set[str] = set()
    cost_multipliers: set[float] = set()
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        totals["call_count"] += 1
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, int):
            totals["prompt_tokens"] += prompt_tokens
            has_prompt = True
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int):
            totals["completion_tokens"] += completion_tokens
            has_completion = True
        total_tokens = usage.get("total_tokens")
        if isinstance(total_tokens, int):
            totals["total_tokens"] += total_tokens
            totals["token_call_count"] += 1
            has_total = True
        estimated_cost = usage.get("estimated_cost_usd")
        if isinstance(estimated_cost, (int, float)):
            totals["estimated_cost_usd"] += float(estimated_cost)
            has_cost = True
        elapsed_seconds = usage.get("elapsed_seconds")
        if isinstance(elapsed_seconds, (int, float)):
            totals["elapsed_seconds"] += float(elapsed_seconds)
            has_elapsed = True
        batch_pricing_applied = usage.get("batch_pricing_applied")
        if isinstance(batch_pricing_applied, bool):
            batch_pricing_flags.add(batch_pricing_applied)
        cost_estimation_mode = usage.get("cost_estimation_mode")
        if isinstance(cost_estimation_mode, str) and cost_estimation_mode:
            cost_modes.add(cost_estimation_mode)
        cost_estimation_multiplier = usage.get("cost_estimation_multiplier")
        if isinstance(cost_estimation_multiplier, (int, float)):
            cost_multipliers.add(float(cost_estimation_multiplier))
    return {
        "prompt_tokens": totals["prompt_tokens"] if has_prompt else None,
        "completion_tokens": totals["completion_tokens"] if has_completion else None,
        "total_tokens": totals["total_tokens"] if has_total else None,
        "estimated_cost_usd": round(totals["estimated_cost_usd"], 10) if has_cost else None,
        "elapsed_seconds": round(totals["elapsed_seconds"], 6) if has_elapsed else None,
        "call_count": totals["call_count"],
        "batch_pricing_applied": next(iter(batch_pricing_flags)) if len(batch_pricing_flags) == 1 else None,
        "cost_estimation_mode": next(iter(cost_modes)) if len(cost_modes) == 1 else None,
        "cost_estimation_multiplier": next(iter(cost_multipliers)) if len(cost_multipliers) == 1 else None,
        "mean_total_tokens_per_call": (
            totals["total_tokens"] / totals["token_call_count"] if totals["token_call_count"] else None
        ),
    }


def _iter_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for row in iter_jsonl(path):
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _load_optional_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    payload = read_json(path)
    if isinstance(payload, dict):
        return payload
    return None
