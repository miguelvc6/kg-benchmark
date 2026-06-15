from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from tqdm import tqdm

from classifier import VIOLATION_TO_CONSTRAINT_MAP, WorldStateStore
from guardian.evaluator import evaluate_benchmark, summarize_trace_iterable, write_json
from guardian.model_provider import BatchModelProvider, ModelProvider, create_model_provider
from guardian.patch_parser import load_schema as load_a_box_schema
from guardian.patch_parser import normalize_proposal as normalize_a_box_proposal
from guardian.prompts import get_prompt_template
from guardian.tbox_parser import KNOWN_CONSTRAINT_TYPE_QIDS
from guardian.tbox_parser import load_schema as load_t_box_schema
from guardian.tbox_parser import normalize_proposal as normalize_t_box_proposal
from guardian.track_parser import load_schema as load_track_schema
from guardian.track_parser import normalize_diagnosis
from lib.benchmark_selection import resolve_case_id_filter
from lib.repair_state import pre_repair_target_state, reconstruct_properties_with_pre_repair_target
from lib.utils import iter_jsonl, normalize_text


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_hash(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


ABLATION_BUNDLES = ("minimal_case", "logic_only", "local_graph")
DIAGNOSIS_ABLATION_BUNDLES = (
    "diagnosis_minimal",
    "diagnosis_logic_neutral",
    "diagnosis_local_neutral",
)
OPENAI_BATCH_COST_ESTIMATION_MULTIPLIER = 0.5
EVALUATION_IN_MEMORY_CASE_THRESHOLD = 10_000
GENERATION_IN_MEMORY_CASE_THRESHOLD = 10_000
PROPERTY_SCOPE_CONSTRAINT_QID = "Q53869507"
ALLOWED_ENTITY_TYPES_CONSTRAINT_QID = "Q52004125"
UNROUTABLE_TRACK = "UNROUTABLE"
RETRYABLE_BATCH_STATUS_CODES = {408, 409, 429}
RUN_CONFIG_FILENAME = "run_config.json"


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
        "batch_pricing_applied": usage.get("batch_pricing_applied"),
        "cost_estimation_mode": usage.get("cost_estimation_mode"),
        "cost_estimation_multiplier": usage.get("cost_estimation_multiplier"),
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


def _resolved_path_str(path_value: str | Path | None) -> str | None:
    if path_value in (None, ""):
        return None
    return str(Path(path_value).resolve())


def _load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _normalize_string_list(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None
    normalized = [value for value in values if isinstance(value, str) and value]
    return normalized or []


def _build_run_config_payload(
    *,
    run_id: str,
    out_dir: Path,
    provider_name: str,
    model_name: str,
    openai_reasoning_effort: str | None,
    execution_mode: str,
    proposal_track_mode: str,
    oracle_diagnosis_mode: str,
    classified_path: str | Path,
    world_state_path: str | Path,
    selection_manifest_path: str | Path | None,
    tracks: Iterable[str] | None,
    max_cases: int | None,
    ablation_bundles: Iterable[str],
    selected_case_ids: Iterable[str],
    expected_request_count: int,
    skipped_diagnosis_count: int,
    generation_strategy: str,
    selected_generation_records_path: Path | None,
    started_at_utc: str,
) -> dict[str, Any]:
    normalized_case_ids = [case_id for case_id in selected_case_ids if isinstance(case_id, str) and case_id]
    return {
        "run_id": run_id,
        "output_dir": str(out_dir.resolve()),
        "provider": provider_name,
        "model": model_name,
        "openai_reasoning_effort": openai_reasoning_effort,
        "execution_mode": execution_mode,
        "proposal_track_mode": proposal_track_mode,
        "oracle_diagnosis_mode": oracle_diagnosis_mode,
        "classified_benchmark": _resolved_path_str(classified_path),
        "world_state": _resolved_path_str(world_state_path),
        "selection_manifest": _resolved_path_str(selection_manifest_path),
        "tracks": sorted(_normalize_string_list(tracks) or []),
        "max_cases": max_cases,
        "ablation_bundles": [bundle for bundle in ablation_bundles if isinstance(bundle, str) and bundle],
        "selected_case_ids": normalized_case_ids,
        "selected_case_count": len(normalized_case_ids),
        "expected_request_count": expected_request_count,
        "skipped_diagnosis_count": skipped_diagnosis_count,
        "generation_strategy": generation_strategy,
        "selected_generation_records_path": (
            str(selected_generation_records_path.resolve())
            if isinstance(selected_generation_records_path, Path)
            else None
        ),
        "started_at_utc": started_at_utc,
    }


def _validate_resume_run_config(
    existing_config: dict[str, Any] | None,
    expected_config: dict[str, Any],
) -> None:
    if not isinstance(existing_config, dict):
        return
    mismatches: list[str] = []
    for key in (
        "provider",
        "model",
        "openai_reasoning_effort",
        "execution_mode",
        "proposal_track_mode",
        "oracle_diagnosis_mode",
        "classified_benchmark",
        "world_state",
        "selection_manifest",
        "tracks",
        "max_cases",
        "ablation_bundles",
        "selected_case_ids",
    ):
        if key == "oracle_diagnosis_mode" and existing_config.get(key) is None and expected_config.get(key) == "run":
            continue
        if existing_config.get(key) != expected_config.get(key):
            mismatches.append(key)
    if mismatches:
        details = ", ".join(mismatches)
        raise ValueError(
            "Resume run configuration does not match the existing run directory. "
            f"Mismatched fields: {details}."
        )


def _load_existing_manifest_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(manifest_path):
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _manifest_rows_by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str | None, str, str], dict[str, Any]]:
    lookup: dict[tuple[str | None, str, str], dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            continue
        bundle = row.get("ablation_bundle")
        bundle_key = bundle if isinstance(bundle, str) and bundle else None
        task_type = row.get("task_type")
        task_key = task_type if isinstance(task_type, str) and task_type else "proposal"
        lookup[(bundle_key, case_id, task_key)] = row
    return lookup


def _manifest_row_for_task(
    manifest_lookup: dict[tuple[str | None, str, str], dict[str, Any]],
    *,
    bundle: str,
    case_id: str,
    task_type: str,
) -> dict[str, Any] | None:
    return manifest_lookup.get((bundle, case_id, task_type))


def _load_existing_track_diagnosis_rows(
    out_dir: Path,
    bundle_list: Iterable[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for bundle in bundle_list:
        diagnosis_path = out_dir / bundle / "track_diagnoses.jsonl"
        if not diagnosis_path.exists():
            continue
        for row in iter_jsonl(diagnosis_path):
            case_id = row.get("case_id") if isinstance(row, dict) else None
            if isinstance(case_id, str) and case_id:
                rows_by_key[(bundle, case_id)] = row
    return rows_by_key


def _completed_case_units(
    manifest_lookup: dict[tuple[str | None, str, str], dict[str, Any]],
    *,
    bundle_list: Iterable[str],
    selected_case_ids: Iterable[str],
) -> int:
    completed = 0
    for bundle in bundle_list:
        for case_id in selected_case_ids:
            if (
                _manifest_row_for_task(manifest_lookup, bundle=bundle, case_id=case_id, task_type="track_diagnosis")
                is not None
                and _manifest_row_for_task(manifest_lookup, bundle=bundle, case_id=case_id, task_type="proposal")
                is not None
            ):
                completed += 1
    return completed


def _routing_info_from_existing_diagnosis(
    *,
    diagnosis_manifest_row: dict[str, Any],
    diagnosis_row: dict[str, Any] | None,
) -> dict[str, Any]:
    context_audit = diagnosis_manifest_row.get("context_audit") or {}
    if diagnosis_manifest_row.get("parse_status") != "normalized":
        return {
            "proposal_track_used": UNROUTABLE_TRACK,
            "routing_source": "diagnosis_prediction",
            "context_audit": context_audit,
        }
    if not isinstance(diagnosis_row, dict):
        raise RuntimeError(
            "Cannot resume diagnosis-routed proposal generation because an existing normalized diagnosis row is "
            "missing."
        )
    predicted_track = diagnosis_row.get("predicted_track")
    if predicted_track not in {"A_BOX", "T_BOX", "AMBIGUOUS"}:
        predicted_track = UNROUTABLE_TRACK
    return {
        "proposal_track_used": predicted_track,
        "routing_source": "diagnosis_prediction",
        "context_audit": context_audit,
    }


def _emit_runtime_status(progress_bar: Any, message: str) -> None:
    formatted = f"[{_utc_now()}] [reasoning-floor] {message}"
    if progress_bar is not None:
        progress_bar.write(formatted)
        return
    print(formatted)


def _disable_generation_progress(*, execution_mode: str, total_requests: int) -> bool:
    return total_requests == 0 or execution_mode == "batch"


def _batch_pricing_applies(*, provider_name: str, execution_mode: str) -> bool:
    return provider_name.strip().lower() == "openai" and execution_mode == "batch"


def _apply_cost_estimation_policy(
    usage: dict[str, Any],
    *,
    provider_name: str,
    execution_mode: str,
) -> dict[str, Any]:
    adjusted = dict(usage)
    batch_pricing_applied = _batch_pricing_applies(
        provider_name=provider_name,
        execution_mode=execution_mode,
    )
    multiplier = OPENAI_BATCH_COST_ESTIMATION_MULTIPLIER if batch_pricing_applied else 1.0
    if batch_pricing_applied:
        for field in (
            "estimated_cost_usd",
            "input_cost_per_1m_tokens_usd",
            "output_cost_per_1m_tokens_usd",
        ):
            value = adjusted.get(field)
            if isinstance(value, (int, float)):
                adjusted[field] = round(float(value) * multiplier, 10)
    adjusted["batch_pricing_applied"] = batch_pricing_applied
    adjusted["cost_estimation_mode"] = (
        "openai_batch_discount_applied" if batch_pricing_applied else "provider_default"
    )
    adjusted["cost_estimation_multiplier"] = multiplier
    return adjusted


def _env_positive_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _ollama_model_size_billions(model_name: str | None) -> float | None:
    if not isinstance(model_name, str):
        return None
    match = re.search(r":(\d+(?:\.\d+)?)b(?:$|[^a-z0-9])", model_name.strip().lower())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _default_parallel_workers(provider_name: str, model_name: str | None) -> tuple[int, str]:
    env_value = _env_positive_int("REASONING_FLOOR_PARALLEL_WORKERS")
    if env_value is not None:
        return env_value, "env"

    normalized_provider = provider_name.strip().lower()
    if normalized_provider == "ollama":
        ollama_parallel_limit = _env_positive_int("OLLAMA_NUM_PARALLEL")
        model_size = _ollama_model_size_billions(model_name)
        recommended = 2 if isinstance(model_size, float) and model_size <= 8.0 else 1
        if ollama_parallel_limit is not None:
            return min(recommended, ollama_parallel_limit), "heuristic_capped_by_ollama_num_parallel"
        return recommended, "ollama_model_heuristic"

    return 4, "generic_default"


@dataclass(frozen=True)
class RequestExecutionResult:
    request_info: dict[str, Any]
    raw_response: Any
    parsed_payload: Any
    usage: dict[str, Any]
    elapsed_seconds: float
    error_message: str | None = None


@dataclass(frozen=True)
class CasePipelineOutcome:
    request_results: list[RequestExecutionResult]
    skipped_diagnoses: list[dict[str, Any]]
    skipped_proposals: list[dict[str, Any]]


def _batch_requests_by_custom_id(path: Path) -> dict[str, dict[str, Any]]:
    request_map: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if not isinstance(row, dict):
            continue
        custom_id = row.get("custom_id")
        if isinstance(custom_id, str) and custom_id:
            request_map[custom_id] = row
    return request_map


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _batch_request_prompt_components(request_row: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    body = request_row.get("body")
    if not isinstance(body, dict):
        return None
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    system_prompt = ""
    prompt = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = _message_content_text(message.get("content"))
        if role == "system" and not system_prompt:
            system_prompt = content
        elif role == "user" and not prompt:
            prompt = content
    response_format = body.get("response_format")
    if not isinstance(response_format, dict):
        response_format = {}
    return prompt, system_prompt, dict(response_format)


def _batch_result_status_code(result_row: dict[str, Any] | None) -> int | None:
    if not isinstance(result_row, dict):
        return None
    response = result_row.get("response")
    status_code = response.get("status_code") if isinstance(response, dict) else None
    return status_code if isinstance(status_code, int) else None


def _is_retryable_batch_error(result_row: dict[str, Any] | None, error_message: str | None) -> bool:
    if not isinstance(error_message, str) or not error_message.strip():
        return False
    status_code = _batch_result_status_code(result_row)
    if status_code is None:
        return False
    return status_code in RETRYABLE_BATCH_STATUS_CODES or status_code >= 500


def _execute_case_requests(
    provider: ModelProvider,
    requests_to_run: list[tuple[PromptBundle, dict[str, Any]]],
    *,
    provider_name: str,
    execution_mode: str,
) -> list[RequestExecutionResult]:
    results: list[RequestExecutionResult] = []
    for prompt_bundle, request_info in requests_to_run:
        started_at = time.perf_counter()
        try:
            raw_response, parsed_payload, usage = provider.generate(
                prompt_bundle.prompt,
                prompt_bundle.system_prompt,
                prompt_bundle.response_format,
                request_info,
            )
            usage = _apply_cost_estimation_policy(
                usage,
                provider_name=provider_name,
                execution_mode=execution_mode,
            )
            error_message = None
            elapsed_seconds = time.perf_counter() - started_at
        except Exception as exc:
            raw_response = None
            parsed_payload = None
            fallback_model = getattr(provider, "model", None) or request_info.get("model") or "unknown"
            usage = _apply_cost_estimation_policy(
                _empty_usage_payload(provider_name, str(fallback_model), request_info),
                provider_name=provider_name,
                execution_mode=execution_mode,
            )
            error_text = str(exc).strip() or exc.__class__.__name__
            error_message = f"{exc.__class__.__name__}: {error_text}"
            elapsed_seconds = time.perf_counter() - started_at
        results.append(
            RequestExecutionResult(
                request_info=request_info,
                raw_response=raw_response,
                parsed_payload=parsed_payload,
                usage=usage,
                elapsed_seconds=elapsed_seconds,
                error_message=error_message,
            )
        )
    return results


@dataclass
class PromptBundle:
    ablation_bundle: str
    prompt_name: str
    prompt: str
    system_prompt: str
    response_format: dict[str, Any]
    context_audit: dict[str, Any] = field(default_factory=dict)


def _iter_leaf_strings(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, bool):
        yield "true" if value else "false"
        return
    if isinstance(value, (int, float)):
        yield str(value)
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_leaf_strings(item)
        return
    if isinstance(value, dict):
        for key in ("qid", "pid", "property_id", "target_qid", "target_pid", "id", "raw", "value"):
            if key in value:
                yield from _iter_leaf_strings(value[key])
        for item in value.values():
            yield from _iter_leaf_strings(item)


def _sanitized_violation_context(record: dict[str, Any]) -> dict[str, Any]:
    violation_context = record.get("violation_context")
    if not isinstance(violation_context, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key in (
        "report_violation_type",
        "report_violation_type_normalized",
        "report_violation_type_raw",
        "report_violation_type_qids",
        "report_page_title",
        "value",
        "value_labels_en",
        "value_descriptions_en",
    ):
        value = violation_context.get(key)
        if value is not None:
            sanitized[key] = value
    return sanitized


def _constraint_type_qid(constraint: dict[str, Any]) -> str | None:
    if not isinstance(constraint, dict):
        return None
    constraint_type = constraint.get("constraint_type")
    if not isinstance(constraint_type, dict):
        return None
    qid = constraint_type.get("qid")
    if isinstance(qid, str) and qid:
        return qid
    return None


def _synthetic_target_values(record: dict[str, Any]) -> list[str]:
    return pre_repair_target_state(record).values


def _synthetic_target_label_entries(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return pre_repair_target_state(record).entity_label_entries()


def _repair_target_constraint_type_qids(record: dict[str, Any]) -> list[str]:
    repair_target = record.get("repair_target")
    if not isinstance(repair_target, dict):
        return []
    constraint_delta = repair_target.get("constraint_delta")
    if not isinstance(constraint_delta, dict):
        return []
    qids: list[str] = []
    for value in constraint_delta.get("changed_constraint_types", []):
        if isinstance(value, str) and value and value not in qids:
            qids.append(value)
    for key in ("signature_before", "signature_after", "old_constraints", "new_constraints"):
        signature = constraint_delta.get(key)
        if not isinstance(signature, list):
            continue
        for entry in signature:
            if not isinstance(entry, dict):
                continue
            constraint_qid = entry.get("constraint_qid")
            if isinstance(constraint_qid, str) and constraint_qid and constraint_qid not in qids:
                qids.append(constraint_qid)
    return qids


def _world_state_constraint_type_qids(world_state_entry: Optional[dict[str, Any]]) -> list[str]:
    if not isinstance(world_state_entry, dict):
        return []
    l4_constraints = world_state_entry.get("L4_constraints")
    if not isinstance(l4_constraints, dict):
        return []
    constraints = l4_constraints.get("constraints")
    if not isinstance(constraints, list):
        return []
    qids: list[str] = []
    for constraint in constraints:
        constraint_qid = _constraint_type_qid(constraint)
        if isinstance(constraint_qid, str) and constraint_qid not in qids:
            qids.append(constraint_qid)
    return qids


def _t_box_constraint_type_qids(record: dict[str, Any], world_state_entry: Optional[dict[str, Any]]) -> list[str]:
    qids: list[str] = sorted(KNOWN_CONSTRAINT_TYPE_QIDS)
    mapped_qid = _mapped_constraint_qid(record)
    if isinstance(mapped_qid, str) and mapped_qid not in qids:
        qids.append(mapped_qid)
    for value in _repair_target_constraint_type_qids(record):
        if value not in qids:
            qids.append(value)
    for value in _world_state_constraint_type_qids(world_state_entry):
        if value not in qids:
            qids.append(value)
    return qids


def _violation_values(record: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(_iter_leaf_strings(_sanitized_violation_context(record).get("value"))))


def _mapped_constraint_qid(record: dict[str, Any]) -> str | None:
    violation_context = _sanitized_violation_context(record)
    raw_value = violation_context.get("report_violation_type_normalized") or violation_context.get(
        "report_violation_type"
    )
    if not isinstance(raw_value, str):
        return None
    normalized_mapping = {normalize_text(key): value for key, value in VIOLATION_TO_CONSTRAINT_MAP.items()}
    return normalized_mapping.get(normalize_text(raw_value))


def _constraint_mentions_tokens(constraint: dict[str, Any], tokens: set[str]) -> bool:
    if not tokens:
        return False
    normalized_tokens = {normalize_text(token) for token in tokens if token}
    if not normalized_tokens:
        return False
    for leaf in _iter_leaf_strings(constraint):
        normalized_leaf = normalize_text(leaf)
        if not normalized_leaf:
            continue
        if normalized_leaf in normalized_tokens:
            return True
        if any(token in normalized_leaf for token in normalized_tokens):
            return True
    return False


def _fallback_constraints(constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_qids: set[str] = set()
    property_scope = next(
        (
            constraint
            for constraint in constraints
            if _constraint_type_qid(constraint) == PROPERTY_SCOPE_CONSTRAINT_QID
        ),
        None,
    )
    if property_scope is not None:
        selected.append(property_scope)
        seen_qids.add(PROPERTY_SCOPE_CONSTRAINT_QID)
    for constraint in constraints:
        qid = _constraint_type_qid(constraint) or f"__fallback_{len(seen_qids)}"
        if qid in seen_qids:
            continue
        selected.append(constraint)
        seen_qids.add(qid)
        if len(selected) >= 5:
            break
    return selected


def _constraint_family_inventory(constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        constraint_type = constraint.get("constraint_type")
        constraint_qid = _constraint_type_qid(constraint)
        fallback_key = json.dumps(constraint_type, ensure_ascii=False, sort_keys=True) if constraint_type else ""
        key = constraint_qid or fallback_key
        if not key or key in seen:
            continue
        seen.add(key)
        entry: dict[str, Any] = {}
        if constraint_qid:
            entry["constraint_qid"] = constraint_qid
        if isinstance(constraint_type, dict):
            label = constraint_type.get("label") or constraint_type.get("label_en")
            if isinstance(label, str) and label:
                entry["label"] = label
        if entry:
            inventory.append(entry)
    return inventory


def _t_box_pre_reform_constraints_payload(
    record: dict[str, Any],
    constraints_payload: dict[str, Any],
    valid_constraints: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], dict[str, Any]] | None:
    if record.get("track") != "T_BOX":
        return None
    repair_target = record.get("repair_target") if isinstance(record.get("repair_target"), dict) else {}
    constraint_delta = (
        repair_target.get("constraint_delta") if isinstance(repair_target.get("constraint_delta"), dict) else {}
    )
    signature_before = constraint_delta.get("signature_before") or constraint_delta.get("old_constraints")
    audit = {
        "constraint_count_before": len(valid_constraints),
        "constraint_count_after": 0,
        "temporal_policy": None,
    }
    if isinstance(signature_before, list) and signature_before:
        before_constraints = [constraint for constraint in signature_before if isinstance(constraint, dict)]
        pruned_payload = {
            key: value
            for key, value in constraints_payload.items()
            if key not in {"constraints", "constraint_count", "property_revision_id"}
        }
        pruned_payload["constraints"] = before_constraints
        audit["constraint_count_after"] = len(before_constraints)
        audit["temporal_policy"] = "pre_change_signature_before"
        return pruned_payload, audit

    compact_payload = {
        "constraint_family_inventory": _constraint_family_inventory(valid_constraints),
        "violation_context": _sanitized_violation_context(record),
    }
    audit["constraint_count_after"] = len(compact_payload["constraint_family_inventory"])
    audit["temporal_policy"] = "compact_inventory_no_pre_change_signature"
    return compact_payload, audit


def _pruned_constraints_payload(
    record: dict[str, Any],
    world_state_entry: Optional[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    constraints_payload = {}
    if isinstance(world_state_entry, dict):
        raw_payload = world_state_entry.get("L4_constraints")
        if isinstance(raw_payload, dict):
            constraints_payload = dict(raw_payload)
    constraints = constraints_payload.get("constraints")
    if not isinstance(constraints, list):
        constraints = []
    valid_constraints = [constraint for constraint in constraints if isinstance(constraint, dict)]
    audit = {
        "constraint_count_before": len(valid_constraints),
        "constraint_count_after": 0,
    }
    t_box_pre_reform_payload = _t_box_pre_reform_constraints_payload(record, constraints_payload, valid_constraints)
    if t_box_pre_reform_payload is not None:
        return t_box_pre_reform_payload
    if not valid_constraints:
        if constraints_payload:
            constraints_payload["constraints"] = []
            return constraints_payload, audit
        return None, audit

    mapped_constraint_qid = _mapped_constraint_qid(record)
    referenced_tokens = set(_violation_values(record)) | set(_synthetic_target_values(record))
    kept_constraints: list[dict[str, Any]] = []
    for constraint in valid_constraints:
        constraint_qid = _constraint_type_qid(constraint)
        if mapped_constraint_qid and constraint_qid == mapped_constraint_qid:
            kept_constraints.append(constraint)
            continue
        if constraint_qid in {PROPERTY_SCOPE_CONSTRAINT_QID, ALLOWED_ENTITY_TYPES_CONSTRAINT_QID}:
            kept_constraints.append(constraint)
            continue
        if _constraint_mentions_tokens(constraint, referenced_tokens):
            kept_constraints.append(constraint)

    if not kept_constraints:
        kept_constraints = _fallback_constraints(valid_constraints)

    pruned_payload = dict(constraints_payload)
    pruned_payload["constraints"] = kept_constraints
    audit["constraint_count_after"] = len(kept_constraints)
    return pruned_payload, audit


def _pruned_l1_ego_node(
    record: dict[str, Any],
    world_state_entry: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not isinstance(world_state_entry, dict):
        return None
    l1_node = world_state_entry.get("L1_ego_node")
    if not isinstance(l1_node, dict):
        return None
    target_pid = record.get("property")
    sanitized = {
        "qid": l1_node.get("qid"),
        "label": l1_node.get("label"),
        "description": l1_node.get("description"),
    }
    properties = reconstruct_properties_with_pre_repair_target(record, l1_node.get("properties"))
    if isinstance(target_pid, str) and target_pid in properties:
        sanitized["properties"] = {target_pid: properties.get(target_pid)}
    return {key: value for key, value in sanitized.items() if value not in (None, {}, [])}


def _collect_reference_ids(value: Any) -> set[str]:
    return {token for token in _iter_leaf_strings(value) if token.startswith(("Q", "P"))}


def _edge_matches_references(edge: dict[str, Any], target_pid: str | None, references: set[str]) -> bool:
    if not isinstance(edge, dict):
        return False
    property_id = edge.get("property_id") or edge.get("pid")
    if isinstance(target_pid, str) and property_id == target_pid:
        return False
    return bool(_collect_reference_ids(edge) & references)


def _pruned_l2_labels(
    labels_payload: Any,
    references: set[str],
    *,
    synthetic_entities: Optional[dict[str, dict[str, Any]]] = None,
) -> tuple[Any, int]:
    fallback_entities = {
        key: value
        for key, value in (synthetic_entities or {}).items()
        if key in references and isinstance(value, dict)
    }
    if not isinstance(labels_payload, dict):
        if not fallback_entities:
            return labels_payload, 0
        return {"entities": fallback_entities}, len(fallback_entities)
    entities = labels_payload.get("entities")
    if isinstance(entities, dict):
        kept = {key: value for key, value in entities.items() if key in references}
        for key, value in fallback_entities.items():
            existing = kept.get(key)
            if isinstance(existing, dict):
                merged = dict(existing)
                for field_name, field_value in value.items():
                    if merged.get(field_name) in (None, "") and field_value is not None:
                        merged[field_name] = field_value
                kept[key] = merged
            else:
                kept[key] = dict(value)
        return {**labels_payload, "entities": kept}, len(kept)
    kept = {key: value for key, value in labels_payload.items() if key in references}
    for key, value in fallback_entities.items():
        if key not in kept:
            kept[key] = dict(value)
    return kept, len(kept)


def _pruned_local_context(
    record: dict[str, Any],
    world_state_entry: Optional[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    l4_constraints, constraint_audit = _pruned_constraints_payload(record, world_state_entry)
    if not isinstance(world_state_entry, dict):
        audit = {
            **constraint_audit,
            "edge_count_after": 0,
            "label_count_after": 0,
        }
        return None, audit

    l1_ego_node = _pruned_l1_ego_node(record, world_state_entry)
    references = _collect_reference_ids(l1_ego_node) | _collect_reference_ids(l4_constraints)
    references.add(record.get("qid")) if isinstance(record.get("qid"), str) else None
    if isinstance(record.get("property"), str):
        references.add(record["property"])

    outgoing_edges = []
    l3_payload = world_state_entry.get("L3_neighborhood")
    if isinstance(l3_payload, dict):
        for edge in l3_payload.get("outgoing_edges", []):
            if _edge_matches_references(edge, record.get("property"), references):
                outgoing_edges.append(edge)
                references.update(_collect_reference_ids(edge))

    l2_payload, label_count_after = _pruned_l2_labels(
        world_state_entry.get("L2_labels"),
        references,
        synthetic_entities=_synthetic_target_label_entries(record),
    )

    local_context = {
        "L1_ego_node": l1_ego_node,
        "L2_labels": l2_payload,
        "L3_neighborhood": {"outgoing_edges": outgoing_edges},
        "L4_constraints": l4_constraints,
    }
    audit = {
        **constraint_audit,
        "edge_count_after": len(outgoing_edges),
        "label_count_after": label_count_after,
    }
    return local_context, audit


_DIAGNOSIS_FORBIDDEN_CONTEXT_KEYS = {
    "classification",
    "repair_target",
    "persistence_check",
    "popularity",
    "sitelinks_count",
    "track",
    "target_constraint_is_changed",
    "changed_constraint_types",
    "signature_before",
    "signature_after",
    "constraint_delta",
}


def _strip_diagnosis_forbidden_context(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_diagnosis_forbidden_context(item)
            for key, item in value.items()
            if key not in _DIAGNOSIS_FORBIDDEN_CONTEXT_KEYS
        }
    if isinstance(value, list):
        return [_strip_diagnosis_forbidden_context(item) for item in value]
    return value


def _diagnosis_neutral_constraints_payload(
    record: dict[str, Any],
    world_state_entry: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    constraints_payload: dict[str, Any] = {}
    if isinstance(world_state_entry, dict) and isinstance(world_state_entry.get("L4_constraints"), dict):
        constraints_payload = dict(world_state_entry["L4_constraints"])

    raw_constraints = constraints_payload.get("constraints")
    valid_constraints = [constraint for constraint in raw_constraints if isinstance(constraint, dict)] if isinstance(raw_constraints, list) else []
    mapped_constraint_qid = _mapped_constraint_qid(record)
    referenced_tokens = set(_violation_values(record))
    kept_constraints: list[dict[str, Any]] = []
    for constraint in valid_constraints:
        constraint_qid = _constraint_type_qid(constraint)
        if mapped_constraint_qid and constraint_qid == mapped_constraint_qid:
            kept_constraints.append(constraint)
            continue
        if constraint_qid in {PROPERTY_SCOPE_CONSTRAINT_QID, ALLOWED_ENTITY_TYPES_CONSTRAINT_QID}:
            kept_constraints.append(constraint)
            continue
        if _constraint_mentions_tokens(constraint, referenced_tokens):
            kept_constraints.append(constraint)

    if not kept_constraints and valid_constraints:
        kept_constraints = _fallback_constraints(valid_constraints)

    payload = {
        "constraints": _strip_diagnosis_forbidden_context(kept_constraints),
        "constraint_family_inventory": _strip_diagnosis_forbidden_context(
            _constraint_family_inventory(valid_constraints)
        ),
    }
    audit = {
        "constraint_count_before": len(valid_constraints),
        "constraint_count_after": len(kept_constraints),
        "temporal_policy": "diagnosis_track_agnostic_current_context",
    }
    return payload, audit


def _diagnosis_neutral_l1_ego_node(
    record: dict[str, Any],
    world_state_entry: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(world_state_entry, dict) or not isinstance(world_state_entry.get("L1_ego_node"), dict):
        return {}
    l1_node = world_state_entry["L1_ego_node"]
    target_pid = record.get("property")
    properties = l1_node.get("properties")
    kept_properties: dict[str, Any] = {}
    if isinstance(properties, dict):
        for pid, values in properties.items():
            if isinstance(target_pid, str) and pid == target_pid:
                continue
            kept_properties[pid] = values
    sanitized = {
        "qid": l1_node.get("qid"),
        "label": l1_node.get("label"),
        "description": l1_node.get("description"),
        "properties": kept_properties,
    }
    return {
        key: _strip_diagnosis_forbidden_context(value)
        for key, value in sanitized.items()
        if value not in (None, {}, [])
    }


def _diagnosis_neutral_local_context(
    record: dict[str, Any],
    world_state_entry: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    l4_constraints, constraint_audit = _diagnosis_neutral_constraints_payload(record, world_state_entry)
    l1_ego_node = _diagnosis_neutral_l1_ego_node(record, world_state_entry)
    references = _collect_reference_ids(l1_ego_node) | _collect_reference_ids(l4_constraints)
    if isinstance(record.get("qid"), str):
        references.add(record["qid"])
    if isinstance(record.get("property"), str):
        references.add(record["property"])

    outgoing_edges: list[dict[str, Any]] = []
    if isinstance(world_state_entry, dict) and isinstance(world_state_entry.get("L3_neighborhood"), dict):
        for edge in world_state_entry["L3_neighborhood"].get("outgoing_edges", []):
            if _edge_matches_references(edge, record.get("property"), references):
                sanitized_edge = _strip_diagnosis_forbidden_context(edge)
                outgoing_edges.append(sanitized_edge)
                references.update(_collect_reference_ids(sanitized_edge))

    l2_payload: Any = {}
    label_count_after = 0
    if isinstance(world_state_entry, dict):
        l2_payload, label_count_after = _pruned_l2_labels(world_state_entry.get("L2_labels"), references)
        l2_payload = _strip_diagnosis_forbidden_context(l2_payload)

    local_context = {
        "L1_ego_node": l1_ego_node,
        "L2_labels": l2_payload,
        "L3_neighborhood": {"outgoing_edges": outgoing_edges},
        "L4_constraints": l4_constraints,
    }
    audit = {
        **constraint_audit,
        "edge_count_after": len(outgoing_edges),
        "label_count_after": label_count_after,
    }
    return local_context, audit


def _sanitized_case_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "qid": record.get("qid"),
        "property": record.get("property"),
        "labels_en": record.get("labels_en"),
        "violation_context": _sanitized_violation_context(record),
    }


def prompt_visible_case_id(raw_case_id: str, index: int | None = None) -> str:
    if isinstance(index, int) and index > 0:
        return f"case_{index:06d}"
    return f"case_{_stable_hash('prompt_visible_case_id', raw_case_id)[:12]}"


def _model_visible_case_payload(case_payload: dict[str, Any], visible_case_id: str | None) -> dict[str, Any]:
    if not visible_case_id:
        return case_payload
    visible_payload = dict(case_payload)
    visible_payload["id"] = visible_case_id
    return visible_payload


def _bundle_payload_and_audit(
    record: dict[str, Any],
    world_state_entry: Optional[dict[str, Any]],
    bundle: str,
    *,
    visible_case_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if bundle not in ABLATION_BUNDLES:
        raise ValueError(f"Unsupported ablation bundle: {bundle}")
    case_payload = _model_visible_case_payload(_sanitized_case_payload(record), visible_case_id)
    if bundle == "minimal_case":
        return case_payload, {"constraint_count_before": 0, "constraint_count_after": 0}
    if bundle == "logic_only":
        logic_context, audit = _pruned_constraints_payload(record, world_state_entry)
        case_payload["logic_context"] = logic_context
        return case_payload, audit
    local_context, audit = _pruned_local_context(record, world_state_entry)
    case_payload["local_context"] = local_context
    return case_payload, audit


def _diagnosis_bundle_payload_and_audit(
    record: dict[str, Any],
    world_state_entry: Optional[dict[str, Any]],
    bundle: str,
    *,
    visible_case_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if bundle not in DIAGNOSIS_ABLATION_BUNDLES:
        raise ValueError(f"Unsupported diagnosis ablation bundle: {bundle}")
    case_payload = _model_visible_case_payload(_sanitized_case_payload(record), visible_case_id)
    case_payload = _strip_diagnosis_forbidden_context(case_payload)
    if bundle == "diagnosis_minimal":
        return case_payload, {
            "constraint_count_before": 0,
            "constraint_count_after": 0,
            "temporal_policy": "diagnosis_track_agnostic_minimal",
        }
    if bundle == "diagnosis_logic_neutral":
        logic_context, audit = _diagnosis_neutral_constraints_payload(record, world_state_entry)
        case_payload["logic_context"] = logic_context
        return case_payload, audit
    local_context, audit = _diagnosis_neutral_local_context(record, world_state_entry)
    case_payload["local_context"] = local_context
    return case_payload, audit


def build_prompt_bundle(
    record: dict[str, Any],
    world_state_entry: Optional[dict[str, Any]],
    bundle: str,
    *,
    proposal_track: str | None = None,
    visible_case_id: str | None = None,
) -> PromptBundle:
    case_payload, context_audit = _bundle_payload_and_audit(
        record,
        world_state_entry,
        bundle,
        visible_case_id=visible_case_id or prompt_visible_case_id(str(record.get("id") or "")),
    )
    effective_track = proposal_track or record.get("track")
    prompt_template_name = (
        "reasoning_floor_t_box_zero_shot" if effective_track == "T_BOX" else "reasoning_floor_a_box_zero_shot"
    )
    prompt_template = get_prompt_template(prompt_template_name)
    return PromptBundle(
        ablation_bundle=bundle,
        prompt_name=prompt_template.name,
        prompt=prompt_template.render(case_payload),
        system_prompt=prompt_template.system_prompt,
        response_format=prompt_template.response_format_copy(),
        context_audit=context_audit,
    )


def build_track_diagnosis_prompt_bundle(
    record: dict[str, Any],
    world_state_entry: Optional[dict[str, Any]],
    bundle: str,
    *,
    visible_case_id: str | None = None,
) -> PromptBundle:
    builder = _diagnosis_bundle_payload_and_audit if bundle in DIAGNOSIS_ABLATION_BUNDLES else _bundle_payload_and_audit
    case_payload, context_audit = builder(
        record,
        world_state_entry,
        bundle,
        visible_case_id=visible_case_id or prompt_visible_case_id(str(record.get("id") or "")),
    )
    prompt_template = get_prompt_template("reasoning_floor_track_diagnosis_zero_shot")
    return PromptBundle(
        ablation_bundle=bundle,
        prompt_name=prompt_template.name,
        prompt=prompt_template.render(case_payload),
        system_prompt=prompt_template.system_prompt,
        response_format=prompt_template.response_format_copy(),
        context_audit=context_audit,
    )


@dataclass(frozen=True)
class MaterializedGenerationSelection:
    case_ids: list[str]
    records: list[dict[str, Any]] | None = None
    records_path: Path | None = None
    strategy: str = "in_memory"


def _track_filter_set(tracks: Optional[Iterable[str]]) -> set[str] | None:
    if not tracks:
        return None
    return {track for track in tracks if isinstance(track, str) and track}


def _collect_selected_records_in_order(
    classified_path: str | Path,
    *,
    case_ids: Optional[Iterable[str]] = None,
    tracks: Optional[Iterable[str]] = None,
    max_cases: Optional[int] = None,
) -> list[dict[str, Any]]:
    ordered_case_ids = [case_id for case_id in case_ids if isinstance(case_id, str) and case_id] if case_ids else None
    track_set = _track_filter_set(tracks)
    limit = max(0, max_cases) if max_cases is not None else None
    if limit == 0:
        return []
    if ordered_case_ids is not None and track_set is None and limit is not None:
        ordered_case_ids = ordered_case_ids[:limit]

    if ordered_case_ids is None:
        selected_records: list[dict[str, Any]] = []
        for record in iter_jsonl(classified_path):
            if not isinstance(record, dict):
                continue
            case_id = record.get("id")
            if not isinstance(case_id, str) or not case_id:
                continue
            if track_set is not None and record.get("track") not in track_set:
                continue
            selected_records.append(record)
            if limit is not None and len(selected_records) >= limit:
                break
        return selected_records

    remaining_case_ids = set(ordered_case_ids)
    matched_records: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(classified_path):
        if not isinstance(record, dict):
            continue
        case_id = record.get("id")
        if not isinstance(case_id, str) or case_id not in remaining_case_ids:
            continue
        if track_set is not None and record.get("track") not in track_set:
            continue
        matched_records[case_id] = record
        remaining_case_ids.remove(case_id)
        if not remaining_case_ids:
            break

    ordered_records = [matched_records[case_id] for case_id in ordered_case_ids if case_id in matched_records]
    if limit is not None:
        ordered_records = ordered_records[:limit]
    return ordered_records


def _iter_selected_records(
    classified_path: str | Path,
    *,
    case_ids: Optional[Iterable[str]] = None,
    tracks: Optional[Iterable[str]] = None,
    max_cases: Optional[int] = None,
) -> Iterable[dict[str, Any]]:
    yield from _collect_selected_records_in_order(
        classified_path,
        case_ids=case_ids,
        tracks=tracks,
        max_cases=max_cases,
    )


def _selected_case_ids(
    classified_path: str | Path,
    *,
    case_ids: Optional[Iterable[str]] = None,
    tracks: Optional[Iterable[str]] = None,
    max_cases: Optional[int] = None,
) -> list[str]:
    return [
        record["id"]
        for record in _collect_selected_records_in_order(
            classified_path,
            case_ids=case_ids,
            tracks=tracks,
            max_cases=max_cases,
        )
    ]


def _write_record_subset(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(destination, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def _materialize_generation_selection(
    classified_path: str | Path,
    *,
    case_ids: Optional[Iterable[str]],
    tracks: Optional[Iterable[str]],
    max_cases: Optional[int],
    output_dir: str | Path,
) -> MaterializedGenerationSelection:
    ordered_records = _collect_selected_records_in_order(
        classified_path,
        case_ids=case_ids,
        tracks=tracks,
        max_cases=max_cases,
    )
    ordered_case_ids = [
        record["id"]
        for record in ordered_records
        if isinstance(record, dict) and isinstance(record.get("id"), str)
    ]
    if len(ordered_records) <= GENERATION_IN_MEMORY_CASE_THRESHOLD:
        return MaterializedGenerationSelection(
            case_ids=ordered_case_ids,
            records=ordered_records,
            records_path=None,
            strategy="in_memory",
        )

    selected_records_path = Path(output_dir) / "selected_generation_records.jsonl"
    _write_record_subset(selected_records_path, ordered_records)
    return MaterializedGenerationSelection(
        case_ids=ordered_case_ids,
        records=None,
        records_path=selected_records_path,
        strategy="stream_from_file",
    )


def _iter_materialized_generation_records(selection: MaterializedGenerationSelection) -> Iterable[dict[str, Any]]:
    if selection.records is not None:
        yield from selection.records
        return
    if selection.records_path is None:
        return
    for record in iter_jsonl(selection.records_path):
        if isinstance(record, dict):
            yield record


def _load_selected_records_for_evaluation(
    classified_path: str | Path,
    selected_case_ids: Iterable[str],
) -> list[dict[str, Any]]:
    ordered_case_ids = [case_id for case_id in selected_case_ids if isinstance(case_id, str) and case_id]
    if not ordered_case_ids:
        return []
    case_id_set = set(ordered_case_ids)
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(classified_path):
        if not isinstance(record, dict):
            continue
        case_id = record.get("id")
        if not isinstance(case_id, str) or case_id not in case_id_set:
            continue
        records_by_id[case_id] = record
    return [records_by_id[case_id] for case_id in ordered_case_ids if case_id in records_by_id]


def _write_selected_records_for_evaluation(
    classified_path: str | Path,
    selected_case_ids: Iterable[str],
    destination_path: str | Path,
) -> int:
    records = _load_selected_records_for_evaluation(classified_path, selected_case_ids)
    return _write_record_subset(destination_path, records)


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
    diagnosis_skipped = 0
    proposal_parse_errors = 0
    proposal_request_errors = 0
    diagnosis_request_errors = 0
    parser_errors: dict[str, int] = {}
    for trace in traces:
        total += 1
        if not trace.get("proposal_valid"):
            invalid += 1
        if not trace.get("proposal_executable"):
            non_executable += 1
        if trace.get("parse_status") == "parse_error":
            proposal_parse_errors += 1
            parser_error = trace.get("details", {}).get("parser_error")
            if isinstance(parser_error, str) and parser_error.strip():
                parser_errors[parser_error.strip()] = parser_errors.get(parser_error.strip(), 0) + 1
        if trace.get("parse_status") == "request_error":
            proposal_request_errors += 1
        diagnosis = trace.get("track_diagnosis") or {}
        if diagnosis.get("parse_status") == "request_error":
            diagnosis_request_errors += 1
        if diagnosis.get("skipped") or diagnosis.get("parse_status") in {"skipped", "not_run"}:
            diagnosis_skipped += 1
        elif not diagnosis.get("exact_track_match"):
            diagnosis_errors += 1
    if total == 0:
        return {
            "missing_or_invalid_proposal_rate": 0.0,
            "non_executable_rate": 0.0,
            "proposal_parse_error_rate": 0.0,
            "proposal_request_error_rate": 0.0,
            "track_diagnosis_request_error_rate": 0.0,
            "track_diagnosis_error_rate": 0.0,
            "track_diagnosis_skipped_rate": 0.0,
            "proposal_parse_errors_by_message": {},
        }
    return {
        "missing_or_invalid_proposal_rate": invalid / total,
        "non_executable_rate": non_executable / total,
        "proposal_parse_error_rate": proposal_parse_errors / total,
        "proposal_request_error_rate": proposal_request_errors / total,
        "track_diagnosis_request_error_rate": diagnosis_request_errors / total,
        "track_diagnosis_error_rate": diagnosis_errors / total,
        "track_diagnosis_skipped_rate": diagnosis_skipped / total,
        "proposal_parse_errors_by_message": parser_errors,
    }


def _proposal_output_name(track: str) -> str:
    return "t_box_proposals.jsonl" if track == "T_BOX" else "a_box_proposals.jsonl"


def _request_metadata(
    *,
    run_id: str,
    case_id: str,
    visible_case_id: str | None,
    bundle: str,
    prompt_name: str,
    historical_track: str | None,
    task_type: str,
    model: str,
    proposal_track_used: str | None = None,
    routing_source: str | None = None,
    context_audit: Optional[dict[str, Any]] = None,
    t_box_constraint_type_qids: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "case_id": case_id,
        "visible_case_id": visible_case_id,
        "ablation_bundle": bundle,
        "prompt_name": prompt_name,
        "track": historical_track,
        "historical_track": historical_track,
        "task_type": task_type,
        "model": model,
        "proposal_track_used": proposal_track_used,
        "routing_source": routing_source,
        "context_audit": dict(context_audit or {}),
    }
    if t_box_constraint_type_qids is not None:
        payload["t_box_constraint_type_qids"] = list(t_box_constraint_type_qids)
    return payload


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


def _prepare_payload_for_case_id(raw_payload: Any, case_id: Any, visible_case_id: Any = None) -> Any:
    if not isinstance(raw_payload, dict):
        return raw_payload
    payload = dict(raw_payload)
    if isinstance(visible_case_id, str) and payload.get("case_id") == visible_case_id:
        payload["case_id"] = case_id
    if (not isinstance(payload.get("case_id"), str) or not payload.get("case_id", "").strip()) and isinstance(
        case_id, str
    ):
        payload["case_id"] = case_id
    return payload


def _proposal_track_for_request(request_info: dict[str, Any]) -> str | None:
    proposal_track = request_info.get("proposal_track_used")
    if isinstance(proposal_track, str) and proposal_track:
        return proposal_track
    historical_track = request_info.get("historical_track")
    if isinstance(historical_track, str) and historical_track:
        return historical_track
    fallback_track = request_info.get("track")
    if isinstance(fallback_track, str) and fallback_track:
        return fallback_track
    return None


def _routed_track_from_diagnosis_payload(parsed_payload: Any, case_id: str, visible_case_id: str | None = None) -> str:
    try:
        normalized = normalize_diagnosis(_prepare_payload_for_case_id(parsed_payload, case_id, visible_case_id))
    except Exception:
        return UNROUTABLE_TRACK
    predicted_track = normalized.predicted_track
    if predicted_track in {"A_BOX", "T_BOX", "AMBIGUOUS"}:
        return predicted_track
    return UNROUTABLE_TRACK


def _record_skipped_proposal_result(
    *,
    request_info: dict[str, Any],
    usage: dict[str, Any],
    raw_log_fh: Any,
    manifest_fh: Any,
    parse_status: str,
    skip_reason: str,
) -> dict[str, Any]:
    manifest_record = {
        "run_id": request_info.get("run_id"),
        "case_id": request_info.get("case_id"),
        "visible_case_id": request_info.get("visible_case_id"),
        "ablation_bundle": request_info.get("ablation_bundle"),
        "prompt_name": request_info.get("prompt_name"),
        "track": request_info.get("track"),
        "historical_track": request_info.get("historical_track"),
        "proposal_track_used": request_info.get("proposal_track_used"),
        "routing_source": request_info.get("routing_source"),
        "context_audit": request_info.get("context_audit") or {},
        "task_type": request_info.get("task_type"),
        "provider": usage.get("provider"),
        "model": usage.get("model"),
        "usage": _usage_block(usage, None),
        "timestamp_utc": _utc_now(),
        "parse_status": parse_status,
        "skip_reason": skip_reason,
    }
    raw_record = {
        "run_id": request_info.get("run_id"),
        "case_id": request_info.get("case_id"),
        "visible_case_id": request_info.get("visible_case_id"),
        "ablation_bundle": request_info.get("ablation_bundle"),
        "prompt_name": request_info.get("prompt_name"),
        "track": request_info.get("track"),
        "historical_track": request_info.get("historical_track"),
        "proposal_track_used": request_info.get("proposal_track_used"),
        "routing_source": request_info.get("routing_source"),
        "task_type": request_info.get("task_type"),
        "raw_response": None,
        "parsed_payload": None,
        "skip_reason": skip_reason,
    }
    _append_jsonl_record(raw_log_fh, raw_record)
    _append_jsonl_record(manifest_fh, manifest_record)
    return manifest_record


def _record_skipped_diagnosis_result(
    *,
    request_info: dict[str, Any],
    usage: dict[str, Any],
    raw_log_fh: Any,
    manifest_fh: Any,
    parse_status: str,
    skip_reason: str,
) -> dict[str, Any]:
    manifest_record = {
        "run_id": request_info.get("run_id"),
        "case_id": request_info.get("case_id"),
        "visible_case_id": request_info.get("visible_case_id"),
        "ablation_bundle": request_info.get("ablation_bundle"),
        "prompt_name": request_info.get("prompt_name"),
        "track": request_info.get("track"),
        "historical_track": request_info.get("historical_track"),
        "proposal_track_used": request_info.get("proposal_track_used"),
        "routing_source": request_info.get("routing_source"),
        "context_audit": request_info.get("context_audit") or {},
        "task_type": request_info.get("task_type"),
        "provider": usage.get("provider"),
        "model": usage.get("model"),
        "usage": _usage_block(usage, None),
        "timestamp_utc": _utc_now(),
        "parse_status": parse_status,
        "skip_reason": skip_reason,
    }
    raw_record = {
        "run_id": request_info.get("run_id"),
        "case_id": request_info.get("case_id"),
        "visible_case_id": request_info.get("visible_case_id"),
        "ablation_bundle": request_info.get("ablation_bundle"),
        "prompt_name": request_info.get("prompt_name"),
        "track": request_info.get("track"),
        "historical_track": request_info.get("historical_track"),
        "proposal_track_used": request_info.get("proposal_track_used"),
        "routing_source": request_info.get("routing_source"),
        "task_type": request_info.get("task_type"),
        "raw_response": None,
        "parsed_payload": None,
        "skip_reason": skip_reason,
    }
    _append_jsonl_record(raw_log_fh, raw_record)
    _append_jsonl_record(manifest_fh, manifest_record)
    return manifest_record


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
    recovery_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_record = {
        "run_id": request_info.get("run_id"),
        "case_id": request_info.get("case_id"),
        "visible_case_id": request_info.get("visible_case_id"),
        "ablation_bundle": request_info.get("ablation_bundle"),
        "prompt_name": request_info.get("prompt_name"),
        "track": request_info.get("track"),
        "historical_track": request_info.get("historical_track"),
        "proposal_track_used": request_info.get("proposal_track_used"),
        "routing_source": request_info.get("routing_source"),
        "context_audit": request_info.get("context_audit") or {},
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
        "visible_case_id": request_info.get("visible_case_id"),
        "ablation_bundle": request_info.get("ablation_bundle"),
        "prompt_name": request_info.get("prompt_name"),
        "track": request_info.get("track"),
        "historical_track": request_info.get("historical_track"),
        "proposal_track_used": request_info.get("proposal_track_used"),
        "routing_source": request_info.get("routing_source"),
        "task_type": request_info.get("task_type"),
        "raw_response": raw_response,
        "parsed_payload": parsed_payload,
    }
    if isinstance(custom_id, str):
        raw_record["custom_id"] = custom_id
        manifest_record["custom_id"] = custom_id
    if isinstance(recovery_details, dict) and recovery_details:
        raw_record["recovery"] = dict(recovery_details)
        manifest_record["recovery"] = dict(recovery_details)
    if error_message:
        raw_record["error"] = error_message
    _append_jsonl_record(raw_log_fh, raw_record)

    if error_message:
        manifest_record["parse_status"] = "request_error"
        manifest_record["provider_error"] = error_message
        _append_jsonl_record(manifest_fh, manifest_record)
        return manifest_record

    try:
        normalization_payload = _prepare_payload_for_case_id(
            parsed_payload,
            request_info.get("case_id"),
            request_info.get("visible_case_id"),
        )
        if request_info.get("task_type") == "track_diagnosis":
            normalized_diagnosis = normalize_diagnosis(normalization_payload)
            _append_jsonl_record(track_fh, normalized_diagnosis.to_dict())
            manifest_record["parse_status"] = "normalized"
            manifest_record["canonical_hash"] = normalized_diagnosis.canonical_hash
        elif _proposal_track_for_request(request_info) == "T_BOX":
            normalized = normalize_t_box_proposal(
                normalization_payload,
                constraint_type_qids=request_info.get("t_box_constraint_type_qids"),
            )
            _append_jsonl_record(t_box_fh, normalized.to_dict())
            manifest_record["parse_status"] = "normalized"
            manifest_record["canonical_hash"] = normalized.canonical_hash
        else:
            normalized = normalize_a_box_proposal(normalization_payload)
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
    resume_run_dir: str | Path | None = None,
    provider: Optional[ModelProvider] = None,
    model_name: str | None = None,
    model_endpoint: str | None = None,
    ablation_bundles: Iterable[str] = ABLATION_BUNDLES,
    case_ids: Optional[Iterable[str]] = None,
    selection_manifest_path: str | Path | None = None,
    tracks: Optional[Iterable[str]] = None,
    max_cases: Optional[int] = None,
    execution_mode: str | None = None,
    parallel_workers: int | None = None,
    batch_completion_window: str = "24h",
    batch_poll_interval_seconds: float = 60.0,
    proposal_track_mode: str = "oracle",
    oracle_diagnosis_mode: str | None = None,
) -> dict[str, Any]:
    run_started_utc = _utc_now()
    run_started_at = time.perf_counter()
    if provider is None:
        provider = create_model_provider(model_name, model_endpoint=model_endpoint)
    selected_model = getattr(provider, "model", None) or model_name or "unknown-model"
    selected_provider = (
        getattr(provider, "provider_name", None) or provider.__class__.__name__.replace("ChatProvider", "").lower()
    )
    openai_reasoning_effort = getattr(provider, "reasoning_effort", None) if selected_provider == "openai" else None

    normalized_execution_mode = (execution_mode or "").strip().lower()
    if not normalized_execution_mode:
        normalized_execution_mode = "batch" if selected_provider == "openai" else "sync"
    if normalized_execution_mode not in {"sync", "parallel", "batch"}:
        raise ValueError(f"Unsupported execution mode: {execution_mode!r}")
    normalized_proposal_track_mode = (proposal_track_mode or "oracle").strip().lower()
    if normalized_proposal_track_mode not in {"oracle", "diagnosis_routed"}:
        raise ValueError(f"Unsupported proposal_track_mode: {proposal_track_mode!r}")
    normalized_oracle_diagnosis_mode = (oracle_diagnosis_mode or "").strip().lower()
    if not normalized_oracle_diagnosis_mode:
        normalized_oracle_diagnosis_mode = "skip" if normalized_proposal_track_mode == "oracle" else "run"
    if normalized_oracle_diagnosis_mode not in {"run", "skip"}:
        raise ValueError(f"Unsupported oracle_diagnosis_mode: {oracle_diagnosis_mode!r}")
    if normalized_proposal_track_mode != "oracle" and normalized_oracle_diagnosis_mode == "skip":
        raise ValueError("--oracle-diagnosis-mode=skip is only valid with proposal_track_mode=oracle.")
    should_run_track_diagnosis = (
        normalized_proposal_track_mode == "diagnosis_routed" or normalized_oracle_diagnosis_mode == "run"
    )
    if normalized_execution_mode == "batch" and not isinstance(provider, BatchModelProvider):
        raise RuntimeError(
            f"Execution mode 'batch' is not supported by provider {provider.__class__.__name__}."
        )
    if normalized_execution_mode == "batch" and selected_provider not in {"openai", "static"}:
        raise RuntimeError(
            "Execution mode 'batch' is currently supported only for the standard OpenAI provider. "
            f"Use --execution-mode sync or --execution-mode parallel for endpoint {selected_provider!r}."
        )
    if parallel_workers is not None and parallel_workers < 1:
        raise ValueError("parallel_workers must be at least 1 when provided.")

    resume_requested = resume_run_dir is not None
    existing_run_config: dict[str, Any] | None = None
    if resume_requested:
        out_dir = Path(resume_run_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        existing_run_config = _load_optional_json(out_dir / RUN_CONFIG_FILENAME)
        if oracle_diagnosis_mode is None and isinstance(existing_run_config, dict):
            existing_oracle_mode = existing_run_config.get("oracle_diagnosis_mode")
            if existing_oracle_mode in {"run", "skip"}:
                normalized_oracle_diagnosis_mode = str(existing_oracle_mode)
            elif normalized_proposal_track_mode == "oracle":
                normalized_oracle_diagnosis_mode = "run"
            should_run_track_diagnosis = (
                normalized_proposal_track_mode == "diagnosis_routed" or normalized_oracle_diagnosis_mode == "run"
            )
        run_id = (
            existing_run_config.get("run_id")
            if isinstance(existing_run_config, dict) and isinstance(existing_run_config.get("run_id"), str)
            else out_dir.name.split("_", 1)[0]
        )
    else:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        run_dir_name = f"{run_id}_{_slugify(selected_provider)}_{_slugify(selected_model)}"
        out_dir = Path(output_dir) / run_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = out_dir / "raw_model_responses.jsonl"
    manifest_path = out_dir / "run_manifest.jsonl"
    run_config_path = out_dir / RUN_CONFIG_FILENAME

    _emit_runtime_status(
        None,
        (
            "Preparing resumed run "
            if resume_requested
            else "Preparing run "
        )
        + f"{run_id} with provider={selected_provider}, model={selected_model}, "
        + f"mode={normalized_execution_mode}, proposal_track_mode={normalized_proposal_track_mode}, "
        + f"oracle_diagnosis_mode={normalized_oracle_diagnosis_mode}."
        + (f" Resume directory: {out_dir}." if resume_requested else ""),
    )

    resolved_case_ids = resolve_case_id_filter(
        case_ids=case_ids,
        selection_manifest_path=selection_manifest_path,
    )
    if resolved_case_ids is None:
        _emit_runtime_status(
            None,
            f"Materializing generation selection from {classified_path} without an explicit case-id filter.",
        )
    else:
        selected_case_count = len(resolved_case_ids)
        selection_source = "selection manifest" if selection_manifest_path else "explicit case ids"
        if max_cases is not None and tracks is None:
            effective_case_count = min(selected_case_count, max(0, max_cases))
            _emit_runtime_status(
                None,
                "Materializing generation selection from "
                f"{classified_path} using {effective_case_count} of {selected_case_count} case ids "
                f"from the {selection_source}.",
            )
        else:
            _emit_runtime_status(
                None,
                "Materializing generation selection from "
                f"{classified_path} using {selected_case_count} case ids from the {selection_source}.",
            )
    generation_selection = _materialize_generation_selection(
        classified_path,
        case_ids=resolved_case_ids,
        tracks=tracks,
        max_cases=max_cases,
        output_dir=out_dir,
    )
    selected_case_ids = generation_selection.case_ids
    visible_case_id_by_raw = {
        case_id: prompt_visible_case_id(case_id, index)
        for index, case_id in enumerate(selected_case_ids, start=1)
    }
    bundle_list = [bundle for bundle in ablation_bundles if bundle in ABLATION_BUNDLES]
    if not bundle_list:
        raise ValueError("At least one supported ablation bundle is required.")
    total_instances = len(selected_case_ids) * len(bundle_list)
    planned_diagnosis_request_count = total_instances if should_run_track_diagnosis else 0
    planned_proposal_request_count = total_instances
    expected_request_count = planned_diagnosis_request_count + planned_proposal_request_count
    skipped_diagnosis_count = 0 if should_run_track_diagnosis else total_instances
    total_work_units = (
        expected_request_count + skipped_diagnosis_count
        if normalized_execution_mode == "batch"
        else total_instances
    )
    expected_run_config = _build_run_config_payload(
        run_id=run_id,
        out_dir=out_dir,
        provider_name=selected_provider,
        model_name=selected_model,
        openai_reasoning_effort=openai_reasoning_effort,
        execution_mode=normalized_execution_mode,
        proposal_track_mode=normalized_proposal_track_mode,
        oracle_diagnosis_mode=normalized_oracle_diagnosis_mode,
        classified_path=classified_path,
        world_state_path=world_state_path,
        selection_manifest_path=selection_manifest_path,
        tracks=tracks,
        max_cases=max_cases,
        ablation_bundles=bundle_list,
        selected_case_ids=selected_case_ids,
        expected_request_count=expected_request_count,
        skipped_diagnosis_count=skipped_diagnosis_count,
        generation_strategy=generation_selection.strategy,
        selected_generation_records_path=generation_selection.records_path,
        started_at_utc=run_started_utc,
    )
    expected_run_config["prompt_visible_case_id_policy"] = "case_000001_ordered_by_generation_selection"
    expected_run_config["visible_case_id_map"] = dict(visible_case_id_by_raw)
    if resume_requested:
        _validate_resume_run_config(existing_run_config, expected_run_config)
        if isinstance(existing_run_config, dict) and isinstance(existing_run_config.get("started_at_utc"), str):
            expected_run_config["started_at_utc"] = existing_run_config["started_at_utc"]
        expected_run_config["last_resumed_at_utc"] = run_started_utc
        if existing_run_config is None:
            _emit_runtime_status(
                None,
                f"No existing {RUN_CONFIG_FILENAME} was found under {out_dir}; resuming in best-effort mode.",
            )
    write_json(run_config_path, expected_run_config)

    existing_manifest_rows = _load_existing_manifest_rows(manifest_path) if resume_requested else []
    existing_manifest_lookup = _manifest_rows_by_key(existing_manifest_rows)
    existing_diagnosis_rows = (
        _load_existing_track_diagnosis_rows(out_dir, bundle_list)
        if resume_requested and normalized_proposal_track_mode == "diagnosis_routed"
        else {}
    )
    existing_completed_work_units = (
        len(existing_manifest_rows)
        if normalized_execution_mode == "batch"
        else _completed_case_units(
            existing_manifest_lookup,
            bundle_list=bundle_list,
            selected_case_ids=selected_case_ids,
        )
    )
    completed_work_units = existing_completed_work_units
    effective_parallel_workers = 1
    parallel_worker_source = "disabled"
    if normalized_execution_mode == "parallel":
        if parallel_workers is None:
            effective_parallel_workers, parallel_worker_source = _default_parallel_workers(
                selected_provider,
                selected_model,
            )
        else:
            effective_parallel_workers = parallel_workers
            parallel_worker_source = "argument"
        effective_parallel_workers = min(effective_parallel_workers, total_instances) if total_instances else 1
    refresh_every = max(1, min(1000, (total_work_units + 9) // 10)) if total_work_units else 1

    world_state_store = WorldStateStore(Path(world_state_path), __import__("logging").getLogger("reasoning_floor"))
    _emit_runtime_status(None, f"Opening world-state index at {world_state_path}.")
    world_state_store.open()
    progress = tqdm(
        total=total_work_units,
        desc="reasoning-floor",
        unit="manifest-row" if normalized_execution_mode == "batch" else "case",
        disable=_disable_generation_progress(
            execution_mode=normalized_execution_mode,
            total_requests=total_work_units,
        ),
    )
    pending_progress = 0
    current_estimated_cost_usd, has_cost_data = _update_cost_tracking(
        (
            row.get("usage")
            for row in existing_manifest_rows
            if isinstance(row.get("usage"), dict)
        ),
        0.0,
        False,
    )
    if completed_work_units:
        progress.update(min(completed_work_units, total_work_units))
        progress.set_postfix(
            {
                "current_cost": _format_cost(current_estimated_cost_usd if has_cost_data else None),
                "est_total_cost": _format_cost(current_estimated_cost_usd if has_cost_data else None),
            },
            refresh=True,
        )
    batch_summary: dict[str, Any] | None = None
    try:
        def build_diagnosis_request(
            record: dict[str, Any],
            bundle: str,
            world_state_entry: Optional[dict[str, Any]],
        ) -> tuple[PromptBundle, dict[str, Any]]:
            visible_case_id = visible_case_id_by_raw.get(record["id"], prompt_visible_case_id(record["id"]))
            diagnosis_bundle = build_track_diagnosis_prompt_bundle(
                record,
                world_state_entry,
                bundle,
                visible_case_id=visible_case_id,
            )
            diagnosis_metadata = _request_metadata(
                run_id=run_id,
                case_id=record["id"],
                visible_case_id=visible_case_id,
                bundle=bundle,
                prompt_name=diagnosis_bundle.prompt_name,
                historical_track=record.get("track"),
                task_type="track_diagnosis",
                model=selected_model,
                proposal_track_used=None,
                routing_source="diagnosis_only",
                context_audit=diagnosis_bundle.context_audit,
            )
            return diagnosis_bundle, diagnosis_metadata

        def build_proposal_request(
            record: dict[str, Any],
            bundle: str,
            world_state_entry: Optional[dict[str, Any]],
            *,
            proposal_track_used: str,
            routing_source: str,
        ) -> tuple[PromptBundle, dict[str, Any]]:
            visible_case_id = visible_case_id_by_raw.get(record["id"], prompt_visible_case_id(record["id"]))
            proposal_bundle = build_prompt_bundle(
                record,
                world_state_entry,
                bundle,
                proposal_track=proposal_track_used,
                visible_case_id=visible_case_id,
            )
            proposal_metadata = _request_metadata(
                run_id=run_id,
                case_id=record["id"],
                visible_case_id=visible_case_id,
                bundle=bundle,
                prompt_name=proposal_bundle.prompt_name,
                historical_track=record.get("track"),
                task_type="proposal",
                model=selected_model,
                proposal_track_used=proposal_track_used,
                routing_source=routing_source,
                context_audit=proposal_bundle.context_audit,
                t_box_constraint_type_qids=(
                    _t_box_constraint_type_qids(record, world_state_entry)
                    if proposal_track_used == "T_BOX"
                    else None
                ),
            )
            return proposal_bundle, proposal_metadata

        def build_skipped_proposal_request(
            record: dict[str, Any],
            bundle: str,
            *,
            proposal_track_used: str,
            routing_source: str,
            context_audit: Optional[dict[str, Any]],
        ) -> dict[str, Any]:
            return _request_metadata(
                run_id=run_id,
                case_id=record["id"],
                visible_case_id=visible_case_id_by_raw.get(record["id"], prompt_visible_case_id(record["id"])),
                bundle=bundle,
                prompt_name="skipped_proposal_no_track",
                historical_track=record.get("track"),
                task_type="proposal",
                model=selected_model,
                proposal_track_used=proposal_track_used,
                routing_source=routing_source,
                context_audit=context_audit,
            )

        def build_skipped_diagnosis_request(
            record: dict[str, Any],
            bundle: str,
            *,
            context_audit: Optional[dict[str, Any]] = None,
        ) -> dict[str, Any]:
            return _request_metadata(
                run_id=run_id,
                case_id=record["id"],
                visible_case_id=visible_case_id_by_raw.get(record["id"], prompt_visible_case_id(record["id"])),
                bundle=bundle,
                prompt_name="skipped_track_diagnosis_oracle",
                historical_track=record.get("track"),
                task_type="track_diagnosis",
                model=selected_model,
                proposal_track_used=None,
                routing_source="oracle_historical_track",
                context_audit=context_audit,
            )

        def mark_completed(units: int = 1) -> None:
            nonlocal completed_work_units
            nonlocal pending_progress
            completed_work_units += units
            pending_progress += units
            if pending_progress >= refresh_every or completed_work_units == total_work_units:
                flush_progress()

        def flush_progress() -> None:
            nonlocal pending_progress
            if pending_progress <= 0:
                return
            estimated_total_cost = None
            if has_cost_data and completed_work_units > 0:
                estimated_total_cost = (current_estimated_cost_usd / completed_work_units) * total_work_units
            progress.update(pending_progress)
            progress.set_postfix(
                {
                    "current_cost": _format_cost(current_estimated_cost_usd if has_cost_data else None),
                    "est_total_cost": _format_cost(estimated_total_cost),
                },
                refresh=True,
            )
            pending_progress = 0

        def record_generation_result(
            *,
            request_info: dict[str, Any],
            raw_response: Any,
            parsed_payload: Any,
            usage: dict[str, Any],
            bundle_handles: dict[str, dict[str, Any]],
            raw_log_fh: Any,
            manifest_fh: Any,
            elapsed_seconds: float | None = None,
            error_message: str | None = None,
            recovery_details: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            nonlocal current_estimated_cost_usd
            nonlocal has_cost_data
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
                elapsed_seconds=elapsed_seconds,
                error_message=error_message,
                recovery_details=recovery_details,
            )
            usage_manifest.append(manifest_record)
            current_estimated_cost_usd, has_cost_data = _update_cost_tracking(
                (usage,),
                current_estimated_cost_usd,
                has_cost_data,
            )
            bundle_key = request_info.get("ablation_bundle")
            case_id = request_info.get("case_id")
            task_type = request_info.get("task_type")
            if isinstance(bundle_key, str) and isinstance(case_id, str) and isinstance(task_type, str):
                existing_manifest_lookup[(bundle_key, case_id, task_type)] = manifest_record
            return manifest_record

        def record_skipped_proposal(
            *,
            request_info: dict[str, Any],
            parse_status: str,
            skip_reason: str,
            raw_log_fh: Any,
            manifest_fh: Any,
        ) -> dict[str, Any]:
            usage = _apply_cost_estimation_policy(
                _empty_usage_payload(selected_provider, selected_model, request_info),
                provider_name=selected_provider,
                execution_mode=normalized_execution_mode,
            )
            manifest_record = _record_skipped_proposal_result(
                request_info=request_info,
                usage=usage,
                raw_log_fh=raw_log_fh,
                manifest_fh=manifest_fh,
                parse_status=parse_status,
                skip_reason=skip_reason,
            )
            usage_manifest.append(manifest_record)
            bundle_key = request_info.get("ablation_bundle")
            case_id = request_info.get("case_id")
            if isinstance(bundle_key, str) and isinstance(case_id, str):
                existing_manifest_lookup[(bundle_key, case_id, "proposal")] = manifest_record
            return manifest_record

        def record_skipped_diagnosis(
            *,
            request_info: dict[str, Any],
            parse_status: str,
            skip_reason: str,
            raw_log_fh: Any,
            manifest_fh: Any,
        ) -> dict[str, Any]:
            usage = _apply_cost_estimation_policy(
                _empty_usage_payload(selected_provider, selected_model, request_info),
                provider_name=selected_provider,
                execution_mode=normalized_execution_mode,
            )
            manifest_record = _record_skipped_diagnosis_result(
                request_info=request_info,
                usage=usage,
                raw_log_fh=raw_log_fh,
                manifest_fh=manifest_fh,
                parse_status=parse_status,
                skip_reason=skip_reason,
            )
            usage_manifest.append(manifest_record)
            bundle_key = request_info.get("ablation_bundle")
            case_id = request_info.get("case_id")
            if isinstance(bundle_key, str) and isinstance(case_id, str):
                existing_manifest_lookup[(bundle_key, case_id, "track_diagnosis")] = manifest_record
            return manifest_record

        def routing_info_for_existing_case(
            *,
            bundle: str,
            case_id: str,
        ) -> dict[str, Any] | None:
            diagnosis_manifest_row = _manifest_row_for_task(
                existing_manifest_lookup,
                bundle=bundle,
                case_id=case_id,
                task_type="track_diagnosis",
            )
            if diagnosis_manifest_row is None:
                return None
            return _routing_info_from_existing_diagnosis(
                diagnosis_manifest_row=diagnosis_manifest_row,
                diagnosis_row=existing_diagnosis_rows.get((bundle, case_id)),
            )

        def execute_case_pipeline(
            record: dict[str, Any],
            bundle: str,
            world_state_entry: Optional[dict[str, Any]],
            *,
            existing_diagnosis_manifest: dict[str, Any] | None = None,
            existing_proposal_manifest: dict[str, Any] | None = None,
        ) -> CasePipelineOutcome:
            request_results: list[RequestExecutionResult] = []
            skipped_diagnoses: list[dict[str, Any]] = []
            skipped_proposals: list[dict[str, Any]] = []
            routing_info: dict[str, Any] | None = None

            if existing_diagnosis_manifest is None:
                if should_run_track_diagnosis:
                    diagnosis_bundle, diagnosis_metadata = build_diagnosis_request(record, bundle, world_state_entry)
                    diagnosis_result = _execute_case_requests(
                        provider,
                        [(diagnosis_bundle, diagnosis_metadata)],
                        provider_name=selected_provider,
                        execution_mode=normalized_execution_mode,
                    )[0]
                    request_results.append(diagnosis_result)
                    routing_info = {
                        "proposal_track_used": _routed_track_from_diagnosis_payload(
                            diagnosis_result.parsed_payload,
                            record["id"],
                            diagnosis_metadata.get("visible_case_id")
                            if isinstance(diagnosis_metadata.get("visible_case_id"), str)
                            else None,
                        ),
                        "routing_source": "diagnosis_prediction",
                        "context_audit": diagnosis_metadata.get("context_audit") or {},
                    }
                else:
                    skipped_diagnoses.append(
                        {
                            "request_info": build_skipped_diagnosis_request(record, bundle),
                            "parse_status": "skipped",
                            "skip_reason": "oracle_mode_track_diagnosis_skipped",
                        }
                    )
            else:
                routing_info = routing_info_for_existing_case(bundle=bundle, case_id=record["id"])

            if normalized_proposal_track_mode == "oracle":
                if existing_proposal_manifest is not None:
                    return CasePipelineOutcome(
                        request_results=request_results,
                        skipped_diagnoses=skipped_diagnoses,
                        skipped_proposals=skipped_proposals,
                    )
                proposal_bundle, proposal_metadata = build_proposal_request(
                    record,
                    bundle,
                    world_state_entry,
                    proposal_track_used=record.get("track") or "A_BOX",
                    routing_source="oracle_historical_track",
                )
                request_results.extend(
                    _execute_case_requests(
                        provider,
                        [(proposal_bundle, proposal_metadata)],
                        provider_name=selected_provider,
                        execution_mode=normalized_execution_mode,
                    )
                )
                return CasePipelineOutcome(
                    request_results=request_results,
                    skipped_diagnoses=skipped_diagnoses,
                    skipped_proposals=skipped_proposals,
                )

            if existing_proposal_manifest is not None:
                return CasePipelineOutcome(
                    request_results=request_results,
                    skipped_diagnoses=skipped_diagnoses,
                    skipped_proposals=skipped_proposals,
                )

            if routing_info is None:
                raise RuntimeError(
                    f"Cannot resume case {record['id']} in bundle {bundle}: routing information is unavailable."
                )
            routed_track = routing_info.get("proposal_track_used")
            context_audit = routing_info.get("context_audit")
            if routed_track == "AMBIGUOUS":
                skipped_request_info = build_skipped_proposal_request(
                    record,
                    bundle,
                    proposal_track_used="AMBIGUOUS",
                    routing_source="diagnosis_prediction",
                    context_audit=context_audit,
                )
                skipped_proposals.append(
                    {
                        "request_info": skipped_request_info,
                        "parse_status": "skipped_ambiguous_track",
                        "skip_reason": "Diagnosis predicted AMBIGUOUS track.",
                    }
                )
                return CasePipelineOutcome(
                    request_results=request_results,
                    skipped_diagnoses=skipped_diagnoses,
                    skipped_proposals=skipped_proposals,
                )
            if routed_track not in {"A_BOX", "T_BOX"}:
                skipped_request_info = build_skipped_proposal_request(
                    record,
                    bundle,
                    proposal_track_used=UNROUTABLE_TRACK,
                    routing_source="diagnosis_prediction",
                    context_audit=context_audit,
                )
                skipped_proposals.append(
                    {
                        "request_info": skipped_request_info,
                        "parse_status": "skipped_unroutable_track",
                        "skip_reason": "Diagnosis output could not be routed to A_BOX or T_BOX.",
                    }
                )
                return CasePipelineOutcome(
                    request_results=request_results,
                    skipped_diagnoses=skipped_diagnoses,
                    skipped_proposals=skipped_proposals,
                )

            proposal_bundle, proposal_metadata = build_proposal_request(
                record,
                bundle,
                world_state_entry,
                proposal_track_used=str(routed_track),
                routing_source="diagnosis_prediction",
            )
            request_results.extend(
                _execute_case_requests(
                    provider,
                    [(proposal_bundle, proposal_metadata)],
                    provider_name=selected_provider,
                    execution_mode=normalized_execution_mode,
                )
            )
            return CasePipelineOutcome(
                request_results=request_results,
                skipped_diagnoses=skipped_diagnoses,
                skipped_proposals=skipped_proposals,
            )

        def move_batch_artifact(path: Path | None, prefix: str) -> Path | None:
            if path is None or not path.exists() or not prefix:
                return path
            destination = path.with_name(f"{prefix}{path.name}")
            if destination.exists():
                destination.unlink()
            shutil.move(str(path), str(destination))
            return destination

        def relocate_batch_phase_artifacts(
            *,
            output_path: Path | None,
            error_path: Path | None,
            phase_prefix: str,
        ) -> tuple[Path | None, Path | None, dict[str, str | None]]:
            relocated_output = move_batch_artifact(output_path, phase_prefix)
            relocated_error = move_batch_artifact(error_path, phase_prefix)
            artifact_paths: dict[str, str | None] = {
                "output_path": str(relocated_output) if relocated_output else None,
                "error_path": str(relocated_error) if relocated_error else None,
            }
            job_path = out_dir / f"{selected_provider}_batch_job.json"
            relocated_job = move_batch_artifact(job_path, phase_prefix)
            if relocated_job is not None and relocated_job.exists():
                artifact_paths["job_path"] = str(relocated_job)
            return relocated_output, relocated_error, artifact_paths

        def build_batch_phase_summary(
            *,
            request_count: int,
            input_path: Path,
            request_manifest_path: Path,
            phase_label: str,
            execution_result: Any,
            elapsed_seconds: float,
            artifact_paths: dict[str, str | None],
            sync_retry_fallback: dict[str, int] | None = None,
        ) -> dict[str, Any]:
            retry_summary = dict(
                sync_retry_fallback
                or {
                    "eligible": 0,
                    "attempted": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "unrecoverable": 0,
                }
            )
            if execution_result is None:
                return {
                    "id": None,
                    "status": "skipped_empty",
                    "completion_window": batch_completion_window,
                    "poll_interval_seconds": batch_poll_interval_seconds,
                    "elapsed_seconds": round(elapsed_seconds, 6),
                    "request_counts": {"total": request_count, "completed": 0, "failed": 0},
                    "input_path": str(input_path),
                    "request_manifest_path": str(request_manifest_path),
                    "output_path": artifact_paths.get("output_path"),
                    "error_path": artifact_paths.get("error_path"),
                    "job_path": artifact_paths.get("job_path"),
                    "sync_retry_fallback": retry_summary,
                    "phase": phase_label,
                }
            return {
                "id": execution_result.batch.get("id"),
                "status": execution_result.batch.get("status"),
                "completion_window": batch_completion_window,
                "poll_interval_seconds": batch_poll_interval_seconds,
                "elapsed_seconds": round(elapsed_seconds, 6),
                "request_counts": execution_result.batch.get("request_counts"),
                "input_path": str(input_path),
                "request_manifest_path": str(request_manifest_path),
                "output_path": artifact_paths.get("output_path"),
                "error_path": artifact_paths.get("error_path"),
                "job_path": artifact_paths.get("job_path"),
                "sync_retry_fallback": retry_summary,
                "phase": phase_label,
            }

        def iter_request_manifest_rows(path: Path) -> dict[str, dict[str, Any]]:
            request_map: dict[str, dict[str, Any]] = {}
            for row in iter_jsonl(path):
                if not isinstance(row, dict):
                    continue
                custom_id = row.get("custom_id")
                metadata = row.get("metadata")
                if isinstance(custom_id, str) and isinstance(metadata, dict):
                    request_map[custom_id] = {"custom_id": custom_id, **metadata}
            return request_map

        def batch_status_from_phases(phases: dict[str, dict[str, Any]]) -> str:
            statuses = [phase.get("status") for phase in phases.values()]
            if not statuses:
                return "skipped_empty"
            for status in statuses:
                if status not in {"completed", "skipped_empty"}:
                    return str(status)
            if all(status == "skipped_empty" for status in statuses):
                return "skipped_empty"
            return "completed"

        def aggregate_sync_retry_fallback(phases: dict[str, dict[str, Any]]) -> dict[str, int]:
            totals = {"eligible": 0, "attempted": 0, "succeeded": 0, "failed": 0, "unrecoverable": 0}
            for phase in phases.values():
                retry_summary = phase.get("sync_retry_fallback")
                if not isinstance(retry_summary, dict):
                    continue
                for key in totals:
                    value = retry_summary.get(key)
                    if isinstance(value, int):
                        totals[key] += value
            return totals

        def rebuild_batch_phase_results(
            *,
            phase_label: str,
            batch_execution: Any,
            batch_input_path: Path,
            request_manifest_path: Path,
            bundle_handles: dict[str, dict[str, Any]],
            raw_log_fh: Any,
            manifest_fh: Any,
            on_result: Callable[[dict[str, Any], Any, str | None], None] | None = None,
        ) -> dict[str, int]:
            request_map = iter_request_manifest_rows(request_manifest_path)
            batch_request_map = _batch_requests_by_custom_id(batch_input_path)
            finalized_custom_ids: set[str] = set()
            pending_retries: dict[str, dict[str, Any]] = {}
            result_paths = ()
            if batch_execution is not None:
                result_paths = (batch_execution.output_path, batch_execution.error_path)

            def finalize_result(
                *,
                request_info: dict[str, Any],
                raw_response: Any,
                parsed_payload: Any,
                usage: dict[str, Any],
                error_message: str | None,
                elapsed_seconds: float | None,
                recovery_details: dict[str, Any] | None = None,
            ) -> None:
                record_generation_result(
                    request_info=request_info,
                    raw_response=raw_response,
                    parsed_payload=parsed_payload,
                    usage=usage,
                    bundle_handles=bundle_handles,
                    raw_log_fh=raw_log_fh,
                    manifest_fh=manifest_fh,
                    elapsed_seconds=elapsed_seconds,
                    error_message=error_message,
                    recovery_details=recovery_details,
                )
                if on_result is not None:
                    on_result(request_info, parsed_payload, error_message)

            for result_path in result_paths:
                if result_path is None:
                    continue
                for result_row in iter_jsonl(result_path):
                    if not isinstance(result_row, dict):
                        continue
                    custom_id = result_row.get("custom_id")
                    if (
                        not isinstance(custom_id, str)
                        or custom_id in finalized_custom_ids
                        or custom_id in pending_retries
                    ):
                        continue
                    request_info = request_map.get(custom_id)
                    if not isinstance(request_info, dict):
                        continue
                    raw_response, parsed_payload, usage, error_message = provider.parse_batch_result(
                        result_row,
                        request_info,
                    )
                    if _is_retryable_batch_error(result_row, error_message):
                        pending_retries[custom_id] = {
                            "request_info": request_info,
                            "batch_request": batch_request_map.get(custom_id),
                            "error_message": error_message,
                            "status_code": _batch_result_status_code(result_row),
                        }
                        continue
                    usage = _apply_cost_estimation_policy(
                        usage,
                        provider_name=selected_provider,
                        execution_mode=normalized_execution_mode,
                    )
                    finalize_result(
                        request_info=request_info,
                        raw_response=raw_response,
                        parsed_payload=parsed_payload,
                        usage=usage,
                        error_message=error_message,
                        elapsed_seconds=None,
                    )
                    finalized_custom_ids.add(custom_id)
                    mark_completed()

            missing_custom_ids = sorted(set(request_map) - finalized_custom_ids - set(pending_retries))
            for custom_id in missing_custom_ids:
                request_info = request_map[custom_id]
                batch_request = batch_request_map.get(custom_id)
                if isinstance(batch_request, dict):
                    pending_retries[custom_id] = {
                        "request_info": request_info,
                        "batch_request": batch_request,
                        "error_message": "Batch result was missing from both output and error files.",
                        "status_code": None,
                    }
                    continue
                usage = _apply_cost_estimation_policy(
                    _empty_usage_payload(selected_provider, selected_model, request_info),
                    provider_name=selected_provider,
                    execution_mode=normalized_execution_mode,
                )
                finalize_result(
                    request_info=request_info,
                    raw_response=None,
                    parsed_payload=None,
                    usage=usage,
                    error_message="Batch result was missing from both output and error files.",
                    elapsed_seconds=None,
                )
                finalized_custom_ids.add(custom_id)
                mark_completed()

            retry_summary = {
                "eligible": len(pending_retries),
                "attempted": 0,
                "succeeded": 0,
                "failed": 0,
                "unrecoverable": 0,
            }
            if pending_retries:
                _emit_runtime_status(
                    progress,
                    f"Retrying {len(pending_retries)} {phase_label} request(s) synchronously after batch failure.",
                )
            for custom_id in sorted(pending_retries):
                retry_entry = pending_retries[custom_id]
                request_info = retry_entry["request_info"]
                original_error = retry_entry.get("error_message")
                status_code = retry_entry.get("status_code")
                recovery_details = {
                    "type": "sync_retry_after_batch_error",
                    "phase": phase_label,
                    "batch_status_code": status_code,
                    "batch_error": original_error,
                }
                request_components = _batch_request_prompt_components(retry_entry.get("batch_request") or {})
                if request_components is None:
                    retry_summary["unrecoverable"] += 1
                    usage = _apply_cost_estimation_policy(
                        _empty_usage_payload(selected_provider, selected_model, request_info),
                        provider_name=selected_provider,
                        execution_mode=normalized_execution_mode,
                    )
                    finalize_result(
                        request_info=request_info,
                        raw_response=None,
                        parsed_payload=None,
                        usage=usage,
                        error_message=(
                            original_error
                            or "Retryable batch failure could not be reconstructed for synchronous retry."
                        ),
                        elapsed_seconds=None,
                        recovery_details={**recovery_details, "attempted": False, "succeeded": False},
                    )
                    finalized_custom_ids.add(custom_id)
                    mark_completed()
                    continue
                retry_summary["attempted"] += 1
                prompt, system_prompt, response_format = request_components
                try:
                    retry_started_at = time.perf_counter()
                    raw_response, parsed_payload, usage = provider.generate(
                        prompt,
                        system_prompt,
                        response_format,
                        request_info,
                    )
                    elapsed_seconds = time.perf_counter() - retry_started_at
                    usage = _apply_cost_estimation_policy(
                        usage,
                        provider_name=selected_provider,
                        execution_mode="sync",
                    )
                    error_message = None
                    retry_summary["succeeded"] += 1
                except Exception as exc:
                    raw_response = None
                    parsed_payload = None
                    elapsed_seconds = None
                    usage = _apply_cost_estimation_policy(
                        _empty_usage_payload(selected_provider, selected_model, request_info),
                        provider_name=selected_provider,
                        execution_mode="sync",
                    )
                    retry_error = str(exc).strip() or exc.__class__.__name__
                    error_message = (
                        f"{original_error} Sync retry failed: {retry_error}"
                        if original_error
                        else f"Sync retry failed: {retry_error}"
                    )
                    retry_summary["failed"] += 1
                finalize_result(
                    request_info=request_info,
                    raw_response=raw_response,
                    parsed_payload=parsed_payload,
                    usage=usage,
                    error_message=error_message,
                    elapsed_seconds=elapsed_seconds,
                    recovery_details={**recovery_details, "attempted": True, "succeeded": error_message is None},
                )
                finalized_custom_ids.add(custom_id)
                mark_completed()
            return retry_summary

        _emit_runtime_status(
            progress,
            ("Starting resumed run " if resume_requested else "Starting run ")
            + f"{run_id} with provider={selected_provider}, model={selected_model}, "
            + f"mode={normalized_execution_mode}, proposal_track_mode={normalized_proposal_track_mode}, "
            f"oracle_diagnosis_mode={normalized_oracle_diagnosis_mode}, "
            f"cases={len(selected_case_ids)}, bundles={len(bundle_list)}, "
            f"inference_requests={expected_request_count}, skipped_diagnosis={skipped_diagnosis_count}.",
        )
        _emit_runtime_status(progress, f"Writing outputs to {out_dir}.")
        _emit_runtime_status(
            progress,
            "Materialized generation selection using "
            f"{generation_selection.strategy} for {len(selected_case_ids)} selected case(s).",
        )
        if resume_requested:
            _emit_runtime_status(
                progress,
                "Loaded existing generation state from "
                f"{manifest_path}: {len(existing_manifest_rows)} request row(s), "
                f"{completed_work_units}/{total_work_units} completed work unit(s).",
            )
        usage_manifest: list[dict[str, Any]] = list(existing_manifest_rows)

        a_box_schema = load_a_box_schema(Path("schemas") / "verified_repair_proposal.schema.json")
        t_box_schema = load_t_box_schema(Path("schemas") / "tbox_reform_proposal.schema.json")
        track_schema = load_track_schema(Path("schemas") / "track_diagnosis.schema.json")
        del a_box_schema, t_box_schema, track_schema

        for bundle in bundle_list:
            bundle_dir = out_dir / bundle
            bundle_dir.mkdir(parents=True, exist_ok=True)
            if not resume_requested:
                (bundle_dir / "a_box_proposals.jsonl").touch()
                (bundle_dir / "t_box_proposals.jsonl").touch()
                (bundle_dir / "track_diagnoses.jsonl").touch()

        output_file_mode = "a" if resume_requested else "w"
        with open(raw_log_path, output_file_mode, encoding="utf-8") as raw_log_fh, open(
            manifest_path, output_file_mode, encoding="utf-8"
        ) as manifest_fh, ExitStack() as stack:
            bundle_handles = {}
            for bundle in bundle_list:
                bundle_dir = out_dir / bundle
                bundle_handles[bundle] = {
                    "a_box": stack.enter_context(
                        open(bundle_dir / "a_box_proposals.jsonl", output_file_mode, encoding="utf-8")
                    ),
                    "t_box": stack.enter_context(
                        open(bundle_dir / "t_box_proposals.jsonl", output_file_mode, encoding="utf-8")
                    ),
                    "track": stack.enter_context(
                        open(bundle_dir / "track_diagnoses.jsonl", output_file_mode, encoding="utf-8")
                    ),
                }
            if normalized_execution_mode in {"sync", "parallel"}:
                if normalized_execution_mode == "sync":
                    for bundle in bundle_list:
                        for record in _iter_materialized_generation_records(generation_selection):
                            case_id = record["id"]
                            existing_diagnosis_manifest = _manifest_row_for_task(
                                existing_manifest_lookup,
                                bundle=bundle,
                                case_id=case_id,
                                task_type="track_diagnosis",
                            )
                            existing_proposal_manifest = _manifest_row_for_task(
                                existing_manifest_lookup,
                                bundle=bundle,
                                case_id=case_id,
                                task_type="proposal",
                            )
                            if existing_diagnosis_manifest is not None and existing_proposal_manifest is not None:
                                continue
                            world_state_entry = world_state_store.get(case_id)
                            outcome = execute_case_pipeline(
                                record,
                                bundle,
                                world_state_entry,
                                existing_diagnosis_manifest=existing_diagnosis_manifest,
                                existing_proposal_manifest=existing_proposal_manifest,
                            )
                            for skipped in outcome.skipped_diagnoses:
                                record_skipped_diagnosis(
                                    request_info=skipped["request_info"],
                                    parse_status=skipped["parse_status"],
                                    skip_reason=skipped["skip_reason"],
                                    raw_log_fh=raw_log_fh,
                                    manifest_fh=manifest_fh,
                                )
                            for request_result in outcome.request_results:
                                record_generation_result(
                                    request_info=request_result.request_info,
                                    raw_response=request_result.raw_response,
                                    parsed_payload=request_result.parsed_payload,
                                    usage=request_result.usage,
                                    bundle_handles=bundle_handles,
                                    raw_log_fh=raw_log_fh,
                                    manifest_fh=manifest_fh,
                                    elapsed_seconds=request_result.elapsed_seconds,
                                    error_message=request_result.error_message,
                                )
                            for skipped in outcome.skipped_proposals:
                                record_skipped_proposal(
                                    request_info=skipped["request_info"],
                                    parse_status=skipped["parse_status"],
                                    skip_reason=skipped["skip_reason"],
                                    raw_log_fh=raw_log_fh,
                                    manifest_fh=manifest_fh,
                                )
                            mark_completed()
                else:
                    with ThreadPoolExecutor(max_workers=effective_parallel_workers) as executor:
                        pending_futures: dict[Future[CasePipelineOutcome], str] = {}

                        def record_future_result(future: Future[CasePipelineOutcome]) -> None:
                            outcome = future.result()
                            for skipped in outcome.skipped_diagnoses:
                                record_skipped_diagnosis(
                                    request_info=skipped["request_info"],
                                    parse_status=skipped["parse_status"],
                                    skip_reason=skipped["skip_reason"],
                                    raw_log_fh=raw_log_fh,
                                    manifest_fh=manifest_fh,
                                )
                            for request_result in outcome.request_results:
                                record_generation_result(
                                    request_info=request_result.request_info,
                                    raw_response=request_result.raw_response,
                                    parsed_payload=request_result.parsed_payload,
                                    usage=request_result.usage,
                                    bundle_handles=bundle_handles,
                                    raw_log_fh=raw_log_fh,
                                    manifest_fh=manifest_fh,
                                    elapsed_seconds=request_result.elapsed_seconds,
                                    error_message=request_result.error_message,
                                )
                            for skipped in outcome.skipped_proposals:
                                record_skipped_proposal(
                                    request_info=skipped["request_info"],
                                    parse_status=skipped["parse_status"],
                                    skip_reason=skipped["skip_reason"],
                                    raw_log_fh=raw_log_fh,
                                    manifest_fh=manifest_fh,
                                )
                            mark_completed()

                        def drain_completed_futures(*, wait_for_all: bool) -> None:
                            if not pending_futures:
                                return
                            if not wait_for_all:
                                done, _ = wait(tuple(pending_futures), return_when=FIRST_COMPLETED)
                            else:
                                done, _ = wait(tuple(pending_futures))
                            for future in done:
                                pending_futures.pop(future, None)
                                record_future_result(future)

                        for bundle in bundle_list:
                            for record in _iter_materialized_generation_records(generation_selection):
                                case_id = record["id"]
                                existing_diagnosis_manifest = _manifest_row_for_task(
                                    existing_manifest_lookup,
                                    bundle=bundle,
                                    case_id=case_id,
                                    task_type="track_diagnosis",
                                )
                                existing_proposal_manifest = _manifest_row_for_task(
                                    existing_manifest_lookup,
                                    bundle=bundle,
                                    case_id=case_id,
                                    task_type="proposal",
                                )
                                if existing_diagnosis_manifest is not None and existing_proposal_manifest is not None:
                                    continue
                                world_state_entry = world_state_store.get(case_id)
                                future = executor.submit(
                                    execute_case_pipeline,
                                    record,
                                    bundle,
                                    world_state_entry,
                                    existing_diagnosis_manifest=existing_diagnosis_manifest,
                                    existing_proposal_manifest=existing_proposal_manifest,
                                )
                                pending_futures[future] = case_id
                                if len(pending_futures) >= effective_parallel_workers:
                                    drain_completed_futures(wait_for_all=False)

                        while pending_futures:
                            drain_completed_futures(wait_for_all=True)
            else:
                assert isinstance(provider, BatchModelProvider)
                if normalized_proposal_track_mode == "oracle":
                    batch_input_path = out_dir / "batch_input.jsonl"
                    request_manifest_path = out_dir / "batch_request_manifest.jsonl"
                    request_counter = 0
                    with open(batch_input_path, "w", encoding="utf-8") as batch_input_fh, open(
                        request_manifest_path, "w", encoding="utf-8"
                    ) as request_manifest_fh:
                        for bundle in bundle_list:
                            for record in _iter_materialized_generation_records(generation_selection):
                                case_id = record["id"]
                                world_state_entry = world_state_store.get(case_id)
                                if (
                                    _manifest_row_for_task(
                                        existing_manifest_lookup,
                                        bundle=bundle,
                                        case_id=case_id,
                                        task_type="track_diagnosis",
                                    )
                                    is None
                                ):
                                    if should_run_track_diagnosis:
                                        diagnosis_bundle, diagnosis_metadata = build_diagnosis_request(
                                            record,
                                            bundle,
                                            world_state_entry,
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
                                            {"custom_id": diagnosis_custom_id, "metadata": diagnosis_metadata},
                                        )
                                    else:
                                        record_skipped_diagnosis(
                                            request_info=build_skipped_diagnosis_request(record, bundle),
                                            parse_status="skipped",
                                            skip_reason="oracle_mode_track_diagnosis_skipped",
                                            raw_log_fh=raw_log_fh,
                                            manifest_fh=manifest_fh,
                                        )
                                        mark_completed()

                                if (
                                    _manifest_row_for_task(
                                        existing_manifest_lookup,
                                        bundle=bundle,
                                        case_id=case_id,
                                        task_type="proposal",
                                    )
                                    is None
                                ):
                                    proposal_bundle, proposal_metadata = build_proposal_request(
                                        record,
                                        bundle,
                                        world_state_entry,
                                        proposal_track_used=record.get("track") or "A_BOX",
                                        routing_source="oracle_historical_track",
                                    )
                                    proposal_custom_id = f"rf_{request_counter:09d}"
                                    request_counter += 1
                                    provider.write_batch_request(
                                        batch_input_fh,
                                        custom_id=proposal_custom_id,
                                        prompt=proposal_bundle.prompt,
                                        system_prompt=proposal_bundle.system_prompt,
                                        response_format=proposal_bundle.response_format,
                                        metadata=proposal_metadata,
                                    )
                                    _append_jsonl_record(
                                        request_manifest_fh,
                                        {"custom_id": proposal_custom_id, "metadata": proposal_metadata},
                                    )

                    batch_execution = None
                    batch_elapsed_seconds = 0.0
                    artifact_paths = {"output_path": None, "error_path": None, "job_path": None}
                    if request_counter == 0:
                        _emit_runtime_status(progress, "No batch requests were generated; skipping provider batch job.")
                    else:
                        _emit_runtime_status(
                            progress,
                            f"Prepared {request_counter} batch requests; submitting provider batch job.",
                        )
                        batch_started_at = time.perf_counter()
                        batch_execution = provider.execute_batch(
                            batch_input_path,
                            request_manifest_path=request_manifest_path,
                            output_dir=out_dir,
                            completion_window=batch_completion_window,
                            poll_interval_seconds=batch_poll_interval_seconds,
                            status_callback=lambda message: _emit_runtime_status(progress, message),
                        )
                        batch_elapsed_seconds = time.perf_counter() - batch_started_at
                        relocated_output, relocated_error, artifact_paths = relocate_batch_phase_artifacts(
                            output_path=batch_execution.output_path,
                            error_path=batch_execution.error_path,
                            phase_prefix="",
                        )
                        batch_execution = type(batch_execution)(
                            batch=batch_execution.batch,
                            output_path=relocated_output,
                            error_path=relocated_error,
                        )
                        _emit_runtime_status(
                            progress,
                            "Provider batch finished; rebuilding normalized outputs from returned records.",
                        )

                    one_stage_retry_summary = rebuild_batch_phase_results(
                        phase_label="combined",
                        batch_execution=batch_execution,
                        batch_input_path=batch_input_path,
                        request_manifest_path=request_manifest_path,
                        bundle_handles=bundle_handles,
                        raw_log_fh=raw_log_fh,
                        manifest_fh=manifest_fh,
                    )

                    one_stage_phase = build_batch_phase_summary(
                        request_count=request_counter,
                        input_path=batch_input_path,
                        request_manifest_path=request_manifest_path,
                        phase_label="combined",
                        execution_result=batch_execution,
                        elapsed_seconds=batch_elapsed_seconds,
                        artifact_paths=artifact_paths,
                        sync_retry_fallback=one_stage_retry_summary,
                    )
                    batch_summary = {
                        "mode": "one_stage",
                        "status": one_stage_phase.get("status"),
                        "overall_status": one_stage_phase.get("status"),
                        "elapsed_seconds": one_stage_phase.get("elapsed_seconds"),
                        "sync_retry_fallback": dict(one_stage_phase.get("sync_retry_fallback") or {}),
                        "phases": {"combined": one_stage_phase},
                    }
                else:
                    phase_summaries: dict[str, dict[str, Any]] = {}
                    routed_tracks: dict[tuple[str, str], dict[str, Any]] = {}
                    for bundle in bundle_list:
                        for case_id in selected_case_ids:
                            routing_info = routing_info_for_existing_case(bundle=bundle, case_id=case_id)
                            if routing_info is not None:
                                routed_tracks[(bundle, case_id)] = routing_info

                    diagnosis_input_path = out_dir / "diagnosis_batch_input.jsonl"
                    diagnosis_manifest_path = out_dir / "diagnosis_batch_request_manifest.jsonl"
                    diagnosis_request_count = 0
                    with open(diagnosis_input_path, "w", encoding="utf-8") as batch_input_fh, open(
                        diagnosis_manifest_path, "w", encoding="utf-8"
                    ) as request_manifest_fh:
                        for bundle in bundle_list:
                            for record in _iter_materialized_generation_records(generation_selection):
                                case_id = record["id"]
                                if (
                                    _manifest_row_for_task(
                                        existing_manifest_lookup,
                                        bundle=bundle,
                                        case_id=case_id,
                                        task_type="track_diagnosis",
                                    )
                                    is not None
                                ):
                                    continue
                                world_state_entry = world_state_store.get(case_id)
                                diagnosis_bundle, diagnosis_metadata = build_diagnosis_request(
                                    record,
                                    bundle,
                                    world_state_entry,
                                )
                                diagnosis_custom_id = f"rf_diag_{diagnosis_request_count:09d}"
                                diagnosis_request_count += 1
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
                                    {"custom_id": diagnosis_custom_id, "metadata": diagnosis_metadata},
                                )

                    diagnosis_execution = None
                    diagnosis_elapsed_seconds = 0.0
                    diagnosis_artifacts = {"output_path": None, "error_path": None, "job_path": None}
                    if diagnosis_request_count > 0:
                        _emit_runtime_status(
                            progress,
                            f"Prepared {diagnosis_request_count} diagnosis batch requests; submitting provider "
                            "batch job.",
                        )
                        batch_started_at = time.perf_counter()
                        diagnosis_execution = provider.execute_batch(
                            diagnosis_input_path,
                            request_manifest_path=diagnosis_manifest_path,
                            output_dir=out_dir,
                            completion_window=batch_completion_window,
                            poll_interval_seconds=batch_poll_interval_seconds,
                            status_callback=lambda message: _emit_runtime_status(progress, message),
                        )
                        diagnosis_elapsed_seconds = time.perf_counter() - batch_started_at
                        relocated_output, relocated_error, diagnosis_artifacts = relocate_batch_phase_artifacts(
                            output_path=diagnosis_execution.output_path,
                            error_path=diagnosis_execution.error_path,
                            phase_prefix="diagnosis_",
                        )
                        diagnosis_execution = type(diagnosis_execution)(
                            batch=diagnosis_execution.batch,
                            output_path=relocated_output,
                            error_path=relocated_error,
                        )
                        _emit_runtime_status(
                            progress,
                            "Diagnosis batch finished; rebuilding normalized diagnosis outputs and routing table.",
                        )

                    diagnosis_retry_summary = rebuild_batch_phase_results(
                        phase_label="diagnosis",
                        batch_execution=diagnosis_execution,
                        batch_input_path=diagnosis_input_path,
                        request_manifest_path=diagnosis_manifest_path,
                        bundle_handles=bundle_handles,
                        raw_log_fh=raw_log_fh,
                        manifest_fh=manifest_fh,
                        on_result=lambda request_info, parsed_payload, error_message: routed_tracks.__setitem__(
                            (request_info["ablation_bundle"], request_info["case_id"]),
                            {
                                "proposal_track_used": (
                                    UNROUTABLE_TRACK
                                    if error_message
                                    else _routed_track_from_diagnosis_payload(
                                        parsed_payload,
                                        str(request_info.get("case_id")),
                                        request_info.get("visible_case_id")
                                        if isinstance(request_info.get("visible_case_id"), str)
                                        else None,
                                    )
                                ),
                                "routing_source": "diagnosis_prediction",
                                "context_audit": request_info.get("context_audit") or {},
                            },
                        ),
                    )

                    phase_summaries["diagnosis"] = build_batch_phase_summary(
                        request_count=diagnosis_request_count,
                        input_path=diagnosis_input_path,
                        request_manifest_path=diagnosis_manifest_path,
                        phase_label="diagnosis",
                        execution_result=diagnosis_execution,
                        elapsed_seconds=diagnosis_elapsed_seconds,
                        artifact_paths=diagnosis_artifacts,
                        sync_retry_fallback=diagnosis_retry_summary,
                    )

                    proposal_input_path = out_dir / "proposal_batch_input.jsonl"
                    proposal_manifest_path = out_dir / "proposal_batch_request_manifest.jsonl"
                    proposal_request_count = 0
                    with open(proposal_input_path, "w", encoding="utf-8") as batch_input_fh, open(
                        proposal_manifest_path, "w", encoding="utf-8"
                    ) as request_manifest_fh:
                        for bundle in bundle_list:
                            for record in _iter_materialized_generation_records(generation_selection):
                                if (
                                    _manifest_row_for_task(
                                        existing_manifest_lookup,
                                        bundle=bundle,
                                        case_id=record["id"],
                                        task_type="proposal",
                                    )
                                    is not None
                                ):
                                    continue
                                routing_info = routed_tracks.get((bundle, record["id"]))
                                if not isinstance(routing_info, dict):
                                    routing_info = {
                                        "proposal_track_used": UNROUTABLE_TRACK,
                                        "routing_source": "diagnosis_prediction",
                                        "context_audit": {},
                                    }
                                proposal_track_used = routing_info.get("proposal_track_used")
                                if proposal_track_used == "AMBIGUOUS":
                                    skipped_request_info = build_skipped_proposal_request(
                                        record,
                                        bundle,
                                        proposal_track_used="AMBIGUOUS",
                                        routing_source="diagnosis_prediction",
                                        context_audit=routing_info.get("context_audit"),
                                    )
                                    record_skipped_proposal(
                                        request_info=skipped_request_info,
                                        parse_status="skipped_ambiguous_track",
                                        skip_reason="Diagnosis predicted AMBIGUOUS track.",
                                        raw_log_fh=raw_log_fh,
                                        manifest_fh=manifest_fh,
                                    )
                                    mark_completed()
                                    continue
                                if proposal_track_used not in {"A_BOX", "T_BOX"}:
                                    skipped_request_info = build_skipped_proposal_request(
                                        record,
                                        bundle,
                                        proposal_track_used=UNROUTABLE_TRACK,
                                        routing_source="diagnosis_prediction",
                                        context_audit=routing_info.get("context_audit"),
                                    )
                                    record_skipped_proposal(
                                        request_info=skipped_request_info,
                                        parse_status="skipped_unroutable_track",
                                        skip_reason="Diagnosis output could not be routed to A_BOX or T_BOX.",
                                        raw_log_fh=raw_log_fh,
                                        manifest_fh=manifest_fh,
                                    )
                                    mark_completed()
                                    continue

                                world_state_entry = world_state_store.get(record["id"])
                                proposal_bundle, proposal_metadata = build_proposal_request(
                                    record,
                                    bundle,
                                    world_state_entry,
                                    proposal_track_used=proposal_track_used,
                                    routing_source="diagnosis_prediction",
                                )
                                proposal_custom_id = f"rf_prop_{proposal_request_count:09d}"
                                proposal_request_count += 1
                                provider.write_batch_request(
                                    batch_input_fh,
                                    custom_id=proposal_custom_id,
                                    prompt=proposal_bundle.prompt,
                                    system_prompt=proposal_bundle.system_prompt,
                                    response_format=proposal_bundle.response_format,
                                    metadata=proposal_metadata,
                                )
                                _append_jsonl_record(
                                    request_manifest_fh,
                                    {"custom_id": proposal_custom_id, "metadata": proposal_metadata},
                                )

                    proposal_execution = None
                    proposal_elapsed_seconds = 0.0
                    proposal_artifacts = {"output_path": None, "error_path": None, "job_path": None}
                    if proposal_request_count > 0:
                        _emit_runtime_status(
                            progress,
                            f"Prepared {proposal_request_count} routed proposal batch requests; submitting provider "
                            "batch job.",
                        )
                        batch_started_at = time.perf_counter()
                        proposal_execution = provider.execute_batch(
                            proposal_input_path,
                            request_manifest_path=proposal_manifest_path,
                            output_dir=out_dir,
                            completion_window=batch_completion_window,
                            poll_interval_seconds=batch_poll_interval_seconds,
                            status_callback=lambda message: _emit_runtime_status(progress, message),
                        )
                        proposal_elapsed_seconds = time.perf_counter() - batch_started_at
                        relocated_output, relocated_error, proposal_artifacts = relocate_batch_phase_artifacts(
                            output_path=proposal_execution.output_path,
                            error_path=proposal_execution.error_path,
                            phase_prefix="proposal_",
                        )
                        proposal_execution = type(proposal_execution)(
                            batch=proposal_execution.batch,
                            output_path=relocated_output,
                            error_path=relocated_error,
                        )
                        _emit_runtime_status(
                            progress,
                            "Proposal batch finished; rebuilding normalized proposal outputs.",
                        )

                    proposal_retry_summary = rebuild_batch_phase_results(
                        phase_label="proposal",
                        batch_execution=proposal_execution,
                        batch_input_path=proposal_input_path,
                        request_manifest_path=proposal_manifest_path,
                        bundle_handles=bundle_handles,
                        raw_log_fh=raw_log_fh,
                        manifest_fh=manifest_fh,
                    )

                    phase_summaries["proposal"] = build_batch_phase_summary(
                        request_count=proposal_request_count,
                        input_path=proposal_input_path,
                        request_manifest_path=proposal_manifest_path,
                        phase_label="proposal",
                        execution_result=proposal_execution,
                        elapsed_seconds=proposal_elapsed_seconds,
                        artifact_paths=proposal_artifacts,
                        sync_retry_fallback=proposal_retry_summary,
                    )
                    batch_summary = {
                        "mode": "two_stage",
                        "status": batch_status_from_phases(phase_summaries),
                        "overall_status": batch_status_from_phases(phase_summaries),
                        "elapsed_seconds": round(
                            sum(
                                phase.get("elapsed_seconds", 0.0)
                                for phase in phase_summaries.values()
                                if isinstance(phase.get("elapsed_seconds"), (int, float))
                            ),
                            6,
                        ),
                        "sync_retry_fallback": aggregate_sync_retry_fallback(phase_summaries),
                        "phases": phase_summaries,
                    }

        evaluation_case_count = len(selected_case_ids)
        evaluation_classified_path = Path(classified_path)
        evaluation_classified_records: list[dict[str, Any]] | None = None
        evaluation_strategy = "memory_cache"
        evaluation_filtered_record_count: int | None = None
        evaluation_filtered_path: Path | None = None
        if evaluation_case_count <= EVALUATION_IN_MEMORY_CASE_THRESHOLD:
            _emit_runtime_status(
                progress,
                "Preparing evaluation inputs with in-memory classified-record cache "
                f"for {evaluation_case_count} selected case(s).",
            )
            evaluation_classified_records = _load_selected_records_for_evaluation(
                classified_path,
                selected_case_ids,
            )
        else:
            evaluation_strategy = "filtered_subset_stream"
            evaluation_filtered_path = out_dir / "selected_classified_records.jsonl"
            _emit_runtime_status(
                progress,
                "Preparing evaluation inputs by writing filtered classified records to "
                f"{evaluation_filtered_path} for {evaluation_case_count} selected case(s).",
            )
            evaluation_filtered_record_count = _write_selected_records_for_evaluation(
                classified_path,
                selected_case_ids,
                evaluation_filtered_path,
            )
            evaluation_classified_path = evaluation_filtered_path

        _emit_runtime_status(
            progress,
            "Generation phase complete. Starting evaluation for "
            f"{len(bundle_list)} bundle(s) across {evaluation_case_count} selected case(s).",
        )
        evaluation_total = len(bundle_list) * len(selected_case_ids)
        evaluation_progress = tqdm(
            total=evaluation_total,
            desc="reasoning-floor eval",
            unit="case",
            disable=evaluation_total == 0,
        )

        try:
            for bundle in bundle_list:
                bundle_dir = out_dir / bundle
                evaluation_progress.set_postfix({"bundle": bundle}, refresh=True)
                _emit_runtime_status(
                    progress,
                    f"Evaluating bundle '{bundle}' ({len(selected_case_ids)} case(s)).",
                )
                evaluate_benchmark(
                    classified_path=evaluation_classified_path,
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
                    progress_callback=lambda _trace: evaluation_progress.update(1),
                    classified_records=evaluation_classified_records,
                    classified_input_path=classified_path,
                )
                _emit_runtime_status(
                    progress,
                    f"Finished evaluating bundle '{bundle}'.",
                )
        finally:
            evaluation_progress.close()

        _emit_runtime_status(progress, "Evaluation phase complete. Building combined run summary.")

        summary = summarize_trace_iterable(
            _iter_bundle_traces(out_dir, bundle_list),
            {
                "classified_benchmark": str(classified_path),
                "world_state": str(world_state_path),
                "run_id": run_id,
                "ablation_bundles": bundle_list,
                "provider": selected_provider,
                "model": selected_model,
                "openai_reasoning_effort": openai_reasoning_effort,
                "output_dir": str(out_dir),
                "selection_manifest": str(selection_manifest_path) if selection_manifest_path else None,
            },
        )
        failure_taxonomy = _failure_taxonomy_from_traces(_iter_bundle_traces(out_dir, bundle_list))
        run_elapsed_seconds = time.perf_counter() - run_started_at
        run_finished_utc = _utc_now()
        run_usage = _aggregate_run_usage(usage_manifest)
        cost_estimation_modes = sorted(
            {
                mode
                for record in usage_manifest
                for mode in [((record.get("usage") or {}).get("cost_estimation_mode"))]
                if isinstance(mode, str) and mode
            }
        )
        cost_estimation_multipliers = sorted(
            {
                round(float(multiplier), 10)
                for record in usage_manifest
                for multiplier in [((record.get("usage") or {}).get("cost_estimation_multiplier"))]
                if isinstance(multiplier, (int, float))
            }
        )
        batch_pricing_applied = any(
            bool((record.get("usage") or {}).get("batch_pricing_applied")) for record in usage_manifest
        )
        summary["run_info"] = {
            "run_id": run_id,
            "provider": selected_provider,
            "model": selected_model,
            "openai_reasoning_effort": openai_reasoning_effort,
            "output_dir": str(out_dir),
            "started_at_utc": run_started_utc,
            "finished_at_utc": run_finished_utc,
            "elapsed_seconds": round(run_elapsed_seconds, 6),
            "generation_elapsed_seconds": run_usage["generation_elapsed_seconds"],
            "execution_mode": normalized_execution_mode,
            "batch_mode_used": normalized_execution_mode == "batch",
            "evaluation": {
                "classified_record_strategy": evaluation_strategy,
                "selected_case_count": evaluation_case_count,
                "memory_cache_case_threshold": EVALUATION_IN_MEMORY_CASE_THRESHOLD,
                "filtered_classified_path": str(evaluation_filtered_path) if evaluation_filtered_path else None,
                "filtered_record_count": evaluation_filtered_record_count,
            },
            "generation": {
                "classified_record_strategy": generation_selection.strategy,
                "selected_generation_records_path": (
                    str(generation_selection.records_path) if generation_selection.records_path else None
                ),
                "selected_case_count": len(selected_case_ids),
                "memory_cache_case_threshold": GENERATION_IN_MEMORY_CASE_THRESHOLD,
            },
            "proposal_track_mode": normalized_proposal_track_mode,
            "oracle_diagnosis_mode": normalized_oracle_diagnosis_mode,
            "expected_request_count": expected_request_count,
            "planned_diagnosis_request_count": planned_diagnosis_request_count,
            "planned_proposal_request_count": planned_proposal_request_count,
            "skipped_diagnosis_count": skipped_diagnosis_count,
            "resume": {
                "enabled": resume_requested,
                "resumed_from": str(out_dir) if resume_requested else None,
                "run_config_path": str(run_config_path),
                "existing_manifest_rows": len(existing_manifest_rows),
                "completed_work_units_before_resume": existing_completed_work_units if resume_requested else 0,
                "original_started_at_utc": (
                    existing_run_config.get("started_at_utc")
                    if isinstance(existing_run_config, dict)
                    else None
                ),
            },
        }
        if normalized_execution_mode == "parallel":
            summary["run_info"]["parallel"] = {
                "workers": effective_parallel_workers,
                "source": parallel_worker_source,
            }
        if batch_summary is not None:
            summary["run_info"]["batch"] = batch_summary
        summary["usage"] = {
            "prompt_tokens": run_usage["prompt_tokens"],
            "completion_tokens": run_usage["completion_tokens"],
            "total_tokens": run_usage["total_tokens"],
            "cached_tokens": run_usage["cached_tokens"],
            "estimated_cost_usd": run_usage["estimated_cost_usd"],
            "batch_pricing_applied": batch_pricing_applied,
            "cost_estimation_mode": (
                cost_estimation_modes[0]
                if len(cost_estimation_modes) == 1
                else ("mixed" if cost_estimation_modes else None)
            ),
            "per_call_cost_estimation_modes": cost_estimation_modes,
            "per_call_cost_estimation_multipliers": cost_estimation_multipliers,
            "cost_estimation_multiplier": (
                cost_estimation_multipliers[0] if len(cost_estimation_multipliers) == 1 else None
            ),
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
        _emit_runtime_status(progress, f"Run complete. Summary written to {out_dir / 'reasoning_floor_summary.json'}.")
        return summary
    finally:
        if pending_progress:
            progress.update(pending_progress)
        progress.close()
        world_state_store.close()
