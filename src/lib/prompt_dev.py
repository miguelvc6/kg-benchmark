from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from classifier import WorldStateStore
from guardian.evaluator import evaluate_benchmark, write_json
from guardian.model_provider import ModelProvider, create_model_provider
from guardian.patch_parser import normalize_proposal as normalize_a_box_proposal
from guardian.reasoning import ABLATION_BUNDLES, _bundle_payload_and_audit, _t_box_constraint_type_qids
from guardian.tbox_parser import normalize_proposal as normalize_t_box_proposal
from guardian.track_parser import normalize_diagnosis
from lib.benchmark_selection import derive_case_metadata, group_key_for_record, load_selection_manifest
from lib.repair_state import derive_value_change_summary, normalize_value_list
from lib.utils import iter_jsonl
from scripts.prompt_dev_templates import (
    PROMPT_DEV_VERSION,
    REPRESENTATIONS,
    render_prompt_dev_prompt,
)

EXAMPLE_POLICIES = (
    "zero_shot",
    "random_same_task_2shot",
    "same_track_2shot",
    "matched_2shot",
)
REPAIR_TRACK_MODES = ("oracle", "diagnosis_routed")
DEFAULT_CONTEXT_BUNDLES = ("logic_only", "local_graph")
DEFAULT_RENDER_TASKS = ("track_diagnosis", "repair_proposal")
SAMPLE_STRATEGIES = ("manifest_order", "stratified", "diverse_stratified")
T_BOX_ACTIONS = {
    "RELAXATION_RANGE_WIDENED",
    "RESTRICTION_RANGE_NARROWED",
    "RELAXATION_SET_EXPANSION",
    "RESTRICTION_SET_CONTRACTION",
    "SCHEMA_UPDATE",
    "COINCIDENTAL_SCHEMA_CHANGE",
}


@dataclass(frozen=True)
class PromptDevMatrixOptions:
    representations: tuple[str, ...] = REPRESENTATIONS
    example_policies: tuple[str, ...] = EXAMPLE_POLICIES
    context_bundles: tuple[str, ...] = DEFAULT_CONTEXT_BUNDLES
    tasks: tuple[str, ...] = DEFAULT_RENDER_TASKS
    repair_track_modes: tuple[str, ...] = REPAIR_TRACK_MODES
    include_abstention: bool = False


@dataclass(frozen=True)
class PromptDevRenderOptions:
    classified_benchmark: Path
    world_state: Path
    dev_manifest: Path
    output_dir: Path
    seed: int = 13
    max_cases: int | None = 24
    representations: tuple[str, ...] = ("hybrid_json_nl",)
    example_policies: tuple[str, ...] = ("zero_shot",)
    context_bundles: tuple[str, ...] = DEFAULT_CONTEXT_BUNDLES
    tasks: tuple[str, ...] = DEFAULT_RENDER_TASKS
    repair_track_modes: tuple[str, ...] = ("oracle",)
    include_abstention: bool = False
    core_manifest: Path | None = None
    example_count: int = 2
    allow_same_property_examples: bool = False
    sample_strategy: str = "stratified"
    allow_core_example_risk: bool = False


@dataclass(frozen=True)
class PromptDevEvaluateOptions:
    classified_benchmark: Path
    world_state: Path
    dev_manifest: Path
    output_dir: Path
    model_endpoint: str | None = None
    model_name: str | None = None
    seed: int = 13
    max_cases: int | None = 24
    representations: tuple[str, ...] = ("hybrid_json_nl",)
    example_policies: tuple[str, ...] = ("zero_shot",)
    context_bundles: tuple[str, ...] = DEFAULT_CONTEXT_BUNDLES
    tasks: tuple[str, ...] = DEFAULT_RENDER_TASKS
    repair_track_modes: tuple[str, ...] = ("oracle",)
    include_abstention: bool = False
    core_manifest: Path | None = None
    example_count: int = 2
    allow_same_property_examples: bool = False
    resume_existing: bool = True
    retry_failures: bool = False
    max_prompt_chars: int | None = None
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    sample_strategy: str = "stratified"
    allow_core_example_risk: bool = False


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_hash(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _ordered_csv(values: Iterable[str] | None, default: Iterable[str]) -> tuple[str, ...]:
    result = tuple(value.strip() for value in values or default if isinstance(value, str) and value.strip())
    return result or tuple(default)


def _classification_parts(record: dict[str, Any]) -> tuple[str, str]:
    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    cls = classification.get("class")
    subtype = classification.get("subtype")
    return (
        cls if isinstance(cls, str) and cls else "unknown",
        subtype if isinstance(subtype, str) and subtype else "unknown",
    )


def _manifest_annotation(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    annotations = manifest.get("case_annotations") if isinstance(manifest.get("case_annotations"), dict) else {}
    annotation = annotations.get(case_id)
    return annotation if isinstance(annotation, dict) else {}


def _selection_stratum(manifest: dict[str, Any], case_id: str) -> str:
    annotation = _manifest_annotation(manifest, case_id)
    value = annotation.get("selection_stratum")
    return value if isinstance(value, str) and value else "unknown"


def _prompt_dev_stratum_key(manifest: dict[str, Any], record: dict[str, Any]) -> tuple[str, str, str, str]:
    cls, subtype = _classification_parts(record)
    return (
        str(record.get("track") or "unknown"),
        cls,
        subtype,
        _selection_stratum(manifest, str(record.get("id") or "")),
    )


def _stratified_case_ids(
    ordered_ids: list[str],
    by_id: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    *,
    max_cases: int,
    seed: int,
    diversify_instances: bool = False,
) -> list[str]:
    if max_cases <= 0:
        return []
    buckets: dict[tuple[str, str, str, str], list[str]] = {}
    for case_id in ordered_ids:
        record = by_id.get(case_id)
        if record is None:
            continue
        buckets.setdefault(_prompt_dev_stratum_key(manifest, record), []).append(case_id)
    for key, values in buckets.items():
        values.sort(key=lambda case_id: (_stable_hash(seed, "prompt_dev_sample", *key, case_id), case_id))

    track_to_keys: dict[str, list[tuple[str, str, str, str]]] = {}
    for key in sorted(buckets):
        track_to_keys.setdefault(key[0], []).append(key)
    track_order = sorted(track_to_keys)
    selected: list[str] = []
    selected_set: set[str] = set()
    used_qids: set[str] = set()
    used_properties: set[str] = set()
    track_positions = {track: 0 for track in track_order}

    def pop_candidate(key: tuple[str, str, str, str]) -> str | None:
        while buckets[key] and buckets[key][0] in selected_set:
            buckets[key].pop(0)
        if not buckets[key]:
            return None
        if not diversify_instances:
            return buckets[key].pop(0)
        best_index = 0
        best_score: tuple[int, int, str] | None = None
        for index, candidate_id in enumerate(buckets[key]):
            if candidate_id in selected_set:
                continue
            candidate = by_id.get(candidate_id) or {}
            qid = candidate.get("qid")
            pid = candidate.get("property")
            score = (
                0 if isinstance(qid, str) and qid in used_qids else 1,
                0 if isinstance(pid, str) and pid in used_properties else 1,
                _stable_hash(seed, "diverse_prompt_dev_sample", candidate_id),
            )
            if best_score is None or score > best_score:
                best_index = index
                best_score = score
        return buckets[key].pop(best_index)

    while len(selected) < max_cases:
        made_progress = False
        for track in track_order:
            keys = track_to_keys[track]
            if not keys:
                continue
            for _ in range(len(keys)):
                key = keys[track_positions[track] % len(keys)]
                track_positions[track] += 1
                case_id = pop_candidate(key)
                if case_id is None:
                    continue
                selected.append(case_id)
                selected_set.add(case_id)
                record = by_id.get(case_id) or {}
                qid = record.get("qid")
                pid = record.get("property")
                if isinstance(qid, str):
                    used_qids.add(qid)
                if isinstance(pid, str):
                    used_properties.add(pid)
                made_progress = True
                break
            if len(selected) >= max_cases:
                break
        if not made_progress:
            break
    return selected


def build_prompt_dev_matrix(options: PromptDevMatrixOptions | None = None) -> dict[str, Any]:
    options = options or PromptDevMatrixOptions()
    rows: list[dict[str, Any]] = []
    for representation in options.representations:
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unsupported representation: {representation}")
        for example_policy in options.example_policies:
            if example_policy not in EXAMPLE_POLICIES:
                raise ValueError(f"Unsupported example policy: {example_policy}")
            for context_bundle in options.context_bundles:
                if context_bundle not in ABLATION_BUNDLES:
                    raise ValueError(f"Unsupported context bundle: {context_bundle}")
                for task in options.tasks:
                    if task == "track_diagnosis":
                        rows.append(
                            _matrix_row(
                                representation=representation,
                                example_policy=example_policy,
                                context_bundle=context_bundle,
                                task=task,
                                track_mode=None,
                                include_abstention=False,
                            )
                        )
                    elif task == "repair_proposal":
                        for track_mode in options.repair_track_modes:
                            rows.append(
                                _matrix_row(
                                    representation=representation,
                                    example_policy=example_policy,
                                    context_bundle=context_bundle,
                                    task=task,
                                    track_mode=track_mode,
                                    include_abstention=options.include_abstention,
                                )
                            )
                    else:
                        raise ValueError(f"Unsupported matrix task: {task}")
    for index, row in enumerate(rows, start=1):
        row["matrix_id"] = f"prompt_dev_{index:03d}_{row['matrix_key']}"
    return {
        "manifest_type": "prompt_dev_matrix",
        "manifest_version": PROMPT_DEV_VERSION,
        "created_at_utc": _utc_now(),
        "counts": {
            "rows": len(rows),
            "by_representation": dict(Counter(row["representation"] for row in rows)),
            "by_example_policy": dict(Counter(row["example_policy"] for row in rows)),
            "by_context_bundle": dict(Counter(row["context_bundle"] for row in rows)),
            "by_task": dict(Counter(row["task"] for row in rows)),
        },
        "selection_criteria": [
            "Select final prompts before main core inference.",
            "Primary: parse validity, proposal executability, exact historical agreement, token count.",
            "Secondary: track diagnosis accuracy, T-box semantic-family success, auditability completeness.",
        ],
        "rows": rows,
    }


def _matrix_row(
    *,
    representation: str,
    example_policy: str,
    context_bundle: str,
    task: str,
    track_mode: str | None,
    include_abstention: bool,
) -> dict[str, Any]:
    key = "_".join(
        part
        for part in (
            representation,
            example_policy,
            context_bundle,
            task,
            track_mode or "diagnosis",
            "abstain" if include_abstention else "no_abstain",
        )
        if part
    )
    return {
        "matrix_key": key,
        "representation": representation,
        "example_policy": example_policy,
        "context_bundle": context_bundle,
        "task": task,
        "track_mode": track_mode,
        "include_abstention": include_abstention,
        "run_scope": "dev_only",
        "metrics": [
            "parse_validity",
            "proposal_contract_validity",
            "proposal_executability",
            "exact_historical_agreement",
            "t_box_target_constraint_hit",
            "t_box_semantic_family_success",
            "track_diagnosis_accuracy",
            "auditability_completeness",
            "provenance_completeness",
            "request_error_rate",
            "tokens_per_case",
            "estimated_cost",
            "latency",
        ],
    }


def write_prompt_dev_matrix(path: str | Path, options: PromptDevMatrixOptions | None = None) -> dict[str, Any]:
    matrix = build_prompt_dev_matrix(options)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(_matrix_markdown(matrix), encoding="utf-8")
    return matrix


def _matrix_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# Prompt Development Matrix",
        "",
        f"Prompt version: `{matrix['manifest_version']}`",
        f"Rows: {matrix['counts']['rows']}",
        "",
        "| Matrix id | Representation | Examples | Context | Task | Track mode | Abstention |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in matrix["rows"]:
        markdown_row = (
            "| {matrix_id} | {representation} | {example_policy} | {context_bundle} | "
            "{task} | {track_mode} | {include_abstention} |"
        )
        lines.append(
            markdown_row.format(
                **{**row, "track_mode": row.get("track_mode") or ""}
            )
        )
    lines.append("")
    return "\n".join(lines)


def _load_manifest_records(
    classified_path: Path,
    manifest_path: Path,
    *,
    max_cases: int | None,
    sample_strategy: str = "stratified",
    seed: int = 13,
) -> list[dict[str, Any]]:
    if sample_strategy not in SAMPLE_STRATEGIES:
        raise ValueError(f"Unsupported prompt-dev sample strategy: {sample_strategy}")
    manifest = load_selection_manifest(manifest_path)
    ids = [case_id for case_id in manifest.get("selected_case_ids", []) if isinstance(case_id, str)]
    id_set = set(ids)
    by_id = {
        record["id"]: record
        for record in iter_jsonl(classified_path)
        if isinstance(record, dict) and isinstance(record.get("id"), str) and record["id"] in id_set
    }
    ids = [case_id for case_id in ids if case_id in by_id]
    if max_cases is not None:
        limit = max(0, max_cases)
        if sample_strategy == "manifest_order":
            ids = ids[:limit]
        else:
            ids = _stratified_case_ids(
                ids,
                by_id,
                manifest,
                max_cases=limit,
                seed=seed,
                diversify_instances=sample_strategy == "diverse_stratified",
            )
    return [by_id[case_id] for case_id in ids if case_id in by_id]


def _load_all_manifest_records(classified_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    return _load_manifest_records(classified_path, manifest_path, max_cases=None, sample_strategy="manifest_order")


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    return derive_case_metadata(record, tier="dev") or derive_case_metadata(record, tier="core") or {}


def _info_condition(record: dict[str, Any]) -> str:
    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    cls = classification.get("class")
    if cls == "TypeA":
        return "IC-L"
    if cls == "TypeB":
        return "IC-G"
    if cls == "TypeC":
        subtype = classification.get("subtype")
        return "IC-E-elim" if subtype == "EXTERNAL_BY_ELIMINATION" else "IC-U"
    if cls == "T_BOX":
        return "T-BOX"
    return "unknown"


def _value_datatype(record: dict[str, Any]) -> str:
    summary = derive_value_change_summary(record)
    values = summary.new_values or summary.old_values
    kinds: set[str] = set()
    for value in values:
        if value.startswith("Q"):
            kinds.add("qid")
        elif value.startswith("P"):
            kinds.add("pid")
        else:
            try:
                float(value)
            except ValueError:
                if value.startswith(("+", "-")) and "T" in value:
                    kinds.add("date")
                else:
                    kinds.add("literal")
            else:
                kinds.add("numeric")
    return next(iter(kinds)) if len(kinds) == 1 else ("mixed" if kinds else "none")


def _repair_action(record: dict[str, Any]) -> str:
    rt = record.get("repair_target") if isinstance(record.get("repair_target"), dict) else {}
    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    return str(rt.get("action") or classification.get("subtype") or "unknown")


def _visible_case_id_map(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for index, record in enumerate(records, start=1):
        case_id = record.get("id")
        if isinstance(case_id, str) and case_id and case_id not in mapping:
            mapping[case_id] = f"case_{index:06d}"
    return mapping


def _replace_exact_string(value: Any, old: str, new: str) -> Any:
    if value == old:
        return new
    if isinstance(value, dict):
        return {key: _replace_exact_string(child, old, new) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_exact_string(child, old, new) for child in value]
    return value


def _model_visible_payload(payload: dict[str, Any], *, raw_case_id: str, visible_case_id: str) -> dict[str, Any]:
    visible = _replace_exact_string(payload, raw_case_id, visible_case_id)
    if isinstance(visible, dict):
        visible["id"] = visible_case_id
    return visible if isinstance(visible, dict) else payload


def _model_visible_output(
    payload: dict[str, Any] | None,
    *,
    raw_case_id: str,
    visible_case_id: str,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    visible = _replace_exact_string(payload, raw_case_id, visible_case_id)
    if isinstance(visible, dict):
        visible["case_id"] = visible_case_id
    return visible if isinstance(visible, dict) else payload


def _case_summary_counts(
    records: Iterable[dict[str, Any]],
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record_list = list(records)
    return {
        "by_track": dict(Counter(str(record.get("track") or "unknown") for record in record_list)),
        "by_class": dict(Counter(_classification_parts(record)[0] for record in record_list)),
        "by_subtype": dict(Counter(_classification_parts(record)[1] for record in record_list)),
        "by_class_subtype": dict(Counter(":".join(_classification_parts(record)) for record in record_list)),
        "by_selection_stratum": dict(
            Counter(
                _selection_stratum(manifest, str(record.get("id") or "")) if manifest is not None else "unknown"
                for record in record_list
            )
        ),
        "unique_qids": len({record.get("qid") for record in record_list if isinstance(record.get("qid"), str)}),
        "unique_properties": len(
            {record.get("property") for record in record_list if isinstance(record.get("property"), str)}
        ),
    }


def _example_sort_key(
    seed: int,
    eval_record: dict[str, Any],
    candidate: dict[str, Any],
    policy: str,
) -> tuple[Any, ...]:
    eval_meta = _metadata(eval_record)
    cand_meta = _metadata(candidate)
    score = 0
    if candidate.get("track") == eval_record.get("track"):
        score += 20
    if cand_meta.get("constraint_family") == eval_meta.get("constraint_family"):
        score += 10
    if cand_meta.get("subtype") == eval_meta.get("subtype"):
        score += 8
    if _repair_action(candidate) == _repair_action(eval_record):
        score += 6
    if _info_condition(candidate) == _info_condition(eval_record):
        score += 4
    if _value_datatype(candidate) == _value_datatype(eval_record):
        score += 2
    if cand_meta.get("popularity_bucket") == eval_meta.get("popularity_bucket"):
        score += 1
    if policy == "random_same_task_2shot":
        score = 0
    elif policy == "same_track_2shot" and candidate.get("track") == eval_record.get("track"):
        score = max(score, 20)
    return (-score, _stable_hash(seed, policy, eval_record.get("id"), candidate.get("id")), candidate.get("id"))


def _blocked_core_sets(core_manifest_path: Path | None) -> dict[str, set[str]]:
    if core_manifest_path is None or not core_manifest_path.exists():
        return {"case_ids": set(), "group_keys": set(), "tbox_revision_keys": set()}
    manifest = load_selection_manifest(core_manifest_path)
    annotations = manifest.get("case_annotations") if isinstance(manifest.get("case_annotations"), dict) else {}
    return {
        "case_ids": set(manifest.get("selected_case_ids", [])),
        "group_keys": {
            ann.get("group_key")
            for ann in annotations.values()
            if isinstance(ann, dict) and isinstance(ann.get("group_key"), str)
        },
        "tbox_revision_keys": {
            ann.get("tbox_revision_key")
            for ann in annotations.values()
            if isinstance(ann, dict) and isinstance(ann.get("tbox_revision_key"), str)
        },
    }


def select_examples(
    *,
    eval_record: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    policy: str,
    task: str,
    seed: int,
    example_count: int = 2,
    blocked_core: dict[str, set[str]] | None = None,
    allow_same_property: bool = False,
) -> list[dict[str, Any]]:
    if policy == "zero_shot" or example_count <= 0:
        return []
    if policy not in EXAMPLE_POLICIES:
        raise ValueError(f"Unsupported example policy: {policy}")
    blocked_core = blocked_core or {"case_ids": set(), "group_keys": set(), "tbox_revision_keys": set()}
    eval_case_id = eval_record.get("id")
    eval_qid = eval_record.get("qid")
    eval_property = eval_record.get("property")
    eval_group_key, eval_tbox_key, _ = group_key_for_record(eval_record)

    candidates: list[dict[str, Any]] = []
    for candidate in candidate_records:
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or candidate_id == eval_case_id:
            continue
        candidate_group_key, candidate_tbox_key, _ = group_key_for_record(candidate)
        if candidate_id in blocked_core["case_ids"]:
            continue
        if candidate_group_key in blocked_core["group_keys"]:
            continue
        if isinstance(candidate_tbox_key, str) and candidate_tbox_key in blocked_core["tbox_revision_keys"]:
            continue
        if candidate.get("qid") == eval_qid and eval_qid:
            continue
        if candidate_tbox_key == eval_tbox_key and eval_tbox_key:
            continue
        if not allow_same_property and candidate.get("property") == eval_property:
            continue
        if policy in {"same_track_2shot", "matched_2shot"} and candidate.get("track") != eval_record.get("track"):
            continue
        candidates.append(candidate)

    candidates.sort(key=lambda candidate: _example_sort_key(seed, eval_record, candidate, policy))
    examples: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    for candidate in candidates:
        candidate_group_key, _, _ = group_key_for_record(candidate)
        if candidate_group_key in used_groups:
            continue
        output_payload = _gold_output_for_task(candidate, task)
        if not output_payload:
            continue
        used_groups.add(candidate_group_key)
        examples.append(
            {
                "case_id": candidate["id"],
                "group_key": candidate_group_key,
                "track": candidate.get("track"),
                "input_payload": {},
                "output_payload": output_payload,
            }
        )
        if len(examples) >= example_count:
            break
    return examples


def _gold_output_for_task(record: dict[str, Any], task: str) -> dict[str, Any] | None:
    if task == "track_diagnosis":
        return {
            "case_id": record.get("id"),
            "predicted_track": record.get("track") if record.get("track") in {"A_BOX", "T_BOX"} else "AMBIGUOUS",
            "confidence": "high",
            "rationale": "The demonstration answer uses this dev example's historical repair locus.",
        }
    if task == "a_box_repair":
        return _gold_a_box_output(record)
    if task == "t_box_repair":
        return _gold_t_box_output(record)
    return None


def _gold_a_box_output(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("track") != "A_BOX":
        return None
    qid = record.get("qid")
    pid = record.get("property")
    if not isinstance(qid, str) or not isinstance(pid, str):
        return None
    repair_target = record.get("repair_target") if isinstance(record.get("repair_target"), dict) else {}
    action = repair_target.get("action")
    new_values = normalize_value_list(repair_target.get("new_value"))
    if not new_values and action in {"CREATE", "UPDATE"}:
        new_values = normalize_value_list(repair_target.get("value"))
    ops: list[dict[str, Any]] = []
    if not new_values:
        ops.append({"op": "DELETE_ALL", "pid": pid})
    elif len(new_values) == 1:
        ops.append({"op": "SET", "pid": pid, "value": new_values[0], "rank": "normal"})
    else:
        ops.append({"op": "DELETE_ALL", "pid": pid})
        ops.extend({"op": "ADD", "pid": pid, "value": value, "rank": "normal"} for value in new_values)
    return {
        "case_id": record.get("id"),
        "target": {"qid": qid, "pid": pid},
        "ops": ops,
        "rationale": "Demonstration answer reconstructed from the dev example's historical repaired value.",
        "provenance": [{"kind": "OTHER", "snippet": "dev example visible repair target"}],
        "uncertainty": {"confidence": 0.95, "notes": "Gold demonstration only; not a model output."},
    }


def _gold_t_box_output(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("track") != "T_BOX":
        return None
    pid = record.get("property")
    repair_target = record.get("repair_target") if isinstance(record.get("repair_target"), dict) else {}
    delta = repair_target.get("constraint_delta") if isinstance(repair_target.get("constraint_delta"), dict) else {}
    signature_after = delta.get("signature_after") or delta.get("new_constraints")
    if not isinstance(pid, str) or not isinstance(signature_after, list) or not signature_after:
        return None
    changed = delta.get("changed_constraint_types")
    target_constraint = changed[0] if isinstance(changed, list) and changed else None
    if not isinstance(target_constraint, str):
        first = signature_after[0] if isinstance(signature_after[0], dict) else {}
        target_constraint = first.get("constraint_qid")
    if not isinstance(target_constraint, str):
        return None
    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    action = classification.get("subtype") if classification.get("subtype") in T_BOX_ACTIONS else "SCHEMA_UPDATE"
    return {
        "case_id": record.get("id"),
        "target": {"pid": pid, "constraint_type_qid": target_constraint},
        "proposal": {"action": action, "signature_after": signature_after},
        "rationale": "Demonstration answer reconstructed from the dev example's historical constraint signature.",
        "provenance": [{"kind": "OTHER", "snippet": "dev example visible property-revision context"}],
        "uncertainty": {"confidence": 0.95, "notes": "Gold demonstration only; not a model output."},
    }


def _task_for_record(matrix_task: str, record: dict[str, Any], track_mode: str | None) -> str | None:
    if matrix_task == "track_diagnosis":
        return "track_diagnosis"
    proposal_track = record.get("track") if track_mode in {None, "oracle", "diagnosis_routed"} else None
    if proposal_track == "A_BOX":
        return "a_box_repair"
    if proposal_track == "T_BOX":
        return "t_box_repair"
    return None


def _uses_few_shot(example_policies: Iterable[str]) -> bool:
    return any(policy != "zero_shot" for policy in example_policies)


def _validate_core_example_guard(options: PromptDevRenderOptions | PromptDevEvaluateOptions) -> None:
    if not _uses_few_shot(options.example_policies):
        return
    if options.core_manifest is not None and options.core_manifest.exists():
        return
    if options.allow_core_example_risk:
        return
    raise ValueError(
        "Few-shot prompt development requires --core-manifest so examples can exclude core cases. "
        "Pass --allow-core-example-risk only for an explicit leakage-risk experiment."
    )


def render_prompt_dev_prompts(options: PromptDevRenderOptions) -> dict[str, Any]:
    log = logging.getLogger("prompt_dev")
    log.info(
        "render: loading manifest records classified=%s manifest=%s max_cases=%s sample_strategy=%s",
        options.classified_benchmark,
        options.dev_manifest,
        options.max_cases,
        options.sample_strategy,
    )
    _validate_core_example_guard(options)
    manifest = load_selection_manifest(options.dev_manifest)
    eval_records = _load_manifest_records(
        options.classified_benchmark,
        options.dev_manifest,
        max_cases=options.max_cases,
        sample_strategy=options.sample_strategy,
        seed=options.seed,
    )
    log.info("render: selected eval records=%s", len(eval_records))
    log.info("render: loading candidate records for examples and visible ids")
    candidate_records = _load_all_manifest_records(options.classified_benchmark, options.dev_manifest)
    log.info("render: loaded candidate records=%s", len(candidate_records))
    visible_case_ids = _visible_case_id_map(candidate_records)
    blocked_core = _blocked_core_sets(options.core_manifest)
    matrix = build_prompt_dev_matrix(
        PromptDevMatrixOptions(
            representations=options.representations,
            example_policies=options.example_policies,
            context_bundles=options.context_bundles,
            tasks=options.tasks,
            repair_track_modes=options.repair_track_modes,
            include_abstention=options.include_abstention,
        )
    )
    log.info("render: matrix rows=%s", matrix["counts"]["rows"])
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / "prompt_dev_rendered_prompts.jsonl"
    summary_path = output_dir / "prompt_dev_render_summary.json"
    review_path = output_dir / "prompt_dev_prompt_review.md"

    rendered_count = 0
    skipped_count = 0
    prompt_records: list[dict[str, Any]] = []
    log.info("render: opening world-state store path=%s", options.world_state)
    with WorldStateStore(options.world_state, log) as world_store:
        log.info("render: world-state store ready entries=%s", len(world_store))
        for record in eval_records:
            raw_case_id = record["id"]
            visible_case_id = visible_case_ids.get(raw_case_id, f"case_{_stable_hash(raw_case_id)[:12]}")
            world_state_entry = world_store.get(record["id"])
            for row in matrix["rows"]:
                task = _task_for_record(row["task"], record, row.get("track_mode"))
                if task is None:
                    skipped_count += 1
                    continue
                case_payload, context_audit = _bundle_payload_and_audit(
                    record,
                    world_state_entry,
                    row["context_bundle"],
                )
                case_payload = _model_visible_payload(
                    case_payload,
                    raw_case_id=raw_case_id,
                    visible_case_id=visible_case_id,
                )
                examples = select_examples(
                    eval_record=record,
                    candidate_records=candidate_records,
                    policy=row["example_policy"],
                    task=task,
                    seed=options.seed,
                    example_count=options.example_count,
                    blocked_core=blocked_core,
                    allow_same_property=options.allow_same_property_examples,
                )
                for example in examples:
                    candidate = next((item for item in candidate_records if item.get("id") == example["case_id"]), None)
                    if candidate is None:
                        continue
                    candidate_raw_case_id = candidate["id"]
                    candidate_visible_case_id = visible_case_ids.get(
                        candidate_raw_case_id,
                        f"case_{_stable_hash(candidate_raw_case_id)[:12]}",
                    )
                    candidate_world = world_store.get(candidate["id"])
                    example_payload, _ = _bundle_payload_and_audit(candidate, candidate_world, row["context_bundle"])
                    example["input_payload"] = _model_visible_payload(
                        example_payload,
                        raw_case_id=candidate_raw_case_id,
                        visible_case_id=candidate_visible_case_id,
                    )
                    visible_output = _model_visible_output(
                        example.get("output_payload") if isinstance(example.get("output_payload"), dict) else None,
                        raw_case_id=candidate_raw_case_id,
                        visible_case_id=candidate_visible_case_id,
                    )
                    if visible_output is not None:
                        example["output_payload"] = visible_output
                    example["visible_case_id"] = candidate_visible_case_id
                rendered = render_prompt_dev_prompt(
                    task=task,
                    representation=row["representation"],
                    case_payload=case_payload,
                    examples=examples,
                    include_abstention=row["include_abstention"],
                )
                prompt_record = {
                    "matrix_id": row["matrix_id"],
                    "case_id": raw_case_id,
                    "visible_case_id": visible_case_id,
                    "task": task,
                    "historical_track": record.get("track"),
                    "representation": row["representation"],
                    "example_policy": row["example_policy"],
                    "context_bundle": row["context_bundle"],
                    "track_mode": row.get("track_mode"),
                    "include_abstention": row["include_abstention"],
                    "prompt_name": rendered.prompt_name,
                    "system_prompt": rendered.system_prompt,
                    "user_prompt": rendered.user_prompt,
                    "response_format": rendered.response_format,
                    "context_audit": context_audit,
                    "examples": [
                        {key: value for key, value in example.items() if key != "input_payload"}
                        for example in examples
                    ],
                }
                prompt_records.append(prompt_record)
                rendered_count += 1

    log.info("render: writing prompt artifacts prompts=%s", rendered_count)
    with prompts_path.open("w", encoding="utf-8") as handle:
        for prompt_record in prompt_records:
            handle.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")

    summary = {
        "manifest_type": "prompt_dev_rendered_prompt_summary",
        "manifest_version": PROMPT_DEV_VERSION,
        "created_at_utc": _utc_now(),
        "inputs": {
            "classified_benchmark": str(options.classified_benchmark),
            "world_state": str(options.world_state),
            "dev_manifest": str(options.dev_manifest),
            "core_manifest": str(options.core_manifest) if options.core_manifest else None,
            "sample_strategy": options.sample_strategy,
            "max_cases": options.max_cases,
        },
        "outputs": {
            "prompts_jsonl": str(prompts_path),
            "review_markdown": str(review_path),
        },
        "counts": {
            "eval_cases": len(eval_records),
            "matrix_rows": len(matrix["rows"]),
            "rendered_prompts": rendered_count,
            "skipped_prompts": skipped_count,
            "by_task": dict(Counter(record["task"] for record in prompt_records)),
            "by_representation": dict(Counter(record["representation"] for record in prompt_records)),
            "by_example_policy": dict(Counter(record["example_policy"] for record in prompt_records)),
            "by_context_bundle": dict(Counter(record["context_bundle"] for record in prompt_records)),
            **_case_summary_counts(eval_records, manifest),
        },
        "note": "No LLM inference was run. These artifacts contain prompts only.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review_path.write_text(_prompt_review_markdown(prompt_records, summary), encoding="utf-8")
    log.info("render: done rendered=%s skipped=%s output_dir=%s", rendered_count, skipped_count, output_dir)
    return summary


def _prompt_review_markdown(prompt_records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Prompt Development Review",
        "",
        "No LLM inference was run for this artifact.",
        "",
        f"Rendered prompts: `{summary['counts']['rendered_prompts']}`",
        "",
    ]
    for record in prompt_records[:12]:
        lines.extend(
            [
                f"## {record['matrix_id']} / {record.get('visible_case_id') or record['case_id']}",
                "",
                f"- Task: `{record['task']}`",
                f"- Representation: `{record['representation']}`",
                f"- Examples: `{record['example_policy']}`",
                f"- Context: `{record['context_bundle']}`",
                "",
                "System prompt:",
                "```text",
                record["system_prompt"],
                "```",
                "",
                "User prompt:",
                "```text",
                record["user_prompt"][:6000],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def freeze_prompt_dev_config(
    *,
    output: str | Path,
    representation: str,
    example_policy: str,
    context_bundles: Iterable[str],
    proposal_track_modes: Iterable[str],
    include_abstention: bool,
    notes: str = "",
) -> dict[str, Any]:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unsupported representation: {representation}")
    if example_policy not in EXAMPLE_POLICIES:
        raise ValueError(f"Unsupported example policy: {example_policy}")
    config = {
        "manifest_type": "prompt_dev_frozen_prompt_config",
        "manifest_version": PROMPT_DEV_VERSION,
        "created_at_utc": _utc_now(),
        "frozen": True,
        "representation": representation,
        "example_policy": example_policy,
        "context_bundles": list(context_bundles),
        "proposal_track_modes": list(proposal_track_modes),
        "include_abstention": include_abstention,
        "prompt_template_script": "scripts/prompt_dev_templates.py",
        "notes": notes,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text(_frozen_config_markdown(config), encoding="utf-8")
    return config


def _frozen_config_markdown(config: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Frozen Prompt Development Configuration",
            "",
            f"Prompt version: `{config['manifest_version']}`",
            f"Representation: `{config['representation']}`",
            f"Example policy: `{config['example_policy']}`",
            f"Context bundles: `{', '.join(config['context_bundles'])}`",
            f"Proposal track modes: `{', '.join(config['proposal_track_modes'])}`",
            f"Abstention enabled: `{config['include_abstention']}`",
            "",
            "Notes:",
            config.get("notes") or "",
            "",
        ]
    )


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def _empty_usage(provider: ModelProvider, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cached_tokens": None,
        "estimated_cost_usd": None,
        "input_cost_per_1m_tokens_usd": None,
        "output_cost_per_1m_tokens_usd": None,
        "model": getattr(provider, "model", metadata.get("model") or "unknown-model"),
        "provider": getattr(provider, "provider_name", provider.__class__.__name__),
        "request_metadata": metadata,
    }


def _payload_with_case_id(payload: Any, case_id: str, visible_case_id: str | None = None) -> Any:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    if visible_case_id and normalized.get("case_id") == visible_case_id:
        normalized["case_id"] = case_id
    if not isinstance(normalized.get("case_id"), str) or not normalized.get("case_id", "").strip():
        normalized["case_id"] = case_id
    return normalized


def _prompt_record_metadata(
    *,
    run_id: str,
    prompt_record: dict[str, Any],
    provider: ModelProvider,
    record: dict[str, Any],
    world_state_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    proposal_track = prompt_record.get("historical_track") if prompt_record.get("task") != "track_diagnosis" else None
    task_type = "track_diagnosis" if prompt_record.get("task") == "track_diagnosis" else "proposal"
    metadata = {
        "run_id": run_id,
        "matrix_id": prompt_record["matrix_id"],
        "case_id": prompt_record["case_id"],
        "visible_case_id": prompt_record.get("visible_case_id"),
        "ablation_bundle": prompt_record["matrix_id"],
        "context_bundle": prompt_record["context_bundle"],
        "representation": prompt_record["representation"],
        "example_policy": prompt_record["example_policy"],
        "track_mode": prompt_record.get("track_mode"),
        "include_abstention": prompt_record["include_abstention"],
        "prompt_name": prompt_record["prompt_name"],
        "track": prompt_record.get("historical_track"),
        "historical_track": prompt_record.get("historical_track"),
        "proposal_track_used": proposal_track,
        "routing_source": "prompt_dev",
        "task_type": task_type,
        "prompt_dev_task": prompt_record["task"],
        "provider": getattr(provider, "provider_name", provider.__class__.__name__),
        "model": getattr(provider, "model", "unknown-model"),
        "context_audit": prompt_record.get("context_audit") or {},
    }
    if proposal_track == "T_BOX":
        metadata["t_box_constraint_type_qids"] = _t_box_constraint_type_qids(record, world_state_entry)
    return metadata


def _record_prompt_dev_result(
    *,
    matrix_dir: Path,
    prompt_record: dict[str, Any],
    metadata: dict[str, Any],
    raw_response: Any,
    parsed_payload: Any,
    usage: dict[str, Any],
    elapsed_seconds: float | None,
    error_message: str | None = None,
) -> dict[str, Any]:
    manifest_record = {
        **{key: metadata.get(key) for key in (
            "run_id",
            "matrix_id",
            "case_id",
            "visible_case_id",
            "ablation_bundle",
            "context_bundle",
            "representation",
            "example_policy",
            "track_mode",
            "include_abstention",
            "prompt_name",
            "track",
            "historical_track",
            "proposal_track_used",
            "routing_source",
            "task_type",
            "prompt_dev_task",
        )},
        "provider": usage.get("provider"),
        "model": usage.get("model"),
        "usage": _usage_block(usage, elapsed_seconds),
        "context_audit": metadata.get("context_audit") or {},
        "timestamp_utc": _utc_now(),
    }
    raw_record = {
        **{key: manifest_record.get(key) for key in (
            "run_id",
            "matrix_id",
            "case_id",
            "visible_case_id",
            "ablation_bundle",
            "prompt_name",
            "track",
            "historical_track",
            "proposal_track_used",
            "routing_source",
            "task_type",
            "prompt_dev_task",
        )},
        "raw_response": raw_response,
        "parsed_payload": parsed_payload,
    }
    if error_message:
        raw_record["error"] = error_message
        manifest_record["parse_status"] = "request_error"
        manifest_record["provider_error"] = error_message
        _append_jsonl(matrix_dir / "raw_model_responses.jsonl", raw_record)
        _append_jsonl(matrix_dir / "run_manifest.jsonl", manifest_record)
        return manifest_record

    try:
        normalized_payload = _payload_with_case_id(
            parsed_payload,
            prompt_record["case_id"],
            prompt_record.get("visible_case_id") if isinstance(prompt_record.get("visible_case_id"), str) else None,
        )
        if metadata["task_type"] == "track_diagnosis":
            normalized = normalize_diagnosis(normalized_payload)
            _append_jsonl(matrix_dir / "track_diagnoses.jsonl", normalized.to_dict())
        elif metadata.get("proposal_track_used") == "T_BOX":
            normalized = normalize_t_box_proposal(
                normalized_payload,
                constraint_type_qids=metadata.get("t_box_constraint_type_qids"),
            )
            _append_jsonl(matrix_dir / "t_box_proposals.jsonl", normalized.to_dict())
        else:
            normalized = normalize_a_box_proposal(normalized_payload)
            _append_jsonl(matrix_dir / "a_box_proposals.jsonl", normalized.to_dict())
        manifest_record["parse_status"] = "normalized"
        manifest_record["canonical_hash"] = normalized.canonical_hash
    except Exception as exc:
        manifest_record["parse_status"] = "parse_error"
        manifest_record["parser_error"] = str(exc)

    _append_jsonl(matrix_dir / "raw_model_responses.jsonl", raw_record)
    _append_jsonl(matrix_dir / "run_manifest.jsonl", manifest_record)
    return manifest_record


def _write_missing_diagnosis_manifest_rows(
    *,
    matrix_dir: Path,
    matrix_id: str,
    run_id: str,
    case_ids: Iterable[str],
    provider: ModelProvider,
) -> None:
    for case_id in sorted(set(case_ids)):
        _append_jsonl(
            matrix_dir / "run_manifest.jsonl",
            {
                "run_id": run_id,
                "matrix_id": matrix_id,
                "case_id": case_id,
                "ablation_bundle": matrix_id,
                "task_type": "track_diagnosis",
                "prompt_dev_task": "track_diagnosis",
                "provider": getattr(provider, "provider_name", provider.__class__.__name__),
                "model": getattr(provider, "model", "unknown-model"),
                "usage": _usage_block(_empty_usage(provider, {"case_id": case_id}), None),
                "timestamp_utc": _utc_now(),
                "parse_status": "missing",
                "skip_reason": "not_run_for_repair_proposal_prompt_matrix",
            },
        )


def _prompt_manifest_key(prompt_record: dict[str, Any]) -> tuple[str, str]:
    task_type = "track_diagnosis" if prompt_record.get("task") == "track_diagnosis" else "proposal"
    return prompt_record["case_id"], task_type


def _latest_manifest_rows(manifest_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not manifest_path.exists():
        return rows
    for row in iter_jsonl(manifest_path):
        if not isinstance(row, dict):
            continue
        case_id = row.get("case_id")
        task_type = row.get("task_type")
        if not isinstance(case_id, str) or not isinstance(task_type, str):
            continue
        if row.get("skip_reason") == "not_run_for_repair_proposal_prompt_matrix":
            continue
        rows[(case_id, task_type)] = row
    return rows


def _should_skip_existing_prompt_result(
    existing_row: dict[str, Any] | None,
    *,
    retry_failures: bool,
) -> tuple[bool, str | None]:
    if not existing_row:
        return False, None
    parse_status = existing_row.get("parse_status")
    if parse_status == "normalized":
        return True, "skipped_existing_normalized"
    if parse_status in {"request_error", "parse_error"} and not retry_failures:
        return True, f"skipped_existing_{parse_status}"
    return False, None


def _status_bucket(parse_status: str) -> str:
    if parse_status in {"normalized", "parse_error", "request_error"}:
        return parse_status
    if parse_status.startswith("skipped"):
        return "skipped"
    return parse_status


def _prompt_char_count(prompt_record: dict[str, Any]) -> int:
    return len(str(prompt_record.get("system_prompt") or "")) + len(str(prompt_record.get("user_prompt") or ""))


def _notify_progress(callback: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    if callback is not None:
        callback(event)


def evaluate_prompt_dev_prompts(
    options: PromptDevEvaluateOptions,
    *,
    provider: ModelProvider | None = None,
) -> dict[str, Any]:
    log = logging.getLogger("prompt_dev")
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = output_dir / "rendered_prompts"
    log.info("evaluate: rendering prompts into %s", render_dir)
    render_summary = render_prompt_dev_prompts(
        PromptDevRenderOptions(
            classified_benchmark=options.classified_benchmark,
            world_state=options.world_state,
            dev_manifest=options.dev_manifest,
            output_dir=render_dir,
            seed=options.seed,
            max_cases=options.max_cases,
            representations=options.representations,
            example_policies=options.example_policies,
            context_bundles=options.context_bundles,
            tasks=options.tasks,
            repair_track_modes=options.repair_track_modes,
            include_abstention=options.include_abstention,
            core_manifest=options.core_manifest,
            example_count=options.example_count,
            allow_same_property_examples=options.allow_same_property_examples,
            sample_strategy=options.sample_strategy,
            allow_core_example_risk=options.allow_core_example_risk,
        )
    )
    log.info("evaluate: render complete rendered_prompts=%s", render_summary["counts"]["rendered_prompts"])
    log.info("evaluate: loading rendered prompt records from %s", render_dir / "prompt_dev_rendered_prompts.jsonl")
    prompt_records = [
        record
        for record in iter_jsonl(render_dir / "prompt_dev_rendered_prompts.jsonl")
        if isinstance(record, dict)
    ]
    log.info("evaluate: loaded prompt records=%s", len(prompt_records))
    log.info("evaluate: loading eval records")
    eval_records = _load_manifest_records(
        options.classified_benchmark,
        options.dev_manifest,
        max_cases=options.max_cases,
        sample_strategy=options.sample_strategy,
        seed=options.seed,
    )
    log.info("evaluate: loaded eval records=%s", len(eval_records))
    records_by_id = {record["id"]: record for record in eval_records if isinstance(record.get("id"), str)}
    log.info("evaluate: creating model provider endpoint=%s model=%s", options.model_endpoint or "env", options.model_name or "env")
    provider = provider or create_model_provider(options.model_name, model_endpoint=options.model_endpoint)
    log.info(
        "evaluate: provider ready provider=%s model=%s",
        getattr(provider, "provider_name", provider.__class__.__name__),
        getattr(provider, "model", options.model_name or "unknown-model"),
    )
    run_id = f"prompt_dev_eval_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    matrices: dict[str, dict[str, Any]] = {}
    prompt_counts = Counter()
    _notify_progress(
        options.progress_callback,
        {
            "event": "start",
            "total": len(prompt_records),
            "unit": "prompt",
            "provider": getattr(provider, "provider_name", provider.__class__.__name__),
            "model": getattr(provider, "model", options.model_name or "unknown-model"),
        },
    )
    log.info("evaluate: opening world-state store path=%s", options.world_state)
    with WorldStateStore(options.world_state, log) as world_store:
        log.info("evaluate: world-state store ready entries=%s; starting model requests", len(world_store))
        for prompt_record in prompt_records:
            matrix_id = prompt_record["matrix_id"]
            matrix_dir = output_dir / "matrices" / matrix_id
            matrix_dir.mkdir(parents=True, exist_ok=True)
            existing_rows = _latest_manifest_rows(matrix_dir / "run_manifest.jsonl") if options.resume_existing else {}
            record = records_by_id[prompt_record["case_id"]]
            world_state_entry = world_store.get(record["id"])
            metadata = _prompt_record_metadata(
                run_id=run_id,
                prompt_record=prompt_record,
                provider=provider,
                record=record,
                world_state_entry=world_state_entry,
            )
            existing_row = existing_rows.get(_prompt_manifest_key(prompt_record))
            should_skip, skip_status = _should_skip_existing_prompt_result(
                existing_row,
                retry_failures=options.retry_failures,
            )
            started = time.perf_counter()
            if should_skip:
                manifest_record = dict(existing_row or {})
                manifest_record["parse_status"] = skip_status
            else:
                prompt_char_count = _prompt_char_count(prompt_record)
                if options.max_prompt_chars is not None and prompt_char_count > options.max_prompt_chars:
                    manifest_record = _record_prompt_dev_result(
                        matrix_dir=matrix_dir,
                        prompt_record=prompt_record,
                        metadata=metadata,
                        raw_response=None,
                        parsed_payload=None,
                        usage=_empty_usage(provider, metadata),
                        elapsed_seconds=None,
                        error_message=(
                            f"prompt length {prompt_char_count} exceeds --max-prompt-chars "
                            f"{options.max_prompt_chars}"
                        ),
                    )
                else:
                    try:
                        raw_response, parsed_payload, usage = provider.generate(
                            prompt_record["user_prompt"],
                            prompt_record["system_prompt"],
                            prompt_record["response_format"],
                            metadata,
                        )
                        elapsed_seconds = time.perf_counter() - started
                        error_message = None
                    except Exception as exc:
                        raw_response = None
                        parsed_payload = None
                        usage = _empty_usage(provider, metadata)
                        elapsed_seconds = time.perf_counter() - started
                        error_message = str(exc)
                    manifest_record = _record_prompt_dev_result(
                        matrix_dir=matrix_dir,
                        prompt_record=prompt_record,
                        metadata=metadata,
                        raw_response=raw_response,
                        parsed_payload=parsed_payload,
                        usage=usage,
                        elapsed_seconds=elapsed_seconds,
                        error_message=error_message,
                    )
            prompt_counts[manifest_record["parse_status"]] += 1
            matrices.setdefault(
                matrix_id,
                {
                    "matrix_id": matrix_id,
                    "output_dir": str(matrix_dir),
                    "representation": prompt_record["representation"],
                    "example_policy": prompt_record["example_policy"],
                    "context_bundle": prompt_record["context_bundle"],
                    "task": "track_diagnosis" if prompt_record["task"] == "track_diagnosis" else "repair_proposal",
                    "track_mode": prompt_record.get("track_mode"),
                    "include_abstention": prompt_record["include_abstention"],
                    "case_ids": [],
                    "counts": Counter(),
                    "status_counts": Counter(),
                    "by_historical_track": Counter(),
                    "by_task": Counter(),
                    "by_context": Counter(),
                },
            )
            matrices[matrix_id]["case_ids"].append(prompt_record["case_id"])
            matrices[matrix_id]["counts"][manifest_record["parse_status"]] += 1
            matrices[matrix_id]["status_counts"][_status_bucket(manifest_record["parse_status"])] += 1
            matrices[matrix_id]["by_historical_track"][prompt_record.get("historical_track") or "unknown"] += 1
            matrices[matrix_id]["by_task"][prompt_record["task"]] += 1
            matrices[matrix_id]["by_context"][prompt_record["context_bundle"]] += 1
            _notify_progress(
                options.progress_callback,
                {
                    "event": "advance",
                    "matrix_id": matrix_id,
                    "case_id": prompt_record["case_id"],
                    "task": prompt_record["task"],
                    "parse_status": manifest_record["parse_status"],
                },
            )

    results: list[dict[str, Any]] = []
    log.info("evaluate: model request loop complete status_counts=%s", dict(prompt_counts))
    for matrix_id, matrix_info in sorted(matrices.items()):
        matrix_dir = Path(matrix_info["output_dir"])
        unique_case_ids = sorted(set(matrix_info["case_ids"]))
        log.info("evaluate: scoring matrix=%s cases=%s", matrix_id, len(unique_case_ids))
        if matrix_info["task"] == "repair_proposal":
            _write_missing_diagnosis_manifest_rows(
                matrix_dir=matrix_dir,
                matrix_id=matrix_id,
                run_id=run_id,
                case_ids=unique_case_ids,
                provider=provider,
            )
        evaluation_error = None
        try:
            _, eval_summary = evaluate_benchmark(
                classified_path=options.classified_benchmark,
                world_state_path=options.world_state,
                a_box_proposals_path=matrix_dir / "a_box_proposals.jsonl",
                t_box_proposals_path=matrix_dir / "t_box_proposals.jsonl",
                track_diagnoses_path=matrix_dir / "track_diagnoses.jsonl",
                run_manifest_path=matrix_dir / "run_manifest.jsonl",
                ablation_bundle=matrix_id,
                case_ids=unique_case_ids,
                out_traces_path=matrix_dir / "evaluation_traces.jsonl",
                out_summary_path=matrix_dir / "evaluation_summary.json",
                collect_traces=False,
                classified_records=eval_records,
                classified_input_path=options.classified_benchmark,
            )
        except Exception as exc:
            evaluation_error = str(exc)
            eval_summary = {
                "manifest_type": "prompt_dev_matrix_evaluation_summary",
                "manifest_version": PROMPT_DEV_VERSION,
                "created_at_utc": _utc_now(),
                "ablation_bundle": matrix_id,
                "overall_metrics": {},
                "parse_errors": {},
                "request_errors": {},
                "evaluation_error": evaluation_error,
            }
            write_json(matrix_dir / "evaluation_summary.json", eval_summary)
            (matrix_dir / "evaluation_traces.jsonl").touch()
        status_counts = (
            matrix_info.get("status_counts") if isinstance(matrix_info.get("status_counts"), Counter) else Counter()
        )
        detailed_counts = {
            "normalized": status_counts.get("normalized", 0),
            "parse_error": status_counts.get("parse_error", 0),
            "request_error": status_counts.get("request_error", 0),
            "skipped": status_counts.get("skipped", 0),
            "by_parse_status": dict(matrix_info["counts"]),
            "by_historical_track": dict(matrix_info.get("by_historical_track", {})),
            "by_task": dict(matrix_info.get("by_task", {})),
            "by_context": dict(matrix_info.get("by_context", {})),
        }
        result = {
            **{
                key: value
                for key, value in matrix_info.items()
                if key not in {"counts", "status_counts", "by_historical_track", "by_task", "by_context"}
            },
            "case_ids": unique_case_ids,
            "counts": detailed_counts,
            "evaluation_summary": str(matrix_dir / "evaluation_summary.json"),
            "overall_metrics": eval_summary.get("overall_metrics", {}),
            "parse_errors": eval_summary.get("parse_errors", {}),
            "request_errors": eval_summary.get("request_errors", {}),
        }
        if evaluation_error is not None:
            result["evaluation_error"] = evaluation_error
        results.append(result)
        log.info("evaluate: scored matrix=%s error=%s", matrix_id, evaluation_error or "none")

    summary = {
        "manifest_type": "prompt_dev_evaluation_summary",
        "manifest_version": PROMPT_DEV_VERSION,
        "created_at_utc": _utc_now(),
        "run_id": run_id,
        "provider": getattr(provider, "provider_name", provider.__class__.__name__),
        "model": getattr(provider, "model", options.model_name or "unknown-model"),
        "inputs": {
            "classified_benchmark": str(options.classified_benchmark),
            "world_state": str(options.world_state),
            "dev_manifest": str(options.dev_manifest),
            "core_manifest": str(options.core_manifest) if options.core_manifest else None,
            "render_summary": str(render_dir / "prompt_dev_render_summary.json"),
            "sample_strategy": options.sample_strategy,
            "max_cases": options.max_cases,
        },
        "outputs": {
            "output_dir": str(output_dir),
            "comparison_markdown": str(output_dir / "prompt_dev_evaluation_comparison.md"),
        },
        "counts": {
            "rendered_prompts": render_summary["counts"]["rendered_prompts"],
            "evaluated_prompts": len(prompt_records),
            "matrix_rows": len(results),
            "by_parse_status": dict(prompt_counts),
        },
        "results": results,
    }
    summary_path = output_dir / "prompt_dev_evaluation_summary.json"
    write_json(summary_path, summary)
    (output_dir / "prompt_dev_evaluation_comparison.md").write_text(
        _evaluation_comparison_markdown(summary),
        encoding="utf-8",
    )
    log.info("evaluate: wrote summary=%s comparison=%s", summary_path, output_dir / "prompt_dev_evaluation_comparison.md")
    return summary


def _metric_text(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return ""


def _evaluation_comparison_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Prompt Development Evaluation",
        "",
        f"Run id: `{summary['run_id']}`",
        f"Provider: `{summary['provider']}`",
        f"Model: `{summary['model']}`",
        f"Evaluated prompts: `{summary['counts']['evaluated_prompts']}`",
        "",
        (
            "| Matrix id | Task | Representation | Examples | Context | Track mode | "
            "Parse errors | Request errors | Functional | Track acc | Audit |"
        ),
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in summary["results"]:
        metrics = result.get("overall_metrics") if isinstance(result.get("overall_metrics"), dict) else {}
        task = result.get("task")
        parse_errors = (result.get("parse_errors") or {}).get("proposal_parse_error_count", 0)
        request_errors = (result.get("request_errors") or {}).get("proposal_request_error_count", 0)
        request_errors += (result.get("request_errors") or {}).get("track_diagnosis_request_error_count", 0)
        functional_text = "n/a" if task == "track_diagnosis" else _metric_text(metrics, "functional_success_rate")
        track_accuracy_text = (
            _metric_text(metrics, "track_diagnosis_accuracy") if task == "track_diagnosis" else "n/a"
        )
        audit_text = "n/a" if task == "track_diagnosis" else _metric_text(metrics, "auditability_complete_rate")
        lines.append(
            " | ".join(
                [
                    f"| `{result['matrix_id']}`",
                    f"`{task}`",
                    f"`{result['representation']}`",
                    f"`{result['example_policy']}`",
                    f"`{result['context_bundle']}`",
                    f"`{result.get('track_mode') or ''}`",
                    str(parse_errors),
                    str(request_errors),
                    functional_text,
                    track_accuracy_text,
                    audit_text + " |",
                ]
            )
        )
    lines.append("")
    return "\n".join(lines)
