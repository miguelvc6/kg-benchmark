from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from jsonschema import Draft202012Validator

from classifier import WorldStateStore
from guardian.evaluator import evaluate_benchmark, write_json
from guardian.model_provider import ModelProvider, create_model_provider
from guardian.patch_parser import normalize_proposal as normalize_a_box_proposal
from guardian.reasoning import (
    ABLATION_BUNDLES,
    DIAGNOSIS_ABLATION_BUNDLES,
    _bundle_payload_and_audit,
    _diagnosis_bundle_payload_and_audit,
    _t_box_constraint_type_qids,
)
from guardian.tbox_taxonomy_patch_evaluator import evaluate_tbox_taxonomy_patch_predictions, load_jsonl
from guardian.tbox_taxonomy_patch_parser import normalize_tbox_taxonomy_patch
from guardian.tbox_parser import normalize_proposal as normalize_t_box_proposal
from guardian.track_parser import normalize_diagnosis
from lib.benchmark_selection import derive_case_metadata, group_key_for_record, load_selection_manifest
from lib.tbox_taxonomy_patch_gold import gold_patch_for_record
from lib.repair_state import derive_value_change_summary, normalize_value_list
from lib.utils import iter_jsonl
from scripts.prompt_dev_templates import (
    PROMPT_DEV_VERSION,
    PROMPT_DEV_TBOX_TAXONOMY_PATCH_VERSION,
    REPRESENTATIONS,
    render_prompt_dev_prompt,
)

EXAMPLE_POLICIES = (
    "zero_shot",
    "static_diverse_kshot",
    "random_same_task_2shot",
    "same_track_2shot",
    "matched_2shot",
)
REPAIR_TRACK_MODES = ("oracle", "diagnosis_routed")
DEFAULT_CONTEXT_BUNDLES = ("logic_only", "local_graph")
DEFAULT_DIAGNOSIS_CONTEXT_BUNDLES: tuple[str, ...] = ()
DEFAULT_RENDER_TASKS = ("track_diagnosis", "repair_proposal")
SAMPLE_STRATEGIES = ("manifest_order", "stratified", "diverse_stratified")
TRACK_FILTERS = ("A_BOX", "T_BOX")
T_BOX_ACTIONS = {
    "RELAXATION_RANGE_WIDENED",
    "RESTRICTION_RANGE_NARROWED",
    "RELAXATION_SET_EXPANSION",
    "RESTRICTION_SET_CONTRACTION",
    "SCHEMA_UPDATE",
    "COINCIDENTAL_SCHEMA_CHANGE",
}
DIAGNOSIS_ACCEPTANCE_GATES = {
    "request_error_rate_max": 0.01,
    "parse_error_rate_max": 0.04,
    "balanced_accuracy_min": 0.70,
    "a_box_recall_min": 0.65,
    "t_box_recall_min": 0.65,
    "ambiguous_rate_max": 0.15,
}


@dataclass(frozen=True)
class PromptDevMatrixOptions:
    representations: tuple[str, ...] = REPRESENTATIONS
    example_policies: tuple[str, ...] = EXAMPLE_POLICIES
    context_bundles: tuple[str, ...] = DEFAULT_CONTEXT_BUNDLES
    diagnosis_context_bundles: tuple[str, ...] = DEFAULT_DIAGNOSIS_CONTEXT_BUNDLES
    tasks: tuple[str, ...] = DEFAULT_RENDER_TASKS
    repair_track_modes: tuple[str, ...] = REPAIR_TRACK_MODES
    include_abstention: bool = False


@dataclass(frozen=True)
class PromptDevRenderOptions:
    classified_benchmark: Path
    world_state: Path
    output_dir: Path
    eval_manifest: Path | None = None
    dev_manifest: Path | None = None
    example_manifest: Path | None = None
    seed: int = 13
    max_cases: int | None = 24
    representations: tuple[str, ...] = ("hybrid_json_nl",)
    example_policies: tuple[str, ...] = ("zero_shot",)
    context_bundles: tuple[str, ...] = DEFAULT_CONTEXT_BUNDLES
    diagnosis_context_bundles: tuple[str, ...] = DEFAULT_DIAGNOSIS_CONTEXT_BUNDLES
    tasks: tuple[str, ...] = DEFAULT_RENDER_TASKS
    repair_track_modes: tuple[str, ...] = ("oracle",)
    include_abstention: bool = False
    core_manifest: Path | None = None
    support_set_manifest: Path | None = None
    example_count: int = 2
    allow_same_property_examples: bool = False
    sample_strategy: str = "stratified"
    allow_core_example_risk: bool = False
    track_filter: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PromptDevEvaluateOptions:
    classified_benchmark: Path
    world_state: Path
    output_dir: Path
    eval_manifest: Path | None = None
    dev_manifest: Path | None = None
    example_manifest: Path | None = None
    model_endpoint: str | None = None
    model_name: str | None = None
    seed: int = 13
    max_cases: int | None = 24
    representations: tuple[str, ...] = ("hybrid_json_nl",)
    example_policies: tuple[str, ...] = ("zero_shot",)
    context_bundles: tuple[str, ...] = DEFAULT_CONTEXT_BUNDLES
    diagnosis_context_bundles: tuple[str, ...] = DEFAULT_DIAGNOSIS_CONTEXT_BUNDLES
    tasks: tuple[str, ...] = DEFAULT_RENDER_TASKS
    repair_track_modes: tuple[str, ...] = ("oracle",)
    include_abstention: bool = False
    core_manifest: Path | None = None
    support_set_manifest: Path | None = None
    example_count: int = 2
    allow_same_property_examples: bool = False
    resume_existing: bool = True
    retry_failures: bool = False
    max_prompt_chars: int | None = None
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    sample_strategy: str = "stratified"
    allow_core_example_risk: bool = False
    track_filter: tuple[str, ...] | None = None
    zero_shot_baseline_summary: Path | None = None


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
            context_axis = tuple(dict.fromkeys((*options.context_bundles, *options.diagnosis_context_bundles)))
            for context_bundle in context_axis:
                if context_bundle not in ABLATION_BUNDLES and context_bundle not in DIAGNOSIS_ABLATION_BUNDLES:
                    raise ValueError(f"Unsupported context bundle: {context_bundle}")
                for task in options.tasks:
                    if task == "track_diagnosis":
                        if options.diagnosis_context_bundles:
                            if context_bundle not in options.diagnosis_context_bundles:
                                continue
                        elif context_bundle not in options.context_bundles:
                            continue
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
                        if context_bundle not in options.context_bundles:
                            continue
                        if context_bundle not in ABLATION_BUNDLES:
                            raise ValueError(f"Repair proposal context bundle must be one of {ABLATION_BUNDLES}: {context_bundle}")
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


def _prompt_payload_builder_for_task(task: str, context_bundle: str) -> Callable[..., tuple[dict[str, Any], dict[str, Any]]]:
    if task == "track_diagnosis" and context_bundle in DIAGNOSIS_ABLATION_BUNDLES:
        return _diagnosis_bundle_payload_and_audit
    return _bundle_payload_and_audit


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
    track_filter: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    if sample_strategy not in SAMPLE_STRATEGIES:
        raise ValueError(f"Unsupported prompt-dev sample strategy: {sample_strategy}")
    normalized_track_filter = _normalize_track_filter(track_filter)
    manifest = load_selection_manifest(manifest_path)
    ids = [case_id for case_id in manifest.get("selected_case_ids", []) if isinstance(case_id, str)]
    id_set = set(ids)
    by_id = {
        record["id"]: record
        for record in iter_jsonl(classified_path)
        if isinstance(record, dict) and isinstance(record.get("id"), str) and record["id"] in id_set
    }
    ids = [case_id for case_id in ids if case_id in by_id]
    if normalized_track_filter is not None:
        ids = [case_id for case_id in ids if by_id[case_id].get("track") in normalized_track_filter]
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


def _normalize_track_filter(track_filter: tuple[str, ...] | None) -> set[str] | None:
    if track_filter is None:
        return None
    normalized = {value.strip().upper() for value in track_filter if isinstance(value, str) and value.strip()}
    if not normalized:
        return None
    invalid = sorted(normalized - set(TRACK_FILTERS))
    if invalid:
        raise ValueError(f"Unsupported prompt-dev track filter: {', '.join(invalid)}")
    return normalized


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


FORBIDDEN_EXAMPLE_KEYS = {
    "repair_target",
    "classification",
    "persistence_check",
    "truth_source",
    "truth_tokens",
    "selection_stratum",
    "group_key",
    "selected_case_ids",
    "case_annotations",
    "historical_track",
}
RAW_CASE_ID_RE = re.compile(r"\b(?:repair|reform)_(?!op\b)[A-Za-z0-9][A-Za-z0-9_.:-]*")
CORE_DEV_LABEL_RE = re.compile(r"\b(?:DEV|CORE)_[A-Za-z0-9][A-Za-z0-9_.:-]*")
FORBIDDEN_EXAMPLE_TEXT_TERMS = tuple(sorted(FORBIDDEN_EXAMPLE_KEYS))
MODEL_VISIBLE_FORBIDDEN_TERMS = (
    "repair_target",
    "classification",
    "persistence_check",
    "truth_source",
    "truth_tokens",
    "selection_stratum",
    "group_key",
    "selected_case_ids",
    "case_annotations",
    "sitelinks_count",
    "popularity",
    "historical_track",
    "TypeA",
    "TypeB",
    "TypeC",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _support_set_schema_path() -> Path:
    return _repo_root() / "schemas" / "few_shot_support_set.schema.json"


def _load_support_set_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    with _support_set_schema_path().open(encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator(schema).validate(manifest)
    return manifest


def _support_task_key(task: str) -> str | None:
    if task == "a_box_repair":
        return "a_box_repair"
    if task == "t_box_repair":
        return "t_box_repair"
    if task == "track_diagnosis":
        return "track_diagnosis"
    return None


def _forbidden_key_paths(value: Any, *, path: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_EXAMPLE_KEYS:
                paths.append(child_path)
            paths.extend(_forbidden_key_paths(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_key_paths(child, path=f"{path}[{index}]"))
    return paths


def _forbidden_string_hits(value: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(value, str):
        hits.extend(RAW_CASE_ID_RE.findall(value))
        hits.extend(CORE_DEV_LABEL_RE.findall(value))
        hits.extend(term for term in FORBIDDEN_EXAMPLE_TEXT_TERMS if term in value)
    elif isinstance(value, dict):
        for child in value.values():
            hits.extend(_forbidden_string_hits(child))
    elif isinstance(value, list):
        for child in value:
            hits.extend(_forbidden_string_hits(child))
    return hits


def _sanitize_example_text(value: Any) -> Any:
    if isinstance(value, str):
        return (
            value.replace("classification confidence", "support confidence")
            .replace("classification", "support metadata")
            .replace("repair_target", "visible target evidence")
            .replace("truth_source", "visible source")
            .replace("truth_tokens", "visible tokens")
        )
    if isinstance(value, dict):
        return {key: _sanitize_example_text(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_example_text(child) for child in value]
    return value


def _scan_examples_for_leakage(examples: list[dict[str, Any]]) -> dict[str, Any]:
    payloads = [
        {
            "input_payload": example.get("input_payload") if isinstance(example.get("input_payload"), dict) else {},
            "output_payload": example.get("output_payload") if isinstance(example.get("output_payload"), dict) else {},
        }
        for example in examples
    ]
    key_hits = sorted({path for payload in payloads for path in _forbidden_key_paths(payload)})
    text_hits = sorted({hit for payload in payloads for hit in _forbidden_string_hits(payload)})
    return {
        "passed": not key_hits and not text_hits,
        "forbidden_key_paths": key_hits,
        "forbidden_text_patterns": text_hits,
    }


def _raise_if_examples_leak(examples: list[dict[str, Any]]) -> dict[str, Any]:
    scan = _scan_examples_for_leakage(examples)
    if not scan["passed"]:
        raise ValueError(
            "Few-shot support examples failed leakage scan: "
            f"keys={scan['forbidden_key_paths']} text={scan['forbidden_text_patterns']}"
        )
    return scan


def _model_visible_text_scan(prompt_records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    hard_matches: list[dict[str, Any]] = []
    benign_matches: list[dict[str, Any]] = []
    for record in prompt_records:
        text = f"{record.get('system_prompt') or ''}\n{record.get('user_prompt') or ''}"
        row_matches: list[dict[str, Any]] = []
        for term in MODEL_VISIBLE_FORBIDDEN_TERMS:
            if term not in text:
                continue
            match = {"kind": "term", "value": term}
            if term == "classification" and f'"{term}"' not in text and f"{term}:" not in text:
                benign_matches.append({"matrix_id": record["matrix_id"], "case_id": record["case_id"], **match})
            else:
                row_matches.append(match)
        for raw_id in sorted(set(RAW_CASE_ID_RE.findall(text))):
            row_matches.append({"kind": "raw_case_id", "value": raw_id})
        for label in sorted(set(CORE_DEV_LABEL_RE.findall(text))):
            row_matches.append({"kind": "core_dev_label", "value": label})
        if row_matches:
            for match in row_matches:
                hard_matches.append({"matrix_id": record["matrix_id"], "case_id": record["case_id"], **match})
        rows.append(
            {
                "matrix_id": record["matrix_id"],
                "case_id": record["case_id"],
                "task": record["task"],
                "example_policy": record["example_policy"],
                "hard_match_count": len(row_matches),
            }
        )
    return {
        "passed": not hard_matches,
        "prompt_count": len(prompt_records),
        "hard_matches": hard_matches,
        "benign_text_matches": benign_matches,
        "rows": rows,
    }


def _tbox_revision_key_for_overlap(record: dict[str, Any] | None) -> str | None:
    if not isinstance(record, dict):
        return None
    _, tbox_key, _ = group_key_for_record(record)
    return tbox_key


def _example_overlap_report(
    *,
    prompt_records: list[dict[str, Any]],
    eval_records_by_id: dict[str, dict[str, Any]],
    example_records_by_id: dict[str, dict[str, Any]],
    blocked_core: dict[str, set[str]],
    allow_same_property_examples: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    hard_failures: list[dict[str, Any]] = []
    property_overlaps: list[dict[str, Any]] = []
    core_case_overlap = 0
    core_tbox_revision_overlap = 0
    for prompt_record in prompt_records:
        eval_case_id = prompt_record.get("case_id")
        eval_record = eval_records_by_id.get(eval_case_id) if isinstance(eval_case_id, str) else None
        eval_qid = eval_record.get("qid") if isinstance(eval_record, dict) else None
        eval_property = eval_record.get("property") if isinstance(eval_record, dict) else None
        eval_tbox_key = _tbox_revision_key_for_overlap(eval_record)
        is_static = any(
            isinstance(example, dict) and example.get("support_source") == "support_set_manifest"
            for example in prompt_record.get("examples", [])
        )
        for example in prompt_record.get("examples", []):
            if not isinstance(example, dict):
                continue
            example_case_id = example.get("case_id")
            example_record = example_records_by_id.get(example_case_id) if isinstance(example_case_id, str) else None
            example_qid = example_record.get("qid") if isinstance(example_record, dict) else None
            example_property = example_record.get("property") if isinstance(example_record, dict) else None
            example_tbox_key = _tbox_revision_key_for_overlap(example_record)
            checks = {
                "case_id_overlap": example_case_id == eval_case_id,
                "qid_overlap": bool(example_qid and eval_qid and example_qid == eval_qid),
                "property_overlap": bool(example_property and eval_property and example_property == eval_property),
                "tbox_revision_overlap": bool(example_tbox_key and eval_tbox_key and example_tbox_key == eval_tbox_key),
                "core_case_overlap": bool(example_case_id and example_case_id in blocked_core["case_ids"]),
                "core_tbox_revision_overlap": bool(
                    example_tbox_key and example_tbox_key in blocked_core["tbox_revision_keys"]
                ),
            }
            if checks["core_case_overlap"]:
                core_case_overlap += 1
            if checks["core_tbox_revision_overlap"]:
                core_tbox_revision_overlap += 1
            failure_keys = [
                key
                for key, failed in checks.items()
                if failed
                and key != "property_overlap"
                and key not in {"core_case_overlap", "core_tbox_revision_overlap"}
            ]
            if checks["core_case_overlap"] or checks["core_tbox_revision_overlap"]:
                failure_keys.extend(
                    key for key in ("core_case_overlap", "core_tbox_revision_overlap") if checks[key]
                )
            if checks["property_overlap"]:
                property_overlap = {
                    "matrix_id": prompt_record["matrix_id"],
                    "eval_case_id": eval_case_id,
                    "example_case_id": example_case_id,
                    "property": example_property,
                    "static_support": is_static,
                    "allowed": is_static or allow_same_property_examples,
                }
                property_overlaps.append(property_overlap)
                if not property_overlap["allowed"]:
                    failure_keys.append("property_overlap")
            row = {
                "matrix_id": prompt_record["matrix_id"],
                "eval_case_id": eval_case_id,
                "example_case_id": example_case_id,
                "visible_case_id": example.get("visible_case_id"),
                "task": prompt_record.get("task"),
                "example_task_schema": example.get("task_schema"),
                "checks": checks,
                "failure_keys": failure_keys,
            }
            if failure_keys:
                hard_failures.append(row)
            rows.append(row)
    return {
        "passed": not hard_failures and core_case_overlap == 0 and core_tbox_revision_overlap == 0,
        "example_count": len(rows),
        "core_case_overlap": core_case_overlap,
        "core_tbox_revision_overlap": core_tbox_revision_overlap,
        "property_overlaps": property_overlaps,
        "hard_failures": hard_failures,
        "rows": rows,
    }


def _overlap_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Few-Shot Overlap Report",
        "",
        f"Passed: `{report['passed']}`",
        f"Examples checked: `{report['example_count']}`",
        f"Core case overlap: `{report['core_case_overlap']}`",
        f"Core T-box revision overlap: `{report['core_tbox_revision_overlap']}`",
        "",
        "## Property Overlaps",
        "",
    ]
    if report["property_overlaps"]:
        lines.extend(
            [
                "| Matrix | Eval case | Example case | Property | Static support | Allowed |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in report["property_overlaps"]:
            lines.append(
                f"| `{row['matrix_id']}` | `{row['eval_case_id']}` | `{row['example_case_id']}` | "
                f"`{row['property']}` | `{row['static_support']}` | `{row['allowed']}` |"
            )
    else:
        lines.append("No property overlaps detected.")
    if report["hard_failures"]:
        lines.extend(["", "## Hard Failures", ""])
        for row in report["hard_failures"]:
            lines.append(
                f"- `{row['matrix_id']}` eval `{row['eval_case_id']}` example `{row['example_case_id']}`: "
                f"{', '.join(row['failure_keys'])}"
            )
    lines.append("")
    return "\n".join(lines)


def _validate_example_output_payload(
    *,
    task: str,
    payload: dict[str, Any],
    record: dict[str, Any],
    world_state_entry: dict[str, Any] | None,
    task_schema: str | None = None,
) -> None:
    if task == "a_box_repair":
        normalize_a_box_proposal(payload)
        return
    if task == "t_box_repair":
        if task_schema == "tbox_taxonomy_patch_v1" or PROMPT_DEV_VERSION == PROMPT_DEV_TBOX_TAXONOMY_PATCH_VERSION:
            normalize_tbox_taxonomy_patch(
                payload,
                constraint_type_qids=_t_box_constraint_type_qids(record, world_state_entry),
            )
            return
        normalize_t_box_proposal(payload, constraint_type_qids=_t_box_constraint_type_qids(record, world_state_entry))
        return
    if task == "track_diagnosis":
        normalize_diagnosis(payload)
        return
    raise ValueError(f"Unsupported support example task: {task}")


def _prepare_example_payloads(
    *,
    example: dict[str, Any],
    candidate: dict[str, Any],
    task: str,
    context_bundle: str,
    world_state_entry: dict[str, Any] | None,
    visible_case_id: str,
) -> dict[str, Any]:
    payload_builder = _prompt_payload_builder_for_task(task, context_bundle)
    input_payload, _ = payload_builder(candidate, world_state_entry, context_bundle)
    task_schema = example.get("task_schema") if isinstance(example.get("task_schema"), str) else None
    if task == "t_box_repair" and task_schema == "tbox_taxonomy_patch_v1":
        output_payload = gold_patch_for_record(candidate)
    else:
        output_payload = _gold_output_for_task(candidate, task)
    if not isinstance(output_payload, dict):
        raise ValueError(f"Support example {candidate.get('id')} has no gold output for task {task}.")
    visible_input = _model_visible_payload(
        input_payload,
        raw_case_id=candidate["id"],
        visible_case_id=visible_case_id,
    )
    visible_output = _model_visible_output(
        output_payload,
        raw_case_id=candidate["id"],
        visible_case_id=visible_case_id,
    )
    if visible_output is None:
        raise ValueError(f"Support example {candidate.get('id')} produced an empty visible output.")
    visible_output = _sanitize_example_text(visible_output)
    _validate_example_output_payload(
        task=task,
        payload=visible_output,
        record=candidate,
        world_state_entry=world_state_entry,
        task_schema=task_schema,
    )
    return {
        **example,
        "input_payload": visible_input,
        "output_payload": visible_output,
        "visible_case_id": visible_case_id,
    }


def _support_set_examples(
    *,
    support_manifest: dict[str, Any],
    eval_record: dict[str, Any],
    task: str,
    records_by_id: dict[str, dict[str, Any]],
    world_store: WorldStateStore,
    context_bundle: str,
    example_count: int,
) -> list[dict[str, Any]]:
    task_key = _support_task_key(task)
    if task_key is None:
        return []
    entries = support_manifest.get("support_sets", {}).get(task_key, [])
    if not isinstance(entries, list):
        raise ValueError(f"Support set for {task_key} must be a list.")
    eval_case_id = eval_record.get("id")
    eval_qid = eval_record.get("qid")
    _, eval_tbox_key, _ = group_key_for_record(eval_record)
    filtered_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_case_id = entry.get("raw_case_id")
        if not isinstance(raw_case_id, str):
            raise ValueError(f"Support set entry for {task_key} must contain raw_case_id.")
        candidate = records_by_id.get(raw_case_id)
        if candidate is None:
            raise ValueError(f"Support example raw_case_id not found in example manifest: {raw_case_id}")
        _, candidate_tbox_key, _ = group_key_for_record(candidate)
        if raw_case_id == eval_case_id:
            continue
        if candidate.get("qid") == eval_qid and eval_qid:
            continue
        if candidate_tbox_key == eval_tbox_key and eval_tbox_key:
            continue
        filtered_entries.append((entry, candidate))
    if not filtered_entries and example_count > 0:
        raise ValueError(f"Support set for {task_key} has no non-overlapping examples for {eval_case_id}.")
    examples: list[dict[str, Any]] = []
    for entry, candidate in filtered_entries[:example_count]:
        raw_case_id = entry.get("raw_case_id")
        visible_case_id = entry.get("visible_example_id")
        if not isinstance(raw_case_id, str) or not isinstance(visible_case_id, str):
            raise ValueError(f"Support set entry for {task_key} must contain raw_case_id and visible_example_id.")
        example = {
            "case_id": raw_case_id,
            "visible_case_id": visible_case_id,
            "role": entry.get("role"),
            "task_schema": entry.get("task_schema"),
            "support_source": "support_set_manifest",
        }
        examples.append(
            _prepare_example_payloads(
                example=example,
                candidate=candidate,
                task=task,
                context_bundle=context_bundle,
                world_state_entry=world_store.get(raw_case_id),
                visible_case_id=visible_case_id,
            )
        )
    _raise_if_examples_leak(examples)
    return examples


def _examples_for_prompt(
    *,
    eval_record: dict[str, Any],
    task: str,
    context_bundle: str,
    policy: str,
    seed: int,
    example_count: int,
    blocked_core: dict[str, set[str]],
    allow_same_property: bool,
    candidate_records: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    visible_case_ids: dict[str, str],
    world_store: WorldStateStore,
    support_manifest: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if policy == "zero_shot" or example_count <= 0:
        return []
    if support_manifest is not None:
        return _support_set_examples(
            support_manifest=support_manifest,
            eval_record=eval_record,
            task=task,
            records_by_id=records_by_id,
            world_store=world_store,
            context_bundle=context_bundle,
            example_count=example_count,
        )
    if policy == "static_diverse_kshot":
        raise ValueError("example_policy=static_diverse_kshot requires --support-set-manifest.")
    selected = select_examples(
        eval_record=eval_record,
        candidate_records=candidate_records,
        policy=policy,
        task=task,
        seed=seed,
        example_count=example_count,
        blocked_core=blocked_core,
        allow_same_property=allow_same_property,
    )
    examples: list[dict[str, Any]] = []
    for example in selected:
        raw_case_id = example["case_id"]
        candidate = records_by_id.get(raw_case_id)
        if candidate is None:
            continue
        visible_case_id = visible_case_ids.get(raw_case_id, f"case_{_stable_hash(raw_case_id)[:12]}")
        examples.append(
            _prepare_example_payloads(
                example=example,
                candidate=candidate,
                task=task,
                context_bundle=context_bundle,
                world_state_entry=world_store.get(raw_case_id),
                visible_case_id=visible_case_id,
            )
        )
    _raise_if_examples_leak(examples)
    return examples


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
        if PROMPT_DEV_VERSION == PROMPT_DEV_TBOX_TAXONOMY_PATCH_VERSION:
            return gold_patch_for_record(record)
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
    if matrix_task == "repair_proposal" and track_mode == "diagnosis_routed":
        return "track_diagnosis"
    proposal_track = record.get("track") if track_mode in {None, "oracle", "diagnosis_routed"} else None
    if proposal_track == "A_BOX":
        return "a_box_repair"
    if proposal_track == "T_BOX":
        return "t_box_repair"
    return None


def _uses_few_shot(example_policies: Iterable[str]) -> bool:
    return any(policy != "zero_shot" for policy in example_policies)


def _default_zero_shot_baseline_summary_path() -> Path | None:
    candidate = (
        _repo_root()
        / "reports"
        / "prompt_dev"
        / f"evaluation_{PROMPT_DEV_VERSION}_holdout96_zero_shot"
        / "prompt_dev_evaluation_summary.json"
    )
    return candidate if candidate.exists() else None


def _load_zero_shot_baseline_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Zero-shot baseline summary does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Zero-shot baseline summary must be a JSON object: {path}")
    payload["_summary_path"] = str(path)
    return payload


def _eval_manifest_path(options: PromptDevRenderOptions | PromptDevEvaluateOptions) -> Path:
    manifest_path = options.eval_manifest or options.dev_manifest
    if manifest_path is None:
        raise ValueError("Prompt development requires --eval-manifest or its legacy alias --dev-manifest.")
    return manifest_path


def _example_manifest_path(options: PromptDevRenderOptions | PromptDevEvaluateOptions) -> Path | None:
    return options.example_manifest


def _validate_core_example_guard(options: PromptDevRenderOptions | PromptDevEvaluateOptions) -> None:
    if not _uses_few_shot(options.example_policies):
        return
    if options.support_set_manifest is not None and not options.support_set_manifest.exists():
        raise ValueError(f"Support-set manifest does not exist: {options.support_set_manifest}")
    if "static_diverse_kshot" in options.example_policies and options.support_set_manifest is None:
        raise ValueError("example_policy=static_diverse_kshot requires --support-set-manifest.")
    if options.example_manifest is None and options.support_set_manifest is None and not options.allow_core_example_risk:
        raise ValueError(
            "Few-shot prompt development requires --example-manifest or --support-set-manifest so examples are "
            "selected outside the evaluation manifest. Pass --allow-core-example-risk only for an explicit "
            "leakage-risk experiment."
        )
    if options.core_manifest is not None and options.core_manifest.exists():
        return
    if options.allow_core_example_risk:
        return
    raise ValueError(
        "Few-shot prompt development requires --core-manifest so examples can exclude core cases. "
        "Pass --allow-core-example-risk only for an explicit leakage-risk experiment."
    )


def _tbox_taxonomy_gold_version_for_manifest(manifest_path: Path) -> str:
    name = manifest_path.name
    if name.startswith("core_"):
        return "tbox_taxonomy_patch_gold_core_v1"
    if "dev_prompt_holdout" in name:
        return "tbox_taxonomy_patch_gold_dev_holdout_v1"
    return "tbox_taxonomy_patch_gold_ad_hoc_v1"


def render_prompt_dev_prompts(options: PromptDevRenderOptions) -> dict[str, Any]:
    log = logging.getLogger("prompt_dev")
    eval_manifest_path = _eval_manifest_path(options)
    example_manifest_path = _example_manifest_path(options)
    support_manifest = _load_support_set_manifest(options.support_set_manifest)
    support_source_manifest = (
        Path(str(support_manifest["source_manifest"]))
        if isinstance(support_manifest, dict) and isinstance(support_manifest.get("source_manifest"), str)
        else None
    )
    candidate_manifest_path = example_manifest_path or support_source_manifest or eval_manifest_path
    log.info(
        "render: loading manifest records classified=%s eval_manifest=%s example_manifest=%s max_cases=%s sample_strategy=%s track_filter=%s",
        options.classified_benchmark,
        eval_manifest_path,
        example_manifest_path,
        options.max_cases,
        options.sample_strategy,
        options.track_filter,
    )
    _validate_core_example_guard(options)
    manifest = load_selection_manifest(eval_manifest_path)
    eval_records = _load_manifest_records(
        options.classified_benchmark,
        eval_manifest_path,
        max_cases=options.max_cases,
        sample_strategy=options.sample_strategy,
        seed=options.seed,
        track_filter=options.track_filter,
    )
    log.info("render: selected eval records=%s", len(eval_records))
    log.info("render: loading candidate records for examples and visible ids")
    candidate_records = _load_all_manifest_records(options.classified_benchmark, candidate_manifest_path)
    log.info("render: loaded candidate records=%s", len(candidate_records))
    records_by_id = {
        record["id"]: record
        for record in candidate_records
        if isinstance(record, dict) and isinstance(record.get("id"), str)
    }
    visible_case_ids = _visible_case_id_map([*eval_records, *candidate_records])
    blocked_core = _blocked_core_sets(options.core_manifest)
    matrix = build_prompt_dev_matrix(
        PromptDevMatrixOptions(
            representations=options.representations,
            example_policies=options.example_policies,
            context_bundles=options.context_bundles,
            diagnosis_context_bundles=options.diagnosis_context_bundles,
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
                payload_builder = _prompt_payload_builder_for_task(task, row["context_bundle"])
                case_payload, context_audit = payload_builder(
                    record,
                    world_state_entry,
                    row["context_bundle"],
                )
                case_payload = _model_visible_payload(
                    case_payload,
                    raw_case_id=raw_case_id,
                    visible_case_id=visible_case_id,
                )
                examples = _examples_for_prompt(
                    eval_record=record,
                    task=task,
                    context_bundle=row["context_bundle"],
                    policy=row["example_policy"],
                    seed=options.seed,
                    example_count=options.example_count,
                    blocked_core=blocked_core,
                    allow_same_property=options.allow_same_property_examples,
                    candidate_records=candidate_records,
                    records_by_id=records_by_id,
                    visible_case_ids=visible_case_ids,
                    world_store=world_store,
                    support_manifest=support_manifest,
                )
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
                    "matrix_task": row["task"],
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
                    "example_leakage_scan": _scan_examples_for_leakage(examples),
                    "examples": [
                        {key: value for key, value in example.items() if key != "input_payload"}
                        for example in examples
                    ],
                }
                prompt_records.append(prompt_record)
                rendered_count += 1

    log.info("render: writing prompt artifacts prompts=%s", rendered_count)
    few_shot_outputs: dict[str, str] = {}
    if _uses_few_shot(options.example_policies):
        leakage_scan = {
            "manifest_type": "few_shot_leakage_scan",
            "manifest_version": PROMPT_DEV_VERSION,
            "created_at_utc": _utc_now(),
            "model_visible_text_scan": _model_visible_text_scan(prompt_records),
            "example_payload_scans": [
                {
                    "matrix_id": record["matrix_id"],
                    "case_id": record["case_id"],
                    "task": record["task"],
                    "example_leakage_scan": record.get("example_leakage_scan") or {},
                }
                for record in prompt_records
            ],
        }
        overlap_report = {
            "manifest_type": "few_shot_overlap_report",
            "manifest_version": PROMPT_DEV_VERSION,
            "created_at_utc": _utc_now(),
            **_example_overlap_report(
                prompt_records=prompt_records,
                eval_records_by_id={
                    record["id"]: record for record in eval_records if isinstance(record.get("id"), str)
                },
                example_records_by_id=records_by_id,
                blocked_core=blocked_core,
                allow_same_property_examples=options.allow_same_property_examples,
            ),
        }
        leakage_path = output_dir / "few_shot_leakage_scan.json"
        overlap_json_path = output_dir / "few_shot_overlap_report.json"
        overlap_md_path = output_dir / "few_shot_overlap_report.md"
        write_json(leakage_path, leakage_scan)
        write_json(overlap_json_path, overlap_report)
        overlap_md_path.write_text(_overlap_report_markdown(overlap_report), encoding="utf-8")
        few_shot_outputs = {
            "few_shot_leakage_scan": str(leakage_path),
            "few_shot_overlap_report_json": str(overlap_json_path),
            "few_shot_overlap_report_markdown": str(overlap_md_path),
        }
        if not leakage_scan["model_visible_text_scan"]["passed"]:
            raise ValueError(f"Few-shot model-visible leakage scan failed: {leakage_path}")
        if any(not row["example_leakage_scan"].get("passed", False) for row in leakage_scan["example_payload_scans"]):
            raise ValueError(f"Few-shot example payload leakage scan failed: {leakage_path}")
        if not overlap_report["passed"]:
            raise ValueError(f"Few-shot overlap report failed: {overlap_json_path}")

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
            "eval_manifest": str(eval_manifest_path),
            "dev_manifest": str(eval_manifest_path),
            "example_manifest": str(example_manifest_path) if example_manifest_path else None,
            "core_manifest": str(options.core_manifest) if options.core_manifest else None,
            "support_set_manifest": str(options.support_set_manifest) if options.support_set_manifest else None,
            "sample_strategy": options.sample_strategy,
            "max_cases": options.max_cases,
            "track_filter": list(options.track_filter) if options.track_filter else None,
        },
        "outputs": {
            "prompts_jsonl": str(prompts_path),
            "review_markdown": str(review_path),
            **few_shot_outputs,
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
    schema_counts: dict[str, Counter[str]] = {}
    example_count_by_task: Counter[str] = Counter()
    for record in prompt_records:
        task = str(record.get("task") or "unknown")
        schema_counts.setdefault(task, Counter())
        for example in record.get("examples", []):
            if not isinstance(example, dict):
                continue
            example_count_by_task[task] += 1
            schema_counts[task].update([str(example.get("task_schema") or "unknown")])
    lines = [
        "# Prompt Development Review",
        "",
        "No LLM inference was run for this artifact.",
        "",
        f"Rendered prompts: `{summary['counts']['rendered_prompts']}`",
        "",
        "## Example Schema Summary",
        "",
        "| Task | Example count | Example schemas |",
        "| --- | ---: | --- |",
    ]
    for task, counts in sorted(schema_counts.items()):
        schema_text = ", ".join(f"`{schema}`: {count}" for schema, count in sorted(counts.items())) or "n/a"
        lines.append(f"| `{task}` | {example_count_by_task[task]} | {schema_text} |")
    lines.append("")
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
    # Prompt-dev requests are one case at a time and use neutral model-visible IDs.
    # Attribute the parsed payload to the request case before parser/evaluator joins,
    # even if the model copied a shortened or otherwise slightly malformed neutral ID.
    if isinstance(case_id, str) and case_id.strip():
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
    proposal_track = prompt_record.get("proposal_track_used")
    if not isinstance(proposal_track, str) or not proposal_track:
        proposal_track = prompt_record.get("historical_track") if prompt_record.get("task") != "track_diagnosis" else None
    task_type = "track_diagnosis" if prompt_record.get("task") == "track_diagnosis" else "proposal"
    prompt_dev_task = prompt_record.get("matrix_task") or prompt_record["task"]
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
        "routing_source": prompt_record.get("routing_source") or (
            "diagnosis_prediction"
            if prompt_record.get("track_mode") == "diagnosis_routed" and task_type == "proposal"
            else "prompt_dev"
        ),
        "task_type": task_type,
        "prompt_dev_task": prompt_dev_task,
        "provider": getattr(provider, "provider_name", provider.__class__.__name__),
        "model": getattr(provider, "model", "unknown-model"),
        "context_audit": prompt_record.get("context_audit") or {},
    }
    if proposal_track == "T_BOX":
        metadata["t_box_constraint_type_qids"] = _t_box_constraint_type_qids(record, world_state_entry)
        if PROMPT_DEV_VERSION == PROMPT_DEV_TBOX_TAXONOMY_PATCH_VERSION:
            metadata["tbox_task_version"] = "tbox_taxonomy_patch_v1"
            metadata["strict_tbox_signature_diagnostic"] = "enabled"
        else:
            metadata["tbox_task_version"] = "strict_signature_after_v1"
            metadata["strict_tbox_signature_diagnostic"] = "headline"
    if proposal_track == "A_BOX":
        metadata["abox_task_version"] = "prompt_dev_v4_spec_only"
    metadata["prompt_version"] = PROMPT_DEV_VERSION
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
            "prompt_version",
            "tbox_task_version",
            "abox_task_version",
            "strict_tbox_signature_diagnostic",
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
            "prompt_version",
            "tbox_task_version",
            "abox_task_version",
            "strict_tbox_signature_diagnostic",
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
        elif (
            metadata.get("proposal_track_used") == "T_BOX"
            and metadata.get("tbox_task_version") == "tbox_taxonomy_patch_v1"
        ):
            normalized = normalize_tbox_taxonomy_patch(
                normalized_payload,
                constraint_type_qids=metadata.get("t_box_constraint_type_qids"),
            )
            _append_jsonl(matrix_dir / "t_box_taxonomy_patch_proposals.jsonl", normalized.schema_dict())
        elif metadata.get("proposal_track_used") == "T_BOX":
            normalized = normalize_t_box_proposal(
                normalized_payload,
                constraint_type_qids=metadata.get("t_box_constraint_type_qids"),
            )
            _append_jsonl(matrix_dir / "t_box_proposals.jsonl", normalized.to_dict())
        else:
            if _is_explicit_empty_a_box_ops(normalized_payload):
                manifest_record["parse_status"] = "non_executable_empty_ops"
                manifest_record["parser_warning"] = "A-box output was valid JSON but contained no executable operations."
                _append_jsonl(matrix_dir / "raw_model_responses.jsonl", raw_record)
                _append_jsonl(matrix_dir / "run_manifest.jsonl", manifest_record)
                return manifest_record
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


def _is_explicit_empty_a_box_ops(payload: Any) -> bool:
    return isinstance(payload, dict) and "ops" in payload and payload.get("ops") == []


def _diagnosis_track_from_payload(parsed_payload: Any, case_id: str, visible_case_id: str | None = None) -> str:
    try:
        normalized = normalize_diagnosis(_payload_with_case_id(parsed_payload, case_id, visible_case_id))
    except Exception:
        return "UNROUTABLE"
    if normalized.predicted_track in {"A_BOX", "T_BOX", "AMBIGUOUS"}:
        return normalized.predicted_track
    return "UNROUTABLE"


def _diagnosis_track_from_file(matrix_dir: Path, case_id: str) -> str:
    path = matrix_dir / "track_diagnoses.jsonl"
    if not path.exists():
        return "UNROUTABLE"
    for row in iter_jsonl(path):
        if not isinstance(row, dict) or row.get("case_id") != case_id:
            continue
        predicted = row.get("predicted_track")
        if predicted in {"A_BOX", "T_BOX", "AMBIGUOUS"}:
            return predicted
    return "UNROUTABLE"


def _record_skipped_routed_proposal_result(
    *,
    matrix_dir: Path,
    prompt_record: dict[str, Any],
    metadata: dict[str, Any],
    usage: dict[str, Any],
    parse_status: str,
    skip_reason: str,
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
        "usage": _usage_block(usage, None),
        "context_audit": metadata.get("context_audit") or {},
        "timestamp_utc": _utc_now(),
        "parse_status": parse_status,
        "skip_reason": skip_reason,
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
        "raw_response": None,
        "parsed_payload": None,
        "skip_reason": skip_reason,
    }
    _append_jsonl(matrix_dir / "raw_model_responses.jsonl", raw_record)
    _append_jsonl(matrix_dir / "run_manifest.jsonl", manifest_record)
    return manifest_record


def _render_routed_proposal_prompt_record(
    *,
    diagnosis_prompt_record: dict[str, Any],
    proposal_track: str,
    record: dict[str, Any],
    world_state_entry: dict[str, Any] | None,
    world_store: WorldStateStore,
    candidate_records: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    visible_case_ids: dict[str, str],
    blocked_core: dict[str, set[str]],
    support_manifest: dict[str, Any] | None,
    options: PromptDevEvaluateOptions,
) -> dict[str, Any]:
    task = "a_box_repair" if proposal_track == "A_BOX" else "t_box_repair"
    raw_case_id = record["id"]
    visible_case_id = diagnosis_prompt_record.get("visible_case_id") or visible_case_ids.get(
        raw_case_id,
        f"case_{_stable_hash(raw_case_id)[:12]}",
    )
    case_payload, context_audit = _bundle_payload_and_audit(record, world_state_entry, diagnosis_prompt_record["context_bundle"])
    case_payload = _model_visible_payload(case_payload, raw_case_id=raw_case_id, visible_case_id=visible_case_id)
    examples = _examples_for_prompt(
        eval_record=record,
        task=task,
        context_bundle=diagnosis_prompt_record["context_bundle"],
        policy=diagnosis_prompt_record["example_policy"],
        seed=options.seed,
        example_count=options.example_count,
        blocked_core=blocked_core,
        allow_same_property=options.allow_same_property_examples,
        candidate_records=candidate_records,
        records_by_id=records_by_id,
        visible_case_ids=visible_case_ids,
        world_store=world_store,
        support_manifest=support_manifest,
    )
    rendered = render_prompt_dev_prompt(
        task=task,
        representation=diagnosis_prompt_record["representation"],
        case_payload=case_payload,
        examples=examples,
        include_abstention=diagnosis_prompt_record["include_abstention"],
    )
    return {
        **diagnosis_prompt_record,
        "task": task,
        "matrix_task": "repair_proposal",
        "proposal_track_used": proposal_track,
        "routing_source": "diagnosis_prediction",
        "prompt_name": rendered.prompt_name,
        "system_prompt": rendered.system_prompt,
        "user_prompt": rendered.user_prompt,
        "response_format": rendered.response_format,
        "context_audit": context_audit,
        "example_leakage_scan": _scan_examples_for_leakage(examples),
        "examples": [
            {key: value for key, value in example.items() if key != "input_payload"}
            for example in examples
        ],
    }


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


def _ensure_matrix_info(
    matrices: dict[str, dict[str, Any]],
    prompt_record: dict[str, Any],
    matrix_dir: Path,
) -> dict[str, Any]:
    matrix_task = prompt_record.get("matrix_task") or (
        "track_diagnosis" if prompt_record["task"] == "track_diagnosis" else "repair_proposal"
    )
    return matrices.setdefault(
        prompt_record["matrix_id"],
        {
            "matrix_id": prompt_record["matrix_id"],
            "output_dir": str(matrix_dir),
            "representation": prompt_record["representation"],
            "example_policy": prompt_record["example_policy"],
            "context_bundle": prompt_record["context_bundle"],
            "task": matrix_task,
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


def _accumulate_prompt_result(
    *,
    matrices: dict[str, dict[str, Any]],
    prompt_counts: Counter,
    prompt_record: dict[str, Any],
    manifest_record: dict[str, Any],
    matrix_dir: Path,
) -> None:
    parse_status = str(manifest_record.get("parse_status") or "unknown")
    prompt_counts[parse_status] += 1
    matrix_info = _ensure_matrix_info(matrices, prompt_record, matrix_dir)
    matrix_info["case_ids"].append(prompt_record["case_id"])
    matrix_info["counts"][parse_status] += 1
    matrix_info["status_counts"][_status_bucket(parse_status)] += 1
    matrix_info["by_historical_track"][prompt_record.get("historical_track") or "unknown"] += 1
    matrix_info["by_task"][prompt_record["task"]] += 1
    matrix_info["by_context"][prompt_record["context_bundle"]] += 1


def evaluate_prompt_dev_prompts(
    options: PromptDevEvaluateOptions,
    *,
    provider: ModelProvider | None = None,
) -> dict[str, Any]:
    log = logging.getLogger("prompt_dev")
    eval_manifest_path = _eval_manifest_path(options)
    example_manifest_path = _example_manifest_path(options)
    support_manifest = _load_support_set_manifest(options.support_set_manifest)
    support_source_manifest = (
        Path(str(support_manifest["source_manifest"]))
        if isinstance(support_manifest, dict) and isinstance(support_manifest.get("source_manifest"), str)
        else None
    )
    candidate_manifest_path = example_manifest_path or support_source_manifest or eval_manifest_path
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = output_dir / "rendered_prompts"
    log.info("evaluate: rendering prompts into %s", render_dir)
    render_summary = render_prompt_dev_prompts(
        PromptDevRenderOptions(
            classified_benchmark=options.classified_benchmark,
            world_state=options.world_state,
            eval_manifest=eval_manifest_path,
            example_manifest=example_manifest_path,
            output_dir=render_dir,
            seed=options.seed,
            max_cases=options.max_cases,
            representations=options.representations,
            example_policies=options.example_policies,
            context_bundles=options.context_bundles,
            diagnosis_context_bundles=options.diagnosis_context_bundles,
            tasks=options.tasks,
            repair_track_modes=options.repair_track_modes,
            include_abstention=options.include_abstention,
            core_manifest=options.core_manifest,
            support_set_manifest=options.support_set_manifest,
            example_count=options.example_count,
            allow_same_property_examples=options.allow_same_property_examples,
            sample_strategy=options.sample_strategy,
            allow_core_example_risk=options.allow_core_example_risk,
            track_filter=options.track_filter,
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
        eval_manifest_path,
        max_cases=options.max_cases,
        sample_strategy=options.sample_strategy,
        seed=options.seed,
        track_filter=options.track_filter,
    )
    eval_manifest_payload = load_selection_manifest(eval_manifest_path)
    log.info("evaluate: loaded eval records=%s", len(eval_records))
    records_by_id = {record["id"]: record for record in eval_records if isinstance(record.get("id"), str)}
    log.info("evaluate: loading candidate records for routed prompts and examples")
    candidate_records = _load_all_manifest_records(options.classified_benchmark, candidate_manifest_path)
    support_records_by_id = {
        record["id"]: record
        for record in candidate_records
        if isinstance(record, dict) and isinstance(record.get("id"), str)
    }
    visible_case_ids = _visible_case_id_map([*eval_records, *candidate_records])
    blocked_core = _blocked_core_sets(options.core_manifest)
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
            _accumulate_prompt_result(
                matrices=matrices,
                prompt_counts=prompt_counts,
                prompt_record=prompt_record,
                manifest_record=manifest_record,
                matrix_dir=matrix_dir,
            )
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
            if (
                prompt_record.get("matrix_task") == "repair_proposal"
                and prompt_record.get("track_mode") == "diagnosis_routed"
                and prompt_record.get("task") == "track_diagnosis"
            ):
                diagnosis_was_normalized = manifest_record.get("parse_status") == "normalized" or (
                    should_skip and skip_status == "skipped_existing_normalized"
                )
                if should_skip and skip_status == "skipped_existing_normalized":
                    routed_track = _diagnosis_track_from_file(matrix_dir, prompt_record["case_id"])
                else:
                    routed_track = _diagnosis_track_from_payload(
                        parsed_payload if not should_skip else None,
                        prompt_record["case_id"],
                        prompt_record.get("visible_case_id")
                        if isinstance(prompt_record.get("visible_case_id"), str)
                        else None,
                    )
                if not diagnosis_was_normalized:
                    routed_track = "UNROUTABLE"
                if routed_track in {"A_BOX", "T_BOX"}:
                    proposal_prompt_record = _render_routed_proposal_prompt_record(
                        diagnosis_prompt_record=prompt_record,
                        proposal_track=routed_track,
                        record=record,
                        world_state_entry=world_state_entry,
                        world_store=world_store,
                        candidate_records=candidate_records,
                        records_by_id=support_records_by_id,
                        visible_case_ids=visible_case_ids,
                        blocked_core=blocked_core,
                        support_manifest=support_manifest,
                        options=options,
                    )
                    proposal_metadata = _prompt_record_metadata(
                        run_id=run_id,
                        prompt_record=proposal_prompt_record,
                        provider=provider,
                        record=record,
                        world_state_entry=world_state_entry,
                    )
                    proposal_key = (proposal_prompt_record["case_id"], "proposal")
                    proposal_existing_row = existing_rows.get(proposal_key)
                    proposal_should_skip, proposal_skip_status = _should_skip_existing_prompt_result(
                        proposal_existing_row,
                        retry_failures=options.retry_failures,
                    )
                    proposal_started = time.perf_counter()
                    if proposal_should_skip:
                        proposal_manifest_record = dict(proposal_existing_row or {})
                        proposal_manifest_record["parse_status"] = proposal_skip_status
                    else:
                        proposal_prompt_char_count = _prompt_char_count(proposal_prompt_record)
                        if options.max_prompt_chars is not None and proposal_prompt_char_count > options.max_prompt_chars:
                            proposal_manifest_record = _record_prompt_dev_result(
                                matrix_dir=matrix_dir,
                                prompt_record=proposal_prompt_record,
                                metadata=proposal_metadata,
                                raw_response=None,
                                parsed_payload=None,
                                usage=_empty_usage(provider, proposal_metadata),
                                elapsed_seconds=None,
                                error_message=(
                                    f"prompt length {proposal_prompt_char_count} exceeds --max-prompt-chars "
                                    f"{options.max_prompt_chars}"
                                ),
                            )
                        else:
                            try:
                                proposal_raw, proposal_parsed, proposal_usage = provider.generate(
                                    proposal_prompt_record["user_prompt"],
                                    proposal_prompt_record["system_prompt"],
                                    proposal_prompt_record["response_format"],
                                    proposal_metadata,
                                )
                                proposal_elapsed = time.perf_counter() - proposal_started
                                proposal_error = None
                            except Exception as exc:
                                proposal_raw = None
                                proposal_parsed = None
                                proposal_usage = _empty_usage(provider, proposal_metadata)
                                proposal_elapsed = time.perf_counter() - proposal_started
                                proposal_error = str(exc)
                            proposal_manifest_record = _record_prompt_dev_result(
                                matrix_dir=matrix_dir,
                                prompt_record=proposal_prompt_record,
                                metadata=proposal_metadata,
                                raw_response=proposal_raw,
                                parsed_payload=proposal_parsed,
                                usage=proposal_usage,
                                elapsed_seconds=proposal_elapsed,
                                error_message=proposal_error,
                            )
                    _accumulate_prompt_result(
                        matrices=matrices,
                        prompt_counts=prompt_counts,
                        prompt_record=proposal_prompt_record,
                        manifest_record=proposal_manifest_record,
                        matrix_dir=matrix_dir,
                    )
                    _notify_progress(
                        options.progress_callback,
                        {
                            "event": "advance",
                            "matrix_id": matrix_id,
                            "case_id": proposal_prompt_record["case_id"],
                            "task": proposal_prompt_record["task"],
                            "parse_status": proposal_manifest_record["parse_status"],
                        },
                    )
                else:
                    skip_reason = (
                        "diagnosis_routed_ambiguous_track"
                        if routed_track == "AMBIGUOUS"
                        else "diagnosis_routed_unroutable_track"
                    )
                    skipped_prompt_record = {
                        **prompt_record,
                        "task": "repair_proposal",
                        "matrix_task": "repair_proposal",
                        "proposal_track_used": routed_track,
                        "routing_source": "diagnosis_prediction",
                    }
                    skipped_metadata = _prompt_record_metadata(
                        run_id=run_id,
                        prompt_record=skipped_prompt_record,
                        provider=provider,
                        record=record,
                        world_state_entry=world_state_entry,
                    )
                    skipped_metadata["proposal_track_used"] = routed_track
                    skipped_metadata["task_type"] = "proposal"
                    skipped_metadata["prompt_dev_task"] = "repair_proposal"
                    skipped_manifest_record = _record_skipped_routed_proposal_result(
                        matrix_dir=matrix_dir,
                        prompt_record=skipped_prompt_record,
                        metadata=skipped_metadata,
                        usage=_empty_usage(provider, skipped_metadata),
                        parse_status=(
                            "skipped_ambiguous_track"
                            if routed_track == "AMBIGUOUS"
                            else "skipped_unroutable_track"
                        ),
                        skip_reason=skip_reason,
                    )
                    _accumulate_prompt_result(
                        matrices=matrices,
                        prompt_counts=prompt_counts,
                        prompt_record=skipped_prompt_record,
                        manifest_record=skipped_manifest_record,
                        matrix_dir=matrix_dir,
                    )
                    _notify_progress(
                        options.progress_callback,
                        {
                            "event": "advance",
                            "matrix_id": matrix_id,
                            "case_id": skipped_prompt_record["case_id"],
                            "task": skipped_prompt_record["task"],
                            "parse_status": skipped_manifest_record["parse_status"],
                        },
                    )

    results: list[dict[str, Any]] = []
    log.info("evaluate: model request loop complete status_counts=%s", dict(prompt_counts))
    for matrix_id, matrix_info in sorted(matrices.items()):
        matrix_dir = Path(matrix_info["output_dir"])
        unique_case_ids = sorted(set(matrix_info["case_ids"]))
        log.info("evaluate: scoring matrix=%s cases=%s", matrix_id, len(unique_case_ids))
        if matrix_info["task"] == "repair_proposal" and matrix_info.get("track_mode") != "diagnosis_routed":
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
        taxonomy_eval_summary = None
        taxonomy_eval_path = matrix_dir / "tbox_taxonomy_patch_evaluation_summary.json"
        if (
            PROMPT_DEV_VERSION == PROMPT_DEV_TBOX_TAXONOMY_PATCH_VERSION
            and matrix_info["task"] == "repair_proposal"
        ):
            tbox_records = [
                record
                for record in eval_records
                if record.get("id") in set(unique_case_ids) and record.get("track") == "T_BOX"
            ]
            gold_rows = [
                gold
                for gold in (
                    gold_patch_for_record(
                        record,
                        annotation=(eval_manifest_payload.get("case_annotations") or {}).get(record.get("id"), {}),
                    )
                    for record in tbox_records
                )
                if gold is not None
            ]
            prediction_path = matrix_dir / "t_box_taxonomy_patch_proposals.jsonl"
            prediction_rows = load_jsonl(prediction_path) if prediction_path.exists() else []
            taxonomy_eval_summary = evaluate_tbox_taxonomy_patch_predictions(
                gold_rows=gold_rows,
                prediction_rows=prediction_rows,
                case_annotations=eval_manifest_payload.get("case_annotations") or {},
                gold_version=_tbox_taxonomy_gold_version_for_manifest(eval_manifest_path),
            )
            write_json(taxonomy_eval_path, taxonomy_eval_summary)
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
        if taxonomy_eval_summary is not None:
            result["tbox_taxonomy_patch_evaluation_summary"] = str(taxonomy_eval_path)
            result["tbox_taxonomy_patch_gold_version"] = taxonomy_eval_summary.get("gold_version")
            result["tbox_taxonomy_patch_metrics"] = taxonomy_eval_summary.get("subsets", {}).get("all_core", {}).get(
                "metrics",
                {},
            )
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
            "eval_manifest": str(eval_manifest_path),
            "dev_manifest": str(eval_manifest_path),
            "example_manifest": str(example_manifest_path) if example_manifest_path else None,
            "core_manifest": str(options.core_manifest) if options.core_manifest else None,
            "support_set_manifest": str(options.support_set_manifest) if options.support_set_manifest else None,
            "zero_shot_baseline_summary": str(options.zero_shot_baseline_summary)
            if options.zero_shot_baseline_summary
            else (
                str(_default_zero_shot_baseline_summary_path())
                if _uses_few_shot(options.example_policies)
                and "zero_shot" not in options.example_policies
                and _default_zero_shot_baseline_summary_path() is not None
                else None
            ),
            "render_summary": str(render_dir / "prompt_dev_render_summary.json"),
            "sample_strategy": options.sample_strategy,
            "max_cases": options.max_cases,
            "track_filter": list(options.track_filter) if options.track_filter else None,
        },
        "outputs": {
            "output_dir": str(output_dir),
            "comparison_markdown": str(output_dir / "prompt_dev_evaluation_comparison.md"),
        },
        "counts": {
            "rendered_prompts": render_summary["counts"]["rendered_prompts"],
            "evaluated_prompts": sum(prompt_counts.values()),
            "matrix_rows": len(results),
            "by_parse_status": dict(prompt_counts),
        },
        "results": results,
    }
    render_outputs = render_summary.get("outputs") if isinstance(render_summary.get("outputs"), dict) else {}
    for key in (
        "few_shot_leakage_scan",
        "few_shot_overlap_report_json",
        "few_shot_overlap_report_markdown",
    ):
        if isinstance(render_outputs.get(key), str):
            summary["outputs"][key] = render_outputs[key]
    diagnosis_report = _write_track_diagnosis_report(
        output_dir=output_dir,
        results=results,
        eval_records=eval_records,
        manifest=eval_manifest_payload,
    )
    if diagnosis_report:
        summary["outputs"]["track_diagnosis_report_json"] = str(output_dir / "track_diagnosis_report.json")
        summary["outputs"]["track_diagnosis_report_markdown"] = str(output_dir / "track_diagnosis_report.md")
        summary["track_diagnosis_report"] = {
            "matrix_count": len(diagnosis_report.get("matrices", [])),
            "gates": diagnosis_report.get("gates"),
        }
    if _uses_few_shot(options.example_policies):
        baseline_path = options.zero_shot_baseline_summary
        if baseline_path is None and "zero_shot" not in options.example_policies:
            baseline_path = _default_zero_shot_baseline_summary_path()
        zero_shot_baseline_summary = _load_zero_shot_baseline_summary(baseline_path)
        few_shot_config, few_shot_report = _few_shot_reports(
            summary=summary,
            output_dir=output_dir,
            diagnosis_report=diagnosis_report,
            zero_shot_baseline_summary=zero_shot_baseline_summary,
        )
        summary["outputs"]["few_shot_run_config"] = str(output_dir / "few_shot_run_config.json")
        summary["outputs"]["few_shot_delta_vs_zero_shot_json"] = str(output_dir / "few_shot_delta_vs_zero_shot.json")
        summary["outputs"]["few_shot_delta_vs_zero_shot_markdown"] = str(output_dir / "few_shot_delta_vs_zero_shot.md")
        summary["few_shot_report"] = {
            "condition_count": len(few_shot_config.get("few_shot_conditions", {})),
            "a_box_comparison_count": len(few_shot_report.get("sections", {}).get("a_box", [])),
            "t_box_comparison_count": len(few_shot_report.get("sections", {}).get("t_box_taxonomy_patch", [])),
            "diagnosis_comparison_count": len(few_shot_report.get("sections", {}).get("diagnosis", [])),
            "unmatched_few_shot_matrix_count": len(few_shot_report.get("unmatched_few_shot_matrix_ids", [])),
        }
    summary_path = output_dir / "prompt_dev_evaluation_summary.json"
    write_json(summary_path, summary)
    (output_dir / "prompt_dev_evaluation_comparison.md").write_text(
        _evaluation_comparison_markdown(summary),
        encoding="utf-8",
    )
    log.info("evaluate: wrote summary=%s comparison=%s", summary_path, output_dir / "prompt_dev_evaluation_comparison.md")
    return summary


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _annotation_bool(annotation: dict[str, Any], key: str) -> bool:
    value = annotation.get(key)
    return value is True or (isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"})


def _record_slice_metadata(record: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    case_id = str(record.get("id") or "")
    annotation = _manifest_annotation(manifest, case_id)
    cls, subtype = _classification_parts(record)
    return {
        "class": annotation.get("class") if isinstance(annotation.get("class"), str) else cls,
        "subtype": annotation.get("subtype") if isinstance(annotation.get("subtype"), str) else subtype,
        "selection_stratum": _selection_stratum(manifest, case_id),
        "main_score": _annotation_bool(annotation, "main_score")
        or _annotation_bool(annotation, "main_score_case")
        or _annotation_bool(annotation, "is_main_score"),
        "diagnostic_only": _annotation_bool(annotation, "diagnostic_only")
        or _annotation_bool(annotation, "diagnostic_case"),
    }


def _read_diagnosis_predictions(matrix_dir: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    path = matrix_dir / "track_diagnoses.jsonl"
    if not path.exists():
        return predictions
    for row in iter_jsonl(path):
        if not isinstance(row, dict):
            continue
        case_id = row.get("case_id")
        predicted = row.get("predicted_track")
        if isinstance(case_id, str) and predicted in {"A_BOX", "T_BOX", "AMBIGUOUS"}:
            predictions[case_id] = predicted
    return predictions


def _diagnosis_metrics_for_matrix(
    *,
    result: dict[str, Any],
    eval_records_by_id: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    matrix_dir = Path(result["output_dir"])
    manifest_path = matrix_dir / "run_manifest.jsonl"
    if not manifest_path.exists():
        return None
    manifest_rows = [
        row for row in iter_jsonl(manifest_path)
        if isinstance(row, dict) and row.get("task_type") == "track_diagnosis"
        and row.get("skip_reason") != "not_run_for_repair_proposal_prompt_matrix"
    ]
    if not manifest_rows:
        return None
    predictions = _read_diagnosis_predictions(matrix_dir)
    confusion: dict[str, Counter] = {"A_BOX": Counter(), "T_BOX": Counter()}
    slice_counters: dict[str, dict[str, Counter]] = {
        "class": {},
        "subtype": {},
        "selection_stratum": {},
        "main_score": {},
        "diagnostic_only": {},
    }
    parse_error_count = 0
    request_error_count = 0
    skipped_count = 0
    ambiguous_count = 0
    wrong_route_count = 0
    normalized_count = 0
    total = 0
    for row in manifest_rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            continue
        record = eval_records_by_id.get(case_id)
        if not isinstance(record, dict):
            continue
        historical = record.get("track")
        if historical not in {"A_BOX", "T_BOX"}:
            continue
        total += 1
        status = row.get("parse_status")
        if status == "normalized":
            normalized_count += 1
            predicted = predictions.get(case_id, "UNPARSED")
        elif status == "parse_error":
            parse_error_count += 1
            predicted = "PARSE_ERROR"
        elif status == "request_error":
            request_error_count += 1
            predicted = "REQUEST_ERROR"
        elif isinstance(status, str) and status.startswith("skipped"):
            skipped_count += 1
            predicted = "SKIPPED"
        else:
            predicted = "UNPARSED"
        confusion[historical][predicted] += 1
        if predicted == "AMBIGUOUS":
            ambiguous_count += 1
        if predicted in {"A_BOX", "T_BOX"} and predicted != historical:
            wrong_route_count += 1
        slice_meta = _record_slice_metadata(record, manifest)
        for slice_name, value in slice_meta.items():
            key = str(value)
            slice_counters[slice_name].setdefault(key, Counter())[f"{historical}->{predicted}"] += 1
            slice_counters[slice_name][key]["total"] += 1

    a_total = sum(confusion["A_BOX"].values())
    t_total = sum(confusion["T_BOX"].values())
    a_recall = _safe_divide(confusion["A_BOX"].get("A_BOX", 0), a_total)
    t_recall = _safe_divide(confusion["T_BOX"].get("T_BOX", 0), t_total)
    recalls = [value for value in (a_recall, t_recall) if value is not None]
    balanced_accuracy = sum(recalls) / len(recalls) if recalls else None
    f1_scores: dict[str, float | None] = {}
    for label in ("A_BOX", "T_BOX"):
        tp = confusion[label].get(label, 0)
        fp = sum(confusion[other].get(label, 0) for other in ("A_BOX", "T_BOX") if other != label)
        fn = sum(count for predicted, count in confusion[label].items() if predicted != label)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1_scores[label] = (
            None
            if precision is None or recall is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    macro_f1_values = [value for value in f1_scores.values() if value is not None]
    macro_f1 = sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else None
    request_error_rate = _safe_divide(request_error_count, total) or 0.0
    parse_error_rate = _safe_divide(parse_error_count, total) or 0.0
    ambiguous_rate = _safe_divide(ambiguous_count, total) or 0.0
    wrong_route_rate = _safe_divide(wrong_route_count, total) or 0.0
    gates = {
        "request_error_rate": {
            "value": request_error_rate,
            "threshold": DIAGNOSIS_ACCEPTANCE_GATES["request_error_rate_max"],
            "passed": request_error_rate <= DIAGNOSIS_ACCEPTANCE_GATES["request_error_rate_max"],
        },
        "parse_error_rate": {
            "value": parse_error_rate,
            "threshold": DIAGNOSIS_ACCEPTANCE_GATES["parse_error_rate_max"],
            "passed": parse_error_rate <= DIAGNOSIS_ACCEPTANCE_GATES["parse_error_rate_max"],
        },
        "balanced_accuracy": {
            "value": balanced_accuracy,
            "threshold": DIAGNOSIS_ACCEPTANCE_GATES["balanced_accuracy_min"],
            "passed": balanced_accuracy is not None
            and balanced_accuracy >= DIAGNOSIS_ACCEPTANCE_GATES["balanced_accuracy_min"],
        },
        "a_box_recall": {
            "value": a_recall,
            "threshold": DIAGNOSIS_ACCEPTANCE_GATES["a_box_recall_min"],
            "passed": a_recall is not None and a_recall >= DIAGNOSIS_ACCEPTANCE_GATES["a_box_recall_min"],
        },
        "t_box_recall": {
            "value": t_recall,
            "threshold": DIAGNOSIS_ACCEPTANCE_GATES["t_box_recall_min"],
            "passed": t_recall is not None and t_recall >= DIAGNOSIS_ACCEPTANCE_GATES["t_box_recall_min"],
        },
        "ambiguous_rate": {
            "value": ambiguous_rate,
            "threshold": DIAGNOSIS_ACCEPTANCE_GATES["ambiguous_rate_max"],
            "passed": ambiguous_rate <= DIAGNOSIS_ACCEPTANCE_GATES["ambiguous_rate_max"],
        },
    }
    return {
        "matrix_id": result["matrix_id"],
        "task": result.get("task"),
        "context_bundle": result.get("context_bundle"),
        "representation": result.get("representation"),
        "example_policy": result.get("example_policy"),
        "track_mode": result.get("track_mode"),
        "counts": {
            "total": total,
            "normalized": normalized_count,
            "parse_error": parse_error_count,
            "request_error": request_error_count,
            "skipped": skipped_count,
            "ambiguous": ambiguous_count,
        },
        "confusion_by_historical_track": {
            track: dict(counter) for track, counter in confusion.items()
        },
        "confusion_by_slice": {
            slice_name: {key: dict(counter) for key, counter in sorted(counters.items())}
            for slice_name, counters in slice_counters.items()
        },
        "metrics": {
            "a_box_recall": a_recall,
            "t_box_recall": t_recall,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "a_box_f1": f1_scores["A_BOX"],
            "t_box_f1": f1_scores["T_BOX"],
            "ambiguous_rate": ambiguous_rate,
            "wrong_route_rate": wrong_route_rate,
            "request_error_rate": request_error_rate,
            "parse_error_rate": parse_error_rate,
        },
        "routed_risk": {
            "wrong_repair_prompt_count": wrong_route_count,
            "wrong_repair_prompt_rate": wrong_route_rate,
            "ambiguous_skipped_count": ambiguous_count,
            "ambiguous_skipped_rate": ambiguous_rate,
        },
        "gates": gates,
        "eligible_for_routed_canary": all(item["passed"] for item in gates.values()),
    }


def _write_track_diagnosis_report(
    *,
    output_dir: Path,
    results: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    eval_records_by_id = {record["id"]: record for record in eval_records if isinstance(record.get("id"), str)}
    matrices = [
        matrix_report
        for result in results
        if (matrix_report := _diagnosis_metrics_for_matrix(
            result=result,
            eval_records_by_id=eval_records_by_id,
            manifest=manifest,
        ))
        is not None
    ]
    if not matrices:
        return None
    report = {
        "manifest_type": "prompt_dev_track_diagnosis_report",
        "manifest_version": PROMPT_DEV_VERSION,
        "created_at_utc": _utc_now(),
        "gates": DIAGNOSIS_ACCEPTANCE_GATES,
        "matrices": matrices,
        "eligible_matrices": [matrix["matrix_id"] for matrix in matrices if matrix["eligible_for_routed_canary"]],
    }
    write_json(output_dir / "track_diagnosis_report.json", report)
    (output_dir / "track_diagnosis_report.md").write_text(_track_diagnosis_report_markdown(report), encoding="utf-8")
    return report


def _result_key(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        result.get("task"),
        result.get("representation"),
        result.get("context_bundle"),
        result.get("track_mode"),
    )


def _few_shot_condition(policy: str) -> dict[str, str]:
    if policy == "static_diverse_kshot":
        return {
            "selection_type": "static_support_set",
            "paper_status": "paper-facing",
            "interpretation": "Static few-shot oracle ablation.",
        }
    return {
        "selection_type": "dynamic_retrieval",
        "paper_status": "exploratory",
        "interpretation": "Dynamic retrieval ablation; do not merge with paper-facing static few-shot results.",
    }


def _metric_value(metrics: dict[str, Any], key: str) -> float | int | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        rate = value.get("rate")
        return rate if isinstance(rate, (int, float)) else None
    return value if isinstance(value, (int, float)) else None


def _metric_delta(zero_value: float | int | None, few_value: float | int | None) -> float | None:
    if not isinstance(zero_value, (int, float)) or not isinstance(few_value, (int, float)):
        return None
    return float(few_value) - float(zero_value)


def _comparison_metric(
    *,
    zero_value: float | int | None,
    few_value: float | int | None,
    source: str,
) -> dict[str, Any]:
    return {
        "zero_shot": zero_value,
        "few_shot": few_value,
        "delta": _metric_delta(zero_value, few_value),
        "source": source,
        "available": isinstance(zero_value, (int, float)) and isinstance(few_value, (int, float)),
    }


def _load_matrix_evaluation_summary(result: dict[str, Any]) -> dict[str, Any]:
    path = result.get("evaluation_summary")
    if not isinstance(path, str) or not path:
        return {}
    summary_path = Path(path)
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _group_metrics(result: dict[str, Any], group_name: str, group_key: str) -> dict[str, Any]:
    summary = _load_matrix_evaluation_summary(result)
    group = summary.get(group_name, {})
    if not isinstance(group, dict):
        return {}
    metrics = group.get(group_key, {})
    return metrics if isinstance(metrics, dict) else {}


def _status_rate(result: dict[str, Any], status: str) -> float | None:
    counts = result.get("counts", {})
    by_status = counts.get("by_parse_status", {}) if isinstance(counts, dict) else {}
    total = sum(value for value in by_status.values() if isinstance(value, int))
    count = by_status.get(status)
    if not isinstance(count, int) or total == 0:
        return None
    return count / total


def _parse_error_rate(result: dict[str, Any]) -> float | None:
    parse_errors = result.get("parse_errors", {})
    if isinstance(parse_errors, dict) and isinstance(parse_errors.get("proposal_parse_error_rate"), (int, float)):
        return parse_errors["proposal_parse_error_rate"]
    metrics = result.get("overall_metrics", {})
    return _metric_value(metrics if isinstance(metrics, dict) else {}, "proposal_parse_error_rate")


def _usage_totals(result: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(result.get("output_dir", "")) / "run_manifest.jsonl"
    rows = list(iter_jsonl(manifest_path)) if manifest_path.exists() else []
    fields = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "estimated_cost_usd")
    totals: dict[str, int | float | None] = {field: 0 for field in fields}
    observed: Counter[str] = Counter()
    elapsed_total = 0.0
    elapsed_count = 0
    for row in rows:
        usage = row.get("usage") if isinstance(row, dict) else None
        if not isinstance(usage, dict):
            continue
        for field in fields:
            value = usage.get(field)
            if isinstance(value, (int, float)):
                totals[field] += value
                observed[field] += 1
        elapsed = usage.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            elapsed_total += float(elapsed)
            elapsed_count += 1
    for field in fields:
        if observed[field] == 0:
            totals[field] = None
    return {
        **totals,
        "elapsed_seconds_total": elapsed_total if elapsed_count else None,
        "elapsed_seconds_mean": elapsed_total / elapsed_count if elapsed_count else None,
        "rows_with_usage": max(observed.values()) if observed else 0,
        "rows_total": len(rows),
    }


def _usage_comparison(zero: dict[str, Any], few: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "estimated_cost_usd",
        "elapsed_seconds_total",
        "elapsed_seconds_mean",
    )
    return {
        field: _comparison_metric(zero_value=zero.get(field), few_value=few.get(field), source="run_manifest.usage")
        for field in fields
    }


def _comparison_key_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": result.get("task"),
        "representation": result.get("representation"),
        "context_bundle": result.get("context_bundle"),
        "track_mode": result.get("track_mode"),
    }


def _a_box_comparison(zero: dict[str, Any], few: dict[str, Any]) -> dict[str, Any]:
    zero_a = _group_metrics(zero, "by_track", "A_BOX") or zero.get("overall_metrics", {})
    few_a = _group_metrics(few, "by_track", "A_BOX") or few.get("overall_metrics", {})
    metrics = {
        "overall_a_box_accepted": _comparison_metric(
            zero_value=_metric_value(zero_a, "accepted_rate"),
            few_value=_metric_value(few_a, "accepted_rate"),
            source="evaluation_summary.by_track.A_BOX.accepted_rate",
        ),
        "typea_accepted": _comparison_metric(
            zero_value=_metric_value(_group_metrics(zero, "by_class", "TypeA"), "accepted_rate"),
            few_value=_metric_value(_group_metrics(few, "by_class", "TypeA"), "accepted_rate"),
            source="evaluation_summary.by_class.TypeA.accepted_rate",
        ),
        "typeb_accepted": _comparison_metric(
            zero_value=_metric_value(_group_metrics(zero, "by_class", "TypeB"), "accepted_rate"),
            few_value=_metric_value(_group_metrics(few, "by_class", "TypeB"), "accepted_rate"),
            source="evaluation_summary.by_class.TypeB.accepted_rate",
        ),
        "typec_accepted": _comparison_metric(
            zero_value=_metric_value(_group_metrics(zero, "by_class", "TypeC"), "accepted_rate"),
            few_value=_metric_value(_group_metrics(few, "by_class", "TypeC"), "accepted_rate"),
            source="evaluation_summary.by_class.TypeC.accepted_rate",
        ),
        "a_box_exact_value": _comparison_metric(
            zero_value=_metric_value(zero_a, "a_box_exact_value_match_rate"),
            few_value=_metric_value(few_a, "a_box_exact_value_match_rate"),
            source="evaluation_summary.by_track.A_BOX.a_box_exact_value_match_rate",
        ),
        "a_box_exact_action": _comparison_metric(
            zero_value=_metric_value(zero_a, "a_box_exact_action_match_rate"),
            few_value=_metric_value(few_a, "a_box_exact_action_match_rate"),
            source="evaluation_summary.by_track.A_BOX.a_box_exact_action_match_rate",
        ),
        "a_box_regression_pass": _comparison_metric(
            zero_value=_metric_value(zero_a, "a_box_regression_pass_rate"),
            few_value=_metric_value(few_a, "a_box_regression_pass_rate"),
            source="evaluation_summary.by_track.A_BOX.a_box_regression_pass_rate",
        ),
        "overdelete_rate": _comparison_metric(zero_value=None, few_value=None, source="not available in summary"),
        "empty_ops_rate": _comparison_metric(
            zero_value=_status_rate(zero, "non_executable_empty_ops"),
            few_value=_status_rate(few, "non_executable_empty_ops"),
            source="run_manifest.parse_status.non_executable_empty_ops",
        ),
        "constraint_type_qid_as_value_rate": _comparison_metric(
            zero_value=None,
            few_value=None,
            source="not available in summary",
        ),
        "parse_error_rate": _comparison_metric(
            zero_value=_parse_error_rate(zero),
            few_value=_parse_error_rate(few),
            source="evaluation_summary.parse_errors.proposal_parse_error_rate",
        ),
    }
    return {
        "comparison_key": _comparison_key_payload(few),
        "zero_shot_matrix_id": zero.get("matrix_id"),
        "few_shot_matrix_id": few.get("matrix_id"),
        "few_shot_policy": few.get("example_policy"),
        **_few_shot_condition(str(few.get("example_policy"))),
        "metrics": metrics,
        "token_cost_latency_overhead": _usage_comparison(_usage_totals(zero), _usage_totals(few)),
    }


def _tbox_metric_comparison(
    zero_metrics: dict[str, Any],
    few_metrics: dict[str, Any],
    key: str,
    source: str = "tbox_taxonomy_patch_metrics",
) -> dict[str, Any]:
    return _comparison_metric(
        zero_value=_metric_value(zero_metrics, key),
        few_value=_metric_value(few_metrics, key),
        source=f"{source}.{key}",
    )


def _load_tbox_taxonomy_summary(result: dict[str, Any]) -> dict[str, Any]:
    path = result.get("tbox_taxonomy_patch_evaluation_summary")
    if not isinstance(path, str) or not path:
        return {}
    summary_path = Path(path)
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _out_of_current_gold_operation_fp_rate(result: dict[str, Any]) -> float | None:
    summary = _load_tbox_taxonomy_summary(result)
    report = (
        summary.get("diagnostic_reports", {})
        .get("out_of_current_gold_operation_false_positive_rates", {})
    )
    value = report.get("overall_out_of_current_gold_false_positive_rate") if isinstance(report, dict) else None
    return value if isinstance(value, (int, float)) else None


def _t_box_comparison(zero: dict[str, Any], few: dict[str, Any]) -> dict[str, Any]:
    zero_metrics = zero.get("tbox_taxonomy_patch_metrics", {})
    few_metrics = few.get("tbox_taxonomy_patch_metrics", {})
    zero_metrics = zero_metrics if isinstance(zero_metrics, dict) else {}
    few_metrics = few_metrics if isinstance(few_metrics, dict) else {}
    metrics = {
        "family_level_success": _tbox_metric_comparison(zero_metrics, few_metrics, "tbox_patch_family_level_success"),
        "schema_decision_match": _tbox_metric_comparison(
            zero_metrics, few_metrics, "tbox_patch_schema_decision_match_rate"
        ),
        "taxonomy_code_match": _tbox_metric_comparison(
            zero_metrics, few_metrics, "tbox_patch_taxonomy_code_exact_match_rate"
        ),
        "taxonomy_level_success": _tbox_metric_comparison(
            zero_metrics, few_metrics, "tbox_patch_taxonomy_level_success"
        ),
        "constraint_family_f1": _tbox_metric_comparison(zero_metrics, few_metrics, "tbox_patch_constraint_family_f1"),
        "repair_op_f1": _tbox_metric_comparison(zero_metrics, few_metrics, "tbox_patch_repair_op_f1"),
        "value_delta_f1_when_applicable": _tbox_metric_comparison(
            zero_metrics,
            few_metrics,
            "tbox_patch_value_delta_f1_when_applicable",
        ),
        "value_delta_claimed_when_gold_absent": _tbox_metric_comparison(
            zero_metrics,
            few_metrics,
            "tbox_patch_value_delta_claimed_when_gold_absent_rate",
        ),
        "family_only_when_value_delta_gold_present": _tbox_metric_comparison(
            zero_metrics,
            few_metrics,
            "tbox_patch_family_only_when_value_delta_gold_present_rate",
        ),
        "out_of_current_gold_operation_false_positive_rate": _comparison_metric(
            zero_value=_out_of_current_gold_operation_fp_rate(zero),
            few_value=_out_of_current_gold_operation_fp_rate(few),
            source=(
                "tbox_taxonomy_patch_evaluation_summary.diagnostic_reports."
                "out_of_current_gold_operation_false_positive_rates"
            ),
        ),
        "parse_error_rate": _tbox_metric_comparison(zero_metrics, few_metrics, "tbox_patch_parse_error_rate"),
    }
    return {
        "comparison_key": _comparison_key_payload(few),
        "zero_shot_matrix_id": zero.get("matrix_id"),
        "few_shot_matrix_id": few.get("matrix_id"),
        "few_shot_policy": few.get("example_policy"),
        **_few_shot_condition(str(few.get("example_policy"))),
        "metrics": metrics,
        "token_cost_latency_overhead": _usage_comparison(_usage_totals(zero), _usage_totals(few)),
    }


def _diagnosis_by_matrix(diagnosis_report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not diagnosis_report:
        return {}
    matrices = diagnosis_report.get("matrices", [])
    return {
        matrix.get("matrix_id"): matrix
        for matrix in matrices
        if isinstance(matrix, dict) and isinstance(matrix.get("matrix_id"), str)
    }


def _diagnosis_comparison(
    zero: dict[str, Any],
    few: dict[str, Any],
    diagnosis_matrices: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    zero_metrics = diagnosis_matrices.get(str(zero.get("matrix_id")), {}).get("metrics", {})
    few_metrics = diagnosis_matrices.get(str(few.get("matrix_id")), {}).get("metrics", {})
    zero_metrics = zero_metrics if isinstance(zero_metrics, dict) else {}
    few_metrics = few_metrics if isinstance(few_metrics, dict) else {}
    return {
        "comparison_key": _comparison_key_payload(few),
        "zero_shot_matrix_id": zero.get("matrix_id"),
        "few_shot_matrix_id": few.get("matrix_id"),
        "few_shot_policy": few.get("example_policy"),
        **_few_shot_condition(str(few.get("example_policy"))),
        "metrics": {
            key: _comparison_metric(
                zero_value=_metric_value(zero_metrics, key),
                few_value=_metric_value(few_metrics, key),
                source=f"track_diagnosis_report.metrics.{key}",
            )
            for key in (
                "balanced_accuracy",
                "a_box_recall",
                "t_box_recall",
                "ambiguous_rate",
                "wrong_route_rate",
                "parse_error_rate",
            )
        },
        "token_cost_latency_overhead": _usage_comparison(_usage_totals(zero), _usage_totals(few)),
    }


def _few_shot_reports(
    *,
    summary: dict[str, Any],
    output_dir: Path,
    diagnosis_report: dict[str, Any] | None,
    zero_shot_baseline_summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    results = [result for result in summary.get("results", []) if isinstance(result, dict)]
    baseline_results = []
    if isinstance(zero_shot_baseline_summary, dict):
        baseline_results = [
            result for result in zero_shot_baseline_summary.get("results", []) if isinstance(result, dict)
        ]
    zero_by_key = {
        _result_key(result): result
        for result in [*baseline_results, *results]
        if result.get("example_policy") == "zero_shot"
    }
    diagnosis_matrices = _diagnosis_by_matrix(diagnosis_report)
    report: dict[str, Any] = {
        "manifest_type": "prompt_dev_few_shot_delta_vs_zero_shot",
        "manifest_version": PROMPT_DEV_VERSION,
        "created_at_utc": _utc_now(),
        "run_id": summary.get("run_id"),
        "provider": summary.get("provider"),
        "model": summary.get("model"),
        "zero_shot_baseline": {
            "source": (
                zero_shot_baseline_summary.get("_summary_path")
                or (
                    zero_shot_baseline_summary.get("outputs", {}).get("summary")
                    if isinstance(zero_shot_baseline_summary.get("outputs"), dict)
                    else None
                )
            )
            if isinstance(zero_shot_baseline_summary, dict)
            else None,
            "run_id": zero_shot_baseline_summary.get("run_id") if isinstance(zero_shot_baseline_summary, dict) else None,
            "manifest_version": zero_shot_baseline_summary.get("manifest_version")
            if isinstance(zero_shot_baseline_summary, dict)
            else None,
        },
        "comparison_rule": "Match few-shot matrices to zero-shot by task, representation, context bundle, and track mode.",
        "headline_policy": "No aggregate A-box/T-box headline is computed.",
        "sections": {"a_box": [], "t_box_taxonomy_patch": [], "diagnosis": []},
        "unmatched_few_shot_matrix_ids": [],
    }
    for few in results:
        policy = few.get("example_policy")
        if not isinstance(policy, str) or policy == "zero_shot":
            continue
        zero = zero_by_key.get(_result_key(few))
        if zero is None:
            report["unmatched_few_shot_matrix_ids"].append(few.get("matrix_id"))
            continue
        if few.get("task") == "track_diagnosis":
            report["sections"]["diagnosis"].append(_diagnosis_comparison(zero, few, diagnosis_matrices))
            continue
        track_counts = few.get("counts", {}).get("by_historical_track", {})
        if isinstance(track_counts, dict) and track_counts.get("A_BOX", 0):
            report["sections"]["a_box"].append(_a_box_comparison(zero, few))
        if isinstance(track_counts, dict) and track_counts.get("T_BOX", 0):
            report["sections"]["t_box_taxonomy_patch"].append(_t_box_comparison(zero, few))

    config = {
        "manifest_type": "prompt_dev_few_shot_run_config",
        "manifest_version": PROMPT_DEV_VERSION,
        "created_at_utc": _utc_now(),
        "run_id": summary.get("run_id"),
        "provider": summary.get("provider"),
        "model": summary.get("model"),
        "inputs": summary.get("inputs", {}),
        "few_shot_conditions": {
            policy: _few_shot_condition(policy)
            for policy in sorted(
                {
                    result.get("example_policy")
                    for result in results
                    if isinstance(result.get("example_policy"), str) and result.get("example_policy") != "zero_shot"
                }
            )
        },
        "artifact_role": (
            "paper-facing only for static_diverse_kshot comparisons; dynamic retrieval policies are exploratory."
        ),
        "zero_shot_baseline": report["zero_shot_baseline"],
        "matrix_usage_totals": {result["matrix_id"]: _usage_totals(result) for result in results if "matrix_id" in result},
    }
    write_json(output_dir / "few_shot_run_config.json", config)
    write_json(output_dir / "few_shot_delta_vs_zero_shot.json", report)
    (output_dir / "few_shot_delta_vs_zero_shot.md").write_text(_few_shot_delta_markdown(report), encoding="utf-8")
    return config, report


def _format_metric_cell(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "n/a"


def _few_shot_delta_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Few-Shot Delta vs Zero-Shot",
        "",
        f"Run id: `{report.get('run_id')}`",
        "",
        "A-box, T-box taxonomy-patch, and diagnosis metrics are reported separately. "
        "No combined A-box/T-box headline is computed.",
        "",
        "Static few-shot (`static_diverse_kshot`) is paper-facing. Dynamic retrieval policies are exploratory.",
        "",
        "Token, cost, and latency overhead are included per comparison from `run_manifest.jsonl` usage fields.",
        "",
    ]
    section_titles = {
        "a_box": "A-Box",
        "t_box_taxonomy_patch": "T-Box Taxonomy Patch",
        "diagnosis": "Diagnosis",
    }
    for section_key, title in section_titles.items():
        lines.extend(
            [
                f"## {title}",
                "",
                "| Few-shot matrix | Zero-shot matrix | Policy | Selection | Paper status | Metric | Zero-shot | Few-shot | Delta |",
                "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        comparisons = report.get("sections", {}).get(section_key, [])
        if not comparisons:
            lines.append("| n/a | n/a | n/a | n/a | n/a | No comparisons available | n/a | n/a | n/a |")
        for comparison in comparisons:
            for metric_name, metric in comparison.get("metrics", {}).items():
                lines.append(
                    " | ".join(
                        [
                            f"| `{comparison.get('few_shot_matrix_id')}`",
                            f"`{comparison.get('zero_shot_matrix_id')}`",
                            f"`{comparison.get('few_shot_policy')}`",
                            f"`{comparison.get('selection_type')}`",
                            comparison.get("paper_status", ""),
                            metric_name,
                            _format_metric_cell(metric.get("zero_shot")),
                            _format_metric_cell(metric.get("few_shot")),
                            _format_metric_cell(metric.get("delta")) + " |",
                        ]
                    )
                )
            overhead = comparison.get("token_cost_latency_overhead", {})
            for metric_name, metric in overhead.items():
                lines.append(
                    " | ".join(
                        [
                            f"| `{comparison.get('few_shot_matrix_id')}`",
                            f"`{comparison.get('zero_shot_matrix_id')}`",
                            f"`{comparison.get('few_shot_policy')}`",
                            f"`{comparison.get('selection_type')}`",
                            comparison.get("paper_status", ""),
                            f"overhead_{metric_name}",
                            _format_metric_cell(metric.get("zero_shot")),
                            _format_metric_cell(metric.get("few_shot")),
                            _format_metric_cell(metric.get("delta")) + " |",
                        ]
                    )
                )
        lines.append("")
    if report.get("unmatched_few_shot_matrix_ids"):
        lines.extend(["## Unmatched Few-Shot Matrices", ""])
        for matrix_id in report["unmatched_few_shot_matrix_ids"]:
            lines.append(f"- `{matrix_id}`")
        lines.append("")
    return "\n".join(lines)


def _format_rate(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "n/a"


def _track_diagnosis_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track Diagnosis Report",
        "",
        "This report is dev-only and gates whether diagnosis-routed repair canaries are interpretable.",
        "",
        "Acceptance gates:",
        (
            f"- Request error rate <= {DIAGNOSIS_ACCEPTANCE_GATES['request_error_rate_max']:.2%}; "
            f"parse error rate <= {DIAGNOSIS_ACCEPTANCE_GATES['parse_error_rate_max']:.2%}; "
            f"balanced accuracy >= {DIAGNOSIS_ACCEPTANCE_GATES['balanced_accuracy_min']:.2f}; "
            f"A_BOX/T_BOX recall >= {DIAGNOSIS_ACCEPTANCE_GATES['a_box_recall_min']:.2f}/"
            f"{DIAGNOSIS_ACCEPTANCE_GATES['t_box_recall_min']:.2f}; "
            f"AMBIGUOUS rate <= {DIAGNOSIS_ACCEPTANCE_GATES['ambiguous_rate_max']:.2%}."
        ),
        "",
        (
            "| Matrix | Context | Track mode | Total | A recall | T recall | Balanced acc | Macro-F1 | "
            "Ambiguous | Wrong-route | Parse err | Request err | Gate |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for matrix in report["matrices"]:
        metrics = matrix["metrics"]
        lines.append(
            " | ".join(
                [
                    f"| `{matrix['matrix_id']}`",
                    f"`{matrix.get('context_bundle')}`",
                    f"`{matrix.get('track_mode') or ''}`",
                    str(matrix["counts"]["total"]),
                    _format_rate(metrics.get("a_box_recall")),
                    _format_rate(metrics.get("t_box_recall")),
                    _format_rate(metrics.get("balanced_accuracy")),
                    _format_rate(metrics.get("macro_f1")),
                    _format_rate(metrics.get("ambiguous_rate")),
                    _format_rate(metrics.get("wrong_route_rate")),
                    _format_rate(metrics.get("parse_error_rate")),
                    _format_rate(metrics.get("request_error_rate")),
                    ("PASS" if matrix["eligible_for_routed_canary"] else "FAIL") + " |",
                ]
            )
        )
    lines.extend(["", "## Confusion Matrices", ""])
    for matrix in report["matrices"]:
        lines.extend(
            [
                f"### `{matrix['matrix_id']}`",
                "",
                "```json",
                json.dumps(matrix["confusion_by_historical_track"], indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _metric_text(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return ""


def _rate_metric_text(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if not isinstance(value, dict):
        return "n/a"
    rate = value.get("rate")
    if isinstance(rate, (int, float)):
        return f"{rate:.3f}"
    return "n/a"


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
            "Parse errors | Request errors | Strict functional | Track acc | Strict audit | "
            "T-box family | T-box decision | T-box taxonomy | T-box value F1 |"
        ),
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in summary["results"]:
        metrics = result.get("overall_metrics") if isinstance(result.get("overall_metrics"), dict) else {}
        tbox_metrics = (
            result.get("tbox_taxonomy_patch_metrics")
            if isinstance(result.get("tbox_taxonomy_patch_metrics"), dict)
            else {}
        )
        task = result.get("task")
        parse_errors = (result.get("parse_errors") or {}).get("proposal_parse_error_count", 0)
        request_errors = (result.get("request_errors") or {}).get("proposal_request_error_count", 0)
        request_errors += (result.get("request_errors") or {}).get("track_diagnosis_request_error_count", 0)
        functional_text = "n/a" if task == "track_diagnosis" else _metric_text(metrics, "functional_success_rate")
        track_accuracy_text = (
            _metric_text(metrics, "track_diagnosis_accuracy") if task == "track_diagnosis" else "n/a"
        )
        audit_text = "n/a" if task == "track_diagnosis" else _metric_text(metrics, "auditability_complete_rate")
        tbox_family_text = _rate_metric_text(tbox_metrics, "tbox_patch_family_level_success")
        tbox_decision_text = _rate_metric_text(tbox_metrics, "tbox_patch_decision_level_success")
        tbox_taxonomy_text = _rate_metric_text(tbox_metrics, "tbox_patch_taxonomy_level_success")
        tbox_value_f1_text = _rate_metric_text(tbox_metrics, "tbox_patch_value_delta_f1_when_applicable")
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
                    audit_text,
                    tbox_family_text,
                    tbox_decision_text,
                    tbox_taxonomy_text,
                    tbox_value_f1_text + " |",
                ]
            )
        )
    lines.append("")
    return "\n".join(lines)
