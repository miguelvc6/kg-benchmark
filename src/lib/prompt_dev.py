from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from classifier import WorldStateStore
from guardian.reasoning import ABLATION_BUNDLES, _bundle_payload_and_audit
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


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_hash(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _ordered_csv(values: Iterable[str] | None, default: Iterable[str]) -> tuple[str, ...]:
    result = tuple(value.strip() for value in values or default if isinstance(value, str) and value.strip())
    return result or tuple(default)


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
) -> list[dict[str, Any]]:
    manifest = load_selection_manifest(manifest_path)
    ids = [case_id for case_id in manifest.get("selected_case_ids", []) if isinstance(case_id, str)]
    if max_cases is not None:
        ids = ids[: max(0, max_cases)]
    id_set = set(ids)
    by_id = {
        record["id"]: record
        for record in iter_jsonl(classified_path)
        if isinstance(record, dict) and isinstance(record.get("id"), str) and record["id"] in id_set
    }
    return [by_id[case_id] for case_id in ids if case_id in by_id]


def _load_all_manifest_records(classified_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    return _load_manifest_records(classified_path, manifest_path, max_cases=None)


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
        "provenance": [{"kind": "HISTORY", "snippet": "dev example historical repair target"}],
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
        "provenance": [{"kind": "HISTORY", "snippet": "dev example historical property revision"}],
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


def render_prompt_dev_prompts(options: PromptDevRenderOptions) -> dict[str, Any]:
    eval_records = _load_manifest_records(
        options.classified_benchmark,
        options.dev_manifest,
        max_cases=options.max_cases,
    )
    candidate_records = _load_all_manifest_records(options.classified_benchmark, options.dev_manifest)
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
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / "prompt_dev_rendered_prompts.jsonl"
    summary_path = output_dir / "prompt_dev_render_summary.json"
    review_path = output_dir / "prompt_dev_prompt_review.md"

    rendered_count = 0
    skipped_count = 0
    prompt_records: list[dict[str, Any]] = []
    log = logging.getLogger("prompt_dev")
    with WorldStateStore(options.world_state, log) as world_store:
        for record in eval_records:
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
                    candidate_world = world_store.get(candidate["id"])
                    example_payload, _ = _bundle_payload_and_audit(candidate, candidate_world, row["context_bundle"])
                    example["input_payload"] = example_payload
                rendered = render_prompt_dev_prompt(
                    task=task,
                    representation=row["representation"],
                    case_payload=case_payload,
                    examples=examples,
                    include_abstention=row["include_abstention"],
                )
                prompt_record = {
                    "matrix_id": row["matrix_id"],
                    "case_id": record["id"],
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
        },
        "note": "No LLM inference was run. These artifacts contain prompts only.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review_path.write_text(_prompt_review_markdown(prompt_records, summary), encoding="utf-8")
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
                f"## {record['matrix_id']} / {record['case_id']}",
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
