#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from lib.benchmark_selection import derive_case_metadata, group_key_for_record, load_selection_manifest
from lib.tbox_taxonomy_patch_gold import gold_patch_for_record
from lib.utils import iter_jsonl

A_BOX_ROLES: tuple[tuple[str, str, Callable[[dict[str, Any], dict[str, Any] | None], bool]], ...] = (
    (
        "a_box_clean_rule",
        "clean rule / constraint rejection / target-required claim",
        lambda record, _gold: _subtype(record)
        in {"SET_MEMBERSHIP_REJECTION", "SELF_LINK_REJECTION", "TARGET_REQUIRED_CLAIM"},
    ),
    (
        "a_box_format_or_literal_normalization",
        "format or literal normalization/pruning",
        lambda record, _gold: _subtype(record)
        in {
            "FORMAT_NORMALIZATION",
            "FORMAT_VALUE_PRUNING",
            "MULTIPLICITY_NORMALIZATION",
            "REJECTION_FORMAT_INVALID",
        },
    ),
    (
        "a_box_local_evidence",
        "local-evidence case",
        lambda record, _gold: _class_name(record) == "TypeB" or _subtype(record).startswith("LOCAL_"),
    ),
)

T_BOX_ROLES: tuple[tuple[str, str, Callable[[dict[str, Any], dict[str, Any] | None], bool]], ...] = (
    (
        "tbox_taxonomy_cq_plus",
        "CAUSAL_SCHEMA_REPAIR + CQ_PLUS / CONSTRAINT_QUALIFIER_ADD",
        lambda _record, gold: _schema_decision(gold) == "CAUSAL_SCHEMA_REPAIR"
        and _has_taxonomy_code(gold, {"CQ_PLUS"}),
    ),
    (
        "tbox_taxonomy_cq_minus_or_replace",
        "CAUSAL_SCHEMA_REPAIR + CQ_MINUS or CQ_REPLACE",
        lambda _record, gold: _schema_decision(gold) == "CAUSAL_SCHEMA_REPAIR"
        and _has_taxonomy_code(gold, {"CQ_MINUS", "CQ_REPLACE"}),
    ),
    (
        "tbox_taxonomy_no_causal_empty",
        "NO_CAUSAL_SCHEMA_REPAIR with repairs=[]",
        lambda _record, gold: _schema_decision(gold) == "NO_CAUSAL_SCHEMA_REPAIR" and not _repairs(gold),
    ),
    (
        "tbox_taxonomy_other_or_family_only",
        "OTHER_TBOX_UPDATE or FAMILY_ONLY / OPERATION_VISIBLE case",
        lambda _record, gold: _has_taxonomy_code(gold, {"OTHER"}) or _has_evidence_level(gold, {"FAMILY_ONLY"}),
    ),
)

DIAGNOSIS_ROLES: tuple[tuple[str, str, Callable[[dict[str, Any], dict[str, Any] | None], bool]], ...] = (
    ("diagnosis_a_box", "A_BOX diagnosis example", lambda record, _gold: record.get("track") == "A_BOX"),
    ("diagnosis_t_box", "T_BOX diagnosis example", lambda record, _gold: record.get("track") == "T_BOX"),
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_hash(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _class_name(record: dict[str, Any]) -> str:
    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    value = classification.get("class")
    return value if isinstance(value, str) else ""


def _subtype(record: dict[str, Any]) -> str:
    classification = record.get("classification") if isinstance(record.get("classification"), dict) else {}
    value = classification.get("subtype")
    return value if isinstance(value, str) else ""


def _repairs(gold: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(gold, dict) or not isinstance(gold.get("repairs"), list):
        return []
    return [repair for repair in gold["repairs"] if isinstance(repair, dict)]


def _schema_decision(gold: dict[str, Any] | None) -> str:
    if not isinstance(gold, dict):
        return ""
    value = gold.get("schema_decision")
    return value if isinstance(value, str) else ""


def _has_taxonomy_code(gold: dict[str, Any] | None, values: set[str]) -> bool:
    return any(repair.get("taxonomy_code") in values for repair in _repairs(gold))


def _has_evidence_level(gold: dict[str, Any] | None, values: set[str]) -> bool:
    return any(repair.get("evidence_level") in values for repair in _repairs(gold))


def _load_records_by_id(classified_benchmark: Path, case_ids: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    remaining = set(case_ids)
    for record in iter_jsonl(classified_benchmark):
        if not remaining or not isinstance(record, dict):
            continue
        case_id = record.get("id")
        if isinstance(case_id, str) and case_id in remaining:
            records[case_id] = record
            remaining.remove(case_id)
    return records


def _records_for_manifest(manifest: dict[str, Any], records_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        records_by_id[case_id]
        for case_id in manifest.get("selected_case_ids", [])
        if isinstance(case_id, str) and case_id in records_by_id
    ]


def _manifest_annotation(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    annotations = manifest.get("case_annotations") if isinstance(manifest.get("case_annotations"), dict) else {}
    annotation = annotations.get(case_id)
    return annotation if isinstance(annotation, dict) else {}


def _record_tbox_key(record: dict[str, Any], manifest: dict[str, Any]) -> str | None:
    annotation = _manifest_annotation(manifest, str(record.get("id") or ""))
    value = annotation.get("tbox_revision_key")
    if isinstance(value, str) and value:
        return value
    _, tbox_key, _ = group_key_for_record(record)
    return tbox_key


def _record_group_key(record: dict[str, Any], manifest: dict[str, Any]) -> str:
    annotation = _manifest_annotation(manifest, str(record.get("id") or ""))
    value = annotation.get("group_key")
    if isinstance(value, str) and value:
        return value
    group_key, _, _ = group_key_for_record(record)
    return group_key


def _blocked_sets(core_records: list[dict[str, Any]], core_manifest: dict[str, Any]) -> dict[str, set[str]]:
    return {
        "case_ids": {
            str(case_id)
            for case_id in core_manifest.get("selected_case_ids", [])
            if isinstance(case_id, str)
        },
        "qids": {record["qid"] for record in core_records if isinstance(record.get("qid"), str)},
        "properties": {record["property"] for record in core_records if isinstance(record.get("property"), str)},
        "tbox_revision_keys": {
            key for record in core_records if isinstance((key := _record_tbox_key(record, core_manifest)), str)
        },
    }


def _candidate_rows(
    *,
    records: list[dict[str, Any]],
    dev_manifest: dict[str, Any],
    core_blocks: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        case_id = record.get("id")
        if not isinstance(case_id, str) or case_id in core_blocks["case_ids"]:
            continue
        tbox_key = _record_tbox_key(record, dev_manifest)
        if isinstance(tbox_key, str) and tbox_key in core_blocks["tbox_revision_keys"]:
            continue
        annotation = _manifest_annotation(dev_manifest, case_id)
        metadata = derive_case_metadata(record, tier="dev") or {}
        gold = (
            gold_patch_for_record(record, annotation=annotation)
            if record.get("track") == "T_BOX"
            else None
        )
        rows.append(
            {
                "record": record,
                "case_id": case_id,
                "track": record.get("track"),
                "class": metadata.get("class") or _class_name(record),
                "subtype": metadata.get("subtype") or _subtype(record),
                "property": record.get("property"),
                "qid": record.get("qid"),
                "group_key": _record_group_key(record, dev_manifest),
                "tbox_revision_key": tbox_key,
                "gold": gold,
            }
        )
    return rows


def _select_for_roles(
    *,
    rows: list[dict[str, Any]],
    roles: tuple[tuple[str, str, Callable[[dict[str, Any], dict[str, Any] | None], bool]], ...],
    seed: int,
    visible_prefix: str,
    task_schema: str,
    gold_version: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    warnings: list[str] = []
    used_case_ids: set[str] = set()
    used_qids: set[str] = set()
    used_properties: set[str] = set()
    used_tbox_keys: set[str] = set()

    for index, (role, role_description, predicate) in enumerate(roles, start=1):
        matches = [
            row
            for row in rows
            if row["case_id"] not in used_case_ids
            and row.get("tbox_revision_key") not in used_tbox_keys
            and predicate(row["record"], row.get("gold"))
        ]
        if not matches:
            warnings.append(f"No support candidate found for role {role}.")
            continue
        matches.sort(
            key=lambda row: (
                1 if row.get("qid") in used_qids else 0,
                1 if row.get("property") in used_properties else 0,
                _stable_hash(seed, role, row["case_id"]),
                row["case_id"],
            )
        )
        chosen = matches[0]
        used_case_ids.add(chosen["case_id"])
        if isinstance(chosen.get("qid"), str):
            used_qids.add(chosen["qid"])
        if isinstance(chosen.get("property"), str):
            used_properties.add(chosen["property"])
        if isinstance(chosen.get("tbox_revision_key"), str):
            used_tbox_keys.add(chosen["tbox_revision_key"])
        example = {
            "raw_case_id": chosen["case_id"],
            "visible_example_id": f"{visible_prefix}_{index:06d}",
            "role": role,
            "task_schema": task_schema,
            "notes": role_description,
        }
        if gold_version is not None:
            example["gold_version"] = gold_version
        selected.append(example)
    return selected, warnings


def _selected_records_by_case_id(
    rows: list[dict[str, Any]],
    support_sets: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    by_id = {row["case_id"]: row for row in rows}
    selected_ids = {
        example["raw_case_id"]
        for examples in support_sets.values()
        for example in examples
        if isinstance(example.get("raw_case_id"), str)
    }
    return {case_id: by_id[case_id] for case_id in sorted(selected_ids) if case_id in by_id}


def _taxonomy_codes(row: dict[str, Any]) -> list[str]:
    gold = row.get("gold")
    if not isinstance(gold, dict):
        return []
    repairs = _repairs(gold)
    if not repairs:
        return ["__EMPTY_REPAIRS__"]
    return [str(repair.get("taxonomy_code") or "unknown") for repair in repairs]


def _selection_counts(selected_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    taxonomy = Counter()
    for row in selected_rows.values():
        taxonomy.update(_taxonomy_codes(row))
    return {
        "selected_examples": len(selected_rows),
        "by_track": dict(Counter(str(row.get("track") or "unknown") for row in selected_rows.values())),
        "by_class_subtype": dict(
            Counter(
                f"{row.get('class') or 'unknown'}:{row.get('subtype') or 'unknown'}"
                for row in selected_rows.values()
            )
        ),
        "by_tbox_taxonomy_code": dict(taxonomy),
        "by_property": dict(Counter(str(row.get("property") or "unknown") for row in selected_rows.values())),
        "by_qid": dict(Counter(str(row.get("qid") or "unknown") for row in selected_rows.values())),
        "by_tbox_revision_key": dict(
            Counter(str(row.get("tbox_revision_key") or "none") for row in selected_rows.values())
        ),
    }


def _overlap_report(
    *,
    selected_rows: dict[str, dict[str, Any]],
    core_blocks: dict[str, set[str]],
) -> dict[str, Any]:
    selected_case_ids = set(selected_rows)
    selected_qids = {row["qid"] for row in selected_rows.values() if isinstance(row.get("qid"), str)}
    selected_properties = {
        row["property"] for row in selected_rows.values() if isinstance(row.get("property"), str)
    }
    selected_tbox_keys = {
        row["tbox_revision_key"]
        for row in selected_rows.values()
        if isinstance(row.get("tbox_revision_key"), str)
    }
    shared_properties = sorted(selected_properties & core_blocks["properties"])
    return {
        "core_case_overlap": len(selected_case_ids & core_blocks["case_ids"]),
        "core_qid_overlap": len(selected_qids & core_blocks["qids"]),
        "core_tbox_revision_overlap": len(selected_tbox_keys & core_blocks["tbox_revision_keys"]),
        "core_property_overlap": len(shared_properties),
        "core_property_overlap_values": shared_properties,
        "shares_property_with_core": bool(shared_properties),
    }


def build_support_manifest(
    *,
    classified_benchmark: Path,
    dev_manifest_path: Path,
    core_manifest_path: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dev_manifest = load_selection_manifest(dev_manifest_path)
    core_manifest = load_selection_manifest(core_manifest_path)
    requested_ids = {
        case_id
        for manifest in (dev_manifest, core_manifest)
        for case_id in manifest.get("selected_case_ids", [])
        if isinstance(case_id, str)
    }
    records_by_id = _load_records_by_id(classified_benchmark, requested_ids)
    dev_records = _records_for_manifest(dev_manifest, records_by_id)
    core_records = _records_for_manifest(core_manifest, records_by_id)
    core_blocks = _blocked_sets(core_records, core_manifest)
    rows = _candidate_rows(records=dev_records, dev_manifest=dev_manifest, core_blocks=core_blocks)
    a_box_rows = [row for row in rows if row.get("track") == "A_BOX"]
    t_box_rows = [row for row in rows if row.get("track") == "T_BOX" and isinstance(row.get("gold"), dict)]

    a_box_support, a_warnings = _select_for_roles(
        rows=a_box_rows,
        roles=A_BOX_ROLES,
        seed=seed,
        visible_prefix="example_a",
        task_schema="a_box_v4_spec_only",
    )
    t_box_support, t_warnings = _select_for_roles(
        rows=t_box_rows,
        roles=T_BOX_ROLES,
        seed=seed,
        visible_prefix="example_t",
        task_schema="tbox_taxonomy_patch_v1",
        gold_version="tbox_taxonomy_patch_gold_dev_v1",
    )
    diagnosis_support, d_warnings = _select_for_roles(
        rows=rows,
        roles=DIAGNOSIS_ROLES,
        seed=seed,
        visible_prefix="example_d",
        task_schema="track_diagnosis_v1",
    )
    support_sets = {
        "a_box_repair": a_box_support,
        "t_box_repair": t_box_support,
        "track_diagnosis": diagnosis_support,
    }
    selected_rows = _selected_records_by_case_id(rows, support_sets)
    overlaps = _overlap_report(selected_rows=selected_rows, core_blocks=core_blocks)
    manifest = {
        "manifest_type": "few_shot_support_set",
        "manifest_version": "static_support_v1",
        "created_at_utc": _utc_now(),
        "source_manifest": str(dev_manifest_path),
        "blocked_manifest": str(core_manifest_path),
        "selection_policy": "static_diverse",
        "support_sets": support_sets,
        "blocked_overlaps": {
            "core_case_overlap": overlaps["core_case_overlap"],
            "core_qid_overlap": overlaps["core_qid_overlap"],
            "core_tbox_revision_overlap": overlaps["core_tbox_revision_overlap"],
        },
    }
    report = {
        "manifest_type": "few_shot_support_selection_report",
        "manifest_version": "static_support_v1",
        "created_at_utc": manifest["created_at_utc"],
        "inputs": {
            "classified_benchmark": str(classified_benchmark),
            "dev_manifest": str(dev_manifest_path),
            "core_manifest": str(core_manifest_path),
            "seed": seed,
        },
        "counts": {
            "dev_records": len(dev_records),
            "eligible_records": len(rows),
            "eligible_a_box_records": len(a_box_rows),
            "eligible_t_box_records": len(t_box_rows),
            **_selection_counts(selected_rows),
        },
        "overlaps": overlaps,
        "warnings": a_warnings + t_warnings + d_warnings,
        "selected_examples": {
            case_id: {
                "track": row.get("track"),
                "class": row.get("class"),
                "subtype": row.get("subtype"),
                "property": row.get("property"),
                "qid": row.get("qid"),
                "tbox_revision_key": row.get("tbox_revision_key"),
                "taxonomy_codes": _taxonomy_codes(row),
            }
            for case_id, row in selected_rows.items()
        },
    }
    return manifest, report


def _validate_manifest(manifest: dict[str, Any], schema_path: Path) -> None:
    with schema_path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = Draft202012Validator(schema)
    validator.validate(manifest)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown_report(manifest: dict[str, Any], report: dict[str, Any]) -> str:
    lines = [
        "# Few-Shot Static Support Selection",
        "",
        f"Created: `{manifest['created_at_utc']}`",
        f"Source manifest: `{manifest['source_manifest']}`",
        f"Blocked manifest: `{manifest['blocked_manifest']}`",
        f"Selection policy: `{manifest['selection_policy']}`",
        "",
        "## Support Sets",
        "",
        "| Task | Visible example | Raw case id | Role |",
        "| --- | --- | --- | --- |",
    ]
    for task, examples in manifest["support_sets"].items():
        for example in examples:
            lines.append(
                f"| `{task}` | `{example['visible_example_id']}` | "
                f"`{example['raw_case_id']}` | `{example['role']}` |"
            )
    lines.extend(["", "## Counts", ""])
    for key, value in report["counts"].items():
        if isinstance(value, dict):
            lines.extend([f"### {key}", ""])
            for item, count in sorted(value.items()):
                lines.append(f"- `{item}`: {count}")
            lines.append("")
        else:
            lines.append(f"- `{key}`: {value}")
    overlaps = report["overlaps"]
    lines.extend(
        [
            "",
            "## Overlap Audit",
            "",
            f"- Core case overlap: `{overlaps['core_case_overlap']}`",
            f"- Core QID overlap: `{overlaps['core_qid_overlap']}`",
            f"- Core T-box revision overlap: `{overlaps['core_tbox_revision_overlap']}`",
            f"- Core property overlap: `{overlaps['core_property_overlap']}`",
            (
                "- Static support examples share properties with core cases: "
                f"`{overlaps['shares_property_with_core']}`"
            ),
        ]
    )
    if overlaps["core_property_overlap_values"]:
        values = ", ".join(f"`{value}`" for value in overlaps["core_property_overlap_values"])
        lines.append(f"- Shared core properties: {values}")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select static few-shot support examples from the dev manifest.")
    parser.add_argument("--classified-benchmark", default="data/04_classified_benchmark.jsonl")
    parser.add_argument("--dev-manifest", default="reports/benchmark_selection/dev_prompt_v1_seed_13.json")
    parser.add_argument("--core-manifest", default="reports/benchmark_selection/core_v1_seed_13.json")
    parser.add_argument("--output-dir", default="reports/prompt_dev/few_shot/static_support_v1")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--schema", default="schemas/few_shot_support_set.schema.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest, report = build_support_manifest(
        classified_benchmark=Path(args.classified_benchmark),
        dev_manifest_path=Path(args.dev_manifest),
        core_manifest_path=Path(args.core_manifest),
        seed=args.seed,
    )
    _validate_manifest(manifest, Path(args.schema))
    write_json(output_dir / "static_support_manifest.json", manifest)
    write_json(output_dir / "support_selection_report.json", report)
    (output_dir / "support_selection_report.md").write_text(
        _markdown_report(manifest, report),
        encoding="utf-8",
    )
    print(f"[done] wrote {output_dir / 'static_support_manifest.json'}")
    print(f"[done] wrote {output_dir / 'support_selection_report.json'}")
    print(f"[done] wrote {output_dir / 'support_selection_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
