"""Shared confirmatory T-box taxonomy-gold preparation and evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from guardian.evaluator import write_json
from guardian.tbox_taxonomy_patch_evaluator import (
    evaluate_tbox_taxonomy_patch_predictions,
    load_jsonl,
)
from lib.tbox_taxonomy_patch_gold import gold_patch_for_record, is_tbox_record, summarize_patches
from lib.utils import iter_jsonl

TBOX_TAXONOMY_GOLD_VERSION = 1
UNSUPPORTED_CONFIRMATORY_REPAIR_OPS = frozenset({"CLASS_HIERARCHY_ADD", "EXCEPTION_ADD"})


@dataclass
class PreparedTBoxTaxonomyGold:
    patches: list[dict[str, Any]]
    case_annotations: dict[str, dict[str, Any]]
    tbox_case_ids: list[str]
    summary: dict[str, Any]
    gold_version: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selection_annotations(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    annotations = payload.get("case_annotations") if isinstance(payload, dict) else None
    if not isinstance(annotations, dict):
        return {}
    return {
        str(case_id): annotation
        for case_id, annotation in annotations.items()
        if isinstance(annotation, dict)
    }


def taxonomy_gold_eligibility(
    record: dict[str, Any], *, annotation: dict[str, Any] | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    """Return mechanically supported taxonomy gold or an exclusion reason."""
    patch = gold_patch_for_record(record, annotation=annotation)
    if patch is None:
        return None, "taxonomy_gold_unextractable"
    repair_ops = {
        repair.get("repair_op")
        for repair in patch.get("repairs", [])
        if isinstance(repair, dict) and isinstance(repair.get("repair_op"), str)
    }
    unsupported = sorted(repair_ops & UNSUPPORTED_CONFIRMATORY_REPAIR_OPS)
    if unsupported:
        return None, "unsupported_taxonomy_operations:" + ",".join(unsupported)
    return patch, None


def prepare_tbox_taxonomy_gold(
    *,
    classified_path: str | Path,
    selected_case_ids: Iterable[str],
    selection_manifest_path: str | Path | None = None,
    require_complete: bool = True,
) -> PreparedTBoxTaxonomyGold:
    selected = [case_id for case_id in selected_case_ids if isinstance(case_id, str) and case_id]
    selected_set = set(selected)
    annotations = _selection_annotations(selection_manifest_path)
    patches: list[dict[str, Any]] = []
    tbox_case_ids: list[str] = []
    unsupported: dict[str, str] = {}
    seen: set[str] = set()
    for record in iter_jsonl(classified_path):
        case_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(case_id, str) or case_id not in selected_set:
            continue
        if case_id in seen:
            raise ValueError(f"Duplicate selected benchmark case while preparing taxonomy gold: {case_id}")
        seen.add(case_id)
        if not is_tbox_record(record):
            continue
        tbox_case_ids.append(case_id)
        patch, reason = taxonomy_gold_eligibility(record, annotation=annotations.get(case_id))
        if patch is None:
            unsupported[case_id] = reason or "taxonomy_gold_unextractable"
        else:
            patches.append(patch)
    missing_selected = sorted(selected_set - seen)
    if missing_selected:
        preview = ", ".join(missing_selected[:10])
        raise ValueError(
            f"Classified benchmark is missing {len(missing_selected)} selected cases while preparing taxonomy gold: "
            f"{preview}"
        )
    summary = summarize_patches(
        patches,
        selected_records=len(selected),
        selected_tbox_records=len(tbox_case_ids),
        unsupported_case_ids=sorted(unsupported),
    )
    summary["unsupported_reasons"] = dict(sorted(unsupported.items()))
    summary["mechanically_supported_gold_complete"] = not unsupported
    summary["unsupported_confirmatory_repair_ops"] = sorted(UNSUPPORTED_CONFIRMATORY_REPAIR_OPS)
    if require_complete and unsupported:
        preview = ", ".join(f"{case_id} ({reason})" for case_id, reason in list(sorted(unsupported.items()))[:10])
        raise ValueError(
            "Confirmatory T-box selection lacks complete mechanically supported taxonomy gold: "
            f"{len(unsupported)} unsupported case(s); first: {preview}"
        )
    patches.sort(key=lambda patch: str(patch.get("case_id", "")))
    tbox_case_ids.sort()
    classified = Path(classified_path).resolve()
    gold_identity = {
        "gold_spec_version": TBOX_TAXONOMY_GOLD_VERSION,
        "classified_sha256": _sha256_file(classified),
        "selected_case_ids": sorted(selected_set),
        "patches_sha256": _canonical_sha256(patches),
    }
    gold_version = f"tbox_taxonomy_patch_gold_{_canonical_sha256(gold_identity)[:16]}_v1"
    summary["gold_version"] = gold_version
    summary["gold_identity"] = gold_identity
    return PreparedTBoxTaxonomyGold(
        patches=patches,
        case_annotations=annotations,
        tbox_case_ids=tbox_case_ids,
        summary=summary,
        gold_version=gold_version,
    )


def evaluate_tbox_taxonomy_patch_bundle(
    *,
    prepared_gold: PreparedTBoxTaxonomyGold,
    predictions_path: str | Path,
    out_traces_path: str | Path,
    out_summary_path: str | Path,
) -> dict[str, Any]:
    prediction_path = Path(predictions_path)
    prediction_rows = load_jsonl(prediction_path) if prediction_path.is_file() else []
    prediction_ids: list[str] = []
    for row in prediction_rows:
        case_id = row.get("case_id") if isinstance(row, dict) else None
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Taxonomy prediction lacks a case_id in {prediction_path}.")
        prediction_ids.append(case_id)
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError(f"Duplicate taxonomy prediction case IDs in {prediction_path}.")
    unexpected = sorted(set(prediction_ids) - set(prepared_gold.tbox_case_ids))
    if unexpected:
        raise ValueError(
            f"Taxonomy predictions contain {len(unexpected)} unselected case IDs; first: {', '.join(unexpected[:10])}"
        )
    result = evaluate_tbox_taxonomy_patch_predictions(
        gold_rows=prepared_gold.patches,
        prediction_rows=prediction_rows,
        case_annotations=prepared_gold.case_annotations,
        gold_version=prepared_gold.gold_version,
    )
    traces = result.pop("traces")
    traces_path = Path(out_traces_path)
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    with traces_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
    result["gold_extraction"] = prepared_gold.summary
    result["inputs"] = {
        "predictions": str(prediction_path.resolve()),
        "prediction_sha256": _sha256_file(prediction_path) if prediction_path.is_file() else None,
        "traces": str(traces_path.resolve()),
    }
    write_json(out_summary_path, result)
    return result
