from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_TBOX_CAP_PER_UPDATE = 100
DEFAULT_SELECTION_SEED = 13
TRACK_TBOX_MARKER = '"track": "T_BOX"'
CASE_ID_RE = re.compile(r'"id"\s*:\s*"([^"]+)"')
PROPERTY_REVISION_RE = re.compile(r'"property_revision_id"\s*:\s*(\d+)')


def _normalized_case_ids(case_ids: Optional[Iterable[str]]) -> Optional[set[str]]:
    if not case_ids:
        return None
    return {case_id for case_id in case_ids if isinstance(case_id, str) and case_id}


def _stable_rank(case_id: str, property_revision_id: int, seed: int) -> str:
    payload = f"{seed}|{property_revision_id}|{case_id}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def build_selection_manifest(
    classified_path: str | Path,
    *,
    tbox_cap_per_update: int = DEFAULT_TBOX_CAP_PER_UPDATE,
    seed: int = DEFAULT_SELECTION_SEED,
    progress_every: int = 0,
) -> dict[str, Any]:
    if tbox_cap_per_update < 0:
        raise ValueError("tbox_cap_per_update must be non-negative.")

    selected_a_box_ids: list[str] = []
    tbox_candidates: dict[int, list[tuple[str, str]]] = {}
    tbox_available_counts: Counter[int] = Counter()
    total_records = 0
    total_a_box = 0
    total_t_box = 0

    with Path(classified_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if progress_every > 0 and line_number % progress_every == 0:
                print(
                    f"[progress] lines={line_number:,} "
                    f"selected_a_box={len(selected_a_box_ids):,} "
                    f"tbox_cases={total_t_box:,} "
                    f"distinct_tbox_updates={len(tbox_candidates):,}"
                )

            case_id_match = CASE_ID_RE.search(line)
            if not case_id_match:
                continue
            case_id = case_id_match.group(1)
            if not case_id:
                continue

            total_records += 1
            if TRACK_TBOX_MARKER not in line:
                total_a_box += 1
                selected_a_box_ids.append(case_id)
                continue

            total_t_box += 1
            revision_match = PROPERTY_REVISION_RE.search(line)
            if not revision_match:
                raise ValueError(f"T-BOX record {case_id} is missing an integer property_revision_id.")
            revision_id = int(revision_match.group(1))
            tbox_available_counts[revision_id] += 1
            tbox_candidates.setdefault(revision_id, []).append((_stable_rank(case_id, revision_id, seed), case_id))

    selected_t_box_ids: list[str] = []
    selected_counts_by_revision: list[dict[str, int]] = []
    for revision_id in sorted(tbox_candidates):
        ranked_cases = sorted(tbox_candidates[revision_id], key=lambda item: (item[0], item[1]))
        selected_pairs = ranked_cases[:tbox_cap_per_update]
        selected_t_box_ids.extend(case_id for _, case_id in selected_pairs)
        selected_counts_by_revision.append(
            {
                "property_revision_id": revision_id,
                "available_case_count": tbox_available_counts[revision_id],
                "selected_case_count": len(selected_pairs),
            }
        )

    selected_case_ids = sorted(selected_a_box_ids + selected_t_box_ids)
    selected_t_box = len(selected_t_box_ids)

    return {
        "manifest_type": "benchmark_case_selection",
        "manifest_version": 1,
        "inputs": {
            "classified_benchmark": str(classified_path),
        },
        "policy": {
            "scope": "paper_eval_subset",
            "selection_strategy": "keep_all_a_box_cap_t_box_per_property_revision",
            "t_box_group_key": "repair_target.property_revision_id",
            "tbox_cap_per_update": tbox_cap_per_update,
            "seed": seed,
            "stable_ordering": "sha1(seed|property_revision_id|case_id) ascending",
        },
        "counts": {
            "total_records": total_records,
            "total_a_box_cases": total_a_box,
            "total_t_box_cases": total_t_box,
            "distinct_t_box_updates": len(tbox_candidates),
            "selected_cases": len(selected_case_ids),
            "selected_a_box_cases": len(selected_a_box_ids),
            "selected_t_box_cases": selected_t_box,
        },
        "t_box_selected_counts_by_revision": selected_counts_by_revision,
        "selected_case_ids": selected_case_ids,
    }


def load_selection_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Selection manifest must be a JSON object.")
    selected_case_ids = manifest.get("selected_case_ids")
    if not isinstance(selected_case_ids, list):
        raise ValueError("Selection manifest must contain selected_case_ids.")
    normalized_ids = []
    seen = set()
    for case_id in selected_case_ids:
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Selection manifest contains an invalid case id.")
        if case_id in seen:
            raise ValueError(f"Selection manifest contains duplicate case id: {case_id}")
        seen.add(case_id)
        normalized_ids.append(case_id)
    manifest["selected_case_ids"] = normalized_ids
    return manifest


def resolve_case_id_filter(
    *,
    case_ids: Optional[Iterable[str]] = None,
    selection_manifest_path: str | Path | None = None,
) -> Optional[list[str]]:
    explicit_ids = _normalized_case_ids(case_ids)
    manifest_ids = None
    if selection_manifest_path:
        manifest = load_selection_manifest(selection_manifest_path)
        manifest_ids = _normalized_case_ids(manifest.get("selected_case_ids"))

    if explicit_ids is None and manifest_ids is None:
        return None
    if explicit_ids is None:
        return sorted(manifest_ids or set())
    if manifest_ids is None:
        return sorted(explicit_ids)
    return sorted(explicit_ids & manifest_ids)
