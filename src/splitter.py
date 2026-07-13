#!/usr/bin/env python3
"""
splitter.py -- deterministic train/dev/test splits for WikidataRepairEval 1.0

Reads:
  - data/04_classified_benchmark.jsonl (or data_sample/...)
Writes:
  - data/05_splits.json
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from lib.benchmark_selection import group_key_for_record, popularity_bucket_for_record
from lib.utils import iter_jsonl

DEFAULT_IN_PATH = "04_classified_benchmark.jsonl"
DEFAULT_OUT_PATH = "05_splits.json"

# Global defaults (tunable constants)
SEED = 13
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1
ALLOW_MISSING_POPULARITY = False
MAX_DELTA = 0.02


def derive_split_group(record: Dict[str, Any]) -> Dict[str, Any]:
    group_key, tbox_revision_key, weak_group_key = group_key_for_record(record)
    return {
        "group_key": group_key,
        "tbox_revision_key": tbox_revision_key,
        "weak_group_key": weak_group_key,
    }


def _stable_hash_seed(seed: int, salt: str) -> int:
    raw = f"{seed}|{salt}".encode("utf-8")
    digest = hashlib.sha1(raw).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _normalize_ratios(train: float, dev: float, test: float) -> Dict[str, float]:
    total = train + dev + test
    if total <= 0:
        raise ValueError("Split ratios must be positive.")
    return {"train": train / total, "dev": dev / total, "test": test / total}


def _allocate_counts(n: int, ratios: Dict[str, float]) -> Dict[str, int]:
    order = ("train", "dev", "test")
    exact = {k: n * ratios[k] for k in order}
    base = {k: int(math.floor(exact[k])) for k in order}
    remaining = n - sum(base.values())
    fracs = sorted(
        ((exact[k] - base[k], idx, k) for idx, k in enumerate(order)),
        key=lambda x: (-x[0], x[1]),
    )
    for i in range(remaining):
        base[fracs[i][2]] += 1
    return base


def _distribution(ids: Iterable[str], by_id: Dict[str, Dict[str, str]], field: str) -> Dict[str, float]:
    counter: Counter = Counter()
    total = 0
    for rid in ids:
        counter[by_id[rid][field]] += 1
        total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def _check_proportions(
    *,
    overall: Dict[str, float],
    split: Dict[str, float],
    split_name: str,
    field: str,
    n_split: int,
    max_delta: float,
) -> List[str]:
    issues: List[str] = []
    keys = sorted(set(overall) | set(split))
    tol = max_delta + (1.0 / max(1, n_split))
    for key in keys:
        o = overall.get(key, 0.0)
        s = split.get(key, 0.0)
        if abs(s - o) > tol:
            issues.append(
                f"{split_name}:{field}:{key} delta={abs(s - o):.4f} tol={tol:.4f} split={s:.4f} overall={o:.4f}"
            )
    return issues


def _assign_grouped_splits(
    entries: list[dict[str, Any]],
    *,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    """Assign complete leakage groups while approximately preserving all strata."""
    split_names = ("train", "dev", "test")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["group_key"]].append(entry)

    stratum_totals: Counter[tuple[str, str, str]] = Counter(
        (entry["class"], entry["track"], entry["popularity_bucket"]) for entry in entries
    )
    stratum_assigned: dict[str, Counter[tuple[str, str, str]]] = {
        name: Counter() for name in split_names
    }
    total_assigned: Counter[str] = Counter()
    target_total = {name: len(entries) * ratios[name] for name in split_names}

    group_rows: list[tuple[str, list[dict[str, Any]], Counter[tuple[str, str, str]]]] = []
    for group_key, rows in grouped.items():
        profile = Counter((row["class"], row["track"], row["popularity_bucket"]) for row in rows)
        group_rows.append((group_key, rows, profile))
    group_rows.sort(
        key=lambda item: (
            -len(item[1]),
            _stable_hash_seed(seed, item[0]),
            item[0],
        )
    )

    splits: dict[str, list[str]] = {name: [] for name in split_names}
    for group_key, rows, profile in group_rows:
        candidate_scores: list[tuple[float, int, str]] = []
        for name in split_names:
            delta = 0.0
            for stratum, count in profile.items():
                target = stratum_totals[stratum] * ratios[name]
                before = stratum_assigned[name][stratum]
                scale = max(target, 1.0)
                delta += ((before + count - target) ** 2 - (before - target) ** 2) / scale
            before_total = total_assigned[name]
            delta += (
                (before_total + len(rows) - target_total[name]) ** 2
                - (before_total - target_total[name]) ** 2
            ) / max(target_total[name], 1.0)
            tie_break = _stable_hash_seed(seed, f"{group_key}|{name}")
            candidate_scores.append((delta, tie_break, name))
        _, _, selected_split = min(candidate_scores)
        splits[selected_split].extend(row["id"] for row in rows)
        total_assigned[selected_split] += len(rows)
        stratum_assigned[selected_split].update(profile)

    return {name: sorted(ids) for name, ids in splits.items()}


def build_split_manifest(
    records: Iterable[dict[str, Any]],
    *,
    seed: int = SEED,
    train_ratio: float = TRAIN_RATIO,
    dev_ratio: float = DEV_RATIO,
    test_ratio: float = TEST_RATIO,
    allow_missing_popularity: bool = ALLOW_MISSING_POPULARITY,
    max_delta: float = MAX_DELTA,
    input_path: str | Path = DEFAULT_IN_PATH,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, str]] = {}
    for rec in records:
        rid = rec.get("id")
        if not isinstance(rid, str) or not rid:
            continue
        if rid in by_id:
            raise ValueError(f"Duplicate id detected: {rid}")
        classification = rec.get("classification") if isinstance(rec.get("classification"), dict) else {}
        cls = classification.get("class") if isinstance(classification, dict) else None
        track = rec.get("track")
        group = derive_split_group(rec)
        entry = {
            "id": rid,
            "class": cls if isinstance(cls, str) and cls else "UNKNOWN",
            "track": track if isinstance(track, str) and track else "UNKNOWN",
            "popularity_bucket": popularity_bucket_for_record(rec),
            **group,
        }
        if entry["popularity_bucket"] == "unknown" and not allow_missing_popularity:
            raise ValueError(f"Missing popularity score for record: {rid}")
        entries.append(entry)
        by_id[rid] = {
            "class": entry["class"],
            "track": entry["track"],
            "popularity_bucket": entry["popularity_bucket"],
        }

    ratios = _normalize_ratios(train_ratio, dev_ratio, test_ratio)
    splits = _assign_grouped_splits(entries, ratios=ratios, seed=seed)

    overall = {
        "class": _distribution(by_id.keys(), by_id, "class"),
        "track": _distribution(by_id.keys(), by_id, "track"),
        "popularity_bucket": _distribution(by_id.keys(), by_id, "popularity_bucket"),
    }
    issues: list[str] = []
    for name, ids in splits.items():
        for field, distribution in overall.items():
            issues.extend(
                _check_proportions(
                    overall=distribution,
                    split=_distribution(ids, by_id, field),
                    split_name=name,
                    field=field,
                    n_split=len(ids),
                    max_delta=max_delta,
                )
            )

    split_for_id = {case_id: name for name, ids in splits.items() for case_id in ids}
    group_splits: dict[str, set[str]] = defaultdict(set)
    weak_groups: set[str] = set()
    for entry in entries:
        group_splits[entry["group_key"]].add(split_for_id[entry["id"]])
        if entry["weak_group_key"]:
            weak_groups.add(entry["group_key"])
    cross_split_groups = sorted(key for key, names in group_splits.items() if len(names) > 1)
    if cross_split_groups:
        raise ValueError(f"Grouped split isolation failed for {len(cross_split_groups)} groups.")

    return {
        "inputs": {"classified_benchmark": str(input_path)},
        "policy": {
            "seed": seed,
            "ratios": ratios,
            "assignment_unit": "ABOX qid-property or TBOX property-revision group",
            "assignment_algorithm": "deterministic greedy stratum-balance v1",
            "popularity_buckets": {
                "policy": "explicit bucket, else score thresholds tail<=1/3 and head>=2/3",
                "missing_policy": "unknown" if allow_missing_popularity else "error",
            },
            "max_delta": max_delta,
        },
        "counts": {
            "total": len(entries),
            "splits": {key: len(value) for key, value in splits.items()},
            "classes": dict(Counter(entry["class"] for entry in entries)),
            "tracks": dict(Counter(entry["track"] for entry in entries)),
            "popularity_buckets": dict(Counter(entry["popularity_bucket"] for entry in entries)),
            "groups": len(group_splits),
            "weak_groups": len(weak_groups),
        },
        "validation": {
            "cross_split_group_count": len(cross_split_groups),
            "cross_split_groups": cross_split_groups,
            "proportion_issues": issues,
            "proportion_checks_passed": not issues,
        },
        "splits": splits,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="Use data_sample/ inputs/outputs instead of data/.")
    args = ap.parse_args()

    folder = Path("data_sample") if args.sample else Path("data")
    in_path = folder / DEFAULT_IN_PATH
    out_path = folder / DEFAULT_OUT_PATH

    output = build_split_manifest(iter_jsonl(in_path), input_path=in_path)

    issues = output["validation"]["proportion_issues"]
    if issues:
        preview = "\n".join(issues[:10])
        raise ValueError(f"Split proportion checks failed (showing up to 10):\n{preview}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
