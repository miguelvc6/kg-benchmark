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
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

DEFAULT_IN_PATH = "04_classified_benchmark.jsonl"
DEFAULT_OUT_PATH = "05_splits.json"

# Global defaults (tunable constants)
SEED = 13
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1
TAIL_FRAC = 0.2
HEAD_FRAC = 0.2
ALLOW_MISSING_POPULARITY = False
MAX_DELTA = 0.02


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _popularity_score(rec: Dict[str, Any]) -> Optional[float]:
    pop = rec.get("popularity")
    if isinstance(pop, dict):
        score = pop.get("score")
        if isinstance(score, (int, float)):
            return float(score)
    return None


def _stable_hash_seed(seed: int, salt: str) -> int:
    raw = f"{seed}|{salt}".encode("utf-8")
    digest = hashlib.sha1(raw).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _assign_popularity_buckets(
    entries: List[Dict[str, Any]],
    *,
    tail_frac: float,
    head_frac: float,
    allow_missing: bool,
) -> Dict[str, str]:
    scored: List[Tuple[float, str]] = []
    missing_ids: List[str] = []
    for e in entries:
        rid = e["id"]
        score = e.get("popularity_score")
        if score is None:
            missing_ids.append(rid)
        else:
            scored.append((score, rid))

    if missing_ids and not allow_missing:
        raise ValueError(f"Missing popularity score for {len(missing_ids)} records.")

    scored.sort(key=lambda x: (x[0], x[1]))
    n = len(scored)
    n_tail = int(n * tail_frac)
    n_head = int(n * head_frac)

    buckets: Dict[str, str] = {}
    for idx, (_, rid) in enumerate(scored):
        if idx < n_tail:
            buckets[rid] = "tail"
        elif idx >= n - n_head:
            buckets[rid] = "head"
        else:
            buckets[rid] = "mid"

    if allow_missing:
        for rid in missing_ids:
            buckets[rid] = "unknown"

    return buckets


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="Use data_sample/ inputs/outputs instead of data/.")
    args = ap.parse_args()

    folder = Path("data_sample") if args.sample else Path("data")
    in_path = folder / DEFAULT_IN_PATH
    out_path = folder / DEFAULT_OUT_PATH

    entries: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, str]] = {}

    for rec in iter_jsonl(in_path):
        rid = rec.get("id")
        if not isinstance(rid, str) or not rid:
            continue
        if rid in by_id:
            raise ValueError(f"Duplicate id detected: {rid}")
        classification = rec.get("classification") if isinstance(rec.get("classification"), dict) else {}
        cls = classification.get("class") if isinstance(classification, dict) else None
        track = rec.get("track")
        entry = {
            "id": rid,
            "class": cls if isinstance(cls, str) and cls else "UNKNOWN",
            "track": track if isinstance(track, str) and track else "UNKNOWN",
            "popularity_score": _popularity_score(rec),
        }
        entries.append(entry)
        by_id[rid] = {"class": entry["class"], "track": entry["track"]}

    buckets = _assign_popularity_buckets(
        entries,
        tail_frac=TAIL_FRAC,
        head_frac=HEAD_FRAC,
        allow_missing=ALLOW_MISSING_POPULARITY,
    )

    for e in entries:
        bucket = buckets.get(e["id"])
        e["popularity_bucket"] = bucket if isinstance(bucket, str) else "unknown"
        by_id[e["id"]]["popularity_bucket"] = e["popularity_bucket"]

    ratios = _normalize_ratios(TRAIN_RATIO, DEV_RATIO, TEST_RATIO)

    strata: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for e in entries:
        key = (e["class"], e["track"], e["popularity_bucket"])
        strata[key].append(e["id"])

    splits: Dict[str, List[str]] = {"train": [], "dev": [], "test": []}

    for key in sorted(strata.keys()):
        ids = sorted(strata[key])
        seed = _stable_hash_seed(SEED, "|".join(key))
        rng = random.Random(seed)
        rng.shuffle(ids)
        counts = _allocate_counts(len(ids), ratios)
        n_train = counts["train"]
        n_dev = counts["dev"]
        splits["train"].extend(ids[:n_train])
        splits["dev"].extend(ids[n_train : n_train + n_dev])
        splits["test"].extend(ids[n_train + n_dev :])

    for name in splits:
        splits[name] = sorted(splits[name])

    overall_class = _distribution(by_id.keys(), by_id, "class")
    overall_track = _distribution(by_id.keys(), by_id, "track")
    overall_pop = _distribution(by_id.keys(), by_id, "popularity_bucket")

    issues: List[str] = []
    for name in ("train", "dev", "test"):
        ids = splits[name]
        issues.extend(
            _check_proportions(
                overall=overall_class,
                split=_distribution(ids, by_id, "class"),
                split_name=name,
                field="class",
                n_split=len(ids),
                max_delta=MAX_DELTA,
            )
        )
        issues.extend(
            _check_proportions(
                overall=overall_track,
                split=_distribution(ids, by_id, "track"),
                split_name=name,
                field="track",
                n_split=len(ids),
                max_delta=MAX_DELTA,
            )
        )
        issues.extend(
            _check_proportions(
                overall=overall_pop,
                split=_distribution(ids, by_id, "popularity_bucket"),
                split_name=name,
                field="popularity_bucket",
                n_split=len(ids),
                max_delta=MAX_DELTA,
            )
        )

    if issues:
        preview = "\n".join(issues[:10])
        raise ValueError(f"Split proportion checks failed (showing up to 10):\n{preview}")

    output = {
        "inputs": {"classified_benchmark": str(in_path)},
        "policy": {
            "seed": SEED,
            "ratios": ratios,
            "popularity_buckets": {
                "tail_frac": TAIL_FRAC,
                "head_frac": HEAD_FRAC,
                "missing_policy": "unknown" if ALLOW_MISSING_POPULARITY else "error",
            },
            "max_delta": MAX_DELTA,
        },
        "counts": {
            "total": len(entries),
            "splits": {k: len(v) for k, v in splits.items()},
            "classes": dict(Counter(e["class"] for e in entries)),
            "tracks": dict(Counter(e["track"] for e in entries)),
            "popularity_buckets": dict(Counter(e["popularity_bucket"] for e in entries)),
        },
        "splits": splits,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
