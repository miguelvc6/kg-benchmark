#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from classifier import WorldStateStore, classify_one


TARGET_OLD = {
    ("TypeB", "LOCAL_TEXT"),
    ("TypeB", "LOCAL_FOCUS_PREREPAIR_PROPERTY"),
    ("TypeB", "LOCAL_TEXT_CONFIRMED"),
    ("TypeB", "LOCAL_FOCUS_QID"),
    ("TypeB", "LOCAL_TEXT_DERIVED"),
    ("TypeC", "EXTERNAL_BY_ELIMINATION"),
    ("TypeC", "UNKNOWN_BAD_TARGET_OR_CONTEXT"),
    ("TypeC", "UNKNOWN_SELECTION_AMBIGUOUS"),
    ("TypeC", "UNKNOWN_FOCUS_QID_DOMAIN_REASONING"),
    ("TypeA", "REJECTION_FORMAT_INVALID"),
    ("TypeA", "DELETE_AMBIGUOUS"),
    ("TypeA", "FORMAT_VALUE_PRUNING"),
    ("TypeA", "FORMAT_NORMALIZATION"),
    ("TypeA", "MULTIPLICITY_NORMALIZATION"),
    ("T_BOX", "SCHEMA_UPDATE"),
    ("T_BOX", "COINCIDENTAL_SCHEMA_CHANGE"),
    ("T_BOX", "RELAXATION_SET_EXPANSION"),
    ("T_BOX", "RESTRICTION_SET_CONTRACTION"),
    ("T_BOX", "UNKNOWN_TBOX_CAUSALITY"),
}
TARGET_SUBTYPE_MARKERS = tuple(f'"subtype": "{subtype}"' for _, subtype in sorted(TARGET_OLD))


def truth_kind(record: dict[str, Any]) -> str:
    diagnostics = record.get("classification", {}).get("diagnostics", {})
    tokens = diagnostics.get("truth_tokens") if isinstance(diagnostics, dict) else []
    if not tokens:
        return "none"
    kinds = set()
    for token in tokens:
        token_s = str(token)
        if token_s.startswith("Q"):
            kinds.add("qid")
        else:
            kinds.add("literal")
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare current Stage 4 labels to hardened classifier labels.")
    parser.add_argument("--input", default="data/04_classified_benchmark.jsonl")
    parser.add_argument("--world-state", default="data/03_world_state.json")
    parser.add_argument("--out", default="reports/classifier_audit/transition_after_delta_hardening.json")
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("classifier_delta_transition")
    transitions: Counter[str] = Counter()
    by_bucket: Counter[str] = Counter()
    scanned = 0
    considered = 0
    store: WorldStateStore | None = None
    try:
        with Path(args.input).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if args.progress_every > 0 and line_number % args.progress_every == 0:
                    print(f"[progress] scanned={line_number:,} considered={considered:,}", flush=True)
                if not line.strip():
                    continue
                scanned += 1
                if not any(marker in line for marker in TARGET_SUBTYPE_MARKERS):
                    continue
                old_record = json.loads(line)
                old_cls = old_record.get("classification", {}).get("class")
                old_subtype = old_record.get("classification", {}).get("subtype")
                if (old_cls, old_subtype) not in TARGET_OLD:
                    continue
                considered += 1
                if old_cls == "T_BOX":
                    ws_entry = {}
                else:
                    if store is None:
                        store = WorldStateStore(Path(args.world_state), logger)
                        store.open()
                    ws_entry = store.get(old_record["id"])
                new_classification, _, _ = classify_one(old_record, ws_entry)
                new_cls = new_classification.get("class")
                new_subtype = new_classification.get("subtype")
                old_key = f"{old_cls}/{old_subtype}"
                if old_cls == "TypeC" and old_subtype == "EXTERNAL_BY_ELIMINATION":
                    old_key = f"{old_key}/{truth_kind(old_record)}_truth"
                transitions[f"{old_key} -> {new_cls}/{new_subtype}"] += 1
                by_bucket[old_key] += 1
    finally:
        if store is not None:
            store.close()

    report = {
        "report_type": "classifier_delta_transition",
        "input": args.input,
        "world_state": args.world_state,
        "records_scanned": scanned,
        "records_considered": considered,
        "target_old_counts": dict(sorted(by_bucket.items())),
        "transitions": dict(sorted(transitions.items())),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] wrote {out}")
    print(f"[done] records_considered={considered:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
