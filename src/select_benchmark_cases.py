#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from lib.benchmark_selection import (
    DEFAULT_SELECTION_SEED,
    DEFAULT_TBOX_CAP_PER_UPDATE,
    build_selection_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic benchmark case-selection manifest that keeps all "
            "A-BOX cases and caps T-BOX cases per property revision."
        )
    )
    parser.add_argument("--classified-benchmark", default="data/04_classified_benchmark.jsonl")
    parser.add_argument(
        "--output",
        default="reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json",
        help="Path to the JSON selection manifest to write.",
    )
    parser.add_argument(
        "--tbox-cap-per-update",
        type=int,
        default=DEFAULT_TBOX_CAP_PER_UPDATE,
        help="Maximum number of T-BOX cases to keep per property revision.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SELECTION_SEED,
        help="Seed used in the stable hash ordering for within-update case selection.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print a progress update every N input lines. Set to 0 to disable.",
    )
    args = parser.parse_args()

    manifest = build_selection_manifest(
        args.classified_benchmark,
        tbox_cap_per_update=args.tbox_cap_per_update,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts = manifest["counts"]
    print(f"[done] wrote {out_path}")
    print(f"[done] selected_cases={counts['selected_cases']:,}")
    print(f"[done] selected_a_box_cases={counts['selected_a_box_cases']:,}")
    print(f"[done] selected_t_box_cases={counts['selected_t_box_cases']:,}")
    print(f"[done] distinct_t_box_updates={counts['distinct_t_box_updates']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
