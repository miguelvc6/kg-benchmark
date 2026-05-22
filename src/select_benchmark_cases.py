#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lib.benchmark_selection import (
    DEFAULT_ABOX_CAP_CORE,
    DEFAULT_ABOX_CAP_DEV,
    DEFAULT_CORE_SIZE,
    DEFAULT_DEV_SIZE,
    DEFAULT_SELECTION_SEED,
    DEFAULT_SELECTED_CASE_ORDER,
    DEFAULT_TBOX_CAP_CORE,
    DEFAULT_TBOX_CAP_DEV,
    SUPPORTED_SELECTED_CASE_ORDERS,
    SelectionOptions,
    build_tier_manifest,
)


def _default_output_for_tier(tier: str, seed: int) -> str:
    name = "dev_prompt" if tier == "dev" else "core"
    return f"reports/benchmark_selection/{name}_v1_seed_{seed}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic Phase C benchmark selection manifest from the "
            "canonical Stage 4 classified benchmark."
        )
    )
    parser.add_argument("--tier", choices=("dev", "core"), required=True)
    parser.add_argument("--classified-benchmark", default="data/04_classified_benchmark.jsonl")
    parser.add_argument("--output", help="Path to the JSON selection manifest to write.")
    parser.add_argument("--exclude-manifest", help="Dev manifest to exclude when building the core tier.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--core-size", type=int, default=DEFAULT_CORE_SIZE)
    parser.add_argument("--dev-size", type=int, default=DEFAULT_DEV_SIZE)
    parser.add_argument("--tbox-cap-core", type=int, default=DEFAULT_TBOX_CAP_CORE)
    parser.add_argument("--tbox-cap-dev", type=int, default=DEFAULT_TBOX_CAP_DEV)
    parser.add_argument("--abox-cap-core", type=int, default=DEFAULT_ABOX_CAP_CORE)
    parser.add_argument("--abox-cap-dev", type=int, default=DEFAULT_ABOX_CAP_DEV)
    parser.add_argument(
        "--selected-case-order",
        choices=sorted(SUPPORTED_SELECTED_CASE_ORDERS),
        default=DEFAULT_SELECTED_CASE_ORDER,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print a progress update every N input lines. Set to 0 to disable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output or _default_output_for_tier(args.tier, args.seed))
    options = SelectionOptions(
        classified_benchmark=Path(args.classified_benchmark),
        tier=args.tier,
        seed=args.seed,
        core_size=args.core_size,
        dev_size=args.dev_size,
        tbox_cap_core=args.tbox_cap_core,
        tbox_cap_dev=args.tbox_cap_dev,
        abox_cap_core=args.abox_cap_core,
        abox_cap_dev=args.abox_cap_dev,
        selected_case_order=args.selected_case_order,
        progress_every=args.progress_every,
        exclude_manifest=Path(args.exclude_manifest) if args.exclude_manifest else None,
    )

    manifest = build_tier_manifest(options)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    counts = manifest["counts"]
    validation = manifest["validation"]
    print(f"[done] wrote {output}")
    print(f"[done] tier={manifest['tier']} seed={manifest['seed']}")
    print(f"[done] selected={counts['selected']:,} main_score={counts['main_score']:,} diagnostic={counts['diagnostic']:,}")
    print(f"[done] underfilled_quotas={len(manifest['underfilled_quotas'])}")
    print(f"[done] hard_validation_passed={validation['hard_validation_passed']}")
    if manifest["warnings"]:
        for warning in manifest["warnings"]:
            print(f"[warning] {warning}")

    if args.tier == "core" and not validation["hard_validation_passed"]:
        print("[error] core manifest failed hard Phase C validation checks")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
