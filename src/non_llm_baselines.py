#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from lib.non_llm_baselines import PhaseEOptions, run_phase_e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate Phase E non-LLM baselines.")
    parser.add_argument("--classified-benchmark", default="data/04_classified_benchmark.jsonl")
    parser.add_argument("--world-state", default="data/03_world_state.json")
    parser.add_argument("--selection-manifest", default="reports/benchmark_selection/core_v1_seed_13.json")
    parser.add_argument("--output-dir", default="reports/non_llm_baselines/core_v1_phase_e")
    parser.add_argument("--tier", default="core")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N selected cases for proposal baselines. Set to 0 to disable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_phase_e(
        PhaseEOptions(
            classified_benchmark=Path(args.classified_benchmark),
            world_state=Path(args.world_state),
            selection_manifest=Path(args.selection_manifest),
            output_dir=Path(args.output_dir),
            tier=args.tier,
            progress_every=args.progress_every,
        )
    )
    print(f"[done] wrote {args.output_dir}")
    for name, baseline in sorted(summary["baselines"].items()):
        accepted = baseline.get("accepted_rate")
        track_accuracy = baseline.get("track_diagnosis_accuracy")
        coverage = baseline.get("coverage", {})
        coverage_value = coverage.get("coverage") if isinstance(coverage, dict) else None
        print(
            "[done] {name}: accepted_rate={accepted} track_accuracy={track_accuracy} coverage={coverage}".format(
                name=name,
                accepted=_fmt(accepted),
                track_accuracy=_fmt(track_accuracy),
                coverage=_fmt(coverage_value),
            )
        )
    return 0


def _fmt(value: object) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
