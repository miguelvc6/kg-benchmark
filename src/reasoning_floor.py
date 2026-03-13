#!/usr/bin/env python3

import argparse

from guardian.reasoning import ABLATION_BUNDLES, run_reasoning_floor


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the zero-shot reasoning-floor baseline.")
    parser.add_argument("--classified-benchmark", default="data/04_classified_benchmark.jsonl")
    parser.add_argument("--world-state", default="data/03_world_state.json")
    parser.add_argument("--output-dir", default="reports/reasoning_floor")
    parser.add_argument("--model", default=None, help="Override the model name configured in .env.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-ids", default=None, help="Comma-separated case ids to include.")
    parser.add_argument(
        "--selection-manifest",
        default=None,
        help="Path to a JSON selection manifest containing selected_case_ids.",
    )
    parser.add_argument("--tracks", default=None, help="Comma-separated track filter, e.g. A_BOX,T_BOX.")
    parser.add_argument(
        "--ablation-bundles",
        default=",".join(ABLATION_BUNDLES),
        help="Comma-separated bundle names.",
    )
    args = parser.parse_args()

    run_reasoning_floor(
        classified_path=args.classified_benchmark,
        world_state_path=args.world_state,
        output_dir=args.output_dir,
        model_name=args.model,
        case_ids=[item.strip() for item in args.case_ids.split(",")] if args.case_ids else None,
        selection_manifest_path=args.selection_manifest,
        tracks=[item.strip() for item in args.tracks.split(",")] if args.tracks else None,
        max_cases=args.max_cases,
        ablation_bundles=[item.strip() for item in args.ablation_bundles.split(",") if item.strip()],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

