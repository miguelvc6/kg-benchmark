#!/usr/bin/env python3

import argparse

from guardian.evaluator import evaluate_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate benchmark proposals against Stage-4 benchmark cases.")
    parser.add_argument("--classified-benchmark", default="data/04_classified_benchmark.jsonl")
    parser.add_argument("--world-state", default="data/03_world_state.json")
    parser.add_argument("--a-box-proposals", default=None)
    parser.add_argument("--t-box-proposals", default=None)
    parser.add_argument("--track-diagnoses", default=None)
    parser.add_argument("--run-manifest", default=None)
    parser.add_argument("--ablation-bundle", default=None)
    parser.add_argument("--out-traces", default="reports/evaluation_traces.jsonl")
    parser.add_argument("--out-summary", default="reports/evaluation_summary.json")
    parser.add_argument("--case-ids", default=None, help="Comma-separated case ids to evaluate.")
    args = parser.parse_args()

    case_ids = [item.strip() for item in args.case_ids.split(",")] if args.case_ids else None
    evaluate_benchmark(
        classified_path=args.classified_benchmark,
        world_state_path=args.world_state,
        a_box_proposals_path=args.a_box_proposals,
        t_box_proposals_path=args.t_box_proposals,
        track_diagnoses_path=args.track_diagnoses,
        run_manifest_path=args.run_manifest,
        ablation_bundle=args.ablation_bundle,
        case_ids=case_ids,
        out_traces_path=args.out_traces,
        out_summary_path=args.out_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
