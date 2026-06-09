#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from lib.prompt_dev import (
    DEFAULT_CONTEXT_BUNDLES,
    DEFAULT_RENDER_TASKS,
    EXAMPLE_POLICIES,
    REPAIR_TRACK_MODES,
    SAMPLE_STRATEGIES,
    PromptDevEvaluateOptions,
    PromptDevMatrixOptions,
    PromptDevRenderOptions,
    evaluate_prompt_dev_prompts,
    freeze_prompt_dev_config,
    render_prompt_dev_prompts,
    write_prompt_dev_matrix,
)
from scripts.prompt_dev_templates import REPRESENTATIONS


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _csv_tuple(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or default


def _status_text(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))


def _make_evaluate_progress_bar(*, disabled: bool):
    counts: Counter[str] = Counter()
    bar = tqdm(total=0, desc="Evaluating prompts", unit="prompt", disable=disabled)

    def progress_callback(event: dict) -> None:
        event_name = event.get("event")
        if event_name == "start":
            bar.reset(total=int(event.get("total") or 0))
            provider = event.get("provider") or "provider"
            model = event.get("model") or "model"
            bar.set_description(f"Evaluating prompts [{provider}/{model}]")
            return
        if event_name != "advance":
            return
        status = str(event.get("parse_status") or "unknown")
        counts[status] += 1
        bar.update(1)
        bar.set_postfix_str(_status_text(counts), refresh=False)

    return bar, progress_callback


def _add_axis_args(parser: argparse.ArgumentParser, *, render_defaults: bool = False) -> None:
    representation_default = ("hybrid_json_nl",) if render_defaults else REPRESENTATIONS
    example_default = ("zero_shot",) if render_defaults else EXAMPLE_POLICIES
    track_mode_default = ("oracle",) if render_defaults else REPAIR_TRACK_MODES
    parser.add_argument(
        "--representations",
        default=",".join(representation_default),
        help=f"Comma-separated representations. Supported: {', '.join(REPRESENTATIONS)}.",
    )
    parser.add_argument(
        "--example-policies",
        default=",".join(example_default),
        help=f"Comma-separated example policies. Supported: {', '.join(EXAMPLE_POLICIES)}.",
    )
    parser.add_argument(
        "--context-bundles",
        default=",".join(DEFAULT_CONTEXT_BUNDLES),
        help="Comma-separated context bundles, e.g. logic_only,local_graph.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_RENDER_TASKS),
        help="Comma-separated tasks: track_diagnosis,repair_proposal.",
    )
    parser.add_argument(
        "--repair-track-modes",
        default=",".join(track_mode_default),
        help="Comma-separated repair proposal modes: oracle,diagnosis_routed.",
    )
    parser.add_argument(
        "--include-abstention",
        action="store_true",
        help="Include the optional abstention contract in repair prompts.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and evaluate Phase F prompt-development artifacts.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Terminal logging level. Logs include timestamps and long-running setup phases.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix_parser = subparsers.add_parser("matrix", help="Write the Phase F prompt-development run matrix.")
    matrix_parser.add_argument("--output", default="reports/prompt_dev/prompt_dev_matrix_v1.json")
    _add_axis_args(matrix_parser)

    render_parser = subparsers.add_parser("render", help="Render prompt-review artifacts over the dev manifest.")
    render_parser.add_argument("--classified-benchmark", default="data/04_classified_benchmark.jsonl")
    render_parser.add_argument("--world-state", default="data/03_world_state.json")
    render_parser.add_argument("--dev-manifest", default="reports/benchmark_selection/dev_prompt_v1_seed_13.json")
    render_parser.add_argument("--core-manifest", default=None)
    render_parser.add_argument("--output-dir", default="reports/prompt_dev/rendered_prompt_dev_v1")
    render_parser.add_argument("--seed", type=int, default=13)
    render_parser.add_argument("--max-cases", type=int, default=24)
    render_parser.add_argument("--example-count", type=int, default=2)
    render_parser.add_argument(
        "--sample-strategy",
        choices=SAMPLE_STRATEGIES,
        default="stratified",
        help=(
            "Case sampling strategy for --max-cases. Use diverse_stratified for broader canaries that prefer unseen "
            "focus QIDs and properties inside balanced strata."
        ),
    )
    render_parser.add_argument(
        "--allow-same-property-examples",
        action="store_true",
        help="Allow few-shot examples from the same property. Disabled by default for leakage control.",
    )
    render_parser.add_argument(
        "--allow-core-example-risk",
        action="store_true",
        help="Allow few-shot prompt rendering without a core manifest. Use only for explicit leakage-risk experiments.",
    )
    _add_axis_args(render_parser, render_defaults=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run dev-manifest prompt variants through a model endpoint and score them.",
    )
    evaluate_parser.add_argument("--classified-benchmark", default="data/04_classified_benchmark.jsonl")
    evaluate_parser.add_argument("--world-state", default="data/03_world_state.json")
    evaluate_parser.add_argument("--dev-manifest", default="reports/benchmark_selection/dev_prompt_v1_seed_13.json")
    evaluate_parser.add_argument("--core-manifest", default=None)
    evaluate_parser.add_argument("--output-dir", default="reports/prompt_dev/evaluation_prompt_dev_v1")
    evaluate_parser.add_argument("--model", default=None, help="Override the model name configured in .env.")
    evaluate_parser.add_argument(
        "--model-endpoint",
        choices=("ollama", "azure", "university", "openai"),
        default=None,
        help="Choose the model endpoint configuration. Defaults to MODEL_ENDPOINT or MODEL_PROVIDER from .env.",
    )
    evaluate_parser.add_argument("--seed", type=int, default=13)
    evaluate_parser.add_argument("--max-cases", type=int, default=24)
    evaluate_parser.add_argument("--example-count", type=int, default=2)
    evaluate_parser.add_argument(
        "--sample-strategy",
        choices=SAMPLE_STRATEGIES,
        default="stratified",
        help=(
            "Case sampling strategy for --max-cases. Use diverse_stratified for broader canaries that prefer unseen "
            "focus QIDs and properties inside balanced strata."
        ),
    )
    evaluate_parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=None,
        help="Skip prompts whose system+user prompt length exceeds this character count.",
    )
    evaluate_parser.add_argument(
        "--allow-same-property-examples",
        action="store_true",
        help="Allow few-shot examples from the same property. Disabled by default for leakage control.",
    )
    evaluate_parser.add_argument(
        "--allow-core-example-risk",
        action="store_true",
        help="Allow few-shot evaluation without a core manifest. Use only for explicit leakage-risk experiments.",
    )
    evaluate_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip existing prompt results in the output directory.",
    )
    evaluate_parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="When resuming, retry existing request_error and parse_error rows instead of leaving them as-is.",
    )
    evaluate_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar for prompt evaluation.",
    )
    _add_axis_args(evaluate_parser, render_defaults=True)

    freeze_parser = subparsers.add_parser("freeze", help="Write a frozen Phase F prompt configuration.")
    freeze_parser.add_argument("--output", default="reports/prompt_dev/final_prompts_prompt_dev_v1.json")
    freeze_parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    freeze_parser.add_argument("--example-policy", choices=EXAMPLE_POLICIES, required=True)
    freeze_parser.add_argument("--context-bundles", default="logic_only,local_graph")
    freeze_parser.add_argument("--proposal-track-modes", default="oracle,diagnosis_routed")
    freeze_parser.add_argument("--include-abstention", action="store_true")
    freeze_parser.add_argument("--notes", default="")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.log_level)
    log = logging.getLogger("prompt_dev.cli")
    log.info("command=%s", args.command)
    if args.command == "matrix":
        log.info("writing prompt-dev matrix output=%s", args.output)
        matrix = write_prompt_dev_matrix(
            args.output,
            PromptDevMatrixOptions(
                representations=_csv_tuple(args.representations, REPRESENTATIONS),
                example_policies=_csv_tuple(args.example_policies, EXAMPLE_POLICIES),
                context_bundles=_csv_tuple(args.context_bundles, DEFAULT_CONTEXT_BUNDLES),
                tasks=_csv_tuple(args.tasks, DEFAULT_RENDER_TASKS),
                repair_track_modes=_csv_tuple(args.repair_track_modes, REPAIR_TRACK_MODES),
                include_abstention=args.include_abstention,
            ),
        )
        print(f"[done] wrote {args.output}")
        print(f"[done] rows={matrix['counts']['rows']}")
        return 0

    if args.command == "render":
        log.info(
            "render start classified=%s world_state=%s dev_manifest=%s output_dir=%s max_cases=%s",
            args.classified_benchmark,
            args.world_state,
            args.dev_manifest,
            args.output_dir,
            args.max_cases,
        )
        summary = render_prompt_dev_prompts(
            PromptDevRenderOptions(
                classified_benchmark=Path(args.classified_benchmark),
                world_state=Path(args.world_state),
                dev_manifest=Path(args.dev_manifest),
                core_manifest=Path(args.core_manifest) if args.core_manifest else None,
                output_dir=Path(args.output_dir),
                seed=args.seed,
                max_cases=args.max_cases,
                representations=_csv_tuple(args.representations, ("hybrid_json_nl",)),
                example_policies=_csv_tuple(args.example_policies, ("zero_shot",)),
                context_bundles=_csv_tuple(args.context_bundles, DEFAULT_CONTEXT_BUNDLES),
                tasks=_csv_tuple(args.tasks, DEFAULT_RENDER_TASKS),
                repair_track_modes=_csv_tuple(args.repair_track_modes, ("oracle",)),
                include_abstention=args.include_abstention,
                example_count=args.example_count,
                allow_same_property_examples=args.allow_same_property_examples,
                sample_strategy=args.sample_strategy,
                allow_core_example_risk=args.allow_core_example_risk,
            )
        )
        print(f"[done] rendered={summary['counts']['rendered_prompts']}")
        print(f"[done] prompts={summary['outputs']['prompts_jsonl']}")
        print(f"[done] review={summary['outputs']['review_markdown']}")
        log.info("render done rendered=%s output_dir=%s", summary["counts"]["rendered_prompts"], args.output_dir)
        return 0

    if args.command == "evaluate":
        log.info(
            "evaluate start endpoint=%s model=%s classified=%s world_state=%s dev_manifest=%s output_dir=%s max_cases=%s",
            args.model_endpoint or "env",
            args.model or "env",
            args.classified_benchmark,
            args.world_state,
            args.dev_manifest,
            args.output_dir,
            args.max_cases,
        )
        progress_bar, progress_callback = _make_evaluate_progress_bar(disabled=args.no_progress)
        with progress_bar:
            summary = evaluate_prompt_dev_prompts(
                PromptDevEvaluateOptions(
                    classified_benchmark=Path(args.classified_benchmark),
                    world_state=Path(args.world_state),
                    dev_manifest=Path(args.dev_manifest),
                    core_manifest=Path(args.core_manifest) if args.core_manifest else None,
                    output_dir=Path(args.output_dir),
                    model_endpoint=args.model_endpoint,
                    model_name=args.model,
                    seed=args.seed,
                    max_cases=args.max_cases,
                    representations=_csv_tuple(args.representations, ("hybrid_json_nl",)),
                    example_policies=_csv_tuple(args.example_policies, ("zero_shot",)),
                    context_bundles=_csv_tuple(args.context_bundles, DEFAULT_CONTEXT_BUNDLES),
                    tasks=_csv_tuple(args.tasks, DEFAULT_RENDER_TASKS),
                    repair_track_modes=_csv_tuple(args.repair_track_modes, ("oracle",)),
                    include_abstention=args.include_abstention,
                    example_count=args.example_count,
                    allow_same_property_examples=args.allow_same_property_examples,
                    resume_existing=not args.no_resume,
                    retry_failures=args.retry_failures,
                    max_prompt_chars=args.max_prompt_chars,
                    progress_callback=progress_callback,
                    sample_strategy=args.sample_strategy,
                    allow_core_example_risk=args.allow_core_example_risk,
                )
            )
        print(f"[done] evaluated_prompts={summary['counts']['evaluated_prompts']}")
        print(f"[done] summary={args.output_dir}/prompt_dev_evaluation_summary.json")
        print(f"[done] comparison={summary['outputs']['comparison_markdown']}")
        log.info("evaluate done evaluated_prompts=%s output_dir=%s", summary["counts"]["evaluated_prompts"], args.output_dir)
        return 0

    if args.command == "freeze":
        log.info("freeze start output=%s", args.output)
        config = freeze_prompt_dev_config(
            output=args.output,
            representation=args.representation,
            example_policy=args.example_policy,
            context_bundles=_csv_tuple(args.context_bundles, DEFAULT_CONTEXT_BUNDLES),
            proposal_track_modes=_csv_tuple(args.proposal_track_modes, REPAIR_TRACK_MODES),
            include_abstention=args.include_abstention,
            notes=args.notes,
        )
        print(f"[done] wrote {args.output}")
        print(f"[done] prompt_version={config['manifest_version']}")
        log.info("freeze done output=%s prompt_version=%s", args.output, config["manifest_version"])
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
