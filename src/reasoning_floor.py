#!/usr/bin/env python3

import argparse

from guardian.reasoning import (
    ABLATION_BUNDLES,
    TBOX_TASK_VERSION_TAXONOMY_PATCH,
    configure_tbox_task_version,
    run_reasoning_floor,
)

DEFAULT_ABLATION_BUNDLES = tuple(bundle for bundle in ABLATION_BUNDLES if bundle != "minimal_case")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the paper reasoning-floor tasks.")
    parser.add_argument("--classified-benchmark", default="dataset/cases.jsonl")
    parser.add_argument("--world-state", default="dataset/source/world-state.jsonl")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument(
        "--resume-run-dir",
        default=None,
        help="Resume an interrupted reasoning-floor run from an existing run directory.",
    )
    parser.add_argument("--model", default=None, help="Override the model name configured in .env.")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        default=None,
        help="Explicit reasoning effort for OpenAI-compatible providers.",
    )
    parser.add_argument("--context-length", type=int, default=None, help="Explicit Ollama context window.")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Explicit completion-token limit (Ollama num_predict or OpenAI max_completion_tokens).",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Explicit Ollama sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Explicit Ollama nucleus-sampling threshold.")
    parser.add_argument("--seed", type=int, default=None, help="Explicit Ollama sampling seed.")
    parser.add_argument(
        "--ollama-think",
        choices=("enabled", "disabled", "low", "medium", "high", "max"),
        default=None,
        help="Pin Ollama's native hidden-reasoning mode; traces are redacted from saved responses.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum exact-request transport retries for providers that support them.",
    )
    parser.add_argument(
        "--model-digest",
        default=None,
        help="Immutable model/deployment revision digest used in provenance and cache identity.",
    )
    parser.add_argument("--protocol", default="paper/protocol.json", help="Paper protocol to validate and bind.")
    parser.add_argument(
        "--generation-cache",
        default=None,
        help="Append-only SQLite cache for exact response reuse across runs and populations.",
    )
    parser.add_argument(
        "--model-endpoint",
        choices=("ollama", "azure", "university", "openai"),
        default=None,
        help=(
            "Choose the model endpoint configuration to use. "
            "Defaults to MODEL_ENDPOINT or MODEL_PROVIDER from .env when omitted."
        ),
    )
    parser.add_argument(
        "--tbox-task-version",
        choices=(TBOX_TASK_VERSION_TAXONOMY_PATCH,),
        default=TBOX_TASK_VERSION_TAXONOMY_PATCH,
        help="Paper T-box answer contract (taxonomy patch).",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("sync", "parallel", "batch"),
        default=None,
        help="Override execution mode. Defaults to batch for the OpenAI provider and sync otherwise.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Concurrent case workers for --execution-mode=parallel.",
    )
    parser.add_argument(
        "--batch-completion-window",
        default="24h",
        help="Completion window to request when --execution-mode=batch.",
    )
    parser.add_argument(
        "--batch-poll-interval-seconds",
        type=float,
        default=60.0,
        help="Polling interval in seconds while waiting for a batch job to finish.",
    )
    parser.add_argument(
        "--batch-sync-retry-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Retry eligible failed batch requests synchronously. Disable for strict batch-only execution."
        ),
    )
    parser.add_argument(
        "--proposal-track-mode",
        choices=("oracle",),
        default="oracle",
        help="Use the audited historical track for proposal generation.",
    )
    parser.add_argument(
        "--oracle-diagnosis-mode",
        choices=("run", "skip"),
        default="run",
        help="Whether to run the independently scored track-diagnosis task alongside oracle-routed proposals.",
    )
    parser.add_argument(
        "--prompt-regime",
        choices=("zero_shot", "static_few_shot"),
        default="zero_shot",
        help="Run the zero-shot contract or inject a deterministic prefix of the support bank.",
    )
    parser.add_argument(
        "--support-bank",
        default=None,
        help="Support-bank manifest required for --prompt-regime=static_few_shot.",
    )
    parser.add_argument("--a-box-examples", type=int, default=4)
    parser.add_argument("--t-box-examples", type=int, default=4)
    parser.add_argument("--diagnosis-examples", type=int, default=2)
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
        default=",".join(DEFAULT_ABLATION_BUNDLES),
        help=(
            "Comma-separated bundle names. Defaults to logic_only,local_graph. "
            "Include minimal_case explicitly if you want to run the no-context bundle."
        ),
    )
    args = parser.parse_args()

    configure_tbox_task_version(args.tbox_task_version)

    run_reasoning_floor(
        classified_path=args.classified_benchmark,
        world_state_path=args.world_state,
        output_dir=args.output_dir,
        resume_run_dir=args.resume_run_dir,
        model_name=args.model,
        model_endpoint=args.model_endpoint,
        reasoning_effort=args.reasoning_effort,
        context_length=args.context_length,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        ollama_think=args.ollama_think,
        max_retries=args.max_retries,
        protocol_path=args.protocol,
        model_digest=args.model_digest,
        generation_cache_path=args.generation_cache,
        execution_mode=args.execution_mode,
        proposal_track_mode=args.proposal_track_mode,
        oracle_diagnosis_mode=args.oracle_diagnosis_mode,
        prompt_regime=args.prompt_regime,
        support_bank_path=args.support_bank,
        a_box_example_count=args.a_box_examples,
        t_box_example_count=args.t_box_examples,
        diagnosis_example_count=args.diagnosis_examples,
        parallel_workers=args.parallel_workers,
        batch_completion_window=args.batch_completion_window,
        batch_poll_interval_seconds=args.batch_poll_interval_seconds,
        batch_sync_retry_fallback=args.batch_sync_retry_fallback,
        case_ids=[item.strip() for item in args.case_ids.split(",")] if args.case_ids else None,
        selection_manifest_path=args.selection_manifest,
        tracks=[item.strip() for item in args.tracks.split(",")] if args.tracks else None,
        max_cases=args.max_cases,
        ablation_bundles=[item.strip() for item in args.ablation_bundles.split(",") if item.strip()],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
