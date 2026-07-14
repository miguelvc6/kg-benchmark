from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

from kg_benchmark.audit.workflow import (
    AuditWorkflowError,
    audit_status,
    prepare_audit,
    run_deterministic_phase,
    run_finalize_phase,
    run_review_phase,
)
from kg_benchmark.dataset.release import (
    canonicalize_acquisition,
    fetch_dataset,
    promote_dataset,
    verify_dataset,
    write_source_provenance,
)
from kg_benchmark.methodology import (
    MethodologyError,
    check_methodology,
    create_methodology_lock,
    require_frozen_methodology,
)
from kg_benchmark.selection.workflow import (
    SelectionWorkflowError,
    expand_population,
    finalize_selection,
    prepare_reserve,
    review_reserve,
    selection_status,
)

LEGACY_COMMANDS = {
    "baseline": "non_llm_baselines",
    "run": "reasoning_floor",
    "score": "rescore_run",
}


def _add_audit_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cases", type=Path, default=Path("work/cases.jsonl"))
    parser.add_argument("--world-state", type=Path, default=Path("work/source/world-state.jsonl"))
    parser.add_argument("--stage2", type=Path, default=Path("work/source/repairs.jsonl"))
    parser.add_argument("--lineage-manifest", type=Path)
    parser.add_argument("--stage4-schema", type=Path, default=Path("schemas/dataset-case.schema.json"))
    parser.add_argument("--protocol", type=Path, default=Path("paper/protocol.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("work/audit"))


def _add_audit_review_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=600)


def _audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kg-benchmark audit")
    subparsers = parser.add_subparsers(dest="audit_command", required=True)
    prepare = subparsers.add_parser("prepare", help="Create the frozen construct sample and canonical prompt render.")
    _add_audit_prepare_arguments(prepare)
    deterministic = subparsers.add_parser("deterministic", help="Run exhaustive deterministic and temporal gates.")
    deterministic.add_argument("--work-dir", type=Path, default=Path("work/audit"))
    review = subparsers.add_parser("review", help="Run the protocol-bound label-hidden Codex review.")
    review.add_argument("--work-dir", type=Path, default=Path("work/audit"))
    _add_audit_review_arguments(review)
    finalize = subparsers.add_parser("finalize", help="Write conservative canonical dispositions and audit report.")
    finalize.add_argument("--work-dir", type=Path, default=Path("work/audit"))
    finalize.add_argument("--report", type=Path, default=Path("audit.md"))
    run = subparsers.add_parser("run", help="Run or resume every canonical audit phase in order.")
    _add_audit_prepare_arguments(run)
    _add_audit_review_arguments(run)
    run.add_argument("--report", type=Path, default=Path("audit.md"))
    status = subparsers.add_parser("status", help="Verify phase hashes and report the resumable audit position.")
    status.add_argument("--work-dir", type=Path, default=Path("work/audit"))
    return parser


def _run_audit_workflow(argv: list[str]) -> int:
    args = _audit_parser().parse_args(argv)
    repo_root = Path.cwd()
    if args.audit_command in {"review", "run"}:
        require_frozen_methodology(repo_root)
    if args.audit_command == "status":
        print(json.dumps(audit_status(work_dir=args.work_dir, repo_root=repo_root), indent=2, sort_keys=True))
        return 0
    if args.audit_command in {"prepare", "run"}:
        state = prepare_audit(
            cases_path=args.cases,
            world_state_path=args.world_state,
            stage2_path=args.stage2,
            lineage_manifest_path=args.lineage_manifest,
            stage4_schema_path=args.stage4_schema,
            protocol_path=args.protocol,
            work_dir=args.work_dir,
            repo_root=repo_root,
        )
        if args.audit_command == "prepare":
            print(json.dumps({"phase": "prepare", "counts": state["phases"]["prepare"]["counts"]}, indent=2))
            return 0
    if args.audit_command in {"deterministic", "run"}:
        state = run_deterministic_phase(work_dir=args.work_dir, repo_root=repo_root)
        if args.audit_command == "deterministic":
            print(json.dumps({"phase": "deterministic", "passed": state["phases"]["deterministic"]["passed"]}, indent=2))
            return 0 if state["phases"]["deterministic"]["passed"] else 1
    if args.audit_command in {"review", "run"}:
        state = run_review_phase(
            work_dir=args.work_dir,
            repo_root=repo_root,
            batch_size=args.batch_size,
            workers=args.workers,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
        )
        if args.audit_command == "review":
            print(json.dumps({"phase": "review", "reviewer": state["phases"]["review"]["reviewer"]}, indent=2))
            return 0
    state = run_finalize_phase(work_dir=args.work_dir, report_path=args.report, repo_root=repo_root)
    print(
        json.dumps(
            {
                "phase": "finalize",
                "dispositions": state["phases"]["finalize"]["artifacts"]["dispositions"],
                "summary": state["phases"]["finalize"]["artifacts"]["summary"],
                "report": state["phases"]["finalize"]["artifacts"]["release_report"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _contains_option(argv: list[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in argv)


def _run_acquire(argv: list[str]) -> int:
    if not any(value in {"-h", "--help"} for value in argv):
        require_frozen_methodology(Path.cwd())
    values = list(argv)
    if not _contains_option(values, "--data-dir"):
        values.extend(["--data-dir", "work/acquisition"])
    if not _contains_option(values, "--cache-dir"):
        values.extend(["--cache-dir", "work/cache"])
    if not _contains_option(values, "--dump-path"):
        values.extend(["--dump-path", "work/latest-all.json.gz"])
    result = _delegate("fetcher", ["acquire", *values])
    config_path = Path("work/acquisition-config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"command": "acquire", "arguments": values}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _run_build(argv: list[str]) -> int:
    if not any(value in {"-h", "--help"} for value in argv):
        require_frozen_methodology(Path.cwd())
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--work-dir", type=Path, default=Path("work"))
    wrapper.add_argument("--acquisition-dir", type=Path, default=Path("work/acquisition"))
    wrapper.add_argument("--dump-path", type=Path, default=Path("work/latest-all.json.gz"))
    wrapper.add_argument("--acquisition-config", type=Path, default=Path("work/acquisition-config.json"))
    wrapper_args, classifier_args = wrapper.parse_known_args(argv)
    acquisition = wrapper_args.acquisition_dir
    work = wrapper_args.work_dir
    defaults = {
        "--repairs-path": acquisition / "02_wikidata_repairs.json",
        "--world-state-path": acquisition / "03_world_state.json",
        "--popularity-path": acquisition / "00_entity_popularity.json",
        "--out-path": work / "cases.jsonl",
        "--stats-path": work / "audit" / "classifier-summary.json",
    }
    values = list(classifier_args)
    for option, path in defaults.items():
        if not _contains_option(values, option):
            values.extend([option, str(path)])
    if "--no-full-output" not in values:
        values.append("--no-full-output")
    result = _delegate("classifier", ["build", *values])
    if result:
        return result
    summary = canonicalize_acquisition(acquisition_dir=acquisition, work_dir=work)
    provenance = write_source_provenance(
        acquisition_dir=acquisition,
        work_dir=work,
        dump_path=wrapper_args.dump_path,
        acquisition_config_path=wrapper_args.acquisition_config,
    )
    print(
        json.dumps(
            {"built": True, "canonicalized_records": summary, "source_provenance": str(provenance)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _delegate(module_name: str, argv: list[str]) -> int:
    module = importlib.import_module(module_name)
    entrypoint: Callable[[], object] | None = getattr(module, "main", None)
    if entrypoint is None:
        raise RuntimeError(f"Delegated module {module_name} has no main().")
    previous = sys.argv
    try:
        sys.argv = [f"kg-benchmark {argv[0]}", *argv[1:]]
        result = entrypoint()
    finally:
        sys.argv = previous
    return int(result) if isinstance(result, int) else 0


def _selection_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kg-benchmark select")
    subparsers = parser.add_subparsers(dest="selection_command", required=True)
    reserve = subparsers.add_parser("reserve", help="Build the independent reserve and render/scan every prompt.")
    reserve.add_argument("--cases", type=Path, default=Path("work/cases.jsonl"))
    reserve.add_argument("--world-state", type=Path, default=Path("work/source/world-state.jsonl"))
    reserve.add_argument("--dispositions", type=Path, default=Path("work/audit/dispositions.jsonl"))
    reserve.add_argument("--audit-summary", type=Path, default=Path("work/audit/summary.json"))
    reserve.add_argument("--exclusions", type=Path, required=True)
    reserve.add_argument("--protocol", type=Path, default=Path("paper/protocol.json"))
    reserve.add_argument("--policy", type=Path, default=Path("paper/selection-policy.json"))
    reserve.add_argument("--output-dir", type=Path, default=Path("work/selections"))

    review = subparsers.add_parser("review", help="Run the fixed 50-case reserve temporal review with Codex.")
    review.add_argument("--output-dir", type=Path, default=Path("work/selections"))
    review.add_argument("--batch-size", type=int, default=10)
    review.add_argument("--workers", type=int, default=1)
    review.add_argument("--retries", type=int, default=2)
    review.add_argument("--timeout-seconds", type=float, default=600)

    finalize = subparsers.add_parser("finalize", help="Replace prompt failures and seal the 1,200/600 populations.")
    finalize.add_argument("--output-dir", type=Path, default=Path("work/selections"))

    expand = subparsers.add_parser("expand", help="Materialize a larger nested prompt-clean population.")
    expand.add_argument("--output-dir", type=Path, default=Path("work/selections"))
    expand.add_argument("--parent", type=Path, required=True)
    expand.add_argument("--name", required=True)
    expand.add_argument("--quota-ic-l", type=int, required=True)
    expand.add_argument("--quota-ic-g", type=int, required=True)
    expand.add_argument("--quota-ic-e-elim", type=int, required=True)
    expand.add_argument("--quota-tbox", type=int, required=True)
    expand.add_argument("--destination", type=Path, required=True)

    status = subparsers.add_parser("status", help="Verify reserve, review, and finalization artifact hashes.")
    status.add_argument("--output-dir", type=Path, default=Path("work/selections"))
    return parser


def _run_selection(argv: list[str]) -> int:
    args = _selection_parser().parse_args(argv)
    repo_root = Path.cwd()
    if args.selection_command == "reserve":
        state = prepare_reserve(
            cases_path=args.cases,
            world_state_path=args.world_state,
            dispositions_path=args.dispositions,
            audit_summary_path=args.audit_summary,
            exclusions_path=args.exclusions,
            protocol_path=args.protocol,
            policy_path=args.policy,
            output_dir=args.output_dir,
            repo_root=repo_root,
        )
        print(json.dumps({"phase": "reserve", "counts": state["phases"]["reserve"]["counts"]}, indent=2))
        return 0
    if args.selection_command == "review":
        require_frozen_methodology(repo_root)
        state = review_reserve(
            output_dir=args.output_dir,
            repo_root=repo_root,
            batch_size=args.batch_size,
            workers=args.workers,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps({"phase": "review", "reviewer": state["phases"]["review"]["reviewer"]}, indent=2))
        return 0
    if args.selection_command == "finalize":
        state = finalize_selection(output_dir=args.output_dir, repo_root=repo_root)
        print(json.dumps({"phase": "finalize", "counts": state["phases"]["finalize"]["counts"]}, indent=2))
        return 0
    if args.selection_command == "status":
        print(json.dumps(selection_status(output_dir=args.output_dir, repo_root=repo_root), indent=2, sort_keys=True))
        return 0
    manifest = expand_population(
        output_dir=args.output_dir,
        parent_path=args.parent,
        name=args.name,
        requested_quotas={
            "IC-L": args.quota_ic_l,
            "IC-G": args.quota_ic_g,
            "IC-E-elim": args.quota_ic_e_elim,
            "TBOX": args.quota_tbox,
        },
        destination=args.destination,
        repo_root=repo_root,
    )
    print(json.dumps({"name": manifest["name"], "case_count": manifest["case_count"], "quotas": manifest["quotas"]}, indent=2))
    return 0


def _run_promote(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="kg-benchmark promote")
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--protocol", type=Path, default=Path("paper/protocol.json"))
    parser.add_argument("--source-provenance", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = promote_dataset(
        work_dir=args.work_dir,
        dataset_dir=args.dataset_dir,
        protocol_path=args.protocol,
        source_provenance_path=args.source_provenance,
    )
    print(json.dumps({"promoted": True, "dataset_id": manifest["dataset_id"]}, indent=2))
    return 0


def _run_verify(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="kg-benchmark verify")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    args = parser.parse_args(argv)
    print(json.dumps(verify_dataset(args.dataset_dir), indent=2, sort_keys=True))
    return 0


def _run_fetch(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="kg-benchmark fetch")
    parser.add_argument("--manifest-url", required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    args = parser.parse_args(argv)
    result = fetch_dataset(
        manifest_url=args.manifest_url,
        manifest_sha256=args.manifest_sha256,
        dataset_dir=args.dataset_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_viewer(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="kg-benchmark viewer")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(repository_root / "scripts" / "reasoning_floor_viewer.py"),
    ]
    if args.server_port is not None:
        command.extend(["--server.port", str(args.server_port)])
    command.extend(
        [
            "--",
            "--reports-root",
            args.runs_dir,
            "--classified-benchmark",
            str(args.dataset_dir / "cases.jsonl"),
            "--world-state",
            str(args.dataset_dir / "source" / "world-state.jsonl"),
        ]
    )
    return subprocess.run(command, check=False).returncode


def _methodology_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kg-benchmark methodology")
    subparsers = parser.add_subparsers(dest="methodology_command", required=True)
    check = subparsers.add_parser("check", help="Validate the methodology candidate and report freeze blockers.")
    check.add_argument("--repo-root", type=Path, default=Path("."))
    check.add_argument("--require-freeze-ready", action="store_true")
    check.add_argument("--output", type=Path)
    freeze = subparsers.add_parser("freeze", help="Create the final methodology lock after all blockers are resolved.")
    freeze.add_argument("--repo-root", type=Path, default=Path("."))
    freeze.add_argument("--output", type=Path)
    return parser


def _run_methodology(argv: list[str]) -> int:
    args = _methodology_parser().parse_args(argv)
    if args.methodology_command == "freeze":
        lock = create_methodology_lock(args.repo_root, args.output)
        print(json.dumps(lock, indent=2, sort_keys=True))
        return 0
    report = check_methodology(args.repo_root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["valid"] and (not args.require_freeze_ready or report["freeze_ready"]) else 1


def _main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        print(
            "usage: kg-benchmark {methodology,acquire,build,audit,select,promote,fetch,verify,run,score,baseline,viewer} ...\n\n"
            "One paper-facing command for dataset construction, verification, execution, and inspection."
        )
        return 0
    command = values[0]
    rest = values[1:]
    if command == "methodology":
        return _run_methodology(rest)
    if command == "select":
        return _run_selection(rest)
    if command == "audit":
        return _run_audit_workflow(rest)
    if command == "acquire":
        return _run_acquire(rest)
    if command == "build":
        return _run_build(rest)
    if command == "promote":
        return _run_promote(rest)
    if command == "verify":
        return _run_verify(rest)
    if command == "fetch":
        return _run_fetch(rest)
    if command == "viewer":
        return _run_viewer(rest)
    module_name = LEGACY_COMMANDS.get(command)
    if module_name is None:
        raise SystemExit(f"Unknown command: {command}")
    if command == "run" and not any(value in {"-h", "--help"} for value in rest):
        require_frozen_methodology(Path.cwd())
    return _delegate(module_name, [command, *rest])


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except (MethodologyError, AuditWorkflowError, SelectionWorkflowError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
