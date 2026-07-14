from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

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
from kg_benchmark.selection.extensible import build_selection_artifacts, materialize_population

LEGACY_COMMANDS = {
    "audit": "automated_consistency_audit",
    "baseline": "non_llm_baselines",
    "run": "reasoning_floor",
    "score": "rescore_run",
}


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
    build = subparsers.add_parser("build", help="Build ordering, support bank, and frozen 1,200/600 populations.")
    build.add_argument("--cases", type=Path, required=True)
    build.add_argument("--dispositions", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, default=Path("work/selections"))
    build.add_argument("--seed", type=int, default=13)
    build.add_argument("--support-capacity", type=int, default=16)
    build.add_argument("--main-quota-ic-l", type=int, default=230)
    build.add_argument("--main-quota-ic-g", type=int, default=375)
    build.add_argument("--main-quota-ic-e-elim", type=int, default=295)
    build.add_argument("--main-quota-tbox", type=int, default=300)
    build.add_argument("--api-quota-ic-l", type=int, default=115)
    build.add_argument("--api-quota-ic-g", type=int, default=188)
    build.add_argument("--api-quota-ic-e-elim", type=int, default=147)
    build.add_argument("--api-quota-tbox", type=int, default=150)

    expand = subparsers.add_parser("expand", help="Materialize a nested population from the frozen eligibility order.")
    expand.add_argument("--eligibility-order", type=Path, required=True)
    expand.add_argument("--support-bank", type=Path, required=True)
    expand.add_argument("--parent", type=Path)
    expand.add_argument("--name", required=True)
    expand.add_argument("--quota-ic-l", type=int, required=True)
    expand.add_argument("--quota-ic-g", type=int, required=True)
    expand.add_argument("--quota-ic-e-elim", type=int, required=True)
    expand.add_argument("--quota-tbox", type=int, required=True)
    expand.add_argument("--output", type=Path, required=True)
    return parser


def _run_selection(argv: list[str]) -> int:
    args = _selection_parser().parse_args(argv)
    if args.selection_command == "build":
        summary = build_selection_artifacts(
            cases_path=args.cases,
            dispositions_path=args.dispositions,
            output_dir=args.output_dir,
            seed=args.seed,
            support_capacity=args.support_capacity,
            main_quotas={
                "IC-L": args.main_quota_ic_l,
                "IC-G": args.main_quota_ic_g,
                "IC-E-elim": args.main_quota_ic_e_elim,
                "TBOX": args.main_quota_tbox,
            },
            api_quotas={
                "IC-L": args.api_quota_ic_l,
                "IC-G": args.api_quota_ic_g,
                "IC-E-elim": args.api_quota_ic_e_elim,
                "TBOX": args.api_quota_tbox,
            },
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    eligibility = [json.loads(line) for line in args.eligibility_order.read_text(encoding="utf-8").splitlines() if line]
    support_bank = json.loads(args.support_bank.read_text(encoding="utf-8"))
    parent = json.loads(args.parent.read_text(encoding="utf-8")) if args.parent else None
    manifest = materialize_population(
        name=args.name,
        quotas={
            "IC-L": args.quota_ic_l,
            "IC-G": args.quota_ic_g,
            "IC-E-elim": args.quota_ic_e_elim,
            "TBOX": args.quota_tbox,
        },
        eligibility_rows=eligibility,
        support_bank=support_bank,
        parent_manifest=parent,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    except MethodologyError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
