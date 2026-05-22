#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from lib.manual_audit import (
    DEFAULT_AUDIT_SEED,
    AuditBuildOptions,
    build_audit_sample,
    write_audit_csv,
    write_audit_jsonl,
    write_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase D manual-audit sample.")
    parser.add_argument("--classified-benchmark", default="data/04_classified_benchmark.jsonl")
    parser.add_argument("--core-manifest", required=True)
    parser.add_argument("--dev-manifest")
    parser.add_argument("--seed", type=int, default=DEFAULT_AUDIT_SEED)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-schema", default="reports/manual_audit/audit_annotation_schema.json")
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, metadata = build_audit_sample(
        AuditBuildOptions(
            classified_benchmark=Path(args.classified_benchmark),
            core_manifest=Path(args.core_manifest),
            dev_manifest=Path(args.dev_manifest) if args.dev_manifest else None,
            seed=args.seed,
            progress_every=args.progress_every,
        )
    )
    write_audit_jsonl(rows, Path(args.output_jsonl))
    write_audit_csv(rows, Path(args.output_csv))
    write_schema(Path(args.output_schema), metadata)

    print(f"[done] wrote {args.output_jsonl}")
    print(f"[done] wrote {args.output_csv}")
    print(f"[done] wrote {args.output_schema}")
    print(f"[done] selected={metadata['counts']['selected']:,}")
    print(f"[done] dev_overlap={metadata['counts']['dev_overlap']:,}")
    print(f"[done] max_tbox_per_revision={metadata['counts']['max_tbox_per_revision']:,}")
    print(f"[done] max_abox_per_group={metadata['counts']['max_abox_per_group']:,}")
    print(f"[done] underfilled_quotas={len(metadata['underfilled_quotas'])}")
    for warning in metadata["warnings"]:
        print(f"[warning] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
