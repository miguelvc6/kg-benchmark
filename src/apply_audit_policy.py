#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lib.manual_audit import apply_audit_policy, read_annotation_csv, write_audit_policy_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Phase D manual-audit recommendations to produce an audit-informed policy report."
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless every audit row has complete human annotation fields.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_annotation_csv(Path(args.annotations))
    policy = apply_audit_policy(rows, require_complete=args.require_complete)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_audit_policy_markdown(policy, Path(args.out_md))
    print(f"[done] wrote {args.out_json}")
    print(f"[done] wrote {args.out_md}")
    print(f"[done] rows={policy['completion']['row_count']:,} complete={policy['completion']['complete_row_count']:,}")
    print(f"[done] status={policy['status']}")
    for warning in policy["warnings"]:
        print(f"[warning] {warning}")
    if args.require_complete and policy["status"] != "ready":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
