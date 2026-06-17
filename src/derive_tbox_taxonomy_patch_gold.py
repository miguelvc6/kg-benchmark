#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys

from lib.tbox_taxonomy_patch_gold import CoverageError, extract_selected_tbox_gold, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive deterministic T-box taxonomy patch gold.")
    parser.add_argument("--classified-benchmark", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--require-coverage", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = extract_selected_tbox_gold(
            classified_benchmark=args.classified_benchmark,
            selection_manifest=args.selection_manifest,
            require_coverage=args.require_coverage,
        )
    except CoverageError as exc:
        write_json(args.out_summary, exc.summary)
        print(json.dumps(exc.summary, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    write_jsonl(args.out_jsonl, result.patches)
    write_json(args.out_summary, result.summary)
    print(f"[done] wrote {args.out_jsonl}")
    print(f"[done] wrote {args.out_summary}")
    print(
        "[done] selected_tbox_records={selected_tbox_records} gold_extracted={gold_extracted} unsupported={unsupported_count}".format(
            **result.summary
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
