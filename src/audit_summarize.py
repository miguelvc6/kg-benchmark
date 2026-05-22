#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lib.manual_audit import read_annotation_csv, summarize_annotations, write_summary_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase D manual-audit annotations.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_annotation_csv(Path(args.annotations))
    summary = summarize_annotations(rows)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_markdown(summary, Path(args.out_md))
    print(f"[done] wrote {args.out_json}")
    print(f"[done] wrote {args.out_md}")
    print(f"[done] rows={summary['row_count']:,} unannotated={summary['unannotated_row_count']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
