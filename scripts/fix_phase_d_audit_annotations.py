#!/usr/bin/env python3
"""Apply deterministic cleanups to the Phase D manual-audit annotation CSV.

This script intentionally performs only mechanical consistency fixes. It does not
invent annotations for unannotated rows and it does not inspect external evidence.

Default input/output:
    reports/manual_audit/audit_phase_d_v1_seed_13_annotated.csv

Fixes applied:
1. LOCAL_SELECTION_CONFIRMED rows annotated as local_derived_confirmed are changed
   to local_confirmed, because selection-confirmed evidence is direct independent
   local support, not deterministic derivation.
2. TBOX_UNKNOWN_TBOX_CAUSALITY rows annotated as coincidental_or_weak are changed
   to unknown_causality, matching the stratum, classifier subtype, and summarizer
   semantics.
3. EXTERNAL_BY_ELIMINATION rows with typec_judgment=external_by_elimination_ok
   and external_evidence_required=yes are made conservative by changing
   external_evidence_required to maybe. This avoids claiming confirmed external
   evidence without retrieval/manual source confirmation.
4. Annotated rows with an annotator_id but blank annotation_timestamp_utc receive
   a deterministic timestamp supplied by --timestamp.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

DEFAULT_CSV = Path("reports/manual_audit/audit_phase_d_v1_seed_13_annotated.csv")
DEFAULT_TIMESTAMP = "2026-06-03T10:00:25Z"
ANNOTATION_FIELDS = [
    "repair_locus_correct",
    "historical_target_well_defined",
    "target_visible_locally",
    "extractor_missed_local_evidence",
    "external_evidence_required",
    "typec_judgment",
    "typea_judgment",
    "typeb_judgment",
    "tbox_judgment",
    "core_recommendation",
]


def _is_annotated(value: object) -> bool:
    return str(value or "").strip() not in {"", "unannotated"}


def _row_has_human_annotation(row: dict[str, str]) -> bool:
    return any(_is_annotated(row.get(field)) for field in ANNOTATION_FIELDS)


def fix_rows(rows: list[dict[str, str]], *, timestamp: str) -> tuple[list[dict[str, str]], Counter[str]]:
    counts: Counter[str] = Counter()

    for row in rows:
        # 1. TypeB local-selection cases are direct local confirmations, not derived confirmations.
        if (
            row.get("selection_stratum") == "TypeB_LOCAL_SELECTION_CONFIRMED"
            and row.get("typeb_judgment") == "local_derived_confirmed"
        ):
            row["typeb_judgment"] = "local_confirmed"
            counts["typeb_local_selection_to_local_confirmed"] += 1

        # 2. Unknown-causality T-box cases should use the matching judgment value.
        if (
            row.get("selection_stratum") == "TBOX_UNKNOWN_TBOX_CAUSALITY"
            and row.get("tbox_judgment") == "coincidental_or_weak"
        ):
            row["tbox_judgment"] = "unknown_causality"
            counts["tbox_unknown_to_unknown_causality"] += 1

        # 3. Conservative TypeC policy: external-by-elimination is not confirmed external evidence.
        if (
            row.get("class") == "TypeC"
            and row.get("subtype") == "EXTERNAL_BY_ELIMINATION"
            and row.get("typec_judgment") == "external_by_elimination_ok"
            and row.get("external_evidence_required") == "yes"
        ):
            row["external_evidence_required"] = "maybe"
            if row.get("notes"):
                row["notes"] = row["notes"].replace(" likely needs ", " may need ")
                row["notes"] = row["notes"].replace("likely needs external/domain/provenance evidence", "may need external/domain/provenance evidence")
            counts["typec_external_required_yes_to_maybe"] += 1

        # 4. Fill timestamp only for already annotated rows with annotator provenance.
        if (
            _row_has_human_annotation(row)
            and str(row.get("annotator_id") or "").strip()
            and not str(row.get("annotation_timestamp_utc") or "").strip()
        ):
            row["annotation_timestamp_utc"] = timestamp
            counts["blank_annotation_timestamp_filled"] += 1

    return rows, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV, help="CSV to read")
    parser.add_argument("--output", type=Path, default=None, help="CSV to write; defaults to --input with --in-place")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input CSV")
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP, help="Timestamp for annotated rows missing annotation_timestamp_utc")
    args = parser.parse_args()

    if args.output is None and not args.in_place:
        parser.error("provide --output or use --in-place")
    output = args.input if args.in_place else args.output
    assert output is not None

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise SystemExit(f"No header found in {args.input}")

    rows, counts = fix_rows(rows, timestamp=args.timestamp)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output}")
    print("Applied fixes:")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")
    if not counts:
        print("  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
