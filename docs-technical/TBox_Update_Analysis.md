# T-BOX Update Analysis

`src/analyze_tbox_updates.py` summarizes how many Stage 4 `T_BOX` benchmark cases point back to the same property-level schema edit.

This answers the question "which T-BOX updates generated these repairs?" more directly than the raw T-BOX case count, because many violations can map to a single `repair_target.property_revision_id`.

For the deterministic capped paper subset derived from this analysis, see [Benchmark Selection](./Benchmark_Selection.md).

## Inputs

- `data/04_classified_benchmark.jsonl`

## Outputs

The script writes a small report bundle under `reports/tbox_update_analysis/` by default:

- `tbox_update_frequency.csv`: one row per `property_revision_id`
- `tbox_survival_by_cap.csv`: survival curve after capping each property revision at `N` repairs
- `summary.json`: aggregate counts plus the top-N revisions
- `tbox_update_frequency_top.svg`: bar chart for the most frequent revisions
- `tbox_survival_by_cap.svg`: line chart for `sum(min(case_count, N))` from `N=max_cap` down to `0`

## Usage

```bash
uv run python src/analyze_tbox_updates.py \
  --input data/04_classified_benchmark.jsonl \
  --output-dir reports/tbox_update_analysis \
  --top-n 25 \
  --max-cap 1000
```

## Memory behavior

The analyzer is intentionally streaming:

- it reads the JSONL file one line at a time
- it only fully decodes JSON for the first occurrence of each distinct T-BOX property revision
- it keeps counters and compact per-revision metadata in memory

It does not load the full benchmark into RAM, which keeps it suitable for multi-gigabyte Stage 4 artifacts.
