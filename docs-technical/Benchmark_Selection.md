# Benchmark Selection

This document describes the deterministic case-selection manifest used when paper-facing runs should evaluate a capped subset of the full benchmark without duplicating the full Stage 4 JSONL artifact.

The canonical benchmark artifact remains `data/04_classified_benchmark.jsonl`.

The selection manifest is a frozen evaluation view over that artifact, not a replacement for it.

## Current Paper Policy

The current paper subset keeps:

- all `A_BOX` cases
- at most `100` `T_BOX` cases per `repair_target.property_revision_id`

The T-box cap is applied independently within each property revision group.

## Deterministic Ordering

Within each T-box update group, cases are ranked by:

- `sha1(seed|property_revision_id|case_id)` in ascending order

This avoids dependence on JSONL row order and avoids Python's process-randomized `hash()` behavior.

The default seed is `13`.

## Selector Script

The selector entry point is `src/select_benchmark_cases.py`.

Default usage:

```bash
uv run python src/select_benchmark_cases.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --output reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json \
  --tbox-cap-per-update 100 \
  --seed 13
```

The output JSON includes:

- input provenance
- selection policy and seed
- aggregate counts
- per-update selected-count summaries for T-box groups
- `selected_case_ids`

## Runtime Integration

Both runtime entry points accept the manifest directly:

- `src/reasoning_floor.py --selection-manifest ...`
- `src/evaluate.py --selection-manifest ...`

When a selection manifest is supplied:

- the runner generates only those cases
- the evaluator scores only those cases
- explicit `--case-ids` are intersected with the manifest rather than replacing it

## Why This Is a Manifest, Not a Second Benchmark Artifact

The manifest approach preserves:

- one canonical Stage 4 benchmark artifact
- reproducible paper subsets
- low disk overhead
- the ability to derive multiple frozen subsets from the same benchmark later

This also keeps the benchmark core independent from protocol-specific runtime choices.
