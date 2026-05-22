# Benchmark Selection

This document describes the Phase C deterministic case-selection manifests used when paper-facing runs should evaluate fixed core/dev tiers without duplicating the full Stage 4 JSONL artifact.

The canonical benchmark artifact remains `data/04_classified_benchmark.jsonl`.

The selection manifest is a frozen evaluation view over that artifact, not a replacement for it.

## Current Paper Policy

The current paper policy writes manifests only:

- `reports/benchmark_selection/dev_prompt_v1_seed_13.json`
- `reports/benchmark_selection/core_v1_seed_13.json`

Core v1 targets 4,800 cases and dev/pilot v1 targets 600 cases. Core excludes dev case ids and dev T-box property-revision groups. A-box `(qid, property)` overlap with dev is avoided when quotas permit and recorded as a warning when unavoidable.

## Deterministic Ordering

Within each selection stratum, cases are ranked by:

- `sha1(seed|tier|selection_stratum|group_key|case_id)` in ascending order

This avoids dependence on JSONL row order and avoids Python's process-randomized `hash()` behavior.

The default seed is `13`.

Selection uses popularity-aware round-robin over `head`, `mid`, `tail`, and `unknown` buckets. Empty buckets are skipped.

## Selector Script

The selector entry point is `src/select_benchmark_cases.py`.

Build the dev manifest first:

```bash
uv run python src/select_benchmark_cases.py \
  --tier dev \
  --output reports/benchmark_selection/dev_prompt_v1_seed_13.json
```

Then build the core manifest against the dev exclusions:

```bash
uv run python src/select_benchmark_cases.py \
  --tier core \
  --exclude-manifest reports/benchmark_selection/dev_prompt_v1_seed_13.json \
  --output reports/benchmark_selection/core_v1_seed_13.json
```

The output JSON includes:

- input provenance
- selection policy and seed
- aggregate counts
- `selected_case_ids`
- `main_score_case_ids`
- `diagnostic_case_ids`
- per-case derived annotations
- validation checks and warnings

`selected_case_ids` ordering is configurable:

- `sorted` keeps the current global `case_id` ordering
- `shuffled` applies a deterministic seeded hash order across the final selected ids

Optional selector controls:

```bash
uv run python src/select_benchmark_cases.py \
  --tier core \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --exclude-manifest reports/benchmark_selection/dev_prompt_v1_seed_13.json \
  --output reports/benchmark_selection/core_v1_seed_13.json \
  --seed 13 \
  --core-size 4800 \
  --dev-size 600 \
  --tbox-cap-core 10 \
  --tbox-cap-dev 3 \
  --abox-cap-core 3 \
  --abox-cap-dev 2 \
  --selected-case-order shuffled
```

## Runtime Integration

Both runtime entry points accept the manifest directly:

- `src/reasoning_floor.py --selection-manifest ...`
- `src/evaluate.py --selection-manifest ...`

When a selection manifest is supplied:

- the runner generates only those cases
- the evaluator scores only those cases
- explicit `--case-ids` are intersected with the manifest rather than replacing it
- manifest order is preserved end-to-end
- `--max-cases` truncates after manifest ordering is resolved, not after a file-order scan of the benchmark

Evaluation aggregation should use `main_score_case_ids` for headline scores and report `diagnostic_case_ids` separately. `EXTERNAL_BY_ELIMINATION` is an IC-E-elim/no-retrieval stress label, not confirmed external evidence; `UNKNOWN_*` TypeC cases are IC-U diagnostics.

## Why This Is a Manifest, Not a Second Benchmark Artifact

The manifest approach preserves:

- one canonical Stage 4 benchmark artifact
- reproducible paper subsets
- low disk overhead
- the ability to derive multiple frozen subsets from the same benchmark later

This also keeps the benchmark core independent from protocol-specific runtime choices.

## Ordered Materialization

The reasoning-floor runner now materializes the final selected generation subset once per run.

- small selections stay in memory
- larger selections are written to `selected_generation_records.jsonl` in the run directory and then streamed from that ordered subset

This guarantees that every ablation bundle sees the same ordered case subset and that later evaluation artifacts can be traced back to the exact generated selection.
