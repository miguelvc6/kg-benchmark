# kg-benchmark

WikidataRepairEval 1.0 is a benchmark of real Wikidata repair events with frozen
context for evaluating knowledge-graph repair and retrieval needs. The pipeline
reconstructs historical fixes, attaches a 2026 world-state snapshot, labels each
case by information necessity (Type A/B/C), and supports downstream evaluation
and reasoning-floor runs.

## Pipeline overview

Stage 1 - Index candidate repairs from Wikidata constraint report diffs.
Stage 2 - Fetch the atomic edit and capture provenance plus A-box/T-box track.
Stage 3 - Build frozen 2026 graph context layers (L1-L4).
Stage 4 - Classify cases into the benchmark taxonomy and emit audit traces.
Stage 5 - Generate deterministic train/dev/test splits.
Stage 6 - Evaluate normalized proposals against the frozen benchmark artifacts.
Stage 7 - Run the zero-shot reasoning-floor baseline, including track diagnosis.

## Documentation map

- `docs-conceptual/`: research framing, benchmark intent, hypotheses, taxonomy, and evaluation goals
- `docs-technical/`: repository structure, scripts, artifacts, schemas, and runtime behavior
- `docs-public/`: shorter public-facing benchmark guidance and release-facing usage notes
- `docs/README.md`: pointer into the primary conceptual and technical documentation areas

Use the conceptual docs when research decisions change. Use the technical docs
when code paths, CLI behavior, artifacts, or engineering choices change.

## Repository layout

- `src/fetcher.py`: stages 1-3 pipeline entry point
- `src/classifier.py`: stage 4 taxonomy labeler
- `src/splitter.py`: stage 5 train/dev/test split generation
- `src/select_benchmark_cases.py`: deterministic paper-subset selection manifest builder
- `src/evaluate.py`: benchmark evaluation entry point
- `src/reasoning_floor.py`: zero-shot baseline runner
- `src/reasoning_floor_viewer.py`: Streamlit viewer for reasoning-floor runs
- `src/analyze_tbox_updates.py`: frequency analysis for Stage 4 T-box update groups
- `src/lib/`: shared pipeline modules
- `src/guardian/`: proposal normalization, evaluation, and reasoning-floor support modules
- `data/`: generated artifacts and caches
- `data_sample/`: sample artifacts for quick inspection and dry runs
- `reports/`: run outputs, benchmark-selection manifests, and summaries
- `schemas/`: JSON schemas for benchmark and proposal artifacts
- `tests/`: parser, evaluator, selection, and reasoning-floor coverage

## Setup

This repository uses `uv` and `pyproject.toml`.

```bash
# Windows
set UV_PROJECT_ENVIRONMENT=.venv
uv sync

# WSL/Linux
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync
```

Optional extras:

- `uv sync --extra dev` for `pytest` and `ruff`
- `uv sync --extra ui` for the Streamlit reasoning-floor viewer
- `uv sync --extra analysis` for plotting and analysis dependencies
- `uv sync --extra notebook` for Jupyter and notebook support

Command convention: prefer `uv run python ...` for repository commands. Bare
`python` is not guaranteed to exist on `PATH`; use `python3` only for ad hoc
commands outside the project's `uv` environment.

The project metadata also installs console entry points such as `kg-fetcher`,
`kg-classifier`, `kg-evaluate`, and `kg-reasoning-floor`, and it packages the
top-level modules so `uv run python -m fetcher`-style execution works in
editable installs and built artifacts.

If you plan to run the reasoning floor, create a local `.env` from
`.env.example` and set the provider variables there. The current runtime
supports `MODEL_PROVIDER=openai` and `MODEL_PROVIDER=ollama`.

## Common workflows

Inspect the live CLI surfaces:

```bash
uv run python src/fetcher.py --help
uv run python src/classifier.py --help
uv run python src/evaluate.py --help
uv run python src/reasoning_floor.py --help
```

Run stages 1-3:

```bash
uv run python src/fetcher.py
```

Debug or resume the fetcher:

```bash
uv run python src/fetcher.py --max-candidates 100
uv run python src/fetcher.py --resume-stats logs/fetcher_stats_YYYYMMDDTHHMMSS.jsonl
uv run python src/fetcher.py --resume-checkpoint logs/resume_checkpoint_YYYYMMDDTHHMMSS.json
uv run python src/fetcher.py --reuse-popularity-artifact
uv run python src/fetcher.py --validate-only
```

Run the benchmark classifier and split generation on sample artifacts:

```bash
uv run python src/classifier.py --sample
uv run python src/splitter.py --sample
uv run python src/classifier.py --self-test
```

Build a deterministic benchmark-selection manifest for paper-facing runs:

```bash
uv run python src/select_benchmark_cases.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --output reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json \
  --tbox-cap-per-update 100 \
  --seed 13
```

Run or resume the reasoning floor:

```bash
uv run python src/reasoning_floor.py \
  --selection-manifest reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json

uv run python src/reasoning_floor.py \
  --resume-run-dir reports/reasoning_floor/<RUN_ID>_<provider>_<model>
```

Evaluate proposal artifacts against the benchmark:

```bash
uv run python src/evaluate.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --selection-manifest reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json \
  --a-box-proposals <path/to/a_box_proposals.jsonl> \
  --t-box-proposals <path/to/t_box_proposals.jsonl> \
  --out-traces reports/evaluation_traces.jsonl \
  --out-summary reports/evaluation_summary.json
```

Launch the reasoning-floor viewer:

```bash
uv sync --extra ui
uv run streamlit run src/reasoning_floor_viewer.py -- --reports-root reports/reasoning_floor
```

Analyze which property-level T-box updates dominate Stage 4:

```bash
uv run python src/analyze_tbox_updates.py --input data/04_classified_benchmark.jsonl
```

Run tests:

```bash
uv sync --extra dev
uv run pytest
```

## Key artifacts

- `data/01_repair_candidates.json`: candidate repair events from report diffs
- `data/02_wikidata_repairs.json` and `data/02_wikidata_repairs.jsonl`: atomic repair events with provenance
- `data/03_world_state.json`: frozen 2026 context keyed by repair id
- `data/04_classified_benchmark.jsonl`: lean Stage 4 benchmark artifact
- `data/04_classified_benchmark_full.jsonl`: Stage 4 benchmark artifact with embedded context
- `reports/benchmark_selection/*.json`: deterministic selection manifests with `selected_case_ids`
- `reports/reasoning_floor/<run_id>_<provider>_<model>/`: reasoning-floor outputs, manifests, and bundle summaries
- `reports/evaluation_traces.jsonl`: per-case evaluator output
- `reports/evaluation_summary.json`: aggregate evaluator summary
- `reports/classifier_stats.json`: classifier counts and diagnostics

Sample outputs live under `data_sample/`, including:

- `data_sample/03_world_state.json`
- `data_sample/04_classified_benchmark.jsonl`
- `data_sample/classified_benchmark_sample.json`

## Notes

- The fetcher hits the live Wikidata API and can take days on a full run.
- Fetcher runs write large caches under `data/cache/` and resume state under `logs/`.
- Stage 3 requires the Wikidata dump at `data/latest-all.json.gz`.
- OpenAI reasoning-floor runs default to batch execution; other providers default to synchronous execution unless overridden.
- For deeper implementation detail, start with `docs-technical/README.md`. For benchmark rationale, start with `docs-conceptual/README.md`. For external-facing benchmark usage, start with `docs-public/README.md`.
