# Paper Execution Plan

This runbook is the command-by-command sequence for producing the benchmark artifacts and zero-shot paper outputs from the current repository.

The commands below assume a bash shell on Linux or WSL and a full build, not a sample run.

## What This Produces

Running this plan produces the current paper-relevant artifacts:

- `data/00_entity_popularity.json`
- `data/01_repair_candidates.json`
- `data/02_wikidata_repairs.json`
- `data/02_wikidata_repairs.jsonl`
- `data/03_world_state.json`
- `data/04_classified_benchmark.jsonl`
- `data/04_classified_benchmark_full.jsonl`
- `data/05_splits.json`
- `reports/classifier_stats.json`
- `reports/reasoning_floor/<RUN_ID>/raw_model_responses.jsonl`
- `reports/reasoning_floor/<RUN_ID>/run_manifest.jsonl`
- `reports/reasoning_floor/<RUN_ID>/<bundle>/a_box_proposals.jsonl`
- `reports/reasoning_floor/<RUN_ID>/<bundle>/t_box_proposals.jsonl`
- `reports/reasoning_floor/<RUN_ID>/<bundle>/track_diagnoses.jsonl`
- `reports/reasoning_floor/<RUN_ID>/<bundle>/evaluation_traces.jsonl`
- `reports/reasoning_floor/<RUN_ID>/<bundle>/evaluation_summary.json`
- `reports/reasoning_floor/<RUN_ID>/reasoning_floor_summary.json`

The reasoning-floor run already performs evaluation, so no extra command is required to obtain paper-ready baseline summaries.

## Prerequisites

You need:

- `uv`
- network access to Wikidata and the model provider
- a local Wikidata dump available as `data/latest-all.json.gz`
- OpenAI-compatible credentials for the reasoning-floor run

If your dump is elsewhere on disk, link it into the repository path used by the code.

## 1. Environment Setup

```bash
cd /home/mvazquez/kg-benchmark

export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync

mkdir -p data reports logs
```

If your dump is not already present at `data/latest-all.json.gz`, create a symlink:

```bash
ln -sf /absolute/path/to/latest-all.json.gz data/latest-all.json.gz
```

## 2. Sanity Checks Before Full Build

```bash
uv run python src/classifier.py --self-test
uv run python -m unittest discover -s tests
```

## 3. Build the Benchmark Artifacts

This single command performs:

- Stage 1 candidate mining
- Stage 2 repair reconstruction
- Stage 3 popularity enrichment
- Stage 3 world-state build and validation

```bash
uv run python src/fetcher.py
```

Validate the generated world state explicitly once the fetcher run completes:

```bash
uv run python src/fetcher.py --validate-only
```

If the fetcher run is interrupted, resume with one of the following:

```bash
uv run python src/fetcher.py --resume-stats logs/fetcher_stats_YYYYMMDDTHHMMSS.jsonl
```

or

```bash
uv run python src/fetcher.py --resume-checkpoint logs/resume_checkpoint_YYYYMMDDTHHMMSS.json
```

If `data/00_entity_popularity.json` is already valid and you want to reuse it:

```bash
uv run python src/fetcher.py --reuse-popularity-artifact
```

## 4. Classify the Benchmark

```bash
uv run python src/classifier.py
```

This produces:

- `data/04_classified_benchmark.jsonl`
- `data/04_classified_benchmark_full.jsonl`
- `reports/classifier_stats.json`

## 5. Build Deterministic Splits

```bash
uv run python src/splitter.py
```

This produces:

- `data/05_splits.json`

## 6. Configure the Model Provider

The reasoning-floor runner currently expects an OpenAI-compatible provider.

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
export OPENAI_MODEL='YOUR_MODEL_NAME'
```

Optional, only if you are using a non-default OpenAI-compatible endpoint:

```bash
export OPENAI_BASE_URL='https://your-compatible-endpoint/v1'
```

## 7. Run the Zero-Shot Reasoning Floor

Create a timestamped output directory for the paper run:

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
```

Run the full zero-shot baseline over the classified benchmark:

```bash
uv run python src/reasoning_floor.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --output-dir reports/reasoning_floor/$RUN_ID
```

This runs all three ablation bundles by default:

- `minimal_case`
- `logic_only`
- `local_graph`

It also performs:

- A-box proposal generation
- T-box proposal generation
- A-box vs T-box track diagnosis
- evaluation of all generated outputs

## 8. Verify the Paper Outputs

Check that the expected outputs exist:

```bash
find reports/reasoning_floor/$RUN_ID -maxdepth 2 -type f | sort
```

Inspect the combined paper summary:

```bash
sed -n '1,240p' reports/reasoning_floor/$RUN_ID/reasoning_floor_summary.json
```

Inspect the per-bundle summaries:

```bash
sed -n '1,240p' reports/reasoning_floor/$RUN_ID/minimal_case/evaluation_summary.json
sed -n '1,240p' reports/reasoning_floor/$RUN_ID/logic_only/evaluation_summary.json
sed -n '1,240p' reports/reasoning_floor/$RUN_ID/local_graph/evaluation_summary.json
```

Inspect the benchmark summary artifacts:

```bash
sed -n '1,240p' reports/classifier_stats.json
sed -n '1,240p' data/05_splits.json
```

## 9. Optional: Rerun Evaluation Only

Use this only if you want to rescore existing proposal outputs without rerunning the model:

```bash
uv run python src/evaluate.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --a-box-proposals reports/reasoning_floor/$RUN_ID/minimal_case/a_box_proposals.jsonl \
  --t-box-proposals reports/reasoning_floor/$RUN_ID/minimal_case/t_box_proposals.jsonl \
  --track-diagnoses reports/reasoning_floor/$RUN_ID/minimal_case/track_diagnoses.jsonl \
  --run-manifest reports/reasoning_floor/$RUN_ID/run_manifest.jsonl \
  --ablation-bundle minimal_case \
  --out-traces reports/reasoning_floor/$RUN_ID/minimal_case/evaluation_traces_rerun.jsonl \
  --out-summary reports/reasoning_floor/$RUN_ID/minimal_case/evaluation_summary_rerun.json
```

Repeat with `logic_only` and `local_graph` if needed.

## 10. Minimal End-to-End Command List

If you only want the shortest strict sequence, run these in order:

```bash
cd /home/mvazquez/kg-benchmark
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync
mkdir -p data reports logs
ln -sf /absolute/path/to/latest-all.json.gz data/latest-all.json.gz
uv run python src/classifier.py --self-test
uv run python -m unittest discover -s tests
uv run python src/fetcher.py
uv run python src/fetcher.py --validate-only
uv run python src/classifier.py
uv run python src/splitter.py
export OPENAI_API_KEY='YOUR_API_KEY'
export OPENAI_MODEL='YOUR_MODEL_NAME'
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
uv run python src/reasoning_floor.py --classified-benchmark data/04_classified_benchmark.jsonl --world-state data/03_world_state.json --output-dir reports/reasoning_floor/$RUN_ID
```

That sequence builds the benchmark, the splits, and the zero-shot paper outputs from the current codebase.
