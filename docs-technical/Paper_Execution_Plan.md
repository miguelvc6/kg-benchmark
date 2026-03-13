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
- `reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json`
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

## 5. Build the Deterministic Paper Subset Manifest

Generate the frozen paper subset that keeps all A-box cases and caps T-box cases at `100` per property revision:

```bash
uv run python src/select_benchmark_cases.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --output reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json \
  --tbox-cap-per-update 100 \
  --seed 13
```

This produces:

- `reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json`

## 6. Build Deterministic Splits

```bash
uv run python src/splitter.py
```

This produces:

- `data/05_splits.json`

## 7. Configure the Model Provider

The reasoning-floor runner supports OpenAI and Ollama providers.

```bash
cp .env.example .env
```

Edit `.env` for one of the supported providers.

OpenAI:

```dotenv
MODEL_PROVIDER=openai
OPENAI_API_KEY=YOUR_API_KEY
OPENAI_MODEL=YOUR_MODEL_NAME
```

Use the raw key value only. Do not paste `OPENAI_API_KEY=...` into the value itself, and do not add a `Bearer ` prefix.

Optional for a non-default OpenAI-compatible endpoint:

```dotenv
OPENAI_BASE_URL=https://your-compatible-endpoint/v1
```

Optional for API cost estimation in summaries:

```dotenv
OPENAI_INPUT_COST_PER_1M_TOKENS=2.50
OPENAI_OUTPUT_COST_PER_1M_TOKENS=10.00
```

Ollama:

```dotenv
MODEL_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
```

Optional for a non-default Ollama endpoint:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434/api
```

If you point `OLLAMA_BASE_URL` at `https://ollama.com/api`, also set:

```dotenv
OLLAMA_API_KEY=your_api_key
```

Optional for cost estimation in summaries:

```dotenv
OLLAMA_INPUT_COST_PER_1M_TOKENS=0.00
OLLAMA_OUTPUT_COST_PER_1M_TOKENS=0.00
```

`src/reasoning_floor.py` auto-loads `.env` from the repository root or a parent directory. Shell-exported variables still override `.env` values when both are set.

## 8. Run the Zero-Shot Reasoning Floor

Create a base output directory for the paper run:

```bash
mkdir -p reports/reasoning_floor
```

Run the zero-shot baseline over the deterministic paper subset. The runner will create a subdirectory named `<run_id>_<provider>_<model>` under the base output directory:

```bash
uv run python src/reasoning_floor.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --selection-manifest reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json \
  --output-dir reports/reasoning_floor
```

If you want to override the `.env` model name for a single run:

```bash
uv run python src/reasoning_floor.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --selection-manifest reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json \
  --output-dir reports/reasoning_floor \
  --model llama3.2:latest
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

## 9. Verify the Paper Outputs

Inspect the generated run directories:

```bash
find reports/reasoning_floor -maxdepth 1 -mindepth 1 -type d | sort
```

Set the run directory you want to inspect:

```bash
RUN_DIR=$(find reports/reasoning_floor -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)
find "$RUN_DIR" -maxdepth 2 -type f | sort
```

Inspect the combined paper summary:

```bash
sed -n '1,240p' "$RUN_DIR"/reasoning_floor_summary.json
```

Inspect the per-bundle summaries:

```bash
sed -n '1,240p' "$RUN_DIR"/minimal_case/evaluation_summary.json
sed -n '1,240p' "$RUN_DIR"/logic_only/evaluation_summary.json
sed -n '1,240p' "$RUN_DIR"/local_graph/evaluation_summary.json
```

Inspect the benchmark summary artifacts:

```bash
sed -n '1,240p' reports/classifier_stats.json
sed -n '1,240p' reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json
sed -n '1,240p' data/05_splits.json
```

## 10. Optional: Rerun Evaluation Only

Use this only if you want to rescore existing proposal outputs without rerunning the model:

```bash
uv run python src/evaluate.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --selection-manifest reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json \
  --a-box-proposals "$RUN_DIR"/minimal_case/a_box_proposals.jsonl \
  --t-box-proposals "$RUN_DIR"/minimal_case/t_box_proposals.jsonl \
  --track-diagnoses "$RUN_DIR"/minimal_case/track_diagnoses.jsonl \
  --run-manifest "$RUN_DIR"/run_manifest.jsonl \
  --ablation-bundle minimal_case \
  --out-traces "$RUN_DIR"/minimal_case/evaluation_traces_rerun.jsonl \
  --out-summary "$RUN_DIR"/minimal_case/evaluation_summary_rerun.json
```

Repeat with `logic_only` and `local_graph` if needed.

## 11. Minimal End-to-End Command List

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
uv run python src/select_benchmark_cases.py --classified-benchmark data/04_classified_benchmark.jsonl --output reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json --tbox-cap-per-update 100 --seed 13
uv run python src/splitter.py
cp .env.example .env
uv run python src/reasoning_floor.py --classified-benchmark data/04_classified_benchmark.jsonl --world-state data/03_world_state.json --selection-manifest reports/benchmark_selection/paper_eval_tbox_cap_100_seed_13.json --output-dir reports/reasoning_floor
```

That sequence builds the benchmark, the frozen paper subset, the splits, and the zero-shot paper outputs from the current codebase.
