# Ollama VM Runbook

This runbook covers the local-Ollama path for running Phase F prompt development and Phase G reasoning-floor experiments
on a Linux VM with shared NVIDIA H100 access.

## Recommended Model

Default recommendation: `gpt-oss:120b`.

Reasoning:

- The Phase F/G tasks require strict instruction following, JSON output, and schema/repair reasoning.
- `gpt-oss:120b` is the highest-capacity Ollama model currently recommended here for a single 80 GB H100-class GPU.
- Use `gpt-oss:20b` for a fast smoke test or if the shared GPU is too busy for the 120B model.

The repository defaults in the VM setup script are conservative:

- `OLLAMA_NUM_PARALLEL=1`
- `REASONING_FLOOR_PARALLEL_WORKERS=1`
- `OLLAMA_CONTEXT_LENGTH=16384`
- `OLLAMA_MAX_OUTPUT_TOKENS=2048`
- `OLLAMA_TEMPERATURE=0`

Increase concurrency only after checking the shared GPU policy and measured latency.

## Copy Files To The VM

Create the remote directory:

```bash
ssh -o KexAlgorithms=curve25519-sha256 \
  -i ~/.ssh/gpu-wu-h100_ed25519 \
  -p 32629 \
  mvazquez@137.208.33.107 \
  'mkdir -p ~/kg-benchmark/data ~/kg-benchmark/reports/benchmark_selection ~/kg-benchmark/reports/prompt_dev'
```

Copy code, docs, tests, and project metadata:

```bash
scp -r -o KexAlgorithms=curve25519-sha256 \
  -i ~/.ssh/gpu-wu-h100_ed25519 \
  -P 32629 \
  pyproject.toml uv.lock requirements.txt .env.example \
  src scripts tests docs-conceptual docs-technical \
  mvazquez@137.208.33.107:~/kg-benchmark/
```

Copy Phase F/G data and manifests:

```bash
scp -o KexAlgorithms=curve25519-sha256 \
  -i ~/.ssh/gpu-wu-h100_ed25519 \
  -P 32629 \
  data/03_world_state.json \
  data/04_classified_benchmark.jsonl \
  mvazquez@137.208.33.107:~/kg-benchmark/data/
```

```bash
scp -o KexAlgorithms=curve25519-sha256 \
  -i ~/.ssh/gpu-wu-h100_ed25519 \
  -P 32629 \
  reports/benchmark_selection/dev_prompt_v1_seed_13.json \
  reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json \
  reports/benchmark_selection/core_v1_seed_13.json \
  mvazquez@137.208.33.107:~/kg-benchmark/reports/benchmark_selection/
```

```bash
scp -o KexAlgorithms=curve25519-sha256 \
  -i ~/.ssh/gpu-wu-h100_ed25519 \
  -P 32629 \
  reports/prompt_dev/prompt_validity_charter.md \
  mvazquez@137.208.33.107:~/kg-benchmark/reports/prompt_dev/
```

Do not copy the local `.env` by default. It may contain Azure or university endpoint secrets that are not needed for the
local Ollama path.

## VM Setup

On the VM:

```bash
cd ~/kg-benchmark
uv sync --extra dev
MODEL=gpt-oss:120b bash scripts/vm_ollama_setup.sh
```

If Ollama is not installed and you want the script to install it:

```bash
ALLOW_OLLAMA_INSTALL=1 MODEL=gpt-oss:120b bash scripts/vm_ollama_setup.sh
```

The setup script starts `ollama serve` if needed, pulls the model, and writes `.env.ollama.vm`.

Smoke test:

```bash
UV_PROJECT_ENVIRONMENT=.venv-vm uv run python scripts/test_llm_endpoint.py ollama \
  --dotenv .env.ollama.vm \
  --timeout 900
```

## Phase F v4 Holdout

Run the spec-only holdout again before Phase G:

```bash
bash scripts/run_phase_f_v4_ollama_holdout.sh
```

Useful overrides:

```bash
MODEL_ENDPOINT=ollama MAX_CASES=24 OUTPUT_DIR=reports/prompt_dev/evaluation_prompt_dev_v4_spec_only_holdout24_ollama_zero_shot \
  bash scripts/run_phase_f_v4_ollama_holdout.sh
```

The script uses `--retry-failures` and resumes existing normalized rows in the output directory.

## Phase G Oracle Dry Run

Run a small oracle-mode dry run first:

```bash
MAX_CASES=16 OUTPUT_DIR=reports/reasoning_floor/ollama_v4_spec_only_oracle_dry_run \
  bash scripts/run_phase_g_ollama_oracle.sh
```

If the dry run is stable, run the full core oracle run by omitting `MAX_CASES`:

```bash
OUTPUT_DIR=reports/reasoning_floor/ollama_v4_spec_only_oracle_core \
  bash scripts/run_phase_g_ollama_oracle.sh
```

Use `PROPOSAL_TRACK_MODE=diagnosis_routed` only as an ablation after oracle mode is stable.

## Resume

Phase F prompt-dev evaluation resumes by output directory. Re-run the same command with the same `OUTPUT_DIR`.

Phase G reasoning-floor runs resume with:

```bash
RESUME_RUN_DIR=reports/reasoning_floor/ollama_v4_spec_only_oracle_core/<RUN_ID> \
  OUTPUT_DIR=reports/reasoning_floor/ollama_v4_spec_only_oracle_core \
  bash scripts/run_phase_g_ollama_oracle.sh
```
