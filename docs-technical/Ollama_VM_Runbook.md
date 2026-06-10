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
scp -p -o KexAlgorithms=curve25519-sha256 \
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

The Phase F/G Ollama scripts default to the full `data/03_world_state.json` artifact. On first use the repository builds
a SQLite lookup next to that file at `data/03_world_state.json.sqlite`; this can take a while but only needs to happen
when the source file signature changes.

## VM Setup

On the VM:

```bash
cd ~/kg-benchmark
UV_PROJECT_ENVIRONMENT=.venv-vm uv sync --extra dev
MODEL=gpt-oss:120b bash scripts/vm_ollama_setup.sh
```

If Ollama is not installed and you want the script to install it:

```bash
ALLOW_OLLAMA_INSTALL=1 MODEL=gpt-oss:120b bash scripts/vm_ollama_setup.sh
```

The default install mode is user-local and does not require root. It downloads the official Linux package and extracts it
under `~/.local`, then runs `~/.local/bin/ollama`. Current Ollama Linux packages are usually `.tar.zst`, so the VM needs
`zstd` for this user-local extraction path. The script falls back to older `.tgz` or direct-binary URLs if available.

Use the system installer only if you have root/sudo permissions:

```bash
OLLAMA_INSTALL_MODE=system ALLOW_OLLAMA_INSTALL=1 MODEL=gpt-oss:120b bash scripts/vm_ollama_setup.sh
```

The setup script writes `.env.ollama.vm`, starts `ollama serve` if needed, and pulls the model. If Ollama is not
installed and `ALLOW_OLLAMA_INSTALL=1` is not set, the script exits after writing `.env.ollama.vm`.

If the user-local install fails because the VM lacks `zstd`, blocks downloading, executing user-local binaries, binding
`127.0.0.1:11434`, or accessing the H100 from the process, contact IT. The useful request is:

> Please install or make available Ollama for this VM/user, install `zstd` if user-local extraction is expected, allow
> running `ollama serve` on localhost, and confirm the process has access to the shared NVIDIA H100 through the installed
> NVIDIA driver/container runtime.

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

The core manifest is T-box-first, so small `MAX_CASES` runs without a track filter may not exercise A-box proposal
generation. Run a separate A-box smoke test before treating the oracle path as fully checked:

```bash
MAX_CASES=16 TRACKS=A_BOX OUTPUT_DIR=reports/reasoning_floor/ollama_v4_spec_only_oracle_abox_dry_run \
  bash scripts/run_phase_g_ollama_oracle.sh
```

The wrapper refuses to run the full selected core set unless full-core execution is explicitly approved. If the dry run
is stable and full Phase G oracle scoring has been approved, run:

```bash
ALLOW_FULL_CORE_RUN=1 OUTPUT_DIR=reports/reasoning_floor/ollama_v4_spec_only_oracle_core \
  bash scripts/run_phase_g_ollama_oracle.sh
```

Do not omit `MAX_CASES` for exploratory dry runs. A bare wrapper invocation is intentionally blocked unless
`ALLOW_FULL_CORE_RUN=1` is set.

Use `PROPOSAL_TRACK_MODE=diagnosis_routed` only as an ablation after oracle mode is stable.

## Resume

Phase F prompt-dev evaluation resumes by output directory. Re-run the same command with the same `OUTPUT_DIR`.

Phase G reasoning-floor runs resume with:

```bash
RESUME_RUN_DIR=reports/reasoning_floor/ollama_v4_spec_only_oracle_core/<RUN_ID> \
  OUTPUT_DIR=reports/reasoning_floor/ollama_v4_spec_only_oracle_core \
  bash scripts/run_phase_g_ollama_oracle.sh
```

If you run `src/reasoning_floor.py` directly instead of the wrapper, source `.env.ollama.vm` first so the Ollama timeout,
retry, context, and model settings are exported:

```bash
set -a
source .env.ollama.vm
set +a
```

A timeout message that says `read timeout=120` means the direct command did not pick up `OLLAMA_TIMEOUT_SECONDS=900` from
`.env.ollama.vm`. Prefer the wrapper for VM runs, or export `OLLAMA_TIMEOUT_SECONDS=900` explicitly before direct
commands. Reasoning-floor records exhausted provider failures as `request_error` rows so a single timed-out request does
not abort the whole run.
