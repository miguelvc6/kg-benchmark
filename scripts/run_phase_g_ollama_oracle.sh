#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env.ollama.vm ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.ollama.vm
  set +a
fi

UV_ENV="${UV_PROJECT_ENVIRONMENT:-.venv-vm}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/reasoning_floor/ollama_v4_spec_only_oracle}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-${REASONING_FLOOR_PARALLEL_WORKERS:-1}}"
WORLD_STATE_PATH="${WORLD_STATE:-data/03_world_state.json}"

if [[ ! -f "${WORLD_STATE_PATH}" ]]; then
  echo "World-state file not found: ${WORLD_STATE_PATH}" >&2
  echo "Copy the full data/03_world_state.json dataset to the VM, or override WORLD_STATE=..." >&2
  exit 1
fi

args=(
  src/reasoning_floor.py
  --classified-benchmark "${CLASSIFIED_BENCHMARK:-data/04_classified_benchmark.jsonl}"
  --world-state "${WORLD_STATE_PATH}"
  --selection-manifest "${SELECTION_MANIFEST:-reports/benchmark_selection/core_v1_seed_13.json}"
  --output-dir "${OUTPUT_DIR}"
  --model-endpoint ollama
  --execution-mode parallel
  --parallel-workers "${PARALLEL_WORKERS}"
  --proposal-track-mode "${PROPOSAL_TRACK_MODE:-oracle}"
  --ablation-bundles "${ABLATION_BUNDLES:-logic_only,local_graph}"
)

if [[ -n "${MAX_CASES:-}" ]]; then
  args+=(--max-cases "${MAX_CASES}")
fi

if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
  args+=(--resume-run-dir "${RESUME_RUN_DIR}")
fi

UV_PROJECT_ENVIRONMENT="${UV_ENV}" uv run python "${args[@]}"
