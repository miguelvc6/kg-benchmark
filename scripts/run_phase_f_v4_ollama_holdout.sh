#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env.ollama.vm ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.ollama.vm
  set +a
fi

export PROMPT_DEV_VERSION="${PROMPT_DEV_VERSION:-prompt_dev_v4_spec_only}"
MAX_CASES="${MAX_CASES:-96}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/prompt_dev/evaluation_prompt_dev_v4_spec_only_holdout96_ollama_zero_shot}"
UV_ENV="${UV_PROJECT_ENVIRONMENT:-.venv-vm}"

UV_PROJECT_ENVIRONMENT="${UV_ENV}" uv run python scripts/test_llm_endpoint.py ollama \
  --dotenv "${DOTENV_PATH:-.env.ollama.vm}" \
  --timeout "${OLLAMA_TIMEOUT_SECONDS:-900}"

UV_PROJECT_ENVIRONMENT="${UV_ENV}" uv run python src/prompt_dev.py evaluate \
  --classified-benchmark "${CLASSIFIED_BENCHMARK:-data/04_classified_benchmark.jsonl}" \
  --world-state "${WORLD_STATE:-data/03_world_state.json}" \
  --dev-manifest "${DEV_MANIFEST:-reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json}" \
  --core-manifest "${CORE_MANIFEST:-reports/benchmark_selection/core_v1_seed_13.json}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-endpoint ollama \
  --max-cases "${MAX_CASES}" \
  --sample-strategy "${SAMPLE_STRATEGY:-manifest_order}" \
  --representations "${REPRESENTATIONS:-hybrid_json_nl}" \
  --example-policies "${EXAMPLE_POLICIES:-zero_shot}" \
  --context-bundles "${CONTEXT_BUNDLES:-logic_only,local_graph}" \
  --tasks "${TASKS:-track_diagnosis,repair_proposal}" \
  --repair-track-modes "${REPAIR_TRACK_MODES:-oracle}" \
  --retry-failures \
  --no-progress
