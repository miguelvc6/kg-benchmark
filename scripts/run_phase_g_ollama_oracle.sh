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
PROPOSAL_TRACK_MODE="${PROPOSAL_TRACK_MODE:-oracle}"

if [[ -z "${MAX_CASES:-}" && "${ALLOW_FULL_CORE_RUN:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
Refusing to run the full selected core set without explicit confirmation.

For a dry run, set MAX_CASES, for example:
  MAX_CASES=16 OUTPUT_DIR=reports/reasoning_floor/ollama_v4_spec_only_oracle_dry_run_16 bash scripts/run_phase_g_ollama_oracle.sh

For an approved full core run, set:
  ALLOW_FULL_CORE_RUN=1
EOF
  exit 2
fi

if [[ ! -f "${WORLD_STATE_PATH}" ]]; then
  echo "World-state file not found: ${WORLD_STATE_PATH}" >&2
  echo "Copy the full data/03_world_state.json dataset to the VM, or override WORLD_STATE=..." >&2
  exit 1
fi

OLLAMA_API_BASE="${OLLAMA_BASE_URL:-http://localhost:11434/api}"
OLLAMA_API_BASE="${OLLAMA_API_BASE%/}"
OLLAMA_SERVER_BASE="${OLLAMA_API_BASE%/api}"
OLLAMA_HEALTH_URL="${OLLAMA_SERVER_BASE}/api/tags"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for the Ollama preflight check." >&2
  exit 3
fi

if ! curl -fsS --max-time 5 "${OLLAMA_HEALTH_URL}" >/dev/null 2>&1; then
  cat >&2 <<EOF
Ollama is not reachable at ${OLLAMA_HEALTH_URL}.

Start or repair the local Ollama server before running Phase G:
  bash scripts/vm_ollama_setup.sh

Then verify:
  UV_PROJECT_ENVIRONMENT=${UV_ENV} uv run python scripts/test_llm_endpoint.py ollama --dotenv .env.ollama.vm --timeout 900
EOF
  exit 3
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
  --proposal-track-mode "${PROPOSAL_TRACK_MODE}"
  --ablation-bundles "${ABLATION_BUNDLES:-logic_only,local_graph}"
)

if [[ "${PROPOSAL_TRACK_MODE}" == "oracle" ]]; then
  args+=(--oracle-diagnosis-mode skip)
fi

if [[ -n "${MAX_CASES:-}" ]]; then
  args+=(--max-cases "${MAX_CASES}")
fi

if [[ -n "${TRACKS:-}" ]]; then
  args+=(--tracks "${TRACKS}")
fi

if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
  args+=(--resume-run-dir "${RESUME_RUN_DIR}")
fi

UV_PROJECT_ENVIRONMENT="${UV_ENV}" uv run python "${args[@]}"
