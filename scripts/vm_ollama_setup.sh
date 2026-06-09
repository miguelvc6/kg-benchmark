#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-gpt-oss:120b}"
OLLAMA_HOST_BIND="${OLLAMA_HOST_BIND:-127.0.0.1:11434}"
OLLAMA_API_BASE="${OLLAMA_API_BASE:-http://localhost:11434/api}"
OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"
OLLAMA_MAX_OUTPUT_TOKENS="${OLLAMA_MAX_OUTPUT_TOKENS:-2048}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-60m}"
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
REASONING_FLOOR_PARALLEL_WORKERS="${REASONING_FLOOR_PARALLEL_WORKERS:-1}"

if ! command -v ollama >/dev/null 2>&1; then
  if [[ "${ALLOW_OLLAMA_INSTALL:-0}" == "1" ]]; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    cat >&2 <<'EOF'
ollama is not installed.

Install it yourself, or rerun with:
  ALLOW_OLLAMA_INSTALL=1 bash scripts/vm_ollama_setup.sh
EOF
    exit 1
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "warning: nvidia-smi not found; verify GPU access before running Phase F/G." >&2
fi

if ! curl -fsS "${OLLAMA_API_BASE%/api}/api/tags" >/dev/null 2>&1; then
  echo "Starting ollama serve on ${OLLAMA_HOST_BIND} ..."
  OLLAMA_HOST="${OLLAMA_HOST_BIND}" OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL}" nohup ollama serve > ollama-serve.log 2>&1 &
  for _ in $(seq 1 30); do
    if curl -fsS "${OLLAMA_API_BASE%/api}/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

echo "Pulling ${MODEL} ..."
ollama pull "${MODEL}"

cat > .env.ollama.vm <<EOF
MODEL_ENDPOINT=ollama
MODEL_PROVIDER=ollama
OLLAMA_MODEL=${MODEL}
OLLAMA_BASE_URL=${OLLAMA_API_BASE}
OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}
OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH}
OLLAMA_MAX_OUTPUT_TOKENS=${OLLAMA_MAX_OUTPUT_TOKENS}
OLLAMA_TIMEOUT_SECONDS=900
OLLAMA_MAX_RETRIES=2
OLLAMA_RETRY_BASE_SECONDS=5
OLLAMA_RETRY_MAX_SECONDS=60
OLLAMA_TEMPERATURE=0
OLLAMA_TOP_P=1
OLLAMA_SEED=13
OLLAMA_INPUT_COST_PER_1M_TOKENS=0.00
OLLAMA_OUTPUT_COST_PER_1M_TOKENS=0.00
REASONING_FLOOR_PARALLEL_WORKERS=${REASONING_FLOOR_PARALLEL_WORKERS}
EOF

echo "Wrote .env.ollama.vm"
echo "Smoke test:"
echo "  UV_PROJECT_ENVIRONMENT=.venv-vm uv run python scripts/test_llm_endpoint.py ollama --dotenv .env.ollama.vm --timeout 900"
