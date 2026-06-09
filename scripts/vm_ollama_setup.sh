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
OLLAMA_INSTALL_MODE="${OLLAMA_INSTALL_MODE:-user}"

install_ollama_user_local() {
  local machine arch tmp_dir archive bin_path
  machine="$(uname -m)"
  case "${machine}" in
    x86_64|amd64) arch="amd64" ;;
    aarch64|arm64) arch="arm64" ;;
    *)
      echo "Unsupported architecture for user-local Ollama install: ${machine}" >&2
      return 1
      ;;
  esac

  tmp_dir="$(mktemp -d)"
  mkdir -p "${HOME}/.local"
  echo "Installing Ollama under ${HOME}/.local from the official Linux ${arch} package ..."
  if curl -fsIL "https://ollama.com/download/ollama-linux-${arch}.tar.zst" >/dev/null 2>&1; then
    if ! command -v zstd >/dev/null 2>&1; then
      echo "Ollama's current Linux package requires zstd for user-local extraction." >&2
      echo "Ask IT to install zstd, install Ollama system-wide, or provide a preinstalled ollama binary." >&2
      return 1
    fi
    archive="${tmp_dir}/ollama-linux-${arch}.tar.zst"
    curl -fL "https://ollama.com/download/ollama-linux-${arch}.tar.zst" -o "${archive}"
    zstd -dc "${archive}" | tar -C "${HOME}/.local" -xf -
  elif curl -fsIL "https://ollama.com/download/ollama-linux-${arch}.tgz" >/dev/null 2>&1; then
    archive="${tmp_dir}/ollama-linux-${arch}.tgz"
    curl -fL "https://ollama.com/download/ollama-linux-${arch}.tgz" -o "${archive}"
    tar -C "${HOME}/.local" -xzf "${archive}"
  else
    bin_path="${HOME}/.local/bin/ollama"
    mkdir -p "${HOME}/.local/bin"
    curl -fL "https://ollama.com/download/ollama-linux-${arch}" -o "${bin_path}"
    chmod +x "${bin_path}"
  fi
  rm -rf "${tmp_dir}"
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! command -v ollama >/dev/null 2>&1; then
    echo "User-local Ollama install finished, but ollama is still not on PATH." >&2
    echo "Try: export PATH=\"${HOME}/.local/bin:\$PATH\"" >&2
    return 1
  fi
}

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

if ! command -v ollama >/dev/null 2>&1; then
  if [[ "${ALLOW_OLLAMA_INSTALL:-0}" == "1" ]]; then
    if [[ "${OLLAMA_INSTALL_MODE}" == "system" ]]; then
      curl -fsSL https://ollama.com/install.sh | sh
    else
      install_ollama_user_local
    fi
  else
    cat >&2 <<'EOF'
ollama is not installed.

Install it yourself, ask IT to install it, or try a user-local install with:
  ALLOW_OLLAMA_INSTALL=1 bash scripts/vm_ollama_setup.sh

Use OLLAMA_INSTALL_MODE=system only if you have sudo/root permissions:
  OLLAMA_INSTALL_MODE=system ALLOW_OLLAMA_INSTALL=1 bash scripts/vm_ollama_setup.sh
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

echo "Smoke test:"
echo "  UV_PROJECT_ENVIRONMENT=.venv-vm uv run python scripts/test_llm_endpoint.py ollama --dotenv .env.ollama.vm --timeout 900"
