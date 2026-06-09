# LLM Endpoint Smoke Test

This repository includes a small endpoint smoke-test command for checking the three inference approaches used during prompt development and reasoning-floor runs.

## Environment Variables

The script reads `.env` through the repository's existing environment loader.

Local Ollama:

```dotenv
OLLAMA_MODEL=gpt-oss:120b
OLLAMA_BASE_URL=http://localhost:11434/api
OLLAMA_API_KEY=
OLLAMA_KEEP_ALIVE=60m
OLLAMA_CONTEXT_LENGTH=16384
OLLAMA_MAX_OUTPUT_TOKENS=2048
OLLAMA_TIMEOUT_SECONDS=900
OLLAMA_MAX_RETRIES=2
OLLAMA_RETRY_BASE_SECONDS=5
OLLAMA_RETRY_MAX_SECONDS=60
OLLAMA_TEMPERATURE=0
OLLAMA_TOP_P=1
OLLAMA_SEED=13
```

Azure AI endpoint:

```dotenv
AZURE_OPENAI_ENDPOINT=https://mvazquez-it-184686-ki.openai.azure.com/openai/v1
AZURE_OPENAI_DEPLOYMENT=gpt-5.4-nano
AZURE_OPENAI_API_KEY=your-api-key
```

University OpenAI-compatible endpoint:

```dotenv
UNIVERSITY_OPENAI_BASE_URL=https://demosite.ml.jku.at/v1
UNIVERSITY_OPENAI_MODEL=Qwen/Qwen3-4B
UNIVERSITY_OPENAI_API_KEY=add-api-key
```

The generic `OPENAI_*` and `MODEL_PROVIDER` variables are still used by `src/reasoning_floor.py`. The smoke-test variables are provider-specific so a quick endpoint check does not accidentally change benchmark-run configuration.

## Commands

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python scripts/test_llm_endpoint.py ollama
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python scripts/test_llm_endpoint.py azure
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python scripts/test_llm_endpoint.py university
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python scripts/test_llm_endpoint.py all
```

The same selector is available as `--model-endpoint`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-test-llm-endpoint --model-endpoint azure
```

After `uv sync`, the console entry point is also available:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-test-llm-endpoint azure
```

Use `--prompt` to override the default test prompt:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-test-llm-endpoint university --prompt "Hello, who are you?"
```

For a Linux VM using the local Ollama configuration written by `scripts/vm_ollama_setup.sh`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-vm uv run python scripts/test_llm_endpoint.py ollama \
  --dotenv .env.ollama.vm \
  --timeout 900
```

The command prints the model answer on success and a concise error on failure. It does not run benchmark evaluation or write reasoning-floor artifacts.
