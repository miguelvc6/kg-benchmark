# Model Execution

The model matrix is configured in `paper/models.json`; populations, tasks, prompt regimes, and contexts are data-driven.
The initial full factorial has repair proposal and track diagnosis under zero/static-few-shot and
`logic_only`/`local_graph`: 9,600 calls per Ollama model and 4,800 sequential Azure calls.

Ollama models are `qwen3:30b`, `llama3.3:70b`, and `gpt-oss:120b`. Qwen thinking is enabled, Llama thinking is
disabled, and GPT-OSS uses high thinking. Ollama uses two exact-request transport retries. Azure uses `gpt-5.6-sol`,
the dated `gpt-5.6-sol-2026-07-09` snapshot, high reasoning effort, sequential synchronous execution, two exact-request
transport retries, and disabled tools.

The Azure deployment name is `gpt-5.6-sol`; the dated `gpt-5.6-sol-2026-07-09` value remains recorded separately as
the observed model revision. Chat Completions sends reasoning effort as the top-level `reasoning_effort` field.
Disabled tools are represented by omitting both `tools` and `tool_choice`, so the model has no tool definitions or
tool-selection capability.

For synchronous OpenAI-compatible calls, the provider encodes the request body once and reuses those bytes for every
transport attempt. Only HTTP 429, 500, 502, 503, and 504 responses plus connection failures and timeouts are retryable;
ordinary 4xx responses, authentication failures, and response-parsing failures are not. Azure uses a 900-second request
timeout by default through `AZURE_OPENAI_TIMEOUT_SECONDS`. Backoff is configured separately through
`AZURE_OPENAI_RETRY_BASE_SECONDS` and `AZURE_OPENAI_RETRY_MAX_SECONDS`. The frozen matrix's explicit retry count wins
over `AZURE_OPENAI_MAX_RETRIES`.

Every installed Ollama artifact is bound by its full local-registry digest. The Azure execution target and cache identity
are bound to the exact dated snapshot returned by the configured Azure data-plane model inventory; mutable model tags
are not accepted as final revision identity.

`kg-benchmark matrix plan` reads the final dataset, model configuration, protocol, and population manifests. It renders
every request and writes a deterministic matrix under `runs/matrices/<matrix-id>/`. A logical cell is one
model × population × task × prompt regime × context bundle. Repair and diagnosis cells with the other dimensions equal
share one physical execution group because the runner produces both task outputs in that pass.

Planning and `matrix dry-run` never construct a provider or make a provider call. The dry run reports logical cells,
physical groups, request memberships, unique requests, cache hits, new requests, and unresolved model revisions. A
request membership records that a population needs a generation; unique counts remove overlap between nested
populations.

```bash
MATRIX_DIR="$(
  UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark matrix plan |
    UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -c \
      'import json,sys; print(json.load(sys.stdin)["matrix_dir"])'
)"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark matrix dry-run \
  --matrix-dir "$MATRIX_DIR"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark matrix execute \
  --matrix-dir "$MATRIX_DIR"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark matrix status \
  --matrix-dir "$MATRIX_DIR"
```

Execution is blocked unless the methodology lock is valid and every selected model has a stable revision. It passes the
frozen Ollama settings exactly. Azure runs sequentially with high reasoning effort, tools disabled, and two exact-request
transport retries. Each physical group resumes in its stable `executions/<group-id>/` directory. Each logical cell gets
a hash-bound manifest under `cells/`; `matrix status` independently recomputes output coverage and generation-cache
coverage rather than trusting a saved success flag.

A manifest row with `parse_status: request_error` is incomplete work, not a cached model answer. Resuming the same
physical group retries that request while retaining the historical error row; completed diagnosis or proposal requests
remain untouched. Successful call records include `transport_attempts` and `transport_retries`, and run summaries total
them for comparison with provider billing. The run configuration records timeout and backoff values under
`transport_settings`; they are operational provenance and do not change the content-addressed generation identity.
There is no paper-execution option to bypass the methodology lock or resume against a different source revision.

Generation keys depend on the rendered request, provider/deployment, model revision, and inference settings rather than
population membership or provider endpoint. A larger nested population therefore reuses all matching earlier requests
and schedules only new keys. To plan extensions, pass one or more explicit population manifests; every selected model
is crossed with each supplied population:

```bash
: "${EXPANDED_POPULATION:?Set EXPANDED_POPULATION to the sealed extension manifest}"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark matrix plan \
  --model-id ollama_qwen3_30b \
  --population dataset/selections/main-1200.json \
  --population "$EXPANDED_POPULATION"
```

The dry run is the proof and preflight: shared parent keys appear as cache hits once the parent has run, while only keys
for newly added cases appear under `unique_new_requests`. Changing a prompt, few-shot count, model revision, or inference
setting correctly defines new requests.

The Azure cost pilot uses a deterministic 48-case population nested inside the frozen 600-case Azure population. Its
eight conditions produce 384 request keys. The pilot and full Azure matrices must use the same dataset, model file,
protocol, and generation-cache SQLite file. Population membership is not part of cache identity, so every successful
pilot response is retained and becomes a cache hit in the later 600-case run; only the remaining 4,416 Azure requests
need generation. A Foundry cost delta for the pilot can therefore be extrapolated by approximately 12.5, with observed
transport retries reported separately.

`kg-benchmark score` writes a new metric-version output for one physical run without mutating or resubmitting
generations. The paper-facing `kg-benchmark analyze replay` applies that replay across a complete matrix and additionally
scores diagnosis for every selected A- and T-box case. Confirmatory and extension populations remain separately
identified even when they share cached responses. See [paper analysis](./Paper_Analysis.md).

`kg-benchmark baseline` retains the deterministic non-LLM comparisons. The Reasoning-floor Streamlit application is
launched with `kg-benchmark viewer` and reads the same dataset and run layout.
