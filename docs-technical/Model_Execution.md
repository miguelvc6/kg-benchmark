# Model Execution

The model matrix is configured in `paper/models.json`; populations, tasks, prompt regimes, and contexts are data-driven.
The initial full factorial has repair proposal and track diagnosis under zero/static-few-shot and
`logic_only`/`local_graph`: 9,600 calls per Ollama model and 4,800 Azure batch calls.

Ollama models are `qwen3:30b`, `llama3.3:70b`, and `gpt-oss:120b`. Qwen thinking is enabled, Llama thinking is
disabled, and GPT-OSS uses high thinking. Ollama uses two exact-request transport retries. Azure uses `gpt-5.6-sol`,
high reasoning effort, batch-only execution, no synchronous fallback, zero transport retries, and disabled tools.

The installed GPT-OSS artifact is bound by its full Ollama digest. The Qwen, Llama, and immutable Azure deployment or
snapshot revisions remain explicit freeze blockers until provisioned and resolved; mutable model tags are not accepted
as final revision identity.

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
frozen Ollama settings exactly. Azure is forced through batch mode with high reasoning effort, tools disabled, zero
transport retries, and no synchronous fallback. Each physical group resumes in its stable `executions/<group-id>/`
directory. Each logical cell gets a hash-bound manifest under `cells/`; `matrix status` independently recomputes output
coverage and generation-cache coverage rather than trusting a saved success flag.

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

`kg-benchmark score` writes a new metric-version output for one physical run without mutating or resubmitting
generations. The paper-facing `kg-benchmark analyze replay` applies that replay across a complete matrix and additionally
scores diagnosis for every selected A- and T-box case. Confirmatory and extension populations remain separately
identified even when they share cached responses. See [paper analysis](./Paper_Analysis.md).

`kg-benchmark baseline` retains the deterministic non-LLM comparisons. The Reasoning-floor Streamlit application is
launched with `kg-benchmark viewer` and reads the same dataset and run layout.
