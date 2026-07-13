# Model Execution Matrix

The paper model and population plan is tracked in
[`experiments/paper_execution_models_v1.json`](../experiments/paper_execution_models_v1.json). It is intentionally
`draft` until the prompt configuration, final selection hashes, and all immutable model revisions are available.
`kg-experiment-plan` validates the matrix and derives workloads and runner arguments; adding a model or population is a
data-only change when it uses an existing provider contract.

## Registered draft conditions

| Provider | Model/deployment | Population | Execution | Requests |
|---|---|---:|---|---:|
| Ollama | `qwen3:30b` | full 1,200 | parallel, one worker | 2,400 |
| Ollama | `llama3.3:70b` | full 1,200 | parallel, one worker | 2,400 |
| Ollama | `gpt-oss:120b` | full 1,200 | parallel, one worker | 2,400 |
| Azure | `gpt-5.6-sol` | nested 600 | batch, high reasoning effort, tools disabled | 1,200 |

The counts assume oracle routing with diagnosis skipped and the `logic_only` and `local_graph` bundles. The Azure
condition disables synchronous fallback: a failed batch item remains a recorded request error and is not silently
reissued through a different execution mode. The OpenAI-compatible request adapter sends `tool_choice: "none"` for both
synchronous and batch payloads.

All four conditions pin `tbox_task_version=tbox_taxonomy_patch_v1`. T-box scoring requires complete mechanically
supported gold and excludes cases requiring unmined class-hierarchy or exception operations. A-box and T-box use separate
metric families and are never collapsed into one repair-success score. Strict-signature reconstruction is not queried by
the confirmatory matrix.

Validate or render the plan with:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-experiment-plan validate
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-experiment-plan plan --output reports/execution_plan.json
```

`--require-frozen` deliberately fails the current draft. Before changing the matrix to `frozen`, resolve the two missing
Ollama manifest digests, bind an immutable Azure deployment revision, replace the snapshot selection placeholders and
their SHA-256 hashes, and set `prompt_configuration` to the prompt freeze chosen in the next methodology step. The
installed `gpt-oss:120b` manifest digest is already recorded. Large missing Ollama models are not pulled as part of
validation.

## Cross-run generation reuse

Every planned model has a dedicated append-only SQLite generation cache under `.cache/model_generations/`. The cache key
binds provider, model, immutable model digest, resolved inference settings, system prompt, user prompt, and response
format. A cache is therefore reusable across retries, population extensions, and selection reorganizations without
allowing responses to cross a changed model revision or changed prompt.

Prompt-visible case IDs are stable hashes of raw case IDs. They no longer depend on selection order, so adding cases to
a population does not alter existing prompts. Cache hits report zero new tokens and cost while preserving the source
generation usage in provenance. A model digest is mandatory whenever caching is enabled.

The runner still supports `--resume-run-dir` for an interrupted identical population. The generation cache is the broader
mechanism for reuse across different run directories and expanded populations.

## Versioned metric replay

Generation and scoring are separable. Apply a revised evaluator to stored normalized proposals without model calls:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-rescore-run \
  --run-dir reports/reasoning_floor/RUN_ID \
  --evaluation-id metrics_v2
```

Outputs are written under `RUN_ID/evaluations/metrics_v2/`. The command refuses to overwrite an existing evaluation ID
and writes a manifest containing source-output hashes, evaluator code hashes, Git state, output hashes, and
`provider_calls: 0`. If data artifacts have moved, pass explicit benchmark, world-state, and selection-manifest paths;
their content is fingerprinted in the replay manifest.

For taxonomy-patch runs, replay writes separate A-box and T-box taxonomy summaries using a newly derived, content-bound
gold version. It refuses incomplete mechanically supported T-box gold and never routes missing strict-signature proposals
through the legacy evaluator.
