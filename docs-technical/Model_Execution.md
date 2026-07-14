# Model Execution

The model matrix is configured in `paper/models.json`; populations, tasks, prompt regimes, and contexts are data-driven.
The initial full factorial has repair proposal and track diagnosis under zero/static-few-shot and
`logic_only`/`local_graph`: 9,600 calls per Ollama model and 4,800 Azure batch calls.

Ollama models are `qwen3:30b`, `llama3.3:70b`, and `gpt-oss:120b`. Azure uses `gpt-5.6-sol`, high reasoning effort,
batch execution, and disabled tools. Exact model/deployment revisions must be filled before methodology freeze.

`kg-benchmark run` writes immutable raw generations and run manifests under ignored `runs/`. Generation keys depend on
the rendered request, model revision, and inference settings rather than population membership. A larger nested
population therefore reuses all matching earlier requests and schedules only new cases. Changing few-shot count changes
the rendered prompt and correctly defines new requests.

`kg-benchmark score` writes a new metric-version output without mutating or resubmitting generations. Confirmatory and
extension populations remain separately identified even when they share cached responses.

`kg-benchmark baseline` retains the deterministic non-LLM comparisons. The Reasoning-floor Streamlit application is
launched with `kg-benchmark viewer` and reads the same dataset and run layout.
