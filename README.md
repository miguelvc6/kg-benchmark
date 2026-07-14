# WikidataRepairEval

WikidataRepairEval is the reproducible implementation for a paper on language-model repair of Wikidata A-box claims
and T-box constraints under controlled information conditions.

The repository has one active protocol under `paper/`, one public command (`kg-benchmark`), one externally distributed
final dataset, and compact paper results. Generated data, caches, logs, prompts, and model responses are not tracked.

## Research design

The benchmark distinguishes A-box claim repair from T-box taxonomy-patch repair. A-box cases are stratified as logical
IC-L, local-graph IC-G, or IC-E-elim, where supported logical/local extractors did not identify the historical target.
IC-E-elim receives no retrieval and is not described as confirmed external-evidence necessity.

The headline matrix evaluates repair proposal and track diagnosis under zero/static-few-shot and
`logic_only`/`local_graph`. Repair uses oracle routing; diagnosis is scored independently. The main population contains
1,200 independent cases and Azure receives a nested 600-case view. Stable eligibility ordering supports larger nested
populations without repeating existing requests.

## Install

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv sync --extra dev --extra analysis --extra ui
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark --help
```

## Workflow

```bash
# Build only after the paper protocol is frozen.
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark acquire --refresh-candidates
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark build
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit ...
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select build \
  --cases work/cases.jsonl \
  --dispositions work/audit/dispositions.jsonl
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark promote \
  --source-provenance work/source-provenance.json

# Clean-clone use after the external release is published.
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark fetch \
  --manifest-url <published-manifest-url> \
  --manifest-sha256 <published-manifest-sha256>
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark verify --dataset-dir dataset
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark run ...
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark score ...
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark baseline ...
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark viewer ...
```

Construction is isolated under ignored `work/`. Promotion is atomic and refuses to overwrite the final dataset. Raw
generations are content-addressed under ignored `runs/`; metric changes do not call providers again.

## Repository map

- `paper/`: current protocol, analysis plan, models, selection policy, and prompt sources.
- `schemas/`: current published artifact and model-response contracts.
- `src/kg_benchmark/`: paper-facing package and CLI.
- `dataset/`: manifest and externally retrieved final dataset.
- `results/`: compact aggregate outputs used in the paper.
- `docs-conceptual/`: research objectives, narrative, taxonomy, design, and limitations.
- `docs-technical/`: reproduction, implementation, artifact reference, and development history.
- `examples/`: tiny synthetic data for inspection and end-to-end tests.

See [conceptual documentation](docs-conceptual/README.md), [technical documentation](docs-technical/README.md), and the
[development history](docs-technical/Development_History.md).
