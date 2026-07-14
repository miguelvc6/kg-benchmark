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
# The candidate may be inspected now; acquisition requires freeze_ready=true.
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark methodology check

# Run only after the final methodology lock has been committed.
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark acquire --refresh-candidates
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark build
# `build` writes canonical source JSONL, source provenance, and work/lineage.json.
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit prepare
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit deterministic
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit review
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit finalize
: "${EVENT_GROUP_EXCLUSIONS:?Set EVENT_GROUP_EXCLUSIONS to the frozen exclusion manifest}"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select reserve \
  --exclusions "$EVENT_GROUP_EXCLUSIONS"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select review
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select finalize
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select status
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark promote \
  --source-provenance work/source-provenance.json \
  --lineage work/lineage.json

# Clean-clone use after the external release is published.
: "${DATASET_MANIFEST_URL:?Set DATASET_MANIFEST_URL to the published manifest URL}"
: "${DATASET_MANIFEST_SHA256:?Set DATASET_MANIFEST_SHA256 to its SHA-256}"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark fetch \
  --manifest-url "$DATASET_MANIFEST_URL" \
  --manifest-sha256 "$DATASET_MANIFEST_SHA256"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark verify --dataset-dir dataset
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
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark analyze replay \
  --matrix-dir "$MATRIX_DIR" \
  --evaluation-id paper-metrics-v1
RESULT_DIR="$(
  UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark analyze run \
    --matrix-dir "$MATRIX_DIR" \
    --evaluation-id paper-metrics-v1 |
    UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -c \
      'import json,sys; print(json.load(sys.stdin)["result_dir"])'
)"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark analyze status \
  --result-dir "$RESULT_DIR"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark viewer
```

The audit and selection phases are hash-verified and resumable. `kg-benchmark audit run` executes or resumes the audit,
including its protocol-bound Codex review. Selection separately constructs and scans the full reserve, runs the fixed
50-case temporal review, replaces failures, and seals the main and nested Azure populations. Both `audit status` and
`select status` verify stored artifacts and their semantic bindings without calling a model.

Construction is isolated under ignored `work/`. Promotion is atomic and refuses to overwrite the final dataset. Raw
generations and evaluation traces are content-addressed under ignored `runs/`; metric changes and paper analysis do not
call providers again. Compact hash-bound aggregate packages are written under `results/`.

Promotion revalidates every published record and reconstructs lineage, audit coverage, independent eligibility,
support and prior-group exclusions, reserve prompt QA, replacements, and both populations. It requires exactly 1,200
main cases and a nested 600-case Azure population, then reproduces the release manifest byte-for-byte before the
temporary directory can become `dataset/`.

The remaining implementation and operational gates are tracked in [checklist.md](checklist.md). Dataset generation is
not part of repository maintenance and should begin only after the checklist's implementation section is complete.

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
