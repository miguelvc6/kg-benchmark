# Paper Analysis and Result Packaging

Paper analysis is a provider-free two-stage workflow. `analyze replay` applies one evaluator version to every complete
physical matrix group. `analyze run` consumes those immutable evaluation manifests and writes one compact,
content-addressed result package under `results/`. Neither command constructs a model provider or submits a request.

```bash
: "${MATRIX_DIR:?Set MATRIX_DIR to the completed matrix directory}"
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
```

Replay is resumable at the physical-group boundary. An absent evaluation is created from immutable generations; an
existing evaluation is reused only after its run configuration, run manifest, output hashes, population size, context,
and zero-provider-call declaration verify. Metric revisions use a new `--evaluation-id`; old evaluations are never
overwritten. Replay emits separate A-box repair, T-box taxonomy-patch, and all-cases track-diagnosis traces. Parse or
schema failures remain completed observations and score incorrect. A transport failure prevents matrix completeness and
therefore blocks replay and analysis.

The analysis implementation reads `paper/analysis.json`. For every applicable endpoint it reports the case-micro mean
and the mean of independent event-cluster means. A-box clusters are QID/property groups; T-box clusters are
property/revision groups. The mixed-locus diagnosis estimate uses the already independent event-group key. Intervals are
the predeclared 95% percentile intervals from 5,000 seed-13 cluster-bootstrap samples.

Every primary comparison is paired by exact case ID and event group. Effects are reported as right minus left. Binary
outcomes use the two-sided exact conditional McNemar test. Holm adjustment is applied to exactly the four predeclared
contrasts within each model × task × stratum × endpoint family. T-box’s two primary endpoints form distinct four-test
families. Models are never pooled, and A-box and T-box endpoints are never collapsed into one repair score.

Base packages require every configured model on its configured population. Ollama models form the confirmatory role on
`main-1200`; Azure forms the paired calibration role on `azure-600`, without cross-model significance tests. A matrix
containing a larger population produces an extension package instead: the parent must be present, hash-bound, and an
exact per-stratum prefix. Parent rows are used only to prove nesting and are not relabeled as new confirmatory results.

Each result directory contains `summary.json`, `estimates.jsonl`, `contrasts.jsonl`, `diagnosis.jsonl`, and compact
Markdown tables for confirmatory, Azure-calibration, extension, and diagnosis reporting. `manifest.json` binds those
outputs to the matrix, all evaluation manifests, frozen analysis configuration, analysis code, and `uv.lock`. The
analysis ID is content-addressed from the same inputs. `analyze status` validates every schema and hash, requires four
rows in every Holm family, and reproduces the manifest bytes.
