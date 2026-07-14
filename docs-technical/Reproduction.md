# Reproduction

All WSL commands use `UV_PROJECT_ENVIRONMENT=.venv-wsl`. Install with `uv sync --extra dev --extra analysis --extra ui`.
The repository exposes only `kg-benchmark`; run `kg-benchmark --help` for the paper workflow.

Before dataset construction, run `kg-benchmark methodology check`. Candidate validity and final freeze readiness are
separate: a valid candidate may still report unresolved model revisions, non-frozen statuses, a dirty worktree, or a
missing lock. `kg-benchmark methodology freeze` writes the deterministic lock only after every other blocker has been
removed. Acquisition, build, and model execution refuse to run without a matching final lock.

A clean clone obtains the published bulk release with `kg-benchmark fetch --manifest-url ... --manifest-sha256 ...`,
validates every byte and record count with
`kg-benchmark verify --dataset-dir dataset`, and uses `kg-benchmark run` and `kg-benchmark score` for execution and
metric replay. Raw generations live under ignored `runs/`; compact aggregate outputs used in the paper live in
`results/`.

Dataset construction uses an empty ignored `work/` directory. Acquisition, build, audit, and selection must complete
before `kg-benchmark promote --source-provenance work/source-provenance.json` atomically creates the immutable
`dataset/`. Promotion requires source provenance and refuses to overwrite an existing dataset.
