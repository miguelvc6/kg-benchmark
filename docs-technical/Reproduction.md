# Reproduction

All WSL commands use `UV_PROJECT_ENVIRONMENT=.venv-wsl`. Install with `uv sync --extra dev --extra analysis --extra ui`.
The repository exposes only `kg-benchmark`; run `kg-benchmark --help` for the paper workflow.

Before dataset construction, run `kg-benchmark methodology check`. Candidate validity and final freeze readiness are
separate: a valid candidate may still report unresolved model revisions, non-frozen statuses, a dirty worktree, or a
missing lock. `kg-benchmark methodology freeze` writes the deterministic lock only after every other blocker has been
removed. Acquisition, build, and model execution refuse to run without a matching final lock.

A clean clone obtains the published bulk release with `kg-benchmark fetch --manifest-url ... --manifest-sha256 ...`,
validates every byte and record count with `kg-benchmark verify --dataset-dir dataset`, plans and executes with
`kg-benchmark matrix`, and uses `kg-benchmark score` for metric replay. Raw generations live under ignored `runs/`;
compact aggregate outputs used in the paper live in `results/`.

Dataset construction uses an empty ignored `work/` directory. Acquisition, build, audit, and selection must complete
before the following command atomically creates the immutable `dataset/`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark promote \
  --source-provenance work/source-provenance.json \
  --lineage work/lineage.json
```

`build` creates both required manifests after canonicalizing the acquired Stage 0–3 artifacts and classifying Stage 4.
Promotion requires the canonical paths shown above, verifies their bindings, and refuses to overwrite an existing
dataset.

After `kg-benchmark build`, run the canonical audit as four explicit resumable phases:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit prepare \
  --lineage-manifest work/lineage.json
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit deterministic
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit review
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit finalize
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit status
```

The equivalent one-command form is `kg-benchmark audit run`. It performs Codex calls during the review phase; `status`
never calls a model. The displayed lineage argument is also the default, but spelling it out makes the release binding
visible in reproduction logs. If a bound artifact changes, preserve the failed workflow for diagnosis, remediate the
source or implementation defect, remove the invalid `work/audit/` run, and restart from `audit prepare`. Do not edit a
sample, review, disposition, or summary in place.

After audit finalization, freeze a complete event-group exclusion artifact and run selection:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select reserve \
  --exclusions <frozen-event-group-exclusions.json>
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select review
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select finalize
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select status
```

`select reserve` renders and scans every reserve prompt. `select review` performs the fixed 50-case Codex temporal
review; the other three commands make no model calls. The resumable manifest is `work/selections/selection-workflow.json`.
If any bound input differs, reuse fails closed. Preserve an invalid workflow for diagnosis and restart in a fresh empty
selection directory instead of editing generated artifacts.

The initial main/Azure sizes come from `paper/selection-policy.json`. Additional prompt-clean nested populations are
materialized with `select expand` and explicit per-stratum quotas; this never repeats prompt review or provider calls and
cannot exceed the audited reserve. Provider request deduplication is handled later by generation identity and cache.

After promotion, materialize and inspect the exact execution matrix before any provider call:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark matrix plan
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark matrix dry-run \
  --matrix-dir runs/matrices/<matrix-id>
```

The dry run must report no missing revisions before execution. Run or resume the matrix, then require complete cell
coverage:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark matrix execute \
  --matrix-dir runs/matrices/<matrix-id>
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark matrix status \
  --matrix-dir runs/matrices/<matrix-id>
```

Planning and status are safe to repeat. Execution skips independently verified complete groups and resumes incomplete
groups in place. To add a population later, first create it with `select expand`, then plan a matrix with explicit
`--population` paths. The shared generation cache means parent cases are not submitted again.

If the audit exposes a systemic implementation defect after methodology freeze, the protocol requires invalidating the
freeze and acquired dataset, fixing the implementation, creating a new freeze, and acquiring again. Restarting only the
audit is appropriate for damaged audit outputs or non-systemic case-level dispositions, not for a changed methodology
or construction rule.

Final promotion is deliberately stricter than phase-local status commands. It validates all source, case, disposition,
ordering, replacement, and population records against the schemas copied into the release; replays Stage 2/3/4 lineage
and deterministic selection; checks complete source and cache provenance against the construction inputs; requires zero
unresolved systemic findings; and proves the 1,200/600 population sizes and nesting. It writes into a temporary sibling,
rebuilds `manifest.json` from the copied bytes, requires an exact byte match, and only then performs the atomic rename.
