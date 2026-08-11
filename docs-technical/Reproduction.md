# Reproduction

All WSL commands use `UV_PROJECT_ENVIRONMENT=.venv-wsl`. Install with `uv sync --extra dev --extra analysis --extra ui`.
The repository exposes only `kg-benchmark`; run `kg-benchmark --help` for the paper workflow.

## Implementation acceptance

Before the final methodology freeze, run the complete offline implementation gate from a clean worktree:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run ruff check .
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run pytest -q
git diff --check
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark methodology check
```

The test suite includes schema and documentation checks, a tracked-files-only clean-clone wheel build/install smoke test,
and the full synthetic lifecycle from build through paper analysis. The lifecycle uses deterministic fake Codex and
model executors: it performs no network access or provider calls, but still proves rejection/replacement, generation
cache reuse, provider-free metric replay, final dataset manifest reproduction, and byte-identical result packages.
The clean-clone smoke test always preserves repository history. When the checkout has tracked modifications, it applies
the binary Git diff over a local clone; untracked artifacts are excluded. This keeps frozen lock revisions available
while allowing an untracked `results/analysis_<hash>/` package during release staging.

Before dataset construction, run `kg-benchmark methodology check`. Candidate validity and final freeze readiness are
separate: a valid candidate may still report unresolved model revisions, non-frozen statuses, a dirty worktree, or a
missing lock. `kg-benchmark methodology freeze` writes the deterministic lock only after every other blocker has been
removed. Acquisition, build, and model execution refuse to run without a matching final lock.

A clean clone obtains the published bulk release with `kg-benchmark fetch`, using the published manifest URL and hash,
validates every byte and record count with `kg-benchmark verify --dataset-dir dataset`, plans and executes with
`kg-benchmark matrix`, and uses `kg-benchmark score` for metric replay. Raw generations live under ignored `runs/`;
compact aggregate outputs used in the paper live in `results/`.

After publication, a clean clone retrieves and verifies the immutable dataset with explicit release metadata:

```bash
: "${DATASET_MANIFEST_URL:?Set DATASET_MANIFEST_URL to the published manifest URL}"
: "${DATASET_MANIFEST_SHA256:?Set DATASET_MANIFEST_SHA256 to its SHA-256}"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark fetch \
  --manifest-url "$DATASET_MANIFEST_URL" \
  --manifest-sha256 "$DATASET_MANIFEST_SHA256"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark verify --dataset-dir dataset
```

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
: "${EVENT_GROUP_EXCLUSIONS:?Set EVENT_GROUP_EXCLUSIONS to the frozen exclusion manifest}"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select reserve \
  --exclusions "$EVENT_GROUP_EXCLUSIONS"
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
MATRIX_DIR="$(
  UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark matrix plan |
    UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -c \
      'import json,sys; print(json.load(sys.stdin)["matrix_dir"])'
)"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark matrix dry-run \
  --matrix-dir "$MATRIX_DIR"
```

The dry run must report no missing revisions before execution. Run or resume the matrix, then require complete cell
coverage:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark matrix execute \
  --matrix-dir "$MATRIX_DIR"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark matrix status \
  --matrix-dir "$MATRIX_DIR"
```

Planning and status are safe to repeat. Execution skips independently verified complete groups and resumes incomplete
groups in place. To add a population later, first create it with `select expand`, then plan a matrix with explicit
`--population` paths. The shared generation cache means parent cases are not submitted again.

After `matrix status` reports complete, replay the frozen evaluator across every physical group and build the paper
package:

```bash
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

Use a new evaluation ID when evaluator metrics change. Replay and analysis make zero provider calls. The result package
binds all evaluation manifests and analysis code, so rerunning the same inputs reproduces the same analysis ID and
manifest bytes. Extension analysis requires the parent and expanded population manifests in the same matrix plan; its
outputs remain separate from the base confirmatory/calibration package.

If the audit exposes a systemic implementation defect after methodology freeze, the protocol requires invalidating the
freeze and acquired dataset, fixing the implementation, creating a new freeze, and acquiring again. Restarting only the
audit is appropriate for damaged audit outputs or non-systemic case-level dispositions, not for a changed methodology
or construction rule.

An execution-transport correction is non-systemic with respect to the acquired dataset only when it changes no dataset,
selection, prompt, model revision, inference setting, generation identity, parser, evaluator, or analysis input. That
classification requires a provider-free re-render proving the same matrix ID and byte-identical request plan, plus a
new clean source lock. The immutable dataset remains bound to its acquisition lock; the replacement source lock binds
the corrected executor. A correction that fails any of these invariants follows the systemic-defect rule above.

## Methodology v7 restart from the completed Stage 2 checkpoint

The v5 temporal audit exposed a systemic scanner defect after Stage 2 had completed. A v6 dry-run then separated most
token coincidences from true leakage but found a small number of genuine future-value and hidden-author occurrences in
otherwise valid cases. Methodology v7 keeps those cases in Stage 0–4 for provenance, assigns them deterministic final
disposition `exclude`, and prevents them from entering any selection population. Preserve the failed audit as historical
evidence, then resume acquisition from the completed checkpoint under the v7 lock. This replay must not use
`--refresh-candidates`: the checkpoint reuses the completed 267,401-candidate Stage 2 scan and recomputes downstream
Stage 3 plus the acquisition configuration binding.

```bash
: "${SOURCE_REPO:?Set SOURCE_REPO to the v7 checkout}"
: "${RUN_ROOT:?Set RUN_ROOT to the retained construction run}"
: "${DUMP_PATH:?Set DUMP_PATH to the 2026 dump}"

cd "$SOURCE_REPO"
if test -d "$RUN_ROOT/work/audit" && test ! -e "$RUN_ROOT/audit-v5-failed"; then
  mv "$RUN_ROOT/work/audit" "$RUN_ROOT/audit-v5-failed"
fi

UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark acquire \
  --dump-path "$DUMP_PATH" \
  --resume-checkpoint "$RUN_ROOT/work/acquisition/logs/resume_checkpoint_20260716T172918.json"

UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark build --dump-path "$DUMP_PATH"

UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit run \
  --lineage-manifest work/lineage.json \
  --report work/audit/audit.md

UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit status | tee "$RUN_ROOT/audit-status.json"
```

`audit run` includes the external Codex review phase. Run the corrected version-4 temporal scanner against the retained
138,312 v5 prompt render in a temporary output directory before the replay if an isolated deterministic comparison is
needed. The reference dry-run identifies 75 high-hit cases, leaving 34,503 cases free of deterministic temporal
exclusions before other audit dispositions. Acceptance requires 138,312 prompts, 34,578 matched cases, every
mutation-sensitivity check true, a passing case-exclusion gate, exact `exclude` dispositions for every high-hit case,
and unchanged lineage, render-coverage, and integrity outcomes. Any newly unexplained high source must be inspected;
do not add a whitelist.

## Methodology v8 restart after the support-render guard defect

The v7 selection reserve exposed a structural-check bug in the few-shot support renderer. One legitimate T-box support
entity had the visible English label `classification`. The renderer searched serialized JSON for the token
`"classification"` and therefore mistook that scalar value for a hidden metadata key, rejecting the static-few-shot
cell for every T-box reserve case. Methodology v8 checks object keys recursively instead: actual `classification` and
`repair_target` keys remain forbidden, while identical visible scalar vocabulary is allowed.

The defect does not change Stage 2 outcomes, but the frozen protocol requires downstream execution to be rebound to the
new lock after a systemic implementation correction. Preserve the completed v7 audit and failed v7 selection, resume
the same completed Stage 2 checkpoint without `--refresh-candidates`, then rebuild and audit before restarting
selection:

```bash
: "${SOURCE_REPO:?Set SOURCE_REPO to the v8 checkout}"
: "${RUN_ROOT:?Set RUN_ROOT to the retained construction run}"
: "${DUMP_PATH:?Set DUMP_PATH to the 2026 dump}"

cd "$SOURCE_REPO"
cp "$RUN_ROOT/work/selections/group-exclusions.json" "$RUN_ROOT/event-group-exclusions.json"
mv "$RUN_ROOT/work/selections" "$RUN_ROOT/selections-v7-failed"
mv "$RUN_ROOT/work/audit" "$RUN_ROOT/audit-v7-complete"

UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark acquire \
  --dump-path "$DUMP_PATH" \
  --resume-checkpoint "$RUN_ROOT/work/acquisition/logs/resume_checkpoint_20260716T172918.json"

UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark build --dump-path "$DUMP_PATH"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit run \
  --lineage-manifest work/lineage.json \
  --report work/audit/audit.md
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit status | tee "$RUN_ROOT/audit-status-v8.json"

UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select reserve \
  --exclusions "$RUN_ROOT/event-group-exclusions.json"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select review --batch-size 5
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select finalize
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select status | tee "$RUN_ROOT/selection-status.json"
```

The five-packet review batch changes only Codex request grouping; it keeps the frozen 50 packets and reviewer model
unchanged while remaining below the service request-size limit. Do not reuse the failed v7 reserve render, its review,
or any partial finalization output.

## Methodology v9 restart after the MISSING-sentinel defect

The v8 reserve rendered all 1,313 cases and 10,504 prompt cells with zero render failures, confirming the structural
few-shot guard repair. Finalization then exposed a temporal-classification defect: 142 IC-L deletion cases were
excluded solely because their hidden target used the generic `MISSING` sentinel and an unrelated support example
visibly used the same sentinel for an absent historical claim. Methodology v9 classifies this non-distinctive
prompt-contract vocabulary as diagnostic. Substantive future values and metadata remain blocking.

The corrected version-5 scanner was run against the complete retained v8 reserve render before freezing. It reduced
the high-risk population from 186 cases to 44 while leaving 400 substantive high-risk occurrences blocking. The clean
reserve has 276 IC-L, 423 IC-G, 345 IC-E-elim, and 225 T-box cases. This deterministically fills the 1,200-case main
population with effective quotas 249/406/320/225 and the nested 600-case Azure population with quotas
115/188/147/150.

Preserve the completed v8 audit and failed v8 selection, then bind downstream construction to v9 from the same
completed Stage 2 checkpoint:

```bash
: "${SOURCE_REPO:?Set SOURCE_REPO to the v9 checkout}"
: "${RUN_ROOT:?Set RUN_ROOT to the retained construction run}"
: "${DUMP_PATH:?Set DUMP_PATH to the 2026 dump}"

cd "$SOURCE_REPO"
mv "$RUN_ROOT/work/selections" "$RUN_ROOT/selections-v8-failed"
mv "$RUN_ROOT/work/audit" "$RUN_ROOT/audit-v8-complete"

UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark acquire \
  --dump-path "$DUMP_PATH" \
  --resume-checkpoint "$RUN_ROOT/work/acquisition/logs/resume_checkpoint_20260716T172918.json"

UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark build --dump-path "$DUMP_PATH"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit run \
  --lineage-manifest work/lineage.json \
  --report work/audit/audit.md
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit status | tee "$RUN_ROOT/audit-status-v9.json"

UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select reserve \
  --exclusions "$RUN_ROOT/event-group-exclusions.json"
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select review --batch-size 5
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select finalize
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select status | tee "$RUN_ROOT/selection-status.json"
```

Do not reuse the v8 prompt audit or review artifacts in the canonical v9 selection. The diagnostic v9 scan of the v8
render is evidence for the correction and quota feasibility only; canonical finalization consumes artifacts generated
under the v9 lock.

Final promotion is deliberately stricter than phase-local status commands. It validates all source, case, disposition,
ordering, replacement, and population records against the schemas copied into the release; replays Stage 2/3/4 lineage
and deterministic selection; checks complete source and cache provenance against the construction inputs; requires zero
unresolved systemic findings; and proves the 1,200/600 population sizes and nesting. It writes into a temporary sibling,
rebuilds `manifest.json` from the copied bytes, requires an exact byte match, and only then performs the atomic rename.
