# Automated Consistency Audit

## Status And Scope

The `kg-automated-audit` command implements the repository's no-human validation fallback. It is an exploratory
consistency and error-discovery gate, not semantic ground-truth validation.

The workflow searches for cross-artifact inconsistencies before release review. It has three subcommands:

1. `run` loads the full inputs, records their identity, and performs deterministic checks.
2. `review-codex` uses Codex to discover plausible errors in review shards.
3. `finalize` consolidates deterministic and model-assisted findings under a conservative disposition policy.

Codex output is error-discovery evidence only. It is not an independent annotation, an adjudication, human construct
validation, semantic gold validation, or authorization to change benchmark labels or paper claims.

## Full-Data Inputs

Run the audit against these current artifacts:

| Role | Path |
| --- | --- |
| Stage 4 classified benchmark | `data/04_classified_benchmark.jsonl` |
| Stage 3 world state | `data/03_world_state.json` |
| Stage 4 schema | `schemas/04_classified_benchmark.schema.json` |
| Current construct sample | `reports/manual_audit/audit_phase_d_v1_seed_13.csv` |
| Current rendered prompts | `reports/temporal_audit/current_holdout96_v4_rendered/prompt_dev_rendered_prompts.jsonl` |
| Current render summary | `reports/temporal_audit/current_holdout96_v4_rendered/prompt_dev_render_summary.json` |

The current full-data checkout has no Stage 2 artifact. The audit must record Stage 2 as unavailable and must not use
the synthetic `data_sample/` or release sample as a substitute. Checks requiring Stage 2 therefore remain not run or
unresolved, never passed by inference.

All generated material belongs under `reports/automated_audit/full_v1/`. Treat that directory as one audit unit: retain
the input manifest, deterministic findings, review shards, Codex responses, consolidated dispositions, errors and
retry records, and final machine-readable and human-readable summaries together. Do not mix artifacts from another
input set or audit version into this directory.

## WSL Commands

Verify the installed interface before starting:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-automated-audit --help
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-automated-audit run --help
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-automated-audit review-codex --help
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-automated-audit finalize --help
```

Run deterministic collection and consistency checks with every currently available full-data input:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-automated-audit run \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --stage4-schema schemas/04_classified_benchmark.schema.json \
  --construct-sample reports/manual_audit/audit_phase_d_v1_seed_13.csv \
  --rendered-prompts reports/temporal_audit/current_holdout96_v4_rendered/prompt_dev_rendered_prompts.jsonl \
  --render-summary reports/temporal_audit/current_holdout96_v4_rendered/prompt_dev_render_summary.json \
  --output-dir reports/automated_audit/full_v1
```

Stage 2 is intentionally omitted because the full artifact is absent. The `run` output must preserve that limitation.

Use the approved Codex review configuration. A shard contains at most 10 review cases, three workers may run in
parallel, and a failed request receives at most two retries:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-automated-audit review-codex \
  --manifest reports/automated_audit/full_v1/manifest.json \
  --model gpt-5.6-sol \
  --shard-size 10 \
  --workers 3 \
  --retries 2
```

After the Codex step completes or its failures are recorded, consolidate the audit:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-automated-audit finalize \
  --manifest reports/automated_audit/full_v1/manifest.json \
  --reviews reports/automated_audit/full_v1/codex_reviews.jsonl
```

Do not delete failed shards or rerun into the same directory with changed inputs, model, or concurrency settings. Start
a new versioned audit directory when any bound input or review configuration changes.

## Conservative Disposition Policy

Every finding retains its source, affected record, evidence, severity, and disposition. Deterministic integrity errors
become `exclude`; deterministic label disagreements and unsupported reconstruction become `diagnostic`; Codex construct
concerns or uncertainty become `diagnostic`; suspected temporal leakage becomes `exclude_pending_rerender`; otherwise the
case remains `include`. The finalizer never relabels a case or emits `EXTERNAL_CONFIRMED`.

Malformed or incomplete Codex responses and exhausted retries fail closed. Fix the runner or start a new versioned audit
directory; never infer missing reviews as passes. Re-run the entire audit after any bound input changes.

## Release Interpretation

A completed audit supports the narrow statement that the supplied artifacts underwent the recorded deterministic and
Codex-assisted consistency checks. It does not establish benchmark correctness, human construct validity,
inter-annotator agreement, semantic correctness, causal necessity, or repair uniqueness. The accepted no-human scope and
remaining release controls are defined in the [Research Release Protocol](./Research_Release_Protocol.md).
