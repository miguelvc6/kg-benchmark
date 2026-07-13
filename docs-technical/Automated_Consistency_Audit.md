# Automated Consistency Audit

Version 2 validates Stage 2 content when `--stage2` is supplied: exact Stage 2/3/4 identity, ordering, and Stage 2-to-lean-
Stage 4 field projection must pass. It also checks format-rule contradictions, cardinality conflicts, unsupported independent
local evidence, missing T-box history, and report/constraint disagreement. The temporal scanner covers embedded, decoded,
encoded, and semantic-normalized values in addition to exact tokens. See the
[Confirmatory Release Runbook](./Confirmatory_Release_Runbook.md) for the release-gated sequence.

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
| Stage 0 popularity | `data/00_entity_popularity.json` |
| Stage 1 candidates | `data/01_repair_candidates.json` |
| Stage 2 compiled JSON | `data/02_wikidata_repairs.json` |
| Stage 2 recovered JSONL | `data/02_wikidata_repairs.jsonl` |
| Stage 4 classified benchmark | `data/04_classified_benchmark.jsonl` |
| Stage 3 world state | `data/03_world_state.json` |
| Stage 0--4 lineage manifest | `reports/lineage/restored_v2.json` |
| Stage 4 schema | `schemas/04_classified_benchmark.schema.json` |
| Current construct sample | `reports/manual_audit/audit_phase_d_v1_seed_13.csv` |
| Current rendered prompts | `reports/temporal_audit/current_holdout96_v4_rendered/prompt_dev_rendered_prompts.jsonl` |
| Current render summary | `reports/temporal_audit/current_holdout96_v4_rendered/prompt_dev_render_summary.json` |

The restored baseline contains two Stage 2 representations. They are validation inputs, not interchangeable sources:
`kg-artifact-lineage validate` must compare them and the audit must bind that result. A representation, provenance, or
Stage 2/3/4 projection failure makes the dataset audit fail even when the compiled Stage 2/3/4 chain itself is exact. Never
rewrite a recovered artifact to make the gate pass, and never substitute `data_sample/` or the release sample.

Use a new versioned directory for every audit iteration. Treat each directory as one audit unit: retain
the input manifest, deterministic findings, review shards, Codex responses, consolidated dispositions, errors and
retry records, and final machine-readable and human-readable summaries together. Do not mix artifacts from another
input set or audit version into this directory.

## Recorded `full_v1` Result

The full audit completed on 2026-07-13. The manifest binds the deterministic run to commit `5753446`, Stage 4 SHA-256
`dfbdc9286cc54e10dfe7092ecdf78f95e9852b9271c564b3dfa2447b4487f344`, and Stage 3 SHA-256
`7e3591a9bfc7289c698d1cc87339604b3c796a042d8ce96bb0d67f435bfa21bf`. It covered all 535,570 Stage 4 rows, all
535,570 Stage 3 entries, and all 384 supplied rendered prompts. Stage 2 remained explicitly unavailable.

Deterministic status counts were:

| Status | Cases | Interpretation |
| --- | ---: | --- |
| `pass` | 473,412 | All implemented checks for the case reproduced. |
| `unsupported` | 60,537 | The current replay does not implement enough evidence to decide; this is not agreement. |
| `disagreement` | 1,621 | The implemented replay contradicted the stored class/subtype evidence. |
| `error` | 0 | No schema, identity, or join failure was found. |

The 1,621 deterministic disagreements comprise 350 Type A rule-replay failures, 109 Type B cases whose local support was
not reproduced, and 1,162 Type C cases with visible local support. Major coverage gaps include 33,525 T-box cases requiring
causality-policy replay, 16,741 unsupported Type A subtypes, 5,402 unknown Type C cases, and 4,993 derived-text cases that
require semantic review. The finding stream also records 42 missing one-of constraints and four missing or invalid format
constraints. Finding categories can overlap and must not be summed as a partition.

The expanded temporal scan checked 3,024 hidden claims across 96 cases and 384 prompts. It found zero deterministic
high-risk hits, 672 diagnostic hits, and 156 expected rule-derived hits. Exact identifier, normalized-label, serialized-ID,
and substring-boundary mutation checks all fired as expected.

Codex CLI `0.144.3` with model `gpt-5.6-sol` reviewed 450 blinded construct packets and 50 temporal packets in 50 batches;
all batches succeeded on their first attempt. Construct verdicts were 221 `pass`, 193 `concern`, and 36 `uncertain`.
Temporal verdicts were 39 `pass`, five `suspected_temporal_leakage`, and six `uncertain`. The five leakage packets map to
four benchmark cases. They include direct target-value exposure, a target identifier embedded in a visible URL, and hidden
replacement labels/descriptions exposed under another QID. These mechanisms explain why a zero-hit deterministic gate is
not a semantic no-leakage guarantee.

Conservative finalization produced 473,222 `include`, 62,344 `diagnostic`, and four `exclude_pending_rerender` case
dispositions. There were no integrity exclusions and no labels were changed. The construct sample is stratified error
discovery, not a population estimate: notably, 148 concerns occurred among deterministic-pass cases, while 18 Codex passes
occurred among deterministic disagreements. This disagreement in both directions is evidence for follow-up, not model
adjudication.

## Recorded Restored-Baseline `full_v2c` Result

The v2c deterministic rerun covered all 535,570 compiled Stage 2/3/4 cases and bound the corrected restored-lineage
manifest. It reported 472,790 pass, 56,761 unsupported, 6,019 disagreement, and zero error cases. The temporal gate passed
for 384 supplied prompts, but the audit manifest failed because the recovered Stage 2 JSON and JSONL representations and
Stage 0 provenance do not pass complete lineage. No v2 dispositions or confirmatory selection were certified from this
run. The full population flow, finding counts, and residual limits are recorded in [`audit.md`](../audit.md).

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
  --output-dir reports/automated_audit/full_v2c \
  --stage2 data/02_wikidata_repairs.json \
  --lineage-manifest reports/lineage/restored_v2.json
```

This restored-baseline command is expected to exit nonzero while its complete lineage manifest fails. It still writes the
full deterministic result for remediation evidence. A confirmatory post-freeze run must use its own passing v2 lineage and
snapshot manifests, not this restored manifest.

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
