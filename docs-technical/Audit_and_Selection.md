# Audit and Selection

Every Stage 4 case receives one unique final disposition. Deterministic checks cover lineage, schema, classification
invariants, evidence support, contradictions, target leakage, temporal aliases, and group validity. A fixed label-hidden
Codex sample may discover new error patterns, but it does not certify ground truth. Only `include` is eligible.

## Canonical audit workflow

The paper-facing interface is a resumable sequence:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit prepare \
  --lineage-manifest work/lineage.json
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit deterministic
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit review
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit finalize
```

`audit run` executes or resumes all four phases, while `audit status` performs only hash verification. A completed phase
is reused only when every bound input and artifact still matches. A different dataset, protocol, schema, prompt source,
sample, review file, or disposition file fails closed instead of being silently mixed into the existing run.
`work/lineage.json` is the default lineage input; the explicit argument above documents the binding that final promotion
requires in the audit summary.

`prepare` ranks case IDs by SHA-256 using the protocol seed and writes a one-column, label-hidden 450-case construct
sample. It streams the current paper prompt builders over every case for repair proposal and track diagnosis under both
`logic_only` and `local_graph`. These audit renders are zero-shot because the independent few-shot support bank is
created by the later selection gate; every support-augmented reserve prompt is audited there. The render summary binds
the prompt sources, response schemas, case and world-state inputs, counts, and renderer contract.

`deterministic` exhaustively checks Stage 2–4 identity when Stage 2 is present, validates every case, scans all rendered
prompts, and creates blinded construct and distinct-case temporal review packets. `review` invokes the protocol-bound
Codex model with ephemeral, read-only, no-rules, no-user-config execution and records its exact CLI version. `finalize`
applies the conservative disposition precedence without relabeling and writes the canonical
`work/audit/dispositions.jsonl`, `work/audit/summary.json`, and repository-root `audit.md`.

The workflow manifest is `work/audit/workflow.json`. It binds all source inputs, active schemas, prompt sources,
samples, deterministic evidence, reviewer packets, Codex run report, normalized reviews, final dispositions, summary,
and release report by size and SHA-256. Lower-level evidence remains under `work/audit/deterministic/`; it is not a
second canonical disposition source.

Selection computes one SHA-256 seed-13 ordering per stratum over independent event groups. The initial main quotas are
230 IC-L, 375 IC-G, 295 IC-E-elim, and at most 300 T-box. If fewer independent T-box groups survive, the deficit is
transferred to A-box using the declared A-box weights and largest-remainder rounding. Other underfill fails. Azure uses
a nested 600-case view and applies the same rule if its T-box prefix underfills.

Before evaluation selection, the selector builds a 16-group A-box and 16-group T-box support bank round-robin over
registered roles. Every support group is excluded from all evaluation populations. Initial quotas are command options,
not source-code edits. Later population manifests select larger per-stratum prefixes with `kg-benchmark select expand`;
nondecreasing quotas must prove that the parent population is a subset.

The frozen policy first constructs a reserve of 276 IC-L, 450 IC-G, 354 IC-E-elim, and up to 360 T-box groups. Every
reserve prompt must pass deterministic scanning and a fixed 50-case temporal review. Failed prompts are replaced by the
next clean case in the frozen stratum order before finalization.

## Reserve and finalization workflow

The public sequence is:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select reserve \
  --exclusions <frozen-event-group-exclusions.json>
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select review
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select finalize
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select status
```

The exclusions file must validate against `group-exclusions.schema.json`. It may be empty only when no earlier study,
development run, or publication used a group; the selector hashes it even when it contains zero keys. A-box keys are
`ABOX|<QID>|<PID>` and T-box keys are `TBOX|<PID>|<revision>`.

`select reserve` validates the whole-data audit binding and complete final dispositions, applies prior-group
exclusions, chooses one representative per event group, and creates the independent support bank. It then materializes
the reserve and renders eight prompt cells per case: repair and diagnosis under both prompt regimes and both context
bundles. Every successfully rendered row is scanned deterministically; a render failure makes that case prompt-failed.
The temporal scanner selects 50 distinct cases by seed-13 ranking and writes blinded review packets.

`select review` is the only selection phase that calls Codex. It requires the final methodology lock and records the
reviewer model, Codex CLI version, run report, normalized reviews, and hashes. Any temporal verdict other than `pass` is
conservatively treated as a prompt failure. `select finalize` combines render, deterministic-scan, and review failures,
then advances through the audited reserve until every requested slot is filled by a clean `include` case. It fails
rather than reducing the declared 1,200/600 totals.

T-box ranks are frozen independently within relaxation-expansion, restriction-contraction, and schema-update queues.
After prompt failures are removed, their deterministic weighted merge restores the declared 130/50/120 main prefix
whenever each category has sufficient clean capacity. If the total T-box prefix underfills, the deficit transfers to
the three A-box strata by the configured weights and largest-remainder rounding. The same total-preserving rule applies
to Azure and later extensions.

The workflow lives under `work/selections/`. Its final population manifests bind the dataset, audit summary and
dispositions, prior-group exclusions, ranking, support bank, eligibility ordering, prompt audit, and complete per-case
eligibility artifact. They additionally embed the eligibility-record SHA-256 for every selected case. The Azure 600 is
proved to be a subset of the main 1,200. `select status` recomputes per-case eligibility digests, validates population
schemas and provenance, and rechecks prompt-clean membership and Azure nesting in addition to verifying file hashes.

Larger experiments use only the already prompt-audited clean reserve:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark select expand \
  --parent work/selections/main-1200.json \
  --name main-expanded \
  --quota-ic-l <N> --quota-ic-g <N> --quota-ic-e-elim <N> --quota-tbox <N> \
  --destination work/selections/main-expanded.json
```

Every requested quota must be nondecreasing from the parent after redistribution, the parent cases must remain nested,
and the destination must not already exist. Extending beyond the audited reserve requires a new, larger predeclared
reserve and repetition of the prompt-quality gate; unreviewed cases are never appended directly.
