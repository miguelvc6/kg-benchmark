# Audit and Selection

Every Stage 4 case receives one unique final disposition. Deterministic checks cover lineage, schema, classification
invariants, evidence support, contradictions, target leakage, temporal aliases, and group validity. A fixed label-hidden
Codex sample may discover new error patterns, but it does not certify ground truth. Only `include` is eligible.

## Canonical audit workflow

The paper-facing interface is a resumable sequence:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit prepare
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit deterministic
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit review
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark audit finalize
```

`audit run` executes or resumes all four phases, while `audit status` performs only hash verification. A completed phase
is reused only when every bound input and artifact still matches. A different dataset, protocol, schema, prompt source,
sample, review file, or disposition file fails closed instead of being silently mixed into the existing run.

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
next case in the frozen stratum order before finalization. The implementation of this reserve/finalize interface is
tracked in the repository-root checklist and must be complete before the final methodology lock.
