# Audit and Selection

Every Stage 4 case receives one unique final disposition. Deterministic checks cover lineage, schema, classification
invariants, evidence support, contradictions, target leakage, temporal aliases, and group validity. A fixed label-hidden
Codex sample may discover new error patterns, but it does not certify ground truth. Only `include` is eligible.

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
