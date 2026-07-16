# Data Pipeline

`kg-benchmark acquire` mines report candidates, reconstructs historical repairs with fail-closed API access, computes
popularity, and extracts world state from a checksum-bound dump. All outputs, caches, logs, and resume state are scoped
to `work/`. A partial JSONL is never treated as complete; completion state is explicit and resumable.

After Stage 1 violation-type deduplication, acquisition groups candidates by property, old/new report revision, and fix
date. It retains every QID from groups of at most 100 and deterministically selects 100 from larger groups using
seed-13 SHA-256 QID ranking. The sampled `01_repair_candidates.json` replaces the temporary full candidate array
atomically before Stage 2 begins. Per-candidate sampling provenance makes the operation idempotent on Stage 2 resume.

Stage 2 retries transient history, snapshot, and persistence requests under the frozen transport policy. If those
retries are exhausted, the candidate is appended to `02_stage2_exclusions.jsonl` as `upstream_unavailable`, flushed,
and checkpointed immediately; it is never logged as `no_history`, `no_diff`, or a terminally missing entity. The final
`02_stage2_exclusions.json` is hash-bound in source provenance with its record count. Exclusion candidate keys are
loaded as completed work on resume. Ordinary progress flushes stats, repairs, exclusions, and an atomic checkpoint every
100 processed candidates, while an upstream exclusion forces an immediate checkpoint.

When Stage 2 resumes from an earlier acquisition attempt, `work/acquisition-config.json` binds the prior acquisition
configuration, sampled candidate artifact, partial repair JSONL, supplied stats/checkpoint, and their hashes under the
`compatible_completed_prefix_v1` policy. This permits the current v4 completed prefix to be reused under a
robustness-only replacement freeze without presenting the prefix as newly computed or reusing the incomplete failing
candidate.

`kg-benchmark build` classifies the acquired records and emits canonical JSONL. The release roles are popularity,
candidates, repairs, world state, and cases. It also records the completed acquisition arguments, methodology lock,
original Stage 0–3 and dump hashes, cache inventory, Git revision, and a passing Stage 0–4 lineage manifest under
`work/`. SQLite may be generated as a disposable read index but is never canonical or published alongside an equivalent
JSON artifact.

The audit and selection gates verify source provenance, record identity, Stage 2–4 projection, disposition coverage,
support exclusions, and selection manifests before promotion. `kg-benchmark promote` then verifies that every canonical
role is present, validates every JSON or JSONL record against the active schema, and independently reconstructs the
lineage, eligibility order, support bank, reserve, prompt-clean order, replacements, and final populations. It copies
only canonical roles, the frozen protocol and lock, and their active schemas to a temporary sibling; verifies original
source/cache provenance while those construction inputs remain available; writes artifact hashes and counts into the
version-2 dataset manifest; reproduces that manifest byte-for-byte; and atomically renames the candidate to `dataset/`.

The released source representation is JSONL only. Stage 2 JSON and JSONL equivalence is proven during build; promotion
also streams the canonical Stage 0, Stage 1, and Stage 3 rows against their provenance-bound construction JSON. The
original acquisition JSON remains a construction input rather than a duplicate published dataset. Schema validation
and representation comparison are streaming or disk-indexed, so the gate does not load the multi-gigabyte source roles
into memory.
