# MPU Implementation Tracker

## Purpose

This document is the living tracker for minimum-publishable-unit gaps between the conceptual program and the implemented repository.

Deletion rule:

- when a workstream reaches its exit criteria, remove it from this document
- move its implementation detail into the normal technical docs

## Current Implemented Scope

The repository now implements:

- Stage 1 candidate mining in `src/fetcher.py` and `src/lib/mining.py`
- Stage 2 repair reconstruction in `src/fetcher.py`
- Stage 3 popularity enrichment and world-state construction in `src/fetcher.py`, `src/lib/popularity.py`, and `src/lib/world_state.py`
- Stage 4 classification in `src/classifier.py`
- Stage 5 deterministic train/dev/test split generation in `src/splitter.py`
- A-box proposal validation in `guardian.patch_parser`
- T-box proposal validation in `guardian.tbox_parser`
- Benchmark evaluation in `src/evaluate.py`
- Zero-shot reasoning-floor execution in `src/reasoning_floor.py`

Implemented details now live in:

- [Proposal Validation](./Proposal_Validation.md)
- [Evaluation Harness](./Evaluation_Harness.md)
- [Reasoning Floor](./Reasoning_Floor.md)
- [Pipeline Implementation](./Pipeline_Implementation.md)

## Open Workstreams

No open MPU workstreams are currently tracked here.

If a future implementation gap appears relative to the conceptual docs, add it back as a new open workstream in this document instead of duplicating backlog detail elsewhere.

## Cross-Cutting Constraints

- Stick to the conceptual docs unless an actual conceptual ambiguity is discovered.
- Do not weaken benchmark validity rules for convenience.
- Prefer frozen benchmark artifacts and deterministic evaluation over live external dependencies.
- Keep the end goal explicit: a publishable research paper and a benchmark that is also practically useful.

## Exit Criteria

A workstream can be deleted from this tracker only when:

- the code path exists and is runnable
- tests cover the core contract
- the relevant technical docs have been updated
- the implementation remains aligned with the conceptual docs
