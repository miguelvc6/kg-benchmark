# MPU Implementation Tracker

## Purpose

This document is the living tracker for minimum-publishable-unit gaps between the conceptual program and the implemented repository.

It now also records the acceptance judgment for the current MPU and the next implementation checkpoints needed to keep building the rest of Track B on top of this repository.

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
- A-box vs T-box diagnosis normalization in `guardian.track_parser`
- Benchmark evaluation in `src/evaluate.py`
- Zero-shot reasoning-floor execution in `src/reasoning_floor.py`

Implemented details now live in:

- [Proposal Validation](./Proposal_Validation.md)
- [Evaluation Harness](./Evaluation_Harness.md)
- [Reasoning Floor](./Reasoning_Floor.md)
- [Track Diagnosis](./Track_Diagnosis.md)
- [Pipeline Implementation](./Pipeline_Implementation.md)

## MPU-1 Assessment

Current judgment:

- the repository is a valid first minimum publishable unit for Track B
- it does not implement the whole Track B research program
- it does implement the correct benchmark, baseline, and evaluation substrate for the rest of Track B

Why this counts as MPU-1-complete:

- the benchmark is built from historical A-box and T-box repair events
- the benchmark has frozen world-state context, temporal controls, and Type A/B/C classification
- the repository can normalize repair proposals and score them against benchmark-native targets
- the repository includes a pre-Guardian reasoning-floor baseline
- the repository includes a separate A-box vs T-box diagnosis task

What is intentionally not part of MPU-1:

- retrieval-augmented repair protocols
- Guardian-style multi-attempt verifier loops
- repair-time tool use
- cost-quality frontier experiments over tool calls and verifier calls
- end-to-end locus-controlled repair where the predicted track determines the repair path

## Track B Continuation Judgment

The current architecture is suitable for continuing Track B without redesigning the conceptual layer.

The repository already separates the main long-term concerns cleanly:

- benchmark construction
- proposal contracts
- offline evaluation
- zero-shot baseline execution

This means the rest of Track B can be added as new protocol layers around the existing benchmark and evaluation machinery rather than by rewriting the benchmark core.

## Open Post-MPU Workstreams

These are not blockers for MPU-1 acceptance. They are the next implementation checkpoints for the rest of Track B.

### Workstream 1: Verifier Runtime

Goal:

- introduce a repair-time verifier interface that can evaluate candidate transactions and emit structured rejection diagnostics during generation

Why it matters:

- Track B is defined as a generator-verifier program, not only as offline post hoc scoring

Current gap:

- [evaluate.py](./Evaluation_Harness.md) scores completed artifacts after generation
- the repository does not yet expose a live verifier loop that can reject a draft and return actionable diagnostics to the generator

Completion condition:

- the repository has a callable verifier component that can be used inside future protocol runners
- verifier outputs are machine-readable and aligned with benchmark validity rules

### Workstream 2: Protocol Runners Beyond the Reasoning Floor

Goal:

- add separate runners for the next Track B protocol families without collapsing them into the reasoning-floor runner

Priority protocol families:

- one-shot with retrieval
- bounded retrieval-augmented generation
- bounded Guardian-style iterative repair

Why it matters:

- the reasoning floor must remain the stable pre-intervention control condition
- later protocols need to be compared against it, not replace it

Completion condition:

- new runners exist beside `src/reasoning_floor.py`
- the zero-shot baseline remains unchanged and comparable

### Workstream 3: Actionable Locus Selection

Goal:

- turn A-box vs T-box diagnosis from a separate analytic task into an optional control point for repair generation

Why it matters:

- the broader Track B proposal treats locus selection as part of the repair problem

Current gap:

- diagnosis is currently scored separately
- proposal generation still uses the historical benchmark `track` to choose the proposal schema and route

Completion condition:

- a future protocol runner can use the predicted track to determine whether to attempt an A-box repair, a T-box reform, or abstention
- evaluation can attribute failures to wrong locus choice versus wrong repair content

### Workstream 4: Evidence and Retrieval Layer

Goal:

- add retrieval-backed evidence collection and structured provenance for Type C and evidence-heavy T-box cases

Why it matters:

- Track B is specifically about verifier-guided LLM repair with evidence and provenance

Current gap:

- provenance fields exist in proposal schemas
- the current baseline does not perform retrieval or evidence-grounded proposal construction

Completion condition:

- protocol runners can attach retrieved evidence to proposals
- evidence provenance is logged in a normalized form
- evaluation can distinguish unsupported proposals from evidence-backed ones

### Workstream 5: Cost-Aware Protocol Evaluation

Goal:

- extend evaluation from token-only accounting to full protocol-cost accounting

Required future measures:

- attempt count
- tool call count
- verifier invocation count
- conversion rate across attempts
- tokens-to-fix

Why it matters:

- Track B is not only about correctness; it is also about reliability under bounded operational cost

Completion condition:

- protocol runs emit structured cost telemetry
- evaluation summaries report cost-quality trade-offs consistently across protocol families

## Repository Design Rules for the Next Phases

To preserve compatibility with the conceptual program and keep the repository useful as a benchmark, future Track B work should follow these rules:

- keep benchmark construction independent from protocol experimentation
- keep the reasoning floor as a stable pre-intervention baseline
- add new protocol runners beside the existing baseline runner instead of overloading it with all future behaviors
- treat offline evaluation as the benchmark authority even after a live verifier runtime is introduced
- keep proposal schemas stable and extend them only when the research design actually requires it
- preserve deterministic artifact generation and link consistency across docs
- do not edit conceptual docs unless a real research decision has changed

## Open Workstreams Policy

There are no open MPU-1 blockers.

The workstreams above are continuation workstreams for Track B after MPU-1. As they are implemented, this document should be shortened again and the implementation details moved into the regular technical docs.

## Cross-Cutting Constraints

- Stick to the conceptual docs unless an actual conceptual ambiguity is discovered.
- Do not weaken benchmark validity rules for convenience.
- Prefer frozen benchmark artifacts and deterministic evaluation over live external dependencies.
- Keep the end goal explicit: a publishable research paper and a benchmark that is also practically useful.

## Exit Criteria

A continuation workstream can be deleted from this tracker only when:

- the code path exists and is runnable
- tests cover the core contract
- the relevant technical docs have been updated
- the implementation remains aligned with the conceptual docs
