# Scientific Vision

WikidataRepairEval studies whether language models can perform knowledge-graph repair as a controlled edit problem. The project is not primarily a leaderboard and should not be framed as a generic LLM benchmark. It is a measurement instrument for testing whether a model can identify the right repair layer, use the right evidence, avoid temporal leakage, and emit an auditable transaction that survives symbolic checks.

## Thesis

Knowledge-graph repair is not the same task as link prediction, triple classification, or question answering. A repair system must decide:

- whether the error is in an entity statement or in a property constraint;
- whether the repair is rule-implied, locally grounded, or dependent on non-local evidence;
- whether the available context is sufficient or the system should abstain or request retrieval;
- whether the proposed edit is executable and preserves useful information;
- whether the rationale, provenance, and uncertainty are auditable.

The benchmark operationalizes this thesis using historical Wikidata repair events, frozen 2026 world-state context, temporal target-property reconstruction, structured proposal contracts, and symbolic evaluation.

## Relation To Prior Repair Work

WikidataRepairEval adopts the A-box/T-box repair distinction and related nomenclature from existing Wikidata repair-taxonomy work, especially Ferranti et al. (2025), *Formalizing Repairs for Wikidata Constraint Violations: A Taxonomy and Empirical Analysis*.

The novelty claim is not that A-box and T-box repair are new. The contribution is that this repair view is turned into an LLM-facing evaluation protocol with:

- repair-locus diagnosis;
- information-access labels;
- controlled context ablations;
- temporal leakage safeguards;
- executable A-box and T-box repair contracts;
- stratified evaluation of parse validity, executability, historical alignment, semantic compatibility, and auditability.

## First-Paper Scope

The first publishable unit should establish the benchmark and the reasoning-floor evaluation, not a full retrieval or Guardian-style repair agent.

It should include:

- benchmark construction from real Wikidata repair events;
- repair-locus and information-access taxonomy alignment;
- classifier audit and label validation;
- full, core, dev/pilot, and audit dataset tiers;
- zero-shot reasoning-floor experiments;
- `logic_only` vs `local_graph` context ablations;
- oracle-track vs diagnosis-routed repair;
- local H100-runnable model baselines;
- a small stratified API reference subset if budget allows;
- failure analysis and limitations.

Retrieval-augmented repair, verifier-guided retry loops, and the full Guardian protocol are natural follow-up work unless a small contingency experiment is needed.

## Research Questions

### RQ1. Does Information Need Predict Model Behavior?

Rule-implied, local-context, and external or unresolved cases should behave differently under controlled context ablations. Local graph context should help most on genuinely local cases, while no-retrieval Type C cases should remain difficult or require abstention.

### RQ2. Can Models Choose The Correct Repair Locus?

Models must distinguish A-box entity repair from T-box schema reform. The gap between oracle-track repair and diagnosis-routed repair measures the cost of choosing the wrong layer.

### RQ3. Do Proposals Survive Symbolic Transaction Checks?

Valid JSON is not enough. A proposal can parse while still targeting the wrong entity or property, deleting useful information, failing executability, hallucinating provenance, or missing the historical repair state.

### RQ4. How Much Does Context And Prompt Design Matter?

The main baseline is zero-shot contract prompting. Few-shot prompting is an ablation that tests precedent adaptation and contract compliance, not the base reasoning floor.

### RQ5. Does Popularity Expose Memorization Risk?

Head entities may be easier in no-retrieval settings because of parametric memory. Tail-entity behavior is more diagnostic of context use, local reasoning, and abstention under insufficient evidence.

## Design Commitments

- Use real historical Wikidata repair events rather than synthetic-only cases.
- Separate repair locus from information-access condition.
- Treat Type C conservatively unless manual audit or retrieval confirms external evidence need.
- Reconstruct the edited target property as a historical pre-repair state instead of exposing current post-repair values.
- Report stratified metrics rather than a single aggregate leaderboard score.
- Control repeated T-box schema reforms so one property revision cannot dominate paper-facing results.
- Treat historical repairs as historically accepted targets, not universal truth.

The taxonomy details are summarized in [Benchmark Taxonomy](./Benchmark_Taxonomy.md), evaluation details in [Evaluation Framework](./Evaluation_Framework.md), and temporal leakage policy in [Temporal Validity](./Temporal_Validity.md).
