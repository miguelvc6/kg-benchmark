# Phase A Completion: Project Alignment and Paper Scope

**Project:** WikidataRepairEval 1.0  
**Phase:** A — project alignment and paper scope  
**Status:** Completed  
**Date:** 2026-05-21

## A1 — Paper 1 scope note

### Paper 1 main claim

WikidataRepairEval operationalizes an existing Wikidata repair taxonomy into an LLM-facing benchmark and evaluation protocol for knowledge-graph repair. The paper studies whether models can choose the correct repair locus, use the appropriate information condition, and emit executable, auditable graph-repair transactions under temporal leakage controls.

A concise paper thesis:

> Knowledge-graph repair is not merely triple prediction. It is a controlled edit problem over repair locus, information access, temporal validity, and transaction safety. WikidataRepairEval measures those dimensions using historical Wikidata repair events and symbolic evaluation of generated repair proposals.

### In-scope contributions for paper 1

1. **Benchmark construction from historical Wikidata repairs.** The paper should describe how historical repair events are reconstructed and converted into evaluation cases.
2. **Adoption and operationalization of prior Wikidata repair taxonomy.** The paper should use the existing A-box/T-box repair-locus vocabulary as the conceptual backbone rather than claiming a new repair taxonomy.
3. **Information-condition layer for LLM evaluation.** The Type A/B/C layer should be described as an information-access condition: rule-implied, local graph-grounded, or external/non-local/unknown-by-elimination.
4. **Temporal leakage controls.** The paper should emphasize that the target property is reconstructed into its pre-repair state for prompt context rather than exposing post-repair target-property values.
5. **Executable repair contracts.** The benchmark evaluates structured A-box and T-box repair proposals, not just free-text answers.
6. **Reasoning-floor experiments.** Main experiments should use zero-shot schema-constrained prompting with logic-only and local-graph ablation bundles.
7. **Oracle-track versus diagnosis-routed comparison.** The paper should distinguish repair ability under the known historical track from full pipeline behavior where the model must choose the repair locus.
8. **Classifier validation and audit.** The paper should include classifier transition analysis and manual audit, especially for Type C and delete/format cases.
9. **Prompt development and few-shot ablation.** Few-shot prompting should be treated as an ablation on the core/dev subset, not as the headline benchmark condition.
10. **Local open-model baseline and small API reference subset.** The H100-backed local experiments should carry the main analysis; API calls should be used for calibration.

### Out-of-scope contributions for paper 1

1. **Full retrieval-augmented repair protocol.** Type C cases may motivate retrieval, but RAG should remain future work unless a minimal contingency experiment is needed.
2. **Full Neuro-Symbolic Guardian agent.** Verifier-guided multi-step repair should be paper 2, not paper 1.
3. **Full cost-quality frontier.** The reasoning-floor runner can record tokens/cost, but cost-quality optimization belongs to the follow-up protocol paper.
4. **Live Wikidata editing or deployment.** The paper evaluates offline repair proposals only.
5. **Claiming historical edits are universal truth.** The benchmark target is the historically accepted repair, not a proof of ontological correctness.
6. **Claiming Type C is confirmed retrieval need without audit.** Current Type C semantics must be qualified as external-by-elimination or unknown unless manually or retrieval-confirmed.
7. **Claiming novelty for the A-box/T-box distinction.** The paper’s novelty is the LLM-facing benchmark/protocol, not the basic repair-locus taxonomy.

### Supervisor-aligned positioning

The paper should explicitly state that it adopts the existing Wikidata repair-taxonomy work as the repair-locus vocabulary. The project contribution is to convert that conceptual taxonomy into an evaluation instrument for LLM-based repair: frozen-context cases, information-condition labels, structured proposal contracts, symbolic evaluators, prompt ablations, and repair-locus diagnosis.

Recommended positioning sentence:

> We build on the existing taxonomy of Wikidata constraint-violation repairs and ask whether language models can operationalize it: identify whether the repair belongs at the instance or schema layer, determine whether the supplied context is sufficient, and emit a transaction that can be checked symbolically.

## A2 — Taxonomy mapping note

### Core distinction

The paper must keep two axes separate:

| Axis | Paper role | Source | What it answers |
|---|---|---|---|
| **Repair locus / repair taxonomy** | Main repair-type vocabulary | Existing Wikidata repair-taxonomy work | Should the repair edit an entity statement, a property constraint, or be treated as ambiguous? |
| **Information condition** | LLM evaluation layer | WikidataRepairEval | What information would a model need to reproduce or justify the historical repair target under controlled context? |

### Repository-to-paper terminology

| Repository term | Paper-facing term | Use in paper | Notes |
|---|---|---|---|
| `A_BOX` | A-box / instance-level repair | Main repair-locus label | Use prior repair-taxonomy nomenclature. Do not claim novelty. |
| `T_BOX` | T-box / schema-level repair | Main repair-locus label | Use prior repair-taxonomy nomenclature. Do not claim novelty. |
| `AMBIGUOUS` | Ambiguous or mixed repair-locus evidence | Diagnosis label / limitation slice | Use when A-box and T-box evidence overlap or the historical causal route is not unique. |
| `TypeA` | IC-L: rule-implied information condition | A-box information-condition label | The repair signal is supplied by the rule, violation shape, or deterministic normalization. |
| `TypeB` | IC-G: local graph-grounded information condition | A-box information-condition label | The historical repair target is visible in the focus node, pre-repair target state, one-hop graph, or local labels/descriptions. |
| `TypeC` | IC-E / IC-U: non-local, external-by-elimination, or unknown information condition | A-box information-condition label | Must be split or qualified. Current Type C is often a residual bucket, not confirmed external evidence. |
| `EXTERNAL` | Avoid as an unqualified paper term | Replace with split subtypes | Unqualified `EXTERNAL` overclaims. |
| `EXTERNAL_BY_ELIMINATION` | IC-E-elim: external by negative local/rule scan | Main Type C subtype after redesign | Means supported local/rule evidence did not identify the target. It is not confirmed retrieval need. |
| `EXTERNAL_CONFIRMED` | IC-E-confirmed: externally confirmed | Manual-audit or retrieval-confirmed subtype | Use only after positive evidence that non-local evidence is required. |
| `UNKNOWN_MISSING_WORLD_STATE` | IC-U missing context | Diagnostic/challenge slice | Should not be part of main core scoring unless separately reported. |
| `UNKNOWN_MISSING_TRUTH` | IC-U missing historical target | Diagnostic/challenge slice | Indicates weak benchmark metadata, not externality. |
| `UNKNOWN_CURRENT_VALUE_FALLBACK` | IC-U current-value fallback | Diagnostic/challenge slice | Indicates classification depended on contemporary value fallback. |
| `UNKNOWN_INCOMPLETE_LOCAL_CONTEXT` | IC-U sparse local context | Diagnostic/challenge slice | Indicates local absence is not strong enough evidence. |
| `REJECTION` | Rule-invalid rejection / delete-as-cleaning | Needs refinement | Do not treat every delete as high-confidence logical rejection. |
| `LOGICAL` | Rule-implied repair | Type A subtype | Should require deterministic rule signal. |
| `LOCAL_*` | Local evidence subtype | Type B subtype | Should record match source and match kind. |
| `RELAXATION_*` / `RESTRICTION_*` | T-box schema-reform direction | T-box subtype | Report separately from A-box information conditions. |
| `SCHEMA_UPDATE` | Generic schema update | T-box subtype | Use lower confidence when direction or causality is weak. |
| `COINCIDENTAL_SCHEMA_CHANGE` | Weakly causal or coincidental schema edit | T-box diagnostic subtype | Report separately from clean T-box reform. |

### Paper-facing terminology policy

Use `Type A/B/C` only if the paper explicitly defines them as information conditions. Prefer `IC-L`, `IC-G`, and `IC-E/IC-U` in conceptual sections. The code may keep `TypeA`, `TypeB`, and `TypeC`, but the paper should avoid presenting them as a rival repair taxonomy.

Recommended definitions:

| Paper label | Definition | Positive evidence required? |
|---|---|---|
| **IC-L: rule-implied** | The historical target can be inferred from a constraint, violation shape, or deterministic normalization. | Yes: supported rule signal. |
| **IC-G: local graph-grounded** | The historical target is available in the supplied local graph context after temporal leakage controls. | Yes: exact id/local value/local text evidence. |
| **IC-E-elim: external by elimination** | The historical target is available from historical repair metadata, but supported rule/local extractors do not find it. | Negative evidence only; must be audited. |
| **IC-E-confirmed: externally confirmed** | Manual audit or retrieval confirms non-local evidence is required. | Yes: external source or expert audit. |
| **IC-U: unknown / weak artifact** | The benchmark artifact is too incomplete or ambiguous to claim local, rule, or external evidence. | No; diagnostic only. |

### Novelty-safe contribution statement

Do not write:

> We introduce a taxonomy of A-box and T-box Wikidata repairs.

Write:

> We instantiate an existing Wikidata repair taxonomy as an LLM evaluation protocol, adding information-condition labels, temporal prompt controls, structured proposal contracts, and symbolic transaction-level evaluation.

## A3 — Finalized research questions, hypotheses, metrics, and contrasts

### Decision on existing hypotheses

Use the hypotheses in `00-kg_llm_benchmark.md` as the **scaffold**, but update them before treating them as paper-final. They are directionally correct, but after the classifier audit they need three changes:

1. Replace unqualified “Type C external-evidence cases” with `IC-E-elim`, `IC-E-confirmed`, and `IC-U` terminology.
2. Add explicit construct-validity hypotheses for classifier redesign and manual audit.
3. Attach every hypothesis to a metric and experimental contrast so that Task A3 is executable rather than rhetorical.

The updated hypotheses below should be used as the paper-facing version.

### Validation question VQ0 — Are benchmark labels reliable enough for scientific claims?

This is a validation precondition rather than the main model-performance question.

| Hypothesis | Metric | Experimental contrast / evidence | Interpretation |
|---|---|---|---|
| **VQ0.1.** Splitting Type C into external-by-elimination and unknown subtypes will reveal that a non-trivial fraction of old Type C cases were artifact weakness rather than confirmed externality. | Old-vs-new transition counts; fraction of old `TypeC/EXTERNAL` routed to `IC-U`; manual audit precision. | Old classifier versus redesigned classifier; audit sample over Type C strata. | Supports a more cautious Type C claim and prevents overclaiming retrieval need. |
| **VQ0.2.** Expanding local evidence extraction will convert some old Type C cases into IC-G local-context cases. | `TypeC -> TypeB/IC-G` transition rate; local match source distribution; audit false-positive rate. | Old classifier versus redesigned classifier after local bucket expansion. | Estimates how much old Type C was caused by extractor incompleteness. |
| **VQ0.3.** Refining delete and format logic will downgrade some old high-confidence Type A cases. | `TypeA -> IC-U/IC-G/IC-E-elim` transition rate; delete/format audit precision. | Old classifier versus redesigned classifier; targeted manual audit. | Prevents overclaiming “logical” repairs when choosing the deletion/replacement needs evidence. |

### RQ1 — Does information condition predict model behavior?

**Question.** Do models behave differently on rule-implied, local graph-grounded, and non-local/unknown repair cases?

| Hypothesis | Metric | Experimental contrast | Expected result |
|---|---|---|---|
| **H1.1.** IC-L rule-implied cases should be easiest under logic-only context. | Accepted repair rate; exact historical agreement; executable proposal rate. | IC-L versus IC-G/IC-E-elim/IC-U under `logic_only`, oracle-track mode. | IC-L should have the highest repair success and lowest need for local context. |
| **H1.2.** IC-G local graph-grounded cases should show the largest gain from local graph context. | `local_graph - logic_only` delta in accepted repair and exact match; paired bootstrap CI. | Same cases under `logic_only` and `local_graph`. | Largest positive delta should appear for IC-G. |
| **H1.3.** IC-E-elim cases should remain difficult without retrieval; IC-U cases should have low repair success and high abstention/uncertainty if abstention is allowed. | Accepted repair rate; exact match; unsupported/hallucinated provenance rate; abstention precision if implemented. | No-retrieval reasoning floor across IC-E-elim and IC-U slices. | High no-retrieval success should trigger leakage, memorization, or classifier-audit review. |
| **H1.4.** Large local-graph gains on IC-L or IC-E-elim indicate either classifier weakness, prompt leakage, or hidden local evidence. | Context-gain anomaly rate; inspected examples. | `local_graph - logic_only` by class/subtype, followed by manual inspection. | Unexpected gains become a diagnostic signal, not just a positive score. |

### RQ2 — Can models choose the correct repair locus?

**Question.** Can a model distinguish entity-level A-box repair from schema-level T-box repair?

| Hypothesis | Metric | Experimental contrast | Expected result |
|---|---|---|---|
| **H2.1.** Models will overuse A-box repairs when the prompt contains a concrete violating entity and value. | Track accuracy; macro-F1; confusion matrix; T-box recall; A-box overprediction rate. | Track-diagnosis outputs versus historical track on core set. | T-box cases will often be misrouted as A-box. |
| **H2.2.** T-box diagnosis will be hardest for generic or weakly causal schema updates. | T-box recall by subtype; macro-average by property revision. | T-box subtype slices: directional reforms versus `SCHEMA_UPDATE`/`COINCIDENTAL_SCHEMA_CHANGE`. | Clear directional reforms should be easier than generic/weakly causal changes. |
| **H2.3.** Local graph context can hurt T-box diagnosis if it focuses the model on the violating entity rather than the constraint. | Track accuracy delta by track and bundle. | Track diagnosis under `logic_only` versus `local_graph`. | Some models may become more A-box-biased with local graph context. |

### RQ3 — Are valid-looking LLM outputs executable and auditable graph transactions?

**Question.** Are generated repairs merely parseable, or do they survive symbolic transaction-level checks?

| Hypothesis | Metric | Experimental contrast | Expected result |
|---|---|---|---|
| **H3.1.** Valid JSON will substantially overestimate true repair success. | Parse validity; schema validity; executability; accepted repair; exact match. | Stepwise funnel from raw output to accepted proposal. | Many proposals will parse but fail later checks. |
| **H3.2.** T-box exact signature match will be low, but semantic-family success will be higher. | Exact signature match; target constraint match; semantic-family success; signature Jaccard. | T-box evaluation under oracle-track mode. | Models may identify the reform family/direction without reproducing the exact historical signature. |
| **H3.3.** Some A-box proposals will satisfy the violation destructively by deleting or overwriting useful information. | Destructive-op rate; information-preservation checks; over-delete rate. | A-box traces by operation type and violation family. | Exact/accepted repair should be reported alongside preservation metrics. |
| **H3.4.** Auditability fields will be weaker than syntactic validity. | Rationale/provenance/uncertainty completeness; hallucinated provenance rate. | Valid proposals versus auditable proposals. | Models can produce valid shapes without trustworthy evidence statements. |

### RQ4 — Does local graph context help the correct classes?

**Question.** Does adding local graph context improve the classes it should improve, rather than producing indiscriminate gains?

| Hypothesis | Metric | Experimental contrast | Expected result |
|---|---|---|---|
| **H4.1.** Local context should improve IC-G more than IC-L. | Context-gain ratio; paired bootstrap CI. | `local_graph` versus `logic_only` by IC-L and IC-G. | Stronger gain for IC-G than IC-L. |
| **H4.2.** Local context should not make IC-U cases look solved unless abstention/uncertainty improves. | Repair success; abstention precision; uncertainty calibration; hallucinated provenance rate. | `local_graph` versus `logic_only` on IC-U. | Better uncertainty is a success; unsupported exact guesses are suspect. |
| **H4.3.** Local context may increase prompt length and parse/format failure for smaller local models. | Token count; parse error rate; latency; valid proposal rate. | `logic_only` versus `local_graph` by model size. | Some smaller models may suffer from longer context even when evidence is useful. |

### RQ5 — Does popularity expose memorization or robustness gaps?

**Question.** Do head and tail entities behave differently, and does context reduce that gap?

| Hypothesis | Metric | Experimental contrast | Expected result |
|---|---|---|---|
| **H5.1.** Head entities will be easier in no-retrieval settings, especially for IC-E-elim cases. | Head/mid/tail repair success; exact match; track accuracy. | Popularity buckets under `logic_only` and no retrieval. | Higher head performance suggests parametric-memory contribution. |
| **H5.2.** Local graph context should reduce the head-tail gap for IC-G. | Difference-in-differences: `(head-tail)_logic_only - (head-tail)_local_graph`. | IC-G head/tail cases under both bundles. | Context should make tail local cases more competitive. |
| **H5.3.** A large head-tail gap on IC-E-elim should be interpreted as memorization risk, not grounded repair. | Head-tail gap; provenance quality; hallucinated evidence rate. | IC-E-elim head versus tail under no retrieval. | High head performance without evidence is not enough to claim repair reasoning. |

### RQ6 — How does prompt design affect measured capability?

**Question.** Does representation or few-shot example selection improve repair quality, and where?

| Hypothesis | Metric | Experimental contrast | Expected result |
|---|---|---|---|
| **H6.1.** Hybrid JSON plus concise natural language should outperform pure natural language and pure Turtle on parse validity and executability. | Parse validity; schema validity; executability; tokens per case. | Dev-set representation ablation: hybrid JSON, natural language, Turtle/table if tested. | Hybrid JSON should be the safest default. |
| **H6.2.** Few-shot examples will primarily improve contract compliance rather than truth discovery. | Parse validity; schema validity; operation-shape accuracy; accepted repair. | Zero-shot versus random same-task few-shot versus matched few-shot on dev/core subset. | Format and operation shape should improve more than semantic repair. |
| **H6.3.** Matched few-shot should help T-box constraint-family targeting more than A-box exact repair. | T-box target constraint match; semantic-family success; A-box exact match. | Matched few-shot versus zero-shot by track. | Precedents help structural T-box contracts but should not solve external evidence. |
| **H6.4.** Same-property or same-revision examples risk becoming implicit retrieval and should not be used in the main few-shot condition. | Leakage-risk count; performance jump under same-property examples; manual inspection. | Matched examples excluding same property/revision versus optional same-property precedent experiment. | If same-property examples dominate gains, report them as precedent retrieval, not reasoning floor. |

### RQ7 — What is the oracle-track versus diagnosis-routed gap?

**Question.** How much repair performance is lost when the model must choose the repair locus before proposing a repair?

| Hypothesis | Metric | Experimental contrast | Expected result |
|---|---|---|---|
| **H7.1.** Diagnosis-routed repair will underperform oracle-track repair. | Accepted repair rate; exact match; semantic-family success; track-diagnosis error attribution. | Same model/prompt/cases under `oracle` and `diagnosis_routed`. | The gap quantifies repair-locus selection cost. |
| **H7.2.** The oracle-routed gap will be largest for T-box and ambiguous cases. | Gap by track/subtype; T-box recall; semantic-family success. | Oracle versus diagnosis-routed by A-box/T-box/subtype. | Wrong-locus errors should dominate T-box failures. |
| **H7.3.** Improving track diagnosis will not fully solve proposal quality. | Decomposition: track-correct but proposal-failed cases. | Diagnosis-routed traces partitioned by track correctness. | Locus choice and transaction construction are distinct capabilities. |

### RQ8 — Are H100-runnable local models enough for the main claims?

This is mainly a feasibility and scope question, but it should still be measured.

| Hypothesis | Metric | Experimental contrast | Expected result |
|---|---|---|---|
| **H8.1.** Local open instruction models are sufficient to expose the benchmark’s structured failure modes. | Same qualitative trends across context, track, class, and prompt axes. | Two or more local models on the core set. | The paper can argue from behavioral structure, not frontier scores. |
| **H8.2.** A small API reference subset is sufficient for calibration. | Rank/trend agreement between API subset and local-model trends; confidence intervals. | API model on 500–1,000 stratified cases versus local models on same subset. | API results should contextualize, not replace, the main local experiments. |
| **H8.3.** Absolute leaderboard performance is secondary to decomposition quality. | Completeness of failure analysis; per-stratum metrics. | Paper analysis emphasis. | Scientific value comes from what the benchmark reveals, not the highest model score. |

## Phase A completion checklist

| Task | Status | Deliverable |
|---|---|---|
| A1 — Freeze paper 1 scope | Completed | Scope note, main claim, in-scope/out-of-scope list, supervisor-aligned positioning. |
| A2 — Adopt prior taxonomy vocabulary | Completed | Repository-to-paper taxonomy mapping and terminology policy. |
| A3 — Finalize RQs and hypotheses | Completed | Updated RQ/hypothesis/metric/contrast section. Existing hypotheses retained as scaffold but revised. |

## Immediate consequences for Phase B

1. The classifier redesign should treat old `TypeC/EXTERNAL` as an unsafe overclaim until split into `EXTERNAL_BY_ELIMINATION`, `EXTERNAL_CONFIRMED`, and `UNKNOWN_*` subtypes.
2. Baseline transition matrices are not optional; they are part of construct validation for VQ0.
3. Manual audit should prioritize Type C, delete, and format cases because these are the highest-risk label sources.
4. Core dataset selection should exclude or separately report `IC-U` cases.
5. Prompt experiments should not use same-property or same-revision examples in the main few-shot condition.
