# WikidataRepairEval: Full Project Description

**Project.** WikidataRepairEval 1.0  
**Purpose.** Benchmarking LLM-assisted knowledge-graph repair using real Wikidata repair events, frozen context, information-access labels, executable proposal contracts, and symbolic evaluation.  
**Current status.** The benchmark substrate, classifier, reasoning-floor runner, proposal contracts, evaluator, deterministic selection manifest, and project documentation exist. The next phase is classifier hardening, core/dev dataset definition, prompt engineering, manual audit, and LLM experiments.

---

## 1. Executive summary

WikidataRepairEval is a benchmark and evaluation framework for studying whether language models can propose trustworthy repairs to knowledge graphs. It is built from historical Wikidata maintenance events: cases where a constraint violation disappeared because either an entity statement changed or the underlying property constraint changed.

The benchmark reconstructs those repairs, attaches a frozen 2026 world-state context, labels cases by repair locus and information requirement, and evaluates model outputs as structured transactions rather than free-text answers. The core scientific claim is that knowledge-graph repair is a multi-dimensional decision problem involving:

- **repair locus**: whether to edit an entity statement or a schema constraint;
- **information condition**: whether the repair is rule-implied, local-context-grounded, or requires non-local evidence;
- **temporal validity**: whether post-repair target values are hidden from model inputs;
- **transaction safety**: whether a proposed edit is executable and does not destroy useful information;
- **auditability**: whether a model provides rationale, provenance, and uncertainty.

The project should not be framed as “another LLM benchmark.” It should be framed as a measurement instrument for evaluating LLM behavior in controlled knowledge-graph repair scenarios.

---

## 2. Conceptual motivation

Knowledge graphs such as Wikidata are structured and auditable, but they are incomplete, noisy, and constantly changing. Language models can generate plausible edits, but they may hallucinate, overfit to surface labels, or repair a constraint violation by deleting useful knowledge. A trustworthy KG repair assistant must do more than output a triple. It must decide what kind of repair is needed, whether the available evidence is sufficient, and whether the proposed edit is safe.

This project targets the gap between:

- **KG completion**, where the goal is usually to predict a missing or plausible triple; and
- **KG repair**, where the goal is to perform a constrained edit that resolves a violation while preserving semantics and respecting modeling practice.

The benchmark therefore evaluates not only *what* value a model predicts, but also *where* it edits, *why* it edits, *what evidence it uses*, and *whether the proposed transaction is executable*.

---

## 3. Relation to prior repair-taxonomy work

A related 2025 ISWC paper, *Formalizing Repairs for Wikidata Constraint Violations: A Taxonomy and Empirical Analysis*, provides a repair taxonomy for Wikidata constraint violations. The project should explicitly adopt this taxonomy and nomenclature rather than compete with it.

The recommended positioning is:

> Prior work formalizes the taxonomy of repairs for Wikidata constraint violations. WikidataRepairEval turns that repair view into an LLM-facing benchmark with frozen context, information-access labels, executable proposal contracts, and evaluation metrics for repair-locus choice, repair content, auditability, and temporal leakage control.

The project therefore has two orthogonal conceptual axes:

| Axis | Meaning | Source |
|---|---|---|
| **Repair locus / repair taxonomy** | Whether the repair edits an entity statement, edits a schema constraint, or is ambiguous. | Prior Wikidata repair-taxonomy work. |
| **Information-access condition** | What kind of information would be required to reproduce or justify the historical repair target under controlled context. | WikidataRepairEval. |

This resolves a potential novelty problem: the project does not claim to invent A-box/T-box repair. It claims to operationalize repair-locus and information-need distinctions for LLM evaluation.

---

## 4. Current benchmark snapshot after Phase B

The Phase B classifier redesign materially changed the benchmark interpretation. The current Stage 4 snapshot contains:

| Slice | Count |
|---|---:|
| Full classified records | 535,570 |
| A-box instance-data repair cases | 78,976 |
| T-box schema-reform cases | 456,594 |
| TypeA cases | 48,085 |
| TypeB local-context cases | 6,059 |
| TypeC / `EXTERNAL_BY_ELIMINATION` cases | 19,430 |
| TypeA / `FORMAT_NORMALIZATION` | 12,807 |
| TypeA / `FORMAT_VALUE_PRUNING` | 4,930 |
| TypeA / `SET_MEMBERSHIP_REJECTION` | 8,971 |
| TypeA / `TARGET_REQUIRED_CLAIM` | 760 |
| TypeA / `DELETE_AMBIGUOUS` | 16,739 |
| TypeB / `LOCAL_TEXT_CONFIRMED` | 88 |
| TypeB / `LOCAL_TEXT_DERIVED` | 4,993 |
| TypeB / `LOCAL_SELECTION_CONFIRMED` | 656 |
| T-box / `RELAXATION_SET_EXPANSION` | 19,118 |
| T-box / `RESTRICTION_SET_CONTRACTION` | 26,602 |
| T-box / `SCHEMA_UPDATE` | 138,799 |
| T-box / `COINCIDENTAL_SCHEMA_CHANGE` | 233,429 |
| T-box / `UNKNOWN_TBOX_CAUSALITY` | 38,646 |

This snapshot has two consequences for the paper:

1. **TypeC is no longer described as unqualified external evidence.** It is `EXTERNAL_BY_ELIMINATION` unless manual audit or retrieval confirms true external evidence need.
2. **TypeA is not mostly logical repair after Phase B.** It is mostly format rejection and ambiguous deletion. Therefore, `DELETE_AMBIGUOUS` must be a diagnostic slice rather than part of the clean TypeA headline score.

The full benchmark remains the release artifact. Paper experiments should use deterministic selection manifests:

| Tier | Target size | Purpose |
|---|---:|---|
| Full | all valid records | Release/statistics. |
| Core v1 | 4,800 | Main LLM experiments, with main-score and diagnostic slices. |
| Dev/Pilot v1 | 600 | Prompt development and debugging. |
| Audit v1 | 450 | Manual label validation. |

The repository writes the Phase C core/dev tiers as manifest JSON files under `reports/benchmark_selection/`, not as additional Stage 4 benchmark files. T-box dominance remains severe: T-box records represent about 85% of the full benchmark. Core selection therefore caps T-box cases per property revision at 10 and marks low-causality `COINCIDENTAL_SCHEMA_CHANGE` cases as diagnostic-only.

---

## 5. Pipeline overview

The full project can be described as a staged pipeline.

### Stage 0. Candidate discovery and de-duplication

The pipeline starts from constraint-violation candidates. Candidates are de-duplicated and merged when multiple report entries refer to the same underlying repair opportunity. The system guards against simple report reorganization by checking whether the same item moved elsewhere rather than being repaired.

### Stage 1/2. Historical repair reconstruction

For each candidate, the system walks Wikidata revision history to identify the edit that caused the violation to disappear.

A repair can be reconstructed as:

- **A-box repair**: the entity statement changed;
- **T-box repair**: the property constraint changed;
- **ambiguous repair**: A-box and T-box evidence overlap or the causal repair is unclear.

For A-box repairs, the system records old value, new value, action type, revision id, editor metadata, and relevant report provenance.

For T-box repairs, the system records the property revision, previous revision, editor metadata, and before/after constraint signature delta.

### Stage 3. Frozen world-state context

Each repair receives a frozen world-state context keyed by case id. The context has four layers:

| Layer | Description |
|---|---|
| **L1 ego node** | Focus entity id, label, description, sitelink count, compact property map, and optional popularity block. |
| **L2 labels** | Id-indexed labels and descriptions for the focus node, target property, neighbors, and constraint-related ids. |
| **L3 neighborhood** | Bounded one-hop outgoing graph context with property ids, target ids, labels, and descriptions. |
| **L4 constraints** | Property constraints, constraint-family ids, qualifiers, and rule summaries. |

T-box cases may also include before/after schema context.

### Stage 3b. Popularity enrichment

Each focus entity receives popularity metadata derived from signals such as pageviews, graph degree, and sitelinks. This allows analysis over head, mid, and tail entities.

Popularity is important because model performance on head entities may reflect parametric memory, whereas tail-entity performance is more diagnostic of context use and reasoning.

### Stage 4. Classification

The deterministic classifier labels each case.

For A-box cases, the current labels are:

- **Type A**: logical/rule-implied repair;
- **Type B**: local-context repair;
- **Type C**: `EXTERNAL_BY_ELIMINATION`, `UNKNOWN_*`, or post-audit `EXTERNAL_CONFIRMED`.

For T-box cases, the classifier assigns schema-change subtypes such as:

- range widened;
- range narrowed;
- allowed set expanded;
- allowed set contracted;
- generic schema update;
- coincidental schema change;
- unknown T-box causality when the reported violation does not map to the changed constraint family or changed qualifier values.

T-box main-score labels require causal constraint-family alignment or type-compatible value/property/language/scope overlap on semantic qualifier changes. Metadata-only qualifier changes, such as constraint status changes, can show that a constraint revision occurred but are not treated as semantic polarity evidence. Directional labels additionally require interpretable polarity for the changed target constraint family; the public directional subtype is coarse, while active `directional_subtype_precise` records allowed, forbidden, required, or exception set semantics only for final directional labels.

### Stage 5. Splits and selection manifests

The project supports deterministic train/dev/test splits and deterministic paper-subset selection. Selection manifests allow experiments to target a frozen subset without creating a second full benchmark artifact.

The current selector keeps all A-box cases and caps T-box cases per property revision. For broader prompt and model experiments, the project should define:

- a full dataset;
- a core dataset;
- a dev/pilot set;
- an audit set.

### Stage 6. Reasoning-floor runs

The reasoning floor is the zero-shot pre-intervention baseline. It evaluates models without retrieval, tools, verifier-guided retries, or memory.

For each case and ablation bundle, the runner performs:

1. track diagnosis: predict `A_BOX`, `T_BOX`, or `AMBIGUOUS`;
2. repair proposal generation, using either historical track (`oracle`) or diagnosed track (`diagnosis_routed`).

The main ablation bundles are:

| Bundle | Contents |
|---|---|
| `minimal_case` | Sanitized case payload only. |
| `logic_only` | Sanitized case plus pruned touched constraints. |
| `local_graph` | Sanitized case plus pruned local graph and constraints. |

The current default uses `logic_only` and `local_graph`, while `minimal_case` remains available.

### Stage 7. Evaluation

The evaluator compares model proposals against benchmark cases. It checks parse validity, executability, exact historical alignment, semantic compatibility, auditability, and track diagnosis.

For A-box repairs, the evaluator reconstructs the pre-repair target-property state, applies proposed operations, and compares the resulting state to the historical repaired state.

For T-box repairs, the evaluator compares the proposed post-reform constraint signature to the historical post-reform signature and also records semantic-family fields.

---

## 6. Temporal validity policy

Temporal leakage is one of the most important methodological concerns.

The frozen world-state snapshot may contain values that were added by the historical repair or changed later. Therefore, the target property cannot simply be shown as it appears in the frozen contemporary graph. The project follows a special **target-property rule**:

> Current graph context may be used for surrounding labels, neighbors, and constraints. The edited target property is the exception: its historical pre-repair state must be reconstructed from repair metadata, not copied from the frozen world-state snapshot.

For local-graph prompts, this means:

- `L1_ego_node.properties[target_pid]` is rewritten to the synthetic pre-repair target state;
- `L3_neighborhood` edges on the target property are omitted or rewritten to avoid exposing post-repair target values;
- missing labels for synthetic pre-repair target ids may be backfilled from Stage 2 mirrors;
- benchmark-only fields such as `repair_target`, `classification`, `persistence_check`, and current target-property values are hidden from the model.

This policy is crucial for credible LLM evaluation.

---

## 7. Repair locus taxonomy

The repair-locus axis follows the existing Wikidata repair-taxonomy tradition.

### A-box repair

An A-box repair edits an entity-level statement. Example actions include:

- deleting an invalid value;
- replacing a value;
- adding a value;
- preserving a valid value while removing an invalid one.

A-box evaluation asks whether the proposed entity edit is executable and whether the resulting target-property state matches the historical repaired state.

### T-box repair

A T-box repair edits the schema or constraint layer. Examples include:

- relaxing a range constraint;
- narrowing a range constraint;
- adding an allowed class/value;
- removing a constraint;
- updating constraint metadata;
- changing property-scope or allowed-entity-type metadata.

T-box evaluation asks whether the proposed constraint signature matches or semantically aligns with the historical post-reform signature.

### Ambiguous repair

Some cases may have both A-box and T-box evidence or weak causal alignment. Ambiguous cases should be explicitly reported rather than forced into a clean class when the historical signal is unclear.

---

## 8. Information-access taxonomy

The A-box information-access labels describe what information would be needed to reproduce or justify the historical repair target.

### Type A: logical / rule-implied

Type A repairs are cases where the rule, violation shape, or internal consistency determines the repair without needing graph traversal or external evidence.

Examples:

- malformed literal corrected according to a format rule;
- singleton one-of constraint where exactly one value is allowed;
- range-boundary correction where the target equals a rule-implied boundary;
- deletion of a value that should not be present.

Phase B status: deletes are no longer automatically treated as clean TypeA rejection. The classifier now distinguishes `REJECTION_RULE_INVALID`, `REJECTION_FORMAT_INVALID`, and `DELETE_AMBIGUOUS`; the ambiguous delete subtype is diagnostic-only in the Phase C core policy.

### Type B: local graph-grounded

Type B repairs are cases where the target truth is independently available in the focus node or immediate graph neighborhood.

Local evidence sources include:

- non-target focus-node properties;
- one-hop neighbor ids;
- focus-node labels or descriptions;
- neighbor labels or descriptions;
- labels for locally referenced ids.

Synthetic pre-repair target-property values remain visible in diagnostics, but retained old target-property values are not independent local support. TypeB remains an audit target through refined subtypes such as `LOCAL_TEXT_CONFIRMED`, `LOCAL_TEXT_DERIVED`, and `LOCAL_SELECTION_CONFIRMED`. Rare focus-QID availability is only TypeB when focus identity itself is sufficient; otherwise it is routed to diagnostic `UNKNOWN_FOCUS_QID_DOMAIN_REASONING`.

### TypeC: `EXTERNAL_BY_ELIMINATION` / IC-U unresolved non-local evidence

After Phase B, ordinary fallback cases are no longer emitted as unqualified `TypeC / EXTERNAL`. The main TypeC subtype is now `EXTERNAL_BY_ELIMINATION`:

```text
not T-box
and not clean rule-deterministic
and target truth not found in supported local buckets
=> TypeC / EXTERNAL_BY_ELIMINATION
```

This label means the supported rule and local-context checks did not identify the historical target. It does **not** prove that external evidence is definitely required. Stronger claims require manual audit or retrieval confirmation.

Phase B TypeC vocabulary:

| Subtype | Meaning | Phase C policy |
|---|---|---|
| `EXTERNAL_BY_ELIMINATION` | Supported rule/local checks did not identify the target. | Main no-retrieval stress slice, reported as IC-E-elim. |
| `UNKNOWN_MISSING_WORLD_STATE` | Required world-state context is absent. | Diagnostic or excluded from main score. |
| `UNKNOWN_MISSING_TRUTH` | Historical target truth tokens are unavailable. | Diagnostic or excluded from main score. |
| `UNKNOWN_CURRENT_VALUE_FALLBACK` | Target would require current-value fallback. | Diagnostic or excluded from main score. |
| `UNKNOWN_INCOMPLETE_LOCAL_CONTEXT` | Local context is too sparse to interpret the label. | Diagnostic or excluded from main score. |
| `UNKNOWN_BAD_TARGET_OR_CONTEXT` | Report shape and repaired target conflict, such as single-value reports with multiple new values. | Diagnostic or excluded from main score. |
| `UNKNOWN_MULTIPLICITY_ARTIFACT` | Multiplicity changed under an unrelated/non-cardinality report. | Diagnostic or excluded from main score. |
| `UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED` | Format pruning removed an invalid-looking value but retained values were not regex-verified. | Diagnostic or excluded from main score. |
| `EXTERNAL_CONFIRMED` | Manual audit or retrieval confirms non-local evidence need. | Future/audit-upgraded label. |

---

## 9. Classifier audit summary after Phase B

The classifier is coherent as a deterministic heuristic, and Phase B hardened the highest-risk branches before final experiments.

### 9.1 Phase B classifier priority order

1. Missing world state -> low-confidence `TypeC / UNKNOWN_MISSING_WORLD_STATE`.
2. T-box repair -> T-box subtype via constraint-delta analysis.
3. A-box delete -> refined delete subtype such as `REJECTION_RULE_INVALID`, `REJECTION_FORMAT_INVALID`, or `DELETE_AMBIGUOUS`.
4. Rule-deterministic match -> clean TypeA subtype only for deterministic cases.
5. Local truth match -> TypeB local subtype with match diagnostics.
6. Fallback -> `TypeC / EXTERNAL_BY_ELIMINATION`.

This priority order is now safer for paper claims, but it still requires manual audit because `EXTERNAL_BY_ELIMINATION` remains a negative-evidence label.

### 9.2 Main classifier risks

| Risk | Consequence | Recommended fix |
|---|---|---|
| TypeC / `EXTERNAL_BY_ELIMINATION` overclaim | Overclaims confirmed external evidence need. | Report as IC-E-elim unless audit/retrieval confirms `EXTERNAL_CONFIRMED`; keep `UNKNOWN_*` as IC-U. |
| 2026/current-value truth fallbacks | Potential post-repair leakage into classification. | Use historical repair target for classification; downgrade fallback cases. |
| Narrow local evidence scan | False TypeC / `EXTERNAL_BY_ELIMINATION` labels. | Expand local buckets to all non-target L1 properties and aligned L2 labels. |
| Literal substring matching | False Type B or false TypeC labels. | Add exact/boundary matching and short-literal safeguards. |
| All deletes are Type A | Hides cases where selecting deletion needs evidence. | Split delete subtypes and downgrade generic deletes. |
| Format constraints over-treated as deterministic | Some format repairs are not unique. | Treat only simple normalizations as high-confidence Type A. |
| Numeric/date range handling may be incomplete | Missed Type A range cases. | Explicitly separate numeric and date bound qualifier properties. |
| Type/value-type T-box qualifiers may be incomplete | T-box subtype errors. | Handle class and relation qualifiers explicitly. |

### 9.3 Recommended classifier output semantics

The classifier should expose enough diagnostics to make labels auditable:

- truth source;
- truth tokens;
- rule families checked;
- local buckets checked;
- match type, e.g. `ID_EXACT`, `LITERAL_EXACT`, `LITERAL_SUBSTRING`, `LABEL_RESOLVED`;
- local graph completeness indicators;
- fallback reason;
- confidence.

### 9.4 Manual classifier audit

Before final experiments, manually audit a stratified sample of approximately 300-500 cases. Include:

| Stratum | Suggested cases |
|---|---:|
| TypeC / `EXTERNAL_BY_ELIMINATION`, QID truth | 50 |
| TypeC / `EXTERNAL_BY_ELIMINATION`, literal truth | 50 |
| TypeC / `UNKNOWN_*` sparse local graph | 50 |
| TypeC / `UNKNOWN_CURRENT_VALUE_FALLBACK` | all or 50 |
| Type A format update | 50 |
| Type A delete under single/unique constraints | 50 |
| Type B local text | 50 |
| T-box generic schema update | 50 |

For each TypeC case, ask:

1. Is the historical target truth actually present in local context?
2. Was it missed because the extractor did not scan the relevant field?
3. Does the case really require external evidence?
4. Is the repair target well-defined?
5. Should this case be core, challenge, or excluded?

---

## 10. Dataset design

The project should maintain multiple dataset tiers.

### 10.1 Full dataset

The full dataset is the canonical historical record. It should remain available for release, descriptive statistics, non-LLM baselines, and future large-scale experiments.

### 10.2 Core dataset

The core dataset should be the main LLM evaluation set. It should be fixed before final experiments and stratified by:

- repair locus;
- information condition;
- subtype;
- constraint family;
- popularity bucket;
- classifier confidence;
- T-box property revision cluster.

Recommended size: 3,000-6,000 cases.

### 10.3 Dev/pilot dataset

The dev/pilot set should be used only for prompt development, representation ablations, and debugging.

Recommended size: 300-800 cases.

### 10.4 Audit dataset

The audit dataset is manually inspected for label quality and failure-mode validation.

Recommended size: 300-500 cases.

### 10.5 Confidence policy

Classifier confidence should not be used as a simple filter. It should be used as a stratification variable.

| Confidence | Recommended treatment |
|---|---|
| High | Include broadly in core. |
| Medium | Include and report separately. |
| Low | Include as diagnostic/challenge slice, not dominant in main score. |
| Missing/weak metadata | Exclude from main score unless explicitly studying artifact weakness. |

---

## 11. Prompt-engineering plan

The prompt-engineering question has three axes:

1. zero-shot vs few-shot;
2. example selection policy;
3. input verbalization format.

### 11.1 Zero-shot baseline

Zero-shot should be the main baseline because it is the cleanest reasoning floor.

The prompt should contain:

- task definition;
- repair-locus definitions;
- allowed operations;
- exact output contract;
- sanitized case context;
- instruction to use only supplied context;
- instruction not to invent external evidence;
- instruction to output only valid JSON;
- short rationale/provenance/uncertainty fields.

### 11.2 Few-shot ablation

Few-shot prompting should be an ablation. It tests precedent adaptation, not the base reasoning floor.

Recommended conditions:

| Condition | Interpretation |
|---|---|
| Zero-shot | Base reasoning floor. |
| Random same-task few-shot | Format learning. |
| Same-track few-shot | Track-specific formatting. |
| Matched few-shot | Precedent-guided repair. |

Matched examples should be selected by:

1. same prediction task;
2. same repair locus;
3. same constraint family;
4. same subtype/action;
5. same information condition;
6. similar local subtype;
7. similar datatype.

Hard exclusions:

- no same case id;
- no same focus entity;
- no same T-box property revision;
- no test/core leakage;
- no benchmark-only labels in examples;
- no historical target of the current instance.

### 11.3 Input representation

The main representation should be hybrid JSON plus concise natural language.

Reasons:

- the evaluator expects structured JSON contracts;
- ids are necessary for exact matching;
- labels/descriptions help semantic interpretation;
- pure Turtle is likely token-expensive and brittle for generation;
- pure natural language loses precision.

Dev-set representation ablations can compare:

- hybrid JSON + controlled natural language;
- pure natural-language verbalization;
- Turtle-like triples;
- compact table/list format.

---

## 12. Reasoning-floor experiment design

The reasoning floor should evaluate models without retrieval, external tools, memory, rejection sampling, or verifier-guided retries.

### Main axes

| Axis | Values |
|---|---|
| Context bundle | `logic_only`, `local_graph`; optionally `minimal_case` for diagnostics. |
| Proposal-track mode | `oracle`, `diagnosis_routed`. |
| Prompt regime | zero-shot contract; few-shot ablation separately. |
| Model | 2-3 local models plus small API reference subset. |
| Dataset | dev/pilot for prompt selection; core for final. |

### Key comparisons

1. `local_graph - logic_only` by information condition.
2. `diagnosis_routed - oracle` by repair locus.
3. T-box exact match vs semantic-family success.
4. Head vs tail entity performance.
5. Zero-shot vs few-shot contract compliance.
6. Valid JSON vs executable proposal vs accepted repair.

---

## 13. Evaluation metrics

### 13.1 General metrics

- proposal presence;
- parse validity;
- normalization success;
- executability;
- acceptance;
- exact historical agreement;
- semantic match where applicable;
- auditability completeness;
- provenance completeness;
- uncertainty availability and calibration proxy;
- tokens per case;
- estimated cost per accepted repair.

### 13.2 A-box metrics

- target entity/property match;
- operation validity;
- post-application state exact match;
- information preservation;
- over-delete rate;
- local constraint-regression pass;
- repair functional success;
- exact historical agreement.

### 13.3 T-box metrics

- target property match;
- target constraint-family match;
- proposed action match;
- exact post-signature match;
- semantic-family match;
- signature overlap;
- whether current values would be admitted by proposed schema;
- generic schema update vs directional reform success.

### 13.4 Track-diagnosis metrics

- exact track match;
- macro-F1;
- confusion matrix;
- A-box false-positive rate on T-box cases;
- T-box false-positive rate on A-box cases;
- ambiguous prediction rate;
- downstream diagnosis-routed repair gap.

---

## 14. Baselines

### 14.1 Symbolic / heuristic baselines

These baselines should be implemented or at least reported where cheap:

- majority track;
- always A-box;
- always T-box;
- constraint-only delete/update heuristic;
- singleton one-of heuristic;
- range-boundary heuristic;
- local-token lookup oracle;
- do-nothing baseline;
- invalid proposal baseline.

They help demonstrate that the benchmark is not only an LLM leaderboard.

### 14.2 LLM baselines

The main experiments should use local H100-runnable models and one API reference subset.

Recommended categories:

- small local instruction model for debugging/lower bound;
- medium local instruction model for main baseline;
- larger local model if feasible;
- one stronger hosted/API model on a stratified subset.

Do not spend limited API credits on the full benchmark. Use API runs to calibrate qualitative patterns.

---

## 15. Failure taxonomy

The analysis should categorize failures, not just report scores.

Candidate failure classes:

| Failure | Description |
|---|---|
| Parse failure | Invalid JSON or non-normalizable output. |
| Wrong track | A-box proposed for T-box or vice versa. |
| Wrong target | Wrong entity, property, or constraint family. |
| Wrong operation | Delete vs update vs add mismatch. |
| Over-delete | Removes useful values to satisfy constraint. |
| Unsupported value | Invents a value not supported by context. |
| Hallucinated provenance | Cites nonexistent or unsupported evidence. |
| Non-auditable | Missing rationale, provenance, or uncertainty. |
| Exact mismatch but semantic match | Directionally plausible but not historically exact. |
| Constraint regression | Fix introduces or preserves supported violations. |
| External-evidence hallucination | Guesses a TypeC / `EXTERNAL_BY_ELIMINATION` or IC-U target without evidence. |

This failure taxonomy is likely to be one of the most scientifically valuable outputs.

---

## 16. Future project direction

The natural continuation is a second paper on neuro-symbolic repair-time control.

### 16.1 Retrieval-augmented repair

Post-audit `EXTERNAL_CONFIRMED` TypeC cases and evidence-heavy T-box cases require retrieval. A future protocol should add:

- query generation;
- evidence retrieval;
- source normalization;
- provenance validation;
- evidence-conditioned repair proposal;
- abstention when evidence is insufficient.

### 16.2 Cost-quality frontier

Because KG repair may be deployed in large-scale maintenance workflows, cost matters. Future experiments should measure:

- tokens per case;
- model cost per accepted repair;
- retrieval calls per accepted repair;
- verifier calls per accepted repair;
- latency;
- batch vs synchronous execution tradeoffs.

### 16.3 Neuro-symbolic Guardian

A Guardian is a repair-time verifier that can reject candidate transactions and return structured diagnostics to an agent that tries again.

Potential comparison:

| System | Retrieval | Verifier | Retry loop |
|---|---:|---:|---:|
| LLM zero-shot | no | no | no |
| LLM + RAG | yes | no | no |
| LLM + verifier | no | yes | yes |
| LLM + RAG + verifier | yes | yes | yes |
| Symbolic heuristic baseline | no | yes | no |

This should be a second paper unless the first paper is repeatedly rejected for lacking a method contribution. A small Guardian-lite experiment can be used as a contingency.

---

## 17. Main risks and mitigations

| Risk | Mitigation |
|---|---|
| “Just another LLM benchmark” | Lead with repair-locus, information condition, transaction evaluation, and temporal controls. |
| Taxonomy novelty challenge | Adopt prior taxonomy explicitly; present information condition as an evaluation axis. |
| TypeC overclaim | Report `EXTERNAL_BY_ELIMINATION` as IC-E-elim, keep `UNKNOWN_*` as IC-U, and manually audit for `EXTERNAL_CONFIRMED`. |
| Historical gold imperfection | Say “historically accepted repair target,” not universal truth. |
| T-box skew | Use core subset caps and property-revision macro-averages. |
| Prompt leakage | Enforce target-property reconstruction and sanitizer tests. |
| API budget limits | Use H100 local models as primary, API subset for calibration. |
| Evaluation partiality | Report exact and semantic metrics; be explicit about supported constraint families. |
| Few-shot leakage | Strict train/dev/test separation and exemplar exclusions. |

---

## 18. Final recommended scope for the first paper

The first paper should include:

1. benchmark construction;
2. taxonomy alignment with existing repair work;
3. information-access labels;
4. classifier audit and label validation;
5. full/core/dev dataset design;
6. zero-shot reasoning floor;
7. logic-only vs local-graph context ablation;
8. oracle-track vs diagnosis-routed repair;
9. local H100 model experiments;
10. small API reference subset;
11. optional few-shot ablation;
12. failure analysis;
13. limitations and future Guardian/RAG roadmap.

The first paper should not try to fully implement RAG, cost-quality frontier analysis, and Guardian-style verifier-guided agents unless needed as a contingency. Those are strong enough for a second paper.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| A-box | Assertional/entity-level data, such as item statements. |
| T-box | Terminological/schema-level data, such as property constraints. |
| Repair locus | The layer where the repair should occur: entity statement or schema constraint. |
| Information condition | The information needed to justify the repair: rule, local graph, or non-local evidence. |
| Reasoning floor | Zero-shot no-tool baseline before RAG, verifier, memory, or retries. |
| Oracle-track | Proposal generation using the historical repair locus. |
| Diagnosis-routed | Proposal generation routed through the model's predicted repair locus. |
| Frozen world state | Contemporary graph context attached to cases for controlled evaluation. |
| Target-property rule | The edited property is reconstructed as pre-repair and not copied from the current snapshot. |
| `EXTERNAL_BY_ELIMINATION` | A TypeC / IC-E-elim case assigned because supported rule/local checks did not find the target. |
| Guardian | Future verifier-guided repair-time controller. |

## Reference anchors for future writing

The final paper should include complete bibliographic entries for:

- Ferranti et al. (2025), **Formalizing Repairs for Wikidata Constraint Violations: A Taxonomy and Empirical Analysis**.
- Wikidata documentation on property constraints and property constraint statement `P2302`.
- Work on formalizing and validating Wikidata constraints with SHACL/SPARQL.
- Knowledge-graph completion and triple-classification benchmarks, used as contrast cases rather than direct competitors.
- LLM structured-output, verification, and neuro-symbolic repair/control literature.
