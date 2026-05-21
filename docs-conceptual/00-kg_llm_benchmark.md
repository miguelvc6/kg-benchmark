# WikidataRepairEval: Research Narrative for the Paper

**Working thesis.** WikidataRepairEval is not primarily a leaderboard for large language models. It is a benchmark and evaluation protocol for studying whether language models can perform **knowledge-graph repair as a controlled edit problem**: decide whether the error lies in an entity statement or in a property constraint, identify what information is needed to justify a repair, and emit an auditable transaction that survives symbolic checks.

The paper should be framed around this claim:

> Knowledge-graph repair is not equivalent to link prediction or question answering. It is a constrained transaction problem over repair locus, evidence availability, temporal validity, schema drift, and auditability. WikidataRepairEval operationalizes this problem using real Wikidata repair events and evaluates how models fail across these dimensions.

This document is intended as the conceptual base for the final paper. It describes the scientific value, research questions, benchmark design, taxonomy, evaluation, model comparisons, and narrative strategy. It deliberately avoids repository-specific schema details except where they are necessary to explain the research contribution.

---

## 1. Brief project description

**WikidataRepairEval 1.0** is a benchmark of real Wikidata repair events with frozen context for evaluating knowledge-graph repair and retrieval needs. The pipeline reconstructs historical fixes, attaches a frozen 2026 world-state snapshot, labels cases by information requirement and repair locus, and supports downstream LLM evaluation through structured repair proposals and track-diagnosis runs.

A benchmark case contains a historical Wikidata maintenance event: a constraint violation disappeared because either an entity statement was repaired (**A-box**) or a property constraint/schema statement was changed (**T-box**). The benchmark records the historical repair target, reconstructs the pre-repair target-property state, attaches controlled local graph context, and evaluates whether a model can propose a repair aligned with the historical outcome.

The benchmark is designed to answer questions such as:

- Can a model tell whether the right repair is to edit an entity fact or to reform the schema rule?
- Does local graph context improve repair quality only when the needed information is actually local?
- Can a model distinguish a rule-implied fix from a case requiring external evidence?
- Are valid-looking JSON repairs actually executable, historically aligned, and auditable?
- Do head entities behave differently from long-tail entities, suggesting memorization or prior exposure?

---

## 2. Main scientific value

The main scientific value is the decomposition of knowledge-graph repair into measurable capabilities.

A weak framing would be:

> We built another benchmark and evaluated LLMs on Wikidata repairs.

A stronger framing is:

> We instantiate an existing Wikidata repair taxonomy into an LLM-facing benchmark that measures repair-locus selection, information sufficiency, temporal leakage control, transaction executability, and auditability.

This provides scientific value beyond a benchmark release because it tests distinct hypotheses about LLM behavior:

1. **Repair locus is a bottleneck.** A model may detect that a violation exists but repair the wrong layer: editing an entity when the rule should change, or changing a rule when the entity value is wrong.
2. **Information need predicts performance.** Rule-implied repairs, locally grounded repairs, and externally grounded repairs should produce different behavior under controlled context ablations.
3. **Syntactic validity is insufficient.** A model can produce valid JSON but still target the wrong entity/property, delete useful information, fail exact historical agreement, or hallucinate provenance.
4. **Temporal control matters.** The frozen contemporary graph can contain post-repair information. The edited target property must therefore be reconstructed as it was before the repair rather than copied from the current snapshot.
5. **Schema repair is not instance repair.** T-box reforms evaluate whether models can reason about property constraints and modeling practice, not only factual triples.

---

## 3. Relation to the 2025 ISWC repair taxonomy

The paper should explicitly build on the 2025 ISWC work *Formalizing Repairs for Wikidata Constraint Violations: A Taxonomy and Empirical Analysis* by Ferranti et al. The supervisor's recommendation to use that taxonomy is methodologically sound.

The contribution should not be presented as a competing taxonomy. Instead, the paper should separate two axes:

| Axis | Source | Meaning in this paper |
|---|---|---|
| **Repair taxonomy / repair locus** | Ferranti et al. and Wikidata repair literature | What kind of repair occurred: A-box entity repair, T-box schema repair, or ambiguous/mixed repair. |
| **Information-access condition** | WikidataRepairEval | What information would be needed for a model to reproduce or justify the historical repair target under controlled context. |

The paper can therefore say:

> We adopt the A-box/T-box repair distinction and related Wikidata repair nomenclature from prior repair-taxonomy work. Our contribution is to operationalize that repair view as an LLM evaluation benchmark with temporally controlled inputs, information-access labels, executable repair contracts, and reasoning-floor experiments.

This avoids the reviewer objection that the A-box/T-box taxonomy is already known.

---

## 4. Research questions and hypotheses

This section uses the hypotheses from the initial narrative as the starting point, but updates them into an operational form. The change is necessary because the classifier audit showed that some Type C cases are not positively confirmed external-evidence cases; they are often **external by elimination** under the current rule/local evidence extractor. The final paper should therefore avoid treating all Type C cases as equally strong evidence of retrieval need.

The paper-facing language should use the following distinction:

- **IC-L**: rule-implied or logical information condition, currently `TypeA` in the repository.
- **IC-G**: local graph-grounded information condition, currently `TypeB` in the repository.
- **IC-E-elim**: external-by-elimination information condition, currently part of `TypeC` but requiring subtype refinement.
- **IC-U**: unknown or insufficient-context cases, also currently mixed into `TypeC` and unsuitable for the main score unless manually audited.
- **T-box repair**: schema-level repair evaluated on a separate repair-locus axis rather than as an A/B/C information condition.

### RQ0. Can the benchmark labels be defended as an information-access instrument?

**Question.** Can the deterministic classifier produce information-condition labels that are sufficiently reliable for model evaluation after splitting unknown cases from external-by-elimination cases?

**H0.1.** Splitting Type C into `EXTERNAL_BY_ELIMINATION` and unknown subtypes will reveal that a non-trivial fraction of current Type C cases should not be interpreted as confirmed external-evidence cases.

- **Metric:** Type C subtype distribution before and after classifier redesign; fraction of Type C routed to `UNKNOWN_MISSING_WORLD_STATE`, `UNKNOWN_MISSING_TRUTH`, `UNKNOWN_CURRENT_VALUE_FALLBACK`, or `UNKNOWN_INCOMPLETE_LOCAL_CONTEXT`.
- **Contrast:** original classifier vs redesigned classifier.

**H0.2.** Expanding local-evidence extraction will convert some current Type C cases into IC-G cases, showing that the previous Type C bucket contained missed local evidence.

- **Metric:** transition matrix from old to new labels, especially `TypeC -> TypeB` / `IC-E-elim -> IC-G` transitions.
- **Contrast:** original local buckets vs expanded local buckets.

**H0.3.** A manually audited core subset will show higher precision for IC-L and IC-G than for IC-E-elim, because IC-E-elim is based on negative evidence rather than direct evidence.

- **Metric:** manual-audit precision by information condition and subtype.
- **Contrast:** IC-L vs IC-G vs IC-E-elim vs IC-U.

---

### RQ1. Does information condition predict model behavior?

**Question.** Do models behave differently on rule-implied, local-context, and external-by-elimination repair cases under controlled context bundles?

**H1.1.** IC-L repairs should have the highest repair success under `logic_only`, because the constraint and violation shape should be sufficient.

- **Metric:** accepted repair rate, exact historical agreement, executable proposal rate, and destructive-repair rate on IC-L.
- **Contrast:** IC-L vs IC-G vs IC-E-elim under `logic_only`.

**H1.2.** IC-G repairs should show the largest gain from `local_graph` over `logic_only`.

- **Metric:** paired difference in accepted repair rate, exact historical agreement, semantic success, and executable proposal rate.
- **Contrast:** `local_graph - logic_only` by information condition.

**H1.3.** IC-E-elim cases should remain difficult without retrieval. High success on IC-E-elim should be interpreted cautiously as possible parametric memory, hidden leakage, lucky guessing, or residual label noise.

- **Metric:** accepted repair rate on IC-E-elim; head-tail split; local-context gain; manual error analysis.
- **Contrast:** IC-E-elim head vs tail entities; IC-E-elim under `logic_only` vs `local_graph`.

**H1.4.** If IC-L improves substantially from `local_graph`, the classifier likely over-includes cases that actually require local evidence.

- **Metric:** IC-L local-context gain and post-hoc audit of high-gain IC-L cases.
- **Contrast:** IC-L cases with large local gain vs IC-L cases solved under `logic_only`.

**H1.5.** If IC-E-elim improves substantially from `local_graph`, the Type C redesign or local-evidence extractor should be audited for missed local evidence or prompt leakage.

- **Metric:** IC-E-elim local-context gain; proportion of successful IC-E-elim cases whose target appears in model-visible local context.
- **Contrast:** successful vs failed IC-E-elim cases under `local_graph`.

---

### RQ2. Can models choose the correct repair locus?

**Question.** Can a model distinguish between an entity-level repair and a schema-level repair?

**H2.1.** Models will overpredict A-box repairs when a concrete violating entity/value is shown, even when the historical repair was T-box.

- **Metric:** track-diagnosis confusion matrix, macro-F1, A-box overprediction rate on T-box cases.
- **Contrast:** track diagnosis on A-box vs T-box cases; `logic_only` vs `local_graph`.

**H2.2.** Diagnosis-routed repair will perform worse than oracle-track repair, and the gap will be largest for T-box cases.

- **Metric:** accepted repair rate, executable proposal rate, exact/semantic success under `oracle` and `diagnosis_routed` modes.
- **Contrast:** `oracle - diagnosis_routed` by repair locus and subtype.

**H2.3.** T-box cases with generic schema updates or weak causal signatures will have lower diagnosis and repair success than directional T-box reforms.

- **Metric:** T-box diagnosis accuracy, semantic-family success, signature similarity, and exact match by T-box subtype.
- **Contrast:** directional reform subtypes vs generic/low-causality schema-update subtypes.

---

### RQ3. Can LLM proposals survive symbolic transaction checks?

**Question.** Are generated repairs merely parseable, or are they executable, historically aligned, and auditable graph transactions?

**H3.1.** Valid JSON will overestimate repair success. Many proposals will parse but fail executability, target alignment, exact historical agreement, information preservation, or auditability.

- **Metric:** funnel from response present -> parse valid -> schema valid -> executable -> exact/semantic success -> auditability pass.
- **Contrast:** syntactic validity vs accepted repair rate across models and context bundles.

**H3.2.** For T-box reforms, exact signature match will be low, but semantic-family success will reveal partial repair understanding.

- **Metric:** exact signature match, target-constraint-family match, reform-direction match, signature Jaccard/overlap, and whether the proposed constraint would admit the historical violating value.
- **Contrast:** exact T-box success vs semantic T-box success.

**H3.3.** Models will sometimes satisfy the violation destructively, for example by deleting values or weakening constraints while losing useful information.

- **Metric:** destructive-repair rate, over-delete rate, information-preservation failures, constraint-regression failures.
- **Contrast:** A-box delete/update operations and T-box relax/restrict operations.

---

### RQ4. How much does prompt design matter?

**Question.** Does prompt format or few-shot selection improve repair quality, and where?

**H4.1.** Hybrid JSON plus concise natural-language descriptions should outperform pure natural language and pure Turtle-like representations on parse validity and executability.

- **Metric:** parse validity, schema validity, executable proposal rate, accepted repair rate, tokens per case.
- **Contrast:** hybrid JSON vs natural-language-only vs Turtle/table variants on the dev set.

**H4.2.** Few-shot examples will primarily improve contract compliance and operation shape, not truth discovery.

- **Metric:** parse validity, schema validity, executable rate, correct operation family, accepted repair rate.
- **Contrast:** zero-shot vs random same-task few-shot vs matched few-shot.

**H4.3.** Matched few-shot examples will help T-box more than A-box because T-box repair contracts and constraint signatures are less familiar to instruction models.

- **Metric:** T-box target-constraint-family match, reform-direction match, signature similarity, exact match.
- **Contrast:** matched few-shot vs zero-shot by repair locus.

**H4.4.** Matched few-shot should not substantially solve IC-E-elim cases unless the examples act as implicit retrieval or memorized precedent.

- **Metric:** IC-E-elim accepted repair rate and provenance plausibility under few-shot prompting.
- **Contrast:** IC-E-elim few-shot gain vs IC-G few-shot gain.

---

### RQ5. Does popularity expose memorization or robustness issues?

**Question.** Do head and tail entities behave differently under no-retrieval repair?

**H5.1.** Head entities should be easier in no-retrieval settings because parametric model memory is more likely to contain relevant facts.

- **Metric:** accepted repair rate, exact historical agreement, and semantic success by popularity bucket.
- **Contrast:** head vs mid vs tail entities under `logic_only` and `local_graph`.

**H5.2.** Local graph context should reduce the head-tail gap for genuine IC-G cases.

- **Metric:** head-tail performance gap on IC-G before and after adding local graph context.
- **Contrast:** `logic_only` head-tail gap vs `local_graph` head-tail gap.

**H5.3.** A large head-tail gap on IC-E-elim should be interpreted as memorization risk, not necessarily as repair reasoning.

- **Metric:** IC-E-elim head-tail gap; success cases with unsupported or hallucinated provenance.
- **Contrast:** IC-E-elim head vs tail cases and IC-E-elim vs IC-G.

---

### RQ6. Are local H100-runnable models sufficient for the main scientific claims?

**Question.** Can the main claims be established without a large API budget?

**H6.1.** Local open instruction models should be sufficient to reveal structured failure patterns across repair locus, information condition, context bundle, and prompt format.

- **Metric:** relative ordering and failure taxonomy across local models.
- **Contrast:** multiple local H100-runnable models on the core set.

**H6.2.** A small API reference subset is sufficient for calibration; the paper does not need a full frontier-model leaderboard.

- **Metric:** same evaluation funnel as local models on a stratified reference subset.
- **Contrast:** best local model vs API reference model on matched core subset.

**H6.3.** The main result should be behavioral decomposition, not absolute leaderboard performance.

- **Metric:** consistency of ablation effects and failure modes across models.
- **Contrast:** model-level score differences vs class/context/locus effects.

---

### RQ7. Can models recognize insufficient evidence?

**Question.** If an abstention option is added, can models avoid hallucinating repairs when the supplied context is insufficient?

This RQ is conditional. It should be included in the main paper only if the output contract is extended to support abstention before the main experiments.

**H7.1.** IC-E-elim and IC-U cases should produce more justified abstentions than IC-L and IC-G cases.

- **Metric:** abstention rate, justified-abstention precision, hallucinated-repair rate.
- **Contrast:** IC-E-elim/IC-U vs IC-L/IC-G.

**H7.2.** Models without abstention will overproduce unsupported repairs on IC-E-elim and IC-U cases.

- **Metric:** unsupported repair rate, hallucinated provenance rate, failed auditability rate.
- **Contrast:** forced-repair contract vs abstention-enabled contract on the same dev/core subset.

If abstention is not implemented, RQ7 should be moved to limitations and future work rather than reported as a main result.

## 5. Benchmark construction narrative

The benchmark construction story should be told as a sequence of controlled transformations.

### 5.1 Historical repair reconstruction

The pipeline starts from Wikidata constraint-violation reports and reconstructs the edit that caused the violation to disappear. The repair may be:

- an **A-box repair**, where the entity statement changed;
- a **T-box repair**, where the property constraint changed;
- an ambiguous case where evidence overlaps or the repair locus is not clean.

For each candidate, the system records the focus entity, property, violation type, historical repair action, old value, new value where available, editor metadata, revision information, and relevant provenance.

### 5.2 Frozen world-state context

Each case receives a frozen world-state context. Conceptually, this context has four layers:

| Layer | Meaning |
|---|---|
| **L1 ego node** | Focus entity identity, label, description, compact property map, and related metadata. |
| **L2 labels** | Deterministic label and description map for referenced ids, properties, neighbors, and constraint-related entities. |
| **L3 neighborhood** | Bounded one-hop outgoing graph context. |
| **L4 constraints** | Property-constraint metadata, including constraint-family ids, qualifiers, and rule summaries. |

The frozen context is contemporary, but the edited target property is special: it must be reconstructed from historical repair metadata as the **pre-repair** target-property state. This avoids leaking the post-repair target value into the model input.

### 5.3 Persistence check

The benchmark does not blindly treat every historical edit as a usable evaluation case. It checks whether the repair target remains meaningful in the frozen contemporary world state. Cases that no longer express the same logical evaluation question should be filtered, flagged, or excluded from core experiments.

### 5.4 Information-access classification

A deterministic classifier assigns A-box cases to information-access classes. T-box cases are classified separately as schema reforms.

The current classifier is useful as a deterministic first pass, but its Type C class requires careful interpretation. In the current design, many Type C cases are effectively **external by elimination**: the target truth was not found by supported rule checks or supported local-context extraction. This does not prove that external evidence is definitely required. The paper should either rename this subtype or report it with manual-audit validation.

### 5.5 Reproducible subsets

The full benchmark is the complete historical record. Because the full dataset is large and skewed, especially by repeated manifestations of the same T-box schema update, the paper should also define a fixed **core dataset** and a smaller **development set**.

Recommended tiers:

| Dataset tier | Purpose |
|---|---|
| **Full dataset** | Canonical release and descriptive statistics. |
| **Core dataset** | Main LLM evaluation across axes. |
| **Dev/Pilot set** | Prompt engineering, representation ablations, debugging, and initial failure analysis. |
| **Audit set** | Manual validation of classifier labels, especially Type C and T-box cases. |

---

## 6. Taxonomy and labels

The paper should distinguish **repair locus** from **information condition**.

### 6.1 Repair locus

| Label | Meaning | Main evaluation question |
|---|---|---|
| **A-box repair** | An entity statement should be edited. | Can the model edit the correct entity/property/value without losing useful information? |
| **T-box repair** | A property constraint/schema statement should be edited. | Can the model reform the correct constraint family and signature? |
| **Ambiguous** | Evidence supports more than one locus or the historical route is not uniquely determined. | Can the model identify uncertainty or avoid overconfident repair? |

This axis should follow the nomenclature of the existing Wikidata repair-taxonomy work.

### 6.2 Information condition for A-box repairs

The current Type A/B/C naming can remain in code, but the paper should consider using neutral names such as `IC-L`, `IC-G`, and `IC-E` to avoid creating a competing taxonomy.

| Current label | Paper-facing label | Meaning | Expected model behavior |
|---|---|---|---|
| **Type A** | **IC-L: logical / rule-implied** | The rule, violation shape, or internal consistency is sufficient. | Should be solvable with logic-only constraints. |
| **Type B** | **IC-G: local graph-grounded** | The repair target is available in focus-node or one-hop local context. | Should benefit most from local graph context. |
| **Type C** | **IC-E: external / non-local evidence** or **IC-U: unresolved external-by-elimination** | The supported local/rule evidence does not identify the target. | Should be hard without retrieval; may require abstention. |

A crucial paper decision: Type C should not be overclaimed. Use one of these formulations:

- conservative: **Type C = not supported by local/rule evidence under the current extraction protocol**;
- stronger, after manual audit: **Type C = external evidence required**;
- best: split into `EXTERNAL_CONFIRMED`, `EXTERNAL_BY_ELIMINATION`, and `UNKNOWN_*` subtypes.

### 6.3 T-box schema-reform subtypes

T-box cases should be reported separately from A/B/C. Candidate subtypes include:

| Subtype | Interpretation |
|---|---|
| `RELAXATION_RANGE_WIDENED` | Numeric/date range became more permissive. |
| `RESTRICTION_RANGE_NARROWED` | Numeric/date range became stricter. |
| `RELAXATION_SET_EXPANSION` | Allowed set or class set expanded. |
| `RESTRICTION_SET_CONTRACTION` | Allowed set or class set contracted. |
| `SCHEMA_UPDATE` | Schema changed but direction is generic or not confidently typed. |
| `COINCIDENTAL_SCHEMA_CHANGE` | Schema changed, but causal relation to the violation is weak. |

---

## 7. Prediction tasks

The benchmark supports two main tasks and one composed setting.

### 7.1 Track diagnosis

The model predicts whether the repair locus is:

```json
{"predicted_track": "A_BOX"}
```

or

```json
{"predicted_track": "T_BOX"}
```

or

```json
{"predicted_track": "AMBIGUOUS"}
```

This measures repair-locus selection independently from repair generation.

### 7.2 Oracle-track repair proposal

The model receives the correct historical track and proposes a structured repair in the corresponding contract:

- A-box proposal: edit entity values.
- T-box proposal: reform property constraint signature.

This measures repair ability when locus selection is not the bottleneck.

### 7.3 Diagnosis-routed repair proposal

The model first predicts the track and then generates the proposal using the predicted track. This measures the full end-to-end repair behavior.

The gap between oracle-track and diagnosis-routed performance is central: it quantifies how much repair quality is lost to locus-selection errors.

---

## 8. Context ablations

The reasoning-floor experiments should use fixed context bundles.

| Bundle | Contents | Purpose |
|---|---|---|
| `minimal_case` | Sanitized case-local payload only. | Tests the no-context lower bound. |
| `logic_only` | Sanitized case payload plus pruned touched constraints. | Tests rule and constraint reasoning. |
| `local_graph` | Sanitized case payload plus pruned L1-L4 context. | Tests local graph grounding. |

The main paper should probably use `logic_only` and `local_graph` as primary bundles. `minimal_case` can be included in pilot or diagnostic analysis.

All bundles must exclude benchmark-only fields such as the classification label, historical repair target, persistence check, popularity, and current/post-repair target-property truth fields. For `local_graph`, the target property must be reconstructed into its pre-repair state and target-property edges in the current neighborhood must be suppressed or carefully rewritten.

---

## 9. Prompting strategy

### 9.1 Main baseline: zero-shot contract prompting

The main reasoning-floor baseline should be **zero-shot, schema-constrained prompting**:

- explicit task description;
- explicit allowed operations;
- explicit output schema;
- ids plus labels;
- short rationale and provenance fields;
- no benchmark-derived demonstrations;
- deterministic decoding.

This is the cleanest baseline because it tests task understanding and context use, not adaptation from selected examples.

### 9.2 Few-shot prompting as an ablation

Few-shot prompting should be an ablation, not the main result. It measures a different capability:

> Can models adapt from repair precedents?

Recommended few-shot policies:

| Policy | Description | Main interpretation |
|---|---|---|
| Zero-shot | No examples. | Reasoning floor. |
| Random same-task examples | Same schema, unrelated cases. | Format learning. |
| Same-track examples | A-box examples for A-box, T-box for T-box. | Repair-locus-specific formatting. |
| Matched examples | Same track, constraint family, subtype, and information condition, with strict exclusions. | Precedent-guided repair. |

Strict exclusions for few-shot examples:

- no same case id;
- no same focus entity;
- no same T-box property revision;
- preferably no same property in the main matched setting;
- no examples from the test/core evaluation slice;
- no benchmark-only fields visible in examples;
- no current target-property value leakage.

### 9.3 Representation choice

The recommended main representation is **hybrid JSON plus controlled natural language**.

Rationale:

- JSON aligns with the evaluator contracts.
- Natural language helps the model interpret labels, constraints, and actions.
- Turtle/RDF is attractive for a Semantic Web audience but may be token-expensive and brittle for structured generation.

A small dev-set ablation can compare:

| Representation | Expected behavior |
|---|---|
| Hybrid JSON + concise natural language | Best overall for parse validity and grounding. |
| Pure natural language | Potentially better for small models, weaker exactness. |
| Turtle-like triples | Semantically familiar for KG venues, likely more token-heavy. |
| Compact table/list | Possibly strong for local graph cases. |

---

## 10. Evaluation protocol

Evaluation should never collapse to a single leaderboard score. The paper should report multiple metric families.

### 10.1 Shared metrics

| Metric | Meaning |
|---|---|
| Parse validity | Did the model produce valid normalized JSON? |
| Proposal presence | Did it produce a proposal at all? |
| Executability | Can the proposal be applied to the intended target? |
| Acceptance | Did it pass the benchmark's strict success definition? |
| Auditability completeness | Does it provide rationale, provenance, and uncertainty? |
| Token/cost telemetry | How expensive is a case, prompt regime, or accepted repair? |

### 10.2 A-box metrics

A-box evaluation should reconstruct the pre-repair target-property state, apply the proposed operations in memory, and compare the result to the historical repaired state.

Important metrics:

- exact historical agreement;
- functional success;
- information preservation;
- supported local constraint regression;
- over-delete rate;
- wrong target entity/property rate;
- provenance completeness;
- auditability completeness.

### 10.3 T-box metrics

T-box evaluation should compare the proposed post-reform constraint signature to the historical post-reform signature.

Important metrics:

- exact action match;
- exact signature match;
- target constraint-family match;
- semantic-family success;
- signature overlap / Jaccard;
- whether the proposed schema would admit the relevant current values;
- provenance completeness;
- auditability completeness.

For T-box, exact match should remain strict, but semantic-family metrics are necessary because a proposed reform may be directionally correct even if it does not exactly reproduce the historical signature.

### 10.4 Track-diagnosis metrics

Important metrics:

- exact track accuracy;
- macro-F1 over A-box/T-box/ambiguous;
- confusion matrix;
- false A-box rate on T-box cases;
- false T-box rate on A-box cases;
- oracle-track vs diagnosis-routed repair gap.

### 10.5 Aggregation

Report metrics by:

- repair locus;
- information condition;
- subtype;
- constraint family;
- context bundle;
- prompt regime;
- model;
- popularity bucket;
- T-box property revision cluster.

T-box metrics should be reported both case-level and cluster-macro-averaged by property revision. Otherwise, repeated manifestations of one schema edit can dominate aggregate scores.

---

## 11. Baselines and models compared

### 11.1 Non-LLM baselines

These are scientifically important because they establish how much can be solved without LLM reasoning.

| Baseline | Purpose |
|---|---|
| Majority track | Lower bound for repair-locus diagnosis. |
| Always A-box | Tests how much score comes from entity-edit bias. |
| Always T-box | Tests schema-reform bias. |
| Constraint-only delete/update heuristic | Estimates rule-deterministic repair coverage. |
| Local-token lookup oracle | Checks whether Type B labels are operational. |
| Do-nothing / invalid proposal | Sanity lower bound. |

### 11.2 Local LLMs on H100

The main evaluation should rely on local open instruction models, because API credits are limited and the scientific claims do not require a full frontier-model leaderboard.

Recommended comparison categories:

| Model category | Role |
|---|---|
| Small local instruction model | Cheap lower bound and prompt-debugging model. |
| Medium H100-runnable instruction model | Main open-model baseline. |
| Larger local model if feasible | Tests whether scale improves repair locus and structured proposal quality. |

The exact models should be chosen near execution time based on availability, context length, inference speed, JSON reliability, and H100 memory constraints. The paper should report model names, parameter sizes, quantization if any, serving stack, context length, decoding settings, and hardware.

### 11.3 API reference subset

Use API credits only for a stratified reference subset, not for the full benchmark.

| API setting | Role |
|---|---|
| One strong commercial or hosted model | Calibration reference on 500-1,000 stratified cases. |
| Optional cheaper API model | Secondary reference if budget allows. |

The API subset should be stratified by class, track, subtype, popularity bucket, and T-box revision cluster.

---

## 12. Dataset tiers for the paper

A recommended design:

| Tier | Size target | Use |
|---|---:|---|
| Full | all valid cases | Release and descriptive statistics. |
| Core | 3,000-6,000 cases | Main LLM experiments. |
| Dev/Pilot | 300-800 cases | Prompt engineering and representation ablations. |
| Audit | 300-500 cases | Manual validation of labels and failure modes. |

The core should not be selected only by classifier confidence. Confidence should be a stratification variable. Low-confidence cases should be included as a diagnostic slice or challenge set, but not silently mixed into the main score.

Recommended T-box policy:

| Dataset | T-box cap per property revision |
|---|---:|
| Full | no cap |
| Extended paper subset | 50-100 |
| Core | 5-20 |
| Dev/Pilot | smaller fixed stratified sample |

---

## 13. Expected results and paper narrative

The strongest paper result is not necessarily high model accuracy. The strongest result is a structured failure map.

A plausible result pattern:

1. Models often produce syntactically valid JSON but fail executability or exact historical agreement.
2. Local graph context improves Type B/IC-G cases more than Type A/IC-L cases.
3. Type C/IC-E cases remain difficult without retrieval or should trigger abstention.
4. T-box repair is harder than A-box repair because models confuse entity values with constraint-family ids or reform the wrong layer.
5. Diagnosis-routed repair is substantially worse than oracle-track repair, showing that repair-locus selection is a real bottleneck.
6. Few-shot prompting improves contract compliance and T-box shape but does not solve external evidence.
7. Head entities perform better than tail entities in no-retrieval settings, indicating possible parametric memory effects.

The paper's discussion should emphasize that these patterns support the benchmark's scientific value even if absolute repair accuracy is low.

---

## 14. Limitations to state explicitly

1. **Historical repair is not universal ground truth.** The benchmark evaluates alignment with a historically accepted repair target, not metaphysical correctness.
2. **Type C is not automatically retrieval-confirmed.** Without manual audit or retrieval evidence, many Type C cases should be treated as external-by-elimination.
3. **Evaluation is partial.** The evaluator checks supported constraints and exact/semantic alignment but cannot reproduce all Wikidata community judgment.
4. **Temporal reconstruction is difficult.** Some entities, properties, and constraints change independently after the repair event.
5. **T-box skew is real.** Many apparent cases can originate from a single schema reform.
6. **Few-shot prompting can become implicit retrieval.** Matched examples must be carefully separated from the test set.
7. **No-retrieval Type C results are hard to interpret.** Success may reflect memory, leakage, or guessing rather than grounded evidence.

---

## 15. Claims to make and claims to avoid

### Safe claims

- WikidataRepairEval operationalizes real Wikidata repair events for LLM evaluation.
- The benchmark separates repair-locus diagnosis from repair proposal quality.
- The benchmark supports controlled context ablations and temporal leakage safeguards.
- The benchmark evaluates executable, auditable graph-repair transactions rather than only free-text answers.
- The information-condition labels predict where local context should or should not help.

### Claims to avoid

- Do not claim the A-box/T-box taxonomy is novel.
- Do not claim historical edits are perfect ground truth.
- Do not claim Type C fully evaluates retrieval unless retrieval is actually implemented.
- Do not claim the evaluator fully validates Wikidata correctness.
- Do not report one aggregate score without stratified breakdowns.

---

## 16. Recommended paper structure

1. **Introduction**
   - Motivation: KG repair is a constrained transaction problem, not just link prediction.
   - Main contribution: temporally controlled LLM-facing repair benchmark.
   - Summary of findings.

2. **Background and related work**
   - Wikidata constraints.
   - Existing Wikidata repair taxonomy.
   - KG completion vs KG repair.
   - LLMs for structured editing and KG maintenance.

3. **Benchmark construction**
   - Historical repair reconstruction.
   - Frozen world-state context.
   - Temporal target-property reconstruction.
   - Persistence checks.
   - Dataset tiers.

4. **Repair and information taxonomy**
   - A-box/T-box repair locus from prior taxonomy.
   - Information-access condition: logical, local, external-by-elimination/confirmed external.
   - T-box subtypes.

5. **Tasks and prompts**
   - Track diagnosis.
   - Oracle-track repair proposal.
   - Diagnosis-routed repair proposal.
   - Context bundles.
   - Prompt contract and representation.

6. **Evaluation**
   - A-box proposal evaluation.
   - T-box proposal evaluation.
   - Track-diagnosis evaluation.
   - Auditability and cost telemetry.

7. **Experiments**
   - Dataset splits and core selection.
   - Non-LLM baselines.
   - Local models.
   - API reference subset.
   - Zero-shot and few-shot ablations.

8. **Results**
   - Stratified performance.
   - Context gains.
   - Oracle vs diagnosis-routed gap.
   - T-box semantic-family results.
   - Popularity analysis.
   - Failure taxonomy.

9. **Discussion**
   - What the failures imply about LLM-assisted KG repair.
   - Why retrieval and verifier-guided protocols are needed.
   - How the benchmark can support future Guardian-style systems.

10. **Limitations and ethics**
    - Historical gold limitations.
    - External evidence limitations.
    - Temporal leakage risks.
    - Dataset skew.

11. **Conclusion**
    - The benchmark as a measurement instrument for trustworthy KG repair.

---

## 17. One-paragraph abstract draft

Knowledge-graph repair requires more than predicting missing triples: a system must decide whether an error lies in an entity statement or a schema constraint, determine what information is needed to justify the edit, and produce an auditable transaction that can be checked symbolically. We introduce WikidataRepairEval, a benchmark of historical Wikidata repair events with frozen world-state context and temporal leakage controls. Building on existing Wikidata repair taxonomy, the benchmark separates A-box entity repairs from T-box schema reforms and labels A-box cases by information condition: rule-implied, locally grounded, or requiring non-local evidence. We evaluate language models on track diagnosis, oracle-track repair, and diagnosis-routed repair under logic-only and local-graph context ablations. Our evaluation checks parse validity, executability, exact historical agreement, semantic schema-family match, auditability, and cost telemetry. The resulting failure map shows whether models repair the right layer, use local context when appropriate, abstain or guess when evidence is insufficient, and produce transactions that survive symbolic verification.

---

## 18. One-sentence positioning

> WikidataRepairEval measures whether language models can perform temporally controlled, evidence-aware, executable knowledge-graph repair rather than merely generate plausible graph edits.

## Reference anchors to include in the final paper

These are not complete BibTeX entries, but they identify the sources the final paper should cite.

- Ferranti et al. (2025), **Formalizing Repairs for Wikidata Constraint Violations: A Taxonomy and Empirical Analysis**. Use this as the repair-taxonomy and A-box/T-box foundation.
- Wikidata documentation, **Help:Property constraints portal** and **Property:P2302**. Use these to describe Wikidata property constraints and their role in violation detection.
- Prior work on Wikidata constraint formalization and validation, especially SHACL/SPARQL-based treatments.
- Knowledge-graph completion benchmarks such as CoDEx, to contrast link prediction/triple classification with executable repair transactions.
- LLM structured-output and tool/verifier literature, to position the proposal contracts and future Guardian protocol.
