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

### RQ1. Does information necessity predict model behavior?

**Question.** Do models behave differently on rule-implied, local-context, and external-evidence repair cases?

**H1.1.** Rule-implied A-box repairs should be easiest under a logic-only context bundle.

**H1.2.** Local graph context should provide the largest gain on local-context A-box repairs.

**H1.3.** External-evidence cases should remain difficult without retrieval. If performance is high in no-retrieval settings, the result may reflect parametric memory, hidden leakage, or lucky guessing rather than grounded repair.

**H1.4.** If Type A performance improves substantially from local graph context, the Type A definition or classifier may be too broad.

**H1.5.** If Type C performance improves substantially from local graph context, the Type C bucket should be audited for missed local evidence or prompt leakage.

---

### RQ2. Can models choose the correct repair locus?

**Question.** Can a model distinguish between an entity-level repair and a schema-level repair?

**H2.1.** Models will overuse A-box repairs, especially when the input case contains a concrete violating entity and value.

**H2.2.** Diagnosis-routed repair will perform worse than oracle-track repair. The gap measures the cost of repair-locus errors.

**H2.3.** T-box cases will be hardest when the schema reform is generic, weakly causal, or repeated across many apparent violations.

---

### RQ3. Can LLM proposals survive symbolic transaction checks?

**Question.** Are generated repairs merely parseable, or are they executable and historically aligned?

**H3.1.** Valid JSON will overestimate repair success. Many proposals will parse but fail executability, target alignment, exact historical agreement, information preservation, or auditability.

**H3.2.** For T-box reforms, exact signature match will be strict and often low, but semantic-family success will reveal whether the model at least identified the correct reform direction or constraint family.

**H3.3.** Models will sometimes satisfy constraints destructively, for example by deleting a problematic value while losing useful surviving information.

---

### RQ4. How much does prompt design matter?

**Question.** Does prompt format or few-shot selection improve repair quality, and where?

**H4.1.** Hybrid JSON plus concise natural-language explanation should outperform pure natural language and pure Turtle on parse validity and executability.

**H4.2.** Few-shot examples will primarily improve contract compliance, operation shape, and T-box constraint-family targeting. They should not substantially solve external-evidence cases unless they become an implicit retrieval mechanism.

**H4.3.** Matched few-shot examples will help T-box more than A-box because T-box repair contracts and constraint signatures are less familiar to generic instruction models.

---

### RQ5. Does popularity expose memorization or robustness issues?

**Question.** Do head and tail entities behave differently?

**H5.1.** Head entities should be easier in no-retrieval settings because parametric model memory is more likely to contain relevant facts.

**H5.2.** Local graph context should reduce the head-tail gap for genuine local-context repairs.

**H5.3.** A large head-tail gap on external-evidence cases should be interpreted as evidence of memorization risk, not necessarily as repair reasoning.

---

### RQ6. Are local H100-runnable models sufficient for the main scientific claims?

**Question.** Can the main claims be established without a large API budget?

**H6.1.** Local open instruction models should be adequate for measuring structured failure patterns, especially across context, track, and class axes.

**H6.2.** A small API reference subset is sufficient for calibration. The paper does not need a full frontier-model leaderboard.

**H6.3.** The main contribution is the behavioral decomposition and evaluation protocol, not the best absolute score.

---

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
- stronger, after manual audit: **Type C = external evidence required only after audit/retrieval confirmation**;
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

## 12. Dataset tiers for the paper after Phase C

Paper experiments should use deterministic manifests over the full Stage 4 benchmark, not separate benchmark files.

| Tier | Size target | Use | Final-score role |
|---|---:|---|---|
| Full | all valid records | Release and descriptive statistics. | Not the default LLM score because of T-box skew. |
| Core v1 | 4,800 cases | Main reasoning-floor experiments. | Yes, using `main_score_case_ids`; diagnostic cases reported separately. |
| Dev/Pilot v1 | 600 cases | Prompt engineering and representation ablations. | No. |
| Audit v1 | 450 cases | Manual validation of labels and failure modes. | No; used to justify label quality and filtering. |

Core v1 is stratified by repair locus, class/subtype, confidence, popularity bucket, constraint family, and T-box property-revision group. It uses seed-13 deterministic SHA-1 ordering and caps T-box property-revision groups at 10 cases in core.

Core v1 target composition:

| Core group | Target cases | Score policy |
|---|---:|---|
| TypeA clean rule/rejection | 700 | main |
| TypeA ambiguous delete | 250 | diagnostic-only |
| TypeB local graph-grounded | 1,150 | main |
| TypeC / `EXTERNAL_BY_ELIMINATION` | 900 | main stress slice, reported as IC-E-elim rather than confirmed external. |
| T-box directional/schema reform | 1,500 | main, with schema-update separated from directional reforms. |
| T-box coincidental schema change | 300 | diagnostic-only |
| **Total** | **4,800** | mixed |

The main paper score must use `main_score_case_ids`. `DELETE_AMBIGUOUS`, `COINCIDENTAL_SCHEMA_CHANGE`, low-confidence, and `UNKNOWN_*` cases may be run as diagnostics, but they should not be silently mixed into the headline score.

Recommended T-box policy:

| Dataset | T-box cap per property revision |
|---|---:|
| Full | no cap |
| Core v1 | 10 |
| Dev/Pilot v1 | 3 |
| Audit v1 | 5 |

Dev and core must have zero case-id overlap and zero T-box property-revision overlap. This protects few-shot and prompt-development experiments from using examples that are effectively the same schema reform as final-evaluation cases.

---

## 13. Expected results and paper narrative

The strongest paper result is not necessarily high model accuracy. The strongest result is a structured failure map.

A plausible result pattern:

1. Models often produce syntactically valid JSON but fail executability or exact historical agreement.
2. Local graph context improves Type B/IC-G cases more than Type A/IC-L cases.
3. Type C/IC-E-elim cases remain difficult without retrieval or should trigger abstention rather than hallucinated repair.
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
