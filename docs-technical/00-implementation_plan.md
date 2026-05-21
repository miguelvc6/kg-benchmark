# WikidataRepairEval Implementation and Execution Plan

**Ordered task plan for repository implementation, evaluation, prompt development, manual audit, experiments, analysis, and paper writing**

## 0. Guiding decisions

The project should proceed with the following decisions fixed unless strong evidence emerges during the pilot:

1. Use the existing Wikidata repair taxonomy and nomenclature as the repair-locus backbone.
2. Treat Type A/B/C as an information-condition layer, not as a competing repair taxonomy.
3. Keep zero-shot contract prompting as the main paper baseline.
4. Treat few-shot prompting as an ablation.
5. Use hybrid JSON + controlled natural language as the main prompt representation unless dev results strongly disprove it.
6. Create full, core, and dev dataset tiers.
7. Redesign Type C to distinguish external-by-elimination from unknown or confirmed external cases.
8. Keep RAG, full cost-quality frontier, and neuro-symbolic Guardian as paper 2 unless a small Guardian-lite contingency is needed.

## 1. Phase A — Project alignment and paper scope

**Completion status:** complete on 2026-05-21  
**Deliverable:** `00-phase_A_completion.md`  
**Narrative update:** Section 4 of `00-kg_llm_benchmark.md` has been updated with the finalized operational RQs/hypotheses.

### Phase A decision summary

Task A3 should **not** reuse the original hypotheses from `00-kg_llm_benchmark.md` verbatim. The original hypotheses are a good conceptual base, but they have now been updated into metric-linked, contrast-linked hypotheses because the classifier audit changed the interpretation of Type C. Type C should be treated as external-by-elimination or unknown/incomplete-context unless it is manually or retrieval-confirmed.

### Task A1 — Freeze paper 1 scope

**Type:** Research planning / writing  
**Dependencies:** none  
**Output:** short internal scope note  
**Status:** complete

Paper 1 is frozen as:

> Benchmark + taxonomy operationalization + classifier validation + reasoning floor + prompt/context/model analysis.

Full RAG, full cost-quality frontier analysis, and the neuro-symbolic Guardian agent loop are excluded from paper 1 except as future work or contingency.

**Acceptance criteria:** complete

- One-paragraph main claim exists in `00-phase_A_completion.md`.
- List of in-scope and out-of-scope contributions exists.
- Supervisor-aligned taxonomy positioning is documented.

### Task A2 — Adopt prior taxonomy vocabulary

**Type:** Research planning / writing  
**Dependencies:** A1  
**Output:** taxonomy mapping note  
**Status:** complete

The paper-facing vocabulary separates repair locus from information condition:

| Repository term | Paper-facing term | Notes |
|---|---|---|
| A_BOX | A-box / instance-level repair | Use prior taxonomy nomenclature. |
| T_BOX | T-box / schema-level repair | Use prior taxonomy nomenclature. |
| TypeA | IC-L / rule-implied information condition | Keep TypeA in code if needed. |
| TypeB | IC-G / local graph-grounded information condition | Keep TypeB in code if needed. |
| TypeC | IC-E-elim or IC-U | Split/qualify; do not overclaim confirmed externality. |

**Acceptance criteria:** complete

- No paper section claims A-box/T-box taxonomy as novel.
- Type A/B/C is described as information access, not repair taxonomy.
- Type C is explicitly split into external-by-elimination and unknown/incomplete-context cases before core experiments.

### Task A3 — Finalize research questions and hypotheses

**Type:** Research planning / writing  
**Dependencies:** A1, A2  
**Output:** RQ/Hypothesis document section  
**Status:** complete

Decision: use the hypotheses in `00-kg_llm_benchmark.md` as the starting point, but update them. The finalized version adds metrics, experimental contrasts, and conservative Type C wording.

Finalized research-question set:

1. Can the benchmark labels be defended as an information-access instrument?
2. Does information condition predict model behavior?
3. Can models choose the correct repair locus?
4. Can LLM proposals survive symbolic transaction checks?
5. How much does prompt design matter?
6. Does popularity expose memorization or robustness issues?
7. Are local H100-runnable models sufficient for the main scientific claims?
8. Can models recognize insufficient evidence, if abstention is implemented?

**Acceptance criteria:** complete

- Each RQ has at least one falsifiable hypothesis.
- Each hypothesis has corresponding metrics and experimental contrast.
- Type C is phrased as external-by-elimination or unresolved unless confirmed by audit/retrieval.
- The updated RQs are stored in `00-phase_A_completion.md` and `00-kg_llm_benchmark.md`.


### Phase A completion output — completed 2026-05-21

Phase A is completed in [`00-phase_a_project_alignment.md`](00-phase_a_project_alignment.md).

**A1 result.** Paper 1 is scoped as benchmark construction, taxonomy operationalization, classifier validation, reasoning-floor evaluation, and prompt/context/model analysis. Full RAG, full Guardian, live editing, and cost-quality frontier analysis are out of scope for the first paper except as future work or contingency.

**A2 result.** A-box/T-box terminology is adopted as the repair-locus vocabulary from existing Wikidata repair-taxonomy work. Type A/B/C is reframed as an information-condition layer: IC-L, IC-G, and IC-E/IC-U. Type C must be split or qualified rather than treated as confirmed external evidence.

**A3 result.** The hypotheses from `00-kg_llm_benchmark.md` were updated rather than copied unchanged. The revised set adds explicit metrics and experimental contrasts, separates prompt-design hypotheses from benchmark-behavior hypotheses, and accounts for `EXTERNAL_BY_ELIMINATION` and `UNKNOWN_*` Type C subtypes.

## 2. Phase B — Repository audit and classifier redesign

### Task B1 — Create classifier audit branch and snapshot current counts

**Type:** Repository implementation / evaluation  
**Dependencies:** A2  
**Output:** baseline classifier count report

Run the current classifier and save:

- counts by class;
- counts by subtype;
- counts by confidence;
- counts by truth source;
- counts by decision-trace branch;
- counts by local subtype;
- counts by repair track.

**Metrics:**

- number of TypeC/EXTERNAL cases;
- number of low-confidence TypeC cases;
- number of TypeC cases whose truth source is current 2026 fallback;
- number of missing world-state fallbacks;
- number of TypeA deletes;
- number of TypeA format repairs.

**Acceptance criteria:**

- Baseline stats are committed or archived.
- Baseline can be compared to redesigned classifier through a transition matrix.

### Task B2 — Split Type C subtypes

**Type:** Repository implementation  
**Dependencies:** B1  
**Output:** updated classifier and schema/docs

Replace ordinary TypeC/EXTERNAL fallback with more precise subtypes:

| New subtype | Trigger |
|---|---|
| EXTERNAL_BY_ELIMINATION | Historical truth exists, local/rule checks fail, context is sufficient enough for negative evidence. |
| UNKNOWN_MISSING_WORLD_STATE | No usable world-state context. |
| UNKNOWN_MISSING_TRUTH | No usable historical truth tokens. |
| UNKNOWN_CURRENT_VALUE_FALLBACK | Truth was taken from 2026/current-value fallback. |
| UNKNOWN_UNSUPPORTED_VALUE_SHAPE | Truth exists but cannot be compared by current logic. |
| UNKNOWN_INCOMPLETE_LOCAL_CONTEXT | Local context is too sparse to claim externality. |
| EXTERNAL_CONFIRMED | Reserved for manual audit or retrieval-confirmed cases. |

**Metrics:**

- TypeC subtype distribution;
- fraction of TypeC that is external-by-elimination vs unknown;
- fraction of TypeC eligible for main core scoring.

**Acceptance criteria:**

- No missing world-state case is labeled ordinary `EXTERNAL`.
- No missing-truth case is labeled ordinary `EXTERNAL`.
- Type C rationale clearly distinguishes negative evidence from confirmed externality.

### Task B3 — Remove or quarantine 2026 truth fallbacks

**Type:** Repository implementation  
**Dependencies:** B2  
**Output:** updated truth extraction function

Classification truth sources should prioritize historical repair target fields:

1. `repair_target.new_value`;
2. `repair_target.value`.

If 2026 fields are retained, route them to low-confidence diagnostics:

- `persistence_check.current_value_2026`;
- `violation_context.value_current_2026`.

**Metrics:**

- number of cases losing truth tokens after removing 2026 fallback;
- number routed to `UNKNOWN_CURRENT_VALUE_FALLBACK`;
- downstream class changes.

**Acceptance criteria:**

- Main classification no longer silently uses 2026 current values as ordinary historical truth.
- Cases needing 2026 fallback are excluded from core or separately reported.

### Task B4 — Expand local evidence extraction

**Type:** Repository implementation  
**Dependencies:** B2  
**Output:** updated local context buckets

Add local-evidence sources beyond the current buckets:

- all non-target L1 properties;
- labels/descriptions for QIDs referenced in local properties;
- L2 labels for locally referenced ids;
- optionally aliases and qualifiers if supported later.

Preserve leakage control:

- do not use current/post-repair values on the target property;
- do not expose target-property L3 edges that leak post-repair values.

**Metrics:**

- TypeC -> TypeB transitions;
- new TypeB local subtype distribution;
- false-positive local matches in manual audit.

**Acceptance criteria:**

- Non-target local property values can produce TypeB when they exactly contain the historical target.
- Target-property current values still cannot produce a local match.

### Task B5 — Tighten local literal matching

**Type:** Repository implementation  
**Dependencies:** B4  
**Output:** updated matcher and tests

Implement stricter matching:

| Value type | Rule |
|---|---|
| QID/PID | Exact id match only, unless label resolution is explicitly proven. |
| Full ISO date | Exact date match at sufficient precision. |
| Short literal < 4 chars | Exact field equality, not substring. |
| Longer literal | Token-boundary match or exact normalized field match. |
| Label-resolved QID | Optional, but record resolution source and confidence. |

**Metrics:**

- TypeB -> TypeC/UNKNOWN transitions due to stricter matching;
- literal substring match rate;
- audit false-positive rate for `LOCAL_TEXT`.

**Acceptance criteria:**

- Short literals do not match arbitrary text substrings.
- All local matches record match kind and source.

### Task B6 — Refine Type A delete logic

**Type:** Repository implementation / classifier design  
**Dependencies:** B2  
**Output:** delete subtype redesign

Replace unconditional high-confidence TypeA/REJECTION for every delete with refined delete categories:

| Subtype | Meaning |
|---|---|
| REJECTION_RULE_INVALID | Rule identifies the value as invalid. |
| REJECTION_FORMAT_INVALID | Format invalidity justifies deletion or trivial rejection. |
| DELETE_SELECTION_LOCAL | Choosing what to delete requires local evidence. |
| DELETE_SELECTION_EXTERNAL | Choosing what to delete requires external evidence. |
| DELETE_AMBIGUOUS | Not enough evidence to classify confidently. |

**Metrics:**

- number of deletes downgraded from high-confidence TypeA;
- audit precision of delete subtypes;
- model performance on refined delete categories.

**Acceptance criteria:**

- Delete is no longer automatically equated with logical rejection in all cases.
- Single-value/unique-value conflicts are handled carefully.

### Task B7 — Refine format repair logic

**Type:** Repository implementation / classifier design  
**Dependencies:** B2  
**Output:** format subtype redesign

Format repairs should be TypeA only when the old-to-new change is deterministic normalization:

- strip trailing illegal character;
- strip whitespace;
- normalize obvious punctuation;
- normalize case if the rule clearly implies it;
- trivial date/literal shape normalization.

Otherwise route to lower-confidence TypeA, TypeB, TypeC, or unknown depending on evidence.

**Metrics:**

- number of format cases downgraded;
- audit precision of high-confidence format TypeA;
- model performance on format subtypes.

**Acceptance criteria:**

- Non-deterministic format updates are not high-confidence logical repairs.

### Task B8 — Fix/verify range and type/value-type handling

**Type:** Repository implementation / tests  
**Dependencies:** B2  
**Output:** corrected constraint handling and tests

Implement explicit qualifier handling:

| Constraint element | Property |
|---|---|
| Minimum quantity/value | P2313 |
| Maximum quantity/value | P2312 |
| Minimum date | P2310 |
| Maximum date | P2311 |
| Type/value class | P2308 |
| Type/value relation | P2309 |

Handle both:

- subject type constraint Q21503250;
- value-type constraint Q21510865.

**Metrics:**

- number of range cases reclassified;
- number of type/value-type T-box cases with meaningful set changes;
- exact/semantic T-box evaluator effect.

**Acceptance criteria:**

- Numeric and date boundaries are not conflated.
- Type and value-type constraints use P2308/P2309 rather than one-of qualifiers.

### Task B9 — Add classifier unit tests

**Type:** Repository implementation / tests  
**Dependencies:** B2–B8  
**Output:** unit-test suite

Add tests for:

1. post-repair target-property edge does not cause local match;
2. non-target L1 property QID can produce TypeB;
3. L2 label does not create QID match unless locally referenced;
4. short literal does not substring-match accidentally;
5. numeric range boundary via P2313/P2312 gives TypeA;
6. date range boundary via P2310/P2311 gives TypeA;
7. format update is TypeA only for simple normalization;
8. missing truth -> UNKNOWN_MISSING_TRUTH;
9. missing world state -> UNKNOWN_MISSING_WORLD_STATE;
10. value-type T-box expansion over P2308 gives set expansion;
11. delete under single-value conflict is not automatically high-confidence rejection.

**Metrics:**

- test pass/fail;
- branch coverage for classifier decision order.

**Acceptance criteria:**

- Tests pass.
- Old leakage self-test still passes.

### Task B10 — Generate classifier transition matrix

**Type:** Evaluation / analysis  
**Dependencies:** B2–B9  
**Output:** classifier transition report

Compare old and new classifications:

| Transition | Interpretation |
|---|---|
| TypeC -> TypeB | Old classifier missed local evidence. |
| TypeC -> TypeA | Old classifier missed rule evidence. |
| TypeC -> UNKNOWN | Old classifier overclaimed externality. |
| TypeA -> TypeB | Old classifier overclaimed logical determinism. |
| TypeA -> TypeC/UNKNOWN | Old classifier overclassified delete/format/range cases. |

**Metrics:**

- transition counts;
- transition rates by constraint family;
- examples per transition.

**Acceptance criteria:**

- Transition matrix is included in classifier audit appendix or internal report.
- New Type C semantics are reflected in dataset documentation.

## 3. Phase C — Dataset tiers and selection manifests

### Task C1 — Define full/core/dev tier policy

**Type:** Evaluation design / repository implementation  
**Dependencies:** B10  
**Output:** tier policy document

Define:

| Tier | Size | Purpose |
|---|---:|---|
| Full | all valid cases | Release/statistics. |
| Core | 3,000–6,000 | Main LLM experiments. |
| Dev/Pilot | 300–800 | Prompt development and debugging. |

**Acceptance criteria:**

- Core and dev are deterministic.
- Core excludes or separately marks low-confidence/unknown cases.
- Dev does not overlap final test/core evaluation if used for prompt tuning.

### Task C2 — Implement core selection manifest

**Type:** Repository implementation  
**Dependencies:** C1  
**Output:** `reports/benchmark_selection/core_*.json`

Selection rules:

- balance A-box information conditions;
- preserve T-box diversity;
- cap T-box cases per property revision at 5–20;
- stratify by subtype and popularity;
- keep medium-confidence cases but report separately;
- include low-confidence cases only as diagnostic/challenge slice.

**Metrics:**

- total selected cases;
- selected cases by track;
- selected cases by class/subtype;
- selected cases by confidence;
- selected cases by popularity bucket;
- T-box revisions and max count per revision;
- overlap with dev/test splits.

**Acceptance criteria:**

- No T-box revision dominates the core.
- All key strata have enough cases for analysis.

### Task C3 — Implement dev/pilot selection manifest

**Type:** Repository implementation / prompt development support  
**Dependencies:** C1  
**Output:** `reports/benchmark_selection/dev_prompt_*.json`

Dev set should include:

- enough examples of each track and information condition;
- enough T-box examples for prompt debugging;
- Type C external-by-elimination and unknown slices;
- head/mid/tail cases;
- cases likely to expose parser and representation issues.

**Metrics:**

- distribution table;
- overlap check with core final test if applicable.

**Acceptance criteria:**

- Dev is representative but small enough for prompt iteration.

### Task C4 — Update splitter if needed

**Type:** Repository implementation  
**Dependencies:** C1–C3  
**Output:** deterministic split artifact

Ensure splits can stratify by:

- repair locus;
- class/subtype;
- confidence;
- popularity bucket;
- T-box property revision;
- constraint family.

**Metrics:**

- split distribution deltas;
- maximum distribution deviation by stratum.

**Acceptance criteria:**

- Train/dev/test distributions are within acceptable deltas.
- T-box property revisions are not split in a way that leaks same-revision patterns when few-shot examples are used.

## 4. Phase D — Manual audit

### Task D1 — Build audit sample

**Type:** Manual audit / repository support  
**Dependencies:** B10, C1  
**Output:** audit sample JSONL/CSV

Sample 300–500 cases:

| Stratum | Target count |
|---|---:|
| TypeC / EXTERNAL_BY_ELIMINATION, QID truth | 50 |
| TypeC / EXTERNAL_BY_ELIMINATION, literal truth | 50 |
| TypeC sparse local graph | 50 |
| TypeC current-value fallback | all or 50 |
| TypeA format update | 50 |
| TypeA delete under single/unique-value constraints | 50 |
| TypeB local text | 50 |
| T-box generic schema update | 50 |

**Metrics:**

- audit sample distribution;
- coverage of constraint families;
- coverage of popularity buckets.

**Acceptance criteria:**

- Sample covers the highest-risk classifier decisions.

### Task D2 — Create annotation template

**Type:** Manual audit  
**Dependencies:** D1  
**Output:** audit spreadsheet or JSON schema

Fields:

- case_id;
- current class/subtype/confidence;
- repair locus correct?;
- target truth well-defined?;
- target visible locally?;
- extractor missed local evidence?;
- external evidence truly required?;
- Type C subtype judgment;
- core/challenge/exclude recommendation;
- notes;
- annotator id;
- timestamp.

**Metrics:**

- annotation completeness rate;
- disagreement rate if multiple annotators.

**Acceptance criteria:**

- Template supports direct computation of label precision and transition suggestions.

### Task D3 — Execute manual audit

**Type:** Manual audit  
**Dependencies:** D2  
**Output:** audit labels and notes

Audit each sampled case.

**Metrics:**

- label precision by stratum;
- Type C confirmed-external rate;
- Type C false-positive rate;
- TypeA overclaim rate;
- TypeB false-positive rate;
- recommended exclusion rate.

**Acceptance criteria:**

- Enough audited examples exist to support paper claims about classifier quality.

### Task D4 — Apply audit-informed filtering/reporting policy

**Type:** Evaluation design / repository implementation  
**Dependencies:** D3  
**Output:** updated core inclusion policy

Define which cases count in main core vs challenge/diagnostic:

- high-confidence confirmed or well-supported cases -> main core;
- medium-confidence external-by-elimination -> main core but separate slice;
- unknown/missing/sparse/current fallback -> challenge or excluded from main score;
- low-causality T-box -> separate slice.

**Metrics:**

- core size after policy;
- diagnostic slice size;
- excluded cases by reason.

**Acceptance criteria:**

- Main score is not dominated by low-confidence or unknown cases.

## 5. Phase E — Non-LLM baselines

### Task E1 — Implement majority and constant-track baselines

**Type:** Repository implementation / evaluation  
**Dependencies:** C2  
**Output:** baseline diagnosis predictions

Baselines:

- majority track;
- always A-box;
- always T-box;
- always ambiguous if useful.

**Metrics:**

- track accuracy;
- macro-F1;
- confusion matrix;
- A-box overuse rate;
- T-box miss rate.

**Acceptance criteria:**

- LLM track-diagnosis results can be compared against trivial baselines.

### Task E2 — Implement constraint-only Type A baseline

**Type:** Repository implementation / evaluation  
**Dependencies:** B8, C2  
**Output:** symbolic repair proposals for supported TypeA cases

Baseline solves only supported deterministic cases:

- one-of singleton;
- range boundary;
- simple format normalization;
- rule-invalid delete when safe.

**Metrics:**

- coverage;
- exact historical agreement;
- precision on covered cases;
- unsupported/abstained cases.

**Acceptance criteria:**

- Provides meaningful lower/upper reference for TypeA.

### Task E3 — Implement local lookup oracle

**Type:** Repository implementation / evaluation  
**Dependencies:** B4, C2  
**Output:** local lookup proposals or oracle labels

This baseline checks whether TypeB labels are operational:

- if target truth is present locally, output it;
- otherwise abstain.

**Metrics:**

- TypeB coverage;
- exact local-match rate;
- false TypeB rate;
- TypeC->local leakage detection.

**Acceptance criteria:**

- TypeB operationality can be validated independently of LLMs.

### Task E4 — Implement invalid/do-nothing baseline

**Type:** Repository implementation / evaluation sanity check  
**Dependencies:** C2  
**Output:** invalid or empty proposals

**Metrics:**

- evaluator lower-bound behavior;
- accepted rate should be near zero.

**Acceptance criteria:**

- Evaluator does not accept invalid/no-op proposals except in explicitly allowed cases.

## 6. Phase F — Prompt development on dev only

### Task F1 — Build prompt-development run matrix

**Type:** Prompt development / evaluation design  
**Dependencies:** C3, E1–E4  
**Output:** prompt dev experiment matrix

Use 300–800 dev cases.

Axes:

| Axis | Values |
|---|---|
| Representation | hybrid JSON+NL, pure NL, compact table/list, optional Turtle/RDF. |
| Examples | zero-shot, random same-task 2-shot, same-track 2-shot, matched 2-shot. |
| Context | logic_only, local_graph, optional minimal_case. |
| Task | track diagnosis, repair proposal. |
| Track mode | oracle for proposal dev; diagnosis_routed after diagnosis prompt is stable. |

**Metrics:**

- parse validity;
- proposal contract validity;
- proposal executability;
- exact historical agreement;
- T-box target-constraint hit;
- T-box semantic-family success;
- track diagnosis accuracy;
- auditability completeness;
- provenance completeness;
- request error rate;
- tokens per case;
- estimated cost;
- latency.

**Acceptance criteria:**

- Matrix is small enough to run with local model and no API spending.
- Final prompt selection criteria are defined before inspecting all results.

### Task F2 — Test representation variants

**Type:** Prompt development  
**Dependencies:** F1  
**Output:** representation comparison report

Compare:

1. hybrid JSON + controlled natural language;
2. pure natural language;
3. compact table/list;
4. RDF/Turtle-like triples if feasible.

**Primary selection metrics:**

- parse validity;
- proposal executability;
- exact historical agreement;
- token count.

**Secondary metrics:**

- track diagnosis;
- T-box semantic-family success;
- auditability.

**Acceptance criteria:**

- Select one representation for main runs.
- If Turtle is worse but included for venue relevance, restrict it to appendix/dev results.

### Task F3 — Test example policies

**Type:** Prompt development / few-shot design  
**Dependencies:** F1, F2  
**Output:** few-shot policy comparison

Compare:

- zero-shot;
- random same-task examples;
- same-track examples;
- matched examples.

Matched example criteria:

1. same task;
2. same repair locus;
3. same constraint family;
4. same subtype/action;
5. same information condition;
6. same value datatype;
7. similar popularity bucket as optional final tie-breaker.

Hard exclusions:

- same case;
- same QID;
- same T-box property revision;
- final test/core leakage;
- same property unless explicitly testing precedent retrieval.

**Metrics:**

- format/schema validity gain;
- exact repair gain;
- semantic T-box gain;
- tokens and cost increase;
- examples causing copying errors.

**Acceptance criteria:**

- Decide whether few-shot belongs in main core run or only as ablation.
- Zero-shot remains the main reasoning floor unless few-shot is redefined as a separate research question.

### Task F4 — Test abstention prompt if implemented

**Type:** Prompt development / repository implementation  
**Dependencies:** B2, F2  
**Output:** abstention schema and dev results

Add structured abstention for insufficient-evidence cases.

**Metrics:**

- justified abstention rate on TypeC;
- false abstention rate on TypeA/TypeB;
- hallucinated repair rate on TypeC;
- repair success when not abstaining;
- calibration of uncertainty.

**Acceptance criteria:**

- If abstention improves TypeC interpretability without destroying TypeA/B repair, include it.
- If abstention is too complex, omit from paper 1 and frame TypeC as no-retrieval stress.

### Task F5 — Freeze final prompts

**Type:** Prompt development / reproducibility  
**Dependencies:** F2–F4  
**Output:** final prompt templates and prompt version id

Freeze:

- diagnosis prompt;
- A-box proposal prompt;
- T-box proposal prompt;
- optional abstention prompt;
- representation format;
- context-bundle rendering;
- generation parameters.

**Acceptance criteria:**

- Prompt templates are versioned.
- No further prompt edits after main results are generated.

## 7. Phase G — Main reasoning-floor experiments

### Task G1 — Dry run on 50–100 cases

**Type:** Evaluation / execution  
**Dependencies:** F5, C2  
**Output:** dry-run report

Run final prompts on a tiny stratified sample.

**Metrics:**

- parser error rate;
- request error rate;
- average tokens;
- evaluator completion;
- obvious prompt leakage or wrong schema usage.

**Acceptance criteria:**

- No blocking parser/evaluator bug.
- Estimated cost/throughput is acceptable.

### Task G2 — Pilot run on 1,000–1,500 cases

**Type:** Evaluation / execution  
**Dependencies:** G1  
**Output:** pilot reasoning-floor summary

Run one local model across:

- logic_only;
- local_graph;
- oracle mode;
- diagnosis-routed mode if diagnosis is stable.

**Metrics:**

- all main evaluation metrics;
- model failure modes;
- context gain by class;
- oracle vs routed gap;
- T-box exact vs semantic gap.

**Acceptance criteria:**

- No major benchmark/evaluator bug discovered.
- Failure taxonomy is refined before full run.

### Task G3 — Main local-model run on core

**Type:** Evaluation / execution  
**Dependencies:** G2  
**Output:** main local-model results

Run two or three local models on the core dataset.

Axes:

| Axis | Values |
|---|---|
| Context | logic_only, local_graph. |
| Track mode | oracle, diagnosis_routed. |
| Prompt | final zero-shot contract prompt. |
| Model | small, medium, and possibly large local instruction models. |

**Metrics:**

- parse validity;
- proposal validity;
- executability;
- exact historical agreement;
- semantic-family success;
- track diagnosis accuracy;
- auditability completeness;
- regression pass;
- tokens/cost/latency;
- request errors.

**Acceptance criteria:**

- All runs use identical core manifest.
- Deterministic generation settings are recorded.
- Run manifests and summaries are archived.

### Task G4 — API reference subset

**Type:** Evaluation / execution  
**Dependencies:** G2  
**Output:** API calibration results

Run one API model on 500–1,000 stratified cases.

**Metrics:**

- same as G3;
- cost per case;
- cost per accepted repair;
- qualitative comparison to local models.

**Acceptance criteria:**

- API subset is stratified and reproducible.
- API is not used as the only evidence for main claims.

### Task G5 — Few-shot ablation on core subset

**Type:** Evaluation / prompt ablation  
**Dependencies:** F3, G3  
**Output:** few-shot ablation results

Run zero-shot vs selected few-shot policies on a representative subset or full core if affordable.

**Metrics:**

- schema compliance gain;
- exact repair gain;
- T-box semantic-family gain;
- TypeC behavior;
- tokens/cost increase;
- example copying errors.

**Acceptance criteria:**

- Few-shot claims are separate from zero-shot reasoning-floor claims.

## 8. Phase H — Analysis and reporting

### Task H1 — Aggregate main results

**Type:** Analysis  
**Dependencies:** G3–G5  
**Output:** main tables

Tables:

1. track diagnosis by model/context;
2. oracle-track repair success by class/subtype;
3. diagnosis-routed repair success by class/subtype;
4. local_graph minus logic_only gain by information condition;
5. T-box exact vs semantic success;
6. parse/executability/auditability decomposition;
7. head/mid/tail performance;
8. tokens/cost per accepted repair.

**Metrics:** all main metrics.

**Acceptance criteria:**

- Results answer each RQ.
- No aggregate hides T-box revision duplication.

### Task H2 — Statistical testing

**Type:** Analysis  
**Dependencies:** H1  
**Output:** confidence intervals and significance notes

Use:

- paired bootstrap confidence intervals for model/context differences;
- macro-average by T-box property revision;
- macro-average by subtype where appropriate;
- head-tail stratified comparisons;
- effect sizes, not just p-values.

**Metrics:**

- confidence intervals for local_graph gain;
- confidence intervals for oracle-routed gap;
- confidence intervals for model differences;
- T-box cluster-robust estimates.

**Acceptance criteria:**

- Main claims are backed by uncertainty estimates.

### Task H3 — Failure taxonomy analysis

**Type:** Analysis / manual review  
**Dependencies:** H1  
**Output:** failure taxonomy table and examples

Code failures into:

- parse failure;
- contract failure;
- wrong locus;
- wrong target;
- wrong operation;
- wrong value;
- over-delete;
- under-repair;
- hallucinated provenance;
- non-auditable;
- constraint regression;
- exact mismatch but semantic match;
- abstention error if applicable.

**Metrics:**

- failure distribution by class/subtype/model;
- top failure modes for TypeC and T-box;
- qualitative examples.

**Acceptance criteria:**

- Paper has more than raw scores; it explains why systems fail.

### Task H4 — Classifier audit report

**Type:** Analysis / writing  
**Dependencies:** D3, B10  
**Output:** classifier validation section or appendix

Report:

- deterministic decision priority;
- Type C redesign;
- manual-audit precision;
- transition matrix;
- low-confidence handling;
- limitations.

**Acceptance criteria:**

- Reviewer concern about heuristic Type C is preempted.

### Task H5 — Dataset card and benchmark documentation

**Type:** Writing / repository documentation  
**Dependencies:** C2, D4, H1  
**Output:** dataset card

Include:

- source and construction;
- intended use;
- limitations;
- fields;
- temporal controls;
- splits;
- core/full/dev distinction;
- label semantics;
- known biases;
- ethical and community considerations;
- compute requirements.

**Acceptance criteria:**

- Dataset release is understandable without reading code.

## 9. Phase I — Paper writing

### Task I1 — Draft introduction and narrative

**Type:** Writing  
**Dependencies:** A1–A3, H1  
**Output:** introduction draft

Core story:

> KG repair requires deciding whether to fix the fact, the context, or the rule; current LLM evaluations do not isolate this decision; WikidataRepairEval provides a temporally controlled benchmark built from historical repairs and executable evaluation.

**Acceptance criteria:**

- Introduction names the scientific problem before describing the dataset.
- Contributions are not just “we built a benchmark.”

### Task I2 — Draft benchmark construction section

**Type:** Writing  
**Dependencies:** B10, C2, D4  
**Output:** methods section draft

Include:

- historical reconstruction;
- world-state context;
- temporal target-property rule;
- classification;
- core/full/dev tiers;
- selection caps;
- audit.

**Acceptance criteria:**

- A reader can understand the benchmark without repository internals.

### Task I3 — Draft tasks and evaluation section

**Type:** Writing  
**Dependencies:** F5, H1  
**Output:** tasks/evaluation section

Include:

- track diagnosis;
- oracle-track repair;
- diagnosis-routed repair;
- A-box proposal contract;
- T-box proposal contract;
- metrics;
- auditability;
- semantic-family evaluation;
- cost telemetry.

**Acceptance criteria:**

- Metrics are linked to RQs.
- Exact and semantic metrics are clearly separated.

### Task I4 — Draft experiments section

**Type:** Writing  
**Dependencies:** G3–G5  
**Output:** experiments section

Include:

- datasets;
- models;
- prompts;
- baselines;
- decoding settings;
- compute setup;
- API-credit limitation;
- statistical methods.

**Acceptance criteria:**

- Experiments are reproducible and not overclaimed.

### Task I5 — Draft results and discussion

**Type:** Writing / analysis  
**Dependencies:** H1–H3  
**Output:** results section

Organize by RQ:

1. information condition effects;
2. locus diagnosis;
3. executable vs valid outputs;
4. prompt/context effects;
5. head-tail effects;
6. few-shot ablation;
7. cost/telemetry.

**Acceptance criteria:**

- Each result links to a hypothesis.
- Negative results are interpreted as benchmark insight, not failure.

### Task I6 — Draft limitations

**Type:** Writing  
**Dependencies:** all analysis tasks
**Output:** limitations section

Include:

- historical repair target imperfection;
- Type C by elimination;
- partial evaluator;
- T-box skew;
- temporal complexity;
- prompt leakage risk;
- limited API comparison;
- no retrieval in paper 1.

**Acceptance criteria:**

- Limitations are explicit enough to preempt obvious reviewer attacks.

### Task I7 — Prepare figures and tables

**Type:** Analysis / writing  
**Dependencies:** H1–H3  
**Output:** camera-ready figures/tables

Recommended figures:

1. conceptual diagram: repair locus × information condition;
2. pipeline diagram: repair reconstruction -> world state -> classification -> prompting -> evaluation;
3. context ablation result plot;
4. oracle vs diagnosis-routed gap plot;
5. T-box exact vs semantic success plot;
6. failure taxonomy stacked chart.

Recommended tables:

1. dataset statistics;
2. core selection distribution;
3. model list and compute settings;
4. main metrics by class;
5. T-box metrics;
6. manual audit results;
7. limitations/mitigations.

**Acceptance criteria:**

- Main paper figures fit the narrative and avoid leaderboard-only framing.

### Task I8 — Internal review and submission package

**Type:** Writing / project management  
**Dependencies:** I1–I7  
**Output:** submission-ready paper and artifacts

Prepare:

- paper draft;
- appendix;
- anonymized repository if needed;
- dataset card;
- run manifests;
- core selection manifest;
- reproducibility checklist;
- ethical considerations;
- limitations.

**Acceptance criteria:**

- Paper can be reviewed without private code access.
- Claims are supported by results and audit evidence.

## 10. Phase J — Contingency and paper 2 preparation

### Task J1 — Guardian-lite contingency experiment

**Type:** Repository implementation / evaluation  
**Dependencies:** G3, H3  
**Output:** small verifier-retry experiment, only if needed

Use only if reviews or internal assessment suggest benchmark alone is too weak.

Protocol:

1. model proposes repair;
2. verifier checks validity, executability, auditability, and simple regression;
3. if rejected, model receives structured diagnostics;
4. one retry allowed;
5. evaluate accepted repair rate and cost.

Use 300–500 core cases.

**Metrics:**

- accepted repair rate improvement;
- retry success rate;
- verifier rejection categories;
- additional tokens/cost;
- hallucinated provenance reduction;
- latency increase.

**Acceptance criteria:**

- Guardian-lite is small enough not to distract from benchmark paper.
- If included, it is framed as demonstration of benchmark utility, not full method contribution.

### Task J2 — Paper 2 design memo

**Type:** Research planning  
**Dependencies:** H3, J1 optional  
**Output:** RAG/Guardian paper outline

Paper 2 question:

> Can retrieval and verifier-guided repair-time control improve accepted KG repair while controlling cost, hallucination, and unsafe transactions?

Compare:

| System | Retrieval | Verifier | Retry loop |
|---|---:|---:|---:|
| LLM zero-shot | no | no | no |
| LLM + RAG | yes | no | no |
| LLM + verifier | no | yes | yes |
| LLM + RAG + verifier | yes | yes | yes |
| Symbolic heuristic | no | yes | no |

**Metrics:**

- accepted repair rate;
- justified abstention;
- hallucinated provenance;
- verifier rejection rate;
- successful retry rate;
- tokens-to-fix;
- retrieval calls;
- verifier calls;
- latency;
- cost per accepted repair.

**Acceptance criteria:**

- Paper 2 scope remains separate from paper 1.

## 11. Execution order summary

The minimum viable order is:

1. Freeze paper scope and taxonomy positioning.
2. Snapshot current classifier counts.
3. Redesign Type C and classifier high-risk branches.
4. Add classifier tests.
5. Generate transition matrix.
6. Build full/core/dev selection manifests.
7. Run manual audit.
8. Apply audit-informed core policy.
9. Implement non-LLM baselines.
10. Run prompt development on dev only.
11. Freeze final prompts.
12. Dry run.
13. Pilot run.
14. Main local-model core run.
15. API reference subset.
16. Few-shot ablation.
17. Aggregate and statistically analyze results.
18. Write classifier audit and dataset card.
19. Draft paper.
20. Add Guardian-lite only if needed.

## 12. Main go/no-go gates

### Gate 1 — Classifier gate

Proceed to LLM experiments only if:

- Type C is no longer a single overstrong external bucket;
- transition matrix is understood;
- manual audit shows acceptable precision for main-core labels;
- low-confidence cases are separated.

### Gate 2 — Prompt gate

Proceed to main core only if dev runs show:

- parse error rate is acceptable;
- proposal contracts are mostly followed;
- no obvious leakage;
- token cost is manageable;
- final prompt is frozen.

### Gate 3 — Pilot gate

Proceed to full core only if pilot shows:

- evaluator runs end-to-end;
- request errors are manageable;
- no systemic wrong-schema issue;
- failure modes are interpretable.

### Gate 4 — Paper gate

Submit paper 1 if results can support at least these claims:

1. information conditions produce distinct model behavior;
2. local graph context helps local cases more than logical cases;
3. repair-locus diagnosis is a bottleneck;
4. valid JSON overestimates repair success;
5. T-box exact and semantic metrics reveal different failure modes;
6. Type C no-retrieval cases expose insufficiency, memorization, or abstention behavior.

If these claims cannot be supported, add Guardian-lite or expand the method contribution.

## 13. Suggested repository TODO checklist

- [ ] Add classifier Type C subtypes.
- [ ] Remove/quarantine 2026 truth fallbacks.
- [ ] Expand local context buckets.
- [ ] Tighten literal matching.
- [ ] Split delete subtypes.
- [ ] Refine format determinism.
- [ ] Fix numeric/date range qualifiers.
- [ ] Handle subject type and value-type constraints with P2308/P2309.
- [ ] Add classifier unit tests.
- [ ] Add old/new transition matrix script.
- [ ] Add core/dev selection manifests.
- [ ] Add audit sample generator.
- [ ] Add audit annotation template.
- [ ] Add non-LLM baselines.
- [ ] Add prompt representation variants.
- [ ] Add few-shot exemplar selector with leakage controls.
- [ ] Add optional abstention schema.
- [ ] Freeze final prompts.
- [ ] Run dry/pilot/main experiments.
- [ ] Add bootstrap and T-box macro-averaging analysis.
- [ ] Generate paper tables and figures.
- [ ] Write dataset card and limitations.
