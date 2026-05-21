# Benchmark Taxonomy

This document summarizes the research-facing labels used by WikidataRepairEval. The full conceptual rationale is in [Full Project Description](./00-full_project_description.md) and the paper narrative is in [KG LLM Benchmark Narrative](./00-kg_llm_benchmark.md). Technical artifact layouts belong in [docs-technical](../docs-technical/README.md).

The benchmark has two orthogonal axes:

- **repair locus**: where the historical repair happened or should happen;
- **information-access condition**: what kind of evidence is needed to justify an A-box repair target under controlled context.

Keeping these axes separate avoids presenting the existing A-box/T-box repair distinction as a new taxonomy.

## Repair Locus

The repair-locus axis follows existing Wikidata repair-taxonomy work.

| Label | Meaning | Evaluation question |
|---|---|---|
| `A_BOX` | The repair edits entity-level instance data. | Can the model edit the correct entity/property/value while preserving useful information? |
| `T_BOX` | The repair edits the property constraint or schema layer. | Can the model reform the correct constraint family and signature? |
| `AMBIGUOUS` | Evidence overlaps or the causal repair locus is unclear. | Can the model represent uncertainty instead of forcing an overconfident edit? |

### A-box Entity Repairs

A-box repairs include deleting invalid values, replacing values, adding missing values, or preserving valid values while removing invalid ones. Evaluation reconstructs the pre-repair target-property state, applies the proposed transaction, and compares the resulting state to the historical repaired state.

### T-box Schema Reforms

T-box repairs include range changes, allowed-set or allowed-class changes, constraint metadata updates, and other property-level schema reforms. Evaluation compares the proposed post-reform constraint signature with the historical post-reform signature and also reports semantic-family metrics.

Repeated T-box manifestations should be controlled in paper-facing subsets because a single property revision can explain many apparent violation rows.

## Information-Access Conditions

Information-access labels apply to A-box repairs. They describe what information would be needed to reproduce or justify the historical repair target.

The code may continue to use `TypeA`, `TypeB`, and `TypeC`, but paper-facing text should also use neutral names such as `IC-L`, `IC-G`, and `IC-E`/`IC-U`.

| Code label | Paper-facing label | Meaning |
|---|---|---|
| `TypeA` | `IC-L: logical / rule-implied` | The rule, violation shape, or internal consistency is enough to determine the repair. |
| `TypeB` | `IC-G: local graph-grounded` | The repair target is available in the focus node or bounded local graph context. |
| `TypeC` | `IC-E` or `IC-U` | Supported rule and local evidence do not identify the target; this may require external evidence or may be unresolved by the current extractor. |

### Type A: Logical / Rule-Implied

Type A cases should be solvable from the violation shape, constraint, or internal consistency without graph traversal or retrieval.

Examples include:

- simple format normalization where the normalized value is uniquely determined;
- singleton one-of constraints where exactly one value is allowed;
- range-boundary corrections where the target is implied by the rule;
- deletion when the rule itself identifies the invalid value.

Not every delete is automatically high-confidence Type A. If selecting which value to delete requires local or external evidence, the case should be downgraded, split into a more specific delete subtype, or audited.

### Type B: Local Graph-Grounded

Type B cases require information available in the focus node or immediate graph neighborhood.

Local evidence can include:

- reconstructed pre-repair target-property values;
- non-target focus-node properties;
- one-hop neighbor ids;
- focus-node labels and descriptions;
- neighbor labels and descriptions;
- labels for locally referenced ids;
- relevant local constraint context.

These cases should benefit most from the `local_graph` context bundle compared with `logic_only`.

### Type C: External, Non-Local, Or Unresolved

Type C must be interpreted conservatively. A fallback label often means only that the current rule/local extractor did not find the target truth. It does not by itself prove that external evidence is required.

Recommended subtypes:

| Subtype | Meaning |
|---|---|
| `EXTERNAL_CONFIRMED` | Manual audit or retrieval shows that non-local evidence is needed. |
| `EXTERNAL_BY_ELIMINATION` | Supported rule/local checks failed to identify the target. |
| `UNKNOWN_MISSING_WORLD_STATE` | Required frozen context is missing or incomplete. |
| `UNKNOWN_MISSING_TRUTH` | The historical repair target is not sufficiently represented. |
| `UNKNOWN_CURRENT_VALUE_FALLBACK` | Classification depended on a current-value fallback and should be leakage-audited. |
| `UNKNOWN_INCOMPLETE_LOCAL_CONTEXT` | Local graph context is too sparse to make a strong claim. |

For final experiments, Type C should be manually audited or reported with these caveats.

## T-box Subtypes

T-box cases should be reported separately from A/B/C information-access labels.

Candidate schema-reform subtypes include:

| Subtype | Interpretation |
|---|---|
| `RELAXATION_RANGE_WIDENED` | Numeric/date range became more permissive. |
| `RESTRICTION_RANGE_NARROWED` | Numeric/date range became stricter. |
| `RELAXATION_SET_EXPANSION` | Allowed value/class set expanded. |
| `RESTRICTION_SET_CONTRACTION` | Allowed value/class set contracted. |
| `SCHEMA_UPDATE` | Schema changed but direction is generic or not confidently typed. |
| `COINCIDENTAL_SCHEMA_CHANGE` | Schema changed, but causal relation to the violation is weak. |

## Dataset Tiers

The full dataset is the canonical historical record. Paper-facing evaluation should use fixed deterministic subsets:

| Tier | Purpose |
|---|---|
| Full dataset | Release and descriptive statistics. |
| Core dataset | Main LLM evaluation, stratified across repair locus, information condition, subtype, popularity, confidence, and T-box clusters. |
| Dev/Pilot set | Prompt engineering, representation ablations, debugging, and failure analysis. |
| Audit set | Manual validation of classifier labels, especially Type C and weak T-box cases. |

Classifier confidence should be a stratification variable, not a simple inclusion filter.
