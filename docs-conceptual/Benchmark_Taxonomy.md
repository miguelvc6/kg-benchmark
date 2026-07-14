# Benchmark Taxonomy

This document summarizes the research-facing labels used by WikidataRepairEval. The rationale and paper narrative are
in [Research Objectives and Paper Narrative](./Research_Objectives_and_Paper_Narrative.md). Technical artifact layouts
belong in [technical documentation](../docs-technical/README.md).

The benchmark has two orthogonal axes:

- **repair locus**: where the recorded historical edit occurred;
- **information-access condition**: which supported evidence channel, if any, identifies an A-box historical target under controlled context.

Keeping these axes separate avoids presenting the existing A-box/T-box repair distinction as a new taxonomy.

## Repair Locus

The repair-locus axis follows existing Wikidata repair-taxonomy work.

| Label | Meaning | Evaluation question |
|---|---|---|
| `A_BOX` | The repair edits entity-level instance data. | Can the model edit the correct entity/property/value while preserving useful information? |
| `T_BOX` | The repair edits the property constraint or schema layer. | Can the model identify the correct constraint family and evidence-supported schema patch? |
| `AMBIGUOUS` | Evidence overlaps or the historical repair locus is unclear. | Can the model represent uncertainty instead of forcing an overconfident edit? |

### A-box Entity Repairs

A-box repairs include deleting invalid values, replacing values, adding missing values, or preserving valid values while removing invalid ones. Evaluation reconstructs the pre-repair target-property state, applies the proposed transaction, and compares the resulting state to the historical repaired state.

### T-box Schema Reforms

T-box repairs include range changes, allowed-set or allowed-class changes, constraint metadata updates, and other
property-level schema reforms. Confirmatory evaluation uses a bounded taxonomy-patch task: it separately measures the
schema decision, affected constraint family, mechanically supported repair operation, and visible value delta. Full
post-reform signature reconstruction is retained only as a legacy exploratory diagnostic.

Repeated T-box manifestations should be controlled in paper-facing subsets because a single property revision can explain many apparent violation rows.

## Information-Access Conditions

Information-access labels apply to A-box repairs. They describe the result of the current supported rule and context checks against the historical target. They are operational extractor labels, not claims about semantic correctness, causal necessity, or repair uniqueness.

The code may continue to use `TypeA`, `TypeB`, and `TypeC`, but paper-facing text should also use neutral names such as `IC-L`, `IC-G`, and `IC-E`/`IC-U`.

| Code label | Paper-facing label | Meaning |
|---|---|---|
| `TypeA` | `IC-L: rule-matched` | A supported rule, violation-shape, or consistency pattern maps to the historical target. |
| `TypeB` | `IC-G: local match` | The extractor matches the historical target in the focus node or bounded local graph context. |
| `TypeC` | `IC-E-elim` or `IC-U` | Supported rule and local checks do not identify the target, or required inputs are unresolved. |

### Type A: Logical / Rule-Implied

Type A cases match supported violation-shape, constraint, or internal-consistency rules without graph traversal. This does not prove that the historical edit is the only semantically valid repair.

Examples include:

- simple format normalization where the implemented rule maps to the historical normalized value;
- singleton one-of constraints where exactly one value is allowed;
- range-boundary corrections where the target is implied by the rule;
- deletion when the rule itself identifies the invalid value.

Not every delete is automatically high-confidence Type A. If the implemented rule does not identify the historical deletion, the case should be downgraded, split into a more specific delete subtype, or routed to diagnostics.

### Type B: Local Graph-Grounded

Type B cases are those for which the supported extractor finds a match in the focus node or immediate graph neighborhood. Match presence does not by itself establish that the field semantically justifies the repair.

Local evidence can include:

- reconstructed pre-repair target-property values for diagnostics only, not as independent local support;
- non-target focus-node properties;
- one-hop neighbor ids;
- focus-node labels and descriptions;
- neighbor labels and descriptions;
- labels for locally referenced ids;
- relevant local constraint context.

These cases should benefit most from the `local_graph` context bundle compared with `logic_only`.

### Type C: Not Identified By Supported Checks Or Unresolved

Type C must be interpreted conservatively. A fallback label means only that the current rule/local extractor did not find the historical target. It does not prove that external evidence or retrieval is required.

Recommended subtypes:

| Subtype | Meaning |
|---|---|
| `EXTERNAL_BY_ELIMINATION` | Supported rule/local checks failed to identify the target. |
| `UNKNOWN_MISSING_WORLD_STATE` | Required frozen context is missing or incomplete. |
| `UNKNOWN_MISSING_TRUTH` | The historical repair target is not sufficiently represented. |
| `UNKNOWN_CURRENT_VALUE_FALLBACK` | Classification depended on a current-value fallback and should be leakage-audited. |
| `UNKNOWN_INCOMPLETE_LOCAL_CONTEXT` | Local graph context is too sparse to make a strong claim. |

The current study does not assign `EXTERNAL_CONFIRMED`. Type C is reported only with the extractor-relative caveat above.

## Validation Without Independent Human Evaluation

No independent human evaluation is available. The validation fallback combines exhaustive automated consistency auditing
over every Stage 4 record and every supplied or final rendered prompt with label-hidden Codex-assisted error discovery.
Missing Stage 2 records or required fields are explicit audit outcomes rather than empty evidence. Codex nominations are
exploratory: they are not gold labels, independent annotations, inter-annotator agreement, or evidence of causal or
semantic uniqueness. The completed `full_v1` run found 1,621 deterministic disagreements and placed model-assisted
concerns in conservative diagnostic or rerender dispositions. That run informed deterministic rules but does not
certify the final dataset; its role is recorded in the
[development history](../docs-technical/Development_History.md).

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
| `COINCIDENTAL_SCHEMA_CHANGE` | Schema changed, but report-to-delta alignment is weak. |

## Dataset Views

The release contains one canonical dataset and several deterministic views of its fully audited eligible pool. The
initial main view has 1,200 independent cases; the Azure view is a nested 600-case prefix. A reserved support bank is
excluded from every evaluation population. Larger experiments use longer prefixes of the same stable per-stratum order,
so they do not redefine or resample earlier cases.
