# T-Box Taxonomy Patch Task

This document defines the implementation-facing specification for the Ferranti-style T-box taxonomy patch task. The task replaces the previous strict T-box post-repair signature reconstruction as the headline T-box proposal task. The old `signature_after` reconstruction metric remains useful as a strict historical diagnostic, but it is not the primary T-box target for taxonomy-patch runs.

The A-box repair proposal task is unchanged.

## Task Shape

A T-box taxonomy patch answer describes whether the historical evidence supports a causal schema repair and, when it does, which schema-level repair operation is visible from the benchmark artifacts. The answer is a compact patch, not a full reconstructed post-repair constraint signature.

The model-visible task should explain every output field and operation in plain language. It must not expose internal benchmark labels, private classifier shortcuts, or dev-derived repair recipes. Gold answers are extracted mechanically from the classified benchmark records and their historical constraint deltas.

## Taxonomy

| Taxonomy code | Operation name | Meaning | Current extraction support |
|---|---|---|---|
| `C_MINUS` | `CONSTRAINT_REMOVE` | Remove a property-constraint statement or constraint family from a property. | Fully supported when the family is present before the historical edit and absent after it. |
| `C_D` | `CONSTRAINT_DEPRECATE` | Deprecate or deactivate a constraint statement by rank or status. | Supported when rank/status fields in the before/after signatures show a transition to deprecated or inactive. |
| `C_PLUS` | `CONSTRAINT_ADD` | Add a property-constraint statement or constraint family to a property. | Fully supported when the family is absent before the historical edit and present after it. |
| `C_REPLACE` | `CONSTRAINT_TYPE_REPLACE` | Replace one constraint family/type with another. This may also be represented as a remove plus an add when pairing is ambiguous. | Supported only when the before/after diff contains a clear one-family removal paired with a clear one-family addition. Otherwise the extractor emits separate remove and add operations. |
| `CQ_PLUS` | `CONSTRAINT_QUALIFIER_ADD` | Add a qualifier value to an existing constraint definition. | Fully supported when qualifier value sets are available before and after and values are added for a qualifier property. |
| `CQ_MINUS` | `CONSTRAINT_QUALIFIER_REMOVE` | Remove a qualifier value from an existing constraint definition. | Fully supported when qualifier value sets are available before and after and values are removed for a qualifier property. |
| `CQ_REPLACE` | `CONSTRAINT_QUALIFIER_REPLACE` | Replace a qualifier value on the same qualifier property, equivalent to a remove plus an add on that qualifier property. | Fully supported when both added and removed values are visible for the same qualifier property. |
| `SUBCLASS_PLUS` | `CLASS_HIERARCHY_ADD` | Add a subclass relation that resolves the violation through class hierarchy rather than by editing the constraint statement itself. | Supported by the schema but not currently extractable from the existing constraint-delta artifacts. It requires class-hierarchy delta mining. |
| `E_PLUS` | `EXCEPTION_ADD` | Add an exception to the constraint so the violating item or value is explicitly exempted. | Partially supported by the schema. Current extraction requires reliable detection of exception qualifier properties and the added exception value; otherwise cases are routed to `OTHER_TBOX_UPDATE`. |
| `OTHER` | `OTHER_TBOX_UPDATE` | Schema-level repair not covered by the listed operations, including visible rank/snaktype/status changes that are not deprecations. | Supported as a fallback for visible schema-level diffs that are not covered by more specific extractable operations. |

`C_PLUS`, `C_REPLACE`, and `OTHER` are engineering extensions. Ferranti et al.'s repair taxonomy is used as the local framing, but the benchmark must represent every selected T-box record with mechanically extractable gold. Additions, replacements, and a conservative fallback are needed for complete diff coverage in this repository's historical artifacts.

## Answer Schema Enums

The schema-level decision field has three values:

| Value | Meaning |
|---|---|
| `CAUSAL_SCHEMA_REPAIR` | The record has evidence that the property schema changed in a way that causally addresses the violation. A non-empty repair list is expected. |
| `NO_CAUSAL_SCHEMA_REPAIR` | The property schema changed historically, but the change is coincidental relative to the selected violation. The repair list may be empty. |
| `UNCLEAR_SCHEMA_EVIDENCE` | The available historical or local evidence is too weak, missing, or ambiguous to assert a causal schema repair. The repair list may be empty. |

Repair operation names map one-to-one to taxonomy codes:

| `repair_op` | `taxonomy_code` |
|---|---|
| `CONSTRAINT_REMOVE` | `C_MINUS` |
| `CONSTRAINT_DEPRECATE` | `C_D` |
| `CONSTRAINT_ADD` | `C_PLUS` |
| `CONSTRAINT_TYPE_REPLACE` | `C_REPLACE` |
| `CONSTRAINT_QUALIFIER_ADD` | `CQ_PLUS` |
| `CONSTRAINT_QUALIFIER_REMOVE` | `CQ_MINUS` |
| `CONSTRAINT_QUALIFIER_REPLACE` | `CQ_REPLACE` |
| `CLASS_HIERARCHY_ADD` | `SUBCLASS_PLUS` |
| `EXCEPTION_ADD` | `E_PLUS` |
| `OTHER_TBOX_UPDATE` | `OTHER` |

Repair evidence levels:

| Value | Meaning |
|---|---|
| `FAMILY_ONLY` | The target constraint family is visible, but the concrete operation or value delta is not available. |
| `OPERATION_VISIBLE` | The operation and constraint family are visible, but concrete added or removed values are not available. |
| `VALUE_DELTA_VISIBLE` | Concrete added and/or removed values are visible in the historical delta and appear in the gold patch. |

Rank values:

| Value | Meaning |
|---|---|
| `normal` | The constraint statement has normal rank after the repair. |
| `preferred` | The constraint statement has preferred rank after the repair. |
| `deprecated` | The constraint statement has deprecated rank after the repair. |
| `null` | Rank is not applicable or not visible for this repair entry. |

Snaktype values:

| Value | Meaning |
|---|---|
| `VALUE` | The repaired statement or qualifier has an explicit value. |
| `SOMEVALUE` | The repaired statement or qualifier indicates an unknown value exists. |
| `NOVALUE` | The repaired statement or qualifier indicates no value exists. |
| `null` | Snaktype is not applicable or not visible for this repair entry. |

Provenance kinds:

| Value | Meaning |
|---|---|
| `KG` | Evidence comes from benchmark-visible knowledge graph fields, including historical constraint signatures and violation context. |
| `OTHER` | Evidence comes from another benchmark-visible artifact that is not a web source or direct KG node. |

## Extractable Fields

The initial gold extractor may use only benchmark-internal historical artifacts:

- record id
- property id
- track
- classification class, subtype, and diagnostics
- repair target kind and property revision id
- constraint-delta `signature_before` and `old_constraints`
- constraint-delta `signature_after` and `new_constraints`
- constraint-delta `changed_constraint_types`
- focus qid
- violation context

The fields above fully support constraint family additions and removals, qualifier value additions and removals, qualifier value replacements, clear one-to-one constraint family replacements, and visible deprecation/rank/snaktype changes. They do not by themselves support arbitrary class-hierarchy repairs because those require before/after subclass relation mining outside the property constraint signature. Exception additions are supported only when the relevant exception qualifier and added exception value are visible in the constraint delta.

## Coincidental And Unclear Cases

Coincidental schema changes are represented with:

```json
{
  "schema_decision": "NO_CAUSAL_SCHEMA_REPAIR",
  "repairs": []
}
```

The target property and constraint family should still be filled when they are visible in the record, because they are useful for analysis. No repair operation is emitted unless the record-level evidence supports a causal schema repair.

Unknown, weak, missing, or ambiguous T-box causality is represented with:

```json
{
  "schema_decision": "UNCLEAR_SCHEMA_EVIDENCE",
  "repairs": []
}
```

If the property or constraint family is visible, the target should be copied into the answer. If no visible family is available, the parser and schema may allow a documented target relaxation for unclear cases only.

## Unsupported But Valid Operations

`CLASS_HIERARCHY_ADD` is a valid schema operation, but the current benchmark artifacts do not mine class-hierarchy deltas. The extractor must not hallucinate subclass additions from violation context alone. If a selected T-box record appears to require a class-hierarchy delta for coverage, that is a coverage blocker until the mining support exists.

`EXCEPTION_ADD` is valid when an exception qualifier is added and the exception value is visible. If the current data cannot reliably distinguish an exception addition from a generic qualifier addition, the extractor should document that limitation in the gold summary and use `OTHER_TBOX_UPDATE` only when a visible schema-level update exists but no more specific supported operation can be assigned.

For all unsupported-but-theoretically-valid operations, the schema may accept the operation, but deterministic gold extraction may emit it only when the required evidence is actually present in the selected record.

## Difference From Strict Signature Reconstruction

The old T-box proposal asked for a full `signature_after` that exactly reconstructed the historical post-repair constraint signature. That target remains available for strict historical diagnostics such as exact signature agreement or signature Jaccard. It is intentionally demoted because many causal schema repairs are easier to identify at the operation or value-delta level than as a complete post-edit signature.

The taxonomy patch task asks for:

- whether the schema edit is causal, coincidental, or unclear;
- the target property and constraint family when visible;
- one or more repair operations with taxonomy codes;
- concrete added and removed values only when the historical delta exposes them;
- a short evidence-based rationale and provenance.

The task does not require the model to construct a full post-repair `signature_after`.
