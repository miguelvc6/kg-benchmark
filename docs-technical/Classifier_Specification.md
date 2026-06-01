# Classifier Specification

This document describes the current behavior of `src/classifier.py`.

## Inputs and Outputs

Inputs:

- `data/02_wikidata_repairs.json` or `.jsonl`
- `data/03_world_state.json`
- optional `data/00_entity_popularity.json`

Outputs:

- `data/04_classified_benchmark.jsonl`
- `data/04_classified_benchmark_full.jsonl`
- `reports/classifier_stats.json`

The CLI also supports explicit path overrides:

- `--repairs-path`
- `--world-state-path`
- `--popularity-path`
- `--out-path`
- `--out-full-path`
- `--no-full-output`
- `--stats-path`

Runtime side effect:

- a SQLite index is built next to the world-state JSON file to support keyed world-state lookup during classification.

## Decision Order

Classification is deterministic and runs in this order:

1. Missing world state -> `TypeC/UNKNOWN_MISSING_WORLD_STATE`.
2. T-box records -> class `T_BOX`, with subtype inferred from the constraint delta.
3. A-box value-delta summary is computed and attached to diagnostics.
4. A-box deletes and deterministic delta repairs -> refined TypeA subtype.
5. Missing historical truth -> `TypeC/UNKNOWN_MISSING_TRUTH`.
6. Current-value-only truth fallback -> `TypeC/UNKNOWN_CURRENT_VALUE_FALLBACK`.
7. Conservative local-text derivations, currently including `P8726` statutory-instrument ids from independent focus/neighbor text, -> `TypeB/LOCAL_TEXT_DERIVED`.
8. Local truth match over delta-specific target tokens -> refined `TypeB/LOCAL_*` subtype or a diagnostic unknown if the match is not sufficient.
9. Remaining cases -> `TypeC/EXTERNAL_BY_ELIMINATION` or an explicit `UNKNOWN_*` diagnostic subtype.

## Type C Semantics

Unqualified `TypeC/EXTERNAL` is no longer emitted by the redesigned classifier. The supported subtypes are:

- `EXTERNAL_BY_ELIMINATION`: historical truth exists, but supported rule and local checks did not find it.
- `UNKNOWN_MISSING_WORLD_STATE`: frozen local context is unavailable.
- `UNKNOWN_MISSING_TRUTH`: no usable historical repair target tokens are available.
- `UNKNOWN_CURRENT_VALUE_FALLBACK`: only a current 2025/2026 value fallback exists.
- `UNKNOWN_INCOMPLETE_LOCAL_CONTEXT`: local context is too sparse for a negative local-evidence claim.
- `UNKNOWN_SELECTION_AMBIGUOUS`: a subset repair retained old values but lacks independent local/rule evidence for why the removed values should be removed.
- `UNKNOWN_MULTIPLICITY_ARTIFACT`: unique values are unchanged and multiplicity increased or otherwise looks like a reconstruction artifact.
- `UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED`: a format subset repair removed a format-failing value, but retained values could not be verified against the available regex.
- `UNKNOWN_BAD_TARGET_OR_CONTEXT`: the repaired target conflicts with the report shape, such as a single-value report followed by multiple new values or a bad-direction format/self-link repair.
- `UNKNOWN_FOCUS_QID_DOMAIN_REASONING`: the new value equals the focus QID, but no target-required-claim or other explicit identity rule makes the focus id sufficient.
- `EXTERNAL_CONFIRMED`: reserved for manual audit or retrieval-confirmed cases.

Current-value fallback fields are diagnostics only. They do not feed rule or local matching.

## Local Evidence Policy

The classifier reconstructs the target property in its pre-repair state to avoid post-repair leakage.

It searches for delta-specific target evidence in:

- focus-node labels and descriptions
- focus aliases when available
- synthetic pre-repair target-property QIDs, explicitly marked as `FOCUS_PREREPAIR_TARGET_PROPERTY_QID`
- synthetic pre-repair target-property literals, explicitly marked as `FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL`
- non-target focus-node property values
- the focus entity id as `FOCUS_QID`
- one-hop neighbor identifiers
- one-hop neighbor labels and descriptions
- labels and descriptions for ids that are locally referenced

Leakage controls:

- current/post-repair target-property values from `L1_ego_node.properties[target_pid]` are excluded
- target-property `L3_neighborhood` edges are excluded
- QID/PID truth tokens require exact id matches, not label matches
- synthetic pre-repair target-property values are not merged into generic `FOCUS_TEXT`
- retained old target-property values alone cannot make a subset repair TypeB
- focus-QID availability alone is not enough for TypeB when the property assertion requires domain-specific reasoning
- `LOCAL_TEXT_DERIVED` requires an explicit deterministic derivation from independent local text, not from pre-repair target-property values

Literal matching is conservative:

- full ISO dates require exact full-date boundary matches
- literals shorter than four normalized characters require exact field equality
- longer literals require exact normalized field equality or token-boundary matches
- every local match records match kind and source in the decision trace

## Delete Policy

Deletes are no longer automatically high-confidence logical rejections.

Supported delete subtypes:

- `REJECTION_RULE_INVALID`: a supported rule family identifies the value as invalid.
- `REJECTION_FORMAT_INVALID`: the violation report is format-related, or the deleted value fails a relevant format rule.
- `SELF_LINK_REJECTION`: the removed value is the focus QID under a self-link violation.
- `SET_MEMBERSHIP_REJECTION`: parsed one-of/none-of constraints prove the removed value is invalid.
- `TARGET_REQUIRED_CLAIM`: a target-required-claim report deterministically requires the focus entity as the target value.
- `DELETE_AMBIGUOUS`: not enough evidence to classify the deletion confidently.

Single-value and unique-value conflicts are routed to `DELETE_AMBIGUOUS` or `UNKNOWN_SELECTION_AMBIGUOUS` unless a stronger rule or independent local-selection signal exists.

## Rule-Deterministic Policy

The classifier treats a case as Type A only when the rule itself uniquely determines the repair.

Supported deterministic families:

- simple format normalizations, such as whitespace stripping, trailing punctuation stripping, case normalization, and trivial date literal normalization
- deterministic literal canonicalizations such as trailing slash removal and `SCHEMBL` prefix stripping
- mixed deterministic format normalization, where one changed old/new pair is normalized while retained values are ignored
- format value pruning where the removed value is the reported format violation or fails the relevant regex and retained values pass available regex checks
- duplicate multiplicity decreases with unchanged unique values under a cardinality or duplicate-related report
- self-link rejection
- target-required-claim focus-QID creation
- one-of constraints with a singleton allowed set
- numeric range boundaries using `P2313` minimum quantity/value and `P2312` maximum quantity/value
- date range boundaries using `P2310` minimum date and `P2311` maximum date

Non-deterministic format updates are not high-confidence Type A repairs.

## Delta-Aware A-box Policy

Every A-box classification diagnostics block contains:

- `value_change_summary`
- `classification_target_tokens`

The classifier uses changed values rather than the full post-repair `new_value` list:

- `CREATE_FROM_MISSING` and `ADD_SUPERSET` use added values.
- `DELETE_SUBSET` and `DELETE_TO_MISSING` use removed values or explicit selection evidence.
- `REPLACE_1_TO_1` uses the replacement relation.
- multiplicity-only changes are TypeA duplicate normalization only when multiplicity decreases under a cardinality or duplicate-related report; unrelated-report multiplicity changes are diagnostic unknown artifacts.
- single-value reports followed by multiple created or added target values are diagnostic bad-target/context cases, not clean `EXTERNAL_BY_ELIMINATION`.

Retained values that were already present before repair are not treated as independent local evidence.

## T-Box Classification

T-box records are classified separately as class `T_BOX`.

The classifier emits subtypes such as:

- `RELAXATION_RANGE_WIDENED`
- `RESTRICTION_RANGE_NARROWED`
- `RELAXATION_SET_EXPANSION`
- `RESTRICTION_SET_CONTRACTION`
- `SCHEMA_UPDATE`
- `COINCIDENTAL_SCHEMA_CHANGE`
- `UNKNOWN_TBOX_CAUSALITY`

Range analysis separates numeric and date bounds. Type and value-type constraints compare `P2308` type/value class and `P2309` relation qualifiers rather than treating them as one-of values.

T-box main-score labels require a selected violation report candidate, a mapped violation family, and either a changed matching constraint family or type-compatible value/property/language/scope overlap with changed qualifier values. The classifier evaluates all report-name candidates from normalized type, raw type, type lists, violation name, and message, then selects the strongest mapping rather than trusting only the first field.

Qualifier-value-only edits contribute to `changed_constraint_qids_all`, but only family-relevant qualifier properties count as semantic added/removed values for causality and polarity. For example, `P1793` is semantic for format constraints, `P424` is semantic for label/description language constraints, `P2308`/`P2309` are semantic for type/value-type families, and `P2316` constraint status is ignored as metadata. Exact-family-only matches are weaker when the report names concrete QIDs, PIDs, or languages: without compatible semantic overlap, these cases are downgraded to `SCHEMA_UPDATE` or `UNKNOWN_TBOX_CAUSALITY` rather than high-confidence directional labels.

The target constraint is the changed constraint that carries the semantic repair evidence. It may differ from the report-mapped constraint when a conservative related-family mapping is used, such as subject type reports aligning with changed allowed-entity-types constraints. Directional public subtypes remain coarse for compatibility; use active `directional_subtype_precise` and `analysis_slice_precise` for polarity-specific analysis only when the final public subtype is directional. Non-directional `SCHEMA_UPDATE` records use `main_tbox_schema_update`; any computed but inactive direction appears only in `potential_directional_*` fields. Directional labels require known polarity for the constraint family: allowed-set expansion is relaxation, allowed-set contraction is restriction, forbidden-set expansion is restriction, forbidden-set contraction is relaxation, and required-set expansion is restriction.

`Q21510862` is treated as the Wikidata symmetric constraint. A-box self-link repairs remain report-level `SELF_LINK_REJECTION` decisions and do not claim `Q21510862` as the decision constraint.

## Audit Reporting

`scripts/classifier_audit_report.py` produces Phase B audit reports:

- counts by class, subtype, confidence, truth source, decision branch, local subtype, and repair track
- Type C subtype metrics
- current-value fallback counts
- Type A delete and format counts
- old/new transition matrices with example case ids

The console entry point is `kg-classifier-audit-report`.

## Output Contract

Each Stage 4 record contains:

- `id`, `qid`, `property`, `track`
- `information_type`
- `labels_en`
- `violation_context`
- `repair_target`
- `persistence_check`
- `popularity`
- `context_ref`
- `classification`
- `build`

The `classification` block also carries decision-family metadata for later audit and analysis:

- `decision_constraint_type_qid`
- `decision_constraint_type_label`
- `decision_constraint_source`
- `classification_rule_family`
- `classification_rule_subfamily`

`constraint_family` remains available for compatibility, but it is not necessarily the decisive classifier rule. Use the decision-family fields for rule-family analysis.

Each `classification` block includes:

```json
{
  "class": "TypeB",
  "subtype": "LOCAL_NEIGHBOR_IDS",
  "confidence": "high",
  "decision_trace": [],
  "rationale": "",
  "constraint_types": [],
  "diagnostics": {},
  "local_subtype": "LOCAL_NEIGHBOR_IDS"
}
```

## Current Verification Status

Confirmed directly from the repository:

- `UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/classifier.py --self-test` passes
- `UV_PROJECT_ENVIRONMENT=.venv-wsl uv run --extra dev python -m pytest tests/test_classifier.py` passes
