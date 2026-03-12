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

Runtime side effect:

- a SQLite index is built next to `data/03_world_state.json` to support keyed world-state lookup during classification

CLI options currently implemented:

- `--sample`
- `--self-test`
- `--verbose`
- `--quiet`
- `--no-progress`

## Decision Order

Classification is deterministic and runs in a fixed order.

1. If the world-state entry is missing, the record is classified as low-confidence `TypeC/EXTERNAL`.
2. T-box records are classified separately as class `T_BOX`, with a subtype inferred from the constraint delta.
3. Delete actions are labeled `TypeA` with subtype `REJECTION`.
4. Rule-deterministic cases are labeled `TypeA`.
5. Repairs whose truth is available in local context are labeled `TypeB`.
6. Remaining applicable cases fall back to `TypeC`.

## T-box Classification

The current code does not collapse T-box records into `UNKNOWN`.

Instead it uses `ConstraintDiffer` to emit:

- class `T_BOX`
- a subtype such as `RELAXATION_RANGE_WIDENED`, `RESTRICTION_RANGE_NARROWED`, `RELAXATION_SET_EXPANSION`, `RESTRICTION_SET_CONTRACTION`, `SCHEMA_UPDATE`, or `COINCIDENTAL_SCHEMA_CHANGE`
- a decision trace specific to the schema-change path

Confidence is adjusted by subtype:

- `high` for clear widening or contraction signals
- `medium` for generic schema updates
- `low` for coincidental or weakly causal schema changes

## Missing Context Behavior

If a world-state entry cannot be found for a repair id, the classifier still emits a Stage 4 record.

In that case it defaults to:

- class `TypeC`
- subtype `EXTERNAL`
- confidence `low`
- error counter `missing_world_state`

This is a fallback, not evidence that the case is truly external.

## Local Evidence Policy

The classifier reconstructs the target property in its pre-repair state to avoid post-repair leakage.

It searches for truth evidence in:

- focus-node labels and descriptions
- pre-repair focus-node property values
- one-hop neighbor identifiers
- one-hop neighbor labels and descriptions

Matching rules:

- QID truths use exact identifier matching.
- Literal truths use normalized text matching.
- Date matches must preserve sufficient precision; a year-only mention does not satisfy a full-date truth.
- post-repair values of the target property are explicitly excluded from local evidence to avoid leakage

## Rule-Deterministic Policy

The current implementation treats a case as Type A only when the rule itself uniquely determines the repair.

Supported deterministic families in the current code:

- format constraints
- one-of constraints with a singleton allowed set
- range constraints when the repair matches a rule-implied boundary

Constraint information is read primarily from `world_state.L4_constraints` and selectively supplemented by the repair metadata when needed.

## Output Contract

Each Stage 4 record written by the current code contains:

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

Each `classification` block includes at least:

```json
{
  "classification": {
    "class": "TypeB",
    "subtype": "LOCAL_NEIGHBOR_IDS",
    "decision_trace": [],
    "rationale": "",
    "constraint_types": [],
    "diagnostics": {},
    "local_subtype": null
  }
}
```

The classifier preserves Stage 2 semantics; it appends classification results rather than rewriting the source repair event.

## Determinism Notes

- Classification does not perform live web calls.
- Given identical Stage 2 input, Stage 3 input, and classifier code, output is deterministic.
- The same script can emit both LEAN and FULL records in one run.

## Current Verification Status

Confirmed directly from the repository:

- `uv run python src/classifier.py --self-test` passes
- Stage 4 output is now consumed by the proposal validator, evaluation harness, and reasoning-floor runner
