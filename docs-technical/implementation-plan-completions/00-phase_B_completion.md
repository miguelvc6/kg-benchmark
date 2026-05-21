# Phase B Completion: Repository Audit and Classifier Redesign

**Project:** WikidataRepairEval 1.0  
**Phase:** B — repository audit and classifier redesign  
**Status:** Completed  
**Date:** 2026-05-21

## B1/B10 — Audit snapshots and transition matrix

The pre-redesign full classifier snapshot in `reports/classifier_stats.json` records 535,570 classified records:

| Slice | Count |
|---|---:|
| T-box | 456,594 |
| A-box total | 78,976 |
| TypeA | 47,216 |
| TypeB | 13,492 |
| TypeC / EXTERNAL | 18,268 |
| TypeA / REJECTION deletes | 17,525 |
| TypeA / LOGICAL | 29,691 |

Phase B audit artifacts were added under `reports/classifier_audit/`:

- `phase_b_baseline_eval_subset_counts.json`
- `phase_b_transition_eval_subset.json`

The transition report compares the first 1,000 existing Stage 4 records against the redesigned classifier using the same world-state entries. Key transitions in that subset:

| Transition | Count |
|---|---:|
| `TypeC/EXTERNAL -> TypeC/EXTERNAL_BY_ELIMINATION` | 28 |
| `TypeA/REJECTION -> TypeA/DELETE_AMBIGUOUS` | 11 |
| `TypeA/REJECTION -> TypeA/REJECTION_FORMAT_INVALID` | 10 |
| `TypeA/LOGICAL -> TypeC/EXTERNAL_BY_ELIMINATION` | 33 |
| `TypeA/LOGICAL -> TypeB/LOCAL_TEXT` | 18 |

This subset is an audit smoke test, not a replacement for regenerating the full Stage 4 artifact before final experiments.

## B2/B3 — Type C split and current-value quarantine

The classifier no longer emits ordinary `TypeC/EXTERNAL` for fallback decisions. It emits:

- `EXTERNAL_BY_ELIMINATION`
- `UNKNOWN_MISSING_WORLD_STATE`
- `UNKNOWN_MISSING_TRUTH`
- `UNKNOWN_CURRENT_VALUE_FALLBACK`
- `UNKNOWN_INCOMPLETE_LOCAL_CONTEXT`

Historical truth extraction now prioritizes only:

1. `repair_target.new_value`
2. `repair_target.value`

Current-value fields are diagnostic fallbacks only. If a case needs `persistence_check.current_value_2026`, `violation_context.value_current_2026`, or corresponding 2025 fields, it is routed to `UNKNOWN_CURRENT_VALUE_FALLBACK` before rule or local matching.

## B4/B5 — Local evidence and matching

Local evidence now includes:

- synthetic pre-repair target-property values
- all non-target `L1_ego_node.properties`
- one-hop neighbor ids
- labels/descriptions for ids actually referenced in local properties or neighborhood edges

Leakage controls remain in place:

- target-property `L1` current values are excluded
- target-property `L3` current edges are excluded
- QID/PID truth tokens require exact id matches

Literal matching was tightened:

- short literals under four normalized characters require exact field equality
- longer literals use exact field equality or token-boundary matches
- full ISO dates require full-date matches
- all local matches record match kind and source

## B6/B7 — Delete and format refinement

Deletes are no longer automatically high-confidence `TypeA/REJECTION`.

Implemented delete subtypes:

- `REJECTION_RULE_INVALID`
- `REJECTION_FORMAT_INVALID`
- `DELETE_AMBIGUOUS`

`DELETE_SELECTION_LOCAL` and `DELETE_SELECTION_EXTERNAL` are reserved for later audit-informed refinements. Single-value and unique-value delete conflicts are downgraded to `DELETE_AMBIGUOUS` unless a stronger rule signal exists.

Format repairs are Type A only for deterministic normalizations such as whitespace stripping, trailing punctuation stripping, case normalization, and trivial date literal normalization. Non-deterministic format updates fall through to local, external-by-elimination, or unknown handling.

## B8 — Range and type/value-type handling

Range handling now separates:

| Constraint element | Property |
|---|---|
| Minimum quantity/value | `P2313` |
| Maximum quantity/value | `P2312` |
| Minimum date | `P2310` |
| Maximum date | `P2311` |

T-box type and value-type constraints now compare `P2308` and `P2309` qualifiers rather than treating them as one-of `P2305` sets.

## B9 — Tests

Added `tests/test_classifier.py`, covering:

1. post-repair target-property edge leakage prevention
2. non-target L1 QID local evidence
3. L2 label non-leakage for unreferenced QIDs
4. short literal substring safeguards
5. numeric range boundaries via `P2313`/`P2312`
6. date range boundaries via `P2310`/`P2311`
7. deterministic-only format Type A
8. missing truth -> `UNKNOWN_MISSING_TRUTH`
9. current-value fallback -> `UNKNOWN_CURRENT_VALUE_FALLBACK`
10. missing world state -> `UNKNOWN_MISSING_WORLD_STATE`
11. value-type T-box expansion over `P2308`
12. single-value delete not treated as high-confidence rejection

Verification:

```text
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/classifier.py --self-test
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run --extra dev python -m pytest tests/test_classifier.py
```

Both commands passed on 2026-05-21.

## Files changed

- `src/classifier.py`
- `tests/test_classifier.py`
- `scripts/classifier_audit_report.py`
- `schemas/04_classified_benchmark.schema.json`
- `docs-technical/Classifier_Specification.md`
- `docs-technical/00-implementation_plan.md`
- `docs-technical/README.md`
- `reports/classifier_audit/phase_b_baseline_eval_subset_counts.json`
- `reports/classifier_audit/phase_b_transition_eval_subset.json`
