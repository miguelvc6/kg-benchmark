# Evaluation Harness

The benchmark evaluation entry point is [evaluate.py](/home/mvazquez/kg-benchmark/src/evaluate.py).

## Inputs

- Stage 4 benchmark records
- Stage 3 world state
- normalized A-box proposal JSONL
- normalized T-box proposal JSONL
- optional normalized track-diagnosis JSONL
- optional reasoning-floor run manifest
- optional benchmark selection manifest containing `selected_case_ids`

## Outputs

- per-case evaluation traces
- aggregate summary JSON

The evaluator can restrict scoring to a frozen subset either by explicit `case_ids` or by `--selection-manifest`. When both are supplied, it evaluates the intersection.

## A-box Evaluation

Current A-box scoring:

- reconstructs the pre-repair focus-node target property from benchmark artifacts
- applies normalized proposal ops in memory
- checks executability against the benchmark target ids
- compares the resulting target property against the historical repair target
- requires auditability completeness for acceptance:
  - non-empty rationale
  - usable provenance
  - proposal-level uncertainty
- computes:
  - functional success
  - exact historical agreement
  - information preservation
  - provenance completeness
  - auditability completeness
  - conversion rate
  - tokens-to-fix
  - token usage

The first-wave regression check uses currently supported local constraint families represented in the stored constraint metadata.

## T-box Evaluation

Current T-box scoring:

- compares normalized `signature_after` against the historical `constraint_delta.signature_after`
- separately scores exact reform match and semantic reform match
- checks executability against the target property and constraint family
- requires auditability completeness for acceptance:
  - non-empty rationale
  - usable provenance
  - proposal-level uncertainty
- computes:
  - functional success
  - exact historical agreement
  - semantic success
  - provenance completeness
  - auditability completeness
  - conversion rate
  - tokens-to-fix
  - token usage
  - T-box proxy metrics such as exact action match and signature overlap

## Summary Splits

The evaluator aggregates results by:

- class
- subtype
- track
- ablation bundle
- popularity bucket

Grouped summaries now also expose metric applicability counts so track-specific fields such as `semantic_success`, `conversion_rate`, and `tokens_to_fix` can be interpreted against the right denominator.

## Track-Diagnosis Evaluation

The evaluator also supports a separate diagnosis task:

- input artifact: normalized track-diagnosis JSONL
- historical target: benchmark `track`
- supported predictions: `A_BOX`, `T_BOX`, `AMBIGUOUS`

Current trace fields include:

- predicted track
- historical track
- exact-track-match
- ambiguous-prediction flag
- diagnosis token usage when supplied by the run manifest

Current summaries expose diagnosis accuracy alongside the other grouped aggregates.
