# Evaluation Harness

The benchmark evaluation entry point is [evaluate.py](/home/mvazquez/kg-benchmark/src/evaluate.py).

## Inputs

- Stage 4 benchmark records
- Stage 3 world state
- normalized A-box proposal JSONL
- normalized T-box proposal JSONL
- optional reasoning-floor run manifest

## Outputs

- per-case evaluation traces
- aggregate summary JSON

## A-box Evaluation

Current A-box scoring:

- reconstructs the pre-repair focus-node target property from benchmark artifacts
- applies normalized proposal ops in memory
- checks executability against the benchmark target ids
- compares the resulting target property against the historical repair target
- computes:
  - functional success
  - exact historical agreement
  - information preservation
  - provenance completeness
  - token usage
- preserves reserved null fields for:
  - conversion rate
  - tokens-to-fix

The first-wave regression check uses currently supported local constraint families represented in the stored constraint metadata.

## T-box Evaluation

Current T-box scoring:

- compares normalized `signature_after` against the historical `constraint_delta.signature_after`
- separately scores exact reform match and semantic reform match
- checks executability against the target property and constraint family
- computes:
  - functional success
  - exact historical agreement
  - provenance completeness
  - token usage

## Summary Splits

The evaluator aggregates results by:

- class
- subtype
- track
- ablation bundle
- popularity bucket
