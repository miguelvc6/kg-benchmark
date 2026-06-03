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

Stored Wikidata format regexes that are not valid Python `re` patterns are treated as non-matching during regression checks instead of aborting evaluation. This keeps benchmark scoring deterministic when Wikidata regex syntax uses features outside Python's regex engine.

## T-box Evaluation

Current T-box scoring:

- compares normalized `signature_after` against the historical `constraint_delta.signature_after`
- keeps exact historical agreement strict: exact action match plus exact normalized `signature_after` match
- derives a family-level semantic signal from the historical T-box delta when possible
- checks executability against the target property and constraint family
- requires auditability completeness for acceptance:
  - non-empty rationale
  - usable provenance
  - proposal-level uncertainty
- computes:
  - functional success
  - exact historical agreement
  - semantic success, now defined as family-level T-box compatibility rather than literal action-label equality
  - semantic-family success as an explicit companion metric
  - provenance completeness
  - auditability completeness
  - conversion rate
  - tokens-to-fix
  - token usage
  - T-box proxy metrics such as literal/exact action match, changed-constraint-type hit, signature overlap, and whether the proposal admits current values when applicable

Family-level T-box compatibility requires:

- the proposal targets one of the historical changed constraint families
- when the evaluator can identify a specific historical target constraint family, the proposal must target that same family
- the proposal action maps to the same semantic family as the historical reform
- the proposed `signature_after` is directionally compatible with that family when the historical `signature_before` makes direction inferable

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
