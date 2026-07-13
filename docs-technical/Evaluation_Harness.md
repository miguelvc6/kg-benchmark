# Evaluation Harness

The benchmark evaluation entry point is `src/evaluate.py`.

## Inputs

- Stage 4 benchmark records
- Stage 3 world state
- normalized A-box proposal JSONL
- normalized T-box proposal JSONL
- optional normalized taxonomy-patch T-box proposal JSONL for the separate taxonomy-patch evaluator
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
- treats target values as an unordered multiset, so serialization order does not change exactness while duplicate counts remain meaningful
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

## Legacy Strict-Signature T-box Evaluation

Legacy strict-signature scoring remains available for exploratory artifact compatibility:

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

## Confirmatory Taxonomy-Patch T-box Evaluation

The taxonomy-patch task is a separate metric family implemented in
`guardian.tbox_taxonomy_patch_evaluator`. It compares normalized taxonomy-patch proposals against a versioned gold
JSONL and reports schema-decision, constraint-family, taxonomy-code, repair-operation, and value-delta metrics with
explicit applicability denominators.

The reasoning-floor runner and `kg-rescore-run` invoke this evaluator automatically when the run records
`tbox_task_version=tbox_taxonomy_patch_v1`. Each bundle writes
`tbox_taxonomy_patch_evaluation_traces.jsonl` and `tbox_taxonomy_patch_evaluation_summary.json`. The ordinary
`evaluation_summary.json` is A-box-only for these runs, so missing strict T-box proposals never become false failures.
The top-level run summary exposes the two task families separately and sets `combined_repair_success_score=false`.

Confirmatory scoring requires complete mechanically supported taxonomy gold for every selected T-box case. Selection
excludes unextractable cases and operations whose required deltas are not mined, currently `CLASS_HIERARCHY_ADD` and
`EXCEPTION_ADD`.

Strict-signature and taxonomy-patch scores measure different tasks and must remain separate in tables and registry
metadata.

## Summary Splits

The evaluator aggregates results by:

- class
- subtype
- track
- ablation bundle
- popularity bucket
- manifest-defined evaluation subset

Popularity buckets use the selection policy's explicit bucket or fixed score thresholds; they are not recomputed on the evaluated slice. `paper_subsets` contains separate `all_selected`, `main_score`, and `diagnostic` aggregates. Grouped summaries also expose metric applicability counts so track-specific fields such as `semantic_success`, `conversion_rate`, and `tokens_to_fix` can be interpreted against the right denominator.

Evaluation fails on duplicate proposal IDs, duplicate selected Stage 4 IDs, or selected IDs missing from Stage 4. Traces carry the selection stratum, analysis slice, confidence, and leakage-group key used by cluster-aware analysis.

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
