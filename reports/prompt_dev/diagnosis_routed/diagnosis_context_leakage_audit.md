# Diagnosis Context Leakage Audit

Verdict: `BLOCKER_FOUND_AND_PATCHED`

The v1 diagnosis-only holdout is not a clean measurement of repair-locus diagnosis. The diagnosis renderer reused the general prompt context path, and that path can branch on the hidden historical track before the model sees the prompt.

## Blocking Finding

- `src/guardian/reasoning.py:_t_box_pre_reform_constraints_payload` branches on `record["track"]`.
- `src/guardian/reasoning.py:_pruned_constraints_payload` calls that helper for `logic_only` and `local_graph`.
- `build_track_diagnosis_prompt_bundle` previously reused the same context builder, so T-box diagnosis prompts could get pre-reform `signature_before` constraints or compact T-box inventory while A-box prompts got normal pruned constraints.

This is model-visible structural conditioning on the gold track. It is a blocker for interpreting the failed v1 diagnosis gate as a pure model/prompt failure.

## Patch

- Added explicit diagnosis-only bundles: `diagnosis_minimal`, `diagnosis_logic_neutral`, `diagnosis_local_neutral`.
- Added `--diagnosis-context-bundles` to prompt-dev so diagnosis-only matrices can use those bundles without changing repair proposal context bundles.
- The diagnosis-neutral path never reads `repair_target.constraint_delta`, never branches on `record["track"]`, strips forbidden fields recursively, and suppresses target-property L1/L3 local graph leakage for every case.
- Oracle repair prompts keep their existing context path and T-box temporal policy.

## Checks

| Check | Result |
| --- | --- |
| Raw `repair_`/`reform_` IDs in model-visible diagnosis prompts | Guarded by neutral IDs and tests |
| `classification`, `repair_target`, `persistence_check`, `popularity`, `track` exposed | Stripped in new diagnosis-neutral branch |
| `signature_before` only for T-box diagnosis cases | Removed from diagnosis-neutral branch |
| Current target-property L1/L3 leakage | Suppressed for all diagnosis-neutral cases |
| Oracle repair context behavior | Unchanged |

Next required measurement is a diagnosis-only rerun on the same 96-case holdout using the three diagnosis-neutral bundles. No routed canary should run unless that gate passes.
