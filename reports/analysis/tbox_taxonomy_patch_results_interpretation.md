# T-Box Taxonomy Patch Results Interpretation

Date: 2026-06-18

## Executive Summary

The T-box taxonomy-patch task is operationally stable and produces usable structured outputs on the full core T-box set. The full run evaluated 596 core T-box records under two contexts, `logic_only` and `local_graph`, for 1192 total prompts. Every response normalized successfully, with 0 parse errors and 0 request errors.

The main empirical result is that the model is substantially better at identifying the affected constraint family than at reconstructing the precise taxonomy operation or concrete value delta. Adding local graph context improves every headline metric, but the improvement is strongest at the family and repair-operation levels, not at exact value-delta recovery.

The recommended paper-facing interpretation is:

- Use taxonomy-patch metrics as the headline T-box repair-reasoning result.
- Treat strict `signature_after` reconstruction as a diagnostic only.
- Report family/schema-decision performance separately from taxonomy-code and value-delta performance.
- Prefer `local_graph` over `logic_only` for the main T-box taxonomy-patch condition.

## Run Validity

| Check | Result |
| --- | --- |
| Full core T-box rows | 596 |
| Total prompts | 1192 |
| Normalized outputs | 1192/1192 |
| Parse errors | 0 |
| Request errors | 0 |
| Gold version | `tbox_taxonomy_patch_gold_core_v1` |
| Prompt version | `prompt_dev_v5_tbox_taxonomy_patch` |
| Task version | `tbox_taxonomy_patch_v1` |

The result is technically clean enough for reporting: failures are metric-level misses, not infrastructure, parser, or request failures.

## Headline Interpretation

| Context | Family success | Schema decision | Taxonomy code | Value-delta F1 |
| --- | ---: | ---: | ---: | ---: |
| `logic_only`, all core | 0.601 | 0.456 | 0.295 | 0.036 |
| `local_graph`, all core | 0.695 | 0.497 | 0.307 | 0.047 |
| `logic_only`, causal main-score subset | 0.686 | 0.679 | 0.115 | 0.048 |
| `local_graph`, causal main-score subset | 0.764 | 0.682 | 0.166 | 0.056 |

The `local_graph` context improves all-core family success by about 9.4 percentage points and main-score family success by about 7.8 points. This is the clearest positive result: local neighborhood and labels help the model identify which constraint family is implicated.

Schema-decision matching improves only modestly in all-core scoring, from 0.456 to 0.497. On the causal main-score subset, both contexts are nearly tied at about 0.68. This suggests that the main difficulty is not only deciding whether a schema repair is causal, but distinguishing causal repairs from no-causal and unclear cases across the full mixed T-box set.

Taxonomy-code matching remains low: 0.295 all-core for `logic_only`, 0.307 all-core for `local_graph`, and only 0.166 on the local-graph main-score subset. This means the model often finds the right general schema area but does not reliably choose the same fine-grained operation as the deterministic gold extractor.

Value-delta F1 is weak in both contexts. Even where value deltas are applicable, the model rarely recovers the exact added/removed values. This should be reported as a separate fine-grained extraction limitation rather than blended into the family-level repair result.

## What The Model Can Do

The strongest capability is coarse schema localization. With `local_graph`, the model reaches:

- 0.695 family-level success on all core T-box rows.
- 0.764 family-level success on causal main-score rows.
- 0.654 constraint-family F1 on all core rows.
- 0.747 constraint-family F1 on causal main-score rows.

This supports the claim that the model can often identify the relevant property-constraint family from visible benchmark evidence, especially when local graph context is available.

The model also avoids collapsing to the fallback operation. `OTHER_TBOX_UPDATE` appears in only 1 logic-only predicted repair and 2 local-graph predicted repairs. This is important because it shows that the taxonomy contract is not being satisfied by a degenerate fallback strategy.

## Where The Model Struggles

The model struggles with operation granularity. The gold core distribution is dominated by qualifier updates:

- `CONSTRAINT_QUALIFIER_ADD`: 138
- `CONSTRAINT_QUALIFIER_REMOVE`: 31
- `CONSTRAINT_QUALIFIER_REPLACE`: 64
- `OTHER_TBOX_UPDATE`: 73

The predictions instead spread mass over operations that are schema-plausible but often not gold-equivalent:

- `CONSTRAINT_REMOVE`
- `CONSTRAINT_ADD`
- `CONSTRAINT_DEPRECATE`
- `EXCEPTION_ADD`

This explains the gap between family success and taxonomy-code success. The model often sees that the schema is implicated, but it tends to propose broad schema edits rather than the exact qualifier-level edit represented in historical gold.

The model also struggles with concrete values. Value-delta F1 is 0.036 for `logic_only` and 0.047 for `local_graph` on all core rows. On the causal main-score subset, the rates are only 0.048 and 0.056. This is the clearest evidence that exact value-delta prediction should not be the headline result.

## Diagnostic Subset

The diagnostic subset contains 300 rows where the gold decision is not a causal schema repair. Performance is much weaker on schema decision:

- `logic_only`: 0.237
- `local_graph`: 0.313

The model often predicts causal edits even when gold marks the case as no causal schema repair or unclear. This matters because over-attributing causality can inflate apparent schema-repair confidence if only family overlap is reported. The diagnostic subset should remain visible in the paper or appendix.

## Recommended Reporting

Use `local_graph` as the primary T-box taxonomy-patch result and present `logic_only` as an ablation.

Recommended main table columns:

- Family success
- Schema-decision match
- Taxonomy-code match
- Value-delta F1 when applicable
- Parse/request error rates

Recommended interpretation text:

> Local graph context improves T-box schema localization, raising all-core family-level success from 0.601 to 0.695 and causal-subset family-level success from 0.686 to 0.764. However, fine-grained taxonomy-code and value-delta recovery remain difficult, indicating that the model more reliably identifies the implicated constraint family than the exact historical qualifier edit.

Avoid saying that the model "repairs T-box constraints" without qualification. The result supports a narrower claim: it often identifies the correct schema locus, but exact operation and value-delta reconstruction are still weak.

## Paper-Safe Claims

Supported:

- The taxonomy-patch task is parse-stable on full core T-box evaluation.
- Local graph context improves T-box family localization.
- Fine-grained taxonomy operation selection is substantially harder than family localization.
- Exact value-delta recovery remains a bottleneck.
- Strict `signature_after` reconstruction should remain diagnostic rather than headline.

Not supported:

- The model reliably reconstructs complete T-box repairs.
- Strict-signature and taxonomy-patch scores are directly comparable.
- Value-level T-box edits are solved by the current prompt.
- No-causal/unclear schema cases are reliably separated from causal repairs.

## Bottom Line

The migration succeeds as an evaluation design: it replaces an overly strict full-signature reconstruction target with a more interpretable taxonomy-patch task and yields stable, complete core artifacts. The empirical result is mixed but useful. The model has meaningful signal for schema-family localization, especially with local graph context, but exact operation and value-delta prediction remain weak and should be presented as harder diagnostic layers rather than as the primary success criterion.
