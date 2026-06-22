# T-box Value-Delta False Positive Analysis

Source: `reports/prompt_dev/few_shot/evaluation_static_v1_core_tbox_canary/`

No additional T-box inference was run. The static support set was not modified.

## Summary

- `logic_only`: 10 FP cases; metric rate 0.159.
- `local_graph`: 14 FP cases; metric rate 0.222.
- Cross-context overlap: 7 shared FP cases, 3 logic-only-only, 7 local-graph-only.
- Final verdict: `DO_NOT_USE_TBOX_FEW_SHOT`.

## Breakdown

### logic_only

- By predicted schema_decision: `{'CAUSAL_SCHEMA_REPAIR': 10}`
- By gold schema_decision: `{'NO_CAUSAL_SCHEMA_REPAIR': 6, 'CAUSAL_SCHEMA_REPAIR': 4}`
- By evidence_level: `{'VALUE_DELTA_VISIBLE': 10}`
- By taxonomy_code: `{'CQ_PLUS': 10}`
- Added vs removed value items: `{'added_values': 61}`
- qualifier_property_id: `{'P2308': 6, '__NULL__': 4}`
- Prompt visibility: `{'visible_only_in_static_examples': 61}`
- Support example pattern matches: `{'tbox_taxonomy_cq_plus': 6}`
- Plausible-but-not-gold review: 0/10 (0.0%) FP cases had at least one false-positive value visible in the held-out input case; 0/10 (0.0%) had at least one value not visible anywhere in the prompt.

### local_graph

- By predicted schema_decision: `{'CAUSAL_SCHEMA_REPAIR': 14}`
- By gold schema_decision: `{'NO_CAUSAL_SCHEMA_REPAIR': 9, 'CAUSAL_SCHEMA_REPAIR': 5}`
- By evidence_level: `{'VALUE_DELTA_VISIBLE': 14}`
- By taxonomy_code: `{'CQ_PLUS': 14}`
- Added vs removed value items: `{'added_values': 56}`
- qualifier_property_id: `{'P2308': 9, '__NULL__': 4, 'P2302': 1}`
- Prompt visibility: `{'not_visible_in_prompt': 1, 'visible_only_in_static_examples': 55}`
- Support example pattern matches: `{'tbox_taxonomy_cq_plus': 9}`
- Local-graph-specific cause counts: `{'unsupported value hallucination despite local graph': 1, 'over-causal schema decision on gold no-repair row': 9, 'missing qualifier_property_id on value-delta repair': 4}`
- Plausible-but-not-gold review: 0/14 (0.0%) FP cases had at least one false-positive value visible in the held-out input case; 1/14 (7.1%) had at least one value not visible anywhere in the prompt.

## Interpretation

- False positives are dominated by `CAUSAL_SCHEMA_REPAIR` predictions on gold `NO_CAUSAL_SCHEMA_REPAIR` rows, meaning the model often turns a visible report token or constraint family into a concrete value delta even when the gold taxonomy says no value delta is supported.
- `local_graph` increases the number of FP cases, but the values still mostly come from the static examples rather than the held-out input case; one local-graph FP value is not visible anywhere in the prompt.
- The static examples cluster the output shape around CQ/value-delta repairs, especially the CQ_PLUS support example. This appears to bias T-box few-shot toward adding concrete qualifier values.
- The evaluator is not merely overcounting plausible non-gold values: in this canary, nearly all FP values are example-visible only, and one is not prompt-visible at all. The false-positive increase is therefore a modeling/support-bias issue, not just a reporting artifact.

## Verdict

`DO_NOT_USE_TBOX_FEW_SHOT`

Do not proceed to full T-box few-shot. If revisited, redesign T-box static support in dev only, with examples that discourage value-delta claims unless qualifier polarity and values are explicitly gold-supported.
