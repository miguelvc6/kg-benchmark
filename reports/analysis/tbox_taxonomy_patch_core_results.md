# T-Box Taxonomy Patch Core Results

Date: 2026-06-17

## Scope

This report covers the full core T-box taxonomy-patch run approved in `reports/analysis/tbox_taxonomy_patch_core_rerun_decision.md`.

- Prompt version: `prompt_dev_v5_tbox_taxonomy_patch`
- Gold version: `tbox_taxonomy_patch_gold_core_v1`
- Model endpoint/model: `ollama` / `gpt-oss:120b`
- Core T-box rows: 596
- Context bundles: `logic_only`, `local_graph`
- Prompt count: 1192
- Output directory: `reports/prompt_dev/evaluation_prompt_dev_v5_tbox_taxonomy_patch_core_tbox_all_zero_shot/`
- Machine-readable report: `reports/analysis/tbox_taxonomy_patch_core_results.json`

## Run Status

| Context | T-box rows | Normalized | Parse errors | Request errors |
| --- | ---: | ---: | ---: | ---: |
| `logic_only` | 596 | 596 | 0 | 0 |
| `local_graph` | 596 | 596 | 0 | 0 |

Both matrices scored successfully, and each matrix wrote 596 raw model responses and 596 `t_box_taxonomy_patch_proposals.jsonl` rows.

## Headline Core Metrics

These are taxonomy-patch metrics. The strict `functional` and `audit` columns in the prompt-dev comparison table are legacy strict-signature diagnostics and are not the paper-facing taxonomy-patch score.

| Context | Subset | Family success | Schema-decision match | Taxonomy-code match | Value-delta F1 when applicable |
| --- | --- | ---: | ---: | ---: | ---: |
| `logic_only` | `all_core` | 358/596 = 0.601 | 272/596 = 0.456 | 176/596 = 0.295 | 35/1960 = 0.036; coverage 0.357 |
| `logic_only` | `taxonomy_main_score` | 203/296 = 0.686 | 201/296 = 0.679 | 34/296 = 0.115 | 35/1455 = 0.048; coverage 0.720 |
| `logic_only` | `taxonomy_diagnostic` | 155/300 = 0.517 | 71/300 = 0.237 | 142/300 = 0.473 | n/a; gold applicability 0.000 |
| `local_graph` | `all_core` | 414/596 = 0.695 | 296/596 = 0.497 | 183/596 = 0.307 | 34/1442 = 0.047; coverage 0.357 |
| `local_graph` | `taxonomy_main_score` | 226/296 = 0.764 | 202/296 = 0.682 | 49/296 = 0.166 | 34/1225 = 0.056; coverage 0.720 |
| `local_graph` | `taxonomy_diagnostic` | 188/300 = 0.627 | 94/300 = 0.313 | 134/300 = 0.447 | n/a; gold applicability 0.000 |

`taxonomy_main_score` is the T-box taxonomy gold subset intersected with the core manifest `main_score_case_ids`, not the full manifest-level `main_score_case_ids` list.

## Additional Diagnostics

| Context | Constraint-family F1 | Repair-op F1 | Value-delta success |
| --- | ---: | ---: | ---: |
| `logic_only` | 515/1753 = 0.588 | 34/666 = 0.102 | 41/596 = 0.069 |
| `local_graph` | 589/1800 = 0.654 | 49/675 = 0.145 | 55/596 = 0.092 |

## Value-Delta Diagnostics

False positives count rows where the model claims an added or removed value delta although gold has no value-delta applicability. Under-specification counts rows where gold has a value delta and the model gets the family-level signal but not the value delta.

| Context | Subset | Value-delta false-positive rate | Value-delta under-specification rate |
| --- | --- | ---: | ---: |
| `logic_only` | `all_core` | 43/383 = 0.112 | 84/213 = 0.394 |
| `logic_only` | `taxonomy_main_score` | 5/83 = 0.060 | 84/213 = 0.394 |
| `logic_only` | `taxonomy_diagnostic` | 38/300 = 0.127 | n/a |
| `local_graph` | `all_core` | 47/383 = 0.123 | 84/213 = 0.394 |
| `local_graph` | `taxonomy_main_score` | 2/83 = 0.024 | 84/213 = 0.394 |
| `local_graph` | `taxonomy_diagnostic` | 45/300 = 0.150 | n/a |

Diagnostic value-delta F1 is reported as n/a because the diagnostic subset has zero gold value-delta-applicable rows. The false-positive rate remains reportable on diagnostic rows because it measures value deltas claimed when gold value deltas are absent.

## Confusion Matrices

Full sparse aligned confusion matrices for schema decision, taxonomy code, repair operation, and qualifier property are serialized in `reports/analysis/tbox_taxonomy_patch_core_results.json`. Multi-repair rows use exact-overlap-first multiset alignment; unmatched gold labels use `__MISSING__`, unmatched predicted labels use `__EXTRA__`.

Schema-decision confusion matrix:

| Context | Gold decision | Predicted causal | Predicted no-causal | Predicted unclear |
| --- | --- | ---: | ---: | ---: |
| `logic_only` | `CAUSAL_SCHEMA_REPAIR` | 201 | 61 | 34 |
| `logic_only` | `NO_CAUSAL_SCHEMA_REPAIR` | 158 | 71 | 71 |
| `local_graph` | `CAUSAL_SCHEMA_REPAIR` | 202 | 67 | 27 |
| `local_graph` | `NO_CAUSAL_SCHEMA_REPAIR` | 166 | 94 | 40 |

Largest aligned taxonomy-code, repair-operation, and qualifier-property confusion cells:

| Context | Matrix | Gold label | Predicted label | Count |
| --- | --- | --- | --- | ---: |
| `logic_only` | `taxonomy_code` | `__NO_REPAIR__` | `__NO_REPAIR__` | 142 |
| `logic_only` | `taxonomy_code` | `__NO_REPAIR__` | `C_PLUS` | 50 |
| `logic_only` | `taxonomy_code` | `CQ_PLUS` | `__NO_REPAIR__` | 40 |
| `logic_only` | `repair_operation` | `__NO_REPAIR__` | `__NO_REPAIR__` | 142 |
| `logic_only` | `repair_operation` | `__NO_REPAIR__` | `CONSTRAINT_ADD` | 50 |
| `logic_only` | `repair_operation` | `CONSTRAINT_QUALIFIER_ADD` | `__NO_REPAIR__` | 40 |
| `logic_only` | `qualifier_property` | `__NO_QUALIFIER__` | `__NO_QUALIFIER__` | 368 |
| `logic_only` | `qualifier_property` | `P2308` | `__NO_QUALIFIER__` | 88 |
| `logic_only` | `qualifier_property` | `P2305` | `__NO_QUALIFIER__` | 41 |
| `local_graph` | `taxonomy_code` | `__NO_REPAIR__` | `__NO_REPAIR__` | 134 |
| `local_graph` | `taxonomy_code` | `__NO_REPAIR__` | `CQ_PLUS` | 56 |
| `local_graph` | `taxonomy_code` | `CQ_PLUS` | `CQ_PLUS` | 48 |
| `local_graph` | `repair_operation` | `__NO_REPAIR__` | `__NO_REPAIR__` | 134 |
| `local_graph` | `repair_operation` | `__NO_REPAIR__` | `CONSTRAINT_QUALIFIER_ADD` | 56 |
| `local_graph` | `repair_operation` | `CONSTRAINT_QUALIFIER_ADD` | `CONSTRAINT_QUALIFIER_ADD` | 48 |
| `local_graph` | `qualifier_property` | `__NO_QUALIFIER__` | `__NO_QUALIFIER__` | 367 |
| `local_graph` | `qualifier_property` | `P2308` | `__NO_QUALIFIER__` | 94 |
| `local_graph` | `qualifier_property` | `P2305` | `__NO_QUALIFIER__` | 41 |

## Out-of-Current-Gold Operation False Positives

Current gold repair operations are `CONSTRAINT_QUALIFIER_ADD`, `CONSTRAINT_QUALIFIER_REMOVE`, `CONSTRAINT_QUALIFIER_REPLACE`, and `OTHER_TBOX_UPDATE`. Predictions using operations outside that set are counted as out-of-current-gold false positives.

| Context | Operation | Predicted count | False-positive rate among predicted repairs |
| --- | --- | ---: | ---: |
| `logic_only` | overall out-of-current-gold | 260 | 260/360 = 0.722 |
| `logic_only` | `CLASS_HIERARCHY_ADD` | 1 | 1/360 = 0.003 |
| `logic_only` | `CONSTRAINT_ADD` | 77 | 77/360 = 0.214 |
| `logic_only` | `CONSTRAINT_DEPRECATE` | 62 | 62/360 = 0.172 |
| `logic_only` | `CONSTRAINT_REMOVE` | 84 | 84/360 = 0.233 |
| `logic_only` | `CONSTRAINT_TYPE_REPLACE` | 4 | 4/360 = 0.011 |
| `logic_only` | `EXCEPTION_ADD` | 32 | 32/360 = 0.089 |
| `local_graph` | overall out-of-current-gold | 244 | 244/369 = 0.661 |
| `local_graph` | `CLASS_HIERARCHY_ADD` | 1 | 1/369 = 0.003 |
| `local_graph` | `CONSTRAINT_ADD` | 21 | 21/369 = 0.057 |
| `local_graph` | `CONSTRAINT_DEPRECATE` | 94 | 94/369 = 0.255 |
| `local_graph` | `CONSTRAINT_REMOVE` | 86 | 86/369 = 0.233 |
| `local_graph` | `CONSTRAINT_TYPE_REPLACE` | 5 | 5/369 = 0.014 |
| `local_graph` | `EXCEPTION_ADD` | 37 | 37/369 = 0.100 |

## Macro Averages

Macro averages are computed over per-property and per-T-box-revision group rates. Value-delta F1 macro averages exclude groups with zero gold value-delta applicability.

| Context | Grouping | Groups | Family success | Schema-decision match | Taxonomy-code match | Value-delta F1 | Value-delta FP | Value-delta under-spec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logic_only` | Property | 99 | 0.622 | 0.428 | 0.325 | 0.035 | 0.112 | 0.431 |
| `logic_only` | T-box revision | 120 | 0.641 | 0.424 | 0.299 | 0.033 | 0.113 | 0.436 |
| `local_graph` | Property | 99 | 0.710 | 0.536 | 0.382 | 0.032 | 0.134 | 0.364 |
| `local_graph` | T-box revision | 120 | 0.740 | 0.541 | 0.373 | 0.030 | 0.132 | 0.354 |

## Subset Audit

The manifest has 3818 top-level `main_score_case_ids` and 982 top-level `diagnostic_case_ids` across all tracks. The T-box taxonomy gold set has 596 case IDs. The evaluator subset previously displayed as `main_score` contains 296 rows and exactly equals the intersection of T-box taxonomy gold rows with manifest `main_score_case_ids`; it does not equal the full manifest-level main-score list. The report therefore uses `taxonomy_main_score` and `taxonomy_diagnostic` labels for the T-box-only subset views while preserving the underlying evaluator key for artifact compatibility.

## Prediction Distribution

| Context | Schema decisions |
| --- | --- |
| `logic_only` | `CAUSAL_SCHEMA_REPAIR`: 359; `NO_CAUSAL_SCHEMA_REPAIR`: 132; `UNCLEAR_SCHEMA_EVIDENCE`: 105 |
| `local_graph` | `CAUSAL_SCHEMA_REPAIR`: 368; `NO_CAUSAL_SCHEMA_REPAIR`: 161; `UNCLEAR_SCHEMA_EVIDENCE`: 67 |

| Context | Predicted repair operations |
| --- | --- |
| `logic_only` | `CONSTRAINT_QUALIFIER_ADD`: 98; `CONSTRAINT_REMOVE`: 84; `CONSTRAINT_ADD`: 77; `CONSTRAINT_DEPRECATE`: 62; `EXCEPTION_ADD`: 32; `CONSTRAINT_TYPE_REPLACE`: 4; `CLASS_HIERARCHY_ADD`: 1; `CONSTRAINT_QUALIFIER_REMOVE`: 1; `OTHER_TBOX_UPDATE`: 1 |
| `local_graph` | `CONSTRAINT_QUALIFIER_ADD`: 123; `CONSTRAINT_DEPRECATE`: 94; `CONSTRAINT_REMOVE`: 86; `EXCEPTION_ADD`: 37; `CONSTRAINT_ADD`: 21; `CONSTRAINT_TYPE_REPLACE`: 5; `OTHER_TBOX_UPDATE`: 2; `CLASS_HIERARCHY_ADD`: 1 |

`OTHER_TBOX_UPDATE` does not dominate model predictions: 1 predicted repair in `logic_only` and 2 predicted repairs in `local_graph`.

## Leakage Scan

The structured scan checked only model-visible `system_prompt` and `user_prompt` text. Hidden benchmark fields and labels were absent: `repair_target`, `popularity`, `sitelinks_count`, `changed_constraint_types`, `target_constraint_is_changed`, `TypeA`, `TypeB`, `TypeC`, `truth_source`, `truth_tokens`, `selected_case_ids`, `case_annotations`, `selection_stratum`, `group_key`, `DEV_`, `CORE_`, and `historical_track`.

The token `classification` appears only as visible Wikidata text in labels or descriptions, such as sports classification wording, not as the hidden benchmark `classification` field.

## Interpretation

The local-graph context improves family-level and taxonomy-level success over logic-only on the full core T-box set. Value-delta extraction remains weak even when applicable, so value-delta metrics should be reported separately from family and schema-decision metrics.

This report does not compare old strict T-box `signature_after` scores to taxonomy-patch scores as the same task. Strict-signature exactness remains a diagnostic for full-signature reconstruction only.

## Artifact Notes

- `run_manifest.jsonl` includes expected missing `track_diagnosis` rows for repair-only matrices; proposal metrics count the 596 T-box proposal rows per matrix.
- All metric numerators, denominators, applicability coverage values, and rates for every emitted T-box metric are serialized in `reports/analysis/tbox_taxonomy_patch_core_results.json`.
- A-box evaluation is outside this full-core T-box-only run and remains governed by the existing A-box task.
