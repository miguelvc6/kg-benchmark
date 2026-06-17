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
| `logic_only` | `main_score` | 203/296 = 0.686 | 201/296 = 0.679 | 34/296 = 0.115 | 35/1455 = 0.048; coverage 0.720 |
| `logic_only` | `diagnostic` | 155/300 = 0.517 | 71/300 = 0.237 | 142/300 = 0.473 | 0/505 = 0.000; coverage 0.000 |
| `local_graph` | `all_core` | 414/596 = 0.695 | 296/596 = 0.497 | 183/596 = 0.307 | 34/1442 = 0.047; coverage 0.357 |
| `local_graph` | `main_score` | 226/296 = 0.764 | 202/296 = 0.682 | 49/296 = 0.166 | 34/1225 = 0.056; coverage 0.720 |
| `local_graph` | `diagnostic` | 188/300 = 0.627 | 94/300 = 0.313 | 134/300 = 0.447 | 0/217 = 0.000; coverage 0.000 |

## Additional Diagnostics

| Context | Constraint-family F1 | Repair-op F1 | Value-delta success |
| --- | ---: | ---: | ---: |
| `logic_only` | 515/1753 = 0.588 | 34/666 = 0.102 | 41/596 = 0.069 |
| `local_graph` | 589/1800 = 0.654 | 49/675 = 0.145 | 55/596 = 0.092 |

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
