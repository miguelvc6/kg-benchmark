# T-Box Taxonomy Patch Metric Design

Date: 2026-06-17

## Purpose

The taxonomy-patch task replaces strict T-box `signature_after` reconstruction as the paper-facing T-box repair-reasoning task. Strict `signature_after` reconstruction remains available only as a historical diagnostic because it asks the model to reconstruct a full post-repair constraint signature, while the taxonomy-patch task asks for a bounded schema-repair decision and Ferranti-derived repair operation.

A-box repair remains a separate task. This metric family does not change the A-box prompt, parser, evaluator, or score interpretation.

## Ferranti-Derived Mapping

| Taxonomy code | Operation | Role in this benchmark |
| --- | --- | --- |
| `C_MINUS` | `CONSTRAINT_REMOVE` | Remove a property-constraint statement or family. |
| `C_D` | `CONSTRAINT_DEPRECATE` | Deprecate or deactivate a constraint statement. |
| `C_PLUS` | `CONSTRAINT_ADD` | Engineering extension for added constraint statements. |
| `C_REPLACE` | `CONSTRAINT_TYPE_REPLACE` | Engineering extension for replacing one constraint family with another. |
| `CQ_PLUS` | `CONSTRAINT_QUALIFIER_ADD` | Add a qualifier value to a constraint definition. |
| `CQ_MINUS` | `CONSTRAINT_QUALIFIER_REMOVE` | Remove a qualifier value from a constraint definition. |
| `CQ_REPLACE` | `CONSTRAINT_QUALIFIER_REPLACE` | Replace qualifier values on the same qualifier property. |
| `SUBCLASS_PLUS` | `CLASS_HIERARCHY_ADD` | Schema-level class hierarchy repair; schema-supported but not mined from current gold artifacts. |
| `E_PLUS` | `EXCEPTION_ADD` | Exception repair; schema-supported but not mined from current gold artifacts. |
| `OTHER` | `OTHER_TBOX_UPDATE` | Schema-level update outside the listed operations. |

Gold extraction currently supports qualifier add/remove/replace and `OTHER_TBOX_UPDATE` for selected records. It does not force unsupported class-hierarchy or exception cases into synthetic labels.

## Metric Hierarchy

Paper-facing T-box headline metrics:

- `tbox_patch_family_level_success`
- `tbox_patch_schema_decision_match_rate`
- `tbox_patch_taxonomy_code_exact_match_rate`
- `tbox_patch_value_delta_f1_when_applicable`

Strict-signature diagnostics:

- `strict_exact_signature_match_rate`
- `strict_exact_historical_agreement_rate`
- `signature_after_jaccard`

The strict diagnostics are not comparable to the taxonomy-patch metrics as identical tasks. They remain useful for understanding historical full-signature reconstruction behavior, not for the new headline T-box repair score.

## Subsets

| Subset | Definition | Reporting role |
| --- | --- | --- |
| `all_core` | All selected core T-box rows with taxonomy gold. | Full T-box core accounting. |
| `main_score` | Gold rows where `schema_decision = CAUSAL_SCHEMA_REPAIR`. | Main taxonomy-patch repair reasoning subset. |
| `diagnostic` | Gold rows where no causal schema repair is present. | Abstention/no-causal diagnostic subset. |

## Metric Definitions

Every metric emitted by `tbox_taxonomy_patch_evaluation_summary.json` and `reports/analysis/tbox_taxonomy_patch_core_results.json` uses this shape:

```json
{
  "numerator": 0,
  "applicable_denominator": 0,
  "total_tbox_rows": 0,
  "applicability_coverage": 0.0,
  "rate": 0.0
}
```

`applicable_denominator` is the denominator for that metric. `total_tbox_rows` is the row count for the subset. `applicability_coverage` is row-bounded coverage where the metric applies to only some rows; micro-averaged item metrics may have item-level denominators larger than the row count.

| Metric | Numerator | Applicable denominator | Applicability |
| --- | --- | --- | --- |
| `tbox_patch_parse_rate` | Parsed T-box outputs | T-box rows | All rows |
| `tbox_patch_contract_valid_rate` | Contract-valid T-box outputs | T-box rows | All rows |
| `tbox_patch_parse_error_rate` | Parse-error T-box outputs | T-box rows | All rows |
| `tbox_patch_target_pid_match_rate` | Predictions matching focus PID | Parsed/contract rows | Parsed rows |
| `tbox_patch_primary_constraint_family_match_rate` | Primary target family matches gold | Parsed/contract rows | Parsed rows |
| `tbox_patch_any_changed_family_hit_rate` | Any predicted changed family hits gold family | Parsed/contract rows | Parsed rows |
| `tbox_patch_schema_decision_match_rate` | Schema decision matches gold | Parsed/contract rows | Parsed rows |
| `tbox_patch_no_causal_schema_repair_match_rate` | Correct `NO_CAUSAL_SCHEMA_REPAIR` decision | Gold no-causal rows | Diagnostic rows with no-causal gold |
| `tbox_patch_unclear_schema_evidence_match_rate` | Correct `UNCLEAR_SCHEMA_EVIDENCE` decision | Gold unclear rows | Rows with unclear gold, if any |
| `tbox_patch_repair_op_exact_match_rate` | Exact operation-set match | Parsed/contract rows | Parsed rows |
| `tbox_patch_taxonomy_code_exact_match_rate` | Exact taxonomy-code set match | Parsed/contract rows | Parsed rows |
| `tbox_patch_qualifier_property_match_rate` | Edited qualifier property matches gold | Rows with qualifier-property gold | Rows where qualifier property is applicable |
| `tbox_patch_evidence_level_exact_match_rate` | Evidence level matches gold | Parsed/contract rows | Parsed rows |
| `tbox_patch_value_delta_claimed_when_gold_absent_rate` | Predicted value delta when gold lacks one | Gold rows without value delta | Diagnostic false-positive rate |
| `tbox_patch_family_only_when_value_delta_gold_present_rate` | Family-only prediction despite value-delta gold | Gold rows with value delta | Diagnostic under-specification rate |
| `tbox_patch_family_level_success` | Family-level success cases | Parsed/contract rows | Parsed rows |
| `tbox_patch_decision_level_success` | Family plus schema-decision success cases | Parsed/contract rows | Parsed rows |
| `tbox_patch_taxonomy_level_success` | Family plus decision plus taxonomy success cases | Parsed/contract rows | Parsed rows |
| `tbox_patch_value_delta_success` | Complete value-delta success cases | Parsed/contract rows | Parsed rows |
| `tbox_patch_constraint_family_precision` | Constraint-family true positives | Predicted family items | Micro item metric |
| `tbox_patch_constraint_family_recall` | Constraint-family true positives | Gold family items | Micro item metric |
| `tbox_patch_constraint_family_f1` | Constraint-family F1 numerator | Precision/recall item denominator | Micro item metric |
| `tbox_patch_repair_op_precision` | Repair-op true positives | Predicted op items | Micro item metric |
| `tbox_patch_repair_op_recall` | Repair-op true positives | Gold op items | Micro item metric |
| `tbox_patch_repair_op_f1` | Repair-op F1 numerator | Precision/recall item denominator | Micro item metric |
| `tbox_patch_added_values_precision` | Added-value true positives | Predicted added values | Micro item metric |
| `tbox_patch_added_values_recall` | Added-value true positives | Gold added values | Micro item metric |
| `tbox_patch_added_values_f1` | Added-value F1 numerator | Precision/recall item denominator | Micro item metric |
| `tbox_patch_removed_values_precision` | Removed-value true positives | Predicted removed values | Micro item metric |
| `tbox_patch_removed_values_recall` | Removed-value true positives | Gold removed values | Micro item metric |
| `tbox_patch_removed_values_f1` | Removed-value F1 numerator | Precision/recall item denominator | Micro item metric |
| `tbox_patch_value_delta_f1_when_applicable` | Added/removed value true positives | Value-delta item denominator | Gold rows with value-delta applicability |

Full numerator, denominator, and applicability values for every metric and subset are serialized in `reports/analysis/tbox_taxonomy_patch_core_results.json`.
