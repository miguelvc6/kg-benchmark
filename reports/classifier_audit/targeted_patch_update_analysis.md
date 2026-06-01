# Targeted Post-Delta Classifier Patch Analysis

Generated from the current post-patch artifacts:

- `reports/classifier_stats.json`
- `reports/classifier_audit/transition_after_targeted_patch.json`
- `reports/benchmark_selection/dev_prompt_v1_seed_13.json`
- `reports/benchmark_selection/core_v1_seed_13.json`
- `reports/manual_audit/audit_phase_d_v1_seed_13.csv`
- `reports/manual_audit/audit_phase_d_v1_results.json`

The classifier stats artifact was built at `2026-05-27T09:39:12Z`.

## Executive Summary

The targeted patch fixed the main issues found in the post-delta manual audit cards without reopening the broad TypeB contamination problem.

Most importantly:

- `TypeB / LOCAL_FOCUS_QID` collapsed from `770` targeted pre-patch cases to `8` final Stage 4 cases.
- `760` former `LOCAL_FOCUS_QID` cases became `TypeA / TARGET_REQUIRED_CLAIM`.
- `2` former `FORMAT_VALUE_PRUNING` cases became `TypeC / UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED`.
- `58` former `MULTIPLICITY_NORMALIZATION` cases became `TypeC / UNKNOWN_MULTIPLICITY_ARTIFACT`.
- `17` former `LOCAL_TEXT_CONFIRMED` cases became `TypeA / FORMAT_NORMALIZATION`.
- `105` former literal `EXTERNAL_BY_ELIMINATION` cases became `TypeC / UNKNOWN_BAD_TARGET_OR_CONTEXT`.
- `61` former literal `EXTERNAL_BY_ELIMINATION` cases became `TypeA / FORMAT_NORMALIZATION`.

The old broad labels remain absent:

| Deprecated or disallowed label | Current count |
| --- | ---: |
| `TypeC / EXTERNAL` | 0 |
| `TypeB / LOCAL_TEXT` | 0 |
| `TypeB / LOCAL_FOCUS_PREREPAIR_PROPERTY` | 0 |

The final Stage 4 file contains `535,570` records and regenerated without classifier errors.

## Stage 4 Class Distribution

| Class | Count | Share |
| --- | ---: | ---: |
| `T_BOX` | 456,594 | 85.25% |
| `TypeA` | 47,128 | 8.80% |
| `TypeC` | 30,769 | 5.75% |
| `TypeB` | 1,079 | 0.20% |

TypeB is now very narrow. This is expected: the patch removed target-required-claim and bad self-link focus-QID cases from TypeB, and moved deterministic wrapper/URL/category normalizations out of `LOCAL_TEXT_CONFIRMED`.

## Stage 4 Subtype Distribution

| Subtype | Count | Share of Stage 4 |
| --- | ---: | ---: |
| `SCHEMA_UPDATE` | 200,840 | 37.50% |
| `COINCIDENTAL_SCHEMA_CHANGE` | 166,856 | 31.15% |
| `RELAXATION_SET_EXPANSION` | 86,875 | 16.22% |
| `EXTERNAL_BY_ELIMINATION` | 28,999 | 5.41% |
| `DELETE_AMBIGUOUS` | 16,739 | 3.13% |
| `FORMAT_NORMALIZATION` | 11,830 | 2.21% |
| `SET_MEMBERSHIP_REJECTION` | 8,971 | 1.67% |
| `FORMAT_VALUE_PRUNING` | 4,930 | 0.92% |
| `SELF_LINK_REJECTION` | 3,237 | 0.60% |
| `RESTRICTION_SET_CONTRACTION` | 2,023 | 0.38% |
| `UNKNOWN_SELECTION_AMBIGUOUS` | 1,474 | 0.28% |
| `TARGET_REQUIRED_CLAIM` | 760 | 0.14% |
| `LOCAL_SELECTION_CONFIRMED` | 656 | 0.12% |
| `MULTIPLICITY_NORMALIZATION` | 330 | 0.06% |
| `REJECTION_FORMAT_INVALID` | 329 | 0.06% |
| `LOCAL_FOCUS_NON_TARGET_PROPERTY` | 304 | 0.06% |
| `UNKNOWN_MULTIPLICITY_ARTIFACT` | 137 | 0.03% |
| `UNKNOWN_BAD_TARGET_OR_CONTEXT` | 108 | 0.02% |
| `LOCAL_TEXT_CONFIRMED` | 88 | 0.02% |
| `UNKNOWN_INCOMPLETE_LOCAL_CONTEXT` | 49 | 0.01% |
| `LOCAL_MIXED` | 21 | <0.01% |
| `LOCAL_FOCUS_QID` | 8 | <0.01% |
| `LOCAL_NEIGHBOR_IDS` | 2 | <0.01% |
| `LOGICAL` | 2 | <0.01% |
| `UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED` | 2 | <0.01% |

The new small diagnostic categories are useful because they isolate exactly the suspicious patterns found during manual inspection:

- retained format-pruning targets that could not be verified;
- bad-target or context-suspect format/self-link direction cases;
- multiplicity changes under non-cardinality reports.

## Targeted Transition Report

The transition report scanned all `535,570` Stage 4 records and reclassified `65,628` targeted records from high-risk post-delta labels.

### Format Value Pruning

Old targeted count: `4,932`

| New label | Count |
| --- | ---: |
| `TypeA / FORMAT_VALUE_PRUNING` | 4,930 |
| `TypeC / UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED` | 2 |

This directly addresses the issue where `FORMAT_VALUE_PRUNING` could be emitted despite `retained_pass_regex=false`. In the regenerated Stage 4, a full scan found:

- `FORMAT_VALUE_PRUNING` cases: `4,930`
- `retained_pass_regex=false` among those cases: `0`

The two unsafe cases are now diagnostic TypeC.

### Focus-QID Local Evidence

Old targeted `TypeB / LOCAL_FOCUS_QID` count: `770`

| New label | Count |
| --- | ---: |
| `TypeA / TARGET_REQUIRED_CLAIM` | 760 |
| `TypeB / LOCAL_FOCUS_QID` | 8 |
| `TypeC / UNKNOWN_BAD_TARGET_OR_CONTEXT` | 2 |

This is the most important label-quality improvement in the patch. The audit revealed that many focus-QID additions were not local-evidence repairs. They were target-required-claim repairs where the rule and case identity determine the target. Those cases are now TypeA.

The two bad self-link or suspect focus-QID cases are no longer main-score TypeB.

### Local Text Confirmed

Old targeted `TypeB / LOCAL_TEXT_CONFIRMED` count: `105`

| New label | Count |
| --- | ---: |
| `TypeB / LOCAL_TEXT_CONFIRMED` | 88 |
| `TypeA / FORMAT_NORMALIZATION` | 17 |

This catches deterministic wrapper, URL, category, or identifier normalizations before local matching. The remaining `LOCAL_TEXT_CONFIRMED` cases are therefore less likely to be deterministic string cleanup mislabeled as local evidence.

### Multiplicity Normalization

Old targeted `TypeA / MULTIPLICITY_NORMALIZATION` count: `388`

| New label | Count |
| --- | ---: |
| `TypeA / MULTIPLICITY_NORMALIZATION` | 330 |
| `TypeC / UNKNOWN_MULTIPLICITY_ARTIFACT` | 58 |

Multiplicity normalization is now main-score TypeA only under cardinality or duplicate-related reports. Multiplicity changes under unrelated reports are diagnostic artifacts.

### Literal External-by-Elimination

Old targeted literal `TypeC / EXTERNAL_BY_ELIMINATION` count: `12,142`

| New label | Count |
| --- | ---: |
| `TypeC / EXTERNAL_BY_ELIMINATION` | 11,976 |
| `TypeC / UNKNOWN_BAD_TARGET_OR_CONTEXT` | 105 |
| `TypeA / FORMAT_NORMALIZATION` | 61 |

This shows two refinements:

- deterministic format direction/wrapper cases are no longer external-by-elimination;
- regex-invalid target directions become diagnostic bad-target/context cases instead of ordinary external cases.

### QID External-by-Elimination

Old targeted QID `TypeC / EXTERNAL_BY_ELIMINATION` count: `17,020`

| New label | Count |
| --- | ---: |
| `TypeC / EXTERNAL_BY_ELIMINATION` | 17,020 |

The QID external-by-elimination slice is unchanged by this targeted patch, which is appropriate because this patch focused on format direction, focus-QID target-required claims, self-link bad targets, and multiplicity/report consistency.

## Decision Trace Counts

| Decision branch | Count |
| --- | ---: |
| `external_by_elimination` | 28,999 |
| `delete_refined` | 17,525 |
| `format_normalization` | 11,616 |
| `set_membership_rejection` | 8,906 |
| `format_value_pruning` | 4,930 |
| `self_link_rejection` | 2,845 |
| `unknown_selection_ambiguous` | 1,460 |
| `target_required_claim` | 760 |
| `local_selection_confirmed` | 656 |
| `local_match` | 423 |
| `multiplicity_normalization` | 330 |
| `rule_deterministic` | 216 |
| `unknown_multiplicity_artifact` | 137 |
| `unknown_bad_target_or_context` | 108 |
| `pre_repair_target_only_not_local` | 63 |
| `unknown_format_pruning_retained_unverified` | 2 |

The new decision trace entries make the high-risk cases easier to audit. In particular, `target_required_claim`, `unknown_bad_target_or_context`, and `unknown_format_pruning_retained_unverified` isolate the exact remaining risk categories raised by manual review.

## Phase C Manifest Effects

### Dev Manifest

The dev manifest selected `600` cases:

| Category | Count |
| --- | ---: |
| Main score | 486 |
| Diagnostic | 114 |
| A-box | 360 |
| T-box | 240 |

Class distribution:

| Class | Count |
| --- | ---: |
| `T_BOX` | 240 |
| `TypeB` | 130 |
| `TypeC` | 120 |
| `TypeA` | 110 |

Hard validation passed:

- selected ids unique: true
- main + diagnostic equals selected: true
- dev/core case overlap: `0`
- dev/core T-box revision overlap: `0`
- dev/core A-box group overlap: `0`
- unknown or low-confidence in main score: `0`
- diagnostic subtypes in main score: `0`
- max T-box per revision: `3`
- max A-box per qid/property: `1`

The only underfilled dev quota remains:

| Stratum | Quota | Selected |
| --- | ---: | ---: |
| `DEV_TBOX_RESTRICTION_SET_CONTRACTION` | 40 | 6 |

### Core Manifest

The core manifest selected `4,800` cases:

| Category | Count |
| --- | ---: |
| Main score | 4,008 |
| Diagnostic | 792 |

Class distribution:

| Class | Count |
| --- | ---: |
| `TypeA` | 1,945 |
| `TypeC` | 1,135 |
| `T_BOX` | 1,032 |
| `TypeB` | 688 |

Hard validation passed:

- selected ids unique: true
- main + diagnostic equals selected: true
- core/dev case overlap: `0`
- core/dev T-box revision overlap: `0`
- core/dev A-box group overlap: `0`
- unknown or low-confidence in main score: `0`
- diagnostic subtypes in main score: `0`
- max T-box per revision: `10`
- max A-box per qid/property: `1`

Core selected subtypes:

| Subtype | Count |
| --- | ---: |
| `EXTERNAL_BY_ELIMINATION` | 1,135 |
| `SCHEMA_UPDATE` | 812 |
| `DELETE_AMBIGUOUS` | 586 |
| `LOCAL_SELECTION_CONFIRMED` | 514 |
| `FORMAT_NORMALIZATION` | 440 |
| `COINCIDENTAL_SCHEMA_CHANGE` | 206 |
| `FORMAT_VALUE_PRUNING` | 207 |
| `REJECTION_FORMAT_INVALID` | 258 |
| `SET_MEMBERSHIP_REJECTION` | 251 |
| `SELF_LINK_REJECTION` | 112 |
| `LOCAL_TEXT_CONFIRMED` | 78 |
| `LOCAL_FOCUS_NON_TARGET_PROPERTY` | 70 |
| `TARGET_REQUIRED_CLAIM` | 49 |
| `MULTIPLICITY_NORMALIZATION` | 40 |
| `LOCAL_MIXED` | 18 |
| `RELAXATION_SET_EXPANSION` | 14 |
| `LOCAL_FOCUS_QID` | 6 |
| `LOGICAL` | 2 |
| `LOCAL_NEIGHBOR_IDS` | 2 |

Core underfilled quotas are now more informative:

| Stratum | Quota | Selected | Interpretation |
| --- | ---: | ---: | --- |
| `TypeA_REJECTION_RULE_INVALID` | 20 | 0 | Current classifier does not emit this as a populated refined stratum. |
| `TypeA_LOGICAL` | 40 | 2 | Rare after the deterministic subtypes were split out. |
| `TypeB_LOCAL_TEXT_CONFIRMED` | 520 | 78 | Independent local text remains scarce after wrapper/URL/category normalization moved to TypeA. |
| `TypeB_LOCAL_TEXT_DERIVED` | 100 | 0 | Derived local text is not currently populated. |
| `TypeB_LOCAL_FOCUS_QID` | 80 | 6 | Most focus-QID cases are now target-required-claim TypeA or diagnostics. |
| `TypeB_LOCAL_MIXED` | 38 | 18 | Rare under stricter local-source handling. |
| `TBOX_RELAXATION_SET_EXPANSION` | 650 | 14 | Limited by T-box grouping/caps and dev exclusion. |
| `TBOX_RESTRICTION_SET_CONTRACTION` | 250 | 0 | Limited by T-box grouping/caps and dev exclusion. |
| `TBOX_SCHEMA_UPDATE` | 600 | 562 | Slightly under target after caps/exclusion. |
| `TBOX_COINCIDENTAL_SCHEMA_CHANGE` | 300 | 206 | Under target after caps/exclusion. |

The most important Phase C effect is that TypeB remains narrow while hard validation still passes.

## Phase D Audit Sample

The regenerated Phase D audit sample contains `450` rows.

Class distribution:

| Class | Count |
| --- | ---: |
| `TypeA` | 174 |
| `TypeC` | 134 |
| `TypeB` | 81 |
| `T_BOX` | 61 |

Main/diagnostic split:

| Slice | Count |
| --- | ---: |
| Main score | 363 |
| Diagnostic only | 87 |

Confidence split:

| Confidence | Count |
| --- | ---: |
| `high` | 205 |
| `medium` | 158 |
| `low` | 87 |

Popularity is still balanced:

| Bucket | Count |
| --- | ---: |
| `mid` | 155 |
| `head` | 149 |
| `tail` | 146 |

Audit stratum distribution:

| Stratum | Count |
| --- | ---: |
| `TypeC_EXTERNAL_BY_ELIMINATION_QID_TRUTH` | 51 |
| `TypeC_EXTERNAL_BY_ELIMINATION_LITERAL_TRUTH` | 46 |
| `TypeB_LOCAL_TEXT_CONFIRMED` | 40 |
| `TypeA_FORMAT_NORMALIZATION` | 36 |
| `TypeB_LOCAL_SELECTION_CONFIRMED` | 35 |
| `TypeA_FORMAT_VALUE_PRUNING` | 35 |
| `TypeA_REJECTION_FORMAT_INVALID` | 33 |
| `TypeA_DELETE_AMBIGUOUS` | 30 |
| `TBOX_SCHEMA_UPDATE` | 21 |
| `TBOX_DIRECTIONAL_RELAXATION_OR_RESTRICTION` | 20 |
| `TypeA_SELF_LINK_REJECTION` | 20 |
| `TBOX_COINCIDENTAL_SCHEMA_CHANGE` | 20 |
| `TypeC_UNKNOWN_SELECTION_AMBIGUOUS` | 15 |
| `TypeC_UNKNOWN_BAD_TARGET_OR_CONTEXT` | 10 |
| `TypeA_TARGET_REQUIRED_CLAIM` | 10 |
| `TypeA_MULTIPLICITY_NORMALIZATION` | 10 |
| `TypeB_LOCAL_FOCUS_QID` | 6 |
| `TypeC_UNKNOWN_MULTIPLICITY_ARTIFACT` | 5 |
| `TypeC_UNKNOWN_OR_SPARSE_DIAGNOSTIC` | 5 |
| `TypeC_UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED` | 2 |

The audit sample underfilled two rare strata:

| Stratum | Requested | Selected |
| --- | ---: | ---: |
| `TypeC_UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED` | 5 | 2 |
| `TypeB_LOCAL_FOCUS_QID` | 10 | 6 |

This underfill is a good sign for the patch: both categories are intentionally rare after the fix. The audit still includes all available examples from these strata.

The audit results file currently reports:

- total rows: `450`
- unannotated rows: `450`
- annotation completeness: `0.0`

So the precision and keep/exclude metrics remain null until the human annotations are filled.

## Case-Card Improvements

The regenerated case cards now expose the delta-aware fields needed for manual review:

- `semantic_action`
- `added_unique_values`
- `removed_unique_values`
- `retained_unique_values`
- `classification_target_tokens`
- `classification_target_reason`
- explicit local source names
- `independent_of_target_property`
- supporting non-target property metadata where available, including `supporting_property_id`

The regenerated case-card directory contains refined strata such as:

- `TypeC_UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED`
- `TypeC_UNKNOWN_BAD_TARGET_OR_CONTEXT`
- `TypeA_TARGET_REQUIRED_CLAIM`

No stale broad audit strata appear in the current audit CSV.

## Interpretation

The patch made the classifier stricter in the right places:

1. `FORMAT_VALUE_PRUNING` is now safer because retained values must pass regex checks when regexes are available.
2. Mixed deterministic format normalizations are no longer ordinary external-by-elimination.
3. Target-required focus-QID repairs are TypeA rule-deterministic cases, not TypeB local-evidence cases.
4. Self-link repairs that add the focus QID are diagnostic bad-target/context cases.
5. Wrapper, category, and URL slug normalizations are caught before local matching.
6. Multiplicity normalization is main-score only when the report is cardinality or duplicate related.
7. Phone-like plus-prefixed literals are no longer treated as dates.
8. TypeB remains narrow and substantially cleaner.

The tradeoff is that local-evidence strata are now smaller. This affects quota fill, especially `LOCAL_TEXT_CONFIRMED` and `LOCAL_FOCUS_QID`, but this is preferable to inflating TypeB with deterministic or diagnostic repairs.

## Residual Risks

1. `LOCAL_TEXT_DERIVED` is still empty. If the benchmark needs a richer local-literal derivation slice, property-specific conservative derivation rules would be needed.
2. The T-box directional strata remain underfilled after caps and dev exclusion.
3. `TypeA_REJECTION_RULE_INVALID` remains unpopulated in core selection policy.
4. The new diagnostic strata are small. Their audit value is high, but rate estimates from them will be noisy.
5. The audit metrics are still uninformative until manual annotations are completed.

## Bottom Line

The targeted post-delta patch achieved its purpose. It removed the remaining audited false-positive pathways into TypeB and TypeA main-score labels, added diagnostics for unsafe format and bad-target cases, preserved hard manifest validation, regenerated Phase D artifacts, and made the case cards more useful for human inspection.
