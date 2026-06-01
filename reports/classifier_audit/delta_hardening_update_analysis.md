# Classifier Delta-Hardening Update Analysis

Generated from the post-update artifacts in this repository:

- `reports/classifier_stats.json`
- `reports/classifier_audit/transition_after_delta_hardening.json`
- `reports/benchmark_selection/dev_prompt_v1_seed_13.json`
- `reports/benchmark_selection/core_v1_seed_13.json`
- `reports/manual_audit/audit_phase_d_v1_seed_13.csv`
- `reports/manual_audit/audit_phase_d_v1_results.json`

The classifier stats artifact was built at `2026-05-26T14:28:31Z`.

## Executive Summary

The update substantially changed A-box classification behavior in the intended direction. The broad contaminated TypeB labels are no longer present in the regenerated Stage 4 output:

| Deprecated or disallowed label | New count |
| --- | ---: |
| `TypeC / EXTERNAL` | 0 |
| `TypeB / LOCAL_TEXT` | 0 |
| `TypeB / LOCAL_FOCUS_PREREPAIR_PROPERTY` | 0 |

The biggest effect is a sharp contraction of TypeB. After delta-hardening, TypeB is only `1,863` records, or `0.35%` of Stage 4. This is expected: most previous TypeB cases were retained old values, format pruning, self-link/set-membership deletes, or ambiguous selections rather than independently grounded local evidence.

The update also split formerly broad labels into more defensible deterministic or diagnostic subtypes:

- `FORMAT_NORMALIZATION`: `11,726`
- `FORMAT_VALUE_PRUNING`: `4,932`
- `SET_MEMBERSHIP_REJECTION`: `8,971`
- `SELF_LINK_REJECTION`: `3,237`
- `MULTIPLICITY_NORMALIZATION`: `388`
- `UNKNOWN_SELECTION_AMBIGUOUS`: `1,474`
- `UNKNOWN_MULTIPLICITY_ARTIFACT`: `79`

## Stage 4 Class Distribution

The regenerated canonical Stage 4 file contains `535,570` records.

| Class | Count | Share |
| --- | ---: | ---: |
| `T_BOX` | 456,594 | 85.25% |
| `TypeA` | 46,324 | 8.65% |
| `TypeC` | 30,789 | 5.75% |
| `TypeB` | 1,863 | 0.35% |

The class mix confirms that the update mostly moved contaminated local cases into TypeA deterministic repairs or TypeC diagnostic/external-by-elimination buckets. TypeB is now a narrow local-grounding class rather than a broad catch-all for values visible in the pre-repair target property.

## Stage 4 Subtype Distribution

| Subtype | Count | Share of Stage 4 |
| --- | ---: | ---: |
| `SCHEMA_UPDATE` | 200,840 | 37.50% |
| `COINCIDENTAL_SCHEMA_CHANGE` | 166,856 | 31.15% |
| `RELAXATION_SET_EXPANSION` | 86,875 | 16.22% |
| `EXTERNAL_BY_ELIMINATION` | 29,165 | 5.45% |
| `DELETE_AMBIGUOUS` | 16,739 | 3.13% |
| `FORMAT_NORMALIZATION` | 11,726 | 2.19% |
| `SET_MEMBERSHIP_REJECTION` | 8,971 | 1.68% |
| `FORMAT_VALUE_PRUNING` | 4,932 | 0.92% |
| `SELF_LINK_REJECTION` | 3,237 | 0.60% |
| `RESTRICTION_SET_CONTRACTION` | 2,023 | 0.38% |
| `UNKNOWN_SELECTION_AMBIGUOUS` | 1,474 | 0.28% |
| `LOCAL_FOCUS_QID` | 770 | 0.14% |
| `LOCAL_SELECTION_CONFIRMED` | 656 | 0.12% |
| `MULTIPLICITY_NORMALIZATION` | 388 | 0.07% |
| `REJECTION_FORMAT_INVALID` | 329 | 0.06% |
| `LOCAL_FOCUS_NON_TARGET_PROPERTY` | 308 | 0.06% |
| `LOCAL_TEXT_CONFIRMED` | 105 | 0.02% |
| `UNKNOWN_MULTIPLICITY_ARTIFACT` | 79 | 0.01% |
| `UNKNOWN_INCOMPLETE_LOCAL_CONTEXT` | 71 | 0.01% |
| `LOCAL_MIXED` | 22 | <0.01% |
| `LOCAL_NEIGHBOR_IDS` | 2 | <0.01% |
| `LOGICAL` | 2 | <0.01% |

The important qualitative result is not just the disappearance of old TypeB labels, but where those cases went. The new subtypes reveal that much of the earlier TypeB mass was actually rule-deterministic repair activity.

## Targeted Transition Analysis

The transition report scanned all `535,570` Stage 4 records and reclassified `78,589` records from the high-risk old buckets.

### `TypeA / REJECTION_FORMAT_INVALID`

Old count: `8,675`

| New label | Count | Share |
| --- | ---: | ---: |
| `TypeA / DELETE_AMBIGUOUS` | 8,346 | 96.2% |
| `TypeA / REJECTION_FORMAT_INVALID` | 329 | 3.8% |

This is the clearest validation of the format-delete fix. The previous classifier was assigning `REJECTION_FORMAT_INVALID` mostly because a property had a format constraint somewhere, not because the actual violation report was a format violation. After the update, only `329` cases remain format-invalid rejections.

### `TypeB / LOCAL_FOCUS_PREREPAIR_PROPERTY`

Old count: `12,541`

| New label | Count | Share |
| --- | ---: | ---: |
| `TypeA / SET_MEMBERSHIP_REJECTION` | 8,906 | 71.0% |
| `TypeA / SELF_LINK_REJECTION` | 2,837 | 22.6% |
| `TypeC / UNKNOWN_SELECTION_AMBIGUOUS` | 689 | 5.5% |
| `TypeA / MULTIPLICITY_NORMALIZATION` | 93 | 0.7% |
| `TypeC / UNKNOWN_MULTIPLICITY_ARTIFACT` | 16 | 0.1% |

This bucket was almost entirely contaminated by retained old target-property values. The new distribution shows that most cases are now explained as deterministic rule rejections, especially set-membership and self-link repairs. The remaining ambiguous selections are now diagnostic TypeC rather than TypeB.

### `TypeB / LOCAL_TEXT`

Old count: `9,546`

| New label | Count | Share |
| --- | ---: | ---: |
| `TypeA / FORMAT_VALUE_PRUNING` | 4,932 | 51.7% |
| `TypeA / FORMAT_NORMALIZATION` | 2,624 | 27.5% |
| `TypeC / UNKNOWN_SELECTION_AMBIGUOUS` | 784 | 8.2% |
| `TypeB / LOCAL_SELECTION_CONFIRMED` | 656 | 6.9% |
| `TypeA / MULTIPLICITY_NORMALIZATION` | 292 | 3.1% |
| `TypeB / LOCAL_TEXT_CONFIRMED` | 105 | 1.1% |
| `TypeC / UNKNOWN_INCOMPLETE_LOCAL_CONTEXT` | 71 | 0.7% |
| `TypeC / UNKNOWN_MULTIPLICITY_ARTIFACT` | 63 | 0.7% |
| Other TypeB local subtypes | 19 | 0.2% |

This is the expected result from separating pre-repair target-property literals from independent local text. Only `105` cases survive as confirmed local text. Most old `LOCAL_TEXT` cases become deterministic format normalization or format value pruning.

### `TypeC / EXTERNAL_BY_ELIMINATION` with literal truth

Old count: `21,217`

| New label | Count | Share |
| --- | ---: | ---: |
| `TypeC / EXTERNAL_BY_ELIMINATION` | 12,142 | 57.2% |
| `TypeA / FORMAT_NORMALIZATION` | 9,072 | 42.8% |
| `TypeA / MULTIPLICITY_NORMALIZATION` | 3 | <0.1% |

Nearly `43%` of literal TypeC cases were deterministic format normalizations. This directly addresses the observed `SCHEMBL... -> numeric id` and trailing slash cases.

### `TypeC / EXTERNAL_BY_ELIMINATION` with QID truth

Old count: `17,791`

| New label | Count | Share |
| --- | ---: | ---: |
| `TypeC / EXTERNAL_BY_ELIMINATION` | 17,020 | 95.7% |
| `TypeB / LOCAL_FOCUS_QID` | 770 | 4.3% |
| `TypeB / LOCAL_MIXED` | 1 | <0.1% |

Most QID TypeC cases remain external-by-elimination. A small but meaningful slice is now local because the added or created QID is the focus entity id itself.

### `TypeA / DELETE_AMBIGUOUS`

Old count: `8,816`

| New label | Count | Share |
| --- | ---: | ---: |
| `TypeA / DELETE_AMBIGUOUS` | 8,390 | 95.2% |
| `TypeA / SELF_LINK_REJECTION` | 392 | 4.4% |
| `TypeA / SET_MEMBERSHIP_REJECTION` | 34 | 0.4% |

Most delete-ambiguous cases remain conservative diagnostics. The update only promotes the small subset with clear self-link or set-membership evidence.

## Decision Trace

The classifier decision trace records the main new branches:

| Decision branch | Count |
| --- | ---: |
| `external_by_elimination` | 29,165 |
| `delete_refined` | 17,525 |
| `format_normalization` | 11,515 |
| `set_membership_rejection` | 8,906 |
| `format_value_pruning` | 4,932 |
| `self_link_rejection` | 2,845 |
| `unknown_selection_ambiguous` | 1,460 |
| `local_match` | 1,207 |
| `local_selection_confirmed` | 656 |
| `multiplicity_normalization` | 388 |
| `rule_deterministic` | 213 |
| `pre_repair_target_only_not_local` | 85 |
| `unknown_multiplicity_artifact` | 79 |

The `pre_repair_target_only_not_local` trace is particularly important. It confirms that the classifier is explicitly detecting cases where the only apparent evidence is the target property's pre-repair state and refusing to use that as local grounding.

## Phase C Manifest Effects

### Dev Manifest

The dev manifest selected `600` cases:

| Category | Count |
| --- | ---: |
| Main score | 488 |
| Diagnostic | 112 |
| A-box | 360 |
| T-box | 240 |

Class distribution:

| Class | Count |
| --- | ---: |
| `T_BOX` | 240 |
| `TypeB` | 130 |
| `TypeC` | 120 |
| `TypeA` | 110 |

Validation passed. Hard checks report:

- selected ids unique: true
- main + diagnostic equals selected: true
- dev/core case overlap: `0`
- dev/core T-box revision overlap: `0`
- dev/core A-box group overlap: `0`
- unknown or low-confidence in main score: `0`
- diagnostic subtypes in main score: `0`
- max T-box per revision: `3`
- max A-box per qid/property: `1`

There is one underfilled dev quota:

| Stratum | Quota | Selected |
| --- | ---: | ---: |
| `DEV_TBOX_RESTRICTION_SET_CONTRACTION` | 40 | 6 |

This reflects the stricter T-box grouping/cap constraints, not an A-box classifier problem.

### Core Manifest

The core manifest selected `4,800` cases:

| Category | Count |
| --- | ---: |
| Main score | 4,003 |
| Diagnostic | 797 |
| A-box | 3,768 |
| T-box | 1,032 |

Class distribution:

| Class | Count |
| --- | ---: |
| `TypeA` | 1,914 |
| `TypeC` | 1,145 |
| `T_BOX` | 1,032 |
| `TypeB` | 709 |

Validation passed. Hard checks report:

- selected ids unique: true
- main + diagnostic equals selected: true
- core/dev case overlap: `0`
- core/dev T-box revision overlap: `0`
- core/dev A-box group overlap: `0`
- unknown or low-confidence in main score: `0`
- diagnostic subtypes in main score: `0`
- max T-box per revision: `10`
- max A-box per qid/property: `1`

Core underfilled quotas are informative:

| Stratum | Quota | Selected | Interpretation |
| --- | ---: | ---: | --- |
| `TypeA_REJECTION_RULE_INVALID` | 20 | 0 | Not produced by the current refined classifier policy. |
| `TypeA_LOGICAL` | 40 | 2 | Rare after the update. |
| `TypeB_LOCAL_TEXT_CONFIRMED` | 520 | 99 | Independent local text evidence is much rarer than old broad `LOCAL_TEXT`. |
| `TypeB_LOCAL_TEXT_DERIVED` | 100 | 0 | Conservative derived-text rule is not currently producing selected cases. |
| `TypeB_LOCAL_MIXED` | 38 | 19 | Rare after source separation. |
| `TBOX_RELAXATION_SET_EXPANSION` | 650 | 14 | Constrained by T-box revision overlap/caps and dev exclusion. |
| `TBOX_RESTRICTION_SET_CONTRACTION` | 250 | 0 | Constrained by T-box revision overlap/caps and available groups. |
| `TBOX_SCHEMA_UPDATE` | 600 | 562 | Slightly under target after caps/exclusion. |
| `TBOX_COINCIDENTAL_SCHEMA_CHANGE` | 300 | 206 | Under target after caps/exclusion. |

The key A-box consequence is that confirmed TypeB local evidence is scarce. This is a positive signal for label quality: old TypeB abundance was mostly an artifact of using retained pre-repair values as evidence.

## Phase D Audit Sample

The regenerated Phase D audit sample contains `450` rows.

| Audit class | Count |
| --- | ---: |
| `TypeA` | 170 |
| `TypeC` | 130 |
| `TypeB` | 90 |
| `T_BOX` | 60 |

Main/diagnostic split:

| Slice | Count |
| --- | ---: |
| Main score | 360 |
| Diagnostic only | 90 |

Confidence split:

| Confidence | Count |
| --- | ---: |
| `medium` | 193 |
| `high` | 167 |
| `low` | 90 |

Audit stratum distribution:

| Stratum | Count |
| --- | ---: |
| `TypeC_EXTERNAL_BY_ELIMINATION_QID_TRUTH` | 50 |
| `TypeC_EXTERNAL_BY_ELIMINATION_LITERAL_TRUTH` | 50 |
| `TypeA_DELETE_AMBIGUOUS` | 40 |
| `TypeB_LOCAL_TEXT_CONFIRMED` | 40 |
| `TypeA_FORMAT_NORMALIZATION` | 35 |
| `TypeB_LOCAL_SELECTION_CONFIRMED` | 35 |
| `TypeA_FORMAT_VALUE_PRUNING` | 35 |
| `TypeA_REJECTION_FORMAT_INVALID` | 30 |
| `TBOX_DIRECTIONAL_RELAXATION_OR_RESTRICTION` | 20 |
| `TypeA_SELF_LINK_REJECTION` | 20 |
| `TBOX_SCHEMA_UPDATE` | 20 |
| `TBOX_COINCIDENTAL_SCHEMA_CHANGE` | 20 |
| `TypeC_UNKNOWN_SELECTION_AMBIGUOUS` | 15 |
| `TypeB_LOCAL_FOCUS_QID` | 15 |
| `TypeA_MULTIPLICITY_NORMALIZATION` | 10 |
| `TypeC_UNKNOWN_OR_SPARSE_DIAGNOSTIC` | 10 |
| `TypeC_UNKNOWN_MULTIPLICITY_ARTIFACT` | 5 |

The audit sample is balanced across popularity buckets:

| Bucket | Count |
| --- | ---: |
| `mid` | 152 |
| `tail` | 149 |
| `head` | 149 |

The audit summary currently reports:

- total rows: `450`
- unannotated rows: `450`
- annotation completeness: `0.0`

Therefore the metric rates in `audit_phase_d_v1_results.json` are currently null because no human annotations have been filled yet. The file is structurally ready for annotation, but it contains no empirical human precision estimates yet.

## Interpretation

The update appears to have fixed the main delta-awareness failure modes:

1. Retained old target-property values are no longer sufficient for TypeB.
2. Pre-repair target-property literals and QIDs are no longer treated as generic local text.
3. One-to-one deterministic literal canonicalizations are now TypeA format normalization.
4. Format-pruning subset repairs are now TypeA format value pruning.
5. Self-link and set-membership subset deletes are now deterministic TypeA when the removed value explains the violation.
6. Ambiguous selection and multiplicity artifacts are diagnostic, not main-score local evidence.
7. `REJECTION_FORMAT_INVALID` has contracted to actual format-delete cases.

The most important quantitative signal is the transition of old TypeB:

- Old `LOCAL_FOCUS_PREREPAIR_PROPERTY`: `12,541` cases, now mostly deterministic TypeA rule repairs.
- Old `LOCAL_TEXT`: `9,546` cases, now mostly deterministic TypeA format repairs.
- New confirmed local TypeB evidence is small:
  - `LOCAL_TEXT_CONFIRMED`: `105`
  - `LOCAL_SELECTION_CONFIRMED`: `656`
  - `LOCAL_FOCUS_QID`: `770`

This suggests the benchmark's local-evidence slice is now much cleaner but also much smaller. The core selector compensates by backfilling from other valid strata while keeping unknown, low-confidence, and diagnostic-only cases out of the main score.

## Residual Risks and Follow-Up Questions

1. `TypeB_LOCAL_TEXT_DERIVED` is currently not contributing to core. If derived local literals such as statutory instrument IDs are important, a conservative property-specific derivation rule could expand this slice.
2. T-box directional strata are heavily underfilled in core after caps and dev exclusion. This may be acceptable, but it means the core distribution is less close to the nominal T-box quotas than the A-box distribution.
3. `TypeA_REJECTION_RULE_INVALID` is not represented. If this stratum is still desired conceptually, it needs either classifier support or removal from the quota policy.
4. The audit sample is ready, but all human-dependent precision metrics remain unknown until annotation is completed.
5. The canonical `data/04_classified_benchmark.jsonl` was regenerated. The larger embedded full-output file was not analyzed here.

## Bottom Line

The update achieved the intended classifier hardening. It removed the contaminated broad TypeB buckets, sharply reduced false format-invalid deletes, promoted deterministic format and rule repairs to TypeA, and kept ambiguous or artifact-like cases diagnostic. The new manifests pass hard validation, and the audit sample now targets the refined high-risk strata that should be reviewed next.
