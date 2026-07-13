# Step 13A A-box Static Few-shot Gate Report

Verdict: `RUN_FULL_ABOX_FEW_SHOT_ABLATION_SUCCESS`

## Render and Leakage Gate
- Rendered prompts: `8408` across `4204` A-box cases and two contexts.
- Leakage scan: passed, hard hits `0`, soft lexical hits `79`.
- Soft terms: `{'classification': 75, 'popularity': 4}`.
- Overlap scan: passed `True`, core case overlap `0`, core T-box revision overlap `0`.

## Gate Checks
- `request_error_le_1pct`: `True`
- `proposal_parse_error_le_4pct`: `True`
- `leakage_scan_pass`: `True`
- `overlap_scan_pass`: `True`
- `schema_outputs_valid`: `True`
- `accepted_or_exact_value_improves`: `True`
- `no_material_typec_hallucination_empty_op_worsening`: `True`
- `token_increase_documented`: `True`

## logic_only
| Metric | Baseline | Few-shot | Delta |
| --- | ---: | ---: | ---: |
| accepted_rate | 54.02% | 60.39% | +6.37 pp |
| exact_value | 54.02% | 60.49% | +6.47 pp |
| exact_action | 63.94% | 66.98% | +3.04 pp |
| regression_pass | 96.48% | 98.12% | +1.64 pp |
| information_preservation | 40.41% | 47.11% | +6.70 pp |
| parse_error | 2.28% | 0.98% | -1.31 pp |
| request_error | 0.00% | 0.00% | +0.00 pp |
| token_usage_total_mean | 2494.3623 | 4895.2376 | 2400.8754 |
| prompt_tokens_mean | 2230.7336 | 4650.1403 | 2419.4068 |
| completion_tokens_mean | 263.6287 | 245.0973 | -18.5314 |
| latency_seconds_mean | 7.6142 | 8.5603 | 0.9461 |

### Diagnostic Behavior
| Diagnostic | Baseline | Few-shot | Delta | Counts |
| --- | ---: | ---: | ---: | ---: |
| empty_ops | 0.00% | 0.74% | +0.74 pp | 0 -> 31 / 4204 |
| hallucinated_value | 28.40% | 7.28% | -21.12 pp | 1194 -> 306 / 4204 |
| overdelete | 26.76% | 26.74% | -0.02 pp | 1125 -> 1124 / 4204 |
| constraint/type-QID-as-value | 12.04% | 3.90% | -8.14 pp | 506 -> 164 / 4204 |

### TypeA/TypeB/TypeC Breakdown
| Group | Count | Accepted delta | Exact value delta | Exact action delta | Regression delta | Parse delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TypeA | 2228 | +6.42 pp | +6.55 pp | +6.01 pp | -0.49 pp | +0.22 pp |
| TypeB | 924 | +13.74 pp | +13.85 pp | +3.79 pp | +2.06 pp | -2.60 pp |
| TypeC | 1052 | -0.19 pp | -0.19 pp | -3.90 pp | +5.80 pp | -3.42 pp |

### Popularity/Main Score Breakdown
| Group | Count | Accepted delta | Exact value delta | Exact action delta | Regression delta | Parse delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| head | 840 | +9.30 pp | +3.86 pp | +6.08 pp | -0.21 pp | +1.38 pp |
| mid | 2524 | +12.75 pp | +6.28 pp | +2.88 pp | +1.56 pp | -1.24 pp |
| tail | 840 | +17.86 pp | +10.14 pp | +0.63 pp | +3.95 pp | -2.99 pp |

### Subtype Breakdown
| Group | Count | Accepted delta | Exact value delta | Exact action delta | Regression delta | Parse delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COINCIDENTAL_SCHEMA_CHANGE | None | n/a | n/a | n/a | n/a | n/a |
| DELETE_AMBIGUOUS | 682 | -6.74 pp | -6.45 pp | -6.45 pp | -0.29 pp | -0.59 pp |
| EXTERNAL_BY_ELIMINATION | 1052 | -0.19 pp | -0.19 pp | -3.90 pp | +5.80 pp | -3.42 pp |
| FORMAT_NORMALIZATION | 542 | +54.43 pp | +54.43 pp | +54.80 pp | +0.18 pp | -0.18 pp |
| FORMAT_VALUE_PRUNING | 229 | -34.50 pp | -34.50 pp | -34.50 pp | +0.00 pp | +0.00 pp |
| LOCAL_FOCUS_NON_TARGET_PROPERTY | 70 | +5.71 pp | +7.14 pp | -1.43 pp | -4.29 pp | -5.71 pp |
| LOCAL_MIXED | 17 | +0.00 pp | +0.00 pp | +5.88 pp | +0.00 pp | +0.00 pp |
| LOCAL_NEIGHBOR_IDS | 2 | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| LOCAL_SELECTION_CONFIRMED | 521 | -0.19 pp | -0.19 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| LOCAL_TEXT_CONFIRMED | 87 | +9.20 pp | +9.20 pp | +11.49 pp | +0.00 pp | +0.00 pp |
| LOCAL_TEXT_DERIVED | 227 | +51.10 pp | +51.10 pp | +11.01 pp | +9.69 pp | -8.81 pp |
| LOGICAL | 2 | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| MULTIPLICITY_NORMALIZATION | 40 | +22.50 pp | +22.50 pp | +20.00 pp | +0.00 pp | +0.00 pp |
| REJECTION_FORMAT_INVALID | 258 | -20.16 pp | -19.77 pp | -19.77 pp | -5.04 pp | +5.04 pp |
| RELAXATION_SET_EXPANSION | None | n/a | n/a | n/a | n/a | n/a |
| SCHEMA_UPDATE | None | n/a | n/a | n/a | n/a | n/a |
| SELF_LINK_REJECTION | 128 | -0.78 pp | -0.78 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| SET_MEMBERSHIP_REJECTION | 293 | -0.34 pp | -0.34 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| TARGET_REQUIRED_CLAIM | 54 | +33.33 pp | +33.33 pp | +5.56 pp | +5.56 pp | -5.56 pp |

## local_graph
| Metric | Baseline | Few-shot | Delta |
| --- | ---: | ---: | ---: |
| accepted_rate | 53.16% | 60.94% | +7.78 pp |
| exact_value | 53.16% | 61.49% | +8.33 pp |
| exact_action | 65.29% | 67.96% | +2.66 pp |
| regression_pass | 94.89% | 96.24% | +1.36 pp |
| information_preservation | 40.09% | 48.81% | +8.72 pp |
| parse_error | 4.09% | 1.81% | -2.28 pp |
| request_error | 0.00% | 0.00% | +0.00 pp |
| token_usage_total_mean | 3249.0497 | 7286.6644 | 4037.6147 |
| prompt_tokens_mean | 2979.5818 | 7047.1330 | 4067.5511 |
| completion_tokens_mean | 269.4679 | 239.5314 | -29.9365 |
| latency_seconds_mean | 9.0139 | 9.9115 | 0.8975 |

### Diagnostic Behavior
| Diagnostic | Baseline | Few-shot | Delta | Counts |
| --- | ---: | ---: | ---: | ---: |
| empty_ops | 0.00% | 1.57% | +1.57 pp | 0 -> 66 / 4204 |
| hallucinated_value | 27.47% | 6.16% | -21.31 pp | 1155 -> 259 / 4204 |
| overdelete | 26.21% | 25.40% | -0.81 pp | 1102 -> 1068 / 4204 |
| constraint/type-QID-as-value | 10.54% | 3.19% | -7.35 pp | 443 -> 134 / 4204 |

### TypeA/TypeB/TypeC Breakdown
| Group | Count | Accepted delta | Exact value delta | Exact action delta | Regression delta | Parse delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TypeA | 2228 | +10.14 pp | +10.77 pp | +9.25 pp | -0.99 pp | +0.54 pp |
| TypeB | 924 | +11.04 pp | +11.80 pp | +1.52 pp | +0.97 pp | -1.52 pp |
| TypeC | 1052 | -0.10 pp | +0.10 pp | -10.27 pp | +6.65 pp | -8.94 pp |

### Popularity/Main Score Breakdown
| Group | Count | Accepted delta | Exact value delta | Exact action delta | Regression delta | Parse delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| head | 840 | +11.41 pp | +6.35 pp | +5.26 pp | -0.70 pp | +0.13 pp |
| mid | 2524 | +14.43 pp | +8.68 pp | +2.77 pp | +1.19 pp | -1.78 pp |
| tail | 840 | +17.17 pp | +9.65 pp | +0.02 pp | +4.23 pp | -4.39 pp |

### Subtype Breakdown
| Group | Count | Accepted delta | Exact value delta | Exact action delta | Regression delta | Parse delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COINCIDENTAL_SCHEMA_CHANGE | None | n/a | n/a | n/a | n/a | n/a |
| DELETE_AMBIGUOUS | 682 | +0.00 pp | +0.15 pp | +0.15 pp | +1.76 pp | -2.49 pp |
| EXTERNAL_BY_ELIMINATION | 1052 | -0.10 pp | +0.10 pp | -10.27 pp | +6.65 pp | -8.94 pp |
| FORMAT_NORMALIZATION | 542 | +40.41 pp | +40.96 pp | +40.59 pp | -0.18 pp | +0.00 pp |
| FORMAT_VALUE_PRUNING | 229 | +18.78 pp | +19.21 pp | +19.21 pp | +0.00 pp | +0.00 pp |
| LOCAL_FOCUS_NON_TARGET_PROPERTY | 70 | +2.86 pp | +4.29 pp | +0.00 pp | +1.43 pp | -8.57 pp |
| LOCAL_MIXED | 17 | +0.00 pp | +0.00 pp | +11.76 pp | +0.00 pp | -5.88 pp |
| LOCAL_NEIGHBOR_IDS | 2 | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| LOCAL_SELECTION_CONFIRMED | 521 | -2.30 pp | -2.11 pp | -1.34 pp | +0.00 pp | +0.00 pp |
| LOCAL_TEXT_CONFIRMED | 87 | +14.94 pp | +14.94 pp | +14.94 pp | +0.00 pp | -2.30 pp |
| LOCAL_TEXT_DERIVED | 227 | +43.61 pp | +45.81 pp | +2.64 pp | +3.52 pp | -2.20 pp |
| LOGICAL | 2 | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| MULTIPLICITY_NORMALIZATION | 40 | +12.50 pp | +12.50 pp | +10.00 pp | +0.00 pp | +0.00 pp |
| REJECTION_FORMAT_INVALID | 258 | -22.09 pp | -21.32 pp | -21.32 pp | -9.30 pp | +8.91 pp |
| RELAXATION_SET_EXPANSION | None | n/a | n/a | n/a | n/a | n/a |
| SCHEMA_UPDATE | None | n/a | n/a | n/a | n/a | n/a |
| SELF_LINK_REJECTION | 128 | -1.56 pp | -0.78 pp | -0.78 pp | -0.78 pp | +0.78 pp |
| SET_MEMBERSHIP_REJECTION | 293 | -1.71 pp | +0.34 pp | -0.34 pp | -0.68 pp | +0.68 pp |
| TARGET_REQUIRED_CLAIM | 54 | +42.59 pp | +42.59 pp | -11.11 pp | -11.11 pp | +5.56 pp |

## Notes
- A-box few-shot improves accepted/exact-value in both contexts versus the frozen zero-shot oracle baseline.
- Token use increased materially as expected for static 4-shot prompting; the report records prompt/completion/total-token and latency deltas.
- Diagnostic behavior rates for hallucinated values and constraint/type-QID-as-value are heuristic: they count non-gold proposed values from evaluator traces and may overcount plausible but non-gold values.
- No T-box few-shot inference or diagnosis-routed run was performed as part of this step.
