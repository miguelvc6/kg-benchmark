# T-box Static Few-shot v1 Decision

Verdict: `DO_NOT_USE_TBOX_FEW_SHOT_STATIC_V1`

Reason: value-delta false positives cluster around `CQ_PLUS` / `VALUE_DELTA_VISIBLE` and are mostly copied from static examples rather than held-out case evidence.

No further T-box inference was run. The static support set was not modified. Zero-shot taxonomy-patch remains the T-box paper-facing result.

Source analysis: `reports/prompt_dev/few_shot/evaluation_static_v1_core_tbox_canary/tbox_value_delta_fp_analysis.json`
