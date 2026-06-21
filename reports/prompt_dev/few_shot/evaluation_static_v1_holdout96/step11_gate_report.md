# Step 11 Gate Report

Status: **PASS_WITH_REVIEW**

Step 11 satisfies the stated hard gates. Do not treat it as a silent go-ahead for the larger core canary: token usage rose sharply, taxonomy-code exact match regressed in both contexts, and local_graph value-delta false positives increased. Proceed to Step 12 only after accepting that cost/regression tradeoff or after choosing a mitigation threshold.

## Run Evidence

- Run ID: `prompt_dev_eval_20260618T145902`
- Provider/model: `ollama` / `gpt-oss:120b`
- Manifest version: `prompt_dev_v5_tbox_taxonomy_patch`
- Rendered/evaluated prompts: 192 / 192
- Parse status counts: `{'normalized': 186, 'parse_error': 1, 'non_executable_empty_ops': 5}`
- Zero-shot baseline: `/home/wucloud/kg-benchmark/reports/prompt_dev/evaluation_prompt_dev_v5_tbox_taxonomy_patch_holdout96_zero_shot/prompt_dev_evaluation_summary.json`

## Gate Checks

| Gate | Result |
|---|---:|
| required artifacts present | pass |
| local only provider | pass |
| request error rate lte 1pct | pass |
| proposal parse error rate lte 4pct | pass |
| leakage scan passed | pass |
| overlap scan passed | pass |
| correct schema outputs present | pass |
| tbox taxonomy metrics and diagnostics emitted | pass |
| delta report compares to zero shot | pass |

## Matrix Error Rates

| Context | Request error | Proposal parse error | A-box proposals | T-box taxonomy proposals |
|---|---:|---:|---:|---:|
| logic_only | 0.0% | 0.0% | True | True |
| local_graph | 0.0% | 1.0% | True | True |

## A-box Delta vs Zero-shot

| Context | Accepted | Exact value | Exact action | Regression pass | Parse error | Total-token delta | Mean-latency delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| logic_only | +12.5 pp | +12.5 pp | +8.3 pp | +2.1 pp | -1.0 pp | 269941.000 | 3.151s |
| local_graph | +6.2 pp | +6.2 pp | +0.0 pp | +2.1 pp | +1.0 pp | 434891.000 | 1.637s |

## T-box Taxonomy Delta vs Zero-shot

| Context | Family | Schema decision | Taxonomy code | Taxonomy level | Repair op F1 | Value-delta F1 | Value-delta FP | Parse error | Total-token delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logic_only | +17.5 pp | +26.3 pp | -2.5 pp | +1.8 pp | +4.2 pp | +8.3 pp | +0.0 pp | -2.1 pp | 269941.000 |
| local_graph | +18.4 pp | +30.5 pp | -6.0 pp | +2.3 pp | +2.1 pp | +11.3 pp | +6.7 pp | +2.1 pp | 434891.000 |

## Diagnostics Emitted

| Context | Metric family | Confusion matrices | Value-delta display | OOC-gold FP rates | Macro averages |
|---|---|---|---:|---:|---:|
| logic_only | `tbox_taxonomy_patch_v1` | qualifier_property, repair_operation, schema_decision, taxonomy_code | True | True | True |
| local_graph | `tbox_taxonomy_patch_v1` | qualifier_property, repair_operation, schema_decision, taxonomy_code | True | True | True |

## Review Items Before Step 12

- `logic_only` `taxonomy_code_match` changed by -2.5 pp.
- `local_graph` `taxonomy_code_match` changed by -6.0 pp.
- `local_graph` `value_delta_claimed_when_gold_absent` changed by +6.7 pp.
- `local_graph` `t_box_parse_error_rate` changed by +2.1 pp.
- `local_graph` `a_box_parse_error_rate` changed by +1.0 pp.

The primary Step 11 gates pass, but the report intentionally keeps the Step 12 decision explicit because the canary is a larger local-model run and the taxonomy-code/value-delta regressions are real.
