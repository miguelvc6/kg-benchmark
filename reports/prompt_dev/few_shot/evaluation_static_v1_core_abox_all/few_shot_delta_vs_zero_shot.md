# Few-Shot Delta vs Zero-Shot

> **INVALID / SUPERSEDED:** the zero-shot baseline uses a 96-case development population while the few-shot run uses
> 4,204 core A-box cases, with zero shared case IDs. The deltas and total-usage differences below are retained only as
> historical diagnostics and must not be cited as a few-shot effect.

Run id: `prompt_dev_eval_20260622T083142`

A-box, T-box taxonomy-patch, and diagnosis metrics are reported separately. No combined A-box/T-box headline is computed.

No comparison in this artifact is paper-eligible.

Token, cost, and latency overhead are included per comparison from `run_manifest.jsonl` usage fields.

## A-Box

| Few-shot matrix | Zero-shot matrix | Policy | Selection | Paper status | Metric | Zero-shot | Few-shot | Delta |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overall_a_box_accepted | 0.396 | 0.604 | 0.208 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | typea_accepted | 0.667 | 0.790 | 0.123 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | typeb_accepted | 0.250 | 0.814 | 0.564 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | typec_accepted | 0.000 | 0.026 | 0.026 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | a_box_exact_value | 0.396 | 0.605 | 0.209 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | a_box_exact_action | 0.562 | 0.670 | 0.107 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | a_box_regression_pass | 0.938 | 0.981 | 0.044 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overdelete_rate | n/a | n/a | n/a |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | empty_ops_rate | 0.021 | 0.007 | -0.013 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | constraint_type_qid_as_value_rate | n/a | n/a | n/a |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | parse_error_rate | 0.010 | 0.010 | -0.001 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_prompt_tokens | 209402.000 | 19549190.000 | 19339788.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_completion_tokens | 25974.000 | 1030389.000 | 1004415.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_total_tokens | 235376.000 | 20579579.000 | 20344203.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_cached_tokens | n/a | n/a | n/a |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_estimated_cost_usd | 0.000 | 0.000 | 0.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_elapsed_seconds_total | 717.473 | 35987.623 | 35270.150 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_elapsed_seconds_mean | 7.474 | 8.560 | 1.087 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overall_a_box_accepted | 0.500 | 0.609 | 0.109 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | typea_accepted | 0.708 | 0.807 | 0.099 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | typeb_accepted | 0.583 | 0.805 | 0.222 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | typec_accepted | 0.000 | 0.019 | 0.019 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | a_box_exact_value | 0.500 | 0.615 | 0.115 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | a_box_exact_action | 0.667 | 0.680 | 0.013 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | a_box_regression_pass | 0.917 | 0.962 | 0.046 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overdelete_rate | n/a | n/a | n/a |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | empty_ops_rate | 0.031 | 0.016 | -0.016 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | constraint_type_qid_as_value_rate | n/a | n/a | n/a |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | parse_error_rate | 0.000 | 0.018 | 0.018 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_prompt_tokens | 266927.000 | 29626147.000 | 29359220.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_completion_tokens | 27273.000 | 1006990.000 | 979717.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_total_tokens | 294200.000 | 30633137.000 | 30338937.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_cached_tokens | n/a | n/a | n/a |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_estimated_cost_usd | 0.000 | 0.000 | 0.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_elapsed_seconds_total | 738.389 | 41667.817 | 40929.428 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | invalid/superseded | overhead_elapsed_seconds_mean | 7.692 | 9.911 | 2.220 |

## T-Box Taxonomy Patch

| Few-shot matrix | Zero-shot matrix | Policy | Selection | Paper status | Metric | Zero-shot | Few-shot | Delta |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| n/a | n/a | n/a | n/a | n/a | No comparisons available | n/a | n/a | n/a |

## Diagnosis

| Few-shot matrix | Zero-shot matrix | Policy | Selection | Paper status | Metric | Zero-shot | Few-shot | Delta |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| n/a | n/a | n/a | n/a | n/a | No comparisons available | n/a | n/a | n/a |
