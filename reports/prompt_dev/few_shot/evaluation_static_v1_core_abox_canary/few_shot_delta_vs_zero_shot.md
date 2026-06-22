# Few-Shot Delta vs Zero-Shot

Run id: `prompt_dev_eval_20260621T160709`

A-box, T-box taxonomy-patch, and diagnosis metrics are reported separately. No combined A-box/T-box headline is computed.

Static few-shot (`static_diverse_kshot`) is paper-facing. Dynamic retrieval policies are exploratory.

Token, cost, and latency overhead are included per comparison from `run_manifest.jsonl` usage fields.

## A-Box

| Few-shot matrix | Zero-shot matrix | Policy | Selection | Paper status | Metric | Zero-shot | Few-shot | Delta |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overall_a_box_accepted | 0.406 | 0.434 | 0.027 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | typea_accepted | 0.814 | 0.802 | -0.012 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | typeb_accepted | 0.341 | 0.424 | 0.082 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | typec_accepted | 0.059 | 0.071 | 0.012 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | a_box_exact_value | 0.410 | 0.434 | 0.023 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | a_box_exact_action | 0.547 | 0.547 | 0.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | a_box_regression_pass | 0.922 | 0.957 | 0.035 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overdelete_rate | n/a | n/a | n/a |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | empty_ops_rate | 0.039 | 0.031 | -0.008 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | constraint_type_qid_as_value_rate | n/a | n/a | n/a |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | parse_error_rate | 0.000 | 0.004 | 0.004 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_prompt_tokens | 564812.000 | 1213396.000 | 648584.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_completion_tokens | 62530.000 | 58775.000 | -3755.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_total_tokens | 627342.000 | 1272171.000 | 644829.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_cached_tokens | n/a | n/a | n/a |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_estimated_cost_usd | 0.000 | 0.000 | 0.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_elapsed_seconds_total | 2390.901 | 2264.343 | -126.558 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_elapsed_seconds_mean | 9.339 | 8.845 | -0.494 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overall_a_box_accepted | 0.398 | 0.457 | 0.059 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | typea_accepted | 0.756 | 0.860 | 0.105 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | typeb_accepted | 0.365 | 0.435 | 0.071 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | typec_accepted | 0.071 | 0.071 | 0.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | a_box_exact_value | 0.398 | 0.461 | 0.062 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | a_box_exact_action | 0.547 | 0.574 | 0.027 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | a_box_regression_pass | 0.875 | 0.922 | 0.047 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overdelete_rate | n/a | n/a | n/a |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | empty_ops_rate | 0.078 | 0.039 | -0.039 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | constraint_type_qid_as_value_rate | n/a | n/a | n/a |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | parse_error_rate | 0.012 | 0.023 | 0.012 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_prompt_tokens | 764969.000 | 1830426.000 | 1065457.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_completion_tokens | 62845.000 | 59664.000 | -3181.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_total_tokens | 827814.000 | 1890090.000 | 1062276.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_cached_tokens | n/a | n/a | n/a |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_estimated_cost_usd | 0.000 | 0.000 | 0.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_elapsed_seconds_total | 2192.168 | 2580.387 | 388.218 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_elapsed_seconds_mean | 8.563 | 10.080 | 1.516 |

## T-Box Taxonomy Patch

| Few-shot matrix | Zero-shot matrix | Policy | Selection | Paper status | Metric | Zero-shot | Few-shot | Delta |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| n/a | n/a | n/a | n/a | n/a | No comparisons available | n/a | n/a | n/a |

## Diagnosis

| Few-shot matrix | Zero-shot matrix | Policy | Selection | Paper status | Metric | Zero-shot | Few-shot | Delta |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| n/a | n/a | n/a | n/a | n/a | No comparisons available | n/a | n/a | n/a |
