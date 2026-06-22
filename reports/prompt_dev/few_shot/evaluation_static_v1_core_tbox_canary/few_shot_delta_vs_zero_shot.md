# Few-Shot Delta vs Zero-Shot

Run id: `prompt_dev_eval_20260621T181453`

A-box, T-box taxonomy-patch, and diagnosis metrics are reported separately. No combined A-box/T-box headline is computed.

Static few-shot (`static_diverse_kshot`) is paper-facing. Dynamic retrieval policies are exploratory.

Token, cost, and latency overhead are included per comparison from `run_manifest.jsonl` usage fields.

## A-Box

| Few-shot matrix | Zero-shot matrix | Policy | Selection | Paper status | Metric | Zero-shot | Few-shot | Delta |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| n/a | n/a | n/a | n/a | n/a | No comparisons available | n/a | n/a | n/a |

## T-Box Taxonomy Patch

| Few-shot matrix | Zero-shot matrix | Policy | Selection | Paper status | Metric | Zero-shot | Few-shot | Delta |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | family_level_success | 0.672 | 0.812 | 0.141 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | schema_decision_match | 0.469 | 0.750 | 0.281 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | taxonomy_code_match | 0.102 | 0.234 | 0.133 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | taxonomy_level_success | 0.023 | 0.203 | 0.180 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | constraint_family_f1 | 0.615 | 0.788 | 0.173 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | repair_op_f1 | 0.043 | 0.215 | 0.172 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | value_delta_f1_when_applicable | 0.052 | 0.055 | 0.004 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | value_delta_claimed_when_gold_absent | 0.079 | 0.159 | 0.079 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | family_only_when_value_delta_gold_present | 0.477 | 0.585 | 0.108 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | out_of_current_gold_operation_false_positive_rate | 0.699 | 0.126 | -0.573 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | parse_error_rate | 0.000 | 0.000 | 0.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_prompt_tokens | 343847.000 | 755528.000 | 411681.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_completion_tokens | 41252.000 | 38436.000 | -2816.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_total_tokens | 385099.000 | 793964.000 | 408865.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_cached_tokens | n/a | n/a | n/a |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_estimated_cost_usd | 0.000 | 0.000 | 0.000 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_elapsed_seconds_total | 1055.482 | 1304.773 | 249.291 |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_elapsed_seconds_mean | 8.246 | 10.194 | 1.948 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | family_level_success | 0.781 | 0.781 | 0.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | schema_decision_match | 0.578 | 0.742 | 0.164 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | taxonomy_code_match | 0.195 | 0.273 | 0.078 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | taxonomy_level_success | 0.117 | 0.172 | 0.055 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | constraint_family_f1 | 0.740 | 0.759 | 0.019 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | repair_op_f1 | 0.116 | 0.247 | 0.131 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | value_delta_f1_when_applicable | 0.053 | 0.060 | 0.007 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | value_delta_claimed_when_gold_absent | 0.032 | 0.222 | 0.190 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | family_only_when_value_delta_gold_present | 0.492 | 0.615 | 0.123 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | out_of_current_gold_operation_false_positive_rate | 0.705 | 0.112 | -0.593 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | parse_error_rate | 0.000 | 0.000 | 0.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_prompt_tokens | 441989.000 | 1102397.000 | 660408.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_completion_tokens | 41363.000 | 36079.000 | -5284.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_total_tokens | 483352.000 | 1138476.000 | 655124.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_cached_tokens | n/a | n/a | n/a |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_estimated_cost_usd | 0.000 | 0.000 | 0.000 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_elapsed_seconds_total | 1115.283 | 1348.137 | 232.854 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `static_diverse_kshot` | `static_support_set` | paper-facing | overhead_elapsed_seconds_mean | 8.713 | 10.532 | 1.819 |

## Diagnosis

| Few-shot matrix | Zero-shot matrix | Policy | Selection | Paper status | Metric | Zero-shot | Few-shot | Delta |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| n/a | n/a | n/a | n/a | n/a | No comparisons available | n/a | n/a | n/a |
