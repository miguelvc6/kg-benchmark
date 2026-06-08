# Prompt v2 48-Case Canary Delta

Compared runs: `reports/prompt_dev/evaluation_prompt_dev_v2_48_zero_shot` against `reports/prompt_dev/evaluation_prompt_dev_v1_96_zero_shot`

## Verdict

Do not freeze v2 and do not scale this exact prompt to v2_96 yet. The canary validates the T-box signature-discipline change, but A-box over-delete worsened and track diagnosis regressed. Build a narrower v3 that keeps the T-box fixes while revising A-box operation/value-source guidance and track-diagnosis wording.

## Operational Stability

- v2 rendered prompts: `192` over `48` cases
- v2 parse statuses: `{"normalized": 191, "parse_error": 1}`
- v2 leakage counters: `{}`

## Targeted Prompt-Bug Rates

| Bug heuristic | v1_96 rate | v2_48 rate | Delta |
|---|---:|---:|---:|
| `a_box_constraint_type_value` | 29.2% | 25.0% | -4.2% |
| `a_box_over_delete` | 40.6% | 43.8% | 3.1% |
| `t_box_directional_without_visible_evidence` | 44.8% | 2.1% | -42.7% |
| `t_box_invented_signature_after` | 63.5% | 2.1% | -61.5% |
| `t_box_report_qid_copy` | 6.2% | 2.1% | -4.2% |
| `typec_concrete_failed_repair` | 22.9% | 20.8% | -2.1% |

## Matrix Metrics

| Run | Matrix | n | Track correct | A-box exact value | T-box semantic family | T-box invented signatures |
|---|---|---:|---:|---:|---:|---:|
| v1_96 | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain` | 96 | 64/96 |  |  |  |
| v1_96 | `prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | 96 |  | 19/48 | 3/48 after evaluator applicability fix | 27/48 |
| v1_96 | `prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain` | 96 | 63/96 |  |  |  |
| v1_96 | `prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | 96 |  | 19/48 | 5/48 after evaluator applicability fix | 34/48 |
| v2_48 | `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain` | 48 | 22/48 |  |  |  |
| v2_48 | `prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | 48 |  | 8/24 | 5/24 | 0/24 |
| v2_48 | `prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain` | 48 | 25/48 |  |  |  |
| v2_48 | `prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | 48 |  | 8/24 | 5/24 | 1/24 |

## Local Graph vs Logic-Only

- v2 track diagnosis improves from 22/48 logic-only to 25/48 local-graph, but both are below the v1_96 rates.
- v2 A-box exact value is unchanged by local graph: 8/24 in both contexts.
- v2 T-box semantic-family success is unchanged by local graph: 5/24 in both contexts.
- v2 T-box invented signature rate is 0/24 logic-only and 1/24 local-graph; this is still a large improvement over v1_96.

## A-box Exact Value By Subtype

### v2_48

- `DELETE_AMBIGUOUS`: 2/4 (50.0%)
- `EXTERNAL_BY_ELIMINATION`: 0/2 (0.0%)
- `FORMAT_NORMALIZATION`: 0/4 (0.0%)
- `FORMAT_VALUE_PRUNING`: 2/4 (50.0%)
- `LOCAL_FOCUS_NON_TARGET_PROPERTY`: 0/2 (0.0%)
- `LOCAL_MIXED`: 0/2 (0.0%)
- `LOCAL_SELECTION_CONFIRMED`: 0/2 (0.0%)
- `LOCAL_TEXT_CONFIRMED`: 0/2 (0.0%)
- `LOCAL_TEXT_DERIVED`: 0/2 (0.0%)
- `MULTIPLICITY_NORMALIZATION`: 3/4 (75.0%)
- `REJECTION_FORMAT_INVALID`: 3/4 (75.0%)
- `SELF_LINK_REJECTION`: 3/4 (75.0%)
- `SET_MEMBERSHIP_REJECTION`: 2/2 (100.0%)
- `TARGET_REQUIRED_CLAIM`: 1/2 (50.0%)
- `UNKNOWN_BAD_TARGET_OR_CONTEXT`: 0/2 (0.0%)
- `UNKNOWN_INCOMPLETE_LOCAL_CONTEXT`: 0/2 (0.0%)
- `UNKNOWN_MULTIPLICITY_ARTIFACT`: 0/2 (0.0%)
- `UNKNOWN_SELECTION_AMBIGUOUS`: 0/2 (0.0%)

## Interpretation

- Comparisons are cautious because v1 has 96 cases and v2 has 48 cases.
- The primary acceptance checks are operational stability, no leakage, and movement in the systematic prompt-bug counters.
- T-box signature discipline improved enough to keep that part of v2.
- A-box and track-diagnosis regressions mean the next step should be a narrower v3, not Phase G main scoring and not this exact v2 at 96 cases.
