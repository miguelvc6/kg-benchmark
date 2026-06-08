# Prompt Development Evaluation

Run id: `prompt_dev_eval_20260607T152438`
Provider: `university`
Model: `Qwen/Qwen3-4B`
Evaluated prompts: `192`

| Matrix id | Task | Representation | Examples | Context | Track mode | Parse errors | Request errors | Functional | Track acc | Audit |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain` | `track_diagnosis` | `hybrid_json_nl` | `zero_shot` | `logic_only` | `` | 0 | 0 | 0.000 | 0.458 | 0.000 |
| `prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `repair_proposal` | `hybrid_json_nl` | `zero_shot` | `logic_only` | `oracle` | 1 | 0 | 0.167 | 0.000 | 0.979 |
| `prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain` | `track_diagnosis` | `hybrid_json_nl` | `zero_shot` | `local_graph` | `` | 0 | 0 | 0.000 | 0.521 | 0.000 |
| `prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `repair_proposal` | `hybrid_json_nl` | `zero_shot` | `local_graph` | `oracle` | 0 | 0 | 0.167 | 0.000 | 1.000 |
