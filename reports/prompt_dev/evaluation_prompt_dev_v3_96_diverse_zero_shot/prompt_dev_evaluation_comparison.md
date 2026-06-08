# Prompt Development Evaluation

Run id: `prompt_dev_eval_20260607T202605`
Provider: `university`
Model: `Qwen/Qwen3-4B`
Evaluated prompts: `384`

| Matrix id | Task | Representation | Examples | Context | Track mode | Parse errors | Request errors | Functional | Track acc | Audit |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain` | `track_diagnosis` | `hybrid_json_nl` | `zero_shot` | `logic_only` | `` | 0 | 0 | 0.000 | 0.562 | 0.000 |
| `prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `repair_proposal` | `hybrid_json_nl` | `zero_shot` | `logic_only` | `oracle` | 2 | 1 | 0.177 | 0.000 | 0.969 |
| `prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain` | `track_diagnosis` | `hybrid_json_nl` | `zero_shot` | `local_graph` | `` | 0 | 0 | 0.000 | 0.594 | 0.000 |
| `prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `repair_proposal` | `hybrid_json_nl` | `zero_shot` | `local_graph` | `oracle` | 0 | 1 | 0.208 | 0.000 | 0.990 |
