# Prompt Development Evaluation

Run id: `prompt_dev_eval_20260609T171656`
Provider: `ollama`
Model: `gpt-oss:120b`
Evaluated prompts: `384`

| Matrix id | Task | Representation | Examples | Context | Track mode | Parse errors | Request errors | Functional | Track acc | Audit |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain` | `track_diagnosis` | `hybrid_json_nl` | `zero_shot` | `logic_only` | `` | 0 | 0 | n/a | 0.490 | n/a |
| `prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_abstain` | `repair_proposal` | `hybrid_json_nl` | `zero_shot` | `logic_only` | `oracle` | 48 | 0 | 0.135 | n/a | 0.500 |
| `prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain` | `track_diagnosis` | `hybrid_json_nl` | `zero_shot` | `local_graph` | `` | 0 | 0 | n/a | 0.479 | n/a |
| `prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_abstain` | `repair_proposal` | `hybrid_json_nl` | `zero_shot` | `local_graph` | `oracle` | 50 | 0 | 0.146 | n/a | 0.479 |
