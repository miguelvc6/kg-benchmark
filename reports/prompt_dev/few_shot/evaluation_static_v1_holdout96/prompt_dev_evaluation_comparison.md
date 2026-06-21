# Prompt Development Evaluation

Run id: `prompt_dev_eval_20260618T145902`
Provider: `ollama`
Model: `gpt-oss:120b`
Evaluated prompts: `192`

| Matrix id | Task | Representation | Examples | Context | Track mode | Parse errors | Request errors | Strict functional | Track acc | Strict audit | T-box family | T-box decision | T-box taxonomy | T-box value F1 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain` | `repair_proposal` | `hybrid_json_nl` | `static_diverse_kshot` | `logic_only` | `oracle` | 0 | 0 | 0.260 | n/a | 0.479 | 0.792 | 0.562 | 0.146 | 0.206 |
| `prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain` | `repair_proposal` | `hybrid_json_nl` | `static_diverse_kshot` | `local_graph` | `oracle` | 1 | 0 | 0.281 | n/a | 0.469 | 0.830 | 0.596 | 0.106 | 0.171 |
