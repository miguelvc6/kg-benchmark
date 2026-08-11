# Track Diagnosis

Diagnosis is scored independently and never routes repair proposals.

| Role | Population | Model | Regime | Context | n | Accuracy | Macro-F1 | False locus | Ambiguous |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| azure_calibration | azure-600 | azure_gpt_5_6_sol_high | static_few_shot | local_graph | 600 | 0.563 | 0.524 | 0.325 | 0.112 |
| confirmatory | main-1200 | ollama_llama3_3_70b | zero_shot | logic_only | 1200 | 0.725 | 0.615 | 0.264 | 0.001 |
| confirmatory | main-1200 | ollama_gpt_oss_120b | zero_shot | logic_only | 1200 | 0.654 | 0.445 | 0.277 | 0.051 |
| confirmatory | main-1200 | ollama_qwen3_30b | zero_shot | logic_only | 1200 | 0.628 | 0.425 | 0.323 | 0.035 |
| confirmatory | main-1200 | ollama_gpt_oss_120b | static_few_shot | logic_only | 1200 | 0.662 | 0.505 | 0.292 | 0.018 |
| confirmatory | main-1200 | ollama_llama3_3_70b | static_few_shot | logic_only | 1200 | 0.847 | 0.747 | 0.142 | 0.000 |
| confirmatory | main-1200 | ollama_llama3_3_70b | zero_shot | local_graph | 1200 | 0.682 | 0.594 | 0.302 | 0.001 |
| confirmatory | main-1200 | ollama_gpt_oss_120b | static_few_shot | local_graph | 1200 | 0.678 | 0.511 | 0.275 | 0.021 |
| confirmatory | main-1200 | ollama_llama3_3_70b | static_few_shot | local_graph | 1200 | 0.855 | 0.789 | 0.131 | 0.000 |
| confirmatory | main-1200 | ollama_qwen3_30b | static_few_shot | logic_only | 1200 | 0.629 | 0.518 | 0.343 | 0.013 |
| confirmatory | main-1200 | ollama_qwen3_30b | zero_shot | local_graph | 1200 | 0.623 | 0.504 | 0.293 | 0.059 |
| confirmatory | main-1200 | ollama_qwen3_30b | static_few_shot | local_graph | 1200 | 0.595 | 0.495 | 0.373 | 0.016 |
| confirmatory | main-1200 | ollama_gpt_oss_120b | zero_shot | local_graph | 1200 | 0.691 | 0.491 | 0.252 | 0.039 |
| azure_calibration | azure-600 | azure_gpt_5_6_sol_high | zero_shot | local_graph | 600 | 0.502 | 0.401 | 0.363 | 0.135 |
| azure_calibration | azure-600 | azure_gpt_5_6_sol_high | zero_shot | logic_only | 600 | 0.508 | 0.437 | 0.257 | 0.235 |
| azure_calibration | azure-600 | azure_gpt_5_6_sol_high | static_few_shot | logic_only | 600 | 0.553 | 0.508 | 0.310 | 0.137 |
