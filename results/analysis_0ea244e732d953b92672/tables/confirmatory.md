# Confirmatory

Primary endpoint estimates; models and repair loci are not pooled.

| Population | Model | Task | Stratum | Endpoint | Regime | Context | n | Case micro (95% CI) | Event macro (95% CI) |
|---|---|---|---|---|---|---|---:|---:|---:|
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-E-elim | accepted_repair | static_few_shot | local_graph | 320 | 0.041 [0.022, 0.066] | 0.041 [0.022, 0.066] |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-E-elim | accepted_repair | static_few_shot | logic_only | 320 | 0.034 [0.016, 0.056] | 0.034 [0.016, 0.056] |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-E-elim | accepted_repair | zero_shot | local_graph | 320 | 0.031 [0.016, 0.053] | 0.031 [0.016, 0.053] |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-E-elim | accepted_repair | zero_shot | logic_only | 320 | 0.031 [0.016, 0.053] | 0.031 [0.016, 0.053] |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-G | accepted_repair | static_few_shot | local_graph | 406 | 0.155 [0.121, 0.190] | 0.155 [0.121, 0.190] |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-G | accepted_repair | static_few_shot | logic_only | 406 | 0.170 [0.133, 0.209] | 0.170 [0.133, 0.209] |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-G | accepted_repair | zero_shot | local_graph | 406 | 0.153 [0.118, 0.187] | 0.153 [0.118, 0.187] |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-G | accepted_repair | zero_shot | logic_only | 406 | 0.126 [0.094, 0.158] | 0.126 [0.094, 0.158] |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-L | accepted_repair | static_few_shot | local_graph | 249 | 0.888 [0.847, 0.928] | 0.888 [0.847, 0.928] |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-L | accepted_repair | static_few_shot | logic_only | 249 | 0.936 [0.904, 0.964] | 0.936 [0.904, 0.964] |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-L | accepted_repair | zero_shot | local_graph | 249 | 0.880 [0.839, 0.920] | 0.880 [0.839, 0.920] |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-L | accepted_repair | zero_shot | logic_only | 249 | 0.880 [0.835, 0.920] | 0.880 [0.835, 0.920] |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | static_few_shot | local_graph | 225 | 0.324 [0.262, 0.387] | 0.324 [0.262, 0.387] |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | static_few_shot | logic_only | 225 | 0.320 [0.258, 0.382] | 0.320 [0.258, 0.382] |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | zero_shot | local_graph | 225 | 0.080 [0.049, 0.116] | 0.080 [0.049, 0.116] |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | zero_shot | logic_only | 225 | 0.111 [0.071, 0.151] | 0.111 [0.071, 0.151] |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | static_few_shot | local_graph | 225 | 0.320 [0.258, 0.382] | 0.320 [0.258, 0.382] |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | static_few_shot | logic_only | 225 | 0.324 [0.262, 0.387] | 0.324 [0.262, 0.387] |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | zero_shot | local_graph | 225 | 0.547 [0.480, 0.613] | 0.547 [0.480, 0.613] |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | zero_shot | logic_only | 225 | 0.591 [0.529, 0.653] | 0.591 [0.529, 0.653] |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | ALL | accuracy | static_few_shot | local_graph | 1200 | 0.678 [0.651, 0.704] | 0.678 [0.651, 0.704] |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | ALL | accuracy | static_few_shot | logic_only | 1200 | 0.662 [0.635, 0.688] | 0.662 [0.635, 0.688] |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | ALL | accuracy | zero_shot | local_graph | 1200 | 0.691 [0.664, 0.717] | 0.691 [0.664, 0.717] |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | ALL | accuracy | zero_shot | logic_only | 1200 | 0.654 [0.627, 0.680] | 0.654 [0.627, 0.680] |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | A_BOX | accuracy | static_few_shot | local_graph | 975 | 0.786 [0.759, 0.811] | 0.786 [0.759, 0.811] |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | A_BOX | accuracy | static_few_shot | logic_only | 975 | 0.762 [0.734, 0.789] | 0.762 [0.734, 0.789] |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | A_BOX | accuracy | zero_shot | local_graph | 975 | 0.818 [0.795, 0.842] | 0.818 [0.795, 0.842] |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | A_BOX | accuracy | zero_shot | logic_only | 975 | 0.787 [0.760, 0.812] | 0.787 [0.760, 0.812] |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | T_BOX | accuracy | static_few_shot | local_graph | 225 | 0.213 [0.160, 0.267] | 0.213 [0.160, 0.267] |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | T_BOX | accuracy | static_few_shot | logic_only | 225 | 0.227 [0.173, 0.284] | 0.227 [0.173, 0.284] |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | T_BOX | accuracy | zero_shot | local_graph | 225 | 0.138 [0.098, 0.187] | 0.138 [0.098, 0.187] |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | T_BOX | accuracy | zero_shot | logic_only | 225 | 0.080 [0.044, 0.116] | 0.080 [0.044, 0.116] |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-E-elim | accepted_repair | static_few_shot | local_graph | 320 | 0.031 [0.016, 0.053] | 0.031 [0.016, 0.053] |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-E-elim | accepted_repair | static_few_shot | logic_only | 320 | 0.047 [0.025, 0.072] | 0.047 [0.025, 0.072] |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-E-elim | accepted_repair | zero_shot | local_graph | 320 | 0.031 [0.016, 0.053] | 0.031 [0.016, 0.053] |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-E-elim | accepted_repair | zero_shot | logic_only | 320 | 0.034 [0.016, 0.056] | 0.034 [0.016, 0.056] |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-G | accepted_repair | static_few_shot | local_graph | 406 | 0.155 [0.121, 0.192] | 0.155 [0.121, 0.192] |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-G | accepted_repair | static_few_shot | logic_only | 406 | 0.170 [0.135, 0.207] | 0.170 [0.135, 0.207] |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-G | accepted_repair | zero_shot | local_graph | 406 | 0.135 [0.103, 0.170] | 0.135 [0.103, 0.170] |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-G | accepted_repair | zero_shot | logic_only | 406 | 0.133 [0.101, 0.165] | 0.133 [0.101, 0.165] |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-L | accepted_repair | static_few_shot | local_graph | 249 | 0.755 [0.699, 0.807] | 0.755 [0.699, 0.807] |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-L | accepted_repair | static_few_shot | logic_only | 249 | 0.867 [0.823, 0.908] | 0.867 [0.823, 0.908] |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-L | accepted_repair | zero_shot | local_graph | 249 | 0.683 [0.627, 0.739] | 0.683 [0.627, 0.739] |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-L | accepted_repair | zero_shot | logic_only | 249 | 0.739 [0.683, 0.791] | 0.739 [0.683, 0.791] |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | static_few_shot | local_graph | 225 | 0.147 [0.102, 0.196] | 0.147 [0.102, 0.196] |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | static_few_shot | logic_only | 225 | 0.147 [0.102, 0.196] | 0.147 [0.102, 0.196] |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | zero_shot | local_graph | 225 | 0.058 [0.027, 0.089] | 0.058 [0.027, 0.089] |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | zero_shot | logic_only | 225 | 0.124 [0.084, 0.169] | 0.124 [0.084, 0.169] |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | static_few_shot | local_graph | 225 | 0.191 [0.138, 0.244] | 0.191 [0.138, 0.244] |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | static_few_shot | logic_only | 225 | 0.191 [0.138, 0.244] | 0.191 [0.138, 0.244] |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | zero_shot | local_graph | 225 | 0.080 [0.044, 0.116] | 0.080 [0.044, 0.116] |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | zero_shot | logic_only | 225 | 0.338 [0.276, 0.400] | 0.338 [0.276, 0.400] |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | ALL | accuracy | static_few_shot | local_graph | 1200 | 0.855 [0.835, 0.875] | 0.855 [0.835, 0.875] |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | ALL | accuracy | static_few_shot | logic_only | 1200 | 0.847 [0.826, 0.868] | 0.847 [0.826, 0.868] |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | ALL | accuracy | zero_shot | local_graph | 1200 | 0.682 [0.656, 0.708] | 0.682 [0.656, 0.708] |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | ALL | accuracy | zero_shot | logic_only | 1200 | 0.725 [0.699, 0.750] | 0.725 [0.699, 0.750] |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | A_BOX | accuracy | static_few_shot | local_graph | 975 | 0.889 [0.869, 0.909] | 0.889 [0.869, 0.909] |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | A_BOX | accuracy | static_few_shot | logic_only | 975 | 0.917 [0.898, 0.934] | 0.917 [0.898, 0.934] |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | A_BOX | accuracy | zero_shot | local_graph | 975 | 0.715 [0.687, 0.744] | 0.715 [0.687, 0.744] |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | A_BOX | accuracy | zero_shot | logic_only | 975 | 0.779 [0.753, 0.805] | 0.779 [0.753, 0.805] |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | T_BOX | accuracy | static_few_shot | local_graph | 225 | 0.707 [0.644, 0.764] | 0.707 [0.644, 0.764] |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | T_BOX | accuracy | static_few_shot | logic_only | 225 | 0.542 [0.476, 0.604] | 0.542 [0.476, 0.604] |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | T_BOX | accuracy | zero_shot | local_graph | 225 | 0.542 [0.476, 0.609] | 0.542 [0.476, 0.609] |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | T_BOX | accuracy | zero_shot | logic_only | 225 | 0.489 [0.422, 0.551] | 0.489 [0.422, 0.551] |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-E-elim | accepted_repair | static_few_shot | local_graph | 320 | 0.034 [0.016, 0.056] | 0.034 [0.016, 0.056] |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-E-elim | accepted_repair | static_few_shot | logic_only | 320 | 0.025 [0.009, 0.044] | 0.025 [0.009, 0.044] |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-E-elim | accepted_repair | zero_shot | local_graph | 320 | 0.031 [0.016, 0.050] | 0.031 [0.016, 0.050] |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-E-elim | accepted_repair | zero_shot | logic_only | 320 | 0.031 [0.016, 0.053] | 0.031 [0.016, 0.053] |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-G | accepted_repair | static_few_shot | local_graph | 406 | 0.131 [0.099, 0.165] | 0.131 [0.099, 0.165] |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-G | accepted_repair | static_few_shot | logic_only | 406 | 0.140 [0.106, 0.175] | 0.140 [0.106, 0.175] |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-G | accepted_repair | zero_shot | local_graph | 406 | 0.153 [0.118, 0.187] | 0.153 [0.118, 0.187] |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-G | accepted_repair | zero_shot | logic_only | 406 | 0.158 [0.123, 0.192] | 0.158 [0.123, 0.192] |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-L | accepted_repair | static_few_shot | local_graph | 249 | 0.855 [0.811, 0.896] | 0.855 [0.811, 0.896] |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-L | accepted_repair | static_few_shot | logic_only | 249 | 0.904 [0.863, 0.940] | 0.904 [0.863, 0.940] |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-L | accepted_repair | zero_shot | local_graph | 249 | 0.687 [0.630, 0.743] | 0.687 [0.630, 0.743] |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-L | accepted_repair | zero_shot | logic_only | 249 | 0.687 [0.627, 0.743] | 0.687 [0.627, 0.743] |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | static_few_shot | local_graph | 225 | 0.204 [0.156, 0.262] | 0.204 [0.156, 0.262] |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | static_few_shot | logic_only | 225 | 0.231 [0.178, 0.289] | 0.231 [0.178, 0.289] |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | zero_shot | local_graph | 225 | 0.044 [0.018, 0.071] | 0.044 [0.018, 0.071] |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | zero_shot | logic_only | 225 | 0.049 [0.022, 0.076] | 0.049 [0.022, 0.076] |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | static_few_shot | local_graph | 225 | 0.196 [0.147, 0.249] | 0.196 [0.147, 0.249] |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | static_few_shot | logic_only | 225 | 0.298 [0.236, 0.360] | 0.298 [0.236, 0.360] |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | zero_shot | local_graph | 225 | 0.569 [0.502, 0.636] | 0.569 [0.502, 0.636] |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | zero_shot | logic_only | 225 | 0.604 [0.542, 0.671] | 0.604 [0.542, 0.671] |
| main-1200 | ollama_qwen3_30b | track_diagnosis | ALL | accuracy | static_few_shot | local_graph | 1200 | 0.595 [0.568, 0.623] | 0.595 [0.568, 0.623] |
| main-1200 | ollama_qwen3_30b | track_diagnosis | ALL | accuracy | static_few_shot | logic_only | 1200 | 0.629 [0.603, 0.657] | 0.629 [0.603, 0.657] |
| main-1200 | ollama_qwen3_30b | track_diagnosis | ALL | accuracy | zero_shot | local_graph | 1200 | 0.623 [0.596, 0.650] | 0.623 [0.596, 0.650] |
| main-1200 | ollama_qwen3_30b | track_diagnosis | ALL | accuracy | zero_shot | logic_only | 1200 | 0.628 [0.601, 0.655] | 0.628 [0.601, 0.655] |
| main-1200 | ollama_qwen3_30b | track_diagnosis | A_BOX | accuracy | static_few_shot | local_graph | 975 | 0.648 [0.617, 0.677] | 0.648 [0.617, 0.677] |
| main-1200 | ollama_qwen3_30b | track_diagnosis | A_BOX | accuracy | static_few_shot | logic_only | 975 | 0.690 [0.662, 0.719] | 0.690 [0.662, 0.719] |
| main-1200 | ollama_qwen3_30b | track_diagnosis | A_BOX | accuracy | zero_shot | local_graph | 975 | 0.704 [0.675, 0.732] | 0.704 [0.675, 0.732] |
| main-1200 | ollama_qwen3_30b | track_diagnosis | A_BOX | accuracy | zero_shot | logic_only | 975 | 0.756 [0.728, 0.783] | 0.756 [0.728, 0.783] |
| main-1200 | ollama_qwen3_30b | track_diagnosis | T_BOX | accuracy | static_few_shot | local_graph | 225 | 0.364 [0.302, 0.427] | 0.364 [0.302, 0.427] |
| main-1200 | ollama_qwen3_30b | track_diagnosis | T_BOX | accuracy | static_few_shot | logic_only | 225 | 0.364 [0.302, 0.427] | 0.364 [0.302, 0.427] |
| main-1200 | ollama_qwen3_30b | track_diagnosis | T_BOX | accuracy | zero_shot | local_graph | 225 | 0.271 [0.218, 0.329] | 0.271 [0.218, 0.329] |
| main-1200 | ollama_qwen3_30b | track_diagnosis | T_BOX | accuracy | zero_shot | logic_only | 225 | 0.076 [0.044, 0.111] | 0.076 [0.044, 0.111] |

## Paired primary contrasts

Effects are right minus left. Holm adjustment is within each model × task × stratum × endpoint family.

| Population | Model | Task | Stratum | Endpoint | Contrast | n | Micro difference (95% CI) | Exact p | Holm p |
|---|---|---|---|---|---|---:|---:|---:|---:|
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-E-elim | accepted_repair | context_within_few_shot | 320 | 0.006 [0.000, 0.016] | 0.5 | 1 |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-E-elim | accepted_repair | context_within_zero_shot | 320 | 0.000 [0.000, 0.000] | 1 | 1 |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-E-elim | accepted_repair | few_shot_within_local_graph | 320 | 0.009 [0.000, 0.022] | 0.25 | 1 |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-E-elim | accepted_repair | few_shot_within_logic_only | 320 | 0.003 [0.000, 0.009] | 1 | 1 |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-G | accepted_repair | context_within_few_shot | 406 | -0.015 [-0.042, 0.010] | 0.3616 | 0.7232 |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-G | accepted_repair | context_within_zero_shot | 406 | 0.027 [0.005, 0.049] | 0.0266 | 0.07981 |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-G | accepted_repair | few_shot_within_local_graph | 406 | 0.002 [-0.020, 0.025] | 1 | 1 |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-G | accepted_repair | few_shot_within_logic_only | 406 | 0.044 [0.020, 0.074] | 0.002102 | 0.00841 |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-L | accepted_repair | context_within_few_shot | 249 | -0.048 [-0.084, -0.016] | 0.007538 | 0.02261 |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-L | accepted_repair | context_within_zero_shot | 249 | 0.000 [-0.040, 0.040] | 1 | 1 |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-L | accepted_repair | few_shot_within_local_graph | 249 | 0.008 [-0.036, 0.052] | 0.8601 | 1 |
| main-1200 | ollama_gpt_oss_120b | a_box_repair | IC-L | accepted_repair | few_shot_within_logic_only | 249 | 0.056 [0.020, 0.092] | 0.004344 | 0.01737 |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | context_within_few_shot | 225 | 0.004 [-0.067, 0.076] | 1 | 1 |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | context_within_zero_shot | 225 | -0.031 [-0.067, 0.004] | 0.1435 | 0.2869 |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | few_shot_within_local_graph | 225 | 0.244 [0.173, 0.320] | 6.803e-10 | 2.721e-09 |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | few_shot_within_logic_only | 225 | 0.209 [0.138, 0.280] | 9.439e-08 | 2.832e-07 |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | context_within_few_shot | 225 | -0.004 [-0.076, 0.067] | 1 | 1 |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | context_within_zero_shot | 225 | -0.044 [-0.093, 0.004] | 0.1102 | 0.2204 |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | few_shot_within_local_graph | 225 | -0.227 [-0.311, -0.133] | 2.673e-06 | 8.02e-06 |
| main-1200 | ollama_gpt_oss_120b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | few_shot_within_logic_only | 225 | -0.267 [-0.351, -0.187] | 3.787e-09 | 1.515e-08 |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | ALL | accuracy | context_within_few_shot | 1200 | 0.017 [-0.008, 0.042] | 0.2042 | 0.6125 |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | ALL | accuracy | context_within_zero_shot | 1200 | 0.037 [0.015, 0.059] | 0.001839 | 0.007355 |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | ALL | accuracy | few_shot_within_local_graph | 1200 | -0.013 [-0.035, 0.011] | 0.3185 | 0.6371 |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | ALL | accuracy | few_shot_within_logic_only | 1200 | 0.007 [-0.018, 0.033] | 0.6049 | 0.6371 |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | A_BOX | accuracy | context_within_few_shot | 975 | 0.024 [-0.003, 0.050] | 0.0922 | 0.1844 |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | A_BOX | accuracy | context_within_zero_shot | 975 | 0.032 [0.006, 0.056] | 0.01708 | 0.05125 |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | A_BOX | accuracy | few_shot_within_local_graph | 975 | -0.033 [-0.057, -0.008] | 0.01281 | 0.05125 |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | A_BOX | accuracy | few_shot_within_logic_only | 975 | -0.025 [-0.051, 0.003] | 0.09319 | 0.1844 |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | T_BOX | accuracy | context_within_few_shot | 225 | -0.013 [-0.076, 0.053] | 0.7838 | 0.7838 |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | T_BOX | accuracy | context_within_zero_shot | 225 | 0.058 [0.009, 0.107] | 0.03508 | 0.07016 |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | T_BOX | accuracy | few_shot_within_local_graph | 225 | 0.076 [0.022, 0.129] | 0.01151 | 0.03452 |
| main-1200 | ollama_gpt_oss_120b | track_diagnosis | T_BOX | accuracy | few_shot_within_logic_only | 225 | 0.147 [0.089, 0.209] | 3.389e-06 | 1.356e-05 |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-E-elim | accepted_repair | context_within_few_shot | 320 | -0.016 [-0.031, -0.003] | 0.0625 | 0.25 |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-E-elim | accepted_repair | context_within_zero_shot | 320 | -0.003 [-0.009, 0.000] | 1 | 1 |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-E-elim | accepted_repair | few_shot_within_local_graph | 320 | 0.000 [0.000, 0.000] | 1 | 1 |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-E-elim | accepted_repair | few_shot_within_logic_only | 320 | 0.013 [-0.003, 0.028] | 0.2188 | 0.6562 |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-G | accepted_repair | context_within_few_shot | 406 | -0.015 [-0.037, 0.007] | 0.2863 | 0.646 |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-G | accepted_repair | context_within_zero_shot | 406 | 0.002 [-0.022, 0.027] | 1 | 1 |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-G | accepted_repair | few_shot_within_local_graph | 406 | 0.020 [-0.007, 0.047] | 0.2153 | 0.646 |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-G | accepted_repair | few_shot_within_logic_only | 406 | 0.037 [0.015, 0.062] | 0.004077 | 0.01631 |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-L | accepted_repair | context_within_few_shot | 249 | -0.112 [-0.157, -0.072] | 2.463e-07 | 9.853e-07 |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-L | accepted_repair | context_within_zero_shot | 249 | -0.056 [-0.108, -0.004] | 0.05408 | 0.1071 |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-L | accepted_repair | few_shot_within_local_graph | 249 | 0.072 [0.004, 0.141] | 0.05354 | 0.1071 |
| main-1200 | ollama_llama3_3_70b | a_box_repair | IC-L | accepted_repair | few_shot_within_logic_only | 249 | 0.129 [0.076, 0.185] | 1.401e-05 | 4.204e-05 |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | context_within_few_shot | 225 | 0.000 [-0.040, 0.040] | 1 | 1 |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | context_within_zero_shot | 225 | -0.067 [-0.116, -0.018] | 0.01353 | 0.04059 |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | few_shot_within_local_graph | 225 | 0.089 [0.036, 0.138] | 0.001193 | 0.004773 |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | few_shot_within_logic_only | 225 | 0.022 [-0.036, 0.084] | 0.5682 | 1 |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | context_within_few_shot | 225 | 0.000 [-0.040, 0.040] | 1 | 1 |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | context_within_zero_shot | 225 | -0.258 [-0.329, -0.191] | 4.334e-12 | 1.734e-11 |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | few_shot_within_local_graph | 225 | 0.111 [0.053, 0.169] | 0.000346 | 0.000692 |
| main-1200 | ollama_llama3_3_70b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | few_shot_within_logic_only | 225 | -0.147 [-0.218, -0.076] | 0.0001123 | 0.0003368 |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | ALL | accuracy | context_within_few_shot | 1200 | 0.008 [-0.013, 0.029] | 0.4796 | 0.4796 |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | ALL | accuracy | context_within_zero_shot | 1200 | -0.043 [-0.072, -0.013] | 0.006215 | 0.01243 |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | ALL | accuracy | few_shot_within_local_graph | 1200 | 0.172 [0.145, 0.200] | 3.393e-34 | 1.357e-33 |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | ALL | accuracy | few_shot_within_logic_only | 1200 | 0.122 [0.098, 0.147] | 1.485e-20 | 4.455e-20 |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | A_BOX | accuracy | context_within_few_shot | 975 | -0.028 [-0.046, -0.010] | 0.004512 | 0.004512 |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | A_BOX | accuracy | context_within_zero_shot | 975 | -0.065 [-0.096, -0.033] | 0.0001158 | 0.0002316 |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | A_BOX | accuracy | few_shot_within_local_graph | 975 | 0.174 [0.146, 0.202] | 2.311e-31 | 9.244e-31 |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | A_BOX | accuracy | few_shot_within_logic_only | 975 | 0.137 [0.112, 0.164] | 1.011e-25 | 3.032e-25 |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | T_BOX | accuracy | context_within_few_shot | 225 | 0.164 [0.093, 0.236] | 2.926e-05 | 0.0001171 |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | T_BOX | accuracy | context_within_zero_shot | 225 | 0.053 [-0.022, 0.129] | 0.2007 | 0.4014 |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | T_BOX | accuracy | few_shot_within_local_graph | 225 | 0.164 [0.089, 0.236] | 2.926e-05 | 0.0001171 |
| main-1200 | ollama_llama3_3_70b | track_diagnosis | T_BOX | accuracy | few_shot_within_logic_only | 225 | 0.053 [-0.022, 0.129] | 0.2127 | 0.4014 |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-E-elim | accepted_repair | context_within_few_shot | 320 | 0.009 [0.000, 0.022] | 0.25 | 1 |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-E-elim | accepted_repair | context_within_zero_shot | 320 | 0.000 [-0.009, 0.009] | 1 | 1 |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-E-elim | accepted_repair | few_shot_within_local_graph | 320 | 0.003 [0.000, 0.009] | 1 | 1 |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-E-elim | accepted_repair | few_shot_within_logic_only | 320 | -0.006 [-0.016, 0.000] | 0.5 | 1 |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-G | accepted_repair | context_within_few_shot | 406 | -0.010 [-0.039, 0.020] | 0.6177 | 1 |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-G | accepted_repair | context_within_zero_shot | 406 | -0.005 [-0.027, 0.017] | 0.8318 | 1 |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-G | accepted_repair | few_shot_within_local_graph | 406 | -0.022 [-0.049, 0.002] | 0.1221 | 0.4883 |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-G | accepted_repair | few_shot_within_logic_only | 406 | -0.017 [-0.042, 0.005] | 0.21 | 0.6301 |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-L | accepted_repair | context_within_few_shot | 249 | -0.048 [-0.080, -0.016] | 0.007538 | 0.01508 |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-L | accepted_repair | context_within_zero_shot | 249 | 0.000 [-0.060, 0.056] | 1 | 1 |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-L | accepted_repair | few_shot_within_local_graph | 249 | 0.169 [0.116, 0.221] | 1.313e-10 | 3.938e-10 |
| main-1200 | ollama_qwen3_30b | a_box_repair | IC-L | accepted_repair | few_shot_within_logic_only | 249 | 0.217 [0.161, 0.273] | 2.592e-13 | 1.037e-12 |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | context_within_few_shot | 225 | -0.027 [-0.089, 0.036] | 0.4885 | 0.9769 |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | context_within_zero_shot | 225 | -0.004 [-0.013, 0.000] | 1 | 1 |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | few_shot_within_local_graph | 225 | 0.160 [0.102, 0.222] | 4.039e-07 | 1.212e-06 |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | few_shot_within_logic_only | 225 | 0.182 [0.129, 0.240] | 8.225e-10 | 3.29e-09 |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | context_within_few_shot | 225 | -0.102 [-0.169, -0.040] | 0.002667 | 0.005335 |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | context_within_zero_shot | 225 | -0.036 [-0.080, 0.004] | 0.1516 | 0.1516 |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | few_shot_within_local_graph | 225 | -0.373 [-0.458, -0.284] | 1.255e-14 | 5.019e-14 |
| main-1200 | ollama_qwen3_30b | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | few_shot_within_logic_only | 225 | -0.307 [-0.391, -0.227] | 4.946e-12 | 1.484e-11 |
| main-1200 | ollama_qwen3_30b | track_diagnosis | ALL | accuracy | context_within_few_shot | 1200 | -0.034 [-0.062, -0.007] | 0.01766 | 0.07065 |
| main-1200 | ollama_qwen3_30b | track_diagnosis | ALL | accuracy | context_within_zero_shot | 1200 | -0.006 [-0.033, 0.020] | 0.7039 | 1 |
| main-1200 | ollama_qwen3_30b | track_diagnosis | ALL | accuracy | few_shot_within_local_graph | 1200 | -0.028 [-0.053, -0.001] | 0.04998 | 0.1499 |
| main-1200 | ollama_qwen3_30b | track_diagnosis | ALL | accuracy | few_shot_within_logic_only | 1200 | 0.001 [-0.028, 0.028] | 1 | 1 |
| main-1200 | ollama_qwen3_30b | track_diagnosis | A_BOX | accuracy | context_within_few_shot | 975 | -0.042 [-0.072, -0.013] | 0.006487 | 0.006487 |
| main-1200 | ollama_qwen3_30b | track_diagnosis | A_BOX | accuracy | context_within_zero_shot | 975 | -0.052 [-0.080, -0.026] | 0.0002359 | 0.0004719 |
| main-1200 | ollama_qwen3_30b | track_diagnosis | A_BOX | accuracy | few_shot_within_local_graph | 975 | -0.055 [-0.083, -0.028] | 0.0001293 | 0.0003878 |
| main-1200 | ollama_qwen3_30b | track_diagnosis | A_BOX | accuracy | few_shot_within_logic_only | 975 | -0.066 [-0.093, -0.038] | 5.671e-06 | 2.269e-05 |
| main-1200 | ollama_qwen3_30b | track_diagnosis | T_BOX | accuracy | context_within_few_shot | 225 | 0.000 [-0.071, 0.071] | 1 | 1 |
| main-1200 | ollama_qwen3_30b | track_diagnosis | T_BOX | accuracy | context_within_zero_shot | 225 | 0.196 [0.133, 0.262] | 1.051e-08 | 3.152e-08 |
| main-1200 | ollama_qwen3_30b | track_diagnosis | T_BOX | accuracy | few_shot_within_local_graph | 225 | 0.093 [0.018, 0.164] | 0.01862 | 0.03724 |
| main-1200 | ollama_qwen3_30b | track_diagnosis | T_BOX | accuracy | few_shot_within_logic_only | 225 | 0.289 [0.218, 0.360] | 1.858e-13 | 7.433e-13 |
