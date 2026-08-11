# Azure Calibration

Primary endpoint estimates; models and repair loci are not pooled.

| Population | Model | Task | Stratum | Endpoint | Regime | Context | n | Case micro (95% CI) | Event macro (95% CI) |
|---|---|---|---|---|---|---|---:|---:|---:|
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-E-elim | accepted_repair | static_few_shot | local_graph | 147 | 0.048 [0.020, 0.082] | 0.048 [0.020, 0.082] |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-E-elim | accepted_repair | static_few_shot | logic_only | 147 | 0.041 [0.014, 0.075] | 0.041 [0.014, 0.075] |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-E-elim | accepted_repair | zero_shot | local_graph | 147 | 0.048 [0.020, 0.082] | 0.048 [0.020, 0.082] |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-E-elim | accepted_repair | zero_shot | logic_only | 147 | 0.041 [0.014, 0.075] | 0.041 [0.014, 0.075] |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-G | accepted_repair | static_few_shot | local_graph | 188 | 0.165 [0.117, 0.218] | 0.165 [0.117, 0.218] |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-G | accepted_repair | static_few_shot | logic_only | 188 | 0.160 [0.112, 0.213] | 0.160 [0.112, 0.213] |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-G | accepted_repair | zero_shot | local_graph | 188 | 0.122 [0.080, 0.170] | 0.122 [0.080, 0.170] |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-G | accepted_repair | zero_shot | logic_only | 188 | 0.122 [0.080, 0.170] | 0.122 [0.080, 0.170] |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-L | accepted_repair | static_few_shot | local_graph | 115 | 0.957 [0.913, 0.991] | 0.957 [0.913, 0.991] |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-L | accepted_repair | static_few_shot | logic_only | 115 | 0.974 [0.939, 1.000] | 0.974 [0.939, 1.000] |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-L | accepted_repair | zero_shot | local_graph | 115 | 0.957 [0.913, 0.991] | 0.957 [0.913, 0.991] |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-L | accepted_repair | zero_shot | logic_only | 115 | 0.983 [0.957, 1.000] | 0.983 [0.957, 1.000] |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | static_few_shot | local_graph | 150 | 0.513 [0.440, 0.593] | 0.513 [0.440, 0.593] |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | static_few_shot | logic_only | 150 | 0.527 [0.447, 0.607] | 0.527 [0.447, 0.607] |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | zero_shot | local_graph | 150 | 0.093 [0.053, 0.140] | 0.093 [0.053, 0.140] |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | zero_shot | logic_only | 150 | 0.100 [0.053, 0.153] | 0.100 [0.053, 0.153] |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | static_few_shot | local_graph | 150 | 0.580 [0.500, 0.660] | 0.580 [0.500, 0.660] |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | static_few_shot | logic_only | 150 | 0.607 [0.527, 0.680] | 0.607 [0.527, 0.680] |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | zero_shot | local_graph | 150 | 0.600 [0.520, 0.680] | 0.600 [0.520, 0.680] |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | zero_shot | logic_only | 150 | 0.600 [0.520, 0.680] | 0.600 [0.520, 0.680] |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | ALL | accuracy | static_few_shot | local_graph | 600 | 0.563 [0.523, 0.602] | 0.563 [0.523, 0.602] |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | ALL | accuracy | static_few_shot | logic_only | 600 | 0.553 [0.513, 0.593] | 0.553 [0.513, 0.593] |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | ALL | accuracy | zero_shot | local_graph | 600 | 0.502 [0.462, 0.543] | 0.502 [0.462, 0.543] |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | ALL | accuracy | zero_shot | logic_only | 600 | 0.508 [0.468, 0.548] | 0.508 [0.468, 0.548] |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | A_BOX | accuracy | static_few_shot | local_graph | 450 | 0.622 [0.578, 0.664] | 0.622 [0.578, 0.664] |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | A_BOX | accuracy | static_few_shot | logic_only | 450 | 0.629 [0.587, 0.671] | 0.629 [0.587, 0.671] |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | A_BOX | accuracy | zero_shot | local_graph | 450 | 0.631 [0.589, 0.676] | 0.631 [0.589, 0.676] |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | A_BOX | accuracy | zero_shot | logic_only | 450 | 0.636 [0.591, 0.678] | 0.636 [0.591, 0.678] |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | T_BOX | accuracy | static_few_shot | local_graph | 150 | 0.387 [0.307, 0.467] | 0.387 [0.307, 0.467] |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | T_BOX | accuracy | static_few_shot | logic_only | 150 | 0.327 [0.253, 0.400] | 0.327 [0.253, 0.400] |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | T_BOX | accuracy | zero_shot | local_graph | 150 | 0.113 [0.067, 0.167] | 0.113 [0.067, 0.167] |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | T_BOX | accuracy | zero_shot | logic_only | 150 | 0.127 [0.073, 0.180] | 0.127 [0.073, 0.180] |

## Paired primary contrasts

Effects are right minus left. Holm adjustment is within each model × task × stratum × endpoint family.

| Population | Model | Task | Stratum | Endpoint | Contrast | n | Micro difference (95% CI) | Exact p | Holm p |
|---|---|---|---|---|---|---:|---:|---:|---:|
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-E-elim | accepted_repair | context_within_few_shot | 147 | 0.007 [0.000, 0.020] | 1 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-E-elim | accepted_repair | context_within_zero_shot | 147 | 0.007 [0.000, 0.020] | 1 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-E-elim | accepted_repair | few_shot_within_local_graph | 147 | 0.000 [-0.020, 0.020] | 1 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-E-elim | accepted_repair | few_shot_within_logic_only | 147 | 0.000 [-0.020, 0.020] | 1 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-G | accepted_repair | context_within_few_shot | 188 | 0.005 [-0.027, 0.043] | 1 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-G | accepted_repair | context_within_zero_shot | 188 | 0.000 [-0.027, 0.027] | 1 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-G | accepted_repair | few_shot_within_local_graph | 188 | 0.043 [0.011, 0.074] | 0.02148 | 0.06445 |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-G | accepted_repair | few_shot_within_logic_only | 188 | 0.037 [0.016, 0.064] | 0.01562 | 0.0625 |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-L | accepted_repair | context_within_few_shot | 115 | -0.017 [-0.052, 0.017] | 0.625 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-L | accepted_repair | context_within_zero_shot | 115 | -0.026 [-0.070, 0.009] | 0.375 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-L | accepted_repair | few_shot_within_local_graph | 115 | 0.000 [-0.026, 0.026] | 1 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | a_box_repair | IC-L | accepted_repair | few_shot_within_logic_only | 115 | -0.009 [-0.043, 0.017] | 1 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | context_within_few_shot | 150 | -0.013 [-0.060, 0.033] | 0.7905 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | context_within_zero_shot | 150 | -0.007 [-0.060, 0.047] | 1 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | few_shot_within_local_graph | 150 | 0.420 [0.327, 0.513] | 9.691e-14 | 2.907e-13 |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_schema_decision_match_rate | few_shot_within_logic_only | 150 | 0.427 [0.333, 0.520] | 6.312e-15 | 2.525e-14 |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | context_within_few_shot | 150 | -0.027 [-0.073, 0.020] | 0.3877 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | context_within_zero_shot | 150 | 0.000 [0.000, 0.000] | 1 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | few_shot_within_local_graph | 150 | -0.020 [-0.100, 0.060] | 0.7493 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | t_box_repair | TBOX | tbox_patch_taxonomy_code_exact_match_rate | few_shot_within_logic_only | 150 | 0.007 [-0.067, 0.073] | 1 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | ALL | accuracy | context_within_few_shot | 600 | 0.010 [-0.015, 0.035] | 0.5258 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | ALL | accuracy | context_within_zero_shot | 600 | -0.007 [-0.035, 0.022] | 0.7275 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | ALL | accuracy | few_shot_within_local_graph | 600 | 0.062 [0.032, 0.092] | 5.974e-05 | 0.0002389 |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | ALL | accuracy | few_shot_within_logic_only | 600 | 0.045 [0.013, 0.075] | 0.005545 | 0.01663 |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | A_BOX | accuracy | context_within_few_shot | 450 | -0.007 [-0.033, 0.020] | 0.7493 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | A_BOX | accuracy | context_within_zero_shot | 450 | -0.004 [-0.036, 0.027] | 0.8899 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | A_BOX | accuracy | few_shot_within_local_graph | 450 | -0.009 [-0.038, 0.016] | 0.6271 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | A_BOX | accuracy | few_shot_within_logic_only | 450 | -0.007 [-0.038, 0.022] | 0.7709 | 1 |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | T_BOX | accuracy | context_within_few_shot | 150 | 0.060 [0.000, 0.120] | 0.09314 | 0.1863 |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | T_BOX | accuracy | context_within_zero_shot | 150 | -0.013 [-0.073, 0.047] | 0.8318 | 0.8318 |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | T_BOX | accuracy | few_shot_within_local_graph | 150 | 0.273 [0.200, 0.347] | 5.889e-11 | 2.356e-10 |
| azure-600 | azure_gpt_5_6_sol_high | track_diagnosis | T_BOX | accuracy | few_shot_within_logic_only | 150 | 0.200 [0.120, 0.280] | 2.829e-06 | 8.487e-06 |
