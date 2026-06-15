# Diagnosis Neutral Context Summary

Verdict: `STOP_DIAGNOSIS_ROUTED_FOR_MODEL`

Provider/model: `ollama` / `gpt-oss:120b`
Prompt version: `prompt_dev_diag_v1_locus_spec`
Holdout manifest: `reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json`
Run scope: dev holdout only; track diagnosis only; no repair proposals; no routed canary.

The neutral context branch removed the identified structural confound, but diagnosis quality remains below even the continuation threshold (`balanced_accuracy >= 0.55` and `T_BOX recall >= 0.25`). Do not run a diagnosis-routed canary and do not run Phase G `diagnosis_routed` for this model.

## Results

| Context | Normalized | Parse err | Request err | A recall | T recall | Balanced acc | Macro-F1 | AMBIGUOUS | Wrong route | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `diagnosis_minimal` | 96 | 0 | 0 | 0.8958 | 0.0625 | 0.4792 | 0.3990 | 0.1458 | 0.3750 | FAIL: balanced_accuracy, t_box_recall |
| `diagnosis_logic_neutral` | 95 | 1 | 0 | 0.8333 | 0.1250 | 0.4792 | 0.4333 | 0.1146 | 0.3958 | FAIL: balanced_accuracy, t_box_recall |
| `diagnosis_local_neutral` | 95 | 1 | 0 | 0.8958 | 0.0833 | 0.4896 | 0.4169 | 0.1042 | 0.3958 | FAIL: balanced_accuracy, t_box_recall |

## Confusion Matrices

`diagnosis_minimal`:

```json
{
  "A_BOX": {
    "AMBIGUOUS": 4,
    "A_BOX": 43,
    "T_BOX": 1
  },
  "T_BOX": {
    "AMBIGUOUS": 10,
    "A_BOX": 35,
    "T_BOX": 3
  }
}
```

`diagnosis_logic_neutral`:

```json
{
  "A_BOX": {
    "A_BOX": 40,
    "T_BOX": 6,
    "AMBIGUOUS": 2
  },
  "T_BOX": {
    "PARSE_ERROR": 1,
    "A_BOX": 32,
    "T_BOX": 6,
    "AMBIGUOUS": 9
  }
}
```

`diagnosis_local_neutral`:

```json
{
  "A_BOX": {
    "T_BOX": 5,
    "A_BOX": 43
  },
  "T_BOX": {
    "PARSE_ERROR": 1,
    "A_BOX": 33,
    "T_BOX": 4,
    "AMBIGUOUS": 10
  }
}
```

## Interpretation

Best balanced accuracy was `0.4896` on `diagnosis_local_neutral` with T-box recall `0.0833`.
The neutral-context rerun does not support a diagnosis prompt v2 under the predeclared rule because no matrix reached balanced accuracy `0.55` and T-box recall `0.25`.
The safest conclusion is that this model/prompt setup is not currently suitable for diagnosis-routed repair routing. Oracle remains the Phase G main mode.

## Artifacts

- Full track diagnosis report: `reports/prompt_dev/diagnosis_routed/evaluation_diag_neutral_context_holdout96_ollama/track_diagnosis_report.json`
- Evaluation summary: `reports/prompt_dev/diagnosis_routed/evaluation_diag_neutral_context_holdout96_ollama/prompt_dev_evaluation_summary.json`
