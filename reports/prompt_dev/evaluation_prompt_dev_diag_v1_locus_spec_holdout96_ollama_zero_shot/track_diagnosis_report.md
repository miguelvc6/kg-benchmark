# Track Diagnosis Report

This report is dev-only and gates whether diagnosis-routed repair canaries are interpretable.

Acceptance gates:
- Request error rate <= 1.00%; parse error rate <= 4.00%; balanced accuracy >= 0.70; A_BOX/T_BOX recall >= 0.65/0.65; AMBIGUOUS rate <= 15.00%.

| Matrix | Context | Track mode | Total | A recall | T recall | Balanced acc | Macro-F1 | Ambiguous | Wrong-route | Parse err | Request err | Gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain` | `logic_only` | `` | 96 | 0.896 | 0.062 | 0.479 | 0.398 | 0.135 | 0.385 | 0.000 | 0.000 | FAIL |
| `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain` | `local_graph` | `` | 96 | 0.833 | 0.042 | 0.438 | 0.373 | 0.198 | 0.365 | 0.000 | 0.000 | FAIL |

## Confusion Matrices

### `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain`

```json
{
  "A_BOX": {
    "AMBIGUOUS": 3,
    "A_BOX": 43,
    "T_BOX": 2
  },
  "T_BOX": {
    "AMBIGUOUS": 10,
    "A_BOX": 35,
    "T_BOX": 3
  }
}
```

### `prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain`

```json
{
  "A_BOX": {
    "AMBIGUOUS": 4,
    "A_BOX": 40,
    "T_BOX": 4
  },
  "T_BOX": {
    "AMBIGUOUS": 15,
    "A_BOX": 31,
    "T_BOX": 2
  }
}
```
