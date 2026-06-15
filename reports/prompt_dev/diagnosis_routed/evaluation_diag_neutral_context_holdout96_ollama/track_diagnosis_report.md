# Track Diagnosis Report

This report is dev-only and gates whether diagnosis-routed repair canaries are interpretable.

Acceptance gates:
- Request error rate <= 1.00%; parse error rate <= 4.00%; balanced accuracy >= 0.70; A_BOX/T_BOX recall >= 0.65/0.65; AMBIGUOUS rate <= 15.00%.

| Matrix | Context | Track mode | Total | A recall | T recall | Balanced acc | Macro-F1 | Ambiguous | Wrong-route | Parse err | Request err | Gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `prompt_dev_001_hybrid_json_nl_zero_shot_diagnosis_minimal_track_diagnosis_diagnosis_no_abstain` | `diagnosis_minimal` | `` | 96 | 0.896 | 0.062 | 0.479 | 0.399 | 0.146 | 0.375 | 0.000 | 0.000 | FAIL |
| `prompt_dev_002_hybrid_json_nl_zero_shot_diagnosis_logic_neutral_track_diagnosis_diagnosis_no_abstain` | `diagnosis_logic_neutral` | `` | 96 | 0.833 | 0.125 | 0.479 | 0.433 | 0.115 | 0.396 | 0.010 | 0.000 | FAIL |
| `prompt_dev_003_hybrid_json_nl_zero_shot_diagnosis_local_neutral_track_diagnosis_diagnosis_no_abstain` | `diagnosis_local_neutral` | `` | 96 | 0.896 | 0.083 | 0.490 | 0.417 | 0.104 | 0.396 | 0.010 | 0.000 | FAIL |

## Confusion Matrices

### `prompt_dev_001_hybrid_json_nl_zero_shot_diagnosis_minimal_track_diagnosis_diagnosis_no_abstain`

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

### `prompt_dev_002_hybrid_json_nl_zero_shot_diagnosis_logic_neutral_track_diagnosis_diagnosis_no_abstain`

```json
{
  "A_BOX": {
    "AMBIGUOUS": 2,
    "A_BOX": 40,
    "T_BOX": 6
  },
  "T_BOX": {
    "AMBIGUOUS": 9,
    "A_BOX": 32,
    "PARSE_ERROR": 1,
    "T_BOX": 6
  }
}
```

### `prompt_dev_003_hybrid_json_nl_zero_shot_diagnosis_local_neutral_track_diagnosis_diagnosis_no_abstain`

```json
{
  "A_BOX": {
    "A_BOX": 43,
    "T_BOX": 5
  },
  "T_BOX": {
    "AMBIGUOUS": 10,
    "A_BOX": 33,
    "PARSE_ERROR": 1,
    "T_BOX": 4
  }
}
```
