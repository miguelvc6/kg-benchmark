# Diagnosis Gate V1 Summary

Verdict: `DIAGNOSIS_GATE_FAILED`

Provider/model: `ollama` / `gpt-oss:120b`
Prompt version: `prompt_dev_diag_v1_locus_spec`
Holdout manifest: `reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json`
Sample: 96 cases, balanced 48 `A_BOX` / 48 `T_BOX`; 192 total prompts across two context bundles.

Routed canary was not run because the diagnosis gate failed.

## Metrics

| Context | A recall | T recall | Balanced acc | Macro-F1 | AMBIGUOUS | Wrong route | Request err | Parse err | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `logic_only` | 0.8958 | 0.0625 | 0.4792 | 0.3979 | 0.1354 | 0.3854 | 0.0000 | 0.0000 | FAIL: balanced_accuracy, t_box_recall |
| `local_graph` | 0.8333 | 0.0417 | 0.4375 | 0.3732 | 0.1979 | 0.3646 | 0.0000 | 0.0000 | FAIL: balanced_accuracy, t_box_recall, ambiguous_rate |

## Confusion Matrices

`logic_only`:

```json
{
  "A_BOX": {
    "A_BOX": 43,
    "AMBIGUOUS": 3,
    "T_BOX": 2
  },
  "T_BOX": {
    "A_BOX": 35,
    "AMBIGUOUS": 10,
    "T_BOX": 3
  }
}
```

`local_graph`:

```json
{
  "A_BOX": {
    "A_BOX": 40,
    "AMBIGUOUS": 4,
    "T_BOX": 4
  },
  "T_BOX": {
    "A_BOX": 31,
    "AMBIGUOUS": 15,
    "T_BOX": 2
  }
}
```

## Baselines

- Frequency-weighted random A/T expected balanced accuracy: `0.5000` on the 48/48 holdout.
- Always `A_BOX`: balanced accuracy `0.5000`, T-box recall `0.0000`.
- Always `T_BOX`: balanced accuracy `0.5000`, A-box recall `0.0000`.
- Uniform `A_BOX`/`T_BOX`/`AMBIGUOUS`: expected balanced accuracy `0.3333`, ambiguity rate `0.3333`.

The observed balanced accuracy is below the simple always-single-label and frequency-weighted random A/T baselines, mainly because T-box recall is near zero and ambiguity consumes additional T-box cases.

## Operational Note

The local copy of the full run artifact is incomplete because VM-to-local bulk transfer hung earlier. These metrics are taken from the authoritative VM run and the existing completion report.
