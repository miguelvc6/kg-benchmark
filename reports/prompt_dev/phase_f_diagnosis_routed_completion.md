# Phase F Diagnosis-Routed Completion

Verdict: `PHASE_F_DIAGNOSIS_BRANCH_COMPLETE_ROUTED_BLOCKED`

I completed the dev-only diagnosis branch with local Ollama inference on the H100 VM. The run used `prompt_dev_diag_v1_locus_spec`, `gpt-oss:120b`, 96 holdout cases, and two context bundles (`logic_only`, `local_graph`), for 192 total diagnosis prompts.

The pipeline is operational: all 192 responses normalized, with request error rate `0.0` and parse error rate `0.0`. I also patched prompt-dev case-id attribution so shortened neutral IDs such as `case_0003` are attributed back to the request case before normalization/evaluation. The completed VM artifact was repaired and rescored without rerunning LLM inference.

## Gate Results

| Context | A recall | T recall | Balanced acc | Macro-F1 | Ambiguous | Wrong route | Request err | Parse err | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `logic_only` | 0.8958 | 0.0625 | 0.4792 | 0.3979 | 0.1354 | 0.3854 | 0.0000 | 0.0000 | FAIL |
| `local_graph` | 0.8333 | 0.0417 | 0.4375 | 0.3732 | 0.1979 | 0.3646 | 0.0000 | 0.0000 | FAIL |

Acceptance gates were:

- request error rate `<= 1%`
- parse error rate `<= 4%`
- balanced accuracy `>= 0.70`
- A-box recall `>= 0.65`
- T-box recall `>= 0.65`
- `AMBIGUOUS` rate `<= 15%`

Both matrices failed the balanced-accuracy and T-box-recall gates. `local_graph` also failed the ambiguity cap.

## Confusion

`logic_only`:

```json
{
  "A_BOX": {"A_BOX": 43, "AMBIGUOUS": 3, "T_BOX": 2},
  "T_BOX": {"A_BOX": 35, "AMBIGUOUS": 10, "T_BOX": 3}
}
```

`local_graph`:

```json
{
  "A_BOX": {"A_BOX": 40, "AMBIGUOUS": 4, "T_BOX": 4},
  "T_BOX": {"A_BOX": 31, "AMBIGUOUS": 15, "T_BOX": 2}
}
```

## Decision

Do not run the diagnosis-routed Phase F canary and do not run Phase G `diagnosis_routed` on core. The diagnosis model is strongly biased toward `A_BOX` on T-box cases, so routed repair metrics would mostly measure wrong-route failure rather than proposal quality.

Oracle remains the validated Phase G main mode. Diagnosis-routed remains a blocked ablation until a future diagnosis prompt or model passes the dev-only gate.

## Artifact Locations

- VM authoritative run: `reports/prompt_dev/evaluation_prompt_dev_diag_v1_locus_spec_holdout96_ollama_zero_shot`
- Local summary: `reports/prompt_dev/phase_f_diagnosis_routed_completion.json`
- Local report: `reports/prompt_dev/phase_f_diagnosis_routed_completion.md`

Note: VM-to-local bulk artifact transfer hung repeatedly in this session (`tar`, `scp`, and direct `cat` redirects). Code sync local-to-VM works, and the VM inference artifact is present and repaired. Key metrics were queried directly over SSH and recorded in this report.
