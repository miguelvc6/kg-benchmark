# v4 Spec-Only Holdout Report

Final verdict: **FREEZE_V4_SPEC_ONLY_FOR_PHASE_G_DRY_RUN**

Source run: `reports/prompt_dev/evaluation_prompt_dev_v4_spec_only_holdout96_ollama_zero_shot`
Provider/model: `ollama` / `gpt-oss:120b`

This report uses the completed Ollama/gpt-oss run. It is written into the originally requested `...holdout96_zero_shot` report path as well as the source Ollama run directory.

## Gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Completion | PASS | 384/384 prompts evaluated |
| Request errors | PASS | 0 total; proposal request error rate 0.0% |
| Parse errors | WARNING | 6/384 total (1.6%); proposal parse rate 3.1% |
| Leakage | PASS | no hits for raw case IDs, hidden class names, scaffolded strategy wording, `sitelinks_count`, or temporal-policy labels |
| Spec-only wording | PASS | v4 prompt remains the clean task specification; no v3 scaffolded wording detected |

## Matrix Metrics

| Matrix | Context | Task | Parse errors | Request errors | Accepted | Exact hist. | Track acc. | A-box value | T-box family |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain` | `logic_only` | `track_diagnosis` | 0 | 0 | 0.0% | 0.0% | 51.0% | 0.0% | 0.0% |
| `prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain` | `logic_only` | `repair_proposal` | 4 | 0 | 20.8% | 20.8% | 0.0% | 41.7% | 14.6% |
| `prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain` | `local_graph` | `track_diagnosis` | 0 | 0 | 0.0% | 0.0% | 47.9% | 0.0% | 0.0% |
| `prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain` | `local_graph` | `repair_proposal` | 2 | 0 | 22.9% | 22.9% | 0.0% | 45.8% | 16.7% |

## Answerability-Aware Rates

| Gold target visible | N | Exact | Exact rate | Accepted/non-exact | Accepted/non-exact rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `false` | 127 | 4 | 3.1% | 19 | 15.0% |
| `true` | 65 | 38 | 58.5% | 38 | 58.5% |

Interpretation: answerable A-box rows are substantially better than non-visible rows, but still only `58.5%` exact/accepted. Non-visible rows have low exactness as expected; TypeC remains a strong case for a separate abstention branch rather than a main-prompt rewrite.

## Class Breakdown

| Class | N | Exact | Exact rate | Accepted/non-exact | Accepted/non-exact rate | Dominant failures |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `TypeA` | 48 | 33 | 68.8% | 33 | 68.8% | exact=33, wrong_value=7, overdelete=4 |
| `TypeB` | 24 | 9 | 37.5% | 9 | 37.5% | exact=9, wrong_value=9, overdelete=5 |
| `TypeC` | 24 | 0 | 0.0% | 0 | 0.0% | wrong_value=14, overdelete=6, hallucinated_replacement=2 |
| `T_BOX` | 96 | 0 | 0.0% | 15 | 15.6% | wrong_operation=65, accepted_non_exact=15, wrong_tbox_family=13 |

## Key Failure Shapes

- TypeA answerable failures: `15` rows.
- TypeB local_graph answerable failures: `2` rows.
- TypeC concrete failed repairs: `23/24` rows (95.8%).
- T-box failure shape: `{"accepted_non_exact": 15, "parse_error": 3, "wrong_operation": 65, "wrong_tbox_family": 13}`.
- Track diagnosis remains weak: logic_only `51.0%`, local_graph `47.9%`; use Phase G oracle mode first.

## v3 Scaffolded Comparison

No same-holdout `prompt_dev_v3_scaffolded` ablation was found. Existing v3 artifacts use a different dev manifest/provider and should not be treated as a fair v3-v4 comparison.

## Phase G Dry-Run Decision

`prompt_dev_v4_spec_only` is operationally stable enough for a **Phase G oracle dry run**, not full main scoring yet. The dry run should monitor proposal parse errors and exact output contracts, and should not use `diagnosis_routed` as the main mode.

The main Phase G prompt should remain the clean v4 specification. v3 scaffolded language should stay diagnostic-only even if it scores higher on dev artifacts.

Recommended next command shape:

```bash
MAX_CASES=16 OUTPUT_DIR=reports/reasoning_floor/ollama_v4_spec_only_oracle_dry_run \
  bash scripts/run_phase_g_ollama_oracle.sh
```
