# Prompt v3 Diverse Canary Delta

Verdict: **CONTINUE PHASE F; DO NOT MOVE TO PHASE G MAIN SCORING**

Run: `reports/prompt_dev/evaluation_prompt_dev_v3_96_diverse_zero_shot`
Provider/model: `university` / `Qwen/Qwen3-4B`
Run id: `prompt_dev_eval_20260607T202605`

## Scope

- Cases: 96 (`A_BOX` 48, `T_BOX` 48)
- Diversity: 96 unique focus QIDs, 81 unique properties
- Classes: `{"T_BOX": 48, "TypeA": 24, "TypeB": 13, "TypeC": 11}`
- Prompts evaluated: 384

## Stability Gates

| Gate | Result |
| --- | --- |
| Proposal parse error rate | 2/192 = 1.0% |
| Request error rate | 2/384 = 0.5% |
| Normalized rows | 380/384 = 99.0% |
| Prompt text leakage | PASS `{}` |
| Raw `<think>` preambles | 382/384 = 99.5% |

## Matrix Metrics

| Context | Task | Normalized | Parse errors | Request errors | Track acc | Accepted | A-box exact value | T-box semantic family | Audit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logic_only | track_diagnosis | 96 | 0 | 0 | 56.2% | n/a | n/a | n/a | n/a |
| local_graph | track_diagnosis | 96 | 0 | 0 | 59.4% | n/a | n/a | n/a | n/a |
| logic_only | repair_proposal | 93 | 2 | 1 | n/a | 17.7% | 35.4% | 22.9% | 96.9% |
| local_graph | repair_proposal | 95 | 0 | 1 | n/a | 20.8% | 41.7% | 22.9% | 99.0% |

## Failure-Mode Delta

The comparison is cautious: v1 and v3 are both 96-case runs, but v3 uses `diverse_stratified`; v2 is only a 48-case canary.

| Failure mode | v1_96 baseline | v2_48 canary | v3_96 diverse | Interpretation |
| --- | ---: | ---: | ---: | --- |
| A-box constraint/type QID as value | 29.2% | 25.0% | 2/96 = 2.1% | lower is better |
| A-box over-delete heuristic | 40.6% | 43.8% | 10/96 DELETE_ALL = 10.4% | v3 reports DELETE_ALL separately |
| TypeC concrete failed repair | 22.9% | 20.8% | 20/96 = 20.8% overall; 20/22 TypeC rows = 90.9% | still high means abstention needed |
| T-box unsupported directional action | 44.8% | 2.1% | 0/96 = 0.0% | v3 should remain low |
| T-box invented signature_after | 63.5% | 2.1% | 0/96 = 0.0% | v3 should remain low |
| T-box report-QID copy | 6.2% | 2.1% | 4/96 = 4.2% | v3 should remain low |

## A-Box Exact Value By Subtype

### logic_only

| Subtype | Exact value | N | Rate |
| --- | ---: | ---: | ---: |
| `DELETE_AMBIGUOUS` | 3 | 3 | 100.0% |
| `EXTERNAL_BY_ELIMINATION` | 0 | 3 | 0.0% |
| `FORMAT_NORMALIZATION` | 0 | 3 | 0.0% |
| `FORMAT_VALUE_PRUNING` | 2 | 3 | 66.7% |
| `LOCAL_FOCUS_NON_TARGET_PROPERTY` | 0 | 3 | 0.0% |
| `LOCAL_MIXED` | 0 | 3 | 0.0% |
| `LOCAL_SELECTION_CONFIRMED` | 1 | 3 | 33.3% |
| `LOCAL_TEXT_CONFIRMED` | 0 | 1 | 0.0% |
| `LOCAL_TEXT_DERIVED` | 0 | 3 | 0.0% |
| `MULTIPLICITY_NORMALIZATION` | 0 | 3 | 0.0% |
| `REJECTION_FORMAT_INVALID` | 3 | 3 | 100.0% |
| `SELF_LINK_REJECTION` | 2 | 3 | 66.7% |
| `SET_MEMBERSHIP_REJECTION` | 3 | 3 | 100.0% |
| `TARGET_REQUIRED_CLAIM` | 2 | 3 | 66.7% |
| `UNKNOWN_BAD_TARGET_OR_CONTEXT` | 0 | 3 | 0.0% |
| `UNKNOWN_INCOMPLETE_LOCAL_CONTEXT` | 0 | 1 | 0.0% |
| `UNKNOWN_MULTIPLICITY_ARTIFACT` | 0 | 1 | 0.0% |
| `UNKNOWN_SELECTION_AMBIGUOUS` | 1 | 3 | 33.3% |

### local_graph

| Subtype | Exact value | N | Rate |
| --- | ---: | ---: | ---: |
| `DELETE_AMBIGUOUS` | 3 | 3 | 100.0% |
| `EXTERNAL_BY_ELIMINATION` | 0 | 3 | 0.0% |
| `FORMAT_NORMALIZATION` | 0 | 3 | 0.0% |
| `FORMAT_VALUE_PRUNING` | 1 | 3 | 33.3% |
| `LOCAL_FOCUS_NON_TARGET_PROPERTY` | 0 | 3 | 0.0% |
| `LOCAL_MIXED` | 0 | 3 | 0.0% |
| `LOCAL_SELECTION_CONFIRMED` | 3 | 3 | 100.0% |
| `LOCAL_TEXT_CONFIRMED` | 0 | 1 | 0.0% |
| `LOCAL_TEXT_DERIVED` | 0 | 3 | 0.0% |
| `MULTIPLICITY_NORMALIZATION` | 1 | 3 | 33.3% |
| `REJECTION_FORMAT_INVALID` | 3 | 3 | 100.0% |
| `SELF_LINK_REJECTION` | 3 | 3 | 100.0% |
| `SET_MEMBERSHIP_REJECTION` | 3 | 3 | 100.0% |
| `TARGET_REQUIRED_CLAIM` | 2 | 3 | 66.7% |
| `UNKNOWN_BAD_TARGET_OR_CONTEXT` | 0 | 3 | 0.0% |
| `UNKNOWN_INCOMPLETE_LOCAL_CONTEXT` | 0 | 1 | 0.0% |
| `UNKNOWN_MULTIPLICITY_ARTIFACT` | 0 | 1 | 0.0% |
| `UNKNOWN_SELECTION_AMBIGUOUS` | 1 | 3 | 33.3% |

## Track Diagnosis Failure Shape

| Context | Correct | A-box predicted T-box | T-box predicted A-box | Ambiguous |
| --- | ---: | ---: | ---: | ---: |
| logic_only | 54/96 (56.2%) | 2 | 38 | 2 |
| local_graph | 57/96 (59.4%) | 2 | 33 | 4 |

## Recommendation

- Do not move to Phase G main scoring: repair proposal quality remains weak even with the broader diverse v3 sample.
- Keep the v3/v2 T-box signature discipline; it appears to have fixed the main invented-signature failure mode.
- Keep the v3 A-box targeted-remove direction, but do not freeze it yet: constraint/type-QID misuse fell sharply and
  `DELETE_ALL` use is much lower, while exact value quality and TypeC behavior remain weak.
- Track diagnosis is still below the v1_96 sanity level; do another Phase F prompt iteration before freezing.
- Provider still emits <think> preambles despite prompt instructions; parser handles them, but this should be reported separately from prompt JSON compliance.

## Files Written

- `reports/prompt_dev/evaluation_prompt_dev_v3_96_diverse_zero_shot/prompt_v3_diverse_delta.json`
- `reports/prompt_dev/evaluation_prompt_dev_v3_96_diverse_zero_shot/prompt_v3_diverse_delta.md`
