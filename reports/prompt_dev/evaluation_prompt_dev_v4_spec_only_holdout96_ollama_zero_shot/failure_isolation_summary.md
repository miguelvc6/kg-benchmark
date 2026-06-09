# Failure Mode Isolation Summary

Run: `reports/prompt_dev/evaluation_prompt_dev_v4_spec_only_holdout96_ollama_zero_shot`

## Exact/Accepted By Gold Visibility

| Gold target visible | N | Exact rate | Accepted/non-exact rate |
| --- | ---: | ---: | ---: |
| `true` | 65 | 0.585 | 0.585 |
| `false` | 127 | 0.031 | 0.150 |

## Key Failure Counts

- TypeA answerable failures: `15`
- TypeB local_graph answerable failures: `2`
- TypeC concrete failed repairs: `23` / `24`
- T-box failure shape: `{"accepted_non_exact": 15, "parse_error": 3, "wrong_operation": 65, "wrong_tbox_family": 13}`

## Track Diagnosis Confusion By Class

```json
{
  "A_BOX::TypeA": {
    "AMBIGUOUS": 8,
    "A_BOX": 40
  },
  "A_BOX::TypeB": {
    "A_BOX": 24
  },
  "A_BOX::TypeC": {
    "AMBIGUOUS": 1,
    "A_BOX": 22,
    "T_BOX": 1
  },
  "T_BOX::T_BOX": {
    "AMBIGUOUS": 9,
    "A_BOX": 78,
    "T_BOX": 9
  }
}
```

## Explicit Answers

### Are TypeA clean/rule cases failing despite answerable evidence?

Yes. Answerable TypeA rows still fail exact repair in 15 of 48 repair-prompt rows.

### Are TypeB local_graph cases failing despite visible local evidence?

Yes. Local-graph TypeB answerable rows still fail exact repair in 2 of 5 rows.

### Are TypeC cases being forced into hallucinated concrete repair?

Yes. TypeC rows are rarely answerable and concrete failed repairs dominate: 23 of 24 TypeC repair-prompt rows.

### Are T-box failures mostly wrong family, wrong action, or impossible exact signature?

They are mostly impossible exact-signature cases under compact temporal context, with remaining failures split as {'wrong_operation': 65, 'wrong_tbox_family': 13, 'parse_error': 3, 'accepted_non_exact': 15}. Target-family/action errors still exist but signature construction is intentionally not visible.

### Are evaluator metrics too strict for plausible non-exact T-box schema updates?

Yes for exact historical agreement: compact-policy T-box prompts cannot infer exact signature_after. Family/action metrics are more scientifically meaningful for these rows.

## Deliverable Verdict

`IMPLEMENT_TARGETED_V4`
