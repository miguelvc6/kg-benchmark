# Failure Mode Isolation Summary

Run: `reports/prompt_dev/evaluation_prompt_dev_v3_96_diverse_zero_shot`

## Exact/Accepted By Gold Visibility

| Gold target visible | N | Exact rate | Accepted/non-exact rate |
| --- | ---: | ---: | ---: |
| `true` | 68 | 0.544 | 0.544 |
| `false` | 124 | 0.000 | 0.177 |

## Key Failure Counts

- TypeA answerable failures: `17`
- TypeB local_graph answerable failures: `4`
- TypeC concrete failed repairs: `20` / `22`
- T-box failure shape: `{"accepted_non_exact": 22, "parse_error": 2, "wrong_operation": 60, "wrong_tbox_family": 12}`

## Track Diagnosis Confusion By Class

```json
{
  "A_BOX::TypeA": {
    "A_BOX": 47,
    "T_BOX": 1
  },
  "A_BOX::TypeB": {
    "A_BOX": 26
  },
  "A_BOX::TypeC": {
    "A_BOX": 19,
    "T_BOX": 3
  },
  "T_BOX::T_BOX": {
    "AMBIGUOUS": 6,
    "A_BOX": 71,
    "T_BOX": 19
  }
}
```

## Explicit Answers

### Are TypeA clean/rule cases failing despite answerable evidence?

Yes. Answerable TypeA rows still fail exact repair in 17 of 48 repair-prompt rows.

### Are TypeB local_graph cases failing despite visible local evidence?

Yes. Local-graph TypeB answerable rows still fail exact repair in 4 of 7 rows.

### Are TypeC cases being forced into hallucinated concrete repair?

Yes. TypeC rows are rarely answerable and concrete failed repairs dominate: 20 of 22 TypeC repair-prompt rows.

### Are T-box failures mostly wrong family, wrong action, or impossible exact signature?

They are mostly impossible exact-signature cases under compact temporal context, with remaining failures split as {'wrong_operation': 60, 'parse_error': 2, 'accepted_non_exact': 22, 'wrong_tbox_family': 12}. Target-family/action errors still exist but signature construction is intentionally not visible.

### Are evaluator metrics too strict for plausible non-exact T-box schema updates?

Yes for exact historical agreement: compact-policy T-box prompts cannot infer exact signature_after. Family/action metrics are more scientifically meaningful for these rows.

## Deliverable Verdict

`IMPLEMENT_TARGETED_V4`
