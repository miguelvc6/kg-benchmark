# Answerability Audit Summary

Rows: `192`

## Counts

- By track: `{"A_BOX": 96, "T_BOX": 96}`
- By context: `{"local_graph": 96, "logic_only": 96}`
- By expected behavior: `{"abstain": 23, "conservative_remove": 2, "diagnostic_only": 48, "exact_repair": 47, "schema_update_low_confidence": 72}`
- By proposal status: `{"accepted_non_exact": 15, "exact": 42, "hallucinated_replacement": 3, "overdelete": 15, "parse_error": 6, "wrong_operation": 68, "wrong_tbox_family": 13, "wrong_value": 30}`

## Exact/Accepted By Visibility

| Gold target visible | N | Exact | Exact rate | Accepted/non-exact | Accepted/non-exact rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `true` | 65 | 38 | 0.585 | 38 | 0.585 |
| `false` | 127 | 4 | 0.031 | 19 | 0.150 |
