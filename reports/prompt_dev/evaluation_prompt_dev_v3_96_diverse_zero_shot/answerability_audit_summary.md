# Answerability Audit Summary

Rows: `192`

## Counts

- By track: `{"A_BOX": 96, "T_BOX": 96}`
- By context: `{"local_graph": 96, "logic_only": 96}`
- By expected behavior: `{"abstain": 20, "conservative_remove": 6, "diagnostic_only": 46, "exact_repair": 48, "schema_update_low_confidence": 72}`
- By proposal status: `{"accepted_non_exact": 22, "exact": 37, "overdelete": 10, "parse_error": 2, "request_error": 2, "wrong_operation": 63, "wrong_tbox_family": 12, "wrong_value": 44}`

## Exact/Accepted By Visibility

| Gold target visible | N | Exact | Exact rate | Accepted/non-exact | Accepted/non-exact rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `true` | 68 | 37 | 0.544 | 37 | 0.544 |
| `false` | 124 | 0 | 0.000 | 22 | 0.177 |
