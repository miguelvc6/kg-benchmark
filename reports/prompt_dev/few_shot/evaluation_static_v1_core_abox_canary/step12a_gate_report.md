# Step 12A A-box Gate Report

Status: **PASS**

## Hard Gates

| Gate | Result |
|---|---:|
| local only provider | pass |
| request error lte 1pct | pass |
| proposal parse error lte 4pct | pass |
| leakage scan pass | pass |
| overlap scan pass | pass |
| correct schema outputs present | pass |
| delta report same case zero shot | pass |

## Effect Gates

| Gate | Result |
|---|---:|
| accepted or exact value improves 3pp in at least one context | pass |
| no exact value regression gt 2pp | pass |
| typec hallucination empty op not materially worse | pass |
| token delta documented | pass |

## Delta By Context

| Context | Accepted | Exact value | TypeC accepted | TypeC empty-op | Parse error | Total-token delta |
|---|---:|---:|---:|---:|---:|---:|
| local_graph | +5.9 pp | +6.2 pp | +0.0 pp | -3.5 pp | +1.2 pp | 1062276.000 |
| logic_only | +2.7 pp | +2.3 pp | +1.2 pp | -0.6 pp | +0.4 pp | 644829.000 |

Leakage and overlap scans passed. Same-case zero-shot baseline comparison passed.
