# Step 12B T-box Taxonomy-Patch Gate Report

Status: **PASS_WITH_REVIEW**

Review flags: `value_delta_false_positive_increase_gt_5pp`

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
| family success improves 5pp | pass |
| schema decision match improves 5pp | pass |
| taxonomy level success does not regress | pass |
| taxonomy code exact regression lte 5pp or review | pass |
| value delta false positive increase lte 5pp or review | pass |
| token delta documented | pass |

## Delta By Context

| Context | Family | Schema decision | Taxonomy level | Taxonomy code | Value-delta FP | Parse error | Total-token delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| local_graph | +0.0 pp | +16.4 pp | +5.5 pp | +7.8 pp | +19.0 pp | +0.0 pp | 655124.000 |
| logic_only | +14.1 pp | +28.1 pp | +18.0 pp | +13.3 pp | +7.9 pp | +0.0 pp | 408865.000 |

Leakage and overlap scans passed. Same-case zero-shot baseline comparison passed.
