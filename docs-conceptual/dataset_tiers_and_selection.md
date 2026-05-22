# Dataset Tiers and Selection Policy

**Project:** WikidataRepairEval 1.0  
**Policy version:** `phase_c_v1`  
**Seed:** `13`  
**Purpose:** Define deterministic full/core/dev/audit tiers for LLM experiments, prompt development, and manual audit.

## 1. Why selection manifests are required

The full WikidataRepairEval benchmark is intentionally large and historically faithful, but it is not suitable as the default LLM experiment set. The Phase B classifier snapshot contains 535,570 records, including 456,594 T-box records and 78,976 A-box records. T-box records are repeated manifestations of property-level schema edits, so naive full-dataset averages would overrepresent a small number of schema-change events.

The paper should therefore use deterministic selection manifests instead of creating alternative Stage 4 files. A manifest keeps the full benchmark canonical while making experimental subsets reproducible.

## 2. Tier definitions

| Tier | Path | Size target | Used for final scores? | Purpose |
|---|---|---:|---|---|
| Full | `data/04_classified_benchmark.jsonl` | all valid records | no, except descriptive stats | Canonical release and full historical coverage. |
| Core v1 | `reports/benchmark_selection/core_v1_seed_13.json` | 4,800 | yes, but with main/diagnostic split | Main reasoning-floor experiments. |
| Dev/Pilot v1 | `reports/benchmark_selection/dev_prompt_v1_seed_13.json` | 600 | no | Prompt engineering and parser debugging. |
| Audit v1 | `reports/manual_audit/audit_phase_d_v1_seed_13.jsonl` and `.csv` | 450 | no | Manual label validation and classifier-risk audit. |

## 3. Main-score and diagnostic split

Every selected case must be assigned:

```text
main_score = true|false
diagnostic_only = true|false
analysis_slice = <stable string>
selection_stratum = <stable string>
```

Diagnostic cases may be evaluated by LLMs, but headline scores must be computed with `main_score_case_ids` unless the table explicitly says it is a challenge/diagnostic table.

### Main-score cases

Main-score cases include:

- `TypeA / REJECTION_FORMAT_INVALID`
- `TypeA / REJECTION_RULE_INVALID`
- `TypeA / LOGICAL`
- `TypeB / LOCAL_*`
- `TypeC / EXTERNAL_BY_ELIMINATION`, reported as IC-E-elim/no-retrieval stress
- `T_BOX / RELAXATION_SET_EXPANSION`
- `T_BOX / RESTRICTION_SET_CONTRACTION`
- `T_BOX / SCHEMA_UPDATE`, reported separately from directional reforms

### Diagnostic-only cases

Diagnostic-only cases include:

- `TypeA / DELETE_AMBIGUOUS`
- `TypeC / UNKNOWN_*`
- `T_BOX / COINCIDENTAL_SCHEMA_CHANGE`
- any low-confidence case not explicitly upgraded by manual audit
- cases with weak or missing split group keys

## 4. Core v1 quotas

Target size: 4,800.

| Group | Subtype(s) | Quota | Main-score? |
|---|---|---:|---|
| IC-L clean rule/rejection | `REJECTION_FORMAT_INVALID`, `REJECTION_RULE_INVALID`, `LOGICAL` | 700 | yes |
| Ambiguous delete diagnostic | `DELETE_AMBIGUOUS` | 250 | no |
| IC-G local graph-grounded | all `LOCAL_*` subtypes | 1,150 | yes |
| IC-E-elim stress | `EXTERNAL_BY_ELIMINATION` | 900 | yes, separately reported |
| T-box directional/schema | expansion, restriction, schema update | 1,500 | yes |
| T-box low-causality diagnostic | coincidental schema change | 300 | no |

Detailed A-box quotas:

| Stratum | Quota |
|---|---:|
| `TypeA_REJECTION_FORMAT_INVALID` | 640 |
| `TypeA_REJECTION_RULE_INVALID` | 20 |
| `TypeA_LOGICAL` | 40 |
| `TypeA_DELETE_AMBIGUOUS` | 250 |
| `TypeB_LOCAL_FOCUS_PREREPAIR_PROPERTY` | 520 |
| `TypeB_LOCAL_TEXT` | 520 |
| `TypeB_LOCAL_FOCUS_NON_TARGET_PROPERTY` | 70 |
| `TypeB_LOCAL_MIXED` | 38 |
| `TypeB_LOCAL_NEIGHBOR_IDS` | 2 |
| `TypeC_EXTERNAL_BY_ELIMINATION` | 900 |

Detailed T-box quotas:

| Stratum | Quota |
|---|---:|
| `TBOX_RELAXATION_SET_EXPANSION` | 650 |
| `TBOX_RESTRICTION_SET_CONTRACTION` | 250 |
| `TBOX_SCHEMA_UPDATE` | 600 |
| `TBOX_COINCIDENTAL_SCHEMA_CHANGE` | 300 |

If a rare stratum underfills, the selector must record the underfill and backfill from the nearest conceptually compatible stratum:

| Underfilled stratum | Backfill priority |
|---|---|
| `REJECTION_RULE_INVALID`, `LOGICAL` | `REJECTION_FORMAT_INVALID` |
| `LOCAL_NEIGHBOR_IDS`, `LOCAL_MIXED`, `LOCAL_FOCUS_NON_TARGET_PROPERTY` | other `LOCAL_*`, prioritizing `LOCAL_TEXT` then `LOCAL_FOCUS_PREREPAIR_PROPERTY` |
| `RESTRICTION_SET_CONTRACTION` | `RELAXATION_SET_EXPANSION`, then `SCHEMA_UPDATE` |
| `UNKNOWN_*` in dev/audit | `EXTERNAL_BY_ELIMINATION` with sparse/incomplete diagnostics if available |

## 5. Dev/Pilot v1 quotas

Target size: 600.

| Group | Quota |
|---|---:|
| TypeA clean rule/rejection | 70 |
| TypeA ambiguous delete | 40 |
| TypeB local graph-grounded | 130 |
| TypeC external-by-elimination or unknown diagnostic | 120 |
| T-box relaxation expansion | 80 |
| T-box restriction contraction | 40 |
| T-box schema update | 80 |
| T-box coincidental diagnostic | 40 |

The dev set is selected first. Core selection must exclude dev case ids and T-box property-revision groups.

## 6. Group keys and leakage control

### T-box group key

Use:

```text
TBOX::{property}::{property_revision_id}
```

Fallbacks:

1. `repair_target.property_revision_id`
2. `repair_target.property_revision_new`
3. `repair_target.revision_id`
4. `constraint_delta.revision_id`
5. `id`, with `weak_group_key=true`

### A-box group key

Use:

```text
ABOX::{qid}::{property}
```

Fallback to record id with `weak_group_key=true` only when either qid or property is missing.

### Caps

| Tier | T-box cap per revision | A-box cap per `(qid, property)` |
|---|---:|---:|
| Dev | 3 | 2 |
| Core | 10 | 3 |
| Audit | 5 | 3 |

### Non-overlap

- Dev and core must have no shared case ids.
- Dev and core must have no shared T-box property-revision group keys.
- Dev and core should have no shared A-box `(qid, property)` group keys. If impossible, the selector must emit a warning and exact overlap count.

## 7. Popularity stratification

Each selected stratum should be approximately balanced over popularity buckets where possible:

```text
head / mid / tail / unknown
```

If popularity is missing or too sparse for a stratum, the selector should report `unknown` counts rather than dropping cases silently.

## 8. Constraint-family stratification

Where available, record and stratify by the first relevant constraint-family id in:

```text
classification.constraint_types[*].qid
violation_context.report_violation_type_qids
constraint_delta.changed_constraint_qids
```

This is mainly for reporting and audit. Do not make hard quota fulfillment depend on constraint family unless enough cases exist.

## 9. Selection algorithm

1. Load Stage 4 records.
2. Compute derived metadata for every record:
   - `selection_stratum`
   - `analysis_slice`
   - `main_score`
   - `diagnostic_only`
   - `group_key`
   - `tbox_revision_key`
   - `popularity_bucket`
   - `constraint_family`
3. Select dev first using dev quotas and dev caps.
4. Select core from remaining cases, excluding dev case ids and dev T-box revision keys.
5. Apply per-stratum quotas with group caps and popularity-aware round-robin.
6. Record underfilled quotas and backfills.
7. Validate selected ids, counts, caps, and non-overlap.
8. Write manifest JSON.

Stable ordering key:

```python
sha1(f"{seed}|{tier}|{selection_stratum}|{group_key}|{case_id}").hexdigest()
```

## 10. Manifest validation checks

Every manifest must include a `validation` object with:

```json
{
  "selected_case_count_matches": true,
  "selected_case_ids_unique": true,
  "main_plus_diagnostic_equals_selected": true,
  "max_tbox_per_revision": 10,
  "max_abox_per_qid_property": 3,
  "dev_core_case_overlap": 0,
  "dev_core_tbox_revision_overlap": 0,
  "unknown_or_low_confidence_in_main_score": 0,
  "diagnostic_subtypes_in_main_score": 0
}
```

## 11. Reporting requirements

Every core experiment table should state whether it uses:

- `selected_case_ids`: all core cases, including diagnostic/challenge;
- `main_score_case_ids`: headline result set;
- `diagnostic_case_ids`: challenge/diagnostic analysis only.

The paper should report at least:

1. counts by track;
2. counts by class/subtype;
3. counts by main/diagnostic flag;
4. counts by confidence;
5. counts by popularity bucket;
6. counts by constraint family;
7. T-box property-revision count distribution;
8. dev/core non-overlap checks.
