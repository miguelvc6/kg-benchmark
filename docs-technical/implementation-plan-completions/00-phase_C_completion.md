# Phase C Completion: Dataset Tiers and Selection Manifests

**Project:** WikidataRepairEval 1.0  
**Phase:** C — dataset tiers and selection manifests  
**Status:** Completed as a design-and-implementation specification  
**Date:** 2026-05-22  
**Depends on:** Phase B classifier redesign and regenerated Stage 4 counts

## 1. Phase C decision summary

Phase C fixes the benchmark selection policy for paper-facing experiments. The full dataset remains the canonical historical artifact, but LLM experiments must use deterministic, stratified manifests so that results are not dominated by T-box repetition, ambiguous deletes, or low-causality schema changes.

The selected policy is:

| Tier | Manifest | Target size | Purpose |
|---|---|---:|---|
| Full | none; use `data/04_classified_benchmark.jsonl` | 535,570 records in the Phase B snapshot | Release, descriptive statistics, large-scale non-LLM analysis. |
| Core v1 | `reports/benchmark_selection/core_v1_seed_13.json` | 4,800 cases | Main reasoning-floor LLM experiments. |
| Dev/Pilot v1 | `reports/benchmark_selection/dev_prompt_v1_seed_13.json` | 600 cases | Prompt development, parser debugging, representation ablations. |
| Audit v1 | `reports/manual_audit/audit_phase_d_v1_seed_13.jsonl` and `.csv` | 450 cases | Manual classifier validation and Phase D evidence audit. |

The core manifest contains both `selected_case_ids` and analysis labels that separate **main-score** cases from **diagnostic/challenge** cases. Diagnostic cases may be run by models but must not be silently mixed into the headline main score.

## 2. Phase B counts used by Phase C

The Phase B classifier snapshot has the following relevant counts:

| Slice | Count |
|---|---:|
| Full records | 535,570 |
| A-box records | 78,976 |
| T-box records | 456,594 |
| TypeA | 17,556 |
| TypeB | 22,409 |
| TypeC / `EXTERNAL_BY_ELIMINATION` | 39,011 |
| T-box / `RELAXATION_SET_EXPANSION` | 86,876 |
| T-box / `RESTRICTION_SET_CONTRACTION` | 2,023 |
| T-box / `SCHEMA_UPDATE` | 213,802 |
| T-box / `COINCIDENTAL_SCHEMA_CHANGE` | 153,893 |

This changes the paper design: TypeA is no longer mostly logical repair. It is mostly format rejection and ambiguous deletion. Therefore, the core selection policy treats clean rule/rejection cases, ambiguous deletes, local cases, external-by-elimination cases, and schema reforms as separate analysis slices.

## 3. Terminology fixed by Phase C

Use these terms consistently in repository docs and paper-facing text:

| Repository label | Paper-facing label | Phase C interpretation |
|---|---|---|
| `TypeA / REJECTION_FORMAT_INVALID` | IC-L clean rule/rejection | Rule or format violation identifies an invalid value; deterministic enough for main scoring. |
| `TypeA / REJECTION_RULE_INVALID` | IC-L clean rule/rejection | Rule violation identifies an invalid value; rare but conceptually important. |
| `TypeA / LOGICAL` | IC-L rule-implied | Very rare after Phase B; include but do not let it define all TypeA. |
| `TypeA / DELETE_AMBIGUOUS` | ambiguous delete diagnostic | Run as diagnostic/challenge; do not mix into headline TypeA score. |
| `TypeB / LOCAL_*` | IC-G local graph-grounded | Main local-context slice; report subtype-specific results. |
| `TypeC / EXTERNAL_BY_ELIMINATION` | IC-E-elim / no-retrieval stress | Main stress slice, but not confirmed external evidence. |
| `TypeC / UNKNOWN_*` | IC-U unknown/incomplete | Diagnostic/challenge or excluded from main score. |
| `T_BOX / RELAXATION_SET_EXPANSION` | T-box directional relaxation | Main T-box schema-reform slice. |
| `T_BOX / RESTRICTION_SET_CONTRACTION` | T-box directional restriction | Main T-box schema-reform slice. |
| `T_BOX / SCHEMA_UPDATE` | T-box generic schema update | Main T-box slice, reported separately from directional reforms. |
| `T_BOX / COINCIDENTAL_SCHEMA_CHANGE` | low-causality T-box diagnostic | Diagnostic/challenge unless manual audit upgrades it. |

## 4. Core v1 selection policy

### 4.1 Target size and composition

`core_v1_seed_13.json` targets **4,800 cases**:

| Core group | Subtype quota | Main-score? | Notes |
|---|---:|---|---|
| TypeA clean rule/rejection | 700 | yes | Mostly `REJECTION_FORMAT_INVALID`; include rare `REJECTION_RULE_INVALID` and `LOGICAL` when available. |
| TypeA ambiguous delete | 250 | no, diagnostic | Included to measure delete ambiguity and overconfident repair. |
| TypeB local graph-grounded | 1,150 | yes | Balance `LOCAL_FOCUS_PREREPAIR_PROPERTY` and `LOCAL_TEXT`; include rare local subtypes. |
| TypeC external-by-elimination | 900 | yes, but reported as IC-E-elim | No-retrieval stress slice; not confirmed external. |
| T-box directional/schema reform | 1,500 | yes | Expansion, restriction, and generic schema update. |
| T-box coincidental schema change | 300 | no, diagnostic | Low-causality schema-change slice. |
| **Total** | **4,800** | mixed | Manifest must expose `main_score_case_ids` and `diagnostic_case_ids`. |

### 4.2 Detailed quotas

A-box quotas:

| Class/subtype | Quota | Fill policy |
|---|---:|---|
| `TypeA / REJECTION_FORMAT_INVALID` | 640 | Fill from same subtype across popularity buckets and constraint families. |
| `TypeA / REJECTION_RULE_INVALID` | 20 | If fewer non-dev eligible cases exist, take all and backfill from `REJECTION_FORMAT_INVALID`. |
| `TypeA / LOGICAL` | 40 | If fewer non-dev eligible cases exist, take all and backfill from `REJECTION_FORMAT_INVALID`. |
| `TypeA / DELETE_AMBIGUOUS` | 250 | Diagnostic-only; stratify by violation family when possible. |
| `TypeB / LOCAL_FOCUS_PREREPAIR_PROPERTY` | 520 | Main score. |
| `TypeB / LOCAL_TEXT` | 520 | Main score. |
| `TypeB / LOCAL_FOCUS_NON_TARGET_PROPERTY` | 70 | Main score, rare local source. |
| `TypeB / LOCAL_MIXED` | 38 | Main score, rare local source. |
| `TypeB / LOCAL_NEIGHBOR_IDS` | 2 | Include all if eligible; otherwise record underfill. |
| `TypeC / EXTERNAL_BY_ELIMINATION` | 900 | Main stress slice; report as IC-E-elim, not confirmed external. |

T-box quotas:

| Class/subtype | Quota | Fill policy |
|---|---:|---|
| `T_BOX / RELAXATION_SET_EXPANSION` | 650 | Main score; cap by property revision. |
| `T_BOX / RESTRICTION_SET_CONTRACTION` | 250 | Main score; cap by property revision. |
| `T_BOX / SCHEMA_UPDATE` | 600 | Main score but separate from directional reforms. |
| `T_BOX / COINCIDENTAL_SCHEMA_CHANGE` | 300 | Diagnostic-only unless audit upgrades. |

### 4.3 Caps and leakage controls

Use deterministic group-level sampling.

| Constraint | Core v1 rule |
|---|---|
| Random seed | `13` |
| Hash order | SHA-1 over `seed | tier | stratum | group_key | case_id` |
| T-box cap | Max 10 cases per property revision in core. |
| A-box cap | Max 3 cases per `(qid, property)` group in core. |
| Dev/core leakage | No dev and core case may share the same T-box property revision group. Prefer no shared `(qid, property)` A-box group. |
| Few-shot leakage | Few-shot examples must not share case id, qid, or T-box property revision with evaluated core cases. |
| Low confidence | Allowed only if explicitly labeled `diagnostic_only`. |
| Unknown TypeC | Diagnostic-only or excluded from main score. |

### 4.4 Required manifest fields

The core manifest must contain at least:

```json
{
  "manifest_type": "benchmark_selection",
  "manifest_version": "phase_c_v1",
  "tier": "core",
  "seed": 13,
  "inputs": {
    "classified_benchmark": "data/04_classified_benchmark.jsonl",
    "world_state": "data/03_world_state.json",
    "phase_b_counts": "reports/classifier_audit/new_full_counts_phase_b.json"
  },
  "policy": {
    "target_size": 4800,
    "strata": {},
    "tbox_cap_per_property_revision": 10,
    "abox_cap_per_qid_property": 3,
    "dev_core_group_disjoint": true,
    "typec_semantics": "EXTERNAL_BY_ELIMINATION is an IC-E-elim/no-retrieval stress label, not confirmed external evidence"
  },
  "selected_case_ids": [],
  "main_score_case_ids": [],
  "diagnostic_case_ids": [],
  "case_annotations": {},
  "counts": {},
  "underfilled_quotas": [],
  "validation": {}
}
```

`case_annotations` should map each selected id to:

```json
{
  "tier": "core",
  "selection_stratum": "TypeB_LOCAL_TEXT",
  "analysis_slice": "main_ic_g_local_text",
  "main_score": true,
  "diagnostic_only": false,
  "group_key": "ABOX::Q...::P...",
  "tbox_revision_key": null,
  "class": "TypeB",
  "subtype": "LOCAL_TEXT",
  "confidence": "medium",
  "track": "A_BOX",
  "popularity_bucket": "head|mid|tail|unknown",
  "constraint_family": "..."
}
```

## 5. Dev/Pilot v1 selection policy

`dev_prompt_v1_seed_13.json` targets **600 cases**. It is used only for prompt development, parser debugging, prompt representation comparison, and few-shot exemplar selection experiments.

| Dev group | Quota |
|---|---:|
| TypeA clean rule/rejection | 70 |
| TypeA ambiguous delete | 40 |
| TypeB local graph-grounded | 130 |
| TypeC external-by-elimination / unknown diagnostic | 120 |
| T-box relaxation expansion | 80 |
| T-box restriction contraction | 40 |
| T-box schema update | 80 |
| T-box coincidental diagnostic | 40 |
| **Total** | **600** |

Rules:

1. Dev is selected before core.
2. Core excludes dev case ids.
3. Core should also exclude dev T-box property-revision groups.
4. For A-box, prefer excluding dev `(qid, property)` groups from core when quotas allow.
5. Dev cases may include harder/diagnostic cases because the purpose is prompt debugging, not final scoring.
6. Dev results must not be reported as final benchmark performance.

## 6. Splitter policy for C4

The existing random stratified splitter is not enough for few-shot and prompt-development safety because it may split the same T-box property revision across train/dev/test. Phase C therefore requires a **group-aware split policy**.

Required split keys:

| Record type | Split group key |
|---|---|
| T-box | `TBOX::{property}::{property_revision_id}`; fallback to `TBOX::{property}::{constraint_delta.revision_id}`; final fallback to `TBOX::{property}::{id}`. |
| A-box | `ABOX::{qid}::{property}`. |
| Missing fields | Fall back to record id and mark `weak_group_key=true`. |

Stratification keys:

```text
track
classification.class
classification.subtype
classification.confidence
popularity_bucket
constraint_family
selection_stratum
```

Acceptance thresholds:

| Check | Required threshold |
|---|---|
| No case-id overlap between dev and core | 0 overlaps |
| No T-box revision overlap between dev and core | 0 overlaps |
| A-box `(qid, property)` overlap | 0 preferred; nonzero only with explicit warning and count. |
| Max core T-box cases per revision | <= 10 |
| Max dev T-box cases per revision | <= 3 |
| Main/diagnostic label coverage | Every selected id has `main_score` and `analysis_slice`. |
| Unknown/low-confidence in headline score | 0 cases |

## 7. Phase D readiness requirements completed by Phase C

Phase C defines the audit sample and annotation template requirements so that Phase D can begin immediately once Codex implements the manifest builders.

Required Phase D artifacts:

| Artifact | Path | Status |
|---|---|---|
| Audit sample policy | `00-manual_audit_phase_D.md` | specified |
| Audit sample manifest | `reports/manual_audit/audit_phase_d_v1_seed_13.jsonl` | implementation prompt provided |
| Audit CSV | `reports/manual_audit/audit_phase_d_v1_seed_13.csv` | implementation prompt provided |
| Audit annotation schema | `reports/manual_audit/audit_annotation_schema.json` | implementation prompt provided |
| Audit metrics script | `src/audit_summarize.py` or equivalent | implementation prompt provided |

The audit sample targets **450 cases** and covers the highest-risk classifier decisions:

| Audit stratum | Target count |
|---|---:|
| `TypeC / EXTERNAL_BY_ELIMINATION`, QID truth | 50 |
| `TypeC / EXTERNAL_BY_ELIMINATION`, literal/date/numeric truth | 50 |
| `TypeC / UNKNOWN_*` or sparse/incomplete diagnostics if present | 30 |
| `TypeA / REJECTION_FORMAT_INVALID` | 50 |
| `TypeA / DELETE_AMBIGUOUS`, especially single/unique conflicts | 50 |
| `TypeB / LOCAL_TEXT` | 50 |
| `TypeB / LOCAL_FOCUS_PREREPAIR_PROPERTY` | 40 |
| `T_BOX / SCHEMA_UPDATE` | 50 |
| `T_BOX / COINCIDENTAL_SCHEMA_CHANGE` | 40 |
| `T_BOX / RELAXATION_SET_EXPANSION` or `RESTRICTION_SET_CONTRACTION` | 40 |
| **Total** | **450** |

## 8. Tests required for Phase C implementation

Codex should add tests that verify:

1. core and dev manifests are deterministic for fixed seed;
2. core and dev case ids do not overlap;
3. core and dev T-box property-revision groups do not overlap;
4. core max T-box cap is 10 per revision;
5. dev max T-box cap is 3 per revision;
6. diagnostic-only subtypes are not included in `main_score_case_ids`;
7. unknown TypeC and low-confidence cases are never in `main_score_case_ids`;
8. rare subtypes can underfill without crashing and produce an `underfilled_quotas` entry;
9. manifest counts match selected ids;
10. audit sample builder emits all required annotation fields.

Minimum command set after implementation:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/classifier.py --self-test
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run --extra dev python -m pytest tests/test_classifier.py
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run --extra dev python -m pytest tests/test_benchmark_selection.py tests/test_splitter.py tests/test_manual_audit.py
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/select_benchmark_cases.py --tier dev --output reports/benchmark_selection/dev_prompt_v1_seed_13.json
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/select_benchmark_cases.py --tier core --exclude-manifest reports/benchmark_selection/dev_prompt_v1_seed_13.json --output reports/benchmark_selection/core_v1_seed_13.json
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/build_audit_sample.py --core-manifest reports/benchmark_selection/core_v1_seed_13.json --dev-manifest reports/benchmark_selection/dev_prompt_v1_seed_13.json --output-jsonl reports/manual_audit/audit_phase_d_v1_seed_13.jsonl --output-csv reports/manual_audit/audit_phase_d_v1_seed_13.csv
```

## 9. Phase C acceptance criteria status

| Acceptance criterion | Status |
|---|---|
| Core and dev are deterministic. | Completed as policy; implementation prompt specifies stable hashing and tests. |
| Core excludes or separately marks low-confidence/unknown cases. | Completed; unknown and low-confidence are diagnostic-only. |
| Dev does not overlap final core evaluation. | Completed; case-id and T-box-revision disjointness required. |
| No T-box revision dominates the core. | Completed; max 10 per property revision. |
| All key strata have enough cases for analysis. | Completed; quotas are fixed and underfills are recorded. |
| Splitter can avoid few-shot leakage. | Completed as group-aware split requirement. |
| Phase D can start. | Completed; audit sample and template requirements are specified. |

## 10. Remaining repository implementation handoff

The conceptual Phase C decisions are complete. The repository still needs the manifest builder and audit builder implemented in code. Use:

- `00-codex_phase_C_selection.md`
- `00-codex_phase_D_audit.md`

as Codex prompts. After those prompts are applied inside the actual repository, run the tests listed above and commit the generated manifests.
