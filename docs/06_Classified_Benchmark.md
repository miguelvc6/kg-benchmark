# 04 Classified Benchmark Artifact (WikidataRepairEval 1.0)

This document defines the final experiment-ready output of Phase 1:

- `data/04_classified_benchmark.jsonl` (LEAN)
- `data/04_classified_benchmark_full.jsonl` (FULL, optional)

These artifacts are produced by `classifier.py` and are the canonical inputs for Phase 2-4 experiments (Writer/Verifier, ablations, and learning loops).

---

## 1. Purpose

The earlier artifacts are provenance-grade and modular:

- `01_repair_candidates.json`: report-diff candidate signal
- `02_wikidata_repairs.json`: forensic repair events (A-Box/T-Box)
- `03_world_state.json`: frozen 2026 context (L1-L4)
- `00_entity_popularity.json`: deterministic head/tail stratification

However, experiments need a single, joined record with an explicit label for:

> Information Necessity (Type A / B / C)

The 04 artifact provides that record, plus a full decision trace for auditability.

---

## 2. Record Unit

One record corresponds to one repair event, keyed by:

- `id` (the repair event id produced in Stage 2)

This is the join key into `03_world_state.json`, which stores context as:

`world_state[id] -> { L1, L2, L3, L4, ... }`

---

## 3. Two Output Variants

### 3.1 LEAN: `04_classified_benchmark.jsonl`

Contains:

- the Stage-2 event metadata
- popularity and English mirrors needed for prompting
- the taxonomy label (Type A/B/C) with decision trace
- a reference to the external world state (`context_ref`)

This variant is recommended for routine experimentation where `03_world_state.json` is available locally.

### 3.2 FULL: `04_classified_benchmark_full.jsonl` (optional)

Same record, but embeds `world_state` (L1-L4) directly.

This is recommended for:
- artifact evaluation packages
- offline runs without access to `03_world_state.json`
- long-term archival stability

---

## 4. Required Fields

A 04 record must include:

- `id`, `qid`, `property`, `track`
- `violation_context`
- `repair_target`
- `persistence_check`
- `labels_en` (qid + property)
- `popularity`
- `classification`
- `context_ref` (LEAN) OR `world_state` (FULL)

The taxonomy script must not modify the meaning of Stage-2 raw fields; it only adds classification and derived summaries.

---

## 5. The Classification Block (Taxonomy of Information Necessity)

The taxonomy label is stored under:

`classification.class` in `{TypeA, TypeB, TypeC}`

and a more operational tag:

`classification.subtype` in `{LOGICAL, LOCAL_NEIGHBOR_IDS, LOCAL_FOCUS_PREREPAIR_PROPERTY, LOCAL_TEXT, LOCAL_MIXED, EXTERNAL, REJECTION, UNKNOWN}`

### 5.1 Decision Trace

Every classification must include a `decision_trace` array. This makes the taxonomy auditable and debuggable.

Steps (in order):

1. `is_delete`
2. `rule_deterministic`
3. `local_availability`
4. `fallback_external`

### 5.2 Constraint Types

`classification.constraint_types` is derived deterministically from:

- `world_state.L4_constraints.constraints[*].constraint_type`
- optionally unioned with T-Box delta signatures when present

This field enables quick stratification by constraint family without reparsing raw signatures.

---

## 6. Determinism Requirements

The 04 artifact is deterministic given identical inputs:

- same `02_wikidata_repairs.json`
- same `03_world_state.json`
- same classifier code + policy constants

No live web calls are allowed during classification.

---

## 7. Recommended Companion Outputs

Although not required, the classifier should also emit:

- `reports/classifier_stats.json` (counts per type, subtype, truth_source, and error reasons)
- `data/05_splits.json` (deterministic train/dev/test IDs stratified by Type A/B/C, head/tail, track)

---

## 8. Compatibility Notes

- `information_type` in Stage-2 records may remain `TBD`; the authoritative label is `classification`.
- `labels_en` excludes aliases by design to avoid multilingual noise and prompt leakage.
- `classification.diagnostics` may include `truth_tokens`, `truth_source`, and `truth_applicable` for auditing.
