# WikidataRepairEval 1.0 – Data Acquisition & Validation Protocol

**Project:** Dynamics of Neuro-Symbolic Alignment (Phase 1)
**Status:** Protocol Definition & Indexing Phase
**Goal:** Build a gold-standard benchmark of historical Wikidata repairs that isolates logical consistency, topological reasoning, and external retrieval gaps.

---

## Overview

We construct WikidataRepairEval 1.0 by replaying real repair events rather than scraping static dumps. The workflow rejects full-history parsing (multi-terabyte XML) and instead follows a hybrid index–fetch pipeline that leverages community-maintained constraint reports as high-signal pointers to repairs. Each stage produces a verifiable artifact that becomes the input for the next stage.

### Pipeline at a Glance

| Stage              | Description                                                                                               | Inputs                                                          | Outputs                                            |
| :----------------- | :-------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------- | :------------------------------------------------- |
| 1. Indexer         | Mine constraint-violation report histories to find entities that truly disappeared between two snapshots. | `Wikidata:Database reports/Constraint violations/*` histories | `data\01_repair_candidates.json`                 |
| 2. Fetcher         | Query revision histories to isolate the exact edit, label it as A-box/T-box, and capture provenance.      | `data\01_repair_candidates.json`, Wikibase REST API           | `data\02_wikidata_repairs.jsonl` + compiled JSON |
| 3. Context Builder | Freeze the 2026 neighborhood, labels, constraint metadata, and (for T-box) constraint deltas.            | `data\02_wikidata_repairs.json`, `latest-all.json.gz` dump  | `world_state.json` keyed by each benchmark case  |

---

## Stage 1 – Indexer (Signal Detection)

- **Purpose:** Detect candidate repairs by monitoring entries that leave constraint-violation pages.
- **Input:** Revision histories from `Wikidata:Database reports/Constraint violations/*`.
- **Process:** Parse each report's history. A candidate is only emitted if entity `E` disappears from the entire page between `T_report` and `T_next`; if it merely moves to another section, it is ignored as a reclassification.
- **Output:** `data\01_repair_candidates.json`, containing tuples `{qid, property_id, violation_type, fix_date, report_revision_old, report_revision_new}` for downstream fetching.
- **Benefits:** Limits the search space to high-quality violations, encodes the relevant constraint class, and preserves report provenance for debugging.

---

## Stage 2 – Fetcher (Forensics)

- **Purpose:** Pinpoint the exact revision that fixed each violation, classify the fix as A-box (entity edit) or T-box (constraint edit), and capture the relevant before/after evidence.
- **Inputs:** `data\01_repair_candidates.json`, Wikibase REST API endpoint `GET /w/rest.php/v1/page/{qid}/history`, plus `Special:EntityData` snapshots.
- **Process:**
  - Deduplicate candidates before scanning to avoid repeated Stage-2 work while preserving all violation types.
  - Walk revisions from newest to oldest and stop at the first property signature diff (the last change in-window).
  - Cache revision histories per QID/window and reuse overlapping windows when possible.
  - Cache entity snapshots per `(qid, revision_id)` on disk (plus short-lived negative caching) so multiple properties share one download.
  - Fetch snapshots with bounded concurrency and a global rate limiter to avoid sustained 429s.
  - Scan entity histories within the 7-day lookback window and store the *latest* property change (action + old/new signatures). Each record now includes `track`, `repair_target.kind`, and a fully explicit before/after payload.
  - If no entity edit exists, scan the property (`Property:{pid}`) for the most recent `P2302` signature change, capturing deterministic SHA1 hashes and optional serialized constraint statements in `constraint_delta`.
  - When an A-box fix is found, run a cheap reverse scan (≤25 revisions) over the property history to detect coincident T-box edits and mark `ambiguous` entries.
- **Persistence Filter:** Live 2026 data is fetched *after* the repair type is known. DELETE actions may legitimately return `None`; all other repairs must still exist in the live graph or the candidate is dropped.
- **Outputs:** Append-only `data\02_wikidata_repairs.jsonl` during execution, compiled into `data\02_wikidata_repairs.json` at the end. Each entry stores:
  - Unique IDs (`repair_{qid}_{rev}` or `reform_{qid}_{pid}_{rev}`), track labels, and report provenance.
  - `violation_context` (offending value or `null`, Stage-1 timestamps, report revisions).
  - `repair_target` discriminated unions for A-box vs. T-box, plus persistence metadata and optional `ambiguous_reasons`.
- **Stats Logging:** Per-candidate stats are buffered in memory and flushed in large JSONL batches to reduce I/O overhead.
- **Note:** This artifact drives both the taxonomy labels and the context builder.
- **Migration Note (2026):** Label and snapshot caches are now stored in SQLite (`data/cache/labels_en.sqlite`, `data/cache/entity_snapshots.sqlite`). Legacy JSON label caches and per-entity snapshot folders are deprecated.

---

## Stage 3 – Context Builder (World State Snapshot)

### Role

Freeze the 2026 local topology so evaluators can test reasoning with the same view of the graph. Acts as the “freezing mechanism” referenced in the Guardian hypothesis.

### Inputs

- `data\02_wikidata_repairs.json` (provides the `Set<QID>` to preserve).
- `latest-all.json.gz` (2026 dump).

### Process

1. **Stream-and-Filter:** Perform a single pass over `latest-all.json.gz`. When an entry’s `id` is in the target set, extract required “context layers”; never load the dump entirely into memory. Duplicate IDs are rejected with a warning to prevent silent overwrites.
2. **Layer Extraction:** Capture four distinct data layers per entity to support downstream ablations:
   - **L1 – Ego Node Properties:** Full statement set to verify persistence and detect merges/deletions.
   - **L2 – Labels & Descriptions:** Resolve IDs into human-readable strings for LLM consumption.
   - **L3 – 1-Hop Neighborhood:** Outgoing 1-hop edges (no recursion) for Type B reasoning checks.
   - **L4 – Constraint Metadata:** Current SHACL (P2302) definitions to confirm the violation logic itself.
   - **Constraint Change Context (T-box only):** Hashes and optional serialized statements before/after the property edit.

### Outputs

- `world_state.json`, keyed by benchmark entry id:
  ```json
  {
    "repair_Q42_123456789": {
      "L1_ego_node": {
        "qid": "Q42",
        "label": "Douglas Adams",
        "description": "English author and humorist",
        "properties": {
          "P31": ["Q5"],
          "P569": ["1952-03-11"]
        }
      },
      "L3_neighborhood": {
        "outgoing_edges": [
          {
            "property_id": "P26",
            "target_qid": "Q12345",
            "target_label": "Jane Belson",
            "target_description": "Wife of Douglas Adams"
          }
        ]
      },
      "L4_constraints": {
        "property_id": "P569",
        "constraints": [
          {
            "constraint_type": {
              "qid": "Q21503250",
              "label": "contemporary constraint"
            },
            "rule_summary": "P569 must be greater than P570 if P570 exists."
          }
        ]
      },
      "constraint_change_context": {
        "property_revision_id": 213456789,
        "signatures": {
          "before": {"hash": "a0c9…"},
          "after": {"hash": "ff09…"}
        }
      }
    }
  }
  ```

---

## Taxonomy Standards

Each repair is classified to enable RQ2 (Information Gap) analysis.

- **Type A – Logical (Internal Consistency):**Detected and fixed purely via literal checks (e.g., End Date < Start Date). No graph traversal or retrieval.
- **Type B – Local (Topological Reasoning):**Requires inspecting statements on the subject or directly connected nodes (≤1 hop). Includes textual descriptions on the node, e.g., verifying that a human has the correct role type.
- **Type C – External (Information Void):**
  Needs information absent from the graph (beyond 2 hops). Relies on frozen Tavily search results packaged with the case.

---

## Quality Control Protocols

### Persistence Filter

- **Risk Addressed:** Time-travel paradox—repairs that no longer make sense in 2026.
- **Rule:** Discard `(E, P, V)` if it no longer exists in the 2026 dump. Sacrifices ~60% of cases to ensure reproducibility.

### Frozen Retrieval Snapshots

- **Scope:** Type C cases only.
- **Action:** Query Tavily once, store the JSON as `retrieval_context`, and forbid live web calls during evaluation to decouple search-engine variance from model reasoning.

### Anti-Gaming Penalty

- **Metric:** Information Preservation Score `S_info`:
  $$
  S_{info} =
  \begin{cases}
  1.0 & \text{if } Action_{Model} = Action_{Human} \\
  -0.5 & \text{if } Action_{Model} = \text{DELETE} \land Action_{Human} = \text{UPDATE} \\
  0.0 & \text{otherwise}
  \end{cases}
  $$
- **Goal:** Penalize deletions that “silence” constraints instead of repairing them.

---
