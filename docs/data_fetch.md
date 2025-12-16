# WikidataRepairEval 1.0 – Data Acquisition & Validation Protocol

**Project:** Dynamics of Neuro-Symbolic Alignment (Phase 1)  
**Date:** December 10, 2025  
**Status:** Protocol Definition & Indexing Phase  
**Goal:** Build a gold-standard benchmark of historical Wikidata repairs that isolates logical consistency, topological reasoning, and external retrieval gaps.

---

## Overview

We construct WikidataRepairEval 1.0 by replaying real repair events rather than scraping static dumps. The workflow rejects full-history parsing (multi-terabyte XML) and instead follows a hybrid index–fetch pipeline that leverages community-maintained constraint reports as high-signal pointers to repairs. Each stage produces a verifiable artifact that becomes the input for the next stage.

### Pipeline at a Glance

| Stage              | Description                                                                                                  | Inputs                                                        | Outputs                                            |
| :----------------- | :----------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------ | :------------------------------------------------- |
| 1. Indexer         | Mine constraint-violation report histories to find entities that were fixed between two snapshots.           | `Wikidata:Database reports/Constraint violations/*` histories | `data\01_repair_candidates.json`                           |
| 2. Fetcher         | Query revision histories to isolate the exact edit that resolved each violation and capture pre/post states. | `data\01_repair_candidates.json`, Wikibase REST API                   | `data\02_wikidata_repairs.json`                    |
| 3. Context Builder | Freeze the 2025 neighborhood, labels, and constraint metadata for every repaired entity.                     | `data\02_wikidata_repairs.json`, `latest-all.json.gz` dump    | `world_state.json` attached to each benchmark case |

---

## Stage 1 – Indexer (Signal Detection)

- **Purpose:** Detect candidate repairs by monitoring entries that leave constraint-violation pages.
- **Input:** Revision histories from `Wikidata:Database reports/Constraint violations/*`.
- **Process:** Parse each report's history. If entity `E` appears at timestamp `T_report` and disappears at `T_next`, flag the interval `(T_report, T_next)` as a repair window for `(E, constraint_id)`.
- **Output:** `data\01_repair_candidates.json`, containing tuples `{qid, constraint_id, violation_type, t_report, t_next}` for downstream fetching.
- **Benefits:** Limits the search space to high-quality violations and encodes the relevant constraint class (e.g., `P569` Date of Birth).

---

## Stage 2 – Fetcher (Forensics)

- **Purpose:** Pinpoint the exact revision that fixed each violation and capture before/after graphs.
- **Inputs:** `data\01_repair_candidates.json`, Wikibase REST API endpoint `GET /w/rest.php/v1/page/{qid}/history`.
- **Process:** For every candidate, pull the revision list within `(t_report, t_next)`, diff adjacent revisions, and select the edit where the violating triple changed. Record both the violating state and the repaired state.
- **Outputs:** `data\02_wikidata_repairs.json`, where each entry stores:
  - `qid`, `revision_id_before`, `revision_id_after`
  - Serialized statements for the specific property before and after
  - Metadata for taxonomy classification (timestamps, editor ids, constraint ids)
- **Note:** This artifact drives both the taxonomy labels and the context builder.

---

## Stage 3 – Context Builder (World State Snapshot)

### Role

Freeze the 2025 local topology so evaluators can test reasoning with the same view of the graph. Acts as the “freezing mechanism” referenced in the Guardian hypothesis.

### Inputs

- `data\02_wikidata_repairs.json` (provides the `Set<QID>` to preserve).
- `latest-all.json.gz` (2025 dump).

### Process

1. **Stream-and-Filter:** Perform a single pass over `latest-all.json.gz`. When an entry’s `id` is in the target set, extract required “context layers”; never load the dump entirely into memory.
2. **Layer Extraction:** Capture four distinct data layers per entity to support downstream ablations:
   - **L1 – Ego Node Properties:** Full statement set to verify persistence and detect merges/deletions.
   - **L2 – Labels & Descriptions:** Resolve IDs into human-readable strings for LLM consumption.
   - **L3 – 1-Hop Neighborhood:** Outgoing 1-hop edges (no recursion) for Type B reasoning checks.
   - **L4 – Constraint Metadata:** Current SHACL (P2302) definitions to confirm the violation logic itself.

### Outputs

- `world_state.json`, keyed by benchmark entry id:
  ```json
  {
    "world_state": {
      "focus_node": {
        "qid": "Q42",
        "label": "Douglas Adams",
        "description": "English author and humorist",
        "properties": {
          "P31": ["Q5"],
          "P569": ["1952-03-11"]
        }
      },
      "neighborhood_snapshot": {
        "outgoing_edges": [
          {
            "property_id": "P26",
            "target_qid": "Q12345",
            "target_label": "Jane Belson",
            "target_description": "Wife of Douglas Adams"
          }
        ]
      },
      "constraint_metadata": {
        "property_id": "P569",
        "constraint_type": "Q21503250",
        "rule_summary": "P569 must be greater than P570 if P570 exists."
      }
    }
  }
  ```

---

## Taxonomy Standards

Each repair is classified to enable RQ2 (Information Gap) analysis.

- **Type A – Logical (Internal Consistency):**  
  Detected and fixed purely via literal checks (e.g., End Date < Start Date). No graph traversal or retrieval.

- **Type B – Local (Topological Reasoning):**  
  Requires inspecting statements on the subject or directly connected nodes (≤1 hop). Includes textual descriptions on the node, e.g., verifying that a human has the correct role type.

- **Type C – External (Information Void):**  
  Needs information absent from the graph (beyond 2 hops). Relies on frozen Tavily search results packaged with the case.

---

## Quality Control Protocols

### Persistence Filter

- **Risk Addressed:** Time-travel paradox—repairs that no longer make sense in 2025.
- **Rule:** Discard `(E, P, V)` if it no longer exists in the 2025 dump. Sacrifices ~60% of cases to ensure reproducibility.

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
