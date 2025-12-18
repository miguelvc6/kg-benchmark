## Overview

The pipeline reconstructs **real historical Wikidata repairs** in three stages.
The goal is to identify *what was fixed*, *how it was fixed*, and *what information was available* at the time of repair, while ensuring the result is still meaningful in today’s graph.

---

## Stage 1 — Indexing Repair Candidates (Signal Detection)

I start from the **Wikidata Database Reports** that community bots generate periodically for constraint violations.

* For each report snapshot (t) and the next snapshot (t+1), I track entities that **disappear entirely** from the violation list.
* If an entity merely moves between sections, it is treated as **reclassification**, not a repair, and discarded.
* Each genuine disappearance is treated as a **candidate repair event**.

This process is applied to **all ~2,000 properties currently monitored by bots**.

---

## Stage 2 — Fetching the Atomic Edit

There is latency between when a human fixes a violation and when bots update the reports.
For each candidate repair:

1. I scan the **entity revision history** within a bounded lookback window to find an **A-box edit** that modified the property implicated in the violation.
2. If no such entity-level edit exists, I switch to the **property revision history** and detect **T-box edits** by tracking changes to `P2302` (constraint) claims.
3. Once the repair type is known, I query the **live Wikidata API** to ensure that the entity and property still exist today.

   * DELETE actions are allowed to remain absent.
   * Non-DELETE repairs must still be present; otherwise the case is discarded.

This yields a single, well-defined **atomic repair event** with full provenance.

---

## Stage 3 — Context Builder (World-State Snapshot)

For each accepted repair, I construct a **frozen 2025 context** that the Guardian and LLMs will reason over.

* The **focus entity** and **property definition** are loaded from the latest Wikidata dump (schema-consistent).
* **Labels and descriptions** of neighboring entities are fetched via the Wikidata API.
* For **T-box repairs**, I additionally record deterministic hashes (and optionally full statements) of the constraint *before* and *after* the change.

This decouples reasoning from live Wikidata access and ensures reproducibility.

---

## Data Layers (What the Model Sees)

Each benchmark case is represented as a `world_state` object with four layers:

* **L1 — Ego Node:**
  All properties of the entity (used for consistency and regression checks).
* **L2 — Labels:**
  Human-readable labels and descriptions (LLM-friendly surface form).
* **L3 — Neighborhood:**
  1-hop outgoing edges (for **Type B: Graph-RAG** reasoning).
* **L4 — Constraints:**
  The full SHACL definition of the violated property (for **Type A: Logic**).

---

## Final Data Structure (Simplified)

```json
{
  "repair_Q42_123456789": {
    "L1_ego_node": {
      "qid": "Q42",
      "label": "Douglas Adams",
      "description": "English author",
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
      "property_revision_prev": 213456700,
      "signatures": {
        "before": {"hash": "a0c9…"},
        "after": {"hash": "ff09…"}
      }
    }
  }
}
```

---

## One-sentence takeaway (useful verbally)

> *The pipeline reconstructs real Wikidata repairs by detecting when violations disappear, identifying the exact entity- or schema-level edit that caused it, and freezing a clean, reproducible world-state snapshot that separates logic, topology, and external information.*

If you want, I can also:

* compress this into a **30-second verbal explanation**, or
* adapt it to a **single slide diagram** for supervision meetings.
