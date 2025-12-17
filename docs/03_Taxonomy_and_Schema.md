# Taxonomy & Data Schema

**Artifact Definition:** WikidataRepairEval 1.0
**Context:** Classification of Knowledge Graph Errors & JSON Specifications

---

## 1. The Taxonomy of Information Necessity

To enable the ablation studies required for **RQ2 (The Information Gap)**, the benchmark algorithmically sorts violations into three complexity classes based on the *source of information* required to resolve them.

### Type A: Logical (Internal Consistency)

**Definition:** The violation is detectable and resolvable purely by analyzing the internal consistency of the statement's literals or the schema definition. No graph traversal is required.

* **Representative Constraints:** Format (Regex), One-of (Enumeration), Range (Min/Max).
* **Hypothesis:** LLMs can solve these with **Zero-Shot Logic** (No retrieval needed).
* **Example:** "End Date (1900) < Start Date (1950)".

### Type B: Local (Topological Reasoning)

**Definition:** The violation requires checking the properties of the *Subject* or *Object* entities involved in the triple (strictly \le 1 hop). This includes textual descriptions stored on the node.

* **Representative Constraints:** Contemporary, Inverse, Distinct Value, Type Constraints.
* **Hypothesis:** LLMs require **Graph RAG** (Topology) to solve these, but text retrieval introduces noise.
* **Example:** "Subject is an instance of 'Human', but Property 'Founded By' expects 'Organization'."

### Type C: External (Information Void)

**Definition:** The violation can only be resolved by retrieving information not present in the local graph (beyond 2 hops).

* **Representative Constraints:** Mandatory Value, Conflicts with External, Open-world errors.
* **Hypothesis:** LLMs will fail unless **Web RAG** (Active Grounding) is enabled.
* **Example:** "Paris is the capital of [Missing Value]."

---

## 2. The "Reformer" Track (A-Box vs. T-Box)

Most benchmarks conflate data errors with schema evolution. We explicitly separate them into two evaluation tracks:

### Track 1: The Cleaner (A-Box Repair)*

**Input:** A violation where the historical fix was a **data update** to the entity.

**Goal:** Satisfy the constraint by modifying the instance data.

* **Action:** `UPDATE`, `DELETE`, or `INSERT` triples on the entity.

### Track 2: The Reformer (T-Box Reform)*

**Input:** A violation where the historical fix was a **modification to the Constraint itself** (e.g., widening a date range or adding an allowed class).

* **Goal:** Detect that the rule is outdated or too strict ("Concept Drift").
* **Action:** Propose a change to the SHACL definition.

---

## 3. Data Schema Specifications

The benchmark produces two primary JSON artifacts. The schema is designed to separate the *Event* (what happened) from the *Context* (the world state).

### 3.1 Repair Index (`data/02_wikidata_repairs.json`)

This file is the "Gold Standard" log of historical repairs. Every entry now carries a `track`
flag describing whether the fix touched the **A-box** (instance data) or the **T-box**
(constraint definition). The identifier is unique across the entire benchmark (`repair_{qid}_{rev}`
or `reform_{qid}_{pid}_{rev}`) so that World State blobs can be stored in a key-value map
without overwriting each other.

```json
{
  "id": "repair_Q42_123456789",          // Unique Benchmark ID
  "qid": "Q42",                          // Entity ID
  "property": "P569",                    // Property ID (e.g., Date of Birth)
  "track": "A_BOX",                      // Either "A_BOX" (Cleaner) or "T_BOX" (Reformer)
  "type": "TBD",                         // Taxonomy Class (Filled by Classifier)
  "violation_context": {
    "value": "+1952-03-11T00:00:00Z",    // The value that violated the constraint
    "report_fix_date": "2025-05-17T08:04:00Z",
    "report_revision_old": 2098075000,   // Report provenance for reproducibility
    "report_revision_new": 2098081111,
    "report_page_title": "Wikidata:Database reports/Constraint violations/P569"
  },
  "repair_target": {
    "kind": "A_BOX",
    "action": "UPDATE",                  // Operation: UPDATE, CREATE, DELETE
    "old_value": "+1952-02-11T00:00:00Z",
    "new_value": "+1952-03-11T00:00:00Z",
    "revision_id": 123456789,            // The Wikidata Revision ID of the fix
    "author": "User:KrBot"               // Who performed the repair
  },
  "persistence_check": {
    "status": "passed",                  // "passed" if entity exists in 2025
    "current_value_2025": ["+1952-03-11T00:00:00Z"]
  }
}

```

T-box entries share the same layout but extend the `repair_target` with property-level metadata:

```json
{
  "track": "T_BOX",
  "violation_context": {
    "value": null,
    "value_current_2025": ["+1952-03-11T00:00:00Z"],
    "... provenance fields omitted ..."
  },
  "repair_target": {
    "kind": "T_BOX",
    "property_revision_id": 213456789,
    "property_revision_prev": 213456700,
    "constraint_delta": {
      "hash_before": "a0c9…",
      "hash_after": "ff09…",
      "signature_before": "...deterministic json...",
      "signature_after": "...",
      "changed_constraint_types": ["Q21503250"],
      "old_constraints": [...],          // Optional: available for full world-state replay
      "new_constraints": [...]
    }
  }
}
```

An optional `ambiguous` flag and `ambiguous_reasons` array appear on A-box entries when both
the entity data and the property constraints changed inside the lookback window.

### 3.2 World State (`data/03_world_state.json`)

This file contains the frozen graph context required for the "Guardian" to reason. It is keyed by the `id` from the Repair Index.

```json
{
  "repair_Q42_123456789": {
    "L1_ego_node": {
      "qid": "Q42",
      "label": "Douglas Adams",          // Resolved English Label
      "description": "English author",   // Resolved English Description
      "properties": {                    // All properties of the entity (L1)
        "P31": ["Q5"],
        "P569": ["1952-03-11"]
      }
    },
    "L3_neighborhood": {
      "outgoing_edges": [                // 1-Hop Neighbors (L3)
        {
          "property_id": "P26",
          "target_qid": "Q12345",
          "target_label": "Jane Belson",
          "target_description": "Wife of Douglas Adams"
        }
      ]
    },
    "L4_constraints": {                  // The "Law" (L4)
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
    "constraint_change_context": {       // Only present for T-box / ambiguous entries
      "property_revision_id": 213456789,
      "property_revision_prev": 213456700,
      "signatures": {
        "before": {"hash": "a0c9…"},
        "after": {"hash": "ff09…"}
      },
      "constraints_before": [...],
      "constraints_after": [...]
    }
  }
}

```
