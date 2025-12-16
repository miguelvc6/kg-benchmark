# Taxonomy & Data Schema

**Artifact Definition:** WikidataRepairEval 1.0
**Context:** Classification of Knowledge Graph Errors & JSON Specifications

---

## 1. The Taxonomy of Information Necessity

To enable the ablation studies required for **RQ2 (The Information Gap)**, the benchmark algorithmically sorts violations into three complexity classes based on the *source of information* required to resolve them.

### Type A: Logical (Internal Consistency)*

**Definition:** The violation is detectable and resolvable purely by analyzing the internal consistency of the statement's literals or the schema definition. No graph traversal is required.

* **Representative Constraints:** Format (Regex), One-of (Enumeration), Range (Min/Max).
* **Hypothesis:** LLMs can solve these with **Zero-Shot Logic** (No retrieval needed).
* **Example:** "End Date (1900) < Start Date (1950)".

### Type B: Local (Topological Reasoning)*

**Definition:** The violation requires checking the properties of the *Subject* or *Object* entities involved in the triple (strictly \le 1 hop). This includes textual descriptions stored on the node.

* **Representative Constraints:** Contemporary, Inverse, Distinct Value, Type Constraints.
* **Hypothesis:** LLMs require **Graph RAG** (Topology) to solve these, but text retrieval introduces noise.
* **Example:** "Subject is an instance of 'Human', but Property 'Founded By' expects 'Organization'."

### Type C: External (Information Void)*

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

### 3.1 Repair Index (`data/02_wikidata_repairs.json`)This file is the "Gold Standard" log of historical repairs.

```json
{
  "id": "repair_Q42_123456789",          // Unique Benchmark ID
  "qid": "Q42",                          // Entity ID
  "property": "P569",                    // Property ID (e.g., Date of Birth)
  "type": "TBD",                         // Taxonomy Class (Filled by Classifier)
  "violation_context": {
    "value": "+1952-03-11T00:00:00Z"     // The value causing the violation
  },
  "repair_target": {
    "action": "UPDATE",                  // Operation: UPDATE, CREATE, DELETE
    "value": "+1952-03-11T00:00:00Z",    // The Fixed Value (Ground Truth)
    "revision_id": 123456789,            // The Wikidata Revision ID of the fix
    "author": "User:KrBot"               // Who performed the repair
  },
  "persistence_check": {
    "status": "passed",                  // "passed" if entity exists in 2025
    "current_value_2025": "..."          // The value in the live graph
  }
}

```

### 3.2 World State (`data/03_world_state.json`)

This file contains the frozen graph context required for the "Guardian" to reason. It is keyed by the `id` from the Repair Index.

```json
{
  "repair_Q42_123456789": {
    "focus_node": {
      "qid": "Q42",
      "label": "Douglas Adams",          // Resolved English Label
      "description": "English author",   // Resolved English Description
      "properties": {                    // All properties of the entity (L1)
        "P31": ["Q5"],
        "P569": ["1952-03-11"]
      }
    },
    "neighborhood_snapshot": {
      "outgoing_edges": [                // 1-Hop Neighbors (L3)
        {
          "property_id": "P26",
          "target_qid": "Q12345",
          "target_label": "Jane Belson",
          "target_description": "Wife of Douglas Adams"
        }
      ]
    },
    "constraint_metadata": {             // The "Law" (L4)
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
    }
  }
}

```
