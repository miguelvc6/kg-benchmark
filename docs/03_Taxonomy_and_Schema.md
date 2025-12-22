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

### 3.1 Naming Conventions for Interpretable Mirrors

All machine-stable fields remain untouched, but we now add deterministic mirrors that make each artifact self-describing. The following suffixes are used consistently across Stage-2 and Stage-3:

| Suffix | Meaning |
| --- | --- |
| `_raw` | Byte-identical value captured from the source (kept for hashing/backwards compatibility). |
| `_qids` | Parsed list of normalized QIDs extracted from a free-form field. |
| `_label_en` / `_labels_en` | Resolved English labels for a single ID or an ordered list. |
| `_description_en` / `_descriptions_en` | English descriptions aligned with the same cardinality as the ID field. |
| `_aliases_en` | Alias list (single ID) or list-of-lists aligned with `_labels_en`. |

The fetcher owns the deterministic label cache stored at `data/cache/id_labels_en.json`. Every lookup goes through this cache to guarantee reproducibility.

### 3.2 Stage-2 Repair Index (`data/02_wikidata_repairs.json`)

Every record still captures the core event metadata (`id`, `qid`, `property`, `track`, etc.) and now includes:

* Human-friendly mirrors for `qid` and `property`.
* Parsed violation types (`report_violation_type_qids`) plus aligned labels/descriptions/aliases.
* Annotated value lists wherever the payload is a QID (`value`, `value_current_2025`, `current_value_2025`, `old_value`, `new_value`, etc.).
* Structured constraint signatures (`signature_before` / `signature_after`) accompanied by their raw string form, readable projections, and templated `rule_summaries_en`.

#### Example (A-box)

```json
{
  "id": "repair_Q2509775_2442825468",
  "qid": "Q2509775",
  "qid_label_en": "Alex Example",
  "qid_description_en": "Austrian illustrator",
  "qid_aliases_en": ["Alexander Example", "A. Example"],
  "property": "P21",
  "property_label_en": "sex or gender",
  "property_description_en": "sex or gender identity of the person or animal",
  "property_aliases_en": ["gender"],
  "track": "A_BOX",
  "violation_context": {
    "report_violation_type": "Type Q|5, Q|6581097",
    "report_violation_type_raw": "Type Q|5, Q|6581097",
    "report_violation_type_qids": ["Q5", "Q6581097"],
    "report_violation_type_labels_en": ["human", "female"],
    "report_violation_type_descriptions_en": [
      "any member of species Homo sapiens",
      "female sex or gender"
    ],
    "report_violation_type_aliases_en": [
      ["person", "people"],
      ["woman", "female person"]
    ],
    "value": ["Q6581072"],
    "value_labels_en": ["male"],
    "value_current_2025": ["Q6581097"],
    "value_current_2025_labels_en": ["female"],
    "report_fix_date": "2025-12-16T15:52:53Z",
    "report_revision_old": 2442751815,
    "report_revision_new": 2443041553
  },
  "repair_target": {
    "kind": "A_BOX",
    "action": "UPDATE",
    "old_value": ["Q6581072"],
    "old_value_labels_en": ["male"],
    "new_value": ["Q6581097"],
    "new_value_labels_en": ["female"],
    "revision_id": 2442825468,
    "author": "Jerimee"
  },
  "persistence_check": {
    "status": "passed",
    "current_value_2025": ["Q6581097"],
    "current_value_2025_labels_en": ["female"]
  }
}
```

#### Example (T-box delta)

```json
"constraint_delta": {
  "hash_before": "61625458bfae6ed69e8a1aa15777d0d0b1199a3c",
  "hash_after": "d74ae4fa10e1831bc2945243c1cc71a36f4af959",
  "signature_before": [
    {
      "constraint_qid": "Q21502838",
      "snaktype": "VALUE",
      "rank": "normal",
      "qualifiers": [
        {"property_id": "P2305", "values": ["Q5"]},
        {"property_id": "P2306", "values": ["P31"]}
      ]
    }
  ],
  "signature_after": [
    {
      "constraint_qid": "Q21502838",
      "snaktype": "VALUE",
      "rank": "normal",
      "qualifiers": [
        {"property_id": "P2305", "values": ["Q5", "Q6581097"]},
        {"property_id": "P2306", "values": ["P31"]}
      ]
    }
  ],
  "signature_before_raw": "[{\"constraint_qid\": ... }]",
  "signature_after_raw": "[{\"constraint_qid\": ... }]",
  "constraints_readable_en": {
    "before": [
      {
        "constraint_type": {"id": "Q21502838", "label_en": "type constraint"},
        "rank": "normal",
        "snaktype": "VALUE",
        "parameters": {
          "P2305": [{"id": "Q5", "label_en": "human"}],
          "P2306": [{"id": "P31", "label_en": "instance of"}]
        }
      }
    ],
    "after": [
      {
        "constraint_type": {"id": "Q21502838", "label_en": "type constraint"},
        "rank": "normal",
        "snaktype": "VALUE",
        "parameters": {
          "P2305": [
            {"id": "Q5", "label_en": "human"},
            {"id": "Q6581097", "label_en": "female"}
          ],
          "P2306": [{"id": "P31", "label_en": "instance of"}]
        }
      }
    ]
  },
  "rule_summaries_en": {
    "before": [
      "type constraint: class of property: human; property: instance of"
    ],
    "after": [
      "type constraint: class of property: human, female; property: instance of"
    ]
  },
  "changed_constraint_types": []
}
```

The hashes continue to reference the canonical serialization (sorted keys, `separators=(",", ":")`). `signature_*_raw` preserves the exact byte sequence that feeds the SHA1 digest.

### 3.3 Stage-3 World State (`data/03_world_state.json`)

World State entries remain keyed by the repair `id` and keep the four-layer contract. The enrichment adds alias metadata to the label layer and exposes structured constraint signatures inside the optional `constraint_change_context`.

```json
{
  "reform_Q2509775_P21_2442825468": {
    "L1_ego_node": {
      "qid": "Q2509775",
      "label": "Alex Example",
      "description": "Austrian illustrator",
      "aliases": ["Alexander Example"],
      "properties": {"P21": ["Q6581097"]}
    },
    "L2_labels": {
      "entities": {
        "Q2509775": {
          "label": "Alex Example",
          "description": "Austrian illustrator",
          "aliases": ["Alexander Example"]
        },
        "P21": {
          "label": "sex or gender",
          "description": "sex or gender identity of the person or animal",
          "aliases": ["gender"]
        }
      }
    },
    "L3_neighborhood": {
      "outgoing_edges": [
        {
          "property_id": "P735",
          "target_qid": "Q1234",
          "target_label": "First name",
          "target_description": "given name",
          "target_aliases": []
        }
      ]
    },
    "L4_constraints": {
      "property_id": "P21",
      "constraints": [
        {
          "constraint_type": {"qid": "Q21502838", "label": "type constraint"},
          "rule_summary": "class of property (P2305): human (Q5)"
        }
      ]
    },
    "constraint_change_context": {
      "property_revision_id": 2442825468,
      "property_revision_prev": 2440297725,
      "signatures": {
        "before": {
          "hash": "61625458bfae6ed69e8a1aa15777d0d0b1199a3c",
          "signature": [
            {"constraint_qid": "Q21502838", "qualifiers": [{"property_id": "P2305", "values": ["Q5"]}]}
          ],
          "signature_raw": "[{\"constraint_qid\": ... }]"
        },
        "after": {
          "hash": "d74ae4fa10e1831bc2945243c1cc71a36f4af959",
          "signature": [
            {"constraint_qid": "Q21502838", "qualifiers": [{"property_id": "P2305", "values": ["Q5","Q6581097"]}]}
          ],
          "signature_raw": "[{\"constraint_qid\": ... }]"
        }
      },
      "constraints_before": [...],
      "constraints_after": [...]
    }
  }
}
```

`constraints_before/after` remain optional heavyweight snapshots for replay experiments. The new signature block guarantees that the structured payload and the canonical raw string travel together.

### 3.4 Canonicalization & Migration Notes

* **Canonical Serializer:** `fetcher.canonicalize_json_structure` renders normalized constraint statements with sorted keys and no whitespace before hashing. This guarantees that identical statements always hash to the same digest regardless of formatting.
* **Backwards compatibility:** The fetcher automatically upgrades historical artifacts by populating `signature_*` (structured) and `signature_*_raw` when only a string was present. `report_violation_type_qids` is emitted even when parsing fails (empty list) so downstream tooling can rely on the field's presence.
* **Label Cache:** `data/cache/id_labels_en.json` is part of the deterministic build. Deleting it forces the resolver to re-query Wikidata; keeping it ensures byte-for-byte identical enriched outputs.

These conventions allow the benchmark to remain hash-stable while finally being readable by humans (and LLMs) without additional post-processing.
