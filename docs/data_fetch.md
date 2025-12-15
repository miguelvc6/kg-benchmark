# Technical Report: Data Acquisition & Validation Protocol for WikidataRepairEval 1.0

**Project:** Dynamics of Neuro-Symbolic Alignment (Phase 1)
**Date:** December 10, 2025
**Status:** Protocol Definition & Indexing Phase
**Context:** Construction of a specialized Knowledge Graph repair benchmark distinguishing between reasoning failures and information voids.

---

## 1. Objective

To construct **WikidataRepairEval 1.0**, a "Gold Standard" benchmark of historical KG repairs. Unlike existing resources that aggregate static dumps, this dataset reconstructs the *dynamic state* of the graph at the moment of repair. It is designed to enable the "Guardian" hypothesis testing by isolating three variables: Logical Consistency, Topological Reasoning, and External Retrieval.

## 2. Methodology: The Hybrid Index-Fetch Strategy

We reject the naive approach of parsing the full Wikidata Revision History (multi-terabyte XML dumps) due to low signal-to-noise ratio. Instead, we implement a **Hybrid Index-Fetch Strategy** that leverages existing community maintenance bots as "pointers" to high-quality repair events.

### 2.1. The Pipeline Architecture

1. **The Indexer (Signal Detection):**
   * **Source:** `Wikidata:Database reports/Constraint violations/*`.
   * **Mechanism:** Wikidata generates static pages listing current violations. We parse the *history* of these report pages. If Entity $E$ is listed in the violation report at time $T_{report}$ and removed at time $T_{next}$, a candidate repair occurred in the interval $(T_{report}, T_{next})$.
   * **Benefit:** This pre-filters the search space to entities known to have violated specific constraints (e.g., `P569` Date of Birth, `P570` Date of Death), ensuring high relevance.
   * **Output** repair_candidates.json
  
2. **The Fetcher (Forensics):**
   * **Source:** Wikibase REST API (`GET /w/rest.php/v1/page/{id}/history`).
   * **Mechanism:** For every candidate $(E, T_{interval})$, we query the API to isolate the exact revision $R$ where the violating statement was modified.
   * **Diff Extraction:** We capture the *Before* state (Constraint Violated) and the *After* state (Constraint Satisfied).
   * **Output** wikidata_repair_eval_raw.json

3. **The Context Builder (World State):**
    We need the full graph neighborhood to test "Type B" (Local) reasoning.
   * **Source:** `latest-all.json.gz` (2025 Dump).
   * **Mechanism:** To ensure the benchmark evaluates *current* reasoning capabilities, we cross-reference the historical repair against the live graph.

### **2.2. Expanded Pipeline: The Context Builder (World State)**

The **Context Builder** acts as the "Freezing Mechanism." Its job is to capture the local topology of the entity *as it exists in the 2025 standard dump* and attach it to the repair event. This creates a static, reproducible test case.

#### **A. The Extraction Strategy: Stream-and-Filter**

We cannot load the 1TB+ JSON dump into memory. Instead, we perform a **single-pass stream** over `latest-all.json.gz`.

  * **Input:** The `wikidata_repair_eval_raw.json` (from Step 2), creating a `Set<QID>` of target entities.
  * **Process:** Stream the dump. If `entry['id']` is in our `Set<QID>`, we do not just save the line; we extract specific "Context Layers."
  * **Output:** A `world_state.json` map linked to each test case.

#### **B. The Data Manifest: What Must Be Extracted?**

[cite_start]To enable the "Ablation Study" (RQ2)[cite: 89], we need to extract four distinct layers of data from the dump for every target entity.

| Layer | Data Element | Why is it needed? |
| :--- | :--- | :--- |
| **L1** | **The Ego Node (Direct Properties)** | [cite_start]**Persistence Filtering[cite: 44].** We must verify that the *current* state of the entity still matches the "Pre-Repair" or "Post-Repair" logic. If the entity has been deleted or merged in the dump, the case is discarded. |
| **L2** | **The Labels & Descriptions** | **LLM Readability.** An LLM cannot reason about `Q42` having `P569`. It needs "Douglas Adams" and "Date of Birth." We must resolve the IDs to English labels. |
| **L3** | **The 1-Hop Neighborhood (Type B)** | [cite_start]**Taxonomy Classification[cite: 34].** To classify a case as **Type B (Local)**, we must check if the solution exists in the immediate neighbors. (e.g., *Is the 'Start Date' present on the spouse entity?*). |
| **L4** | **The Constraint Definition (P2302)** | [cite_start]**The "Guardian" Logic[cite: 5].** We need the *current* SHACL definition of the constraint (e.g., "P569 domain: Human") to verify if the violation is real or if the rule itself changed. |

#### **C. The JSON Schema for "World State"**

The final "Context Object" attached to every benchmark case must look like this to support the experiments defined in your proposal:

```json
"world_state": {
  "focus_node": {
    "qid": "Q42",
    "label": "Douglas Adams",
    "description": "English author and humorist", 
    "properties": {
      "P31": ["Q5"], // Instance of Human
      "P569": ["1952-03-11"] // Date of Birth
    }
  },
  "neighborhood_snapshot": { 
    [cite_start]// This supports "Graph RAG" (Type B) evaluation [cite: 34, 131]
    "outgoing_edges": [
      {
        "property_id": "P26", // Spouse
        "target_qid": "Q12345",
        "target_label": "Jane Belson",
        "target_description": "Wife of Douglas Adams" 
        // Note: We do NOT recurse deeper than 1 hop.
      }
    ]
  },
  "constraint_metadata": {
    [cite_start]// This supports the "Guardian" logic verification [cite: 5, 19]
    "property_id": "P569",
    "constraint_type": "Q21503250", // "Constraint: contemporary"
    "rule_summary": "P569 must be greater than P570 (Date of Death) if P570 exists."
  }
}
```


## 3. Taxonomy Standards (The Classifiers)

To satisfy the requirements of **RQ2 (The Information Gap)**, every fetched repair is algorithmically classified into one of three complexity tiers. These definitions are now finalized based on the project's specific constraints.

### Type A: Logical (Internal Consistency)

* **Definition:** The violation is detectable and resolvable purely by analyzing the internal consistency of the statement's literals.
* **Example:** "End Date (1900) < Start Date (1950)".
* **Requirement:** No graph traversal; no external search.

### Type B: Local (Topological Reasoning)

* **Definition:** The violation requires checking the properties of the *Subject* or *Object* entities involved in the triple.
* **Decision:** Textual descriptions (e.g., `schema:description`) stored within the node are classified as **Local Topology**.
* **Hop Limit:** Strictly $< 2$ hops.
* **Example:** "Subject is an instance of 'Human', but Property 'Founded By' expects 'Organization'."

### Type C: External (Information Void)

* **Definition:** The violation can only be resolved by retrieving information not present in the graph (or present only in $> 2$ hop neighborhoods).
* **Requirement:** Access to the **Tavily Search API**.
* **Example:** "Paris is the capital of [Missing Value]."

## 4. Quality Control & Validity Protocols

To ensure the benchmark measures *reasoning* rather than *gaming* or *hallucination*, we implement the following strict filters.

### 4.1. The Persistence Filter (Time-Travel Paradox)

We address the risk of "stale knowledge" (evaluating 2019 repairs that are no longer valid in 2025).

* **Protocol:** Strict Persistence.
* **Logic:** A historical repair candidate $(E, P, V)$ is **discarded** if the tuple does not exist in the current 2025 Wikidata dump.
* **Implication:** We accept a significant reduction in dataset size (estimated ~60% drop) in exchange for absolute ecological validity. The benchmark will only contain errors that are theoretically reproducible on the live graph today.

### 4.2. Frozen Retrieval Snapshots

To decouple the variance of the search engine from the variance of the model.

* **Protocol:** For all **Type C** cases, the search results are pre-fetched and frozen.
* **Tool:** **Tavily API**.
* **Artifact:** The benchmark dataset will include a `retrieval_context` field containing the static JSON response from Tavily. Future evaluations must use this field, not live web search.

### 4.3. The Anti-Gaming Penalty

To penalize "Lazy Repairs" (e.g., deleting a node to silence a constraint error).

* **Metric:** `Information Preservation Score` ($S_{info}$).
* **Formula:**
  $$
  _{info} = \begin{cases} 
  1.0 & \text{if } Action_{Model} = Action_{Human} \\
  -0.5 & \text{if } Action_{Model} = \text{DELETE} \land Action_{Human} = \text{UPDATE} \\
  0.0 & \text{otherwise}
  \end{cases}
  $$
* **Rationale:** A penalty of `-0.5` ensures that a model which systematically deletes data to pass validation will score lower than a model that simply fails to repair.

## 5. Next Steps

1. **Execution of `indexer.py`:** Mine the `repair_candidates.json` list using the `mwclient` library.
2. **Implementation of `fetcher.py`:** Build the REST API client to extract the precise diffs.
3. **JSON Schema Validation:** Define the strict output schema for `WikidataRepairEval_Entry` to ensure compatibility with the training loop in Phase 2.
