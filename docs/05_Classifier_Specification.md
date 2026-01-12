# Classifier Specification: The Information Necessity Labeler

**Component:** `classifier.py`
**Input Artifacts:** `data/02_wikidata_repairs.json`, `data/03_world_state.json`
**Output Artifact:** `data/04_classified_benchmark.json`

---

## 1. Objective

To algorithmically assign a **Complexity Class** (Type A, B, or C) to each repair event based on the *Information Necessity* principle. The classifier acts as the "Ground Truth" for the ablation studies, determining whether a repair *should* have required retrieval to solve.

## 2. The Logic: Hierarchical Subtraction

We do not classify by constraint type alone. Instead, we use a **Subtraction Logic**: we test if the information was available locally; if not, we test if it was self-evident; only then do we declare it "External."

### Decision Tree Pseudo-Code

```python
def classify(repair_event, world_state):
    truth = repair_event['repair_target']['value']
  
    # 1. Negative Sampling Check
    if repair_event['repair_target']['action'] == 'DELETE':
        return "Type A (Rejection)" # Deleting bad data is a logical rejection

    # 2. Local Availability Check (Type B)
    if search_local_context(truth, world_state):
        return "Type B (Local)"

    # 3. Intrinsic Logic Check (Type A)
    if is_logical_constraint(world_state['constraint_metadata']):
        return "Type A (Logical)"

    # 4. Fallback (Type C)
    return "Type C (External)"

```

---

## 3. Detailed Logic Specifications

###3.1 Step 1: The Local Search (Type B Detection)
**Goal:** Determine if the answer (the "Truth") was already present in the graph topology or node text.

* **Search Scope:**
* **Focus Node:** Start with the Stage-2 mirrors (`qid_label_en`, `qid_description_en`) and fall back to the World State `L1_ego_node` for additional literal evidence. Aliases are intentionally excluded to reduce multilingual noise and leakage.
* **1-Hop Neighbors:** Target QID Labels and Descriptions from `neighborhood_snapshot`.
* **Matching Logic:**
* **Exact ID Match:** If the Truth is a QID (e.g., `Q123`), check if `Q123` exists in the `outgoing_edges` list.
* **Fuzzy String Match:** If the Truth is a string/date/quantity, perform a normalized substring check against the text fields.
* *Normalization:* Lowercase, strip punctuation.
* *Threshold:* Exact substring required (to avoid false positives).
* **Edge Case:** "Date Precision." If Truth is `1952-03-11`, finding `1952` in the description is **INSUFFICIENT**. The context must provide equal or greater precision.

### 3.2 Step 2: The Constraint Whitelist (Type A Detection)

**Goal:** Identify repairs that are solvable purely by understanding the rule, without needing data.

* **Logic:** Check the `constraint_type` QID against a hardcoded whitelist.
* **Type A Whitelist:**
* **Format Constraint (`Q21502404`):** Regex violations (e.g., "URL must start with https").
* **One-of Constraint (`Q21502402`):** Allowed values are enumerated in the constraint definition itself.
* **Range Constraint (`Q21510860`):** Numerical bounds (Min/Max).
* **Inverse Constraint (`Q21510855`):** (Debatable, but often implies a logical symmetry fix if the inverse edge exists). *Decision:* Treat as Type B if the inverse edge exists, otherwise Type A if it's a schema fix.
* **Readable metadata:** Prefer the Stage-2 `constraints_readable_en` and `rule_summaries_en` projections before attempting to parse raw constraint statements. They already contain `{id,label}` tuples for every qualifier.

### 3.3 Step 3: The External Void (Type C)

**Goal:** Catch everything else.

* **Definition:** If the information isn't in the neighborhood and isn't self-evident from the rule, it must come from the outside world.
* **Typical Cases:** `Mandatory Value`, `Contemporary`, `Conflicts With`.

---

## 4. Implementation Guidelines

### 4.1 Input Handling* Load `02_wikidata_repairs.json` (The list of events).

* Load `03_world_state.json` (The dictionary of context).
* **Join:** Iterate through `02`, using the `id` field to look up the context in `03`.

Stage-2 entries already expose deterministic mirrors for all IDs (`qid_label_en`, `property_description_en`, `value_current_2025_labels_en`, `report_violation_type_qids`, `constraints_readable_en`, etc.). Use these fields before resorting to bespoke label resolution—they are backed by the shared label cache, include English labels/descriptions, and omit aliases per the benchmark policy.

"Aliases are not stored by design to avoid multilingual noise, prompt bloat, and unintended information leakage. Labels and descriptions are sufficient for all Phase-1 experiments."

### 4.2 Handling DELETE Operations* A `DELETE` operation usually implies the existing value was wrong.

* **Classification:**
* If the constraint was "Format" or "Range" \rightarrow **Type A** (Cleaning noise).
* If the constraint was "Mandatory Value" \rightarrow **Type A** (Rejecting the schema applicability).
* *Default Policy:* Label all DELETEs as **Type A (Rejection)** for the purpose of the "Writer" baseline, as they require recognizing falsity rather than retrieving truth.

### 4.3 Output Schema

The script should produce a new file `data/04_classified_benchmark.json` which clones the input repair object and adds a `classification` block:

```json
{
  "id": "repair_Q42_...",
  ...
  "classification": {
    "class": "Type B",
    "reason": "Found exact match for 'Q123' in neighborhood snapshot.",
    "constraint_type_qid": "Q21503250" 
  }
}

```
