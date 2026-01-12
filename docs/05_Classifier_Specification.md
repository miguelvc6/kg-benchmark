# Classifier Specification: The Information Necessity Labeler

**Component:** `classifier.py`
**Input Artifacts:** `data/02_wikidata_repairs.json` (or `.jsonl`), `data/03_world_state.json`
**Output Artifacts:** `data/04_classified_benchmark.jsonl` (LEAN), `data/04_classified_benchmark_full.jsonl` (FULL, optional)

---

## 1. Objective

To algorithmically assign a **Complexity Class** (Type A, B, or C) to each repair event based on the *Information Necessity* principle. The classifier acts as the "Ground Truth" for the ablation studies, determining whether a repair *should* have required retrieval to solve.

## 2. The Logic: Hierarchical Subtraction (Rule-Deterministic First)

We do not classify by constraint type alone. Instead, we use a **Subtraction Logic**: we test if the information was available locally; if not, we test if it was self-evident; only then do we declare it "External."

### Decision Tree Pseudo-Code

```python
def classify(repair_event, world_state):
    truth, truth_source, truth_applicable = get_truth_info(repair_event)

    # 1. Negative Sampling Check
    if repair_event['repair_target']['action'] == 'DELETE':
        return "Type A (Rejection)"

    # 2. Rule-Deterministic Check (Type A)
    if rule_deterministic_constraint(repair_event, world_state, truth):
        return "Type A (Logical)"

    # 3. Local Availability Check (Type B)
    if search_local_context(truth, world_state):
        return "Type B (Local)"

    # 4. Fallback (Type C)
    return "Type C (External)"

```

---

## 3. Detailed Logic Specifications

### 3.1 Step 1: The Local Search (Type B Detection)
**Goal:** Determine if the answer (the "Truth") was already present in the graph topology or node text.

* **Search Scope:**
* **Focus Node:** Start with the Stage-2 mirrors (`qid_label_en`, `qid_description_en`) and fall back to the World State `L1_ego_node` for additional literal evidence. Aliases are intentionally excluded.
* **1-Hop Neighbors:** Target QID labels and descriptions from `L3_neighborhood`.
* **Synthetic Pre-Repair Values:** The target property on the focus node is reconstructed using `repair_target.old_value` (or `violation_context.value`) to avoid post-repair leakage.
* **Matching Logic:**
* **Exact ID Match:** If the Truth is a QID (e.g., `Q123`), check in neighbor IDs or synthetic pre-repair focus-property IDs.
* **Fuzzy String Match:** If the Truth is a string/date/quantity, perform a normalized substring check against focus text or neighbor text.
* *Normalization:* Lowercase, strip punctuation.
* *Threshold:* Exact substring required (to avoid false positives).
* **Edge Case:** "Date Precision." If Truth is `1952-03-11`, finding `1952` in the description is **INSUFFICIENT**. The context must provide equal or greater precision.

### 3.2 Step 2: Rule-Deterministic Constraints (Type A Detection)

**Goal:** Identify repairs that are solvable purely by understanding the rule, without needing data.

* **Logic:** A repair is Type A only when the rule itself uniquely determines the fix.
* **Deterministic cases:**
* **Format Constraint (`Q21502404`):** Treated as deterministic (rule-driven).
* **One-of Constraint (`Q21510859`, `Q21502402`):** Deterministic only if allowed set size is 1.
* **Range Constraint (`Q21510860`):** Deterministic only when the repair equals a boundary value implied by min/max.
* **Readable metadata:** Use `L4_constraints.constraints[*]` where available; fall back to report type only for format.

### 3.3 Step 3: The External Void (Type C)

**Goal:** Catch everything else.

* **Definition:** If the information isn't in the neighborhood and isn't self-evident from the rule, it must come from the outside world.
* **Typical Cases:** `Mandatory Value`, `Contemporary`, `Conflicts With`.

---

## 4. Implementation Guidelines

### 4.1 Input Handling
* Load `02_wikidata_repairs.json` or `.jsonl` (streamed).

* Load `03_world_state.json` (The dictionary of context).
* **Join:** Iterate through `02`, using the `id` field to look up the context in `03`.

Stage-2 entries already expose deterministic mirrors for all IDs (`qid_label_en`, `property_description_en`, `value_current_2025_labels_en`, `report_violation_type_qids`, `constraints_readable_en`, etc.). Use these fields before resorting to bespoke label resolution—they are backed by the shared label cache, include English labels/descriptions, and omit aliases per the benchmark policy.

"Aliases are not stored by design to avoid multilingual noise, prompt bloat, and unintended information leakage. Labels and descriptions are sufficient for all Phase-1 experiments."

### 4.2 Handling DELETE Operations
* A `DELETE` operation usually implies the existing value was wrong.

* **Classification:**
* If the constraint was "Format" or "Range" \rightarrow **Type A** (Cleaning noise).
* If the constraint was "Mandatory Value" \rightarrow **Type A** (Rejecting the schema applicability).
* *Default Policy:* Label all DELETEs as **Type A (Rejection)** for the purpose of the "Writer" baseline, as they require recognizing falsity rather than retrieving truth.

### 4.3 Output Schema

The script should produce a new file `data/04_classified_benchmark.jsonl` which clones the input repair object and adds a `classification` block:

```json
{
  "id": "repair_Q42_...",
  ...
  "classification": {
    "class": "TypeB",
    "subtype": "LOCAL_NEIGHBOR_IDS",
    "confidence": "high",
    "decision_trace": [...],
    "rationale": "Truth tokens matched neighbor identifiers.",
    "constraint_types": [...],
    "local_subtype": "LOCAL_NEIGHBOR_IDS",
    "diagnostics": {
      "truth_applicable": true,
      "truth_tokens": ["Q123"],
      "truth_source": "repair_target.new_value"
    }
  }
}

```
