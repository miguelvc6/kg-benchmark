# Architecture & Data Pipeline

**Component:** WikidataRepairEval 1.0 Data Engine
**Implementation:** `fetcher.py`

---

## 1. System Overview: The Hybrid Index-Fetch Strategy

We reject the traditional approach of parsing the full Wikidata Revision History (multi-terabyte XML dumps) due to its low signal-to-noise ratio. Instead, we implement a **Hybrid Index-Fetch Strategy** that leverages existing community maintenance bots as "pointers" to high-quality repair events.

### The Core Pipeline

The system operates in three distinct, sequential stages. Each stage produces an immutable JSON artifact that serves as the input for the next.

| Stage       | Component                 | Responsibility                                                                                                        | Input Source                    | Output Artifact                    |
| ----------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ---------------------------------- |
| **1** | **Indexer**         | **Signal Detection:** Mines reports to find "Candidate Repairs" (entities that stopped violating a constraint). | `Wikidata:Database reports/*` | `data/01_repair_candidates.json` |
| **2** | **Fetcher**         | **Forensics:** Queries API history to find the exact diff and verifies persistence in 2025.                     | Wikibase REST API               | `data/02_wikidata_repairs.json`  |
| **3** | **Context Builder** | **World State:** Streams the 2025 JSON dump to attach frozen topological context.                               | `latest-all.json.gz`          | `data/03_world_state.json`       |

---

## 2. Stage I: The Indexer (Signal Detection)

### Objective

To identify a high-precision search space of potential repairs without scanning the entire graph history.

### Mechanism

The system monitors the revision history of **Wikidata Database Reports** (specifically pages under `Wikidata:Database reports/Constraint violations/`). These pages are updated periodically by community bots (e.g., KrBot).

**Logic:** We treat the report page history as a time-series signal.

* Let $R_t$ be the set of QIDs listed in the report at time $t$.
* Let $R_{t+1}$ be the set of QIDs listed at time $t+1$.
* **Candidate Repair:** Any entity e such that $e \in R_t \land e \notin R_{t+1}$ **and is also missing from the entire report at $t_{+1}$**. If the QID merely moved to a different constraint section, the candidate is discarded as a section reclassification rather than a fix. This prevents false positives that would otherwise pollute the fetcher stage.
* **Auto-Discovery:** The script automatically scrapes the "Constraint Violations Summary" page to identify all ~2,000 properties currently tracked by bots, ensuring the benchmark covers the "Long Tail" of knowledge, not just popular properties like "Date of Birth".

---

## 3. Stage II: The Fetcher

### Objective

To isolate the exact "Atomic Edit" that resolved the violation and verify its validity in the current graph. Stage 2 now labels each case as either an **A-box** (Cleaner) or **T-box** (Reformer) repair and preserves provenance so downstream evaluations can differentiate schema vs. data evolution.

### The "Lookback" Protocol

A critical engineering challenge is the latency between the **Repair Event** (User fixes the data) and the **Reporting Event** (Bot updates the list).

* **Analysis:** Statistical analysis of bot schedules reveals gaps of up to 6 days between report updates.
* **Configuration:** We set `REVISION_LOOKBACK_DAYS = 7`. This window ensures that even if a bot delays reporting, the fetcher will scan far enough back to find the user's edit.
* **Scan:** Stage II starts from each `(qid, property_id, fix_date)` to anchor a 7‑day window around the report disappearance. Within that window it inspects the entity’s own history for an A‑box edit on the subject entity that touched the property implicated by the violation. If no entity-level change is found, Stage II switches to the property’s history fetching P2302 claims for each revision to detect changes to the constraint definition (T‑box edits) and stores the before/after constraint hashes plus snapshots.
* **Ambiguity Guard:** When an A-box repair is found, the fetcher performs a secondary, signature-only scan from the end of the property history (bounded to ~25 revisions) to detect whether the constraint definition changed in the same window. Such cases are flagged as `ambiguous` to inform evaluation splits.

### The Persistence Filter (Time-Travel Protection)

To ensure ecological validity, we enforce **Strict Persistence** *after* identifying the candidate repair. This sequencing allows legitimate DELETE actions to remain in the dataset while still rejecting stale candidates that no longer exist in 2025.

* **Rule:** Once the repair type is known, the system queries the **live** 2025 Wikidata API for the affected entity/property.
* **Check:** Does the entity still exist? Does the property definition still exist? If the human fix was a DELETE, the absence of the property in 2025 is acceptable.
* **Outcome:** Non-DELETE repairs with missing live values are discarded, preserving reproducibility without biasing against deletion cases.

### Dual-Track Detection & Provenance

* **A-box Path:** Diffs the entity revision history to capture action (`CREATE/UPDATE/DELETE`), old/new value snapshots, and the repair revision ID.
* **T-box Path:** Diffs the property page history (P2302 claims) and records the *latest* constraint signature change within the window, storing deterministic SHA1 hashes plus optional snapshots of the previous and new constraint statements.
* **Ambiguous Cases:** When both paths detect changes, the entry receives `ambiguous` metadata and contributes to a dedicated stats counter.
* **Report Provenance:** Every record includes report fix timestamps and revision IDs from Stage 1 so failures can be traced back to the originating bot diff.

### Deterministic Label Resolution & Mirroring

Stage 2 now emits human-readable mirrors for every machine-stable identifier without sacrificing determinism.

* **Label Resolver Module:** `fetcher.py` instantiates a `LabelResolver` that batches IDs through `wbgetentities`, persists the responses inside `data/cache/id_labels_en.json`, and reuses the cache on subsequent runs. The resolver now returns `{label_en, description_en}` only and falls back to `null` values with a warning whenever Wikidata cannot resolve an ID. The cache file is canonicalized (`sort_keys=True`) to keep hashes stable.
* **Naming Convention:** Raw fields remain unchanged; the pipeline adds siblings such as `qid_label_en`, `property_description_en`, `report_violation_type_qids`, `value_current_2025_labels_en`, etc. `_raw` preserves the original string, `_qids` lists parsed IDs, and `_label_en/_labels_en` plus `_description_en/_descriptions_en` carry the interpreted mirrors while aliases remain intentionally excluded.
  "Aliases are not stored by design to avoid multilingual noise, prompt bloat, and unintended information leakage. Labels and descriptions are sufficient for all Phase-1 experiments."
* **Constraint Introspection:** P2302 qualifier IDs are now expanded into `{id,label_en,...}` tuples under `constraints_readable_en`, and the same lookup powers the templated `rule_summaries_en` strings used in docs and classifiers.

### Structured Constraint Signatures

The old design stored deterministic constraint signatures as JSON strings. We now keep both representations:

* **Canonical serialization:** `canonicalize_json_structure()` renders normalized statements using sorted keys and `separators=(",", ":")` prior to hashing, giving us whitespace-free hashes.
* **Dual fields:** `signature_before` / `signature_after` are structured lists of qualifier dicts, while `signature_before_raw` / `signature_after_raw` retain the byte-identical string used in the original benchmark. `constraints_readable_en` and `rule_summaries_en` are derived strictly from the structured payload to avoid re-parsing strings downstream.

---

## 4. Stage III: The Context Builder (World State Snapshot)

### Objective

To construct the "frozen" informational environment required for the Guardian evaluation. This stage decouples the *state of the graph* from the *retrieval process*.

### The Single-Pass Streaming Architecture

Loading the full Wikidata JSON dump (>100GB compressed) into RAM is impossible. We utilize a **Single-Pass Stream-and-Filter** architecture using `ijson`.

1. **Target Loading:** The script loads the set of `focus_ids` (from Stage 2) into a hash set.
2. **Streaming:** It reads `latest-all.json.gz` line-by-line.

* If `entry['id']` is in the target set, the entity is fully materialized.
* If not, it is skipped immediately.

3. **Output Buffering:** To prevent OOM (Out of Memory) crashes, the system flushes the built contexts to `data/03_world_state.json` in batches of 10 entries.

### The "Hybrid Context" Strategy

To efficiently capture the 1-hop neighborhood without scanning the dump twice (which would take ~16 hours):

* **Primary Data:** The Focus Node and the Property Definition are loaded from the **Dump** (ensuring schema consistency).
* **Secondary Data:** The labels and descriptions of **Neighbors** (Target QIDs) are fetched via the **Wikidata API** in batch mode (50 IDs/request).
* **Constraint Change Context:** For T-box entries the builder also records the hashes (and optionally the serialized statements) before/after the constraint change under `constraint_change_context`, enabling schema-evolution experiments.
* **Result:** This reduces the runtime from >16 hours to ~8 hours while maintaining high data fidelity and ensuring that no world-state entry is overwritten (duplicate IDs are rejected with a warning).

### Data Layers

For every benchmark case, the `world_state` object contains four distinct layers:

* **L1 (Ego Node):** All properties of the entity (for consistency checks).
* **L2 (Labels):** Human-readable names plus descriptions for every QID/PID touched by the context. Aliases remain intentionally excluded to keep the context deterministic and compact.
* **L3 (Neighborhood):** 1-hop outgoing edges (for testing **Type B: Graph RAG**).
* **L4 (Constraints):** The full SHACL definition of the violated property (for testing **Type A: Logic**).
