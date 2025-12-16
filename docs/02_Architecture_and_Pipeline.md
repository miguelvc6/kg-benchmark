# Architecture & Data Pipeline

**Component:** WikidataRepairEval 1.0 Data Engine
**Implementation:** `fetcher.py`

---

## 1. System Overview: The Hybrid Index-Fetch Strategy

We reject the traditional approach of parsing the full Wikidata Revision History (multi-terabyte XML dumps) due to its low signal-to-noise ratio. Instead, we implement a **Hybrid Index-Fetch Strategy** that leverages existing community maintenance bots as "pointers" to high-quality repair events.

### The Core Pipeline

The system operates in three distinct, sequential stages. Each stage produces an immutable JSON artifact that serves as the input for the next.

| Stage       | Component                 | Responsibility                                                                                                            | Input Source                    | Output Artifact                    |
| ----------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ---------------------------------- |
| **1** | **Indexer**         | **Signal Detection:** Mines bot reports to find "Candidate Repairs" (entities that stopped violating a constraint). | `Wikidata:Database reports/*` | `data/01_repair_candidates.json` |
| **2** | **Fetcher**         | **Forensics:** Queries API history to find the exact diff and verifies persistence in 2025.                         | Wikibase REST API               | `data/02_wikidata_repairs.json`  |
| **3** | **Context Builder** | **World State:** Streams the 2025 JSON dump to attach frozen topological context.                                   | `latest-all.json.gz`          | `data/03_world_state.json`       |

---

## 2. Stage 1: The Indexer (Signal Detection)

### Objective

To identify a high-precision search space of potential repairs without scanning the entire graph history.

### Mechanism

The system monitors the revision history of **Wikidata Database Reports** (specifically pages under `Wikidata:Database reports/Constraint violations/`). These pages are updated periodically by community bots (e.g., KrBot).

* **Logic:** We treat the report page history as a time-series signal.
* Let $R_t$ be the set of QIDs listed in the report at time $t$.
* Let $R_{t+1}$ be the set of QIDs listed at time $t+1$.
* **Candidate Repair:** Any entity e such that $e \in R_t \land e \notin R_{t+1}$.
* **Auto-Discovery:** The script automatically scrapes the "Constraint Violations Summary" page to identify all ~2,000 properties currently tracked by bots, ensuring the benchmark covers the "Long Tail" of knowledge, not just popular properties like "Date of Birth".

---

## 3. Stage 2: The Fetcher (Forensics)

### Objective

To isolate the exact "Atomic Edit" that resolved the violation and verify its validity in the current graph.

### The "Lookback" Protocol

A critical engineering challenge is the latency between the **Repair Event** (User fixes the data) and the **Reporting Event** (Bot updates the list).

* **Analysis:** Statistical analysis of bot schedules reveals gaps of up to 6 days between report updates.
* **Configuration:** We set `REVISION_LOOKBACK_DAYS = 7`. This window ensures that even if a bot delays reporting, the fetcher will scan far enough back to find the user's edit.

### The Persistence Filter (Time-Travel Protection)

To ensure ecological validity, we enforce **Strict Persistence**.

* **Rule:** Before accepting a candidate, the system queries the **live** 2025 Wikidata API.
* **Check:** Does the Entity still exist? Does the Property still exist?
* **Outcome:** If the entity was deleted or merged in 2025, the historical repair is discarded. This sacrifices dataset size (~60% drop) to guarantee that all test cases are reproducible on the modern graph.

---

## 4. Stage 3: The Context Builder (World State Snapshot)

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
* **Result:** This reduces the runtime from >16 hours to ~8 hours while maintaining high data fidelity.

### Data Layers

For every benchmark case, the `world_state` object contains four distinct layers:

* **L1 (Ego Node):** All properties of the entity (for consistency checks).
* **L2 (Labels):** Human-readable names for LLM consumption.
* **L3 (Neighborhood):** 1-hop outgoing edges (for testing **Type B: Graph RAG**).
* **L4 (Constraints):** The full SHACL definition of the violated property (for testing **Type A: Logic**).
