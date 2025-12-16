# Scientific Vision: The Dynamics of Neuro-Symbolic Alignment

**Project Scope:** Phase 1 – The Unified Benchmark (WikidataRepairEval)
**Core Thesis:** Sustainable Neuro-Symbolic systems require a formalized "Guardian" mechanism to mediate the flow of knowledge between stochastic LLMs and rigid Knowledge Graphs.

---

## 1. The Core Problem

Large Language Models (LLMs) and Knowledge Graphs (KGs) have complementary failure modes:

* **LLMs** suffer from hallucination and "knowledge staleness".
* **KGs** suffer from incompleteness and brittleness.

Current approaches treat these as separate problems: **RAG** uses KGs to patch LLMs, while **Information Extraction** uses LLMs to patch KGs.
**My thesis is that these must be treated as a single, coupled dynamic system**. An LLM that learns only from its own verified output (a self-consuming loop) will eventually suffer model collapse unless grounded by a continuous stream of external, structure-aware verification.

---

## 2. The "Guardian" Hypothesis

I argue that a sustainable neuro-symbolic system requires a **formalized transaction protocol**—a "Guardian"—that mediates the flow of information.

### Hypothesis 1: The Rejection Filter

Symbolic constraints (SHACL/OWL) acting as a rejection sampling filter are **strictly necessary** to prevent semantic drift in LLM-generated knowledge. Without this "immune system," the probabilistic nature of the model inevitably degrades the graph's integrity.

### Hypothesis 2: The Cycle

Improving the **semantics** (truthfulness) of a model, rather than just its **syntax** (SPARQL generation), requires an "Active Grounding" retrieval layer. The stream of strictly verified repairs becomes the safe training data for subsequent fine-tuning.

---

## 3. Research Questions (The Scientific Roadmap)

This project (WikidataRepairEval) is the experimental apparatus designed to answer the following specific hypotheses:

### RQ1: The Protocol Definition*

**Question:** What are the necessary semantic components of a "Knowledge Transaction"?

* **Investigation:** Moving beyond simple triples to a formal schema that encapsulates *Proposal, Rationale, Provenance, and Uncertainty*.

### RQ2: The Information Gap (Ablation Study)*

**Question:** What is the marginal utility of unstructured context (Text RAG) versus structured topology (Graph RAG) in resolving knowledge violations?

* **Investigation:** We distinguish between errors caused by **reasoning failures** (logic) vs. **informational voids** (missing data). Does adding Wikipedia text actually help verify a constraint, or does it introduce noise compared to the graph neighborhood?.

### RQ3: The Verification Trade-off*

**Question:** To what extent can Automated Constraint Verification (SHACL) replace human oversight without compromising KG integrity?

* **Metric:** The **False Positive Acceptance Rate**—how often does the system accept a plausible-sounding but factually wrong patch?.

### RQ4: Loop Dynamics & Model Collapse*

**Question:** Does fine-tuning an LLM on its own verified repairs lead to genuine capability gains, or mere overfitting to the constraint logic?

---

## 4. The "Delta": Why a New Benchmark?

Existing resources (Tanon, Ferranti, Lin) are fragmented, outdated, or purely functional. They measure **if** a repair was made, but not **what information** was required to make it.

**WikidataRepairEval 1.0** differentiates itself through:

1. **Taxonomy of Information Necessity:** Instead of grouping by "Constraint Type" (e.g., Cardinality), we group by the information source required to solve the problem:

* **Type A (Logical):** Solvable by internal consistency (Zero-Shot).
* **Type B (Local):** Solvable by graph topology (Graph RAG).
* **Type C (External):** Solvable only by retrieval (Web RAG).

2. **Ecological Validity:** Unlike synthetic datasets, we use historical human repairs filtered for **persistence** in the 2025 graph, ensuring the benchmark reflects real-world modeling challenges.
3. **Differentiation of Repair Semantics:** We explicitly separate **A-Box Repairs** (correcting data) from **T-Box Reforms** (correcting the schema), enabling the first rigorous study of "Concept Drift" detection.
