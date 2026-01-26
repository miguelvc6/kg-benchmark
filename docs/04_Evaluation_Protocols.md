# Evaluation Protocols & Metrics**

**Artifact Definition:** WikidataRepairEval 1.0
**Context:** Success Criteria, Guardrails, and Validity Checks for the "Guardian" Experiment.

---

## 1. Validity Filters (Quality Control)

Before calculating performance metrics, every model output must pass a series of **Validity Filters**. These protocols ensure that the evaluation measures *reasoning* rather than "gaming" or "luck."

### 1.1 The Persistence Filter (Time-Travel Paradox)

**Risk:** Evaluating a 2019 repair on a 2026 KG snapshot is invalid if the entity's state has mutated or the constraint has changed.

* **Protocol:** The benchmark pipeline re-runs the exact SHACL shape from the dataset on the current live KG.
* **Rule:** If the violation does not exist or has structurally changed (e.g., a different conflicting value is present), the sample is discarded. We prioritize "Persistent Violations" that have remained unsolved or re-occurred.

### 1.2 Sub-graph Regression Testing (The Ripple Effect)

**Risk:** A repair fixes the target constraint but creates a new violation *on the focus node itself* (e.g., changing a date fixes *Range* but breaks *Ordering* with another property on the same node).

* **Protocol:** The verification harness re-validates the **Focus Node** after the patch using the full L1 property set.
* **Rule:** A repair is only "Accepted" if **$Violations(FocusNode)_{after} \le Violations(FocusNode)_{before}$**.

---

## 2. Success Metrics (The "What")

We move beyond simple "Accuracy" to metrics that capture the health of the neuro-symbolic flow.

### 2.1 Pass@K (Functional Success)*

**Definition:** The standard probability that at least one of the top-k generated patches satisfies the SHACL constraint.

* **Application:** Used primarily for the **One-Shot** (Baseline) condition.

### 2.2 Conversion Rate (Feedback Utility)*

**Definition:** The percentage of cases where the Guardian rejected a draft, provided feedback, and the Agent *successfully* repaired it in the next turn.

* **Significance:** Measures the utility of the "Symbolic Immune System." A high conversion rate implies the model is "coachable" via formal logic.

### 2.3 Revert Rate (Human Alignment)*

**Definition:** The ultimate proxy for "Human Trust." Using historical data, would this system's edit have been reverted by a human editor?.

* **Calculation:** If $Action_{Model} \neq Action_{Human}$, we check if the Model's action corresponds to a known "reverted state" in the history logs.

---

## 3. Efficiency & Safety Metrics (The "How")

### 3.1 Information Preservation Score ($S_{info}$)

**Risk:** The "Lazy Repair" Loophole. An agent can satisfy a constraint like Death > Birth by simply deleting the Death Date, maximizing "Constraint F1" while destroying knowledge.

* **Formula:**
* **Significance:** Penalizes purely syntactic fixes that result in information loss.

### 3.2 Tokens-to-Fix (The Price of Truth)

**Definition:** The total count of input/output tokens consumed to reach a verified state.

* **Significance:** Quantifies the efficiency of the "Active Grounding." A model that requires 10 retrieval loops to fix a fact is less viable than one that fixes it in 2, even if both eventually succeed.

### 3.3 Provenance Completeness

**Definition:** The percentage of accepted edits that carry a machine-verifiable citation chain (Reference URL or Node ID).

* **Goal:** Ensures the system is not just "guessing correctly" but "citing correctly."
