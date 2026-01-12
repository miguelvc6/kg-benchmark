# The Time-Travel Paradox (Temporal Inconsistency)

## Why This Matters

The **Time-Travel Paradox** is a fundamental validity issue in benchmarks that replay **historical knowledge graph repairs** but evaluate them against a **current snapshot** of the world (e.g., Wikidata 2026).

If this issue is not handled correctly, the benchmark no longer measures **reasoning ability** or **information access**, but instead measures **ontology drift** and **world evolution**. This silently invalidates conclusions about model performance.

This document explains the paradox, why it arises, and why the **Persistence Filter** used in WikidataRepairEval is a scientifically correct mitigation.

---

## 1. Core Intuition

> A repair that was *correct in the past* can appear *incorrect today* because the world has changed.

This creates a paradox:

- The **human editor** acted rationally and correctly.
- The **current KG state** no longer matches the assumptions under which the repair was made.
- Evaluating the repair today falsely penalizes correct reasoning.

This is not an edge case. Wikidata evolves continuously.

---

## 2. Formal Statement of the Paradox

Let:

- $S_t$: the constraint (SHACL shape) at time $t$
- $E_t$: the entity state at time $t$
- $E_{now}$: the entity state in the current snapshot (e.g. 2026)
- $V(S, E) \in \{\text{valid}, \text{invalid}\}$

A historical repair is triggered because:

$$
V(S_t, E_t) = \text{invalid}
$$

A naïve evaluation checks:

$$
V(S_{now}, E_{now})
$$

This is **invalid**, because it compares:

- a past action
- against a different law
- in a different world

Correct evaluation must instead ask:

$$
V(S_t, E_{now})
$$

> “Does the *same logical violation* still exist if the *old law* is applied to the *current world*?”

If not, the case is **temporally inconsistent** and cannot be evaluated.

---

## 3. Concrete Example

### Historical Context (2019)

- Entity: `Q123`
- Property: `P39` (position held)
- Constraint: value must be instance of *political office*

At the time:

```

Minister of Digital Affairs → instance of political office

```

A human editor adds:

```

Q123 P39 Minister of Digital Affairs

```

The violation disappears. This is a **correct repair**.

---

### World Evolution (2022–2024)

Later, Wikidata evolves:

```

Minister of Digital Affairs → instance of government role

```

The constraint is also tightened.

---

### Naïve Evaluation (Incorrect)

Replaying the 2019 repair in 2026 yields:

```

Constraint violated again

```

This leads to the false conclusion:

> “The repair was wrong.”

In reality, the **world changed**. The repair did not.

This is the Time-Travel Paradox.

---

## 4. What “Re-Running the SHACL Shape” Means

In WikidataRepairEval, each repair stores the **exact constraint logic** that caused the violation, serialized deterministically.

Re-running means executing:

```

validate(
shape = constraint_at_time_of_violation,
entity = entity_state_2026
)

```

This is a **counterfactual check**:

> “If the old rule were still applied today, would the same violation exist?”

---

## 5. When a Violation “No Longer Exists”

A violation is considered gone if:

```

validate(shape_t, entity_2026) == valid

```

This can happen because:

### Case A — Independent World Fix

Another editor fixed the issue later.

### Case B — Entity Deletion or Merge

The entity no longer exists.

### Case C — Legitimate DELETE Repair

If the historical repair was a DELETE, absence is expected and allowed.

Only DELETE cases bypass existence checks.

---

## 6. What “Structurally Changed” Means

A violation may still exist but be **logically different**.

### Example 1 — Different Conflicting Value

- Same constraint
- Different offending value
- Different repair action required

### Example 2 — Constraint Semantics Drift

- Additional conditions added
- Rule logic no longer matches

### Example 3 — Cardinality Relaxation

- Schema evolution resolves violation without data change

In all cases, the **problem definition has changed**.
These cases are discarded.

---

## 7. Why Discarding Is the Only Correct Action

### Why Not Update the Ground Truth?

- Breaks reproducibility
- Changes task definition retroactively
- Destroys causal attribution

### Why Not “Map” Old Violations to New Ones?

- Requires human interpretation
- Introduces subjectivity
- Not automatable

### Why Discarding Preserves Science

Discarding enforces:

> Only evaluate cases where the logical problem is invariant across time.

This guarantees:

- Comparability across models
- Valid attribution of failure
- Clean separation of reasoning vs. world drift

---

## 8. Why This Is Central

Without resolving temporal inconsistency:

- Guardian effectiveness becomes uninterpretable
- RAG ablations are noisy
- Self-training appears to “collapse” spuriously
- Negative results are meaningless

With the Persistence Filter:

- Observed failures are attributable to **reasoning or information gaps**
- Not to ontology drift or temporal leakage

This transforms the benchmark from:

> an engineering artifact into a **scientific instrument**

---

## 9. One-Sentence Definition

**Temporal Inconsistency (Time-Travel Paradox):**
The invalid evaluation of a historically correct repair under a future world state in which the underlying facts or constraints have changed, conflating reasoning error with ontology drift.

---

## 10. Mental Model

> Freeze the **law**, then ask whether the **same crime still exists** in today’s world.
> If not, the case is inadmissible as experimental evidence.
