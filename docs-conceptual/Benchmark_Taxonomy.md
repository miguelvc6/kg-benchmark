# Benchmark Taxonomy

This document defines the benchmark at the research-design level. Technical artifact layouts live in [docs-technical/Artifact_Schemas.md](../docs-technical/Artifact_Schemas.md).

## Information Necessity Classes

The benchmark groups repair cases by the source of information required to resolve them.

### Type A: Logical

The violation is detectable and repairable from internal consistency or the rule itself. No graph traversal or external retrieval is required.

Typical examples:

- format or enumeration constraints
- boundary cases where the rule uniquely determines the fix
- deletion-as-rejection when the task is to reject an invalid value rather than retrieve a new truth

Research claim: these cases test zero-retrieval reasoning rather than knowledge access.

### Type B: Local

The repair depends on information already available in the focus node or its immediate neighborhood.

Typical examples:

- type compatibility
- inverse or contemporary checks
- cases where the truth is visible in neighboring identifiers, labels, or descriptions

Research claim: these cases test graph-local reasoning and should benefit from structured context more than open-web retrieval.

### Type C: External

The graph-local context is insufficient, so the model must retrieve information not present in the immediate graph view.

Typical examples:

- missing facts
- open-world conflicts
- cases where local graph evidence does not identify the correct replacement

Research claim: these cases isolate true external grounding.

## Repair Tracks

The benchmark separates two kinds of historical fixes that many datasets conflate.

### Cleaner Track: A-box Repairs

The historical action changed instance data on the subject entity.

This track studies whether the system can repair factual graph content while preserving surrounding consistency.

### Reformer Track: T-box Reforms

The historical action changed the constraint or schema itself.

This track studies concept drift: whether the rule became outdated, incomplete, or too strict.

## Benchmark Design Decisions

Several methodological decisions belong to the conceptual layer because they define what the benchmark is trying to measure.

### Real Historical Repairs

The dataset is built from historical human edits to maximize ecological validity.

### Persistence in the Current Graph

The benchmark prefers cases that remain meaningful in the current graph so present-day evaluation does not silently collapse into ontology drift. The reasoning behind this choice is detailed in [Temporal Validity](./Temporal_Validity.md).

### Popularity Stratification

Entity popularity is treated as a controlled variable so common-knowledge entities and long-tail entities can be compared without conflating difficulty with visibility.

### Auditability

Every classification and evaluation decision should be explainable. Decision traces and provenance are therefore part of the benchmark contract, not optional implementation details.

### Pre-Intervention Baseline

The taxonomy is not only for enriched systems. It also defines the inputs for the reasoning floor baseline.

That baseline uses the same benchmark cases and Phase 2 ablation structure, but with zero-shot prompting only and without tools, rejection sampling, or learning. This is necessary so later Guardian results can be interpreted as interventions on a stable task rather than as evaluations on a different task definition.
