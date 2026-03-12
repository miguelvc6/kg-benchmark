# Evaluation Framework

This document describes what successful evaluation should measure. Script behavior and output schemas are described in [docs-technical](../docs-technical/README.md).

## Evaluation Goal

The benchmark is not only about whether a model can emit a patch. It is about whether a neuro-symbolic workflow can:

- identify the right information source
- produce a repair that survives symbolic validation
- preserve knowledge rather than exploit shortcuts
- remain aligned with human-maintained graph quality

## Mandatory Evaluation Layers

The first publishable unit of the project requires three evaluation-facing capabilities:

- benchmark-ready cases produced from the historical repair pipeline
- evaluation protocols that test validity, safety, and alignment
- a reasoning-floor baseline that measures model performance before protocol support is introduced

The reasoning floor is mandatory because the project is not only evaluating systems with intervention. It must also establish what models can already do without intervention.

## The Reasoning Floor

### Objective

The reasoning floor establishes the pre-Guardian baseline: what current LLMs can do without protocol, memory, tools, rejection sampling, or learning.

Conceptually, this defines the upper bound of unaided reasoning available before the neuro-symbolic transaction layer begins to help.

### Experimental Setup

The reasoning floor uses:

- zero-shot prompting only
- no tool use
- no rejection sampling
- inputs that strictly follow the Phase 2 ablation conditions

### Role in the Research Program

The reasoning floor is the control condition for later intervention studies.

It exists to answer:

- which cases models can already solve unaided
- where failure begins before Guardian support
- whether later improvements come from genuine protocol value rather than from an artificially weak baseline

## Validity Guardrails

### Temporal Validity

Historical repairs cannot be judged naively against a newer world snapshot. Cases whose logical problem no longer exists should be filtered out rather than repurposed. See [Temporal Validity](./Temporal_Validity.md).

### Focus-Node Regression Control

A repair should not be considered successful if it fixes one constraint while creating new violations on the same focus node. The evaluation therefore cares about local side effects, not just target-constraint satisfaction.

### Provenance and Auditability

Accepted repairs should be inspectable. The benchmark values decision traces, provenance chains, and machine-verifiable evidence because these are prerequisites for trustworthy KG updates.

## Core Success Metrics

### Pass@K

Measures whether at least one of the top-k candidate patches satisfies the constraint.

For the reasoning floor, this should be interpreted as the best zero-shot performance obtainable without intervention.

### Conversion Rate

Measures whether verifier or guardian feedback helps the model move from a rejected draft to a valid repair on the next attempt.

### Revert Rate

Uses historical alignment as the main trust proxy: would a human-maintained workflow likely reject or revert the model's action?

## Efficiency and Safety Metrics

### Information Preservation

The benchmark penalizes repairs that satisfy a rule by deleting useful knowledge instead of fixing it.

### Tokens-to-Fix

Measures how much interaction and retrieval cost is required before a verified repair is reached.

### Provenance Completeness

Measures whether accepted repairs are backed by an auditable citation or evidence chain rather than unsupported guesses.

## Interpretation Principle

The benchmark should support conclusions about reasoning, retrieval need, and verification quality. Metrics that blur those questions should be treated as secondary.

This includes a strict comparison rule: post-Guardian or tool-enabled systems should be compared against the reasoning floor, not against an informal baseline.
