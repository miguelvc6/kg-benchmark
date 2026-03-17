# Dataset Card

## Name

WikidataRepairEval / `kg-benchmark`

## What It Is

This benchmark reconstructs real historical Wikidata repair events and pairs
them with a frozen present-day graph snapshot for controlled evaluation of
knowledge-graph repair systems.

The benchmark is designed to study:

- whether a system can identify the right information source for a repair
- whether it can produce a historically aligned repair
- whether it can do so without unsafe shortcuts or silent semantic drift

## Core Units

Each case is centered on:

- a focus entity or property
- a violating property/constraint report context
- a historical repair target
- a frozen world-state context
- a benchmark label describing information necessity and repair track

## Main Artifact Layers

- Stage 2: historical repair record with target repair metadata
- Stage 3: frozen world-state context
- Stage 4: classified benchmark record
- Stage 5: deterministic train/dev/test split metadata

## Intended Uses

- benchmarking A-box and T-box repair systems
- studying information-necessity classes such as logical, local, and external cases
- evaluating zero-shot or protocol-guided KG repair workflows
- auditing model behavior against historical human-maintained repairs

## Recommended Evaluation Practice

- use the official evaluator when possible
- preserve the distinction between historical target state and current contextual snapshot
- use the benchmark record, not raw world state alone, to reconstruct the target property's historical before-state

## Important Limitation

The benchmark combines historical repair targets with a frozen later world-state
snapshot. That is a deliberate design choice, but it means users must follow the
benchmark's temporal policy carefully. The main failure mode is target-property
leakage from the frozen world-state snapshot into prompts or derived training
examples.

## What To Cite or Describe in Papers

At minimum, external users should report:

- which benchmark artifact layer they used
- whether they used the official evaluator
- whether prompts were built with official repository logic or custom logic
- whether the experiment used benchmark-only data or also used protocol/runtime artifacts

## Where To Read Next

- [Benchmark Invariants](./Benchmark_Invariants.md)
- [Correct Usage and Pitfalls](./Correct_Usage_and_Pitfalls.md)
- [Release Structure](./Release_Structure.md)
