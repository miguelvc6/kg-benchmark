# Release Structure

External users should think about this repository as two related releases.

## 1. Benchmark Release

This is the release you use when your goal is to benchmark repair behavior on a
stable task definition.

It includes:

- historical repair records
- frozen world-state context
- classified benchmark labels
- split/selection metadata
- schemas and public documentation

This layer defines:

- what the task is
- what the canonical target is
- what counts as a valid evaluation setup

## 2. Protocol and Runtime Release

This is the release you use when your goal is to reproduce or extend the
repository's experimental runners.

It includes:

- reasoning-floor prompt construction
- proposal and diagnosis normalization
- evaluation harness implementation
- run manifests and protocol artifacts
- debugging and viewer tooling

This layer defines:

- how a specific baseline or protocol was run
- what prompt context a model saw
- how a released baseline result was produced

## Practical Rule

- If you are comparing models on the benchmark, start from the benchmark release.
- If you are reproducing published baseline numbers from this repository, use the protocol/runtime release in addition to the benchmark release.
- If you modify prompt construction or protocol behavior, describe the result as a derived protocol setting rather than the untouched benchmark baseline.

## Why This Separation Matters

It keeps external users from confusing:

- benchmark validity
- protocol behavior
- implementation shortcuts

That separation makes it easier to publish the dataset openly without forcing
every researcher to understand every internal implementation detail before they
can use it responsibly.
