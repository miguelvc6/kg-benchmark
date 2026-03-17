# Public Documentation

This directory is the public-facing entry point for researchers who want to use
the benchmark without first reading the full internal repository docs.

It is intentionally shorter and more policy-oriented than the material in
`docs-conceptual/` and `docs-technical/`.

Start here:

- [Dataset Card](./Dataset_Card.md): what the dataset is, what it contains, and what it is for
- [Benchmark Invariants](./Benchmark_Invariants.md): the rules that must stay true when loading, prompting, and evaluating cases
- [Correct Usage and Pitfalls](./Correct_Usage_and_Pitfalls.md): the shortest path to using the benchmark correctly and the most common failure modes
- [Release Structure](./Release_Structure.md): how to think about the benchmark release versus protocol/runtime releases

Canonical deeper references:

- [Conceptual docs](../docs-conceptual/README.md): research framing, benchmark intent, and evaluation philosophy
- [Technical docs](../docs-technical/README.md): implementation details, artifact schemas, pipeline behavior, and runner semantics

Use this public documentation as the first stop for external use and citation.
Use the conceptual and technical docs when reproducing pipeline behavior or
auditing implementation details.
