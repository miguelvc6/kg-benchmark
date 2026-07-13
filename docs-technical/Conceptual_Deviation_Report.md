# Implemented Boundary And Remaining Gaps

## Purpose

This document records the current engineering boundary between the implemented repository and the broader research
program. It is not the operational release checklist; use [Research Release Protocol](./Research_Release_Protocol.md)
for release and confirmatory-run gates.

## Current Implemented Scope

The repository implements:

- historical Wikidata candidate mining, repair reconstruction, popularity enrichment, and frozen world-state building;
- Stage 4 classification and schema validation;
- group-aware train/dev/test splitting and deterministic dev/core selection manifests;
- A-box, strict-signature T-box, taxonomy-patch T-box, and track-diagnosis normalization;
- deterministic non-LLM baselines and zero-shot/few-shot model evaluation;
- oracle proposal routing and optional diagnosis-routed generation;
- blinded multi-reviewer annotation assignment, agreement reporting, and adjudication support;
- paired cluster-bootstrap analysis with exact paired-binary tests;
- content-addressed release manifests, frozen protocol manifests, and an experiment registry;
- protocol-bound untouched-test allocation from a post-freeze benchmark snapshot.

Implementation details live in:

- [Pipeline Implementation](./Pipeline_Implementation.md)
- [Proposal Validation](./Proposal_Validation.md)
- [Evaluation Harness](./Evaluation_Harness.md)
- [Reasoning Floor](./Reasoning_Floor.md)
- [Track Diagnosis](./Track_Diagnosis.md)
- [T-Box Taxonomy Patch Task](./TBox_Taxonomy_Patch_Task.md)
- [Few-Shot Evaluation](./Few_Shot_Evaluation.md)

## Current Release Judgment

The repository supplies the engineering substrate needed to create a paper-eligible release, but existing core/model
artifacts remain exploratory. The controls added after those runs do not retroactively make development-contaminated
results confirmatory.

The following work still requires deliberate execution and review:

- exhaustive automated consistency auditing plus label-hidden Codex-assisted error discovery under the documented
  no-human scope;
- explicit reporting that extracted T-box taxonomy gold is not independently human-validated;
- creation of a new post-freeze benchmark snapshot and sealed untouched-test manifest;
- a clean, content-addressed release manifest and frozen protocol;
- confirmatory model execution with stable model digests;
- registered statistical analyses and paper tables;
- dataset card, license, citation metadata, ethics statement, limitations, and archival deposit.

## Research Features Not Implemented

These remain outside the current paper-release substrate:

### Live Verifier Loop

`src/evaluate.py` scores completed outputs. The repository does not provide a repair-time loop that returns verifier
diagnostics to a generator and allows bounded proposal revision.

### Retrieval And Tool Use

The model runners do not perform external evidence retrieval, live Wikidata editing, or repair-time tool use. Provenance
fields describe evidence supplied in the frozen task context; they are not evidence that a retrieval system ran.

### Full Diagnosis-Routed Main Condition

The runtime can route proposals from track-diagnosis predictions, including explicit `AMBIGUOUS` skips. The current
diagnosis prompt did not pass the development gate, so oracle routing remains the supported main condition and
diagnosis-routed execution remains exploratory.

### Cost-Quality Protocol Frontier

Runs capture tokens, latency, request failures, and configured cost estimates. They do not yet compare bounded
multi-attempt, retrieval, tool-call, and verifier-call protocols because those protocol runners do not exist.

## Engineering Rules

- Keep benchmark construction independent from protocol experiments.
- Keep exploratory and confirmatory runs distinct in `experiments/registry.json`.
- Preserve the reasoning floor as a stable pre-intervention control.
- Treat offline evaluation as the benchmark authority.
- Bind releases and runs to hashes, code commits, prompt/schema versions, and model digests.
- Do not infer untouchedness from code-level exclusions on the development snapshot; use a post-freeze snapshot.
- Update this document when an implementation boundary changes, and put operational commands in the release protocol.
