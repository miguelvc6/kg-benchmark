# Confirmatory Analysis Plan

The local 1,200-case population is the headline population. The nested 600-case Azure population is a paired
calibration reference. Larger population manifests are registered extensions and never replace or overwrite the frozen
1,200/600 results. Models are analyzed separately rather than pooled.

The machine-readable source for these decisions is `paper/analysis.json`; this document explains their paper-facing
meaning.

## Conditions

Every selected case is evaluated for oracle-routed repair proposal and track diagnosis under the full factorial of
zero-shot/static-few-shot and `logic_only`/`local_graph`. Diagnosis is scored separately and never routes the proposal
task. The initial workload is 9,600 calls per Ollama model and 4,800 Azure calls.

Few-shot prompts use deterministic prefixes of the published support bank. The defaults are four A-box examples, four
T-box examples, and two diagnosis examples. Changing the demonstration count defines a new prompt condition. Expanding
only the evaluated population reuses existing content-addressed generations.

## Predeclared contrasts and inference

Four paired contrasts are primary within every task, model, and reporting stratum:

1. `local_graph` versus `logic_only` within zero-shot;
2. `local_graph` versus `logic_only` within static few-shot;
3. static few-shot versus zero-shot within `logic_only`;
4. static few-shot versus zero-shot within `local_graph`.

Report case-micro and independent event-cluster macro estimates with 95% percentile intervals from 5,000 seed-13
cluster-bootstrap samples. Binary paired comparisons use exact McNemar tests. Apply Holm correction across the four
primary contrasts within each task × model × stratum family. Do not pool models into a single confirmatory estimate.

## Task metrics

For A-box repair, the primary endpoint is accepted repair. Report paired effects and component metrics separately for
IC-L, IC-G, and IC-E-elim. IC-E-elim is an extractor-relative no-retrieval stress condition, not confirmed
external-evidence necessity.

For T-box repair, the primary endpoints are schema-decision match and taxonomy-code exact match. Repair-operation,
evidence-level, qualifier, and value-delta metrics are secondary. A-box and T-box scores are never collapsed into one
repair score.

For diagnosis, the primary endpoint is accuracy. Macro-F1, confusion matrix, false-locus rates, and ambiguous prediction
rate are secondary.

## Failures, replay, and extensions

Parse or response-schema failures count as incorrect. Transport failures are incomplete rather than scored; incomplete
confirmatory pairs or condition matrices block analysis and release. Transport retries replay only the exact request,
Azure is executed sequentially with at most two exact-request transport retries, and semantic retries are disabled.

Raw generations are immutable and metric versions can be recomputed without provider calls. Azure results are a paired
calibration reference and receive no cross-model significance tests. A population extension must reference a parent
manifest and prove per-stratum prefix nesting; extension results are reported separately and never replace confirmatory
results.
