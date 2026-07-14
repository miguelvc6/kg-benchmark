# Confirmatory Analysis Plan

The local 1,200-case population is the headline population. The nested 600-case Azure population is a paired
calibration reference. Larger population manifests are registered extensions and never replace or overwrite the frozen
1,200/600 results.

## Conditions

Every selected case is evaluated for oracle-routed repair proposal and track diagnosis under the full factorial of
zero-shot/static-few-shot and `logic_only`/`local_graph`. Diagnosis is scored separately and never routes the proposal
task. The initial workload is 9,600 calls per Ollama model and 4,800 Azure calls.

Few-shot prompts use deterministic prefixes of the published support bank. The defaults are four A-box examples, four
T-box examples, and two diagnosis examples. Changing the demonstration count defines a new prompt condition. Expanding
only the evaluated population reuses existing content-addressed generations.

## Primary analyses

For A-box repair, report paired context and prompt-regime effects on accepted repair rate using case-micro and
event-cluster macro estimates, 95% percentile intervals from 5,000 seed-13 cluster bootstrap samples, and exact McNemar
tests for paired binary comparisons. Report IC-L, IC-G, and IC-E-elim separately. IC-E-elim is an extractor-relative
no-retrieval stress condition, not confirmed external-evidence necessity.

For T-box repair, report taxonomy-patch operation, taxonomy-code, evidence-level, qualifier, and value-delta metrics.
A-box and T-box scores are never collapsed into one repair score. For diagnosis, report accuracy, macro-F1, confusion
matrix, false-locus rates, and ambiguous prediction rate.

## Replay and extensions

Incomplete confirmatory pairs fail the analysis. Transport retries replay the exact request and semantic retries are
disabled. Raw generations are immutable; metric versions can be recomputed without provider calls. A population
extension must reference a parent manifest and prove per-stratum prefix nesting before it may reuse the parent results.
