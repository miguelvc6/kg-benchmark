# Confirmatory Evaluation Design

The confirmatory benchmark is quality-gated rather than quota-first. Historical extraction is not treated as correct merely
because a record is present: machine-checkable lineage, schema, classification, evidence, and prompt-leakage checks must pass,
and the final audit disposition must be `include`. Diagnostic, unsupported, disagreement, malformed, and rerender-pending
cases remain useful for aggregate error analysis but never enter main-score evaluation.

## Post-freeze acquisition

Prompts, schemas, classification rules, audit rules, allocation policy, metrics, model identities, and the analysis plan are
frozen before confirmatory candidates are mined. The confirmatory snapshot is then acquired into an isolated directory and
is bound to source timestamps, checksums, cache provenance, configuration, the freeze manifest, and a Git revision. A systemic
implementation defect invalidates that freeze; it is fixed before a new freeze and a new acquisition. Filtering a previously
used population cannot create an untouched confirmatory population.

## Independent sampling unit

The selection unit is an edit event, not an individual row. At most one A-box case is selected for each QID/property group,
and at most one T-box case is selected for each property/revision group. Previously used events are excluded. Property-wide
holdout is not required because the scientific independence claim is event-group isolation, not zero property overlap.

The target population contains exactly 1,200 main-score cases. Its nominal composition is 230 TypeA/IC-L, 375 TypeB/IC-G,
295 TypeC/IC-E-elim, and no more than 300 T-box cases. T-box selection aims for approximately 130 relaxation expansions,
50 restriction contractions, and 120 schema updates. When fewer than 300 independent T-box events survive, the deficit is
transferred across TypeA, TypeB, and TypeC in the fixed 230:375:295 ratio using largest-remainder rounding. Fewer than 1,200
eligible independent cases is a failed release, not permission to weaken a gate.

## Automated validation boundary

Exhaustive deterministic checks replace unavailable human evaluation for release gating. A fixed label-hidden Codex sample
is used only to discover error mechanisms that can then become deterministic rules. It cannot certify semantic truth, causal
necessity, uniqueness, or external-evidence dependence. These remain explicit limitations of the paper.

Every local Ollama model evaluates all 1,200 cases with oracle routing in both `logic_only` and `local_graph`, or 2,400
calls per model. External API reference models evaluate a deterministic, nested 600-case subset in both context bundles,
or 1,200 calls. Diagnosis-routed and few-shot runs are not part of this confirmatory workload; the extensible execution
layer may add them later as separately frozen ablations without repeating the existing generations. Local results are the
headline analysis; API results are paired calibration evidence rather than a substitute population.

Implementation commands and artifact contracts are in the
[Confirmatory Release Runbook](../docs-technical/Confirmatory_Release_Runbook.md).
