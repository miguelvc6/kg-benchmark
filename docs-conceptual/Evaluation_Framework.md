# Evaluation Framework

WikidataRepairEval evaluates knowledge-graph repair as an executable, auditable transaction problem. It should not collapse results into a single leaderboard score. The important output is a stratified failure map across repair locus, information condition, context bundle, model, popularity bucket, and schema-reform cluster.

Implementation details, schemas, and scripts live in [docs-technical](../docs-technical/README.md).

## Evaluation Goal

The benchmark asks whether a model can:

- choose the correct repair locus: A-box, T-box, or ambiguous;
- use the available evidence appropriately;
- avoid leaking post-repair target values from frozen contemporary context;
- produce a parseable and executable proposal;
- preserve useful graph information rather than deleting it to satisfy a rule;
- align with the historically accepted repair target;
- provide rationale, provenance, and uncertainty that make the proposal auditable.

## Reasoning Floor

The reasoning floor is the zero-shot, no-tool, pre-intervention baseline. It measures what current models can do before retrieval, memory, verifier-guided retries, rejection sampling, learning, or a Guardian-style controller is added.

The main reasoning-floor setup uses:

- deterministic zero-shot contract prompting;
- sanitized benchmark inputs only;
- `logic_only` and `local_graph` context bundles as primary conditions;
- `minimal_case` as an optional diagnostic condition;
- oracle-track and diagnosis-routed proposal modes.

Few-shot prompting is an ablation, not the main baseline. It tests precedent adaptation and contract compliance rather than the pure reasoning floor.

## Prediction Tasks

### Track Diagnosis

The model predicts whether the repair locus is `A_BOX`, `T_BOX`, or `AMBIGUOUS`. This measures layer selection independently from repair generation.

### Oracle-Track Repair

The model receives the historical repair locus and proposes a repair using the corresponding contract. This measures repair quality when layer selection is not the bottleneck.

### Diagnosis-Routed Repair

The model first predicts the repair locus and then generates a proposal using that predicted route. The gap between oracle-track and diagnosis-routed performance measures the cost of repair-locus errors.

## Context Ablations

| Bundle | Contents | Purpose |
|---|---|---|
| `minimal_case` | Sanitized case payload only. | No-context diagnostic lower bound. |
| `logic_only` | Sanitized payload plus pruned touched constraints. | Tests rule and constraint reasoning. |
| `local_graph` | Sanitized payload plus pruned local graph and constraints. | Tests local graph grounding. |

All bundles must hide benchmark-only fields such as classification labels, historical repair targets, persistence checks, and current/post-repair target-property truth. The `local_graph` bundle must follow the target-property rule described in [Temporal Validity](./Temporal_Validity.md).

## Metric Families

### Shared Metrics

- proposal presence;
- parse validity;
- normalization success;
- executability;
- acceptance under strict benchmark criteria;
- exact historical alignment where applicable;
- auditability completeness;
- provenance completeness;
- uncertainty availability and calibration proxy;
- tokens, latency, and cost per case or accepted repair.

### A-box Metrics

A-box evaluation reconstructs the pre-repair target-property state, applies the proposed operations in memory, and compares the result to the historical repaired state.

Important metrics:

- target entity/property match;
- operation validity;
- post-application state exact match;
- information preservation;
- over-delete rate;
- local constraint-regression pass;
- wrong target rate;
- functional success.

### T-box Metrics

T-box evaluation compares the proposed post-reform constraint signature with the historical post-reform signature.

Important metrics:

- target property match;
- target constraint-family match;
- proposed action match;
- exact post-signature match;
- semantic-family match;
- signature overlap;
- whether the proposed schema would admit relevant current values;
- case-level and property-revision cluster-macro averages.

Exact T-box signature match should remain strict, but semantic-family metrics are necessary because a proposal may be directionally correct without exactly reproducing the historical signature.

### Track-Diagnosis Metrics

- exact track accuracy;
- macro-F1 over A-box/T-box/ambiguous;
- confusion matrix;
- false A-box rate on T-box cases;
- false T-box rate on A-box cases;
- ambiguous prediction rate;
- oracle-track vs diagnosis-routed repair gap.

## Baselines

Non-LLM baselines are scientifically useful because they show how much can be solved without language-model repair reasoning:

- majority track;
- always A-box;
- always T-box;
- constraint-only delete/update heuristic;
- singleton one-of heuristic;
- range-boundary heuristic;
- local-token lookup oracle;
- do-nothing or invalid proposal baseline.

The main LLM experiments should use local H100-runnable instruction models. A small stratified API subset can serve as a calibration reference, but the paper does not require a full frontier-model leaderboard.

## Interpretation Rules

- Report metrics stratified by repair locus, information condition, subtype, context bundle, prompt regime, model, popularity bucket, and T-box property revision cluster.
- Treat Type C results carefully unless manual audit or retrieval confirms external evidence need.
- Treat historical repairs as historically accepted targets, not universal truth.
- Compare future RAG or Guardian-style systems against the reasoning floor, not against an informal baseline.
