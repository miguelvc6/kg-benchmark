# Development History

This document preserves the decisions and evidence produced while developing WikidataRepairEval. It is historical
context for writing the paper, not an executable protocol. The current protocol lives under `paper/`; the complete
pre-restructure tree is preserved by Git tag `archive/pre-paper-restructure-20260714` at commit
`bf9129bb05053925e03c4d780fe4fd3687d12698`.

## Research framing and initial implementation

The project began as a benchmark of whether language models can reconstruct historically accepted Wikidata repairs
from controlled information conditions. The pipeline mined constraint-report pages, reconstructed nearby entity and
property revisions, checked persistence, extracted a frozen world state, classified information requirements, selected
controlled populations, generated proposals, and evaluated the proposals mechanically.

The core distinction between A-box claim repair and T-box schema reform was retained throughout development. The
TypeA/TypeB/TypeC implementation vocabulary evolved into the paper-facing IC-L, IC-G, IC-E-elim, and IC-U conditions.
Type C was narrowed from an informal `EXTERNAL` label to `EXTERNAL_BY_ELIMINATION`: supported rule and local extractors
did not identify the historical target. This does not establish that external evidence is necessary.

## Phase A: scope and paper narrative

Phase A aligned the benchmark with the first-paper questions, separated repair locus from information condition, and
defined controlled context bundles. It introduced a reasoning-floor framing: models receive frozen visible evidence
without retrieval, hidden labels, post-repair truth, or verifier-guided retries. It also established stratified reporting,
temporal validity, popularity analysis, and the distinction between confirmatory and exploratory work.

## Phase B: classifier redesign

Phase B audited and hardened the Stage 4 classifier. It split ordinary Type C fallback from explicit `UNKNOWN_*`
diagnostics; quarantined current-value-only truth; expanded supported local evidence; added boundary-aware literal and
identifier matching; and revised format, cardinality, deletion, set-membership, and target-required-claim rules.
Classifier transition matrices were repeatedly generated to understand the effect of each change. Those matrices are
archived history and are not part of the final paper workflow.

## Phase C: dataset tiers and independent selection

Phase C added deterministic selection manifests, development/core separation, popularity and constraint-family strata,
event-level group keys, caps, and non-overlap checks. Early manifests used fixed `core_v1` and development quotas. The
confirmatory design later replaced these with one fully audited population, a 1,200-case main view, and a nested
600-case API view. The final design stores a stable eligibility order so larger nested populations can be materialized
without changing prior selections.

## Phase D: manual and automated audit development

An early manual-audit workflow generated stratified case cards and annotation artifacts for Type C, ambiguous A-box,
and T-box cases. It exposed risks such as missed local evidence, incomplete truth, selection ambiguity, multiplicity
artifacts, bad target/context combinations, and coincidental schema changes. Independent human certification was later
removed from the paper scope.

The replacement automated audit combined deterministic whole-data checks with fixed, label-hidden Codex samples for
error discovery. `full_v1` recorded the first complete audit and remains historical evidence rather than a release
certificate. `full_v2c` validated the restored baseline after lineage recovery. `full_v3` incorporated deterministic
rules derived from recurring findings, including target leakage and temporal-token patterns. The final policy requires
one unique disposition for every case and makes only `include` eligible.

## Phase E: deterministic baselines

Development included majority/constant track baselines, invalid and do-nothing proposals, constraint-oriented Type A
heuristics, and local lookup oracles. These established mechanical lower bounds and exercised the same evaluator used
for model outputs. The baseline implementations remain useful, but historical raw traces and duplicated reports are
archived rather than retained in the active repository.

## Prompt-development rounds

Prompt work compared representation formats, zero-shot contracts, abstention behavior, context sanitization, diagnosis
prompts, and proposal schemas. Prompt versions progressed through increasingly strict JSON contracts. The maintained
A-box prompt requires target, operations, rationale, provenance, and uncertainty. Model-visible input omits hidden class,
subtype, truth, persistence, popularity, and selection fields.

Few-shot development introduced fixed support manifests, neutral example IDs, task-specific safe outputs, example/core
group isolation, hidden-field leakage scans, and zero/few-shot paired reports. Static diverse support was retained;
dynamic similarity and hidden-metadata matching were treated as exploratory. The final design publishes a reserved
support bank, excludes every support group from evaluation, and makes example count a prompt configuration parameter.

## Track diagnosis

Track diagnosis was implemented as a separate `A_BOX`, `T_BOX`, or `AMBIGUOUS` task. Development evaluated both direct
diagnosis and diagnosis-routed proposals. Routing compounded diagnosis and repair errors and did not become the
confirmatory proposal mode. The final design scores diagnosis as a headline task while retaining oracle routing for
repair proposals.

## T-box task migration

The first T-box task requested an exact post-repair constraint signature. Historical mining often lacked a complete
pre-repair signature, making exact reconstruction unsupported. Development therefore introduced the taxonomy-patch
task: causal schema decision, constraint family, bounded repair operation, taxonomy code, qualifier/value deltas, and
evidence level. Class-hierarchy and exception operations were excluded because the acquisition pipeline did not mine
their required evidence. The strict-signature implementation and reports are archived as exploratory history.

## Temporal validity and leakage remediation

The pipeline reconstructed prompts from the pre-repair target-property state and removed later target values from local
context. Auditing was extended from exact target tokens to embedded identifiers, URLs, paths, queries, encoded values,
normalized literals, descriptions, and aliases exposed under other identifiers. T-box cases without a pre-repair
signature omit later constraint inventories and explicitly expose missing historical context. Recurring patterns found by
model-assisted review were converted into deterministic scans.

## Restored baseline and lineage

The original large Stage 0–4 files were restored under `data/` and intentionally not rewritten. Lineage tooling compared
Stage 2 JSON and JSONL by count, unique IDs, order, canonical digests, and documented representation differences. It
validated Stage 0 popularity payloads, Stage 1 candidate provenance, exact Stage 2/3/4 identity, and the Stage 2
projection of Stage 4. Versioned source-provenance, reconciliation, and lineage manifests recorded the recovered state.

The restored baseline was useful for finding implementation defects and validating rules, but it is not the final
confirmatory dataset. Its canonical artifacts and scientifically relevant reports are preserved in the external
development archive; caches and derived indexes are not.

## Execution and replay design

The runner gained Ollama and external-provider adapters, exact-request transport retry, immutable raw generations,
content-addressed caching, batch support, model/version provenance, resumption, and evaluator-only replay. The selected
paper models became `qwen3:30b`, `llama3.3:70b`, `gpt-oss:120b`, and Azure `gpt-5.6-sol` with high reasoning effort,
batch execution, and tools disabled.

The final matrix uses repair and diagnosis under zero/few-shot and logic/local conditions. Cache identity is based on
case payload, rendered prompt, context, task, model revision, and inference parameters—not population membership—so
larger populations schedule only new requests and metric revisions require no provider queries.

The paper-facing orchestration subsequently materialized that design instead of relying on handwritten runner command
combinations. It introduced deterministic matrix and exact-request plans, provider-free cache/revision preflight,
stable physical execution groups, per-logical-cell manifests, independent completeness replay, and explicit extension
planning. Tests use fake executors to prove the frozen Ollama/Azure argument policies and that a nested population adds
only generation keys absent from the parent cache.

## Release and freeze work

Research-release tooling added schemas, manifests, artifact hashing, clean-clone acquisition, experiment registration,
protocol freezing, and paper-eligibility checks. Multiple methodology manifests accumulated during this process. The
restructured repository replaces them with one current protocol and relies on Git tags for history.

The final promotion boundary was then made independent of phase-local success flags. A version-2 release gate copies
the frozen protocol, lock, active schemas, canonical source/case data, audit outputs, and selection evidence into a
temporary dataset; revalidates every record; replays lineage and deterministic selection; checks original acquisition
and cache provenance; requires complete dispositions with no unresolved systemic findings; enforces the exact nested
1,200/600 populations; and reproduces the manifest bytes before the atomic rename. Synthetic mutations exercise each
fail-closed boundary without generating the paper dataset or making model calls.

The first post-freeze acquisition, `post_freeze_20260714T062500Z_wd20260706`, was invalidated after discovering that
exhausted history, pageview, and dump errors could be interpreted as missing or partial data and that the methodology
freeze omitted transitive modules. Commit `cba9349` made acquisition fail closed, and `bf9129b` created a broader freeze.

The replacement acquisition, `post_freeze_20260714T064200Z_wd20260706`, downloaded and checksum-verified the 2026-07-06
Wikidata dump and began a fresh report crawl. It was stopped during Stage 1 when inspection found that an interrupted
Stage 2 JSONL could be compiled and treated as complete on restart. Resume telemetry also remained outside the isolated
data directory. This attempt is invalid and contributes no paper cases.

## Repository simplification

By July 2026 the working tree mixed 96 GB of baseline data, 41 GB of reports, 2.6 GB of logs, hundreds of prompt and
audit artifacts, 25 CLI entry points, and overlapping schema/protocol versions. The paper-focused restructure preserves
the scientific record in this document, Git history, and an external checksum-indexed archive while exposing one CLI,
one protocol, one canonical dataset release, compact results, and the retained research capabilities.

## Deferred or explicitly excluded features

External retrieval is not supplied to IC-E-elim in the current paper. Diagnosis-routed proposal generation, dynamic
few-shot retrieval, hidden-metadata example matching, semantic response retries, verifier-guided retries, and
class-hierarchy/exception T-box reconstruction remain possible extensions. They must be registered as new conditions
and cannot silently alter the frozen confirmatory results.
