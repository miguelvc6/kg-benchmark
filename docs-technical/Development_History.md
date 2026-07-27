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
and tools disabled. The freeze candidate initially required Azure batch execution; when the selected deployment proved
not to offer batch, the paper condition was changed before freeze to sequential synchronous execution with two exact-
request transport retries. The Azure data-plane inventory exposed `gpt-5.6-sol-2026-07-09`, which became the immutable
execution and cache revision. General batch support remains available for registered future experiments.

The final matrix uses repair and diagnosis under zero/few-shot and logic/local conditions. Cache identity is based on
case payload, rendered prompt, context, task, model revision, and inference parameters—not population membership—so
larger populations schedule only new requests and metric revisions require no provider queries.

The paper-facing orchestration subsequently materialized that design instead of relying on handwritten runner command
combinations. It introduced deterministic matrix and exact-request plans, provider-free cache/revision preflight,
stable physical execution groups, per-logical-cell manifests, independent completeness replay, and explicit extension
planning. Tests use fake executors to prove the frozen Ollama/Azure argument policies and that a nested population adds
only generation keys absent from the parent cache.

The final analysis layer closed the replay-to-paper boundary. Replay gained an all-selected-cases diagnosis artifact so
T-box diagnosis was no longer lost behind the separate taxonomy evaluator, and taxonomy parse failures were included as
incorrect rather than silently removed from endpoint denominators. A provider-free matrix replay command, predeclared
cluster inference, exact McNemar/Holm contrasts, extension-prefix enforcement, compact role-specific tables, and
content-addressed result manifests replaced manual aggregation notebooks.

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

The final acceptance pass joined those phase-level checks into one offline synthetic lifecycle. It builds a frozen
1,620-case source population through the public interface, validates lineage and the exhaustive audit, rejects a
would-be selected prompt and replaces it from the declared reserve order, promotes exact nested 1,200/600 populations,
materializes and completes the experiment matrix through a fake executor, replays revised metrics without generation,
and reproduces the paper-analysis manifests byte for byte. A separate tracked-files-only smoke test builds and installs
the wheel in a clean temporary clone, then checks the installed CLI and frozen prompts without relying on ignored local
state. Both tests are provider-free and do not certify the real paper dataset.

That pass exposed three engineering defects before execution. Canonical Stage 3 JSONL began with an object token and
was incorrectly routed through the streaming JSON-object reader; suffix-first dispatch now preserves JSONL record
boundaries. Stage 4 cases retained the construction machine's absolute world-state path; build canonicalization now
publishes `source/world-state.jsonl`. Finally, frozen prompt files and schemas were available only from a source tree;
they are now installed as package data and resolved through one repository-or-installation resource lookup. Regression
tests preserve all three fixes.

The final paper methodology was frozen after the complete offline acceptance suite passed with 363 tests and 124
subtests. Source commit `adbfa0fc037bb536554b291f886f0ccd923dec3e` resolves all four model revisions and binds the
sequential Azure policy. Lock-only commit `e3d9059` adds `paper/methodology.lock.json`, whose freeze-scope digest is
`aa8d9c8d3e0ac2219921cd91450e984eb626db698d1c7f2861790742e3216323`. The annotated Git tag
`paper-methodology-v1` identifies the completed freeze state from which the single final dataset must be acquired.

The first post-freeze acquisition, `post_freeze_20260714T062500Z_wd20260706`, was invalidated after discovering that
exhausted history, pageview, and dump errors could be interpreted as missing or partial data and that the methodology
freeze omitted transitive modules. Commit `cba9349` made acquisition fail closed, and `bf9129b` created a broader freeze.

The replacement acquisition, `post_freeze_20260714T064200Z_wd20260706`, downloaded and checksum-verified the 2026-07-06
Wikidata dump and began a fresh report crawl. It was stopped during Stage 1 when inspection found that an interrupted
Stage 2 JSONL could be compiled and treated as complete on restart. Resume telemetry also remained outside the isolated
data directory. This attempt is invalid and contributes no paper cases.

The next clean acquisition under `/run/kg-benchmark-paper-v1` stopped during Stage 1 at property `P11029` after the
Wikidata Action API returned HTTP 429. Candidate mining had short fixed retries, did not honor `Retry-After`, did not
pace the metadata request hidden in each `mwclient` page construction, and retained all Stage 1 progress only in memory.
Source commit `20664a3dc12abf7aa91f79b207271477da3182eb` adds process-wide request pacing, rate-aware retry with
exponential fallback, and an atomic append-only Stage 1 checkpoint after every completed property. Lock-only commit
`c06a2ca9a9eb34ffc8044c79871e3ef90d4d0566` binds the replacement freeze to scope digest
`3b4fca75d3eac67cb5567db83f9def185e1c5152d5c0ab58840c3c3a0a50c04b`; annotated tag `paper-methodology-v2`
identifies the restartable paper acquisition state. The earlier `paper-methodology-v1` tag remains unchanged as the
historical pre-repair freeze.

The first v2 Stage 2 pass then exposed two additional acquisition defects. A deleted focus entity returned a terminal
REST history 404, but the generic JSON helper retried it and raised `TransientAPIError`, aborting the run. More
importantly, Stage 1 treated report-bot revisions with comments such as `error while update`—including full report pages
collapsing to roughly one hundred bytes—as if every disappeared QID represented a repaired violation. This inflated the
candidate list with millions of false report-disappearance events. The corrective implementation classifies an initial
history 404 as a terminal missing entity, excludes explicit report-update errors, and conservatively rejects unmarked
large-to-tiny report collapses. All other upstream failures remain fail-closed.

The v3 crawl removed roughly one million explicit error-page rows, but its remaining 2,813,427 deduplicated candidates
were still dominated by correlated bulk events: 83.5% came from 664 report transitions containing at least 1,000 QIDs.
After 51,299 Stage 2 candidates, the observed throughput projected a five-to-six-week acquisition even though final
selection requires only 1,200 cases and treats T-box property/revision groups as the independent sampling unit. The v3
attempt was stopped and archived without contributing paper cases. The replacement policy preserves every report event
and every candidate from events of at most 100 QIDs, while larger events retain a seed-13 SHA-256-ranked sample of 100.
This reduces the observed v3 candidate universe to approximately 267,000 candidates without globally truncating the
source stream or discarding bulk T-box events. Source commit
`a755959a62ae2d4adf97447a729b33e4ba7831c6` implements and documents the event cap; lock-only commit
`d09ddfd1e04c5bddcd04fefac0976052456d0dcd` binds freeze-scope digest
`c56a69be441c3eb0d56a6cc7d7b57e4b02331a1ceb1cfe9a80966123ebcb93f9`; annotated tag
`paper-methodology-v4` identifies the event-balanced acquisition state.

The v4 Stage 2 run then stopped after 1,130 completed candidates when historical snapshot `P888@2483634740` exhausted
four 30-second read attempts. The revision remained valid and returned immediately on a later probe, so the failure was
transient; however, `SnapshotFetchError` escaped the T-box scan and terminated the entire acquisition. The replacement
policy treats exhausted transient Stage 2 access as explicit reconstruction nonresponse. It records the candidate,
phase, and error in a durable exclusion artifact, never maps the failure to missing/no-history/no-diff evidence, flushes
and checkpoints the exclusion immediately, and writes ordinary checkpoints every 100 processed candidates. Source
provenance binds the completed exclusion array and its count.

The 1,130 v4 stats rows form a contiguous unique-key prefix with zero transient exclusions, and the failed P888
candidate has no completed stats row or repair output. The v5 resume therefore reuses only completed outcomes and binds
the prior acquisition configuration, candidate artifact, partial repairs, and supplied resume stats in the new
acquisition configuration. Source commit `95f64af4f69c70303b715ba8a4ae62f438a978c8` implements the exclusion and
compatible-prefix policy. Lock-only commit `61950263de185ab39dd2b7cd80139c69fd281622` binds freeze-scope digest
`030930ba71ec7ecd407dc8ed9009fb37a2ba740130a2d013cef858414690024f`; the lock SHA-256 is
`6c26ad28e27c12ea5614cbf190a348aa6c7a13a806b8eff7f02f812beabc32e3`. The complete acceptance suite passed with
376 tests and 125 subtests. Annotated tag `paper-methodology-v5` identifies the transient-safe acquisition state.

The completed v5 acquisition subsequently rendered 138,312 audit prompts for 34,578 cases, but the temporal gate
treated any target-token coincidence as future leakage. That incorrectly blocked retained values and aliases during
mixed updates, shared historical aliases, target-required focus identity, format-normalized substrings embedded in
historical literals, and Type B values backed by model-visible independent local evidence. The failed v5 audit remains
an archived construction artifact and is never promoted.

Methodology v6 replaces that gate with occurrence-level source/span classification. It distinguishes unexplained
future or hidden values from expected historical, rule-derived, local-evidence, and diagnostic matches; only the first
category blocks. It retains positive detection for future-only identifiers and aliases, encoded values, revision and
author metadata, after-signatures, and uncovered normalized occurrences. The temporal report contract is version 3 and
records raw hits plus unique-case counts for every severity. Deterministic and temporal phases now emit starts,
completions, and 60-second heartbeats. `PopularityCalculator` also resolves the runtime dump path when instantiated, so
the checkpoint replay honors `--dump-path` directly.

Source commit `93f335d63b04cfb24ba14ac090b4d56f36aeaa2f` implements the repair and removes the v5 lock. Lock-only commit
`f6640eb5dd3578125d955a3d32ef8e6b07fde67a` binds freeze-scope digest
`9b3b23ca5184ecf8f149803ec9b180710ccb8253e2b4536f3e79e6691575d7b6`; the lock SHA-256 is
`269e14b0855f40cdd7f5c3945d78e9d44842ceef825c93785859a851ad29324d`. The complete acceptance suite passed with
380 tests and 125 subtests. Annotated tag `paper-methodology-v6` identifies the source-aware temporal-audit state. The
publication replay resumes the completed Stage 2 checkpoint and does not repeat candidate discovery.

The v6 dry-run then found two narrower source-classification defects: JSON-escaped historical descriptions were not
recognized as covering spans, and labels aligned to trace-backed Type B values were not connected to their visible
local scalar. After those corrections, 75 of 34,578 cases still contained genuinely uncovered future-value aliases or
distinctive hidden-author strings. Because evaluation uses only a small deterministic subset, methodology v7 retains
all Stage 0–4 rows for exact lineage but permanently assigns those cases disposition `exclude`; only `include` remains
selection eligible. The temporal report contract is version 4 and separately reports its zero-high clean-inclusion
gate and its exhaustive case-exclusion gate.

Source commit `4e8c1e80d89ddeeaf2bd06eba4024d4a75bd44a3` implements and documents the temporal-exclusion policy and removes
the v6 lock. Lock-only commit `73c1f15c966535bda4f3f15662b1caf3af93809d` binds freeze-scope digest
`99cab40dc0ebf111d7880627d0a7f58ff16a8920b9df9019afc6eef704bedbac`; the lock SHA-256 is
`91098e985de63c7532e512cfd6356ab2c14e4bbd917751a44b4cc1bd44f1703b`. The complete acceptance suite passed with
384 tests and 125 subtests. Annotated tag `paper-methodology-v7` identifies the deterministic temporal-exclusion state.
The publication replay still resumes the completed Stage 2 checkpoint and does not repeat candidate discovery.

The v7 selection reserve then exposed a false positive in the few-shot hidden-metadata guard. A legitimate T-box
support entity had the visible label `classification`; serialized-substring matching treated that scalar vocabulary as
if it were a hidden `classification` key and rejected the static-few-shot cell for all 233 T-box reserve cases. With no
prompt-clean T-box prefix, total-preserving quota redistribution correctly failed rather than silently reducing the
1,200-case population. Methodology v8 makes this check structural: actual nested `classification` and `repair_target`
keys and raw support IDs remain forbidden, while identical visible scalar values are accepted.

Source commit `619c1d5a7e54b33406620152902a85c31354e870` implements and documents the correction and removes the v7 lock.
Lock-only commit `712b20fc67a30ccacc3372d325cb9aa2bcebf6f6` binds freeze-scope digest
`093cc526a764f28aadee4034896cb771dc218abbdee11b723fb89e4bc6bb7c0e`; the lock SHA-256 is
`6a156ae3f44855ab35b1c7a9bb20c66f3c6cce93fd7640d8299b0c9fe58329fd`. The complete acceptance suite passed with
385 tests and 125 subtests. Annotated tag `paper-methodology-v8` identifies the structural support-render state.
Publication execution again resumes the completed Stage 2 checkpoint without repeating candidate discovery, then
rebuilds and audits before restarting selection.

The v8 reserve confirmed the support-render repair with all 1,313 cases and 10,504 prompt cells rendered, but
finalization exposed a separate non-distinctive-token defect. The temporal scanner treated the generic `MISSING`
sentinel in unrelated visible support inputs as if it disclosed the hidden deletion target for 142 IC-L cases.
Methodology v9 adds that sentinel to the existing prompt-contract vocabulary category, where occurrences are diagnostic
and non-blocking. A complete scan of the retained v8 reserve under the correction left 44 genuinely high-risk cases
and proved exact capacity for both the 1,200-case main and nested 600-case Azure populations.

Source commit `f1de73908202ab050fa06e140bb614a75c270632` implements the version-5 temporal report correction and removes
the v8 lock. Lock-only commit `9afe3a94bbda18383a8565bf0b6d1718ae38103d` binds freeze-scope digest
`dd854165580e86d0ec382f15d17103fefe59b4d979e1c17bcfdd738f90f8e652`; the lock SHA-256 is
`32345ba54e9c124f4dcad9a1dea09c7c4e1768f2d65b7b642f8997bb981d607f`. The complete acceptance suite passed with
386 tests and 125 subtests. Annotated tag `paper-methodology-v9` identifies the missing-sentinel temporal state.
Publication execution again resumes the completed Stage 2 checkpoint without repeating candidate discovery, then
rebuilds, audits, and restarts selection.

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
