# Post-Update Repository Audit: WikidataRepairEval

**Re-audit date:** 2026-07-13

**Repository:** `/home/wucloud/kg-benchmark`

**Branch / audited commit:** `main` / `e93eccb84fb95060c8d5c4632cc17ca12ac5c1fd`

**Worktree state:** dirty. The audit covers the working tree, including uncommitted remediation and prompt-development artifacts, not only the named commit.

**Objective:** reassess implementation progress, defects, reproducibility, and readiness for defensible academic claims after the repository updates.

## Executive Verdict

The updates resolve several of the first audit's most important engineering defects. Group-aware splitting works on the complete 535,570-case artifact; all Stage 4 records pass the current schema; the selected core passes recomputed caps and dev/core isolation; `local_graph` now contains the intended non-target L1 evidence; evaluator code distinguishes main and diagnostic subsets; the test suite passes; CI and a tracked smoke fixture now exist; and run/release/annotation/analysis governance tools have been added.

The repository is nevertheless **not ready for confirmatory paper claims or a public benchmark release**. The main reason is that tooling has advanced further than the scientific evidence and, in several places, the new tools claim stronger guarantees than they enforce:

1. The current core remains development-contaminated, and no post-freeze untouched test exists.
2. The paper-facing few-shot report compares disjoint populations: a 96-case dev zero-shot run and a 4,204-case core A-box few-shot run have zero shared case IDs.
3. The experiment registry accepts fabricated hashes, a fake commit, and a nonexistent protocol as `paper_eligible`.
4. Release validation checks only selected Stage 4 rows while fingerprinting the whole file; an invalid unselected row does not fail a release candidate.
5. The annotation protocol says it is blind to track while retaining `repair_target.kind`, which directly reveals `A_BOX` or `T_BOX`, and it implements neither agreement calculation nor adjudication.
6. The statistical tool emits a paired binary exact test for arbitrary continuous metrics, silently drops incomplete pairs, and gives no interval for event-macro or weighted effects.
7. Existing model outputs predate the corrected evaluator/context/provenance contracts and are not registered; the experiment registry is empty.
8. Governing documentation remains contradictory and several public-release essentials are absent.

The appropriate current status is **a strong exploratory benchmark implementation with verified data plumbing, pending independent construct validation, a clean protocol freeze, a genuinely held-out evaluation, corrected analysis, and an archival release**.

## Readiness Summary

| Area | Current state | Re-audit assessment |
|---|---|---|
| Research framing | Coherent A/T locus and evidence-access taxonomy | Strong, provided claims remain operational and extractor-relative. |
| Stage 4 artifact | 535,570 records; current v2 schema | Fully schema-valid in this audit. |
| Grouped splitter | Implemented and full-artifact tested | Passes with zero cross-split groups and acceptable proportions. |
| Core/dev manifests | 4,800 core; 600 dev | Current manifests pass selected-row schema, caps, partition, and overlap checks. |
| Evidence bundles | `logic_only` and corrected `local_graph` | Contract is improved, but effects must be rerun on an untouched population. |
| Evaluator | Main/diagnostic metadata and stricter identity checks implemented | New semantics are tested; historical result files were not regenerated. |
| Few-shot study | Full A-box run exists | Current delta report is invalid because comparator populations are disjoint. |
| Diagnosis-routed study | Runner support exists | No confirmatory routed-vs-oracle evidence exists. |
| T-box task | Taxonomy-patch task and full core reports exist | Gold remains extractor-derived and core-developed. |
| Independent annotation | Assignment generator exists | Blinding is incomplete; no second annotations, agreement, or adjudication exist. |
| Statistical inference | Initial paired cluster analysis tool exists | Not yet adequate as a general paper analysis layer. |
| Reproducibility | Hash capture, release, registry, CI tools exist | Release/registry guarantees are incomplete and no actual release is present. |
| Public artifact | Public docs exist | No deposited benchmark, DOI, license, citation file, checksums, or complete dataset card. |
| Paper claim readiness | Exploratory only | Confirmatory claims should not be made from current outputs. |

## Scope And Method

The re-audit covered:

- all research framing under `docs-conceptual/`;
- technical plans, implementation descriptions, completion notes, and runbooks under `docs-technical/`;
- public documentation under `docs-public/`;
- the updated classifier, grouped splitter, selector, reasoning context, evaluator, prompt-development, release, registry, annotation, untouched-test, and analysis code;
- schemas, core/dev manifests, prompt-development summaries, full-core reasoning outputs, manual-audit artifacts, CI, and sample data;
- full-artifact schema validation and grouped splitting;
- adversarial checks of release validation, experiment eligibility, annotation blinding, and statistical metric behavior.

No model endpoint or external LLM API was called. No benchmark or result artifact was changed. Temporary smoke/adversarial outputs were removed after verification; this file is the only audit deliverable.

## Research Direction And Safe Claims

The intended contribution remains a structured evaluation protocol for real historical Wikidata repair events:

- distinguish entity-level A-box repair from schema-level T-box repair;
- classify cases by whether the target is rule-implied, supported by supplied local evidence, or unresolved by supplied evidence;
- compare `logic_only` and `local_graph` evidence bundles;
- compare oracle locus with diagnosis-routed repair;
- reconstruct the historical pre-repair target property to reduce target leakage;
- require parseable, executable, historically aligned, and auditable proposals;
- report main and diagnostic slices and stratified failures rather than one leaderboard score.

The strongest defensible current characterization is:

> WikidataRepairEval operationalizes historical Wikidata repair events as structured, temporally sanitized evaluation cases and provides tested infrastructure for comparing repair locus, supplied-evidence conditions, and executable historical alignment.

Claims that remain unsupported include "leakage-free," "external evidence required," "causally correct repair," "few-shot improves repair," "local graph improves repair," "diagnosis routing is effective," "popularity demonstrates memorization," or general performance claims about LLMs.

## Verified Remediation Since The First Audit

### Full-Artifact Data And Split Integrity

- All **535,570** lines in `data/04_classified_benchmark.jsonl` parsed and passed `schemas/04_classified_benchmark.schema.json`; invalid records: **0**.
- Stage 4 SHA-256 is `dfbdc9286cc54e10dfe7092ecdf78f95e9852b9271c564b3dfa2447b4487f344`.
- Stage 3 world-state SHA-256 is `7e3591a9bfc7289c698d1cc87339604b3c796a042d8ce96bb0d67f435bfa21bf`.
- A pure full-artifact grouped split completed in 84.8 seconds: train **427,865**, dev **54,053**, test **53,652**.
- The full split contained **79,243** groups, no weak groups, **0** cross-split groups, and no proportion warnings.
- `src/splitter.py` now allocates complete A-box `(qid, property)` and T-box `(property, revision)` groups; this is no longer only a helper-level claim.

### Core/Dev And Evaluator Contracts

- Core contains **4,800** cases: 4,204 A-box and 596 T-box; 3,818 main-score and 982 diagnostic.
- Dev contains **600** cases: 360 A-box and 240 T-box; 481 main-score and 119 diagnostic.
- Current manifests report zero case, A-box-group, and T-box-revision overlap.
- Recomputed core release-input validation found all 4,800 IDs, zero duplicates, zero selected-row schema errors, maximum A-box group size 1, maximum T-box revision size 10, and zero exclusion overlap.
- The evaluator now carries evaluation subset, selection metadata, and cluster keys; it rejects missing requested IDs and duplicates and uses order-insensitive A-box value comparison.
- Popularity bucketing is now stable across selection and evaluation.
- The Type A audit denominator implementation now covers all non-ambiguous Type A subtypes.

### Evidence, Testing, And Developer Workflow

- Proposal `local_graph` construction retains all non-target L1 focus properties while removing current target-property values.
- The complete suite passes: **349 tests and 76 subtests**.
- The tracked synthetic classifier/splitter workflow passes and reports zero group overlap.
- The five new modules and their focused tests pass Ruff.
- GitHub Actions runs tests, critical error/name lint checks, and the sample workflow.
- All new governance CLIs import and expose entry points through `pyproject.toml`.

These changes materially improve implementation reliability. They do not validate old results or turn the current core into a held-out test.

## Critical Findings

### C1. Current Core Results Remain Development-Contaminated

The current core has informed repeated prompt, parser, gate, and task decisions. Full-core oracle runs, core canaries, full T-box taxonomy-patch development, and the full A-box few-shot run have all exposed core behavior. `docs-technical/Research_Release_Protocol.md:5-7` correctly recognizes this.

No post-freeze Stage 4 snapshot, sealed test manifest, preregistration artifact, or untouched model result exists. The `sealed/` and `releases/` outputs described by the new runbook are absent.

**Impact:** current core metrics can support exploratory error analysis, not confirmatory generalization or final model selection.

**Required action:** freeze prompts, schemas, models, generation settings, primary metrics, exclusions, and analysis; then build a new post-freeze snapshot and group/property-isolated test without inspecting outcomes until execution is complete.

### C2. The Paper-Facing Few-Shot Delta Compares Disjoint Populations

`src/lib/prompt_dev.py:3270-3276` matches matrices only by task, representation, context, and track mode. It does not require equal case sets. The generated report states the same comparison rule (`few_shot_delta_vs_zero_shot.json`).

Direct artifact inspection found:

- zero-shot baseline: 96 dev cases per matrix, comprising 48 A-box and 48 T-box cases;
- few-shot treatment: all 4,204 core A-box cases per matrix;
- shared case IDs between the corresponding matrices: **0**;
- total rendered prompts compared at run level: 192 versus 8,408.

The report nevertheless labels static few-shot as `paper-facing` and subtracts aggregate accepted rates and token totals. These differences mix treatment, dataset tier, track composition, case difficulty, and sample size.

**Impact:** every reported few-shot delta and total token/latency overhead in this artifact is scientifically invalid as a few-shot effect.

**Required action:** run zero-shot and few-shot on the identical frozen case set and join by `(case_id, context_bundle, task_version, track_mode, evaluation_subset)`. Fail report generation on unequal IDs or missing pairs. Report per-case token differences or normalized tokens/request, not unmatched run totals.

### C3. The Experiment Registry Can Falsely Certify A Run

`src/experiment_registry.py:12-35` tests only whether provenance-shaped strings are present. It does not resolve referenced files, recompute their hashes, validate the Git commit, require an existing protocol, bind to a release manifest, verify selected counts, or confirm the model digest.

An adversarial summary containing `not-a-real-digest`, `not-a-real-commit`, three `fake` hashes, one claimed main-score case, and protocol `nonexistent-protocol` was registered as:

```json
{"paper_eligible": true, "ineligibility_reasons": []}
```

The caller also chooses `--status confirmatory`; the registry does not derive that status. `experiments/registry.json` currently has no entries, so none of the historical runs has even passed this weak gate.

**Impact:** `paper_eligible` is currently a self-asserted metadata completeness flag, not an integrity guarantee. Treating it as a paper gate risks accidental certification of fabricated or stale evidence.

**Required action:** register an immutable release/protocol ID; resolve and rehash every referenced file; verify the commit exists and is clean; verify prompt/schema hashes, model/server identity, expected case population, result completeness, and supersession; make confirmatory status derivable only after all checks pass.

## High-Severity Findings

### H1. Release Validation Does Not Validate The Release It Fingerprints

`src/artifact_release.py:61-76` validates schema only for IDs selected by the supplied manifest, but `build_release_manifest()` fingerprints the complete Stage 4 file and labels it a benchmark release candidate. A temporary Stage 4 file containing one selected valid row and one unselected schema-invalid row still returned `release_validation_passed: true`.

Additional gaps:

- Git dirty state is recorded but does not fail candidate creation.
- Stage 2 historical repairs and Stage 3 world state are optional `--artifact` arguments, although the public benchmark release says it includes them.
- Added artifacts are hashed but not schema-checked or cross-validated against Stage 4 references.
- Paths are stored as machine-specific absolute paths.
- No command verifies an existing manifest and its hashes, despite the archived-reproduction runbook requiring verification.
- No actual release manifest or archive is present.

The current canonical Stage 4 happens to pass full validation, so this is a release-gate design defect rather than evidence of current data corruption.

**Required action:** distinguish dataset-wide and selected-evaluation releases; validate every released row and required artifact; require clean Git for confirmatory candidates; store release-relative logical paths; add `verify-release`; and cross-check Stage 2/3/4 IDs, schemas, snapshot identifiers, and selection references.

### H2. Annotation Assignments Are Not Blind To Repair Locus

`src/annotation_protocol.py:95-107` removes top-level `track`, `classification`, and `information_type`, but retains the complete `repair_target`. A generated evidence card exposed:

```json
{
  "kind": "A_BOX",
  "action": "UPDATE",
  "old_value": ["Q2"],
  "new_value": ["Q3"]
}
```

Thus track/locus is directly visible through `repair_target.kind`; historical author and exact repaired values are also visible. Exact target values may be appropriate for assessing whether supplied evidence supports a known target, but they make this a target-conditioned evidence audit, not a blind repair or locus annotation.

The module creates assignments only. It does not ingest completed reviews, validate allowed values, calculate agreement, surface disagreements, adjudicate, or produce confidence intervals. The private map is stored beside assignments with ordinary filesystem permissions and absolute paths.

**Impact:** the protocol cannot independently validate A-box/T-box diagnosis as documented, and no independent validity evidence has yet been produced.

**Required action:** define separate tasks: blind locus/repair inference, and target-conditioned evidence-sufficiency review. Redact `repair_target.kind` for locus review, randomize field order where relevant, isolate the re-identification map operationally, and implement merge, agreement, adjudication, and signed final-label artifacts.

### H3. The Statistical Tool Is Not Yet A General Paper Analysis Layer

`src/analyze_results.py` is a useful start, but several semantics are unsafe:

- `paired_binary_test` is emitted for every metric. For a synthetic continuous `information_preservation` effect of +0.4, the tool emitted zero discordant successes and `p=null` rather than rejecting the binary test as inapplicable.
- Cases missing either condition are silently omitted; only the final paired count reveals attrition.
- The cluster bootstrap interval is attached to the pooled case-micro effect. The event-macro effect has no interval.
- Population-weighted effects have no interval.
- There is no primary-hypothesis declaration, multiplicity handling, repeated-run/model hierarchy, or standardized effect table.
- The analysis tool has not been run on a valid untouched result set.

**Impact:** indiscriminate CLI use can create misleading inferential fields, and the current output is insufficient for the planned multi-outcome paper.

**Required action:** type metrics as binary/continuous; reject incompatible tests; report expected, observed, paired, and dropped IDs; bootstrap both declared micro and macro estimands; add uncertainty for weights; support run/model clustering or explicitly narrow the estimand; and encode primary versus exploratory comparisons.

### H4. Updated Measurement Semantics Have Not Been Applied To Existing Results

The latest full-core reasoning summary predates the remediation. It has no `paper_subsets`, artifact fingerprints, Git state, model digest, or new run provenance. Its bundle summaries each aggregate 4,800 cases. Historical outputs also used the previous `local_graph` evidence contract.

The experiment registry is empty, the audit summary was not regenerated after its denominator fix, and no new release/analysis artifacts exist. Code-level fixes cannot retroactively alter saved metrics.

**Impact:** existing tables and reports remain exploratory historical artifacts and must not be cited as results from the corrected protocol.

**Required action:** mark them explicitly exploratory/superseded, regenerate descriptive reports only if useful for development, and produce confirmatory outputs solely from the frozen untouched run.

### H5. Construct Validity Is Still Extractor-Relative And Single-Annotator

The original 450-row audit remains single-annotator. No independent agreement, adjudication, or clustered uncertainty exists. No audited Type C case established true external evidence need, so `EXTERNAL_BY_ELIMINATION` remains a negative result of supported local/rule checks, not evidence of necessity.

T-box taxonomy-patch gold remains derived from benchmark-internal constraint deltas and classifier policy. Exact historical alignment is not equivalent to semantic validity, causal necessity, or uniqueness of the historical repair.

**Impact:** performance against current gold primarily measures agreement with the extraction and canonicalization pipeline.

**Required action:** independently review a stratified property/revision sample; report agreement and adjudication by stratum; annotate causal linkage and alternative valid repairs; retain `EXTERNAL_BY_ELIMINATION` as an unresolved/no-retrieval stress label unless retrieval or domain review confirms otherwise.

### H6. Governing Documentation Contradicts The Current Repository

Material examples include:

- `docs-conceptual/00-full_project_description.md:5` says the next phase is classifier hardening, dataset definition, prompt engineering, and manual audit, all of which already occurred.
- The same document at line 168 describes the legacy keep-all-A-box selector.
- `docs-conceptual/dataset_tiers_and_selection.md:160` permits warned A-box overlap, while current code hard-fails it; its example validation block omits A-box overlap.
- `docs-technical/Pipeline_Implementation.md:170` documents obsolete `TypeC/EXTERNAL`, and lines 192-195 describe the legacy selection policy.
- `docs-technical/Conceptual_Deviation_Report.md:130-138` says diagnosis is scored separately and cannot route proposals, although `diagnosis_routed` exists.
- `README.md:116-145` still advertises a legacy `paper_eval_tbox_cap_100_seed_13.json` workflow instead of the current core/release protocol.
- `docs-technical/Paper_Execution_Plan.md` treats the contaminated core and external-provider credentials as the paper execution path, conflicting with the newer release protocol.
- `docs-technical/Reasoning_Floor.md` retains six absolute `/mnt/c/Code/...` Markdown links; two more machine-specific working-directory commands remain elsewhere.
- `docs-technical/Research_Release_Protocol.md:63`, 98, and 112 overstate annotation blinding, analysis coverage, and derived paper eligibility.

**Impact:** a collaborator can follow a documented but obsolete or scientifically invalid workflow. This is a research-governance defect, not merely editorial debt.

**Required action:** designate one current protocol index; mark historical plans as superseded; update conceptual status without changing research decisions; rewrite technical behavior to match code; and add documentation link/command checks to CI.

### H7. Temporal Control And Evaluation Semantics Remain Partial

Historical target-property reconstruction is a meaningful safeguard, but surrounding labels, descriptions, non-target properties, neighbors, and constraints are from a later snapshot. Current-state constraint persistence is not a repair-time regression oracle. Auditability/provenance metrics still primarily check structure and field presence rather than factual support.

**Impact:** the task is a historically targeted repair under a later frozen context, not a complete reconstruction of what was knowable at repair time. Audit-field completeness must not be reported as evidence correctness.

**Required action:** archive snapshot identity, run field-level target/leakage scans, audit a stratified prompt sample, distinguish structural provenance completeness from supported citation accuracy, and qualify temporal claims explicitly.

## Medium-Severity Findings

### M1. Untouched-Test Selection Is A Candidate Generator, Not A Seal

The selector correctly excludes known case/group identities, can hold out properties, rejects weak keys, and warns that operational access control is required. However:

- the manifest itself contains all selected IDs and labels while its `status` says sealed;
- exclusion beyond exact case IDs depends on prior manifests having complete `case_annotations`;
- selection is hash-order group packing with no declared class/track/subtype/popularity targets;
- it reuses core eligibility policy, so the resulting test can have a different and uncontrolled estimand;
- `excluded_group_overlap` is computed after excluded groups were already removed, not independently reconstructed from selected records.

Add an encrypted/private allocation workflow, predeclared composition constraints, independent post-selection validation, and a public blinded aggregate manifest.

### M2. Track-Diagnosis Reporting Is Incomplete

The evaluator reports diagnosis accuracy, presence, ambiguous rate, and errors, but the conceptual plan calls for confusion-oriented and macro metrics. Final analysis should include a confusion matrix, balanced accuracy or macro-F1, per-locus recall, ambiguity/abstention, and downstream routing attribution on the identical case set.

### M3. Repository-Wide Lint Still Fails

The full test suite passes and new modules are clean, but repository-wide Ruff reports **258** findings, mainly line length and import ordering. CI intentionally runs only critical error/name checks. This is not a scientific blocker, but it weakens maintenance and can hide new style regressions among existing debt. Establish a clean scoped baseline or reduce debt before release.

### M4. Public Release Metadata Is Incomplete

`docs-public/Dataset_Card.md` is still a short overview. It lacks exact snapshot/dump identifiers, selection funnel and distributions, source and derived-data licensing, editor/privacy treatment, bot and language bias, missingness/network failure analysis, maintenance/versioning, contamination, subgroup limitations, and checksums. There is no `LICENSE`, `CITATION.cff`, DOI, versioned archive, or repository release manifest. The project version remains `0.0.0`.

### M5. The Full Pipeline Is Not Reconstructible From The Current Workspace

Local `data/` contains Stage 3, its SQLite sidecar, and lean Stage 4, but not Stages 0-2 or a canonical Stage 5 split. The workspace holds about 44 GB under `data/`, 41 GB under `reports/`, and 2.6 GB under `logs/`; headline data and reasoning outputs are ignored by Git. A clean clone cannot rebuild or reproduce the benchmark without an external artifact source that has not yet been published.

## Scientific Claim Matrix

| Candidate claim | Current support | Decision |
|---|---|---|
| Historical repairs can be represented as structured A/T repair cases | Pipeline, schemas, and artifacts support this | Defensible with extraction and temporal limitations. |
| Group-aware partitioning prevents event-group leakage | Full 535,570-row split passed | Defensible for the implemented group keys; document key semantics. |
| `local_graph` supplies the intended independent L1 evidence | Source/tests support the new contract | Defensible as an implementation claim, not yet as a model-effect claim. |
| Local context improves repair | No untouched rerun under corrected context | Unsupported. |
| Static few-shot improves repair | Current comparison has zero shared cases | Invalid current evidence. |
| Diagnosis-routed repair is competitive with oracle routing | No confirmatory paired run | Unsupported. |
| T-box taxonomy patches measure semantically valid schema repair | Gold is extractor-derived | Exploratory until independent validation. |
| Type C requires external evidence | Negative local/rule search only | Do not claim; retain `EXTERNAL_BY_ELIMINATION`. |
| Popularity effects demonstrate memorization | Strong property/class/context confounding remains | Unsupported causal interpretation. |
| Released results are reproducible from immutable artifacts | No release, empty registry, weak eligibility gate | Unsupported. |

## Recommendations For Scientific Value

1. **Pre-register a narrow primary study.** Name one primary paired contrast, one primary outcome, the evaluation subset, cluster unit, exclusion rules, and multiplicity policy. Treat the remaining slices as exploratory.
2. **Build a truly new test population.** Use a post-freeze snapshot, complete event groups, optional property holdout, declared composition targets, and operational separation between allocation and analysis teams.
3. **Make every ablation paired.** Zero/few-shot, logic/local, and oracle/routed conditions must use identical IDs and task versions. Report join failures as fatal.
4. **Separate capability and population estimands.** Publish case-micro, event/property-macro, and population-weighted effects separately. Explain what distribution each estimates.
5. **Redesign independent validation.** Split blind locus/repair annotation from target-conditioned evidence sufficiency; double-annotate; adjudicate; publish agreement and clustered intervals.
6. **Validate T-box gold at the property-revision level.** Include causal link, operation family, value delta where knowable, and alternative acceptable repairs.
7. **Add evidence controls.** Compare real local context with token-matched shuffled/decoy context, record prompt length/truncation, and assert whether gold-supporting evidence is visible per case.
8. **Evaluate semantic alternatives.** Keep exact historical alignment, but add independently reviewed semantic validity and over/under-repair outcomes rather than treating one historical edit as uniquely correct.
9. **Narrow or broaden model claims deliberately.** Either run multiple frozen local models/replicates or explicitly frame the work as a single-model case study. Do not infer general LLM behavior from one serving stack.
10. **Publish the selection funnel.** Report candidate, reconstructed, persistent, classified, eligible, selected, main, diagnostic, and excluded counts with network/missingness causes.

## Prioritized Completion Gates

### Gate 0: Remove Invalid Paper Artifacts

- Relabel the current few-shot delta report as invalid/superseded.
- Register all historical runs as exploratory only after strengthening the registry.
- Prevent any table builder from consuming unregistered, unequal-population, or all-selected aggregates.

### Gate 1: Strengthen Governance Code

- Make registry eligibility cryptographically and semantically verified.
- Validate the entire released dataset and all required linked artifacts.
- Add release-manifest verification and portable paths.
- Split annotation tasks and implement agreement/adjudication.
- Type statistical metrics and report pairing attrition and full uncertainty.

### Gate 2: Freeze And Validate The Protocol

- Resolve documentation contradictions and publish one authoritative runbook.
- Complete independent A-box/T-box/information-condition review.
- Complete temporal leakage and provenance-support audits.
- Freeze prompts, models, digests, runtime settings, schemas, metrics, and analysis code in a clean commit.

### Gate 3: Execute The Untouched Study

- Build and privately allocate a post-freeze test.
- Run identical-case zero/few, logic/local, and oracle/routed conditions as preregistered.
- Capture complete run provenance and immediately verify/register outputs.
- Do not adapt prompts, parsers, or metrics after viewing test outcomes.

### Gate 4: Release And Report

- Produce primary paired tables with cluster-aware intervals and transparent attrition.
- Publish main and diagnostic results separately.
- Deposit immutable Stage 2/3/4, manifests, schemas, prompts, raw/normalized outputs as allowed, checksums, and verification instructions.
- Add license, citation metadata, dataset card, ethics/privacy, limitations, version, and DOI/archive reference.

## Verification Record

Commands used the WSL project environment where applicable:

```text
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run pytest -q
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run ruff check .
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run ruff check <new modules and tests>
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m classifier --sample --no-progress --no-full-output
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m splitter --sample
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python <full Stage 4 schema scan>
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python <full grouped split build>
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python <selected-core release validation>
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python <registry, release, annotation, and analysis adversarial checks>
sha256sum data/04_classified_benchmark.jsonl data/03_world_state.json
```

Results:

- tests: **349 passed; 76 subtests passed**;
- focused new-module lint: **passed**;
- repository-wide lint: **258 findings**;
- sample classifier/splitter: **passed**;
- full Stage 4: **535,570 records; 0 invalid**;
- full grouped splitter: **79,243 groups; 0 cross-split groups; 0 proportion issues**;
- selected core: **4,800/4,800 observed; 0 schema errors; all current release-input checks passed**;
- few-shot comparator overlap: **0 case IDs**;
- fabricated confirmatory summary: **incorrectly registered as paper eligible**;
- release file with an unselected invalid record: **incorrectly passed release validation**;
- continuous metric: **incorrectly received a binary-test block**;
- generated annotation card: **retained direct A-box locus and historical target fields**.

## Bottom Line

The repository has moved from having major implementation-contract defects to having a credible, well-tested benchmark substrate. The remaining blockers are concentrated at the boundary between software checks and scientific guarantees. The next milestone should not be another model run on the current core. It should be a governance-hardening pass, independent construct validation, and a clean preregistered untouched study. Until those are complete, current model results and few-shot reports must remain exploratory.
