# Repository Audit: WikidataRepairEval

**Audit date:** 2026-07-13

**Branch:** `agent/post-update-audit-remediation`

**Scope:** the repository state containing this audit, including code, tests, schemas, documentation, tracked sample artifacts,
and locally available research artifacts.

## Executive Verdict

WikidataRepairEval is a mature exploratory research implementation, but it is not yet a completed confirmatory study or a
public benchmark release. The repository now has credible controls for immutable releases, two-phase protocol freezing,
private untouched-test allocation, paired analysis, optional independent annotation, temporal leakage auditing, provenance
support, and experiment registration. The high- and medium-severity implementation defects from the previous audit have
been remediated and tested. The approved no-human fallback has also been implemented and executed over the full available
Stage 4/3 population and the current rendered prompts.

The remaining blockers are scientific execution and deposition rather than missing core software:

1. Four cases are excluded pending prompt rerender after model-assisted temporal-leakage discovery.
2. No new post-freeze snapshot and privately allocated untouched test have been used for a confirmatory model run.
3. Independent human construct validation is unavailable. Automated replay and label-hidden Codex review can discover
   inconsistencies, but cannot establish semantic correctness, causal necessity, repair uniqueness, or agreement.
4. The full research artifacts have no published immutable URLs, DOI, final derived-data license, or complete upstream
   snapshot identifiers.
5. Historical model outputs predate corrected measurement contracts. They are registered as superseded and are not
   paper-eligible.

The repository can support a defensible, explicitly no-human paper after the remaining release and untouched-execution
gates are completed without adapting prompts, parsers, metrics, or analysis in response to untouched-test outcomes. Its
claims must stay narrower than a study with independent construct validation.

## Research Direction

The conceptual documentation defines a benchmark built from real historical Wikidata repairs. Its principal research axes
are:

- repair locus: entity-level A-box versus schema-level T-box repair;
- information condition: rule-implied, supported by supplied local evidence, or unresolved by supplied evidence;
- evidence bundle: `logic_only` versus `local_graph`;
- routing: oracle locus versus diagnosis-routed repair;
- output quality: parseability, executability, historical alignment, auditability, and factual provenance support;
- population reporting: main-score and diagnostic subsets, with event/property-aware uncertainty and stratified errors.

The temporally accurate description is a **historically targeted repair under later frozen context**, not a complete
reconstruction of what an editor knew at repair time. `EXTERNAL_BY_ELIMINATION` means unresolved by implemented rule and
local-evidence checks; it does not prove external evidence was causally necessary.

## Current Implementation State

### Data And Selection

- The locally available Stage 4 artifact contains 535,570 schema-valid records. The prior full scan found zero invalid rows.
- The current core has 4,800 cases: 4,204 A-box and 596 T-box; 3,818 main-score and 982 diagnostic.
- The current development tier has 600 cases: 360 A-box and 240 T-box; 481 main-score and 119 diagnostic.
- Group-aware splitting and selection enforce A-box `(qid, property)` and T-box `(property, revision)` isolation.
- The current core has already influenced task and prompt development. It is development data, not an untouched test.
- A selection CLI now creates a private allocation from an allocation-phase protocol, enforces predeclared composition and
  main-score counts, checks exclusions, and emits a separately publishable blinded aggregate.

### Exhaustive Automated Audit

- The `full_v1` run replayed all 535,570 Stage 4 rows against all 535,570 Stage 3 entries. It found no schema, identity, join,
  or missing-ID errors. Stage 2 is absent and is explicitly recorded as unavailable.
- Deterministic outcomes were 473,412 pass (88.394%), 60,537 unsupported (11.303%), and 1,621 disagreement (0.303%).
  Unsupported means insufficient implemented replay coverage, not agreement.
- The disagreements are 350 Type A rule-replay failures, 109 Type B cases whose claimed local support was not reproduced,
  and 1,162 Type C cases for which visible local support was found.
- The largest coverage gaps are 33,525 T-box causality-policy replays, 16,741 unsupported Type A subtypes, 5,402 unknown
  Type C cases, and 4,993 derived-text cases needing semantic review. Finding codes overlap and are not a partition.
- The deterministic prompt audit covered all 384 supplied prompts and 3,024 hidden claims. It found zero high-risk exact or
  normalized hits, 672 diagnostic hits, and 156 expected rule-derived hits; all four mutation-sensitivity checks passed.
- Codex reviewed 450 blinded construct packets and 50 temporal packets. Construct verdicts were 221 pass, 193 concern, and
  36 uncertain. Temporal verdicts were 39 pass, five suspected leakage, and six uncertain. All 50 batches succeeded on the
  first attempt.
- Conservative finalization retained 473,222 cases as `include`, placed 62,344 in `diagnostic`, and marked four cases
  `exclude_pending_rerender`. It did not relabel any case or emit an integrity exclusion.

The model-assisted review found error mechanisms outside the deterministic scanner's supported representation: a hidden
identifier embedded in a visible URL and hidden replacement labels/descriptions exposed under different QIDs. It also found
one direct target-value exposure. The five leakage packets map to four cases because one case appeared in two sampled
prompts. This is the clearest new audit result: a zero-hit lexical gate was necessary but not sufficient.

### Evaluation And Analysis

- The evaluator distinguishes main and diagnostic populations and checks requested-ID completeness and uniqueness.
- Track diagnosis includes confusion counts, per-locus recall, macro-F1, balanced accuracy, ambiguity, and abstention.
- Structural provenance completeness is reported separately from factual support by evaluation-visible evidence.
- The analysis CLI types binary and continuous metrics, rejects incompatible exact tests, reports pairing attrition, and
  provides cluster-aware micro, event-macro, and weighted uncertainty where applicable.
- Few-shot comparisons now require identical, unique case populations and normalized per-request resource metrics.
- The previously published disjoint-population few-shot delta is explicitly marked `invalid_superseded`.

### Governance And Reproducibility

- Dataset and evaluation releases validate every Stage 4 row and cross-check Stage 2, Stage 3, Stage 4, schemas, snapshot
  identity, selection references, hashes, and required roles.
- Release paths are portable and a verifier rehashes and revalidates an existing release.
- Confirmatory releases require a clean Git state and a real source commit.
- Protocol freezing has two phases: allocation protocols bind a dataset release; execution protocols bind the selected
  evaluation release. This removes the former release/selection dependency cycle.
- Confirmatory registry entries require a verified execution protocol, verified evaluation release, clean real commit,
  expected population, complete results, prompt/schema hashes, and model/server identity.
- Three historical runs are registered as `superseded`, with hashes and explicit ineligibility reasons.
- A tracked synthetic release proves clean-clone release verification without pretending to be research data.
- An acquisition CLI validates distribution manifests and downloads through temporary files with path, size, and SHA-256
  checks before atomic installation. The research distribution template deliberately fails closed while URLs and hashes
  remain unresolved.

## Finding Reassessment

### High-Severity Findings

#### H1. Release Validation Scope And Identity: Resolved In Code

Release validation now scans the entire released Stage 4 file, requires Stage 2, Stage 3, the Stage 4 schema, and a snapshot
manifest, and cross-validates record identities and context references. The snapshot manifest binds source identifiers,
retrieval times, limitations, and Stage 2/3/4 hashes. `artifact_release verify` rejects missing, altered, escaped, or
inconsistent artifacts and rejects a fake confirmatory commit.

Evidence: `src/artifact_release.py`, `src/snapshot_manifest.py`, `schemas/release_manifest.schema.json`,
`schemas/snapshot_manifest.schema.json`, and `release/sample-v0.1.0/`.

Residual gate: build the actual research release from a post-freeze snapshot in a clean commit. The tracked sample is
synthetic and candidate-only.

#### H2. Annotation Blinding And Adjudication: Resolved In Tooling

The annotation workflow separates blind locus/repair inference from target-conditioned evidence sufficiency. Locus cards no
longer expose `track` or `repair_target.kind`. The tooling validates completed reviews, merges assignments, measures raw
agreement and chance-corrected agreement, identifies disagreements, supports adjudication, and emits a signed final-label
artifact. Private re-identification material is operationally separated from reviewer assignments.

Evidence: `src/annotation_protocol.py`, its schemas, tests, and the authoritative research release protocol.

Residual gate: recruit independent reviewers, double-annotate the declared strata, adjudicate disagreements, and report
agreement with uncertainty. A protocol cannot substitute for independent evidence.

#### H3. Statistical Semantics: Resolved In Tooling

Metric declarations now distinguish binary and continuous outcomes. Binary exact tests are not emitted for continuous
metrics. Pairing reports expected, observed, paired, and dropped IDs instead of silently hiding attrition. The analysis
supports declared case-micro, event-macro, and population-weighted estimands with uncertainty, and records primary versus
exploratory comparisons and multiplicity policy.

Evidence: `src/analyze_results.py`, its schema/tests, and `docs-technical/Research_Release_Protocol.md`.

Residual gate: apply the frozen analysis only to a complete identical-case confirmatory result set. No current output is
valid confirmatory evidence.

#### H4. Stale Measurement Semantics In Existing Results: Contained

Historical reports were not silently rewritten under new semantics. The registry contains the full-core oracle reasoning,
T-box taxonomy-patch, and A-box few-shot runs as `superseded` and `paper_eligible: false`. Their summaries are hash-bound and
carry reasons explaining development contamination or invalid comparisons. Corrected semantics therefore apply
prospectively.

Evidence: `experiments/registry.json` and the invalid/superseded few-shot JSON and Markdown reports.

Residual gate: rerun only after the execution protocol is frozen. Historical metrics must not be presented as results from
the remediated evaluator or evidence contracts.

#### H5. Construct Validation: Automated Fallback Complete, Human Evidence Unavailable

The repository supports an optional independent-review workflow, but the current study has no access to human evaluation.
The implemented fallback exhaustively replays machine-checkable Stage 4/3 invariants and exports a stratified,
label-hidden construct sample for Codex-assisted error discovery. Codex can nominate concerns but cannot relabel cases,
produce `EXTERNAL_CONFIRMED`, or serve as an independent annotator. Documentation consistently limits extractor-relative
gold, historical alignment, and `EXTERNAL_BY_ELIMINATION` claims.

Residual limitation: causal linkage, alternative valid repairs, T-box semantic validity, and evidence necessity remain
unvalidated. This is a permanent scope limitation for the no-human paper, not a software gate that code or model review can
close.

#### H6. Contradictory Documentation: Resolved

`docs-technical/README.md` is the current implementation index and `Research_Release_Protocol.md` is the authoritative paper
workflow. The former execution plan is marked historical/superseded. Conceptual status, selection isolation, evidence
contracts, routed diagnosis, schemas, release phases, environment commands, and portability references now match the code.
Machine-specific absolute paths and obsolete credential-dependent instructions were removed. CI checks local documentation
links, protocol hierarchy, and prohibited machine paths.

Evidence: conceptual and technical documentation plus `tests/test_documentation.py`.

#### H7. Temporal Control And Provenance Semantics: Resolved In Tooling, Claim Narrowed

Proposal and diagnosis contexts prune post-repair-only target atoms from non-target properties, constraints, graph edges,
and labels while preserving legitimate reconstructed history and expected Type A rule visibility. A snapshot manifest makes
the later-context boundary explicit. The expanded temporal audit scans rendered prompts against hidden target/current
identifiers, labels, descriptions, normalized forms, serialized forms, and token boundaries; it also runs mutation checks
and produces a deterministic stratified sample for label-hidden Codex error discovery.

The current audit scanned 3,024 forbidden claims in 384 prompts across 96 cases and found zero deterministic high-risk hits.
Codex nevertheless nominated five suspected-leakage packets affecting four cases, including URL-mediated and alias-based
exposure. Those cases are now `exclude_pending_rerender`; six additional temporal reviews are diagnostic uncertainties.

Provenance reporting now separates structurally auditable output from claims supported by evaluation-visible evidence.
Neither measure proves real-world causal correctness.

Evidence: `src/temporal_audit.py`, `src/guardian/reasoning.py`, `src/guardian/evaluator.py`,
`reports/temporal_audit/current_holdout96_v4_audit.json`, and `docs-conceptual/Temporal_Validity.md`.

### Medium-Severity Findings

#### M1. Untouched-Test Allocation: Resolved In Code

Allocation is bound to a frozen dataset release and allocation-phase protocol, enforces the predeclared main-score count and
composition policy, verifies group exclusions from selected records, and separates private IDs from public aggregates.
Execution requires a later execution-phase protocol bound to the selected evaluation release.

Residual gate: operational secrecy and access control are human/infrastructure responsibilities. No repository flag can
make a manifest secret once its contents are disclosed.

#### M2. Diagnosis Reporting: Resolved

The evaluator now provides the required confusion-oriented metrics, per-locus recall, macro-F1, balanced accuracy,
ambiguity/abstention, and routing attribution on the evaluated population.

#### M3. Repository-Wide Lint: Resolved

Configured Ruff checks pass over `src`, `scripts`, and `tests`. CI runs that full configured scope. The line-length exception
is explicit (`E501`) rather than hidden by a critical-errors-only command.

#### M4. Public Metadata: Substantially Resolved, Publication Fields Pending

The project is versioned `0.1.0` and now includes `LICENSE`, `DATA_LICENSE.md`, `CITATION.cff`, `CHANGELOG.md`, a substantially
expanded dataset card, snapshot/release schemas, and checksum-oriented acquisition documentation. The files avoid inventing
unknown authors, a DOI, derived-data terms, or snapshot identifiers.

Residual gate: maintainers must choose the final derived-data license, supply contributor metadata, publish an immutable
archive, and add its DOI and exact upstream snapshot provenance. The dataset card must be finalized against that release.

#### M5. Clean-Clone Reconstruction: Resolved For Mechanism, Research Deposit Pending

The acquisition manifest schema and CLI provide a deterministic, checksum-bound clean-clone path. A synthetic release is
tracked and verified in CI. The draft research distribution manifest exposes every missing URL, checksum, size, license, and
snapshot value and fails closed rather than fabricating reproducibility.

Residual gate: upload the large Stage 2/3/4/5 artifacts and publish a populated distribution manifest. A clean clone cannot
currently obtain the full research dataset because that external deposit does not exist.

## Additional Critical Rechecks

- **Development contamination remains explicit.** The current core is exploratory. The governance workflow prevents it from
  being registered as a confirmatory execution.
- **The invalid few-shot contrast is blocked at source.** Report generation now rejects unequal or duplicate populations;
  the old report is marked invalid/superseded.
- **Registry self-certification is closed.** Fabricated hashes, nonexistent commits/protocols, wrong release phases, dirty
  confirmatory code, incomplete populations, and missing model provenance fail eligibility tests.

## Residual Risks And Bugs

1. The generic reasoning-floor summary does not score the specialized T-box taxonomy-patch task; that task currently uses
   its dedicated parser/evaluator. A unified paper table must join those outputs by case ID or the generic runner must gain
   an explicit taxonomy-patch evaluation path.
2. Automated temporal scans can miss paraphrased, alias-based, or semantically equivalent leakage. The 50-item label-hidden
   Codex sample reduces this blind spot but remains model-assisted error discovery, not independent validation.
3. Later snapshot labels, descriptions, unrelated claims, and graph structure may encode post-repair information even after
   target-atom pruning. The paper must retain the later-frozen-context qualification.
4. Exact historical alignment is only one outcome. Without independent semantic-validity and over/under-repair review, the
   paper cannot make strong claims about alternative correct repairs.
5. A single model/server configuration cannot support broad claims about LLM behavior. Either freeze multiple models and
   replicates or explicitly frame the experiment as a single-model case study.

## Scientific Claim Matrix

| Candidate claim | Current decision |
| --- | --- |
| Historical repairs can be represented as structured A/T cases | Defensible with extraction and temporal limitations. |
| Implemented group keys prevent cross-split event leakage | Defensible as an implementation claim. |
| Release and protocol artifacts are content-addressed and verifiable | Defensible for the tooling and synthetic sample. |
| The deterministic scanner found zero high-risk target-field hits | Defensible for the audited 96-case/384-prompt population; not equivalent to no leakage because Codex nominated four affected cases. |
| `local_graph` improves model repair | Unsupported until an untouched paired run. |
| Few-shot prompting improves repair | Unsupported; the historical comparison is invalid/superseded. |
| Diagnosis routing is competitive with oracle routing | Unsupported until an untouched paired run. |
| T-box gold is semantically or causally unique | Unsupported pending independent validation. |
| Type C cases require external evidence | Do not claim; use `EXTERNAL_BY_ELIMINATION`. |
| Popularity effects demonstrate memorization | Unsupported causal interpretation. |
| The full benchmark is publicly reproducible | Not yet; distribution metadata remains unresolved. |

## Recommended Paper Execution Order

1. Treat the completed exhaustive automated audit and label-hidden Codex review as an error-discovery gate. Apply its
   conservative dispositions, rerender excluded temporal cases, and retain the explicit no-human construct-validity
   limitation; do not infer replacement labels from model output.
2. Freeze a confirmatory dataset release from an exactly identified post-freeze snapshot in a clean commit.
3. Freeze an allocation protocol and privately select the untouched population with declared strata and exclusions.
4. Build the selected evaluation release, then freeze the execution protocol, models, prompts, metrics, and analysis.
5. Run every declared contrast on identical case IDs. Fail immediately on missing, duplicate, or unmatched cases.
6. Register and verify outputs before analysis; do not change the protocol after viewing outcomes.
7. Report case-micro, event/property-macro, and population-weighted estimands separately with attrition and uncertainty.
8. Deposit immutable artifacts, checksums, licenses, citation metadata, and a DOI; then finalize the dataset card and paper.

## Verification Record

Commands used the WSL project environment as required:

```text
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run pytest -q
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run ruff check src scripts tests
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m artifact_release verify \
  --manifest release/sample-v0.1.0/release_manifest.json \
  --release-root release/sample-v0.1.0
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m artifact_acquisition \
  --manifest release/artifact_distribution.template.json status --allow-unresolved
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-automated-audit finalize \
  --manifest reports/automated_audit/full_v1/manifest.json \
  --reviews reports/automated_audit/full_v1/codex_reviews.jsonl
git -c core.whitespace=cr-at-eol diff --check
```

Results:

- **399 tests passed; 82 subtests passed**.
- Repository-wide configured Ruff: **passed**.
- Documentation portability/link checks: **passed**.
- Synthetic release schema, hashes, complete-data validation, cross-artifact identity, and snapshot binding: **passed**.
- Full deterministic audit: **535,570 Stage 4/3 cases; 473,412 pass, 60,537 unsupported, 1,621 disagreement, 0 errors**.
- Temporal audit: **384 prompts, 96 cases, 3,024 forbidden claims, 0 deterministic high-risk hits, 50 AI reviews**.
- Codex review: **500/500 packets; 229 construct concerns/uncertainties, 5 suspected temporal leaks, 6 temporal
  uncertainties; 4 cases excluded pending rerender**.
- Git whitespace check (treating the repository's existing CRLF files correctly): **passed**.

## Bottom Line

The repository has moved beyond the high- and medium-severity implementation failures identified in the previous audit.
Its main risk is now overclaiming what the improved machinery has demonstrated. The software establishes enforceable
contracts for a confirmatory study and a reproducible no-human error-discovery audit; it does not retroactively validate
contaminated outputs, create human construct validity, or create an archival release. The paper should remain explicitly
exploratory until the untouched execution and public deposit gates are complete, and its construct-validity limitation must
remain explicit thereafter.
