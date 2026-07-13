# Research Release Protocol

This document is the authoritative engineering runbook for converting the current exploratory repository state into a
reproducible paper artifact. Research rationale remains in `docs-conceptual/`; this file covers executable controls.
The older [Paper Execution Plan](./Paper_Execution_Plan.md) is historical and superseded.
The concrete v2 lineage, post-freeze acquisition, reserve, and 1,200/600 finalization commands are in the
[Confirmatory Release Runbook](./Confirmatory_Release_Runbook.md); where older allocation examples below differ, that
runbook governs confirmatory releases.

## Current Status

The existing core and its model outputs are development-contaminated and remain exploratory. The tools below repair
measurement and governance contracts, but they do not retroactively make old results confirmatory. Independent human
evaluation is unavailable; the accepted fallback is exhaustive automated consistency checking plus label-hidden
Codex-assisted error discovery with narrower claims. A post-freeze snapshot, untouched-test execution, and final release
materials still require execution.

The historical `full_v1` fallback audit completed on 2026-07-13. It covered all 535,570 Stage 4/3 cases and 384 supplied prompts,
then conservatively marked four cases `exclude_pending_rerender` after model-assisted temporal error discovery. Those cases
must be rerendered and the v2 audit rerun against restored Stage 2 content before the current prompt set can enter a release candidate. See
[Automated Consistency Audit](./Automated_Consistency_Audit.md) for measured results.

The governance workflow has two temporal freeze boundaries. A pre-acquisition **methodology protocol** binds methods,
policies, analysis, prompts, model identifiers, and inference settings without depending on a not-yet-created release or
selection. After private allocation, an **execution protocol** binds the selected evaluation release, selection hashes,
and immutable deployment revisions before any model run. A checksum-bound dataset/allocation protocol is still produced
between them to authorize private selection. The registry accepts only execution protocols for confirmatory results.

## Local Smoke Workflow

The tracked `data_sample/` fixture is synthetic and never enters paper results:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m classifier --sample --no-progress --no-full-output
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m splitter --sample
```

Sample statistics are written under `data_sample/`, not into canonical reports.

## Measurement And Schema Gates

- `src/splitter.py` assigns complete A-box QID/property and T-box property/revision groups.
- `local_graph` reconstructs the historical target property and retains independent non-target L1 evidence.
- selection and evaluation share a fixed popularity-bucket policy.
- evaluator outputs distinguish `main_score`, `diagnostic`, and `all_selected`.
- `schemas/04_classified_benchmark.schema.json` is the current Stage 4 v2 contract.
- all repository schemas are checked by `tests/test_json_schemas.py`.

Taxonomy-patch reasoning runs automatically score versioned, content-bound gold with the separate evaluator described in
[Evaluation Harness](./Evaluation_Harness.md). Their ordinary bundle evaluation is A-box-only, taxonomy metrics are
written separately, and no combined A-box/T-box repair-success score is emitted.

Paper runs additionally require the content-addressed prompt profile. It freezes the actual A-box and T-box execution
templates, supported T-box operation vocabulary, zero-shot representation, context bundles, oracle routing, response
policy, and implementation hashes. `kg-experiment-plan` verifies its file digest, and `kg-reasoning-floor` revalidates
and records it before inference. Parse failures receive no semantic repair retry.

Before building a release, write a snapshot manifest that names the data sources, retrieval time, temporal-context
policy, and SHA-256 digests for Stage 2, Stage 3, and Stage 4. It must validate against
`schemas/snapshot_manifest.schema.json`. The release builder verifies those digests rather than trusting the snapshot
label.

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m snapshot_manifest build \
  --snapshot-id SNAPSHOT_ID \
  --stage2 data/02_wikidata_repairs.json \
  --world-state data/03_world_state.json \
  --stage4 data/04_classified_benchmark.jsonl \
  --source wikidata-history=SOURCE_ID=RETRIEVED_AT_UTC \
  --source wikidata-context=SOURCE_ID=RETRIEVED_AT_UTC \
  --limitation "Later frozen context; only the target property is historically reconstructed." \
  --output releases/snapshot_manifest.json
```

Build a release candidate only after the full Stage 2/3/4 identity graph, Stage 4 schema, selection caps, group
isolation, subset partition, and snapshot binding pass:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m artifact_release build \
  --release-root . \
  --release-kind evaluation \
  --release-status candidate \
  --stage2 data/02_wikidata_repairs.jsonl \
  --world-state data/03_world_state.json \
  --stage4 data/04_classified_benchmark.jsonl \
  --schema schemas/04_classified_benchmark.schema.json \
  --snapshot-manifest releases/snapshot_manifest.json \
  --selection-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --output releases/candidate_manifest.json
```

The command writes portable paths, file sizes, SHA-256 digests, snapshot identity, and Git state, and refuses a failing
candidate. A confirmatory release additionally requires `--release-status confirmatory` from a clean Git commit.

Verify an archived release without fetching live data:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m artifact_release verify \
  --manifest releases/candidate_manifest.json \
  --release-root .
```

Render every frozen prompt before inference and run the hidden-field scan. The command checks exact and normalized target
identifiers, labels, descriptions, serialized forms, and boundary behavior, and retains a deterministic stratified sample
for Codex-assisted error discovery:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m temporal_audit \
  --rendered-prompts reports/PROMPT_RUN/prompt_dev_rendered_prompts.jsonl \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --sample-size 50 \
  --output reports/PROMPT_RUN/temporal_audit.json
```

Expected Type A rule-derived visibility is reported separately from high-risk later-context leakage. A passing automated
gate does not turn the Codex-assisted sample into ground truth or prove the absence of semantic leakage.

## Automated Consistency Audit

Run the three-step `kg-automated-audit` workflow before release review against the full Stage 4 and Stage 3 artifacts,
the Stage 4 schema, the current construct sample, and the current rendered-prompt artifacts. The detailed inputs,
commands, outputs, and disposition policy are in
[Automated Consistency Audit](./Automated_Consistency_Audit.md).

This workflow combines deterministic checks with Codex-assisted error discovery. It is the accepted replacement for an
unavailable human study, but it does not establish human construct validity or semantic ground truth. Missing inputs remain
explicitly unavailable. A supplied Stage 2 path receives a pass only after Stage 2/3/4 identity, ordering, and lean-projection
content validation; file presence is never sufficient.

## Run Provenance

New reasoning-floor runs store in `run_config.json` and the final summary:

- Git commit and dirty state;
- SHA-256, size, and absolute path for Stage 4, world state, selection manifest, and output schemas;
- resolved context length, output limit, temperature, top-p, seed, retries, and reasoning effort;
- model digest when provided by the provider or `OLLAMA_MODEL_DIGEST`/`MODEL_DIGEST`;
- prompt/task versions, routing mode, selected IDs, and visible-ID mapping.

A missing model digest or dirty worktree makes a run ineligible for confirmatory registration.

The extensible model/population configuration, generation-cache identity, and evaluation replay procedure are specified
in [Model Execution Matrix](./Model_Execution.md). Its `--require-methodology-frozen` validation is the pre-acquisition
gate; `--require-frozen` is the execution gate. The tracked matrix is methodology-frozen but remains intentionally
ineligible for execution while final selection hashes and unresolved model revisions are pending.

## Construct-Validity Boundary

The repository retains annotation tooling for possible future work, but human annotation is not a gate for the current
study. The automated audit exports label-hidden construct packets and uses Codex only to nominate errors. A deterministic or
Codex disagreement downgrades a case conservatively; it does not create a replacement label. T-box taxonomy gold remains
extractor-relative, and exact historical alignment does not prove semantic validity, causal necessity, or uniqueness.

## Untouched Test And Two-Boundary Freeze

Freeze prompts, model identifiers/settings, metrics, analysis code, acquisition procedure, and allocation policy before
acquiring candidates. This methodology protocol has `protocol_phase=methodology`, has `release: null`, and allows model
digests to remain unresolved. Use a post-freeze
benchmark snapshot; code-level exclusions on the existing snapshot are not sufficient to establish untouchedness.

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-protocol-freeze build \
  --protocol-root . \
  --protocol-id methodology_v1 \
  --protocol-phase methodology \
  --model qwen3:30b \
  --model llama3.3:70b \
  --model gpt-oss:120b \
  --model gpt-5.6-sol \
  --condition logic_only \
  --condition local_graph \
  --prompt experiments/paper_prompt_profile_v1.json \
  --schema schemas/verified_repair_proposal.schema.json \
  --schema schemas/tbox_taxonomy_patch_proposal.schema.json \
  --methodology-file experiments/paper_execution_models_v1.json \
  --methodology-file protocols/post_freeze_acquisition_v1.json \
  --methodology-file protocols/selection_policy_v1.json \
  --analysis-plan protocols/analysis_plan_v1.md \
  --expected-selected-count 1200 \
  --expected-main-score-count 1200 \
  --status frozen \
  --output protocols/methodology_v1.json
```

After acquisition and audit, build a **dataset** release without a selection manifest, then create the allocation protocol. Every model must use
a stable digest, and every prompt, schema, and analysis-plan file is hashed into the protocol:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m protocol_freeze build \
  --protocol-root . \
  --protocol-id allocation_v1 \
  --protocol-phase allocation \
  --release-manifest releases/dataset_confirmatory_manifest.json \
  --model gpt-oss:120b=MODEL_DIGEST \
  --condition logic_only \
  --condition local_graph \
  --prompt src/guardian/prompts.py \
  --schema schemas/verified_repair_proposal.schema.json \
  --schema schemas/tbox_taxonomy_patch_proposal.schema.json \
  --analysis-plan protocols/analysis_plan_v1.md \
  --expected-selected-count 1200 \
  --expected-main-score-count EXPECTED_MAIN_COUNT \
  --status frozen \
  --output protocols/allocation_v1.json
```

The placeholder counts and digest must be replaced with predeclared values. The referenced analysis plan must exist
before freezing; edits after freezing invalidate verification.

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m select_untouched_test \
  --classified-benchmark data_post_freeze/04_classified_benchmark.jsonl \
  --exclude-manifest reports/benchmark_selection/dev_prompt_v1_seed_13.json \
  --exclude-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --stratum-targets protocols/untouched_test_strata_v1.json \
  --protocol-manifest protocols/allocation_v1.json \
  --protocol-root . \
  --property-holdout \
  --private-output sealed/untouched_test_v1.json \
  --public-output releases/untouched_test_v1_public.json
```

The selector excludes prior case, event-group, and optionally property identities; weak group keys are ineligible. Treat the output as sealed until the protocol is registered.

With the private selection in place, build an **evaluation** release that includes that selection manifest. Then freeze
an execution protocol with `--protocol-phase execution` and the evaluation release. The execution protocol may be built
from a later clean commit than the dataset release; both commit identities remain recorded. Do not run a model until the
execution protocol verifies. Confirmatory registry entries reject allocation protocols and non-evaluation releases.

No external API is required by these governance tools. Model execution may use local Ollama. Any API reference condition, if retained in the research scope, must be separately approved and recorded.

## Statistical Analysis

Combine the two per-bundle trace files into one analysis input, then compute a paired cluster bootstrap over manifest
event groups:

```bash
mkdir -p reports/paper
sed -n '1,$p' reports/reasoning_floor/RUN/logic_only/evaluation_traces.jsonl \
  reports/reasoning_floor/RUN/local_graph/evaluation_traces.jsonl \
  > reports/paper/RUN_paired_traces.jsonl

UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m analyze_results \
  --traces reports/paper/RUN_paired_traces.jsonl \
  --baseline-bundle logic_only \
  --treatment-bundle local_graph \
  --comparison-id local_graph_vs_logic_only_accepted \
  --hypothesis-role primary \
  --metric accepted \
  --metric-type binary \
  --evaluation-subset main_score \
  --output reports/paper/local_evidence_effect.json
```

The report separates case-micro and event-macro effects, supplies a paired exact binary test, and computes population weighting only when every paired case has an explicit positive weight. Do not describe equal bucket averages as population weighted.

## Experiment Registry

Register every run, including failures and exploratory runs:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m experiment_registry \
  --run-summary reports/reasoning_floor/RUN/reasoning_floor_summary.json \
  --status exploratory \
  --notes "Development-contaminated core; not paper eligible"
```

`experiments/registry.json` is schema validated. Paper eligibility is derived from status and provenance rather than manually asserted. Use `superseded` plus `--supersedes` to retain history instead of deleting contradictory runs.

## Archived Reproduction Versus Live Rebuild

Archived reproduction uses the exact release-manifest files and verifies all hashes before evaluation. It must not
fetch live Wikidata data. A release is also bound to a validated snapshot manifest, including source identifiers and
retrieval timestamps.

For a clean clone, obtain large immutable files through the published, checksum-bound workflow in
[Artifact Acquisition](./Artifact_Acquisition.md). The current distribution manifest is an unresolved template and is
not a release.

A live rebuild intentionally reruns fetch and classification against a named new snapshot. It produces a new dataset version, new hashes, and new selection manifests; it is not a byte-for-byte reproduction of the archived release. Report both the snapshot date and the resulting release manifest.

## CI And Remaining Gates

GitHub Actions runs the full configured Ruff rule set, tests, documentation portability checks, schema checks through
the test suite, and the tracked sample release workflow. Line-length enforcement is explicitly excluded from the Ruff
policy; import, syntax, undefined-name, and unused-code checks have a clean repository-wide baseline.

Before paper submission, also complete the dataset card, license, citation metadata, ethics statement, limitations,
independent label/gold results, temporal leakage audit, multi-model or explicitly narrowed model scope, untouched-test
execution, and archival deposit.
