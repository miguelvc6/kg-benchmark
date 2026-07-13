# Research Release Protocol

This document is the engineering runbook for converting the current exploratory repository state into a reproducible paper artifact. Research rationale remains in `docs-conceptual/`; this file covers executable controls.

## Current Status

The existing core and its model outputs are development-contaminated and remain exploratory. The tools below repair measurement and governance contracts, but they do not retroactively make old results confirmatory. Independent annotation, a post-freeze data snapshot, untouched-test execution, and final release materials still require human execution.

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

Build a release candidate only after the selection manifest and selected Stage 4 rows pass recomputed caps, group isolation, subset partition, and schema checks:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m artifact_release \
  --selection-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --artifact data/03_world_state.json \
  --output releases/candidate_manifest.json
```

The command writes file sizes and SHA-256 digests and refuses a failing candidate.

## Run Provenance

New reasoning-floor runs store in `run_config.json` and the final summary:

- Git commit and dirty state;
- SHA-256, size, and absolute path for Stage 4, world state, selection manifest, and output schemas;
- resolved context length, output limit, temperature, top-p, seed, retries, and reasoning effort;
- model digest when provided by the provider or `OLLAMA_MODEL_DIGEST`/`MODEL_DIGEST`;
- prompt/task versions, routing mode, selected IDs, and visible-ID mapping.

A missing model digest or dirty worktree makes a run ineligible for confirmatory registration.

## Independent Annotation

Create blind, two-reviewer assignments from the stratified audit sample:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m annotation_protocol \
  --audit-csv reports/manual_audit/audit_phase_d_v1_seed_13.csv \
  --output-dir reports/manual_audit/double_review_v1 \
  --annotators reviewer_a,reviewer_b,reviewer_c
```

The builder removes raw IDs, classification, confidence, track, strata, information type, and build metadata from evidence cards. It writes the re-identification map separately. Every case receives two distinct reviewers, and disagreements require adjudication. Access controls for the private map and adjudicator identity are operational responsibilities outside the repository.

T-box taxonomy gold still requires independent property/revision-level validation; extractor output alone is not independent gold.

## Untouched Test

Freeze prompts, models, metrics, analysis code, and protocol before creating a test manifest. Use a post-freeze benchmark snapshot; code-level exclusions on the existing snapshot are not sufficient to establish untouchedness.

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m select_untouched_test \
  --classified-benchmark data_post_freeze/04_classified_benchmark.jsonl \
  --exclude-manifest reports/benchmark_selection/dev_prompt_v1_seed_13.json \
  --exclude-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --target-size 1200 \
  --property-holdout \
  --output sealed/untouched_test_v1.json
```

The selector excludes prior case, event-group, and optionally property identities; weak group keys are ineligible. Treat the output as sealed until the protocol is registered.

No external API is required by these governance tools. Model execution may use local Ollama. Any API reference condition, if retained in the research scope, must be separately approved and recorded.

## Statistical Analysis

Compute a paired cluster bootstrap over manifest event groups:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python -m analyze_results \
  --traces reports/reasoning_floor/RUN/evaluation_traces.jsonl \
  --baseline-bundle logic_only \
  --treatment-bundle local_graph \
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
  --protocol-id preregistration_v1 \
  --notes "Development-contaminated core; not paper eligible"
```

`experiments/registry.json` is schema validated. Paper eligibility is derived from status and provenance rather than manually asserted. Use `superseded` plus `--supersedes` to retain history instead of deleting contradictory runs.

## Archived Reproduction Versus Live Rebuild

Archived reproduction uses the exact release-manifest files and verifies all hashes before evaluation. It must not fetch live Wikidata data.

A live rebuild intentionally reruns fetch and classification against a named new snapshot. It produces a new dataset version, new hashes, and new selection manifests; it is not a byte-for-byte reproduction of the archived release. Report both the snapshot date and the resulting release manifest.

## CI And Remaining Gates

GitHub Actions runs tests, critical syntax/name lint checks, schema checks through the test suite, and the tracked sample workflow. Full style lint remains a known cleanup item because of pre-existing line-length debt.

Before paper submission, also complete the dataset card, license, citation metadata, ethics statement, limitations, independent label/gold results, temporal leakage audit, multi-model or explicitly narrowed model scope, untouched-test execution, and archival deposit.
