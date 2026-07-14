# Paper Readiness Checklist

This checklist separates repository implementation from paper execution. The repository is ready for execution only
when every item under **Implementation readiness** is complete and the final methodology lock verifies. Dataset
acquisition and model calls are deliberately listed later as operational execution, not as missing software.

## Implementation readiness

### 1. Methodology closure and freeze enforcement

- [x] Encode the methodology as one active freeze candidate under `paper/`.
- [x] Make all prompt regimes, context bundles, few-shot counts, audit sample sizes, failure policies, selection quotas,
  statistical contrasts, and inference settings explicit.
- [x] Pin the installed `gpt-oss:120b` artifact by its full Ollama digest.
- [x] Add strict JSON Schema and cross-file validation.
- [x] Add `kg-benchmark methodology check` with deterministic freeze-scope hashes and workload calculations.
- [x] Add `kg-benchmark methodology freeze` for the eventual immutable lock.
- [x] Prevent dataset acquisition/build and model execution without a valid frozen lock.

### 2. Canonical audit workflow

- [x] Generate the deterministic label-hidden 450-case construct sample through the public CLI.
- [x] Render canonical audit prompts and their render summary without relying on removed prompt-development systems.
- [x] Connect deterministic audit, Codex error discovery, and conservative finalization into one resumable workflow.
- [x] Write canonical `work/audit/dispositions.jsonl` and `work/audit/summary.json` artifacts directly.
- [x] Bind reviewer model, Codex CLI version, inputs, samples, schemas, reviews, and dispositions by hash.
- [x] Generate the final release-facing `audit.md` from the completed audit artifacts.

### 3. Reserve, prompt-quality gate, and selection finalization

- [x] Implement `select reserve` with quotas 276 IC-L, 450 IC-G, 354 IC-E-elim, and up to 360 T-box.
- [x] Accept and hash a previously-used event-group exclusion artifact.
- [x] Render and deterministically scan every reserve prompt.
- [x] Require the fixed seed-13 50-case temporal prompt review.
- [x] Replace failed prompts with the next eligible case in the frozen per-stratum order.
- [x] Implement `select finalize`; only prompt-clean final disposition `include` cases may enter evaluation.
- [x] Target the T-box main composition of 130 relaxation expansions, 50 restriction contractions, and 120 schema
  updates, subject to the declared eligible-prefix rule.
- [x] Bind dataset, audit, exclusions, ranking, support bank, eligibility order, prompt audit, and per-case eligibility
  hashes in selection manifests.
- [x] Apply the declared T-box deficit redistribution rule to initial and extension populations.

### 4. Final dataset promotion gates

- [x] Require a valid methodology lock and matching protocol hash.
- [x] Require exact Stage 2/3/4 identity, complete source provenance, and passing lineage validation.
- [x] Validate every published record against its active schema.
- [x] Require complete unique dispositions and zero unresolved systemic findings.
- [x] Verify independent groups, prior-group and support exclusions, reserve finalization, and prompt QA.
- [x] Require exactly 1,200 main cases and exactly 600 Azure cases nested within the main population.
- [x] Reproduce the dataset manifest byte-for-byte before promotion succeeds.

### 5. Experiment-matrix orchestration

- [x] Read `paper/models.json` and materialize models × tasks × prompt regimes × contexts × populations.
- [x] Add a no-call dry run reporting new requests, cache hits, missing revisions, and expected workload.
- [x] Add resumable per-cell execution manifests and matrix completeness checks.
- [x] Enforce the configured Ollama inference parameters and Azure sequential/no-tools policy.
- [x] Prove that population extensions schedule only request keys absent from the generation cache.

### 6. Paper-level analysis and result packaging

- [x] Aggregate immutable run/evaluation manifests across models and conditions without provider calls.
- [x] Implement case-micro and event-cluster macro estimates.
- [x] Implement the 5,000-sample seed-13 percentile cluster bootstrap.
- [x] Implement exact McNemar tests and Holm adjustment for the four predeclared contrasts.
- [x] Keep A-box, T-box, diagnosis, models, confirmatory results, Azure calibration, and extensions separate.
- [x] Generate compact paper tables, machine-readable summaries, and provenance manifests under `results/`.

### 7. Acceptance and reproduction

- [x] Add one end-to-end synthetic test covering build, lineage, audit, reserve, prompt rejection/replacement,
  finalization, promotion, matrix planning, cached execution, rescoring, and paper analysis.
- [x] Add a clean-clone reproduction smoke test.
- [x] Replace remaining workflow ellipses with executable commands after their public interfaces exist.
- [x] Run the complete test/schema/documentation suite and verify byte-reproducible manifests.

### 8. Final methodology freeze gate

Complete this gate only after sections 2–7 are finished; otherwise later freeze-scoped implementation changes would
immediately invalidate the lock.

- [x] Resolve and record the full `qwen3:30b` Ollama digest.
- [x] Resolve and record the full `llama3.3:70b` Ollama digest.
- [x] Resolve and record the immutable Azure `gpt-5.6-sol-2026-07-09` snapshot revision.
- [x] Change all methodology component statuses to `frozen`, commit a clean source revision, create
  `paper/methodology.lock.json`, commit the lock alone, verify `freeze_ready: true`, and tag the freeze.

## Operational paper execution

Run these only after implementation readiness is complete. The long-running dataset commands should be executed by the
repository operator, not implicitly by an implementation agent.

- [ ] Acquire the single final Wikidata dataset into ignored `work/`.
- [ ] Build, audit, review, reserve, finalize, and promote the final dataset.
- [ ] Run deterministic non-LLM baselines.
- [ ] Execute the complete Ollama matrix over the final 1,200 cases.
- [ ] Execute the sequential Azure matrix over the nested 600 cases.
- [ ] Replay metrics, run the predeclared analysis, and generate compact paper results.
- [ ] Transfer the historical development artifacts to the checksum-indexed external archive and update
  `paper/development-archive.json`; only then may the preserved local bulk be deleted.
- [ ] Publish the final dataset, manifest checksum, code revision, freeze tag, and result artifacts.

## Current methodology check

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-benchmark methodology check
```

The command must report `valid: true`. Before acquisition it must additionally report `freeze_ready: true`; unresolved
model revisions, a dirty worktree, a non-frozen status, or a missing/mismatched lock are hard blockers.
