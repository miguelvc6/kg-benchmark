# Confirmatory Release Runbook

This runbook implements the quality-gated, post-freeze 1,200-case release. Bulk data and private selections remain outside
Git. Run repository commands from WSL with `UV_PROJECT_ENVIRONMENT=.venv-wsl`.

## 1. Validate the restored baseline

Do not rewrite the recovered files. The compiled JSON array is the authoritative Stage 2 consumed by Stages 3 and 4; the
larger JSONL is a recovered pre-popularity precursor. The lineage command proves the checksum-bound declared relationship:
every compiled row must be an ordered precursor row with only `popularity` added, while precursor-only rows and the known
multi-value physical line remain recorded. It validates Stage 0 popularity payload equality and Stage 1 event provenance
against authoritative Stage 2 and requires
exact Stage 2/3/4 IDs plus field equality between Stage 2 and the lean Stage 4 projection.

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-artifact-lineage validate \
  --stage0 data/00_entity_popularity.json \
  --stage1 data/01_repair_candidates.json \
  --stage2-json data/02_wikidata_repairs.json \
  --stage2-jsonl data/02_wikidata_repairs.jsonl \
  --stage3 data/03_world_state.json \
  --stage4 data/04_classified_benchmark.jsonl \
  --source-provenance release/restored_source_provenance_v1.json \
  --reconciliation-policy release/restored_stage2_reconciliation_v1.json \
  --output reports/lineage/restored_v3.json
```

The command exits nonzero on any undeclared difference, checksum mismatch, duplicate or missing ID, reordered successor,
missing provenance, identity gap, or projection mutation. The v3 manifest records file sizes, SHA-256 hashes, counts,
provenance, Git revision, strict-equivalence diagnostics, and the authoritative reconciliation result.
`kg-automated-audit run --stage2 ...` now repeats the Stage 2/3/4 content gate;
it no longer records Stage 2 presence as if that were validation.

If only a lineage validator rule is strengthened after an exhaustive pass, `kg-artifact-lineage refresh-provenance`
may recompute Stage 0/1 provenance while retaining the prior relationship and identity results. It first rehashes every
Stage 0--4 artifact and refuses changed bytes, failed prior subchecks, or an invalid prior manifest. The new manifest
records exactly which checks were reused and recomputed; this is not an artifact-repair mechanism.

## 2. Freeze and acquire an isolated snapshot

The first freeze boundary is the pre-acquisition methodology freeze. It is defined by
[Model Execution Matrix](./Model_Execution.md) and `experiments/paper_execution_models_v1.json`. Validate it before the
snapshot is acquired. It binds prompts, schemas, classifier/audit/selection/evaluation code, acquisition and allocation
policies, model identifiers and inference settings, and `protocols/analysis_plan_v1.md`. It intentionally does not bind a
dataset release, final selections, or deployment digests that cannot exist yet.

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-experiment-plan validate
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-paper-prompt-profile
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-experiment-plan validate --require-methodology-frozen
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-protocol-freeze verify \
  --manifest protocols/methodology_v1.json --protocol-root .
```

Both methodology checks must pass before acquisition.

Use an unused snapshot directory and an isolated cache. Candidate
refresh is explicit; resume statistics and checkpoints remain supported.

```bash
SNAPSHOT_ID=post_freeze_YYYYMMDD
SNAPSHOT_DIR="data_post_freeze/${SNAPSHOT_ID}"

UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-fetcher \
  --data-dir "${SNAPSHOT_DIR}" \
  --cache-dir "${SNAPSHOT_DIR}/cache" \
  --dump-path "${SNAPSHOT_DIR}/latest-all.json.gz" \
  --refresh-candidates
```

Acquisition is fail-closed. Exhausted report, history, historical-snapshot, or pageview retries must terminate the run;
they must never be interpreted as an empty report, missing claims, no repair, or zero popularity. Resume Stage 2 from its
last snapshot-local run checkpoint after the upstream service recovers. Successful pageview responses are persisted in
batches so Stage 0 can resume without repeating them. A gzip/JSON/CRC error while scanning the dump invalidates the whole
Stage 3 build; partial world state is never selection eligible.

The methodology freeze must bind the complete transitive acquisition and evaluation implementation, not only top-level
entry points. The current freeze enumerates all repository Python modules under `src/`, all JSON schemas, `pyproject.toml`,
and `uv.lock`; adding or changing any of them requires a new pre-acquisition freeze.

The dump is acquired separately, checksummed before use, and never placed in the restored `data/` directory. Classify into
`04_classified_benchmark.jsonl`, then build a v2 snapshot manifest with `--stage0`, `--stage1`, `--stage2`, `--dump-path`,
`--world-state`, `--stage4`, `--freeze-manifest`, `--configuration`, and `--cache-provenance`. Source entries should include
retrieval timestamps and upstream checksums using `NAME=SOURCE_ID=RETRIEVED_AT_UTC=SHA256`. Missing any v2 binding fails
manifest construction.

If a whole-data audit exposes a systemic implementation defect, discard this snapshot for confirmatory use, fix the defect,
create a clean new freeze, and acquire a new snapshot ID.

## 3. Exhaustive audit and dispositions

Run Stage 2/3/4 lineage, Stage 4 schema/classification/evidence checks, and prompt rendering over the complete candidate
population. The deterministic audit flags format-rule contradictions, cardinality conflicts, unsupported local-evidence
claims, T-box history gaps, and report/constraint disagreements. Missing T-box `signature_before` is made explicit and no
later constraint inventory is substituted. Prompt scanning detects exact, embedded, normalized, URL-decoded, HTML-decoded,
hex, and base64 forms. Label-hidden Codex review is error discovery only.

Final disposition JSONL is a v2 complete, unique partition. Only `include` is selection eligible. Any recurring AI finding
must first become a deterministic regression rule; rerender and rerun until no systemic finding remains.

## 4. Reserve and final selection

Reserve allocation requires the v2 snapshot, passing dataset and temporal audits, complete final dispositions, and all
prior selection manifests:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-select-untouched-test reserve \
  --classified-benchmark "${SNAPSHOT_DIR}/04_classified_benchmark.jsonl" \
  --dispositions reports/automated_audit/POST_FREEZE/audit_dispositions.jsonl \
  --dataset-audit reports/automated_audit/POST_FREEZE/deterministic_summary.json \
  --temporal-audit reports/automated_audit/POST_FREEZE/temporal_audit.json \
  --exclude-manifest reports/benchmark_selection/dev_prompt_v1_seed_13.json \
  --exclude-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --snapshot-manifest "${SNAPSHOT_DIR}/snapshot_manifest.json" \
  --seed 13 \
  --private-output sealed/reserve_v3.json \
  --public-output releases/reserve_v3_public.json
```

The reserve is capped at 276 TypeA, 450 TypeB, 354 TypeC, and 360 T-box independent events. T-box reserve eligibility
also requires complete mechanically supported `tbox_taxonomy_patch_v1` gold; cases requiring unmined class-hierarchy or
exception operations are excluded before ranking. Render every reserve prompt,
run the deterministic scan, and complete the fixed 50-case temporal error-discovery sample. Record rejected cases in the
prompt-review artifact and finalize without changing reserve order:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-select-untouched-test finalize \
  --reserve-manifest sealed/reserve_v3.json \
  --temporal-audit reports/reserve_prompt_audit/temporal_audit.json \
  --prompt-review reports/reserve_prompt_audit/review.json \
  --seed 13 \
  --private-output sealed/final_1200_v3.json \
  --public-output releases/final_1200_v3_public.json
```

Finalization removes prompt failures, transfers any T-box deficit with largest-remainder rounding, and fails unless exactly
1,200 unique events remain. It also writes the deterministic nested 600-case API subset. Private manifests bind snapshot,
audit, exclusion, ranking, prompt-audit, and per-case eligibility hashes; public manifests expose commitments and aggregates
without case IDs.

After final selection, build the evaluation release, bind its selection hashes and immutable deployment revisions, change
the execution matrix status to `frozen`, and create the execution protocol. `kg-experiment-plan validate --require-frozen`
and the execution-protocol verifier are the second freeze boundary and must pass before any model request. The intermediate
dataset/allocation binding protects private selection but does not replace either freeze boundary.

## 5. Execution accounting and recovery

Each Ollama model makes 2,400 oracle-routed calls for two bundles across 1,200 cases. The external API reference model
makes 1,200 oracle-routed calls for two bundles across the nested 600. Few-shot or diagnosis-routed extensions require a
separate frozen matrix expansion and reuse the existing generation cache rather than changing these headline counts.
Resume only from
hash-matching manifests and model identities. A shortage, lineage failure, incomplete disposition partition, repeated or
previously used event, prompt leak, dirty freeze, or changed model identifier fails the release rather than triggering an
ad hoc fallback.

Materialize the frozen execution plan with `kg-experiment-plan plan`. Each model uses a persistent, model-specific
generation cache, so a later population extension submits only new prompt identities. Azure runs are batch-only with high
reasoning effort and tool calls disabled. Updated metrics use `kg-rescore-run` against stored generations and do not issue
provider requests.
