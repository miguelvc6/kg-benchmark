# Confirmatory Release Runbook

This runbook implements the quality-gated, post-freeze 1,200-case release. Bulk data and private selections remain outside
Git. Run repository commands from WSL with `UV_PROJECT_ENVIRONMENT=.venv-wsl`.

## 1. Validate the restored baseline

Do not rewrite the recovered files. The lineage command streams both Stage 2 representations, compares canonical record
digests and order, validates Stage 0 popularity and Stage 1 candidate provenance, and requires exact Stage 2/3/4 IDs plus
field equality between Stage 2 and the lean Stage 4 projection.

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-artifact-lineage validate \
  --stage0 data/00_entity_popularity.json \
  --stage1 data/01_repair_candidates.json \
  --stage2-json data/02_wikidata_repairs.json \
  --stage2-jsonl data/02_wikidata_repairs.jsonl \
  --stage3 data/03_world_state.json \
  --stage4 data/04_classified_benchmark.jsonl \
  --output reports/lineage/restored_v2.json
```

The command exits nonzero on duplicate or missing IDs, reordered rows, canonical representation differences, missing
provenance, identity gaps, or projection mutation. The v2 manifest records file sizes, SHA-256 hashes, counts, provenance,
Git revision, and every validation result. `kg-automated-audit run --stage2 ...` now repeats the Stage 2/3/4 content gate;
it no longer records Stage 2 presence as if that were validation.

## 2. Freeze and acquire an isolated snapshot

The model/population portion of the methodology freeze is defined by
[Model Execution Matrix](./Model_Execution.md) and `experiments/paper_execution_models_v1.json`. Validate it before the
allocation freeze. The current matrix remains a draft until prompt configuration is decided, selection hashes exist, and
all four immutable model/deployment revisions are bound; do not acquire the post-freeze snapshot while those fields are
unresolved.

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-experiment-plan validate
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-experiment-plan validate --require-frozen
```

The second command is the confirmatory gate and must pass immediately before the freeze commit.

Create and commit the allocation freeze before mining. Use an unused snapshot directory and an isolated cache. Candidate
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

## 5. Execution accounting and recovery

Each Ollama model makes 2,400 calls for two bundles across 1,200 cases; both routing modes require 4,800. An external API
reference model makes 1,200 calls for two bundles across the nested 600, or 2,400 with both routing modes. Resume only from
hash-matching manifests and model identities. A shortage, lineage failure, incomplete disposition partition, repeated or
previously used event, prompt leak, dirty freeze, or changed model identifier fails the release rather than triggering an
ad hoc fallback.

Materialize the frozen execution plan with `kg-experiment-plan plan`. Each model uses a persistent, model-specific
generation cache, so a later population extension submits only new prompt identities. Azure runs are batch-only with high
reasoning effort and tool calls disabled. Updated metrics use `kg-rescore-run` against stored generations and do not issue
provider requests.
