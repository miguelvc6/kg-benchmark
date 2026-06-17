# T-Box Taxonomy Patch Core Rerun Decision

Date: 2026-06-17

## Decision

`RUN_CORE_TBOX_TAXONOMY_PATCH`

The full core T-box taxonomy-patch run is approved. All Step 12 prerequisites have direct evidence, and no blocker remains before running all selected core T-box rows.

## Prerequisite Audit

| Requirement | Evidence | Decision |
| --- | --- | --- |
| Gold extraction coverage is 100% | `reports/gold/tbox_taxonomy_patch_gold_core_v1_summary.json`: `selected_tbox_records = 596`, `gold_extracted = 596`, `unsupported_count = 0` | Pass |
| Dev validation passed | `reports/analysis/tbox_taxonomy_patch_dev_validation.md` records Step 10 pass; request errors 0, T-box contract valid >= 95%, interpretable failure modes | Pass |
| Core dry run passed | `reports/analysis/tbox_taxonomy_patch_core_tbox64_dry_run.md` records Step 11 pass; 128/128 normalized, request errors 0, parse errors 0 | Pass |
| Evaluator metrics are interpretable | Dev and dry-run reports include numerator, denominator, applicability, family/decision/taxonomy/value-delta metrics | Pass |
| No prompt validity violations found | Dev and dry-run structured leakage scans found no hidden benchmark metadata in model-visible prompt text; only visible Wikidata label/description uses of `classification` occurred | Pass |
| Compute budget is acceptable | Core T-box gold count is 596; the approved run is 596 cases x 2 contexts = 1192 prompts, using the already validated local `gpt-oss:120b` Ollama setup | Pass |
| Old strict G3 artifacts are preserved | New outputs use a taxonomy-patch-specific prompt-dev directory and separate `t_box_taxonomy_patch_proposals.jsonl` files | Pass |

## Approved Full-Core Command

```bash
PROMPT_DEV_VERSION=prompt_dev_v5_tbox_taxonomy_patch \
OLLAMA_MODEL=gpt-oss:120b \
OLLAMA_TIMEOUT_SECONDS=300 \
OLLAMA_MAX_RETRIES=0 \
OLLAMA_TEMPERATURE=0 \
OLLAMA_CONTEXT_LENGTH=16384 \
OLLAMA_MAX_OUTPUT_TOKENS=2048 \
OLLAMA_KEEP_ALIVE=30m \
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/prompt_dev.py evaluate \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --dev-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --core-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --output-dir reports/prompt_dev/evaluation_prompt_dev_v5_tbox_taxonomy_patch_core_tbox_all_zero_shot \
  --model-endpoint ollama \
  --model gpt-oss:120b \
  --max-cases 596 \
  --track-filter T_BOX \
  --sample-strategy manifest_order \
  --representations hybrid_json_nl \
  --example-policies zero_shot \
  --context-bundles logic_only,local_graph \
  --tasks repair_proposal \
  --repair-track-modes oracle \
  --no-progress
```

## Guardrails

- Do not overwrite strict-signature G3 artifacts.
- Keep `prompt_dev_v5_tbox_taxonomy_patch` outputs in taxonomy-patch-specific directories.
- Keep strict `signature_after` reconstruction metrics as diagnostics only.
- Do not tune prompts from full-core results; use full-core results for reporting and governance checks.
