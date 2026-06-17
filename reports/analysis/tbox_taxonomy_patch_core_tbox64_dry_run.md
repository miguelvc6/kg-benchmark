# T-Box Taxonomy Patch Core T-Box-Only Dry Run

Date: 2026-06-17

## Scope

This is the Step 11 bounded core dry run for `prompt_dev_v5_tbox_taxonomy_patch`.

- Manifest: `reports/benchmark_selection/core_v1_seed_13.json`
- Track filter: `T_BOX`
- Max cases: 64
- Context bundles: `logic_only`, `local_graph`
- Example policy: `zero_shot`
- Repair track mode: `oracle`
- Model endpoint/model: `ollama` / `gpt-oss:120b`
- Output directory: `reports/prompt_dev/evaluation_prompt_dev_v5_tbox_taxonomy_patch_core_tbox64_zero_shot/`

The final clean run used `OLLAMA_CONTEXT_LENGTH=16384`. An earlier 8192-context attempt produced four local-graph length failures; no prompt text was changed before rerunning cleanly with the larger context window.

## Command

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
  --output-dir reports/prompt_dev/evaluation_prompt_dev_v5_tbox_taxonomy_patch_core_tbox64_zero_shot \
  --model-endpoint ollama \
  --model gpt-oss:120b \
  --max-cases 64 \
  --track-filter T_BOX \
  --sample-strategy manifest_order \
  --representations hybrid_json_nl \
  --example-policies zero_shot \
  --context-bundles logic_only,local_graph \
  --tasks repair_proposal \
  --repair-track-modes oracle \
  --no-progress
```

## Gate Results

Step 11 passes.

| Gate | Result |
| --- | --- |
| Core T-box-only run completes | Pass: 128 prompts evaluated |
| New T-box metrics computed | Pass: `tbox_taxonomy_patch_evaluation_summary.json` exists for both matrices |
| Does not overwrite old G3 artifacts | Pass: new Step11-specific output directory |
| Request errors below threshold | Pass: 0/128 |
| Parse errors below threshold | Pass: 0/128 |
| No prompt leakage | Pass: no hidden benchmark metadata terms in model-visible prompt text |
| Summary distinguishes strict diagnostics from taxonomy headlines | Pass: comparison markdown has `Strict functional` / `Strict audit` and separate `T-box family`, `T-box decision`, `T-box taxonomy`, `T-box value F1` columns |

## Headline Metrics

The strict `functional` and `audit` columns in `prompt_dev_evaluation_comparison.md` are legacy strict-signature diagnostics. The taxonomy-patch headline metrics are the T-box columns and the `tbox_patch_*` metrics below.

| Context | Parse valid | Request errors | Family success | Decision success | Taxonomy success | Value-delta F1 when applicable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `logic_only` | 64/64 = 1.000 | 0/64 = 0.000 | 44/64 = 0.688 | 24/64 = 0.375 | 11/64 = 0.172 | 7/251 = 0.056 |
| `local_graph` | 64/64 = 1.000 | 0/64 = 0.000 | 49/64 = 0.766 | 30/64 = 0.469 | 15/64 = 0.234 | 9/133 = 0.135 |

## Leakage Scan

The structured scan checked only `system_prompt` and `user_prompt` fields in `prompt_dev_rendered_prompts.jsonl`.

- Hidden benchmark fields/labels absent: `repair_target`, `popularity`, `sitelinks_count`, `changed_constraint_types`, `target_constraint_is_changed`, `TypeA`, `TypeB`, `TypeC`, `truth_source`, `truth_tokens`, `selected_case_ids`, `case_annotations`, `selection_stratum`, `group_key`, `DEV_`, `CORE_`, `historical_track`.
- `classification` appears twice only as visible Wikidata text for `competition class` / `classification in sports` in `case_000038`.

## Artifact Notes

- Each matrix has 64 raw model responses and 64 T-box taxonomy patch proposal rows.
- `run_manifest.jsonl` has 128 rows per matrix because prompt-dev also records 64 expected-but-missing `track_diagnosis` rows for this repair-only matrix; these are not request or parse errors and are not counted in the proposal metrics.
- No core-result-driven prompt tuning was performed.
