# T-Box Taxonomy Patch Dev Validation

**Run date:** 2026-06-17  
**Prompt version:** `prompt_dev_v5_tbox_taxonomy_patch`  
**Model endpoint:** `ollama`  
**Model:** `gpt-oss:120b`  
**Output directory:** `reports/prompt_dev/evaluation_prompt_dev_v5_tbox_taxonomy_patch_holdout96_zero_shot/`

## Command

```bash
PROMPT_DEV_VERSION=prompt_dev_v5_tbox_taxonomy_patch \
OLLAMA_MODEL=gpt-oss:120b \
OLLAMA_TIMEOUT_SECONDS=300 \
OLLAMA_MAX_RETRIES=0 \
OLLAMA_TEMPERATURE=0 \
OLLAMA_CONTEXT_LENGTH=8192 \
OLLAMA_MAX_OUTPUT_TOKENS=2048 \
OLLAMA_KEEP_ALIVE=30m \
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/prompt_dev.py evaluate \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --dev-manifest reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json \
  --core-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --output-dir reports/prompt_dev/evaluation_prompt_dev_v5_tbox_taxonomy_patch_holdout96_zero_shot \
  --model-endpoint ollama \
  --model gpt-oss:120b \
  --max-cases 96 \
  --sample-strategy manifest_order \
  --representations hybrid_json_nl \
  --example-policies zero_shot \
  --context-bundles logic_only,local_graph \
  --tasks repair_proposal \
  --repair-track-modes oracle \
  --no-progress
```

## Gate Summary

- Dev evaluation completed: yes, 192 prompts evaluated.
- Full core inference run: no.
- Proposal request error rate: 0.0 in both matrices.
- Proposal parse errors: 1/192 total. The only parser failure is one T-box `NO_CAUSAL_SCHEMA_REPAIR` response with `target.constraint_type_qid = null`, which is invalid unless the decision is `UNCLEAR_SCHEMA_EVIDENCE`.
- Prompt leakage: no hidden benchmark metadata was found in model-visible system/user prompt text. Literal `classification` occurrences are ordinary visible entity/property descriptions, not benchmark `classification` fields.
- A-box prompt: unchanged v4 prompt. Five A-box responses were valid JSON with explicit `ops: []`; these are recorded as `non_executable_empty_ops` rather than parser failures and are not converted into fabricated proposal operations.
- `OTHER_TBOX_UPDATE` prediction rate: 0 in both matrices.

## T-Box Taxonomy Metrics

| Context | T-box parse rate | Contract-valid rate | Family-level success | Schema-decision match | Taxonomy-level success | Value-delta F1 when applicable |
|---|---:|---:|---:|---:|---:|---:|
| `logic_only` | 47/48 = 0.979 | 47/48 = 0.979 | 29/47 = 0.617 | 17/47 = 0.362 | 6/47 = 0.128 | 0.123, coverage 33/48 |
| `local_graph` | 48/48 = 1.000 | 48/48 = 1.000 | 31/48 = 0.646 | 17/48 = 0.354 | 4/48 = 0.083 | 0.058, coverage 33/48 |

## Operation Distribution Review

Gold dev T-box repairs contain mostly qualifier updates: 20 qualifier adds, 14 qualifier removes, 3 qualifier replaces, and 1 `OTHER_TBOX_UPDATE`.

Predicted operations did not collapse into `OTHER_TBOX_UPDATE`:

- `logic_only`: `CONSTRAINT_QUALIFIER_ADD` 7, `EXCEPTION_ADD` 6, `CONSTRAINT_REMOVE` 3, `CONSTRAINT_ADD` 3, `CONSTRAINT_DEPRECATE` 2.
- `local_graph`: `CONSTRAINT_QUALIFIER_ADD` 11, `CONSTRAINT_DEPRECATE` 7, `CONSTRAINT_REMOVE` 3, `EXCEPTION_ADD` 1.

The dominant predicted operation is qualifier add, which is consistent with the largest gold operation family. The model also overuses exception/deprecate operations relative to gold; this is an interpretable taxonomy-confusion failure mode, not a parser or contract failure.

## Decision

Step 10 dev validation passes the implementation-plan gates for continuing to a bounded core T-box dry run. The next run must remain bounded and must not overwrite strict-signature G3 artifacts.
