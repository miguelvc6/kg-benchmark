# Prompt Dev v4 Spec-Only Holdout Report

Verdict: **BLOCKED**

The `prompt_dev_v4_spec_only` candidate is renderable, spec-only on prompt wording, and clean on the leakage scan. The holdout inference run is not complete: the university endpoint path completed only 31 of 384 prompt rows before I stopped it with SIGTERM to avoid leaving a long-running process active.

This is an operational block, not evidence that v4 proposal quality is good or bad. The partial proposal sample is too small and too biased toward the first manifest rows to support a quality conclusion.

## What Changed

- Created `reports/prompt_dev/prompt_validity_charter.md`.
- Added `prompt_dev_v4_spec_only` as the default Phase F prompt candidate in `scripts/prompt_dev_templates.py`.
- Preserved v3 as `prompt_dev_v3_scaffolded`, selectable with `PROMPT_DEV_VERSION=prompt_dev_v3_scaffolded`.
- Mirrored only spec-level wording into Phase G prompts in `src/guardian/prompts.py`.
- Removed prompt-visible T-box temporal-policy implementation labels from `src/guardian/reasoning.py`; the policy remains in internal audit metadata.
- Documented the v3/v4 split and model-visible temporal policy boundary in `docs-technical/Prompt_Development.md` and `docs-technical/Reasoning_Floor.md`.
- Created disjoint holdout manifest `reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json`.

## Holdout Composition

| Field | Count |
| --- | ---: |
| Cases | 96 |
| Rendered prompts | 384 |
| A_BOX cases | 48 |
| T_BOX cases | 48 |
| TypeA cases | 24 |
| TypeB cases | 12 |
| TypeC cases | 12 |
| Unique focus QIDs | 96 |
| Unique properties | 74 |
| Overlap with v3 96-case run | 0 |

The holdout preserves A_BOX/T_BOX balance and excludes the 96 cases used in `reports/prompt_dev/evaluation_prompt_dev_v3_96_diverse_zero_shot/`.

## Gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Render gate | PASS | 384 prompts rendered over 96 cases |
| Leakage gate | PASS | no raw `repair_`/`reform_` IDs, no `sitelinks_count` |
| Spec-only wording gate | PASS | no `TypeA`, `TypeB`, `TypeC`, `targeted REMOVE`, or action-decision-tree wording in rendered prompts |
| Temporal-label gate | PASS | no model-visible `compact_inventory_no_pre_change_signature` or `pre_change_signature_before` labels after rerender |
| Parse gate | PARTIAL PASS | 0 parse errors in 31 completed rows |
| Request stability gate | FAIL | 4 request errors in 31 completed rows; 4 request errors in 15 completed proposal rows |
| Completion gate | FAIL | run stopped at 31/384 rows due endpoint throughput |
| Test gate | PASS | 106 tests passed |

## Partial Inference Snapshot

| Matrix | Completed | Main partial signal |
| --- | ---: | --- |
| logic_only track diagnosis | 8 | track diagnosis accuracy 0.500 |
| logic_only repair proposal | 8 | accepted 0.125, request error 0.125 |
| local_graph track diagnosis | 8 | track diagnosis accuracy 0.625 |
| local_graph repair proposal | 7 | accepted 0.000, request error 0.429 |

Overall partial status:

- Completed rows: 31/384.
- Normalized rows: 27.
- Parse errors: 0.
- Request errors: 4.
- Observed request error rate: 12.9%.
- Observed repair-proposal request error rate: 26.7%.
- Repair-proposal call latency averaged about 384-387 seconds in the partial rows, with max single calls around 633 seconds.

These rates fail the Phase F stability gate. They should not be interpreted as a full v4 quality result.

## Answerability-Aware Rates

Not computed for v4. The run completed too few proposal rows to make answerability-aware exact/accepted rates meaningful. Running the answerability audit on this partial result would mainly measure endpoint incompletion and manifest order, not prompt behavior.

## v3 Scaffolded Comparison

Not run. The optional v3 scaffolded holdout ablation was skipped because the main v4 endpoint execution did not complete. v3 remains a scaffolded diagnostic ablation only and should not be promoted to Phase G main solely on dev or holdout score.

## Commands Run

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/prompt_dev.py evaluate \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --dev-manifest reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json \
  --core-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --output-dir reports/prompt_dev/evaluation_prompt_dev_v4_spec_only_holdout96_zero_shot \
  --model-endpoint university \
  --max-cases 96 \
  --sample-strategy manifest_order \
  --representations hybrid_json_nl \
  --example-policies zero_shot \
  --context-bundles logic_only,local_graph \
  --tasks track_diagnosis,repair_proposal \
  --repair-track-modes oracle \
  --no-progress
```

Stopped after 31/384 rows because the endpoint path was too slow for a complete synchronous run.

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/prompt_dev.py render \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --dev-manifest reports/benchmark_selection/dev_prompt_holdout_spec_v4_96_seed_17.json \
  --core-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --output-dir reports/prompt_dev/evaluation_prompt_dev_v4_spec_only_holdout96_zero_shot/rendered_prompts \
  --max-cases 96 \
  --sample-strategy manifest_order \
  --representations hybrid_json_nl \
  --example-policies zero_shot \
  --context-bundles logic_only,local_graph \
  --tasks track_diagnosis,repair_proposal \
  --repair-track-modes oracle
```

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run --extra dev python -m pytest \
  tests/test_prompt_dev.py \
  tests/test_model_provider.py \
  tests/test_reasoning_floor.py \
  tests/test_track_parser.py \
  tests/test_tbox_parser.py \
  tests/test_patch_parser.py
```

Result: 106 passed.

## Recommendation

Do not move to Phase G. The correct next step is to resolve endpoint throughput/request stability or add a controlled execution mode that does not alter prompt content. After that, rerun the full v4 spec-only holdout before deciding whether to freeze it for a Phase G dry run.

Final verdict option: **BLOCKED**
