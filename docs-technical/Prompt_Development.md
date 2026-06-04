# Phase F Prompt Development

Phase F prepares prompt-development artifacts on the dev manifest only. It does not run LLM inference by default.

## Prompt Template Script

Prompt text and representation renderers live in:

- `scripts/prompt_dev_templates.py`

This script defines the Phase F prompt version, supported representations, task contracts, optional abstention contract, and prompt rendering function. Keep prompt wording changes there so templates are easy to review before any main-core run.

Supported representations:

- `hybrid_json_nl`
- `pure_nl`
- `compact_table`
- `turtle`

Supported example policies:

- `zero_shot`
- `random_same_task_2shot`
- `same_track_2shot`
- `matched_2shot`

## CLI

The entry point is:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-prompt-dev --help
```

The equivalent source invocation is:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/prompt_dev.py --help
```

## Build The Matrix

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-prompt-dev matrix \
  --output reports/prompt_dev/prompt_dev_matrix_v1.json
```

This writes:

- `reports/prompt_dev/prompt_dev_matrix_v1.json`
- `reports/prompt_dev/prompt_dev_matrix_v1.md`

The matrix records representation, example policy, context bundle, task, track mode, abstention mode, and planned metrics. It is a design artifact only; it does not contact a model provider.

## Render Prompts For Review

Render a small review pack over the dev manifest:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-prompt-dev render \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --dev-manifest reports/benchmark_selection/dev_prompt_v1_seed_13.json \
  --core-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --output-dir reports/prompt_dev/rendered_prompt_dev_v1 \
  --max-cases 24 \
  --representations hybrid_json_nl \
  --example-policies zero_shot \
  --context-bundles logic_only,local_graph \
  --tasks track_diagnosis,repair_proposal \
  --repair-track-modes oracle
```

This writes:

- `prompt_dev_rendered_prompts.jsonl`
- `prompt_dev_render_summary.json`
- `prompt_dev_prompt_review.md`

The render command opens the Stage 4 artifact and world-state index, builds the same sanitized context bundles used by the reasoning-floor runner, selects dev-only few-shot examples when requested, and writes prompts for manual review. It does not run LLM inference.

## Evaluate Prompt Variants On The Dev Manifest

After static review, run selected prompt variants on the dev manifest only:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-prompt-dev evaluate \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --dev-manifest reports/benchmark_selection/dev_prompt_v1_seed_13.json \
  --core-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --output-dir reports/prompt_dev/evaluation_prompt_dev_v1 \
  --model-endpoint ollama \
  --max-cases 24 \
  --representations hybrid_json_nl,pure_nl \
  --example-policies zero_shot,matched_2shot \
  --context-bundles logic_only,local_graph \
  --tasks track_diagnosis,repair_proposal \
  --repair-track-modes oracle
```

`evaluate` runs LLM inference. Keep it restricted to the dev manifest and do not use it on the frozen core selection.

The command writes:

- `rendered_prompts/` with the prompt pack used for the run
- `matrices/<matrix_id>/raw_model_responses.jsonl`
- `matrices/<matrix_id>/run_manifest.jsonl`
- `matrices/<matrix_id>/a_box_proposals.jsonl`, `t_box_proposals.jsonl`, or `track_diagnoses.jsonl`
- `matrices/<matrix_id>/evaluation_traces.jsonl`
- `matrices/<matrix_id>/evaluation_summary.json`
- `prompt_dev_evaluation_summary.json`
- `prompt_dev_evaluation_comparison.md`

Each matrix row is scored in its own directory so prompt variants for the same case cannot overwrite each other.

## Few-Shot Leakage Controls

The example selector excludes:

- same case id;
- same focus QID;
- same T-box property-revision group;
- same property by default;
- cases or groups listed in the optional core manifest.

Matched examples rank candidates by:

- same repair locus;
- same constraint family;
- same subtype/action;
- same information condition;
- same value datatype;
- same popularity bucket.

Use `--allow-same-property-examples` only for explicit precedent-retrieval experiments.

## Freeze Final Prompt Configuration

After dev results are reviewed, freeze the chosen prompt settings before main-core inference:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-prompt-dev freeze \
  --output reports/prompt_dev/final_prompts_prompt_dev_v1.json \
  --representation hybrid_json_nl \
  --example-policy zero_shot \
  --context-bundles logic_only,local_graph \
  --proposal-track-modes oracle,diagnosis_routed \
  --notes "Frozen after dev prompt comparison."
```

This writes JSON and Markdown config artifacts. Freezing records the prompt version and selected axes; it does not run inference.

## Relationship To Reasoning Floor

Phase F prompt preparation is separate from `src/reasoning_floor.py`. The reasoning-floor runner remains the execution path for model calls and evaluation. Phase F artifacts are used to inspect and choose prompt settings before those runs.
