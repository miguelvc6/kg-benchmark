# Phase F Prompt Development

Phase F prepares prompt-development artifacts on the dev manifest only. It does not run LLM inference by default.

## Prompt Template Script

Prompt text and representation renderers live in:

- `scripts/prompt_dev_templates.py`

This script defines the Phase F prompt version, supported representations, task contracts, optional abstention contract, and prompt rendering function. Keep prompt wording changes there so templates are easy to review before any main-core run.

Current main prompt candidate: `prompt_dev_v4_spec_only`.

Prompt validity is governed by `reports/prompt_dev/prompt_validity_charter.md`. The main prompt may define tasks,
fields, operation semantics, the visible-evidence boundary, JSON contracts, and neutral placeholder examples. It must
not encode hidden benchmark classes/subtypes, answerability-audit rules, dev-set repair recipes, or prompt wording
optimized to known dev failures.

`prompt_dev_v2` was introduced after the 96-case zero-shot failure taxonomy found systematic prompt-level failures:

- A-box proposals used constraint-family, allowed-type, or report-type QIDs as repaired claim values.
- A-box proposals over-deleted values instead of preserving visible retained values.
- T-box proposals invented `signature_after` values under compact temporal context.
- T-box proposals chose directional actions when no pre-change signature or changed value evidence was visible.
- Track diagnosis over-predicted T-box from constraint-report vocabulary alone.

The v2 canary sharply reduced T-box invented-signature and unsupported-directional-action failures, but it regressed
A-box over-deletion and track diagnosis. `prompt_dev_v3_scaffolded` keeps the v2 T-box discipline and revises the A-box
and diagnosis wording:

- A-box prompts now prefer targeted `REMOVE` over `DELETE_ALL` when only one visible value is bad.
- A-box prompts explicitly preserve retained values and reserve `DELETE_ALL` for evidence that all current values should
  be removed.
- Track diagnosis still warns that a constraint report alone does not imply T-box, but also states that visible
  property-level schema-change evidence should support T-box.

`prompt_dev_v3_scaffolded` is retained as a diagnostic ablation only. It is not the main Phase G reasoning-floor prompt
candidate, even if it scores higher than a clean specification prompt.

`prompt_dev_v4_spec_only` removes scaffolded failure-mode strategies and keeps only the task specification, output
schemas, field definitions, QID role separation, and visible-evidence boundary. It is the fair main candidate for a
Phase G dry run if it is operationally stable on a disjoint dev holdout.

T-box context still follows the pre-reform temporal policy: prompts expose `signature_before` constraints when they are
available, otherwise they expose only a compact constraint-family inventory plus visible violation context. The
implementation records the chosen temporal policy in internal audit metadata only; model-visible prompt payloads do not
include policy labels such as `compact_inventory_no_pre_change_signature`.

To explicitly render or evaluate the scaffolded ablation, set `PROMPT_DEV_VERSION=prompt_dev_v3_scaffolded` in the
environment for that command. The default is `prompt_dev_v4_spec_only`.

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
  --sample-strategy diverse_stratified \
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

`--sample-strategy stratified` is the prompt-development default. It samples the dev manifest by rotating across
historical track, classification class/subtype, and the manifest `selection_stratum` before applying `--max-cases`.
For broader canaries, use `--sample-strategy diverse_stratified`; it keeps the same stratum rotation while preferring
previously unseen focus QIDs and properties inside each stratum. Use `--sample-strategy manifest_order` only when
reproducing the literal manifest order. Render summaries report case counts by track, class, subtype, class/subtype,
selection stratum, plus unique focus-QID and property counts.

Prompt-visible case IDs are neutralized to values such as `case_000001`. The prompt asks the model to copy this neutral
ID, and the evaluation path maps it back to the internal benchmark case ID before normalization and scoring. The raw
benchmark IDs remain internal run metadata and should not appear in model-visible prompt text.

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
  --sample-strategy stratified \
  --representations hybrid_json_nl,pure_nl \
  --example-policies zero_shot,matched_2shot \
  --context-bundles logic_only,local_graph \
  --tasks track_diagnosis,repair_proposal \
  --repair-track-modes oracle
```

`evaluate` runs LLM inference. Keep it restricted to the dev manifest and do not use it on the frozen core selection.

During evaluation, the CLI shows a `tqdm` progress bar over rendered prompt requests. The postfix reports prompt outcomes
as they are observed, including normalized rows, skipped resumed rows, request errors, and parse errors. Use
`--no-progress` when writing logs to files or running in an environment where terminal progress output is undesirable.

When using the university endpoint, the Responses provider supports endpoint-resilience settings in `.env`:

```dotenv
UNIVERSITY_OPENAI_TIMEOUT_SECONDS=240
UNIVERSITY_OPENAI_MAX_OUTPUT_TOKENS=1024
UNIVERSITY_OPENAI_MAX_RETRIES=4
UNIVERSITY_OPENAI_RETRY_BASE_SECONDS=2
UNIVERSITY_OPENAI_RETRY_MAX_SECONDS=20
```

Retryable failures are HTTP `429`, `500`, `502`, `503`, `504`, read timeouts, connection timeouts, and temporary
connection errors. HTTP `400` context-window failures are not retried; reduce prompt size instead.

For local Ollama on the H100 VM, use [Ollama VM Runbook](./Ollama_VM_Runbook.md). The prompt-dev Ollama script is:

```bash
bash scripts/run_phase_f_v4_ollama_holdout.sh
```

It reads `.env.ollama.vm` when present and uses the same v4 spec-only prompt candidate by default.

The evaluate command resumes by default when the output directory already contains matrix artifacts. Existing normalized
rows are skipped. Existing `request_error` and `parse_error` rows are also left in place unless you explicitly retry
them:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-prompt-dev evaluate \
  --output-dir reports/prompt_dev/evaluation_prompt_dev_v1 \
  --model-endpoint university \
  --retry-failures
```

Use `--no-resume` only when you intentionally want to append fresh attempts for every rendered prompt in the output
directory.

Use `--max-prompt-chars` to avoid submitting prompts that are likely to exceed the endpoint context window:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-prompt-dev evaluate \
  --output-dir reports/prompt_dev/evaluation_prompt_dev_v1 \
  --model-endpoint university \
  --max-prompt-chars 120000
```

Oversized prompts are recorded as local `request_error` rows with a provider-error message explaining the character
limit, and no endpoint request is made for those prompts.

Top-level `prompt_dev_evaluation_summary.json` and `prompt_dev_evaluation_comparison.md` are written even when an
individual matrix scoring step fails. Per-matrix result blocks include normalized, parse-error, request-error, and
skipped counts, plus counters by historical track, prompt task, and context bundle.

For `track_diagnosis`-only matrices, proposal files are intentionally absent. Interpret those rows by
`track_diagnosis_accuracy`, `track_diagnosis_present_rate`, and request/parse status; generic proposal-quality columns
such as functional success, auditability, and T-box target/signature proxy metrics are not the decision metric for those
matrices.

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

For few-shot prompt development, `--core-manifest` is required by default so examples can exclude core cases and core
T-box property-revision groups. If you intentionally want to render or evaluate few-shot prompts without this guard,
pass `--allow-core-example-risk` and treat the output as a leakage-risk experiment, not a paper-facing prompt-selection
run.

## T-Box Temporal Context Policy

T-box repair prompts must not expose post-reform target-property constraints that already contain the answer. When a
benchmark case has `repair_target.constraint_delta.signature_before` or `old_constraints`, prompt development exposes
that pre-reform constraint signature. When no pre-reform signature is available, prompt development exposes only a
compact constraint-family inventory plus the visible violation context, not the full current L4 target-property
constraint payload. Local graph prompts follow the same L4 temporal policy and do not expose `sitelinks_count`.

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
