# Reasoning Floor

The zero-shot baseline runner is [reasoning_floor.py](/home/mvazquez/kg-benchmark/src/reasoning_floor.py).

## Objective

This runner implements the pre-Guardian reasoning floor described in the conceptual docs.

It executes one track-diagnosis call per case per ablation bundle and then generates a proposal either from the historical track or from the diagnosed track, depending on `--proposal-track-mode`.

It also executes a separate diagnostic call that asks the model to classify each case as `A_BOX`, `T_BOX`, or `AMBIGUOUS`.

The runner also accepts `--selection-manifest` so paper runs can target a deterministic benchmark subset without creating a second Stage 4 JSONL artifact.

The default execution mode is provider-aware. OpenAI runs default to batch mode, while other providers default to synchronous execution. You can still override this with `--execution-mode`.

## Ablation Bundles

The first-wave bundles are:

- `minimal_case`: sanitized case-local payload only
- `logic_only`: sanitized case-local payload plus pruned touched `L4_constraints`
- `local_graph`: sanitized case-local payload plus pruned touched `L1` through `L4`

`minimal_case` remains supported, but the CLI in [reasoning_floor.py](/mnt/c/Code/kg-benchmark/src/reasoning_floor.py) now defaults to `logic_only,local_graph`. Include `minimal_case` explicitly with `--ablation-bundles` when you want to run the no-context bundle.

Prompt payloads are now hard-sanitized before rendering. Model-visible inputs exclude benchmark-only fields such as:

- `track`
- `classification`
- `persistence_check`
- `repair_target`
- `build`
- `popularity`
- post-hoc `truth_*` and `*_current_2026*` fields

The pruned `logic_only` and `local_graph` bundles keep only constraint and graph context that is directly implicated by the current violation type, the current violating value, or the synthetic pre-repair target-property state reconstructed from benchmark repair metadata. This avoids leaking post-repair target values from the stored world-state snapshot into reasoning-floor prompts while still preserving current surrounding graph and constraint context.

For `local_graph`, the prompt builder now:

- rewrites `L1_ego_node.properties[target_pid]` to the synthetic pre-repair target state
- omits `L3_neighborhood` edges on the target property instead of exposing current/post-repair target edges
- backfills missing `L2_labels.entities` entries for synthetic pre-repair target QIDs from the benchmark's resolved Stage 2 mirrors when world state does not already provide them

This temporal policy is shared by proposal and track-diagnosis prompt bundles. Manifest rows still record pruning audit counts such as `constraint_count_before`, `constraint_count_after`, and for `local_graph` also `edge_count_after` and `label_count_after`.

## Provider Interface

The runner uses a provider adapter boundary defined in `guardian.model_provider`.

Current implementations:

- `OpenAIChatProvider`
- `OllamaChatProvider`
- `StaticResponseProvider` for deterministic tests

The default provider is selected from `.env` or the shell with `MODEL_PROVIDER`.

Supported runtime settings:

- `MODEL_PROVIDER=openai`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- optional `OPENAI_BASE_URL`
- `MODEL_PROVIDER=ollama`
- `OLLAMA_MODEL`
- optional `OLLAMA_BASE_URL` (defaults to `http://localhost:11434/api`)
- optional `OLLAMA_API_KEY` for direct `ollama.com/api` access
- optional `OLLAMA_KEEP_ALIVE` to keep a model resident between requests
- optional `OLLAMA_CONTEXT_LENGTH` to send `num_ctx` on each Ollama chat request
- optional `REASONING_FLOOR_PARALLEL_WORKERS` to override the runner's inferred worker count in `parallel` mode

Process environment variables still take precedence over values loaded from `.env`.

For provider API keys, use only the raw secret value in `.env`. Do not include the variable name again or a `Bearer ` prefix.

`src/reasoning_floor.py` also accepts `--model` to override the model name from `.env` without changing provider selection.

Execution CLI settings:

- `--execution-mode sync|parallel|batch`
- `--proposal-track-mode oracle|diagnosis_routed`
- `--parallel-workers` for `parallel` mode
- `--batch-completion-window` (defaults to `24h`)
- `--batch-poll-interval-seconds` (defaults to `60`)
- `--resume-run-dir` to continue an interrupted run from an existing run directory

If `--execution-mode` is omitted:

- `MODEL_PROVIDER=openai` defaults to batch execution
- other providers default to synchronous execution

`parallel` mode keeps the existing per-case request pattern but overlaps multiple cases with a bounded thread pool. This is the recommended throughput mode for Ollama because Ollama does not expose a provider batch API for text generation in this repository.

When `--execution-mode parallel` is used and `--parallel-workers` is omitted:

- Ollama models up to `8b` default to `2` workers
- larger or unknown Ollama models default to `1` worker
- non-Ollama providers default to `4` workers
- if `OLLAMA_NUM_PARALLEL` is present in the runner environment, the inferred Ollama worker count is capped to that value

The runner's `parallel` worker count is separate from Ollama server concurrency. `OLLAMA_NUM_PARALLEL` must be configured in the environment that launches `ollama serve`.

The adapter contract is:

`generate(prompt, system_prompt, response_format, metadata) -> raw_response, parsed_payload, usage`

Providers that support batch execution may additionally implement a batch contract used by the reasoning-floor runner to:

- write provider-specific batch-request JSONL lines
- submit and poll a batch job
- parse returned batch result records back into the same raw response and usage shape used by synchronous execution

Named prompt templates for the reasoning floor now live in [src/guardian/prompts.py](/mnt/c/Code/kg-benchmark/src/guardian/prompts.py). This keeps prompt text out of the runner itself and gives each prompt a stable descriptive name that can be reused across callers.

Those prompts now spell out the exact normalized JSON contract expected by the proposal and diagnosis parsers. The runner still requests JSON objects from providers, but prompt text now explicitly forbids wrapper shapes such as `proposal_id`, `summary`, `actions`, or `proposed_changes` in place of the canonical benchmark schema.

The proposal prompts now require `rationale`, `provenance`, and a proposal-level `uncertainty` object. `provenance` must be a JSON array of canonical provenance objects, and `uncertainty.confidence` must normalize to a numeric `0.0-1.0` score.

The T-box prompt now uses placeholder-only contrastive examples rather than a single concrete `Q21510859 + P2305:[Q5,Q43229]` worked example. It also states explicitly that:

- `target.constraint_type_qid` and `proposal.signature_after[*].constraint_qid` must come from constraint-family QIDs present in the supplied context
- violating item/type QIDs must not be copied into those constraint-family fields unless they actually appear as constraint families
- `SCHEMA_UPDATE` is the fallback when the payload shows a schema change but does not support a narrower directional reform family confidently

Those placeholders no longer mimic Wikidata `Q...` ids, so copied template tokens are less likely to be misread as valid-but-wrong constraint-family guesses.

The track-diagnosis prompt now includes explicit contrastive guidance for schema-vs-claim confusion cases such as allowed-entity-types and property-scope disputes.

As a safety net, the normalization layer also accepts several legacy rich-JSON shapes that appeared in earlier reasoning-floor runs. This compatibility path recovers common aliases such as `proposal_id` or `repair_id`, nested property fields, numeric track-diagnosis confidence values, proposal-level confidence aliases that can be rewritten into `uncertainty`, and common proposal wrappers when they can be mapped back to the public schemas.

Proposal normalization also now coerces common provenance variants into the canonical list form. Plain strings become one-item `OTHER` provenance lists, singleton objects with `url`, `node_id`, or `revision_id` are preserved with inferred `WEB`, `KG`, or `HISTORY` kinds, and mixed provenance lists are normalized entry-by-entry instead of failing the whole proposal.

For the OpenAI adapter, request payloads are encoded locally with strict JSON rules before the HTTP call. Non-finite numeric values or invalid Unicode now fail fast with run and case metadata in the error message instead of surfacing later as a remote `400 Bad Request`.

The OpenAI adapter also disables tool calling at the API level by sending `tool_choice: "none"` on each Chat Completions request, and it raises an error if the response still attempts to return a tool call.
 
## Outputs

A reasoning-floor run writes:

- raw model responses
- run manifest
- `run_config.json`
- a model-specific run directory named `<run_id>_<provider>_<model>`
- normalized track-diagnosis JSONL per ablation bundle
- normalized proposal JSONL per ablation bundle
- per-bundle evaluation traces and summaries
- one combined reasoning-floor summary with paper-facing breakdowns

Batch runs also write provider batch artifacts at the top level of the run directory:

- `batch_input.jsonl`
- `batch_request_manifest.jsonl`
- provider-specific batch job metadata, output JSONL, and error JSONL when available

`run_config.json` stores the immutable run settings and the ordered selected case ids so a later `--resume-run-dir` invocation can verify that the resumed command still targets the same provider, model, selection, execution mode, proposal-track mode, and ablation bundles.

The run manifest stores per-call token usage, cached prompt tokens when available, elapsed seconds when available, estimated cost when provider token pricing is configured in `.env`, cost-estimation metadata, and the prompt template name used for that call. When a retryable batch failure is recovered by a synchronous retry, the recovered raw and manifest rows also carry a `recovery` block describing the fallback path.

The combined summary stores run-level provider, model, output directory, execution mode, an explicit `batch_mode_used` flag, total elapsed time, aggregate prompt/completion/total tokens, aggregate cached tokens, aggregate estimated cost, and cost-estimation metadata. For OpenAI batch calls, the runner applies a built-in `0.5` multiplier to estimated costs to reflect batch pricing. When a batch run includes synchronous retry fallbacks, the summary marks `usage.cost_estimation_mode` as `mixed` and also records `usage.per_call_cost_estimation_modes` and `usage.per_call_cost_estimation_multipliers`. Parallel runs also record `run_info.parallel.workers` and its configuration source. Batch runs also record provider batch metadata in `run_info.batch`. Resumed runs also record `run_info.resume`, including the originating run directory, the persisted `run_config.json` path, and how much generation work had already completed before the resumed process started.

Run and per-bundle summaries now also expose proposal parser visibility directly:

- `parse_errors.proposal_parse_error_count`
- `parse_errors.proposal_parse_error_rate`
- `parse_errors.by_message`
- per-group `proposal_parse_error_count`, `proposal_parse_error_rate`, and `proposal_parse_errors_by_message`

This makes proposal parser regressions diagnosable from the top-level summary JSON without opening traces.

Run and per-bundle summaries also expose request-level transport failures directly:

- `request_errors.proposal_request_error_count`
- `request_errors.proposal_request_error_rate`
- `request_errors.track_diagnosis_request_error_count`
- `request_errors.track_diagnosis_request_error_rate`
- per-group `proposal_request_error_count`, `proposal_request_error_rate`, `track_diagnosis_request_error_count`, and `track_diagnosis_request_error_rate`

When a selection manifest is used, the run summary records its path under `inputs.selection_manifest`.

During execution, synchronous and parallel runs show a `tqdm` generation progress bar with elapsed time, ETA, current estimated cost, and estimated total cost. It refreshes in chunks of `min(1000 cases, 10% of total cases)`. Batch runs suppress that generation bar because provider work completes remotely rather than incrementally in-process.

The runner also prints UTC-timestamped phase-level status lines for run startup, batch submission and polling milestones, bundle evaluation start and completion, and final summary output. In batch mode, provider status updates are emitted when batch state or request counts change, and those progress lines now include a request-based ETA whenever the provider has reported enough completed work to estimate one.

Startup status now begins before generation-selection materialization and before the world-state index is opened, so long scans over large Stage 4 JSONL inputs no longer appear completely silent at process start.

After generation completes, the runner shows a second `tqdm` bar for bundle evaluation so the terminal does not appear idle while per-bundle summaries and traces are still being computed.

In synchronous mode, the runner appends raw responses, manifest rows, normalized proposals, and evaluation traces incrementally over one stable ordered selected subset. In parallel mode, it keeps the same outputs but executes multiple cases concurrently with bounded in-flight work.

When `--resume-run-dir` is used, the runner opens the existing JSONL artifacts in append mode, loads prior completion state from `run_manifest.jsonl`, and only submits the missing request(s). This works for sync, parallel, and batch execution. In `diagnosis_routed` mode, proposal resumption is driven from the existing normalized diagnosis artifacts, so the runner can skip already-finished diagnosis calls and submit only the missing proposals or synthetic skips.

In batch mode:

- `--proposal-track-mode oracle` keeps the original one-stage batch flow
- `--proposal-track-mode diagnosis_routed` uses a two-stage batch flow

The two-stage flow writes:

- `diagnosis_batch_input.jsonl`
- `diagnosis_batch_request_manifest.jsonl`
- `proposal_batch_input.jsonl`
- `proposal_batch_request_manifest.jsonl`
- provider batch artifacts renamed with `diagnosis_` and `proposal_` prefixes

When a provider batch artifact contains retryable `408`, `409`, `429`, or `5xx` request failures, the runner reconstructs the original prompt from the batch input artifact and retries that request synchronously. Each batch phase summary records the fallback outcome under `run_info.batch.phases[*].sync_retry_fallback`, and the top-level batch summary also carries an aggregated `sync_retry_fallback` block.

When `diagnosis_routed` predicts `AMBIGUOUS`, the runner does not submit a proposal request. It still writes synthetic raw/manifest proposal rows with `parse_status="skipped_ambiguous_track"` and `proposal_track_used="AMBIGUOUS"`.

During evaluation, the runner now switches classified-benchmark access strategy by selected-case count:

- for `10,000` selected cases or fewer, it loads the selected Stage 4 records into an in-memory cache once and reuses them across bundle evaluation
- above `10,000` selected cases, it streams the original Stage 4 benchmark once into a run-local `selected_classified_records.jsonl` artifact and then evaluates bundles by streaming that filtered subset from disk

This keeps small evaluation runs fast without rescanning the full Stage 4 file for every bundle, while avoiding large in-memory record caches on paper-scale selections.

When `--selection-manifest` and `--max-cases` are combined without a track filter, the runner trims the manifest-ordered case-id list before scanning the classified benchmark. This avoids reading the full Stage 4 JSONL when only the first manifest-ordered subset is needed.

## Evaluation Semantics

`A_BOX` acceptance is still based on executable repairs that reproduce the historical repaired value without increasing supported constraint violations.

To make A-box evaluation easier to debug, grouped and overall summaries now also expose `a_box_exact_action_match_rate`, `a_box_exact_value_match_rate`, and `a_box_regression_pass_rate` instead of forcing all A-box diagnosis through `accepted` or `exact_historical_agreement` alone.

`T_BOX` evaluation now separates exact, family-level semantic, and proxy signals:

- `accepted`, `functional_success`, and `exact_historical_agreement` require both exact action match and exact normalized `signature_after` match
- `semantic_success` now reports family-level compatibility, not literal action-label equality
- traces also expose `comparison.literal_action_match`, `comparison.semantic_family_match`, `comparison.target_constraint_match`, `comparison.exact_action_match`, `comparison.exact_signature_match`, `comparison.changed_constraint_type_hit`, `metrics.semantic_family_success`, `metrics.signature_after_jaccard`, `metrics.t_box_target_constraint_match`, and `details.proposal_admits_current_values` where applicable

For new reasoning-floor runs, T-box normalization is also strict about constraint-family IDs. The runner passes a case-local allowlist into the T-box parser, so malformed outputs such as entity/type QIDs used in `constraint_type_qid` become proposal `parse_error` rows and are omitted from normalized T-box proposal JSONL artifacts.

Grouped and overall summaries now expose metric applicability counts so T-box-only metrics can be interpreted with their true denominator.

This prevents summaries from collapsing plausible narrower T-box reforms to zero when the historical label is coarse `SCHEMA_UPDATE`, while still keeping exact historical agreement strict and avoiding credit for editing the wrong changed constraint family.

## Test Coverage

The dry-run integration path is covered by [tests/test_reasoning_floor.py](/home/mvazquez/kg-benchmark/tests/test_reasoning_floor.py).

The diagnosis normalization path is covered by [tests/test_track_parser.py](/home/mvazquez/kg-benchmark/tests/test_track_parser.py).
