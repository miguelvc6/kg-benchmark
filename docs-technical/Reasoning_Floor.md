# Reasoning Floor

The zero-shot baseline runner is [reasoning_floor.py](/home/mvazquez/kg-benchmark/src/reasoning_floor.py).

## Objective

This runner implements the pre-Guardian reasoning floor described in the conceptual docs.

It executes one proposal call and one track-diagnosis call per case per ablation bundle, without tools, rejection sampling, or memory.

It also executes a separate diagnostic call that asks the model to classify each case as `A_BOX`, `T_BOX`, or `AMBIGUOUS`.

The runner also accepts `--selection-manifest` so paper runs can target a deterministic benchmark subset without creating a second Stage 4 JSONL artifact.

The default execution mode is provider-aware. OpenAI runs default to batch mode, while other providers default to synchronous execution. You can still override this with `--execution-mode`.

## Ablation Bundles

The first-wave bundles are:

- `minimal_case`: Stage 4 case metadata only
- `logic_only`: minimal case plus `L4_constraints`
- `local_graph`: minimal case plus `L1` through `L4`

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
- `--parallel-workers` for `parallel` mode
- `--batch-completion-window` (defaults to `24h`)
- `--batch-poll-interval-seconds` (defaults to `60`)

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

As a safety net, the normalization layer also accepts several legacy rich-JSON shapes that appeared in earlier reasoning-floor runs. This compatibility path recovers common aliases such as `proposal_id` or `repair_id`, nested property fields, numeric track-diagnosis confidence values, and common proposal wrappers when they can be mapped back to the public schemas.

For the OpenAI adapter, request payloads are encoded locally with strict JSON rules before the HTTP call. Non-finite numeric values or invalid Unicode now fail fast with run and case metadata in the error message instead of surfacing later as a remote `400 Bad Request`.

The OpenAI adapter also disables tool calling at the API level by sending `tool_choice: "none"` on each Chat Completions request, and it raises an error if the response still attempts to return a tool call.
 
## Outputs

A reasoning-floor run writes:

- raw model responses
- run manifest
- a model-specific run directory named `<run_id>_<provider>_<model>`
- normalized track-diagnosis JSONL per ablation bundle
- normalized proposal JSONL per ablation bundle
- per-bundle evaluation traces and summaries
- one combined reasoning-floor summary with paper-facing breakdowns

Batch runs also write provider batch artifacts at the top level of the run directory:

- `batch_input.jsonl`
- `batch_request_manifest.jsonl`
- provider-specific batch job metadata, output JSONL, and error JSONL when available

The run manifest stores per-call token usage, cached prompt tokens when available, elapsed seconds when available, estimated cost when provider token pricing is configured in `.env`, cost-estimation metadata, and the prompt template name used for that call.

The combined summary stores run-level provider, model, output directory, execution mode, an explicit `batch_mode_used` flag, total elapsed time, aggregate prompt/completion/total tokens, aggregate cached tokens, aggregate estimated cost, and cost-estimation metadata. For OpenAI batch runs, the runner applies a built-in `0.5` multiplier to estimated costs to reflect batch pricing. Parallel runs also record `run_info.parallel.workers` and its configuration source. Batch runs also record provider batch metadata in `run_info.batch`.

When a selection manifest is used, the run summary records its path under `inputs.selection_manifest`.

During execution, the runner shows a `tqdm` generation progress bar with elapsed time, ETA, current estimated cost, and estimated total cost. It refreshes in chunks of `min(1000 cases, 10% of total cases)`.

The runner also prints phase-level status lines for run startup, batch submission and polling milestones, bundle evaluation starts, and final summary output. In batch mode, provider status updates are emitted when batch state or request counts change.

After generation completes, the runner shows a second `tqdm` bar for bundle evaluation so the terminal does not appear idle while per-bundle summaries and traces are still being computed.

In synchronous mode, the runner streams Stage 4 cases from disk and appends raw responses, manifest rows, normalized proposals, and evaluation traces incrementally. In parallel mode, it keeps the same outputs but executes multiple cases concurrently with bounded in-flight work. In batch mode, it first writes provider batch-input artifacts, then reconstructs the normal reasoning-floor outputs from the completed batch results.

## Test Coverage

The dry-run integration path is covered by [tests/test_reasoning_floor.py](/home/mvazquez/kg-benchmark/tests/test_reasoning_floor.py).

The diagnosis normalization path is covered by [tests/test_track_parser.py](/home/mvazquez/kg-benchmark/tests/test_track_parser.py).
