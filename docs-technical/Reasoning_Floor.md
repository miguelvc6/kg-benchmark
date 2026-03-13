# Reasoning Floor

The zero-shot baseline runner is [reasoning_floor.py](/home/mvazquez/kg-benchmark/src/reasoning_floor.py).

## Objective

This runner implements the pre-Guardian reasoning floor described in the conceptual docs.

It executes one model call per case per ablation bundle, without tools, rejection sampling, or memory.

It also executes a separate diagnostic call that asks the model to classify each case as `A_BOX`, `T_BOX`, or `AMBIGUOUS`.

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

Process environment variables still take precedence over values loaded from `.env`.

For provider API keys, use only the raw secret value in `.env`. Do not include the variable name again or a `Bearer ` prefix.

`src/reasoning_floor.py` also accepts `--model` to override the model name from `.env` without changing provider selection.

The adapter contract is:

`generate(prompt, system_prompt, response_format, metadata) -> raw_response, parsed_payload, usage`

Named prompt templates for the reasoning floor now live in [src/guardian/prompts.py](/mnt/c/Code/kg-benchmark/src/guardian/prompts.py). This keeps prompt text out of the runner itself and gives each prompt a stable descriptive name that can be reused across callers.

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

The run manifest stores per-call token usage, elapsed seconds, estimated cost when provider token pricing is configured in `.env`, and the prompt template name used for that call.

The combined summary stores run-level provider, model, output directory, total elapsed time, aggregate prompt/completion/total tokens, and aggregate estimated cost.

During execution, the runner shows a `tqdm` progress bar with elapsed time, ETA, current estimated cost, and estimated total cost. It refreshes in chunks of `min(1000 cases, 10% of total cases)`.

The runner streams Stage 4 cases from disk and appends raw responses, manifest rows, normalized proposals, and evaluation traces incrementally. It does not retain the full classified benchmark or full model-output payloads in memory during generation, which keeps long runs bounded by per-case prompt size rather than total dataset size.

## Test Coverage

The dry-run integration path is covered by [tests/test_reasoning_floor.py](/home/mvazquez/kg-benchmark/tests/test_reasoning_floor.py).

The diagnosis normalization path is covered by [tests/test_track_parser.py](/home/mvazquez/kg-benchmark/tests/test_track_parser.py).
