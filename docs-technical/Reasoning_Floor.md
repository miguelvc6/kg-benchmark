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
- `StaticResponseProvider` for deterministic tests

When `OpenAIChatProvider` is used, it auto-loads a repository `.env` file if present and reads:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- optional `OPENAI_BASE_URL`

Process environment variables still take precedence over values loaded from `.env`.

The adapter contract is:

`generate(prompt, system_prompt, response_format, metadata) -> raw_response, parsed_payload, usage`

## Outputs

A reasoning-floor run writes:

- raw model responses
- run manifest
- normalized track-diagnosis JSONL per ablation bundle
- normalized proposal JSONL per ablation bundle
- per-bundle evaluation traces and summaries
- one combined reasoning-floor summary with paper-facing breakdowns

## Test Coverage

The dry-run integration path is covered by [tests/test_reasoning_floor.py](/home/mvazquez/kg-benchmark/tests/test_reasoning_floor.py).

The diagnosis normalization path is covered by [tests/test_track_parser.py](/home/mvazquez/kg-benchmark/tests/test_track_parser.py).
