# Reasoning Floor

The zero-shot baseline runner is [reasoning_floor.py](/home/mvazquez/kg-benchmark/src/reasoning_floor.py).

## Objective

This runner implements the pre-Guardian reasoning floor described in the conceptual docs.

It executes one model call per case per ablation bundle, without tools, rejection sampling, or memory.

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

The adapter contract is:

`generate(prompt, system_prompt, response_format, metadata) -> raw_response, parsed_payload, usage`

## Outputs

A reasoning-floor run writes:

- raw model responses
- run manifest
- normalized proposal JSONL per ablation bundle
- per-bundle evaluation traces and summaries
- one combined reasoning-floor summary with paper-facing breakdowns

## Test Coverage

The dry-run integration path is covered by [tests/test_reasoning_floor.py](/home/mvazquez/kg-benchmark/tests/test_reasoning_floor.py).
