# Conceptual Deviation Report

This report captures mismatches between the current conceptual documentation and the implemented repository as of March 12, 2026. The codebase was treated as the source of truth.

## Summary

The repository currently implements benchmark construction, classification, and deterministic split generation. It does not yet implement the full evaluation layer or the reasoning-floor baseline described in the conceptual docs.

## Confirmed Implemented Scope

The code currently provides:

- Stage 1 candidate mining in `src/fetcher.py` and `src/lib/mining.py`
- Stage 2 repair reconstruction in `src/fetcher.py`
- Stage 3 popularity enrichment and world-state construction in `src/fetcher.py`, `src/lib/popularity.py`, and `src/lib/world_state.py`
- Stage 4 classification in `src/classifier.py`
- Stage 5 deterministic train/dev/test split generation in `src/splitter.py`

## Deviations from Conceptual Claims

### 1. No Executable Evaluation Harness

Conceptual docs currently describe repository-level evaluation procedures and metrics as part of the minimum publishable unit.

Implemented reality:

- there is no evaluation script or package that executes benchmark proposals against the classified benchmark
- there is no code for Pass@K, conversion-rate measurement, revert-rate measurement, or information-preservation scoring
- there is no verifier loop or acceptance harness wired into the repository runtime

Affected conceptual files:

- [Scientific_Vision.md](../docs-conceptual/Scientific_Vision.md)
- [Evaluation_Framework.md](../docs-conceptual/Evaluation_Framework.md)
- [README.md](../docs-conceptual/README.md)

### 2. No Reasoning-Floor Baseline Runner

Conceptual docs state that the repository should implement a pre-Guardian reasoning-floor baseline using zero-shot prompting.

Implemented reality:

- there is no script, module, or CLI entry point that runs zero-shot model prompting over benchmark cases
- there is no prompt assembly code, model adapter, result recorder, or baseline output artifact in the repository
- there is no experimental harness that consumes the classified benchmark for baseline measurement

Affected conceptual files:

- [Scientific_Vision.md](../docs-conceptual/Scientific_Vision.md)
- [Evaluation_Framework.md](../docs-conceptual/Evaluation_Framework.md)
- [Benchmark_Taxonomy.md](../docs-conceptual/Benchmark_Taxonomy.md)
- [README.md](../docs-conceptual/README.md)

### 3. Guardian / Proposal Path Exists Only as Design Assets

The conceptual layer discusses Guardian-style intervention and proposal semantics.

Implemented reality:

- `schemas/verified_repair_proposal.schema.json` exists, but no production module in this repository loads it as part of a runnable benchmark workflow
- `tests/test_patch_parser.py` expects `guardian.patch_parser`, but that module does not exist in the repository
- `uv run python -m unittest tests/test_patch_parser.py` fails with `ModuleNotFoundError: No module named 'guardian'`

This means the proposal-validation path is planned or partially stubbed, not implemented.

## Code-Verified Notes

The following checks were run during documentation refresh:

- `uv run python src/classifier.py --self-test` passed
- `uv run python -m unittest tests/test_patch_parser.py` failed because `guardian.patch_parser` is missing

## Recommendation

Either:

- reduce the conceptual claims so they describe these pieces as planned next steps rather than current repository capabilities

or:

- implement the missing evaluation harness, reasoning-floor baseline, and proposal-validation path so the conceptual layer matches reality
