# Phase E Completion: Non-LLM Baselines

**Completion date:** 2026-06-03  
**Implementation:** `src/lib/non_llm_baselines.py`, `src/non_llm_baselines.py`  
**Generated outputs:** `reports/non_llm_baselines/core_v1_phase_e/`

## Completed Tasks

### E1. Majority and constant-track baselines

Implemented:

- `majority_track`
- `always_a_box`
- `always_t_box`
- `always_ambiguous`

Each baseline writes normalized `track_diagnoses.jsonl`, evaluator traces, evaluator summary, and baseline-level track metrics including accuracy, macro-F1, confusion matrix, A-box overuse rate, and T-box miss rate.

### E2. Constraint-only TypeA baseline

Implemented `constraint_only_typea`.

The baseline emits normalized A-box proposals for supported deterministic TypeA repairs and abstains on unsupported or overlarge cases. Supported families include simple format normalization, set-membership rejection, safe format rejection/pruning, self-link rejection, target-required-claim creation, multiplicity normalization, and limited rule/range handling.

Current core-v1 result:

- TypeA coverage: 0.6423
- full-core accepted rate: 0.2942

### E3. Local lookup oracle

Implemented `local_lookup_oracle`.

The baseline emits normalized A-box proposals for TypeB cases when the target value is directly visible in local context or deterministically derivable from local text, including the P8726 statutory-instrument-id derivation.

Current core-v1 result:

- TypeB coverage: 0.9621
- full-core accepted rate: 0.1844

### E4. Invalid/do-nothing baseline

Implemented:

- `do_nothing_pre_repair`
- `invalid_empty`

`invalid_empty` records parse-error rows through the run manifest instead of feeding malformed proposals to the normalizers.

Current core-v1 result:

- `invalid_empty` accepted rate: 0.0000
- `do_nothing_pre_repair` accepted rate: 0.0083

## Verification

Tests:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run --extra dev python -m pytest tests/test_non_llm_baselines.py tests/test_evaluator.py
```

Generation:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run python src/non_llm_baselines.py \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json \
  --selection-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --output-dir reports/non_llm_baselines/core_v1_phase_e \
  --tier core
```

The generated top-level summary is `reports/non_llm_baselines/core_v1_phase_e/baseline_summary.md`.

## Evaluator Robustness Fix

During Phase E generation, a stored Wikidata format regex used syntax unsupported by Python `re`. The evaluator now treats Python-incompatible Wikidata regex patterns as non-matching rather than aborting regression checks.
