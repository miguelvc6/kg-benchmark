# Reasoning Floor Viewer

The reasoning-floor viewer is a local Streamlit app for browsing runs under `reports/reasoning_floor/`.

It is optimized for debugging individual cases while still surfacing aggregate metrics for the selected run and ablation bundle.

## Launch

Install the UI extra, then start the app:

```bash
uv sync --extra ui
uv run streamlit run scripts/reasoning_floor_viewer.py -- --reports-root reports/reasoning_floor
```

Optional overrides are available when the app needs to reconstruct prompts or compute evaluation metrics for runs that do not have `reasoning_floor_summary.json`, `evaluation_traces.jsonl`, or `evaluation_summary.json` on disk:

```bash
uv run streamlit run scripts/reasoning_floor_viewer.py -- \
  --reports-root reports/reasoning_floor \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --world-state data/03_world_state.json
```

## What The App Reads

For the selected run and ablation bundle, the viewer reads:

- `run_manifest.jsonl`
- `raw_model_responses.jsonl`
- `<bundle>/track_diagnoses.jsonl`
- `<bundle>/a_box_proposals.jsonl`
- `<bundle>/t_box_proposals.jsonl`
- `<bundle>/evaluation_traces.jsonl` when present
- `<bundle>/evaluation_summary.json` when present
- `reasoning_floor_summary.json` when present

The viewer stays read-only. It does not rewrite or materialize report artifacts.

## On-The-Fly Evaluation

If bundle evaluation artifacts are missing, the viewer computes evaluation in memory with the existing evaluator.

Input paths are resolved in this order:

1. explicit CLI override
2. `reasoning_floor_summary.json` inputs
3. repository defaults:
   - `data/04_classified_benchmark.jsonl`
   - `data/03_world_state.json`

If the required benchmark inputs cannot be found, the app still shows raw and normalized outputs, but evaluation metrics are marked unavailable.

## Main Views

- Run and bundle summary metrics with parse-status breakdowns
- Filtered case list with next and previous navigation
- Filters for historical track, proposal parse status, proposal type, A-box classification type, T-box subtype, classification confidence, and acceptance status
- Instance inputs, with the full Stage 4 benchmark record and frozen world-state context loaded on demand so initial render stays responsive on large artifacts
- Prompt inputs for track diagnosis and repair proposal
- Single-case diagnosis and proposal outputs, including raw provider payloads
- Single-case evaluation trace and raw JSON inspectors

For batch runs, per-call `usage.elapsed_seconds` values in `run_manifest.jsonl` may be null because provider work completes remotely. In that case, the main elapsed-seconds metric falls back to `run_info.generation_elapsed_seconds` or `run_info.elapsed_seconds` from `reasoning_floor_summary.json`.

## Related Docs

- [Reasoning Floor](./Reasoning_Floor.md)
- [Evaluation Harness](./Evaluation_Harness.md)
- [Artifact Schemas](./Artifact_Schemas.md)
