# Track Diagnosis

The repository now implements a separate diagnostic task for predicting whether a benchmark case belongs to the `A_BOX` or `T_BOX` repair track.

## Purpose

This task is separate from proposal generation.

It answers a higher-level question:

- should the case be treated as instance repair
- should it be treated as schema reform
- or is the case ambiguous

## Runtime Components

- `guardian.track_parser`: normalization for diagnosis outputs
- `schemas/track_diagnosis.schema.json`: public diagnosis contract
- `src/reasoning_floor.py`: generates zero-shot diagnosis outputs and can optionally route proposal generation through them
- `src/evaluate.py`: scores diagnosis outputs against the historical benchmark track

Diagnosis prompts share the same sanitized bundle builder as proposal prompts. For `logic_only` and `local_graph`, the focus target property is reconstructed from synthetic pre-repair benchmark state rather than copied directly from current world-state target values, so diagnosis does not get post-repair target-property leakage that proposal prompting is supposed to avoid.

## Output Contract

Normalized diagnosis records contain:

- `case_id`
- `predicted_track`
- optional `confidence`
- optional `rationale`
- `canonical_hash`

`confidence` may be provided as either a descriptive string such as `high` or a numeric score such as `0.9`. The normalizer stores numeric values as strings in the normalized output so downstream consumers keep a single field type.

Supported predictions:

- `A_BOX`
- `T_BOX`
- `AMBIGUOUS`

## Evaluation Semantics

Current diagnosis scoring uses the historical benchmark `track` as the target label.

First-wave outputs include:

- exact-track-match
- ambiguous-prediction flag
- grouped diagnosis accuracy by class, subtype, track, ablation bundle, and popularity bucket

Reasoning-floor summaries now also surface proposal parser failure counts and exact-vs-semantic T-box success alongside diagnosis metrics, so diagnosis quality can be interpreted next to proposal execution quality from the same top-level report.

`AMBIGUOUS` is preserved as a legitimate model output, but it does not count as an exact match against historical `A_BOX` or `T_BOX`.

## Proposal Routing

`src/reasoning_floor.py` now exposes:

- `--proposal-track-mode oracle`
- `--proposal-track-mode diagnosis_routed`

`oracle` keeps proposal prompting on the historical benchmark track.

`diagnosis_routed` runs diagnosis first for every case and bundle, then:

- routes proposal generation to the diagnosed `A_BOX` or `T_BOX` track
- skips proposal generation when diagnosis returns `AMBIGUOUS`
- records `historical_track`, `proposal_track_used`, and `routing_source` in raw and manifest artifacts

Skipped routed proposals are still visible in run artifacts with synthetic proposal rows, so downstream evaluation and viewer tooling can distinguish `AMBIGUOUS` routing skips from parser failures or missing provider results.
