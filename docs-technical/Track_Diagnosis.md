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
- `src/reasoning_floor.py`: generates zero-shot diagnosis outputs
- `src/evaluate.py`: scores diagnosis outputs against the historical benchmark track

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

`AMBIGUOUS` is preserved as a legitimate model output, but it does not count as an exact match against historical `A_BOX` or `T_BOX`.
