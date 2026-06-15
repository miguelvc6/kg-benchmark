# Track Diagnosis

The repository now implements a separate diagnostic task for predicting whether a benchmark case belongs to the `A_BOX` or `T_BOX` repair track.

Operational status: track diagnosis is not yet validated for main routing. Oracle repair remains the Phase G main
condition. `diagnosis_routed` is a dev-gated ablation and must not be run on the core as a paper-facing condition until
the Phase F diagnosis branch passes the gates below.

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
- `scripts/prompt_dev_templates.py`: contains the dev-only `prompt_dev_diag_v1_locus_spec` diagnosis prompt candidate
- `src/evaluate.py`: scores diagnosis outputs against the historical benchmark track

Prompt-dev diagnosis experiments should use the explicit diagnosis-neutral context bundles rather than the ordinary
repair proposal bundles. The diagnosis-neutral bundles are:

- `diagnosis_minimal`: neutral case ID, QID, property, English labels, and violation context
- `diagnosis_logic_neutral`: minimal payload plus a structurally neutral current constraint context
- `diagnosis_local_neutral`: logic-neutral payload plus local graph context with target-property post-repair L1/L3 leakage suppressed

These bundles do not branch on the historical `track`, do not use `repair_target.constraint_delta`, and do not expose
`classification`, `repair_target`, `persistence_check`, `popularity`, raw `repair_`/`reform_` case IDs, or target-property
post-repair local graph edges. Ordinary `logic_only` and `local_graph` repair bundles are still used for oracle proposal
prompting, where the T-box pre-reform temporal policy is intentional.

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

Prompt-dev diagnosis evaluations additionally write `track_diagnosis_report.json` and `track_diagnosis_report.md`.
These reports compute:

- confusion matrix by historical track;
- confusion by class, subtype, selection stratum, main-score flag, and diagnostic-only flag;
- A-box recall, T-box recall, balanced accuracy, and macro-F1;
- `AMBIGUOUS` rate and where ambiguity is used;
- routed-risk counts for wrong repair prompt selection or skipped proposal routing.

The diagnosis prompt is eligible for a diagnosis-routed dev canary only if all of these gates pass:

- request error rate `<= 1%`;
- parse error rate `<= 4%`;
- balanced accuracy `>= 0.70`;
- A-box recall `>= 0.65`;
- T-box recall `>= 0.65`;
- `AMBIGUOUS` rate `<= 15%`, unless a diagnostic report explicitly justifies a higher cap.

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

The current v4 spec-only diagnosis prompt failed the Phase F routing gate on the v4 holdout, with accuracy close to
chance. A follow-up audit found that the legacy diagnosis render path reused repair proposal context bundles whose
constraint structure could be conditioned on the historical track. Use `prompt_dev_diag_v1_locus_spec` with
`--diagnosis-context-bundles diagnosis_minimal,diagnosis_logic_neutral,diagnosis_local_neutral` for the next dev-only
diagnosis validation. If that branch fails, do not run Phase G `diagnosis_routed`; write a blocked report instead.
