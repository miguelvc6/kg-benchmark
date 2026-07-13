# Automated Audit Finalization

This report combines deterministic gates with Codex-assisted review. AI review never changes benchmark labels
and cannot establish `EXTERNAL_CONFIRMED`.

## Counts

- Cases: 535570
- AI reviews: 500
- Cases with disagreement or concern flags: 6233
- `include`: 472605
- `diagnostic`: 62948
- `exclude_pending_rerender`: 17
- `exclude`: 0

## Policy

Deterministic integrity errors are excluded. Deterministic label disagreements and AI-only construct
concerns or uncertainty are diagnostic. Suspected temporal leakage is excluded pending rerender. These
automated decisions are error-discovery dispositions, not human annotation or semantic ground truth.
