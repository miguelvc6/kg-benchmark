# Phase F Prompt Validity Charter

This charter defines what may appear in the main Phase F prompt candidate for a fair reasoning-floor experiment.

## Allowed In Main Prompt

- Task definitions.
- Field definitions.
- Operation semantics.
- `A_BOX`, `T_BOX`, and `AMBIGUOUS` definitions.
- Visible-evidence boundary.
- JSON schema and output contract.
- Neutral placeholder examples only.

## Disallowed In Main Prompt

- Hidden class or subtype names such as `TypeA`, `TypeB`, or `TypeC`.
- Repair recipes derived from dev failure analysis.
- Operation heuristics such as "`targeted REMOVE` is safer".
- Instructions that mirror classifier taxonomy.
- Answerability-audit-derived rules.
- Prompt wording optimized to known dev failures.
- Core data inspection.

## Current Candidate Policy

`prompt_dev_v4_spec_only` is the main prompt candidate. It is intended to be an honest task specification, not a
scaffold tuned to dev-set failure modes.

`prompt_dev_v3_scaffolded` is retained only as a diagnostic ablation. It may be useful for understanding model behavior,
but it should not become the main Phase G reasoning-floor prompt solely because it scores higher on dev or holdout.
