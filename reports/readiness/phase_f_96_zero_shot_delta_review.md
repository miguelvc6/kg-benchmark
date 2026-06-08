# Phase F 96-Case Zero-Shot Delta Review

Created: 2026-06-07

Reference readiness report: `reports/readiness/phase_f_g_readiness_review.md`

Evaluated run: `reports/prompt_dev/evaluation_prompt_dev_v1_96_zero_shot`

## Verdict

READY TO CONTINUE PHASE F; NOT READY TO FREEZE FOR PHASE G MAIN SCORING.

The 96-case run satisfies the main readiness follow-up request from the earlier review: it is balanced, larger than the
24-case smoke run, has no parse/request errors, and has no prompt-visible leakage. It also narrowly clears the
local-graph track-diagnosis sanity threshold. However, proposal quality remains too weak to freeze prompts for Phase G:
A-box exact value recovery is below 0.40, accepted rate is below 0.20, and T-box semantic-family success remains low even
after applying the evaluator fix for missing historical T-box signatures.

## Comparison To Previous Readiness Criteria

| Criterion from readiness review | Result | Status |
|---|---:|---|
| Larger balanced dev-only run, preferably 96 cases | 96 cases, 48 A_BOX / 48 T_BOX | PASS |
| Proposal parse error rate <= 2% | 0 / 192 proposal prompts | PASS |
| Request error rate <= 1% after retry/backoff | 0 request errors | PASS |
| Local-graph track diagnosis macro-F1 or balanced accuracy >= 0.65 | balanced accuracy 0.6563; macro-F1 0.6579 | PASS, narrow |
| Proposal normalization >= 98% | 192 / 192 proposal prompts normalized | PASS |
| No prompt leakage | no raw `repair_` / `reform_`, no `sitelinks_count`, no benchmark metadata leakage | PASS |
| Token budget comfortable | max prompt chars 18,398; max observed total tokens 5,916 | PASS |
| Prompt quality strong enough to freeze | accepted rate 0.1979; T-box semantic-family 0.0625-0.1042 | FAIL |

## Run Snapshot

- Run id: `prompt_dev_eval_20260607T110915`
- Provider/model: `university` / `Qwen/Qwen3-4B`
- Rendered prompts: 384
- Matrix rows: 4
- Eval cases: 96
- Representation: `hybrid_json_nl`
- Example policy: `zero_shot`
- Context bundles: `logic_only`, `local_graph`
- Repair mode: oracle

The top-level prompt-dev summary reports `skipped_existing_normalized: 383` and `normalized: 1`, which means this was
mostly a resumed/reused run. That is fine for interpreting the outputs, but request/normalization counters in the
top-level summary describe work performed during that invocation, not the total historical number of endpoint calls.

## Matrix Metrics

Metrics below use the current patched evaluator semantics, so missing historical T-box target/signature details are
treated as non-applicable for target/signature proxy metrics instead of failed.

| Matrix | Main task | Parse errors | Request errors | Key result |
|---|---:|---:|---:|---|
| logic-only track diagnosis | diagnosis | 0 | 0 | accuracy 0.6667; balanced accuracy 0.6667; macro-F1 0.6870 |
| local-graph track diagnosis | diagnosis | 0 | 0 | accuracy 0.6563; balanced accuracy 0.6563; macro-F1 0.6579 |
| logic-only oracle repair | proposal | 0 | 0 | accepted 0.1979; A-box exact value 0.3958; T-box semantic-family 0.0625 |
| local-graph oracle repair | proposal | 0 | 0 | accepted 0.1979; A-box exact value 0.3958; T-box semantic-family 0.1042 |

## Leakage Check

Model-visible prompt text passed the leakage checks:

- raw `repair_` / `reform_` case IDs: 0 hits
- `sitelinks_count`: 0 hits
- benchmark-only metadata terms such as `historical_track`, `repair_target`, `persistence_check`, and
  `classification_label`: 0 hits

Four broad `build` string hits were inspected and are false positives from the entity description "state capitol
building", not benchmark build metadata.

## Interpretation

The new run resolves the operational warnings from the 24-case smoke run: endpoint stability is good, parsing is stable,
the sample is properly balanced, and prompt leakage controls hold.

The remaining weakness is model/prompt quality:

- Track diagnosis is usable as a sanity baseline but fragile; local-graph barely clears the threshold and often predicts
  T_BOX for A_BOX cases.
- A-box repair proposals recover the exact target value for only 19 / 48 cases in both repair matrices.
- T-box exact scoring is limited by lean Stage 4 missing exact signatures, but semantic-family success is still low:
  3 / 48 logic-only and 5 / 48 local-graph after the evaluator applicability fix.
- Local graph helps T-box semantic family slightly, but not enough to freeze.

## Recommendation

Continue Phase F prompt development before Phase G main scoring.

Recommended next step:

1. Keep the 96-case run as the zero-shot baseline candidate.
2. Revise prompts specifically for:
   - reducing A_BOX over-broad delete/set behavior;
   - improving T_BOX action-family choice;
   - discouraging defaulting to generic `SCHEMA_UPDATE` or unrelated range/type constraint families.
3. Re-run another 96-case dev-only zero-shot evaluation after prompt changes.
4. Freeze only if proposal quality improves materially while preserving the current parse/request/leakage stability.

Phase G dry-run preparation can continue in parallel, but Phase G main scoring should wait for another Phase F prompt
iteration.
