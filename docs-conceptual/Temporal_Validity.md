# Temporal Validity

Temporal validity is a core methodological guardrail for WikidataRepairEval. Historical repairs cannot be evaluated naively against a newer graph snapshot, because the current graph may contain values added by the repair itself or by later unrelated edits.

## The Problem

A repair that was correct at time `t` can appear different in the frozen 2026 world state because:

- the entity changed after the repair;
- the relevant property constraint changed;
- the entity was merged, deleted, or remodeled;
- another editor resolved the issue independently;
- the repaired target value is now visible in the contemporary graph.

Without temporal controls, evaluation can stop measuring repair reasoning and start measuring world evolution or prompt leakage.

## Persistence Check

The benchmark should not blindly treat every historical repair as a usable evaluation case. It should check whether the repaired violation still expresses the same logical evaluation question in the frozen contemporary context.

Cases whose current context no longer supports the same evaluation question should be filtered, flagged, or kept out of core experiments rather than silently reinterpreted.

## Target-Property Rule

The edited target property is the main leakage risk. The benchmark therefore follows a special rule:

> Current graph context may be used for surrounding labels, neighbors, and constraints, but the edited target property must be reconstructed from historical repair metadata as the pre-repair state.

For model inputs, this means:

- `L1_ego_node.properties[target_pid]` is rewritten to the synthetic historical pre-repair target state;
- current `L3_neighborhood` edges on the target property are omitted or rewritten so they do not reveal post-repair values;
- labels for synthetic pre-repair target ids may be backfilled from historical mirrors when needed;
- benchmark-only fields such as `repair_target`, `classification`, `persistence_check`, and current target-property truth are hidden.

This rule allows frozen 2026 context to be useful without exposing the answer.

## Correct Evaluation Question

The relevant question is not:

> Does the entity violate today's rule in today's graph?

It is:

> Given a temporally sanitized case representation, can the model propose the historically accepted repair transaction without access to post-repair target values?

This framing preserves causal attribution between the original violation, the historical repair, and the evaluated proposal.

## Conceptual Consequence

Temporal filtering and target-property reconstruction are not implementation conveniences. They define the benchmark's scientific validity. Without them, local graph context could leak the answer and no-retrieval success would be hard to distinguish from memorization, leakage, or lucky guessing.
