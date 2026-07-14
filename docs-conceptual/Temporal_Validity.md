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
- benchmark-only fields such as `repair_target`, `classification`, `persistence_check`, and current target-property values are hidden.

This rule allows frozen 2026 context to be useful while reducing direct target exposure. It does not rule out every indirect or semantic leak from later context.

## Correct Evaluation Question

The relevant question is not:

> Does the entity violate today's rule in today's graph?

It is:

> Given a temporally sanitized case representation, can the model propose the historically accepted repair transaction without access to post-repair target values?

This framing evaluates alignment with the recorded historical transaction. It does not establish that the transaction caused the report disappearance, was semantically correct, or was the unique valid repair.

## Scope Of The Temporal Claim

The benchmark does **not** reconstruct the complete graph as it existed at the repair timestamp. Labels, descriptions,
non-target properties, neighborhood structure, and constraint context come from a later frozen snapshot. Only the edited
target property is reconstructed from historical repair metadata. The scientific task must therefore be described as a
**historically targeted repair under later frozen context**, not as repair from a complete historical world state.

Before release or confirmatory evaluation, the dataset must bind Stage 2, the world state, and Stage 4 to a snapshot
manifest with source identifiers and artifact hashes. Exhaustive automated consistency auditing must cover every Stage 4
record and every supplied or final rendered prompt. Stage 2 availability is an explicit audit outcome: missing records or
fields must be reported and handled by declared policy rather than silently interpreted as empty evidence. Prompt scans
must test the supported exact and normalized target representations across all model-visible fields. A passing scan rules
out only the tested direct leaks; it does not establish that every surrounding field was knowable at repair time or that
no semantic paraphrase reveals the answer.

No independent human evaluation is available. Label-hidden Codex-assisted review may be used to discover suspected
reconstruction, context, or leakage errors, but its findings are exploratory nominations rather than ground truth,
independent annotation, or inter-annotator agreement. Neither automated scans nor Codex review establish causal or
semantic uniqueness. In the historical `full_v1` audit, the deterministic scanner found zero high-risk hits but Codex
nominated five suspected-leakage prompts affecting four cases; this demonstrates why passing lexical scans cannot support
a universal no-leakage claim. The audit's methodological role is preserved in the
[development history](../docs-technical/Development_History.md).

Provenance evaluation follows the same distinction. Structural completeness asks whether a proposal supplies a rationale,
citation-shaped provenance, and uncertainty. Provenance support asks whether the cited identifier or snippet occurs in the
model-visible evidence. Neither measure alone establishes that a citation is factually correct in the historical world.

## Conceptual Consequence

Temporal filtering and target-property reconstruction are not implementation conveniences. They define the benchmark's scientific validity. Without them, local graph context could leak the answer and no-retrieval success would be hard to distinguish from memorization, leakage, or lucky guessing.
