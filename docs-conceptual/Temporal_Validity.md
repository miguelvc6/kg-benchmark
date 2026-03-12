# Temporal Validity and the Time-Travel Paradox

Historical KG repairs become invalid evaluation examples when the benchmark quietly compares a past action against a different present-day world.

## The Problem

A repair that was correct at time `t` can appear incorrect in 2026 because:

- the entity changed
- the constraint changed
- the entity was merged or deleted
- another editor resolved the issue independently

Without a temporal validity rule, evaluation stops measuring reasoning quality and starts measuring world evolution.

## Correct Evaluation Question

The relevant question is not "does the entity violate today's rule in today's graph?".

It is:

"If the historical rule is applied to the current graph snapshot, does the same logical problem still exist?"

If the answer is no, the case is temporally inconsistent and should be discarded.

## Why Discarding Is Necessary

Discarding inconsistent cases preserves:

- reproducibility
- a stable task definition
- causal attribution between the original violation and the evaluated repair

Updating the ground truth after the fact would turn benchmark construction into subjective reinterpretation.

## Conceptual Consequence

Temporal filtering is not a convenience optimization. It is part of the scientific design of the benchmark because it defines which historical events still support valid present-day evaluation.
