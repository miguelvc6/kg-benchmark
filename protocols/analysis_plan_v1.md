# Confirmatory Analysis Plan v1

This plan is frozen before acquisition of the confirmatory snapshot. The 1,200-case local population is the headline
population; the nested 600-case Azure population is a paired calibration reference and is not substituted for missing
local results.

## Populations and conditions

- Every enabled Ollama model evaluates all 1,200 selected cases under `logic_only` and `local_graph`.
- Azure `gpt-5.6-sol` evaluates the deterministic nested 600 under the same two bundles, using batch execution, high
  reasoning effort, and no tools.
- Oracle proposal routing is the confirmatory mode. Diagnosis-routed or few-shot runs are separately identified
  extensions and do not alter the headline population.
- A-box and T-box results are separate metric families. No combined repair-success score is reported.

## Primary analysis

The primary A-box estimand is the paired `local_graph - logic_only` difference in accepted repair rate on main-score
cases. Report the case-micro effect and the event-cluster macro effect with 95% percentile intervals from 5,000 cluster
bootstrap samples using seed 13. Report the two-sided exact McNemar test for paired binary outcomes. Incomplete pairs fail
the confirmatory analysis; they are not silently dropped.

## T-box analysis

Evaluate `tbox_taxonomy_patch_v1` only where complete mechanically supported gold exists. Report taxonomy-patch metrics
separately, including repair-operation exact match, taxonomy-code exact match, evidence-level exact match, and the
registered component metrics emitted by the frozen evaluator. `CLASS_HIERARCHY_ADD` and `EXCEPTION_ADD` are outside the
confirmatory operation vocabulary. The legacy strict-signature task is exploratory only.

## Secondary and calibration analyses

- Report model-specific bundle effects and descriptive results by registered A-box class and T-box subtype.
- Report Azure results only on the exact nested 600 and compare bundles within that paired population.
- Population-weighted effects are reported only if every paired unit has a positive preregistered weight.
- Diagnostic, unsupported, disagreement, malformed, and prompt-rejected cases are summarized in aggregate and never
  enter main-score denominators.

## Multiplicity, missingness, and replay

Use Holm adjustment within any declared family of multiple confirmatory binary comparisons. Report all failures,
missing responses, parse failures, and incomplete pairs. Transport retries must replay the exact request; semantic
repair retries are disabled. Stored generations are immutable inputs to rescoring, allowing evaluator and metric updates
without another provider request. Any analysis not named above is exploratory.
