# Artifact Schemas

This document describes the artifacts produced by the current code. Conceptual benchmark categories live in [docs-conceptual/Benchmark_Taxonomy.md](../docs-conceptual/Benchmark_Taxonomy.md).

When this document and a schema file disagree, the Python implementation is the source of truth.

## Naming Conventions

Machine-stable fields remain untouched, but the pipeline adds deterministic human-readable mirrors.

| Suffix | Meaning |
| --- | --- |
| `_raw` | Original source value preserved for hashing or backward compatibility |
| `_qids` | Parsed and normalized QID list extracted from a field |
| `_label_en` / `_labels_en` | English label mirror for one id or a list of ids |
| `_description_en` / `_descriptions_en` | English descriptions aligned with the source field |

Aliases are intentionally excluded from these mirrors to keep artifacts compact and deterministic.

## Stage 2: `data/02_wikidata_repairs.json(.jsonl)`

Stage 2 is the provenance-grade repair index. Each record stores:

- record identity: `id`, `qid`, `property`, `track`
- report provenance inside `violation_context`
- repair payload inside `repair_target`
- persistence outcome inside `persistence_check`
- deterministic English mirrors for ids and values
- optional `constraint_delta` for T-box cases
- optional ambiguity metadata when A-box and T-box evidence overlap
- attached `popularity` block once Stage 3 popularity enrichment completes

Representative top-level shape:

```json
{
  "id": "repair_Q42_123456789",
  "qid": "Q42",
  "property": "P569",
  "track": "A_BOX",
  "violation_context": {},
  "repair_target": {},
  "persistence_check": {},
  "popularity": {}
}
```

Common Stage 2 fields added by the current implementation:

- top level: `qid_label_en`, `qid_description_en`, `property_label_en`, `property_description_en`
- `violation_context`: `report_fix_date`, `report_revision_old`, `report_revision_new`, `report_page_title`, `report_violation_type_qids`
- `repair_target` for A-box: `kind`, `action`, `old_value`, `new_value`, `value`, `revision_id`, `author`
- `repair_target` for T-box: `kind`, `property_revision_id`, `property_revision_prev`, `author`, `constraint_delta`
- `persistence_check.current_value_2026`

`information_type` is present but remains a placeholder populated as `"TBD"` by `src/fetcher.py`.

## Stage 3: `data/03_world_state.json`

Stage 3 is a JSON object keyed by Stage 2 `id`.

Each entry follows the four-layer contract:

- `L1_ego_node`: focus entity identity and properties
- `L2_labels`: labels and descriptions for referenced ids
- `L3_neighborhood`: outgoing one-hop graph context
- `L4_constraints`: property-constraint metadata used during classification and evaluation
- `constraint_change_context`: optional T-box before/after context

Current required fields enforced by `src/lib/world_state.py`:

- `L1_ego_node.qid`
- `L1_ego_node.label`
- `L1_ego_node.description`
- `L1_ego_node.properties`
- `L3_neighborhood.outgoing_edges[*].property_id`
- `L3_neighborhood.outgoing_edges[*].target_qid`
- `L3_neighborhood.outgoing_edges[*].target_label`
- `L3_neighborhood.outgoing_edges[*].target_description`

Minimal shape:

```json
{
  "repair_Q42_123456789": {
    "L1_ego_node": {},
    "L2_labels": {},
    "L3_neighborhood": {},
    "L4_constraints": {}
  }
}
```

## Stage 4: `data/04_classified_benchmark.jsonl`

The LEAN Stage 4 artifact clones the Stage 2 repair record and adds:

- `classification`
- `context_ref`
- `labels_en`
- `build`
- `popularity` copied or backfilled from the popularity artifact when available

`context_ref` points back to the world-state entry in `03_world_state.json`.

The canonical classification fields are:

- `classification.class` in `{TypeA, TypeB, TypeC, T_BOX}`
- `classification.subtype`
- `classification.confidence`
- `classification.decision_trace`
- `classification.rationale`
- `classification.constraint_types`
- `classification.diagnostics`
- `classification.local_subtype`

After the delta-hardening pass, A-box diagnostics include:

- `classification.diagnostics.value_change_summary`
- `classification.diagnostics.classification_target_tokens`

`value_change_summary.semantic_action` is one of `CREATE_FROM_MISSING`, `DELETE_TO_MISSING`, `DELETE_SUBSET`, `ADD_SUPERSET`, `REPLACE_1_TO_1`, `MULTIPLICITY_DECREASE_SAME_UNIQUE`, `MULTIPLICITY_INCREASE_SAME_UNIQUE`, `MULTIPLICITY_CHANGE_SAME_UNIQUE`, `MIXED_UPDATE`, or `NO_CHANGE_OR_REORDER_ONLY`.

Local-match diagnostics use explicit source names such as `FOCUS_LABEL`, `FOCUS_DESCRIPTION`, `FOCUS_PREREPAIR_TARGET_PROPERTY_QID`, `FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL`, `FOCUS_QID`, `NEIGHBOR_ID`, and `NEIGHBOR_LABEL`. Synthetic pre-repair target-property values are not reported as generic `FOCUS_TEXT`.

Important runtime behavior:

- T-box records are emitted as class `T_BOX`, not `UNKNOWN`
- missing world-state entries still produce a Stage 4 record, defaulting to low-confidence `TypeC/UNKNOWN_MISSING_WORLD_STATE` after the Phase B redesign

## Stage 4 FULL: `data/04_classified_benchmark_full.jsonl`

The FULL variant is identical except it embeds the world-state payload directly under `world_state` instead of using only `context_ref`.

Current operational convention: `data/04_classified_benchmark.jsonl` is the canonical Stage 4 input for selection, audit, and tests. When classifier regeneration is run with `--no-full-output`, any existing `data/04_classified_benchmark_full.jsonl` should be treated as stale/deprecated until explicitly regenerated.

Use cases:

- offline experiment bundles
- archival exports
- environments where `03_world_state.json` is not available separately

## Stage 5: `data/05_splits.json`

The splitter reads Stage 4 JSONL and writes a deterministic summary containing:

- input provenance
- split policy
- overall counts
- sorted `train`, `dev`, and `test` id lists

Current stratification uses:

- `classification.class`
- `track`
- popularity bucket derived from the Stage 4 popularity score

## Selection Manifest: `reports/benchmark_selection/*.json`

The deterministic subset selector writes JSON manifests rather than creating alternate Stage 4 benchmark files. Phase C defines two primary manifests:

- `reports/benchmark_selection/dev_prompt_v1_seed_13.json`
- `reports/benchmark_selection/core_v1_seed_13.json`

Required top-level fields:

- `manifest_type`: `benchmark_selection`
- `manifest_version`: `phase_c_v1`
- `tier`: `core` or `dev`
- `seed`
- `created_at_utc`
- `inputs.classified_benchmark`
- `inputs.exclude_manifest`
- `policy`
- `selected_case_ids`
- `main_score_case_ids`
- `diagnostic_case_ids`
- `case_annotations`
- `counts`
- `underfilled_quotas`
- `warnings`
- `validation`

`case_annotations` maps each selected case id to derived selection metadata:

- `case_id`
- `tier`
- `track`
- `class`
- `subtype`
- `confidence`
- `selection_stratum`
- `analysis_slice`
- `main_score`
- `diagnostic_only`
- `group_key`
- `tbox_revision_key`
- `weak_group_key`
- `popularity_bucket`
- `constraint_family`
- `decision_constraint_type_qid`
- `decision_constraint_type_label`
- `decision_constraint_source`
- `classification_rule_family`
- `classification_rule_subfamily`
- `truth_source`
- `truth_token_kind`

The current core policy treats refined labels such as `FORMAT_NORMALIZATION`, `FORMAT_VALUE_PRUNING`, `SELF_LINK_REJECTION`, `SET_MEMBERSHIP_REJECTION`, `TARGET_REQUIRED_CLAIM`, `MULTIPLICITY_NORMALIZATION`, `LOCAL_TEXT_CONFIRMED`, `LOCAL_TEXT_DERIVED`, and `LOCAL_SELECTION_CONFIRMED` as main-score candidates when confidence is not low. Diagnostic labels such as `DELETE_AMBIGUOUS`, `UNKNOWN_SELECTION_AMBIGUOUS`, `UNKNOWN_MULTIPLICITY_ARTIFACT`, `UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED`, `UNKNOWN_BAD_TARGET_OR_CONTEXT`, `UNKNOWN_FOCUS_QID_DOMAIN_REASONING`, other `UNKNOWN_*` TypeC labels, `COINCIDENTAL_SCHEMA_CHANGE`, and `UNKNOWN_TBOX_CAUSALITY` are excluded from `main_score_case_ids`.

`constraint_family` is retained for backward compatibility and may describe a property-level or first-observed constraint family rather than the rule that decided the classifier label. For rule-family analyses, use `classification_rule_family`, `classification_rule_subfamily`, and `decision_constraint_type_qid`.

For T-box records, `classification.decision_trace` includes a `tbox_causality` step with `selected_violation_name`, `candidate_violation_names`, `candidate_violation_mappings_preview`, mapped-report constraint fields, changed target-constraint fields, semantic and ignored qualifier-change fields, compatible and incompatible overlap fields, active direction fields, and optional potential direction fields. Lean Stage 4 also stores `classification.diagnostics.tbox_diff_summary` as the compact replacement for pruned full constraint signatures.

T-box qualifier changes are split into semantic and ignored groups before polarity analysis. Metadata/status qualifiers such as `P2316` can mark that a constraint changed, but they are not semantic added/removed values and should not be used as directional evidence. The coarse public subtype remains available for compatibility. Active `directional_subtype_precise` and directional `analysis_slice_precise` are populated only when the final public subtype is directional; non-directional `SCHEMA_UPDATE` records use `main_tbox_schema_update` and may expose `potential_directional_*` fields for debugging only.

The manifest must distinguish headline evaluation cases from diagnostic/challenge cases. The main paper score should use `main_score_case_ids`, while `diagnostic_case_ids` should be reported separately.

Phase C uses deterministic SHA-1 ordering within each stratum:

```text
sha1(seed|tier|selection_stratum|group_key|case_id)
```

The selector writes manifests from `data/04_classified_benchmark.jsonl`; it must not create a second Stage 4 benchmark file. `EXTERNAL_BY_ELIMINATION` is reported as IC-E-elim/no-retrieval stress, while `UNKNOWN_*` TypeC cases are IC-U diagnostics.

Hard validation checks for `core_v1`:

- no duplicate selected ids;
- no case-id overlap with `dev_prompt_v1`;
- no T-box property-revision overlap with `dev_prompt_v1`;
- max 10 T-box cases per property revision;
- no `UNKNOWN_*` TypeC cases in `main_score_case_ids`;
- no low-confidence cases in `main_score_case_ids` unless explicitly upgraded by policy;
- no `DELETE_AMBIGUOUS`, `COINCIDENTAL_SCHEMA_CHANGE`, or `UNKNOWN_TBOX_CAUSALITY` cases in `main_score_case_ids`.

## Manual Audit Artifacts: `reports/manual_audit/*`

Phase D writes audit artifacts for validating high-risk classifier decisions:

- `audit_phase_d_v1_seed_13.jsonl`
- `audit_phase_d_v1_seed_13.csv`
- `audit_annotation_schema.json`
- `audit_phase_d_v1_results.json`
- `audit_phase_d_v1_summary.md`

Audit rows include prefilled case metadata, classifier diagnostics, and empty human annotation fields. The required annotation fields are documented in `00-manual_audit_phase_D.md`.

## Proposal Artifacts

Two normalized proposal contracts are now implemented:

- `schemas/verified_repair_proposal.schema.json` for A-box entity repairs
- `schemas/tbox_reform_proposal.schema.json` for T-box schema reforms
- `schemas/track_diagnosis.schema.json` for A-box vs T-box diagnosis outputs

Normalized A-box proposal JSONL records contain:

- `case_id`
- `target.qid`
- `target.pid`
- `ops`
- required `rationale`, `provenance`, `uncertainty`
- optional `metadata`
- `canonical_hash`

Normalized T-box proposal JSONL records contain:

- `case_id`
- `target.pid`
- `target.constraint_type_qid`
- `proposal.action`
- `proposal.signature_after`
- required `rationale`, `provenance`, `uncertainty`
- optional `metadata`
- `canonical_hash`

Normalized track-diagnosis JSONL records contain:

- `case_id`
- `predicted_track` in `{A_BOX, T_BOX, AMBIGUOUS}`
- optional `confidence`
- optional `rationale`
- `canonical_hash`

## Evaluation Artifacts

`src/evaluate.py` writes:

- `reports/evaluation_traces.jsonl`
- `reports/evaluation_summary.json`

It also accepts an optional selection manifest so evaluation can be restricted to a frozen subset without rewriting Stage 4.

Each trace includes:

- case identity and benchmark labels
- proposal presence, validity, and executability
- acceptance decision
- exact-match and semantic-match comparison fields
- for T-box traces, family-level semantic fields such as `comparison.semantic_family_match`, `comparison.target_constraint_match`, `comparison.literal_action_match`, and `metrics.semantic_family_success`
- track-diagnosis fields including predicted track, historical track, and exact-track-match
- metric fields including auditability completeness, conversion rate, and tokens-to-fix

The summary aggregates results by:

- class
- subtype
- track
- ablation bundle
- popularity bucket

## Reasoning-Floor Artifacts

`src/reasoning_floor.py` writes a run directory under `reports/reasoning_floor/` containing:

- a top-level directory named `<run_id>_<provider>_<model>`
- `raw_model_responses.jsonl`
- `run_manifest.jsonl`
- `run_config.json`
- one subdirectory per ablation bundle
- normalized proposal JSONL files per bundle
- evaluation traces and summaries per bundle
- `reasoning_floor_summary.json`

When the runner uses `--execution-mode batch`, the same run directory also includes:

- `batch_input.jsonl`
- `batch_request_manifest.jsonl`
- provider-specific batch job metadata and downloaded output or error files when the provider exposes them
- per-phase batch artifacts such as `diagnosis_*` and `proposal_*` when `--proposal-track-mode diagnosis_routed` is used

`run_manifest.jsonl` includes per-call provider, model, token usage, cached token counts when available, elapsed seconds when available, estimated cost when pricing metadata is configured, and cost-estimation metadata including whether batch pricing was applied. Recovered batch rows may also include a `recovery` object describing a synchronous retry after a retryable batch failure.

`run_config.json` stores the stable run configuration, including provider/model choice, the optional OpenAI reasoning-effort setting, execution mode, proposal-track mode, selected case ids, and selected generation artifact path when present. The runner uses this file to validate `--resume-run-dir` invocations before it appends new results into an interrupted run directory.

`reasoning_floor_summary.json` includes aggregated run-level token totals, cached token totals when available, estimated cost, elapsed time, provider, model, the OpenAI reasoning-effort setting when applicable, execution mode, an explicit `run_info.batch_mode_used` flag, output directory, cost-estimation metadata, and input references including the optional selection manifest path. The same OpenAI reasoning-effort value is also stored in the summary input block when present. OpenAI batch calls apply a built-in `0.5` cost-estimation multiplier. If some batch failures are retried synchronously, the summary reports `usage.cost_estimation_mode: "mixed"` and includes both `usage.per_call_cost_estimation_modes` and `usage.per_call_cost_estimation_multipliers`. Batch runs also include provider batch metadata under `run_info.batch`, including `run_info.batch.sync_retry_fallback` and per-phase `run_info.batch.phases[*].sync_retry_fallback`. Resumed runs also include `run_info.resume` metadata describing the reused run directory and the amount of generation work already completed before the resumed process began.

The combined and per-bundle evaluation summaries now also expose:

- `request_errors.proposal_request_error_count`
- `request_errors.proposal_request_error_rate`
- `request_errors.track_diagnosis_request_error_count`
- `request_errors.track_diagnosis_request_error_rate`
- A-box diagnostic rates such as `a_box_exact_action_match_rate`, `a_box_exact_value_match_rate`, and `a_box_regression_pass_rate`
- T-box proxy rates such as `t_box_target_constraint_match_rate`

## Schema Files in `schemas/`

Three schema files exist in the repository:

- `schemas/04_classified_benchmark.schema.json`: intended to describe the lean Stage 4 artifact, but it is not kept in lockstep with the current Python output
- `schemas/verified_repair_proposal.schema.json`: implemented by `guardian.patch_parser`
- `schemas/tbox_reform_proposal.schema.json`: implemented by `guardian.tbox_parser`
- `schemas/track_diagnosis.schema.json`: implemented by `guardian.track_parser`

Treat `schemas/04_classified_benchmark.schema.json` as a design asset unless it is brought back into sync with the classifier output.

## Related Docs

- Classification behavior: [Classifier Specification](./Classifier_Specification.md)
- Pipeline and command flow: [Pipeline Implementation](./Pipeline_Implementation.md)
- Proposal contracts: [Proposal Validation](./Proposal_Validation.md)
- Evaluation semantics: [Evaluation Harness](./Evaluation_Harness.md)
- Reasoning-floor outputs: [Reasoning Floor](./Reasoning_Floor.md)
- Track-diagnosis task: [Track Diagnosis](./Track_Diagnosis.md)
