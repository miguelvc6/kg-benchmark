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

Important runtime behavior:

- T-box records are emitted as class `T_BOX`, not `UNKNOWN`
- missing world-state entries still produce a Stage 4 record, defaulting to low-confidence `TypeC/EXTERNAL`

## Stage 4 FULL: `data/04_classified_benchmark_full.jsonl`

The FULL variant is identical except it embeds the world-state payload directly under `world_state` instead of using only `context_ref`.

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

The deterministic paper-subset selector writes a small JSON manifest rather than a second full benchmark artifact.

Current manifest fields include:

- `manifest_type`
- `manifest_version`
- `inputs.classified_benchmark`
- `policy.scope`
- `policy.selection_strategy`
- `policy.t_box_group_key`
- `policy.tbox_cap_per_update`
- `policy.seed`
- `policy.stable_ordering`
- aggregate `counts`
- `t_box_selected_counts_by_revision`
- `selected_case_ids`

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
- optional `rationale`, `provenance`, `metadata`
- `canonical_hash`

Normalized T-box proposal JSONL records contain:

- `case_id`
- `target.pid`
- `target.constraint_type_qid`
- `proposal.action`
- `proposal.signature_after`
- optional `rationale`, `provenance`, `metadata`
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
- track-diagnosis fields including predicted track, historical track, and exact-track-match
- metric fields including reserved nulls for later multi-turn runs

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
- one subdirectory per ablation bundle
- normalized proposal JSONL files per bundle
- evaluation traces and summaries per bundle
- `reasoning_floor_summary.json`

When the runner uses `--execution-mode batch`, the same run directory also includes:

- `batch_input.jsonl`
- `batch_request_manifest.jsonl`
- provider-specific batch job metadata and downloaded output or error files when the provider exposes them

`run_manifest.jsonl` includes per-call provider, model, token usage, cached token counts when available, elapsed seconds when available, estimated cost when pricing metadata is configured, and cost-estimation metadata including whether batch pricing was applied.

`reasoning_floor_summary.json` includes aggregated run-level token totals, cached token totals when available, estimated cost, elapsed time, provider, model, execution mode, an explicit `run_info.batch_mode_used` flag, output directory, cost-estimation metadata, and input references including the optional selection manifest path. OpenAI batch runs apply a built-in `0.5` cost-estimation multiplier. Batch runs also include provider batch metadata under `run_info.batch`.

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
