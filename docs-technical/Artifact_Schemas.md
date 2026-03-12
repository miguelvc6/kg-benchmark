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

## Schema Files in `schemas/`

Two schema files exist in the repository, but they do not both describe implemented runtime behavior:

- `schemas/04_classified_benchmark.schema.json`: intended to describe the lean Stage 4 artifact, but it is not kept in lockstep with the current Python output
- `schemas/verified_repair_proposal.schema.json`: defines a future repair-proposal format, but no production code in this repository currently emits or validates that proposal format

Treat these files as design assets unless and until the corresponding runtime modules exist and are wired into the pipeline.

## Related Docs

- Classification behavior: [Classifier Specification](./Classifier_Specification.md)
- Pipeline and command flow: [Pipeline Implementation](./Pipeline_Implementation.md)
- Conceptual mismatches: [Conceptual Deviation Report](./Conceptual_Deviation_Report.md)
