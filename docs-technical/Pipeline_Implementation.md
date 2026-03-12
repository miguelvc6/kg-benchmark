# Pipeline Implementation

This document describes the current repository workflow. Research motivation is documented separately in [docs-conceptual](../docs-conceptual/README.md).

## Current Repository Scope

The implemented repository currently provides:

- benchmark construction from report mining through classified benchmark generation
- deterministic train/dev/test split generation
- schema and test assets for a future repair-proposal interface

The repository does not currently provide:

- an executable benchmark evaluation harness
- a reasoning-floor baseline runner
- a Guardian or verifier loop
- an implemented `guardian.patch_parser` module, even though related tests and schema files exist

## Entry Points

- `src/fetcher.py`: stages 1-3 of benchmark construction plus world-state validation.
- `src/classifier.py`: stage 4 taxonomy labeling and benchmark materialization.
- `src/splitter.py`: stage 5 deterministic train/dev/test splits.

## Current Pipeline

| Stage | Script | Purpose | Primary outputs |
| --- | --- | --- | --- |
| 1 | `src/fetcher.py` | Mine Wikidata constraint-report diffs for candidate repairs | `data/01_repair_candidates.json` |
| 2 | `src/fetcher.py` | Reconstruct the atomic repair event and persistence metadata | `data/02_wikidata_repairs.jsonl`, `data/02_wikidata_repairs.json` |
| 3a | `src/fetcher.py` | Attach deterministic popularity metadata | `data/00_entity_popularity.json` |
| 3b | `src/fetcher.py` | Build and validate frozen world-state context | `data/03_world_state.json` |
| 4 | `src/classifier.py` | Assign Type A/B/C labels and write benchmark records | `data/04_classified_benchmark.jsonl`, `data/04_classified_benchmark_full.jsonl`, `reports/classifier_stats.json` |
| 5 | `src/splitter.py` | Create deterministic train/dev/test splits from Stage 4 output | `data/05_splits.json` |

The stage numbering above reflects the actual code flow. There is no separate implemented evaluation stage after `src/splitter.py`.

## Stage 1: Candidate Mining

Stage 1 mines revision histories of `Wikidata:Database reports/Constraint violations/*`.

Current behavior:

- load `data/01_repair_candidates.json` if it already exists and is non-empty
- otherwise auto-discover properties from the Wikidata constraint-violations summary page unless `TARGET_PROPERTIES` is set explicitly
- detect QIDs that disappear between adjacent report snapshots
- discard section reclassifications by checking whether the QID still exists elsewhere on the page
- preserve report provenance for later debugging
- deduplicate candidate records before Stage 2 while keeping the associated violation-type evidence

## Stage 2: Repair Reconstruction

Stage 2 resolves each candidate into one repair event.

### A-box path

- scan entity history inside the configured lookback window (`REVISION_LOOKBACK_DAYS`)
- compare property signatures across revisions
- store the latest in-window change as the repair target
- emit action, old/new value snapshots, revision id, and author

### T-box path

- if no entity-level repair is found, scan the property entity history
- detect `P2302` signature changes
- store deterministic before/after hashes and optional serialized signatures in `constraint_delta`
- emit T-box records with ids shaped as `reform_{qid}_{pid}_{property_revision_id}`

### Ambiguity guard

When an A-box repair is found, the fetcher runs a bounded reverse scan over the property history to detect coincident T-box changes and marks ambiguous cases explicitly.

### Persistence checks

Stage 2 fetches the live 2026 state after the repair type is known.

- delete actions may legitimately remain absent
- non-delete repairs must still exist in the live graph
- failed persistence checks are dropped before downstream artifacts are built
- T-box repairs are also dropped when the current entity/property state cannot be fetched and strict persistence is enabled

### Operational details

- Stage 2 writes append-only JSONL during execution, then compiles to JSON.
- Resume support exists through `--resume-stats` and `--resume-checkpoint`.
- Per-candidate diagnostics are buffered into `logs/fetcher_stats_<run>.jsonl`.
- Caches live under `data/cache/`, including `labels_en.sqlite` and `entity_snapshots.sqlite`.
- A per-run summary is written to `logs/run_summary_<run>.json`.
- `src/fetcher.py --validate-only` validates only `data/03_world_state.json` against `data/02_wikidata_repairs.json`.

## Stage 3: Popularity and World State

After Stage 2, the fetcher enriches repair entries with popularity metadata and builds the frozen world-state artifact.

### Popularity artifact

`data/00_entity_popularity.json` maps each focus QID to a popularity block. The current score combines:

- `pageviews_365d`
- `out_degree`
- `sitelinks_count`

The artifact also stores percentile-normalized versions of those components and request metadata such as:

- pageviews project, access, agent, and granularity
- pageview window start and end
- `wiki`

The normalized composite score is later copied into Stage 2 and Stage 4 records.

### World-state builder

The world-state builder in `src/lib/world_state.py` streams `data/latest-all.json.gz` and materializes only target entities.

Each entry includes:

- `L1_ego_node`
- `L2_labels`
- `L3_neighborhood`
- `L4_constraints`
- optional `constraint_change_context` for T-box cases

Implementation details that matter for downstream tooling:

- `L1_ego_node` includes `qid`, `label`, `description`, `sitelinks_count`, and `properties`
- `L1_ego_node` also includes `popularity` when the Stage 2 record already carries it
- `L2_labels.entities` is a flat id-indexed map built from the focus node, property, neighborhood, and constraint references
- `L4_constraints.constraints[*]` stores `constraint_type`, `rank`, `snaktype`, `qualifiers`, and a human-readable `rule_summary`

The builder validates uniqueness and schema consistency before `data/03_world_state.json` is finalized.

## Stage 4: Classification

`src/classifier.py` reads Stage 2 repairs and Stage 3 world state, then writes:

- LEAN benchmark records that reference `03_world_state.json`
- FULL benchmark records that embed the world-state payload
- `reports/classifier_stats.json`

Classification is offline and deterministic for fixed inputs. No live web calls are made during this stage.

Implementation details that are easy to miss:

- the classifier builds a SQLite sidecar index next to `03_world_state.json` for keyed lookup during classification
- missing world-state entries are classified as low-confidence `TypeC/EXTERNAL`
- T-box entries are not mapped to `UNKNOWN`; they receive class `T_BOX` and a schema-change subtype
- each Stage 4 record gets a `build` block with classifier version and build timestamp

The decision logic itself is documented in [Classifier Specification](./Classifier_Specification.md).

## Stage 5: Splits

`src/splitter.py` creates `data/05_splits.json` from the classified benchmark.

Current stratification dimensions:

- `classification.class`
- `track`
- popularity bucket derived from the Stage 4 popularity score

The splitter raises an error if split proportions drift beyond the configured tolerance or if popularity is missing while `ALLOW_MISSING_POPULARITY` is `False`.

## Verification Notes

Executable checks currently confirmed from the repository:

- `uv run python src/classifier.py --self-test` passes

Repository limitations observed during verification:

- there is no installed `pytest` entry point in the current environment
- `uv run python -m unittest tests/test_patch_parser.py` fails because `guardian.patch_parser` does not exist in the repository

## Common Commands

```bash
uv run python src/fetcher.py
uv run python src/fetcher.py --max-candidates 100
uv run python src/fetcher.py --resume-stats logs/fetcher_stats_YYYYMMDDTHHMMSS.jsonl
uv run python src/fetcher.py --resume-checkpoint logs/resume_checkpoint_YYYYMMDDTHHMMSS.json
uv run python src/fetcher.py --reuse-popularity-artifact
uv run python src/fetcher.py --validate-only
uv run python src/classifier.py --sample
uv run python src/classifier.py --self-test
uv run python src/splitter.py --sample
uv run python -m unittest tests/test_patch_parser.py
```

## Related Docs

- Artifact structure: [Artifact Schemas](./Artifact_Schemas.md)
- Conceptual benchmark framing: [docs-conceptual/Benchmark_Taxonomy.md](../docs-conceptual/Benchmark_Taxonomy.md)
- Scope mismatch summary: [Conceptual Deviation Report](./Conceptual_Deviation_Report.md)
