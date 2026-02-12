# kg-benchmark

WikidataRepairEval 1.0: a benchmark of real Wikidata repair events with frozen
context for evaluating knowledge graph repair and retrieval needs. The pipeline
reconstructs historical fixes, attaches a 2026 world-state snapshot, and labels
each case by information necessity (Type A/B/C).

## Pipeline overview

Stage 1 - Indexer: find candidate repairs from Wikidata constraint report diffs.
Stage 2 - Fetcher: locate the atomic edit and capture provenance (A-box or T-box).
Stage 3 - Context builder: freeze 2026 graph context (L1-L4).
Stage 4 - Classifier: assign Type A/B/C and emit audit traces.

## Repo layout

- `src/fetcher.py`: stages 1-3 (index, fetch, world state build)
- `src/classifier.py`: stage 4 taxonomy labeler
- `src/lib/`: shared pipeline modules
- `data/`: generated artifacts (large)
- `data_sample/`: sample artifacts for quick runs (large)
- `reports/`: classifier stats and run summaries
- `docs/`: design notes, pipeline details, taxonomy specification

## Quick start

Setup (uv + pyproject.toml):

```bash
# Windows
set UV_PROJECT_ENVIRONMENT=.venv
uv sync

# WSL/Linux
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync
```

Run the full pipeline (stages 1-3):

```bash
uv run python src/fetcher.py
```

Debug run with a cap:

```bash
uv run python src/fetcher.py --max-candidates 100
```

Resume an interrupted fetcher run:

```bash
# Preferred: resume from a prior stats log (new runs include candidate_key)
uv run python src/fetcher.py --resume-stats logs/fetcher_stats_YYYYMMDDTHHMMSS.jsonl

# Resume from a checkpoint file (written every 5k candidates by default)
uv run python src/fetcher.py --resume-checkpoint logs/resume_checkpoint_YYYYMMDDTHHMMSS.json

# Reuse existing popularity artifact instead of rebuilding data/00_entity_popularity.json
uv run python src/fetcher.py --reuse-popularity-artifact
```

Validate an existing world state:

```bash
uv run python src/fetcher.py --validate-only
```

Run the classifier (stage 4) on sample data:

```bash
uv run python src/classifier.py --sample
```

Minimal self-test for classifier logic:

```bash
uv run python src/classifier.py --self-test
```

## Key artifacts

- `data/01_repair_candidates.json`: candidate repair events from report diffs
- `data/02_wikidata_repairs.json(.jsonl)`: atomic repair events with provenance
- `data/03_world_state.json`: frozen 2026 context keyed by repair id
- `data/04_classified_benchmark.jsonl`: labeled benchmark (lean)
- `data/04_classified_benchmark_full.jsonl`: labeled benchmark with embedded context
- `reports/classifier_stats.json`: summary counts and diagnostics

Sample outputs live in `data_sample/`, plus a small `classified_benchmark_sample.json`
at repo root for inspection.

## Notes

- The fetcher hits the live Wikidata API and can take days. It writes large
  artifacts and caches under `data/cache/`.
- Fetcher runs write resume checkpoints to `logs/resume_checkpoint_<run>.json`
  unless `--no-checkpoint` is set.
- Stage 3 requires the `data/latest-all.json.gz` Wikidata dump.
- See `docs/` for detailed protocol and taxonomy documentation.
