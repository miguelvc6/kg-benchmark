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

- `fetcher.py`: stages 1-3 (index, fetch, world state build)
- `classifier.py`: stage 4 taxonomy labeler
- `data/`: generated artifacts (large)
- `data_sample/`: sample artifacts for quick runs (large)
- `reports/`: classifier stats and run summaries
- `docs/`: design notes, pipeline details, taxonomy specification

## Quick start

Setup:

```bash
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt
```

Run the full pipeline (stages 1-3):

```bash
python fetcher.py
```

Debug run with a cap:

```bash
python fetcher.py --max-candidates 100
```

Validate an existing world state:

```bash
python fetcher.py --validate-only
```

Run the classifier (stage 4) on sample data:

```bash
python classifier.py --sample
```

Minimal self-test for classifier logic:

```bash
python classifier.py --self-test
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
- Stage 3 requires the `data/latest-all.json.gz` Wikidata dump.
- See `docs/` for detailed protocol and taxonomy documentation.
