# Confidence Frequency Report

`scripts/confidence_frequency_report.py` streams a Stage 4 classified benchmark
JSONL artifact and counts `classification.confidence` values.

## Default Usage

```bash
uv run python scripts/confidence_frequency_report.py
```

Defaults:

- input: `data/04_classified_benchmark.jsonl`
- output: `reports/confidence_frequency_report.json`

## Custom Usage

```bash
uv run python scripts/confidence_frequency_report.py \
  --input data/04_classified_benchmark.jsonl \
  --output reports/confidence_frequency_report.json
```

Use `--progress-every 0` to disable progress output.

## Report Structure

The JSON report contains:

- `overall`: total confidence counts and fractions across all records
- `strata.by_class`: counts grouped by `classification.class`
- `strata.by_track`: counts grouped by `track`
- `strata.by_subtype`: counts grouped by `classification.subtype`
- `strata.by_class_and_track`: counts grouped by class and track
- `strata.by_class_and_subtype`: counts grouped by class and subtype
- `missing_classification_lines`: input lines where the classification block could not be found

The script streams the input file, extracts only the needed fields, and only
keeps counters in memory, so it is safe to run on the full Stage 4 artifact.
