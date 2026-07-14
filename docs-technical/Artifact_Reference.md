# Artifact Reference

The one final dataset contains `manifest.json`, canonical JSONL under `source/`, `cases.jsonl`, audit dispositions and
summary, eligibility ordering, support bank, and evaluation-population manifests. `dataset-manifest.schema.json` binds
each role to path, size, record count where applicable, and SHA-256.

Generation identity is independent of population membership. It contains case-payload, rendered-prompt, context,
model-revision, inference-parameter, and task hashes. Run manifests record which population requested those generation
keys. This permits population expansion and metric replay without submitting existing requests again.

Bulk data and raw generations are distributed externally. Git contains manifests, retrieval metadata, compact results,
schemas, protocol, code, documentation, and a tiny synthetic example.
