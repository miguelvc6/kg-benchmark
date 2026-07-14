# Artifact Reference

The one final dataset contains `manifest.json`, canonical JSONL under `source/`, `cases.jsonl`, audit dispositions and
summary, eligibility ordering, support bank, and evaluation-population manifests. `dataset-manifest.schema.json` binds
each role to path, size, record count where applicable, and SHA-256.

During construction, `work/audit/workflow.json` is the resumable audit control manifest. Preparation writes
`construct-sample.csv`, `rendered-prompts.jsonl`, and `render-summary.json`. Deterministic evidence and blinded review
packets live under `work/audit/deterministic/`. Finalization writes the only promotion-facing audit artifacts directly
as `work/audit/dispositions.jsonl` and `work/audit/summary.json`; `audit.md` is rendered from that summary.

`audit-workflow.schema.json`, `rendered-audit-prompt.schema.json`, `audit-disposition.schema.json`, and
`audit-summary.schema.json` are the active contracts. The workflow manifest records the exact Codex reviewer model and
CLI version and hash-binds every input, schema, sample, prompt render, review artifact, disposition, and report.

Selection construction is controlled by `work/selections/selection-workflow.json`. The reserve phase writes
`ranking.json`, `eligibility-order.jsonl`, `support-bank.json`, `reserve.json`, `reserve-prompts.jsonl`, render failures
and summary, the deterministic pre-review prompt audit, and 50 blinded temporal packets. Review writes a Codex run
manifest and normalized review rows. Finalization writes `prompt-audit.json`, `per-case-eligibility.jsonl`,
`prompt-clean-eligibility-order.jsonl`, `replacements.jsonl`, `main-1200.json`, and `azure-600.json`.

The active selection contracts are `group-exclusions.schema.json`, `reserve-manifest.schema.json`,
`per-case-eligibility.schema.json`, `selection-prompt-audit.schema.json`, `selection-workflow.schema.json`, and
`selection-manifest.schema.json`. Version-2 population manifests contain file-level provenance hashes and one canonical
eligibility-decision hash per selected case. A later extension adds a parent-population hash and an explicit nesting
relationship.

Generation identity is independent of population membership. It contains case-payload, rendered-prompt, context,
model-revision, inference-parameter, and task hashes. Run manifests record which population requested those generation
keys. This permits population expansion and metric replay without submitting existing requests again.

Bulk data and raw generations are distributed externally. Git contains manifests, retrieval metadata, compact results,
schemas, protocol, code, documentation, and a tiny synthetic example.
