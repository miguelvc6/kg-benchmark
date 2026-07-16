# Artifact Reference

The one final dataset contains `manifest.json`; canonical popularity, candidate, repair, and world-state JSONL under
`source/`; `cases.jsonl`; source provenance and lineage manifests; audit dispositions and summary; ranking, prior-group
exclusions, eligibility orders, support bank, reserve, prompt audit, replacement records, and evaluation-population
manifests; the frozen protocol and methodology lock; and every schema needed to validate those published roles.
`dataset-manifest.schema.json` binds each role to its canonical path, byte size, record count where applicable, and
SHA-256. Manifest version 2 also binds the protocol, methodology freeze scope and source revision, source provenance,
lineage, and successful semantic and byte-reproduction gates.

Source provenance additionally binds the Stage-2 upstream-exclusion artifact and its record count. This construction
diagnostic remains separate from the repair and case populations: each row documents a candidate that could not be
reconstructed after transient API retries were exhausted, and no row is treated as negative repair evidence.

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
`selection-manifest.schema.json`; promotion additionally validates the ranking, eligibility-order, support-bank, and
replacement row contracts. Version-2 population manifests contain file-level provenance hashes and one canonical
eligibility-decision hash per selected case. A later extension adds a parent-population hash and an explicit nesting
relationship.

`source-provenance.json` keeps the original construction-file and dump hashes, completed acquisition configuration,
cache inventory, and acquisition Git revision. `lineage.json` proves Stage 0/1 provenance, exact Stage 2 representation
equivalence, exact Stage 2/3/4 case identity, and the Stage 2 projection of every Stage 4 case. The final release keeps
only canonical JSONL source roles; promotion additionally compares their Stage 0, Stage 1, and Stage 3 content with the
original construction JSON. Each published case refers to the world state by the portable dataset-relative path
`source/world-state.jsonl`, never by a construction-machine path. The original JSON and cache remain outside the
published dataset.

The Python distribution includes the active schemas and frozen prompt templates as package data. Runtime resource
lookup supports both a repository checkout and an installed wheel, so the paper-facing CLI does not depend on the
caller's working directory or on untracked files.

Generation identity is independent of population membership. It contains case-payload, rendered-prompt, context,
model-revision, inference-parameter, and task hashes. Run manifests record which population requested those generation
keys. This permits population expansion and metric replay without submitting existing requests again.

Execution planning writes `runs/matrices/<matrix-id>/matrix.json` and `requests.jsonl`. The matrix binds the final
dataset inputs, model and protocol files, every population, the complete logical-cell cross-product, physical execution
groups, and workload totals. Each request row records the six audit identities plus the exact cache request
specification; unresolved revisions have a null request key and cannot execute. The active contracts are
`model-matrix.schema.json`, `experiment-matrix.schema.json`, and `matrix-request.schema.json`.

Physical runner artifacts live under `executions/<group-id>/`. Hash-bound logical-cell manifests live under `cells/`
and use `matrix-cell.schema.json`; they bind the matrix and request plan, runner configuration, runner manifest, summary,
observed completion/failure counts, and cache coverage. These matrix artifacts and the shared SQLite generation cache
are operational bulk under ignored `runs/`, not part of the published dataset.

Each physical group may contain immutable `evaluations/<evaluation-id>/` directories. Their
`evaluation_manifest.json` uses `evaluation-replay.schema.json`, declares zero provider calls, and binds the source run,
evaluator implementation, separate A-box/T-box/diagnosis traces and summaries, and combined display summary. A new
metric definition creates a new evaluation ID rather than mutating an existing directory.

Compact paper outputs live under `results/<analysis-id>/`. `analysis-result-manifest.schema.json` binds the matrix,
evaluation manifests, `paper/analysis.json`, analysis workflow/statistics code, dependency lock, machine-readable rows,
and Markdown tables. `analysis-estimate.schema.json`, `analysis-contrast.schema.json`,
`analysis-diagnosis.schema.json`, and `analysis-summary.schema.json` validate the output rows. Raw generations and
evaluation traces remain under ignored `runs/` and are never copied into `results/`.

Bulk data and raw generations are distributed externally. Git contains manifests, retrieval metadata, compact results,
schemas, protocol, code, documentation, and a tiny synthetic example.
