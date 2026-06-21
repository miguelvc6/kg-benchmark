# Few-Shot Evaluation Governance

This document fixes the operational policy for Phase G5 few-shot evaluation before any few-shot inference is run. It is implementation governance for prompt rendering, support-set construction, leakage controls, and result labeling.

## Role

The zero-shot oracle evaluation remains the main reasoning-floor result. Few-shot evaluation is not the main zero-shot floor and must not replace, relabel, or be merged into the zero-shot oracle baseline.

Few-shot runs are Phase G5 ablations. Their purpose is to measure precedent adaptation and contract-following changes after support examples are added, while preserving the frozen zero-shot result as the primary baseline.

The first paper-facing few-shot condition is static diverse support examples. Dynamic visible-similarity retrieval is exploratory unless later validation proves that its retrieval inputs are clean and reproducible. Hidden-metadata matched retrieval is an oracle-metadata appendix condition only, because it may use benchmark fields that are not model-visible in the main task.

## Example Source Policy

Few-shot examples must come from `reports/benchmark_selection/dev_prompt_v1_seed_13.json` unless a later technical document explicitly defines a different support pool and its leakage controls.

Core cases must never be used as examples. Core T-box revision groups must never be used as examples. Selection and validation code must block both direct case overlap and grouped overlap with `reports/benchmark_selection/core_v1_seed_13.json`.

Raw case identifiers such as `repair_...` and `reform_...` are internal metadata only. They must not appear in model-visible prompt text. Rendered prompts must use neutral example identifiers when an example needs a visible label.

## Manifest Separation

Few-shot prompt development distinguishes evaluation manifests from example manifests:

- `--eval-manifest` identifies cases to render, evaluate, and score.
- `--dev-manifest` is only a backward-compatible alias for `--eval-manifest`.
- `--example-manifest` identifies the candidate pool for selecting examples.
- `--core-manifest` identifies forbidden final-evaluation cases and T-box revision groups.
- `--support-set-manifest` identifies a fixed preselected support set when static examples are used.

Few-shot runs must not silently select examples from the evaluation manifest. If any example policy other than
`zero_shot` is active, prompt rendering and evaluation require `--example-manifest` or `--support-set-manifest` unless
`--allow-core-example-risk` is explicitly passed for a leakage-risk experiment. Run summaries must record
`eval_manifest` and `example_manifest` as distinct fields.

## Prompt Validity

Few-shot inputs obey the same leakage boundary as zero-shot inputs. Support examples may show the prompt-visible input for the example and the correct task output for that example, but must not expose hidden benchmark fields, selection strata, raw case IDs, future-only gold metadata, or support-set roles.

The task schemas for few-shot runs are the same schemas used by the selected zero-shot prompts:

- A-box repair uses A-box v4 / `prompt_dev_v4_spec_only`.
- T-box repair uses T-box taxonomy-patch v5 / `prompt_dev_v5_tbox_taxonomy_patch`.

Example selection logic must be declared in the run artifacts. Static support sets must point to a fixed support-set manifest. Dynamic support selection must record the retrieval policy, candidate pool, blocked core manifest, and per-case selected example identifiers.

## Condition Taxonomy

`static_diverse_kshot` is the first paper-facing Phase G5 condition. It uses a preselected, auditable, immutable
support-set manifest built from the dev prompt manifest. Rendering this policy requires `--support-set-manifest`; the
first up to `k` examples in each task-specific support-set section are selected according to `--example-count`.

`visible_similarity_kshot` is an exploratory dynamic retrieval condition. It may use only model-visible input features for retrieval unless a later governance update narrows or expands the allowed feature set.

`matched_metadata_oracle_kshot` is an oracle-metadata appendix condition. It may use hidden subtype, action, information-condition, or related benchmark metadata for support matching, and must be reported separately from paper-facing static examples.

`diagnosis_static_kshot` is a diagnosis-routed rescue condition. It remains development-only unless the diagnosis gate defined for the evaluation phase passes and a later readiness review approves it for broader reporting.

## Static Support Schema

Static support-set manifests must validate against `schemas/few_shot_support_set.schema.json`. The schema separates
`a_box_repair`, `t_box_repair`, and `track_diagnosis` support sets, stores raw case IDs only as internal metadata, and
requires neutral visible example IDs such as `example_a_000001`, `example_t_000001`, and `example_d_000001`.

Generate the default static support set with:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run kg-select-few-shot-support \
  --classified-benchmark data/04_classified_benchmark.jsonl \
  --dev-manifest reports/benchmark_selection/dev_prompt_v1_seed_13.json \
  --core-manifest reports/benchmark_selection/core_v1_seed_13.json \
  --output-dir reports/prompt_dev/few_shot/static_support_v1 \
  --seed 13
```

This writes `static_support_manifest.json`, `support_selection_report.json`, and `support_selection_report.md`. The
selection report includes counts by track, class/subtype, T-box taxonomy code, property, QID, and T-box revision key,
plus an explicit statement about whether static examples share properties with core cases.

## Safe Example Payloads

Rendered support examples must use the task schema declared in the support manifest:

- `a_box_v4_spec_only` examples render canonical A-box proposal outputs with `target`, `ops`, `rationale`,
  `provenance`, and `uncertainty`.
- `tbox_taxonomy_patch_v1` examples render taxonomy-patch outputs with `schema_decision`, `target`, `repairs`,
  `rationale`, `provenance`, and `uncertainty`; strict-signature `proposal.signature_after` examples are not used.
- `track_diagnosis_v1` examples render diagnosis outputs with `predicted_track`, `confidence`, and `rationale`.

The renderer validates every support example output with the corresponding parser before writing prompt artifacts.
Rendered rows include `example_leakage_scan`, and rendering fails if example input or output payloads contain forbidden
hidden fields, raw benchmark case IDs, or core/dev selection labels.

## Leakage And Overlap Scans

Every few-shot render writes `few_shot_leakage_scan.json`, `few_shot_overlap_report.json`, and
`few_shot_overlap_report.md`. The leakage scan checks model-visible system/user prompt text and rendered example
payloads for hidden benchmark fields, raw `repair_...` / `reform_...` case IDs, `DEV_` / `CORE_` labels,
`sitelinks_count`, `popularity`, and hidden TypeA/TypeB/TypeC labels. Benign visible text matches are documented
separately from hard failures.

The overlap report checks support examples against evaluated cases and the core manifest by raw case ID, focus QID,
property, and T-box revision key. Core case overlap and core T-box revision overlap must be zero. Static support
property overlap is allowed only when disclosed; dynamic same-property overlap remains a failure unless explicitly
allowed with `--allow-same-property-examples`.

Static support rendering filters the fixed support set per evaluated case before prompt construction. Raw case, focus
QID, and T-box revision overlaps with the current evaluated case are skipped; disclosed same-property overlaps remain
allowed for static support. This lets a fixed 3-shot A-box / 4-shot T-box support set be reused safely across a dev
holdout manifest that may contain some support cases or sibling T-box revisions.

## Reporting

Evaluation runs that include any non-zero-shot example policy write these few-shot-specific artifacts:

- `few_shot_run_config.json`
- `few_shot_delta_vs_zero_shot.json`
- `few_shot_delta_vs_zero_shot.md`

The delta report matches each few-shot matrix to its zero-shot baseline by task, representation, context bundle, and
track mode. It reports A-box, T-box taxonomy-patch, and diagnosis comparisons in separate sections and intentionally
does not compute a combined A-box/T-box headline.

When a few-shot evaluation run includes only non-zero-shot policies, the evaluator can load an existing zero-shot
`prompt_dev_evaluation_summary.json` for delta reporting instead of rerunning the baseline. Pass
`--zero-shot-baseline-summary` to select that artifact explicitly. If the option is omitted for the standard v5
holdout static run and `reports/prompt_dev/evaluation_prompt_dev_v5_tbox_taxonomy_patch_holdout96_zero_shot/`
exists, that frozen zero-shot summary is used automatically.

`static_diverse_kshot` is labeled `static_support_set` and paper-facing. Per-case retrieval policies such as
`random_same_task_2shot`, `same_track_2shot`, and `matched_2shot` are labeled `dynamic_retrieval` and exploratory.
The Markdown report repeats this distinction so static few-shot ablations are not confused with dynamic retrieval
experiments.

Each comparison includes token, cost, and latency overhead from the matrix `run_manifest.jsonl` usage fields. Missing
provider usage remains `n/a` rather than being inferred.

T-box taxonomy-patch evaluation summaries also expose report-only diagnostics for confusion matrices, value-delta
display metrics, out-of-current-gold operation false positives, macro averages by property and T-box revision, and
subset label interpretation. These diagnostics do not change prompts or gold extraction.
