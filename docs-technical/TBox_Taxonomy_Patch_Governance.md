# T-Box Taxonomy Patch Governance

This document records the versioning and artifact-separation rules for the T-box taxonomy-patch migration.

## Version Labels

Use explicit labels so strict-signature T-box reconstruction and taxonomy-patch repair reasoning cannot be mixed accidentally.

| Field | Value | Meaning |
| --- | --- | --- |
| `tbox_task_version` | `strict_signature_v1` | Historical strict T-box full `signature_after` reconstruction task. |
| `tbox_task_version` | `tbox_taxonomy_patch_v1` | Prompt-dev taxonomy-patch T-box task version recorded in run manifests. |
| `metric_family` | `tbox_taxonomy_patch_v1` | Taxonomy-patch evaluator metric family. |
| `prompt_version` | `prompt_dev_v5_tbox_taxonomy_patch` | Prompt-dev version that keeps A-box v4 unchanged and routes T-box repairs to the taxonomy-patch contract. |
| `gold_version` | `tbox_taxonomy_patch_gold_dev_holdout_v1` | Dev holdout taxonomy-patch gold artifact version. |
| `gold_version` | `tbox_taxonomy_patch_gold_core_v1` | Core taxonomy-patch gold artifact version. |

The prompt-dev run manifest field is currently `tbox_task_version = tbox_taxonomy_patch_v1`. The implementation plan used the shorter label `taxonomy_patch_v1`; reports should use the emitted repository label and may state that it is the taxonomy-patch v1 task.

## Artifact Separation

Strict-signature artifacts remain in their existing locations. Taxonomy-patch prompt-dev outputs must use taxonomy-patch-specific directories, for example:

```text
reports/prompt_dev/evaluation_prompt_dev_v5_tbox_taxonomy_patch_holdout96_zero_shot/
reports/prompt_dev/evaluation_prompt_dev_v5_tbox_taxonomy_patch_core_tbox64_zero_shot/
reports/prompt_dev/evaluation_prompt_dev_v5_tbox_taxonomy_patch_core_tbox_all_zero_shot/
```

Taxonomy-patch predictions are written to:

```text
t_box_taxonomy_patch_proposals.jsonl
```

They must not be written to or interpreted as the old strict:

```text
t_box_proposals.jsonl
```

Each taxonomy-patch matrix writes:

```text
tbox_taxonomy_patch_evaluation_summary.json
```

That summary records `metric_family`, `gold_version`, and `strict_signature_metrics_role = diagnostic_only`.

## Metric Interpretation

Do not compare strict-signature and taxonomy-patch scores as if they are the same task.

- Strict-signature metrics ask whether a model reconstructed a full historical post-repair constraint signature.
- Taxonomy-patch metrics ask whether a model identified the repair family, schema decision, taxonomy operation, and visible value deltas.

Strict-signature metrics remain useful diagnostics, but they are not headline scores for taxonomy-patch runs.

## A-Box Separation

The taxonomy-patch migration does not change A-box repair. Under `prompt_dev_v5_tbox_taxonomy_patch`, A-box prompts still use the `prompt_dev_v4_spec_only` repair contract. T-box taxonomy-patch reports must not aggregate A-box and T-box scores into a single headline.

## Required Checks For New Runs

Before publishing a taxonomy-patch run:

- confirm the output directory is taxonomy-patch-specific;
- confirm `prompt_version = prompt_dev_v5_tbox_taxonomy_patch`;
- confirm proposal rows use `t_box_taxonomy_patch_proposals.jsonl`;
- confirm run manifests include `tbox_task_version = tbox_taxonomy_patch_v1`;
- confirm taxonomy evaluation summaries include the expected `gold_version`;
- confirm comparison reports label strict-signature diagnostics separately from taxonomy-patch headline metrics.
