# prompt_dev_v3_abstain Branch Design

This is a separate abstention branch. It does not replace the no-abstain `prompt_dev_v3` baseline.

Recommended configuration:

```json
{
  "context_bundles": [
    "logic_only",
    "local_graph"
  ],
  "example_policies": [
    "zero_shot"
  ],
  "include_abstention": true,
  "repair_track_modes": [
    "oracle"
  ],
  "representations": [
    "hybrid_json_nl"
  ],
  "sample_strategy": "diverse_stratified",
  "tasks": [
    "track_diagnosis",
    "repair_proposal"
  ]
}
```

Metrics to compute when run:

- TypeC justified abstention rate
- TypeA false abstention rate
- TypeB false abstention rate
- repair success conditional on not abstaining
- hallucinated TypeC repair rate
- diagnostic/unknown abstention rate

Run only after the answerability audit confirms abstention is the right isolation step.
