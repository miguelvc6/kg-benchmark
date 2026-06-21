# Few-Shot Static Support Selection

Created: `2026-06-18T12:45:07Z`
Source manifest: `reports/benchmark_selection/dev_prompt_v1_seed_13.json`
Blocked manifest: `reports/benchmark_selection/core_v1_seed_13.json`
Selection policy: `static_diverse`

## Support Sets

| Task | Visible example | Raw case id | Role |
| --- | --- | --- | --- |
| `a_box_repair` | `example_a_000001` | `repair_Q95552174_2447348752` | `a_box_clean_rule` |
| `a_box_repair` | `example_a_000002` | `repair_Q136738478_2443608153` | `a_box_format_or_literal_normalization` |
| `a_box_repair` | `example_a_000003` | `repair_Q100548289_1295700461` | `a_box_local_evidence` |
| `t_box_repair` | `example_t_000001` | `reform_Q17555344_P613_2439848360` | `tbox_taxonomy_cq_plus` |
| `t_box_repair` | `example_t_000002` | `reform_Q29041706_P356_2447249541` | `tbox_taxonomy_cq_minus_or_replace` |
| `t_box_repair` | `example_t_000003` | `reform_Q7199901_P3137_2428712581` | `tbox_taxonomy_no_causal_empty` |
| `t_box_repair` | `example_t_000004` | `reform_Q85808015_P1552_2405574367` | `tbox_taxonomy_other_or_family_only` |
| `track_diagnosis` | `example_d_000001` | `repair_Q15484095_2443725779` | `diagnosis_a_box` |
| `track_diagnosis` | `example_d_000002` | `reform_Q364189_P1368_2329221897` | `diagnosis_t_box` |

## Counts

- `dev_records`: 600
- `eligible_records`: 600
- `eligible_a_box_records`: 360
- `eligible_t_box_records`: 240
- `selected_examples`: 9
### by_track

- `A_BOX`: 4
- `T_BOX`: 5

### by_class_subtype

- `T_BOX:COINCIDENTAL_SCHEMA_CHANGE`: 1
- `T_BOX:RELAXATION_SET_EXPANSION`: 1
- `T_BOX:SCHEMA_UPDATE`: 3
- `TypeA:MULTIPLICITY_NORMALIZATION`: 1
- `TypeA:SELF_LINK_REJECTION`: 1
- `TypeB:LOCAL_SELECTION_CONFIRMED`: 1
- `TypeB:LOCAL_TEXT_DERIVED`: 1

### by_tbox_taxonomy_code

- `CQ_MINUS`: 1
- `CQ_PLUS`: 2
- `OTHER`: 1
- `__EMPTY_REPAIRS__`: 1

### by_property

- `P1368`: 1
- `P1552`: 1
- `P3137`: 1
- `P356`: 1
- `P5749`: 1
- `P613`: 1
- `P8726`: 1
- `P921`: 1
- `P961`: 1

### by_qid

- `Q100548289`: 1
- `Q136738478`: 1
- `Q15484095`: 1
- `Q17555344`: 1
- `Q29041706`: 1
- `Q364189`: 1
- `Q7199901`: 1
- `Q85808015`: 1
- `Q95552174`: 1

### by_tbox_revision_key

- `TBOX::P1368::2329221897`: 1
- `TBOX::P1552::2405574367`: 1
- `TBOX::P3137::2428712581`: 1
- `TBOX::P356::2447249541`: 1
- `TBOX::P613::2439848360`: 1
- `none`: 4


## Overlap Audit

- Core case overlap: `0`
- Core QID overlap: `0`
- Core T-box revision overlap: `0`
- Core property overlap: `8`
- Static support examples share properties with core cases: `True`
- Shared core properties: `P1368`, `P1552`, `P3137`, `P356`, `P5749`, `P8726`, `P921`, `P961`
