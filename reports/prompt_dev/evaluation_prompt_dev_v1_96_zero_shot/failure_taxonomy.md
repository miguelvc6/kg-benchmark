# Phase F 96-Case Zero-Shot Failure Taxonomy

Run directory: `reports/prompt_dev/evaluation_prompt_dev_v1_96_zero_shot`

## Executive Summary

The 96-case run is operationally stable but shows systematic prompt-level failures in A-box value sourcing, A-box operation choice, and T-box signature/action discipline. T-box exact-signature failures are also partly an expected limitation of the compact temporal policy because the prompt does not expose pre/post reform signatures.

## Task 2 Answers

- **Are TypeA clean/rule cases failing?** Yes. TypeA/clean-rule A-box cases still show wrong-value and operation errors, especially constraint/type QIDs being used as replacement values. This is a prompt bug.
- **Are TypeB local_graph cases failing to use visible local evidence?** Yes. Local-graph TypeB failures remain and are counted under missing local evidence use. This is a prompt bug.
- **Are TypeC cases hallucinating repairs?** Yes. TypeC A-box cases produce concrete repairs despite insufficient evidence. This supports testing abstention after the zero-shot repair prompt is stable.
- **Are T-box failures mostly target-family correct but action/signature wrong?** No for exact target-family because most compact cases have no gold target family. Failures are dominated by invented signature_after and directional action choices without visible pre-change evidence.
- **Are T-box compact-policy cases impossible to match exactly from visible evidence?** Yes. The sampled T-box prompts use compact_inventory_no_pre_change_signature, so exact signature matching is not visible-evidence feasible and should be separated in reporting.

## Failure Mode Counts

### A-box

- `wrong value`: 58
- `over-delete`: 39
- `under-repair`: 35
- `correct deterministic normalization`: 32
- `constraint/type QID used as repaired entity value`: 28
- `wrong operation`: 23
- `expected TypeC insufficient-evidence failure`: 22
- `missing local evidence use`: 11
- `hallucinated value`: 9

### T-box

- `compact temporal policy makes exact signature impossible`: 96
- `invented signature_after`: 61
- `exact mismatch but plausible schema update`: 53
- `directional action chosen without visible evidence`: 43
- `wrong constraint family`: 14
- `report_violation_type_qids copied into signature_after`: 6

### Track Diagnosis

- `correct_track`: 127
- `A-box predicted as T-box due to constraint-report framing`: 42
- `ambiguous overused`: 12
- `T-box predicted as A-box due to concrete violating value`: 8
- `ambiguous used appropriately`: 3

## Key Slice Observations

### context_bundle

- `local_graph`: a_box::constraint/type QID used as repaired entity value=12, a_box::wrong value=29, n=192, t_box::invented signature_after=34, track::A-box predicted as T-box due to constraint-report framing=21, track::T-box predicted as A-box due to concrete violating value=5, track::ambiguous overused=6, track::ambiguous used appropriately=1, track::correct_track=63
- `logic_only`: a_box::constraint/type QID used as repaired entity value=16, a_box::wrong value=29, n=192, t_box::invented signature_after=27, track::A-box predicted as T-box due to constraint-report framing=21, track::T-box predicted as A-box due to concrete violating value=3, track::ambiguous overused=6, track::ambiguous used appropriately=2, track::correct_track=64

### task

- `repair_proposal`: a_box::constraint/type QID used as repaired entity value=28, a_box::wrong value=58, n=192, t_box::invented signature_after=61
- `track_diagnosis`: n=192, track::A-box predicted as T-box due to constraint-report framing=42, track::T-box predicted as A-box due to concrete violating value=8, track::ambiguous overused=12, track::ambiguous used appropriately=3, track::correct_track=127

### track

- `A_BOX`: a_box::constraint/type QID used as repaired entity value=28, a_box::wrong value=58, n=192, track::A-box predicted as T-box due to constraint-report framing=42, track::ambiguous overused=8, track::ambiguous used appropriately=2, track::correct_track=44
- `T_BOX`: n=192, t_box::invented signature_after=61, track::T-box predicted as A-box due to concrete violating value=8, track::ambiguous overused=4, track::ambiguous used appropriately=1, track::correct_track=83

### class

- `T_BOX`: n=192, t_box::invented signature_after=61, track::T-box predicted as A-box due to concrete violating value=8, track::ambiguous overused=4, track::ambiguous used appropriately=1, track::correct_track=83
- `TypeA`: a_box::constraint/type QID used as repaired entity value=15, a_box::wrong value=16, n=96, track::A-box predicted as T-box due to constraint-report framing=21, track::ambiguous overused=3, track::ambiguous used appropriately=1, track::correct_track=23
- `TypeB`: a_box::constraint/type QID used as repaired entity value=6, a_box::wrong value=20, n=52, track::A-box predicted as T-box due to constraint-report framing=9, track::ambiguous overused=5, track::correct_track=12
- `TypeC`: a_box::constraint/type QID used as repaired entity value=7, a_box::wrong value=22, n=44, track::A-box predicted as T-box due to constraint-report framing=12, track::ambiguous used appropriately=1, track::correct_track=9

### selection_stratum

- `DEV_TBOX_COINCIDENTAL_SCHEMA_CHANGE`: n=48, t_box::invented signature_after=15, track::T-box predicted as A-box due to concrete violating value=2, track::ambiguous used appropriately=1, track::correct_track=21
- `DEV_TBOX_RELAXATION_SET_EXPANSION`: n=48, t_box::invented signature_after=12, track::T-box predicted as A-box due to concrete violating value=3, track::ambiguous overused=1, track::correct_track=20
- `DEV_TBOX_RESTRICTION_SET_CONTRACTION`: n=48, t_box::invented signature_after=15, track::T-box predicted as A-box due to concrete violating value=3, track::ambiguous overused=2, track::correct_track=19
- `DEV_TBOX_SCHEMA_UPDATE`: n=48, t_box::invented signature_after=19, track::ambiguous overused=1, track::correct_track=23
- `DEV_TYPEA_AMBIGUOUS_DELETE`: a_box::constraint/type QID used as repaired entity value=4, a_box::wrong value=2, n=12, track::A-box predicted as T-box due to constraint-report framing=3, track::ambiguous used appropriately=1, track::correct_track=2
- `DEV_TYPEA_CLEAN_RULE_REJECTION`: a_box::constraint/type QID used as repaired entity value=11, a_box::wrong value=14, n=84, track::A-box predicted as T-box due to constraint-report framing=18, track::ambiguous overused=3, track::correct_track=21
- `DEV_TYPEB_LOCAL`: a_box::constraint/type QID used as repaired entity value=6, a_box::wrong value=20, n=52, track::A-box predicted as T-box due to constraint-report framing=9, track::ambiguous overused=5, track::correct_track=12
- `DEV_TYPEC_EXTERNAL_OR_UNKNOWN`: a_box::constraint/type QID used as repaired entity value=7, a_box::wrong value=22, n=44, track::A-box predicted as T-box due to constraint-report framing=12, track::ambiguous used appropriately=1, track::correct_track=9

### score_slice

- `diagnostic_only`: a_box::constraint/type QID used as repaired entity value=7, a_box::wrong value=18, n=92, t_box::invented signature_after=15, track::A-box predicted as T-box due to constraint-report framing=12, track::T-box predicted as A-box due to concrete violating value=2, track::ambiguous used appropriately=3, track::correct_track=29
- `main_score`: a_box::constraint/type QID used as repaired entity value=21, a_box::wrong value=40, n=292, t_box::invented signature_after=46, track::A-box predicted as T-box due to constraint-report framing=30, track::T-box predicted as A-box due to concrete violating value=6, track::ambiguous overused=12, track::correct_track=98

### tbox_temporal_policy

- `compact_inventory_no_pre_change_signature`: n=96, t_box::invented signature_after=61

## Representative Examples

### a_box

- `constraint/type QID used as repaired entity value`
  - `repair_Q2131024_2445785456`: {"case_id": "repair_Q2131024_2445785456", "class": "TypeA", "context": "logic_only", "expected": [], "ops": ["SET"], "score_slice": "diagnostic_only", "subtype": "DELETE_AMBIGUOUS", "values": ["Q29934200"]}
  - `repair_Q63898844_2333536236`: {"case_id": "repair_Q63898844_2333536236", "class": "TypeA", "context": "logic_only", "expected": ["Q99868032"], "ops": ["REMOVE"], "score_slice": "main_score", "subtype": "SET_MEMBERSHIP_REJECTION", "values": ["Q59496158"]}
  - `repair_Q979055_2441520750`: {"case_id": "repair_Q979055_2441520750", "class": "TypeB", "context": "logic_only", "expected": ["Q35896"], "ops": ["REMOVE", "ADD"], "score_slice": "main_score", "subtype": "LOCAL_FOCUS_NON_TARGET_PROPERTY", "values": ["Q125097315", "Q29934200"]}
- `under-repair`
  - `repair_Q2131024_2445785456`: {"case_id": "repair_Q2131024_2445785456", "class": "TypeA", "context": "logic_only", "expected": [], "ops": ["SET"], "score_slice": "diagnostic_only", "subtype": "DELETE_AMBIGUOUS", "values": ["Q29934200"]}
  - `repair_Q117185537_2442854831`: {"case_id": "repair_Q117185537_2442854831", "class": "TypeB", "context": "logic_only", "expected": ["Q2359069", "Q259167", "Q5229639"], "ops": ["SET"], "score_slice": "main_score", "subtype": "LOCAL_MIXED", "values": ["Q54828449"]}
  - `repair_Q100536719_1295663835`: {"case_id": "repair_Q100536719_1295663835", "class": "TypeB", "context": "logic_only", "expected": ["2013/si/333/made"], "ops": ["SET"], "score_slice": "main_score", "subtype": "LOCAL_TEXT_DERIVED", "values": ["S.I. No. 333/2013"]}
- `wrong operation`
  - `repair_Q2131024_2445785456`: {"case_id": "repair_Q2131024_2445785456", "class": "TypeA", "context": "logic_only", "expected": [], "ops": ["SET"], "score_slice": "diagnostic_only", "subtype": "DELETE_AMBIGUOUS", "values": ["Q29934200"]}
  - `repair_Q97573807_2403110875`: {"case_id": "repair_Q97573807_2403110875", "class": "TypeA", "context": "logic_only", "expected": ["244833559"], "ops": ["REMOVE"], "score_slice": "main_score", "subtype": "MULTIPLICITY_NORMALIZATION", "values": ["244833559"]}
  - `repair_Q117185537_2442854831`: {"case_id": "repair_Q117185537_2442854831", "class": "TypeB", "context": "logic_only", "expected": ["Q2359069", "Q259167", "Q5229639"], "ops": ["SET"], "score_slice": "main_score", "subtype": "LOCAL_MIXED", "values": ["Q54828449"]}
- `wrong value`
  - `repair_Q2131024_2445785456`: {"case_id": "repair_Q2131024_2445785456", "class": "TypeA", "context": "logic_only", "expected": [], "ops": ["SET"], "score_slice": "diagnostic_only", "subtype": "DELETE_AMBIGUOUS", "values": ["Q29934200"]}
  - `repair_Q97573807_2403110875`: {"case_id": "repair_Q97573807_2403110875", "class": "TypeA", "context": "logic_only", "expected": ["244833559"], "ops": ["REMOVE"], "score_slice": "main_score", "subtype": "MULTIPLICITY_NORMALIZATION", "values": ["244833559"]}
  - `repair_Q979055_2441520750`: {"case_id": "repair_Q979055_2441520750", "class": "TypeB", "context": "logic_only", "expected": ["Q35896"], "ops": ["REMOVE", "ADD"], "score_slice": "main_score", "subtype": "LOCAL_FOCUS_NON_TARGET_PROPERTY", "values": ["Q125097315", "Q29934200"]}
- `over-delete`
  - `repair_Q55682174_2297178353`: {"case_id": "repair_Q55682174_2297178353", "class": "TypeA", "context": "logic_only", "expected": ["1051188857"], "ops": ["REMOVE"], "score_slice": "main_score", "subtype": "FORMAT_VALUE_PRUNING", "values": ["pkd0121"]}
  - `repair_Q97573807_2403110875`: {"case_id": "repair_Q97573807_2403110875", "class": "TypeA", "context": "logic_only", "expected": ["244833559"], "ops": ["REMOVE"], "score_slice": "main_score", "subtype": "MULTIPLICITY_NORMALIZATION", "values": ["244833559"]}
  - `repair_Q131148728_2447350025`: {"case_id": "repair_Q131148728_2447350025", "class": "TypeA", "context": "logic_only", "expected": ["Q131148694", "Q131148704"], "ops": ["REMOVE"], "score_slice": "main_score", "subtype": "SELF_LINK_REJECTION", "values": ["Q131148728"]}
- `expected TypeC insufficient-evidence failure`
  - `repair_Q135504042_2388911715`: {"case_id": "repair_Q135504042_2388911715", "class": "TypeC", "context": "logic_only", "expected": ["Q200", "Q203", "Q37136"], "ops": ["REMOVE", "REMOVE", "ADD", "ADD", "ADD"], "score_slice": "main_score", "subtype": "EXTERNAL_BY_ELIMINATION", "values": ["Q200", "Q203", "Q2", "Q5", "Q11"]}
  - `repair_Q27126389_2425267918`: {"case_id": "repair_Q27126389_2425267918", "class": "TypeC", "context": "logic_only", "expected": ["30414747", "3836830"], "ops": ["SET"], "score_slice": "diagnostic_only", "subtype": "UNKNOWN_BAD_TARGET_OR_CONTEXT", "values": ["Q27126389"]}
  - `repair_Q137441618_2444110605`: {"case_id": "repair_Q137441618_2444110605", "class": "TypeC", "context": "logic_only", "expected": ["Heftye family"], "ops": ["SET"], "score_slice": "diagnostic_only", "subtype": "UNKNOWN_INCOMPLETE_LOCAL_CONTEXT", "values": ["Heftye"]}
- `hallucinated value`
  - `repair_Q135504042_2388911715`: {"case_id": "repair_Q135504042_2388911715", "class": "TypeC", "context": "logic_only", "expected": ["Q200", "Q203", "Q37136"], "ops": ["REMOVE", "REMOVE", "ADD", "ADD", "ADD"], "score_slice": "main_score", "subtype": "EXTERNAL_BY_ELIMINATION", "values": ["Q200", "Q203", "Q2", "Q5", "Q11"]}
  - `repair_Q100536073_1295665587`: {"case_id": "repair_Q100536073_1295665587", "class": "TypeB", "context": "logic_only", "expected": ["2013/si/157/made"], "ops": ["SET"], "score_slice": "main_score", "subtype": "LOCAL_TEXT_DERIVED", "values": ["si/157/2013"]}
  - `repair_Q1546466_2297182718`: {"case_id": "repair_Q1546466_2297182718", "class": "TypeA", "context": "logic_only", "expected": [], "ops": ["REMOVE", "ADD"], "score_slice": "main_score", "subtype": "REJECTION_FORMAT_INVALID", "values": ["pka0194", "10000000"]}
- `missing local evidence use`
  - `repair_Q979055_2441520750`: {"case_id": "repair_Q979055_2441520750", "class": "TypeB", "context": "local_graph", "expected": ["Q35896"], "ops": ["SET"], "score_slice": "main_score", "subtype": "LOCAL_FOCUS_NON_TARGET_PROPERTY", "values": ["Q125097315"]}
  - `repair_Q117185537_2442854831`: {"case_id": "repair_Q117185537_2442854831", "class": "TypeB", "context": "local_graph", "expected": ["Q2359069", "Q259167", "Q5229639"], "ops": ["DELETE_ALL", "DELETE_ALL", "DELETE_ALL"], "score_slice": "main_score", "subtype": "LOCAL_MIXED", "values": []}
  - `repair_Q94762277_2446321758`: {"case_id": "repair_Q94762277_2446321758", "class": "TypeB", "context": "local_graph", "expected": ["1012560481"], "ops": ["DELETE_ALL", "DELETE_ALL"], "score_slice": "main_score", "subtype": "LOCAL_SELECTION_CONFIRMED", "values": []}

### t_box

- `compact temporal policy makes exact signature impossible`
  - `reform_Q254745_P1346_1580835843`: {"action": "RESTRICTION_RANGE_NARROWED", "case_id": "reform_Q254745_P1346_1580835843", "changed_constraint_types": [], "context": "logic_only", "score_slice": "diagnostic_only", "signature_after_len": 1, "subtype": "COINCIDENTAL_SCHEMA_CHANGE", "target": "Q21510855", "temporal_policy": "compact_inventory_no_pre_change_signature"}
  - `reform_Q96483507_P2529_2442137584`: {"action": "SCHEMA_UPDATE", "case_id": "reform_Q96483507_P2529_2442137584", "changed_constraint_types": ["Q21503247"], "context": "logic_only", "score_slice": "main_score", "signature_after_len": 0, "subtype": "RELAXATION_SET_EXPANSION", "target": "Q19474404", "temporal_policy": "compact_inventory_no_pre_change_signature"}
  - `reform_Q20888952_P4216_2361580316`: {"action": "RELAXATION_RANGE_WIDENED", "case_id": "reform_Q20888952_P4216_2361580316", "changed_constraint_types": [], "context": "logic_only", "score_slice": "main_score", "signature_after_len": 1, "subtype": "RESTRICTION_SET_CONTRACTION", "target": "Q53869507", "temporal_policy": "compact_inventory_no_pre_change_signature"}
- `directional action chosen without visible evidence`
  - `reform_Q254745_P1346_1580835843`: {"action": "RESTRICTION_RANGE_NARROWED", "case_id": "reform_Q254745_P1346_1580835843", "changed_constraint_types": [], "context": "logic_only", "score_slice": "diagnostic_only", "signature_after_len": 1, "subtype": "COINCIDENTAL_SCHEMA_CHANGE", "target": "Q21510855", "temporal_policy": "compact_inventory_no_pre_change_signature"}
  - `reform_Q20888952_P4216_2361580316`: {"action": "RELAXATION_RANGE_WIDENED", "case_id": "reform_Q20888952_P4216_2361580316", "changed_constraint_types": [], "context": "logic_only", "score_slice": "main_score", "signature_after_len": 1, "subtype": "RESTRICTION_SET_CONTRACTION", "target": "Q53869507", "temporal_policy": "compact_inventory_no_pre_change_signature"}
  - `reform_Q95217667_P950_2330325818`: {"action": "RESTRICTION_RANGE_NARROWED", "case_id": "reform_Q95217667_P950_2330325818", "changed_constraint_types": [], "context": "logic_only", "score_slice": "main_score", "signature_after_len": 1, "subtype": "SCHEMA_UPDATE", "target": "Q108139345", "temporal_policy": "compact_inventory_no_pre_change_signature"}
- `invented signature_after`
  - `reform_Q254745_P1346_1580835843`: {"action": "RESTRICTION_RANGE_NARROWED", "case_id": "reform_Q254745_P1346_1580835843", "changed_constraint_types": [], "context": "logic_only", "score_slice": "diagnostic_only", "signature_after_len": 1, "subtype": "COINCIDENTAL_SCHEMA_CHANGE", "target": "Q21510855", "temporal_policy": "compact_inventory_no_pre_change_signature"}
  - `reform_Q20888952_P4216_2361580316`: {"action": "RELAXATION_RANGE_WIDENED", "case_id": "reform_Q20888952_P4216_2361580316", "changed_constraint_types": [], "context": "logic_only", "score_slice": "main_score", "signature_after_len": 1, "subtype": "RESTRICTION_SET_CONTRACTION", "target": "Q53869507", "temporal_policy": "compact_inventory_no_pre_change_signature"}
  - `reform_Q95217667_P950_2330325818`: {"action": "RESTRICTION_RANGE_NARROWED", "case_id": "reform_Q95217667_P950_2330325818", "changed_constraint_types": [], "context": "logic_only", "score_slice": "main_score", "signature_after_len": 1, "subtype": "SCHEMA_UPDATE", "target": "Q108139345", "temporal_policy": "compact_inventory_no_pre_change_signature"}
- `exact mismatch but plausible schema update`
  - `reform_Q96483507_P2529_2442137584`: {"action": "SCHEMA_UPDATE", "case_id": "reform_Q96483507_P2529_2442137584", "changed_constraint_types": ["Q21503247"], "context": "logic_only", "score_slice": "main_score", "signature_after_len": 0, "subtype": "RELAXATION_SET_EXPANSION", "target": "Q19474404", "temporal_policy": "compact_inventory_no_pre_change_signature"}
  - `reform_Q33203_P2347_2236583436`: {"action": "SCHEMA_UPDATE", "case_id": "reform_Q33203_P2347_2236583436", "changed_constraint_types": [], "context": "logic_only", "score_slice": "diagnostic_only", "signature_after_len": 0, "subtype": "COINCIDENTAL_SCHEMA_CHANGE", "target": "Q19474404", "temporal_policy": "compact_inventory_no_pre_change_signature"}
  - `reform_Q115933554_P31_2442704771`: {"action": "SCHEMA_UPDATE", "case_id": "reform_Q115933554_P31_2442704771", "changed_constraint_types": [], "context": "logic_only", "score_slice": "main_score", "signature_after_len": 0, "subtype": "RELAXATION_SET_EXPANSION", "target": "Q52558054", "temporal_policy": "compact_inventory_no_pre_change_signature"}
- `wrong constraint family`
  - `reform_Q96483507_P2529_2442137584`: {"action": "SCHEMA_UPDATE", "case_id": "reform_Q96483507_P2529_2442137584", "changed_constraint_types": ["Q21503247"], "context": "logic_only", "score_slice": "main_score", "signature_after_len": 0, "subtype": "RELAXATION_SET_EXPANSION", "target": "Q19474404", "temporal_policy": "compact_inventory_no_pre_change_signature"}
  - `reform_Q1016312_P18_2402814023`: {"action": "SCHEMA_UPDATE", "case_id": "reform_Q1016312_P18_2402814023", "changed_constraint_types": ["Q52558054"], "context": "logic_only", "score_slice": "diagnostic_only", "signature_after_len": 1, "subtype": "COINCIDENTAL_SCHEMA_CHANGE", "target": "Q21502404", "temporal_policy": "compact_inventory_no_pre_change_signature"}
  - `reform_Q23231924_P645_722137891`: {"action": "RELAXATION_RANGE_WIDENED", "case_id": "reform_Q23231924_P645_722137891", "changed_constraint_types": ["Q21502410"], "context": "logic_only", "score_slice": "main_score", "signature_after_len": 1, "subtype": "SCHEMA_UPDATE", "target": "Q53869507", "temporal_policy": "compact_inventory_no_pre_change_signature"}
- `report_violation_type_qids copied into signature_after`
  - `reform_Q1510431_P10225_2403749973`: {"action": "RELAXATION_RANGE_WIDENED", "case_id": "reform_Q1510431_P10225_2403749973", "changed_constraint_types": [], "context": "local_graph", "score_slice": "diagnostic_only", "signature_after_len": 1, "subtype": "COINCIDENTAL_SCHEMA_CHANGE", "target": "Q21503250", "temporal_policy": "compact_inventory_no_pre_change_signature"}
  - `reform_Q109769851_P11318_2217144931`: {"action": "RESTRICTION_RANGE_NARROWED", "case_id": "reform_Q109769851_P11318_2217144931", "changed_constraint_types": [], "context": "local_graph", "score_slice": "diagnostic_only", "signature_after_len": 1, "subtype": "COINCIDENTAL_SCHEMA_CHANGE", "target": "Q21503250", "temporal_policy": "compact_inventory_no_pre_change_signature"}
  - `reform_Q71167967_P410_2444658197`: {"action": "RESTRICTION_RANGE_NARROWED", "case_id": "reform_Q71167967_P410_2444658197", "changed_constraint_types": [], "context": "local_graph", "score_slice": "main_score", "signature_after_len": 1, "subtype": "RELAXATION_SET_EXPANSION", "target": "Q21510865", "temporal_policy": "compact_inventory_no_pre_change_signature"}

### track_diagnosis

- `A-box predicted as T-box due to constraint-report framing`
  - `repair_Q423757_2425335643`: {"case_id": "repair_Q423757_2425335643", "class": "TypeA", "context": "logic_only", "predicted": "T_BOX", "score_slice": "main_score", "subtype": "FORMAT_NORMALIZATION", "track": "A_BOX"}
  - `repair_Q97573807_2403110875`: {"case_id": "repair_Q97573807_2403110875", "class": "TypeA", "context": "logic_only", "predicted": "T_BOX", "score_slice": "main_score", "subtype": "MULTIPLICITY_NORMALIZATION", "track": "A_BOX"}
  - `repair_Q979055_2441520750`: {"case_id": "repair_Q979055_2441520750", "class": "TypeB", "context": "logic_only", "predicted": "T_BOX", "score_slice": "main_score", "subtype": "LOCAL_FOCUS_NON_TARGET_PROPERTY", "track": "A_BOX"}
- `ambiguous overused`
  - `reform_Q20888952_P4216_2361580316`: {"case_id": "reform_Q20888952_P4216_2361580316", "class": "T_BOX", "context": "logic_only", "predicted": "AMBIGUOUS", "score_slice": "main_score", "subtype": "RESTRICTION_SET_CONTRACTION", "track": "T_BOX"}
  - `repair_Q131148728_2447350025`: {"case_id": "repair_Q131148728_2447350025", "class": "TypeA", "context": "logic_only", "predicted": "AMBIGUOUS", "score_slice": "main_score", "subtype": "SELF_LINK_REJECTION", "track": "A_BOX"}
  - `reform_Q439712_P1770_2433393062`: {"case_id": "reform_Q439712_P1770_2433393062", "class": "T_BOX", "context": "logic_only", "predicted": "AMBIGUOUS", "score_slice": "main_score", "subtype": "RESTRICTION_SET_CONTRACTION", "track": "T_BOX"}
- `T-box predicted as A-box due to concrete violating value`
  - `reform_Q1510431_P10225_2403749973`: {"case_id": "reform_Q1510431_P10225_2403749973", "class": "T_BOX", "context": "logic_only", "predicted": "A_BOX", "score_slice": "diagnostic_only", "subtype": "COINCIDENTAL_SCHEMA_CHANGE", "track": "T_BOX"}
  - `reform_Q58311923_P6004_1800952303`: {"case_id": "reform_Q58311923_P6004_1800952303", "class": "T_BOX", "context": "logic_only", "predicted": "A_BOX", "score_slice": "main_score", "subtype": "RESTRICTION_SET_CONTRACTION", "track": "T_BOX"}
  - `reform_Q1056905_P166_2254793027`: {"case_id": "reform_Q1056905_P166_2254793027", "class": "T_BOX", "context": "logic_only", "predicted": "A_BOX", "score_slice": "diagnostic_only", "subtype": "COINCIDENTAL_SCHEMA_CHANGE", "track": "T_BOX"}
- `ambiguous used appropriately`
  - `reform_Q107438498_P2092_2333296438`: {"case_id": "reform_Q107438498_P2092_2333296438", "class": "T_BOX", "context": "logic_only", "predicted": "AMBIGUOUS", "score_slice": "diagnostic_only", "subtype": "COINCIDENTAL_SCHEMA_CHANGE", "track": "T_BOX"}
  - `repair_Q278283_2086421334`: {"case_id": "repair_Q278283_2086421334", "class": "TypeC", "context": "logic_only", "predicted": "AMBIGUOUS", "score_slice": "diagnostic_only", "subtype": "UNKNOWN_BAD_TARGET_OR_CONTEXT", "track": "A_BOX"}
  - `repair_Q670165_2166553005`: {"case_id": "repair_Q670165_2166553005", "class": "TypeA", "context": "local_graph", "predicted": "AMBIGUOUS", "score_slice": "diagnostic_only", "subtype": "DELETE_AMBIGUOUS", "track": "A_BOX"}

## Prompt Iteration Decision

The taxonomy supports a focused `prompt_dev_v2` iteration. Required changes are: stronger A-box value-source rules, an explicit A-box operation rubric, a T-box action decision tree, stricter T-box signature discipline, and a track-diagnosis warning that constraint-report vocabulary alone does not imply T-box.
