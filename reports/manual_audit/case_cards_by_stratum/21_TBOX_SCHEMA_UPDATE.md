# TBOX_SCHEMA_UPDATE

Cases: 26

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `reform_Q106814418_P1346_1577277439`

| Field | Value |
|---|---|
| qid | Q106814418 |
| property | P1346 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21510865 value-type constraint |
| group_key | TBOX::P1346::1577277439 |
| tbox_revision_key | TBOX::P1346::1577277439 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "value-type constraint",
  "decision_constraint_type_qid": "Q21510865"
}
```

#### Repair Target

```json
{
  "author": "Lucio Luiz",
  "kind": "T_BOX",
  "property_revision_id": 1577277439,
  "property_revision_prev": 1575232987
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-02-15T10:36:41",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1346",
  "report_revision_new": 1577696536,
  "report_revision_old": 1577200212,
  "report_violation_type": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "social entity established to meet needs or pursue goals",
    "community of people who share a common language, culture, ethnicity, descent, or history",
    "large landmass identified by convention",
    "domesticated four-footed mammal from the equine family",
    "publication type, serial publication that appears in a new edition on a regular schedule",
    "computer designed for playing chess",
    "place of any size, in which people permanently live",
    "territorial entity for administration purposes, with or without its own local government",
    "fictional human or non-human character in a narrative work of art",
    "medium for recording information (words or images) typically on bound pages or more abstractly in electronic or audio form",
    "creative work in which images and text convey information such as narratives",
    "intellectual or artistic creation"
  ],
  "report_violation_type_labels_en": [
    "human",
    "organization",
    "nation",
    "continent",
    "horse",
    "periodical",
    "chess computer",
    "human settlement",
    "administrative territorial entity",
    "character",
    "book",
    "comics",
    "work"
  ],
  "report_violation_type_normalized": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
  "report_violation_type_qids": [
    "Q5",
    "Q43229",
    "Q6266",
    "Q5107",
    "Q726",
    "Q1002697",
    "Q1364192",
    "Q486972",
    "Q56061",
    "Q95074",
    "Q571",
    "Q1004",
    "Q386724"
  ],
  "report_violation_type_raw": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
  "value": null,
  "value_current_2026": [
    "Q288333"
  ],
  "value_current_2026_descriptions_en": [
    "Italian diver"
  ],
  "value_current_2026_labels_en": [
    "Tania Cagnotto"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "winner of a competition or similar event, not to be used from the awardees record (instead use \"award received\" (P166), possibly qualified with \"for work\" (P1686)) nor for wars or battles",
    "label": "winner"
  },
  "qid": {
    "description": null,
    "label": "diving at the 2008 European Aquatics Championships – women's 10 metre platform"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "inverse constraint",
    "qid": "Q21510855"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21510865",
      "mapped_violation_family": "value_type",
      "violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724"
    }
  ],
  "candidate_violation_names": [
    "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "value-type constraint",
  "mapped_violation_constraint_qid": "Q21510865",
  "mapped_violation_family": "value_type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q1656682"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "value-type constraint",
  "target_constraint_qid": "Q21510865",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q1656682"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "changed_qualifier_properties": [
    "P2308"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
  "semantic_added_values": [
    "Q1656682"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "value-type constraint",
  "target_constraint_qid": "Q21510865",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": true
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Lucio Luiz",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "543e6e09dcb8cc7824fdf0db30d1c54aa8810398",
  "hash_before": "33ebba8f595b7d9221d8dad0f7114ae058a6f3d5",
  "property_revision_id": 1577277439,
  "property_revision_prev": 1575232987,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "mapped_violation_constraint_qid": "Q21510865",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724"
  },
  {
    "result": "Q21510865",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q1656682"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21510865",
        "mapped_violation_family": "value_type",
        "violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724"
      }
    ],
    "candidate_violation_names": [
      "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21510865"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510865"
    ],
    "changed_qualifier_properties": [
      "P2308"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "value-type constraint",
    "mapped_report_constraint_qid": "Q21510865",
    "mapped_report_family": "value_type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "value-type constraint",
    "mapped_violation_constraint_qid": "Q21510865",
    "mapped_violation_family": "value_type",
    "mapped_violation_reason": "value_type_prefix_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q1656682"
    ],
    "semantic_changed_qualifier_properties": [
      "P2308"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "value-type constraint",
    "target_constraint_qid": "Q21510865",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "value_type",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724"
  }
]
```

---

## 002. `reform_Q106825114_P1346_1577277439`

| Field | Value |
|---|---|
| qid | Q106825114 |
| property | P1346 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21510865 value-type constraint |
| group_key | TBOX::P1346::1577277439 |
| tbox_revision_key | TBOX::P1346::1577277439 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "value-type constraint",
  "decision_constraint_type_qid": "Q21510865"
}
```

#### Repair Target

```json
{
  "author": "Lucio Luiz",
  "kind": "T_BOX",
  "property_revision_id": 1577277439,
  "property_revision_prev": 1575232987
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-02-15T10:36:41",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1346",
  "report_revision_new": 1577696536,
  "report_revision_old": 1577200212,
  "report_violation_type": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "social entity established to meet needs or pursue goals",
    "community of people who share a common language, culture, ethnicity, descent, or history",
    "large landmass identified by convention",
    "domesticated four-footed mammal from the equine family",
    "publication type, serial publication that appears in a new edition on a regular schedule",
    "computer designed for playing chess",
    "place of any size, in which people permanently live",
    "territorial entity for administration purposes, with or without its own local government",
    "fictional human or non-human character in a narrative work of art",
    "medium for recording information (words or images) typically on bound pages or more abstractly in electronic or audio form",
    "creative work in which images and text convey information such as narratives",
    "intellectual or artistic creation"
  ],
  "report_violation_type_labels_en": [
    "human",
    "organization",
    "nation",
    "continent",
    "horse",
    "periodical",
    "chess computer",
    "human settlement",
    "administrative territorial entity",
    "character",
    "book",
    "comics",
    "work"
  ],
  "report_violation_type_normalized": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
  "report_violation_type_qids": [
    "Q5",
    "Q43229",
    "Q6266",
    "Q5107",
    "Q726",
    "Q1002697",
    "Q1364192",
    "Q486972",
    "Q56061",
    "Q95074",
    "Q571",
    "Q1004",
    "Q386724"
  ],
  "report_violation_type_raw": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
  "value": null,
  "value_current_2026": [
    "Q106825037"
  ],
  "value_current_2026_descriptions_en": [
    "Russian diver"
  ],
  "value_current_2026_labels_en": [
    "Anna Konanykhina"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "winner of a competition or similar event, not to be used from the awardees record (instead use \"award received\" (P166), possibly qualified with \"for work\" (P1686)) nor for wars or battles",
    "label": "winner"
  },
  "qid": {
    "description": "موسم رياضى فى المجر",
    "label": "diving at the 2020 European Aquatics Championships – women's 10 m platform"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "inverse constraint",
    "qid": "Q21510855"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21510865",
      "mapped_violation_family": "value_type",
      "violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724"
    }
  ],
  "candidate_violation_names": [
    "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "value-type constraint",
  "mapped_violation_constraint_qid": "Q21510865",
  "mapped_violation_family": "value_type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q1656682"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "value-type constraint",
  "target_constraint_qid": "Q21510865",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q1656682"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "changed_qualifier_properties": [
    "P2308"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
  "semantic_added_values": [
    "Q1656682"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "value-type constraint",
  "target_constraint_qid": "Q21510865",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": true
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Lucio Luiz",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "543e6e09dcb8cc7824fdf0db30d1c54aa8810398",
  "hash_before": "33ebba8f595b7d9221d8dad0f7114ae058a6f3d5",
  "property_revision_id": 1577277439,
  "property_revision_prev": 1575232987,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "mapped_violation_constraint_qid": "Q21510865",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724"
  },
  {
    "result": "Q21510865",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q1656682"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21510865",
        "mapped_violation_family": "value_type",
        "violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724"
      }
    ],
    "candidate_violation_names": [
      "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21510865"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510865"
    ],
    "changed_qualifier_properties": [
      "P2308"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "value-type constraint",
    "mapped_report_constraint_qid": "Q21510865",
    "mapped_report_family": "value_type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "value-type constraint",
    "mapped_violation_constraint_qid": "Q21510865",
    "mapped_violation_family": "value_type",
    "mapped_violation_reason": "value_type_prefix_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q1656682"
    ],
    "semantic_changed_qualifier_properties": [
      "P2308"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "value-type constraint",
    "target_constraint_qid": "Q21510865",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "value_type",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724"
  }
]
```

---

## 003. `reform_Q107006876_P190_2440746684`

| Field | Value |
|---|---|
| qid | Q107006876 |
| property | P190 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P190::2440746684 |
| tbox_revision_key | TBOX::P190::2440746684 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Jpnbot",
  "kind": "T_BOX",
  "property_revision_id": 2440746684,
  "property_revision_prev": 2440687529
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-11T13:55:51",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P190",
  "report_revision_new": 2440879198,
  "report_revision_old": 2440425151,
  "report_violation_type": "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
  "report_violation_type_descriptions_en": [
    "territorial entity for administration purposes, with or without its own local government",
    "large human settlement",
    "type of human settlement",
    "geographically localized community within a larger city, town or suburb",
    "place of any size, in which people permanently live",
    "railway facility where trains regularly stop to load or unload passengers and/or freight",
    "institution for the education of students by teachers",
    "imposing structure created to commemorate a person or event, or used for that purpose"
  ],
  "report_violation_type_labels_en": [
    "administrative territorial entity",
    "city",
    "city or town",
    "neighborhood",
    "human settlement",
    "railway station",
    "school",
    "monument"
  ],
  "report_violation_type_normalized": "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
  "report_violation_type_qids": [
    "Q56061",
    "Q515",
    "Q7930989",
    "Q123705",
    "Q486972",
    "Q55488",
    "Q3914",
    "Q4989906"
  ],
  "report_violation_type_raw": "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
  "report_violation_types": [
    "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
    "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
  ],
  "value": null,
  "value_current_2026": [
    "Q64921067"
  ],
  "value_current_2026_descriptions_en": [
    "fictional eldritch town, narrative location of the Welcome to Night Vale podcast"
  ],
  "value_current_2026_labels_en": [
    "Night Vale"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "twin towns, sister cities, twinned municipalities and other localities that have a partnership or cooperative agreement, either legally or informally acknowledged by their governments",
    "label": "twinned administrative body"
  },
  "qid": {
    "description": "fictional locale in Welcome to Night Vale",
    "label": "Desert Bluffs"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "symmetric constraint",
    "qid": "Q21510862"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
    },
    {
      "candidate_causality_match_level": "value_or_property_overlap_only",
      "candidate_score": 20,
      "mapped_violation_constraint_qid": "Q21510865",
      "mapped_violation_family": "value_type",
      "violation_name": "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
    }
  ],
  "candidate_violation_names": [
    "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
    "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
  ],
  "causality_match_level": "exact_constraint_and_value_match",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "type_compatible_report_argument_overlap",
  "compatible_overlap_used": true,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [
    "Q123705",
    "Q515",
    "Q7930989"
  ],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q3895768"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_value_count": 3,
  "semantic_removed_values": [
    "Q123705",
    "Q515",
    "Q7930989"
  ],
  "set_operation": "mixed",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "type constraint",
  "target_constraint_qid": "Q21503250",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [
    "Q123705",
    "Q515",
    "Q7930989"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q3895768"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": [
    "P2308"
  ],
  "compatible_overlap_reason": "type_compatible_report_argument_overlap",
  "compatible_overlap_used": true,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 3,
  "removed_values": [
    "Q123705",
    "Q515",
    "Q7930989"
  ],
  "selected_violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
  "semantic_added_values": [
    "Q3895768"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_values": [
    "Q123705",
    "Q515",
    "Q7930989"
  ],
  "set_operation": "mixed",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "type constraint",
  "target_constraint_qid": "Q21503250",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Jpnbot",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "e0b80a748b26f7c5d8e261f31f900416046d827e",
  "hash_before": "8a70cd3a253177cd6d177081d5c6dc307edcd89f",
  "property_revision_id": 2440746684,
  "property_revision_prev": 2440687529,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_and_value_match",
    "mapped_violation_constraint_qid": "Q21503250",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 3,
    "result": "SCHEMA_UPDATE",
    "set_operation": "mixed",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q3895768"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
      },
      {
        "candidate_causality_match_level": "value_or_property_overlap_only",
        "candidate_score": 20,
        "mapped_violation_constraint_qid": "Q21510865",
        "mapped_violation_family": "value_type",
        "violation_name": "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
      }
    ],
    "candidate_violation_names": [
      "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
      "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
    ],
    "causality_match_level": "exact_constraint_and_value_match",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21503250"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503250"
    ],
    "changed_qualifier_properties": [
      "P2308"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "type_compatible_report_argument_overlap",
    "compatible_overlap_used": true,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [
      "Q123705",
      "Q515",
      "Q7930989"
    ],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "type constraint",
    "mapped_report_constraint_qid": "Q21503250",
    "mapped_report_family": "type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "type constraint",
    "mapped_violation_constraint_qid": "Q21503250",
    "mapped_violation_family": "type",
    "mapped_violation_reason": "type_prefix_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 3,
    "removed_values": [
      "Q123705",
      "Q515",
      "Q7930989"
    ],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q3895768"
    ],
    "semantic_changed_qualifier_properties": [
      "P2308"
    ],
    "semantic_removed_value_count": 3,
    "semantic_removed_values": [
      "Q123705",
      "Q515",
      "Q7930989"
    ],
    "set_operation": "mixed",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "type constraint",
    "target_constraint_qid": "Q21503250",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "type",
    "value_overlap_with_report_qids": [
      "Q123705",
      "Q515",
      "Q7930989"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
  }
]
```

---

## 004. `reform_Q11385613_P1786_2050398987`

| Field | Value |
|---|---|
| qid | Q11385613 |
| property | P1786 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q108139345 label in language constraint |
| group_key | TBOX::P1786::2050398987 |
| tbox_revision_key | TBOX::P1786::2050398987 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "label in language constraint",
  "decision_constraint_type_qid": "Q108139345"
}
```

#### Repair Target

```json
{
  "author": "Trường Mộc",
  "kind": "T_BOX",
  "property_revision_id": 2050398987,
  "property_revision_prev": 2026755025
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-01-10T11:36:15",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1786",
  "report_revision_new": 2050511635,
  "report_revision_old": 2047711677,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
  "report_violation_types": [
    "Allowed qualifiers",
    "Mandatory Qualifiers",
    "Label in zh language"
  ],
  "value": null,
  "value_current_2026": [
    "月輪大師"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "name given to a person after death (East Asia)",
    "label": "posthumous name"
  },
  "qid": {
    "description": "kalligraaf",
    "label": "Shunjō"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "label in language constraint",
    "qid": "Q108139345"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q108139345",
      "mapped_violation_family": "label_in_language",
      "violation_name": "Label in zh language"
    },
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21510851",
      "mapped_violation_family": "allowed_qualifier",
      "violation_name": "Allowed qualifiers"
    },
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21510856",
      "mapped_violation_family": "mandatory_qualifier",
      "violation_name": "Mandatory Qualifiers"
    }
  ],
  "candidate_violation_names": [
    "Allowed qualifiers",
    "Mandatory Qualifiers",
    "Label in zh language"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q108139345"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q108139345"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "label-in-language constraint",
  "mapped_report_constraint_qid": "Q108139345",
  "mapped_report_family": "label_in_language",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "label-in-language constraint",
  "mapped_violation_constraint_qid": "Q108139345",
  "mapped_violation_family": "label_in_language",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Label in zh language",
  "semantic_added_value_count": 3,
  "semantic_added_values": [
    "ja",
    "ko",
    "vi"
  ],
  "semantic_changed_qualifier_properties": [
    "P424"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "label-in-language constraint",
  "target_constraint_qid": "Q108139345",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 3,
  "added_values": [
    "ja",
    "ko",
    "vi"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q108139345"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q108139345"
  ],
  "changed_qualifier_properties": [
    "P424"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "label-in-language constraint",
  "mapped_report_constraint_qid": "Q108139345",
  "mapped_report_family": "label_in_language",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Label in zh language",
  "semantic_added_values": [
    "ja",
    "ko",
    "vi"
  ],
  "semantic_changed_qualifier_properties": [
    "P424"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "label-in-language constraint",
  "target_constraint_qid": "Q108139345",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": true
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Trường Mộc",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "a59a1d3b5d2a5e9147602c8ae7089004484502c2",
  "hash_before": "5c79afefc2e6c59b4f686ea67fdd14ec922d6ccd",
  "property_revision_id": 2050398987,
  "property_revision_prev": 2026755025,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "mapped_violation_constraint_qid": "Q108139345",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Label in zh language"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 3,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P424"
    ],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 3,
    "added_values": [
      "ja",
      "ko",
      "vi"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q108139345",
        "mapped_violation_family": "label_in_language",
        "violation_name": "Label in zh language"
      },
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21510851",
        "mapped_violation_family": "allowed_qualifier",
        "violation_name": "Allowed qualifiers"
      },
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21510856",
        "mapped_violation_family": "mandatory_qualifier",
        "violation_name": "Mandatory Qualifiers"
      }
    ],
    "candidate_violation_names": [
      "Allowed qualifiers",
      "Mandatory Qualifiers",
      "Label in zh language"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q108139345"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q108139345"
    ],
    "changed_qualifier_properties": [
      "P424"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "label-in-language constraint",
    "mapped_report_constraint_qid": "Q108139345",
    "mapped_report_family": "label_in_language",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "label-in-language constraint",
    "mapped_violation_constraint_qid": "Q108139345",
    "mapped_violation_family": "label_in_language",
    "mapped_violation_reason": "label_language_report_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P424"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Label in zh language",
    "semantic_added_value_count": 3,
    "semantic_added_values": [
      "ja",
      "ko",
      "vi"
    ],
    "semantic_changed_qualifier_properties": [
      "P424"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "label-in-language constraint",
    "target_constraint_qid": "Q108139345",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "label_in_language",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Label in zh language"
  }
]
```

---

## 005. `reform_Q114622900_P9559_1751255376`

| Field | Value |
|---|---|
| qid | Q114622900 |
| property | P9559 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q21502410 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21502404 format constraint |
| group_key | TBOX::P9559::1751255376 |
| tbox_revision_key | TBOX::P9559::1751255376 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "author": "Vojtěch Dostál",
  "kind": "T_BOX",
  "property_revision_id": 1751255376,
  "property_revision_prev": 1550090372
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-10-15T07:45:23",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P9559",
  "report_revision_new": 1751296861,
  "report_revision_old": 1750687285,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Unique value"
  ],
  "value": null,
  "value_current_2026": [
    "ABH002"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "persistent identifier of Czech libraries",
    "label": "Sigla ID"
  },
  "qid": {
    "description": "knihovna v Praze",
    "label": "Library of African Information Centre"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21502404",
      "mapped_violation_family": "format",
      "violation_name": "Format"
    },
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21502410",
      "mapped_violation_family": "distinct_values",
      "violation_name": "Unique value"
    }
  ],
  "candidate_violation_names": [
    "Format",
    "Unique value"
  ],
  "causality_match_level": "exact_constraint_and_value_match",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21502404"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502404"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "format_regex_qualifier_changed",
  "compatible_overlap_used": true,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "format constraint",
  "mapped_report_constraint_qid": "Q21502404",
  "mapped_report_family": "format",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "format constraint",
  "mapped_violation_constraint_qid": "Q21502404",
  "mapped_violation_family": "format",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "unknown",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Format",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "[A-Z]{2}[A-H][0-9]{3}"
  ],
  "semantic_changed_qualifier_properties": [
    "P1793"
  ],
  "semantic_removed_value_count": 1,
  "semantic_removed_values": [
    "[A-Z]{2}[A-G][0-9]{3}"
  ],
  "set_operation": "mixed",
  "set_semantics": "unknown",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "format constraint",
  "target_constraint_qid": "Q21502404",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "[A-Z]{2}[A-H][0-9]{3}"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21502404"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502404"
  ],
  "changed_qualifier_properties": [
    "P1793"
  ],
  "compatible_overlap_reason": "format_regex_qualifier_changed",
  "compatible_overlap_used": true,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "format constraint",
  "mapped_report_constraint_qid": "Q21502404",
  "mapped_report_family": "format",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "unknown",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 1,
  "removed_values": [
    "[A-Z]{2}[A-G][0-9]{3}"
  ],
  "selected_violation_name": "Format",
  "semantic_added_values": [
    "[A-Z]{2}[A-H][0-9]{3}"
  ],
  "semantic_changed_qualifier_properties": [
    "P1793"
  ],
  "semantic_removed_values": [
    "[A-Z]{2}[A-G][0-9]{3}"
  ],
  "set_operation": "mixed",
  "set_semantics": "unknown",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "format constraint",
  "target_constraint_qid": "Q21502404",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Vojtěch Dostál",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "60692028ddd3567752cb63c98c6b958095dec70a",
  "hash_before": "4c679cf431eca53a4e908793ca0c592caa61c6a6",
  "property_revision_id": 1751255376,
  "property_revision_prev": 1550090372,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_and_value_match",
    "mapped_violation_constraint_qid": "Q21502404",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Format"
  },
  {
    "result": "Q21502404",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "unknown",
    "property_ids": [
      "P1793"
    ],
    "removed_value_count": 1,
    "result": "SCHEMA_UPDATE",
    "set_operation": "mixed",
    "set_semantics": "unknown",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "[A-Z]{2}[A-H][0-9]{3}"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21502404",
        "mapped_violation_family": "format",
        "violation_name": "Format"
      },
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21502410",
        "mapped_violation_family": "distinct_values",
        "violation_name": "Unique value"
      }
    ],
    "candidate_violation_names": [
      "Format",
      "Unique value"
    ],
    "causality_match_level": "exact_constraint_and_value_match",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21502404"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21502404"
    ],
    "changed_qualifier_properties": [
      "P1793"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "format_regex_qualifier_changed",
    "compatible_overlap_used": true,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "format constraint",
    "mapped_report_constraint_qid": "Q21502404",
    "mapped_report_family": "format",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "format constraint",
    "mapped_violation_constraint_qid": "Q21502404",
    "mapped_violation_family": "format",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "unknown",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P1793"
    ],
    "removed_value_count": 1,
    "removed_values": [
      "[A-Z]{2}[A-G][0-9]{3}"
    ],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Format",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "[A-Z]{2}[A-H][0-9]{3}"
    ],
    "semantic_changed_qualifier_properties": [
      "P1793"
    ],
    "semantic_removed_value_count": 1,
    "semantic_removed_values": [
      "[A-Z]{2}[A-G][0-9]{3}"
    ],
    "set_operation": "mixed",
    "set_semantics": "unknown",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "format constraint",
    "target_constraint_qid": "Q21502404",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "format",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Format"
  }
]
```

---

## 006. `reform_Q11940_P190_2440746684`

| Field | Value |
|---|---|
| qid | Q11940 |
| property | P190 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P190::2440746684 |
| tbox_revision_key | TBOX::P190::2440746684 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Jpnbot",
  "kind": "T_BOX",
  "property_revision_id": 2440746684,
  "property_revision_prev": 2440687529
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-11T13:55:51",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P190",
  "report_revision_new": 2440879198,
  "report_revision_old": 2440425151,
  "report_violation_type": "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
  "report_violation_type_descriptions_en": [
    "territorial entity for administration purposes, with or without its own local government",
    "large human settlement",
    "type of human settlement",
    "geographically localized community within a larger city, town or suburb",
    "place of any size, in which people permanently live",
    "railway facility where trains regularly stop to load or unload passengers and/or freight",
    "institution for the education of students by teachers",
    "imposing structure created to commemorate a person or event, or used for that purpose"
  ],
  "report_violation_type_labels_en": [
    "administrative territorial entity",
    "city",
    "city or town",
    "neighborhood",
    "human settlement",
    "railway station",
    "school",
    "monument"
  ],
  "report_violation_type_normalized": "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
  "report_violation_type_qids": [
    "Q56061",
    "Q515",
    "Q7930989",
    "Q123705",
    "Q486972",
    "Q55488",
    "Q3914",
    "Q4989906"
  ],
  "report_violation_type_raw": "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
  "report_violation_types": [
    "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
    "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
  ],
  "value": null,
  "value_current_2026": [
    "Q3967502"
  ],
  "value_current_2026_descriptions_en": [
    "fictional city where Darkwing Duck is set"
  ],
  "value_current_2026_labels_en": [
    "Saint Canard"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "twin towns, sister cities, twinned municipalities and other localities that have a partnership or cooperative agreement, either legally or informally acknowledged by their governments",
    "label": "twinned administrative body"
  },
  "qid": {
    "description": "narrative location of the Donald Duck universe",
    "label": "Duckburg"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "symmetric constraint",
    "qid": "Q21510862"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
    },
    {
      "candidate_causality_match_level": "value_or_property_overlap_only",
      "candidate_score": 20,
      "mapped_violation_constraint_qid": "Q21510865",
      "mapped_violation_family": "value_type",
      "violation_name": "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
    }
  ],
  "candidate_violation_names": [
    "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
    "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
  ],
  "causality_match_level": "exact_constraint_and_value_match",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "type_compatible_report_argument_overlap",
  "compatible_overlap_used": true,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [
    "Q123705",
    "Q515",
    "Q7930989"
  ],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q3895768"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_value_count": 3,
  "semantic_removed_values": [
    "Q123705",
    "Q515",
    "Q7930989"
  ],
  "set_operation": "mixed",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "type constraint",
  "target_constraint_qid": "Q21503250",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [
    "Q123705",
    "Q515",
    "Q7930989"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q3895768"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": [
    "P2308"
  ],
  "compatible_overlap_reason": "type_compatible_report_argument_overlap",
  "compatible_overlap_used": true,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 3,
  "removed_values": [
    "Q123705",
    "Q515",
    "Q7930989"
  ],
  "selected_violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
  "semantic_added_values": [
    "Q3895768"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_values": [
    "Q123705",
    "Q515",
    "Q7930989"
  ],
  "set_operation": "mixed",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "type constraint",
  "target_constraint_qid": "Q21503250",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Jpnbot",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "e0b80a748b26f7c5d8e261f31f900416046d827e",
  "hash_before": "8a70cd3a253177cd6d177081d5c6dc307edcd89f",
  "property_revision_id": 2440746684,
  "property_revision_prev": 2440687529,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_and_value_match",
    "mapped_violation_constraint_qid": "Q21503250",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 3,
    "result": "SCHEMA_UPDATE",
    "set_operation": "mixed",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q3895768"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
      },
      {
        "candidate_causality_match_level": "value_or_property_overlap_only",
        "candidate_score": 20,
        "mapped_violation_constraint_qid": "Q21510865",
        "mapped_violation_family": "value_type",
        "violation_name": "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
      }
    ],
    "candidate_violation_names": [
      "Value type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
      "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
    ],
    "causality_match_level": "exact_constraint_and_value_match",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21503250"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503250"
    ],
    "changed_qualifier_properties": [
      "P2308"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "type_compatible_report_argument_overlap",
    "compatible_overlap_used": true,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [
      "Q123705",
      "Q515",
      "Q7930989"
    ],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "type constraint",
    "mapped_report_constraint_qid": "Q21503250",
    "mapped_report_family": "type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "type constraint",
    "mapped_violation_constraint_qid": "Q21503250",
    "mapped_violation_family": "type",
    "mapped_violation_reason": "type_prefix_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 3,
    "removed_values": [
      "Q123705",
      "Q515",
      "Q7930989"
    ],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q3895768"
    ],
    "semantic_changed_qualifier_properties": [
      "P2308"
    ],
    "semantic_removed_value_count": 3,
    "semantic_removed_values": [
      "Q123705",
      "Q515",
      "Q7930989"
    ],
    "set_operation": "mixed",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "type constraint",
    "target_constraint_qid": "Q21503250",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "type",
    "value_overlap_with_report_qids": [
      "Q123705",
      "Q515",
      "Q7930989"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|56061, Q|515, Q|7930989, Q|123705, Q|486972, Q|55488, Q|3914, Q|4989906"
  }
]
```

---

## 007. `reform_Q12106333_P31_2440675178`

| Field | Value |
|---|---|
| qid | Q12106333 |
| property | P31 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q21510851 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q52558054 none-of constraint |
| group_key | TBOX::P31::2440675178 |
| tbox_revision_key | TBOX::P31::2440675178 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "none-of constraint",
  "decision_constraint_type_qid": "Q52558054"
}
```

#### Repair Target

```json
{
  "author": "Clemens Dulcis",
  "kind": "T_BOX",
  "property_revision_id": 2440675178,
  "property_revision_prev": 2439474684
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-11T19:29:00",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
  "report_revision_new": 2440991063,
  "report_revision_old": 2440479307,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": null,
  "value_current_2026": [
    "Q124423387",
    "Q4263830"
  ],
  "value_current_2026_descriptions_en": [
    "language used in the field of literary studies",
    "category of literary works distinguished by formal characteristics without consideration of content"
  ],
  "value_current_2026_labels_en": [
    "literary terminology",
    "literary form"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
    "label": "instance of"
  },
  "qid": {
    "description": "collection of poems by a single author published together. See also poetry anthology (Q19357149)",
    "label": "poem collection"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q52558054",
      "mapped_violation_family": "none_of",
      "violation_name": "None of"
    }
  ],
  "candidate_violation_names": [
    "None of"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q52558054"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q52558054"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Use \"album\" (Q482994) with \"instance of\" (P31), use the specific form with \"form of creative work\" (P7937).@en"
  ],
  "ignored_changed_qualifier_properties": [
    "P6607"
  ],
  "ignored_removed_values": [
    "Use Q482994 with P31, use the specific form with P7937@en"
  ],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "none-of constraint",
  "mapped_report_constraint_qid": "Q52558054",
  "mapped_report_family": "none_of",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "none-of constraint",
  "mapped_violation_constraint_qid": "Q52558054",
  "mapped_violation_family": "none_of",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "forbidden",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "None of",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "forbidden",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "none-of constraint",
  "target_constraint_qid": "Q52558054",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 0,
  "added_values": [],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q52558054"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q52558054"
  ],
  "changed_qualifier_properties": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Use \"album\" (Q482994) with \"instance of\" (P31), use the specific form with \"form of creative work\" (P7937).@en"
  ],
  "ignored_changed_qualifier_properties": [
    "P6607"
  ],
  "ignored_removed_values": [
    "Use Q482994 with P31, use the specific form with P7937@en"
  ],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "none-of constraint",
  "mapped_report_constraint_qid": "Q52558054",
  "mapped_report_family": "none_of",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "forbidden",
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "None of",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "forbidden",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "none-of constraint",
  "target_constraint_qid": "Q52558054",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Clemens Dulcis",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "a812be9148ef19ef44ea6a8a21b9f347000b2c31",
  "hash_before": "ce4232e77bc7d6290fc000803a238c5704d79301",
  "property_revision_id": 2440675178,
  "property_revision_prev": 2439474684,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only",
    "mapped_violation_constraint_qid": "Q52558054",
    "result": true,
    "step": "causality_filter",
    "violation_name": "None of"
  },
  {
    "result": "Q52558054",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 0,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "forbidden",
    "property_ids": [],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "unchanged",
    "set_semantics": "forbidden",
    "step": "set_semantics"
  },
  {
    "added_value_count": 0,
    "added_values": [],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q52558054",
        "mapped_violation_family": "none_of",
        "violation_name": "None of"
      }
    ],
    "candidate_violation_names": [
      "None of"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q52558054"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q52558054"
    ],
    "changed_qualifier_properties": [],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [
      "Use \"album\" (Q482994) with \"instance of\" (P31), use the specific form with \"form of creative work\" (P7937).@en"
    ],
    "ignored_changed_qualifier_properties": [
      "P6607"
    ],
    "ignored_removed_values": [
      "Use Q482994 with P31, use the specific form with P7937@en"
    ],
    "ignored_value_count": 2,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "none-of constraint",
    "mapped_report_constraint_qid": "Q52558054",
    "mapped_report_family": "none_of",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "none-of constraint",
    "mapped_violation_constraint_qid": "Q52558054",
    "mapped_violation_family": "none_of",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "forbidden",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "None of",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "unchanged",
    "set_semantics": "forbidden",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "none-of constraint",
    "target_constraint_qid": "Q52558054",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "none_of",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "None of"
  }
]
```

---

## 008. `reform_Q123338933_P1786_2050398987`

| Field | Value |
|---|---|
| qid | Q123338933 |
| property | P1786 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q108139345 label in language constraint |
| group_key | TBOX::P1786::2050398987 |
| tbox_revision_key | TBOX::P1786::2050398987 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "label in language constraint",
  "decision_constraint_type_qid": "Q108139345"
}
```

#### Repair Target

```json
{
  "author": "Trường Mộc",
  "kind": "T_BOX",
  "property_revision_id": 2050398987,
  "property_revision_prev": 2026755025
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-01-10T11:36:15",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1786",
  "report_revision_new": 2050511635,
  "report_revision_old": 2047711677,
  "report_violation_type": "Mandatory Qualifiers",
  "report_violation_type_normalized": "Mandatory Qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Mandatory Qualifiers",
  "report_violation_types": [
    "Mandatory Qualifiers",
    "Label in zh language",
    "Item P|570"
  ],
  "value": null,
  "value_current_2026": [
    "Từ Hạnh"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "name given to a person after death (East Asia)",
    "label": "posthumous name"
  },
  "qid": {
    "description": null,
    "label": "Đặng Từ Hạnh"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "label in language constraint",
    "qid": "Q108139345"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q108139345",
      "mapped_violation_family": "label_in_language",
      "violation_name": "Label in zh language"
    },
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21510856",
      "mapped_violation_family": "mandatory_qualifier",
      "violation_name": "Mandatory Qualifiers"
    },
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|570"
    }
  ],
  "candidate_violation_names": [
    "Mandatory Qualifiers",
    "Label in zh language",
    "Item P|570"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q108139345"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q108139345"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "label-in-language constraint",
  "mapped_report_constraint_qid": "Q108139345",
  "mapped_report_family": "label_in_language",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "label-in-language constraint",
  "mapped_violation_constraint_qid": "Q108139345",
  "mapped_violation_family": "label_in_language",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Label in zh language",
  "semantic_added_value_count": 3,
  "semantic_added_values": [
    "ja",
    "ko",
    "vi"
  ],
  "semantic_changed_qualifier_properties": [
    "P424"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "label-in-language constraint",
  "target_constraint_qid": "Q108139345",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 3,
  "added_values": [
    "ja",
    "ko",
    "vi"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q108139345"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q108139345"
  ],
  "changed_qualifier_properties": [
    "P424"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "label-in-language constraint",
  "mapped_report_constraint_qid": "Q108139345",
  "mapped_report_family": "label_in_language",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Label in zh language",
  "semantic_added_values": [
    "ja",
    "ko",
    "vi"
  ],
  "semantic_changed_qualifier_properties": [
    "P424"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "label-in-language constraint",
  "target_constraint_qid": "Q108139345",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": true
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Trường Mộc",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "a59a1d3b5d2a5e9147602c8ae7089004484502c2",
  "hash_before": "5c79afefc2e6c59b4f686ea67fdd14ec922d6ccd",
  "property_revision_id": 2050398987,
  "property_revision_prev": 2026755025,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "mapped_violation_constraint_qid": "Q108139345",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Label in zh language"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 3,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P424"
    ],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 3,
    "added_values": [
      "ja",
      "ko",
      "vi"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q108139345",
        "mapped_violation_family": "label_in_language",
        "violation_name": "Label in zh language"
      },
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21510856",
        "mapped_violation_family": "mandatory_qualifier",
        "violation_name": "Mandatory Qualifiers"
      },
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|570"
      }
    ],
    "candidate_violation_names": [
      "Mandatory Qualifiers",
      "Label in zh language",
      "Item P|570"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q108139345"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q108139345"
    ],
    "changed_qualifier_properties": [
      "P424"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "label-in-language constraint",
    "mapped_report_constraint_qid": "Q108139345",
    "mapped_report_family": "label_in_language",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "label-in-language constraint",
    "mapped_violation_constraint_qid": "Q108139345",
    "mapped_violation_family": "label_in_language",
    "mapped_violation_reason": "label_language_report_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P424"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Label in zh language",
    "semantic_added_value_count": 3,
    "semantic_added_values": [
      "ja",
      "ko",
      "vi"
    ],
    "semantic_changed_qualifier_properties": [
      "P424"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "label-in-language constraint",
    "target_constraint_qid": "Q108139345",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "label_in_language",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Label in zh language"
  }
]
```

---

## 009. `reform_Q136214804_P921_2447527451`

| Field | Value |
|---|---|
| qid | Q136214804 |
| property | P921 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P921::2447527451 |
| tbox_revision_key | TBOX::P921::2447527451 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "YotaMoteuchi",
  "kind": "T_BOX",
  "property_revision_id": 2447527451,
  "property_revision_prev": 2438998167
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T11:28:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
  "report_revision_new": 2447751779,
  "report_revision_old": 2447354743,
  "report_violation_type": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "any set of human beings",
    "reference in one place in a book to information at another place in the same work",
    "category of creative works based on stylistic, thematic or technical criteria",
    "temporary and scheduled happening, like a conference, festival, competition or similar",
    "program of study, or unit of teaching that typically lasts one academic term",
    "intangible asset consisting of ownership of ideas and processes",
    "theme or subject in a work of art",
    "a restaurant based around a concept or intellectual property",
    "non-repayable funds disbursed by one party to a recipient",
    "human subject research in medicine",
    "process that attempts to determine the facts of a crime and circumstances",
    "single content rating in a rating system",
    "transgression or alleged transgression resulting in public outrage",
    "disclosure of confidential or nonpublic information to unauthorized parties",
    "collection of materials with some unifying characteristic, housed in an archive",
    "set of purposely gathered physical or digital objects with some common characteristics",
    "experience of intense sexual arousal to atypical objects, situations, or individuals",
    "section of learning or teaching into which a wider learning content is divided",
    "process that has the aim of augmenting knowledge, resolving doubt, or solving a problem",
    "word or an unspaced phrase prefixed with the number sign, used to categorise a topic",
    "project of one or more scientists, or of an organization in a scientific field",
    "scientific procedure carried out to support, refute, or validate a hypothesis",
    "topic viewed from a historical point of view",
    "... omitted 62 items"
  ],
  "report_violation_type_labels_en": [
    "work",
    "group of humans",
    "cross-reference",
    "genre",
    "event",
    "course",
    "intellectual property",
    "artistic theme",
    "theme restaurant",
    "grant",
    "clinical trial",
    "criminal investigation",
    "content rating category",
    "scandal",
    "information leak",
    "archival collection",
    "collection",
    "paraphilia",
    "lesson",
    "inquiry",
    "#",
    "science project",
    "experiment",
    "aspect of history",
    "... omitted 62 items"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
  "report_violation_type_qids": [
    "Q386724",
    "Q16334295",
    "Q1302249",
    "Q483394",
    "Q1656682",
    "Q600134",
    "Q131257",
    "Q1406161",
    "Q676586",
    "Q230788",
    "Q30612",
    "Q1964968",
    "Q23649976",
    "Q192909",
    "Q2904148",
    "Q9388534",
    "Q2668072",
    "Q178059",
    "Q379833",
    "Q21004260",
    "Q278485",
    "Q1298668",
    "Q101965",
    "Q17524420",
    "... omitted 62 items"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
  "value": null,
  "value_current_2026": [
    "Q17172850"
  ],
  "value_current_2026_descriptions_en": [
    "human voice as musical instrument"
  ],
  "value_current_2026_labels_en": [
    "voice"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "primary topic of a work or act of communication",
    "label": "main subject"
  },
  "qid": {
    "description": null,
    "label": "SOVT Exercises"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]"
    }
  ],
  "candidate_violation_names": [
    "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q3249551"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "type constraint",
  "target_constraint_qid": "Q21503250",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q3249551"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": [
    "P2308"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
  "semantic_added_values": [
    "Q3249551"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "type constraint",
  "target_constraint_qid": "Q21503250",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": true
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "YotaMoteuchi",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "d5ea1158d61a890f5b9b5973dbea33604cae9572",
  "hash_before": "27f65b7446fdc00965554e75fb3f409357651989",
  "property_revision_id": 2447527451,
  "property_revision_prev": 2438998167,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "mapped_violation_constraint_qid": "Q21503250",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q3249551"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]"
      }
    ],
    "candidate_violation_names": [
      "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503250"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503250"
    ],
    "changed_qualifier_properties": [
      "P2308"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "type constraint",
    "mapped_report_constraint_qid": "Q21503250",
    "mapped_report_family": "type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "type constraint",
    "mapped_violation_constraint_qid": "Q21503250",
    "mapped_violation_family": "type",
    "mapped_violation_reason": "type_prefix_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q3249551"
    ],
    "semantic_changed_qualifier_properties": [
      "P2308"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "type constraint",
    "target_constraint_qid": "Q21503250",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "type",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]"
  }
]
```

---

## 010. `reform_Q136451000_P710_2438088264`

| Field | Value |
|---|---|
| qid | Q136451000 |
| property | P710 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21510865 value-type constraint |
| group_key | TBOX::P710::2438088264 |
| tbox_revision_key | TBOX::P710::2438088264 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "value-type constraint",
  "decision_constraint_type_qid": "Q21510865"
}
```

#### Repair Target

```json
{
  "author": "Einar Myre",
  "kind": "T_BOX",
  "property_revision_id": 2438088264,
  "property_revision_prev": 2435131607
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-10T09:26:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P710",
  "report_revision_new": 2440401012,
  "report_revision_old": 2439956468,
  "report_violation_type": "Value type Q|24229398, Q|16334298, Q|795052, Q|115537581",
  "report_violation_type_descriptions_en": [
    "distinct and identifiable entity with agency, capable of performing actions",
    "set of live individual entities of any nature",
    "individual person or organism",
    "character known only from narrations (fictional or in a factual manner) without a proof of existence; includes fictional, mythical, legendary or religious characters and similar"
  ],
  "report_violation_type_labels_en": [
    "being",
    "group of living things",
    "individual",
    "imaginary character"
  ],
  "report_violation_type_normalized": "Value type Q|24229398, Q|16334298, Q|795052, Q|115537581",
  "report_violation_type_qids": [
    "Q24229398",
    "Q16334298",
    "Q795052",
    "Q115537581"
  ],
  "report_violation_type_raw": "Value type Q|24229398, Q|16334298, Q|795052, Q|115537581",
  "value": null,
  "value_current_2026": [
    "Q12557",
    "Q2662180",
    "Q1059046",
    "Q728528"
  ],
  "value_current_2026_descriptions_en": [
    "13th- and 14th-century empire originating in Mongolia",
    "Uyghur kingdom (832-1132)",
    "tribal confederation of 12th century Mongolia",
    "Extinct Turkic people"
  ],
  "value_current_2026_labels_en": [
    "Mongol Empire",
    "Qocho",
    "Merkit people",
    "Kankalis"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "person, group of people or organization (object) that actively takes/took part in an event or process (subject). Preferably qualify with \"object has role\" (P3831). Use P1923 for participants that are teams.",
    "label": "participant"
  },
  "qid": {
    "description": "13th-century battle",
    "label": "Chem River Battle"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21510865",
      "mapped_violation_family": "value_type",
      "violation_name": "Value type Q|24229398, Q|16334298, Q|795052, Q|115537581"
    }
  ],
  "candidate_violation_names": [
    "Value type Q|24229398, Q|16334298, Q|795052, Q|115537581"
  ],
  "causality_match_level": "exact_constraint_and_value_match",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "value_type_compatible_report_argument_overlap",
  "compatible_overlap_used": true,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [
    "Q115537581"
  ],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "value-type constraint",
  "mapped_violation_constraint_qid": "Q21510865",
  "mapped_violation_family": "value_type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Value type Q|24229398, Q|16334298, Q|795052, Q|115537581",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q115537581"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_value_count": 1,
  "semantic_removed_values": [
    "Q95074"
  ],
  "set_operation": "mixed",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "value-type constraint",
  "target_constraint_qid": "Q21510865",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [
    "Q115537581"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q115537581"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "changed_qualifier_properties": [
    "P2308"
  ],
  "compatible_overlap_reason": "value_type_compatible_report_argument_overlap",
  "compatible_overlap_used": true,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 1,
  "removed_values": [
    "Q95074"
  ],
  "selected_violation_name": "Value type Q|24229398, Q|16334298, Q|795052, Q|115537581",
  "semantic_added_values": [
    "Q115537581"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_values": [
    "Q95074"
  ],
  "set_operation": "mixed",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "value-type constraint",
  "target_constraint_qid": "Q21510865",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Einar Myre",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "ccb72590df61392e35f8f274464b81fd4a1fc5da",
  "hash_before": "f46e315093b951edb7de4e6add897b21d6dae6df",
  "property_revision_id": 2438088264,
  "property_revision_prev": 2435131607,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_and_value_match",
    "mapped_violation_constraint_qid": "Q21510865",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Value type Q|24229398, Q|16334298, Q|795052, Q|115537581"
  },
  {
    "result": "Q21510865",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 1,
    "result": "SCHEMA_UPDATE",
    "set_operation": "mixed",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q115537581"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21510865",
        "mapped_violation_family": "value_type",
        "violation_name": "Value type Q|24229398, Q|16334298, Q|795052, Q|115537581"
      }
    ],
    "candidate_violation_names": [
      "Value type Q|24229398, Q|16334298, Q|795052, Q|115537581"
    ],
    "causality_match_level": "exact_constraint_and_value_match",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510865"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510865"
    ],
    "changed_qualifier_properties": [
      "P2308"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "value_type_compatible_report_argument_overlap",
    "compatible_overlap_used": true,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [
      "Q115537581"
    ],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "value-type constraint",
    "mapped_report_constraint_qid": "Q21510865",
    "mapped_report_family": "value_type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "value-type constraint",
    "mapped_violation_constraint_qid": "Q21510865",
    "mapped_violation_family": "value_type",
    "mapped_violation_reason": "value_type_prefix_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 1,
    "removed_values": [
      "Q95074"
    ],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Value type Q|24229398, Q|16334298, Q|795052, Q|115537581",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q115537581"
    ],
    "semantic_changed_qualifier_properties": [
      "P2308"
    ],
    "semantic_removed_value_count": 1,
    "semantic_removed_values": [
      "Q95074"
    ],
    "set_operation": "mixed",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "value-type constraint",
    "target_constraint_qid": "Q21510865",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "value_type",
    "value_overlap_with_report_qids": [
      "Q115537581"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Value type Q|24229398, Q|16334298, Q|795052, Q|115537581"
  }
]
```

---

## 011. `reform_Q136791783_P856_2442691260`

| Field | Value |
|---|---|
| qid | Q136791783 |
| property | P856 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21502410 distinct-values constraint |
| group_key | TBOX::P856::2442691260 |
| tbox_revision_key | TBOX::P856::2442691260 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "distinct-values constraint",
  "decision_constraint_type_qid": "Q21502410"
}
```

#### Repair Target

```json
{
  "author": "Legonin",
  "kind": "T_BOX",
  "property_revision_id": 2442691260,
  "property_revision_prev": 2442687184
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-18T17:47:44",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P856",
  "report_revision_new": 2443817852,
  "report_revision_old": 2443378012,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "http://dx.doi.org/10.1158/0008-5472.25224631.v1"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "URL of the official page of an item (current or former). Usage: If a listed URL no longer points to the official website, do not remove it, but see the \"Hijacked or dead websites\" section of the Talk page",
    "label": "official website"
  },
  "qid": {
    "description": null,
    "label": "Table S11 from R-Loop Accumulation in Spliceosome Mutant Leukemias Confers Sensitivity to PARP1 Inhibition by Triggering Transcription–Replication Conflicts"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21502410",
      "mapped_violation_family": "distinct_values",
      "violation_name": "Unique value"
    }
  ],
  "candidate_violation_names": [
    "Unique value"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21502410"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502410"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q2994765",
    "Q76267992"
  ],
  "ignored_changed_qualifier_properties": [
    "P2303"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "distinct-values constraint",
  "mapped_report_constraint_qid": "Q21502410",
  "mapped_report_family": "distinct_values",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "distinct-values constraint",
  "mapped_violation_constraint_qid": "Q21502410",
  "mapped_violation_family": "distinct_values",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Unique value",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "distinct-values constraint",
  "target_constraint_qid": "Q21502410",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 0,
  "added_values": [],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21502410"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502410"
  ],
  "changed_qualifier_properties": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q2994765",
    "Q76267992"
  ],
  "ignored_changed_qualifier_properties": [
    "P2303"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "distinct-values constraint",
  "mapped_report_constraint_qid": "Q21502410",
  "mapped_report_family": "distinct_values",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Unique value",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "distinct-values constraint",
  "target_constraint_qid": "Q21502410",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Legonin",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "9cebba9c089543e647dd61637ec2b182fddf9841",
  "hash_before": "eb15cf149d21b5d909a97543682e7756361d93c0",
  "property_revision_id": 2442691260,
  "property_revision_prev": 2442687184,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only",
    "mapped_violation_constraint_qid": "Q21502410",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Unique value"
  },
  {
    "result": "Q21502410",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 0,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_ids": [],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 0,
    "added_values": [],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21502410",
        "mapped_violation_family": "distinct_values",
        "violation_name": "Unique value"
      }
    ],
    "candidate_violation_names": [
      "Unique value"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21502410"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21502410"
    ],
    "changed_qualifier_properties": [],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [
      "Q2994765",
      "Q76267992"
    ],
    "ignored_changed_qualifier_properties": [
      "P2303"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 2,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "distinct-values constraint",
    "mapped_report_constraint_qid": "Q21502410",
    "mapped_report_family": "distinct_values",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "distinct-values constraint",
    "mapped_violation_constraint_qid": "Q21502410",
    "mapped_violation_family": "distinct_values",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Unique value",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "distinct-values constraint",
    "target_constraint_qid": "Q21502410",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "distinct_values",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Unique value"
  }
]
```

---

## 012. `reform_Q15963668_P735_2442352845`

| Field | Value |
|---|---|
| qid | Q15963668 |
| property | P735 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q53869507 property scope constraint |
| group_key | TBOX::P735::2442352845 |
| tbox_revision_key | TBOX::P735::2442352845 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "property scope constraint",
  "decision_constraint_type_qid": "Q53869507"
}
```

#### Repair Target

```json
{
  "author": "Swpb",
  "kind": "T_BOX",
  "property_revision_id": 2442352845,
  "property_revision_prev": 2427927742
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-15T10:39:57",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P735",
  "report_revision_new": 2442615698,
  "report_revision_old": 2442267914,
  "report_violation_type": "Scope",
  "report_violation_type_normalized": "Scope",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Scope",
  "value": null,
  "value_current_2026": [
    "Q6968341",
    "Q6968300"
  ],
  "value_current_2026_descriptions_en": [
    "female given name (Наталья)",
    "female given name (Наталія)"
  ],
  "value_current_2026_labels_en": [
    "Natalya",
    "Nataliia"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "first name or another given name of this person; values used with the property should not link disambiguations nor family names",
    "label": "given name"
  },
  "qid": {
    "description": "Russian lawyer, politician and diplomat",
    "label": "Natalia Poklonskaya"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q53869507",
      "mapped_violation_family": "property_scope",
      "violation_name": "Scope"
    }
  ],
  "candidate_violation_names": [
    "Scope"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q53869507"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q54828449"
  ],
  "ignored_changed_qualifier_properties": [
    "P5314"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "property scope constraint",
  "mapped_report_constraint_qid": "Q53869507",
  "mapped_report_family": "property_scope",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "property scope constraint",
  "mapped_violation_constraint_qid": "Q53869507",
  "mapped_violation_family": "property_scope",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Scope",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "property scope constraint",
  "target_constraint_qid": "Q53869507",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 0,
  "added_values": [],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q53869507"
  ],
  "changed_qualifier_properties": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q54828449"
  ],
  "ignored_changed_qualifier_properties": [
    "P5314"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "property scope constraint",
  "mapped_report_constraint_qid": "Q53869507",
  "mapped_report_family": "property_scope",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Scope",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "property scope constraint",
  "target_constraint_qid": "Q53869507",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Swpb",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "0b94ae84accb701e29d60b04208ef59e98f5d398",
  "hash_before": "48220aad70210b2f09dd68709f0cc66601d47c2b",
  "property_revision_id": 2442352845,
  "property_revision_prev": 2427927742,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only",
    "mapped_violation_constraint_qid": "Q53869507",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Scope"
  },
  {
    "result": "Q53869507",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 0,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_ids": [],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 0,
    "added_values": [],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q53869507",
        "mapped_violation_family": "property_scope",
        "violation_name": "Scope"
      }
    ],
    "candidate_violation_names": [
      "Scope"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q53869507"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q53869507"
    ],
    "changed_qualifier_properties": [],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [
      "Q54828449"
    ],
    "ignored_changed_qualifier_properties": [
      "P5314"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "property scope constraint",
    "mapped_report_constraint_qid": "Q53869507",
    "mapped_report_family": "property_scope",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "property scope constraint",
    "mapped_violation_constraint_qid": "Q53869507",
    "mapped_violation_family": "property_scope",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P4680"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Scope",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "property scope constraint",
    "target_constraint_qid": "Q53869507",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "property_scope",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Scope"
  }
]
```

---

## 013. `reform_Q168356_P800_2439776691`

| Field | Value |
|---|---|
| qid | Q168356 |
| property | P800 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q21510865 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21510865 value-type constraint |
| group_key | TBOX::P800::2439776691 |
| tbox_revision_key | TBOX::P800::2439776691 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "value-type constraint",
  "decision_constraint_type_qid": "Q21510865"
}
```

#### Repair Target

```json
{
  "author": "AVDLCZ",
  "kind": "T_BOX",
  "property_revision_id": 2439776691,
  "property_revision_prev": 2436002170
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-10T09:20:50",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P800",
  "report_revision_new": 2440399321,
  "report_revision_old": 2439954550,
  "report_violation_type": "Value type Q|386724, Q|1151067, Q|151885, Q|11862829, Q|211364, Q|1293220, Q|17737, Q|1799072, Q|14208553, Q|246672, Q|1298668, Q|170584, Q|36534, Q|12772819, Q|65943, Q|2424752, Q|43229, Q|2668072, Q|8261, Q|618779, Q|9081, Q|1711593, Q|10541491, Q|7406919, Q|820655, Q|2095, Q|1914636, Q|1656682, Q|16510064, Q|22811462, Q|185925, Q|853614, Q|47461344, Q|27058, Q|193946, Q|12280, Q|124301146, Q|3658341, Q|131841, Q|41176, Q|811979, Q|483394, Q|106822736, Q|1731969... [truncated 90 chars]",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "prescription, including laws, regulations, instructions, guidelines, and social conventions; determinate method for performing any operation",
    "semantic unit understood in different ways, e.g. as mental representation, ability or abstract object (philosophy)",
    "academic field of study or profession",
    "guiding rule or inevitable consequence of something, such as the laws observed in nature",
    "phenomenon of the material world",
    "contemplative and rational type of abstract or generalizing thinking, or the results of such thinking",
    "planned path to reaching an objective",
    "unique or novel device, method, composition or process",
    "abstract object in mathematics",
    "project of one or more scientists, or of an organization in a scientific field",
    "collaborative enterprise, frequently involving research or design, that is carefully planned to achieve a particular aim",
    "process that results in the interconversion of chemical species",
    "act of detecting something new",
    "in mathematics, a statement that has been proved",
    "anything that can be offered to a market",
    "social entity established to meet needs or pursue goals",
    "set of purposely gathered physical or digital objects with some common characteristics",
    "narrative text, normally of a substantial length and in the form of prose describing a fictional and sequential story",
    "something given to a person or a group of people to recognize their merit or excellence",
    "mental possession of information or skills, often contributing to understanding",
    "collection of scholarly or scientific chapters written by different authors",
    "type of legal entity within certain legal system",
    "economic product that directly satisfies wants without producing a lasting asset",
    "... omitted 28 items"
  ],
  "report_violation_type_labels_en": [
    "work",
    "rule",
    "concept",
    "academic discipline",
    "principle",
    "physical phenomenon",
    "theory",
    "method",
    "invention",
    "mathematical object",
    "science project",
    "project",
    "chemical reaction",
    "discovery",
    "theorem",
    "product",
    "organization",
    "collection",
    "novel",
    "award",
    "knowledge",
    "edited volume",
    "legal form",
    "service",
    "... omitted 28 items"
  ],
  "report_violation_type_normalized": "Value type Q|386724, Q|1151067, Q|151885, Q|11862829, Q|211364, Q|1293220, Q|17737, Q|1799072, Q|14208553, Q|246672, Q|1298668, Q|170584, Q|36534, Q|12772819, Q|65943, Q|2424752, Q|43229, Q|2668072, Q|8261, Q|618779, Q|9081, Q|1711593, Q|10541491, Q|7406919, Q|820655, Q|2095, Q|1914636, Q|1656682, Q|16510064, Q|22811462, Q|185925, Q|853614, Q|47461344, Q|27058, Q|193946, Q|12280, Q|124301146, Q|3658341, Q|131841, Q|41176, Q|811979, Q|483394, Q|106822736, Q|1731969... [truncated 90 chars]",
  "report_violation_type_qids": [
    "Q386724",
    "Q1151067",
    "Q151885",
    "Q11862829",
    "Q211364",
    "Q1293220",
    "Q17737",
    "Q1799072",
    "Q14208553",
    "Q246672",
    "Q1298668",
    "Q170584",
    "Q36534",
    "Q12772819",
    "Q65943",
    "Q2424752",
    "Q43229",
    "Q2668072",
    "Q8261",
    "Q618779",
    "Q9081",
    "Q1711593",
    "Q10541491",
    "Q7406919",
    "... omitted 28 items"
  ],
  "report_violation_type_raw": "Value type Q|386724, Q|1151067, Q|151885, Q|11862829, Q|211364, Q|1293220, Q|17737, Q|1799072, Q|14208553, Q|246672, Q|1298668, Q|170584, Q|36534, Q|12772819, Q|65943, Q|2424752, Q|43229, Q|2668072, Q|8261, Q|618779, Q|9081, Q|1711593, Q|10541491, Q|7406919, Q|820655, Q|2095, Q|1914636, Q|1656682, Q|16510064, Q|22811462, Q|185925, Q|853614, Q|47461344, Q|27058, Q|193946, Q|12280, Q|124301146, Q|3658341, Q|131841, Q|41176, Q|811979, Q|483394, Q|106822736, Q|1731969... [truncated 90 chars]",
  "value": null,
  "value_current_2026": [
    "Q11973189",
    "Q165747",
    "Q11957214"
  ],
  "value_current_2026_descriptions_en": [
    "1802 poem written by Adam Oehlenschläger",
    "National anthem of Denmark",
    "1805 play by Adam Oehlenschläger"
  ],
  "value_current_2026_labels_en": [
    "The golden horns",
    "Der er et yndigt land",
    "Aladdin"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "notable scientific, artistic or literary work, or other work of significance among subject's works",
    "label": "notable work"
  },
  "qid": {
    "description": "Danish poet and playwright (1779–1850)",
    "label": "Adam Oehlenschläger"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21510865",
      "mapped_violation_family": "value_type",
      "violation_name": "Value type Q|386724, Q|1151067, Q|151885, Q|11862829, Q|211364, Q|1293220, Q|17737, Q|1799072, Q|14208553, Q|246672, Q|1298668, Q|170584, Q|36534, Q|12772819, Q|65943, Q|2424752, Q|43229, Q|2668072, Q|8261, Q|618779, Q|9081, Q|1711593, Q|10541491, Q|7406919, Q|820655, Q|2095, Q|1914636, Q|1656682, Q|16510064, Q|22811462, Q|185925, Q|853614, Q|47461344, Q|27058, Q|193946, Q|12280, Q|124301146, Q|3658341, Q|131841, Q|41176, Q|811979, Q|483394, Q|106822736, Q|1731969... [truncated 90 chars]"
    }
  ],
  "candidate_violation_names": [
    "Value type Q|386724, Q|1151067, Q|151885, Q|11862829, Q|211364, Q|1293220, Q|17737, Q|1799072, Q|14208553, Q|246672, Q|1298668, Q|170584, Q|36534, Q|12772819, Q|65943, Q|2424752, Q|43229, Q|2668072, Q|8261, Q|618779, Q|9081, Q|1711593, Q|10541491, Q|7406919, Q|820655, Q|2095, Q|1914636, Q|1656682, Q|16510064, Q|22811462, Q|185925, Q|853614, Q|47461344, Q|27058, Q|193946, Q|12280, Q|124301146, Q|3658341, Q|131841, Q|41176, Q|811979, Q|483394, Q|106822736, Q|1731969... [truncated 90 chars]"
  ],
  "causality_match_level": "exact_constraint_and_value_match",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "value_type_compatible_report_argument_overlap",
  "compatible_overlap_used": true,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [
    "Q246672"
  ],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "value-type constraint",
  "mapped_violation_constraint_qid": "Q21510865",
  "mapped_violation_family": "value_type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Value type Q|386724, Q|1151067, Q|151885, Q|11862829, Q|211364, Q|1293220, Q|17737, Q|1799072, Q|14208553, Q|246672, Q|1298668, Q|170584, Q|36534, Q|12772819, Q|65943, Q|2424752, Q|43229, Q|2668072, Q|8261, Q|618779, Q|9081, Q|1711593, Q|10541491, Q|7406919, Q|820655, Q|2095, Q|1914636, Q|1656682, Q|16510064, Q|22811462, Q|185925, Q|853614, Q|47461344, Q|27058, Q|193946, Q|12280, Q|124301146, Q|3658341, Q|131841, Q|41176, Q|811979, Q|483394, Q|106822736, Q|1731969... [truncated 90 chars]",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q246672"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_value_count": 1,
  "semantic_removed_values": [
    "Q486902"
  ],
  "set_operation": "mixed",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "value-type constraint",
  "target_constraint_qid": "Q21510865",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [
    "Q246672"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q246672"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "changed_qualifier_properties": [
    "P2308"
  ],
  "compatible_overlap_reason": "value_type_compatible_report_argument_overlap",
  "compatible_overlap_used": true,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 1,
  "removed_values": [
    "Q486902"
  ],
  "selected_violation_name": "Value type Q|386724, Q|1151067, Q|151885, Q|11862829, Q|211364, Q|1293220, Q|17737, Q|1799072, Q|14208553, Q|246672, Q|1298668, Q|170584, Q|36534, Q|12772819, Q|65943, Q|2424752, Q|43229, Q|2668072, Q|8261, Q|618779, Q|9081, Q|1711593, Q|10541491, Q|7406919, Q|820655, Q|2095, Q|1914636, Q|1656682, Q|16510064, Q|22811462, Q|185925, Q|853614, Q|47461344, Q|27058, Q|193946, Q|12280, Q|124301146, Q|3658341, Q|131841, Q|41176, Q|811979, Q|483394, Q|106822736, Q|1731969... [truncated 90 chars]",
  "semantic_added_values": [
    "Q246672"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_values": [
    "Q486902"
  ],
  "set_operation": "mixed",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "value-type constraint",
  "target_constraint_qid": "Q21510865",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "AVDLCZ",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "8fa55dea5c851c51af5c1fcf4f0d1bfa28d99b71",
  "hash_before": "8738bc50e1b24f536e9f5b63f53b789741ea3148",
  "property_revision_id": 2439776691,
  "property_revision_prev": 2436002170,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_and_value_match",
    "mapped_violation_constraint_qid": "Q21510865",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Value type Q|386724, Q|1151067, Q|151885, Q|11862829, Q|211364, Q|1293220, Q|17737, Q|1799072, Q|14208553, Q|246672, Q|1298668, Q|170584, Q|36534, Q|12772819, Q|65943, Q|2424752, Q|43229, Q|2668072, Q|8261, Q|618779, Q|9081, Q|1711593, Q|10541491, Q|7406919, Q|820655, Q|2095, Q|1914636, Q|1656682, Q|16510064, Q|22811462, Q|185925, Q|853614, Q|47461344, Q|27058, Q|193946, Q|12280, Q|124301146, Q|3658341, Q|131841, Q|41176, Q|811979, Q|483394, Q|106822736, Q|1731969... [truncated 90 chars]"
  },
  {
    "result": "Q21510865",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 1,
    "result": "SCHEMA_UPDATE",
    "set_operation": "mixed",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q246672"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21510865",
        "mapped_violation_family": "value_type",
        "violation_name": "Value type Q|386724, Q|1151067, Q|151885, Q|11862829, Q|211364, Q|1293220, Q|17737, Q|1799072, Q|14208553, Q|246672, Q|1298668, Q|170584, Q|36534, Q|12772819, Q|65943, Q|2424752, Q|43229, Q|2668072, Q|8261, Q|618779, Q|9081, Q|1711593, Q|10541491, Q|7406919, Q|820655, Q|2095, Q|1914636, Q|1656682, Q|16510064, Q|22811462, Q|185925, Q|853614, Q|47461344, Q|27058, Q|193946, Q|12280, Q|124301146, Q|3658341, Q|131841, Q|41176, Q|811979, Q|483394, Q|106822736, Q|1731969... [truncated 90 chars]"
      }
    ],
    "candidate_violation_names": [
      "Value type Q|386724, Q|1151067, Q|151885, Q|11862829, Q|211364, Q|1293220, Q|17737, Q|1799072, Q|14208553, Q|246672, Q|1298668, Q|170584, Q|36534, Q|12772819, Q|65943, Q|2424752, Q|43229, Q|2668072, Q|8261, Q|618779, Q|9081, Q|1711593, Q|10541491, Q|7406919, Q|820655, Q|2095, Q|1914636, Q|1656682, Q|16510064, Q|22811462, Q|185925, Q|853614, Q|47461344, Q|27058, Q|193946, Q|12280, Q|124301146, Q|3658341, Q|131841, Q|41176, Q|811979, Q|483394, Q|106822736, Q|1731969... [truncated 90 chars]"
    ],
    "causality_match_level": "exact_constraint_and_value_match",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510865"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510865"
    ],
    "changed_qualifier_properties": [
      "P2308"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "value_type_compatible_report_argument_overlap",
    "compatible_overlap_used": true,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [
      "Q246672"
    ],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "value-type constraint",
    "mapped_report_constraint_qid": "Q21510865",
    "mapped_report_family": "value_type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "value-type constraint",
    "mapped_violation_constraint_qid": "Q21510865",
    "mapped_violation_family": "value_type",
    "mapped_violation_reason": "value_type_prefix_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 1,
    "removed_values": [
      "Q486902"
    ],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Value type Q|386724, Q|1151067, Q|151885, Q|11862829, Q|211364, Q|1293220, Q|17737, Q|1799072, Q|14208553, Q|246672, Q|1298668, Q|170584, Q|36534, Q|12772819, Q|65943, Q|2424752, Q|43229, Q|2668072, Q|8261, Q|618779, Q|9081, Q|1711593, Q|10541491, Q|7406919, Q|820655, Q|2095, Q|1914636, Q|1656682, Q|16510064, Q|22811462, Q|185925, Q|853614, Q|47461344, Q|27058, Q|193946, Q|12280, Q|124301146, Q|3658341, Q|131841, Q|41176, Q|811979, Q|483394, Q|106822736, Q|1731969... [truncated 90 chars]",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q246672"
    ],
    "semantic_changed_qualifier_properties": [
      "P2308"
    ],
    "semantic_removed_value_count": 1,
    "semantic_removed_values": [
      "Q486902"
    ],
    "set_operation": "mixed",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "value-type constraint",
    "target_constraint_qid": "Q21510865",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "value_type",
    "value_overlap_with_report_qids": [
      "Q246672"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Value type Q|386724, Q|1151067, Q|151885, Q|11862829, Q|211364, Q|1293220, Q|17737, Q|1799072, Q|14208553, Q|246672, Q|1298668, Q|170584, Q|36534, Q|12772819, Q|65943, Q|2424752, Q|43229, Q|2668072, Q|8261, Q|618779, Q|9081, Q|1711593, Q|10541491, Q|7406919, Q|820655, Q|2095, Q|1914636, Q|1656682, Q|16510064, Q|22811462, Q|185925, Q|853614, Q|47461344, Q|27058, Q|193946, Q|12280, Q|124301146, Q|3658341, Q|131841, Q|41176, Q|811979, Q|483394, Q|106822736, Q|1731969... [truncated 90 chars]"
  }
]
```

---

## 014. `reform_Q17479235_P12347_2444981085`

| Field | Value |
|---|---|
| qid | Q17479235 |
| property | P12347 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21502404 format constraint |
| group_key | TBOX::P12347::2444981085 |
| tbox_revision_key | TBOX::P12347::2444981085 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "author": "Ontogon",
  "kind": "T_BOX",
  "property_revision_id": 2444981085,
  "property_revision_prev": 2427580269
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-22T04:07:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P12347",
  "report_revision_new": 2445329991,
  "report_revision_old": 2444769289,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": null,
  "value_current_2026": [
    "du-gehörst-mir"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a movie at NientePopCorn.it",
    "label": "NientePopCorn movie ID"
  },
  "qid": {
    "description": "1959 film",
    "label": "Your Body Belongs to Me"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "label in language constraint",
    "qid": "Q108139345"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21502404",
      "mapped_violation_family": "format",
      "violation_name": "Format"
    }
  ],
  "candidate_violation_names": [
    "Format"
  ],
  "causality_match_level": "exact_constraint_and_value_match",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21502404"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502404"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "format_regex_qualifier_changed",
  "compatible_overlap_used": true,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "format constraint",
  "mapped_report_constraint_qid": "Q21502404",
  "mapped_report_family": "format",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "format constraint",
  "mapped_violation_constraint_qid": "Q21502404",
  "mapped_violation_family": "format",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "unknown",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Format",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "[a-zäöüéèø0-9\\-]*"
  ],
  "semantic_changed_qualifier_properties": [
    "P1793"
  ],
  "semantic_removed_value_count": 1,
  "semantic_removed_values": [
    "[a-zäüéèø0-9\\-]*"
  ],
  "set_operation": "mixed",
  "set_semantics": "unknown",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "format constraint",
  "target_constraint_qid": "Q21502404",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "[a-zäöüéèø0-9\\-]*"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21502404"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502404"
  ],
  "changed_qualifier_properties": [
    "P1793"
  ],
  "compatible_overlap_reason": "format_regex_qualifier_changed",
  "compatible_overlap_used": true,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "format constraint",
  "mapped_report_constraint_qid": "Q21502404",
  "mapped_report_family": "format",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "unknown",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 1,
  "removed_values": [
    "[a-zäüéèø0-9\\-]*"
  ],
  "selected_violation_name": "Format",
  "semantic_added_values": [
    "[a-zäöüéèø0-9\\-]*"
  ],
  "semantic_changed_qualifier_properties": [
    "P1793"
  ],
  "semantic_removed_values": [
    "[a-zäüéèø0-9\\-]*"
  ],
  "set_operation": "mixed",
  "set_semantics": "unknown",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "format constraint",
  "target_constraint_qid": "Q21502404",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Ontogon",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "6606f5f69217055e56dbf3d7c7ea1f0ad077cdb9",
  "hash_before": "ec39d9bce56b76c9848e8ac7cc4de4c93dd1bde5",
  "property_revision_id": 2444981085,
  "property_revision_prev": 2427580269,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_and_value_match",
    "mapped_violation_constraint_qid": "Q21502404",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Format"
  },
  {
    "result": "Q21502404",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "unknown",
    "property_ids": [
      "P1793"
    ],
    "removed_value_count": 1,
    "result": "SCHEMA_UPDATE",
    "set_operation": "mixed",
    "set_semantics": "unknown",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "[a-zäöüéèø0-9\\-]*"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21502404",
        "mapped_violation_family": "format",
        "violation_name": "Format"
      }
    ],
    "candidate_violation_names": [
      "Format"
    ],
    "causality_match_level": "exact_constraint_and_value_match",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21502404"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21502404"
    ],
    "changed_qualifier_properties": [
      "P1793"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "format_regex_qualifier_changed",
    "compatible_overlap_used": true,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "format constraint",
    "mapped_report_constraint_qid": "Q21502404",
    "mapped_report_family": "format",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "format constraint",
    "mapped_violation_constraint_qid": "Q21502404",
    "mapped_violation_family": "format",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "unknown",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P1793"
    ],
    "removed_value_count": 1,
    "removed_values": [
      "[a-zäüéèø0-9\\-]*"
    ],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Format",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "[a-zäöüéèø0-9\\-]*"
    ],
    "semantic_changed_qualifier_properties": [
      "P1793"
    ],
    "semantic_removed_value_count": 1,
    "semantic_removed_values": [
      "[a-zäüéèø0-9\\-]*"
    ],
    "set_operation": "mixed",
    "set_semantics": "unknown",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "format constraint",
    "target_constraint_qid": "Q21502404",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "format",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Format"
  }
]
```

---

## 015. `reform_Q2115968_P921_2447527451`

| Field | Value |
|---|---|
| qid | Q2115968 |
| property | P921 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P921::2447527451 |
| tbox_revision_key | TBOX::P921::2447527451 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "YotaMoteuchi",
  "kind": "T_BOX",
  "property_revision_id": 2447527451,
  "property_revision_prev": 2438998167
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T11:28:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
  "report_revision_new": 2447751779,
  "report_revision_old": 2447354743,
  "report_violation_type": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "any set of human beings",
    "reference in one place in a book to information at another place in the same work",
    "category of creative works based on stylistic, thematic or technical criteria",
    "temporary and scheduled happening, like a conference, festival, competition or similar",
    "program of study, or unit of teaching that typically lasts one academic term",
    "intangible asset consisting of ownership of ideas and processes",
    "theme or subject in a work of art",
    "a restaurant based around a concept or intellectual property",
    "non-repayable funds disbursed by one party to a recipient",
    "human subject research in medicine",
    "process that attempts to determine the facts of a crime and circumstances",
    "single content rating in a rating system",
    "transgression or alleged transgression resulting in public outrage",
    "disclosure of confidential or nonpublic information to unauthorized parties",
    "collection of materials with some unifying characteristic, housed in an archive",
    "set of purposely gathered physical or digital objects with some common characteristics",
    "experience of intense sexual arousal to atypical objects, situations, or individuals",
    "section of learning or teaching into which a wider learning content is divided",
    "process that has the aim of augmenting knowledge, resolving doubt, or solving a problem",
    "word or an unspaced phrase prefixed with the number sign, used to categorise a topic",
    "project of one or more scientists, or of an organization in a scientific field",
    "scientific procedure carried out to support, refute, or validate a hypothesis",
    "topic viewed from a historical point of view",
    "... omitted 62 items"
  ],
  "report_violation_type_labels_en": [
    "work",
    "group of humans",
    "cross-reference",
    "genre",
    "event",
    "course",
    "intellectual property",
    "artistic theme",
    "theme restaurant",
    "grant",
    "clinical trial",
    "criminal investigation",
    "content rating category",
    "scandal",
    "information leak",
    "archival collection",
    "collection",
    "paraphilia",
    "lesson",
    "inquiry",
    "#",
    "science project",
    "experiment",
    "aspect of history",
    "... omitted 62 items"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
  "report_violation_type_qids": [
    "Q386724",
    "Q16334295",
    "Q1302249",
    "Q483394",
    "Q1656682",
    "Q600134",
    "Q131257",
    "Q1406161",
    "Q676586",
    "Q230788",
    "Q30612",
    "Q1964968",
    "Q23649976",
    "Q192909",
    "Q2904148",
    "Q9388534",
    "Q2668072",
    "Q178059",
    "Q379833",
    "Q21004260",
    "Q278485",
    "Q1298668",
    "Q101965",
    "Q17524420",
    "... omitted 62 items"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
  "value": null,
  "value_current_2026": [
    "Q9418"
  ],
  "value_current_2026_descriptions_en": [
    "study of mental functions and behaviours"
  ],
  "value_current_2026_labels_en": [
    "psychology"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "primary topic of a work or act of communication",
    "label": "main subject"
  },
  "qid": {
    "description": "educación que permite obtener el título de psicólogo",
    "label": "studies of psychology"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]"
    }
  ],
  "candidate_violation_names": [
    "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q3249551"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "type constraint",
  "target_constraint_qid": "Q21503250",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q3249551"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": [
    "P2308"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
  "semantic_added_values": [
    "Q3249551"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "type constraint",
  "target_constraint_qid": "Q21503250",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": true
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "YotaMoteuchi",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "d5ea1158d61a890f5b9b5973dbea33604cae9572",
  "hash_before": "27f65b7446fdc00965554e75fb3f409357651989",
  "property_revision_id": 2447527451,
  "property_revision_prev": 2438998167,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "mapped_violation_constraint_qid": "Q21503250",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q3249551"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]"
      }
    ],
    "candidate_violation_names": [
      "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503250"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503250"
    ],
    "changed_qualifier_properties": [
      "P2308"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "type constraint",
    "mapped_report_constraint_qid": "Q21503250",
    "mapped_report_family": "type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "type constraint",
    "mapped_violation_constraint_qid": "Q21503250",
    "mapped_violation_family": "type",
    "mapped_violation_reason": "type_prefix_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q3249551"
    ],
    "semantic_changed_qualifier_properties": [
      "P2308"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "type constraint",
    "target_constraint_qid": "Q21503250",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "type",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]"
  }
]
```

---

## 016. `reform_Q27696267_P640_2317152167`

| Field | Value |
|---|---|
| qid | Q27696267 |
| property | P640 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q108139345 label in language constraint |
| group_key | TBOX::P640::2317152167 |
| tbox_revision_key | TBOX::P640::2317152167 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "label in language constraint",
  "decision_constraint_type_qid": "Q108139345"
}
```

#### Repair Target

```json
{
  "author": "Bob08",
  "kind": "T_BOX",
  "property_revision_id": 2317152167,
  "property_revision_prev": 2313446717
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-02-26T12:30:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P640",
  "report_revision_new": 2317274491,
  "report_revision_old": 2316736418,
  "report_violation_type": "Item P|19",
  "report_violation_type_normalized": "Item P|19",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|19",
  "report_violation_types": [
    "Item P|19",
    "Item P|27",
    "Item P|166 one of Q|10855195, Q|10855212, Q|10855216, Q|10855226, Q|10855271",
    "Label in fr language"
  ],
  "value": null,
  "value_current_2026": [
    "19800035/558/63637"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of a person in the Léonore database",
    "label": "Léonore ID"
  },
  "qid": {
    "description": null,
    "label": "Léonce Charles Moineville"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "label in language constraint",
    "qid": "Q108139345"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q108139345",
      "mapped_violation_family": "label_in_language",
      "violation_name": "Label in fr language"
    },
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|19"
    },
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|27"
    },
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|166 one of Q|10855195, Q|10855212, Q|10855216, Q|10855226, Q|10855271"
    }
  ],
  "candidate_violation_names": [
    "Item P|19",
    "Item P|27",
    "Item P|166 one of Q|10855195, Q|10855212, Q|10855216, Q|10855226, Q|10855271",
    "Label in fr language"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q108139345"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q108139345"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "label-in-language constraint",
  "mapped_report_constraint_qid": "Q108139345",
  "mapped_report_family": "label_in_language",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "label-in-language constraint",
  "mapped_violation_constraint_qid": "Q108139345",
  "mapped_violation_family": "label_in_language",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Label in fr language",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "mul"
  ],
  "semantic_changed_qualifier_properties": [
    "P424"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "label-in-language constraint",
  "target_constraint_qid": "Q108139345",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "mul"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q108139345"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q108139345"
  ],
  "changed_qualifier_properties": [
    "P424"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "label-in-language constraint",
  "mapped_report_constraint_qid": "Q108139345",
  "mapped_report_family": "label_in_language",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Label in fr language",
  "semantic_added_values": [
    "mul"
  ],
  "semantic_changed_qualifier_properties": [
    "P424"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "label-in-language constraint",
  "target_constraint_qid": "Q108139345",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": true
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Bob08",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "b3fce829a0d59e3f6b3d66ff8be02ec76d0eb344",
  "hash_before": "b59dc677f5719e4088c2bfd0b02d669c53b06638",
  "property_revision_id": 2317152167,
  "property_revision_prev": 2313446717,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "mapped_violation_constraint_qid": "Q108139345",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Label in fr language"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P424"
    ],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "mul"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q108139345",
        "mapped_violation_family": "label_in_language",
        "violation_name": "Label in fr language"
      },
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|19"
      },
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|27"
      },
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|166 one of Q|10855195, Q|10855212, Q|10855216, Q|10855226, Q|10855271"
      }
    ],
    "candidate_violation_names": [
      "Item P|19",
      "Item P|27",
      "Item P|166 one of Q|10855195, Q|10855212, Q|10855216, Q|10855226, Q|10855271",
      "Label in fr language"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q108139345"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q108139345"
    ],
    "changed_qualifier_properties": [
      "P424"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "label-in-language constraint",
    "mapped_report_constraint_qid": "Q108139345",
    "mapped_report_family": "label_in_language",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "label-in-language constraint",
    "mapped_violation_constraint_qid": "Q108139345",
    "mapped_violation_family": "label_in_language",
    "mapped_violation_reason": "label_language_report_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P424"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Label in fr language",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "mul"
    ],
    "semantic_changed_qualifier_properties": [
      "P424"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "label-in-language constraint",
    "target_constraint_qid": "Q108139345",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "label_in_language",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Label in fr language"
  }
]
```

---

## 017. `reform_Q3137802_P269_2445523281`

| Field | Value |
|---|---|
| qid | Q3137802 |
| property | P269 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21502410 distinct-values constraint |
| group_key | TBOX::P269::2445523281 |
| tbox_revision_key | TBOX::P269::2445523281 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "distinct-values constraint",
  "decision_constraint_type_qid": "Q21502410"
}
```

#### Repair Target

```json
{
  "author": "Thomas Kerboul (BGE)",
  "kind": "T_BOX",
  "property_revision_id": 2445523281,
  "property_revision_prev": 2445451375
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-24T12:51:30",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P269",
  "report_revision_new": 2446541658,
  "report_revision_old": 2446069538,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "027401448",
    "256094608"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for authority control in the French collaborative library catalog (see also P1025). Format: 8 digits followed by a digit or \"X\"",
    "label": "IdRef ID"
  },
  "qid": {
    "description": "state in western Europe (1034–1848)",
    "label": "Principality of Neuchâtel"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21502410",
      "mapped_violation_family": "distinct_values",
      "violation_name": "Unique value"
    }
  ],
  "candidate_violation_names": [
    "Unique value"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21502410"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502410"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [
    "P2303"
  ],
  "ignored_removed_values": [
    "Q12738",
    "Q159",
    "Q2184",
    "Q2370801",
    "Q3137802",
    "Q34266",
    "Q4345832"
  ],
  "ignored_value_count": 7,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "distinct-values constraint",
  "mapped_report_constraint_qid": "Q21502410",
  "mapped_report_family": "distinct_values",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "distinct-values constraint",
  "mapped_violation_constraint_qid": "Q21502410",
  "mapped_violation_family": "distinct_values",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Unique value",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "distinct-values constraint",
  "target_constraint_qid": "Q21502410",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 0,
  "added_values": [],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21502410"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502410"
  ],
  "changed_qualifier_properties": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [
    "P2303"
  ],
  "ignored_removed_values": [
    "Q12738",
    "Q159",
    "Q2184",
    "Q2370801",
    "Q3137802",
    "Q34266",
    "Q4345832"
  ],
  "ignored_value_count": 7,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "distinct-values constraint",
  "mapped_report_constraint_qid": "Q21502410",
  "mapped_report_family": "distinct_values",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Unique value",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "distinct-values constraint",
  "target_constraint_qid": "Q21502410",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Thomas Kerboul (BGE)",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "885bab3ddcf610be0c045df2830bd44b8d0037de",
  "hash_before": "7001e5b91ee685cbf6467bb34973749c2e5258df",
  "property_revision_id": 2445523281,
  "property_revision_prev": 2445451375,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only",
    "mapped_violation_constraint_qid": "Q21502410",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Unique value"
  },
  {
    "result": "Q21502410",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 0,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_ids": [],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 0,
    "added_values": [],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21502410",
        "mapped_violation_family": "distinct_values",
        "violation_name": "Unique value"
      }
    ],
    "candidate_violation_names": [
      "Unique value"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21502410"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21502410"
    ],
    "changed_qualifier_properties": [],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [
      "P2303"
    ],
    "ignored_removed_values": [
      "Q12738",
      "Q159",
      "Q2184",
      "Q2370801",
      "Q3137802",
      "Q34266",
      "Q4345832"
    ],
    "ignored_value_count": 7,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "distinct-values constraint",
    "mapped_report_constraint_qid": "Q21502410",
    "mapped_report_family": "distinct_values",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "distinct-values constraint",
    "mapped_violation_constraint_qid": "Q21502410",
    "mapped_violation_family": "distinct_values",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Unique value",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "distinct-values constraint",
    "target_constraint_qid": "Q21502410",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "distinct_values",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Unique value"
  }
]
```

---

## 018. `reform_Q347201_P735_2442352845`

| Field | Value |
|---|---|
| qid | Q347201 |
| property | P735 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q53869507 property scope constraint |
| group_key | TBOX::P735::2442352845 |
| tbox_revision_key | TBOX::P735::2442352845 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "property scope constraint",
  "decision_constraint_type_qid": "Q53869507"
}
```

#### Repair Target

```json
{
  "author": "Swpb",
  "kind": "T_BOX",
  "property_revision_id": 2442352845,
  "property_revision_prev": 2427927742
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-15T10:39:57",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P735",
  "report_revision_new": 2442615698,
  "report_revision_old": 2442267914,
  "report_violation_type": "Scope",
  "report_violation_type_normalized": "Scope",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Scope",
  "value": null,
  "value_current_2026": [
    "Q4927589",
    "Q19688695"
  ],
  "value_current_2026_descriptions_en": [
    "male given name",
    "male given name"
  ],
  "value_current_2026_labels_en": [
    "Marcel",
    "Marcellin"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "first name or another given name of this person; values used with the property should not link disambiguations nor family names",
    "label": "given name"
  },
  "qid": {
    "description": "French boxer (1916-1949)",
    "label": "Marcel Cerdan"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q53869507",
      "mapped_violation_family": "property_scope",
      "violation_name": "Scope"
    }
  ],
  "candidate_violation_names": [
    "Scope"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q53869507"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q54828449"
  ],
  "ignored_changed_qualifier_properties": [
    "P5314"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "property scope constraint",
  "mapped_report_constraint_qid": "Q53869507",
  "mapped_report_family": "property_scope",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "property scope constraint",
  "mapped_violation_constraint_qid": "Q53869507",
  "mapped_violation_family": "property_scope",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Scope",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "property scope constraint",
  "target_constraint_qid": "Q53869507",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 0,
  "added_values": [],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q53869507"
  ],
  "changed_qualifier_properties": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q54828449"
  ],
  "ignored_changed_qualifier_properties": [
    "P5314"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "property scope constraint",
  "mapped_report_constraint_qid": "Q53869507",
  "mapped_report_family": "property_scope",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Scope",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "property scope constraint",
  "target_constraint_qid": "Q53869507",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Swpb",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "0b94ae84accb701e29d60b04208ef59e98f5d398",
  "hash_before": "48220aad70210b2f09dd68709f0cc66601d47c2b",
  "property_revision_id": 2442352845,
  "property_revision_prev": 2427927742,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only",
    "mapped_violation_constraint_qid": "Q53869507",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Scope"
  },
  {
    "result": "Q53869507",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 0,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_ids": [],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 0,
    "added_values": [],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q53869507",
        "mapped_violation_family": "property_scope",
        "violation_name": "Scope"
      }
    ],
    "candidate_violation_names": [
      "Scope"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q53869507"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q53869507"
    ],
    "changed_qualifier_properties": [],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [
      "Q54828449"
    ],
    "ignored_changed_qualifier_properties": [
      "P5314"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "property scope constraint",
    "mapped_report_constraint_qid": "Q53869507",
    "mapped_report_family": "property_scope",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "property scope constraint",
    "mapped_violation_constraint_qid": "Q53869507",
    "mapped_violation_family": "property_scope",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P4680"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Scope",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "property scope constraint",
    "target_constraint_qid": "Q53869507",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "property_scope",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Scope"
  }
]
```

---

## 019. `reform_Q3520317_P400_2355127724`

| Field | Value |
|---|---|
| qid | Q3520317 |
| property | P400 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q21510865 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q52558054 none-of constraint |
| group_key | TBOX::P400::2355127724 |
| tbox_revision_key | TBOX::P400::2355127724 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "none-of constraint",
  "decision_constraint_type_qid": "Q52558054"
}
```

#### Repair Target

```json
{
  "author": "Trade",
  "kind": "T_BOX",
  "property_revision_id": 2355127724,
  "property_revision_prev": 2355127546
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-06-02T11:38:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P400",
  "report_revision_new": 2356023718,
  "report_revision_old": 2355590472,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": null,
  "value_current_2026": [
    "Q48263",
    "Q10683",
    "Q1406"
  ],
  "value_current_2026_descriptions_en": [
    "Microsoft's seventh-generation and second home video game console",
    "video game console developed Sony Interactive Entertainment",
    "family of computer operating systems developed by Microsoft"
  ],
  "value_current_2026_labels_en": [
    "எக்ஸ் பாக்ஸ் 360",
    "PlayStation 3",
    "Microsoft Windows"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "platform for which a work was developed or released, or the specific platform version of a software product",
    "label": "platform"
  },
  "qid": {
    "description": "2008 video game",
    "label": "The Club"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q52558054",
      "mapped_violation_family": "none_of",
      "violation_name": "None of"
    }
  ],
  "candidate_violation_names": [
    "None of"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q52558054"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q52558054"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q863516"
  ],
  "ignored_changed_qualifier_properties": [
    "P9729"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "none-of constraint",
  "mapped_report_constraint_qid": "Q52558054",
  "mapped_report_family": "none_of",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "none-of constraint",
  "mapped_violation_constraint_qid": "Q52558054",
  "mapped_violation_family": "none_of",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "forbidden",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "None of",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "forbidden",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "none-of constraint",
  "target_constraint_qid": "Q52558054",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 0,
  "added_values": [],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q52558054"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q52558054"
  ],
  "changed_qualifier_properties": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q863516"
  ],
  "ignored_changed_qualifier_properties": [
    "P9729"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "none-of constraint",
  "mapped_report_constraint_qid": "Q52558054",
  "mapped_report_family": "none_of",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "forbidden",
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "None of",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "forbidden",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "none-of constraint",
  "target_constraint_qid": "Q52558054",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Trade",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "fd8998e99a76e16dd5985b5dae7f07e6b892dde1",
  "hash_before": "2a0fcd019beedf9ea03197f3bcf9821f5dacfda3",
  "property_revision_id": 2355127724,
  "property_revision_prev": 2355127546,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only",
    "mapped_violation_constraint_qid": "Q52558054",
    "result": true,
    "step": "causality_filter",
    "violation_name": "None of"
  },
  {
    "result": "Q52558054",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 0,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "forbidden",
    "property_ids": [],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "unchanged",
    "set_semantics": "forbidden",
    "step": "set_semantics"
  },
  {
    "added_value_count": 0,
    "added_values": [],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q52558054",
        "mapped_violation_family": "none_of",
        "violation_name": "None of"
      }
    ],
    "candidate_violation_names": [
      "None of"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q52558054"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q52558054"
    ],
    "changed_qualifier_properties": [],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [
      "Q863516"
    ],
    "ignored_changed_qualifier_properties": [
      "P9729"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "none-of constraint",
    "mapped_report_constraint_qid": "Q52558054",
    "mapped_report_family": "none_of",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "none-of constraint",
    "mapped_violation_constraint_qid": "Q52558054",
    "mapped_violation_family": "none_of",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "forbidden",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "None of",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "unchanged",
    "set_semantics": "forbidden",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "none-of constraint",
    "target_constraint_qid": "Q52558054",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "none_of",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "None of"
  }
]
```

---

## 020. `reform_Q435328_P421_2440930250`

| Field | Value |
|---|---|
| qid | Q435328 |
| property | P421 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21502838 conflicts-with constraint |
| group_key | TBOX::P421::2440930250 |
| tbox_revision_key | TBOX::P421::2440930250 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "conflicts-with constraint",
  "decision_constraint_type_qid": "Q21502838"
}
```

#### Repair Target

```json
{
  "author": "Herzi Pinki",
  "kind": "T_BOX",
  "property_revision_id": 2440930250,
  "property_revision_prev": 2431249521
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-12T10:56:23",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P421",
  "report_revision_new": 2441204479,
  "report_revision_old": 2440851512,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": null,
  "value_current_2026": [
    "Q843589"
  ],
  "value_current_2026_descriptions_en": [
    "time zone"
  ],
  "value_current_2026_labels_en": [
    "Western European Time"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "time zone for this item",
    "label": "located in time zone"
  },
  "qid": {
    "description": "civil parish in Loulé",
    "label": "Alte"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21502838",
      "mapped_violation_family": "conflicts_with",
      "violation_name": "Conflicts with P|31"
    }
  ],
  "candidate_violation_names": [
    "Conflicts with P|31"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21502838"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502838"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "conflicts-with constraint",
  "mapped_report_constraint_qid": "Q21502838",
  "mapped_report_family": "conflicts_with",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "conflicts-with constraint",
  "mapped_violation_constraint_qid": "Q21502838",
  "mapped_violation_family": "conflicts_with",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "forbidden set expansion",
  "potential_directional_subtype_precise": "RESTRICTION_FORBIDDEN_SET_EXPANSION",
  "potential_polarity": "restriction",
  "potential_polarity_basis": "forbidden set gained prohibited values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "forbidden",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Conflicts with P|31",
  "semantic_added_value_count": 4,
  "semantic_added_values": [
    "Q11183787",
    "Q17376095",
    "Q20871353",
    "Q253326"
  ],
  "semantic_changed_qualifier_properties": [
    "P2305"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "forbidden",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "conflicts-with constraint",
  "target_constraint_qid": "Q21502838",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 4,
  "added_values": [
    "Q11183787",
    "Q17376095",
    "Q20871353",
    "Q253326"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21502838"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502838"
  ],
  "changed_qualifier_properties": [
    "P2305"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "conflicts-with constraint",
  "mapped_report_constraint_qid": "Q21502838",
  "mapped_report_family": "conflicts_with",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "forbidden set expansion",
  "potential_directional_subtype_precise": "RESTRICTION_FORBIDDEN_SET_EXPANSION",
  "potential_polarity": "restriction",
  "potential_polarity_basis": "forbidden set gained prohibited values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "forbidden",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Conflicts with P|31",
  "semantic_added_values": [
    "Q11183787",
    "Q17376095",
    "Q20871353",
    "Q253326"
  ],
  "semantic_changed_qualifier_properties": [
    "P2305"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "forbidden",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "conflicts-with constraint",
  "target_constraint_qid": "Q21502838",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": true
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Herzi Pinki",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "59a57149d7aa11ef4a244d7f3f250a78c26cbb62",
  "hash_before": "1d4d35a0022e0a13a6087df913821789d9e8aa1d",
  "property_revision_id": 2440930250,
  "property_revision_prev": 2431249521,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "mapped_violation_constraint_qid": "Q21502838",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Conflicts with P|31"
  },
  {
    "result": "Q21502838",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 4,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "forbidden set expansion",
    "potential_directional_subtype_precise": "RESTRICTION_FORBIDDEN_SET_EXPANSION",
    "potential_polarity": "restriction",
    "potential_polarity_basis": "forbidden set gained prohibited values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "forbidden",
    "property_ids": [
      "P2305"
    ],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "expansion",
    "set_semantics": "forbidden",
    "step": "set_semantics"
  },
  {
    "added_value_count": 4,
    "added_values": [
      "Q11183787",
      "Q17376095",
      "Q20871353",
      "Q253326"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21502838",
        "mapped_violation_family": "conflicts_with",
        "violation_name": "Conflicts with P|31"
      }
    ],
    "candidate_violation_names": [
      "Conflicts with P|31"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21502838"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21502838"
    ],
    "changed_qualifier_properties": [
      "P2305"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "conflicts-with constraint",
    "mapped_report_constraint_qid": "Q21502838",
    "mapped_report_family": "conflicts_with",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "conflicts-with constraint",
    "mapped_violation_constraint_qid": "Q21502838",
    "mapped_violation_family": "conflicts_with",
    "mapped_violation_reason": "conflicts_with_prefix_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "forbidden set expansion",
    "potential_directional_subtype_precise": "RESTRICTION_FORBIDDEN_SET_EXPANSION",
    "potential_polarity": "restriction",
    "potential_polarity_basis": "forbidden set gained prohibited values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "forbidden",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Conflicts with P|31",
    "semantic_added_value_count": 4,
    "semantic_added_values": [
      "Q11183787",
      "Q17376095",
      "Q20871353",
      "Q253326"
    ],
    "semantic_changed_qualifier_properties": [
      "P2305"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "forbidden",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "conflicts-with constraint",
    "target_constraint_qid": "Q21502838",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "conflicts_with",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Conflicts with P|31"
  }
]
```

---

## 021. `reform_Q449317_P735_2442352845`

| Field | Value |
|---|---|
| qid | Q449317 |
| property | P735 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q53869507 property scope constraint |
| group_key | TBOX::P735::2442352845 |
| tbox_revision_key | TBOX::P735::2442352845 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "property scope constraint",
  "decision_constraint_type_qid": "Q53869507"
}
```

#### Repair Target

```json
{
  "author": "Swpb",
  "kind": "T_BOX",
  "property_revision_id": 2442352845,
  "property_revision_prev": 2427927742
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-15T10:39:57",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P735",
  "report_revision_new": 2442615698,
  "report_revision_old": 2442267914,
  "report_violation_type": "Scope",
  "report_violation_type_normalized": "Scope",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Scope",
  "value": null,
  "value_current_2026": [
    "Q16281024"
  ],
  "value_current_2026_descriptions_en": [
    "female given name"
  ],
  "value_current_2026_labels_en": [
    "Rosie"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "first name or another given name of this person; values used with the property should not link disambiguations nor family names",
    "label": "given name"
  },
  "qid": {
    "description": "American singer-songwriter and comedian",
    "label": "Rosie Thomas"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q53869507",
      "mapped_violation_family": "property_scope",
      "violation_name": "Scope"
    }
  ],
  "candidate_violation_names": [
    "Scope"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q53869507"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q54828449"
  ],
  "ignored_changed_qualifier_properties": [
    "P5314"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "property scope constraint",
  "mapped_report_constraint_qid": "Q53869507",
  "mapped_report_family": "property_scope",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "property scope constraint",
  "mapped_violation_constraint_qid": "Q53869507",
  "mapped_violation_family": "property_scope",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Scope",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "property scope constraint",
  "target_constraint_qid": "Q53869507",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 0,
  "added_values": [],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q53869507"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q53869507"
  ],
  "changed_qualifier_properties": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q54828449"
  ],
  "ignored_changed_qualifier_properties": [
    "P5314"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "property scope constraint",
  "mapped_report_constraint_qid": "Q53869507",
  "mapped_report_family": "property_scope",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Scope",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "property scope constraint",
  "target_constraint_qid": "Q53869507",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Swpb",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "0b94ae84accb701e29d60b04208ef59e98f5d398",
  "hash_before": "48220aad70210b2f09dd68709f0cc66601d47c2b",
  "property_revision_id": 2442352845,
  "property_revision_prev": 2427927742,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only",
    "mapped_violation_constraint_qid": "Q53869507",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Scope"
  },
  {
    "result": "Q53869507",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 0,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_ids": [],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 0,
    "added_values": [],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q53869507",
        "mapped_violation_family": "property_scope",
        "violation_name": "Scope"
      }
    ],
    "candidate_violation_names": [
      "Scope"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q53869507"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q53869507"
    ],
    "changed_qualifier_properties": [],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [
      "Q54828449"
    ],
    "ignored_changed_qualifier_properties": [
      "P5314"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "property scope constraint",
    "mapped_report_constraint_qid": "Q53869507",
    "mapped_report_family": "property_scope",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "property scope constraint",
    "mapped_violation_constraint_qid": "Q53869507",
    "mapped_violation_family": "property_scope",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P4680"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Scope",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "property scope constraint",
    "target_constraint_qid": "Q53869507",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "property_scope",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Scope"
  }
]
```

---

## 022. `reform_Q56250199_P282_1713906839`

| Field | Value |
|---|---|
| qid | Q56250199 |
| property | P282 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q21510865 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | TBOX::P282::1713906839 |
| tbox_revision_key | TBOX::P282::1713906839 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 1713906839,
  "property_revision_prev": 1708678075
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-08-27T17:43:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P282",
  "report_revision_new": 1714619320,
  "report_revision_old": 1714058185,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": null,
  "value_current_2026": [
    "Q8196"
  ],
  "value_current_2026_descriptions_en": [
    "alphabet specifically codified for writing the Arabic language"
  ],
  "value_current_2026_labels_en": [
    "Arabic alphabet"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface",
    "label": "writing system"
  },
  "qid": {
    "description": "female given name",
    "label": "اعتماد"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21510859",
      "mapped_violation_family": "one_of",
      "violation_name": "One of"
    }
  ],
  "candidate_violation_names": [
    "One of"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510859"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510859"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "one-of constraint",
  "mapped_report_constraint_qid": "Q21510859",
  "mapped_report_family": "one_of",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "one-of constraint",
  "mapped_violation_constraint_qid": "Q21510859",
  "mapped_violation_family": "one_of",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "One of",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q1828555"
  ],
  "semantic_changed_qualifier_properties": [
    "P2305"
  ],
  "semantic_removed_value_count": 1,
  "semantic_removed_values": [
    "Q8196"
  ],
  "set_operation": "mixed",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "one-of constraint",
  "target_constraint_qid": "Q21510859",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q1828555"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21510859"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510859"
  ],
  "changed_qualifier_properties": [
    "P2305"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "one-of constraint",
  "mapped_report_constraint_qid": "Q21510859",
  "mapped_report_family": "one_of",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 1,
  "removed_values": [
    "Q8196"
  ],
  "selected_violation_name": "One of",
  "semantic_added_values": [
    "Q1828555"
  ],
  "semantic_changed_qualifier_properties": [
    "P2305"
  ],
  "semantic_removed_values": [
    "Q8196"
  ],
  "set_operation": "mixed",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "one-of constraint",
  "target_constraint_qid": "Q21510859",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "41b0efd0045833d67dfbaa50a5f74ea899707a20",
  "hash_before": "282fa125cd4e285b482bd691b211bd14fdc7f383",
  "property_revision_id": 1713906839,
  "property_revision_prev": 1708678075,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only",
    "mapped_violation_constraint_qid": "Q21510859",
    "result": true,
    "step": "causality_filter",
    "violation_name": "One of"
  },
  {
    "result": "Q21510859",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P2305"
    ],
    "removed_value_count": 1,
    "result": "SCHEMA_UPDATE",
    "set_operation": "mixed",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q1828555"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21510859",
        "mapped_violation_family": "one_of",
        "violation_name": "One of"
      }
    ],
    "candidate_violation_names": [
      "One of"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510859"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510859"
    ],
    "changed_qualifier_properties": [
      "P2305"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "one-of constraint",
    "mapped_report_constraint_qid": "Q21510859",
    "mapped_report_family": "one_of",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "one-of constraint",
    "mapped_violation_constraint_qid": "Q21510859",
    "mapped_violation_family": "one_of",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2305"
    ],
    "removed_value_count": 1,
    "removed_values": [
      "Q8196"
    ],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "One of",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q1828555"
    ],
    "semantic_changed_qualifier_properties": [
      "P2305"
    ],
    "semantic_removed_value_count": 1,
    "semantic_removed_values": [
      "Q8196"
    ],
    "set_operation": "mixed",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "one-of constraint",
    "target_constraint_qid": "Q21510859",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "one_of",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "One of"
  }
]
```

---

## 023. `reform_Q60853265_P12379_2356114887`

| Field | Value |
|---|---|
| qid | Q60853265 |
| property | P12379 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q19474404 single-value constraint |
| group_key | TBOX::P12379::2356114887 |
| tbox_revision_key | TBOX::P12379::2356114887 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

```json
{
  "author": "JhowieNitnek",
  "kind": "T_BOX",
  "property_revision_id": 2356114887,
  "property_revision_prev": 2355961915
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-06-03T04:31:08",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P12379",
  "report_revision_new": 2356346495,
  "report_revision_old": 2355885205,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "report_violation_types": [
    "Single value",
    "Item P|131"
  ],
  "value": null,
  "value_current_2026": [
    "streets/10006078",
    "streets/10501930"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for architectural heritage in the Brussels-Capital Region, Belgium",
    "label": "Brussels Inventory of Architectural Heritage ID"
  },
  "qid": {
    "description": "street in Ixelles, Brussels and Uccle, Belgium",
    "label": "Avenue Legrand - Legrandlaan"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q19474404",
      "mapped_violation_family": "single_value",
      "violation_name": "Single value"
    },
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|131"
    }
  ],
  "candidate_violation_names": [
    "Single value",
    "Item P|131"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q19474404"
  ],
  "changed_constraint_qids_from_entries": [
    "Q19474404"
  ],
  "changed_constraint_qids_from_qualifier_changes": [],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "single-value constraint",
  "mapped_report_constraint_qid": "Q19474404",
  "mapped_report_family": "single_value",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "single-value constraint",
  "mapped_violation_constraint_qid": "Q19474404",
  "mapped_violation_family": "single_value",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Single value",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "single-value constraint",
  "target_constraint_qid": "Q19474404",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 0,
  "added_values": [],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q19474404"
  ],
  "changed_constraint_qids_from_entries": [
    "Q19474404"
  ],
  "changed_constraint_qids_from_qualifier_changes": [],
  "changed_qualifier_properties": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "single-value constraint",
  "mapped_report_constraint_qid": "Q19474404",
  "mapped_report_family": "single_value",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Single value",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "single-value constraint",
  "target_constraint_qid": "Q19474404",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "JhowieNitnek",
  "before_constraint_count": 0,
  "changed_constraint_types": [
    "Q19474404"
  ],
  "constraints_readable_en": null,
  "hash_after": "9d56cc7c7b0c4597f7585d9f174f4e71d9f791d5",
  "hash_before": "746f020dab38b308c8d2cf5d1c4b1f106fc133d2",
  "property_revision_id": 2356114887,
  "property_revision_prev": 2355961915,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only",
    "mapped_violation_constraint_qid": "Q19474404",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Single value"
  },
  {
    "result": "Q19474404",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 0,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_ids": [],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 0,
    "added_values": [],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q19474404",
        "mapped_violation_family": "single_value",
        "violation_name": "Single value"
      },
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|131"
      }
    ],
    "candidate_violation_names": [
      "Single value",
      "Item P|131"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q19474404"
    ],
    "changed_constraint_qids_from_entries": [
      "Q19474404"
    ],
    "changed_constraint_qids_from_qualifier_changes": [],
    "changed_qualifier_properties": [],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "single-value constraint",
    "mapped_report_constraint_qid": "Q19474404",
    "mapped_report_family": "single_value",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "single-value constraint",
    "mapped_violation_constraint_qid": "Q19474404",
    "mapped_violation_family": "single_value",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P4155"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Single value",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "single-value constraint",
    "target_constraint_qid": "Q19474404",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "single_value",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Single value"
  }
]
```

---

## 024. `reform_Q6290537_P108_2437412687`

| Field | Value |
|---|---|
| qid | Q6290537 |
| property | P108 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q21510865 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21510865 value-type constraint |
| group_key | TBOX::P108::2437412687 |
| tbox_revision_key | TBOX::P108::2437412687 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "value-type constraint",
  "decision_constraint_type_qid": "Q21510865"
}
```

#### Repair Target

```json
{
  "author": "Andre Engels",
  "kind": "T_BOX",
  "property_revision_id": 2437412687,
  "property_revision_prev": 2433444260
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-09T13:55:48",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P108",
  "report_revision_new": 2440044970,
  "report_revision_old": 2439577929,
  "report_violation_type": "Value type Q|43229, Q|14623646, Q|16334295, Q|5, Q|95074, Q|327333, Q|1248784, Q|1002697, Q|102496, Q|13002315, Q|5446565, Q|35127, Q|783794, Q|10648343, Q|15632617, Q|11812394, Q|2659904, Q|15416, Q|16560, Q|24634210, Q|15911314, Q|2001305, Q|3152824, Q|56061, Q|11032, Q|1555508, Q|245016, Q|820477, Q|2385804, Q|13235160, Q|13226383, Q|6056746, Q|1474440, Q|22988604, Q|21070568, Q|167037, Q|2085381, Q|431289, Q|15265344",
  "report_violation_type_descriptions_en": [
    "social entity established to meet needs or pursue goals",
    "organization which only appears in works of fiction",
    "any set of human beings",
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "fictional human or non-human character in a narrative work of art",
    "organisational part of a government responsible for specific public services, such as health, judiciary, education, transportation, foreign affairs, etc",
    "location where aircraft take off and land with extended support facilities, mostly for commercial air transport",
    "publication type, serial publication that appears in a new edition on a regular schedule",
    "type of ecclesiastical subdivision of a diocese",
    "named person or animal that appears in legends that have some claim to be historical",
    "business organization which only exists in fiction",
    "set of related web pages served from a single web domain",
    "legal entity representing an association of people, whether natural, legal or a mixture of both, with a specific objective",
    "two individuals who work together",
    "human being that only exists in fictional works",
    "organization that produces theatrical performances",
    "organization administrated by a government authority or agency",
    "segment of audiovisual content intended for broadcast and streaming on television",
    "grand residence, especially a royal or episcopal residence",
    "episode-based program (audio or video) distributed asynchronously on the Internet, typically via an RSS feed or downloadable files",
    "group of individuals (natural persons) or entities (legal persons) of all kinds",
    "terrestrial frequency or virtual number over which a television station or television network is distributed",
    "organization that works for the preservation or promotion of culture",
    "territorial entity for administration purposes, with or without its own local government",
    "... omitted 15 items"
  ],
  "report_violation_type_labels_en": [
    "organization",
    "fictional organization",
    "group of humans",
    "human",
    "character",
    "government agency",
    "airport",
    "periodical",
    "parish",
    "legendary figure",
    "fictional company",
    "website",
    "company",
    "duo",
    "fictional human",
    "theatre company",
    "government organization",
    "television program",
    "palace",
    "podcast show",
    "association",
    "television channel",
    "cultural institution",
    "administrative territorial entity",
    "... omitted 15 items"
  ],
  "report_violation_type_normalized": "Value type Q|43229, Q|14623646, Q|16334295, Q|5, Q|95074, Q|327333, Q|1248784, Q|1002697, Q|102496, Q|13002315, Q|5446565, Q|35127, Q|783794, Q|10648343, Q|15632617, Q|11812394, Q|2659904, Q|15416, Q|16560, Q|24634210, Q|15911314, Q|2001305, Q|3152824, Q|56061, Q|11032, Q|1555508, Q|245016, Q|820477, Q|2385804, Q|13235160, Q|13226383, Q|6056746, Q|1474440, Q|22988604, Q|21070568, Q|167037, Q|2085381, Q|431289, Q|15265344",
  "report_violation_type_qids": [
    "Q43229",
    "Q14623646",
    "Q16334295",
    "Q5",
    "Q95074",
    "Q327333",
    "Q1248784",
    "Q1002697",
    "Q102496",
    "Q13002315",
    "Q5446565",
    "Q35127",
    "Q783794",
    "Q10648343",
    "Q15632617",
    "Q11812394",
    "Q2659904",
    "Q15416",
    "Q16560",
    "Q24634210",
    "Q15911314",
    "Q2001305",
    "Q3152824",
    "Q56061",
    "... omitted 15 items"
  ],
  "report_violation_type_raw": "Value type Q|43229, Q|14623646, Q|16334295, Q|5, Q|95074, Q|327333, Q|1248784, Q|1002697, Q|102496, Q|13002315, Q|5446565, Q|35127, Q|783794, Q|10648343, Q|15632617, Q|11812394, Q|2659904, Q|15416, Q|16560, Q|24634210, Q|15911314, Q|2001305, Q|3152824, Q|56061, Q|11032, Q|1555508, Q|245016, Q|820477, Q|2385804, Q|13235160, Q|13226383, Q|6056746, Q|1474440, Q|22988604, Q|21070568, Q|167037, Q|2085381, Q|431289, Q|15265344",
  "value": null,
  "value_current_2026": [
    "Q6823840"
  ],
  "value_current_2026_descriptions_en": [
    "Christian conversion effort in U.S. state"
  ],
  "value_current_2026_labels_en": [
    "Methodist Mission"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "person or organization for which the subject works or worked",
    "label": "employer"
  },
  "qid": {
    "description": "American pioneer",
    "label": "Josiah Lamberson Parrish"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21510865",
      "mapped_violation_family": "value_type",
      "violation_name": "Value type Q|43229, Q|14623646, Q|16334295, Q|5, Q|95074, Q|327333, Q|1248784, Q|1002697, Q|102496, Q|13002315, Q|5446565, Q|35127, Q|783794, Q|10648343, Q|15632617, Q|11812394, Q|2659904, Q|15416, Q|16560, Q|24634210, Q|15911314, Q|2001305, Q|3152824, Q|56061, Q|11032, Q|1555508, Q|245016, Q|820477, Q|2385804, Q|13235160, Q|13226383, Q|6056746, Q|1474440, Q|22988604, Q|21070568, Q|167037, Q|2085381, Q|431289, Q|15265344"
    }
  ],
  "candidate_violation_names": [
    "Value type Q|43229, Q|14623646, Q|16334295, Q|5, Q|95074, Q|327333, Q|1248784, Q|1002697, Q|102496, Q|13002315, Q|5446565, Q|35127, Q|783794, Q|10648343, Q|15632617, Q|11812394, Q|2659904, Q|15416, Q|16560, Q|24634210, Q|15911314, Q|2001305, Q|3152824, Q|56061, Q|11032, Q|1555508, Q|245016, Q|820477, Q|2385804, Q|13235160, Q|13226383, Q|6056746, Q|1474440, Q|22988604, Q|21070568, Q|167037, Q|2085381, Q|431289, Q|15265344"
  ],
  "causality_match_level": "exact_constraint_and_value_match",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "value_type_compatible_report_argument_overlap",
  "compatible_overlap_used": true,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [
    "Q15265344"
  ],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "value-type constraint",
  "mapped_violation_constraint_qid": "Q21510865",
  "mapped_violation_family": "value_type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Value type Q|43229, Q|14623646, Q|16334295, Q|5, Q|95074, Q|327333, Q|1248784, Q|1002697, Q|102496, Q|13002315, Q|5446565, Q|35127, Q|783794, Q|10648343, Q|15632617, Q|11812394, Q|2659904, Q|15416, Q|16560, Q|24634210, Q|15911314, Q|2001305, Q|3152824, Q|56061, Q|11032, Q|1555508, Q|245016, Q|820477, Q|2385804, Q|13235160, Q|13226383, Q|6056746, Q|1474440, Q|22988604, Q|21070568, Q|167037, Q|2085381, Q|431289, Q|15265344",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q15265344"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_value_count": 4,
  "semantic_removed_values": [
    "Q105420",
    "Q1370598",
    "Q3895768",
    "Q4164871"
  ],
  "set_operation": "mixed",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "value-type constraint",
  "target_constraint_qid": "Q21510865",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [
    "Q15265344"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q15265344"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "changed_qualifier_properties": [
    "P2308"
  ],
  "compatible_overlap_reason": "value_type_compatible_report_argument_overlap",
  "compatible_overlap_used": true,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "mixed qualifier-value change",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "both_added_and_removed_values",
  "potential_set_operation": "mixed",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 4,
  "removed_values": [
    "Q105420",
    "Q1370598",
    "Q3895768",
    "Q4164871"
  ],
  "selected_violation_name": "Value type Q|43229, Q|14623646, Q|16334295, Q|5, Q|95074, Q|327333, Q|1248784, Q|1002697, Q|102496, Q|13002315, Q|5446565, Q|35127, Q|783794, Q|10648343, Q|15632617, Q|11812394, Q|2659904, Q|15416, Q|16560, Q|24634210, Q|15911314, Q|2001305, Q|3152824, Q|56061, Q|11032, Q|1555508, Q|245016, Q|820477, Q|2385804, Q|13235160, Q|13226383, Q|6056746, Q|1474440, Q|22988604, Q|21070568, Q|167037, Q|2085381, Q|431289, Q|15265344",
  "semantic_added_values": [
    "Q15265344"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_values": [
    "Q105420",
    "Q1370598",
    "Q3895768",
    "Q4164871"
  ],
  "set_operation": "mixed",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "value-type constraint",
  "target_constraint_qid": "Q21510865",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Andre Engels",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "e63d4c64338b2bd9c36a260f5756f78d691d9e0d",
  "hash_before": "8f57ec6ac84b5b83e7938210b45343032c5e6154",
  "property_revision_id": 2437412687,
  "property_revision_prev": 2433444260,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_and_value_match",
    "mapped_violation_constraint_qid": "Q21510865",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Value type Q|43229, Q|14623646, Q|16334295, Q|5, Q|95074, Q|327333, Q|1248784, Q|1002697, Q|102496, Q|13002315, Q|5446565, Q|35127, Q|783794, Q|10648343, Q|15632617, Q|11812394, Q|2659904, Q|15416, Q|16560, Q|24634210, Q|15911314, Q|2001305, Q|3152824, Q|56061, Q|11032, Q|1555508, Q|245016, Q|820477, Q|2385804, Q|13235160, Q|13226383, Q|6056746, Q|1474440, Q|22988604, Q|21070568, Q|167037, Q|2085381, Q|431289, Q|15265344"
  },
  {
    "result": "Q21510865",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 4,
    "result": "SCHEMA_UPDATE",
    "set_operation": "mixed",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q15265344"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21510865",
        "mapped_violation_family": "value_type",
        "violation_name": "Value type Q|43229, Q|14623646, Q|16334295, Q|5, Q|95074, Q|327333, Q|1248784, Q|1002697, Q|102496, Q|13002315, Q|5446565, Q|35127, Q|783794, Q|10648343, Q|15632617, Q|11812394, Q|2659904, Q|15416, Q|16560, Q|24634210, Q|15911314, Q|2001305, Q|3152824, Q|56061, Q|11032, Q|1555508, Q|245016, Q|820477, Q|2385804, Q|13235160, Q|13226383, Q|6056746, Q|1474440, Q|22988604, Q|21070568, Q|167037, Q|2085381, Q|431289, Q|15265344"
      }
    ],
    "candidate_violation_names": [
      "Value type Q|43229, Q|14623646, Q|16334295, Q|5, Q|95074, Q|327333, Q|1248784, Q|1002697, Q|102496, Q|13002315, Q|5446565, Q|35127, Q|783794, Q|10648343, Q|15632617, Q|11812394, Q|2659904, Q|15416, Q|16560, Q|24634210, Q|15911314, Q|2001305, Q|3152824, Q|56061, Q|11032, Q|1555508, Q|245016, Q|820477, Q|2385804, Q|13235160, Q|13226383, Q|6056746, Q|1474440, Q|22988604, Q|21070568, Q|167037, Q|2085381, Q|431289, Q|15265344"
    ],
    "causality_match_level": "exact_constraint_and_value_match",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510865"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510865"
    ],
    "changed_qualifier_properties": [
      "P2308"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "value_type_compatible_report_argument_overlap",
    "compatible_overlap_used": true,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [
      "Q15265344"
    ],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "value-type constraint",
    "mapped_report_constraint_qid": "Q21510865",
    "mapped_report_family": "value_type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "value-type constraint",
    "mapped_violation_constraint_qid": "Q21510865",
    "mapped_violation_family": "value_type",
    "mapped_violation_reason": "value_type_prefix_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "mixed qualifier-value change",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "both_added_and_removed_values",
    "potential_set_operation": "mixed",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 4,
    "removed_values": [
      "Q105420",
      "Q1370598",
      "Q3895768",
      "Q4164871"
    ],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Value type Q|43229, Q|14623646, Q|16334295, Q|5, Q|95074, Q|327333, Q|1248784, Q|1002697, Q|102496, Q|13002315, Q|5446565, Q|35127, Q|783794, Q|10648343, Q|15632617, Q|11812394, Q|2659904, Q|15416, Q|16560, Q|24634210, Q|15911314, Q|2001305, Q|3152824, Q|56061, Q|11032, Q|1555508, Q|245016, Q|820477, Q|2385804, Q|13235160, Q|13226383, Q|6056746, Q|1474440, Q|22988604, Q|21070568, Q|167037, Q|2085381, Q|431289, Q|15265344",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q15265344"
    ],
    "semantic_changed_qualifier_properties": [
      "P2308"
    ],
    "semantic_removed_value_count": 4,
    "semantic_removed_values": [
      "Q105420",
      "Q1370598",
      "Q3895768",
      "Q4164871"
    ],
    "set_operation": "mixed",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "value-type constraint",
    "target_constraint_qid": "Q21510865",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "value_type",
    "value_overlap_with_report_qids": [
      "Q15265344"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Value type Q|43229, Q|14623646, Q|16334295, Q|5, Q|95074, Q|327333, Q|1248784, Q|1002697, Q|102496, Q|13002315, Q|5446565, Q|35127, Q|783794, Q|10648343, Q|15632617, Q|11812394, Q|2659904, Q|15416, Q|16560, Q|24634210, Q|15911314, Q|2001305, Q|3152824, Q|56061, Q|11032, Q|1555508, Q|245016, Q|820477, Q|2385804, Q|13235160, Q|13226383, Q|6056746, Q|1474440, Q|22988604, Q|21070568, Q|167037, Q|2085381, Q|431289, Q|15265344"
  }
]
```

---

## 025. `reform_Q89098418_P370_1714119687`

| Field | Value |
|---|---|
| qid | Q89098418 |
| property | P370 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q52004125 allowed entity types constraint |
| group_key | TBOX::P370::1714119687 |
| tbox_revision_key | TBOX::P370::1714119687 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "allowed entity types constraint",
  "decision_constraint_type_qid": "Q52004125"
}
```

#### Repair Target

```json
{
  "author": "JesseW",
  "kind": "T_BOX",
  "property_revision_id": 1714119687,
  "property_revision_prev": 1696340793
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-08-27T17:24:29",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P370",
  "report_revision_new": 1714610346,
  "report_revision_old": 1703730507,
  "report_violation_type": "Entity types",
  "report_violation_type_normalized": "Entity types",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Entity types",
  "value": null,
  "value_current_2026": [
    "MSTTDALITGAEASPASPATPPSPPTSPAATGPASTAAVTTYTVGIDIGSATTRVAIMDPAKLMPTIIRNALGNEATSTVVSFAANEARSFGENAAARQVTKASETIVDLAPWIFGYTSGEQVPKDTAVSLQVVSPATAMTEAVVLKSRQLGTQEHLTHPAQVAAFYIKSLLQFLPDKEMIRTSPVCLAVPSAACAASFEALRQAAFLAGVPQEKTIIAHSDEATAVYFHHLQYRSLPAKHEGAAVPVVLIDIGQSCSVASLIIASQPRVEKVGCQTLRMGSEYIDTLLCSHVYSELGKKFGAAADPLRGDIKSFRKILRECRKAKEVLSTADETQVQLEGLSGDIDIIVSVTRAMMEQAALPFLQAVRAMLTAIKEKLPEPKATEDGSQAAAVPPRVEVIGGGWRSVCVMEAIREVLGITRVGVSLDANLSVAEGSAILAEVRRLTIARQQREQDEEQTAQDTAASS... [truncated 314 chars]"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "Sandbox property for the data value type \"String\"",
    "label": "Sandbox-String"
  },
  "qid": {
    "description": "Leishmania infantum protein-coding gene",
    "label": "LINF_290018300"
  }
}
```

### Constraint Types

_empty_

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q52004125",
      "mapped_violation_family": "allowed_entity_types",
      "violation_name": "Entity types"
    }
  ],
  "candidate_violation_names": [
    "Entity types"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q52004125"
  ],
  "changed_constraint_qids_from_entries": [
    "Q52004125"
  ],
  "changed_constraint_qids_from_qualifier_changes": [],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "allowed entity types constraint",
  "mapped_report_constraint_qid": "Q52004125",
  "mapped_report_family": "allowed_entity_types",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "allowed entity types constraint",
  "mapped_violation_constraint_qid": "Q52004125",
  "mapped_violation_family": "allowed_entity_types",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Entity types",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed entity types constraint",
  "target_constraint_qid": "Q52004125",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 0,
  "added_values": [],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q52004125"
  ],
  "changed_constraint_qids_from_entries": [
    "Q52004125"
  ],
  "changed_constraint_qids_from_qualifier_changes": [],
  "changed_qualifier_properties": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "allowed entity types constraint",
  "mapped_report_constraint_qid": "Q52004125",
  "mapped_report_family": "allowed_entity_types",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "causal family match without interpretable polarity",
  "potential_directional_subtype_precise": null,
  "potential_polarity": "unknown",
  "potential_polarity_basis": "unknown constraint-family polarity",
  "potential_set_operation": "unchanged",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Entity types",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": "unchanged",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed entity types constraint",
  "target_constraint_qid": "Q52004125",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "JesseW",
  "before_constraint_count": 0,
  "changed_constraint_types": [
    "Q52004125"
  ],
  "constraints_readable_en": null,
  "hash_after": "f7189ae451113c8aa58775454620c956b93fc50b",
  "hash_before": "b6a585920c5dbc697b1df3acf4c4edcfb28ddcf2",
  "property_revision_id": 1714119687,
  "property_revision_prev": 1696340793,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only",
    "mapped_violation_constraint_qid": "Q52004125",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Entity types"
  },
  {
    "result": "Q52004125",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 0,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_ids": [],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 0,
    "added_values": [],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q52004125",
        "mapped_violation_family": "allowed_entity_types",
        "violation_name": "Entity types"
      }
    ],
    "candidate_violation_names": [
      "Entity types"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q52004125"
    ],
    "changed_constraint_qids_from_entries": [
      "Q52004125"
    ],
    "changed_constraint_qids_from_qualifier_changes": [],
    "changed_qualifier_properties": [],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "allowed entity types constraint",
    "mapped_report_constraint_qid": "Q52004125",
    "mapped_report_family": "allowed_entity_types",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "allowed entity types constraint",
    "mapped_violation_constraint_qid": "Q52004125",
    "mapped_violation_family": "allowed_entity_types",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "causal family match without interpretable polarity",
    "potential_directional_subtype_precise": null,
    "potential_polarity": "unknown",
    "potential_polarity_basis": "unknown constraint-family polarity",
    "potential_set_operation": "unchanged",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Entity types",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "unchanged",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "allowed entity types constraint",
    "target_constraint_qid": "Q52004125",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "allowed_entity_types",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Entity types"
  }
]
```

---

## 026. `reform_Q97382769_P123_2442705670`

| Field | Value |
|---|---|
| qid | Q97382769 |
| property | P123 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | schema_update |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P123::2442705670 |
| tbox_revision_key | TBOX::P123::2442705670 |

### Annotation Focus

- Check whether the changed constraint family matches the reported violation family.
- Use value/property/language overlap to decide whether the schema edit is causal, coincidental, or unknown.
- For directional labels, verify that polarity is correct for allowed, forbidden, or required set semantics.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | [] |
| classification_target_reason | no changed semantic value tokens |
| decision_branch |  |
| rationale | Constraint family matched the violation, but polarity could not be interpreted directionally. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "no changed semantic value tokens",
  "classification_target_role": "none",
  "classification_target_tokens": [],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "NO_CHANGE_OR_REORDER_ONLY"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "tbox_schema_causality",
  "classification_rule_subfamily": "schema_update",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Trade",
  "kind": "T_BOX",
  "property_revision_id": 2442705670,
  "property_revision_prev": 2433642579
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-16T13:48:39",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P123",
  "report_revision_new": 2443007244,
  "report_revision_old": 2442702725,
  "report_violation_type": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645",
  "report_violation_type_descriptions_en": [
    "prescription, including laws, regulations, instructions, guidelines, and social conventions; determinate method for performing any operation",
    "type of list that is ranked by some criteria",
    "a group of software licenses",
    "formal attestation of certain characteristics of an object, person, or organization",
    "non-tangible executable component of a computer",
    "physical or digital embodiment of an information artifact",
    "set of manifestations as defined in FRBR",
    "medium for recording information (words or images) typically on bound pages or more abstractly in electronic or audio form",
    "structured form of play",
    "Wikidata metaclass; legal concept of uniquely identifiable piece or work of music, either vocal or instrumental; NOT applicable to recordings, broadcasts, or individual publications of music in printed or digital form or on physical media",
    "multiple video games marketed under the same series name",
    "two or more compositions published under, or otherwise known by, a common name",
    "סוג קבוצת משחקי וידאו",
    "work manifested on the Internet",
    "presentation of a series of still images",
    "compilation of software, in most cases, from the same developer",
    "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
    "artistic work drawn with the aid of a computer",
    "online service provided on Minitel",
    "list of flora and/or fauna of a place or in a particular taxonomic group",
    "section of a work, most commonly a book",
    "content made available to the general public",
    "widely-accepted foundational work in a specific field, profession, or discipline",
    "unbound collection of visual artworks housed in a binder, folder or other container",
    "... omitted 2 items"
  ],
  "report_violation_type_labels_en": [
    "rule",
    "ranked list",
    "license scheme",
    "certification",
    "software",
    "manifestation",
    "group of manifestations",
    "book",
    "game",
    "musical work/composition",
    "video game series",
    "group of musical works",
    "group of video games often treated as a singular game",
    "online publication",
    "slide show",
    "software bundle",
    "version, edition or translation",
    "digital artistic drawing",
    "telematics service",
    "checklist",
    "chapter",
    "publication",
    "standard work",
    "portfolio",
    "... omitted 2 items"
  ],
  "report_violation_type_normalized": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645",
  "report_violation_type_qids": [
    "Q1151067",
    "Q80793969",
    "Q95107111",
    "Q374814",
    "Q7397",
    "Q286583",
    "Q17538690",
    "Q571",
    "Q11410",
    "Q105543609",
    "Q7058673",
    "Q115473170",
    "Q116779426",
    "Q1714118",
    "Q904997",
    "Q62651817",
    "Q3331189",
    "Q97180164",
    "Q124030631",
    "Q106140535",
    "Q1980247",
    "Q732577",
    "Q1748756",
    "Q49094714",
    "... omitted 2 items"
  ],
  "report_violation_type_raw": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645",
  "value": null,
  "value_current_2026": [
    "Q248326"
  ],
  "value_current_2026_descriptions_en": [
    "Soviet and Russian publishing house"
  ],
  "value_current_2026_labels_en": [
    "Nauka"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "organization or person responsible for publishing a work, such as a book, periodical, printed music, podcast, game or software",
    "label": "publisher"
  },
  "qid": {
    "description": null,
    "label": "Русские фамилии тюркского происхождения"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_schema_update",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645"
    }
  ],
  "candidate_violation_names": [
    "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": null,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q8274"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "type constraint",
  "target_constraint_qid": "Q21503250",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q8274"
  ],
  "analysis_slice_precise": "main_tbox_schema_update",
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": [
    "P2308"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "unknown",
  "polarity_basis": "not active because final T-box subtype is non-directional",
  "potential_directional_subtype_basis": "allowed set expansion",
  "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "potential_polarity": "relaxation",
  "potential_polarity_basis": "allowed set gained values",
  "potential_set_operation": "expansion",
  "potential_set_semantics": "allowed",
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645",
  "semantic_added_values": [
    "Q8274"
  ],
  "semantic_changed_qualifier_properties": [
    "P2308"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "type constraint",
  "target_constraint_qid": "Q21503250",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_specific_without_overlap": true
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Trade",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "43b1fc0c8abf83553480e2e6e6f43a40a4839f91",
  "hash_before": "b3587ce8e60c5825467ef95c0c087e052fd7cb39",
  "property_revision_id": 2442705670,
  "property_revision_prev": 2433642579,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "mapped_violation_constraint_qid": "Q21503250",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_schema_update",
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "SCHEMA_UPDATE",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q8274"
    ],
    "analysis_slice_precise": "main_tbox_schema_update",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645"
      }
    ],
    "candidate_violation_names": [
      "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503250"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503250"
    ],
    "changed_qualifier_properties": [
      "P2308"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": null,
    "directional_subtype_precise": null,
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "type constraint",
    "mapped_report_constraint_qid": "Q21503250",
    "mapped_report_family": "type",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "type constraint",
    "mapped_violation_constraint_qid": "Q21503250",
    "mapped_violation_family": "type",
    "mapped_violation_reason": "type_prefix_mapping",
    "polarity": "unknown",
    "polarity_basis": "not active because final T-box subtype is non-directional",
    "potential_directional_subtype_basis": "allowed set expansion",
    "potential_directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "potential_polarity": "relaxation",
    "potential_polarity_basis": "allowed set gained values",
    "potential_set_operation": "expansion",
    "potential_set_semantics": "allowed",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "SCHEMA_UPDATE",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q8274"
    ],
    "semantic_changed_qualifier_properties": [
      "P2308"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "type constraint",
    "target_constraint_qid": "Q21503250",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "type",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645"
  }
]
```

---
