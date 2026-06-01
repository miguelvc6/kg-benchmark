# TBOX_DIRECTIONAL_RELAXATION_OR_RESTRICTION

Cases: 25

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `reform_Q105336230_P4945_2438810266`

| Field | Value |
|---|---|
| qid | Q105336230 |
| property | P4945 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P4945::2438810266 |
| tbox_revision_key | TBOX::P4945::2438810266 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Eurohunter",
  "kind": "T_BOX",
  "property_revision_id": 2438810266,
  "property_revision_prev": 2392458714
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-09T07:22:23",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4945",
  "report_revision_new": 2439902615,
  "report_revision_old": 2439494672,
  "report_violation_type": "Type Q|386724, Q|49848, Q|694975, Q|17709279",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "form for preservation of structured and identified information",
    "type of document stored as a computer file",
    "digital or physical document or other work which is attached to, but not transcluded in, another work"
  ],
  "report_violation_type_labels_en": [
    "work",
    "document",
    "electronic document",
    "attachment"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|49848, Q|694975, Q|17709279",
  "report_violation_type_qids": [
    "Q386724",
    "Q49848",
    "Q694975",
    "Q17709279"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|49848, Q|694975, Q|17709279",
  "value": null,
  "value_current_2026": [
    "http://leish-esp.cbm.uam.es/browser_donovani2.php"
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
    "description": "URL which can be used to download a work",
    "label": "download URL"
  },
  "qid": {
    "description": "Leishmania donovani protein-coding gene",
    "label": "LDHU3_19.1470"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|386724, Q|49848, Q|694975, Q|17709279"
    }
  ],
  "candidate_violation_names": [
    "Type Q|386724, Q|49848, Q|694975, Q|17709279"
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
    "Q17709279",
    "Q49848",
    "Q694975"
  ],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|386724, Q|49848, Q|694975, Q|17709279",
  "semantic_added_value_count": 3,
  "semantic_added_values": [
    "Q17709279",
    "Q49848",
    "Q694975"
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
  "value_overlap_with_report_qids": [
    "Q17709279",
    "Q49848",
    "Q694975"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 3,
  "added_values": [
    "Q17709279",
    "Q49848",
    "Q694975"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
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
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Type Q|386724, Q|49848, Q|694975, Q|17709279",
  "semantic_added_values": [
    "Q17709279",
    "Q49848",
    "Q694975"
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
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Eurohunter",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "9705da1767e183dcb1c9bb1d6d0875e353872540",
  "hash_before": "3fe081598bd314d0ddae280a5786d3212a7c56c5",
  "property_revision_id": 2438810266,
  "property_revision_prev": 2392458714,
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
    "violation_name": "Type Q|386724, Q|49848, Q|694975, Q|17709279"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 3,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 3,
    "added_values": [
      "Q17709279",
      "Q49848",
      "Q694975"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|386724, Q|49848, Q|694975, Q|17709279"
      }
    ],
    "candidate_violation_names": [
      "Type Q|386724, Q|49848, Q|694975, Q|17709279"
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
      "Q17709279",
      "Q49848",
      "Q694975"
    ],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|386724, Q|49848, Q|694975, Q|17709279",
    "semantic_added_value_count": 3,
    "semantic_added_values": [
      "Q17709279",
      "Q49848",
      "Q694975"
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
    "value_overlap_with_report_qids": [
      "Q17709279",
      "Q49848",
      "Q694975"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|386724, Q|49848, Q|694975, Q|17709279"
  }
]
```

---

## 002. `reform_Q107447197_P170_2440830812`

| Field | Value |
|---|---|
| qid | Q107447197 |
| property | P170 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21510865 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21510851 allowed qualifiers constraint |
| group_key | TBOX::P170::2440830812 |
| tbox_revision_key | TBOX::P170::2440830812 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "allowed qualifiers constraint",
  "decision_constraint_type_qid": "Q21510851"
}
```

#### Repair Target

```json
{
  "author": "DaxServer",
  "kind": "T_BOX",
  "property_revision_id": 2440830812,
  "property_revision_prev": 2436332374
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-13T11:35:03",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P170",
  "report_revision_new": 2441803264,
  "report_revision_old": 2441245748,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
  "value": null,
  "value_current_2026": [
    "Q568760",
    "Q499934"
  ],
  "value_current_2026_descriptions_en": [
    "German painter (1500-1543)",
    "German painter and engraver (1502-1540)"
  ],
  "value_current_2026_labels_en": [
    "Master of Messkirch",
    "Barthel Beham"
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
    "description": "maker of this creative work or other object (where no more specific property exists)",
    "label": "creator"
  },
  "qid": {
    "description": "painting by Meister von Meßkirch",
    "label": "Der heilige Crispinianus"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21510851",
      "mapped_violation_family": "allowed_qualifier",
      "violation_name": "Allowed qualifiers"
    }
  ],
  "candidate_violation_names": [
    "Allowed qualifiers"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "allowed qualifiers constraint",
  "mapped_violation_constraint_qid": "Q21510851",
  "mapped_violation_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "P13988"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
    "P13988"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": [
    "P2306"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_values": [
    "P13988"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
  "author": "DaxServer",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "c21c78328e24d9a9a3cfdee120c374c0a31e2473",
  "hash_before": "1741eab8b4ac0408db08129916e7d7fc1e714511",
  "property_revision_id": 2440830812,
  "property_revision_prev": 2436332374,
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
    "mapped_violation_constraint_qid": "Q21510851",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Allowed qualifiers"
  },
  {
    "result": "Q21510851",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2306"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "P13988"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21510851",
        "mapped_violation_family": "allowed_qualifier",
        "violation_name": "Allowed qualifiers"
      }
    ],
    "candidate_violation_names": [
      "Allowed qualifiers"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "changed_qualifier_properties": [
      "P2306"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "allowed qualifiers constraint",
    "mapped_report_constraint_qid": "Q21510851",
    "mapped_report_family": "allowed_qualifier",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "allowed qualifiers constraint",
    "mapped_violation_constraint_qid": "Q21510851",
    "mapped_violation_family": "allowed_qualifier",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Allowed qualifiers",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "P13988"
    ],
    "semantic_changed_qualifier_properties": [
      "P2306"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "allowed qualifiers constraint",
    "target_constraint_qid": "Q21510851",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "allowed_qualifier",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Allowed qualifiers"
  }
]
```

---

## 003. `reform_Q110852008_P1787_2445837416`

| Field | Value |
|---|---|
| qid | Q110852008 |
| property | P1787 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21503247 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21510851 allowed qualifiers constraint |
| group_key | TBOX::P1787::2445837416 |
| tbox_revision_key | TBOX::P1787::2445837416 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "allowed qualifiers constraint",
  "decision_constraint_type_qid": "Q21510851"
}
```

#### Repair Target

```json
{
  "author": "Clemens Dulcis",
  "kind": "T_BOX",
  "property_revision_id": 2445837416,
  "property_revision_prev": 2433734615
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-23T10:53:13",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1787",
  "report_revision_new": 2445981692,
  "report_revision_old": 2445411959,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
  "value": null,
  "value_current_2026": [
    "青山 ネモフィラ"
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
    "description": "type of pseudonym traditionally adopted by writers and artists in East Asia",
    "label": "art name"
  },
  "qid": {
    "description": "Japanese idol and Nogizaka46 member",
    "label": "Aruno Nakanishi"
  }
}
```

### Constraint Types

```json
[
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21510851",
      "mapped_violation_family": "allowed_qualifier",
      "violation_name": "Allowed qualifiers"
    }
  ],
  "candidate_violation_names": [
    "Allowed qualifiers"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "allowed qualifiers constraint",
  "mapped_violation_constraint_qid": "Q21510851",
  "mapped_violation_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_value_count": 3,
  "semantic_added_values": [
    "P1319",
    "P580",
    "P8555"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 3,
  "added_values": [
    "P1319",
    "P580",
    "P8555"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": [
    "P2306"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_values": [
    "P1319",
    "P580",
    "P8555"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
  "hash_after": "9461c17df98d935a5696d9c36dbf7f90d755eb4d",
  "hash_before": "6bedcc4c24683cc9243a0535971a2d1305123727",
  "property_revision_id": 2445837416,
  "property_revision_prev": 2433734615,
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
    "mapped_violation_constraint_qid": "Q21510851",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Allowed qualifiers"
  },
  {
    "result": "Q21510851",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 3,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2306"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 3,
    "added_values": [
      "P1319",
      "P580",
      "P8555"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21510851",
        "mapped_violation_family": "allowed_qualifier",
        "violation_name": "Allowed qualifiers"
      }
    ],
    "candidate_violation_names": [
      "Allowed qualifiers"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "changed_qualifier_properties": [
      "P2306"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "allowed qualifiers constraint",
    "mapped_report_constraint_qid": "Q21510851",
    "mapped_report_family": "allowed_qualifier",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "allowed qualifiers constraint",
    "mapped_violation_constraint_qid": "Q21510851",
    "mapped_violation_family": "allowed_qualifier",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Allowed qualifiers",
    "semantic_added_value_count": 3,
    "semantic_added_values": [
      "P1319",
      "P580",
      "P8555"
    ],
    "semantic_changed_qualifier_properties": [
      "P2306"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "allowed qualifiers constraint",
    "target_constraint_qid": "Q21510851",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "allowed_qualifier",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Allowed qualifiers"
  }
]
```

---

## 004. `reform_Q11802292_P31_2436953325`

| Field | Value |
|---|---|
| qid | Q11802292 |
| property | P31 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21510851 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21510851 allowed qualifiers constraint |
| group_key | TBOX::P31::2436953325 |
| tbox_revision_key | TBOX::P31::2436953325 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "allowed qualifiers constraint",
  "decision_constraint_type_qid": "Q21510851"
}
```

#### Repair Target

```json
{
  "author": "Tokrkbot",
  "kind": "T_BOX",
  "property_revision_id": 2436953325,
  "property_revision_prev": 2435928020
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-08T16:07:16",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
  "report_revision_new": 2439641943,
  "report_revision_old": 2439287245,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
  "value": null,
  "value_current_2026": [
    "Q3055118"
  ],
  "value_current_2026_descriptions_en": [
    "Spanish statistical unit of human settlements"
  ],
  "value_current_2026_labels_en": [
    "single entity of population"
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
    "description": "human settlement in Spain",
    "label": "Palagret"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21510851",
      "mapped_violation_family": "allowed_qualifier",
      "violation_name": "Allowed qualifiers"
    }
  ],
  "candidate_violation_names": [
    "Allowed qualifiers"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "allowed qualifiers constraint",
  "mapped_violation_constraint_qid": "Q21510851",
  "mapped_violation_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "P1793"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
    "P1793"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": [
    "P2306"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_values": [
    "P1793"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
  "author": "Tokrkbot",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "ce4232e77bc7d6290fc000803a238c5704d79301",
  "hash_before": "f05c09c794215817aa2b794c67557405cbc8e889",
  "property_revision_id": 2436953325,
  "property_revision_prev": 2435928020,
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
    "mapped_violation_constraint_qid": "Q21510851",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Allowed qualifiers"
  },
  {
    "result": "Q21510851",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2306"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "P1793"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21510851",
        "mapped_violation_family": "allowed_qualifier",
        "violation_name": "Allowed qualifiers"
      }
    ],
    "candidate_violation_names": [
      "Allowed qualifiers"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "changed_qualifier_properties": [
      "P2306"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "allowed qualifiers constraint",
    "mapped_report_constraint_qid": "Q21510851",
    "mapped_report_family": "allowed_qualifier",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "allowed qualifiers constraint",
    "mapped_violation_constraint_qid": "Q21510851",
    "mapped_violation_family": "allowed_qualifier",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Allowed qualifiers",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "P1793"
    ],
    "semantic_changed_qualifier_properties": [
      "P2306"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "allowed qualifiers constraint",
    "target_constraint_qid": "Q21510851",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "allowed_qualifier",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Allowed qualifiers"
  }
]
```

---

## 005. `reform_Q1192156_P2534_2441000798`

| Field | Value |
|---|---|
| qid | Q1192156 |
| property | P2534 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21510851 allowed qualifiers constraint |
| group_key | TBOX::P2534::2441000798 |
| tbox_revision_key | TBOX::P2534::2441000798 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "allowed qualifiers constraint",
  "decision_constraint_type_qid": "Q21510851"
}
```

#### Repair Target

```json
{
  "author": "Swpb",
  "kind": "T_BOX",
  "property_revision_id": 2441000798,
  "property_revision_prev": 2419406526
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-17T09:34:39",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2534",
  "report_revision_new": 2443350017,
  "report_revision_old": 2442585791,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
  "report_violation_types": [
    "Allowed qualifiers",
    "Item P|7235"
  ],
  "value": null,
  "value_current_2026": [
    "L_{C}^2=4\\pi A_{C}+8\\pi\\big|\\widetilde{A}_{0.5}\\big|"
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
    "description": "mathematical formula representing a theorem or law",
    "label": "defining formula"
  },
  "qid": {
    "description": "convex planar shape whose width is the same regardless of the orientation of the curve",
    "label": "curve of constant width"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21510851",
      "mapped_violation_family": "allowed_qualifier",
      "violation_name": "Allowed qualifiers"
    },
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|7235"
    }
  ],
  "candidate_violation_names": [
    "Allowed qualifiers",
    "Item P|7235"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "allowed qualifiers constraint",
  "mapped_violation_constraint_qid": "Q21510851",
  "mapped_violation_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "P3831"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
    "P3831"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": [
    "P2306"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_values": [
    "P3831"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
  "hash_after": "f50a7da29f58ef236325cfebd2f9189b626ba3b1",
  "hash_before": "7186e131b564b7c502df5ebebfa527765e13ee13",
  "property_revision_id": 2441000798,
  "property_revision_prev": 2419406526,
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
    "mapped_violation_constraint_qid": "Q21510851",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Allowed qualifiers"
  },
  {
    "result": "Q21510851",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2306"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "P3831"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21510851",
        "mapped_violation_family": "allowed_qualifier",
        "violation_name": "Allowed qualifiers"
      },
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|7235"
      }
    ],
    "candidate_violation_names": [
      "Allowed qualifiers",
      "Item P|7235"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "changed_qualifier_properties": [
      "P2306"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "allowed qualifiers constraint",
    "mapped_report_constraint_qid": "Q21510851",
    "mapped_report_family": "allowed_qualifier",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "allowed qualifiers constraint",
    "mapped_violation_constraint_qid": "Q21510851",
    "mapped_violation_family": "allowed_qualifier",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Allowed qualifiers",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "P3831"
    ],
    "semantic_changed_qualifier_properties": [
      "P2306"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "allowed qualifiers constraint",
    "target_constraint_qid": "Q21510851",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "allowed_qualifier",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Allowed qualifiers"
  }
]
```

---

## 006. `reform_Q120081121_P856_2446852358`

| Field | Value |
|---|---|
| qid | Q120081121 |
| property | P856 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21510851 allowed qualifiers constraint |
| group_key | TBOX::P856::2446852358 |
| tbox_revision_key | TBOX::P856::2446852358 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "allowed qualifiers constraint",
  "decision_constraint_type_qid": "Q21510851"
}
```

#### Repair Target

```json
{
  "author": "Clemens Dulcis",
  "kind": "T_BOX",
  "property_revision_id": 2446852358,
  "property_revision_prev": 2443941926
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-25T18:00:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P856",
  "report_revision_new": 2447046423,
  "report_revision_old": 2446480257,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
  "value": null,
  "value_current_2026": [
    "https://lobones.com/"
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
    "description": "abandoned village in the province of Segovia, Spain",
    "label": "Lobones"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21510851",
      "mapped_violation_family": "allowed_qualifier",
      "violation_name": "Allowed qualifiers"
    }
  ],
  "candidate_violation_names": [
    "Allowed qualifiers"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "allowed qualifiers constraint",
  "mapped_violation_constraint_qid": "Q21510851",
  "mapped_violation_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "P7081"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
    "P7081"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": [
    "P2306"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_values": [
    "P7081"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
  "hash_after": "a36a61f0e86f73f6deb18c55577835a1b9bbf650",
  "hash_before": "a24e4427528acfa48d593fe8b3d3db4c276c2009",
  "property_revision_id": 2446852358,
  "property_revision_prev": 2443941926,
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
    "mapped_violation_constraint_qid": "Q21510851",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Allowed qualifiers"
  },
  {
    "result": "Q21510851",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2306"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "P7081"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21510851",
        "mapped_violation_family": "allowed_qualifier",
        "violation_name": "Allowed qualifiers"
      }
    ],
    "candidate_violation_names": [
      "Allowed qualifiers"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "changed_qualifier_properties": [
      "P2306"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "allowed qualifiers constraint",
    "mapped_report_constraint_qid": "Q21510851",
    "mapped_report_family": "allowed_qualifier",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "allowed qualifiers constraint",
    "mapped_violation_constraint_qid": "Q21510851",
    "mapped_violation_family": "allowed_qualifier",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Allowed qualifiers",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "P7081"
    ],
    "semantic_changed_qualifier_properties": [
      "P2306"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "allowed qualifiers constraint",
    "target_constraint_qid": "Q21510851",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "allowed_qualifier",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Allowed qualifiers"
  }
]
```

---

## 007. `reform_Q126723126_P675_2441373454`

| Field | Value |
|---|---|
| qid | Q126723126 |
| property | P675 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P675::2441373454 |
| tbox_revision_key | TBOX::P675::2441373454 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Ferran Mir",
  "kind": "T_BOX",
  "property_revision_id": 2441373454,
  "property_revision_prev": 2441373019
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-19T10:19:10",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P675",
  "report_revision_new": 2444030376,
  "report_revision_old": 2443820615,
  "report_violation_type": "Type Q|3331189, Q|57933693, Q|136832029, Q|732577, Q|136747113, Q|1266946",
  "report_violation_type_descriptions_en": [
    "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
    "edition of a book",
    "edition of a manga",
    "content made available to the general public",
    "edition of a light novel",
    "work by academic candidate"
  ],
  "report_violation_type_labels_en": [
    "version, edition or translation",
    "book edition",
    "manga edition",
    "publication",
    "light novel edition",
    "thesis"
  ],
  "report_violation_type_normalized": "Type Q|3331189, Q|57933693, Q|136832029, Q|732577, Q|136747113, Q|1266946",
  "report_violation_type_qids": [
    "Q3331189",
    "Q57933693",
    "Q136832029",
    "Q732577",
    "Q136747113",
    "Q1266946"
  ],
  "report_violation_type_raw": "Type Q|3331189, Q|57933693, Q|136832029, Q|732577, Q|136747113, Q|1266946",
  "report_violation_types": [
    "Type Q|3331189, Q|57933693, Q|136832029, Q|732577, Q|136747113, Q|1266946",
    "Conflicts with P|31"
  ],
  "value": null,
  "value_current_2026": [
    "Ncy7AAAAIAAJ"
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
    "description": "identifier for a book edition in Google Books",
    "label": "Google Books ID"
  },
  "qid": {
    "description": "1905 book",
    "label": "Educational History of Ohio"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|3331189, Q|57933693, Q|136832029, Q|732577, Q|136747113, Q|1266946"
    },
    {
      "candidate_causality_match_level": "constraint_family_mismatch",
      "candidate_score": 5,
      "mapped_violation_constraint_qid": "Q21502838",
      "mapped_violation_family": "conflicts_with",
      "violation_name": "Conflicts with P|31"
    }
  ],
  "candidate_violation_names": [
    "Type Q|3331189, Q|57933693, Q|136832029, Q|732577, Q|136747113, Q|1266946",
    "Conflicts with P|31"
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
    "Q1266946"
  ],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|3331189, Q|57933693, Q|136832029, Q|732577, Q|136747113, Q|1266946",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q1266946"
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
  "value_overlap_with_report_qids": [
    "Q1266946"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q1266946"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
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
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Type Q|3331189, Q|57933693, Q|136832029, Q|732577, Q|136747113, Q|1266946",
  "semantic_added_values": [
    "Q1266946"
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
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Ferran Mir",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "09178c4af278d40d32a8355e9767af3fee78b81d",
  "hash_before": "bab0ba7a85b0843ceb8ef544c0e296eb4bef788d",
  "property_revision_id": 2441373454,
  "property_revision_prev": 2441373019,
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
    "violation_name": "Type Q|3331189, Q|57933693, Q|136832029, Q|732577, Q|136747113, Q|1266946"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q1266946"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|3331189, Q|57933693, Q|136832029, Q|732577, Q|136747113, Q|1266946"
      },
      {
        "candidate_causality_match_level": "constraint_family_mismatch",
        "candidate_score": 5,
        "mapped_violation_constraint_qid": "Q21502838",
        "mapped_violation_family": "conflicts_with",
        "violation_name": "Conflicts with P|31"
      }
    ],
    "candidate_violation_names": [
      "Type Q|3331189, Q|57933693, Q|136832029, Q|732577, Q|136747113, Q|1266946",
      "Conflicts with P|31"
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
      "Q1266946"
    ],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|3331189, Q|57933693, Q|136832029, Q|732577, Q|136747113, Q|1266946",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q1266946"
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
    "value_overlap_with_report_qids": [
      "Q1266946"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|3331189, Q|57933693, Q|136832029, Q|732577, Q|136747113, Q|1266946"
  }
]
```

---

## 008. `reform_Q131862896_P179_2445358570`

| Field | Value |
|---|---|
| qid | Q131862896 |
| property | P179 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21510851 allowed qualifiers constraint |
| group_key | TBOX::P179::2445358570 |
| tbox_revision_key | TBOX::P179::2445358570 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "allowed qualifiers constraint",
  "decision_constraint_type_qid": "Q21510851"
}
```

#### Repair Target

```json
{
  "author": "Clemens Dulcis",
  "kind": "T_BOX",
  "property_revision_id": 2445358570,
  "property_revision_prev": 2435838794
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-22T11:51:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P179",
  "report_revision_new": 2445486749,
  "report_revision_old": 2444924362,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
  "value": null,
  "value_current_2026": [
    "Q7771505"
  ],
  "value_current_2026_descriptions_en": [
    "series of young adult fantasy novels"
  ],
  "value_current_2026_labels_en": [
    "The Unicorn Chronicles"
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
    "description": "series which contains the subject",
    "label": "part of the series"
  },
  "qid": {
    "description": "2022 novel by Bruce Coville",
    "label": "The Gathered Glory"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21510851",
      "mapped_violation_family": "allowed_qualifier",
      "violation_name": "Allowed qualifiers"
    }
  ],
  "candidate_violation_names": [
    "Allowed qualifiers"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "allowed qualifiers constraint",
  "mapped_violation_constraint_qid": "Q21510851",
  "mapped_violation_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "P3831"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
    "P3831"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": [
    "P2306"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_values": [
    "P3831"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
  "hash_after": "75aef94efdbb1c8b9c43a6d5808ad70ee801124f",
  "hash_before": "0a0324197eb07ad7983405cb72b991a5997520a4",
  "property_revision_id": 2445358570,
  "property_revision_prev": 2435838794,
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
    "mapped_violation_constraint_qid": "Q21510851",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Allowed qualifiers"
  },
  {
    "result": "Q21510851",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2306"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "P3831"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21510851",
        "mapped_violation_family": "allowed_qualifier",
        "violation_name": "Allowed qualifiers"
      }
    ],
    "candidate_violation_names": [
      "Allowed qualifiers"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "changed_qualifier_properties": [
      "P2306"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "allowed qualifiers constraint",
    "mapped_report_constraint_qid": "Q21510851",
    "mapped_report_family": "allowed_qualifier",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "allowed qualifiers constraint",
    "mapped_violation_constraint_qid": "Q21510851",
    "mapped_violation_family": "allowed_qualifier",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Allowed qualifiers",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "P3831"
    ],
    "semantic_changed_qualifier_properties": [
      "P2306"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "allowed qualifiers constraint",
    "target_constraint_qid": "Q21510851",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "allowed_qualifier",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Allowed qualifiers"
  }
]
```

---

## 009. `reform_Q135957623_P2094_2443310519`

| Field | Value |
|---|---|
| qid | Q135957623 |
| property | P2094 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P2094::2443310519 |
| tbox_revision_key | TBOX::P2094::2443310519 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Mitte27",
  "kind": "T_BOX",
  "property_revision_id": 2443310519,
  "property_revision_prev": 2436452255
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-23T10:38:38",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2094",
  "report_revision_new": 2445975769,
  "report_revision_old": 2445407551,
  "report_violation_type": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "report_violation_type_descriptions_en": [
    "event, during which one or more sporting events are held",
    "organization involved primarily in sports",
    "object used for sport or exercise",
    "classification of age groups, sexes and levels within sports",
    "mobile machine used for transport, whether it has an engine or not, including wheeled and tracked vehicles, air-, water-, and space-craft",
    "specific vehicle design of which all instances are produced to identical specifications",
    "quantified value of an event that is more extreme than that of all comparable events",
    "event in sports",
    "any set of human beings",
    "object (cup, medal...) awarded to a sports winner",
    "sports event scheduled to recur within a decided interval",
    "view of the navigation template of the Wikimedia project",
    "participation of a nation at sports competition",
    "sports practiced within a specific region",
    null
  ],
  "report_violation_type_labels_en": [
    "sports competition",
    "sports organization",
    "sports equipment",
    "classification in sports",
    "vehicle",
    "vehicle model",
    "record",
    "sporting event",
    "group of humans",
    "sports award",
    "recurring sporting event",
    "Wikimedia navigational template for sports team squad",
    "nation at sport competition",
    "sport in a geographic region",
    "список участников спортивного турнира"
  ],
  "report_violation_type_normalized": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "report_violation_type_qids": [
    "Q13406554",
    "Q4438121",
    "Q768186",
    "Q1744559",
    "Q42889",
    "Q29048322",
    "Q1241356",
    "Q16510064",
    "Q16334295",
    "Q15229207",
    "Q18608583",
    "Q107285679",
    "Q46135307",
    "Q29791211",
    "Q137424753"
  ],
  "report_violation_type_raw": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "value": null,
  "value_current_2026": [
    "Q606060"
  ],
  "value_current_2026_descriptions_en": [
    "association football when played by women"
  ],
  "value_current_2026_labels_en": [
    "women's association football"
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
    "description": "official classification by a regulating body under which the subject (events, teams, participants, or equipment) qualifies for inclusion",
    "label": "competition class"
  },
  "qid": {
    "description": "football tournament season",
    "label": "2025–26 Copa de la Reina de Fútbol"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
    }
  ],
  "candidate_violation_names": [
    "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
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
    "Q137424753"
  ],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q137424753"
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
  "value_overlap_with_report_qids": [
    "Q137424753"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q137424753"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
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
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "semantic_added_values": [
    "Q137424753"
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
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Mitte27",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "30dbd2dbb202d34107cdea391b1a7a596cbfaf87",
  "hash_before": "e054bcd1226272b2c0a6a49a829d9fde160ba75f",
  "property_revision_id": 2443310519,
  "property_revision_prev": 2436452255,
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
    "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q137424753"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
      }
    ],
    "candidate_violation_names": [
      "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
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
      "Q137424753"
    ],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q137424753"
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
    "value_overlap_with_report_qids": [
      "Q137424753"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
  }
]
```

---

## 010. `reform_Q136650135_P1399_2443999770`

| Field | Value |
|---|---|
| qid | Q136650135 |
| property | P1399 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21510865 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21510851 allowed qualifiers constraint |
| group_key | TBOX::P1399::2443999770 |
| tbox_revision_key | TBOX::P1399::2443999770 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "allowed qualifiers constraint",
  "decision_constraint_type_qid": "Q21510851"
}
```

#### Repair Target

```json
{
  "author": "Pallor",
  "kind": "T_BOX",
  "property_revision_id": 2443999770,
  "property_revision_prev": 2442311007
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-19T09:39:49",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1399",
  "report_revision_new": 2444017206,
  "report_revision_old": 2443809810,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
  "value": null,
  "value_current_2026": [
    "Q234213"
  ],
  "value_current_2026_descriptions_en": [
    "extramarital sex without the consent of the married participant's spouse"
  ],
  "value_current_2026_labels_en": [
    "adultery"
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
    "description": "crime a person or organization was convicted of",
    "label": "convicted of"
  },
  "qid": {
    "description": "surgeon (1791–1866)",
    "label": "William Hulke"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21510851",
      "mapped_violation_family": "allowed_qualifier",
      "violation_name": "Allowed qualifiers"
    }
  ],
  "candidate_violation_names": [
    "Allowed qualifiers"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "allowed qualifiers constraint",
  "mapped_violation_constraint_qid": "Q21510851",
  "mapped_violation_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "P1480"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
    "P1480"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": [
    "P2306"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_values": [
    "P1480"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
  "author": "Pallor",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "f768a342feebc74b8fe530ae847548b7708af59e",
  "hash_before": "e44744442fb6d0e5fe651cd7ba7f0363a08d1d25",
  "property_revision_id": 2443999770,
  "property_revision_prev": 2442311007,
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
    "mapped_violation_constraint_qid": "Q21510851",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Allowed qualifiers"
  },
  {
    "result": "Q21510851",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2306"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "P1480"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21510851",
        "mapped_violation_family": "allowed_qualifier",
        "violation_name": "Allowed qualifiers"
      }
    ],
    "candidate_violation_names": [
      "Allowed qualifiers"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "changed_qualifier_properties": [
      "P2306"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "allowed qualifiers constraint",
    "mapped_report_constraint_qid": "Q21510851",
    "mapped_report_family": "allowed_qualifier",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "allowed qualifiers constraint",
    "mapped_violation_constraint_qid": "Q21510851",
    "mapped_violation_family": "allowed_qualifier",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Allowed qualifiers",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "P1480"
    ],
    "semantic_changed_qualifier_properties": [
      "P2306"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "allowed qualifiers constraint",
    "target_constraint_qid": "Q21510851",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "allowed_qualifier",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Allowed qualifiers"
  }
]
```

---

## 011. `reform_Q137425880_P2094_2443310519`

| Field | Value |
|---|---|
| qid | Q137425880 |
| property | P2094 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P2094::2443310519 |
| tbox_revision_key | TBOX::P2094::2443310519 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Mitte27",
  "kind": "T_BOX",
  "property_revision_id": 2443310519,
  "property_revision_prev": 2436452255
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-19T09:10:40",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2094",
  "report_revision_new": 2444010073,
  "report_revision_old": 2443802695,
  "report_violation_type": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "report_violation_type_descriptions_en": [
    "event, during which one or more sporting events are held",
    "organization involved primarily in sports",
    "object used for sport or exercise",
    "classification of age groups, sexes and levels within sports",
    "mobile machine used for transport, whether it has an engine or not, including wheeled and tracked vehicles, air-, water-, and space-craft",
    "specific vehicle design of which all instances are produced to identical specifications",
    "quantified value of an event that is more extreme than that of all comparable events",
    "event in sports",
    "any set of human beings",
    "object (cup, medal...) awarded to a sports winner",
    "sports event scheduled to recur within a decided interval",
    "view of the navigation template of the Wikimedia project",
    "participation of a nation at sports competition",
    "sports practiced within a specific region",
    null
  ],
  "report_violation_type_labels_en": [
    "sports competition",
    "sports organization",
    "sports equipment",
    "classification in sports",
    "vehicle",
    "vehicle model",
    "record",
    "sporting event",
    "group of humans",
    "sports award",
    "recurring sporting event",
    "Wikimedia navigational template for sports team squad",
    "nation at sport competition",
    "sport in a geographic region",
    "список участников спортивного турнира"
  ],
  "report_violation_type_normalized": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "report_violation_type_qids": [
    "Q13406554",
    "Q4438121",
    "Q768186",
    "Q1744559",
    "Q42889",
    "Q29048322",
    "Q1241356",
    "Q16510064",
    "Q16334295",
    "Q15229207",
    "Q18608583",
    "Q107285679",
    "Q46135307",
    "Q29791211",
    "Q137424753"
  ],
  "report_violation_type_raw": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "value": null,
  "value_current_2026": [
    "Q63891772"
  ],
  "value_current_2026_descriptions_en": [
    "competition class in badminton"
  ],
  "value_current_2026_labels_en": [
    "national championship"
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
    "description": "official classification by a regulating body under which the subject (events, teams, participants, or equipment) qualifies for inclusion",
    "label": "competition class"
  },
  "qid": {
    "description": "badminton championships",
    "label": "Badminton at the 2025 Chinese Paralympic Games and Special Olympics – Special Olympics"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
    }
  ],
  "candidate_violation_names": [
    "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
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
    "Q137424753"
  ],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q137424753"
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
  "value_overlap_with_report_qids": [
    "Q137424753"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q137424753"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
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
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "semantic_added_values": [
    "Q137424753"
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
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Mitte27",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "30dbd2dbb202d34107cdea391b1a7a596cbfaf87",
  "hash_before": "e054bcd1226272b2c0a6a49a829d9fde160ba75f",
  "property_revision_id": 2443310519,
  "property_revision_prev": 2436452255,
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
    "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q137424753"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
      }
    ],
    "candidate_violation_names": [
      "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
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
      "Q137424753"
    ],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q137424753"
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
    "value_overlap_with_report_qids": [
      "Q137424753"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
  }
]
```

---

## 012. `reform_Q2337361_P1344_2444117279`

| Field | Value |
|---|---|
| qid | Q2337361 |
| property | P1344 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21510851 allowed qualifiers constraint |
| group_key | TBOX::P1344::2444117279 |
| tbox_revision_key | TBOX::P1344::2444117279 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "allowed qualifiers constraint",
  "decision_constraint_type_qid": "Q21510851"
}
```

#### Repair Target

```json
{
  "author": "Swpb",
  "kind": "T_BOX",
  "property_revision_id": 2444117279,
  "property_revision_prev": 2439333402
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-21T08:46:16",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1344",
  "report_revision_new": 2444860386,
  "report_revision_old": 2444430366,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
  "value": null,
  "value_current_2026": [
    "Q8450",
    "Q8438",
    "Q201971",
    "Q201971",
    "Q8444",
    "Q8531",
    "Q8544",
    "Q8558",
    "Q8577"
  ],
  "value_current_2026_descriptions_en": [
    "Games of the XXII Olympiad, in Moscow, USSR",
    "Games of the XX Olympiad, in Munich, West Germany",
    "international water polo tournament",
    "international water polo tournament",
    "Games of the XXI Olympiad, in Montréal, Canada",
    "Games of the XXVI Olympiad, in Atlanta, USA",
    "Games of the XXVII Olympiad, in Sydney, Australia",
    "Games of the XXVIII Olympiad, in Athens, Greece",
    "Games of the XXX Olympiad, in London, United Kingdom"
  ],
  "value_current_2026_labels_en": [
    "1980 Summer Olympics",
    "1972 Summer Olympics",
    "water polo at the World Aquatics Championships",
    "water polo at the World Aquatics Championships",
    "1976 Summer Olympics",
    "1996 Summer Olympics",
    "2000 Summer Olympics",
    "2004 Summer Olympics",
    "2012 Summer Olympics"
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
    "description": "event in which a person, organization or creative work was/is a participant; inverse of P710 or P1923",
    "label": "participant in"
  },
  "qid": {
    "description": "Russian water polo player and coach (1948—2020)",
    "label": "Alexander Kabanov"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21510851",
      "mapped_violation_family": "allowed_qualifier",
      "violation_name": "Allowed qualifiers"
    }
  ],
  "candidate_violation_names": [
    "Allowed qualifiers"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "allowed qualifiers constraint",
  "mapped_violation_constraint_qid": "Q21510851",
  "mapped_violation_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "P121"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
    "P121"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": [
    "P2306"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_values": [
    "P121"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
  "hash_after": "6545e4945ca6c09cf83514519b18146d39a5f7ce",
  "hash_before": "ca50d932ee6ddd7258eaedaed5c8392f6306bddd",
  "property_revision_id": 2444117279,
  "property_revision_prev": 2439333402,
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
    "mapped_violation_constraint_qid": "Q21510851",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Allowed qualifiers"
  },
  {
    "result": "Q21510851",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2306"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "P121"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21510851",
        "mapped_violation_family": "allowed_qualifier",
        "violation_name": "Allowed qualifiers"
      }
    ],
    "candidate_violation_names": [
      "Allowed qualifiers"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "changed_qualifier_properties": [
      "P2306"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "allowed qualifiers constraint",
    "mapped_report_constraint_qid": "Q21510851",
    "mapped_report_family": "allowed_qualifier",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "allowed qualifiers constraint",
    "mapped_violation_constraint_qid": "Q21510851",
    "mapped_violation_family": "allowed_qualifier",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Allowed qualifiers",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "P121"
    ],
    "semantic_changed_qualifier_properties": [
      "P2306"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "allowed qualifiers constraint",
    "target_constraint_qid": "Q21510851",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "allowed_qualifier",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Allowed qualifiers"
  }
]
```

---

## 013. `reform_Q2543976_P2094_2443310519`

| Field | Value |
|---|---|
| qid | Q2543976 |
| property | P2094 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P2094::2443310519 |
| tbox_revision_key | TBOX::P2094::2443310519 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Mitte27",
  "kind": "T_BOX",
  "property_revision_id": 2443310519,
  "property_revision_prev": 2436452255
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-20T08:30:12",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2094",
  "report_revision_new": 2444422254,
  "report_revision_old": 2444010073,
  "report_violation_type": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "report_violation_type_descriptions_en": [
    "event, during which one or more sporting events are held",
    "organization involved primarily in sports",
    "object used for sport or exercise",
    "classification of age groups, sexes and levels within sports",
    "mobile machine used for transport, whether it has an engine or not, including wheeled and tracked vehicles, air-, water-, and space-craft",
    "specific vehicle design of which all instances are produced to identical specifications",
    "quantified value of an event that is more extreme than that of all comparable events",
    "event in sports",
    "any set of human beings",
    "object (cup, medal...) awarded to a sports winner",
    "sports event scheduled to recur within a decided interval",
    "view of the navigation template of the Wikimedia project",
    "participation of a nation at sports competition",
    "sports practiced within a specific region",
    null
  ],
  "report_violation_type_labels_en": [
    "sports competition",
    "sports organization",
    "sports equipment",
    "classification in sports",
    "vehicle",
    "vehicle model",
    "record",
    "sporting event",
    "group of humans",
    "sports award",
    "recurring sporting event",
    "Wikimedia navigational template for sports team squad",
    "nation at sport competition",
    "sport in a geographic region",
    "список участников спортивного турнира"
  ],
  "report_violation_type_normalized": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "report_violation_type_qids": [
    "Q13406554",
    "Q4438121",
    "Q768186",
    "Q1744559",
    "Q42889",
    "Q29048322",
    "Q1241356",
    "Q16510064",
    "Q16334295",
    "Q15229207",
    "Q18608583",
    "Q107285679",
    "Q46135307",
    "Q29791211",
    "Q137424753"
  ],
  "report_violation_type_raw": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "value": null,
  "value_current_2026": [
    "Q31930761"
  ],
  "value_current_2026_descriptions_en": [
    "association football when played by men"
  ],
  "value_current_2026_labels_en": [
    "men's association football"
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
    "description": "official classification by a regulating body under which the subject (events, teams, participants, or equipment) qualifies for inclusion",
    "label": "competition class"
  },
  "qid": {
    "description": "جائزة رياضية",
    "label": "SFWA Footballer of the Year"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
    }
  ],
  "candidate_violation_names": [
    "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
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
    "Q137424753"
  ],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q137424753"
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
  "value_overlap_with_report_qids": [
    "Q137424753"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q137424753"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
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
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "semantic_added_values": [
    "Q137424753"
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
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Mitte27",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "30dbd2dbb202d34107cdea391b1a7a596cbfaf87",
  "hash_before": "e054bcd1226272b2c0a6a49a829d9fde160ba75f",
  "property_revision_id": 2443310519,
  "property_revision_prev": 2436452255,
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
    "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q137424753"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
      }
    ],
    "candidate_violation_names": [
      "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
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
      "Q137424753"
    ],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q137424753"
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
    "value_overlap_with_report_qids": [
      "Q137424753"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
  }
]
```

---

## 014. `reform_Q28439711_P2094_2443310519`

| Field | Value |
|---|---|
| qid | Q28439711 |
| property | P2094 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P2094::2443310519 |
| tbox_revision_key | TBOX::P2094::2443310519 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Mitte27",
  "kind": "T_BOX",
  "property_revision_id": 2443310519,
  "property_revision_prev": 2436452255
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-18T16:41:13",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2094",
  "report_revision_new": 2443802695,
  "report_revision_old": 2443355035,
  "report_violation_type": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "report_violation_type_descriptions_en": [
    "event, during which one or more sporting events are held",
    "organization involved primarily in sports",
    "object used for sport or exercise",
    "classification of age groups, sexes and levels within sports",
    "mobile machine used for transport, whether it has an engine or not, including wheeled and tracked vehicles, air-, water-, and space-craft",
    "specific vehicle design of which all instances are produced to identical specifications",
    "quantified value of an event that is more extreme than that of all comparable events",
    "event in sports",
    "any set of human beings",
    "object (cup, medal...) awarded to a sports winner",
    "sports event scheduled to recur within a decided interval",
    "view of the navigation template of the Wikimedia project",
    "participation of a nation at sports competition",
    "sports practiced within a specific region",
    null
  ],
  "report_violation_type_labels_en": [
    "sports competition",
    "sports organization",
    "sports equipment",
    "classification in sports",
    "vehicle",
    "vehicle model",
    "record",
    "sporting event",
    "group of humans",
    "sports award",
    "recurring sporting event",
    "Wikimedia navigational template for sports team squad",
    "nation at sport competition",
    "sport in a geographic region",
    "список участников спортивного турнира"
  ],
  "report_violation_type_normalized": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "report_violation_type_qids": [
    "Q13406554",
    "Q4438121",
    "Q768186",
    "Q1744559",
    "Q42889",
    "Q29048322",
    "Q1241356",
    "Q16510064",
    "Q16334295",
    "Q15229207",
    "Q18608583",
    "Q107285679",
    "Q46135307",
    "Q29791211",
    "Q137424753"
  ],
  "report_violation_type_raw": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "value": null,
  "value_current_2026": [
    "Q8031140"
  ],
  "value_current_2026_descriptions_en": [
    "cricket when played by girls/women"
  ],
  "value_current_2026_labels_en": [
    "women's cricket"
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
    "description": "official classification by a regulating body under which the subject (events, teams, participants, or equipment) qualifies for inclusion",
    "label": "competition class"
  },
  "qid": {
    "description": "Wikimedia list article",
    "label": "2000 Women's Cricket World Cup squads"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
    }
  ],
  "candidate_violation_names": [
    "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
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
    "Q137424753"
  ],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q137424753"
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
  "value_overlap_with_report_qids": [
    "Q137424753"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q137424753"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
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
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "semantic_added_values": [
    "Q137424753"
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
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Mitte27",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "30dbd2dbb202d34107cdea391b1a7a596cbfaf87",
  "hash_before": "e054bcd1226272b2c0a6a49a829d9fde160ba75f",
  "property_revision_id": 2443310519,
  "property_revision_prev": 2436452255,
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
    "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q137424753"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
      }
    ],
    "candidate_violation_names": [
      "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
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
      "Q137424753"
    ],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q137424753"
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
    "value_overlap_with_report_qids": [
      "Q137424753"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
  }
]
```

---

## 015. `reform_Q3045248_P1399_2443999770`

| Field | Value |
|---|---|
| qid | Q3045248 |
| property | P1399 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | head |
| constraint_family | Q21510865 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21510851 allowed qualifiers constraint |
| group_key | TBOX::P1399::2443999770 |
| tbox_revision_key | TBOX::P1399::2443999770 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "allowed qualifiers constraint",
  "decision_constraint_type_qid": "Q21510851"
}
```

#### Repair Target

```json
{
  "author": "Pallor",
  "kind": "T_BOX",
  "property_revision_id": 2443999770,
  "property_revision_prev": 2442311007
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-19T09:39:49",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1399",
  "report_revision_new": 2444017206,
  "report_revision_old": 2443809810,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
  "value": null,
  "value_current_2026": [
    "Q69996547"
  ],
  "value_current_2026_descriptions_en": [
    "the charge/crime of criminally negligent manslaughter in the United States"
  ],
  "value_current_2026_labels_en": [
    "criminally negligent homicide"
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
    "description": "crime a person or organization was convicted of",
    "label": "convicted of"
  },
  "qid": {
    "description": "American songwriter and visual artist",
    "label": "Gail Collins"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21510851",
      "mapped_violation_family": "allowed_qualifier",
      "violation_name": "Allowed qualifiers"
    }
  ],
  "candidate_violation_names": [
    "Allowed qualifiers"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "allowed qualifiers constraint",
  "mapped_violation_constraint_qid": "Q21510851",
  "mapped_violation_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "P1480"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
    "P1480"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": [
    "P2306"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_values": [
    "P1480"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
  "author": "Pallor",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "f768a342feebc74b8fe530ae847548b7708af59e",
  "hash_before": "e44744442fb6d0e5fe651cd7ba7f0363a08d1d25",
  "property_revision_id": 2443999770,
  "property_revision_prev": 2442311007,
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
    "mapped_violation_constraint_qid": "Q21510851",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Allowed qualifiers"
  },
  {
    "result": "Q21510851",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2306"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "P1480"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21510851",
        "mapped_violation_family": "allowed_qualifier",
        "violation_name": "Allowed qualifiers"
      }
    ],
    "candidate_violation_names": [
      "Allowed qualifiers"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "changed_qualifier_properties": [
      "P2306"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "allowed qualifiers constraint",
    "mapped_report_constraint_qid": "Q21510851",
    "mapped_report_family": "allowed_qualifier",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "allowed qualifiers constraint",
    "mapped_violation_constraint_qid": "Q21510851",
    "mapped_violation_family": "allowed_qualifier",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Allowed qualifiers",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "P1480"
    ],
    "semantic_changed_qualifier_properties": [
      "P2306"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "allowed qualifiers constraint",
    "target_constraint_qid": "Q21510851",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "allowed_qualifier",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Allowed qualifiers"
  }
]
```

---

## 016. `reform_Q38276179_P123_2442705670`

| Field | Value |
|---|---|
| qid | Q38276179 |
| property | P123 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
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
  "report_fix_date": "2025-12-18T20:39:48",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P123",
  "report_revision_new": 2443864425,
  "report_revision_old": 2443439013,
  "report_violation_type": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274",
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
    "... omitted 3 items"
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
    "... omitted 3 items"
  ],
  "report_violation_type_normalized": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274",
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
    "... omitted 3 items"
  ],
  "report_violation_type_raw": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274",
  "value": null,
  "value_current_2026": [
    "Q391534"
  ],
  "value_current_2026_descriptions_en": [
    "Japanese confectionery company"
  ],
  "value_current_2026_labels_en": [
    "Ezaki Glico"
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
    "description": "江崎グリコのアイスクリーム",
    "label": "牧場しぼり"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274"
    }
  ],
  "candidate_violation_names": [
    "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274"
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
    "Q8274"
  ],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274",
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
  "value_overlap_with_report_qids": [
    "Q8274"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q8274"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
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
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274",
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
    "causality_match_level": "exact_constraint_and_value_match",
    "mapped_violation_constraint_qid": "Q21503250",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q8274"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274"
      }
    ],
    "candidate_violation_names": [
      "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274"
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
      "Q8274"
    ],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274",
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
    "value_overlap_with_report_qids": [
      "Q8274"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274"
  }
]
```

---

## 017. `reform_Q413471_P2877_2438975304`

| Field | Value |
|---|---|
| qid | Q413471 |
| property | P2877 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q19474404 single-value constraint |
| group_key | TBOX::P2877::2438975304 |
| tbox_revision_key | TBOX::P2877::2438975304 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

```json
{
  "author": "Wostr",
  "kind": "T_BOX",
  "property_revision_id": 2438975304,
  "property_revision_prev": 2433431025
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-09T08:13:49",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2439916240,
  "report_revision_old": 2439508509,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": null,
  "value_current_2026": [
    "22049",
    "22050"
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
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "inorganic chemical compound",
    "label": "titanium tetrachloride"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
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
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q19474404",
      "mapped_violation_family": "single_value",
      "violation_name": "Single value"
    }
  ],
  "candidate_violation_names": [
    "Single value"
  ],
  "causality_match_level": "exact_constraint_and_value_match",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q19474404"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q19474404"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
  "compatible_overlap_used": true,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Single value",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "P1013"
  ],
  "semantic_changed_qualifier_properties": [
    "P4155"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
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
  "added_value_count": 1,
  "added_values": [
    "P1013"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "changed_constraint_qids_all": [
    "Q19474404"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q19474404"
  ],
  "changed_qualifier_properties": [
    "P4155"
  ],
  "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
  "compatible_overlap_used": true,
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "single-value constraint",
  "mapped_report_constraint_qid": "Q19474404",
  "mapped_report_family": "single_value",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Single value",
  "semantic_added_values": [
    "P1013"
  ],
  "semantic_changed_qualifier_properties": [
    "P4155"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
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
  "author": "Wostr",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "f5405b49b65b7934fd9e137ef8f147af068f2bf5",
  "hash_before": "32c7b5c7444e03defeaf44d5a32dc6f0a79c3f86",
  "property_revision_id": 2438975304,
  "property_revision_prev": 2433431025,
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
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P4155"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "P1013"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q19474404",
        "mapped_violation_family": "single_value",
        "violation_name": "Single value"
      }
    ],
    "candidate_violation_names": [
      "Single value"
    ],
    "causality_match_level": "exact_constraint_and_value_match",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q19474404"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q19474404"
    ],
    "changed_qualifier_properties": [
      "P4155"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
    "compatible_overlap_used": true,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P4155"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Single value",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "P1013"
    ],
    "semantic_changed_qualifier_properties": [
      "P4155"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
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

## 018. `reform_Q45319423_P1346_1577277439`

| Field | Value |
|---|---|
| qid | Q45319423 |
| property | P1346 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
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
  "report_fix_date": "2022-02-19T11:39:16",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1346",
  "report_revision_new": 1579704875,
  "report_revision_old": 1579145383,
  "report_violation_type": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724, Q|1656682",
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
    "intellectual or artistic creation",
    "temporary and scheduled happening, like a conference, festival, competition or similar"
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
    "work",
    "event"
  ],
  "report_violation_type_normalized": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724, Q|1656682",
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
    "Q386724",
    "Q1656682"
  ],
  "report_violation_type_raw": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724, Q|1656682",
  "value": null,
  "value_current_2026": [
    "Q28402361"
  ],
  "value_current_2026_descriptions_en": [
    "American esports organization"
  ],
  "value_current_2026_labels_en": [
    "Chaos Esports Club"
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
    "description": "esports championship tournament",
    "label": "ESL One Genting 2017"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21510865",
      "mapped_violation_family": "value_type",
      "violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724, Q|1656682"
    }
  ],
  "candidate_violation_names": [
    "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724, Q|1656682"
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
    "Q1656682"
  ],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724, Q|1656682",
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
  "value_overlap_with_report_qids": [
    "Q1656682"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q1656682"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
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
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724, Q|1656682",
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
  "value_specific_without_overlap": false
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
    "causality_match_level": "exact_constraint_and_value_match",
    "mapped_violation_constraint_qid": "Q21510865",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724, Q|1656682"
  },
  {
    "result": "Q21510865",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q1656682"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21510865",
        "mapped_violation_family": "value_type",
        "violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724, Q|1656682"
      }
    ],
    "candidate_violation_names": [
      "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724, Q|1656682"
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
      "Q1656682"
    ],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724, Q|1656682",
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
    "value_overlap_with_report_qids": [
      "Q1656682"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724, Q|1656682"
  }
]
```

---

## 019. `reform_Q63504997_P175_2289317152`

| Field | Value |
|---|---|
| qid | Q63504997 |
| property | P175 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21510851 allowed qualifiers constraint |
| group_key | TBOX::P175::2289317152 |
| tbox_revision_key | TBOX::P175::2289317152 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "allowed qualifiers constraint",
  "decision_constraint_type_qid": "Q21510851"
}
```

#### Repair Target

```json
{
  "author": "Trade",
  "kind": "T_BOX",
  "property_revision_id": 2289317152,
  "property_revision_prev": 2281043870
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-12-24T09:50:40",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P175",
  "report_revision_new": 2290976877,
  "report_revision_old": 2290718782,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
  "value": null,
  "value_current_2026": [
    "Q14917975",
    "Q44479972"
  ],
  "value_current_2026_descriptions_en": [
    "French rapper of Algerian descent",
    "American rapper from New York (born 1996)"
  ],
  "value_current_2026_labels_en": [
    "Lacrim",
    "6ix9ine"
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
    "description": "actor, musician, band or other performer associated with this role or musical work",
    "label": "performer"
  },
  "qid": {
    "description": "2019 song by Lacrim",
    "label": "Bloody"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21510851",
      "mapped_violation_family": "allowed_qualifier",
      "violation_name": "Allowed qualifiers"
    }
  ],
  "candidate_violation_names": [
    "Allowed qualifiers"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "allowed qualifiers constraint",
  "mapped_violation_constraint_qid": "Q21510851",
  "mapped_violation_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "P13187"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
    "P13187"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": [
    "P2306"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_values": [
    "P13187"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
  "hash_after": "3dc455e8a8adf7fb4510c94a4c5b1a4e913b8cd7",
  "hash_before": "6ec6445c098411b1515d783b722ab338b09679d7",
  "property_revision_id": 2289317152,
  "property_revision_prev": 2281043870,
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
    "mapped_violation_constraint_qid": "Q21510851",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Allowed qualifiers"
  },
  {
    "result": "Q21510851",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2306"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "P13187"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21510851",
        "mapped_violation_family": "allowed_qualifier",
        "violation_name": "Allowed qualifiers"
      }
    ],
    "candidate_violation_names": [
      "Allowed qualifiers"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "changed_qualifier_properties": [
      "P2306"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "allowed qualifiers constraint",
    "mapped_report_constraint_qid": "Q21510851",
    "mapped_report_family": "allowed_qualifier",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "allowed qualifiers constraint",
    "mapped_violation_constraint_qid": "Q21510851",
    "mapped_violation_family": "allowed_qualifier",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Allowed qualifiers",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "P13187"
    ],
    "semantic_changed_qualifier_properties": [
      "P2306"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "allowed qualifiers constraint",
    "target_constraint_qid": "Q21510851",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "allowed_qualifier",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Allowed qualifiers"
  }
]
```

---

## 020. `reform_Q6596715_P2094_2443310519`

| Field | Value |
|---|---|
| qid | Q6596715 |
| property | P2094 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P2094::2443310519 |
| tbox_revision_key | TBOX::P2094::2443310519 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Mitte27",
  "kind": "T_BOX",
  "property_revision_id": 2443310519,
  "property_revision_prev": 2436452255
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-20T08:30:12",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2094",
  "report_revision_new": 2444422254,
  "report_revision_old": 2444010073,
  "report_violation_type": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "report_violation_type_descriptions_en": [
    "event, during which one or more sporting events are held",
    "organization involved primarily in sports",
    "object used for sport or exercise",
    "classification of age groups, sexes and levels within sports",
    "mobile machine used for transport, whether it has an engine or not, including wheeled and tracked vehicles, air-, water-, and space-craft",
    "specific vehicle design of which all instances are produced to identical specifications",
    "quantified value of an event that is more extreme than that of all comparable events",
    "event in sports",
    "any set of human beings",
    "object (cup, medal...) awarded to a sports winner",
    "sports event scheduled to recur within a decided interval",
    "view of the navigation template of the Wikimedia project",
    "participation of a nation at sports competition",
    "sports practiced within a specific region",
    null
  ],
  "report_violation_type_labels_en": [
    "sports competition",
    "sports organization",
    "sports equipment",
    "classification in sports",
    "vehicle",
    "vehicle model",
    "record",
    "sporting event",
    "group of humans",
    "sports award",
    "recurring sporting event",
    "Wikimedia navigational template for sports team squad",
    "nation at sport competition",
    "sport in a geographic region",
    "список участников спортивного турнира"
  ],
  "report_violation_type_normalized": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "report_violation_type_qids": [
    "Q13406554",
    "Q4438121",
    "Q768186",
    "Q1744559",
    "Q42889",
    "Q29048322",
    "Q1241356",
    "Q16510064",
    "Q16334295",
    "Q15229207",
    "Q18608583",
    "Q107285679",
    "Q46135307",
    "Q29791211",
    "Q137424753"
  ],
  "report_violation_type_raw": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "value": null,
  "value_current_2026": [
    "Q2887217"
  ],
  "value_current_2026_descriptions_en": [
    "basketball played by women"
  ],
  "value_current_2026_labels_en": [
    "women's basketball"
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
    "description": "official classification by a regulating body under which the subject (events, teams, participants, or equipment) qualifies for inclusion",
    "label": "competition class"
  },
  "qid": {
    "description": "page de liste de Wikipédia",
    "label": "List of Senior CLASS Award women's basketball winners"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
    }
  ],
  "candidate_violation_names": [
    "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
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
    "Q137424753"
  ],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q137424753"
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
  "value_overlap_with_report_qids": [
    "Q137424753"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q137424753"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
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
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
  "semantic_added_values": [
    "Q137424753"
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
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Mitte27",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "30dbd2dbb202d34107cdea391b1a7a596cbfaf87",
  "hash_before": "e054bcd1226272b2c0a6a49a829d9fde160ba75f",
  "property_revision_id": 2443310519,
  "property_revision_prev": 2436452255,
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
    "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q137424753"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
      }
    ],
    "candidate_violation_names": [
      "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
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
      "Q137424753"
    ],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q137424753"
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
    "value_overlap_with_report_qids": [
      "Q137424753"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|13406554, Q|4438121, Q|768186, Q|1744559, Q|42889, Q|29048322, Q|1241356, Q|16510064, Q|16334295, Q|15229207, Q|18608583, Q|107285679, Q|46135307, Q|29791211, Q|137424753"
  }
]
```

---

## 021. `reform_Q6663_P5905_2440998983`

| Field | Value |
|---|---|
| qid | Q6663 |
| property | P5905 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P5905::2440998983 |
| tbox_revision_key | TBOX::P5905::2440998983 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Jwillikers",
  "kind": "T_BOX",
  "property_revision_id": 2440998983,
  "property_revision_prev": 2433975755
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-18T14:57:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5905",
  "report_revision_new": 2443776066,
  "report_revision_old": 2443324880,
  "report_violation_type": "Type Q|1760610, Q|11424, Q|15416, Q|21191270, Q|5, Q|14897293, Q|8274, Q|14406742, Q|747381, Q|18706315, Q|618123, Q|16334295, Q|151885, Q|19958368, Q|125632018, Q|3331189",
  "report_violation_type_descriptions_en": [
    "publication type, book of comic art",
    "sequence of images that give the impression of movement, stored on film stock",
    "segment of audiovisual content intended for broadcast and streaming on television",
    "single installment of a television series",
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "entity that only exists in a work of fiction",
    "comics employing a set of Japanese stylistic conventions, produced in Japan or elsewhere",
    "group of published comic stories typically sharing the same title",
    "Japanese novella-type storytelling in conjunction with illustrations, geared toward young adults",
    "entity whose existence is possible, but not proven",
    "components of planets that can be geographically located",
    "any set of human beings",
    "semantic unit understood in different ways, e.g. as mental representation, ability or abstract object (philosophy)",
    "distinct human activities and values that take place in, or originate from, a geographic location",
    "physical or digital compilation of chapters from a manga series",
    "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])"
  ],
  "report_violation_type_labels_en": [
    "comic book",
    "film",
    "television program",
    "television series episode",
    "human",
    "fictional entity",
    "manga",
    "comic book series",
    "light novel",
    "hypothetical entity",
    "geographical feature",
    "group of humans",
    "concept",
    "culture of an area",
    "manga volume",
    "version, edition or translation"
  ],
  "report_violation_type_normalized": "Type Q|1760610, Q|11424, Q|15416, Q|21191270, Q|5, Q|14897293, Q|8274, Q|14406742, Q|747381, Q|18706315, Q|618123, Q|16334295, Q|151885, Q|19958368, Q|125632018, Q|3331189",
  "report_violation_type_qids": [
    "Q1760610",
    "Q11424",
    "Q15416",
    "Q21191270",
    "Q5",
    "Q14897293",
    "Q8274",
    "Q14406742",
    "Q747381",
    "Q18706315",
    "Q618123",
    "Q16334295",
    "Q151885",
    "Q19958368",
    "Q125632018",
    "Q3331189"
  ],
  "report_violation_type_raw": "Type Q|1760610, Q|11424, Q|15416, Q|21191270, Q|5, Q|14897293, Q|8274, Q|14406742, Q|747381, Q|18706315, Q|618123, Q|16334295, Q|151885, Q|19958368, Q|125632018, Q|3331189",
  "value": null,
  "value_current_2026": [
    "4055-56620"
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
    "description": "identifier at the Comic Vine database of comic books, fictional characters, people, films and television series/episodes",
    "label": "Comic Vine ID"
  },
  "qid": {
    "description": "American sandwich of ground beef patty",
    "label": "hamburger"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|1760610, Q|11424, Q|15416, Q|21191270, Q|5, Q|14897293, Q|8274, Q|14406742, Q|747381, Q|18706315, Q|618123, Q|16334295, Q|151885, Q|19958368, Q|125632018, Q|3331189"
    }
  ],
  "candidate_violation_names": [
    "Type Q|1760610, Q|11424, Q|15416, Q|21191270, Q|5, Q|14897293, Q|8274, Q|14406742, Q|747381, Q|18706315, Q|618123, Q|16334295, Q|151885, Q|19958368, Q|125632018, Q|3331189"
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
    "Q3331189"
  ],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|1760610, Q|11424, Q|15416, Q|21191270, Q|5, Q|14897293, Q|8274, Q|14406742, Q|747381, Q|18706315, Q|618123, Q|16334295, Q|151885, Q|19958368, Q|125632018, Q|3331189",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q3331189"
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
  "value_overlap_with_report_qids": [
    "Q3331189"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q3331189"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
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
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Type Q|1760610, Q|11424, Q|15416, Q|21191270, Q|5, Q|14897293, Q|8274, Q|14406742, Q|747381, Q|18706315, Q|618123, Q|16334295, Q|151885, Q|19958368, Q|125632018, Q|3331189",
  "semantic_added_values": [
    "Q3331189"
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
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Jwillikers",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "0f6bbfcb8d6915242c5e4cd11e4521035c08e909",
  "hash_before": "c52ec5d0754d5fa76cff87e79a211d857b472ce3",
  "property_revision_id": 2440998983,
  "property_revision_prev": 2433975755,
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
    "violation_name": "Type Q|1760610, Q|11424, Q|15416, Q|21191270, Q|5, Q|14897293, Q|8274, Q|14406742, Q|747381, Q|18706315, Q|618123, Q|16334295, Q|151885, Q|19958368, Q|125632018, Q|3331189"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q3331189"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|1760610, Q|11424, Q|15416, Q|21191270, Q|5, Q|14897293, Q|8274, Q|14406742, Q|747381, Q|18706315, Q|618123, Q|16334295, Q|151885, Q|19958368, Q|125632018, Q|3331189"
      }
    ],
    "candidate_violation_names": [
      "Type Q|1760610, Q|11424, Q|15416, Q|21191270, Q|5, Q|14897293, Q|8274, Q|14406742, Q|747381, Q|18706315, Q|618123, Q|16334295, Q|151885, Q|19958368, Q|125632018, Q|3331189"
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
      "Q3331189"
    ],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|1760610, Q|11424, Q|15416, Q|21191270, Q|5, Q|14897293, Q|8274, Q|14406742, Q|747381, Q|18706315, Q|618123, Q|16334295, Q|151885, Q|19958368, Q|125632018, Q|3331189",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q3331189"
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
    "value_overlap_with_report_qids": [
      "Q3331189"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|1760610, Q|11424, Q|15416, Q|21191270, Q|5, Q|14897293, Q|8274, Q|14406742, Q|747381, Q|18706315, Q|618123, Q|16334295, Q|151885, Q|19958368, Q|125632018, Q|3331189"
  }
]
```

---

## 022. `reform_Q68457257_P123_2442705670`

| Field | Value |
|---|---|
| qid | Q68457257 |
| property | P123 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
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
  "report_fix_date": "2025-12-19T12:11:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P123",
  "report_revision_new": 2444049316,
  "report_revision_old": 2443864425,
  "report_violation_type": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274",
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
    "... omitted 3 items"
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
    "... omitted 3 items"
  ],
  "report_violation_type_normalized": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274",
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
    "... omitted 3 items"
  ],
  "report_violation_type_raw": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274",
  "value": null,
  "value_current_2026": [
    "Q22497966"
  ],
  "value_current_2026_descriptions_en": [
    "Belarusian publishing house"
  ],
  "value_current_2026_labels_en": [
    "Januskevic"
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
    "label": "Урб@н.М. Адзін дзень не майго жыцця"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274"
    }
  ],
  "candidate_violation_names": [
    "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274"
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
    "Q8274"
  ],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274",
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
  "value_overlap_with_report_qids": [
    "Q8274"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q8274"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
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
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274",
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
    "causality_match_level": "exact_constraint_and_value_match",
    "mapped_violation_constraint_qid": "Q21503250",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q8274"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274"
      }
    ],
    "candidate_violation_names": [
      "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274"
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
      "Q8274"
    ],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274",
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
    "value_overlap_with_report_qids": [
      "Q8274"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645, Q|8274"
  }
]
```

---

## 023. `reform_Q7043025_P121_2442147265`

| Field | Value |
|---|---|
| qid | Q7043025 |
| property | P121 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21510865 value-type constraint |
| group_key | TBOX::P121::2442147265 |
| tbox_revision_key | TBOX::P121::2442147265 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "value-type constraint",
  "decision_constraint_type_qid": "Q21510865"
}
```

#### Repair Target

```json
{
  "author": "LBLaiSiNanHai",
  "kind": "T_BOX",
  "property_revision_id": 2442147265,
  "property_revision_prev": 2431827995
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-16T13:38:25",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P121",
  "report_revision_new": 2443005059,
  "report_revision_old": 2442697572,
  "report_violation_type": "Value type Q|2897903, Q|2668072, Q|3150005, Q|13226383, Q|8513, Q|19541, Q|11660, Q|431289, Q|39546, Q|1322323, Q|340169, Q|728937, Q|1055307, Q|994895, Q|47311934, Q|61974339, Q|12973014, Q|476028, Q|623109, Q|14296, Q|507619, Q|11019, Q|3918, Q|778575, Q|14635346, Q|14897293, Q|170584, Q|18643213, Q|3604202, Q|63981612",
  "report_violation_type_descriptions_en": [
    "outcome of human efforts to meet the wants and needs of people",
    "set of purposely gathered physical or digital objects with some common characteristics",
    "formally executed written document",
    "place, equipment, or service to support a specific function",
    "organized collection of data in computing",
    "system to help searching for information",
    "field of computer science that develops and studies software enabling machines to exhibit intelligent behavior",
    "identification for a good or service",
    "physical item that can be used to achieve a goal",
    "route of a journey",
    "storage and delivery agent of information or data",
    "constructional unit in rail transport, the route or way of rail tracks between defined locations",
    "technology existing only in fictional works",
    "system created through engineering",
    "group of vehicles operated by an organization",
    "individual team that competes in professional gaming tournaments",
    "individual team that plays sports",
    "sports club devoted to association football (soccer)",
    "group of sports teams or individual athletes that compete against each other",
    "domain at the highest level of the DNS hierarchy",
    "retail outlets that share a brand and central management, and usually have standardized business methods and practices",
    "tool containing one or more parts that uses energy to perform an intended action",
    "academic institution for further education",
    "large company involved in many industries",
    "... omitted 6 items"
  ],
  "report_violation_type_labels_en": [
    "goods and services",
    "collection",
    "legal instrument",
    "facility",
    "database",
    "search engine",
    "artificial intelligence",
    "brand",
    "physical tool",
    "itinerary",
    "communications media",
    "railway line",
    "fictional technology",
    "technical system",
    "fleet",
    "esports team",
    "sports team",
    "association football club",
    "sports league",
    "top-level domain",
    "retail chain",
    "machine",
    "university",
    "conglomerate",
    "... omitted 6 items"
  ],
  "report_violation_type_normalized": "Value type Q|2897903, Q|2668072, Q|3150005, Q|13226383, Q|8513, Q|19541, Q|11660, Q|431289, Q|39546, Q|1322323, Q|340169, Q|728937, Q|1055307, Q|994895, Q|47311934, Q|61974339, Q|12973014, Q|476028, Q|623109, Q|14296, Q|507619, Q|11019, Q|3918, Q|778575, Q|14635346, Q|14897293, Q|170584, Q|18643213, Q|3604202, Q|63981612",
  "report_violation_type_qids": [
    "Q2897903",
    "Q2668072",
    "Q3150005",
    "Q13226383",
    "Q8513",
    "Q19541",
    "Q11660",
    "Q431289",
    "Q39546",
    "Q1322323",
    "Q340169",
    "Q728937",
    "Q1055307",
    "Q994895",
    "Q47311934",
    "Q61974339",
    "Q12973014",
    "Q476028",
    "Q623109",
    "Q14296",
    "Q507619",
    "Q11019",
    "Q3918",
    "Q778575",
    "... omitted 6 items"
  ],
  "report_violation_type_raw": "Value type Q|2897903, Q|2668072, Q|3150005, Q|13226383, Q|8513, Q|19541, Q|11660, Q|431289, Q|39546, Q|1322323, Q|340169, Q|728937, Q|1055307, Q|994895, Q|47311934, Q|61974339, Q|12973014, Q|476028, Q|623109, Q|14296, Q|507619, Q|11019, Q|3918, Q|778575, Q|14635346, Q|14897293, Q|170584, Q|18643213, Q|3604202, Q|63981612",
  "value": null,
  "value_current_2026": [
    "Q880762",
    "Q1860313",
    "Q694521",
    "Q887246",
    "Q850179",
    "Q799582",
    "Q1373850",
    "Q771742",
    "Q4037086"
  ],
  "value_current_2026_descriptions_en": [
    "carrier-based torpedo bomber aircraft",
    "Dutch light reconnaissance and bomber biplane aircraft",
    "1926 airliner family",
    "airplane",
    "1934 fighter aircraft family by Gloster",
    "fighter",
    "1936 army cooperation aircraft family by Westland",
    "Military plane",
    "light utility aircraft"
  ],
  "value_current_2026_labels_en": [
    "Blackburn Ripon",
    "Fokker C.V",
    "Junkers W 34",
    "Fokker C.X",
    "Gloster Gladiator",
    "Fokker D.X",
    "Lysander",
    "VL Myrsky",
    "Moth"
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
    "description": "equipment, installation or service operated by the subject",
    "label": "item operated"
  },
  "qid": {
    "description": "1939–1944 Finnish Air Force reconnaissance squadron",
    "label": "Squadron 16"
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
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q21510865",
      "mapped_violation_family": "value_type",
      "violation_name": "Value type Q|2897903, Q|2668072, Q|3150005, Q|13226383, Q|8513, Q|19541, Q|11660, Q|431289, Q|39546, Q|1322323, Q|340169, Q|728937, Q|1055307, Q|994895, Q|47311934, Q|61974339, Q|12973014, Q|476028, Q|623109, Q|14296, Q|507619, Q|11019, Q|3918, Q|778575, Q|14635346, Q|14897293, Q|170584, Q|18643213, Q|3604202, Q|63981612"
    }
  ],
  "candidate_violation_names": [
    "Value type Q|2897903, Q|2668072, Q|3150005, Q|13226383, Q|8513, Q|19541, Q|11660, Q|431289, Q|39546, Q|1322323, Q|340169, Q|728937, Q|1055307, Q|994895, Q|47311934, Q|61974339, Q|12973014, Q|476028, Q|623109, Q|14296, Q|507619, Q|11019, Q|3918, Q|778575, Q|14635346, Q|14897293, Q|170584, Q|18643213, Q|3604202, Q|63981612"
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
    "Q63981612"
  ],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Value type Q|2897903, Q|2668072, Q|3150005, Q|13226383, Q|8513, Q|19541, Q|11660, Q|431289, Q|39546, Q|1322323, Q|340169, Q|728937, Q|1055307, Q|994895, Q|47311934, Q|61974339, Q|12973014, Q|476028, Q|623109, Q|14296, Q|507619, Q|11019, Q|3918, Q|778575, Q|14635346, Q|14897293, Q|170584, Q|18643213, Q|3604202, Q|63981612",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "Q63981612"
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
  "value_overlap_with_report_qids": [
    "Q63981612"
  ],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": 1,
  "added_values": [
    "Q63981612"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
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
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Value type Q|2897903, Q|2668072, Q|3150005, Q|13226383, Q|8513, Q|19541, Q|11660, Q|431289, Q|39546, Q|1322323, Q|340169, Q|728937, Q|1055307, Q|994895, Q|47311934, Q|61974339, Q|12973014, Q|476028, Q|623109, Q|14296, Q|507619, Q|11019, Q|3918, Q|778575, Q|14635346, Q|14897293, Q|170584, Q|18643213, Q|3604202, Q|63981612",
  "semantic_added_values": [
    "Q63981612"
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
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "LBLaiSiNanHai",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "91f4dae117a3fb5af4ade635095e90c4ffc4a7b8",
  "hash_before": "341a5a0a3b32c6863ad6fb34bc0f624b66c821a1",
  "property_revision_id": 2442147265,
  "property_revision_prev": 2431827995,
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
    "violation_name": "Value type Q|2897903, Q|2668072, Q|3150005, Q|13226383, Q|8513, Q|19541, Q|11660, Q|431289, Q|39546, Q|1322323, Q|340169, Q|728937, Q|1055307, Q|994895, Q|47311934, Q|61974339, Q|12973014, Q|476028, Q|623109, Q|14296, Q|507619, Q|11019, Q|3918, Q|778575, Q|14635346, Q|14897293, Q|170584, Q|18643213, Q|3604202, Q|63981612"
  },
  {
    "result": "Q21510865",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2308"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "Q63981612"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q21510865",
        "mapped_violation_family": "value_type",
        "violation_name": "Value type Q|2897903, Q|2668072, Q|3150005, Q|13226383, Q|8513, Q|19541, Q|11660, Q|431289, Q|39546, Q|1322323, Q|340169, Q|728937, Q|1055307, Q|994895, Q|47311934, Q|61974339, Q|12973014, Q|476028, Q|623109, Q|14296, Q|507619, Q|11019, Q|3918, Q|778575, Q|14635346, Q|14897293, Q|170584, Q|18643213, Q|3604202, Q|63981612"
      }
    ],
    "candidate_violation_names": [
      "Value type Q|2897903, Q|2668072, Q|3150005, Q|13226383, Q|8513, Q|19541, Q|11660, Q|431289, Q|39546, Q|1322323, Q|340169, Q|728937, Q|1055307, Q|994895, Q|47311934, Q|61974339, Q|12973014, Q|476028, Q|623109, Q|14296, Q|507619, Q|11019, Q|3918, Q|778575, Q|14635346, Q|14897293, Q|170584, Q|18643213, Q|3604202, Q|63981612"
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
      "Q63981612"
    ],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Value type Q|2897903, Q|2668072, Q|3150005, Q|13226383, Q|8513, Q|19541, Q|11660, Q|431289, Q|39546, Q|1322323, Q|340169, Q|728937, Q|1055307, Q|994895, Q|47311934, Q|61974339, Q|12973014, Q|476028, Q|623109, Q|14296, Q|507619, Q|11019, Q|3918, Q|778575, Q|14635346, Q|14897293, Q|170584, Q|18643213, Q|3604202, Q|63981612",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "Q63981612"
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
    "value_overlap_with_report_qids": [
      "Q63981612"
    ],
    "value_specific_without_overlap": false,
    "violation_name": "Value type Q|2897903, Q|2668072, Q|3150005, Q|13226383, Q|8513, Q|19541, Q|11660, Q|431289, Q|39546, Q|1322323, Q|340169, Q|728937, Q|1055307, Q|994895, Q|47311934, Q|61974339, Q|12973014, Q|476028, Q|623109, Q|14296, Q|507619, Q|11019, Q|3918, Q|778575, Q|14635346, Q|14897293, Q|170584, Q|18643213, Q|3604202, Q|63981612"
  }
]
```

---

## 024. `reform_Q80729951_P1019_2443162511`

| Field | Value |
|---|---|
| qid | Q80729951 |
| property | P1019 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q21510851 allowed qualifiers constraint |
| group_key | TBOX::P1019::2443162511 |
| tbox_revision_key | TBOX::P1019::2443162511 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "allowed qualifiers constraint",
  "decision_constraint_type_qid": "Q21510851"
}
```

#### Repair Target

```json
{
  "author": "Wd-Ryan",
  "kind": "T_BOX",
  "property_revision_id": 2443162511,
  "property_revision_prev": 2439607694
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-17T10:52:25",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1019",
  "report_revision_new": 2443373363,
  "report_revision_old": 2442960955,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
  "value": null,
  "value_current_2026": [
    "https://media.ccc.de/news.atom",
    "https://media.ccc.de/podcast-audio-only.xml",
    "https://media.ccc.de/podcast-hq.xml",
    "https://media.ccc.de/podcast-lq.xml",
    "https://media.ccc.de/updates.rdf",
    "https://media.ccc.de/podcast-archive-hq.xml"
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
    "description": "news feed (RSS, Atom, etc.) of this person/organisation/project",
    "label": "web feed URL"
  },
  "qid": {
    "description": "Video Streaming Portal of the Chaos Computer Club",
    "label": "media.ccc.de"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 30,
      "mapped_violation_constraint_qid": "Q21510851",
      "mapped_violation_family": "allowed_qualifier",
      "violation_name": "Allowed qualifiers"
    }
  ],
  "candidate_violation_names": [
    "Allowed qualifiers"
  ],
  "causality_match_level": "exact_constraint_family_only",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "mapped_violation_confidence": "high",
  "mapped_violation_constraint_label": "allowed qualifiers constraint",
  "mapped_violation_constraint_qid": "Q21510851",
  "mapped_violation_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "P3831"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
    "P3831"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": [
    "P2306"
  ],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "allowed qualifiers constraint",
  "mapped_report_constraint_qid": "Q21510851",
  "mapped_report_family": "allowed_qualifier",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Allowed qualifiers",
  "semantic_added_values": [
    "P3831"
  ],
  "semantic_changed_qualifier_properties": [
    "P2306"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
  "set_semantics": "allowed",
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "allowed qualifiers constraint",
  "target_constraint_qid": "Q21510851",
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
  "author": "Wd-Ryan",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "1a7aa391bc68f3dfcf99a2319241c899e388295e",
  "hash_before": "8e5c45c7db053a4a60485408fdfc60e548adabfe",
  "property_revision_id": 2443162511,
  "property_revision_prev": 2439607694,
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
    "mapped_violation_constraint_qid": "Q21510851",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Allowed qualifiers"
  },
  {
    "result": "Q21510851",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P2306"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "P3831"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 30,
        "mapped_violation_constraint_qid": "Q21510851",
        "mapped_violation_family": "allowed_qualifier",
        "violation_name": "Allowed qualifiers"
      }
    ],
    "candidate_violation_names": [
      "Allowed qualifiers"
    ],
    "causality_match_level": "exact_constraint_family_only",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q21510851"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21510851"
    ],
    "changed_qualifier_properties": [
      "P2306"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "ignored_added_values": [],
    "ignored_changed_qualifier_properties": [],
    "ignored_removed_values": [],
    "ignored_value_count": 0,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "allowed qualifiers constraint",
    "mapped_report_constraint_qid": "Q21510851",
    "mapped_report_family": "allowed_qualifier",
    "mapped_violation_confidence": "high",
    "mapped_violation_constraint_label": "allowed qualifiers constraint",
    "mapped_violation_constraint_qid": "Q21510851",
    "mapped_violation_family": "allowed_qualifier",
    "mapped_violation_reason": "exact_violation_type_mapping",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P2306"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Allowed qualifiers",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "P3831"
    ],
    "semantic_changed_qualifier_properties": [
      "P2306"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "allowed qualifiers constraint",
    "target_constraint_qid": "Q21510851",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "allowed_qualifier",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Allowed qualifiers"
  }
]
```

---

## 025. `reform_Q97279443_P373_2443872580`

| Field | Value |
|---|---|
| qid | Q97279443 |
| property | P373 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | relaxation_set_expansion |
| decision_constraint_type | Q19474404 single-value constraint |
| group_key | TBOX::P373::2443872580 |
| tbox_revision_key | TBOX::P373::2443872580 |

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
| rationale | Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint. |
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
  "classification_rule_subfamily": "relaxation_set_expansion",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

```json
{
  "author": "Clemens Dulcis",
  "kind": "T_BOX",
  "property_revision_id": 2443872580,
  "property_revision_prev": 2442604045
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-20T10:59:36",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2444464305,
  "report_revision_old": 2444037630,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": null,
  "value_current_2026": [
    "Grolla d'oro for Best Best Direction",
    "Grolla d'oro for Best Direction"
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
    "description": "name of the Wikimedia Commons category containing files related to this item (without the prefix \"Category:\")",
    "label": "Commons category"
  },
  "qid": {
    "description": "category of the Italian film award (1953–1983, 1990–2001)",
    "label": "Grolla d'oro for Best Direction"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
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
    "label_en": "Commons link constraint",
    "qid": "Q21510852"
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility. Use active `directional_subtype_precise` for polarity-specific analysis only when the final subtype is directional; `potential_directional_*` fields are debugging context for non-directional schema updates._

```json
{
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_and_value_match",
      "candidate_score": 40,
      "mapped_violation_constraint_qid": "Q19474404",
      "mapped_violation_family": "single_value",
      "violation_name": "Single value"
    }
  ],
  "candidate_violation_names": [
    "Single value"
  ],
  "causality_match_level": "exact_constraint_and_value_match",
  "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
  "changed_constraint_qids_all": [
    "Q19474404"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q19474404"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
  "compatible_overlap_used": true,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "directional_subtype_basis": "allowed set expansion",
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Single value",
  "semantic_added_value_count": 1,
  "semantic_added_values": [
    "P1545"
  ],
  "semantic_changed_qualifier_properties": [
    "P4155"
  ],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "set_operation": "expansion",
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
  "added_value_count": 1,
  "added_values": [
    "P1545"
  ],
  "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
  "changed_constraint_qids_all": [
    "Q19474404"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q19474404"
  ],
  "changed_qualifier_properties": [
    "P4155"
  ],
  "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
  "compatible_overlap_used": true,
  "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
  "ignored_added_values": [],
  "ignored_changed_qualifier_properties": [],
  "ignored_removed_values": [],
  "ignored_value_count": 0,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "single-value constraint",
  "mapped_report_constraint_qid": "Q19474404",
  "mapped_report_family": "single_value",
  "polarity": "relaxation",
  "polarity_basis": "allowed set gained values",
  "potential_directional_subtype_basis": null,
  "potential_directional_subtype_precise": null,
  "potential_polarity": null,
  "potential_polarity_basis": null,
  "potential_set_operation": null,
  "potential_set_semantics": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
  "removed_value_count": 0,
  "removed_values": [],
  "selected_violation_name": "Single value",
  "semantic_added_values": [
    "P1545"
  ],
  "semantic_changed_qualifier_properties": [
    "P4155"
  ],
  "semantic_removed_values": [],
  "set_operation": "expansion",
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
  "author": "Clemens Dulcis",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "e5ff6f21e501dc868e71379e3a3627bbca239e66",
  "hash_before": "847cc606259b0433da724dc728d5a1a47ae1647c",
  "property_revision_id": 2443872580,
  "property_revision_prev": 2442604045,
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
    "added_value_count": 1,
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_ids": [
      "P4155"
    ],
    "removed_value_count": 0,
    "result": "RELAXATION_SET_EXPANSION",
    "set_operation": "expansion",
    "set_semantics": "allowed",
    "step": "set_semantics"
  },
  {
    "added_value_count": 1,
    "added_values": [
      "P1545"
    ],
    "analysis_slice_precise": "main_tbox_relaxation_allowed_set_expansion",
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_and_value_match",
        "candidate_score": 40,
        "mapped_violation_constraint_qid": "Q19474404",
        "mapped_violation_family": "single_value",
        "violation_name": "Single value"
      }
    ],
    "candidate_violation_names": [
      "Single value"
    ],
    "causality_match_level": "exact_constraint_and_value_match",
    "causality_match_reason": "mapped constraint family and compatible changed values support the violation report",
    "changed_constraint_qids_all": [
      "Q19474404"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q19474404"
    ],
    "changed_qualifier_properties": [
      "P4155"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "family_specific_semantic_qualifier_changed",
    "compatible_overlap_used": true,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "directional_subtype_basis": "allowed set expansion",
    "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
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
    "polarity": "relaxation",
    "polarity_basis": "allowed set gained values",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_semantic_for_family",
    "relevant_qualifier_properties": [
      "P4155"
    ],
    "removed_value_count": 0,
    "removed_values": [],
    "result": "RELAXATION_SET_EXPANSION",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Single value",
    "semantic_added_value_count": 1,
    "semantic_added_values": [
      "P1545"
    ],
    "semantic_changed_qualifier_properties": [
      "P4155"
    ],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "set_operation": "expansion",
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
