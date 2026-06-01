# TBOX_UNKNOWN_TBOX_CAUSALITY

Cases: 30

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `reform_Q110738289_P166_2254793027`

| Field | Value |
|---|---|
| qid | Q110738289 |
| property | P166 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | mid |
| constraint_family | Q21510851 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P166::2254793027 |
| tbox_revision_key | TBOX::P166::2254793027 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 2254793027,
  "property_revision_prev": 2254791750
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-10-01T12:00:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P166",
  "report_revision_new": 2255073635,
  "report_revision_old": 2254683010,
  "report_violation_type": "empty",
  "report_violation_type_normalized": "empty",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "empty",
  "report_violation_types": [
    "empty",
    "Item P|31"
  ],
  "value": null,
  "value_current_2026": [
    "Q478850",
    "Q104520671"
  ],
  "value_current_2026_descriptions_en": [
    "order of the Soviet Union",
    null
  ],
  "value_current_2026_labels_en": [
    "Order of the Red Banner of Labour",
    "Awards of Kazan"
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
    "description": "award or recognition received by a person, organization or creative work",
    "label": "award received"
  },
  "qid": {
    "description": null,
    "label": "Radioelectronika (company)"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|31"
    },
    {
      "candidate_causality_match_level": "unmapped_violation",
      "candidate_score": 0,
      "mapped_violation_constraint_qid": null,
      "mapped_violation_family": "unknown",
      "violation_name": "empty"
    }
  ],
  "candidate_violation_names": [
    "empty",
    "Item P|31"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|31",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|31",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "dd3141826e6c4ed30fee5d8ea1843f38a994a620",
  "hash_before": "74a153b04c2c54f8592801c6187afd44d4d41016",
  "property_revision_id": 2254793027,
  "property_revision_prev": 2254791750,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|31"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|31"
      },
      {
        "candidate_causality_match_level": "unmapped_violation",
        "candidate_score": 0,
        "mapped_violation_constraint_qid": null,
        "mapped_violation_family": "unknown",
        "violation_name": "empty"
      }
    ],
    "candidate_violation_names": [
      "empty",
      "Item P|31"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q29934200"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|31",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|31"
  }
]
```

---

## 002. `reform_Q11148810_P400_2356390580`

| Field | Value |
|---|---|
| qid | Q11148810 |
| property | P400 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | head |
| constraint_family | Q21510865 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type |   |
| group_key | TBOX::P400::2356390580 |
| tbox_revision_key | TBOX::P400::2356390580 |

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
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "author": "Trade",
  "kind": "T_BOX",
  "property_revision_id": 2356390580,
  "property_revision_prev": 2356390454
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-06-03T10:25:59",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P400",
  "report_revision_new": 2356424542,
  "report_revision_old": 2356023718,
  "report_violation_type": "Q|7188",
  "report_violation_type_descriptions_en": [
    "system or group of people governing an organized community, often a state"
  ],
  "report_violation_type_labels_en": [
    "government"
  ],
  "report_violation_type_normalized": "Q|7188",
  "report_violation_type_qids": [
    "Q7188"
  ],
  "report_violation_type_raw": "Q|7188",
  "value": null,
  "value_current_2026": [
    "Q48493"
  ],
  "value_current_2026_descriptions_en": [
    "mobile operating system by Apple Inc."
  ],
  "value_current_2026_labels_en": [
    "iOS"
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
    "description": "government of Zhejiang",
    "label": "People's Government of Zhejiang Province"
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

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "unmapped_violation",
      "candidate_score": 0,
      "mapped_violation_constraint_qid": null,
      "mapped_violation_family": "unknown",
      "violation_name": "Q|7188"
    }
  ],
  "candidate_violation_names": [
    "Q|7188"
  ],
  "causality_match_level": "unmapped_violation",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
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
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "mapped_violation_confidence": "none",
  "mapped_violation_constraint_label": null,
  "mapped_violation_constraint_qid": null,
  "mapped_violation_family": "unknown",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Q|7188",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q52558054"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q52558054"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Q|7188",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
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
  "hash_after": "4c688d9e9946021f54ff2d42e6a2a03a85f4936d",
  "hash_before": "2a9a8192bbb1e51c571b94c0ccda6b8a1c18bda9",
  "property_revision_id": 2356390580,
  "property_revision_prev": 2356390454,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "unmapped_violation",
    "mapped_violation_constraint_qid": null,
    "result": false,
    "step": "causality_filter",
    "violation_name": "Q|7188"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "unmapped_violation",
        "candidate_score": 0,
        "mapped_violation_constraint_qid": null,
        "mapped_violation_family": "unknown",
        "violation_name": "Q|7188"
      }
    ],
    "candidate_violation_names": [
      "Q|7188"
    ],
    "causality_match_level": "unmapped_violation",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
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
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": null,
    "mapped_report_constraint_qid": null,
    "mapped_report_family": "unknown",
    "mapped_violation_confidence": "none",
    "mapped_violation_constraint_label": null,
    "mapped_violation_constraint_qid": null,
    "mapped_violation_family": "unknown",
    "mapped_violation_reason": "unmapped_violation_type",
    "property_overlap_with_report_pids": [],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Q|7188",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Q|7188"
  }
]
```

---

## 003. `reform_Q114664650_P9899_2316258676`

| Field | Value |
|---|---|
| qid | Q114664650 |
| property | P9899 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P9899::2316258676 |
| tbox_revision_key | TBOX::P9899::2316258676 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "Kefr4000",
  "kind": "T_BOX",
  "property_revision_id": 2316258676,
  "property_revision_prev": 2316258527
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-02-25T05:54:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P9899",
  "report_revision_new": 2316632014,
  "report_revision_old": 2316203965,
  "report_violation_type": "Item P|1476",
  "report_violation_type_normalized": "Item P|1476",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|1476",
  "value": null,
  "value_current_2026": [
    "Q11263801"
  ],
  "value_current_2026_descriptions_en": [
    "2000 Indian film by Neeraj Vora"
  ],
  "value_current_2026_labels_en": [
    "Khiladi 420"
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
    "description": "a work, event, etc. for which a musical composition was created (e.g., a play for which incidental music was composed; a ballet for which ballet music was written; a film for which motion picture music was created)",
    "label": "music created for"
  },
  "qid": {
    "description": null,
    "label": "Meri Biwi Ka Jawab Nahin"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|1476"
    }
  ],
  "candidate_violation_names": [
    "Item P|1476"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q30208840"
  ],
  "ignored_changed_qualifier_properties": [
    "P2309",
    "P5314"
  ],
  "ignored_removed_values": [
    "Q30208840"
  ],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|1476",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q30208840"
  ],
  "ignored_changed_qualifier_properties": [
    "P2309",
    "P5314"
  ],
  "ignored_removed_values": [
    "Q30208840"
  ],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|1476",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "Kefr4000",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "74346a7b032275583625d99042483d15be936f4d",
  "hash_before": "ec86bf0473d200b413988a0fa5458a14df6cff6c",
  "property_revision_id": 2316258676,
  "property_revision_prev": 2316258527,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|1476"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|1476"
      }
    ],
    "candidate_violation_names": [
      "Item P|1476"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q30208840"
    ],
    "ignored_changed_qualifier_properties": [
      "P2309",
      "P5314"
    ],
    "ignored_removed_values": [
      "Q30208840"
    ],
    "ignored_value_count": 2,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|1476",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|1476"
  }
]
```

---

## 004. `reform_Q114669459_P9899_2316258676`

| Field | Value |
|---|---|
| qid | Q114669459 |
| property | P9899 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P9899::2316258676 |
| tbox_revision_key | TBOX::P9899::2316258676 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "Kefr4000",
  "kind": "T_BOX",
  "property_revision_id": 2316258676,
  "property_revision_prev": 2316258527
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-02-25T05:54:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P9899",
  "report_revision_new": 2316632014,
  "report_revision_old": 2316203965,
  "report_violation_type": "Item P|1476",
  "report_violation_type_normalized": "Item P|1476",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|1476",
  "value": null,
  "value_current_2026": [
    "Q7144913"
  ],
  "value_current_2026_descriptions_en": [
    "2011 film by Nikhil Advani"
  ],
  "value_current_2026_labels_en": [
    "Patiala House"
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
    "description": "a work, event, etc. for which a musical composition was created (e.g., a play for which incidental music was composed; a ballet for which ballet music was written; a film for which motion picture music was created)",
    "label": "music created for"
  },
  "qid": {
    "description": null,
    "label": "Rola Pe Gaya (orig & remix)"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|1476"
    }
  ],
  "candidate_violation_names": [
    "Item P|1476"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q30208840"
  ],
  "ignored_changed_qualifier_properties": [
    "P2309",
    "P5314"
  ],
  "ignored_removed_values": [
    "Q30208840"
  ],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|1476",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q30208840"
  ],
  "ignored_changed_qualifier_properties": [
    "P2309",
    "P5314"
  ],
  "ignored_removed_values": [
    "Q30208840"
  ],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|1476",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "Kefr4000",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "74346a7b032275583625d99042483d15be936f4d",
  "hash_before": "ec86bf0473d200b413988a0fa5458a14df6cff6c",
  "property_revision_id": 2316258676,
  "property_revision_prev": 2316258527,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|1476"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|1476"
      }
    ],
    "candidate_violation_names": [
      "Item P|1476"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q30208840"
    ],
    "ignored_changed_qualifier_properties": [
      "P2309",
      "P5314"
    ],
    "ignored_removed_values": [
      "Q30208840"
    ],
    "ignored_value_count": 2,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|1476",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|1476"
  }
]
```

---

## 005. `reform_Q114705356_P9899_2316258676`

| Field | Value |
|---|---|
| qid | Q114705356 |
| property | P9899 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P9899::2316258676 |
| tbox_revision_key | TBOX::P9899::2316258676 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "Kefr4000",
  "kind": "T_BOX",
  "property_revision_id": 2316258676,
  "property_revision_prev": 2316258527
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-02-25T05:54:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P9899",
  "report_revision_new": 2316632014,
  "report_revision_old": 2316203965,
  "report_violation_type": "Item P|1476",
  "report_violation_type_normalized": "Item P|1476",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|1476",
  "value": null,
  "value_current_2026": [
    "Q21646665"
  ],
  "value_current_2026_descriptions_en": [
    "2016 film"
  ],
  "value_current_2026_labels_en": [
    "Loveshhuda"
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
    "description": "a work, event, etc. for which a musical composition was created (e.g., a play for which incidental music was composed; a ballet for which ballet music was written; a film for which motion picture music was created)",
    "label": "music created for"
  },
  "qid": {
    "description": null,
    "label": "Mar Jaayen (Reprise)"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|1476"
    }
  ],
  "candidate_violation_names": [
    "Item P|1476"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q30208840"
  ],
  "ignored_changed_qualifier_properties": [
    "P2309",
    "P5314"
  ],
  "ignored_removed_values": [
    "Q30208840"
  ],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|1476",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q30208840"
  ],
  "ignored_changed_qualifier_properties": [
    "P2309",
    "P5314"
  ],
  "ignored_removed_values": [
    "Q30208840"
  ],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|1476",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "Kefr4000",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "74346a7b032275583625d99042483d15be936f4d",
  "hash_before": "ec86bf0473d200b413988a0fa5458a14df6cff6c",
  "property_revision_id": 2316258676,
  "property_revision_prev": 2316258527,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|1476"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|1476"
      }
    ],
    "candidate_violation_names": [
      "Item P|1476"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q30208840"
    ],
    "ignored_changed_qualifier_properties": [
      "P2309",
      "P5314"
    ],
    "ignored_removed_values": [
      "Q30208840"
    ],
    "ignored_value_count": 2,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|1476",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|1476"
  }
]
```

---

## 006. `reform_Q114835835_P9899_2316258676`

| Field | Value |
|---|---|
| qid | Q114835835 |
| property | P9899 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P9899::2316258676 |
| tbox_revision_key | TBOX::P9899::2316258676 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "Kefr4000",
  "kind": "T_BOX",
  "property_revision_id": 2316258676,
  "property_revision_prev": 2316258527
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-02-25T05:54:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P9899",
  "report_revision_new": 2316632014,
  "report_revision_old": 2316203965,
  "report_violation_type": "Item P|1476",
  "report_violation_type_normalized": "Item P|1476",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|1476",
  "value": null,
  "value_current_2026": [
    "Q11016537"
  ],
  "value_current_2026_descriptions_en": [
    "1983 film"
  ],
  "value_current_2026_labels_en": [
    "Daulat Ke Dushman"
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
    "description": "a work, event, etc. for which a musical composition was created (e.g., a play for which incidental music was composed; a ballet for which ballet music was written; a film for which motion picture music was created)",
    "label": "music created for"
  },
  "qid": {
    "description": null,
    "label": "Jo Bhi Hua Hai"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|1476"
    }
  ],
  "candidate_violation_names": [
    "Item P|1476"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q30208840"
  ],
  "ignored_changed_qualifier_properties": [
    "P2309",
    "P5314"
  ],
  "ignored_removed_values": [
    "Q30208840"
  ],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|1476",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q30208840"
  ],
  "ignored_changed_qualifier_properties": [
    "P2309",
    "P5314"
  ],
  "ignored_removed_values": [
    "Q30208840"
  ],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|1476",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "Kefr4000",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "74346a7b032275583625d99042483d15be936f4d",
  "hash_before": "ec86bf0473d200b413988a0fa5458a14df6cff6c",
  "property_revision_id": 2316258676,
  "property_revision_prev": 2316258527,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|1476"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|1476"
      }
    ],
    "candidate_violation_names": [
      "Item P|1476"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q30208840"
    ],
    "ignored_changed_qualifier_properties": [
      "P2309",
      "P5314"
    ],
    "ignored_removed_values": [
      "Q30208840"
    ],
    "ignored_value_count": 2,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|1476",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|1476"
  }
]
```

---

## 007. `reform_Q118802327_P175_2292837202`

| Field | Value |
|---|---|
| qid | Q118802327 |
| property | P175 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21510865 value-type constraint |
| group_key | TBOX::P175::2292837202 |
| tbox_revision_key | TBOX::P175::2292837202 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "value-type constraint",
  "decision_constraint_type_qid": "Q21510865"
}
```

#### Repair Target

```json
{
  "author": "Trade",
  "kind": "T_BOX",
  "property_revision_id": 2292837202,
  "property_revision_prev": 2289317152
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-12-30T10:56:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P175",
  "report_revision_new": 2292897440,
  "report_revision_old": 2292473966,
  "report_violation_type": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "group of people who perform instrumental and/or vocal music, with the ensemble typically known by a distinct name",
    "human being that only exists in fictional works",
    "session musician or backing band participating on a recording but whose name is not on the cover",
    "musical ensemble which performs music",
    "musical ensemble that exists only in fiction",
    "fictional character who appears in animated films, television, and other animated works",
    "fictional character who appears in a television series",
    "type of artist credit used by MusicBrainz when the original creator of a musical work is unknown, anonymous, or in the public domain",
    "any group of artists working together in the field of comedy",
    "organized group that plays theatre",
    "group of people performing in a theatrical production, motion picture, television programme or musical composition",
    "artificial intelligence that only exists in a work of fiction",
    "character appearing in digital media that acts and looks like real people",
    "set of fictional characters",
    "individual team that plays sports",
    "group of dancers",
    "business that finds work for show business people",
    "hobbyist club in Japan",
    "singular named exemplar of an animal (e.g., the gorilla named Koko; the cat named Socks)",
    "kingdom of multicellular eukaryotic organisms",
    "fictional character appearing in a video game"
  ],
  "report_violation_type_labels_en": [
    "human",
    "musical ensemble",
    "fictional human",
    "uncredited musical artist",
    "musical group",
    "fictional musical ensemble",
    "animated character",
    "television character",
    "special purpose artist",
    "comedy troupe",
    "theatre troupe",
    "cast",
    "fictional artificial intelligence",
    "virtual character",
    "group of fictional characters",
    "sports team",
    "dance troupe",
    "talent agency",
    "dōjin circle",
    "individual animal",
    "Animalia",
    "video game character"
  ],
  "report_violation_type_normalized": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167",
  "report_violation_type_qids": [
    "Q5",
    "Q2088357",
    "Q15632617",
    "Q60614352",
    "Q215380",
    "Q6619719",
    "Q15711870",
    "Q15773317",
    "Q59755569",
    "Q18510489",
    "Q2416217",
    "Q15267437",
    "Q66481339",
    "Q65209857",
    "Q14514600",
    "Q12973014",
    "Q2393314",
    "Q5354754",
    "Q2013644",
    "Q26401003",
    "Q729",
    "Q1569167"
  ],
  "report_violation_type_raw": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167",
  "value": null,
  "value_current_2026": [
    "Q118784631",
    "Q118785927"
  ],
  "value_current_2026_descriptions_en": [
    null,
    null
  ],
  "value_current_2026_labels_en": [
    "Adn Del Ánima",
    "Mudit Grau"
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
    "description": null,
    "label": "l’adn de l’ ànima"
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

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21510865",
      "mapped_violation_family": "value_type",
      "violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167"
    }
  ],
  "candidate_violation_names": [
    "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167"
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
  "ignored_added_values": [
    "Q84310035"
  ],
  "ignored_changed_qualifier_properties": [
    "P2305"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "value-type constraint",
  "mapped_violation_constraint_qid": "Q21510865",
  "mapped_violation_family": "value_type",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
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
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q84310035"
  ],
  "ignored_changed_qualifier_properties": [
    "P2305"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
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
  "author": "Trade",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "4057f3a67f654cae6ddfc690d91087b6901d88d9",
  "hash_before": "3dc455e8a8adf7fb4510c94a4c5b1a4e913b8cd7",
  "property_revision_id": 2292837202,
  "property_revision_prev": 2289317152,
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
    "violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167"
  },
  {
    "result": "Q21510865",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21510865",
        "mapped_violation_family": "value_type",
        "violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167"
      }
    ],
    "candidate_violation_names": [
      "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167"
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
    "ignored_added_values": [
      "Q84310035"
    ],
    "ignored_changed_qualifier_properties": [
      "P2305"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
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
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
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
    "violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167"
  }
]
```

---

## 008. `reform_Q11948144_P1296_1648181364`

| Field | Value |
|---|---|
| qid | Q11948144 |
| property | P1296 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P1296::1648181364 |
| tbox_revision_key | TBOX::P1296::1648181364 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "Nikki",
  "kind": "T_BOX",
  "property_revision_id": 1648181364,
  "property_revision_prev": 1638585077
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-06-01T10:55:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1296",
  "report_revision_new": 1651853720,
  "report_revision_old": 1647499267,
  "report_violation_type": "Type Q|35120",
  "report_violation_type_descriptions_en": [
    "anything that can be considered, discussed, or observed"
  ],
  "report_violation_type_labels_en": [
    "entity"
  ],
  "report_violation_type_normalized": "Type Q|35120",
  "report_violation_type_qids": [
    "Q35120"
  ],
  "report_violation_type_raw": "Type Q|35120",
  "value": null,
  "value_current_2026": [
    "0141643"
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
    "description": "identifier for an item in the Gran Enciclopèdia Catalana. Replaced with \"Gran Enciclopèdia Catalana ID (P12385)\"",
    "label": "Gran Enciclopèdia Catalana ID (former scheme)"
  },
  "qid": {
    "description": null,
    "label": "Semicaducifoli"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "lexeme requires language constraint",
    "qid": "Q55819106"
  },
  {
    "label_en": "label in language constraint",
    "qid": "Q108139345"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|35120"
    }
  ],
  "candidate_violation_names": [
    "Type Q|35120"
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
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|35120",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
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
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Type Q|35120",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
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
  "author": "Nikki",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "48ae3a4b24989e2f2e751e655c874dceb0e12c5f",
  "hash_before": "02755eb4bf2c3dd6a77ae6165cb9c4894a779a45",
  "property_revision_id": 1648181364,
  "property_revision_prev": 1638585077,
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
    "violation_name": "Type Q|35120"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|35120"
      }
    ],
    "candidate_violation_names": [
      "Type Q|35120"
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
    "ignored_added_values": [
      "Q29934200"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
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
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|35120",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
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
    "violation_name": "Type Q|35120"
  }
]
```

---

## 009. `reform_Q123474656_P9899_2316258676`

| Field | Value |
|---|---|
| qid | Q123474656 |
| property | P9899 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P9899::2316258676 |
| tbox_revision_key | TBOX::P9899::2316258676 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "Kefr4000",
  "kind": "T_BOX",
  "property_revision_id": 2316258676,
  "property_revision_prev": 2316258527
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-02-25T05:54:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P9899",
  "report_revision_new": 2316632014,
  "report_revision_old": 2316203965,
  "report_violation_type": "Item P|1476",
  "report_violation_type_normalized": "Item P|1476",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|1476",
  "value": null,
  "value_current_2026": [
    "Q7907820"
  ],
  "value_current_2026_descriptions_en": [
    "2015 Tamil film directed by Vijayachander"
  ],
  "value_current_2026_labels_en": [
    "Vaalu"
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
    "description": "a work, event, etc. for which a musical composition was created (e.g., a play for which incidental music was composed; a ballet for which ballet music was written; a film for which motion picture music was created)",
    "label": "music created for"
  },
  "qid": {
    "description": "Tamil song from Vaalu movie",
    "label": "You're My Darling"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|1476"
    }
  ],
  "candidate_violation_names": [
    "Item P|1476"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q30208840"
  ],
  "ignored_changed_qualifier_properties": [
    "P2309",
    "P5314"
  ],
  "ignored_removed_values": [
    "Q30208840"
  ],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|1476",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q30208840"
  ],
  "ignored_changed_qualifier_properties": [
    "P2309",
    "P5314"
  ],
  "ignored_removed_values": [
    "Q30208840"
  ],
  "ignored_value_count": 2,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|1476",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "Kefr4000",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "74346a7b032275583625d99042483d15be936f4d",
  "hash_before": "ec86bf0473d200b413988a0fa5458a14df6cff6c",
  "property_revision_id": 2316258676,
  "property_revision_prev": 2316258527,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|1476"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|1476"
      }
    ],
    "candidate_violation_names": [
      "Item P|1476"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q30208840"
    ],
    "ignored_changed_qualifier_properties": [
      "P2309",
      "P5314"
    ],
    "ignored_removed_values": [
      "Q30208840"
    ],
    "ignored_value_count": 2,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|1476",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|1476"
  }
]
```

---

## 010. `reform_Q16253469_P175_2292837202`

| Field | Value |
|---|---|
| qid | Q16253469 |
| property | P175 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21510865 value-type constraint |
| group_key | TBOX::P175::2292837202 |
| tbox_revision_key | TBOX::P175::2292837202 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "value-type constraint",
  "decision_constraint_type_qid": "Q21510865"
}
```

#### Repair Target

```json
{
  "author": "Trade",
  "kind": "T_BOX",
  "property_revision_id": 2292837202,
  "property_revision_prev": 2289317152
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-12-30T10:56:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P175",
  "report_revision_new": 2292897440,
  "report_revision_old": 2292473966,
  "report_violation_type": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "group of people who perform instrumental and/or vocal music, with the ensemble typically known by a distinct name",
    "human being that only exists in fictional works",
    "session musician or backing band participating on a recording but whose name is not on the cover",
    "musical ensemble which performs music",
    "musical ensemble that exists only in fiction",
    "fictional character who appears in animated films, television, and other animated works",
    "fictional character who appears in a television series",
    "type of artist credit used by MusicBrainz when the original creator of a musical work is unknown, anonymous, or in the public domain",
    "any group of artists working together in the field of comedy",
    "organized group that plays theatre",
    "group of people performing in a theatrical production, motion picture, television programme or musical composition",
    "artificial intelligence that only exists in a work of fiction",
    "character appearing in digital media that acts and looks like real people",
    "set of fictional characters",
    "individual team that plays sports",
    "group of dancers",
    "business that finds work for show business people",
    "hobbyist club in Japan",
    "singular named exemplar of an animal (e.g., the gorilla named Koko; the cat named Socks)",
    "kingdom of multicellular eukaryotic organisms",
    "fictional character appearing in a video game"
  ],
  "report_violation_type_labels_en": [
    "human",
    "musical ensemble",
    "fictional human",
    "uncredited musical artist",
    "musical group",
    "fictional musical ensemble",
    "animated character",
    "television character",
    "special purpose artist",
    "comedy troupe",
    "theatre troupe",
    "cast",
    "fictional artificial intelligence",
    "virtual character",
    "group of fictional characters",
    "sports team",
    "dance troupe",
    "talent agency",
    "dōjin circle",
    "individual animal",
    "Animalia",
    "video game character"
  ],
  "report_violation_type_normalized": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167",
  "report_violation_type_qids": [
    "Q5",
    "Q2088357",
    "Q15632617",
    "Q60614352",
    "Q215380",
    "Q6619719",
    "Q15711870",
    "Q15773317",
    "Q59755569",
    "Q18510489",
    "Q2416217",
    "Q15267437",
    "Q66481339",
    "Q65209857",
    "Q14514600",
    "Q12973014",
    "Q2393314",
    "Q5354754",
    "Q2013644",
    "Q26401003",
    "Q729",
    "Q1569167"
  ],
  "report_violation_type_raw": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167",
  "value": null,
  "value_current_2026": [
    "Q58917"
  ],
  "value_current_2026_descriptions_en": [
    "horizontal entrance shaft to an underground mine"
  ],
  "value_current_2026_labels_en": [
    "adit"
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
    "description": "2014 film by Iftakar Chowdhury",
    "label": "Rajotto"
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

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21510865",
      "mapped_violation_family": "value_type",
      "violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167"
    }
  ],
  "candidate_violation_names": [
    "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167"
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
  "ignored_added_values": [
    "Q84310035"
  ],
  "ignored_changed_qualifier_properties": [
    "P2305"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "value-type constraint",
  "mapped_violation_constraint_qid": "Q21510865",
  "mapped_violation_family": "value_type",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
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
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21510865"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510865"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q84310035"
  ],
  "ignored_changed_qualifier_properties": [
    "P2305"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "value-type constraint",
  "mapped_report_constraint_qid": "Q21510865",
  "mapped_report_family": "value_type",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
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
  "author": "Trade",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "4057f3a67f654cae6ddfc690d91087b6901d88d9",
  "hash_before": "3dc455e8a8adf7fb4510c94a4c5b1a4e913b8cd7",
  "property_revision_id": 2292837202,
  "property_revision_prev": 2289317152,
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
    "violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167"
  },
  {
    "result": "Q21510865",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21510865",
        "mapped_violation_family": "value_type",
        "violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167"
      }
    ],
    "candidate_violation_names": [
      "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167"
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
    "ignored_added_values": [
      "Q84310035"
    ],
    "ignored_changed_qualifier_properties": [
      "P2305"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
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
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
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
    "violation_name": "Value type Q|5, Q|2088357, Q|15632617, Q|60614352, Q|215380, Q|6619719, Q|15711870, Q|15773317, Q|59755569, Q|18510489, Q|2416217, Q|15267437, Q|66481339, Q|65209857, Q|14514600, Q|12973014, Q|2393314, Q|5354754, Q|2013644, Q|26401003, Q|729, Q|1569167"
  }
]
```

---

## 011. `reform_Q178806_P242_1687213130`

| Field | Value |
|---|---|
| qid | Q178806 |
| property | P242 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | head |
| constraint_family | Q21502410 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P242::1687213130 |
| tbox_revision_key | TBOX::P242::1687213130 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 1687213130,
  "property_revision_prev": 1687212271
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-07-25T12:44:07",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P242",
  "report_revision_new": 1687848379,
  "report_revision_old": 1686708951,
  "report_violation_type": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885, Q|7884789",
  "report_violation_type_descriptions_en": [
    "point, line or area on or near Earth",
    "relations between two subjects of public international law",
    "territorial entity for administration purposes, with or without its own local government",
    "organization established by treaty between governments",
    "alliance between different states with the purpose to cooperate militarily",
    "2D or 3D defined space on something, mainly in terrestrial and astrophysics sciences",
    "occurrence of a fact or object in space-time; instantiation of a property in an object",
    "image on the celestial sphere consisting of stars according to any current of historical system or description",
    "group of independent or autonomous territories sharing a given set of traits",
    "structured system of communication",
    "place that exists only in fiction and not in reality",
    "taxonomic rank between family and genus",
    "socially defined category of people who identify with each other",
    "human-designed and -made structure",
    "group of languages related through descent from a common ancestor",
    "semantic unit understood in different ways, e.g. as mental representation, ability or abstract object (philosophy)",
    "proper name of a place"
  ],
  "report_violation_type_labels_en": [
    "geographic location",
    "bilateral relation",
    "administrative territorial entity",
    "international organization",
    "military alliance",
    "region",
    "occurrence",
    "constellation",
    "geopolitical group",
    "language",
    "fictional location",
    "tribe",
    "ethnic group",
    "architectural structure",
    "language family",
    "concept",
    "toponym"
  ],
  "report_violation_type_normalized": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885, Q|7884789",
  "report_violation_type_qids": [
    "Q2221906",
    "Q15221623",
    "Q56061",
    "Q484652",
    "Q1127126",
    "Q82794",
    "Q1190554",
    "Q8928",
    "Q52110228",
    "Q315",
    "Q3895768",
    "Q227936",
    "Q41710",
    "Q811979",
    "Q25295",
    "Q151885",
    "Q7884789"
  ],
  "report_violation_type_raw": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885, Q|7884789",
  "value": null,
  "value_current_2026": [
    "Middle Dutch with dialect distribution.png"
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
    "description": "geographic map image which highlights the location of the subject within some larger entity",
    "label": "locator map image"
  },
  "qid": {
    "description": "collective name of Dutch dialects of the High and Late Middle Ages",
    "label": "Middle Dutch"
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
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "Commons link constraint",
    "qid": "Q21510852"
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
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885, Q|7884789"
    }
  ],
  "candidate_violation_names": [
    "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885, Q|7884789"
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
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885, Q|7884789",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
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
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885, Q|7884789",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
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
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "b10e15e88aead4e1ff900d2102ae20739fac1392",
  "hash_before": "69bc2e802f252f396bfac375980dad4b68acfad6",
  "property_revision_id": 1687213130,
  "property_revision_prev": 1687212271,
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
    "violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885, Q|7884789"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885, Q|7884789"
      }
    ],
    "candidate_violation_names": [
      "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885, Q|7884789"
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
    "ignored_added_values": [
      "Q29934200"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
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
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885, Q|7884789",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
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
    "violation_name": "Type Q|2221906, Q|15221623, Q|56061, Q|484652, Q|1127126, Q|82794, Q|1190554, Q|8928, Q|52110228, Q|315, Q|3895768, Q|227936, Q|41710, Q|811979, Q|25295, Q|151885, Q|7884789"
  }
]
```

---

## 012. `reform_Q1802655_P18_2402814023`

| Field | Value |
|---|---|
| qid | Q1802655 |
| property | P18 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type |   |
| group_key | TBOX::P18::2402814023 |
| tbox_revision_key | TBOX::P18::2402814023 |

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
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "author": "Trade",
  "kind": "T_BOX",
  "property_revision_id": 2402814023,
  "property_revision_prev": 2395987900
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-09-09T14:31:46",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P18",
  "report_revision_new": 2402826781,
  "report_revision_old": 2402323739,
  "report_violation_type": "Q|13406463",
  "report_violation_type_descriptions_en": [
    "page of a Wikimedia project with a list of something"
  ],
  "report_violation_type_labels_en": [
    "Wikimedia list article"
  ],
  "report_violation_type_normalized": "Q|13406463",
  "report_violation_type_qids": [
    "Q13406463"
  ],
  "report_violation_type_raw": "Q|13406463",
  "value": null,
  "value_current_2026": [
    "L47-(08)-Noviand-Beim Linnengraben (1).jpg"
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
    "description": "image of relevant illustration of the subject; if available, also use more specific properties (sample: coat of arms image, locator map, flag image, signature image, logo image, collage image)",
    "label": "image"
  },
  "qid": {
    "description": "Wikimedia list article",
    "label": "state roads in Rheinland-Pfalz"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "Commons link constraint",
    "qid": "Q21510852"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "unmapped_violation",
      "candidate_score": 0,
      "mapped_violation_constraint_qid": null,
      "mapped_violation_family": "unknown",
      "violation_name": "Q|13406463"
    }
  ],
  "candidate_violation_names": [
    "Q|13406463"
  ],
  "causality_match_level": "unmapped_violation",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q52558054"
  ],
  "changed_constraint_qids_from_entries": [
    "Q52558054"
  ],
  "changed_constraint_qids_from_qualifier_changes": [],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "mapped_violation_confidence": "none",
  "mapped_violation_constraint_label": null,
  "mapped_violation_constraint_qid": null,
  "mapped_violation_family": "unknown",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Q|13406463",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q52558054"
  ],
  "changed_constraint_qids_from_entries": [
    "Q52558054"
  ],
  "changed_constraint_qids_from_qualifier_changes": [],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Q|13406463",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
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
  "changed_constraint_types": [
    "Q52558054"
  ],
  "constraints_readable_en": null,
  "hash_after": "17cae97285ddfcaff28dd6eafd2b6674886a02b7",
  "hash_before": "e3303e74183b70ca25ccbb0c41b3be659021a3b8",
  "property_revision_id": 2402814023,
  "property_revision_prev": 2395987900,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "unmapped_violation",
    "mapped_violation_constraint_qid": null,
    "result": false,
    "step": "causality_filter",
    "violation_name": "Q|13406463"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "unmapped_violation",
        "candidate_score": 0,
        "mapped_violation_constraint_qid": null,
        "mapped_violation_family": "unknown",
        "violation_name": "Q|13406463"
      }
    ],
    "candidate_violation_names": [
      "Q|13406463"
    ],
    "causality_match_level": "unmapped_violation",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q52558054"
    ],
    "changed_constraint_qids_from_entries": [
      "Q52558054"
    ],
    "changed_constraint_qids_from_qualifier_changes": [],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": null,
    "mapped_report_constraint_qid": null,
    "mapped_report_family": "unknown",
    "mapped_violation_confidence": "none",
    "mapped_violation_constraint_label": null,
    "mapped_violation_constraint_qid": null,
    "mapped_violation_family": "unknown",
    "mapped_violation_reason": "unmapped_violation_type",
    "property_overlap_with_report_pids": [],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Q|13406463",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Q|13406463"
  }
]
```

---

## 013. `reform_Q1867263_P485_2386344088`

| Field | Value |
|---|---|
| qid | Q1867263 |
| property | P485 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | mid |
| constraint_family | Q21510865 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type |   |
| group_key | TBOX::P485::2386344088 |
| tbox_revision_key | TBOX::P485::2386344088 |

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
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "author": "Bliekt82",
  "kind": "T_BOX",
  "property_revision_id": 2386344088,
  "property_revision_prev": 2376417160
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-07-31T16:57:30",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P485",
  "report_revision_new": 2386422158,
  "report_revision_old": 2386020128,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": null,
  "value_current_2026": [
    "Q1867263"
  ],
  "value_current_2026_descriptions_en": [
    "Roeselare, West Flanders, Belgium"
  ],
  "value_current_2026_labels_en": [
    "Klein Seminarie Roeselare"
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
    "description": "the institution holding the subject's archives",
    "label": "archives at"
  },
  "qid": {
    "description": "Roeselare, West Flanders, Belgium",
    "label": "Klein Seminarie Roeselare"
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

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "unmapped_violation",
      "candidate_score": 0,
      "mapped_violation_constraint_qid": null,
      "mapped_violation_family": "unknown",
      "violation_name": "Self link"
    }
  ],
  "candidate_violation_names": [
    "Self link"
  ],
  "causality_match_level": "unmapped_violation",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
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
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "mapped_violation_confidence": "none",
  "mapped_violation_constraint_label": null,
  "mapped_violation_constraint_qid": null,
  "mapped_violation_family": "unknown",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Self link",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21510851"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21510851"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Self link",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Bliekt82",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "ba235e52595612023455365c26eecb9a4e401470",
  "hash_before": "99b49c44f1a2bce8859352d427a348e4af498f83",
  "property_revision_id": 2386344088,
  "property_revision_prev": 2376417160,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "unmapped_violation",
    "mapped_violation_constraint_qid": null,
    "result": false,
    "step": "causality_filter",
    "violation_name": "Self link"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "unmapped_violation",
        "candidate_score": 0,
        "mapped_violation_constraint_qid": null,
        "mapped_violation_family": "unknown",
        "violation_name": "Self link"
      }
    ],
    "candidate_violation_names": [
      "Self link"
    ],
    "causality_match_level": "unmapped_violation",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
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
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": null,
    "mapped_report_constraint_qid": null,
    "mapped_report_family": "unknown",
    "mapped_violation_confidence": "none",
    "mapped_violation_constraint_label": null,
    "mapped_violation_constraint_qid": null,
    "mapped_violation_family": "unknown",
    "mapped_violation_reason": "unmapped_violation_type",
    "property_overlap_with_report_pids": [],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Self link",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Self link"
  }
]
```

---

## 014. `reform_Q1929552_P2521_1725922197`

| Field | Value |
|---|---|
| qid | Q1929552 |
| property | P2521 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | head |
| constraint_family | Q53869507 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P2521::1725922197 |
| tbox_revision_key | TBOX::P2521::1725922197 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 1725922197,
  "property_revision_prev": 1719625275
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-09-11T11:10:29",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2521",
  "report_revision_new": 1726203339,
  "report_revision_old": 1725211201,
  "report_violation_type": "Item P|3321",
  "report_violation_type_normalized": "Item P|3321",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|3321",
  "value": null,
  "value_current_2026": [
    "Primaballerina@de",
    "première danseuse@fr"
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
    "description": "female form of name or title (for male use P3321)",
    "label": "female form of label"
  },
  "qid": {
    "description": "dancer at the highest rank within a professional dance company",
    "label": "principal dancer"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
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
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|3321"
    }
  ],
  "candidate_violation_names": [
    "Item P|3321"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|3321",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|3321",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "f0cc45546da8768eeffa00936c0e3a553c2aca4b",
  "hash_before": "42e4e2c51d366fbf19084c4ab0a0759e9306eafa",
  "property_revision_id": 1725922197,
  "property_revision_prev": 1719625275,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|3321"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|3321"
      }
    ],
    "candidate_violation_names": [
      "Item P|3321"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q29934200"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|3321",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|3321"
  }
]
```

---

## 015. `reform_Q21487823_P6597_2040528587`

| Field | Value |
|---|---|
| qid | Q21487823 |
| property | P6597 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P6597::2040528587 |
| tbox_revision_key | TBOX::P6597::2040528587 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "Nikki",
  "kind": "T_BOX",
  "property_revision_id": 2040528587,
  "property_revision_prev": 2040528338
}
```

### Violation Context

```json
{
  "report_fix_date": "2023-12-31T07:00:53",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6597",
  "report_revision_new": 2041211978,
  "report_revision_old": 2040400362,
  "report_violation_type": "Item P|407",
  "report_violation_type_normalized": "Item P|407",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|407",
  "value": null,
  "value_current_2026": [
    "119288"
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
    "description": "ID of corresponding entry in the DFD online dictionary of family names",
    "label": "DFD ID"
  },
  "qid": {
    "description": "family name",
    "label": "Hauge"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "lexeme requires language constraint",
    "qid": "Q55819106"
  },
  {
    "label_en": "label in language constraint",
    "qid": "Q108139345"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|407"
    }
  ],
  "candidate_violation_names": [
    "Item P|407"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|407",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|407",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "Nikki",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "8901b5ad38dd448c8371c1ec9b6a8dc79d75ba9a",
  "hash_before": "077995b0705d0cd9e93a85d257d18baf836c2888",
  "property_revision_id": 2040528587,
  "property_revision_prev": 2040528338,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|407"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|407"
      }
    ],
    "candidate_violation_names": [
      "Item P|407"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q29934200"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|407",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|407"
  }
]
```

---

## 016. `reform_Q23838605_P166_2254793027`

| Field | Value |
|---|---|
| qid | Q23838605 |
| property | P166 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | head |
| constraint_family | Q21510851 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type |   |
| group_key | TBOX::P166::2254793027 |
| tbox_revision_key | TBOX::P166::2254793027 |

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
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 2254793027,
  "property_revision_prev": 2254791750
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-10-01T12:00:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P166",
  "report_revision_new": 2255073635,
  "report_revision_old": 2254683010,
  "report_violation_type": "Q|41176",
  "report_violation_type_descriptions_en": [
    "structure, typically with a roof and walls, standing more or less permanently in one place"
  ],
  "report_violation_type_labels_en": [
    "building"
  ],
  "report_violation_type_normalized": "Q|41176",
  "report_violation_type_qids": [
    "Q41176"
  ],
  "report_violation_type_raw": "Q|41176",
  "value": null,
  "value_current_2026": [
    "Q14557467",
    "Q11753260"
  ],
  "value_current_2026_descriptions_en": [
    "Wikimedia list article",
    "architectural award"
  ],
  "value_current_2026_labels_en": [
    "list of World Architecture Festival winners",
    "CTBUH Skyscraper Award"
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
    "description": "award or recognition received by a person, organization or creative work",
    "label": "award received"
  },
  "qid": {
    "description": "The artistic Jenga-block apartments of Singapore",
    "label": "The Interlace"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "unmapped_violation",
      "candidate_score": 0,
      "mapped_violation_constraint_qid": null,
      "mapped_violation_family": "unknown",
      "violation_name": "Q|41176"
    }
  ],
  "candidate_violation_names": [
    "Q|41176"
  ],
  "causality_match_level": "unmapped_violation",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "mapped_violation_confidence": "none",
  "mapped_violation_constraint_label": null,
  "mapped_violation_constraint_qid": null,
  "mapped_violation_family": "unknown",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Q|41176",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Q|41176",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
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
  "hash_after": "dd3141826e6c4ed30fee5d8ea1843f38a994a620",
  "hash_before": "74a153b04c2c54f8592801c6187afd44d4d41016",
  "property_revision_id": 2254793027,
  "property_revision_prev": 2254791750,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "unmapped_violation",
    "mapped_violation_constraint_qid": null,
    "result": false,
    "step": "causality_filter",
    "violation_name": "Q|41176"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "unmapped_violation",
        "candidate_score": 0,
        "mapped_violation_constraint_qid": null,
        "mapped_violation_family": "unknown",
        "violation_name": "Q|41176"
      }
    ],
    "candidate_violation_names": [
      "Q|41176"
    ],
    "causality_match_level": "unmapped_violation",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": null,
    "mapped_report_constraint_qid": null,
    "mapped_report_family": "unknown",
    "mapped_violation_confidence": "none",
    "mapped_violation_constraint_label": null,
    "mapped_violation_constraint_qid": null,
    "mapped_violation_family": "unknown",
    "mapped_violation_reason": "unmapped_violation_type",
    "property_overlap_with_report_pids": [],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Q|41176",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Q|41176"
  }
]
```

---

## 017. `reform_Q25486814_P140_1858269104`

| Field | Value |
|---|---|
| qid | Q25486814 |
| property | P140 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P140::1858269104 |
| tbox_revision_key | TBOX::P140::1858269104 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 1858269104,
  "property_revision_prev": 1858268019
}
```

### Violation Context

```json
{
  "report_fix_date": "2023-03-22T13:59:00",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P140",
  "report_revision_new": 1858281604,
  "report_revision_old": 1857743994,
  "report_violation_type": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "entity that only exists in a work of fiction",
    "social entity established to meet needs or pursue goals",
    "non-natural geographic entities such as settlements, infrastructure, and excavations",
    "place of burial",
    "any set of human beings",
    "social role with a set of powers and responsibilities within an organization",
    "object used in a religion",
    "behaviour motivated by religious belief",
    "single established symbolic acts, is a part of ceremony",
    "topic viewed from a historical point of view",
    "aesthetic item or artistic creation",
    "human who is hypothesized to exist, but where evidence is not conclusive",
    "time of special importance marked by adherents of some religion",
    "transgression or alleged transgression resulting in public outrage",
    "entity that only exists in myth, legends and folklore",
    "icon representing a particular religion",
    "use of symbols, themes, and subject matter in the visual arts",
    "invocation or act that seeks to activate a rapport with a deity",
    "abstract object associated with religion",
    "physical object made or shaped by humans",
    "architectural practices used in places of worship",
    "occupation or profession that serves a purpose within the context of a religion",
    "idea or tenet that is part of a faith; refers to attitudes towards mythological, supernatural, or spiritual aspects of a religion; is usually codified",
    "... omitted 6 items"
  ],
  "report_violation_type_labels_en": [
    "human",
    "fictional entity",
    "organization",
    "artificial geographic entity",
    "cemetery",
    "group of humans",
    "position",
    "religious object",
    "religious behaviour",
    "rite",
    "aspect of history",
    "work of art",
    "human whose existence is disputed",
    "religious holiday",
    "scandal",
    "mythical entity",
    "religious symbol",
    "iconography",
    "prayer",
    "religious concept",
    "artificial physical object",
    "sacred architecture",
    "religious occupation",
    "religious belief",
    "... omitted 6 items"
  ],
  "report_violation_type_normalized": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819",
  "report_violation_type_qids": [
    "Q5",
    "Q14897293",
    "Q43229",
    "Q27096235",
    "Q39614",
    "Q16334295",
    "Q4164871",
    "Q21029893",
    "Q2110808",
    "Q628455",
    "Q17524420",
    "Q838948",
    "Q21070568",
    "Q375011",
    "Q192909",
    "Q24334685",
    "Q60469796",
    "Q208145",
    "Q40953",
    "Q23847174",
    "Q8205328",
    "Q47848",
    "Q63187345",
    "Q2728698",
    "... omitted 6 items"
  ],
  "report_violation_type_raw": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819",
  "value": null,
  "value_current_2026": [
    "Q683724"
  ],
  "value_current_2026_descriptions_en": [
    "national church of the Armenian people"
  ],
  "value_current_2026_labels_en": [
    "Armenian Apostolic Church"
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
    "description": "religion of a person, organization or religious building, or associated with this subject",
    "label": "religion or worldview"
  },
  "qid": {
    "description": "Monastery complex in Hrazdan, Armenia",
    "label": "Surb Ach Monastery"
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
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
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

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819"
    }
  ],
  "candidate_violation_names": [
    "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819"
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
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
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
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
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
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "9350eb1efaf050760d7a6f09673cbcca6d90d25d",
  "hash_before": "76a46513d7cd812bbd1598a668bf57ca51785c36",
  "property_revision_id": 1858269104,
  "property_revision_prev": 1858268019,
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
    "violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819"
      }
    ],
    "candidate_violation_names": [
      "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819"
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
    "ignored_added_values": [
      "Q29934200"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
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
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
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
    "violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819"
  }
]
```

---

## 018. `reform_Q28008281_P5800_1839493915`

| Field | Value |
|---|---|
| qid | Q28008281 |
| property | P5800 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | tail |
| constraint_family | Q21510865 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P5800::1839493915 |
| tbox_revision_key | TBOX::P5800::1839493915 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "OmegaFallon",
  "kind": "T_BOX",
  "property_revision_id": 1839493915,
  "property_revision_prev": 1839493614
}
```

### Violation Context

```json
{
  "report_fix_date": "2023-02-23T20:14:20",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5800",
  "report_revision_new": 1840033229,
  "report_revision_old": 1839112494,
  "report_violation_type": "Type Q|386724",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation"
  ],
  "report_violation_type_labels_en": [
    "work"
  ],
  "report_violation_type_normalized": "Type Q|386724",
  "report_violation_type_qids": [
    "Q386724"
  ],
  "report_violation_type_raw": "Type Q|386724",
  "value": null,
  "value_current_2026": [
    "Q27623618"
  ],
  "value_current_2026_descriptions_en": [
    "character being of minor importance to the story"
  ],
  "value_current_2026_labels_en": [
    "minor character"
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
    "description": "narrative role of this character (should be used as a qualifier with P674 or restricted to a certain work using P10663)",
    "label": "narrative role"
  },
  "qid": {
    "description": "character from Disney's Wreck-It Ralph",
    "label": "Mary"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|386724"
    }
  ],
  "candidate_violation_names": [
    "Type Q|386724"
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
  "ignored_added_values": [
    "Q46466787"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|386724",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
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
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q46466787"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Type Q|386724",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
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
  "author": "OmegaFallon",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "427147685b3b5f5354652f600db4a0a14896f66b",
  "hash_before": "f4c0d35976e743caeb893ae9e4ab41709d66736c",
  "property_revision_id": 1839493915,
  "property_revision_prev": 1839493614,
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
    "violation_name": "Type Q|386724"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|386724"
      }
    ],
    "candidate_violation_names": [
      "Type Q|386724"
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
    "ignored_added_values": [
      "Q46466787"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
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
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|386724",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
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
    "violation_name": "Type Q|386724"
  }
]
```

---

## 019. `reform_Q28151040_P6597_2040528587`

| Field | Value |
|---|---|
| qid | Q28151040 |
| property | P6597 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | tail |
| constraint_family | Q19474404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P6597::2040528587 |
| tbox_revision_key | TBOX::P6597::2040528587 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "Nikki",
  "kind": "T_BOX",
  "property_revision_id": 2040528587,
  "property_revision_prev": 2040528338
}
```

### Violation Context

```json
{
  "report_fix_date": "2023-12-31T07:00:53",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6597",
  "report_revision_new": 2041211978,
  "report_revision_old": 2040400362,
  "report_violation_type": "Item P|407",
  "report_violation_type_normalized": "Item P|407",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|407",
  "value": null,
  "value_current_2026": [
    "180680"
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
    "description": "ID of corresponding entry in the DFD online dictionary of family names",
    "label": "DFD ID"
  },
  "qid": {
    "description": "family name",
    "label": "Myller"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "lexeme requires language constraint",
    "qid": "Q55819106"
  },
  {
    "label_en": "label in language constraint",
    "qid": "Q108139345"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|407"
    }
  ],
  "candidate_violation_names": [
    "Item P|407"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|407",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|407",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "Nikki",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "8901b5ad38dd448c8371c1ec9b6a8dc79d75ba9a",
  "hash_before": "077995b0705d0cd9e93a85d257d18baf836c2888",
  "property_revision_id": 2040528587,
  "property_revision_prev": 2040528338,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|407"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|407"
      }
    ],
    "candidate_violation_names": [
      "Item P|407"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q29934200"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|407",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|407"
  }
]
```

---

## 020. `reform_Q28667156_P140_1858269104`

| Field | Value |
|---|---|
| qid | Q28667156 |
| property | P140 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P140::1858269104 |
| tbox_revision_key | TBOX::P140::1858269104 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 1858269104,
  "property_revision_prev": 1858268019
}
```

### Violation Context

```json
{
  "report_fix_date": "2023-03-22T13:59:00",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P140",
  "report_revision_new": 1858281604,
  "report_revision_old": 1857743994,
  "report_violation_type": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "entity that only exists in a work of fiction",
    "social entity established to meet needs or pursue goals",
    "non-natural geographic entities such as settlements, infrastructure, and excavations",
    "place of burial",
    "any set of human beings",
    "social role with a set of powers and responsibilities within an organization",
    "object used in a religion",
    "behaviour motivated by religious belief",
    "single established symbolic acts, is a part of ceremony",
    "topic viewed from a historical point of view",
    "aesthetic item or artistic creation",
    "human who is hypothesized to exist, but where evidence is not conclusive",
    "time of special importance marked by adherents of some religion",
    "transgression or alleged transgression resulting in public outrage",
    "entity that only exists in myth, legends and folklore",
    "icon representing a particular religion",
    "use of symbols, themes, and subject matter in the visual arts",
    "invocation or act that seeks to activate a rapport with a deity",
    "abstract object associated with religion",
    "physical object made or shaped by humans",
    "architectural practices used in places of worship",
    "occupation or profession that serves a purpose within the context of a religion",
    "idea or tenet that is part of a faith; refers to attitudes towards mythological, supernatural, or spiritual aspects of a religion; is usually codified",
    "... omitted 6 items"
  ],
  "report_violation_type_labels_en": [
    "human",
    "fictional entity",
    "organization",
    "artificial geographic entity",
    "cemetery",
    "group of humans",
    "position",
    "religious object",
    "religious behaviour",
    "rite",
    "aspect of history",
    "work of art",
    "human whose existence is disputed",
    "religious holiday",
    "scandal",
    "mythical entity",
    "religious symbol",
    "iconography",
    "prayer",
    "religious concept",
    "artificial physical object",
    "sacred architecture",
    "religious occupation",
    "religious belief",
    "... omitted 6 items"
  ],
  "report_violation_type_normalized": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819",
  "report_violation_type_qids": [
    "Q5",
    "Q14897293",
    "Q43229",
    "Q27096235",
    "Q39614",
    "Q16334295",
    "Q4164871",
    "Q21029893",
    "Q2110808",
    "Q628455",
    "Q17524420",
    "Q838948",
    "Q21070568",
    "Q375011",
    "Q192909",
    "Q24334685",
    "Q60469796",
    "Q208145",
    "Q40953",
    "Q23847174",
    "Q8205328",
    "Q47848",
    "Q63187345",
    "Q2728698",
    "... omitted 6 items"
  ],
  "report_violation_type_raw": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819",
  "value": null,
  "value_current_2026": [
    "Q9268"
  ],
  "value_current_2026_descriptions_en": [
    "Abrahamic monotheistic ethnic religion of the Jews"
  ],
  "value_current_2026_labels_en": [
    "Judaism"
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
    "description": "religion of a person, organization or religious building, or associated with this subject",
    "label": "religion or worldview"
  },
  "qid": {
    "description": null,
    "label": "Синагога ремесленников"
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
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
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

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819"
    }
  ],
  "candidate_violation_names": [
    "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819"
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
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
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
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
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
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "9350eb1efaf050760d7a6f09673cbcca6d90d25d",
  "hash_before": "76a46513d7cd812bbd1598a668bf57ca51785c36",
  "property_revision_id": 1858269104,
  "property_revision_prev": 1858268019,
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
    "violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819"
      }
    ],
    "candidate_violation_names": [
      "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819"
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
    "ignored_added_values": [
      "Q29934200"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
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
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
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
    "violation_name": "Type Q|5, Q|14897293, Q|43229, Q|27096235, Q|39614, Q|16334295, Q|4164871, Q|21029893, Q|2110808, Q|628455, Q|17524420, Q|838948, Q|21070568, Q|375011, Q|192909, Q|24334685, Q|60469796, Q|208145, Q|40953, Q|23847174, Q|8205328, Q|47848, Q|63187345, Q|2728698, Q|178885, Q|511056, Q|2627975, Q|3071477, Q|1131696, Q|189819"
  }
]
```

---

## 021. `reform_Q3580099_P2521_1725922197`

| Field | Value |
|---|---|
| qid | Q3580099 |
| property | P2521 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | tail |
| constraint_family | Q53869507 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P2521::1725922197 |
| tbox_revision_key | TBOX::P2521::1725922197 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 1725922197,
  "property_revision_prev": 1719625275
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-09-11T11:10:29",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2521",
  "report_revision_new": 1726203339,
  "report_revision_old": 1725211201,
  "report_violation_type": "Item P|3321",
  "report_violation_type_normalized": "Item P|3321",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|3321",
  "value": null,
  "value_current_2026": [
    "éducatrice de la protection judiciaire de la jeunesse@fr"
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
    "description": "female form of name or title (for male use P3321)",
    "label": "female form of label"
  },
  "qid": {
    "description": "fonctionnaire d'État français de catégorie A",
    "label": "éducateur de la protection judiciaire de la jeunesse"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
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
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|3321"
    }
  ],
  "candidate_violation_names": [
    "Item P|3321"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|3321",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|3321",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "f0cc45546da8768eeffa00936c0e3a553c2aca4b",
  "hash_before": "42e4e2c51d366fbf19084c4ab0a0759e9306eafa",
  "property_revision_id": 1725922197,
  "property_revision_prev": 1719625275,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|3321"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|3321"
      }
    ],
    "candidate_violation_names": [
      "Item P|3321"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q29934200"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|3321",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|3321"
  }
]
```

---

## 022. `reform_Q37039974_P6597_2040528587`

| Field | Value |
|---|---|
| qid | Q37039974 |
| property | P6597 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P6597::2040528587 |
| tbox_revision_key | TBOX::P6597::2040528587 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "Nikki",
  "kind": "T_BOX",
  "property_revision_id": 2040528587,
  "property_revision_prev": 2040528338
}
```

### Violation Context

```json
{
  "report_fix_date": "2023-12-31T07:00:53",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6597",
  "report_revision_new": 2041211978,
  "report_revision_old": 2040400362,
  "report_violation_type": "Item P|407",
  "report_violation_type_normalized": "Item P|407",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|407",
  "value": null,
  "value_current_2026": [
    "11860"
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
    "description": "ID of corresponding entry in the DFD online dictionary of family names",
    "label": "DFD ID"
  },
  "qid": {
    "description": "family name",
    "label": "Vaas"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "lexeme requires language constraint",
    "qid": "Q55819106"
  },
  {
    "label_en": "label in language constraint",
    "qid": "Q108139345"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|407"
    }
  ],
  "candidate_violation_names": [
    "Item P|407"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|407",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|407",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "Nikki",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "8901b5ad38dd448c8371c1ec9b6a8dc79d75ba9a",
  "hash_before": "077995b0705d0cd9e93a85d257d18baf836c2888",
  "property_revision_id": 2040528587,
  "property_revision_prev": 2040528338,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|407"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|407"
      }
    ],
    "candidate_violation_names": [
      "Item P|407"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q29934200"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|407",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|407"
  }
]
```

---

## 023. `reform_Q4200853_P2521_1725922197`

| Field | Value |
|---|---|
| qid | Q4200853 |
| property | P2521 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | head |
| constraint_family | Q53869507 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P2521::1725922197 |
| tbox_revision_key | TBOX::P2521::1725922197 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 1725922197,
  "property_revision_prev": 1719625275
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-09-11T11:10:29",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2521",
  "report_revision_new": 1726203339,
  "report_revision_old": 1725211201,
  "report_violation_type": "Item P|3321",
  "report_violation_type_normalized": "Item P|3321",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|3321",
  "value": null,
  "value_current_2026": [
    "indonesias@gl",
    "indonesias@es",
    "Indonesierin@de",
    "Indoneesierin@lb"
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
    "description": "female form of name or title (for male use P3321)",
    "label": "female form of label"
  },
  "qid": {
    "description": "citizens or residents of Indonesia",
    "label": "Indonesians"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
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
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|3321"
    }
  ],
  "candidate_violation_names": [
    "Item P|3321"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|3321",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|3321",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "f0cc45546da8768eeffa00936c0e3a553c2aca4b",
  "hash_before": "42e4e2c51d366fbf19084c4ab0a0759e9306eafa",
  "property_revision_id": 1725922197,
  "property_revision_prev": 1719625275,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|3321"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|3321"
      }
    ],
    "candidate_violation_names": [
      "Item P|3321"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q29934200"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|3321",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|3321"
  }
]
```

---

## 024. `reform_Q4520451_P18_2393757166`

| Field | Value |
|---|---|
| qid | Q4520451 |
| property | P18 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type |   |
| group_key | TBOX::P18::2393757166 |
| tbox_revision_key | TBOX::P18::2393757166 |

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
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "author": "Verdy p",
  "kind": "T_BOX",
  "property_revision_id": 2393757166,
  "property_revision_prev": 2393756905
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-23T11:22:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P18",
  "report_revision_new": 2395642478,
  "report_revision_old": 2395357701,
  "report_violation_type": "Q|13406463",
  "report_violation_type_descriptions_en": [
    "page of a Wikimedia project with a list of something"
  ],
  "report_violation_type_labels_en": [
    "Wikimedia list article"
  ],
  "report_violation_type_normalized": "Q|13406463",
  "report_violation_type_qids": [
    "Q13406463"
  ],
  "report_violation_type_raw": "Q|13406463",
  "value": null,
  "value_current_2026": [
    "20130821 Joods monument Rijssen.jpg",
    "Canadian War Cemetery Holten Hoofdingang Close-up.JPG",
    "Gedenkstenen Haarlerweg Espelo.jpg",
    "Oorlogsmonument Molenbelterweg Holten Zijaanzicht.jpg",
    "Plaquette station Holten.jpg"
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
    "description": "image of relevant illustration of the subject; if available, also use more specific properties (sample: coat of arms image, locator map, flag image, signature image, logo image, collage image)",
    "label": "image"
  },
  "qid": {
    "description": "Wikimedia list article",
    "label": "list of war memorials in Rijssen-Holten"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "Commons link constraint",
    "qid": "Q21510852"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "unmapped_violation",
      "candidate_score": 0,
      "mapped_violation_constraint_qid": null,
      "mapped_violation_family": "unknown",
      "violation_name": "Q|13406463"
    }
  ],
  "candidate_violation_names": [
    "Q|13406463"
  ],
  "causality_match_level": "unmapped_violation",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q52060874"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q52060874"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "mapped_violation_confidence": "none",
  "mapped_violation_constraint_label": null,
  "mapped_violation_constraint_qid": null,
  "mapped_violation_family": "unknown",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Q|13406463",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q52060874"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q52060874"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Q|13406463",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_specific_without_overlap": false
}
```

### T-box Constraint Diff

_Lean Stage 4 may prune full constraint signatures. Prefer the compact diff summary above when full signatures are empty._

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 0,
  "author": "Verdy p",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "e3303e74183b70ca25ccbb0c41b3be659021a3b8",
  "hash_before": "939de6ad1bd17bae20218d86952ce698c42403a1",
  "property_revision_id": 2393757166,
  "property_revision_prev": 2393756905,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "unmapped_violation",
    "mapped_violation_constraint_qid": null,
    "result": false,
    "step": "causality_filter",
    "violation_name": "Q|13406463"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "unmapped_violation",
        "candidate_score": 0,
        "mapped_violation_constraint_qid": null,
        "mapped_violation_family": "unknown",
        "violation_name": "Q|13406463"
      }
    ],
    "candidate_violation_names": [
      "Q|13406463"
    ],
    "causality_match_level": "unmapped_violation",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q52060874"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q52060874"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": null,
    "mapped_report_constraint_qid": null,
    "mapped_report_family": "unknown",
    "mapped_violation_confidence": "none",
    "mapped_violation_constraint_label": null,
    "mapped_violation_constraint_qid": null,
    "mapped_violation_family": "unknown",
    "mapped_violation_reason": "unmapped_violation_type",
    "property_overlap_with_report_pids": [],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Q|13406463",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Q|13406463"
  }
]
```

---

## 025. `reform_Q463882_P166_2254793027`

| Field | Value |
|---|---|
| qid | Q463882 |
| property | P166 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | head |
| constraint_family | Q21510851 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type |   |
| group_key | TBOX::P166::2254793027 |
| tbox_revision_key | TBOX::P166::2254793027 |

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
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 2254793027,
  "property_revision_prev": 2254791750
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-10-01T12:00:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P166",
  "report_revision_new": 2255073635,
  "report_revision_old": 2254683010,
  "report_violation_type": "Q|5",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo"
  ],
  "report_violation_type_labels_en": [
    "human"
  ],
  "report_violation_type_normalized": "Q|5",
  "report_violation_type_qids": [
    "Q5"
  ],
  "report_violation_type_raw": "Q|5",
  "value": null,
  "value_current_2026": [
    "Q17130755"
  ],
  "value_current_2026_descriptions_en": [
    null
  ],
  "value_current_2026_labels_en": [
    "Grotius Lectures"
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
    "description": "award or recognition received by a person, organization or creative work",
    "label": "award received"
  },
  "qid": {
    "description": "American law professor and writer (born 1962)",
    "label": "Amy Chua"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "unmapped_violation",
      "candidate_score": 0,
      "mapped_violation_constraint_qid": null,
      "mapped_violation_family": "unknown",
      "violation_name": "Q|5"
    }
  ],
  "candidate_violation_names": [
    "Q|5"
  ],
  "causality_match_level": "unmapped_violation",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "mapped_violation_confidence": "none",
  "mapped_violation_constraint_label": null,
  "mapped_violation_constraint_qid": null,
  "mapped_violation_family": "unknown",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Q|5",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Q|5",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
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
  "hash_after": "dd3141826e6c4ed30fee5d8ea1843f38a994a620",
  "hash_before": "74a153b04c2c54f8592801c6187afd44d4d41016",
  "property_revision_id": 2254793027,
  "property_revision_prev": 2254791750,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "unmapped_violation",
    "mapped_violation_constraint_qid": null,
    "result": false,
    "step": "causality_filter",
    "violation_name": "Q|5"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "unmapped_violation",
        "candidate_score": 0,
        "mapped_violation_constraint_qid": null,
        "mapped_violation_family": "unknown",
        "violation_name": "Q|5"
      }
    ],
    "candidate_violation_names": [
      "Q|5"
    ],
    "causality_match_level": "unmapped_violation",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": null,
    "mapped_report_constraint_qid": null,
    "mapped_report_family": "unknown",
    "mapped_violation_confidence": "none",
    "mapped_violation_constraint_label": null,
    "mapped_violation_constraint_qid": null,
    "mapped_violation_family": "unknown",
    "mapped_violation_reason": "unmapped_violation_type",
    "property_overlap_with_report_pids": [],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Q|5",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Q|5"
  }
]
```

---

## 026. `reform_Q55716932_P5800_1839493915`

| Field | Value |
|---|---|
| qid | Q55716932 |
| property | P5800 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | mid |
| constraint_family | Q21510865 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503250 subject type constraint |
| group_key | TBOX::P5800::1839493915 |
| tbox_revision_key | TBOX::P5800::1839493915 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "subject type constraint",
  "decision_constraint_type_qid": "Q21503250"
}
```

#### Repair Target

```json
{
  "author": "OmegaFallon",
  "kind": "T_BOX",
  "property_revision_id": 1839493915,
  "property_revision_prev": 1839493614
}
```

### Violation Context

```json
{
  "report_fix_date": "2023-02-23T20:14:20",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5800",
  "report_revision_new": 1840033229,
  "report_revision_old": 1839112494,
  "report_violation_type": "Type Q|386724",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation"
  ],
  "report_violation_type_labels_en": [
    "work"
  ],
  "report_violation_type_normalized": "Type Q|386724",
  "report_violation_type_qids": [
    "Q386724"
  ],
  "report_violation_type_raw": "Type Q|386724",
  "value": null,
  "value_current_2026": [
    "Q3246821",
    "Q245204"
  ],
  "value_current_2026_descriptions_en": [
    "character who is named or referred to in the title, performance part that gives the title to the piece",
    "character of a work actively opposing the protagonist"
  ],
  "value_current_2026_labels_en": [
    "title character",
    "antagonist"
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
    "description": "narrative role of this character (should be used as a qualifier with P674 or restricted to a certain work using P10663)",
    "label": "narrative role"
  },
  "qid": {
    "description": "fictional character from The Big Lebowski, namesake of the film's protagonist",
    "label": "Jeffrey Lebowski"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503250",
      "mapped_violation_family": "type",
      "violation_name": "Type Q|386724"
    }
  ],
  "candidate_violation_names": [
    "Type Q|386724"
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
  "ignored_added_values": [
    "Q46466787"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "type constraint",
  "mapped_violation_constraint_qid": "Q21503250",
  "mapped_violation_family": "type",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Type Q|386724",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
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
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503250"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503250"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q46466787"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "type constraint",
  "mapped_report_constraint_qid": "Q21503250",
  "mapped_report_family": "type",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Type Q|386724",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
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
  "author": "OmegaFallon",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "427147685b3b5f5354652f600db4a0a14896f66b",
  "hash_before": "f4c0d35976e743caeb893ae9e4ab41709d66736c",
  "property_revision_id": 1839493915,
  "property_revision_prev": 1839493614,
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
    "violation_name": "Type Q|386724"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503250",
        "mapped_violation_family": "type",
        "violation_name": "Type Q|386724"
      }
    ],
    "candidate_violation_names": [
      "Type Q|386724"
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
    "ignored_added_values": [
      "Q46466787"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
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
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2308",
      "P2309"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Type Q|386724",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
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
    "violation_name": "Type Q|386724"
  }
]
```

---

## 027. `reform_Q59315167_P21_2442825468`

| Field | Value |
|---|---|
| qid | Q59315167 |
| property | P21 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21502838 conflicts-with constraint |
| group_key | TBOX::P21::2442825468 |
| tbox_revision_key | TBOX::P21::2442825468 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "conflicts-with constraint",
  "decision_constraint_type_qid": "Q21502838"
}
```

#### Repair Target

```json
{
  "author": "Jerimee",
  "kind": "T_BOX",
  "property_revision_id": 2442825468,
  "property_revision_prev": 2440297725
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-16T15:52:53",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P21",
  "report_revision_new": 2443041553,
  "report_revision_old": 2442751815,
  "report_violation_type": "Conflicts with P|17",
  "report_violation_type_normalized": "Conflicts with P|17",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|17",
  "value": null,
  "value_current_2026": [
    "Q6581097"
  ],
  "value_current_2026_descriptions_en": [
    "to be used in \"sex or gender\" (P21) to indicate that the human subject is a male or \"semantic gender\" (P10339) to indicate that a word refers to a male person"
  ],
  "value_current_2026_labels_en": [
    "male"
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
    "description": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
    "label": "sex or gender"
  },
  "qid": {
    "description": "Russian politician",
    "label": "Valery Usatyuk"
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
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21502838",
      "mapped_violation_family": "conflicts_with",
      "violation_name": "Conflicts with P|17"
    }
  ],
  "candidate_violation_names": [
    "Conflicts with P|17"
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
  "ignored_added_values": [
    "Q181360"
  ],
  "ignored_changed_qualifier_properties": [
    "P2303"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "conflicts-with constraint",
  "mapped_report_constraint_qid": "Q21502838",
  "mapped_report_family": "conflicts_with",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "conflicts-with constraint",
  "mapped_violation_constraint_qid": "Q21502838",
  "mapped_violation_family": "conflicts_with",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Conflicts with P|17",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
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
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21502838"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21502838"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q181360"
  ],
  "ignored_changed_qualifier_properties": [
    "P2303"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "conflicts-with constraint",
  "mapped_report_constraint_qid": "Q21502838",
  "mapped_report_family": "conflicts_with",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Conflicts with P|17",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
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
  "author": "Jerimee",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "9ffdd87dbef8eb12ce4d41577040011ac2ef7891",
  "hash_before": "4153e44669496bf040a48eb9d9fecb724aee43d6",
  "property_revision_id": 2442825468,
  "property_revision_prev": 2440297725,
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
    "violation_name": "Conflicts with P|17"
  },
  {
    "result": "Q21502838",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21502838",
        "mapped_violation_family": "conflicts_with",
        "violation_name": "Conflicts with P|17"
      }
    ],
    "candidate_violation_names": [
      "Conflicts with P|17"
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
    "ignored_added_values": [
      "Q181360"
    ],
    "ignored_changed_qualifier_properties": [
      "P2303"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
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
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Conflicts with P|17",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
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
    "violation_name": "Conflicts with P|17"
  }
]
```

---

## 028. `reform_Q686591_P166_2254793027`

| Field | Value |
|---|---|
| qid | Q686591 |
| property | P166 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | head |
| constraint_family | Q21510851 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type |   |
| group_key | TBOX::P166::2254793027 |
| tbox_revision_key | TBOX::P166::2254793027 |

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
| rationale | The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "no_matching_changed_constraint",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 2254793027,
  "property_revision_prev": 2254791750
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-10-01T12:00:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P166",
  "report_revision_new": 2255073635,
  "report_revision_old": 2254683010,
  "report_violation_type": "Q|484170",
  "report_violation_type_descriptions_en": [
    "France territorial subdivision for municipalities"
  ],
  "report_violation_type_labels_en": [
    "commune of France"
  ],
  "report_violation_type_normalized": "Q|484170",
  "report_violation_type_qids": [
    "Q484170"
  ],
  "report_violation_type_raw": "Q|484170",
  "value": null,
  "value_current_2026": [
    "Q125510668"
  ],
  "value_current_2026_descriptions_en": [
    null
  ],
  "value_current_2026_labels_en": [
    "Mistralian City"
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
    "description": "award or recognition received by a person, organization or creative work",
    "label": "award received"
  },
  "qid": {
    "description": "commune in Vaucluse, France",
    "label": "Le Thor"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "unmapped_violation",
      "candidate_score": 0,
      "mapped_violation_constraint_qid": null,
      "mapped_violation_family": "unknown",
      "violation_name": "Q|484170"
    }
  ],
  "candidate_violation_names": [
    "Q|484170"
  ],
  "causality_match_level": "unmapped_violation",
  "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "mapped_violation_confidence": "none",
  "mapped_violation_constraint_label": null,
  "mapped_violation_constraint_qid": null,
  "mapped_violation_family": "unknown",
  "property_overlap_with_report_pids": [],
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Q|484170",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_confidence": "low",
  "target_constraint_selection_reason": "no_matching_changed_constraint",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": false
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": null,
  "ignored_changed_qualifier_properties": null,
  "ignored_removed_values": null,
  "ignored_value_count": null,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": null,
  "mapped_report_constraint_qid": null,
  "mapped_report_family": "unknown",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": null,
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Q|484170",
  "semantic_added_values": null,
  "semantic_changed_qualifier_properties": null,
  "semantic_removed_values": null,
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": false,
  "target_constraint_is_related_family": false,
  "target_constraint_label": null,
  "target_constraint_qid": null,
  "target_constraint_selection_reason": "no_matching_changed_constraint",
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
  "hash_after": "dd3141826e6c4ed30fee5d8ea1843f38a994a620",
  "hash_before": "74a153b04c2c54f8592801c6187afd44d4d41016",
  "property_revision_id": 2254793027,
  "property_revision_prev": 2254791750,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": null
}
```

### Decision Trace

```json
[
  {
    "causality_match_level": "unmapped_violation",
    "mapped_violation_constraint_qid": null,
    "result": false,
    "step": "causality_filter",
    "violation_name": "Q|484170"
  },
  {
    "result": null,
    "step": "target_constraint",
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "unmapped_violation",
        "candidate_score": 0,
        "mapped_violation_constraint_qid": null,
        "mapped_violation_family": "unknown",
        "violation_name": "Q|484170"
      }
    ],
    "candidate_violation_names": [
      "Q|484170"
    ],
    "causality_match_level": "unmapped_violation",
    "causality_match_reason": "changed constraints do not establish a causal link to the reported violation",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": null,
    "mapped_report_constraint_qid": null,
    "mapped_report_family": "unknown",
    "mapped_violation_confidence": "none",
    "mapped_violation_constraint_label": null,
    "mapped_violation_constraint_qid": null,
    "mapped_violation_family": "unknown",
    "mapped_violation_reason": "unmapped_violation_type",
    "property_overlap_with_report_pids": [],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Q|484170",
    "step": "tbox_causality",
    "target_constraint_is_changed": false,
    "target_constraint_is_related_family": false,
    "target_constraint_label": null,
    "target_constraint_qid": null,
    "target_constraint_selection_confidence": "low",
    "target_constraint_selection_reason": "no_matching_changed_constraint",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": false,
    "violation_name": "Q|484170"
  }
]
```

---

## 029. `reform_Q954457_P2521_1725922197`

| Field | Value |
|---|---|
| qid | Q954457 |
| property | P2521 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | mid |
| constraint_family | Q53869507 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P2521::1725922197 |
| tbox_revision_key | TBOX::P2521::1725922197 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 1725922197,
  "property_revision_prev": 1719625275
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-09-11T11:10:29",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2521",
  "report_revision_new": 1726203339,
  "report_revision_old": 1725211201,
  "report_violation_type": "Item P|3321",
  "report_violation_type_normalized": "Item P|3321",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|3321",
  "value": null,
  "value_current_2026": [
    "dirigente pubblica@it"
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
    "description": "female form of name or title (for male use P3321)",
    "label": "female form of label"
  },
  "qid": {
    "description": "in Italia, pubblico funzionario avente grado e funzioni di dirigente",
    "label": "Italian senior civil servant"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
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
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|3321"
    }
  ],
  "candidate_violation_names": [
    "Item P|3321"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|3321",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|3321",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "f0cc45546da8768eeffa00936c0e3a553c2aca4b",
  "hash_before": "42e4e2c51d366fbf19084c4ab0a0759e9306eafa",
  "property_revision_id": 1725922197,
  "property_revision_prev": 1719625275,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|3321"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|3321"
      }
    ],
    "candidate_violation_names": [
      "Item P|3321"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q29934200"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|3321",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|3321"
  }
]
```

---

## 030. `reform_Q974170_P166_2254793027`

| Field | Value |
|---|---|
| qid | Q974170 |
| property | P166 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / UNKNOWN_TBOX_CAUSALITY / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_unknown_causality |
| popularity_bucket | mid |
| constraint_family | Q21510851 |
| classification_rule_family | tbox_schema_causality |
| classification_rule_subfamily | unknown_tbox_causality |
| decision_constraint_type | Q21503247 item-requires-statement constraint |
| group_key | TBOX::P166::2254793027 |
| tbox_revision_key | TBOX::P166::2254793027 |

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
| rationale | Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change. |
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
  "classification_rule_subfamily": "unknown_tbox_causality",
  "constraint_family": null,
  "decision_constraint_source": "mapped_violation_constraint_changed",
  "decision_constraint_type_label": "item-requires-statement constraint",
  "decision_constraint_type_qid": "Q21503247"
}
```

#### Repair Target

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 2254793027,
  "property_revision_prev": 2254791750
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-10-01T12:00:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P166",
  "report_revision_new": 2255073635,
  "report_revision_old": 2254683010,
  "report_violation_type": "Item P|31",
  "report_violation_type_normalized": "Item P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|31",
  "value": null,
  "value_current_2026": [
    "Q80589"
  ],
  "value_current_2026_descriptions_en": [
    "title of honor"
  ],
  "value_current_2026_labels_en": [
    "People's Hero of Yugoslavia"
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
    "description": "award or recognition received by a person, organization or creative work",
    "label": "award received"
  },
  "qid": {
    "description": null,
    "label": "1st Lika Brigade"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Causality

_Public directional subtype is coarse for backward compatibility; use `directional_subtype_precise` for polarity-specific analysis._

```json
{
  "candidate_violation_mappings_preview": [
    {
      "candidate_causality_match_level": "exact_constraint_family_only",
      "candidate_score": 25,
      "mapped_violation_constraint_qid": "Q21503247",
      "mapped_violation_family": "required_statement",
      "violation_name": "Item P|31"
    }
  ],
  "candidate_violation_names": [
    "Item P|31"
  ],
  "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
  "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "compatible_language_overlap_with_report_langs": [],
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "compatible_property_overlap_with_report_pids": [],
  "compatible_scope_overlap_with_report_values": [],
  "compatible_value_overlap_with_report_qids": [],
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "language_overlap_with_report_langs": [],
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "mapped_violation_confidence": "medium",
  "mapped_violation_constraint_label": "item requires statement constraint",
  "mapped_violation_constraint_qid": "Q21503247",
  "mapped_violation_family": "required_statement",
  "property_overlap_with_report_pids": [],
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "scope_overlap_with_report_values": [],
  "selected_violation_name": "Item P|31",
  "semantic_added_value_count": 0,
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_value_count": 0,
  "semantic_removed_values": [],
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
  "target_constraint_selection_confidence": "high",
  "target_constraint_selection_reason": "mapped_violation_constraint_changed",
  "value_overlap_with_report_qids": [],
  "value_specific_without_overlap": true
}
```

### T-box Compact Diff Summary

```json
{
  "added_value_count": null,
  "added_values": null,
  "analysis_slice_precise": null,
  "changed_constraint_qids_all": [
    "Q21503247"
  ],
  "changed_constraint_qids_from_entries": [],
  "changed_constraint_qids_from_qualifier_changes": [
    "Q21503247"
  ],
  "changed_qualifier_properties": null,
  "compatible_overlap_reason": "no_type_compatible_overlap",
  "compatible_overlap_used": false,
  "directional_subtype_precise": null,
  "ignored_added_values": [
    "Q29934200"
  ],
  "ignored_changed_qualifier_properties": [
    "P4680"
  ],
  "ignored_removed_values": [],
  "ignored_value_count": 1,
  "incompatible_overlap_ignored": {},
  "lean_stage4_pruned_full_signatures": true,
  "mapped_report_constraint_label": "item requires statement constraint",
  "mapped_report_constraint_qid": "Q21503247",
  "mapped_report_family": "required_statement",
  "polarity": null,
  "polarity_basis": null,
  "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
  "removed_value_count": null,
  "removed_values": null,
  "selected_violation_name": "Item P|31",
  "semantic_added_values": [],
  "semantic_changed_qualifier_properties": [],
  "semantic_removed_values": [],
  "set_operation": null,
  "set_semantics": null,
  "source": "classification.decision_trace.tbox_causality",
  "target_constraint_is_changed": true,
  "target_constraint_is_related_family": false,
  "target_constraint_label": "item requires statement constraint",
  "target_constraint_qid": "Q21503247",
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
  "author": "عُثمان",
  "before_constraint_count": 0,
  "changed_constraint_types": [],
  "constraints_readable_en": null,
  "hash_after": "dd3141826e6c4ed30fee5d8ea1843f38a994a620",
  "hash_before": "74a153b04c2c54f8592801c6187afd44d4d41016",
  "property_revision_id": 2254793027,
  "property_revision_prev": 2254791750,
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
    "mapped_violation_constraint_qid": "Q21503247",
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|31"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed"
  },
  {
    "candidate_violation_mappings_preview": [
      {
        "candidate_causality_match_level": "exact_constraint_family_only",
        "candidate_score": 25,
        "mapped_violation_constraint_qid": "Q21503247",
        "mapped_violation_family": "required_statement",
        "violation_name": "Item P|31"
      }
    ],
    "candidate_violation_names": [
      "Item P|31"
    ],
    "causality_match_level": "exact_constraint_family_only_no_compatible_overlap",
    "causality_match_reason": "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report",
    "changed_constraint_qids_all": [
      "Q21503247"
    ],
    "changed_constraint_qids_from_entries": [],
    "changed_constraint_qids_from_qualifier_changes": [
      "Q21503247"
    ],
    "compatible_language_overlap_with_report_langs": [],
    "compatible_overlap_reason": "no_type_compatible_overlap",
    "compatible_overlap_used": false,
    "compatible_property_overlap_with_report_pids": [],
    "compatible_scope_overlap_with_report_values": [],
    "compatible_value_overlap_with_report_qids": [],
    "ignored_added_values": [
      "Q29934200"
    ],
    "ignored_changed_qualifier_properties": [
      "P4680"
    ],
    "ignored_removed_values": [],
    "ignored_value_count": 1,
    "incompatible_overlap_ignored": {},
    "language_overlap_with_report_langs": [],
    "mapped_report_constraint_label": "item requires statement constraint",
    "mapped_report_constraint_qid": "Q21503247",
    "mapped_report_family": "required_statement",
    "mapped_violation_confidence": "medium",
    "mapped_violation_constraint_label": "item requires statement constraint",
    "mapped_violation_constraint_qid": "Q21503247",
    "mapped_violation_family": "required_statement",
    "mapped_violation_reason": "item_requires_statement_prefix_mapping",
    "property_overlap_with_report_pids": [],
    "qualifier_filter_reason": "all_changed_qualifiers_are_metadata_or_irrelevant_for_family",
    "relevant_qualifier_properties": [
      "P2305",
      "P2306"
    ],
    "result": "UNKNOWN_TBOX_CAUSALITY",
    "scope_overlap_with_report_values": [],
    "selected_violation_name": "Item P|31",
    "semantic_added_value_count": 0,
    "semantic_added_values": [],
    "semantic_changed_qualifier_properties": [],
    "semantic_removed_value_count": 0,
    "semantic_removed_values": [],
    "step": "tbox_causality",
    "target_constraint_is_changed": true,
    "target_constraint_is_related_family": false,
    "target_constraint_label": "item requires statement constraint",
    "target_constraint_qid": "Q21503247",
    "target_constraint_selection_confidence": "high",
    "target_constraint_selection_reason": "mapped_violation_constraint_changed",
    "target_family": "required_statement",
    "value_overlap_with_report_qids": [],
    "value_specific_without_overlap": true,
    "violation_name": "Item P|31"
  }
]
```

---
