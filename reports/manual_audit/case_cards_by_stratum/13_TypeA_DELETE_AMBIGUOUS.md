# TypeA_DELETE_AMBIGUOUS

Cases: 27

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q107352535_2443599373`

| Field | Value |
|---|---|
| qid | Q107352535 |
| property | P4404 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q107352535::P4404 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["04188afa-fc58-4f44-b97f-c3b86f62abe7"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "04188afa-fc58-4f44-b97f-c3b86f62abe7"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "04188afa-fc58-4f44-b97f-c3b86f62abe7"
  ],
  "removed_unique_values": [
    "04188afa-fc58-4f44-b97f-c3b86f62abe7"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Iamcarbon",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "04188afa-fc58-4f44-b97f-c3b86f62abe7"
  ],
  "revision_id": 2443599373,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "04188afa-fc58-4f44-b97f-c3b86f62abe7": 1
    },
    "old_unique": [
      "04188afa-fc58-4f44-b97f-c3b86f62abe7"
    ],
    "old_values": [
      "04188afa-fc58-4f44-b97f-c3b86f62abe7"
    ],
    "old_values_raw": [
      "04188afa-fc58-4f44-b97f-c3b86f62abe7"
    ],
    "removed_unique_values": [
      "04188afa-fc58-4f44-b97f-c3b86f62abe7"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "04188afa-fc58-4f44-b97f-c3b86f62abe7": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-19T08:04:17",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4404",
  "report_revision_new": 2443995683,
  "report_revision_old": 2443785052,
  "report_violation_type": "Item P|2550",
  "report_violation_type_normalized": "Item P|2550",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|2550",
  "report_violation_types": [
    "Item P|2550",
    "Type Q|7302866, Q|193977",
    "Conflicts with P|31"
  ],
  "value": [
    "04188afa-fc58-4f44-b97f-c3b86f62abe7"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "identifier for a recording in the MusicBrainz open music encyclopedia",
    "label": "MusicBrainz recording ID"
  },
  "qid": {
    "description": "Flash and the Pan song",
    "label": "Hey, St. Peter"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "item p 2550"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q109563704_2442329832`

| Field | Value |
|---|---|
| qid | Q109563704 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q21510851 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q109563704::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q252"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q252"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q252"
  ],
  "removed_unique_values": [
    "Q252"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q252"
  ],
  "old_value_descriptions_en": [
    "island country in Southeast Asia and Oceania"
  ],
  "old_value_labels_en": [
    "Indonesia"
  ],
  "revision_id": 2442329832,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q252": 1
    },
    "old_unique": [
      "Q252"
    ],
    "old_values": [
      "Q252"
    ],
    "old_values_raw": [
      "Q252"
    ],
    "removed_unique_values": [
      "Q252"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q252": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-16T16:27:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2443054962,
  "report_revision_old": 2442766763,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q252"
  ],
  "value_descriptions_en": [
    "island country in Southeast Asia and Oceania"
  ],
  "value_labels_en": [
    "Indonesia"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "label": "country"
  },
  "qid": {
    "description": null,
    "label": "Irfan AB"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q114342236_2441117871`

| Field | Value |
|---|---|
| qid | Q114342236 |
| property | P106 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q53869507 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q114342236::P106 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q1650915"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q1650915"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q1650915"
  ],
  "removed_unique_values": [
    "Q1650915"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q1650915"
  ],
  "old_value_descriptions_en": [
    "person who engages in research, professionally or otherwise. If a more specific occupation is known, use that instead"
  ],
  "old_value_labels_en": [
    "researcher"
  ],
  "revision_id": 2441117871,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q1650915": 1
    },
    "old_unique": [
      "Q1650915"
    ],
    "old_values": [
      "Q1650915"
    ],
    "old_values_raw": [
      "Q1650915"
    ],
    "removed_unique_values": [
      "Q1650915"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q1650915": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-13T12:33:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P106",
  "report_revision_new": 2441832579,
  "report_revision_old": 2441298994,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q1650915"
  ],
  "value_descriptions_en": [
    "person who engages in research, professionally or otherwise. If a more specific occupation is known, use that instead"
  ],
  "value_labels_en": [
    "researcher"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "occupation of a person. See also \"field of work\" (Property:P101), \"position held\" (Property:P39). Not for groups of people. There, use \"field of work\" (Property:P101), \"industry\" (Property:P452), \"members have occupation\" (Property:P3989).",
    "label": "occupation"
  },
  "qid": {
    "description": "дослідник",
    "label": "Information Network of Departments of Dermatology (IVDK)"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
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
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q1166362_2422275679`

| Field | Value |
|---|---|
| qid | Q1166362 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q1166362::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["bourgogne-franche-comte/yonne/mairie-89281-01"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "bourgogne-franche-comte/yonne/mairie-89281-01"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "bourgogne-franche-comte/yonne/mairie-89281-01"
  ],
  "removed_unique_values": [
    "bourgogne-franche-comte/yonne/mairie-89281-01"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "bourgogne-franche-comte/yonne/mairie-89281-01"
  ],
  "revision_id": 2422275679,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "bourgogne-franche-comte/yonne/mairie-89281-01": 1
    },
    "old_unique": [
      "bourgogne-franche-comte/yonne/mairie-89281-01"
    ],
    "old_values": [
      "bourgogne-franche-comte/yonne/mairie-89281-01"
    ],
    "old_values_raw": [
      "bourgogne-franche-comte/yonne/mairie-89281-01"
    ],
    "removed_unique_values": [
      "bourgogne-franche-comte/yonne/mairie-89281-01"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "bourgogne-franche-comte/yonne/mairie-89281-01": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "bourgogne-franche-comte/yonne/mairie-89281-01"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Yonne, France",
    "label": "Les Ormes"
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
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q118288310_2444210911`

| Field | Value |
|---|---|
| qid | Q118288310 |
| property | P4404 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q118288310::P4404 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["ae50d552-e6a2-4ece-976e-47bccd072fb4"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "ae50d552-e6a2-4ece-976e-47bccd072fb4"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "ae50d552-e6a2-4ece-976e-47bccd072fb4"
  ],
  "removed_unique_values": [
    "ae50d552-e6a2-4ece-976e-47bccd072fb4"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Iamcarbon",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "ae50d552-e6a2-4ece-976e-47bccd072fb4"
  ],
  "revision_id": 2444210911,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "ae50d552-e6a2-4ece-976e-47bccd072fb4": 1
    },
    "old_unique": [
      "ae50d552-e6a2-4ece-976e-47bccd072fb4"
    ],
    "old_values": [
      "ae50d552-e6a2-4ece-976e-47bccd072fb4"
    ],
    "old_values_raw": [
      "ae50d552-e6a2-4ece-976e-47bccd072fb4"
    ],
    "removed_unique_values": [
      "ae50d552-e6a2-4ece-976e-47bccd072fb4"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "ae50d552-e6a2-4ece-976e-47bccd072fb4": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-21T06:55:38",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4404",
  "report_revision_new": 2444823785,
  "report_revision_old": 2444408016,
  "report_violation_type": "Item P|2550",
  "report_violation_type_normalized": "Item P|2550",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|2550",
  "report_violation_types": [
    "Item P|2550",
    "Type Q|7302866, Q|193977",
    "Conflicts with P|31"
  ],
  "value": [
    "ae50d552-e6a2-4ece-976e-47bccd072fb4"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "identifier for a recording in the MusicBrainz open music encyclopedia",
    "label": "MusicBrainz recording ID"
  },
  "qid": {
    "description": "2012 song by Shinhwa",
    "label": "This Love"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "item p 2550"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q123254229_2166663398`

| Field | Value |
|---|---|
| qid | Q123254229 |
| property | P6425 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q19474404 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q123254229::P6425 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["216"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "216"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "216"
  ],
  "removed_unique_values": [
    "216"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "RamSeraph",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "216"
  ],
  "revision_id": 2166663398,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "216": 1
    },
    "old_unique": [
      "216"
    ],
    "old_values": [
      "216"
    ],
    "old_values_raw": [
      "216"
    ],
    "removed_unique_values": [
      "216"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "216": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-05-30T08:09:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6425",
  "report_revision_new": 2168535356,
  "report_revision_old": 2155799388,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "216"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "code used for local government bodies by the Local Government Directory, a website run by the Government of India, which maintains directory of rural and urban local governments in India and their political territorial entities",
    "label": "LGD local body code"
  },
  "qid": {
    "description": "tehsil in Hoshiarpur district of Punjab, India",
    "label": "Hoshiarpur tehsil"
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
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q123256793_2166719508`

| Field | Value |
|---|---|
| qid | Q123256793 |
| property | P6425 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q19474404 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q123256793::P6425 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["675"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "675"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "675"
  ],
  "removed_unique_values": [
    "675"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "RamSeraph",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "675"
  ],
  "revision_id": 2166719508,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "675": 1
    },
    "old_unique": [
      "675"
    ],
    "old_values": [
      "675"
    ],
    "old_values_raw": [
      "675"
    ],
    "removed_unique_values": [
      "675"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "675": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-05-30T08:09:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6425",
  "report_revision_new": 2168535356,
  "report_revision_old": 2155799388,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "675"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "code used for local government bodies by the Local Government Directory, a website run by the Government of India, which maintains directory of rural and urban local governments in India and their political territorial entities",
    "label": "LGD local body code"
  },
  "qid": {
    "description": "tehsil in Baran district of Rajasthan, India",
    "label": "Shahbad tehsil"
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
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q124966080_2445323477`

| Field | Value |
|---|---|
| qid | Q124966080 |
| property | P176 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q21510865 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q124966080::P176 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q137286046"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q137286046"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q137286046"
  ],
  "removed_unique_values": [
    "Q137286046"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Meno25",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q137286046"
  ],
  "old_value_descriptions_en": [
    null
  ],
  "old_value_labels_en": [
    null
  ],
  "revision_id": 2445323477,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q137286046": 1
    },
    "old_unique": [
      "Q137286046"
    ],
    "old_values": [
      "Q137286046"
    ],
    "old_values_raw": [
      "Q137286046"
    ],
    "removed_unique_values": [
      "Q137286046"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q137286046": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-23T15:04:55",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P176",
  "report_revision_new": 2446087617,
  "report_revision_old": 2445487255,
  "report_violation_type": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
  "report_violation_type_descriptions_en": [
    "social entity established to meet needs or pursue goals",
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "fictional human or non-human character in a narrative work of art",
    "set of fictional characters",
    "people who work in the arts, a handicraft, or a skilled trade, particularly when utilizing traditional or non-mechanized methods",
    "occupation requiring specialized training",
    "kingdom of multicellular eukaryotic organisms",
    "group of firms that produce a closely related set of raw materials, goods, or services",
    "facility where goods are industrially made, or processed",
    "label applied to a person based on an activity they participate in",
    "room or building, with tools, used to repair or make goods",
    "group of one or more organism(s), which a taxonomist adjudges to be a unit"
  ],
  "report_violation_type_labels_en": [
    "organization",
    "human",
    "character",
    "group of fictional characters",
    "artisan",
    "profession",
    "Animalia",
    "industry",
    "factory",
    "occupation",
    "workshop",
    "taxon"
  ],
  "report_violation_type_normalized": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
  "report_violation_type_qids": [
    "Q43229",
    "Q5",
    "Q95074",
    "Q14514600",
    "Q1294787",
    "Q28640",
    "Q729",
    "Q268592",
    "Q83405",
    "Q12737077",
    "Q656720",
    "Q16521"
  ],
  "report_violation_type_raw": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
  "value": [
    "Q137286046"
  ],
  "value_descriptions_en": [
    null
  ],
  "value_labels_en": [
    null
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "(main or final) manufacturer or producer of this product",
    "label": "manufacturer"
  },
  "qid": {
    "description": "bright orange nail polish made by essie",
    "label": "Tangerine Tease"
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
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "value type q 43229 q 5 q 95074 q 14514600 q 1294787 q 28640 q 729 q 268592 q 83405 q 12737077 q 656720 q 16521"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q126904533_2438902655`

| Field | Value |
|---|---|
| qid | Q126904533 |
| property | P6367 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21502410 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q126904533::P6367 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["140187"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "140187"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "140187"
  ],
  "removed_unique_values": [
    "140187"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Sd5605",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "140187"
  ],
  "revision_id": 2438902655,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "140187": 1
    },
    "old_unique": [
      "140187"
    ],
    "old_values": [
      "140187"
    ],
    "old_values_raw": [
      "140187"
    ],
    "removed_unique_values": [
      "140187"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "140187": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-08T05:40:11",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6367",
  "report_revision_new": 2439487116,
  "report_revision_old": 2439077210,
  "report_violation_type": "Item P|13424",
  "report_violation_type_normalized": "Item P|13424",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|13424",
  "value": [
    "140187"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "a Taiwanese database for Games, Animation, Comic, Light Novels",
    "label": "Bahamut Gamer's Community ACG Database ID"
  },
  "qid": {
    "description": "2024 Japanese anime television series",
    "label": "Ranma ½"
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
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
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
    "label_en": "label in language constraint",
    "qid": "Q108139345"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "item p 13424"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q132181527_2442323278`

| Field | Value |
|---|---|
| qid | Q132181527 |
| property | P131 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q132181527::P131 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q488326"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q488326"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q488326"
  ],
  "removed_unique_values": [
    "Q488326"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q488326"
  ],
  "old_value_descriptions_en": [
    "province of the Democratic Republic of the Congo"
  ],
  "old_value_labels_en": [
    "South Kivu"
  ],
  "revision_id": 2442323278,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q488326": 1
    },
    "old_unique": [
      "Q488326"
    ],
    "old_values": [
      "Q488326"
    ],
    "old_values_raw": [
      "Q488326"
    ],
    "removed_unique_values": [
      "Q488326"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q488326": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-16T13:51:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
  "report_revision_new": 2443007807,
  "report_revision_old": 2442705396,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q488326"
  ],
  "value_descriptions_en": [
    "province of the Democratic Republic of the Congo"
  ],
  "value_labels_en": [
    "South Kivu"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
    "label": "located in the administrative territorial entity"
  },
  "qid": {
    "description": "politician of the Democratic Republic of the Congo",
    "label": "Janvier Msenyibwa Apele"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 011. `repair_Q136734934_2440052973`

| Field | Value |
|---|---|
| qid | Q136734934 |
| property | P123 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q136734934::P123 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q914456"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q914456"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q914456"
  ],
  "removed_unique_values": [
    "Q914456"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Dla archiv 1",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q914456"
  ],
  "old_value_descriptions_en": [
    "German publisher"
  ],
  "old_value_labels_en": [
    "Goldmann"
  ],
  "revision_id": 2440052973,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q914456": 1
    },
    "old_unique": [
      "Q914456"
    ],
    "old_values": [
      "Q914456"
    ],
    "old_values_raw": [
      "Q914456"
    ],
    "removed_unique_values": [
      "Q914456"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q914456": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-11T15:15:00",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P123",
  "report_revision_new": 2440910119,
  "report_revision_old": 2440433467,
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
  "value": [
    "Q914456"
  ],
  "value_descriptions_en": [
    "German publisher"
  ],
  "value_labels_en": [
    "Goldmann"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "Sachbuch von Richard David Precht",
    "label": "Angststillstand"
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

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "type q 1151067 q 80793969 q 95107111 q 374814 q 7397 q 286583 q 17538690 q 571 q 11410 q 105543609 q 7058673 q 115473170 q 116779426 q 1714118 q 904997 q 62651817 q 3331189 q 97180164 q 124030631 q 106140535 q 1980247 q 732577 q 1748756 q 49094714 q 7725310 q 1348645"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 012. `repair_Q1388921_2422270407`

| Field | Value |
|---|---|
| qid | Q1388921 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q1388921::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["auvergne-rhone-alpes/rhone/mairie-69104-01"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "auvergne-rhone-alpes/rhone/mairie-69104-01"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "auvergne-rhone-alpes/rhone/mairie-69104-01"
  ],
  "removed_unique_values": [
    "auvergne-rhone-alpes/rhone/mairie-69104-01"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "auvergne-rhone-alpes/rhone/mairie-69104-01"
  ],
  "revision_id": 2422270407,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "auvergne-rhone-alpes/rhone/mairie-69104-01": 1
    },
    "old_unique": [
      "auvergne-rhone-alpes/rhone/mairie-69104-01"
    ],
    "old_values": [
      "auvergne-rhone-alpes/rhone/mairie-69104-01"
    ],
    "old_values_raw": [
      "auvergne-rhone-alpes/rhone/mairie-69104-01"
    ],
    "removed_unique_values": [
      "auvergne-rhone-alpes/rhone/mairie-69104-01"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "auvergne-rhone-alpes/rhone/mairie-69104-01": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "auvergne-rhone-alpes/rhone/mairie-69104-01"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Rhône, France",
    "label": "Jullié"
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
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 013. `repair_Q1465561_2422275630`

| Field | Value |
|---|---|
| qid | Q1465561 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q1465561::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["bourgogne-franche-comte/saone-et-loire/mairie-71291-01"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "bourgogne-franche-comte/saone-et-loire/mairie-71291-01"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "bourgogne-franche-comte/saone-et-loire/mairie-71291-01"
  ],
  "removed_unique_values": [
    "bourgogne-franche-comte/saone-et-loire/mairie-71291-01"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "bourgogne-franche-comte/saone-et-loire/mairie-71291-01"
  ],
  "revision_id": 2422275630,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "bourgogne-franche-comte/saone-et-loire/mairie-71291-01": 1
    },
    "old_unique": [
      "bourgogne-franche-comte/saone-et-loire/mairie-71291-01"
    ],
    "old_values": [
      "bourgogne-franche-comte/saone-et-loire/mairie-71291-01"
    ],
    "old_values_raw": [
      "bourgogne-franche-comte/saone-et-loire/mairie-71291-01"
    ],
    "removed_unique_values": [
      "bourgogne-franche-comte/saone-et-loire/mairie-71291-01"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "bourgogne-franche-comte/saone-et-loire/mairie-71291-01": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "bourgogne-franche-comte/saone-et-loire/mairie-71291-01"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Saône-et-Loire, France",
    "label": "Melay"
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
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 014. `repair_Q21613_2422271583`

| Field | Value |
|---|---|
| qid | Q21613 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q21613::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["grand-est/moselle/mairie-57166-01"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "grand-est/moselle/mairie-57166-01"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "grand-est/moselle/mairie-57166-01"
  ],
  "removed_unique_values": [
    "grand-est/moselle/mairie-57166-01"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "grand-est/moselle/mairie-57166-01"
  ],
  "revision_id": 2422271583,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "grand-est/moselle/mairie-57166-01": 1
    },
    "old_unique": [
      "grand-est/moselle/mairie-57166-01"
    ],
    "old_values": [
      "grand-est/moselle/mairie-57166-01"
    ],
    "old_values_raw": [
      "grand-est/moselle/mairie-57166-01"
    ],
    "removed_unique_values": [
      "grand-est/moselle/mairie-57166-01"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "grand-est/moselle/mairie-57166-01": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "grand-est/moselle/mairie-57166-01"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Moselle, France",
    "label": "Dalhain"
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
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 015. `repair_Q21783_2422270906`

| Field | Value |
|---|---|
| qid | Q21783 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q21783::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["grand-est/moselle/mairie-57033-01"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "grand-est/moselle/mairie-57033-01"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "grand-est/moselle/mairie-57033-01"
  ],
  "removed_unique_values": [
    "grand-est/moselle/mairie-57033-01"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "grand-est/moselle/mairie-57033-01"
  ],
  "revision_id": 2422270906,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "grand-est/moselle/mairie-57033-01": 1
    },
    "old_unique": [
      "grand-est/moselle/mairie-57033-01"
    ],
    "old_values": [
      "grand-est/moselle/mairie-57033-01"
    ],
    "old_values_raw": [
      "grand-est/moselle/mairie-57033-01"
    ],
    "removed_unique_values": [
      "grand-est/moselle/mairie-57033-01"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "grand-est/moselle/mairie-57033-01": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "grand-est/moselle/mairie-57033-01"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Moselle, France",
    "label": "Arzviller"
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
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 016. `repair_Q22758_2422264431`

| Field | Value |
|---|---|
| qid | Q22758 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q22758::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["grand-est/bas-rhin/mairie-67021-01"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "grand-est/bas-rhin/mairie-67021-01"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "grand-est/bas-rhin/mairie-67021-01"
  ],
  "removed_unique_values": [
    "grand-est/bas-rhin/mairie-67021-01"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "grand-est/bas-rhin/mairie-67021-01"
  ],
  "revision_id": 2422264431,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "grand-est/bas-rhin/mairie-67021-01": 1
    },
    "old_unique": [
      "grand-est/bas-rhin/mairie-67021-01"
    ],
    "old_values": [
      "grand-est/bas-rhin/mairie-67021-01"
    ],
    "old_values_raw": [
      "grand-est/bas-rhin/mairie-67021-01"
    ],
    "removed_unique_values": [
      "grand-est/bas-rhin/mairie-67021-01"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "grand-est/bas-rhin/mairie-67021-01": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "grand-est/bas-rhin/mairie-67021-01"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Bas-Rhin, Alsace, France",
    "label": "Barr"
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
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 017. `repair_Q28170070_2442333753`

| Field | Value |
|---|---|
| qid | Q28170070 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21510851 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q28170070::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q668"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q668"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q668"
  ],
  "removed_unique_values": [
    "Q668"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q668"
  ],
  "old_value_descriptions_en": [
    "country in South Asia"
  ],
  "old_value_labels_en": [
    "India"
  ],
  "revision_id": 2442333753,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q668": 1
    },
    "old_unique": [
      "Q668"
    ],
    "old_values": [
      "Q668"
    ],
    "old_values_raw": [
      "Q668"
    ],
    "removed_unique_values": [
      "Q668"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q668": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-16T16:27:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2443054962,
  "report_revision_old": 2442766763,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q668"
  ],
  "value_descriptions_en": [
    "country in South Asia"
  ],
  "value_labels_en": [
    "India"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "label": "country"
  },
  "qid": {
    "description": "chahamana king",
    "label": "Chandraraja I"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 018. `repair_Q33411200_2442408556`

| Field | Value |
|---|---|
| qid | Q33411200 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q33411200::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Husova 110 (Prachatice)"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Husova 110 (Prachatice)"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Husova 110 (Prachatice)"
  ],
  "removed_unique_values": [
    "Husova 110 (Prachatice)"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "JAnDbot",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Husova 110 (Prachatice)"
  ],
  "revision_id": 2442408556,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Husova 110 (Prachatice)": 1
    },
    "old_unique": [
      "Husova 110 (Prachatice)"
    ],
    "old_values": [
      "Husova 110 (Prachatice)"
    ],
    "old_values_raw": [
      "Husova 110 (Prachatice)"
    ],
    "removed_unique_values": [
      "Husova 110 (Prachatice)"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Husova 110 (Prachatice)": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-16T11:51:46",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2442981743,
  "report_revision_old": 2442645840,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Husova 110 (Prachatice)"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "house in Prachatice I, Czech Republic",
    "label": "Dům čp. 110"
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

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "commons link"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 019. `repair_Q42390642_2440887808`

| Field | Value |
|---|---|
| qid | Q42390642 |
| property | P7937 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q42390642::P7937 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q3328821"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q3328821"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q3328821"
  ],
  "removed_unique_values": [
    "Q3328821"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Maxlath",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q3328821"
  ],
  "old_value_descriptions_en": [
    "literary form characterized by the description of a sequence of events in a certain order"
  ],
  "old_value_labels_en": [
    "narration"
  ],
  "revision_id": 2440887808,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q3328821": 1
    },
    "old_unique": [
      "Q3328821"
    ],
    "old_values": [
      "Q3328821"
    ],
    "old_values_raw": [
      "Q3328821"
    ],
    "removed_unique_values": [
      "Q3328821"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q3328821": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-13T06:34:27",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
  "report_revision_new": 2441661090,
  "report_revision_old": 2441108460,
  "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "any set of works",
    "creative work which only appears in works of fiction"
  ],
  "report_violation_type_labels_en": [
    "work",
    "group of works",
    "fictional creative work"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_qids": [
    "Q386724",
    "Q17489659",
    "Q15306849"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_types": [
    "Type Q|386724, Q|17489659, Q|15306849",
    "None of"
  ],
  "value": [
    "Q3328821"
  ],
  "value_descriptions_en": [
    "literary form characterized by the description of a sequence of events in a certain order"
  ],
  "value_labels_en": [
    "narration"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "structure of a creative work",
    "label": "form of creative work"
  },
  "qid": {
    "description": "Uruguayan writer",
    "label": "Nedy Varela Cetani"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "type q 386724 q 17489659 q 15306849"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 020. `repair_Q4272898_2442334771`

| Field | Value |
|---|---|
| qid | Q4272898 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21510851 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q4272898::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q159"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q159"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q159"
  ],
  "removed_unique_values": [
    "Q159"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q159"
  ],
  "old_value_descriptions_en": [
    "country in Eastern Europe and Northern Asia"
  ],
  "old_value_labels_en": [
    "Russia"
  ],
  "revision_id": 2442334771,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q159": 1
    },
    "old_unique": [
      "Q159"
    ],
    "old_values": [
      "Q159"
    ],
    "old_values_raw": [
      "Q159"
    ],
    "removed_unique_values": [
      "Q159"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q159": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-16T16:27:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2443054962,
  "report_revision_old": 2442766763,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q159"
  ],
  "value_descriptions_en": [
    "country in Eastern Europe and Northern Asia"
  ],
  "value_labels_en": [
    "Russia"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "label": "country"
  },
  "qid": {
    "description": "Russian scientist",
    "label": "Boris Lyovin"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 021. `repair_Q4303556_2440875959`

| Field | Value |
|---|---|
| qid | Q4303556 |
| property | P7937 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q4303556::P7937 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q25379"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q25379"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q25379"
  ],
  "removed_unique_values": [
    "Q25379"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Maxlath",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q25379"
  ],
  "old_value_descriptions_en": [
    "theatrical dramatic work intended to be performed by actors (in theatre, radio or recorded for TV)"
  ],
  "old_value_labels_en": [
    "play"
  ],
  "revision_id": 2440875959,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q25379": 1
    },
    "old_unique": [
      "Q25379"
    ],
    "old_values": [
      "Q25379"
    ],
    "old_values_raw": [
      "Q25379"
    ],
    "removed_unique_values": [
      "Q25379"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q25379": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-13T06:34:27",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
  "report_revision_new": 2441661090,
  "report_revision_old": 2441108460,
  "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "any set of works",
    "creative work which only appears in works of fiction"
  ],
  "report_violation_type_labels_en": [
    "work",
    "group of works",
    "fictional creative work"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_qids": [
    "Q386724",
    "Q17489659",
    "Q15306849"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849",
  "value": [
    "Q25379"
  ],
  "value_descriptions_en": [
    "theatrical dramatic work intended to be performed by actors (in theatre, radio or recorded for TV)"
  ],
  "value_labels_en": [
    "play"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "structure of a creative work",
    "label": "form of creative work"
  },
  "qid": {
    "description": "Georgian and Soviet screenwriter and writer (1896-1954)",
    "label": "Ilo Mosashvili"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "type q 386724 q 17489659 q 15306849"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 022. `repair_Q4355725_2439906127`

| Field | Value |
|---|---|
| qid | Q4355725 |
| property | P212 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q21502410 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q4355725::P212 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["978-0-440-36149-7"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "978-0-440-36149-7"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "978-0-440-36149-7"
  ],
  "removed_unique_values": [
    "978-0-440-36149-7"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Pfadintegral",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "978-0-440-36149-7"
  ],
  "revision_id": 2439906127,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "978-0-440-36149-7": 1
    },
    "old_unique": [
      "978-0-440-36149-7"
    ],
    "old_values": [
      "978-0-440-36149-7"
    ],
    "old_values_raw": [
      "978-0-440-36149-7"
    ],
    "removed_unique_values": [
      "978-0-440-36149-7"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "978-0-440-36149-7": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-10T11:10:27",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P212",
  "report_revision_new": 2440424839,
  "report_revision_old": 2440015789,
  "report_violation_type": "Conflicts with P|8383",
  "report_violation_type_normalized": "Conflicts with P|8383",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|8383",
  "report_violation_types": [
    "Conflicts with P|8383",
    "Type Q|3331189, Q|187685, Q|732577, Q|7889, Q|317623, Q|1711593, Q|131436, Q|1266946",
    "Conflicts with P|31",
    "Item P|629",
    "Conflicts with P|7937"
  ],
  "value": [
    "978-0-440-36149-7"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "identifier for a book (edition), thirteen digit",
    "label": "International Standard Book Number-13"
  },
  "qid": {
    "description": "1975 novel by Philip José Farmer",
    "label": "Venus on the Half-Shell"
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
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 8383"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 023. `repair_Q4524043_2442335157`

| Field | Value |
|---|---|
| qid | Q4524043 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21510851 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q4524043::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q159"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q159"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q159"
  ],
  "removed_unique_values": [
    "Q159"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q159"
  ],
  "old_value_descriptions_en": [
    "country in Eastern Europe and Northern Asia"
  ],
  "old_value_labels_en": [
    "Russia"
  ],
  "revision_id": 2442335157,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q159": 1
    },
    "old_unique": [
      "Q159"
    ],
    "old_values": [
      "Q159"
    ],
    "old_values_raw": [
      "Q159"
    ],
    "removed_unique_values": [
      "Q159"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q159": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-16T16:27:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2443054962,
  "report_revision_old": 2442766763,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q159"
  ],
  "value_descriptions_en": [
    "country in Eastern Europe and Northern Asia"
  ],
  "value_labels_en": [
    "Russia"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "label": "country"
  },
  "qid": {
    "description": "Russian politician",
    "label": "Maksim Shingarkin"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 024. `repair_Q4545128_2442078917`

| Field | Value |
|---|---|
| qid | Q4545128 |
| property | P212 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21502410 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q4545128::P212 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["978-1-59740-165-4"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "978-1-59740-165-4"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "978-1-59740-165-4"
  ],
  "removed_unique_values": [
    "978-1-59740-165-4"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Pfadintegral",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "978-1-59740-165-4"
  ],
  "revision_id": 2442078917,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "978-1-59740-165-4": 1
    },
    "old_unique": [
      "978-1-59740-165-4"
    ],
    "old_values": [
      "978-1-59740-165-4"
    ],
    "old_values_raw": [
      "978-1-59740-165-4"
    ],
    "removed_unique_values": [
      "978-1-59740-165-4"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "978-1-59740-165-4": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-15T13:11:03",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P212",
  "report_revision_new": 2442672664,
  "report_revision_old": 2442291443,
  "report_violation_type": "Item P|629",
  "report_violation_type_normalized": "Item P|629",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|629",
  "report_violation_types": [
    "Item P|629",
    "Conflicts with P|8383",
    "Type Q|3331189, Q|187685, Q|732577, Q|7889, Q|317623, Q|1711593, Q|131436, Q|1266946",
    "Conflicts with P|31"
  ],
  "value": [
    "978-1-59740-165-4"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "identifier for a book (edition), thirteen digit",
    "label": "International Standard Book Number-13"
  },
  "qid": {
    "description": "1985 nonfiction book",
    "label": "...The Heavens and the Earth"
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
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "item p 629"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 025. `repair_Q59322_2422268846`

| Field | Value |
|---|---|
| qid | Q59322 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q59322::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["grand-est/meuse/mairie-55525-01"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "grand-est/meuse/mairie-55525-01"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "grand-est/meuse/mairie-55525-01"
  ],
  "removed_unique_values": [
    "grand-est/meuse/mairie-55525-01"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "grand-est/meuse/mairie-55525-01"
  ],
  "revision_id": 2422268846,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "grand-est/meuse/mairie-55525-01": 1
    },
    "old_unique": [
      "grand-est/meuse/mairie-55525-01"
    ],
    "old_values": [
      "grand-est/meuse/mairie-55525-01"
    ],
    "old_values_raw": [
      "grand-est/meuse/mairie-55525-01"
    ],
    "removed_unique_values": [
      "grand-est/meuse/mairie-55525-01"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "grand-est/meuse/mairie-55525-01": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "grand-est/meuse/mairie-55525-01"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Meuse, France",
    "label": "Vadelaincourt"
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
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 026. `repair_Q6160283_2443968709`

| Field | Value |
|---|---|
| qid | Q6160283 |
| property | P6802 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q21510852 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q6160283::P6802 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Jarkko Laine graffiti.jpg"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Jarkko Laine graffiti.jpg"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Jarkko Laine graffiti.jpg"
  ],
  "removed_unique_values": [
    "Jarkko Laine graffiti.jpg"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "CommonsDelinker",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Jarkko Laine graffiti.jpg"
  ],
  "revision_id": 2443968709,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Jarkko Laine graffiti.jpg": 1
    },
    "old_unique": [
      "Jarkko Laine graffiti.jpg"
    ],
    "old_values": [
      "Jarkko Laine graffiti.jpg"
    ],
    "old_values_raw": [
      "Jarkko Laine graffiti.jpg"
    ],
    "removed_unique_values": [
      "Jarkko Laine graffiti.jpg"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Jarkko Laine graffiti.jpg": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-20T06:24:59",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6802",
  "report_revision_new": 2444398259,
  "report_revision_old": 2443982698,
  "report_violation_type": "Conflicts with P|18",
  "report_violation_type_normalized": "Conflicts with P|18",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|18",
  "value": [
    "Jarkko Laine graffiti.jpg"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "less fitting image, used only because a better alternative is not available. If an appropriate image of the item is available, use P18 instead. Value should not be a generic placeholder",
    "label": "related image"
  },
  "qid": {
    "description": "Finnish author, poet and translator (1947–2006)",
    "label": "Jarkko Laine"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "Commons link constraint",
    "qid": "Q21510852"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 18"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 027. `repair_Q71307_2422269300`

| Field | Value |
|---|---|
| qid | Q71307 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | rule_or_logical |
| classification_rule_subfamily | delete_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q71307::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["bretagne/morbihan/mairie-56030-01"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "bretagne/morbihan/mairie-56030-01"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "bretagne/morbihan/mairie-56030-01"
  ],
  "removed_unique_values": [
    "bretagne/morbihan/mairie-56030-01"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "rule_or_logical",
  "classification_rule_subfamily": "delete_ambiguous",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "bretagne/morbihan/mairie-56030-01"
  ],
  "revision_id": 2422269300,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "bretagne/morbihan/mairie-56030-01": 1
    },
    "old_unique": [
      "bretagne/morbihan/mairie-56030-01"
    ],
    "old_values": [
      "bretagne/morbihan/mairie-56030-01"
    ],
    "old_values_raw": [
      "bretagne/morbihan/mairie-56030-01"
    ],
    "removed_unique_values": [
      "bretagne/morbihan/mairie-56030-01"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "bretagne/morbihan/mairie-56030-01": {
        "new": 0,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "bretagne/morbihan/mairie-56030-01"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Morbihan, France",
    "label": "Camoël"
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
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---
