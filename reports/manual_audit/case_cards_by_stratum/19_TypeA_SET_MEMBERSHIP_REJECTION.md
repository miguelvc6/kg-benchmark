# TypeA_SET_MEMBERSHIP_REJECTION

Cases: 26

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q103820103_2441757143`

| Field | Value |
|---|---|
| qid | Q103820103 |
| property | P421 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | none_of |
| decision_constraint_type | Q52558054 none-of constraint |
| group_key | ABOX::Q103820103::P421 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q6723", "Q6760", "Q190252"] |
| classification_target_tokens | ["Q36669"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q36669"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q36669"
  ],
  "removed_unique_values": [
    "Q36669"
  ],
  "retained_support_tokens": [
    "Q190252",
    "Q6723",
    "Q6760"
  ],
  "retained_unique_values": [
    "Q190252",
    "Q6723",
    "Q6760"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "none_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "none-of constraint",
  "decision_constraint_type_qid": "Q52558054"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q6723",
    "Q6760",
    "Q190252"
  ],
  "new_value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3",
    "time zone"
  ],
  "new_value_labels_en": [
    "UTC+02:00",
    "UTC+03:00",
    "Eastern European Time"
  ],
  "old_value": [
    "Q6723",
    "Q6760",
    "Q190252",
    "Q36669"
  ],
  "old_value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3",
    "time zone",
    "seasonal adjustment of clocks"
  ],
  "old_value_labels_en": [
    "UTC+02:00",
    "UTC+03:00",
    "Eastern European Time",
    "daylight saving time"
  ],
  "revision_id": 2441757143,
  "value": [
    "Q6723",
    "Q6760",
    "Q190252"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q190252": 1,
      "Q6723": 1,
      "Q6760": 1
    },
    "new_unique": [
      "Q190252",
      "Q6723",
      "Q6760"
    ],
    "new_values": [
      "Q6723",
      "Q6760",
      "Q190252"
    ],
    "new_values_raw": [
      "Q6723",
      "Q6760",
      "Q190252"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q190252": 1,
      "Q36669": 1,
      "Q6723": 1,
      "Q6760": 1
    },
    "old_unique": [
      "Q190252",
      "Q36669",
      "Q6723",
      "Q6760"
    ],
    "old_values": [
      "Q6723",
      "Q6760",
      "Q190252",
      "Q36669"
    ],
    "old_values_raw": [
      "Q6723",
      "Q6760",
      "Q190252",
      "Q36669"
    ],
    "removed_unique_values": [
      "Q36669"
    ],
    "retained_unique_values": [
      "Q190252",
      "Q6723",
      "Q6760"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q36669": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3",
    "time zone"
  ],
  "value_labels_en": [
    "UTC+02:00",
    "UTC+03:00",
    "Eastern European Time"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T12:17:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P421",
  "report_revision_new": 2442278494,
  "report_revision_old": 2441776312,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "report_violation_types": [
    "None of",
    "Value type Q|12143"
  ],
  "value": [
    "Q6723",
    "Q6760",
    "Q190252",
    "Q36669"
  ],
  "value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3",
    "time zone",
    "seasonal adjustment of clocks"
  ],
  "value_labels_en": [
    "UTC+02:00",
    "UTC+03:00",
    "Eastern European Time",
    "daylight saving time"
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
  "truth_tokens": [
    "Q6723",
    "Q6760",
    "Q190252"
  ],
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
    "description": "raion in Dnipropetrovsk Oblast, Ukraine (established in 2020)",
    "label": "Dnipro Raion"
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

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q190252": 1,
        "Q6723": 1,
        "Q6760": 1
      },
      "new_unique": [
        "Q190252",
        "Q6723",
        "Q6760"
      ],
      "new_values": [
        "Q6723",
        "Q6760",
        "Q190252"
      ],
      "new_values_raw": [
        "Q6723",
        "Q6760",
        "Q190252"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q190252": 1,
        "Q36669": 1,
        "Q6723": 1,
        "Q6760": 1
      },
      "old_unique": [
        "Q190252",
        "Q36669",
        "Q6723",
        "Q6760"
      ],
      "old_values": [
        "Q6723",
        "Q6760",
        "Q190252",
        "Q36669"
      ],
      "old_values_raw": [
        "Q6723",
        "Q6760",
        "Q190252",
        "Q36669"
      ],
      "removed_unique_values": [
        "Q36669"
      ],
      "retained_unique_values": [
        "Q190252",
        "Q6723",
        "Q6760"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q36669": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q36669"
      ],
      "removed_values": [
        "Q36669"
      ],
      "report_type": "none of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q125220316_2442330860`

| Field | Value |
|---|---|
| qid | Q125220316 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | tail |
| constraint_family | Q21510851 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | none_of |
| decision_constraint_type | Q52558054 none-of constraint |
| group_key | ABOX::Q125220316::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q42620"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value proven invalid by one-of/none-of set membership. |
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
    "Q42620"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q42620"
  ],
  "removed_unique_values": [
    "Q42620"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "none_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "none-of constraint",
  "decision_constraint_type_qid": "Q52558054"
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
    "Q42620"
  ],
  "old_value_descriptions_en": [
    "interim government in Western Asia, governing West Bank Areas A and B since 1994 and, until 2006, the Gaza Strip"
  ],
  "old_value_labels_en": [
    "Palestinian National Authority"
  ],
  "revision_id": 2442330860,
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
      "Q42620": 1
    },
    "old_unique": [
      "Q42620"
    ],
    "old_values": [
      "Q42620"
    ],
    "old_values_raw": [
      "Q42620"
    ],
    "removed_unique_values": [
      "Q42620"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q42620": {
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
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "report_violation_types": [
    "None of",
    "Conflicts with P|31"
  ],
  "value": [
    "Q42620"
  ],
  "value_descriptions_en": [
    "interim government in Western Asia, governing West Bank Areas A and B since 1994 and, until 2006, the Gaza Strip"
  ],
  "value_labels_en": [
    "Palestinian National Authority"
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
    "description": "Palestinian militant (1972–2022)",
    "label": "Raad Thabet"
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
      "delete_reason": "set_membership",
      "known_set": [
        "Q108151641",
        "Q113403829",
        "Q11772",
        "Q14773",
        "Q15132899",
        "Q1657833",
        "Q17252",
        "Q2184",
        "Q219817",
        "Q219825",
        "Q221457",
        "Q22890",
        "Q23334",
        "Q2454900",
        "Q25594375",
        "Q29520",
        "Q42620",
        "Q458",
        "Q47588",
        "Q4843341",
        "Q518617",
        "Q5481",
        "Q5705",
        "Q5710",
        "... omitted 12 items"
      ],
      "removed_values": [
        "Q42620"
      ],
      "report_type": "none of"
    },
    "result": "SET_MEMBERSHIP_REJECTION",
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

## 003. `repair_Q16879762_2403237614`

| Field | Value |
|---|---|
| qid | Q16879762 |
| property | P1552 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | none_of |
| decision_constraint_type | Q52558054 none-of constraint |
| group_key | ABOX::Q16879762::P1552 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q66624698"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value proven invalid by one-of/none-of set membership. |
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
    "Q66624698"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q66624698"
  ],
  "removed_unique_values": [
    "Q66624698"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "none_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "none-of constraint",
  "decision_constraint_type_qid": "Q52558054"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Moebeus",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q66624698"
  ],
  "old_value_descriptions_en": [
    "conflation of definitions of what a \"debut single\" is. see \"disjoint union of\" (P2738) for what classes to replace this class with"
  ],
  "old_value_labels_en": [
    "debut single"
  ],
  "revision_id": 2403237614,
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
      "Q66624698": 1
    },
    "old_unique": [
      "Q66624698"
    ],
    "old_values": [
      "Q66624698"
    ],
    "old_values_raw": [
      "Q66624698"
    ],
    "removed_unique_values": [
      "Q66624698"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q66624698": {
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
  "report_fix_date": "2025-09-12T08:35:00",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1552",
  "report_revision_new": 2403882294,
  "report_revision_old": 2401727319,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": [
    "Q66624698"
  ],
  "value_descriptions_en": [
    "conflation of definitions of what a \"debut single\" is. see \"disjoint union of\" (P2738) for what classes to replace this class with"
  ],
  "value_labels_en": [
    "debut single"
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
    "description": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
    "label": "has characteristic"
  },
  "qid": {
    "description": "1992 single by Del the Funky Homosapien",
    "label": "Mistadobalina"
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
      "delete_reason": "set_membership",
      "known_set": [
        "Q10436169",
        "Q104438884",
        "Q104438889",
        "Q104438898",
        "Q104438918",
        "Q104439050",
        "Q104439055",
        "Q105933250",
        "Q1062345",
        "Q106612454",
        "Q106612650",
        "Q106826394",
        "Q107124972",
        "Q107316855",
        "Q107532692",
        "Q107737653",
        "Q109663871",
        "Q110736830",
        "Q110910",
        "Q110914098",
        "Q110914171",
        "Q110918476",
        "Q111223304",
        "Q113028955",
        "... omitted 104 items"
      ],
      "removed_values": [
        "Q66624698"
      ],
      "report_type": "none of"
    },
    "result": "SET_MEMBERSHIP_REJECTION",
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

## 004. `repair_Q218393_2441757615`

| Field | Value |
|---|---|
| qid | Q218393 |
| property | P421 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | none_of |
| decision_constraint_type | Q52558054 none-of constraint |
| group_key | ABOX::Q218393::P421 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q6723", "Q6760"] |
| classification_target_tokens | ["Q36669"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q36669"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q36669"
  ],
  "removed_unique_values": [
    "Q36669"
  ],
  "retained_support_tokens": [
    "Q6723",
    "Q6760"
  ],
  "retained_unique_values": [
    "Q6723",
    "Q6760"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "none_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "none-of constraint",
  "decision_constraint_type_qid": "Q52558054"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q6723",
    "Q6760"
  ],
  "new_value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3"
  ],
  "new_value_labels_en": [
    "UTC+02:00",
    "UTC+03:00"
  ],
  "old_value": [
    "Q6723",
    "Q6760",
    "Q36669"
  ],
  "old_value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3",
    "seasonal adjustment of clocks"
  ],
  "old_value_labels_en": [
    "UTC+02:00",
    "UTC+03:00",
    "daylight saving time"
  ],
  "revision_id": 2441757615,
  "value": [
    "Q6723",
    "Q6760"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q6723": 1,
      "Q6760": 1
    },
    "new_unique": [
      "Q6723",
      "Q6760"
    ],
    "new_values": [
      "Q6723",
      "Q6760"
    ],
    "new_values_raw": [
      "Q6723",
      "Q6760"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q36669": 1,
      "Q6723": 1,
      "Q6760": 1
    },
    "old_unique": [
      "Q36669",
      "Q6723",
      "Q6760"
    ],
    "old_values": [
      "Q6723",
      "Q6760",
      "Q36669"
    ],
    "old_values_raw": [
      "Q6723",
      "Q6760",
      "Q36669"
    ],
    "removed_unique_values": [
      "Q36669"
    ],
    "retained_unique_values": [
      "Q6723",
      "Q6760"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q36669": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3"
  ],
  "value_labels_en": [
    "UTC+02:00",
    "UTC+03:00"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T12:17:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P421",
  "report_revision_new": 2442278494,
  "report_revision_old": 2441776312,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "report_violation_types": [
    "None of",
    "Value type Q|12143"
  ],
  "value": [
    "Q6723",
    "Q6760",
    "Q36669"
  ],
  "value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3",
    "seasonal adjustment of clocks"
  ],
  "value_labels_en": [
    "UTC+02:00",
    "UTC+03:00",
    "daylight saving time"
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
  "truth_tokens": [
    "Q6723",
    "Q6760"
  ],
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
    "description": "city in Poltava Oblast, Ukraine",
    "label": "Karlivka"
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

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q6723": 1,
        "Q6760": 1
      },
      "new_unique": [
        "Q6723",
        "Q6760"
      ],
      "new_values": [
        "Q6723",
        "Q6760"
      ],
      "new_values_raw": [
        "Q6723",
        "Q6760"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q36669": 1,
        "Q6723": 1,
        "Q6760": 1
      },
      "old_unique": [
        "Q36669",
        "Q6723",
        "Q6760"
      ],
      "old_values": [
        "Q6723",
        "Q6760",
        "Q36669"
      ],
      "old_values_raw": [
        "Q6723",
        "Q6760",
        "Q36669"
      ],
      "removed_unique_values": [
        "Q36669"
      ],
      "retained_unique_values": [
        "Q6723",
        "Q6760"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q36669": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q36669"
      ],
      "removed_values": [
        "Q36669"
      ],
      "report_type": "none of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q327775_2444990087`

| Field | Value |
|---|---|
| qid | Q327775 |
| property | P136 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | none_of |
| decision_constraint_type | Q52558054 none-of constraint |
| group_key | ABOX::Q327775::P136 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q2143665", "Q157443"] |
| classification_target_tokens | ["Q52207399"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q52207399"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q52207399"
  ],
  "removed_unique_values": [
    "Q52207399"
  ],
  "retained_support_tokens": [
    "Q157443",
    "Q2143665"
  ],
  "retained_unique_values": [
    "Q157443",
    "Q2143665"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "none_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "none-of constraint",
  "decision_constraint_type_qid": "Q52558054"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Ezzex",
  "kind": "A_BOX",
  "new_value": [
    "Q2143665",
    "Q157443"
  ],
  "new_value_descriptions_en": [
    "film genre",
    "genre of film in which the main emphasis is on humour"
  ],
  "new_value_labels_en": [
    "children's film",
    "comedy film"
  ],
  "old_value": [
    "Q2143665",
    "Q157443",
    "Q52207399"
  ],
  "old_value_descriptions_en": [
    "film genre",
    "genre of film in which the main emphasis is on humour",
    "film based on a specific literary genre, the novel"
  ],
  "old_value_labels_en": [
    "children's film",
    "comedy film",
    "film based on a novel"
  ],
  "revision_id": 2444990087,
  "value": [
    "Q2143665",
    "Q157443"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q157443": 1,
      "Q2143665": 1
    },
    "new_unique": [
      "Q157443",
      "Q2143665"
    ],
    "new_values": [
      "Q2143665",
      "Q157443"
    ],
    "new_values_raw": [
      "Q2143665",
      "Q157443"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q157443": 1,
      "Q2143665": 1,
      "Q52207399": 1
    },
    "old_unique": [
      "Q157443",
      "Q2143665",
      "Q52207399"
    ],
    "old_values": [
      "Q2143665",
      "Q157443",
      "Q52207399"
    ],
    "old_values_raw": [
      "Q2143665",
      "Q157443",
      "Q52207399"
    ],
    "removed_unique_values": [
      "Q52207399"
    ],
    "retained_unique_values": [
      "Q157443",
      "Q2143665"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q52207399": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "film genre",
    "genre of film in which the main emphasis is on humour"
  ],
  "value_labels_en": [
    "children's film",
    "comedy film"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-23T18:19:34",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P136",
  "report_revision_new": 2446156898,
  "report_revision_old": 2445541075,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": [
    "Q2143665",
    "Q157443",
    "Q52207399"
  ],
  "value_descriptions_en": [
    "film genre",
    "genre of film in which the main emphasis is on humour",
    "film based on a specific literary genre, the novel"
  ],
  "value_labels_en": [
    "children's film",
    "comedy film",
    "film based on a novel"
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
  "truth_tokens": [
    "Q2143665",
    "Q157443"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
    "label": "genre"
  },
  "qid": {
    "description": "1972 film by Olle Hellbom",
    "label": "New Mischief by Emil"
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
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
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
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  }
]
```

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q157443": 1,
        "Q2143665": 1
      },
      "new_unique": [
        "Q157443",
        "Q2143665"
      ],
      "new_values": [
        "Q2143665",
        "Q157443"
      ],
      "new_values_raw": [
        "Q2143665",
        "Q157443"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q157443": 1,
        "Q2143665": 1,
        "Q52207399": 1
      },
      "old_unique": [
        "Q157443",
        "Q2143665",
        "Q52207399"
      ],
      "old_values": [
        "Q2143665",
        "Q157443",
        "Q52207399"
      ],
      "old_values_raw": [
        "Q2143665",
        "Q157443",
        "Q52207399"
      ],
      "removed_unique_values": [
        "Q52207399"
      ],
      "retained_unique_values": [
        "Q157443",
        "Q2143665"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q52207399": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q100749465",
        "Q101716172",
        "Q1028181",
        "Q104482419",
        "Q104969456",
        "Q106542313",
        "Q10701290",
        "Q107022156",
        "Q108197410",
        "Q109501952",
        "Q1097630",
        "Q110263445",
        "Q110416422",
        "Q110955215",
        "Q11190",
        "Q11287467",
        "Q1132127",
        "Q1140363",
        "Q11416",
        "Q11417",
        "Q11425",
        "Q11639",
        "Q117216668",
        "Q117717390",
        "... omitted 158 items"
      ],
      "removed_values": [
        "Q52207399"
      ],
      "report_type": "none of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q3316321_2443811609`

| Field | Value |
|---|---|
| qid | Q3316321 |
| property | P7937 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | none_of |
| decision_constraint_type | Q52558054 none-of constraint |
| group_key | ABOX::Q3316321::P7937 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q35760"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value proven invalid by one-of/none-of set membership. |
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
    "Q35760"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q35760"
  ],
  "removed_unique_values": [
    "Q35760"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "none_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "none-of constraint",
  "decision_constraint_type_qid": "Q52558054"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q35760"
  ],
  "old_value_descriptions_en": [
    "piece of writing often written from an author's personal point of view"
  ],
  "old_value_labels_en": [
    "essay"
  ],
  "revision_id": 2443811609,
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
      "Q35760": 1
    },
    "old_unique": [
      "Q35760"
    ],
    "old_values": [
      "Q35760"
    ],
    "old_values_raw": [
      "Q35760"
    ],
    "removed_unique_values": [
      "Q35760"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q35760": {
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
  "report_fix_date": "2025-12-20T06:03:18",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
  "report_revision_new": 2444393110,
  "report_revision_old": 2443978175,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": [
    "Q35760"
  ],
  "value_descriptions_en": [
    "piece of writing often written from an author's personal point of view"
  ],
  "value_labels_en": [
    "essay"
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
    "description": "by Marguerite Yourcenar",
    "label": "Mishima: A Vision of the Void"
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
      "delete_reason": "set_membership",
      "known_set": [
        "Q106612454",
        "Q106612650",
        "Q106826394",
        "Q107124972",
        "Q109663871",
        "Q110736830",
        "Q113577985",
        "Q1138081",
        "Q117717390",
        "Q1193470",
        "Q123177475",
        "Q1318295",
        "Q134556",
        "Q15982450",
        "Q169930",
        "Q178588",
        "Q178985",
        "Q193977",
        "Q2073093",
        "Q217199",
        "Q269415",
        "Q270171",
        "Q29168811",
        "Q3246734",
        "... omitted 50 items"
      ],
      "removed_values": [
        "Q35760"
      ],
      "report_type": "none of"
    },
    "result": "SET_MEMBERSHIP_REJECTION",
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

## 007. `repair_Q3544052_2445294722`

| Field | Value |
|---|---|
| qid | Q3544052 |
| property | P136 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | none_of |
| decision_constraint_type | Q52558054 none-of constraint |
| group_key | ABOX::Q3544052::P136 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q2143665", "Q157443", "Q28026639"] |
| classification_target_tokens | ["Q52207310"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q52207310"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q52207310"
  ],
  "removed_unique_values": [
    "Q52207310"
  ],
  "retained_support_tokens": [
    "Q157443",
    "Q2143665",
    "Q28026639"
  ],
  "retained_unique_values": [
    "Q157443",
    "Q2143665",
    "Q28026639"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "none_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "none-of constraint",
  "decision_constraint_type_qid": "Q52558054"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Ezzex",
  "kind": "A_BOX",
  "new_value": [
    "Q2143665",
    "Q157443",
    "Q28026639"
  ],
  "new_value_descriptions_en": [
    "film genre",
    "genre of film in which the main emphasis is on humour",
    "film genre associated with Christmas"
  ],
  "new_value_labels_en": [
    "children's film",
    "comedy film",
    "Christmas film"
  ],
  "old_value": [
    "Q2143665",
    "Q157443",
    "Q28026639",
    "Q52207310"
  ],
  "old_value_descriptions_en": [
    "film genre",
    "genre of film in which the main emphasis is on humour",
    "film genre associated with Christmas",
    "type of film adaptation"
  ],
  "old_value_labels_en": [
    "children's film",
    "comedy film",
    "Christmas film",
    "film based on book"
  ],
  "revision_id": 2445294722,
  "value": [
    "Q2143665",
    "Q157443",
    "Q28026639"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q157443": 1,
      "Q2143665": 1,
      "Q28026639": 1
    },
    "new_unique": [
      "Q157443",
      "Q2143665",
      "Q28026639"
    ],
    "new_values": [
      "Q2143665",
      "Q157443",
      "Q28026639"
    ],
    "new_values_raw": [
      "Q2143665",
      "Q157443",
      "Q28026639"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q157443": 1,
      "Q2143665": 1,
      "Q28026639": 1,
      "Q52207310": 1
    },
    "old_unique": [
      "Q157443",
      "Q2143665",
      "Q28026639",
      "Q52207310"
    ],
    "old_values": [
      "Q2143665",
      "Q157443",
      "Q28026639",
      "Q52207310"
    ],
    "old_values_raw": [
      "Q2143665",
      "Q157443",
      "Q28026639",
      "Q52207310"
    ],
    "removed_unique_values": [
      "Q52207310"
    ],
    "retained_unique_values": [
      "Q157443",
      "Q2143665",
      "Q28026639"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q52207310": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "film genre",
    "genre of film in which the main emphasis is on humour",
    "film genre associated with Christmas"
  ],
  "value_labels_en": [
    "children's film",
    "comedy film",
    "Christmas film"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-23T18:19:34",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P136",
  "report_revision_new": 2446156898,
  "report_revision_old": 2445541075,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": [
    "Q2143665",
    "Q157443",
    "Q28026639",
    "Q52207310"
  ],
  "value_descriptions_en": [
    "film genre",
    "genre of film in which the main emphasis is on humour",
    "film genre associated with Christmas",
    "type of film adaptation"
  ],
  "value_labels_en": [
    "children's film",
    "comedy film",
    "Christmas film",
    "film based on book"
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
  "truth_tokens": [
    "Q2143665",
    "Q157443",
    "Q28026639"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
    "label": "genre"
  },
  "qid": {
    "description": "1987 film directed by Lasse Hallström",
    "label": "More About the Children of Noisy Village"
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
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
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
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  }
]
```

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q157443": 1,
        "Q2143665": 1,
        "Q28026639": 1
      },
      "new_unique": [
        "Q157443",
        "Q2143665",
        "Q28026639"
      ],
      "new_values": [
        "Q2143665",
        "Q157443",
        "Q28026639"
      ],
      "new_values_raw": [
        "Q2143665",
        "Q157443",
        "Q28026639"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q157443": 1,
        "Q2143665": 1,
        "Q28026639": 1,
        "Q52207310": 1
      },
      "old_unique": [
        "Q157443",
        "Q2143665",
        "Q28026639",
        "Q52207310"
      ],
      "old_values": [
        "Q2143665",
        "Q157443",
        "Q28026639",
        "Q52207310"
      ],
      "old_values_raw": [
        "Q2143665",
        "Q157443",
        "Q28026639",
        "Q52207310"
      ],
      "removed_unique_values": [
        "Q52207310"
      ],
      "retained_unique_values": [
        "Q157443",
        "Q2143665",
        "Q28026639"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q52207310": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q100749465",
        "Q101716172",
        "Q1028181",
        "Q104482419",
        "Q104969456",
        "Q106542313",
        "Q10701290",
        "Q107022156",
        "Q108197410",
        "Q109501952",
        "Q1097630",
        "Q110263445",
        "Q110416422",
        "Q110955215",
        "Q11190",
        "Q11287467",
        "Q1132127",
        "Q1140363",
        "Q11416",
        "Q11417",
        "Q11425",
        "Q11639",
        "Q117216668",
        "Q117717390",
        "... omitted 158 items"
      ],
      "removed_values": [
        "Q52207310"
      ],
      "report_type": "none of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q4222180_2441758443`

| Field | Value |
|---|---|
| qid | Q4222180 |
| property | P421 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | none_of |
| decision_constraint_type | Q52558054 none-of constraint |
| group_key | ABOX::Q4222180::P421 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q6723", "Q6760"] |
| classification_target_tokens | ["Q36669"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q36669"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q36669"
  ],
  "removed_unique_values": [
    "Q36669"
  ],
  "retained_support_tokens": [
    "Q6723",
    "Q6760"
  ],
  "retained_unique_values": [
    "Q6723",
    "Q6760"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "none_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "none-of constraint",
  "decision_constraint_type_qid": "Q52558054"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q6723",
    "Q6760"
  ],
  "new_value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3"
  ],
  "new_value_labels_en": [
    "UTC+02:00",
    "UTC+03:00"
  ],
  "old_value": [
    "Q6723",
    "Q6760",
    "Q36669"
  ],
  "old_value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3",
    "seasonal adjustment of clocks"
  ],
  "old_value_labels_en": [
    "UTC+02:00",
    "UTC+03:00",
    "daylight saving time"
  ],
  "revision_id": 2441758443,
  "value": [
    "Q6723",
    "Q6760"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q6723": 1,
      "Q6760": 1
    },
    "new_unique": [
      "Q6723",
      "Q6760"
    ],
    "new_values": [
      "Q6723",
      "Q6760"
    ],
    "new_values_raw": [
      "Q6723",
      "Q6760"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q36669": 1,
      "Q6723": 1,
      "Q6760": 1
    },
    "old_unique": [
      "Q36669",
      "Q6723",
      "Q6760"
    ],
    "old_values": [
      "Q6723",
      "Q6760",
      "Q36669"
    ],
    "old_values_raw": [
      "Q6723",
      "Q6760",
      "Q36669"
    ],
    "removed_unique_values": [
      "Q36669"
    ],
    "retained_unique_values": [
      "Q6723",
      "Q6760"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q36669": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3"
  ],
  "value_labels_en": [
    "UTC+02:00",
    "UTC+03:00"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T12:17:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P421",
  "report_revision_new": 2442278494,
  "report_revision_old": 2441776312,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "report_violation_types": [
    "None of",
    "Value type Q|12143"
  ],
  "value": [
    "Q6723",
    "Q6760",
    "Q36669"
  ],
  "value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3",
    "seasonal adjustment of clocks"
  ],
  "value_labels_en": [
    "UTC+02:00",
    "UTC+03:00",
    "daylight saving time"
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
  "truth_tokens": [
    "Q6723",
    "Q6760"
  ],
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
    "description": "village in Nikopol Raion (district), Dnipropetrovsk Oblast (province), Ukraine",
    "label": "Kyslychuvata"
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

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q6723": 1,
        "Q6760": 1
      },
      "new_unique": [
        "Q6723",
        "Q6760"
      ],
      "new_values": [
        "Q6723",
        "Q6760"
      ],
      "new_values_raw": [
        "Q6723",
        "Q6760"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q36669": 1,
        "Q6723": 1,
        "Q6760": 1
      },
      "old_unique": [
        "Q36669",
        "Q6723",
        "Q6760"
      ],
      "old_values": [
        "Q6723",
        "Q6760",
        "Q36669"
      ],
      "old_values_raw": [
        "Q6723",
        "Q6760",
        "Q36669"
      ],
      "removed_unique_values": [
        "Q36669"
      ],
      "retained_unique_values": [
        "Q6723",
        "Q6760"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q36669": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q36669"
      ],
      "removed_values": [
        "Q36669"
      ],
      "report_type": "none of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q4297496_2441758545`

| Field | Value |
|---|---|
| qid | Q4297496 |
| property | P421 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | none_of |
| decision_constraint_type | Q52558054 none-of constraint |
| group_key | ABOX::Q4297496::P421 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q6723", "Q6760"] |
| classification_target_tokens | ["Q36669"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q36669"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q36669"
  ],
  "removed_unique_values": [
    "Q36669"
  ],
  "retained_support_tokens": [
    "Q6723",
    "Q6760"
  ],
  "retained_unique_values": [
    "Q6723",
    "Q6760"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "none_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "none-of constraint",
  "decision_constraint_type_qid": "Q52558054"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q6723",
    "Q6760"
  ],
  "new_value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3"
  ],
  "new_value_labels_en": [
    "UTC+02:00",
    "UTC+03:00"
  ],
  "old_value": [
    "Q6723",
    "Q6760",
    "Q36669"
  ],
  "old_value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3",
    "seasonal adjustment of clocks"
  ],
  "old_value_labels_en": [
    "UTC+02:00",
    "UTC+03:00",
    "daylight saving time"
  ],
  "revision_id": 2441758545,
  "value": [
    "Q6723",
    "Q6760"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q6723": 1,
      "Q6760": 1
    },
    "new_unique": [
      "Q6723",
      "Q6760"
    ],
    "new_values": [
      "Q6723",
      "Q6760"
    ],
    "new_values_raw": [
      "Q6723",
      "Q6760"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q36669": 1,
      "Q6723": 1,
      "Q6760": 1
    },
    "old_unique": [
      "Q36669",
      "Q6723",
      "Q6760"
    ],
    "old_values": [
      "Q6723",
      "Q6760",
      "Q36669"
    ],
    "old_values_raw": [
      "Q6723",
      "Q6760",
      "Q36669"
    ],
    "removed_unique_values": [
      "Q36669"
    ],
    "retained_unique_values": [
      "Q6723",
      "Q6760"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q36669": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3"
  ],
  "value_labels_en": [
    "UTC+02:00",
    "UTC+03:00"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T12:17:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P421",
  "report_revision_new": 2442278494,
  "report_revision_old": 2441776312,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "report_violation_types": [
    "None of",
    "Value type Q|12143"
  ],
  "value": [
    "Q6723",
    "Q6760",
    "Q36669"
  ],
  "value_descriptions_en": [
    "identifier for a time offset from UTC of +2",
    "identifier for a time offset from UTC of +3",
    "seasonal adjustment of clocks"
  ],
  "value_labels_en": [
    "UTC+02:00",
    "UTC+03:00",
    "daylight saving time"
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
  "truth_tokens": [
    "Q6723",
    "Q6760"
  ],
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
    "description": "village in Vilniansk Raion (district), Zaporizhia Oblast (province), Ukraine",
    "label": "Mykhailivka"
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

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q6723": 1,
        "Q6760": 1
      },
      "new_unique": [
        "Q6723",
        "Q6760"
      ],
      "new_values": [
        "Q6723",
        "Q6760"
      ],
      "new_values_raw": [
        "Q6723",
        "Q6760"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q36669": 1,
        "Q6723": 1,
        "Q6760": 1
      },
      "old_unique": [
        "Q36669",
        "Q6723",
        "Q6760"
      ],
      "old_values": [
        "Q6723",
        "Q6760",
        "Q36669"
      ],
      "old_values_raw": [
        "Q6723",
        "Q6760",
        "Q36669"
      ],
      "removed_unique_values": [
        "Q36669"
      ],
      "retained_unique_values": [
        "Q6723",
        "Q6760"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q36669": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q36669"
      ],
      "removed_values": [
        "Q36669"
      ],
      "report_type": "none of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q499891_2444344191`

| Field | Value |
|---|---|
| qid | Q499891 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | head |
| constraint_family | Q21510851 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | none_of |
| decision_constraint_type | Q52558054 none-of constraint |
| group_key | ABOX::Q499891::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| classification_target_tokens | ["Q458"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value proven invalid by one-of/none-of set membership. |
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
    "Q458"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q458"
  ],
  "removed_unique_values": [
    "Q458"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "none_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "none-of constraint",
  "decision_constraint_type_qid": "Q52558054"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Clemens Dulcis",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q458"
  ],
  "old_value_descriptions_en": [
    "political and economic union of 27 European states"
  ],
  "old_value_labels_en": [
    "European Union"
  ],
  "revision_id": 2444344191,
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
      "Q458": 1
    },
    "old_unique": [
      "Q458"
    ],
    "old_values": [
      "Q458"
    ],
    "old_values_raw": [
      "Q458"
    ],
    "removed_unique_values": [
      "Q458"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Q458": {
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
  "report_fix_date": "2025-12-21T16:32:40",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2445031042,
  "report_revision_old": 2444558160,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": [
    "Q458"
  ],
  "value_descriptions_en": [
    "political and economic union of 27 European states"
  ],
  "value_labels_en": [
    "European Union"
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
    "description": "1988 film award ceremony on the 26th of November at the Theater des Westens in West Berlin, Germany",
    "label": "1st European Film Awards"
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
      "delete_reason": "set_membership",
      "known_set": [
        "Q108151641",
        "Q113403829",
        "Q11772",
        "Q14773",
        "Q15132899",
        "Q1657833",
        "Q17252",
        "Q2184",
        "Q219817",
        "Q219825",
        "Q221457",
        "Q22890",
        "Q23334",
        "Q2454900",
        "Q25594375",
        "Q29520",
        "Q42620",
        "Q458",
        "Q47588",
        "Q4843341",
        "Q518617",
        "Q5481",
        "Q5705",
        "Q5710",
        "... omitted 12 items"
      ],
      "removed_values": [
        "Q458"
      ],
      "report_type": "none of"
    },
    "result": "SET_MEMBERSHIP_REJECTION",
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

## 011. `repair_Q636262_2441757834`

| Field | Value |
|---|---|
| qid | Q636262 |
| property | P421 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | none_of |
| decision_constraint_type | Q52558054 none-of constraint |
| group_key | ABOX::Q636262::P421 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q6655", "Q6723"] |
| classification_target_tokens | ["Q36669"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q36669"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q36669"
  ],
  "removed_unique_values": [
    "Q36669"
  ],
  "retained_support_tokens": [
    "Q6655",
    "Q6723"
  ],
  "retained_unique_values": [
    "Q6655",
    "Q6723"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "none_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "none-of constraint",
  "decision_constraint_type_qid": "Q52558054"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q6655",
    "Q6723"
  ],
  "new_value_descriptions_en": [
    "identifier for a time offset from UTC of +1",
    "identifier for a time offset from UTC of +2"
  ],
  "new_value_labels_en": [
    "UTC+01:00",
    "UTC+02:00"
  ],
  "old_value": [
    "Q6655",
    "Q36669",
    "Q6723"
  ],
  "old_value_descriptions_en": [
    "identifier for a time offset from UTC of +1",
    "seasonal adjustment of clocks",
    "identifier for a time offset from UTC of +2"
  ],
  "old_value_labels_en": [
    "UTC+01:00",
    "daylight saving time",
    "UTC+02:00"
  ],
  "revision_id": 2441757834,
  "value": [
    "Q6655",
    "Q6723"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q6655": 1,
      "Q6723": 1
    },
    "new_unique": [
      "Q6655",
      "Q6723"
    ],
    "new_values": [
      "Q6655",
      "Q6723"
    ],
    "new_values_raw": [
      "Q6655",
      "Q6723"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q36669": 1,
      "Q6655": 1,
      "Q6723": 1
    },
    "old_unique": [
      "Q36669",
      "Q6655",
      "Q6723"
    ],
    "old_values": [
      "Q6655",
      "Q36669",
      "Q6723"
    ],
    "old_values_raw": [
      "Q6655",
      "Q36669",
      "Q6723"
    ],
    "removed_unique_values": [
      "Q36669"
    ],
    "retained_unique_values": [
      "Q6655",
      "Q6723"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q36669": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "identifier for a time offset from UTC of +1",
    "identifier for a time offset from UTC of +2"
  ],
  "value_labels_en": [
    "UTC+01:00",
    "UTC+02:00"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T12:17:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P421",
  "report_revision_new": 2442278494,
  "report_revision_old": 2441776312,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "report_violation_types": [
    "None of",
    "Value type Q|12143"
  ],
  "value": [
    "Q6655",
    "Q36669",
    "Q6723"
  ],
  "value_descriptions_en": [
    "identifier for a time offset from UTC of +1",
    "seasonal adjustment of clocks",
    "identifier for a time offset from UTC of +2"
  ],
  "value_labels_en": [
    "UTC+01:00",
    "daylight saving time",
    "UTC+02:00"
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
  "truth_tokens": [
    "Q6655",
    "Q6723"
  ],
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
    "description": "commune in Isère, France",
    "label": "Vizille"
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

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q6655": 1,
        "Q6723": 1
      },
      "new_unique": [
        "Q6655",
        "Q6723"
      ],
      "new_values": [
        "Q6655",
        "Q6723"
      ],
      "new_values_raw": [
        "Q6655",
        "Q6723"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q36669": 1,
        "Q6655": 1,
        "Q6723": 1
      },
      "old_unique": [
        "Q36669",
        "Q6655",
        "Q6723"
      ],
      "old_values": [
        "Q6655",
        "Q36669",
        "Q6723"
      ],
      "old_values_raw": [
        "Q6655",
        "Q36669",
        "Q6723"
      ],
      "removed_unique_values": [
        "Q36669"
      ],
      "retained_unique_values": [
        "Q6655",
        "Q6723"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q36669": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q36669"
      ],
      "removed_values": [
        "Q36669"
      ],
      "report_type": "none of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 012. `repair_Q63896481_2333526281`

| Field | Value |
|---|---|
| qid | Q63896481 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | mid |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q63896481::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99868032"] |
| classification_target_tokens | ["Q59496158"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q59496158"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q59496158"
  ],
  "removed_unique_values": [
    "Q59496158"
  ],
  "retained_support_tokens": [
    "Q99868032"
  ],
  "retained_unique_values": [
    "Q99868032"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99868032"
  ],
  "new_value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "undetermined use restriction"
  ],
  "old_value": [
    "Q59496158",
    "Q99868032"
  ],
  "old_value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
  ],
  "revision_id": 2333526281,
  "value": [
    "Q99868032"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99868032": 1
    },
    "new_unique": [
      "Q99868032"
    ],
    "new_values": [
      "Q99868032"
    ],
    "new_values_raw": [
      "Q99868032"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q59496158": 1,
      "Q99868032": 1
    },
    "old_unique": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values_raw": [
      "Q59496158",
      "Q99868032"
    ],
    "removed_unique_values": [
      "Q59496158"
    ],
    "retained_unique_values": [
      "Q99868032"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q59496158": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "undetermined use restriction"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-03T16:10:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334257368,
  "report_revision_old": 2324105015,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q59496158",
    "Q99868032"
  ],
  "value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
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
  "truth_tokens": [
    "Q99868032"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "series in the National Archives and Records Administration's holdings",
    "label": "Alan Hoffman's Files (NAID 68123414)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99868032": 1
      },
      "new_unique": [
        "Q99868032"
      ],
      "new_values": [
        "Q99868032"
      ],
      "new_values_raw": [
        "Q99868032"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59496158": 1,
        "Q99868032": 1
      },
      "old_unique": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values_raw": [
        "Q59496158",
        "Q99868032"
      ],
      "removed_unique_values": [
        "Q59496158"
      ],
      "retained_unique_values": [
        "Q99868032"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q59496158": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q59496158"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 013. `repair_Q63897137_2333529722`

| Field | Value |
|---|---|
| qid | Q63897137 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | mid |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q63897137::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99868032"] |
| classification_target_tokens | ["Q59496158"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q59496158"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q59496158"
  ],
  "removed_unique_values": [
    "Q59496158"
  ],
  "retained_support_tokens": [
    "Q99868032"
  ],
  "retained_unique_values": [
    "Q99868032"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99868032"
  ],
  "new_value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "undetermined use restriction"
  ],
  "old_value": [
    "Q59496158",
    "Q99868032"
  ],
  "old_value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
  ],
  "revision_id": 2333529722,
  "value": [
    "Q99868032"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99868032": 1
    },
    "new_unique": [
      "Q99868032"
    ],
    "new_values": [
      "Q99868032"
    ],
    "new_values_raw": [
      "Q99868032"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q59496158": 1,
      "Q99868032": 1
    },
    "old_unique": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values_raw": [
      "Q59496158",
      "Q99868032"
    ],
    "removed_unique_values": [
      "Q59496158"
    ],
    "retained_unique_values": [
      "Q99868032"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q59496158": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "undetermined use restriction"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-03T16:10:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334257368,
  "report_revision_old": 2324105015,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q59496158",
    "Q99868032"
  ],
  "value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
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
  "truth_tokens": [
    "Q99868032"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "series in the National Archives and Records Administration's holdings",
    "label": "Nancy Clark's and Jon Glassman's Chronological Files (NAID 2579438)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99868032": 1
      },
      "new_unique": [
        "Q99868032"
      ],
      "new_values": [
        "Q99868032"
      ],
      "new_values_raw": [
        "Q99868032"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59496158": 1,
        "Q99868032": 1
      },
      "old_unique": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values_raw": [
        "Q59496158",
        "Q99868032"
      ],
      "removed_unique_values": [
        "Q59496158"
      ],
      "retained_unique_values": [
        "Q99868032"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q59496158": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q59496158"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 014. `repair_Q63898673_2333535407`

| Field | Value |
|---|---|
| qid | Q63898673 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | mid |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q63898673::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99868032"] |
| classification_target_tokens | ["Q59496158"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q59496158"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q59496158"
  ],
  "removed_unique_values": [
    "Q59496158"
  ],
  "retained_support_tokens": [
    "Q99868032"
  ],
  "retained_unique_values": [
    "Q99868032"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99868032"
  ],
  "new_value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "undetermined use restriction"
  ],
  "old_value": [
    "Q59496158",
    "Q99868032"
  ],
  "old_value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
  ],
  "revision_id": 2333535407,
  "value": [
    "Q99868032"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99868032": 1
    },
    "new_unique": [
      "Q99868032"
    ],
    "new_values": [
      "Q99868032"
    ],
    "new_values_raw": [
      "Q99868032"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q59496158": 1,
      "Q99868032": 1
    },
    "old_unique": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values_raw": [
      "Q59496158",
      "Q99868032"
    ],
    "removed_unique_values": [
      "Q59496158"
    ],
    "retained_unique_values": [
      "Q99868032"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q59496158": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "undetermined use restriction"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-03T16:10:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334257368,
  "report_revision_old": 2324105015,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q59496158",
    "Q99868032"
  ],
  "value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
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
  "truth_tokens": [
    "Q99868032"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "series in the National Archives and Records Administration's holdings",
    "label": "Melissa Solsbury's Subject Files (NAID 83860087)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99868032": 1
      },
      "new_unique": [
        "Q99868032"
      ],
      "new_values": [
        "Q99868032"
      ],
      "new_values_raw": [
        "Q99868032"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59496158": 1,
        "Q99868032": 1
      },
      "old_unique": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values_raw": [
        "Q59496158",
        "Q99868032"
      ],
      "removed_unique_values": [
        "Q59496158"
      ],
      "retained_unique_values": [
        "Q99868032"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q59496158": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q59496158"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 015. `repair_Q63904494_2333564594`

| Field | Value |
|---|---|
| qid | Q63904494 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | mid |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q63904494::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99868032"] |
| classification_target_tokens | ["Q59496158"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q59496158"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q59496158"
  ],
  "removed_unique_values": [
    "Q59496158"
  ],
  "retained_support_tokens": [
    "Q99868032"
  ],
  "retained_unique_values": [
    "Q99868032"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99868032"
  ],
  "new_value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "undetermined use restriction"
  ],
  "old_value": [
    "Q59496158",
    "Q99868032"
  ],
  "old_value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
  ],
  "revision_id": 2333564594,
  "value": [
    "Q99868032"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99868032": 1
    },
    "new_unique": [
      "Q99868032"
    ],
    "new_values": [
      "Q99868032"
    ],
    "new_values_raw": [
      "Q99868032"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q59496158": 1,
      "Q99868032": 1
    },
    "old_unique": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values_raw": [
      "Q59496158",
      "Q99868032"
    ],
    "removed_unique_values": [
      "Q59496158"
    ],
    "retained_unique_values": [
      "Q99868032"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q59496158": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "undetermined use restriction"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-03T16:10:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334257368,
  "report_revision_old": 2324105015,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q59496158",
    "Q99868032"
  ],
  "value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
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
  "truth_tokens": [
    "Q99868032"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "series in the National Archives and Records Administration's holdings",
    "label": "Subject Files on Meetings and Conferences (NAID 5730618)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99868032": 1
      },
      "new_unique": [
        "Q99868032"
      ],
      "new_values": [
        "Q99868032"
      ],
      "new_values_raw": [
        "Q99868032"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59496158": 1,
        "Q99868032": 1
      },
      "old_unique": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values_raw": [
        "Q59496158",
        "Q99868032"
      ],
      "removed_unique_values": [
        "Q59496158"
      ],
      "retained_unique_values": [
        "Q99868032"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q59496158": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q59496158"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 016. `repair_Q63904593_2333565369`

| Field | Value |
|---|---|
| qid | Q63904593 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | mid |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q63904593::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99868032"] |
| classification_target_tokens | ["Q59496158"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q59496158"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q59496158"
  ],
  "removed_unique_values": [
    "Q59496158"
  ],
  "retained_support_tokens": [
    "Q99868032"
  ],
  "retained_unique_values": [
    "Q99868032"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99868032"
  ],
  "new_value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "undetermined use restriction"
  ],
  "old_value": [
    "Q59496158",
    "Q99868032"
  ],
  "old_value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
  ],
  "revision_id": 2333565369,
  "value": [
    "Q99868032"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99868032": 1
    },
    "new_unique": [
      "Q99868032"
    ],
    "new_values": [
      "Q99868032"
    ],
    "new_values_raw": [
      "Q99868032"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q59496158": 1,
      "Q99868032": 1
    },
    "old_unique": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values_raw": [
      "Q59496158",
      "Q99868032"
    ],
    "removed_unique_values": [
      "Q59496158"
    ],
    "retained_unique_values": [
      "Q99868032"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q59496158": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "undetermined use restriction"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-03T16:10:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334257368,
  "report_revision_old": 2324105015,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q59496158",
    "Q99868032"
  ],
  "value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
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
  "truth_tokens": [
    "Q99868032"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "series in the National Archives and Records Administration's holdings",
    "label": "Subject Files on Real Property (NAID 5730643)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99868032": 1
      },
      "new_unique": [
        "Q99868032"
      ],
      "new_values": [
        "Q99868032"
      ],
      "new_values_raw": [
        "Q99868032"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59496158": 1,
        "Q99868032": 1
      },
      "old_unique": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values_raw": [
        "Q59496158",
        "Q99868032"
      ],
      "removed_unique_values": [
        "Q59496158"
      ],
      "retained_unique_values": [
        "Q99868032"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q59496158": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q59496158"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 017. `repair_Q63904849_2333513671`

| Field | Value |
|---|---|
| qid | Q63904849 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | tail |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q63904849::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99867894"] |
| classification_target_tokens | ["Q66739849"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q66739849"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q66739849"
  ],
  "removed_unique_values": [
    "Q66739849"
  ],
  "retained_support_tokens": [
    "Q99867894"
  ],
  "retained_unique_values": [
    "Q99867894"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99867894"
  ],
  "new_value_descriptions_en": [
    "some of the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "partly restricted use"
  ],
  "old_value": [
    "Q66739849",
    "Q99867894"
  ],
  "old_value_descriptions_en": [
    "access restriction status",
    "some of the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "partly restricted access",
    "partly restricted use"
  ],
  "revision_id": 2333513671,
  "value": [
    "Q99867894"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99867894": 1
    },
    "new_unique": [
      "Q99867894"
    ],
    "new_values": [
      "Q99867894"
    ],
    "new_values_raw": [
      "Q99867894"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q66739849": 1,
      "Q99867894": 1
    },
    "old_unique": [
      "Q66739849",
      "Q99867894"
    ],
    "old_values": [
      "Q66739849",
      "Q99867894"
    ],
    "old_values_raw": [
      "Q66739849",
      "Q99867894"
    ],
    "removed_unique_values": [
      "Q66739849"
    ],
    "retained_unique_values": [
      "Q99867894"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q66739849": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "some of the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "partly restricted use"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-03T16:10:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334257368,
  "report_revision_old": 2324105015,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q66739849",
    "Q99867894"
  ],
  "value_descriptions_en": [
    "access restriction status",
    "some of the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "partly restricted access",
    "partly restricted use"
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
  "truth_tokens": [
    "Q99867894"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "series in the National Archives and Records Administration's holdings",
    "label": "White House Subject Files on Invitations (NAID 591582)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99867894": 1
      },
      "new_unique": [
        "Q99867894"
      ],
      "new_values": [
        "Q99867894"
      ],
      "new_values_raw": [
        "Q99867894"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q66739849": 1,
        "Q99867894": 1
      },
      "old_unique": [
        "Q66739849",
        "Q99867894"
      ],
      "old_values": [
        "Q66739849",
        "Q99867894"
      ],
      "old_values_raw": [
        "Q66739849",
        "Q99867894"
      ],
      "removed_unique_values": [
        "Q66739849"
      ],
      "retained_unique_values": [
        "Q99867894"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q66739849": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q66739849"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 018. `repair_Q63906101_2333573477`

| Field | Value |
|---|---|
| qid | Q63906101 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | mid |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q63906101::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99868032"] |
| classification_target_tokens | ["Q59496158"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q59496158"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q59496158"
  ],
  "removed_unique_values": [
    "Q59496158"
  ],
  "retained_support_tokens": [
    "Q99868032"
  ],
  "retained_unique_values": [
    "Q99868032"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99868032"
  ],
  "new_value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "undetermined use restriction"
  ],
  "old_value": [
    "Q59496158",
    "Q99868032"
  ],
  "old_value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
  ],
  "revision_id": 2333573477,
  "value": [
    "Q99868032"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99868032": 1
    },
    "new_unique": [
      "Q99868032"
    ],
    "new_values": [
      "Q99868032"
    ],
    "new_values_raw": [
      "Q99868032"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q59496158": 1,
      "Q99868032": 1
    },
    "old_unique": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values_raw": [
      "Q59496158",
      "Q99868032"
    ],
    "removed_unique_values": [
      "Q59496158"
    ],
    "retained_unique_values": [
      "Q99868032"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q59496158": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "undetermined use restriction"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-03T16:10:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334257368,
  "report_revision_old": 2324105015,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q59496158",
    "Q99868032"
  ],
  "value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
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
  "truth_tokens": [
    "Q99868032"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "series in the National Archives and Records Administration's holdings",
    "label": "Sydney Johnson's Files (NAID 60061771)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99868032": 1
      },
      "new_unique": [
        "Q99868032"
      ],
      "new_values": [
        "Q99868032"
      ],
      "new_values_raw": [
        "Q99868032"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59496158": 1,
        "Q99868032": 1
      },
      "old_unique": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values_raw": [
        "Q59496158",
        "Q99868032"
      ],
      "removed_unique_values": [
        "Q59496158"
      ],
      "retained_unique_values": [
        "Q99868032"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q59496158": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q59496158"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 019. `repair_Q63906311_2333514581`

| Field | Value |
|---|---|
| qid | Q63906311 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | tail |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q63906311::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99867894"] |
| classification_target_tokens | ["Q66739849"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q66739849"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q66739849"
  ],
  "removed_unique_values": [
    "Q66739849"
  ],
  "retained_support_tokens": [
    "Q99867894"
  ],
  "retained_unique_values": [
    "Q99867894"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99867894"
  ],
  "new_value_descriptions_en": [
    "some of the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "partly restricted use"
  ],
  "old_value": [
    "Q66739849",
    "Q99867894"
  ],
  "old_value_descriptions_en": [
    "access restriction status",
    "some of the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "partly restricted access",
    "partly restricted use"
  ],
  "revision_id": 2333514581,
  "value": [
    "Q99867894"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99867894": 1
    },
    "new_unique": [
      "Q99867894"
    ],
    "new_values": [
      "Q99867894"
    ],
    "new_values_raw": [
      "Q99867894"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q66739849": 1,
      "Q99867894": 1
    },
    "old_unique": [
      "Q66739849",
      "Q99867894"
    ],
    "old_values": [
      "Q66739849",
      "Q99867894"
    ],
    "old_values_raw": [
      "Q66739849",
      "Q99867894"
    ],
    "removed_unique_values": [
      "Q66739849"
    ],
    "retained_unique_values": [
      "Q99867894"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q66739849": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "some of the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "partly restricted use"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-03T16:10:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334257368,
  "report_revision_old": 2324105015,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q66739849",
    "Q99867894"
  ],
  "value_descriptions_en": [
    "access restriction status",
    "some of the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "partly restricted access",
    "partly restricted use"
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
  "truth_tokens": [
    "Q99867894"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "series in the National Archives and Records Administration's holdings",
    "label": "Files from the Georgetown Office and Residence (NAID 577110)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99867894": 1
      },
      "new_unique": [
        "Q99867894"
      ],
      "new_values": [
        "Q99867894"
      ],
      "new_values_raw": [
        "Q99867894"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q66739849": 1,
        "Q99867894": 1
      },
      "old_unique": [
        "Q66739849",
        "Q99867894"
      ],
      "old_values": [
        "Q66739849",
        "Q99867894"
      ],
      "old_values_raw": [
        "Q66739849",
        "Q99867894"
      ],
      "removed_unique_values": [
        "Q66739849"
      ],
      "retained_unique_values": [
        "Q99867894"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q66739849": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q66739849"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 020. `repair_Q66351775_2333788345`

| Field | Value |
|---|---|
| qid | Q66351775 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | tail |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q66351775::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99868032"] |
| classification_target_tokens | ["Q59496158"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q59496158"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q59496158"
  ],
  "removed_unique_values": [
    "Q59496158"
  ],
  "retained_support_tokens": [
    "Q99868032"
  ],
  "retained_unique_values": [
    "Q99868032"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99868032"
  ],
  "new_value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "undetermined use restriction"
  ],
  "old_value": [
    "Q59496158",
    "Q99868032"
  ],
  "old_value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
  ],
  "revision_id": 2333788345,
  "value": [
    "Q99868032"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99868032": 1
    },
    "new_unique": [
      "Q99868032"
    ],
    "new_values": [
      "Q99868032"
    ],
    "new_values_raw": [
      "Q99868032"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q59496158": 1,
      "Q99868032": 1
    },
    "old_unique": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values_raw": [
      "Q59496158",
      "Q99868032"
    ],
    "removed_unique_values": [
      "Q59496158"
    ],
    "retained_unique_values": [
      "Q99868032"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q59496158": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "undetermined use restriction"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-04T06:32:45",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334427348,
  "report_revision_old": 2334257368,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q59496158",
    "Q99868032"
  ],
  "value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
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
  "truth_tokens": [
    "Q99868032"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "item in the National Archives and Records Administration's holdings",
    "label": "100TH U.S. CONGRESS, 1ST SESSION, 1987, SENATE. PRDDAT=21 Apr 1987 (NAID 8101)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99868032": 1
      },
      "new_unique": [
        "Q99868032"
      ],
      "new_values": [
        "Q99868032"
      ],
      "new_values_raw": [
        "Q99868032"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59496158": 1,
        "Q99868032": 1
      },
      "old_unique": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values_raw": [
        "Q59496158",
        "Q99868032"
      ],
      "removed_unique_values": [
        "Q59496158"
      ],
      "retained_unique_values": [
        "Q99868032"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q59496158": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q59496158"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 021. `repair_Q66352015_2333792188`

| Field | Value |
|---|---|
| qid | Q66352015 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | tail |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q66352015::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99868032"] |
| classification_target_tokens | ["Q59496158"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q59496158"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q59496158"
  ],
  "removed_unique_values": [
    "Q59496158"
  ],
  "retained_support_tokens": [
    "Q99868032"
  ],
  "retained_unique_values": [
    "Q99868032"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99868032"
  ],
  "new_value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "undetermined use restriction"
  ],
  "old_value": [
    "Q59496158",
    "Q99868032"
  ],
  "old_value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
  ],
  "revision_id": 2333792188,
  "value": [
    "Q99868032"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99868032": 1
    },
    "new_unique": [
      "Q99868032"
    ],
    "new_values": [
      "Q99868032"
    ],
    "new_values_raw": [
      "Q99868032"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q59496158": 1,
      "Q99868032": 1
    },
    "old_unique": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values_raw": [
      "Q59496158",
      "Q99868032"
    ],
    "removed_unique_values": [
      "Q59496158"
    ],
    "retained_unique_values": [
      "Q99868032"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q59496158": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "undetermined use restriction"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-04T06:32:45",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334427348,
  "report_revision_old": 2334257368,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q59496158",
    "Q99868032"
  ],
  "value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
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
  "truth_tokens": [
    "Q99868032"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "item in the National Archives and Records Administration's holdings",
    "label": "100TH U.S. CONGRESS, 1ST SESSION, 1987, SENATE, STATE OF THEUNION. PRDDAT=27 Jan 1987 (NAID 7870)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99868032": 1
      },
      "new_unique": [
        "Q99868032"
      ],
      "new_values": [
        "Q99868032"
      ],
      "new_values_raw": [
        "Q99868032"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59496158": 1,
        "Q99868032": 1
      },
      "old_unique": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values_raw": [
        "Q59496158",
        "Q99868032"
      ],
      "removed_unique_values": [
        "Q59496158"
      ],
      "retained_unique_values": [
        "Q99868032"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q59496158": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q59496158"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 022. `repair_Q67405708_2333805645`

| Field | Value |
|---|---|
| qid | Q67405708 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | tail |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q67405708::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99868032"] |
| classification_target_tokens | ["Q59496158"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q59496158"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q59496158"
  ],
  "removed_unique_values": [
    "Q59496158"
  ],
  "retained_support_tokens": [
    "Q99868032"
  ],
  "retained_unique_values": [
    "Q99868032"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99868032"
  ],
  "new_value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "undetermined use restriction"
  ],
  "old_value": [
    "Q59496158",
    "Q99868032"
  ],
  "old_value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
  ],
  "revision_id": 2333805645,
  "value": [
    "Q99868032"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99868032": 1
    },
    "new_unique": [
      "Q99868032"
    ],
    "new_values": [
      "Q99868032"
    ],
    "new_values_raw": [
      "Q99868032"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q59496158": 1,
      "Q99868032": 1
    },
    "old_unique": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values_raw": [
      "Q59496158",
      "Q99868032"
    ],
    "removed_unique_values": [
      "Q59496158"
    ],
    "retained_unique_values": [
      "Q99868032"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q59496158": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "undetermined use restriction"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-04T06:32:45",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334427348,
  "report_revision_old": 2334257368,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q59496158",
    "Q99868032"
  ],
  "value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
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
  "truth_tokens": [
    "Q99868032"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "item in the National Archives and Records Administration's holdings",
    "label": "DECISION AT DELANO (NAID 100493)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99868032": 1
      },
      "new_unique": [
        "Q99868032"
      ],
      "new_values": [
        "Q99868032"
      ],
      "new_values_raw": [
        "Q99868032"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59496158": 1,
        "Q99868032": 1
      },
      "old_unique": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values_raw": [
        "Q59496158",
        "Q99868032"
      ],
      "removed_unique_values": [
        "Q59496158"
      ],
      "retained_unique_values": [
        "Q99868032"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q59496158": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q59496158"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 023. `repair_Q67406069_2333806631`

| Field | Value |
|---|---|
| qid | Q67406069 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | tail |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q67406069::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99868032"] |
| classification_target_tokens | ["Q59496158"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q59496158"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q59496158"
  ],
  "removed_unique_values": [
    "Q59496158"
  ],
  "retained_support_tokens": [
    "Q99868032"
  ],
  "retained_unique_values": [
    "Q99868032"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99868032"
  ],
  "new_value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "undetermined use restriction"
  ],
  "old_value": [
    "Q59496158",
    "Q99868032"
  ],
  "old_value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
  ],
  "revision_id": 2333806631,
  "value": [
    "Q99868032"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99868032": 1
    },
    "new_unique": [
      "Q99868032"
    ],
    "new_values": [
      "Q99868032"
    ],
    "new_values_raw": [
      "Q99868032"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q59496158": 1,
      "Q99868032": 1
    },
    "old_unique": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values_raw": [
      "Q59496158",
      "Q99868032"
    ],
    "removed_unique_values": [
      "Q59496158"
    ],
    "retained_unique_values": [
      "Q99868032"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q59496158": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "undetermined use restriction"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-04T06:32:45",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334427348,
  "report_revision_old": 2334257368,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q59496158",
    "Q99868032"
  ],
  "value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
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
  "truth_tokens": [
    "Q99868032"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "item in the National Archives and Records Administration's holdings",
    "label": "101ST U.S. CONGRESS, 1ST SESSION, 1989, SENATE (NAID 10057)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99868032": 1
      },
      "new_unique": [
        "Q99868032"
      ],
      "new_values": [
        "Q99868032"
      ],
      "new_values_raw": [
        "Q99868032"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59496158": 1,
        "Q99868032": 1
      },
      "old_unique": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values_raw": [
        "Q59496158",
        "Q99868032"
      ],
      "removed_unique_values": [
        "Q59496158"
      ],
      "retained_unique_values": [
        "Q99868032"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q59496158": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q59496158"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 024. `repair_Q67409192_2333809678`

| Field | Value |
|---|---|
| qid | Q67409192 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | tail |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q67409192::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99868032"] |
| classification_target_tokens | ["Q59496158"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q59496158"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q59496158"
  ],
  "removed_unique_values": [
    "Q59496158"
  ],
  "retained_support_tokens": [
    "Q99868032"
  ],
  "retained_unique_values": [
    "Q99868032"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99868032"
  ],
  "new_value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "undetermined use restriction"
  ],
  "old_value": [
    "Q59496158",
    "Q99868032"
  ],
  "old_value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
  ],
  "revision_id": 2333809678,
  "value": [
    "Q99868032"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99868032": 1
    },
    "new_unique": [
      "Q99868032"
    ],
    "new_values": [
      "Q99868032"
    ],
    "new_values_raw": [
      "Q99868032"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q59496158": 1,
      "Q99868032": 1
    },
    "old_unique": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values_raw": [
      "Q59496158",
      "Q99868032"
    ],
    "removed_unique_values": [
      "Q59496158"
    ],
    "retained_unique_values": [
      "Q99868032"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q59496158": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "undetermined use restriction"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-04T06:32:45",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334427348,
  "report_revision_old": 2334257368,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q59496158",
    "Q99868032"
  ],
  "value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
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
  "truth_tokens": [
    "Q99868032"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "item in the National Archives and Records Administration's holdings",
    "label": "CBS SATURDAY EVENING NEWS (NAID 101098)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99868032": 1
      },
      "new_unique": [
        "Q99868032"
      ],
      "new_values": [
        "Q99868032"
      ],
      "new_values_raw": [
        "Q99868032"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59496158": 1,
        "Q99868032": 1
      },
      "old_unique": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values_raw": [
        "Q59496158",
        "Q99868032"
      ],
      "removed_unique_values": [
        "Q59496158"
      ],
      "retained_unique_values": [
        "Q99868032"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q59496158": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q59496158"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 025. `repair_Q67411422_2333813782`

| Field | Value |
|---|---|
| qid | Q67411422 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | mid |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q67411422::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99868032"] |
| classification_target_tokens | ["Q59496158"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q59496158"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q59496158"
  ],
  "removed_unique_values": [
    "Q59496158"
  ],
  "retained_support_tokens": [
    "Q99868032"
  ],
  "retained_unique_values": [
    "Q99868032"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99868032"
  ],
  "new_value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "undetermined use restriction"
  ],
  "old_value": [
    "Q59496158",
    "Q99868032"
  ],
  "old_value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
  ],
  "revision_id": 2333813782,
  "value": [
    "Q99868032"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99868032": 1
    },
    "new_unique": [
      "Q99868032"
    ],
    "new_values": [
      "Q99868032"
    ],
    "new_values_raw": [
      "Q99868032"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q59496158": 1,
      "Q99868032": 1
    },
    "old_unique": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values_raw": [
      "Q59496158",
      "Q99868032"
    ],
    "removed_unique_values": [
      "Q59496158"
    ],
    "retained_unique_values": [
      "Q99868032"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q59496158": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "undetermined use restriction"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-04T06:32:45",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334427348,
  "report_revision_old": 2334257368,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q59496158",
    "Q99868032"
  ],
  "value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
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
  "truth_tokens": [
    "Q99868032"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "item in the National Archives and Records Administration's holdings",
    "label": "SECRETARY JUANITA KREPS NEWS CONFERENCE-TOWN HALL OFCALIFORNIA (NAID 101470)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99868032": 1
      },
      "new_unique": [
        "Q99868032"
      ],
      "new_values": [
        "Q99868032"
      ],
      "new_values_raw": [
        "Q99868032"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59496158": 1,
        "Q99868032": 1
      },
      "old_unique": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values_raw": [
        "Q59496158",
        "Q99868032"
      ],
      "removed_unique_values": [
        "Q59496158"
      ],
      "retained_unique_values": [
        "Q99868032"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q59496158": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q59496158"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---

## 026. `repair_Q67512614_2333830506`

| Field | Value |
|---|---|
| qid | Q67512614 |
| property | P7261 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SET_MEMBERSHIP_REJECTION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_set_membership_rejection |
| popularity_bucket | mid |
| constraint_family | Q21510859 |
| classification_rule_family | set_membership |
| classification_rule_subfamily | one_of |
| decision_constraint_type | Q21510859 one-of constraint |
| group_key | ABOX::Q67512614::P7261 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the removed value is directly ruled out by a set-membership constraint.
- If choosing the removed value requires local or external evidence, mark overclaimed / needs evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q99868032"] |
| classification_target_tokens | ["Q59496158"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | set_membership_rejection |
| rationale | Subset repair removes a value proven invalid by parsed one-of/none-of constraints. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q59496158"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q59496158"
  ],
  "removed_unique_values": [
    "Q59496158"
  ],
  "retained_support_tokens": [
    "Q99868032"
  ],
  "retained_unique_values": [
    "Q99868032"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "set_membership",
  "classification_rule_subfamily": "one_of",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "one-of constraint",
  "decision_constraint_type_qid": "Q21510859"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "Q99868032"
  ],
  "new_value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "new_value_labels_en": [
    "undetermined use restriction"
  ],
  "old_value": [
    "Q59496158",
    "Q99868032"
  ],
  "old_value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "old_value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
  ],
  "revision_id": 2333830506,
  "value": [
    "Q99868032"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q99868032": 1
    },
    "new_unique": [
      "Q99868032"
    ],
    "new_values": [
      "Q99868032"
    ],
    "new_values_raw": [
      "Q99868032"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q59496158": 1,
      "Q99868032": 1
    },
    "old_unique": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values": [
      "Q59496158",
      "Q99868032"
    ],
    "old_values_raw": [
      "Q59496158",
      "Q99868032"
    ],
    "removed_unique_values": [
      "Q59496158"
    ],
    "retained_unique_values": [
      "Q99868032"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q59496158": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "undetermined use restriction"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-04T06:32:45",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
  "report_revision_new": 2334427348,
  "report_revision_old": 2334257368,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q59496158",
    "Q99868032"
  ],
  "value_descriptions_en": [
    "entity for which the value has not been determined yet (for \"cannot be determined\" use unknown/Q24238356)",
    "unknown if the archival materials have a use restriction"
  ],
  "value_labels_en": [
    "not yet determined",
    "undetermined use restriction"
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
  "truth_tokens": [
    "Q99868032"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "status of any use restrictions on the object, collection, or materials",
    "label": "use restriction status"
  },
  "qid": {
    "description": "item in the National Archives and Records Administration's holdings",
    "label": "SUBSEQUENT TRIAL INTERROGATIONS OF FLUEGGE, W. VON (VW) (NAID 102590)"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q99868032": 1
      },
      "new_unique": [
        "Q99868032"
      ],
      "new_values": [
        "Q99868032"
      ],
      "new_values_raw": [
        "Q99868032"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59496158": 1,
        "Q99868032": 1
      },
      "old_unique": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values": [
        "Q59496158",
        "Q99868032"
      ],
      "old_values_raw": [
        "Q59496158",
        "Q99868032"
      ],
      "removed_unique_values": [
        "Q59496158"
      ],
      "retained_unique_values": [
        "Q99868032"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q59496158": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "known_set": [
        "Q99867853",
        "Q99867894",
        "Q99867969",
        "Q99868032",
        "Q99868068"
      ],
      "removed_values": [
        "Q59496158"
      ],
      "report_type": "one of"
    },
    "kind": "SET_MEMBERSHIP_REJECTION",
    "result": true,
    "step": "rule_deterministic"
  },
  {
    "result": null,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "set_membership_rejection",
    "step": "branch"
  }
]
```

---
