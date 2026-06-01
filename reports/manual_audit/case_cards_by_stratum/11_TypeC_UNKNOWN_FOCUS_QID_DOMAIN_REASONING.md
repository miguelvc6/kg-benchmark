# TypeC_UNKNOWN_FOCUS_QID_DOMAIN_REASONING

Cases: 8

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q135499020_2388486490`

| Field | Value |
|---|---|
| qid | Q135499020 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_FOCUS_QID_DOMAIN_REASONING / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_focus_qid_domain_reasoning |
| decision_constraint_type |   |
| group_key | ABOX::Q135499020::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135499020"] |
| classification_target_tokens | ["Q135499020"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | unknown_focus_qid_domain_reasoning |
| rationale | Focus QID is locally available, but local identity alone does not justify asserting this property value; correctness requires a domain/property-specific rule or external/domain reasoning. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135499020"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135499020"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "CREATE_FROM_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_focus_qid_domain_reasoning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q135499020"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "11027"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2388486490,
  "value": [
    "Q135499020"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135499020"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135499020": 1
    },
    "new_unique": [
      "Q135499020"
    ],
    "new_values": [
      "Q135499020"
    ],
    "new_values_raw": [
      "Q135499020"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {},
    "old_unique": [],
    "old_values": [],
    "old_values_raw": [
      "MISSING"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [],
    "semantic_action": "CREATE_FROM_MISSING",
    "value_multiplicity_changes": {
      "Q135499020": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "11027"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-10T06:26:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2389998295,
  "report_revision_old": 2388961222,
  "report_violation_type": "Value type Q|49008",
  "report_violation_type_descriptions_en": [
    "positive integer with exactly two divisors, 1 and itself"
  ],
  "report_violation_type_labels_en": [
    "prime number"
  ],
  "report_violation_type_normalized": "Value type Q|49008",
  "report_violation_type_qids": [
    "Q49008"
  ],
  "report_violation_type_raw": "Value type Q|49008",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 15,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135499020"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q135499020"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135499020"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "one of the prime numbers that can be multiplied to give this number",
    "label": "prime factor"
  },
  "qid": {
    "description": "natural number",
    "label": "11027"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Q135499020"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135499020": 1
      },
      "new_unique": [
        "Q135499020"
      ],
      "new_values": [
        "Q135499020"
      ],
      "new_values_raw": [
        "Q135499020"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {},
      "old_unique": [],
      "old_values": [],
      "old_values_raw": [
        "MISSING"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [],
      "semantic_action": "CREATE_FROM_MISSING",
      "value_multiplicity_changes": {
        "Q135499020": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "CREATE_FROM_MISSING",
    "step": "value_delta"
  },
  {
    "detail": {
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 1,
      "independent_match_count": 1,
      "local_ids_count": 15,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135499020"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_QID"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "MISSING"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_focus_qid_domain_reasoning",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q135500748_2388530711`

| Field | Value |
|---|---|
| qid | Q135500748 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_FOCUS_QID_DOMAIN_REASONING / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_focus_qid_domain_reasoning |
| decision_constraint_type |   |
| group_key | ABOX::Q135500748::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135500748"] |
| classification_target_tokens | ["Q135500748"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | unknown_focus_qid_domain_reasoning |
| rationale | Focus QID is locally available, but local identity alone does not justify asserting this property value; correctness requires a domain/property-specific rule or external/domain reasoning. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135500748"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135500748"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "CREATE_FROM_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_focus_qid_domain_reasoning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q135500748"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "12289"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2388530711,
  "value": [
    "Q135500748"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135500748"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135500748": 1
    },
    "new_unique": [
      "Q135500748"
    ],
    "new_values": [
      "Q135500748"
    ],
    "new_values_raw": [
      "Q135500748"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {},
    "old_unique": [],
    "old_values": [],
    "old_values_raw": [
      "MISSING"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [],
    "semantic_action": "CREATE_FROM_MISSING",
    "value_multiplicity_changes": {
      "Q135500748": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "12289"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-10T06:26:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2389998295,
  "report_revision_old": 2388961222,
  "report_violation_type": "Value type Q|49008",
  "report_violation_type_descriptions_en": [
    "positive integer with exactly two divisors, 1 and itself"
  ],
  "report_violation_type_labels_en": [
    "prime number"
  ],
  "report_violation_type_normalized": "Value type Q|49008",
  "report_violation_type_qids": [
    "Q49008"
  ],
  "report_violation_type_raw": "Value type Q|49008",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 15,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135500748"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q135500748"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135500748"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "one of the prime numbers that can be multiplied to give this number",
    "label": "prime factor"
  },
  "qid": {
    "description": "natural number",
    "label": "12289"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Q135500748"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135500748": 1
      },
      "new_unique": [
        "Q135500748"
      ],
      "new_values": [
        "Q135500748"
      ],
      "new_values_raw": [
        "Q135500748"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {},
      "old_unique": [],
      "old_values": [],
      "old_values_raw": [
        "MISSING"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [],
      "semantic_action": "CREATE_FROM_MISSING",
      "value_multiplicity_changes": {
        "Q135500748": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "CREATE_FROM_MISSING",
    "step": "value_delta"
  },
  {
    "detail": {
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 1,
      "independent_match_count": 1,
      "local_ids_count": 15,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135500748"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_QID"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "MISSING"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_focus_qid_domain_reasoning",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q135505807_2388473838`

| Field | Value |
|---|---|
| qid | Q135505807 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_FOCUS_QID_DOMAIN_REASONING / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_focus_qid_domain_reasoning |
| decision_constraint_type |   |
| group_key | ABOX::Q135505807::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135505807"] |
| classification_target_tokens | ["Q135505807"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | unknown_focus_qid_domain_reasoning |
| rationale | Focus QID is locally available, but local identity alone does not justify asserting this property value; correctness requires a domain/property-specific rule or external/domain reasoning. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135505807"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135505807"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "CREATE_FROM_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_focus_qid_domain_reasoning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q135505807"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "15149"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2388473838,
  "value": [
    "Q135505807"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135505807"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135505807": 1
    },
    "new_unique": [
      "Q135505807"
    ],
    "new_values": [
      "Q135505807"
    ],
    "new_values_raw": [
      "Q135505807"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {},
    "old_unique": [],
    "old_values": [],
    "old_values_raw": [
      "MISSING"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [],
    "semantic_action": "CREATE_FROM_MISSING",
    "value_multiplicity_changes": {
      "Q135505807": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "15149"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-10T06:26:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2389998295,
  "report_revision_old": 2388961222,
  "report_violation_type": "Value type Q|49008",
  "report_violation_type_descriptions_en": [
    "positive integer with exactly two divisors, 1 and itself"
  ],
  "report_violation_type_labels_en": [
    "prime number"
  ],
  "report_violation_type_normalized": "Value type Q|49008",
  "report_violation_type_qids": [
    "Q49008"
  ],
  "report_violation_type_raw": "Value type Q|49008",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 15,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135505807"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q135505807"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135505807"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "one of the prime numbers that can be multiplied to give this number",
    "label": "prime factor"
  },
  "qid": {
    "description": "natural number",
    "label": "15149"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Q135505807"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135505807": 1
      },
      "new_unique": [
        "Q135505807"
      ],
      "new_values": [
        "Q135505807"
      ],
      "new_values_raw": [
        "Q135505807"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {},
      "old_unique": [],
      "old_values": [],
      "old_values_raw": [
        "MISSING"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [],
      "semantic_action": "CREATE_FROM_MISSING",
      "value_multiplicity_changes": {
        "Q135505807": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "CREATE_FROM_MISSING",
    "step": "value_delta"
  },
  {
    "detail": {
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 1,
      "independent_match_count": 1,
      "local_ids_count": 15,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135505807"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_QID"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "MISSING"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_focus_qid_domain_reasoning",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q135505906_2388475459`

| Field | Value |
|---|---|
| qid | Q135505906 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_FOCUS_QID_DOMAIN_REASONING / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_focus_qid_domain_reasoning |
| decision_constraint_type |   |
| group_key | ABOX::Q135505906::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135505906"] |
| classification_target_tokens | ["Q135505906"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | unknown_focus_qid_domain_reasoning |
| rationale | Focus QID is locally available, but local identity alone does not justify asserting this property value; correctness requires a domain/property-specific rule or external/domain reasoning. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135505906"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135505906"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "CREATE_FROM_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_focus_qid_domain_reasoning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q135505906"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "15233"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2388475459,
  "value": [
    "Q135505906"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135505906"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135505906": 1
    },
    "new_unique": [
      "Q135505906"
    ],
    "new_values": [
      "Q135505906"
    ],
    "new_values_raw": [
      "Q135505906"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {},
    "old_unique": [],
    "old_values": [],
    "old_values_raw": [
      "MISSING"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [],
    "semantic_action": "CREATE_FROM_MISSING",
    "value_multiplicity_changes": {
      "Q135505906": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "15233"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-10T06:26:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2389998295,
  "report_revision_old": 2388961222,
  "report_violation_type": "Value type Q|49008",
  "report_violation_type_descriptions_en": [
    "positive integer with exactly two divisors, 1 and itself"
  ],
  "report_violation_type_labels_en": [
    "prime number"
  ],
  "report_violation_type_normalized": "Value type Q|49008",
  "report_violation_type_qids": [
    "Q49008"
  ],
  "report_violation_type_raw": "Value type Q|49008",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 15,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135505906"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q135505906"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135505906"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "one of the prime numbers that can be multiplied to give this number",
    "label": "prime factor"
  },
  "qid": {
    "description": "natural number",
    "label": "15233"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Q135505906"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135505906": 1
      },
      "new_unique": [
        "Q135505906"
      ],
      "new_values": [
        "Q135505906"
      ],
      "new_values_raw": [
        "Q135505906"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {},
      "old_unique": [],
      "old_values": [],
      "old_values_raw": [
        "MISSING"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [],
      "semantic_action": "CREATE_FROM_MISSING",
      "value_multiplicity_changes": {
        "Q135505906": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "CREATE_FROM_MISSING",
    "step": "value_delta"
  },
  {
    "detail": {
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 1,
      "independent_match_count": 1,
      "local_ids_count": 15,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135505906"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_QID"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "MISSING"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_focus_qid_domain_reasoning",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q135505978_2388475839`

| Field | Value |
|---|---|
| qid | Q135505978 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_FOCUS_QID_DOMAIN_REASONING / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_focus_qid_domain_reasoning |
| decision_constraint_type |   |
| group_key | ABOX::Q135505978::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135505978"] |
| classification_target_tokens | ["Q135505978"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | unknown_focus_qid_domain_reasoning |
| rationale | Focus QID is locally available, but local identity alone does not justify asserting this property value; correctness requires a domain/property-specific rule or external/domain reasoning. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135505978"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135505978"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "CREATE_FROM_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_focus_qid_domain_reasoning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q135505978"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "15263"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2388475839,
  "value": [
    "Q135505978"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135505978"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135505978": 1
    },
    "new_unique": [
      "Q135505978"
    ],
    "new_values": [
      "Q135505978"
    ],
    "new_values_raw": [
      "Q135505978"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {},
    "old_unique": [],
    "old_values": [],
    "old_values_raw": [
      "MISSING"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [],
    "semantic_action": "CREATE_FROM_MISSING",
    "value_multiplicity_changes": {
      "Q135505978": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "15263"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-10T06:26:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2389998295,
  "report_revision_old": 2388961222,
  "report_violation_type": "Value type Q|49008",
  "report_violation_type_descriptions_en": [
    "positive integer with exactly two divisors, 1 and itself"
  ],
  "report_violation_type_labels_en": [
    "prime number"
  ],
  "report_violation_type_normalized": "Value type Q|49008",
  "report_violation_type_qids": [
    "Q49008"
  ],
  "report_violation_type_raw": "Value type Q|49008",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 15,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135505978"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q135505978"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135505978"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "one of the prime numbers that can be multiplied to give this number",
    "label": "prime factor"
  },
  "qid": {
    "description": "natural number",
    "label": "15263"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Q135505978"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135505978": 1
      },
      "new_unique": [
        "Q135505978"
      ],
      "new_values": [
        "Q135505978"
      ],
      "new_values_raw": [
        "Q135505978"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {},
      "old_unique": [],
      "old_values": [],
      "old_values_raw": [
        "MISSING"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [],
      "semantic_action": "CREATE_FROM_MISSING",
      "value_multiplicity_changes": {
        "Q135505978": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "CREATE_FROM_MISSING",
    "step": "value_delta"
  },
  {
    "detail": {
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 1,
      "independent_match_count": 1,
      "local_ids_count": 15,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135505978"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_QID"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "MISSING"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_focus_qid_domain_reasoning",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q135506033_2388476144`

| Field | Value |
|---|---|
| qid | Q135506033 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_FOCUS_QID_DOMAIN_REASONING / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_focus_qid_domain_reasoning |
| decision_constraint_type |   |
| group_key | ABOX::Q135506033::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135506033"] |
| classification_target_tokens | ["Q135506033"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | unknown_focus_qid_domain_reasoning |
| rationale | Focus QID is locally available, but local identity alone does not justify asserting this property value; correctness requires a domain/property-specific rule or external/domain reasoning. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135506033"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135506033"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "CREATE_FROM_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_focus_qid_domain_reasoning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q135506033"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "15287"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2388476144,
  "value": [
    "Q135506033"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135506033"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135506033": 1
    },
    "new_unique": [
      "Q135506033"
    ],
    "new_values": [
      "Q135506033"
    ],
    "new_values_raw": [
      "Q135506033"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {},
    "old_unique": [],
    "old_values": [],
    "old_values_raw": [
      "MISSING"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [],
    "semantic_action": "CREATE_FROM_MISSING",
    "value_multiplicity_changes": {
      "Q135506033": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "15287"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-10T06:26:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2389998295,
  "report_revision_old": 2388961222,
  "report_violation_type": "Value type Q|49008",
  "report_violation_type_descriptions_en": [
    "positive integer with exactly two divisors, 1 and itself"
  ],
  "report_violation_type_labels_en": [
    "prime number"
  ],
  "report_violation_type_normalized": "Value type Q|49008",
  "report_violation_type_qids": [
    "Q49008"
  ],
  "report_violation_type_raw": "Value type Q|49008",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 15,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135506033"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q135506033"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135506033"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "one of the prime numbers that can be multiplied to give this number",
    "label": "prime factor"
  },
  "qid": {
    "description": "natural number",
    "label": "15287"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Q135506033"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135506033": 1
      },
      "new_unique": [
        "Q135506033"
      ],
      "new_values": [
        "Q135506033"
      ],
      "new_values_raw": [
        "Q135506033"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {},
      "old_unique": [],
      "old_values": [],
      "old_values_raw": [
        "MISSING"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [],
      "semantic_action": "CREATE_FROM_MISSING",
      "value_multiplicity_changes": {
        "Q135506033": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "CREATE_FROM_MISSING",
    "step": "value_delta"
  },
  {
    "detail": {
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 1,
      "independent_match_count": 1,
      "local_ids_count": 15,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135506033"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_QID"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "MISSING"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_focus_qid_domain_reasoning",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q135506119_2388477037`

| Field | Value |
|---|---|
| qid | Q135506119 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_FOCUS_QID_DOMAIN_REASONING / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_focus_qid_domain_reasoning |
| decision_constraint_type |   |
| group_key | ABOX::Q135506119::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135506119"] |
| classification_target_tokens | ["Q135506119"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | unknown_focus_qid_domain_reasoning |
| rationale | Focus QID is locally available, but local identity alone does not justify asserting this property value; correctness requires a domain/property-specific rule or external/domain reasoning. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135506119"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135506119"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "CREATE_FROM_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_focus_qid_domain_reasoning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q135506119"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "15331"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2388477037,
  "value": [
    "Q135506119"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135506119"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135506119": 1
    },
    "new_unique": [
      "Q135506119"
    ],
    "new_values": [
      "Q135506119"
    ],
    "new_values_raw": [
      "Q135506119"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {},
    "old_unique": [],
    "old_values": [],
    "old_values_raw": [
      "MISSING"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [],
    "semantic_action": "CREATE_FROM_MISSING",
    "value_multiplicity_changes": {
      "Q135506119": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "15331"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-10T06:26:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2389998295,
  "report_revision_old": 2388961222,
  "report_violation_type": "Value type Q|49008",
  "report_violation_type_descriptions_en": [
    "positive integer with exactly two divisors, 1 and itself"
  ],
  "report_violation_type_labels_en": [
    "prime number"
  ],
  "report_violation_type_normalized": "Value type Q|49008",
  "report_violation_type_qids": [
    "Q49008"
  ],
  "report_violation_type_raw": "Value type Q|49008",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 15,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135506119"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q135506119"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135506119"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "one of the prime numbers that can be multiplied to give this number",
    "label": "prime factor"
  },
  "qid": {
    "description": "natural number",
    "label": "15331"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Q135506119"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135506119": 1
      },
      "new_unique": [
        "Q135506119"
      ],
      "new_values": [
        "Q135506119"
      ],
      "new_values_raw": [
        "Q135506119"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {},
      "old_unique": [],
      "old_values": [],
      "old_values_raw": [
        "MISSING"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [],
      "semantic_action": "CREATE_FROM_MISSING",
      "value_multiplicity_changes": {
        "Q135506119": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "CREATE_FROM_MISSING",
    "step": "value_delta"
  },
  {
    "detail": {
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 1,
      "independent_match_count": 1,
      "local_ids_count": 15,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135506119"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_QID"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "MISSING"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_focus_qid_domain_reasoning",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q135506962_2388530023`

| Field | Value |
|---|---|
| qid | Q135506962 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_FOCUS_QID_DOMAIN_REASONING / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_focus_qid_domain_reasoning |
| decision_constraint_type |   |
| group_key | ABOX::Q135506962::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135506962"] |
| classification_target_tokens | ["Q135506962"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | unknown_focus_qid_domain_reasoning |
| rationale | Focus QID is locally available, but local identity alone does not justify asserting this property value; correctness requires a domain/property-specific rule or external/domain reasoning. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135506962"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135506962"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "CREATE_FROM_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_focus_qid_domain_reasoning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q135506962"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "15937"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2388530023,
  "value": [
    "Q135506962"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135506962"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135506962": 1
    },
    "new_unique": [
      "Q135506962"
    ],
    "new_values": [
      "Q135506962"
    ],
    "new_values_raw": [
      "Q135506962"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {},
    "old_unique": [],
    "old_values": [],
    "old_values_raw": [
      "MISSING"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [],
    "semantic_action": "CREATE_FROM_MISSING",
    "value_multiplicity_changes": {
      "Q135506962": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "15937"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-10T06:26:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2389998295,
  "report_revision_old": 2388961222,
  "report_violation_type": "Value type Q|49008",
  "report_violation_type_descriptions_en": [
    "positive integer with exactly two divisors, 1 and itself"
  ],
  "report_violation_type_labels_en": [
    "prime number"
  ],
  "report_violation_type_normalized": "Value type Q|49008",
  "report_violation_type_qids": [
    "Q49008"
  ],
  "report_violation_type_raw": "Value type Q|49008",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 15,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135506962"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q135506962"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135506962"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "one of the prime numbers that can be multiplied to give this number",
    "label": "prime factor"
  },
  "qid": {
    "description": "natural number",
    "label": "15937"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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

### Decision Trace

```json
[
  {
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Q135506962"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135506962": 1
      },
      "new_unique": [
        "Q135506962"
      ],
      "new_values": [
        "Q135506962"
      ],
      "new_values_raw": [
        "Q135506962"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {},
      "old_unique": [],
      "old_values": [],
      "old_values_raw": [
        "MISSING"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [],
      "semantic_action": "CREATE_FROM_MISSING",
      "value_multiplicity_changes": {
        "Q135506962": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "CREATE_FROM_MISSING",
    "step": "value_delta"
  },
  {
    "detail": {
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 1,
      "independent_match_count": 1,
      "local_ids_count": 15,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135506962"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_QID"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "MISSING"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_focus_qid_domain_reasoning",
    "step": "branch"
  }
]
```

---
