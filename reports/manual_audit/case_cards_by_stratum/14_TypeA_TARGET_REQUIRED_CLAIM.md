# TypeA_TARGET_REQUIRED_CLAIM

Cases: 10

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q135498394_2404098994`

| Field | Value |
|---|---|
| qid | Q135498394 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeA / TARGET_REQUIRED_CLAIM / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_target_required_claim |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | target_required_claim |
| classification_rule_subfamily | target_required_claim |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | ABOX::Q135498394::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm that the report is a target-required-claim violation and that the required target is the focus QID.
- This should be rule-deterministic, not local-evidence TypeB.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135498394"] |
| classification_target_tokens | ["Q135498394"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | target_required_claim |
| rationale | Target-required-claim violation deterministically requires the focus entity as the target value. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135498394"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135498394"
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
  "classification_rule_family": "target_required_claim",
  "classification_rule_subfamily": "target_required_claim",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135498394"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "10501"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404098994,
  "value": [
    "Q135498394"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135498394"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135498394": 1
    },
    "new_unique": [
      "Q135498394"
    ],
    "new_values": [
      "Q135498394"
    ],
    "new_values_raw": [
      "Q135498394"
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
      "Q135498394": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "10501"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-09-14T06:35:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2404606348,
  "report_revision_old": 2403072583,
  "report_violation_type": "Target required claim P|5236",
  "report_violation_type_normalized": "Target required claim P|5236",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|5236",
  "value": [
    "MISSING"
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
    "Q135498394"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "label": "10501"
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
        "Q135498394"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135498394": 1
      },
      "new_unique": [
        "Q135498394"
      ],
      "new_values": [
        "Q135498394"
      ],
      "new_values_raw": [
        "Q135498394"
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
        "Q135498394": {
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
      "target_required_property": "P5236"
    },
    "kind": "TARGET_REQUIRED_CLAIM",
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
    "result": "target_required_claim",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q135502768_2388853034`

| Field | Value |
|---|---|
| qid | Q135502768 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeA / TARGET_REQUIRED_CLAIM / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_target_required_claim |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | target_required_claim |
| classification_rule_subfamily | target_required_claim |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | ABOX::Q135502768::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm that the report is a target-required-claim violation and that the required target is the focus QID.
- This should be rule-deterministic, not local-evidence TypeB.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135502768"] |
| classification_target_tokens | ["Q135502768"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | target_required_claim |
| rationale | Target-required-claim violation deterministically requires the focus entity as the target value. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135502768"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135502768"
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
  "classification_rule_family": "target_required_claim",
  "classification_rule_subfamily": "target_required_claim",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q135502768"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "13217"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2388853034,
  "value": [
    "Q135502768"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135502768"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135502768": 1
    },
    "new_unique": [
      "Q135502768"
    ],
    "new_values": [
      "Q135502768"
    ],
    "new_values_raw": [
      "Q135502768"
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
      "Q135502768": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "13217"
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
  "report_violation_type": "Target required claim P|5236",
  "report_violation_type_normalized": "Target required claim P|5236",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|5236",
  "value": [
    "MISSING"
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
    "Q135502768"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "label": "13217"
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
        "Q135502768"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135502768": 1
      },
      "new_unique": [
        "Q135502768"
      ],
      "new_values": [
        "Q135502768"
      ],
      "new_values_raw": [
        "Q135502768"
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
        "Q135502768": {
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
      "target_required_property": "P5236"
    },
    "kind": "TARGET_REQUIRED_CLAIM",
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
    "result": "target_required_claim",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q135503571_2388896127`

| Field | Value |
|---|---|
| qid | Q135503571 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeA / TARGET_REQUIRED_CLAIM / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_target_required_claim |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | target_required_claim |
| classification_rule_subfamily | target_required_claim |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | ABOX::Q135503571::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm that the report is a target-required-claim violation and that the required target is the focus QID.
- This should be rule-deterministic, not local-evidence TypeB.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135503571"] |
| classification_target_tokens | ["Q135503571"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | target_required_claim |
| rationale | Target-required-claim violation deterministically requires the focus entity as the target value. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135503571"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135503571"
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
  "classification_rule_family": "target_required_claim",
  "classification_rule_subfamily": "target_required_claim",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q135503571"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "13691"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2388896127,
  "value": [
    "Q135503571"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135503571"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135503571": 1
    },
    "new_unique": [
      "Q135503571"
    ],
    "new_values": [
      "Q135503571"
    ],
    "new_values_raw": [
      "Q135503571"
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
      "Q135503571": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "13691"
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
  "report_violation_type": "Target required claim P|5236",
  "report_violation_type_normalized": "Target required claim P|5236",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|5236",
  "value": [
    "MISSING"
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
    "Q135503571"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "label": "13691"
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
        "Q135503571"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135503571": 1
      },
      "new_unique": [
        "Q135503571"
      ],
      "new_values": [
        "Q135503571"
      ],
      "new_values_raw": [
        "Q135503571"
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
        "Q135503571": {
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
      "target_required_property": "P5236"
    },
    "kind": "TARGET_REQUIRED_CLAIM",
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
    "result": "target_required_claim",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q135514464_2404101043`

| Field | Value |
|---|---|
| qid | Q135514464 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeA / TARGET_REQUIRED_CLAIM / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_target_required_claim |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | target_required_claim |
| classification_rule_subfamily | target_required_claim |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | ABOX::Q135514464::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm that the report is a target-required-claim violation and that the required target is the focus QID.
- This should be rule-deterministic, not local-evidence TypeB.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135514464"] |
| classification_target_tokens | ["Q135514464"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | target_required_claim |
| rationale | Target-required-claim violation deterministically requires the focus entity as the target value. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135514464"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135514464"
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
  "classification_rule_family": "target_required_claim",
  "classification_rule_subfamily": "target_required_claim",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135514464"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "21673"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404101043,
  "value": [
    "Q135514464"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135514464"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135514464": 1
    },
    "new_unique": [
      "Q135514464"
    ],
    "new_values": [
      "Q135514464"
    ],
    "new_values_raw": [
      "Q135514464"
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
      "Q135514464": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "21673"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-09-14T06:35:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2404606348,
  "report_revision_old": 2403072583,
  "report_violation_type": "Target required claim P|5236",
  "report_violation_type_normalized": "Target required claim P|5236",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|5236",
  "value": [
    "MISSING"
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
    "Q135514464"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "label": "21673"
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
        "Q135514464"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135514464": 1
      },
      "new_unique": [
        "Q135514464"
      ],
      "new_values": [
        "Q135514464"
      ],
      "new_values_raw": [
        "Q135514464"
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
        "Q135514464": {
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
      "target_required_property": "P5236"
    },
    "kind": "TARGET_REQUIRED_CLAIM",
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
    "result": "target_required_claim",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q135515516_2404100420`

| Field | Value |
|---|---|
| qid | Q135515516 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeA / TARGET_REQUIRED_CLAIM / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_target_required_claim |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | target_required_claim |
| classification_rule_subfamily | target_required_claim |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | ABOX::Q135515516::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm that the report is a target-required-claim violation and that the required target is the focus QID.
- This should be rule-deterministic, not local-evidence TypeB.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135515516"] |
| classification_target_tokens | ["Q135515516"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | target_required_claim |
| rationale | Target-required-claim violation deterministically requires the focus entity as the target value. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135515516"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135515516"
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
  "classification_rule_family": "target_required_claim",
  "classification_rule_subfamily": "target_required_claim",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135515516"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "22343"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404100420,
  "value": [
    "Q135515516"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135515516"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135515516": 1
    },
    "new_unique": [
      "Q135515516"
    ],
    "new_values": [
      "Q135515516"
    ],
    "new_values_raw": [
      "Q135515516"
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
      "Q135515516": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "22343"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-09-14T06:35:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2404606348,
  "report_revision_old": 2403072583,
  "report_violation_type": "Target required claim P|5236",
  "report_violation_type_normalized": "Target required claim P|5236",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|5236",
  "value": [
    "MISSING"
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
    "Q135515516"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "label": "22343"
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
        "Q135515516"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135515516": 1
      },
      "new_unique": [
        "Q135515516"
      ],
      "new_values": [
        "Q135515516"
      ],
      "new_values_raw": [
        "Q135515516"
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
        "Q135515516": {
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
      "target_required_property": "P5236"
    },
    "kind": "TARGET_REQUIRED_CLAIM",
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
    "result": "target_required_claim",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q135515596_2404099131`

| Field | Value |
|---|---|
| qid | Q135515596 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeA / TARGET_REQUIRED_CLAIM / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_target_required_claim |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | target_required_claim |
| classification_rule_subfamily | target_required_claim |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | ABOX::Q135515596::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm that the report is a target-required-claim violation and that the required target is the focus QID.
- This should be rule-deterministic, not local-evidence TypeB.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135515596"] |
| classification_target_tokens | ["Q135515596"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | target_required_claim |
| rationale | Target-required-claim violation deterministically requires the focus entity as the target value. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135515596"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135515596"
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
  "classification_rule_family": "target_required_claim",
  "classification_rule_subfamily": "target_required_claim",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135515596"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "22391"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404099131,
  "value": [
    "Q135515596"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135515596"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135515596": 1
    },
    "new_unique": [
      "Q135515596"
    ],
    "new_values": [
      "Q135515596"
    ],
    "new_values_raw": [
      "Q135515596"
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
      "Q135515596": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "22391"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-09-14T06:35:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2404606348,
  "report_revision_old": 2403072583,
  "report_violation_type": "Target required claim P|5236",
  "report_violation_type_normalized": "Target required claim P|5236",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|5236",
  "value": [
    "MISSING"
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
    "Q135515596"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "label": "22391"
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
        "Q135515596"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135515596": 1
      },
      "new_unique": [
        "Q135515596"
      ],
      "new_values": [
        "Q135515596"
      ],
      "new_values_raw": [
        "Q135515596"
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
        "Q135515596": {
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
      "target_required_property": "P5236"
    },
    "kind": "TARGET_REQUIRED_CLAIM",
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
    "result": "target_required_claim",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q135516207_2404100217`

| Field | Value |
|---|---|
| qid | Q135516207 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeA / TARGET_REQUIRED_CLAIM / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_target_required_claim |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | target_required_claim |
| classification_rule_subfamily | target_required_claim |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | ABOX::Q135516207::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm that the report is a target-required-claim violation and that the required target is the focus QID.
- This should be rule-deterministic, not local-evidence TypeB.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135516207"] |
| classification_target_tokens | ["Q135516207"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | target_required_claim |
| rationale | Target-required-claim violation deterministically requires the focus entity as the target value. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135516207"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135516207"
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
  "classification_rule_family": "target_required_claim",
  "classification_rule_subfamily": "target_required_claim",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135516207"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "22787"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404100217,
  "value": [
    "Q135516207"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135516207"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135516207": 1
    },
    "new_unique": [
      "Q135516207"
    ],
    "new_values": [
      "Q135516207"
    ],
    "new_values_raw": [
      "Q135516207"
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
      "Q135516207": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "22787"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-09-14T06:35:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2404606348,
  "report_revision_old": 2403072583,
  "report_violation_type": "Target required claim P|5236",
  "report_violation_type_normalized": "Target required claim P|5236",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|5236",
  "value": [
    "MISSING"
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
    "Q135516207"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "label": "22787"
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
        "Q135516207"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135516207": 1
      },
      "new_unique": [
        "Q135516207"
      ],
      "new_values": [
        "Q135516207"
      ],
      "new_values_raw": [
        "Q135516207"
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
        "Q135516207": {
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
      "target_required_property": "P5236"
    },
    "kind": "TARGET_REQUIRED_CLAIM",
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
    "result": "target_required_claim",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q135516988_2404098460`

| Field | Value |
|---|---|
| qid | Q135516988 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeA / TARGET_REQUIRED_CLAIM / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_target_required_claim |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | target_required_claim |
| classification_rule_subfamily | target_required_claim |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | ABOX::Q135516988::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm that the report is a target-required-claim violation and that the required target is the focus QID.
- This should be rule-deterministic, not local-evidence TypeB.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135516988"] |
| classification_target_tokens | ["Q135516988"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | target_required_claim |
| rationale | Target-required-claim violation deterministically requires the focus entity as the target value. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135516988"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135516988"
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
  "classification_rule_family": "target_required_claim",
  "classification_rule_subfamily": "target_required_claim",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135516988"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "17881"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404098460,
  "value": [
    "Q135516988"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135516988"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135516988": 1
    },
    "new_unique": [
      "Q135516988"
    ],
    "new_values": [
      "Q135516988"
    ],
    "new_values_raw": [
      "Q135516988"
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
      "Q135516988": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "17881"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-09-14T06:35:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2404606348,
  "report_revision_old": 2403072583,
  "report_violation_type": "Target required claim P|5236",
  "report_violation_type_normalized": "Target required claim P|5236",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|5236",
  "value": [
    "MISSING"
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
    "Q135516988"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "label": "17881"
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
        "Q135516988"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135516988": 1
      },
      "new_unique": [
        "Q135516988"
      ],
      "new_values": [
        "Q135516988"
      ],
      "new_values_raw": [
        "Q135516988"
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
        "Q135516988": {
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
      "target_required_property": "P5236"
    },
    "kind": "TARGET_REQUIRED_CLAIM",
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
    "result": "target_required_claim",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q135517157_2404099555`

| Field | Value |
|---|---|
| qid | Q135517157 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeA / TARGET_REQUIRED_CLAIM / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_target_required_claim |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | target_required_claim |
| classification_rule_subfamily | target_required_claim |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | ABOX::Q135517157::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm that the report is a target-required-claim violation and that the required target is the focus QID.
- This should be rule-deterministic, not local-evidence TypeB.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135517157"] |
| classification_target_tokens | ["Q135517157"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | target_required_claim |
| rationale | Target-required-claim violation deterministically requires the focus entity as the target value. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135517157"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135517157"
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
  "classification_rule_family": "target_required_claim",
  "classification_rule_subfamily": "target_required_claim",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135517157"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "23297"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404099555,
  "value": [
    "Q135517157"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135517157"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135517157": 1
    },
    "new_unique": [
      "Q135517157"
    ],
    "new_values": [
      "Q135517157"
    ],
    "new_values_raw": [
      "Q135517157"
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
      "Q135517157": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "23297"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-09-14T06:35:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2404606348,
  "report_revision_old": 2403072583,
  "report_violation_type": "Target required claim P|5236",
  "report_violation_type_normalized": "Target required claim P|5236",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|5236",
  "value": [
    "MISSING"
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
    "Q135517157"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "label": "23297"
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
        "Q135517157"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135517157": 1
      },
      "new_unique": [
        "Q135517157"
      ],
      "new_values": [
        "Q135517157"
      ],
      "new_values_raw": [
        "Q135517157"
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
        "Q135517157": {
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
      "target_required_property": "P5236"
    },
    "kind": "TARGET_REQUIRED_CLAIM",
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
    "result": "target_required_claim",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q135633158_2404099838`

| Field | Value |
|---|---|
| qid | Q135633158 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeA / TARGET_REQUIRED_CLAIM / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_target_required_claim |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | target_required_claim |
| classification_rule_subfamily | target_required_claim |
| decision_constraint_type | Q21510864 value-requires-statement constraint |
| group_key | ABOX::Q135633158::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm that the report is a target-required-claim violation and that the required target is the focus QID.
- This should be rule-deterministic, not local-evidence TypeB.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135633158"] |
| classification_target_tokens | ["Q135633158"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | target_required_claim |
| rationale | Target-required-claim violation deterministically requires the focus entity as the target value. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q135633158"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q135633158"
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
  "classification_rule_family": "target_required_claim",
  "classification_rule_subfamily": "target_required_claim",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "value-requires-statement constraint",
  "decision_constraint_type_qid": "Q21510864"
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135633158"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "21101"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404099838,
  "value": [
    "Q135633158"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q135633158"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q135633158": 1
    },
    "new_unique": [
      "Q135633158"
    ],
    "new_values": [
      "Q135633158"
    ],
    "new_values_raw": [
      "Q135633158"
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
      "Q135633158": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "21101"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-09-14T06:35:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2404606348,
  "report_revision_old": 2403072583,
  "report_violation_type": "Target required claim P|5236",
  "report_violation_type_normalized": "Target required claim P|5236",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|5236",
  "value": [
    "MISSING"
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
    "Q135633158"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "label": "21101"
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
        "Q135633158"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q135633158": 1
      },
      "new_unique": [
        "Q135633158"
      ],
      "new_values": [
        "Q135633158"
      ],
      "new_values_raw": [
        "Q135633158"
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
        "Q135633158": {
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
      "target_required_property": "P5236"
    },
    "kind": "TARGET_REQUIRED_CLAIM",
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
    "result": "target_required_claim",
    "step": "branch"
  }
]
```

---
