# TypeB_LOCAL_TEXT_CONFIRMED

Cases: 25

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q117320046_2446824475`

| Field | Value |
|---|---|
| qid | Q117320046 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q117320046::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Gavansky Residential Complex for Workers"] |
| classification_target_tokens | ["Gavansky Residential Complex for Workers"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | NEIGHBOR_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Gavansky Residential Complex for Workers"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Gavansky Residential Complex for Workers"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Gavansky Residental Complex for Workers"
  ],
  "removed_unique_values": [
    "Gavansky Residental Complex for Workers"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Екатерина Борисова",
  "kind": "A_BOX",
  "new_value": [
    "Gavansky Residential Complex for Workers"
  ],
  "old_value": [
    "Gavansky Residental Complex for Workers"
  ],
  "revision_id": 2446824475,
  "value": [
    "Gavansky Residential Complex for Workers"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Gavansky Residential Complex for Workers"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Gavansky Residential Complex for Workers": 1
    },
    "new_unique": [
      "Gavansky Residential Complex for Workers"
    ],
    "new_values": [
      "Gavansky Residential Complex for Workers"
    ],
    "new_values_raw": [
      "Gavansky Residential Complex for Workers"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Gavansky Residental Complex for Workers": 1
    },
    "old_unique": [
      "Gavansky Residental Complex for Workers"
    ],
    "old_values": [
      "Gavansky Residental Complex for Workers"
    ],
    "old_values_raw": [
      "Gavansky Residental Complex for Workers"
    ],
    "removed_unique_values": [
      "Gavansky Residental Complex for Workers"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Gavansky Residental Complex for Workers": {
        "new": 0,
        "old": 1
      },
      "Gavansky Residential Complex for Workers": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-26T13:06:48",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447382517,
  "report_revision_old": 2447067079,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Gavansky Residental Complex for Workers"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 13,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "gavansky residential complex for workers",
      "raw_match_text": "Gavansky Residential Complex for Workers",
      "source": "NEIGHBOR_LABEL",
      "token": "Gavansky Residential Complex for Workers"
    }
  ],
  "needed": 1,
  "sources_used": [
    "NEIGHBOR_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Gavansky Residental Complex for Workers"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Gavansky Residential Complex for Workers"
  ],
  "truth_tokens_in_recorded_matches": [
    "Gavansky Residential Complex for Workers"
  ],
  "used_literal_substring": false
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
    "description": null,
    "label": "Жилой корпус"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Gavansky Residential Complex for Workers"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Gavansky Residential Complex for Workers": 1
      },
      "new_unique": [
        "Gavansky Residential Complex for Workers"
      ],
      "new_values": [
        "Gavansky Residential Complex for Workers"
      ],
      "new_values_raw": [
        "Gavansky Residential Complex for Workers"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Gavansky Residental Complex for Workers": 1
      },
      "old_unique": [
        "Gavansky Residental Complex for Workers"
      ],
      "old_values": [
        "Gavansky Residental Complex for Workers"
      ],
      "old_values_raw": [
        "Gavansky Residental Complex for Workers"
      ],
      "removed_unique_values": [
        "Gavansky Residental Complex for Workers"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Gavansky Residental Complex for Workers": {
          "new": 0,
          "old": 1
        },
        "Gavansky Residential Complex for Workers": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 13,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "gavansky residential complex for workers",
          "raw_match_text": "Gavansky Residential Complex for Workers",
          "source": "NEIGHBOR_LABEL",
          "token": "Gavansky Residential Complex for Workers"
        }
      ],
      "needed": 1,
      "sources_used": [
        "NEIGHBOR_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Gavansky Residental Complex for Workers"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q117320048_2446824658`

| Field | Value |
|---|---|
| qid | Q117320048 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q117320048::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Gavansky Residential Complex for Workers"] |
| classification_target_tokens | ["Gavansky Residential Complex for Workers"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | NEIGHBOR_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Gavansky Residential Complex for Workers"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Gavansky Residential Complex for Workers"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Gavansky Residental Complex for Workers"
  ],
  "removed_unique_values": [
    "Gavansky Residental Complex for Workers"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Екатерина Борисова",
  "kind": "A_BOX",
  "new_value": [
    "Gavansky Residential Complex for Workers"
  ],
  "old_value": [
    "Gavansky Residental Complex for Workers"
  ],
  "revision_id": 2446824658,
  "value": [
    "Gavansky Residential Complex for Workers"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Gavansky Residential Complex for Workers"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Gavansky Residential Complex for Workers": 1
    },
    "new_unique": [
      "Gavansky Residential Complex for Workers"
    ],
    "new_values": [
      "Gavansky Residential Complex for Workers"
    ],
    "new_values_raw": [
      "Gavansky Residential Complex for Workers"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Gavansky Residental Complex for Workers": 1
    },
    "old_unique": [
      "Gavansky Residental Complex for Workers"
    ],
    "old_values": [
      "Gavansky Residental Complex for Workers"
    ],
    "old_values_raw": [
      "Gavansky Residental Complex for Workers"
    ],
    "removed_unique_values": [
      "Gavansky Residental Complex for Workers"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Gavansky Residental Complex for Workers": {
        "new": 0,
        "old": 1
      },
      "Gavansky Residential Complex for Workers": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-26T13:06:48",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447382517,
  "report_revision_old": 2447067079,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Gavansky Residental Complex for Workers"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 13,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "gavansky residential complex for workers",
      "raw_match_text": "Gavansky Residential Complex for Workers",
      "source": "NEIGHBOR_LABEL",
      "token": "Gavansky Residential Complex for Workers"
    }
  ],
  "needed": 1,
  "sources_used": [
    "NEIGHBOR_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Gavansky Residental Complex for Workers"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Gavansky Residential Complex for Workers"
  ],
  "truth_tokens_in_recorded_matches": [
    "Gavansky Residential Complex for Workers"
  ],
  "used_literal_substring": false
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
    "description": null,
    "label": "Жилой корпус"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Gavansky Residential Complex for Workers"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Gavansky Residential Complex for Workers": 1
      },
      "new_unique": [
        "Gavansky Residential Complex for Workers"
      ],
      "new_values": [
        "Gavansky Residential Complex for Workers"
      ],
      "new_values_raw": [
        "Gavansky Residential Complex for Workers"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Gavansky Residental Complex for Workers": 1
      },
      "old_unique": [
        "Gavansky Residental Complex for Workers"
      ],
      "old_values": [
        "Gavansky Residental Complex for Workers"
      ],
      "old_values_raw": [
        "Gavansky Residental Complex for Workers"
      ],
      "removed_unique_values": [
        "Gavansky Residental Complex for Workers"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Gavansky Residental Complex for Workers": {
          "new": 0,
          "old": 1
        },
        "Gavansky Residential Complex for Workers": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 13,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "gavansky residential complex for workers",
          "raw_match_text": "Gavansky Residential Complex for Workers",
          "source": "NEIGHBOR_LABEL",
          "token": "Gavansky Residential Complex for Workers"
        }
      ],
      "needed": 1,
      "sources_used": [
        "NEIGHBOR_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Gavansky Residental Complex for Workers"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q123734475_2447259961`

| Field | Value |
|---|---|
| qid | Q123734475 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q123734475::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Frosinone Calcio v US Città di Palermo, 16 June 2018"] |
| classification_target_tokens | ["Frosinone Calcio v US Città di Palermo, 16 June 2018"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Frosinone Calcio v US Città di Palermo, 16 June 2018"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Frosinone Calcio v US Città di Palermo, 16 June 2018"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
  ],
  "removed_unique_values": [
    "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Pi bot",
  "kind": "A_BOX",
  "new_value": [
    "Frosinone Calcio v US Città di Palermo, 16 June 2018"
  ],
  "old_value": [
    "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
  ],
  "revision_id": 2447259961,
  "value": [
    "Frosinone Calcio v US Città di Palermo, 16 June 2018"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Frosinone Calcio v US Città di Palermo, 16 June 2018"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Frosinone Calcio v US Città di Palermo, 16 June 2018": 1
    },
    "new_unique": [
      "Frosinone Calcio v US Città di Palermo, 16 June 2018"
    ],
    "new_values": [
      "Frosinone Calcio v US Città di Palermo, 16 June 2018"
    ],
    "new_values_raw": [
      "Frosinone Calcio v US Città di Palermo, 16 June 2018"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)": 1
    },
    "old_unique": [
      "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
    ],
    "old_values": [
      "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
    ],
    "old_values_raw": [
      "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
    ],
    "removed_unique_values": [
      "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)": {
        "new": 0,
        "old": 1
      },
      "Frosinone Calcio v US Città di Palermo, 16 June 2018": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T12:35:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447772714,
  "report_revision_old": 2447382517,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 5,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "frosinone calcio v us città di palermo 16 june 2018",
      "raw_match_text": "Frosinone Calcio v US Città di Palermo, 16 June 2018",
      "source": "FOCUS_LABEL",
      "token": "Frosinone Calcio v US Città di Palermo, 16 June 2018"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Frosinone Calcio v US Città di Palermo, 16 June 2018"
  ],
  "truth_tokens_in_recorded_matches": [
    "Frosinone Calcio v US Città di Palermo, 16 June 2018"
  ],
  "used_literal_substring": false
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
    "description": null,
    "label": "Frosinone Calcio v US Città di Palermo, 16 June 2018"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Frosinone Calcio v US Città di Palermo, 16 June 2018"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Frosinone Calcio v US Città di Palermo, 16 June 2018": 1
      },
      "new_unique": [
        "Frosinone Calcio v US Città di Palermo, 16 June 2018"
      ],
      "new_values": [
        "Frosinone Calcio v US Città di Palermo, 16 June 2018"
      ],
      "new_values_raw": [
        "Frosinone Calcio v US Città di Palermo, 16 June 2018"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)": 1
      },
      "old_unique": [
        "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
      ],
      "old_values": [
        "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
      ],
      "old_values_raw": [
        "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
      ],
      "removed_unique_values": [
        "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)": {
          "new": 0,
          "old": 1
        },
        "Frosinone Calcio v US Città di Palermo, 16 June 2018": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 5,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "frosinone calcio v us città di palermo 16 june 2018",
          "raw_match_text": "Frosinone Calcio v US Città di Palermo, 16 June 2018",
          "source": "FOCUS_LABEL",
          "token": "Frosinone Calcio v US Città di Palermo, 16 June 2018"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q137217940_2439242692`

| Field | Value |
|---|---|
| qid | Q137217940 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q137217940::P225 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Magnadigita"] |
| classification_target_tokens | ["Magnadigita"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Magnadigita"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Magnadigita"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Bolitoglossa"
  ],
  "removed_unique_values": [
    "Bolitoglossa"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "PieterJanR",
  "kind": "A_BOX",
  "new_value": [
    "Magnadigita"
  ],
  "old_value": [
    "Bolitoglossa"
  ],
  "revision_id": 2439242692,
  "value": [
    "Magnadigita"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Magnadigita"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Magnadigita": 1
    },
    "new_unique": [
      "Magnadigita"
    ],
    "new_values": [
      "Magnadigita"
    ],
    "new_values_raw": [
      "Magnadigita"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Bolitoglossa": 1
    },
    "old_unique": [
      "Bolitoglossa"
    ],
    "old_values": [
      "Bolitoglossa"
    ],
    "old_values_raw": [
      "Bolitoglossa"
    ],
    "removed_unique_values": [
      "Bolitoglossa"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Bolitoglossa": {
        "new": 0,
        "old": 1
      },
      "Magnadigita": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-09T12:33:44",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2440014373,
  "report_revision_old": 2439564746,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "report_violation_types": [
    "Unique value",
    "Item P|105"
  ],
  "value": [
    "Bolitoglossa"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "magnadigita",
      "raw_match_text": "Magnadigita",
      "source": "FOCUS_LABEL",
      "token": "Magnadigita"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Bolitoglossa"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Magnadigita"
  ],
  "truth_tokens_in_recorded_matches": [
    "Magnadigita"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "correct scientific name of a taxon (according to the reference given)",
    "label": "taxon name"
  },
  "qid": {
    "description": "subgenus of Bolitoglossa",
    "label": "Magnadigita"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
        "Magnadigita"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Magnadigita": 1
      },
      "new_unique": [
        "Magnadigita"
      ],
      "new_values": [
        "Magnadigita"
      ],
      "new_values_raw": [
        "Magnadigita"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Bolitoglossa": 1
      },
      "old_unique": [
        "Bolitoglossa"
      ],
      "old_values": [
        "Bolitoglossa"
      ],
      "old_values_raw": [
        "Bolitoglossa"
      ],
      "removed_unique_values": [
        "Bolitoglossa"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Bolitoglossa": {
          "new": 0,
          "old": 1
        },
        "Magnadigita": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "magnadigita",
          "raw_match_text": "Magnadigita",
          "source": "FOCUS_LABEL",
          "token": "Magnadigita"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Bolitoglossa"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q137288654_2440360795`

| Field | Value |
|---|---|
| qid | Q137288654 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q137288654::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Ruinas de la presa de la Estanca"] |
| classification_target_tokens | ["Ruinas de la presa de la Estanca"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Ruinas de la presa de la Estanca"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Ruinas de la presa de la Estanca"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Ruinas de la presa de la Estanca (Cascante)"
  ],
  "removed_unique_values": [
    "Ruinas de la presa de la Estanca (Cascante)"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Pi bot",
  "kind": "A_BOX",
  "new_value": [
    "Ruinas de la presa de la Estanca"
  ],
  "old_value": [
    "Ruinas de la presa de la Estanca (Cascante)"
  ],
  "revision_id": 2440360795,
  "value": [
    "Ruinas de la presa de la Estanca"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Ruinas de la presa de la Estanca"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Ruinas de la presa de la Estanca": 1
    },
    "new_unique": [
      "Ruinas de la presa de la Estanca"
    ],
    "new_values": [
      "Ruinas de la presa de la Estanca"
    ],
    "new_values_raw": [
      "Ruinas de la presa de la Estanca"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Ruinas de la presa de la Estanca (Cascante)": 1
    },
    "old_unique": [
      "Ruinas de la presa de la Estanca (Cascante)"
    ],
    "old_values": [
      "Ruinas de la presa de la Estanca (Cascante)"
    ],
    "old_values_raw": [
      "Ruinas de la presa de la Estanca (Cascante)"
    ],
    "removed_unique_values": [
      "Ruinas de la presa de la Estanca (Cascante)"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Ruinas de la presa de la Estanca": {
        "new": 1,
        "old": 0
      },
      "Ruinas de la presa de la Estanca (Cascante)": {
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
  "report_fix_date": "2025-12-11T12:56:58",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2440854653,
  "report_revision_old": 2440415829,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Ruinas de la presa de la Estanca (Cascante)"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "ruinas de la presa de la estanca",
      "raw_match_text": "Ruinas de la presa de la Estanca",
      "source": "FOCUS_LABEL",
      "token": "Ruinas de la presa de la Estanca"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Ruinas de la presa de la Estanca (Cascante)"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Ruinas de la presa de la Estanca"
  ],
  "truth_tokens_in_recorded_matches": [
    "Ruinas de la presa de la Estanca"
  ],
  "used_literal_substring": false
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
    "description": "Restos de ingeniería hidráulica",
    "label": "Ruinas de la presa de la Estanca"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Ruinas de la presa de la Estanca"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Ruinas de la presa de la Estanca": 1
      },
      "new_unique": [
        "Ruinas de la presa de la Estanca"
      ],
      "new_values": [
        "Ruinas de la presa de la Estanca"
      ],
      "new_values_raw": [
        "Ruinas de la presa de la Estanca"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Ruinas de la presa de la Estanca (Cascante)": 1
      },
      "old_unique": [
        "Ruinas de la presa de la Estanca (Cascante)"
      ],
      "old_values": [
        "Ruinas de la presa de la Estanca (Cascante)"
      ],
      "old_values_raw": [
        "Ruinas de la presa de la Estanca (Cascante)"
      ],
      "removed_unique_values": [
        "Ruinas de la presa de la Estanca (Cascante)"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Ruinas de la presa de la Estanca": {
          "new": 1,
          "old": 0
        },
        "Ruinas de la presa de la Estanca (Cascante)": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "ruinas de la presa de la estanca",
          "raw_match_text": "Ruinas de la presa de la Estanca",
          "source": "FOCUS_LABEL",
          "token": "Ruinas de la presa de la Estanca"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Ruinas de la presa de la Estanca (Cascante)"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q137379963_2443215624`

| Field | Value |
|---|---|
| qid | Q137379963 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q137379963::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["The Rewynd"] |
| classification_target_tokens | ["The Rewynd"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "The Rewynd"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "The Rewynd"
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
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Armat272",
  "kind": "A_BOX",
  "new_value": [
    "The Rewynd"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2443215624,
  "value": [
    "The Rewynd"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "The Rewynd"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "The Rewynd": 1
    },
    "new_unique": [
      "The Rewynd"
    ],
    "new_values": [
      "The Rewynd"
    ],
    "new_values_raw": [
      "The Rewynd"
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
      "The Rewynd": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-17T12:39:08",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2443399922,
  "report_revision_old": 2442981743,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
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
  "local_ids_count": 31,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "the rewynd",
      "raw_match_text": "The Rewynd",
      "source": "FOCUS_LABEL",
      "token": "The Rewynd"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "The Rewynd"
  ],
  "truth_tokens_in_recorded_matches": [
    "The Rewynd"
  ],
  "used_literal_substring": false
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
    "description": "Indian singer-songwriter and record producer",
    "label": "The Rewynd"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "The Rewynd"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "The Rewynd": 1
      },
      "new_unique": [
        "The Rewynd"
      ],
      "new_values": [
        "The Rewynd"
      ],
      "new_values_raw": [
        "The Rewynd"
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
        "The Rewynd": {
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
      "local_ids_count": 31,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "the rewynd",
          "raw_match_text": "The Rewynd",
          "source": "FOCUS_LABEL",
          "token": "The Rewynd"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q137385306_2442996855`

| Field | Value |
|---|---|
| qid | Q137385306 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q137385306::P225 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Pachyspathe"] |
| classification_target_tokens | ["Pachyspathe"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Pachyspathe"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Pachyspathe"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Nephus"
  ],
  "removed_unique_values": [
    "Nephus"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "PieterJanR",
  "kind": "A_BOX",
  "new_value": [
    "Pachyspathe"
  ],
  "old_value": [
    "Nephus"
  ],
  "revision_id": 2442996855,
  "value": [
    "Pachyspathe"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Pachyspathe"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Pachyspathe": 1
    },
    "new_unique": [
      "Pachyspathe"
    ],
    "new_values": [
      "Pachyspathe"
    ],
    "new_values_raw": [
      "Pachyspathe"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Nephus": 1
    },
    "old_unique": [
      "Nephus"
    ],
    "old_values": [
      "Nephus"
    ],
    "old_values_raw": [
      "Nephus"
    ],
    "removed_unique_values": [
      "Nephus"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Nephus": {
        "new": 0,
        "old": 1
      },
      "Pachyspathe": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-18T19:37:14",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2443848675,
  "report_revision_old": 2443413974,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "report_violation_types": [
    "Unique value",
    "Item P|105"
  ],
  "value": [
    "Nephus"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "pachyspathe",
      "raw_match_text": "Pachyspathe",
      "source": "FOCUS_LABEL",
      "token": "Pachyspathe"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Nephus"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Pachyspathe"
  ],
  "truth_tokens_in_recorded_matches": [
    "Pachyspathe"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "correct scientific name of a taxon (according to the reference given)",
    "label": "taxon name"
  },
  "qid": {
    "description": "subgenus of Nephus",
    "label": "Pachyspathe"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
        "Pachyspathe"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Pachyspathe": 1
      },
      "new_unique": [
        "Pachyspathe"
      ],
      "new_values": [
        "Pachyspathe"
      ],
      "new_values_raw": [
        "Pachyspathe"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Nephus": 1
      },
      "old_unique": [
        "Nephus"
      ],
      "old_values": [
        "Nephus"
      ],
      "old_values_raw": [
        "Nephus"
      ],
      "removed_unique_values": [
        "Nephus"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Nephus": {
          "new": 0,
          "old": 1
        },
        "Pachyspathe": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "pachyspathe",
          "raw_match_text": "Pachyspathe",
          "source": "FOCUS_LABEL",
          "token": "Pachyspathe"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Nephus"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q137392742_2442642262`

| Field | Value |
|---|---|
| qid | Q137392742 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q137392742::P225 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Vermeulenia"] |
| classification_target_tokens | ["Vermeulenia"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Vermeulenia"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Vermeulenia"
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
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Brya",
  "kind": "A_BOX",
  "new_value": [
    "Vermeulenia"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2442642262,
  "value": [
    "Vermeulenia"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Vermeulenia"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Vermeulenia": 1
    },
    "new_unique": [
      "Vermeulenia"
    ],
    "new_values": [
      "Vermeulenia"
    ],
    "new_values_raw": [
      "Vermeulenia"
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
      "Vermeulenia": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-17T13:40:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2443413974,
  "report_revision_old": 2442991823,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
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
  "local_ids_count": 7,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "vermeulenia",
      "raw_match_text": "Vermeulenia",
      "source": "FOCUS_LABEL",
      "token": "Vermeulenia"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Vermeulenia"
  ],
  "truth_tokens_in_recorded_matches": [
    "Vermeulenia"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "correct scientific name of a taxon (according to the reference given)",
    "label": "taxon name"
  },
  "qid": {
    "description": "genus of plants",
    "label": "Vermeulenia"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
        "Vermeulenia"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Vermeulenia": 1
      },
      "new_unique": [
        "Vermeulenia"
      ],
      "new_values": [
        "Vermeulenia"
      ],
      "new_values_raw": [
        "Vermeulenia"
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
        "Vermeulenia": {
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "vermeulenia",
          "raw_match_text": "Vermeulenia",
          "source": "FOCUS_LABEL",
          "token": "Vermeulenia"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q137397574_2442915141`

| Field | Value |
|---|---|
| qid | Q137397574 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q137397574::P225 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Trochilus rubricauda"] |
| classification_target_tokens | ["Trochilus rubricauda"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Trochilus rubricauda"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Trochilus rubricauda"
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
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Brya",
  "kind": "A_BOX",
  "new_value": [
    "Trochilus rubricauda"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2442915141,
  "value": [
    "Trochilus rubricauda"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Trochilus rubricauda"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Trochilus rubricauda": 1
    },
    "new_unique": [
      "Trochilus rubricauda"
    ],
    "new_values": [
      "Trochilus rubricauda"
    ],
    "new_values_raw": [
      "Trochilus rubricauda"
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
      "Trochilus rubricauda": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-20T11:45:08",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2444477567,
  "report_revision_old": 2444041771,
  "report_violation_type": "Allowed qualifiers",
  "report_violation_type_normalized": "Allowed qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Allowed qualifiers",
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
  "local_ids_count": 11,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "trochilus rubricauda",
      "raw_match_text": "Trochilus rubricauda",
      "source": "FOCUS_LABEL",
      "token": "Trochilus rubricauda"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Trochilus rubricauda"
  ],
  "truth_tokens_in_recorded_matches": [
    "Trochilus rubricauda"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "correct scientific name of a taxon (according to the reference given)",
    "label": "taxon name"
  },
  "qid": {
    "description": "species of bird",
    "label": "Trochilus rubricauda"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
        "Trochilus rubricauda"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Trochilus rubricauda": 1
      },
      "new_unique": [
        "Trochilus rubricauda"
      ],
      "new_values": [
        "Trochilus rubricauda"
      ],
      "new_values_raw": [
        "Trochilus rubricauda"
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
        "Trochilus rubricauda": {
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
      "local_ids_count": 11,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "trochilus rubricauda",
          "raw_match_text": "Trochilus rubricauda",
          "source": "FOCUS_LABEL",
          "token": "Trochilus rubricauda"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q144617_2445327292`

| Field | Value |
|---|---|
| qid | Q144617 |
| property | P4264 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q52060874 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q144617::P4264 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["tim"] |
| classification_target_tokens | ["tim"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_normalized_exact |
| local_match_source | NEIGHBOR_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "tim"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "tim"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "telecom-italia"
  ],
  "removed_unique_values": [
    "telecom-italia"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Scip.",
  "kind": "A_BOX",
  "new_value": [
    "tim"
  ],
  "old_value": [
    "telecom-italia"
  ],
  "revision_id": 2445327292,
  "value": [
    "tim"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "tim"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "tim": 1
    },
    "new_unique": [
      "tim"
    ],
    "new_values": [
      "tim"
    ],
    "new_values_raw": [
      "tim"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "telecom-italia": 1
    },
    "old_unique": [
      "telecom-italia"
    ],
    "old_values": [
      "telecom-italia"
    ],
    "old_values_raw": [
      "telecom-italia"
    ],
    "removed_unique_values": [
      "telecom-italia"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "telecom-italia": {
        "new": 0,
        "old": 1
      },
      "tim": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-22T06:58:16",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4264",
  "report_revision_new": 2445384768,
  "report_revision_old": 2444824898,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "telecom-italia"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 95,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_normalized_exact",
      "normalized_match_text": "tim",
      "raw_match_text": "TIM",
      "source": "NEIGHBOR_LABEL",
      "token": "tim"
    }
  ],
  "needed": 1,
  "sources_used": [
    "NEIGHBOR_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "telecom-italia"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "tim"
  ],
  "truth_tokens_in_recorded_matches": [
    "tim"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for an official company, school, organisation page, or showcase page, on LinkedIn",
    "label": "LinkedIn company or organization ID"
  },
  "qid": {
    "description": "Italian telecommunications company",
    "label": "TIM Group"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "tim"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "tim": 1
      },
      "new_unique": [
        "tim"
      ],
      "new_values": [
        "tim"
      ],
      "new_values_raw": [
        "tim"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "telecom-italia": 1
      },
      "old_unique": [
        "telecom-italia"
      ],
      "old_values": [
        "telecom-italia"
      ],
      "old_values_raw": [
        "telecom-italia"
      ],
      "removed_unique_values": [
        "telecom-italia"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "telecom-italia": {
          "new": 0,
          "old": 1
        },
        "tim": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 95,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_normalized_exact",
          "normalized_match_text": "tim",
          "raw_match_text": "TIM",
          "source": "NEIGHBOR_LABEL",
          "token": "tim"
        }
      ],
      "needed": 1,
      "sources_used": [
        "NEIGHBOR_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "telecom-italia"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 011. `repair_Q1578657_2447257297`

| Field | Value |
|---|---|
| qid | Q1578657 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q1578657::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Sony Ericsson W995"] |
| classification_target_tokens | ["Sony Ericsson W995"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Sony Ericsson W995"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Sony Ericsson W995"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Sony Ericsson W995i"
  ],
  "removed_unique_values": [
    "Sony Ericsson W995i"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Pi bot",
  "kind": "A_BOX",
  "new_value": [
    "Sony Ericsson W995"
  ],
  "old_value": [
    "Sony Ericsson W995i"
  ],
  "revision_id": 2447257297,
  "value": [
    "Sony Ericsson W995"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Sony Ericsson W995"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Sony Ericsson W995": 1
    },
    "new_unique": [
      "Sony Ericsson W995"
    ],
    "new_values": [
      "Sony Ericsson W995"
    ],
    "new_values_raw": [
      "Sony Ericsson W995"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Sony Ericsson W995i": 1
    },
    "old_unique": [
      "Sony Ericsson W995i"
    ],
    "old_values": [
      "Sony Ericsson W995i"
    ],
    "old_values_raw": [
      "Sony Ericsson W995i"
    ],
    "removed_unique_values": [
      "Sony Ericsson W995i"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Sony Ericsson W995": {
        "new": 1,
        "old": 0
      },
      "Sony Ericsson W995i": {
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
  "report_fix_date": "2025-12-27T12:35:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447772714,
  "report_revision_old": 2447382517,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Sony Ericsson W995i"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 5,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "sony ericsson w995",
      "raw_match_text": "Sony Ericsson W995",
      "source": "FOCUS_LABEL",
      "token": "Sony Ericsson W995"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Sony Ericsson W995i"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Sony Ericsson W995"
  ],
  "truth_tokens_in_recorded_matches": [
    "Sony Ericsson W995"
  ],
  "used_literal_substring": false
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
    "description": "cell phone model",
    "label": "Sony Ericsson W995"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Sony Ericsson W995"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Sony Ericsson W995": 1
      },
      "new_unique": [
        "Sony Ericsson W995"
      ],
      "new_values": [
        "Sony Ericsson W995"
      ],
      "new_values_raw": [
        "Sony Ericsson W995"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Sony Ericsson W995i": 1
      },
      "old_unique": [
        "Sony Ericsson W995i"
      ],
      "old_values": [
        "Sony Ericsson W995i"
      ],
      "old_values_raw": [
        "Sony Ericsson W995i"
      ],
      "removed_unique_values": [
        "Sony Ericsson W995i"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Sony Ericsson W995": {
          "new": 1,
          "old": 0
        },
        "Sony Ericsson W995i": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 5,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "sony ericsson w995",
          "raw_match_text": "Sony Ericsson W995",
          "source": "FOCUS_LABEL",
          "token": "Sony Ericsson W995"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Sony Ericsson W995i"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 012. `repair_Q20428060_2442783962`

| Field | Value |
|---|---|
| qid | Q20428060 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q20428060::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Stations of the Cross in Dobrá Voda u Českých Budějovic"] |
| classification_target_tokens | ["Stations of the Cross in Dobrá Voda u Českých Budějovic"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Stations of the Cross in Dobrá Voda u Českých Budějovic"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Stations of the Cross in Dobrá Voda u Českých Budějovic"
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
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Stations of the Cross in Dobrá Voda u Českých Budějovic"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2442783962,
  "value": [
    "Stations of the Cross in Dobrá Voda u Českých Budějovic"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Stations of the Cross in Dobrá Voda u Českých Budějovic"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Stations of the Cross in Dobrá Voda u Českých Budějovic": 1
    },
    "new_unique": [
      "Stations of the Cross in Dobrá Voda u Českých Budějovic"
    ],
    "new_values": [
      "Stations of the Cross in Dobrá Voda u Českých Budějovic"
    ],
    "new_values_raw": [
      "Stations of the Cross in Dobrá Voda u Českých Budějovic"
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
      "Stations of the Cross in Dobrá Voda u Českých Budějovic": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-17T12:39:08",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2443399922,
  "report_revision_old": 2442981743,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Commons link"
  ],
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
  "local_ids_count": 39,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "stations of the cross in dobrá voda u českých budějovic",
      "raw_match_text": "Stations of the Cross in Dobrá Voda u Českých Budějovic",
      "source": "FOCUS_LABEL",
      "token": "Stations of the Cross in Dobrá Voda u Českých Budějovic"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Stations of the Cross in Dobrá Voda u Českých Budějovic"
  ],
  "truth_tokens_in_recorded_matches": [
    "Stations of the Cross in Dobrá Voda u Českých Budějovic"
  ],
  "used_literal_substring": false
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
    "description": "kulturní památka České republiky v obci Dobrá Voda u Českých Budějovic",
    "label": "Stations of the Cross in Dobrá Voda u Českých Budějovic"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Stations of the Cross in Dobrá Voda u Českých Budějovic"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Stations of the Cross in Dobrá Voda u Českých Budějovic": 1
      },
      "new_unique": [
        "Stations of the Cross in Dobrá Voda u Českých Budějovic"
      ],
      "new_values": [
        "Stations of the Cross in Dobrá Voda u Českých Budějovic"
      ],
      "new_values_raw": [
        "Stations of the Cross in Dobrá Voda u Českých Budějovic"
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
        "Stations of the Cross in Dobrá Voda u Českých Budějovic": {
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
      "local_ids_count": 39,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "stations of the cross in dobrá voda u českých budějovic",
          "raw_match_text": "Stations of the Cross in Dobrá Voda u Českých Budějovic",
          "source": "FOCUS_LABEL",
          "token": "Stations of the Cross in Dobrá Voda u Českých Budějovic"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 013. `repair_Q205953_2441113979`

| Field | Value |
|---|---|
| qid | Q205953 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q205953::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Fedor Emelianenko"] |
| classification_target_tokens | ["Fedor Emelianenko"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Fedor Emelianenko"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Fedor Emelianenko"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Fedor Emelianenkov"
  ],
  "removed_unique_values": [
    "Fedor Emelianenkov"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Pi bot",
  "kind": "A_BOX",
  "new_value": [
    "Fedor Emelianenko"
  ],
  "old_value": [
    "Fedor Emelianenkov"
  ],
  "revision_id": 2441113979,
  "value": [
    "Fedor Emelianenko"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Fedor Emelianenko"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Fedor Emelianenko": 1
    },
    "new_unique": [
      "Fedor Emelianenko"
    ],
    "new_values": [
      "Fedor Emelianenko"
    ],
    "new_values_raw": [
      "Fedor Emelianenko"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Fedor Emelianenkov": 1
    },
    "old_unique": [
      "Fedor Emelianenkov"
    ],
    "old_values": [
      "Fedor Emelianenkov"
    ],
    "old_values_raw": [
      "Fedor Emelianenkov"
    ],
    "removed_unique_values": [
      "Fedor Emelianenkov"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Fedor Emelianenko": {
        "new": 1,
        "old": 0
      },
      "Fedor Emelianenkov": {
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
  "report_fix_date": "2025-12-13T10:46:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2441779130,
  "report_revision_old": 2441207555,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Fedor Emelianenkov"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 78,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "fedor emelianenko",
      "raw_match_text": "Fedor Emelianenko",
      "source": "FOCUS_LABEL",
      "token": "Fedor Emelianenko"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Fedor Emelianenkov"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Fedor Emelianenko"
  ],
  "truth_tokens_in_recorded_matches": [
    "Fedor Emelianenko"
  ],
  "used_literal_substring": false
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
    "description": "Russian mixed martial arts fighter",
    "label": "Fedor Emelianenko"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Fedor Emelianenko"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Fedor Emelianenko": 1
      },
      "new_unique": [
        "Fedor Emelianenko"
      ],
      "new_values": [
        "Fedor Emelianenko"
      ],
      "new_values_raw": [
        "Fedor Emelianenko"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Fedor Emelianenkov": 1
      },
      "old_unique": [
        "Fedor Emelianenkov"
      ],
      "old_values": [
        "Fedor Emelianenkov"
      ],
      "old_values_raw": [
        "Fedor Emelianenkov"
      ],
      "removed_unique_values": [
        "Fedor Emelianenkov"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Fedor Emelianenko": {
          "new": 1,
          "old": 0
        },
        "Fedor Emelianenkov": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 78,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "fedor emelianenko",
          "raw_match_text": "Fedor Emelianenko",
          "source": "FOCUS_LABEL",
          "token": "Fedor Emelianenko"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Fedor Emelianenkov"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 014. `repair_Q25239177_2447091414`

| Field | Value |
|---|---|
| qid | Q25239177 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q25239177::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Folk festivals in Japan"] |
| classification_target_tokens | ["Folk festivals in Japan"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Folk festivals in Japan"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Folk festivals in Japan"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Folk festivals of Japan"
  ],
  "removed_unique_values": [
    "Folk festivals of Japan"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Folk festivals in Japan"
  ],
  "old_value": [
    "Folk festivals of Japan"
  ],
  "revision_id": 2447091414,
  "value": [
    "Folk festivals in Japan"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Folk festivals in Japan"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Folk festivals in Japan": 1
    },
    "new_unique": [
      "Folk festivals in Japan"
    ],
    "new_values": [
      "Folk festivals in Japan"
    ],
    "new_values_raw": [
      "Folk festivals in Japan"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Folk festivals of Japan": 1
    },
    "old_unique": [
      "Folk festivals of Japan"
    ],
    "old_values": [
      "Folk festivals of Japan"
    ],
    "old_values_raw": [
      "Folk festivals of Japan"
    ],
    "removed_unique_values": [
      "Folk festivals of Japan"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Folk festivals in Japan": {
        "new": 1,
        "old": 0
      },
      "Folk festivals of Japan": {
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
  "report_fix_date": "2025-12-27T12:35:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447772714,
  "report_revision_old": 2447382517,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Folk festivals of Japan"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_boundary",
      "normalized_match_text": "category:folk festivals in japan",
      "raw_match_text": "Category:Folk festivals in Japan",
      "source": "FOCUS_LABEL",
      "token": "Folk festivals in Japan"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Folk festivals of Japan"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Folk festivals in Japan"
  ],
  "truth_tokens_in_recorded_matches": [
    "Folk festivals in Japan"
  ],
  "used_literal_substring": true
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
    "description": "Wikimedia category",
    "label": "Category:Folk festivals in Japan"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Folk festivals in Japan"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Folk festivals in Japan": 1
      },
      "new_unique": [
        "Folk festivals in Japan"
      ],
      "new_values": [
        "Folk festivals in Japan"
      ],
      "new_values_raw": [
        "Folk festivals in Japan"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Folk festivals of Japan": 1
      },
      "old_unique": [
        "Folk festivals of Japan"
      ],
      "old_values": [
        "Folk festivals of Japan"
      ],
      "old_values_raw": [
        "Folk festivals of Japan"
      ],
      "removed_unique_values": [
        "Folk festivals of Japan"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Folk festivals in Japan": {
          "new": 1,
          "old": 0
        },
        "Folk festivals of Japan": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_boundary",
          "normalized_match_text": "category:folk festivals in japan",
          "raw_match_text": "Category:Folk festivals in Japan",
          "source": "FOCUS_LABEL",
          "token": "Folk festivals in Japan"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
      ],
      "used_literal_substring": true
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Folk festivals of Japan"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 015. `repair_Q2756_2447374243`

| Field | Value |
|---|---|
| qid | Q2756 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q2756::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Siena FC"] |
| classification_target_tokens | ["Siena FC"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Siena FC"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Siena FC"
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
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "חזרתי",
  "kind": "A_BOX",
  "new_value": [
    "Siena FC"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2447374243,
  "value": [
    "Siena FC"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Siena FC"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Siena FC": 1
    },
    "new_unique": [
      "Siena FC"
    ],
    "new_values": [
      "Siena FC"
    ],
    "new_values_raw": [
      "Siena FC"
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
      "Siena FC": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T12:35:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447772714,
  "report_revision_old": 2447382517,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
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
  "local_ids_count": 39,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "siena fc",
      "raw_match_text": "Siena FC",
      "source": "FOCUS_LABEL",
      "token": "Siena FC"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Siena FC"
  ],
  "truth_tokens_in_recorded_matches": [
    "Siena FC"
  ],
  "used_literal_substring": false
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
    "description": "Italian football club based in Siena, Tuscany",
    "label": "Siena FC"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Siena FC"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Siena FC": 1
      },
      "new_unique": [
        "Siena FC"
      ],
      "new_values": [
        "Siena FC"
      ],
      "new_values_raw": [
        "Siena FC"
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
        "Siena FC": {
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
      "local_ids_count": 39,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "siena fc",
          "raw_match_text": "Siena FC",
          "source": "FOCUS_LABEL",
          "token": "Siena FC"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 016. `repair_Q32250854_2447091819`

| Field | Value |
|---|---|
| qid | Q32250854 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q32250854::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Vikulovsky District"] |
| classification_target_tokens | ["Vikulovsky District"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | NEIGHBOR_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Vikulovsky District"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Vikulovsky District"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Vikulovo rayon"
  ],
  "removed_unique_values": [
    "Vikulovo rayon"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Vikulovsky District"
  ],
  "old_value": [
    "Vikulovo rayon"
  ],
  "revision_id": 2447091819,
  "value": [
    "Vikulovsky District"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Vikulovsky District"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Vikulovsky District": 1
    },
    "new_unique": [
      "Vikulovsky District"
    ],
    "new_values": [
      "Vikulovsky District"
    ],
    "new_values_raw": [
      "Vikulovsky District"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Vikulovo rayon": 1
    },
    "old_unique": [
      "Vikulovo rayon"
    ],
    "old_values": [
      "Vikulovo rayon"
    ],
    "old_values_raw": [
      "Vikulovo rayon"
    ],
    "removed_unique_values": [
      "Vikulovo rayon"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Vikulovo rayon": {
        "new": 0,
        "old": 1
      },
      "Vikulovsky District": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T12:35:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447772714,
  "report_revision_old": 2447382517,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Vikulovo rayon"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 5,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "vikulovsky district",
      "raw_match_text": "Vikulovsky District",
      "source": "NEIGHBOR_LABEL",
      "token": "Vikulovsky District"
    }
  ],
  "needed": 1,
  "sources_used": [
    "NEIGHBOR_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Vikulovo rayon"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Vikulovsky District"
  ],
  "truth_tokens_in_recorded_matches": [
    "Vikulovsky District"
  ],
  "used_literal_substring": false
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
    "description": "Wikimedia category",
    "label": "Category:Vikulovo rayon"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Vikulovsky District"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Vikulovsky District": 1
      },
      "new_unique": [
        "Vikulovsky District"
      ],
      "new_values": [
        "Vikulovsky District"
      ],
      "new_values_raw": [
        "Vikulovsky District"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Vikulovo rayon": 1
      },
      "old_unique": [
        "Vikulovo rayon"
      ],
      "old_values": [
        "Vikulovo rayon"
      ],
      "old_values_raw": [
        "Vikulovo rayon"
      ],
      "removed_unique_values": [
        "Vikulovo rayon"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Vikulovo rayon": {
          "new": 0,
          "old": 1
        },
        "Vikulovsky District": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 5,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "vikulovsky district",
          "raw_match_text": "Vikulovsky District",
          "source": "NEIGHBOR_LABEL",
          "token": "Vikulovsky District"
        }
      ],
      "needed": 1,
      "sources_used": [
        "NEIGHBOR_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Vikulovo rayon"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 017. `repair_Q334056_2447087909`

| Field | Value |
|---|---|
| qid | Q334056 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q334056::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Muri-Gries Abbey"] |
| classification_target_tokens | ["Muri-Gries Abbey"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Muri-Gries Abbey"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Muri-Gries Abbey"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Abtei Muri-Gries"
  ],
  "removed_unique_values": [
    "Abtei Muri-Gries"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Muri-Gries Abbey"
  ],
  "old_value": [
    "Abtei Muri-Gries"
  ],
  "revision_id": 2447087909,
  "value": [
    "Muri-Gries Abbey"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Muri-Gries Abbey"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Muri-Gries Abbey": 1
    },
    "new_unique": [
      "Muri-Gries Abbey"
    ],
    "new_values": [
      "Muri-Gries Abbey"
    ],
    "new_values_raw": [
      "Muri-Gries Abbey"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Abtei Muri-Gries": 1
    },
    "old_unique": [
      "Abtei Muri-Gries"
    ],
    "old_values": [
      "Abtei Muri-Gries"
    ],
    "old_values_raw": [
      "Abtei Muri-Gries"
    ],
    "removed_unique_values": [
      "Abtei Muri-Gries"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Abtei Muri-Gries": {
        "new": 0,
        "old": 1
      },
      "Muri-Gries Abbey": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T12:35:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447772714,
  "report_revision_old": 2447382517,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Abtei Muri-Gries"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 21,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "muri-gries abbey",
      "raw_match_text": "Muri-Gries Abbey",
      "source": "FOCUS_LABEL",
      "token": "Muri-Gries Abbey"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Abtei Muri-Gries"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Muri-Gries Abbey"
  ],
  "truth_tokens_in_recorded_matches": [
    "Muri-Gries Abbey"
  ],
  "used_literal_substring": false
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
    "description": "abbey in South Tyrol (Italy)",
    "label": "Muri-Gries Abbey"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Muri-Gries Abbey"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Muri-Gries Abbey": 1
      },
      "new_unique": [
        "Muri-Gries Abbey"
      ],
      "new_values": [
        "Muri-Gries Abbey"
      ],
      "new_values_raw": [
        "Muri-Gries Abbey"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Abtei Muri-Gries": 1
      },
      "old_unique": [
        "Abtei Muri-Gries"
      ],
      "old_values": [
        "Abtei Muri-Gries"
      ],
      "old_values_raw": [
        "Abtei Muri-Gries"
      ],
      "removed_unique_values": [
        "Abtei Muri-Gries"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Abtei Muri-Gries": {
          "new": 0,
          "old": 1
        },
        "Muri-Gries Abbey": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 21,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "muri-gries abbey",
          "raw_match_text": "Muri-Gries Abbey",
          "source": "FOCUS_LABEL",
          "token": "Muri-Gries Abbey"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Abtei Muri-Gries"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 018. `repair_Q3797851_2442499584`

| Field | Value |
|---|---|
| qid | Q3797851 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q3797851::P225 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Mopalia schrencki"] |
| classification_target_tokens | ["Mopalia schrencki"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Mopalia schrencki"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Mopalia schrencki"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Mopalia schrenck"
  ],
  "removed_unique_values": [
    "Mopalia schrenck"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Brya",
  "kind": "A_BOX",
  "new_value": [
    "Mopalia schrencki"
  ],
  "old_value": [
    "Mopalia schrenck"
  ],
  "revision_id": 2442499584,
  "value": [
    "Mopalia schrencki"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Mopalia schrencki"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Mopalia schrencki": 1
    },
    "new_unique": [
      "Mopalia schrencki"
    ],
    "new_values": [
      "Mopalia schrencki"
    ],
    "new_values_raw": [
      "Mopalia schrencki"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Mopalia schrenck": 1
    },
    "old_unique": [
      "Mopalia schrenck"
    ],
    "old_values": [
      "Mopalia schrenck"
    ],
    "old_values_raw": [
      "Mopalia schrenck"
    ],
    "removed_unique_values": [
      "Mopalia schrenck"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Mopalia schrenck": {
        "new": 0,
        "old": 1
      },
      "Mopalia schrencki": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-16T12:38:46",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2442991823,
  "report_revision_old": 2442670671,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Mopalia schrenck"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "mopalia schrencki",
      "raw_match_text": "Mopalia schrencki",
      "source": "FOCUS_LABEL",
      "token": "Mopalia schrencki"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Mopalia schrenck"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Mopalia schrencki"
  ],
  "truth_tokens_in_recorded_matches": [
    "Mopalia schrencki"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "correct scientific name of a taxon (according to the reference given)",
    "label": "taxon name"
  },
  "qid": {
    "description": "species of mollusc",
    "label": "Mopalia schrencki"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
        "Mopalia schrencki"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Mopalia schrencki": 1
      },
      "new_unique": [
        "Mopalia schrencki"
      ],
      "new_values": [
        "Mopalia schrencki"
      ],
      "new_values_raw": [
        "Mopalia schrencki"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Mopalia schrenck": 1
      },
      "old_unique": [
        "Mopalia schrenck"
      ],
      "old_values": [
        "Mopalia schrenck"
      ],
      "old_values_raw": [
        "Mopalia schrenck"
      ],
      "removed_unique_values": [
        "Mopalia schrenck"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Mopalia schrenck": {
          "new": 0,
          "old": 1
        },
        "Mopalia schrencki": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "mopalia schrencki",
          "raw_match_text": "Mopalia schrencki",
          "source": "FOCUS_LABEL",
          "token": "Mopalia schrencki"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Mopalia schrenck"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 019. `repair_Q5360360_2447089476`

| Field | Value |
|---|---|
| qid | Q5360360 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q5360360::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Elia, Keryneias"] |
| classification_target_tokens | ["Elia, Keryneias"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_normalized_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Elia, Keryneias"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Elia, Keryneias"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Elia Keryneias"
  ],
  "removed_unique_values": [
    "Elia Keryneias"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Elia, Keryneias"
  ],
  "old_value": [
    "Elia Keryneias"
  ],
  "revision_id": 2447089476,
  "value": [
    "Elia, Keryneias"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Elia, Keryneias"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Elia, Keryneias": 1
    },
    "new_unique": [
      "Elia, Keryneias"
    ],
    "new_values": [
      "Elia, Keryneias"
    ],
    "new_values_raw": [
      "Elia, Keryneias"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Elia Keryneias": 1
    },
    "old_unique": [
      "Elia Keryneias"
    ],
    "old_values": [
      "Elia Keryneias"
    ],
    "old_values_raw": [
      "Elia Keryneias"
    ],
    "removed_unique_values": [
      "Elia Keryneias"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Elia Keryneias": {
        "new": 0,
        "old": 1
      },
      "Elia, Keryneias": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T12:35:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447772714,
  "report_revision_old": 2447382517,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Elia Keryneias"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 23,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_normalized_exact",
      "normalized_match_text": "elia keryneias",
      "raw_match_text": "Elia Keryneias",
      "source": "FOCUS_LABEL",
      "token": "Elia, Keryneias"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Elia Keryneias"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Elia, Keryneias"
  ],
  "truth_tokens_in_recorded_matches": [
    "Elia, Keryneias"
  ],
  "used_literal_substring": false
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
    "description": "village in Kyrenia District, Cyprus",
    "label": "Elia Keryneias"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Elia, Keryneias"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Elia, Keryneias": 1
      },
      "new_unique": [
        "Elia, Keryneias"
      ],
      "new_values": [
        "Elia, Keryneias"
      ],
      "new_values_raw": [
        "Elia, Keryneias"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Elia Keryneias": 1
      },
      "old_unique": [
        "Elia Keryneias"
      ],
      "old_values": [
        "Elia Keryneias"
      ],
      "old_values_raw": [
        "Elia Keryneias"
      ],
      "removed_unique_values": [
        "Elia Keryneias"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Elia Keryneias": {
          "new": 0,
          "old": 1
        },
        "Elia, Keryneias": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 23,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_normalized_exact",
          "normalized_match_text": "elia keryneias",
          "raw_match_text": "Elia Keryneias",
          "source": "FOCUS_LABEL",
          "token": "Elia, Keryneias"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Elia Keryneias"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 020. `repair_Q61872875_2447258973`

| Field | Value |
|---|---|
| qid | Q61872875 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q61872875::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Dr. Fazıl Küçük Museum"] |
| classification_target_tokens | ["Dr. Fazıl Küçük Museum"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Dr. Fazıl Küçük Museum"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Dr. Fazıl Küçük Museum"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Dr. Fazıl Küçük Müzesi"
  ],
  "removed_unique_values": [
    "Dr. Fazıl Küçük Müzesi"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Pi bot",
  "kind": "A_BOX",
  "new_value": [
    "Dr. Fazıl Küçük Museum"
  ],
  "old_value": [
    "Dr. Fazıl Küçük Müzesi"
  ],
  "revision_id": 2447258973,
  "value": [
    "Dr. Fazıl Küçük Museum"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Dr. Fazıl Küçük Museum"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Dr. Fazıl Küçük Museum": 1
    },
    "new_unique": [
      "Dr. Fazıl Küçük Museum"
    ],
    "new_values": [
      "Dr. Fazıl Küçük Museum"
    ],
    "new_values_raw": [
      "Dr. Fazıl Küçük Museum"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Dr. Fazıl Küçük Müzesi": 1
    },
    "old_unique": [
      "Dr. Fazıl Küçük Müzesi"
    ],
    "old_values": [
      "Dr. Fazıl Küçük Müzesi"
    ],
    "old_values_raw": [
      "Dr. Fazıl Küçük Müzesi"
    ],
    "removed_unique_values": [
      "Dr. Fazıl Küçük Müzesi"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Dr. Fazıl Küçük Museum": {
        "new": 1,
        "old": 0
      },
      "Dr. Fazıl Küçük Müzesi": {
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
  "report_fix_date": "2025-12-27T12:35:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447772714,
  "report_revision_old": 2447382517,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Dr. Fazıl Küçük Müzesi"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "dr. fazıl küçük museum",
      "raw_match_text": "Dr. Fazıl Küçük Museum",
      "source": "FOCUS_LABEL",
      "token": "Dr. Fazıl Küçük Museum"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Dr. Fazıl Küçük Müzesi"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Dr. Fazıl Küçük Museum"
  ],
  "truth_tokens_in_recorded_matches": [
    "Dr. Fazıl Küçük Museum"
  ],
  "used_literal_substring": false
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
    "description": "museum in Northern Cyprus",
    "label": "Dr. Fazıl Küçük Museum"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Dr. Fazıl Küçük Museum"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Dr. Fazıl Küçük Museum": 1
      },
      "new_unique": [
        "Dr. Fazıl Küçük Museum"
      ],
      "new_values": [
        "Dr. Fazıl Küçük Museum"
      ],
      "new_values_raw": [
        "Dr. Fazıl Küçük Museum"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Dr. Fazıl Küçük Müzesi": 1
      },
      "old_unique": [
        "Dr. Fazıl Küçük Müzesi"
      ],
      "old_values": [
        "Dr. Fazıl Küçük Müzesi"
      ],
      "old_values_raw": [
        "Dr. Fazıl Küçük Müzesi"
      ],
      "removed_unique_values": [
        "Dr. Fazıl Küçük Müzesi"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Dr. Fazıl Küçük Museum": {
          "new": 1,
          "old": 0
        },
        "Dr. Fazıl Küçük Müzesi": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "dr. fazıl küçük museum",
          "raw_match_text": "Dr. Fazıl Küçük Museum",
          "source": "FOCUS_LABEL",
          "token": "Dr. Fazıl Küçük Museum"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Dr. Fazıl Küçük Müzesi"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 021. `repair_Q624028_2447088024`

| Field | Value |
|---|---|
| qid | Q624028 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q624028::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Credit default swap"] |
| classification_target_tokens | ["Credit default swap"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_normalized_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Credit default swap"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Credit default swap"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Credit defaut swap"
  ],
  "removed_unique_values": [
    "Credit defaut swap"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Credit default swap"
  ],
  "old_value": [
    "Credit defaut swap"
  ],
  "revision_id": 2447088024,
  "value": [
    "Credit default swap"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Credit default swap"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Credit default swap": 1
    },
    "new_unique": [
      "Credit default swap"
    ],
    "new_values": [
      "Credit default swap"
    ],
    "new_values_raw": [
      "Credit default swap"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Credit defaut swap": 1
    },
    "old_unique": [
      "Credit defaut swap"
    ],
    "old_values": [
      "Credit defaut swap"
    ],
    "old_values_raw": [
      "Credit defaut swap"
    ],
    "removed_unique_values": [
      "Credit defaut swap"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Credit default swap": {
        "new": 1,
        "old": 0
      },
      "Credit defaut swap": {
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
  "report_fix_date": "2025-12-27T12:35:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447772714,
  "report_revision_old": 2447382517,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Credit defaut swap"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_normalized_exact",
      "normalized_match_text": "credit default swap",
      "raw_match_text": "credit default swap",
      "source": "FOCUS_LABEL",
      "token": "Credit default swap"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Credit defaut swap"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Credit default swap"
  ],
  "truth_tokens_in_recorded_matches": [
    "Credit default swap"
  ],
  "used_literal_substring": false
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
    "description": "financial swap agreement in case of default",
    "label": "credit default swap"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Credit default swap"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Credit default swap": 1
      },
      "new_unique": [
        "Credit default swap"
      ],
      "new_values": [
        "Credit default swap"
      ],
      "new_values_raw": [
        "Credit default swap"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Credit defaut swap": 1
      },
      "old_unique": [
        "Credit defaut swap"
      ],
      "old_values": [
        "Credit defaut swap"
      ],
      "old_values_raw": [
        "Credit defaut swap"
      ],
      "removed_unique_values": [
        "Credit defaut swap"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Credit default swap": {
          "new": 1,
          "old": 0
        },
        "Credit defaut swap": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_normalized_exact",
          "normalized_match_text": "credit default swap",
          "raw_match_text": "credit default swap",
          "source": "FOCUS_LABEL",
          "token": "Credit default swap"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Credit defaut swap"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 022. `repair_Q6767_2447372569`

| Field | Value |
|---|---|
| qid | Q6767 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q6767::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["US Livorno 1915"] |
| classification_target_tokens | ["US Livorno 1915"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "US Livorno 1915"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "US Livorno 1915"
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
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "חזרתי",
  "kind": "A_BOX",
  "new_value": [
    "US Livorno 1915"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2447372569,
  "value": [
    "US Livorno 1915"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "US Livorno 1915"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "US Livorno 1915": 1
    },
    "new_unique": [
      "US Livorno 1915"
    ],
    "new_values": [
      "US Livorno 1915"
    ],
    "new_values_raw": [
      "US Livorno 1915"
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
      "US Livorno 1915": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T12:35:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447772714,
  "report_revision_old": 2447382517,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
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
  "local_ids_count": 29,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "us livorno 1915",
      "raw_match_text": "US Livorno 1915",
      "source": "FOCUS_LABEL",
      "token": "US Livorno 1915"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "US Livorno 1915"
  ],
  "truth_tokens_in_recorded_matches": [
    "US Livorno 1915"
  ],
  "used_literal_substring": false
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
    "description": "Italian football club based in Livorno, Tuscany",
    "label": "US Livorno 1915"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "US Livorno 1915"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "US Livorno 1915": 1
      },
      "new_unique": [
        "US Livorno 1915"
      ],
      "new_values": [
        "US Livorno 1915"
      ],
      "new_values_raw": [
        "US Livorno 1915"
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
        "US Livorno 1915": {
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
      "local_ids_count": 29,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "us livorno 1915",
          "raw_match_text": "US Livorno 1915",
          "source": "FOCUS_LABEL",
          "token": "US Livorno 1915"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 023. `repair_Q7886778_2447089920`

| Field | Value |
|---|---|
| qid | Q7886778 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q7886778::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Union stations"] |
| classification_target_tokens | ["Union stations"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_boundary |
| local_match_source | NEIGHBOR_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Union stations"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Union stations"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Union train stations"
  ],
  "removed_unique_values": [
    "Union train stations"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Union stations"
  ],
  "old_value": [
    "Union train stations"
  ],
  "revision_id": 2447089920,
  "value": [
    "Union stations"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Union stations"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Union stations": 1
    },
    "new_unique": [
      "Union stations"
    ],
    "new_values": [
      "Union stations"
    ],
    "new_values_raw": [
      "Union stations"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Union train stations": 1
    },
    "old_unique": [
      "Union train stations"
    ],
    "old_values": [
      "Union train stations"
    ],
    "old_values_raw": [
      "Union train stations"
    ],
    "removed_unique_values": [
      "Union train stations"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Union stations": {
        "new": 1,
        "old": 0
      },
      "Union train stations": {
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
  "report_fix_date": "2025-12-27T12:35:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447772714,
  "report_revision_old": 2447382517,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Union train stations"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_boundary",
      "normalized_match_text": "category:union stations",
      "raw_match_text": "Category:Union stations",
      "source": "NEIGHBOR_LABEL",
      "token": "Union stations"
    }
  ],
  "needed": 1,
  "sources_used": [
    "NEIGHBOR_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Union train stations"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Union stations"
  ],
  "truth_tokens_in_recorded_matches": [
    "Union stations"
  ],
  "used_literal_substring": true
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
    "description": "railway station where tracks and facilities are shared by two or more separate railway companies",
    "label": "union station"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Union stations"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Union stations": 1
      },
      "new_unique": [
        "Union stations"
      ],
      "new_values": [
        "Union stations"
      ],
      "new_values_raw": [
        "Union stations"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Union train stations": 1
      },
      "old_unique": [
        "Union train stations"
      ],
      "old_values": [
        "Union train stations"
      ],
      "old_values_raw": [
        "Union train stations"
      ],
      "removed_unique_values": [
        "Union train stations"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Union stations": {
          "new": 1,
          "old": 0
        },
        "Union train stations": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_boundary",
          "normalized_match_text": "category:union stations",
          "raw_match_text": "Category:Union stations",
          "source": "NEIGHBOR_LABEL",
          "token": "Union stations"
        }
      ],
      "needed": 1,
      "sources_used": [
        "NEIGHBOR_LABEL"
      ],
      "used_literal_substring": true
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Union train stations"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 024. `repair_Q8866784_2447090198`

| Field | Value |
|---|---|
| qid | Q8866784 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q8866784::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Union stations"] |
| classification_target_tokens | ["Union stations"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Union stations"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Union stations"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Union train stations"
  ],
  "removed_unique_values": [
    "Union train stations"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Union stations"
  ],
  "old_value": [
    "Union train stations"
  ],
  "revision_id": 2447090198,
  "value": [
    "Union stations"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Union stations"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Union stations": 1
    },
    "new_unique": [
      "Union stations"
    ],
    "new_values": [
      "Union stations"
    ],
    "new_values_raw": [
      "Union stations"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Union train stations": 1
    },
    "old_unique": [
      "Union train stations"
    ],
    "old_values": [
      "Union train stations"
    ],
    "old_values_raw": [
      "Union train stations"
    ],
    "removed_unique_values": [
      "Union train stations"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Union stations": {
        "new": 1,
        "old": 0
      },
      "Union train stations": {
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
  "report_fix_date": "2025-12-27T12:35:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447772714,
  "report_revision_old": 2447382517,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Union train stations"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 5,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_boundary",
      "normalized_match_text": "category:union stations",
      "raw_match_text": "Category:Union stations",
      "source": "FOCUS_LABEL",
      "token": "Union stations"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Union train stations"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Union stations"
  ],
  "truth_tokens_in_recorded_matches": [
    "Union stations"
  ],
  "used_literal_substring": true
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
    "description": "Wikimedia category",
    "label": "Category:Union stations"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Union stations"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Union stations": 1
      },
      "new_unique": [
        "Union stations"
      ],
      "new_values": [
        "Union stations"
      ],
      "new_values_raw": [
        "Union stations"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Union train stations": 1
      },
      "old_unique": [
        "Union train stations"
      ],
      "old_values": [
        "Union train stations"
      ],
      "old_values_raw": [
        "Union train stations"
      ],
      "removed_unique_values": [
        "Union train stations"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Union stations": {
          "new": 1,
          "old": 0
        },
        "Union train stations": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "REPLACE_1_TO_1",
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
      "local_ids_count": 5,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_boundary",
          "normalized_match_text": "category:union stations",
          "raw_match_text": "Category:Union stations",
          "source": "FOCUS_LABEL",
          "token": "Union stations"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
      ],
      "used_literal_substring": true
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Union train stations"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 025. `repair_Q910283_2440849086`

| Field | Value |
|---|---|
| qid | Q910283 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_raw |
| decision_constraint_type |   |
| group_key | ABOX::Q910283::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_TEXT_DERIVED, verify that the target is deterministically derived from independent local text.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that focus identity alone is sufficient and no domain-specific reasoning is missing.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Nippon Railway"] |
| classification_target_tokens | ["Nippon Railway"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Nippon Railway"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Nippon Railway"
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
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_text_raw",
  "constraint_family": null,
  "decision_constraint_source": "local_context",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Soren Bradley",
  "kind": "A_BOX",
  "new_value": [
    "Nippon Railway"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2440849086,
  "value": [
    "Nippon Railway"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Nippon Railway"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Nippon Railway": 1
    },
    "new_unique": [
      "Nippon Railway"
    ],
    "new_values": [
      "Nippon Railway"
    ],
    "new_values_raw": [
      "Nippon Railway"
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
      "Nippon Railway": {
        "new": 1,
        "old": 0
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-13T10:46:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2441779130,
  "report_revision_old": 2441207555,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
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
  "local_ids_count": 19,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "nippon railway",
      "raw_match_text": "Nippon Railway",
      "source": "FOCUS_LABEL",
      "token": "Nippon Railway"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Nippon Railway"
  ],
  "truth_tokens_in_recorded_matches": [
    "Nippon Railway"
  ],
  "used_literal_substring": false
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
    "description": "かつて東京府東京市に存在した鉄道事業者",
    "label": "Nippon Railway"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [
        "Nippon Railway"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Nippon Railway": 1
      },
      "new_unique": [
        "Nippon Railway"
      ],
      "new_values": [
        "Nippon Railway"
      ],
      "new_values_raw": [
        "Nippon Railway"
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
        "Nippon Railway": {
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
      "local_ids_count": 19,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "nippon railway",
          "raw_match_text": "Nippon Railway",
          "source": "FOCUS_LABEL",
          "token": "Nippon Railway"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_LABEL"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---
