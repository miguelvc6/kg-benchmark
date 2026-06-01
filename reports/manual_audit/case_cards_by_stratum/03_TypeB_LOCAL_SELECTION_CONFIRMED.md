# TypeB_LOCAL_SELECTION_CONFIRMED

Cases: 30

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q10288134_2445317449`

| Field | Value |
|---|---|
| qid | Q10288134 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q10288134::P225 |
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
| truth_tokens_preview | ["Gallinago faeroeensis"] |
| classification_target_tokens | ["Gallinago gallinago faeroeensis"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Gallinago gallinago faeroeensis"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Gallinago gallinago faeroeensis"
  ],
  "removed_unique_values": [
    "Gallinago gallinago faeroeensis"
  ],
  "retained_support_tokens": [
    "Gallinago faeroeensis"
  ],
  "retained_unique_values": [
    "Gallinago faeroeensis"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Gallinago faeroeensis"
  ],
  "old_value": [
    "Gallinago faeroeensis",
    "Gallinago gallinago faeroeensis"
  ],
  "revision_id": 2445317449,
  "value": [
    "Gallinago faeroeensis"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Gallinago faeroeensis": 1
    },
    "new_unique": [
      "Gallinago faeroeensis"
    ],
    "new_values": [
      "Gallinago faeroeensis"
    ],
    "new_values_raw": [
      "Gallinago faeroeensis"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Gallinago faeroeensis": 1,
      "Gallinago gallinago faeroeensis": 1
    },
    "old_unique": [
      "Gallinago faeroeensis",
      "Gallinago gallinago faeroeensis"
    ],
    "old_values": [
      "Gallinago faeroeensis",
      "Gallinago gallinago faeroeensis"
    ],
    "old_values_raw": [
      "Gallinago faeroeensis",
      "Gallinago gallinago faeroeensis"
    ],
    "removed_unique_values": [
      "Gallinago gallinago faeroeensis"
    ],
    "retained_unique_values": [
      "Gallinago faeroeensis"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Gallinago gallinago faeroeensis": {
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
  "report_fix_date": "2025-12-23T14:31:41",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2446076213,
  "report_revision_old": 2445477126,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Gallinago faeroeensis",
    "Gallinago gallinago faeroeensis"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "gallinago faeroeensis",
      "raw_match_text": "Gallinago faeroeensis",
      "source": "FOCUS_LABEL",
      "token": "Gallinago faeroeensis"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "gallinago faeroeensis",
      "raw_match_text": "Gallinago faeroeensis",
      "source": "FOCUS_LABEL",
      "token": "Gallinago faeroeensis"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Gallinago faeroeensis",
      "Gallinago gallinago faeroeensis"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Gallinago faeroeensis"
  ],
  "truth_tokens_in_recorded_matches": [
    "Gallinago faeroeensis"
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
    "label": "Gallinago faeroeensis"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Gallinago faeroeensis": 1
      },
      "new_unique": [
        "Gallinago faeroeensis"
      ],
      "new_values": [
        "Gallinago faeroeensis"
      ],
      "new_values_raw": [
        "Gallinago faeroeensis"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Gallinago faeroeensis": 1,
        "Gallinago gallinago faeroeensis": 1
      },
      "old_unique": [
        "Gallinago faeroeensis",
        "Gallinago gallinago faeroeensis"
      ],
      "old_values": [
        "Gallinago faeroeensis",
        "Gallinago gallinago faeroeensis"
      ],
      "old_values_raw": [
        "Gallinago faeroeensis",
        "Gallinago gallinago faeroeensis"
      ],
      "removed_unique_values": [
        "Gallinago gallinago faeroeensis"
      ],
      "retained_unique_values": [
        "Gallinago faeroeensis"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Gallinago gallinago faeroeensis": {
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
          "normalized_match_text": "gallinago faeroeensis",
          "raw_match_text": "Gallinago faeroeensis",
          "source": "FOCUS_LABEL",
          "token": "Gallinago faeroeensis"
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
        "Gallinago faeroeensis",
        "Gallinago gallinago faeroeensis"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q106310031_2446825884`

| Field | Value |
|---|---|
| qid | Q106310031 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q106310031::P225 |
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
| truth_tokens_preview | ["Chaetopleura destituta"] |
| classification_target_tokens | ["Chætopleura destituta"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Chætopleura destituta"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Chætopleura destituta"
  ],
  "removed_unique_values": [
    "Chætopleura destituta"
  ],
  "retained_support_tokens": [
    "Chaetopleura destituta"
  ],
  "retained_unique_values": [
    "Chaetopleura destituta"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Chaetopleura destituta"
  ],
  "old_value": [
    "Chaetopleura destituta",
    "Chætopleura destituta"
  ],
  "revision_id": 2446825884,
  "value": [
    "Chaetopleura destituta"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Chaetopleura destituta": 1
    },
    "new_unique": [
      "Chaetopleura destituta"
    ],
    "new_values": [
      "Chaetopleura destituta"
    ],
    "new_values_raw": [
      "Chaetopleura destituta"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Chaetopleura destituta": 1,
      "Chætopleura destituta": 1
    },
    "old_unique": [
      "Chaetopleura destituta",
      "Chætopleura destituta"
    ],
    "old_values": [
      "Chaetopleura destituta",
      "Chætopleura destituta"
    ],
    "old_values_raw": [
      "Chaetopleura destituta",
      "Chætopleura destituta"
    ],
    "removed_unique_values": [
      "Chætopleura destituta"
    ],
    "retained_unique_values": [
      "Chaetopleura destituta"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Chætopleura destituta": {
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
  "report_fix_date": "2025-12-26T13:55:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2447398980,
  "report_revision_old": 2447090178,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Chaetopleura destituta",
    "Chætopleura destituta"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "chaetopleura destituta",
      "raw_match_text": "Chaetopleura destituta",
      "source": "FOCUS_LABEL",
      "token": "Chaetopleura destituta"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "chaetopleura destituta",
      "raw_match_text": "Chaetopleura destituta",
      "source": "FOCUS_LABEL",
      "token": "Chaetopleura destituta"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Chaetopleura destituta",
      "Chætopleura destituta"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Chaetopleura destituta"
  ],
  "truth_tokens_in_recorded_matches": [
    "Chaetopleura destituta"
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
    "label": "Chaetopleura destituta"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Chaetopleura destituta": 1
      },
      "new_unique": [
        "Chaetopleura destituta"
      ],
      "new_values": [
        "Chaetopleura destituta"
      ],
      "new_values_raw": [
        "Chaetopleura destituta"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Chaetopleura destituta": 1,
        "Chætopleura destituta": 1
      },
      "old_unique": [
        "Chaetopleura destituta",
        "Chætopleura destituta"
      ],
      "old_values": [
        "Chaetopleura destituta",
        "Chætopleura destituta"
      ],
      "old_values_raw": [
        "Chaetopleura destituta",
        "Chætopleura destituta"
      ],
      "removed_unique_values": [
        "Chætopleura destituta"
      ],
      "retained_unique_values": [
        "Chaetopleura destituta"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Chætopleura destituta": {
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
          "normalized_match_text": "chaetopleura destituta",
          "raw_match_text": "Chaetopleura destituta",
          "source": "FOCUS_LABEL",
          "token": "Chaetopleura destituta"
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
        "Chaetopleura destituta",
        "Chætopleura destituta"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q107520288_2444909581`

| Field | Value |
|---|---|
| qid | Q107520288 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q107520288::P225 |
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
| truth_tokens_preview | ["Beauveria petelotii"] |
| classification_target_tokens | ["Beauveria peteloti"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Beauveria peteloti"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Beauveria peteloti"
  ],
  "removed_unique_values": [
    "Beauveria peteloti"
  ],
  "retained_support_tokens": [
    "Beauveria petelotii"
  ],
  "retained_unique_values": [
    "Beauveria petelotii"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Beauveria petelotii"
  ],
  "old_value": [
    "Beauveria petelotii",
    "Beauveria peteloti"
  ],
  "revision_id": 2444909581,
  "value": [
    "Beauveria petelotii"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Beauveria petelotii": 1
    },
    "new_unique": [
      "Beauveria petelotii"
    ],
    "new_values": [
      "Beauveria petelotii"
    ],
    "new_values_raw": [
      "Beauveria petelotii"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Beauveria peteloti": 1,
      "Beauveria petelotii": 1
    },
    "old_unique": [
      "Beauveria peteloti",
      "Beauveria petelotii"
    ],
    "old_values": [
      "Beauveria petelotii",
      "Beauveria peteloti"
    ],
    "old_values_raw": [
      "Beauveria petelotii",
      "Beauveria peteloti"
    ],
    "removed_unique_values": [
      "Beauveria peteloti"
    ],
    "retained_unique_values": [
      "Beauveria petelotii"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Beauveria peteloti": {
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
  "report_fix_date": "2025-12-22T11:22:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2445477126,
  "report_revision_old": 2444910674,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Beauveria petelotii",
    "Beauveria peteloti"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "beauveria petelotii",
      "raw_match_text": "Beauveria petelotii",
      "source": "FOCUS_LABEL",
      "token": "Beauveria petelotii"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "beauveria petelotii",
      "raw_match_text": "Beauveria petelotii",
      "source": "FOCUS_LABEL",
      "token": "Beauveria petelotii"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Beauveria petelotii",
      "Beauveria peteloti"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Beauveria petelotii"
  ],
  "truth_tokens_in_recorded_matches": [
    "Beauveria petelotii"
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
    "description": "species of fungi",
    "label": "Beauveria petelotii"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Beauveria petelotii": 1
      },
      "new_unique": [
        "Beauveria petelotii"
      ],
      "new_values": [
        "Beauveria petelotii"
      ],
      "new_values_raw": [
        "Beauveria petelotii"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Beauveria peteloti": 1,
        "Beauveria petelotii": 1
      },
      "old_unique": [
        "Beauveria peteloti",
        "Beauveria petelotii"
      ],
      "old_values": [
        "Beauveria petelotii",
        "Beauveria peteloti"
      ],
      "old_values_raw": [
        "Beauveria petelotii",
        "Beauveria peteloti"
      ],
      "removed_unique_values": [
        "Beauveria peteloti"
      ],
      "retained_unique_values": [
        "Beauveria petelotii"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Beauveria peteloti": {
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
          "normalized_match_text": "beauveria petelotii",
          "raw_match_text": "Beauveria petelotii",
          "source": "FOCUS_LABEL",
          "token": "Beauveria petelotii"
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
        "Beauveria petelotii",
        "Beauveria peteloti"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q107546502_2444908123`

| Field | Value |
|---|---|
| qid | Q107546502 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q107546502::P225 |
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
| truth_tokens_preview | ["Candida prunicola"] |
| classification_target_tokens | ["Candida pruni"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Candida pruni"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Candida pruni"
  ],
  "removed_unique_values": [
    "Candida pruni"
  ],
  "retained_support_tokens": [
    "Candida prunicola"
  ],
  "retained_unique_values": [
    "Candida prunicola"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Candida prunicola"
  ],
  "old_value": [
    "Candida prunicola",
    "Candida pruni"
  ],
  "revision_id": 2444908123,
  "value": [
    "Candida prunicola"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Candida prunicola": 1
    },
    "new_unique": [
      "Candida prunicola"
    ],
    "new_values": [
      "Candida prunicola"
    ],
    "new_values_raw": [
      "Candida prunicola"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Candida pruni": 1,
      "Candida prunicola": 1
    },
    "old_unique": [
      "Candida pruni",
      "Candida prunicola"
    ],
    "old_values": [
      "Candida prunicola",
      "Candida pruni"
    ],
    "old_values_raw": [
      "Candida prunicola",
      "Candida pruni"
    ],
    "removed_unique_values": [
      "Candida pruni"
    ],
    "retained_unique_values": [
      "Candida prunicola"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Candida pruni": {
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
  "report_fix_date": "2025-12-22T11:22:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2445477126,
  "report_revision_old": 2444910674,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Candida prunicola",
    "Candida pruni"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "candida prunicola",
      "raw_match_text": "Candida prunicola",
      "source": "FOCUS_LABEL",
      "token": "Candida prunicola"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "candida prunicola",
      "raw_match_text": "Candida prunicola",
      "source": "FOCUS_LABEL",
      "token": "Candida prunicola"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Candida prunicola",
      "Candida pruni"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Candida prunicola"
  ],
  "truth_tokens_in_recorded_matches": [
    "Candida prunicola"
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
    "description": "species of fungi",
    "label": "Candida prunicola"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Candida prunicola": 1
      },
      "new_unique": [
        "Candida prunicola"
      ],
      "new_values": [
        "Candida prunicola"
      ],
      "new_values_raw": [
        "Candida prunicola"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Candida pruni": 1,
        "Candida prunicola": 1
      },
      "old_unique": [
        "Candida pruni",
        "Candida prunicola"
      ],
      "old_values": [
        "Candida prunicola",
        "Candida pruni"
      ],
      "old_values_raw": [
        "Candida prunicola",
        "Candida pruni"
      ],
      "removed_unique_values": [
        "Candida pruni"
      ],
      "retained_unique_values": [
        "Candida prunicola"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Candida pruni": {
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
          "normalized_match_text": "candida prunicola",
          "raw_match_text": "Candida prunicola",
          "source": "FOCUS_LABEL",
          "token": "Candida prunicola"
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
        "Candida prunicola",
        "Candida pruni"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q108256260_2445830247`

| Field | Value |
|---|---|
| qid | Q108256260 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q108256260::P225 |
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
| truth_tokens_preview | ["Pleospora gaeumannii"] |
| classification_target_tokens | ["Pleospora gaeumanni"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Pleospora gaeumanni"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Pleospora gaeumanni"
  ],
  "removed_unique_values": [
    "Pleospora gaeumanni"
  ],
  "retained_support_tokens": [
    "Pleospora gaeumannii"
  ],
  "retained_unique_values": [
    "Pleospora gaeumannii"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Pleospora gaeumannii"
  ],
  "old_value": [
    "Pleospora gaeumannii",
    "Pleospora gaeumanni"
  ],
  "revision_id": 2445830247,
  "value": [
    "Pleospora gaeumannii"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Pleospora gaeumannii": 1
    },
    "new_unique": [
      "Pleospora gaeumannii"
    ],
    "new_values": [
      "Pleospora gaeumannii"
    ],
    "new_values_raw": [
      "Pleospora gaeumannii"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Pleospora gaeumanni": 1,
      "Pleospora gaeumannii": 1
    },
    "old_unique": [
      "Pleospora gaeumanni",
      "Pleospora gaeumannii"
    ],
    "old_values": [
      "Pleospora gaeumannii",
      "Pleospora gaeumanni"
    ],
    "old_values_raw": [
      "Pleospora gaeumannii",
      "Pleospora gaeumanni"
    ],
    "removed_unique_values": [
      "Pleospora gaeumanni"
    ],
    "retained_unique_values": [
      "Pleospora gaeumannii"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Pleospora gaeumanni": {
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
  "report_fix_date": "2025-12-24T13:09:37",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2446547742,
  "report_revision_old": 2446076213,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Pleospora gaeumannii",
    "Pleospora gaeumanni"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "pleospora gaeumannii",
      "raw_match_text": "Pleospora gaeumannii",
      "source": "FOCUS_LABEL",
      "token": "Pleospora gaeumannii"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "pleospora gaeumannii",
      "raw_match_text": "Pleospora gaeumannii",
      "source": "FOCUS_LABEL",
      "token": "Pleospora gaeumannii"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Pleospora gaeumannii",
      "Pleospora gaeumanni"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Pleospora gaeumannii"
  ],
  "truth_tokens_in_recorded_matches": [
    "Pleospora gaeumannii"
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
    "description": "species",
    "label": "Pleospora gaeumannii"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Pleospora gaeumannii": 1
      },
      "new_unique": [
        "Pleospora gaeumannii"
      ],
      "new_values": [
        "Pleospora gaeumannii"
      ],
      "new_values_raw": [
        "Pleospora gaeumannii"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Pleospora gaeumanni": 1,
        "Pleospora gaeumannii": 1
      },
      "old_unique": [
        "Pleospora gaeumanni",
        "Pleospora gaeumannii"
      ],
      "old_values": [
        "Pleospora gaeumannii",
        "Pleospora gaeumanni"
      ],
      "old_values_raw": [
        "Pleospora gaeumannii",
        "Pleospora gaeumanni"
      ],
      "removed_unique_values": [
        "Pleospora gaeumanni"
      ],
      "retained_unique_values": [
        "Pleospora gaeumannii"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Pleospora gaeumanni": {
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
          "normalized_match_text": "pleospora gaeumannii",
          "raw_match_text": "Pleospora gaeumannii",
          "source": "FOCUS_LABEL",
          "token": "Pleospora gaeumannii"
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
        "Pleospora gaeumannii",
        "Pleospora gaeumanni"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q11330209_2445317178`

| Field | Value |
|---|---|
| qid | Q11330209 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q11330209::P225 |
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
| truth_tokens_preview | ["Daphniphyllum teijsmannii"] |
| classification_target_tokens | ["Daphniphyllum teysmannii"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Daphniphyllum teysmannii"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Daphniphyllum teysmannii"
  ],
  "removed_unique_values": [
    "Daphniphyllum teysmannii"
  ],
  "retained_support_tokens": [
    "Daphniphyllum teijsmannii"
  ],
  "retained_unique_values": [
    "Daphniphyllum teijsmannii"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Daphniphyllum teijsmannii"
  ],
  "old_value": [
    "Daphniphyllum teijsmannii",
    "Daphniphyllum teysmannii"
  ],
  "revision_id": 2445317178,
  "value": [
    "Daphniphyllum teijsmannii"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Daphniphyllum teijsmannii": 1
    },
    "new_unique": [
      "Daphniphyllum teijsmannii"
    ],
    "new_values": [
      "Daphniphyllum teijsmannii"
    ],
    "new_values_raw": [
      "Daphniphyllum teijsmannii"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Daphniphyllum teijsmannii": 1,
      "Daphniphyllum teysmannii": 1
    },
    "old_unique": [
      "Daphniphyllum teijsmannii",
      "Daphniphyllum teysmannii"
    ],
    "old_values": [
      "Daphniphyllum teijsmannii",
      "Daphniphyllum teysmannii"
    ],
    "old_values_raw": [
      "Daphniphyllum teijsmannii",
      "Daphniphyllum teysmannii"
    ],
    "removed_unique_values": [
      "Daphniphyllum teysmannii"
    ],
    "retained_unique_values": [
      "Daphniphyllum teijsmannii"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Daphniphyllum teysmannii": {
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
  "report_fix_date": "2025-12-23T14:31:41",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2446076213,
  "report_revision_old": 2445477126,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Daphniphyllum teijsmannii",
    "Daphniphyllum teysmannii"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 9,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "daphniphyllum teijsmannii",
      "raw_match_text": "Daphniphyllum teijsmannii",
      "source": "FOCUS_LABEL",
      "token": "Daphniphyllum teijsmannii"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "daphniphyllum teijsmannii",
      "raw_match_text": "Daphniphyllum teijsmannii",
      "source": "FOCUS_LABEL",
      "token": "Daphniphyllum teijsmannii"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Daphniphyllum teijsmannii",
      "Daphniphyllum teysmannii"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Daphniphyllum teijsmannii"
  ],
  "truth_tokens_in_recorded_matches": [
    "Daphniphyllum teijsmannii"
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
    "description": "species of plant",
    "label": "Daphniphyllum teijsmannii"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Daphniphyllum teijsmannii": 1
      },
      "new_unique": [
        "Daphniphyllum teijsmannii"
      ],
      "new_values": [
        "Daphniphyllum teijsmannii"
      ],
      "new_values_raw": [
        "Daphniphyllum teijsmannii"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Daphniphyllum teijsmannii": 1,
        "Daphniphyllum teysmannii": 1
      },
      "old_unique": [
        "Daphniphyllum teijsmannii",
        "Daphniphyllum teysmannii"
      ],
      "old_values": [
        "Daphniphyllum teijsmannii",
        "Daphniphyllum teysmannii"
      ],
      "old_values_raw": [
        "Daphniphyllum teijsmannii",
        "Daphniphyllum teysmannii"
      ],
      "removed_unique_values": [
        "Daphniphyllum teysmannii"
      ],
      "retained_unique_values": [
        "Daphniphyllum teijsmannii"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Daphniphyllum teysmannii": {
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
      "local_ids_count": 9,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "daphniphyllum teijsmannii",
          "raw_match_text": "Daphniphyllum teijsmannii",
          "source": "FOCUS_LABEL",
          "token": "Daphniphyllum teijsmannii"
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
        "Daphniphyllum teijsmannii",
        "Daphniphyllum teysmannii"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q119112_2442434378`

| Field | Value |
|---|---|
| qid | Q119112 |
| property | P7902 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q119112::P7902 |
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
| truth_token_kind | numeric |
| truth_tokens_preview | ["119371456"] |
| classification_target_tokens | ["116658029"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_NON_TARGET_PROPERTY_TEXT |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "116658029"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "116658029"
  ],
  "removed_unique_values": [
    "116658029"
  ],
  "retained_support_tokens": [
    "119371456"
  ],
  "retained_unique_values": [
    "119371456"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
  "author": "Kolja21",
  "kind": "A_BOX",
  "new_value": [
    "119371456"
  ],
  "old_value": [
    "119371456",
    "116658029"
  ],
  "revision_id": 2442434378,
  "value": [
    "119371456"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "119371456": 1
    },
    "new_unique": [
      "119371456"
    ],
    "new_values": [
      "119371456"
    ],
    "new_values_raw": [
      "119371456"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "116658029": 1,
      "119371456": 1
    },
    "old_unique": [
      "116658029",
      "119371456"
    ],
    "old_values": [
      "119371456",
      "116658029"
    ],
    "old_values_raw": [
      "119371456",
      "116658029"
    ],
    "removed_unique_values": [
      "116658029"
    ],
    "retained_unique_values": [
      "119371456"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "116658029": {
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
  "report_fix_date": "2025-12-16T06:54:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7902",
  "report_revision_new": 2442916914,
  "report_revision_old": 2442552506,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "119371456",
    "116658029"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 32,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "119371456",
      "raw_match_text": "119371456",
      "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
      "supporting_property_id": "P227",
      "supporting_value": "119371456",
      "token": "119371456"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "119371456",
      "raw_match_text": "119371456",
      "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
      "supporting_property_id": "P227",
      "supporting_value": "119371456",
      "token": "119371456"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_NON_TARGET_PROPERTY_TEXT"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "119371456",
      "116658029"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "119371456"
  ],
  "truth_tokens_in_recorded_matches": [
    "119371456"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "GND identifier for an item in the Deutsche Biographie",
    "label": "Deutsche Biographie (GND) ID"
  },
  "qid": {
    "description": "German author (1865-1947)",
    "label": "Alexander von Gleichen-Rußwurm"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
  },
  {
    "label_en": "label in language constraint",
    "qid": "Q108139345"
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
        "119371456": 1
      },
      "new_unique": [
        "119371456"
      ],
      "new_values": [
        "119371456"
      ],
      "new_values_raw": [
        "119371456"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "116658029": 1,
        "119371456": 1
      },
      "old_unique": [
        "116658029",
        "119371456"
      ],
      "old_values": [
        "119371456",
        "116658029"
      ],
      "old_values_raw": [
        "119371456",
        "116658029"
      ],
      "removed_unique_values": [
        "116658029"
      ],
      "retained_unique_values": [
        "119371456"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "116658029": {
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
      "local_ids_count": 32,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "119371456",
          "raw_match_text": "119371456",
          "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
          "supporting_property_id": "P227",
          "supporting_value": "119371456",
          "token": "119371456"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_NON_TARGET_PROPERTY_TEXT"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "119371456",
        "116658029"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q135484336_2442690377`

| Field | Value |
|---|---|
| qid | Q135484336 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q135484336::P225 |
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
| truth_tokens_preview | ["Clavicollis laevipennis"] |
| classification_target_tokens | ["Clavicollis"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Clavicollis"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Clavicollis"
  ],
  "removed_unique_values": [
    "Clavicollis"
  ],
  "retained_support_tokens": [
    "Clavicollis laevipennis"
  ],
  "retained_unique_values": [
    "Clavicollis laevipennis"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Clavicollis laevipennis"
  ],
  "old_value": [
    "Clavicollis laevipennis",
    "Clavicollis"
  ],
  "revision_id": 2442690377,
  "value": [
    "Clavicollis laevipennis"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Clavicollis laevipennis": 1
    },
    "new_unique": [
      "Clavicollis laevipennis"
    ],
    "new_values": [
      "Clavicollis laevipennis"
    ],
    "new_values_raw": [
      "Clavicollis laevipennis"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Clavicollis": 1,
      "Clavicollis laevipennis": 1
    },
    "old_unique": [
      "Clavicollis",
      "Clavicollis laevipennis"
    ],
    "old_values": [
      "Clavicollis laevipennis",
      "Clavicollis"
    ],
    "old_values_raw": [
      "Clavicollis laevipennis",
      "Clavicollis"
    ],
    "removed_unique_values": [
      "Clavicollis"
    ],
    "retained_unique_values": [
      "Clavicollis laevipennis"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Clavicollis": {
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
  "report_fix_date": "2025-12-17T13:40:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2443413974,
  "report_revision_old": 2442991823,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "report_violation_types": [
    "Single value",
    "Unique value"
  ],
  "value": [
    "Clavicollis laevipennis",
    "Clavicollis"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "clavicollis laevipennis",
      "raw_match_text": "Clavicollis laevipennis",
      "source": "FOCUS_LABEL",
      "token": "Clavicollis laevipennis"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "clavicollis laevipennis",
      "raw_match_text": "Clavicollis laevipennis",
      "source": "FOCUS_LABEL",
      "token": "Clavicollis laevipennis"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Clavicollis laevipennis",
      "Clavicollis"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Clavicollis laevipennis"
  ],
  "truth_tokens_in_recorded_matches": [
    "Clavicollis laevipennis"
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
    "description": "taxon in the family Anthicidae",
    "label": "Clavicollis laevipennis"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Clavicollis laevipennis": 1
      },
      "new_unique": [
        "Clavicollis laevipennis"
      ],
      "new_values": [
        "Clavicollis laevipennis"
      ],
      "new_values_raw": [
        "Clavicollis laevipennis"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Clavicollis": 1,
        "Clavicollis laevipennis": 1
      },
      "old_unique": [
        "Clavicollis",
        "Clavicollis laevipennis"
      ],
      "old_values": [
        "Clavicollis laevipennis",
        "Clavicollis"
      ],
      "old_values_raw": [
        "Clavicollis laevipennis",
        "Clavicollis"
      ],
      "removed_unique_values": [
        "Clavicollis"
      ],
      "retained_unique_values": [
        "Clavicollis laevipennis"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Clavicollis": {
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
          "normalized_match_text": "clavicollis laevipennis",
          "raw_match_text": "Clavicollis laevipennis",
          "source": "FOCUS_LABEL",
          "token": "Clavicollis laevipennis"
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
        "Clavicollis laevipennis",
        "Clavicollis"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q1363545_2439874718`

| Field | Value |
|---|---|
| qid | Q1363545 |
| property | P227 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q1363545::P227 |
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
| truth_tokens_preview | ["11744894X"] |
| classification_target_tokens | ["117703508"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_NON_TARGET_PROPERTY_TEXT |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "117703508"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "117703508"
  ],
  "removed_unique_values": [
    "117703508"
  ],
  "retained_support_tokens": [
    "11744894X"
  ],
  "retained_unique_values": [
    "11744894X"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
  "author": "Kolja21",
  "kind": "A_BOX",
  "new_value": [
    "11744894X"
  ],
  "old_value": [
    "11744894X",
    "117703508"
  ],
  "revision_id": 2439874718,
  "value": [
    "11744894X"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "11744894X": 1
    },
    "new_unique": [
      "11744894X"
    ],
    "new_values": [
      "11744894X"
    ],
    "new_values_raw": [
      "11744894X"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "11744894X": 1,
      "117703508": 1
    },
    "old_unique": [
      "11744894X",
      "117703508"
    ],
    "old_values": [
      "11744894X",
      "117703508"
    ],
    "old_values_raw": [
      "11744894X",
      "117703508"
    ],
    "removed_unique_values": [
      "117703508"
    ],
    "retained_unique_values": [
      "11744894X"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "117703508": {
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
  "report_fix_date": "2025-12-10T11:06:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P227",
  "report_revision_new": 2440424022,
  "report_revision_old": 2440013759,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "11744894X",
    "117703508"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 31,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "11744894x",
      "raw_match_text": "11744894X",
      "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
      "supporting_property_id": "P7902",
      "supporting_value": "11744894X",
      "token": "11744894X"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "11744894x",
      "raw_match_text": "11744894X",
      "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
      "supporting_property_id": "P7902",
      "supporting_value": "11744894X",
      "token": "11744894X"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_NON_TARGET_PROPERTY_TEXT"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "11744894X",
      "117703508"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "11744894X"
  ],
  "truth_tokens_in_recorded_matches": [
    "11744894X"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier from the Gemeinsame Normdatei authority file of names, subjects, and organizations",
    "label": "GND ID"
  },
  "qid": {
    "description": "German painter (1775-1849)",
    "label": "Gottfried Wilhelm Voelcker"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
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
    "label_en": "label in language constraint",
    "qid": "Q108139345"
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
        "11744894X": 1
      },
      "new_unique": [
        "11744894X"
      ],
      "new_values": [
        "11744894X"
      ],
      "new_values_raw": [
        "11744894X"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "11744894X": 1,
        "117703508": 1
      },
      "old_unique": [
        "11744894X",
        "117703508"
      ],
      "old_values": [
        "11744894X",
        "117703508"
      ],
      "old_values_raw": [
        "11744894X",
        "117703508"
      ],
      "removed_unique_values": [
        "117703508"
      ],
      "retained_unique_values": [
        "11744894X"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "117703508": {
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
          "normalized_match_text": "11744894x",
          "raw_match_text": "11744894X",
          "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
          "supporting_property_id": "P7902",
          "supporting_value": "11744894X",
          "token": "11744894X"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_NON_TARGET_PROPERTY_TEXT"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "11744894X",
        "117703508"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q14410310_2443596718`

| Field | Value |
|---|---|
| qid | Q14410310 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q14410310::P225 |
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
| truth_tokens_preview | ["Strumigenys datissa"] |
| classification_target_tokens | ["Pyramica datissa"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Pyramica datissa"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Pyramica datissa"
  ],
  "removed_unique_values": [
    "Pyramica datissa"
  ],
  "retained_support_tokens": [
    "Strumigenys datissa"
  ],
  "retained_unique_values": [
    "Strumigenys datissa"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Strumigenys datissa"
  ],
  "old_value": [
    "Strumigenys datissa",
    "Pyramica datissa"
  ],
  "revision_id": 2443596718,
  "value": [
    "Strumigenys datissa"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Strumigenys datissa": 1
    },
    "new_unique": [
      "Strumigenys datissa"
    ],
    "new_values": [
      "Strumigenys datissa"
    ],
    "new_values_raw": [
      "Strumigenys datissa"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Pyramica datissa": 1,
      "Strumigenys datissa": 1
    },
    "old_unique": [
      "Pyramica datissa",
      "Strumigenys datissa"
    ],
    "old_values": [
      "Strumigenys datissa",
      "Pyramica datissa"
    ],
    "old_values_raw": [
      "Strumigenys datissa",
      "Pyramica datissa"
    ],
    "removed_unique_values": [
      "Pyramica datissa"
    ],
    "retained_unique_values": [
      "Strumigenys datissa"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Pyramica datissa": {
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
  "report_fix_date": "2025-12-19T11:31:53",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2444041771,
  "report_revision_old": 2443848675,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Strumigenys datissa",
    "Pyramica datissa"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "strumigenys datissa",
      "raw_match_text": "Strumigenys datissa",
      "source": "FOCUS_LABEL",
      "token": "Strumigenys datissa"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "strumigenys datissa",
      "raw_match_text": "Strumigenys datissa",
      "source": "FOCUS_LABEL",
      "token": "Strumigenys datissa"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Strumigenys datissa",
      "Pyramica datissa"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Strumigenys datissa"
  ],
  "truth_tokens_in_recorded_matches": [
    "Strumigenys datissa"
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
    "description": "species of insect",
    "label": "Strumigenys datissa"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Strumigenys datissa": 1
      },
      "new_unique": [
        "Strumigenys datissa"
      ],
      "new_values": [
        "Strumigenys datissa"
      ],
      "new_values_raw": [
        "Strumigenys datissa"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Pyramica datissa": 1,
        "Strumigenys datissa": 1
      },
      "old_unique": [
        "Pyramica datissa",
        "Strumigenys datissa"
      ],
      "old_values": [
        "Strumigenys datissa",
        "Pyramica datissa"
      ],
      "old_values_raw": [
        "Strumigenys datissa",
        "Pyramica datissa"
      ],
      "removed_unique_values": [
        "Pyramica datissa"
      ],
      "retained_unique_values": [
        "Strumigenys datissa"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Pyramica datissa": {
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
          "normalized_match_text": "strumigenys datissa",
          "raw_match_text": "Strumigenys datissa",
          "source": "FOCUS_LABEL",
          "token": "Strumigenys datissa"
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
        "Strumigenys datissa",
        "Pyramica datissa"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 011. `repair_Q16438784_2446326778`

| Field | Value |
|---|---|
| qid | Q16438784 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q16438784::P373 |
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
| truth_tokens_preview | ["Prosternini"] |
| classification_target_tokens | ["Prosterninae"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Prosterninae"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Prosterninae"
  ],
  "removed_unique_values": [
    "Prosterninae"
  ],
  "retained_support_tokens": [
    "Prosternini"
  ],
  "retained_unique_values": [
    "Prosternini"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Prosternini"
  ],
  "old_value": [
    "Prosternini",
    "Prosterninae"
  ],
  "revision_id": 2446326778,
  "value": [
    "Prosternini"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Prosternini": 1
    },
    "new_unique": [
      "Prosternini"
    ],
    "new_values": [
      "Prosternini"
    ],
    "new_values_raw": [
      "Prosternini"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Prosterninae": 1,
      "Prosternini": 1
    },
    "old_unique": [
      "Prosterninae",
      "Prosternini"
    ],
    "old_values": [
      "Prosternini",
      "Prosterninae"
    ],
    "old_values_raw": [
      "Prosternini",
      "Prosterninae"
    ],
    "removed_unique_values": [
      "Prosterninae"
    ],
    "retained_unique_values": [
      "Prosternini"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Prosterninae": {
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
  "report_fix_date": "2025-12-25T19:21:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447067079,
  "report_revision_old": 2446526020,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Prosternini",
    "Prosterninae"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 9,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "prosternini",
      "raw_match_text": "Prosternini",
      "source": "FOCUS_LABEL",
      "token": "Prosternini"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "prosternini",
      "raw_match_text": "Prosternini",
      "source": "FOCUS_LABEL",
      "token": "Prosternini"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Prosternini",
      "Prosterninae"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Prosternini"
  ],
  "truth_tokens_in_recorded_matches": [
    "Prosternini"
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
    "description": "tribe of insects",
    "label": "Prosternini"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Prosternini": 1
      },
      "new_unique": [
        "Prosternini"
      ],
      "new_values": [
        "Prosternini"
      ],
      "new_values_raw": [
        "Prosternini"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Prosterninae": 1,
        "Prosternini": 1
      },
      "old_unique": [
        "Prosterninae",
        "Prosternini"
      ],
      "old_values": [
        "Prosternini",
        "Prosterninae"
      ],
      "old_values_raw": [
        "Prosternini",
        "Prosterninae"
      ],
      "removed_unique_values": [
        "Prosterninae"
      ],
      "retained_unique_values": [
        "Prosternini"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Prosterninae": {
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
      "local_ids_count": 9,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "prosternini",
          "raw_match_text": "Prosternini",
          "source": "FOCUS_LABEL",
          "token": "Prosternini"
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
        "Prosternini",
        "Prosterninae"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 012. `repair_Q21355438_2444770922`

| Field | Value |
|---|---|
| qid | Q21355438 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q21355438::P225 |
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
| truth_tokens_preview | ["Scymnus syoitii"] |
| classification_target_tokens | ["Scymnus shoitii"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Scymnus shoitii"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Scymnus shoitii"
  ],
  "removed_unique_values": [
    "Scymnus shoitii"
  ],
  "retained_support_tokens": [
    "Scymnus syoitii"
  ],
  "retained_unique_values": [
    "Scymnus syoitii"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Scymnus syoitii"
  ],
  "old_value": [
    "Scymnus shoitii",
    "Scymnus syoitii"
  ],
  "revision_id": 2444770922,
  "value": [
    "Scymnus syoitii"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Scymnus syoitii": 1
    },
    "new_unique": [
      "Scymnus syoitii"
    ],
    "new_values": [
      "Scymnus syoitii"
    ],
    "new_values_raw": [
      "Scymnus syoitii"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Scymnus shoitii": 1,
      "Scymnus syoitii": 1
    },
    "old_unique": [
      "Scymnus shoitii",
      "Scymnus syoitii"
    ],
    "old_values": [
      "Scymnus shoitii",
      "Scymnus syoitii"
    ],
    "old_values_raw": [
      "Scymnus shoitii",
      "Scymnus syoitii"
    ],
    "removed_unique_values": [
      "Scymnus shoitii"
    ],
    "retained_unique_values": [
      "Scymnus syoitii"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Scymnus shoitii": {
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
  "report_fix_date": "2025-12-22T11:22:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2445477126,
  "report_revision_old": 2444910674,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Scymnus shoitii",
    "Scymnus syoitii"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "scymnus syoitii",
      "raw_match_text": "Scymnus syoitii",
      "source": "FOCUS_LABEL",
      "token": "Scymnus syoitii"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "scymnus syoitii",
      "raw_match_text": "Scymnus syoitii",
      "source": "FOCUS_LABEL",
      "token": "Scymnus syoitii"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Scymnus shoitii",
      "Scymnus syoitii"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Scymnus syoitii"
  ],
  "truth_tokens_in_recorded_matches": [
    "Scymnus syoitii"
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
    "description": "species of beetle",
    "label": "Scymnus syoitii"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Scymnus syoitii": 1
      },
      "new_unique": [
        "Scymnus syoitii"
      ],
      "new_values": [
        "Scymnus syoitii"
      ],
      "new_values_raw": [
        "Scymnus syoitii"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Scymnus shoitii": 1,
        "Scymnus syoitii": 1
      },
      "old_unique": [
        "Scymnus shoitii",
        "Scymnus syoitii"
      ],
      "old_values": [
        "Scymnus shoitii",
        "Scymnus syoitii"
      ],
      "old_values_raw": [
        "Scymnus shoitii",
        "Scymnus syoitii"
      ],
      "removed_unique_values": [
        "Scymnus shoitii"
      ],
      "retained_unique_values": [
        "Scymnus syoitii"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Scymnus shoitii": {
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
          "normalized_match_text": "scymnus syoitii",
          "raw_match_text": "Scymnus syoitii",
          "source": "FOCUS_LABEL",
          "token": "Scymnus syoitii"
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
        "Scymnus shoitii",
        "Scymnus syoitii"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 013. `repair_Q2168044_2442522650`

| Field | Value |
|---|---|
| qid | Q2168044 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q2168044::P225 |
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
| truth_tokens_preview | ["Rotaria rotatoria"] |
| classification_target_tokens | ["Rotaria rotaria"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Rotaria rotaria"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Rotaria rotaria"
  ],
  "removed_unique_values": [
    "Rotaria rotaria"
  ],
  "retained_support_tokens": [
    "Rotaria rotatoria"
  ],
  "retained_unique_values": [
    "Rotaria rotatoria"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Rotaria rotatoria"
  ],
  "old_value": [
    "Rotaria rotatoria",
    "Rotaria rotaria"
  ],
  "revision_id": 2442522650,
  "value": [
    "Rotaria rotatoria"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Rotaria rotatoria": 1
    },
    "new_unique": [
      "Rotaria rotatoria"
    ],
    "new_values": [
      "Rotaria rotatoria"
    ],
    "new_values_raw": [
      "Rotaria rotatoria"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Rotaria rotaria": 1,
      "Rotaria rotatoria": 1
    },
    "old_unique": [
      "Rotaria rotaria",
      "Rotaria rotatoria"
    ],
    "old_values": [
      "Rotaria rotatoria",
      "Rotaria rotaria"
    ],
    "old_values_raw": [
      "Rotaria rotatoria",
      "Rotaria rotaria"
    ],
    "removed_unique_values": [
      "Rotaria rotaria"
    ],
    "retained_unique_values": [
      "Rotaria rotatoria"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Rotaria rotaria": {
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
  "report_fix_date": "2025-12-16T12:38:46",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2442991823,
  "report_revision_old": 2442670671,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Rotaria rotatoria",
    "Rotaria rotaria"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 9,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "rotaria rotatoria",
      "raw_match_text": "Rotaria rotatoria",
      "source": "FOCUS_LABEL",
      "token": "Rotaria rotatoria"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "rotaria rotatoria",
      "raw_match_text": "Rotaria rotatoria",
      "source": "FOCUS_LABEL",
      "token": "Rotaria rotatoria"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Rotaria rotatoria",
      "Rotaria rotaria"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Rotaria rotatoria"
  ],
  "truth_tokens_in_recorded_matches": [
    "Rotaria rotatoria"
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
    "description": "species of rotifers",
    "label": "Rotaria rotatoria"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Rotaria rotatoria": 1
      },
      "new_unique": [
        "Rotaria rotatoria"
      ],
      "new_values": [
        "Rotaria rotatoria"
      ],
      "new_values_raw": [
        "Rotaria rotatoria"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Rotaria rotaria": 1,
        "Rotaria rotatoria": 1
      },
      "old_unique": [
        "Rotaria rotaria",
        "Rotaria rotatoria"
      ],
      "old_values": [
        "Rotaria rotatoria",
        "Rotaria rotaria"
      ],
      "old_values_raw": [
        "Rotaria rotatoria",
        "Rotaria rotaria"
      ],
      "removed_unique_values": [
        "Rotaria rotaria"
      ],
      "retained_unique_values": [
        "Rotaria rotatoria"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Rotaria rotaria": {
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
      "local_ids_count": 9,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "rotaria rotatoria",
          "raw_match_text": "Rotaria rotatoria",
          "source": "FOCUS_LABEL",
          "token": "Rotaria rotatoria"
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
        "Rotaria rotatoria",
        "Rotaria rotaria"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 014. `repair_Q2591227_2440580371`

| Field | Value |
|---|---|
| qid | Q2591227 |
| property | P227 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q2591227::P227 |
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
| truth_token_kind | numeric |
| truth_tokens_preview | ["1190610213"] |
| classification_target_tokens | ["13034334X"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_NON_TARGET_PROPERTY_TEXT |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "13034334X"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "13034334X"
  ],
  "removed_unique_values": [
    "13034334X"
  ],
  "retained_support_tokens": [
    "1190610213"
  ],
  "retained_unique_values": [
    "1190610213"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
  "author": "Epìdosis",
  "kind": "A_BOX",
  "new_value": [
    "1190610213"
  ],
  "old_value": [
    "1190610213",
    "13034334X"
  ],
  "revision_id": 2440580371,
  "value": [
    "1190610213"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1190610213": 1
    },
    "new_unique": [
      "1190610213"
    ],
    "new_values": [
      "1190610213"
    ],
    "new_values_raw": [
      "1190610213"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1190610213": 1,
      "13034334X": 1
    },
    "old_unique": [
      "1190610213",
      "13034334X"
    ],
    "old_values": [
      "1190610213",
      "13034334X"
    ],
    "old_values_raw": [
      "1190610213",
      "13034334X"
    ],
    "removed_unique_values": [
      "13034334X"
    ],
    "retained_unique_values": [
      "1190610213"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "13034334X": {
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
  "report_fix_date": "2025-12-12T12:09:04",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P227",
  "report_revision_new": 2441229556,
  "report_revision_old": 2440873708,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "1190610213",
    "13034334X"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 30,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "1190610213",
      "raw_match_text": "1190610213",
      "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
      "supporting_property_id": "P13226",
      "supporting_value": "1190610213",
      "token": "1190610213"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "1190610213",
      "raw_match_text": "1190610213",
      "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
      "supporting_property_id": "P13226",
      "supporting_value": "1190610213",
      "token": "1190610213"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_NON_TARGET_PROPERTY_TEXT"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "1190610213",
      "13034334X"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "1190610213"
  ],
  "truth_tokens_in_recorded_matches": [
    "1190610213"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier from the Gemeinsame Normdatei authority file of names, subjects, and organizations",
    "label": "GND ID"
  },
  "qid": {
    "description": "German politician (1934–2000)",
    "label": "Wolfgang Schmidt"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
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
    "label_en": "label in language constraint",
    "qid": "Q108139345"
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
        "1190610213": 1
      },
      "new_unique": [
        "1190610213"
      ],
      "new_values": [
        "1190610213"
      ],
      "new_values_raw": [
        "1190610213"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1190610213": 1,
        "13034334X": 1
      },
      "old_unique": [
        "1190610213",
        "13034334X"
      ],
      "old_values": [
        "1190610213",
        "13034334X"
      ],
      "old_values_raw": [
        "1190610213",
        "13034334X"
      ],
      "removed_unique_values": [
        "13034334X"
      ],
      "retained_unique_values": [
        "1190610213"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "13034334X": {
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
      "local_ids_count": 30,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "1190610213",
          "raw_match_text": "1190610213",
          "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
          "supporting_property_id": "P13226",
          "supporting_value": "1190610213",
          "token": "1190610213"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_NON_TARGET_PROPERTY_TEXT"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "1190610213",
        "13034334X"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 015. `repair_Q2709763_2444372614`

| Field | Value |
|---|---|
| qid | Q2709763 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q2709763::P225 |
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
| truth_tokens_preview | ["Haematommataceae"] |
| classification_target_tokens | ["Haematommaceae"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Haematommaceae"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Haematommaceae"
  ],
  "removed_unique_values": [
    "Haematommaceae"
  ],
  "retained_support_tokens": [
    "Haematommataceae"
  ],
  "retained_unique_values": [
    "Haematommataceae"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Haematommataceae"
  ],
  "old_value": [
    "Haematommataceae",
    "Haematommaceae"
  ],
  "revision_id": 2444372614,
  "value": [
    "Haematommataceae"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Haematommataceae": 1
    },
    "new_unique": [
      "Haematommataceae"
    ],
    "new_values": [
      "Haematommataceae"
    ],
    "new_values_raw": [
      "Haematommataceae"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Haematommaceae": 1,
      "Haematommataceae": 1
    },
    "old_unique": [
      "Haematommaceae",
      "Haematommataceae"
    ],
    "old_values": [
      "Haematommataceae",
      "Haematommaceae"
    ],
    "old_values_raw": [
      "Haematommataceae",
      "Haematommaceae"
    ],
    "removed_unique_values": [
      "Haematommaceae"
    ],
    "retained_unique_values": [
      "Haematommataceae"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Haematommaceae": {
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
  "report_fix_date": "2025-12-21T11:55:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2444910674,
  "report_revision_old": 2444477567,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Haematommataceae",
    "Haematommaceae"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 9,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "haematommataceae",
      "raw_match_text": "Haematommataceae",
      "source": "FOCUS_LABEL",
      "token": "Haematommataceae"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "haematommataceae",
      "raw_match_text": "Haematommataceae",
      "source": "FOCUS_LABEL",
      "token": "Haematommataceae"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Haematommataceae",
      "Haematommaceae"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Haematommataceae"
  ],
  "truth_tokens_in_recorded_matches": [
    "Haematommataceae"
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
    "description": "family of fungi",
    "label": "Haematommataceae"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Haematommataceae": 1
      },
      "new_unique": [
        "Haematommataceae"
      ],
      "new_values": [
        "Haematommataceae"
      ],
      "new_values_raw": [
        "Haematommataceae"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Haematommaceae": 1,
        "Haematommataceae": 1
      },
      "old_unique": [
        "Haematommaceae",
        "Haematommataceae"
      ],
      "old_values": [
        "Haematommataceae",
        "Haematommaceae"
      ],
      "old_values_raw": [
        "Haematommataceae",
        "Haematommaceae"
      ],
      "removed_unique_values": [
        "Haematommaceae"
      ],
      "retained_unique_values": [
        "Haematommataceae"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Haematommaceae": {
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
      "local_ids_count": 9,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "haematommataceae",
          "raw_match_text": "Haematommataceae",
          "source": "FOCUS_LABEL",
          "token": "Haematommataceae"
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
        "Haematommataceae",
        "Haematommaceae"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 016. `repair_Q284994_2439397359`

| Field | Value |
|---|---|
| qid | Q284994 |
| property | P227 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q284994::P227 |
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
| truth_token_kind | numeric |
| truth_tokens_preview | ["118560263"] |
| classification_target_tokens | ["189575476"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_NON_TARGET_PROPERTY_TEXT |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "189575476"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "189575476"
  ],
  "removed_unique_values": [
    "189575476"
  ],
  "retained_support_tokens": [
    "118560263"
  ],
  "retained_unique_values": [
    "118560263"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
  "author": "Kolja21",
  "kind": "A_BOX",
  "new_value": [
    "118560263"
  ],
  "old_value": [
    "118560263",
    "189575476"
  ],
  "revision_id": 2439397359,
  "value": [
    "118560263"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "118560263": 1
    },
    "new_unique": [
      "118560263"
    ],
    "new_values": [
      "118560263"
    ],
    "new_values_raw": [
      "118560263"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "118560263": 1,
      "189575476": 1
    },
    "old_unique": [
      "118560263",
      "189575476"
    ],
    "old_values": [
      "118560263",
      "189575476"
    ],
    "old_values_raw": [
      "118560263",
      "189575476"
    ],
    "removed_unique_values": [
      "189575476"
    ],
    "retained_unique_values": [
      "118560263"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "189575476": {
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
  "report_fix_date": "2025-12-09T12:32:17",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P227",
  "report_revision_new": 2440013759,
  "report_revision_old": 2439564475,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "118560263",
    "189575476"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 55,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "118560263",
      "raw_match_text": "118560263",
      "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
      "supporting_property_id": "P7902",
      "supporting_value": "118560263",
      "token": "118560263"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "118560263",
      "raw_match_text": "118560263",
      "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
      "supporting_property_id": "P7902",
      "supporting_value": "118560263",
      "token": "118560263"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_NON_TARGET_PROPERTY_TEXT"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "118560263",
      "189575476"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "118560263"
  ],
  "truth_tokens_in_recorded_matches": [
    "118560263"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier from the Gemeinsame Normdatei authority file of names, subjects, and organizations",
    "label": "GND ID"
  },
  "qid": {
    "description": "ancient Greek philosopher",
    "label": "Carneades"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
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
    "label_en": "label in language constraint",
    "qid": "Q108139345"
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
        "118560263": 1
      },
      "new_unique": [
        "118560263"
      ],
      "new_values": [
        "118560263"
      ],
      "new_values_raw": [
        "118560263"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "118560263": 1,
        "189575476": 1
      },
      "old_unique": [
        "118560263",
        "189575476"
      ],
      "old_values": [
        "118560263",
        "189575476"
      ],
      "old_values_raw": [
        "118560263",
        "189575476"
      ],
      "removed_unique_values": [
        "189575476"
      ],
      "retained_unique_values": [
        "118560263"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "189575476": {
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
      "local_ids_count": 55,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "118560263",
          "raw_match_text": "118560263",
          "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
          "supporting_property_id": "P7902",
          "supporting_value": "118560263",
          "token": "118560263"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_NON_TARGET_PROPERTY_TEXT"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "118560263",
        "189575476"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 017. `repair_Q2917788_2444349622`

| Field | Value |
|---|---|
| qid | Q2917788 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q2917788::P225 |
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
| truth_tokens_preview | ["Osteobrama belangeri"] |
| classification_target_tokens | ["Osteobrama brevipectoralis"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Osteobrama brevipectoralis"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Osteobrama brevipectoralis"
  ],
  "removed_unique_values": [
    "Osteobrama brevipectoralis"
  ],
  "retained_support_tokens": [
    "Osteobrama belangeri"
  ],
  "retained_unique_values": [
    "Osteobrama belangeri"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Osteobrama belangeri"
  ],
  "old_value": [
    "Osteobrama belangeri",
    "Osteobrama brevipectoralis"
  ],
  "revision_id": 2444349622,
  "value": [
    "Osteobrama belangeri"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Osteobrama belangeri": 1
    },
    "new_unique": [
      "Osteobrama belangeri"
    ],
    "new_values": [
      "Osteobrama belangeri"
    ],
    "new_values_raw": [
      "Osteobrama belangeri"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Osteobrama belangeri": 1,
      "Osteobrama brevipectoralis": 1
    },
    "old_unique": [
      "Osteobrama belangeri",
      "Osteobrama brevipectoralis"
    ],
    "old_values": [
      "Osteobrama belangeri",
      "Osteobrama brevipectoralis"
    ],
    "old_values_raw": [
      "Osteobrama belangeri",
      "Osteobrama brevipectoralis"
    ],
    "removed_unique_values": [
      "Osteobrama brevipectoralis"
    ],
    "retained_unique_values": [
      "Osteobrama belangeri"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Osteobrama brevipectoralis": {
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
  "report_fix_date": "2025-12-21T11:55:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2444910674,
  "report_revision_old": 2444477567,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Osteobrama belangeri",
    "Osteobrama brevipectoralis"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 9,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "osteobrama belangeri",
      "raw_match_text": "Osteobrama belangeri",
      "source": "FOCUS_LABEL",
      "token": "Osteobrama belangeri"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "osteobrama belangeri",
      "raw_match_text": "Osteobrama belangeri",
      "source": "FOCUS_LABEL",
      "token": "Osteobrama belangeri"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Osteobrama belangeri",
      "Osteobrama brevipectoralis"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Osteobrama belangeri"
  ],
  "truth_tokens_in_recorded_matches": [
    "Osteobrama belangeri"
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
    "description": "species of fish",
    "label": "Osteobrama belangeri"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Osteobrama belangeri": 1
      },
      "new_unique": [
        "Osteobrama belangeri"
      ],
      "new_values": [
        "Osteobrama belangeri"
      ],
      "new_values_raw": [
        "Osteobrama belangeri"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Osteobrama belangeri": 1,
        "Osteobrama brevipectoralis": 1
      },
      "old_unique": [
        "Osteobrama belangeri",
        "Osteobrama brevipectoralis"
      ],
      "old_values": [
        "Osteobrama belangeri",
        "Osteobrama brevipectoralis"
      ],
      "old_values_raw": [
        "Osteobrama belangeri",
        "Osteobrama brevipectoralis"
      ],
      "removed_unique_values": [
        "Osteobrama brevipectoralis"
      ],
      "retained_unique_values": [
        "Osteobrama belangeri"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Osteobrama brevipectoralis": {
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
      "local_ids_count": 9,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "osteobrama belangeri",
          "raw_match_text": "Osteobrama belangeri",
          "source": "FOCUS_LABEL",
          "token": "Osteobrama belangeri"
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
        "Osteobrama belangeri",
        "Osteobrama brevipectoralis"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 018. `repair_Q321749_2444524922`

| Field | Value |
|---|---|
| qid | Q321749 |
| property | P227 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q321749::P227 |
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
| truth_tokens_preview | ["6014425-7"] |
| classification_target_tokens | ["1254038671"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_NON_TARGET_PROPERTY_TEXT |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "1254038671"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "1254038671"
  ],
  "removed_unique_values": [
    "1254038671"
  ],
  "retained_support_tokens": [
    "6014425-7"
  ],
  "retained_unique_values": [
    "6014425-7"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
  "author": "Kolja21",
  "kind": "A_BOX",
  "new_value": [
    "6014425-7"
  ],
  "old_value": [
    "6014425-7",
    "1254038671"
  ],
  "revision_id": 2444524922,
  "value": [
    "6014425-7"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "6014425-7": 1
    },
    "new_unique": [
      "6014425-7"
    ],
    "new_values": [
      "6014425-7"
    ],
    "new_values_raw": [
      "6014425-7"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1254038671": 1,
      "6014425-7": 1
    },
    "old_unique": [
      "1254038671",
      "6014425-7"
    ],
    "old_values": [
      "6014425-7",
      "1254038671"
    ],
    "old_values_raw": [
      "6014425-7",
      "1254038671"
    ],
    "removed_unique_values": [
      "1254038671"
    ],
    "retained_unique_values": [
      "6014425-7"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "1254038671": {
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
  "report_fix_date": "2025-12-22T11:21:46",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P227",
  "report_revision_new": 2445476724,
  "report_revision_old": 2444910132,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "6014425-7",
    "1254038671"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 17,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "6014425-7",
      "raw_match_text": "6014425-7",
      "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
      "supporting_property_id": "P9964",
      "supporting_value": "6014425-7",
      "token": "6014425-7"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "6014425-7",
      "raw_match_text": "6014425-7",
      "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
      "supporting_property_id": "P9964",
      "supporting_value": "6014425-7",
      "token": "6014425-7"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_NON_TARGET_PROPERTY_TEXT"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "6014425-7",
      "1254038671"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "6014425-7"
  ],
  "truth_tokens_in_recorded_matches": [
    "6014425-7"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier from the Gemeinsame Normdatei authority file of names, subjects, and organizations",
    "label": "GND ID"
  },
  "qid": {
    "description": "Reform Jewish congregation",
    "label": "Hamburg Temple"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
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
    "label_en": "label in language constraint",
    "qid": "Q108139345"
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
        "6014425-7": 1
      },
      "new_unique": [
        "6014425-7"
      ],
      "new_values": [
        "6014425-7"
      ],
      "new_values_raw": [
        "6014425-7"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1254038671": 1,
        "6014425-7": 1
      },
      "old_unique": [
        "1254038671",
        "6014425-7"
      ],
      "old_values": [
        "6014425-7",
        "1254038671"
      ],
      "old_values_raw": [
        "6014425-7",
        "1254038671"
      ],
      "removed_unique_values": [
        "1254038671"
      ],
      "retained_unique_values": [
        "6014425-7"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "1254038671": {
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
      "local_ids_count": 17,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact_raw",
          "normalized_match_text": "6014425-7",
          "raw_match_text": "6014425-7",
          "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
          "supporting_property_id": "P9964",
          "supporting_value": "6014425-7",
          "token": "6014425-7"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_NON_TARGET_PROPERTY_TEXT"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "6014425-7",
        "1254038671"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 019. `repair_Q5130878_2446021932`

| Field | Value |
|---|---|
| qid | Q5130878 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q5130878::P373 |
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
| truth_tokens_preview | ["Clearing the cervical spine"] |
| classification_target_tokens | ["Canadian C-Spine Rule"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Canadian C-Spine Rule"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Canadian C-Spine Rule"
  ],
  "removed_unique_values": [
    "Canadian C-Spine Rule"
  ],
  "retained_support_tokens": [
    "Clearing the cervical spine"
  ],
  "retained_unique_values": [
    "Clearing the cervical spine"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
  "author": "Louperibot",
  "kind": "A_BOX",
  "new_value": [
    "Clearing the cervical spine"
  ],
  "old_value": [
    "Clearing the cervical spine",
    "Canadian C-Spine Rule"
  ],
  "revision_id": 2446021932,
  "value": [
    "Clearing the cervical spine"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Clearing the cervical spine": 1
    },
    "new_unique": [
      "Clearing the cervical spine"
    ],
    "new_values": [
      "Clearing the cervical spine"
    ],
    "new_values_raw": [
      "Clearing the cervical spine"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Canadian C-Spine Rule": 1,
      "Clearing the cervical spine": 1
    },
    "old_unique": [
      "Canadian C-Spine Rule",
      "Clearing the cervical spine"
    ],
    "old_values": [
      "Clearing the cervical spine",
      "Canadian C-Spine Rule"
    ],
    "old_values_raw": [
      "Clearing the cervical spine",
      "Canadian C-Spine Rule"
    ],
    "removed_unique_values": [
      "Canadian C-Spine Rule"
    ],
    "retained_unique_values": [
      "Clearing the cervical spine"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Canadian C-Spine Rule": {
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
  "report_fix_date": "2025-12-25T19:21:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447067079,
  "report_revision_old": 2446526020,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Clearing the cervical spine",
    "Canadian C-Spine Rule"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 5,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "clearing the cervical spine",
      "raw_match_text": "Clearing the cervical spine",
      "source": "FOCUS_LABEL",
      "token": "Clearing the cervical spine"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "clearing the cervical spine",
      "raw_match_text": "Clearing the cervical spine",
      "source": "FOCUS_LABEL",
      "token": "Clearing the cervical spine"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Clearing the cervical spine",
      "Canadian C-Spine Rule"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Clearing the cervical spine"
  ],
  "truth_tokens_in_recorded_matches": [
    "Clearing the cervical spine"
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
    "description": "process of determining the existence of a cervical spine injury",
    "label": "Clearing the cervical spine"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Clearing the cervical spine": 1
      },
      "new_unique": [
        "Clearing the cervical spine"
      ],
      "new_values": [
        "Clearing the cervical spine"
      ],
      "new_values_raw": [
        "Clearing the cervical spine"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Canadian C-Spine Rule": 1,
        "Clearing the cervical spine": 1
      },
      "old_unique": [
        "Canadian C-Spine Rule",
        "Clearing the cervical spine"
      ],
      "old_values": [
        "Clearing the cervical spine",
        "Canadian C-Spine Rule"
      ],
      "old_values_raw": [
        "Clearing the cervical spine",
        "Canadian C-Spine Rule"
      ],
      "removed_unique_values": [
        "Canadian C-Spine Rule"
      ],
      "retained_unique_values": [
        "Clearing the cervical spine"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Canadian C-Spine Rule": {
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
          "normalized_match_text": "clearing the cervical spine",
          "raw_match_text": "Clearing the cervical spine",
          "source": "FOCUS_LABEL",
          "token": "Clearing the cervical spine"
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
        "Clearing the cervical spine",
        "Canadian C-Spine Rule"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 020. `repair_Q575874_2445923132`

| Field | Value |
|---|---|
| qid | Q575874 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q575874::P225 |
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
| truth_tokens_preview | ["Didymiaceae"] |
| classification_target_tokens | ["Didymiidae"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Didymiidae"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Didymiidae"
  ],
  "removed_unique_values": [
    "Didymiidae"
  ],
  "retained_support_tokens": [
    "Didymiaceae"
  ],
  "retained_unique_values": [
    "Didymiaceae"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
  "author": "Xavidvila",
  "kind": "A_BOX",
  "new_value": [
    "Didymiaceae"
  ],
  "old_value": [
    "Didymiaceae",
    "Didymiidae"
  ],
  "revision_id": 2445923132,
  "value": [
    "Didymiaceae"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Didymiaceae": 1
    },
    "new_unique": [
      "Didymiaceae"
    ],
    "new_values": [
      "Didymiaceae"
    ],
    "new_values_raw": [
      "Didymiaceae"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Didymiaceae": 1,
      "Didymiidae": 1
    },
    "old_unique": [
      "Didymiaceae",
      "Didymiidae"
    ],
    "old_values": [
      "Didymiaceae",
      "Didymiidae"
    ],
    "old_values_raw": [
      "Didymiaceae",
      "Didymiidae"
    ],
    "removed_unique_values": [
      "Didymiidae"
    ],
    "retained_unique_values": [
      "Didymiaceae"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Didymiidae": {
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
  "report_fix_date": "2025-12-24T13:09:37",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2446547742,
  "report_revision_old": 2446076213,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Didymiaceae",
    "Didymiidae"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 11,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "didymiaceae",
      "raw_match_text": "Didymiaceae",
      "source": "FOCUS_LABEL",
      "token": "Didymiaceae"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "didymiaceae",
      "raw_match_text": "Didymiaceae",
      "source": "FOCUS_LABEL",
      "token": "Didymiaceae"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Didymiaceae",
      "Didymiidae"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Didymiaceae"
  ],
  "truth_tokens_in_recorded_matches": [
    "Didymiaceae"
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
    "description": "family of slime moulds",
    "label": "Didymiaceae"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Didymiaceae": 1
      },
      "new_unique": [
        "Didymiaceae"
      ],
      "new_values": [
        "Didymiaceae"
      ],
      "new_values_raw": [
        "Didymiaceae"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Didymiaceae": 1,
        "Didymiidae": 1
      },
      "old_unique": [
        "Didymiaceae",
        "Didymiidae"
      ],
      "old_values": [
        "Didymiaceae",
        "Didymiidae"
      ],
      "old_values_raw": [
        "Didymiaceae",
        "Didymiidae"
      ],
      "removed_unique_values": [
        "Didymiidae"
      ],
      "retained_unique_values": [
        "Didymiaceae"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Didymiidae": {
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
          "normalized_match_text": "didymiaceae",
          "raw_match_text": "Didymiaceae",
          "source": "FOCUS_LABEL",
          "token": "Didymiaceae"
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
        "Didymiaceae",
        "Didymiidae"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 021. `repair_Q63428288_2444875683`

| Field | Value |
|---|---|
| qid | Q63428288 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q63428288::P225 |
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
| truth_tokens_preview | ["Caecibacterium"] |
| classification_target_tokens | ["Caecibacter"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Caecibacter"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Caecibacter"
  ],
  "removed_unique_values": [
    "Caecibacter"
  ],
  "retained_support_tokens": [
    "Caecibacterium"
  ],
  "retained_unique_values": [
    "Caecibacterium"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Caecibacterium"
  ],
  "old_value": [
    "Caecibacterium",
    "Caecibacter"
  ],
  "revision_id": 2444875683,
  "value": [
    "Caecibacterium"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Caecibacterium": 1
    },
    "new_unique": [
      "Caecibacterium"
    ],
    "new_values": [
      "Caecibacterium"
    ],
    "new_values_raw": [
      "Caecibacterium"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Caecibacter": 1,
      "Caecibacterium": 1
    },
    "old_unique": [
      "Caecibacter",
      "Caecibacterium"
    ],
    "old_values": [
      "Caecibacterium",
      "Caecibacter"
    ],
    "old_values_raw": [
      "Caecibacterium",
      "Caecibacter"
    ],
    "removed_unique_values": [
      "Caecibacter"
    ],
    "retained_unique_values": [
      "Caecibacterium"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Caecibacter": {
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
  "report_fix_date": "2025-12-22T11:22:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2445477126,
  "report_revision_old": 2444910674,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Caecibacterium",
    "Caecibacter"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "caecibacterium",
      "raw_match_text": "Caecibacterium",
      "source": "FOCUS_LABEL",
      "token": "Caecibacterium"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "caecibacterium",
      "raw_match_text": "Caecibacterium",
      "source": "FOCUS_LABEL",
      "token": "Caecibacterium"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Caecibacterium",
      "Caecibacter"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Caecibacterium"
  ],
  "truth_tokens_in_recorded_matches": [
    "Caecibacterium"
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
    "description": "genus of bacteria",
    "label": "Caecibacterium"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Caecibacterium": 1
      },
      "new_unique": [
        "Caecibacterium"
      ],
      "new_values": [
        "Caecibacterium"
      ],
      "new_values_raw": [
        "Caecibacterium"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Caecibacter": 1,
        "Caecibacterium": 1
      },
      "old_unique": [
        "Caecibacter",
        "Caecibacterium"
      ],
      "old_values": [
        "Caecibacterium",
        "Caecibacter"
      ],
      "old_values_raw": [
        "Caecibacterium",
        "Caecibacter"
      ],
      "removed_unique_values": [
        "Caecibacter"
      ],
      "retained_unique_values": [
        "Caecibacterium"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Caecibacter": {
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
          "normalized_match_text": "caecibacterium",
          "raw_match_text": "Caecibacterium",
          "source": "FOCUS_LABEL",
          "token": "Caecibacterium"
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
        "Caecibacterium",
        "Caecibacter"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 022. `repair_Q6515511_2444900157`

| Field | Value |
|---|---|
| qid | Q6515511 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q6515511::P225 |
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
| truth_tokens_preview | ["Lithobius porathi"] |
| classification_target_tokens | ["Lithobius dziadoszi"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Lithobius dziadoszi"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Lithobius dziadoszi"
  ],
  "removed_unique_values": [
    "Lithobius dziadoszi"
  ],
  "retained_support_tokens": [
    "Lithobius porathi"
  ],
  "retained_unique_values": [
    "Lithobius porathi"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Lithobius porathi"
  ],
  "old_value": [
    "Lithobius porathi",
    "Lithobius dziadoszi"
  ],
  "revision_id": 2444900157,
  "value": [
    "Lithobius porathi"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Lithobius porathi": 1
    },
    "new_unique": [
      "Lithobius porathi"
    ],
    "new_values": [
      "Lithobius porathi"
    ],
    "new_values_raw": [
      "Lithobius porathi"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Lithobius dziadoszi": 1,
      "Lithobius porathi": 1
    },
    "old_unique": [
      "Lithobius dziadoszi",
      "Lithobius porathi"
    ],
    "old_values": [
      "Lithobius porathi",
      "Lithobius dziadoszi"
    ],
    "old_values_raw": [
      "Lithobius porathi",
      "Lithobius dziadoszi"
    ],
    "removed_unique_values": [
      "Lithobius dziadoszi"
    ],
    "retained_unique_values": [
      "Lithobius porathi"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Lithobius dziadoszi": {
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
  "report_fix_date": "2025-12-22T11:22:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2445477126,
  "report_revision_old": 2444910674,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Lithobius porathi",
    "Lithobius dziadoszi"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "lithobius porathi",
      "raw_match_text": "Lithobius porathi",
      "source": "FOCUS_LABEL",
      "token": "Lithobius porathi"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "lithobius porathi",
      "raw_match_text": "Lithobius porathi",
      "source": "FOCUS_LABEL",
      "token": "Lithobius porathi"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Lithobius porathi",
      "Lithobius dziadoszi"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Lithobius porathi"
  ],
  "truth_tokens_in_recorded_matches": [
    "Lithobius porathi"
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
    "description": "species of arthropod",
    "label": "Lithobius porathi"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Lithobius porathi": 1
      },
      "new_unique": [
        "Lithobius porathi"
      ],
      "new_values": [
        "Lithobius porathi"
      ],
      "new_values_raw": [
        "Lithobius porathi"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Lithobius dziadoszi": 1,
        "Lithobius porathi": 1
      },
      "old_unique": [
        "Lithobius dziadoszi",
        "Lithobius porathi"
      ],
      "old_values": [
        "Lithobius porathi",
        "Lithobius dziadoszi"
      ],
      "old_values_raw": [
        "Lithobius porathi",
        "Lithobius dziadoszi"
      ],
      "removed_unique_values": [
        "Lithobius dziadoszi"
      ],
      "retained_unique_values": [
        "Lithobius porathi"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Lithobius dziadoszi": {
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
          "normalized_match_text": "lithobius porathi",
          "raw_match_text": "Lithobius porathi",
          "source": "FOCUS_LABEL",
          "token": "Lithobius porathi"
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
        "Lithobius porathi",
        "Lithobius dziadoszi"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 023. `repair_Q6668972_2439412579`

| Field | Value |
|---|---|
| qid | Q6668972 |
| property | P227 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q6668972::P227 |
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
| truth_token_kind | numeric |
| truth_tokens_preview | ["102397392"] |
| classification_target_tokens | ["1055449728"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_NON_TARGET_PROPERTY_TEXT |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "1055449728"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "1055449728"
  ],
  "removed_unique_values": [
    "1055449728"
  ],
  "retained_support_tokens": [
    "102397392"
  ],
  "retained_unique_values": [
    "102397392"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
  "author": "Kolja21",
  "kind": "A_BOX",
  "new_value": [
    "102397392"
  ],
  "old_value": [
    "102397392",
    "1055449728"
  ],
  "revision_id": 2439412579,
  "value": [
    "102397392"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "102397392": 1
    },
    "new_unique": [
      "102397392"
    ],
    "new_values": [
      "102397392"
    ],
    "new_values_raw": [
      "102397392"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "102397392": 1,
      "1055449728": 1
    },
    "old_unique": [
      "102397392",
      "1055449728"
    ],
    "old_values": [
      "102397392",
      "1055449728"
    ],
    "old_values_raw": [
      "102397392",
      "1055449728"
    ],
    "removed_unique_values": [
      "1055449728"
    ],
    "retained_unique_values": [
      "102397392"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "1055449728": {
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
  "report_fix_date": "2025-12-09T12:32:17",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P227",
  "report_revision_new": 2440013759,
  "report_revision_old": 2439564475,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "102397392",
    "1055449728"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 21,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "102397392",
      "raw_match_text": "102397392",
      "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
      "supporting_property_id": "P7902",
      "supporting_value": "102397392",
      "token": "102397392"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "102397392",
      "raw_match_text": "102397392",
      "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
      "supporting_property_id": "P7902",
      "supporting_value": "102397392",
      "token": "102397392"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_NON_TARGET_PROPERTY_TEXT"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "102397392",
      "1055449728"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "102397392"
  ],
  "truth_tokens_in_recorded_matches": [
    "102397392"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier from the Gemeinsame Normdatei authority file of names, subjects, and organizations",
    "label": "GND ID"
  },
  "qid": {
    "description": "1st century AD Greek epigrammatist",
    "label": "Lollius Bassus"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
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
    "label_en": "label in language constraint",
    "qid": "Q108139345"
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
        "102397392": 1
      },
      "new_unique": [
        "102397392"
      ],
      "new_values": [
        "102397392"
      ],
      "new_values_raw": [
        "102397392"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "102397392": 1,
        "1055449728": 1
      },
      "old_unique": [
        "102397392",
        "1055449728"
      ],
      "old_values": [
        "102397392",
        "1055449728"
      ],
      "old_values_raw": [
        "102397392",
        "1055449728"
      ],
      "removed_unique_values": [
        "1055449728"
      ],
      "retained_unique_values": [
        "102397392"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "1055449728": {
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
          "normalized_match_text": "102397392",
          "raw_match_text": "102397392",
          "source": "FOCUS_NON_TARGET_PROPERTY_TEXT",
          "supporting_property_id": "P7902",
          "supporting_value": "102397392",
          "token": "102397392"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_NON_TARGET_PROPERTY_TEXT"
      ],
      "used_literal_substring": false
    },
    "result": true,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "102397392",
        "1055449728"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 024. `repair_Q7095178_2444955974`

| Field | Value |
|---|---|
| qid | Q7095178 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q7095178::P225 |
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
| truth_tokens_preview | ["Onythes"] |
| classification_target_tokens | ["Onthyes"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Onthyes"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Onthyes"
  ],
  "removed_unique_values": [
    "Onthyes"
  ],
  "retained_support_tokens": [
    "Onythes"
  ],
  "retained_unique_values": [
    "Onythes"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Onythes"
  ],
  "old_value": [
    "Onythes",
    "Onthyes"
  ],
  "revision_id": 2444955974,
  "value": [
    "Onythes"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Onythes": 1
    },
    "new_unique": [
      "Onythes"
    ],
    "new_values": [
      "Onythes"
    ],
    "new_values_raw": [
      "Onythes"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Onthyes": 1,
      "Onythes": 1
    },
    "old_unique": [
      "Onthyes",
      "Onythes"
    ],
    "old_values": [
      "Onythes",
      "Onthyes"
    ],
    "old_values_raw": [
      "Onythes",
      "Onthyes"
    ],
    "removed_unique_values": [
      "Onthyes"
    ],
    "retained_unique_values": [
      "Onythes"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Onthyes": {
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
  "report_fix_date": "2025-12-23T14:31:41",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2446076213,
  "report_revision_old": 2445477126,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Onythes",
    "Onthyes"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 13,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "onythes",
      "raw_match_text": "Onythes",
      "source": "FOCUS_LABEL",
      "token": "Onythes"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "onythes",
      "raw_match_text": "Onythes",
      "source": "FOCUS_LABEL",
      "token": "Onythes"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Onythes",
      "Onthyes"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Onythes"
  ],
  "truth_tokens_in_recorded_matches": [
    "Onythes"
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
    "description": "genus of insects",
    "label": "Onythes"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Onythes": 1
      },
      "new_unique": [
        "Onythes"
      ],
      "new_values": [
        "Onythes"
      ],
      "new_values_raw": [
        "Onythes"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Onthyes": 1,
        "Onythes": 1
      },
      "old_unique": [
        "Onthyes",
        "Onythes"
      ],
      "old_values": [
        "Onythes",
        "Onthyes"
      ],
      "old_values_raw": [
        "Onythes",
        "Onthyes"
      ],
      "removed_unique_values": [
        "Onthyes"
      ],
      "retained_unique_values": [
        "Onythes"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Onthyes": {
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
          "normalized_match_text": "onythes",
          "raw_match_text": "Onythes",
          "source": "FOCUS_LABEL",
          "token": "Onythes"
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
        "Onythes",
        "Onthyes"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 025. `repair_Q7201945_2444408422`

| Field | Value |
|---|---|
| qid | Q7201945 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q7201945::P225 |
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
| truth_tokens_preview | ["Hepatocystis pteropi"] |
| classification_target_tokens | ["Hepatocystis pteropti"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Hepatocystis pteropti"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Hepatocystis pteropti"
  ],
  "removed_unique_values": [
    "Hepatocystis pteropti"
  ],
  "retained_support_tokens": [
    "Hepatocystis pteropi"
  ],
  "retained_unique_values": [
    "Hepatocystis pteropi"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Hepatocystis pteropi"
  ],
  "old_value": [
    "Hepatocystis pteropti",
    "Hepatocystis pteropi"
  ],
  "revision_id": 2444408422,
  "value": [
    "Hepatocystis pteropi"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Hepatocystis pteropi": 1
    },
    "new_unique": [
      "Hepatocystis pteropi"
    ],
    "new_values": [
      "Hepatocystis pteropi"
    ],
    "new_values_raw": [
      "Hepatocystis pteropi"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Hepatocystis pteropi": 1,
      "Hepatocystis pteropti": 1
    },
    "old_unique": [
      "Hepatocystis pteropi",
      "Hepatocystis pteropti"
    ],
    "old_values": [
      "Hepatocystis pteropti",
      "Hepatocystis pteropi"
    ],
    "old_values_raw": [
      "Hepatocystis pteropti",
      "Hepatocystis pteropi"
    ],
    "removed_unique_values": [
      "Hepatocystis pteropti"
    ],
    "retained_unique_values": [
      "Hepatocystis pteropi"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Hepatocystis pteropti": {
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
  "report_fix_date": "2025-12-21T11:55:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2444910674,
  "report_revision_old": 2444477567,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Hepatocystis pteropti",
    "Hepatocystis pteropi"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "hepatocystis pteropi",
      "raw_match_text": "Hepatocystis pteropi",
      "source": "FOCUS_LABEL",
      "token": "Hepatocystis pteropi"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "hepatocystis pteropi",
      "raw_match_text": "Hepatocystis pteropi",
      "source": "FOCUS_LABEL",
      "token": "Hepatocystis pteropi"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Hepatocystis pteropti",
      "Hepatocystis pteropi"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Hepatocystis pteropi"
  ],
  "truth_tokens_in_recorded_matches": [
    "Hepatocystis pteropi"
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
    "description": "taxon, (Breinl, 1913) species of plasmodiidae",
    "label": "Hepatocystis pteropi"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Hepatocystis pteropi": 1
      },
      "new_unique": [
        "Hepatocystis pteropi"
      ],
      "new_values": [
        "Hepatocystis pteropi"
      ],
      "new_values_raw": [
        "Hepatocystis pteropi"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Hepatocystis pteropi": 1,
        "Hepatocystis pteropti": 1
      },
      "old_unique": [
        "Hepatocystis pteropi",
        "Hepatocystis pteropti"
      ],
      "old_values": [
        "Hepatocystis pteropti",
        "Hepatocystis pteropi"
      ],
      "old_values_raw": [
        "Hepatocystis pteropti",
        "Hepatocystis pteropi"
      ],
      "removed_unique_values": [
        "Hepatocystis pteropti"
      ],
      "retained_unique_values": [
        "Hepatocystis pteropi"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Hepatocystis pteropti": {
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
          "normalized_match_text": "hepatocystis pteropi",
          "raw_match_text": "Hepatocystis pteropi",
          "source": "FOCUS_LABEL",
          "token": "Hepatocystis pteropi"
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
        "Hepatocystis pteropti",
        "Hepatocystis pteropi"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 026. `repair_Q7305306_2442871123`

| Field | Value |
|---|---|
| qid | Q7305306 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q7305306::P225 |
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
| truth_tokens_preview | ["Pseudophycis bachus"] |
| classification_target_tokens | ["Physiculus bachus"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Physiculus bachus"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Physiculus bachus"
  ],
  "removed_unique_values": [
    "Physiculus bachus"
  ],
  "retained_support_tokens": [
    "Pseudophycis bachus"
  ],
  "retained_unique_values": [
    "Pseudophycis bachus"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Pseudophycis bachus"
  ],
  "old_value": [
    "Pseudophycis bachus",
    "Physiculus bachus"
  ],
  "revision_id": 2442871123,
  "value": [
    "Pseudophycis bachus"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Pseudophycis bachus": 1
    },
    "new_unique": [
      "Pseudophycis bachus"
    ],
    "new_values": [
      "Pseudophycis bachus"
    ],
    "new_values_raw": [
      "Pseudophycis bachus"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Physiculus bachus": 1,
      "Pseudophycis bachus": 1
    },
    "old_unique": [
      "Physiculus bachus",
      "Pseudophycis bachus"
    ],
    "old_values": [
      "Pseudophycis bachus",
      "Physiculus bachus"
    ],
    "old_values_raw": [
      "Pseudophycis bachus",
      "Physiculus bachus"
    ],
    "removed_unique_values": [
      "Physiculus bachus"
    ],
    "retained_unique_values": [
      "Pseudophycis bachus"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Physiculus bachus": {
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
  "report_fix_date": "2025-12-17T13:40:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2443413974,
  "report_revision_old": 2442991823,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Pseudophycis bachus",
    "Physiculus bachus"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 13,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "pseudophycis bachus",
      "raw_match_text": "Pseudophycis bachus",
      "source": "FOCUS_LABEL",
      "token": "Pseudophycis bachus"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "pseudophycis bachus",
      "raw_match_text": "Pseudophycis bachus",
      "source": "FOCUS_LABEL",
      "token": "Pseudophycis bachus"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Pseudophycis bachus",
      "Physiculus bachus"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Pseudophycis bachus"
  ],
  "truth_tokens_in_recorded_matches": [
    "Pseudophycis bachus"
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
    "description": "species of fish",
    "label": "Pseudophycis bachus"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Pseudophycis bachus": 1
      },
      "new_unique": [
        "Pseudophycis bachus"
      ],
      "new_values": [
        "Pseudophycis bachus"
      ],
      "new_values_raw": [
        "Pseudophycis bachus"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Physiculus bachus": 1,
        "Pseudophycis bachus": 1
      },
      "old_unique": [
        "Physiculus bachus",
        "Pseudophycis bachus"
      ],
      "old_values": [
        "Pseudophycis bachus",
        "Physiculus bachus"
      ],
      "old_values_raw": [
        "Pseudophycis bachus",
        "Physiculus bachus"
      ],
      "removed_unique_values": [
        "Physiculus bachus"
      ],
      "retained_unique_values": [
        "Pseudophycis bachus"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Physiculus bachus": {
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
          "normalized_match_text": "pseudophycis bachus",
          "raw_match_text": "Pseudophycis bachus",
          "source": "FOCUS_LABEL",
          "token": "Pseudophycis bachus"
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
        "Pseudophycis bachus",
        "Physiculus bachus"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 027. `repair_Q74577630_2444407214`

| Field | Value |
|---|---|
| qid | Q74577630 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q74577630::P225 |
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
| truth_tokens_preview | ["Isoëtes laurentiana"] |
| classification_target_tokens | ["Isoetes laurentiana"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Isoetes laurentiana"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Isoetes laurentiana"
  ],
  "removed_unique_values": [
    "Isoetes laurentiana"
  ],
  "retained_support_tokens": [
    "Isoëtes laurentiana"
  ],
  "retained_unique_values": [
    "Isoëtes laurentiana"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Isoëtes laurentiana"
  ],
  "old_value": [
    "Isoëtes laurentiana",
    "Isoetes laurentiana"
  ],
  "revision_id": 2444407214,
  "value": [
    "Isoëtes laurentiana"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Isoëtes laurentiana": 1
    },
    "new_unique": [
      "Isoëtes laurentiana"
    ],
    "new_values": [
      "Isoëtes laurentiana"
    ],
    "new_values_raw": [
      "Isoëtes laurentiana"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Isoetes laurentiana": 1,
      "Isoëtes laurentiana": 1
    },
    "old_unique": [
      "Isoetes laurentiana",
      "Isoëtes laurentiana"
    ],
    "old_values": [
      "Isoëtes laurentiana",
      "Isoetes laurentiana"
    ],
    "old_values_raw": [
      "Isoëtes laurentiana",
      "Isoetes laurentiana"
    ],
    "removed_unique_values": [
      "Isoetes laurentiana"
    ],
    "retained_unique_values": [
      "Isoëtes laurentiana"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Isoetes laurentiana": {
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
  "report_fix_date": "2025-12-21T11:55:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2444910674,
  "report_revision_old": 2444477567,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Isoëtes laurentiana",
    "Isoetes laurentiana"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "isoëtes laurentiana",
      "raw_match_text": "Isoëtes laurentiana",
      "source": "FOCUS_LABEL",
      "token": "Isoëtes laurentiana"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "isoëtes laurentiana",
      "raw_match_text": "Isoëtes laurentiana",
      "source": "FOCUS_LABEL",
      "token": "Isoëtes laurentiana"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Isoëtes laurentiana",
      "Isoetes laurentiana"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Isoëtes laurentiana"
  ],
  "truth_tokens_in_recorded_matches": [
    "Isoëtes laurentiana"
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
    "description": "species of plant",
    "label": "Isoëtes laurentiana"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Isoëtes laurentiana": 1
      },
      "new_unique": [
        "Isoëtes laurentiana"
      ],
      "new_values": [
        "Isoëtes laurentiana"
      ],
      "new_values_raw": [
        "Isoëtes laurentiana"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Isoetes laurentiana": 1,
        "Isoëtes laurentiana": 1
      },
      "old_unique": [
        "Isoetes laurentiana",
        "Isoëtes laurentiana"
      ],
      "old_values": [
        "Isoëtes laurentiana",
        "Isoetes laurentiana"
      ],
      "old_values_raw": [
        "Isoëtes laurentiana",
        "Isoetes laurentiana"
      ],
      "removed_unique_values": [
        "Isoetes laurentiana"
      ],
      "retained_unique_values": [
        "Isoëtes laurentiana"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Isoetes laurentiana": {
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
          "normalized_match_text": "isoëtes laurentiana",
          "raw_match_text": "Isoëtes laurentiana",
          "source": "FOCUS_LABEL",
          "token": "Isoëtes laurentiana"
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
        "Isoëtes laurentiana",
        "Isoetes laurentiana"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 028. `repair_Q7734948_2442882905`

| Field | Value |
|---|---|
| qid | Q7734948 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q7734948::P225 |
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
| truth_tokens_preview | ["Protaetia orientalis submarmorea"] |
| classification_target_tokens | ["Protaetia orientalis submarumorea"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Protaetia orientalis submarumorea"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Protaetia orientalis submarumorea"
  ],
  "removed_unique_values": [
    "Protaetia orientalis submarumorea"
  ],
  "retained_support_tokens": [
    "Protaetia orientalis submarmorea"
  ],
  "retained_unique_values": [
    "Protaetia orientalis submarmorea"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Protaetia orientalis submarmorea"
  ],
  "old_value": [
    "Protaetia orientalis submarumorea",
    "Protaetia orientalis submarmorea"
  ],
  "revision_id": 2442882905,
  "value": [
    "Protaetia orientalis submarmorea"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Protaetia orientalis submarmorea": 1
    },
    "new_unique": [
      "Protaetia orientalis submarmorea"
    ],
    "new_values": [
      "Protaetia orientalis submarmorea"
    ],
    "new_values_raw": [
      "Protaetia orientalis submarmorea"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Protaetia orientalis submarmorea": 1,
      "Protaetia orientalis submarumorea": 1
    },
    "old_unique": [
      "Protaetia orientalis submarmorea",
      "Protaetia orientalis submarumorea"
    ],
    "old_values": [
      "Protaetia orientalis submarumorea",
      "Protaetia orientalis submarmorea"
    ],
    "old_values_raw": [
      "Protaetia orientalis submarumorea",
      "Protaetia orientalis submarmorea"
    ],
    "removed_unique_values": [
      "Protaetia orientalis submarumorea"
    ],
    "retained_unique_values": [
      "Protaetia orientalis submarmorea"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Protaetia orientalis submarumorea": {
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
  "report_fix_date": "2025-12-17T13:40:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2443413974,
  "report_revision_old": 2442991823,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Protaetia orientalis submarumorea",
    "Protaetia orientalis submarmorea"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "protaetia orientalis submarmorea",
      "raw_match_text": "Protaetia orientalis submarmorea",
      "source": "FOCUS_LABEL",
      "token": "Protaetia orientalis submarmorea"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "protaetia orientalis submarmorea",
      "raw_match_text": "Protaetia orientalis submarmorea",
      "source": "FOCUS_LABEL",
      "token": "Protaetia orientalis submarmorea"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Protaetia orientalis submarumorea",
      "Protaetia orientalis submarmorea"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Protaetia orientalis submarmorea"
  ],
  "truth_tokens_in_recorded_matches": [
    "Protaetia orientalis submarmorea"
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
    "description": "subspecies of insect",
    "label": "Protaetia orientalis submarmorea"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Protaetia orientalis submarmorea": 1
      },
      "new_unique": [
        "Protaetia orientalis submarmorea"
      ],
      "new_values": [
        "Protaetia orientalis submarmorea"
      ],
      "new_values_raw": [
        "Protaetia orientalis submarmorea"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Protaetia orientalis submarmorea": 1,
        "Protaetia orientalis submarumorea": 1
      },
      "old_unique": [
        "Protaetia orientalis submarmorea",
        "Protaetia orientalis submarumorea"
      ],
      "old_values": [
        "Protaetia orientalis submarumorea",
        "Protaetia orientalis submarmorea"
      ],
      "old_values_raw": [
        "Protaetia orientalis submarumorea",
        "Protaetia orientalis submarmorea"
      ],
      "removed_unique_values": [
        "Protaetia orientalis submarumorea"
      ],
      "retained_unique_values": [
        "Protaetia orientalis submarmorea"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Protaetia orientalis submarumorea": {
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
          "normalized_match_text": "protaetia orientalis submarmorea",
          "raw_match_text": "Protaetia orientalis submarmorea",
          "source": "FOCUS_LABEL",
          "token": "Protaetia orientalis submarmorea"
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
        "Protaetia orientalis submarumorea",
        "Protaetia orientalis submarmorea"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 029. `repair_Q88358722_2445360305`

| Field | Value |
|---|---|
| qid | Q88358722 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q88358722::P225 |
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
| truth_tokens_preview | ["Muscari sabihapinariae"] |
| classification_target_tokens | ["Muscari sabihapinari"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Muscari sabihapinari"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Muscari sabihapinari"
  ],
  "removed_unique_values": [
    "Muscari sabihapinari"
  ],
  "retained_support_tokens": [
    "Muscari sabihapinariae"
  ],
  "retained_unique_values": [
    "Muscari sabihapinariae"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Muscari sabihapinariae"
  ],
  "old_value": [
    "Muscari sabihapinariae",
    "Muscari sabihapinari"
  ],
  "revision_id": 2445360305,
  "value": [
    "Muscari sabihapinariae"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Muscari sabihapinariae": 1
    },
    "new_unique": [
      "Muscari sabihapinariae"
    ],
    "new_values": [
      "Muscari sabihapinariae"
    ],
    "new_values_raw": [
      "Muscari sabihapinariae"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Muscari sabihapinari": 1,
      "Muscari sabihapinariae": 1
    },
    "old_unique": [
      "Muscari sabihapinari",
      "Muscari sabihapinariae"
    ],
    "old_values": [
      "Muscari sabihapinariae",
      "Muscari sabihapinari"
    ],
    "old_values_raw": [
      "Muscari sabihapinariae",
      "Muscari sabihapinari"
    ],
    "removed_unique_values": [
      "Muscari sabihapinari"
    ],
    "retained_unique_values": [
      "Muscari sabihapinariae"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Muscari sabihapinari": {
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
  "report_fix_date": "2025-12-23T14:31:41",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2446076213,
  "report_revision_old": 2445477126,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Muscari sabihapinariae",
    "Muscari sabihapinari"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "muscari sabihapinariae",
      "raw_match_text": "Muscari sabihapinariae",
      "source": "FOCUS_LABEL",
      "token": "Muscari sabihapinariae"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "muscari sabihapinariae",
      "raw_match_text": "Muscari sabihapinariae",
      "source": "FOCUS_LABEL",
      "token": "Muscari sabihapinariae"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Muscari sabihapinariae",
      "Muscari sabihapinari"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Muscari sabihapinariae"
  ],
  "truth_tokens_in_recorded_matches": [
    "Muscari sabihapinariae"
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
    "description": "species of plant",
    "label": "Muscari sabihapinariae"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Muscari sabihapinariae": 1
      },
      "new_unique": [
        "Muscari sabihapinariae"
      ],
      "new_values": [
        "Muscari sabihapinariae"
      ],
      "new_values_raw": [
        "Muscari sabihapinariae"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Muscari sabihapinari": 1,
        "Muscari sabihapinariae": 1
      },
      "old_unique": [
        "Muscari sabihapinari",
        "Muscari sabihapinariae"
      ],
      "old_values": [
        "Muscari sabihapinariae",
        "Muscari sabihapinari"
      ],
      "old_values_raw": [
        "Muscari sabihapinariae",
        "Muscari sabihapinari"
      ],
      "removed_unique_values": [
        "Muscari sabihapinari"
      ],
      "retained_unique_values": [
        "Muscari sabihapinariae"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Muscari sabihapinari": {
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
          "normalized_match_text": "muscari sabihapinariae",
          "raw_match_text": "Muscari sabihapinariae",
          "source": "FOCUS_LABEL",
          "token": "Muscari sabihapinariae"
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
        "Muscari sabihapinariae",
        "Muscari sabihapinari"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---

## 030. `repair_Q92285687_2444849890`

| Field | Value |
|---|---|
| qid | Q92285687 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_SELECTION_CONFIRMED / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_selection_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_selection_confirmed |
| decision_constraint_type |   |
| group_key | ABOX::Q92285687::P225 |
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
| truth_tokens_preview | ["Leontocebus nigrifrons"] |
| classification_target_tokens | ["Saguinus nigrifrons"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | local_selection_confirmed |
| rationale | Retained value in a subset repair has independent local support; pre-repair target values alone are not counted. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_LABEL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Saguinus nigrifrons"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Saguinus nigrifrons"
  ],
  "removed_unique_values": [
    "Saguinus nigrifrons"
  ],
  "retained_support_tokens": [
    "Leontocebus nigrifrons"
  ],
  "retained_unique_values": [
    "Leontocebus nigrifrons"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "local_evidence",
  "classification_rule_subfamily": "local_selection_confirmed",
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
    "Leontocebus nigrifrons"
  ],
  "old_value": [
    "Leontocebus nigrifrons",
    "Saguinus nigrifrons"
  ],
  "revision_id": 2444849890,
  "value": [
    "Leontocebus nigrifrons"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Leontocebus nigrifrons": 1
    },
    "new_unique": [
      "Leontocebus nigrifrons"
    ],
    "new_values": [
      "Leontocebus nigrifrons"
    ],
    "new_values_raw": [
      "Leontocebus nigrifrons"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Leontocebus nigrifrons": 1,
      "Saguinus nigrifrons": 1
    },
    "old_unique": [
      "Leontocebus nigrifrons",
      "Saguinus nigrifrons"
    ],
    "old_values": [
      "Leontocebus nigrifrons",
      "Saguinus nigrifrons"
    ],
    "old_values_raw": [
      "Leontocebus nigrifrons",
      "Saguinus nigrifrons"
    ],
    "removed_unique_values": [
      "Saguinus nigrifrons"
    ],
    "retained_unique_values": [
      "Leontocebus nigrifrons"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Saguinus nigrifrons": {
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
  "report_fix_date": "2025-12-22T11:22:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2445477126,
  "report_revision_old": 2444910674,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Leontocebus nigrifrons",
    "Saguinus nigrifrons"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 11,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "leontocebus nigrifrons",
      "raw_match_text": "Leontocebus nigrifrons",
      "source": "FOCUS_LABEL",
      "token": "Leontocebus nigrifrons"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact_raw",
      "normalized_match_text": "leontocebus nigrifrons",
      "raw_match_text": "Leontocebus nigrifrons",
      "source": "FOCUS_LABEL",
      "token": "Leontocebus nigrifrons"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Leontocebus nigrifrons",
      "Saguinus nigrifrons"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Leontocebus nigrifrons"
  ],
  "truth_tokens_in_recorded_matches": [
    "Leontocebus nigrifrons"
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
    "description": "species of mammal",
    "label": "Leontocebus nigrifrons"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Leontocebus nigrifrons": 1
      },
      "new_unique": [
        "Leontocebus nigrifrons"
      ],
      "new_values": [
        "Leontocebus nigrifrons"
      ],
      "new_values_raw": [
        "Leontocebus nigrifrons"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Leontocebus nigrifrons": 1,
        "Saguinus nigrifrons": 1
      },
      "old_unique": [
        "Leontocebus nigrifrons",
        "Saguinus nigrifrons"
      ],
      "old_values": [
        "Leontocebus nigrifrons",
        "Saguinus nigrifrons"
      ],
      "old_values_raw": [
        "Leontocebus nigrifrons",
        "Saguinus nigrifrons"
      ],
      "removed_unique_values": [
        "Saguinus nigrifrons"
      ],
      "retained_unique_values": [
        "Leontocebus nigrifrons"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Saguinus nigrifrons": {
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
          "normalized_match_text": "leontocebus nigrifrons",
          "raw_match_text": "Leontocebus nigrifrons",
          "source": "FOCUS_LABEL",
          "token": "Leontocebus nigrifrons"
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
        "Leontocebus nigrifrons",
        "Saguinus nigrifrons"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_selection_confirmed",
    "step": "branch"
  }
]
```

---
