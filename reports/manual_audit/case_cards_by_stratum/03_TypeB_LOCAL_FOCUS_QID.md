# TypeB_LOCAL_FOCUS_QID

Cases: 15

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q135499095_2404099124`

| Field | Value |
|---|---|
| qid | Q135499095 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135499095::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135499095"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135499095"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "11087"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404099124,
  "value": [
    "Q135499095"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q135499095"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q135499095"
    ],
    "new_value": [
      "Q135499095"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q135499095": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "11087"
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
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 13,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135499095"
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
    "Q135499095"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135499095"
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
    "label": "11087"
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
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135499095"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q135501847_2404100003`

| Field | Value |
|---|---|
| qid | Q135501847 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135501847::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135501847"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135501847"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "12619"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404100003,
  "value": [
    "Q135501847"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q135501847"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q135501847"
    ],
    "new_value": [
      "Q135501847"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q135501847": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "12619"
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
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 15,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135501847"
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
    "Q135501847"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135501847"
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
    "label": "12619"
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
          "token": "Q135501847"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q135501905_2388789166`

| Field | Value |
|---|---|
| qid | Q135501905 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135501905::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135501905"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q135501905"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "12671"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2388789166,
  "value": [
    "Q135501905"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q135501905"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q135501905"
    ],
    "new_value": [
      "Q135501905"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q135501905": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "12671"
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
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 15,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135501905"
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
    "Q135501905"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135501905"
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
    "label": "12671"
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
          "token": "Q135501905"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q135502896_2388871353`

| Field | Value |
|---|---|
| qid | Q135502896 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135502896::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135502896"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q135502896"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "13259"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2388871353,
  "value": [
    "Q135502896"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q135502896"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q135502896"
    ],
    "new_value": [
      "Q135502896"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q135502896": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "13259"
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
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 15,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135502896"
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
    "Q135502896"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135502896"
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
    "label": "13259"
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
          "token": "Q135502896"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q135503247_2404101603`

| Field | Value |
|---|---|
| qid | Q135503247 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135503247::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135503247"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135503247"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "13513"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404101603,
  "value": [
    "Q135503247"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q135503247"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q135503247"
    ],
    "new_value": [
      "Q135503247"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q135503247": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "13513"
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
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 13,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135503247"
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
    "Q135503247"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135503247"
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
    "label": "13513"
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
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135503247"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q135504124_2404098581`

| Field | Value |
|---|---|
| qid | Q135504124 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135504124::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135504124"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135504124"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "14143"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404098581,
  "value": [
    "Q135504124"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q135504124"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q135504124"
    ],
    "new_value": [
      "Q135504124"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q135504124": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "14143"
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
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 13,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135504124"
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
    "Q135504124"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135504124"
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
    "label": "14143"
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
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135504124"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q135514311_2404099664`

| Field | Value |
|---|---|
| qid | Q135514311 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135514311::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135514311"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135514311"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "21391"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404099664,
  "value": [
    "Q135514311"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q135514311"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q135514311"
    ],
    "new_value": [
      "Q135514311"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q135514311": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "21391"
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
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 13,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135514311"
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
    "Q135514311"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135514311"
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
    "label": "21391"
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
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135514311"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q135515103_2404101292`

| Field | Value |
|---|---|
| qid | Q135515103 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135515103::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135515103"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135515103"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "22111"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404101292,
  "value": [
    "Q135515103"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q135515103"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q135515103"
    ],
    "new_value": [
      "Q135515103"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q135515103": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "22111"
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
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 13,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135515103"
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
    "Q135515103"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135515103"
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
    "label": "22111"
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
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135515103"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q135516242_2404100053`

| Field | Value |
|---|---|
| qid | Q135516242 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135516242::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135516242"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135516242"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "22811"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404100053,
  "value": [
    "Q135516242"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q135516242"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q135516242"
    ],
    "new_value": [
      "Q135516242"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q135516242": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "22811"
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
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 13,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135516242"
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
    "Q135516242"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135516242"
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
    "label": "22811"
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
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135516242"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q135516516_2404100959`

| Field | Value |
|---|---|
| qid | Q135516516 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135516516::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135516516"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135516516"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "22993"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404100959,
  "value": [
    "Q135516516"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q135516516"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q135516516"
    ],
    "new_value": [
      "Q135516516"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q135516516": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "22993"
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
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 13,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135516516"
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
    "Q135516516"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135516516"
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
    "label": "22993"
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
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135516516"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 011. `repair_Q135517447_2404098544`

| Field | Value |
|---|---|
| qid | Q135517447 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135517447::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135517447"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135517447"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "23431"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404098544,
  "value": [
    "Q135517447"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q135517447"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q135517447"
    ],
    "new_value": [
      "Q135517447"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q135517447": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "23431"
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
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 13,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135517447"
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
    "Q135517447"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135517447"
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
    "label": "23431"
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
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q135517447"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 012. `repair_Q135520071_2388856522`

| Field | Value |
|---|---|
| qid | Q135520071 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135520071::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135520071"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q135520071"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "18719"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2388856522,
  "value": [
    "Q135520071"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q135520071"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q135520071"
    ],
    "new_value": [
      "Q135520071"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q135520071": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "18719"
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
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 15,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135520071"
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
    "Q135520071"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135520071"
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
    "label": "18719"
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
          "token": "Q135520071"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 013. `repair_Q135536493_2404100553`

| Field | Value |
|---|---|
| qid | Q135536493 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135536493::P5236 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q135536493"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "Q135536493"
  ],
  "new_value_descriptions_en": [
    "natural number"
  ],
  "new_value_labels_en": [
    "32363"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2404100553,
  "value": [
    "Q135536493"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q135536493"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q135536493"
    ],
    "new_value": [
      "Q135536493"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q135536493": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "32363"
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
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 15,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q135536493"
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
    "Q135536493"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q135536493"
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
    "label": "32363"
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
          "token": "Q135536493"
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
    "result": "local_match",
    "step": "branch"
  }
]
```

---

## 014. `repair_Q43845_2441128800`

| Field | Value |
|---|---|
| qid | Q43845 |
| property | P279 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| group_key | ABOX::Q43845::P279 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q108286992", "Q43845"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Q108286992",
    "Q43845"
  ],
  "new_value_descriptions_en": [
    "druh zaměstnání podle ISCO",
    "person involved in activities for the purpose of generating revenue"
  ],
  "new_value_labels_en": [
    "business and administration professionals",
    "businessperson"
  ],
  "old_value": [
    "Q108286992"
  ],
  "old_value_descriptions_en": [
    "druh zaměstnání podle ISCO"
  ],
  "old_value_labels_en": [
    "business and administration professionals"
  ],
  "revision_id": 2441128800,
  "value": [
    "Q108286992",
    "Q43845"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q43845"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "Q108286992",
      "Q43845"
    ],
    "new_value": [
      "Q108286992",
      "Q43845"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q108286992"
    ],
    "old_value": [
      "Q108286992"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "Q43845": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "druh zaměstnání podle ISCO",
    "person involved in activities for the purpose of generating revenue"
  ],
  "value_labels_en": [
    "business and administration professionals",
    "businessperson"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-13T11:16:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P279",
  "report_revision_new": 2441794953,
  "report_revision_old": 2441232862,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q108286992"
  ],
  "value_descriptions_en": [
    "druh zaměstnání podle ISCO"
  ],
  "value_labels_en": [
    "business and administration professionals"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 24,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q43845"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q108286992"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q108286992",
    "Q43845"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q43845"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "this item is a subclass (subset) of that item; ALL instances of this item are instances of that item; different from P31 (instance of), e.g.: volcano is a subclass of mountain; Everest is an instance of mountain",
    "label": "subclass of"
  },
  "qid": {
    "description": "person involved in activities for the purpose of generating revenue",
    "label": "businessperson"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
    "result": false,
    "step": "is_delete"
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
      "local_ids_count": 24,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q43845"
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
        "Q108286992"
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

## 015. `repair_Q924335_2446691985`

| Field | Value |
|---|---|
| qid | Q924335 |
| property | P6379 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_FOCUS_QID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_focus_qid |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q924335::P6379 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether the recorded local match actually supports the historical target.
- For LOCAL_TEXT_CONFIRMED, check that the match is independent focus/neighbor text rather than a retained target-property value.
- For LOCAL_SELECTION_CONFIRMED, confirm that retained-value support is independent of the pre-repair target-property list.
- For LOCAL_FOCUS_QID, confirm that the added/created value really equals the focus entity id.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q13882236", "Q3983824", "Q679527", "Q924335"] |
| decision_branch | local_match |
| rationale | Repair target matched the focus entity id. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_QID |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Q13882236",
    "Q3983824",
    "Q679527",
    "Q924335"
  ],
  "new_value_descriptions_en": [
    "museum in Hattem, the Netherlands",
    "museum in Tilburg, the Netherlands about textile design and history",
    "art museum in Rotterdam, Netherlands",
    "art museum in Amsterdam, Netherlands"
  ],
  "new_value_labels_en": [
    "Bakkerij Museum",
    "TextielMuseum",
    "Museum Boijmans Van Beuningen",
    "Stedelijk Museum Amsterdam"
  ],
  "old_value": [
    "Q13882236",
    "Q3983824",
    "Q679527"
  ],
  "old_value_descriptions_en": [
    "museum in Hattem, the Netherlands",
    "museum in Tilburg, the Netherlands about textile design and history",
    "art museum in Rotterdam, Netherlands"
  ],
  "old_value_labels_en": [
    "Bakkerij Museum",
    "TextielMuseum",
    "Museum Boijmans Van Beuningen"
  ],
  "revision_id": 2446691985,
  "value": [
    "Q13882236",
    "Q3983824",
    "Q679527",
    "Q924335"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q924335"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 4,
    "new_unique": [
      "Q13882236",
      "Q3983824",
      "Q679527",
      "Q924335"
    ],
    "new_value": [
      "Q13882236",
      "Q3983824",
      "Q679527",
      "Q924335"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 3,
    "old_unique": [
      "Q13882236",
      "Q3983824",
      "Q679527"
    ],
    "old_value": [
      "Q13882236",
      "Q3983824",
      "Q679527"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "Q924335": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "museum in Hattem, the Netherlands",
    "museum in Tilburg, the Netherlands about textile design and history",
    "art museum in Rotterdam, Netherlands",
    "art museum in Amsterdam, Netherlands"
  ],
  "value_labels_en": [
    "Bakkerij Museum",
    "TextielMuseum",
    "Museum Boijmans Van Beuningen",
    "Stedelijk Museum Amsterdam"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-25T14:33:14",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6379",
  "report_revision_new": 2446977520,
  "report_revision_old": 2446399910,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q13882236",
    "Q3983824",
    "Q679527"
  ],
  "value_descriptions_en": [
    "museum in Hattem, the Netherlands",
    "museum in Tilburg, the Netherlands about textile design and history",
    "art museum in Rotterdam, Netherlands"
  ],
  "value_labels_en": [
    "Bakkerij Museum",
    "TextielMuseum",
    "Museum Boijmans Van Beuningen"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 80,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "id_exact",
      "source": "FOCUS_QID",
      "token": "Q924335"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q13882236",
      "Q3983824",
      "Q679527"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q13882236",
    "Q3983824",
    "Q679527",
    "Q924335"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q924335"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "collection that has works of this person or organisation (use archives at [P485] for archives)",
    "label": "has works in the collection"
  },
  "qid": {
    "description": "art museum in Amsterdam, Netherlands",
    "label": "Stedelijk Museum Amsterdam"
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
      "local_ids_count": 80,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "id_exact",
          "source": "FOCUS_QID",
          "token": "Q924335"
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
        "Q13882236",
        "Q3983824",
        "Q679527"
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
