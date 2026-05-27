# TypeB_LOCAL_TEXT_CONFIRMED

Cases: 40

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
| group_key | ABOX::Q117320046::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Gavansky Residential Complex for Workers"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | NEIGHBOR_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Gavansky Residential Complex for Workers"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Gavansky Residential Complex for Workers"
    ],
    "new_value": [
      "Gavansky Residential Complex for Workers"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Gavansky Residental Complex for Workers"
    ],
    "old_value": [
      "Gavansky Residental Complex for Workers"
    ],
    "removed_unique_values": [
      "Gavansky Residental Complex for Workers"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 002. `repair_Q117320047_2446824555`

| Field | Value |
|---|---|
| qid | Q117320047 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| group_key | ABOX::Q117320047::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Gavansky Residential Complex for Workers"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | NEIGHBOR_LABEL |

### What Changed

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
  "revision_id": 2446824555,
  "value": [
    "Gavansky Residential Complex for Workers"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Gavansky Residential Complex for Workers"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Gavansky Residential Complex for Workers"
    ],
    "new_value": [
      "Gavansky Residential Complex for Workers"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Gavansky Residental Complex for Workers"
    ],
    "old_value": [
      "Gavansky Residental Complex for Workers"
    ],
    "removed_unique_values": [
      "Gavansky Residental Complex for Workers"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 003. `repair_Q117320048_2446824658`

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
| group_key | ABOX::Q117320048::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Gavansky Residential Complex for Workers"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | NEIGHBOR_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Gavansky Residential Complex for Workers"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Gavansky Residential Complex for Workers"
    ],
    "new_value": [
      "Gavansky Residential Complex for Workers"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Gavansky Residental Complex for Workers"
    ],
    "old_value": [
      "Gavansky Residental Complex for Workers"
    ],
    "removed_unique_values": [
      "Gavansky Residental Complex for Workers"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 004. `repair_Q123734475_2447259961`

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
| group_key | ABOX::Q123734475::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Frosinone Calcio v US Città di Palermo, 16 June 2018"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Frosinone Calcio v US Città di Palermo, 16 June 2018"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Frosinone Calcio v US Città di Palermo, 16 June 2018"
    ],
    "new_value": [
      "Frosinone Calcio v US Città di Palermo, 16 June 2018"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
    ],
    "old_value": [
      "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
    ],
    "removed_unique_values": [
      "2017-18 Serie B - Frosinone Calcio v Palermo (finale Play-off)"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 005. `repair_Q123749072_2447260069`

| Field | Value |
|---|---|
| qid | Q123749072 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q123749072::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Paris Saint-Germain FC v ASJ Soyaux-Charente, 25 August 2019"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Pi bot",
  "kind": "A_BOX",
  "new_value": [
    "Paris Saint-Germain FC v ASJ Soyaux-Charente, 25 August 2019"
  ],
  "old_value": [
    "PSG - Soyaux, 25 August 2019"
  ],
  "revision_id": 2447260069,
  "value": [
    "Paris Saint-Germain FC v ASJ Soyaux-Charente, 25 August 2019"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Paris Saint-Germain FC v ASJ Soyaux-Charente, 25 August 2019"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Paris Saint-Germain FC v ASJ Soyaux-Charente, 25 August 2019"
    ],
    "new_value": [
      "Paris Saint-Germain FC v ASJ Soyaux-Charente, 25 August 2019"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "PSG - Soyaux, 25 August 2019"
    ],
    "old_value": [
      "PSG - Soyaux, 25 August 2019"
    ],
    "removed_unique_values": [
      "PSG - Soyaux, 25 August 2019"
    ],
    "value_multiplicity_changes": {
      "PSG - Soyaux, 25 August 2019": {
        "new": 0,
        "old": 1
      },
      "Paris Saint-Germain FC v ASJ Soyaux-Charente, 25 August 2019": {
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
    "PSG - Soyaux, 25 August 2019"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 3,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "Paris Saint-Germain FC v ASJ Soyaux-Charente, 25 August 2019"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "PSG - Soyaux, 25 August 2019"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Paris Saint-Germain FC v ASJ Soyaux-Charente, 25 August 2019"
  ],
  "truth_tokens_in_recorded_matches": [
    "Paris Saint-Germain FC v ASJ Soyaux-Charente, 25 August 2019"
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
    "label": "Paris Saint-Germain FC v ASJ Soyaux-Charente, 25 August 2019"
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
      "local_ids_count": 3,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "Paris Saint-Germain FC v ASJ Soyaux-Charente, 25 August 2019"
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
        "PSG - Soyaux, 25 August 2019"
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

## 006. `repair_Q137217940_2439242692`

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
| group_key | ABOX::Q137217940::P225 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Magnadigita"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Magnadigita"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Magnadigita"
    ],
    "new_value": [
      "Magnadigita"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Bolitoglossa"
    ],
    "old_value": [
      "Bolitoglossa"
    ],
    "removed_unique_values": [
      "Bolitoglossa"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 007. `repair_Q137217945_2439242986`

| Field | Value |
|---|---|
| qid | Q137217945 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q137217945::P225 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Mayamandra"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "PieterJanR",
  "kind": "A_BOX",
  "new_value": [
    "Mayamandra"
  ],
  "old_value": [
    "Bolitoglossa"
  ],
  "revision_id": 2439242986,
  "value": [
    "Mayamandra"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Mayamandra"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Mayamandra"
    ],
    "new_value": [
      "Mayamandra"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Bolitoglossa"
    ],
    "old_value": [
      "Bolitoglossa"
    ],
    "removed_unique_values": [
      "Bolitoglossa"
    ],
    "value_multiplicity_changes": {
      "Bolitoglossa": {
        "new": 0,
        "old": 1
      },
      "Mayamandra": {
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "Mayamandra"
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
    "Mayamandra"
  ],
  "truth_tokens_in_recorded_matches": [
    "Mayamandra"
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
    "label": "Mayamandra"
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
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "Mayamandra"
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

## 008. `repair_Q137219619_2438468445`

| Field | Value |
|---|---|
| qid | Q137219619 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q137219619::P225 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Hypnum parietinum"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "CREATE",
  "author": "FL0RA 1234",
  "kind": "A_BOX",
  "new_value": [
    "Hypnum parietinum"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2438468445,
  "value": [
    "Hypnum parietinum"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Hypnum parietinum"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Hypnum parietinum"
    ],
    "new_value": [
      "Hypnum parietinum"
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
      "Hypnum parietinum": {
        "new": 1,
        "old": 0
      },
      "MISSING": {
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
  "report_fix_date": "2025-12-09T12:33:44",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2440014373,
  "report_revision_old": 2439564746,
  "report_violation_type": "Item P|105",
  "report_violation_type_normalized": "Item P|105",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|105",
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
  "local_ids_count": 5,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "Hypnum parietinum"
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
    "Hypnum parietinum"
  ],
  "truth_tokens_in_recorded_matches": [
    "Hypnum parietinum"
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
    "description": "species of the genus Hypnum",
    "label": "Hypnum parietinum"
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
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "Hypnum parietinum"
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

## 009. `repair_Q137288654_2440360795`

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
| group_key | ABOX::Q137288654::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Ruinas de la presa de la Estanca"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Ruinas de la presa de la Estanca"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Ruinas de la presa de la Estanca"
    ],
    "new_value": [
      "Ruinas de la presa de la Estanca"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Ruinas de la presa de la Estanca (Cascante)"
    ],
    "old_value": [
      "Ruinas de la presa de la Estanca (Cascante)"
    ],
    "removed_unique_values": [
      "Ruinas de la presa de la Estanca (Cascante)"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 010. `repair_Q137366682_2443217037`

| Field | Value |
|---|---|
| qid | Q137366682 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| group_key | ABOX::Q137366682::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["2025 Bulgarian budget protests"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "2025 Bulgarian budget protests"
  ],
  "old_value": [
    "Category:2025 Bulgarian budget protests"
  ],
  "revision_id": 2443217037,
  "value": [
    "2025 Bulgarian budget protests"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "2025 Bulgarian budget protests"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2025 Bulgarian budget protests"
    ],
    "new_value": [
      "2025 Bulgarian budget protests"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Category:2025 Bulgarian budget protests"
    ],
    "old_value": [
      "Category:2025 Bulgarian budget protests"
    ],
    "removed_unique_values": [
      "Category:2025 Bulgarian budget protests"
    ],
    "value_multiplicity_changes": {
      "2025 Bulgarian budget protests": {
        "new": 1,
        "old": 0
      },
      "Category:2025 Bulgarian budget protests": {
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
  "report_fix_date": "2025-12-18T18:54:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2443834913,
  "report_revision_old": 2443399922,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Commons link"
  ],
  "value": [
    "Category:2025 Bulgarian budget protests"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 33,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "2025 Bulgarian budget protests"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Category:2025 Bulgarian budget protests"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "2025 Bulgarian budget protests"
  ],
  "truth_tokens_in_recorded_matches": [
    "2025 Bulgarian budget protests"
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
    "description": "Gen Z protests over increasing taxes",
    "label": "2025 Bulgarian budget protests"
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
      "local_ids_count": 33,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "2025 Bulgarian budget protests"
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
        "Category:2025 Bulgarian budget protests"
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

## 011. `repair_Q137375874_2442357718`

| Field | Value |
|---|---|
| qid | Q137375874 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| group_key | ABOX::Q137375874::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Nabila Idris"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Nabila Idris"
  ],
  "old_value": [
    "Category:Nabila Idris"
  ],
  "revision_id": 2442357718,
  "value": [
    "Nabila Idris"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Nabila Idris"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Nabila Idris"
    ],
    "new_value": [
      "Nabila Idris"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Category:Nabila Idris"
    ],
    "old_value": [
      "Category:Nabila Idris"
    ],
    "removed_unique_values": [
      "Category:Nabila Idris"
    ],
    "value_multiplicity_changes": {
      "Category:Nabila Idris": {
        "new": 0,
        "old": 1
      },
      "Nabila Idris": {
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
  "report_fix_date": "2025-12-16T11:51:46",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2442981743,
  "report_revision_old": 2442645840,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Commons link"
  ],
  "value": [
    "Category:Nabila Idris"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 27,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "Nabila Idris"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Category:Nabila Idris"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Nabila Idris"
  ],
  "truth_tokens_in_recorded_matches": [
    "Nabila Idris"
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
    "description": "A Bangladeshi academic, social protection expert and human rights activist, she received the Begum Rokeya Medal in 2025 in recognition of her distinguished contributions to the field of human rights.",
    "label": "Nabila Idris"
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
      "local_ids_count": 27,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": true,
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "Nabila Idris"
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
        "Category:Nabila Idris"
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

## 012. `repair_Q137379963_2443215624`

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
| group_key | ABOX::Q137379963::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["The Rewynd"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "The Rewynd"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "The Rewynd"
    ],
    "new_value": [
      "The Rewynd"
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 013. `repair_Q137385306_2442996855`

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
| group_key | ABOX::Q137385306::P225 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Pachyspathe"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Pachyspathe"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Pachyspathe"
    ],
    "new_value": [
      "Pachyspathe"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Nephus"
    ],
    "old_value": [
      "Nephus"
    ],
    "removed_unique_values": [
      "Nephus"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 014. `repair_Q137392742_2442642262`

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
| group_key | ABOX::Q137392742::P225 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Vermeulenia"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "Vermeulenia"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Vermeulenia"
    ],
    "new_value": [
      "Vermeulenia"
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 015. `repair_Q137397574_2442915141`

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
| group_key | ABOX::Q137397574::P225 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Trochilus rubricauda"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "Trochilus rubricauda"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Trochilus rubricauda"
    ],
    "new_value": [
      "Trochilus rubricauda"
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 016. `repair_Q14215493_2447090540`

| Field | Value |
|---|---|
| qid | Q14215493 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q14215493::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Festivals in Kerala"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Festivals in Kerala"
  ],
  "old_value": [
    "Festivals of Kerala"
  ],
  "revision_id": 2447090540,
  "value": [
    "Festivals in Kerala"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Festivals in Kerala"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Festivals in Kerala"
    ],
    "new_value": [
      "Festivals in Kerala"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Festivals of Kerala"
    ],
    "old_value": [
      "Festivals of Kerala"
    ],
    "removed_unique_values": [
      "Festivals of Kerala"
    ],
    "value_multiplicity_changes": {
      "Festivals in Kerala": {
        "new": 1,
        "old": 0
      },
      "Festivals of Kerala": {
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
    "Festivals of Kerala"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 9,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "Festivals in Kerala"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Festivals of Kerala"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Festivals in Kerala"
  ],
  "truth_tokens_in_recorded_matches": [
    "Festivals in Kerala"
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
    "description": "part of the culture of Kerala, India",
    "label": "festivals in Kerala"
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
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "Festivals in Kerala"
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
        "Festivals of Kerala"
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

## 017. `repair_Q144617_2445327292`

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
| group_key | ABOX::Q144617::P4264 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["tim"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | NEIGHBOR_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "tim"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "tim"
    ],
    "new_value": [
      "tim"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "telecom-italia"
    ],
    "old_value": [
      "telecom-italia"
    ],
    "removed_unique_values": [
      "telecom-italia"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 018. `repair_Q1578657_2447257297`

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
| group_key | ABOX::Q1578657::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Sony Ericsson W995"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Sony Ericsson W995"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Sony Ericsson W995"
    ],
    "new_value": [
      "Sony Ericsson W995"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Sony Ericsson W995i"
    ],
    "old_value": [
      "Sony Ericsson W995i"
    ],
    "removed_unique_values": [
      "Sony Ericsson W995i"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 019. `repair_Q17514215_2443430384`

| Field | Value |
|---|---|
| qid | Q17514215 |
| property | P4264 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q52060874 |
| group_key | ABOX::Q17514215::P4264 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["vacasa"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "vacasa"
  ],
  "old_value": [
    "https://www.linkedin.com/company/vacasa/"
  ],
  "revision_id": 2443430384,
  "value": [
    "vacasa"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "vacasa"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "vacasa"
    ],
    "new_value": [
      "vacasa"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "https://www.linkedin.com/company/vacasa/"
    ],
    "old_value": [
      "https://www.linkedin.com/company/vacasa/"
    ],
    "removed_unique_values": [
      "https://www.linkedin.com/company/vacasa/"
    ],
    "value_multiplicity_changes": {
      "https://www.linkedin.com/company/vacasa/": {
        "new": 0,
        "old": 1
      },
      "vacasa": {
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
  "report_fix_date": "2025-12-19T08:06:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4264",
  "report_revision_new": 2443996360,
  "report_revision_old": 2443785860,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "https://www.linkedin.com/company/vacasa/"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 9,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "vacasa"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "https://www.linkedin.com/company/vacasa/"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "vacasa"
  ],
  "truth_tokens_in_recorded_matches": [
    "vacasa"
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
    "description": "vacation rental management company",
    "label": "Vacasa"
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
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "vacasa"
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
        "https://www.linkedin.com/company/vacasa/"
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

## 020. `repair_Q1753812_2440724915`

| Field | Value |
|---|---|
| qid | Q1753812 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| group_key | ABOX::Q1753812::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Vikulovsky District"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Pi bot",
  "kind": "A_BOX",
  "new_value": [
    "Vikulovsky District"
  ],
  "old_value": [
    "Category:Vikulovsky District"
  ],
  "revision_id": 2440724915,
  "value": [
    "Vikulovsky District"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Vikulovsky District"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Vikulovsky District"
    ],
    "new_value": [
      "Vikulovsky District"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Category:Vikulovsky District"
    ],
    "old_value": [
      "Category:Vikulovsky District"
    ],
    "removed_unique_values": [
      "Category:Vikulovsky District"
    ],
    "value_multiplicity_changes": {
      "Category:Vikulovsky District": {
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
  "report_fix_date": "2025-12-12T11:04:55",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2441207555,
  "report_revision_old": 2440854653,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Commons link"
  ],
  "value": [
    "Category:Vikulovsky District"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 23,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "Vikulovsky District"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Category:Vikulovsky District"
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
    "description": "human settlement in Russia",
    "label": "Vikulovsky District"
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
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "Vikulovsky District"
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
        "Category:Vikulovsky District"
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

## 021. `repair_Q18597760_2447258320`

| Field | Value |
|---|---|
| qid | Q18597760 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q18597760::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["AS Montigny-le-Bretonneux"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | NEIGHBOR_LABEL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Pi bot",
  "kind": "A_BOX",
  "new_value": [
    "AS Montigny-le-Bretonneux"
  ],
  "old_value": [
    "Association sportive de Montigny-le-Bretonneux"
  ],
  "revision_id": 2447258320,
  "value": [
    "AS Montigny-le-Bretonneux"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "AS Montigny-le-Bretonneux"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "AS Montigny-le-Bretonneux"
    ],
    "new_value": [
      "AS Montigny-le-Bretonneux"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Association sportive de Montigny-le-Bretonneux"
    ],
    "old_value": [
      "Association sportive de Montigny-le-Bretonneux"
    ],
    "removed_unique_values": [
      "Association sportive de Montigny-le-Bretonneux"
    ],
    "value_multiplicity_changes": {
      "AS Montigny-le-Bretonneux": {
        "new": 1,
        "old": 0
      },
      "Association sportive de Montigny-le-Bretonneux": {
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
    "Association sportive de Montigny-le-Bretonneux"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 5,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
      "source": "NEIGHBOR_LABEL",
      "token": "AS Montigny-le-Bretonneux"
    }
  ],
  "needed": 1,
  "sources_used": [
    "NEIGHBOR_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Association sportive de Montigny-le-Bretonneux"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "AS Montigny-le-Bretonneux"
  ],
  "truth_tokens_in_recorded_matches": [
    "AS Montigny-le-Bretonneux"
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
    "label": "Category:Association sportive de Montigny-le-Bretonneux"
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
          "kind": "literal_exact",
          "source": "NEIGHBOR_LABEL",
          "token": "AS Montigny-le-Bretonneux"
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
        "Association sportive de Montigny-le-Bretonneux"
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

## 022. `repair_Q205953_2441113979`

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
| group_key | ABOX::Q205953::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Fedor Emelianenko"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Fedor Emelianenko"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Fedor Emelianenko"
    ],
    "new_value": [
      "Fedor Emelianenko"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Fedor Emelianenkov"
    ],
    "old_value": [
      "Fedor Emelianenkov"
    ],
    "removed_unique_values": [
      "Fedor Emelianenkov"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 023. `repair_Q25239177_2447091414`

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
| group_key | ABOX::Q25239177::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Folk festivals in Japan"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Folk festivals in Japan"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Folk festivals in Japan"
    ],
    "new_value": [
      "Folk festivals in Japan"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Folk festivals of Japan"
    ],
    "old_value": [
      "Folk festivals of Japan"
    ],
    "removed_unique_values": [
      "Folk festivals of Japan"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_boundary",
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

## 024. `repair_Q2756_2447374243`

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
| group_key | ABOX::Q2756::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Siena FC"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "Siena FC"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Siena FC"
    ],
    "new_value": [
      "Siena FC"
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 025. `repair_Q31868187_2442784183`

| Field | Value |
|---|---|
| qid | Q31868187 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| group_key | ABOX::Q31868187::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Plástovice čp. 24"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "CREATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Plástovice čp. 24"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2442784183,
  "value": [
    "Plástovice čp. 24"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Plástovice čp. 24"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Plástovice čp. 24"
    ],
    "new_value": [
      "Plástovice čp. 24"
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
      "Plástovice čp. 24": {
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
  "local_ids_count": 11,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "Plástovice čp. 24"
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
    "Plástovice čp. 24"
  ],
  "truth_tokens_in_recorded_matches": [
    "Plástovice čp. 24"
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
    "description": "usedlost",
    "label": "Plástovice čp. 24"
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
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "Plástovice čp. 24"
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

## 026. `repair_Q32250854_2447091819`

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
| group_key | ABOX::Q32250854::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Vikulovsky District"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | NEIGHBOR_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Vikulovsky District"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Vikulovsky District"
    ],
    "new_value": [
      "Vikulovsky District"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Vikulovo rayon"
    ],
    "old_value": [
      "Vikulovo rayon"
    ],
    "removed_unique_values": [
      "Vikulovo rayon"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 027. `repair_Q334056_2447087909`

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
| group_key | ABOX::Q334056::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Muri-Gries Abbey"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Muri-Gries Abbey"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Muri-Gries Abbey"
    ],
    "new_value": [
      "Muri-Gries Abbey"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Abtei Muri-Gries"
    ],
    "old_value": [
      "Abtei Muri-Gries"
    ],
    "removed_unique_values": [
      "Abtei Muri-Gries"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 028. `repair_Q3797851_2442499584`

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
| group_key | ABOX::Q3797851::P225 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Mopalia schrencki"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Mopalia schrencki"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Mopalia schrencki"
    ],
    "new_value": [
      "Mopalia schrencki"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Mopalia schrenck"
    ],
    "old_value": [
      "Mopalia schrenck"
    ],
    "removed_unique_values": [
      "Mopalia schrenck"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 029. `repair_Q5360360_2447089476`

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
| group_key | ABOX::Q5360360::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Elia, Keryneias"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Elia, Keryneias"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Elia, Keryneias"
    ],
    "new_value": [
      "Elia, Keryneias"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Elia Keryneias"
    ],
    "old_value": [
      "Elia Keryneias"
    ],
    "removed_unique_values": [
      "Elia Keryneias"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 030. `repair_Q56276464_2439595160`

| Field | Value |
|---|---|
| qid | Q56276464 |
| property | P2088 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502410 |
| group_key | ABOX::Q56276464::P2088 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["blueground"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "~2025-39122-72",
  "kind": "A_BOX",
  "new_value": [
    "blueground"
  ],
  "old_value": [
    "http://www.crunchbase.com/organization/blueground"
  ],
  "revision_id": 2439595160,
  "value": [
    "blueground"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "blueground"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "blueground"
    ],
    "new_value": [
      "blueground"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "http://www.crunchbase.com/organization/blueground"
    ],
    "old_value": [
      "http://www.crunchbase.com/organization/blueground"
    ],
    "removed_unique_values": [
      "http://www.crunchbase.com/organization/blueground"
    ],
    "value_multiplicity_changes": {
      "blueground": {
        "new": 1,
        "old": 0
      },
      "http://www.crunchbase.com/organization/blueground": {
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
  "report_fix_date": "2025-12-10T08:07:36",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2088",
  "report_revision_new": 2440381917,
  "report_revision_old": 2439924877,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "http://www.crunchbase.com/organization/blueground"
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
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "blueground"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "http://www.crunchbase.com/organization/blueground"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "blueground"
  ],
  "truth_tokens_in_recorded_matches": [
    "blueground"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "Identifier for an organization, in the Crunchbase database of companies and start-ups, operated by TechCrunch",
    "label": "Crunchbase organization ID"
  },
  "qid": {
    "description": "A global PropTech company offering fully furnished, turnkey apartments for flexible short‑ to long‑term stays, available in many cities worldwide.",
    "label": "Blueground"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
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
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "blueground"
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
        "http://www.crunchbase.com/organization/blueground"
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

## 031. `repair_Q5769641_2447089552`

| Field | Value |
|---|---|
| qid | Q5769641 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| group_key | ABOX::Q5769641::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Opisthostoma everetti"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Opisthostoma everetti"
  ],
  "old_value": [
    "Opisthostoma everettii"
  ],
  "revision_id": 2447089552,
  "value": [
    "Opisthostoma everetti"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Opisthostoma everetti"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Opisthostoma everetti"
    ],
    "new_value": [
      "Opisthostoma everetti"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Opisthostoma everettii"
    ],
    "old_value": [
      "Opisthostoma everettii"
    ],
    "removed_unique_values": [
      "Opisthostoma everettii"
    ],
    "value_multiplicity_changes": {
      "Opisthostoma everetti": {
        "new": 1,
        "old": 0
      },
      "Opisthostoma everettii": {
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
    "Opisthostoma everettii"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "Opisthostoma everetti"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Opisthostoma everettii"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Opisthostoma everetti"
  ],
  "truth_tokens_in_recorded_matches": [
    "Opisthostoma everetti"
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
    "description": "species of snail",
    "label": "Opisthostoma everetti"
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
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "Opisthostoma everetti"
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
        "Opisthostoma everettii"
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

## 032. `repair_Q6073972_2447257679`

| Field | Value |
|---|---|
| qid | Q6073972 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| group_key | ABOX::Q6073972::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Peñalcázar"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Pi bot",
  "kind": "A_BOX",
  "new_value": [
    "Peñalcázar"
  ],
  "old_value": [
    "Peñalcazar"
  ],
  "revision_id": 2447257679,
  "value": [
    "Peñalcázar"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Peñalcázar"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Peñalcázar"
    ],
    "new_value": [
      "Peñalcázar"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Peñalcazar"
    ],
    "old_value": [
      "Peñalcazar"
    ],
    "removed_unique_values": [
      "Peñalcazar"
    ],
    "value_multiplicity_changes": {
      "Peñalcazar": {
        "new": 0,
        "old": 1
      },
      "Peñalcázar": {
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
    "Peñalcazar"
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
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "Peñalcázar"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Peñalcazar"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Peñalcázar"
  ],
  "truth_tokens_in_recorded_matches": [
    "Peñalcázar"
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
    "description": "depopulated of Spain in the municipality of La Quiñonería",
    "label": "Peñalcázar"
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
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "Peñalcázar"
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
        "Peñalcazar"
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

## 033. `repair_Q61872875_2447258973`

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
| group_key | ABOX::Q61872875::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Dr. Fazıl Küçük Museum"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Dr. Fazıl Küçük Museum"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Dr. Fazıl Küçük Museum"
    ],
    "new_value": [
      "Dr. Fazıl Küçük Museum"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Dr. Fazıl Küçük Müzesi"
    ],
    "old_value": [
      "Dr. Fazıl Küçük Müzesi"
    ],
    "removed_unique_values": [
      "Dr. Fazıl Küçük Müzesi"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 034. `repair_Q624028_2447088024`

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
| group_key | ABOX::Q624028::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Credit default swap"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Credit default swap"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Credit default swap"
    ],
    "new_value": [
      "Credit default swap"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Credit defaut swap"
    ],
    "old_value": [
      "Credit defaut swap"
    ],
    "removed_unique_values": [
      "Credit defaut swap"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 035. `repair_Q6767_2447372569`

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
| group_key | ABOX::Q6767::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["US Livorno 1915"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "US Livorno 1915"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "US Livorno 1915"
    ],
    "new_value": [
      "US Livorno 1915"
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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

## 036. `repair_Q7495220_2444599258`

| Field | Value |
|---|---|
| qid | Q7495220 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| group_key | ABOX::Q7495220::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Sherman Oaks Galleria"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Sherman Oaks Galleria"
  ],
  "old_value": [
    "Category:Sherman Oaks Galleria"
  ],
  "revision_id": 2444599258,
  "value": [
    "Sherman Oaks Galleria"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Sherman Oaks Galleria"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Sherman Oaks Galleria"
    ],
    "new_value": [
      "Sherman Oaks Galleria"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Category:Sherman Oaks Galleria"
    ],
    "old_value": [
      "Category:Sherman Oaks Galleria"
    ],
    "removed_unique_values": [
      "Category:Sherman Oaks Galleria"
    ],
    "value_multiplicity_changes": {
      "Category:Sherman Oaks Galleria": {
        "new": 0,
        "old": 1
      },
      "Sherman Oaks Galleria": {
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
  "report_fix_date": "2025-12-22T10:34:48",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2445460508,
  "report_revision_old": 2444891710,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Commons link"
  ],
  "value": [
    "Category:Sherman Oaks Galleria"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 7,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "Sherman Oaks Galleria"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Category:Sherman Oaks Galleria"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Sherman Oaks Galleria"
  ],
  "truth_tokens_in_recorded_matches": [
    "Sherman Oaks Galleria"
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
    "description": "Shopping mall in Los Angeles, California, United States",
    "label": "Sherman Oaks Galleria"
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
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "Sherman Oaks Galleria"
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
        "Category:Sherman Oaks Galleria"
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

## 037. `repair_Q7886778_2447089920`

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
| group_key | ABOX::Q7886778::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Union stations"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_boundary |
| local_match_source | NEIGHBOR_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Union stations"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Union stations"
    ],
    "new_value": [
      "Union stations"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Union train stations"
    ],
    "old_value": [
      "Union train stations"
    ],
    "removed_unique_values": [
      "Union train stations"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_boundary",
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

## 038. `repair_Q7986873_2446818080`

| Field | Value |
|---|---|
| qid | Q7986873 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_CONFIRMED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_confirmed |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| group_key | ABOX::Q7986873::P225 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Nemopalpus"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Brya",
  "kind": "A_BOX",
  "new_value": [
    "Nemopalpus"
  ],
  "old_value": [
    "Nemapalpus"
  ],
  "revision_id": 2446818080,
  "value": [
    "Nemopalpus"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Nemopalpus"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Nemopalpus"
    ],
    "new_value": [
      "Nemopalpus"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Nemapalpus"
    ],
    "old_value": [
      "Nemapalpus"
    ],
    "removed_unique_values": [
      "Nemapalpus"
    ],
    "value_multiplicity_changes": {
      "Nemapalpus": {
        "new": 0,
        "old": 1
      },
      "Nemopalpus": {
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
  "report_fix_date": "2025-12-26T13:55:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2447398980,
  "report_revision_old": 2447090178,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Nemapalpus"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": true,
  "local_ids_count": 9,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
      "source": "FOCUS_LABEL",
      "token": "Nemopalpus"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_LABEL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Nemapalpus"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Nemopalpus"
  ],
  "truth_tokens_in_recorded_matches": [
    "Nemopalpus"
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
    "label": "Nemopalpus"
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
          "kind": "literal_exact",
          "source": "FOCUS_LABEL",
          "token": "Nemopalpus"
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
        "Nemapalpus"
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

## 039. `repair_Q8866784_2447090198`

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
| group_key | ABOX::Q8866784::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Union stations"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Union stations"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Union stations"
    ],
    "new_value": [
      "Union stations"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Union train stations"
    ],
    "old_value": [
      "Union train stations"
    ],
    "removed_unique_values": [
      "Union train stations"
    ],
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_boundary",
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

## 040. `repair_Q910283_2440849086`

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
| group_key | ABOX::Q910283::P373 |
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
| truth_token_kind | literal |
| truth_tokens_preview | ["Nippon Railway"] |
| decision_branch | local_match |
| rationale | Truth tokens matched independent local text context. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_LABEL |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "Nippon Railway"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Nippon Railway"
    ],
    "new_value": [
      "Nippon Railway"
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
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": true,
      "kind": "literal_exact",
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
          "kind": "literal_exact",
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
