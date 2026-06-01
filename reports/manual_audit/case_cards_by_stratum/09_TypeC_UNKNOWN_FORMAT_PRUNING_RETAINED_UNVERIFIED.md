# TypeC_UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED

Cases: 2

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q220090_2446235130`

| Field | Value |
|---|---|
| qid | Q220090 |
| property | P395 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_format_pruning_retained_unverified |
| decision_constraint_type |   |
| group_key | ABOX::Q220090::P395 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["M", "MU"] |
| classification_target_tokens | ["M, AIB, WOR"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | unknown_format_pruning_retained_unverified |
| rationale | Format subset pruning removed invalid-looking values, but retained values were not verified against the format regex. |
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
    "M, AIB, WOR"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "M, AIB, WOR"
  ],
  "removed_unique_values": [
    "M, AIB, WOR"
  ],
  "retained_support_tokens": [
    "M",
    "MU"
  ],
  "retained_unique_values": [
    "M",
    "MU"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_format_pruning_retained_unverified",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Michaelt1964",
  "kind": "A_BOX",
  "new_value": [
    "M",
    "MU"
  ],
  "old_value": [
    "M, AIB, WOR",
    "M",
    "MU"
  ],
  "revision_id": 2446235130,
  "value": [
    "M",
    "MU"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "M": 1,
      "MU": 1
    },
    "new_unique": [
      "M",
      "MU"
    ],
    "new_values": [
      "M",
      "MU"
    ],
    "new_values_raw": [
      "M",
      "MU"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "M": 1,
      "M, AIB, WOR": 1,
      "MU": 1
    },
    "old_unique": [
      "M",
      "M, AIB, WOR",
      "MU"
    ],
    "old_values": [
      "M, AIB, WOR",
      "M",
      "MU"
    ],
    "old_values_raw": [
      "M, AIB, WOR",
      "M",
      "MU"
    ],
    "removed_unique_values": [
      "M, AIB, WOR"
    ],
    "retained_unique_values": [
      "M",
      "MU"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "M, AIB, WOR": {
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
  "report_fix_date": "2025-12-25T19:18:04",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P395",
  "report_revision_new": 2447065856,
  "report_revision_old": 2446522457,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "M, AIB, WOR",
    "M",
    "MU"
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
    "M",
    "MU"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "distinguishing signs or parts of license plate associated with the subject. For countries: international licence plate country code or distinguishing sign of vehicles",
    "label": "licence plate code"
  },
  "qid": {
    "description": "municipality of Germany",
    "label": "Grünwald"
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
        "M": 1,
        "MU": 1
      },
      "new_unique": [
        "M",
        "MU"
      ],
      "new_values": [
        "M",
        "MU"
      ],
      "new_values_raw": [
        "M",
        "MU"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "M": 1,
        "M, AIB, WOR": 1,
        "MU": 1
      },
      "old_unique": [
        "M",
        "M, AIB, WOR",
        "MU"
      ],
      "old_values": [
        "M, AIB, WOR",
        "M",
        "MU"
      ],
      "old_values_raw": [
        "M, AIB, WOR",
        "M",
        "MU"
      ],
      "removed_unique_values": [
        "M, AIB, WOR"
      ],
      "retained_unique_values": [
        "M",
        "MU"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "M, AIB, WOR": {
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
      "reason": "retained_values_do_not_all_pass_format_regex",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "M, AIB, WOR"
      ],
      "report_values": [
        "M",
        "M, AIB, WOR",
        "MU"
      ],
      "retained_pass_regex": false,
      "retained_values": [
        "M",
        "MU"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
    "result": false,
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
    "result": "unknown_format_pruning_retained_unverified",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q50810375_2443313212`

| Field | Value |
|---|---|
| qid | Q50810375 |
| property | P2473 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_format_pruning_retained_unverified |
| decision_constraint_type |   |
| group_key | ABOX::Q50810375::P2473 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["03.093-9999-000042"] |
| classification_target_tokens | ["03.093-9999-42"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | unknown_format_pruning_retained_unverified |
| rationale | Format subset pruning removed invalid-looking values, but retained values were not verified against the format regex. |
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
    "03.093-9999-42"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "03.093-9999-42"
  ],
  "removed_unique_values": [
    "03.093-9999-42"
  ],
  "retained_support_tokens": [
    "03.093-9999-000042"
  ],
  "retained_unique_values": [
    "03.093-9999-000042"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_format_pruning_retained_unverified",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "B25es",
  "kind": "A_BOX",
  "new_value": [
    "03.093-9999-000042"
  ],
  "old_value": [
    "03.093-9999-42",
    "03.093-9999-000042"
  ],
  "revision_id": 2443313212,
  "value": [
    "03.093-9999-000042"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "03.093-9999-000042": 1
    },
    "new_unique": [
      "03.093-9999-000042"
    ],
    "new_values": [
      "03.093-9999-000042"
    ],
    "new_values_raw": [
      "03.093-9999-000042"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "03.093-9999-000042": 1,
      "03.093-9999-42": 1
    },
    "old_unique": [
      "03.093-9999-000042",
      "03.093-9999-42"
    ],
    "old_values": [
      "03.093-9999-42",
      "03.093-9999-000042"
    ],
    "old_values_raw": [
      "03.093-9999-42",
      "03.093-9999-000042"
    ],
    "removed_unique_values": [
      "03.093-9999-42"
    ],
    "retained_unique_values": [
      "03.093-9999-000042"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "03.093-9999-42": {
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
  "report_fix_date": "2025-12-19T08:58:40",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2473",
  "report_revision_new": 2444007257,
  "report_revision_old": 2443799780,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Item P|18"
  ],
  "value": [
    "03.093-9999-42",
    "03.093-9999-000042"
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
    "03.093-9999-000042"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "cultural heritage identifier in the Inventario General del Patrimonio Cultural Valenciano",
    "label": "IGPCV ID"
  },
  "qid": {
    "description": "building in Novelda (Alicante), Spain",
    "label": "Centro Cultural Gómez-Tortosa"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
        "03.093-9999-000042": 1
      },
      "new_unique": [
        "03.093-9999-000042"
      ],
      "new_values": [
        "03.093-9999-000042"
      ],
      "new_values_raw": [
        "03.093-9999-000042"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "03.093-9999-000042": 1,
        "03.093-9999-42": 1
      },
      "old_unique": [
        "03.093-9999-000042",
        "03.093-9999-42"
      ],
      "old_values": [
        "03.093-9999-42",
        "03.093-9999-000042"
      ],
      "old_values_raw": [
        "03.093-9999-42",
        "03.093-9999-000042"
      ],
      "removed_unique_values": [
        "03.093-9999-42"
      ],
      "retained_unique_values": [
        "03.093-9999-000042"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "03.093-9999-42": {
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
      "reason": "retained_values_do_not_all_pass_format_regex",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "03.093-9999-42"
      ],
      "report_values": [
        "03.093-9999-000042",
        "03.093-9999-42"
      ],
      "retained_pass_regex": false,
      "retained_values": [
        "03.093-9999-000042"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
    "result": false,
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
    "result": "unknown_format_pruning_retained_unverified",
    "step": "branch"
  }
]
```

---
