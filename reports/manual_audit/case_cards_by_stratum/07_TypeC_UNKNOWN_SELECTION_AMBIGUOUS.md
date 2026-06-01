# TypeC_UNKNOWN_SELECTION_AMBIGUOUS

Cases: 10

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q135499226_2388908540`

| Field | Value |
|---|---|
| qid | Q135499226 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_SELECTION_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_selection_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q135499226::P5236 |
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
| truth_tokens_preview | ["Q200", "Q203", "Q23350"] |
| classification_target_tokens | ["Q202"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | unknown_selection_ambiguous |
| rationale | Subset repair only shows retained values in the pre-repair target property; this is not independent local grounding. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q202"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q202"
  ],
  "removed_unique_values": [
    "Q202"
  ],
  "retained_support_tokens": [
    "Q200",
    "Q203",
    "Q23350"
  ],
  "retained_unique_values": [
    "Q200",
    "Q203",
    "Q23350"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_selection_ambiguous",
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
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q200",
    "Q203",
    "Q23350"
  ],
  "new_value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number"
  ],
  "new_value_labels_en": [
    "ҩба",
    "5",
    "7"
  ],
  "old_value": [
    "Q200",
    "Q202",
    "Q203",
    "Q23350"
  ],
  "old_value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number",
    "natural number"
  ],
  "old_value_labels_en": [
    "ҩба",
    "4",
    "5",
    "7"
  ],
  "revision_id": 2388908540,
  "value": [
    "Q200",
    "Q203",
    "Q23350"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q200": 1,
      "Q203": 1,
      "Q23350": 1
    },
    "new_unique": [
      "Q200",
      "Q203",
      "Q23350"
    ],
    "new_values": [
      "Q200",
      "Q203",
      "Q23350"
    ],
    "new_values_raw": [
      "Q200",
      "Q203",
      "Q23350"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q200": 1,
      "Q202": 1,
      "Q203": 1,
      "Q23350": 1
    },
    "old_unique": [
      "Q200",
      "Q202",
      "Q203",
      "Q23350"
    ],
    "old_values": [
      "Q200",
      "Q202",
      "Q203",
      "Q23350"
    ],
    "old_values_raw": [
      "Q200",
      "Q202",
      "Q203",
      "Q23350"
    ],
    "removed_unique_values": [
      "Q202"
    ],
    "retained_unique_values": [
      "Q200",
      "Q203",
      "Q23350"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q202": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number"
  ],
  "value_labels_en": [
    "ҩба",
    "5",
    "7"
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
    "Q200",
    "Q202",
    "Q203",
    "Q23350"
  ],
  "value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number",
    "natural number"
  ],
  "value_labels_en": [
    "ҩба",
    "4",
    "5",
    "7"
  ]
}
```

### Local Evidence

```json
{
  "found": 3,
  "local_availability_result": false,
  "local_ids_count": 19,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q200"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q203"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q23350"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q200"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q203"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q23350"
    }
  ],
  "needed": 3,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q200",
      "Q202",
      "Q203",
      "Q23350"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q200",
    "Q203",
    "Q23350"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q200",
    "Q203",
    "Q23350"
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
    "label": "11200"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q200": 1,
        "Q203": 1,
        "Q23350": 1
      },
      "new_unique": [
        "Q200",
        "Q203",
        "Q23350"
      ],
      "new_values": [
        "Q200",
        "Q203",
        "Q23350"
      ],
      "new_values_raw": [
        "Q200",
        "Q203",
        "Q23350"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q200": 1,
        "Q202": 1,
        "Q203": 1,
        "Q23350": 1
      },
      "old_unique": [
        "Q200",
        "Q202",
        "Q203",
        "Q23350"
      ],
      "old_values": [
        "Q200",
        "Q202",
        "Q203",
        "Q23350"
      ],
      "old_values_raw": [
        "Q200",
        "Q202",
        "Q203",
        "Q23350"
      ],
      "removed_unique_values": [
        "Q202"
      ],
      "retained_unique_values": [
        "Q200",
        "Q203",
        "Q23350"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q202": {
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
      "found": 3,
      "independent_match_count": 0,
      "local_ids_count": 19,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q200"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q203"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q23350"
        }
      ],
      "needed": 3,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q200",
        "Q202",
        "Q203",
        "Q23350"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_selection_ambiguous",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q136232073_2405255069`

| Field | Value |
|---|---|
| qid | Q136232073 |
| property | P1006 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_SELECTION_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_selection_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q136232073::P1006 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["328663352"] |
| classification_target_tokens | ["363033033"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | unknown_selection_ambiguous |
| rationale | Subset repair only shows retained values in the pre-repair target property; this is not independent local grounding. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "363033033"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "363033033"
  ],
  "removed_unique_values": [
    "363033033"
  ],
  "retained_support_tokens": [
    "328663352"
  ],
  "retained_unique_values": [
    "328663352"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_selection_ambiguous",
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
  "author": "Thomas Kerboul (BGE)",
  "kind": "A_BOX",
  "new_value": [
    "328663352"
  ],
  "old_value": [
    "328663352",
    "363033033"
  ],
  "revision_id": 2405255069,
  "value": [
    "328663352"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "328663352": 1
    },
    "new_unique": [
      "328663352"
    ],
    "new_values": [
      "328663352"
    ],
    "new_values_raw": [
      "328663352"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "328663352": 1,
      "363033033": 1
    },
    "old_unique": [
      "328663352",
      "363033033"
    ],
    "old_values": [
      "328663352",
      "363033033"
    ],
    "old_values_raw": [
      "328663352",
      "363033033"
    ],
    "removed_unique_values": [
      "363033033"
    ],
    "retained_unique_values": [
      "328663352"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "363033033": {
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
  "report_fix_date": "2025-09-19T22:59:13",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1006",
  "report_revision_new": 2407507647,
  "report_revision_old": 2407109254,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "328663352",
    "363033033"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 17,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "328663352",
      "raw_match_text": "328663352",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "328663352"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "328663352",
      "raw_match_text": "328663352",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "328663352"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "328663352",
      "363033033"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "328663352"
  ],
  "truth_tokens_in_recorded_matches": [
    "328663352"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for person names (not: works nor organisations) from the Dutch National Thesaurus for Author names (which also contains non-authors)",
    "label": "Nationale Thesaurus voor Auteursnamen ID"
  },
  "qid": {
    "description": "geographer and university professor",
    "label": "Noukpo Agossou"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
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
        "328663352": 1
      },
      "new_unique": [
        "328663352"
      ],
      "new_values": [
        "328663352"
      ],
      "new_values_raw": [
        "328663352"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "328663352": 1,
        "363033033": 1
      },
      "old_unique": [
        "328663352",
        "363033033"
      ],
      "old_values": [
        "328663352",
        "363033033"
      ],
      "old_values_raw": [
        "328663352",
        "363033033"
      ],
      "removed_unique_values": [
        "363033033"
      ],
      "retained_unique_values": [
        "328663352"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "363033033": {
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
      "independent_match_count": 0,
      "local_ids_count": 17,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_exact_raw",
          "normalized_match_text": "328663352",
          "raw_match_text": "328663352",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "328663352"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "328663352",
        "363033033"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_selection_ambiguous",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q136750874_2429986390`

| Field | Value |
|---|---|
| qid | Q136750874 |
| property | P662 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_SELECTION_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_selection_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q136750874::P662 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["131954322", "135627718", "154723801", "154725694", "172677304"] |
| classification_target_tokens | ["154723800"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | unknown_selection_ambiguous |
| rationale | Subset repair only shows retained values in the pre-repair target property; this is not independent local grounding. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "154723800"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "154723800"
  ],
  "removed_unique_values": [
    "154723800"
  ],
  "retained_support_tokens": [
    "131954322",
    "135627718",
    "154723801",
    "154725694",
    "172677304"
  ],
  "retained_unique_values": [
    "131954322",
    "135627718",
    "154723801",
    "154725694",
    "172677304"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_selection_ambiguous",
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
  "author": "Egon Willighagen",
  "kind": "A_BOX",
  "new_value": [
    "131954322",
    "135627718",
    "154723801",
    "154725694",
    "172677304"
  ],
  "old_value": [
    "131954322",
    "135627718",
    "154723801",
    "154725694",
    "172677304",
    "154723800"
  ],
  "revision_id": 2429986390,
  "value": [
    "131954322",
    "135627718",
    "154723801",
    "154725694",
    "172677304"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "131954322": 1,
      "135627718": 1,
      "154723801": 1,
      "154725694": 1,
      "172677304": 1
    },
    "new_unique": [
      "131954322",
      "135627718",
      "154723801",
      "154725694",
      "172677304"
    ],
    "new_values": [
      "131954322",
      "135627718",
      "154723801",
      "154725694",
      "172677304"
    ],
    "new_values_raw": [
      "131954322",
      "135627718",
      "154723801",
      "154725694",
      "172677304"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "131954322": 1,
      "135627718": 1,
      "154723800": 1,
      "154723801": 1,
      "154725694": 1,
      "172677304": 1
    },
    "old_unique": [
      "131954322",
      "135627718",
      "154723800",
      "154723801",
      "154725694",
      "172677304"
    ],
    "old_values": [
      "131954322",
      "135627718",
      "154723801",
      "154725694",
      "172677304",
      "154723800"
    ],
    "old_values_raw": [
      "131954322",
      "135627718",
      "154723801",
      "154725694",
      "172677304",
      "154723800"
    ],
    "removed_unique_values": [
      "154723800"
    ],
    "retained_unique_values": [
      "131954322",
      "135627718",
      "154723801",
      "154725694",
      "172677304"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "154723800": {
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
  "report_fix_date": "2025-11-17T10:02:49",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P662",
  "report_revision_new": 2430653855,
  "report_revision_old": 2430258636,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "131954322",
    "135627718",
    "154723801",
    "154725694",
    "172677304",
    "154723800"
  ]
}
```

### Local Evidence

```json
{
  "found": 5,
  "local_availability_result": false,
  "local_ids_count": 3,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "131954322",
      "raw_match_text": "131954322",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "131954322"
    },
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "135627718",
      "raw_match_text": "135627718",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "135627718"
    },
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "154723801",
      "raw_match_text": "154723801",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "154723801"
    },
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "154725694",
      "raw_match_text": "154725694",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "154725694"
    },
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "172677304",
      "raw_match_text": "172677304",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "172677304"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "131954322",
      "raw_match_text": "131954322",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "131954322"
    },
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "135627718",
      "raw_match_text": "135627718",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "135627718"
    },
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "154723801",
      "raw_match_text": "154723801",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "154723801"
    },
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "154725694",
      "raw_match_text": "154725694",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "154725694"
    },
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "172677304",
      "raw_match_text": "172677304",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "172677304"
    }
  ],
  "needed": 5,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "131954322",
      "135627718",
      "154723801",
      "154725694",
      "172677304",
      "154723800"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "131954322",
    "135627718",
    "154723801",
    "154725694",
    "172677304"
  ],
  "truth_tokens_in_recorded_matches": [
    "131954322",
    "135627718",
    "154723801",
    "154725694",
    "172677304"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier from database of chemical molecules and their activities in biological assays (Compound ID number)",
    "label": "PubChem CID"
  },
  "qid": {
    "description": null,
    "label": "2-[2-[3-[2-(1,3-Dihydro-1,3,3-trimethyl-2H-indol-2-ylidene)ethylidene]-1-cyclohexen-1-yl]ethenyl]-3,3-dimethyl-1-[6-oxo-6-(2-propyn-1-ylamino)hexyl]-3H-indolium"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
        "131954322": 1,
        "135627718": 1,
        "154723801": 1,
        "154725694": 1,
        "172677304": 1
      },
      "new_unique": [
        "131954322",
        "135627718",
        "154723801",
        "154725694",
        "172677304"
      ],
      "new_values": [
        "131954322",
        "135627718",
        "154723801",
        "154725694",
        "172677304"
      ],
      "new_values_raw": [
        "131954322",
        "135627718",
        "154723801",
        "154725694",
        "172677304"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "131954322": 1,
        "135627718": 1,
        "154723800": 1,
        "154723801": 1,
        "154725694": 1,
        "172677304": 1
      },
      "old_unique": [
        "131954322",
        "135627718",
        "154723800",
        "154723801",
        "154725694",
        "172677304"
      ],
      "old_values": [
        "131954322",
        "135627718",
        "154723801",
        "154725694",
        "172677304",
        "154723800"
      ],
      "old_values_raw": [
        "131954322",
        "135627718",
        "154723801",
        "154725694",
        "172677304",
        "154723800"
      ],
      "removed_unique_values": [
        "154723800"
      ],
      "retained_unique_values": [
        "131954322",
        "135627718",
        "154723801",
        "154725694",
        "172677304"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "154723800": {
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
      "found": 5,
      "independent_match_count": 0,
      "local_ids_count": 3,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_exact_raw",
          "normalized_match_text": "131954322",
          "raw_match_text": "131954322",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "131954322"
        },
        {
          "independent_of_target_property": false,
          "kind": "literal_exact_raw",
          "normalized_match_text": "135627718",
          "raw_match_text": "135627718",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "135627718"
        },
        {
          "independent_of_target_property": false,
          "kind": "literal_exact_raw",
          "normalized_match_text": "154723801",
          "raw_match_text": "154723801",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "154723801"
        },
        {
          "independent_of_target_property": false,
          "kind": "literal_exact_raw",
          "normalized_match_text": "154725694",
          "raw_match_text": "154725694",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "154725694"
        },
        {
          "independent_of_target_property": false,
          "kind": "literal_exact_raw",
          "normalized_match_text": "172677304",
          "raw_match_text": "172677304",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "172677304"
        }
      ],
      "needed": 5,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "131954322",
        "135627718",
        "154723801",
        "154725694",
        "172677304",
        "154723800"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_selection_ambiguous",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q26293016_2444319553`

| Field | Value |
|---|---|
| qid | Q26293016 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_SELECTION_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_selection_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q26293016::P225 |
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
| truth_tokens_preview | ["Streptomyces thermospinisporus"] |
| classification_target_tokens | ["Streptomyces thermospinosisporus"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | unknown_selection_ambiguous |
| rationale | Subset repair only shows retained values in the pre-repair target property; this is not independent local grounding. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Streptomyces thermospinosisporus"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Streptomyces thermospinosisporus"
  ],
  "removed_unique_values": [
    "Streptomyces thermospinosisporus"
  ],
  "retained_support_tokens": [
    "Streptomyces thermospinisporus"
  ],
  "retained_unique_values": [
    "Streptomyces thermospinisporus"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_selection_ambiguous",
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
  "author": "Brya",
  "kind": "A_BOX",
  "new_value": [
    "Streptomyces thermospinisporus"
  ],
  "old_value": [
    "Streptomyces thermospinosisporus",
    "Streptomyces thermospinisporus"
  ],
  "revision_id": 2444319553,
  "value": [
    "Streptomyces thermospinisporus"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Streptomyces thermospinisporus": 1
    },
    "new_unique": [
      "Streptomyces thermospinisporus"
    ],
    "new_values": [
      "Streptomyces thermospinisporus"
    ],
    "new_values_raw": [
      "Streptomyces thermospinisporus"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Streptomyces thermospinisporus": 1,
      "Streptomyces thermospinosisporus": 1
    },
    "old_unique": [
      "Streptomyces thermospinisporus",
      "Streptomyces thermospinosisporus"
    ],
    "old_values": [
      "Streptomyces thermospinosisporus",
      "Streptomyces thermospinisporus"
    ],
    "old_values_raw": [
      "Streptomyces thermospinosisporus",
      "Streptomyces thermospinisporus"
    ],
    "removed_unique_values": [
      "Streptomyces thermospinosisporus"
    ],
    "retained_unique_values": [
      "Streptomyces thermospinisporus"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Streptomyces thermospinosisporus": {
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
    "Streptomyces thermospinosisporus",
    "Streptomyces thermospinisporus"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 9,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "streptomyces thermospinisporus",
      "raw_match_text": "Streptomyces thermospinisporus",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "Streptomyces thermospinisporus"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "streptomyces thermospinisporus",
      "raw_match_text": "Streptomyces thermospinisporus",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "Streptomyces thermospinisporus"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Streptomyces thermospinosisporus",
      "Streptomyces thermospinisporus"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Streptomyces thermospinisporus"
  ],
  "truth_tokens_in_recorded_matches": [
    "Streptomyces thermospinisporus"
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
    "description": "species of Actinobacteria",
    "label": "Streptomyces thermospinosisporus"
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
        "Streptomyces thermospinisporus": 1
      },
      "new_unique": [
        "Streptomyces thermospinisporus"
      ],
      "new_values": [
        "Streptomyces thermospinisporus"
      ],
      "new_values_raw": [
        "Streptomyces thermospinisporus"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Streptomyces thermospinisporus": 1,
        "Streptomyces thermospinosisporus": 1
      },
      "old_unique": [
        "Streptomyces thermospinisporus",
        "Streptomyces thermospinosisporus"
      ],
      "old_values": [
        "Streptomyces thermospinosisporus",
        "Streptomyces thermospinisporus"
      ],
      "old_values_raw": [
        "Streptomyces thermospinosisporus",
        "Streptomyces thermospinisporus"
      ],
      "removed_unique_values": [
        "Streptomyces thermospinosisporus"
      ],
      "retained_unique_values": [
        "Streptomyces thermospinisporus"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Streptomyces thermospinosisporus": {
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
      "independent_match_count": 0,
      "local_ids_count": 9,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_exact_raw",
          "normalized_match_text": "streptomyces thermospinisporus",
          "raw_match_text": "Streptomyces thermospinisporus",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "Streptomyces thermospinisporus"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Streptomyces thermospinosisporus",
        "Streptomyces thermospinisporus"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_selection_ambiguous",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q3859_2444563007`

| Field | Value |
|---|---|
| qid | Q3859 |
| property | P856 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_SELECTION_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_selection_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q3859::P856 |
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
| truth_tokens_preview | ["https://www.kigalicity.gov.rw/"] |
| classification_target_tokens | ["http://www.kigalicity.gov.rw/"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | unknown_selection_ambiguous |
| rationale | Subset repair only shows retained values in the pre-repair target property; this is not independent local grounding. |
| local_match_kind | literal_exact_raw |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "http://www.kigalicity.gov.rw/"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "http://www.kigalicity.gov.rw/"
  ],
  "removed_unique_values": [
    "http://www.kigalicity.gov.rw/"
  ],
  "retained_support_tokens": [
    "https://www.kigalicity.gov.rw/"
  ],
  "retained_unique_values": [
    "https://www.kigalicity.gov.rw/"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_selection_ambiguous",
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
  "author": "Irifasum",
  "kind": "A_BOX",
  "new_value": [
    "https://www.kigalicity.gov.rw/"
  ],
  "old_value": [
    "https://www.kigalicity.gov.rw/",
    "http://www.kigalicity.gov.rw/"
  ],
  "revision_id": 2444563007,
  "value": [
    "https://www.kigalicity.gov.rw/"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "https://www.kigalicity.gov.rw/": 1
    },
    "new_unique": [
      "https://www.kigalicity.gov.rw/"
    ],
    "new_values": [
      "https://www.kigalicity.gov.rw/"
    ],
    "new_values_raw": [
      "https://www.kigalicity.gov.rw/"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "http://www.kigalicity.gov.rw/": 1,
      "https://www.kigalicity.gov.rw/": 1
    },
    "old_unique": [
      "http://www.kigalicity.gov.rw/",
      "https://www.kigalicity.gov.rw/"
    ],
    "old_values": [
      "https://www.kigalicity.gov.rw/",
      "http://www.kigalicity.gov.rw/"
    ],
    "old_values_raw": [
      "https://www.kigalicity.gov.rw/",
      "http://www.kigalicity.gov.rw/"
    ],
    "removed_unique_values": [
      "http://www.kigalicity.gov.rw/"
    ],
    "retained_unique_values": [
      "https://www.kigalicity.gov.rw/"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "http://www.kigalicity.gov.rw/": {
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
  "report_fix_date": "2025-12-22T09:19:07",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P856",
  "report_revision_new": 2445433685,
  "report_revision_old": 2444870679,
  "report_violation_type": "Mandatory Qualifiers",
  "report_violation_type_normalized": "Mandatory Qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Mandatory Qualifiers",
  "report_violation_types": [
    "Mandatory Qualifiers",
    "Single value"
  ],
  "value": [
    "https://www.kigalicity.gov.rw/",
    "http://www.kigalicity.gov.rw/"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 105,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "https://www.kigalicity.gov.rw/",
      "raw_match_text": "https://www.kigalicity.gov.rw/",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "https://www.kigalicity.gov.rw/"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_exact_raw",
      "normalized_match_text": "https://www.kigalicity.gov.rw/",
      "raw_match_text": "https://www.kigalicity.gov.rw/",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "https://www.kigalicity.gov.rw/"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "https://www.kigalicity.gov.rw/",
      "http://www.kigalicity.gov.rw/"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "https://www.kigalicity.gov.rw/"
  ],
  "truth_tokens_in_recorded_matches": [
    "https://www.kigalicity.gov.rw/"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "URL of the official page of an item (current or former). Usage: If a listed URL no longer points to the official website, do not remove it, but see the \"Hijacked or dead websites\" section of the Talk page",
    "label": "official website"
  },
  "qid": {
    "description": "province-level city, capital and largest city of Rwanda",
    "label": "Kigali"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
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
        "https://www.kigalicity.gov.rw/": 1
      },
      "new_unique": [
        "https://www.kigalicity.gov.rw/"
      ],
      "new_values": [
        "https://www.kigalicity.gov.rw/"
      ],
      "new_values_raw": [
        "https://www.kigalicity.gov.rw/"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "http://www.kigalicity.gov.rw/": 1,
        "https://www.kigalicity.gov.rw/": 1
      },
      "old_unique": [
        "http://www.kigalicity.gov.rw/",
        "https://www.kigalicity.gov.rw/"
      ],
      "old_values": [
        "https://www.kigalicity.gov.rw/",
        "http://www.kigalicity.gov.rw/"
      ],
      "old_values_raw": [
        "https://www.kigalicity.gov.rw/",
        "http://www.kigalicity.gov.rw/"
      ],
      "removed_unique_values": [
        "http://www.kigalicity.gov.rw/"
      ],
      "retained_unique_values": [
        "https://www.kigalicity.gov.rw/"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "http://www.kigalicity.gov.rw/": {
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
      "independent_match_count": 0,
      "local_ids_count": 105,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_exact_raw",
          "normalized_match_text": "https://www.kigalicity.gov.rw/",
          "raw_match_text": "https://www.kigalicity.gov.rw/",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "https://www.kigalicity.gov.rw/"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "https://www.kigalicity.gov.rw/",
        "http://www.kigalicity.gov.rw/"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_selection_ambiguous",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q53896845_2445336866`

| Field | Value |
|---|---|
| qid | Q53896845 |
| property | P50 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_SELECTION_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_selection_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q53896845::P50 |
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
| truth_tokens_preview | ["Q61516239"] |
| classification_target_tokens | ["Q67908402"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | unknown_selection_ambiguous |
| rationale | Subset repair only shows retained values in the pre-repair target property; this is not independent local grounding. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q67908402"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q67908402"
  ],
  "removed_unique_values": [
    "Q67908402"
  ],
  "retained_support_tokens": [
    "Q61516239"
  ],
  "retained_unique_values": [
    "Q61516239"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_selection_ambiguous",
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
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Q61516239",
    "Q61516239"
  ],
  "new_value_descriptions_en": [
    "researcher ORCID ID = 0000-0001-9196-8516",
    "researcher ORCID ID = 0000-0001-9196-8516"
  ],
  "new_value_labels_en": [
    "James G Hamilton",
    "James G Hamilton"
  ],
  "old_value": [
    "Q61516239",
    "Q67908402"
  ],
  "old_value_descriptions_en": [
    "researcher ORCID ID = 0000-0001-9196-8516",
    "researcher ORCID ID = 0000-0001-9196-8516"
  ],
  "old_value_labels_en": [
    "James G Hamilton",
    "James G Hamilton"
  ],
  "revision_id": 2445336866,
  "value": [
    "Q61516239",
    "Q61516239"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q61516239": 2
    },
    "new_unique": [
      "Q61516239"
    ],
    "new_values": [
      "Q61516239",
      "Q61516239"
    ],
    "new_values_raw": [
      "Q61516239",
      "Q61516239"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q61516239": 1,
      "Q67908402": 1
    },
    "old_unique": [
      "Q61516239",
      "Q67908402"
    ],
    "old_values": [
      "Q61516239",
      "Q67908402"
    ],
    "old_values_raw": [
      "Q61516239",
      "Q67908402"
    ],
    "removed_unique_values": [
      "Q67908402"
    ],
    "retained_unique_values": [
      "Q61516239"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q61516239": {
        "new": 2,
        "old": 1
      },
      "Q67908402": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "researcher ORCID ID = 0000-0001-9196-8516",
    "researcher ORCID ID = 0000-0001-9196-8516"
  ],
  "value_labels_en": [
    "James G Hamilton",
    "James G Hamilton"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-23T16:42:06",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P50",
  "report_revision_new": 2446124069,
  "report_revision_old": 2445516136,
  "report_violation_type": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "report_violation_type_descriptions_en": [
    "being that has certain capacities or attributes constituting personhood (for humans, use Q5 [human] with P31 [instance of])",
    "fictitious name that a person or group assumes for a particular purpose, which differs from their original or true name (orthonym)",
    "social entity established to meet needs or pursue goals",
    "any set of human beings",
    "fictional human or non-human character in a narrative work of art",
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "field of computer science that develops and studies software enabling machines to exhibit intelligent behavior",
    "human being whose existence is not directly attested, but is deduced by other evidence",
    "intelligent actor with unnatural origin",
    "distinct and identifiable entity with agency, capable of performing actions",
    "copyright is owned or retained by the party that commissioned it or by the employer of the person who produced it",
    null
  ],
  "report_violation_type_labels_en": [
    "person",
    "pseudonym",
    "organization",
    "group of humans",
    "character",
    "human",
    "artificial intelligence",
    "hypothetical person",
    "artificially intelligent entity",
    "being",
    "corporate authorship",
    "corporate author"
  ],
  "report_violation_type_normalized": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "report_violation_type_qids": [
    "Q215627",
    "Q61002",
    "Q43229",
    "Q16334295",
    "Q95074",
    "Q5",
    "Q11660",
    "Q75855169",
    "Q107307291",
    "Q24229398",
    "Q78056559",
    "Q60557912"
  ],
  "report_violation_type_raw": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "value": [
    "Q61516239",
    "Q67908402"
  ],
  "value_descriptions_en": [
    "researcher ORCID ID = 0000-0001-9196-8516",
    "researcher ORCID ID = 0000-0001-9196-8516"
  ],
  "value_labels_en": [
    "James G Hamilton",
    "James G Hamilton"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 21,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q61516239"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q61516239"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q61516239",
      "Q67908402"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q61516239"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q61516239"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "main creator(s) of a written work (use on works, not humans); use P2093 (author name string) when Wikidata item is unknown or does not exist",
    "label": "author"
  },
  "qid": {
    "description": "scientific article",
    "label": "Multichemical defense of plant bugHotea gambiae (Westwood) (Heteroptera: Scutelleridae): (E)-2-hexenol from abdominal gland in adults"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q61516239": 2
      },
      "new_unique": [
        "Q61516239"
      ],
      "new_values": [
        "Q61516239",
        "Q61516239"
      ],
      "new_values_raw": [
        "Q61516239",
        "Q61516239"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q61516239": 1,
        "Q67908402": 1
      },
      "old_unique": [
        "Q61516239",
        "Q67908402"
      ],
      "old_values": [
        "Q61516239",
        "Q67908402"
      ],
      "old_values_raw": [
        "Q61516239",
        "Q67908402"
      ],
      "removed_unique_values": [
        "Q67908402"
      ],
      "retained_unique_values": [
        "Q61516239"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q61516239": {
          "new": 2,
          "old": 1
        },
        "Q67908402": {
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
      "independent_match_count": 0,
      "local_ids_count": 21,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q61516239"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q61516239",
        "Q67908402"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_selection_ambiguous",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q65775814_2446541662`

| Field | Value |
|---|---|
| qid | Q65775814 |
| property | P580 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_SELECTION_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q52060874 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_selection_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q65775814::P580 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | date |
| truth_tokens_preview | ["+1988-06-17T00:00:00Z"] |
| classification_target_tokens | ["+1965-06-19T00:00:00Z"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | unknown_selection_ambiguous |
| rationale | Subset repair only shows retained values in the pre-repair target property; this is not independent local grounding. |
| local_match_kind | date_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "+1965-06-19T00:00:00Z"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "+1965-06-19T00:00:00Z"
  ],
  "removed_unique_values": [
    "+1965-06-19T00:00:00Z"
  ],
  "retained_support_tokens": [
    "+1988-06-17T00:00:00Z"
  ],
  "retained_unique_values": [
    "+1988-06-17T00:00:00Z"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_selection_ambiguous",
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
  "author": "Florentyna",
  "kind": "A_BOX",
  "new_value": [
    "+1988-06-17T00:00:00Z"
  ],
  "old_value": [
    "+1988-06-17T00:00:00Z",
    "+1965-06-19T00:00:00Z"
  ],
  "revision_id": 2446541662,
  "value": [
    "+1988-06-17T00:00:00Z"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "+1988-06-17T00:00:00Z": 1
    },
    "new_unique": [
      "+1988-06-17T00:00:00Z"
    ],
    "new_values": [
      "+1988-06-17T00:00:00Z"
    ],
    "new_values_raw": [
      "+1988-06-17T00:00:00Z"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "+1965-06-19T00:00:00Z": 1,
      "+1988-06-17T00:00:00Z": 1
    },
    "old_unique": [
      "+1965-06-19T00:00:00Z",
      "+1988-06-17T00:00:00Z"
    ],
    "old_values": [
      "+1988-06-17T00:00:00Z",
      "+1965-06-19T00:00:00Z"
    ],
    "old_values_raw": [
      "+1988-06-17T00:00:00Z",
      "+1965-06-19T00:00:00Z"
    ],
    "removed_unique_values": [
      "+1965-06-19T00:00:00Z"
    ],
    "retained_unique_values": [
      "+1988-06-17T00:00:00Z"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "+1965-06-19T00:00:00Z": {
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
  "report_fix_date": "2025-12-26T11:58:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P580",
  "report_revision_new": 2447358423,
  "report_revision_old": 2447052294,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "+1988-06-17T00:00:00Z",
    "+1965-06-19T00:00:00Z"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 19,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": false,
      "kind": "date_exact",
      "normalized_match_text": "1988-06-17t00:00:00z",
      "raw_match_text": "+1988-06-17T00:00:00Z",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "+1988-06-17T00:00:00Z"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "date_exact",
      "normalized_match_text": "1988-06-17t00:00:00z",
      "raw_match_text": "+1988-06-17T00:00:00Z",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "+1988-06-17T00:00:00Z"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "+1988-06-17T00:00:00Z",
      "+1965-06-19T00:00:00Z"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "+1988-06-17T00:00:00Z"
  ],
  "truth_tokens_in_recorded_matches": [
    "+1988-06-17T00:00:00Z"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "time an entity begins to exist or a statement starts being valid",
    "label": "start time"
  },
  "qid": {
    "description": "badminton championships",
    "label": "1987/1988 German Students' Badminton Championships – teams"
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
    "result": false,
    "step": "is_delete"
  },
  {
    "detail": {
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "+1988-06-17T00:00:00Z": 1
      },
      "new_unique": [
        "+1988-06-17T00:00:00Z"
      ],
      "new_values": [
        "+1988-06-17T00:00:00Z"
      ],
      "new_values_raw": [
        "+1988-06-17T00:00:00Z"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "+1965-06-19T00:00:00Z": 1,
        "+1988-06-17T00:00:00Z": 1
      },
      "old_unique": [
        "+1965-06-19T00:00:00Z",
        "+1988-06-17T00:00:00Z"
      ],
      "old_values": [
        "+1988-06-17T00:00:00Z",
        "+1965-06-19T00:00:00Z"
      ],
      "old_values_raw": [
        "+1988-06-17T00:00:00Z",
        "+1965-06-19T00:00:00Z"
      ],
      "removed_unique_values": [
        "+1965-06-19T00:00:00Z"
      ],
      "retained_unique_values": [
        "+1988-06-17T00:00:00Z"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "+1965-06-19T00:00:00Z": {
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
      "independent_match_count": 0,
      "local_ids_count": 19,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "date_exact",
          "normalized_match_text": "1988-06-17t00:00:00z",
          "raw_match_text": "+1988-06-17T00:00:00Z",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "+1988-06-17T00:00:00Z"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "+1988-06-17T00:00:00Z",
        "+1965-06-19T00:00:00Z"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_selection_ambiguous",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q7783986_1580936803`

| Field | Value |
|---|---|
| qid | Q7783986 |
| property | P1346 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_SELECTION_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_selection_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q7783986::P1346 |
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
| truth_tokens_preview | ["Q22277554", "Q108871755", "Q1421302", "Q16197959", "Q55687800", "...(+16)"] |
| classification_target_tokens | ["Q110864939"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | unknown_selection_ambiguous |
| rationale | Subset repair only shows retained values in the pre-repair target property; this is not independent local grounding. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q110864939"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q110864939"
  ],
  "removed_unique_values": [
    "Q110864939"
  ],
  "retained_support_tokens": [
    "Q108871755",
    "Q110864324",
    "Q110864359",
    "Q110864575",
    "Q110864632",
    "Q110864757",
    "Q110865581",
    "Q110865641",
    "Q110865711",
    "Q110865792",
    "Q110865872",
    "Q1421302",
    "Q16197959",
    "Q16234559",
    "Q18324728",
    "Q22277554",
    "Q25095457",
    "Q55687800",
    "Q58494969",
    "Q61070629",
    "Q66686322"
  ],
  "retained_unique_values": [
    "Q108871755",
    "Q110864324",
    "Q110864359",
    "Q110864575",
    "Q110864632",
    "Q110864757",
    "Q110865581",
    "Q110865641",
    "Q110865711",
    "Q110865792",
    "Q110865872",
    "Q1421302",
    "Q16197959",
    "Q16234559",
    "Q18324728",
    "Q22277554",
    "Q25095457",
    "Q55687800",
    "Q58494969",
    "Q61070629",
    "Q66686322"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_selection_ambiguous",
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
  "author": "Lymantria",
  "kind": "A_BOX",
  "new_value": [
    "Q22277554",
    "Q108871755",
    "Q1421302",
    "Q16197959",
    "Q55687800",
    "Q61070629",
    "Q25095457",
    "Q16234559",
    "Q58494969",
    "Q18324728",
    "Q66686322",
    "Q110864757",
    "Q110865581",
    "Q110865641",
    "Q110865711",
    "Q110865792",
    "Q110865872",
    "Q110864632",
    "Q110864359",
    "Q110864575",
    "Q110864324"
  ],
  "new_value_descriptions_en": [
    "Venture capitalist",
    "American entrepreneur",
    "American student, scientist",
    "Russian-Canadian computer scientist",
    "American businessperson",
    "Indian entrepreneur",
    "Canadian entrepreneur",
    "American entrepreneur, speaker and author",
    "Canadian entrepreneur",
    "Dutch inventor and environmentalist",
    "American entrepreneur (born 1999)",
    "Founder at Paladin Drones",
    "Founder and CEO at ExpressionMed",
    "Founder of Docbot",
    "Founder of Ramp USA",
    "Cofounder of WindBorne Systems",
    "Co-founder, CEO at Compocket",
    "CEO at Epic Aerospace",
    "Founder of Vitable Health",
    "Co-founder of Dandy",
    "Cofounder and CTO of Dreambound"
  ],
  "new_value_labels_en": [
    "Laura Deming",
    "Dylan Field",
    "Taylor Wilson",
    "Vitalik Buterin",
    "Austin Russell",
    "Ritesh Agarwal",
    "Simon Tian",
    "Stacey Ferreira",
    "Cathy Tie",
    "Boyan Slat",
    "Erin Smith",
    "Divyaditya Shrivastava",
    "Meghan Sharkus",
    "Andrew Ninh",
    "Melvin Du",
    "Paige Brown",
    "Ilayda Buyukdogan",
    "Ignacio Belieres Montero",
    "Joseph Kitonga",
    "Toni Oloko",
    "Brandon Wang"
  ],
  "old_value": [
    "Q22277554",
    "Q108871755",
    "Q1421302",
    "Q16197959",
    "Q55687800",
    "Q61070629",
    "Q25095457",
    "Q16234559",
    "Q58494969",
    "Q18324728",
    "Q66686322",
    "Q110864757",
    "Q110865581",
    "Q110865641",
    "Q110865711",
    "Q110865792",
    "Q110865872",
    "Q110864632",
    "Q110864359",
    "Q110864575",
    "Q110864939",
    "Q110864324"
  ],
  "old_value_descriptions_en": [
    "Venture capitalist",
    "American entrepreneur",
    "American student, scientist",
    "Russian-Canadian computer scientist",
    "American businessperson",
    "Indian entrepreneur",
    "Canadian entrepreneur",
    "American entrepreneur, speaker and author",
    "Canadian entrepreneur",
    "Dutch inventor and environmentalist",
    "American entrepreneur (born 1999)",
    "Founder at Paladin Drones",
    "Founder and CEO at ExpressionMed",
    "Founder of Docbot",
    "Founder of Ramp USA",
    "Cofounder of WindBorne Systems",
    "Co-founder, CEO at Compocket",
    "CEO at Epic Aerospace",
    "Founder of Vitable Health",
    "Co-founder of Dandy",
    null,
    "Cofounder and CTO of Dreambound"
  ],
  "old_value_labels_en": [
    "Laura Deming",
    "Dylan Field",
    "Taylor Wilson",
    "Vitalik Buterin",
    "Austin Russell",
    "Ritesh Agarwal",
    "Simon Tian",
    "Stacey Ferreira",
    "Cathy Tie",
    "Boyan Slat",
    "Erin Smith",
    "Divyaditya Shrivastava",
    "Meghan Sharkus",
    "Andrew Ninh",
    "Melvin Du",
    "Paige Brown",
    "Ilayda Buyukdogan",
    "Ignacio Belieres Montero",
    "Joseph Kitonga",
    "Toni Oloko",
    null,
    "Brandon Wang"
  ],
  "revision_id": 1580936803,
  "value": [
    "Q22277554",
    "Q108871755",
    "Q1421302",
    "Q16197959",
    "Q55687800",
    "Q61070629",
    "Q25095457",
    "Q16234559",
    "Q58494969",
    "Q18324728",
    "Q66686322",
    "Q110864757",
    "Q110865581",
    "Q110865641",
    "Q110865711",
    "Q110865792",
    "Q110865872",
    "Q110864632",
    "Q110864359",
    "Q110864575",
    "Q110864324"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q108871755": 1,
      "Q110864324": 1,
      "Q110864359": 1,
      "Q110864575": 1,
      "Q110864632": 1,
      "Q110864757": 1,
      "Q110865581": 1,
      "Q110865641": 1,
      "Q110865711": 1,
      "Q110865792": 1,
      "Q110865872": 1,
      "Q1421302": 1,
      "Q16197959": 1,
      "Q16234559": 1,
      "Q18324728": 1,
      "Q22277554": 1,
      "Q25095457": 1,
      "Q55687800": 1,
      "Q58494969": 1,
      "Q61070629": 1,
      "Q66686322": 1
    },
    "new_unique": [
      "Q108871755",
      "Q110864324",
      "Q110864359",
      "Q110864575",
      "Q110864632",
      "Q110864757",
      "Q110865581",
      "Q110865641",
      "Q110865711",
      "Q110865792",
      "Q110865872",
      "Q1421302",
      "Q16197959",
      "Q16234559",
      "Q18324728",
      "Q22277554",
      "Q25095457",
      "Q55687800",
      "Q58494969",
      "Q61070629",
      "Q66686322"
    ],
    "new_values": [
      "Q22277554",
      "Q108871755",
      "Q1421302",
      "Q16197959",
      "Q55687800",
      "Q61070629",
      "Q25095457",
      "Q16234559",
      "Q58494969",
      "Q18324728",
      "Q66686322",
      "Q110864757",
      "Q110865581",
      "Q110865641",
      "Q110865711",
      "Q110865792",
      "Q110865872",
      "Q110864632",
      "Q110864359",
      "Q110864575",
      "Q110864324"
    ],
    "new_values_raw": [
      "Q22277554",
      "Q108871755",
      "Q1421302",
      "Q16197959",
      "Q55687800",
      "Q61070629",
      "Q25095457",
      "Q16234559",
      "Q58494969",
      "Q18324728",
      "Q66686322",
      "Q110864757",
      "Q110865581",
      "Q110865641",
      "Q110865711",
      "Q110865792",
      "Q110865872",
      "Q110864632",
      "Q110864359",
      "Q110864575",
      "Q110864324"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q108871755": 1,
      "Q110864324": 1,
      "Q110864359": 1,
      "Q110864575": 1,
      "Q110864632": 1,
      "Q110864757": 1,
      "Q110864939": 1,
      "Q110865581": 1,
      "Q110865641": 1,
      "Q110865711": 1,
      "Q110865792": 1,
      "Q110865872": 1,
      "Q1421302": 1,
      "Q16197959": 1,
      "Q16234559": 1,
      "Q18324728": 1,
      "Q22277554": 1,
      "Q25095457": 1,
      "Q55687800": 1,
      "Q58494969": 1,
      "Q61070629": 1,
      "Q66686322": 1
    },
    "old_unique": [
      "Q108871755",
      "Q110864324",
      "Q110864359",
      "Q110864575",
      "Q110864632",
      "Q110864757",
      "Q110864939",
      "Q110865581",
      "Q110865641",
      "Q110865711",
      "Q110865792",
      "Q110865872",
      "Q1421302",
      "Q16197959",
      "Q16234559",
      "Q18324728",
      "Q22277554",
      "Q25095457",
      "Q55687800",
      "Q58494969",
      "Q61070629",
      "Q66686322"
    ],
    "old_values": [
      "Q22277554",
      "Q108871755",
      "Q1421302",
      "Q16197959",
      "Q55687800",
      "Q61070629",
      "Q25095457",
      "Q16234559",
      "Q58494969",
      "Q18324728",
      "Q66686322",
      "Q110864757",
      "Q110865581",
      "Q110865641",
      "Q110865711",
      "Q110865792",
      "Q110865872",
      "Q110864632",
      "Q110864359",
      "Q110864575",
      "Q110864939",
      "Q110864324"
    ],
    "old_values_raw": [
      "Q22277554",
      "Q108871755",
      "Q1421302",
      "Q16197959",
      "Q55687800",
      "Q61070629",
      "Q25095457",
      "Q16234559",
      "Q58494969",
      "Q18324728",
      "Q66686322",
      "Q110864757",
      "Q110865581",
      "Q110865641",
      "Q110865711",
      "Q110865792",
      "Q110865872",
      "Q110864632",
      "Q110864359",
      "Q110864575",
      "Q110864939",
      "Q110864324"
    ],
    "removed_unique_values": [
      "Q110864939"
    ],
    "retained_unique_values": [
      "Q108871755",
      "Q110864324",
      "Q110864359",
      "Q110864575",
      "Q110864632",
      "Q110864757",
      "Q110865581",
      "Q110865641",
      "Q110865711",
      "Q110865792",
      "Q110865872",
      "Q1421302",
      "Q16197959",
      "Q16234559",
      "Q18324728",
      "Q22277554",
      "Q25095457",
      "Q55687800",
      "Q58494969",
      "Q61070629",
      "Q66686322"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q110864939": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "Venture capitalist",
    "American entrepreneur",
    "American student, scientist",
    "Russian-Canadian computer scientist",
    "American businessperson",
    "Indian entrepreneur",
    "Canadian entrepreneur",
    "American entrepreneur, speaker and author",
    "Canadian entrepreneur",
    "Dutch inventor and environmentalist",
    "American entrepreneur (born 1999)",
    "Founder at Paladin Drones",
    "Founder and CEO at ExpressionMed",
    "Founder of Docbot",
    "Founder of Ramp USA",
    "Cofounder of WindBorne Systems",
    "Co-founder, CEO at Compocket",
    "CEO at Epic Aerospace",
    "Founder of Vitable Health",
    "Co-founder of Dandy",
    "Cofounder and CTO of Dreambound"
  ],
  "value_labels_en": [
    "Laura Deming",
    "Dylan Field",
    "Taylor Wilson",
    "Vitalik Buterin",
    "Austin Russell",
    "Ritesh Agarwal",
    "Simon Tian",
    "Stacey Ferreira",
    "Cathy Tie",
    "Boyan Slat",
    "Erin Smith",
    "Divyaditya Shrivastava",
    "Meghan Sharkus",
    "Andrew Ninh",
    "Melvin Du",
    "Paige Brown",
    "Ilayda Buyukdogan",
    "Ignacio Belieres Montero",
    "Joseph Kitonga",
    "Toni Oloko",
    "Brandon Wang"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-02-22T08:57:12",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1346",
  "report_revision_new": 1581266996,
  "report_revision_old": 1580828847,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Q22277554",
    "Q108871755",
    "Q1421302",
    "Q16197959",
    "Q55687800",
    "Q61070629",
    "Q25095457",
    "Q16234559",
    "Q58494969",
    "Q18324728",
    "Q66686322",
    "Q110864757",
    "Q110865581",
    "Q110865641",
    "Q110865711",
    "Q110865792",
    "Q110865872",
    "Q110864632",
    "Q110864359",
    "Q110864575",
    "Q110864939",
    "Q110864324"
  ],
  "value_descriptions_en": [
    "Venture capitalist",
    "American entrepreneur",
    "American student, scientist",
    "Russian-Canadian computer scientist",
    "American businessperson",
    "Indian entrepreneur",
    "Canadian entrepreneur",
    "American entrepreneur, speaker and author",
    "Canadian entrepreneur",
    "Dutch inventor and environmentalist",
    "American entrepreneur (born 1999)",
    "Founder at Paladin Drones",
    "Founder and CEO at ExpressionMed",
    "Founder of Docbot",
    "Founder of Ramp USA",
    "Cofounder of WindBorne Systems",
    "Co-founder, CEO at Compocket",
    "CEO at Epic Aerospace",
    "Founder of Vitable Health",
    "Co-founder of Dandy",
    null,
    "Cofounder and CTO of Dreambound"
  ],
  "value_labels_en": [
    "Laura Deming",
    "Dylan Field",
    "Taylor Wilson",
    "Vitalik Buterin",
    "Austin Russell",
    "Ritesh Agarwal",
    "Simon Tian",
    "Stacey Ferreira",
    "Cathy Tie",
    "Boyan Slat",
    "Erin Smith",
    "Divyaditya Shrivastava",
    "Meghan Sharkus",
    "Andrew Ninh",
    "Melvin Du",
    "Paige Brown",
    "Ilayda Buyukdogan",
    "Ignacio Belieres Montero",
    "Joseph Kitonga",
    "Toni Oloko",
    null,
    "Brandon Wang"
  ]
}
```

### Local Evidence

```json
{
  "found": 21,
  "local_availability_result": false,
  "local_ids_count": 33,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q108871755"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110864324"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110864359"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110864575"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110864632"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110864757"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110865581"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110865641"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110865711"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110865792"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110865872"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q1421302"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q16197959"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q16234559"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q18324728"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q22277554"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q25095457"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q55687800"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q58494969"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q61070629"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q66686322"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q108871755"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110864324"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110864359"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110864575"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110864632"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110864757"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110865581"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110865641"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110865711"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110865792"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q110865872"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q1421302"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q16197959"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q16234559"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q18324728"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q22277554"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q25095457"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q55687800"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q58494969"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q61070629"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q66686322"
    }
  ],
  "needed": 21,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q22277554",
      "Q108871755",
      "Q1421302",
      "Q16197959",
      "Q55687800",
      "Q61070629",
      "Q25095457",
      "Q16234559",
      "Q58494969",
      "Q18324728",
      "Q66686322",
      "Q110864757",
      "Q110865581",
      "Q110865641",
      "Q110865711",
      "Q110865792",
      "Q110865872",
      "Q110864632",
      "Q110864359",
      "Q110864575",
      "Q110864939",
      "Q110864324"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q22277554",
    "Q108871755",
    "Q1421302",
    "Q16197959",
    "Q55687800",
    "Q61070629",
    "Q25095457",
    "Q16234559",
    "Q58494969",
    "Q18324728",
    "Q66686322",
    "Q110864757",
    "Q110865581",
    "Q110865641",
    "Q110865711",
    "Q110865792",
    "Q110865872",
    "Q110864632",
    "Q110864359",
    "Q110864575",
    "Q110864324"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q108871755",
    "Q110864324",
    "Q110864359",
    "Q110864575",
    "Q110864632",
    "Q110864757",
    "Q110865581",
    "Q110865641",
    "Q110865711",
    "Q110865792",
    "Q110865872",
    "Q1421302",
    "Q16197959",
    "Q16234559",
    "Q18324728",
    "Q22277554",
    "Q25095457",
    "Q55687800",
    "Q58494969",
    "Q61070629",
    "Q66686322"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "winner of a competition or similar event, not to be used from the awardees record (instead use \"award received\" (P166), possibly qualified with \"for work\" (P1686)) nor for wars or battles",
    "label": "winner"
  },
  "qid": {
    "description": "scholarship founded by Peter Thiel",
    "label": "Thiel Fellowship"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "inverse constraint",
    "qid": "Q21510855"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Q108871755": 1,
        "Q110864324": 1,
        "Q110864359": 1,
        "Q110864575": 1,
        "Q110864632": 1,
        "Q110864757": 1,
        "Q110865581": 1,
        "Q110865641": 1,
        "Q110865711": 1,
        "Q110865792": 1,
        "Q110865872": 1,
        "Q1421302": 1,
        "Q16197959": 1,
        "Q16234559": 1,
        "Q18324728": 1,
        "Q22277554": 1,
        "Q25095457": 1,
        "Q55687800": 1,
        "Q58494969": 1,
        "Q61070629": 1,
        "Q66686322": 1
      },
      "new_unique": [
        "Q108871755",
        "Q110864324",
        "Q110864359",
        "Q110864575",
        "Q110864632",
        "Q110864757",
        "Q110865581",
        "Q110865641",
        "Q110865711",
        "Q110865792",
        "Q110865872",
        "Q1421302",
        "Q16197959",
        "Q16234559",
        "Q18324728",
        "Q22277554",
        "Q25095457",
        "Q55687800",
        "Q58494969",
        "Q61070629",
        "Q66686322"
      ],
      "new_values": [
        "Q22277554",
        "Q108871755",
        "Q1421302",
        "Q16197959",
        "Q55687800",
        "Q61070629",
        "Q25095457",
        "Q16234559",
        "Q58494969",
        "Q18324728",
        "Q66686322",
        "Q110864757",
        "Q110865581",
        "Q110865641",
        "Q110865711",
        "Q110865792",
        "Q110865872",
        "Q110864632",
        "Q110864359",
        "Q110864575",
        "Q110864324"
      ],
      "new_values_raw": [
        "Q22277554",
        "Q108871755",
        "Q1421302",
        "Q16197959",
        "Q55687800",
        "Q61070629",
        "Q25095457",
        "Q16234559",
        "Q58494969",
        "Q18324728",
        "Q66686322",
        "Q110864757",
        "Q110865581",
        "Q110865641",
        "Q110865711",
        "Q110865792",
        "Q110865872",
        "Q110864632",
        "Q110864359",
        "Q110864575",
        "Q110864324"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q108871755": 1,
        "Q110864324": 1,
        "Q110864359": 1,
        "Q110864575": 1,
        "Q110864632": 1,
        "Q110864757": 1,
        "Q110864939": 1,
        "Q110865581": 1,
        "Q110865641": 1,
        "Q110865711": 1,
        "Q110865792": 1,
        "Q110865872": 1,
        "Q1421302": 1,
        "Q16197959": 1,
        "Q16234559": 1,
        "Q18324728": 1,
        "Q22277554": 1,
        "Q25095457": 1,
        "Q55687800": 1,
        "Q58494969": 1,
        "Q61070629": 1,
        "Q66686322": 1
      },
      "old_unique": [
        "Q108871755",
        "Q110864324",
        "Q110864359",
        "Q110864575",
        "Q110864632",
        "Q110864757",
        "Q110864939",
        "Q110865581",
        "Q110865641",
        "Q110865711",
        "Q110865792",
        "Q110865872",
        "Q1421302",
        "Q16197959",
        "Q16234559",
        "Q18324728",
        "Q22277554",
        "Q25095457",
        "Q55687800",
        "Q58494969",
        "Q61070629",
        "Q66686322"
      ],
      "old_values": [
        "Q22277554",
        "Q108871755",
        "Q1421302",
        "Q16197959",
        "Q55687800",
        "Q61070629",
        "Q25095457",
        "Q16234559",
        "Q58494969",
        "Q18324728",
        "Q66686322",
        "Q110864757",
        "Q110865581",
        "Q110865641",
        "Q110865711",
        "Q110865792",
        "Q110865872",
        "Q110864632",
        "Q110864359",
        "Q110864575",
        "Q110864939",
        "Q110864324"
      ],
      "old_values_raw": [
        "Q22277554",
        "Q108871755",
        "Q1421302",
        "Q16197959",
        "Q55687800",
        "Q61070629",
        "Q25095457",
        "Q16234559",
        "Q58494969",
        "Q18324728",
        "Q66686322",
        "Q110864757",
        "Q110865581",
        "Q110865641",
        "Q110865711",
        "Q110865792",
        "Q110865872",
        "Q110864632",
        "Q110864359",
        "Q110864575",
        "Q110864939",
        "Q110864324"
      ],
      "removed_unique_values": [
        "Q110864939"
      ],
      "retained_unique_values": [
        "Q108871755",
        "Q110864324",
        "Q110864359",
        "Q110864575",
        "Q110864632",
        "Q110864757",
        "Q110865581",
        "Q110865641",
        "Q110865711",
        "Q110865792",
        "Q110865872",
        "Q1421302",
        "Q16197959",
        "Q16234559",
        "Q18324728",
        "Q22277554",
        "Q25095457",
        "Q55687800",
        "Q58494969",
        "Q61070629",
        "Q66686322"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q110864939": {
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
      "found": 21,
      "independent_match_count": 0,
      "local_ids_count": 33,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q108871755"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q110864324"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q110864359"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q110864575"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q110864632"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q110864757"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q110865581"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q110865641"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q110865711"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q110865792"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q110865872"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q1421302"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q16197959"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q16234559"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q18324728"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q22277554"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q25095457"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q55687800"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q58494969"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q61070629"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q66686322"
        }
      ],
      "needed": 21,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q22277554",
        "Q108871755",
        "Q1421302",
        "Q16197959",
        "Q55687800",
        "Q61070629",
        "Q25095457",
        "Q16234559",
        "Q58494969",
        "Q18324728",
        "Q66686322",
        "Q110864757",
        "Q110865581",
        "Q110865641",
        "Q110865711",
        "Q110865792",
        "Q110865872",
        "Q110864632",
        "Q110864359",
        "Q110864575",
        "Q110864939",
        "Q110864324"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_selection_ambiguous",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q82550_2443411347`

| Field | Value |
|---|---|
| qid | Q82550 |
| property | P1889 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_SELECTION_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | head |
| constraint_family | Q21510851 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_selection_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q82550::P1889 |
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
| truth_tokens_preview | ["Q65708270", "Q9351759"] |
| classification_target_tokens | ["Q66746920"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | unknown_selection_ambiguous |
| rationale | Subset repair only shows retained values in the pre-repair target property; this is not independent local grounding. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q66746920"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q66746920"
  ],
  "removed_unique_values": [
    "Q66746920"
  ],
  "retained_support_tokens": [
    "Q65708270",
    "Q9351759"
  ],
  "retained_unique_values": [
    "Q65708270",
    "Q9351759"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_selection_ambiguous",
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
  "author": "S41pdrc4od",
  "kind": "A_BOX",
  "new_value": [
    "Q65708270",
    "Q9351759"
  ],
  "new_value_descriptions_en": [
    "type of subject heading in a controlled vocabulary",
    "easy to remember, memorable, catchy word or phrase in a song"
  ],
  "new_value_labels_en": [
    "topical term",
    "buzzword"
  ],
  "old_value": [
    "Q65708270",
    "Q9351759",
    "Q66746920"
  ],
  "old_value_descriptions_en": [
    "type of subject heading in a controlled vocabulary",
    "easy to remember, memorable, catchy word or phrase in a song",
    "word or phrase used with specialized meaning in a particular field, with little or different wider usage"
  ],
  "old_value_labels_en": [
    "topical term",
    "buzzword",
    "jargon term"
  ],
  "revision_id": 2443411347,
  "value": [
    "Q65708270",
    "Q9351759"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q65708270": 1,
      "Q9351759": 1
    },
    "new_unique": [
      "Q65708270",
      "Q9351759"
    ],
    "new_values": [
      "Q65708270",
      "Q9351759"
    ],
    "new_values_raw": [
      "Q65708270",
      "Q9351759"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q65708270": 1,
      "Q66746920": 1,
      "Q9351759": 1
    },
    "old_unique": [
      "Q65708270",
      "Q66746920",
      "Q9351759"
    ],
    "old_values": [
      "Q65708270",
      "Q9351759",
      "Q66746920"
    ],
    "old_values_raw": [
      "Q65708270",
      "Q9351759",
      "Q66746920"
    ],
    "removed_unique_values": [
      "Q66746920"
    ],
    "retained_unique_values": [
      "Q65708270",
      "Q9351759"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q66746920": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "type of subject heading in a controlled vocabulary",
    "easy to remember, memorable, catchy word or phrase in a song"
  ],
  "value_labels_en": [
    "topical term",
    "buzzword"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-19T09:19:14",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1889",
  "report_revision_new": 2444012051,
  "report_revision_old": 2443804788,
  "report_violation_type": "Symmetric",
  "report_violation_type_normalized": "Symmetric",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Symmetric",
  "value": [
    "Q65708270",
    "Q9351759",
    "Q66746920"
  ],
  "value_descriptions_en": [
    "type of subject heading in a controlled vocabulary",
    "easy to remember, memorable, catchy word or phrase in a song",
    "word or phrase used with specialized meaning in a particular field, with little or different wider usage"
  ],
  "value_labels_en": [
    "topical term",
    "buzzword",
    "jargon term"
  ]
}
```

### Local Evidence

```json
{
  "found": 2,
  "local_availability_result": false,
  "local_ids_count": 16,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q65708270"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q9351759"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q65708270"
    },
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q9351759"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q65708270",
      "Q9351759",
      "Q66746920"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q65708270",
    "Q9351759"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q65708270",
    "Q9351759"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "item that is different from another item, with which it may be confused",
    "label": "different from"
  },
  "qid": {
    "description": "fashionable word or phrase used to impress rather than for its technical meaning",
    "label": "buzzword"
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
    "label_en": "symmetric constraint",
    "qid": "Q21510862"
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
        "Q65708270": 1,
        "Q9351759": 1
      },
      "new_unique": [
        "Q65708270",
        "Q9351759"
      ],
      "new_values": [
        "Q65708270",
        "Q9351759"
      ],
      "new_values_raw": [
        "Q65708270",
        "Q9351759"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q65708270": 1,
        "Q66746920": 1,
        "Q9351759": 1
      },
      "old_unique": [
        "Q65708270",
        "Q66746920",
        "Q9351759"
      ],
      "old_values": [
        "Q65708270",
        "Q9351759",
        "Q66746920"
      ],
      "old_values_raw": [
        "Q65708270",
        "Q9351759",
        "Q66746920"
      ],
      "removed_unique_values": [
        "Q66746920"
      ],
      "retained_unique_values": [
        "Q65708270",
        "Q9351759"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q66746920": {
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
      "found": 2,
      "independent_match_count": 0,
      "local_ids_count": 16,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q65708270"
        },
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q9351759"
        }
      ],
      "needed": 2,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q65708270",
        "Q9351759",
        "Q66746920"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_selection_ambiguous",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q83823_2441288381`

| Field | Value |
|---|---|
| qid | Q83823 |
| property | P1454 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_SELECTION_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | head |
| constraint_family | Q21510865 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_selection_ambiguous |
| decision_constraint_type |   |
| group_key | ABOX::Q83823::P1454 |
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
| truth_tokens_preview | ["Q3742388"] |
| classification_target_tokens | ["Q891723"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | unknown_selection_ambiguous |
| rationale | Subset repair only shows retained values in the pre-repair target property; this is not independent local grounding. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "subset deletion is explained by removed values, not retained values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Q891723"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q891723"
  ],
  "removed_unique_values": [
    "Q891723"
  ],
  "retained_support_tokens": [
    "Q3742388"
  ],
  "retained_unique_values": [
    "Q3742388"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_selection_ambiguous",
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
  "author": "ZioNicco",
  "kind": "A_BOX",
  "new_value": [
    "Q3742388"
  ],
  "new_value_descriptions_en": [
    "Italian company owned by shareholders"
  ],
  "new_value_labels_en": [
    "società per azioni"
  ],
  "old_value": [
    "Q3742388",
    "Q891723"
  ],
  "old_value_descriptions_en": [
    "Italian company owned by shareholders",
    "company that offers its securities for sale to the general public"
  ],
  "old_value_labels_en": [
    "società per azioni",
    "public company"
  ],
  "revision_id": 2441288381,
  "value": [
    "Q3742388"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q3742388": 1
    },
    "new_unique": [
      "Q3742388"
    ],
    "new_values": [
      "Q3742388"
    ],
    "new_values_raw": [
      "Q3742388"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q3742388": 1,
      "Q891723": 1
    },
    "old_unique": [
      "Q3742388",
      "Q891723"
    ],
    "old_values": [
      "Q3742388",
      "Q891723"
    ],
    "old_values_raw": [
      "Q3742388",
      "Q891723"
    ],
    "removed_unique_values": [
      "Q891723"
    ],
    "retained_unique_values": [
      "Q3742388"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "Q891723": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "Italian company owned by shareholders"
  ],
  "value_labels_en": [
    "società per azioni"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T09:55:31",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1454",
  "report_revision_new": 2442252909,
  "report_revision_old": 2441731976,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Q3742388",
    "Q891723"
  ],
  "value_descriptions_en": [
    "Italian company owned by shareholders",
    "company that offers its securities for sale to the general public"
  ],
  "value_labels_en": [
    "società per azioni",
    "public company"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 82,
  "local_support_for_retained_value": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q3742388"
    }
  ],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q3742388"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q3742388",
      "Q891723"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q3742388"
  ],
  "truth_tokens_in_recorded_matches": [
    "Q3742388"
  ],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "legal form of an entity",
    "label": "legal form"
  },
  "qid": {
    "description": "national rail operator of Italy",
    "label": "Ferrovie dello Stato Italiane"
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
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
        "Q3742388": 1
      },
      "new_unique": [
        "Q3742388"
      ],
      "new_values": [
        "Q3742388"
      ],
      "new_values_raw": [
        "Q3742388"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q3742388": 1,
        "Q891723": 1
      },
      "old_unique": [
        "Q3742388",
        "Q891723"
      ],
      "old_values": [
        "Q3742388",
        "Q891723"
      ],
      "old_values_raw": [
        "Q3742388",
        "Q891723"
      ],
      "removed_unique_values": [
        "Q891723"
      ],
      "retained_unique_values": [
        "Q3742388"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q891723": {
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
      "independent_match_count": 0,
      "local_ids_count": 82,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q3742388"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q3742388",
        "Q891723"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "unknown_selection_ambiguous",
    "step": "branch"
  }
]
```

---
