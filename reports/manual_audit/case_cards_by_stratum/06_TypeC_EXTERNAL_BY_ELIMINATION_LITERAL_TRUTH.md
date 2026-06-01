# TypeC_EXTERNAL_BY_ELIMINATION_LITERAL_TRUTH

Cases: 31

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q122748457_2440237243`

| Field | Value |
|---|---|
| qid | Q122748457 |
| property | P170 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21510865 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q122748457::P170 |
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
| truth_tokens_preview | ["SOMEVALUE"] |
| classification_target_tokens | ["SOMEVALUE"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "SOMEVALUE"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "SOMEVALUE"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q4233718"
  ],
  "removed_unique_values": [
    "Q4233718"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Hannolans",
  "kind": "A_BOX",
  "new_value": [
    "SOMEVALUE"
  ],
  "old_value": [
    "Q4233718"
  ],
  "old_value_descriptions_en": [
    "unknown creator of a work (do not use as value of P50; use \"unknown value\" instead)"
  ],
  "old_value_labels_en": [
    "anonymous"
  ],
  "revision_id": 2440237243,
  "value": [
    "SOMEVALUE"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "SOMEVALUE"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "SOMEVALUE": 1
    },
    "new_unique": [
      "SOMEVALUE"
    ],
    "new_values": [
      "SOMEVALUE"
    ],
    "new_values_raw": [
      "SOMEVALUE"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q4233718": 1
    },
    "old_unique": [
      "Q4233718"
    ],
    "old_values": [
      "Q4233718"
    ],
    "old_values_raw": [
      "Q4233718"
    ],
    "removed_unique_values": [
      "Q4233718"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q4233718": {
        "new": 0,
        "old": 1
      },
      "SOMEVALUE": {
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
  "report_fix_date": "2025-12-11T14:30:59",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P170",
  "report_revision_new": 2440893955,
  "report_revision_old": 2440429522,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": [
    "Q4233718"
  ],
  "value_descriptions_en": [
    "unknown creator of a work (do not use as value of P50; use \"unknown value\" instead)"
  ],
  "value_labels_en": [
    "anonymous"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 14,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q4233718"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "SOMEVALUE"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "maker of this creative work or other object (where no more specific property exists)",
    "label": "creator"
  },
  "qid": {
    "description": "anonymous drawing",
    "label": "Academic expression study: indignation"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 14,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q4233718"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q122922876_2440236445`

| Field | Value |
|---|---|
| qid | Q122922876 |
| property | P170 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21510865 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q122922876::P170 |
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
| truth_tokens_preview | ["SOMEVALUE"] |
| classification_target_tokens | ["SOMEVALUE"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "SOMEVALUE"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "SOMEVALUE"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q4233718"
  ],
  "removed_unique_values": [
    "Q4233718"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Hannolans",
  "kind": "A_BOX",
  "new_value": [
    "SOMEVALUE"
  ],
  "old_value": [
    "Q4233718"
  ],
  "old_value_descriptions_en": [
    "unknown creator of a work (do not use as value of P50; use \"unknown value\" instead)"
  ],
  "old_value_labels_en": [
    "anonymous"
  ],
  "revision_id": 2440236445,
  "value": [
    "SOMEVALUE"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "SOMEVALUE"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "SOMEVALUE": 1
    },
    "new_unique": [
      "SOMEVALUE"
    ],
    "new_values": [
      "SOMEVALUE"
    ],
    "new_values_raw": [
      "SOMEVALUE"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q4233718": 1
    },
    "old_unique": [
      "Q4233718"
    ],
    "old_values": [
      "Q4233718"
    ],
    "old_values_raw": [
      "Q4233718"
    ],
    "removed_unique_values": [
      "Q4233718"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q4233718": {
        "new": 0,
        "old": 1
      },
      "SOMEVALUE": {
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
  "report_fix_date": "2025-12-11T14:30:59",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P170",
  "report_revision_new": 2440893955,
  "report_revision_old": 2440429522,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": [
    "Q4233718"
  ],
  "value_descriptions_en": [
    "unknown creator of a work (do not use as value of P50; use \"unknown value\" instead)"
  ],
  "value_labels_en": [
    "anonymous"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 14,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q4233718"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "SOMEVALUE"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "maker of this creative work or other object (where no more specific property exists)",
    "label": "creator"
  },
  "qid": {
    "description": "anonymous drawing (0022.GRO0505.II)",
    "label": "Academic life drawing study: standing male nude, seen from behind"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 14,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q4233718"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q122922920_2440236730`

| Field | Value |
|---|---|
| qid | Q122922920 |
| property | P170 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21510865 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q122922920::P170 |
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
| truth_tokens_preview | ["SOMEVALUE"] |
| classification_target_tokens | ["SOMEVALUE"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "SOMEVALUE"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "SOMEVALUE"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q4233718"
  ],
  "removed_unique_values": [
    "Q4233718"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Hannolans",
  "kind": "A_BOX",
  "new_value": [
    "SOMEVALUE"
  ],
  "old_value": [
    "Q4233718"
  ],
  "old_value_descriptions_en": [
    "unknown creator of a work (do not use as value of P50; use \"unknown value\" instead)"
  ],
  "old_value_labels_en": [
    "anonymous"
  ],
  "revision_id": 2440236730,
  "value": [
    "SOMEVALUE"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "SOMEVALUE"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "SOMEVALUE": 1
    },
    "new_unique": [
      "SOMEVALUE"
    ],
    "new_values": [
      "SOMEVALUE"
    ],
    "new_values_raw": [
      "SOMEVALUE"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q4233718": 1
    },
    "old_unique": [
      "Q4233718"
    ],
    "old_values": [
      "Q4233718"
    ],
    "old_values_raw": [
      "Q4233718"
    ],
    "removed_unique_values": [
      "Q4233718"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q4233718": {
        "new": 0,
        "old": 1
      },
      "SOMEVALUE": {
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
  "report_fix_date": "2025-12-11T14:30:59",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P170",
  "report_revision_new": 2440893955,
  "report_revision_old": 2440429522,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": [
    "Q4233718"
  ],
  "value_descriptions_en": [
    "unknown creator of a work (do not use as value of P50; use \"unknown value\" instead)"
  ],
  "value_labels_en": [
    "anonymous"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 14,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q4233718"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "SOMEVALUE"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "maker of this creative work or other object (where no more specific property exists)",
    "label": "creator"
  },
  "qid": {
    "description": "anonymous drawing (0022.GRO0648.II)",
    "label": "Academic study after a print: standing male nude, seen from behind"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 14,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q4233718"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q122922949_2440236775`

| Field | Value |
|---|---|
| qid | Q122922949 |
| property | P170 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21510865 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q122922949::P170 |
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
| truth_tokens_preview | ["SOMEVALUE"] |
| classification_target_tokens | ["SOMEVALUE"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "SOMEVALUE"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "SOMEVALUE"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q4233718"
  ],
  "removed_unique_values": [
    "Q4233718"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Hannolans",
  "kind": "A_BOX",
  "new_value": [
    "SOMEVALUE"
  ],
  "old_value": [
    "Q4233718"
  ],
  "old_value_descriptions_en": [
    "unknown creator of a work (do not use as value of P50; use \"unknown value\" instead)"
  ],
  "old_value_labels_en": [
    "anonymous"
  ],
  "revision_id": 2440236775,
  "value": [
    "SOMEVALUE"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "SOMEVALUE"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "SOMEVALUE": 1
    },
    "new_unique": [
      "SOMEVALUE"
    ],
    "new_values": [
      "SOMEVALUE"
    ],
    "new_values_raw": [
      "SOMEVALUE"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q4233718": 1
    },
    "old_unique": [
      "Q4233718"
    ],
    "old_values": [
      "Q4233718"
    ],
    "old_values_raw": [
      "Q4233718"
    ],
    "removed_unique_values": [
      "Q4233718"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q4233718": {
        "new": 0,
        "old": 1
      },
      "SOMEVALUE": {
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
  "report_fix_date": "2025-12-11T14:30:59",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P170",
  "report_revision_new": 2440893955,
  "report_revision_old": 2440429522,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": [
    "Q4233718"
  ],
  "value_descriptions_en": [
    "unknown creator of a work (do not use as value of P50; use \"unknown value\" instead)"
  ],
  "value_labels_en": [
    "anonymous"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 16,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q4233718"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "SOMEVALUE"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "maker of this creative work or other object (where no more specific property exists)",
    "label": "creator"
  },
  "qid": {
    "description": "anonymous drawing (0022.GRO1176.II)",
    "label": "Academic study after a sculpture: head of a man looking up"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 16,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q4233718"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q123754467_2443050600`

| Field | Value |
|---|---|
| qid | Q123754467 |
| property | P1146 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q123754467::P1146 |
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
| truth_tokens_preview | ["14681420"] |
| classification_target_tokens | ["14681420"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "14681420"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "14681420"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Kevin Scannell",
  "kind": "A_BOX",
  "new_value": [
    "14681420"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2443050600,
  "value": [
    "14681420"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "14681420"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "14681420": 1
    },
    "new_unique": [
      "14681420"
    ],
    "new_values": [
      "14681420"
    ],
    "new_values_raw": [
      "14681420"
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
      "14681420": {
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
  "report_fix_date": "2025-12-22T08:54:49",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1146",
  "report_revision_new": 2445425502,
  "report_revision_old": 2444863501,
  "report_violation_type": "Item P|106",
  "report_violation_type_normalized": "Item P|106",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|106",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 25,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "14681420"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for athletes in World Athletics database and website",
    "label": "World Athletics athlete ID"
  },
  "qid": {
    "description": "American runner",
    "label": "Cassia Hameline"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
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
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 25,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q124482829_2442848415`

| Field | Value |
|---|---|
| qid | Q124482829 |
| property | P2508 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502410 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q124482829::P2508 |
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
| truth_tokens_preview | ["98746"] |
| classification_target_tokens | ["98746"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "98746"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "98746"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "90209"
  ],
  "removed_unique_values": [
    "90209"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Mcampany",
  "kind": "A_BOX",
  "new_value": [
    "98746"
  ],
  "old_value": [
    "90209"
  ],
  "revision_id": 2442848415,
  "value": [
    "98746"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "98746"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "98746": 1
    },
    "new_unique": [
      "98746"
    ],
    "new_values": [
      "98746"
    ],
    "new_values_raw": [
      "98746"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "90209": 1
    },
    "old_unique": [
      "90209"
    ],
    "old_values": [
      "90209"
    ],
    "old_values_raw": [
      "90209"
    ],
    "removed_unique_values": [
      "90209"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "90209": {
        "new": 0,
        "old": 1
      },
      "98746": {
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
  "report_fix_date": "2025-12-17T09:36:16",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2508",
  "report_revision_new": 2443350503,
  "report_revision_old": 2442945922,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "90209"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 9,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "90209"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "98746"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of a film in the KINENOTE movie database",
    "label": "KINENOTE film ID"
  },
  "qid": {
    "description": "film directed by Shun Konii",
    "label": "いまダンスをするのは誰だ?"
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
    "label_en": "format constraint",
    "qid": "Q21502404"
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
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
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
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 9,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "90209"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q132348179_2439090729`

| Field | Value |
|---|---|
| qid | Q132348179 |
| property | P1559 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q132348179::P1559 |
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
| truth_tokens_preview | ["aurora@de"] |
| classification_target_tokens | ["aurora@de"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "aurora@de"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "aurora@de"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Ameisenigel",
  "kind": "A_BOX",
  "new_value": [
    "aurora@de"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2439090729,
  "value": [
    "aurora@de"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "aurora@de"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "aurora@de": 1
    },
    "new_unique": [
      "aurora@de"
    ],
    "new_values": [
      "aurora@de"
    ],
    "new_values_raw": [
      "aurora@de"
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
      "aurora@de": {
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
  "report_fix_date": "2025-12-09T09:10:20",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1559",
  "report_revision_new": 2439931044,
  "report_revision_old": 2439522677,
  "report_violation_type": "Type Q|5, Q|95074, Q|21070568, Q|15619164, Q|21070598, Q|2239243, Q|64520857, Q|64643615, Q|75855169, Q|16334295",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "fictional human or non-human character in a narrative work of art",
    "human who is hypothesized to exist, but where evidence is not conclusive",
    "entity that has no physical realisation",
    "character who is hypothesized to exist, but where evidence is not conclusive",
    "supernatural animal, generally a hybrid, sometimes part human, whose existence cannot be proven, described in legends, myths, fables, folklore",
    "people who never existed but were once thought to have",
    "person listed in modern scholarship, now shown not to have existed",
    "human being whose existence is not directly attested, but is deduced by other evidence",
    "any set of human beings"
  ],
  "report_violation_type_labels_en": [
    "human",
    "character",
    "human whose existence is disputed",
    "abstract being",
    "figure that may or may not be fictional",
    "mythical creature",
    "fictional human formerly considered to be historical",
    "prosopographical phantom",
    "hypothetical person",
    "group of humans"
  ],
  "report_violation_type_normalized": "Type Q|5, Q|95074, Q|21070568, Q|15619164, Q|21070598, Q|2239243, Q|64520857, Q|64643615, Q|75855169, Q|16334295",
  "report_violation_type_qids": [
    "Q5",
    "Q95074",
    "Q21070568",
    "Q15619164",
    "Q21070598",
    "Q2239243",
    "Q64520857",
    "Q64643615",
    "Q75855169",
    "Q16334295"
  ],
  "report_violation_type_raw": "Type Q|5, Q|95074, Q|21070568, Q|15619164, Q|21070598, Q|2239243, Q|64520857, Q|64643615, Q|75855169, Q|16334295",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 9,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "aurora@de"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "name of a person in their native language",
    "label": "name in native language"
  },
  "qid": {
    "description": null,
    "label": "aurora"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 9,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q133745212_2422784427`

| Field | Value |
|---|---|
| qid | Q133745212 |
| property | P865 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q133745212::P865 |
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
| truth_tokens_preview | ["w1221"] |
| classification_target_tokens | ["w1221"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "w1221"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "w1221"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Epìdosis",
  "kind": "A_BOX",
  "new_value": [
    "w1221"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2422784427,
  "value": [
    "w1221"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "w1221"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "w1221": 1
    },
    "new_unique": [
      "w1221"
    ],
    "new_values": [
      "w1221"
    ],
    "new_values_raw": [
      "w1221"
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
      "w1221": {
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
  "report_fix_date": "2025-10-29T16:26:31",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P865",
  "report_revision_new": 2423382507,
  "report_revision_old": 2422996905,
  "report_violation_type": "Item P|106",
  "report_violation_type_normalized": "Item P|106",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|106",
  "report_violation_types": [
    "Item P|106",
    "Item P|735"
  ],
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 19,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "w1221"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of Bayerisches Musiker-Lexikon Online",
    "label": "BMLO ID"
  },
  "qid": {
    "description": "American soprano (1939-)",
    "label": "Patricia Wells"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 19,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q136537051_2433478275`

| Field | Value |
|---|---|
| qid | Q136537051 |
| property | P9818 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q19474404 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q136537051::P9818 |
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
| truth_tokens_preview | ["710633"] |
| classification_target_tokens | ["710633"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "710633"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "710633"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Iamcarbon",
  "kind": "A_BOX",
  "new_value": [
    "710633"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2433478275,
  "value": [
    "710633"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "710633"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "710633": 1
    },
    "new_unique": [
      "710633"
    ],
    "new_values": [
      "710633"
    ],
    "new_values_raw": [
      "710633"
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
      "710633": {
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
  "report_fix_date": "2025-11-30T04:46:58",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P9818",
  "report_revision_new": 2435916802,
  "report_revision_old": 2435426806,
  "report_violation_type": "Conflicts with P|747",
  "report_violation_type_normalized": "Conflicts with P|747",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|747",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "710633"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a book/publication on the Penguin Random House website",
    "label": "Penguin Random House work ID"
  },
  "qid": {
    "description": "non-fiction work by Mark Shepard",
    "label": "There Are No Facts"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 5,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q137150011_2447057063`

| Field | Value |
|---|---|
| qid | Q137150011 |
| property | P213 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q137150011::P213 |
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
| truth_tokens_preview | ["0000000108151397"] |
| classification_target_tokens | ["0000000108151397"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "0000000108151397"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "0000000108151397"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Epìdosis",
  "kind": "A_BOX",
  "new_value": [
    "0000000108151397"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2447057063,
  "value": [
    "0000000108151397"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "0000000108151397"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "0000000108151397": 1
    },
    "new_unique": [
      "0000000108151397"
    ],
    "new_values": [
      "0000000108151397"
    ],
    "new_values_raw": [
      "0000000108151397"
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
      "0000000108151397": {
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
  "report_fix_date": "2025-12-27T13:27:38",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P213",
  "report_revision_new": 2447790547,
  "report_revision_old": 2447400320,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 27,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "0000000108151397"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "International Standard Name Identifier for an identity. Starting with 0000.",
    "label": "ISNI"
  },
  "qid": {
    "description": "German jurist (1680-1736)",
    "label": "Franz Ernst Vogt"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 27,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 011. `repair_Q137300743_2442980512`

| Field | Value |
|---|---|
| qid | Q137300743 |
| property | P625 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q137300743::P625 |
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
| truth_tokens_preview | ["55.598134,-2.433272"] |
| classification_target_tokens | ["55.598134,-2.433272"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "55.598134,-2.433272"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "55.598134,-2.433272"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Back ache",
  "kind": "A_BOX",
  "new_value": [
    "55.598134,-2.433272"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2442980512,
  "value": [
    "55.598134,-2.433272"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "55.598134,-2.433272"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "55.598134,-2.433272": 1
    },
    "new_unique": [
      "55.598134,-2.433272"
    ],
    "new_values": [
      "55.598134,-2.433272"
    ],
    "new_values_raw": [
      "55.598134,-2.433272"
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
      "55.598134,-2.433272": {
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
  "report_fix_date": "2025-12-20T10:04:01",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P625",
  "report_revision_new": 2444447211,
  "report_revision_old": 2444031676,
  "report_violation_type": "Q|163740, Q|43229",
  "report_violation_type_descriptions_en": [
    "organization operated for a collective benefit",
    "social entity established to meet needs or pursue goals"
  ],
  "report_violation_type_labels_en": [
    "nonprofit organization",
    "organization"
  ],
  "report_violation_type_normalized": "Q|163740, Q|43229",
  "report_violation_type_qids": [
    "Q163740",
    "Q43229"
  ],
  "report_violation_type_raw": "Q|163740, Q|43229",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 9,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "55.598134,-2.433272"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
    "label": "coordinate location"
  },
  "qid": {
    "description": "organization for the preservation of Hume Castle",
    "label": "Hume Castle Preservation Trust"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 9,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 012. `repair_Q137460393_2444071356`

| Field | Value |
|---|---|
| qid | Q137460393 |
| property | P569 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q137460393::P569 |
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
| truth_tokens_preview | ["+1881-08-31T00:00:00Z"] |
| classification_target_tokens | ["+1881-08-31T00:00:00Z"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "+1881-08-31T00:00:00Z"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "+1881-08-31T00:00:00Z"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Артемий Александров",
  "kind": "A_BOX",
  "new_value": [
    "+1881-08-31T00:00:00Z"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2444071356,
  "value": [
    "+1881-08-31T00:00:00Z"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "+1881-08-31T00:00:00Z"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "+1881-08-31T00:00:00Z": 1
    },
    "new_unique": [
      "+1881-08-31T00:00:00Z"
    ],
    "new_values": [
      "+1881-08-31T00:00:00Z"
    ],
    "new_values_raw": [
      "+1881-08-31T00:00:00Z"
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
      "+1881-08-31T00:00:00Z": {
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
  "report_fix_date": "2025-12-23T12:32:07",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P569",
  "report_revision_new": 2446031583,
  "report_revision_old": 2445442393,
  "report_violation_type": "Type Q|5, Q|95074, Q|21070568, Q|146, Q|144, Q|726, Q|3542731, Q|30017383, Q|13002315, Q|75855169, Q|64520857, Q|64643615, Q|18347143, Q|2345820, Q|57812611, Q|611804, Q|729, Q|10855152",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "fictional human or non-human character in a narrative work of art",
    "human who is hypothesized to exist, but where evidence is not conclusive",
    "small domesticated carnivorous mammal",
    "domesticated species of canid",
    "domesticated four-footed mammal from the equine family",
    "class of individual animals which are fictional",
    "class of individual fictional characters in the form of an organism",
    "named person or animal that appears in legends that have some claim to be historical",
    "human being whose existence is not directly attested, but is deduced by other evidence",
    "people who never existed but were once thought to have",
    "person listed in modern scholarship, now shown not to have existed",
    "subclass of fossil",
    "individual who died before or during birth",
    "mammal in captivity",
    "group of genetically identical plants, fungi, or bacteria, originating vegetatively from a single ancestor, growing at a single site",
    "kingdom of multicellular eukaryotic organisms",
    "individual living thing"
  ],
  "report_violation_type_labels_en": [
    "human",
    "character",
    "human whose existence is disputed",
    "cat",
    "dog",
    "horse",
    "fictional animal character",
    "fictional organism",
    "legendary figure",
    "hypothetical person",
    "fictional human formerly considered to be historical",
    "prosopographical phantom",
    "Hominin fossil",
    "stillborn child",
    "captive mammal",
    "clonal colony",
    "Animalia",
    "individual organism"
  ],
  "report_violation_type_normalized": "Type Q|5, Q|95074, Q|21070568, Q|146, Q|144, Q|726, Q|3542731, Q|30017383, Q|13002315, Q|75855169, Q|64520857, Q|64643615, Q|18347143, Q|2345820, Q|57812611, Q|611804, Q|729, Q|10855152",
  "report_violation_type_qids": [
    "Q5",
    "Q95074",
    "Q21070568",
    "Q146",
    "Q144",
    "Q726",
    "Q3542731",
    "Q30017383",
    "Q13002315",
    "Q75855169",
    "Q64520857",
    "Q64643615",
    "Q18347143",
    "Q2345820",
    "Q57812611",
    "Q611804",
    "Q729",
    "Q10855152"
  ],
  "report_violation_type_raw": "Type Q|5, Q|95074, Q|21070568, Q|146, Q|144, Q|726, Q|3542731, Q|30017383, Q|13002315, Q|75855169, Q|64520857, Q|64643615, Q|18347143, Q|2345820, Q|57812611, Q|611804, Q|729, Q|10855152",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "+1881-08-31T00:00:00Z"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "date on which the subject was born",
    "label": "date of birth"
  },
  "qid": {
    "description": "Russian and Soviet theater and sinema actress.",
    "label": "Anna Esipovich"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
    "label_en": "range constraint",
    "qid": "Q21510860"
  },
  {
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
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
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 11,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 013. `repair_Q1733505_2443482750`

| Field | Value |
|---|---|
| qid | Q1733505 |
| property | P213 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q1733505::P213 |
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
| truth_tokens_preview | ["0000000529102913"] |
| classification_target_tokens | ["0000000529102913"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "0000000529102913"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "0000000529102913"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "0000000034296656"
  ],
  "removed_unique_values": [
    "0000000034296656"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Epìdosis",
  "kind": "A_BOX",
  "new_value": [
    "0000000529102913"
  ],
  "old_value": [
    "0000000034296656"
  ],
  "revision_id": 2443482750,
  "value": [
    "0000000529102913"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "0000000529102913"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "0000000529102913": 1
    },
    "new_unique": [
      "0000000529102913"
    ],
    "new_values": [
      "0000000529102913"
    ],
    "new_values_raw": [
      "0000000529102913"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "0000000034296656": 1
    },
    "old_unique": [
      "0000000034296656"
    ],
    "old_values": [
      "0000000034296656"
    ],
    "old_values_raw": [
      "0000000034296656"
    ],
    "removed_unique_values": [
      "0000000034296656"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "0000000034296656": {
        "new": 0,
        "old": 1
      },
      "0000000529102913": {
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
  "report_fix_date": "2025-12-19T11:33:06",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P213",
  "report_revision_new": 2444041942,
  "report_revision_old": 2443850051,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "0000000034296656"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 30,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "0000000034296656"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "0000000529102913"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "International Standard Name Identifier for an identity. Starting with 0000.",
    "label": "ISNI"
  },
  "qid": {
    "description": "German actor and director (1893–1950)",
    "label": "Karl Wüstenhagen"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 30,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "0000000034296656"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 014. `repair_Q2121419_2311240375`

| Field | Value |
|---|---|
| qid | Q2121419 |
| property | P5838 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q2121419::P5838 |
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
| truth_tokens_preview | ["PC9J18"] |
| classification_target_tokens | ["PC9J18"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "PC9J18"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "PC9J18"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "NoInkling",
  "kind": "A_BOX",
  "new_value": [
    "PC9J18"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2311240375,
  "value": [
    "PC9J18"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "PC9J18"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "PC9J18": 1
    },
    "new_unique": [
      "PC9J18"
    ],
    "new_values": [
      "PC9J18"
    ],
    "new_values_raw": [
      "PC9J18"
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
      "PC9J18": {
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
  "report_fix_date": "2025-02-16T07:34:43",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5838",
  "report_revision_new": 2312208675,
  "report_revision_old": 2311723062,
  "report_violation_type": "Mandatory Qualifiers",
  "report_violation_type_normalized": "Mandatory Qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Mandatory Qualifiers",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 33,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "PC9J18"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "six-alphanumeric-character Nintendo GameID for a specific game on the GameCube or Wii",
    "label": "Nintendo GameID (GameCube/Wii)"
  },
  "qid": {
    "description": "1988 video game",
    "label": "The NewZealand Story"
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
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 33,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 015. `repair_Q2332808_2309914266`

| Field | Value |
|---|---|
| qid | Q2332808 |
| property | P5838 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q2332808::P5838 |
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
| truth_tokens_preview | ["FFJJ01"] |
| classification_target_tokens | ["FFJJ01"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "FFJJ01"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "FFJJ01"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "NoInkling",
  "kind": "A_BOX",
  "new_value": [
    "FFJJ01"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2309914266,
  "value": [
    "FFJJ01"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "FFJJ01"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "FFJJ01": 1
    },
    "new_unique": [
      "FFJJ01"
    ],
    "new_values": [
      "FFJJ01"
    ],
    "new_values_raw": [
      "FFJJ01"
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
      "FFJJ01": {
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
  "report_fix_date": "2025-02-16T07:34:43",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5838",
  "report_revision_new": 2312208675,
  "report_revision_old": 2311723062,
  "report_violation_type": "Mandatory Qualifiers",
  "report_violation_type_normalized": "Mandatory Qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Mandatory Qualifiers",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 25,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "FFJJ01"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "six-alphanumeric-character Nintendo GameID for a specific game on the GameCube or Wii",
    "label": "Nintendo GameID (GameCube/Wii)"
  },
  "qid": {
    "description": "1991 video game",
    "label": "Metal Max"
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
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 25,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 016. `repair_Q27179108_2395899652`

| Field | Value |
|---|---|
| qid | Q27179108 |
| property | P18 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q27179108::P18 |
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
| truth_tokens_preview | ["Hans de Winiwarter (1946).jpg"] |
| classification_target_tokens | ["Hans de Winiwarter (1946).jpg"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Hans de Winiwarter (1946).jpg"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Hans de Winiwarter (1946).jpg"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Galpas79",
  "kind": "A_BOX",
  "new_value": [
    "Hans de Winiwarter (1946).jpg"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2395899652,
  "value": [
    "Hans de Winiwarter (1946).jpg"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Hans de Winiwarter (1946).jpg"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Hans de Winiwarter (1946).jpg": 1
    },
    "new_unique": [
      "Hans de Winiwarter (1946).jpg"
    ],
    "new_values": [
      "Hans de Winiwarter (1946).jpg"
    ],
    "new_values_raw": [
      "Hans de Winiwarter (1946).jpg"
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
      "Hans de Winiwarter (1946).jpg": {
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
  "report_fix_date": "2025-08-25T13:10:27",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P18",
  "report_revision_new": 2396416041,
  "report_revision_old": 2396089597,
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 29,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Hans de Winiwarter (1946).jpg"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "image of relevant illustration of the subject; if available, also use more specific properties (sample: coat of arms image, locator map, flag image, signature image, logo image, collage image)",
    "label": "image"
  },
  "qid": {
    "description": "Belgian embryologist and university teacher",
    "label": "Hans de Winiwarter"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "Commons link constraint",
    "qid": "Q21510852"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 29,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 017. `repair_Q28079683_2396039573`

| Field | Value |
|---|---|
| qid | Q28079683 |
| property | P18 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q28079683::P18 |
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
| truth_tokens_preview | ["Tianlin Railway Station 20250812.jpg"] |
| classification_target_tokens | ["Tianlin Railway Station 20250812.jpg"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Tianlin Railway Station 20250812.jpg"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Tianlin Railway Station 20250812.jpg"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "TIAN LIN.jpg"
  ],
  "removed_unique_values": [
    "TIAN LIN.jpg"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "TimWu007",
  "kind": "A_BOX",
  "new_value": [
    "Tianlin Railway Station 20250812.jpg"
  ],
  "old_value": [
    "TIAN LIN.jpg"
  ],
  "revision_id": 2396039573,
  "value": [
    "Tianlin Railway Station 20250812.jpg"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Tianlin Railway Station 20250812.jpg"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Tianlin Railway Station 20250812.jpg": 1
    },
    "new_unique": [
      "Tianlin Railway Station 20250812.jpg"
    ],
    "new_values": [
      "Tianlin Railway Station 20250812.jpg"
    ],
    "new_values_raw": [
      "Tianlin Railway Station 20250812.jpg"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "TIAN LIN.jpg": 1
    },
    "old_unique": [
      "TIAN LIN.jpg"
    ],
    "old_values": [
      "TIAN LIN.jpg"
    ],
    "old_values_raw": [
      "TIAN LIN.jpg"
    ],
    "removed_unique_values": [
      "TIAN LIN.jpg"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "TIAN LIN.jpg": {
        "new": 0,
        "old": 1
      },
      "Tianlin Railway Station 20250812.jpg": {
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
  "report_fix_date": "2025-08-26T13:03:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P18",
  "report_revision_new": 2396759629,
  "report_revision_old": 2396416041,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "TIAN LIN.jpg"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 19,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "TIAN LIN.jpg"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Tianlin Railway Station 20250812.jpg"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "image of relevant illustration of the subject; if available, also use more specific properties (sample: coat of arms image, locator map, flag image, signature image, logo image, collage image)",
    "label": "image"
  },
  "qid": {
    "description": "railway station in Tianlin, Baise, Guangxi, People's Republic of China",
    "label": "Tianlin Railway Station"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "Commons link constraint",
    "qid": "Q21510852"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 19,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "TIAN LIN.jpg"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 018. `repair_Q28400196_2447258614`

| Field | Value |
|---|---|
| qid | Q28400196 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q28400196::P373 |
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
| truth_tokens_preview | ["Coaches of OL Lyonnes"] |
| classification_target_tokens | ["Coaches of OL Lyonnes"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Coaches of OL Lyonnes"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Coaches of OL Lyonnes"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Coaches of Olympique Lyonnais Féminin"
  ],
  "removed_unique_values": [
    "Coaches of Olympique Lyonnais Féminin"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Pi bot",
  "kind": "A_BOX",
  "new_value": [
    "Coaches of OL Lyonnes"
  ],
  "old_value": [
    "Coaches of Olympique Lyonnais Féminin"
  ],
  "revision_id": 2447258614,
  "value": [
    "Coaches of OL Lyonnes"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Coaches of OL Lyonnes"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Coaches of OL Lyonnes": 1
    },
    "new_unique": [
      "Coaches of OL Lyonnes"
    ],
    "new_values": [
      "Coaches of OL Lyonnes"
    ],
    "new_values_raw": [
      "Coaches of OL Lyonnes"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Coaches of Olympique Lyonnais Féminin": 1
    },
    "old_unique": [
      "Coaches of Olympique Lyonnais Féminin"
    ],
    "old_values": [
      "Coaches of Olympique Lyonnais Féminin"
    ],
    "old_values_raw": [
      "Coaches of Olympique Lyonnais Féminin"
    ],
    "removed_unique_values": [
      "Coaches of Olympique Lyonnais Féminin"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Coaches of OL Lyonnes": {
        "new": 1,
        "old": 0
      },
      "Coaches of Olympique Lyonnais Féminin": {
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
    "Coaches of Olympique Lyonnais Féminin"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Coaches of Olympique Lyonnais Féminin"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Coaches of OL Lyonnes"
  ],
  "truth_tokens_in_recorded_matches": [],
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
    "label": "Category:OL Lyonnes managers"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 5,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Coaches of Olympique Lyonnais Féminin"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 019. `repair_Q28410501_2446378409`

| Field | Value |
|---|---|
| qid | Q28410501 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q28410501::P373 |
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
| truth_tokens_preview | ["Route 17 (Quanzhou Bus)"] |
| classification_target_tokens | ["Route 17 (Quanzhou Bus)"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Route 17 (Quanzhou Bus)"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Route 17 (Quanzhou Bus)"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Line 17 (Quanzhou Bus)"
  ],
  "removed_unique_values": [
    "Line 17 (Quanzhou Bus)"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Pi bot",
  "kind": "A_BOX",
  "new_value": [
    "Route 17 (Quanzhou Bus)"
  ],
  "old_value": [
    "Line 17 (Quanzhou Bus)"
  ],
  "revision_id": 2446378409,
  "value": [
    "Route 17 (Quanzhou Bus)"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Route 17 (Quanzhou Bus)"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Route 17 (Quanzhou Bus)": 1
    },
    "new_unique": [
      "Route 17 (Quanzhou Bus)"
    ],
    "new_values": [
      "Route 17 (Quanzhou Bus)"
    ],
    "new_values_raw": [
      "Route 17 (Quanzhou Bus)"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Line 17 (Quanzhou Bus)": 1
    },
    "old_unique": [
      "Line 17 (Quanzhou Bus)"
    ],
    "old_values": [
      "Line 17 (Quanzhou Bus)"
    ],
    "old_values_raw": [
      "Line 17 (Quanzhou Bus)"
    ],
    "removed_unique_values": [
      "Line 17 (Quanzhou Bus)"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Line 17 (Quanzhou Bus)": {
        "new": 0,
        "old": 1
      },
      "Route 17 (Quanzhou Bus)": {
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
  "report_fix_date": "2025-12-25T19:21:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447067079,
  "report_revision_old": 2446526020,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Line 17 (Quanzhou Bus)"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 7,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Line 17 (Quanzhou Bus)"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Route 17 (Quanzhou Bus)"
  ],
  "truth_tokens_in_recorded_matches": [],
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
    "description": "Buslinie in der Volksrepublik China",
    "label": "泉州公交17路"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 7,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Line 17 (Quanzhou Bus)"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 020. `repair_Q29890456_2444941612`

| Field | Value |
|---|---|
| qid | Q29890456 |
| property | P381 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q29890456::P381 |
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
| truth_tokens_preview | ["05891"] |
| classification_target_tokens | ["05891"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "05891"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "05891"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "5891"
  ],
  "removed_unique_values": [
    "5891"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "AnBuKu",
  "kind": "A_BOX",
  "new_value": [
    "05891"
  ],
  "old_value": [
    "5891"
  ],
  "revision_id": 2444941612,
  "value": [
    "05891"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "05891"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "05891": 1
    },
    "new_unique": [
      "05891"
    ],
    "new_values": [
      "05891"
    ],
    "new_values_raw": [
      "05891"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "5891": 1
    },
    "old_unique": [
      "5891"
    ],
    "old_values": [
      "5891"
    ],
    "old_values_raw": [
      "5891"
    ],
    "removed_unique_values": [
      "5891"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "05891": {
        "new": 1,
        "old": 0
      },
      "5891": {
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
  "report_fix_date": "2025-12-23T13:28:51",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P381",
  "report_revision_new": 2446054236,
  "report_revision_old": 2445457863,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "5891"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 13,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "5891"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "05891"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for cultural properties in Switzerland",
    "label": "PCP reference number"
  },
  "qid": {
    "description": "school building in Aigle in the canton of Vaud, Switzerland",
    "label": "secondary school"
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
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 13,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "5891"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 021. `repair_Q3067784_2310365235`

| Field | Value |
|---|---|
| qid | Q3067784 |
| property | P5838 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q3067784::P5838 |
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
| truth_tokens_preview | ["SEPE41", "SEPP41", "SEPX41", "SEPZ41"] |
| classification_target_tokens | ["SEPZ41"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "SEPZ41"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "SEPZ41"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "SEPE41",
    "SEPP41",
    "SEPX41"
  ],
  "retained_unique_values": [
    "SEPE41",
    "SEPP41",
    "SEPX41"
  ],
  "semantic_action": "ADD_SUPERSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "NoInkling",
  "kind": "A_BOX",
  "new_value": [
    "SEPE41",
    "SEPP41",
    "SEPX41",
    "SEPZ41"
  ],
  "old_value": [
    "SEPE41",
    "SEPP41",
    "SEPX41"
  ],
  "revision_id": 2310365235,
  "value": [
    "SEPE41",
    "SEPP41",
    "SEPX41",
    "SEPZ41"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "SEPZ41"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "SEPE41": 1,
      "SEPP41": 1,
      "SEPX41": 1,
      "SEPZ41": 1
    },
    "new_unique": [
      "SEPE41",
      "SEPP41",
      "SEPX41",
      "SEPZ41"
    ],
    "new_values": [
      "SEPE41",
      "SEPP41",
      "SEPX41",
      "SEPZ41"
    ],
    "new_values_raw": [
      "SEPE41",
      "SEPP41",
      "SEPX41",
      "SEPZ41"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SEPE41": 1,
      "SEPP41": 1,
      "SEPX41": 1
    },
    "old_unique": [
      "SEPE41",
      "SEPP41",
      "SEPX41"
    ],
    "old_values": [
      "SEPE41",
      "SEPP41",
      "SEPX41"
    ],
    "old_values_raw": [
      "SEPE41",
      "SEPP41",
      "SEPX41"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "SEPE41",
      "SEPP41",
      "SEPX41"
    ],
    "semantic_action": "ADD_SUPERSET",
    "value_multiplicity_changes": {
      "SEPZ41": {
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
  "report_fix_date": "2025-02-16T07:34:43",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5838",
  "report_revision_new": 2312208675,
  "report_revision_old": 2311723062,
  "report_violation_type": "Mandatory Qualifiers",
  "report_violation_type_normalized": "Mandatory Qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Mandatory Qualifiers",
  "value": [
    "SEPE41",
    "SEPP41",
    "SEPX41"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 35,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "SEPE41",
      "SEPP41",
      "SEPX41"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "SEPE41",
    "SEPP41",
    "SEPX41",
    "SEPZ41"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "six-alphanumeric-character Nintendo GameID for a specific game on the GameCube or Wii",
    "label": "Nintendo GameID (GameCube/Wii)"
  },
  "qid": {
    "description": "2011 video game",
    "label": "The Black Eyed Peas Experience"
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
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 35,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "SEPE41",
        "SEPP41",
        "SEPX41"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 022. `repair_Q312751_2334676628`

| Field | Value |
|---|---|
| qid | Q312751 |
| property | P7699 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q312751::P7699 |
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
| truth_tokens_preview | ["LNB:BD4i;=BJ"] |
| classification_target_tokens | ["LNB:BD4i;=BJ"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "LNB:BD4i;=BJ"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "LNB:BD4i;=BJ"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Harmonia Amanda",
  "kind": "A_BOX",
  "new_value": [
    "LNB:BD4i;=BJ"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2334676628,
  "value": [
    "LNB:BD4i;=BJ"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "LNB:BD4i;=BJ"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "LNB:BD4i;=BJ": 1
    },
    "new_unique": [
      "LNB:BD4i;=BJ"
    ],
    "new_values": [
      "LNB:BD4i;=BJ"
    ],
    "new_values_raw": [
      "LNB:BD4i;=BJ"
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
      "LNB:BD4i;=BJ": {
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
  "report_fix_date": "2025-04-09T06:13:31",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7699",
  "report_revision_new": 2336146311,
  "report_revision_old": 2335760439,
  "report_violation_type": "Label in lt language",
  "report_violation_type_normalized": "Label in lt language",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Label in lt language",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 51,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "LNB:BD4i;=BJ"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "authority id at the National Library of Lithuania, part of VIAF (code LIH)",
    "label": "National Library of Lithuania ID"
  },
  "qid": {
    "description": "American filmmaker and novelist",
    "label": "Charlie Kaufman"
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
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
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
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 51,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 023. `repair_Q31854135_2442784175`

| Field | Value |
|---|---|
| qid | Q31854135 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q31854135::P373 |
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
| truth_tokens_preview | ["Náměstí Míru 251 (Týn nad Vltavou)"] |
| classification_target_tokens | ["Náměstí Míru 251 (Týn nad Vltavou)"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Náměstí Míru 251 (Týn nad Vltavou)"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Náměstí Míru 251 (Týn nad Vltavou)"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Náměstí Míru 251 (Týn nad Vltavou)"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2442784175,
  "value": [
    "Náměstí Míru 251 (Týn nad Vltavou)"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Náměstí Míru 251 (Týn nad Vltavou)"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Náměstí Míru 251 (Týn nad Vltavou)": 1
    },
    "new_unique": [
      "Náměstí Míru 251 (Týn nad Vltavou)"
    ],
    "new_values": [
      "Náměstí Míru 251 (Týn nad Vltavou)"
    ],
    "new_values_raw": [
      "Náměstí Míru 251 (Týn nad Vltavou)"
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
      "Náměstí Míru 251 (Týn nad Vltavou)": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 17,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Náměstí Míru 251 (Týn nad Vltavou)"
  ],
  "truth_tokens_in_recorded_matches": [],
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
    "description": "kulturní památka České republiky na území obce Týn nad Vltavou",
    "label": "Náměstí Míru 251"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 17,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 024. `repair_Q50329510_2438127958`

| Field | Value |
|---|---|
| qid | Q50329510 |
| property | P7439 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q50329510::P7439 |
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
| truth_tokens_preview | ["1518570"] |
| classification_target_tokens | ["1518570"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "1518570"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "1518570"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "HedgeHog",
  "kind": "A_BOX",
  "new_value": [
    "1518570"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2438127958,
  "value": [
    "1518570"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "1518570"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1518570": 1
    },
    "new_unique": [
      "1518570"
    ],
    "new_values": [
      "1518570"
    ],
    "new_values_raw": [
      "1518570"
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
      "1518570": {
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
  "report_fix_date": "2025-12-11T07:58:59",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7439",
  "report_revision_new": 2440757500,
  "report_revision_old": 2440359350,
  "report_violation_type": "Type Q|47461344",
  "report_violation_type_descriptions_en": [
    "any work expressed in writing, such as inscriptions, manuscripts, documents or maps"
  ],
  "report_violation_type_labels_en": [
    "written work"
  ],
  "report_violation_type_normalized": "Type Q|47461344",
  "report_violation_type_qids": [
    "Q47461344"
  ],
  "report_violation_type_raw": "Type Q|47461344",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 13,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "1518570"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for the literary work in FantLab",
    "label": "FantLab work ID"
  },
  "qid": {
    "description": "poem by Robert Burns",
    "label": "O were my love yon lilac fair"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 13,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 025. `repair_Q50414229_703355254`

| Field | Value |
|---|---|
| qid | Q50414229 |
| property | P644 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q50414229::P644 |
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
| truth_tokens_preview | ["1205730"] |
| classification_target_tokens | ["1205730"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "1205730"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "1205730"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Muhammad elhossary",
  "kind": "A_BOX",
  "new_value": [
    "1205730"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 703355254,
  "value": [
    "1205730"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "1205730"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1205730": 1
    },
    "new_unique": [
      "1205730"
    ],
    "new_values": [
      "1205730"
    ],
    "new_values_raw": [
      "1205730"
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
      "1205730": {
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
  "report_fix_date": "2018-07-01T19:28:45",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P644",
  "report_revision_new": 705159203,
  "report_revision_old": 704829822,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 13,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "1205730"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "genomic starting coordinate of the biological sequence (e.g. a gene)",
    "label": "genomic start"
  },
  "qid": {
    "description": "Microbial gene (pseudogene) encodes a protein found in Escherichia coli str. K-12 substr. MG1655, Note: pseudogene, portal protein family, e14 prophage;Phage or Prophage Related",
    "label": "beeE encodes: pseudogene, portal protein family, e14 prophage;Phage or Prophage Related b1151"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 13,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 026. `repair_Q540943_2398875501`

| Field | Value |
|---|---|
| qid | Q540943 |
| property | P18 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q540943::P18 |
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
| truth_tokens_preview | ["Rage Against the Machine cover.jpg"] |
| classification_target_tokens | ["Rage Against the Machine cover.jpg"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Rage Against the Machine cover.jpg"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Rage Against the Machine cover.jpg"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "RageAgainsttheMachineRageAgainsttheMachine.jpg"
  ],
  "removed_unique_values": [
    "RageAgainsttheMachineRageAgainsttheMachine.jpg"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Trade",
  "kind": "A_BOX",
  "new_value": [
    "Rage Against the Machine cover.jpg"
  ],
  "old_value": [
    "RageAgainsttheMachineRageAgainsttheMachine.jpg"
  ],
  "revision_id": 2398875501,
  "value": [
    "Rage Against the Machine cover.jpg"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Rage Against the Machine cover.jpg"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Rage Against the Machine cover.jpg": 1
    },
    "new_unique": [
      "Rage Against the Machine cover.jpg"
    ],
    "new_values": [
      "Rage Against the Machine cover.jpg"
    ],
    "new_values_raw": [
      "Rage Against the Machine cover.jpg"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "RageAgainsttheMachineRageAgainsttheMachine.jpg": 1
    },
    "old_unique": [
      "RageAgainsttheMachineRageAgainsttheMachine.jpg"
    ],
    "old_values": [
      "RageAgainsttheMachineRageAgainsttheMachine.jpg"
    ],
    "old_values_raw": [
      "RageAgainsttheMachineRageAgainsttheMachine.jpg"
    ],
    "removed_unique_values": [
      "RageAgainsttheMachineRageAgainsttheMachine.jpg"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Rage Against the Machine cover.jpg": {
        "new": 1,
        "old": 0
      },
      "RageAgainsttheMachineRageAgainsttheMachine.jpg": {
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
  "report_fix_date": "2025-09-02T13:56:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P18",
  "report_revision_new": 2399622179,
  "report_revision_old": 2399167844,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "RageAgainsttheMachineRageAgainsttheMachine.jpg"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 35,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "RageAgainsttheMachineRageAgainsttheMachine.jpg"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Rage Against the Machine cover.jpg"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "image of relevant illustration of the subject; if available, also use more specific properties (sample: coat of arms image, locator map, flag image, signature image, logo image, collage image)",
    "label": "image"
  },
  "qid": {
    "description": "1992 debut studio album by Rage Against the Machine",
    "label": "Rage Against the Machine"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "Commons link constraint",
    "qid": "Q21510852"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 35,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "RageAgainsttheMachineRageAgainsttheMachine.jpg"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 027. `repair_Q5591950_2309445074`

| Field | Value |
|---|---|
| qid | Q5591950 |
| property | P5838 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q5591950::P5838 |
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
| truth_tokens_preview | ["WGDEA4", "WGDPA4", "WGDJA4"] |
| classification_target_tokens | ["WGDJA4"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "WGDJA4"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "WGDJA4"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "WGDEA4",
    "WGDPA4"
  ],
  "retained_unique_values": [
    "WGDEA4",
    "WGDPA4"
  ],
  "semantic_action": "ADD_SUPERSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "NoInkling",
  "kind": "A_BOX",
  "new_value": [
    "WGDEA4",
    "WGDPA4",
    "WGDJA4"
  ],
  "old_value": [
    "WGDEA4",
    "WGDPA4"
  ],
  "revision_id": 2309445074,
  "value": [
    "WGDEA4",
    "WGDPA4",
    "WGDJA4"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "WGDJA4"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "WGDEA4": 1,
      "WGDJA4": 1,
      "WGDPA4": 1
    },
    "new_unique": [
      "WGDEA4",
      "WGDJA4",
      "WGDPA4"
    ],
    "new_values": [
      "WGDEA4",
      "WGDPA4",
      "WGDJA4"
    ],
    "new_values_raw": [
      "WGDEA4",
      "WGDPA4",
      "WGDJA4"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "WGDEA4": 1,
      "WGDPA4": 1
    },
    "old_unique": [
      "WGDEA4",
      "WGDPA4"
    ],
    "old_values": [
      "WGDEA4",
      "WGDPA4"
    ],
    "old_values_raw": [
      "WGDEA4",
      "WGDPA4"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "WGDEA4",
      "WGDPA4"
    ],
    "semantic_action": "ADD_SUPERSET",
    "value_multiplicity_changes": {
      "WGDJA4": {
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
  "report_fix_date": "2025-02-16T07:34:43",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5838",
  "report_revision_new": 2312208675,
  "report_revision_old": 2311723062,
  "report_violation_type": "Mandatory Qualifiers",
  "report_violation_type_normalized": "Mandatory Qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Mandatory Qualifiers",
  "value": [
    "WGDEA4",
    "WGDPA4"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 25,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "WGDEA4",
      "WGDPA4"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "WGDEA4",
    "WGDPA4",
    "WGDJA4"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "six-alphanumeric-character Nintendo GameID for a specific game on the GameCube or Wii",
    "label": "Nintendo GameID (GameCube/Wii)"
  },
  "qid": {
    "description": "2008 video game",
    "label": "Gradius ReBirth"
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
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 25,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "WGDEA4",
        "WGDPA4"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 028. `repair_Q67629_2335501612`

| Field | Value |
|---|---|
| qid | Q67629 |
| property | P7699 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q67629::P7699 |
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
| truth_tokens_preview | ["LNB:Wcb;=Bs"] |
| classification_target_tokens | ["LNB:Wcb;=Bs"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "LNB:Wcb;=Bs"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "LNB:Wcb;=Bs"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "RonaldCarrierNU",
  "kind": "A_BOX",
  "new_value": [
    "LNB:Wcb;=Bs"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2335501612,
  "value": [
    "LNB:Wcb;=Bs"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "LNB:Wcb;=Bs"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "LNB:Wcb;=Bs": 1
    },
    "new_unique": [
      "LNB:Wcb;=Bs"
    ],
    "new_values": [
      "LNB:Wcb;=Bs"
    ],
    "new_values_raw": [
      "LNB:Wcb;=Bs"
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
      "LNB:Wcb;=Bs": {
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
  "report_fix_date": "2025-04-09T06:13:31",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7699",
  "report_revision_new": 2336146311,
  "report_revision_old": 2335760439,
  "report_violation_type": "Label in lt language",
  "report_violation_type_normalized": "Label in lt language",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Label in lt language",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 83,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "LNB:Wcb;=Bs"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "authority id at the National Library of Lithuania, part of VIAF (code LIH)",
    "label": "National Library of Lithuania ID"
  },
  "qid": {
    "description": "Turkish-German writer (born 1964)",
    "label": "Feridun Zaimoğlu"
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
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
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
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 83,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 029. `repair_Q721001_2311223818`

| Field | Value |
|---|---|
| qid | Q721001 |
| property | P5838 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q721001::P5838 |
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
| truth_tokens_preview | ["MB8E8P", "MB8J8P", "MB8P8P"] |
| classification_target_tokens | ["MB8P8P"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "MB8P8P"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "MB8P8P"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "MB8E8P",
    "MB8J8P"
  ],
  "retained_unique_values": [
    "MB8E8P",
    "MB8J8P"
  ],
  "semantic_action": "ADD_SUPERSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "NoInkling",
  "kind": "A_BOX",
  "new_value": [
    "MB8E8P",
    "MB8J8P",
    "MB8P8P"
  ],
  "old_value": [
    "MB8E8P",
    "MB8J8P"
  ],
  "revision_id": 2311223818,
  "value": [
    "MB8E8P",
    "MB8J8P",
    "MB8P8P"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "MB8P8P"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "MB8E8P": 1,
      "MB8J8P": 1,
      "MB8P8P": 1
    },
    "new_unique": [
      "MB8E8P",
      "MB8J8P",
      "MB8P8P"
    ],
    "new_values": [
      "MB8E8P",
      "MB8J8P",
      "MB8P8P"
    ],
    "new_values_raw": [
      "MB8E8P",
      "MB8J8P",
      "MB8P8P"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "MB8E8P": 1,
      "MB8J8P": 1
    },
    "old_unique": [
      "MB8E8P",
      "MB8J8P"
    ],
    "old_values": [
      "MB8E8P",
      "MB8J8P"
    ],
    "old_values_raw": [
      "MB8E8P",
      "MB8J8P"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "MB8E8P",
      "MB8J8P"
    ],
    "semantic_action": "ADD_SUPERSET",
    "value_multiplicity_changes": {
      "MB8P8P": {
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
  "report_fix_date": "2025-02-16T07:34:43",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5838",
  "report_revision_new": 2312208675,
  "report_revision_old": 2311723062,
  "report_violation_type": "Mandatory Qualifiers",
  "report_violation_type_normalized": "Mandatory Qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Mandatory Qualifiers",
  "value": [
    "MB8E8P",
    "MB8J8P"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 59,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MB8E8P",
      "MB8J8P"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "MB8E8P",
    "MB8J8P",
    "MB8P8P"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "six-alphanumeric-character Nintendo GameID for a specific game on the GameCube or Wii",
    "label": "Nintendo GameID (GameCube/Wii)"
  },
  "qid": {
    "description": "1989 video game",
    "label": "Phantasy Star II"
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
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 59,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "MB8E8P",
        "MB8J8P"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 030. `repair_Q8030771_2433495254`

| Field | Value |
|---|---|
| qid | Q8030771 |
| property | P9818 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q8030771::P9818 |
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
| truth_tokens_preview | ["120774"] |
| classification_target_tokens | ["120774"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "120774"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "120774"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Iamcarbon",
  "kind": "A_BOX",
  "new_value": [
    "120774"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2433495254,
  "value": [
    "120774"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "120774"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "120774": 1
    },
    "new_unique": [
      "120774"
    ],
    "new_values": [
      "120774"
    ],
    "new_values_raw": [
      "120774"
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
      "120774": {
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
  "report_fix_date": "2025-11-30T04:46:58",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P9818",
  "report_revision_new": 2435916802,
  "report_revision_old": 2435426806,
  "report_violation_type": "Conflicts with P|747",
  "report_violation_type_normalized": "Conflicts with P|747",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|747",
  "report_violation_types": [
    "Conflicts with P|747",
    "Conflicts with P|8383"
  ],
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 25,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "120774"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a book/publication on the Penguin Random House website",
    "label": "Penguin Random House work ID"
  },
  "qid": {
    "description": "2009 crime novel by Håkan Nesser",
    "label": "Woman with Birthmark"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 25,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---

## 031. `repair_Q8963433_2309428741`

| Field | Value |
|---|---|
| qid | Q8963433 |
| property | P5838 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q8963433::P5838 |
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
| truth_tokens_preview | ["FF6J01"] |
| classification_target_tokens | ["FF6J01"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "FF6J01"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "FF6J01"
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
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "NoInkling",
  "kind": "A_BOX",
  "new_value": [
    "FF6J01"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2309428741,
  "value": [
    "FF6J01"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "FF6J01"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "FF6J01": 1
    },
    "new_unique": [
      "FF6J01"
    ],
    "new_values": [
      "FF6J01"
    ],
    "new_values_raw": [
      "FF6J01"
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
      "FF6J01": {
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
  "report_fix_date": "2025-02-16T07:34:43",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5838",
  "report_revision_new": 2312208675,
  "report_revision_old": 2311723062,
  "report_violation_type": "Mandatory Qualifiers",
  "report_violation_type_normalized": "Mandatory Qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Mandatory Qualifiers",
  "value": [
    "MISSING"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 19,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "FF6J01"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "six-alphanumeric-character Nintendo GameID for a specific game on the GameCube or Wii",
    "label": "Nintendo GameID (GameCube/Wii)"
  },
  "qid": {
    "description": "NES video game",
    "label": "Ganbare Goemon 2"
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
      "constraint_type": null,
      "signal": "L4_constraints"
    },
    "result": false,
    "step": "rule_deterministic"
  },
  {
    "evidence": {
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 19,
      "matched": false,
      "matches": [],
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
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
    "result": true,
    "step": "fallback_external"
  },
  {
    "result": "external_by_elimination",
    "step": "branch"
  }
]
```

---
