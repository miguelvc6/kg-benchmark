# TypeA_FORMAT_VALUE_PRUNING

Cases: 26

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q119245_2297179246`

| Field | Value |
|---|---|
| qid | Q119245 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q119245::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["119169762"] |
| classification_target_tokens | ["pta1024"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pta1024"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pta1024"
  ],
  "removed_unique_values": [
    "pta1024"
  ],
  "retained_support_tokens": [
    "119169762"
  ],
  "retained_unique_values": [
    "119169762"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "119169762"
  ],
  "old_value": [
    "pta1024",
    "119169762"
  ],
  "revision_id": 2297179246,
  "value": [
    "119169762"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "119169762": 1
    },
    "new_unique": [
      "119169762"
    ],
    "new_values": [
      "119169762"
    ],
    "new_values_raw": [
      "119169762"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "119169762": 1,
      "pta1024": 1
    },
    "old_unique": [
      "119169762",
      "pta1024"
    ],
    "old_values": [
      "pta1024",
      "119169762"
    ],
    "old_values_raw": [
      "pta1024",
      "119169762"
    ],
    "removed_unique_values": [
      "pta1024"
    ],
    "retained_unique_values": [
      "119169762"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pta1024": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pta1024",
    "119169762"
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
    "119169762"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German architect (1899–1974)",
    "label": "Hans Bernhard Reichow"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "119169762": 1
      },
      "new_unique": [
        "119169762"
      ],
      "new_values": [
        "119169762"
      ],
      "new_values_raw": [
        "119169762"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "119169762": 1,
        "pta1024": 1
      },
      "old_unique": [
        "119169762",
        "pta1024"
      ],
      "old_values": [
        "pta1024",
        "119169762"
      ],
      "old_values_raw": [
        "pta1024",
        "119169762"
      ],
      "removed_unique_values": [
        "pta1024"
      ],
      "retained_unique_values": [
        "119169762"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pta1024": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pta1024"
      ],
      "report_values": [
        "119169762",
        "pta1024"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "119169762"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q1201231_2297177514`

| Field | Value |
|---|---|
| qid | Q1201231 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q1201231::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["130475173"] |
| classification_target_tokens | ["pkc0062"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pkc0062"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pkc0062"
  ],
  "removed_unique_values": [
    "pkc0062"
  ],
  "retained_support_tokens": [
    "130475173"
  ],
  "retained_unique_values": [
    "130475173"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "130475173"
  ],
  "old_value": [
    "pkc0062",
    "130475173"
  ],
  "revision_id": 2297177514,
  "value": [
    "130475173"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "130475173": 1
    },
    "new_unique": [
      "130475173"
    ],
    "new_values": [
      "130475173"
    ],
    "new_values_raw": [
      "130475173"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "130475173": 1,
      "pkc0062": 1
    },
    "old_unique": [
      "130475173",
      "pkc0062"
    ],
    "old_values": [
      "pkc0062",
      "130475173"
    ],
    "old_values_raw": [
      "pkc0062",
      "130475173"
    ],
    "removed_unique_values": [
      "pkc0062"
    ],
    "retained_unique_values": [
      "130475173"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pkc0062": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pkc0062",
    "130475173"
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
    "130475173"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German politician (1905-1941)",
    "label": "Detlef Dern"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "130475173": 1
      },
      "new_unique": [
        "130475173"
      ],
      "new_values": [
        "130475173"
      ],
      "new_values_raw": [
        "130475173"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "130475173": 1,
        "pkc0062": 1
      },
      "old_unique": [
        "130475173",
        "pkc0062"
      ],
      "old_values": [
        "pkc0062",
        "130475173"
      ],
      "old_values_raw": [
        "pkc0062",
        "130475173"
      ],
      "removed_unique_values": [
        "pkc0062"
      ],
      "retained_unique_values": [
        "130475173"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pkc0062": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pkc0062"
      ],
      "report_values": [
        "130475173",
        "pkc0062"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "130475173"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q1280703_2297190959`

| Field | Value |
|---|---|
| qid | Q1280703 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q1280703::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["134421434"] |
| classification_target_tokens | ["pk00593"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pk00593"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pk00593"
  ],
  "removed_unique_values": [
    "pk00593"
  ],
  "retained_support_tokens": [
    "134421434"
  ],
  "retained_unique_values": [
    "134421434"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "134421434"
  ],
  "old_value": [
    "pk00593",
    "134421434"
  ],
  "revision_id": 2297190959,
  "value": [
    "134421434"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "134421434": 1
    },
    "new_unique": [
      "134421434"
    ],
    "new_values": [
      "134421434"
    ],
    "new_values_raw": [
      "134421434"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "134421434": 1,
      "pk00593": 1
    },
    "old_unique": [
      "134421434",
      "pk00593"
    ],
    "old_values": [
      "pk00593",
      "134421434"
    ],
    "old_values_raw": [
      "pk00593",
      "134421434"
    ],
    "removed_unique_values": [
      "pk00593"
    ],
    "retained_unique_values": [
      "134421434"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pk00593": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pk00593",
    "134421434"
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
    "134421434"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German musician",
    "label": "Hubert Käppel"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "134421434": 1
      },
      "new_unique": [
        "134421434"
      ],
      "new_values": [
        "134421434"
      ],
      "new_values_raw": [
        "134421434"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "134421434": 1,
        "pk00593": 1
      },
      "old_unique": [
        "134421434",
        "pk00593"
      ],
      "old_values": [
        "pk00593",
        "134421434"
      ],
      "old_values_raw": [
        "pk00593",
        "134421434"
      ],
      "removed_unique_values": [
        "pk00593"
      ],
      "retained_unique_values": [
        "134421434"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pk00593": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pk00593"
      ],
      "report_values": [
        "134421434",
        "pk00593"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "134421434"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q1446515_2297190314`

| Field | Value |
|---|---|
| qid | Q1446515 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q1446515::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1036869474"] |
| classification_target_tokens | ["pta0394"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pta0394"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pta0394"
  ],
  "removed_unique_values": [
    "pta0394"
  ],
  "retained_support_tokens": [
    "1036869474"
  ],
  "retained_unique_values": [
    "1036869474"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "1036869474"
  ],
  "old_value": [
    "pta0394",
    "1036869474"
  ],
  "revision_id": 2297190314,
  "value": [
    "1036869474"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1036869474": 1
    },
    "new_unique": [
      "1036869474"
    ],
    "new_values": [
      "1036869474"
    ],
    "new_values_raw": [
      "1036869474"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1036869474": 1,
      "pta0394": 1
    },
    "old_unique": [
      "1036869474",
      "pta0394"
    ],
    "old_values": [
      "pta0394",
      "1036869474"
    ],
    "old_values_raw": [
      "pta0394",
      "1036869474"
    ],
    "removed_unique_values": [
      "pta0394"
    ],
    "retained_unique_values": [
      "1036869474"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pta0394": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pta0394",
    "1036869474"
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
    "1036869474"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German politician (1788-1865)",
    "label": "Franz Damian Görtz"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "1036869474": 1
      },
      "new_unique": [
        "1036869474"
      ],
      "new_values": [
        "1036869474"
      ],
      "new_values_raw": [
        "1036869474"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1036869474": 1,
        "pta0394": 1
      },
      "old_unique": [
        "1036869474",
        "pta0394"
      ],
      "old_values": [
        "pta0394",
        "1036869474"
      ],
      "old_values_raw": [
        "pta0394",
        "1036869474"
      ],
      "removed_unique_values": [
        "pta0394"
      ],
      "retained_unique_values": [
        "1036869474"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pta0394": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pta0394"
      ],
      "report_values": [
        "1036869474",
        "pta0394"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "1036869474"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q14865565_2386559161`

| Field | Value |
|---|---|
| qid | Q14865565 |
| property | P351 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q14865565::P351 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1956"] |
| classification_target_tokens | ["https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
  ],
  "removed_unique_values": [
    "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
  ],
  "retained_support_tokens": [
    "1956"
  ],
  "retained_unique_values": [
    "1956"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "170.85.55.79",
  "kind": "A_BOX",
  "new_value": [
    "1956"
  ],
  "old_value": [
    "1956",
    "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
  ],
  "revision_id": 2386559161,
  "value": [
    "1956"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1956": 1
    },
    "new_unique": [
      "1956"
    ],
    "new_values": [
      "1956"
    ],
    "new_values_raw": [
      "1956"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1956": 1,
      "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956": 1
    },
    "old_unique": [
      "1956",
      "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
    ],
    "old_values": [
      "1956",
      "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
    ],
    "old_values_raw": [
      "1956",
      "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
    ],
    "removed_unique_values": [
      "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
    ],
    "retained_unique_values": [
      "1956"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956": {
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
  "report_fix_date": "2025-08-02T10:34:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P351",
  "report_revision_new": 2387142204,
  "report_revision_old": 2356019188,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Single value"
  ],
  "value": [
    "1956",
    "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
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
    "1956"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a gene per the NCBI Entrez database",
    "label": "Entrez Gene ID"
  },
  "qid": {
    "description": "protein-coding gene in the species Homo sapiens",
    "label": "EGFR"
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
    "label_en": "single-value constraint",
    "qid": "Q19474404"
  },
  {
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
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
        "1956": 1
      },
      "new_unique": [
        "1956"
      ],
      "new_values": [
        "1956"
      ],
      "new_values_raw": [
        "1956"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1956": 1,
        "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956": 1
      },
      "old_unique": [
        "1956",
        "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
      ],
      "old_values": [
        "1956",
        "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
      ],
      "old_values_raw": [
        "1956",
        "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
      ],
      "removed_unique_values": [
        "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
      ],
      "retained_unique_values": [
        "1956"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
      ],
      "report_values": [
        "1956",
        "https://ncbi.nlm.nih.gov/gene?cmd=retrieve&dopt=default&rn=1&list_uids=1956"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "1956"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q1706115_2297189384`

| Field | Value |
|---|---|
| qid | Q1706115 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q1706115::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1051215536"] |
| classification_target_tokens | ["pta1489"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pta1489"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pta1489"
  ],
  "removed_unique_values": [
    "pta1489"
  ],
  "retained_support_tokens": [
    "1051215536"
  ],
  "retained_unique_values": [
    "1051215536"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "1051215536"
  ],
  "old_value": [
    "pta1489",
    "1051215536"
  ],
  "revision_id": 2297189384,
  "value": [
    "1051215536"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1051215536": 1
    },
    "new_unique": [
      "1051215536"
    ],
    "new_values": [
      "1051215536"
    ],
    "new_values_raw": [
      "1051215536"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1051215536": 1,
      "pta1489": 1
    },
    "old_unique": [
      "1051215536",
      "pta1489"
    ],
    "old_values": [
      "pta1489",
      "1051215536"
    ],
    "old_values_raw": [
      "pta1489",
      "1051215536"
    ],
    "removed_unique_values": [
      "pta1489"
    ],
    "retained_unique_values": [
      "1051215536"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pta1489": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pta1489",
    "1051215536"
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
    "1051215536"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German priest (1872–1942)",
    "label": "Josef Zilliken"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "1051215536": 1
      },
      "new_unique": [
        "1051215536"
      ],
      "new_values": [
        "1051215536"
      ],
      "new_values_raw": [
        "1051215536"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1051215536": 1,
        "pta1489": 1
      },
      "old_unique": [
        "1051215536",
        "pta1489"
      ],
      "old_values": [
        "pta1489",
        "1051215536"
      ],
      "old_values_raw": [
        "pta1489",
        "1051215536"
      ],
      "removed_unique_values": [
        "pta1489"
      ],
      "retained_unique_values": [
        "1051215536"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pta1489": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pta1489"
      ],
      "report_values": [
        "1051215536",
        "pta1489"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "1051215536"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q1730530_2297189465`

| Field | Value |
|---|---|
| qid | Q1730530 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q1730530::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["116245123"] |
| classification_target_tokens | ["pkd0092"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pkd0092"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pkd0092"
  ],
  "removed_unique_values": [
    "pkd0092"
  ],
  "retained_support_tokens": [
    "116245123"
  ],
  "retained_unique_values": [
    "116245123"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "116245123"
  ],
  "old_value": [
    "pkd0092",
    "116245123"
  ],
  "revision_id": 2297189465,
  "value": [
    "116245123"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "116245123": 1
    },
    "new_unique": [
      "116245123"
    ],
    "new_values": [
      "116245123"
    ],
    "new_values_raw": [
      "116245123"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "116245123": 1,
      "pkd0092": 1
    },
    "old_unique": [
      "116245123",
      "pkd0092"
    ],
    "old_values": [
      "pkd0092",
      "116245123"
    ],
    "old_values_raw": [
      "pkd0092",
      "116245123"
    ],
    "removed_unique_values": [
      "pkd0092"
    ],
    "retained_unique_values": [
      "116245123"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pkd0092": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pkd0092",
    "116245123"
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
    "116245123"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German politician (1802-1877)",
    "label": "Karl Boost"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "116245123": 1
      },
      "new_unique": [
        "116245123"
      ],
      "new_values": [
        "116245123"
      ],
      "new_values_raw": [
        "116245123"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "116245123": 1,
        "pkd0092": 1
      },
      "old_unique": [
        "116245123",
        "pkd0092"
      ],
      "old_values": [
        "pkd0092",
        "116245123"
      ],
      "old_values_raw": [
        "pkd0092",
        "116245123"
      ],
      "removed_unique_values": [
        "pkd0092"
      ],
      "retained_unique_values": [
        "116245123"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pkd0092": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pkd0092"
      ],
      "report_values": [
        "116245123",
        "pkd0092"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "116245123"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q2076637_2297194746`

| Field | Value |
|---|---|
| qid | Q2076637 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q2076637::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["116026715"] |
| classification_target_tokens | ["pma0277"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pma0277"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pma0277"
  ],
  "removed_unique_values": [
    "pma0277"
  ],
  "retained_support_tokens": [
    "116026715"
  ],
  "retained_unique_values": [
    "116026715"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "116026715"
  ],
  "old_value": [
    "pma0277",
    "116026715"
  ],
  "revision_id": 2297194746,
  "value": [
    "116026715"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "116026715": 1
    },
    "new_unique": [
      "116026715"
    ],
    "new_values": [
      "116026715"
    ],
    "new_values_raw": [
      "116026715"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "116026715": 1,
      "pma0277": 1
    },
    "old_unique": [
      "116026715",
      "pma0277"
    ],
    "old_values": [
      "pma0277",
      "116026715"
    ],
    "old_values_raw": [
      "pma0277",
      "116026715"
    ],
    "removed_unique_values": [
      "pma0277"
    ],
    "retained_unique_values": [
      "116026715"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pma0277": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pma0277",
    "116026715"
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
    "116026715"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "Catholic bishop of Mainz",
    "label": "Peter Leopold Kaiser"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "116026715": 1
      },
      "new_unique": [
        "116026715"
      ],
      "new_values": [
        "116026715"
      ],
      "new_values_raw": [
        "116026715"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "116026715": 1,
        "pma0277": 1
      },
      "old_unique": [
        "116026715",
        "pma0277"
      ],
      "old_values": [
        "pma0277",
        "116026715"
      ],
      "old_values_raw": [
        "pma0277",
        "116026715"
      ],
      "removed_unique_values": [
        "pma0277"
      ],
      "retained_unique_values": [
        "116026715"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pma0277": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pma0277"
      ],
      "report_values": [
        "116026715",
        "pma0277"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "116026715"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q3083115_2297181877`

| Field | Value |
|---|---|
| qid | Q3083115 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q3083115::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["118707795"] |
| classification_target_tokens | ["pka0439"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pka0439"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pka0439"
  ],
  "removed_unique_values": [
    "pka0439"
  ],
  "retained_support_tokens": [
    "118707795"
  ],
  "retained_unique_values": [
    "118707795"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "118707795"
  ],
  "old_value": [
    "pka0439",
    "118707795"
  ],
  "revision_id": 2297181877,
  "value": [
    "118707795"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "118707795": 1
    },
    "new_unique": [
      "118707795"
    ],
    "new_values": [
      "118707795"
    ],
    "new_values_raw": [
      "118707795"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "118707795": 1,
      "pka0439": 1
    },
    "old_unique": [
      "118707795",
      "pka0439"
    ],
    "old_values": [
      "pka0439",
      "118707795"
    ],
    "old_values_raw": [
      "pka0439",
      "118707795"
    ],
    "removed_unique_values": [
      "pka0439"
    ],
    "retained_unique_values": [
      "118707795"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pka0439": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pka0439",
    "118707795"
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
    "118707795"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German pianist and composer",
    "label": "Franz Hünten"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "118707795": 1
      },
      "new_unique": [
        "118707795"
      ],
      "new_values": [
        "118707795"
      ],
      "new_values_raw": [
        "118707795"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "118707795": 1,
        "pka0439": 1
      },
      "old_unique": [
        "118707795",
        "pka0439"
      ],
      "old_values": [
        "pka0439",
        "118707795"
      ],
      "old_values_raw": [
        "pka0439",
        "118707795"
      ],
      "removed_unique_values": [
        "pka0439"
      ],
      "retained_unique_values": [
        "118707795"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pka0439": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pka0439"
      ],
      "report_values": [
        "118707795",
        "pka0439"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "118707795"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q31191189_2297172995`

| Field | Value |
|---|---|
| qid | Q31191189 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q31191189::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1079181458"] |
| classification_target_tokens | ["pk05896"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pk05896"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pk05896"
  ],
  "removed_unique_values": [
    "pk05896"
  ],
  "retained_support_tokens": [
    "1079181458"
  ],
  "retained_unique_values": [
    "1079181458"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "1079181458"
  ],
  "old_value": [
    "pk05896",
    "1079181458"
  ],
  "revision_id": 2297172995,
  "value": [
    "1079181458"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1079181458": 1
    },
    "new_unique": [
      "1079181458"
    ],
    "new_values": [
      "1079181458"
    ],
    "new_values_raw": [
      "1079181458"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1079181458": 1,
      "pk05896": 1
    },
    "old_unique": [
      "1079181458",
      "pk05896"
    ],
    "old_values": [
      "pk05896",
      "1079181458"
    ],
    "old_values_raw": [
      "pk05896",
      "1079181458"
    ],
    "removed_unique_values": [
      "pk05896"
    ],
    "retained_unique_values": [
      "1079181458"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pk05896": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pk05896",
    "1079181458"
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
    "1079181458"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "Spanish general in the Thirty Years' War (1584–1664)",
    "label": "Ernest of Isenburg-Grenzau"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "1079181458": 1
      },
      "new_unique": [
        "1079181458"
      ],
      "new_values": [
        "1079181458"
      ],
      "new_values_raw": [
        "1079181458"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1079181458": 1,
        "pk05896": 1
      },
      "old_unique": [
        "1079181458",
        "pk05896"
      ],
      "old_values": [
        "pk05896",
        "1079181458"
      ],
      "old_values_raw": [
        "pk05896",
        "1079181458"
      ],
      "removed_unique_values": [
        "pk05896"
      ],
      "retained_unique_values": [
        "1079181458"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pk05896": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pk05896"
      ],
      "report_values": [
        "1079181458",
        "pk05896"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "1079181458"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 011. `repair_Q50821_2297174102`

| Field | Value |
|---|---|
| qid | Q50821 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q50821::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["116372230"] |
| classification_target_tokens | ["pka0031"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pka0031"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pka0031"
  ],
  "removed_unique_values": [
    "pka0031"
  ],
  "retained_support_tokens": [
    "116372230"
  ],
  "retained_unique_values": [
    "116372230"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "116372230"
  ],
  "old_value": [
    "pka0031",
    "116372230"
  ],
  "revision_id": 2297174102,
  "value": [
    "116372230"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "116372230": 1
    },
    "new_unique": [
      "116372230"
    ],
    "new_values": [
      "116372230"
    ],
    "new_values_raw": [
      "116372230"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "116372230": 1,
      "pka0031": 1
    },
    "old_unique": [
      "116372230",
      "pka0031"
    ],
    "old_values": [
      "pka0031",
      "116372230"
    ],
    "old_values_raw": [
      "pka0031",
      "116372230"
    ],
    "removed_unique_values": [
      "pka0031"
    ],
    "retained_unique_values": [
      "116372230"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pka0031": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pka0031",
    "116372230"
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
    "116372230"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "Prussian, Saxon and Russian general (1778-1855)",
    "label": "Ernst Ludwig von Aster"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "116372230": 1
      },
      "new_unique": [
        "116372230"
      ],
      "new_values": [
        "116372230"
      ],
      "new_values_raw": [
        "116372230"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "116372230": 1,
        "pka0031": 1
      },
      "old_unique": [
        "116372230",
        "pka0031"
      ],
      "old_values": [
        "pka0031",
        "116372230"
      ],
      "old_values_raw": [
        "pka0031",
        "116372230"
      ],
      "removed_unique_values": [
        "pka0031"
      ],
      "retained_unique_values": [
        "116372230"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pka0031": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pka0031"
      ],
      "report_values": [
        "116372230",
        "pka0031"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "116372230"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 012. `repair_Q55680619_2297193357`

| Field | Value |
|---|---|
| qid | Q55680619 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q55680619::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1029798338"] |
| classification_target_tokens | ["pk03004"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pk03004"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pk03004"
  ],
  "removed_unique_values": [
    "pk03004"
  ],
  "retained_support_tokens": [
    "1029798338"
  ],
  "retained_unique_values": [
    "1029798338"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "1029798338"
  ],
  "old_value": [
    "pk03004",
    "1029798338"
  ],
  "revision_id": 2297193357,
  "value": [
    "1029798338"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1029798338": 1
    },
    "new_unique": [
      "1029798338"
    ],
    "new_values": [
      "1029798338"
    ],
    "new_values_raw": [
      "1029798338"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1029798338": 1,
      "pk03004": 1
    },
    "old_unique": [
      "1029798338",
      "pk03004"
    ],
    "old_values": [
      "pk03004",
      "1029798338"
    ],
    "old_values_raw": [
      "pk03004",
      "1029798338"
    ],
    "removed_unique_values": [
      "pk03004"
    ],
    "retained_unique_values": [
      "1029798338"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pk03004": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pk03004",
    "1029798338"
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
    "1029798338"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": null,
    "label": "Carsten Koppke"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "1029798338": 1
      },
      "new_unique": [
        "1029798338"
      ],
      "new_values": [
        "1029798338"
      ],
      "new_values_raw": [
        "1029798338"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1029798338": 1,
        "pk03004": 1
      },
      "old_unique": [
        "1029798338",
        "pk03004"
      ],
      "old_values": [
        "pk03004",
        "1029798338"
      ],
      "old_values_raw": [
        "pk03004",
        "1029798338"
      ],
      "removed_unique_values": [
        "pk03004"
      ],
      "retained_unique_values": [
        "1029798338"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pk03004": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pk03004"
      ],
      "report_values": [
        "1029798338",
        "pk03004"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "1029798338"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 013. `repair_Q55681930_2297197411`

| Field | Value |
|---|---|
| qid | Q55681930 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q55681930::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["105114793X"] |
| classification_target_tokens | ["pk00222"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pk00222"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pk00222"
  ],
  "removed_unique_values": [
    "pk00222"
  ],
  "retained_support_tokens": [
    "105114793X"
  ],
  "retained_unique_values": [
    "105114793X"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "105114793X"
  ],
  "old_value": [
    "pk00222",
    "105114793X"
  ],
  "revision_id": 2297197411,
  "value": [
    "105114793X"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "105114793X": 1
    },
    "new_unique": [
      "105114793X"
    ],
    "new_values": [
      "105114793X"
    ],
    "new_values_raw": [
      "105114793X"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "105114793X": 1,
      "pk00222": 1
    },
    "old_unique": [
      "105114793X",
      "pk00222"
    ],
    "old_values": [
      "pk00222",
      "105114793X"
    ],
    "old_values_raw": [
      "pk00222",
      "105114793X"
    ],
    "removed_unique_values": [
      "pk00222"
    ],
    "retained_unique_values": [
      "105114793X"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pk00222": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pk00222",
    "105114793X"
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
    "105114793X"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "onderwijzer",
    "label": "Hansgeorg Rack"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "105114793X": 1
      },
      "new_unique": [
        "105114793X"
      ],
      "new_values": [
        "105114793X"
      ],
      "new_values_raw": [
        "105114793X"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "105114793X": 1,
        "pk00222": 1
      },
      "old_unique": [
        "105114793X",
        "pk00222"
      ],
      "old_values": [
        "pk00222",
        "105114793X"
      ],
      "old_values_raw": [
        "pk00222",
        "105114793X"
      ],
      "removed_unique_values": [
        "pk00222"
      ],
      "retained_unique_values": [
        "105114793X"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pk00222": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pk00222"
      ],
      "report_values": [
        "105114793X",
        "pk00222"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "105114793X"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 014. `repair_Q55681948_2297183181`

| Field | Value |
|---|---|
| qid | Q55681948 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q55681948::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1051150027"] |
| classification_target_tokens | ["pk00726"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pk00726"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pk00726"
  ],
  "removed_unique_values": [
    "pk00726"
  ],
  "retained_support_tokens": [
    "1051150027"
  ],
  "retained_unique_values": [
    "1051150027"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "1051150027"
  ],
  "old_value": [
    "pk00726",
    "1051150027"
  ],
  "revision_id": 2297183181,
  "value": [
    "1051150027"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1051150027": 1
    },
    "new_unique": [
      "1051150027"
    ],
    "new_values": [
      "1051150027"
    ],
    "new_values_raw": [
      "1051150027"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1051150027": 1,
      "pk00726": 1
    },
    "old_unique": [
      "1051150027",
      "pk00726"
    ],
    "old_values": [
      "pk00726",
      "1051150027"
    ],
    "old_values_raw": [
      "pk00726",
      "1051150027"
    ],
    "removed_unique_values": [
      "pk00726"
    ],
    "retained_unique_values": [
      "1051150027"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pk00726": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pk00726",
    "1051150027"
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
    "1051150027"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "onderwijzer",
    "label": "Karl Wilhelm Johannes Goetzke"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "1051150027": 1
      },
      "new_unique": [
        "1051150027"
      ],
      "new_values": [
        "1051150027"
      ],
      "new_values_raw": [
        "1051150027"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1051150027": 1,
        "pk00726": 1
      },
      "old_unique": [
        "1051150027",
        "pk00726"
      ],
      "old_values": [
        "pk00726",
        "1051150027"
      ],
      "old_values_raw": [
        "pk00726",
        "1051150027"
      ],
      "removed_unique_values": [
        "pk00726"
      ],
      "retained_unique_values": [
        "1051150027"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pk00726": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pk00726"
      ],
      "report_values": [
        "1051150027",
        "pk00726"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "1051150027"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 015. `repair_Q55682053_2297173199`

| Field | Value |
|---|---|
| qid | Q55682053 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q55682053::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1051177413"] |
| classification_target_tokens | ["pka0051"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pka0051"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pka0051"
  ],
  "removed_unique_values": [
    "pka0051"
  ],
  "retained_support_tokens": [
    "1051177413"
  ],
  "retained_unique_values": [
    "1051177413"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "1051177413"
  ],
  "old_value": [
    "pka0051",
    "1051177413"
  ],
  "revision_id": 2297173199,
  "value": [
    "1051177413"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1051177413": 1
    },
    "new_unique": [
      "1051177413"
    ],
    "new_values": [
      "1051177413"
    ],
    "new_values_raw": [
      "1051177413"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1051177413": 1,
      "pka0051": 1
    },
    "old_unique": [
      "1051177413",
      "pka0051"
    ],
    "old_values": [
      "pka0051",
      "1051177413"
    ],
    "old_values_raw": [
      "pka0051",
      "1051177413"
    ],
    "removed_unique_values": [
      "pka0051"
    ],
    "retained_unique_values": [
      "1051177413"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pka0051": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pka0051",
    "1051177413"
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
    "1051177413"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "redacteur",
    "label": "Friedrich Bahne"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "1051177413": 1
      },
      "new_unique": [
        "1051177413"
      ],
      "new_values": [
        "1051177413"
      ],
      "new_values_raw": [
        "1051177413"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1051177413": 1,
        "pka0051": 1
      },
      "old_unique": [
        "1051177413",
        "pka0051"
      ],
      "old_values": [
        "pka0051",
        "1051177413"
      ],
      "old_values_raw": [
        "pka0051",
        "1051177413"
      ],
      "removed_unique_values": [
        "pka0051"
      ],
      "retained_unique_values": [
        "1051177413"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pka0051": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pka0051"
      ],
      "report_values": [
        "1051177413",
        "pka0051"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "1051177413"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 016. `repair_Q55682097_2297173221`

| Field | Value |
|---|---|
| qid | Q55682097 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q55682097::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1051178894"] |
| classification_target_tokens | ["pka0449"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pka0449"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pka0449"
  ],
  "removed_unique_values": [
    "pka0449"
  ],
  "retained_support_tokens": [
    "1051178894"
  ],
  "retained_unique_values": [
    "1051178894"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "1051178894"
  ],
  "old_value": [
    "pka0449",
    "1051178894"
  ],
  "revision_id": 2297173221,
  "value": [
    "1051178894"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1051178894": 1
    },
    "new_unique": [
      "1051178894"
    ],
    "new_values": [
      "1051178894"
    ],
    "new_values_raw": [
      "1051178894"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1051178894": 1,
      "pka0449": 1
    },
    "old_unique": [
      "1051178894",
      "pka0449"
    ],
    "old_values": [
      "pka0449",
      "1051178894"
    ],
    "old_values_raw": [
      "pka0449",
      "1051178894"
    ],
    "removed_unique_values": [
      "pka0449"
    ],
    "retained_unique_values": [
      "1051178894"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pka0449": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pka0449",
    "1051178894"
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
    "1051178894"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "monnik",
    "label": "Arsenius Jakobs"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "1051178894": 1
      },
      "new_unique": [
        "1051178894"
      ],
      "new_values": [
        "1051178894"
      ],
      "new_values_raw": [
        "1051178894"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1051178894": 1,
        "pka0449": 1
      },
      "old_unique": [
        "1051178894",
        "pka0449"
      ],
      "old_values": [
        "pka0449",
        "1051178894"
      ],
      "old_values_raw": [
        "pka0449",
        "1051178894"
      ],
      "removed_unique_values": [
        "pka0449"
      ],
      "retained_unique_values": [
        "1051178894"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pka0449": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pka0449"
      ],
      "report_values": [
        "1051178894",
        "pka0449"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "1051178894"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 017. `repair_Q55682115_2297176580`

| Field | Value |
|---|---|
| qid | Q55682115 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q55682115::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1051179718"] |
| classification_target_tokens | ["pka0660"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pka0660"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pka0660"
  ],
  "removed_unique_values": [
    "pka0660"
  ],
  "retained_support_tokens": [
    "1051179718"
  ],
  "retained_unique_values": [
    "1051179718"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "1051179718"
  ],
  "old_value": [
    "pka0660",
    "1051179718"
  ],
  "revision_id": 2297176580,
  "value": [
    "1051179718"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1051179718": 1
    },
    "new_unique": [
      "1051179718"
    ],
    "new_values": [
      "1051179718"
    ],
    "new_values_raw": [
      "1051179718"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1051179718": 1,
      "pka0660": 1
    },
    "old_unique": [
      "1051179718",
      "pka0660"
    ],
    "old_values": [
      "pka0660",
      "1051179718"
    ],
    "old_values_raw": [
      "pka0660",
      "1051179718"
    ],
    "removed_unique_values": [
      "pka0660"
    ],
    "retained_unique_values": [
      "1051179718"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pka0660": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pka0660",
    "1051179718"
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
    "1051179718"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": null,
    "label": "Josef Mündnich"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "1051179718": 1
      },
      "new_unique": [
        "1051179718"
      ],
      "new_values": [
        "1051179718"
      ],
      "new_values_raw": [
        "1051179718"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1051179718": 1,
        "pka0660": 1
      },
      "old_unique": [
        "1051179718",
        "pka0660"
      ],
      "old_values": [
        "pka0660",
        "1051179718"
      ],
      "old_values_raw": [
        "pka0660",
        "1051179718"
      ],
      "removed_unique_values": [
        "pka0660"
      ],
      "retained_unique_values": [
        "1051179718"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pka0660": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pka0660"
      ],
      "report_values": [
        "1051179718",
        "pka0660"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "1051179718"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 018. `repair_Q55682145_2297195571`

| Field | Value |
|---|---|
| qid | Q55682145 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q55682145::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1051180694"] |
| classification_target_tokens | ["pka0876"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pka0876"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pka0876"
  ],
  "removed_unique_values": [
    "pka0876"
  ],
  "retained_support_tokens": [
    "1051180694"
  ],
  "retained_unique_values": [
    "1051180694"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "1051180694"
  ],
  "old_value": [
    "pka0876",
    "1051180694"
  ],
  "revision_id": 2297195571,
  "value": [
    "1051180694"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1051180694": 1
    },
    "new_unique": [
      "1051180694"
    ],
    "new_values": [
      "1051180694"
    ],
    "new_values_raw": [
      "1051180694"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1051180694": 1,
      "pka0876": 1
    },
    "old_unique": [
      "1051180694",
      "pka0876"
    ],
    "old_values": [
      "pka0876",
      "1051180694"
    ],
    "old_values_raw": [
      "pka0876",
      "1051180694"
    ],
    "removed_unique_values": [
      "pka0876"
    ],
    "retained_unique_values": [
      "1051180694"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pka0876": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pka0876",
    "1051180694"
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
    "1051180694"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "actor",
    "label": "Heinrich Alex Gustav Turrian"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "1051180694": 1
      },
      "new_unique": [
        "1051180694"
      ],
      "new_values": [
        "1051180694"
      ],
      "new_values_raw": [
        "1051180694"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1051180694": 1,
        "pka0876": 1
      },
      "old_unique": [
        "1051180694",
        "pka0876"
      ],
      "old_values": [
        "pka0876",
        "1051180694"
      ],
      "old_values_raw": [
        "pka0876",
        "1051180694"
      ],
      "removed_unique_values": [
        "pka0876"
      ],
      "retained_unique_values": [
        "1051180694"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pka0876": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pka0876"
      ],
      "report_values": [
        "1051180694",
        "pka0876"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "1051180694"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 019. `repair_Q55682184_2297173272`

| Field | Value |
|---|---|
| qid | Q55682184 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q55682184::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1051189918"] |
| classification_target_tokens | ["pkd0203"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pkd0203"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pkd0203"
  ],
  "removed_unique_values": [
    "pkd0203"
  ],
  "retained_support_tokens": [
    "1051189918"
  ],
  "retained_unique_values": [
    "1051189918"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "1051189918"
  ],
  "old_value": [
    "pkd0203",
    "1051189918"
  ],
  "revision_id": 2297173272,
  "value": [
    "1051189918"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1051189918": 1
    },
    "new_unique": [
      "1051189918"
    ],
    "new_values": [
      "1051189918"
    ],
    "new_values_raw": [
      "1051189918"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1051189918": 1,
      "pkd0203": 1
    },
    "old_unique": [
      "1051189918",
      "pkd0203"
    ],
    "old_values": [
      "pkd0203",
      "1051189918"
    ],
    "old_values_raw": [
      "pkd0203",
      "1051189918"
    ],
    "removed_unique_values": [
      "pkd0203"
    ],
    "retained_unique_values": [
      "1051189918"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pkd0203": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pkd0203",
    "1051189918"
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
    "1051189918"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "Kath. Pfarrer",
    "label": "Joseph Ebertz"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "1051189918": 1
      },
      "new_unique": [
        "1051189918"
      ],
      "new_values": [
        "1051189918"
      ],
      "new_values_raw": [
        "1051189918"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1051189918": 1,
        "pkd0203": 1
      },
      "old_unique": [
        "1051189918",
        "pkd0203"
      ],
      "old_values": [
        "pkd0203",
        "1051189918"
      ],
      "old_values_raw": [
        "pkd0203",
        "1051189918"
      ],
      "removed_unique_values": [
        "pkd0203"
      ],
      "retained_unique_values": [
        "1051189918"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pkd0203": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pkd0203"
      ],
      "report_values": [
        "1051189918",
        "pkd0203"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "1051189918"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 020. `repair_Q55682214_2297173284`

| Field | Value |
|---|---|
| qid | Q55682214 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q55682214::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1051193907"] |
| classification_target_tokens | ["pkd0501"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pkd0501"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pkd0501"
  ],
  "removed_unique_values": [
    "pkd0501"
  ],
  "retained_support_tokens": [
    "1051193907"
  ],
  "retained_unique_values": [
    "1051193907"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "1051193907"
  ],
  "old_value": [
    "pkd0501",
    "1051193907"
  ],
  "revision_id": 2297173284,
  "value": [
    "1051193907"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1051193907": 1
    },
    "new_unique": [
      "1051193907"
    ],
    "new_values": [
      "1051193907"
    ],
    "new_values_raw": [
      "1051193907"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1051193907": 1,
      "pkd0501": 1
    },
    "old_unique": [
      "1051193907",
      "pkd0501"
    ],
    "old_values": [
      "pkd0501",
      "1051193907"
    ],
    "old_values_raw": [
      "pkd0501",
      "1051193907"
    ],
    "removed_unique_values": [
      "pkd0501"
    ],
    "retained_unique_values": [
      "1051193907"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pkd0501": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pkd0501",
    "1051193907"
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
    "1051193907"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "Kath. Pfarrer",
    "label": "Johann Adam Klütsch"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "1051193907": 1
      },
      "new_unique": [
        "1051193907"
      ],
      "new_values": [
        "1051193907"
      ],
      "new_values_raw": [
        "1051193907"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1051193907": 1,
        "pkd0501": 1
      },
      "old_unique": [
        "1051193907",
        "pkd0501"
      ],
      "old_values": [
        "pkd0501",
        "1051193907"
      ],
      "old_values_raw": [
        "pkd0501",
        "1051193907"
      ],
      "removed_unique_values": [
        "pkd0501"
      ],
      "retained_unique_values": [
        "1051193907"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pkd0501": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pkd0501"
      ],
      "report_values": [
        "1051193907",
        "pkd0501"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "1051193907"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 021. `repair_Q55682260_2297186727`

| Field | Value |
|---|---|
| qid | Q55682260 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q55682260::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1051201063"] |
| classification_target_tokens | ["pkd0950"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pkd0950"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pkd0950"
  ],
  "removed_unique_values": [
    "pkd0950"
  ],
  "retained_support_tokens": [
    "1051201063"
  ],
  "retained_unique_values": [
    "1051201063"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "1051201063"
  ],
  "old_value": [
    "pkd0950",
    "1051201063"
  ],
  "revision_id": 2297186727,
  "value": [
    "1051201063"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "1051201063": 1
    },
    "new_unique": [
      "1051201063"
    ],
    "new_values": [
      "1051201063"
    ],
    "new_values_raw": [
      "1051201063"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "1051201063": 1,
      "pkd0950": 1
    },
    "old_unique": [
      "1051201063",
      "pkd0950"
    ],
    "old_values": [
      "pkd0950",
      "1051201063"
    ],
    "old_values_raw": [
      "pkd0950",
      "1051201063"
    ],
    "removed_unique_values": [
      "pkd0950"
    ],
    "retained_unique_values": [
      "1051201063"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pkd0950": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pkd0950",
    "1051201063"
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
    "1051201063"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "Kath. Pfarrer",
    "label": "Johannes Theis"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "1051201063": 1
      },
      "new_unique": [
        "1051201063"
      ],
      "new_values": [
        "1051201063"
      ],
      "new_values_raw": [
        "1051201063"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "1051201063": 1,
        "pkd0950": 1
      },
      "old_unique": [
        "1051201063",
        "pkd0950"
      ],
      "old_values": [
        "pkd0950",
        "1051201063"
      ],
      "old_values_raw": [
        "pkd0950",
        "1051201063"
      ],
      "removed_unique_values": [
        "pkd0950"
      ],
      "retained_unique_values": [
        "1051201063"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pkd0950": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pkd0950"
      ],
      "report_values": [
        "1051201063",
        "pkd0950"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "1051201063"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 022. `repair_Q821147_2297167109`

| Field | Value |
|---|---|
| qid | Q821147 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q821147::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["118754270"] |
| classification_target_tokens | ["pma0475"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pma0475"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pma0475"
  ],
  "removed_unique_values": [
    "pma0475"
  ],
  "retained_support_tokens": [
    "118754270"
  ],
  "retained_unique_values": [
    "118754270"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "118754270"
  ],
  "old_value": [
    "pma0475",
    "118754270"
  ],
  "revision_id": 2297167109,
  "value": [
    "118754270"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "118754270": 1
    },
    "new_unique": [
      "118754270"
    ],
    "new_values": [
      "118754270"
    ],
    "new_values_raw": [
      "118754270"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "118754270": 1,
      "pma0475": 1
    },
    "old_unique": [
      "118754270",
      "pma0475"
    ],
    "old_values": [
      "pma0475",
      "118754270"
    ],
    "old_values_raw": [
      "pma0475",
      "118754270"
    ],
    "removed_unique_values": [
      "pma0475"
    ],
    "retained_unique_values": [
      "118754270"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pma0475": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pma0475",
    "118754270"
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
    "118754270"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German artist (1699-1756)",
    "label": "Christoph Thomas Scheffler"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "118754270": 1
      },
      "new_unique": [
        "118754270"
      ],
      "new_values": [
        "118754270"
      ],
      "new_values_raw": [
        "118754270"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "118754270": 1,
        "pma0475": 1
      },
      "old_unique": [
        "118754270",
        "pma0475"
      ],
      "old_values": [
        "pma0475",
        "118754270"
      ],
      "old_values_raw": [
        "pma0475",
        "118754270"
      ],
      "removed_unique_values": [
        "pma0475"
      ],
      "retained_unique_values": [
        "118754270"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pma0475": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pma0475"
      ],
      "report_values": [
        "118754270",
        "pma0475"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "118754270"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 023. `repair_Q825149_2297173730`

| Field | Value |
|---|---|
| qid | Q825149 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q825149::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["143787411"] |
| classification_target_tokens | ["pka0333"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pka0333"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pka0333"
  ],
  "removed_unique_values": [
    "pka0333"
  ],
  "retained_support_tokens": [
    "143787411"
  ],
  "retained_unique_values": [
    "143787411"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "143787411"
  ],
  "old_value": [
    "pka0333",
    "143787411"
  ],
  "revision_id": 2297173730,
  "value": [
    "143787411"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "143787411": 1
    },
    "new_unique": [
      "143787411"
    ],
    "new_values": [
      "143787411"
    ],
    "new_values_raw": [
      "143787411"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "143787411": 1,
      "pka0333": 1
    },
    "old_unique": [
      "143787411",
      "pka0333"
    ],
    "old_values": [
      "pka0333",
      "143787411"
    ],
    "old_values_raw": [
      "pka0333",
      "143787411"
    ],
    "removed_unique_values": [
      "pka0333"
    ],
    "retained_unique_values": [
      "143787411"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pka0333": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pka0333",
    "143787411"
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
    "143787411"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German politician (1906-1981)",
    "label": "Bernhard Günther"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "143787411": 1
      },
      "new_unique": [
        "143787411"
      ],
      "new_values": [
        "143787411"
      ],
      "new_values_raw": [
        "143787411"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "143787411": 1,
        "pka0333": 1
      },
      "old_unique": [
        "143787411",
        "pka0333"
      ],
      "old_values": [
        "pka0333",
        "143787411"
      ],
      "old_values_raw": [
        "pka0333",
        "143787411"
      ],
      "removed_unique_values": [
        "pka0333"
      ],
      "retained_unique_values": [
        "143787411"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pka0333": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pka0333"
      ],
      "report_values": [
        "143787411",
        "pka0333"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "143787411"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 024. `repair_Q90269_2297175264`

| Field | Value |
|---|---|
| qid | Q90269 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q90269::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["116729228"] |
| classification_target_tokens | ["pk00152"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pk00152"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pk00152"
  ],
  "removed_unique_values": [
    "pk00152"
  ],
  "retained_support_tokens": [
    "116729228"
  ],
  "retained_unique_values": [
    "116729228"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "116729228"
  ],
  "old_value": [
    "pk00152",
    "116729228"
  ],
  "revision_id": 2297175264,
  "value": [
    "116729228"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "116729228": 1
    },
    "new_unique": [
      "116729228"
    ],
    "new_values": [
      "116729228"
    ],
    "new_values_raw": [
      "116729228"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "116729228": 1,
      "pk00152": 1
    },
    "old_unique": [
      "116729228",
      "pk00152"
    ],
    "old_values": [
      "pk00152",
      "116729228"
    ],
    "old_values_raw": [
      "pk00152",
      "116729228"
    ],
    "removed_unique_values": [
      "pk00152"
    ],
    "retained_unique_values": [
      "116729228"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pk00152": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pk00152",
    "116729228"
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
    "116729228"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German historian and writer (1805–1852)",
    "label": "Guido Görres"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "116729228": 1
      },
      "new_unique": [
        "116729228"
      ],
      "new_values": [
        "116729228"
      ],
      "new_values_raw": [
        "116729228"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "116729228": 1,
        "pk00152": 1
      },
      "old_unique": [
        "116729228",
        "pk00152"
      ],
      "old_values": [
        "pk00152",
        "116729228"
      ],
      "old_values_raw": [
        "pk00152",
        "116729228"
      ],
      "removed_unique_values": [
        "pk00152"
      ],
      "retained_unique_values": [
        "116729228"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pk00152": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pk00152"
      ],
      "report_values": [
        "116729228",
        "pk00152"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "116729228"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 025. `repair_Q90720_2297190110`

| Field | Value |
|---|---|
| qid | Q90720 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q90720::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["118503200"] |
| classification_target_tokens | ["ps01037"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "ps01037"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "ps01037"
  ],
  "removed_unique_values": [
    "ps01037"
  ],
  "retained_support_tokens": [
    "118503200"
  ],
  "retained_unique_values": [
    "118503200"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "118503200"
  ],
  "old_value": [
    "ps01037",
    "118503200"
  ],
  "revision_id": 2297190110,
  "value": [
    "118503200"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "118503200": 1
    },
    "new_unique": [
      "118503200"
    ],
    "new_values": [
      "118503200"
    ],
    "new_values_raw": [
      "118503200"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "118503200": 1,
      "ps01037": 1
    },
    "old_unique": [
      "118503200",
      "ps01037"
    ],
    "old_values": [
      "ps01037",
      "118503200"
    ],
    "old_values_raw": [
      "ps01037",
      "118503200"
    ],
    "removed_unique_values": [
      "ps01037"
    ],
    "retained_unique_values": [
      "118503200"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "ps01037": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "ps01037",
    "118503200"
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
    "118503200"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German feminist (1817–1884)",
    "label": "Mathilde Franziska Anneke"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "118503200": 1
      },
      "new_unique": [
        "118503200"
      ],
      "new_values": [
        "118503200"
      ],
      "new_values_raw": [
        "118503200"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "118503200": 1,
        "ps01037": 1
      },
      "old_unique": [
        "118503200",
        "ps01037"
      ],
      "old_values": [
        "ps01037",
        "118503200"
      ],
      "old_values_raw": [
        "ps01037",
        "118503200"
      ],
      "removed_unique_values": [
        "ps01037"
      ],
      "retained_unique_values": [
        "118503200"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "ps01037": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "ps01037"
      ],
      "report_values": [
        "118503200",
        "ps01037"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "118503200"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---

## 026. `repair_Q91627_2297168815`

| Field | Value |
|---|---|
| qid | Q91627 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_VALUE_PRUNING / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_value_pruning |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | format_value_pruning |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q91627::P8748 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["128906561"] |
| classification_target_tokens | ["pk00165"] |
| classification_target_reason | subset deletion is explained by removed values, not retained values |
| decision_branch | format_value_pruning |
| rationale | Subset repair removes the value indicated by a format violation. |
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
    "pk00165"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pk00165"
  ],
  "removed_unique_values": [
    "pk00165"
  ],
  "retained_support_tokens": [
    "128906561"
  ],
  "retained_unique_values": [
    "128906561"
  ],
  "semantic_action": "DELETE_SUBSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "format_value_pruning",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "128906561"
  ],
  "old_value": [
    "pk00165",
    "128906561"
  ],
  "revision_id": 2297168815,
  "value": [
    "128906561"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "128906561": 1
    },
    "new_unique": [
      "128906561"
    ],
    "new_values": [
      "128906561"
    ],
    "new_values_raw": [
      "128906561"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "128906561": 1,
      "pk00165": 1
    },
    "old_unique": [
      "128906561",
      "pk00165"
    ],
    "old_values": [
      "pk00165",
      "128906561"
    ],
    "old_values_raw": [
      "pk00165",
      "128906561"
    ],
    "removed_unique_values": [
      "pk00165"
    ],
    "retained_unique_values": [
      "128906561"
    ],
    "semantic_action": "DELETE_SUBSET",
    "value_multiplicity_changes": {
      "pk00165": {
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
  "report_fix_date": "2025-01-13T04:30:19",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8748",
  "report_revision_new": 2297608175,
  "report_revision_old": 2297241617,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "pk00165",
    "128906561"
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
    "128906561"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German photographer",
    "label": "Karsten Thormaehlen"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "128906561": 1
      },
      "new_unique": [
        "128906561"
      ],
      "new_values": [
        "128906561"
      ],
      "new_values_raw": [
        "128906561"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "128906561": 1,
        "pk00165": 1
      },
      "old_unique": [
        "128906561",
        "pk00165"
      ],
      "old_values": [
        "pk00165",
        "128906561"
      ],
      "old_values_raw": [
        "pk00165",
        "128906561"
      ],
      "removed_unique_values": [
        "pk00165"
      ],
      "retained_unique_values": [
        "128906561"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "pk00165": {
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
      "reason": "removed_values_reported_as_offending_values",
      "regexes_present": true,
      "removed_fail_regex": true,
      "removed_reported": true,
      "removed_values": [
        "pk00165"
      ],
      "report_values": [
        "128906561",
        "pk00165"
      ],
      "retained_pass_regex": true,
      "retained_values": [
        "128906561"
      ]
    },
    "kind": "FORMAT_VALUE_PRUNING",
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
    "result": "format_value_pruning",
    "step": "branch"
  }
]
```

---
