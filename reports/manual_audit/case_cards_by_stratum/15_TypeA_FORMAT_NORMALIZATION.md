# TypeA_FORMAT_NORMALIZATION

Cases: 26

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q105176924_2425187483`

| Field | Value |
|---|---|
| qid | Q105176924 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q105176924::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["26835617", "26835656"] |
| classification_target_tokens | ["SCHEMBL26835656", "26835656"] |
| classification_target_reason | mixed update classification uses the deterministic changed pair while ignoring retained values |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "26835656"
  ],
  "classification_target_reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
  "classification_target_role": "changed_pair",
  "classification_target_tokens": [
    "SCHEMBL26835656",
    "26835656"
  ],
  "new_changed_value": "26835656",
  "old_changed_value": "SCHEMBL26835656",
  "removed_target_tokens": [
    "SCHEMBL26835656"
  ],
  "removed_unique_values": [
    "SCHEMBL26835656"
  ],
  "retained_support_tokens": [
    "26835617"
  ],
  "retained_unique_values": [
    "26835617"
  ],
  "semantic_action": "MIXED_UPDATE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "26835617",
    "26835656"
  ],
  "old_value": [
    "26835617",
    "SCHEMBL26835656"
  ],
  "revision_id": 2425187483,
  "value": [
    "26835617",
    "26835656"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "26835656"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "26835617": 1,
      "26835656": 1
    },
    "new_unique": [
      "26835617",
      "26835656"
    ],
    "new_values": [
      "26835617",
      "26835656"
    ],
    "new_values_raw": [
      "26835617",
      "26835656"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "26835617": 1,
      "SCHEMBL26835656": 1
    },
    "old_unique": [
      "26835617",
      "SCHEMBL26835656"
    ],
    "old_values": [
      "26835617",
      "SCHEMBL26835656"
    ],
    "old_values_raw": [
      "26835617",
      "SCHEMBL26835656"
    ],
    "removed_unique_values": [
      "SCHEMBL26835656"
    ],
    "retained_unique_values": [
      "26835617"
    ],
    "semantic_action": "MIXED_UPDATE",
    "value_multiplicity_changes": {
      "26835656": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL26835656": {
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
  "report_fix_date": "2025-11-04T10:40:02",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425435358,
  "report_revision_old": 2411348025,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "26835617",
    "SCHEMBL26835656"
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
    "26835617",
    "26835656"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "group of stereoisomers with the chemical formula C₂₄H₂₂O₁₅",
    "label": "Quercetin 3-(6''-malonyl-glucoside)"
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
        "26835656"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "26835617": 1,
        "26835656": 1
      },
      "new_unique": [
        "26835617",
        "26835656"
      ],
      "new_values": [
        "26835617",
        "26835656"
      ],
      "new_values_raw": [
        "26835617",
        "26835656"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "26835617": 1,
        "SCHEMBL26835656": 1
      },
      "old_unique": [
        "26835617",
        "SCHEMBL26835656"
      ],
      "old_values": [
        "26835617",
        "SCHEMBL26835656"
      ],
      "old_values_raw": [
        "26835617",
        "SCHEMBL26835656"
      ],
      "removed_unique_values": [
        "SCHEMBL26835656"
      ],
      "retained_unique_values": [
        "26835617"
      ],
      "semantic_action": "MIXED_UPDATE",
      "value_multiplicity_changes": {
        "26835656": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL26835656": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "MIXED_UPDATE",
    "step": "value_delta"
  },
  {
    "detail": {
      "mixed_update": true,
      "new_changed": "26835656",
      "new_pass_regex": true,
      "new_value": "26835656",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL26835656",
      "old_pass_regex": false,
      "old_value": "SCHEMBL26835656",
      "regexes_present": true,
      "retained_values": [
        "26835617"
      ]
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q110331423_2442356891`

| Field | Value |
|---|---|
| qid | Q110331423 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_category_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q110331423::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Cemetery of Asnelles"] |
| classification_target_tokens | ["Cemetery of Asnelles"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Cemetery of Asnelles"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Cemetery of Asnelles"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Category:Cemetery of Asnelles"
  ],
  "removed_unique_values": [
    "Category:Cemetery of Asnelles"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_category_prefix",
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
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Cemetery of Asnelles"
  ],
  "old_value": [
    "Category:Cemetery of Asnelles"
  ],
  "revision_id": 2442356891,
  "value": [
    "Cemetery of Asnelles"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Cemetery of Asnelles"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Cemetery of Asnelles": 1
    },
    "new_unique": [
      "Cemetery of Asnelles"
    ],
    "new_values": [
      "Cemetery of Asnelles"
    ],
    "new_values_raw": [
      "Cemetery of Asnelles"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Category:Cemetery of Asnelles": 1
    },
    "old_unique": [
      "Category:Cemetery of Asnelles"
    ],
    "old_values": [
      "Category:Cemetery of Asnelles"
    ],
    "old_values_raw": [
      "Category:Cemetery of Asnelles"
    ],
    "removed_unique_values": [
      "Category:Cemetery of Asnelles"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Category:Cemetery of Asnelles": {
        "new": 0,
        "old": 1
      },
      "Cemetery of Asnelles": {
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
    "Category:Cemetery of Asnelles"
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
    "Cemetery of Asnelles"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "description": "cemetery located in Calvados, in France",
    "label": "cimetière d'Asnelles (Chemin du Magasin)"
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
        "Cemetery of Asnelles"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Cemetery of Asnelles": 1
      },
      "new_unique": [
        "Cemetery of Asnelles"
      ],
      "new_values": [
        "Cemetery of Asnelles"
      ],
      "new_values_raw": [
        "Cemetery of Asnelles"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Category:Cemetery of Asnelles": 1
      },
      "old_unique": [
        "Category:Cemetery of Asnelles"
      ],
      "old_values": [
        "Category:Cemetery of Asnelles"
      ],
      "old_values_raw": [
        "Category:Cemetery of Asnelles"
      ],
      "removed_unique_values": [
        "Category:Cemetery of Asnelles"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Category:Cemetery of Asnelles": {
          "new": 0,
          "old": 1
        },
        "Cemetery of Asnelles": {
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
      "new_changed": "Cemetery of Asnelles",
      "new_pass_regex": true,
      "new_value": "Cemetery of Asnelles",
      "normalization_kind": "strip_category_prefix",
      "normalization_rule": "strip_category_prefix",
      "old_changed": "Category:Cemetery of Asnelles",
      "old_pass_regex": false,
      "old_value": "Category:Cemetery of Asnelles",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q137423767_2444114469`

| Field | Value |
|---|---|
| qid | Q137423767 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | format |
| classification_rule_subfamily | collapse_whitespace |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q137423767::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Comics artists from Costa Rica"] |
| classification_target_tokens | ["Comics artists from Costa Rica"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | rule_deterministic |
| rationale | Rule-deterministic format constraint fix. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Comics artists from Costa Rica"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Comics artists from Costa Rica"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Comics artists from Costa  Rica"
  ],
  "removed_unique_values": [
    "Comics artists from Costa  Rica"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "collapse_whitespace",
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
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Comics artists from Costa Rica"
  ],
  "old_value": [
    "Comics artists from Costa  Rica"
  ],
  "revision_id": 2444114469,
  "value": [
    "Comics artists from Costa Rica"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Comics artists from Costa Rica"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Comics artists from Costa Rica": 1
    },
    "new_unique": [
      "Comics artists from Costa Rica"
    ],
    "new_values": [
      "Comics artists from Costa Rica"
    ],
    "new_values_raw": [
      "Comics artists from Costa Rica"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Comics artists from Costa  Rica": 1
    },
    "old_unique": [
      "Comics artists from Costa  Rica"
    ],
    "old_values": [
      "Comics artists from Costa  Rica"
    ],
    "old_values_raw": [
      "Comics artists from Costa  Rica"
    ],
    "removed_unique_values": [
      "Comics artists from Costa  Rica"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Comics artists from Costa  Rica": {
        "new": 0,
        "old": 1
      },
      "Comics artists from Costa Rica": {
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
  "report_fix_date": "2025-12-21T10:58:44",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2444891710,
  "report_revision_old": 2444464305,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Comics artists from Costa  Rica"
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
    "Comics artists from Costa Rica"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "label": "Category:Costa Rican comics artists"
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
      "constraint_type": {
        "label": "format constraint",
        "qid": "Q21502404"
      },
      "new_value": "Comics artists from Costa Rica",
      "normalization_kind": "collapse_whitespace",
      "old_value": "Comics artists from Costa  Rica",
      "pre_repair_source": "repair_target.old_value",
      "signal": "L4_constraints",
      "truth_source": "replacement_new"
    },
    "kind": "FORMAT",
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
    "result": "rule_deterministic",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q15502554_2086442380`

| Field | Value |
|---|---|
| qid | Q15502554 |
| property | P3795 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_trailing_slash |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q15502554::P3795 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["ARGUNI"] |
| classification_target_tokens | ["ARGUNI"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "ARGUNI"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "ARGUNI"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "ARGUNI/"
  ],
  "removed_unique_values": [
    "ARGUNI/"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_trailing_slash",
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
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "ARGUNI"
  ],
  "old_value": [
    "ARGUNI/"
  ],
  "revision_id": 2086442380,
  "value": [
    "ARGUNI"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "ARGUNI"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "ARGUNI": 1
    },
    "new_unique": [
      "ARGUNI"
    ],
    "new_values": [
      "ARGUNI"
    ],
    "new_values_raw": [
      "ARGUNI"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "ARGUNI/": 1
    },
    "old_unique": [
      "ARGUNI/"
    ],
    "old_values": [
      "ARGUNI/"
    ],
    "old_values_raw": [
      "ARGUNI/"
    ],
    "removed_unique_values": [
      "ARGUNI/"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "ARGUNI": {
        "new": 1,
        "old": 0
      },
      "ARGUNI/": {
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
  "report_fix_date": "2024-02-26T12:55:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P3795",
  "report_revision_new": 2087785506,
  "report_revision_old": 2082360484,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "ARGUNI/"
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
    "ARGUNI"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a plant taxon or cultivar in the Flora of Israel Online database",
    "label": "Flora of Israel Online plant ID"
  },
  "qid": {
    "description": "species of plant",
    "label": "Argyrolobium uniflorum"
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
      "added_unique_values": [
        "ARGUNI"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "ARGUNI": 1
      },
      "new_unique": [
        "ARGUNI"
      ],
      "new_values": [
        "ARGUNI"
      ],
      "new_values_raw": [
        "ARGUNI"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "ARGUNI/": 1
      },
      "old_unique": [
        "ARGUNI/"
      ],
      "old_values": [
        "ARGUNI/"
      ],
      "old_values_raw": [
        "ARGUNI/"
      ],
      "removed_unique_values": [
        "ARGUNI/"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "ARGUNI": {
          "new": 1,
          "old": 0
        },
        "ARGUNI/": {
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
      "new_changed": "ARGUNI",
      "new_pass_regex": true,
      "new_value": "ARGUNI",
      "normalization_kind": "strip_trailing_slash",
      "normalization_rule": "strip_trailing_slash",
      "old_changed": "ARGUNI/",
      "old_pass_regex": false,
      "old_value": "ARGUNI/",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q15564178_2086445496`

| Field | Value |
|---|---|
| qid | Q15564178 |
| property | P3795 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_trailing_slash |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q15564178::P3795 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["CREACU"] |
| classification_target_tokens | ["CREACU"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "CREACU"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "CREACU"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "CREACU/"
  ],
  "removed_unique_values": [
    "CREACU/"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_trailing_slash",
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
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "CREACU"
  ],
  "old_value": [
    "CREACU/"
  ],
  "revision_id": 2086445496,
  "value": [
    "CREACU"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "CREACU"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "CREACU": 1
    },
    "new_unique": [
      "CREACU"
    ],
    "new_values": [
      "CREACU"
    ],
    "new_values_raw": [
      "CREACU"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "CREACU/": 1
    },
    "old_unique": [
      "CREACU/"
    ],
    "old_values": [
      "CREACU/"
    ],
    "old_values_raw": [
      "CREACU/"
    ],
    "removed_unique_values": [
      "CREACU/"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "CREACU": {
        "new": 1,
        "old": 0
      },
      "CREACU/": {
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
  "report_fix_date": "2024-02-26T12:55:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P3795",
  "report_revision_new": 2087785506,
  "report_revision_old": 2082360484,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "CREACU/"
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
    "CREACU"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a plant taxon or cultivar in the Flora of Israel Online database",
    "label": "Flora of Israel Online plant ID"
  },
  "qid": {
    "description": "species of plant",
    "label": "Crepis aculeata"
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
      "added_unique_values": [
        "CREACU"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "CREACU": 1
      },
      "new_unique": [
        "CREACU"
      ],
      "new_values": [
        "CREACU"
      ],
      "new_values_raw": [
        "CREACU"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "CREACU/": 1
      },
      "old_unique": [
        "CREACU/"
      ],
      "old_values": [
        "CREACU/"
      ],
      "old_values_raw": [
        "CREACU/"
      ],
      "removed_unique_values": [
        "CREACU/"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "CREACU": {
          "new": 1,
          "old": 0
        },
        "CREACU/": {
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
      "new_changed": "CREACU",
      "new_pass_regex": true,
      "new_value": "CREACU",
      "normalization_kind": "strip_trailing_slash",
      "normalization_rule": "strip_trailing_slash",
      "old_changed": "CREACU/",
      "old_pass_regex": false,
      "old_value": "CREACU/",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q16121727_2086447164`

| Field | Value |
|---|---|
| qid | Q16121727 |
| property | P3795 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_trailing_slash |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q16121727::P3795 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["podalp"] |
| classification_target_tokens | ["podalp"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "podalp"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "podalp"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "podalp/"
  ],
  "removed_unique_values": [
    "podalp/"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_trailing_slash",
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
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "podalp"
  ],
  "old_value": [
    "podalp/"
  ],
  "revision_id": 2086447164,
  "value": [
    "podalp"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "podalp"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "podalp": 1
    },
    "new_unique": [
      "podalp"
    ],
    "new_values": [
      "podalp"
    ],
    "new_values_raw": [
      "podalp"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "podalp/": 1
    },
    "old_unique": [
      "podalp/"
    ],
    "old_values": [
      "podalp/"
    ],
    "old_values_raw": [
      "podalp/"
    ],
    "removed_unique_values": [
      "podalp/"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "podalp": {
        "new": 1,
        "old": 0
      },
      "podalp/": {
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
  "report_fix_date": "2024-02-26T12:55:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P3795",
  "report_revision_new": 2087785506,
  "report_revision_old": 2082360484,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "podalp/"
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
    "podalp"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a plant taxon or cultivar in the Flora of Israel Online database",
    "label": "Flora of Israel Online plant ID"
  },
  "qid": {
    "description": "species of plant",
    "label": "Podospermum alpigenum"
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
      "added_unique_values": [
        "podalp"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "podalp": 1
      },
      "new_unique": [
        "podalp"
      ],
      "new_values": [
        "podalp"
      ],
      "new_values_raw": [
        "podalp"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "podalp/": 1
      },
      "old_unique": [
        "podalp/"
      ],
      "old_values": [
        "podalp/"
      ],
      "old_values_raw": [
        "podalp/"
      ],
      "removed_unique_values": [
        "podalp/"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "podalp": {
          "new": 1,
          "old": 0
        },
        "podalp/": {
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
      "new_changed": "podalp",
      "new_pass_regex": true,
      "new_value": "podalp",
      "normalization_kind": "strip_trailing_slash",
      "normalization_rule": "strip_trailing_slash",
      "old_changed": "podalp/",
      "old_pass_regex": false,
      "old_value": "podalp/",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q20060274_2086450311`

| Field | Value |
|---|---|
| qid | Q20060274 |
| property | P3795 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_trailing_slash |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q20060274::P3795 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["MEDITA"] |
| classification_target_tokens | ["MEDITA"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "MEDITA"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "MEDITA"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "MEDITA/"
  ],
  "removed_unique_values": [
    "MEDITA/"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_trailing_slash",
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
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "MEDITA"
  ],
  "old_value": [
    "MEDITA/"
  ],
  "revision_id": 2086450311,
  "value": [
    "MEDITA"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "MEDITA"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "MEDITA": 1
    },
    "new_unique": [
      "MEDITA"
    ],
    "new_values": [
      "MEDITA"
    ],
    "new_values_raw": [
      "MEDITA"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "MEDITA/": 1
    },
    "old_unique": [
      "MEDITA/"
    ],
    "old_values": [
      "MEDITA/"
    ],
    "old_values_raw": [
      "MEDITA/"
    ],
    "removed_unique_values": [
      "MEDITA/"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "MEDITA": {
        "new": 1,
        "old": 0
      },
      "MEDITA/": {
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
  "report_fix_date": "2024-02-26T12:55:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P3795",
  "report_revision_new": 2087785506,
  "report_revision_old": 2082360484,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "MEDITA/"
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
    "MEDITA"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a plant taxon or cultivar in the Flora of Israel Online database",
    "label": "Flora of Israel Online plant ID"
  },
  "qid": {
    "description": "species of plant",
    "label": "Medicago italica"
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
      "added_unique_values": [
        "MEDITA"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "MEDITA": 1
      },
      "new_unique": [
        "MEDITA"
      ],
      "new_values": [
        "MEDITA"
      ],
      "new_values_raw": [
        "MEDITA"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "MEDITA/": 1
      },
      "old_unique": [
        "MEDITA/"
      ],
      "old_values": [
        "MEDITA/"
      ],
      "old_values_raw": [
        "MEDITA/"
      ],
      "removed_unique_values": [
        "MEDITA/"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "MEDITA": {
          "new": 1,
          "old": 0
        },
        "MEDITA/": {
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
      "new_changed": "MEDITA",
      "new_pass_regex": true,
      "new_value": "MEDITA",
      "normalization_kind": "strip_trailing_slash",
      "normalization_rule": "strip_trailing_slash",
      "old_changed": "MEDITA/",
      "old_pass_regex": false,
      "old_value": "MEDITA/",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q2636110_2425228276`

| Field | Value |
|---|---|
| qid | Q2636110 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q2636110::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["141581"] |
| classification_target_tokens | ["141581"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "141581"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "141581"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL141581"
  ],
  "removed_unique_values": [
    "SCHEMBL141581"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "141581"
  ],
  "old_value": [
    "SCHEMBL141581"
  ],
  "revision_id": 2425228276,
  "value": [
    "141581"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "141581"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "141581": 1
    },
    "new_unique": [
      "141581"
    ],
    "new_values": [
      "141581"
    ],
    "new_values_raw": [
      "141581"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL141581": 1
    },
    "old_unique": [
      "SCHEMBL141581"
    ],
    "old_values": [
      "SCHEMBL141581"
    ],
    "old_values_raw": [
      "SCHEMBL141581"
    ],
    "removed_unique_values": [
      "SCHEMBL141581"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "141581": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL141581": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "SCHEMBL141581"
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
    "141581"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "midecamycin"
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
        "141581"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "141581": 1
      },
      "new_unique": [
        "141581"
      ],
      "new_values": [
        "141581"
      ],
      "new_values_raw": [
        "141581"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL141581": 1
      },
      "old_unique": [
        "SCHEMBL141581"
      ],
      "old_values": [
        "SCHEMBL141581"
      ],
      "old_values_raw": [
        "SCHEMBL141581"
      ],
      "removed_unique_values": [
        "SCHEMBL141581"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "141581": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL141581": {
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
      "new_changed": "141581",
      "new_pass_regex": true,
      "new_value": "141581",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL141581",
      "old_pass_regex": false,
      "old_value": "SCHEMBL141581",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q414547_2425330494`

| Field | Value |
|---|---|
| qid | Q414547 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q414547::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["4603"] |
| classification_target_tokens | ["4603"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "4603"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "4603"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL4603"
  ],
  "removed_unique_values": [
    "SCHEMBL4603"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "4603"
  ],
  "old_value": [
    "SCHEMBL4603"
  ],
  "revision_id": 2425330494,
  "value": [
    "4603"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "4603"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "4603": 1
    },
    "new_unique": [
      "4603"
    ],
    "new_values": [
      "4603"
    ],
    "new_values_raw": [
      "4603"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL4603": 1
    },
    "old_unique": [
      "SCHEMBL4603"
    ],
    "old_values": [
      "SCHEMBL4603"
    ],
    "old_values_raw": [
      "SCHEMBL4603"
    ],
    "removed_unique_values": [
      "SCHEMBL4603"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "4603": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL4603": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "SCHEMBL4603"
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
    "4603"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "sodium salicylate"
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
        "4603"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "4603": 1
      },
      "new_unique": [
        "4603"
      ],
      "new_values": [
        "4603"
      ],
      "new_values_raw": [
        "4603"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL4603": 1
      },
      "old_unique": [
        "SCHEMBL4603"
      ],
      "old_values": [
        "SCHEMBL4603"
      ],
      "old_values_raw": [
        "SCHEMBL4603"
      ],
      "removed_unique_values": [
        "SCHEMBL4603"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "4603": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL4603": {
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
      "new_changed": "4603",
      "new_pass_regex": true,
      "new_value": "4603",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL4603",
      "old_pass_regex": false,
      "old_value": "SCHEMBL4603",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q421322_2425334359`

| Field | Value |
|---|---|
| qid | Q421322 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q421322::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["34316"] |
| classification_target_tokens | ["34316"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "34316"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "34316"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL34316"
  ],
  "removed_unique_values": [
    "SCHEMBL34316"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "34316"
  ],
  "old_value": [
    "SCHEMBL34316"
  ],
  "revision_id": 2425334359,
  "value": [
    "34316"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "34316"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "34316": 1
    },
    "new_unique": [
      "34316"
    ],
    "new_values": [
      "34316"
    ],
    "new_values_raw": [
      "34316"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL34316": 1
    },
    "old_unique": [
      "SCHEMBL34316"
    ],
    "old_values": [
      "SCHEMBL34316"
    ],
    "old_values_raw": [
      "SCHEMBL34316"
    ],
    "removed_unique_values": [
      "SCHEMBL34316"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "34316": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL34316": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "SCHEMBL34316"
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
    "34316"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "atosiban"
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
        "34316"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "34316": 1
      },
      "new_unique": [
        "34316"
      ],
      "new_values": [
        "34316"
      ],
      "new_values_raw": [
        "34316"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL34316": 1
      },
      "old_unique": [
        "SCHEMBL34316"
      ],
      "old_values": [
        "SCHEMBL34316"
      ],
      "old_values_raw": [
        "SCHEMBL34316"
      ],
      "removed_unique_values": [
        "SCHEMBL34316"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "34316": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL34316": {
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
      "new_changed": "34316",
      "new_pass_regex": true,
      "new_value": "34316",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL34316",
      "old_pass_regex": false,
      "old_value": "SCHEMBL34316",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 011. `repair_Q422582_2425335103`

| Field | Value |
|---|---|
| qid | Q422582 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q422582::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["37405"] |
| classification_target_tokens | ["37405"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "37405"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "37405"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL37405"
  ],
  "removed_unique_values": [
    "SCHEMBL37405"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "37405"
  ],
  "old_value": [
    "SCHEMBL37405"
  ],
  "revision_id": 2425335103,
  "value": [
    "37405"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "37405"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "37405": 1
    },
    "new_unique": [
      "37405"
    ],
    "new_values": [
      "37405"
    ],
    "new_values_raw": [
      "37405"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL37405": 1
    },
    "old_unique": [
      "SCHEMBL37405"
    ],
    "old_values": [
      "SCHEMBL37405"
    ],
    "old_values_raw": [
      "SCHEMBL37405"
    ],
    "removed_unique_values": [
      "SCHEMBL37405"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "37405": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL37405": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "SCHEMBL37405"
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
    "37405"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "guanosine 5'-diphosphate (RRSR form)"
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
        "37405"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "37405": 1
      },
      "new_unique": [
        "37405"
      ],
      "new_values": [
        "37405"
      ],
      "new_values_raw": [
        "37405"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL37405": 1
      },
      "old_unique": [
        "SCHEMBL37405"
      ],
      "old_values": [
        "SCHEMBL37405"
      ],
      "old_values_raw": [
        "SCHEMBL37405"
      ],
      "removed_unique_values": [
        "SCHEMBL37405"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "37405": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL37405": {
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
      "new_changed": "37405",
      "new_pass_regex": true,
      "new_value": "37405",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL37405",
      "old_pass_regex": false,
      "old_value": "SCHEMBL37405",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 012. `repair_Q5332578_2086433858`

| Field | Value |
|---|---|
| qid | Q5332578 |
| property | P3795 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_trailing_slash |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q5332578::P3795 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["ECHADE"] |
| classification_target_tokens | ["ECHADE"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "ECHADE"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "ECHADE"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "ECHADE/"
  ],
  "removed_unique_values": [
    "ECHADE/"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_trailing_slash",
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
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "ECHADE"
  ],
  "old_value": [
    "ECHADE/"
  ],
  "revision_id": 2086433858,
  "value": [
    "ECHADE"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "ECHADE"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "ECHADE": 1
    },
    "new_unique": [
      "ECHADE"
    ],
    "new_values": [
      "ECHADE"
    ],
    "new_values_raw": [
      "ECHADE"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "ECHADE/": 1
    },
    "old_unique": [
      "ECHADE/"
    ],
    "old_values": [
      "ECHADE/"
    ],
    "old_values_raw": [
      "ECHADE/"
    ],
    "removed_unique_values": [
      "ECHADE/"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "ECHADE": {
        "new": 1,
        "old": 0
      },
      "ECHADE/": {
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
  "report_fix_date": "2024-02-26T12:55:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P3795",
  "report_revision_new": 2087785506,
  "report_revision_old": 2082360484,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "ECHADE/"
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
    "ECHADE"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a plant taxon or cultivar in the Flora of Israel Online database",
    "label": "Flora of Israel Online plant ID"
  },
  "qid": {
    "description": "species of plant",
    "label": "Echinops adenocaulos"
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
      "added_unique_values": [
        "ECHADE"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "ECHADE": 1
      },
      "new_unique": [
        "ECHADE"
      ],
      "new_values": [
        "ECHADE"
      ],
      "new_values_raw": [
        "ECHADE"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "ECHADE/": 1
      },
      "old_unique": [
        "ECHADE/"
      ],
      "old_values": [
        "ECHADE/"
      ],
      "old_values_raw": [
        "ECHADE/"
      ],
      "removed_unique_values": [
        "ECHADE/"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "ECHADE": {
          "new": 1,
          "old": 0
        },
        "ECHADE/": {
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
      "new_changed": "ECHADE",
      "new_pass_regex": true,
      "new_value": "ECHADE",
      "normalization_kind": "strip_trailing_slash",
      "normalization_rule": "strip_trailing_slash",
      "old_changed": "ECHADE/",
      "old_pass_regex": false,
      "old_value": "ECHADE/",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 013. `repair_Q67192867_2086453916`

| Field | Value |
|---|---|
| qid | Q67192867 |
| property | P3795 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_trailing_slash |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q67192867::P3795 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["glecor"] |
| classification_target_tokens | ["glecor"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "glecor"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "glecor"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "glecor/"
  ],
  "removed_unique_values": [
    "glecor/"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_trailing_slash",
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
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "glecor"
  ],
  "old_value": [
    "glecor/"
  ],
  "revision_id": 2086453916,
  "value": [
    "glecor"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "glecor"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "glecor": 1
    },
    "new_unique": [
      "glecor"
    ],
    "new_values": [
      "glecor"
    ],
    "new_values_raw": [
      "glecor"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "glecor/": 1
    },
    "old_unique": [
      "glecor/"
    ],
    "old_values": [
      "glecor/"
    ],
    "old_values_raw": [
      "glecor/"
    ],
    "removed_unique_values": [
      "glecor/"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "glecor": {
        "new": 1,
        "old": 0
      },
      "glecor/": {
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
  "report_fix_date": "2024-02-26T12:55:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P3795",
  "report_revision_new": 2087785506,
  "report_revision_old": 2082360484,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "glecor/"
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
    "glecor"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a plant taxon or cultivar in the Flora of Israel Online database",
    "label": "Flora of Israel Online plant ID"
  },
  "qid": {
    "description": "species of plant",
    "label": "Glebionis coronarium"
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
      "added_unique_values": [
        "glecor"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "glecor": 1
      },
      "new_unique": [
        "glecor"
      ],
      "new_values": [
        "glecor"
      ],
      "new_values_raw": [
        "glecor"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "glecor/": 1
      },
      "old_unique": [
        "glecor/"
      ],
      "old_values": [
        "glecor/"
      ],
      "old_values_raw": [
        "glecor/"
      ],
      "removed_unique_values": [
        "glecor/"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "glecor": {
          "new": 1,
          "old": 0
        },
        "glecor/": {
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
      "new_changed": "glecor",
      "new_pass_regex": true,
      "new_value": "glecor",
      "normalization_kind": "strip_trailing_slash",
      "normalization_rule": "strip_trailing_slash",
      "old_changed": "glecor/",
      "old_pass_regex": false,
      "old_value": "glecor/",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 014. `repair_Q72461546_2425387483`

| Field | Value |
|---|---|
| qid | Q72461546 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q72461546::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["214196"] |
| classification_target_tokens | ["214196"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "214196"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "214196"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL214196"
  ],
  "removed_unique_values": [
    "SCHEMBL214196"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "214196"
  ],
  "old_value": [
    "SCHEMBL214196"
  ],
  "revision_id": 2425387483,
  "value": [
    "214196"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "214196"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "214196": 1
    },
    "new_unique": [
      "214196"
    ],
    "new_values": [
      "214196"
    ],
    "new_values_raw": [
      "214196"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL214196": 1
    },
    "old_unique": [
      "SCHEMBL214196"
    ],
    "old_values": [
      "SCHEMBL214196"
    ],
    "old_values_raw": [
      "SCHEMBL214196"
    ],
    "removed_unique_values": [
      "SCHEMBL214196"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "214196": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL214196": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "SCHEMBL214196"
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
    "214196"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "1,9-Decadiene-maleic anhydride-methyl vinyl ether copolymer"
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
        "214196"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "214196": 1
      },
      "new_unique": [
        "214196"
      ],
      "new_values": [
        "214196"
      ],
      "new_values_raw": [
        "214196"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL214196": 1
      },
      "old_unique": [
        "SCHEMBL214196"
      ],
      "old_values": [
        "SCHEMBL214196"
      ],
      "old_values_raw": [
        "SCHEMBL214196"
      ],
      "removed_unique_values": [
        "SCHEMBL214196"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "214196": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL214196": {
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
      "new_changed": "214196",
      "new_pass_regex": true,
      "new_value": "214196",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL214196",
      "old_pass_regex": false,
      "old_value": "SCHEMBL214196",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 015. `repair_Q72491370_2425414361`

| Field | Value |
|---|---|
| qid | Q72491370 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q72491370::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["122885"] |
| classification_target_tokens | ["122885"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "122885"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "122885"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL122885"
  ],
  "removed_unique_values": [
    "SCHEMBL122885"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "122885"
  ],
  "old_value": [
    "SCHEMBL122885"
  ],
  "revision_id": 2425414361,
  "value": [
    "122885"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "122885"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "122885": 1
    },
    "new_unique": [
      "122885"
    ],
    "new_values": [
      "122885"
    ],
    "new_values_raw": [
      "122885"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL122885": 1
    },
    "old_unique": [
      "SCHEMBL122885"
    ],
    "old_values": [
      "SCHEMBL122885"
    ],
    "old_values_raw": [
      "SCHEMBL122885"
    ],
    "removed_unique_values": [
      "SCHEMBL122885"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "122885": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL122885": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "SCHEMBL122885"
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
    "122885"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemische verbinding",
    "label": "Methyl 2-hydroxy-3-methylbutanoate"
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
        "122885"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "122885": 1
      },
      "new_unique": [
        "122885"
      ],
      "new_values": [
        "122885"
      ],
      "new_values_raw": [
        "122885"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL122885": 1
      },
      "old_unique": [
        "SCHEMBL122885"
      ],
      "old_values": [
        "SCHEMBL122885"
      ],
      "old_values_raw": [
        "SCHEMBL122885"
      ],
      "removed_unique_values": [
        "SCHEMBL122885"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "122885": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL122885": {
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
      "new_changed": "122885",
      "new_pass_regex": true,
      "new_value": "122885",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL122885",
      "old_pass_regex": false,
      "old_value": "SCHEMBL122885",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 016. `repair_Q743661_2425443354`

| Field | Value |
|---|---|
| qid | Q743661 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q743661::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["187397"] |
| classification_target_tokens | ["187397"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "187397"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "187397"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL187397"
  ],
  "removed_unique_values": [
    "SCHEMBL187397"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "187397"
  ],
  "old_value": [
    "SCHEMBL187397"
  ],
  "revision_id": 2425443354,
  "value": [
    "187397"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "187397"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "187397": 1
    },
    "new_unique": [
      "187397"
    ],
    "new_values": [
      "187397"
    ],
    "new_values_raw": [
      "187397"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL187397": 1
    },
    "old_unique": [
      "SCHEMBL187397"
    ],
    "old_values": [
      "SCHEMBL187397"
    ],
    "old_values_raw": [
      "SCHEMBL187397"
    ],
    "removed_unique_values": [
      "SCHEMBL187397"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "187397": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL187397": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "SCHEMBL187397"
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
    "187397"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "muco-inositol"
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
        "187397"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "187397": 1
      },
      "new_unique": [
        "187397"
      ],
      "new_values": [
        "187397"
      ],
      "new_values_raw": [
        "187397"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL187397": 1
      },
      "old_unique": [
        "SCHEMBL187397"
      ],
      "old_values": [
        "SCHEMBL187397"
      ],
      "old_values_raw": [
        "SCHEMBL187397"
      ],
      "removed_unique_values": [
        "SCHEMBL187397"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "187397": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL187397": {
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
      "new_changed": "187397",
      "new_pass_regex": true,
      "new_value": "187397",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL187397",
      "old_pass_regex": false,
      "old_value": "SCHEMBL187397",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 017. `repair_Q775073_2425452244`

| Field | Value |
|---|---|
| qid | Q775073 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q775073::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["44957"] |
| classification_target_tokens | ["44957"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "44957"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "44957"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL44957"
  ],
  "removed_unique_values": [
    "SCHEMBL44957"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "44957"
  ],
  "old_value": [
    "SCHEMBL44957"
  ],
  "revision_id": 2425452244,
  "value": [
    "44957"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "44957"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "44957": 1
    },
    "new_unique": [
      "44957"
    ],
    "new_values": [
      "44957"
    ],
    "new_values_raw": [
      "44957"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL44957": 1
    },
    "old_unique": [
      "SCHEMBL44957"
    ],
    "old_values": [
      "SCHEMBL44957"
    ],
    "old_values_raw": [
      "SCHEMBL44957"
    ],
    "removed_unique_values": [
      "SCHEMBL44957"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "44957": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL44957": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "SCHEMBL44957"
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
    "44957"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "cinnarizine"
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
        "44957"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "44957": 1
      },
      "new_unique": [
        "44957"
      ],
      "new_values": [
        "44957"
      ],
      "new_values_raw": [
        "44957"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL44957": 1
      },
      "old_unique": [
        "SCHEMBL44957"
      ],
      "old_values": [
        "SCHEMBL44957"
      ],
      "old_values_raw": [
        "SCHEMBL44957"
      ],
      "removed_unique_values": [
        "SCHEMBL44957"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "44957": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL44957": {
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
      "new_changed": "44957",
      "new_pass_regex": true,
      "new_value": "44957",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL44957",
      "old_pass_regex": false,
      "old_value": "SCHEMBL44957",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 018. `repair_Q82006684_2425465303`

| Field | Value |
|---|---|
| qid | Q82006684 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q82006684::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["16322828"] |
| classification_target_tokens | ["16322828"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "16322828"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "16322828"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL16322828"
  ],
  "removed_unique_values": [
    "SCHEMBL16322828"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "16322828"
  ],
  "old_value": [
    "SCHEMBL16322828"
  ],
  "revision_id": 2425465303,
  "value": [
    "16322828"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "16322828"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "16322828": 1
    },
    "new_unique": [
      "16322828"
    ],
    "new_values": [
      "16322828"
    ],
    "new_values_raw": [
      "16322828"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL16322828": 1
    },
    "old_unique": [
      "SCHEMBL16322828"
    ],
    "old_values": [
      "SCHEMBL16322828"
    ],
    "old_values_raw": [
      "SCHEMBL16322828"
    ],
    "removed_unique_values": [
      "SCHEMBL16322828"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "16322828": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL16322828": {
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
  "report_fix_date": "2025-11-06T09:25:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426458092,
  "report_revision_old": 2425837272,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "SCHEMBL16322828"
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
    "16322828"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "N-(4-Chloro-2-methylphenyl)-2-[(E)-(4-chloro-2-nitrophenyl)diazenyl]-3-oxobutanamide"
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
        "16322828"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "16322828": 1
      },
      "new_unique": [
        "16322828"
      ],
      "new_values": [
        "16322828"
      ],
      "new_values_raw": [
        "16322828"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL16322828": 1
      },
      "old_unique": [
        "SCHEMBL16322828"
      ],
      "old_values": [
        "SCHEMBL16322828"
      ],
      "old_values_raw": [
        "SCHEMBL16322828"
      ],
      "removed_unique_values": [
        "SCHEMBL16322828"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "16322828": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL16322828": {
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
      "new_changed": "16322828",
      "new_pass_regex": true,
      "new_value": "16322828",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL16322828",
      "old_pass_regex": false,
      "old_value": "SCHEMBL16322828",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 019. `repair_Q82009517_2425465994`

| Field | Value |
|---|---|
| qid | Q82009517 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q82009517::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["968307"] |
| classification_target_tokens | ["968307"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "968307"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "968307"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL968307"
  ],
  "removed_unique_values": [
    "SCHEMBL968307"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "968307"
  ],
  "old_value": [
    "SCHEMBL968307"
  ],
  "revision_id": 2425465994,
  "value": [
    "968307"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "968307"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "968307": 1
    },
    "new_unique": [
      "968307"
    ],
    "new_values": [
      "968307"
    ],
    "new_values_raw": [
      "968307"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL968307": 1
    },
    "old_unique": [
      "SCHEMBL968307"
    ],
    "old_values": [
      "SCHEMBL968307"
    ],
    "old_values_raw": [
      "SCHEMBL968307"
    ],
    "removed_unique_values": [
      "SCHEMBL968307"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "968307": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL968307": {
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
  "report_fix_date": "2025-11-06T09:25:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426458092,
  "report_revision_old": 2425837272,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "SCHEMBL968307"
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
    "968307"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "butyl methylcarbamate"
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
        "968307"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "968307": 1
      },
      "new_unique": [
        "968307"
      ],
      "new_values": [
        "968307"
      ],
      "new_values_raw": [
        "968307"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL968307": 1
      },
      "old_unique": [
        "SCHEMBL968307"
      ],
      "old_values": [
        "SCHEMBL968307"
      ],
      "old_values_raw": [
        "SCHEMBL968307"
      ],
      "removed_unique_values": [
        "SCHEMBL968307"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "968307": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL968307": {
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
      "new_changed": "968307",
      "new_pass_regex": true,
      "new_value": "968307",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL968307",
      "old_pass_regex": false,
      "old_value": "SCHEMBL968307",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 020. `repair_Q82015154_2425466908`

| Field | Value |
|---|---|
| qid | Q82015154 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q82015154::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["4386493"] |
| classification_target_tokens | ["4386493"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "4386493"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "4386493"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL4386493"
  ],
  "removed_unique_values": [
    "SCHEMBL4386493"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "4386493"
  ],
  "old_value": [
    "SCHEMBL4386493"
  ],
  "revision_id": 2425466908,
  "value": [
    "4386493"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "4386493"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "4386493": 1
    },
    "new_unique": [
      "4386493"
    ],
    "new_values": [
      "4386493"
    ],
    "new_values_raw": [
      "4386493"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL4386493": 1
    },
    "old_unique": [
      "SCHEMBL4386493"
    ],
    "old_values": [
      "SCHEMBL4386493"
    ],
    "old_values_raw": [
      "SCHEMBL4386493"
    ],
    "removed_unique_values": [
      "SCHEMBL4386493"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "4386493": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL4386493": {
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
  "report_fix_date": "2025-11-06T09:25:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426458092,
  "report_revision_old": 2425837272,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "SCHEMBL4386493"
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
    "4386493"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "4-hexylcyclohexan-1-ol"
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
        "4386493"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "4386493": 1
      },
      "new_unique": [
        "4386493"
      ],
      "new_values": [
        "4386493"
      ],
      "new_values_raw": [
        "4386493"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL4386493": 1
      },
      "old_unique": [
        "SCHEMBL4386493"
      ],
      "old_values": [
        "SCHEMBL4386493"
      ],
      "old_values_raw": [
        "SCHEMBL4386493"
      ],
      "removed_unique_values": [
        "SCHEMBL4386493"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "4386493": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL4386493": {
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
      "new_changed": "4386493",
      "new_pass_regex": true,
      "new_value": "4386493",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL4386493",
      "old_pass_regex": false,
      "old_value": "SCHEMBL4386493",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 021. `repair_Q82081964_2425481690`

| Field | Value |
|---|---|
| qid | Q82081964 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q82081964::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["190428"] |
| classification_target_tokens | ["190428"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "190428"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "190428"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL190428"
  ],
  "removed_unique_values": [
    "SCHEMBL190428"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "190428"
  ],
  "old_value": [
    "SCHEMBL190428"
  ],
  "revision_id": 2425481690,
  "value": [
    "190428"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "190428"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "190428": 1
    },
    "new_unique": [
      "190428"
    ],
    "new_values": [
      "190428"
    ],
    "new_values_raw": [
      "190428"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL190428": 1
    },
    "old_unique": [
      "SCHEMBL190428"
    ],
    "old_values": [
      "SCHEMBL190428"
    ],
    "old_values_raw": [
      "SCHEMBL190428"
    ],
    "removed_unique_values": [
      "SCHEMBL190428"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "190428": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL190428": {
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
  "report_fix_date": "2025-11-06T09:25:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426458092,
  "report_revision_old": 2425837272,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "SCHEMBL190428"
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
    "190428"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "METHYLPHENYLPHOSPHINIC ACID"
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
        "190428"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "190428": 1
      },
      "new_unique": [
        "190428"
      ],
      "new_values": [
        "190428"
      ],
      "new_values_raw": [
        "190428"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL190428": 1
      },
      "old_unique": [
        "SCHEMBL190428"
      ],
      "old_values": [
        "SCHEMBL190428"
      ],
      "old_values_raw": [
        "SCHEMBL190428"
      ],
      "removed_unique_values": [
        "SCHEMBL190428"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "190428": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL190428": {
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
      "new_changed": "190428",
      "new_pass_regex": true,
      "new_value": "190428",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL190428",
      "old_pass_regex": false,
      "old_value": "SCHEMBL190428",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 022. `repair_Q82091736_2425483914`

| Field | Value |
|---|---|
| qid | Q82091736 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q82091736::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["223337"] |
| classification_target_tokens | ["223337"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "223337"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "223337"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL223337"
  ],
  "removed_unique_values": [
    "SCHEMBL223337"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "223337"
  ],
  "old_value": [
    "SCHEMBL223337"
  ],
  "revision_id": 2425483914,
  "value": [
    "223337"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "223337"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "223337": 1
    },
    "new_unique": [
      "223337"
    ],
    "new_values": [
      "223337"
    ],
    "new_values_raw": [
      "223337"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL223337": 1
    },
    "old_unique": [
      "SCHEMBL223337"
    ],
    "old_values": [
      "SCHEMBL223337"
    ],
    "old_values_raw": [
      "SCHEMBL223337"
    ],
    "removed_unique_values": [
      "SCHEMBL223337"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "223337": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL223337": {
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
  "report_fix_date": "2025-11-06T09:25:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426458092,
  "report_revision_old": 2425837272,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "SCHEMBL223337"
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
    "223337"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "2-methoxy-4-morpholinoaniline"
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
        "223337"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "223337": 1
      },
      "new_unique": [
        "223337"
      ],
      "new_values": [
        "223337"
      ],
      "new_values_raw": [
        "223337"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL223337": 1
      },
      "old_unique": [
        "SCHEMBL223337"
      ],
      "old_values": [
        "SCHEMBL223337"
      ],
      "old_values_raw": [
        "SCHEMBL223337"
      ],
      "removed_unique_values": [
        "SCHEMBL223337"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "223337": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL223337": {
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
      "new_changed": "223337",
      "new_pass_regex": true,
      "new_value": "223337",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL223337",
      "old_pass_regex": false,
      "old_value": "SCHEMBL223337",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 023. `repair_Q82101968_2425487135`

| Field | Value |
|---|---|
| qid | Q82101968 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q82101968::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["11984337"] |
| classification_target_tokens | ["11984337"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "11984337"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "11984337"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL11984337"
  ],
  "removed_unique_values": [
    "SCHEMBL11984337"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "11984337"
  ],
  "old_value": [
    "SCHEMBL11984337"
  ],
  "revision_id": 2425487135,
  "value": [
    "11984337"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "11984337"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "11984337": 1
    },
    "new_unique": [
      "11984337"
    ],
    "new_values": [
      "11984337"
    ],
    "new_values_raw": [
      "11984337"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL11984337": 1
    },
    "old_unique": [
      "SCHEMBL11984337"
    ],
    "old_values": [
      "SCHEMBL11984337"
    ],
    "old_values_raw": [
      "SCHEMBL11984337"
    ],
    "removed_unique_values": [
      "SCHEMBL11984337"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "11984337": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL11984337": {
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
  "report_fix_date": "2025-11-06T09:25:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426458092,
  "report_revision_old": 2425837272,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "SCHEMBL11984337"
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
    "11984337"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "p-methoxybenzylidene-benzyl-amine"
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
        "11984337"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "11984337": 1
      },
      "new_unique": [
        "11984337"
      ],
      "new_values": [
        "11984337"
      ],
      "new_values_raw": [
        "11984337"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL11984337": 1
      },
      "old_unique": [
        "SCHEMBL11984337"
      ],
      "old_values": [
        "SCHEMBL11984337"
      ],
      "old_values_raw": [
        "SCHEMBL11984337"
      ],
      "removed_unique_values": [
        "SCHEMBL11984337"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "11984337": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL11984337": {
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
      "new_changed": "11984337",
      "new_pass_regex": true,
      "new_value": "11984337",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL11984337",
      "old_pass_regex": false,
      "old_value": "SCHEMBL11984337",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 024. `repair_Q82107958_2425489486`

| Field | Value |
|---|---|
| qid | Q82107958 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q82107958::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["490162"] |
| classification_target_tokens | ["490162"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "490162"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "490162"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL490162"
  ],
  "removed_unique_values": [
    "SCHEMBL490162"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "490162"
  ],
  "old_value": [
    "SCHEMBL490162"
  ],
  "revision_id": 2425489486,
  "value": [
    "490162"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "490162"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "490162": 1
    },
    "new_unique": [
      "490162"
    ],
    "new_values": [
      "490162"
    ],
    "new_values_raw": [
      "490162"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL490162": 1
    },
    "old_unique": [
      "SCHEMBL490162"
    ],
    "old_values": [
      "SCHEMBL490162"
    ],
    "old_values_raw": [
      "SCHEMBL490162"
    ],
    "removed_unique_values": [
      "SCHEMBL490162"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "490162": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL490162": {
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
  "report_fix_date": "2025-11-06T09:25:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426458092,
  "report_revision_old": 2425837272,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "SCHEMBL490162"
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
    "490162"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "Methyl 8-nonenoate"
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
        "490162"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "490162": 1
      },
      "new_unique": [
        "490162"
      ],
      "new_values": [
        "490162"
      ],
      "new_values_raw": [
        "490162"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL490162": 1
      },
      "old_unique": [
        "SCHEMBL490162"
      ],
      "old_values": [
        "SCHEMBL490162"
      ],
      "old_values_raw": [
        "SCHEMBL490162"
      ],
      "removed_unique_values": [
        "SCHEMBL490162"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "490162": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL490162": {
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
      "new_changed": "490162",
      "new_pass_regex": true,
      "new_value": "490162",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL490162",
      "old_pass_regex": false,
      "old_value": "SCHEMBL490162",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 025. `repair_Q82893540_2425667305`

| Field | Value |
|---|---|
| qid | Q82893540 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q82893540::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["124166"] |
| classification_target_tokens | ["124166"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "124166"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "124166"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL124166"
  ],
  "removed_unique_values": [
    "SCHEMBL124166"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "124166"
  ],
  "old_value": [
    "SCHEMBL124166"
  ],
  "revision_id": 2425667305,
  "value": [
    "124166"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "124166"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "124166": 1
    },
    "new_unique": [
      "124166"
    ],
    "new_values": [
      "124166"
    ],
    "new_values_raw": [
      "124166"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL124166": 1
    },
    "old_unique": [
      "SCHEMBL124166"
    ],
    "old_values": [
      "SCHEMBL124166"
    ],
    "old_values_raw": [
      "SCHEMBL124166"
    ],
    "removed_unique_values": [
      "SCHEMBL124166"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "124166": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL124166": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "SCHEMBL124166"
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
    "124166"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "Acetic acid--lysine (1/1)"
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
        "124166"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "124166": 1
      },
      "new_unique": [
        "124166"
      ],
      "new_values": [
        "124166"
      ],
      "new_values_raw": [
        "124166"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL124166": 1
      },
      "old_unique": [
        "SCHEMBL124166"
      ],
      "old_values": [
        "SCHEMBL124166"
      ],
      "old_values_raw": [
        "SCHEMBL124166"
      ],
      "removed_unique_values": [
        "SCHEMBL124166"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "124166": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL124166": {
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
      "new_changed": "124166",
      "new_pass_regex": true,
      "new_value": "124166",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL124166",
      "old_pass_regex": false,
      "old_value": "SCHEMBL124166",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---

## 026. `repair_Q90536038_2425735162`

| Field | Value |
|---|---|
| qid | Q90536038 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | strip_schembl_prefix |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q90536038::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["14388592"] |
| classification_target_tokens | ["14388592"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "14388592"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "14388592"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "SCHEMBL14388592"
  ],
  "removed_unique_values": [
    "SCHEMBL14388592"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "strip_schembl_prefix",
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
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "14388592"
  ],
  "old_value": [
    "SCHEMBL14388592"
  ],
  "revision_id": 2425735162,
  "value": [
    "14388592"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "14388592"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "14388592": 1
    },
    "new_unique": [
      "14388592"
    ],
    "new_values": [
      "14388592"
    ],
    "new_values_raw": [
      "14388592"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "SCHEMBL14388592": 1
    },
    "old_unique": [
      "SCHEMBL14388592"
    ],
    "old_values": [
      "SCHEMBL14388592"
    ],
    "old_values_raw": [
      "SCHEMBL14388592"
    ],
    "removed_unique_values": [
      "SCHEMBL14388592"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "14388592": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL14388592": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "SCHEMBL14388592"
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
    "14388592"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
    "label": "SureChEMBL ID"
  },
  "qid": {
    "description": "chemical compound",
    "label": "2,4(1H,3H)-Pyrimidinedione, 5-methyl-1-β-D-xylofuranosyl-"
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
        "14388592"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "14388592": 1
      },
      "new_unique": [
        "14388592"
      ],
      "new_values": [
        "14388592"
      ],
      "new_values_raw": [
        "14388592"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL14388592": 1
      },
      "old_unique": [
        "SCHEMBL14388592"
      ],
      "old_values": [
        "SCHEMBL14388592"
      ],
      "old_values_raw": [
        "SCHEMBL14388592"
      ],
      "removed_unique_values": [
        "SCHEMBL14388592"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "14388592": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL14388592": {
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
      "new_changed": "14388592",
      "new_pass_regex": true,
      "new_value": "14388592",
      "normalization_kind": "strip_schembl_prefix",
      "normalization_rule": "strip_schembl_prefix",
      "old_changed": "SCHEMBL14388592",
      "old_pass_regex": false,
      "old_value": "SCHEMBL14388592",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "format_normalization",
    "step": "branch"
  }
]
```

---
