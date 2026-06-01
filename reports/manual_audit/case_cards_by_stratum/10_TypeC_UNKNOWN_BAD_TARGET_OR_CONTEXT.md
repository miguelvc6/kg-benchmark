# TypeC_UNKNOWN_BAD_TARGET_OR_CONTEXT

Cases: 10

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q1462_2442788470`

| Field | Value |
|---|---|
| qid | Q1462 |
| property | P1549 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_BAD_TARGET_OR_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_bad_target_or_context |
| decision_constraint_type |   |
| group_key | ABOX::Q1462::P1549 |
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
| truth_tokens_preview | ["Sardijn@nl", "Sardiniër@nl", "Sard@nl", "sardi@it", "sardos@sc", "...(+10)"] |
| classification_target_tokens | ["Sardinnen@de", "Sardinen@de"] |
| classification_target_reason | mixed update classification uses the deterministic changed pair while ignoring retained values |
| decision_branch | single_value_report_multiple_new_values |
| rationale | Single-value violation is followed by multiple created/added target values; treated as report-context mismatch or bad target rather than clean external evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Sardinen@de"
  ],
  "classification_target_reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
  "classification_target_role": "changed_pair",
  "classification_target_tokens": [
    "Sardinnen@de",
    "Sardinen@de"
  ],
  "new_changed_value": "Sardinen@de",
  "old_changed_value": "Sardinnen@de",
  "removed_target_tokens": [
    "Sardinnen@de"
  ],
  "removed_unique_values": [
    "Sardinnen@de"
  ],
  "retained_support_tokens": [
    "Sard@nl",
    "Sarde@de",
    "Sarden@de",
    "Sardijn@nl",
    "Sardin@de",
    "Sardiniër@nl",
    "sarda@it",
    "sarda@sc",
    "sardas@sc",
    "sarde@it",
    "sardi@it",
    "sardo@it",
    "sardos@sc",
    "sardu@sc"
  ],
  "retained_unique_values": [
    "Sard@nl",
    "Sarde@de",
    "Sarden@de",
    "Sardijn@nl",
    "Sardin@de",
    "Sardiniër@nl",
    "sarda@it",
    "sarda@sc",
    "sardas@sc",
    "sarde@it",
    "sardi@it",
    "sardo@it",
    "sardos@sc",
    "sardu@sc"
  ],
  "semantic_action": "MIXED_UPDATE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_bad_target_or_context",
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
  "author": "Labant",
  "kind": "A_BOX",
  "new_value": [
    "Sardijn@nl",
    "Sardiniër@nl",
    "Sard@nl",
    "sardi@it",
    "sardos@sc",
    "sardo@it",
    "sarda@it",
    "sarde@it",
    "sardu@sc",
    "sarda@sc",
    "sardas@sc",
    "Sarde@de",
    "Sardin@de",
    "Sarden@de",
    "Sardinen@de"
  ],
  "old_value": [
    "Sardijn@nl",
    "Sardiniër@nl",
    "Sard@nl",
    "sardi@it",
    "sardos@sc",
    "sardo@it",
    "sarda@it",
    "sarde@it",
    "sardu@sc",
    "sarda@sc",
    "sardas@sc",
    "Sarde@de",
    "Sardin@de",
    "Sarden@de",
    "Sardinnen@de"
  ],
  "revision_id": 2442788470,
  "value": [
    "Sardijn@nl",
    "Sardiniër@nl",
    "Sard@nl",
    "sardi@it",
    "sardos@sc",
    "sardo@it",
    "sarda@it",
    "sarde@it",
    "sardu@sc",
    "sarda@sc",
    "sardas@sc",
    "Sarde@de",
    "Sardin@de",
    "Sarden@de",
    "Sardinen@de"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Sardinen@de"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Sard@nl": 1,
      "Sarde@de": 1,
      "Sarden@de": 1,
      "Sardijn@nl": 1,
      "Sardin@de": 1,
      "Sardinen@de": 1,
      "Sardiniër@nl": 1,
      "sarda@it": 1,
      "sarda@sc": 1,
      "sardas@sc": 1,
      "sarde@it": 1,
      "sardi@it": 1,
      "sardo@it": 1,
      "sardos@sc": 1,
      "sardu@sc": 1
    },
    "new_unique": [
      "Sard@nl",
      "Sarde@de",
      "Sarden@de",
      "Sardijn@nl",
      "Sardin@de",
      "Sardinen@de",
      "Sardiniër@nl",
      "sarda@it",
      "sarda@sc",
      "sardas@sc",
      "sarde@it",
      "sardi@it",
      "sardo@it",
      "sardos@sc",
      "sardu@sc"
    ],
    "new_values": [
      "Sardijn@nl",
      "Sardiniër@nl",
      "Sard@nl",
      "sardi@it",
      "sardos@sc",
      "sardo@it",
      "sarda@it",
      "sarde@it",
      "sardu@sc",
      "sarda@sc",
      "sardas@sc",
      "Sarde@de",
      "Sardin@de",
      "Sarden@de",
      "Sardinen@de"
    ],
    "new_values_raw": [
      "Sardijn@nl",
      "Sardiniër@nl",
      "Sard@nl",
      "sardi@it",
      "sardos@sc",
      "sardo@it",
      "sarda@it",
      "sarde@it",
      "sardu@sc",
      "sarda@sc",
      "sardas@sc",
      "Sarde@de",
      "Sardin@de",
      "Sarden@de",
      "Sardinen@de"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Sard@nl": 1,
      "Sarde@de": 1,
      "Sarden@de": 1,
      "Sardijn@nl": 1,
      "Sardin@de": 1,
      "Sardiniër@nl": 1,
      "Sardinnen@de": 1,
      "sarda@it": 1,
      "sarda@sc": 1,
      "sardas@sc": 1,
      "sarde@it": 1,
      "sardi@it": 1,
      "sardo@it": 1,
      "sardos@sc": 1,
      "sardu@sc": 1
    },
    "old_unique": [
      "Sard@nl",
      "Sarde@de",
      "Sarden@de",
      "Sardijn@nl",
      "Sardin@de",
      "Sardiniër@nl",
      "Sardinnen@de",
      "sarda@it",
      "sarda@sc",
      "sardas@sc",
      "sarde@it",
      "sardi@it",
      "sardo@it",
      "sardos@sc",
      "sardu@sc"
    ],
    "old_values": [
      "Sardijn@nl",
      "Sardiniër@nl",
      "Sard@nl",
      "sardi@it",
      "sardos@sc",
      "sardo@it",
      "sarda@it",
      "sarde@it",
      "sardu@sc",
      "sarda@sc",
      "sardas@sc",
      "Sarde@de",
      "Sardin@de",
      "Sarden@de",
      "Sardinnen@de"
    ],
    "old_values_raw": [
      "Sardijn@nl",
      "Sardiniër@nl",
      "Sard@nl",
      "sardi@it",
      "sardos@sc",
      "sardo@it",
      "sarda@it",
      "sarde@it",
      "sardu@sc",
      "sarda@sc",
      "sardas@sc",
      "Sarde@de",
      "Sardin@de",
      "Sarden@de",
      "Sardinnen@de"
    ],
    "removed_unique_values": [
      "Sardinnen@de"
    ],
    "retained_unique_values": [
      "Sard@nl",
      "Sarde@de",
      "Sarden@de",
      "Sardijn@nl",
      "Sardin@de",
      "Sardiniër@nl",
      "sarda@it",
      "sarda@sc",
      "sardas@sc",
      "sarde@it",
      "sardi@it",
      "sardo@it",
      "sardos@sc",
      "sardu@sc"
    ],
    "semantic_action": "MIXED_UPDATE",
    "value_multiplicity_changes": {
      "Sardinen@de": {
        "new": 1,
        "old": 0
      },
      "Sardinnen@de": {
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
  "report_fix_date": "2025-12-17T10:13:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1549",
  "report_revision_new": 2443362146,
  "report_revision_old": 2442954081,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Sardijn@nl",
    "Sardiniër@nl",
    "Sard@nl",
    "sardi@it",
    "sardos@sc",
    "sardo@it",
    "sarda@it",
    "sarde@it",
    "sardu@sc",
    "sarda@sc",
    "sardas@sc",
    "Sarde@de",
    "Sardin@de",
    "Sarden@de",
    "Sardinnen@de"
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
    "Sardijn@nl",
    "Sardiniër@nl",
    "Sard@nl",
    "sardi@it",
    "sardos@sc",
    "sardo@it",
    "sarda@it",
    "sarde@it",
    "sardu@sc",
    "sarda@sc",
    "sardas@sc",
    "Sarde@de",
    "Sardin@de",
    "Sarden@de",
    "Sardinen@de"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "name for people or things associated with a given place, usually based off the place name",
    "label": "demonym"
  },
  "qid": {
    "description": "autonomous region of Italy",
    "label": "Sardinia"
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
        "Sardinen@de"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Sard@nl": 1,
        "Sarde@de": 1,
        "Sarden@de": 1,
        "Sardijn@nl": 1,
        "Sardin@de": 1,
        "Sardinen@de": 1,
        "Sardiniër@nl": 1,
        "sarda@it": 1,
        "sarda@sc": 1,
        "sardas@sc": 1,
        "sarde@it": 1,
        "sardi@it": 1,
        "sardo@it": 1,
        "sardos@sc": 1,
        "sardu@sc": 1
      },
      "new_unique": [
        "Sard@nl",
        "Sarde@de",
        "Sarden@de",
        "Sardijn@nl",
        "Sardin@de",
        "Sardinen@de",
        "Sardiniër@nl",
        "sarda@it",
        "sarda@sc",
        "sardas@sc",
        "sarde@it",
        "sardi@it",
        "sardo@it",
        "sardos@sc",
        "sardu@sc"
      ],
      "new_values": [
        "Sardijn@nl",
        "Sardiniër@nl",
        "Sard@nl",
        "sardi@it",
        "sardos@sc",
        "sardo@it",
        "sarda@it",
        "sarde@it",
        "sardu@sc",
        "sarda@sc",
        "sardas@sc",
        "Sarde@de",
        "Sardin@de",
        "Sarden@de",
        "Sardinen@de"
      ],
      "new_values_raw": [
        "Sardijn@nl",
        "Sardiniër@nl",
        "Sard@nl",
        "sardi@it",
        "sardos@sc",
        "sardo@it",
        "sarda@it",
        "sarde@it",
        "sardu@sc",
        "sarda@sc",
        "sardas@sc",
        "Sarde@de",
        "Sardin@de",
        "Sarden@de",
        "Sardinen@de"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Sard@nl": 1,
        "Sarde@de": 1,
        "Sarden@de": 1,
        "Sardijn@nl": 1,
        "Sardin@de": 1,
        "Sardiniër@nl": 1,
        "Sardinnen@de": 1,
        "sarda@it": 1,
        "sarda@sc": 1,
        "sardas@sc": 1,
        "sarde@it": 1,
        "sardi@it": 1,
        "sardo@it": 1,
        "sardos@sc": 1,
        "sardu@sc": 1
      },
      "old_unique": [
        "Sard@nl",
        "Sarde@de",
        "Sarden@de",
        "Sardijn@nl",
        "Sardin@de",
        "Sardiniër@nl",
        "Sardinnen@de",
        "sarda@it",
        "sarda@sc",
        "sardas@sc",
        "sarde@it",
        "sardi@it",
        "sardo@it",
        "sardos@sc",
        "sardu@sc"
      ],
      "old_values": [
        "Sardijn@nl",
        "Sardiniër@nl",
        "Sard@nl",
        "sardi@it",
        "sardos@sc",
        "sardo@it",
        "sarda@it",
        "sarde@it",
        "sardu@sc",
        "sarda@sc",
        "sardas@sc",
        "Sarde@de",
        "Sardin@de",
        "Sarden@de",
        "Sardinnen@de"
      ],
      "old_values_raw": [
        "Sardijn@nl",
        "Sardiniër@nl",
        "Sard@nl",
        "sardi@it",
        "sardos@sc",
        "sardo@it",
        "sarda@it",
        "sarde@it",
        "sardu@sc",
        "sarda@sc",
        "sardas@sc",
        "Sarde@de",
        "Sardin@de",
        "Sarden@de",
        "Sardinnen@de"
      ],
      "removed_unique_values": [
        "Sardinnen@de"
      ],
      "retained_unique_values": [
        "Sard@nl",
        "Sarde@de",
        "Sarden@de",
        "Sardijn@nl",
        "Sardin@de",
        "Sardiniër@nl",
        "sarda@it",
        "sarda@sc",
        "sardas@sc",
        "sarde@it",
        "sardi@it",
        "sardo@it",
        "sardos@sc",
        "sardu@sc"
      ],
      "semantic_action": "MIXED_UPDATE",
      "value_multiplicity_changes": {
        "Sardinen@de": {
          "new": 1,
          "old": 0
        },
        "Sardinnen@de": {
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
      "new_unique": [
        "Sard@nl",
        "Sarde@de",
        "Sarden@de",
        "Sardijn@nl",
        "Sardin@de",
        "Sardinen@de",
        "Sardiniër@nl",
        "sarda@it",
        "sarda@sc",
        "sardas@sc",
        "sarde@it",
        "sardi@it",
        "sardo@it",
        "sardos@sc",
        "sardu@sc"
      ],
      "report_type": "single value"
    },
    "kind": "SINGLE_VALUE_MULTIPLE_NEW_VALUES",
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
    "result": "single_value_report_multiple_new_values",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q18254725_704618355`

| Field | Value |
|---|---|
| qid | Q18254725 |
| property | P671 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_BAD_TARGET_OR_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_bad_target_or_context |
| decision_constraint_type |   |
| group_key | ABOX::Q18254725::P671 |
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
| truth_tokens_preview | ["MGI:104013", "MGI:104012"] |
| classification_target_tokens | ["MGI:104012"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | single_value_report_multiple_new_values |
| rationale | Single-value violation is followed by multiple created/added target values; treated as report-context mismatch or bad target rather than clean external evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "MGI:104012"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "MGI:104012"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "MGI:104013"
  ],
  "retained_unique_values": [
    "MGI:104013"
  ],
  "semantic_action": "ADD_SUPERSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_bad_target_or_context",
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
  "author": "Nonoxb",
  "kind": "A_BOX",
  "new_value": [
    "MGI:104013",
    "MGI:104012"
  ],
  "old_value": [
    "MGI:104013"
  ],
  "revision_id": 704618355,
  "value": [
    "MGI:104013",
    "MGI:104012"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "MGI:104012"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "MGI:104012": 1,
      "MGI:104013": 1
    },
    "new_unique": [
      "MGI:104012",
      "MGI:104013"
    ],
    "new_values": [
      "MGI:104013",
      "MGI:104012"
    ],
    "new_values_raw": [
      "MGI:104013",
      "MGI:104012"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "MGI:104013": 1
    },
    "old_unique": [
      "MGI:104013"
    ],
    "old_values": [
      "MGI:104013"
    ],
    "old_values_raw": [
      "MGI:104013"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "MGI:104013"
    ],
    "semantic_action": "ADD_SUPERSET",
    "value_multiplicity_changes": {
      "MGI:104012": {
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
  "report_fix_date": "2018-07-01T19:27:23",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P671",
  "report_revision_new": 705158681,
  "report_revision_old": 704830146,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "MGI:104013"
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
    "MGI:104013",
    "MGI:104012"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a gene in the Mouse Genome Informatics database",
    "label": "Mouse Genome Informatics ID"
  },
  "qid": {
    "description": "mouse gene",
    "label": "T(8C3;16B5)164Dn"
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
        "MGI:104012"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "MGI:104012": 1,
        "MGI:104013": 1
      },
      "new_unique": [
        "MGI:104012",
        "MGI:104013"
      ],
      "new_values": [
        "MGI:104013",
        "MGI:104012"
      ],
      "new_values_raw": [
        "MGI:104013",
        "MGI:104012"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "MGI:104013": 1
      },
      "old_unique": [
        "MGI:104013"
      ],
      "old_values": [
        "MGI:104013"
      ],
      "old_values_raw": [
        "MGI:104013"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "MGI:104013"
      ],
      "semantic_action": "ADD_SUPERSET",
      "value_multiplicity_changes": {
        "MGI:104012": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "ADD_SUPERSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "new_unique": [
        "MGI:104012",
        "MGI:104013"
      ],
      "report_type": "single value"
    },
    "kind": "SINGLE_VALUE_MULTIPLE_NEW_VALUES",
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
    "result": "single_value_report_multiple_new_values",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q27108472_2425154288`

| Field | Value |
|---|---|
| qid | Q27108472 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_BAD_TARGET_OR_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_bad_target_or_context |
| decision_constraint_type |   |
| group_key | ABOX::Q27108472::P2877 |
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
| truth_tokens_preview | ["10595245", "29406006"] |
| classification_target_tokens | ["10595245", "29406006"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | single_value_report_multiple_new_values |
| rationale | Single-value violation is followed by multiple created/added target values; treated as report-context mismatch or bad target rather than clean external evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "10595245",
    "29406006"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "10595245",
    "29406006"
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
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_bad_target_or_context",
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
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "10595245",
    "29406006"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2425154288,
  "value": [
    "10595245",
    "29406006"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "10595245",
      "29406006"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "10595245": 1,
      "29406006": 1
    },
    "new_unique": [
      "10595245",
      "29406006"
    ],
    "new_values": [
      "10595245",
      "29406006"
    ],
    "new_values_raw": [
      "10595245",
      "29406006"
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
      "10595245": {
        "new": 1,
        "old": 0
      },
      "29406006": {
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
  "report_fix_date": "2025-11-07T07:50:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426830853,
  "report_revision_old": 2426458092,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
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
    "10595245",
    "29406006"
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
    "label": "Tricrozarin A"
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
        "10595245",
        "29406006"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "10595245": 1,
        "29406006": 1
      },
      "new_unique": [
        "10595245",
        "29406006"
      ],
      "new_values": [
        "10595245",
        "29406006"
      ],
      "new_values_raw": [
        "10595245",
        "29406006"
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
        "10595245": {
          "new": 1,
          "old": 0
        },
        "29406006": {
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
      "new_unique": [
        "10595245",
        "29406006"
      ],
      "report_type": "single value"
    },
    "kind": "SINGLE_VALUE_MULTIPLE_NEW_VALUES",
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
    "result": "single_value_report_multiple_new_values",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q27861942_2424784938`

| Field | Value |
|---|---|
| qid | Q27861942 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_BAD_TARGET_OR_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_bad_target_or_context |
| decision_constraint_type |   |
| group_key | ABOX::Q27861942::P2877 |
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
| truth_tokens_preview | ["27561558", "19671", "599693"] |
| classification_target_tokens | ["19671", "27561558", "599693"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | single_value_report_multiple_new_values |
| rationale | Single-value violation is followed by multiple created/added target values; treated as report-context mismatch or bad target rather than clean external evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "19671",
    "27561558",
    "599693"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "19671",
    "27561558",
    "599693"
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
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_bad_target_or_context",
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
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "27561558",
    "19671",
    "599693"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2424784938,
  "value": [
    "27561558",
    "19671",
    "599693"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "19671",
      "27561558",
      "599693"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "19671": 1,
      "27561558": 1,
      "599693": 1
    },
    "new_unique": [
      "19671",
      "27561558",
      "599693"
    ],
    "new_values": [
      "27561558",
      "19671",
      "599693"
    ],
    "new_values_raw": [
      "27561558",
      "19671",
      "599693"
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
      "19671": {
        "new": 1,
        "old": 0
      },
      "27561558": {
        "new": 1,
        "old": 0
      },
      "599693": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
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
    "27561558",
    "19671",
    "599693"
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
    "label": "aluminium(III) triacetate"
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
        "19671",
        "27561558",
        "599693"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "19671": 1,
        "27561558": 1,
        "599693": 1
      },
      "new_unique": [
        "19671",
        "27561558",
        "599693"
      ],
      "new_values": [
        "27561558",
        "19671",
        "599693"
      ],
      "new_values_raw": [
        "27561558",
        "19671",
        "599693"
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
        "19671": {
          "new": 1,
          "old": 0
        },
        "27561558": {
          "new": 1,
          "old": 0
        },
        "599693": {
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
      "new_unique": [
        "19671",
        "27561558",
        "599693"
      ],
      "report_type": "single value"
    },
    "kind": "SINGLE_VALUE_MULTIPLE_NEW_VALUES",
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
    "result": "single_value_report_multiple_new_values",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q3156354_2086430607`

| Field | Value |
|---|---|
| qid | Q3156354 |
| property | P3795 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_BAD_TARGET_OR_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_bad_target_or_context |
| decision_constraint_type |   |
| group_key | ABOX::Q3156354::P3795 |
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
| truth_tokens_preview | ["LOLRIG/"] |
| classification_target_tokens | ["LOLRIG/"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | unknown_bad_target_or_context |
| rationale | Format update moves from a regex-valid value to a regex-invalid value; target or context is suspect. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "LOLRIG/"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "LOLRIG/"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "LOLRIG"
  ],
  "removed_unique_values": [
    "LOLRIG"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_bad_target_or_context",
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
    "LOLRIG/"
  ],
  "old_value": [
    "LOLRIG"
  ],
  "revision_id": 2086430607,
  "value": [
    "LOLRIG/"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "LOLRIG/"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "LOLRIG/": 1
    },
    "new_unique": [
      "LOLRIG/"
    ],
    "new_values": [
      "LOLRIG/"
    ],
    "new_values_raw": [
      "LOLRIG/"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "LOLRIG": 1
    },
    "old_unique": [
      "LOLRIG"
    ],
    "old_values": [
      "LOLRIG"
    ],
    "old_values_raw": [
      "LOLRIG"
    ],
    "removed_unique_values": [
      "LOLRIG"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "LOLRIG": {
        "new": 0,
        "old": 1
      },
      "LOLRIG/": {
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
  "report_fix_date": "2024-02-26T12:55:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P3795",
  "report_revision_new": 2087785506,
  "report_revision_old": 2082360484,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "LOLRIG"
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
    "LOLRIG/"
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
    "label": "Lolium rigidum"
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
        "LOLRIG/"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "LOLRIG/": 1
      },
      "new_unique": [
        "LOLRIG/"
      ],
      "new_values": [
        "LOLRIG/"
      ],
      "new_values_raw": [
        "LOLRIG/"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "LOLRIG": 1
      },
      "old_unique": [
        "LOLRIG"
      ],
      "old_values": [
        "LOLRIG"
      ],
      "old_values_raw": [
        "LOLRIG"
      ],
      "removed_unique_values": [
        "LOLRIG"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "LOLRIG": {
          "new": 0,
          "old": 1
        },
        "LOLRIG/": {
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
      "new_changed": "LOLRIG/",
      "new_pass_regex": false,
      "new_value": "LOLRIG/",
      "normalization_kind": "append_trailing_slash",
      "normalization_rule": "append_trailing_slash",
      "old_changed": "LOLRIG",
      "old_pass_regex": true,
      "old_value": "LOLRIG",
      "reason": "old_value_passes_regex_new_value_fails",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "unknown_bad_target_or_context",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q3273836_2086430809`

| Field | Value |
|---|---|
| qid | Q3273836 |
| property | P3795 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_BAD_TARGET_OR_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_bad_target_or_context |
| decision_constraint_type |   |
| group_key | ABOX::Q3273836::P3795 |
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
| truth_tokens_preview | ["PHYNOD/"] |
| classification_target_tokens | ["PHYNOD/"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | unknown_bad_target_or_context |
| rationale | Format update moves from a regex-valid value to a regex-invalid value; target or context is suspect. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "PHYNOD/"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "PHYNOD/"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "PHYNOD"
  ],
  "removed_unique_values": [
    "PHYNOD"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_bad_target_or_context",
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
    "PHYNOD/"
  ],
  "old_value": [
    "PHYNOD"
  ],
  "revision_id": 2086430809,
  "value": [
    "PHYNOD/"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "PHYNOD/"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "PHYNOD/": 1
    },
    "new_unique": [
      "PHYNOD/"
    ],
    "new_values": [
      "PHYNOD/"
    ],
    "new_values_raw": [
      "PHYNOD/"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "PHYNOD": 1
    },
    "old_unique": [
      "PHYNOD"
    ],
    "old_values": [
      "PHYNOD"
    ],
    "old_values_raw": [
      "PHYNOD"
    ],
    "removed_unique_values": [
      "PHYNOD"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "PHYNOD": {
        "new": 0,
        "old": 1
      },
      "PHYNOD/": {
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
  "report_fix_date": "2024-02-26T12:55:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P3795",
  "report_revision_new": 2087785506,
  "report_revision_old": 2082360484,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "PHYNOD"
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
    "PHYNOD/"
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
    "label": "Phyla nodiflora"
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
        "PHYNOD/"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "PHYNOD/": 1
      },
      "new_unique": [
        "PHYNOD/"
      ],
      "new_values": [
        "PHYNOD/"
      ],
      "new_values_raw": [
        "PHYNOD/"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "PHYNOD": 1
      },
      "old_unique": [
        "PHYNOD"
      ],
      "old_values": [
        "PHYNOD"
      ],
      "old_values_raw": [
        "PHYNOD"
      ],
      "removed_unique_values": [
        "PHYNOD"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "PHYNOD": {
          "new": 0,
          "old": 1
        },
        "PHYNOD/": {
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
      "new_changed": "PHYNOD/",
      "new_pass_regex": false,
      "new_value": "PHYNOD/",
      "normalization_kind": "append_trailing_slash",
      "normalization_rule": "append_trailing_slash",
      "old_changed": "PHYNOD",
      "old_pass_regex": true,
      "old_value": "PHYNOD",
      "reason": "old_value_passes_regex_new_value_fails",
      "regexes_present": true
    },
    "kind": "FORMAT_NORMALIZATION",
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
    "result": "unknown_bad_target_or_context",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q72434839_2424832256`

| Field | Value |
|---|---|
| qid | Q72434839 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_BAD_TARGET_OR_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_bad_target_or_context |
| decision_constraint_type |   |
| group_key | ABOX::Q72434839::P2877 |
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
| truth_tokens_preview | ["98338", "29498305"] |
| classification_target_tokens | ["29498305", "98338"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | single_value_report_multiple_new_values |
| rationale | Single-value violation is followed by multiple created/added target values; treated as report-context mismatch or bad target rather than clean external evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "29498305",
    "98338"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "29498305",
    "98338"
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
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_bad_target_or_context",
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
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "98338",
    "29498305"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2424832256,
  "value": [
    "98338",
    "29498305"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "29498305",
      "98338"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "29498305": 1,
      "98338": 1
    },
    "new_unique": [
      "29498305",
      "98338"
    ],
    "new_values": [
      "98338",
      "29498305"
    ],
    "new_values_raw": [
      "98338",
      "29498305"
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
      "29498305": {
        "new": 1,
        "old": 0
      },
      "98338": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
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
    "98338",
    "29498305"
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
    "label": "5-Acetyl-8-benzyloxy-1H-quinolin-2-one"
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
        "29498305",
        "98338"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "29498305": 1,
        "98338": 1
      },
      "new_unique": [
        "29498305",
        "98338"
      ],
      "new_values": [
        "98338",
        "29498305"
      ],
      "new_values_raw": [
        "98338",
        "29498305"
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
        "29498305": {
          "new": 1,
          "old": 0
        },
        "98338": {
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
      "new_unique": [
        "29498305",
        "98338"
      ],
      "report_type": "single value"
    },
    "kind": "SINGLE_VALUE_MULTIPLE_NEW_VALUES",
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
    "result": "single_value_report_multiple_new_values",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q72462571_2424917231`

| Field | Value |
|---|---|
| qid | Q72462571 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_BAD_TARGET_OR_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_bad_target_or_context |
| decision_constraint_type |   |
| group_key | ABOX::Q72462571::P2877 |
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
| truth_tokens_preview | ["102964", "17697293"] |
| classification_target_tokens | ["102964", "17697293"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | single_value_report_multiple_new_values |
| rationale | Single-value violation is followed by multiple created/added target values; treated as report-context mismatch or bad target rather than clean external evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "102964",
    "17697293"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "102964",
    "17697293"
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
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_bad_target_or_context",
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
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "102964",
    "17697293"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2424917231,
  "value": [
    "102964",
    "17697293"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "102964",
      "17697293"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "102964": 1,
      "17697293": 1
    },
    "new_unique": [
      "102964",
      "17697293"
    ],
    "new_values": [
      "102964",
      "17697293"
    ],
    "new_values_raw": [
      "102964",
      "17697293"
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
      "102964": {
        "new": 1,
        "old": 0
      },
      "17697293": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
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
    "102964",
    "17697293"
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
    "label": "5,6-Difluoro-1H-benzotriazole"
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
        "102964",
        "17697293"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "102964": 1,
        "17697293": 1
      },
      "new_unique": [
        "102964",
        "17697293"
      ],
      "new_values": [
        "102964",
        "17697293"
      ],
      "new_values_raw": [
        "102964",
        "17697293"
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
        "102964": {
          "new": 1,
          "old": 0
        },
        "17697293": {
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
      "new_unique": [
        "102964",
        "17697293"
      ],
      "report_type": "single value"
    },
    "kind": "SINGLE_VALUE_MULTIPLE_NEW_VALUES",
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
    "result": "single_value_report_multiple_new_values",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q72494996_2424564036`

| Field | Value |
|---|---|
| qid | Q72494996 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_BAD_TARGET_OR_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_bad_target_or_context |
| decision_constraint_type |   |
| group_key | ABOX::Q72494996::P2877 |
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
| truth_tokens_preview | ["1843227", "2971895", "8589087", "13804583"] |
| classification_target_tokens | ["13804583", "1843227", "2971895", "8589087"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | single_value_report_multiple_new_values |
| rationale | Single-value violation is followed by multiple created/added target values; treated as report-context mismatch or bad target rather than clean external evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "13804583",
    "1843227",
    "2971895",
    "8589087"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "13804583",
    "1843227",
    "2971895",
    "8589087"
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
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_bad_target_or_context",
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
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "1843227",
    "2971895",
    "8589087",
    "13804583"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2424564036,
  "value": [
    "1843227",
    "2971895",
    "8589087",
    "13804583"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "13804583",
      "1843227",
      "2971895",
      "8589087"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "13804583": 1,
      "1843227": 1,
      "2971895": 1,
      "8589087": 1
    },
    "new_unique": [
      "13804583",
      "1843227",
      "2971895",
      "8589087"
    ],
    "new_values": [
      "1843227",
      "2971895",
      "8589087",
      "13804583"
    ],
    "new_values_raw": [
      "1843227",
      "2971895",
      "8589087",
      "13804583"
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
      "13804583": {
        "new": 1,
        "old": 0
      },
      "1843227": {
        "new": 1,
        "old": 0
      },
      "2971895": {
        "new": 1,
        "old": 0
      },
      "8589087": {
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
  "report_fix_date": "2025-11-05T07:45:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2425837272,
  "report_revision_old": 2425435358,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
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
    "1843227",
    "2971895",
    "8589087",
    "13804583"
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
    "label": "N-Nitro-S-methyl isothiourea"
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
        "13804583",
        "1843227",
        "2971895",
        "8589087"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "13804583": 1,
        "1843227": 1,
        "2971895": 1,
        "8589087": 1
      },
      "new_unique": [
        "13804583",
        "1843227",
        "2971895",
        "8589087"
      ],
      "new_values": [
        "1843227",
        "2971895",
        "8589087",
        "13804583"
      ],
      "new_values_raw": [
        "1843227",
        "2971895",
        "8589087",
        "13804583"
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
        "13804583": {
          "new": 1,
          "old": 0
        },
        "1843227": {
          "new": 1,
          "old": 0
        },
        "2971895": {
          "new": 1,
          "old": 0
        },
        "8589087": {
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
      "new_unique": [
        "13804583",
        "1843227",
        "2971895",
        "8589087"
      ],
      "report_type": "single value"
    },
    "kind": "SINGLE_VALUE_MULTIPLE_NEW_VALUES",
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
    "result": "single_value_report_multiple_new_values",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q87000952_2442340658`

| Field | Value |
|---|---|
| qid | Q87000952 |
| property | P2473 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_BAD_TARGET_OR_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_bad_target_or_context |
| decision_constraint_type |   |
| group_key | ABOX::Q87000952::P2473 |
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
| truth_tokens_preview | ["46.17.213-093", "46.213-9999-000022"] |
| classification_target_tokens | ["46.213-9999-000022"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | single_value_report_multiple_new_values |
| rationale | Single-value violation is followed by multiple created/added target values; treated as report-context mismatch or bad target rather than clean external evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "46.213-9999-000022"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "46.213-9999-000022"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "46.17.213-093"
  ],
  "retained_unique_values": [
    "46.17.213-093"
  ],
  "semantic_action": "ADD_SUPERSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_bad_target_or_context",
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
    "46.17.213-093",
    "46.213-9999-000022"
  ],
  "old_value": [
    "46.17.213-093"
  ],
  "revision_id": 2442340658,
  "value": [
    "46.17.213-093",
    "46.213-9999-000022"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "46.213-9999-000022"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "46.17.213-093": 1,
      "46.213-9999-000022": 1
    },
    "new_unique": [
      "46.17.213-093",
      "46.213-9999-000022"
    ],
    "new_values": [
      "46.17.213-093",
      "46.213-9999-000022"
    ],
    "new_values_raw": [
      "46.17.213-093",
      "46.213-9999-000022"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "46.17.213-093": 1
    },
    "old_unique": [
      "46.17.213-093"
    ],
    "old_values": [
      "46.17.213-093"
    ],
    "old_values_raw": [
      "46.17.213-093"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "46.17.213-093"
    ],
    "semantic_action": "ADD_SUPERSET",
    "value_multiplicity_changes": {
      "46.213-9999-000022": {
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
  "report_fix_date": "2025-12-19T08:58:40",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2473",
  "report_revision_new": 2444007257,
  "report_revision_old": 2443799780,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "report_violation_types": [
    "Single value",
    "Item P|18"
  ],
  "value": [
    "46.17.213-093"
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
    "46.17.213-093",
    "46.213-9999-000022"
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
    "description": "old agricultural settlement in Requena",
    "label": "Casas de Sisternas"
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
      "added_unique_values": [
        "46.213-9999-000022"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "46.17.213-093": 1,
        "46.213-9999-000022": 1
      },
      "new_unique": [
        "46.17.213-093",
        "46.213-9999-000022"
      ],
      "new_values": [
        "46.17.213-093",
        "46.213-9999-000022"
      ],
      "new_values_raw": [
        "46.17.213-093",
        "46.213-9999-000022"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "46.17.213-093": 1
      },
      "old_unique": [
        "46.17.213-093"
      ],
      "old_values": [
        "46.17.213-093"
      ],
      "old_values_raw": [
        "46.17.213-093"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "46.17.213-093"
      ],
      "semantic_action": "ADD_SUPERSET",
      "value_multiplicity_changes": {
        "46.213-9999-000022": {
          "new": 1,
          "old": 0
        }
      }
    },
    "result": "ADD_SUPERSET",
    "step": "value_delta"
  },
  {
    "detail": {
      "new_unique": [
        "46.17.213-093",
        "46.213-9999-000022"
      ],
      "report_type": "single value"
    },
    "kind": "SINGLE_VALUE_MULTIPLE_NEW_VALUES",
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
    "result": "single_value_report_multiple_new_values",
    "step": "branch"
  }
]
```

---
