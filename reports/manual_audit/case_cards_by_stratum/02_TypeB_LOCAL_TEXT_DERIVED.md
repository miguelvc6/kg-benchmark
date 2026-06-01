# TypeB_LOCAL_TEXT_DERIVED

Cases: 22

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q100536294_1295663860`

| Field | Value |
|---|---|
| qid | Q100536294 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100536294::P8726 |
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
| truth_tokens_preview | ["2007/si/278/made"] |
| classification_target_tokens | ["2007/si/278/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2007/si/278/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2007/si/278/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2007/si/278/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295663860,
  "value": [
    "2007/si/278/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2007/si/278/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2007/si/278/made": 1
    },
    "new_unique": [
      "2007/si/278/made"
    ],
    "new_values": [
      "2007/si/278/made"
    ],
    "new_values_raw": [
      "2007/si/278/made"
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
      "2007/si/278/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2007/si/278/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2007/si/278/made",
      "extracted_number": "278",
      "extracted_year": "2007",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 278/2007",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 278/2007"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2007/si/278/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 278/2007",
    "label": "European Communities (Drinking Water) (No. 2) Regulations 2007"
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
      "added_unique_values": [
        "2007/si/278/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2007/si/278/made": 1
      },
      "new_unique": [
        "2007/si/278/made"
      ],
      "new_values": [
        "2007/si/278/made"
      ],
      "new_values_raw": [
        "2007/si/278/made"
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
        "2007/si/278/made": {
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
    "detail": {
      "actual_target": "2007/si/278/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2007/si/278/made",
      "extracted_number": "278",
      "extracted_year": "2007",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 278/2007",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 278/2007"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2007/si/278/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2007/si/278/made",
          "extracted_number": "278",
          "extracted_year": "2007",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 278/2007",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 278/2007"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q100536913_1295668128`

| Field | Value |
|---|---|
| qid | Q100536913 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100536913::P8726 |
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
| truth_tokens_preview | ["2007/si/436/made"] |
| classification_target_tokens | ["2007/si/436/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2007/si/436/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2007/si/436/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2007/si/436/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295668128,
  "value": [
    "2007/si/436/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2007/si/436/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2007/si/436/made": 1
    },
    "new_unique": [
      "2007/si/436/made"
    ],
    "new_values": [
      "2007/si/436/made"
    ],
    "new_values_raw": [
      "2007/si/436/made"
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
      "2007/si/436/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2007/si/436/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2007/si/436/made",
      "extracted_number": "436",
      "extracted_year": "2007",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 436/2007",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 436/2007"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2007/si/436/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 436/2007",
    "label": "Taxes Consolidation Act 1997 (Qualifying town Renewal Areas) (Sixmilebridge, County Clare) (Amendment) Order 2007"
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
      "added_unique_values": [
        "2007/si/436/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2007/si/436/made": 1
      },
      "new_unique": [
        "2007/si/436/made"
      ],
      "new_values": [
        "2007/si/436/made"
      ],
      "new_values_raw": [
        "2007/si/436/made"
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
        "2007/si/436/made": {
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
    "detail": {
      "actual_target": "2007/si/436/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2007/si/436/made",
      "extracted_number": "436",
      "extracted_year": "2007",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 436/2007",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 436/2007"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2007/si/436/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2007/si/436/made",
          "extracted_number": "436",
          "extracted_year": "2007",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 436/2007",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 436/2007"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q100537367_1295674942`

| Field | Value |
|---|---|
| qid | Q100537367 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100537367::P8726 |
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
| truth_tokens_preview | ["2007/si/621/made"] |
| classification_target_tokens | ["2007/si/621/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2007/si/621/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2007/si/621/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2007/si/621/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295674942,
  "value": [
    "2007/si/621/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2007/si/621/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2007/si/621/made": 1
    },
    "new_unique": [
      "2007/si/621/made"
    ],
    "new_values": [
      "2007/si/621/made"
    ],
    "new_values_raw": [
      "2007/si/621/made"
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
      "2007/si/621/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2007/si/621/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2007/si/621/made",
      "extracted_number": "621",
      "extracted_year": "2007",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 621/2007",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 621/2007"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2007/si/621/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 621/2007",
    "label": "Trade Marks (Amendment) Rules 2007"
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
      "added_unique_values": [
        "2007/si/621/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2007/si/621/made": 1
      },
      "new_unique": [
        "2007/si/621/made"
      ],
      "new_values": [
        "2007/si/621/made"
      ],
      "new_values_raw": [
        "2007/si/621/made"
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
        "2007/si/621/made": {
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
    "detail": {
      "actual_target": "2007/si/621/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2007/si/621/made",
      "extracted_number": "621",
      "extracted_year": "2007",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 621/2007",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 621/2007"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2007/si/621/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2007/si/621/made",
          "extracted_number": "621",
          "extracted_year": "2007",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 621/2007",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 621/2007"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q100537571_1295676959`

| Field | Value |
|---|---|
| qid | Q100537571 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100537571::P8726 |
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
| truth_tokens_preview | ["2013/si/523/made"] |
| classification_target_tokens | ["2013/si/523/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2013/si/523/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2013/si/523/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2013/si/523/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295676959,
  "value": [
    "2013/si/523/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2013/si/523/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2013/si/523/made": 1
    },
    "new_unique": [
      "2013/si/523/made"
    ],
    "new_values": [
      "2013/si/523/made"
    ],
    "new_values_raw": [
      "2013/si/523/made"
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
      "2013/si/523/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2013/si/523/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2013/si/523/made",
      "extracted_number": "523",
      "extracted_year": "2013",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 523/2013",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 523/2013"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2013/si/523/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 523/2013",
    "label": "Voluntary Health Insurance (Amendment) Act 2008 (Appointment of date pursuant to subsection (5)(b) of section 2 of the Voluntary Health Insurance (Amendment) Act 1996) Order 2013"
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
      "added_unique_values": [
        "2013/si/523/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2013/si/523/made": 1
      },
      "new_unique": [
        "2013/si/523/made"
      ],
      "new_values": [
        "2013/si/523/made"
      ],
      "new_values_raw": [
        "2013/si/523/made"
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
        "2013/si/523/made": {
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
    "detail": {
      "actual_target": "2013/si/523/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2013/si/523/made",
      "extracted_number": "523",
      "extracted_year": "2013",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 523/2013",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 523/2013"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2013/si/523/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2013/si/523/made",
          "extracted_number": "523",
          "extracted_year": "2013",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 523/2013",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 523/2013"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q100538337_1295690533`

| Field | Value |
|---|---|
| qid | Q100538337 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100538337::P8726 |
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
| truth_tokens_preview | ["2014/si/146/made"] |
| classification_target_tokens | ["2014/si/146/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2014/si/146/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2014/si/146/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2014/si/146/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295690533,
  "value": [
    "2014/si/146/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2014/si/146/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2014/si/146/made": 1
    },
    "new_unique": [
      "2014/si/146/made"
    ],
    "new_values": [
      "2014/si/146/made"
    ],
    "new_values_raw": [
      "2014/si/146/made"
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
      "2014/si/146/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2014/si/146/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2014/si/146/made",
      "extracted_number": "146",
      "extracted_year": "2014",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 146/2014",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 146/2014"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2014/si/146/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 146/2014",
    "label": "Local Government Reform Act 2014 (Commencement of Certain Provisions) (No. 2) Order 2014"
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
      "added_unique_values": [
        "2014/si/146/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2014/si/146/made": 1
      },
      "new_unique": [
        "2014/si/146/made"
      ],
      "new_values": [
        "2014/si/146/made"
      ],
      "new_values_raw": [
        "2014/si/146/made"
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
        "2014/si/146/made": {
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
    "detail": {
      "actual_target": "2014/si/146/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2014/si/146/made",
      "extracted_number": "146",
      "extracted_year": "2014",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 146/2014",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 146/2014"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2014/si/146/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2014/si/146/made",
          "extracted_number": "146",
          "extracted_year": "2014",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 146/2014",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 146/2014"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q100538432_1295690908`

| Field | Value |
|---|---|
| qid | Q100538432 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100538432::P8726 |
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
| truth_tokens_preview | ["2014/si/173/made"] |
| classification_target_tokens | ["2014/si/173/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2014/si/173/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2014/si/173/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2014/si/173/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295690908,
  "value": [
    "2014/si/173/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2014/si/173/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2014/si/173/made": 1
    },
    "new_unique": [
      "2014/si/173/made"
    ],
    "new_values": [
      "2014/si/173/made"
    ],
    "new_values_raw": [
      "2014/si/173/made"
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
      "2014/si/173/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2014/si/173/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2014/si/173/made",
      "extracted_number": "173",
      "extracted_year": "2014",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 173/2014",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 173/2014"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2014/si/173/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 173/2014",
    "label": "County Enterprise Boards (Dissolution) Act 2014 (Section 2) (No. 4) Order 2014"
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
      "added_unique_values": [
        "2014/si/173/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2014/si/173/made": 1
      },
      "new_unique": [
        "2014/si/173/made"
      ],
      "new_values": [
        "2014/si/173/made"
      ],
      "new_values_raw": [
        "2014/si/173/made"
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
        "2014/si/173/made": {
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
    "detail": {
      "actual_target": "2014/si/173/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2014/si/173/made",
      "extracted_number": "173",
      "extracted_year": "2014",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 173/2014",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 173/2014"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2014/si/173/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2014/si/173/made",
          "extracted_number": "173",
          "extracted_year": "2014",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 173/2014",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 173/2014"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q100538731_1295666552`

| Field | Value |
|---|---|
| qid | Q100538731 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100538731::P8726 |
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
| truth_tokens_preview | ["2014/si/274/made"] |
| classification_target_tokens | ["2014/si/274/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2014/si/274/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2014/si/274/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2014/si/274/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295666552,
  "value": [
    "2014/si/274/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2014/si/274/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2014/si/274/made": 1
    },
    "new_unique": [
      "2014/si/274/made"
    ],
    "new_values": [
      "2014/si/274/made"
    ],
    "new_values_raw": [
      "2014/si/274/made"
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
      "2014/si/274/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2014/si/274/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2014/si/274/made",
      "extracted_number": "274",
      "extracted_year": "2014",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 274/2014",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 274/2014"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2014/si/274/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 274/2014",
    "label": "Education (Miscellaneous Provisions) Act 2007 (Commencement) Order 2014"
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
      "added_unique_values": [
        "2014/si/274/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2014/si/274/made": 1
      },
      "new_unique": [
        "2014/si/274/made"
      ],
      "new_values": [
        "2014/si/274/made"
      ],
      "new_values_raw": [
        "2014/si/274/made"
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
        "2014/si/274/made": {
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
    "detail": {
      "actual_target": "2014/si/274/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2014/si/274/made",
      "extracted_number": "274",
      "extracted_year": "2014",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 274/2014",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 274/2014"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2014/si/274/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2014/si/274/made",
          "extracted_number": "274",
          "extracted_year": "2014",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 274/2014",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 274/2014"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q100538984_1295670012`

| Field | Value |
|---|---|
| qid | Q100538984 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100538984::P8726 |
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
| truth_tokens_preview | ["2014/si/335/made"] |
| classification_target_tokens | ["2014/si/335/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2014/si/335/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2014/si/335/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2014/si/335/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295670012,
  "value": [
    "2014/si/335/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2014/si/335/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2014/si/335/made": 1
    },
    "new_unique": [
      "2014/si/335/made"
    ],
    "new_values": [
      "2014/si/335/made"
    ],
    "new_values_raw": [
      "2014/si/335/made"
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
      "2014/si/335/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2014/si/335/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2014/si/335/made",
      "extracted_number": "335",
      "extracted_year": "2014",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 335/2014",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 335/2014"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2014/si/335/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 335/2014",
    "label": "Central Bank Act 1942 (Section 32D) Regulations 2014"
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
      "added_unique_values": [
        "2014/si/335/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2014/si/335/made": 1
      },
      "new_unique": [
        "2014/si/335/made"
      ],
      "new_values": [
        "2014/si/335/made"
      ],
      "new_values_raw": [
        "2014/si/335/made"
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
        "2014/si/335/made": {
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
    "detail": {
      "actual_target": "2014/si/335/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2014/si/335/made",
      "extracted_number": "335",
      "extracted_year": "2014",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 335/2014",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 335/2014"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2014/si/335/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2014/si/335/made",
          "extracted_number": "335",
          "extracted_year": "2014",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 335/2014",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 335/2014"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q100539627_1295668067`

| Field | Value |
|---|---|
| qid | Q100539627 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100539627::P8726 |
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
| truth_tokens_preview | ["2014/si/489/made"] |
| classification_target_tokens | ["2014/si/489/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2014/si/489/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2014/si/489/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2014/si/489/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295668067,
  "value": [
    "2014/si/489/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2014/si/489/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2014/si/489/made": 1
    },
    "new_unique": [
      "2014/si/489/made"
    ],
    "new_values": [
      "2014/si/489/made"
    ],
    "new_values_raw": [
      "2014/si/489/made"
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
      "2014/si/489/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2014/si/489/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2014/si/489/made",
      "extracted_number": "489",
      "extracted_year": "2014",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 489/2014",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 489/2014"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2014/si/489/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 489/2014",
    "label": "Health (Provision of Food Allergen Information to Consumers in respect of Non-Prepacked Food) Regulations 2014"
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
      "added_unique_values": [
        "2014/si/489/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2014/si/489/made": 1
      },
      "new_unique": [
        "2014/si/489/made"
      ],
      "new_values": [
        "2014/si/489/made"
      ],
      "new_values_raw": [
        "2014/si/489/made"
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
        "2014/si/489/made": {
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
    "detail": {
      "actual_target": "2014/si/489/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2014/si/489/made",
      "extracted_number": "489",
      "extracted_year": "2014",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 489/2014",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 489/2014"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2014/si/489/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2014/si/489/made",
          "extracted_number": "489",
          "extracted_year": "2014",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 489/2014",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 489/2014"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q100540730_1295678400`

| Field | Value |
|---|---|
| qid | Q100540730 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100540730::P8726 |
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
| truth_tokens_preview | ["2015/si/197/made"] |
| classification_target_tokens | ["2015/si/197/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2015/si/197/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2015/si/197/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2015/si/197/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295678400,
  "value": [
    "2015/si/197/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2015/si/197/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2015/si/197/made": 1
    },
    "new_unique": [
      "2015/si/197/made"
    ],
    "new_values": [
      "2015/si/197/made"
    ],
    "new_values_raw": [
      "2015/si/197/made"
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
      "2015/si/197/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2015/si/197/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2015/si/197/made",
      "extracted_number": "197",
      "extracted_year": "2015",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 197/2015",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 197/2015"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2015/si/197/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 197/2015",
    "label": "Waste Management (Collection Permit) (Amendment) Regulations 2015"
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
      "added_unique_values": [
        "2015/si/197/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2015/si/197/made": 1
      },
      "new_unique": [
        "2015/si/197/made"
      ],
      "new_values": [
        "2015/si/197/made"
      ],
      "new_values_raw": [
        "2015/si/197/made"
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
        "2015/si/197/made": {
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
    "detail": {
      "actual_target": "2015/si/197/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2015/si/197/made",
      "extracted_number": "197",
      "extracted_year": "2015",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 197/2015",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 197/2015"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2015/si/197/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2015/si/197/made",
          "extracted_number": "197",
          "extracted_year": "2015",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 197/2015",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 197/2015"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 011. `repair_Q100540812_1295678864`

| Field | Value |
|---|---|
| qid | Q100540812 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100540812::P8726 |
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
| truth_tokens_preview | ["2008/si/405/made"] |
| classification_target_tokens | ["2008/si/405/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2008/si/405/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2008/si/405/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2008/si/405/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295678864,
  "value": [
    "2008/si/405/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2008/si/405/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2008/si/405/made": 1
    },
    "new_unique": [
      "2008/si/405/made"
    ],
    "new_values": [
      "2008/si/405/made"
    ],
    "new_values_raw": [
      "2008/si/405/made"
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
      "2008/si/405/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2008/si/405/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2008/si/405/made",
      "extracted_number": "405",
      "extracted_year": "2008",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 405/2008",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 405/2008"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2008/si/405/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 405/2008",
    "label": "Public Health (tobacco) (Amendment) Act 2004 (Commencement) Order 2008"
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
      "added_unique_values": [
        "2008/si/405/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2008/si/405/made": 1
      },
      "new_unique": [
        "2008/si/405/made"
      ],
      "new_values": [
        "2008/si/405/made"
      ],
      "new_values_raw": [
        "2008/si/405/made"
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
        "2008/si/405/made": {
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
    "detail": {
      "actual_target": "2008/si/405/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2008/si/405/made",
      "extracted_number": "405",
      "extracted_year": "2008",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 405/2008",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 405/2008"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2008/si/405/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2008/si/405/made",
          "extracted_number": "405",
          "extracted_year": "2008",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 405/2008",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 405/2008"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 012. `repair_Q100540814_1295678865`

| Field | Value |
|---|---|
| qid | Q100540814 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100540814::P8726 |
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
| truth_tokens_preview | ["2015/si/224/made"] |
| classification_target_tokens | ["2015/si/224/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2015/si/224/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2015/si/224/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2015/si/224/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295678865,
  "value": [
    "2015/si/224/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2015/si/224/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2015/si/224/made": 1
    },
    "new_unique": [
      "2015/si/224/made"
    ],
    "new_values": [
      "2015/si/224/made"
    ],
    "new_values_raw": [
      "2015/si/224/made"
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
      "2015/si/224/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2015/si/224/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2015/si/224/made",
      "extracted_number": "224",
      "extracted_year": "2015",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 224/2015",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 224/2015"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2015/si/224/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 224/2015",
    "label": "European Communities (Accounts) (Amendment) Regulations 2015"
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
      "added_unique_values": [
        "2015/si/224/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2015/si/224/made": 1
      },
      "new_unique": [
        "2015/si/224/made"
      ],
      "new_values": [
        "2015/si/224/made"
      ],
      "new_values_raw": [
        "2015/si/224/made"
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
        "2015/si/224/made": {
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
    "detail": {
      "actual_target": "2015/si/224/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2015/si/224/made",
      "extracted_number": "224",
      "extracted_year": "2015",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 224/2015",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 224/2015"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2015/si/224/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2015/si/224/made",
          "extracted_number": "224",
          "extracted_year": "2015",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 224/2015",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 224/2015"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 013. `repair_Q100541013_1295683303`

| Field | Value |
|---|---|
| qid | Q100541013 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100541013::P8726 |
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
| truth_tokens_preview | ["2015/si/287/made"] |
| classification_target_tokens | ["2015/si/287/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2015/si/287/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2015/si/287/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2015/si/287/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295683303,
  "value": [
    "2015/si/287/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2015/si/287/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2015/si/287/made": 1
    },
    "new_unique": [
      "2015/si/287/made"
    ],
    "new_values": [
      "2015/si/287/made"
    ],
    "new_values_raw": [
      "2015/si/287/made"
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
      "2015/si/287/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2015/si/287/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2015/si/287/made",
      "extracted_number": "287",
      "extracted_year": "2015",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 287/2015",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 287/2015"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2015/si/287/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 287/2015",
    "label": "National Vehicle and Driver File (Access) (No. 2) Regulations 2015"
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
      "added_unique_values": [
        "2015/si/287/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2015/si/287/made": 1
      },
      "new_unique": [
        "2015/si/287/made"
      ],
      "new_values": [
        "2015/si/287/made"
      ],
      "new_values_raw": [
        "2015/si/287/made"
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
        "2015/si/287/made": {
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
    "detail": {
      "actual_target": "2015/si/287/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2015/si/287/made",
      "extracted_number": "287",
      "extracted_year": "2015",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 287/2015",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 287/2015"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2015/si/287/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2015/si/287/made",
          "extracted_number": "287",
          "extracted_year": "2015",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 287/2015",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 287/2015"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 014. `repair_Q100542539_1295674509`

| Field | Value |
|---|---|
| qid | Q100542539 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100542539::P8726 |
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
| truth_tokens_preview | ["2016/si/152/made"] |
| classification_target_tokens | ["2016/si/152/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2016/si/152/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2016/si/152/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2016/si/152/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295674509,
  "value": [
    "2016/si/152/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2016/si/152/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2016/si/152/made": 1
    },
    "new_unique": [
      "2016/si/152/made"
    ],
    "new_values": [
      "2016/si/152/made"
    ],
    "new_values_raw": [
      "2016/si/152/made"
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
      "2016/si/152/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2016/si/152/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2016/si/152/made",
      "extracted_number": "152",
      "extracted_year": "2016",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 152/2016",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 152/2016"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2016/si/152/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 152/2016",
    "label": "Occupational Pension Schemes (Revaluation) Regulations 2016"
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
      "added_unique_values": [
        "2016/si/152/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2016/si/152/made": 1
      },
      "new_unique": [
        "2016/si/152/made"
      ],
      "new_values": [
        "2016/si/152/made"
      ],
      "new_values_raw": [
        "2016/si/152/made"
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
        "2016/si/152/made": {
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
    "detail": {
      "actual_target": "2016/si/152/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2016/si/152/made",
      "extracted_number": "152",
      "extracted_year": "2016",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 152/2016",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 152/2016"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2016/si/152/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2016/si/152/made",
          "extracted_number": "152",
          "extracted_year": "2016",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 152/2016",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 152/2016"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 015. `repair_Q100542634_1295676108`

| Field | Value |
|---|---|
| qid | Q100542634 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100542634::P8726 |
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
| truth_tokens_preview | ["2016/si/186/made"] |
| classification_target_tokens | ["2016/si/186/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2016/si/186/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2016/si/186/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2016/si/186/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295676108,
  "value": [
    "2016/si/186/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2016/si/186/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2016/si/186/made": 1
    },
    "new_unique": [
      "2016/si/186/made"
    ],
    "new_values": [
      "2016/si/186/made"
    ],
    "new_values_raw": [
      "2016/si/186/made"
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
      "2016/si/186/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2016/si/186/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2016/si/186/made",
      "extracted_number": "186",
      "extracted_year": "2016",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 186/2016",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 186/2016"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2016/si/186/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 186/2016",
    "label": "European Union Habitats (Lisduff Fen Special Area of Conservation 002147) Regulations 2016"
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
      "added_unique_values": [
        "2016/si/186/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2016/si/186/made": 1
      },
      "new_unique": [
        "2016/si/186/made"
      ],
      "new_values": [
        "2016/si/186/made"
      ],
      "new_values_raw": [
        "2016/si/186/made"
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
        "2016/si/186/made": {
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
    "detail": {
      "actual_target": "2016/si/186/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2016/si/186/made",
      "extracted_number": "186",
      "extracted_year": "2016",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 186/2016",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 186/2016"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2016/si/186/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2016/si/186/made",
          "extracted_number": "186",
          "extracted_year": "2016",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 186/2016",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 186/2016"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 016. `repair_Q100543725_1295694734`

| Field | Value |
|---|---|
| qid | Q100543725 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100543725::P8726 |
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
| truth_tokens_preview | ["2016/si/450/made"] |
| classification_target_tokens | ["2016/si/450/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2016/si/450/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2016/si/450/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2016/si/450/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295694734,
  "value": [
    "2016/si/450/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2016/si/450/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2016/si/450/made": 1
    },
    "new_unique": [
      "2016/si/450/made"
    ],
    "new_values": [
      "2016/si/450/made"
    ],
    "new_values_raw": [
      "2016/si/450/made"
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
      "2016/si/450/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2016/si/450/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2016/si/450/made",
      "extracted_number": "450",
      "extracted_year": "2016",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 450/2016",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 450/2016"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2016/si/450/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 450/2016",
    "label": "Rules of the Superior Courts (Construction Contracts Act 2013) 2016"
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
      "added_unique_values": [
        "2016/si/450/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2016/si/450/made": 1
      },
      "new_unique": [
        "2016/si/450/made"
      ],
      "new_values": [
        "2016/si/450/made"
      ],
      "new_values_raw": [
        "2016/si/450/made"
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
        "2016/si/450/made": {
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
    "detail": {
      "actual_target": "2016/si/450/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2016/si/450/made",
      "extracted_number": "450",
      "extracted_year": "2016",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 450/2016",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 450/2016"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2016/si/450/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2016/si/450/made",
          "extracted_number": "450",
          "extracted_year": "2016",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 450/2016",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 450/2016"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 017. `repair_Q100544069_1295697798`

| Field | Value |
|---|---|
| qid | Q100544069 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100544069::P8726 |
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
| truth_tokens_preview | ["2009/si/485/made"] |
| classification_target_tokens | ["2009/si/485/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2009/si/485/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2009/si/485/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2009/si/485/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295697798,
  "value": [
    "2009/si/485/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2009/si/485/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2009/si/485/made": 1
    },
    "new_unique": [
      "2009/si/485/made"
    ],
    "new_values": [
      "2009/si/485/made"
    ],
    "new_values_raw": [
      "2009/si/485/made"
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
      "2009/si/485/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2009/si/485/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2009/si/485/made",
      "extracted_number": "485",
      "extracted_year": "2009",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 485/2009",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 485/2009"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2009/si/485/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 485/2009",
    "label": "Finance Act 2008 (Commencement of Section 111) Order 2009"
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
      "added_unique_values": [
        "2009/si/485/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2009/si/485/made": 1
      },
      "new_unique": [
        "2009/si/485/made"
      ],
      "new_values": [
        "2009/si/485/made"
      ],
      "new_values_raw": [
        "2009/si/485/made"
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
        "2009/si/485/made": {
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
    "detail": {
      "actual_target": "2009/si/485/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2009/si/485/made",
      "extracted_number": "485",
      "extracted_year": "2009",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 485/2009",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 485/2009"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2009/si/485/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2009/si/485/made",
          "extracted_number": "485",
          "extracted_year": "2009",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 485/2009",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 485/2009"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 018. `repair_Q100545698_1295696018`

| Field | Value |
|---|---|
| qid | Q100545698 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100545698::P8726 |
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
| truth_tokens_preview | ["2017/si/173/made"] |
| classification_target_tokens | ["2017/si/173/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2017/si/173/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2017/si/173/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2017/si/173/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295696018,
  "value": [
    "2017/si/173/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2017/si/173/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2017/si/173/made": 1
    },
    "new_unique": [
      "2017/si/173/made"
    ],
    "new_values": [
      "2017/si/173/made"
    ],
    "new_values_raw": [
      "2017/si/173/made"
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
      "2017/si/173/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2017/si/173/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2017/si/173/made",
      "extracted_number": "173",
      "extracted_year": "2017",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 173/2017",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 173/2017"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2017/si/173/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 173/2017",
    "label": "Misuse of Drugs Regulations 2017"
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
      "added_unique_values": [
        "2017/si/173/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2017/si/173/made": 1
      },
      "new_unique": [
        "2017/si/173/made"
      ],
      "new_values": [
        "2017/si/173/made"
      ],
      "new_values_raw": [
        "2017/si/173/made"
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
        "2017/si/173/made": {
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
    "detail": {
      "actual_target": "2017/si/173/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2017/si/173/made",
      "extracted_number": "173",
      "extracted_year": "2017",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 173/2017",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 173/2017"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2017/si/173/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2017/si/173/made",
          "extracted_number": "173",
          "extracted_year": "2017",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 173/2017",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 173/2017"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 019. `repair_Q100547272_1295689115`

| Field | Value |
|---|---|
| qid | Q100547272 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100547272::P8726 |
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
| truth_tokens_preview | ["2017/si/429/made"] |
| classification_target_tokens | ["2017/si/429/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2017/si/429/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2017/si/429/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2017/si/429/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295689115,
  "value": [
    "2017/si/429/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2017/si/429/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2017/si/429/made": 1
    },
    "new_unique": [
      "2017/si/429/made"
    ],
    "new_values": [
      "2017/si/429/made"
    ],
    "new_values_raw": [
      "2017/si/429/made"
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
      "2017/si/429/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2017/si/429/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2017/si/429/made",
      "extracted_number": "429",
      "extracted_year": "2017",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 429/2017",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 429/2017"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2017/si/429/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 429/2017",
    "label": "Health Act 2007 (Commencement) Order 2017"
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
      "added_unique_values": [
        "2017/si/429/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2017/si/429/made": 1
      },
      "new_unique": [
        "2017/si/429/made"
      ],
      "new_values": [
        "2017/si/429/made"
      ],
      "new_values_raw": [
        "2017/si/429/made"
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
        "2017/si/429/made": {
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
    "detail": {
      "actual_target": "2017/si/429/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2017/si/429/made",
      "extracted_number": "429",
      "extracted_year": "2017",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 429/2017",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 429/2017"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2017/si/429/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2017/si/429/made",
          "extracted_number": "429",
          "extracted_year": "2017",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 429/2017",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 429/2017"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 020. `repair_Q100547370_1295689210`

| Field | Value |
|---|---|
| qid | Q100547370 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100547370::P8726 |
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
| truth_tokens_preview | ["2017/si/446/made"] |
| classification_target_tokens | ["2017/si/446/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2017/si/446/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2017/si/446/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2017/si/446/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295689210,
  "value": [
    "2017/si/446/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2017/si/446/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2017/si/446/made": 1
    },
    "new_unique": [
      "2017/si/446/made"
    ],
    "new_values": [
      "2017/si/446/made"
    ],
    "new_values_raw": [
      "2017/si/446/made"
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
      "2017/si/446/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2017/si/446/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2017/si/446/made",
      "extracted_number": "446",
      "extracted_year": "2017",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 446/2017",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 446/2017"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2017/si/446/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 446/2017",
    "label": "European Union Habitats (Lough Cutra Special Area of Conservation 000299) Regulations 2017"
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
      "added_unique_values": [
        "2017/si/446/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2017/si/446/made": 1
      },
      "new_unique": [
        "2017/si/446/made"
      ],
      "new_values": [
        "2017/si/446/made"
      ],
      "new_values_raw": [
        "2017/si/446/made"
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
        "2017/si/446/made": {
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
    "detail": {
      "actual_target": "2017/si/446/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2017/si/446/made",
      "extracted_number": "446",
      "extracted_year": "2017",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 446/2017",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 446/2017"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2017/si/446/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2017/si/446/made",
          "extracted_number": "446",
          "extracted_year": "2017",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 446/2017",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 446/2017"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 021. `repair_Q100547668_1295692923`

| Field | Value |
|---|---|
| qid | Q100547668 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100547668::P8726 |
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
| truth_tokens_preview | ["2017/si/498/made"] |
| classification_target_tokens | ["2017/si/498/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2017/si/498/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2017/si/498/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2017/si/498/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295692923,
  "value": [
    "2017/si/498/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2017/si/498/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2017/si/498/made": 1
    },
    "new_unique": [
      "2017/si/498/made"
    ],
    "new_values": [
      "2017/si/498/made"
    ],
    "new_values_raw": [
      "2017/si/498/made"
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
      "2017/si/498/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2017/si/498/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2017/si/498/made",
      "extracted_number": "498",
      "extracted_year": "2017",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 498/2017",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 498/2017"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2017/si/498/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 498/2017",
    "label": "Forestry (Amendment) Regulations 2017"
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
      "added_unique_values": [
        "2017/si/498/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2017/si/498/made": 1
      },
      "new_unique": [
        "2017/si/498/made"
      ],
      "new_values": [
        "2017/si/498/made"
      ],
      "new_values_raw": [
        "2017/si/498/made"
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
        "2017/si/498/made": {
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
    "detail": {
      "actual_target": "2017/si/498/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2017/si/498/made",
      "extracted_number": "498",
      "extracted_year": "2017",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 498/2017",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 498/2017"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2017/si/498/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2017/si/498/made",
          "extracted_number": "498",
          "extracted_year": "2017",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 498/2017",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 498/2017"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---

## 022. `repair_Q100547947_1295696805`

| Field | Value |
|---|---|
| qid | Q100547947 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeB / LOCAL_TEXT_DERIVED / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_g_local_text_derived |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | local_evidence |
| classification_rule_subfamily | local_text_derived |
| decision_constraint_type |   |
| group_key | ABOX::Q100547947::P8726 |
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
| truth_tokens_preview | ["2017/si/546/made"] |
| classification_target_tokens | ["2017/si/546/made"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | local_text_derived |
| rationale | Target literal derived from independent local text by deterministic property-specific transformation. |
| local_match_kind |  |
| local_match_source | FOCUS_DESCRIPTION |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2017/si/546/made"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "2017/si/546/made"
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
  "classification_rule_subfamily": "local_text_derived",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2017/si/546/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295696805,
  "value": [
    "2017/si/546/made"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "2017/si/546/made"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2017/si/546/made": 1
    },
    "new_unique": [
      "2017/si/546/made"
    ],
    "new_values": [
      "2017/si/546/made"
    ],
    "new_values_raw": [
      "2017/si/546/made"
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
      "2017/si/546/made": {
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
  "report_fix_date": "2020-10-26T07:19:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
  "report_revision_new": 1297675684,
  "report_revision_old": 1295775049,
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
  "found": null,
  "local_availability_result": true,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
  "matched": null,
  "matches": [
    {
      "actual_target": "2017/si/546/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2017/si/546/made",
      "extracted_number": "546",
      "extracted_year": "2017",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 546/2017",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 546/2017"
    }
  ],
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "2017/si/546/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of legislation on the irishstatutebook.ie website",
    "label": "Irish Statute Book ID"
  },
  "qid": {
    "description": "Irish Statutory Instrument S.I. No. 546/2017",
    "label": "Social Welfare Act 2016 (Section 4) (Commencement) Order 2017"
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
      "added_unique_values": [
        "2017/si/546/made"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2017/si/546/made": 1
      },
      "new_unique": [
        "2017/si/546/made"
      ],
      "new_values": [
        "2017/si/546/made"
      ],
      "new_values_raw": [
        "2017/si/546/made"
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
        "2017/si/546/made": {
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
    "detail": {
      "actual_target": "2017/si/546/made",
      "classification_target_role": "added",
      "derivation_rule": "p8726_statutory_instrument_id",
      "derived_target": "2017/si/546/made",
      "extracted_number": "546",
      "extracted_year": "2017",
      "independent_of_target_property": true,
      "raw_matched_text": "S.I. No. 546/2017",
      "source": "FOCUS_DESCRIPTION",
      "source_text": "Irish Statutory Instrument S.I. No. 546/2017"
    },
    "independent_of_target_property": true,
    "result": true,
    "step": "local_text_derived"
  },
  {
    "evidence": {
      "matches": [
        {
          "actual_target": "2017/si/546/made",
          "classification_target_role": "added",
          "derivation_rule": "p8726_statutory_instrument_id",
          "derived_target": "2017/si/546/made",
          "extracted_number": "546",
          "extracted_year": "2017",
          "independent_of_target_property": true,
          "raw_matched_text": "S.I. No. 546/2017",
          "source": "FOCUS_DESCRIPTION",
          "source_text": "Irish Statutory Instrument S.I. No. 546/2017"
        }
      ]
    },
    "result": true,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "local_text_derived",
    "step": "branch"
  }
]
```

---
