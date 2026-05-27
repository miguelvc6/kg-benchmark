# TypeC_EXTERNAL_BY_ELIMINATION_LITERAL_TRUTH

Cases: 50

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q100537036_1295668682`

| Field | Value |
|---|---|
| qid | Q100537036 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100537036::P8726 |
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
| truth_tokens_preview | ["2007/si/483/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2007/si/483/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295668682,
  "value": [
    "2007/si/483/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2007/si/483/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2007/si/483/made"
    ],
    "new_value": [
      "2007/si/483/made"
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
      "2007/si/483/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
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
    "2007/si/483/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 483/2007",
    "label": "Taxes Consolidation Act 1997 (Qualifying Urban Renewal Areas) (New Ross, County Wexford) (Amendment) Order 2007"
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

## 002. `repair_Q100537126_1295671742`

| Field | Value |
|---|---|
| qid | Q100537126 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100537126::P8726 |
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
| truth_tokens_preview | ["2007/si/538/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2007/si/538/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295671742,
  "value": [
    "2007/si/538/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2007/si/538/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2007/si/538/made"
    ],
    "new_value": [
      "2007/si/538/made"
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
      "2007/si/538/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 9,
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
    "2007/si/538/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 538/2007",
    "label": "Medicinal Products (Control of Wholesale Distribution) Regulations 2007"
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

## 003. `repair_Q100537290_1295674850`

| Field | Value |
|---|---|
| qid | Q100537290 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100537290::P8726 |
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
| truth_tokens_preview | ["2007/si/598/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2007/si/598/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295674850,
  "value": [
    "2007/si/598/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2007/si/598/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2007/si/598/made"
    ],
    "new_value": [
      "2007/si/598/made"
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
      "2007/si/598/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
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
    "2007/si/598/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 598/2007",
    "label": "European Communities (Human Tissues and Cells Traceability Requirements, Notification of Serious Adverse Reactions and Events and Certain Technical Requirements) Regulations 2007"
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

## 004. `repair_Q100537373_1295674950`

| Field | Value |
|---|---|
| qid | Q100537373 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100537373::P8726 |
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
| truth_tokens_preview | ["2007/si/623/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2007/si/623/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295674950,
  "value": [
    "2007/si/623/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2007/si/623/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2007/si/623/made"
    ],
    "new_value": [
      "2007/si/623/made"
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
      "2007/si/623/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 9,
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
    "2007/si/623/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 623/2007",
    "label": "Contract Cleaning (Excluding the City and County of Dublin) Joint Labour Committee (Abolition) Order, 2007"
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

## 005. `repair_Q100537767_1295681249`

| Field | Value |
|---|---|
| qid | Q100537767 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100537767::P8726 |
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
| truth_tokens_preview | ["2013/si/572/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2013/si/572/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295681249,
  "value": [
    "2013/si/572/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2013/si/572/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2013/si/572/made"
    ],
    "new_value": [
      "2013/si/572/made"
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
      "2013/si/572/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
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
    "2013/si/572/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 572/2013",
    "label": "Safety, Health and Welfare at Work (Biological Agents) Regulations 2013"
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

## 006. `repair_Q100539216_1295673234`

| Field | Value |
|---|---|
| qid | Q100539216 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100539216::P8726 |
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
| truth_tokens_preview | ["2014/si/398/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2014/si/398/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295673234,
  "value": [
    "2014/si/398/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2014/si/398/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2014/si/398/made"
    ],
    "new_value": [
      "2014/si/398/made"
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
      "2014/si/398/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
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
    "2014/si/398/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 398/2014",
    "label": "European Union (Paints, Varnishes, Vehicle Refinishing Products and Activities) (Amendment) Regulations 2014"
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

## 007. `repair_Q100540595_1295672660`

| Field | Value |
|---|---|
| qid | Q100540595 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100540595::P8726 |
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
| truth_tokens_preview | ["2008/si/364/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2008/si/364/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295672660,
  "value": [
    "2008/si/364/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2008/si/364/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2008/si/364/made"
    ],
    "new_value": [
      "2008/si/364/made"
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
      "2008/si/364/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
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
    "2008/si/364/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 364/2008",
    "label": "Hepatitis C Compensation Tribunal (Insurance Scheme for Relevant Claimants) (Amendment) Regulations 2008"
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

## 008. `repair_Q100540631_1295672719`

| Field | Value |
|---|---|
| qid | Q100540631 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100540631::P8726 |
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
| truth_tokens_preview | ["2015/si/165/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2015/si/165/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295672719,
  "value": [
    "2015/si/165/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2015/si/165/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2015/si/165/made"
    ],
    "new_value": [
      "2015/si/165/made"
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
      "2015/si/165/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
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
    "2015/si/165/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 165/2015",
    "label": "Statistics (Community Innovation Survey) Order 2015"
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

## 009. `repair_Q100541522_1295693853`

| Field | Value |
|---|---|
| qid | Q100541522 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100541522::P8726 |
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
| truth_tokens_preview | ["2015/si/443/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2015/si/443/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295693853,
  "value": [
    "2015/si/443/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2015/si/443/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2015/si/443/made"
    ],
    "new_value": [
      "2015/si/443/made"
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
      "2015/si/443/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 13,
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
    "2015/si/443/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 443/2015",
    "label": "Health (Miscellaneous Provisions) Act 2014 (Commencement) Order 2015"
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

## 010. `repair_Q100542941_1295680747`

| Field | Value |
|---|---|
| qid | Q100542941 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100542941::P8726 |
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
| truth_tokens_preview | ["2016/si/256/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2016/si/256/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295680747,
  "value": [
    "2016/si/256/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2016/si/256/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2016/si/256/made"
    ],
    "new_value": [
      "2016/si/256/made"
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
      "2016/si/256/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
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
    "2016/si/256/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 256/2016",
    "label": "European Union Habitats (Ballintra Special Area of Conservation 000115) Regulations 2016"
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

## 011. `repair_Q100543000_1295683720`

| Field | Value |
|---|---|
| qid | Q100543000 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100543000::P8726 |
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
| truth_tokens_preview | ["2016/si/267/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2016/si/267/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295683720,
  "value": [
    "2016/si/267/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2016/si/267/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2016/si/267/made"
    ],
    "new_value": [
      "2016/si/267/made"
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
      "2016/si/267/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
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
    "2016/si/267/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 267/2016",
    "label": "European Union Habitats (Lisduff Turlough Special Area of Conservation 000609) Regulations 2016"
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

## 012. `repair_Q100545012_1295686192`

| Field | Value |
|---|---|
| qid | Q100545012 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100545012::P8726 |
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
| truth_tokens_preview | ["2010/si/152/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2010/si/152/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295686192,
  "value": [
    "2010/si/152/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2010/si/152/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2010/si/152/made"
    ],
    "new_value": [
      "2010/si/152/made"
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
      "2010/si/152/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
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
    "2010/si/152/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 152/2010",
    "label": "European Communities (Marketing of Fruit Plant Propagating Material) Regulations 2010"
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

## 013. `repair_Q100546769_1295708986`

| Field | Value |
|---|---|
| qid | Q100546769 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100546769::P8726 |
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
| truth_tokens_preview | ["2010/si/522/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2010/si/522/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295708986,
  "value": [
    "2010/si/522/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2010/si/522/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2010/si/522/made"
    ],
    "new_value": [
      "2010/si/522/made"
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
      "2010/si/522/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
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
    "2010/si/522/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 522/2010",
    "label": "European Communities (Additives, Colours and Sweeteners in Foodstuffs) (Amendment) Regulations 2010"
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

## 014. `repair_Q100548202_1295700384`

| Field | Value |
|---|---|
| qid | Q100548202 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100548202::P8726 |
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
| truth_tokens_preview | ["2011/si/105/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2011/si/105/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295700384,
  "value": [
    "2011/si/105/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2011/si/105/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2011/si/105/made"
    ],
    "new_value": [
      "2011/si/105/made"
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
      "2011/si/105/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
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
    "2011/si/105/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 105/2011",
    "label": "European Communities (Plastics and other materials) (Contact with Foodstuffs) (Amendment) Regulations 2011"
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

## 015. `repair_Q100548208_1295700391`

| Field | Value |
|---|---|
| qid | Q100548208 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q100548208::P8726 |
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
| truth_tokens_preview | ["2011/si/106/made"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2011/si/106/made"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 1295700391,
  "value": [
    "2011/si/106/made"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2011/si/106/made"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2011/si/106/made"
    ],
    "new_value": [
      "2011/si/106/made"
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
      "2011/si/106/made": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
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
    "2011/si/106/made"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Irish Statutory Instrument S.I. No. 106/2011",
    "label": "Agriculture Appeals Act 2001 (Amendment of Schedule) Regulations 2011"
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

## 016. `repair_Q11372574_2309440932`

| Field | Value |
|---|---|
| qid | Q11372574 |
| property | P5838 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q11372574::P5838 |
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
| truth_tokens_preview | ["FADJ01"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "NoInkling",
  "kind": "A_BOX",
  "new_value": [
    "FADJ01"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2309440932,
  "value": [
    "FADJ01"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "FADJ01"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "FADJ01"
    ],
    "new_value": [
      "FADJ01"
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
      "FADJ01": {
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
  "report_fix_date": "2025-02-15T07:41:53",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5838",
  "report_revision_new": 2311723062,
  "report_revision_old": 2311277564,
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
  "local_ids_count": 17,
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
    "FADJ01"
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
    "description": "Japanese 1983 puzzle video game",
    "label": "Gomoku Narabe Renju"
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

## 017. `repair_Q122922920_2440236730`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "SOMEVALUE"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "SOMEVALUE"
    ],
    "new_value": [
      "SOMEVALUE"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q4233718"
    ],
    "old_value": [
      "Q4233718"
    ],
    "removed_unique_values": [
      "Q4233718"
    ],
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

## 018. `repair_Q1409450_2406214621`

| Field | Value |
|---|---|
| qid | Q1409450 |
| property | P1006 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| group_key | ABOX::Q1409450::P1006 |
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
| truth_tokens_preview | ["405347847", "413636712"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Susmuffin",
  "kind": "A_BOX",
  "new_value": [
    "405347847",
    "413636712"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2406214621,
  "value": [
    "405347847",
    "413636712"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "405347847",
      "413636712"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "405347847",
      "413636712"
    ],
    "new_value": [
      "405347847",
      "413636712"
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
      "405347847": {
        "new": 1,
        "old": 0
      },
      "413636712": {
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
  "report_fix_date": "2025-09-19T22:59:13",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1006",
  "report_revision_new": 2407507647,
  "report_revision_old": 2407109254,
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 35,
  "matched": false,
  "matches": [],
  "needed": 2,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "405347847",
    "413636712"
  ],
  "truth_tokens_in_recorded_matches": [],
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
    "description": "German economist",
    "label": "Marc Oliver Opresnik"
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
      "needed": 2,
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

## 019. `repair_Q16176248_2440335985`

| Field | Value |
|---|---|
| qid | Q16176248 |
| property | P569 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| group_key | ABOX::Q16176248::P569 |
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
| truth_tokens_preview | ["+1977-10-04T00:00:00Z"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Bovlb",
  "kind": "A_BOX",
  "new_value": [
    "+1977-10-04T00:00:00Z"
  ],
  "old_value": [
    "+1986-00-00T00:00:00Z"
  ],
  "revision_id": 2440335985,
  "value": [
    "+1977-10-04T00:00:00Z"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "+1977-10-04T00:00:00Z"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "+1977-10-04T00:00:00Z"
    ],
    "new_value": [
      "+1977-10-04T00:00:00Z"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "+1986-00-00T00:00:00Z"
    ],
    "old_value": [
      "+1986-00-00T00:00:00Z"
    ],
    "removed_unique_values": [
      "+1986-00-00T00:00:00Z"
    ],
    "value_multiplicity_changes": {
      "+1977-10-04T00:00:00Z": {
        "new": 1,
        "old": 0
      },
      "+1986-00-00T00:00:00Z": {
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
  "report_fix_date": "2025-12-11T11:58:25",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P569",
  "report_revision_new": 2440836223,
  "report_revision_old": 2440403690,
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
    "+1986-00-00T00:00:00Z"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 65,
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "+1986-00-00T00:00:00Z"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "+1977-10-04T00:00:00Z"
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
    "description": "Catalan writer and translator",
    "label": "Bel Olid"
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
      "local_ids_count": 65,
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
        "+1986-00-00T00:00:00Z"
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

## 020. `repair_Q2121419_2311240375`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "PC9J18"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "PC9J18"
    ],
    "new_value": [
      "PC9J18"
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

## 021. `repair_Q2332808_2309914266`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "FFJJ01"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "FFJJ01"
    ],
    "new_value": [
      "FFJJ01"
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
      "FFJJ01": {
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

## 022. `repair_Q2364078_2310352975`

| Field | Value |
|---|---|
| qid | Q2364078 |
| property | P5838 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q2364078::P5838 |
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
| truth_tokens_preview | ["ECNEJ8", "ECNJJ8", "ECNPJ8"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "NoInkling",
  "kind": "A_BOX",
  "new_value": [
    "ECNEJ8",
    "ECNJJ8",
    "ECNPJ8"
  ],
  "old_value": [
    "ECNEJ8",
    "ECNJJ8"
  ],
  "revision_id": 2310352975,
  "value": [
    "ECNEJ8",
    "ECNJJ8",
    "ECNPJ8"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "ECNPJ8"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "ECNEJ8",
      "ECNJJ8",
      "ECNPJ8"
    ],
    "new_value": [
      "ECNEJ8",
      "ECNJJ8",
      "ECNPJ8"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "ECNEJ8",
      "ECNJJ8"
    ],
    "old_value": [
      "ECNEJ8",
      "ECNJJ8"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "ECNPJ8": {
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
    "ECNEJ8",
    "ECNJJ8"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 33,
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "ECNEJ8",
      "ECNJJ8"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "ECNEJ8",
    "ECNJJ8",
    "ECNPJ8"
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
    "description": "2001 video game",
    "label": "Sengoku 3"
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
        "ECNEJ8",
        "ECNJJ8"
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

## 023. `repair_Q27107545_2425741998`

| Field | Value |
|---|---|
| qid | Q27107545 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q27107545::P2877 |
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
| truth_tokens_preview | ["21088422", "13592134"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "21088422",
    "13592134"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2425741998,
  "value": [
    "21088422",
    "13592134"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "13592134",
      "21088422"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "13592134",
      "21088422"
    ],
    "new_value": [
      "21088422",
      "13592134"
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
      "13592134": {
        "new": 1,
        "old": 0
      },
      "21088422": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 17,
  "matched": false,
  "matches": [],
  "needed": 2,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "21088422",
    "13592134"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "Norhyoscyamine"
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
      "needed": 2,
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

## 024. `repair_Q27163854_2425080233`

| Field | Value |
|---|---|
| qid | Q27163854 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q27163854::P2877 |
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
| truth_tokens_preview | ["31060198", "9904268"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "31060198",
    "9904268"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2425080233,
  "value": [
    "31060198",
    "9904268"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "31060198",
      "9904268"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "31060198",
      "9904268"
    ],
    "new_value": [
      "31060198",
      "9904268"
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
      "31060198": {
        "new": 1,
        "old": 0
      },
      "9904268": {
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
  "report_fix_date": "2025-11-06T09:25:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426458092,
  "report_revision_old": 2425837272,
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [],
  "needed": 2,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "31060198",
    "9904268"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "7-hydroxy-2,3,4,5-tetrahydrobenzofuro[2,3-c]azepin-1-one"
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
      "needed": 2,
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

## 025. `repair_Q27186241_2425162245`

| Field | Value |
|---|---|
| qid | Q27186241 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q27186241::P2877 |
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
| truth_tokens_preview | ["29831923", "3460177"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "29831923",
    "3460177"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2425162245,
  "value": [
    "29831923",
    "3460177"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "29831923",
      "3460177"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "29831923",
      "3460177"
    ],
    "new_value": [
      "29831923",
      "3460177"
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
      "29831923": {
        "new": 1,
        "old": 0
      },
      "3460177": {
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
  "report_fix_date": "2025-11-06T09:25:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426458092,
  "report_revision_old": 2425837272,
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [],
  "needed": 2,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "29831923",
    "3460177"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "2-pyridin-4-yl-1H-quinazolin-4-one"
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
      "needed": 2,
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

## 026. `repair_Q27236384_2425108945`

| Field | Value |
|---|---|
| qid | Q27236384 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q27236384::P2877 |
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
| truth_tokens_preview | ["7638175", "379310", "145459"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "7638175",
    "379310",
    "145459"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2425108945,
  "value": [
    "7638175",
    "379310",
    "145459"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "145459",
      "379310",
      "7638175"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "145459",
      "379310",
      "7638175"
    ],
    "new_value": [
      "7638175",
      "379310",
      "145459"
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
      "145459": {
        "new": 1,
        "old": 0
      },
      "379310": {
        "new": 1,
        "old": 0
      },
      "7638175": {
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
  "report_fix_date": "2025-11-06T09:25:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426458092,
  "report_revision_old": 2425837272,
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [],
  "needed": 3,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "7638175",
    "379310",
    "145459"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "urea sulfate"
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
      "needed": 3,
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

## 027. `repair_Q27251608_2425062294`

| Field | Value |
|---|---|
| qid | Q27251608 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q27251608::P2877 |
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
| truth_tokens_preview | ["29620269", "1329944"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "29620269",
    "1329944"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2425062294,
  "value": [
    "29620269",
    "1329944"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "1329944",
      "29620269"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "1329944",
      "29620269"
    ],
    "new_value": [
      "29620269",
      "1329944"
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
      "1329944": {
        "new": 1,
        "old": 0
      },
      "29620269": {
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
  "report_fix_date": "2025-11-06T09:25:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426458092,
  "report_revision_old": 2425837272,
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [],
  "needed": 2,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "29620269",
    "1329944"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "ethanolamine O-dodecylbenzenesulfonate"
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
      "needed": 2,
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

## 028. `repair_Q27261912_2425248992`

| Field | Value |
|---|---|
| qid | Q27261912 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q27261912::P2877 |
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
| truth_tokens_preview | ["30946630", "8672200"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "30946630",
    "8672200"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2425248992,
  "value": [
    "30946630",
    "8672200"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "30946630",
      "8672200"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "30946630",
      "8672200"
    ],
    "new_value": [
      "30946630",
      "8672200"
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
      "30946630": {
        "new": 1,
        "old": 0
      },
      "8672200": {
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
  "report_fix_date": "2025-11-06T09:25:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426458092,
  "report_revision_old": 2425837272,
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [],
  "needed": 2,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "30946630",
    "8672200"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "sodium 3-dodecylbenzenesulfonate"
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
      "needed": 2,
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

## 029. `repair_Q27265292_2425214491`

| Field | Value |
|---|---|
| qid | Q27265292 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q27265292::P2877 |
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
| truth_tokens_preview | ["29947935", "29948010", "21065732"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "29947935",
    "29948010",
    "21065732"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2425214491,
  "value": [
    "29947935",
    "29948010",
    "21065732"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "21065732",
      "29947935",
      "29948010"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "21065732",
      "29947935",
      "29948010"
    ],
    "new_value": [
      "29947935",
      "29948010",
      "21065732"
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
      "21065732": {
        "new": 1,
        "old": 0
      },
      "29947935": {
        "new": 1,
        "old": 0
      },
      "29948010": {
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
  "report_fix_date": "2025-11-06T09:25:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2426458092,
  "report_revision_old": 2425837272,
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [],
  "needed": 3,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "29947935",
    "29948010",
    "21065732"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "3-(4-hydroxyphenyl)pyrazin-2-ol"
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
      "needed": 3,
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

## 030. `repair_Q27265860_2424906008`

| Field | Value |
|---|---|
| qid | Q27265860 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q27265860::P2877 |
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
| truth_tokens_preview | ["126107", "3987020"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "126107",
    "3987020"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2424906008,
  "value": [
    "126107",
    "3987020"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "126107",
      "3987020"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "126107",
      "3987020"
    ],
    "new_value": [
      "126107",
      "3987020"
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
      "126107": {
        "new": 1,
        "old": 0
      },
      "3987020": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [],
  "needed": 2,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "126107",
    "3987020"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "ethylamine hydrochloride"
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
      "needed": 2,
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

## 031. `repair_Q27280905_2424811351`

| Field | Value |
|---|---|
| qid | Q27280905 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q27280905::P2877 |
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
| truth_tokens_preview | ["3683272", "29617021"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "3683272",
    "29617021"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2424811351,
  "value": [
    "3683272",
    "29617021"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "29617021",
      "3683272"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "29617021",
      "3683272"
    ],
    "new_value": [
      "3683272",
      "29617021"
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
      "29617021": {
        "new": 1,
        "old": 0
      },
      "3683272": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [],
  "needed": 2,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "3683272",
    "29617021"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "diethyl diphenate"
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
      "needed": 2,
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

## 032. `repair_Q27282475_2424591149`

| Field | Value |
|---|---|
| qid | Q27282475 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| group_key | ABOX::Q27282475::P2877 |
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
| truth_tokens_preview | ["36504", "36507", "25841720", "25841723"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "36504",
    "36507",
    "25841720",
    "25841723"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2424591149,
  "value": [
    "36504",
    "36507",
    "25841720",
    "25841723"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "25841720",
      "25841723",
      "36504",
      "36507"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 4,
    "new_unique": [
      "25841720",
      "25841723",
      "36504",
      "36507"
    ],
    "new_value": [
      "36504",
      "36507",
      "25841720",
      "25841723"
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
      "25841720": {
        "new": 1,
        "old": 0
      },
      "25841723": {
        "new": 1,
        "old": 0
      },
      "36504": {
        "new": 1,
        "old": 0
      },
      "36507": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [],
  "needed": 4,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "36504",
    "36507",
    "25841720",
    "25841723"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "1,3,5-triazinane-2,4,6-trithione"
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
      "needed": 4,
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

## 033. `repair_Q27284218_2424863137`

| Field | Value |
|---|---|
| qid | Q27284218 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q27284218::P2877 |
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
| truth_tokens_preview | ["48542", "7193099"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "48542",
    "7193099"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2424863137,
  "value": [
    "48542",
    "7193099"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "48542",
      "7193099"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "48542",
      "7193099"
    ],
    "new_value": [
      "48542",
      "7193099"
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
      "48542": {
        "new": 1,
        "old": 0
      },
      "7193099": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [],
  "needed": 2,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "48542",
    "7193099"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "dibutyltin bis(2-ethylhexanoate)"
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
      "needed": 2,
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

## 034. `repair_Q27455803_2424813218`

| Field | Value |
|---|---|
| qid | Q27455803 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q27455803::P2877 |
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
| truth_tokens_preview | ["3793360", "2777107", "2787496", "5181913"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "3793360",
    "2777107",
    "2787496",
    "5181913"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2424813218,
  "value": [
    "3793360",
    "2777107",
    "2787496",
    "5181913"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "2777107",
      "2787496",
      "3793360",
      "5181913"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 4,
    "new_unique": [
      "2777107",
      "2787496",
      "3793360",
      "5181913"
    ],
    "new_value": [
      "3793360",
      "2777107",
      "2787496",
      "5181913"
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
      "2777107": {
        "new": 1,
        "old": 0
      },
      "2787496": {
        "new": 1,
        "old": 0
      },
      "3793360": {
        "new": 1,
        "old": 0
      },
      "5181913": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [],
  "needed": 4,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "3793360",
    "2777107",
    "2787496",
    "5181913"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "2,8-dithioxo-1,2,3,7,8,9-hexahydro-6H-purin-6-one"
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
      "needed": 4,
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

## 035. `repair_Q28400196_2447258614`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Coaches of OL Lyonnes"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Coaches of OL Lyonnes"
    ],
    "new_value": [
      "Coaches of OL Lyonnes"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Coaches of Olympique Lyonnais Féminin"
    ],
    "old_value": [
      "Coaches of Olympique Lyonnais Féminin"
    ],
    "removed_unique_values": [
      "Coaches of Olympique Lyonnais Féminin"
    ],
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

## 036. `repair_Q3067784_2310365235`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "SEPZ41"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 4,
    "new_unique": [
      "SEPE41",
      "SEPP41",
      "SEPX41",
      "SEPZ41"
    ],
    "new_value": [
      "SEPE41",
      "SEPP41",
      "SEPX41",
      "SEPZ41"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 3,
    "old_unique": [
      "SEPE41",
      "SEPP41",
      "SEPX41"
    ],
    "old_value": [
      "SEPE41",
      "SEPP41",
      "SEPX41"
    ],
    "removed_unique_values": [],
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

## 037. `repair_Q312751_2334676628`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "LNB:BD4i;=BJ"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "LNB:BD4i;=BJ"
    ],
    "new_value": [
      "LNB:BD4i;=BJ"
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
      "LNB:BD4i;=BJ": {
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

## 038. `repair_Q31854135_2442784175`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "Náměstí Míru 251 (Týn nad Vltavou)"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Náměstí Míru 251 (Týn nad Vltavou)"
    ],
    "new_value": [
      "Náměstí Míru 251 (Týn nad Vltavou)"
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

## 039. `repair_Q48942_1724863168`

| Field | Value |
|---|---|
| qid | Q48942 |
| property | P2521 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q53869507 |
| group_key | ABOX::Q48942::P2521 |
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
| truth_tokens_preview | ["ученица@ru", "Schülerin@de", "alumna@es", "alunna@it", "μαθήτρια@el", "...(+14)"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Bjankuloski06",
  "kind": "A_BOX",
  "new_value": [
    "ученица@ru",
    "Schülerin@de",
    "alumna@es",
    "alunna@it",
    "μαθήτρια@el",
    "leerlinge@nl",
    "lernantino@eo",
    "учениця@uk",
    "تلميذة@ar",
    "תלמידה@he",
    "dijakinja@sl",
    "Schülerin@lb",
    "alumna@ast",
    "aluna@pt-br",
    "aluna@pt",
    "nữ học sinh@vi",
    "alumna@ca",
    "žákyně@cs",
    "ученичка@mk"
  ],
  "old_value": [
    "ученица@ru",
    "Schülerin@de",
    "alumna@es",
    "alunna@it",
    "μαθήτρια@el",
    "leerlinge@nl",
    "lernantino@eo",
    "учениця@uk",
    "تلميذة@ar",
    "תלמידה@he",
    "dijakinja@sl",
    "Schülerin@lb",
    "alumna@ast",
    "aluna@pt-br",
    "aluna@pt",
    "nữ học sinh@vi",
    "alumna@ca",
    "žákyně@cs"
  ],
  "revision_id": 1724863168,
  "value": [
    "ученица@ru",
    "Schülerin@de",
    "alumna@es",
    "alunna@it",
    "μαθήτρια@el",
    "leerlinge@nl",
    "lernantino@eo",
    "учениця@uk",
    "تلميذة@ar",
    "תלמידה@he",
    "dijakinja@sl",
    "Schülerin@lb",
    "alumna@ast",
    "aluna@pt-br",
    "aluna@pt",
    "nữ học sinh@vi",
    "alumna@ca",
    "žákyně@cs",
    "ученичка@mk"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "ученичка@mk"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 19,
    "new_unique": [
      "Schülerin@de",
      "Schülerin@lb",
      "alumna@ast",
      "alumna@ca",
      "alumna@es",
      "aluna@pt",
      "aluna@pt-br",
      "alunna@it",
      "dijakinja@sl",
      "leerlinge@nl",
      "lernantino@eo",
      "nữ học sinh@vi",
      "žákyně@cs",
      "μαθήτρια@el",
      "ученица@ru",
      "учениця@uk",
      "ученичка@mk",
      "תלמידה@he",
      "تلميذة@ar"
    ],
    "new_value": [
      "ученица@ru",
      "Schülerin@de",
      "alumna@es",
      "alunna@it",
      "μαθήτρια@el",
      "leerlinge@nl",
      "lernantino@eo",
      "учениця@uk",
      "تلميذة@ar",
      "תלמידה@he",
      "dijakinja@sl",
      "Schülerin@lb",
      "alumna@ast",
      "aluna@pt-br",
      "aluna@pt",
      "nữ học sinh@vi",
      "alumna@ca",
      "žákyně@cs",
      "ученичка@mk"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 18,
    "old_unique": [
      "Schülerin@de",
      "Schülerin@lb",
      "alumna@ast",
      "alumna@ca",
      "alumna@es",
      "aluna@pt",
      "aluna@pt-br",
      "alunna@it",
      "dijakinja@sl",
      "leerlinge@nl",
      "lernantino@eo",
      "nữ học sinh@vi",
      "žákyně@cs",
      "μαθήτρια@el",
      "ученица@ru",
      "учениця@uk",
      "תלמידה@he",
      "تلميذة@ar"
    ],
    "old_value": [
      "ученица@ru",
      "Schülerin@de",
      "alumna@es",
      "alunna@it",
      "μαθήτρια@el",
      "leerlinge@nl",
      "lernantino@eo",
      "учениця@uk",
      "تلميذة@ar",
      "תלמידה@he",
      "dijakinja@sl",
      "Schülerin@lb",
      "alumna@ast",
      "aluna@pt-br",
      "aluna@pt",
      "nữ học sinh@vi",
      "alumna@ca",
      "žákyně@cs"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "ученичка@mk": {
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
  "report_fix_date": "2022-09-11T11:10:29",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2521",
  "report_revision_new": 1726203339,
  "report_revision_old": 1725211201,
  "report_violation_type": "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359",
  "report_violation_type_descriptions_en": [
    "occupation requiring specialized training",
    "social role with a set of powers and responsibilities within an organization",
    "label applied to a person based on an activity they participate in",
    "field of work that requires particular skills and knowledge of skilled work",
    "subclass of noble titles",
    "legal privilege given to some members in monarchial and princely societies",
    "titles given in an organization to show what duties and responsibilities a person has",
    "human relationship term; web of social relationships that form an important part of the lives of most humans in most societies; form of social connection",
    "level in a hierarchy",
    "profession in fictional stories",
    "part of a naming scheme for individuals, used in many cultures worldwide",
    "métaclasse d'ambassadeurs",
    "part of statements according to the Wikidata data model, appearing as the 2nd item in the statement triple",
    "title bestowed upon individuals or organizations as an award in recognition of their merits",
    "title to indicate the completion of a course of study or the extent of academic achievement",
    "enumeration value for a Wikidata property",
    "name of a employee's role assigned by their employer",
    "type of identity create by a type of religious belief",
    "joint arrangement of a team on its field of play and/or the standardized place of any individual player",
    "name for a resident of a locality",
    "character or part played by a performer",
    "Item with label/aliases for inverse relation of property. This helps the related items gadget - which you can enable in your wikidata preferences - to function.",
    "person who is paid to undertake a specialized set of tasks and to complete them for a fee",
    "words or grammatical forms that denote a positive affect",
    "... omitted 8 items"
  ],
  "report_violation_type_labels_en": [
    "profession",
    "position",
    "occupation",
    "craft",
    "hereditary title",
    "noble title",
    "corporate title",
    "kinship",
    "rank",
    "fictional profession",
    "family name",
    "class of ambassadors",
    "Wikidata property",
    "title of honor",
    "academic title",
    "Wikidata enumeration value",
    "job title",
    "religious identity",
    "position",
    "demonym",
    "role",
    "inverse property label item",
    "professional",
    "approbative",
    "... omitted 8 items"
  ],
  "report_violation_type_normalized": "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359",
  "report_violation_type_qids": [
    "Q28640",
    "Q4164871",
    "Q12737077",
    "Q2207288",
    "Q5737899",
    "Q355567",
    "Q11488158",
    "Q171318",
    "Q4120621",
    "Q17305127",
    "Q101352",
    "Q29918287",
    "Q18616576",
    "Q3320743",
    "Q3529618",
    "Q21874278",
    "Q828803",
    "Q4392985",
    "Q1781513",
    "Q217438",
    "Q1707847",
    "Q65932995",
    "Q702269",
    "Q4781727",
    "... omitted 8 items"
  ],
  "report_violation_type_raw": "Type Q|28640, Q|4164871, Q|12737077, Q|2207288, Q|5737899, Q|355567, Q|11488158, Q|171318, Q|4120621, Q|17305127, Q|101352, Q|29918287, Q|18616576, Q|3320743, Q|3529618, Q|21874278, Q|828803, Q|4392985, Q|1781513, Q|217438, Q|1707847, Q|65932995, Q|702269, Q|4781727, Q|545779, Q|22116852, Q|1377295, Q|231002, Q|33829, Q|15978876, Q|15978856, Q|51591359",
  "value": [
    "ученица@ru",
    "Schülerin@de",
    "alumna@es",
    "alunna@it",
    "μαθήτρια@el",
    "leerlinge@nl",
    "lernantino@eo",
    "учениця@uk",
    "تلميذة@ar",
    "תלמידה@he",
    "dijakinja@sl",
    "Schülerin@lb",
    "alumna@ast",
    "aluna@pt-br",
    "aluna@pt",
    "nữ học sinh@vi",
    "alumna@ca",
    "žákyně@cs"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 20,
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "ученица@ru",
      "Schülerin@de",
      "alumna@es",
      "alunna@it",
      "μαθήτρια@el",
      "leerlinge@nl",
      "lernantino@eo",
      "учениця@uk",
      "تلميذة@ar",
      "תלמידה@he",
      "dijakinja@sl",
      "Schülerin@lb",
      "alumna@ast",
      "aluna@pt-br",
      "aluna@pt",
      "nữ học sinh@vi",
      "alumna@ca",
      "žákyně@cs"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "ученица@ru",
    "Schülerin@de",
    "alumna@es",
    "alunna@it",
    "μαθήτρια@el",
    "leerlinge@nl",
    "lernantino@eo",
    "учениця@uk",
    "تلميذة@ar",
    "תלמידה@he",
    "dijakinja@sl",
    "Schülerin@lb",
    "alumna@ast",
    "aluna@pt-br",
    "aluna@pt",
    "nữ học sinh@vi",
    "alumna@ca",
    "žákyně@cs",
    "ученичка@mk"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "female form of name or title (for male use P3321)",
    "label": "female form of label"
  },
  "qid": {
    "description": "child studying in a school",
    "label": "schoolchild"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
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
      "local_ids_count": 20,
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
        "ученица@ru",
        "Schülerin@de",
        "alumna@es",
        "alunna@it",
        "μαθήτρια@el",
        "leerlinge@nl",
        "lernantino@eo",
        "учениця@uk",
        "تلميذة@ar",
        "תלמידה@he",
        "dijakinja@sl",
        "Schülerin@lb",
        "alumna@ast",
        "aluna@pt-br",
        "aluna@pt",
        "nữ học sinh@vi",
        "alumna@ca",
        "žákyně@cs"
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

## 040. `repair_Q50414229_703355254`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "1205730"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "1205730"
    ],
    "new_value": [
      "1205730"
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
      "1205730": {
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

## 041. `repair_Q540943_2398875501`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Rage Against the Machine cover.jpg"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Rage Against the Machine cover.jpg"
    ],
    "new_value": [
      "Rage Against the Machine cover.jpg"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "RageAgainsttheMachineRageAgainsttheMachine.jpg"
    ],
    "old_value": [
      "RageAgainsttheMachineRageAgainsttheMachine.jpg"
    ],
    "removed_unique_values": [
      "RageAgainsttheMachineRageAgainsttheMachine.jpg"
    ],
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

## 042. `repair_Q5591950_2309445074`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "WGDJA4"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "WGDEA4",
      "WGDJA4",
      "WGDPA4"
    ],
    "new_value": [
      "WGDEA4",
      "WGDPA4",
      "WGDJA4"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "WGDEA4",
      "WGDPA4"
    ],
    "old_value": [
      "WGDEA4",
      "WGDPA4"
    ],
    "removed_unique_values": [],
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

## 043. `repair_Q6663140_2444164235`

| Field | Value |
|---|---|
| qid | Q6663140 |
| property | P214 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| group_key | ABOX::Q6663140::P214 |
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
| truth_tokens_preview | ["121821044", "1613166414901902740000"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Silewe",
  "kind": "A_BOX",
  "new_value": [
    "121821044",
    "1613166414901902740000"
  ],
  "old_value": [
    "121821044"
  ],
  "revision_id": 2444164235,
  "value": [
    "121821044",
    "1613166414901902740000"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "1613166414901902740000"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "121821044",
      "1613166414901902740000"
    ],
    "new_value": [
      "121821044",
      "1613166414901902740000"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "121821044"
    ],
    "old_value": [
      "121821044"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "1613166414901902740000": {
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
  "report_fix_date": "2025-12-21T11:58:25",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P214",
  "report_revision_new": 2444911498,
  "report_revision_old": 2444478181,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "121821044"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 19,
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "121821044"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "121821044",
    "1613166414901902740000"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for the Virtual International Authority File database [format: up to 22 digits]; please note: VIAF is a cluster, the ID can include multiple items",
    "label": "VIAF cluster ID"
  },
  "qid": {
    "description": "Hong Kong martial artist and actor",
    "label": "Lo Mang"
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
        "121821044"
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

## 044. `repair_Q721001_2311223818`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "MB8P8P"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "MB8E8P",
      "MB8J8P",
      "MB8P8P"
    ],
    "new_value": [
      "MB8E8P",
      "MB8J8P",
      "MB8P8P"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "MB8E8P",
      "MB8J8P"
    ],
    "old_value": [
      "MB8E8P",
      "MB8J8P"
    ],
    "removed_unique_values": [],
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

## 045. `repair_Q72469678_2424590690`

| Field | Value |
|---|---|
| qid | Q72469678 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q72469678::P2877 |
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
| truth_tokens_preview | ["29363120", "9319847"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "29363120",
    "9319847"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2424590690,
  "value": [
    "29363120",
    "9319847"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "29363120",
      "9319847"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "29363120",
      "9319847"
    ],
    "new_value": [
      "29363120",
      "9319847"
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
      "29363120": {
        "new": 1,
        "old": 0
      },
      "9319847": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [],
  "needed": 2,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "29363120",
    "9319847"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "Disperse Blue 183"
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
      "needed": 2,
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

## 046. `repair_Q72482750_2425407227`

| Field | Value |
|---|---|
| qid | Q72482750 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q72482750::P2877 |
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
| truth_tokens_preview | ["117208", "7582475"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "117208",
    "7582475"
  ],
  "old_value": [
    "SCHEMBL117208",
    "7582475"
  ],
  "revision_id": 2425407227,
  "value": [
    "117208",
    "7582475"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "117208"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "117208",
      "7582475"
    ],
    "new_value": [
      "117208",
      "7582475"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "7582475",
      "SCHEMBL117208"
    ],
    "old_value": [
      "SCHEMBL117208",
      "7582475"
    ],
    "removed_unique_values": [
      "SCHEMBL117208"
    ],
    "value_multiplicity_changes": {
      "117208": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL117208": {
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
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "SCHEMBL117208",
    "7582475"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "SCHEMBL117208"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "SCHEMBL117208",
      "7582475"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "117208",
    "7582475"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "5-Indazolamine"
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
      "local_ids_count": 5,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "SCHEMBL117208"
        }
      ],
      "needed": 2,
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
        "SCHEMBL117208",
        "7582475"
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

## 047. `repair_Q72518501_2424863878`

| Field | Value |
|---|---|
| qid | Q72518501 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q72518501::P2877 |
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
| truth_tokens_preview | ["10762405", "3653849"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "AdrianoRutz",
  "kind": "A_BOX",
  "new_value": [
    "10762405",
    "3653849"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2424863878,
  "value": [
    "10762405",
    "3653849"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "10762405",
      "3653849"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "10762405",
      "3653849"
    ],
    "new_value": [
      "10762405",
      "3653849"
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
      "10762405": {
        "new": 1,
        "old": 0
      },
      "3653849": {
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
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": false,
  "matches": [],
  "needed": 2,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "MISSING"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "10762405",
    "3653849"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "4-Chloro-1,2-dihydro-2-oxo-3-pyridinecarboxylic acid"
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
      "needed": 2,
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

## 048. `repair_Q8030771_2433495254`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "120774"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "120774"
    ],
    "new_value": [
      "120774"
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
      "120774": {
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

## 049. `repair_Q8963433_2309428741`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "FF6J01"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "FF6J01"
    ],
    "new_value": [
      "FF6J01"
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
      "FF6J01": {
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

## 050. `repair_Q9051804_2086436363`

| Field | Value |
|---|---|
| qid | Q9051804 |
| property | P3795 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q9051804::P3795 |
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
| truth_tokens_preview | ["OENDRU/"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "OENDRU/"
  ],
  "old_value": [
    "OENDRU"
  ],
  "revision_id": 2086436363,
  "value": [
    "OENDRU/"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "OENDRU/"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "OENDRU/"
    ],
    "new_value": [
      "OENDRU/"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "OENDRU"
    ],
    "old_value": [
      "OENDRU"
    ],
    "removed_unique_values": [
      "OENDRU"
    ],
    "value_multiplicity_changes": {
      "OENDRU": {
        "new": 0,
        "old": 1
      },
      "OENDRU/": {
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
    "OENDRU"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 19,
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "OENDRU"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "OENDRU/"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "label": "Oenothera drummondii"
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
        "OENDRU"
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
