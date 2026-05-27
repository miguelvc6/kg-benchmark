# TypeC_UNKNOWN_OR_SPARSE_DIAGNOSTIC

Cases: 10

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q109490797_2444113068`

| Field | Value |
|---|---|
| qid | Q109490797 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_INCOMPLETE_LOCAL_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q109490797::P373 |
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
| truth_tokens_preview | ["Antonio Trotti Bentivoglio (1627-1684)"] |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Antonio Trotti Bentivoglio (1627-1684)"
  ],
  "old_value": [
    "Category:Antonio Trotti Bentivoglio (1627-1684)"
  ],
  "revision_id": 2444113068,
  "value": [
    "Antonio Trotti Bentivoglio (1627-1684)"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Antonio Trotti Bentivoglio (1627-1684)"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Antonio Trotti Bentivoglio (1627-1684)"
    ],
    "new_value": [
      "Antonio Trotti Bentivoglio (1627-1684)"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Category:Antonio Trotti Bentivoglio (1627-1684)"
    ],
    "old_value": [
      "Category:Antonio Trotti Bentivoglio (1627-1684)"
    ],
    "removed_unique_values": [
      "Category:Antonio Trotti Bentivoglio (1627-1684)"
    ],
    "value_multiplicity_changes": {
      "Antonio Trotti Bentivoglio (1627-1684)": {
        "new": 1,
        "old": 0
      },
      "Category:Antonio Trotti Bentivoglio (1627-1684)": {
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
  "report_fix_date": "2025-12-21T10:58:44",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2444891710,
  "report_revision_old": 2444464305,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Commons link"
  ],
  "value": [
    "Category:Antonio Trotti Bentivoglio (1627-1684)"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_boundary",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "Antonio Trotti Bentivoglio (1627-1684)"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Category:Antonio Trotti Bentivoglio (1627-1684)"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Antonio Trotti Bentivoglio (1627-1684)"
  ],
  "truth_tokens_in_recorded_matches": [
    "Antonio Trotti Bentivoglio (1627-1684)"
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
    "description": "italian nobleman and military officer, master of the field general of the spanish army.",
    "label": "Antonio Trotti Bentivoglio"
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
        "Antonio Trotti Bentivoglio (1627-1684)"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Antonio Trotti Bentivoglio (1627-1684)": 1
      },
      "new_unique": [
        "Antonio Trotti Bentivoglio (1627-1684)"
      ],
      "new_values": [
        "Antonio Trotti Bentivoglio (1627-1684)"
      ],
      "new_values_raw": [
        "Antonio Trotti Bentivoglio (1627-1684)"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Category:Antonio Trotti Bentivoglio (1627-1684)": 1
      },
      "old_unique": [
        "Category:Antonio Trotti Bentivoglio (1627-1684)"
      ],
      "old_values": [
        "Category:Antonio Trotti Bentivoglio (1627-1684)"
      ],
      "old_values_raw": [
        "Category:Antonio Trotti Bentivoglio (1627-1684)"
      ],
      "removed_unique_values": [
        "Category:Antonio Trotti Bentivoglio (1627-1684)"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Antonio Trotti Bentivoglio (1627-1684)": {
          "new": 1,
          "old": 0
        },
        "Category:Antonio Trotti Bentivoglio (1627-1684)": {
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
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_boundary",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "Antonio Trotti Bentivoglio (1627-1684)"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": true
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Category:Antonio Trotti Bentivoglio (1627-1684)"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "pre_repair_target_only_not_local",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q109947010_2440551998`

| Field | Value |
|---|---|
| qid | Q109947010 |
| property | P675 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_INCOMPLETE_LOCAL_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q109947010::P675 |
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
| truth_tokens_preview | ["qdpxzgEACAAJ"] |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Gerwoman",
  "kind": "A_BOX",
  "new_value": [
    "qdpxzgEACAAJ"
  ],
  "old_value": [
    "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it"
  ],
  "revision_id": 2440551998,
  "value": [
    "qdpxzgEACAAJ"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "qdpxzgEACAAJ"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "qdpxzgEACAAJ"
    ],
    "new_value": [
      "qdpxzgEACAAJ"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it"
    ],
    "old_value": [
      "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it"
    ],
    "removed_unique_values": [
      "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it"
    ],
    "value_multiplicity_changes": {
      "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it": {
        "new": 0,
        "old": 1
      },
      "qdpxzgEACAAJ": {
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
  "report_fix_date": "2025-12-12T09:45:25",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P675",
  "report_revision_new": 2441179352,
  "report_revision_old": 2440831862,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 5,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_boundary",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "qdpxzgEACAAJ"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "qdpxzgEACAAJ"
  ],
  "truth_tokens_in_recorded_matches": [
    "qdpxzgEACAAJ"
  ],
  "used_literal_substring": true
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a book edition in Google Books",
    "label": "Google Books ID"
  },
  "qid": {
    "description": "art book by Franco Loi",
    "label": "Cronaca di una mostra Mario Bardi"
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
      "added_unique_values": [
        "qdpxzgEACAAJ"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "qdpxzgEACAAJ": 1
      },
      "new_unique": [
        "qdpxzgEACAAJ"
      ],
      "new_values": [
        "qdpxzgEACAAJ"
      ],
      "new_values_raw": [
        "qdpxzgEACAAJ"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it": 1
      },
      "old_unique": [
        "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it"
      ],
      "old_values": [
        "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it"
      ],
      "old_values_raw": [
        "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it"
      ],
      "removed_unique_values": [
        "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it": {
          "new": 0,
          "old": 1
        },
        "qdpxzgEACAAJ": {
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
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_boundary",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "qdpxzgEACAAJ"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": true
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "https://www.google.it/books/edition/Cronaca_di_una_mostra/qdpxzgEACAAJ?hl=it"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "pre_repair_target_only_not_local",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q110331423_2442356891`

| Field | Value |
|---|---|
| qid | Q110331423 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_INCOMPLETE_LOCAL_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q110331423::P373 |
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
| truth_tokens_preview | ["Cemetery of Asnelles"] |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Cemetery of Asnelles"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Cemetery of Asnelles"
    ],
    "new_value": [
      "Cemetery of Asnelles"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Category:Cemetery of Asnelles"
    ],
    "old_value": [
      "Category:Cemetery of Asnelles"
    ],
    "removed_unique_values": [
      "Category:Cemetery of Asnelles"
    ],
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
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 7,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_boundary",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "Cemetery of Asnelles"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Category:Cemetery of Asnelles"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Cemetery of Asnelles"
  ],
  "truth_tokens_in_recorded_matches": [
    "Cemetery of Asnelles"
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_boundary",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "Cemetery of Asnelles"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": true
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Category:Cemetery of Asnelles"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "pre_repair_target_only_not_local",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q135903816_2444882711`

| Field | Value |
|---|---|
| qid | Q135903816 |
| property | P345 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_INCOMPLETE_LOCAL_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q135903816::P345 |
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
| truth_tokens_preview | ["tt11730114"] |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Koffeinfrei-org",
  "kind": "A_BOX",
  "new_value": [
    "tt11730114"
  ],
  "old_value": [
    "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2"
  ],
  "revision_id": 2444882711,
  "value": [
    "tt11730114"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "tt11730114"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "tt11730114"
    ],
    "new_value": [
      "tt11730114"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2"
    ],
    "old_value": [
      "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2"
    ],
    "removed_unique_values": [
      "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2"
    ],
    "value_multiplicity_changes": {
      "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2": {
        "new": 0,
        "old": 1
      },
      "tt11730114": {
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
  "report_fix_date": "2025-12-22T10:47:55",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P345",
  "report_revision_new": 2445465851,
  "report_revision_old": 2444895883,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 9,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_boundary",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "tt11730114"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "tt11730114"
  ],
  "truth_tokens_in_recorded_matches": [
    "tt11730114"
  ],
  "used_literal_substring": true
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for the IMDb (with prefix 'tt', 'nm', 'co', 'ev', 'ch' or 'ni')",
    "label": "IMDb ID"
  },
  "qid": {
    "description": "film by Denis Gheerbrant",
    "label": "Amour Rue De Lappe"
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
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "tt11730114"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "tt11730114": 1
      },
      "new_unique": [
        "tt11730114"
      ],
      "new_values": [
        "tt11730114"
      ],
      "new_values_raw": [
        "tt11730114"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2": 1
      },
      "old_unique": [
        "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2"
      ],
      "old_values": [
        "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2"
      ],
      "old_values_raw": [
        "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2"
      ],
      "removed_unique_values": [
        "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2": {
          "new": 0,
          "old": 1
        },
        "tt11730114": {
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
          "kind": "literal_boundary",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "tt11730114"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": true
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "https://www.imdb.com/it/title/tt11730114/?ref_=nm_flmg_job_1_cdt_c_2"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "pre_repair_target_only_not_local",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q135989124_2444889115`

| Field | Value |
|---|---|
| qid | Q135989124 |
| property | P345 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_INCOMPLETE_LOCAL_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q135989124::P345 |
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
| truth_tokens_preview | ["tt38048840"] |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Koffeinfrei-org",
  "kind": "A_BOX",
  "new_value": [
    "tt38048840"
  ],
  "old_value": [
    "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk"
  ],
  "revision_id": 2444889115,
  "value": [
    "tt38048840"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "tt38048840"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "tt38048840"
    ],
    "new_value": [
      "tt38048840"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk"
    ],
    "old_value": [
      "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk"
    ],
    "removed_unique_values": [
      "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk"
    ],
    "value_multiplicity_changes": {
      "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk": {
        "new": 0,
        "old": 1
      },
      "tt38048840": {
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
  "report_fix_date": "2025-12-22T10:47:55",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P345",
  "report_revision_new": 2445465851,
  "report_revision_old": 2444895883,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 9,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_boundary",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "tt38048840"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "tt38048840"
  ],
  "truth_tokens_in_recorded_matches": [
    "tt38048840"
  ],
  "used_literal_substring": true
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for the IMDb (with prefix 'tt', 'nm', 'co', 'ev', 'ch' or 'ni')",
    "label": "IMDb ID"
  },
  "qid": {
    "description": "Argentine parody short film, released on May 28, 2019",
    "label": "The Truman Parody Show INSSC"
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
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "tt38048840"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "tt38048840": 1
      },
      "new_unique": [
        "tt38048840"
      ],
      "new_values": [
        "tt38048840"
      ],
      "new_values_raw": [
        "tt38048840"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk": 1
      },
      "old_unique": [
        "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk"
      ],
      "old_values": [
        "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk"
      ],
      "old_values_raw": [
        "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk"
      ],
      "removed_unique_values": [
        "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk": {
          "new": 0,
          "old": 1
        },
        "tt38048840": {
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
          "kind": "literal_boundary",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "tt38048840"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": true
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "https://www.imdb.com/es-es/title/tt38048840/?ref_=ext_shr_lnk"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "pre_repair_target_only_not_local",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q137437438_2443816791`

| Field | Value |
|---|---|
| qid | Q137437438 |
| property | P4985 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_INCOMPLETE_LOCAL_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q19474404 |
| group_key | ABOX::Q137437438::P4985 |
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
| truth_tokens_preview | ["3895926"] |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "3895926"
  ],
  "old_value": [
    "3895926-jennifer-trejo"
  ],
  "revision_id": 2443816791,
  "value": [
    "3895926"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "3895926"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "3895926"
    ],
    "new_value": [
      "3895926"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "3895926-jennifer-trejo"
    ],
    "old_value": [
      "3895926-jennifer-trejo"
    ],
    "removed_unique_values": [
      "3895926-jennifer-trejo"
    ],
    "value_multiplicity_changes": {
      "3895926": {
        "new": 1,
        "old": 0
      },
      "3895926-jennifer-trejo": {
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
  "report_fix_date": "2025-12-20T07:06:29",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4985",
  "report_revision_new": 2444405812,
  "report_revision_old": 2443992605,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "3895926-jennifer-trejo"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 7,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_boundary",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "3895926"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "3895926-jennifer-trejo"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "3895926"
  ],
  "truth_tokens_in_recorded_matches": [
    "3895926"
  ],
  "used_literal_substring": true
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a person at The Movie Database",
    "label": "TMDB person ID"
  },
  "qid": {
    "description": "actor",
    "label": "Jennifer Trejo"
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
        "3895926"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "3895926": 1
      },
      "new_unique": [
        "3895926"
      ],
      "new_values": [
        "3895926"
      ],
      "new_values_raw": [
        "3895926"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "3895926-jennifer-trejo": 1
      },
      "old_unique": [
        "3895926-jennifer-trejo"
      ],
      "old_values": [
        "3895926-jennifer-trejo"
      ],
      "old_values_raw": [
        "3895926-jennifer-trejo"
      ],
      "removed_unique_values": [
        "3895926-jennifer-trejo"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "3895926": {
          "new": 1,
          "old": 0
        },
        "3895926-jennifer-trejo": {
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_boundary",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "3895926"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": true
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "3895926-jennifer-trejo"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "pre_repair_target_only_not_local",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q137442818_2444601682`

| Field | Value |
|---|---|
| qid | Q137442818 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_INCOMPLETE_LOCAL_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q137442818::P373 |
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
| truth_tokens_preview | ["Government of SSC-Khatumo"] |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Government of SSC-Khatumo"
  ],
  "old_value": [
    "Category:Government of SSC-Khatumo"
  ],
  "revision_id": 2444601682,
  "value": [
    "Government of SSC-Khatumo"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Government of SSC-Khatumo"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Government of SSC-Khatumo"
    ],
    "new_value": [
      "Government of SSC-Khatumo"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Category:Government of SSC-Khatumo"
    ],
    "old_value": [
      "Category:Government of SSC-Khatumo"
    ],
    "removed_unique_values": [
      "Category:Government of SSC-Khatumo"
    ],
    "value_multiplicity_changes": {
      "Category:Government of SSC-Khatumo": {
        "new": 0,
        "old": 1
      },
      "Government of SSC-Khatumo": {
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
    "Category:Government of SSC-Khatumo"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 3,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_boundary",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "Government of SSC-Khatumo"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Category:Government of SSC-Khatumo"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Government of SSC-Khatumo"
  ],
  "truth_tokens_in_recorded_matches": [
    "Government of SSC-Khatumo"
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
    "description": "Somali subnational government",
    "label": "government of Waqooyi Bari"
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
        "Government of SSC-Khatumo"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Government of SSC-Khatumo": 1
      },
      "new_unique": [
        "Government of SSC-Khatumo"
      ],
      "new_values": [
        "Government of SSC-Khatumo"
      ],
      "new_values_raw": [
        "Government of SSC-Khatumo"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Category:Government of SSC-Khatumo": 1
      },
      "old_unique": [
        "Category:Government of SSC-Khatumo"
      ],
      "old_values": [
        "Category:Government of SSC-Khatumo"
      ],
      "old_values_raw": [
        "Category:Government of SSC-Khatumo"
      ],
      "removed_unique_values": [
        "Category:Government of SSC-Khatumo"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Category:Government of SSC-Khatumo": {
          "new": 0,
          "old": 1
        },
        "Government of SSC-Khatumo": {
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
      "local_ids_count": 3,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_boundary",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "Government of SSC-Khatumo"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": true
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Category:Government of SSC-Khatumo"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "pre_repair_target_only_not_local",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q137539453_2446042791`

| Field | Value |
|---|---|
| qid | Q137539453 |
| property | P4985 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_INCOMPLETE_LOCAL_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q19474404 |
| group_key | ABOX::Q137539453::P4985 |
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
| truth_tokens_preview | ["2278168"] |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "2278168"
  ],
  "old_value": [
    "2278168-stella-nwimo"
  ],
  "revision_id": 2446042791,
  "value": [
    "2278168"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "2278168"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "2278168"
    ],
    "new_value": [
      "2278168"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "2278168-stella-nwimo"
    ],
    "old_value": [
      "2278168-stella-nwimo"
    ],
    "removed_unique_values": [
      "2278168-stella-nwimo"
    ],
    "value_multiplicity_changes": {
      "2278168": {
        "new": 1,
        "old": 0
      },
      "2278168-stella-nwimo": {
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
  "report_fix_date": "2025-12-25T15:05:44",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4985",
  "report_revision_new": 2446982127,
  "report_revision_old": 2446410002,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "2278168-stella-nwimo"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 7,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_boundary",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "2278168"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "2278168-stella-nwimo"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "2278168"
  ],
  "truth_tokens_in_recorded_matches": [
    "2278168"
  ],
  "used_literal_substring": true
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a person at The Movie Database",
    "label": "TMDB person ID"
  },
  "qid": {
    "description": "film producer",
    "label": "Stella Nwimo"
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
        "2278168"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2278168": 1
      },
      "new_unique": [
        "2278168"
      ],
      "new_values": [
        "2278168"
      ],
      "new_values_raw": [
        "2278168"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "2278168-stella-nwimo": 1
      },
      "old_unique": [
        "2278168-stella-nwimo"
      ],
      "old_values": [
        "2278168-stella-nwimo"
      ],
      "old_values_raw": [
        "2278168-stella-nwimo"
      ],
      "removed_unique_values": [
        "2278168-stella-nwimo"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "2278168": {
          "new": 1,
          "old": 0
        },
        "2278168-stella-nwimo": {
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_boundary",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "2278168"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": true
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "2278168-stella-nwimo"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "pre_repair_target_only_not_local",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q50409009_2447258829`

| Field | Value |
|---|---|
| qid | Q50409009 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_INCOMPLETE_LOCAL_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| group_key | ABOX::Q50409009::P373 |
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
| truth_tokens_preview | ["Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"] |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Pi bot",
  "kind": "A_BOX",
  "new_value": [
    "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
  ],
  "old_value": [
    "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)"
  ],
  "revision_id": 2447258829,
  "value": [
    "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
    ],
    "new_value": [
      "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)"
    ],
    "old_value": [
      "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)"
    ],
    "removed_unique_values": [
      "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)"
    ],
    "value_multiplicity_changes": {
      "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)": {
        "new": 0,
        "old": 1
      },
      "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)": {
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
    "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 7,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
  ],
  "truth_tokens_in_recorded_matches": [
    "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
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
    "label": "Pond with steppe areas \""
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
        "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)": 1
      },
      "new_unique": [
        "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
      ],
      "new_values": [
        "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
      ],
      "new_values_raw": [
        "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)": 1
      },
      "old_unique": [
        "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)"
      ],
      "old_values": [
        "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)"
      ],
      "old_values_raw": [
        "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)"
      ],
      "removed_unique_values": [
        "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)": {
          "new": 0,
          "old": 1
        },
        "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)": {
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
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
        "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "pre_repair_target_only_not_local",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q9837227_2445827867`

| Field | Value |
|---|---|
| qid | Q9837227 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_INCOMPLETE_LOCAL_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q9837227::P373 |
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
| truth_tokens_preview | ["Local authority associations"] |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Local authority associations"
  ],
  "old_value": [
    ":Local authority associations"
  ],
  "revision_id": 2445827867,
  "value": [
    "Local authority associations"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Local authority associations"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Local authority associations"
    ],
    "new_value": [
      "Local authority associations"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      ":Local authority associations"
    ],
    "old_value": [
      ":Local authority associations"
    ],
    "removed_unique_values": [
      ":Local authority associations"
    ],
    "value_multiplicity_changes": {
      ":Local authority associations": {
        "new": 0,
        "old": 1
      },
      "Local authority associations": {
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
  "report_fix_date": "2025-12-24T12:12:14",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2446526020,
  "report_revision_old": 2446056750,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    ":Local authority associations"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 7,
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_boundary",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "Local authority associations"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      ":Local authority associations"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Local authority associations"
  ],
  "truth_tokens_in_recorded_matches": [
    "Local authority associations"
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
    "label": "Category:Associations of local governments"
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
        "Local authority associations"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Local authority associations": 1
      },
      "new_unique": [
        "Local authority associations"
      ],
      "new_values": [
        "Local authority associations"
      ],
      "new_values_raw": [
        "Local authority associations"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        ":Local authority associations": 1
      },
      "old_unique": [
        ":Local authority associations"
      ],
      "old_values": [
        ":Local authority associations"
      ],
      "old_values_raw": [
        ":Local authority associations"
      ],
      "removed_unique_values": [
        ":Local authority associations"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        ":Local authority associations": {
          "new": 0,
          "old": 1
        },
        "Local authority associations": {
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
      "local_ids_count": 7,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_boundary",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "Local authority associations"
        }
      ],
      "needed": 1,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
      ],
      "used_literal_substring": true
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        ":Local authority associations"
      ],
      "used_pre_repair_value": true
    }
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "pre_repair_target_only_not_local",
    "step": "branch"
  }
]
```

---
