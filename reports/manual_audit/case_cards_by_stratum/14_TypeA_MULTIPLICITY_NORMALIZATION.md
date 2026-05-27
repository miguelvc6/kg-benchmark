# TypeA_MULTIPLICITY_NORMALIZATION

Cases: 10

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q13626898_2445861529`

| Field | Value |
|---|---|
| qid | Q13626898 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| group_key | ABOX::Q13626898::P225 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Euryphorus nordmannii"] |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Brya",
  "kind": "A_BOX",
  "new_value": [
    "Euryphorus nordmannii"
  ],
  "old_value": [
    "Euryphorus nordmannii",
    "Euryphorus nordmannii"
  ],
  "revision_id": 2445861529,
  "value": [
    "Euryphorus nordmannii"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Euryphorus nordmannii"
    ],
    "new_value": [
      "Euryphorus nordmannii"
    ],
    "normalized_unique_values_unchanged": true,
    "old_count": 2,
    "old_unique": [
      "Euryphorus nordmannii"
    ],
    "old_value": [
      "Euryphorus nordmannii",
      "Euryphorus nordmannii"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "Euryphorus nordmannii": {
        "new": 1,
        "old": 2
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-24T13:09:37",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2446547742,
  "report_revision_old": 2446076213,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Euryphorus nordmannii",
    "Euryphorus nordmannii"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "Euryphorus nordmannii"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "description": "species of crustacean",
    "label": "Euryphorus nordmannii"
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
        "Euryphorus nordmannii": 1
      },
      "new_unique": [
        "Euryphorus nordmannii"
      ],
      "new_values": [
        "Euryphorus nordmannii"
      ],
      "new_values_raw": [
        "Euryphorus nordmannii"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "Euryphorus nordmannii": 2
      },
      "old_unique": [
        "Euryphorus nordmannii"
      ],
      "old_values": [
        "Euryphorus nordmannii",
        "Euryphorus nordmannii"
      ],
      "old_values_raw": [
        "Euryphorus nordmannii",
        "Euryphorus nordmannii"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "Euryphorus nordmannii"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "Euryphorus nordmannii": {
          "new": 1,
          "old": 2
        }
      }
    },
    "result": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "kind": "MULTIPLICITY",
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
    "result": "multiplicity_normalization",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q136881481_2439730444`

| Field | Value |
|---|---|
| qid | Q136881481 |
| property | P212 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502410 |
| group_key | ABOX::Q136881481::P212 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["978-82-530-3646-5"] |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Pfadintegral",
  "kind": "A_BOX",
  "new_value": [
    "978-82-530-3646-5"
  ],
  "old_value": [
    "978-82-530-3646-5",
    "978-82-530-3646-5"
  ],
  "revision_id": 2439730444,
  "value": [
    "978-82-530-3646-5"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "978-82-530-3646-5"
    ],
    "new_value": [
      "978-82-530-3646-5"
    ],
    "normalized_unique_values_unchanged": true,
    "old_count": 2,
    "old_unique": [
      "978-82-530-3646-5"
    ],
    "old_value": [
      "978-82-530-3646-5",
      "978-82-530-3646-5"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "978-82-530-3646-5": {
        "new": 1,
        "old": 2
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-10T11:10:27",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P212",
  "report_revision_new": 2440424839,
  "report_revision_old": 2440015789,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "978-82-530-3646-5",
    "978-82-530-3646-5"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "978-82-530-3646-5"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a book (edition), thirteen digit",
    "label": "International Standard Book Number-13"
  },
  "qid": {
    "description": "bok",
    "label": "Tenke, fort og langsomt"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "978-82-530-3646-5": 1
      },
      "new_unique": [
        "978-82-530-3646-5"
      ],
      "new_values": [
        "978-82-530-3646-5"
      ],
      "new_values_raw": [
        "978-82-530-3646-5"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "978-82-530-3646-5": 2
      },
      "old_unique": [
        "978-82-530-3646-5"
      ],
      "old_values": [
        "978-82-530-3646-5",
        "978-82-530-3646-5"
      ],
      "old_values_raw": [
        "978-82-530-3646-5",
        "978-82-530-3646-5"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "978-82-530-3646-5"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "978-82-530-3646-5": {
          "new": 1,
          "old": 2
        }
      }
    },
    "result": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "kind": "MULTIPLICITY",
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
    "result": "multiplicity_normalization",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q18554077_2393607898`

| Field | Value |
|---|---|
| qid | Q18554077 |
| property | P2892 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | tail |
| constraint_family | Q19474404 |
| group_key | ABOX::Q18554077::P2892 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["C0152207"] |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Mahir256",
  "kind": "A_BOX",
  "new_value": [
    "C0152207"
  ],
  "old_value": [
    "C0152207",
    "C0152207"
  ],
  "revision_id": 2393607898,
  "value": [
    "C0152207"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "C0152207"
    ],
    "new_value": [
      "C0152207"
    ],
    "normalized_unique_values_unchanged": true,
    "old_count": 2,
    "old_unique": [
      "C0152207"
    ],
    "old_value": [
      "C0152207",
      "C0152207"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "C0152207": {
        "new": 1,
        "old": 2
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-21T11:13:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2892",
  "report_revision_new": 2395046684,
  "report_revision_old": 2390014418,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "report_violation_types": [
    "Single value",
    "Unique value"
  ],
  "value": [
    "C0152207",
    "C0152207"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "C0152207"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "NLM Unified Medical Language System (UMLS) controlled biomedical vocabulary unique identifier",
    "label": "UMLS CUI"
  },
  "qid": {
    "description": "Human disease",
    "label": "alternating exotropia"
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
        "C0152207": 1
      },
      "new_unique": [
        "C0152207"
      ],
      "new_values": [
        "C0152207"
      ],
      "new_values_raw": [
        "C0152207"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "C0152207": 2
      },
      "old_unique": [
        "C0152207"
      ],
      "old_values": [
        "C0152207",
        "C0152207"
      ],
      "old_values_raw": [
        "C0152207",
        "C0152207"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "C0152207"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "C0152207": {
          "new": 1,
          "old": 2
        }
      }
    },
    "result": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "kind": "MULTIPLICITY",
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
    "result": "multiplicity_normalization",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q3588573_2440234611`

| Field | Value |
|---|---|
| qid | Q3588573 |
| property | P8392 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| group_key | ABOX::Q3588573::P8392 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["album/ange/emile-jacotey"] |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Duffseb",
  "kind": "A_BOX",
  "new_value": [
    "album/ange/emile-jacotey"
  ],
  "old_value": [
    "album/ange/emile-jacotey",
    "album/ange/emile-jacotey"
  ],
  "revision_id": 2440234611,
  "value": [
    "album/ange/emile-jacotey"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "album/ange/emile-jacotey"
    ],
    "new_value": [
      "album/ange/emile-jacotey"
    ],
    "normalized_unique_values_unchanged": true,
    "old_count": 2,
    "old_unique": [
      "album/ange/emile-jacotey"
    ],
    "old_value": [
      "album/ange/emile-jacotey",
      "album/ange/emile-jacotey"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "album/ange/emile-jacotey": {
        "new": 1,
        "old": 2
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-10T05:35:17",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8392",
  "report_revision_new": 2440357832,
  "report_revision_old": 2439878496,
  "report_violation_type": "Label in mul language",
  "report_violation_type_normalized": "Label in mul language",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Label in mul language",
  "value": [
    "album/ange/emile-jacotey",
    "album/ange/emile-jacotey"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "album/ange/emile-jacotey"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a music release on Rate Your Music site",
    "label": "Rate Your Music release ID"
  },
  "qid": {
    "description": "1975 album by Ange",
    "label": "Émile Jacotey"
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
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
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
        "album/ange/emile-jacotey": 1
      },
      "new_unique": [
        "album/ange/emile-jacotey"
      ],
      "new_values": [
        "album/ange/emile-jacotey"
      ],
      "new_values_raw": [
        "album/ange/emile-jacotey"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "album/ange/emile-jacotey": 2
      },
      "old_unique": [
        "album/ange/emile-jacotey"
      ],
      "old_values": [
        "album/ange/emile-jacotey",
        "album/ange/emile-jacotey"
      ],
      "old_values_raw": [
        "album/ange/emile-jacotey",
        "album/ange/emile-jacotey"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "album/ange/emile-jacotey"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "album/ange/emile-jacotey": {
          "new": 1,
          "old": 2
        }
      }
    },
    "result": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "kind": "MULTIPLICITY",
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
    "result": "multiplicity_normalization",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q3630746_2447040400`

| Field | Value |
|---|---|
| qid | Q3630746 |
| property | P10760 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q3630746::P10760 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1668"] |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Elya",
  "kind": "A_BOX",
  "new_value": [
    "1668"
  ],
  "old_value": [
    "1668",
    "1668"
  ],
  "revision_id": 2447040400,
  "value": [
    "1668"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "1668"
    ],
    "new_value": [
      "1668"
    ],
    "normalized_unique_values_unchanged": true,
    "old_count": 2,
    "old_unique": [
      "1668"
    ],
    "old_value": [
      "1668",
      "1668"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "1668": {
        "new": 1,
        "old": 2
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T06:45:38",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P10760",
  "report_revision_new": 2447680554,
  "report_revision_old": 2447268842,
  "report_violation_type": "Item P|170",
  "report_violation_type_normalized": "Item P|170",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|170",
  "value": [
    "1668",
    "1668"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "1668"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier in the Munich Central Collecting Point database",
    "label": "MCCP ID"
  },
  "qid": {
    "description": "painting by Peter Paul Rubens",
    "label": "Self-Portrait in a Circle of Friends in Mantua"
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
        "1668": 1
      },
      "new_unique": [
        "1668"
      ],
      "new_values": [
        "1668"
      ],
      "new_values_raw": [
        "1668"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "1668": 2
      },
      "old_unique": [
        "1668"
      ],
      "old_values": [
        "1668",
        "1668"
      ],
      "old_values_raw": [
        "1668",
        "1668"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "1668"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "1668": {
          "new": 1,
          "old": 2
        }
      }
    },
    "result": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "kind": "MULTIPLICITY",
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
    "result": "multiplicity_normalization",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q510595_2442836549`

| Field | Value |
|---|---|
| qid | Q510595 |
| property | P136 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q510595::P136 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q2297927", "Q52162262", "Q130232", "Q11304653", "Q157443", "...(+8)"] |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "BigJack20",
  "kind": "A_BOX",
  "new_value": [
    "Q2297927",
    "Q52162262",
    "Q130232",
    "Q11304653",
    "Q157443",
    "Q5778924",
    "Q113485322",
    "Q19367312",
    "Q1054574",
    "Q2484376",
    "Q959790",
    "Q1200678",
    "Q188473"
  ],
  "new_value_descriptions_en": [
    "film genre",
    "film where work of literature makes up the basic",
    "film genre",
    "film characterized by suspense",
    "genre of film in which the main emphasis is on humour",
    "film genre",
    "film genre",
    "film genre",
    "film genre",
    "film genre that evokes excitement and suspense in the audience",
    "film genre",
    "sub-genre of the more general category of crime film and at times the thriller genre",
    "film genre"
  ],
  "new_value_labels_en": [
    "spy film",
    "film based on literature",
    "drama film",
    "suspense film",
    "comedy film",
    "black comedy film",
    "crime drama film",
    "crime thriller film",
    "romance film",
    "thriller film",
    "crime film",
    "mystery film",
    "action film"
  ],
  "old_value": [
    "Q2297927",
    "Q52162262",
    "Q130232",
    "Q11304653",
    "Q157443",
    "Q2297927",
    "Q5778924",
    "Q113485322",
    "Q19367312",
    "Q1054574",
    "Q2484376",
    "Q959790",
    "Q1200678",
    "Q188473"
  ],
  "old_value_descriptions_en": [
    "film genre",
    "film where work of literature makes up the basic",
    "film genre",
    "film characterized by suspense",
    "genre of film in which the main emphasis is on humour",
    "film genre",
    "film genre",
    "film genre",
    "film genre",
    "film genre",
    "film genre that evokes excitement and suspense in the audience",
    "film genre",
    "sub-genre of the more general category of crime film and at times the thriller genre",
    "film genre"
  ],
  "old_value_labels_en": [
    "spy film",
    "film based on literature",
    "drama film",
    "suspense film",
    "comedy film",
    "spy film",
    "black comedy film",
    "crime drama film",
    "crime thriller film",
    "romance film",
    "thriller film",
    "crime film",
    "mystery film",
    "action film"
  ],
  "revision_id": 2442836549,
  "value": [
    "Q2297927",
    "Q52162262",
    "Q130232",
    "Q11304653",
    "Q157443",
    "Q5778924",
    "Q113485322",
    "Q19367312",
    "Q1054574",
    "Q2484376",
    "Q959790",
    "Q1200678",
    "Q188473"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 13,
    "new_unique": [
      "Q1054574",
      "Q11304653",
      "Q113485322",
      "Q1200678",
      "Q130232",
      "Q157443",
      "Q188473",
      "Q19367312",
      "Q2297927",
      "Q2484376",
      "Q52162262",
      "Q5778924",
      "Q959790"
    ],
    "new_value": [
      "Q2297927",
      "Q52162262",
      "Q130232",
      "Q11304653",
      "Q157443",
      "Q5778924",
      "Q113485322",
      "Q19367312",
      "Q1054574",
      "Q2484376",
      "Q959790",
      "Q1200678",
      "Q188473"
    ],
    "normalized_unique_values_unchanged": true,
    "old_count": 14,
    "old_unique": [
      "Q1054574",
      "Q11304653",
      "Q113485322",
      "Q1200678",
      "Q130232",
      "Q157443",
      "Q188473",
      "Q19367312",
      "Q2297927",
      "Q2484376",
      "Q52162262",
      "Q5778924",
      "Q959790"
    ],
    "old_value": [
      "Q2297927",
      "Q52162262",
      "Q130232",
      "Q11304653",
      "Q157443",
      "Q2297927",
      "Q5778924",
      "Q113485322",
      "Q19367312",
      "Q1054574",
      "Q2484376",
      "Q959790",
      "Q1200678",
      "Q188473"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "Q2297927": {
        "new": 1,
        "old": 2
      }
    }
  },
  "value_descriptions_en": [
    "film genre",
    "film where work of literature makes up the basic",
    "film genre",
    "film characterized by suspense",
    "genre of film in which the main emphasis is on humour",
    "film genre",
    "film genre",
    "film genre",
    "film genre",
    "film genre that evokes excitement and suspense in the audience",
    "film genre",
    "sub-genre of the more general category of crime film and at times the thriller genre",
    "film genre"
  ],
  "value_labels_en": [
    "spy film",
    "film based on literature",
    "drama film",
    "suspense film",
    "comedy film",
    "black comedy film",
    "crime drama film",
    "crime thriller film",
    "romance film",
    "thriller film",
    "crime film",
    "mystery film",
    "action film"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-18T22:24:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P136",
  "report_revision_new": 2443886611,
  "report_revision_old": 2443886501,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "report_violation_types": [
    "None of",
    "Value type Q|483394, Q|11921029, Q|1406161, Q|1792644, Q|17155032, Q|17955, Q|3030248, Q|53001749, Q|862597, Q|1499017, Q|223393, Q|188451, Q|1991869, Q|6806507, Q|4263830, Q|107356781, Q|317623, Q|56055944, Q|64138195, Q|108676140, Q|128093, Q|82753, Q|110562117, Q|10428845, Q|126125231, Q|1078597, Q|47433, Q|37930, Q|25679497, Q|109551565, Q|735, Q|104822033, Q|41207, Q|855973, Q|104624828, Q|116474095"
  ],
  "value": [
    "Q2297927",
    "Q52162262",
    "Q130232",
    "Q11304653",
    "Q157443",
    "Q2297927",
    "Q5778924",
    "Q113485322",
    "Q19367312",
    "Q1054574",
    "Q2484376",
    "Q959790",
    "Q1200678",
    "Q188473"
  ],
  "value_descriptions_en": [
    "film genre",
    "film where work of literature makes up the basic",
    "film genre",
    "film characterized by suspense",
    "genre of film in which the main emphasis is on humour",
    "film genre",
    "film genre",
    "film genre",
    "film genre",
    "film genre",
    "film genre that evokes excitement and suspense in the audience",
    "film genre",
    "sub-genre of the more general category of crime film and at times the thriller genre",
    "film genre"
  ],
  "value_labels_en": [
    "spy film",
    "film based on literature",
    "drama film",
    "suspense film",
    "comedy film",
    "spy film",
    "black comedy film",
    "crime drama film",
    "crime thriller film",
    "romance film",
    "thriller film",
    "crime film",
    "mystery film",
    "action film"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "Q2297927",
    "Q52162262",
    "Q130232",
    "Q11304653",
    "Q157443",
    "Q5778924",
    "Q113485322",
    "Q19367312",
    "Q1054574",
    "Q2484376",
    "Q959790",
    "Q1200678",
    "Q188473"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
    "label": "genre"
  },
  "qid": {
    "description": "1936 film by Alfred Hitchcock",
    "label": "Secret Agent"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
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
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
        "Q1054574": 1,
        "Q11304653": 1,
        "Q113485322": 1,
        "Q1200678": 1,
        "Q130232": 1,
        "Q157443": 1,
        "Q188473": 1,
        "Q19367312": 1,
        "Q2297927": 1,
        "Q2484376": 1,
        "Q52162262": 1,
        "Q5778924": 1,
        "Q959790": 1
      },
      "new_unique": [
        "Q1054574",
        "Q11304653",
        "Q113485322",
        "Q1200678",
        "Q130232",
        "Q157443",
        "Q188473",
        "Q19367312",
        "Q2297927",
        "Q2484376",
        "Q52162262",
        "Q5778924",
        "Q959790"
      ],
      "new_values": [
        "Q2297927",
        "Q52162262",
        "Q130232",
        "Q11304653",
        "Q157443",
        "Q5778924",
        "Q113485322",
        "Q19367312",
        "Q1054574",
        "Q2484376",
        "Q959790",
        "Q1200678",
        "Q188473"
      ],
      "new_values_raw": [
        "Q2297927",
        "Q52162262",
        "Q130232",
        "Q11304653",
        "Q157443",
        "Q5778924",
        "Q113485322",
        "Q19367312",
        "Q1054574",
        "Q2484376",
        "Q959790",
        "Q1200678",
        "Q188473"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "Q1054574": 1,
        "Q11304653": 1,
        "Q113485322": 1,
        "Q1200678": 1,
        "Q130232": 1,
        "Q157443": 1,
        "Q188473": 1,
        "Q19367312": 1,
        "Q2297927": 2,
        "Q2484376": 1,
        "Q52162262": 1,
        "Q5778924": 1,
        "Q959790": 1
      },
      "old_unique": [
        "Q1054574",
        "Q11304653",
        "Q113485322",
        "Q1200678",
        "Q130232",
        "Q157443",
        "Q188473",
        "Q19367312",
        "Q2297927",
        "Q2484376",
        "Q52162262",
        "Q5778924",
        "Q959790"
      ],
      "old_values": [
        "Q2297927",
        "Q52162262",
        "Q130232",
        "Q11304653",
        "Q157443",
        "Q2297927",
        "Q5778924",
        "Q113485322",
        "Q19367312",
        "Q1054574",
        "Q2484376",
        "Q959790",
        "Q1200678",
        "Q188473"
      ],
      "old_values_raw": [
        "Q2297927",
        "Q52162262",
        "Q130232",
        "Q11304653",
        "Q157443",
        "Q2297927",
        "Q5778924",
        "Q113485322",
        "Q19367312",
        "Q1054574",
        "Q2484376",
        "Q959790",
        "Q1200678",
        "Q188473"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "Q1054574",
        "Q11304653",
        "Q113485322",
        "Q1200678",
        "Q130232",
        "Q157443",
        "Q188473",
        "Q19367312",
        "Q2297927",
        "Q2484376",
        "Q52162262",
        "Q5778924",
        "Q959790"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "Q2297927": {
          "new": 1,
          "old": 2
        }
      }
    },
    "result": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "kind": "MULTIPLICITY",
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
    "result": "multiplicity_normalization",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q6034386_2443074497`

| Field | Value |
|---|---|
| qid | Q6034386 |
| property | P8189 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q6034386::P8189 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["987013321227905171"] |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "987013321227905171"
  ],
  "old_value": [
    "987013321227905171",
    "987013321227905171"
  ],
  "revision_id": 2443074497,
  "value": [
    "987013321227905171"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "987013321227905171"
    ],
    "new_value": [
      "987013321227905171"
    ],
    "normalized_unique_values_unchanged": true,
    "old_count": 2,
    "old_unique": [
      "987013321227905171"
    ],
    "old_value": [
      "987013321227905171",
      "987013321227905171"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "987013321227905171": {
        "new": 1,
        "old": 2
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-18T14:09:07",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8189",
  "report_revision_new": 2443764355,
  "report_revision_old": 2443312104,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "report_violation_types": [
    "Single value",
    "Unique value"
  ],
  "value": [
    "987013321227905171",
    "987013321227905171"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "987013321227905171"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier used by the National Library of Israel",
    "label": "National Library of Israel J9U ID"
  },
  "qid": {
    "description": "museum in Autonomous City of Buenos Aires, Argentina",
    "label": "Pink House Museum"
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
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "987013321227905171": 1
      },
      "new_unique": [
        "987013321227905171"
      ],
      "new_values": [
        "987013321227905171"
      ],
      "new_values_raw": [
        "987013321227905171"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "987013321227905171": 2
      },
      "old_unique": [
        "987013321227905171"
      ],
      "old_values": [
        "987013321227905171",
        "987013321227905171"
      ],
      "old_values_raw": [
        "987013321227905171",
        "987013321227905171"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "987013321227905171"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "987013321227905171": {
          "new": 1,
          "old": 2
        }
      }
    },
    "result": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "kind": "MULTIPLICITY",
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
    "result": "multiplicity_normalization",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q60716478_2441533380`

| Field | Value |
|---|---|
| qid | Q60716478 |
| property | P27 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | tail |
| constraint_family | Q21510865 |
| group_key | ABOX::Q60716478::P27 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q148"] |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Dsp13",
  "kind": "A_BOX",
  "new_value": [
    "Q148"
  ],
  "new_value_descriptions_en": [
    "country in East Asia"
  ],
  "new_value_labels_en": [
    "People's Republic of China"
  ],
  "old_value": [
    "Q148",
    "Q148"
  ],
  "old_value_descriptions_en": [
    "country in East Asia",
    "country in East Asia"
  ],
  "old_value_labels_en": [
    "People's Republic of China",
    "People's Republic of China"
  ],
  "revision_id": 2441533380,
  "value": [
    "Q148"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q148"
    ],
    "new_value": [
      "Q148"
    ],
    "normalized_unique_values_unchanged": true,
    "old_count": 2,
    "old_unique": [
      "Q148"
    ],
    "old_value": [
      "Q148",
      "Q148"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "Q148": {
        "new": 1,
        "old": 2
      }
    }
  },
  "value_descriptions_en": [
    "country in East Asia"
  ],
  "value_labels_en": [
    "People's Republic of China"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T17:43:10",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P27",
  "report_revision_new": 2442355696,
  "report_revision_old": 2441850092,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": [
    "Q148",
    "Q148"
  ],
  "value_descriptions_en": [
    "country in East Asia",
    "country in East Asia"
  ],
  "value_labels_en": [
    "People's Republic of China",
    "People's Republic of China"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "Q148"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the object is a country that recognizes the subject as its citizen",
    "label": "country of citizenship"
  },
  "qid": {
    "description": "Chinese military nurse, Florence Nightingale Medal recipient",
    "label": "Yumei Suo"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
        "Q148": 1
      },
      "new_unique": [
        "Q148"
      ],
      "new_values": [
        "Q148"
      ],
      "new_values_raw": [
        "Q148"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "Q148": 2
      },
      "old_unique": [
        "Q148"
      ],
      "old_values": [
        "Q148",
        "Q148"
      ],
      "old_values_raw": [
        "Q148",
        "Q148"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "Q148"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "Q148": {
          "new": 1,
          "old": 2
        }
      }
    },
    "result": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "kind": "MULTIPLICITY",
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
    "result": "multiplicity_normalization",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q61612369_2444763348`

| Field | Value |
|---|---|
| qid | Q61612369 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| group_key | ABOX::Q61612369::P225 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Oncorhynchus clarkii virginalis"] |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Brya",
  "kind": "A_BOX",
  "new_value": [
    "Oncorhynchus clarkii virginalis"
  ],
  "old_value": [
    "Oncorhynchus clarkii virginalis",
    "Oncorhynchus clarkii virginalis"
  ],
  "revision_id": 2444763348,
  "value": [
    "Oncorhynchus clarkii virginalis"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Oncorhynchus clarkii virginalis"
    ],
    "new_value": [
      "Oncorhynchus clarkii virginalis"
    ],
    "normalized_unique_values_unchanged": true,
    "old_count": 2,
    "old_unique": [
      "Oncorhynchus clarkii virginalis"
    ],
    "old_value": [
      "Oncorhynchus clarkii virginalis",
      "Oncorhynchus clarkii virginalis"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "Oncorhynchus clarkii virginalis": {
        "new": 1,
        "old": 2
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-22T11:22:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2445477126,
  "report_revision_old": 2444910674,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Oncorhynchus clarkii virginalis",
    "Oncorhynchus clarkii virginalis"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "Oncorhynchus clarkii virginalis"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "description": "subspecies of fish",
    "label": "Oncorhynchus clarkii virginalis"
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
        "Oncorhynchus clarkii virginalis": 1
      },
      "new_unique": [
        "Oncorhynchus clarkii virginalis"
      ],
      "new_values": [
        "Oncorhynchus clarkii virginalis"
      ],
      "new_values_raw": [
        "Oncorhynchus clarkii virginalis"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "Oncorhynchus clarkii virginalis": 2
      },
      "old_unique": [
        "Oncorhynchus clarkii virginalis"
      ],
      "old_values": [
        "Oncorhynchus clarkii virginalis",
        "Oncorhynchus clarkii virginalis"
      ],
      "old_values_raw": [
        "Oncorhynchus clarkii virginalis",
        "Oncorhynchus clarkii virginalis"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "Oncorhynchus clarkii virginalis"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "Oncorhynchus clarkii virginalis": {
          "new": 1,
          "old": 2
        }
      }
    },
    "result": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "kind": "MULTIPLICITY",
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
    "result": "multiplicity_normalization",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q83302727_2439739710`

| Field | Value |
|---|---|
| qid | Q83302727 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| group_key | ABOX::Q83302727::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Portrait of Jan Hubrecht by Pieter van Slingelandt"] |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Portrait of Jan Hubrecht by Pieter van Slingelandt"
  ],
  "old_value": [
    "Portrait of Jan Hubrecht by Pieter van Slingelandt",
    "Portrait of Jan Hubrecht by Pieter van Slingelandt"
  ],
  "revision_id": 2439739710,
  "value": [
    "Portrait of Jan Hubrecht by Pieter van Slingelandt"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Portrait of Jan Hubrecht by Pieter van Slingelandt"
    ],
    "new_value": [
      "Portrait of Jan Hubrecht by Pieter van Slingelandt"
    ],
    "normalized_unique_values_unchanged": true,
    "old_count": 2,
    "old_unique": [
      "Portrait of Jan Hubrecht by Pieter van Slingelandt"
    ],
    "old_value": [
      "Portrait of Jan Hubrecht by Pieter van Slingelandt",
      "Portrait of Jan Hubrecht by Pieter van Slingelandt"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "Portrait of Jan Hubrecht by Pieter van Slingelandt": {
        "new": 1,
        "old": 2
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-10T10:22:34",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2440415829,
  "report_revision_old": 2439985888,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Portrait of Jan Hubrecht by Pieter van Slingelandt",
    "Portrait of Jan Hubrecht by Pieter van Slingelandt"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": null,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [
    "Portrait of Jan Hubrecht by Pieter van Slingelandt"
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
    "description": "painting by Pieter van Slingelandt",
    "label": "Portrait of Jan Hubrecht (1606-1669)"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "Portrait of Jan Hubrecht by Pieter van Slingelandt": 1
      },
      "new_unique": [
        "Portrait of Jan Hubrecht by Pieter van Slingelandt"
      ],
      "new_values": [
        "Portrait of Jan Hubrecht by Pieter van Slingelandt"
      ],
      "new_values_raw": [
        "Portrait of Jan Hubrecht by Pieter van Slingelandt"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "Portrait of Jan Hubrecht by Pieter van Slingelandt": 2
      },
      "old_unique": [
        "Portrait of Jan Hubrecht by Pieter van Slingelandt"
      ],
      "old_values": [
        "Portrait of Jan Hubrecht by Pieter van Slingelandt",
        "Portrait of Jan Hubrecht by Pieter van Slingelandt"
      ],
      "old_values_raw": [
        "Portrait of Jan Hubrecht by Pieter van Slingelandt",
        "Portrait of Jan Hubrecht by Pieter van Slingelandt"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "Portrait of Jan Hubrecht by Pieter van Slingelandt"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "Portrait of Jan Hubrecht by Pieter van Slingelandt": {
          "new": 1,
          "old": 2
        }
      }
    },
    "result": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "kind": "MULTIPLICITY",
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
    "result": "multiplicity_normalization",
    "step": "branch"
  }
]
```

---
