# TypeA_SELF_LINK_REJECTION

Cases: 20

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q1025801_2443864989`

| Field | Value |
|---|---|
| qid | Q1025801 |
| property | P131 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| group_key | ABOX::Q1025801::P131 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q498443"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "TallerTecleando3.0",
  "kind": "A_BOX",
  "new_value": [
    "Q498443"
  ],
  "new_value_descriptions_en": [
    "department in central Nicaragua with the capital Juigalpa, bordering Lake Nicaragua, known for agriculture and cattle ranching"
  ],
  "new_value_labels_en": [
    "Chontales Department"
  ],
  "old_value": [
    "Q498443",
    "Q1025801"
  ],
  "old_value_descriptions_en": [
    "department in central Nicaragua with the capital Juigalpa, bordering Lake Nicaragua, known for agriculture and cattle ranching",
    "municipality in Chontales Department, Nicaragua"
  ],
  "old_value_labels_en": [
    "Chontales Department",
    "Juigalpa"
  ],
  "revision_id": 2443864989,
  "value": [
    "Q498443"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q498443"
    ],
    "new_value": [
      "Q498443"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "Q1025801",
      "Q498443"
    ],
    "old_value": [
      "Q498443",
      "Q1025801"
    ],
    "removed_unique_values": [
      "Q1025801"
    ],
    "value_multiplicity_changes": {
      "Q1025801": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "department in central Nicaragua with the capital Juigalpa, bordering Lake Nicaragua, known for agriculture and cattle ranching"
  ],
  "value_labels_en": [
    "Chontales Department"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-20T12:58:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
  "report_revision_new": 2444506587,
  "report_revision_old": 2444049858,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q498443",
    "Q1025801"
  ],
  "value_descriptions_en": [
    "department in central Nicaragua with the capital Juigalpa, bordering Lake Nicaragua, known for agriculture and cattle ranching",
    "municipality in Chontales Department, Nicaragua"
  ],
  "value_labels_en": [
    "Chontales Department",
    "Juigalpa"
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
    "Q498443"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
    "label": "located in the administrative territorial entity"
  },
  "qid": {
    "description": "municipality in Chontales Department, Nicaragua",
    "label": "Juigalpa"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
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
        "Q498443": 1
      },
      "new_unique": [
        "Q498443"
      ],
      "new_values": [
        "Q498443"
      ],
      "new_values_raw": [
        "Q498443"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q1025801": 1,
        "Q498443": 1
      },
      "old_unique": [
        "Q1025801",
        "Q498443"
      ],
      "old_values": [
        "Q498443",
        "Q1025801"
      ],
      "old_values_raw": [
        "Q498443",
        "Q1025801"
      ],
      "removed_unique_values": [
        "Q1025801"
      ],
      "retained_unique_values": [
        "Q498443"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q1025801": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q121768149_2447349561`

| Field | Value |
|---|---|
| qid | Q121768149 |
| property | P921 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q121768149::P921 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q121768149"
  ],
  "old_value_descriptions_en": [
    "Spanish television series (2023-2025)"
  ],
  "old_value_labels_en": [
    "La Moderna"
  ],
  "revision_id": 2447349561,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "action": "DELETE",
    "added_unique_values": [
      "MISSING"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "MISSING"
    ],
    "new_value": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q121768149"
    ],
    "old_value": [
      "Q121768149"
    ],
    "removed_unique_values": [
      "Q121768149"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q121768149": {
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
  "report_fix_date": "2025-12-27T11:28:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
  "report_revision_new": 2447751779,
  "report_revision_old": 2447354743,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q121768149"
  ],
  "value_descriptions_en": [
    "Spanish television series (2023-2025)"
  ],
  "value_labels_en": [
    "La Moderna"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "primary topic of a work or act of communication",
    "label": "main subject"
  },
  "qid": {
    "description": "Spanish television series (2023-2025)",
    "label": "La Moderna"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "self_link",
      "removed_values": [
        "Q121768149"
      ],
      "report_type": "self link"
    },
    "result": "SELF_LINK_REJECTION",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q132710289_2447350068`

| Field | Value |
|---|---|
| qid | Q132710289 |
| property | P921 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | ABOX::Q132710289::P921 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q132710289"
  ],
  "old_value_descriptions_en": [
    "historical region in Italy"
  ],
  "old_value_labels_en": [
    "Mezzogiorno"
  ],
  "revision_id": 2447350068,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "action": "DELETE",
    "added_unique_values": [
      "MISSING"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "MISSING"
    ],
    "new_value": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q132710289"
    ],
    "old_value": [
      "Q132710289"
    ],
    "removed_unique_values": [
      "Q132710289"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q132710289": {
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
  "report_fix_date": "2025-12-27T11:28:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
  "report_revision_new": 2447751779,
  "report_revision_old": 2447354743,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q132710289"
  ],
  "value_descriptions_en": [
    "historical region in Italy"
  ],
  "value_labels_en": [
    "Mezzogiorno"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "primary topic of a work or act of communication",
    "label": "main subject"
  },
  "qid": {
    "description": "historical region in Italy",
    "label": "Mezzogiorno"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "self_link",
      "removed_values": [
        "Q132710289"
      ],
      "report_type": "self link"
    },
    "result": "SELF_LINK_REJECTION",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q135639210_2447350413`

| Field | Value |
|---|---|
| qid | Q135639210 |
| property | P921 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135639210::P921 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q135639210"
  ],
  "old_value_descriptions_en": [
    null
  ],
  "old_value_labels_en": [
    "Peloriarca"
  ],
  "revision_id": 2447350413,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "action": "DELETE",
    "added_unique_values": [
      "MISSING"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "MISSING"
    ],
    "new_value": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q135639210"
    ],
    "old_value": [
      "Q135639210"
    ],
    "removed_unique_values": [
      "Q135639210"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q135639210": {
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
  "report_fix_date": "2025-12-27T11:28:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
  "report_revision_new": 2447751779,
  "report_revision_old": 2447354743,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q135639210"
  ],
  "value_descriptions_en": [
    null
  ],
  "value_labels_en": [
    "Peloriarca"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "primary topic of a work or act of communication",
    "label": "main subject"
  },
  "qid": {
    "description": null,
    "label": "Peloriarca"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "self_link",
      "removed_values": [
        "Q135639210"
      ],
      "report_type": "self link"
    },
    "result": "SELF_LINK_REJECTION",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q18332808_2439816265`

| Field | Value |
|---|---|
| qid | Q18332808 |
| property | P31 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | tail |
| constraint_family | Q21510851 |
| group_key | ABOX::Q18332808::P31 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q7278", "Q176799"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Q7278",
    "Q176799"
  ],
  "new_value_descriptions_en": [
    "organization that seeks to influence government policy and actions and be elected to directly take part on government or legislation",
    "organization formed as part of an armed force"
  ],
  "new_value_labels_en": [
    "political party",
    "military unit"
  ],
  "old_value": [
    "Q7278",
    "Q176799",
    "Q18332808"
  ],
  "old_value_descriptions_en": [
    "organization that seeks to influence government policy and actions and be elected to directly take part on government or legislation",
    "organization formed as part of an armed force",
    "partai politik"
  ],
  "old_value_labels_en": [
    "political party",
    "military unit",
    "SS-Abschnitt"
  ],
  "revision_id": 2439816265,
  "value": [
    "Q7278",
    "Q176799"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "Q176799",
      "Q7278"
    ],
    "new_value": [
      "Q7278",
      "Q176799"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 3,
    "old_unique": [
      "Q176799",
      "Q18332808",
      "Q7278"
    ],
    "old_value": [
      "Q7278",
      "Q176799",
      "Q18332808"
    ],
    "removed_unique_values": [
      "Q18332808"
    ],
    "value_multiplicity_changes": {
      "Q18332808": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "organization that seeks to influence government policy and actions and be elected to directly take part on government or legislation",
    "organization formed as part of an armed force"
  ],
  "value_labels_en": [
    "political party",
    "military unit"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-10T15:45:17",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
  "report_revision_new": 2440479307,
  "report_revision_old": 2440179879,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q7278",
    "Q176799",
    "Q18332808"
  ],
  "value_descriptions_en": [
    "organization that seeks to influence government policy and actions and be elected to directly take part on government or legislation",
    "organization formed as part of an armed force",
    "partai politik"
  ],
  "value_labels_en": [
    "political party",
    "military unit",
    "SS-Abschnitt"
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
    "Q7278",
    "Q176799"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
    "label": "instance of"
  },
  "qid": {
    "description": "partai politik",
    "label": "SS-Abschnitt"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
        "Q176799": 1,
        "Q7278": 1
      },
      "new_unique": [
        "Q176799",
        "Q7278"
      ],
      "new_values": [
        "Q7278",
        "Q176799"
      ],
      "new_values_raw": [
        "Q7278",
        "Q176799"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q176799": 1,
        "Q18332808": 1,
        "Q7278": 1
      },
      "old_unique": [
        "Q176799",
        "Q18332808",
        "Q7278"
      ],
      "old_values": [
        "Q7278",
        "Q176799",
        "Q18332808"
      ],
      "old_values_raw": [
        "Q7278",
        "Q176799",
        "Q18332808"
      ],
      "removed_unique_values": [
        "Q18332808"
      ],
      "retained_unique_values": [
        "Q176799",
        "Q7278"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q18332808": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 006. `repair_Q2041168_2447343829`

| Field | Value |
|---|---|
| qid | Q2041168 |
| property | P921 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q2041168::P921 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q5775114", "Q3130", "Q64510815"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q5775114",
    "Q3130",
    "Q64510815"
  ],
  "new_value_descriptions_en": [
    "period of history of Australia from 1788 to 1850",
    "capital city of New South Wales, Australia",
    "first in an achievement or position in a social group"
  ],
  "new_value_labels_en": [
    "history of Australia (1788–1850)",
    "Sydney",
    "historic first"
  ],
  "old_value": [
    "Q5775114",
    "Q3130",
    "Q2041168",
    "Q64510815"
  ],
  "old_value_descriptions_en": [
    "period of history of Australia from 1788 to 1850",
    "capital city of New South Wales, Australia",
    "Australian 19th century newspaper",
    "first in an achievement or position in a social group"
  ],
  "old_value_labels_en": [
    "history of Australia (1788–1850)",
    "Sydney",
    "Sydney Gazette",
    "historic first"
  ],
  "revision_id": 2447343829,
  "value": [
    "Q5775114",
    "Q3130",
    "Q64510815"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "Q3130",
      "Q5775114",
      "Q64510815"
    ],
    "new_value": [
      "Q5775114",
      "Q3130",
      "Q64510815"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 4,
    "old_unique": [
      "Q2041168",
      "Q3130",
      "Q5775114",
      "Q64510815"
    ],
    "old_value": [
      "Q5775114",
      "Q3130",
      "Q2041168",
      "Q64510815"
    ],
    "removed_unique_values": [
      "Q2041168"
    ],
    "value_multiplicity_changes": {
      "Q2041168": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "period of history of Australia from 1788 to 1850",
    "capital city of New South Wales, Australia",
    "first in an achievement or position in a social group"
  ],
  "value_labels_en": [
    "history of Australia (1788–1850)",
    "Sydney",
    "historic first"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T11:28:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
  "report_revision_new": 2447751779,
  "report_revision_old": 2447354743,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q5775114",
    "Q3130",
    "Q2041168",
    "Q64510815"
  ],
  "value_descriptions_en": [
    "period of history of Australia from 1788 to 1850",
    "capital city of New South Wales, Australia",
    "Australian 19th century newspaper",
    "first in an achievement or position in a social group"
  ],
  "value_labels_en": [
    "history of Australia (1788–1850)",
    "Sydney",
    "Sydney Gazette",
    "historic first"
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
    "Q5775114",
    "Q3130",
    "Q64510815"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "primary topic of a work or act of communication",
    "label": "main subject"
  },
  "qid": {
    "description": "Australian 19th century newspaper",
    "label": "Sydney Gazette"
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
        "Q3130": 1,
        "Q5775114": 1,
        "Q64510815": 1
      },
      "new_unique": [
        "Q3130",
        "Q5775114",
        "Q64510815"
      ],
      "new_values": [
        "Q5775114",
        "Q3130",
        "Q64510815"
      ],
      "new_values_raw": [
        "Q5775114",
        "Q3130",
        "Q64510815"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q2041168": 1,
        "Q3130": 1,
        "Q5775114": 1,
        "Q64510815": 1
      },
      "old_unique": [
        "Q2041168",
        "Q3130",
        "Q5775114",
        "Q64510815"
      ],
      "old_values": [
        "Q5775114",
        "Q3130",
        "Q2041168",
        "Q64510815"
      ],
      "old_values_raw": [
        "Q5775114",
        "Q3130",
        "Q2041168",
        "Q64510815"
      ],
      "removed_unique_values": [
        "Q2041168"
      ],
      "retained_unique_values": [
        "Q3130",
        "Q5775114",
        "Q64510815"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q2041168": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 007. `repair_Q41973432_2446940773`

| Field | Value |
|---|---|
| qid | Q41973432 |
| property | P2860 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | mid |
| constraint_family | Q21510864 |
| group_key | ABOX::Q41973432::P2860 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q1460420", "Q39113842", "Q42048116"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q1460420",
    "Q39113842",
    "Q42048116"
  ],
  "new_value_descriptions_en": [
    "discipline of dealing with and avoiding both natural and man-made disasters, with the goal of reducing the harmful effects",
    "scientific article published on July 1996",
    "scientific article"
  ],
  "new_value_labels_en": [
    "emergency management",
    "Mortality of Kauai residents in the 12-month period following Hurricane Iniki",
    "Increased acute myocardial infarction mortality following the 1995 Great Hanshin-Awaji earthquake in Japan"
  ],
  "old_value": [
    "Q41973432",
    "Q1460420",
    "Q39113842",
    "Q42048116"
  ],
  "old_value_descriptions_en": [
    "scientific article published on October 4, 2012",
    "discipline of dealing with and avoiding both natural and man-made disasters, with the goal of reducing the harmful effects",
    "scientific article published on July 1996",
    "scientific article"
  ],
  "old_value_labels_en": [
    "Disaster Planning Considerations Involving the Geriatric Patient: Part II",
    "emergency management",
    "Mortality of Kauai residents in the 12-month period following Hurricane Iniki",
    "Increased acute myocardial infarction mortality following the 1995 Great Hanshin-Awaji earthquake in Japan"
  ],
  "revision_id": 2446940773,
  "value": [
    "Q1460420",
    "Q39113842",
    "Q42048116"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "Q1460420",
      "Q39113842",
      "Q42048116"
    ],
    "new_value": [
      "Q1460420",
      "Q39113842",
      "Q42048116"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 4,
    "old_unique": [
      "Q1460420",
      "Q39113842",
      "Q41973432",
      "Q42048116"
    ],
    "old_value": [
      "Q41973432",
      "Q1460420",
      "Q39113842",
      "Q42048116"
    ],
    "removed_unique_values": [
      "Q41973432"
    ],
    "value_multiplicity_changes": {
      "Q41973432": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "discipline of dealing with and avoiding both natural and man-made disasters, with the goal of reducing the harmful effects",
    "scientific article published on July 1996",
    "scientific article"
  ],
  "value_labels_en": [
    "emergency management",
    "Mortality of Kauai residents in the 12-month period following Hurricane Iniki",
    "Increased acute myocardial infarction mortality following the 1995 Great Hanshin-Awaji earthquake in Japan"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-26T10:07:30",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2860",
  "report_revision_new": 2447318381,
  "report_revision_old": 2447008974,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q41973432",
    "Q1460420",
    "Q39113842",
    "Q42048116"
  ],
  "value_descriptions_en": [
    "scientific article published on October 4, 2012",
    "discipline of dealing with and avoiding both natural and man-made disasters, with the goal of reducing the harmful effects",
    "scientific article published on July 1996",
    "scientific article"
  ],
  "value_labels_en": [
    "Disaster Planning Considerations Involving the Geriatric Patient: Part II",
    "emergency management",
    "Mortality of Kauai residents in the 12-month period following Hurricane Iniki",
    "Increased acute myocardial infarction mortality following the 1995 Great Hanshin-Awaji earthquake in Japan"
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
    "Q1460420",
    "Q39113842",
    "Q42048116"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "citation from one creative or scholarly work to another",
    "label": "cites work"
  },
  "qid": {
    "description": "scientific article published on October 4, 2012",
    "label": "Disaster Planning Considerations Involving the Geriatric Patient: Part II"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "Q1460420": 1,
        "Q39113842": 1,
        "Q42048116": 1
      },
      "new_unique": [
        "Q1460420",
        "Q39113842",
        "Q42048116"
      ],
      "new_values": [
        "Q1460420",
        "Q39113842",
        "Q42048116"
      ],
      "new_values_raw": [
        "Q1460420",
        "Q39113842",
        "Q42048116"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q1460420": 1,
        "Q39113842": 1,
        "Q41973432": 1,
        "Q42048116": 1
      },
      "old_unique": [
        "Q1460420",
        "Q39113842",
        "Q41973432",
        "Q42048116"
      ],
      "old_values": [
        "Q41973432",
        "Q1460420",
        "Q39113842",
        "Q42048116"
      ],
      "old_values_raw": [
        "Q41973432",
        "Q1460420",
        "Q39113842",
        "Q42048116"
      ],
      "removed_unique_values": [
        "Q41973432"
      ],
      "retained_unique_values": [
        "Q1460420",
        "Q39113842",
        "Q42048116"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q41973432": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 008. `repair_Q44146526_2447344755`

| Field | Value |
|---|---|
| qid | Q44146526 |
| property | P921 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | ABOX::Q44146526::P921 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q4932206"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q4932206"
  ],
  "new_value_descriptions_en": [
    "theoretical study of law, by philosophers and social scientists"
  ],
  "new_value_labels_en": [
    "jurisprudence"
  ],
  "old_value": [
    "Q44146526",
    "Q4932206"
  ],
  "old_value_descriptions_en": [
    "scientific article published on February 25, 1987",
    "theoretical study of law, by philosophers and social scientists"
  ],
  "old_value_labels_en": [
    "In re Eric B",
    "jurisprudence"
  ],
  "revision_id": 2447344755,
  "value": [
    "Q4932206"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q4932206"
    ],
    "new_value": [
      "Q4932206"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "Q44146526",
      "Q4932206"
    ],
    "old_value": [
      "Q44146526",
      "Q4932206"
    ],
    "removed_unique_values": [
      "Q44146526"
    ],
    "value_multiplicity_changes": {
      "Q44146526": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "theoretical study of law, by philosophers and social scientists"
  ],
  "value_labels_en": [
    "jurisprudence"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T11:28:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
  "report_revision_new": 2447751779,
  "report_revision_old": 2447354743,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q44146526",
    "Q4932206"
  ],
  "value_descriptions_en": [
    "scientific article published on February 25, 1987",
    "theoretical study of law, by philosophers and social scientists"
  ],
  "value_labels_en": [
    "In re Eric B",
    "jurisprudence"
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
    "Q4932206"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "primary topic of a work or act of communication",
    "label": "main subject"
  },
  "qid": {
    "description": "scientific article published on February 25, 1987",
    "label": "In re Eric B"
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
        "Q4932206": 1
      },
      "new_unique": [
        "Q4932206"
      ],
      "new_values": [
        "Q4932206"
      ],
      "new_values_raw": [
        "Q4932206"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q44146526": 1,
        "Q4932206": 1
      },
      "old_unique": [
        "Q44146526",
        "Q4932206"
      ],
      "old_values": [
        "Q44146526",
        "Q4932206"
      ],
      "old_values_raw": [
        "Q44146526",
        "Q4932206"
      ],
      "removed_unique_values": [
        "Q44146526"
      ],
      "retained_unique_values": [
        "Q4932206"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q44146526": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 009. `repair_Q45282055_2446941597`

| Field | Value |
|---|---|
| qid | Q45282055 |
| property | P2860 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | mid |
| constraint_family | Q21510864 |
| group_key | ABOX::Q45282055::P2860 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q31169204", "Q51338136", "Q60764157"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q31169204",
    "Q51338136",
    "Q60764157"
  ],
  "new_value_descriptions_en": [
    "scientific article",
    "scientific article published in August 2005",
    "scientific article published on 01 October 2005"
  ],
  "new_value_labels_en": [
    "Quantitative time- and frequency-domain analysis of the two-pulse COSY revamped by asymmetric Z-gradient echo detection NMR experiment: Theoretical and experimental aspects, time-zero data truncation artifacts, and radiation damping",
    "Theoretical formalism and experimental verification of line shapes of NMR intermolecular multiple-quantum coherence spectra",
    "Intermolecular multiple quantum coherences at high magnetic field: The nonlinear regime"
  ],
  "old_value": [
    "Q31169204",
    "Q51338136",
    "Q45282055",
    "Q60764157"
  ],
  "old_value_descriptions_en": [
    "scientific article",
    "scientific article published in August 2005",
    "scientific article published on September 21, 2010",
    "scientific article published on 01 October 2005"
  ],
  "old_value_labels_en": [
    "Quantitative time- and frequency-domain analysis of the two-pulse COSY revamped by asymmetric Z-gradient echo detection NMR experiment: Theoretical and experimental aspects, time-zero data truncation artifacts, and radiation damping",
    "Theoretical formalism and experimental verification of line shapes of NMR intermolecular multiple-quantum coherence spectra",
    "Erratum: \"Quantitative time- and frequency-domain analysis of the two-pulse COSY revamped by asymmetic Z-gradient echo detection NMR experiment: theoretical and experimental aspects, time-zero data truncation artifacts, and radiation damping\" [J. Ch",
    "Intermolecular multiple quantum coherences at high magnetic field: The nonlinear regime"
  ],
  "revision_id": 2446941597,
  "value": [
    "Q31169204",
    "Q51338136",
    "Q60764157"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "Q31169204",
      "Q51338136",
      "Q60764157"
    ],
    "new_value": [
      "Q31169204",
      "Q51338136",
      "Q60764157"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 4,
    "old_unique": [
      "Q31169204",
      "Q45282055",
      "Q51338136",
      "Q60764157"
    ],
    "old_value": [
      "Q31169204",
      "Q51338136",
      "Q45282055",
      "Q60764157"
    ],
    "removed_unique_values": [
      "Q45282055"
    ],
    "value_multiplicity_changes": {
      "Q45282055": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "scientific article",
    "scientific article published in August 2005",
    "scientific article published on 01 October 2005"
  ],
  "value_labels_en": [
    "Quantitative time- and frequency-domain analysis of the two-pulse COSY revamped by asymmetric Z-gradient echo detection NMR experiment: Theoretical and experimental aspects, time-zero data truncation artifacts, and radiation damping",
    "Theoretical formalism and experimental verification of line shapes of NMR intermolecular multiple-quantum coherence spectra",
    "Intermolecular multiple quantum coherences at high magnetic field: The nonlinear regime"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-26T10:07:30",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2860",
  "report_revision_new": 2447318381,
  "report_revision_old": 2447008974,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q31169204",
    "Q51338136",
    "Q45282055",
    "Q60764157"
  ],
  "value_descriptions_en": [
    "scientific article",
    "scientific article published in August 2005",
    "scientific article published on September 21, 2010",
    "scientific article published on 01 October 2005"
  ],
  "value_labels_en": [
    "Quantitative time- and frequency-domain analysis of the two-pulse COSY revamped by asymmetric Z-gradient echo detection NMR experiment: Theoretical and experimental aspects, time-zero data truncation artifacts, and radiation damping",
    "Theoretical formalism and experimental verification of line shapes of NMR intermolecular multiple-quantum coherence spectra",
    "Erratum: \"Quantitative time- and frequency-domain analysis of the two-pulse COSY revamped by asymmetic Z-gradient echo detection NMR experiment: theoretical and experimental aspects, time-zero data truncation artifacts, and radiation damping\" [J. Ch",
    "Intermolecular multiple quantum coherences at high magnetic field: The nonlinear regime"
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
    "Q31169204",
    "Q51338136",
    "Q60764157"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "citation from one creative or scholarly work to another",
    "label": "cites work"
  },
  "qid": {
    "description": "scientific article published on September 21, 2010",
    "label": "Erratum: \"Quantitative time- and frequency-domain analysis of the two-pulse COSY revamped by asymmetic Z-gradient echo detection NMR experiment: theoretical and experimental aspects, time-zero data truncation artifacts, and radiation damping\" [J. Ch"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "Q31169204": 1,
        "Q51338136": 1,
        "Q60764157": 1
      },
      "new_unique": [
        "Q31169204",
        "Q51338136",
        "Q60764157"
      ],
      "new_values": [
        "Q31169204",
        "Q51338136",
        "Q60764157"
      ],
      "new_values_raw": [
        "Q31169204",
        "Q51338136",
        "Q60764157"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q31169204": 1,
        "Q45282055": 1,
        "Q51338136": 1,
        "Q60764157": 1
      },
      "old_unique": [
        "Q31169204",
        "Q45282055",
        "Q51338136",
        "Q60764157"
      ],
      "old_values": [
        "Q31169204",
        "Q51338136",
        "Q45282055",
        "Q60764157"
      ],
      "old_values_raw": [
        "Q31169204",
        "Q51338136",
        "Q45282055",
        "Q60764157"
      ],
      "removed_unique_values": [
        "Q45282055"
      ],
      "retained_unique_values": [
        "Q31169204",
        "Q51338136",
        "Q60764157"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q45282055": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 010. `repair_Q46787778_2446941757`

| Field | Value |
|---|---|
| qid | Q46787778 |
| property | P2860 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | mid |
| constraint_family | Q21510864 |
| group_key | ABOX::Q46787778::P2860 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q38096425", "Q35809840", "Q43583135", "Q28265604", "Q34288599", "...(+6)"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q38096425",
    "Q35809840",
    "Q43583135",
    "Q28265604",
    "Q34288599",
    "Q73959236",
    "Q24203878",
    "Q40804478",
    "Q37135229",
    "Q37457924",
    "Q37330650"
  ],
  "new_value_descriptions_en": [
    "scientific article published on May 1, 2013",
    "scientific article published on July 2004",
    "scientific article published on 12 January 2006",
    "scientific article",
    "scientific article",
    "scientific article published on 01 June 2001",
    "scientific article",
    "scientific article published on January 1999",
    "scientific article published on 20 January 2009",
    "scientific article published on March 2009",
    "scientific article published on 21 July 2009"
  ],
  "new_value_labels_en": [
    "Rituximab in the treatment of non-Hodgkin's lymphoma – a critical evaluation of randomized controlled trials",
    "Treatment of Burkitt's/Burkitt-like lymphoma in adolescents and adults: a 20-year experience from the Norwegian Radium Hospital with the use of three successive regimens",
    "Early treatment-related mortality in adult autologous stem cell transplant recipients: a nation-wide survey of 1482 transplanted patients.",
    "The EBMT activity survey: 1990-2010",
    "Autologous bone marrow transplantation as compared with salvage chemotherapy in relapses of chemotherapy-sensitive non-Hodgkin's lymphoma",
    "High-dose therapy and autologous stem-cell transplantation versus conventional-dose consolidation/maintenance therapy as postremission therapy for adult patients with lymphoblastic lymphoma: results of a randomized trial of the European Group for Bl",
    "High-dose therapy with autologous stem cell transplantation versus chemotherapy or immuno-chemotherapy for follicular lymphoma in adults",
    "Late medical sequelae after therapy for supradiaphragmatic Hodgkin's disease",
    "Gonadal function in male patients after treatment for malignant lymphomas, with emphasis on chemotherapy",
    "Late effects of Hodgkin's disease and its treatment",
    "Valvular dysfunction and left ventricular changes in Hodgkin's lymphoma survivors. A longitudinal study"
  ],
  "old_value": [
    "Q46787778",
    "Q38096425",
    "Q35809840",
    "Q43583135",
    "Q28265604",
    "Q34288599",
    "Q73959236",
    "Q24203878",
    "Q40804478",
    "Q37135229",
    "Q37457924",
    "Q37330650"
  ],
  "old_value_descriptions_en": [
    "scientific article published on September 3, 2013",
    "scientific article published on May 1, 2013",
    "scientific article published on July 2004",
    "scientific article published on 12 January 2006",
    "scientific article",
    "scientific article",
    "scientific article published on 01 June 2001",
    "scientific article",
    "scientific article published on January 1999",
    "scientific article published on 20 January 2009",
    "scientific article published on March 2009",
    "scientific article published on 21 July 2009"
  ],
  "old_value_labels_en": [
    "High-dose therapy with autologous stem cell support for lymphoma in Norway 1987-2008",
    "Rituximab in the treatment of non-Hodgkin's lymphoma – a critical evaluation of randomized controlled trials",
    "Treatment of Burkitt's/Burkitt-like lymphoma in adolescents and adults: a 20-year experience from the Norwegian Radium Hospital with the use of three successive regimens",
    "Early treatment-related mortality in adult autologous stem cell transplant recipients: a nation-wide survey of 1482 transplanted patients.",
    "The EBMT activity survey: 1990-2010",
    "Autologous bone marrow transplantation as compared with salvage chemotherapy in relapses of chemotherapy-sensitive non-Hodgkin's lymphoma",
    "High-dose therapy and autologous stem-cell transplantation versus conventional-dose consolidation/maintenance therapy as postremission therapy for adult patients with lymphoblastic lymphoma: results of a randomized trial of the European Group for Bl",
    "High-dose therapy with autologous stem cell transplantation versus chemotherapy or immuno-chemotherapy for follicular lymphoma in adults",
    "Late medical sequelae after therapy for supradiaphragmatic Hodgkin's disease",
    "Gonadal function in male patients after treatment for malignant lymphomas, with emphasis on chemotherapy",
    "Late effects of Hodgkin's disease and its treatment",
    "Valvular dysfunction and left ventricular changes in Hodgkin's lymphoma survivors. A longitudinal study"
  ],
  "revision_id": 2446941757,
  "value": [
    "Q38096425",
    "Q35809840",
    "Q43583135",
    "Q28265604",
    "Q34288599",
    "Q73959236",
    "Q24203878",
    "Q40804478",
    "Q37135229",
    "Q37457924",
    "Q37330650"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 11,
    "new_unique": [
      "Q24203878",
      "Q28265604",
      "Q34288599",
      "Q35809840",
      "Q37135229",
      "Q37330650",
      "Q37457924",
      "Q38096425",
      "Q40804478",
      "Q43583135",
      "Q73959236"
    ],
    "new_value": [
      "Q38096425",
      "Q35809840",
      "Q43583135",
      "Q28265604",
      "Q34288599",
      "Q73959236",
      "Q24203878",
      "Q40804478",
      "Q37135229",
      "Q37457924",
      "Q37330650"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 12,
    "old_unique": [
      "Q24203878",
      "Q28265604",
      "Q34288599",
      "Q35809840",
      "Q37135229",
      "Q37330650",
      "Q37457924",
      "Q38096425",
      "Q40804478",
      "Q43583135",
      "Q46787778",
      "Q73959236"
    ],
    "old_value": [
      "Q46787778",
      "Q38096425",
      "Q35809840",
      "Q43583135",
      "Q28265604",
      "Q34288599",
      "Q73959236",
      "Q24203878",
      "Q40804478",
      "Q37135229",
      "Q37457924",
      "Q37330650"
    ],
    "removed_unique_values": [
      "Q46787778"
    ],
    "value_multiplicity_changes": {
      "Q46787778": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "scientific article published on May 1, 2013",
    "scientific article published on July 2004",
    "scientific article published on 12 January 2006",
    "scientific article",
    "scientific article",
    "scientific article published on 01 June 2001",
    "scientific article",
    "scientific article published on January 1999",
    "scientific article published on 20 January 2009",
    "scientific article published on March 2009",
    "scientific article published on 21 July 2009"
  ],
  "value_labels_en": [
    "Rituximab in the treatment of non-Hodgkin's lymphoma – a critical evaluation of randomized controlled trials",
    "Treatment of Burkitt's/Burkitt-like lymphoma in adolescents and adults: a 20-year experience from the Norwegian Radium Hospital with the use of three successive regimens",
    "Early treatment-related mortality in adult autologous stem cell transplant recipients: a nation-wide survey of 1482 transplanted patients.",
    "The EBMT activity survey: 1990-2010",
    "Autologous bone marrow transplantation as compared with salvage chemotherapy in relapses of chemotherapy-sensitive non-Hodgkin's lymphoma",
    "High-dose therapy and autologous stem-cell transplantation versus conventional-dose consolidation/maintenance therapy as postremission therapy for adult patients with lymphoblastic lymphoma: results of a randomized trial of the European Group for Bl",
    "High-dose therapy with autologous stem cell transplantation versus chemotherapy or immuno-chemotherapy for follicular lymphoma in adults",
    "Late medical sequelae after therapy for supradiaphragmatic Hodgkin's disease",
    "Gonadal function in male patients after treatment for malignant lymphomas, with emphasis on chemotherapy",
    "Late effects of Hodgkin's disease and its treatment",
    "Valvular dysfunction and left ventricular changes in Hodgkin's lymphoma survivors. A longitudinal study"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-26T10:07:30",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2860",
  "report_revision_new": 2447318381,
  "report_revision_old": 2447008974,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q46787778",
    "Q38096425",
    "Q35809840",
    "Q43583135",
    "Q28265604",
    "Q34288599",
    "Q73959236",
    "Q24203878",
    "Q40804478",
    "Q37135229",
    "Q37457924",
    "Q37330650"
  ],
  "value_descriptions_en": [
    "scientific article published on September 3, 2013",
    "scientific article published on May 1, 2013",
    "scientific article published on July 2004",
    "scientific article published on 12 January 2006",
    "scientific article",
    "scientific article",
    "scientific article published on 01 June 2001",
    "scientific article",
    "scientific article published on January 1999",
    "scientific article published on 20 January 2009",
    "scientific article published on March 2009",
    "scientific article published on 21 July 2009"
  ],
  "value_labels_en": [
    "High-dose therapy with autologous stem cell support for lymphoma in Norway 1987-2008",
    "Rituximab in the treatment of non-Hodgkin's lymphoma – a critical evaluation of randomized controlled trials",
    "Treatment of Burkitt's/Burkitt-like lymphoma in adolescents and adults: a 20-year experience from the Norwegian Radium Hospital with the use of three successive regimens",
    "Early treatment-related mortality in adult autologous stem cell transplant recipients: a nation-wide survey of 1482 transplanted patients.",
    "The EBMT activity survey: 1990-2010",
    "Autologous bone marrow transplantation as compared with salvage chemotherapy in relapses of chemotherapy-sensitive non-Hodgkin's lymphoma",
    "High-dose therapy and autologous stem-cell transplantation versus conventional-dose consolidation/maintenance therapy as postremission therapy for adult patients with lymphoblastic lymphoma: results of a randomized trial of the European Group for Bl",
    "High-dose therapy with autologous stem cell transplantation versus chemotherapy or immuno-chemotherapy for follicular lymphoma in adults",
    "Late medical sequelae after therapy for supradiaphragmatic Hodgkin's disease",
    "Gonadal function in male patients after treatment for malignant lymphomas, with emphasis on chemotherapy",
    "Late effects of Hodgkin's disease and its treatment",
    "Valvular dysfunction and left ventricular changes in Hodgkin's lymphoma survivors. A longitudinal study"
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
    "Q38096425",
    "Q35809840",
    "Q43583135",
    "Q28265604",
    "Q34288599",
    "Q73959236",
    "Q24203878",
    "Q40804478",
    "Q37135229",
    "Q37457924",
    "Q37330650"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "citation from one creative or scholarly work to another",
    "label": "cites work"
  },
  "qid": {
    "description": "scientific article published on September 3, 2013",
    "label": "High-dose therapy with autologous stem cell support for lymphoma in Norway 1987-2008"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "Q24203878": 1,
        "Q28265604": 1,
        "Q34288599": 1,
        "Q35809840": 1,
        "Q37135229": 1,
        "Q37330650": 1,
        "Q37457924": 1,
        "Q38096425": 1,
        "Q40804478": 1,
        "Q43583135": 1,
        "Q73959236": 1
      },
      "new_unique": [
        "Q24203878",
        "Q28265604",
        "Q34288599",
        "Q35809840",
        "Q37135229",
        "Q37330650",
        "Q37457924",
        "Q38096425",
        "Q40804478",
        "Q43583135",
        "Q73959236"
      ],
      "new_values": [
        "Q38096425",
        "Q35809840",
        "Q43583135",
        "Q28265604",
        "Q34288599",
        "Q73959236",
        "Q24203878",
        "Q40804478",
        "Q37135229",
        "Q37457924",
        "Q37330650"
      ],
      "new_values_raw": [
        "Q38096425",
        "Q35809840",
        "Q43583135",
        "Q28265604",
        "Q34288599",
        "Q73959236",
        "Q24203878",
        "Q40804478",
        "Q37135229",
        "Q37457924",
        "Q37330650"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q24203878": 1,
        "Q28265604": 1,
        "Q34288599": 1,
        "Q35809840": 1,
        "Q37135229": 1,
        "Q37330650": 1,
        "Q37457924": 1,
        "Q38096425": 1,
        "Q40804478": 1,
        "Q43583135": 1,
        "Q46787778": 1,
        "Q73959236": 1
      },
      "old_unique": [
        "Q24203878",
        "Q28265604",
        "Q34288599",
        "Q35809840",
        "Q37135229",
        "Q37330650",
        "Q37457924",
        "Q38096425",
        "Q40804478",
        "Q43583135",
        "Q46787778",
        "Q73959236"
      ],
      "old_values": [
        "Q46787778",
        "Q38096425",
        "Q35809840",
        "Q43583135",
        "Q28265604",
        "Q34288599",
        "Q73959236",
        "Q24203878",
        "Q40804478",
        "Q37135229",
        "Q37457924",
        "Q37330650"
      ],
      "old_values_raw": [
        "Q46787778",
        "Q38096425",
        "Q35809840",
        "Q43583135",
        "Q28265604",
        "Q34288599",
        "Q73959236",
        "Q24203878",
        "Q40804478",
        "Q37135229",
        "Q37457924",
        "Q37330650"
      ],
      "removed_unique_values": [
        "Q46787778"
      ],
      "retained_unique_values": [
        "Q24203878",
        "Q28265604",
        "Q34288599",
        "Q35809840",
        "Q37135229",
        "Q37330650",
        "Q37457924",
        "Q38096425",
        "Q40804478",
        "Q43583135",
        "Q73959236"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q46787778": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 011. `repair_Q596688_2443865072`

| Field | Value |
|---|---|
| qid | Q596688 |
| property | P131 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| group_key | ABOX::Q596688::P131 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q728099"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "TallerTecleando3.0",
  "kind": "A_BOX",
  "new_value": [
    "Q728099"
  ],
  "new_value_descriptions_en": [
    "department in central Nicaragua"
  ],
  "new_value_labels_en": [
    "Matagalpa Department"
  ],
  "old_value": [
    "Q728099",
    "Q596688"
  ],
  "old_value_descriptions_en": [
    "department in central Nicaragua",
    "municipality in Matagalpa Department, Nicaragua"
  ],
  "old_value_labels_en": [
    "Matagalpa Department",
    "Matagalpa"
  ],
  "revision_id": 2443865072,
  "value": [
    "Q728099"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q728099"
    ],
    "new_value": [
      "Q728099"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "Q596688",
      "Q728099"
    ],
    "old_value": [
      "Q728099",
      "Q596688"
    ],
    "removed_unique_values": [
      "Q596688"
    ],
    "value_multiplicity_changes": {
      "Q596688": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "department in central Nicaragua"
  ],
  "value_labels_en": [
    "Matagalpa Department"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-20T12:58:47",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
  "report_revision_new": 2444506587,
  "report_revision_old": 2444049858,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q728099",
    "Q596688"
  ],
  "value_descriptions_en": [
    "department in central Nicaragua",
    "municipality in Matagalpa Department, Nicaragua"
  ],
  "value_labels_en": [
    "Matagalpa Department",
    "Matagalpa"
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
    "Q728099"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
    "label": "located in the administrative territorial entity"
  },
  "qid": {
    "description": "municipality in Matagalpa Department, Nicaragua",
    "label": "Matagalpa"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
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
        "Q728099": 1
      },
      "new_unique": [
        "Q728099"
      ],
      "new_values": [
        "Q728099"
      ],
      "new_values_raw": [
        "Q728099"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q596688": 1,
        "Q728099": 1
      },
      "old_unique": [
        "Q596688",
        "Q728099"
      ],
      "old_values": [
        "Q728099",
        "Q596688"
      ],
      "old_values_raw": [
        "Q728099",
        "Q596688"
      ],
      "removed_unique_values": [
        "Q596688"
      ],
      "retained_unique_values": [
        "Q728099"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q596688": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 012. `repair_Q639669_2441128807`

| Field | Value |
|---|---|
| qid | Q639669 |
| property | P279 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| group_key | ABOX::Q639669::P279 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q483501", "Q17307272"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Q483501",
    "Q17307272"
  ],
  "new_value_descriptions_en": [
    "person who engages in any form of artistic creation or practice",
    "profession"
  ],
  "new_value_labels_en": [
    "artist",
    "circus performer"
  ],
  "old_value": [
    "Q483501",
    "Q17307272",
    "Q639669"
  ],
  "old_value_descriptions_en": [
    "person who engages in any form of artistic creation or practice",
    "profession",
    "person who composes, conducts or performs music"
  ],
  "old_value_labels_en": [
    "artist",
    "circus performer",
    "musician"
  ],
  "revision_id": 2441128807,
  "value": [
    "Q483501",
    "Q17307272"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "Q17307272",
      "Q483501"
    ],
    "new_value": [
      "Q483501",
      "Q17307272"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 3,
    "old_unique": [
      "Q17307272",
      "Q483501",
      "Q639669"
    ],
    "old_value": [
      "Q483501",
      "Q17307272",
      "Q639669"
    ],
    "removed_unique_values": [
      "Q639669"
    ],
    "value_multiplicity_changes": {
      "Q639669": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "person who engages in any form of artistic creation or practice",
    "profession"
  ],
  "value_labels_en": [
    "artist",
    "circus performer"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-13T11:16:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P279",
  "report_revision_new": 2441794953,
  "report_revision_old": 2441232862,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q483501",
    "Q17307272",
    "Q639669"
  ],
  "value_descriptions_en": [
    "person who engages in any form of artistic creation or practice",
    "profession",
    "person who composes, conducts or performs music"
  ],
  "value_labels_en": [
    "artist",
    "circus performer",
    "musician"
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
    "Q483501",
    "Q17307272"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "this item is a subclass (subset) of that item; ALL instances of this item are instances of that item; different from P31 (instance of), e.g.: volcano is a subclass of mountain; Everest is an instance of mountain",
    "label": "subclass of"
  },
  "qid": {
    "description": "person who composes, conducts or performs music",
    "label": "musician"
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
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
        "Q17307272": 1,
        "Q483501": 1
      },
      "new_unique": [
        "Q17307272",
        "Q483501"
      ],
      "new_values": [
        "Q483501",
        "Q17307272"
      ],
      "new_values_raw": [
        "Q483501",
        "Q17307272"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q17307272": 1,
        "Q483501": 1,
        "Q639669": 1
      },
      "old_unique": [
        "Q17307272",
        "Q483501",
        "Q639669"
      ],
      "old_values": [
        "Q483501",
        "Q17307272",
        "Q639669"
      ],
      "old_values_raw": [
        "Q483501",
        "Q17307272",
        "Q639669"
      ],
      "removed_unique_values": [
        "Q639669"
      ],
      "retained_unique_values": [
        "Q17307272",
        "Q483501"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q639669": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 013. `repair_Q661122_2441114026`

| Field | Value |
|---|---|
| qid | Q661122 |
| property | P937 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q661122::P937 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Samoasambia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q661122"
  ],
  "old_value_descriptions_en": [
    "Italian painter (1859-1932)"
  ],
  "old_value_labels_en": [
    "Emilio Longoni"
  ],
  "revision_id": 2441114026,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "action": "DELETE",
    "added_unique_values": [
      "MISSING"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "MISSING"
    ],
    "new_value": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q661122"
    ],
    "old_value": [
      "Q661122"
    ],
    "removed_unique_values": [
      "Q661122"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q661122": {
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
  "report_fix_date": "2025-12-13T09:49:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P937",
  "report_revision_new": 2441745882,
  "report_revision_old": 2441170808,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "report_violation_types": [
    "Self link",
    "Value type Q|17334923, Q|3895768, Q|1229765, Q|56061, Q|2221906, Q|628858, Q|41176, Q|486972, Q|82794, Q|43229, Q|2385804"
  ],
  "value": [
    "Q661122"
  ],
  "value_descriptions_en": [
    "Italian painter (1859-1932)"
  ],
  "value_labels_en": [
    "Emilio Longoni"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "location where persons or organisations were actively participating in employment, business or other work",
    "label": "work location"
  },
  "qid": {
    "description": "Italian painter (1859-1932)",
    "label": "Emilio Longoni"
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
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "self_link",
      "removed_values": [
        "Q661122"
      ],
      "report_type": "self link"
    },
    "result": "SELF_LINK_REJECTION",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 014. `repair_Q71821029_2447345727`

| Field | Value |
|---|---|
| qid | Q71821029 |
| property | P921 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | ABOX::Q71821029::P921 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q45051448"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q45051448"
  ],
  "new_value_descriptions_en": [
    "human infant born at less than 37 weeks after conception"
  ],
  "new_value_labels_en": [
    "preterm infant"
  ],
  "old_value": [
    "Q71821029",
    "Q45051448"
  ],
  "old_value_descriptions_en": [
    "scientific article published on 01 January 1995",
    "human infant born at less than 37 weeks after conception"
  ],
  "old_value_labels_en": [
    "Nguyen v. Sacred Heart Medical Center",
    "preterm infant"
  ],
  "revision_id": 2447345727,
  "value": [
    "Q45051448"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q45051448"
    ],
    "new_value": [
      "Q45051448"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "Q45051448",
      "Q71821029"
    ],
    "old_value": [
      "Q71821029",
      "Q45051448"
    ],
    "removed_unique_values": [
      "Q71821029"
    ],
    "value_multiplicity_changes": {
      "Q71821029": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "human infant born at less than 37 weeks after conception"
  ],
  "value_labels_en": [
    "preterm infant"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T11:28:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
  "report_revision_new": 2447751779,
  "report_revision_old": 2447354743,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q71821029",
    "Q45051448"
  ],
  "value_descriptions_en": [
    "scientific article published on 01 January 1995",
    "human infant born at less than 37 weeks after conception"
  ],
  "value_labels_en": [
    "Nguyen v. Sacred Heart Medical Center",
    "preterm infant"
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
    "Q45051448"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "primary topic of a work or act of communication",
    "label": "main subject"
  },
  "qid": {
    "description": "scientific article published on 01 January 1995",
    "label": "Nguyen v. Sacred Heart Medical Center"
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
        "Q45051448": 1
      },
      "new_unique": [
        "Q45051448"
      ],
      "new_values": [
        "Q45051448"
      ],
      "new_values_raw": [
        "Q45051448"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q45051448": 1,
        "Q71821029": 1
      },
      "old_unique": [
        "Q45051448",
        "Q71821029"
      ],
      "old_values": [
        "Q71821029",
        "Q45051448"
      ],
      "old_values_raw": [
        "Q71821029",
        "Q45051448"
      ],
      "removed_unique_values": [
        "Q71821029"
      ],
      "retained_unique_values": [
        "Q45051448"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q71821029": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 015. `repair_Q72049593_2446943599`

| Field | Value |
|---|---|
| qid | Q72049593 |
| property | P2860 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | mid |
| constraint_family | Q21510864 |
| group_key | ABOX::Q72049593::P2860 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q59019016", "Q72922814"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q59019016",
    "Q72922814"
  ],
  "new_value_descriptions_en": [
    "scientific article published in Nature",
    "scientific article published on 01 November 1966"
  ],
  "new_value_labels_en": [
    "Direct Proportionality of Urinary Excretion Rate and Serum Level of Tetracycline in Human Subjects",
    "Apparent renal tubular secretion of riboflavin in man"
  ],
  "old_value": [
    "Q72049593",
    "Q59019016",
    "Q72922814"
  ],
  "old_value_descriptions_en": [
    "scientific article published on 01 January 1968",
    "scientific article published in Nature",
    "scientific article published on 01 November 1966"
  ],
  "old_value_labels_en": [
    "Concept of a Volume of Distribution and Possible Errors in Evaluation of This Parameter",
    "Direct Proportionality of Urinary Excretion Rate and Serum Level of Tetracycline in Human Subjects",
    "Apparent renal tubular secretion of riboflavin in man"
  ],
  "revision_id": 2446943599,
  "value": [
    "Q59019016",
    "Q72922814"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "Q59019016",
      "Q72922814"
    ],
    "new_value": [
      "Q59019016",
      "Q72922814"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 3,
    "old_unique": [
      "Q59019016",
      "Q72049593",
      "Q72922814"
    ],
    "old_value": [
      "Q72049593",
      "Q59019016",
      "Q72922814"
    ],
    "removed_unique_values": [
      "Q72049593"
    ],
    "value_multiplicity_changes": {
      "Q72049593": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "scientific article published in Nature",
    "scientific article published on 01 November 1966"
  ],
  "value_labels_en": [
    "Direct Proportionality of Urinary Excretion Rate and Serum Level of Tetracycline in Human Subjects",
    "Apparent renal tubular secretion of riboflavin in man"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-26T10:07:30",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2860",
  "report_revision_new": 2447318381,
  "report_revision_old": 2447008974,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q72049593",
    "Q59019016",
    "Q72922814"
  ],
  "value_descriptions_en": [
    "scientific article published on 01 January 1968",
    "scientific article published in Nature",
    "scientific article published on 01 November 1966"
  ],
  "value_labels_en": [
    "Concept of a Volume of Distribution and Possible Errors in Evaluation of This Parameter",
    "Direct Proportionality of Urinary Excretion Rate and Serum Level of Tetracycline in Human Subjects",
    "Apparent renal tubular secretion of riboflavin in man"
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
    "Q59019016",
    "Q72922814"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "citation from one creative or scholarly work to another",
    "label": "cites work"
  },
  "qid": {
    "description": "scientific article published on 01 January 1968",
    "label": "Concept of a Volume of Distribution and Possible Errors in Evaluation of This Parameter"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "Q59019016": 1,
        "Q72922814": 1
      },
      "new_unique": [
        "Q59019016",
        "Q72922814"
      ],
      "new_values": [
        "Q59019016",
        "Q72922814"
      ],
      "new_values_raw": [
        "Q59019016",
        "Q72922814"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q59019016": 1,
        "Q72049593": 1,
        "Q72922814": 1
      },
      "old_unique": [
        "Q59019016",
        "Q72049593",
        "Q72922814"
      ],
      "old_values": [
        "Q72049593",
        "Q59019016",
        "Q72922814"
      ],
      "old_values_raw": [
        "Q72049593",
        "Q59019016",
        "Q72922814"
      ],
      "removed_unique_values": [
        "Q72049593"
      ],
      "retained_unique_values": [
        "Q59019016",
        "Q72922814"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q72049593": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 016. `repair_Q731126_2441779392`

| Field | Value |
|---|---|
| qid | Q731126 |
| property | P6379 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q731126::P6379 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Mondo",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q731126"
  ],
  "old_value_descriptions_en": [
    "art museum in Los Angeles, California"
  ],
  "old_value_labels_en": [
    "J. Paul Getty Museum"
  ],
  "revision_id": 2441779392,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "action": "DELETE",
    "added_unique_values": [
      "MISSING"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "MISSING"
    ],
    "new_value": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q731126"
    ],
    "old_value": [
      "Q731126"
    ],
    "removed_unique_values": [
      "Q731126"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q731126": {
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
  "report_fix_date": "2025-12-14T07:32:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6379",
  "report_revision_new": 2442225437,
  "report_revision_old": 2441674069,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q731126"
  ],
  "value_descriptions_en": [
    "art museum in Los Angeles, California"
  ],
  "value_labels_en": [
    "J. Paul Getty Museum"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "matched": null,
  "matches": null,
  "needed": null,
  "sources_used": null,
  "synthetic_pre_repair": {},
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "collection that has works of this person or organisation (use archives at [P485] for archives)",
    "label": "has works in the collection"
  },
  "qid": {
    "description": "art museum in Los Angeles, California",
    "label": "J. Paul Getty Museum"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  }
]
```

### Decision Trace

```json
[
  {
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "self_link",
      "removed_values": [
        "Q731126"
      ],
      "report_type": "self link"
    },
    "result": "SELF_LINK_REJECTION",
    "step": "delete_classification"
  },
  {
    "result": null,
    "step": "rule_deterministic"
  },
  {
    "result": false,
    "step": "local_availability"
  },
  {
    "result": false,
    "step": "fallback_external"
  },
  {
    "result": "delete_refined",
    "step": "branch"
  }
]
```

---

## 017. `repair_Q74775001_2447346252`

| Field | Value |
|---|---|
| qid | Q74775001 |
| property | P921 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | ABOX::Q74775001::P921 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q1216998", "Q4932206"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q1216998",
    "Q4932206"
  ],
  "new_value_descriptions_en": [
    "abnormal mass of tissue as a result of abnormal growth or division of cells",
    "theoretical study of law, by philosophers and social scientists"
  ],
  "new_value_labels_en": [
    "neoplasm",
    "jurisprudence"
  ],
  "old_value": [
    "Q74775001",
    "Q1216998",
    "Q4932206"
  ],
  "old_value_descriptions_en": [
    "scientific article published on 01 April 1991",
    "abnormal mass of tissue as a result of abnormal growth or division of cells",
    "theoretical study of law, by philosophers and social scientists"
  ],
  "old_value_labels_en": [
    "Newmark v. Williams",
    "neoplasm",
    "jurisprudence"
  ],
  "revision_id": 2447346252,
  "value": [
    "Q1216998",
    "Q4932206"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "Q1216998",
      "Q4932206"
    ],
    "new_value": [
      "Q1216998",
      "Q4932206"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 3,
    "old_unique": [
      "Q1216998",
      "Q4932206",
      "Q74775001"
    ],
    "old_value": [
      "Q74775001",
      "Q1216998",
      "Q4932206"
    ],
    "removed_unique_values": [
      "Q74775001"
    ],
    "value_multiplicity_changes": {
      "Q74775001": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "abnormal mass of tissue as a result of abnormal growth or division of cells",
    "theoretical study of law, by philosophers and social scientists"
  ],
  "value_labels_en": [
    "neoplasm",
    "jurisprudence"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T11:28:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
  "report_revision_new": 2447751779,
  "report_revision_old": 2447354743,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q74775001",
    "Q1216998",
    "Q4932206"
  ],
  "value_descriptions_en": [
    "scientific article published on 01 April 1991",
    "abnormal mass of tissue as a result of abnormal growth or division of cells",
    "theoretical study of law, by philosophers and social scientists"
  ],
  "value_labels_en": [
    "Newmark v. Williams",
    "neoplasm",
    "jurisprudence"
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
    "Q1216998",
    "Q4932206"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "primary topic of a work or act of communication",
    "label": "main subject"
  },
  "qid": {
    "description": "scientific article published on 01 April 1991",
    "label": "Newmark v. Williams"
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
        "Q1216998": 1,
        "Q4932206": 1
      },
      "new_unique": [
        "Q1216998",
        "Q4932206"
      ],
      "new_values": [
        "Q1216998",
        "Q4932206"
      ],
      "new_values_raw": [
        "Q1216998",
        "Q4932206"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q1216998": 1,
        "Q4932206": 1,
        "Q74775001": 1
      },
      "old_unique": [
        "Q1216998",
        "Q4932206",
        "Q74775001"
      ],
      "old_values": [
        "Q74775001",
        "Q1216998",
        "Q4932206"
      ],
      "old_values_raw": [
        "Q74775001",
        "Q1216998",
        "Q4932206"
      ],
      "removed_unique_values": [
        "Q74775001"
      ],
      "retained_unique_values": [
        "Q1216998",
        "Q4932206"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q74775001": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 018. `repair_Q74775133_2447346381`

| Field | Value |
|---|---|
| qid | Q74775133 |
| property | P921 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | ABOX::Q74775133::P921 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q44275059", "Q54973000", "Q12128", "Q4932206", "Q59283400", "...(+1)"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q44275059",
    "Q54973000",
    "Q12128",
    "Q4932206",
    "Q59283400",
    "Q70442066"
  ],
  "new_value_descriptions_en": [
    "scientific article published on December 1, 1971",
    "development of neutralizing antibodies in individuals who have been exposed to the human immunodeficiency virus",
    "branch of medicine dealing with oral health and teeth",
    "theoretical study of law, by philosophers and social scientists",
    "persons working in the field of health",
    "refusal of a health professional to initiate or continue treatment of a patient or group of patients"
  ],
  "new_value_labels_en": [
    "Health care and public health",
    "HIV seropositivity",
    "dentistry",
    "jurisprudence",
    "health personnel",
    "refusal to treat"
  ],
  "old_value": [
    "Q44275059",
    "Q74775133",
    "Q54973000",
    "Q12128",
    "Q4932206",
    "Q59283400",
    "Q70442066"
  ],
  "old_value_descriptions_en": [
    "scientific article published on December 1, 1971",
    "scientific article published on March 23, 1995",
    "development of neutralizing antibodies in individuals who have been exposed to the human immunodeficiency virus",
    "branch of medicine dealing with oral health and teeth",
    "theoretical study of law, by philosophers and social scientists",
    "persons working in the field of health",
    "refusal of a health professional to initiate or continue treatment of a patient or group of patients"
  ],
  "old_value_labels_en": [
    "Health care and public health",
    "United States v. Morvant",
    "HIV seropositivity",
    "dentistry",
    "jurisprudence",
    "health personnel",
    "refusal to treat"
  ],
  "revision_id": 2447346381,
  "value": [
    "Q44275059",
    "Q54973000",
    "Q12128",
    "Q4932206",
    "Q59283400",
    "Q70442066"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 6,
    "new_unique": [
      "Q12128",
      "Q44275059",
      "Q4932206",
      "Q54973000",
      "Q59283400",
      "Q70442066"
    ],
    "new_value": [
      "Q44275059",
      "Q54973000",
      "Q12128",
      "Q4932206",
      "Q59283400",
      "Q70442066"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 7,
    "old_unique": [
      "Q12128",
      "Q44275059",
      "Q4932206",
      "Q54973000",
      "Q59283400",
      "Q70442066",
      "Q74775133"
    ],
    "old_value": [
      "Q44275059",
      "Q74775133",
      "Q54973000",
      "Q12128",
      "Q4932206",
      "Q59283400",
      "Q70442066"
    ],
    "removed_unique_values": [
      "Q74775133"
    ],
    "value_multiplicity_changes": {
      "Q74775133": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "scientific article published on December 1, 1971",
    "development of neutralizing antibodies in individuals who have been exposed to the human immunodeficiency virus",
    "branch of medicine dealing with oral health and teeth",
    "theoretical study of law, by philosophers and social scientists",
    "persons working in the field of health",
    "refusal of a health professional to initiate or continue treatment of a patient or group of patients"
  ],
  "value_labels_en": [
    "Health care and public health",
    "HIV seropositivity",
    "dentistry",
    "jurisprudence",
    "health personnel",
    "refusal to treat"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T11:28:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
  "report_revision_new": 2447751779,
  "report_revision_old": 2447354743,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q44275059",
    "Q74775133",
    "Q54973000",
    "Q12128",
    "Q4932206",
    "Q59283400",
    "Q70442066"
  ],
  "value_descriptions_en": [
    "scientific article published on December 1, 1971",
    "scientific article published on March 23, 1995",
    "development of neutralizing antibodies in individuals who have been exposed to the human immunodeficiency virus",
    "branch of medicine dealing with oral health and teeth",
    "theoretical study of law, by philosophers and social scientists",
    "persons working in the field of health",
    "refusal of a health professional to initiate or continue treatment of a patient or group of patients"
  ],
  "value_labels_en": [
    "Health care and public health",
    "United States v. Morvant",
    "HIV seropositivity",
    "dentistry",
    "jurisprudence",
    "health personnel",
    "refusal to treat"
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
    "Q44275059",
    "Q54973000",
    "Q12128",
    "Q4932206",
    "Q59283400",
    "Q70442066"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "primary topic of a work or act of communication",
    "label": "main subject"
  },
  "qid": {
    "description": "scientific article published on March 23, 1995",
    "label": "United States v. Morvant"
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
        "Q12128": 1,
        "Q44275059": 1,
        "Q4932206": 1,
        "Q54973000": 1,
        "Q59283400": 1,
        "Q70442066": 1
      },
      "new_unique": [
        "Q12128",
        "Q44275059",
        "Q4932206",
        "Q54973000",
        "Q59283400",
        "Q70442066"
      ],
      "new_values": [
        "Q44275059",
        "Q54973000",
        "Q12128",
        "Q4932206",
        "Q59283400",
        "Q70442066"
      ],
      "new_values_raw": [
        "Q44275059",
        "Q54973000",
        "Q12128",
        "Q4932206",
        "Q59283400",
        "Q70442066"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q12128": 1,
        "Q44275059": 1,
        "Q4932206": 1,
        "Q54973000": 1,
        "Q59283400": 1,
        "Q70442066": 1,
        "Q74775133": 1
      },
      "old_unique": [
        "Q12128",
        "Q44275059",
        "Q4932206",
        "Q54973000",
        "Q59283400",
        "Q70442066",
        "Q74775133"
      ],
      "old_values": [
        "Q44275059",
        "Q74775133",
        "Q54973000",
        "Q12128",
        "Q4932206",
        "Q59283400",
        "Q70442066"
      ],
      "old_values_raw": [
        "Q44275059",
        "Q74775133",
        "Q54973000",
        "Q12128",
        "Q4932206",
        "Q59283400",
        "Q70442066"
      ],
      "removed_unique_values": [
        "Q74775133"
      ],
      "retained_unique_values": [
        "Q12128",
        "Q44275059",
        "Q4932206",
        "Q54973000",
        "Q59283400",
        "Q70442066"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q74775133": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 019. `repair_Q74775312_2447346626`

| Field | Value |
|---|---|
| qid | Q74775312 |
| property | P921 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | ABOX::Q74775312::P921 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q67178686", "Q5665173", "Q8452", "Q67120553", "Q4932206"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q67178686",
    "Q5665173",
    "Q8452",
    "Q67120553",
    "Q4932206"
  ],
  "new_value_descriptions_en": [
    "organized services to provide health care to women, beyond maternal care services",
    "1980 United States Supreme Court case",
    "intentional ending of a pregnancy",
    "control exerted normally through written codes by organizations of society, such as established institutions and the law",
    "theoretical study of law, by philosophers and social scientists"
  ],
  "new_value_labels_en": [
    "women's health services",
    "Harris v. McRae",
    "abortion",
    "formal social control",
    "jurisprudence"
  ],
  "old_value": [
    "Q67178686",
    "Q5665173",
    "Q74775312",
    "Q8452",
    "Q67120553",
    "Q4932206"
  ],
  "old_value_descriptions_en": [
    "organized services to provide health care to women, beyond maternal care services",
    "1980 United States Supreme Court case",
    "scientific article published on 01 May 1981",
    "intentional ending of a pregnancy",
    "control exerted normally through written codes by organizations of society, such as established institutions and the law",
    "theoretical study of law, by philosophers and social scientists"
  ],
  "old_value_labels_en": [
    "women's health services",
    "Harris v. McRae",
    "Women's Health Services v. Maher",
    "abortion",
    "formal social control",
    "jurisprudence"
  ],
  "revision_id": 2447346626,
  "value": [
    "Q67178686",
    "Q5665173",
    "Q8452",
    "Q67120553",
    "Q4932206"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 5,
    "new_unique": [
      "Q4932206",
      "Q5665173",
      "Q67120553",
      "Q67178686",
      "Q8452"
    ],
    "new_value": [
      "Q67178686",
      "Q5665173",
      "Q8452",
      "Q67120553",
      "Q4932206"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 6,
    "old_unique": [
      "Q4932206",
      "Q5665173",
      "Q67120553",
      "Q67178686",
      "Q74775312",
      "Q8452"
    ],
    "old_value": [
      "Q67178686",
      "Q5665173",
      "Q74775312",
      "Q8452",
      "Q67120553",
      "Q4932206"
    ],
    "removed_unique_values": [
      "Q74775312"
    ],
    "value_multiplicity_changes": {
      "Q74775312": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "organized services to provide health care to women, beyond maternal care services",
    "1980 United States Supreme Court case",
    "intentional ending of a pregnancy",
    "control exerted normally through written codes by organizations of society, such as established institutions and the law",
    "theoretical study of law, by philosophers and social scientists"
  ],
  "value_labels_en": [
    "women's health services",
    "Harris v. McRae",
    "abortion",
    "formal social control",
    "jurisprudence"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T11:28:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
  "report_revision_new": 2447751779,
  "report_revision_old": 2447354743,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q67178686",
    "Q5665173",
    "Q74775312",
    "Q8452",
    "Q67120553",
    "Q4932206"
  ],
  "value_descriptions_en": [
    "organized services to provide health care to women, beyond maternal care services",
    "1980 United States Supreme Court case",
    "scientific article published on 01 May 1981",
    "intentional ending of a pregnancy",
    "control exerted normally through written codes by organizations of society, such as established institutions and the law",
    "theoretical study of law, by philosophers and social scientists"
  ],
  "value_labels_en": [
    "women's health services",
    "Harris v. McRae",
    "Women's Health Services v. Maher",
    "abortion",
    "formal social control",
    "jurisprudence"
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
    "Q67178686",
    "Q5665173",
    "Q8452",
    "Q67120553",
    "Q4932206"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "primary topic of a work or act of communication",
    "label": "main subject"
  },
  "qid": {
    "description": "scientific article published on 01 May 1981",
    "label": "Women's Health Services v. Maher"
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
        "Q4932206": 1,
        "Q5665173": 1,
        "Q67120553": 1,
        "Q67178686": 1,
        "Q8452": 1
      },
      "new_unique": [
        "Q4932206",
        "Q5665173",
        "Q67120553",
        "Q67178686",
        "Q8452"
      ],
      "new_values": [
        "Q67178686",
        "Q5665173",
        "Q8452",
        "Q67120553",
        "Q4932206"
      ],
      "new_values_raw": [
        "Q67178686",
        "Q5665173",
        "Q8452",
        "Q67120553",
        "Q4932206"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q4932206": 1,
        "Q5665173": 1,
        "Q67120553": 1,
        "Q67178686": 1,
        "Q74775312": 1,
        "Q8452": 1
      },
      "old_unique": [
        "Q4932206",
        "Q5665173",
        "Q67120553",
        "Q67178686",
        "Q74775312",
        "Q8452"
      ],
      "old_values": [
        "Q67178686",
        "Q5665173",
        "Q74775312",
        "Q8452",
        "Q67120553",
        "Q4932206"
      ],
      "old_values_raw": [
        "Q67178686",
        "Q5665173",
        "Q74775312",
        "Q8452",
        "Q67120553",
        "Q4932206"
      ],
      "removed_unique_values": [
        "Q74775312"
      ],
      "retained_unique_values": [
        "Q4932206",
        "Q5665173",
        "Q67120553",
        "Q67178686",
        "Q8452"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q74775312": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---

## 020. `repair_Q78529461_2446944294`

| Field | Value |
|---|---|
| qid | Q78529461 |
| property | P2860 |
| track | A_BOX |
| class / subtype / confidence | TypeA / SELF_LINK_REJECTION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_self_link_rejection |
| popularity_bucket | mid |
| constraint_family | Q21510864 |
| group_key | ABOX::Q78529461::P2860 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q49182336", "Q74634188", "Q83292930", "Q74807075", "Q78882049", "...(+10)"] |
| decision_branch | self_link_rejection |
| rationale | Subset repair removes the focus entity from a self-link violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "RVA2869",
  "kind": "A_BOX",
  "new_value": [
    "Q49182336",
    "Q74634188",
    "Q83292930",
    "Q74807075",
    "Q78882049",
    "Q76248372",
    "Q39109486",
    "Q78289407",
    "Q79409952",
    "Q73645037",
    "Q24652974",
    "Q39246143",
    "Q60271532",
    "Q63874550",
    "Q79272812"
  ],
  "new_value_descriptions_en": [
    "scientific article published in September 1948",
    "scientific article published on 01 January 1957",
    "scientific article published on 01 November 1946",
    "scientific article published on 01 February 1958",
    "scientific article published on 01 December 1960",
    "scientific article published on 01 June 1952",
    "scientific article published on March 19, 2010",
    "scientific article published on 01 June 1958",
    "scientific article published on 01 October 1961",
    "scientific article published on 01 July 1954",
    "scientific article",
    "scientific article",
    "scientific article published in Nature",
    "written work by P. Karrer, E. Jucker",
    "scientific article published on 01 August 1959"
  ],
  "new_value_labels_en": [
    "The effect of alpha-tocopherol on the utilization of carotene by the rat.",
    "The question of the vitamin A blood level as affected by the menstrual cycle",
    "The determination of vitamin A and carotene in small quantities of blood serum",
    "Observations on carotenemia",
    "[Clinical and biochemical studies on mentally deficient children with mental defect and mental retardation.]",
    "Method of determination of vitamin A in blood",
    "The estimation of serum vitamin A with activated glycerol dichlorohydrin",
    "Artificial feeding of infants & vitamin A administration",
    "Features of amino acid metabolism in mentally deficient children in comparison to the normal population",
    "The mechanism of the transformation of beta-carotene into vitamin A in vivo",
    "The incidence of alkaptonuria: a study in chemical individuality. 1902 [classical article]",
    "The antithrombin activity of heparin",
    "Interrelationship of Vitamins A and E",
    "Carotinoide",
    "[Thin-layer chromatography. IV. Insertion scheme, marginal effect, \"acid and base\" layers, gradation technic]"
  ],
  "old_value": [
    "Q49182336",
    "Q74634188",
    "Q83292930",
    "Q74807075",
    "Q78882049",
    "Q76248372",
    "Q39109486",
    "Q78289407",
    "Q79409952",
    "Q73645037",
    "Q24652974",
    "Q78529461",
    "Q39246143",
    "Q60271532",
    "Q63874550",
    "Q79272812"
  ],
  "old_value_descriptions_en": [
    "scientific article published in September 1948",
    "scientific article published on 01 January 1957",
    "scientific article published on 01 November 1946",
    "scientific article published on 01 February 1958",
    "scientific article published on 01 December 1960",
    "scientific article published on 01 June 1952",
    "scientific article published on March 19, 2010",
    "scientific article published on 01 June 1958",
    "scientific article published on 01 October 1961",
    "scientific article published on 01 July 1954",
    "scientific article",
    "scientific article published on 01 January 1964",
    "scientific article",
    "scientific article published in Nature",
    "written work by P. Karrer, E. Jucker",
    "scientific article published on 01 August 1959"
  ],
  "old_value_labels_en": [
    "The effect of alpha-tocopherol on the utilization of carotene by the rat.",
    "The question of the vitamin A blood level as affected by the menstrual cycle",
    "The determination of vitamin A and carotene in small quantities of blood serum",
    "Observations on carotenemia",
    "[Clinical and biochemical studies on mentally deficient children with mental defect and mental retardation.]",
    "Method of determination of vitamin A in blood",
    "The estimation of serum vitamin A with activated glycerol dichlorohydrin",
    "Artificial feeding of infants & vitamin A administration",
    "Features of amino acid metabolism in mentally deficient children in comparison to the normal population",
    "The mechanism of the transformation of beta-carotene into vitamin A in vivo",
    "The incidence of alkaptonuria: a study in chemical individuality. 1902 [classical article]",
    "CLINICAL AND GENETIC STUDIES OF SOME CAROTENOIDS OF THE HUMAN BLOOD SERUM",
    "The antithrombin activity of heparin",
    "Interrelationship of Vitamins A and E",
    "Carotinoide",
    "[Thin-layer chromatography. IV. Insertion scheme, marginal effect, \"acid and base\" layers, gradation technic]"
  ],
  "revision_id": 2446944294,
  "value": [
    "Q49182336",
    "Q74634188",
    "Q83292930",
    "Q74807075",
    "Q78882049",
    "Q76248372",
    "Q39109486",
    "Q78289407",
    "Q79409952",
    "Q73645037",
    "Q24652974",
    "Q39246143",
    "Q60271532",
    "Q63874550",
    "Q79272812"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 15,
    "new_unique": [
      "Q24652974",
      "Q39109486",
      "Q39246143",
      "Q49182336",
      "Q60271532",
      "Q63874550",
      "Q73645037",
      "Q74634188",
      "Q74807075",
      "Q76248372",
      "Q78289407",
      "Q78882049",
      "Q79272812",
      "Q79409952",
      "Q83292930"
    ],
    "new_value": [
      "Q49182336",
      "Q74634188",
      "Q83292930",
      "Q74807075",
      "Q78882049",
      "Q76248372",
      "Q39109486",
      "Q78289407",
      "Q79409952",
      "Q73645037",
      "Q24652974",
      "Q39246143",
      "Q60271532",
      "Q63874550",
      "Q79272812"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 16,
    "old_unique": [
      "Q24652974",
      "Q39109486",
      "Q39246143",
      "Q49182336",
      "Q60271532",
      "Q63874550",
      "Q73645037",
      "Q74634188",
      "Q74807075",
      "Q76248372",
      "Q78289407",
      "Q78529461",
      "Q78882049",
      "Q79272812",
      "Q79409952",
      "Q83292930"
    ],
    "old_value": [
      "Q49182336",
      "Q74634188",
      "Q83292930",
      "Q74807075",
      "Q78882049",
      "Q76248372",
      "Q39109486",
      "Q78289407",
      "Q79409952",
      "Q73645037",
      "Q24652974",
      "Q78529461",
      "Q39246143",
      "Q60271532",
      "Q63874550",
      "Q79272812"
    ],
    "removed_unique_values": [
      "Q78529461"
    ],
    "value_multiplicity_changes": {
      "Q78529461": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "scientific article published in September 1948",
    "scientific article published on 01 January 1957",
    "scientific article published on 01 November 1946",
    "scientific article published on 01 February 1958",
    "scientific article published on 01 December 1960",
    "scientific article published on 01 June 1952",
    "scientific article published on March 19, 2010",
    "scientific article published on 01 June 1958",
    "scientific article published on 01 October 1961",
    "scientific article published on 01 July 1954",
    "scientific article",
    "scientific article",
    "scientific article published in Nature",
    "written work by P. Karrer, E. Jucker",
    "scientific article published on 01 August 1959"
  ],
  "value_labels_en": [
    "The effect of alpha-tocopherol on the utilization of carotene by the rat.",
    "The question of the vitamin A blood level as affected by the menstrual cycle",
    "The determination of vitamin A and carotene in small quantities of blood serum",
    "Observations on carotenemia",
    "[Clinical and biochemical studies on mentally deficient children with mental defect and mental retardation.]",
    "Method of determination of vitamin A in blood",
    "The estimation of serum vitamin A with activated glycerol dichlorohydrin",
    "Artificial feeding of infants & vitamin A administration",
    "Features of amino acid metabolism in mentally deficient children in comparison to the normal population",
    "The mechanism of the transformation of beta-carotene into vitamin A in vivo",
    "The incidence of alkaptonuria: a study in chemical individuality. 1902 [classical article]",
    "The antithrombin activity of heparin",
    "Interrelationship of Vitamins A and E",
    "Carotinoide",
    "[Thin-layer chromatography. IV. Insertion scheme, marginal effect, \"acid and base\" layers, gradation technic]"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-26T10:07:30",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2860",
  "report_revision_new": 2447318381,
  "report_revision_old": 2447008974,
  "report_violation_type": "Self link",
  "report_violation_type_normalized": "Self link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Self link",
  "value": [
    "Q49182336",
    "Q74634188",
    "Q83292930",
    "Q74807075",
    "Q78882049",
    "Q76248372",
    "Q39109486",
    "Q78289407",
    "Q79409952",
    "Q73645037",
    "Q24652974",
    "Q78529461",
    "Q39246143",
    "Q60271532",
    "Q63874550",
    "Q79272812"
  ],
  "value_descriptions_en": [
    "scientific article published in September 1948",
    "scientific article published on 01 January 1957",
    "scientific article published on 01 November 1946",
    "scientific article published on 01 February 1958",
    "scientific article published on 01 December 1960",
    "scientific article published on 01 June 1952",
    "scientific article published on March 19, 2010",
    "scientific article published on 01 June 1958",
    "scientific article published on 01 October 1961",
    "scientific article published on 01 July 1954",
    "scientific article",
    "scientific article published on 01 January 1964",
    "scientific article",
    "scientific article published in Nature",
    "written work by P. Karrer, E. Jucker",
    "scientific article published on 01 August 1959"
  ],
  "value_labels_en": [
    "The effect of alpha-tocopherol on the utilization of carotene by the rat.",
    "The question of the vitamin A blood level as affected by the menstrual cycle",
    "The determination of vitamin A and carotene in small quantities of blood serum",
    "Observations on carotenemia",
    "[Clinical and biochemical studies on mentally deficient children with mental defect and mental retardation.]",
    "Method of determination of vitamin A in blood",
    "The estimation of serum vitamin A with activated glycerol dichlorohydrin",
    "Artificial feeding of infants & vitamin A administration",
    "Features of amino acid metabolism in mentally deficient children in comparison to the normal population",
    "The mechanism of the transformation of beta-carotene into vitamin A in vivo",
    "The incidence of alkaptonuria: a study in chemical individuality. 1902 [classical article]",
    "CLINICAL AND GENETIC STUDIES OF SOME CAROTENOIDS OF THE HUMAN BLOOD SERUM",
    "The antithrombin activity of heparin",
    "Interrelationship of Vitamins A and E",
    "Carotinoide",
    "[Thin-layer chromatography. IV. Insertion scheme, marginal effect, \"acid and base\" layers, gradation technic]"
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
    "Q49182336",
    "Q74634188",
    "Q83292930",
    "Q74807075",
    "Q78882049",
    "Q76248372",
    "Q39109486",
    "Q78289407",
    "Q79409952",
    "Q73645037",
    "Q24652974",
    "Q39246143",
    "Q60271532",
    "Q63874550",
    "Q79272812"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "citation from one creative or scholarly work to another",
    "label": "cites work"
  },
  "qid": {
    "description": "scientific article published on 01 January 1964",
    "label": "CLINICAL AND GENETIC STUDIES OF SOME CAROTENOIDS OF THE HUMAN BLOOD SERUM"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
        "Q24652974": 1,
        "Q39109486": 1,
        "Q39246143": 1,
        "Q49182336": 1,
        "Q60271532": 1,
        "Q63874550": 1,
        "Q73645037": 1,
        "Q74634188": 1,
        "Q74807075": 1,
        "Q76248372": 1,
        "Q78289407": 1,
        "Q78882049": 1,
        "Q79272812": 1,
        "Q79409952": 1,
        "Q83292930": 1
      },
      "new_unique": [
        "Q24652974",
        "Q39109486",
        "Q39246143",
        "Q49182336",
        "Q60271532",
        "Q63874550",
        "Q73645037",
        "Q74634188",
        "Q74807075",
        "Q76248372",
        "Q78289407",
        "Q78882049",
        "Q79272812",
        "Q79409952",
        "Q83292930"
      ],
      "new_values": [
        "Q49182336",
        "Q74634188",
        "Q83292930",
        "Q74807075",
        "Q78882049",
        "Q76248372",
        "Q39109486",
        "Q78289407",
        "Q79409952",
        "Q73645037",
        "Q24652974",
        "Q39246143",
        "Q60271532",
        "Q63874550",
        "Q79272812"
      ],
      "new_values_raw": [
        "Q49182336",
        "Q74634188",
        "Q83292930",
        "Q74807075",
        "Q78882049",
        "Q76248372",
        "Q39109486",
        "Q78289407",
        "Q79409952",
        "Q73645037",
        "Q24652974",
        "Q39246143",
        "Q60271532",
        "Q63874550",
        "Q79272812"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "Q24652974": 1,
        "Q39109486": 1,
        "Q39246143": 1,
        "Q49182336": 1,
        "Q60271532": 1,
        "Q63874550": 1,
        "Q73645037": 1,
        "Q74634188": 1,
        "Q74807075": 1,
        "Q76248372": 1,
        "Q78289407": 1,
        "Q78529461": 1,
        "Q78882049": 1,
        "Q79272812": 1,
        "Q79409952": 1,
        "Q83292930": 1
      },
      "old_unique": [
        "Q24652974",
        "Q39109486",
        "Q39246143",
        "Q49182336",
        "Q60271532",
        "Q63874550",
        "Q73645037",
        "Q74634188",
        "Q74807075",
        "Q76248372",
        "Q78289407",
        "Q78529461",
        "Q78882049",
        "Q79272812",
        "Q79409952",
        "Q83292930"
      ],
      "old_values": [
        "Q49182336",
        "Q74634188",
        "Q83292930",
        "Q74807075",
        "Q78882049",
        "Q76248372",
        "Q39109486",
        "Q78289407",
        "Q79409952",
        "Q73645037",
        "Q24652974",
        "Q78529461",
        "Q39246143",
        "Q60271532",
        "Q63874550",
        "Q79272812"
      ],
      "old_values_raw": [
        "Q49182336",
        "Q74634188",
        "Q83292930",
        "Q74807075",
        "Q78882049",
        "Q76248372",
        "Q39109486",
        "Q78289407",
        "Q79409952",
        "Q73645037",
        "Q24652974",
        "Q78529461",
        "Q39246143",
        "Q60271532",
        "Q63874550",
        "Q79272812"
      ],
      "removed_unique_values": [
        "Q78529461"
      ],
      "retained_unique_values": [
        "Q24652974",
        "Q39109486",
        "Q39246143",
        "Q49182336",
        "Q60271532",
        "Q63874550",
        "Q73645037",
        "Q74634188",
        "Q74807075",
        "Q76248372",
        "Q78289407",
        "Q78882049",
        "Q79272812",
        "Q79409952",
        "Q83292930"
      ],
      "semantic_action": "DELETE_SUBSET",
      "value_multiplicity_changes": {
        "Q78529461": {
          "new": 0,
          "old": 1
        }
      }
    },
    "result": "DELETE_SUBSET",
    "step": "value_delta"
  },
  {
    "kind": "SELF_LINK",
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
    "result": "self_link_rejection",
    "step": "branch"
  }
]
```

---
