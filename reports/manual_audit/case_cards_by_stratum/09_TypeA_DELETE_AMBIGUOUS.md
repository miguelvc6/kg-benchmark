# TypeA_DELETE_AMBIGUOUS

Cases: 40

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q107578706_2444210633`

| Field | Value |
|---|---|
| qid | Q107578706 |
| property | P4404 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| group_key | ABOX::Q107578706::P4404 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Iamcarbon",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "9d80aebb-55f6-4c33-8741-f45301cf82b6"
  ],
  "revision_id": 2444210633,
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
      "9d80aebb-55f6-4c33-8741-f45301cf82b6"
    ],
    "old_value": [
      "9d80aebb-55f6-4c33-8741-f45301cf82b6"
    ],
    "removed_unique_values": [
      "9d80aebb-55f6-4c33-8741-f45301cf82b6"
    ],
    "value_multiplicity_changes": {
      "9d80aebb-55f6-4c33-8741-f45301cf82b6": {
        "new": 0,
        "old": 1
      },
      "MISSING": {
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
  "report_fix_date": "2025-12-21T06:55:38",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4404",
  "report_revision_new": 2444823785,
  "report_revision_old": 2444408016,
  "report_violation_type": "Item P|2550",
  "report_violation_type_normalized": "Item P|2550",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|2550",
  "report_violation_types": [
    "Item P|2550",
    "Conflicts with P|31"
  ],
  "value": [
    "9d80aebb-55f6-4c33-8741-f45301cf82b6"
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
    "description": "identifier for a recording in the MusicBrainz open music encyclopedia",
    "label": "MusicBrainz recording ID"
  },
  "qid": {
    "description": "Hank the Knife & the Jets song",
    "label": "Guitar King"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "item p 2550"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 002. `repair_Q109563704_2442329832`

| Field | Value |
|---|---|
| qid | Q109563704 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q21510851 |
| group_key | ABOX::Q109563704::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
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
    "Q252"
  ],
  "old_value_descriptions_en": [
    "island country in Southeast Asia and Oceania"
  ],
  "old_value_labels_en": [
    "Indonesia"
  ],
  "revision_id": 2442329832,
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
      "Q252"
    ],
    "old_value": [
      "Q252"
    ],
    "removed_unique_values": [
      "Q252"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q252": {
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
  "report_fix_date": "2025-12-16T16:27:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2443054962,
  "report_revision_old": 2442766763,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q252"
  ],
  "value_descriptions_en": [
    "island country in Southeast Asia and Oceania"
  ],
  "value_labels_en": [
    "Indonesia"
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
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "label": "country"
  },
  "qid": {
    "description": null,
    "label": "Irfan AB"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 003. `repair_Q114321801_2441116718`

| Field | Value |
|---|---|
| qid | Q114321801 |
| property | P106 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q53869507 |
| group_key | ABOX::Q114321801::P106 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q1650915"
  ],
  "old_value_descriptions_en": [
    "person who engages in research, professionally or otherwise. If a more specific occupation is known, use that instead"
  ],
  "old_value_labels_en": [
    "researcher"
  ],
  "revision_id": 2441116718,
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
      "Q1650915"
    ],
    "old_value": [
      "Q1650915"
    ],
    "removed_unique_values": [
      "Q1650915"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q1650915": {
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
  "report_fix_date": "2025-12-13T12:33:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P106",
  "report_revision_new": 2441832579,
  "report_revision_old": 2441298994,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q1650915"
  ],
  "value_descriptions_en": [
    "person who engages in research, professionally or otherwise. If a more specific occupation is known, use that instead"
  ],
  "value_labels_en": [
    "researcher"
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
    "description": "occupation of a person. See also \"field of work\" (Property:P101), \"position held\" (Property:P39). Not for groups of people. There, use \"field of work\" (Property:P101), \"industry\" (Property:P452), \"members have occupation\" (Property:P3989).",
    "label": "occupation"
  },
  "qid": {
    "description": "дослідник",
    "label": "OPTIMIZE-HF Investigators and Coordinators"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
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
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 004. `repair_Q114342236_2441117871`

| Field | Value |
|---|---|
| qid | Q114342236 |
| property | P106 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q53869507 |
| group_key | ABOX::Q114342236::P106 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Quesotiotyo",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q1650915"
  ],
  "old_value_descriptions_en": [
    "person who engages in research, professionally or otherwise. If a more specific occupation is known, use that instead"
  ],
  "old_value_labels_en": [
    "researcher"
  ],
  "revision_id": 2441117871,
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
      "Q1650915"
    ],
    "old_value": [
      "Q1650915"
    ],
    "removed_unique_values": [
      "Q1650915"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q1650915": {
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
  "report_fix_date": "2025-12-13T12:33:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P106",
  "report_revision_new": 2441832579,
  "report_revision_old": 2441298994,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q1650915"
  ],
  "value_descriptions_en": [
    "person who engages in research, professionally or otherwise. If a more specific occupation is known, use that instead"
  ],
  "value_labels_en": [
    "researcher"
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
    "description": "occupation of a person. See also \"field of work\" (Property:P101), \"position held\" (Property:P39). Not for groups of people. There, use \"field of work\" (Property:P101), \"industry\" (Property:P452), \"members have occupation\" (Property:P3989).",
    "label": "occupation"
  },
  "qid": {
    "description": "дослідник",
    "label": "Information Network of Departments of Dermatology (IVDK)"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
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
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 005. `repair_Q1166362_2422275679`

| Field | Value |
|---|---|
| qid | Q1166362 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q1166362::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "bourgogne-franche-comte/yonne/mairie-89281-01"
  ],
  "revision_id": 2422275679,
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
      "bourgogne-franche-comte/yonne/mairie-89281-01"
    ],
    "old_value": [
      "bourgogne-franche-comte/yonne/mairie-89281-01"
    ],
    "removed_unique_values": [
      "bourgogne-franche-comte/yonne/mairie-89281-01"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "bourgogne-franche-comte/yonne/mairie-89281-01": {
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
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "bourgogne-franche-comte/yonne/mairie-89281-01"
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Yonne, France",
    "label": "Les Ormes"
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
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 006. `repair_Q118288310_2444210911`

| Field | Value |
|---|---|
| qid | Q118288310 |
| property | P4404 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| group_key | ABOX::Q118288310::P4404 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Iamcarbon",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "ae50d552-e6a2-4ece-976e-47bccd072fb4"
  ],
  "revision_id": 2444210911,
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
      "ae50d552-e6a2-4ece-976e-47bccd072fb4"
    ],
    "old_value": [
      "ae50d552-e6a2-4ece-976e-47bccd072fb4"
    ],
    "removed_unique_values": [
      "ae50d552-e6a2-4ece-976e-47bccd072fb4"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "ae50d552-e6a2-4ece-976e-47bccd072fb4": {
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
  "report_fix_date": "2025-12-21T06:55:38",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4404",
  "report_revision_new": 2444823785,
  "report_revision_old": 2444408016,
  "report_violation_type": "Item P|2550",
  "report_violation_type_normalized": "Item P|2550",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|2550",
  "report_violation_types": [
    "Item P|2550",
    "Type Q|7302866, Q|193977",
    "Conflicts with P|31"
  ],
  "value": [
    "ae50d552-e6a2-4ece-976e-47bccd072fb4"
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
    "description": "identifier for a recording in the MusicBrainz open music encyclopedia",
    "label": "MusicBrainz recording ID"
  },
  "qid": {
    "description": "2012 song by Shinhwa",
    "label": "This Love"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "item p 2550"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 007. `repair_Q123254229_2166663398`

| Field | Value |
|---|---|
| qid | Q123254229 |
| property | P6425 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q19474404 |
| group_key | ABOX::Q123254229::P6425 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "RamSeraph",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "216"
  ],
  "revision_id": 2166663398,
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
      "216"
    ],
    "old_value": [
      "216"
    ],
    "removed_unique_values": [
      "216"
    ],
    "value_multiplicity_changes": {
      "216": {
        "new": 0,
        "old": 1
      },
      "MISSING": {
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
  "report_fix_date": "2024-05-30T08:09:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6425",
  "report_revision_new": 2168535356,
  "report_revision_old": 2155799388,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "216"
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
    "description": "code used for local government bodies by the Local Government Directory, a website run by the Government of India, which maintains directory of rural and urban local governments in India and their political territorial entities",
    "label": "LGD local body code"
  },
  "qid": {
    "description": "tehsil in Hoshiarpur district of Punjab, India",
    "label": "Hoshiarpur tehsil"
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
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 008. `repair_Q123256793_2166719508`

| Field | Value |
|---|---|
| qid | Q123256793 |
| property | P6425 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q19474404 |
| group_key | ABOX::Q123256793::P6425 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "RamSeraph",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "675"
  ],
  "revision_id": 2166719508,
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
      "675"
    ],
    "old_value": [
      "675"
    ],
    "removed_unique_values": [
      "675"
    ],
    "value_multiplicity_changes": {
      "675": {
        "new": 0,
        "old": 1
      },
      "MISSING": {
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
  "report_fix_date": "2024-05-30T08:09:52",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6425",
  "report_revision_new": 2168535356,
  "report_revision_old": 2155799388,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "675"
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
    "description": "code used for local government bodies by the Local Government Directory, a website run by the Government of India, which maintains directory of rural and urban local governments in India and their political territorial entities",
    "label": "LGD local body code"
  },
  "qid": {
    "description": "tehsil in Baran district of Rajasthan, India",
    "label": "Shahbad tehsil"
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
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 009. `repair_Q124966080_2445323477`

| Field | Value |
|---|---|
| qid | Q124966080 |
| property | P176 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q21510865 |
| group_key | ABOX::Q124966080::P176 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Meno25",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q137286046"
  ],
  "old_value_descriptions_en": [
    null
  ],
  "old_value_labels_en": [
    null
  ],
  "revision_id": 2445323477,
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
      "Q137286046"
    ],
    "old_value": [
      "Q137286046"
    ],
    "removed_unique_values": [
      "Q137286046"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q137286046": {
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
  "report_fix_date": "2025-12-23T15:04:55",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P176",
  "report_revision_new": 2446087617,
  "report_revision_old": 2445487255,
  "report_violation_type": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
  "report_violation_type_descriptions_en": [
    "social entity established to meet needs or pursue goals",
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "fictional human or non-human character in a narrative work of art",
    "set of fictional characters",
    "people who work in the arts, a handicraft, or a skilled trade, particularly when utilizing traditional or non-mechanized methods",
    "occupation requiring specialized training",
    "kingdom of multicellular eukaryotic organisms",
    "group of firms that produce a closely related set of raw materials, goods, or services",
    "facility where goods are industrially made, or processed",
    "label applied to a person based on an activity they participate in",
    "room or building, with tools, used to repair or make goods",
    "group of one or more organism(s), which a taxonomist adjudges to be a unit"
  ],
  "report_violation_type_labels_en": [
    "organization",
    "human",
    "character",
    "group of fictional characters",
    "artisan",
    "profession",
    "Animalia",
    "industry",
    "factory",
    "occupation",
    "workshop",
    "taxon"
  ],
  "report_violation_type_normalized": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
  "report_violation_type_qids": [
    "Q43229",
    "Q5",
    "Q95074",
    "Q14514600",
    "Q1294787",
    "Q28640",
    "Q729",
    "Q268592",
    "Q83405",
    "Q12737077",
    "Q656720",
    "Q16521"
  ],
  "report_violation_type_raw": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
  "value": [
    "Q137286046"
  ],
  "value_descriptions_en": [
    null
  ],
  "value_labels_en": [
    null
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
    "description": "(main or final) manufacturer or producer of this product",
    "label": "manufacturer"
  },
  "qid": {
    "description": "bright orange nail polish made by essie",
    "label": "Tangerine Tease"
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
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "value type q 43229 q 5 q 95074 q 14514600 q 1294787 q 28640 q 729 q 268592 q 83405 q 12737077 q 656720 q 16521"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 010. `repair_Q126283015_2445297216`

| Field | Value |
|---|---|
| qid | Q126283015 |
| property | P4404 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| group_key | ABOX::Q126283015::P4404 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Iamcarbon",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "135025b1-7f68-48bf-a97d-e2b013a13d75"
  ],
  "revision_id": 2445297216,
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
      "135025b1-7f68-48bf-a97d-e2b013a13d75"
    ],
    "old_value": [
      "135025b1-7f68-48bf-a97d-e2b013a13d75"
    ],
    "removed_unique_values": [
      "135025b1-7f68-48bf-a97d-e2b013a13d75"
    ],
    "value_multiplicity_changes": {
      "135025b1-7f68-48bf-a97d-e2b013a13d75": {
        "new": 0,
        "old": 1
      },
      "MISSING": {
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
  "report_fix_date": "2025-12-23T08:55:06",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4404",
  "report_revision_new": 2445939761,
  "report_revision_old": 2445383999,
  "report_violation_type": "Item P|2550",
  "report_violation_type_normalized": "Item P|2550",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|2550",
  "report_violation_types": [
    "Item P|2550",
    "Type Q|7302866, Q|193977",
    "Conflicts with P|31"
  ],
  "value": [
    "135025b1-7f68-48bf-a97d-e2b013a13d75"
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
    "description": "identifier for a recording in the MusicBrainz open music encyclopedia",
    "label": "MusicBrainz recording ID"
  },
  "qid": {
    "description": "2023 song by Knocked Loose",
    "label": "Deep in the Willow"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "item p 2550"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 011. `repair_Q126904533_2438902655`

| Field | Value |
|---|---|
| qid | Q126904533 |
| property | P6367 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21502410 |
| group_key | ABOX::Q126904533::P6367 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Sd5605",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "140187"
  ],
  "revision_id": 2438902655,
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
      "140187"
    ],
    "old_value": [
      "140187"
    ],
    "removed_unique_values": [
      "140187"
    ],
    "value_multiplicity_changes": {
      "140187": {
        "new": 0,
        "old": 1
      },
      "MISSING": {
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
  "report_fix_date": "2025-12-08T05:40:11",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6367",
  "report_revision_new": 2439487116,
  "report_revision_old": 2439077210,
  "report_violation_type": "Item P|13424",
  "report_violation_type_normalized": "Item P|13424",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|13424",
  "value": [
    "140187"
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
    "description": "a Taiwanese database for Games, Animation, Comic, Light Novels",
    "label": "Bahamut Gamer's Community ACG Database ID"
  },
  "qid": {
    "description": "2024 Japanese anime television series",
    "label": "Ranma ½"
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
  },
  {
    "label_en": "label in language constraint",
    "qid": "Q108139345"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "item p 13424"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 012. `repair_Q12758849_2442331064`

| Field | Value |
|---|---|
| qid | Q12758849 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21510851 |
| group_key | ABOX::Q12758849::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
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
    "Q403"
  ],
  "old_value_descriptions_en": [
    "country in Southeast Europe"
  ],
  "old_value_labels_en": [
    "Serbia"
  ],
  "revision_id": 2442331064,
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
      "Q403"
    ],
    "old_value": [
      "Q403"
    ],
    "removed_unique_values": [
      "Q403"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q403": {
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
  "report_fix_date": "2025-12-16T16:27:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2443054962,
  "report_revision_old": 2442766763,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q403"
  ],
  "value_descriptions_en": [
    "country in Southeast Europe"
  ],
  "value_labels_en": [
    "Serbia"
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
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "label": "country"
  },
  "qid": {
    "description": "obchodník pocházející ze staré dubrovnické rodiny, autor slovníku",
    "label": "Simo Budmani"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 013. `repair_Q132181527_2442323278`

| Field | Value |
|---|---|
| qid | Q132181527 |
| property | P131 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q132181527::P131 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
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
    "Q488326"
  ],
  "old_value_descriptions_en": [
    "province of the Democratic Republic of the Congo"
  ],
  "old_value_labels_en": [
    "South Kivu"
  ],
  "revision_id": 2442323278,
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
      "Q488326"
    ],
    "old_value": [
      "Q488326"
    ],
    "removed_unique_values": [
      "Q488326"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q488326": {
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
  "report_fix_date": "2025-12-16T13:51:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
  "report_revision_new": 2443007807,
  "report_revision_old": 2442705396,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q488326"
  ],
  "value_descriptions_en": [
    "province of the Democratic Republic of the Congo"
  ],
  "value_labels_en": [
    "South Kivu"
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
    "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
    "label": "located in the administrative territorial entity"
  },
  "qid": {
    "description": "politician of the Democratic Republic of the Congo",
    "label": "Janvier Msenyibwa Apele"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 014. `repair_Q136734934_2440052973`

| Field | Value |
|---|---|
| qid | Q136734934 |
| property | P123 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| group_key | ABOX::Q136734934::P123 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Dla archiv 1",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q914456"
  ],
  "old_value_descriptions_en": [
    "German publisher"
  ],
  "old_value_labels_en": [
    "Goldmann"
  ],
  "revision_id": 2440052973,
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
      "Q914456"
    ],
    "old_value": [
      "Q914456"
    ],
    "removed_unique_values": [
      "Q914456"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q914456": {
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
  "report_fix_date": "2025-12-11T15:15:00",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P123",
  "report_revision_new": 2440910119,
  "report_revision_old": 2440433467,
  "report_violation_type": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645",
  "report_violation_type_descriptions_en": [
    "prescription, including laws, regulations, instructions, guidelines, and social conventions; determinate method for performing any operation",
    "type of list that is ranked by some criteria",
    "a group of software licenses",
    "formal attestation of certain characteristics of an object, person, or organization",
    "non-tangible executable component of a computer",
    "physical or digital embodiment of an information artifact",
    "set of manifestations as defined in FRBR",
    "medium for recording information (words or images) typically on bound pages or more abstractly in electronic or audio form",
    "structured form of play",
    "Wikidata metaclass; legal concept of uniquely identifiable piece or work of music, either vocal or instrumental; NOT applicable to recordings, broadcasts, or individual publications of music in printed or digital form or on physical media",
    "multiple video games marketed under the same series name",
    "two or more compositions published under, or otherwise known by, a common name",
    "סוג קבוצת משחקי וידאו",
    "work manifested on the Internet",
    "presentation of a series of still images",
    "compilation of software, in most cases, from the same developer",
    "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
    "artistic work drawn with the aid of a computer",
    "online service provided on Minitel",
    "list of flora and/or fauna of a place or in a particular taxonomic group",
    "section of a work, most commonly a book",
    "content made available to the general public",
    "widely-accepted foundational work in a specific field, profession, or discipline",
    "unbound collection of visual artworks housed in a binder, folder or other container",
    "... omitted 2 items"
  ],
  "report_violation_type_labels_en": [
    "rule",
    "ranked list",
    "license scheme",
    "certification",
    "software",
    "manifestation",
    "group of manifestations",
    "book",
    "game",
    "musical work/composition",
    "video game series",
    "group of musical works",
    "group of video games often treated as a singular game",
    "online publication",
    "slide show",
    "software bundle",
    "version, edition or translation",
    "digital artistic drawing",
    "telematics service",
    "checklist",
    "chapter",
    "publication",
    "standard work",
    "portfolio",
    "... omitted 2 items"
  ],
  "report_violation_type_normalized": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645",
  "report_violation_type_qids": [
    "Q1151067",
    "Q80793969",
    "Q95107111",
    "Q374814",
    "Q7397",
    "Q286583",
    "Q17538690",
    "Q571",
    "Q11410",
    "Q105543609",
    "Q7058673",
    "Q115473170",
    "Q116779426",
    "Q1714118",
    "Q904997",
    "Q62651817",
    "Q3331189",
    "Q97180164",
    "Q124030631",
    "Q106140535",
    "Q1980247",
    "Q732577",
    "Q1748756",
    "Q49094714",
    "... omitted 2 items"
  ],
  "report_violation_type_raw": "Type Q|1151067, Q|80793969, Q|95107111, Q|374814, Q|7397, Q|286583, Q|17538690, Q|571, Q|11410, Q|105543609, Q|7058673, Q|115473170, Q|116779426, Q|1714118, Q|904997, Q|62651817, Q|3331189, Q|97180164, Q|124030631, Q|106140535, Q|1980247, Q|732577, Q|1748756, Q|49094714, Q|7725310, Q|1348645",
  "value": [
    "Q914456"
  ],
  "value_descriptions_en": [
    "German publisher"
  ],
  "value_labels_en": [
    "Goldmann"
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
    "description": "organization or person responsible for publishing a work, such as a book, periodical, printed music, podcast, game or software",
    "label": "publisher"
  },
  "qid": {
    "description": "Sachbuch von Richard David Precht",
    "label": "Angststillstand"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
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
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "type q 1151067 q 80793969 q 95107111 q 374814 q 7397 q 286583 q 17538690 q 571 q 11410 q 105543609 q 7058673 q 115473170 q 116779426 q 1714118 q 904997 q 62651817 q 3331189 q 97180164 q 124030631 q 106140535 q 1980247 q 732577 q 1748756 q 49094714 q 7725310 q 1348645"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 015. `repair_Q1388921_2422270407`

| Field | Value |
|---|---|
| qid | Q1388921 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q1388921::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "auvergne-rhone-alpes/rhone/mairie-69104-01"
  ],
  "revision_id": 2422270407,
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
      "auvergne-rhone-alpes/rhone/mairie-69104-01"
    ],
    "old_value": [
      "auvergne-rhone-alpes/rhone/mairie-69104-01"
    ],
    "removed_unique_values": [
      "auvergne-rhone-alpes/rhone/mairie-69104-01"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "auvergne-rhone-alpes/rhone/mairie-69104-01": {
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
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "auvergne-rhone-alpes/rhone/mairie-69104-01"
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Rhône, France",
    "label": "Jullié"
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
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 016. `repair_Q20876514_2440875090`

| Field | Value |
|---|---|
| qid | Q20876514 |
| property | P7937 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| group_key | ABOX::Q20876514::P7937 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Maxlath",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q8261"
  ],
  "old_value_descriptions_en": [
    "narrative text, normally of a substantial length and in the form of prose describing a fictional and sequential story"
  ],
  "old_value_labels_en": [
    "novel"
  ],
  "revision_id": 2440875090,
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
      "Q8261"
    ],
    "old_value": [
      "Q8261"
    ],
    "removed_unique_values": [
      "Q8261"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q8261": {
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
  "report_fix_date": "2025-12-13T06:34:27",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
  "report_revision_new": 2441661090,
  "report_revision_old": 2441108460,
  "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "any set of works",
    "creative work which only appears in works of fiction"
  ],
  "report_violation_type_labels_en": [
    "work",
    "group of works",
    "fictional creative work"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_qids": [
    "Q386724",
    "Q17489659",
    "Q15306849"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849",
  "value": [
    "Q8261"
  ],
  "value_descriptions_en": [
    "narrative text, normally of a substantial length and in the form of prose describing a fictional and sequential story"
  ],
  "value_labels_en": [
    "novel"
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
    "description": "structure of a creative work",
    "label": "form of creative work"
  },
  "qid": {
    "description": "Catalan writer",
    "label": "Albert Pijuan"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "type q 386724 q 17489659 q 15306849"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 017. `repair_Q215524_2422268373`

| Field | Value |
|---|---|
| qid | Q215524 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q215524::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "grand-est/meuse/mairie-55429-01"
  ],
  "revision_id": 2422268373,
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
      "grand-est/meuse/mairie-55429-01"
    ],
    "old_value": [
      "grand-est/meuse/mairie-55429-01"
    ],
    "removed_unique_values": [
      "grand-est/meuse/mairie-55429-01"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "grand-est/meuse/mairie-55429-01": {
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
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "grand-est/meuse/mairie-55429-01"
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Meuse, France",
    "label": "Riaville"
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
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 018. `repair_Q21783_2422270906`

| Field | Value |
|---|---|
| qid | Q21783 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q21783::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "grand-est/moselle/mairie-57033-01"
  ],
  "revision_id": 2422270906,
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
      "grand-est/moselle/mairie-57033-01"
    ],
    "old_value": [
      "grand-est/moselle/mairie-57033-01"
    ],
    "removed_unique_values": [
      "grand-est/moselle/mairie-57033-01"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "grand-est/moselle/mairie-57033-01": {
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
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "grand-est/moselle/mairie-57033-01"
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Moselle, France",
    "label": "Arzviller"
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
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 019. `repair_Q220857_2422274946`

| Field | Value |
|---|---|
| qid | Q220857 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q220857::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "bourgogne-franche-comte/yonne/mairie-89122-01"
  ],
  "revision_id": 2422274946,
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
      "bourgogne-franche-comte/yonne/mairie-89122-01"
    ],
    "old_value": [
      "bourgogne-franche-comte/yonne/mairie-89122-01"
    ],
    "removed_unique_values": [
      "bourgogne-franche-comte/yonne/mairie-89122-01"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "bourgogne-franche-comte/yonne/mairie-89122-01": {
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
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "bourgogne-franche-comte/yonne/mairie-89122-01"
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Yonne, France",
    "label": "Courgenay"
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
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 020. `repair_Q22148_2422273290`

| Field | Value |
|---|---|
| qid | Q22148 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q22148::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "grand-est/moselle/mairie-57508-01"
  ],
  "revision_id": 2422273290,
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
      "grand-est/moselle/mairie-57508-01"
    ],
    "old_value": [
      "grand-est/moselle/mairie-57508-01"
    ],
    "removed_unique_values": [
      "grand-est/moselle/mairie-57508-01"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "grand-est/moselle/mairie-57508-01": {
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
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "grand-est/moselle/mairie-57508-01"
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Moselle, France",
    "label": "Nilvange"
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
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 021. `repair_Q22758_2422264431`

| Field | Value |
|---|---|
| qid | Q22758 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q22758::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "grand-est/bas-rhin/mairie-67021-01"
  ],
  "revision_id": 2422264431,
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
      "grand-est/bas-rhin/mairie-67021-01"
    ],
    "old_value": [
      "grand-est/bas-rhin/mairie-67021-01"
    ],
    "removed_unique_values": [
      "grand-est/bas-rhin/mairie-67021-01"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "grand-est/bas-rhin/mairie-67021-01": {
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
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "grand-est/bas-rhin/mairie-67021-01"
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Bas-Rhin, Alsace, France",
    "label": "Barr"
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
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 022. `repair_Q24010600_2396268896`

| Field | Value |
|---|---|
| qid | Q24010600 |
| property | P18 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q24010600::P18 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Румянцева Ольга Владимировна.JPG"
  ],
  "revision_id": 2396268896,
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
      "Румянцева Ольга Владимировна.JPG"
    ],
    "old_value": [
      "Румянцева Ольга Владимировна.JPG"
    ],
    "removed_unique_values": [
      "Румянцева Ольга Владимировна.JPG"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Румянцева Ольга Владимировна.JPG": {
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
  "report_fix_date": "2025-08-26T13:03:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P18",
  "report_revision_new": 2396759629,
  "report_revision_old": 2396416041,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Румянцева Ольга Владимировна.JPG"
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
    "description": "image of relevant illustration of the subject; if available, also use more specific properties (sample: coat of arms image, locator map, flag image, signature image, logo image, collage image)",
    "label": "image"
  },
  "qid": {
    "description": "Russian scientist",
    "label": "Olʹga Vladimirovna Rumânceva"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "commons link"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 023. `repair_Q28170070_2442333753`

| Field | Value |
|---|---|
| qid | Q28170070 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21510851 |
| group_key | ABOX::Q28170070::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
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
    "Q668"
  ],
  "old_value_descriptions_en": [
    "country in South Asia"
  ],
  "old_value_labels_en": [
    "India"
  ],
  "revision_id": 2442333753,
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
      "Q668"
    ],
    "old_value": [
      "Q668"
    ],
    "removed_unique_values": [
      "Q668"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q668": {
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
  "report_fix_date": "2025-12-16T16:27:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2443054962,
  "report_revision_old": 2442766763,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q668"
  ],
  "value_descriptions_en": [
    "country in South Asia"
  ],
  "value_labels_en": [
    "India"
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
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "label": "country"
  },
  "qid": {
    "description": "chahamana king",
    "label": "Chandraraja I"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 024. `repair_Q28497346_2442333932`

| Field | Value |
|---|---|
| qid | Q28497346 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21510851 |
| group_key | ABOX::Q28497346::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
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
    "Q159"
  ],
  "old_value_descriptions_en": [
    "country in Eastern Europe and Northern Asia"
  ],
  "old_value_labels_en": [
    "Russia"
  ],
  "revision_id": 2442333932,
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
      "Q159"
    ],
    "old_value": [
      "Q159"
    ],
    "removed_unique_values": [
      "Q159"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q159": {
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
  "report_fix_date": "2025-12-16T16:27:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2443054962,
  "report_revision_old": 2442766763,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q159"
  ],
  "value_descriptions_en": [
    "country in Eastern Europe and Northern Asia"
  ],
  "value_labels_en": [
    "Russia"
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
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "label": "country"
  },
  "qid": {
    "description": "politicus",
    "label": "Gasanov Magomedkadi"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 025. `repair_Q30922055_2442334217`

| Field | Value |
|---|---|
| qid | Q30922055 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q21510851 |
| group_key | ABOX::Q30922055::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
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
    "Q881"
  ],
  "old_value_descriptions_en": [
    "country in Southeast Asia"
  ],
  "old_value_labels_en": [
    "Vietnam"
  ],
  "revision_id": 2442334217,
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
      "Q881"
    ],
    "old_value": [
      "Q881"
    ],
    "removed_unique_values": [
      "Q881"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q881": {
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
  "report_fix_date": "2025-12-16T16:27:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2443054962,
  "report_revision_old": 2442766763,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q881"
  ],
  "value_descriptions_en": [
    "country in Southeast Asia"
  ],
  "value_labels_en": [
    "Vietnam"
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
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "label": "country"
  },
  "qid": {
    "description": "Vietnamese Cai Luong singer",
    "label": "Thoại Miêu"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 026. `repair_Q4151509_2440872063`

| Field | Value |
|---|---|
| qid | Q4151509 |
| property | P7937 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| group_key | ABOX::Q4151509::P7937 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Maxlath",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q8261"
  ],
  "old_value_descriptions_en": [
    "narrative text, normally of a substantial length and in the form of prose describing a fictional and sequential story"
  ],
  "old_value_labels_en": [
    "novel"
  ],
  "revision_id": 2440872063,
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
      "Q8261"
    ],
    "old_value": [
      "Q8261"
    ],
    "removed_unique_values": [
      "Q8261"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q8261": {
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
  "report_fix_date": "2025-12-13T06:34:27",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
  "report_revision_new": 2441661090,
  "report_revision_old": 2441108460,
  "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "any set of works",
    "creative work which only appears in works of fiction"
  ],
  "report_violation_type_labels_en": [
    "work",
    "group of works",
    "fictional creative work"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_qids": [
    "Q386724",
    "Q17489659",
    "Q15306849"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849",
  "value": [
    "Q8261"
  ],
  "value_descriptions_en": [
    "narrative text, normally of a substantial length and in the form of prose describing a fictional and sequential story"
  ],
  "value_labels_en": [
    "novel"
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
    "description": "structure of a creative work",
    "label": "form of creative work"
  },
  "qid": {
    "description": "Soviet writer (1908-1969)",
    "label": "Aleksandras Gudaitis Guzevičius"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "type q 386724 q 17489659 q 15306849"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 027. `repair_Q42398026_2440876933`

| Field | Value |
|---|---|
| qid | Q42398026 |
| property | P7937 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| group_key | ABOX::Q42398026::P7937 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Maxlath",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q3328821"
  ],
  "old_value_descriptions_en": [
    "literary form characterized by the description of a sequence of events in a certain order"
  ],
  "old_value_labels_en": [
    "narration"
  ],
  "revision_id": 2440876933,
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
      "Q3328821"
    ],
    "old_value": [
      "Q3328821"
    ],
    "removed_unique_values": [
      "Q3328821"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q3328821": {
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
  "report_fix_date": "2025-12-13T06:34:27",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
  "report_revision_new": 2441661090,
  "report_revision_old": 2441108460,
  "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "any set of works",
    "creative work which only appears in works of fiction"
  ],
  "report_violation_type_labels_en": [
    "work",
    "group of works",
    "fictional creative work"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_qids": [
    "Q386724",
    "Q17489659",
    "Q15306849"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_types": [
    "Type Q|386724, Q|17489659, Q|15306849",
    "None of"
  ],
  "value": [
    "Q3328821"
  ],
  "value_descriptions_en": [
    "literary form characterized by the description of a sequence of events in a certain order"
  ],
  "value_labels_en": [
    "narration"
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
    "description": "structure of a creative work",
    "label": "form of creative work"
  },
  "qid": {
    "description": "Uruguayan writer",
    "label": "Gerardo Circelli"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "type q 386724 q 17489659 q 15306849"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 028. `repair_Q42401453_2440888611`

| Field | Value |
|---|---|
| qid | Q42401453 |
| property | P7937 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| group_key | ABOX::Q42401453::P7937 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Maxlath",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q3328821"
  ],
  "old_value_descriptions_en": [
    "literary form characterized by the description of a sequence of events in a certain order"
  ],
  "old_value_labels_en": [
    "narration"
  ],
  "revision_id": 2440888611,
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
      "Q3328821"
    ],
    "old_value": [
      "Q3328821"
    ],
    "removed_unique_values": [
      "Q3328821"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q3328821": {
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
  "report_fix_date": "2025-12-13T06:34:27",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
  "report_revision_new": 2441661090,
  "report_revision_old": 2441108460,
  "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "any set of works",
    "creative work which only appears in works of fiction"
  ],
  "report_violation_type_labels_en": [
    "work",
    "group of works",
    "fictional creative work"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_qids": [
    "Q386724",
    "Q17489659",
    "Q15306849"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_types": [
    "Type Q|386724, Q|17489659, Q|15306849",
    "None of"
  ],
  "value": [
    "Q3328821"
  ],
  "value_descriptions_en": [
    "literary form characterized by the description of a sequence of events in a certain order"
  ],
  "value_labels_en": [
    "narration"
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
    "description": "structure of a creative work",
    "label": "form of creative work"
  },
  "qid": {
    "description": "Uruguayan writer",
    "label": "Sharon Musselli"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "type q 386724 q 17489659 q 15306849"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 029. `repair_Q42404324_2440889361`

| Field | Value |
|---|---|
| qid | Q42404324 |
| property | P7937 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| group_key | ABOX::Q42404324::P7937 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Maxlath",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q3328821"
  ],
  "old_value_descriptions_en": [
    "literary form characterized by the description of a sequence of events in a certain order"
  ],
  "old_value_labels_en": [
    "narration"
  ],
  "revision_id": 2440889361,
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
      "Q3328821"
    ],
    "old_value": [
      "Q3328821"
    ],
    "removed_unique_values": [
      "Q3328821"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q3328821": {
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
  "report_fix_date": "2025-12-13T06:34:27",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
  "report_revision_new": 2441661090,
  "report_revision_old": 2441108460,
  "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "any set of works",
    "creative work which only appears in works of fiction"
  ],
  "report_violation_type_labels_en": [
    "work",
    "group of works",
    "fictional creative work"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_qids": [
    "Q386724",
    "Q17489659",
    "Q15306849"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_types": [
    "Type Q|386724, Q|17489659, Q|15306849",
    "None of"
  ],
  "value": [
    "Q3328821"
  ],
  "value_descriptions_en": [
    "literary form characterized by the description of a sequence of events in a certain order"
  ],
  "value_labels_en": [
    "narration"
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
    "description": "structure of a creative work",
    "label": "form of creative work"
  },
  "qid": {
    "description": "Uruguayan writer",
    "label": "Elsa Ramade de Paiva"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "type q 386724 q 17489659 q 15306849"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 030. `repair_Q4272898_2442334771`

| Field | Value |
|---|---|
| qid | Q4272898 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21510851 |
| group_key | ABOX::Q4272898::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
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
    "Q159"
  ],
  "old_value_descriptions_en": [
    "country in Eastern Europe and Northern Asia"
  ],
  "old_value_labels_en": [
    "Russia"
  ],
  "revision_id": 2442334771,
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
      "Q159"
    ],
    "old_value": [
      "Q159"
    ],
    "removed_unique_values": [
      "Q159"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q159": {
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
  "report_fix_date": "2025-12-16T16:27:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2443054962,
  "report_revision_old": 2442766763,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q159"
  ],
  "value_descriptions_en": [
    "country in Eastern Europe and Northern Asia"
  ],
  "value_labels_en": [
    "Russia"
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
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "label": "country"
  },
  "qid": {
    "description": "Russian scientist",
    "label": "Boris Lyovin"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 031. `repair_Q4303556_2440875959`

| Field | Value |
|---|---|
| qid | Q4303556 |
| property | P7937 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| group_key | ABOX::Q4303556::P7937 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Maxlath",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Q25379"
  ],
  "old_value_descriptions_en": [
    "theatrical dramatic work intended to be performed by actors (in theatre, radio or recorded for TV)"
  ],
  "old_value_labels_en": [
    "play"
  ],
  "revision_id": 2440875959,
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
      "Q25379"
    ],
    "old_value": [
      "Q25379"
    ],
    "removed_unique_values": [
      "Q25379"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q25379": {
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
  "report_fix_date": "2025-12-13T06:34:27",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
  "report_revision_new": 2441661090,
  "report_revision_old": 2441108460,
  "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "any set of works",
    "creative work which only appears in works of fiction"
  ],
  "report_violation_type_labels_en": [
    "work",
    "group of works",
    "fictional creative work"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
  "report_violation_type_qids": [
    "Q386724",
    "Q17489659",
    "Q15306849"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849",
  "value": [
    "Q25379"
  ],
  "value_descriptions_en": [
    "theatrical dramatic work intended to be performed by actors (in theatre, radio or recorded for TV)"
  ],
  "value_labels_en": [
    "play"
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
    "description": "structure of a creative work",
    "label": "form of creative work"
  },
  "qid": {
    "description": "Georgian and Soviet screenwriter and writer (1896-1954)",
    "label": "Ilo Mosashvili"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "type q 386724 q 17489659 q 15306849"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 032. `repair_Q4524043_2442335157`

| Field | Value |
|---|---|
| qid | Q4524043 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21510851 |
| group_key | ABOX::Q4524043::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
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
    "Q159"
  ],
  "old_value_descriptions_en": [
    "country in Eastern Europe and Northern Asia"
  ],
  "old_value_labels_en": [
    "Russia"
  ],
  "revision_id": 2442335157,
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
      "Q159"
    ],
    "old_value": [
      "Q159"
    ],
    "removed_unique_values": [
      "Q159"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q159": {
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
  "report_fix_date": "2025-12-16T16:27:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2443054962,
  "report_revision_old": 2442766763,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q159"
  ],
  "value_descriptions_en": [
    "country in Eastern Europe and Northern Asia"
  ],
  "value_labels_en": [
    "Russia"
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
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "label": "country"
  },
  "qid": {
    "description": "Russian politician",
    "label": "Maksim Shingarkin"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 033. `repair_Q4545128_2442078917`

| Field | Value |
|---|---|
| qid | Q4545128 |
| property | P212 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21502410 |
| group_key | ABOX::Q4545128::P212 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Pfadintegral",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "978-1-59740-165-4"
  ],
  "revision_id": 2442078917,
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
      "978-1-59740-165-4"
    ],
    "old_value": [
      "978-1-59740-165-4"
    ],
    "removed_unique_values": [
      "978-1-59740-165-4"
    ],
    "value_multiplicity_changes": {
      "978-1-59740-165-4": {
        "new": 0,
        "old": 1
      },
      "MISSING": {
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
  "report_fix_date": "2025-12-15T13:11:03",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P212",
  "report_revision_new": 2442672664,
  "report_revision_old": 2442291443,
  "report_violation_type": "Item P|629",
  "report_violation_type_normalized": "Item P|629",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|629",
  "report_violation_types": [
    "Item P|629",
    "Conflicts with P|8383",
    "Type Q|3331189, Q|187685, Q|732577, Q|7889, Q|317623, Q|1711593, Q|131436, Q|1266946",
    "Conflicts with P|31"
  ],
  "value": [
    "978-1-59740-165-4"
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
    "description": "identifier for a book (edition), thirteen digit",
    "label": "International Standard Book Number-13"
  },
  "qid": {
    "description": "1985 nonfiction book",
    "label": "...The Heavens and the Earth"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "item p 629"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 034. `repair_Q476584_2422275323`

| Field | Value |
|---|---|
| qid | Q476584 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q476584::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "bourgogne-franche-comte/yonne/mairie-89206-01"
  ],
  "revision_id": 2422275323,
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
      "bourgogne-franche-comte/yonne/mairie-89206-01"
    ],
    "old_value": [
      "bourgogne-franche-comte/yonne/mairie-89206-01"
    ],
    "removed_unique_values": [
      "bourgogne-franche-comte/yonne/mairie-89206-01"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "bourgogne-franche-comte/yonne/mairie-89206-01": {
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
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "bourgogne-franche-comte/yonne/mairie-89206-01"
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Yonne, France",
    "label": "Joigny"
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
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 035. `repair_Q512428_2422742156`

| Field | Value |
|---|---|
| qid | Q512428 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q512428::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "bourgogne-franche-comte/saone-et-loire/mairie-71115-01"
  ],
  "revision_id": 2422742156,
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
      "bourgogne-franche-comte/saone-et-loire/mairie-71115-01"
    ],
    "old_value": [
      "bourgogne-franche-comte/saone-et-loire/mairie-71115-01"
    ],
    "removed_unique_values": [
      "bourgogne-franche-comte/saone-et-loire/mairie-71115-01"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "bourgogne-franche-comte/saone-et-loire/mairie-71115-01": {
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
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "bourgogne-franche-comte/saone-et-loire/mairie-71115-01"
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Saône-et-Loire, France",
    "label": "Châtel-Moron"
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
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 036. `repair_Q55104097_2442335665`

| Field | Value |
|---|---|
| qid | Q55104097 |
| property | P17 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21510851 |
| group_key | ABOX::Q55104097::P17 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
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
    "Q159"
  ],
  "old_value_descriptions_en": [
    "country in Eastern Europe and Northern Asia"
  ],
  "old_value_labels_en": [
    "Russia"
  ],
  "revision_id": 2442335665,
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
      "Q159"
    ],
    "old_value": [
      "Q159"
    ],
    "removed_unique_values": [
      "Q159"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "Q159": {
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
  "report_fix_date": "2025-12-16T16:27:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
  "report_revision_new": 2443054962,
  "report_revision_old": 2442766763,
  "report_violation_type": "Conflicts with P|31",
  "report_violation_type_normalized": "Conflicts with P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|31",
  "value": [
    "Q159"
  ],
  "value_descriptions_en": [
    "country in Eastern Europe and Northern Asia"
  ],
  "value_labels_en": [
    "Russia"
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
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "label": "country"
  },
  "qid": {
    "description": "politicus",
    "label": "Grigory Balykhin"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 31"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 037. `repair_Q5903361_2394262286`

| Field | Value |
|---|---|
| qid | Q5903361 |
| property | P18 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q5903361::P18 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "CommonsDelinker",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Hotel Glam.jpg"
  ],
  "revision_id": 2394262286,
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
      "Hotel Glam.jpg"
    ],
    "old_value": [
      "Hotel Glam.jpg"
    ],
    "removed_unique_values": [
      "Hotel Glam.jpg"
    ],
    "value_multiplicity_changes": {
      "Hotel Glam.jpg": {
        "new": 0,
        "old": 1
      },
      "MISSING": {
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
  "report_fix_date": "2025-08-22T09:20:17",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P18",
  "report_revision_new": 2395357701,
  "report_revision_old": 2395141195,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": [
    "Hotel Glam.jpg"
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
    "description": "image of relevant illustration of the subject; if available, also use more specific properties (sample: coat of arms image, locator map, flag image, signature image, logo image, collage image)",
    "label": "image"
  },
  "qid": {
    "description": "televisieprogramma",
    "label": "Hotel Glam (programa de televisión)"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "commons link"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 038. `repair_Q59322_2422268846`

| Field | Value |
|---|---|
| qid | Q59322 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q59322::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "grand-est/meuse/mairie-55525-01"
  ],
  "revision_id": 2422268846,
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
      "grand-est/meuse/mairie-55525-01"
    ],
    "old_value": [
      "grand-est/meuse/mairie-55525-01"
    ],
    "removed_unique_values": [
      "grand-est/meuse/mairie-55525-01"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "grand-est/meuse/mairie-55525-01": {
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
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "grand-est/meuse/mairie-55525-01"
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Meuse, France",
    "label": "Vadelaincourt"
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
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 039. `repair_Q6160283_2443968709`

| Field | Value |
|---|---|
| qid | Q6160283 |
| property | P6802 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q21510852 |
| group_key | ABOX::Q6160283::P6802 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "CommonsDelinker",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Jarkko Laine graffiti.jpg"
  ],
  "revision_id": 2443968709,
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
      "Jarkko Laine graffiti.jpg"
    ],
    "old_value": [
      "Jarkko Laine graffiti.jpg"
    ],
    "removed_unique_values": [
      "Jarkko Laine graffiti.jpg"
    ],
    "value_multiplicity_changes": {
      "Jarkko Laine graffiti.jpg": {
        "new": 0,
        "old": 1
      },
      "MISSING": {
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
  "report_fix_date": "2025-12-20T06:24:59",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6802",
  "report_revision_new": 2444398259,
  "report_revision_old": 2443982698,
  "report_violation_type": "Conflicts with P|18",
  "report_violation_type_normalized": "Conflicts with P|18",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|18",
  "value": [
    "Jarkko Laine graffiti.jpg"
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
    "description": "less fitting image, used only because a better alternative is not available. If an appropriate image of the item is available, use P18 instead. Value should not be a generic placeholder",
    "label": "related image"
  },
  "qid": {
    "description": "Finnish author, poet and translator (1947–2006)",
    "label": "Jarkko Laine"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "Commons link constraint",
    "qid": "Q21510852"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "insufficient_delete_evidence",
      "report_type": "conflicts with p 18"
    },
    "result": "DELETE_AMBIGUOUS",
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

## 040. `repair_Q71307_2422269300`

| Field | Value |
|---|---|
| qid | Q71307 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / DELETE_AMBIGUOUS / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ambiguous_delete |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q71307::P6671 |
| tbox_revision_key |  |

### Annotation Focus

- Confirm whether delete ambiguity is real. If the rule alone forces deletion, mark clean_rule_or_format; otherwise delete_ambiguous_ok.
- This stratum should normally remain diagnostic unless audit shows a clean split.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch | delete_refined |
| rationale | A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "bretagne/morbihan/mairie-56030-01"
  ],
  "revision_id": 2422269300,
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
      "bretagne/morbihan/mairie-56030-01"
    ],
    "old_value": [
      "bretagne/morbihan/mairie-56030-01"
    ],
    "removed_unique_values": [
      "bretagne/morbihan/mairie-56030-01"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 1,
        "old": 0
      },
      "bretagne/morbihan/mairie-56030-01": {
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
  "report_fix_date": "2025-10-28T07:23:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P6671",
  "report_revision_new": 2422947515,
  "report_revision_old": 2419240415,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "bretagne/morbihan/mairie-56030-01"
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
    "description": "identifier for French public services",
    "label": "French public service directory ID"
  },
  "qid": {
    "description": "commune in Morbihan, France",
    "label": "Camoël"
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
      "delete_reason": "selection_conflict",
      "report_type": "unique value"
    },
    "result": "DELETE_AMBIGUOUS",
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
