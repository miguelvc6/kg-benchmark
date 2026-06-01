# TypeA_REJECTION_FORMAT_INVALID

Cases: 23

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q1026102_2297167189`

| Field | Value |
|---|---|
| qid | Q1026102 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q1026102::P8748 |
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
| classification_target_tokens | ["pta1298"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "pta1298"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pta1298"
  ],
  "removed_unique_values": [
    "pta1298"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "pta1298"
  ],
  "revision_id": 2297167189,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "pta1298": 1
    },
    "old_unique": [
      "pta1298"
    ],
    "old_values": [
      "pta1298"
    ],
    "old_values_raw": [
      "pta1298"
    ],
    "removed_unique_values": [
      "pta1298"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "pta1298": {
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
    "pta1298"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German politician (1797-1874)",
    "label": "Johann Peter Kajus zu Stolberg-Stolberg"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 002. `repair_Q105765_2297199427`

| Field | Value |
|---|---|
| qid | Q105765 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q105765::P8748 |
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
| classification_target_tokens | ["pka0832"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "pka0832"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pka0832"
  ],
  "removed_unique_values": [
    "pka0832"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "pka0832"
  ],
  "revision_id": 2297199427,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "pka0832": 1
    },
    "old_unique": [
      "pka0832"
    ],
    "old_values": [
      "pka0832"
    ],
    "old_values_raw": [
      "pka0832"
    ],
    "removed_unique_values": [
      "pka0832"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "pka0832": {
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
    "pka0832"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "deutscher Industrieller",
    "label": "Friedrich Albert Carl Spaeter"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 003. `repair_Q105793314_2354400897`

| Field | Value |
|---|---|
| qid | Q105793314 |
| property | P3628 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q105793314::P3628 |
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
| classification_target_tokens | ["yorks/north/vol2/pp157-160#p10"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "yorks/north/vol2/pp157-160#p10"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "yorks/north/vol2/pp157-160#p10"
  ],
  "removed_unique_values": [
    "yorks/north/vol2/pp157-160#p10"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Peter James",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "yorks/north/vol2/pp157-160#p10"
  ],
  "revision_id": 2354400897,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "yorks/north/vol2/pp157-160#p10": 1
    },
    "old_unique": [
      "yorks/north/vol2/pp157-160#p10"
    ],
    "old_values": [
      "yorks/north/vol2/pp157-160#p10"
    ],
    "old_values_raw": [
      "yorks/north/vol2/pp157-160#p10"
    ],
    "removed_unique_values": [
      "yorks/north/vol2/pp157-160#p10"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "yorks/north/vol2/pp157-160#p10": {
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
  "report_fix_date": "2025-05-31T07:10:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P3628",
  "report_revision_new": 2354963425,
  "report_revision_old": 2348416937,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "yorks/north/vol2/pp157-160#p10"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "identifier of a place, in the British History Online digitisation of the Victoria County History",
    "label": "British History Online VCH ID"
  },
  "qid": {
    "description": "former manorial estate in Hull, Yorkshire",
    "label": "Myton Manor"
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
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 004. `repair_Q1193791_2430051362`

| Field | Value |
|---|---|
| qid | Q1193791 |
| property | P2833 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q1193791::P2833 |
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
| classification_target_tokens | ["oak/quercus-alnifolia"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "oak/quercus-alnifolia"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "oak/quercus-alnifolia"
  ],
  "removed_unique_values": [
    "oak/quercus-alnifolia"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "David Newton",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "oak/quercus-alnifolia"
  ],
  "revision_id": 2430051362,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "oak/quercus-alnifolia": 1
    },
    "old_unique": [
      "oak/quercus-alnifolia"
    ],
    "old_values": [
      "oak/quercus-alnifolia"
    ],
    "old_values_raw": [
      "oak/quercus-alnifolia"
    ],
    "removed_unique_values": [
      "oak/quercus-alnifolia"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "oak/quercus-alnifolia": {
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
  "report_fix_date": "2025-11-17T08:18:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2833",
  "report_revision_new": 2430620673,
  "report_revision_old": 2426460495,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Single value"
  ],
  "value": [
    "oak/quercus-alnifolia"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "identifier for a taxon, in the ARKive database",
    "label": "ARKive ID (archived)"
  },
  "qid": {
    "description": "species of plant",
    "label": "Carvalho-dourado"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 005. `repair_Q136162983_2446041991`

| Field | Value |
|---|---|
| qid | Q136162983 |
| property | P2003 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q136162983::P2003 |
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
| classification_target_tokens | ["explore/locations/643817915642915"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "explore/locations/643817915642915"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "explore/locations/643817915642915"
  ],
  "removed_unique_values": [
    "explore/locations/643817915642915"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "explore/locations/643817915642915"
  ],
  "revision_id": 2446041991,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "explore/locations/643817915642915": 1
    },
    "old_unique": [
      "explore/locations/643817915642915"
    ],
    "old_values": [
      "explore/locations/643817915642915"
    ],
    "old_values_raw": [
      "explore/locations/643817915642915"
    ],
    "removed_unique_values": [
      "explore/locations/643817915642915"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "explore/locations/643817915642915": {
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
  "report_fix_date": "2025-12-25T16:38:27",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2003",
  "report_revision_new": 2447016775,
  "report_revision_old": 2446441637,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Type Q|5, Q|43229, Q|386724, Q|1190554, Q|340169, Q|15617994, Q|18127, Q|11012, Q|16979650, Q|308905, Q|104921473, Q|11664239, Q|65676181, Q|102345381, Q|26401003, Q|15632617, Q|431289, Q|56061, Q|2906862, Q|118188043, Q|105416259, Q|4164871, Q|95074",
    "Item P|31"
  ],
  "value": [
    "explore/locations/643817915642915"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "item's username on Instagram",
    "label": "Instagram username"
  },
  "qid": {
    "description": null,
    "label": "岩津城"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 006. `repair_Q1364250_2422270387`

| Field | Value |
|---|---|
| qid | Q1364250 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q1364250::P6671 |
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
| classification_target_tokens | ["auvergne-rhone-alpes/rhone/6783fd62-ee8e-4c33-a66f-c53589d6f669"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "auvergne-rhone-alpes/rhone/6783fd62-ee8e-4c33-a66f-c53589d6f669"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "auvergne-rhone-alpes/rhone/6783fd62-ee8e-4c33-a66f-c53589d6f669"
  ],
  "removed_unique_values": [
    "auvergne-rhone-alpes/rhone/6783fd62-ee8e-4c33-a66f-c53589d6f669"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "auvergne-rhone-alpes/rhone/6783fd62-ee8e-4c33-a66f-c53589d6f669"
  ],
  "revision_id": 2422270387,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "auvergne-rhone-alpes/rhone/6783fd62-ee8e-4c33-a66f-c53589d6f669": 1
    },
    "old_unique": [
      "auvergne-rhone-alpes/rhone/6783fd62-ee8e-4c33-a66f-c53589d6f669"
    ],
    "old_values": [
      "auvergne-rhone-alpes/rhone/6783fd62-ee8e-4c33-a66f-c53589d6f669"
    ],
    "old_values_raw": [
      "auvergne-rhone-alpes/rhone/6783fd62-ee8e-4c33-a66f-c53589d6f669"
    ],
    "removed_unique_values": [
      "auvergne-rhone-alpes/rhone/6783fd62-ee8e-4c33-a66f-c53589d6f669"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "auvergne-rhone-alpes/rhone/6783fd62-ee8e-4c33-a66f-c53589d6f669": {
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
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Unique value"
  ],
  "value": [
    "auvergne-rhone-alpes/rhone/6783fd62-ee8e-4c33-a66f-c53589d6f669"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "commune in the metropolis of Lyon, France",
    "label": "Irigny"
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
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 007. `repair_Q137274428_2441014897`

| Field | Value |
|---|---|
| qid | Q137274428 |
| property | P244 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q137274428::P244 |
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
| classification_target_tokens | ["2024060501"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "2024060501"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "2024060501"
  ],
  "removed_unique_values": [
    "2024060501"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Mcampany-emco",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "2024060501"
  ],
  "revision_id": 2441014897,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "2024060501": 1
    },
    "old_unique": [
      "2024060501"
    ],
    "old_values": [
      "2024060501"
    ],
    "old_values_raw": [
      "2024060501"
    ],
    "removed_unique_values": [
      "2024060501"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "2024060501": {
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
  "report_fix_date": "2025-12-13T11:08:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P244",
  "report_revision_new": 2441790697,
  "report_revision_old": 2441226306,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "2024060501"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "Library of Congress name authority (persons, families, corporate bodies, events, places, works and expressions) and subject authority identifier [Format: 1-2 specific letters followed by 8-10 digits (see regex). For manifestations, use P1144]",
    "label": "Library of Congress authority ID"
  },
  "qid": {
    "description": "2025 book by Tim Wu",
    "label": "The Age of Extraction"
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
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 008. `repair_Q137295996_2439998315`

| Field | Value |
|---|---|
| qid | Q137295996 |
| property | P2397 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q137295996::P2397 |
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
| classification_target_tokens | ["RukayatuIssaka1996"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "RukayatuIssaka1996"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "RukayatuIssaka1996"
  ],
  "removed_unique_values": [
    "RukayatuIssaka1996"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Mbchbot",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "RukayatuIssaka1996"
  ],
  "revision_id": 2439998315,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "RukayatuIssaka1996": 1
    },
    "old_unique": [
      "RukayatuIssaka1996"
    ],
    "old_values": [
      "RukayatuIssaka1996"
    ],
    "old_values_raw": [
      "RukayatuIssaka1996"
    ],
    "removed_unique_values": [
      "RukayatuIssaka1996"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "RukayatuIssaka1996": {
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
  "report_fix_date": "2025-12-11T10:07:09",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2397",
  "report_revision_new": 2440798845,
  "report_revision_old": 2440379958,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Type Q|1067164, Q|5, Q|43229, Q|16334295, Q|8513, Q|386724, Q|1190554, Q|4164871, Q|17558136, Q|618779, Q|2001305, Q|95074, Q|24634210, Q|102345381, Q|43501, Q|24354, Q|431289, Q|2281788, Q|16970, Q|1060829, Q|340169, Q|2906862, Q|13226383, Q|16887380, Q|7406919, Q|56061, Q|14514600, Q|55155641, Q|2424752, Q|79913, Q|170584"
  ],
  "value": [
    "RukayatuIssaka1996"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "ID of the YouTube channel of a person or organisation (not to be confused with the name of the channel) The ID can also be used for music.youtube.com IDs",
    "label": "YouTube channel ID"
  },
  "qid": {
    "description": "Ghanaian marketing strategist and Marketing Director at Adam Ro Music Ltd",
    "label": "Rukayatu Issaka"
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
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 009. `repair_Q137473626_2447321878`

| Field | Value |
|---|---|
| qid | Q137473626 |
| property | P244 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q137473626::P244 |
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
| classification_target_tokens | ["2002141599"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "2002141599"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "2002141599"
  ],
  "removed_unique_values": [
    "2002141599"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Dcflyer",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "2002141599"
  ],
  "revision_id": 2447321878,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "2002141599": 1
    },
    "old_unique": [
      "2002141599"
    ],
    "old_values": [
      "2002141599"
    ],
    "old_values_raw": [
      "2002141599"
    ],
    "removed_unique_values": [
      "2002141599"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "2002141599": {
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
  "report_fix_date": "2025-12-27T13:14:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P244",
  "report_revision_new": 2447786198,
  "report_revision_old": 2447395127,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "2002141599"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "Library of Congress name authority (persons, families, corporate bodies, events, places, works and expressions) and subject authority identifier [Format: 1-2 specific letters followed by 8-10 digits (see regex). For manifestations, use P1144]",
    "label": "Library of Congress authority ID"
  },
  "qid": {
    "description": "2003 book by Lizabeth Cohen",
    "label": "Consumers' Republic: The Politics of Mass Consumption in Postwar America"
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
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 010. `repair_Q137530083_2445489249`

| Field | Value |
|---|---|
| qid | Q137530083 |
| property | P2397 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q137530083::P2397 |
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
| classification_target_tokens | ["https://www.youtube.com/@danylsemmache"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "https://www.youtube.com/@danylsemmache"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "https://www.youtube.com/@danylsemmache"
  ],
  "removed_unique_values": [
    "https://www.youtube.com/@danylsemmache"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Mbchbot",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "https://www.youtube.com/@danylsemmache"
  ],
  "revision_id": 2445489249,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "https://www.youtube.com/@danylsemmache": 1
    },
    "old_unique": [
      "https://www.youtube.com/@danylsemmache"
    ],
    "old_values": [
      "https://www.youtube.com/@danylsemmache"
    ],
    "old_values_raw": [
      "https://www.youtube.com/@danylsemmache"
    ],
    "removed_unique_values": [
      "https://www.youtube.com/@danylsemmache"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "https://www.youtube.com/@danylsemmache": {
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
  "report_fix_date": "2025-12-24T09:07:46",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2397",
  "report_revision_new": 2446436781,
  "report_revision_old": 2445971129,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "https://www.youtube.com/@danylsemmache"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "ID of the YouTube channel of a person or organisation (not to be confused with the name of the channel) The ID can also be used for music.youtube.com IDs",
    "label": "YouTube channel ID"
  },
  "qid": {
    "description": "Entrepreneur",
    "label": "Danyl Semmache"
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
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 011. `repair_Q137552005_2446015904`

| Field | Value |
|---|---|
| qid | Q137552005 |
| property | P2397 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q137552005::P2397 |
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
| classification_target_tokens | ["officialsanyogitatyagi"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "officialsanyogitatyagi"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "officialsanyogitatyagi"
  ],
  "removed_unique_values": [
    "officialsanyogitatyagi"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Mbchbot",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "officialsanyogitatyagi"
  ],
  "revision_id": 2446015904,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "officialsanyogitatyagi": 1
    },
    "old_unique": [
      "officialsanyogitatyagi"
    ],
    "old_values": [
      "officialsanyogitatyagi"
    ],
    "old_values_raw": [
      "officialsanyogitatyagi"
    ],
    "removed_unique_values": [
      "officialsanyogitatyagi"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "officialsanyogitatyagi": {
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
  "report_fix_date": "2025-12-25T16:25:00",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2397",
  "report_revision_new": 2447010208,
  "report_revision_old": 2446436781,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "officialsanyogitatyagi"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "ID of the YouTube channel of a person or organisation (not to be confused with the name of the channel) The ID can also be used for music.youtube.com IDs",
    "label": "YouTube channel ID"
  },
  "qid": {
    "description": "Actor, Producer, Director, Voice-over artist with decades of experience in the industry.",
    "label": "Sanyogita Tyagi"
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
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 012. `repair_Q1509115_2297181722`

| Field | Value |
|---|---|
| qid | Q1509115 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q1509115::P8748 |
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
| classification_target_tokens | ["ps00520"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "ps00520"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "ps00520"
  ],
  "removed_unique_values": [
    "ps00520"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "ps00520"
  ],
  "revision_id": 2297181722,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "ps00520": 1
    },
    "old_unique": [
      "ps00520"
    ],
    "old_values": [
      "ps00520"
    ],
    "old_values_raw": [
      "ps00520"
    ],
    "removed_unique_values": [
      "ps00520"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "ps00520": {
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
    "ps00520"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German archeologist and geographer (1886–1945)",
    "label": "Albert Herrmann"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 013. `repair_Q184918_2433408800`

| Field | Value |
|---|---|
| qid | Q184918 |
| property | P487 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q184918::P487 |
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
| classification_target_tokens | ["\\n"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "\\n"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "\\n"
  ],
  "removed_unique_values": [
    "\\n"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Drmccreedy",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "\\n"
  ],
  "revision_id": 2433408800,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "\\n": 1
    },
    "old_unique": [
      "\\n"
    ],
    "old_values": [
      "\\n"
    ],
    "old_values_raw": [
      "\\n"
    ],
    "removed_unique_values": [
      "\\n"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "\\n": {
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
  "report_fix_date": "2025-11-26T11:03:41",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P487",
  "report_revision_new": 2434210094,
  "report_revision_old": 2433724519,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "\\n"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "Unicode character representing the item (only if this is not a control character or a compatiblity character: in that case, use only P4213)",
    "label": "Unicode character"
  },
  "qid": {
    "description": "control character indicating the end of a line in a text file, considered as one of the 'blank characters' in a character set",
    "label": "line feed"
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
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 014. `repair_Q1905223_2297178941`

| Field | Value |
|---|---|
| qid | Q1905223 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q1905223::P8748 |
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
| classification_target_tokens | ["pk00742"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "pk00742"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pk00742"
  ],
  "removed_unique_values": [
    "pk00742"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "pk00742"
  ],
  "revision_id": 2297178941,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "pk00742": 1
    },
    "old_unique": [
      "pk00742"
    ],
    "old_values": [
      "pk00742"
    ],
    "old_values_raw": [
      "pk00742"
    ],
    "removed_unique_values": [
      "pk00742"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "pk00742": {
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
    "pk00742"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German musician",
    "label": "Martin Weller"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 015. `repair_Q216694_2297179307`

| Field | Value |
|---|---|
| qid | Q216694 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q216694::P8748 |
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
| classification_target_tokens | ["pkc0149"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "pkc0149"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pkc0149"
  ],
  "removed_unique_values": [
    "pkc0149"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "pkc0149"
  ],
  "revision_id": 2297179307,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "pkc0149": 1
    },
    "old_unique": [
      "pkc0149"
    ],
    "old_values": [
      "pkc0149"
    ],
    "old_values_raw": [
      "pkc0149"
    ],
    "removed_unique_values": [
      "pkc0149"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "pkc0149": {
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
    "pkc0149"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German judge and politician (1892-1945)",
    "label": "Dietrich von Jagow"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 016. `repair_Q23000600_2297183062`

| Field | Value |
|---|---|
| qid | Q23000600 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q23000600::P8748 |
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
| classification_target_tokens | ["pka0361"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "pka0361"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pka0361"
  ],
  "removed_unique_values": [
    "pka0361"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "pka0361"
  ],
  "revision_id": 2297183062,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "pka0361": 1
    },
    "old_unique": [
      "pka0361"
    ],
    "old_values": [
      "pka0361"
    ],
    "old_values_raw": [
      "pka0361"
    ],
    "removed_unique_values": [
      "pka0361"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "pka0361": {
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
    "pka0361"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "kunstschilder",
    "label": "Heinrich Hartung IV."
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 017. `repair_Q386122_2422271267`

| Field | Value |
|---|---|
| qid | Q386122 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q386122::P6671 |
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
| classification_target_tokens | ["auvergne-rhone-alpes/rhone/0d4d58f0-8805-464a-8617-21b3d4cf1014"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "auvergne-rhone-alpes/rhone/0d4d58f0-8805-464a-8617-21b3d4cf1014"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "auvergne-rhone-alpes/rhone/0d4d58f0-8805-464a-8617-21b3d4cf1014"
  ],
  "removed_unique_values": [
    "auvergne-rhone-alpes/rhone/0d4d58f0-8805-464a-8617-21b3d4cf1014"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "auvergne-rhone-alpes/rhone/0d4d58f0-8805-464a-8617-21b3d4cf1014"
  ],
  "revision_id": 2422271267,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "auvergne-rhone-alpes/rhone/0d4d58f0-8805-464a-8617-21b3d4cf1014": 1
    },
    "old_unique": [
      "auvergne-rhone-alpes/rhone/0d4d58f0-8805-464a-8617-21b3d4cf1014"
    ],
    "old_values": [
      "auvergne-rhone-alpes/rhone/0d4d58f0-8805-464a-8617-21b3d4cf1014"
    ],
    "old_values_raw": [
      "auvergne-rhone-alpes/rhone/0d4d58f0-8805-464a-8617-21b3d4cf1014"
    ],
    "removed_unique_values": [
      "auvergne-rhone-alpes/rhone/0d4d58f0-8805-464a-8617-21b3d4cf1014"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "auvergne-rhone-alpes/rhone/0d4d58f0-8805-464a-8617-21b3d4cf1014": {
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
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Unique value"
  ],
  "value": [
    "auvergne-rhone-alpes/rhone/0d4d58f0-8805-464a-8617-21b3d4cf1014"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "commune in the metropolis of Lyon, France",
    "label": "Rillieux-la-Pape"
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
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 018. `repair_Q47087898_2416380397`

| Field | Value |
|---|---|
| qid | Q47087898 |
| property | P4244 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q47087898::P4244 |
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
| classification_target_tokens | ["ODB_S00054674"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "ODB_S00054674"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "ODB_S00054674"
  ],
  "removed_unique_values": [
    "ODB_S00054674"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Z thomas",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "ODB_S00054674"
  ],
  "revision_id": 2416380397,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "ODB_S00054674": 1
    },
    "old_unique": [
      "ODB_S00054674"
    ],
    "old_values": [
      "ODB_S00054674"
    ],
    "old_values_raw": [
      "ODB_S00054674"
    ],
    "removed_unique_values": [
      "ODB_S00054674"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "ODB_S00054674": {
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
  "report_fix_date": "2025-10-16T09:11:18",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4244",
  "report_revision_new": 2417079931,
  "report_revision_old": 2416692506,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Item P|1435 one of Q|17297633, Q|97154904, Q|97155914, Q|98100466, Q|104528530, Q|106541984"
  ],
  "value": [
    "ODB_S00054674"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "identifier for cultural heritage monuments (ensembles, buildings and grounds) in Bavaria, issued by the Bavarian State Office for the Preservation of Monuments",
    "label": "Bavarian monument authority ID"
  },
  "qid": {
    "description": "human settlement in Germany",
    "label": "Felsenkeller (Wörnitz)"
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
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 019. `repair_Q55676663_2297173046`

| Field | Value |
|---|---|
| qid | Q55676663 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q55676663::P8748 |
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
| classification_target_tokens | ["pka0925"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "pka0925"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pka0925"
  ],
  "removed_unique_values": [
    "pka0925"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "pka0925"
  ],
  "revision_id": 2297173046,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "pka0925": 1
    },
    "old_unique": [
      "pka0925"
    ],
    "old_values": [
      "pka0925"
    ],
    "old_values_raw": [
      "pka0925"
    ],
    "removed_unique_values": [
      "pka0925"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "pka0925": {
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
    "pka0925"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "Dt. Kommunikationswissenschaftler",
    "label": "Günther Wohlers"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 020. `repair_Q55681957_2297169871`

| Field | Value |
|---|---|
| qid | Q55681957 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q55681957::P8748 |
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
| classification_target_tokens | ["pk01109"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "pk01109"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pk01109"
  ],
  "removed_unique_values": [
    "pk01109"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "pk01109"
  ],
  "revision_id": 2297169871,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "pk01109": 1
    },
    "old_unique": [
      "pk01109"
    ],
    "old_values": [
      "pk01109"
    ],
    "old_values_raw": [
      "pk01109"
    ],
    "removed_unique_values": [
      "pk01109"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "pk01109": {
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
    "pk01109"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "Prussian District Administrator",
    "label": "Hans Karl Heuberger"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 021. `repair_Q71252_2297168661`

| Field | Value |
|---|---|
| qid | Q71252 |
| property | P8748 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q71252::P8748 |
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
| classification_target_tokens | ["pkc0027"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "pkc0027"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "pkc0027"
  ],
  "removed_unique_values": [
    "pkc0027"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Philippe Kayser",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "pkc0027"
  ],
  "revision_id": 2297168661,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "pkc0027": 1
    },
    "old_unique": [
      "pkc0027"
    ],
    "old_values": [
      "pkc0027"
    ],
    "old_values_raw": [
      "pkc0027"
    ],
    "removed_unique_values": [
      "pkc0027"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "pkc0027": {
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
    "pkc0027"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "static identifier (\"Zitierlink\") of an entry in Rheinland-Pfälzische Personendatenbank (the \"recnums\" is not stable!)",
    "label": "Rheinland-Pfälzische Personendatenbank (GND) ID"
  },
  "qid": {
    "description": "German general (1894–1943)",
    "label": "Theodor Berkelmann"
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
    "result": true,
    "step": "is_delete"
  },
  {
    "detail": {
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 022. `repair_Q7395508_2446370905`

| Field | Value |
|---|---|
| qid | Q7395508 |
| property | P154 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q7395508::P154 |
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
| classification_target_tokens | ["Coat of arms of Sa Đéc Province.svg"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "Coat of arms of Sa Đéc Province.svg"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Coat of arms of Sa Đéc Province.svg"
  ],
  "removed_unique_values": [
    "Coat of arms of Sa Đéc Province.svg"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Henrydat",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "Coat of arms of Sa Đéc Province.svg"
  ],
  "revision_id": 2446370905,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Coat of arms of Sa Đéc Province.svg": 1
    },
    "old_unique": [
      "Coat of arms of Sa Đéc Province.svg"
    ],
    "old_values": [
      "Coat of arms of Sa Đéc Province.svg"
    ],
    "old_values_raw": [
      "Coat of arms of Sa Đéc Province.svg"
    ],
    "removed_unique_values": [
      "Coat of arms of Sa Đéc Province.svg"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "Coat of arms of Sa Đéc Province.svg": {
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
  "report_fix_date": "2025-12-25T21:15:00",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P154",
  "report_revision_new": 2447111397,
  "report_revision_old": 2446569067,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "Coat of arms of Sa Đéc Province.svg"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "graphic mark or emblem commonly used by commercial enterprises, organizations and products",
    "label": "logo image"
  },
  "qid": {
    "description": "province of the Republic of Vietnam",
    "label": "Sa Đéc"
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
    "label_en": "Commons link constraint",
    "qid": "Q21510852"
  },
  {
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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

## 023. `repair_Q840195_2422269778`

| Field | Value |
|---|---|
| qid | Q840195 |
| property | P6671 |
| track | A_BOX |
| class / subtype / confidence | TypeA / REJECTION_FORMAT_INVALID / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_rejection_format_invalid |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| classification_rule_family | format |
| classification_rule_subfamily | rejection_format_invalid |
| decision_constraint_type | Q21502404 format constraint |
| group_key | ABOX::Q840195::P6671 |
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
| classification_target_tokens | ["auvergne-rhone-alpes/rhone/937ec7bd-f476-40fc-9b3a-ce47220d4d43"] |
| classification_target_reason | delete-to-missing is explained by deleted old values |
| decision_branch | delete_refined |
| rationale | A-Box DELETE removes a value reported as a format violation. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "delete-to-missing is explained by deleted old values",
  "classification_target_role": "removed",
  "classification_target_tokens": [
    "auvergne-rhone-alpes/rhone/937ec7bd-f476-40fc-9b3a-ce47220d4d43"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "auvergne-rhone-alpes/rhone/937ec7bd-f476-40fc-9b3a-ce47220d4d43"
  ],
  "removed_unique_values": [
    "auvergne-rhone-alpes/rhone/937ec7bd-f476-40fc-9b3a-ce47220d4d43"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "DELETE_TO_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "format",
  "classification_rule_subfamily": "rejection_format_invalid",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "format constraint",
  "decision_constraint_type_qid": "Q21502404"
}
```

#### Repair Target

```json
{
  "action": "DELETE",
  "author": "Arpyia",
  "kind": "A_BOX",
  "new_value": [
    "MISSING"
  ],
  "old_value": [
    "auvergne-rhone-alpes/rhone/937ec7bd-f476-40fc-9b3a-ce47220d4d43"
  ],
  "revision_id": 2422269778,
  "value": [
    "MISSING"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {},
    "new_unique": [],
    "new_values": [],
    "new_values_raw": [
      "MISSING"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "auvergne-rhone-alpes/rhone/937ec7bd-f476-40fc-9b3a-ce47220d4d43": 1
    },
    "old_unique": [
      "auvergne-rhone-alpes/rhone/937ec7bd-f476-40fc-9b3a-ce47220d4d43"
    ],
    "old_values": [
      "auvergne-rhone-alpes/rhone/937ec7bd-f476-40fc-9b3a-ce47220d4d43"
    ],
    "old_values_raw": [
      "auvergne-rhone-alpes/rhone/937ec7bd-f476-40fc-9b3a-ce47220d4d43"
    ],
    "removed_unique_values": [
      "auvergne-rhone-alpes/rhone/937ec7bd-f476-40fc-9b3a-ce47220d4d43"
    ],
    "retained_unique_values": [],
    "semantic_action": "DELETE_TO_MISSING",
    "value_multiplicity_changes": {
      "auvergne-rhone-alpes/rhone/937ec7bd-f476-40fc-9b3a-ce47220d4d43": {
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
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "report_violation_types": [
    "Format",
    "Unique value"
  ],
  "value": [
    "auvergne-rhone-alpes/rhone/937ec7bd-f476-40fc-9b3a-ce47220d4d43"
  ]
}
```

### Local Evidence

```json
{
  "found": null,
  "local_availability_result": false,
  "local_ids_count": null,
  "local_support_for_retained_value": [],
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
    "description": "commune in the metropolis of Lyon, France",
    "label": "Albigny-sur-Saône"
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
      "delete_reason": "format_invalid_report",
      "report_type": "format"
    },
    "result": "REJECTION_FORMAT_INVALID",
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
