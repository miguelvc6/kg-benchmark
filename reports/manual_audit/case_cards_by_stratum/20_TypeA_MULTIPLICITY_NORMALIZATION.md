# TypeA_MULTIPLICITY_NORMALIZATION

Cases: 10

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q105_2291217826`

| Field | Value |
|---|---|
| qid | Q105 |
| property | P2888 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | head |
| constraint_family | Q21502410 |
| classification_rule_family | multiplicity |
| classification_rule_subfamily | multiplicity_normalization |
| decision_constraint_type | Q19474404 single-value constraint |
| group_key | ABOX::Q105::P2888 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["http://dati.beniculturali.it/cis/Monday", "https://schema.org/Monday", "http://www.w3.org/2006/time#Monday"] |
| classification_target_tokens | ["http://dati.beniculturali.it/cis/Monday", "http://www.w3.org/2006/time#Monday", "https://schema.org/Monday"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "unique values are unchanged; only multiplicity changed",
  "classification_target_role": "multiplicity",
  "classification_target_tokens": [
    "http://dati.beniculturali.it/cis/Monday",
    "http://www.w3.org/2006/time#Monday",
    "https://schema.org/Monday"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "http://dati.beniculturali.it/cis/Monday",
    "http://www.w3.org/2006/time#Monday",
    "https://schema.org/Monday"
  ],
  "retained_unique_values": [
    "http://dati.beniculturali.it/cis/Monday",
    "http://www.w3.org/2006/time#Monday",
    "https://schema.org/Monday"
  ],
  "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "multiplicity",
  "classification_rule_subfamily": "multiplicity_normalization",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "http://dati.beniculturali.it/cis/Monday",
    "https://schema.org/Monday",
    "http://www.w3.org/2006/time#Monday"
  ],
  "old_value": [
    "http://dati.beniculturali.it/cis/Monday",
    "https://schema.org/Monday",
    "http://www.w3.org/2006/time#Monday",
    "https://schema.org/Monday"
  ],
  "revision_id": 2291217826,
  "value": [
    "http://dati.beniculturali.it/cis/Monday",
    "https://schema.org/Monday",
    "http://www.w3.org/2006/time#Monday"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "http://dati.beniculturali.it/cis/Monday": 1,
      "http://www.w3.org/2006/time#Monday": 1,
      "https://schema.org/Monday": 1
    },
    "new_unique": [
      "http://dati.beniculturali.it/cis/Monday",
      "http://www.w3.org/2006/time#Monday",
      "https://schema.org/Monday"
    ],
    "new_values": [
      "http://dati.beniculturali.it/cis/Monday",
      "https://schema.org/Monday",
      "http://www.w3.org/2006/time#Monday"
    ],
    "new_values_raw": [
      "http://dati.beniculturali.it/cis/Monday",
      "https://schema.org/Monday",
      "http://www.w3.org/2006/time#Monday"
    ],
    "normalized_unique_values_unchanged": true,
    "old_counts": {
      "http://dati.beniculturali.it/cis/Monday": 1,
      "http://www.w3.org/2006/time#Monday": 1,
      "https://schema.org/Monday": 2
    },
    "old_unique": [
      "http://dati.beniculturali.it/cis/Monday",
      "http://www.w3.org/2006/time#Monday",
      "https://schema.org/Monday"
    ],
    "old_values": [
      "http://dati.beniculturali.it/cis/Monday",
      "https://schema.org/Monday",
      "http://www.w3.org/2006/time#Monday",
      "https://schema.org/Monday"
    ],
    "old_values_raw": [
      "http://dati.beniculturali.it/cis/Monday",
      "https://schema.org/Monday",
      "http://www.w3.org/2006/time#Monday",
      "https://schema.org/Monday"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "http://dati.beniculturali.it/cis/Monday",
      "http://www.w3.org/2006/time#Monday",
      "https://schema.org/Monday"
    ],
    "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "value_multiplicity_changes": {
      "https://schema.org/Monday": {
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
  "report_fix_date": "2024-12-26T08:10:02",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2888",
  "report_revision_new": 2291405740,
  "report_revision_old": 2290949078,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "http://dati.beniculturali.it/cis/Monday",
    "https://schema.org/Monday",
    "http://www.w3.org/2006/time#Monday",
    "https://schema.org/Monday"
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
    "http://dati.beniculturali.it/cis/Monday",
    "https://schema.org/Monday",
    "http://www.w3.org/2006/time#Monday"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "URL used to link two items or properties, indicating a high degree of confidence that the concepts can be used interchangeably",
    "label": "exact match"
  },
  "qid": {
    "description": "day of the week",
    "label": "Monday"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
        "http://dati.beniculturali.it/cis/Monday": 1,
        "http://www.w3.org/2006/time#Monday": 1,
        "https://schema.org/Monday": 1
      },
      "new_unique": [
        "http://dati.beniculturali.it/cis/Monday",
        "http://www.w3.org/2006/time#Monday",
        "https://schema.org/Monday"
      ],
      "new_values": [
        "http://dati.beniculturali.it/cis/Monday",
        "https://schema.org/Monday",
        "http://www.w3.org/2006/time#Monday"
      ],
      "new_values_raw": [
        "http://dati.beniculturali.it/cis/Monday",
        "https://schema.org/Monday",
        "http://www.w3.org/2006/time#Monday"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "http://dati.beniculturali.it/cis/Monday": 1,
        "http://www.w3.org/2006/time#Monday": 1,
        "https://schema.org/Monday": 2
      },
      "old_unique": [
        "http://dati.beniculturali.it/cis/Monday",
        "http://www.w3.org/2006/time#Monday",
        "https://schema.org/Monday"
      ],
      "old_values": [
        "http://dati.beniculturali.it/cis/Monday",
        "https://schema.org/Monday",
        "http://www.w3.org/2006/time#Monday",
        "https://schema.org/Monday"
      ],
      "old_values_raw": [
        "http://dati.beniculturali.it/cis/Monday",
        "https://schema.org/Monday",
        "http://www.w3.org/2006/time#Monday",
        "https://schema.org/Monday"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "http://dati.beniculturali.it/cis/Monday",
        "http://www.w3.org/2006/time#Monday",
        "https://schema.org/Monday"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "https://schema.org/Monday": {
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

## 002. `repair_Q114656709_2443894763`

| Field | Value |
|---|---|
| qid | Q114656709 |
| property | P648 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | multiplicity |
| classification_rule_subfamily | multiplicity_normalization |
| decision_constraint_type | Q19474404 single-value constraint |
| group_key | ABOX::Q114656709::P648 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["OL2827598A"] |
| classification_target_tokens | ["OL2827598A"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "unique values are unchanged; only multiplicity changed",
  "classification_target_role": "multiplicity",
  "classification_target_tokens": [
    "OL2827598A"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "OL2827598A"
  ],
  "retained_unique_values": [
    "OL2827598A"
  ],
  "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "multiplicity",
  "classification_rule_subfamily": "multiplicity_normalization",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "InventaireBot",
  "kind": "A_BOX",
  "new_value": [
    "OL2827598A"
  ],
  "old_value": [
    "OL2827598A",
    "OL2827598A"
  ],
  "revision_id": 2443894763,
  "value": [
    "OL2827598A"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "OL2827598A": 1
    },
    "new_unique": [
      "OL2827598A"
    ],
    "new_values": [
      "OL2827598A"
    ],
    "new_values_raw": [
      "OL2827598A"
    ],
    "normalized_unique_values_unchanged": true,
    "old_counts": {
      "OL2827598A": 2
    },
    "old_unique": [
      "OL2827598A"
    ],
    "old_values": [
      "OL2827598A",
      "OL2827598A"
    ],
    "old_values_raw": [
      "OL2827598A",
      "OL2827598A"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "OL2827598A"
    ],
    "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "value_multiplicity_changes": {
      "OL2827598A": {
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
  "report_fix_date": "2025-12-22T09:35:31",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P648",
  "report_revision_new": 2445439961,
  "report_revision_old": 2444875237,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "OL2827598A",
    "OL2827598A"
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
    "OL2827598A"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a work (\"W\"), edition (\"M\") or author (\"A\") for book data of the Internet Archive",
    "label": "Open Library ID"
  },
  "qid": {
    "description": "American author and editor",
    "label": "Leigh Ronald Grossman"
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
        "OL2827598A": 1
      },
      "new_unique": [
        "OL2827598A"
      ],
      "new_values": [
        "OL2827598A"
      ],
      "new_values_raw": [
        "OL2827598A"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "OL2827598A": 2
      },
      "old_unique": [
        "OL2827598A"
      ],
      "old_values": [
        "OL2827598A",
        "OL2827598A"
      ],
      "old_values_raw": [
        "OL2827598A",
        "OL2827598A"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "OL2827598A"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "OL2827598A": {
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

## 003. `repair_Q136738512_2443608164`

| Field | Value |
|---|---|
| qid | Q136738512 |
| property | P5749 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | tail |
| constraint_family | Q52060874 |
| classification_rule_family | multiplicity |
| classification_rule_subfamily | multiplicity_normalization |
| decision_constraint_type | Q19474404 single-value constraint |
| group_key | ABOX::Q136738512::P5749 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["0199336741"] |
| classification_target_tokens | ["0199336741"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "unique values are unchanged; only multiplicity changed",
  "classification_target_role": "multiplicity",
  "classification_target_tokens": [
    "0199336741"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "0199336741"
  ],
  "retained_unique_values": [
    "0199336741"
  ],
  "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "multiplicity",
  "classification_rule_subfamily": "multiplicity_normalization",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "0199336741"
  ],
  "old_value": [
    "0199336741",
    "0199336741"
  ],
  "revision_id": 2443608164,
  "value": [
    "0199336741"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "0199336741": 1
    },
    "new_unique": [
      "0199336741"
    ],
    "new_values": [
      "0199336741"
    ],
    "new_values_raw": [
      "0199336741"
    ],
    "normalized_unique_values_unchanged": true,
    "old_counts": {
      "0199336741": 2
    },
    "old_unique": [
      "0199336741"
    ],
    "old_values": [
      "0199336741",
      "0199336741"
    ],
    "old_values_raw": [
      "0199336741",
      "0199336741"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "0199336741"
    ],
    "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "value_multiplicity_changes": {
      "0199336741": {
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
  "report_fix_date": "2025-12-19T07:32:37",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5749",
  "report_revision_new": 2443988062,
  "report_revision_old": 2443776901,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "report_violation_types": [
    "Single value",
    "Unique value"
  ],
  "value": [
    "0199336741",
    "0199336741"
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
    "0199336741"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a product on Amazon.com websites",
    "label": "Amazon Standard Identification Number"
  },
  "qid": {
    "description": "2016 hardcover edition",
    "label": "The Oxford Handbook of Metamemory"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
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
        "0199336741": 1
      },
      "new_unique": [
        "0199336741"
      ],
      "new_values": [
        "0199336741"
      ],
      "new_values_raw": [
        "0199336741"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "0199336741": 2
      },
      "old_unique": [
        "0199336741"
      ],
      "old_values": [
        "0199336741",
        "0199336741"
      ],
      "old_values_raw": [
        "0199336741",
        "0199336741"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "0199336741"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "0199336741": {
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

## 004. `repair_Q16030740_2441562219`

| Field | Value |
|---|---|
| qid | Q16030740 |
| property | P535 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| classification_rule_family | multiplicity |
| classification_rule_subfamily | multiplicity_normalization |
| decision_constraint_type | Q19474404 single-value constraint |
| group_key | ABOX::Q16030740::P535 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["94950918"] |
| classification_target_tokens | ["94950918"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "unique values are unchanged; only multiplicity changed",
  "classification_target_role": "multiplicity",
  "classification_target_tokens": [
    "94950918"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "94950918"
  ],
  "retained_unique_values": [
    "94950918"
  ],
  "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "multiplicity",
  "classification_rule_subfamily": "multiplicity_normalization",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "94950918"
  ],
  "old_value": [
    "94950918",
    "94950918"
  ],
  "revision_id": 2441562219,
  "value": [
    "94950918"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "94950918": 1
    },
    "new_unique": [
      "94950918"
    ],
    "new_values": [
      "94950918"
    ],
    "new_values_raw": [
      "94950918"
    ],
    "normalized_unique_values_unchanged": true,
    "old_counts": {
      "94950918": 2
    },
    "old_unique": [
      "94950918"
    ],
    "old_values": [
      "94950918",
      "94950918"
    ],
    "old_values_raw": [
      "94950918",
      "94950918"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "94950918"
    ],
    "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "value_multiplicity_changes": {
      "94950918": {
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
  "report_fix_date": "2025-12-14T11:23:18",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P535",
  "report_revision_new": 2442270781,
  "report_revision_old": 2441762179,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "report_violation_types": [
    "Single value",
    "Unique value"
  ],
  "value": [
    "94950918",
    "94950918"
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
    "94950918"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier of an individual's burial place in the Find a Grave database",
    "label": "Find a Grave memorial ID"
  },
  "qid": {
    "description": "American manufacturer and inventor (1836-1903)",
    "label": "Eliphalet Williams Bliss"
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
        "94950918": 1
      },
      "new_unique": [
        "94950918"
      ],
      "new_values": [
        "94950918"
      ],
      "new_values_raw": [
        "94950918"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "94950918": 2
      },
      "old_unique": [
        "94950918"
      ],
      "old_values": [
        "94950918",
        "94950918"
      ],
      "old_values_raw": [
        "94950918",
        "94950918"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "94950918"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "94950918": {
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

## 005. `repair_Q18554077_2393607898`

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
| classification_rule_family | multiplicity |
| classification_rule_subfamily | multiplicity_normalization |
| decision_constraint_type | Q19474404 single-value constraint |
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
| classification_target_tokens | ["C0152207"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "unique values are unchanged; only multiplicity changed",
  "classification_target_role": "multiplicity",
  "classification_target_tokens": [
    "C0152207"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "C0152207"
  ],
  "retained_unique_values": [
    "C0152207"
  ],
  "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "multiplicity",
  "classification_rule_subfamily": "multiplicity_normalization",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

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
  "local_support_for_retained_value": [],
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

## 006. `repair_Q18554530_2393608520`

| Field | Value |
|---|---|
| qid | Q18554530 |
| property | P2892 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| classification_rule_family | multiplicity |
| classification_rule_subfamily | multiplicity_normalization |
| decision_constraint_type | Q19474404 single-value constraint |
| group_key | ABOX::Q18554530::P2892 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["C0153427"] |
| classification_target_tokens | ["C0153427"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "unique values are unchanged; only multiplicity changed",
  "classification_target_role": "multiplicity",
  "classification_target_tokens": [
    "C0153427"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "C0153427"
  ],
  "retained_unique_values": [
    "C0153427"
  ],
  "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "multiplicity",
  "classification_rule_subfamily": "multiplicity_normalization",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Mahir256",
  "kind": "A_BOX",
  "new_value": [
    "C0153427"
  ],
  "old_value": [
    "C0153427",
    "C0153427"
  ],
  "revision_id": 2393608520,
  "value": [
    "C0153427"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "C0153427": 1
    },
    "new_unique": [
      "C0153427"
    ],
    "new_values": [
      "C0153427"
    ],
    "new_values_raw": [
      "C0153427"
    ],
    "normalized_unique_values_unchanged": true,
    "old_counts": {
      "C0153427": 2
    },
    "old_unique": [
      "C0153427"
    ],
    "old_values": [
      "C0153427",
      "C0153427"
    ],
    "old_values_raw": [
      "C0153427",
      "C0153427"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "C0153427"
    ],
    "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "value_multiplicity_changes": {
      "C0153427": {
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
    "C0153427",
    "C0153427"
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
    "C0153427"
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
    "description": "human disease",
    "label": "jejunal cancer"
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
        "C0153427": 1
      },
      "new_unique": [
        "C0153427"
      ],
      "new_values": [
        "C0153427"
      ],
      "new_values_raw": [
        "C0153427"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "C0153427": 2
      },
      "old_unique": [
        "C0153427"
      ],
      "old_values": [
        "C0153427",
        "C0153427"
      ],
      "old_values_raw": [
        "C0153427",
        "C0153427"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "C0153427"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "C0153427": {
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

## 007. `repair_Q25292_2442308808`

| Field | Value |
|---|---|
| qid | Q25292 |
| property | P225 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | multiplicity |
| classification_rule_subfamily | multiplicity_normalization |
| decision_constraint_type | Q19474404 single-value constraint |
| group_key | ABOX::Q25292::P225 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Phoenix dactylifera"] |
| classification_target_tokens | ["Phoenix dactylifera"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "unique values are unchanged; only multiplicity changed",
  "classification_target_role": "multiplicity",
  "classification_target_tokens": [
    "Phoenix dactylifera"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "Phoenix dactylifera"
  ],
  "retained_unique_values": [
    "Phoenix dactylifera"
  ],
  "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "multiplicity",
  "classification_rule_subfamily": "multiplicity_normalization",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "Brya",
  "kind": "A_BOX",
  "new_value": [
    "Phoenix dactylifera"
  ],
  "old_value": [
    "Phoenix dactylifera",
    "Phoenix dactylifera"
  ],
  "revision_id": 2442308808,
  "value": [
    "Phoenix dactylifera"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Phoenix dactylifera": 1
    },
    "new_unique": [
      "Phoenix dactylifera"
    ],
    "new_values": [
      "Phoenix dactylifera"
    ],
    "new_values_raw": [
      "Phoenix dactylifera"
    ],
    "normalized_unique_values_unchanged": true,
    "old_counts": {
      "Phoenix dactylifera": 2
    },
    "old_unique": [
      "Phoenix dactylifera"
    ],
    "old_values": [
      "Phoenix dactylifera",
      "Phoenix dactylifera"
    ],
    "old_values_raw": [
      "Phoenix dactylifera",
      "Phoenix dactylifera"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "Phoenix dactylifera"
    ],
    "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "value_multiplicity_changes": {
      "Phoenix dactylifera": {
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
  "report_fix_date": "2025-12-16T12:38:46",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P225",
  "report_revision_new": 2442991823,
  "report_revision_old": 2442670671,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "report_violation_types": [
    "Single value",
    "Unique value"
  ],
  "value": [
    "Phoenix dactylifera",
    "Phoenix dactylifera"
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
    "Phoenix dactylifera"
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
    "description": "palm tree cultivated for its edible sweet fruit",
    "label": "Phoenix dactylifera"
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
        "Phoenix dactylifera": 1
      },
      "new_unique": [
        "Phoenix dactylifera"
      ],
      "new_values": [
        "Phoenix dactylifera"
      ],
      "new_values_raw": [
        "Phoenix dactylifera"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "Phoenix dactylifera": 2
      },
      "old_unique": [
        "Phoenix dactylifera"
      ],
      "old_values": [
        "Phoenix dactylifera",
        "Phoenix dactylifera"
      ],
      "old_values_raw": [
        "Phoenix dactylifera",
        "Phoenix dactylifera"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "Phoenix dactylifera"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "Phoenix dactylifera": {
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

## 008. `repair_Q6034386_2443074497`

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
| classification_rule_family | multiplicity |
| classification_rule_subfamily | multiplicity_normalization |
| decision_constraint_type | Q19474404 single-value constraint |
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
| classification_target_tokens | ["987013321227905171"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "unique values are unchanged; only multiplicity changed",
  "classification_target_role": "multiplicity",
  "classification_target_tokens": [
    "987013321227905171"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "987013321227905171"
  ],
  "retained_unique_values": [
    "987013321227905171"
  ],
  "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "multiplicity",
  "classification_rule_subfamily": "multiplicity_normalization",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

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
  "local_support_for_retained_value": [],
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
| classification_rule_family | multiplicity |
| classification_rule_subfamily | multiplicity_normalization |
| decision_constraint_type | Q19474404 single-value constraint |
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
| classification_target_tokens | ["Oncorhynchus clarkii virginalis"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "unique values are unchanged; only multiplicity changed",
  "classification_target_role": "multiplicity",
  "classification_target_tokens": [
    "Oncorhynchus clarkii virginalis"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "Oncorhynchus clarkii virginalis"
  ],
  "retained_unique_values": [
    "Oncorhynchus clarkii virginalis"
  ],
  "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "multiplicity",
  "classification_rule_subfamily": "multiplicity_normalization",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

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
  "local_support_for_retained_value": [],
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

## 010. `repair_Q9591607_2447297574`

| Field | Value |
|---|---|
| qid | Q9591607 |
| property | P373 |
| track | A_BOX |
| class / subtype / confidence | TypeA / MULTIPLICITY_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_multiplicity_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | multiplicity |
| classification_rule_subfamily | multiplicity_normalization |
| decision_constraint_type | Q19474404 single-value constraint |
| group_key | ABOX::Q9591607::P373 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["Carnivals Producers of São Paulo by samba school"] |
| classification_target_tokens | ["Carnivals Producers of São Paulo by samba school"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | multiplicity_normalization |
| rationale | Unique values are unchanged and duplicate multiplicity decreases. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [],
  "classification_target_reason": "unique values are unchanged; only multiplicity changed",
  "classification_target_role": "multiplicity",
  "classification_target_tokens": [
    "Carnivals Producers of São Paulo by samba school"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "Carnivals Producers of São Paulo by samba school"
  ],
  "retained_unique_values": [
    "Carnivals Producers of São Paulo by samba school"
  ],
  "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "multiplicity",
  "classification_rule_subfamily": "multiplicity_normalization",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": "single-value constraint",
  "decision_constraint_type_qid": "Q19474404"
}
```

#### Repair Target

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Carnivals Producers of São Paulo by samba school"
  ],
  "old_value": [
    "Carnivals Producers of São Paulo by samba school",
    "Carnivals Producers of São Paulo by samba school"
  ],
  "revision_id": 2447297574,
  "value": [
    "Carnivals Producers of São Paulo by samba school"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Carnivals Producers of São Paulo by samba school": 1
    },
    "new_unique": [
      "Carnivals Producers of São Paulo by samba school"
    ],
    "new_values": [
      "Carnivals Producers of São Paulo by samba school"
    ],
    "new_values_raw": [
      "Carnivals Producers of São Paulo by samba school"
    ],
    "normalized_unique_values_unchanged": true,
    "old_counts": {
      "Carnivals Producers of São Paulo by samba school": 2
    },
    "old_unique": [
      "Carnivals Producers of São Paulo by samba school"
    ],
    "old_values": [
      "Carnivals Producers of São Paulo by samba school",
      "Carnivals Producers of São Paulo by samba school"
    ],
    "old_values_raw": [
      "Carnivals Producers of São Paulo by samba school",
      "Carnivals Producers of São Paulo by samba school"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "Carnivals Producers of São Paulo by samba school"
    ],
    "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
    "value_multiplicity_changes": {
      "Carnivals Producers of São Paulo by samba school": {
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
  "report_fix_date": "2025-12-27T12:35:05",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2447772714,
  "report_revision_old": 2447382517,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": [
    "Carnivals Producers of São Paulo by samba school",
    "Carnivals Producers of São Paulo by samba school"
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
    "Carnivals Producers of São Paulo by samba school"
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
    "label": "Category:Carnivals Producers of São Paulo by samba schools"
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
        "Carnivals Producers of São Paulo by samba school": 1
      },
      "new_unique": [
        "Carnivals Producers of São Paulo by samba school"
      ],
      "new_values": [
        "Carnivals Producers of São Paulo by samba school"
      ],
      "new_values_raw": [
        "Carnivals Producers of São Paulo by samba school"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "Carnivals Producers of São Paulo by samba school": 2
      },
      "old_unique": [
        "Carnivals Producers of São Paulo by samba school"
      ],
      "old_values": [
        "Carnivals Producers of São Paulo by samba school",
        "Carnivals Producers of São Paulo by samba school"
      ],
      "old_values_raw": [
        "Carnivals Producers of São Paulo by samba school",
        "Carnivals Producers of São Paulo by samba school"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "Carnivals Producers of São Paulo by samba school"
      ],
      "semantic_action": "MULTIPLICITY_DECREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "Carnivals Producers of São Paulo by samba school": {
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
