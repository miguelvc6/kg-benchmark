# TypeA_FORMAT_NORMALIZATION

Cases: 35

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q113482439_2439266147`

| Field | Value |
|---|---|
| qid | Q113482439 |
| property | P243 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q113482439::P243 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1050609649"] |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Louperibot",
  "kind": "A_BOX",
  "new_value": [
    "1050609649"
  ],
  "old_value": [
    "on1050609649"
  ],
  "revision_id": 2439266147,
  "value": [
    "1050609649"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "1050609649"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "1050609649"
    ],
    "new_value": [
      "1050609649"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "on1050609649"
    ],
    "old_value": [
      "on1050609649"
    ],
    "removed_unique_values": [
      "on1050609649"
    ],
    "value_multiplicity_changes": {
      "1050609649": {
        "new": 1,
        "old": 0
      },
      "on1050609649": {
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
  "report_fix_date": "2025-12-09T12:24:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P243",
  "report_revision_new": 2440010228,
  "report_revision_old": 2439563322,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "on1050609649"
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
    "1050609649"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a bibliographic record in OCLC WorldCat",
    "label": "OCLC number"
  },
  "qid": {
    "description": "book published in 2018",
    "label": "Crypto economy: how blockchain, cryptocurrency, and token-economy are disrupting the financial world"
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
        "1050609649"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "1050609649": 1
      },
      "new_unique": [
        "1050609649"
      ],
      "new_values": [
        "1050609649"
      ],
      "new_values_raw": [
        "1050609649"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "on1050609649": 1
      },
      "old_unique": [
        "on1050609649"
      ],
      "old_values": [
        "on1050609649"
      ],
      "old_values_raw": [
        "on1050609649"
      ],
      "removed_unique_values": [
        "on1050609649"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "1050609649": {
          "new": 1,
          "old": 0
        },
        "on1050609649": {
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
      "new_value": "1050609649",
      "normalization_kind": "strip_alpha_prefix",
      "old_value": "on1050609649"
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

## 002. `repair_Q137423767_2444114469`

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
| decision_branch | rule_deterministic |
| rationale | Rule-deterministic format constraint fix. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Comics artists from Costa Rica"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Comics artists from Costa Rica"
    ],
    "new_value": [
      "Comics artists from Costa Rica"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Comics artists from Costa  Rica"
    ],
    "old_value": [
      "Comics artists from Costa  Rica"
    ],
    "removed_unique_values": [
      "Comics artists from Costa  Rica"
    ],
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

## 003. `repair_Q15345411_2086440704`

| Field | Value |
|---|---|
| qid | Q15345411 |
| property | P3795 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| group_key | ABOX::Q15345411::P3795 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["MICMYR"] |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "MICMYR"
  ],
  "old_value": [
    "MICMYR/"
  ],
  "revision_id": 2086440704,
  "value": [
    "MICMYR"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "MICMYR"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "MICMYR"
    ],
    "new_value": [
      "MICMYR"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MICMYR/"
    ],
    "old_value": [
      "MICMYR/"
    ],
    "removed_unique_values": [
      "MICMYR/"
    ],
    "value_multiplicity_changes": {
      "MICMYR": {
        "new": 1,
        "old": 0
      },
      "MICMYR/": {
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
    "MICMYR/"
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
    "MICMYR"
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
    "label": "Micromeria myrtifolia"
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
        "MICMYR"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "MICMYR": 1
      },
      "new_unique": [
        "MICMYR"
      ],
      "new_values": [
        "MICMYR"
      ],
      "new_values_raw": [
        "MICMYR"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "MICMYR/": 1
      },
      "old_unique": [
        "MICMYR/"
      ],
      "old_values": [
        "MICMYR/"
      ],
      "old_values_raw": [
        "MICMYR/"
      ],
      "removed_unique_values": [
        "MICMYR/"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "MICMYR": {
          "new": 1,
          "old": 0
        },
        "MICMYR/": {
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
      "new_value": "MICMYR",
      "normalization_kind": "strip_trailing_slash",
      "old_value": "MICMYR/"
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

## 004. `repair_Q15489283_2086442287`

| Field | Value |
|---|---|
| qid | Q15489283 |
| property | P3795 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| group_key | ABOX::Q15489283::P3795 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["CUSKOT"] |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "CUSKOT"
  ],
  "old_value": [
    "CUSKOT/"
  ],
  "revision_id": 2086442287,
  "value": [
    "CUSKOT"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "CUSKOT"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "CUSKOT"
    ],
    "new_value": [
      "CUSKOT"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "CUSKOT/"
    ],
    "old_value": [
      "CUSKOT/"
    ],
    "removed_unique_values": [
      "CUSKOT/"
    ],
    "value_multiplicity_changes": {
      "CUSKOT": {
        "new": 1,
        "old": 0
      },
      "CUSKOT/": {
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
    "CUSKOT/"
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
    "CUSKOT"
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
    "label": "Cuscuta kotschyana"
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
        "CUSKOT"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "CUSKOT": 1
      },
      "new_unique": [
        "CUSKOT"
      ],
      "new_values": [
        "CUSKOT"
      ],
      "new_values_raw": [
        "CUSKOT"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "CUSKOT/": 1
      },
      "old_unique": [
        "CUSKOT/"
      ],
      "old_values": [
        "CUSKOT/"
      ],
      "old_values_raw": [
        "CUSKOT/"
      ],
      "removed_unique_values": [
        "CUSKOT/"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "CUSKOT": {
          "new": 1,
          "old": 0
        },
        "CUSKOT/": {
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
      "new_value": "CUSKOT",
      "normalization_kind": "strip_trailing_slash",
      "old_value": "CUSKOT/"
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

## 005. `repair_Q15502554_2086442380`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "ARGUNI"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "ARGUNI"
    ],
    "new_value": [
      "ARGUNI"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "ARGUNI/"
    ],
    "old_value": [
      "ARGUNI/"
    ],
    "removed_unique_values": [
      "ARGUNI/"
    ],
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
      "new_value": "ARGUNI",
      "normalization_kind": "strip_trailing_slash",
      "old_value": "ARGUNI/"
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

## 006. `repair_Q15564178_2086445496`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "CREACU"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "CREACU"
    ],
    "new_value": [
      "CREACU"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "CREACU/"
    ],
    "old_value": [
      "CREACU/"
    ],
    "removed_unique_values": [
      "CREACU/"
    ],
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
      "new_value": "CREACU",
      "normalization_kind": "strip_trailing_slash",
      "old_value": "CREACU/"
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

## 007. `repair_Q16121727_2086447164`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "podalp"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "podalp"
    ],
    "new_value": [
      "podalp"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "podalp/"
    ],
    "old_value": [
      "podalp/"
    ],
    "removed_unique_values": [
      "podalp/"
    ],
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
      "new_value": "podalp",
      "normalization_kind": "strip_trailing_slash",
      "old_value": "podalp/"
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

## 008. `repair_Q161735_2086419003`

| Field | Value |
|---|---|
| qid | Q161735 |
| property | P3795 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q161735::P3795 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["DACGLO"] |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "DACGLO"
  ],
  "old_value": [
    "DACGLO/"
  ],
  "revision_id": 2086419003,
  "value": [
    "DACGLO"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "DACGLO"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "DACGLO"
    ],
    "new_value": [
      "DACGLO"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "DACGLO/"
    ],
    "old_value": [
      "DACGLO/"
    ],
    "removed_unique_values": [
      "DACGLO/"
    ],
    "value_multiplicity_changes": {
      "DACGLO": {
        "new": 1,
        "old": 0
      },
      "DACGLO/": {
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
    "DACGLO/"
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
    "DACGLO"
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
    "label": "Dactylis glomerata"
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
        "DACGLO"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "DACGLO": 1
      },
      "new_unique": [
        "DACGLO"
      ],
      "new_values": [
        "DACGLO"
      ],
      "new_values_raw": [
        "DACGLO"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "DACGLO/": 1
      },
      "old_unique": [
        "DACGLO/"
      ],
      "old_values": [
        "DACGLO/"
      ],
      "old_values_raw": [
        "DACGLO/"
      ],
      "removed_unique_values": [
        "DACGLO/"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "DACGLO": {
          "new": 1,
          "old": 0
        },
        "DACGLO/": {
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
      "new_value": "DACGLO",
      "normalization_kind": "strip_trailing_slash",
      "old_value": "DACGLO/"
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

## 009. `repair_Q20060274_2086450311`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "MEDITA"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "MEDITA"
    ],
    "new_value": [
      "MEDITA"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MEDITA/"
    ],
    "old_value": [
      "MEDITA/"
    ],
    "removed_unique_values": [
      "MEDITA/"
    ],
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
      "new_value": "MEDITA",
      "normalization_kind": "strip_trailing_slash",
      "old_value": "MEDITA/"
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

## 010. `repair_Q2060696_2425219743`

| Field | Value |
|---|---|
| qid | Q2060696 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| group_key | ABOX::Q2060696::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["4708"] |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "4708"
  ],
  "old_value": [
    "SCHEMBL4708"
  ],
  "revision_id": 2425219743,
  "value": [
    "4708"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "4708"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "4708"
    ],
    "new_value": [
      "4708"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL4708"
    ],
    "old_value": [
      "SCHEMBL4708"
    ],
    "removed_unique_values": [
      "SCHEMBL4708"
    ],
    "value_multiplicity_changes": {
      "4708": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL4708": {
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
    "SCHEMBL4708"
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
    "4708"
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
    "label": "pirfenidone"
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
        "4708"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "4708": 1
      },
      "new_unique": [
        "4708"
      ],
      "new_values": [
        "4708"
      ],
      "new_values_raw": [
        "4708"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL4708": 1
      },
      "old_unique": [
        "SCHEMBL4708"
      ],
      "old_values": [
        "SCHEMBL4708"
      ],
      "old_values_raw": [
        "SCHEMBL4708"
      ],
      "removed_unique_values": [
        "SCHEMBL4708"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "4708": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL4708": {
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
      "new_value": "4708",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL4708"
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

## 011. `repair_Q2636110_2425228276`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "141581"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "141581"
    ],
    "new_value": [
      "141581"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL141581"
    ],
    "old_value": [
      "SCHEMBL141581"
    ],
    "removed_unique_values": [
      "SCHEMBL141581"
    ],
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
      "new_value": "141581",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL141581"
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

## 012. `repair_Q374027_2086421946`

| Field | Value |
|---|---|
| qid | Q374027 |
| property | P3795 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | head |
| constraint_family | Q19474404 |
| group_key | ABOX::Q374027::P3795 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | literal |
| truth_tokens_preview | ["ANOCRI"] |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "ANOCRI"
  ],
  "old_value": [
    "ANOCRI/"
  ],
  "revision_id": 2086421946,
  "value": [
    "ANOCRI"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "ANOCRI"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "ANOCRI"
    ],
    "new_value": [
      "ANOCRI"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "ANOCRI/"
    ],
    "old_value": [
      "ANOCRI/"
    ],
    "removed_unique_values": [
      "ANOCRI/"
    ],
    "value_multiplicity_changes": {
      "ANOCRI": {
        "new": 1,
        "old": 0
      },
      "ANOCRI/": {
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
    "ANOCRI/"
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
    "ANOCRI"
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
    "label": "Anoda cristata"
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
        "ANOCRI"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "ANOCRI": 1
      },
      "new_unique": [
        "ANOCRI"
      ],
      "new_values": [
        "ANOCRI"
      ],
      "new_values_raw": [
        "ANOCRI"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "ANOCRI/": 1
      },
      "old_unique": [
        "ANOCRI/"
      ],
      "old_values": [
        "ANOCRI/"
      ],
      "old_values_raw": [
        "ANOCRI/"
      ],
      "removed_unique_values": [
        "ANOCRI/"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "ANOCRI": {
          "new": 1,
          "old": 0
        },
        "ANOCRI/": {
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
      "new_value": "ANOCRI",
      "normalization_kind": "strip_trailing_slash",
      "old_value": "ANOCRI/"
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

## 013. `repair_Q415024_2425330731`

| Field | Value |
|---|---|
| qid | Q415024 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| group_key | ABOX::Q415024::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["27981"] |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "27981"
  ],
  "old_value": [
    "SCHEMBL27981"
  ],
  "revision_id": 2425330731,
  "value": [
    "27981"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "27981"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "27981"
    ],
    "new_value": [
      "27981"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL27981"
    ],
    "old_value": [
      "SCHEMBL27981"
    ],
    "removed_unique_values": [
      "SCHEMBL27981"
    ],
    "value_multiplicity_changes": {
      "27981": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL27981": {
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
    "SCHEMBL27981"
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
    "27981"
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
    "label": "p-phenylenediamine"
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
        "27981"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "27981": 1
      },
      "new_unique": [
        "27981"
      ],
      "new_values": [
        "27981"
      ],
      "new_values_raw": [
        "27981"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL27981": 1
      },
      "old_unique": [
        "SCHEMBL27981"
      ],
      "old_values": [
        "SCHEMBL27981"
      ],
      "old_values_raw": [
        "SCHEMBL27981"
      ],
      "removed_unique_values": [
        "SCHEMBL27981"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "27981": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL27981": {
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
      "new_value": "27981",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL27981"
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

## 014. `repair_Q421322_2425334359`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "34316"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "34316"
    ],
    "new_value": [
      "34316"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL34316"
    ],
    "old_value": [
      "SCHEMBL34316"
    ],
    "removed_unique_values": [
      "SCHEMBL34316"
    ],
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
      "new_value": "34316",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL34316"
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

## 015. `repair_Q422582_2425335103`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "37405"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "37405"
    ],
    "new_value": [
      "37405"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL37405"
    ],
    "old_value": [
      "SCHEMBL37405"
    ],
    "removed_unique_values": [
      "SCHEMBL37405"
    ],
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
      "new_value": "37405",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL37405"
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

## 016. `repair_Q5332578_2086433858`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "ECHADE"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "ECHADE"
    ],
    "new_value": [
      "ECHADE"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "ECHADE/"
    ],
    "old_value": [
      "ECHADE/"
    ],
    "removed_unique_values": [
      "ECHADE/"
    ],
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
      "new_value": "ECHADE",
      "normalization_kind": "strip_trailing_slash",
      "old_value": "ECHADE/"
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

## 017. `repair_Q67192867_2086453916`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "glecor"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "glecor"
    ],
    "new_value": [
      "glecor"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "glecor/"
    ],
    "old_value": [
      "glecor/"
    ],
    "removed_unique_values": [
      "glecor/"
    ],
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
      "new_value": "glecor",
      "normalization_kind": "strip_trailing_slash",
      "old_value": "glecor/"
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

## 018. `repair_Q72461546_2425387483`

| Field | Value |
|---|---|
| qid | Q72461546 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
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
| decision_branch | rule_deterministic |
| rationale | Rule-deterministic format constraint fix. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "214196"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "214196"
    ],
    "new_value": [
      "214196"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL214196"
    ],
    "old_value": [
      "SCHEMBL214196"
    ],
    "removed_unique_values": [
      "SCHEMBL214196"
    ],
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
      "constraint_type": {
        "label": "format constraint",
        "qid": "Q21502404"
      },
      "new_value": "214196",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL214196",
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

## 019. `repair_Q72484825_2425408699`

| Field | Value |
|---|---|
| qid | Q72484825 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q72484825::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["134326"] |
| decision_branch | rule_deterministic |
| rationale | Rule-deterministic format constraint fix. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "134326"
  ],
  "old_value": [
    "SCHEMBL134326"
  ],
  "revision_id": 2425408699,
  "value": [
    "134326"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "134326"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "134326"
    ],
    "new_value": [
      "134326"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL134326"
    ],
    "old_value": [
      "SCHEMBL134326"
    ],
    "removed_unique_values": [
      "SCHEMBL134326"
    ],
    "value_multiplicity_changes": {
      "134326": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL134326": {
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
    "SCHEMBL134326"
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
    "134326"
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
    "label": "DL-4-Hydroxy-3-methoxymandelic acid"
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
      "constraint_type": {
        "label": "format constraint",
        "qid": "Q21502404"
      },
      "new_value": "134326",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL134326",
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

## 020. `repair_Q72491370_2425414361`

| Field | Value |
|---|---|
| qid | Q72491370 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
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
| decision_branch | rule_deterministic |
| rationale | Rule-deterministic format constraint fix. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "122885"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "122885"
    ],
    "new_value": [
      "122885"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL122885"
    ],
    "old_value": [
      "SCHEMBL122885"
    ],
    "removed_unique_values": [
      "SCHEMBL122885"
    ],
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
      "constraint_type": {
        "label": "format constraint",
        "qid": "Q21502404"
      },
      "new_value": "122885",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL122885",
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

## 021. `repair_Q743661_2425443354`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "187397"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "187397"
    ],
    "new_value": [
      "187397"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL187397"
    ],
    "old_value": [
      "SCHEMBL187397"
    ],
    "removed_unique_values": [
      "SCHEMBL187397"
    ],
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
      "new_value": "187397",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL187397"
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

## 022. `repair_Q775073_2425452244`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "44957"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "44957"
    ],
    "new_value": [
      "44957"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL44957"
    ],
    "old_value": [
      "SCHEMBL44957"
    ],
    "removed_unique_values": [
      "SCHEMBL44957"
    ],
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
      "new_value": "44957",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL44957"
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

## 023. `repair_Q82006684_2425465303`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "16322828"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "16322828"
    ],
    "new_value": [
      "16322828"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL16322828"
    ],
    "old_value": [
      "SCHEMBL16322828"
    ],
    "removed_unique_values": [
      "SCHEMBL16322828"
    ],
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
      "new_value": "16322828",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL16322828"
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

## 024. `repair_Q82078299_2425481249`

| Field | Value |
|---|---|
| qid | Q82078299 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q82078299::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["247478"] |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "247478"
  ],
  "old_value": [
    "SCHEMBL247478"
  ],
  "revision_id": 2425481249,
  "value": [
    "247478"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "247478"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "247478"
    ],
    "new_value": [
      "247478"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL247478"
    ],
    "old_value": [
      "SCHEMBL247478"
    ],
    "removed_unique_values": [
      "SCHEMBL247478"
    ],
    "value_multiplicity_changes": {
      "247478": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL247478": {
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
    "SCHEMBL247478"
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
    "247478"
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
    "label": "1-Oxo-1λ⁵-1,5-naphthyridine"
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
        "247478"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "247478": 1
      },
      "new_unique": [
        "247478"
      ],
      "new_values": [
        "247478"
      ],
      "new_values_raw": [
        "247478"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL247478": 1
      },
      "old_unique": [
        "SCHEMBL247478"
      ],
      "old_values": [
        "SCHEMBL247478"
      ],
      "old_values_raw": [
        "SCHEMBL247478"
      ],
      "removed_unique_values": [
        "SCHEMBL247478"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "247478": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL247478": {
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
      "new_value": "247478",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL247478"
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

## 025. `repair_Q82081964_2425481690`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "190428"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "190428"
    ],
    "new_value": [
      "190428"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL190428"
    ],
    "old_value": [
      "SCHEMBL190428"
    ],
    "removed_unique_values": [
      "SCHEMBL190428"
    ],
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
      "new_value": "190428",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL190428"
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

## 026. `repair_Q82090948_2425483739`

| Field | Value |
|---|---|
| qid | Q82090948 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q82090948::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["44063"] |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "44063"
  ],
  "old_value": [
    "SCHEMBL44063"
  ],
  "revision_id": 2425483739,
  "value": [
    "44063"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "44063"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "44063"
    ],
    "new_value": [
      "44063"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL44063"
    ],
    "old_value": [
      "SCHEMBL44063"
    ],
    "removed_unique_values": [
      "SCHEMBL44063"
    ],
    "value_multiplicity_changes": {
      "44063": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL44063": {
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
    "SCHEMBL44063"
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
    "44063"
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
    "label": "3-Aminocyclopentanecarboxylic acid"
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
        "44063"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "44063": 1
      },
      "new_unique": [
        "44063"
      ],
      "new_values": [
        "44063"
      ],
      "new_values_raw": [
        "44063"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL44063": 1
      },
      "old_unique": [
        "SCHEMBL44063"
      ],
      "old_values": [
        "SCHEMBL44063"
      ],
      "old_values_raw": [
        "SCHEMBL44063"
      ],
      "removed_unique_values": [
        "SCHEMBL44063"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "44063": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL44063": {
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
      "new_value": "44063",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL44063"
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

## 027. `repair_Q82091736_2425483914`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "223337"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "223337"
    ],
    "new_value": [
      "223337"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL223337"
    ],
    "old_value": [
      "SCHEMBL223337"
    ],
    "removed_unique_values": [
      "SCHEMBL223337"
    ],
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
      "new_value": "223337",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL223337"
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

## 028. `repair_Q82101968_2425487135`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "11984337"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "11984337"
    ],
    "new_value": [
      "11984337"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL11984337"
    ],
    "old_value": [
      "SCHEMBL11984337"
    ],
    "removed_unique_values": [
      "SCHEMBL11984337"
    ],
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
      "new_value": "11984337",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL11984337"
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

## 029. `repair_Q82101982_2425487170`

| Field | Value |
|---|---|
| qid | Q82101982 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q82101982::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["443280"] |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "443280"
  ],
  "old_value": [
    "SCHEMBL443280"
  ],
  "revision_id": 2425487170,
  "value": [
    "443280"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "443280"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "443280"
    ],
    "new_value": [
      "443280"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL443280"
    ],
    "old_value": [
      "SCHEMBL443280"
    ],
    "removed_unique_values": [
      "SCHEMBL443280"
    ],
    "value_multiplicity_changes": {
      "443280": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL443280": {
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
    "SCHEMBL443280"
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
    "443280"
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
    "label": "Benzenamine, N-[(4-methylphenyl)methylene]-"
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
        "443280"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "443280": 1
      },
      "new_unique": [
        "443280"
      ],
      "new_values": [
        "443280"
      ],
      "new_values_raw": [
        "443280"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL443280": 1
      },
      "old_unique": [
        "SCHEMBL443280"
      ],
      "old_values": [
        "SCHEMBL443280"
      ],
      "old_values_raw": [
        "SCHEMBL443280"
      ],
      "removed_unique_values": [
        "SCHEMBL443280"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "443280": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL443280": {
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
      "new_value": "443280",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL443280"
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

## 030. `repair_Q82107958_2425489486`

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
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "490162"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "490162"
    ],
    "new_value": [
      "490162"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL490162"
    ],
    "old_value": [
      "SCHEMBL490162"
    ],
    "removed_unique_values": [
      "SCHEMBL490162"
    ],
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
      "new_value": "490162",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL490162"
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

## 031. `repair_Q82114718_2425492935`

| Field | Value |
|---|---|
| qid | Q82114718 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | ABOX::Q82114718::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["276645"] |
| decision_branch | format_normalization |
| rationale | One-to-one literal update is a deterministic format normalization. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "276645"
  ],
  "old_value": [
    "SCHEMBL276645"
  ],
  "revision_id": 2425492935,
  "value": [
    "276645"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "276645"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "276645"
    ],
    "new_value": [
      "276645"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL276645"
    ],
    "old_value": [
      "SCHEMBL276645"
    ],
    "removed_unique_values": [
      "SCHEMBL276645"
    ],
    "value_multiplicity_changes": {
      "276645": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL276645": {
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
    "SCHEMBL276645"
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
    "276645"
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
    "label": "6-Quinazolinol"
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
        "276645"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "276645": 1
      },
      "new_unique": [
        "276645"
      ],
      "new_values": [
        "276645"
      ],
      "new_values_raw": [
        "276645"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "SCHEMBL276645": 1
      },
      "old_unique": [
        "SCHEMBL276645"
      ],
      "old_values": [
        "SCHEMBL276645"
      ],
      "old_values_raw": [
        "SCHEMBL276645"
      ],
      "removed_unique_values": [
        "SCHEMBL276645"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "276645": {
          "new": 1,
          "old": 0
        },
        "SCHEMBL276645": {
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
      "new_value": "276645",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL276645"
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

## 032. `repair_Q82893540_2425667305`

| Field | Value |
|---|---|
| qid | Q82893540 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
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
| decision_branch | rule_deterministic |
| rationale | Rule-deterministic format constraint fix. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "124166"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "124166"
    ],
    "new_value": [
      "124166"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL124166"
    ],
    "old_value": [
      "SCHEMBL124166"
    ],
    "removed_unique_values": [
      "SCHEMBL124166"
    ],
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
      "constraint_type": {
        "label": "format constraint",
        "qid": "Q21502404"
      },
      "new_value": "124166",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL124166",
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

## 033. `repair_Q82919594_2425672958`

| Field | Value |
|---|---|
| qid | Q82919594 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q82919594::P2877 |
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
| decision_branch | rule_deterministic |
| rationale | Rule-deterministic format constraint fix. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
  "revision_id": 2425672958,
  "value": [
    "214196"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "214196"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "214196"
    ],
    "new_value": [
      "214196"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL214196"
    ],
    "old_value": [
      "SCHEMBL214196"
    ],
    "removed_unique_values": [
      "SCHEMBL214196"
    ],
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
    "label": "Furan-2,5-dione--methoxyethene (1/1)"
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
      "constraint_type": {
        "label": "format constraint",
        "qid": "Q21502404"
      },
      "new_value": "214196",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL214196",
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

## 034. `repair_Q82994476_2425694886`

| Field | Value |
|---|---|
| qid | Q82994476 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | ABOX::Q82994476::P2877 |
| tbox_revision_key |  |

### Annotation Focus

- Check whether the rule or format alone justifies the repair.
- If choosing the target requires local or external evidence, mark overclaimed / needs_local_evidence / needs_external_evidence.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | numeric |
| truth_tokens_preview | ["1491698"] |
| decision_branch | rule_deterministic |
| rationale | Rule-deterministic format constraint fix. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "NPImporterBot",
  "kind": "A_BOX",
  "new_value": [
    "1491698"
  ],
  "old_value": [
    "SCHEMBL1491698"
  ],
  "revision_id": 2425694886,
  "value": [
    "1491698"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "1491698"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "1491698"
    ],
    "new_value": [
      "1491698"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL1491698"
    ],
    "old_value": [
      "SCHEMBL1491698"
    ],
    "removed_unique_values": [
      "SCHEMBL1491698"
    ],
    "value_multiplicity_changes": {
      "1491698": {
        "new": 1,
        "old": 0
      },
      "SCHEMBL1491698": {
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
    "SCHEMBL1491698"
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
    "1491698"
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
    "label": "Methyl 1,2,3,4-tetrahydroisoquinoline-3-carboxylate"
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
      "constraint_type": {
        "label": "format constraint",
        "qid": "Q21502404"
      },
      "new_value": "1491698",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL1491698",
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

## 035. `repair_Q90536038_2425735162`

| Field | Value |
|---|---|
| qid | Q90536038 |
| property | P2877 |
| track | A_BOX |
| class / subtype / confidence | TypeA / FORMAT_NORMALIZATION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_l_format_normalization |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
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
| decision_branch | rule_deterministic |
| rationale | Rule-deterministic format constraint fix. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "14388592"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "14388592"
    ],
    "new_value": [
      "14388592"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "SCHEMBL14388592"
    ],
    "old_value": [
      "SCHEMBL14388592"
    ],
    "removed_unique_values": [
      "SCHEMBL14388592"
    ],
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
      "constraint_type": {
        "label": "format constraint",
        "qid": "Q21502404"
      },
      "new_value": "14388592",
      "normalization_kind": "strip_schembl_prefix",
      "old_value": "SCHEMBL14388592",
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
