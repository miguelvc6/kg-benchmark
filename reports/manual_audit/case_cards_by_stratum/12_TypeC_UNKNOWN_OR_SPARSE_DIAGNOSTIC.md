# TypeC_UNKNOWN_OR_SPARSE_DIAGNOSTIC

Cases: 5

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q136734150_2444706516`

| Field | Value |
|---|---|
| qid | Q136734150 |
| property | P434 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_INCOMPLETE_LOCAL_CONTEXT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_incomplete_local_context |
| decision_constraint_type |   |
| group_key | ABOX::Q136734150::P434 |
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
| truth_tokens_preview | ["4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"] |
| classification_target_tokens | ["4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
  ],
  "removed_unique_values": [
    "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
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
  "classification_rule_subfamily": "unknown_incomplete_local_context",
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
  "author": "Louperivois",
  "kind": "A_BOX",
  "new_value": [
    "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
  ],
  "old_value": [
    "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
  ],
  "revision_id": 2444706516,
  "value": [
    "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc": 1
    },
    "new_unique": [
      "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
    ],
    "new_values": [
      "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
    ],
    "new_values_raw": [
      "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit": 1
    },
    "old_unique": [
      "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
    ],
    "old_values": [
      "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
    ],
    "old_values_raw": [
      "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
    ],
    "removed_unique_values": [
      "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc": {
        "new": 1,
        "old": 0
      },
      "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit": {
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
  "report_fix_date": "2025-12-22T10:20:58",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P434",
  "report_revision_new": 2445454540,
  "report_revision_old": 2444886284,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": [
    "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 9,
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_boundary",
      "normalized_match_text": "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit",
      "raw_match_text": "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
      "token": "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
    }
  ],
  "needed": 1,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
  ],
  "truth_tokens_in_recorded_matches": [
    "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
  ],
  "used_literal_substring": true
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for an artist in the MusicBrainz open music encyclopedia",
    "label": "MusicBrainz artist ID"
  },
  "qid": {
    "description": "indonesian singer",
    "label": "Assia Keva"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
      "added_unique_values": [
        "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
      ],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc": 1
      },
      "new_unique": [
        "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
      ],
      "new_values": [
        "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
      ],
      "new_values_raw": [
        "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
      ],
      "normalized_unique_values_unchanged": false,
      "old_counts": {
        "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit": 1
      },
      "old_unique": [
        "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
      ],
      "old_values": [
        "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
      ],
      "old_values_raw": [
        "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
      ],
      "removed_unique_values": [
        "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
      ],
      "retained_unique_values": [],
      "semantic_action": "REPLACE_1_TO_1",
      "value_multiplicity_changes": {
        "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc": {
          "new": 1,
          "old": 0
        },
        "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit": {
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
      "local_ids_count": 9,
      "matched": true,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "literal_boundary",
          "normalized_match_text": "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit",
          "raw_match_text": "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL",
          "token": "4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc"
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
        "https://musicbrainz.org/artist/4d0c767b-1763-4d20-bf4f-8c41b8f7c3dc/edit"
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

## 002. `repair_Q137437438_2443816791`

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
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_incomplete_local_context |
| decision_constraint_type |   |
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
| classification_target_tokens | ["3895926"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "3895926"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "3895926"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "3895926-jennifer-trejo"
  ],
  "removed_unique_values": [
    "3895926-jennifer-trejo"
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
  "classification_rule_subfamily": "unknown_incomplete_local_context",
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
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_boundary",
      "normalized_match_text": "3895926-jennifer-trejo",
      "raw_match_text": "3895926-jennifer-trejo",
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
          "normalized_match_text": "3895926-jennifer-trejo",
          "raw_match_text": "3895926-jennifer-trejo",
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

## 003. `repair_Q137539453_2446042791`

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
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_incomplete_local_context |
| decision_constraint_type |   |
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
| classification_target_tokens | ["2278168"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "2278168"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "2278168"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "2278168-stella-nwimo"
  ],
  "removed_unique_values": [
    "2278168-stella-nwimo"
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
  "classification_rule_subfamily": "unknown_incomplete_local_context",
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
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_boundary",
      "normalized_match_text": "2278168-stella-nwimo",
      "raw_match_text": "2278168-stella-nwimo",
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
          "normalized_match_text": "2278168-stella-nwimo",
          "raw_match_text": "2278168-stella-nwimo",
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

## 004. `repair_Q50409009_2447258829`

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
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_incomplete_local_context |
| decision_constraint_type |   |
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
| classification_target_tokens | ["Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_normalized_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Pond with steppe areas (safety zone of 30 m) (Landscape Preserve)"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)"
  ],
  "removed_unique_values": [
    "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)"
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
  "classification_rule_subfamily": "unknown_incomplete_local_context",
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
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_normalized_exact",
      "normalized_match_text": "pond with steppe areas safety zone of 30 m landscape preserve",
      "raw_match_text": "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)",
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
          "kind": "literal_normalized_exact",
          "normalized_match_text": "pond with steppe areas safety zone of 30 m landscape preserve",
          "raw_match_text": "Pond with steppe areas \"(safety zone of 30 m) (Landscape Preserve)",
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

## 005. `repair_Q9837227_2445827867`

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
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_incomplete_local_context |
| decision_constraint_type |   |
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
| classification_target_tokens | ["Local authority associations"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | pre_repair_target_only_not_local |
| rationale | Only synthetic pre-repair target-property values matched; this is not independent local evidence. |
| local_match_kind | literal_boundary |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Local authority associations"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Local authority associations"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    ":Local authority associations"
  ],
  "removed_unique_values": [
    ":Local authority associations"
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
  "classification_rule_subfamily": "unknown_incomplete_local_context",
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
  "local_support_for_retained_value": [],
  "matched": true,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "literal_boundary",
      "normalized_match_text": ":local authority associations",
      "raw_match_text": ":Local authority associations",
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
          "normalized_match_text": ":local authority associations",
          "raw_match_text": ":Local authority associations",
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
