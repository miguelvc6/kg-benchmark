# TypeC_UNKNOWN_MULTIPLICITY_ARTIFACT

Cases: 5

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q100541319_1295689984`

| Field | Value |
|---|---|
| qid | Q100541319 |
| property | P8726 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_MULTIPLICITY_ARTIFACT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_multiplicity_artifact |
| decision_constraint_type |   |
| group_key | ABOX::Q100541319::P8726 |
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
| truth_tokens_preview | ["2015/si/385/made"] |
| classification_target_tokens | ["2015/si/385/made"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | unknown_multiplicity_artifact |
| rationale | Unique values are unchanged; the repair appears to be a multiplicity or reconstruction artifact. |
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
    "2015/si/385/made"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "2015/si/385/made"
  ],
  "retained_unique_values": [
    "2015/si/385/made"
  ],
  "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_multiplicity_artifact",
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
  "author": "Tagishsimon",
  "kind": "A_BOX",
  "new_value": [
    "2015/si/385/made",
    "2015/si/385/made"
  ],
  "old_value": [
    "2015/si/385/made"
  ],
  "revision_id": 1295689984,
  "value": [
    "2015/si/385/made",
    "2015/si/385/made"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "2015/si/385/made": 2
    },
    "new_unique": [
      "2015/si/385/made"
    ],
    "new_values": [
      "2015/si/385/made",
      "2015/si/385/made"
    ],
    "new_values_raw": [
      "2015/si/385/made",
      "2015/si/385/made"
    ],
    "normalized_unique_values_unchanged": true,
    "old_counts": {
      "2015/si/385/made": 1
    },
    "old_unique": [
      "2015/si/385/made"
    ],
    "old_values": [
      "2015/si/385/made"
    ],
    "old_values_raw": [
      "2015/si/385/made"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "2015/si/385/made"
    ],
    "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
    "value_multiplicity_changes": {
      "2015/si/385/made": {
        "new": 2,
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
    "2015/si/385/made"
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
    "2015/si/385/made"
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
    "description": "Irish Statutory Instrument S.I. No. 385/2015",
    "label": "Industrial Relations Act 1976 (Section 8) Order 2015"
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
      "added_unique_values": [],
      "exact_value_lists_unchanged": false,
      "new_counts": {
        "2015/si/385/made": 2
      },
      "new_unique": [
        "2015/si/385/made"
      ],
      "new_values": [
        "2015/si/385/made",
        "2015/si/385/made"
      ],
      "new_values_raw": [
        "2015/si/385/made",
        "2015/si/385/made"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "2015/si/385/made": 1
      },
      "old_unique": [
        "2015/si/385/made"
      ],
      "old_values": [
        "2015/si/385/made"
      ],
      "old_values_raw": [
        "2015/si/385/made"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "2015/si/385/made"
      ],
      "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "2015/si/385/made": {
          "new": 2,
          "old": 1
        }
      }
    },
    "result": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "result": false,
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
    "result": "unknown_multiplicity_artifact",
    "step": "branch"
  }
]
```

---

## 002. `repair_Q4770432_2437105937`

| Field | Value |
|---|---|
| qid | Q4770432 |
| property | P7293 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_MULTIPLICITY_ARTIFACT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_multiplicity_artifact |
| decision_constraint_type |   |
| group_key | ABOX::Q4770432::P7293 |
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
| truth_tokens_preview | ["9810677291005606"] |
| classification_target_tokens | ["9810677291005606"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | unknown_multiplicity_artifact |
| rationale | Unique values are unchanged; the repair appears to be a multiplicity or reconstruction artifact. |
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
    "9810677291005606"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "9810677291005606"
  ],
  "retained_unique_values": [
    "9810677291005606"
  ],
  "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_multiplicity_artifact",
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
    "9810677291005606",
    "9810677291005606"
  ],
  "old_value": [
    "9810677291005606"
  ],
  "revision_id": 2437105937,
  "value": [
    "9810677291005606",
    "9810677291005606"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "9810677291005606": 2
    },
    "new_unique": [
      "9810677291005606"
    ],
    "new_values": [
      "9810677291005606",
      "9810677291005606"
    ],
    "new_values_raw": [
      "9810677291005606",
      "9810677291005606"
    ],
    "normalized_unique_values_unchanged": true,
    "old_counts": {
      "9810677291005606": 1
    },
    "old_unique": [
      "9810677291005606"
    ],
    "old_values": [
      "9810677291005606"
    ],
    "old_values_raw": [
      "9810677291005606"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "9810677291005606"
    ],
    "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
    "value_multiplicity_changes": {
      "9810677291005606": {
        "new": 2,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-03T10:05:15",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P7293",
  "report_revision_new": 2437601105,
  "report_revision_old": 2437327920,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "report_violation_types": [
    "Single value",
    "Unique value"
  ],
  "value": [
    "9810677291005606"
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
    "9810677291005606"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "National Library of Poland record no. identifier. Format: \"98\", followed by 10 digits, then ending with \"5606\"",
    "label": "National Library of Poland MMS ID"
  },
  "qid": {
    "description": "political scientist",
    "label": "Anoush Ehteshami"
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
        "9810677291005606": 2
      },
      "new_unique": [
        "9810677291005606"
      ],
      "new_values": [
        "9810677291005606",
        "9810677291005606"
      ],
      "new_values_raw": [
        "9810677291005606",
        "9810677291005606"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "9810677291005606": 1
      },
      "old_unique": [
        "9810677291005606"
      ],
      "old_values": [
        "9810677291005606"
      ],
      "old_values_raw": [
        "9810677291005606"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "9810677291005606"
      ],
      "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "9810677291005606": {
          "new": 2,
          "old": 1
        }
      }
    },
    "result": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "result": false,
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
    "result": "unknown_multiplicity_artifact",
    "step": "branch"
  }
]
```

---

## 003. `repair_Q4908168_2440166740`

| Field | Value |
|---|---|
| qid | Q4908168 |
| property | P39 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_MULTIPLICITY_ARTIFACT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_multiplicity_artifact |
| decision_constraint_type |   |
| group_key | ABOX::Q4908168::P39 |
| tbox_revision_key |  |

### Annotation Focus

- Decide whether this is truly non-local: typec_judgment is usually external_by_elimination_ok, external_confirmed, local_missed, unknown_or_incomplete, or bad_target.
- Check local evidence summary: if truth tokens appear in local matches or obvious local context, mark local_missed.
- If the target is not well-defined or local context is too sparse, mark unknown_or_incomplete and recommend diagnostic/exclude.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | repair_target.new_value |
| truth_token_kind | qid |
| truth_tokens_preview | ["Q15964890", "Q16933914", "Q56063852", "Q3253693", "Q109283894", "...(+2)"] |
| classification_target_tokens | ["Q109283894", "Q15964890", "Q16933914", "Q3253693", "Q3315114", "Q486839", "Q56063852"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | unknown_multiplicity_artifact |
| rationale | Unique values are unchanged; the repair appears to be a multiplicity or reconstruction artifact. |
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
    "Q109283894",
    "Q15964890",
    "Q16933914",
    "Q3253693",
    "Q3315114",
    "Q486839",
    "Q56063852"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "Q109283894",
    "Q15964890",
    "Q16933914",
    "Q3253693",
    "Q3315114",
    "Q486839",
    "Q56063852"
  ],
  "retained_unique_values": [
    "Q109283894",
    "Q15964890",
    "Q16933914",
    "Q3253693",
    "Q3315114",
    "Q486839",
    "Q56063852"
  ],
  "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_multiplicity_artifact",
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
  "author": "Amqui",
  "kind": "A_BOX",
  "new_value": [
    "Q15964890",
    "Q16933914",
    "Q15964890",
    "Q15964890",
    "Q56063852",
    "Q3253693",
    "Q109283894",
    "Q486839",
    "Q3315114",
    "Q3315114"
  ],
  "new_value_descriptions_en": [
    "representative elected by the Canadian voters",
    "minister in the Cabinet of Canada",
    "representative elected by the Canadian voters",
    "representative elected by the Canadian voters",
    "former Canadian cabinet ministerial position",
    "Canadian cabinet position",
    "Canadian government executive",
    "representative of the voters to a parliament",
    "minister in the Cabinet of Canada",
    "minister in the Cabinet of Canada"
  ],
  "new_value_labels_en": [
    "member of the House of Commons of Canada",
    "Minister of Public Safety",
    "member of the House of Commons of Canada",
    "member of the House of Commons of Canada",
    "Minister of Border Security and Organized Crime Reduction",
    "President of the King's Privy Council for Canada",
    "Minister of Emergency Preparedness",
    "member of parliament",
    "Minister of National Defence",
    "Minister of National Defence"
  ],
  "old_value": [
    "Q15964890",
    "Q16933914",
    "Q15964890",
    "Q15964890",
    "Q56063852",
    "Q3253693",
    "Q109283894",
    "Q486839",
    "Q3315114"
  ],
  "old_value_descriptions_en": [
    "representative elected by the Canadian voters",
    "minister in the Cabinet of Canada",
    "representative elected by the Canadian voters",
    "representative elected by the Canadian voters",
    "former Canadian cabinet ministerial position",
    "Canadian cabinet position",
    "Canadian government executive",
    "representative of the voters to a parliament",
    "minister in the Cabinet of Canada"
  ],
  "old_value_labels_en": [
    "member of the House of Commons of Canada",
    "Minister of Public Safety",
    "member of the House of Commons of Canada",
    "member of the House of Commons of Canada",
    "Minister of Border Security and Organized Crime Reduction",
    "President of the King's Privy Council for Canada",
    "Minister of Emergency Preparedness",
    "member of parliament",
    "Minister of National Defence"
  ],
  "revision_id": 2440166740,
  "value": [
    "Q15964890",
    "Q16933914",
    "Q15964890",
    "Q15964890",
    "Q56063852",
    "Q3253693",
    "Q109283894",
    "Q486839",
    "Q3315114",
    "Q3315114"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q109283894": 1,
      "Q15964890": 3,
      "Q16933914": 1,
      "Q3253693": 1,
      "Q3315114": 2,
      "Q486839": 1,
      "Q56063852": 1
    },
    "new_unique": [
      "Q109283894",
      "Q15964890",
      "Q16933914",
      "Q3253693",
      "Q3315114",
      "Q486839",
      "Q56063852"
    ],
    "new_values": [
      "Q15964890",
      "Q16933914",
      "Q15964890",
      "Q15964890",
      "Q56063852",
      "Q3253693",
      "Q109283894",
      "Q486839",
      "Q3315114",
      "Q3315114"
    ],
    "new_values_raw": [
      "Q15964890",
      "Q16933914",
      "Q15964890",
      "Q15964890",
      "Q56063852",
      "Q3253693",
      "Q109283894",
      "Q486839",
      "Q3315114",
      "Q3315114"
    ],
    "normalized_unique_values_unchanged": true,
    "old_counts": {
      "Q109283894": 1,
      "Q15964890": 3,
      "Q16933914": 1,
      "Q3253693": 1,
      "Q3315114": 1,
      "Q486839": 1,
      "Q56063852": 1
    },
    "old_unique": [
      "Q109283894",
      "Q15964890",
      "Q16933914",
      "Q3253693",
      "Q3315114",
      "Q486839",
      "Q56063852"
    ],
    "old_values": [
      "Q15964890",
      "Q16933914",
      "Q15964890",
      "Q15964890",
      "Q56063852",
      "Q3253693",
      "Q109283894",
      "Q486839",
      "Q3315114"
    ],
    "old_values_raw": [
      "Q15964890",
      "Q16933914",
      "Q15964890",
      "Q15964890",
      "Q56063852",
      "Q3253693",
      "Q109283894",
      "Q486839",
      "Q3315114"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "Q109283894",
      "Q15964890",
      "Q16933914",
      "Q3253693",
      "Q3315114",
      "Q486839",
      "Q56063852"
    ],
    "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
    "value_multiplicity_changes": {
      "Q3315114": {
        "new": 2,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "representative elected by the Canadian voters",
    "minister in the Cabinet of Canada",
    "representative elected by the Canadian voters",
    "representative elected by the Canadian voters",
    "former Canadian cabinet ministerial position",
    "Canadian cabinet position",
    "Canadian government executive",
    "representative of the voters to a parliament",
    "minister in the Cabinet of Canada",
    "minister in the Cabinet of Canada"
  ],
  "value_labels_en": [
    "member of the House of Commons of Canada",
    "Minister of Public Safety",
    "member of the House of Commons of Canada",
    "member of the House of Commons of Canada",
    "Minister of Border Security and Organized Crime Reduction",
    "President of the King's Privy Council for Canada",
    "Minister of Emergency Preparedness",
    "member of parliament",
    "Minister of National Defence",
    "Minister of National Defence"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-12T15:17:49",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P39",
  "report_revision_new": 2441293385,
  "report_revision_old": 2440933504,
  "report_violation_type": "Value type Q|4164871, Q|21451536, Q|355567, Q|3687335, Q|81752537, Q|11452125, Q|16631188, Q|21114371, Q|28640, Q|114962596, Q|713223, Q|480319, Q|124467070, Q|124515690, Q|11773926, Q|124466786, Q|136649946",
  "report_violation_type_descriptions_en": [
    "social role with a set of powers and responsibilities within an organization",
    "fictional political, administrative or civil servant position",
    "legal privilege given to some members in monarchial and princely societies",
    "government institution with variable function depending on the jurisdiction",
    "position held by a university professor",
    "ancient government official in Asia",
    "assignment of a person's place in a military organization defining responsibilities and privileges",
    "תואר",
    "occupation requiring specialized training",
    "charge, function or position that existed at a time in history, but that currently does not exist. social role with a set of powers and responsibilities within a private or public organization or the state",
    "program that invites artists to work at a specific venue or place for a period of time",
    "official designation of a position held in an organization associated with certain duties of authority",
    "missing, lost or suppressed episcopal function",
    null,
    "titel",
    "megszűnt pozíció valamelyik egyházban",
    "metaclass for position"
  ],
  "report_violation_type_labels_en": [
    "position",
    "fictional office, position, or title",
    "noble title",
    "Council of state",
    "professorship",
    "East Asian government position",
    "military position",
    "episcopal title",
    "profession",
    "historical position",
    "artist-in-residence",
    "title of authority",
    "historical episcopal title",
    "Anglican episcopal title",
    "ecclesiastical occupation",
    "historical ecclesiastical position",
    "type of position"
  ],
  "report_violation_type_normalized": "Value type Q|4164871, Q|21451536, Q|355567, Q|3687335, Q|81752537, Q|11452125, Q|16631188, Q|21114371, Q|28640, Q|114962596, Q|713223, Q|480319, Q|124467070, Q|124515690, Q|11773926, Q|124466786, Q|136649946",
  "report_violation_type_qids": [
    "Q4164871",
    "Q21451536",
    "Q355567",
    "Q3687335",
    "Q81752537",
    "Q11452125",
    "Q16631188",
    "Q21114371",
    "Q28640",
    "Q114962596",
    "Q713223",
    "Q480319",
    "Q124467070",
    "Q124515690",
    "Q11773926",
    "Q124466786",
    "Q136649946"
  ],
  "report_violation_type_raw": "Value type Q|4164871, Q|21451536, Q|355567, Q|3687335, Q|81752537, Q|11452125, Q|16631188, Q|21114371, Q|28640, Q|114962596, Q|713223, Q|480319, Q|124467070, Q|124515690, Q|11773926, Q|124466786, Q|136649946",
  "value": [
    "Q15964890",
    "Q16933914",
    "Q15964890",
    "Q15964890",
    "Q56063852",
    "Q3253693",
    "Q109283894",
    "Q486839",
    "Q3315114"
  ],
  "value_descriptions_en": [
    "representative elected by the Canadian voters",
    "minister in the Cabinet of Canada",
    "representative elected by the Canadian voters",
    "representative elected by the Canadian voters",
    "former Canadian cabinet ministerial position",
    "Canadian cabinet position",
    "Canadian government executive",
    "representative of the voters to a parliament",
    "minister in the Cabinet of Canada"
  ],
  "value_labels_en": [
    "member of the House of Commons of Canada",
    "Minister of Public Safety",
    "member of the House of Commons of Canada",
    "member of the House of Commons of Canada",
    "Minister of Border Security and Organized Crime Reduction",
    "President of the King's Privy Council for Canada",
    "Minister of Emergency Preparedness",
    "member of parliament",
    "Minister of National Defence"
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
    "Q15964890",
    "Q16933914",
    "Q56063852",
    "Q3253693",
    "Q109283894",
    "Q486839",
    "Q3315114"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "subject currently or formerly holds the object position or public office",
    "label": "position held"
  },
  "qid": {
    "description": "Canadian politician, former police chief of Toronto",
    "label": "Bill Blair"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
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
        "Q109283894": 1,
        "Q15964890": 3,
        "Q16933914": 1,
        "Q3253693": 1,
        "Q3315114": 2,
        "Q486839": 1,
        "Q56063852": 1
      },
      "new_unique": [
        "Q109283894",
        "Q15964890",
        "Q16933914",
        "Q3253693",
        "Q3315114",
        "Q486839",
        "Q56063852"
      ],
      "new_values": [
        "Q15964890",
        "Q16933914",
        "Q15964890",
        "Q15964890",
        "Q56063852",
        "Q3253693",
        "Q109283894",
        "Q486839",
        "Q3315114",
        "Q3315114"
      ],
      "new_values_raw": [
        "Q15964890",
        "Q16933914",
        "Q15964890",
        "Q15964890",
        "Q56063852",
        "Q3253693",
        "Q109283894",
        "Q486839",
        "Q3315114",
        "Q3315114"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "Q109283894": 1,
        "Q15964890": 3,
        "Q16933914": 1,
        "Q3253693": 1,
        "Q3315114": 1,
        "Q486839": 1,
        "Q56063852": 1
      },
      "old_unique": [
        "Q109283894",
        "Q15964890",
        "Q16933914",
        "Q3253693",
        "Q3315114",
        "Q486839",
        "Q56063852"
      ],
      "old_values": [
        "Q15964890",
        "Q16933914",
        "Q15964890",
        "Q15964890",
        "Q56063852",
        "Q3253693",
        "Q109283894",
        "Q486839",
        "Q3315114"
      ],
      "old_values_raw": [
        "Q15964890",
        "Q16933914",
        "Q15964890",
        "Q15964890",
        "Q56063852",
        "Q3253693",
        "Q109283894",
        "Q486839",
        "Q3315114"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "Q109283894",
        "Q15964890",
        "Q16933914",
        "Q3253693",
        "Q3315114",
        "Q486839",
        "Q56063852"
      ],
      "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "Q3315114": {
          "new": 2,
          "old": 1
        }
      }
    },
    "result": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "result": false,
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
    "result": "unknown_multiplicity_artifact",
    "step": "branch"
  }
]
```

---

## 004. `repair_Q57907259_2447300850`

| Field | Value |
|---|---|
| qid | Q57907259 |
| property | P1053 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_MULTIPLICITY_ARTIFACT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_multiplicity_artifact |
| decision_constraint_type |   |
| group_key | ABOX::Q57907259::P1053 |
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
| truth_tokens_preview | ["H-6394-2014"] |
| classification_target_tokens | ["H-6394-2014"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | unknown_multiplicity_artifact |
| rationale | Unique values are unchanged; the repair appears to be a multiplicity or reconstruction artifact. |
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
    "H-6394-2014"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "H-6394-2014"
  ],
  "retained_unique_values": [
    "H-6394-2014"
  ],
  "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_multiplicity_artifact",
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
    "H-6394-2014",
    "H-6394-2014"
  ],
  "old_value": [
    "H-6394-2014"
  ],
  "revision_id": 2447300850,
  "value": [
    "H-6394-2014",
    "H-6394-2014"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "H-6394-2014": 2
    },
    "new_unique": [
      "H-6394-2014"
    ],
    "new_values": [
      "H-6394-2014",
      "H-6394-2014"
    ],
    "new_values_raw": [
      "H-6394-2014",
      "H-6394-2014"
    ],
    "normalized_unique_values_unchanged": true,
    "old_counts": {
      "H-6394-2014": 1
    },
    "old_unique": [
      "H-6394-2014"
    ],
    "old_values": [
      "H-6394-2014"
    ],
    "old_values_raw": [
      "H-6394-2014"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "H-6394-2014"
    ],
    "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
    "value_multiplicity_changes": {
      "H-6394-2014": {
        "new": 2,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T11:01:59",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1053",
  "report_revision_new": 2447744778,
  "report_revision_old": 2447041014,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "report_violation_types": [
    "Single value",
    "Unique value"
  ],
  "value": [
    "H-6394-2014"
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
    "H-6394-2014"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a researcher in a system for scientific authors, redirects to a Web of Science ID, along with P3829",
    "label": "ResearcherID"
  },
  "qid": {
    "description": "Czech geologist",
    "label": "Zuzana Roxerová"
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
        "H-6394-2014": 2
      },
      "new_unique": [
        "H-6394-2014"
      ],
      "new_values": [
        "H-6394-2014",
        "H-6394-2014"
      ],
      "new_values_raw": [
        "H-6394-2014",
        "H-6394-2014"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "H-6394-2014": 1
      },
      "old_unique": [
        "H-6394-2014"
      ],
      "old_values": [
        "H-6394-2014"
      ],
      "old_values_raw": [
        "H-6394-2014"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "H-6394-2014"
      ],
      "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "H-6394-2014": {
          "new": 2,
          "old": 1
        }
      }
    },
    "result": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "result": false,
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
    "result": "unknown_multiplicity_artifact",
    "step": "branch"
  }
]
```

---

## 005. `repair_Q58881928_2447300866`

| Field | Value |
|---|---|
| qid | Q58881928 |
| property | P1153 |
| track | A_BOX |
| class / subtype / confidence | TypeC / UNKNOWN_MULTIPLICITY_ARTIFACT / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_ic_u |
| popularity_bucket | mid |
| constraint_family | Q21502410 |
| classification_rule_family | diagnostic_unknown |
| classification_rule_subfamily | unknown_multiplicity_artifact |
| decision_constraint_type |   |
| group_key | ABOX::Q58881928::P1153 |
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
| truth_tokens_preview | ["56104440000"] |
| classification_target_tokens | ["56104440000"] |
| classification_target_reason | unique values are unchanged; only multiplicity changed |
| decision_branch | unknown_multiplicity_artifact |
| rationale | Unique values are unchanged; the repair appears to be a multiplicity or reconstruction artifact. |
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
    "56104440000"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "56104440000"
  ],
  "retained_unique_values": [
    "56104440000"
  ],
  "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "diagnostic_unknown",
  "classification_rule_subfamily": "unknown_multiplicity_artifact",
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
    "56104440000",
    "56104440000"
  ],
  "old_value": [
    "56104440000"
  ],
  "revision_id": 2447300866,
  "value": [
    "56104440000",
    "56104440000"
  ],
  "value_change_summary": {
    "added_unique_values": [],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "56104440000": 2
    },
    "new_unique": [
      "56104440000"
    ],
    "new_values": [
      "56104440000",
      "56104440000"
    ],
    "new_values_raw": [
      "56104440000",
      "56104440000"
    ],
    "normalized_unique_values_unchanged": true,
    "old_counts": {
      "56104440000": 1
    },
    "old_unique": [
      "56104440000"
    ],
    "old_values": [
      "56104440000"
    ],
    "old_values_raw": [
      "56104440000"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "56104440000"
    ],
    "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
    "value_multiplicity_changes": {
      "56104440000": {
        "new": 2,
        "old": 1
      }
    }
  }
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T10:49:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1153",
  "report_revision_new": 2447741876,
  "report_revision_old": 2447340701,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": [
    "56104440000"
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
    "56104440000"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for an author assigned in Scopus bibliographic database",
    "label": "Scopus author ID"
  },
  "qid": {
    "description": "Czech researcher and sociologist",
    "label": "Marie Pospíšilová"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
        "56104440000": 2
      },
      "new_unique": [
        "56104440000"
      ],
      "new_values": [
        "56104440000",
        "56104440000"
      ],
      "new_values_raw": [
        "56104440000",
        "56104440000"
      ],
      "normalized_unique_values_unchanged": true,
      "old_counts": {
        "56104440000": 1
      },
      "old_unique": [
        "56104440000"
      ],
      "old_values": [
        "56104440000"
      ],
      "old_values_raw": [
        "56104440000"
      ],
      "removed_unique_values": [],
      "retained_unique_values": [
        "56104440000"
      ],
      "semantic_action": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
      "value_multiplicity_changes": {
        "56104440000": {
          "new": 2,
          "old": 1
        }
      }
    },
    "result": "MULTIPLICITY_INCREASE_SAME_UNIQUE",
    "step": "value_delta"
  },
  {
    "result": false,
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
    "result": "unknown_multiplicity_artifact",
    "step": "branch"
  }
]
```

---
