# TypeC_EXTERNAL_BY_ELIMINATION_QID_TRUTH

Cases: 31

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q107062481_2423947686`

| Field | Value |
|---|---|
| qid | Q107062481 |
| property | P1560 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21510862 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q107062481::P1560 |
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
| truth_tokens_preview | ["Q24716740", "Q24712367"] |
| classification_target_tokens | ["Q24712367"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q24712367"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q24712367"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "Q24716740"
  ],
  "retained_unique_values": [
    "Q24716740"
  ],
  "semantic_action": "ADD_SUPERSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "StarTrekker",
  "kind": "A_BOX",
  "new_value": [
    "Q24716740",
    "Q24712367"
  ],
  "new_value_descriptions_en": [
    "male given name",
    "male given name"
  ],
  "new_value_labels_en": [
    "Eustasius",
    "Eustasio"
  ],
  "old_value": [
    "Q24716740"
  ],
  "old_value_descriptions_en": [
    "male given name"
  ],
  "old_value_labels_en": [
    "Eustasius"
  ],
  "revision_id": 2423947686,
  "value": [
    "Q24716740",
    "Q24712367"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q24712367"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q24712367": 1,
      "Q24716740": 1
    },
    "new_unique": [
      "Q24712367",
      "Q24716740"
    ],
    "new_values": [
      "Q24716740",
      "Q24712367"
    ],
    "new_values_raw": [
      "Q24716740",
      "Q24712367"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q24716740": 1
    },
    "old_unique": [
      "Q24716740"
    ],
    "old_values": [
      "Q24716740"
    ],
    "old_values_raw": [
      "Q24716740"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "Q24716740"
    ],
    "semantic_action": "ADD_SUPERSET",
    "value_multiplicity_changes": {
      "Q24712367": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "male given name",
    "male given name"
  ],
  "value_labels_en": [
    "Eustasius",
    "Eustasio"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-11-01T07:18:15",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1560",
  "report_revision_new": 2424184513,
  "report_revision_old": 2423583650,
  "report_violation_type": "Mandatory Qualifiers",
  "report_violation_type_normalized": "Mandatory Qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Mandatory Qualifiers",
  "report_violation_types": [
    "Mandatory Qualifiers",
    "Item P|407"
  ],
  "value": [
    "Q24716740"
  ],
  "value_descriptions_en": [
    "male given name"
  ],
  "value_labels_en": [
    "Eustasius"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 12,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q24716740"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q24716740",
    "Q24712367"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "equivalent name (with respect to the meaning of the name) in the same language: female version of a male first name, male version of a female first name. Add primarily the closest matching one",
    "label": "given name version for other gender"
  },
  "qid": {
    "description": "female given name",
    "label": "Eustasia"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "symmetric constraint",
    "qid": "Q21510862"
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
      "local_ids_count": 12,
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
        "Q24716740"
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

## 002. `repair_Q108308406_2447220088`

| Field | Value |
|---|---|
| qid | Q108308406 |
| property | P50 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q108308406::P50 |
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
| truth_tokens_preview | ["Q57413151", "Q26972386"] |
| classification_target_tokens | ["Q57902495", "Q26972386"] |
| classification_target_reason | mixed update classification uses the deterministic changed pair while ignoring retained values |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q26972386"
  ],
  "classification_target_reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
  "classification_target_role": "changed_pair",
  "classification_target_tokens": [
    "Q57902495",
    "Q26972386"
  ],
  "new_changed_value": "Q26972386",
  "old_changed_value": "Q57902495",
  "removed_target_tokens": [
    "Q57902495"
  ],
  "removed_unique_values": [
    "Q57902495"
  ],
  "retained_support_tokens": [
    "Q57413151"
  ],
  "retained_unique_values": [
    "Q57413151"
  ],
  "semantic_action": "MIXED_UPDATE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q57413151",
    "Q26972386"
  ],
  "new_value_descriptions_en": [
    "researcher ORCID id 0000-0002-3100-0877",
    "Italian geologist (1957-)"
  ],
  "new_value_labels_en": [
    "Jason Phipps Morgan",
    "Carlo Doglioni"
  ],
  "old_value": [
    "Q57413151",
    "Q57902495"
  ],
  "old_value_descriptions_en": [
    "researcher ORCID id 0000-0002-3100-0877",
    "Italian geologist (1957-)"
  ],
  "old_value_labels_en": [
    "Jason Phipps Morgan",
    "Carlo Doglioni"
  ],
  "revision_id": 2447220088,
  "value": [
    "Q57413151",
    "Q26972386"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q26972386"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q26972386": 1,
      "Q57413151": 1
    },
    "new_unique": [
      "Q26972386",
      "Q57413151"
    ],
    "new_values": [
      "Q57413151",
      "Q26972386"
    ],
    "new_values_raw": [
      "Q57413151",
      "Q26972386"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q57413151": 1,
      "Q57902495": 1
    },
    "old_unique": [
      "Q57413151",
      "Q57902495"
    ],
    "old_values": [
      "Q57413151",
      "Q57902495"
    ],
    "old_values_raw": [
      "Q57413151",
      "Q57902495"
    ],
    "removed_unique_values": [
      "Q57902495"
    ],
    "retained_unique_values": [
      "Q57413151"
    ],
    "semantic_action": "MIXED_UPDATE",
    "value_multiplicity_changes": {
      "Q26972386": {
        "new": 1,
        "old": 0
      },
      "Q57902495": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "researcher ORCID id 0000-0002-3100-0877",
    "Italian geologist (1957-)"
  ],
  "value_labels_en": [
    "Jason Phipps Morgan",
    "Carlo Doglioni"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T15:21:58",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P50",
  "report_revision_new": 2447821364,
  "report_revision_old": 2447436458,
  "report_violation_type": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "report_violation_type_descriptions_en": [
    "being that has certain capacities or attributes constituting personhood (for humans, use Q5 [human] with P31 [instance of])",
    "fictitious name that a person or group assumes for a particular purpose, which differs from their original or true name (orthonym)",
    "social entity established to meet needs or pursue goals",
    "any set of human beings",
    "fictional human or non-human character in a narrative work of art",
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "field of computer science that develops and studies software enabling machines to exhibit intelligent behavior",
    "human being whose existence is not directly attested, but is deduced by other evidence",
    "intelligent actor with unnatural origin",
    "distinct and identifiable entity with agency, capable of performing actions",
    "copyright is owned or retained by the party that commissioned it or by the employer of the person who produced it",
    null
  ],
  "report_violation_type_labels_en": [
    "person",
    "pseudonym",
    "organization",
    "group of humans",
    "character",
    "human",
    "artificial intelligence",
    "hypothetical person",
    "artificially intelligent entity",
    "being",
    "corporate authorship",
    "corporate author"
  ],
  "report_violation_type_normalized": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "report_violation_type_qids": [
    "Q215627",
    "Q61002",
    "Q43229",
    "Q16334295",
    "Q95074",
    "Q5",
    "Q11660",
    "Q75855169",
    "Q107307291",
    "Q24229398",
    "Q78056559",
    "Q60557912"
  ],
  "report_violation_type_raw": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "value": [
    "Q57413151",
    "Q57902495"
  ],
  "value_descriptions_en": [
    "researcher ORCID id 0000-0002-3100-0877",
    "Italian geologist (1957-)"
  ],
  "value_labels_en": [
    "Jason Phipps Morgan",
    "Carlo Doglioni"
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
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q57902495"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q57413151",
      "Q57902495"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q57413151",
    "Q26972386"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "main creator(s) of a written work (use on works, not humans); use P2093 (author name string) when Wikidata item is unknown or does not exist",
    "label": "author"
  },
  "qid": {
    "description": "editorial published in Terra Nova in January 2013",
    "label": "Editorial"
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
      "found": 1,
      "independent_match_count": 0,
      "local_ids_count": 7,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q57902495"
        }
      ],
      "needed": 2,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q57413151",
        "Q57902495"
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

## 003. `repair_Q11084_2443681378`

| Field | Value |
|---|---|
| qid | Q11084 |
| property | P1313 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q11084::P1313 |
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
| truth_tokens_preview | ["Q109563825"] |
| classification_target_tokens | ["Q109563825"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q109563825"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q109563825"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "CREATE_FROM_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Danil Satria",
  "kind": "A_BOX",
  "new_value": [
    "Q109563825"
  ],
  "new_value_descriptions_en": [
    "political position held by the regional leader of Kediri Regency"
  ],
  "new_value_labels_en": [
    "Regent of Kediri"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2443681378,
  "value": [
    "Q109563825"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q109563825"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q109563825": 1
    },
    "new_unique": [
      "Q109563825"
    ],
    "new_values": [
      "Q109563825"
    ],
    "new_values_raw": [
      "Q109563825"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {},
    "old_unique": [],
    "old_values": [],
    "old_values_raw": [
      "MISSING"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [],
    "semantic_action": "CREATE_FROM_MISSING",
    "value_multiplicity_changes": {
      "Q109563825": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "political position held by the regional leader of Kediri Regency"
  ],
  "value_labels_en": [
    "Regent of Kediri"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-20T09:08:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1313",
  "report_revision_new": 2444431131,
  "report_revision_old": 2444018890,
  "report_violation_type": "Target required claim P|31 one of Q|294414, Q|4164871, Q|5663900, Q|11773926, Q|17279032, Q|21114371, Q|21451536, Q|114962596, Q|124466786, Q|124467070",
  "report_violation_type_descriptions_en": [
    "elected or appointed political position",
    "social role with a set of powers and responsibilities within an organization",
    "highest authority of a municipality in Spain and medieval Portugal",
    "titel",
    "class of positions where the incumbent is selected by means of an election",
    "תואר",
    "fictional political, administrative or civil servant position",
    "charge, function or position that existed at a time in history, but that currently does not exist. social role with a set of powers and responsibilities within a private or public organization or the state",
    "megszűnt pozíció valamelyik egyházban",
    "missing, lost or suppressed episcopal function"
  ],
  "report_violation_type_labels_en": [
    "public office",
    "position",
    "alcalde",
    "ecclesiastical occupation",
    "elective office",
    "episcopal title",
    "fictional office, position, or title",
    "historical position",
    "historical ecclesiastical position",
    "historical episcopal title"
  ],
  "report_violation_type_normalized": "Target required claim P|31 one of Q|294414, Q|4164871, Q|5663900, Q|11773926, Q|17279032, Q|21114371, Q|21451536, Q|114962596, Q|124466786, Q|124467070",
  "report_violation_type_qids": [
    "Q294414",
    "Q4164871",
    "Q5663900",
    "Q11773926",
    "Q17279032",
    "Q21114371",
    "Q21451536",
    "Q114962596",
    "Q124466786",
    "Q124467070"
  ],
  "report_violation_type_raw": "Target required claim P|31 one of Q|294414, Q|4164871, Q|5663900, Q|11773926, Q|17279032, Q|21114371, Q|21451536, Q|114962596, Q|124466786, Q|124467070",
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
  "local_ids_count": 66,
  "local_support_for_retained_value": [],
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
    "Q109563825"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "political office that is fulfilled by the head of the government of this item",
    "label": "office held by head of government"
  },
  "qid": {
    "description": "regency in East Java Province, Indonesia",
    "label": "Kediri"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
      "local_ids_count": 66,
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

## 004. `repair_Q11724997_2445332870`

| Field | Value |
|---|---|
| qid | Q11724997 |
| property | P184 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q11724997::P184 |
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
| truth_tokens_preview | ["Q11181530"] |
| classification_target_tokens | ["Q11181530"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q11181530"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Q11181530"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q102117969"
  ],
  "removed_unique_values": [
    "Q102117969"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q11181530"
  ],
  "new_value_descriptions_en": [
    "Polish physicist (1916-1994)"
  ],
  "new_value_labels_en": [
    "Jan Rzewuski"
  ],
  "old_value": [
    "Q102117969"
  ],
  "old_value_descriptions_en": [
    "Polish physicist (1916-1994)"
  ],
  "old_value_labels_en": [
    "Jan Rzewuski"
  ],
  "revision_id": 2445332870,
  "value": [
    "Q11181530"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q11181530"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q11181530": 1
    },
    "new_unique": [
      "Q11181530"
    ],
    "new_values": [
      "Q11181530"
    ],
    "new_values_raw": [
      "Q11181530"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q102117969": 1
    },
    "old_unique": [
      "Q102117969"
    ],
    "old_values": [
      "Q102117969"
    ],
    "old_values_raw": [
      "Q102117969"
    ],
    "removed_unique_values": [
      "Q102117969"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q102117969": {
        "new": 0,
        "old": 1
      },
      "Q11181530": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "Polish physicist (1916-1994)"
  ],
  "value_labels_en": [
    "Jan Rzewuski"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-23T14:42:01",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P184",
  "report_revision_new": 2446079329,
  "report_revision_old": 2445480859,
  "report_violation_type": "Value type Q|5, Q|15632617",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "human being that only exists in fictional works"
  ],
  "report_violation_type_labels_en": [
    "human",
    "fictional human"
  ],
  "report_violation_type_normalized": "Value type Q|5, Q|15632617",
  "report_violation_type_qids": [
    "Q5",
    "Q15632617"
  ],
  "report_violation_type_raw": "Value type Q|5, Q|15632617",
  "report_violation_types": [
    "Value type Q|5, Q|15632617",
    "Inverse"
  ],
  "value": [
    "Q102117969"
  ],
  "value_descriptions_en": [
    "Polish physicist (1916-1994)"
  ],
  "value_labels_en": [
    "Jan Rzewuski"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 40,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q102117969"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q11181530"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "person who supervised the doctorate or PhD thesis of the subject",
    "label": "doctoral advisor"
  },
  "qid": {
    "description": "Polish physicist",
    "label": "Jerzy Lukierski"
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
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "inverse constraint",
    "qid": "Q21510855"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
      "local_ids_count": 40,
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
        "Q102117969"
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

## 005. `repair_Q135499656_2388912528`

| Field | Value |
|---|---|
| qid | Q135499656 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q135499656::P5236 |
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
| truth_tokens_preview | ["Q200", "Q201", "Q60101"] |
| classification_target_tokens | ["Q60101"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q60101"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q60101"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "Q200",
    "Q201"
  ],
  "retained_unique_values": [
    "Q200",
    "Q201"
  ],
  "semantic_action": "ADD_SUPERSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q200",
    "Q201",
    "Q60101"
  ],
  "new_value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number"
  ],
  "new_value_labels_en": [
    "ҩба",
    "3",
    "107"
  ],
  "old_value": [
    "Q200",
    "Q201"
  ],
  "old_value_descriptions_en": [
    "natural number",
    "natural number"
  ],
  "old_value_labels_en": [
    "ҩба",
    "3"
  ],
  "revision_id": 2388912528,
  "value": [
    "Q200",
    "Q201",
    "Q60101"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q60101"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q200": 1,
      "Q201": 1,
      "Q60101": 1
    },
    "new_unique": [
      "Q200",
      "Q201",
      "Q60101"
    ],
    "new_values": [
      "Q200",
      "Q201",
      "Q60101"
    ],
    "new_values_raw": [
      "Q200",
      "Q201",
      "Q60101"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q200": 1,
      "Q201": 1
    },
    "old_unique": [
      "Q200",
      "Q201"
    ],
    "old_values": [
      "Q200",
      "Q201"
    ],
    "old_values_raw": [
      "Q200",
      "Q201"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "Q200",
      "Q201"
    ],
    "semantic_action": "ADD_SUPERSET",
    "value_multiplicity_changes": {
      "Q60101": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number"
  ],
  "value_labels_en": [
    "ҩба",
    "3",
    "107"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-10T06:26:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2389998295,
  "report_revision_old": 2388961222,
  "report_violation_type": "Value type Q|49008",
  "report_violation_type_descriptions_en": [
    "positive integer with exactly two divisors, 1 and itself"
  ],
  "report_violation_type_labels_en": [
    "prime number"
  ],
  "report_violation_type_normalized": "Value type Q|49008",
  "report_violation_type_qids": [
    "Q49008"
  ],
  "report_violation_type_raw": "Value type Q|49008",
  "value": [
    "Q200",
    "Q201"
  ],
  "value_descriptions_en": [
    "natural number",
    "natural number"
  ],
  "value_labels_en": [
    "ҩба",
    "3"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 17,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q200",
      "Q201"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q200",
    "Q201",
    "Q60101"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "one of the prime numbers that can be multiplied to give this number",
    "label": "prime factor"
  },
  "qid": {
    "description": "natural number",
    "label": "11556"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
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
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q200",
        "Q201"
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

## 006. `repair_Q135501528_2388922197`

| Field | Value |
|---|---|
| qid | Q135501528 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q135501528::P5236 |
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
| truth_tokens_preview | ["Q200", "Q201", "Q203", "Q713181"] |
| classification_target_tokens | ["Q713181"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q713181"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q713181"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "Q200",
    "Q201",
    "Q203"
  ],
  "retained_unique_values": [
    "Q200",
    "Q201",
    "Q203"
  ],
  "semantic_action": "ADD_SUPERSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q200",
    "Q201",
    "Q203",
    "Q713181"
  ],
  "new_value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number",
    "natural number"
  ],
  "new_value_labels_en": [
    "ҩба",
    "3",
    "5",
    "83"
  ],
  "old_value": [
    "Q200",
    "Q201",
    "Q203"
  ],
  "old_value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number"
  ],
  "old_value_labels_en": [
    "ҩба",
    "3",
    "5"
  ],
  "revision_id": 2388922197,
  "value": [
    "Q200",
    "Q201",
    "Q203",
    "Q713181"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q713181"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q200": 1,
      "Q201": 1,
      "Q203": 1,
      "Q713181": 1
    },
    "new_unique": [
      "Q200",
      "Q201",
      "Q203",
      "Q713181"
    ],
    "new_values": [
      "Q200",
      "Q201",
      "Q203",
      "Q713181"
    ],
    "new_values_raw": [
      "Q200",
      "Q201",
      "Q203",
      "Q713181"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q200": 1,
      "Q201": 1,
      "Q203": 1
    },
    "old_unique": [
      "Q200",
      "Q201",
      "Q203"
    ],
    "old_values": [
      "Q200",
      "Q201",
      "Q203"
    ],
    "old_values_raw": [
      "Q200",
      "Q201",
      "Q203"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "Q200",
      "Q201",
      "Q203"
    ],
    "semantic_action": "ADD_SUPERSET",
    "value_multiplicity_changes": {
      "Q713181": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number",
    "natural number"
  ],
  "value_labels_en": [
    "ҩба",
    "3",
    "5",
    "83"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-10T06:26:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2389998295,
  "report_revision_old": 2388961222,
  "report_violation_type": "Value type Q|49008",
  "report_violation_type_descriptions_en": [
    "positive integer with exactly two divisors, 1 and itself"
  ],
  "report_violation_type_labels_en": [
    "prime number"
  ],
  "report_violation_type_normalized": "Value type Q|49008",
  "report_violation_type_qids": [
    "Q49008"
  ],
  "report_violation_type_raw": "Value type Q|49008",
  "value": [
    "Q200",
    "Q201",
    "Q203"
  ],
  "value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number"
  ],
  "value_labels_en": [
    "ҩба",
    "3",
    "5"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 18,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q200",
      "Q201",
      "Q203"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q200",
    "Q201",
    "Q203",
    "Q713181"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "one of the prime numbers that can be multiplied to give this number",
    "label": "prime factor"
  },
  "qid": {
    "description": "natural number",
    "label": "12450"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
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
      "local_ids_count": 18,
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
        "Q200",
        "Q201",
        "Q203"
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

## 007. `repair_Q135504583_2388914308`

| Field | Value |
|---|---|
| qid | Q135504583 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q135504583::P5236 |
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
| truth_tokens_preview | ["Q200", "Q201", "Q19242243"] |
| classification_target_tokens | ["Q19242243"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q19242243"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q19242243"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "Q200",
    "Q201"
  ],
  "retained_unique_values": [
    "Q200",
    "Q201"
  ],
  "semantic_action": "ADD_SUPERSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q200",
    "Q201",
    "Q19242243"
  ],
  "new_value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number"
  ],
  "new_value_labels_en": [
    "ҩба",
    "3",
    "2383"
  ],
  "old_value": [
    "Q200",
    "Q201"
  ],
  "old_value_descriptions_en": [
    "natural number",
    "natural number"
  ],
  "old_value_labels_en": [
    "ҩба",
    "3"
  ],
  "revision_id": 2388914308,
  "value": [
    "Q200",
    "Q201",
    "Q19242243"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q19242243"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q19242243": 1,
      "Q200": 1,
      "Q201": 1
    },
    "new_unique": [
      "Q19242243",
      "Q200",
      "Q201"
    ],
    "new_values": [
      "Q200",
      "Q201",
      "Q19242243"
    ],
    "new_values_raw": [
      "Q200",
      "Q201",
      "Q19242243"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q200": 1,
      "Q201": 1
    },
    "old_unique": [
      "Q200",
      "Q201"
    ],
    "old_values": [
      "Q200",
      "Q201"
    ],
    "old_values_raw": [
      "Q200",
      "Q201"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "Q200",
      "Q201"
    ],
    "semantic_action": "ADD_SUPERSET",
    "value_multiplicity_changes": {
      "Q19242243": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number"
  ],
  "value_labels_en": [
    "ҩба",
    "3",
    "2383"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-10T06:26:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2389998295,
  "report_revision_old": 2388961222,
  "report_violation_type": "Value type Q|49008",
  "report_violation_type_descriptions_en": [
    "positive integer with exactly two divisors, 1 and itself"
  ],
  "report_violation_type_labels_en": [
    "prime number"
  ],
  "report_violation_type_normalized": "Value type Q|49008",
  "report_violation_type_qids": [
    "Q49008"
  ],
  "report_violation_type_raw": "Value type Q|49008",
  "value": [
    "Q200",
    "Q201"
  ],
  "value_descriptions_en": [
    "natural number",
    "natural number"
  ],
  "value_labels_en": [
    "ҩба",
    "3"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 17,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q200",
      "Q201"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q200",
    "Q201",
    "Q19242243"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "one of the prime numbers that can be multiplied to give this number",
    "label": "prime factor"
  },
  "qid": {
    "description": "natural number",
    "label": "14298"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
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
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q200",
        "Q201"
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

## 008. `repair_Q135569038_2389882363`

| Field | Value |
|---|---|
| qid | Q135569038 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q135569038::P5236 |
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
| truth_tokens_preview | ["Q200", "Q201", "Q369163"] |
| classification_target_tokens | ["Q369163"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q369163"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q369163"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "Q200",
    "Q201"
  ],
  "retained_unique_values": [
    "Q200",
    "Q201"
  ],
  "semantic_action": "ADD_SUPERSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q200",
    "Q201",
    "Q369163"
  ],
  "new_value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number"
  ],
  "new_value_labels_en": [
    "ҩба",
    "3",
    "631"
  ],
  "old_value": [
    "Q200",
    "Q201"
  ],
  "old_value_descriptions_en": [
    "natural number",
    "natural number"
  ],
  "old_value_labels_en": [
    "ҩба",
    "3"
  ],
  "revision_id": 2389882363,
  "value": [
    "Q200",
    "Q201",
    "Q369163"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q369163"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q200": 1,
      "Q201": 1,
      "Q369163": 1
    },
    "new_unique": [
      "Q200",
      "Q201",
      "Q369163"
    ],
    "new_values": [
      "Q200",
      "Q201",
      "Q369163"
    ],
    "new_values_raw": [
      "Q200",
      "Q201",
      "Q369163"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q200": 1,
      "Q201": 1
    },
    "old_unique": [
      "Q200",
      "Q201"
    ],
    "old_values": [
      "Q200",
      "Q201"
    ],
    "old_values_raw": [
      "Q200",
      "Q201"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "Q200",
      "Q201"
    ],
    "semantic_action": "ADD_SUPERSET",
    "value_multiplicity_changes": {
      "Q369163": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number"
  ],
  "value_labels_en": [
    "ҩба",
    "3",
    "631"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-11T06:00:40",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2390538152,
  "report_revision_old": 2389998295,
  "report_violation_type": "Value type Q|49008",
  "report_violation_type_descriptions_en": [
    "positive integer with exactly two divisors, 1 and itself"
  ],
  "report_violation_type_labels_en": [
    "prime number"
  ],
  "report_violation_type_normalized": "Value type Q|49008",
  "report_violation_type_qids": [
    "Q49008"
  ],
  "report_violation_type_raw": "Value type Q|49008",
  "value": [
    "Q200",
    "Q201"
  ],
  "value_descriptions_en": [
    "natural number",
    "natural number"
  ],
  "value_labels_en": [
    "ҩба",
    "3"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 17,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q200",
      "Q201"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q200",
    "Q201",
    "Q369163"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "one of the prime numbers that can be multiplied to give this number",
    "label": "prime factor"
  },
  "qid": {
    "description": "natural number",
    "label": "60576"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
    "label_en": "none-of constraint",
    "qid": "Q52558054"
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
      "needed": 1,
      "sources_used": [],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q200",
        "Q201"
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

## 009. `repair_Q136338282_2443310307`

| Field | Value |
|---|---|
| qid | Q136338282 |
| property | P22 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q136338282::P22 |
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
| truth_tokens_preview | ["Q2646258"] |
| classification_target_tokens | ["Q2646258"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q2646258"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Q2646258"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q136338283"
  ],
  "removed_unique_values": [
    "Q136338283"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q2646258"
  ],
  "new_value_descriptions_en": [
    "German politician"
  ],
  "new_value_labels_en": [
    "Alfred Strachwitz"
  ],
  "old_value": [
    "Q136338283"
  ],
  "old_value_descriptions_en": [
    "German politician"
  ],
  "old_value_labels_en": [
    "Alfred Strachwitz"
  ],
  "revision_id": 2443310307,
  "value": [
    "Q2646258"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q2646258"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q2646258": 1
    },
    "new_unique": [
      "Q2646258"
    ],
    "new_values": [
      "Q2646258"
    ],
    "new_values_raw": [
      "Q2646258"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q136338283": 1
    },
    "old_unique": [
      "Q136338283"
    ],
    "old_values": [
      "Q136338283"
    ],
    "old_values_raw": [
      "Q136338283"
    ],
    "removed_unique_values": [
      "Q136338283"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q136338283": {
        "new": 0,
        "old": 1
      },
      "Q2646258": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "German politician"
  ],
  "value_labels_en": [
    "Alfred Strachwitz"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-18T21:39:46",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P22",
  "report_revision_new": 2443878362,
  "report_revision_old": 2443471646,
  "report_violation_type": "Value type Q|215627, Q|95074, Q|4271324, Q|729, Q|21070598, Q|21070568, Q|13002315, Q|795052",
  "report_violation_type_descriptions_en": [
    "being that has certain capacities or attributes constituting personhood (for humans, use Q5 [human] with P31 [instance of])",
    "fictional human or non-human character in a narrative work of art",
    "character from mythology",
    "kingdom of multicellular eukaryotic organisms",
    "character who is hypothesized to exist, but where evidence is not conclusive",
    "human who is hypothesized to exist, but where evidence is not conclusive",
    "named person or animal that appears in legends that have some claim to be historical",
    "individual person or organism"
  ],
  "report_violation_type_labels_en": [
    "person",
    "character",
    "mythical character",
    "Animalia",
    "figure that may or may not be fictional",
    "human whose existence is disputed",
    "legendary figure",
    "individual"
  ],
  "report_violation_type_normalized": "Value type Q|215627, Q|95074, Q|4271324, Q|729, Q|21070598, Q|21070568, Q|13002315, Q|795052",
  "report_violation_type_qids": [
    "Q215627",
    "Q95074",
    "Q4271324",
    "Q729",
    "Q21070598",
    "Q21070568",
    "Q13002315",
    "Q795052"
  ],
  "report_violation_type_raw": "Value type Q|215627, Q|95074, Q|4271324, Q|729, Q|21070598, Q|21070568, Q|13002315, Q|795052",
  "report_violation_types": [
    "Value type Q|215627, Q|95074, Q|4271324, Q|729, Q|21070598, Q|21070568, Q|13002315, Q|795052",
    "Target required claim P|21 one of Q|44148, Q|48270, Q|1052281, Q|2449503, Q|6581097, Q|11287467, Q|15145778, Q|69990794"
  ],
  "value": [
    "Q136338283"
  ],
  "value_descriptions_en": [
    "German politician"
  ],
  "value_labels_en": [
    "Alfred Strachwitz"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 10,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q136338283"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q2646258"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "male parent of the subject. For stepfather, use \"stepparent\" (P3448)",
    "label": "father"
  },
  "qid": {
    "description": "20 May 1890 Bertelsdorf bei Lauban - 18 Apr 1967 Grabenstätt am Chiemsee",
    "label": "Beatrice, Gräfin Strachwitz von Gross-Zauche und Camminetz"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "inverse constraint",
    "qid": "Q21510855"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
      "local_ids_count": 10,
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
        "Q136338283"
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

## 010. `repair_Q14906850_2426562812`

| Field | Value |
|---|---|
| qid | Q14906850 |
| property | P682 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q14906850::P682 |
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
| truth_tokens_preview | ["Q14906839", "Q14860466", "Q14906846", "Q14860535", "Q14864949", "...(+18)"] |
| classification_target_tokens | ["Q14819288", "Q471817"] |
| classification_target_reason | mixed update classification uses the deterministic changed pair while ignoring retained values |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q471817"
  ],
  "classification_target_reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
  "classification_target_role": "changed_pair",
  "classification_target_tokens": [
    "Q14819288",
    "Q471817"
  ],
  "new_changed_value": "Q471817",
  "old_changed_value": "Q14819288",
  "removed_target_tokens": [
    "Q14819288"
  ],
  "removed_unique_values": [
    "Q14819288"
  ],
  "retained_support_tokens": [
    "Q14599571",
    "Q14645729",
    "Q14818023",
    "Q14819465",
    "Q14819468",
    "Q14819480",
    "Q14859587",
    "Q14860466",
    "Q14860535",
    "Q14863387",
    "Q14864579",
    "Q14864949",
    "Q14865650",
    "Q14873943",
    "Q14877292",
    "Q14881723",
    "Q14906839",
    "Q14906846",
    "Q1509074",
    "Q21095555",
    "Q21132859",
    "Q332154"
  ],
  "retained_unique_values": [
    "Q14599571",
    "Q14645729",
    "Q14818023",
    "Q14819465",
    "Q14819468",
    "Q14819480",
    "Q14859587",
    "Q14860466",
    "Q14860535",
    "Q14863387",
    "Q14864579",
    "Q14864949",
    "Q14865650",
    "Q14873943",
    "Q14877292",
    "Q14881723",
    "Q14906839",
    "Q14906846",
    "Q1509074",
    "Q21095555",
    "Q21132859",
    "Q332154"
  ],
  "semantic_action": "MIXED_UPDATE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q14906839",
    "Q14860466",
    "Q14906846",
    "Q14860535",
    "Q14864949",
    "Q14864579",
    "Q332154",
    "Q1509074",
    "Q471817",
    "Q14819465",
    "Q14865650",
    "Q14859587",
    "Q14645729",
    "Q14881723",
    "Q14873943",
    "Q14863387",
    "Q14819480",
    "Q14877292",
    "Q14818023",
    "Q21095555",
    "Q14599571",
    "Q21132859",
    "Q14819468"
  ],
  "new_value_descriptions_en": [
    "Any process that activates or increases the frequency, rate or extent of Rho protein signal transduction.",
    "process whose specific outcome is the progression of the skeleton over time, from its formation to the mature structure",
    "Any process that stops, prevents or reduces the frequency, rate or extent of neuron migration.",
    "Any process that results in a change in state or activity of a cell or an organism (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of a cytokine stimulus.",
    "Any process that determines the size and arrangement of collagen fibrils within an extracellular matrix.",
    "Any process that results in a change in state or activity of a cell or an organism (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of a mechanical stimulus.",
    "biological processes in a human body coupled to increasing age",
    "series of events that restore integrity to a damaged tissue, following an injury",
    "The process whose specific outcome is the progression of the heart over time, from its formation to the mature structure. The heart is a hollow, muscular organ, which, by contracting rhythmically, keeps up the circulation of the blood.",
    "The formation of a covalent cross-link between or within protein chains.",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of the immune response, the immunological reaction of an organism to an immunogenic stimulus.",
    "The process whose specific outcome is the progression of a blood vessel over time, from its formation to the mature structure. The blood vessel is the vasculature carrying blood.",
    "The progression of the cerebral cortex over time from its initial formation until its mature state. The cerebral cortex is the outer layered region of the telencephalon.",
    "Any process that results in a change in state or activity of a cell (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of an amino acid stimulus. An amino acid is a carboxylic acids containing one or more amino gr",
    "series of molecular signals initiated by the binding of extracellular ligand to an integrin on the surface of a target cell, and ending with regulation of a downstream cellular process, e.g. transcription",
    "Any process that results in a change in state or activity of a cell or an organism (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of an electromagnetic radiation stimulus. Electromagnetic radiation is a propag",
    "The binding of a cell to the extracellular matrix via adhesion molecules.",
    "The process whose specific outcome is the progression of the digestive tract over time, from its formation to the mature structure. The digestive tract is the anatomical structure through which food passes and is processed.",
    "series of molecular signals initiated by the binding of an extracellular ligand to a transforming growth factor beta receptor",
    "The process in which the structure of the smooth muscle tissue surrounding the aorta is generated and organized. An aorta is an artery that carries blood from the heart to other parts of the body.",
    "The process whose specific outcome is the progression of the skin over time, from its formation to the mature structure. The skin is the external membranous integument of an animal. In vertebrates the skin generally consists of two layers, an outer n",
    "process that is carried out at the cellular level which results in the assembly, arrangement of constituent parts, or disassembly of a supramolecular fiber, a polymer consisting of an indefinite number of protein or protein complex subunits that ha",
    "process that is carried out at the cellular level which results in the assembly, arrangement of constituent parts, or disassembly of an extracellular matrix"
  ],
  "new_value_labels_en": [
    "positive regulation of Rho protein signal transduction",
    "skeletal system development",
    "negative regulation of neuron migration",
    "response to cytokine",
    "collagen fibril organization",
    "response to mechanical stimulus",
    "ageing",
    "wound healing",
    "development of the heart",
    "peptide cross-linking",
    "negative regulation of immune response",
    "blood vessel development",
    "cerebral cortex development",
    "cellular response to amino acid stimulus",
    "integrin-mediated signaling pathway",
    "response to radiation",
    "cell-matrix adhesion",
    "digestive tract development",
    "transforming growth factor beta receptor signaling pathway",
    "aorta smooth muscle tissue morphogenesis",
    "skin development",
    "supramolecular fiber organization",
    "extracellular matrix organization"
  ],
  "old_value": [
    "Q14906839",
    "Q14860466",
    "Q14906846",
    "Q14860535",
    "Q14864949",
    "Q14864579",
    "Q332154",
    "Q1509074",
    "Q14819288",
    "Q14819465",
    "Q14865650",
    "Q14859587",
    "Q14645729",
    "Q14881723",
    "Q14873943",
    "Q14863387",
    "Q14819480",
    "Q14877292",
    "Q14818023",
    "Q21095555",
    "Q14599571",
    "Q21132859",
    "Q14819468"
  ],
  "old_value_descriptions_en": [
    "Any process that activates or increases the frequency, rate or extent of Rho protein signal transduction.",
    "process whose specific outcome is the progression of the skeleton over time, from its formation to the mature structure",
    "Any process that stops, prevents or reduces the frequency, rate or extent of neuron migration.",
    "Any process that results in a change in state or activity of a cell or an organism (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of a cytokine stimulus.",
    "Any process that determines the size and arrangement of collagen fibrils within an extracellular matrix.",
    "Any process that results in a change in state or activity of a cell or an organism (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of a mechanical stimulus.",
    "biological processes in a human body coupled to increasing age",
    "series of events that restore integrity to a damaged tissue, following an injury",
    "The process whose specific outcome is the progression of the heart over time, from its formation to the mature structure. The heart is a hollow, muscular organ, which, by contracting rhythmically, keeps up the circulation of the blood.",
    "The formation of a covalent cross-link between or within protein chains.",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of the immune response, the immunological reaction of an organism to an immunogenic stimulus.",
    "The process whose specific outcome is the progression of a blood vessel over time, from its formation to the mature structure. The blood vessel is the vasculature carrying blood.",
    "The progression of the cerebral cortex over time from its initial formation until its mature state. The cerebral cortex is the outer layered region of the telencephalon.",
    "Any process that results in a change in state or activity of a cell (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of an amino acid stimulus. An amino acid is a carboxylic acids containing one or more amino gr",
    "series of molecular signals initiated by the binding of extracellular ligand to an integrin on the surface of a target cell, and ending with regulation of a downstream cellular process, e.g. transcription",
    "Any process that results in a change in state or activity of a cell or an organism (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of an electromagnetic radiation stimulus. Electromagnetic radiation is a propag",
    "The binding of a cell to the extracellular matrix via adhesion molecules.",
    "The process whose specific outcome is the progression of the digestive tract over time, from its formation to the mature structure. The digestive tract is the anatomical structure through which food passes and is processed.",
    "series of molecular signals initiated by the binding of an extracellular ligand to a transforming growth factor beta receptor",
    "The process in which the structure of the smooth muscle tissue surrounding the aorta is generated and organized. An aorta is an artery that carries blood from the heart to other parts of the body.",
    "The process whose specific outcome is the progression of the skin over time, from its formation to the mature structure. The skin is the external membranous integument of an animal. In vertebrates the skin generally consists of two layers, an outer n",
    "process that is carried out at the cellular level which results in the assembly, arrangement of constituent parts, or disassembly of a supramolecular fiber, a polymer consisting of an indefinite number of protein or protein complex subunits that ha",
    "process that is carried out at the cellular level which results in the assembly, arrangement of constituent parts, or disassembly of an extracellular matrix"
  ],
  "old_value_labels_en": [
    "positive regulation of Rho protein signal transduction",
    "skeletal system development",
    "negative regulation of neuron migration",
    "response to cytokine",
    "collagen fibril organization",
    "response to mechanical stimulus",
    "ageing",
    "wound healing",
    "development of the heart",
    "peptide cross-linking",
    "negative regulation of immune response",
    "blood vessel development",
    "cerebral cortex development",
    "cellular response to amino acid stimulus",
    "integrin-mediated signaling pathway",
    "response to radiation",
    "cell-matrix adhesion",
    "digestive tract development",
    "transforming growth factor beta receptor signaling pathway",
    "aorta smooth muscle tissue morphogenesis",
    "skin development",
    "supramolecular fiber organization",
    "extracellular matrix organization"
  ],
  "revision_id": 2426562812,
  "value": [
    "Q14906839",
    "Q14860466",
    "Q14906846",
    "Q14860535",
    "Q14864949",
    "Q14864579",
    "Q332154",
    "Q1509074",
    "Q471817",
    "Q14819465",
    "Q14865650",
    "Q14859587",
    "Q14645729",
    "Q14881723",
    "Q14873943",
    "Q14863387",
    "Q14819480",
    "Q14877292",
    "Q14818023",
    "Q21095555",
    "Q14599571",
    "Q21132859",
    "Q14819468"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q471817"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q14599571": 1,
      "Q14645729": 1,
      "Q14818023": 1,
      "Q14819465": 1,
      "Q14819468": 1,
      "Q14819480": 1,
      "Q14859587": 1,
      "Q14860466": 1,
      "Q14860535": 1,
      "Q14863387": 1,
      "Q14864579": 1,
      "Q14864949": 1,
      "Q14865650": 1,
      "Q14873943": 1,
      "Q14877292": 1,
      "Q14881723": 1,
      "Q14906839": 1,
      "Q14906846": 1,
      "Q1509074": 1,
      "Q21095555": 1,
      "Q21132859": 1,
      "Q332154": 1,
      "Q471817": 1
    },
    "new_unique": [
      "Q14599571",
      "Q14645729",
      "Q14818023",
      "Q14819465",
      "Q14819468",
      "Q14819480",
      "Q14859587",
      "Q14860466",
      "Q14860535",
      "Q14863387",
      "Q14864579",
      "Q14864949",
      "Q14865650",
      "Q14873943",
      "Q14877292",
      "Q14881723",
      "Q14906839",
      "Q14906846",
      "Q1509074",
      "Q21095555",
      "Q21132859",
      "Q332154",
      "Q471817"
    ],
    "new_values": [
      "Q14906839",
      "Q14860466",
      "Q14906846",
      "Q14860535",
      "Q14864949",
      "Q14864579",
      "Q332154",
      "Q1509074",
      "Q471817",
      "Q14819465",
      "Q14865650",
      "Q14859587",
      "Q14645729",
      "Q14881723",
      "Q14873943",
      "Q14863387",
      "Q14819480",
      "Q14877292",
      "Q14818023",
      "Q21095555",
      "Q14599571",
      "Q21132859",
      "Q14819468"
    ],
    "new_values_raw": [
      "Q14906839",
      "Q14860466",
      "Q14906846",
      "Q14860535",
      "Q14864949",
      "Q14864579",
      "Q332154",
      "Q1509074",
      "Q471817",
      "Q14819465",
      "Q14865650",
      "Q14859587",
      "Q14645729",
      "Q14881723",
      "Q14873943",
      "Q14863387",
      "Q14819480",
      "Q14877292",
      "Q14818023",
      "Q21095555",
      "Q14599571",
      "Q21132859",
      "Q14819468"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q14599571": 1,
      "Q14645729": 1,
      "Q14818023": 1,
      "Q14819288": 1,
      "Q14819465": 1,
      "Q14819468": 1,
      "Q14819480": 1,
      "Q14859587": 1,
      "Q14860466": 1,
      "Q14860535": 1,
      "Q14863387": 1,
      "Q14864579": 1,
      "Q14864949": 1,
      "Q14865650": 1,
      "Q14873943": 1,
      "Q14877292": 1,
      "Q14881723": 1,
      "Q14906839": 1,
      "Q14906846": 1,
      "Q1509074": 1,
      "Q21095555": 1,
      "Q21132859": 1,
      "Q332154": 1
    },
    "old_unique": [
      "Q14599571",
      "Q14645729",
      "Q14818023",
      "Q14819288",
      "Q14819465",
      "Q14819468",
      "Q14819480",
      "Q14859587",
      "Q14860466",
      "Q14860535",
      "Q14863387",
      "Q14864579",
      "Q14864949",
      "Q14865650",
      "Q14873943",
      "Q14877292",
      "Q14881723",
      "Q14906839",
      "Q14906846",
      "Q1509074",
      "Q21095555",
      "Q21132859",
      "Q332154"
    ],
    "old_values": [
      "Q14906839",
      "Q14860466",
      "Q14906846",
      "Q14860535",
      "Q14864949",
      "Q14864579",
      "Q332154",
      "Q1509074",
      "Q14819288",
      "Q14819465",
      "Q14865650",
      "Q14859587",
      "Q14645729",
      "Q14881723",
      "Q14873943",
      "Q14863387",
      "Q14819480",
      "Q14877292",
      "Q14818023",
      "Q21095555",
      "Q14599571",
      "Q21132859",
      "Q14819468"
    ],
    "old_values_raw": [
      "Q14906839",
      "Q14860466",
      "Q14906846",
      "Q14860535",
      "Q14864949",
      "Q14864579",
      "Q332154",
      "Q1509074",
      "Q14819288",
      "Q14819465",
      "Q14865650",
      "Q14859587",
      "Q14645729",
      "Q14881723",
      "Q14873943",
      "Q14863387",
      "Q14819480",
      "Q14877292",
      "Q14818023",
      "Q21095555",
      "Q14599571",
      "Q21132859",
      "Q14819468"
    ],
    "removed_unique_values": [
      "Q14819288"
    ],
    "retained_unique_values": [
      "Q14599571",
      "Q14645729",
      "Q14818023",
      "Q14819465",
      "Q14819468",
      "Q14819480",
      "Q14859587",
      "Q14860466",
      "Q14860535",
      "Q14863387",
      "Q14864579",
      "Q14864949",
      "Q14865650",
      "Q14873943",
      "Q14877292",
      "Q14881723",
      "Q14906839",
      "Q14906846",
      "Q1509074",
      "Q21095555",
      "Q21132859",
      "Q332154"
    ],
    "semantic_action": "MIXED_UPDATE",
    "value_multiplicity_changes": {
      "Q14819288": {
        "new": 0,
        "old": 1
      },
      "Q471817": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "Any process that activates or increases the frequency, rate or extent of Rho protein signal transduction.",
    "process whose specific outcome is the progression of the skeleton over time, from its formation to the mature structure",
    "Any process that stops, prevents or reduces the frequency, rate or extent of neuron migration.",
    "Any process that results in a change in state or activity of a cell or an organism (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of a cytokine stimulus.",
    "Any process that determines the size and arrangement of collagen fibrils within an extracellular matrix.",
    "Any process that results in a change in state or activity of a cell or an organism (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of a mechanical stimulus.",
    "biological processes in a human body coupled to increasing age",
    "series of events that restore integrity to a damaged tissue, following an injury",
    "The process whose specific outcome is the progression of the heart over time, from its formation to the mature structure. The heart is a hollow, muscular organ, which, by contracting rhythmically, keeps up the circulation of the blood.",
    "The formation of a covalent cross-link between or within protein chains.",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of the immune response, the immunological reaction of an organism to an immunogenic stimulus.",
    "The process whose specific outcome is the progression of a blood vessel over time, from its formation to the mature structure. The blood vessel is the vasculature carrying blood.",
    "The progression of the cerebral cortex over time from its initial formation until its mature state. The cerebral cortex is the outer layered region of the telencephalon.",
    "Any process that results in a change in state or activity of a cell (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of an amino acid stimulus. An amino acid is a carboxylic acids containing one or more amino gr",
    "series of molecular signals initiated by the binding of extracellular ligand to an integrin on the surface of a target cell, and ending with regulation of a downstream cellular process, e.g. transcription",
    "Any process that results in a change in state or activity of a cell or an organism (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of an electromagnetic radiation stimulus. Electromagnetic radiation is a propag",
    "The binding of a cell to the extracellular matrix via adhesion molecules.",
    "The process whose specific outcome is the progression of the digestive tract over time, from its formation to the mature structure. The digestive tract is the anatomical structure through which food passes and is processed.",
    "series of molecular signals initiated by the binding of an extracellular ligand to a transforming growth factor beta receptor",
    "The process in which the structure of the smooth muscle tissue surrounding the aorta is generated and organized. An aorta is an artery that carries blood from the heart to other parts of the body.",
    "The process whose specific outcome is the progression of the skin over time, from its formation to the mature structure. The skin is the external membranous integument of an animal. In vertebrates the skin generally consists of two layers, an outer n",
    "process that is carried out at the cellular level which results in the assembly, arrangement of constituent parts, or disassembly of a supramolecular fiber, a polymer consisting of an indefinite number of protein or protein complex subunits that ha",
    "process that is carried out at the cellular level which results in the assembly, arrangement of constituent parts, or disassembly of an extracellular matrix"
  ],
  "value_labels_en": [
    "positive regulation of Rho protein signal transduction",
    "skeletal system development",
    "negative regulation of neuron migration",
    "response to cytokine",
    "collagen fibril organization",
    "response to mechanical stimulus",
    "ageing",
    "wound healing",
    "development of the heart",
    "peptide cross-linking",
    "negative regulation of immune response",
    "blood vessel development",
    "cerebral cortex development",
    "cellular response to amino acid stimulus",
    "integrin-mediated signaling pathway",
    "response to radiation",
    "cell-matrix adhesion",
    "digestive tract development",
    "transforming growth factor beta receptor signaling pathway",
    "aorta smooth muscle tissue morphogenesis",
    "skin development",
    "supramolecular fiber organization",
    "extracellular matrix organization"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-11-08T10:02:20",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P682",
  "report_revision_new": 2427161271,
  "report_revision_old": 2423947249,
  "report_violation_type": "Target required claim P|686",
  "report_violation_type_normalized": "Target required claim P|686",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|686",
  "value": [
    "Q14906839",
    "Q14860466",
    "Q14906846",
    "Q14860535",
    "Q14864949",
    "Q14864579",
    "Q332154",
    "Q1509074",
    "Q14819288",
    "Q14819465",
    "Q14865650",
    "Q14859587",
    "Q14645729",
    "Q14881723",
    "Q14873943",
    "Q14863387",
    "Q14819480",
    "Q14877292",
    "Q14818023",
    "Q21095555",
    "Q14599571",
    "Q21132859",
    "Q14819468"
  ],
  "value_descriptions_en": [
    "Any process that activates or increases the frequency, rate or extent of Rho protein signal transduction.",
    "process whose specific outcome is the progression of the skeleton over time, from its formation to the mature structure",
    "Any process that stops, prevents or reduces the frequency, rate or extent of neuron migration.",
    "Any process that results in a change in state or activity of a cell or an organism (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of a cytokine stimulus.",
    "Any process that determines the size and arrangement of collagen fibrils within an extracellular matrix.",
    "Any process that results in a change in state or activity of a cell or an organism (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of a mechanical stimulus.",
    "biological processes in a human body coupled to increasing age",
    "series of events that restore integrity to a damaged tissue, following an injury",
    "The process whose specific outcome is the progression of the heart over time, from its formation to the mature structure. The heart is a hollow, muscular organ, which, by contracting rhythmically, keeps up the circulation of the blood.",
    "The formation of a covalent cross-link between or within protein chains.",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of the immune response, the immunological reaction of an organism to an immunogenic stimulus.",
    "The process whose specific outcome is the progression of a blood vessel over time, from its formation to the mature structure. The blood vessel is the vasculature carrying blood.",
    "The progression of the cerebral cortex over time from its initial formation until its mature state. The cerebral cortex is the outer layered region of the telencephalon.",
    "Any process that results in a change in state or activity of a cell (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of an amino acid stimulus. An amino acid is a carboxylic acids containing one or more amino gr",
    "series of molecular signals initiated by the binding of extracellular ligand to an integrin on the surface of a target cell, and ending with regulation of a downstream cellular process, e.g. transcription",
    "Any process that results in a change in state or activity of a cell or an organism (in terms of movement, secretion, enzyme production, gene expression, etc.) as a result of an electromagnetic radiation stimulus. Electromagnetic radiation is a propag",
    "The binding of a cell to the extracellular matrix via adhesion molecules.",
    "The process whose specific outcome is the progression of the digestive tract over time, from its formation to the mature structure. The digestive tract is the anatomical structure through which food passes and is processed.",
    "series of molecular signals initiated by the binding of an extracellular ligand to a transforming growth factor beta receptor",
    "The process in which the structure of the smooth muscle tissue surrounding the aorta is generated and organized. An aorta is an artery that carries blood from the heart to other parts of the body.",
    "The process whose specific outcome is the progression of the skin over time, from its formation to the mature structure. The skin is the external membranous integument of an animal. In vertebrates the skin generally consists of two layers, an outer n",
    "process that is carried out at the cellular level which results in the assembly, arrangement of constituent parts, or disassembly of a supramolecular fiber, a polymer consisting of an indefinite number of protein or protein complex subunits that ha",
    "process that is carried out at the cellular level which results in the assembly, arrangement of constituent parts, or disassembly of an extracellular matrix"
  ],
  "value_labels_en": [
    "positive regulation of Rho protein signal transduction",
    "skeletal system development",
    "negative regulation of neuron migration",
    "response to cytokine",
    "collagen fibril organization",
    "response to mechanical stimulus",
    "ageing",
    "wound healing",
    "development of the heart",
    "peptide cross-linking",
    "negative regulation of immune response",
    "blood vessel development",
    "cerebral cortex development",
    "cellular response to amino acid stimulus",
    "integrin-mediated signaling pathway",
    "response to radiation",
    "cell-matrix adhesion",
    "digestive tract development",
    "transforming growth factor beta receptor signaling pathway",
    "aorta smooth muscle tissue morphogenesis",
    "skin development",
    "supramolecular fiber organization",
    "extracellular matrix organization"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 67,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q14819288"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q14906839",
      "Q14860466",
      "Q14906846",
      "Q14860535",
      "Q14864949",
      "Q14864579",
      "Q332154",
      "Q1509074",
      "Q14819288",
      "Q14819465",
      "Q14865650",
      "Q14859587",
      "Q14645729",
      "Q14881723",
      "Q14873943",
      "Q14863387",
      "Q14819480",
      "Q14877292",
      "Q14818023",
      "Q21095555",
      "Q14599571",
      "Q21132859",
      "Q14819468"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q14906839",
    "Q14860466",
    "Q14906846",
    "Q14860535",
    "Q14864949",
    "Q14864579",
    "Q332154",
    "Q1509074",
    "Q471817",
    "Q14819465",
    "Q14865650",
    "Q14859587",
    "Q14645729",
    "Q14881723",
    "Q14873943",
    "Q14863387",
    "Q14819480",
    "Q14877292",
    "Q14818023",
    "Q21095555",
    "Q14599571",
    "Q21132859",
    "Q14819468"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "is involved in the biological process",
    "label": "biological process"
  },
  "qid": {
    "description": "mammalian protein found in Mus musculus",
    "label": "Collagen, type III, alpha 1"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
      "local_ids_count": 67,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q14819288"
        }
      ],
      "needed": 2,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q14906839",
        "Q14860466",
        "Q14906846",
        "Q14860535",
        "Q14864949",
        "Q14864579",
        "Q332154",
        "Q1509074",
        "Q14819288",
        "Q14819465",
        "Q14865650",
        "Q14859587",
        "Q14645729",
        "Q14881723",
        "Q14873943",
        "Q14863387",
        "Q14819480",
        "Q14877292",
        "Q14818023",
        "Q21095555",
        "Q14599571",
        "Q21132859",
        "Q14819468"
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

## 011. `repair_Q18205762_2444271422`

| Field | Value |
|---|---|
| qid | Q18205762 |
| property | P735 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q18205762::P735 |
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
| truth_tokens_preview | ["Q4881614"] |
| classification_target_tokens | ["Q4881614"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q4881614"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Q4881614"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q116550025"
  ],
  "removed_unique_values": [
    "Q116550025"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q4881614"
  ],
  "new_value_descriptions_en": [
    "male given name"
  ],
  "new_value_labels_en": [
    "Bekim"
  ],
  "old_value": [
    "Q116550025"
  ],
  "old_value_descriptions_en": [
    "male given name"
  ],
  "old_value_labels_en": [
    "Bekim"
  ],
  "revision_id": 2444271422,
  "value": [
    "Q4881614"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q4881614"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q4881614": 1
    },
    "new_unique": [
      "Q4881614"
    ],
    "new_values": [
      "Q4881614"
    ],
    "new_values_raw": [
      "Q4881614"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q116550025": 1
    },
    "old_unique": [
      "Q116550025"
    ],
    "old_values": [
      "Q116550025"
    ],
    "old_values_raw": [
      "Q116550025"
    ],
    "removed_unique_values": [
      "Q116550025"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q116550025": {
        "new": 0,
        "old": 1
      },
      "Q4881614": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "male given name"
  ],
  "value_labels_en": [
    "Bekim"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-21T09:42:13",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P735",
  "report_revision_new": 2444873428,
  "report_revision_old": 2444444466,
  "report_violation_type": "Target required claim P|1705",
  "report_violation_type_normalized": "Target required claim P|1705",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|1705",
  "report_violation_types": [
    "Target required claim P|1705",
    "Target required claim P|282",
    "Value type Q|202444, Q|49614, Q|122067883"
  ],
  "value": [
    "Q116550025"
  ],
  "value_descriptions_en": [
    "male given name"
  ],
  "value_labels_en": [
    "Bekim"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 24,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q116550025"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q4881614"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "first name or another given name of this person; values used with the property should not link disambiguations nor family names",
    "label": "given name"
  },
  "qid": {
    "description": "Albanian footballer",
    "label": "Bekim Dema"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
      "local_ids_count": 24,
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
        "Q116550025"
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

## 012. `repair_Q2326494_2441312347`

| Field | Value |
|---|---|
| qid | Q2326494 |
| property | P31 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21510851 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q2326494::P31 |
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
| truth_tokens_preview | ["Q1573906"] |
| classification_target_tokens | ["Q1573906"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q1573906"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Q1573906"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q12538685"
  ],
  "removed_unique_values": [
    "Q12538685"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q1573906"
  ],
  "new_value_descriptions_en": [
    "series of concerts by an artist or group of artists in different venues"
  ],
  "new_value_labels_en": [
    "concert tour"
  ],
  "old_value": [
    "Q12538685"
  ],
  "old_value_descriptions_en": [
    "series of concerts by an artist or group of artists in different venues"
  ],
  "old_value_labels_en": [
    "concert tour"
  ],
  "revision_id": 2441312347,
  "value": [
    "Q1573906"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q1573906"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q1573906": 1
    },
    "new_unique": [
      "Q1573906"
    ],
    "new_values": [
      "Q1573906"
    ],
    "new_values_raw": [
      "Q1573906"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q12538685": 1
    },
    "old_unique": [
      "Q12538685"
    ],
    "old_values": [
      "Q12538685"
    ],
    "old_values_raw": [
      "Q12538685"
    ],
    "removed_unique_values": [
      "Q12538685"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q12538685": {
        "new": 0,
        "old": 1
      },
      "Q1573906": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "series of concerts by an artist or group of artists in different venues"
  ],
  "value_labels_en": [
    "concert tour"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T19:49:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
  "report_revision_new": 2442387659,
  "report_revision_old": 2441898308,
  "report_violation_type": "Target required claim P|279",
  "report_violation_type_normalized": "Target required claim P|279",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|279",
  "value": [
    "Q12538685"
  ],
  "value_descriptions_en": [
    "series of concerts by an artist or group of artists in different venues"
  ],
  "value_labels_en": [
    "concert tour"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 10,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q12538685"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q1573906"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "Gira/Tour musical de Madonna",
    "label": "Hung Up Promo Tour"
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
      "local_ids_count": 10,
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
        "Q12538685"
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

## 013. `repair_Q23547_2440621158`

| Field | Value |
|---|---|
| qid | Q23547 |
| property | P69 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q23547::P69 |
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
| truth_tokens_preview | ["Q6682369", "Q5033157"] |
| classification_target_tokens | ["Q1050232", "Q5033157"] |
| classification_target_reason | mixed update classification uses the deterministic changed pair while ignoring retained values |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q5033157"
  ],
  "classification_target_reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
  "classification_target_role": "changed_pair",
  "classification_target_tokens": [
    "Q1050232",
    "Q5033157"
  ],
  "new_changed_value": "Q5033157",
  "old_changed_value": "Q1050232",
  "removed_target_tokens": [
    "Q1050232"
  ],
  "removed_unique_values": [
    "Q1050232"
  ],
  "retained_support_tokens": [
    "Q6682369"
  ],
  "retained_unique_values": [
    "Q6682369"
  ],
  "semantic_action": "MIXED_UPDATE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Simon Villeneuve",
  "kind": "A_BOX",
  "new_value": [
    "Q6682369",
    "Q5033157"
  ],
  "new_value_descriptions_en": [
    "community college in Los Angeles, California, United States",
    "high school in Canoga Park, Los Angeles, California'"
  ],
  "new_value_labels_en": [
    "Los Angeles Valley College",
    "Canoga Park High School"
  ],
  "old_value": [
    "Q6682369",
    "Q1050232"
  ],
  "old_value_descriptions_en": [
    "community college in Los Angeles, California, United States",
    "database of university professors at the University of Rostock"
  ],
  "old_value_labels_en": [
    "Los Angeles Valley College",
    "Catalogus Professorum Rostochiensium"
  ],
  "revision_id": 2440621158,
  "value": [
    "Q6682369",
    "Q5033157"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q5033157"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q5033157": 1,
      "Q6682369": 1
    },
    "new_unique": [
      "Q5033157",
      "Q6682369"
    ],
    "new_values": [
      "Q6682369",
      "Q5033157"
    ],
    "new_values_raw": [
      "Q6682369",
      "Q5033157"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q1050232": 1,
      "Q6682369": 1
    },
    "old_unique": [
      "Q1050232",
      "Q6682369"
    ],
    "old_values": [
      "Q6682369",
      "Q1050232"
    ],
    "old_values_raw": [
      "Q6682369",
      "Q1050232"
    ],
    "removed_unique_values": [
      "Q1050232"
    ],
    "retained_unique_values": [
      "Q6682369"
    ],
    "semantic_action": "MIXED_UPDATE",
    "value_multiplicity_changes": {
      "Q1050232": {
        "new": 0,
        "old": 1
      },
      "Q5033157": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "community college in Los Angeles, California, United States",
    "high school in Canoga Park, Los Angeles, California'"
  ],
  "value_labels_en": [
    "Los Angeles Valley College",
    "Canoga Park High School"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-12T14:30:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P69",
  "report_revision_new": 2441272706,
  "report_revision_old": 2440920529,
  "report_violation_type": "Value type Q|2385804, Q|15690029, Q|23005223, Q|44613, Q|1128397, Q|2023606, Q|16519632, Q|16917, Q|5341295, Q|504703, Q|20860083, Q|15893266, Q|55097243, Q|96086516, Q|32053225",
  "report_violation_type_descriptions_en": [
    "institution that provides education",
    "educational institution existing only in fictional story",
    "partition of an educational institution by level, academic field, history, or another reason",
    "complex of buildings comprising the domestic quarters and workplace(s) of monks or nuns",
    "religious establishment, where clerics lead a religious life in community",
    "structured arrangement of educational activities",
    "group organized for the purpose of scientific research and development",
    "health care facility",
    "body with an aim of education",
    "form of academic instruction",
    "facility for educational activities",
    "entity that no longer operates or is terminated",
    "defunct, destroyed, demolished, or discontinued organization, establishment, group, etc.",
    "former institution that provided education",
    "natural or legal person providing continuous professional training"
  ],
  "report_violation_type_labels_en": [
    "educational institution",
    "fictional educational institution",
    "division of an educational institution",
    "monastery",
    "convent",
    "education program",
    "scientific organization",
    "hospital",
    "educational organization",
    "seminar",
    "educational facility",
    "former entity",
    "defunct organization",
    "former educational institution",
    "training organization"
  ],
  "report_violation_type_normalized": "Value type Q|2385804, Q|15690029, Q|23005223, Q|44613, Q|1128397, Q|2023606, Q|16519632, Q|16917, Q|5341295, Q|504703, Q|20860083, Q|15893266, Q|55097243, Q|96086516, Q|32053225",
  "report_violation_type_qids": [
    "Q2385804",
    "Q15690029",
    "Q23005223",
    "Q44613",
    "Q1128397",
    "Q2023606",
    "Q16519632",
    "Q16917",
    "Q5341295",
    "Q504703",
    "Q20860083",
    "Q15893266",
    "Q55097243",
    "Q96086516",
    "Q32053225"
  ],
  "report_violation_type_raw": "Value type Q|2385804, Q|15690029, Q|23005223, Q|44613, Q|1128397, Q|2023606, Q|16519632, Q|16917, Q|5341295, Q|504703, Q|20860083, Q|15893266, Q|55097243, Q|96086516, Q|32053225",
  "value": [
    "Q6682369",
    "Q1050232"
  ],
  "value_descriptions_en": [
    "community college in Los Angeles, California, United States",
    "database of university professors at the University of Rostock"
  ],
  "value_labels_en": [
    "Los Angeles Valley College",
    "Catalogus Professorum Rostochiensium"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 78,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q1050232"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q6682369",
      "Q1050232"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q6682369",
    "Q5033157"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "educational institution attended by subject",
    "label": "educated at"
  },
  "qid": {
    "description": "American actor, director, and producer",
    "label": "Bryan Cranston"
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
      "local_ids_count": 78,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q1050232"
        }
      ],
      "needed": 2,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q6682369",
        "Q1050232"
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

## 014. `repair_Q28606479_2443314174`

| Field | Value |
|---|---|
| qid | Q28606479 |
| property | P2860 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21510864 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q28606479::P2860 |
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
| truth_tokens_preview | ["Q7810920", "Q5188679", "Q27700202", "Q27008841", "Q30978201", "...(+3)"] |
| classification_target_tokens | ["Q26995469", "Q7810920"] |
| classification_target_reason | mixed update classification uses the deterministic changed pair while ignoring retained values |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q7810920"
  ],
  "classification_target_reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
  "classification_target_role": "changed_pair",
  "classification_target_tokens": [
    "Q26995469",
    "Q7810920"
  ],
  "new_changed_value": "Q7810920",
  "old_changed_value": "Q26995469",
  "removed_target_tokens": [
    "Q26995469"
  ],
  "removed_unique_values": [
    "Q26995469"
  ],
  "retained_support_tokens": [
    "Q27008841",
    "Q27700202",
    "Q30978201",
    "Q33576085",
    "Q35669497",
    "Q43951182",
    "Q5188679"
  ],
  "retained_unique_values": [
    "Q27008841",
    "Q27700202",
    "Q30978201",
    "Q33576085",
    "Q35669497",
    "Q43951182",
    "Q5188679"
  ],
  "semantic_action": "MIXED_UPDATE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q7810920",
    "Q5188679",
    "Q27700202",
    "Q27008841",
    "Q30978201",
    "Q35669497",
    "Q33576085",
    "Q43951182"
  ],
  "new_value_descriptions_en": [
    "landmark report in medicine",
    "book by Institute of Medicine",
    "scientific article (publication date: 2013)",
    "scientific article published on May 3, 2013",
    "scientific article",
    "scientific article published on June 2015",
    "scientific article published on April 25, 2014",
    "scientific article"
  ],
  "new_value_labels_en": [
    "To Err is Human: Building a Safer Health System",
    "Crossing the Quality Chasm",
    "Reassessing Google Flu Trends Data for Detection of Seasonal and Pandemic Influenza: A Comparative Epidemiological Study at Three Geographic Scales",
    "Abacavir Pharmacogenetics – From Initial Reports to Standard of Care",
    "Google Flu Trends in Canada: a comparison of digital disease surveillance data with physician consultations and respiratory virus surveillance data, 2010-2014.",
    "Physicians' behavior following changes in LDL cholesterol target goals",
    "Low rate of non-attenders to primary care providers in Israel - a retrospective longitudinal study",
    "Data curation: Act to staunch loss of research data"
  ],
  "old_value": [
    "Q26995469",
    "Q5188679",
    "Q27700202",
    "Q27008841",
    "Q30978201",
    "Q35669497",
    "Q33576085",
    "Q43951182"
  ],
  "old_value_descriptions_en": [
    "landmark report in medicine",
    "book by Institute of Medicine",
    "scientific article (publication date: 2013)",
    "scientific article published on May 3, 2013",
    "scientific article",
    "scientific article published on June 2015",
    "scientific article published on April 25, 2014",
    "scientific article"
  ],
  "old_value_labels_en": [
    "To Err is Human: Building a Safer Health System",
    "Crossing the Quality Chasm",
    "Reassessing Google Flu Trends Data for Detection of Seasonal and Pandemic Influenza: A Comparative Epidemiological Study at Three Geographic Scales",
    "Abacavir Pharmacogenetics – From Initial Reports to Standard of Care",
    "Google Flu Trends in Canada: a comparison of digital disease surveillance data with physician consultations and respiratory virus surveillance data, 2010-2014.",
    "Physicians' behavior following changes in LDL cholesterol target goals",
    "Low rate of non-attenders to primary care providers in Israel - a retrospective longitudinal study",
    "Data curation: Act to staunch loss of research data"
  ],
  "revision_id": 2443314174,
  "value": [
    "Q7810920",
    "Q5188679",
    "Q27700202",
    "Q27008841",
    "Q30978201",
    "Q35669497",
    "Q33576085",
    "Q43951182"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q7810920"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q27008841": 1,
      "Q27700202": 1,
      "Q30978201": 1,
      "Q33576085": 1,
      "Q35669497": 1,
      "Q43951182": 1,
      "Q5188679": 1,
      "Q7810920": 1
    },
    "new_unique": [
      "Q27008841",
      "Q27700202",
      "Q30978201",
      "Q33576085",
      "Q35669497",
      "Q43951182",
      "Q5188679",
      "Q7810920"
    ],
    "new_values": [
      "Q7810920",
      "Q5188679",
      "Q27700202",
      "Q27008841",
      "Q30978201",
      "Q35669497",
      "Q33576085",
      "Q43951182"
    ],
    "new_values_raw": [
      "Q7810920",
      "Q5188679",
      "Q27700202",
      "Q27008841",
      "Q30978201",
      "Q35669497",
      "Q33576085",
      "Q43951182"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q26995469": 1,
      "Q27008841": 1,
      "Q27700202": 1,
      "Q30978201": 1,
      "Q33576085": 1,
      "Q35669497": 1,
      "Q43951182": 1,
      "Q5188679": 1
    },
    "old_unique": [
      "Q26995469",
      "Q27008841",
      "Q27700202",
      "Q30978201",
      "Q33576085",
      "Q35669497",
      "Q43951182",
      "Q5188679"
    ],
    "old_values": [
      "Q26995469",
      "Q5188679",
      "Q27700202",
      "Q27008841",
      "Q30978201",
      "Q35669497",
      "Q33576085",
      "Q43951182"
    ],
    "old_values_raw": [
      "Q26995469",
      "Q5188679",
      "Q27700202",
      "Q27008841",
      "Q30978201",
      "Q35669497",
      "Q33576085",
      "Q43951182"
    ],
    "removed_unique_values": [
      "Q26995469"
    ],
    "retained_unique_values": [
      "Q27008841",
      "Q27700202",
      "Q30978201",
      "Q33576085",
      "Q35669497",
      "Q43951182",
      "Q5188679"
    ],
    "semantic_action": "MIXED_UPDATE",
    "value_multiplicity_changes": {
      "Q26995469": {
        "new": 0,
        "old": 1
      },
      "Q7810920": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "landmark report in medicine",
    "book by Institute of Medicine",
    "scientific article (publication date: 2013)",
    "scientific article published on May 3, 2013",
    "scientific article",
    "scientific article published on June 2015",
    "scientific article published on April 25, 2014",
    "scientific article"
  ],
  "value_labels_en": [
    "To Err is Human: Building a Safer Health System",
    "Crossing the Quality Chasm",
    "Reassessing Google Flu Trends Data for Detection of Seasonal and Pandemic Influenza: A Comparative Epidemiological Study at Three Geographic Scales",
    "Abacavir Pharmacogenetics – From Initial Reports to Standard of Care",
    "Google Flu Trends in Canada: a comparison of digital disease surveillance data with physician consultations and respiratory virus surveillance data, 2010-2014.",
    "Physicians' behavior following changes in LDL cholesterol target goals",
    "Low rate of non-attenders to primary care providers in Israel - a retrospective longitudinal study",
    "Data curation: Act to staunch loss of research data"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-18T16:27:57",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2860",
  "report_revision_new": 2443799318,
  "report_revision_old": 2443351534,
  "report_violation_type": "Target required claim P|1476",
  "report_violation_type_normalized": "Target required claim P|1476",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|1476",
  "value": [
    "Q26995469",
    "Q5188679",
    "Q27700202",
    "Q27008841",
    "Q30978201",
    "Q35669497",
    "Q33576085",
    "Q43951182"
  ],
  "value_descriptions_en": [
    "landmark report in medicine",
    "book by Institute of Medicine",
    "scientific article (publication date: 2013)",
    "scientific article published on May 3, 2013",
    "scientific article",
    "scientific article published on June 2015",
    "scientific article published on April 25, 2014",
    "scientific article"
  ],
  "value_labels_en": [
    "To Err is Human: Building a Safer Health System",
    "Crossing the Quality Chasm",
    "Reassessing Google Flu Trends Data for Detection of Seasonal and Pandemic Influenza: A Comparative Epidemiological Study at Three Geographic Scales",
    "Abacavir Pharmacogenetics – From Initial Reports to Standard of Care",
    "Google Flu Trends in Canada: a comparison of digital disease surveillance data with physician consultations and respiratory virus surveillance data, 2010-2014.",
    "Physicians' behavior following changes in LDL cholesterol target goals",
    "Low rate of non-attenders to primary care providers in Israel - a retrospective longitudinal study",
    "Data curation: Act to staunch loss of research data"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 23,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q26995469"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q26995469",
      "Q5188679",
      "Q27700202",
      "Q27008841",
      "Q30978201",
      "Q35669497",
      "Q33576085",
      "Q43951182"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q7810920",
    "Q5188679",
    "Q27700202",
    "Q27008841",
    "Q30978201",
    "Q35669497",
    "Q33576085",
    "Q43951182"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "scientific article",
    "label": "Big Data in Israeli healthcare: hopes and challenges report of an international workshop"
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
      "local_ids_count": 23,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q26995469"
        }
      ],
      "needed": 2,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q26995469",
        "Q5188679",
        "Q27700202",
        "Q27008841",
        "Q30978201",
        "Q35669497",
        "Q33576085",
        "Q43951182"
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

## 015. `repair_Q2889467_2441312395`

| Field | Value |
|---|---|
| qid | Q2889467 |
| property | P31 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21510851 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q2889467::P31 |
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
| truth_tokens_preview | ["Q1573906"] |
| classification_target_tokens | ["Q1573906"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q1573906"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Q1573906"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q12538685"
  ],
  "removed_unique_values": [
    "Q12538685"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q1573906"
  ],
  "new_value_descriptions_en": [
    "series of concerts by an artist or group of artists in different venues"
  ],
  "new_value_labels_en": [
    "concert tour"
  ],
  "old_value": [
    "Q12538685"
  ],
  "old_value_descriptions_en": [
    "series of concerts by an artist or group of artists in different venues"
  ],
  "old_value_labels_en": [
    "concert tour"
  ],
  "revision_id": 2441312395,
  "value": [
    "Q1573906"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q1573906"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q1573906": 1
    },
    "new_unique": [
      "Q1573906"
    ],
    "new_values": [
      "Q1573906"
    ],
    "new_values_raw": [
      "Q1573906"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q12538685": 1
    },
    "old_unique": [
      "Q12538685"
    ],
    "old_values": [
      "Q12538685"
    ],
    "old_values_raw": [
      "Q12538685"
    ],
    "removed_unique_values": [
      "Q12538685"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q12538685": {
        "new": 0,
        "old": 1
      },
      "Q1573906": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "series of concerts by an artist or group of artists in different venues"
  ],
  "value_labels_en": [
    "concert tour"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T19:49:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
  "report_revision_new": 2442387659,
  "report_revision_old": 2441898308,
  "report_violation_type": "Target required claim P|279",
  "report_violation_type_normalized": "Target required claim P|279",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|279",
  "value": [
    "Q12538685"
  ],
  "value_descriptions_en": [
    "series of concerts by an artist or group of artists in different venues"
  ],
  "value_labels_en": [
    "concert tour"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 10,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q12538685"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q1573906"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "tour by Paramore",
    "label": "The Final Riot! Tour"
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
      "local_ids_count": 10,
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
        "Q12538685"
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

## 016. `repair_Q308952_2444448124`

| Field | Value |
|---|---|
| qid | Q308952 |
| property | P749 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q308952::P749 |
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
| truth_tokens_preview | ["Q137477114"] |
| classification_target_tokens | ["Q137477114"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q137477114"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q137477114"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "CREATE_FROM_MISSING"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
  "constraint_family": null,
  "decision_constraint_source": "classifier_rule",
  "decision_constraint_type_label": null,
  "decision_constraint_type_qid": null
}
```

#### Repair Target

```json
{
  "action": "CREATE",
  "author": "Rulwarih",
  "kind": "A_BOX",
  "new_value": [
    "Q137477114"
  ],
  "new_value_descriptions_en": [
    "aviation company in Malaysia"
  ],
  "new_value_labels_en": [
    "Malaysia Aviation Group"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2444448124,
  "value": [
    "Q137477114"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q137477114"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q137477114": 1
    },
    "new_unique": [
      "Q137477114"
    ],
    "new_values": [
      "Q137477114"
    ],
    "new_values_raw": [
      "Q137477114"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {},
    "old_unique": [],
    "old_values": [],
    "old_values_raw": [
      "MISSING"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [],
    "semantic_action": "CREATE_FROM_MISSING",
    "value_multiplicity_changes": {
      "Q137477114": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "aviation company in Malaysia"
  ],
  "value_labels_en": [
    "Malaysia Aviation Group"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-21T09:37:10",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P749",
  "report_revision_new": 2444872115,
  "report_revision_old": 2444443672,
  "report_violation_type": "Inverse",
  "report_violation_type_normalized": "Inverse",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Inverse",
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
  "local_ids_count": 53,
  "local_support_for_retained_value": [],
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
    "Q137477114"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "parent organization or unit of an organization or unit, opposite of child organization or unit (P355); use instance of (P31) to distinguish organization (Q43229) and organization unit (Q10387680)",
    "label": "parent organization or unit"
  },
  "qid": {
    "description": "flag-carrier airline of Malaysia",
    "label": "Malaysia Airlines"
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
    "label_en": "inverse constraint",
    "qid": "Q21510855"
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
      "local_ids_count": 53,
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

## 017. `repair_Q3182098_2445431242`

| Field | Value |
|---|---|
| qid | Q3182098 |
| property | P54 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21510865 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q3182098::P54 |
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
| truth_tokens_preview | ["Q55801", "Q104901906"] |
| classification_target_tokens | ["Q217635", "Q104901906"] |
| classification_target_reason | mixed update classification uses the deterministic changed pair while ignoring retained values |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q104901906"
  ],
  "classification_target_reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
  "classification_target_role": "changed_pair",
  "classification_target_tokens": [
    "Q217635",
    "Q104901906"
  ],
  "new_changed_value": "Q104901906",
  "old_changed_value": "Q217635",
  "removed_target_tokens": [
    "Q217635"
  ],
  "removed_unique_values": [
    "Q217635"
  ],
  "retained_support_tokens": [
    "Q55801"
  ],
  "retained_unique_values": [
    "Q55801"
  ],
  "semantic_action": "MIXED_UPDATE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Eugo",
  "kind": "A_BOX",
  "new_value": [
    "Q55801",
    "Q104901906"
  ],
  "new_value_descriptions_en": [
    "men's rugby union team of New Zealand",
    "rugby union team"
  ],
  "new_value_labels_en": [
    "New Zealand national rugby union team",
    "Waikato Rugby Union Team"
  ],
  "old_value": [
    "Q55801",
    "Q217635"
  ],
  "old_value_descriptions_en": [
    "men's rugby union team of New Zealand",
    "sports club"
  ],
  "old_value_labels_en": [
    "New Zealand national rugby union team",
    "Waikato RU"
  ],
  "revision_id": 2445431242,
  "value": [
    "Q55801",
    "Q104901906"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q104901906"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q104901906": 1,
      "Q55801": 1
    },
    "new_unique": [
      "Q104901906",
      "Q55801"
    ],
    "new_values": [
      "Q55801",
      "Q104901906"
    ],
    "new_values_raw": [
      "Q55801",
      "Q104901906"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q217635": 1,
      "Q55801": 1
    },
    "old_unique": [
      "Q217635",
      "Q55801"
    ],
    "old_values": [
      "Q55801",
      "Q217635"
    ],
    "old_values_raw": [
      "Q55801",
      "Q217635"
    ],
    "removed_unique_values": [
      "Q217635"
    ],
    "retained_unique_values": [
      "Q55801"
    ],
    "semantic_action": "MIXED_UPDATE",
    "value_multiplicity_changes": {
      "Q104901906": {
        "new": 1,
        "old": 0
      },
      "Q217635": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "men's rugby union team of New Zealand",
    "rugby union team"
  ],
  "value_labels_en": [
    "New Zealand national rugby union team",
    "Waikato Rugby Union Team"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-23T16:24:38",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P54",
  "report_revision_new": 2446118185,
  "report_revision_old": 2445511063,
  "report_violation_type": "Value type Q|847017, Q|12973014, Q|115898316, Q|989470, Q|98767736, Q|56850094, Q|25178247, Q|1066670",
  "report_violation_type_descriptions_en": [
    "organization for the purpose of playing one or more sports",
    "individual team that plays sports",
    "part of a sports club that is usually focused on one sport or competition category",
    "organized group of video game players",
    "sports club only appearing in works of fiction",
    "sports team in a work of fiction",
    "group of 3 or more professional wrestlers who work together",
    "team of multiple wrestlers"
  ],
  "report_violation_type_labels_en": [
    "sports club",
    "sports team",
    "department of a sports club",
    "clan",
    "fictional sports club",
    "fictional sports team",
    "professional wrestling stable",
    "tag team"
  ],
  "report_violation_type_normalized": "Value type Q|847017, Q|12973014, Q|115898316, Q|989470, Q|98767736, Q|56850094, Q|25178247, Q|1066670",
  "report_violation_type_qids": [
    "Q847017",
    "Q12973014",
    "Q115898316",
    "Q989470",
    "Q98767736",
    "Q56850094",
    "Q25178247",
    "Q1066670"
  ],
  "report_violation_type_raw": "Value type Q|847017, Q|12973014, Q|115898316, Q|989470, Q|98767736, Q|56850094, Q|25178247, Q|1066670",
  "value": [
    "Q55801",
    "Q217635"
  ],
  "value_descriptions_en": [
    "men's rugby union team of New Zealand",
    "sports club"
  ],
  "value_labels_en": [
    "New Zealand national rugby union team",
    "Waikato RU"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 27,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q217635"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q55801",
      "Q217635"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q55801",
    "Q104901906"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "sports teams or clubs that the subject represents or represented",
    "label": "member of sports team"
  },
  "qid": {
    "description": "New Zealand rugby union footballer and coach (b. 1964)",
    "label": "John Mitchell"
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
      "found": 1,
      "independent_match_count": 0,
      "local_ids_count": 27,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q217635"
        }
      ],
      "needed": 2,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q55801",
        "Q217635"
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

## 018. `repair_Q34318765_2441936274`

| Field | Value |
|---|---|
| qid | Q34318765 |
| property | P50 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q34318765::P50 |
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
| truth_tokens_preview | ["Q58941021"] |
| classification_target_tokens | ["Q58941021"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q58941021"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Q58941021"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q66829366"
  ],
  "removed_unique_values": [
    "Q66829366"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q58941021"
  ],
  "new_value_descriptions_en": [
    "researcher"
  ],
  "new_value_labels_en": [
    "Shyam Sundar"
  ],
  "old_value": [
    "Q66829366"
  ],
  "old_value_descriptions_en": [
    "researcher"
  ],
  "old_value_labels_en": [
    "Shyam Sundar"
  ],
  "revision_id": 2441936274,
  "value": [
    "Q58941021"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q58941021"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q58941021": 1
    },
    "new_unique": [
      "Q58941021"
    ],
    "new_values": [
      "Q58941021"
    ],
    "new_values_raw": [
      "Q58941021"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q66829366": 1
    },
    "old_unique": [
      "Q66829366"
    ],
    "old_values": [
      "Q66829366"
    ],
    "old_values_raw": [
      "Q66829366"
    ],
    "removed_unique_values": [
      "Q66829366"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q58941021": {
        "new": 1,
        "old": 0
      },
      "Q66829366": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "researcher"
  ],
  "value_labels_en": [
    "Shyam Sundar"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-15T15:28:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P50",
  "report_revision_new": 2442717404,
  "report_revision_old": 2442328968,
  "report_violation_type": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "report_violation_type_descriptions_en": [
    "being that has certain capacities or attributes constituting personhood (for humans, use Q5 [human] with P31 [instance of])",
    "fictitious name that a person or group assumes for a particular purpose, which differs from their original or true name (orthonym)",
    "social entity established to meet needs or pursue goals",
    "any set of human beings",
    "fictional human or non-human character in a narrative work of art",
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "field of computer science that develops and studies software enabling machines to exhibit intelligent behavior",
    "human being whose existence is not directly attested, but is deduced by other evidence",
    "intelligent actor with unnatural origin",
    "distinct and identifiable entity with agency, capable of performing actions",
    "copyright is owned or retained by the party that commissioned it or by the employer of the person who produced it",
    null
  ],
  "report_violation_type_labels_en": [
    "person",
    "pseudonym",
    "organization",
    "group of humans",
    "character",
    "human",
    "artificial intelligence",
    "hypothetical person",
    "artificially intelligent entity",
    "being",
    "corporate authorship",
    "corporate author"
  ],
  "report_violation_type_normalized": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "report_violation_type_qids": [
    "Q215627",
    "Q61002",
    "Q43229",
    "Q16334295",
    "Q95074",
    "Q5",
    "Q11660",
    "Q75855169",
    "Q107307291",
    "Q24229398",
    "Q78056559",
    "Q60557912"
  ],
  "report_violation_type_raw": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "value": [
    "Q66829366"
  ],
  "value_descriptions_en": [
    "researcher"
  ],
  "value_labels_en": [
    "Shyam Sundar"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 69,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q66829366"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q58941021"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "main creator(s) of a written work (use on works, not humans); use P2093 (author name string) when Wikidata item is unknown or does not exist",
    "label": "author"
  },
  "qid": {
    "description": "scientific article published on December 21, 2012",
    "label": "Leishmaniasis: an update of current pharmacotherapy"
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
      "local_ids_count": 69,
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
        "Q66829366"
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

## 019. `repair_Q353637_2445405927`

| Field | Value |
|---|---|
| qid | Q353637 |
| property | P19 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q353637::P19 |
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
| truth_tokens_preview | ["Q12358639"] |
| classification_target_tokens | ["Q12358639"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q12358639"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Q12358639"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q720155"
  ],
  "removed_unique_values": [
    "Q720155"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "~2025-32979-98",
  "kind": "A_BOX",
  "new_value": [
    "Q12358639"
  ],
  "new_value_descriptions_en": [
    "former municipality of Estonia (1866–1939)"
  ],
  "new_value_labels_en": [
    "Albu Rural Municipality"
  ],
  "old_value": [
    "Q720155"
  ],
  "old_value_descriptions_en": [
    "former municipality of Estonia (1991–2017)"
  ],
  "old_value_labels_en": [
    "Albu Rural Municipality"
  ],
  "revision_id": 2445405927,
  "value": [
    "Q12358639"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q12358639"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q12358639": 1
    },
    "new_unique": [
      "Q12358639"
    ],
    "new_values": [
      "Q12358639"
    ],
    "new_values_raw": [
      "Q12358639"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q720155": 1
    },
    "old_unique": [
      "Q720155"
    ],
    "old_values": [
      "Q720155"
    ],
    "old_values_raw": [
      "Q720155"
    ],
    "removed_unique_values": [
      "Q720155"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q12358639": {
        "new": 1,
        "old": 0
      },
      "Q720155": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "former municipality of Estonia (1866–1939)"
  ],
  "value_labels_en": [
    "Albu Rural Municipality"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-23T18:24:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P19",
  "report_revision_new": 2446158327,
  "report_revision_old": 2445541860,
  "report_violation_type": "Contemporary",
  "report_violation_type_normalized": "Contemporary",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Contemporary",
  "value": [
    "Q720155"
  ],
  "value_descriptions_en": [
    "former municipality of Estonia (1991–2017)"
  ],
  "value_labels_en": [
    "Albu Rural Municipality"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 64,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q720155"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q12358639"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "most specific known birth location of a person, animal or fictional character",
    "label": "place of birth"
  },
  "qid": {
    "description": "Estonian writer (1878–1940)",
    "label": "A. H. Tammsaare"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
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
      "local_ids_count": 64,
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
        "Q720155"
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

## 020. `repair_Q3617809_2441932428`

| Field | Value |
|---|---|
| qid | Q3617809 |
| property | P734 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q3617809::P734 |
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
| truth_tokens_preview | ["Q63116520"] |
| classification_target_tokens | ["Q63116520"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q63116520"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Q63116520"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q113394122"
  ],
  "removed_unique_values": [
    "Q113394122"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q63116520"
  ],
  "new_value_descriptions_en": [
    "family name"
  ],
  "new_value_labels_en": [
    "Rigon"
  ],
  "old_value": [
    "Q113394122"
  ],
  "old_value_descriptions_en": [
    "family name"
  ],
  "old_value_labels_en": [
    "Rigon"
  ],
  "revision_id": 2441932428,
  "value": [
    "Q63116520"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q63116520"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q63116520": 1
    },
    "new_unique": [
      "Q63116520"
    ],
    "new_values": [
      "Q63116520"
    ],
    "new_values_raw": [
      "Q63116520"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q113394122": 1
    },
    "old_unique": [
      "Q113394122"
    ],
    "old_values": [
      "Q113394122"
    ],
    "old_values_raw": [
      "Q113394122"
    ],
    "removed_unique_values": [
      "Q113394122"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q113394122": {
        "new": 0,
        "old": 1
      },
      "Q63116520": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "family name"
  ],
  "value_labels_en": [
    "Rigon"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-15T10:41:13",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P734",
  "report_revision_new": 2442616055,
  "report_revision_old": 2442267617,
  "report_violation_type": "Value type Q|101352, Q|110874, Q|66475447",
  "report_violation_type_descriptions_en": [
    "part of a naming scheme for individuals, used in many cultures worldwide",
    "component of a personal name based on the given name of one's father or other male ancestor",
    "word that is attached to a person's last name, indicating ethnic, familiar, or geographical origin"
  ],
  "report_violation_type_labels_en": [
    "family name",
    "patronymic",
    "family name affix"
  ],
  "report_violation_type_normalized": "Value type Q|101352, Q|110874, Q|66475447",
  "report_violation_type_qids": [
    "Q101352",
    "Q110874",
    "Q66475447"
  ],
  "report_violation_type_raw": "Value type Q|101352, Q|110874, Q|66475447",
  "value": [
    "Q113394122"
  ],
  "value_descriptions_en": [
    "family name"
  ],
  "value_labels_en": [
    "Rigon"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 16,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q113394122"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q63116520"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "part of full name of person",
    "label": "family name"
  },
  "qid": {
    "description": "Italian model",
    "label": "Anna Rigon"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
      "local_ids_count": 16,
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
        "Q113394122"
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

## 021. `repair_Q41286795_2441933756`

| Field | Value |
|---|---|
| qid | Q41286795 |
| property | P50 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q41286795::P50 |
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
| truth_tokens_preview | ["Q87195557", "Q17129531", "Q102145078", "Q90182471", "Q89341552", "...(+3)"] |
| classification_target_tokens | ["Q102228927", "Q17129531"] |
| classification_target_reason | mixed update classification uses the deterministic changed pair while ignoring retained values |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q17129531"
  ],
  "classification_target_reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
  "classification_target_role": "changed_pair",
  "classification_target_tokens": [
    "Q102228927",
    "Q17129531"
  ],
  "new_changed_value": "Q17129531",
  "old_changed_value": "Q102228927",
  "removed_target_tokens": [
    "Q102228927"
  ],
  "removed_unique_values": [
    "Q102228927"
  ],
  "retained_support_tokens": [
    "Q102145078",
    "Q114422322",
    "Q114422323",
    "Q85798921",
    "Q87195557",
    "Q89341552",
    "Q90182471"
  ],
  "retained_unique_values": [
    "Q102145078",
    "Q114422322",
    "Q114422323",
    "Q85798921",
    "Q87195557",
    "Q89341552",
    "Q90182471"
  ],
  "semantic_action": "MIXED_UPDATE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q87195557",
    "Q17129531",
    "Q102145078",
    "Q90182471",
    "Q89341552",
    "Q114422322",
    "Q114422323",
    "Q85798921"
  ],
  "new_value_descriptions_en": [
    "researcher",
    "Japanese ophthalmologist",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher"
  ],
  "new_value_labels_en": [
    "Chie Sotozono",
    "Shigeru Kinoshita",
    "Tsutomu Inatomi",
    "Koji Kitazawa",
    "Osamu Hieda",
    "Kanae Kayukawa",
    "Koichi Wakimasu",
    "Isao Yokota"
  ],
  "old_value": [
    "Q87195557",
    "Q102228927",
    "Q102145078",
    "Q90182471",
    "Q89341552",
    "Q114422322",
    "Q114422323",
    "Q85798921"
  ],
  "old_value_descriptions_en": [
    "researcher",
    "Japanese ophthalmologist",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher"
  ],
  "old_value_labels_en": [
    "Chie Sotozono",
    "Shigeru Kinoshita",
    "Tsutomu Inatomi",
    "Koji Kitazawa",
    "Osamu Hieda",
    "Kanae Kayukawa",
    "Koichi Wakimasu",
    "Isao Yokota"
  ],
  "revision_id": 2441933756,
  "value": [
    "Q87195557",
    "Q17129531",
    "Q102145078",
    "Q90182471",
    "Q89341552",
    "Q114422322",
    "Q114422323",
    "Q85798921"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q17129531"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q102145078": 1,
      "Q114422322": 1,
      "Q114422323": 1,
      "Q17129531": 1,
      "Q85798921": 1,
      "Q87195557": 1,
      "Q89341552": 1,
      "Q90182471": 1
    },
    "new_unique": [
      "Q102145078",
      "Q114422322",
      "Q114422323",
      "Q17129531",
      "Q85798921",
      "Q87195557",
      "Q89341552",
      "Q90182471"
    ],
    "new_values": [
      "Q87195557",
      "Q17129531",
      "Q102145078",
      "Q90182471",
      "Q89341552",
      "Q114422322",
      "Q114422323",
      "Q85798921"
    ],
    "new_values_raw": [
      "Q87195557",
      "Q17129531",
      "Q102145078",
      "Q90182471",
      "Q89341552",
      "Q114422322",
      "Q114422323",
      "Q85798921"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q102145078": 1,
      "Q102228927": 1,
      "Q114422322": 1,
      "Q114422323": 1,
      "Q85798921": 1,
      "Q87195557": 1,
      "Q89341552": 1,
      "Q90182471": 1
    },
    "old_unique": [
      "Q102145078",
      "Q102228927",
      "Q114422322",
      "Q114422323",
      "Q85798921",
      "Q87195557",
      "Q89341552",
      "Q90182471"
    ],
    "old_values": [
      "Q87195557",
      "Q102228927",
      "Q102145078",
      "Q90182471",
      "Q89341552",
      "Q114422322",
      "Q114422323",
      "Q85798921"
    ],
    "old_values_raw": [
      "Q87195557",
      "Q102228927",
      "Q102145078",
      "Q90182471",
      "Q89341552",
      "Q114422322",
      "Q114422323",
      "Q85798921"
    ],
    "removed_unique_values": [
      "Q102228927"
    ],
    "retained_unique_values": [
      "Q102145078",
      "Q114422322",
      "Q114422323",
      "Q85798921",
      "Q87195557",
      "Q89341552",
      "Q90182471"
    ],
    "semantic_action": "MIXED_UPDATE",
    "value_multiplicity_changes": {
      "Q102228927": {
        "new": 0,
        "old": 1
      },
      "Q17129531": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "researcher",
    "Japanese ophthalmologist",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher"
  ],
  "value_labels_en": [
    "Chie Sotozono",
    "Shigeru Kinoshita",
    "Tsutomu Inatomi",
    "Koji Kitazawa",
    "Osamu Hieda",
    "Kanae Kayukawa",
    "Koichi Wakimasu",
    "Isao Yokota"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-15T15:28:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P50",
  "report_revision_new": 2442717404,
  "report_revision_old": 2442328968,
  "report_violation_type": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "report_violation_type_descriptions_en": [
    "being that has certain capacities or attributes constituting personhood (for humans, use Q5 [human] with P31 [instance of])",
    "fictitious name that a person or group assumes for a particular purpose, which differs from their original or true name (orthonym)",
    "social entity established to meet needs or pursue goals",
    "any set of human beings",
    "fictional human or non-human character in a narrative work of art",
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "field of computer science that develops and studies software enabling machines to exhibit intelligent behavior",
    "human being whose existence is not directly attested, but is deduced by other evidence",
    "intelligent actor with unnatural origin",
    "distinct and identifiable entity with agency, capable of performing actions",
    "copyright is owned or retained by the party that commissioned it or by the employer of the person who produced it",
    null
  ],
  "report_violation_type_labels_en": [
    "person",
    "pseudonym",
    "organization",
    "group of humans",
    "character",
    "human",
    "artificial intelligence",
    "hypothetical person",
    "artificially intelligent entity",
    "being",
    "corporate authorship",
    "corporate author"
  ],
  "report_violation_type_normalized": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "report_violation_type_qids": [
    "Q215627",
    "Q61002",
    "Q43229",
    "Q16334295",
    "Q95074",
    "Q5",
    "Q11660",
    "Q75855169",
    "Q107307291",
    "Q24229398",
    "Q78056559",
    "Q60557912"
  ],
  "report_violation_type_raw": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "value": [
    "Q87195557",
    "Q102228927",
    "Q102145078",
    "Q90182471",
    "Q89341552",
    "Q114422322",
    "Q114422323",
    "Q85798921"
  ],
  "value_descriptions_en": [
    "researcher",
    "Japanese ophthalmologist",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher"
  ],
  "value_labels_en": [
    "Chie Sotozono",
    "Shigeru Kinoshita",
    "Tsutomu Inatomi",
    "Koji Kitazawa",
    "Osamu Hieda",
    "Kanae Kayukawa",
    "Koichi Wakimasu",
    "Isao Yokota"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 57,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q102228927"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q87195557",
      "Q102228927",
      "Q102145078",
      "Q90182471",
      "Q89341552",
      "Q114422322",
      "Q114422323",
      "Q85798921"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q87195557",
    "Q17129531",
    "Q102145078",
    "Q90182471",
    "Q89341552",
    "Q114422322",
    "Q114422323",
    "Q85798921"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "main creator(s) of a written work (use on works, not humans); use P2093 (author name string) when Wikidata item is unknown or does not exist",
    "label": "author"
  },
  "qid": {
    "description": "scientific article",
    "label": "Predictive clinical factors of cystoid macular edema in patients with Descemet's stripping automated endothelial keratoplasty"
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
      "found": 1,
      "independent_match_count": 0,
      "local_ids_count": 57,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q102228927"
        }
      ],
      "needed": 2,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q87195557",
        "Q102228927",
        "Q102145078",
        "Q90182471",
        "Q89341552",
        "Q114422322",
        "Q114422323",
        "Q85798921"
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

## 022. `repair_Q462843_2437787877`

| Field | Value |
|---|---|
| qid | Q462843 |
| property | P22 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q462843::P22 |
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
| truth_tokens_preview | ["Q69148050"] |
| classification_target_tokens | ["Q69148050"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q69148050"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Q69148050"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q83349710"
  ],
  "removed_unique_values": [
    "Q83349710"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Codename Noreste",
  "kind": "A_BOX",
  "new_value": [
    "Q69148050"
  ],
  "new_value_descriptions_en": [
    null
  ],
  "new_value_labels_en": [
    "Tu Liangui"
  ],
  "old_value": [
    "Q83349710"
  ],
  "old_value_descriptions_en": [
    "family name"
  ],
  "old_value_labels_en": [
    "Follardt"
  ],
  "revision_id": 2437787877,
  "value": [
    "Q69148050"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q69148050"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q69148050": 1
    },
    "new_unique": [
      "Q69148050"
    ],
    "new_values": [
      "Q69148050"
    ],
    "new_values_raw": [
      "Q69148050"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q83349710": 1
    },
    "old_unique": [
      "Q83349710"
    ],
    "old_values": [
      "Q83349710"
    ],
    "old_values_raw": [
      "Q83349710"
    ],
    "removed_unique_values": [
      "Q83349710"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q69148050": {
        "new": 1,
        "old": 0
      },
      "Q83349710": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    null
  ],
  "value_labels_en": [
    "Tu Liangui"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-10T13:13:23",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P22",
  "report_revision_new": 2440444433,
  "report_revision_old": 2440089347,
  "report_violation_type": "Target required claim P|21 one of Q|44148, Q|48270, Q|1052281, Q|2449503, Q|6581097, Q|11287467, Q|15145778, Q|69990794",
  "report_violation_type_descriptions_en": [
    "organism of the male sex",
    "gender identity that exists outside of the gender binary",
    "woman assigned male at birth",
    "man who was assigned female at birth",
    "to be used in \"sex or gender\" (P21) to indicate that the human subject is a male or \"semantic gender\" (P10339) to indicate that a word refers to a male person",
    "person born with any of several variations in sex characteristics including chromosomes, gonads, sex hormones or genitals that do not fit the typical definitions for male or female bodies",
    "man who was assigned male at birth and identifies as male",
    "group of persons with non-binary gender"
  ],
  "report_violation_type_labels_en": [
    "male organism",
    "non-binary",
    "trans woman",
    "trans man",
    "male",
    "intersex person",
    "cisgender man",
    "non-binary people"
  ],
  "report_violation_type_normalized": "Target required claim P|21 one of Q|44148, Q|48270, Q|1052281, Q|2449503, Q|6581097, Q|11287467, Q|15145778, Q|69990794",
  "report_violation_type_qids": [
    "Q44148",
    "Q48270",
    "Q1052281",
    "Q2449503",
    "Q6581097",
    "Q11287467",
    "Q15145778",
    "Q69990794"
  ],
  "report_violation_type_raw": "Target required claim P|21 one of Q|44148, Q|48270, Q|1052281, Q|2449503, Q|6581097, Q|11287467, Q|15145778, Q|69990794",
  "value": [
    "Q83349710"
  ],
  "value_descriptions_en": [
    "family name"
  ],
  "value_labels_en": [
    "Follardt"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 80,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q83349710"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q69148050"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "male parent of the subject. For stepfather, use \"stepparent\" (P3448)",
    "label": "father"
  },
  "qid": {
    "description": "Chinese medical scientist",
    "label": "Tu Youyou"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
  },
  {
    "label_en": "inverse constraint",
    "qid": "Q21510855"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
      "local_ids_count": 80,
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
        "Q83349710"
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

## 023. `repair_Q4765237_2441298332`

| Field | Value |
|---|---|
| qid | Q4765237 |
| property | P735 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q4765237::P735 |
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
| truth_tokens_preview | ["Q4587552", "Q3061675"] |
| classification_target_tokens | ["Q15731812", "Q3061675"] |
| classification_target_reason | mixed update classification uses the deterministic changed pair while ignoring retained values |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q3061675"
  ],
  "classification_target_reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
  "classification_target_role": "changed_pair",
  "classification_target_tokens": [
    "Q15731812",
    "Q3061675"
  ],
  "new_changed_value": "Q3061675",
  "old_changed_value": "Q15731812",
  "removed_target_tokens": [
    "Q15731812"
  ],
  "removed_unique_values": [
    "Q15731812"
  ],
  "retained_support_tokens": [
    "Q4587552"
  ],
  "retained_unique_values": [
    "Q4587552"
  ],
  "semantic_action": "MIXED_UPDATE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q4587552",
    "Q3061675"
  ],
  "new_value_descriptions_en": [
    "female given name",
    "female given name"
  ],
  "new_value_labels_en": [
    "Anine",
    "Ewa"
  ],
  "old_value": [
    "Q4587552",
    "Q15731812"
  ],
  "old_value_descriptions_en": [
    "female given name",
    "female given name"
  ],
  "old_value_labels_en": [
    "Anine",
    "Ewa"
  ],
  "revision_id": 2441298332,
  "value": [
    "Q4587552",
    "Q3061675"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q3061675"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q3061675": 1,
      "Q4587552": 1
    },
    "new_unique": [
      "Q3061675",
      "Q4587552"
    ],
    "new_values": [
      "Q4587552",
      "Q3061675"
    ],
    "new_values_raw": [
      "Q4587552",
      "Q3061675"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q15731812": 1,
      "Q4587552": 1
    },
    "old_unique": [
      "Q15731812",
      "Q4587552"
    ],
    "old_values": [
      "Q4587552",
      "Q15731812"
    ],
    "old_values_raw": [
      "Q4587552",
      "Q15731812"
    ],
    "removed_unique_values": [
      "Q15731812"
    ],
    "retained_unique_values": [
      "Q4587552"
    ],
    "semantic_action": "MIXED_UPDATE",
    "value_multiplicity_changes": {
      "Q15731812": {
        "new": 0,
        "old": 1
      },
      "Q3061675": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "female given name",
    "female given name"
  ],
  "value_labels_en": [
    "Anine",
    "Ewa"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T11:09:13",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P735",
  "report_revision_new": 2442267914,
  "report_revision_old": 2441753351,
  "report_violation_type": "Target required claim P|1705",
  "report_violation_type_normalized": "Target required claim P|1705",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|1705",
  "report_violation_types": [
    "Target required claim P|1705",
    "Target required claim P|282",
    "Value type Q|202444, Q|49614, Q|122067883"
  ],
  "value": [
    "Q4587552",
    "Q15731812"
  ],
  "value_descriptions_en": [
    "female given name",
    "female given name"
  ],
  "value_labels_en": [
    "Anine",
    "Ewa"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 47,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q15731812"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q4587552",
      "Q15731812"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q4587552",
    "Q3061675"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "first name or another given name of this person; values used with the property should not link disambiguations nor family names",
    "label": "given name"
  },
  "qid": {
    "description": "Norwegian lawyer",
    "label": "Anine Kierulf"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
      "local_ids_count": 47,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q15731812"
        }
      ],
      "needed": 2,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q4587552",
        "Q15731812"
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

## 024. `repair_Q48082861_2442856329`

| Field | Value |
|---|---|
| qid | Q48082861 |
| property | P50 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q48082861::P50 |
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
| truth_tokens_preview | ["Q64763841", "Q1917336"] |
| classification_target_tokens | ["Q90797849", "Q1917336"] |
| classification_target_reason | mixed update classification uses the deterministic changed pair while ignoring retained values |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q1917336"
  ],
  "classification_target_reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
  "classification_target_role": "changed_pair",
  "classification_target_tokens": [
    "Q90797849",
    "Q1917336"
  ],
  "new_changed_value": "Q1917336",
  "old_changed_value": "Q90797849",
  "removed_target_tokens": [
    "Q90797849"
  ],
  "removed_unique_values": [
    "Q90797849"
  ],
  "retained_support_tokens": [
    "Q64763841"
  ],
  "retained_unique_values": [
    "Q64763841"
  ],
  "semantic_action": "MIXED_UPDATE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q64763841",
    "Q1917336"
  ],
  "new_value_descriptions_en": [
    "Turkish medical researcher",
    "Turkish surgeon (born 1944)"
  ],
  "new_value_labels_en": [
    "Gökhan Moray",
    "Mehmet Haberal"
  ],
  "old_value": [
    "Q64763841",
    "Q90797849"
  ],
  "old_value_descriptions_en": [
    "Turkish medical researcher",
    "Turkish surgeon (born 1944)"
  ],
  "old_value_labels_en": [
    "Gökhan Moray",
    "Mehmet Haberal"
  ],
  "revision_id": 2442856329,
  "value": [
    "Q64763841",
    "Q1917336"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q1917336"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q1917336": 1,
      "Q64763841": 1
    },
    "new_unique": [
      "Q1917336",
      "Q64763841"
    ],
    "new_values": [
      "Q64763841",
      "Q1917336"
    ],
    "new_values_raw": [
      "Q64763841",
      "Q1917336"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q64763841": 1,
      "Q90797849": 1
    },
    "old_unique": [
      "Q64763841",
      "Q90797849"
    ],
    "old_values": [
      "Q64763841",
      "Q90797849"
    ],
    "old_values_raw": [
      "Q64763841",
      "Q90797849"
    ],
    "removed_unique_values": [
      "Q90797849"
    ],
    "retained_unique_values": [
      "Q64763841"
    ],
    "semantic_action": "MIXED_UPDATE",
    "value_multiplicity_changes": {
      "Q1917336": {
        "new": 1,
        "old": 0
      },
      "Q90797849": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "Turkish medical researcher",
    "Turkish surgeon (born 1944)"
  ],
  "value_labels_en": [
    "Gökhan Moray",
    "Mehmet Haberal"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-17T16:00:46",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P50",
  "report_revision_new": 2443454316,
  "report_revision_old": 2443014505,
  "report_violation_type": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "report_violation_type_descriptions_en": [
    "being that has certain capacities or attributes constituting personhood (for humans, use Q5 [human] with P31 [instance of])",
    "fictitious name that a person or group assumes for a particular purpose, which differs from their original or true name (orthonym)",
    "social entity established to meet needs or pursue goals",
    "any set of human beings",
    "fictional human or non-human character in a narrative work of art",
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "field of computer science that develops and studies software enabling machines to exhibit intelligent behavior",
    "human being whose existence is not directly attested, but is deduced by other evidence",
    "intelligent actor with unnatural origin",
    "distinct and identifiable entity with agency, capable of performing actions",
    "copyright is owned or retained by the party that commissioned it or by the employer of the person who produced it",
    null
  ],
  "report_violation_type_labels_en": [
    "person",
    "pseudonym",
    "organization",
    "group of humans",
    "character",
    "human",
    "artificial intelligence",
    "hypothetical person",
    "artificially intelligent entity",
    "being",
    "corporate authorship",
    "corporate author"
  ],
  "report_violation_type_normalized": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "report_violation_type_qids": [
    "Q215627",
    "Q61002",
    "Q43229",
    "Q16334295",
    "Q95074",
    "Q5",
    "Q11660",
    "Q75855169",
    "Q107307291",
    "Q24229398",
    "Q78056559",
    "Q60557912"
  ],
  "report_violation_type_raw": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398, Q|78056559, Q|60557912",
  "value": [
    "Q64763841",
    "Q90797849"
  ],
  "value_descriptions_en": [
    "Turkish medical researcher",
    "Turkish surgeon (born 1944)"
  ],
  "value_labels_en": [
    "Gökhan Moray",
    "Mehmet Haberal"
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
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q90797849"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q64763841",
      "Q90797849"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q64763841",
    "Q1917336"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "main creator(s) of a written work (use on works, not humans); use P2093 (author name string) when Wikidata item is unknown or does not exist",
    "label": "author"
  },
  "qid": {
    "description": "scientific article published in March 2014",
    "label": "The history of liver transplantation in Turkey."
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
      "found": 1,
      "independent_match_count": 0,
      "local_ids_count": 9,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q90797849"
        }
      ],
      "needed": 2,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q64763841",
        "Q90797849"
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

## 025. `repair_Q4941561_2426561999`

| Field | Value |
|---|---|
| qid | Q4941561 |
| property | P682 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q4941561::P682 |
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
| truth_tokens_preview | ["Q14633911", "Q14903145", "Q14859574", "Q14901698", "Q14858821", "...(+27)"] |
| classification_target_tokens | ["Q14819288", "Q471817"] |
| classification_target_reason | mixed update classification uses the deterministic changed pair while ignoring retained values |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q471817"
  ],
  "classification_target_reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
  "classification_target_role": "changed_pair",
  "classification_target_tokens": [
    "Q14819288",
    "Q471817"
  ],
  "new_changed_value": "Q471817",
  "old_changed_value": "Q14819288",
  "removed_target_tokens": [
    "Q14819288"
  ],
  "removed_unique_values": [
    "Q14819288"
  ],
  "retained_support_tokens": [
    "Q14599698",
    "Q14633893",
    "Q14633911",
    "Q14645705",
    "Q14818066",
    "Q14852037",
    "Q14858821",
    "Q14859574",
    "Q14859611",
    "Q14859937",
    "Q14859963",
    "Q14864202",
    "Q14864806",
    "Q14865277",
    "Q14865337",
    "Q14865464",
    "Q14888623",
    "Q14889336",
    "Q14901689",
    "Q14901698",
    "Q14903088",
    "Q14903145",
    "Q14903147",
    "Q14903586",
    "... omitted 7 items"
  ],
  "retained_unique_values": [
    "Q14599698",
    "Q14633893",
    "Q14633911",
    "Q14645705",
    "Q14818066",
    "Q14852037",
    "Q14858821",
    "Q14859574",
    "Q14859611",
    "Q14859937",
    "Q14859963",
    "Q14864202",
    "Q14864806",
    "Q14865277",
    "Q14865337",
    "Q14865464",
    "Q14888623",
    "Q14889336",
    "Q14901689",
    "Q14901698",
    "Q14903088",
    "Q14903145",
    "Q14903147",
    "Q14903586",
    "... omitted 7 items"
  ],
  "semantic_action": "MIXED_UPDATE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q14633911",
    "Q14903145",
    "Q14859574",
    "Q14901698",
    "Q14858821",
    "Q14901689",
    "Q14865277",
    "Q14903088",
    "Q14903590",
    "Q14903591",
    "Q14889336",
    "Q14859611",
    "Q14903586",
    "Q14818066",
    "Q14852037",
    "Q14645705",
    "Q471817",
    "Q14633893",
    "Q187640",
    "Q14859937",
    "Q14865337",
    "Q14903147",
    "Q14599698",
    "Q21171694",
    "... omitted 10 items"
  ],
  "new_value_descriptions_en": [
    "any process that modulates the occurrence or rate of cell death by apoptotic process",
    "The process of introducing a phosphate group on to a pathway restricted SMAD protein. A pathway restricted SMAD protein is an effector protein that acts directly downstream of the transforming growth factor family receptor.",
    "Any process that modulates the frequency, rate or extent of signal transduction mediated by the MAP kinase (MAPK) cascade.",
    "The process of creating a trabecula in the heart. A trabecula is a tissue element in the form of a small beam, strut or rod.",
    "The cascade of processes by which a signal interacts with a receptor, causing a change in the activity of a SMAD protein, and ultimately effecting a change in the functioning of the cell.",
    "Any process that decreases the rate, frequency or extent of the enlargement or overgrowth of all or part of the heart due to an increase in size (not length) of individual cardiac muscle fibers, without cell division.",
    "The process whose specific outcome is the progression of the adult heart over time, from its formation to the mature structure.",
    "Any process that increases the rate, frequency or extent of pathway-restricted SMAD protein phosphorylation. Pathway-restricted SMAD proteins and common-partner SMAD proteins are involved in the transforming growth factor beta receptor signaling path",
    "Any process that increases the rate, frequency or extent myofibril assembly by organization of muscle actomyosin into sarcomeres. The sarcomere is the repeating unit of a myofibril in a muscle cell, composed of an array of overlapping thick and thin",
    "Any process that activates or increases the frequency, rate or extent of cell proliferation involved in heart morphogenesis.",
    "The expansion of a cardiac muscle cell population by cell division.",
    "Any process that increases the rate, frequency or extent of the enlargement or overgrowth of all or part of the heart due to an increase in size (not length) of individual cardiac muscle fibers, without cell division.",
    "Any process that decreases the rate, frequency, or extent of the orderly movement of an endothelial cell into the extracellular matrix to form an endothelium.",
    "any process that activates or increases the frequency, rate or extent of cellular DNA-templated transcription",
    "The process in which the anatomical structures of cardiac ventricle muscle is generated and organized.",
    "biological process whose specific outcome is the progression of a multicellular organism over time from an initial condition (e.g. a zygote or a young adult) to a later condition (e.g. a multicellular animal or an aged adult)",
    "The process whose specific outcome is the progression of the heart over time, from its formation to the mature structure. The heart is a hollow, muscular organ, which, by contracting rhythmically, keeps up the circulation of the blood.",
    "any process that increases the frequency, rate or extent of gene expression",
    "attachment of a cell, to another cell or to an underlying substrate, by cell adhesion molecules",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of cell migration.",
    "Any process that activates or increases the frequency, rate or extent of cardiac muscle cell proliferation.",
    "Any process that increases the rate, frequency, or extent of cartilage development, the process whose specific outcome is the progression of the cartilage over time, from its formation to the mature structure. Cartilage is a connective tissue dominat",
    "Any process that stops, prevents, or reduces the frequency, rate, extent or direction of cell growth.",
    "Any process that modulates the frequency, rate or extent of cardiac muscle hypertrophy in response to stress.",
    "... omitted 10 items"
  ],
  "new_value_labels_en": [
    "regulation of apoptotic process",
    "pathway-restricted SMAD protein phosphorylation",
    "regulation of MAPK cascade",
    "heart trabecula formation",
    "SMAD protein signal transduction",
    "negative regulation of cardiac muscle hypertrophy",
    "adult heart development",
    "positive regulation of pathway-restricted SMAD protein phosphorylation",
    "positive regulation of sarcomere organization",
    "positive regulation of cell proliferation involved in heart morphogenesis",
    "cardiac muscle cell proliferation",
    "positive regulation of cardiac muscle hypertrophy",
    "negative regulation of endothelial cell migration",
    "positive regulation of transcription, DNA-templated",
    "ventricular cardiac muscle tissue morphogenesis",
    "multicellular organism development",
    "development of the heart",
    "positive regulation of gene expression",
    "cell adhesion",
    "negative regulation of cell migration",
    "positive regulation of cardiac muscle cell proliferation",
    "positive regulation of cartilage development",
    "negative regulation of cell growth",
    "regulation of cardiac muscle hypertrophy in response to stress",
    "... omitted 10 items"
  ],
  "old_value": [
    "Q14633911",
    "Q14903145",
    "Q14859574",
    "Q14901698",
    "Q14858821",
    "Q14901689",
    "Q14865277",
    "Q14903088",
    "Q14903590",
    "Q14903591",
    "Q14889336",
    "Q14859611",
    "Q14903586",
    "Q14818066",
    "Q14852037",
    "Q14645705",
    "Q14819288",
    "Q14633893",
    "Q187640",
    "Q14859937",
    "Q14865337",
    "Q14903147",
    "Q14599698",
    "Q21171694",
    "... omitted 10 items"
  ],
  "old_value_descriptions_en": [
    "any process that modulates the occurrence or rate of cell death by apoptotic process",
    "The process of introducing a phosphate group on to a pathway restricted SMAD protein. A pathway restricted SMAD protein is an effector protein that acts directly downstream of the transforming growth factor family receptor.",
    "Any process that modulates the frequency, rate or extent of signal transduction mediated by the MAP kinase (MAPK) cascade.",
    "The process of creating a trabecula in the heart. A trabecula is a tissue element in the form of a small beam, strut or rod.",
    "The cascade of processes by which a signal interacts with a receptor, causing a change in the activity of a SMAD protein, and ultimately effecting a change in the functioning of the cell.",
    "Any process that decreases the rate, frequency or extent of the enlargement or overgrowth of all or part of the heart due to an increase in size (not length) of individual cardiac muscle fibers, without cell division.",
    "The process whose specific outcome is the progression of the adult heart over time, from its formation to the mature structure.",
    "Any process that increases the rate, frequency or extent of pathway-restricted SMAD protein phosphorylation. Pathway-restricted SMAD proteins and common-partner SMAD proteins are involved in the transforming growth factor beta receptor signaling path",
    "Any process that increases the rate, frequency or extent myofibril assembly by organization of muscle actomyosin into sarcomeres. The sarcomere is the repeating unit of a myofibril in a muscle cell, composed of an array of overlapping thick and thin",
    "Any process that activates or increases the frequency, rate or extent of cell proliferation involved in heart morphogenesis.",
    "The expansion of a cardiac muscle cell population by cell division.",
    "Any process that increases the rate, frequency or extent of the enlargement or overgrowth of all or part of the heart due to an increase in size (not length) of individual cardiac muscle fibers, without cell division.",
    "Any process that decreases the rate, frequency, or extent of the orderly movement of an endothelial cell into the extracellular matrix to form an endothelium.",
    "any process that activates or increases the frequency, rate or extent of cellular DNA-templated transcription",
    "The process in which the anatomical structures of cardiac ventricle muscle is generated and organized.",
    "biological process whose specific outcome is the progression of a multicellular organism over time from an initial condition (e.g. a zygote or a young adult) to a later condition (e.g. a multicellular animal or an aged adult)",
    "The process whose specific outcome is the progression of the heart over time, from its formation to the mature structure. The heart is a hollow, muscular organ, which, by contracting rhythmically, keeps up the circulation of the blood.",
    "any process that increases the frequency, rate or extent of gene expression",
    "attachment of a cell, to another cell or to an underlying substrate, by cell adhesion molecules",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of cell migration.",
    "Any process that activates or increases the frequency, rate or extent of cardiac muscle cell proliferation.",
    "Any process that increases the rate, frequency, or extent of cartilage development, the process whose specific outcome is the progression of the cartilage over time, from its formation to the mature structure. Cartilage is a connective tissue dominat",
    "Any process that stops, prevents, or reduces the frequency, rate, extent or direction of cell growth.",
    "Any process that modulates the frequency, rate or extent of cardiac muscle hypertrophy in response to stress.",
    "... omitted 10 items"
  ],
  "old_value_labels_en": [
    "regulation of apoptotic process",
    "pathway-restricted SMAD protein phosphorylation",
    "regulation of MAPK cascade",
    "heart trabecula formation",
    "SMAD protein signal transduction",
    "negative regulation of cardiac muscle hypertrophy",
    "adult heart development",
    "positive regulation of pathway-restricted SMAD protein phosphorylation",
    "positive regulation of sarcomere organization",
    "positive regulation of cell proliferation involved in heart morphogenesis",
    "cardiac muscle cell proliferation",
    "positive regulation of cardiac muscle hypertrophy",
    "negative regulation of endothelial cell migration",
    "positive regulation of transcription, DNA-templated",
    "ventricular cardiac muscle tissue morphogenesis",
    "multicellular organism development",
    "development of the heart",
    "positive regulation of gene expression",
    "cell adhesion",
    "negative regulation of cell migration",
    "positive regulation of cardiac muscle cell proliferation",
    "positive regulation of cartilage development",
    "negative regulation of cell growth",
    "regulation of cardiac muscle hypertrophy in response to stress",
    "... omitted 10 items"
  ],
  "revision_id": 2426561999,
  "value": [
    "Q14633911",
    "Q14903145",
    "Q14859574",
    "Q14901698",
    "Q14858821",
    "Q14901689",
    "Q14865277",
    "Q14903088",
    "Q14903590",
    "Q14903591",
    "Q14889336",
    "Q14859611",
    "Q14903586",
    "Q14818066",
    "Q14852037",
    "Q14645705",
    "Q471817",
    "Q14633893",
    "Q187640",
    "Q14859937",
    "Q14865337",
    "Q14903147",
    "Q14599698",
    "Q21171694",
    "... omitted 10 items"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q471817"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q14599698": 1,
      "Q14633893": 1,
      "Q14633911": 1,
      "Q14645705": 1,
      "Q14818066": 1,
      "Q14852037": 1,
      "Q14858821": 1,
      "Q14859574": 1,
      "Q14859611": 1,
      "Q14859937": 1,
      "Q14859963": 1,
      "Q14864202": 1,
      "Q14864806": 2,
      "Q14865277": 1,
      "Q14865337": 1,
      "Q14865464": 1,
      "Q14888623": 1,
      "Q14889336": 1,
      "Q14901689": 1,
      "Q14901698": 1,
      "Q14903088": 2,
      "Q14903145": 1,
      "Q14903147": 1,
      "Q14903586": 1,
      "Q14903587": 1,
      "Q14903588": 1,
      "Q14903589": 1,
      "Q14903590": 1,
      "Q14903591": 1,
      "Q187640": 1,
      "Q21171694": 1,
      "Q471817": 1
    },
    "new_unique": [
      "Q14599698",
      "Q14633893",
      "Q14633911",
      "Q14645705",
      "Q14818066",
      "Q14852037",
      "Q14858821",
      "Q14859574",
      "Q14859611",
      "Q14859937",
      "Q14859963",
      "Q14864202",
      "Q14864806",
      "Q14865277",
      "Q14865337",
      "Q14865464",
      "Q14888623",
      "Q14889336",
      "Q14901689",
      "Q14901698",
      "Q14903088",
      "Q14903145",
      "Q14903147",
      "Q14903586",
      "... omitted 8 items"
    ],
    "new_values": [
      "Q14633911",
      "Q14903145",
      "Q14859574",
      "Q14901698",
      "Q14858821",
      "Q14901689",
      "Q14865277",
      "Q14903088",
      "Q14903590",
      "Q14903591",
      "Q14889336",
      "Q14859611",
      "Q14903586",
      "Q14818066",
      "Q14852037",
      "Q14645705",
      "Q471817",
      "Q14633893",
      "Q187640",
      "Q14859937",
      "Q14865337",
      "Q14903147",
      "Q14599698",
      "Q21171694",
      "... omitted 10 items"
    ],
    "new_values_raw": [
      "Q14633911",
      "Q14903145",
      "Q14859574",
      "Q14901698",
      "Q14858821",
      "Q14901689",
      "Q14865277",
      "Q14903088",
      "Q14903590",
      "Q14903591",
      "Q14889336",
      "Q14859611",
      "Q14903586",
      "Q14818066",
      "Q14852037",
      "Q14645705",
      "Q471817",
      "Q14633893",
      "Q187640",
      "Q14859937",
      "Q14865337",
      "Q14903147",
      "Q14599698",
      "Q21171694",
      "... omitted 10 items"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q14599698": 1,
      "Q14633893": 1,
      "Q14633911": 1,
      "Q14645705": 1,
      "Q14818066": 1,
      "Q14819288": 1,
      "Q14852037": 1,
      "Q14858821": 1,
      "Q14859574": 1,
      "Q14859611": 1,
      "Q14859937": 1,
      "Q14859963": 1,
      "Q14864202": 1,
      "Q14864806": 2,
      "Q14865277": 1,
      "Q14865337": 1,
      "Q14865464": 1,
      "Q14888623": 1,
      "Q14889336": 1,
      "Q14901689": 1,
      "Q14901698": 1,
      "Q14903088": 2,
      "Q14903145": 1,
      "Q14903147": 1,
      "Q14903586": 1,
      "Q14903587": 1,
      "Q14903588": 1,
      "Q14903589": 1,
      "Q14903590": 1,
      "Q14903591": 1,
      "Q187640": 1,
      "Q21171694": 1
    },
    "old_unique": [
      "Q14599698",
      "Q14633893",
      "Q14633911",
      "Q14645705",
      "Q14818066",
      "Q14819288",
      "Q14852037",
      "Q14858821",
      "Q14859574",
      "Q14859611",
      "Q14859937",
      "Q14859963",
      "Q14864202",
      "Q14864806",
      "Q14865277",
      "Q14865337",
      "Q14865464",
      "Q14888623",
      "Q14889336",
      "Q14901689",
      "Q14901698",
      "Q14903088",
      "Q14903145",
      "Q14903147",
      "... omitted 8 items"
    ],
    "old_values": [
      "Q14633911",
      "Q14903145",
      "Q14859574",
      "Q14901698",
      "Q14858821",
      "Q14901689",
      "Q14865277",
      "Q14903088",
      "Q14903590",
      "Q14903591",
      "Q14889336",
      "Q14859611",
      "Q14903586",
      "Q14818066",
      "Q14852037",
      "Q14645705",
      "Q14819288",
      "Q14633893",
      "Q187640",
      "Q14859937",
      "Q14865337",
      "Q14903147",
      "Q14599698",
      "Q21171694",
      "... omitted 10 items"
    ],
    "old_values_raw": [
      "Q14633911",
      "Q14903145",
      "Q14859574",
      "Q14901698",
      "Q14858821",
      "Q14901689",
      "Q14865277",
      "Q14903088",
      "Q14903590",
      "Q14903591",
      "Q14889336",
      "Q14859611",
      "Q14903586",
      "Q14818066",
      "Q14852037",
      "Q14645705",
      "Q14819288",
      "Q14633893",
      "Q187640",
      "Q14859937",
      "Q14865337",
      "Q14903147",
      "Q14599698",
      "Q21171694",
      "... omitted 10 items"
    ],
    "removed_unique_values": [
      "Q14819288"
    ],
    "retained_unique_values": [
      "Q14599698",
      "Q14633893",
      "Q14633911",
      "Q14645705",
      "Q14818066",
      "Q14852037",
      "Q14858821",
      "Q14859574",
      "Q14859611",
      "Q14859937",
      "Q14859963",
      "Q14864202",
      "Q14864806",
      "Q14865277",
      "Q14865337",
      "Q14865464",
      "Q14888623",
      "Q14889336",
      "Q14901689",
      "Q14901698",
      "Q14903088",
      "Q14903145",
      "Q14903147",
      "Q14903586",
      "... omitted 7 items"
    ],
    "semantic_action": "MIXED_UPDATE",
    "value_multiplicity_changes": {
      "Q14819288": {
        "new": 0,
        "old": 1
      },
      "Q471817": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "any process that modulates the occurrence or rate of cell death by apoptotic process",
    "The process of introducing a phosphate group on to a pathway restricted SMAD protein. A pathway restricted SMAD protein is an effector protein that acts directly downstream of the transforming growth factor family receptor.",
    "Any process that modulates the frequency, rate or extent of signal transduction mediated by the MAP kinase (MAPK) cascade.",
    "The process of creating a trabecula in the heart. A trabecula is a tissue element in the form of a small beam, strut or rod.",
    "The cascade of processes by which a signal interacts with a receptor, causing a change in the activity of a SMAD protein, and ultimately effecting a change in the functioning of the cell.",
    "Any process that decreases the rate, frequency or extent of the enlargement or overgrowth of all or part of the heart due to an increase in size (not length) of individual cardiac muscle fibers, without cell division.",
    "The process whose specific outcome is the progression of the adult heart over time, from its formation to the mature structure.",
    "Any process that increases the rate, frequency or extent of pathway-restricted SMAD protein phosphorylation. Pathway-restricted SMAD proteins and common-partner SMAD proteins are involved in the transforming growth factor beta receptor signaling path",
    "Any process that increases the rate, frequency or extent myofibril assembly by organization of muscle actomyosin into sarcomeres. The sarcomere is the repeating unit of a myofibril in a muscle cell, composed of an array of overlapping thick and thin",
    "Any process that activates or increases the frequency, rate or extent of cell proliferation involved in heart morphogenesis.",
    "The expansion of a cardiac muscle cell population by cell division.",
    "Any process that increases the rate, frequency or extent of the enlargement or overgrowth of all or part of the heart due to an increase in size (not length) of individual cardiac muscle fibers, without cell division.",
    "Any process that decreases the rate, frequency, or extent of the orderly movement of an endothelial cell into the extracellular matrix to form an endothelium.",
    "any process that activates or increases the frequency, rate or extent of cellular DNA-templated transcription",
    "The process in which the anatomical structures of cardiac ventricle muscle is generated and organized.",
    "biological process whose specific outcome is the progression of a multicellular organism over time from an initial condition (e.g. a zygote or a young adult) to a later condition (e.g. a multicellular animal or an aged adult)",
    "The process whose specific outcome is the progression of the heart over time, from its formation to the mature structure. The heart is a hollow, muscular organ, which, by contracting rhythmically, keeps up the circulation of the blood.",
    "any process that increases the frequency, rate or extent of gene expression",
    "attachment of a cell, to another cell or to an underlying substrate, by cell adhesion molecules",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of cell migration.",
    "Any process that activates or increases the frequency, rate or extent of cardiac muscle cell proliferation.",
    "Any process that increases the rate, frequency, or extent of cartilage development, the process whose specific outcome is the progression of the cartilage over time, from its formation to the mature structure. Cartilage is a connective tissue dominat",
    "Any process that stops, prevents, or reduces the frequency, rate, extent or direction of cell growth.",
    "Any process that modulates the frequency, rate or extent of cardiac muscle hypertrophy in response to stress.",
    "... omitted 10 items"
  ],
  "value_labels_en": [
    "regulation of apoptotic process",
    "pathway-restricted SMAD protein phosphorylation",
    "regulation of MAPK cascade",
    "heart trabecula formation",
    "SMAD protein signal transduction",
    "negative regulation of cardiac muscle hypertrophy",
    "adult heart development",
    "positive regulation of pathway-restricted SMAD protein phosphorylation",
    "positive regulation of sarcomere organization",
    "positive regulation of cell proliferation involved in heart morphogenesis",
    "cardiac muscle cell proliferation",
    "positive regulation of cardiac muscle hypertrophy",
    "negative regulation of endothelial cell migration",
    "positive regulation of transcription, DNA-templated",
    "ventricular cardiac muscle tissue morphogenesis",
    "multicellular organism development",
    "development of the heart",
    "positive regulation of gene expression",
    "cell adhesion",
    "negative regulation of cell migration",
    "positive regulation of cardiac muscle cell proliferation",
    "positive regulation of cartilage development",
    "negative regulation of cell growth",
    "regulation of cardiac muscle hypertrophy in response to stress",
    "... omitted 10 items"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-11-08T10:02:20",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P682",
  "report_revision_new": 2427161271,
  "report_revision_old": 2423947249,
  "report_violation_type": "Target required claim P|686",
  "report_violation_type_normalized": "Target required claim P|686",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|686",
  "value": [
    "Q14633911",
    "Q14903145",
    "Q14859574",
    "Q14901698",
    "Q14858821",
    "Q14901689",
    "Q14865277",
    "Q14903088",
    "Q14903590",
    "Q14903591",
    "Q14889336",
    "Q14859611",
    "Q14903586",
    "Q14818066",
    "Q14852037",
    "Q14645705",
    "Q14819288",
    "Q14633893",
    "Q187640",
    "Q14859937",
    "Q14865337",
    "Q14903147",
    "Q14599698",
    "Q21171694",
    "... omitted 10 items"
  ],
  "value_descriptions_en": [
    "any process that modulates the occurrence or rate of cell death by apoptotic process",
    "The process of introducing a phosphate group on to a pathway restricted SMAD protein. A pathway restricted SMAD protein is an effector protein that acts directly downstream of the transforming growth factor family receptor.",
    "Any process that modulates the frequency, rate or extent of signal transduction mediated by the MAP kinase (MAPK) cascade.",
    "The process of creating a trabecula in the heart. A trabecula is a tissue element in the form of a small beam, strut or rod.",
    "The cascade of processes by which a signal interacts with a receptor, causing a change in the activity of a SMAD protein, and ultimately effecting a change in the functioning of the cell.",
    "Any process that decreases the rate, frequency or extent of the enlargement or overgrowth of all or part of the heart due to an increase in size (not length) of individual cardiac muscle fibers, without cell division.",
    "The process whose specific outcome is the progression of the adult heart over time, from its formation to the mature structure.",
    "Any process that increases the rate, frequency or extent of pathway-restricted SMAD protein phosphorylation. Pathway-restricted SMAD proteins and common-partner SMAD proteins are involved in the transforming growth factor beta receptor signaling path",
    "Any process that increases the rate, frequency or extent myofibril assembly by organization of muscle actomyosin into sarcomeres. The sarcomere is the repeating unit of a myofibril in a muscle cell, composed of an array of overlapping thick and thin",
    "Any process that activates or increases the frequency, rate or extent of cell proliferation involved in heart morphogenesis.",
    "The expansion of a cardiac muscle cell population by cell division.",
    "Any process that increases the rate, frequency or extent of the enlargement or overgrowth of all or part of the heart due to an increase in size (not length) of individual cardiac muscle fibers, without cell division.",
    "Any process that decreases the rate, frequency, or extent of the orderly movement of an endothelial cell into the extracellular matrix to form an endothelium.",
    "any process that activates or increases the frequency, rate or extent of cellular DNA-templated transcription",
    "The process in which the anatomical structures of cardiac ventricle muscle is generated and organized.",
    "biological process whose specific outcome is the progression of a multicellular organism over time from an initial condition (e.g. a zygote or a young adult) to a later condition (e.g. a multicellular animal or an aged adult)",
    "The process whose specific outcome is the progression of the heart over time, from its formation to the mature structure. The heart is a hollow, muscular organ, which, by contracting rhythmically, keeps up the circulation of the blood.",
    "any process that increases the frequency, rate or extent of gene expression",
    "attachment of a cell, to another cell or to an underlying substrate, by cell adhesion molecules",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of cell migration.",
    "Any process that activates or increases the frequency, rate or extent of cardiac muscle cell proliferation.",
    "Any process that increases the rate, frequency, or extent of cartilage development, the process whose specific outcome is the progression of the cartilage over time, from its formation to the mature structure. Cartilage is a connective tissue dominat",
    "Any process that stops, prevents, or reduces the frequency, rate, extent or direction of cell growth.",
    "Any process that modulates the frequency, rate or extent of cardiac muscle hypertrophy in response to stress.",
    "... omitted 10 items"
  ],
  "value_labels_en": [
    "regulation of apoptotic process",
    "pathway-restricted SMAD protein phosphorylation",
    "regulation of MAPK cascade",
    "heart trabecula formation",
    "SMAD protein signal transduction",
    "negative regulation of cardiac muscle hypertrophy",
    "adult heart development",
    "positive regulation of pathway-restricted SMAD protein phosphorylation",
    "positive regulation of sarcomere organization",
    "positive regulation of cell proliferation involved in heart morphogenesis",
    "cardiac muscle cell proliferation",
    "positive regulation of cardiac muscle hypertrophy",
    "negative regulation of endothelial cell migration",
    "positive regulation of transcription, DNA-templated",
    "ventricular cardiac muscle tissue morphogenesis",
    "multicellular organism development",
    "development of the heart",
    "positive regulation of gene expression",
    "cell adhesion",
    "negative regulation of cell migration",
    "positive regulation of cardiac muscle cell proliferation",
    "positive regulation of cartilage development",
    "negative regulation of cell growth",
    "regulation of cardiac muscle hypertrophy in response to stress",
    "... omitted 10 items"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 71,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q14819288"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q14633911",
      "Q14903145",
      "Q14859574",
      "Q14901698",
      "Q14858821",
      "Q14901689",
      "Q14865277",
      "Q14903088",
      "Q14903590",
      "Q14903591",
      "Q14889336",
      "Q14859611",
      "Q14903586",
      "Q14818066",
      "Q14852037",
      "Q14645705",
      "Q14819288",
      "Q14633893",
      "Q187640",
      "Q14859937",
      "Q14865337",
      "Q14903147",
      "Q14599698",
      "Q21171694",
      "... omitted 8 items"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q14633911",
    "Q14903145",
    "Q14859574",
    "Q14901698",
    "Q14858821",
    "Q14901689",
    "Q14865277",
    "Q14903088",
    "Q14903590",
    "Q14903591",
    "Q14889336",
    "Q14859611",
    "Q14903586",
    "Q14818066",
    "Q14852037",
    "Q14645705",
    "Q471817",
    "Q14633893",
    "Q187640",
    "Q14859937",
    "Q14865337",
    "Q14903147",
    "Q14599698",
    "Q21171694",
    "... omitted 8 items"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "is involved in the biological process",
    "label": "biological process"
  },
  "qid": {
    "description": "mammalian protein found in Homo sapiens",
    "label": "Bone morphogenetic protein 10"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
      "local_ids_count": 71,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q14819288"
        }
      ],
      "needed": 2,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q14633911",
        "Q14903145",
        "Q14859574",
        "Q14901698",
        "Q14858821",
        "Q14901689",
        "Q14865277",
        "Q14903088",
        "Q14903590",
        "Q14903591",
        "Q14889336",
        "Q14859611",
        "Q14903586",
        "Q14818066",
        "Q14852037",
        "Q14645705",
        "Q14819288",
        "Q14633893",
        "Q187640",
        "Q14859937",
        "Q14865337",
        "Q14903147",
        "Q14599698",
        "Q21171694",
        "... omitted 8 items"
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

## 026. `repair_Q5774674_2439465564`

| Field | Value |
|---|---|
| qid | Q5774674 |
| property | P131 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q5774674::P131 |
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
| truth_tokens_preview | ["Q1261", "Q16554", "Q13052701"] |
| classification_target_tokens | ["Q13140165", "Q13052701"] |
| classification_target_reason | mixed update classification uses the deterministic changed pair while ignoring retained values |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q13052701"
  ],
  "classification_target_reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
  "classification_target_role": "changed_pair",
  "classification_target_tokens": [
    "Q13140165",
    "Q13052701"
  ],
  "new_changed_value": "Q13052701",
  "old_changed_value": "Q13140165",
  "removed_target_tokens": [
    "Q13140165"
  ],
  "removed_unique_values": [
    "Q13140165"
  ],
  "retained_support_tokens": [
    "Q1261",
    "Q16554"
  ],
  "retained_unique_values": [
    "Q1261",
    "Q16554"
  ],
  "semantic_action": "MIXED_UPDATE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q1261",
    "Q16554",
    "Q13052701"
  ],
  "new_value_descriptions_en": [
    "state of the United States of America",
    "consolidated city-county and capital of Colorado, United States",
    "county in Colorado, United States, coterminous with the City of Denver"
  ],
  "new_value_labels_en": [
    "Colorado",
    "Denver",
    "Denver County"
  ],
  "old_value": [
    "Q1261",
    "Q16554",
    "Q13140165"
  ],
  "old_value_descriptions_en": [
    "state of the United States of America",
    "consolidated city-county and capital of Colorado, United States",
    "county in Colorado, United States, coterminous with the City of Denver"
  ],
  "old_value_labels_en": [
    "Colorado",
    "Denver",
    "Denver County"
  ],
  "revision_id": 2439465564,
  "value": [
    "Q1261",
    "Q16554",
    "Q13052701"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q13052701"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q1261": 1,
      "Q13052701": 1,
      "Q16554": 1
    },
    "new_unique": [
      "Q1261",
      "Q13052701",
      "Q16554"
    ],
    "new_values": [
      "Q1261",
      "Q16554",
      "Q13052701"
    ],
    "new_values_raw": [
      "Q1261",
      "Q16554",
      "Q13052701"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q1261": 1,
      "Q13140165": 1,
      "Q16554": 1
    },
    "old_unique": [
      "Q1261",
      "Q13140165",
      "Q16554"
    ],
    "old_values": [
      "Q1261",
      "Q16554",
      "Q13140165"
    ],
    "old_values_raw": [
      "Q1261",
      "Q16554",
      "Q13140165"
    ],
    "removed_unique_values": [
      "Q13140165"
    ],
    "retained_unique_values": [
      "Q1261",
      "Q16554"
    ],
    "semantic_action": "MIXED_UPDATE",
    "value_multiplicity_changes": {
      "Q13052701": {
        "new": 1,
        "old": 0
      },
      "Q13140165": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "state of the United States of America",
    "consolidated city-county and capital of Colorado, United States",
    "county in Colorado, United States, coterminous with the City of Denver"
  ],
  "value_labels_en": [
    "Colorado",
    "Denver",
    "Denver County"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-09T14:12:23",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
  "report_revision_new": 2440051055,
  "report_revision_old": 2439581737,
  "report_violation_type": "Target required claim P|17",
  "report_violation_type_normalized": "Target required claim P|17",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|17",
  "value": [
    "Q1261",
    "Q16554",
    "Q13140165"
  ],
  "value_descriptions_en": [
    "state of the United States of America",
    "consolidated city-county and capital of Colorado, United States",
    "county in Colorado, United States, coterminous with the City of Denver"
  ],
  "value_labels_en": [
    "Colorado",
    "Denver",
    "Denver County"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 29,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q13140165"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q1261",
      "Q16554",
      "Q13140165"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q1261",
    "Q16554",
    "Q13052701"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
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
    "description": "historical society",
    "label": "History Colorado"
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
      "local_ids_count": 29,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q13140165"
        }
      ],
      "needed": 2,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q1261",
        "Q16554",
        "Q13140165"
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

## 027. `repair_Q7628515_2444220154`

| Field | Value |
|---|---|
| qid | Q7628515 |
| property | P658 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q7628515::P658 |
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
| truth_tokens_preview | ["Q137465662", "Q137465696", "Q137465698", "Q137465701", "Q137465702", "...(+6)"] |
| classification_target_tokens | ["Q137465724"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q137465724"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q137465724"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "Q137465662",
    "Q137465696",
    "Q137465698",
    "Q137465701",
    "Q137465702",
    "Q137465703",
    "Q137465704",
    "Q137465706",
    "Q137465720",
    "Q137465723"
  ],
  "retained_unique_values": [
    "Q137465662",
    "Q137465696",
    "Q137465698",
    "Q137465701",
    "Q137465702",
    "Q137465703",
    "Q137465704",
    "Q137465706",
    "Q137465720",
    "Q137465723"
  ],
  "semantic_action": "ADD_SUPERSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Iamcarbon",
  "kind": "A_BOX",
  "new_value": [
    "Q137465662",
    "Q137465696",
    "Q137465698",
    "Q137465701",
    "Q137465702",
    "Q137465703",
    "Q137465704",
    "Q137465706",
    "Q137465720",
    "Q137465723",
    "Q137465724"
  ],
  "new_value_descriptions_en": [
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland"
  ],
  "new_value_labels_en": [
    "Numb",
    "Elmo",
    "Porno Mouth",
    "Water",
    "Coward",
    "The Box",
    "U.F.O.",
    "Mystery Song",
    "Just in Me",
    "Twisty Mirror",
    "I Won’t Stay"
  ],
  "old_value": [
    "Q137465662",
    "Q137465696",
    "Q137465698",
    "Q137465701",
    "Q137465702",
    "Q137465703",
    "Q137465704",
    "Q137465706",
    "Q137465720",
    "Q137465723"
  ],
  "old_value_descriptions_en": [
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland"
  ],
  "old_value_labels_en": [
    "Numb",
    "Elmo",
    "Porno Mouth",
    "Water",
    "Coward",
    "The Box",
    "U.F.O.",
    "Mystery Song",
    "Just in Me",
    "Twisty Mirror"
  ],
  "revision_id": 2444220154,
  "value": [
    "Q137465662",
    "Q137465696",
    "Q137465698",
    "Q137465701",
    "Q137465702",
    "Q137465703",
    "Q137465704",
    "Q137465706",
    "Q137465720",
    "Q137465723",
    "Q137465724"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q137465724"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q137465662": 1,
      "Q137465696": 1,
      "Q137465698": 1,
      "Q137465701": 1,
      "Q137465702": 1,
      "Q137465703": 1,
      "Q137465704": 1,
      "Q137465706": 1,
      "Q137465720": 1,
      "Q137465723": 1,
      "Q137465724": 1
    },
    "new_unique": [
      "Q137465662",
      "Q137465696",
      "Q137465698",
      "Q137465701",
      "Q137465702",
      "Q137465703",
      "Q137465704",
      "Q137465706",
      "Q137465720",
      "Q137465723",
      "Q137465724"
    ],
    "new_values": [
      "Q137465662",
      "Q137465696",
      "Q137465698",
      "Q137465701",
      "Q137465702",
      "Q137465703",
      "Q137465704",
      "Q137465706",
      "Q137465720",
      "Q137465723",
      "Q137465724"
    ],
    "new_values_raw": [
      "Q137465662",
      "Q137465696",
      "Q137465698",
      "Q137465701",
      "Q137465702",
      "Q137465703",
      "Q137465704",
      "Q137465706",
      "Q137465720",
      "Q137465723",
      "Q137465724"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q137465662": 1,
      "Q137465696": 1,
      "Q137465698": 1,
      "Q137465701": 1,
      "Q137465702": 1,
      "Q137465703": 1,
      "Q137465704": 1,
      "Q137465706": 1,
      "Q137465720": 1,
      "Q137465723": 1
    },
    "old_unique": [
      "Q137465662",
      "Q137465696",
      "Q137465698",
      "Q137465701",
      "Q137465702",
      "Q137465703",
      "Q137465704",
      "Q137465706",
      "Q137465720",
      "Q137465723"
    ],
    "old_values": [
      "Q137465662",
      "Q137465696",
      "Q137465698",
      "Q137465701",
      "Q137465702",
      "Q137465703",
      "Q137465704",
      "Q137465706",
      "Q137465720",
      "Q137465723"
    ],
    "old_values_raw": [
      "Q137465662",
      "Q137465696",
      "Q137465698",
      "Q137465701",
      "Q137465702",
      "Q137465703",
      "Q137465704",
      "Q137465706",
      "Q137465720",
      "Q137465723"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "Q137465662",
      "Q137465696",
      "Q137465698",
      "Q137465701",
      "Q137465702",
      "Q137465703",
      "Q137465704",
      "Q137465706",
      "Q137465720",
      "Q137465723"
    ],
    "semantic_action": "ADD_SUPERSET",
    "value_multiplicity_changes": {
      "Q137465724": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland"
  ],
  "value_labels_en": [
    "Numb",
    "Elmo",
    "Porno Mouth",
    "Water",
    "Coward",
    "The Box",
    "U.F.O.",
    "Mystery Song",
    "Just in Me",
    "Twisty Mirror",
    "I Won’t Stay"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-21T09:46:18",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P658",
  "report_revision_new": 2444874438,
  "report_revision_old": 2444445900,
  "report_violation_type": "Item P|2635",
  "report_violation_type_normalized": "Item P|2635",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|2635",
  "report_violation_types": [
    "Item P|2635",
    "Target required claim P|1476"
  ],
  "value": [
    "Q137465662",
    "Q137465696",
    "Q137465698",
    "Q137465701",
    "Q137465702",
    "Q137465703",
    "Q137465704",
    "Q137465706",
    "Q137465720",
    "Q137465723"
  ],
  "value_descriptions_en": [
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland",
    "vocal track by Holly McNarland"
  ],
  "value_labels_en": [
    "Numb",
    "Elmo",
    "Porno Mouth",
    "Water",
    "Coward",
    "The Box",
    "U.F.O.",
    "Mystery Song",
    "Just in Me",
    "Twisty Mirror"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 25,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q137465662",
      "Q137465696",
      "Q137465698",
      "Q137465701",
      "Q137465702",
      "Q137465703",
      "Q137465704",
      "Q137465706",
      "Q137465720",
      "Q137465723"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q137465662",
    "Q137465696",
    "Q137465698",
    "Q137465701",
    "Q137465702",
    "Q137465703",
    "Q137465704",
    "Q137465706",
    "Q137465720",
    "Q137465723",
    "Q137465724"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "audio tracks contained in this release",
    "label": "tracklist"
  },
  "qid": {
    "description": "1997 studio album by Holly McNarland",
    "label": "Stuff"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
    "label_en": "inverse constraint",
    "qid": "Q21510855"
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
        "Q137465662",
        "Q137465696",
        "Q137465698",
        "Q137465701",
        "Q137465702",
        "Q137465703",
        "Q137465704",
        "Q137465706",
        "Q137465720",
        "Q137465723"
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

## 028. `repair_Q85404_2254631024`

| Field | Value |
|---|---|
| qid | Q85404 |
| property | P166 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21510851 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q85404::P166 |
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
| truth_tokens_preview | ["Q18080431", "Q18080429", "Q18080427", "Q18080423", "Q51067", "...(+27)"] |
| classification_target_tokens | ["Q1971214"] |
| classification_target_reason | created or added values are the changed repair target |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q1971214"
  ],
  "classification_target_reason": "created or added values are the changed repair target",
  "classification_target_role": "added",
  "classification_target_tokens": [
    "Q1971214"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [],
  "removed_unique_values": [],
  "retained_support_tokens": [
    "Q10855212",
    "Q10855271",
    "Q1185606",
    "Q1319984",
    "Q14539884",
    "Q18080423",
    "Q18080427",
    "Q18080429",
    "Q18080431",
    "Q2268261",
    "Q29017281",
    "Q29017353",
    "Q30317051",
    "Q4146631",
    "Q4286770",
    "Q4287143",
    "Q4287194",
    "Q4336014",
    "Q4375455",
    "Q4375550",
    "Q47452387",
    "Q478850",
    "Q51067",
    "Q56305784",
    "... omitted 7 items"
  ],
  "retained_unique_values": [
    "Q10855212",
    "Q10855271",
    "Q1185606",
    "Q1319984",
    "Q14539884",
    "Q18080423",
    "Q18080427",
    "Q18080429",
    "Q18080431",
    "Q2268261",
    "Q29017281",
    "Q29017353",
    "Q30317051",
    "Q4146631",
    "Q4286770",
    "Q4287143",
    "Q4287194",
    "Q4336014",
    "Q4375455",
    "Q4375550",
    "Q47452387",
    "Q478850",
    "Q51067",
    "Q56305784",
    "... omitted 7 items"
  ],
  "semantic_action": "ADD_SUPERSET"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
  "author": "Infovarius",
  "kind": "A_BOX",
  "new_value": [
    "Q18080431",
    "Q18080429",
    "Q18080427",
    "Q18080423",
    "Q51067",
    "Q4146631",
    "Q478850",
    "Q612907",
    "Q29017281",
    "Q29017353",
    "Q2268261",
    "Q1185606",
    "Q7704960",
    "Q63965771",
    "Q47452387",
    "Q10855212",
    "Q10855271",
    "Q14539884",
    "Q1319984",
    "Q4286770",
    "Q4375550",
    "Q4336014",
    "Q97344253",
    "Q4375455",
    "... omitted 8 items"
  ],
  "new_value_descriptions_en": [
    "grade of an order",
    "grade of an order",
    "grade of an order",
    "grade of an order",
    "one of the most prestigious awards of the Soviet Union",
    "award in Russia (prize in science and technology, in literature and the arts, in humanitarian activities)",
    "order of the Soviet Union",
    "award of the Russian Federation",
    "третья степень украинского ордена «За заслуги»",
    "highest class of the Ukrainian Order of Merit",
    "state award of the Russian Federation, since 1994",
    "national scientific award in Russia",
    "mathematics award",
    "class of award",
    "степень украинского ордена князя Ярослава Мудрого",
    "third rank of the French Legion of Honour",
    "first rank of the French Legion of Honour",
    "decorazione di seconda classe dell'Ordine al Merito della Repubblica Italiana",
    "a service award conferred by the Government of Vietnam",
    "ведомственная медаль Министерства юстиции России",
    "honorary title in Sverdlovsk Oblast, Russia",
    "Award of the Russian Orthodox Church",
    null,
    "Russian honorary award",
    "... omitted 8 items"
  ],
  "new_value_labels_en": [
    "Order \"For Merit to the Fatherland\", 1st class",
    "Order \"For Merit to the Fatherland\", 2nd class",
    "Order \"For Merit to the Fatherland\", 3rd class",
    "Order \"For Merit to the Fatherland\", 4th class",
    "Lenin Prize",
    "State Prize of the Russian Federation",
    "Order of the Red Banner of Labour",
    "Order of Alexander Nevsky",
    "Order of Merit (Ukraine), 3rd class",
    "Order of Merit, 1st class",
    "Order of Honour",
    "Demidov Prize",
    "Leonard Euler Gold Medal",
    "Stolypin Medal, 1st class",
    "Order of Prince Yaroslav the Wise, 4th class",
    "Commander of the Legion of Honour",
    "Knight of the Legion of Honour",
    "Grand Officer of the Order of Merit of the Italian Republic",
    "Friendship Order",
    "медаль «В память 200-летия Минюста России»",
    "honorary citizen of Sverdlovsk Oblast",
    "Order of Holy Prince Daniel of Moscow",
    "Знак К. Э. Циолковского",
    "Russian Federation Government Certificate of Honour",
    "... omitted 8 items"
  ],
  "old_value": [
    "Q18080431",
    "Q18080429",
    "Q18080427",
    "Q18080423",
    "Q51067",
    "Q4146631",
    "Q478850",
    "Q612907",
    "Q29017281",
    "Q29017353",
    "Q2268261",
    "Q1185606",
    "Q7704960",
    "Q63965771",
    "Q47452387",
    "Q10855212",
    "Q10855271",
    "Q14539884",
    "Q1319984",
    "Q4286770",
    "Q4375550",
    "Q4336014",
    "Q97344253",
    "Q4375455",
    "... omitted 7 items"
  ],
  "old_value_descriptions_en": [
    "grade of an order",
    "grade of an order",
    "grade of an order",
    "grade of an order",
    "one of the most prestigious awards of the Soviet Union",
    "award in Russia (prize in science and technology, in literature and the arts, in humanitarian activities)",
    "order of the Soviet Union",
    "award of the Russian Federation",
    "третья степень украинского ордена «За заслуги»",
    "highest class of the Ukrainian Order of Merit",
    "state award of the Russian Federation, since 1994",
    "national scientific award in Russia",
    "mathematics award",
    "class of award",
    "степень украинского ордена князя Ярослава Мудрого",
    "third rank of the French Legion of Honour",
    "first rank of the French Legion of Honour",
    "decorazione di seconda classe dell'Ordine al Merito della Repubblica Italiana",
    "a service award conferred by the Government of Vietnam",
    "ведомственная медаль Министерства юстиции России",
    "honorary title in Sverdlovsk Oblast, Russia",
    "Award of the Russian Orthodox Church",
    null,
    "Russian honorary award",
    "... omitted 7 items"
  ],
  "old_value_labels_en": [
    "Order \"For Merit to the Fatherland\", 1st class",
    "Order \"For Merit to the Fatherland\", 2nd class",
    "Order \"For Merit to the Fatherland\", 3rd class",
    "Order \"For Merit to the Fatherland\", 4th class",
    "Lenin Prize",
    "State Prize of the Russian Federation",
    "Order of the Red Banner of Labour",
    "Order of Alexander Nevsky",
    "Order of Merit (Ukraine), 3rd class",
    "Order of Merit, 1st class",
    "Order of Honour",
    "Demidov Prize",
    "Leonard Euler Gold Medal",
    "Stolypin Medal, 1st class",
    "Order of Prince Yaroslav the Wise, 4th class",
    "Commander of the Legion of Honour",
    "Knight of the Legion of Honour",
    "Grand Officer of the Order of Merit of the Italian Republic",
    "Friendship Order",
    "медаль «В память 200-летия Минюста России»",
    "honorary citizen of Sverdlovsk Oblast",
    "Order of Holy Prince Daniel of Moscow",
    "Знак К. Э. Циолковского",
    "Russian Federation Government Certificate of Honour",
    "... omitted 7 items"
  ],
  "revision_id": 2254631024,
  "value": [
    "Q18080431",
    "Q18080429",
    "Q18080427",
    "Q18080423",
    "Q51067",
    "Q4146631",
    "Q478850",
    "Q612907",
    "Q29017281",
    "Q29017353",
    "Q2268261",
    "Q1185606",
    "Q7704960",
    "Q63965771",
    "Q47452387",
    "Q10855212",
    "Q10855271",
    "Q14539884",
    "Q1319984",
    "Q4286770",
    "Q4375550",
    "Q4336014",
    "Q97344253",
    "Q4375455",
    "... omitted 8 items"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q1971214"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q10855212": 1,
      "Q10855271": 1,
      "Q1185606": 1,
      "Q1319984": 1,
      "Q14539884": 1,
      "Q18080423": 1,
      "Q18080427": 1,
      "Q18080429": 1,
      "Q18080431": 1,
      "Q1971214": 1,
      "Q2268261": 1,
      "Q29017281": 1,
      "Q29017353": 1,
      "Q30317051": 1,
      "Q4146631": 1,
      "Q4286770": 1,
      "Q4287143": 1,
      "Q4287194": 1,
      "Q4336014": 1,
      "Q4375455": 1,
      "Q4375550": 1,
      "Q47452387": 1,
      "Q478850": 1,
      "Q51067": 1,
      "Q56305784": 1,
      "Q612907": 1,
      "Q63965771": 1,
      "Q7704960": 1,
      "Q8706404": 1,
      "Q93982": 1,
      "Q97344253": 1,
      "Q97344254": 1
    },
    "new_unique": [
      "Q10855212",
      "Q10855271",
      "Q1185606",
      "Q1319984",
      "Q14539884",
      "Q18080423",
      "Q18080427",
      "Q18080429",
      "Q18080431",
      "Q1971214",
      "Q2268261",
      "Q29017281",
      "Q29017353",
      "Q30317051",
      "Q4146631",
      "Q4286770",
      "Q4287143",
      "Q4287194",
      "Q4336014",
      "Q4375455",
      "Q4375550",
      "Q47452387",
      "Q478850",
      "Q51067",
      "... omitted 8 items"
    ],
    "new_values": [
      "Q18080431",
      "Q18080429",
      "Q18080427",
      "Q18080423",
      "Q51067",
      "Q4146631",
      "Q478850",
      "Q612907",
      "Q29017281",
      "Q29017353",
      "Q2268261",
      "Q1185606",
      "Q7704960",
      "Q63965771",
      "Q47452387",
      "Q10855212",
      "Q10855271",
      "Q14539884",
      "Q1319984",
      "Q4286770",
      "Q4375550",
      "Q4336014",
      "Q97344253",
      "Q4375455",
      "... omitted 8 items"
    ],
    "new_values_raw": [
      "Q18080431",
      "Q18080429",
      "Q18080427",
      "Q18080423",
      "Q51067",
      "Q4146631",
      "Q478850",
      "Q612907",
      "Q29017281",
      "Q29017353",
      "Q2268261",
      "Q1185606",
      "Q7704960",
      "Q63965771",
      "Q47452387",
      "Q10855212",
      "Q10855271",
      "Q14539884",
      "Q1319984",
      "Q4286770",
      "Q4375550",
      "Q4336014",
      "Q97344253",
      "Q4375455",
      "... omitted 8 items"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q10855212": 1,
      "Q10855271": 1,
      "Q1185606": 1,
      "Q1319984": 1,
      "Q14539884": 1,
      "Q18080423": 1,
      "Q18080427": 1,
      "Q18080429": 1,
      "Q18080431": 1,
      "Q2268261": 1,
      "Q29017281": 1,
      "Q29017353": 1,
      "Q30317051": 1,
      "Q4146631": 1,
      "Q4286770": 1,
      "Q4287143": 1,
      "Q4287194": 1,
      "Q4336014": 1,
      "Q4375455": 1,
      "Q4375550": 1,
      "Q47452387": 1,
      "Q478850": 1,
      "Q51067": 1,
      "Q56305784": 1,
      "Q612907": 1,
      "Q63965771": 1,
      "Q7704960": 1,
      "Q8706404": 1,
      "Q93982": 1,
      "Q97344253": 1,
      "Q97344254": 1
    },
    "old_unique": [
      "Q10855212",
      "Q10855271",
      "Q1185606",
      "Q1319984",
      "Q14539884",
      "Q18080423",
      "Q18080427",
      "Q18080429",
      "Q18080431",
      "Q2268261",
      "Q29017281",
      "Q29017353",
      "Q30317051",
      "Q4146631",
      "Q4286770",
      "Q4287143",
      "Q4287194",
      "Q4336014",
      "Q4375455",
      "Q4375550",
      "Q47452387",
      "Q478850",
      "Q51067",
      "Q56305784",
      "... omitted 7 items"
    ],
    "old_values": [
      "Q18080431",
      "Q18080429",
      "Q18080427",
      "Q18080423",
      "Q51067",
      "Q4146631",
      "Q478850",
      "Q612907",
      "Q29017281",
      "Q29017353",
      "Q2268261",
      "Q1185606",
      "Q7704960",
      "Q63965771",
      "Q47452387",
      "Q10855212",
      "Q10855271",
      "Q14539884",
      "Q1319984",
      "Q4286770",
      "Q4375550",
      "Q4336014",
      "Q97344253",
      "Q4375455",
      "... omitted 7 items"
    ],
    "old_values_raw": [
      "Q18080431",
      "Q18080429",
      "Q18080427",
      "Q18080423",
      "Q51067",
      "Q4146631",
      "Q478850",
      "Q612907",
      "Q29017281",
      "Q29017353",
      "Q2268261",
      "Q1185606",
      "Q7704960",
      "Q63965771",
      "Q47452387",
      "Q10855212",
      "Q10855271",
      "Q14539884",
      "Q1319984",
      "Q4286770",
      "Q4375550",
      "Q4336014",
      "Q97344253",
      "Q4375455",
      "... omitted 7 items"
    ],
    "removed_unique_values": [],
    "retained_unique_values": [
      "Q10855212",
      "Q10855271",
      "Q1185606",
      "Q1319984",
      "Q14539884",
      "Q18080423",
      "Q18080427",
      "Q18080429",
      "Q18080431",
      "Q2268261",
      "Q29017281",
      "Q29017353",
      "Q30317051",
      "Q4146631",
      "Q4286770",
      "Q4287143",
      "Q4287194",
      "Q4336014",
      "Q4375455",
      "Q4375550",
      "Q47452387",
      "Q478850",
      "Q51067",
      "Q56305784",
      "... omitted 7 items"
    ],
    "semantic_action": "ADD_SUPERSET",
    "value_multiplicity_changes": {
      "Q1971214": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "grade of an order",
    "grade of an order",
    "grade of an order",
    "grade of an order",
    "one of the most prestigious awards of the Soviet Union",
    "award in Russia (prize in science and technology, in literature and the arts, in humanitarian activities)",
    "order of the Soviet Union",
    "award of the Russian Federation",
    "третья степень украинского ордена «За заслуги»",
    "highest class of the Ukrainian Order of Merit",
    "state award of the Russian Federation, since 1994",
    "national scientific award in Russia",
    "mathematics award",
    "class of award",
    "степень украинского ордена князя Ярослава Мудрого",
    "third rank of the French Legion of Honour",
    "first rank of the French Legion of Honour",
    "decorazione di seconda classe dell'Ordine al Merito della Repubblica Italiana",
    "a service award conferred by the Government of Vietnam",
    "ведомственная медаль Министерства юстиции России",
    "honorary title in Sverdlovsk Oblast, Russia",
    "Award of the Russian Orthodox Church",
    null,
    "Russian honorary award",
    "... omitted 8 items"
  ],
  "value_labels_en": [
    "Order \"For Merit to the Fatherland\", 1st class",
    "Order \"For Merit to the Fatherland\", 2nd class",
    "Order \"For Merit to the Fatherland\", 3rd class",
    "Order \"For Merit to the Fatherland\", 4th class",
    "Lenin Prize",
    "State Prize of the Russian Federation",
    "Order of the Red Banner of Labour",
    "Order of Alexander Nevsky",
    "Order of Merit (Ukraine), 3rd class",
    "Order of Merit, 1st class",
    "Order of Honour",
    "Demidov Prize",
    "Leonard Euler Gold Medal",
    "Stolypin Medal, 1st class",
    "Order of Prince Yaroslav the Wise, 4th class",
    "Commander of the Legion of Honour",
    "Knight of the Legion of Honour",
    "Grand Officer of the Order of Merit of the Italian Republic",
    "Friendship Order",
    "медаль «В память 200-летия Минюста России»",
    "honorary citizen of Sverdlovsk Oblast",
    "Order of Holy Prince Daniel of Moscow",
    "Знак К. Э. Циолковского",
    "Russian Federation Government Certificate of Honour",
    "... omitted 8 items"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-10-01T12:00:28",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P166",
  "report_revision_new": 2255073635,
  "report_revision_old": 2254683010,
  "report_violation_type": "Q|5",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo"
  ],
  "report_violation_type_labels_en": [
    "human"
  ],
  "report_violation_type_normalized": "Q|5",
  "report_violation_type_qids": [
    "Q5"
  ],
  "report_violation_type_raw": "Q|5",
  "value": [
    "Q18080431",
    "Q18080429",
    "Q18080427",
    "Q18080423",
    "Q51067",
    "Q4146631",
    "Q478850",
    "Q612907",
    "Q29017281",
    "Q29017353",
    "Q2268261",
    "Q1185606",
    "Q7704960",
    "Q63965771",
    "Q47452387",
    "Q10855212",
    "Q10855271",
    "Q14539884",
    "Q1319984",
    "Q4286770",
    "Q4375550",
    "Q4336014",
    "Q97344253",
    "Q4375455",
    "... omitted 7 items"
  ],
  "value_descriptions_en": [
    "grade of an order",
    "grade of an order",
    "grade of an order",
    "grade of an order",
    "one of the most prestigious awards of the Soviet Union",
    "award in Russia (prize in science and technology, in literature and the arts, in humanitarian activities)",
    "order of the Soviet Union",
    "award of the Russian Federation",
    "третья степень украинского ордена «За заслуги»",
    "highest class of the Ukrainian Order of Merit",
    "state award of the Russian Federation, since 1994",
    "national scientific award in Russia",
    "mathematics award",
    "class of award",
    "степень украинского ордена князя Ярослава Мудрого",
    "third rank of the French Legion of Honour",
    "first rank of the French Legion of Honour",
    "decorazione di seconda classe dell'Ordine al Merito della Repubblica Italiana",
    "a service award conferred by the Government of Vietnam",
    "ведомственная медаль Министерства юстиции России",
    "honorary title in Sverdlovsk Oblast, Russia",
    "Award of the Russian Orthodox Church",
    null,
    "Russian honorary award",
    "... omitted 7 items"
  ],
  "value_labels_en": [
    "Order \"For Merit to the Fatherland\", 1st class",
    "Order \"For Merit to the Fatherland\", 2nd class",
    "Order \"For Merit to the Fatherland\", 3rd class",
    "Order \"For Merit to the Fatherland\", 4th class",
    "Lenin Prize",
    "State Prize of the Russian Federation",
    "Order of the Red Banner of Labour",
    "Order of Alexander Nevsky",
    "Order of Merit (Ukraine), 3rd class",
    "Order of Merit, 1st class",
    "Order of Honour",
    "Demidov Prize",
    "Leonard Euler Gold Medal",
    "Stolypin Medal, 1st class",
    "Order of Prince Yaroslav the Wise, 4th class",
    "Commander of the Legion of Honour",
    "Knight of the Legion of Honour",
    "Grand Officer of the Order of Merit of the Italian Republic",
    "Friendship Order",
    "медаль «В память 200-летия Минюста России»",
    "honorary citizen of Sverdlovsk Oblast",
    "Order of Holy Prince Daniel of Moscow",
    "Знак К. Э. Циолковского",
    "Russian Federation Government Certificate of Honour",
    "... omitted 7 items"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 104,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q18080431",
      "Q18080429",
      "Q18080427",
      "Q18080423",
      "Q51067",
      "Q4146631",
      "Q478850",
      "Q612907",
      "Q29017281",
      "Q29017353",
      "Q2268261",
      "Q1185606",
      "Q7704960",
      "Q63965771",
      "Q47452387",
      "Q10855212",
      "Q10855271",
      "Q14539884",
      "Q1319984",
      "Q4286770",
      "Q4375550",
      "Q4336014",
      "Q97344253",
      "Q4375455",
      "... omitted 7 items"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q18080431",
    "Q18080429",
    "Q18080427",
    "Q18080423",
    "Q51067",
    "Q4146631",
    "Q478850",
    "Q612907",
    "Q29017281",
    "Q29017353",
    "Q2268261",
    "Q1185606",
    "Q7704960",
    "Q63965771",
    "Q47452387",
    "Q10855212",
    "Q10855271",
    "Q14539884",
    "Q1319984",
    "Q4286770",
    "Q4375550",
    "Q4336014",
    "Q97344253",
    "Q4375455",
    "... omitted 8 items"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "award or recognition received by a person, organization or creative work",
    "label": "award received"
  },
  "qid": {
    "description": "Russian mathematician (born 1936)",
    "label": "Yury Osipov"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
      "local_ids_count": 104,
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
        "Q18080431",
        "Q18080429",
        "Q18080427",
        "Q18080423",
        "Q51067",
        "Q4146631",
        "Q478850",
        "Q612907",
        "Q29017281",
        "Q29017353",
        "Q2268261",
        "Q1185606",
        "Q7704960",
        "Q63965771",
        "Q47452387",
        "Q10855212",
        "Q10855271",
        "Q14539884",
        "Q1319984",
        "Q4286770",
        "Q4375550",
        "Q4336014",
        "Q97344253",
        "Q4375455",
        "... omitted 7 items"
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

## 029. `repair_Q95967818_2441306184`

| Field | Value |
|---|---|
| qid | Q95967818 |
| property | P735 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q95967818::P735 |
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
| truth_tokens_preview | ["Q3061675"] |
| classification_target_tokens | ["Q3061675"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q3061675"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Q3061675"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q15731812"
  ],
  "removed_unique_values": [
    "Q15731812"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q3061675"
  ],
  "new_value_descriptions_en": [
    "female given name"
  ],
  "new_value_labels_en": [
    "Ewa"
  ],
  "old_value": [
    "Q15731812"
  ],
  "old_value_descriptions_en": [
    "female given name"
  ],
  "old_value_labels_en": [
    "Ewa"
  ],
  "revision_id": 2441306184,
  "value": [
    "Q3061675"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q3061675"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q3061675": 1
    },
    "new_unique": [
      "Q3061675"
    ],
    "new_values": [
      "Q3061675"
    ],
    "new_values_raw": [
      "Q3061675"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q15731812": 1
    },
    "old_unique": [
      "Q15731812"
    ],
    "old_values": [
      "Q15731812"
    ],
    "old_values_raw": [
      "Q15731812"
    ],
    "removed_unique_values": [
      "Q15731812"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q15731812": {
        "new": 0,
        "old": 1
      },
      "Q3061675": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "female given name"
  ],
  "value_labels_en": [
    "Ewa"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T11:09:13",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P735",
  "report_revision_new": 2442267914,
  "report_revision_old": 2441753351,
  "report_violation_type": "Target required claim P|1705",
  "report_violation_type_normalized": "Target required claim P|1705",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|1705",
  "report_violation_types": [
    "Target required claim P|1705",
    "Value type Q|202444, Q|49614, Q|122067883"
  ],
  "value": [
    "Q15731812"
  ],
  "value_descriptions_en": [
    "female given name"
  ],
  "value_labels_en": [
    "Ewa"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 8,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q15731812"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q3061675"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "first name or another given name of this person; values used with the property should not link disambiguations nor family names",
    "label": "given name"
  },
  "qid": {
    "description": "researcher",
    "label": "Ewa Wnuk"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
      "local_ids_count": 8,
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
        "Q15731812"
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

## 030. `repair_Q97689673_2441306589`

| Field | Value |
|---|---|
| qid | Q97689673 |
| property | P735 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q97689673::P735 |
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
| truth_tokens_preview | ["Q20087949", "Q3061675"] |
| classification_target_tokens | ["Q15731812", "Q3061675"] |
| classification_target_reason | mixed update classification uses the deterministic changed pair while ignoring retained values |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q3061675"
  ],
  "classification_target_reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
  "classification_target_role": "changed_pair",
  "classification_target_tokens": [
    "Q15731812",
    "Q3061675"
  ],
  "new_changed_value": "Q3061675",
  "old_changed_value": "Q15731812",
  "removed_target_tokens": [
    "Q15731812"
  ],
  "removed_unique_values": [
    "Q15731812"
  ],
  "retained_support_tokens": [
    "Q20087949"
  ],
  "retained_unique_values": [
    "Q20087949"
  ],
  "semantic_action": "MIXED_UPDATE"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q20087949",
    "Q3061675"
  ],
  "new_value_descriptions_en": [
    "female given name",
    "female given name"
  ],
  "new_value_labels_en": [
    "Dagmara",
    "Ewa"
  ],
  "old_value": [
    "Q20087949",
    "Q15731812"
  ],
  "old_value_descriptions_en": [
    "female given name",
    "female given name"
  ],
  "old_value_labels_en": [
    "Dagmara",
    "Ewa"
  ],
  "revision_id": 2441306589,
  "value": [
    "Q20087949",
    "Q3061675"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q3061675"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q20087949": 1,
      "Q3061675": 1
    },
    "new_unique": [
      "Q20087949",
      "Q3061675"
    ],
    "new_values": [
      "Q20087949",
      "Q3061675"
    ],
    "new_values_raw": [
      "Q20087949",
      "Q3061675"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q15731812": 1,
      "Q20087949": 1
    },
    "old_unique": [
      "Q15731812",
      "Q20087949"
    ],
    "old_values": [
      "Q20087949",
      "Q15731812"
    ],
    "old_values_raw": [
      "Q20087949",
      "Q15731812"
    ],
    "removed_unique_values": [
      "Q15731812"
    ],
    "retained_unique_values": [
      "Q20087949"
    ],
    "semantic_action": "MIXED_UPDATE",
    "value_multiplicity_changes": {
      "Q15731812": {
        "new": 0,
        "old": 1
      },
      "Q3061675": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "female given name",
    "female given name"
  ],
  "value_labels_en": [
    "Dagmara",
    "Ewa"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T11:09:13",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P735",
  "report_revision_new": 2442267914,
  "report_revision_old": 2441753351,
  "report_violation_type": "Target required claim P|1705",
  "report_violation_type_normalized": "Target required claim P|1705",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|1705",
  "report_violation_types": [
    "Target required claim P|1705",
    "Value type Q|202444, Q|49614, Q|122067883"
  ],
  "value": [
    "Q20087949",
    "Q15731812"
  ],
  "value_descriptions_en": [
    "female given name",
    "female given name"
  ],
  "value_labels_en": [
    "Dagmara",
    "Ewa"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 11,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q15731812"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q20087949",
      "Q15731812"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q20087949",
    "Q3061675"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "first name or another given name of this person; values used with the property should not link disambiguations nor family names",
    "label": "given name"
  },
  "qid": {
    "description": "researcher",
    "label": "Dagmara Wasiuk-Zowada"
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
    "label_en": "value-requires-statement constraint",
    "qid": "Q21510864"
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
      "local_ids_count": 11,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q15731812"
        }
      ],
      "needed": 2,
      "sources_used": [
        "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
      ],
      "used_literal_substring": false
    },
    "result": false,
    "step": "local_availability",
    "synthetic": {
      "pre_repair_source": "repair_target.old_value",
      "tokens": [
        "Q20087949",
        "Q15731812"
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

## 031. `repair_Q99870964_2444734107`

| Field | Value |
|---|---|
| qid | Q99870964 |
| property | P749 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| classification_rule_family | negative_rule_and_local_scan |
| classification_rule_subfamily | external_by_elimination |
| decision_constraint_type |   |
| group_key | ABOX::Q99870964::P749 |
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
| truth_tokens_preview | ["Q27346385"] |
| classification_target_tokens | ["Q27346385"] |
| classification_target_reason | one-to-one replacement is classified from the replacement relation |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

#### Delta Summary

```json
{
  "added_unique_values": [
    "Q27346385"
  ],
  "classification_target_reason": "one-to-one replacement is classified from the replacement relation",
  "classification_target_role": "replacement_new",
  "classification_target_tokens": [
    "Q27346385"
  ],
  "new_changed_value": null,
  "old_changed_value": null,
  "removed_target_tokens": [
    "Q99872290"
  ],
  "removed_unique_values": [
    "Q99872290"
  ],
  "retained_support_tokens": [],
  "retained_unique_values": [],
  "semantic_action": "REPLACE_1_TO_1"
}
```

#### Classifier Rule Metadata

```json
{
  "classification_rule_family": "negative_rule_and_local_scan",
  "classification_rule_subfamily": "external_by_elimination",
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
    "Q27346385"
  ],
  "new_value_descriptions_en": [
    "Cayman Islands holding company"
  ],
  "new_value_labels_en": [
    "Manchester United plc"
  ],
  "old_value": [
    "Q99872290"
  ],
  "old_value_descriptions_en": [
    "Cayman Islands holding company"
  ],
  "old_value_labels_en": [
    "Manchester United plc"
  ],
  "revision_id": 2444734107,
  "value": [
    "Q27346385"
  ],
  "value_change_summary": {
    "added_unique_values": [
      "Q27346385"
    ],
    "exact_value_lists_unchanged": false,
    "new_counts": {
      "Q27346385": 1
    },
    "new_unique": [
      "Q27346385"
    ],
    "new_values": [
      "Q27346385"
    ],
    "new_values_raw": [
      "Q27346385"
    ],
    "normalized_unique_values_unchanged": false,
    "old_counts": {
      "Q99872290": 1
    },
    "old_unique": [
      "Q99872290"
    ],
    "old_values": [
      "Q99872290"
    ],
    "old_values_raw": [
      "Q99872290"
    ],
    "removed_unique_values": [
      "Q99872290"
    ],
    "retained_unique_values": [],
    "semantic_action": "REPLACE_1_TO_1",
    "value_multiplicity_changes": {
      "Q27346385": {
        "new": 1,
        "old": 0
      },
      "Q99872290": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "Cayman Islands holding company"
  ],
  "value_labels_en": [
    "Manchester United plc"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-22T09:25:01",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P749",
  "report_revision_new": 2445435850,
  "report_revision_old": 2444872115,
  "report_violation_type": "Value type Q|43229, Q|14623646, Q|170584, Q|1530022, Q|16519632, Q|70363582, Q|10387680, Q|895526",
  "report_violation_type_descriptions_en": [
    "social entity established to meet needs or pursue goals",
    "organization which only appears in works of fiction",
    "collaborative enterprise, frequently involving research or design, that is carefully planned to achieve a particular aim",
    "organization that supports the practice of a religion",
    "group organized for the purpose of scientific research and development",
    "organizations representing specialized fields and accepted as authoritative",
    "частка ўстановы",
    "a designated body with authority"
  ],
  "report_violation_type_labels_en": [
    "organization",
    "fictional organization",
    "project",
    "religious organization",
    "scientific organization",
    "academies and institutes",
    "organization unit",
    "governing body"
  ],
  "report_violation_type_normalized": "Value type Q|43229, Q|14623646, Q|170584, Q|1530022, Q|16519632, Q|70363582, Q|10387680, Q|895526",
  "report_violation_type_qids": [
    "Q43229",
    "Q14623646",
    "Q170584",
    "Q1530022",
    "Q16519632",
    "Q70363582",
    "Q10387680",
    "Q895526"
  ],
  "report_violation_type_raw": "Value type Q|43229, Q|14623646, Q|170584, Q|1530022, Q|16519632, Q|70363582, Q|10387680, Q|895526",
  "value": [
    "Q99872290"
  ],
  "value_descriptions_en": [
    "Cayman Islands holding company"
  ],
  "value_labels_en": [
    "Manchester United plc"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 14,
  "local_support_for_retained_value": [],
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q99872290"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q27346385"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "parent organization or unit of an organization or unit, opposite of child organization or unit (P355); use instance of (P31) to distinguish organization (Q43229) and organization unit (Q10387680)",
    "label": "parent organization or unit"
  },
  "qid": {
    "description": "English sport company",
    "label": "Manchester United Limited"
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
    "label_en": "inverse constraint",
    "qid": "Q21510855"
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
        "Q99872290"
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
