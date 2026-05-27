# TypeC_EXTERNAL_BY_ELIMINATION_QID_TRUTH

Cases: 50

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `repair_Q105401946_2441307367`

| Field | Value |
|---|---|
| qid | Q105401946 |
| property | P735 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| group_key | ABOX::Q105401946::P735 |
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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
  "revision_id": 2441307367,
  "value": [
    "Q3061675"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q3061675"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q3061675"
    ],
    "new_value": [
      "Q3061675"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q15731812"
    ],
    "old_value": [
      "Q15731812"
    ],
    "removed_unique_values": [
      "Q15731812"
    ],
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
  "local_ids_count": 10,
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
    "description": "jewelry designer",
    "label": "Ewa Skrzyńska"
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

## 002. `repair_Q106359079_2443408133`

| Field | Value |
|---|---|
| qid | Q106359079 |
| property | P50 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| group_key | ABOX::Q106359079::P50 |
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
| truth_tokens_preview | ["Q28828105", "Q58476347", "Q57948604", "Q29306140", "Q87007111", "...(+79)"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Simon Villeneuve",
  "kind": "A_BOX",
  "new_value": [
    "Q28828105",
    "Q58476347",
    "Q57948604",
    "Q29306140",
    "Q87007111",
    "Q1676381",
    "Q116790553",
    "Q53970604",
    "Q114339533",
    "Q71233967",
    "Q87189799",
    "Q58926058",
    "Q58375926",
    "Q58479882",
    "Q59673601",
    "Q53971672",
    "Q58481281",
    "Q97651764",
    "Q45319488",
    "Q56562330",
    "Q1507296",
    "Q61284527",
    "Q11974059",
    "Q58923634",
    "... omitted 60 items"
  ],
  "new_value_descriptions_en": [
    "Algerian cosmologist",
    "researcher (ORCID 0000-0002-2407-7956)",
    "researcher, ORCID id # 0000-0003-4481-3559",
    "cosmologist (born 1969)",
    "researcher",
    "Canadian astrophysicist and cosmologist",
    "Ecole Normale Superieure; France",
    "Italian astronomer (1962-)",
    "Italian British observational cosmologist",
    "French physicist",
    "university professor and physicist",
    "researcher ORCID ID = 0000-0003-4572-7732",
    "researcher, ORCID id # 0000-0001-6487-1866",
    "researcher",
    "researcher ORCID ID = 0000-0002-4650-8518",
    "researcher",
    "researcher ORCID ID = 0000-0003-2868-2595",
    "French astrophysicist",
    "astronomer",
    "researcher",
    "British astronomer (born 1955)",
    "researcher ORCID ID = 0000-0002-1838-7288",
    "Norwegian physicist",
    "researcher ORCID ID = 0000-0001-6185-7903",
    "... omitted 60 items"
  ],
  "new_value_labels_en": [
    "Nabila Aghanim",
    "Yashar Akrami",
    "Mario Ballardini",
    "Richard Battye",
    "Pawel Bielewicz",
    "J. Richard Bond",
    "Francois Boulanger",
    "Carlo Burigana",
    "Erminia Calabrese",
    "Jean-François Cardoso",
    "H. Cynthia Chiang",
    "Loris Colombo",
    "Celine Combet",
    "Dagoberto Contreras",
    "Brendan P. Crill",
    "Francesco Cuttaia",
    "Gianfranco de Zotti",
    "Jacques Delabrouille",
    "Olivier Doré",
    "Marian Douspis",
    "George Efstathiou",
    "Franz Elsner",
    "Hans Kristian K. Eriksen",
    "Raul Fernandez-Cobos",
    "... omitted 60 items"
  ],
  "old_value": [
    "Q28828105",
    "Q58476347",
    "Q57948604",
    "Q29306140",
    "Q87007111",
    "Q1676381",
    "Q116790553",
    "Q53970604",
    "Q114339533",
    "Q71233967",
    "Q87189799",
    "Q58926058",
    "Q58375926",
    "Q58479882",
    "Q59673601",
    "Q53971672",
    "Q58481281",
    "Q97651764",
    "Q45319488",
    "Q56562330",
    "Q1507296",
    "Q61284527",
    "Q11974059",
    "Q58923634",
    "... omitted 59 items"
  ],
  "old_value_descriptions_en": [
    "Algerian cosmologist",
    "researcher (ORCID 0000-0002-2407-7956)",
    "researcher, ORCID id # 0000-0003-4481-3559",
    "cosmologist (born 1969)",
    "researcher",
    "Canadian astrophysicist and cosmologist",
    "Ecole Normale Superieure; France",
    "Italian astronomer (1962-)",
    "Italian British observational cosmologist",
    "French physicist",
    "university professor and physicist",
    "researcher ORCID ID = 0000-0003-4572-7732",
    "researcher, ORCID id # 0000-0001-6487-1866",
    "researcher",
    "researcher ORCID ID = 0000-0002-4650-8518",
    "researcher",
    "researcher ORCID ID = 0000-0003-2868-2595",
    "French astrophysicist",
    "astronomer",
    "researcher",
    "British astronomer (born 1955)",
    "researcher ORCID ID = 0000-0002-1838-7288",
    "Norwegian physicist",
    "researcher ORCID ID = 0000-0001-6185-7903",
    "... omitted 59 items"
  ],
  "old_value_labels_en": [
    "Nabila Aghanim",
    "Yashar Akrami",
    "Mario Ballardini",
    "Richard Battye",
    "Pawel Bielewicz",
    "J. Richard Bond",
    "Francois Boulanger",
    "Carlo Burigana",
    "Erminia Calabrese",
    "Jean-François Cardoso",
    "H. Cynthia Chiang",
    "Loris Colombo",
    "Celine Combet",
    "Dagoberto Contreras",
    "Brendan P. Crill",
    "Francesco Cuttaia",
    "Gianfranco de Zotti",
    "Jacques Delabrouille",
    "Olivier Doré",
    "Marian Douspis",
    "George Efstathiou",
    "Franz Elsner",
    "Hans Kristian K. Eriksen",
    "Raul Fernandez-Cobos",
    "... omitted 59 items"
  ],
  "revision_id": 2443408133,
  "value": [
    "Q28828105",
    "Q58476347",
    "Q57948604",
    "Q29306140",
    "Q87007111",
    "Q1676381",
    "Q116790553",
    "Q53970604",
    "Q114339533",
    "Q71233967",
    "Q87189799",
    "Q58926058",
    "Q58375926",
    "Q58479882",
    "Q59673601",
    "Q53971672",
    "Q58481281",
    "Q97651764",
    "Q45319488",
    "Q56562330",
    "Q1507296",
    "Q61284527",
    "Q11974059",
    "Q58923634",
    "... omitted 60 items"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q133482954"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 84,
    "new_unique": [
      "Q106359001",
      "Q106359037",
      "Q106359090",
      "Q113657557",
      "Q114339533",
      "Q116790553",
      "Q11974059",
      "Q133482954",
      "Q135261398",
      "Q1507296",
      "Q1676381",
      "Q28828105",
      "Q29306140",
      "Q3372196",
      "Q45319488",
      "Q53442393",
      "Q53953423",
      "Q53969251",
      "Q53969295",
      "Q53969376",
      "Q53969416",
      "Q53969553",
      "Q53969677",
      "Q53969952",
      "... omitted 60 items"
    ],
    "new_value": [
      "Q28828105",
      "Q58476347",
      "Q57948604",
      "Q29306140",
      "Q87007111",
      "Q1676381",
      "Q116790553",
      "Q53970604",
      "Q114339533",
      "Q71233967",
      "Q87189799",
      "Q58926058",
      "Q58375926",
      "Q58479882",
      "Q59673601",
      "Q53971672",
      "Q58481281",
      "Q97651764",
      "Q45319488",
      "Q56562330",
      "Q1507296",
      "Q61284527",
      "Q11974059",
      "Q58923634",
      "... omitted 60 items"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 83,
    "old_unique": [
      "Q106359001",
      "Q106359037",
      "Q106359090",
      "Q113657557",
      "Q114339533",
      "Q116790553",
      "Q11974059",
      "Q135261398",
      "Q1507296",
      "Q1676381",
      "Q28828105",
      "Q29306140",
      "Q3372196",
      "Q45319488",
      "Q53442393",
      "Q53953423",
      "Q53969251",
      "Q53969295",
      "Q53969376",
      "Q53969416",
      "Q53969553",
      "Q53969677",
      "Q53969952",
      "Q53969992",
      "... omitted 59 items"
    ],
    "old_value": [
      "Q28828105",
      "Q58476347",
      "Q57948604",
      "Q29306140",
      "Q87007111",
      "Q1676381",
      "Q116790553",
      "Q53970604",
      "Q114339533",
      "Q71233967",
      "Q87189799",
      "Q58926058",
      "Q58375926",
      "Q58479882",
      "Q59673601",
      "Q53971672",
      "Q58481281",
      "Q97651764",
      "Q45319488",
      "Q56562330",
      "Q1507296",
      "Q61284527",
      "Q11974059",
      "Q58923634",
      "... omitted 59 items"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "Q133482954": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "Algerian cosmologist",
    "researcher (ORCID 0000-0002-2407-7956)",
    "researcher, ORCID id # 0000-0003-4481-3559",
    "cosmologist (born 1969)",
    "researcher",
    "Canadian astrophysicist and cosmologist",
    "Ecole Normale Superieure; France",
    "Italian astronomer (1962-)",
    "Italian British observational cosmologist",
    "French physicist",
    "university professor and physicist",
    "researcher ORCID ID = 0000-0003-4572-7732",
    "researcher, ORCID id # 0000-0001-6487-1866",
    "researcher",
    "researcher ORCID ID = 0000-0002-4650-8518",
    "researcher",
    "researcher ORCID ID = 0000-0003-2868-2595",
    "French astrophysicist",
    "astronomer",
    "researcher",
    "British astronomer (born 1955)",
    "researcher ORCID ID = 0000-0002-1838-7288",
    "Norwegian physicist",
    "researcher ORCID ID = 0000-0001-6185-7903",
    "... omitted 60 items"
  ],
  "value_labels_en": [
    "Nabila Aghanim",
    "Yashar Akrami",
    "Mario Ballardini",
    "Richard Battye",
    "Pawel Bielewicz",
    "J. Richard Bond",
    "Francois Boulanger",
    "Carlo Burigana",
    "Erminia Calabrese",
    "Jean-François Cardoso",
    "H. Cynthia Chiang",
    "Loris Colombo",
    "Celine Combet",
    "Dagoberto Contreras",
    "Brendan P. Crill",
    "Francesco Cuttaia",
    "Gianfranco de Zotti",
    "Jacques Delabrouille",
    "Olivier Doré",
    "Marian Douspis",
    "George Efstathiou",
    "Franz Elsner",
    "Hans Kristian K. Eriksen",
    "Raul Fernandez-Cobos",
    "... omitted 60 items"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-21T14:05:39",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P50",
  "report_revision_new": 2444961705,
  "report_revision_old": 2444519544,
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
    "Q28828105",
    "Q58476347",
    "Q57948604",
    "Q29306140",
    "Q87007111",
    "Q1676381",
    "Q116790553",
    "Q53970604",
    "Q114339533",
    "Q71233967",
    "Q87189799",
    "Q58926058",
    "Q58375926",
    "Q58479882",
    "Q59673601",
    "Q53971672",
    "Q58481281",
    "Q97651764",
    "Q45319488",
    "Q56562330",
    "Q1507296",
    "Q61284527",
    "Q11974059",
    "Q58923634",
    "... omitted 59 items"
  ],
  "value_descriptions_en": [
    "Algerian cosmologist",
    "researcher (ORCID 0000-0002-2407-7956)",
    "researcher, ORCID id # 0000-0003-4481-3559",
    "cosmologist (born 1969)",
    "researcher",
    "Canadian astrophysicist and cosmologist",
    "Ecole Normale Superieure; France",
    "Italian astronomer (1962-)",
    "Italian British observational cosmologist",
    "French physicist",
    "university professor and physicist",
    "researcher ORCID ID = 0000-0003-4572-7732",
    "researcher, ORCID id # 0000-0001-6487-1866",
    "researcher",
    "researcher ORCID ID = 0000-0002-4650-8518",
    "researcher",
    "researcher ORCID ID = 0000-0003-2868-2595",
    "French astrophysicist",
    "astronomer",
    "researcher",
    "British astronomer (born 1955)",
    "researcher ORCID ID = 0000-0002-1838-7288",
    "Norwegian physicist",
    "researcher ORCID ID = 0000-0001-6185-7903",
    "... omitted 59 items"
  ],
  "value_labels_en": [
    "Nabila Aghanim",
    "Yashar Akrami",
    "Mario Ballardini",
    "Richard Battye",
    "Pawel Bielewicz",
    "J. Richard Bond",
    "Francois Boulanger",
    "Carlo Burigana",
    "Erminia Calabrese",
    "Jean-François Cardoso",
    "H. Cynthia Chiang",
    "Loris Colombo",
    "Celine Combet",
    "Dagoberto Contreras",
    "Brendan P. Crill",
    "Francesco Cuttaia",
    "Gianfranco de Zotti",
    "Jacques Delabrouille",
    "Olivier Doré",
    "Marian Douspis",
    "George Efstathiou",
    "Franz Elsner",
    "Hans Kristian K. Eriksen",
    "Raul Fernandez-Cobos",
    "... omitted 59 items"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 92,
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q28828105",
      "Q58476347",
      "Q57948604",
      "Q29306140",
      "Q87007111",
      "Q1676381",
      "Q116790553",
      "Q53970604",
      "Q114339533",
      "Q71233967",
      "Q87189799",
      "Q58926058",
      "Q58375926",
      "Q58479882",
      "Q59673601",
      "Q53971672",
      "Q58481281",
      "Q97651764",
      "Q45319488",
      "Q56562330",
      "Q1507296",
      "Q61284527",
      "Q11974059",
      "Q58923634",
      "... omitted 59 items"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q28828105",
    "Q58476347",
    "Q57948604",
    "Q29306140",
    "Q87007111",
    "Q1676381",
    "Q116790553",
    "Q53970604",
    "Q114339533",
    "Q71233967",
    "Q87189799",
    "Q58926058",
    "Q58375926",
    "Q58479882",
    "Q59673601",
    "Q53971672",
    "Q58481281",
    "Q97651764",
    "Q45319488",
    "Q56562330",
    "Q1507296",
    "Q61284527",
    "Q11974059",
    "Q58923634",
    "... omitted 60 items"
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
    "description": "scholarly article",
    "label": "Planck 2018 results: VI. Cosmological parameters"
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
      "local_ids_count": 92,
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
        "Q28828105",
        "Q58476347",
        "Q57948604",
        "Q29306140",
        "Q87007111",
        "Q1676381",
        "Q116790553",
        "Q53970604",
        "Q114339533",
        "Q71233967",
        "Q87189799",
        "Q58926058",
        "Q58375926",
        "Q58479882",
        "Q59673601",
        "Q53971672",
        "Q58481281",
        "Q97651764",
        "Q45319488",
        "Q56562330",
        "Q1507296",
        "Q61284527",
        "Q11974059",
        "Q58923634",
        "... omitted 59 items"
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

## 003. `repair_Q108308406_2447220088`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q26972386"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "Q26972386",
      "Q57413151"
    ],
    "new_value": [
      "Q57413151",
      "Q26972386"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "Q57413151",
      "Q57902495"
    ],
    "old_value": [
      "Q57413151",
      "Q57902495"
    ],
    "removed_unique_values": [
      "Q57902495"
    ],
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

## 004. `repair_Q11084_2443681378`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "CREATE",
    "added_unique_values": [
      "Q109563825"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q109563825"
    ],
    "new_value": [
      "Q109563825"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
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

## 005. `repair_Q11724997_2445332870`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q11181530"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q11181530"
    ],
    "new_value": [
      "Q11181530"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q102117969"
    ],
    "old_value": [
      "Q102117969"
    ],
    "removed_unique_values": [
      "Q102117969"
    ],
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

## 006. `repair_Q1227431_2442982249`

| Field | Value |
|---|---|
| qid | Q1227431 |
| property | P197 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q1227431::P197 |
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
| truth_tokens_preview | ["Q533236", "Q3836339", "Q666041", "Q15642563"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Bouzinac",
  "kind": "A_BOX",
  "new_value": [
    "Q533236",
    "Q3836339",
    "Q666041",
    "Q15642563"
  ],
  "new_value_descriptions_en": [
    "Rome Metro station",
    "Rome Metro station",
    "Rome Metro station",
    "metro station in Rome"
  ],
  "new_value_labels_en": [
    "Re di Roma",
    "Lodi",
    "Manzoni",
    "Porta Metronia metro station"
  ],
  "old_value": [
    "Q533236",
    "Q3836339",
    "Q666041"
  ],
  "old_value_descriptions_en": [
    "Rome Metro station",
    "Rome Metro station",
    "Rome Metro station"
  ],
  "old_value_labels_en": [
    "Re di Roma",
    "Lodi",
    "Manzoni"
  ],
  "revision_id": 2442982249,
  "value": [
    "Q533236",
    "Q3836339",
    "Q666041",
    "Q15642563"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q15642563"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 4,
    "new_unique": [
      "Q15642563",
      "Q3836339",
      "Q533236",
      "Q666041"
    ],
    "new_value": [
      "Q533236",
      "Q3836339",
      "Q666041",
      "Q15642563"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 3,
    "old_unique": [
      "Q3836339",
      "Q533236",
      "Q666041"
    ],
    "old_value": [
      "Q533236",
      "Q3836339",
      "Q666041"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "Q15642563": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "Rome Metro station",
    "Rome Metro station",
    "Rome Metro station",
    "metro station in Rome"
  ],
  "value_labels_en": [
    "Re di Roma",
    "Lodi",
    "Manzoni",
    "Porta Metronia metro station"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-17T13:46:41",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P197",
  "report_revision_new": 2443415626,
  "report_revision_old": 2442993000,
  "report_violation_type": "Symmetric",
  "report_violation_type_normalized": "Symmetric",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Symmetric",
  "value": [
    "Q533236",
    "Q3836339",
    "Q666041"
  ],
  "value_descriptions_en": [
    "Rome Metro station",
    "Rome Metro station",
    "Rome Metro station"
  ],
  "value_labels_en": [
    "Re di Roma",
    "Lodi",
    "Manzoni"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 36,
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q533236",
      "Q3836339",
      "Q666041"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q533236",
    "Q3836339",
    "Q666041",
    "Q15642563"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the stations next to this station, sharing the same line(s)",
    "label": "adjacent station"
  },
  "qid": {
    "description": "Rome Metro station",
    "label": "San Giovanni"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
  },
  {
    "label_en": "symmetric constraint",
    "qid": "Q21510862"
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
      "local_ids_count": 36,
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
        "Q533236",
        "Q3836339",
        "Q666041"
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

## 007. `repair_Q123521_2445442127`

| Field | Value |
|---|---|
| qid | Q123521 |
| property | P19 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| group_key | ABOX::Q123521::P19 |
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
| truth_tokens_preview | ["Q70202"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Epìdosis",
  "kind": "A_BOX",
  "new_value": [
    "Q70202"
  ],
  "new_value_descriptions_en": [
    "municipality in the canton of Vaud in Switzerland"
  ],
  "new_value_labels_en": [
    "Moudon"
  ],
  "old_value": [
    "Q2708674"
  ],
  "old_value_descriptions_en": [
    "Dragon Ball character"
  ],
  "old_value_labels_en": [
    "Dr. Gero"
  ],
  "revision_id": 2445442127,
  "value": [
    "Q70202"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q70202"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q70202"
    ],
    "new_value": [
      "Q70202"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q2708674"
    ],
    "old_value": [
      "Q2708674"
    ],
    "removed_unique_values": [
      "Q2708674"
    ],
    "value_multiplicity_changes": {
      "Q2708674": {
        "new": 0,
        "old": 1
      },
      "Q70202": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "municipality in the canton of Vaud in Switzerland"
  ],
  "value_labels_en": [
    "Moudon"
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
  "report_violation_type": "Value type Q|2221906, Q|3895768, Q|27096213, Q|3238337, Q|6999, Q|11446, Q|16391167, Q|18670171, Q|811979, Q|115095765, Q|4130, Q|219858",
  "report_violation_type_descriptions_en": [
    "point, line or area on or near Earth",
    "place that exists only in fiction and not in reality",
    "relatively stationary place or entity that can be geographically identified, located, or described",
    "place that only exists in myths, legends and folklore",
    "physical body of astronomically-significant size, mass, or role, naturally occurring in a universe",
    "large buoyant watercraft",
    "craft designed for transportation on or through air, water, or space",
    "vessel which only exists in fiction",
    "human-designed and -made structure",
    "location of something (be it physical, virtual, digital, real or fictional)",
    "curved path of an object around a point",
    "part of a larger area or volume with some distinguishing characteristic"
  ],
  "report_violation_type_labels_en": [
    "geographic location",
    "fictional location",
    "geographic entity",
    "mythical location",
    "astronomical object",
    "ship",
    "vessel",
    "fictional vessel",
    "architectural structure",
    "location",
    "orbit",
    "zone"
  ],
  "report_violation_type_normalized": "Value type Q|2221906, Q|3895768, Q|27096213, Q|3238337, Q|6999, Q|11446, Q|16391167, Q|18670171, Q|811979, Q|115095765, Q|4130, Q|219858",
  "report_violation_type_qids": [
    "Q2221906",
    "Q3895768",
    "Q27096213",
    "Q3238337",
    "Q6999",
    "Q11446",
    "Q16391167",
    "Q18670171",
    "Q811979",
    "Q115095765",
    "Q4130",
    "Q219858"
  ],
  "report_violation_type_raw": "Value type Q|2221906, Q|3895768, Q|27096213, Q|3238337, Q|6999, Q|11446, Q|16391167, Q|18670171, Q|811979, Q|115095765, Q|4130, Q|219858",
  "value": [
    "Q2708674"
  ],
  "value_descriptions_en": [
    "Dragon Ball character"
  ],
  "value_labels_en": [
    "Dr. Gero"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 82,
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q2708674"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q70202"
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
    "description": "Swiss poet (1925-2021)",
    "label": "Philippe Jaccottet"
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
      "local_ids_count": 82,
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
        "Q2708674"
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

## 008. `repair_Q131430652_2286428807`

| Field | Value |
|---|---|
| qid | Q131430652 |
| property | P175 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| group_key | ABOX::Q131430652::P175 |
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
| truth_tokens_preview | ["Q23471"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "The-anyel",
  "kind": "A_BOX",
  "new_value": [
    "Q23471"
  ],
  "new_value_descriptions_en": [
    "Finnish rock band"
  ],
  "new_value_labels_en": [
    "Lordi"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2286428807,
  "value": [
    "Q23471"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q23471"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q23471"
    ],
    "new_value": [
      "Q23471"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q23471": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "Finnish rock band"
  ],
  "value_labels_en": [
    "Lordi"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-12-15T10:06:39",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P175",
  "report_revision_new": 2287615514,
  "report_revision_old": 2287158081,
  "report_violation_type": "Type Q|386724, Q|1656682, Q|95074, Q|1707847, Q|98216532, Q|115668795, Q|7725310, Q|14514600",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "temporary and scheduled happening, like a conference, festival, competition or similar",
    "fictional human or non-human character in a narrative work of art",
    "character or part played by a performer",
    "period of time devoted to recording music in a studio",
    null,
    "ordered set of creative works",
    "set of fictional characters"
  ],
  "report_violation_type_labels_en": [
    "work",
    "event",
    "character",
    "role",
    "recording session",
    "audiovisual release",
    "series of creative works",
    "group of fictional characters"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|1656682, Q|95074, Q|1707847, Q|98216532, Q|115668795, Q|7725310, Q|14514600",
  "report_violation_type_qids": [
    "Q386724",
    "Q1656682",
    "Q95074",
    "Q1707847",
    "Q98216532",
    "Q115668795",
    "Q7725310",
    "Q14514600"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|1656682, Q|95074, Q|1707847, Q|98216532, Q|115668795, Q|7725310, Q|14514600",
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
  "local_ids_count": 25,
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
    "Q23471"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "actor, musician, band or other performer associated with this role or musical work",
    "label": "performer"
  },
  "qid": {
    "description": "2025 studio album by Lordi",
    "label": "Limited Deadition"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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

## 009. `repair_Q135499656_2388912528`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q60101"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "Q200",
      "Q201",
      "Q60101"
    ],
    "new_value": [
      "Q200",
      "Q201",
      "Q60101"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "Q200",
      "Q201"
    ],
    "old_value": [
      "Q200",
      "Q201"
    ],
    "removed_unique_values": [],
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

## 010. `repair_Q135501528_2388922197`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q713181"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 4,
    "new_unique": [
      "Q200",
      "Q201",
      "Q203",
      "Q713181"
    ],
    "new_value": [
      "Q200",
      "Q201",
      "Q203",
      "Q713181"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 3,
    "old_unique": [
      "Q200",
      "Q201",
      "Q203"
    ],
    "old_value": [
      "Q200",
      "Q201",
      "Q203"
    ],
    "removed_unique_values": [],
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

## 011. `repair_Q135504583_2388914308`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q19242243"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "Q19242243",
      "Q200",
      "Q201"
    ],
    "new_value": [
      "Q200",
      "Q201",
      "Q19242243"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "Q200",
      "Q201"
    ],
    "old_value": [
      "Q200",
      "Q201"
    ],
    "removed_unique_values": [],
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

## 012. `repair_Q135507014_2388932488`

| Field | Value |
|---|---|
| qid | Q135507014 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135507014::P5236 |
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
| truth_tokens_preview | ["Q200", "Q859303"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q200",
    "Q859303"
  ],
  "new_value_descriptions_en": [
    "natural number",
    "natural number"
  ],
  "new_value_labels_en": [
    "ҩба",
    "1997"
  ],
  "old_value": [
    "Q200"
  ],
  "old_value_descriptions_en": [
    "natural number"
  ],
  "old_value_labels_en": [
    "ҩба"
  ],
  "revision_id": 2388932488,
  "value": [
    "Q200",
    "Q859303"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q859303"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "Q200",
      "Q859303"
    ],
    "new_value": [
      "Q200",
      "Q859303"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q200"
    ],
    "old_value": [
      "Q200"
    ],
    "removed_unique_values": [],
    "value_multiplicity_changes": {
      "Q859303": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "natural number",
    "natural number"
  ],
  "value_labels_en": [
    "ҩба",
    "1997"
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
    "Q200"
  ],
  "value_descriptions_en": [
    "natural number"
  ],
  "value_labels_en": [
    "ҩба"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 16,
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q200"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q200",
    "Q859303"
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
    "label": "15976"
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
        "Q200"
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

## 013. `repair_Q135568024_2389811068`

| Field | Value |
|---|---|
| qid | Q135568024 |
| property | P5236 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | ABOX::Q135568024::P5236 |
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
| truth_tokens_preview | ["Q200", "Q201", "Q19246426"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Dolphyb",
  "kind": "A_BOX",
  "new_value": [
    "Q200",
    "Q201",
    "Q19246426"
  ],
  "new_value_descriptions_en": [
    "natural number",
    "natural number",
    "number"
  ],
  "new_value_labels_en": [
    "ҩба",
    "3",
    "4861"
  ],
  "old_value": [
    "Q200",
    "Q201",
    "Q202"
  ],
  "old_value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number"
  ],
  "old_value_labels_en": [
    "ҩба",
    "3",
    "4"
  ],
  "revision_id": 2389811068,
  "value": [
    "Q200",
    "Q201",
    "Q19246426"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q19246426"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "Q19246426",
      "Q200",
      "Q201"
    ],
    "new_value": [
      "Q200",
      "Q201",
      "Q19246426"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 3,
    "old_unique": [
      "Q200",
      "Q201",
      "Q202"
    ],
    "old_value": [
      "Q200",
      "Q201",
      "Q202"
    ],
    "removed_unique_values": [
      "Q202"
    ],
    "value_multiplicity_changes": {
      "Q19246426": {
        "new": 1,
        "old": 0
      },
      "Q202": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "natural number",
    "natural number",
    "number"
  ],
  "value_labels_en": [
    "ҩба",
    "3",
    "4861"
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
    "Q201",
    "Q202"
  ],
  "value_descriptions_en": [
    "natural number",
    "natural number",
    "natural number"
  ],
  "value_labels_en": [
    "ҩба",
    "3",
    "4"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 18,
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q202"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q200",
      "Q201",
      "Q202"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q200",
    "Q201",
    "Q19246426"
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
    "label": "58332"
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
      "found": 1,
      "independent_match_count": 0,
      "local_ids_count": 18,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q202"
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
        "Q200",
        "Q201",
        "Q202"
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

## 014. `repair_Q136008663_2442459290`

| Field | Value |
|---|---|
| qid | Q136008663 |
| property | P40 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q136008663::P40 |
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
| truth_tokens_preview | ["Q27431168", "Q60843482"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Q27431168",
    "Q60843482"
  ],
  "new_value_descriptions_en": [
    "Swiss theologian and philosopher",
    "Swiss physician"
  ],
  "new_value_labels_en": [
    "David Constant",
    "Jacob Constant de Rebecque"
  ],
  "old_value": [
    "Q136008662",
    "Q60843482"
  ],
  "old_value_descriptions_en": [
    "Swiss theologian and philosopher",
    "Swiss physician"
  ],
  "old_value_labels_en": [
    "David Constant",
    "Jacob Constant de Rebecque"
  ],
  "revision_id": 2442459290,
  "value": [
    "Q27431168",
    "Q60843482"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q27431168"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "Q27431168",
      "Q60843482"
    ],
    "new_value": [
      "Q27431168",
      "Q60843482"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "Q136008662",
      "Q60843482"
    ],
    "old_value": [
      "Q136008662",
      "Q60843482"
    ],
    "removed_unique_values": [
      "Q136008662"
    ],
    "value_multiplicity_changes": {
      "Q136008662": {
        "new": 0,
        "old": 1
      },
      "Q27431168": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "Swiss theologian and philosopher",
    "Swiss physician"
  ],
  "value_labels_en": [
    "David Constant",
    "Jacob Constant de Rebecque"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-16T14:17:16",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P40",
  "report_revision_new": 2443012791,
  "report_revision_old": 2442714053,
  "report_violation_type": "Target required claim P|21",
  "report_violation_type_normalized": "Target required claim P|21",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|21",
  "report_violation_types": [
    "Target required claim P|21",
    "Value type Q|5, Q|95074, Q|4271324, Q|729, Q|178885, Q|24334299, Q|21070598, Q|16979650, Q|21070568, Q|13002315, Q|207174, Q|2135501, Q|21191150, Q|75855169, Q|4886, Q|64520857, Q|115537581, Q|795052"
  ],
  "value": [
    "Q136008662",
    "Q60843482"
  ],
  "value_descriptions_en": [
    "Swiss theologian and philosopher",
    "Swiss physician"
  ],
  "value_labels_en": [
    "David Constant",
    "Jacob Constant de Rebecque"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 11,
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q136008662"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q136008662",
      "Q60843482"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q27431168",
    "Q60843482"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "subject has object as child. Do not use for stepchildren—use \"relative\" (P1038), qualified with \"type of kinship\" (P1039)",
    "label": "child"
  },
  "qid": {
    "description": "Aug 1615 Lausanne - 2 Mar 1678 Lausanne",
    "label": "Philibert Constant"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
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
      "local_ids_count": 11,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q136008662"
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
        "Q136008662",
        "Q60843482"
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

## 015. `repair_Q136338282_2443310307`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q2646258"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q2646258"
    ],
    "new_value": [
      "Q2646258"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q136338283"
    ],
    "old_value": [
      "Q136338283"
    ],
    "removed_unique_values": [
      "Q136338283"
    ],
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

## 016. `repair_Q14391_2443678173`

| Field | Value |
|---|---|
| qid | Q14391 |
| property | P1313 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q14391::P1313 |
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
| truth_tokens_preview | ["Q28484757"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Danil Satria",
  "kind": "A_BOX",
  "new_value": [
    "Q28484757"
  ],
  "new_value_descriptions_en": [
    "Wikimedia list article"
  ],
  "new_value_labels_en": [
    "list of Regents of South Barito"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2443678173,
  "value": [
    "Q28484757"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q28484757"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q28484757"
    ],
    "new_value": [
      "Q28484757"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q28484757": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "Wikimedia list article"
  ],
  "value_labels_en": [
    "list of Regents of South Barito"
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
  "local_ids_count": 41,
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
    "Q28484757"
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
    "description": "regency in Central Kalimantan Province, Indonesia",
    "label": "South Barito"
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
      "local_ids_count": 41,
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

## 017. `repair_Q15359_2443679501`

| Field | Value |
|---|---|
| qid | Q15359 |
| property | P1313 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q15359::P1313 |
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
| truth_tokens_preview | ["Q28485026"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Danil Satria",
  "kind": "A_BOX",
  "new_value": [
    "Q28485026"
  ],
  "new_value_descriptions_en": [
    "political position held by the regional leader of Konawe Regency"
  ],
  "new_value_labels_en": [
    "Regent of Konawe"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2443679501,
  "value": [
    "Q28485026"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q28485026"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q28485026"
    ],
    "new_value": [
      "Q28485026"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q28485026": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "political position held by the regional leader of Konawe Regency"
  ],
  "value_labels_en": [
    "Regent of Konawe"
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
  "local_ids_count": 70,
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
    "Q28485026"
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
    "description": "regency in Southeast Sulawesi Province, Indonesia",
    "label": "Konawe"
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
      "local_ids_count": 70,
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

## 018. `repair_Q18205762_2444271422`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q4881614"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q4881614"
    ],
    "new_value": [
      "Q4881614"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q116550025"
    ],
    "old_value": [
      "Q116550025"
    ],
    "removed_unique_values": [
      "Q116550025"
    ],
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

## 019. `repair_Q19698037_2441300303`

| Field | Value |
|---|---|
| qid | Q19698037 |
| property | P735 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| group_key | ABOX::Q19698037::P735 |
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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
  "revision_id": 2441300303,
  "value": [
    "Q3061675"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q3061675"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q3061675"
    ],
    "new_value": [
      "Q3061675"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q15731812"
    ],
    "old_value": [
      "Q15731812"
    ],
    "removed_unique_values": [
      "Q15731812"
    ],
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
    "Target required claim P|282",
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
  "local_ids_count": 36,
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
    "description": "Polish poet",
    "label": "Ewa Karbowska"
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
      "local_ids_count": 36,
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

## 020. `repair_Q2078535_2441060815`

| Field | Value |
|---|---|
| qid | Q2078535 |
| property | P1464 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21502838 |
| group_key | ABOX::Q2078535::P1464 |
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
| truth_tokens_preview | ["Q65599042"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Mercenario97",
  "kind": "A_BOX",
  "new_value": [
    "Q65599042"
  ],
  "new_value_descriptions_en": [
    "Wikimedia category"
  ],
  "new_value_labels_en": [
    "Category:Births in Chorley"
  ],
  "old_value": [
    "Q122648087"
  ],
  "old_value_descriptions_en": [
    "Wikimedia category"
  ],
  "old_value_labels_en": [
    "Category:Births in Chorley"
  ],
  "revision_id": 2441060815,
  "value": [
    "Q65599042"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q65599042"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q65599042"
    ],
    "new_value": [
      "Q65599042"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q122648087"
    ],
    "old_value": [
      "Q122648087"
    ],
    "removed_unique_values": [
      "Q122648087"
    ],
    "value_multiplicity_changes": {
      "Q122648087": {
        "new": 0,
        "old": 1
      },
      "Q65599042": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "Wikimedia category"
  ],
  "value_labels_en": [
    "Category:Births in Chorley"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-13T09:20:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1464",
  "report_revision_new": 2441731472,
  "report_revision_old": 2441152789,
  "report_violation_type": "Entity types",
  "report_violation_type_normalized": "Entity types",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Entity types",
  "value": [
    "Q122648087"
  ],
  "value_descriptions_en": [
    "Wikimedia category"
  ],
  "value_labels_en": [
    "Category:Births in Chorley"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 32,
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q122648087"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q65599042"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "category item that groups people born in this place",
    "label": "category for people born here"
  },
  "qid": {
    "description": "town in Lancashire, England",
    "label": "Chorley"
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
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
      "local_ids_count": 32,
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
        "Q122648087"
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

## 021. `repair_Q21496758_2396444791`

| Field | Value |
|---|---|
| qid | Q21496758 |
| property | P682 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | ABOX::Q21496758::P682 |
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
| truth_tokens_preview | ["Q14874273", "Q14903270", "Q21123869", "Q21111296", "Q4374357", "...(+29)"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Q14874273",
    "Q14903270",
    "Q21123869",
    "Q21111296",
    "Q4374357",
    "Q21107103",
    "Q14878788",
    "Q14860192",
    "Q14633883",
    "Q21101535",
    "Q14818001",
    "Q14819859",
    "Q14904615",
    "Q14763010",
    "Q518328",
    "Q21100570",
    "Q14890445",
    "Q21174980",
    "Q14885189",
    "Q14865486",
    "Q14633878",
    "Q14911565",
    "Q21103310",
    "Q14645705",
    "... omitted 11 items"
  ],
  "new_value_descriptions_en": [
    "The process whose specific outcome is the progression of an immature germ cell over time, from its formation to the mature structure (gamete). A germ cell is any reproductive cell in a multicellular organism.",
    "progression of a cardiac septum over time, from its initial formation to the mature structure",
    "The progression of a heart valve over time, from its formation to the mature structure. A heart valve is a structure that restricts the flow of blood to different regions of the heart and forms from an endocardial cushion.",
    "The process whose specific outcome is the progression of a columnar/cuboidal epithelial cell of the intestine over time, from its formation to the mature structure.",
    "The process whose specific outcome is the progression of the organism over time, from the completion of embryonic development to the mature structure. See embryonic development.",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of signaling in response to detection of lipopolysaccharide.",
    "The process in which a relatively unspecialized cell acquires specialized features of a trophoblast giant cell of the placenta. Trophoblast giant cells are the cell of the placenta that line the maternal decidua.",
    "commitment of cells to specific cell fates and their capacity to differentiate into particular kinds of cells",
    "Any process that modulates the frequency, rate or extent of cellular DNA-templated transcription.",
    "The progression of the ventricular septum over time from its formation to the mature structure.",
    "The process whose specific outcome is the progression of the embryo in the uterus over time, from formation of the zygote in the oviduct, to birth. An example of this process is found in Mus musculus.",
    "Any process that decreases the frequency, rate or extent of gene expression. Gene expression is the process in which a gene's coding sequence is converted into a mature gene product or products (proteins or RNA). This includes the production of an RN",
    "The process in which the anatomical structures of branches are generated and organized. A branch is a division or offshoot from a main stem. Examples in animals would include blood vessels, nerves, lymphatics and other endothelial or epithelial tubes",
    "The cellular synthesis of RNA on a template of DNA.",
    "process in which a methyl group is covalently attached to a molecule",
    "Maternally driven process whose specific outcome is the progression of the placenta over time, from its formation to the mature structure. The placenta is an organ of metabolic interchange between fetus and mother, partly of embryonic origin and part",
    "The process in which the anatomical structures of arterial blood vessels are generated and organized. Arteries are blood vessels that transport blood from the heart to the body and its organs.",
    "The multiplication or reproduction of sebocytes by cell division, resulting in the expansion of their population. A sebocyte is an epithelial cell that makes up the sebaceous glands, and secrete sebum.",
    "Any process that activates or increases the frequency, rate or extent of B cell differentiation.",
    "The embryonically driven process whose specific outcome is the progression of the placenta over time, from its formation to the mature structure. The placenta is an organ of metabolic interchange between fetus and mother, partly of embryonic origin a",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of transcription from an RNA polymerase II promoter.",
    "The process whose specific outcome is the progression of the blood vessels of the heart over time, from its formation to the mature structure.",
    "The progression of the aorta over time, from its initial formation to the mature structure. An aorta is an artery that carries blood from the heart to other parts of the body.",
    "biological process whose specific outcome is the progression of a multicellular organism over time from an initial condition (e.g. a zygote or a young adult) to a later condition (e.g. a multicellular animal or an aged adult)",
    "... omitted 11 items"
  ],
  "new_value_labels_en": [
    "germ cell development",
    "cardiac septum development",
    "heart valve development",
    "intestinal epithelial cell development",
    "post-embryonic development",
    "negative regulation of lipopolysaccharide-mediated signaling pathway",
    "trophoblast giant cell differentiation",
    "cell fate commitment",
    "regulation of transcription, DNA-templated",
    "ventricular septum development",
    "in utero embryonic development",
    "negative regulation of gene expression",
    "morphogenesis of a branching structure",
    "transcription, DNA-templated",
    "methylation",
    "maternal placenta development",
    "artery morphogenesis",
    "sebum secreting cell proliferation",
    "positive regulation of B cell differentiation",
    "embryonic placenta development",
    "negative regulation of transcription by RNA polymerase II",
    "coronary vasculature development",
    "aorta development",
    "multicellular organism development",
    "... omitted 11 items"
  ],
  "old_value": [
    "Q14874273",
    "Q14903270",
    "Q21123869",
    "Q21111296",
    "Q14820052",
    "Q21107103",
    "Q14878788",
    "Q14860192",
    "Q14633883",
    "Q21101535",
    "Q14818001",
    "Q14819859",
    "Q14904615",
    "Q14763010",
    "Q518328",
    "Q21100570",
    "Q14890445",
    "Q21174980",
    "Q14885189",
    "Q14865486",
    "Q14633878",
    "Q14911565",
    "Q21103310",
    "Q14645705",
    "... omitted 11 items"
  ],
  "old_value_descriptions_en": [
    "The process whose specific outcome is the progression of an immature germ cell over time, from its formation to the mature structure (gamete). A germ cell is any reproductive cell in a multicellular organism.",
    "progression of a cardiac septum over time, from its initial formation to the mature structure",
    "The progression of a heart valve over time, from its formation to the mature structure. A heart valve is a structure that restricts the flow of blood to different regions of the heart and forms from an endocardial cushion.",
    "The process whose specific outcome is the progression of a columnar/cuboidal epithelial cell of the intestine over time, from its formation to the mature structure.",
    "The process whose specific outcome is the progression of the organism over time, from the completion of embryonic development to the mature structure. See embryonic development.",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of signaling in response to detection of lipopolysaccharide.",
    "The process in which a relatively unspecialized cell acquires specialized features of a trophoblast giant cell of the placenta. Trophoblast giant cells are the cell of the placenta that line the maternal decidua.",
    "commitment of cells to specific cell fates and their capacity to differentiate into particular kinds of cells",
    "Any process that modulates the frequency, rate or extent of cellular DNA-templated transcription.",
    "The progression of the ventricular septum over time from its formation to the mature structure.",
    "The process whose specific outcome is the progression of the embryo in the uterus over time, from formation of the zygote in the oviduct, to birth. An example of this process is found in Mus musculus.",
    "Any process that decreases the frequency, rate or extent of gene expression. Gene expression is the process in which a gene's coding sequence is converted into a mature gene product or products (proteins or RNA). This includes the production of an RN",
    "The process in which the anatomical structures of branches are generated and organized. A branch is a division or offshoot from a main stem. Examples in animals would include blood vessels, nerves, lymphatics and other endothelial or epithelial tubes",
    "The cellular synthesis of RNA on a template of DNA.",
    "process in which a methyl group is covalently attached to a molecule",
    "Maternally driven process whose specific outcome is the progression of the placenta over time, from its formation to the mature structure. The placenta is an organ of metabolic interchange between fetus and mother, partly of embryonic origin and part",
    "The process in which the anatomical structures of arterial blood vessels are generated and organized. Arteries are blood vessels that transport blood from the heart to the body and its organs.",
    "The multiplication or reproduction of sebocytes by cell division, resulting in the expansion of their population. A sebocyte is an epithelial cell that makes up the sebaceous glands, and secrete sebum.",
    "Any process that activates or increases the frequency, rate or extent of B cell differentiation.",
    "The embryonically driven process whose specific outcome is the progression of the placenta over time, from its formation to the mature structure. The placenta is an organ of metabolic interchange between fetus and mother, partly of embryonic origin a",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of transcription from an RNA polymerase II promoter.",
    "The process whose specific outcome is the progression of the blood vessels of the heart over time, from its formation to the mature structure.",
    "The progression of the aorta over time, from its initial formation to the mature structure. An aorta is an artery that carries blood from the heart to other parts of the body.",
    "biological process whose specific outcome is the progression of a multicellular organism over time from an initial condition (e.g. a zygote or a young adult) to a later condition (e.g. a multicellular animal or an aged adult)",
    "... omitted 11 items"
  ],
  "old_value_labels_en": [
    "germ cell development",
    "cardiac septum development",
    "heart valve development",
    "intestinal epithelial cell development",
    "post-embryonic development",
    "negative regulation of lipopolysaccharide-mediated signaling pathway",
    "trophoblast giant cell differentiation",
    "cell fate commitment",
    "regulation of transcription, DNA-templated",
    "ventricular septum development",
    "in utero embryonic development",
    "negative regulation of gene expression",
    "morphogenesis of a branching structure",
    "transcription, DNA-templated",
    "methylation",
    "maternal placenta development",
    "artery morphogenesis",
    "sebum secreting cell proliferation",
    "positive regulation of B cell differentiation",
    "embryonic placenta development",
    "negative regulation of transcription by RNA polymerase II",
    "coronary vasculature development",
    "aorta development",
    "multicellular organism development",
    "... omitted 11 items"
  ],
  "revision_id": 2396444791,
  "value": [
    "Q14874273",
    "Q14903270",
    "Q21123869",
    "Q21111296",
    "Q4374357",
    "Q21107103",
    "Q14878788",
    "Q14860192",
    "Q14633883",
    "Q21101535",
    "Q14818001",
    "Q14819859",
    "Q14904615",
    "Q14763010",
    "Q518328",
    "Q21100570",
    "Q14890445",
    "Q21174980",
    "Q14885189",
    "Q14865486",
    "Q14633878",
    "Q14911565",
    "Q21103310",
    "Q14645705",
    "... omitted 11 items"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q4374357"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 35,
    "new_unique": [
      "Q14633878",
      "Q14633883",
      "Q14633893",
      "Q14645705",
      "Q14763010",
      "Q14818001",
      "Q14818120",
      "Q14819859",
      "Q14860192",
      "Q14864446",
      "Q14865486",
      "Q14874273",
      "Q14878788",
      "Q14885189",
      "Q14887714",
      "Q14890445",
      "Q14903270",
      "Q14904615",
      "Q14911565",
      "Q14914022",
      "Q14916317",
      "Q21100570",
      "Q21101535",
      "Q21103310",
      "... omitted 10 items"
    ],
    "new_value": [
      "Q14874273",
      "Q14903270",
      "Q21123869",
      "Q21111296",
      "Q4374357",
      "Q21107103",
      "Q14878788",
      "Q14860192",
      "Q14633883",
      "Q21101535",
      "Q14818001",
      "Q14819859",
      "Q14904615",
      "Q14763010",
      "Q518328",
      "Q21100570",
      "Q14890445",
      "Q21174980",
      "Q14885189",
      "Q14865486",
      "Q14633878",
      "Q14911565",
      "Q21103310",
      "Q14645705",
      "... omitted 11 items"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 35,
    "old_unique": [
      "Q14633878",
      "Q14633883",
      "Q14633893",
      "Q14645705",
      "Q14763010",
      "Q14818001",
      "Q14818120",
      "Q14819859",
      "Q14820052",
      "Q14860192",
      "Q14864446",
      "Q14865486",
      "Q14874273",
      "Q14878788",
      "Q14885189",
      "Q14887714",
      "Q14890445",
      "Q14903270",
      "Q14904615",
      "Q14911565",
      "Q14914022",
      "Q14916317",
      "Q21100570",
      "Q21101535",
      "... omitted 10 items"
    ],
    "old_value": [
      "Q14874273",
      "Q14903270",
      "Q21123869",
      "Q21111296",
      "Q14820052",
      "Q21107103",
      "Q14878788",
      "Q14860192",
      "Q14633883",
      "Q21101535",
      "Q14818001",
      "Q14819859",
      "Q14904615",
      "Q14763010",
      "Q518328",
      "Q21100570",
      "Q14890445",
      "Q21174980",
      "Q14885189",
      "Q14865486",
      "Q14633878",
      "Q14911565",
      "Q21103310",
      "Q14645705",
      "... omitted 11 items"
    ],
    "removed_unique_values": [
      "Q14820052"
    ],
    "value_multiplicity_changes": {
      "Q14820052": {
        "new": 0,
        "old": 1
      },
      "Q4374357": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "The process whose specific outcome is the progression of an immature germ cell over time, from its formation to the mature structure (gamete). A germ cell is any reproductive cell in a multicellular organism.",
    "progression of a cardiac septum over time, from its initial formation to the mature structure",
    "The progression of a heart valve over time, from its formation to the mature structure. A heart valve is a structure that restricts the flow of blood to different regions of the heart and forms from an endocardial cushion.",
    "The process whose specific outcome is the progression of a columnar/cuboidal epithelial cell of the intestine over time, from its formation to the mature structure.",
    "The process whose specific outcome is the progression of the organism over time, from the completion of embryonic development to the mature structure. See embryonic development.",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of signaling in response to detection of lipopolysaccharide.",
    "The process in which a relatively unspecialized cell acquires specialized features of a trophoblast giant cell of the placenta. Trophoblast giant cells are the cell of the placenta that line the maternal decidua.",
    "commitment of cells to specific cell fates and their capacity to differentiate into particular kinds of cells",
    "Any process that modulates the frequency, rate or extent of cellular DNA-templated transcription.",
    "The progression of the ventricular septum over time from its formation to the mature structure.",
    "The process whose specific outcome is the progression of the embryo in the uterus over time, from formation of the zygote in the oviduct, to birth. An example of this process is found in Mus musculus.",
    "Any process that decreases the frequency, rate or extent of gene expression. Gene expression is the process in which a gene's coding sequence is converted into a mature gene product or products (proteins or RNA). This includes the production of an RN",
    "The process in which the anatomical structures of branches are generated and organized. A branch is a division or offshoot from a main stem. Examples in animals would include blood vessels, nerves, lymphatics and other endothelial or epithelial tubes",
    "The cellular synthesis of RNA on a template of DNA.",
    "process in which a methyl group is covalently attached to a molecule",
    "Maternally driven process whose specific outcome is the progression of the placenta over time, from its formation to the mature structure. The placenta is an organ of metabolic interchange between fetus and mother, partly of embryonic origin and part",
    "The process in which the anatomical structures of arterial blood vessels are generated and organized. Arteries are blood vessels that transport blood from the heart to the body and its organs.",
    "The multiplication or reproduction of sebocytes by cell division, resulting in the expansion of their population. A sebocyte is an epithelial cell that makes up the sebaceous glands, and secrete sebum.",
    "Any process that activates or increases the frequency, rate or extent of B cell differentiation.",
    "The embryonically driven process whose specific outcome is the progression of the placenta over time, from its formation to the mature structure. The placenta is an organ of metabolic interchange between fetus and mother, partly of embryonic origin a",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of transcription from an RNA polymerase II promoter.",
    "The process whose specific outcome is the progression of the blood vessels of the heart over time, from its formation to the mature structure.",
    "The progression of the aorta over time, from its initial formation to the mature structure. An aorta is an artery that carries blood from the heart to other parts of the body.",
    "biological process whose specific outcome is the progression of a multicellular organism over time from an initial condition (e.g. a zygote or a young adult) to a later condition (e.g. a multicellular animal or an aged adult)",
    "... omitted 11 items"
  ],
  "value_labels_en": [
    "germ cell development",
    "cardiac septum development",
    "heart valve development",
    "intestinal epithelial cell development",
    "post-embryonic development",
    "negative regulation of lipopolysaccharide-mediated signaling pathway",
    "trophoblast giant cell differentiation",
    "cell fate commitment",
    "regulation of transcription, DNA-templated",
    "ventricular septum development",
    "in utero embryonic development",
    "negative regulation of gene expression",
    "morphogenesis of a branching structure",
    "transcription, DNA-templated",
    "methylation",
    "maternal placenta development",
    "artery morphogenesis",
    "sebum secreting cell proliferation",
    "positive regulation of B cell differentiation",
    "embryonic placenta development",
    "negative regulation of transcription by RNA polymerase II",
    "coronary vasculature development",
    "aorta development",
    "multicellular organism development",
    "... omitted 11 items"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-27T09:38:33",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P682",
  "report_revision_new": 2397202759,
  "report_revision_old": 2395081787,
  "report_violation_type": "Target required claim P|686",
  "report_violation_type_normalized": "Target required claim P|686",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|686",
  "value": [
    "Q14874273",
    "Q14903270",
    "Q21123869",
    "Q21111296",
    "Q14820052",
    "Q21107103",
    "Q14878788",
    "Q14860192",
    "Q14633883",
    "Q21101535",
    "Q14818001",
    "Q14819859",
    "Q14904615",
    "Q14763010",
    "Q518328",
    "Q21100570",
    "Q14890445",
    "Q21174980",
    "Q14885189",
    "Q14865486",
    "Q14633878",
    "Q14911565",
    "Q21103310",
    "Q14645705",
    "... omitted 11 items"
  ],
  "value_descriptions_en": [
    "The process whose specific outcome is the progression of an immature germ cell over time, from its formation to the mature structure (gamete). A germ cell is any reproductive cell in a multicellular organism.",
    "progression of a cardiac septum over time, from its initial formation to the mature structure",
    "The progression of a heart valve over time, from its formation to the mature structure. A heart valve is a structure that restricts the flow of blood to different regions of the heart and forms from an endocardial cushion.",
    "The process whose specific outcome is the progression of a columnar/cuboidal epithelial cell of the intestine over time, from its formation to the mature structure.",
    "The process whose specific outcome is the progression of the organism over time, from the completion of embryonic development to the mature structure. See embryonic development.",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of signaling in response to detection of lipopolysaccharide.",
    "The process in which a relatively unspecialized cell acquires specialized features of a trophoblast giant cell of the placenta. Trophoblast giant cells are the cell of the placenta that line the maternal decidua.",
    "commitment of cells to specific cell fates and their capacity to differentiate into particular kinds of cells",
    "Any process that modulates the frequency, rate or extent of cellular DNA-templated transcription.",
    "The progression of the ventricular septum over time from its formation to the mature structure.",
    "The process whose specific outcome is the progression of the embryo in the uterus over time, from formation of the zygote in the oviduct, to birth. An example of this process is found in Mus musculus.",
    "Any process that decreases the frequency, rate or extent of gene expression. Gene expression is the process in which a gene's coding sequence is converted into a mature gene product or products (proteins or RNA). This includes the production of an RN",
    "The process in which the anatomical structures of branches are generated and organized. A branch is a division or offshoot from a main stem. Examples in animals would include blood vessels, nerves, lymphatics and other endothelial or epithelial tubes",
    "The cellular synthesis of RNA on a template of DNA.",
    "process in which a methyl group is covalently attached to a molecule",
    "Maternally driven process whose specific outcome is the progression of the placenta over time, from its formation to the mature structure. The placenta is an organ of metabolic interchange between fetus and mother, partly of embryonic origin and part",
    "The process in which the anatomical structures of arterial blood vessels are generated and organized. Arteries are blood vessels that transport blood from the heart to the body and its organs.",
    "The multiplication or reproduction of sebocytes by cell division, resulting in the expansion of their population. A sebocyte is an epithelial cell that makes up the sebaceous glands, and secrete sebum.",
    "Any process that activates or increases the frequency, rate or extent of B cell differentiation.",
    "The embryonically driven process whose specific outcome is the progression of the placenta over time, from its formation to the mature structure. The placenta is an organ of metabolic interchange between fetus and mother, partly of embryonic origin a",
    "Any process that stops, prevents, or reduces the frequency, rate or extent of transcription from an RNA polymerase II promoter.",
    "The process whose specific outcome is the progression of the blood vessels of the heart over time, from its formation to the mature structure.",
    "The progression of the aorta over time, from its initial formation to the mature structure. An aorta is an artery that carries blood from the heart to other parts of the body.",
    "biological process whose specific outcome is the progression of a multicellular organism over time from an initial condition (e.g. a zygote or a young adult) to a later condition (e.g. a multicellular animal or an aged adult)",
    "... omitted 11 items"
  ],
  "value_labels_en": [
    "germ cell development",
    "cardiac septum development",
    "heart valve development",
    "intestinal epithelial cell development",
    "post-embryonic development",
    "negative regulation of lipopolysaccharide-mediated signaling pathway",
    "trophoblast giant cell differentiation",
    "cell fate commitment",
    "regulation of transcription, DNA-templated",
    "ventricular septum development",
    "in utero embryonic development",
    "negative regulation of gene expression",
    "morphogenesis of a branching structure",
    "transcription, DNA-templated",
    "methylation",
    "maternal placenta development",
    "artery morphogenesis",
    "sebum secreting cell proliferation",
    "positive regulation of B cell differentiation",
    "embryonic placenta development",
    "negative regulation of transcription by RNA polymerase II",
    "coronary vasculature development",
    "aorta development",
    "multicellular organism development",
    "... omitted 11 items"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 71,
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q14820052"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q14874273",
      "Q14903270",
      "Q21123869",
      "Q21111296",
      "Q14820052",
      "Q21107103",
      "Q14878788",
      "Q14860192",
      "Q14633883",
      "Q21101535",
      "Q14818001",
      "Q14819859",
      "Q14904615",
      "Q14763010",
      "Q518328",
      "Q21100570",
      "Q14890445",
      "Q21174980",
      "Q14885189",
      "Q14865486",
      "Q14633878",
      "Q14911565",
      "Q21103310",
      "Q14645705",
      "... omitted 10 items"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q14874273",
    "Q14903270",
    "Q21123869",
    "Q21111296",
    "Q4374357",
    "Q21107103",
    "Q14878788",
    "Q14860192",
    "Q14633883",
    "Q21101535",
    "Q14818001",
    "Q14819859",
    "Q14904615",
    "Q14763010",
    "Q518328",
    "Q21100570",
    "Q14890445",
    "Q21174980",
    "Q14885189",
    "Q14865486",
    "Q14633878",
    "Q14911565",
    "Q21103310",
    "Q14645705",
    "... omitted 10 items"
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
    "label": "PR domain containing 1, with ZNF domain"
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
          "token": "Q14820052"
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
        "Q14874273",
        "Q14903270",
        "Q21123869",
        "Q21111296",
        "Q14820052",
        "Q21107103",
        "Q14878788",
        "Q14860192",
        "Q14633883",
        "Q21101535",
        "Q14818001",
        "Q14819859",
        "Q14904615",
        "Q14763010",
        "Q518328",
        "Q21100570",
        "Q14890445",
        "Q21174980",
        "Q14885189",
        "Q14865486",
        "Q14633878",
        "Q14911565",
        "Q21103310",
        "Q14645705",
        "... omitted 10 items"
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

## 022. `repair_Q21604265_2388176344`

| Field | Value |
|---|---|
| qid | Q21604265 |
| property | P682 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | ABOX::Q21604265::P682 |
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
| truth_tokens_preview | ["Q11533907"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Q11533907"
  ],
  "new_value_descriptions_en": [
    "The progressive compaction of dispersed interphase chromatin into threadlike chromosomes prior to mitotic or meiotic nuclear division, or during apoptosis, in eukaryotic cells."
  ],
  "new_value_labels_en": [
    "chromosome condensation"
  ],
  "old_value": [
    "Q15311670"
  ],
  "old_value_descriptions_en": [
    "The progressive compaction of dispersed interphase chromatin into threadlike chromosomes prior to mitotic or meiotic nuclear division, or during apoptosis, in eukaryotic cells."
  ],
  "old_value_labels_en": [
    "chromosome condensation"
  ],
  "revision_id": 2388176344,
  "value": [
    "Q11533907"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q11533907"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q11533907"
    ],
    "new_value": [
      "Q11533907"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q15311670"
    ],
    "old_value": [
      "Q15311670"
    ],
    "removed_unique_values": [
      "Q15311670"
    ],
    "value_multiplicity_changes": {
      "Q11533907": {
        "new": 1,
        "old": 0
      },
      "Q15311670": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "The progressive compaction of dispersed interphase chromatin into threadlike chromosomes prior to mitotic or meiotic nuclear division, or during apoptosis, in eukaryotic cells."
  ],
  "value_labels_en": [
    "chromosome condensation"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-06T10:50:02",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P682",
  "report_revision_new": 2388580464,
  "report_revision_old": 2388275621,
  "report_violation_type": "Target required claim P|686",
  "report_violation_type_normalized": "Target required claim P|686",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Target required claim P|686",
  "value": [
    "Q15311670"
  ],
  "value_descriptions_en": [
    "The progressive compaction of dispersed interphase chromatin into threadlike chromosomes prior to mitotic or meiotic nuclear division, or during apoptosis, in eukaryotic cells."
  ],
  "value_labels_en": [
    "chromosome condensation"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 11,
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q15311670"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q11533907"
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
    "description": "microbial protein found in Borreliella burgdorferi B31",
    "label": "DNA-binding protein HU BB_0232"
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
      "found": 0,
      "independent_match_count": 0,
      "local_ids_count": 11,
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
        "Q15311670"
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

## 023. `repair_Q2326494_2441312347`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q1573906"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q1573906"
    ],
    "new_value": [
      "Q1573906"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q12538685"
    ],
    "old_value": [
      "Q12538685"
    ],
    "removed_unique_values": [
      "Q12538685"
    ],
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

## 024. `repair_Q23547_2440621158`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q5033157"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "Q5033157",
      "Q6682369"
    ],
    "new_value": [
      "Q6682369",
      "Q5033157"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "Q1050232",
      "Q6682369"
    ],
    "old_value": [
      "Q6682369",
      "Q1050232"
    ],
    "removed_unique_values": [
      "Q1050232"
    ],
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

## 025. `repair_Q28606479_2443314174`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q7810920"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 8,
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
    "normalized_unique_values_unchanged": false,
    "old_count": 8,
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
    "removed_unique_values": [
      "Q26995469"
    ],
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

## 026. `repair_Q2889467_2441312395`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q1573906"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q1573906"
    ],
    "new_value": [
      "Q1573906"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q12538685"
    ],
    "old_value": [
      "Q12538685"
    ],
    "removed_unique_values": [
      "Q12538685"
    ],
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

## 027. `repair_Q3182098_2445431242`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q104901906"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "Q104901906",
      "Q55801"
    ],
    "new_value": [
      "Q55801",
      "Q104901906"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "Q217635",
      "Q55801"
    ],
    "old_value": [
      "Q55801",
      "Q217635"
    ],
    "removed_unique_values": [
      "Q217635"
    ],
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

## 028. `repair_Q34318765_2441936274`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q58941021"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q58941021"
    ],
    "new_value": [
      "Q58941021"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q66829366"
    ],
    "old_value": [
      "Q66829366"
    ],
    "removed_unique_values": [
      "Q66829366"
    ],
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

## 029. `repair_Q3617809_2441932428`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q63116520"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q63116520"
    ],
    "new_value": [
      "Q63116520"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q113394122"
    ],
    "old_value": [
      "Q113394122"
    ],
    "removed_unique_values": [
      "Q113394122"
    ],
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

## 030. `repair_Q41286795_2441933756`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q17129531"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 8,
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
    "normalized_unique_values_unchanged": false,
    "old_count": 8,
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
    "removed_unique_values": [
      "Q102228927"
    ],
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

## 031. `repair_Q42105171_2439944167`

| Field | Value |
|---|---|
| qid | Q42105171 |
| property | P50 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| group_key | ABOX::Q42105171::P50 |
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
| truth_tokens_preview | ["Q56810336", "Q115280707", "Q56786969", "Q87836927"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Q56810336",
    "Q115280707",
    "Q56786969",
    "Q87836927"
  ],
  "new_value_descriptions_en": [
    "researcher",
    "researcher",
    "researcher",
    "researcher"
  ],
  "new_value_labels_en": [
    "Esa-Pekka Kumpula",
    "Juha Vahokoski",
    "Inari Kursula",
    "Saligram P Bhargav"
  ],
  "old_value": [
    "Q56810336",
    "Q115280707",
    "Q87836936",
    "Q87836927"
  ],
  "old_value_descriptions_en": [
    "researcher",
    "researcher",
    "researcher",
    "researcher"
  ],
  "old_value_labels_en": [
    "Esa-Pekka Kumpula",
    "Juha Vahokoski",
    "Inari Kursula",
    "Saligram P Bhargav"
  ],
  "revision_id": 2439944167,
  "value": [
    "Q56810336",
    "Q115280707",
    "Q56786969",
    "Q87836927"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q56786969"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 4,
    "new_unique": [
      "Q115280707",
      "Q56786969",
      "Q56810336",
      "Q87836927"
    ],
    "new_value": [
      "Q56810336",
      "Q115280707",
      "Q56786969",
      "Q87836927"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 4,
    "old_unique": [
      "Q115280707",
      "Q56810336",
      "Q87836927",
      "Q87836936"
    ],
    "old_value": [
      "Q56810336",
      "Q115280707",
      "Q87836936",
      "Q87836927"
    ],
    "removed_unique_values": [
      "Q87836936"
    ],
    "value_multiplicity_changes": {
      "Q56786969": {
        "new": 1,
        "old": 0
      },
      "Q87836936": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "researcher",
    "researcher",
    "researcher",
    "researcher"
  ],
  "value_labels_en": [
    "Esa-Pekka Kumpula",
    "Juha Vahokoski",
    "Inari Kursula",
    "Saligram P Bhargav"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-10T12:43:26",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P50",
  "report_revision_new": 2440438419,
  "report_revision_old": 2440068412,
  "report_violation_type": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398",
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
    "distinct and identifiable entity with agency, capable of performing actions"
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
    "being"
  ],
  "report_violation_type_normalized": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398",
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
    "Q24229398"
  ],
  "report_violation_type_raw": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398",
  "value": [
    "Q56810336",
    "Q115280707",
    "Q87836936",
    "Q87836927"
  ],
  "value_descriptions_en": [
    "researcher",
    "researcher",
    "researcher",
    "researcher"
  ],
  "value_labels_en": [
    "Esa-Pekka Kumpula",
    "Juha Vahokoski",
    "Inari Kursula",
    "Saligram P Bhargav"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 70,
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q87836936"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q56810336",
      "Q115280707",
      "Q87836936",
      "Q87836927"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q56810336",
    "Q115280707",
    "Q56786969",
    "Q87836927"
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
    "label": "Crystallization and preliminary structural characterization of the two actin isoforms of the malaria parasite"
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
      "local_ids_count": 70,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q87836936"
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
        "Q56810336",
        "Q115280707",
        "Q87836936",
        "Q87836927"
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

## 032. `repair_Q44554219_2447217691`

| Field | Value |
|---|---|
| qid | Q44554219 |
| property | P50 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| group_key | ABOX::Q44554219::P50 |
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
| truth_tokens_preview | ["Q42739028", "Q11591785", "Q117252006", "Q44685820", "Q116792644", "...(+6)"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Q42739028",
    "Q11591785",
    "Q117252006",
    "Q44685820",
    "Q116792644",
    "Q117252007",
    "Q117249650",
    "Q59702826",
    "Q117252010",
    "Q117252011",
    "Q117252012"
  ],
  "new_value_descriptions_en": [
    "researcher",
    "Japanese physical chemist (1950-)",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher"
  ],
  "new_value_labels_en": [
    "Kei Ohkubo",
    "Shun'ichi Fukuzumi",
    "Kiyoshi Fukuhara",
    "Ikuo Nakanishi",
    "Toshihiko Ozawa",
    "Shiro Urano",
    "Haruhiro Okuda",
    "Kentaro Miyazaki",
    "Nobuo Ikota",
    "Yoshihiro Uto",
    "Hitoshi Hori"
  ],
  "old_value": [
    "Q42739028",
    "Q67623707",
    "Q117252006",
    "Q44685820",
    "Q116792644",
    "Q117252007",
    "Q117249650",
    "Q59702826",
    "Q117252010",
    "Q117252011",
    "Q117252012"
  ],
  "old_value_descriptions_en": [
    "researcher",
    "Japanese physical chemist (1950-)",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher"
  ],
  "old_value_labels_en": [
    "Kei Ohkubo",
    "Shun'ichi Fukuzumi",
    "Kiyoshi Fukuhara",
    "Ikuo Nakanishi",
    "Toshihiko Ozawa",
    "Shiro Urano",
    "Haruhiro Okuda",
    "Kentaro Miyazaki",
    "Nobuo Ikota",
    "Yoshihiro Uto",
    "Hitoshi Hori"
  ],
  "revision_id": 2447217691,
  "value": [
    "Q42739028",
    "Q11591785",
    "Q117252006",
    "Q44685820",
    "Q116792644",
    "Q117252007",
    "Q117249650",
    "Q59702826",
    "Q117252010",
    "Q117252011",
    "Q117252012"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q11591785"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 11,
    "new_unique": [
      "Q11591785",
      "Q116792644",
      "Q117249650",
      "Q117252006",
      "Q117252007",
      "Q117252010",
      "Q117252011",
      "Q117252012",
      "Q42739028",
      "Q44685820",
      "Q59702826"
    ],
    "new_value": [
      "Q42739028",
      "Q11591785",
      "Q117252006",
      "Q44685820",
      "Q116792644",
      "Q117252007",
      "Q117249650",
      "Q59702826",
      "Q117252010",
      "Q117252011",
      "Q117252012"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 11,
    "old_unique": [
      "Q116792644",
      "Q117249650",
      "Q117252006",
      "Q117252007",
      "Q117252010",
      "Q117252011",
      "Q117252012",
      "Q42739028",
      "Q44685820",
      "Q59702826",
      "Q67623707"
    ],
    "old_value": [
      "Q42739028",
      "Q67623707",
      "Q117252006",
      "Q44685820",
      "Q116792644",
      "Q117252007",
      "Q117249650",
      "Q59702826",
      "Q117252010",
      "Q117252011",
      "Q117252012"
    ],
    "removed_unique_values": [
      "Q67623707"
    ],
    "value_multiplicity_changes": {
      "Q11591785": {
        "new": 1,
        "old": 0
      },
      "Q67623707": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "researcher",
    "Japanese physical chemist (1950-)",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher"
  ],
  "value_labels_en": [
    "Kei Ohkubo",
    "Shun'ichi Fukuzumi",
    "Kiyoshi Fukuhara",
    "Ikuo Nakanishi",
    "Toshihiko Ozawa",
    "Shiro Urano",
    "Haruhiro Okuda",
    "Kentaro Miyazaki",
    "Nobuo Ikota",
    "Yoshihiro Uto",
    "Hitoshi Hori"
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
    "Q42739028",
    "Q67623707",
    "Q117252006",
    "Q44685820",
    "Q116792644",
    "Q117252007",
    "Q117249650",
    "Q59702826",
    "Q117252010",
    "Q117252011",
    "Q117252012"
  ],
  "value_descriptions_en": [
    "researcher",
    "Japanese physical chemist (1950-)",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher",
    "researcher"
  ],
  "value_labels_en": [
    "Kei Ohkubo",
    "Shun'ichi Fukuzumi",
    "Kiyoshi Fukuhara",
    "Ikuo Nakanishi",
    "Toshihiko Ozawa",
    "Shiro Urano",
    "Haruhiro Okuda",
    "Kentaro Miyazaki",
    "Nobuo Ikota",
    "Yoshihiro Uto",
    "Hitoshi Hori"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 16,
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q67623707"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q42739028",
      "Q67623707",
      "Q117252006",
      "Q44685820",
      "Q116792644",
      "Q117252007",
      "Q117249650",
      "Q59702826",
      "Q117252010",
      "Q117252011",
      "Q117252012"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q42739028",
    "Q11591785",
    "Q117252006",
    "Q44685820",
    "Q116792644",
    "Q117252007",
    "Q117249650",
    "Q59702826",
    "Q117252010",
    "Q117252011",
    "Q117252012"
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
    "description": "scientific article published in May 2003",
    "label": "Efficient radical scavenging ability of artepillin C, a major component of Brazilian propolis, and the mechanism"
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
      "local_ids_count": 16,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q67623707"
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
        "Q42739028",
        "Q67623707",
        "Q117252006",
        "Q44685820",
        "Q116792644",
        "Q117252007",
        "Q117249650",
        "Q59702826",
        "Q117252010",
        "Q117252011",
        "Q117252012"
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

## 033. `repair_Q462843_2437787877`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q69148050"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q69148050"
    ],
    "new_value": [
      "Q69148050"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q83349710"
    ],
    "old_value": [
      "Q83349710"
    ],
    "removed_unique_values": [
      "Q83349710"
    ],
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

## 034. `repair_Q46481_2443676387`

| Field | Value |
|---|---|
| qid | Q46481 |
| property | P1313 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q46481::P1313 |
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
| truth_tokens_preview | ["Q25466564"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Danil Satria",
  "kind": "A_BOX",
  "new_value": [
    "Q25466564"
  ],
  "new_value_descriptions_en": [
    "political position held by the regional leader of West Coast Regency"
  ],
  "new_value_labels_en": [
    "Regent of West Coast"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2443676387,
  "value": [
    "Q25466564"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q25466564"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q25466564"
    ],
    "new_value": [
      "Q25466564"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q25466564": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "political position held by the regional leader of West Coast Regency"
  ],
  "value_labels_en": [
    "Regent of West Coast"
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
  "local_ids_count": 41,
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
    "Q25466564"
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
    "description": "regency in Lampung Province, Indonesia",
    "label": "Pesisir Barat"
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
      "local_ids_count": 41,
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

## 035. `repair_Q47192993_2439464108`

| Field | Value |
|---|---|
| qid | Q47192993 |
| property | P50 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| group_key | ABOX::Q47192993::P50 |
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
| truth_tokens_preview | ["Q25190163", "Q57072590", "Q114404609"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Q25190163",
    "Q57072590",
    "Q114404609"
  ],
  "new_value_descriptions_en": [
    "Indian anthropologist",
    "researcher",
    "researcher"
  ],
  "new_value_labels_en": [
    "Kewal Krishan",
    "Tanuj Kanchan",
    "Michael S Nirenberg"
  ],
  "old_value": [
    "Q42131135",
    "Q57072590",
    "Q114404609"
  ],
  "old_value_descriptions_en": [
    "Indian anthropologist",
    "researcher",
    "researcher"
  ],
  "old_value_labels_en": [
    "Kewal Krishan",
    "Tanuj Kanchan",
    "Michael S Nirenberg"
  ],
  "revision_id": 2439464108,
  "value": [
    "Q25190163",
    "Q57072590",
    "Q114404609"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q25190163"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "Q114404609",
      "Q25190163",
      "Q57072590"
    ],
    "new_value": [
      "Q25190163",
      "Q57072590",
      "Q114404609"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 3,
    "old_unique": [
      "Q114404609",
      "Q42131135",
      "Q57072590"
    ],
    "old_value": [
      "Q42131135",
      "Q57072590",
      "Q114404609"
    ],
    "removed_unique_values": [
      "Q42131135"
    ],
    "value_multiplicity_changes": {
      "Q25190163": {
        "new": 1,
        "old": 0
      },
      "Q42131135": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "Indian anthropologist",
    "researcher",
    "researcher"
  ],
  "value_labels_en": [
    "Kewal Krishan",
    "Tanuj Kanchan",
    "Michael S Nirenberg"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-09T14:51:46",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P50",
  "report_revision_new": 2440068412,
  "report_revision_old": 2439590541,
  "report_violation_type": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398",
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
    "distinct and identifiable entity with agency, capable of performing actions"
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
    "being"
  ],
  "report_violation_type_normalized": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398",
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
    "Q24229398"
  ],
  "report_violation_type_raw": "Value type Q|215627, Q|61002, Q|43229, Q|16334295, Q|95074, Q|5, Q|11660, Q|75855169, Q|107307291, Q|24229398",
  "value": [
    "Q42131135",
    "Q57072590",
    "Q114404609"
  ],
  "value_descriptions_en": [
    "Indian anthropologist",
    "researcher",
    "researcher"
  ],
  "value_labels_en": [
    "Kewal Krishan",
    "Tanuj Kanchan",
    "Michael S Nirenberg"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 10,
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q42131135"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q42131135",
      "Q57072590",
      "Q114404609"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q25190163",
    "Q57072590",
    "Q114404609"
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
    "description": "scientific article published on 11 September 2017",
    "label": "A metric study of insole foot impressions in footwear of identical twins."
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
      "local_ids_count": 10,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q42131135"
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
        "Q42131135",
        "Q57072590",
        "Q114404609"
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

## 036. `repair_Q4941561_2426561999`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q471817"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 34,
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
    "normalized_unique_values_unchanged": false,
    "old_count": 34,
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
    "removed_unique_values": [
      "Q14819288"
    ],
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

## 037. `repair_Q53222835_2447218429`

| Field | Value |
|---|---|
| qid | Q53222835 |
| property | P50 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| group_key | ABOX::Q53222835::P50 |
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
| truth_tokens_preview | ["Q42739028", "Q11591785", "Q117252025"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Q42739028",
    "Q11591785",
    "Q117252025"
  ],
  "new_value_descriptions_en": [
    "researcher",
    "Japanese physical chemist (1950-)",
    "researcher"
  ],
  "new_value_labels_en": [
    "Kei Ohkubo",
    "Shun'ichi Fukuzumi",
    "Kentaro Mase"
  ],
  "old_value": [
    "Q42739028",
    "Q67623707",
    "Q117252025"
  ],
  "old_value_descriptions_en": [
    "researcher",
    "Japanese physical chemist (1950-)",
    "researcher"
  ],
  "old_value_labels_en": [
    "Kei Ohkubo",
    "Shun'ichi Fukuzumi",
    "Kentaro Mase"
  ],
  "revision_id": 2447218429,
  "value": [
    "Q42739028",
    "Q11591785",
    "Q117252025"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q11591785"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "Q11591785",
      "Q117252025",
      "Q42739028"
    ],
    "new_value": [
      "Q42739028",
      "Q11591785",
      "Q117252025"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 3,
    "old_unique": [
      "Q117252025",
      "Q42739028",
      "Q67623707"
    ],
    "old_value": [
      "Q42739028",
      "Q67623707",
      "Q117252025"
    ],
    "removed_unique_values": [
      "Q67623707"
    ],
    "value_multiplicity_changes": {
      "Q11591785": {
        "new": 1,
        "old": 0
      },
      "Q67623707": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "researcher",
    "Japanese physical chemist (1950-)",
    "researcher"
  ],
  "value_labels_en": [
    "Kei Ohkubo",
    "Shun'ichi Fukuzumi",
    "Kentaro Mase"
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
    "Q42739028",
    "Q67623707",
    "Q117252025"
  ],
  "value_descriptions_en": [
    "researcher",
    "Japanese physical chemist (1950-)",
    "researcher"
  ],
  "value_labels_en": [
    "Kei Ohkubo",
    "Shun'ichi Fukuzumi",
    "Kentaro Mase"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 38,
  "matched": false,
  "matches": [
    {
      "independent_of_target_property": false,
      "kind": "id_exact",
      "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
      "token": "Q67623707"
    }
  ],
  "needed": 2,
  "sources_used": [
    "FOCUS_PREREPAIR_TARGET_PROPERTY_QID"
  ],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q42739028",
      "Q67623707",
      "Q117252025"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q42739028",
    "Q11591785",
    "Q117252025"
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
    "description": "scientific article published on 29 May 2015",
    "label": "Selective electrochemical reduction of CO2 to CO with a cobalt chlorin complex adsorbed on multi-walled carbon nanotubes in water"
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
      "local_ids_count": 38,
      "matched": false,
      "matches": [
        {
          "independent_of_target_property": false,
          "kind": "id_exact",
          "source": "FOCUS_PREREPAIR_TARGET_PROPERTY_QID",
          "token": "Q67623707"
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
        "Q42739028",
        "Q67623707",
        "Q117252025"
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

## 038. `repair_Q53306951_2441301223`

| Field | Value |
|---|---|
| qid | Q53306951 |
| property | P735 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| group_key | ABOX::Q53306951::P735 |
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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
  "revision_id": 2441301223,
  "value": [
    "Q3061675"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q3061675"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q3061675"
    ],
    "new_value": [
      "Q3061675"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q15731812"
    ],
    "old_value": [
      "Q15731812"
    ],
    "removed_unique_values": [
      "Q15731812"
    ],
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
  "local_ids_count": 12,
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
    "label": "Ewa Jablonska"
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

## 039. `repair_Q5774674_2439465564`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q13052701"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "Q1261",
      "Q13052701",
      "Q16554"
    ],
    "new_value": [
      "Q1261",
      "Q16554",
      "Q13052701"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 3,
    "old_unique": [
      "Q1261",
      "Q13140165",
      "Q16554"
    ],
    "old_value": [
      "Q1261",
      "Q16554",
      "Q13140165"
    ],
    "removed_unique_values": [
      "Q13140165"
    ],
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

## 040. `repair_Q57881513_2447219940`

| Field | Value |
|---|---|
| qid | Q57881513 |
| property | P50 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q52004125 |
| group_key | ABOX::Q57881513::P50 |
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
| truth_tokens_preview | ["Q57921226", "Q57413081", "Q26972386"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "KrBot",
  "kind": "A_BOX",
  "new_value": [
    "Q57921226",
    "Q57413081",
    "Q26972386"
  ],
  "new_value_descriptions_en": [
    "researcher, ORCID id # 0000-0001-9167-5406",
    "researcher ORCID id 0000-0002-4487-2359",
    "Italian geologist (1957-)"
  ],
  "new_value_labels_en": [
    "Erwan Gueguen",
    "Manuel Fernandez",
    "Carlo Doglioni"
  ],
  "old_value": [
    "Q57921226",
    "Q57413081",
    "Q57902495"
  ],
  "old_value_descriptions_en": [
    "researcher, ORCID id # 0000-0001-9167-5406",
    "researcher ORCID id 0000-0002-4487-2359",
    "Italian geologist (1957-)"
  ],
  "old_value_labels_en": [
    "Erwan Gueguen",
    "Manuel Fernandez",
    "Carlo Doglioni"
  ],
  "revision_id": 2447219940,
  "value": [
    "Q57921226",
    "Q57413081",
    "Q26972386"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q26972386"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 3,
    "new_unique": [
      "Q26972386",
      "Q57413081",
      "Q57921226"
    ],
    "new_value": [
      "Q57921226",
      "Q57413081",
      "Q26972386"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 3,
    "old_unique": [
      "Q57413081",
      "Q57902495",
      "Q57921226"
    ],
    "old_value": [
      "Q57921226",
      "Q57413081",
      "Q57902495"
    ],
    "removed_unique_values": [
      "Q57902495"
    ],
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
    "researcher, ORCID id # 0000-0001-9167-5406",
    "researcher ORCID id 0000-0002-4487-2359",
    "Italian geologist (1957-)"
  ],
  "value_labels_en": [
    "Erwan Gueguen",
    "Manuel Fernandez",
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
    "Q57921226",
    "Q57413081",
    "Q57902495"
  ],
  "value_descriptions_en": [
    "researcher, ORCID id # 0000-0001-9167-5406",
    "researcher ORCID id 0000-0002-4487-2359",
    "Italian geologist (1957-)"
  ],
  "value_labels_en": [
    "Erwan Gueguen",
    "Manuel Fernandez",
    "Carlo Doglioni"
  ]
}
```

### Local Evidence

```json
{
  "found": 1,
  "local_availability_result": false,
  "local_ids_count": 10,
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
      "Q57921226",
      "Q57413081",
      "Q57902495"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q57921226",
    "Q57413081",
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
    "description": "scientific article published in 1997",
    "label": "Lithospheric boudinage in the Western Mediterranean back-arc basin"
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
      "local_ids_count": 10,
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
        "Q57921226",
        "Q57413081",
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

## 041. `repair_Q63051575_2441521243`

| Field | Value |
|---|---|
| qid | Q63051575 |
| property | P2354 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21510865 |
| group_key | ABOX::Q63051575::P2354 |
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
| truth_tokens_preview | ["Q98835845"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "MatSuBot",
  "kind": "A_BOX",
  "new_value": [
    "Q98835845"
  ],
  "new_value_descriptions_en": [
    "Wikimedia list article"
  ],
  "new_value_labels_en": [
    "list of Ministers of Foreign Affairs of Moldova"
  ],
  "old_value": [
    "Q137219240"
  ],
  "old_value_descriptions_en": [
    "Wikimedia list article"
  ],
  "old_value_labels_en": [
    "list of Ministers of Foreign Affairs of Moldova"
  ],
  "revision_id": 2441521243,
  "value": [
    "Q98835845"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q98835845"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q98835845"
    ],
    "new_value": [
      "Q98835845"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q137219240"
    ],
    "old_value": [
      "Q137219240"
    ],
    "removed_unique_values": [
      "Q137219240"
    ],
    "value_multiplicity_changes": {
      "Q137219240": {
        "new": 0,
        "old": 1
      },
      "Q98835845": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "Wikimedia list article"
  ],
  "value_labels_en": [
    "list of Ministers of Foreign Affairs of Moldova"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-14T09:20:43",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2354",
  "report_revision_new": 2442246601,
  "report_revision_old": 2441716423,
  "report_violation_type": "Value type Q|13406463",
  "report_violation_type_descriptions_en": [
    "page of a Wikimedia project with a list of something"
  ],
  "report_violation_type_labels_en": [
    "Wikimedia list article"
  ],
  "report_violation_type_normalized": "Value type Q|13406463",
  "report_violation_type_qids": [
    "Q13406463"
  ],
  "report_violation_type_raw": "Value type Q|13406463",
  "report_violation_types": [
    "Value type Q|13406463",
    "Target required claim P|360"
  ],
  "value": [
    "Q137219240"
  ],
  "value_descriptions_en": [
    "Wikimedia list article"
  ],
  "value_labels_en": [
    "list of Ministers of Foreign Affairs of Moldova"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 12,
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q137219240"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q98835845"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "Wikimedia list related to this subject",
    "label": "has list"
  },
  "qid": {
    "description": "head of diplomacy of Moldova",
    "label": "Minister of Foreign Affairs"
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
        "Q137219240"
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

## 042. `repair_Q75358673_2441905199`

| Field | Value |
|---|---|
| qid | Q75358673 |
| property | P26 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | ABOX::Q75358673::P26 |
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
| truth_tokens_preview | ["Q75358675"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Melderick",
  "kind": "A_BOX",
  "new_value": [
    "Q75358675"
  ],
  "new_value_descriptions_en": [
    "(born 1962)"
  ],
  "new_value_labels_en": [
    "Paulo Ibrahim Mansour"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2441905199,
  "value": [
    "Q75358675"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q75358675"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q75358675"
    ],
    "new_value": [
      "Q75358675"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q75358675": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "(born 1962)"
  ],
  "value_labels_en": [
    "Paulo Ibrahim Mansour"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-15T16:06:03",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P26",
  "report_revision_new": 2442729620,
  "report_revision_old": 2442339590,
  "report_violation_type": "Symmetric",
  "report_violation_type_normalized": "Symmetric",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Symmetric",
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
  "local_ids_count": 11,
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
    "Q75358675"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "the subject has the object as their spouse (husband, wife, partner, etc.). Use \"unmarried partner\" (P451) for non-married companions",
    "label": "spouse"
  },
  "qid": {
    "description": "(born 1971)",
    "label": "Princess Anna Luíza of Orléans-Braganza"
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
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
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
      "local_ids_count": 11,
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

## 043. `repair_Q7628515_2444220154`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q137465724"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 11,
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
    "normalized_unique_values_unchanged": false,
    "old_count": 10,
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
    "removed_unique_values": [],
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

## 044. `repair_Q7734_1702618078`

| Field | Value |
|---|---|
| qid | Q7734 |
| property | P140 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q7734::P140 |
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
| truth_tokens_preview | ["Q3324698"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "UPDATE",
  "author": "Sinequonen",
  "kind": "A_BOX",
  "new_value": [
    "Q3324698"
  ],
  "new_value_descriptions_en": [
    "faith attributed to Moses"
  ],
  "new_value_labels_en": [
    "Mosaic Judaism"
  ],
  "old_value": [
    "Q47280"
  ],
  "old_value_descriptions_en": [
    "category of religions considered as coming from the legacy of Abraham"
  ],
  "old_value_labels_en": [
    "Abrahamic religion"
  ],
  "revision_id": 1702618078,
  "value": [
    "Q3324698"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q3324698"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q3324698"
    ],
    "new_value": [
      "Q3324698"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q47280"
    ],
    "old_value": [
      "Q47280"
    ],
    "removed_unique_values": [
      "Q47280"
    ],
    "value_multiplicity_changes": {
      "Q3324698": {
        "new": 1,
        "old": 0
      },
      "Q47280": {
        "new": 0,
        "old": 1
      }
    }
  },
  "value_descriptions_en": [
    "faith attributed to Moses"
  ],
  "value_labels_en": [
    "Mosaic Judaism"
  ]
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-08-15T10:41:49",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P140",
  "report_revision_new": 1704555119,
  "report_revision_old": 1703751372,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": [
    "Q47280"
  ],
  "value_descriptions_en": [
    "category of religions considered as coming from the legacy of Abraham"
  ],
  "value_labels_en": [
    "Abrahamic religion"
  ]
}
```

### Local Evidence

```json
{
  "found": 0,
  "local_availability_result": false,
  "local_ids_count": 73,
  "matched": false,
  "matches": [],
  "needed": 1,
  "sources_used": [],
  "synthetic_pre_repair": {
    "pre_repair_source": "repair_target.old_value",
    "tokens": [
      "Q47280"
    ],
    "used_pre_repair_value": true
  },
  "truth_tokens": [
    "Q3324698"
  ],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": false
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "religion of a person, organization or religious building, or associated with this subject",
    "label": "religion or worldview"
  },
  "qid": {
    "description": "figure in the Torah",
    "label": "Joshua"
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
    "label_en": "one-of constraint",
    "qid": "Q21510859"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
  },
  {
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
  },
  {
    "label_en": "contemporary constraint",
    "qid": "Q25796498"
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
      "local_ids_count": 73,
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
        "Q47280"
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

## 045. `repair_Q8120_2443681018`

| Field | Value |
|---|---|
| qid | Q8120 |
| property | P1313 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | ABOX::Q8120::P1313 |
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
| truth_tokens_preview | ["Q65213614"] |
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "action": "CREATE",
  "author": "Danil Satria",
  "kind": "A_BOX",
  "new_value": [
    "Q65213614"
  ],
  "new_value_descriptions_en": [
    "Wikimedia list article"
  ],
  "new_value_labels_en": [
    "list of Regents of Musi Banyuasin"
  ],
  "old_value": [
    "MISSING"
  ],
  "revision_id": 2443681018,
  "value": [
    "Q65213614"
  ],
  "value_change_summary": {
    "action": "CREATE",
    "added_unique_values": [
      "Q65213614"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q65213614"
    ],
    "new_value": [
      "Q65213614"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "MISSING"
    ],
    "old_value": [
      "MISSING"
    ],
    "removed_unique_values": [
      "MISSING"
    ],
    "value_multiplicity_changes": {
      "MISSING": {
        "new": 0,
        "old": 1
      },
      "Q65213614": {
        "new": 1,
        "old": 0
      }
    }
  },
  "value_descriptions_en": [
    "Wikimedia list article"
  ],
  "value_labels_en": [
    "list of Regents of Musi Banyuasin"
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
  "local_ids_count": 54,
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
    "Q65213614"
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
    "description": "regency in South Sumatra Province, Indonesia",
    "label": "Musi Banyuasin"
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
      "local_ids_count": 54,
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

## 046. `repair_Q85404_2254631024`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q1971214"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 32,
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
    "normalized_unique_values_unchanged": false,
    "old_count": 31,
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
    "removed_unique_values": [],
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

## 047. `repair_Q90353730_2441305196`

| Field | Value |
|---|---|
| qid | Q90353730 |
| property | P735 |
| track | A_BOX |
| class / subtype / confidence | TypeC / EXTERNAL_BY_ELIMINATION / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_ic_e_elim |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| group_key | ABOX::Q90353730::P735 |
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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
  "revision_id": 2441305196,
  "value": [
    "Q3061675"
  ],
  "value_change_summary": {
    "action": "UPDATE",
    "added_unique_values": [
      "Q3061675"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q3061675"
    ],
    "new_value": [
      "Q3061675"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q15731812"
    ],
    "old_value": [
      "Q15731812"
    ],
    "removed_unique_values": [
      "Q15731812"
    ],
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
  "local_ids_count": 26,
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
    "description": "Polish ophthalmologist and researcher",
    "label": "Ewa Kosior-Jarecka"
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
      "local_ids_count": 26,
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

## 048. `repair_Q95967818_2441306184`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q3061675"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q3061675"
    ],
    "new_value": [
      "Q3061675"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q15731812"
    ],
    "old_value": [
      "Q15731812"
    ],
    "removed_unique_values": [
      "Q15731812"
    ],
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

## 049. `repair_Q97689673_2441306589`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind | id_exact |
| local_match_source | FOCUS_PREREPAIR_TARGET_PROPERTY_QID |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q3061675"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 2,
    "new_unique": [
      "Q20087949",
      "Q3061675"
    ],
    "new_value": [
      "Q20087949",
      "Q3061675"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 2,
    "old_unique": [
      "Q15731812",
      "Q20087949"
    ],
    "old_value": [
      "Q20087949",
      "Q15731812"
    ],
    "removed_unique_values": [
      "Q15731812"
    ],
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

## 050. `repair_Q99870964_2444734107`

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
| decision_branch | external_by_elimination |
| rationale | Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

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
    "action": "UPDATE",
    "added_unique_values": [
      "Q27346385"
    ],
    "deleted_value": [],
    "exact_value_lists_unchanged": false,
    "kind": "A_BOX",
    "new_count": 1,
    "new_unique": [
      "Q27346385"
    ],
    "new_value": [
      "Q27346385"
    ],
    "normalized_unique_values_unchanged": false,
    "old_count": 1,
    "old_unique": [
      "Q99872290"
    ],
    "old_value": [
      "Q99872290"
    ],
    "removed_unique_values": [
      "Q99872290"
    ],
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
