# TBOX_COINCIDENTAL_SCHEMA_CHANGE

Cases: 20

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `reform_Q114850411_P1225_2327498904`

| Field | Value |
|---|---|
| qid | Q114850411 |
| property | P1225 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | TBOX::P1225::2327498904 |
| tbox_revision_key | TBOX::P1225::2327498904 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Tokrkbot",
  "kind": "T_BOX",
  "property_revision_id": 2327498904,
  "property_revision_prev": 2318152286
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-03-20T15:59:41",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1225",
  "report_revision_new": 2327706937,
  "report_revision_old": 2324197202,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": null,
  "value_current_2026": [
    "10472863",
    "10494202"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for the United States National Archives and Records Administration's online catalog",
    "label": "U.S. National Archives Identifier"
  },
  "qid": {
    "description": null,
    "label": "Collection District of Marblehead, Massachusetts"
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
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q19474404",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q125191",
            "Q16930315",
            "Q1700320",
            "Q18149496",
            "Q3331518",
            "Q56523001",
            "Q7336423"
          ]
        },
        {
          "property_id": "P4155",
          "values": [
            "P1810",
            "P1932"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 4,
  "author": "Tokrkbot",
  "before_constraint_count": 4,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "image created by light falling on a light-sensitive surface",
              "id": "Q125191",
              "label_en": "photograph"
            },
            {
              "description_en": "former United States federal agency",
              "id": "Q16930315",
              "label_en": "United States Weather Bureau"
            },
            {
              "description_en": "reservoir in North Carolina and Virginia, United States",
              "id": "Q1700320",
              "label_en": "Kerr Lake"
            },
            {
              "description_en": "historic district in North Carolina",
              "id": "Q18149496",
              "label_en": "Downtown Asheville Historic District"
            },
            {
              "description_en": "former customs service of the United States (1789-2003)",
              "id": "Q3331518",
              "label_en": "United States Customs Service"
            },
            {
              "description_en": "American attorney and U.S. Commissioner of Internal Revenue (1927-2018)",
              "id": "Q56523001",
              "label_en": "Sheldon S. Cohen"
            },
            {
              "description_en": "reservoir in the Strawberry Valley in Wasatch County, Utah, United States that absorbed the former Soldier Creek Reservoir",
              "id": "Q7336423",
              "label_en": "Strawberry Reservoir"
            }
          ],
          "P4155": [
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "([1-9]\\d{0,8})"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            },
            {
              "description_en": "Wikibase entity type for Wikimedia Commons",
              "id": "Q59712033",
              "label_en": "МэдыяІнфа Вікібазы"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "image created by light falling on a light-sensitive surface",
              "id": "Q125191",
              "label_en": "photograph"
            },
            {
              "description_en": "former United States federal agency",
              "id": "Q16930315",
              "label_en": "United States Weather Bureau"
            },
            {
              "description_en": "reservoir in North Carolina and Virginia, United States",
              "id": "Q1700320",
              "label_en": "Kerr Lake"
            },
            {
              "description_en": "historic district in North Carolina",
              "id": "Q18149496",
              "label_en": "Downtown Asheville Historic District"
            },
            {
              "description_en": "former customs service of the United States (1789-2003)",
              "id": "Q3331518",
              "label_en": "United States Customs Service"
            },
            {
              "description_en": "American attorney and U.S. Commissioner of Internal Revenue (1927-2018)",
              "id": "Q56523001",
              "label_en": "Sheldon S. Cohen"
            },
            {
              "description_en": "reservoir in the Strawberry Valley in Wasatch County, Utah, United States that absorbed the former Soldier Creek Reservoir",
              "id": "Q7336423",
              "label_en": "Strawberry Reservoir"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "([1-9]\\d{0,8})"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            },
            {
              "description_en": "Wikibase entity type for Wikimedia Commons",
              "id": "Q59712033",
              "label_en": "МэдыяІнфа Вікібазы"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "1f7576271bb52fa68632087c4ff831d16a3b5e3d",
  "hash_before": "d5d771519de4446dc0f3b34cc1f488947ae99bd8",
  "property_revision_id": 2327498904,
  "property_revision_prev": 2318152286,
  "qualifier_value_changes": [
    {
      "added_values": [
        "P1810",
        "P1932"
      ],
      "constraint_qid": "Q19474404",
      "qualifier_property": "P4155",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q19474404",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q125191",
            "Q16930315",
            "Q1700320",
            "Q18149496",
            "Q3331518",
            "Q56523001",
            "Q7336423"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: exception to constraint: photograph, United States Weather Bureau, Kerr Lake, Downtown Asheville Historic District, United States Customs Service, Sheldon S. Cohen, Strawberry Reservoir; separator: subject named as, object named as",
      "format constraint: format as a regular expression: ([1-9]\\d{0,8}); constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы"
    ],
    "before": [
      "single-value constraint: exception to constraint: photograph, United States Weather Bureau, Kerr Lake, Downtown Asheville Historic District, United States Customs Service, Sheldon S. Cohen, Strawberry Reservoir",
      "format constraint: format as a regular expression: ([1-9]\\d{0,8}); constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q19474404",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Single value"
  }
]
```

---

## 002. `reform_Q114871034_P1225_2327498904`

| Field | Value |
|---|---|
| qid | Q114871034 |
| property | P1225 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | TBOX::P1225::2327498904 |
| tbox_revision_key | TBOX::P1225::2327498904 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Tokrkbot",
  "kind": "T_BOX",
  "property_revision_id": 2327498904,
  "property_revision_prev": 2318152286
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-03-20T15:59:41",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1225",
  "report_revision_new": 2327706937,
  "report_revision_old": 2324197202,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": null,
  "value_current_2026": [
    "10467528",
    "10467725",
    "10467726"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for the United States National Archives and Records Administration's online catalog",
    "label": "U.S. National Archives Identifier"
  },
  "qid": {
    "description": null,
    "label": "Collection District of Maine and New Hampshire"
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
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q19474404",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q125191",
            "Q16930315",
            "Q1700320",
            "Q18149496",
            "Q3331518",
            "Q56523001",
            "Q7336423"
          ]
        },
        {
          "property_id": "P4155",
          "values": [
            "P1810",
            "P1932"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 4,
  "author": "Tokrkbot",
  "before_constraint_count": 4,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "image created by light falling on a light-sensitive surface",
              "id": "Q125191",
              "label_en": "photograph"
            },
            {
              "description_en": "former United States federal agency",
              "id": "Q16930315",
              "label_en": "United States Weather Bureau"
            },
            {
              "description_en": "reservoir in North Carolina and Virginia, United States",
              "id": "Q1700320",
              "label_en": "Kerr Lake"
            },
            {
              "description_en": "historic district in North Carolina",
              "id": "Q18149496",
              "label_en": "Downtown Asheville Historic District"
            },
            {
              "description_en": "former customs service of the United States (1789-2003)",
              "id": "Q3331518",
              "label_en": "United States Customs Service"
            },
            {
              "description_en": "American attorney and U.S. Commissioner of Internal Revenue (1927-2018)",
              "id": "Q56523001",
              "label_en": "Sheldon S. Cohen"
            },
            {
              "description_en": "reservoir in the Strawberry Valley in Wasatch County, Utah, United States that absorbed the former Soldier Creek Reservoir",
              "id": "Q7336423",
              "label_en": "Strawberry Reservoir"
            }
          ],
          "P4155": [
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "([1-9]\\d{0,8})"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            },
            {
              "description_en": "Wikibase entity type for Wikimedia Commons",
              "id": "Q59712033",
              "label_en": "МэдыяІнфа Вікібазы"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "image created by light falling on a light-sensitive surface",
              "id": "Q125191",
              "label_en": "photograph"
            },
            {
              "description_en": "former United States federal agency",
              "id": "Q16930315",
              "label_en": "United States Weather Bureau"
            },
            {
              "description_en": "reservoir in North Carolina and Virginia, United States",
              "id": "Q1700320",
              "label_en": "Kerr Lake"
            },
            {
              "description_en": "historic district in North Carolina",
              "id": "Q18149496",
              "label_en": "Downtown Asheville Historic District"
            },
            {
              "description_en": "former customs service of the United States (1789-2003)",
              "id": "Q3331518",
              "label_en": "United States Customs Service"
            },
            {
              "description_en": "American attorney and U.S. Commissioner of Internal Revenue (1927-2018)",
              "id": "Q56523001",
              "label_en": "Sheldon S. Cohen"
            },
            {
              "description_en": "reservoir in the Strawberry Valley in Wasatch County, Utah, United States that absorbed the former Soldier Creek Reservoir",
              "id": "Q7336423",
              "label_en": "Strawberry Reservoir"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "([1-9]\\d{0,8})"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            },
            {
              "description_en": "Wikibase entity type for Wikimedia Commons",
              "id": "Q59712033",
              "label_en": "МэдыяІнфа Вікібазы"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "1f7576271bb52fa68632087c4ff831d16a3b5e3d",
  "hash_before": "d5d771519de4446dc0f3b34cc1f488947ae99bd8",
  "property_revision_id": 2327498904,
  "property_revision_prev": 2318152286,
  "qualifier_value_changes": [
    {
      "added_values": [
        "P1810",
        "P1932"
      ],
      "constraint_qid": "Q19474404",
      "qualifier_property": "P4155",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q19474404",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q125191",
            "Q16930315",
            "Q1700320",
            "Q18149496",
            "Q3331518",
            "Q56523001",
            "Q7336423"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: exception to constraint: photograph, United States Weather Bureau, Kerr Lake, Downtown Asheville Historic District, United States Customs Service, Sheldon S. Cohen, Strawberry Reservoir; separator: subject named as, object named as",
      "format constraint: format as a regular expression: ([1-9]\\d{0,8}); constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы"
    ],
    "before": [
      "single-value constraint: exception to constraint: photograph, United States Weather Bureau, Kerr Lake, Downtown Asheville Historic District, United States Customs Service, Sheldon S. Cohen, Strawberry Reservoir",
      "format constraint: format as a regular expression: ([1-9]\\d{0,8}); constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q19474404",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Single value"
  }
]
```

---

## 003. `reform_Q12633819_P434_2439609481`

| Field | Value |
|---|---|
| qid | Q12633819 |
| property | P434 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| group_key | TBOX::P434::2439609481 |
| tbox_revision_key | TBOX::P434::2439609481 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Revi C.",
  "kind": "T_BOX",
  "property_revision_id": 2439609481,
  "property_revision_prev": 2404361461
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-13T10:39:10",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P434",
  "report_revision_new": 2441775107,
  "report_revision_old": 2441202272,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "f781ac7d-f745-4d5f-9cf6-0b09e5b0e237"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
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
    "description": "Serbian composer and conductor",
    "label": "Jovan Adamov"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502838",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q1377647",
            "Q15139437"
          ]
        },
        {
          "property_id": "P2306",
          "values": [
            "P966"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 17,
  "author": "Revi C.",
  "before_constraint_count": 17,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "(qualifier) label or alias name to which the claim applies (subject of the statement). To refer to name of the value, use \"applies to name of object\" (P8338)",
              "id": "P5168",
              "label_en": "applies to name of subject"
            },
            {
              "description_en": "alias used by someone (for nickname use P1449)",
              "id": "P742",
              "label_en": "pseudonym"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[0-9a-f]{8}-[0-9a-f]{4}-[4][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ],
          "P2916": [
            {
              "value": "UUIDv4 format string, 36 characters, see [[Q73747105]]@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "qualifier, to be used on external identifier IDs, indicating another Wikidata item is also matched to this ID",
              "id": "P4070",
              "label_en": "identifier shared with"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "Japanese video game developer",
              "id": "Q1377647",
              "label_en": "Team Shanghai Alice"
            },
            {
              "description_en": "South Korean subsidiary of NEXON",
              "id": "Q15139437",
              "label_en": "NEXON Korea"
            }
          ],
          "P2306": [
            {
              "description_en": "identifier for a label in the MusicBrainz open music encyclopedia",
              "id": "P966",
              "label_en": "MusicBrainz label ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "German public broadcaster",
              "id": "Q23565",
              "label_en": "Hessischer Rundfunk"
            }
          ],
          "P2306": [
            {
              "description_en": "Identifier for a place in the MusicBrainz open music encyclopedia",
              "id": "P1004",
              "label_en": "MusicBrainz place ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "page of a Wikimedia project with a list of something",
              "id": "Q13406463",
              "label_en": "Wikimedia list article"
            },
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            },
            {
              "description_en": "use with 'instance of' (P31) for Wikimedia category",
              "id": "Q4167836",
              "label_en": "Wikimedia category"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for an instrument on the music encyclopedia MusicBrainz",
              "id": "P1330",
              "label_en": "MusicBrainz instrument ID"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a series per the MusicBrainz open music encyclopedia",
              "id": "P1407",
              "label_en": "MusicBrainz series ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a work per the MusicBrainz open music encyclopedia",
              "id": "P435",
              "label_en": "MusicBrainz work ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a release group per the MusicBrainz open music encyclopedia (album, single, etc.)",
              "id": "P436",
              "label_en": "MusicBrainz release group ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a recording in the MusicBrainz open music encyclopedia",
              "id": "P4404",
              "label_en": "MusicBrainz recording ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for an area in the MusicBrainz open music database",
              "id": "P982",
              "label_en": "MusicBrainz area ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "set of fictional characters",
              "id": "Q14514600",
              "label_en": "group of fictional characters"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "group of people who perform instrumental and/or vocal music, with the ensemble typically known by a distinct name",
              "id": "Q2088357",
              "label_en": "musical ensemble"
            },
            {
              "description_en": "human who is hypothesized to exist, but where evidence is not conclusive",
              "id": "Q21070568",
              "label_en": "human whose existence is disputed"
            },
            {
              "description_en": "episode-based program (audio or video) distributed asynchronously on the Internet, typically via an RSS feed or downloadable files",
              "id": "Q24634210",
              "label_en": "podcast show"
            },
            {
              "description_en": "identification for a good or service",
              "id": "Q431289",
              "label_en": "brand"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            {
              "description_en": "YouTuber or livestreamer that uses a digital avatar",
              "id": "Q55155641",
              "label_en": "VTuber"
            },
            {
              "description_en": "type of artist credit used by MusicBrainz when the original creator of a musical work is unknown, anonymous, or in the public domain",
              "id": "Q59755569",
              "label_en": "special purpose artist"
            },
            {
              "description_en": "alias used by a band or recording artist for the purpose of recording or releasing recorded music",
              "id": "Q87189273",
              "label_en": "recording alias"
            },
            {
              "description_en": "fictional human or non-human character in a narrative work of art",
              "id": "Q95074",
              "label_en": "character"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type constraints",
              "id": "Q21503252",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "qualifier, to be used on external identifier IDs, indicating another Wikidata item is also matched to this ID",
              "id": "P4070",
              "label_en": "identifier shared with"
            },
            {
              "description_en": "qualifier for alternative name(s), given for a subject in a database entry, or preserved in references (even these are no longer the preferred name)",
              "id": "P4970",
              "label_en": "alternative name"
            },
            {
              "description_en": "(qualifier) label or alias name to which the claim applies (subject of the statement). To refer to name of the value, use \"applies to name of object\" (P8338)",
              "id": "P5168",
              "label_en": "applies to name of subject"
            },
            {
              "description_en": "qualifier to allow the reason to be indicated why a particular statement should be considered preferred",
              "id": "P7452",
              "label_en": "reason for preferred rank"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "(qualifier) label or alias name to which the claim applies (subject of the statement). To refer to name of the value, use \"applies to name of object\" (P8338)",
              "id": "P5168",
              "label_en": "applies to name of subject"
            },
            {
              "description_en": "alias used by someone (for nickname use P1449)",
              "id": "P742",
              "label_en": "pseudonym"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[0-9a-f]{8}-[0-9a-f]{4}-[4][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ],
          "P2916": [
            {
              "value": "UUIDv4 format string, 36 characters, see [[Q73747105]]@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "qualifier, to be used on external identifier IDs, indicating another Wikidata item is also matched to this ID",
              "id": "P4070",
              "label_en": "identifier shared with"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "Japanese video game developer",
              "id": "Q1377647",
              "label_en": "Team Shanghai Alice"
            }
          ],
          "P2306": [
            {
              "description_en": "identifier for a label in the MusicBrainz open music encyclopedia",
              "id": "P966",
              "label_en": "MusicBrainz label ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "German public broadcaster",
              "id": "Q23565",
              "label_en": "Hessischer Rundfunk"
            }
          ],
          "P2306": [
            {
              "description_en": "Identifier for a place in the MusicBrainz open music encyclopedia",
              "id": "P1004",
              "label_en": "MusicBrainz place ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "page of a Wikimedia project with a list of something",
              "id": "Q13406463",
              "label_en": "Wikimedia list article"
            },
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            },
            {
              "description_en": "use with 'instance of' (P31) for Wikimedia category",
              "id": "Q4167836",
              "label_en": "Wikimedia category"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for an instrument on the music encyclopedia MusicBrainz",
              "id": "P1330",
              "label_en": "MusicBrainz instrument ID"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a series per the MusicBrainz open music encyclopedia",
              "id": "P1407",
              "label_en": "MusicBrainz series ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a work per the MusicBrainz open music encyclopedia",
              "id": "P435",
              "label_en": "MusicBrainz work ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a release group per the MusicBrainz open music encyclopedia (album, single, etc.)",
              "id": "P436",
              "label_en": "MusicBrainz release group ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a recording in the MusicBrainz open music encyclopedia",
              "id": "P4404",
              "label_en": "MusicBrainz recording ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for an area in the MusicBrainz open music database",
              "id": "P982",
              "label_en": "MusicBrainz area ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "set of fictional characters",
              "id": "Q14514600",
              "label_en": "group of fictional characters"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "group of people who perform instrumental and/or vocal music, with the ensemble typically known by a distinct name",
              "id": "Q2088357",
              "label_en": "musical ensemble"
            },
            {
              "description_en": "human who is hypothesized to exist, but where evidence is not conclusive",
              "id": "Q21070568",
              "label_en": "human whose existence is disputed"
            },
            {
              "description_en": "episode-based program (audio or video) distributed asynchronously on the Internet, typically via an RSS feed or downloadable files",
              "id": "Q24634210",
              "label_en": "podcast show"
            },
            {
              "description_en": "identification for a good or service",
              "id": "Q431289",
              "label_en": "brand"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            {
              "description_en": "YouTuber or livestreamer that uses a digital avatar",
              "id": "Q55155641",
              "label_en": "VTuber"
            },
            {
              "description_en": "type of artist credit used by MusicBrainz when the original creator of a musical work is unknown, anonymous, or in the public domain",
              "id": "Q59755569",
              "label_en": "special purpose artist"
            },
            {
              "description_en": "alias used by a band or recording artist for the purpose of recording or releasing recorded music",
              "id": "Q87189273",
              "label_en": "recording alias"
            },
            {
              "description_en": "fictional human or non-human character in a narrative work of art",
              "id": "Q95074",
              "label_en": "character"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type constraints",
              "id": "Q21503252",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "qualifier, to be used on external identifier IDs, indicating another Wikidata item is also matched to this ID",
              "id": "P4070",
              "label_en": "identifier shared with"
            },
            {
              "description_en": "qualifier for alternative name(s), given for a subject in a database entry, or preserved in references (even these are no longer the preferred name)",
              "id": "P4970",
              "label_en": "alternative name"
            },
            {
              "description_en": "(qualifier) label or alias name to which the claim applies (subject of the statement). To refer to name of the value, use \"applies to name of object\" (P8338)",
              "id": "P5168",
              "label_en": "applies to name of subject"
            },
            {
              "description_en": "qualifier to allow the reason to be indicated why a particular statement should be considered preferred",
              "id": "P7452",
              "label_en": "reason for preferred rank"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "c05cecd7e93c005a95b408db95e8f0714897bf4c",
  "hash_before": "6aaef4c10a7dad74702c1f25ae4069cca5e13ccd",
  "property_revision_id": 2439609481,
  "property_revision_prev": 2404361461,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q15139437"
      ],
      "constraint_qid": "Q21502838",
      "qualifier_property": "P2303",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502838",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q1377647"
          ]
        },
        {
          "property_id": "P2306",
          "values": [
            "P966"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: separator: subject named as, applies to name of subject, pseudonym",
      "format constraint: format as a regular expression: [0-9a-f]{8}-[0-9a-f]{4}-[4][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}; constraint status: mandatory constraint; syntax clarification: UUIDv4 format string, 36 characters, see [[Q73747105]]@en",
      "distinct-values constraint: separator: identifier shared with",
      "conflicts-with constraint: exception to constraint: Team Shanghai Alice, NEXON Korea; property: MusicBrainz label ID",
      "conflicts-with constraint: exception to constraint: Hessischer Rundfunk; property: MusicBrainz place ID",
      "conflicts-with constraint: item of property constraint: Wikimedia list article, Wikimedia disambiguation page, Wikimedia category; property: instance of; constraint status: mandatory constraint",
      "conflicts-with constraint: property: MusicBrainz instrument ID; constraint status: mandatory constraint",
      "conflicts-with constraint: property: MusicBrainz series ID",
      "conflicts-with constraint: property: MusicBrainz work ID",
      "conflicts-with constraint: property: MusicBrainz release group ID",
      "conflicts-with constraint: property: MusicBrainz recording ID",
      "conflicts-with constraint: property: MusicBrainz area ID",
      "item-requires-statement constraint: property: instance of",
      "subject type constraint: class: group of fictional characters, group of humans, musical ensemble, human whose existence is disputed, podcast show, brand, human, VTuber, special purpose artist, recording alias, character; relation: instance of",
      "allowed qualifiers constraint: property: subject named as, reason for deprecated rank, identifier shared with, alternative name, applies to name of subject, reason for preferred rank",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ],
    "before": [
      "single-value constraint: separator: subject named as, applies to name of subject, pseudonym",
      "format constraint: format as a regular expression: [0-9a-f]{8}-[0-9a-f]{4}-[4][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}; constraint status: mandatory constraint; syntax clarification: UUIDv4 format string, 36 characters, see [[Q73747105]]@en",
      "distinct-values constraint: separator: identifier shared with",
      "conflicts-with constraint: exception to constraint: Team Shanghai Alice; property: MusicBrainz label ID",
      "conflicts-with constraint: exception to constraint: Hessischer Rundfunk; property: MusicBrainz place ID",
      "conflicts-with constraint: item of property constraint: Wikimedia list article, Wikimedia disambiguation page, Wikimedia category; property: instance of; constraint status: mandatory constraint",
      "conflicts-with constraint: property: MusicBrainz instrument ID; constraint status: mandatory constraint",
      "conflicts-with constraint: property: MusicBrainz series ID",
      "conflicts-with constraint: property: MusicBrainz work ID",
      "conflicts-with constraint: property: MusicBrainz release group ID",
      "conflicts-with constraint: property: MusicBrainz recording ID",
      "conflicts-with constraint: property: MusicBrainz area ID",
      "item-requires-statement constraint: property: instance of",
      "subject type constraint: class: group of fictional characters, group of humans, musical ensemble, human whose existence is disputed, podcast show, brand, human, VTuber, special purpose artist, recording alias, character; relation: instance of",
      "allowed qualifiers constraint: property: subject named as, reason for deprecated rank, identifier shared with, alternative name, applies to name of subject, reason for preferred rank",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q21502410",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Unique value"
  }
]
```

---

## 004. `reform_Q12738_P269_2445523281`

| Field | Value |
|---|---|
| qid | Q12738 |
| property | P269 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| group_key | TBOX::P269::2445523281 |
| tbox_revision_key | TBOX::P269::2445523281 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Thomas Kerboul (BGE)",
  "kind": "T_BOX",
  "property_revision_id": 2445523281,
  "property_revision_prev": 2445451375
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-24T12:51:30",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P269",
  "report_revision_new": 2446541658,
  "report_revision_old": 2446069538,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "027401448",
    "161195717"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for authority control in the French collaborative library catalog (see also P1025). Format: 8 digits followed by a digit or \"X\"",
    "label": "IdRef ID"
  },
  "qid": {
    "description": "canton of Switzerland",
    "label": "Canton of Neuchâtel"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502410",
      "qualifiers": [
        {
          "property_id": "P4155",
          "values": [
            "P4070"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 6,
  "author": "Thomas Kerboul (BGE)",
  "before_constraint_count": 6,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "qualifier for alternative name(s), given for a subject in a database entry, or preserved in references (even these are no longer the preferred name)",
              "id": "P4970",
              "label_en": "alternative name"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "([0-9]{8}[\\dX]|)"
            }
          ],
          "P2916": [
            {
              "value": "numeric string, 8 digits, suffixed by X or another digit@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "qualifier, to be used on external identifier IDs, indicating another Wikidata item is also matched to this ID",
              "id": "P4070",
              "label_en": "identifier shared with"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
              "id": "Q3331189",
              "label_en": "version, edition or translation"
            },
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "qualifier for alternative name(s), given for a subject in a database entry, or preserved in references (even these are no longer the preferred name)",
              "id": "P4970",
              "label_en": "alternative name"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "([0-9]{8}[\\dX]|)"
            }
          ],
          "P2916": [
            {
              "value": "numeric string, 8 digits, suffixed by X or another digit@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "canton of Switzerland",
              "id": "Q12738",
              "label_en": "Canton of Neuchâtel"
            },
            {
              "description_en": "country in Eastern Europe and Northern Asia",
              "id": "Q159",
              "label_en": "Russia"
            },
            {
              "description_en": "constituent republic of the Soviet Union (1922–1991)",
              "id": "Q2184",
              "label_en": "Russian Soviet Federative Socialist Republic"
            },
            {
              "description_en": "scientific institution of the Soviet Union (1925–1991)",
              "id": "Q2370801",
              "label_en": "Academy of Sciences of the USSR"
            },
            {
              "description_en": "state in western Europe (1034–1848)",
              "id": "Q3137802",
              "label_en": "Principality of Neuchâtel"
            },
            {
              "description_en": "former empire in Eurasia and North America (1721–1917)",
              "id": "Q34266",
              "label_en": "Russian Empire"
            },
            {
              "description_en": "historical academy (1724–1917)",
              "id": "Q4345832",
              "label_en": "Saint Petersburg Academy of Sciences"
            }
          ],
          "P4155": [
            {
              "description_en": "qualifier, to be used on external identifier IDs, indicating another Wikidata item is also matched to this ID",
              "id": "P4070",
              "label_en": "identifier shared with"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
              "id": "Q3331189",
              "label_en": "version, edition or translation"
            },
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "885bab3ddcf610be0c045df2830bd44b8d0037de",
  "hash_before": "7001e5b91ee685cbf6467bb34973749c2e5258df",
  "property_revision_id": 2445523281,
  "property_revision_prev": 2445451375,
  "qualifier_value_changes": [
    {
      "added_values": [],
      "constraint_qid": "Q21502410",
      "qualifier_property": "P2303",
      "removed_values": [
        "Q12738",
        "Q159",
        "Q2184",
        "Q2370801",
        "Q3137802",
        "Q34266",
        "Q4345832"
      ],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502410",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q12738",
            "Q159",
            "Q2184",
            "Q2370801",
            "Q3137802",
            "Q34266",
            "Q4345832"
          ]
        },
        {
          "property_id": "P4155",
          "values": [
            "P4070"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: separator: subject named as, object of statement has role, alternative name",
      "format constraint: format as a regular expression: ([0-9]{8}[\\dX]|); syntax clarification: numeric string, 8 digits, suffixed by X or another digit@en",
      "distinct-values constraint: separator: identifier shared with",
      "conflicts-with constraint: item of property constraint: version, edition or translation, Wikimedia disambiguation page; property: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ],
    "before": [
      "single-value constraint: separator: subject named as, object of statement has role, alternative name",
      "format constraint: format as a regular expression: ([0-9]{8}[\\dX]|); syntax clarification: numeric string, 8 digits, suffixed by X or another digit@en",
      "distinct-values constraint: exception to constraint: Canton of Neuchâtel, Russia, Russian Soviet Federative Socialist Republic, Academy of Sciences of the USSR, Principality of Neuchâtel, Russian Empire, Saint Petersburg Academy of Sciences; separator: identifier shared with",
      "conflicts-with constraint: item of property constraint: version, edition or translation, Wikimedia disambiguation page; property: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q21502410",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Unique value"
  }
]
```

---

## 005. `reform_Q131431598_P1225_2327498904`

| Field | Value |
|---|---|
| qid | Q131431598 |
| property | P1225 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | TBOX::P1225::2327498904 |
| tbox_revision_key | TBOX::P1225::2327498904 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Tokrkbot",
  "kind": "T_BOX",
  "property_revision_id": 2327498904,
  "property_revision_prev": 2318152286
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-03-20T15:59:41",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1225",
  "report_revision_new": 2327706937,
  "report_revision_old": 2324197202,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": null,
  "value_current_2026": [
    "10515473",
    "10515472"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for the United States National Archives and Records Administration's online catalog",
    "label": "U.S. National Archives Identifier"
  },
  "qid": {
    "description": "former agency in the United States Department of Agriculture and Department of Commerce",
    "label": "Bureau of Public Roads"
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
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q19474404",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q125191",
            "Q16930315",
            "Q1700320",
            "Q18149496",
            "Q3331518",
            "Q56523001",
            "Q7336423"
          ]
        },
        {
          "property_id": "P4155",
          "values": [
            "P1810",
            "P1932"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 4,
  "author": "Tokrkbot",
  "before_constraint_count": 4,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "image created by light falling on a light-sensitive surface",
              "id": "Q125191",
              "label_en": "photograph"
            },
            {
              "description_en": "former United States federal agency",
              "id": "Q16930315",
              "label_en": "United States Weather Bureau"
            },
            {
              "description_en": "reservoir in North Carolina and Virginia, United States",
              "id": "Q1700320",
              "label_en": "Kerr Lake"
            },
            {
              "description_en": "historic district in North Carolina",
              "id": "Q18149496",
              "label_en": "Downtown Asheville Historic District"
            },
            {
              "description_en": "former customs service of the United States (1789-2003)",
              "id": "Q3331518",
              "label_en": "United States Customs Service"
            },
            {
              "description_en": "American attorney and U.S. Commissioner of Internal Revenue (1927-2018)",
              "id": "Q56523001",
              "label_en": "Sheldon S. Cohen"
            },
            {
              "description_en": "reservoir in the Strawberry Valley in Wasatch County, Utah, United States that absorbed the former Soldier Creek Reservoir",
              "id": "Q7336423",
              "label_en": "Strawberry Reservoir"
            }
          ],
          "P4155": [
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "([1-9]\\d{0,8})"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            },
            {
              "description_en": "Wikibase entity type for Wikimedia Commons",
              "id": "Q59712033",
              "label_en": "МэдыяІнфа Вікібазы"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "image created by light falling on a light-sensitive surface",
              "id": "Q125191",
              "label_en": "photograph"
            },
            {
              "description_en": "former United States federal agency",
              "id": "Q16930315",
              "label_en": "United States Weather Bureau"
            },
            {
              "description_en": "reservoir in North Carolina and Virginia, United States",
              "id": "Q1700320",
              "label_en": "Kerr Lake"
            },
            {
              "description_en": "historic district in North Carolina",
              "id": "Q18149496",
              "label_en": "Downtown Asheville Historic District"
            },
            {
              "description_en": "former customs service of the United States (1789-2003)",
              "id": "Q3331518",
              "label_en": "United States Customs Service"
            },
            {
              "description_en": "American attorney and U.S. Commissioner of Internal Revenue (1927-2018)",
              "id": "Q56523001",
              "label_en": "Sheldon S. Cohen"
            },
            {
              "description_en": "reservoir in the Strawberry Valley in Wasatch County, Utah, United States that absorbed the former Soldier Creek Reservoir",
              "id": "Q7336423",
              "label_en": "Strawberry Reservoir"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "([1-9]\\d{0,8})"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            },
            {
              "description_en": "Wikibase entity type for Wikimedia Commons",
              "id": "Q59712033",
              "label_en": "МэдыяІнфа Вікібазы"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "1f7576271bb52fa68632087c4ff831d16a3b5e3d",
  "hash_before": "d5d771519de4446dc0f3b34cc1f488947ae99bd8",
  "property_revision_id": 2327498904,
  "property_revision_prev": 2318152286,
  "qualifier_value_changes": [
    {
      "added_values": [
        "P1810",
        "P1932"
      ],
      "constraint_qid": "Q19474404",
      "qualifier_property": "P4155",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q19474404",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q125191",
            "Q16930315",
            "Q1700320",
            "Q18149496",
            "Q3331518",
            "Q56523001",
            "Q7336423"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: exception to constraint: photograph, United States Weather Bureau, Kerr Lake, Downtown Asheville Historic District, United States Customs Service, Sheldon S. Cohen, Strawberry Reservoir; separator: subject named as, object named as",
      "format constraint: format as a regular expression: ([1-9]\\d{0,8}); constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы"
    ],
    "before": [
      "single-value constraint: exception to constraint: photograph, United States Weather Bureau, Kerr Lake, Downtown Asheville Historic District, United States Customs Service, Sheldon S. Cohen, Strawberry Reservoir",
      "format constraint: format as a regular expression: ([1-9]\\d{0,8}); constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q19474404",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Single value"
  }
]
```

---

## 006. `reform_Q136796116_P856_2446852358`

| Field | Value |
|---|---|
| qid | Q136796116 |
| property | P856 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | TBOX::P856::2446852358 |
| tbox_revision_key | TBOX::P856::2446852358 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Clemens Dulcis",
  "kind": "T_BOX",
  "property_revision_id": 2446852358,
  "property_revision_prev": 2443941926
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-25T18:00:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P856",
  "report_revision_new": 2447046423,
  "report_revision_old": 2446480257,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "http://dx.doi.org/10.1101/2022.03.25.485874"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "URL of the official page of an item (current or former). Usage: If a listed URL no longer points to the official website, do not remove it, but see the \"Hijacked or dead websites\" section of the Talk page",
    "label": "official website"
  },
  "qid": {
    "description": null,
    "label": "RNA-targeting CRISPR-Cas13 Provides Broad-spectrum Phage Immunity"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21510851",
      "qualifiers": [
        {
          "property_id": "P2306",
          "values": [
            "P1001",
            "P1019",
            "P1065",
            "P108",
            "P123",
            "P12506",
            "P126",
            "P127",
            "P13044",
            "P1319",
            "P1326",
            "P13337",
            "P1343",
            "P13597",
            "P137",
            "P13768",
            "P138",
            "P1433",
            "P1476",
            "P1480",
            "P1534",
            "P1535",
            "P1552",
            "P1680",
            "... omitted 46 items"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 12,
  "author": "Clemens Dulcis",
  "before_constraint_count": 12,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(?i)((?!\\b(://web\\.archive\\.org/)).)*"
            }
          ],
          "P2303": [
            {
              "description_en": "digital archive founded by the Internet Archive",
              "id": "Q648266",
              "label_en": "Wayback Machine"
            }
          ],
          "P2916": [
            {
              "value": "Die Adresse der Webseite sollte nicht den Ausdruck 'web.archive.org' enthalten.@de"
            },
            {
              "value": "a string not including 'web.archive.org'@en"
            }
          ],
          "P6607": [
            {
              "value": "Add archive link with 'archive URL' (P1065) and qualify former official website with 'end time' (P582) instead.@en"
            },
            {
              "value": "Ändere bitte die Aussage zur ehemaligen offiziellen Webseite und füge den Archivlink über die Eigentschaft P1065 als Qualifikator hinzu.@de"
            },
            {
              "value": "请不要包含“web.archive.org”，请改为正常添加原始链接，将存档链接添加为“存档URL”（P1065）并加限定符“结束日期”（P582）至前官方网站@zh-cn"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(https?|ftps?)://\\S+"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "^((?!google\\.com/search\\?).)*$"
            }
          ],
          "P2303": [
            {
              "description_en": "feature of Google Search",
              "id": "Q135474941",
              "label_en": "AI Mode"
            }
          ],
          "P2916": [
            {
              "value": "Die Adresse der Webseite sollte nicht den Ausdruck 'google.com/search' enthalten.@de"
            },
            {
              "value": "a string not including 'google.com/search'@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "^(?!https?://www\\.$).+"
            }
          ],
          "P2916": [
            {
              "value": "Die Adresse enthält keinen Domainnamen@de"
            },
            {
              "value": "empty value not including a domain name@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "vocabulary encoding schemes for specified RDA properties",
              "id": "Q104417848",
              "label_en": "RDA value vocabularies"
            },
            {
              "description_en": null,
              "id": "Q104533647",
              "label_en": "RDA/ONIX framework value vocabularies"
            },
            {
              "description_en": "thesaurus of terms relating to Australian Aboriginal and Torres Strait Islander studies",
              "id": "Q105536068",
              "label_en": "AIATSIS Subject Thesaurus"
            },
            {
              "description_en": "thesaurus of Australian place names, using the Indigenous place name first wherever possible",
              "id": "Q105550536",
              "label_en": "AIATSIS Place Thesaurus"
            },
            {
              "description_en": "website of the Washington State Parks agency",
              "id": "Q106426688",
              "label_en": "Washington State Parks website"
            },
            {
              "description_en": "official website of ABBA band",
              "id": "Q108001817",
              "label_en": "ABBAsite.com"
            },
            {
              "description_en": "French political party",
              "id": "Q111225168",
              "label_en": "Gauche démocratique et sociale"
            },
            {
              "description_en": "website of the Cornell Lab of Ornithology Macaulay Library",
              "id": "Q111982999",
              "label_en": "Macaulay Library"
            },
            {
              "description_en": "University library in Bologna, Italy",
              "id": "Q112121329",
              "label_en": "Navile's University Library. Section of Chemistry \"Giacomo Ciamician\""
            },
            {
              "description_en": "University library in Bologna, Italy",
              "id": "Q112126763",
              "label_en": "Navile's University Library. Section of the Department of Pharmacy and Biotechnology"
            },
            {
              "description_en": "fictional character by miHoYo",
              "id": "Q113117660",
              "label_en": "Lumi"
            },
            {
              "description_en": "executive agency of the state of Delaware",
              "id": "Q113133561",
              "label_en": "Delaware Department of State"
            },
            {
              "description_en": "American textile artist and academic (born 1971)",
              "id": "Q113293536",
              "label_en": "Rowland Ricketts"
            },
            {
              "description_en": "Japanese American weaver who crafts traditional narrow-width yardage for kimono and obi using historical kasuri (ikat) techniques",
              "id": "Q113297824",
              "label_en": "Chinami Ricketts"
            },
            {
              "description_en": "Altiusrt website for Hockey New Zealand",
              "id": "Q113634129",
              "label_en": "Hockey New Zealand: Altiusrt"
            },
            {
              "description_en": "website of the Maryland Biodiversity Project",
              "id": "Q113634693",
              "label_en": "Maryland Biodiversity Project"
            },
            {
              "description_en": "website of the Nebraska Invasive Species Program",
              "id": "Q113685547",
              "label_en": "Nebraska Invasive Species Program"
            },
            {
              "description_en": "program affiliated with the University of Nebraska–Lincoln",
              "id": "Q113685550",
              "label_en": "Nebraska Invasive Species Program"
            },
            {
              "description_en": "deutsches Unternehmen mit Sitz Nürburg und Betreiber des Nürburgrings",
              "id": "Q114458607",
              "label_en": "Nürburgring 1927 GmbH & Co. KG"
            },
            {
              "description_en": "website of the Australian Bureau of Meteorology",
              "id": "Q114882717",
              "label_en": "Bureau of Meteorology"
            },
            {
              "description_en": "website",
              "id": "Q115336397",
              "label_en": "BestWestern.com"
            },
            {
              "description_en": "travel website",
              "id": "Q115384119",
              "label_en": "TravelWeekly.com"
            },
            {
              "description_en": "website of the publication Livres Hebdo",
              "id": "Q115443051",
              "label_en": "livreshebdo.fr"
            },
            {
              "description_en": "official website of French weekly news magazine Le Point",
              "id": "Q115553783",
              "label_en": "lePoint.fr"
            },
            "... omitted 88 items"
          ],
          "P4155": [
            {
              "description_en": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
              "id": "P1476",
              "label_en": "title"
            },
            {
              "description_en": "qualifier to specify the item that this property is shared with",
              "id": "P1706",
              "label_en": "together with"
            },
            {
              "description_en": "role held by the item the statement appears on (\"subject\") in the context of that statement. For the role of the statement object/value, use P3831 (\"object has role\"). For acting roles, use P453 (\"character role\"). For persons, use P39.",
              "id": "P2868",
              "label_en": "subject has role"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
            },
            {
              "description_en": "paragraph, or other kind of special indication to find information on a page or on a document (legal texts etc.)",
              "id": "P958",
              "label_en": "section"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "Wikimedia list article",
              "id": "Q11548107",
              "label_en": "list of natural events named by the Japan Meteorological Agency"
            }
          ],
          "P2305": [
            {
              "description_en": "type of page in the Wikimedia system. Use with P31 'instance of' for template pages",
              "id": "Q11266439",
              "label_en": "Wikimedia template"
            },
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            },
            {
              "description_en": "use with 'instance of' (P31) for Wikimedia category",
              "id": "Q4167836",
              "label_en": "Wikimedia category"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "news feed (RSS, Atom, etc.) of this person/organisation/project",
              "id": "P1019",
              "label_en": "web feed URL"
            },
            {
              "description_en": "URL to the archived web page specified with URL property",
              "id": "P1065",
              "label_en": "archive URL"
            },
            {
              "description_en": "person or organization for which the subject works or worked",
              "id": "P108",
              "label_en": "employer"
            },
            {
              "description_en": "organization or person responsible for publishing a work, such as a book, periodical, printed music, podcast, game or software",
              "id": "P123",
              "label_en": "publisher"
            },
            {
              "description_en": "latest date beyond which the statement could no longer be true",
              "id": "P12506",
              "label_en": "latest end date"
            },
            {
              "description_en": "person or organization in charge of keeping the subject (for instance an infrastructure) in functioning order",
              "id": "P126",
              "label_en": "maintained by"
            },
            {
              "description_en": "owner of the subject",
              "id": "P127",
              "label_en": "owned by"
            },
            {
              "description_en": "subject or main statement value is a characteristic, quality, property, or state of this object (use a more specific property where possible)",
              "id": "P13044",
              "label_en": "characteristic of"
            },
            {
              "description_en": "earliest date at which an event could have happened. Use as qualifier for other date properties",
              "id": "P1319",
              "label_en": "earliest date"
            },
            {
              "description_en": "latest possible time that something could have occurred. Use as qualifier for other date properties",
              "id": "P1326",
              "label_en": "latest date"
            },
            {
              "description_en": "registered internet domain name that identifies this entity",
              "id": "P13337",
              "label_en": "domain name"
            },
            {
              "description_en": "work where this item is described, in statistical contexts, a methodological note describing the data",
              "id": "P1343",
              "label_en": "described by source"
            },
            {
              "description_en": "identifier of an article on the Polish Minecraft Wiki",
              "id": "P13597",
              "label_en": "MCW-PL article ID"
            },
            {
              "description_en": "person, profession, organization or entity that operates the equipment, facility, or service",
              "id": "P137",
              "label_en": "operator"
            },
            {
              "description_en": "physical design paradigm this object is aligned with",
              "id": "P13768",
              "label_en": "form factor"
            },
            {
              "description_en": "entity or event that inspired the subject's name, or namesake (in at least one language). Qualifier \"applies to name\" (P5168) can be used to indicate which one",
              "id": "P138",
              "label_en": "named after"
            },
            {
              "description_en": "larger work that a given work was published in, like a journal, a website, a collection, a book or a music album",
              "id": "P1433",
              "label_en": "published in"
            },
            {
              "description_en": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
              "id": "P1476",
              "label_en": "title"
            },
            {
              "description_en": "qualification of the truth or accuracy of a source: circa (Q5727902), near (Q21818619), presumably (Q18122778), etc.",
              "id": "P1480",
              "label_en": "sourcing circumstances"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "item or concept that makes use of the subject (use sub-properties when appropriate)",
              "id": "P1535",
              "label_en": "used by"
            },
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
            {
              "description_en": "for works, when the title is followed by a subtitle",
              "id": "P1680",
              "label_en": "subtitle"
            },
            "... omitted 46 items"
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the listed qualifier has to be used",
          "id": "Q21510856",
          "label_en": "required qualifier constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "organization or person responsible for publishing a work, such as a book, periodical, printed music, podcast, game or software",
              "id": "P123",
              "label_en": "publisher"
            },
            {
              "description_en": "owner of the subject",
              "id": "P127",
              "label_en": "owned by"
            },
            {
              "description_en": "person, profession, organization or entity that operates the equipment, facility, or service",
              "id": "P137",
              "label_en": "operator"
            },
            {
              "description_en": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
              "id": "P1476",
              "label_en": "title"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "intended audience or user of this work, product, object, or event",
              "id": "P2360",
              "label_en": "intended public"
            },
            {
              "description_en": "role held by the item the statement appears on (\"subject\") in the context of that statement. For the role of the statement object/value, use P3831 (\"object has role\"). For acting roles, use P453 (\"character role\"). For persons, use P39.",
              "id": "P2868",
              "label_en": "subject has role"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "contents, item or substance located within this wrapping item, but not part of this receptacle or container; does not describe the ingredients or components of this container",
              "id": "P4330",
              "label_en": "contains"
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
            },
            {
              "description_en": "time an entity begins to exist or a statement starts being valid",
              "id": "P580",
              "label_en": "start time"
            },
            {
              "description_en": "moment when an entity ceases to exist and a statement stops being entirely valid or no longer be true",
              "id": "P582",
              "label_en": "end time"
            },
            {
              "description_en": "date something took place, existed or a statement was true; for providing time use the \"refine date\" property (P4241)",
              "id": "P585",
              "label_en": "point in time"
            },
            {
              "description_en": "primary topic of a work or act of communication",
              "id": "P921",
              "label_en": "main subject"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            }
          ],
          "P6607": [
            {
              "value": "For qualifiers use URL (P2699), for references use reference URL (P854)@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(?i)((?!\\b(://web\\.archive\\.org/)).)*"
            }
          ],
          "P2303": [
            {
              "description_en": "digital archive founded by the Internet Archive",
              "id": "Q648266",
              "label_en": "Wayback Machine"
            }
          ],
          "P2916": [
            {
              "value": "Die Adresse der Webseite sollte nicht den Ausdruck 'web.archive.org' enthalten.@de"
            },
            {
              "value": "a string not including 'web.archive.org'@en"
            }
          ],
          "P6607": [
            {
              "value": "Add archive link with 'archive URL' (P1065) and qualify former official website with 'end time' (P582) instead.@en"
            },
            {
              "value": "Ändere bitte die Aussage zur ehemaligen offiziellen Webseite und füge den Archivlink über die Eigentschaft P1065 als Qualifikator hinzu.@de"
            },
            {
              "value": "请不要包含“web.archive.org”，请改为正常添加原始链接，将存档链接添加为“存档URL”（P1065）并加限定符“结束日期”（P582）至前官方网站@zh-cn"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(https?|ftps?)://\\S+"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "^((?!google\\.com/search\\?).)*$"
            }
          ],
          "P2303": [
            {
              "description_en": "feature of Google Search",
              "id": "Q135474941",
              "label_en": "AI Mode"
            }
          ],
          "P2916": [
            {
              "value": "Die Adresse der Webseite sollte nicht den Ausdruck 'google.com/search' enthalten.@de"
            },
            {
              "value": "a string not including 'google.com/search'@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "^(?!https?://www\\.$).+"
            }
          ],
          "P2916": [
            {
              "value": "Die Adresse enthält keinen Domainnamen@de"
            },
            {
              "value": "empty value not including a domain name@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "vocabulary encoding schemes for specified RDA properties",
              "id": "Q104417848",
              "label_en": "RDA value vocabularies"
            },
            {
              "description_en": null,
              "id": "Q104533647",
              "label_en": "RDA/ONIX framework value vocabularies"
            },
            {
              "description_en": "thesaurus of terms relating to Australian Aboriginal and Torres Strait Islander studies",
              "id": "Q105536068",
              "label_en": "AIATSIS Subject Thesaurus"
            },
            {
              "description_en": "thesaurus of Australian place names, using the Indigenous place name first wherever possible",
              "id": "Q105550536",
              "label_en": "AIATSIS Place Thesaurus"
            },
            {
              "description_en": "website of the Washington State Parks agency",
              "id": "Q106426688",
              "label_en": "Washington State Parks website"
            },
            {
              "description_en": "official website of ABBA band",
              "id": "Q108001817",
              "label_en": "ABBAsite.com"
            },
            {
              "description_en": "French political party",
              "id": "Q111225168",
              "label_en": "Gauche démocratique et sociale"
            },
            {
              "description_en": "website of the Cornell Lab of Ornithology Macaulay Library",
              "id": "Q111982999",
              "label_en": "Macaulay Library"
            },
            {
              "description_en": "University library in Bologna, Italy",
              "id": "Q112121329",
              "label_en": "Navile's University Library. Section of Chemistry \"Giacomo Ciamician\""
            },
            {
              "description_en": "University library in Bologna, Italy",
              "id": "Q112126763",
              "label_en": "Navile's University Library. Section of the Department of Pharmacy and Biotechnology"
            },
            {
              "description_en": "fictional character by miHoYo",
              "id": "Q113117660",
              "label_en": "Lumi"
            },
            {
              "description_en": "executive agency of the state of Delaware",
              "id": "Q113133561",
              "label_en": "Delaware Department of State"
            },
            {
              "description_en": "American textile artist and academic (born 1971)",
              "id": "Q113293536",
              "label_en": "Rowland Ricketts"
            },
            {
              "description_en": "Japanese American weaver who crafts traditional narrow-width yardage for kimono and obi using historical kasuri (ikat) techniques",
              "id": "Q113297824",
              "label_en": "Chinami Ricketts"
            },
            {
              "description_en": "Altiusrt website for Hockey New Zealand",
              "id": "Q113634129",
              "label_en": "Hockey New Zealand: Altiusrt"
            },
            {
              "description_en": "website of the Maryland Biodiversity Project",
              "id": "Q113634693",
              "label_en": "Maryland Biodiversity Project"
            },
            {
              "description_en": "website of the Nebraska Invasive Species Program",
              "id": "Q113685547",
              "label_en": "Nebraska Invasive Species Program"
            },
            {
              "description_en": "program affiliated with the University of Nebraska–Lincoln",
              "id": "Q113685550",
              "label_en": "Nebraska Invasive Species Program"
            },
            {
              "description_en": "deutsches Unternehmen mit Sitz Nürburg und Betreiber des Nürburgrings",
              "id": "Q114458607",
              "label_en": "Nürburgring 1927 GmbH & Co. KG"
            },
            {
              "description_en": "website of the Australian Bureau of Meteorology",
              "id": "Q114882717",
              "label_en": "Bureau of Meteorology"
            },
            {
              "description_en": "website",
              "id": "Q115336397",
              "label_en": "BestWestern.com"
            },
            {
              "description_en": "travel website",
              "id": "Q115384119",
              "label_en": "TravelWeekly.com"
            },
            {
              "description_en": "website of the publication Livres Hebdo",
              "id": "Q115443051",
              "label_en": "livreshebdo.fr"
            },
            {
              "description_en": "official website of French weekly news magazine Le Point",
              "id": "Q115553783",
              "label_en": "lePoint.fr"
            },
            "... omitted 88 items"
          ],
          "P4155": [
            {
              "description_en": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
              "id": "P1476",
              "label_en": "title"
            },
            {
              "description_en": "qualifier to specify the item that this property is shared with",
              "id": "P1706",
              "label_en": "together with"
            },
            {
              "description_en": "role held by the item the statement appears on (\"subject\") in the context of that statement. For the role of the statement object/value, use P3831 (\"object has role\"). For acting roles, use P453 (\"character role\"). For persons, use P39.",
              "id": "P2868",
              "label_en": "subject has role"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
            },
            {
              "description_en": "paragraph, or other kind of special indication to find information on a page or on a document (legal texts etc.)",
              "id": "P958",
              "label_en": "section"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "Wikimedia list article",
              "id": "Q11548107",
              "label_en": "list of natural events named by the Japan Meteorological Agency"
            }
          ],
          "P2305": [
            {
              "description_en": "type of page in the Wikimedia system. Use with P31 'instance of' for template pages",
              "id": "Q11266439",
              "label_en": "Wikimedia template"
            },
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            },
            {
              "description_en": "use with 'instance of' (P31) for Wikimedia category",
              "id": "Q4167836",
              "label_en": "Wikimedia category"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "news feed (RSS, Atom, etc.) of this person/organisation/project",
              "id": "P1019",
              "label_en": "web feed URL"
            },
            {
              "description_en": "URL to the archived web page specified with URL property",
              "id": "P1065",
              "label_en": "archive URL"
            },
            {
              "description_en": "person or organization for which the subject works or worked",
              "id": "P108",
              "label_en": "employer"
            },
            {
              "description_en": "organization or person responsible for publishing a work, such as a book, periodical, printed music, podcast, game or software",
              "id": "P123",
              "label_en": "publisher"
            },
            {
              "description_en": "latest date beyond which the statement could no longer be true",
              "id": "P12506",
              "label_en": "latest end date"
            },
            {
              "description_en": "person or organization in charge of keeping the subject (for instance an infrastructure) in functioning order",
              "id": "P126",
              "label_en": "maintained by"
            },
            {
              "description_en": "owner of the subject",
              "id": "P127",
              "label_en": "owned by"
            },
            {
              "description_en": "subject or main statement value is a characteristic, quality, property, or state of this object (use a more specific property where possible)",
              "id": "P13044",
              "label_en": "characteristic of"
            },
            {
              "description_en": "earliest date at which an event could have happened. Use as qualifier for other date properties",
              "id": "P1319",
              "label_en": "earliest date"
            },
            {
              "description_en": "latest possible time that something could have occurred. Use as qualifier for other date properties",
              "id": "P1326",
              "label_en": "latest date"
            },
            {
              "description_en": "registered internet domain name that identifies this entity",
              "id": "P13337",
              "label_en": "domain name"
            },
            {
              "description_en": "work where this item is described, in statistical contexts, a methodological note describing the data",
              "id": "P1343",
              "label_en": "described by source"
            },
            {
              "description_en": "identifier of an article on the Polish Minecraft Wiki",
              "id": "P13597",
              "label_en": "MCW-PL article ID"
            },
            {
              "description_en": "person, profession, organization or entity that operates the equipment, facility, or service",
              "id": "P137",
              "label_en": "operator"
            },
            {
              "description_en": "physical design paradigm this object is aligned with",
              "id": "P13768",
              "label_en": "form factor"
            },
            {
              "description_en": "entity or event that inspired the subject's name, or namesake (in at least one language). Qualifier \"applies to name\" (P5168) can be used to indicate which one",
              "id": "P138",
              "label_en": "named after"
            },
            {
              "description_en": "larger work that a given work was published in, like a journal, a website, a collection, a book or a music album",
              "id": "P1433",
              "label_en": "published in"
            },
            {
              "description_en": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
              "id": "P1476",
              "label_en": "title"
            },
            {
              "description_en": "qualification of the truth or accuracy of a source: circa (Q5727902), near (Q21818619), presumably (Q18122778), etc.",
              "id": "P1480",
              "label_en": "sourcing circumstances"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "item or concept that makes use of the subject (use sub-properties when appropriate)",
              "id": "P1535",
              "label_en": "used by"
            },
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
            {
              "description_en": "for works, when the title is followed by a subtitle",
              "id": "P1680",
              "label_en": "subtitle"
            },
            "... omitted 45 items"
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the listed qualifier has to be used",
          "id": "Q21510856",
          "label_en": "required qualifier constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "organization or person responsible for publishing a work, such as a book, periodical, printed music, podcast, game or software",
              "id": "P123",
              "label_en": "publisher"
            },
            {
              "description_en": "owner of the subject",
              "id": "P127",
              "label_en": "owned by"
            },
            {
              "description_en": "person, profession, organization or entity that operates the equipment, facility, or service",
              "id": "P137",
              "label_en": "operator"
            },
            {
              "description_en": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
              "id": "P1476",
              "label_en": "title"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "intended audience or user of this work, product, object, or event",
              "id": "P2360",
              "label_en": "intended public"
            },
            {
              "description_en": "role held by the item the statement appears on (\"subject\") in the context of that statement. For the role of the statement object/value, use P3831 (\"object has role\"). For acting roles, use P453 (\"character role\"). For persons, use P39.",
              "id": "P2868",
              "label_en": "subject has role"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "contents, item or substance located within this wrapping item, but not part of this receptacle or container; does not describe the ingredients or components of this container",
              "id": "P4330",
              "label_en": "contains"
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
            },
            {
              "description_en": "time an entity begins to exist or a statement starts being valid",
              "id": "P580",
              "label_en": "start time"
            },
            {
              "description_en": "moment when an entity ceases to exist and a statement stops being entirely valid or no longer be true",
              "id": "P582",
              "label_en": "end time"
            },
            {
              "description_en": "date something took place, existed or a statement was true; for providing time use the \"refine date\" property (P4241)",
              "id": "P585",
              "label_en": "point in time"
            },
            {
              "description_en": "primary topic of a work or act of communication",
              "id": "P921",
              "label_en": "main subject"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            }
          ],
          "P6607": [
            {
              "value": "For qualifiers use URL (P2699), for references use reference URL (P854)@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "a36a61f0e86f73f6deb18c55577835a1b9bbf650",
  "hash_before": "a24e4427528acfa48d593fe8b3d3db4c276c2009",
  "property_revision_id": 2446852358,
  "property_revision_prev": 2443941926,
  "qualifier_value_changes": [
    {
      "added_values": [
        "P7081"
      ],
      "constraint_qid": "Q21510851",
      "qualifier_property": "P2306",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21510851",
      "qualifiers": [
        {
          "property_id": "P2306",
          "values": [
            "P1001",
            "P1019",
            "P1065",
            "P108",
            "P123",
            "P12506",
            "P126",
            "P127",
            "P13044",
            "P1319",
            "P1326",
            "P13337",
            "P1343",
            "P13597",
            "P137",
            "P13768",
            "P138",
            "P1433",
            "P1476",
            "P1480",
            "P1534",
            "P1535",
            "P1552",
            "P1680",
            "... omitted 45 items"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "format constraint: format as a regular expression: (?i)((?!\\b(://web\\.archive\\.org/)).)*; exception to constraint: Wayback Machine; syntax clarification: Die Adresse der Webseite sollte nicht den Ausdruck 'web.archive.org' enthalten.@de, a string not including 'web.archive.org'@en; constraint clarification: Add archive link with 'archive URL' (P1065) and qualify former official website with 'end time' (P582) instead.@en, Ändere bitte die Aussage zur ehemaligen off... [truncated 184 chars]",
      "format constraint: format as a regular expression: (https?|ftps?)://\\S+",
      "format constraint: format as a regular expression: ^((?!google\\.com/search\\?).)*$; exception to constraint: AI Mode; syntax clarification: Die Adresse der Webseite sollte nicht den Ausdruck 'google.com/search' enthalten.@de, a string not including 'google.com/search'@en",
      "format constraint: format as a regular expression: ^(?!https?://www\\.$).+; syntax clarification: Die Adresse enthält keinen Domainnamen@de, empty value not including a domain name@en",
      "distinct-values constraint: exception to constraint: RDA value vocabularies, RDA/ONIX framework value vocabularies, AIATSIS Subject Thesaurus, AIATSIS Place Thesaurus, Washington State Parks website, ABBAsite.com, Gauche démocratique et sociale, Macaulay Library, Navile's University Library. Section of Chemistry \"Giacomo Ciamician\", Navile's University Library. Section of the Department of Pharmacy and Biotechnology, Lumi, Delaware Department of State, Rowland Ric... [truncated 2436 chars]",
      "conflicts-with constraint: exception to constraint: list of natural events named by the Japan Meteorological Agency; item of property constraint: Wikimedia template, Wikimedia disambiguation page, Wikimedia category; property: instance of",
      "item-requires-statement constraint: property: instance of",
      "allowed qualifiers constraint: property: applies to jurisdiction, web feed URL, archive URL, employer, publisher, latest end date, maintained by, owned by, characteristic of, earliest date, latest date, domain name, described by source, MCW-PL article ID, operator, form factor, named after, published in, title, sourcing circumstances, end cause, used by, has characteristic, subtitle, together with, subject named as, object named as, reason for deprecated rank, int... [truncated 752 chars]",
      "required qualifier constraint: property: language of work or name; constraint status: suggestion constraint",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "single-best-value constraint: separator: applies to jurisdiction, publisher, owned by, operator, title, subject named as, intended public, subject has role, object of statement has role, language of work or name, contains, applies to part, start time, end time, point in time, main subject",
      "property scope constraint: property scope: as main value; constraint clarification: For qualifiers use URL (P2699), for references use reference URL (P854)@en"
    ],
    "before": [
      "format constraint: format as a regular expression: (?i)((?!\\b(://web\\.archive\\.org/)).)*; exception to constraint: Wayback Machine; syntax clarification: Die Adresse der Webseite sollte nicht den Ausdruck 'web.archive.org' enthalten.@de, a string not including 'web.archive.org'@en; constraint clarification: Add archive link with 'archive URL' (P1065) and qualify former official website with 'end time' (P582) instead.@en, Ändere bitte die Aussage zur ehemaligen off... [truncated 184 chars]",
      "format constraint: format as a regular expression: (https?|ftps?)://\\S+",
      "format constraint: format as a regular expression: ^((?!google\\.com/search\\?).)*$; exception to constraint: AI Mode; syntax clarification: Die Adresse der Webseite sollte nicht den Ausdruck 'google.com/search' enthalten.@de, a string not including 'google.com/search'@en",
      "format constraint: format as a regular expression: ^(?!https?://www\\.$).+; syntax clarification: Die Adresse enthält keinen Domainnamen@de, empty value not including a domain name@en",
      "distinct-values constraint: exception to constraint: RDA value vocabularies, RDA/ONIX framework value vocabularies, AIATSIS Subject Thesaurus, AIATSIS Place Thesaurus, Washington State Parks website, ABBAsite.com, Gauche démocratique et sociale, Macaulay Library, Navile's University Library. Section of Chemistry \"Giacomo Ciamician\", Navile's University Library. Section of the Department of Pharmacy and Biotechnology, Lumi, Delaware Department of State, Rowland Ric... [truncated 2436 chars]",
      "conflicts-with constraint: exception to constraint: list of natural events named by the Japan Meteorological Agency; item of property constraint: Wikimedia template, Wikimedia disambiguation page, Wikimedia category; property: instance of",
      "item-requires-statement constraint: property: instance of",
      "allowed qualifiers constraint: property: applies to jurisdiction, web feed URL, archive URL, employer, publisher, latest end date, maintained by, owned by, characteristic of, earliest date, latest date, domain name, described by source, MCW-PL article ID, operator, form factor, named after, published in, title, sourcing circumstances, end cause, used by, has characteristic, subtitle, together with, subject named as, object named as, reason for deprecated rank, int... [truncated 730 chars]",
      "required qualifier constraint: property: language of work or name; constraint status: suggestion constraint",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "single-best-value constraint: separator: applies to jurisdiction, publisher, owned by, operator, title, subject named as, intended public, subject has role, object of statement has role, language of work or name, contains, applies to part, start time, end time, point in time, main subject",
      "property scope constraint: property scope: as main value; constraint clarification: For qualifiers use URL (P2699), for references use reference URL (P854)@en"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q21502410",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Unique value"
  }
]
```

---

## 007. `reform_Q136925578_P856_2447213274`

| Field | Value |
|---|---|
| qid | Q136925578 |
| property | P856 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | TBOX::P856::2447213274 |
| tbox_revision_key | TBOX::P856::2447213274 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Trivialist",
  "kind": "T_BOX",
  "property_revision_id": 2447213274,
  "property_revision_prev": 2446852358
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-26T11:39:43",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P856",
  "report_revision_new": 2447351370,
  "report_revision_old": 2447046423,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "http://dx.doi.org/10.59350/1r9b8-v4y82"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "URL of the official page of an item (current or former). Usage: If a listed URL no longer points to the official website, do not remove it, but see the \"Hijacked or dead websites\" section of the Talk page",
    "label": "official website"
  },
  "qid": {
    "description": null,
    "label": "A databank of molecular dynamics reaction trajectories (DDT) focused on undergraduate teaching."
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21510851",
      "qualifiers": [
        {
          "property_id": "P2306",
          "values": [
            "P1001",
            "P1019",
            "P1065",
            "P108",
            "P123",
            "P12506",
            "P126",
            "P127",
            "P13044",
            "P1319",
            "P1326",
            "P13337",
            "P1343",
            "P13589",
            "P13597",
            "P137",
            "P13768",
            "P138",
            "P1433",
            "P1476",
            "P1480",
            "P1534",
            "P1535",
            "P1552",
            "... omitted 47 items"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 12,
  "author": "Trivialist",
  "before_constraint_count": 12,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(?i)((?!\\b(://web\\.archive\\.org/)).)*"
            }
          ],
          "P2303": [
            {
              "description_en": "digital archive founded by the Internet Archive",
              "id": "Q648266",
              "label_en": "Wayback Machine"
            }
          ],
          "P2916": [
            {
              "value": "Die Adresse der Webseite sollte nicht den Ausdruck 'web.archive.org' enthalten.@de"
            },
            {
              "value": "a string not including 'web.archive.org'@en"
            }
          ],
          "P6607": [
            {
              "value": "Add archive link with 'archive URL' (P1065) and qualify former official website with 'end time' (P582) instead.@en"
            },
            {
              "value": "Ändere bitte die Aussage zur ehemaligen offiziellen Webseite und füge den Archivlink über die Eigentschaft P1065 als Qualifikator hinzu.@de"
            },
            {
              "value": "请不要包含“web.archive.org”，请改为正常添加原始链接，将存档链接添加为“存档URL”（P1065）并加限定符“结束日期”（P582）至前官方网站@zh-cn"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(https?|ftps?)://\\S+"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "^((?!google\\.com/search\\?).)*$"
            }
          ],
          "P2303": [
            {
              "description_en": "feature of Google Search",
              "id": "Q135474941",
              "label_en": "AI Mode"
            }
          ],
          "P2916": [
            {
              "value": "Die Adresse der Webseite sollte nicht den Ausdruck 'google.com/search' enthalten.@de"
            },
            {
              "value": "a string not including 'google.com/search'@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "^(?!https?://www\\.$).+"
            }
          ],
          "P2916": [
            {
              "value": "Die Adresse enthält keinen Domainnamen@de"
            },
            {
              "value": "empty value not including a domain name@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "vocabulary encoding schemes for specified RDA properties",
              "id": "Q104417848",
              "label_en": "RDA value vocabularies"
            },
            {
              "description_en": null,
              "id": "Q104533647",
              "label_en": "RDA/ONIX framework value vocabularies"
            },
            {
              "description_en": "thesaurus of terms relating to Australian Aboriginal and Torres Strait Islander studies",
              "id": "Q105536068",
              "label_en": "AIATSIS Subject Thesaurus"
            },
            {
              "description_en": "thesaurus of Australian place names, using the Indigenous place name first wherever possible",
              "id": "Q105550536",
              "label_en": "AIATSIS Place Thesaurus"
            },
            {
              "description_en": "website of the Washington State Parks agency",
              "id": "Q106426688",
              "label_en": "Washington State Parks website"
            },
            {
              "description_en": "official website of ABBA band",
              "id": "Q108001817",
              "label_en": "ABBAsite.com"
            },
            {
              "description_en": "French political party",
              "id": "Q111225168",
              "label_en": "Gauche démocratique et sociale"
            },
            {
              "description_en": "website of the Cornell Lab of Ornithology Macaulay Library",
              "id": "Q111982999",
              "label_en": "Macaulay Library"
            },
            {
              "description_en": "University library in Bologna, Italy",
              "id": "Q112121329",
              "label_en": "Navile's University Library. Section of Chemistry \"Giacomo Ciamician\""
            },
            {
              "description_en": "University library in Bologna, Italy",
              "id": "Q112126763",
              "label_en": "Navile's University Library. Section of the Department of Pharmacy and Biotechnology"
            },
            {
              "description_en": "fictional character by miHoYo",
              "id": "Q113117660",
              "label_en": "Lumi"
            },
            {
              "description_en": "executive agency of the state of Delaware",
              "id": "Q113133561",
              "label_en": "Delaware Department of State"
            },
            {
              "description_en": "American textile artist and academic (born 1971)",
              "id": "Q113293536",
              "label_en": "Rowland Ricketts"
            },
            {
              "description_en": "Japanese American weaver who crafts traditional narrow-width yardage for kimono and obi using historical kasuri (ikat) techniques",
              "id": "Q113297824",
              "label_en": "Chinami Ricketts"
            },
            {
              "description_en": "Altiusrt website for Hockey New Zealand",
              "id": "Q113634129",
              "label_en": "Hockey New Zealand: Altiusrt"
            },
            {
              "description_en": "website of the Maryland Biodiversity Project",
              "id": "Q113634693",
              "label_en": "Maryland Biodiversity Project"
            },
            {
              "description_en": "website of the Nebraska Invasive Species Program",
              "id": "Q113685547",
              "label_en": "Nebraska Invasive Species Program"
            },
            {
              "description_en": "program affiliated with the University of Nebraska–Lincoln",
              "id": "Q113685550",
              "label_en": "Nebraska Invasive Species Program"
            },
            {
              "description_en": "deutsches Unternehmen mit Sitz Nürburg und Betreiber des Nürburgrings",
              "id": "Q114458607",
              "label_en": "Nürburgring 1927 GmbH & Co. KG"
            },
            {
              "description_en": "website of the Australian Bureau of Meteorology",
              "id": "Q114882717",
              "label_en": "Bureau of Meteorology"
            },
            {
              "description_en": "website",
              "id": "Q115336397",
              "label_en": "BestWestern.com"
            },
            {
              "description_en": "travel website",
              "id": "Q115384119",
              "label_en": "TravelWeekly.com"
            },
            {
              "description_en": "website of the publication Livres Hebdo",
              "id": "Q115443051",
              "label_en": "livreshebdo.fr"
            },
            {
              "description_en": "official website of French weekly news magazine Le Point",
              "id": "Q115553783",
              "label_en": "lePoint.fr"
            },
            "... omitted 88 items"
          ],
          "P4155": [
            {
              "description_en": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
              "id": "P1476",
              "label_en": "title"
            },
            {
              "description_en": "qualifier to specify the item that this property is shared with",
              "id": "P1706",
              "label_en": "together with"
            },
            {
              "description_en": "role held by the item the statement appears on (\"subject\") in the context of that statement. For the role of the statement object/value, use P3831 (\"object has role\"). For acting roles, use P453 (\"character role\"). For persons, use P39.",
              "id": "P2868",
              "label_en": "subject has role"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
            },
            {
              "description_en": "paragraph, or other kind of special indication to find information on a page or on a document (legal texts etc.)",
              "id": "P958",
              "label_en": "section"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "Wikimedia list article",
              "id": "Q11548107",
              "label_en": "list of natural events named by the Japan Meteorological Agency"
            }
          ],
          "P2305": [
            {
              "description_en": "type of page in the Wikimedia system. Use with P31 'instance of' for template pages",
              "id": "Q11266439",
              "label_en": "Wikimedia template"
            },
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            },
            {
              "description_en": "use with 'instance of' (P31) for Wikimedia category",
              "id": "Q4167836",
              "label_en": "Wikimedia category"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "news feed (RSS, Atom, etc.) of this person/organisation/project",
              "id": "P1019",
              "label_en": "web feed URL"
            },
            {
              "description_en": "URL to the archived web page specified with URL property",
              "id": "P1065",
              "label_en": "archive URL"
            },
            {
              "description_en": "person or organization for which the subject works or worked",
              "id": "P108",
              "label_en": "employer"
            },
            {
              "description_en": "organization or person responsible for publishing a work, such as a book, periodical, printed music, podcast, game or software",
              "id": "P123",
              "label_en": "publisher"
            },
            {
              "description_en": "latest date beyond which the statement could no longer be true",
              "id": "P12506",
              "label_en": "latest end date"
            },
            {
              "description_en": "person or organization in charge of keeping the subject (for instance an infrastructure) in functioning order",
              "id": "P126",
              "label_en": "maintained by"
            },
            {
              "description_en": "owner of the subject",
              "id": "P127",
              "label_en": "owned by"
            },
            {
              "description_en": "subject or main statement value is a characteristic, quality, property, or state of this object (use a more specific property where possible)",
              "id": "P13044",
              "label_en": "characteristic of"
            },
            {
              "description_en": "earliest date at which an event could have happened. Use as qualifier for other date properties",
              "id": "P1319",
              "label_en": "earliest date"
            },
            {
              "description_en": "latest possible time that something could have occurred. Use as qualifier for other date properties",
              "id": "P1326",
              "label_en": "latest date"
            },
            {
              "description_en": "registered internet domain name that identifies this entity",
              "id": "P13337",
              "label_en": "domain name"
            },
            {
              "description_en": "work where this item is described, in statistical contexts, a methodological note describing the data",
              "id": "P1343",
              "label_en": "described by source"
            },
            {
              "description_en": "qualifier property to be used with statements having the object \"no value\", given to provide a reason for \"no value\"",
              "id": "P13589",
              "label_en": "‎reason for no value"
            },
            {
              "description_en": "identifier of an article on the Polish Minecraft Wiki",
              "id": "P13597",
              "label_en": "MCW-PL article ID"
            },
            {
              "description_en": "person, profession, organization or entity that operates the equipment, facility, or service",
              "id": "P137",
              "label_en": "operator"
            },
            {
              "description_en": "physical design paradigm this object is aligned with",
              "id": "P13768",
              "label_en": "form factor"
            },
            {
              "description_en": "entity or event that inspired the subject's name, or namesake (in at least one language). Qualifier \"applies to name\" (P5168) can be used to indicate which one",
              "id": "P138",
              "label_en": "named after"
            },
            {
              "description_en": "larger work that a given work was published in, like a journal, a website, a collection, a book or a music album",
              "id": "P1433",
              "label_en": "published in"
            },
            {
              "description_en": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
              "id": "P1476",
              "label_en": "title"
            },
            {
              "description_en": "qualification of the truth or accuracy of a source: circa (Q5727902), near (Q21818619), presumably (Q18122778), etc.",
              "id": "P1480",
              "label_en": "sourcing circumstances"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "item or concept that makes use of the subject (use sub-properties when appropriate)",
              "id": "P1535",
              "label_en": "used by"
            },
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
            "... omitted 47 items"
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the listed qualifier has to be used",
          "id": "Q21510856",
          "label_en": "required qualifier constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "organization or person responsible for publishing a work, such as a book, periodical, printed music, podcast, game or software",
              "id": "P123",
              "label_en": "publisher"
            },
            {
              "description_en": "owner of the subject",
              "id": "P127",
              "label_en": "owned by"
            },
            {
              "description_en": "person, profession, organization or entity that operates the equipment, facility, or service",
              "id": "P137",
              "label_en": "operator"
            },
            {
              "description_en": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
              "id": "P1476",
              "label_en": "title"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "intended audience or user of this work, product, object, or event",
              "id": "P2360",
              "label_en": "intended public"
            },
            {
              "description_en": "role held by the item the statement appears on (\"subject\") in the context of that statement. For the role of the statement object/value, use P3831 (\"object has role\"). For acting roles, use P453 (\"character role\"). For persons, use P39.",
              "id": "P2868",
              "label_en": "subject has role"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "contents, item or substance located within this wrapping item, but not part of this receptacle or container; does not describe the ingredients or components of this container",
              "id": "P4330",
              "label_en": "contains"
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
            },
            {
              "description_en": "time an entity begins to exist or a statement starts being valid",
              "id": "P580",
              "label_en": "start time"
            },
            {
              "description_en": "moment when an entity ceases to exist and a statement stops being entirely valid or no longer be true",
              "id": "P582",
              "label_en": "end time"
            },
            {
              "description_en": "date something took place, existed or a statement was true; for providing time use the \"refine date\" property (P4241)",
              "id": "P585",
              "label_en": "point in time"
            },
            {
              "description_en": "primary topic of a work or act of communication",
              "id": "P921",
              "label_en": "main subject"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            }
          ],
          "P6607": [
            {
              "value": "For qualifiers use URL (P2699), for references use reference URL (P854)@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(?i)((?!\\b(://web\\.archive\\.org/)).)*"
            }
          ],
          "P2303": [
            {
              "description_en": "digital archive founded by the Internet Archive",
              "id": "Q648266",
              "label_en": "Wayback Machine"
            }
          ],
          "P2916": [
            {
              "value": "Die Adresse der Webseite sollte nicht den Ausdruck 'web.archive.org' enthalten.@de"
            },
            {
              "value": "a string not including 'web.archive.org'@en"
            }
          ],
          "P6607": [
            {
              "value": "Add archive link with 'archive URL' (P1065) and qualify former official website with 'end time' (P582) instead.@en"
            },
            {
              "value": "Ändere bitte die Aussage zur ehemaligen offiziellen Webseite und füge den Archivlink über die Eigentschaft P1065 als Qualifikator hinzu.@de"
            },
            {
              "value": "请不要包含“web.archive.org”，请改为正常添加原始链接，将存档链接添加为“存档URL”（P1065）并加限定符“结束日期”（P582）至前官方网站@zh-cn"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(https?|ftps?)://\\S+"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "^((?!google\\.com/search\\?).)*$"
            }
          ],
          "P2303": [
            {
              "description_en": "feature of Google Search",
              "id": "Q135474941",
              "label_en": "AI Mode"
            }
          ],
          "P2916": [
            {
              "value": "Die Adresse der Webseite sollte nicht den Ausdruck 'google.com/search' enthalten.@de"
            },
            {
              "value": "a string not including 'google.com/search'@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "^(?!https?://www\\.$).+"
            }
          ],
          "P2916": [
            {
              "value": "Die Adresse enthält keinen Domainnamen@de"
            },
            {
              "value": "empty value not including a domain name@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "vocabulary encoding schemes for specified RDA properties",
              "id": "Q104417848",
              "label_en": "RDA value vocabularies"
            },
            {
              "description_en": null,
              "id": "Q104533647",
              "label_en": "RDA/ONIX framework value vocabularies"
            },
            {
              "description_en": "thesaurus of terms relating to Australian Aboriginal and Torres Strait Islander studies",
              "id": "Q105536068",
              "label_en": "AIATSIS Subject Thesaurus"
            },
            {
              "description_en": "thesaurus of Australian place names, using the Indigenous place name first wherever possible",
              "id": "Q105550536",
              "label_en": "AIATSIS Place Thesaurus"
            },
            {
              "description_en": "website of the Washington State Parks agency",
              "id": "Q106426688",
              "label_en": "Washington State Parks website"
            },
            {
              "description_en": "official website of ABBA band",
              "id": "Q108001817",
              "label_en": "ABBAsite.com"
            },
            {
              "description_en": "French political party",
              "id": "Q111225168",
              "label_en": "Gauche démocratique et sociale"
            },
            {
              "description_en": "website of the Cornell Lab of Ornithology Macaulay Library",
              "id": "Q111982999",
              "label_en": "Macaulay Library"
            },
            {
              "description_en": "University library in Bologna, Italy",
              "id": "Q112121329",
              "label_en": "Navile's University Library. Section of Chemistry \"Giacomo Ciamician\""
            },
            {
              "description_en": "University library in Bologna, Italy",
              "id": "Q112126763",
              "label_en": "Navile's University Library. Section of the Department of Pharmacy and Biotechnology"
            },
            {
              "description_en": "fictional character by miHoYo",
              "id": "Q113117660",
              "label_en": "Lumi"
            },
            {
              "description_en": "executive agency of the state of Delaware",
              "id": "Q113133561",
              "label_en": "Delaware Department of State"
            },
            {
              "description_en": "American textile artist and academic (born 1971)",
              "id": "Q113293536",
              "label_en": "Rowland Ricketts"
            },
            {
              "description_en": "Japanese American weaver who crafts traditional narrow-width yardage for kimono and obi using historical kasuri (ikat) techniques",
              "id": "Q113297824",
              "label_en": "Chinami Ricketts"
            },
            {
              "description_en": "Altiusrt website for Hockey New Zealand",
              "id": "Q113634129",
              "label_en": "Hockey New Zealand: Altiusrt"
            },
            {
              "description_en": "website of the Maryland Biodiversity Project",
              "id": "Q113634693",
              "label_en": "Maryland Biodiversity Project"
            },
            {
              "description_en": "website of the Nebraska Invasive Species Program",
              "id": "Q113685547",
              "label_en": "Nebraska Invasive Species Program"
            },
            {
              "description_en": "program affiliated with the University of Nebraska–Lincoln",
              "id": "Q113685550",
              "label_en": "Nebraska Invasive Species Program"
            },
            {
              "description_en": "deutsches Unternehmen mit Sitz Nürburg und Betreiber des Nürburgrings",
              "id": "Q114458607",
              "label_en": "Nürburgring 1927 GmbH & Co. KG"
            },
            {
              "description_en": "website of the Australian Bureau of Meteorology",
              "id": "Q114882717",
              "label_en": "Bureau of Meteorology"
            },
            {
              "description_en": "website",
              "id": "Q115336397",
              "label_en": "BestWestern.com"
            },
            {
              "description_en": "travel website",
              "id": "Q115384119",
              "label_en": "TravelWeekly.com"
            },
            {
              "description_en": "website of the publication Livres Hebdo",
              "id": "Q115443051",
              "label_en": "livreshebdo.fr"
            },
            {
              "description_en": "official website of French weekly news magazine Le Point",
              "id": "Q115553783",
              "label_en": "lePoint.fr"
            },
            "... omitted 88 items"
          ],
          "P4155": [
            {
              "description_en": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
              "id": "P1476",
              "label_en": "title"
            },
            {
              "description_en": "qualifier to specify the item that this property is shared with",
              "id": "P1706",
              "label_en": "together with"
            },
            {
              "description_en": "role held by the item the statement appears on (\"subject\") in the context of that statement. For the role of the statement object/value, use P3831 (\"object has role\"). For acting roles, use P453 (\"character role\"). For persons, use P39.",
              "id": "P2868",
              "label_en": "subject has role"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
            },
            {
              "description_en": "paragraph, or other kind of special indication to find information on a page or on a document (legal texts etc.)",
              "id": "P958",
              "label_en": "section"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "Wikimedia list article",
              "id": "Q11548107",
              "label_en": "list of natural events named by the Japan Meteorological Agency"
            }
          ],
          "P2305": [
            {
              "description_en": "type of page in the Wikimedia system. Use with P31 'instance of' for template pages",
              "id": "Q11266439",
              "label_en": "Wikimedia template"
            },
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            },
            {
              "description_en": "use with 'instance of' (P31) for Wikimedia category",
              "id": "Q4167836",
              "label_en": "Wikimedia category"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "news feed (RSS, Atom, etc.) of this person/organisation/project",
              "id": "P1019",
              "label_en": "web feed URL"
            },
            {
              "description_en": "URL to the archived web page specified with URL property",
              "id": "P1065",
              "label_en": "archive URL"
            },
            {
              "description_en": "person or organization for which the subject works or worked",
              "id": "P108",
              "label_en": "employer"
            },
            {
              "description_en": "organization or person responsible for publishing a work, such as a book, periodical, printed music, podcast, game or software",
              "id": "P123",
              "label_en": "publisher"
            },
            {
              "description_en": "latest date beyond which the statement could no longer be true",
              "id": "P12506",
              "label_en": "latest end date"
            },
            {
              "description_en": "person or organization in charge of keeping the subject (for instance an infrastructure) in functioning order",
              "id": "P126",
              "label_en": "maintained by"
            },
            {
              "description_en": "owner of the subject",
              "id": "P127",
              "label_en": "owned by"
            },
            {
              "description_en": "subject or main statement value is a characteristic, quality, property, or state of this object (use a more specific property where possible)",
              "id": "P13044",
              "label_en": "characteristic of"
            },
            {
              "description_en": "earliest date at which an event could have happened. Use as qualifier for other date properties",
              "id": "P1319",
              "label_en": "earliest date"
            },
            {
              "description_en": "latest possible time that something could have occurred. Use as qualifier for other date properties",
              "id": "P1326",
              "label_en": "latest date"
            },
            {
              "description_en": "registered internet domain name that identifies this entity",
              "id": "P13337",
              "label_en": "domain name"
            },
            {
              "description_en": "work where this item is described, in statistical contexts, a methodological note describing the data",
              "id": "P1343",
              "label_en": "described by source"
            },
            {
              "description_en": "identifier of an article on the Polish Minecraft Wiki",
              "id": "P13597",
              "label_en": "MCW-PL article ID"
            },
            {
              "description_en": "person, profession, organization or entity that operates the equipment, facility, or service",
              "id": "P137",
              "label_en": "operator"
            },
            {
              "description_en": "physical design paradigm this object is aligned with",
              "id": "P13768",
              "label_en": "form factor"
            },
            {
              "description_en": "entity or event that inspired the subject's name, or namesake (in at least one language). Qualifier \"applies to name\" (P5168) can be used to indicate which one",
              "id": "P138",
              "label_en": "named after"
            },
            {
              "description_en": "larger work that a given work was published in, like a journal, a website, a collection, a book or a music album",
              "id": "P1433",
              "label_en": "published in"
            },
            {
              "description_en": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
              "id": "P1476",
              "label_en": "title"
            },
            {
              "description_en": "qualification of the truth or accuracy of a source: circa (Q5727902), near (Q21818619), presumably (Q18122778), etc.",
              "id": "P1480",
              "label_en": "sourcing circumstances"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "item or concept that makes use of the subject (use sub-properties when appropriate)",
              "id": "P1535",
              "label_en": "used by"
            },
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
            {
              "description_en": "for works, when the title is followed by a subtitle",
              "id": "P1680",
              "label_en": "subtitle"
            },
            "... omitted 46 items"
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the listed qualifier has to be used",
          "id": "Q21510856",
          "label_en": "required qualifier constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "organization or person responsible for publishing a work, such as a book, periodical, printed music, podcast, game or software",
              "id": "P123",
              "label_en": "publisher"
            },
            {
              "description_en": "owner of the subject",
              "id": "P127",
              "label_en": "owned by"
            },
            {
              "description_en": "person, profession, organization or entity that operates the equipment, facility, or service",
              "id": "P137",
              "label_en": "operator"
            },
            {
              "description_en": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
              "id": "P1476",
              "label_en": "title"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "intended audience or user of this work, product, object, or event",
              "id": "P2360",
              "label_en": "intended public"
            },
            {
              "description_en": "role held by the item the statement appears on (\"subject\") in the context of that statement. For the role of the statement object/value, use P3831 (\"object has role\"). For acting roles, use P453 (\"character role\"). For persons, use P39.",
              "id": "P2868",
              "label_en": "subject has role"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "contents, item or substance located within this wrapping item, but not part of this receptacle or container; does not describe the ingredients or components of this container",
              "id": "P4330",
              "label_en": "contains"
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
            },
            {
              "description_en": "time an entity begins to exist or a statement starts being valid",
              "id": "P580",
              "label_en": "start time"
            },
            {
              "description_en": "moment when an entity ceases to exist and a statement stops being entirely valid or no longer be true",
              "id": "P582",
              "label_en": "end time"
            },
            {
              "description_en": "date something took place, existed or a statement was true; for providing time use the \"refine date\" property (P4241)",
              "id": "P585",
              "label_en": "point in time"
            },
            {
              "description_en": "primary topic of a work or act of communication",
              "id": "P921",
              "label_en": "main subject"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            }
          ],
          "P6607": [
            {
              "value": "For qualifiers use URL (P2699), for references use reference URL (P854)@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "a3c6b9ae9ecf7dae4721f27e9093f68129421a13",
  "hash_before": "a36a61f0e86f73f6deb18c55577835a1b9bbf650",
  "property_revision_id": 2447213274,
  "property_revision_prev": 2446852358,
  "qualifier_value_changes": [
    {
      "added_values": [
        "P13589"
      ],
      "constraint_qid": "Q21510851",
      "qualifier_property": "P2306",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21510851",
      "qualifiers": [
        {
          "property_id": "P2306",
          "values": [
            "P1001",
            "P1019",
            "P1065",
            "P108",
            "P123",
            "P12506",
            "P126",
            "P127",
            "P13044",
            "P1319",
            "P1326",
            "P13337",
            "P1343",
            "P13597",
            "P137",
            "P13768",
            "P138",
            "P1433",
            "P1476",
            "P1480",
            "P1534",
            "P1535",
            "P1552",
            "P1680",
            "... omitted 46 items"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "format constraint: format as a regular expression: (?i)((?!\\b(://web\\.archive\\.org/)).)*; exception to constraint: Wayback Machine; syntax clarification: Die Adresse der Webseite sollte nicht den Ausdruck 'web.archive.org' enthalten.@de, a string not including 'web.archive.org'@en; constraint clarification: Add archive link with 'archive URL' (P1065) and qualify former official website with 'end time' (P582) instead.@en, Ändere bitte die Aussage zur ehemaligen off... [truncated 184 chars]",
      "format constraint: format as a regular expression: (https?|ftps?)://\\S+",
      "format constraint: format as a regular expression: ^((?!google\\.com/search\\?).)*$; exception to constraint: AI Mode; syntax clarification: Die Adresse der Webseite sollte nicht den Ausdruck 'google.com/search' enthalten.@de, a string not including 'google.com/search'@en",
      "format constraint: format as a regular expression: ^(?!https?://www\\.$).+; syntax clarification: Die Adresse enthält keinen Domainnamen@de, empty value not including a domain name@en",
      "distinct-values constraint: exception to constraint: RDA value vocabularies, RDA/ONIX framework value vocabularies, AIATSIS Subject Thesaurus, AIATSIS Place Thesaurus, Washington State Parks website, ABBAsite.com, Gauche démocratique et sociale, Macaulay Library, Navile's University Library. Section of Chemistry \"Giacomo Ciamician\", Navile's University Library. Section of the Department of Pharmacy and Biotechnology, Lumi, Delaware Department of State, Rowland Ric... [truncated 2436 chars]",
      "conflicts-with constraint: exception to constraint: list of natural events named by the Japan Meteorological Agency; item of property constraint: Wikimedia template, Wikimedia disambiguation page, Wikimedia category; property: instance of",
      "item-requires-statement constraint: property: instance of",
      "allowed qualifiers constraint: property: applies to jurisdiction, web feed URL, archive URL, employer, publisher, latest end date, maintained by, owned by, characteristic of, earliest date, latest date, domain name, described by source, ‎reason for no value, MCW-PL article ID, operator, form factor, named after, published in, title, sourcing circumstances, end cause, used by, has characteristic, subtitle, together with, subject named as, object named as, reason fo... [truncated 774 chars]",
      "required qualifier constraint: property: language of work or name; constraint status: suggestion constraint",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "single-best-value constraint: separator: applies to jurisdiction, publisher, owned by, operator, title, subject named as, intended public, subject has role, object of statement has role, language of work or name, contains, applies to part, start time, end time, point in time, main subject",
      "property scope constraint: property scope: as main value; constraint clarification: For qualifiers use URL (P2699), for references use reference URL (P854)@en"
    ],
    "before": [
      "format constraint: format as a regular expression: (?i)((?!\\b(://web\\.archive\\.org/)).)*; exception to constraint: Wayback Machine; syntax clarification: Die Adresse der Webseite sollte nicht den Ausdruck 'web.archive.org' enthalten.@de, a string not including 'web.archive.org'@en; constraint clarification: Add archive link with 'archive URL' (P1065) and qualify former official website with 'end time' (P582) instead.@en, Ändere bitte die Aussage zur ehemaligen off... [truncated 184 chars]",
      "format constraint: format as a regular expression: (https?|ftps?)://\\S+",
      "format constraint: format as a regular expression: ^((?!google\\.com/search\\?).)*$; exception to constraint: AI Mode; syntax clarification: Die Adresse der Webseite sollte nicht den Ausdruck 'google.com/search' enthalten.@de, a string not including 'google.com/search'@en",
      "format constraint: format as a regular expression: ^(?!https?://www\\.$).+; syntax clarification: Die Adresse enthält keinen Domainnamen@de, empty value not including a domain name@en",
      "distinct-values constraint: exception to constraint: RDA value vocabularies, RDA/ONIX framework value vocabularies, AIATSIS Subject Thesaurus, AIATSIS Place Thesaurus, Washington State Parks website, ABBAsite.com, Gauche démocratique et sociale, Macaulay Library, Navile's University Library. Section of Chemistry \"Giacomo Ciamician\", Navile's University Library. Section of the Department of Pharmacy and Biotechnology, Lumi, Delaware Department of State, Rowland Ric... [truncated 2436 chars]",
      "conflicts-with constraint: exception to constraint: list of natural events named by the Japan Meteorological Agency; item of property constraint: Wikimedia template, Wikimedia disambiguation page, Wikimedia category; property: instance of",
      "item-requires-statement constraint: property: instance of",
      "allowed qualifiers constraint: property: applies to jurisdiction, web feed URL, archive URL, employer, publisher, latest end date, maintained by, owned by, characteristic of, earliest date, latest date, domain name, described by source, MCW-PL article ID, operator, form factor, named after, published in, title, sourcing circumstances, end cause, used by, has characteristic, subtitle, together with, subject named as, object named as, reason for deprecated rank, int... [truncated 752 chars]",
      "required qualifier constraint: property: language of work or name; constraint status: suggestion constraint",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "single-best-value constraint: separator: applies to jurisdiction, publisher, owned by, operator, title, subject named as, intended public, subject has role, object of statement has role, language of work or name, contains, applies to part, start time, end time, point in time, main subject",
      "property scope constraint: property scope: as main value; constraint clarification: For qualifiers use URL (P2699), for references use reference URL (P854)@en"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q21502410",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Unique value"
  }
]
```

---

## 008. `reform_Q17104923_P1930_696610230`

| Field | Value |
|---|---|
| qid | Q17104923 |
| property | P1930 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| group_key | TBOX::P1930::696610230 |
| tbox_revision_key | TBOX::P1930::696610230 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "MisterSynergy",
  "kind": "T_BOX",
  "property_revision_id": 696610230,
  "property_revision_prev": 683950161
}
```

### Violation Context

```json
{
  "report_fix_date": "2018-06-19T21:15:17",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1930",
  "report_revision_new": 698761773,
  "report_revision_old": 698391088,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "307.59"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for a mental disorder in the 5th edition of Diagnostic and Statistical Manual of Mental Disorders",
    "label": "DSM-5 (identifier)"
  },
  "qid": {
    "description": "disorder",
    "label": "other specified feeding or eating disorder"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q53869507",
      "qualifiers": [
        {
          "property_id": "P2316",
          "values": [
            "Q21502408"
          ]
        },
        {
          "property_id": "P5314",
          "values": [
            "Q54828448",
            "Q54828450"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 4,
  "author": "MisterSynergy",
  "before_constraint_count": 4,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "\\d\\d\\d\\.\\d\\d"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ],
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "\\d\\d\\d\\.\\d\\d"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ],
          "P4680": [
            {
              "description_en": "scope for constraints that should be checked on the main value of a statement",
              "id": "Q46466787",
              "label_en": "constraint checked on main value"
            },
            {
              "description_en": "scope for constraints that should be checked on the references of a statement",
              "id": "Q46466805",
              "label_en": "constraint checked on references"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "602fb00163576434409ef878b19bac342eb41b33",
  "hash_before": "27304173eefd26265c337073340ea9a5b3312e41",
  "property_revision_id": 696610230,
  "property_revision_prev": 683950161,
  "qualifier_value_changes": [
    {
      "added_values": [],
      "constraint_qid": "Q53869507",
      "qualifier_property": "P4680",
      "removed_values": [
        "Q46466787",
        "Q46466805"
      ],
      "same_qid_index": 0
    },
    {
      "added_values": [
        "Q54828448",
        "Q54828450"
      ],
      "constraint_qid": "Q53869507",
      "qualifier_property": "P5314",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q53869507",
      "qualifiers": [
        {
          "property_id": "P2316",
          "values": [
            "Q21502408"
          ]
        },
        {
          "property_id": "P4680",
          "values": [
            "Q46466787",
            "Q46466805"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: \\d\\d\\d\\.\\d\\d",
      "distinct-values constraint: no qualifiers recorded",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as reference"
    ],
    "before": [
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: \\d\\d\\d\\.\\d\\d",
      "distinct-values constraint: no qualifiers recorded",
      "property scope constraint: constraint status: mandatory constraint; constraint scope: constraint checked on main value, constraint checked on references"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q21502410",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Unique value"
  }
]
```

---

## 009. `reform_Q18967000_P2249_2265468275`

| Field | Value |
|---|---|
| qid | Q18967000 |
| property | P2249 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q21502410 |
| group_key | TBOX::P2249::2265468275 |
| tbox_revision_key | TBOX::P2249::2265468275 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "ChristianKl",
  "kind": "T_BOX",
  "property_revision_id": 2265468275,
  "property_revision_prev": 2259997813
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-10-27T06:44:04",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2249",
  "report_revision_new": 2265688877,
  "report_revision_old": 2260484921,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": null,
  "value_current_2026": [
    "NC_013266",
    "NC_013267",
    "NC_013268"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "ID in the RefSeq Genome database",
    "label": "RefSeq genome ID"
  },
  "qid": {
    "description": "species of virus",
    "label": "Melandrium yellow fleck virus"
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
    "label_en": "format constraint",
    "qid": "Q21502404"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 5,
  "author": "ChristianKl",
  "before_constraint_count": 6,
  "changed_constraint_types": [
    "Q53869507"
  ],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[A-Z]{2}.+"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[A-Z]{2}_\\d+(\\.\\d{1,2})?"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[A-Z]{2}.+"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[A-Z]{2}_\\d+(\\.\\d{1,2})?"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "514df148fe06fa0fbc1b5efd9fca0eb88be89f4c",
  "hash_before": "2dfc9c713d597654da3f6f97ff7b2cf2b8d81662",
  "property_revision_id": 2265468275,
  "property_revision_prev": 2259997813,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q53869507",
      "qualifiers": [],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: [A-Z]{2}.+; constraint status: mandatory constraint",
      "format constraint: format as a regular expression: [A-Z]{2}_\\d+(\\.\\d{1,2})?",
      "distinct-values constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item"
    ],
    "before": [
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: [A-Z]{2}.+; constraint status: mandatory constraint",
      "format constraint: format as a regular expression: [A-Z]{2}_\\d+(\\.\\d{1,2})?",
      "distinct-values constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: no qualifiers recorded"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [
      "Q53869507"
    ],
    "mapped_constraint_qid": "Q19474404",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Single value"
  }
]
```

---

## 010. `reform_Q22661176_P4203_664001720`

| Field | Value |
|---|---|
| qid | Q22661176 |
| property | P4203 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| group_key | TBOX::P4203::664001720 |
| tbox_revision_key | TBOX::P4203::664001720 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Manu1400",
  "kind": "T_BOX",
  "property_revision_id": 664001720,
  "property_revision_prev": 644405397
}
```

### Violation Context

```json
{
  "report_fix_date": "2018-04-16T18:44:03",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4203",
  "report_revision_new": 666108991,
  "report_revision_old": 665886352,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": null,
  "value_current_2026": [
    "682",
    "429"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier in the Registry of Open Access Repository Mandates and Policies",
    "label": "ROARMAP ID"
  },
  "qid": {
    "description": "policy adopted in 2015",
    "label": "Tri-Agency Open Access Policy on Publications"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502404",
      "qualifiers": [
        {
          "property_id": "P1793",
          "values": [
            "[1-9]\\d*"
          ]
        },
        {
          "property_id": "P2316",
          "values": [
            "Q21502408"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 3,
  "author": "Manu1400",
  "before_constraint_count": 3,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[1-9]\\d*"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "\\d+"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "d093f8a48918a74018aa24d3c7e23f0956450c68",
  "hash_before": "db4996fcf2be467196953fca4f4d966e518b8ae5",
  "property_revision_id": 664001720,
  "property_revision_prev": 644405397,
  "qualifier_value_changes": [
    {
      "added_values": [
        "[1-9]\\d*"
      ],
      "constraint_qid": "Q21502404",
      "qualifier_property": "P1793",
      "removed_values": [
        "\\d+"
      ],
      "same_qid_index": 0
    },
    {
      "added_values": [
        "Q21502408"
      ],
      "constraint_qid": "Q21502404",
      "qualifier_property": "P2316",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502404",
      "qualifiers": [
        {
          "property_id": "P1793",
          "values": [
            "\\d+"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: [1-9]\\d*; constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded"
    ],
    "before": [
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: \\d+",
      "distinct-values constraint: no qualifiers recorded"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q19474404",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Single value"
  }
]
```

---

## 011. `reform_Q2994387_P10717_1636003825`

| Field | Value |
|---|---|
| qid | Q2994387 |
| property | P10717 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q21502410 |
| group_key | TBOX::P10717::1636003825 |
| tbox_revision_key | TBOX::P10717::1636003825 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Pamputt",
  "kind": "T_BOX",
  "property_revision_id": 1636003825,
  "property_revision_prev": 1636003746
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-05-11T07:03:09",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P10717",
  "report_revision_new": 1636602215,
  "report_revision_old": 1636540467,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "counseling"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "Hebrew-language encyclopedia describing philosophical ideas in daily life",
    "label": "Encyclopedia of Ideas ID"
  },
  "qid": {
    "description": "person with more and deeper knowledge in a specific area, who is part of the leadership; for consultant use Q15978655 who is instead fulfilling functional roles",
    "label": "adviser"
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
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q19474404",
      "qualifiers": [],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 2,
  "author": "Pamputt",
  "before_constraint_count": 1,
  "changed_constraint_types": [
    "Q19474404"
  ],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "2bce93b2236f390231e5262059cdb155a5795b0e",
  "hash_before": "d0d87360eefd70543b57752961a26f509b3ff453",
  "property_revision_id": 1636003825,
  "property_revision_prev": 1636003746,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: no qualifiers recorded",
      "distinct-values constraint: no qualifiers recorded"
    ],
    "before": [
      "distinct-values constraint: no qualifiers recorded"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [
      "Q19474404"
    ],
    "mapped_constraint_qid": "Q21502410",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Unique value"
  }
]
```

---

## 012. `reform_Q3644856_P400_2355127724`

| Field | Value |
|---|---|
| qid | Q3644856 |
| property | P400 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q21510865 |
| group_key | TBOX::P400::2355127724 |
| tbox_revision_key | TBOX::P400::2355127724 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Trade",
  "kind": "T_BOX",
  "property_revision_id": 2355127724,
  "property_revision_prev": 2355127546
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-06-02T11:38:21",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P400",
  "report_revision_new": 2356023718,
  "report_revision_old": 2355590472,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": null,
  "value_current_2026": [
    "Q1406",
    "Q10683",
    "Q14116",
    "Q388"
  ],
  "value_current_2026_descriptions_en": [
    "family of computer operating systems developed by Microsoft",
    "video game console developed Sony Interactive Entertainment",
    "operating system for Apple computers",
    "family of Unix-like operating systems"
  ],
  "value_current_2026_labels_en": [
    "Microsoft Windows",
    "PlayStation 3",
    "macOS",
    "Linux"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "platform for which a work was developed or released, or the specific platform version of a software product",
    "label": "platform"
  },
  "qid": {
    "description": "2013 Tower Defense/FPS video game",
    "label": "Sanctum 2"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q52558054",
      "qualifiers": [
        {
          "property_id": "P2305",
          "values": [
            "Q17517"
          ]
        },
        {
          "property_id": "P6607",
          "values": [
            "This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134621319@en"
          ]
        },
        {
          "property_id": "P9729",
          "values": [
            "Q171819",
            "Q193828",
            "Q48493",
            "Q4885200",
            "Q863516",
            "Q94"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 27,
  "author": "Trade",
  "before_constraint_count": 27,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "set of episodes produced for an anime television series",
              "id": "Q100269041",
              "label_en": "anime television series season"
            },
            {
              "description_en": "series of light novels published in Japan",
              "id": "Q104213567",
              "label_en": "light novel series"
            },
            {
              "description_en": "anime released directly online",
              "id": "Q1047299",
              "label_en": "original net animation"
            },
            {
              "description_en": "television program",
              "id": "Q11086742",
              "label_en": "anime television program"
            },
            {
              "description_en": "use of a creative work across several different media",
              "id": "Q196600",
              "label_en": "media franchise"
            },
            {
              "description_en": "animated film from Japan or in Japanese anime style",
              "id": "Q20650540",
              "label_en": "anime film"
            },
            {
              "description_en": "series of comics employing Japanese stylistic conventions that are that are formally identified together",
              "id": "Q21198342",
              "label_en": "manga series"
            },
            {
              "description_en": "Japanese anime designed for release in home-video formats",
              "id": "Q220898",
              "label_en": "original video animation"
            },
            {
              "description_en": "connected set of television program episodes under the same title",
              "id": "Q5398426",
              "label_en": "television series"
            },
            {
              "description_en": "imaginary, typically self-consistent world with its own rules and characters, different from the real world; often used as a background or basis in story telling",
              "id": "Q559618",
              "label_en": "fictional universe"
            },
            {
              "description_en": "Japanese animated television series",
              "id": "Q63952888",
              "label_en": "anime television series"
            },
            {
              "description_en": "Japanese novella-type storytelling in conjunction with illustrations, geared toward young adults",
              "id": "Q747381",
              "label_en": "light novel"
            },
            {
              "description_en": "comics employing a set of Japanese stylistic conventions, produced in Japan or elsewhere",
              "id": "Q8274",
              "label_en": "manga"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "organization that provides access to the Internet",
              "id": "Q11371",
              "label_en": "internet service provider"
            },
            {
              "description_en": "organization that provides telephone and/or other telecommunications service",
              "id": "Q1266169",
              "label_en": "telephone company"
            },
            {
              "description_en": "cellular service provider",
              "id": "Q1941618",
              "label_en": "mobile network operator"
            },
            {
              "description_en": "financial institution that accepts deposits",
              "id": "Q22687",
              "label_en": "bank"
            },
            {
              "description_en": "social entity established to meet needs or pursue goals",
              "id": "Q43229",
              "label_en": "organization"
            },
            {
              "description_en": "organization undertaking commercial, industrial, or professional activity",
              "id": "Q4830453",
              "label_en": "business"
            },
            {
              "description_en": "organizational unit producing goods or services, which benefits from a certain degree of autonomy in decision-making, especially for the allocation of its current resources",
              "id": "Q6881511",
              "label_en": "enterprise"
            },
            {
              "description_en": "company that offers its securities for sale to the general public",
              "id": "Q891723",
              "label_en": "public company"
            },
            {
              "description_en": "specific type of business organization, which was granted a right to perform banking operations by Central Bank of Russia",
              "id": "Q93429702",
              "label_en": "credit organization (business in Russia)"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "portable device to make telephone calls using a radio link",
              "id": "Q17517",
              "label_en": "mobile phone"
            },
            {
              "description_en": "device that plays DVD discs",
              "id": "Q3783103",
              "label_en": "DVD player"
            },
            {
              "description_en": null,
              "id": "Q61448957",
              "label_en": "PlayStation Theme"
            },
            {
              "description_en": "theme for the PlayStation 4 SHAREfactory app",
              "id": "Q61449115",
              "label_en": "SHAREfactory Theme"
            }
          ],
          "P2304": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ],
          "P2308": [
            {
              "description_en": "electronic device connected via different wireless protocols to its environment",
              "id": "Q11253473",
              "label_en": "smart device"
            },
            {
              "description_en": "company that publishes video games",
              "id": "Q1137109",
              "label_en": "video game publisher"
            },
            {
              "description_en": "any set of video games",
              "id": "Q116741534",
              "label_en": "group of video games"
            },
            {
              "description_en": "group or company that develops applications",
              "id": "Q125251322",
              "label_en": "application developer"
            },
            {
              "description_en": "number of distinct pixels in each dimension that can be displayed",
              "id": "Q12538706",
              "label_en": "display resolution"
            },
            {
              "description_en": "book-length publication in digital form",
              "id": "Q128093",
              "label_en": "ebook"
            },
            {
              "description_en": "puzzles, board games, or video games based on language",
              "id": "Q15220419",
              "label_en": "word game"
            },
            {
              "description_en": "software interface between computers and/or programs",
              "id": "Q165194",
              "label_en": "application programming interface"
            },
            {
              "description_en": "literary genre consisting of works of literature that originate within digital environments and require digital computation",
              "id": "Q173167",
              "label_en": "electronic literature"
            },
            {
              "description_en": "coin-operated video game machine",
              "id": "Q192851",
              "label_en": "arcade video game machine"
            },
            {
              "description_en": "software development organization specializing in the creation of video games (for person use Q58287519)",
              "id": "Q210167",
              "label_en": "video game developer"
            },
            {
              "description_en": "album type consisting of works chosen for a particular purpose or theme",
              "id": "Q222910",
              "label_en": "compilation album"
            },
            {
              "description_en": "any type of game under a common theme or story",
              "id": "Q28114058",
              "label_en": "game franchise"
            },
            {
              "description_en": "meta-goal in a video game",
              "id": "Q2988681",
              "label_en": "achievement"
            },
            {
              "description_en": "sequence of instructions written in programming language to perform a specified task with a computer",
              "id": "Q40056",
              "label_en": "computer program"
            },
            {
              "description_en": "fictional videogame in a narrative form of arts",
              "id": "Q40213094",
              "label_en": "fictional video game"
            },
            {
              "description_en": "electronic magazine to be read using computers",
              "id": "Q416",
              "label_en": "disk magazine"
            },
            {
              "description_en": "means by which a user interacts with and controls a machine",
              "id": "Q47146",
              "label_en": "user interface"
            },
            {
              "description_en": "group of people in the demoscene",
              "id": "Q5256141",
              "label_en": "demogroup"
            },
            {
              "description_en": "learning method",
              "id": "Q535741",
              "label_en": "tutorial"
            },
            {
              "description_en": "art software with graphics, music and animation",
              "id": "Q5610543",
              "label_en": "demo"
            },
            {
              "description_en": "compilation of software, in most cases, from the same developer",
              "id": "Q62651817",
              "label_en": "software bundle"
            },
            {
              "description_en": "non-tangible executable component of a computer",
              "id": "Q7397",
              "label_en": "software"
            },
            {
              "description_en": "electronic game with user interface and visual feedback",
              "id": "Q7889",
              "label_en": "video game"
            },
            "... omitted 5 items"
          ],
          "P2309": [
            {
              "description_en": "relation of type or value type constraint",
              "id": "Q30208840",
              "label_en": "instance or subclass of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "any type of firmware used in a computer for low-level hardware initialization during the boot process",
              "id": "Q102676349",
              "label_en": "computer firmware"
            },
            {
              "description_en": "Apple Macintosh’s original operating system (1984–2002)",
              "id": "Q13522376",
              "label_en": "Classic Mac OS"
            },
            {
              "description_en": "family of computer operating systems developed by Microsoft",
              "id": "Q1406",
              "label_en": "Microsoft Windows"
            },
            {
              "description_en": "operating system for Apple computers",
              "id": "Q14116",
              "label_en": "macOS"
            },
            {
              "description_en": "group of closely-related PC-compatible operating systems",
              "id": "Q170434",
              "label_en": "DOS"
            },
            {
              "description_en": "set of rules and methods that describe the functionality, organization and implementation of computer systems",
              "id": "Q173212",
              "label_en": "computer architecture"
            },
            {
              "description_en": "environment with a graphical user interface using the desktop metaphor",
              "id": "Q205020",
              "label_en": "desktop environment"
            },
            {
              "description_en": "environment in which a piece of software is executed",
              "id": "Q241317",
              "label_en": "computing platform"
            },
            {
              "description_en": "speaker with features or services that go beyond audio playback",
              "id": "Q26884850",
              "label_en": "smart speaker"
            },
            {
              "description_en": "software for creating online stores",
              "id": "Q2916479",
              "label_en": "shopping cart software"
            },
            {
              "description_en": "family of Unix-like operating systems",
              "id": "Q388",
              "label_en": "Linux"
            },
            {
              "description_en": "online publishing environment",
              "id": "Q4202064",
              "label_en": "publishing platform"
            },
            {
              "description_en": "mobile operating system by Apple Inc.",
              "id": "Q48493",
              "label_en": "iOS"
            },
            {
              "description_en": "type of operating system",
              "id": "Q600659",
              "label_en": "disk operating system"
            },
            {
              "description_en": "video game console in a narrative form of arts",
              "id": "Q60644919",
              "label_en": "fictional video game console"
            },
            {
              "description_en": "software application for retrieving, presenting, and traversing information resources on the World Wide Web",
              "id": "Q6368",
              "label_en": "web browser"
            },
            {
              "description_en": "interactive entertainment computer or customized computer system for running video games",
              "id": "Q8076",
              "label_en": "video game console"
            },
            {
              "description_en": "software that manages computer hardware resources",
              "id": "Q9135",
              "label_en": "operating system"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type or value type constraint",
              "id": "Q30208840",
              "label_en": "instance or subclass of"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ],
          "P6607": [
            {
              "value": "see Wikidata:Project_chat/Archive/2018/06#relation_between_platform_(P400)_and_operating_system_(P306)@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "1986 video game released by Activision",
              "id": "Q3613155",
              "label_en": "Alter Ego"
            }
          ],
          "P2305": [
            {
              "description_en": "line of smartphones developed and marketed by Apple Inc.",
              "id": "Q2766",
              "label_en": "آيفون"
            },
            {
              "description_en": "line of tablet computers developed by Apple",
              "id": "Q2796",
              "label_en": "آي باد"
            },
            {
              "description_en": "line of portable media players by Apple Inc.",
              "id": "Q9479",
              "label_en": "IPod"
            }
          ],
          "P6607": [
            {
              "value": "For Apple smartphones use \"iOS \" (Q48493) instead@en"
            },
            {
              "value": "Pour les téléphones intelligents d'Apple, utiliser « iOS » (Q48493) à la place@fr"
            }
          ],
          "P9729": [
            {
              "description_en": "mobile operating system by Apple Inc.",
              "id": "Q48493",
              "label_en": "iOS"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "digital media store",
              "id": "Q1052025",
              "label_en": "PlayStation Store"
            },
            {
              "description_en": "virtual market designed for Xbox",
              "id": "Q1204827",
              "label_en": "Xbox Games Store"
            },
            {
              "description_en": "digital distribution platform operated by Microsoft",
              "id": "Q135288",
              "label_en": "Microsoft Store"
            },
            {
              "description_en": "video game platform",
              "id": "Q1486288",
              "label_en": "Good Old Games"
            },
            {
              "description_en": null,
              "id": "Q16991786",
              "label_en": "Amazon Digital Game Store"
            },
            {
              "description_en": "game distribution platform",
              "id": "Q17238705",
              "label_en": "Playism"
            },
            {
              "description_en": "adult video game distribution platform",
              "id": "Q22079967",
              "label_en": "Nutaku"
            },
            {
              "description_en": "website for distributing video games",
              "id": "Q22905933",
              "label_en": "itch.io"
            },
            {
              "description_en": "instant messaging and VoIP software",
              "id": "Q22907849",
              "label_en": "Discord"
            },
            {
              "description_en": "video game distribution platform",
              "id": "Q28057499",
              "label_en": "Facebook Gameroom"
            },
            {
              "description_en": "hosting service for freeware and commercial video games",
              "id": "Q28444637",
              "label_en": "Game Jolt"
            },
            {
              "description_en": "online computer game distribution platform for Nintendo Switch 1 and 2",
              "id": "Q3070866",
              "label_en": "Nintendo eShop"
            },
            {
              "description_en": "content delivery software by Electronic Arts",
              "id": "Q31708",
              "label_en": "Origin"
            },
            {
              "description_en": "video game store and digital distribution platform among other services",
              "id": "Q337535",
              "label_en": "Steam"
            },
            {
              "description_en": "digital app distribution platform for iOS/iPadOS",
              "id": "Q368215",
              "label_en": "앱 스토어"
            },
            {
              "description_en": "digital storefront for video games by Humble Bundle Inc.",
              "id": "Q42328566",
              "label_en": "Humble Store"
            },
            {
              "description_en": "app store",
              "id": "Q456078",
              "label_en": "Amazon Appstore"
            },
            {
              "description_en": "digital video game download service available through the Xbox Games Store; moniker for games released for download over the Xbox and Xbox 360",
              "id": "Q49612",
              "label_en": "Xbox Live Arcade"
            },
            {
              "description_en": "American online video game rental subscription service",
              "id": "Q5519774",
              "label_en": "GameFly"
            },
            {
              "description_en": "Digital distribution platform for computer games",
              "id": "Q5520058",
              "label_en": "GamersGate"
            },
            {
              "description_en": "online video game retailer",
              "id": "Q5602848",
              "label_en": "Green Man Gaming"
            },
            {
              "description_en": "digital media store",
              "id": "Q58379130",
              "label_en": "Nintendo Game Store"
            },
            {
              "description_en": "digital games storefront",
              "id": "Q59510068",
              "label_en": "Epic Games Store"
            },
            {
              "description_en": "video game platform",
              "id": "Q68307628",
              "label_en": "Rockstar Games Launcher"
            },
            "... omitted 10 items"
          ],
          "P6607": [
            {
              "value": "For video game distribution platforms use a more specific property like “distributed by” (P750) instead, not to be confused with “distribution format” (P437).@en"
            },
            {
              "value": "Pour les plateformes de distribution de jeux vidéo utilisez une propriété plus précise comme « distribué par » (P750), à ne pas confondre avec « format de distribution » (P437).@fr"
            }
          ],
          "P6824": [
            {
              "description_en": "distributor of a creative work; distributor for a record label; news agency; film distributor",
              "id": "P750",
              "label_en": "distributed by"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "personal computer operating system by Microsoft released in 2021",
              "id": "Q107269746",
              "label_en": "윈도우 11"
            },
            {
              "description_en": "personal computer operating system by Microsoft that was released in 2007",
              "id": "Q11230",
              "label_en": "Microsoft Windows Vista"
            },
            {
              "description_en": "personal computer operating system developed by Microsoft",
              "id": "Q11248",
              "label_en": "ويندوز إكس بي"
            },
            {
              "description_en": "personal computer operating system by Microsoft that was released in 2015",
              "id": "Q18168774",
              "label_en": "ویندوز ۱۰"
            },
            {
              "description_en": "personal computer operating system by Microsoft that was released in 2013",
              "id": "Q3569290",
              "label_en": "Windows 8.1"
            },
            {
              "description_en": "personal computer operating system by Microsoft",
              "id": "Q483132",
              "label_en": "Microsoft Windows 98"
            },
            {
              "description_en": "personal computer operating system by Microsoft",
              "id": "Q483881",
              "label_en": "Microsoft Windows 2000"
            },
            {
              "description_en": "personal computer operating system by Microsoft released in 2000",
              "id": "Q484892",
              "label_en": "Microsoft Windows Me"
            },
            {
              "description_en": "personal computer operating system by Microsoft that was released in 2012",
              "id": "Q5046",
              "label_en": "Ƿindoƿs 8"
            },
            {
              "description_en": "personal computer operating system by Microsoft",
              "id": "Q83370",
              "label_en": "Windows 95"
            }
          ],
          "P6607": [
            {
              "value": "use 'Microsoft Windows' instead; other values too specific@en"
            },
            {
              "value": "utiliser « Microsoft Windows » à la place ; les autres valeurs sont trop spécifiaues@fr"
            }
          ],
          "P9729": [
            {
              "description_en": "family of computer operating systems developed by Microsoft",
              "id": "Q1406",
              "label_en": "Microsoft Windows"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "handheld gaming PC developed by Valve",
              "id": "Q107542665",
              "label_en": "สตีมเด็ค"
            }
          ],
          "P6824": [
            {
              "description_en": "this work, product, object or standard can interact with another work, product, object or standard",
              "id": "P8956",
              "label_en": "compatible with"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "2023 virtual reality headset developed by Sony Interactive Entertainment",
              "id": "Q107944338",
              "label_en": "PlayStation VR2"
            },
            {
              "description_en": "virtual reality headset by Meta",
              "id": "Q119090835",
              "label_en": "Meta Quest 3"
            },
            {
              "description_en": "2016 virtual reality headset developed by Sony Interactive Entertainment",
              "id": "Q16011703",
              "label_en": "PlayStation VR"
            },
            {
              "description_en": "mobile virtual reality headset developed by Samsung Electronics",
              "id": "Q18031572",
              "label_en": "Gear VR"
            },
            {
              "description_en": "mixed reality computing platform by Microsoft",
              "id": "Q18844946",
              "label_en": "Windows Mixed Reality"
            },
            {
              "description_en": "virtual reality headset by Meta Platforms",
              "id": "Q3274429",
              "label_en": "Oculus Rift"
            },
            {
              "description_en": "2018 standalone virtual reality headset by Facebook Technologies",
              "id": "Q56353330",
              "label_en": "Oculus Go"
            },
            {
              "description_en": "untethered virtual reality headset by Meta Platforms",
              "id": "Q63777286",
              "label_en": "Meta Quest"
            },
            {
              "description_en": "virtual reality software",
              "id": "Q65164802",
              "label_en": "SteamVR"
            },
            {
              "description_en": "virtual reality headset by Facebook Technologies",
              "id": "Q99620218",
              "label_en": "Meta Quest 2"
            }
          ],
          "P6607": [
            {
              "value": "This constraint does not apply when used as an qualifier on Internet Game Database game ID (P5794)@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "video gaming in which the game runs on a remote cloud server",
              "id": "Q1102629",
              "label_en": "cloud gaming"
            },
            {
              "description_en": "premium subscription service offered by Sony as part of the PlayStation Network",
              "id": "Q15304599",
              "label_en": "PlayStation Plus"
            },
            {
              "description_en": "Sony PlayStation cloud gaming service",
              "id": "Q15614041",
              "label_en": "PlayStation Now"
            },
            {
              "description_en": "subscription-based video gaming service",
              "id": "Q17492436",
              "label_en": "EA Play"
            },
            {
              "description_en": "video game distribution platform",
              "id": "Q28057499",
              "label_en": "Facebook Gameroom"
            },
            {
              "description_en": "Nvidia's cloud-based game streaming service",
              "id": "Q28133964",
              "label_en": "GeForce Now"
            },
            {
              "description_en": "subscription service from Microsoft",
              "id": "Q29578331",
              "label_en": "Xbox Game Pass"
            },
            {
              "description_en": "paid online service for the Nintendo Switch and Nintendo Switch 2, which provides several functions",
              "id": "Q30943865",
              "label_en": "Nintendo Switch Online"
            },
            {
              "description_en": "video game subscription service by Apple",
              "id": "Q62513347",
              "label_en": "Apple Arcade"
            },
            {
              "description_en": "Xbox cloud game-streaming service by Microsoft",
              "id": "Q64487188",
              "label_en": "Xbox Cloud Gaming"
            },
            {
              "description_en": "type of online gaming service that runs games on remote servers and streams them directly to a user’s device",
              "id": "Q85632250",
              "label_en": "cloud gaming service"
            },
            {
              "description_en": "Mobile video game subscription service",
              "id": "Q96379060",
              "label_en": "GameClub"
            },
            {
              "description_en": "premium subscription service for PlayStation 5 offered by Sony Interactive Entertainment",
              "id": "Q99447158",
              "label_en": "PlayStation Plus Collection"
            },
            {
              "description_en": "cloud gaming and streaming service",
              "id": "Q99582374",
              "label_en": "Amazon Luna"
            },
            {
              "description_en": "Amazon Prime service",
              "id": "Q99740231",
              "label_en": "Prime Gaming"
            }
          ],
          "P6607": [
            {
              "value": "For cloud gaming and subscription services use a more specific property like “distributed by” (P750) instead, since they shouldn't be considered as gaming platforms - see discussion on Property talk:P400@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "virtual reality headset",
              "id": "Q115368568",
              "label_en": "Meta Quest Pro"
            },
            {
              "description_en": "virtual reality headset by Meta",
              "id": "Q130437846",
              "label_en": "Meta Quest 3S"
            },
            {
              "description_en": "computer-simulated environment simulating physical presence in real or imagined worlds",
              "id": "Q170519",
              "label_en": "virtual reality"
            },
            {
              "description_en": "virtual reality head-mounted smartphone mount made of cardboard, designed by Google",
              "id": "Q17558104",
              "label_en": "Google Cardboard"
            },
            {
              "description_en": "mixed reality device from Microsoft",
              "id": "Q18844389",
              "label_en": "Microsoft HoloLens"
            },
            {
              "description_en": "virtual reality headset produced by HTC and Valve",
              "id": "Q19414112",
              "label_en": "HTC Vive"
            },
            {
              "description_en": "head-mounted device that provides virtual reality for the wearer",
              "id": "Q19600329",
              "label_en": "virtual reality headset"
            },
            {
              "description_en": "open source project",
              "id": "Q20707759",
              "label_en": "Open Source Virtual Reality"
            },
            {
              "description_en": "virtual reality headset by Valve Corporation",
              "id": "Q63847959",
              "label_en": "Valve Index"
            },
            {
              "description_en": "virtual reality headset by Facebook Technologies",
              "id": "Q64021532",
              "label_en": "Oculus Rift S"
            },
            {
              "description_en": "Augmented reality headset by Microsoft",
              "id": "Q65077112",
              "label_en": "HoloLens 2"
            },
            {
              "description_en": "virtual reality headset released in 2016",
              "id": "Q69464718",
              "label_en": "Oculus Rift"
            },
            {
              "description_en": "2019 virtual reality headset by HTC",
              "id": "Q84453920",
              "label_en": "HTC Vive Cosmos"
            },
            {
              "description_en": "virtual reality headset HTC",
              "id": "Q84454740",
              "label_en": "HTC Vive Pro"
            }
          ],
          "P6607": [
            {
              "value": "For virtual reality headsets use a more specific property like \"output method\" (P5196) instead, as they aren't gaming platforms but output devices. Also it shouldn't be confused with the opposite property ”input method” (P479).@en"
            },
            {
              "value": "This constraint does not apply when used as an qualifier on Internet Game Database game ID (P5794)@en"
            }
          ],
          "P6824": [
            {
              "description_en": "output device used to interact with a software, video game console or video card",
              "id": "P5196",
              "label_en": "output device"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "video gaming brand owned by Sony Interactive Entertainment",
              "id": "Q1323662",
              "label_en": "PlayStation"
            }
          ],
          "P9729": [
            {
              "description_en": "1994 5th generation video game console by Sony Interactive Entertainment",
              "id": "Q10677",
              "label_en": "بلاي ستيشن"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "microconsole manufactured by Sony Computer Entertainment",
              "id": "Q14918005",
              "label_en": "PlayStation TV"
            }
          ],
          "P9729": [
            {
              "description_en": "portable game console developed by Sony Computer Entertainment",
              "id": "Q188808",
              "label_en": "PlayStation Vita"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "gaming Linux distribution",
              "id": "Q14944085",
              "label_en": "SteamOS"
            },
            {
              "description_en": "operating system based on GNU and the Linux kernel",
              "id": "Q3251801",
              "label_en": "GNU/Linukso"
            },
            {
              "description_en": "Linux distribution developed by Canonical",
              "id": "Q381",
              "label_en": "ኡቡንቱ"
            },
            {
              "description_en": "Unix-like operating system",
              "id": "Q44571",
              "label_en": "GNU"
            }
          ],
          "P9729": [
            {
              "description_en": "family of Unix-like operating systems",
              "id": "Q388",
              "label_en": "Linux"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "Microsoft's video gaming brand",
              "id": "Q15281614",
              "label_en": "إكس بوكس"
            }
          ],
          "P9729": [
            {
              "description_en": "video game console by Microsoft",
              "id": "Q132020",
              "label_en": "எக்ஸ் பாக்ஸ்"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "computer intended for use by an individual person",
              "id": "Q16338",
              "label_en": "personal computer"
            }
          ],
          "P6607": [
            {
              "value": "Personal computer is not a platform. Apple II, Mac, Amiga and Raspberry Pi 400 are personal computers too.@en"
            },
            {
              "value": "This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134611548@en"
            },
            {
              "value": "Персональный компьютер — не архитектура, платформа или что-то подобное. Apple II, Mac, Amiga и Raspberry Pi 400 — тоже персональные компьютеры.@ru"
            }
          ],
          "P9729": [
            {
              "description_en": "computers similar to the IBM PC and its derivatives",
              "id": "Q751046",
              "label_en": "IBM PC compatible"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "deprecated multimedia platform used to add animation and interactivity to web pages",
              "id": "Q165658",
              "label_en": "Adobe Flash"
            }
          ],
          "P6824": [
            {
              "description_en": "software engine employed by the subject item",
              "id": "P408",
              "label_en": "software engine"
            }
          ],
          "P9729": [
            {
              "description_en": "software application for retrieving, presenting, and traversing information resources on the World Wide Web",
              "id": "Q6368",
              "label_en": "web browser"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "portable device to make telephone calls using a radio link",
              "id": "Q17517",
              "label_en": "mobile phone"
            }
          ],
          "P6607": [
            {
              "value": "This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134621319@en"
            }
          ],
          "P9729": [
            {
              "description_en": "line of wireless handheld devices and services",
              "id": "Q171819",
              "label_en": "BlackBerry"
            },
            {
              "description_en": "computing platform",
              "id": "Q193828",
              "label_en": "Java Platform, Micro Edition"
            },
            {
              "description_en": "mobile operating system by Apple Inc.",
              "id": "Q48493",
              "label_en": "iOS"
            },
            {
              "description_en": "operating system for mobile devices created by Microsoft",
              "id": "Q4885200",
              "label_en": "Windows Phone"
            },
            {
              "description_en": "application development platform for mobile phones that was originally developed by Qualcomm Corp",
              "id": "Q863516",
              "label_en": "Binary Runtime Environment for Wireless"
            },
            {
              "description_en": "operating system created by Google for use in mobile devices",
              "id": "Q94",
              "label_en": "Android"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "Apple operating system family",
              "id": "Q43627",
              "label_en": "Mac OS operating systems"
            },
            {
              "description_en": "family of personal computers designed, manufactured, and sold by Apple Inc.",
              "id": "Q75687",
              "label_en": "Mac"
            }
          ],
          "P9729": [
            {
              "description_en": "Apple Macintosh’s original operating system (1984–2002)",
              "id": "Q13522376",
              "label_en": "Classic Mac OS"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "class of microcomputers of the 1980s, designed for private use at home; first type of computer ever which gained broad popularity amongst consumers, was replaced in the 1990s by personal computers with MS-DOS and later Microsoft Windows",
              "id": "Q473708",
              "label_en": "home computer"
            },
            {
              "description_en": "personal computer in a form intended for regular use at a single location desk/table",
              "id": "Q56155",
              "label_en": "desktop computer"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "Google cloud gaming service",
              "id": "Q60309635",
              "label_en": "Google Stadia"
            }
          ],
          "P6607": [
            {
              "value": "This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134611584@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "video game console developed by Microsoft",
              "id": "Q64513817",
              "label_en": "Xbox Series X"
            },
            {
              "description_en": "video game console developed by Microsoft",
              "id": "Q98967383",
              "label_en": "Xbox Series S"
            }
          ],
          "P6607": [
            {
              "value": "For the ninth generation of Xbox consoles use \"Xbox Series X and Series S \" (Q98973368) instead@en"
            }
          ],
          "P9729": [
            {
              "description_en": "home video game consoles developed by Microsoft",
              "id": "Q98973368",
              "label_en": "Xbox Series X and Series S"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "video game console developed by Microsoft; original version of the Xbox One console",
              "id": "Q66712951",
              "label_en": "Xbox One"
            }
          ],
          "P9729": [
            {
              "description_en": "home video game console developed by Microsoft",
              "id": "Q13361286",
              "label_en": "ایکس‌باکس وان"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      "... omitted 3 items"
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "set of episodes produced for an anime television series",
              "id": "Q100269041",
              "label_en": "anime television series season"
            },
            {
              "description_en": "series of light novels published in Japan",
              "id": "Q104213567",
              "label_en": "light novel series"
            },
            {
              "description_en": "anime released directly online",
              "id": "Q1047299",
              "label_en": "original net animation"
            },
            {
              "description_en": "television program",
              "id": "Q11086742",
              "label_en": "anime television program"
            },
            {
              "description_en": "use of a creative work across several different media",
              "id": "Q196600",
              "label_en": "media franchise"
            },
            {
              "description_en": "animated film from Japan or in Japanese anime style",
              "id": "Q20650540",
              "label_en": "anime film"
            },
            {
              "description_en": "series of comics employing Japanese stylistic conventions that are that are formally identified together",
              "id": "Q21198342",
              "label_en": "manga series"
            },
            {
              "description_en": "Japanese anime designed for release in home-video formats",
              "id": "Q220898",
              "label_en": "original video animation"
            },
            {
              "description_en": "connected set of television program episodes under the same title",
              "id": "Q5398426",
              "label_en": "television series"
            },
            {
              "description_en": "imaginary, typically self-consistent world with its own rules and characters, different from the real world; often used as a background or basis in story telling",
              "id": "Q559618",
              "label_en": "fictional universe"
            },
            {
              "description_en": "Japanese animated television series",
              "id": "Q63952888",
              "label_en": "anime television series"
            },
            {
              "description_en": "Japanese novella-type storytelling in conjunction with illustrations, geared toward young adults",
              "id": "Q747381",
              "label_en": "light novel"
            },
            {
              "description_en": "comics employing a set of Japanese stylistic conventions, produced in Japan or elsewhere",
              "id": "Q8274",
              "label_en": "manga"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "organization that provides access to the Internet",
              "id": "Q11371",
              "label_en": "internet service provider"
            },
            {
              "description_en": "organization that provides telephone and/or other telecommunications service",
              "id": "Q1266169",
              "label_en": "telephone company"
            },
            {
              "description_en": "cellular service provider",
              "id": "Q1941618",
              "label_en": "mobile network operator"
            },
            {
              "description_en": "financial institution that accepts deposits",
              "id": "Q22687",
              "label_en": "bank"
            },
            {
              "description_en": "social entity established to meet needs or pursue goals",
              "id": "Q43229",
              "label_en": "organization"
            },
            {
              "description_en": "organization undertaking commercial, industrial, or professional activity",
              "id": "Q4830453",
              "label_en": "business"
            },
            {
              "description_en": "organizational unit producing goods or services, which benefits from a certain degree of autonomy in decision-making, especially for the allocation of its current resources",
              "id": "Q6881511",
              "label_en": "enterprise"
            },
            {
              "description_en": "company that offers its securities for sale to the general public",
              "id": "Q891723",
              "label_en": "public company"
            },
            {
              "description_en": "specific type of business organization, which was granted a right to perform banking operations by Central Bank of Russia",
              "id": "Q93429702",
              "label_en": "credit organization (business in Russia)"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "portable device to make telephone calls using a radio link",
              "id": "Q17517",
              "label_en": "mobile phone"
            },
            {
              "description_en": "device that plays DVD discs",
              "id": "Q3783103",
              "label_en": "DVD player"
            },
            {
              "description_en": null,
              "id": "Q61448957",
              "label_en": "PlayStation Theme"
            },
            {
              "description_en": "theme for the PlayStation 4 SHAREfactory app",
              "id": "Q61449115",
              "label_en": "SHAREfactory Theme"
            }
          ],
          "P2304": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ],
          "P2308": [
            {
              "description_en": "electronic device connected via different wireless protocols to its environment",
              "id": "Q11253473",
              "label_en": "smart device"
            },
            {
              "description_en": "company that publishes video games",
              "id": "Q1137109",
              "label_en": "video game publisher"
            },
            {
              "description_en": "any set of video games",
              "id": "Q116741534",
              "label_en": "group of video games"
            },
            {
              "description_en": "group or company that develops applications",
              "id": "Q125251322",
              "label_en": "application developer"
            },
            {
              "description_en": "number of distinct pixels in each dimension that can be displayed",
              "id": "Q12538706",
              "label_en": "display resolution"
            },
            {
              "description_en": "book-length publication in digital form",
              "id": "Q128093",
              "label_en": "ebook"
            },
            {
              "description_en": "puzzles, board games, or video games based on language",
              "id": "Q15220419",
              "label_en": "word game"
            },
            {
              "description_en": "software interface between computers and/or programs",
              "id": "Q165194",
              "label_en": "application programming interface"
            },
            {
              "description_en": "literary genre consisting of works of literature that originate within digital environments and require digital computation",
              "id": "Q173167",
              "label_en": "electronic literature"
            },
            {
              "description_en": "coin-operated video game machine",
              "id": "Q192851",
              "label_en": "arcade video game machine"
            },
            {
              "description_en": "software development organization specializing in the creation of video games (for person use Q58287519)",
              "id": "Q210167",
              "label_en": "video game developer"
            },
            {
              "description_en": "album type consisting of works chosen for a particular purpose or theme",
              "id": "Q222910",
              "label_en": "compilation album"
            },
            {
              "description_en": "any type of game under a common theme or story",
              "id": "Q28114058",
              "label_en": "game franchise"
            },
            {
              "description_en": "meta-goal in a video game",
              "id": "Q2988681",
              "label_en": "achievement"
            },
            {
              "description_en": "sequence of instructions written in programming language to perform a specified task with a computer",
              "id": "Q40056",
              "label_en": "computer program"
            },
            {
              "description_en": "fictional videogame in a narrative form of arts",
              "id": "Q40213094",
              "label_en": "fictional video game"
            },
            {
              "description_en": "electronic magazine to be read using computers",
              "id": "Q416",
              "label_en": "disk magazine"
            },
            {
              "description_en": "means by which a user interacts with and controls a machine",
              "id": "Q47146",
              "label_en": "user interface"
            },
            {
              "description_en": "group of people in the demoscene",
              "id": "Q5256141",
              "label_en": "demogroup"
            },
            {
              "description_en": "learning method",
              "id": "Q535741",
              "label_en": "tutorial"
            },
            {
              "description_en": "art software with graphics, music and animation",
              "id": "Q5610543",
              "label_en": "demo"
            },
            {
              "description_en": "compilation of software, in most cases, from the same developer",
              "id": "Q62651817",
              "label_en": "software bundle"
            },
            {
              "description_en": "non-tangible executable component of a computer",
              "id": "Q7397",
              "label_en": "software"
            },
            {
              "description_en": "electronic game with user interface and visual feedback",
              "id": "Q7889",
              "label_en": "video game"
            },
            "... omitted 5 items"
          ],
          "P2309": [
            {
              "description_en": "relation of type or value type constraint",
              "id": "Q30208840",
              "label_en": "instance or subclass of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "any type of firmware used in a computer for low-level hardware initialization during the boot process",
              "id": "Q102676349",
              "label_en": "computer firmware"
            },
            {
              "description_en": "Apple Macintosh’s original operating system (1984–2002)",
              "id": "Q13522376",
              "label_en": "Classic Mac OS"
            },
            {
              "description_en": "family of computer operating systems developed by Microsoft",
              "id": "Q1406",
              "label_en": "Microsoft Windows"
            },
            {
              "description_en": "operating system for Apple computers",
              "id": "Q14116",
              "label_en": "macOS"
            },
            {
              "description_en": "group of closely-related PC-compatible operating systems",
              "id": "Q170434",
              "label_en": "DOS"
            },
            {
              "description_en": "set of rules and methods that describe the functionality, organization and implementation of computer systems",
              "id": "Q173212",
              "label_en": "computer architecture"
            },
            {
              "description_en": "environment with a graphical user interface using the desktop metaphor",
              "id": "Q205020",
              "label_en": "desktop environment"
            },
            {
              "description_en": "environment in which a piece of software is executed",
              "id": "Q241317",
              "label_en": "computing platform"
            },
            {
              "description_en": "speaker with features or services that go beyond audio playback",
              "id": "Q26884850",
              "label_en": "smart speaker"
            },
            {
              "description_en": "software for creating online stores",
              "id": "Q2916479",
              "label_en": "shopping cart software"
            },
            {
              "description_en": "family of Unix-like operating systems",
              "id": "Q388",
              "label_en": "Linux"
            },
            {
              "description_en": "online publishing environment",
              "id": "Q4202064",
              "label_en": "publishing platform"
            },
            {
              "description_en": "mobile operating system by Apple Inc.",
              "id": "Q48493",
              "label_en": "iOS"
            },
            {
              "description_en": "type of operating system",
              "id": "Q600659",
              "label_en": "disk operating system"
            },
            {
              "description_en": "video game console in a narrative form of arts",
              "id": "Q60644919",
              "label_en": "fictional video game console"
            },
            {
              "description_en": "software application for retrieving, presenting, and traversing information resources on the World Wide Web",
              "id": "Q6368",
              "label_en": "web browser"
            },
            {
              "description_en": "interactive entertainment computer or customized computer system for running video games",
              "id": "Q8076",
              "label_en": "video game console"
            },
            {
              "description_en": "software that manages computer hardware resources",
              "id": "Q9135",
              "label_en": "operating system"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type or value type constraint",
              "id": "Q30208840",
              "label_en": "instance or subclass of"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ],
          "P6607": [
            {
              "value": "see Wikidata:Project_chat/Archive/2018/06#relation_between_platform_(P400)_and_operating_system_(P306)@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "1986 video game released by Activision",
              "id": "Q3613155",
              "label_en": "Alter Ego"
            }
          ],
          "P2305": [
            {
              "description_en": "line of smartphones developed and marketed by Apple Inc.",
              "id": "Q2766",
              "label_en": "آيفون"
            },
            {
              "description_en": "line of tablet computers developed by Apple",
              "id": "Q2796",
              "label_en": "آي باد"
            },
            {
              "description_en": "line of portable media players by Apple Inc.",
              "id": "Q9479",
              "label_en": "IPod"
            }
          ],
          "P6607": [
            {
              "value": "For Apple smartphones use \"iOS \" (Q48493) instead@en"
            },
            {
              "value": "Pour les téléphones intelligents d'Apple, utiliser « iOS » (Q48493) à la place@fr"
            }
          ],
          "P9729": [
            {
              "description_en": "mobile operating system by Apple Inc.",
              "id": "Q48493",
              "label_en": "iOS"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "digital media store",
              "id": "Q1052025",
              "label_en": "PlayStation Store"
            },
            {
              "description_en": "virtual market designed for Xbox",
              "id": "Q1204827",
              "label_en": "Xbox Games Store"
            },
            {
              "description_en": "digital distribution platform operated by Microsoft",
              "id": "Q135288",
              "label_en": "Microsoft Store"
            },
            {
              "description_en": "video game platform",
              "id": "Q1486288",
              "label_en": "Good Old Games"
            },
            {
              "description_en": null,
              "id": "Q16991786",
              "label_en": "Amazon Digital Game Store"
            },
            {
              "description_en": "game distribution platform",
              "id": "Q17238705",
              "label_en": "Playism"
            },
            {
              "description_en": "adult video game distribution platform",
              "id": "Q22079967",
              "label_en": "Nutaku"
            },
            {
              "description_en": "website for distributing video games",
              "id": "Q22905933",
              "label_en": "itch.io"
            },
            {
              "description_en": "instant messaging and VoIP software",
              "id": "Q22907849",
              "label_en": "Discord"
            },
            {
              "description_en": "video game distribution platform",
              "id": "Q28057499",
              "label_en": "Facebook Gameroom"
            },
            {
              "description_en": "hosting service for freeware and commercial video games",
              "id": "Q28444637",
              "label_en": "Game Jolt"
            },
            {
              "description_en": "online computer game distribution platform for Nintendo Switch 1 and 2",
              "id": "Q3070866",
              "label_en": "Nintendo eShop"
            },
            {
              "description_en": "content delivery software by Electronic Arts",
              "id": "Q31708",
              "label_en": "Origin"
            },
            {
              "description_en": "video game store and digital distribution platform among other services",
              "id": "Q337535",
              "label_en": "Steam"
            },
            {
              "description_en": "digital app distribution platform for iOS/iPadOS",
              "id": "Q368215",
              "label_en": "앱 스토어"
            },
            {
              "description_en": "digital storefront for video games by Humble Bundle Inc.",
              "id": "Q42328566",
              "label_en": "Humble Store"
            },
            {
              "description_en": "app store",
              "id": "Q456078",
              "label_en": "Amazon Appstore"
            },
            {
              "description_en": "digital video game download service available through the Xbox Games Store; moniker for games released for download over the Xbox and Xbox 360",
              "id": "Q49612",
              "label_en": "Xbox Live Arcade"
            },
            {
              "description_en": "American online video game rental subscription service",
              "id": "Q5519774",
              "label_en": "GameFly"
            },
            {
              "description_en": "Digital distribution platform for computer games",
              "id": "Q5520058",
              "label_en": "GamersGate"
            },
            {
              "description_en": "online video game retailer",
              "id": "Q5602848",
              "label_en": "Green Man Gaming"
            },
            {
              "description_en": "digital media store",
              "id": "Q58379130",
              "label_en": "Nintendo Game Store"
            },
            {
              "description_en": "digital games storefront",
              "id": "Q59510068",
              "label_en": "Epic Games Store"
            },
            {
              "description_en": "video game platform",
              "id": "Q68307628",
              "label_en": "Rockstar Games Launcher"
            },
            "... omitted 10 items"
          ],
          "P6607": [
            {
              "value": "For video game distribution platforms use a more specific property like “distributed by” (P750) instead, not to be confused with “distribution format” (P437).@en"
            },
            {
              "value": "Pour les plateformes de distribution de jeux vidéo utilisez une propriété plus précise comme « distribué par » (P750), à ne pas confondre avec « format de distribution » (P437).@fr"
            }
          ],
          "P6824": [
            {
              "description_en": "distributor of a creative work; distributor for a record label; news agency; film distributor",
              "id": "P750",
              "label_en": "distributed by"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "personal computer operating system by Microsoft released in 2021",
              "id": "Q107269746",
              "label_en": "윈도우 11"
            },
            {
              "description_en": "personal computer operating system by Microsoft that was released in 2007",
              "id": "Q11230",
              "label_en": "Microsoft Windows Vista"
            },
            {
              "description_en": "personal computer operating system developed by Microsoft",
              "id": "Q11248",
              "label_en": "ويندوز إكس بي"
            },
            {
              "description_en": "personal computer operating system by Microsoft that was released in 2015",
              "id": "Q18168774",
              "label_en": "ویندوز ۱۰"
            },
            {
              "description_en": "personal computer operating system by Microsoft that was released in 2013",
              "id": "Q3569290",
              "label_en": "Windows 8.1"
            },
            {
              "description_en": "personal computer operating system by Microsoft",
              "id": "Q483132",
              "label_en": "Microsoft Windows 98"
            },
            {
              "description_en": "personal computer operating system by Microsoft",
              "id": "Q483881",
              "label_en": "Microsoft Windows 2000"
            },
            {
              "description_en": "personal computer operating system by Microsoft released in 2000",
              "id": "Q484892",
              "label_en": "Microsoft Windows Me"
            },
            {
              "description_en": "personal computer operating system by Microsoft that was released in 2012",
              "id": "Q5046",
              "label_en": "Ƿindoƿs 8"
            },
            {
              "description_en": "personal computer operating system by Microsoft",
              "id": "Q83370",
              "label_en": "Windows 95"
            }
          ],
          "P6607": [
            {
              "value": "use 'Microsoft Windows' instead; other values too specific@en"
            },
            {
              "value": "utiliser « Microsoft Windows » à la place ; les autres valeurs sont trop spécifiaues@fr"
            }
          ],
          "P9729": [
            {
              "description_en": "family of computer operating systems developed by Microsoft",
              "id": "Q1406",
              "label_en": "Microsoft Windows"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "handheld gaming PC developed by Valve",
              "id": "Q107542665",
              "label_en": "สตีมเด็ค"
            }
          ],
          "P6824": [
            {
              "description_en": "this work, product, object or standard can interact with another work, product, object or standard",
              "id": "P8956",
              "label_en": "compatible with"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "2023 virtual reality headset developed by Sony Interactive Entertainment",
              "id": "Q107944338",
              "label_en": "PlayStation VR2"
            },
            {
              "description_en": "virtual reality headset by Meta",
              "id": "Q119090835",
              "label_en": "Meta Quest 3"
            },
            {
              "description_en": "2016 virtual reality headset developed by Sony Interactive Entertainment",
              "id": "Q16011703",
              "label_en": "PlayStation VR"
            },
            {
              "description_en": "mobile virtual reality headset developed by Samsung Electronics",
              "id": "Q18031572",
              "label_en": "Gear VR"
            },
            {
              "description_en": "mixed reality computing platform by Microsoft",
              "id": "Q18844946",
              "label_en": "Windows Mixed Reality"
            },
            {
              "description_en": "virtual reality headset by Meta Platforms",
              "id": "Q3274429",
              "label_en": "Oculus Rift"
            },
            {
              "description_en": "2018 standalone virtual reality headset by Facebook Technologies",
              "id": "Q56353330",
              "label_en": "Oculus Go"
            },
            {
              "description_en": "untethered virtual reality headset by Meta Platforms",
              "id": "Q63777286",
              "label_en": "Meta Quest"
            },
            {
              "description_en": "virtual reality software",
              "id": "Q65164802",
              "label_en": "SteamVR"
            },
            {
              "description_en": "virtual reality headset by Facebook Technologies",
              "id": "Q99620218",
              "label_en": "Meta Quest 2"
            }
          ],
          "P6607": [
            {
              "value": "This constraint does not apply when used as an qualifier on Internet Game Database game ID (P5794)@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "video gaming in which the game runs on a remote cloud server",
              "id": "Q1102629",
              "label_en": "cloud gaming"
            },
            {
              "description_en": "premium subscription service offered by Sony as part of the PlayStation Network",
              "id": "Q15304599",
              "label_en": "PlayStation Plus"
            },
            {
              "description_en": "Sony PlayStation cloud gaming service",
              "id": "Q15614041",
              "label_en": "PlayStation Now"
            },
            {
              "description_en": "subscription-based video gaming service",
              "id": "Q17492436",
              "label_en": "EA Play"
            },
            {
              "description_en": "video game distribution platform",
              "id": "Q28057499",
              "label_en": "Facebook Gameroom"
            },
            {
              "description_en": "Nvidia's cloud-based game streaming service",
              "id": "Q28133964",
              "label_en": "GeForce Now"
            },
            {
              "description_en": "subscription service from Microsoft",
              "id": "Q29578331",
              "label_en": "Xbox Game Pass"
            },
            {
              "description_en": "paid online service for the Nintendo Switch and Nintendo Switch 2, which provides several functions",
              "id": "Q30943865",
              "label_en": "Nintendo Switch Online"
            },
            {
              "description_en": "video game subscription service by Apple",
              "id": "Q62513347",
              "label_en": "Apple Arcade"
            },
            {
              "description_en": "Xbox cloud game-streaming service by Microsoft",
              "id": "Q64487188",
              "label_en": "Xbox Cloud Gaming"
            },
            {
              "description_en": "type of online gaming service that runs games on remote servers and streams them directly to a user’s device",
              "id": "Q85632250",
              "label_en": "cloud gaming service"
            },
            {
              "description_en": "Mobile video game subscription service",
              "id": "Q96379060",
              "label_en": "GameClub"
            },
            {
              "description_en": "premium subscription service for PlayStation 5 offered by Sony Interactive Entertainment",
              "id": "Q99447158",
              "label_en": "PlayStation Plus Collection"
            },
            {
              "description_en": "cloud gaming and streaming service",
              "id": "Q99582374",
              "label_en": "Amazon Luna"
            },
            {
              "description_en": "Amazon Prime service",
              "id": "Q99740231",
              "label_en": "Prime Gaming"
            }
          ],
          "P6607": [
            {
              "value": "For cloud gaming and subscription services use a more specific property like “distributed by” (P750) instead, since they shouldn't be considered as gaming platforms - see discussion on Property talk:P400@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "virtual reality headset",
              "id": "Q115368568",
              "label_en": "Meta Quest Pro"
            },
            {
              "description_en": "virtual reality headset by Meta",
              "id": "Q130437846",
              "label_en": "Meta Quest 3S"
            },
            {
              "description_en": "computer-simulated environment simulating physical presence in real or imagined worlds",
              "id": "Q170519",
              "label_en": "virtual reality"
            },
            {
              "description_en": "virtual reality head-mounted smartphone mount made of cardboard, designed by Google",
              "id": "Q17558104",
              "label_en": "Google Cardboard"
            },
            {
              "description_en": "mixed reality device from Microsoft",
              "id": "Q18844389",
              "label_en": "Microsoft HoloLens"
            },
            {
              "description_en": "virtual reality headset produced by HTC and Valve",
              "id": "Q19414112",
              "label_en": "HTC Vive"
            },
            {
              "description_en": "head-mounted device that provides virtual reality for the wearer",
              "id": "Q19600329",
              "label_en": "virtual reality headset"
            },
            {
              "description_en": "open source project",
              "id": "Q20707759",
              "label_en": "Open Source Virtual Reality"
            },
            {
              "description_en": "virtual reality headset by Valve Corporation",
              "id": "Q63847959",
              "label_en": "Valve Index"
            },
            {
              "description_en": "virtual reality headset by Facebook Technologies",
              "id": "Q64021532",
              "label_en": "Oculus Rift S"
            },
            {
              "description_en": "Augmented reality headset by Microsoft",
              "id": "Q65077112",
              "label_en": "HoloLens 2"
            },
            {
              "description_en": "virtual reality headset released in 2016",
              "id": "Q69464718",
              "label_en": "Oculus Rift"
            },
            {
              "description_en": "2019 virtual reality headset by HTC",
              "id": "Q84453920",
              "label_en": "HTC Vive Cosmos"
            },
            {
              "description_en": "virtual reality headset HTC",
              "id": "Q84454740",
              "label_en": "HTC Vive Pro"
            }
          ],
          "P6607": [
            {
              "value": "For virtual reality headsets use a more specific property like \"output method\" (P5196) instead, as they aren't gaming platforms but output devices. Also it shouldn't be confused with the opposite property ”input method” (P479).@en"
            },
            {
              "value": "This constraint does not apply when used as an qualifier on Internet Game Database game ID (P5794)@en"
            }
          ],
          "P6824": [
            {
              "description_en": "output device used to interact with a software, video game console or video card",
              "id": "P5196",
              "label_en": "output device"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "video gaming brand owned by Sony Interactive Entertainment",
              "id": "Q1323662",
              "label_en": "PlayStation"
            }
          ],
          "P9729": [
            {
              "description_en": "1994 5th generation video game console by Sony Interactive Entertainment",
              "id": "Q10677",
              "label_en": "بلاي ستيشن"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "microconsole manufactured by Sony Computer Entertainment",
              "id": "Q14918005",
              "label_en": "PlayStation TV"
            }
          ],
          "P9729": [
            {
              "description_en": "portable game console developed by Sony Computer Entertainment",
              "id": "Q188808",
              "label_en": "PlayStation Vita"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "gaming Linux distribution",
              "id": "Q14944085",
              "label_en": "SteamOS"
            },
            {
              "description_en": "operating system based on GNU and the Linux kernel",
              "id": "Q3251801",
              "label_en": "GNU/Linukso"
            },
            {
              "description_en": "Linux distribution developed by Canonical",
              "id": "Q381",
              "label_en": "ኡቡንቱ"
            },
            {
              "description_en": "Unix-like operating system",
              "id": "Q44571",
              "label_en": "GNU"
            }
          ],
          "P9729": [
            {
              "description_en": "family of Unix-like operating systems",
              "id": "Q388",
              "label_en": "Linux"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "Microsoft's video gaming brand",
              "id": "Q15281614",
              "label_en": "إكس بوكس"
            }
          ],
          "P9729": [
            {
              "description_en": "video game console by Microsoft",
              "id": "Q132020",
              "label_en": "எக்ஸ் பாக்ஸ்"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "computer intended for use by an individual person",
              "id": "Q16338",
              "label_en": "personal computer"
            }
          ],
          "P6607": [
            {
              "value": "Personal computer is not a platform. Apple II, Mac, Amiga and Raspberry Pi 400 are personal computers too.@en"
            },
            {
              "value": "This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134611548@en"
            },
            {
              "value": "Персональный компьютер — не архитектура, платформа или что-то подобное. Apple II, Mac, Amiga и Raspberry Pi 400 — тоже персональные компьютеры.@ru"
            }
          ],
          "P9729": [
            {
              "description_en": "computers similar to the IBM PC and its derivatives",
              "id": "Q751046",
              "label_en": "IBM PC compatible"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "deprecated multimedia platform used to add animation and interactivity to web pages",
              "id": "Q165658",
              "label_en": "Adobe Flash"
            }
          ],
          "P6824": [
            {
              "description_en": "software engine employed by the subject item",
              "id": "P408",
              "label_en": "software engine"
            }
          ],
          "P9729": [
            {
              "description_en": "software application for retrieving, presenting, and traversing information resources on the World Wide Web",
              "id": "Q6368",
              "label_en": "web browser"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "portable device to make telephone calls using a radio link",
              "id": "Q17517",
              "label_en": "mobile phone"
            }
          ],
          "P6607": [
            {
              "value": "This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134621319@en"
            }
          ],
          "P9729": [
            {
              "description_en": "line of wireless handheld devices and services",
              "id": "Q171819",
              "label_en": "BlackBerry"
            },
            {
              "description_en": "computing platform",
              "id": "Q193828",
              "label_en": "Java Platform, Micro Edition"
            },
            {
              "description_en": "mobile operating system by Apple Inc.",
              "id": "Q48493",
              "label_en": "iOS"
            },
            {
              "description_en": "operating system for mobile devices created by Microsoft",
              "id": "Q4885200",
              "label_en": "Windows Phone"
            },
            {
              "description_en": "operating system created by Google for use in mobile devices",
              "id": "Q94",
              "label_en": "Android"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "Apple operating system family",
              "id": "Q43627",
              "label_en": "Mac OS operating systems"
            },
            {
              "description_en": "family of personal computers designed, manufactured, and sold by Apple Inc.",
              "id": "Q75687",
              "label_en": "Mac"
            }
          ],
          "P9729": [
            {
              "description_en": "Apple Macintosh’s original operating system (1984–2002)",
              "id": "Q13522376",
              "label_en": "Classic Mac OS"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "class of microcomputers of the 1980s, designed for private use at home; first type of computer ever which gained broad popularity amongst consumers, was replaced in the 1990s by personal computers with MS-DOS and later Microsoft Windows",
              "id": "Q473708",
              "label_en": "home computer"
            },
            {
              "description_en": "personal computer in a form intended for regular use at a single location desk/table",
              "id": "Q56155",
              "label_en": "desktop computer"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "Google cloud gaming service",
              "id": "Q60309635",
              "label_en": "Google Stadia"
            }
          ],
          "P6607": [
            {
              "value": "This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134611584@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "video game console developed by Microsoft",
              "id": "Q64513817",
              "label_en": "Xbox Series X"
            },
            {
              "description_en": "video game console developed by Microsoft",
              "id": "Q98967383",
              "label_en": "Xbox Series S"
            }
          ],
          "P6607": [
            {
              "value": "For the ninth generation of Xbox consoles use \"Xbox Series X and Series S \" (Q98973368) instead@en"
            }
          ],
          "P9729": [
            {
              "description_en": "home video game consoles developed by Microsoft",
              "id": "Q98973368",
              "label_en": "Xbox Series X and Series S"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "video game console developed by Microsoft; original version of the Xbox One console",
              "id": "Q66712951",
              "label_en": "Xbox One"
            }
          ],
          "P9729": [
            {
              "description_en": "home video game console developed by Microsoft",
              "id": "Q13361286",
              "label_en": "ایکس‌باکس وان"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      "... omitted 3 items"
    ]
  },
  "hash_after": "fd8998e99a76e16dd5985b5dae7f07e6b892dde1",
  "hash_before": "2a0fcd019beedf9ea03197f3bcf9821f5dacfda3",
  "property_revision_id": 2355127724,
  "property_revision_prev": 2355127546,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q863516"
      ],
      "constraint_qid": "Q52558054",
      "qualifier_property": "P9729",
      "removed_values": [],
      "same_qid_index": 13
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q52558054",
      "qualifiers": [
        {
          "property_id": "P2305",
          "values": [
            "Q17517"
          ]
        },
        {
          "property_id": "P6607",
          "values": [
            "This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134621319@en"
          ]
        },
        {
          "property_id": "P9729",
          "values": [
            "Q171819",
            "Q193828",
            "Q48493",
            "Q4885200",
            "Q94"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "conflicts-with constraint: item of property constraint: anime television series season, light novel series, original net animation, anime television program, media franchise, anime film, manga series, original video animation, television series, fictional universe, anime television series, light novel, manga; property: instance of",
      "conflicts-with constraint: item of property constraint: internet service provider, telephone company, mobile network operator, bank, organization, business, enterprise, public company, credit organization (business in Russia); property: instance of",
      "subject type constraint: exception to constraint: mobile phone, DVD player, PlayStation Theme, SHAREfactory Theme; group by: instance of; class: smart device, video game publisher, group of video games, application developer, display resolution, ebook, word game, application programming interface, electronic literature, arcade video game machine, video game developer, compilation album, game franchise, achievement, computer program, fictional video game, disk maga... [truncated 211 chars]",
      "value-type constraint: class: computer firmware, Classic Mac OS, Microsoft Windows, macOS, DOS, computer architecture, desktop environment, computing platform, smart speaker, shopping cart software, Linux, publishing platform, iOS, disk operating system, fictional video game console, web browser, video game console, operating system; relation: instance or subclass of; constraint status: suggestion constraint; constraint clarification: see Wikidata:Project_chat/Arc... [truncated 76 chars]",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "none-of constraint: exception to constraint: Alter Ego; item of property constraint: آيفون, آي باد, IPod; constraint clarification: For Apple smartphones use \"iOS \" (Q48493) instead@en, Pour les téléphones intelligents d'Apple, utiliser « iOS » (Q48493) à la place@fr; replacement value: iOS",
      "none-of constraint: item of property constraint: PlayStation Store, Xbox Games Store, Microsoft Store, Good Old Games, Amazon Digital Game Store, Playism, Nutaku, itch.io, Discord, Facebook Gameroom, Game Jolt, Nintendo eShop, Origin, Steam, 앱 스토어, Humble Store, Amazon Appstore, Xbox Live Arcade, GameFly, GamersGate, Green Man Gaming, Nintendo Game Store, Epic Games Store, Rockstar Games Launcher, PlayStation Network, Pokki, Battle.net, يوبلاي, Google Play, Groupe... [truncated 466 chars]",
      "none-of constraint: item of property constraint: 윈도우 11, Microsoft Windows Vista, ويندوز إكس بي, ویندوز ۱۰, Windows 8.1, Microsoft Windows 98, Microsoft Windows 2000, Microsoft Windows Me, Ƿindoƿs 8, Windows 95; constraint clarification: use 'Microsoft Windows' instead; other values too specific@en, utiliser « Microsoft Windows » à la place ; les autres valeurs sont trop spécifiaues@fr; replacement value: Microsoft Windows",
      "none-of constraint: item of property constraint: สตีมเด็ค; replacement property: compatible with",
      "none-of constraint: item of property constraint: PlayStation VR2, Meta Quest 3, PlayStation VR, Gear VR, Windows Mixed Reality, Oculus Rift, Oculus Go, Meta Quest, SteamVR, Meta Quest 2; constraint clarification: This constraint does not apply when used as an qualifier on Internet Game Database game ID (P5794)@en",
      "none-of constraint: item of property constraint: cloud gaming, PlayStation Plus, PlayStation Now, EA Play, Facebook Gameroom, GeForce Now, Xbox Game Pass, Nintendo Switch Online, Apple Arcade, Xbox Cloud Gaming, cloud gaming service, GameClub, PlayStation Plus Collection, Amazon Luna, Prime Gaming; constraint clarification: For cloud gaming and subscription services use a more specific property like “distributed by” (P750) instead, since they shouldn't be consider... [truncated 64 chars]",
      "none-of constraint: item of property constraint: Meta Quest Pro, Meta Quest 3S, virtual reality, Google Cardboard, Microsoft HoloLens, HTC Vive, virtual reality headset, Open Source Virtual Reality, Valve Index, Oculus Rift S, HoloLens 2, Oculus Rift, HTC Vive Cosmos, HTC Vive Pro; constraint clarification: For virtual reality headsets use a more specific property like \"output method\" (P5196) instead, as they aren't gaming platforms but output devices. Also it sho... [truncated 211 chars]",
      "none-of constraint: item of property constraint: PlayStation; replacement value: بلاي ستيشن",
      "none-of constraint: item of property constraint: PlayStation TV; replacement value: PlayStation Vita",
      "none-of constraint: item of property constraint: SteamOS, GNU/Linukso, ኡቡንቱ, GNU; replacement value: Linux",
      "none-of constraint: item of property constraint: إكس بوكس; replacement value: எக்ஸ் பாக்ஸ்",
      "none-of constraint: item of property constraint: personal computer; constraint clarification: Personal computer is not a platform. Apple II, Mac, Amiga and Raspberry Pi 400 are personal computers too.@en, This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134611548@en, Персональный компьютер — не архитектура, платформа или что-то подобное. Apple II, Mac, Amiga и Raspberry Pi 400 — тоже персональные компьютеры.@ru; re... [truncated 34 chars]",
      "none-of constraint: item of property constraint: Adobe Flash; replacement property: software engine; replacement value: web browser",
      "none-of constraint: item of property constraint: mobile phone; constraint clarification: This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134621319@en; replacement value: BlackBerry, Java Platform, Micro Edition, iOS, Windows Phone, Binary Runtime Environment for Wireless, Android",
      "none-of constraint: item of property constraint: Mac OS operating systems, Mac; replacement value: Classic Mac OS",
      "none-of constraint: item of property constraint: home computer, desktop computer",
      "none-of constraint: item of property constraint: Google Stadia; constraint clarification: This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134611584@en",
      "none-of constraint: item of property constraint: Xbox Series X, Xbox Series S; constraint clarification: For the ninth generation of Xbox consoles use \"Xbox Series X and Series S \" (Q98973368) instead@en; replacement value: Xbox Series X and Series S",
      "none-of constraint: item of property constraint: Xbox One; replacement value: ایکس‌باکس وان",
      "... omitted 3 items"
    ],
    "before": [
      "conflicts-with constraint: item of property constraint: anime television series season, light novel series, original net animation, anime television program, media franchise, anime film, manga series, original video animation, television series, fictional universe, anime television series, light novel, manga; property: instance of",
      "conflicts-with constraint: item of property constraint: internet service provider, telephone company, mobile network operator, bank, organization, business, enterprise, public company, credit organization (business in Russia); property: instance of",
      "subject type constraint: exception to constraint: mobile phone, DVD player, PlayStation Theme, SHAREfactory Theme; group by: instance of; class: smart device, video game publisher, group of video games, application developer, display resolution, ebook, word game, application programming interface, electronic literature, arcade video game machine, video game developer, compilation album, game franchise, achievement, computer program, fictional video game, disk maga... [truncated 211 chars]",
      "value-type constraint: class: computer firmware, Classic Mac OS, Microsoft Windows, macOS, DOS, computer architecture, desktop environment, computing platform, smart speaker, shopping cart software, Linux, publishing platform, iOS, disk operating system, fictional video game console, web browser, video game console, operating system; relation: instance or subclass of; constraint status: suggestion constraint; constraint clarification: see Wikidata:Project_chat/Arc... [truncated 76 chars]",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "none-of constraint: exception to constraint: Alter Ego; item of property constraint: آيفون, آي باد, IPod; constraint clarification: For Apple smartphones use \"iOS \" (Q48493) instead@en, Pour les téléphones intelligents d'Apple, utiliser « iOS » (Q48493) à la place@fr; replacement value: iOS",
      "none-of constraint: item of property constraint: PlayStation Store, Xbox Games Store, Microsoft Store, Good Old Games, Amazon Digital Game Store, Playism, Nutaku, itch.io, Discord, Facebook Gameroom, Game Jolt, Nintendo eShop, Origin, Steam, 앱 스토어, Humble Store, Amazon Appstore, Xbox Live Arcade, GameFly, GamersGate, Green Man Gaming, Nintendo Game Store, Epic Games Store, Rockstar Games Launcher, PlayStation Network, Pokki, Battle.net, يوبلاي, Google Play, Groupe... [truncated 466 chars]",
      "none-of constraint: item of property constraint: 윈도우 11, Microsoft Windows Vista, ويندوز إكس بي, ویندوز ۱۰, Windows 8.1, Microsoft Windows 98, Microsoft Windows 2000, Microsoft Windows Me, Ƿindoƿs 8, Windows 95; constraint clarification: use 'Microsoft Windows' instead; other values too specific@en, utiliser « Microsoft Windows » à la place ; les autres valeurs sont trop spécifiaues@fr; replacement value: Microsoft Windows",
      "none-of constraint: item of property constraint: สตีมเด็ค; replacement property: compatible with",
      "none-of constraint: item of property constraint: PlayStation VR2, Meta Quest 3, PlayStation VR, Gear VR, Windows Mixed Reality, Oculus Rift, Oculus Go, Meta Quest, SteamVR, Meta Quest 2; constraint clarification: This constraint does not apply when used as an qualifier on Internet Game Database game ID (P5794)@en",
      "none-of constraint: item of property constraint: cloud gaming, PlayStation Plus, PlayStation Now, EA Play, Facebook Gameroom, GeForce Now, Xbox Game Pass, Nintendo Switch Online, Apple Arcade, Xbox Cloud Gaming, cloud gaming service, GameClub, PlayStation Plus Collection, Amazon Luna, Prime Gaming; constraint clarification: For cloud gaming and subscription services use a more specific property like “distributed by” (P750) instead, since they shouldn't be consider... [truncated 64 chars]",
      "none-of constraint: item of property constraint: Meta Quest Pro, Meta Quest 3S, virtual reality, Google Cardboard, Microsoft HoloLens, HTC Vive, virtual reality headset, Open Source Virtual Reality, Valve Index, Oculus Rift S, HoloLens 2, Oculus Rift, HTC Vive Cosmos, HTC Vive Pro; constraint clarification: For virtual reality headsets use a more specific property like \"output method\" (P5196) instead, as they aren't gaming platforms but output devices. Also it sho... [truncated 211 chars]",
      "none-of constraint: item of property constraint: PlayStation; replacement value: بلاي ستيشن",
      "none-of constraint: item of property constraint: PlayStation TV; replacement value: PlayStation Vita",
      "none-of constraint: item of property constraint: SteamOS, GNU/Linukso, ኡቡንቱ, GNU; replacement value: Linux",
      "none-of constraint: item of property constraint: إكس بوكس; replacement value: எக்ஸ் பாக்ஸ்",
      "none-of constraint: item of property constraint: personal computer; constraint clarification: Personal computer is not a platform. Apple II, Mac, Amiga and Raspberry Pi 400 are personal computers too.@en, This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134611548@en, Персональный компьютер — не архитектура, платформа или что-то подобное. Apple II, Mac, Amiga и Raspberry Pi 400 — тоже персональные компьютеры.@ru; re... [truncated 34 chars]",
      "none-of constraint: item of property constraint: Adobe Flash; replacement property: software engine; replacement value: web browser",
      "none-of constraint: item of property constraint: mobile phone; constraint clarification: This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134621319@en; replacement value: BlackBerry, Java Platform, Micro Edition, iOS, Windows Phone, Android",
      "none-of constraint: item of property constraint: Mac OS operating systems, Mac; replacement value: Classic Mac OS",
      "none-of constraint: item of property constraint: home computer, desktop computer",
      "none-of constraint: item of property constraint: Google Stadia; constraint clarification: This constraint does not apply when used as an qualifier on properties whose instance of (P31) is Q134611584@en",
      "none-of constraint: item of property constraint: Xbox Series X, Xbox Series S; constraint clarification: For the ninth generation of Xbox consoles use \"Xbox Series X and Series S \" (Q98973368) instead@en; replacement value: Xbox Series X and Series S",
      "none-of constraint: item of property constraint: Xbox One; replacement value: ایکس‌باکس وان",
      "... omitted 3 items"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q52558054",
    "result": false,
    "step": "causality_filter",
    "violation_name": "None of"
  }
]
```

---

## 013. `reform_Q410612_P2877_2433431025`

| Field | Value |
|---|---|
| qid | Q410612 |
| property | P2877 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| group_key | TBOX::P2877::2433431025 |
| tbox_revision_key | TBOX::P2877::2433431025 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Wostr",
  "kind": "T_BOX",
  "property_revision_id": 2433431025,
  "property_revision_prev": 2425181358
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-11-25T07:48:31",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
  "report_revision_new": 2433684576,
  "report_revision_old": 2430237820,
  "report_violation_type": "Single value",
  "report_violation_type_normalized": "Single value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Single value",
  "value": null,
  "value_current_2026": [
    "14722063",
    "1357"
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
  "truth_tokens": [],
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
    "label": "fluoroform"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q19474404",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q243547",
            "Q27077129",
            "Q407883",
            "Q410612",
            "Q45044"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 5,
  "author": "Wostr",
  "before_constraint_count": 5,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "group of stereoisomers",
              "id": "Q243547",
              "label_en": "(RS)-ketamine"
            },
            {
              "description_en": "PI3K inhibitor",
              "id": "Q27077129",
              "label_en": "duvelisib"
            },
            {
              "description_en": "chemical compound",
              "id": "Q407883",
              "label_en": "ketoconazole"
            },
            {
              "description_en": "chemical compound",
              "id": "Q410612",
              "label_en": "fluoroform"
            },
            {
              "description_en": "chemical compound",
              "id": "Q45044",
              "label_en": "magnesium citrate"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "\\d+"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "group of stereoisomers",
              "id": "Q243547",
              "label_en": "(RS)-ketamine"
            },
            {
              "description_en": "PI3K inhibitor",
              "id": "Q27077129",
              "label_en": "duvelisib"
            },
            {
              "description_en": "chemical compound",
              "id": "Q407883",
              "label_en": "ketoconazole"
            },
            {
              "description_en": "chemical compound",
              "id": "Q45044",
              "label_en": "magnesium citrate"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "\\d+"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "32c7b5c7444e03defeaf44d5a32dc6f0a79c3f86",
  "hash_before": "d888370954dd4fe246fb2536adb4e79fd074fe62",
  "property_revision_id": 2433431025,
  "property_revision_prev": 2425181358,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q410612"
      ],
      "constraint_qid": "Q19474404",
      "qualifier_property": "P2303",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q19474404",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q243547",
            "Q27077129",
            "Q407883",
            "Q45044"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: exception to constraint: (RS)-ketamine, duvelisib, ketoconazole, fluoroform, magnesium citrate",
      "format constraint: format as a regular expression: \\d+; constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ],
    "before": [
      "single-value constraint: exception to constraint: (RS)-ketamine, duvelisib, ketoconazole, magnesium citrate",
      "format constraint: format as a regular expression: \\d+; constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q19474404",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Single value"
  }
]
```

---

## 014. `reform_Q450675_P8748_2297581722`

| Field | Value |
|---|---|
| qid | Q450675 |
| property | P8748 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| group_key | TBOX::P8748::2297581722 |
| tbox_revision_key | TBOX::P8748::2297581722 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Kolja21",
  "kind": "T_BOX",
  "property_revision_id": 2297581722,
  "property_revision_prev": 2297581668
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
  "value": null,
  "value_current_2026": [
    "1032307897"
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
    "description": "266th pope of the Catholic Church (2013–2025)",
    "label": "Pope Francis"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21503250",
      "qualifiers": [
        {
          "property_id": "P2308",
          "values": [
            "Q13417114",
            "Q5",
            "Q8436"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q21503252"
          ]
        },
        {
          "property_id": "P2316",
          "values": [
            "Q21502408"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 8,
  "author": "Kolja21",
  "before_constraint_count": 8,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "^1[0-9]{7,8}[0-9X]$"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for the Virtual International Authority File database [format: up to 22 digits]; please note: VIAF is a cluster, the ID can include multiple items",
              "id": "P214",
              "label_en": "VIAF cluster ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier from the Gemeinsame Normdatei authority file of names, subjects, and organizations",
              "id": "P227",
              "label_en": "GND ID"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "family part of the nobility of a region or country",
              "id": "Q13417114",
              "label_en": "noble family"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            {
              "description_en": "group of people affiliated by consanguinity, law, affinity, or co-residence",
              "id": "Q8436",
              "label_en": "family"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type constraints",
              "id": "Q21503252",
              "label_en": "instance of"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ],
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "^1[0-9]{7,8}[0-9X]$"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for the Virtual International Authority File database [format: up to 22 digits]; please note: VIAF is a cluster, the ID can include multiple items",
              "id": "P214",
              "label_en": "VIAF cluster ID"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier from the Gemeinsame Normdatei authority file of names, subjects, and organizations",
              "id": "P227",
              "label_en": "GND ID"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            {
              "description_en": "group of people affiliated by consanguinity, law, affinity, or co-residence",
              "id": "Q8436",
              "label_en": "family"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type constraints",
              "id": "Q21503252",
              "label_en": "instance of"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ],
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "5cc9ccf118135357ba50676dc7f5bddd1d4f74c5",
  "hash_before": "5c95a8861bf74b34235a96de5e0a88be268d75bf",
  "property_revision_id": 2297581722,
  "property_revision_prev": 2297581668,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q13417114"
      ],
      "constraint_qid": "Q21503250",
      "qualifier_property": "P2308",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21503250",
      "qualifiers": [
        {
          "property_id": "P2308",
          "values": [
            "Q5",
            "Q8436"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q21503252"
          ]
        },
        {
          "property_id": "P2316",
          "values": [
            "Q21502408"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: ^1[0-9]{7,8}[0-9X]$; constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "item-requires-statement constraint: property: VIAF cluster ID",
      "item-requires-statement constraint: property: GND ID; constraint status: mandatory constraint",
      "subject type constraint: class: noble family, human, family; relation: instance of; constraint status: mandatory constraint",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as reference"
    ],
    "before": [
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: ^1[0-9]{7,8}[0-9X]$; constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "item-requires-statement constraint: property: VIAF cluster ID",
      "item-requires-statement constraint: property: GND ID; constraint status: mandatory constraint",
      "subject type constraint: class: human, family; relation: instance of; constraint status: mandatory constraint",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as reference"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q21502404",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Format"
  }
]
```

---

## 015. `reform_Q453346_P31_2440675178`

| Field | Value |
|---|---|
| qid | Q453346 |
| property | P31 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | head |
| constraint_family | Q21510851 |
| group_key | TBOX::P31::2440675178 |
| tbox_revision_key | TBOX::P31::2440675178 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Clemens Dulcis",
  "kind": "T_BOX",
  "property_revision_id": 2440675178,
  "property_revision_prev": 2439474684
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-11T19:29:00",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
  "report_revision_new": 2440991063,
  "report_revision_old": 2440479307,
  "report_violation_type": "None of",
  "report_violation_type_normalized": "None of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "None of",
  "value": null,
  "value_current_2026": [
    "Q5"
  ],
  "value_current_2026_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo"
  ],
  "value_current_2026_labels_en": [
    "human"
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
  "truth_tokens": [],
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
    "description": "French nobleman",
    "label": "Amaury III de Montfort"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q52558054",
      "qualifiers": [
        {
          "property_id": "P2305",
          "values": [
            "Q108064011",
            "Q208569",
            "Q209939",
            "Q222910",
            "Q4176708",
            "Q4712779",
            "Q60030240",
            "Q68902449"
          ]
        },
        {
          "property_id": "P6607",
          "values": [
            "Use \"album\" (Q482994) with \"instance of\" (P31), use the specific form with \"form of creative work\" (P7937).@en"
          ]
        },
        {
          "property_id": "P6824",
          "values": [
            "P7937"
          ]
        },
        {
          "property_id": "P9729",
          "values": [
            "Q482994"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 197,
  "author": "Clemens Dulcis",
  "before_constraint_count": 197,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "specialization of a person or organization; see P106 for the occupation",
              "id": "P101",
              "label_en": "field of work"
            },
            {
              "description_en": "usually used as a qualifier",
              "id": "P1011",
              "label_en": "excluding"
            },
            {
              "description_en": "usually used as a qualifier",
              "id": "P1012",
              "label_en": "including"
            },
            {
              "description_en": "property by which a distinction or classification is made",
              "id": "P1013",
              "label_en": "criterion used"
            },
            {
              "description_en": "the taxon of an individual named organism (animal, plant)",
              "id": "P10241",
              "label_en": "individual of taxon"
            },
            {
              "description_en": "person or organization who grants an award, certification, grant, or role",
              "id": "P1027",
              "label_en": "conferred by"
            },
            {
              "description_en": "material or product, including services, produced or provided by an organization, industry, facility, or process",
              "id": "P1056",
              "label_en": "product or material produced"
            },
            {
              "description_en": "work or narration for or in which this statement is true",
              "id": "P10663",
              "label_en": "applies to work"
            },
            {
              "description_en": "to be used as a qualifier, value must be between 0 and 1",
              "id": "P1107",
              "label_en": "proportion"
            },
            {
              "description_en": "number of instances of this subject in the universe of the subject (the actual number of instances in Wikidata may be lower or higher)",
              "id": "P1114",
              "label_en": "quantity"
            },
            {
              "description_en": "league or competition in which team or player has played, or in which an event occurs",
              "id": "P118",
              "label_en": "league or competition"
            },
            {
              "description_en": "service stopping at a station",
              "id": "P1192",
              "label_en": "connecting service"
            },
            {
              "description_en": "item simulated, imitated, or made to appear real by this item",
              "id": "P12328",
              "label_en": "simulates"
            },
            {
              "description_en": "latest date beyond which the statement could no longer be true",
              "id": "P12506",
              "label_en": "latest end date"
            },
            {
              "description_en": "time period when a statement is valid",
              "id": "P1264",
              "label_en": "valid in period"
            },
            {
              "description_en": "organization, individual, or concept that an entity represents",
              "id": "P1268",
              "label_en": "represents"
            },
            {
              "description_en": "topic of which this item is an aspect; item that offers a broader perspective on the same topic",
              "id": "P1269",
              "label_en": "facet of"
            },
            {
              "description_en": "specific object to which an occurrence or class of occurrences applies",
              "id": "P12912",
              "label_en": "object of occurrence"
            },
            {
              "description_en": "class that includes the object(s) to which this occurrence (or class of occurrence) occurs or occurred",
              "id": "P12913",
              "label_en": "class of object(s) of occurrence"
            },
            {
              "description_en": "role that the object(s) of this occurrence take on in the context of this occurrence",
              "id": "P12992",
              "label_en": "objects of occurrence have role"
            },
            {
              "description_en": "role that animate agents of this action take on in the context of this action",
              "id": "P12993",
              "label_en": "role of agent(s) of action"
            },
            {
              "description_en": "class of animate items that may initiate this action or class of actions (for roles of agents, use P12993)",
              "id": "P12994",
              "label_en": "class of agent(s) of action"
            },
            {
              "description_en": "particular animate item that initiates this action or class of actions",
              "id": "P12995",
              "label_en": "agent of action"
            },
            "... omitted 90 items"
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to be one of a given set of items",
          "id": "Q21510859",
          "label_en": "one-of constraint"
        },
        "parameters": {
          "P2241": [
            {
              "description_en": "reason for deprecation of a Wikidata property constraint",
              "id": "Q99460987",
              "label_en": "constraint provides suggestions for manual input"
            }
          ],
          "P2305": [
            {
              "description_en": "part of a naming scheme for individuals, used in many cultures worldwide",
              "id": "Q101352",
              "label_en": "family name"
            },
            {
              "description_en": "Wikidata metaclass; legal concept of uniquely identifiable piece or work of music, either vocal or instrumental; NOT applicable to recordings, broadcasts, or individual publications of music in printed or digital form or on physical media",
              "id": "Q105543609",
              "label_en": "musical work/composition"
            },
            {
              "description_en": "sequence of images that give the impression of movement, stored on film stock",
              "id": "Q11424",
              "label_en": "film"
            },
            {
              "description_en": "article in an academic publication, usually peer reviewed",
              "id": "Q13442814",
              "label_en": "scholarly article"
            },
            {
              "description_en": "group of one or more organism(s), which a taxonomist adjudges to be a unit",
              "id": "Q16521",
              "label_en": "taxon"
            },
            {
              "description_en": "use of a creative work across several different media",
              "id": "Q196600",
              "label_en": "media franchise"
            },
            {
              "description_en": "collection of musical recordings released in a specific format for consumption",
              "id": "Q2031291",
              "label_en": "musical release"
            },
            {
              "description_en": "singular named exemplar of an animal (e.g., the gorilla named Koko; the cat named Socks)",
              "id": "Q26401003",
              "label_en": "individual animal"
            },
            {
              "description_en": "visual artwork, surface artistically covered with paint",
              "id": "Q3305213",
              "label_en": "painting"
            },
            {
              "description_en": "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
              "id": "Q3331189",
              "label_en": "version, edition or translation"
            },
            {
              "description_en": "building usually intended for living in",
              "id": "Q3947",
              "label_en": "house"
            },
            {
              "description_en": "structure, typically with a roof and walls, standing more or less permanently in one place",
              "id": "Q41176",
              "label_en": "building"
            },
            {
              "description_en": "social role with a set of powers and responsibilities within an organization",
              "id": "Q4164871",
              "label_en": "position"
            },
            {
              "description_en": "use with 'instance of' (P31) for Wikimedia category",
              "id": "Q4167836",
              "label_en": "Wikimedia category"
            },
            {
              "description_en": "organization undertaking commercial, industrial, or professional activity",
              "id": "Q4830453",
              "label_en": "business"
            },
            {
              "description_en": "place of any size, in which people permanently live",
              "id": "Q486972",
              "label_en": "human settlement"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            {
              "description_en": "small clustered human settlement smaller than a town",
              "id": "Q532",
              "label_en": "village"
            },
            {
              "description_en": "written work read for enjoyment or edification",
              "id": "Q7725634",
              "label_en": "literary work"
            },
            {
              "description_en": "three-dimensional work of art",
              "id": "Q860861",
              "label_en": "sculpture"
            }
          ]
        },
        "rank": "deprecated",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item should have a statement with a given property",
          "id": "Q21510864",
          "label_en": "value-requires-statement constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "anything that can be considered, discussed, or observed",
              "id": "Q35120",
              "label_en": "entity"
            }
          ],
          "P2306": [
            {
              "description_en": "this item is a subclass (subset) of that item; ALL instances of this item are instances of that item; different from P31 (instance of), e.g.: volcano is a subclass of mountain; Everest is an instance of mountain",
              "id": "P279",
              "label_en": "subclass of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the subject and the object have to coincide or coexist at some point of history",
          "id": "Q25796498",
          "label_en": "contemporary constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            },
            {
              "description_en": "entity type in Wikibase",
              "id": "Q29934218",
              "label_en": "Wikibase property"
            },
            {
              "description_en": "Wikibase entity type for lexemes",
              "id": "Q51885771",
              "label_en": "Wikibase lexeme"
            },
            {
              "description_en": "Wikibase property value datatype",
              "id": "Q54285143",
              "label_en": "Wikibase form"
            },
            {
              "description_en": "Wikibase entity type for lexicographic senses",
              "id": "Q54285715",
              "label_en": "Wikibase sense"
            },
            {
              "description_en": "Wikibase entity type for Wikimedia Commons",
              "id": "Q59712033",
              "label_en": "МэдыяІнфа Вікібазы"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "Canadian video game development and consultation studio",
              "id": "Q107327507",
              "label_en": "Sweet Baby Inc."
            }
          ],
          "P2305": [
            {
              "description_en": "company working in the video game industry",
              "id": "Q112042224",
              "label_en": "video game company"
            }
          ],
          "P9729": [
            {
              "description_en": "group or corporation that translates video games",
              "id": "Q100588475",
              "label_en": "video game translation company"
            },
            {
              "description_en": "company that publishes video games",
              "id": "Q1137109",
              "label_en": "video game publisher"
            },
            {
              "description_en": "software development organization specializing in the creation of video games (for person use Q58287519)",
              "id": "Q210167",
              "label_en": "video game developer"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "adherent of the religion of Hinduism",
              "id": "Q10090",
              "label_en": "Hindu"
            },
            {
              "description_en": "person who adheres to Christianity",
              "id": "Q106039",
              "label_en": "Christian"
            },
            {
              "description_en": "member of any of the 24 churches that make up the Roman Catholic Church",
              "id": "Q17549077",
              "label_en": "Catholic"
            },
            {
              "description_en": "adherent of the religion of Islam",
              "id": "Q47740",
              "label_en": "Muslim"
            },
            {
              "description_en": "adherent of the religion of Buddhism",
              "id": "Q6926246",
              "label_en": "Buddhists"
            },
            {
              "description_en": "ethnoreligious group and nation from the Levant",
              "id": "Q7325",
              "label_en": "Jewish people"
            }
          ],
          "P6824": [
            {
              "description_en": "religion of a person, organization or religious building, or associated with this subject",
              "id": "P140",
              "label_en": "religion or worldview"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "fictional character of male sex",
              "id": "Q100911674",
              "label_en": "male character"
            },
            {
              "description_en": "to be used in \"sex or gender\" (P21) to indicate that the human subject is a male or \"semantic gender\" (P10339) to indicate that a word refers to a male person",
              "id": "Q6581097",
              "label_en": "male"
            },
            {
              "description_en": "human of the male sex",
              "id": "Q84048850",
              "label_en": "male human"
            },
            {
              "description_en": "male adult human",
              "id": "Q8441",
              "label_en": "man"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ],
          "P6607": [
            {
              "value": "Das Geschlecht wird über die Eigenschaft P21 spezifiziert.@de"
            },
            {
              "value": "Specify gender using P21 (sex or gender)@en"
            },
            {
              "value": "Utiliser P21 pour le genre ou le sexe@fr"
            }
          ],
          "P6824": [
            {
              "description_en": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
              "id": "P21",
              "label_en": "sex or gender"
            }
          ],
          "P9729": [
            {
              "description_en": "to be used in \"sex or gender\" (P21) to indicate that the human subject is a male or \"semantic gender\" (P10339) to indicate that a word refers to a male person",
              "id": "Q6581097",
              "label_en": "male"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "type of a paperback book",
              "id": "Q1009641",
              "label_en": "bunkobon"
            },
            {
              "description_en": "新書判の叢書・本",
              "id": "Q11502500",
              "label_en": "shinsho"
            },
            {
              "description_en": "book with two works bound back-to-back, rotated 180 degrees",
              "id": "Q124685562",
              "label_en": "tête-bêche"
            },
            {
              "description_en": "Type of hardback",
              "id": "Q12566525",
              "label_en": "Тыс"
            },
            {
              "description_en": "book-length publication in digital form",
              "id": "Q128093",
              "label_en": "ebook"
            },
            {
              "description_en": "electronic publication avilable over a network",
              "id": "Q1294318",
              "label_en": "online book"
            },
            {
              "description_en": "book with pages bound using a metal or plastic coil",
              "id": "Q16929794",
              "label_en": "ring connection"
            },
            {
              "description_en": "small-size book which could fit in a reader's pocket",
              "id": "Q17994250",
              "label_en": "pocket edition"
            },
            {
              "description_en": "book with a paper or paperboard cover",
              "id": "Q193934",
              "label_en": "paperback"
            },
            {
              "description_en": "book bound with a rigid protective cover",
              "id": "Q193955",
              "label_en": "hardcover"
            },
            {
              "description_en": "Japanese term for a book",
              "id": "Q241996",
              "label_en": "tankōbon"
            },
            {
              "description_en": "short, inexpensive booklet; type of street literature printed in early modern Europe",
              "id": "Q2558308",
              "label_en": "chapbook"
            },
            {
              "description_en": "booklet comprised of alternately folded pages",
              "id": "Q361880",
              "label_en": "leporello book"
            },
            {
              "description_en": "quality paperback book",
              "id": "Q990683",
              "label_en": "softcover"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "type of special limited edition of a video game, often comprizing some extra bonuses/addons, or higher quality support medium or packaging",
              "id": "Q101552184",
              "label_en": "collector's edition"
            },
            {
              "description_en": "video game which contains the main game along with all downloadable content and additional content not found in the original release",
              "id": "Q105760475",
              "label_en": "definitive edition"
            },
            {
              "description_en": "type of video game edition",
              "id": "Q107458055",
              "label_en": "director's cut"
            },
            {
              "description_en": "video game edition",
              "id": "Q108028700",
              "label_en": "digital deluxe edition"
            },
            {
              "description_en": "downloadable content that allow one to upgrade the edition of their game",
              "id": "Q108028709",
              "label_en": "video game edition upgrade"
            },
            {
              "description_en": "type of special limited edition of a video game, often comprizing some extra bonuses/addons, or higher quality support medium or packaging",
              "id": "Q108308863",
              "label_en": "anniversary edition"
            },
            {
              "description_en": "type of special limited edition of a software or other electronic media, often comprizing some extra bonuses/addons, or higher quality support medium or packaging",
              "id": "Q58806721",
              "label_en": "collector's edition"
            },
            {
              "description_en": "repackaged version of a video game honoring an award",
              "id": "Q96604496",
              "label_en": "Game of the Year edition"
            }
          ],
          "P6824": [
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "television program adapted from another work",
              "id": "Q101716172",
              "label_en": "television adaptation"
            },
            {
              "description_en": "video game that is adapted from another work",
              "id": "Q117216668",
              "label_en": "video game adaptation"
            },
            {
              "description_en": "films adapted from another work",
              "id": "Q1257444",
              "label_en": "film adaptation"
            },
            {
              "description_en": "feature film based on actual events",
              "id": "Q28146524",
              "label_en": "film based on actual events"
            },
            {
              "description_en": "film where work of literature makes up the basic",
              "id": "Q52162262",
              "label_en": "film based on literature"
            },
            {
              "description_en": "type of film adaptation",
              "id": "Q52207310",
              "label_en": "film based on book"
            },
            {
              "description_en": "film based on a specific literary genre, the novel",
              "id": "Q52207399",
              "label_en": "film based on a novel"
            }
          ],
          "P6607": [
            {
              "value": "Too broad, use a more specific subclass instead and move it to \"has characteristic (P1552)\"@en"
            }
          ],
          "P6824": [
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "type of mill",
              "id": "Q1018126",
              "label_en": "butter-mill"
            },
            {
              "description_en": "wind powered sawmill, commonly found in the Netherlands",
              "id": "Q12013554",
              "label_en": "wind powered sawmill"
            },
            {
              "description_en": "mill to produce ceramic glaze",
              "id": "Q135509402",
              "label_en": "glazuurmolen"
            },
            {
              "description_en": null,
              "id": "Q18640194",
              "label_en": "cocoa mill"
            },
            {
              "description_en": "special paint mill",
              "id": "Q1877290",
              "label_en": "white lead mill"
            },
            {
              "description_en": "molen waar graan to mout voor de jenever- en bierproductie gemalen wordt",
              "id": "Q1976835",
              "label_en": "malting mill"
            },
            {
              "description_en": "machine",
              "id": "Q1987723",
              "label_en": "fulling mill"
            },
            {
              "description_en": "type of mill",
              "id": "Q2084126",
              "label_en": "iron mill"
            },
            {
              "description_en": "Mühle, die zum Schleifen von Werkstoffen dient",
              "id": "Q2238889",
              "label_en": "grinding mill"
            },
            {
              "description_en": "molen die krijt maalt",
              "id": "Q2254524",
              "label_en": "chalk mill"
            },
            {
              "description_en": "mill to produce snuff tobacco",
              "id": "Q2351283",
              "label_en": "snuff mill"
            },
            {
              "description_en": "molen die tufsteen tot tras maalt ten behoeve van de productie van cement en mortel",
              "id": "Q2456069",
              "label_en": "trass mill"
            },
            {
              "description_en": "mill where paints are ground",
              "id": "Q2598061",
              "label_en": "paint mill"
            },
            {
              "description_en": "windpump used to pump water out of a polder",
              "id": "Q2695327",
              "label_en": "polder windmill"
            },
            {
              "description_en": "type of mill",
              "id": "Q2768798",
              "label_en": "smalt mill"
            },
            {
              "description_en": "type of mill where materials are crushed and thus made more flexible or broken",
              "id": "Q2868354",
              "label_en": "stamping mill"
            },
            {
              "description_en": "industrial mill that extracts oil from seeds or vegetable material",
              "id": "Q297163",
              "label_en": "oil mill"
            },
            {
              "description_en": null,
              "id": "Q3257863",
              "label_en": "copper water mill"
            },
            {
              "description_en": "culinary tool for grinding spices",
              "id": "Q356838",
              "label_en": "spice mill"
            },
            {
              "description_en": "also known as Catskill’s mill",
              "id": "Q374116",
              "label_en": "bark mill"
            },
            {
              "description_en": "mill where gunpowder is made from sulfur, saltpeter and charcoal",
              "id": "Q385422",
              "label_en": "powder mill"
            },
            {
              "description_en": "mill used to store and grind whole peppercorns",
              "id": "Q474686",
              "label_en": "pepper mill"
            },
            {
              "description_en": "facilities that are primarily engaged in the manufacture of paper or paper products, usually from wood, recycled paper, or other fiber pulp. Originally water-driven, modern paper mills are typically powered electrically, for use with P366 (has use)",
              "id": "Q56316683",
              "label_en": "paper mill"
            },
            {
              "description_en": "mill where oil is extracted from olives",
              "id": "Q61978174",
              "label_en": "olive oil mill"
            },
            "... omitted 3 items"
          ],
          "P6607": [
            {
              "value": "Use P366 instead for the function(s) the mill had, and use for P31 'mill', 'watermill', 'windmill', etc.@en"
            }
          ],
          "P9729": [
            {
              "description_en": "device that breaks solid materials into smaller pieces by grinding, crushing, or cutting",
              "id": "Q44494",
              "label_en": "mill"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "song protesting against war",
              "id": "Q102189017",
              "label_en": "anti-war song"
            },
            {
              "description_en": "melody with anonymous or unknown composer without authorized original version; melody that is part of a musical tradition",
              "id": "Q105575966",
              "label_en": "traditional melody"
            },
            {
              "description_en": "song, which is parody",
              "id": "Q23817363",
              "label_en": "parody song"
            },
            {
              "description_en": "type of song",
              "id": "Q25452063",
              "label_en": "political song"
            },
            {
              "description_en": "song associated with a sports team and military conflicts",
              "id": "Q261434",
              "label_en": "fight song"
            },
            {
              "description_en": "song that advocates or praises a revolution",
              "id": "Q265147",
              "label_en": "revolutionary song"
            },
            {
              "description_en": "piece of music closely connected to a form of work",
              "id": "Q502658",
              "label_en": "work song"
            },
            {
              "description_en": "song intended to be performed for the Christmas and holiday season",
              "id": "Q56572789",
              "label_en": "Christmas-themed song"
            },
            {
              "description_en": "musical composition considered an important part of the musical repertoire of jazz musicians; composition that is widely known, performed, and recorded by jazz musicians, and widely known by listeners",
              "id": "Q591990",
              "label_en": "jazz standard"
            },
            {
              "description_en": "song that is associated with a movement for social change",
              "id": "Q829147",
              "label_en": "protest song"
            },
            {
              "description_en": "song with anonymous or unknown writer without authorized original version; song that is part of the oral tradition",
              "id": "Q943929",
              "label_en": "traditional folk song"
            }
          ],
          "P6824": [
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            }
          ],
          "P9729": [
            {
              "description_en": "Wikidata metaclass; legal concept of uniquely identifiable piece or work of music, either vocal or instrumental; NOT applicable to recordings, broadcasts, or individual publications of music in printed or digital form or on physical media",
              "id": "Q105543609",
              "label_en": "musical work/composition"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "Christian song of praise with lyrics from biblical or holy texts other than the Psalms",
              "id": "Q1033831",
              "label_en": "canticle"
            },
            {
              "description_en": "musical piece for a single voice as part of a larger work",
              "id": "Q178122",
              "label_en": "aria"
            },
            {
              "description_en": "secular vocal music composition of the Renaissance and early Baroque eras",
              "id": "Q193217",
              "label_en": "madrigal"
            },
            {
              "description_en": "musical form in opera, cantata, mass or oratorio",
              "id": "Q202534",
              "label_en": "recitative"
            }
          ],
          "P6824": [
            {
              "description_en": "structure of a creative work",
              "id": "P7937",
              "label_en": "form of creative work"
            }
          ],
          "P9729": [
            {
              "description_en": "Wikidata metaclass; legal concept of uniquely identifiable piece or work of music, either vocal or instrumental; NOT applicable to recordings, broadcasts, or individual publications of music in printed or digital form or on physical media",
              "id": "Q105543609",
              "label_en": "musical work/composition"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "piece of music to be performed by a single pianist",
              "id": "Q10349334",
              "label_en": "piano solo"
            },
            {
              "description_en": "composition for piano and two instruments, usually a violin and a cello",
              "id": "Q1414262",
              "label_en": "piano trio"
            },
            {
              "description_en": "musical work for two pianists, sometimes with accompanying instruments",
              "id": "Q17126392",
              "label_en": "piano duet"
            },
            {
              "description_en": "piece of music for solo piano",
              "id": "Q1746015",
              "label_en": "piano piece"
            },
            {
              "description_en": "musical composition for 5 performers including a piano; genre of art music played by such groups",
              "id": "Q1746023",
              "label_en": "piano quintet"
            },
            {
              "description_en": "chamber music composition for piano and three other instruments",
              "id": "Q1746025",
              "label_en": "piano quartet"
            },
            {
              "description_en": "composition for piano and five other musical instruments",
              "id": "Q7190206",
              "label_en": "piano sextet"
            }
          ],
          "P6824": [
            {
              "description_en": "structure of a creative work",
              "id": "P7937",
              "label_en": "form of creative work"
            }
          ],
          "P9729": [
            {
              "description_en": "Wikidata metaclass; legal concept of uniquely identifiable piece or work of music, either vocal or instrumental; NOT applicable to recordings, broadcasts, or individual publications of music in printed or digital form or on physical media",
              "id": "Q105543609",
              "label_en": "musical work/composition"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "national heritage site in France",
              "id": "Q10387575",
              "label_en": "monument historique inscrit"
            },
            {
              "description_en": "national heritage site in France",
              "id": "Q10387684",
              "label_en": "classified historical monument"
            },
            {
              "description_en": "collection of Rijksmonumenten designated as a complex",
              "id": "Q13423591",
              "label_en": "Rijksmonument complex"
            },
            {
              "description_en": "national heritage site of the Netherlands",
              "id": "Q916333",
              "label_en": "Rijksmonument"
            },
            {
              "description_en": "protected French building as a Historical Monument (use « classified Historical Monument » and « inscribed Historical Monument »)",
              "id": "Q916475",
              "label_en": "Historical Monument"
            }
          ],
          "P6607": [
            {
              "value": "Use heritage designation (P1435) instead@en"
            }
          ],
          "P6824": [
            {
              "description_en": "heritage designation of a cultural or natural site",
              "id": "P1435",
              "label_en": "heritage designation"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "victim of a crime",
              "id": "Q10436169",
              "label_en": "crime victim"
            },
            {
              "description_en": "person who have been killed or severely wounded by the act or arson",
              "id": "Q107737414",
              "label_en": "arson victim"
            },
            {
              "description_en": "people who have been injured or killed in an act of vigilantism",
              "id": "Q107973786",
              "label_en": "vigilantism victim"
            },
            {
              "description_en": "people who have been killed by the act of homicide",
              "id": "Q108295395",
              "label_en": "homicide victim"
            },
            {
              "description_en": "person/entity held by a belligerent party to another or seized for carrying out agreement",
              "id": "Q192620",
              "label_en": "hostage"
            },
            {
              "description_en": "person who has disappeared and whose status as alive or dead cannot be confirmed",
              "id": "Q388505",
              "label_en": "missing person"
            },
            {
              "description_en": "person who was killed or severely wounded by the act of war crimes",
              "id": "Q46076028",
              "label_en": "war crime victim"
            },
            {
              "description_en": "person who was harmed by the act of rape",
              "id": "Q67630359",
              "label_en": "rape victim"
            },
            {
              "description_en": "person who has been killed by the act of murder",
              "id": "Q73153647",
              "label_en": "murder victim"
            },
            {
              "description_en": "person who was harmed by the act of sexual abuse",
              "id": "Q73155718",
              "label_en": "sexual abuse victim"
            },
            {
              "description_en": "person who was killed or severely wounded in an incident of terrorism",
              "id": "Q73164596",
              "label_en": "terrorism victim"
            },
            {
              "description_en": "person who has been kidnapped",
              "id": "Q73168492",
              "label_en": "kidnapping victim"
            },
            {
              "description_en": "person who was killed or severely wounded by the act of torture",
              "id": "Q73169841",
              "label_en": "torture victim"
            },
            {
              "description_en": "person who was wounded or killed by the act of assault",
              "id": "Q89061424",
              "label_en": "assault victim"
            },
            {
              "description_en": "person who was harmed by the act of sex trafficking or sexual slavery",
              "id": "Q98966335",
              "label_en": "sex trafficking victim"
            },
            {
              "description_en": "person who was harmed by the act of human trafficking",
              "id": "Q98966337",
              "label_en": "human trafficking victim"
            },
            {
              "description_en": "people who have been a victim of or claim to be a victim of cyberbullying at least once",
              "id": "Q98966375",
              "label_en": "cyber bullying victim"
            },
            {
              "description_en": "people who have been injured or killed in a hate crime",
              "id": "Q98966378",
              "label_en": "hate crime victim"
            },
            {
              "description_en": "people who have been affected by identity theft",
              "id": "Q98966385",
              "label_en": "identity theft victim"
            }
          ],
          "P6824": [
            {
              "description_en": "significant or notable events associated with the subject",
              "id": "P793",
              "label_en": "significant event"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "video game produced some time in the past of which no surviving copies or accessible downloads are known to exist",
              "id": "Q104438884",
              "label_en": "lost video game"
            },
            {
              "description_en": "television series produced some time in the past of which no surviving copies are known to exist",
              "id": "Q104438889",
              "label_en": "lost television series"
            },
            {
              "description_en": "television series episode produced some time in the past of which no surviving copies are known to exist.  Use as value for \"signficiant event\" (P793)",
              "id": "Q104438898",
              "label_en": "lost television series episode"
            },
            {
              "description_en": "book produced some time in the past of which no surviving copies are known to exist",
              "id": "Q104438918",
              "label_en": "lost book"
            },
            {
              "description_en": "musical work produced some time in the past of which no surviving copies are known to exist",
              "id": "Q104439050",
              "label_en": "lost musical work"
            },
            {
              "description_en": "radio program produced some time in the past of which no surviving copies are known to exist",
              "id": "Q104439055",
              "label_en": "lost radio program"
            },
            {
              "description_en": "podcast for which the RSS feed and/or episode files are no longer available",
              "id": "Q107737653",
              "label_en": "lost podcast"
            },
            {
              "description_en": "formerly lost media that has since been found",
              "id": "Q122965806",
              "label_en": "formerly lost media"
            },
            {
              "description_en": "podcast episode produced some time in the past of which no surviving copies are known to exist",
              "id": "Q123490564",
              "label_en": "lost podcast episode"
            },
            {
              "description_en": "feature or short film of which no surviving copies are known to exist",
              "id": "Q1268687",
              "label_en": "lost film"
            },
            {
              "description_en": "literary work produced some time in the past of which no surviving copies are known to exist",
              "id": "Q1585442",
              "label_en": "lost literary work"
            },
            {
              "description_en": "piece of art that once existed",
              "id": "Q4140840",
              "label_en": "lost artwork"
            }
          ],
          "P6824": [
            {
              "description_en": "significant or notable events associated with the subject",
              "id": "P793",
              "label_en": "significant event"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "type of anime",
              "id": "Q104541980",
              "label_en": "anime based on a manga"
            },
            {
              "description_en": "type of manga",
              "id": "Q107408274",
              "label_en": "manga based on a anime"
            }
          ],
          "P6824": [
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "musical form, miniature of sonata",
              "id": "Q1045984",
              "label_en": "sonatina"
            },
            {
              "description_en": "type of musical composition, usually for a solo instrument or a small instrumental ensemble",
              "id": "Q131269",
              "label_en": "sonata"
            },
            {
              "description_en": "sonata written for solo piano",
              "id": "Q1546995",
              "label_en": "piano sonata"
            },
            {
              "description_en": "musical composition for clarinet",
              "id": "Q1999051",
              "label_en": "clarinet sonata"
            },
            {
              "description_en": "musical composition for cello and piano",
              "id": "Q2308166",
              "label_en": "sonata for cello and piano"
            },
            {
              "description_en": "sonata for flute and accompanying instrument",
              "id": "Q3070575",
              "label_en": "flute sonata"
            },
            {
              "description_en": "instrumental composition dating from the Baroque period",
              "id": "Q543020",
              "label_en": "sonata da chiesa"
            },
            {
              "description_en": "musical composition for violin",
              "id": "Q7933385",
              "label_en": "violin sonata"
            },
            {
              "description_en": "sonata written for cello",
              "id": "Q857538",
              "label_en": "cello sonata"
            },
            {
              "description_en": "Baroque sonata for two or three melody instruments and continuo",
              "id": "Q903425",
              "label_en": "trio sonata"
            }
          ],
          "P6824": [
            {
              "description_en": "structure of a creative work",
              "id": "P7937",
              "label_en": "form of creative work"
            }
          ],
          "P9729": [
            {
              "description_en": "Wikidata metaclass; legal concept of uniquely identifiable piece or work of music, either vocal or instrumental; NOT applicable to recordings, broadcasts, or individual publications of music in printed or digital form or on physical media",
              "id": "Q105543609",
              "label_en": "musical work/composition"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "type of Japanese video game featuring erotica",
              "id": "Q1046788",
              "label_en": "eroge"
            },
            {
              "description_en": "Japanese pornographic animation, comics, and video games",
              "id": "Q172067",
              "label_en": "hentai"
            },
            {
              "description_en": "interactive fiction game",
              "id": "Q689445",
              "label_en": "visual novel"
            }
          ],
          "P6824": [
            {
              "description_en": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
              "id": "P136",
              "label_en": "genre"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "collection of written creative works chosen by the compiler",
              "id": "Q105420",
              "label_en": "anthology"
            },
            {
              "description_en": "book containing multiple novelettes",
              "id": "Q111180384",
              "label_en": "novelette collection"
            },
            {
              "description_en": "collection of dramas published together.",
              "id": "Q119318561",
              "label_en": "drama collection"
            },
            {
              "description_en": "collection of poems by a single author published together. See also poetry anthology (Q19357149)",
              "id": "Q12106333",
              "label_en": "poem collection"
            },
            {
              "description_en": "book containing several short stories by different authors",
              "id": "Q125544547",
              "label_en": "short story anthology"
            },
            {
              "description_en": "book containing several short stories by a single author",
              "id": "Q1279564",
              "label_en": "short story collection"
            },
            {
              "description_en": "book containing multiple novels",
              "id": "Q133500522",
              "label_en": "novel collection"
            },
            {
              "description_en": "written, fictional, prose narrative normally longer than a short story but shorter than a novel",
              "id": "Q149537",
              "label_en": "novella"
            },
            {
              "description_en": "works from multiple poets chosen by an editor",
              "id": "Q19357149",
              "label_en": "poetry anthology"
            },
            {
              "description_en": "book containing multiple novellas",
              "id": "Q20024995",
              "label_en": "novella collection"
            },
            {
              "description_en": "narrative prose fiction shorter than a novella and longer than a short story",
              "id": "Q472808",
              "label_en": "novelette"
            },
            {
              "description_en": "brief work of literature, usually written in narrative prose",
              "id": "Q49084",
              "label_en": "short story"
            },
            {
              "description_en": "style of fictional literature or fiction of extreme brevity",
              "id": "Q5457615",
              "label_en": "flash fiction"
            },
            {
              "description_en": "publication, usually a book, containing a compilation of letters written by a real person",
              "id": "Q65085460",
              "label_en": "letter collection"
            },
            {
              "description_en": "form of poetry with fourteen lines and strict rhyming structure",
              "id": "Q80056",
              "label_en": "sonnet"
            },
            {
              "description_en": "narrative text, normally of a substantial length and in the form of prose describing a fictional and sequential story",
              "id": "Q8261",
              "label_en": "novel"
            }
          ],
          "P6607": [
            {
              "value": "Move to P7937 and use Q7725634 as value instead@en"
            },
            {
              "value": "À déplacer vers P7937 et utiliser  Q7725634 à la place@fr"
            }
          ],
          "P6824": [
            {
              "description_en": "structure of a creative work",
              "id": "P7937",
              "label_en": "form of creative work"
            }
          ],
          "P9729": [
            {
              "description_en": "written work read for enjoyment or edification",
              "id": "Q7725634",
              "label_en": "literary work"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "opera that depicts aspects of ‘simple’ (generally rural) life, usually in opposition to that of the court or city, or that is expressive of its atmosphere or values",
              "id": "Q105906205",
              "label_en": "pastoral opera"
            },
            {
              "description_en": "opera genre",
              "id": "Q13220650",
              "label_en": "comic opera"
            },
            {
              "description_en": "opera genre",
              "id": "Q1457481",
              "label_en": "Spieloper"
            },
            {
              "description_en": "theatrical entertainment which began in Paris towards the end of the 17th century",
              "id": "Q17084138",
              "label_en": "comédie en vaudevilles"
            },
            {
              "description_en": "Italian opera genre",
              "id": "Q208080",
              "label_en": "opera buffa"
            },
            {
              "description_en": "style of Italian opera",
              "id": "Q210675",
              "label_en": "opera seria"
            },
            {
              "description_en": "French opérette subgenre",
              "id": "Q24678689",
              "label_en": "vaudeville-opérette"
            },
            {
              "description_en": "French opera genre",
              "id": "Q3084465",
              "label_en": "opéra bouffe"
            },
            {
              "description_en": "opera genre",
              "id": "Q377258",
              "label_en": "singspiel"
            },
            {
              "description_en": "French versions of Italian opera buffa",
              "id": "Q7099147",
              "label_en": "opéra bouffon"
            },
            {
              "description_en": "musical theatre work with rock music",
              "id": "Q7354827",
              "label_en": "rock musical"
            },
            {
              "description_en": "genre of French opera",
              "id": "Q785479",
              "label_en": "opéra comique"
            },
            {
              "description_en": "french opera genre",
              "id": "Q908705",
              "label_en": "tragédie en musique"
            }
          ],
          "P6607": [
            {
              "value": "Use dramatico-musical work (Q58483083) for P31, and use the specific type with P136 (genre)@en"
            }
          ],
          "P6824": [
            {
              "description_en": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
              "id": "P136",
              "label_en": "genre"
            }
          ],
          "P9729": [
            {
              "description_en": "opera, musical play or show, revue or pantomime for which music has been specially written; for ballet use \"choreographic work\" (Q58483088)",
              "id": "Q58483083",
              "label_en": "dramatico-musical work"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "目的語なしで用いられたり，本来の目的語が主語の位置に移された他動詞",
              "id": "Q105933250",
              "label_en": "pseudo-transitive verb"
            },
            {
              "description_en": "verb that takes no grammatical objects",
              "id": "Q1166153",
              "label_en": "intransitive verb"
            },
            {
              "description_en": "verb that requires one or more objects in a sentence",
              "id": "Q1774805",
              "label_en": "transitive verb"
            },
            {
              "description_en": "verb that may or may not take a grammatical object without changing form",
              "id": "Q4115075",
              "label_en": "ambitransitive verb"
            },
            {
              "description_en": "ambitransitive verb whose subject when intransitive corresponds to its direct object when transitive",
              "id": "Q623554",
              "label_en": "ergative verb"
            }
          ],
          "P6824": [
            {
              "description_en": "possible grammatical property of a verb accepting an object complement",
              "id": "P9295",
              "label_en": "transitivity"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      "... omitted 173 items"
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "specialization of a person or organization; see P106 for the occupation",
              "id": "P101",
              "label_en": "field of work"
            },
            {
              "description_en": "usually used as a qualifier",
              "id": "P1011",
              "label_en": "excluding"
            },
            {
              "description_en": "usually used as a qualifier",
              "id": "P1012",
              "label_en": "including"
            },
            {
              "description_en": "property by which a distinction or classification is made",
              "id": "P1013",
              "label_en": "criterion used"
            },
            {
              "description_en": "the taxon of an individual named organism (animal, plant)",
              "id": "P10241",
              "label_en": "individual of taxon"
            },
            {
              "description_en": "person or organization who grants an award, certification, grant, or role",
              "id": "P1027",
              "label_en": "conferred by"
            },
            {
              "description_en": "material or product, including services, produced or provided by an organization, industry, facility, or process",
              "id": "P1056",
              "label_en": "product or material produced"
            },
            {
              "description_en": "work or narration for or in which this statement is true",
              "id": "P10663",
              "label_en": "applies to work"
            },
            {
              "description_en": "to be used as a qualifier, value must be between 0 and 1",
              "id": "P1107",
              "label_en": "proportion"
            },
            {
              "description_en": "number of instances of this subject in the universe of the subject (the actual number of instances in Wikidata may be lower or higher)",
              "id": "P1114",
              "label_en": "quantity"
            },
            {
              "description_en": "league or competition in which team or player has played, or in which an event occurs",
              "id": "P118",
              "label_en": "league or competition"
            },
            {
              "description_en": "service stopping at a station",
              "id": "P1192",
              "label_en": "connecting service"
            },
            {
              "description_en": "item simulated, imitated, or made to appear real by this item",
              "id": "P12328",
              "label_en": "simulates"
            },
            {
              "description_en": "latest date beyond which the statement could no longer be true",
              "id": "P12506",
              "label_en": "latest end date"
            },
            {
              "description_en": "time period when a statement is valid",
              "id": "P1264",
              "label_en": "valid in period"
            },
            {
              "description_en": "organization, individual, or concept that an entity represents",
              "id": "P1268",
              "label_en": "represents"
            },
            {
              "description_en": "topic of which this item is an aspect; item that offers a broader perspective on the same topic",
              "id": "P1269",
              "label_en": "facet of"
            },
            {
              "description_en": "specific object to which an occurrence or class of occurrences applies",
              "id": "P12912",
              "label_en": "object of occurrence"
            },
            {
              "description_en": "class that includes the object(s) to which this occurrence (or class of occurrence) occurs or occurred",
              "id": "P12913",
              "label_en": "class of object(s) of occurrence"
            },
            {
              "description_en": "role that the object(s) of this occurrence take on in the context of this occurrence",
              "id": "P12992",
              "label_en": "objects of occurrence have role"
            },
            {
              "description_en": "role that animate agents of this action take on in the context of this action",
              "id": "P12993",
              "label_en": "role of agent(s) of action"
            },
            {
              "description_en": "class of animate items that may initiate this action or class of actions (for roles of agents, use P12993)",
              "id": "P12994",
              "label_en": "class of agent(s) of action"
            },
            {
              "description_en": "particular animate item that initiates this action or class of actions",
              "id": "P12995",
              "label_en": "agent of action"
            },
            "... omitted 90 items"
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to be one of a given set of items",
          "id": "Q21510859",
          "label_en": "one-of constraint"
        },
        "parameters": {
          "P2241": [
            {
              "description_en": "reason for deprecation of a Wikidata property constraint",
              "id": "Q99460987",
              "label_en": "constraint provides suggestions for manual input"
            }
          ],
          "P2305": [
            {
              "description_en": "part of a naming scheme for individuals, used in many cultures worldwide",
              "id": "Q101352",
              "label_en": "family name"
            },
            {
              "description_en": "Wikidata metaclass; legal concept of uniquely identifiable piece or work of music, either vocal or instrumental; NOT applicable to recordings, broadcasts, or individual publications of music in printed or digital form or on physical media",
              "id": "Q105543609",
              "label_en": "musical work/composition"
            },
            {
              "description_en": "sequence of images that give the impression of movement, stored on film stock",
              "id": "Q11424",
              "label_en": "film"
            },
            {
              "description_en": "article in an academic publication, usually peer reviewed",
              "id": "Q13442814",
              "label_en": "scholarly article"
            },
            {
              "description_en": "group of one or more organism(s), which a taxonomist adjudges to be a unit",
              "id": "Q16521",
              "label_en": "taxon"
            },
            {
              "description_en": "use of a creative work across several different media",
              "id": "Q196600",
              "label_en": "media franchise"
            },
            {
              "description_en": "collection of musical recordings released in a specific format for consumption",
              "id": "Q2031291",
              "label_en": "musical release"
            },
            {
              "description_en": "singular named exemplar of an animal (e.g., the gorilla named Koko; the cat named Socks)",
              "id": "Q26401003",
              "label_en": "individual animal"
            },
            {
              "description_en": "visual artwork, surface artistically covered with paint",
              "id": "Q3305213",
              "label_en": "painting"
            },
            {
              "description_en": "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
              "id": "Q3331189",
              "label_en": "version, edition or translation"
            },
            {
              "description_en": "building usually intended for living in",
              "id": "Q3947",
              "label_en": "house"
            },
            {
              "description_en": "structure, typically with a roof and walls, standing more or less permanently in one place",
              "id": "Q41176",
              "label_en": "building"
            },
            {
              "description_en": "social role with a set of powers and responsibilities within an organization",
              "id": "Q4164871",
              "label_en": "position"
            },
            {
              "description_en": "use with 'instance of' (P31) for Wikimedia category",
              "id": "Q4167836",
              "label_en": "Wikimedia category"
            },
            {
              "description_en": "organization undertaking commercial, industrial, or professional activity",
              "id": "Q4830453",
              "label_en": "business"
            },
            {
              "description_en": "place of any size, in which people permanently live",
              "id": "Q486972",
              "label_en": "human settlement"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            {
              "description_en": "small clustered human settlement smaller than a town",
              "id": "Q532",
              "label_en": "village"
            },
            {
              "description_en": "written work read for enjoyment or edification",
              "id": "Q7725634",
              "label_en": "literary work"
            },
            {
              "description_en": "three-dimensional work of art",
              "id": "Q860861",
              "label_en": "sculpture"
            }
          ]
        },
        "rank": "deprecated",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item should have a statement with a given property",
          "id": "Q21510864",
          "label_en": "value-requires-statement constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "anything that can be considered, discussed, or observed",
              "id": "Q35120",
              "label_en": "entity"
            }
          ],
          "P2306": [
            {
              "description_en": "this item is a subclass (subset) of that item; ALL instances of this item are instances of that item; different from P31 (instance of), e.g.: volcano is a subclass of mountain; Everest is an instance of mountain",
              "id": "P279",
              "label_en": "subclass of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the subject and the object have to coincide or coexist at some point of history",
          "id": "Q25796498",
          "label_en": "contemporary constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            },
            {
              "description_en": "entity type in Wikibase",
              "id": "Q29934218",
              "label_en": "Wikibase property"
            },
            {
              "description_en": "Wikibase entity type for lexemes",
              "id": "Q51885771",
              "label_en": "Wikibase lexeme"
            },
            {
              "description_en": "Wikibase property value datatype",
              "id": "Q54285143",
              "label_en": "Wikibase form"
            },
            {
              "description_en": "Wikibase entity type for lexicographic senses",
              "id": "Q54285715",
              "label_en": "Wikibase sense"
            },
            {
              "description_en": "Wikibase entity type for Wikimedia Commons",
              "id": "Q59712033",
              "label_en": "МэдыяІнфа Вікібазы"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "Canadian video game development and consultation studio",
              "id": "Q107327507",
              "label_en": "Sweet Baby Inc."
            }
          ],
          "P2305": [
            {
              "description_en": "company working in the video game industry",
              "id": "Q112042224",
              "label_en": "video game company"
            }
          ],
          "P9729": [
            {
              "description_en": "group or corporation that translates video games",
              "id": "Q100588475",
              "label_en": "video game translation company"
            },
            {
              "description_en": "company that publishes video games",
              "id": "Q1137109",
              "label_en": "video game publisher"
            },
            {
              "description_en": "software development organization specializing in the creation of video games (for person use Q58287519)",
              "id": "Q210167",
              "label_en": "video game developer"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "adherent of the religion of Hinduism",
              "id": "Q10090",
              "label_en": "Hindu"
            },
            {
              "description_en": "person who adheres to Christianity",
              "id": "Q106039",
              "label_en": "Christian"
            },
            {
              "description_en": "member of any of the 24 churches that make up the Roman Catholic Church",
              "id": "Q17549077",
              "label_en": "Catholic"
            },
            {
              "description_en": "adherent of the religion of Islam",
              "id": "Q47740",
              "label_en": "Muslim"
            },
            {
              "description_en": "adherent of the religion of Buddhism",
              "id": "Q6926246",
              "label_en": "Buddhists"
            },
            {
              "description_en": "ethnoreligious group and nation from the Levant",
              "id": "Q7325",
              "label_en": "Jewish people"
            }
          ],
          "P6824": [
            {
              "description_en": "religion of a person, organization or religious building, or associated with this subject",
              "id": "P140",
              "label_en": "religion or worldview"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "fictional character of male sex",
              "id": "Q100911674",
              "label_en": "male character"
            },
            {
              "description_en": "to be used in \"sex or gender\" (P21) to indicate that the human subject is a male or \"semantic gender\" (P10339) to indicate that a word refers to a male person",
              "id": "Q6581097",
              "label_en": "male"
            },
            {
              "description_en": "human of the male sex",
              "id": "Q84048850",
              "label_en": "male human"
            },
            {
              "description_en": "male adult human",
              "id": "Q8441",
              "label_en": "man"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ],
          "P6607": [
            {
              "value": "Das Geschlecht wird über die Eigenschaft P21 spezifiziert.@de"
            },
            {
              "value": "Specify gender using P21 (sex or gender)@en"
            },
            {
              "value": "Utiliser P21 pour le genre ou le sexe@fr"
            }
          ],
          "P6824": [
            {
              "description_en": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
              "id": "P21",
              "label_en": "sex or gender"
            }
          ],
          "P9729": [
            {
              "description_en": "to be used in \"sex or gender\" (P21) to indicate that the human subject is a male or \"semantic gender\" (P10339) to indicate that a word refers to a male person",
              "id": "Q6581097",
              "label_en": "male"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "type of a paperback book",
              "id": "Q1009641",
              "label_en": "bunkobon"
            },
            {
              "description_en": "新書判の叢書・本",
              "id": "Q11502500",
              "label_en": "shinsho"
            },
            {
              "description_en": "book with two works bound back-to-back, rotated 180 degrees",
              "id": "Q124685562",
              "label_en": "tête-bêche"
            },
            {
              "description_en": "Type of hardback",
              "id": "Q12566525",
              "label_en": "Тыс"
            },
            {
              "description_en": "book-length publication in digital form",
              "id": "Q128093",
              "label_en": "ebook"
            },
            {
              "description_en": "electronic publication avilable over a network",
              "id": "Q1294318",
              "label_en": "online book"
            },
            {
              "description_en": "book with pages bound using a metal or plastic coil",
              "id": "Q16929794",
              "label_en": "ring connection"
            },
            {
              "description_en": "small-size book which could fit in a reader's pocket",
              "id": "Q17994250",
              "label_en": "pocket edition"
            },
            {
              "description_en": "book with a paper or paperboard cover",
              "id": "Q193934",
              "label_en": "paperback"
            },
            {
              "description_en": "book bound with a rigid protective cover",
              "id": "Q193955",
              "label_en": "hardcover"
            },
            {
              "description_en": "Japanese term for a book",
              "id": "Q241996",
              "label_en": "tankōbon"
            },
            {
              "description_en": "short, inexpensive booklet; type of street literature printed in early modern Europe",
              "id": "Q2558308",
              "label_en": "chapbook"
            },
            {
              "description_en": "booklet comprised of alternately folded pages",
              "id": "Q361880",
              "label_en": "leporello book"
            },
            {
              "description_en": "quality paperback book",
              "id": "Q990683",
              "label_en": "softcover"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "type of special limited edition of a video game, often comprizing some extra bonuses/addons, or higher quality support medium or packaging",
              "id": "Q101552184",
              "label_en": "collector's edition"
            },
            {
              "description_en": "video game which contains the main game along with all downloadable content and additional content not found in the original release",
              "id": "Q105760475",
              "label_en": "definitive edition"
            },
            {
              "description_en": "type of video game edition",
              "id": "Q107458055",
              "label_en": "director's cut"
            },
            {
              "description_en": "video game edition",
              "id": "Q108028700",
              "label_en": "digital deluxe edition"
            },
            {
              "description_en": "downloadable content that allow one to upgrade the edition of their game",
              "id": "Q108028709",
              "label_en": "video game edition upgrade"
            },
            {
              "description_en": "type of special limited edition of a video game, often comprizing some extra bonuses/addons, or higher quality support medium or packaging",
              "id": "Q108308863",
              "label_en": "anniversary edition"
            },
            {
              "description_en": "type of special limited edition of a software or other electronic media, often comprizing some extra bonuses/addons, or higher quality support medium or packaging",
              "id": "Q58806721",
              "label_en": "collector's edition"
            },
            {
              "description_en": "repackaged version of a video game honoring an award",
              "id": "Q96604496",
              "label_en": "Game of the Year edition"
            }
          ],
          "P6824": [
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "television program adapted from another work",
              "id": "Q101716172",
              "label_en": "television adaptation"
            },
            {
              "description_en": "video game that is adapted from another work",
              "id": "Q117216668",
              "label_en": "video game adaptation"
            },
            {
              "description_en": "films adapted from another work",
              "id": "Q1257444",
              "label_en": "film adaptation"
            },
            {
              "description_en": "feature film based on actual events",
              "id": "Q28146524",
              "label_en": "film based on actual events"
            },
            {
              "description_en": "film where work of literature makes up the basic",
              "id": "Q52162262",
              "label_en": "film based on literature"
            },
            {
              "description_en": "type of film adaptation",
              "id": "Q52207310",
              "label_en": "film based on book"
            },
            {
              "description_en": "film based on a specific literary genre, the novel",
              "id": "Q52207399",
              "label_en": "film based on a novel"
            }
          ],
          "P6607": [
            {
              "value": "Too broad, use a more specific subclass instead and move it to \"has characteristic (P1552)\"@en"
            }
          ],
          "P6824": [
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "type of mill",
              "id": "Q1018126",
              "label_en": "butter-mill"
            },
            {
              "description_en": "wind powered sawmill, commonly found in the Netherlands",
              "id": "Q12013554",
              "label_en": "wind powered sawmill"
            },
            {
              "description_en": "mill to produce ceramic glaze",
              "id": "Q135509402",
              "label_en": "glazuurmolen"
            },
            {
              "description_en": null,
              "id": "Q18640194",
              "label_en": "cocoa mill"
            },
            {
              "description_en": "special paint mill",
              "id": "Q1877290",
              "label_en": "white lead mill"
            },
            {
              "description_en": "molen waar graan to mout voor de jenever- en bierproductie gemalen wordt",
              "id": "Q1976835",
              "label_en": "malting mill"
            },
            {
              "description_en": "machine",
              "id": "Q1987723",
              "label_en": "fulling mill"
            },
            {
              "description_en": "type of mill",
              "id": "Q2084126",
              "label_en": "iron mill"
            },
            {
              "description_en": "Mühle, die zum Schleifen von Werkstoffen dient",
              "id": "Q2238889",
              "label_en": "grinding mill"
            },
            {
              "description_en": "molen die krijt maalt",
              "id": "Q2254524",
              "label_en": "chalk mill"
            },
            {
              "description_en": "mill to produce snuff tobacco",
              "id": "Q2351283",
              "label_en": "snuff mill"
            },
            {
              "description_en": "molen die tufsteen tot tras maalt ten behoeve van de productie van cement en mortel",
              "id": "Q2456069",
              "label_en": "trass mill"
            },
            {
              "description_en": "mill where paints are ground",
              "id": "Q2598061",
              "label_en": "paint mill"
            },
            {
              "description_en": "windpump used to pump water out of a polder",
              "id": "Q2695327",
              "label_en": "polder windmill"
            },
            {
              "description_en": "type of mill",
              "id": "Q2768798",
              "label_en": "smalt mill"
            },
            {
              "description_en": "type of mill where materials are crushed and thus made more flexible or broken",
              "id": "Q2868354",
              "label_en": "stamping mill"
            },
            {
              "description_en": "industrial mill that extracts oil from seeds or vegetable material",
              "id": "Q297163",
              "label_en": "oil mill"
            },
            {
              "description_en": null,
              "id": "Q3257863",
              "label_en": "copper water mill"
            },
            {
              "description_en": "culinary tool for grinding spices",
              "id": "Q356838",
              "label_en": "spice mill"
            },
            {
              "description_en": "also known as Catskill’s mill",
              "id": "Q374116",
              "label_en": "bark mill"
            },
            {
              "description_en": "mill where gunpowder is made from sulfur, saltpeter and charcoal",
              "id": "Q385422",
              "label_en": "powder mill"
            },
            {
              "description_en": "mill used to store and grind whole peppercorns",
              "id": "Q474686",
              "label_en": "pepper mill"
            },
            {
              "description_en": "facilities that are primarily engaged in the manufacture of paper or paper products, usually from wood, recycled paper, or other fiber pulp. Originally water-driven, modern paper mills are typically powered electrically, for use with P366 (has use)",
              "id": "Q56316683",
              "label_en": "paper mill"
            },
            {
              "description_en": "mill where oil is extracted from olives",
              "id": "Q61978174",
              "label_en": "olive oil mill"
            },
            "... omitted 3 items"
          ],
          "P6607": [
            {
              "value": "Use P366 instead for the function(s) the mill had, and use for P31 'mill', 'watermill', 'windmill', etc.@en"
            }
          ],
          "P9729": [
            {
              "description_en": "device that breaks solid materials into smaller pieces by grinding, crushing, or cutting",
              "id": "Q44494",
              "label_en": "mill"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "song protesting against war",
              "id": "Q102189017",
              "label_en": "anti-war song"
            },
            {
              "description_en": "melody with anonymous or unknown composer without authorized original version; melody that is part of a musical tradition",
              "id": "Q105575966",
              "label_en": "traditional melody"
            },
            {
              "description_en": "song, which is parody",
              "id": "Q23817363",
              "label_en": "parody song"
            },
            {
              "description_en": "type of song",
              "id": "Q25452063",
              "label_en": "political song"
            },
            {
              "description_en": "song associated with a sports team and military conflicts",
              "id": "Q261434",
              "label_en": "fight song"
            },
            {
              "description_en": "song that advocates or praises a revolution",
              "id": "Q265147",
              "label_en": "revolutionary song"
            },
            {
              "description_en": "piece of music closely connected to a form of work",
              "id": "Q502658",
              "label_en": "work song"
            },
            {
              "description_en": "song intended to be performed for the Christmas and holiday season",
              "id": "Q56572789",
              "label_en": "Christmas-themed song"
            },
            {
              "description_en": "musical composition considered an important part of the musical repertoire of jazz musicians; composition that is widely known, performed, and recorded by jazz musicians, and widely known by listeners",
              "id": "Q591990",
              "label_en": "jazz standard"
            },
            {
              "description_en": "song that is associated with a movement for social change",
              "id": "Q829147",
              "label_en": "protest song"
            },
            {
              "description_en": "song with anonymous or unknown writer without authorized original version; song that is part of the oral tradition",
              "id": "Q943929",
              "label_en": "traditional folk song"
            }
          ],
          "P6824": [
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            }
          ],
          "P9729": [
            {
              "description_en": "Wikidata metaclass; legal concept of uniquely identifiable piece or work of music, either vocal or instrumental; NOT applicable to recordings, broadcasts, or individual publications of music in printed or digital form or on physical media",
              "id": "Q105543609",
              "label_en": "musical work/composition"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "Christian song of praise with lyrics from biblical or holy texts other than the Psalms",
              "id": "Q1033831",
              "label_en": "canticle"
            },
            {
              "description_en": "musical piece for a single voice as part of a larger work",
              "id": "Q178122",
              "label_en": "aria"
            },
            {
              "description_en": "secular vocal music composition of the Renaissance and early Baroque eras",
              "id": "Q193217",
              "label_en": "madrigal"
            },
            {
              "description_en": "musical form in opera, cantata, mass or oratorio",
              "id": "Q202534",
              "label_en": "recitative"
            }
          ],
          "P6824": [
            {
              "description_en": "structure of a creative work",
              "id": "P7937",
              "label_en": "form of creative work"
            }
          ],
          "P9729": [
            {
              "description_en": "Wikidata metaclass; legal concept of uniquely identifiable piece or work of music, either vocal or instrumental; NOT applicable to recordings, broadcasts, or individual publications of music in printed or digital form or on physical media",
              "id": "Q105543609",
              "label_en": "musical work/composition"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "piece of music to be performed by a single pianist",
              "id": "Q10349334",
              "label_en": "piano solo"
            },
            {
              "description_en": "composition for piano and two instruments, usually a violin and a cello",
              "id": "Q1414262",
              "label_en": "piano trio"
            },
            {
              "description_en": "musical work for two pianists, sometimes with accompanying instruments",
              "id": "Q17126392",
              "label_en": "piano duet"
            },
            {
              "description_en": "piece of music for solo piano",
              "id": "Q1746015",
              "label_en": "piano piece"
            },
            {
              "description_en": "musical composition for 5 performers including a piano; genre of art music played by such groups",
              "id": "Q1746023",
              "label_en": "piano quintet"
            },
            {
              "description_en": "chamber music composition for piano and three other instruments",
              "id": "Q1746025",
              "label_en": "piano quartet"
            },
            {
              "description_en": "composition for piano and five other musical instruments",
              "id": "Q7190206",
              "label_en": "piano sextet"
            }
          ],
          "P6824": [
            {
              "description_en": "structure of a creative work",
              "id": "P7937",
              "label_en": "form of creative work"
            }
          ],
          "P9729": [
            {
              "description_en": "Wikidata metaclass; legal concept of uniquely identifiable piece or work of music, either vocal or instrumental; NOT applicable to recordings, broadcasts, or individual publications of music in printed or digital form or on physical media",
              "id": "Q105543609",
              "label_en": "musical work/composition"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "national heritage site in France",
              "id": "Q10387575",
              "label_en": "monument historique inscrit"
            },
            {
              "description_en": "national heritage site in France",
              "id": "Q10387684",
              "label_en": "classified historical monument"
            },
            {
              "description_en": "collection of Rijksmonumenten designated as a complex",
              "id": "Q13423591",
              "label_en": "Rijksmonument complex"
            },
            {
              "description_en": "national heritage site of the Netherlands",
              "id": "Q916333",
              "label_en": "Rijksmonument"
            },
            {
              "description_en": "protected French building as a Historical Monument (use « classified Historical Monument » and « inscribed Historical Monument »)",
              "id": "Q916475",
              "label_en": "Historical Monument"
            }
          ],
          "P6607": [
            {
              "value": "Use heritage designation (P1435) instead@en"
            }
          ],
          "P6824": [
            {
              "description_en": "heritage designation of a cultural or natural site",
              "id": "P1435",
              "label_en": "heritage designation"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "victim of a crime",
              "id": "Q10436169",
              "label_en": "crime victim"
            },
            {
              "description_en": "person who have been killed or severely wounded by the act or arson",
              "id": "Q107737414",
              "label_en": "arson victim"
            },
            {
              "description_en": "people who have been injured or killed in an act of vigilantism",
              "id": "Q107973786",
              "label_en": "vigilantism victim"
            },
            {
              "description_en": "people who have been killed by the act of homicide",
              "id": "Q108295395",
              "label_en": "homicide victim"
            },
            {
              "description_en": "person/entity held by a belligerent party to another or seized for carrying out agreement",
              "id": "Q192620",
              "label_en": "hostage"
            },
            {
              "description_en": "person who has disappeared and whose status as alive or dead cannot be confirmed",
              "id": "Q388505",
              "label_en": "missing person"
            },
            {
              "description_en": "person who was killed or severely wounded by the act of war crimes",
              "id": "Q46076028",
              "label_en": "war crime victim"
            },
            {
              "description_en": "person who was harmed by the act of rape",
              "id": "Q67630359",
              "label_en": "rape victim"
            },
            {
              "description_en": "person who has been killed by the act of murder",
              "id": "Q73153647",
              "label_en": "murder victim"
            },
            {
              "description_en": "person who was harmed by the act of sexual abuse",
              "id": "Q73155718",
              "label_en": "sexual abuse victim"
            },
            {
              "description_en": "person who was killed or severely wounded in an incident of terrorism",
              "id": "Q73164596",
              "label_en": "terrorism victim"
            },
            {
              "description_en": "person who has been kidnapped",
              "id": "Q73168492",
              "label_en": "kidnapping victim"
            },
            {
              "description_en": "person who was killed or severely wounded by the act of torture",
              "id": "Q73169841",
              "label_en": "torture victim"
            },
            {
              "description_en": "person who was wounded or killed by the act of assault",
              "id": "Q89061424",
              "label_en": "assault victim"
            },
            {
              "description_en": "person who was harmed by the act of sex trafficking or sexual slavery",
              "id": "Q98966335",
              "label_en": "sex trafficking victim"
            },
            {
              "description_en": "person who was harmed by the act of human trafficking",
              "id": "Q98966337",
              "label_en": "human trafficking victim"
            },
            {
              "description_en": "people who have been a victim of or claim to be a victim of cyberbullying at least once",
              "id": "Q98966375",
              "label_en": "cyber bullying victim"
            },
            {
              "description_en": "people who have been injured or killed in a hate crime",
              "id": "Q98966378",
              "label_en": "hate crime victim"
            },
            {
              "description_en": "people who have been affected by identity theft",
              "id": "Q98966385",
              "label_en": "identity theft victim"
            }
          ],
          "P6824": [
            {
              "description_en": "significant or notable events associated with the subject",
              "id": "P793",
              "label_en": "significant event"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "video game produced some time in the past of which no surviving copies or accessible downloads are known to exist",
              "id": "Q104438884",
              "label_en": "lost video game"
            },
            {
              "description_en": "television series produced some time in the past of which no surviving copies are known to exist",
              "id": "Q104438889",
              "label_en": "lost television series"
            },
            {
              "description_en": "television series episode produced some time in the past of which no surviving copies are known to exist.  Use as value for \"signficiant event\" (P793)",
              "id": "Q104438898",
              "label_en": "lost television series episode"
            },
            {
              "description_en": "book produced some time in the past of which no surviving copies are known to exist",
              "id": "Q104438918",
              "label_en": "lost book"
            },
            {
              "description_en": "musical work produced some time in the past of which no surviving copies are known to exist",
              "id": "Q104439050",
              "label_en": "lost musical work"
            },
            {
              "description_en": "radio program produced some time in the past of which no surviving copies are known to exist",
              "id": "Q104439055",
              "label_en": "lost radio program"
            },
            {
              "description_en": "podcast for which the RSS feed and/or episode files are no longer available",
              "id": "Q107737653",
              "label_en": "lost podcast"
            },
            {
              "description_en": "formerly lost media that has since been found",
              "id": "Q122965806",
              "label_en": "formerly lost media"
            },
            {
              "description_en": "podcast episode produced some time in the past of which no surviving copies are known to exist",
              "id": "Q123490564",
              "label_en": "lost podcast episode"
            },
            {
              "description_en": "feature or short film of which no surviving copies are known to exist",
              "id": "Q1268687",
              "label_en": "lost film"
            },
            {
              "description_en": "literary work produced some time in the past of which no surviving copies are known to exist",
              "id": "Q1585442",
              "label_en": "lost literary work"
            },
            {
              "description_en": "piece of art that once existed",
              "id": "Q4140840",
              "label_en": "lost artwork"
            }
          ],
          "P6824": [
            {
              "description_en": "significant or notable events associated with the subject",
              "id": "P793",
              "label_en": "significant event"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "type of anime",
              "id": "Q104541980",
              "label_en": "anime based on a manga"
            },
            {
              "description_en": "type of manga",
              "id": "Q107408274",
              "label_en": "manga based on a anime"
            }
          ],
          "P6824": [
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "musical form, miniature of sonata",
              "id": "Q1045984",
              "label_en": "sonatina"
            },
            {
              "description_en": "type of musical composition, usually for a solo instrument or a small instrumental ensemble",
              "id": "Q131269",
              "label_en": "sonata"
            },
            {
              "description_en": "sonata written for solo piano",
              "id": "Q1546995",
              "label_en": "piano sonata"
            },
            {
              "description_en": "musical composition for clarinet",
              "id": "Q1999051",
              "label_en": "clarinet sonata"
            },
            {
              "description_en": "musical composition for cello and piano",
              "id": "Q2308166",
              "label_en": "sonata for cello and piano"
            },
            {
              "description_en": "sonata for flute and accompanying instrument",
              "id": "Q3070575",
              "label_en": "flute sonata"
            },
            {
              "description_en": "instrumental composition dating from the Baroque period",
              "id": "Q543020",
              "label_en": "sonata da chiesa"
            },
            {
              "description_en": "musical composition for violin",
              "id": "Q7933385",
              "label_en": "violin sonata"
            },
            {
              "description_en": "sonata written for cello",
              "id": "Q857538",
              "label_en": "cello sonata"
            },
            {
              "description_en": "Baroque sonata for two or three melody instruments and continuo",
              "id": "Q903425",
              "label_en": "trio sonata"
            }
          ],
          "P6824": [
            {
              "description_en": "structure of a creative work",
              "id": "P7937",
              "label_en": "form of creative work"
            }
          ],
          "P9729": [
            {
              "description_en": "Wikidata metaclass; legal concept of uniquely identifiable piece or work of music, either vocal or instrumental; NOT applicable to recordings, broadcasts, or individual publications of music in printed or digital form or on physical media",
              "id": "Q105543609",
              "label_en": "musical work/composition"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "type of Japanese video game featuring erotica",
              "id": "Q1046788",
              "label_en": "eroge"
            },
            {
              "description_en": "Japanese pornographic animation, comics, and video games",
              "id": "Q172067",
              "label_en": "hentai"
            },
            {
              "description_en": "interactive fiction game",
              "id": "Q689445",
              "label_en": "visual novel"
            }
          ],
          "P6824": [
            {
              "description_en": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
              "id": "P136",
              "label_en": "genre"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "collection of written creative works chosen by the compiler",
              "id": "Q105420",
              "label_en": "anthology"
            },
            {
              "description_en": "book containing multiple novelettes",
              "id": "Q111180384",
              "label_en": "novelette collection"
            },
            {
              "description_en": "collection of dramas published together.",
              "id": "Q119318561",
              "label_en": "drama collection"
            },
            {
              "description_en": "collection of poems by a single author published together. See also poetry anthology (Q19357149)",
              "id": "Q12106333",
              "label_en": "poem collection"
            },
            {
              "description_en": "book containing several short stories by different authors",
              "id": "Q125544547",
              "label_en": "short story anthology"
            },
            {
              "description_en": "book containing several short stories by a single author",
              "id": "Q1279564",
              "label_en": "short story collection"
            },
            {
              "description_en": "book containing multiple novels",
              "id": "Q133500522",
              "label_en": "novel collection"
            },
            {
              "description_en": "written, fictional, prose narrative normally longer than a short story but shorter than a novel",
              "id": "Q149537",
              "label_en": "novella"
            },
            {
              "description_en": "works from multiple poets chosen by an editor",
              "id": "Q19357149",
              "label_en": "poetry anthology"
            },
            {
              "description_en": "book containing multiple novellas",
              "id": "Q20024995",
              "label_en": "novella collection"
            },
            {
              "description_en": "narrative prose fiction shorter than a novella and longer than a short story",
              "id": "Q472808",
              "label_en": "novelette"
            },
            {
              "description_en": "brief work of literature, usually written in narrative prose",
              "id": "Q49084",
              "label_en": "short story"
            },
            {
              "description_en": "style of fictional literature or fiction of extreme brevity",
              "id": "Q5457615",
              "label_en": "flash fiction"
            },
            {
              "description_en": "publication, usually a book, containing a compilation of letters written by a real person",
              "id": "Q65085460",
              "label_en": "letter collection"
            },
            {
              "description_en": "form of poetry with fourteen lines and strict rhyming structure",
              "id": "Q80056",
              "label_en": "sonnet"
            },
            {
              "description_en": "narrative text, normally of a substantial length and in the form of prose describing a fictional and sequential story",
              "id": "Q8261",
              "label_en": "novel"
            }
          ],
          "P6607": [
            {
              "value": "Move to P7937 and use Q7725634 as value instead@en"
            },
            {
              "value": "À déplacer vers P7937 et utiliser  Q7725634 à la place@fr"
            }
          ],
          "P6824": [
            {
              "description_en": "structure of a creative work",
              "id": "P7937",
              "label_en": "form of creative work"
            }
          ],
          "P9729": [
            {
              "description_en": "written work read for enjoyment or edification",
              "id": "Q7725634",
              "label_en": "literary work"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "opera that depicts aspects of ‘simple’ (generally rural) life, usually in opposition to that of the court or city, or that is expressive of its atmosphere or values",
              "id": "Q105906205",
              "label_en": "pastoral opera"
            },
            {
              "description_en": "opera genre",
              "id": "Q13220650",
              "label_en": "comic opera"
            },
            {
              "description_en": "opera genre",
              "id": "Q1457481",
              "label_en": "Spieloper"
            },
            {
              "description_en": "theatrical entertainment which began in Paris towards the end of the 17th century",
              "id": "Q17084138",
              "label_en": "comédie en vaudevilles"
            },
            {
              "description_en": "Italian opera genre",
              "id": "Q208080",
              "label_en": "opera buffa"
            },
            {
              "description_en": "style of Italian opera",
              "id": "Q210675",
              "label_en": "opera seria"
            },
            {
              "description_en": "French opérette subgenre",
              "id": "Q24678689",
              "label_en": "vaudeville-opérette"
            },
            {
              "description_en": "French opera genre",
              "id": "Q3084465",
              "label_en": "opéra bouffe"
            },
            {
              "description_en": "opera genre",
              "id": "Q377258",
              "label_en": "singspiel"
            },
            {
              "description_en": "French versions of Italian opera buffa",
              "id": "Q7099147",
              "label_en": "opéra bouffon"
            },
            {
              "description_en": "musical theatre work with rock music",
              "id": "Q7354827",
              "label_en": "rock musical"
            },
            {
              "description_en": "genre of French opera",
              "id": "Q785479",
              "label_en": "opéra comique"
            },
            {
              "description_en": "french opera genre",
              "id": "Q908705",
              "label_en": "tragédie en musique"
            }
          ],
          "P6607": [
            {
              "value": "Use dramatico-musical work (Q58483083) for P31, and use the specific type with P136 (genre)@en"
            }
          ],
          "P6824": [
            {
              "description_en": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
              "id": "P136",
              "label_en": "genre"
            }
          ],
          "P9729": [
            {
              "description_en": "opera, musical play or show, revue or pantomime for which music has been specially written; for ballet use \"choreographic work\" (Q58483088)",
              "id": "Q58483083",
              "label_en": "dramatico-musical work"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "目的語なしで用いられたり，本来の目的語が主語の位置に移された他動詞",
              "id": "Q105933250",
              "label_en": "pseudo-transitive verb"
            },
            {
              "description_en": "verb that takes no grammatical objects",
              "id": "Q1166153",
              "label_en": "intransitive verb"
            },
            {
              "description_en": "verb that requires one or more objects in a sentence",
              "id": "Q1774805",
              "label_en": "transitive verb"
            },
            {
              "description_en": "verb that may or may not take a grammatical object without changing form",
              "id": "Q4115075",
              "label_en": "ambitransitive verb"
            },
            {
              "description_en": "ambitransitive verb whose subject when intransitive corresponds to its direct object when transitive",
              "id": "Q623554",
              "label_en": "ergative verb"
            }
          ],
          "P6824": [
            {
              "description_en": "possible grammatical property of a verb accepting an object complement",
              "id": "P9295",
              "label_en": "transitivity"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      "... omitted 173 items"
    ]
  },
  "hash_after": "a812be9148ef19ef44ea6a8a21b9f347000b2c31",
  "hash_before": "ce4232e77bc7d6290fc000803a238c5704d79301",
  "property_revision_id": 2440675178,
  "property_revision_prev": 2439474684,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Use \"album\" (Q482994) with \"instance of\" (P31), use the specific form with \"form of creative work\" (P7937).@en"
      ],
      "constraint_qid": "Q52558054",
      "qualifier_property": "P6607",
      "removed_values": [
        "Use Q482994 with P31, use the specific form with P7937@en"
      ],
      "same_qid_index": 27
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q52558054",
      "qualifiers": [
        {
          "property_id": "P2305",
          "values": [
            "Q108064011",
            "Q208569",
            "Q209939",
            "Q222910",
            "Q4176708",
            "Q4712779",
            "Q60030240",
            "Q68902449"
          ]
        },
        {
          "property_id": "P6607",
          "values": [
            "Use Q482994 with P31, use the specific form with P7937@en"
          ]
        },
        {
          "property_id": "P6824",
          "values": [
            "P7937"
          ]
        },
        {
          "property_id": "P9729",
          "values": [
            "Q482994"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "allowed qualifiers constraint: property: applies to jurisdiction, field of work, excluding, including, criterion used, individual of taxon, conferred by, product or material produced, applies to work, proportion, quantity, league or competition, connecting service, simulates, latest end date, valid in period, represents, facet of, object of occurrence, class of object(s) of occurrence, objects of occurrence have role, role of agent(s) of action, class of agent(s) ... [truncated 1693 chars]",
      "one-of constraint: reason for deprecated rank: constraint provides suggestions for manual input; item of property constraint: family name, musical work/composition, film, scholarly article, taxon, media franchise, musical release, individual animal, painting, version, edition or translation, house, building, position, Wikimedia category, business, human settlement, human, village, literary work, sculpture",
      "value-requires-statement constraint: exception to constraint: entity; property: subclass of",
      "contemporary constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, Wikibase property, Wikibase lexeme, Wikibase form, Wikibase sense, МэдыяІнфа Вікібазы; constraint status: mandatory constraint",
      "none-of constraint: exception to constraint: Sweet Baby Inc.; item of property constraint: video game company; replacement value: video game translation company, video game publisher, video game developer",
      "none-of constraint: item of property constraint: Hindu, Christian, Catholic, Muslim, Buddhists, Jewish people; replacement property: religion or worldview",
      "none-of constraint: item of property constraint: male character, male, male human, man; constraint status: mandatory constraint; constraint clarification: Das Geschlecht wird über die Eigenschaft P21 spezifiziert.@de, Specify gender using P21 (sex or gender)@en, Utiliser P21 pour le genre ou le sexe@fr; replacement property: sex or gender; replacement value: male",
      "none-of constraint: item of property constraint: bunkobon, shinsho, tête-bêche, Тыс, ebook, online book, ring connection, pocket edition, paperback, hardcover, tankōbon, chapbook, leporello book, softcover",
      "none-of constraint: item of property constraint: collector's edition, definitive edition, director's cut, digital deluxe edition, video game edition upgrade, anniversary edition, collector's edition, Game of the Year edition; replacement property: has characteristic",
      "none-of constraint: item of property constraint: television adaptation, video game adaptation, film adaptation, film based on actual events, film based on literature, film based on book, film based on a novel; constraint clarification: Too broad, use a more specific subclass instead and move it to \"has characteristic (P1552)\"@en; replacement property: has characteristic",
      "none-of constraint: item of property constraint: butter-mill, wind powered sawmill, glazuurmolen, cocoa mill, white lead mill, malting mill, fulling mill, iron mill, grinding mill, chalk mill, snuff mill, trass mill, paint mill, polder windmill, smalt mill, stamping mill, oil mill, copper water mill, spice mill, bark mill, powder mill, pepper mill, paper mill, olive oil mill, gristmill, mustard mill, historical hulling mill; constraint clarification: Use P366 inst... [truncated 119 chars]",
      "none-of constraint: item of property constraint: anti-war song, traditional melody, parody song, political song, fight song, revolutionary song, work song, Christmas-themed song, jazz standard, protest song, traditional folk song; replacement property: has characteristic; replacement value: musical work/composition",
      "none-of constraint: item of property constraint: canticle, aria, madrigal, recitative; replacement property: form of creative work; replacement value: musical work/composition",
      "none-of constraint: item of property constraint: piano solo, piano trio, piano duet, piano piece, piano quintet, piano quartet, piano sextet; replacement property: form of creative work; replacement value: musical work/composition",
      "none-of constraint: item of property constraint: monument historique inscrit, classified historical monument, Rijksmonument complex, Rijksmonument, Historical Monument; constraint clarification: Use heritage designation (P1435) instead@en; replacement property: heritage designation",
      "none-of constraint: item of property constraint: crime victim, arson victim, vigilantism victim, homicide victim, hostage, missing person, war crime victim, rape victim, murder victim, sexual abuse victim, terrorism victim, kidnapping victim, torture victim, assault victim, sex trafficking victim, human trafficking victim, cyber bullying victim, hate crime victim, identity theft victim; replacement property: significant event",
      "none-of constraint: item of property constraint: lost video game, lost television series, lost television series episode, lost book, lost musical work, lost radio program, lost podcast, formerly lost media, lost podcast episode, lost film, lost literary work, lost artwork; replacement property: significant event",
      "none-of constraint: item of property constraint: anime based on a manga, manga based on a anime; replacement property: has characteristic",
      "none-of constraint: item of property constraint: sonatina, sonata, piano sonata, clarinet sonata, sonata for cello and piano, flute sonata, sonata da chiesa, violin sonata, cello sonata, trio sonata; replacement property: form of creative work; replacement value: musical work/composition",
      "none-of constraint: item of property constraint: eroge, hentai, visual novel; replacement property: genre",
      "none-of constraint: item of property constraint: anthology, novelette collection, drama collection, poem collection, short story anthology, short story collection, novel collection, novella, poetry anthology, novella collection, novelette, short story, flash fiction, letter collection, sonnet, novel; constraint clarification: Move to P7937 and use Q7725634 as value instead@en, À déplacer vers P7937 et utiliser  Q7725634 à la place@fr; replacement property: form of... [truncated 48 chars]",
      "none-of constraint: item of property constraint: pastoral opera, comic opera, Spieloper, comédie en vaudevilles, opera buffa, opera seria, vaudeville-opérette, opéra bouffe, singspiel, opéra bouffon, rock musical, opéra comique, tragédie en musique; constraint clarification: Use dramatico-musical work (Q58483083) for P31, and use the specific type with P136 (genre)@en; replacement property: genre; replacement value: dramatico-musical work",
      "none-of constraint: item of property constraint: pseudo-transitive verb, intransitive verb, transitive verb, ambitransitive verb, ergative verb; replacement property: transitivity",
      "... omitted 173 items"
    ],
    "before": [
      "allowed qualifiers constraint: property: applies to jurisdiction, field of work, excluding, including, criterion used, individual of taxon, conferred by, product or material produced, applies to work, proportion, quantity, league or competition, connecting service, simulates, latest end date, valid in period, represents, facet of, object of occurrence, class of object(s) of occurrence, objects of occurrence have role, role of agent(s) of action, class of agent(s) ... [truncated 1693 chars]",
      "one-of constraint: reason for deprecated rank: constraint provides suggestions for manual input; item of property constraint: family name, musical work/composition, film, scholarly article, taxon, media franchise, musical release, individual animal, painting, version, edition or translation, house, building, position, Wikimedia category, business, human settlement, human, village, literary work, sculpture",
      "value-requires-statement constraint: exception to constraint: entity; property: subclass of",
      "contemporary constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, Wikibase property, Wikibase lexeme, Wikibase form, Wikibase sense, МэдыяІнфа Вікібазы; constraint status: mandatory constraint",
      "none-of constraint: exception to constraint: Sweet Baby Inc.; item of property constraint: video game company; replacement value: video game translation company, video game publisher, video game developer",
      "none-of constraint: item of property constraint: Hindu, Christian, Catholic, Muslim, Buddhists, Jewish people; replacement property: religion or worldview",
      "none-of constraint: item of property constraint: male character, male, male human, man; constraint status: mandatory constraint; constraint clarification: Das Geschlecht wird über die Eigenschaft P21 spezifiziert.@de, Specify gender using P21 (sex or gender)@en, Utiliser P21 pour le genre ou le sexe@fr; replacement property: sex or gender; replacement value: male",
      "none-of constraint: item of property constraint: bunkobon, shinsho, tête-bêche, Тыс, ebook, online book, ring connection, pocket edition, paperback, hardcover, tankōbon, chapbook, leporello book, softcover",
      "none-of constraint: item of property constraint: collector's edition, definitive edition, director's cut, digital deluxe edition, video game edition upgrade, anniversary edition, collector's edition, Game of the Year edition; replacement property: has characteristic",
      "none-of constraint: item of property constraint: television adaptation, video game adaptation, film adaptation, film based on actual events, film based on literature, film based on book, film based on a novel; constraint clarification: Too broad, use a more specific subclass instead and move it to \"has characteristic (P1552)\"@en; replacement property: has characteristic",
      "none-of constraint: item of property constraint: butter-mill, wind powered sawmill, glazuurmolen, cocoa mill, white lead mill, malting mill, fulling mill, iron mill, grinding mill, chalk mill, snuff mill, trass mill, paint mill, polder windmill, smalt mill, stamping mill, oil mill, copper water mill, spice mill, bark mill, powder mill, pepper mill, paper mill, olive oil mill, gristmill, mustard mill, historical hulling mill; constraint clarification: Use P366 inst... [truncated 119 chars]",
      "none-of constraint: item of property constraint: anti-war song, traditional melody, parody song, political song, fight song, revolutionary song, work song, Christmas-themed song, jazz standard, protest song, traditional folk song; replacement property: has characteristic; replacement value: musical work/composition",
      "none-of constraint: item of property constraint: canticle, aria, madrigal, recitative; replacement property: form of creative work; replacement value: musical work/composition",
      "none-of constraint: item of property constraint: piano solo, piano trio, piano duet, piano piece, piano quintet, piano quartet, piano sextet; replacement property: form of creative work; replacement value: musical work/composition",
      "none-of constraint: item of property constraint: monument historique inscrit, classified historical monument, Rijksmonument complex, Rijksmonument, Historical Monument; constraint clarification: Use heritage designation (P1435) instead@en; replacement property: heritage designation",
      "none-of constraint: item of property constraint: crime victim, arson victim, vigilantism victim, homicide victim, hostage, missing person, war crime victim, rape victim, murder victim, sexual abuse victim, terrorism victim, kidnapping victim, torture victim, assault victim, sex trafficking victim, human trafficking victim, cyber bullying victim, hate crime victim, identity theft victim; replacement property: significant event",
      "none-of constraint: item of property constraint: lost video game, lost television series, lost television series episode, lost book, lost musical work, lost radio program, lost podcast, formerly lost media, lost podcast episode, lost film, lost literary work, lost artwork; replacement property: significant event",
      "none-of constraint: item of property constraint: anime based on a manga, manga based on a anime; replacement property: has characteristic",
      "none-of constraint: item of property constraint: sonatina, sonata, piano sonata, clarinet sonata, sonata for cello and piano, flute sonata, sonata da chiesa, violin sonata, cello sonata, trio sonata; replacement property: form of creative work; replacement value: musical work/composition",
      "none-of constraint: item of property constraint: eroge, hentai, visual novel; replacement property: genre",
      "none-of constraint: item of property constraint: anthology, novelette collection, drama collection, poem collection, short story anthology, short story collection, novel collection, novella, poetry anthology, novella collection, novelette, short story, flash fiction, letter collection, sonnet, novel; constraint clarification: Move to P7937 and use Q7725634 as value instead@en, À déplacer vers P7937 et utiliser  Q7725634 à la place@fr; replacement property: form of... [truncated 48 chars]",
      "none-of constraint: item of property constraint: pastoral opera, comic opera, Spieloper, comédie en vaudevilles, opera buffa, opera seria, vaudeville-opérette, opéra bouffe, singspiel, opéra bouffon, rock musical, opéra comique, tragédie en musique; constraint clarification: Use dramatico-musical work (Q58483083) for P31, and use the specific type with P136 (genre)@en; replacement property: genre; replacement value: dramatico-musical work",
      "none-of constraint: item of property constraint: pseudo-transitive verb, intransitive verb, transitive verb, ambitransitive verb, ergative verb; replacement property: transitivity",
      "... omitted 173 items"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q52558054",
    "result": false,
    "step": "causality_filter",
    "violation_name": "None of"
  }
]
```

---

## 016. `reform_Q49815542_P4866_696601643`

| Field | Value |
|---|---|
| qid | Q49815542 |
| property | P4866 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | TBOX::P4866::696601643 |
| tbox_revision_key | TBOX::P4866::696601643 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "MisterSynergy",
  "kind": "T_BOX",
  "property_revision_id": 696601643,
  "property_revision_prev": 683457669
}
```

### Violation Context

```json
{
  "report_fix_date": "2018-06-19T20:09:38",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4866",
  "report_revision_new": 698737102,
  "report_revision_old": 698423991,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "2569"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "ID in REBASE (Restriction Enzyme Database)",
    "label": "REBASE Enzyme Number"
  },
  "qid": {
    "description": "restriction enzyme",
    "label": "AceI"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q53869507",
      "qualifiers": [
        {
          "property_id": "P2316",
          "values": [
            "Q21502408"
          ]
        },
        {
          "property_id": "P5314",
          "values": [
            "Q54828448",
            "Q54828450"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 5,
  "author": "MisterSynergy",
  "before_constraint_count": 5,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[1-9]\\d*"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "class of enzymes that cleaves DNA into fragments at or near specific recognition sites within the molecule known as restriction sites",
              "id": "Q219715",
              "label_en": "restriction enzyme"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type constraint",
              "id": "Q21514624",
              "label_en": "subclass of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ],
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[1-9]\\d*"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "class of enzymes that cleaves DNA into fragments at or near specific recognition sites within the molecule known as restriction sites",
              "id": "Q219715",
              "label_en": "restriction enzyme"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type constraint",
              "id": "Q21514624",
              "label_en": "subclass of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ],
          "P4680": [
            {
              "description_en": "scope for constraints that should be checked on the main value of a statement",
              "id": "Q46466787",
              "label_en": "constraint checked on main value"
            },
            {
              "description_en": "scope for constraints that should be checked on the references of a statement",
              "id": "Q46466805",
              "label_en": "constraint checked on references"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "fadc01a02b68826671c0897391ff329efe3c79f9",
  "hash_before": "0c6894c33808e43db305ac6d55f1aec969bf9da6",
  "property_revision_id": 696601643,
  "property_revision_prev": 683457669,
  "qualifier_value_changes": [
    {
      "added_values": [],
      "constraint_qid": "Q53869507",
      "qualifier_property": "P4680",
      "removed_values": [
        "Q46466787",
        "Q46466805"
      ],
      "same_qid_index": 0
    },
    {
      "added_values": [
        "Q54828448",
        "Q54828450"
      ],
      "constraint_qid": "Q53869507",
      "qualifier_property": "P5314",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q53869507",
      "qualifiers": [
        {
          "property_id": "P2316",
          "values": [
            "Q21502408"
          ]
        },
        {
          "property_id": "P4680",
          "values": [
            "Q46466787",
            "Q46466805"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: [1-9]\\d*",
      "distinct-values constraint: no qualifiers recorded",
      "subject type constraint: class: restriction enzyme; relation: subclass of",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as reference"
    ],
    "before": [
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: [1-9]\\d*",
      "distinct-values constraint: no qualifiers recorded",
      "subject type constraint: class: restriction enzyme; relation: subclass of",
      "property scope constraint: constraint status: mandatory constraint; constraint scope: constraint checked on main value, constraint checked on references"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q21502410",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Unique value"
  }
]
```

---

## 017. `reform_Q55846055_P282_1707966882`

| Field | Value |
|---|---|
| qid | Q55846055 |
| property | P282 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q21510865 |
| group_key | TBOX::P282::1707966882 |
| tbox_revision_key | TBOX::P282::1707966882 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "عُثمان",
  "kind": "T_BOX",
  "property_revision_id": 1707966882,
  "property_revision_prev": 1707966314
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-08-21T13:55:48",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P282",
  "report_revision_new": 1709513560,
  "report_revision_old": 1708709995,
  "report_violation_type": "One of",
  "report_violation_type_normalized": "One of",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "One of",
  "value": null,
  "value_current_2026": [
    "Q8201",
    "Q1147857"
  ],
  "value_current_2026_descriptions_en": [
    "logographic writing system with Han origin used in the Sinosphere for Chinese, Japanese, Korean and traditional Vietnamese languages",
    "traditional form of kanji used before 1946"
  ],
  "value_current_2026_labels_en": [
    "Chinese characters",
    "kyūjitai"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface",
    "label": "writing system"
  },
  "qid": {
    "description": "CJK (hanzi/kanji/hanja) character",
    "label": "海 (U+FA45)"
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
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q53869507",
      "qualifiers": [
        {
          "property_id": "P5314",
          "values": [
            "Q54828448",
            "Q54828449",
            "Q54828450"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 4,
  "author": "عُثمان",
  "before_constraint_count": 3,
  "changed_constraint_types": [
    "Q53869507"
  ],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to be one of a given set of items",
          "id": "Q21510859",
          "label_en": "one-of constraint"
        },
        "parameters": {
          "P2241": [
            {
              "description_en": "reason for deprecation of a Wikidata property constraint",
              "id": "Q99460987",
              "label_en": "constraint provides suggestions for manual input"
            }
          ],
          "P2305": [
            {
              "description_en": "modified form of Arabic script which forms the basis of Arabic script orthography for the Persian language and in turn several other Indo-Aryan languages",
              "id": "Q112887344",
              "label_en": "Perso-Arabic script"
            },
            {
              "description_en": "Perso-Arabic script based orthography for Saraiki, extended from the Shahmukhi orthography for Punjabi",
              "id": "Q113406611",
              "label_en": "Saraiki Shahmukhi"
            },
            {
              "description_en": "Latin script based orthography for writing Saraiki",
              "id": "Q113406959",
              "label_en": "Saraiki Latin"
            },
            {
              "description_en": "standard Perso-Arabic script based orthography for writing the Balochi language in Iran",
              "id": "Q113557437",
              "label_en": "Balochi Standard Orthography"
            },
            {
              "description_en": "alphabet used to write the Armenian language",
              "id": "Q11932",
              "label_en": "Armenian alphabet"
            },
            {
              "description_en": "writing system for Punjabi using a Perso-Arabic based script",
              "id": "Q133800",
              "label_en": "Shahmukhi"
            },
            {
              "description_en": "alphabetic writing systems mostly used to transcribe the Georgian language and other languages of the Caucasus region",
              "id": "Q161428",
              "label_en": "Georgian alphabet"
            },
            {
              "description_en": "Semitic alphabet used for writing Hebrew, Samaritan, Yiddish, Judaeo-Spanish, and other Jewish languages",
              "id": "Q33513",
              "label_en": "Hebrew alphabet"
            },
            {
              "description_en": "writing system used to write most North Indian and Nepalese languages",
              "id": "Q38592",
              "label_en": "Devanagari"
            },
            {
              "description_en": "Brahmic script used to write the Punjabi language; commonly used to write Punjabi in India; prominent component of Sikh religious literature",
              "id": "Q689894",
              "label_en": "Gurmukhi"
            },
            {
              "description_en": "alphabet specifically codified for writing the Arabic language",
              "id": "Q8196",
              "label_en": "Arabic alphabet"
            },
            {
              "description_en": "logographic writing system with Han origin used in the Sinosphere for Chinese, Japanese, Korean and traditional Vietnamese languages",
              "id": "Q8201",
              "label_en": "Chinese characters"
            },
            {
              "description_en": "writing system developed in Bulgaria and used for various oriental Eurasian languages",
              "id": "Q8209",
              "label_en": "Cyrillic script"
            },
            {
              "description_en": "alphabet used to write the ancient or modern Greek language",
              "id": "Q8216",
              "label_en": "Greek alphabet"
            },
            {
              "description_en": "native alphabet of the Korean language",
              "id": "Q8222",
              "label_en": "Hangul"
            },
            {
              "description_en": "writing system used to write most Western, Northern and Central European languages",
              "id": "Q8229",
              "label_en": "Latin script"
            },
            {
              "description_en": "adopted logographic Chinese characters used in the modern Japanese writing system",
              "id": "Q82772",
              "label_en": "kanji"
            }
          ]
        },
        "rank": "deprecated",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "transcription method that employs the standard spelling system of each target language",
              "id": "Q1055909",
              "label_en": "orthographic transcription"
            },
            {
              "description_en": "set of conventions for writing a language",
              "id": "Q43091",
              "label_en": "orthography"
            },
            {
              "description_en": "expression of a language in a particular alphabet",
              "id": "Q64362969",
              "label_en": "language in script"
            },
            {
              "description_en": "any conventional method of visually representing verbal or signed communication",
              "id": "Q8192",
              "label_en": "writing system"
            },
            {
              "description_en": "standard set of letters present in some written languages",
              "id": "Q9779",
              "label_en": "alphabet"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type constraints",
              "id": "Q21503252",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            },
            {
              "description_en": "Wikibase entity type for lexemes",
              "id": "Q51885771",
              "label_en": "Wikibase lexeme"
            },
            {
              "description_en": "Wikibase property value datatype",
              "id": "Q54285143",
              "label_en": "Wikibase form"
            },
            {
              "description_en": "Wikibase entity type for lexicographic senses",
              "id": "Q54285715",
              "label_en": "Wikibase sense"
            },
            {
              "description_en": "Wikibase entity type for Wikimedia Commons",
              "id": "Q59712033",
              "label_en": "МэдыяІнфа Вікібазы"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828449",
              "label_en": "as qualifier"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to be one of a given set of items",
          "id": "Q21510859",
          "label_en": "one-of constraint"
        },
        "parameters": {
          "P2241": [
            {
              "description_en": "reason for deprecation of a Wikidata property constraint",
              "id": "Q99460987",
              "label_en": "constraint provides suggestions for manual input"
            }
          ],
          "P2305": [
            {
              "description_en": "modified form of Arabic script which forms the basis of Arabic script orthography for the Persian language and in turn several other Indo-Aryan languages",
              "id": "Q112887344",
              "label_en": "Perso-Arabic script"
            },
            {
              "description_en": "Perso-Arabic script based orthography for Saraiki, extended from the Shahmukhi orthography for Punjabi",
              "id": "Q113406611",
              "label_en": "Saraiki Shahmukhi"
            },
            {
              "description_en": "Latin script based orthography for writing Saraiki",
              "id": "Q113406959",
              "label_en": "Saraiki Latin"
            },
            {
              "description_en": "standard Perso-Arabic script based orthography for writing the Balochi language in Iran",
              "id": "Q113557437",
              "label_en": "Balochi Standard Orthography"
            },
            {
              "description_en": "alphabet used to write the Armenian language",
              "id": "Q11932",
              "label_en": "Armenian alphabet"
            },
            {
              "description_en": "writing system for Punjabi using a Perso-Arabic based script",
              "id": "Q133800",
              "label_en": "Shahmukhi"
            },
            {
              "description_en": "alphabetic writing systems mostly used to transcribe the Georgian language and other languages of the Caucasus region",
              "id": "Q161428",
              "label_en": "Georgian alphabet"
            },
            {
              "description_en": "Semitic alphabet used for writing Hebrew, Samaritan, Yiddish, Judaeo-Spanish, and other Jewish languages",
              "id": "Q33513",
              "label_en": "Hebrew alphabet"
            },
            {
              "description_en": "writing system used to write most North Indian and Nepalese languages",
              "id": "Q38592",
              "label_en": "Devanagari"
            },
            {
              "description_en": "Brahmic script used to write the Punjabi language; commonly used to write Punjabi in India; prominent component of Sikh religious literature",
              "id": "Q689894",
              "label_en": "Gurmukhi"
            },
            {
              "description_en": "alphabet specifically codified for writing the Arabic language",
              "id": "Q8196",
              "label_en": "Arabic alphabet"
            },
            {
              "description_en": "logographic writing system with Han origin used in the Sinosphere for Chinese, Japanese, Korean and traditional Vietnamese languages",
              "id": "Q8201",
              "label_en": "Chinese characters"
            },
            {
              "description_en": "writing system developed in Bulgaria and used for various oriental Eurasian languages",
              "id": "Q8209",
              "label_en": "Cyrillic script"
            },
            {
              "description_en": "alphabet used to write the ancient or modern Greek language",
              "id": "Q8216",
              "label_en": "Greek alphabet"
            },
            {
              "description_en": "native alphabet of the Korean language",
              "id": "Q8222",
              "label_en": "Hangul"
            },
            {
              "description_en": "writing system used to write most Western, Northern and Central European languages",
              "id": "Q8229",
              "label_en": "Latin script"
            },
            {
              "description_en": "adopted logographic Chinese characters used in the modern Japanese writing system",
              "id": "Q82772",
              "label_en": "kanji"
            }
          ]
        },
        "rank": "deprecated",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "transcription method that employs the standard spelling system of each target language",
              "id": "Q1055909",
              "label_en": "orthographic transcription"
            },
            {
              "description_en": "set of conventions for writing a language",
              "id": "Q43091",
              "label_en": "orthography"
            },
            {
              "description_en": "expression of a language in a particular alphabet",
              "id": "Q64362969",
              "label_en": "language in script"
            },
            {
              "description_en": "any conventional method of visually representing verbal or signed communication",
              "id": "Q8192",
              "label_en": "writing system"
            },
            {
              "description_en": "standard set of letters present in some written languages",
              "id": "Q9779",
              "label_en": "alphabet"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type constraints",
              "id": "Q21503252",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            },
            {
              "description_en": "Wikibase entity type for lexemes",
              "id": "Q51885771",
              "label_en": "Wikibase lexeme"
            },
            {
              "description_en": "Wikibase property value datatype",
              "id": "Q54285143",
              "label_en": "Wikibase form"
            },
            {
              "description_en": "Wikibase entity type for lexicographic senses",
              "id": "Q54285715",
              "label_en": "Wikibase sense"
            },
            {
              "description_en": "Wikibase entity type for Wikimedia Commons",
              "id": "Q59712033",
              "label_en": "МэдыяІнфа Вікібазы"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "282fa125cd4e285b482bd691b211bd14fdc7f383",
  "hash_before": "620fed1a7a66089120fd787a677da8e49f56e3f0",
  "property_revision_id": 1707966882,
  "property_revision_prev": 1707966314,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": {
    "after": [
      "one-of constraint: reason for deprecated rank: constraint provides suggestions for manual input; item of property constraint: Perso-Arabic script, Saraiki Shahmukhi, Saraiki Latin, Balochi Standard Orthography, Armenian alphabet, Shahmukhi, Georgian alphabet, Hebrew alphabet, Devanagari, Gurmukhi, Arabic alphabet, Chinese characters, Cyrillic script, Greek alphabet, Hangul, Latin script, kanji",
      "value-type constraint: class: orthographic transcription, orthography, language in script, writing system, alphabet; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, Wikibase lexeme, Wikibase form, Wikibase sense, МэдыяІнфа Вікібазы",
      "property scope constraint: property scope: as main value, as qualifier, as reference"
    ],
    "before": [
      "one-of constraint: reason for deprecated rank: constraint provides suggestions for manual input; item of property constraint: Perso-Arabic script, Saraiki Shahmukhi, Saraiki Latin, Balochi Standard Orthography, Armenian alphabet, Shahmukhi, Georgian alphabet, Hebrew alphabet, Devanagari, Gurmukhi, Arabic alphabet, Chinese characters, Cyrillic script, Greek alphabet, Hangul, Latin script, kanji",
      "value-type constraint: class: orthographic transcription, orthography, language in script, writing system, alphabet; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, Wikibase lexeme, Wikibase form, Wikibase sense, МэдыяІнфа Вікібазы"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [
      "Q53869507"
    ],
    "mapped_constraint_qid": "Q21510859",
    "result": false,
    "step": "causality_filter",
    "violation_name": "One of"
  }
]
```

---

## 018. `reform_Q56443809_P356_2447249541`

| Field | Value |
|---|---|
| qid | Q56443809 |
| property | P356 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | TBOX::P356::2447249541 |
| tbox_revision_key | TBOX::P356::2447249541 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Aceehinrt",
  "kind": "T_BOX",
  "property_revision_id": 2447249541,
  "property_revision_prev": 2445410203
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-26T13:11:24",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P356",
  "report_revision_new": 2447384258,
  "report_revision_old": 2447071553,
  "report_violation_type": "Format",
  "report_violation_type_normalized": "Format",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Format",
  "value": null,
  "value_current_2026": [
    "10.1145/1899661.1869637"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "serial code used to uniquely identify digital objects like academic papers (use upper case letters only)",
    "label": "DOI"
  },
  "qid": {
    "description": "наукова стаття, опублікована в грудні 2010",
    "label": "Factor"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 14,
  "author": "Aceehinrt",
  "before_constraint_count": 15,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
            {
              "description_en": "organisation or other agent that issues or allocates an identifier, code, classification number, etc.",
              "id": "P2378",
              "label_en": "issued by"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(10\\.[0-9]{4,}(?:\\.[0-9]+)*/(?:(?![\"&'])\\S)+)|"
            }
          ],
          "P2303": [
            {
              "description_en": "rigorózní práce",
              "id": "Q107814480",
              "label_en": "České město v procesu modernizace mezi lety 1850 až 1914 na příkladu Kolína"
            },
            {
              "description_en": "diplomová práce",
              "id": "Q108532806",
              "label_en": "František Norbert Hrachovský, O.Praem (1879-1943)"
            },
            {
              "description_en": "2015年書籍",
              "id": "Q112031173",
              "label_en": "鋼鐵風景：宋璽德"
            },
            {
              "description_en": "scientific article published on August 2006",
              "id": "Q39347033",
              "label_en": "Surgical simulation of instrumented posterior occipitocervical fusion in a child with congenital skeletal anomaly: case report"
            },
            {
              "description_en": "scientific article published on May 2006",
              "id": "Q42684434",
              "label_en": "Operative failure of percutaneous endoscopic lumbar discectomy: a radiologic analysis of 55 cases"
            },
            {
              "description_en": "wetenschappelijk artikel",
              "id": "Q56806485",
              "label_en": "A biomimetic study of the explosive discharge of the bombardier beetle"
            },
            {
              "description_en": "thesis published in 2014",
              "id": "Q58222124",
              "label_en": "Woman-to-woman sexual assault : a situational analysis"
            },
            {
              "description_en": "wetenschappelijk artikel",
              "id": "Q58850552",
              "label_en": "Computing urban mobile landscapes through monitoring population density based on cell-phone chatting"
            },
            {
              "description_en": "scientific article published on 22 August 2008",
              "id": "Q62139877",
              "label_en": "On the relationship between energy efficiency and complexity: insight on the causality chain"
            },
            {
              "description_en": "thèse de doctorat de Marcel Hamelin soumise à l'Université Laval",
              "id": "Q63646670",
              "label_en": "L'Assemblée législative de la province de Québec : 1867-1878"
            },
            {
              "description_en": "thèse de doctorat de Jean Provencher soumise à l'Université Laval",
              "id": "Q63648939",
              "label_en": "Joseph-Ernest Grégoire, quatre années de vie politique"
            },
            {
              "description_en": "boek van Florence Piron",
              "id": "Q63862260",
              "label_en": "Et si la recherche scientifique ne pouvait pas être neutre?"
            },
            {
              "description_en": "boek",
              "id": "Q63862333",
              "label_en": "Classiques des sciences sociales : 25 ans de partage des savoirs dans la francophonie"
            },
            {
              "description_en": "boek van Denis Jeffrey",
              "id": "Q63862605",
              "label_en": "Laïcité et signes religieux à l'école"
            },
            {
              "description_en": "boek",
              "id": "Q63862850",
              "label_en": "Les catholiques québécois et la laïcité"
            },
            {
              "description_en": "boek van Jocelyn Maclure",
              "id": "Q63862938",
              "label_en": "Penser la laïcité québécoise : fondements et défense d'une laïcité ouverte au Québec"
            },
            {
              "description_en": "boek van Daniel Baril",
              "id": "Q63863048",
              "label_en": "For a recognition of secularism in Quebec: philosophical, political and legal issues"
            },
            {
              "description_en": "boek van Patrick Taillon",
              "id": "Q63863087",
              "label_en": "Jean-Charles-Bonenfant et l'esprit des institutions"
            },
            {
              "description_en": "boek van Raymond Lemieux",
              "id": "Q63863120",
              "label_en": "Le catholicisme québécois"
            },
            {
              "description_en": "boek van Musée national des beaux-arts du Québec",
              "id": "Q63863193",
              "label_en": "Peinture et société au Québec"
            },
            {
              "description_en": "boek van Denis Jeffrey",
              "id": "Q63863366",
              "label_en": "Jeunes et djihadisme : les conversions interdites"
            },
            {
              "description_en": "boek van Jean-Frédéric Morin",
              "id": "Q63871923",
              "label_en": "Political science in motion"
            },
            {
              "description_en": "boek van Jean-Frédéric Morin",
              "id": "Q63871975",
              "label_en": "Essential concepts of global environmental governance"
            },
            {
              "description_en": "boek van André Couture",
              "id": "Q63872021",
              "label_en": "Ces anges qui nous reviennent"
            },
            "... omitted 4 items"
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(?i)((?!\\b(%)).)*"
            }
          ],
          "P2916": [
            {
              "value": "test, attempts to find URL encoded values (%20 etc).@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(?i)((?!\\b(&)).)*"
            }
          ],
          "P2303": [
            {
              "description_en": "wetenschappelijk artikel",
              "id": "Q56806485",
              "label_en": "A biomimetic study of the explosive discharge of the bombardier beetle"
            },
            {
              "description_en": "wetenschappelijk artikel",
              "id": "Q58850552",
              "label_en": "Computing urban mobile landscapes through monitoring population density based on cell-phone chatting"
            },
            {
              "description_en": "scientific article published on 22 August 2008",
              "id": "Q62139877",
              "label_en": "On the relationship between energy efficiency and complexity: insight on the causality chain"
            }
          ],
          "P2916": [
            {
              "value": "another test, to find HTML-encoded values (&LT; etc)@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[^–]*"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ],
          "P2916": [
            {
              "value": "do not use long dash@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "^(?!10\\.5555).*"
            }
          ],
          "P2916": [
            {
              "value": "DOIs starting with 10.5555 are intended for private use. Please find a better DOI or deprecate the current one.@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "first edition of the IEEE 754 standard for floating-point arithmetic",
              "id": "Q14954905",
              "label_en": "IEEE 754-1985: IEEE Standard for Binary Floating-Point Arithmetic"
            },
            {
              "description_en": "second edition of the IEEE 754 standard for floating-point arithmetic",
              "id": "Q951059",
              "label_en": "IEEE 754-2008 revision"
            }
          ],
          "P2306": [
            {
              "description_en": "this item is a subclass (subset) of that item; ALL instances of this item are instances of that item; different from P31 (instance of), e.g.: volcano is a subclass of mountain; Everest is an instance of mountain",
              "id": "P279",
              "label_en": "subclass of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "type of page in the Wikimedia system. Use with P31 'instance of' for template pages",
              "id": "Q11266439",
              "label_en": "Wikimedia template"
            },
            {
              "description_en": "use with 'instance of' (P31) for Wikimedia category",
              "id": "Q4167836",
              "label_en": "Wikimedia category"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "occupation of a person. See also \"field of work\" (Property:P101), \"position held\" (Property:P39). Not for groups of people. There, use \"field of work\" (Property:P101), \"industry\" (Property:P452), \"members have occupation\" (Property:P3989).",
              "id": "P106",
              "label_en": "occupation"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "International Standard Name Identifier for an identity. Starting with 0000.",
              "id": "P213",
              "label_en": "ISNI"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "organisation or other agent that issues or allocates an identifier, code, classification number, etc.",
              "id": "P2378",
              "label_en": "issued by"
            },
            {
              "description_en": "fee or toll payable to use, transit or enter the subject (only for one-time fees, do NOT use it for an ongoing fee, tuition fee or trading fee)",
              "id": "P2555",
              "label_en": "fee"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "number of an edition (first, second, ... as 1, 2, ...) or event",
              "id": "P393",
              "label_en": "edition number"
            },
            {
              "description_en": "property or qualifier for an ID property indicating whether linked content is directly readable online",
              "id": "P6954",
              "label_en": "online access status"
            },
            {
              "description_en": "qualifier to allow the reason to be indicated why a particular statement should be considered preferred",
              "id": "P7452",
              "label_en": "reason for preferred rank"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            },
            {
              "description_en": "Wikibase entity type for Wikimedia Commons",
              "id": "Q59712033",
              "label_en": "МэдыяІнфа Вікібазы"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828449",
              "label_en": "as qualifier"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
            {
              "description_en": "organisation or other agent that issues or allocates an identifier, code, classification number, etc.",
              "id": "P2378",
              "label_en": "issued by"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(10\\.[0-9]{4,}(?:\\.[0-9]+)*/(?:(?![\"&'])\\S)+)|"
            }
          ],
          "P2303": [
            {
              "description_en": "rigorózní práce",
              "id": "Q107814480",
              "label_en": "České město v procesu modernizace mezi lety 1850 až 1914 na příkladu Kolína"
            },
            {
              "description_en": "diplomová práce",
              "id": "Q108532806",
              "label_en": "František Norbert Hrachovský, O.Praem (1879-1943)"
            },
            {
              "description_en": "2015年書籍",
              "id": "Q112031173",
              "label_en": "鋼鐵風景：宋璽德"
            },
            {
              "description_en": "scientific article published on August 2006",
              "id": "Q39347033",
              "label_en": "Surgical simulation of instrumented posterior occipitocervical fusion in a child with congenital skeletal anomaly: case report"
            },
            {
              "description_en": "scientific article published on May 2006",
              "id": "Q42684434",
              "label_en": "Operative failure of percutaneous endoscopic lumbar discectomy: a radiologic analysis of 55 cases"
            },
            {
              "description_en": "wetenschappelijk artikel",
              "id": "Q56806485",
              "label_en": "A biomimetic study of the explosive discharge of the bombardier beetle"
            },
            {
              "description_en": "thesis published in 2014",
              "id": "Q58222124",
              "label_en": "Woman-to-woman sexual assault : a situational analysis"
            },
            {
              "description_en": "wetenschappelijk artikel",
              "id": "Q58850552",
              "label_en": "Computing urban mobile landscapes through monitoring population density based on cell-phone chatting"
            },
            {
              "description_en": "scientific article published on 22 August 2008",
              "id": "Q62139877",
              "label_en": "On the relationship between energy efficiency and complexity: insight on the causality chain"
            },
            {
              "description_en": "thèse de doctorat de Marcel Hamelin soumise à l'Université Laval",
              "id": "Q63646670",
              "label_en": "L'Assemblée législative de la province de Québec : 1867-1878"
            },
            {
              "description_en": "thèse de doctorat de Jean Provencher soumise à l'Université Laval",
              "id": "Q63648939",
              "label_en": "Joseph-Ernest Grégoire, quatre années de vie politique"
            },
            {
              "description_en": "boek van Florence Piron",
              "id": "Q63862260",
              "label_en": "Et si la recherche scientifique ne pouvait pas être neutre?"
            },
            {
              "description_en": "boek",
              "id": "Q63862333",
              "label_en": "Classiques des sciences sociales : 25 ans de partage des savoirs dans la francophonie"
            },
            {
              "description_en": "boek van Denis Jeffrey",
              "id": "Q63862605",
              "label_en": "Laïcité et signes religieux à l'école"
            },
            {
              "description_en": "boek",
              "id": "Q63862850",
              "label_en": "Les catholiques québécois et la laïcité"
            },
            {
              "description_en": "boek van Jocelyn Maclure",
              "id": "Q63862938",
              "label_en": "Penser la laïcité québécoise : fondements et défense d'une laïcité ouverte au Québec"
            },
            {
              "description_en": "boek van Daniel Baril",
              "id": "Q63863048",
              "label_en": "For a recognition of secularism in Quebec: philosophical, political and legal issues"
            },
            {
              "description_en": "boek van Patrick Taillon",
              "id": "Q63863087",
              "label_en": "Jean-Charles-Bonenfant et l'esprit des institutions"
            },
            {
              "description_en": "boek van Raymond Lemieux",
              "id": "Q63863120",
              "label_en": "Le catholicisme québécois"
            },
            {
              "description_en": "boek van Musée national des beaux-arts du Québec",
              "id": "Q63863193",
              "label_en": "Peinture et société au Québec"
            },
            {
              "description_en": "boek van Denis Jeffrey",
              "id": "Q63863366",
              "label_en": "Jeunes et djihadisme : les conversions interdites"
            },
            {
              "description_en": "boek van Jean-Frédéric Morin",
              "id": "Q63871923",
              "label_en": "Political science in motion"
            },
            {
              "description_en": "boek van Jean-Frédéric Morin",
              "id": "Q63871975",
              "label_en": "Essential concepts of global environmental governance"
            },
            {
              "description_en": "boek van André Couture",
              "id": "Q63872021",
              "label_en": "Ces anges qui nous reviennent"
            },
            "... omitted 4 items"
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(?i)((?!\\b(%)).)*"
            }
          ],
          "P2916": [
            {
              "value": "test, attempts to find URL encoded values (%20 etc).@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "(?i)((?!\\b(&)).)*"
            }
          ],
          "P2303": [
            {
              "description_en": "wetenschappelijk artikel",
              "id": "Q56806485",
              "label_en": "A biomimetic study of the explosive discharge of the bombardier beetle"
            },
            {
              "description_en": "wetenschappelijk artikel",
              "id": "Q58850552",
              "label_en": "Computing urban mobile landscapes through monitoring population density based on cell-phone chatting"
            },
            {
              "description_en": "scientific article published on 22 August 2008",
              "id": "Q62139877",
              "label_en": "On the relationship between energy efficiency and complexity: insight on the causality chain"
            }
          ],
          "P2916": [
            {
              "value": "another test, to find HTML-encoded values (&LT; etc)@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[^–]*"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ],
          "P2916": [
            {
              "value": "do not use long dash@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "^(?!10\\.1145).*"
            }
          ],
          "P2916": [
            {
              "value": "DOIs starting with 10.1145 are intended for private use. Please find a better DOI or deprecate the current one.@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "^(?!10\\.5555).*"
            }
          ],
          "P2916": [
            {
              "value": "DOIs starting with 10.5555 are intended for private use. Please find a better DOI or deprecate the current one.@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {},
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "first edition of the IEEE 754 standard for floating-point arithmetic",
              "id": "Q14954905",
              "label_en": "IEEE 754-1985: IEEE Standard for Binary Floating-Point Arithmetic"
            },
            {
              "description_en": "second edition of the IEEE 754 standard for floating-point arithmetic",
              "id": "Q951059",
              "label_en": "IEEE 754-2008 revision"
            }
          ],
          "P2306": [
            {
              "description_en": "this item is a subclass (subset) of that item; ALL instances of this item are instances of that item; different from P31 (instance of), e.g.: volcano is a subclass of mountain; Everest is an instance of mountain",
              "id": "P279",
              "label_en": "subclass of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "type of page in the Wikimedia system. Use with P31 'instance of' for template pages",
              "id": "Q11266439",
              "label_en": "Wikimedia template"
            },
            {
              "description_en": "use with 'instance of' (P31) for Wikimedia category",
              "id": "Q4167836",
              "label_en": "Wikimedia category"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "occupation of a person. See also \"field of work\" (Property:P101), \"position held\" (Property:P39). Not for groups of people. There, use \"field of work\" (Property:P101), \"industry\" (Property:P452), \"members have occupation\" (Property:P3989).",
              "id": "P106",
              "label_en": "occupation"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "International Standard Name Identifier for an identity. Starting with 0000.",
              "id": "P213",
              "label_en": "ISNI"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "organisation or other agent that issues or allocates an identifier, code, classification number, etc.",
              "id": "P2378",
              "label_en": "issued by"
            },
            {
              "description_en": "fee or toll payable to use, transit or enter the subject (only for one-time fees, do NOT use it for an ongoing fee, tuition fee or trading fee)",
              "id": "P2555",
              "label_en": "fee"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "number of an edition (first, second, ... as 1, 2, ...) or event",
              "id": "P393",
              "label_en": "edition number"
            },
            {
              "description_en": "property or qualifier for an ID property indicating whether linked content is directly readable online",
              "id": "P6954",
              "label_en": "online access status"
            },
            {
              "description_en": "qualifier to allow the reason to be indicated why a particular statement should be considered preferred",
              "id": "P7452",
              "label_en": "reason for preferred rank"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            },
            {
              "description_en": "Wikibase entity type for Wikimedia Commons",
              "id": "Q59712033",
              "label_en": "МэдыяІнфа Вікібазы"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828449",
              "label_en": "as qualifier"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "5cf55bb7e171e2d1e14836aae52873a6c022631b",
  "hash_before": "d031ea02a7b0db039216fefe8f4327579c311afd",
  "property_revision_id": 2447249541,
  "property_revision_prev": 2445410203,
  "qualifier_value_changes": [
    {
      "added_values": [
        "^(?!10\\.5555).*"
      ],
      "constraint_qid": "Q21502404",
      "qualifier_property": "P1793",
      "removed_values": [
        "^(?!10\\.1145).*"
      ],
      "same_qid_index": 4
    },
    {
      "added_values": [
        "DOIs starting with 10.5555 are intended for private use. Please find a better DOI or deprecate the current one.@en"
      ],
      "constraint_qid": "Q21502404",
      "qualifier_property": "P2916",
      "removed_values": [
        "DOIs starting with 10.1145 are intended for private use. Please find a better DOI or deprecate the current one.@en"
      ],
      "same_qid_index": 4
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502404",
      "qualifiers": [
        {
          "property_id": "P1793",
          "values": [
            "^(?!10\\.1145).*"
          ]
        },
        {
          "property_id": "P2916",
          "values": [
            "DOIs starting with 10.1145 are intended for private use. Please find a better DOI or deprecate the current one.@en"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: separator: has characteristic, issued by, object of statement has role",
      "format constraint: format as a regular expression: (10\\.[0-9]{4,}(?:\\.[0-9]+)*/(?:(?![\"&'])\\S)+)|; exception to constraint: České město v procesu modernizace mezi lety 1850 až 1914 na příkladu Kolína, František Norbert Hrachovský, O.Praem (1879-1943), 鋼鐵風景：宋璽德, Surgical simulation of instrumented posterior occipitocervical fusion in a child with congenital skeletal anomaly: case report, Operative failure of percutaneous endoscopic lumbar discectomy: a radiologic a... [truncated 1501 chars]",
      "format constraint: format as a regular expression: (?i)((?!\\b(%)).)*; syntax clarification: test, attempts to find URL encoded values (%20 etc).@en",
      "format constraint: format as a regular expression: (?i)((?!\\b(&)).)*; exception to constraint: A biomimetic study of the explosive discharge of the bombardier beetle, Computing urban mobile landscapes through monitoring population density based on cell-phone chatting, On the relationship between energy efficiency and complexity: insight on the causality chain; syntax clarification: another test, to find HTML-encoded values (&LT; etc)@en",
      "format constraint: format as a regular expression: [^–]*; constraint status: suggestion constraint; syntax clarification: do not use long dash@en",
      "format constraint: format as a regular expression: ^(?!10\\.5555).*; syntax clarification: DOIs starting with 10.5555 are intended for private use. Please find a better DOI or deprecate the current one.@en",
      "distinct-values constraint: no qualifiers recorded",
      "conflicts-with constraint: exception to constraint: IEEE 754-1985: IEEE Standard for Binary Floating-Point Arithmetic, IEEE 754-2008 revision; property: subclass of",
      "conflicts-with constraint: item of property constraint: Wikimedia template, Wikimedia category, human; property: instance of; constraint status: mandatory constraint",
      "conflicts-with constraint: property: occupation",
      "conflicts-with constraint: property: ISNI",
      "allowed qualifiers constraint: property: has characteristic, object named as, reason for deprecated rank, issued by, fee, object of statement has role, edition number, online access status, reason for preferred rank",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы",
      "property scope constraint: property scope: as main value, as qualifier, as reference"
    ],
    "before": [
      "single-value constraint: separator: has characteristic, issued by, object of statement has role",
      "format constraint: format as a regular expression: (10\\.[0-9]{4,}(?:\\.[0-9]+)*/(?:(?![\"&'])\\S)+)|; exception to constraint: České město v procesu modernizace mezi lety 1850 až 1914 na příkladu Kolína, František Norbert Hrachovský, O.Praem (1879-1943), 鋼鐵風景：宋璽德, Surgical simulation of instrumented posterior occipitocervical fusion in a child with congenital skeletal anomaly: case report, Operative failure of percutaneous endoscopic lumbar discectomy: a radiologic a... [truncated 1501 chars]",
      "format constraint: format as a regular expression: (?i)((?!\\b(%)).)*; syntax clarification: test, attempts to find URL encoded values (%20 etc).@en",
      "format constraint: format as a regular expression: (?i)((?!\\b(&)).)*; exception to constraint: A biomimetic study of the explosive discharge of the bombardier beetle, Computing urban mobile landscapes through monitoring population density based on cell-phone chatting, On the relationship between energy efficiency and complexity: insight on the causality chain; syntax clarification: another test, to find HTML-encoded values (&LT; etc)@en",
      "format constraint: format as a regular expression: [^–]*; constraint status: suggestion constraint; syntax clarification: do not use long dash@en",
      "format constraint: format as a regular expression: ^(?!10\\.1145).*; syntax clarification: DOIs starting with 10.1145 are intended for private use. Please find a better DOI or deprecate the current one.@en",
      "format constraint: format as a regular expression: ^(?!10\\.5555).*; syntax clarification: DOIs starting with 10.5555 are intended for private use. Please find a better DOI or deprecate the current one.@en",
      "distinct-values constraint: no qualifiers recorded",
      "conflicts-with constraint: exception to constraint: IEEE 754-1985: IEEE Standard for Binary Floating-Point Arithmetic, IEEE 754-2008 revision; property: subclass of",
      "conflicts-with constraint: item of property constraint: Wikimedia template, Wikimedia category, human; property: instance of; constraint status: mandatory constraint",
      "conflicts-with constraint: property: occupation",
      "conflicts-with constraint: property: ISNI",
      "allowed qualifiers constraint: property: has characteristic, object named as, reason for deprecated rank, issued by, fee, object of statement has role, edition number, online access status, reason for preferred rank",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы",
      "property scope constraint: property scope: as main value, as qualifier, as reference"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q21502404",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Format"
  }
]
```

---

## 019. `reform_Q65825337_P2517_2442912347`

| Field | Value |
|---|---|
| qid | Q65825337 |
| property | P2517 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | TBOX::P2517::2442912347 |
| tbox_revision_key | TBOX::P2517::2442912347 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Clemens Dulcis",
  "kind": "T_BOX",
  "property_revision_id": 2442912347,
  "property_revision_prev": 2442912167
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-16T09:07:43",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2517",
  "report_revision_new": 2442945713,
  "report_revision_old": 2442585925,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "Q9603280"
  ],
  "value_current_2026_descriptions_en": [
    "Wikimedia category"
  ],
  "value_current_2026_labels_en": [
    "Category:Recipients of the Order of the White Eagle (Third Polish Republic)"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "link to Wikimedia category for recipients of this award",
    "label": "category for recipients of this award"
  },
  "qid": {
    "description": "the Polish order of merit since 1992",
    "label": "Order of the White Eagle (Third Polish Republic)"
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
    "label_en": "distinct-values constraint",
    "qid": "Q21502410"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "none-of constraint",
    "qid": "Q52558054"
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q19474404",
      "qualifiers": [
        {
          "property_id": "P4155",
          "values": [
            "P1545",
            "P4224"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 11,
  "author": "Clemens Dulcis",
  "before_constraint_count": 11,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "position of an item in its parent series (most frequently a 1-based index), generally to be used as a qualifier (different from \"rank\" defined as a class, and from \"ranking\" defined as a property for evaluating a quality)",
              "id": "P1545",
              "label_en": "series ordinal"
            },
            {
              "description_en": "category contains elements that are instances of this item",
              "id": "P4224",
              "label_en": "category contains"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "qualifier to specify the item that this property is shared with",
              "id": "P1706",
              "label_en": "together with"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2304": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ],
          "P2305": [
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "event, during which one or more sporting events are held",
              "id": "Q13406554",
              "label_en": "sports competition"
            },
            {
              "description_en": null,
              "id": "Q134882735",
              "label_en": "ice climbing competition"
            },
            {
              "description_en": "competitive exhibition based on aesthetically-pleasing physical attributes",
              "id": "Q2658935",
              "label_en": "beauty contest"
            },
            {
              "description_en": "competition of esports",
              "id": "Q48004378",
              "label_en": "esport competition"
            },
            {
              "description_en": "beauty pageant with female contestants",
              "id": "Q58863414",
              "label_en": "female beauty pageant"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ],
          "P6607": [
            {
              "value": "use \"related category\" (P7084)@mul"
            }
          ],
          "P6824": [
            {
              "description_en": "Wikimedia category is related to this item",
              "id": "P7084",
              "label_en": "related category"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "type of award classified by stylistic, thematic or technical criteria",
              "id": "Q107467117",
              "label_en": "type of award"
            },
            {
              "description_en": "award group or series",
              "id": "Q107655869",
              "label_en": "group of awards"
            },
            {
              "description_en": "qualified name, rank, or other indication of a class or role given to or inherited by a person, often affixed to a person's name",
              "id": "Q216353",
              "label_en": "title"
            },
            {
              "description_en": "class of award (order, medal, etc.)",
              "id": "Q38033430",
              "label_en": "class of award"
            },
            {
              "description_en": "something given to a person or a group of people to recognize their merit or excellence",
              "id": "Q618779",
              "label_en": "award"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type constraints",
              "id": "Q21503252",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item has to refer back to this item with the given inverse property",
          "id": "Q21510855",
          "label_en": "inverse constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "this category combines (intersects) these two or more topics",
              "id": "P971",
              "label_en": "category combines topics"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item should have a statement with a given property",
          "id": "Q21510864",
          "label_en": "value-requires-statement constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "person who has received an award",
              "id": "Q21096945",
              "label_en": "award winner"
            }
          ],
          "P2306": [
            {
              "description_en": "this category combines (intersects) these two or more topics",
              "id": "P971",
              "label_en": "category combines topics"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "use with 'instance of' (P31) for Wikimedia category",
              "id": "Q4167836",
              "label_en": "Wikimedia category"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type constraints",
              "id": "Q21503252",
              "label_en": "instance of"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "Wikimedia category",
              "id": "Q7866116",
              "label_en": "Катэгорыя:Народныя артысты Расіі"
            }
          ],
          "P6607": [
            {
              "value": "value must not be a metacategory@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "category contains elements that are instances of this item",
              "id": "P4224",
              "label_en": "category contains"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "qualifier to specify the item that this property is shared with",
              "id": "P1706",
              "label_en": "together with"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2304": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ],
          "P2305": [
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "event, during which one or more sporting events are held",
              "id": "Q13406554",
              "label_en": "sports competition"
            },
            {
              "description_en": null,
              "id": "Q134882735",
              "label_en": "ice climbing competition"
            },
            {
              "description_en": "competitive exhibition based on aesthetically-pleasing physical attributes",
              "id": "Q2658935",
              "label_en": "beauty contest"
            },
            {
              "description_en": "competition of esports",
              "id": "Q48004378",
              "label_en": "esport competition"
            },
            {
              "description_en": "beauty pageant with female contestants",
              "id": "Q58863414",
              "label_en": "female beauty pageant"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ],
          "P6607": [
            {
              "value": "use \"related category\" (P7084)@mul"
            }
          ],
          "P6824": [
            {
              "description_en": "Wikimedia category is related to this item",
              "id": "P7084",
              "label_en": "related category"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "type of award classified by stylistic, thematic or technical criteria",
              "id": "Q107467117",
              "label_en": "type of award"
            },
            {
              "description_en": "award group or series",
              "id": "Q107655869",
              "label_en": "group of awards"
            },
            {
              "description_en": "qualified name, rank, or other indication of a class or role given to or inherited by a person, often affixed to a person's name",
              "id": "Q216353",
              "label_en": "title"
            },
            {
              "description_en": "class of award (order, medal, etc.)",
              "id": "Q38033430",
              "label_en": "class of award"
            },
            {
              "description_en": "something given to a person or a group of people to recognize their merit or excellence",
              "id": "Q618779",
              "label_en": "award"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type constraints",
              "id": "Q21503252",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item has to refer back to this item with the given inverse property",
          "id": "Q21510855",
          "label_en": "inverse constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "this category combines (intersects) these two or more topics",
              "id": "P971",
              "label_en": "category combines topics"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item should have a statement with a given property",
          "id": "Q21510864",
          "label_en": "value-requires-statement constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "person who has received an award",
              "id": "Q21096945",
              "label_en": "award winner"
            }
          ],
          "P2306": [
            {
              "description_en": "this category combines (intersects) these two or more topics",
              "id": "P971",
              "label_en": "category combines topics"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "use with 'instance of' (P31) for Wikimedia category",
              "id": "Q4167836",
              "label_en": "Wikimedia category"
            }
          ],
          "P2309": [
            {
              "description_en": "relation of type constraints",
              "id": "Q21503252",
              "label_en": "instance of"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "Wikimedia category",
              "id": "Q7866116",
              "label_en": "Катэгорыя:Народныя артысты Расіі"
            }
          ],
          "P6607": [
            {
              "value": "value must not be a metacategory@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "c2b4fff6be93b6bc072701efb9765dbbeab50b30",
  "hash_before": "e6dcffe0b7d5f0baeda89acddd85faacc02c561c",
  "property_revision_id": 2442912347,
  "property_revision_prev": 2442912167,
  "qualifier_value_changes": [
    {
      "added_values": [
        "P1545"
      ],
      "constraint_qid": "Q19474404",
      "qualifier_property": "P4155",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q19474404",
      "qualifiers": [
        {
          "property_id": "P4155",
          "values": [
            "P4224"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: separator: series ordinal, category contains",
      "distinct-values constraint: separator: together with",
      "conflicts-with constraint: group by: instance of; item of property constraint: Wikimedia disambiguation page, human; property: instance of; constraint status: mandatory constraint",
      "conflicts-with constraint: item of property constraint: sports competition, ice climbing competition, beauty contest, esport competition, female beauty pageant; property: instance of; constraint clarification: use \"related category\" (P7084)@mul; replacement property: related category",
      "subject type constraint: class: type of award, group of awards, title, class of award, award; relation: instance of",
      "inverse constraint: property: category combines topics",
      "value-requires-statement constraint: item of property constraint: award winner; property: category combines topics",
      "value-type constraint: class: Wikimedia category; relation: instance of; constraint status: mandatory constraint",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "none-of constraint: item of property constraint: Катэгорыя:Народныя артысты Расіі; constraint clarification: value must not be a metacategory@en",
      "property scope constraint: property scope: as main value"
    ],
    "before": [
      "single-value constraint: separator: category contains",
      "distinct-values constraint: separator: together with",
      "conflicts-with constraint: group by: instance of; item of property constraint: Wikimedia disambiguation page, human; property: instance of; constraint status: mandatory constraint",
      "conflicts-with constraint: item of property constraint: sports competition, ice climbing competition, beauty contest, esport competition, female beauty pageant; property: instance of; constraint clarification: use \"related category\" (P7084)@mul; replacement property: related category",
      "subject type constraint: class: type of award, group of awards, title, class of award, award; relation: instance of",
      "inverse constraint: property: category combines topics",
      "value-requires-statement constraint: item of property constraint: award winner; property: category combines topics",
      "value-type constraint: class: Wikimedia category; relation: instance of; constraint status: mandatory constraint",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "none-of constraint: item of property constraint: Катэгорыя:Народныя артысты Расіі; constraint clarification: value must not be a metacategory@en",
      "property scope constraint: property scope: as main value"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q21502410",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Unique value"
  }
]
```

---

## 020. `reform_Q65960782_P269_2445523281`

| Field | Value |
|---|---|
| qid | Q65960782 |
| property | P269 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / COINCIDENTAL_SCHEMA_CHANGE / low |
| main_score / diagnostic_only | False / True |
| analysis_slice | diagnostic_tbox_coincidental |
| popularity_bucket | mid |
| constraint_family | Q52004125 |
| group_key | TBOX::P269::2445523281 |
| tbox_revision_key | TBOX::P269::2445523281 |

### Annotation Focus

- Check whether the constraint diff plausibly caused the violation disappearance.
- Use qualifier_value_changes to see what actually changed without reading raw full signatures.
- COINCIDENTAL_SCHEMA_CHANGE should usually remain diagnostic unless the causal link is strong.

### Classifier Summary

| Field | Value |
|---|---|
| truth_source | none_expected |
| truth_token_kind | none_expected |
| truth_tokens_preview | [] |
| decision_branch |  |
| rationale | Violation type did not map to the changed constraint types; treated as coincidental schema change. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Thomas Kerboul (BGE)",
  "kind": "T_BOX",
  "property_revision_id": 2445523281,
  "property_revision_prev": 2445451375
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T13:10:18",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P269",
  "report_revision_new": 2447784711,
  "report_revision_old": 2447392530,
  "report_violation_type": "Unique value",
  "report_violation_type_normalized": "Unique value",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Unique value",
  "value": null,
  "value_current_2026": [
    "18978203X"
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
  "truth_tokens": [],
  "truth_tokens_in_recorded_matches": [],
  "used_literal_substring": null
}
```

### Labels / Human-Readable Context

```json
{
  "property": {
    "description": "identifier for authority control in the French collaborative library catalog (see also P1025). Format: 8 digits followed by a digit or \"X\"",
    "label": "IdRef ID"
  },
  "qid": {
    "description": "Dutch scientific instrument maker (1658–1741)",
    "label": "Simon van de Moolen"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502410",
      "qualifiers": [
        {
          "property_id": "P4155",
          "values": [
            "P4070"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 6,
  "author": "Thomas Kerboul (BGE)",
  "before_constraint_count": 6,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "qualifier for alternative name(s), given for a subject in a database entry, or preserved in references (even these are no longer the preferred name)",
              "id": "P4970",
              "label_en": "alternative name"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "([0-9]{8}[\\dX]|)"
            }
          ],
          "P2916": [
            {
              "value": "numeric string, 8 digits, suffixed by X or another digit@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "qualifier, to be used on external identifier IDs, indicating another Wikidata item is also matched to this ID",
              "id": "P4070",
              "label_en": "identifier shared with"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
              "id": "Q3331189",
              "label_en": "version, edition or translation"
            },
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P4155": [
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "qualifier for alternative name(s), given for a subject in a database entry, or preserved in references (even these are no longer the preferred name)",
              "id": "P4970",
              "label_en": "alternative name"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "([0-9]{8}[\\dX]|)"
            }
          ],
          "P2916": [
            {
              "value": "numeric string, 8 digits, suffixed by X or another digit@en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "canton of Switzerland",
              "id": "Q12738",
              "label_en": "Canton of Neuchâtel"
            },
            {
              "description_en": "country in Eastern Europe and Northern Asia",
              "id": "Q159",
              "label_en": "Russia"
            },
            {
              "description_en": "constituent republic of the Soviet Union (1922–1991)",
              "id": "Q2184",
              "label_en": "Russian Soviet Federative Socialist Republic"
            },
            {
              "description_en": "scientific institution of the Soviet Union (1925–1991)",
              "id": "Q2370801",
              "label_en": "Academy of Sciences of the USSR"
            },
            {
              "description_en": "state in western Europe (1034–1848)",
              "id": "Q3137802",
              "label_en": "Principality of Neuchâtel"
            },
            {
              "description_en": "former empire in Eurasia and North America (1721–1917)",
              "id": "Q34266",
              "label_en": "Russian Empire"
            },
            {
              "description_en": "historical academy (1724–1917)",
              "id": "Q4345832",
              "label_en": "Saint Petersburg Academy of Sciences"
            }
          ],
          "P4155": [
            {
              "description_en": "qualifier, to be used on external identifier IDs, indicating another Wikidata item is also matched to this ID",
              "id": "P4070",
              "label_en": "identifier shared with"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
              "id": "Q3331189",
              "label_en": "version, edition or translation"
            },
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            }
          ],
          "P2306": [
            {
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "id": "Q52004125",
          "label_en": "allowed-entity-types constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "entity type for Wikibase items",
              "id": "Q29934200",
              "label_en": "Wikibase item"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "id": "Q53869507",
          "label_en": "property scope constraint"
        },
        "parameters": {
          "P5314": [
            {
              "description_en": "property scope type",
              "id": "Q54828448",
              "label_en": "as main value"
            },
            {
              "description_en": "property scope type",
              "id": "Q54828450",
              "label_en": "as reference"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "885bab3ddcf610be0c045df2830bd44b8d0037de",
  "hash_before": "7001e5b91ee685cbf6467bb34973749c2e5258df",
  "property_revision_id": 2445523281,
  "property_revision_prev": 2445451375,
  "qualifier_value_changes": [
    {
      "added_values": [],
      "constraint_qid": "Q21502410",
      "qualifier_property": "P2303",
      "removed_values": [
        "Q12738",
        "Q159",
        "Q2184",
        "Q2370801",
        "Q3137802",
        "Q34266",
        "Q4345832"
      ],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502410",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q12738",
            "Q159",
            "Q2184",
            "Q2370801",
            "Q3137802",
            "Q34266",
            "Q4345832"
          ]
        },
        {
          "property_id": "P4155",
          "values": [
            "P4070"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: separator: subject named as, object of statement has role, alternative name",
      "format constraint: format as a regular expression: ([0-9]{8}[\\dX]|); syntax clarification: numeric string, 8 digits, suffixed by X or another digit@en",
      "distinct-values constraint: separator: identifier shared with",
      "conflicts-with constraint: item of property constraint: version, edition or translation, Wikimedia disambiguation page; property: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ],
    "before": [
      "single-value constraint: separator: subject named as, object of statement has role, alternative name",
      "format constraint: format as a regular expression: ([0-9]{8}[\\dX]|); syntax clarification: numeric string, 8 digits, suffixed by X or another digit@en",
      "distinct-values constraint: exception to constraint: Canton of Neuchâtel, Russia, Russian Soviet Federative Socialist Republic, Academy of Sciences of the USSR, Principality of Neuchâtel, Russian Empire, Saint Petersburg Academy of Sciences; separator: identifier shared with",
      "conflicts-with constraint: item of property constraint: version, edition or translation, Wikimedia disambiguation page; property: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": "Q21502410",
    "result": false,
    "step": "causality_filter",
    "violation_name": "Unique value"
  }
]
```

---
