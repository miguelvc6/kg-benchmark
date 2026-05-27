# TBOX_DIRECTIONAL_RELAXATION_OR_RESTRICTION

Cases: 20

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `reform_Q106368599_P4969_2435927232`

| Field | Value |
|---|---|
| qid | Q106368599 |
| property | P4969 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | TBOX::P4969::2435927232 |
| tbox_revision_key | TBOX::P4969::2435927232 |

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
| rationale | Constraint qualifiers compared with generic set semantics. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Trade",
  "kind": "T_BOX",
  "property_revision_id": 2435927232,
  "property_revision_prev": 2435926155
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-06T06:28:43",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4969",
  "report_revision_new": 2438638194,
  "report_revision_old": 2438318580,
  "report_violation_type": "Type Q|386724, Q|4886, Q|43099500, Q|95074, Q|18706315, Q|16686448, Q|116779428, Q|11424, Q|48708989, Q|15142894, Q|47451145",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "plant or grouping of plants selected for desirable characteristics",
    "production of the performing arts, consisting of a series of quasi-identical performances of the same performance work",
    "fictional human or non-human character in a narrative work of art",
    "entity whose existence is possible, but not proven",
    "anything created by humans (either material or mental)",
    "סוג קבוצת יצירות",
    "sequence of images that give the impression of movement, stored on film stock",
    "group of related ammunition cartridges which share basic design elements",
    "specific weapon design, pattern, or version of which all examples are essentially identical",
    "recurring, self-sufficient plot or motif grouping, unit of classification in the Aarne–Thompson classification systems"
  ],
  "report_violation_type_labels_en": [
    "work",
    "cultivar",
    "performing arts production",
    "character",
    "hypothetical entity",
    "artificial object",
    "group of works often treated as a singular work",
    "film",
    "cartridge family",
    "weapon model",
    "tale type"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|4886, Q|43099500, Q|95074, Q|18706315, Q|16686448, Q|116779428, Q|11424, Q|48708989, Q|15142894, Q|47451145",
  "report_violation_type_qids": [
    "Q386724",
    "Q4886",
    "Q43099500",
    "Q95074",
    "Q18706315",
    "Q16686448",
    "Q116779428",
    "Q11424",
    "Q48708989",
    "Q15142894",
    "Q47451145"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|4886, Q|43099500, Q|95074, Q|18706315, Q|16686448, Q|116779428, Q|11424, Q|48708989, Q|15142894, Q|47451145",
  "value": null,
  "value_current_2026": [
    "Q106368218"
  ],
  "value_current_2026_descriptions_en": [
    "18th‐century chapbook"
  ],
  "value_current_2026_labels_en": [
    "The Pleaſant Hiſtory of Jack Horner"
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
    "description": "new work of art (film, book, software, etc.) derived from major part of this work",
    "label": "derivative work"
  },
  "qid": {
    "description": "15th‐century poem and fantasy tale",
    "label": "Jack and His Stepdame"
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
    "label_en": "inverse constraint",
    "qid": "Q21510855"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502838",
      "qualifiers": [
        {
          "property_id": "P2305",
          "values": [
            "Q136747113",
            "Q136832029",
            "Q3331189",
            "Q57933693"
          ]
        },
        {
          "property_id": "P2306",
          "values": [
            "P31"
          ]
        },
        {
          "property_id": "P2316",
          "values": [
            "Q21502408"
          ]
        },
        {
          "property_id": "P6607",
          "values": [
            "works should have this property instead of editions, use in Q7725634@en"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 6,
  "author": "Trade",
  "before_constraint_count": 6,
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
              "description_en": "Japanese style of animation",
              "id": "Q1107",
              "label_en": "anime"
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
          ],
          "P9729": [
            {
              "description_en": "series of light novels published in Japan",
              "id": "Q104213567",
              "label_en": "light novel series"
            },
            {
              "description_en": "series of comics employing Japanese stylistic conventions that are that are formally identified together",
              "id": "Q21198342",
              "label_en": "manga series"
            },
            {
              "description_en": "Japanese animated television series",
              "id": "Q63952888",
              "label_en": "anime television series"
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
              "description_en": "edition of a light novel",
              "id": "Q136747113",
              "label_en": "light novel edition"
            },
            {
              "description_en": "edition of a manga",
              "id": "Q136832029",
              "label_en": "manga edition"
            },
            {
              "description_en": "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
              "id": "Q3331189",
              "label_en": "version, edition or translation"
            },
            {
              "description_en": "edition of a book",
              "id": "Q57933693",
              "label_en": "book edition"
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
          ],
          "P6607": [
            {
              "value": "works should have this property instead of editions, use in Q7725634@en"
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
              "description_en": "sequence of images that give the impression of movement, stored on film stock",
              "id": "Q11424",
              "label_en": "film"
            },
            {
              "description_en": "סוג קבוצת יצירות",
              "id": "Q116779428",
              "label_en": "group of works often treated as a singular work"
            },
            {
              "description_en": "specific weapon design, pattern, or version of which all examples are essentially identical",
              "id": "Q15142894",
              "label_en": "weapon model"
            },
            {
              "description_en": "anything created by humans (either material or mental)",
              "id": "Q16686448",
              "label_en": "artificial object"
            },
            {
              "description_en": "entity whose existence is possible, but not proven",
              "id": "Q18706315",
              "label_en": "hypothetical entity"
            },
            {
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
            },
            {
              "description_en": "production of the performing arts, consisting of a series of quasi-identical performances of the same performance work",
              "id": "Q43099500",
              "label_en": "performing arts production"
            },
            {
              "description_en": "recurring, self-sufficient plot or motif grouping, unit of classification in the Aarne–Thompson classification systems",
              "id": "Q47451145",
              "label_en": "tale type"
            },
            {
              "description_en": "group of related ammunition cartridges which share basic design elements",
              "id": "Q48708989",
              "label_en": "cartridge family"
            },
            {
              "description_en": "plant or grouping of plants selected for desirable characteristics",
              "id": "Q4886",
              "label_en": "cultivar"
            },
            {
              "description_en": "fictional human or non-human character in a narrative work of art",
              "id": "Q95074",
              "label_en": "character"
            }
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item has to refer back to this item with the given inverse property",
          "id": "Q21510855",
          "label_en": "inverse constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the work(s) or inputs used as the basis for subject item; for fictional analog use P1074",
              "id": "P144",
              "label_en": "based on"
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
              "id": "Q54828449",
              "label_en": "as qualifier"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "Japanese style of animation",
              "id": "Q1107",
              "label_en": "anime"
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
          ],
          "P9729": [
            {
              "description_en": "series of light novels published in Japan",
              "id": "Q104213567",
              "label_en": "light novel series"
            },
            {
              "description_en": "series of comics employing Japanese stylistic conventions that are that are formally identified together",
              "id": "Q21198342",
              "label_en": "manga series"
            },
            {
              "description_en": "Japanese animated television series",
              "id": "Q63952888",
              "label_en": "anime television series"
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
              "description_en": "edition of a manga",
              "id": "Q136832029",
              "label_en": "manga edition"
            },
            {
              "description_en": "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
              "id": "Q3331189",
              "label_en": "version, edition or translation"
            },
            {
              "description_en": "edition of a book",
              "id": "Q57933693",
              "label_en": "book edition"
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
          ],
          "P6607": [
            {
              "value": "works should have this property instead of editions, use in Q7725634@en"
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
              "description_en": "sequence of images that give the impression of movement, stored on film stock",
              "id": "Q11424",
              "label_en": "film"
            },
            {
              "description_en": "סוג קבוצת יצירות",
              "id": "Q116779428",
              "label_en": "group of works often treated as a singular work"
            },
            {
              "description_en": "specific weapon design, pattern, or version of which all examples are essentially identical",
              "id": "Q15142894",
              "label_en": "weapon model"
            },
            {
              "description_en": "anything created by humans (either material or mental)",
              "id": "Q16686448",
              "label_en": "artificial object"
            },
            {
              "description_en": "entity whose existence is possible, but not proven",
              "id": "Q18706315",
              "label_en": "hypothetical entity"
            },
            {
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
            },
            {
              "description_en": "production of the performing arts, consisting of a series of quasi-identical performances of the same performance work",
              "id": "Q43099500",
              "label_en": "performing arts production"
            },
            {
              "description_en": "recurring, self-sufficient plot or motif grouping, unit of classification in the Aarne–Thompson classification systems",
              "id": "Q47451145",
              "label_en": "tale type"
            },
            {
              "description_en": "group of related ammunition cartridges which share basic design elements",
              "id": "Q48708989",
              "label_en": "cartridge family"
            },
            {
              "description_en": "plant or grouping of plants selected for desirable characteristics",
              "id": "Q4886",
              "label_en": "cultivar"
            },
            {
              "description_en": "fictional human or non-human character in a narrative work of art",
              "id": "Q95074",
              "label_en": "character"
            }
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item has to refer back to this item with the given inverse property",
          "id": "Q21510855",
          "label_en": "inverse constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the work(s) or inputs used as the basis for subject item; for fictional analog use P1074",
              "id": "P144",
              "label_en": "based on"
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
              "id": "Q54828449",
              "label_en": "as qualifier"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "299ae3d11d3719afbe8480654eb40a568a11f3df",
  "hash_before": "068118b0b02e8854ee54a1f3bb75df62a2e858cc",
  "property_revision_id": 2435927232,
  "property_revision_prev": 2435926155,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q136747113"
      ],
      "constraint_qid": "Q21502838",
      "qualifier_property": "P2305",
      "removed_values": [],
      "same_qid_index": 1
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502838",
      "qualifiers": [
        {
          "property_id": "P2305",
          "values": [
            "Q136832029",
            "Q3331189",
            "Q57933693"
          ]
        },
        {
          "property_id": "P2306",
          "values": [
            "P31"
          ]
        },
        {
          "property_id": "P2316",
          "values": [
            "Q21502408"
          ]
        },
        {
          "property_id": "P6607",
          "values": [
            "works should have this property instead of editions, use in Q7725634@en"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "conflicts-with constraint: item of property constraint: anime, light novel, manga; property: instance of; replacement value: light novel series, manga series, anime television series",
      "conflicts-with constraint: item of property constraint: light novel edition, manga edition, version, edition or translation, book edition; property: instance of; constraint status: mandatory constraint; constraint clarification: works should have this property instead of editions, use in Q7725634@en",
      "subject type constraint: class: film, group of works often treated as a singular work, weapon model, artificial object, hypothetical entity, work, performing arts production, tale type, cartridge family, cultivar, character; relation: instance or subclass of",
      "inverse constraint: property: based on",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as qualifier"
    ],
    "before": [
      "conflicts-with constraint: item of property constraint: anime, light novel, manga; property: instance of; replacement value: light novel series, manga series, anime television series",
      "conflicts-with constraint: item of property constraint: manga edition, version, edition or translation, book edition; property: instance of; constraint status: mandatory constraint; constraint clarification: works should have this property instead of editions, use in Q7725634@en",
      "subject type constraint: class: film, group of works often treated as a singular work, weapon model, artificial object, hypothetical entity, work, performing arts production, tale type, cartridge family, cultivar, character; relation: instance or subclass of",
      "inverse constraint: property: based on",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as qualifier"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|386724, Q|4886, Q|43099500, Q|95074, Q|18706315, Q|16686448, Q|116779428, Q|11424, Q|48708989, Q|15142894, Q|47451145"
  },
  {
    "result": "Q21502838",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---

## 002. `reform_Q1131356_P2517_2442912347`

| Field | Value |
|---|---|
| qid | Q1131356 |
| property | P2517 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | head |
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
| rationale | Constraint qualifiers compared with generic set semantics. |
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
  "report_violation_type": "Inverse",
  "report_violation_type_normalized": "Inverse",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Inverse",
  "value": null,
  "value_current_2026": [
    "Q8850671"
  ],
  "value_current_2026_descriptions_en": [
    "Wikimedia category"
  ],
  "value_current_2026_labels_en": [
    "Category:Theatre World Award winners"
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
    "description": "American theatre award (1945–)",
    "label": "Theatre World Award"
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Inverse"
  },
  {
    "result": "Q19474404",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---

## 003. `reform_Q117538666_P2092_2333296438`

| Field | Value |
|---|---|
| qid | Q117538666 |
| property | P2092 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | TBOX::P2092::2333296438 |
| tbox_revision_key | TBOX::P2092::2333296438 |

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
| rationale | Constraint qualifiers compared with generic set semantics. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Bob08",
  "kind": "T_BOX",
  "property_revision_id": 2333296438,
  "property_revision_prev": 2318183653
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-04-02T03:23:57",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2092",
  "report_revision_new": 2333569678,
  "report_revision_old": 2333235475,
  "report_violation_type": "Item P|136",
  "report_violation_type_normalized": "Item P|136",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|136",
  "report_violation_types": [
    "Item P|136",
    "Item P|6216",
    "Item P|180"
  ],
  "value": null,
  "value_current_2026": [
    "00240116"
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
    "description": "identifier for an artwork in Bildindex",
    "label": "Bildindex der Kunst und Architektur ID"
  },
  "qid": {
    "description": "sculpture by anonymous at the Germanisches Nationalmuseum, Nuremberg, Germany",
    "label": "Death as a Gravedigger"
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
  },
  {
    "label_en": "label in language constraint",
    "qid": "Q108139345"
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
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P424",
          "values": [
            "de",
            "fr"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 18,
  "author": "Bob08",
  "before_constraint_count": 18,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "de"
            },
            {
              "value": "fr"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
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
        "parameters": {
          "P2303": [
            {
              "description_en": "painting by David Teniers the Younger (Schleissheim Palace)",
              "id": "Q19960947",
              "label_en": "Archduke Leopold Wilhelm in his Gallery in Brussels (IV)"
            },
            {
              "description_en": "painting by David Teniers the Younger (Schleissheim Palace)",
              "id": "Q19960948",
              "label_en": "The Gallery of Archduke Leopold in Brussels (II)"
            },
            {
              "description_en": "painting by follower of Dieric Bouts",
              "id": "Q27518439",
              "label_en": "The Arrest of Christ with kiss of Judas and ear of Malchus"
            },
            {
              "description_en": "painting of a follower of Dieric Bouts",
              "id": "Q27519765",
              "label_en": "The Resurrection"
            },
            {
              "description_en": "gallery painting by Teniers",
              "id": "Q27919340",
              "label_en": "The Gallery of Archduke Leopold in Brussels (III)"
            },
            {
              "description_en": "painting by David Teniers the Younger in Schloss Schleißheim",
              "id": "Q27919851",
              "label_en": "The Gallery of Archduke Leopold in Brussels (I)"
            },
            {
              "description_en": "dismembered altarpiece by Luca Signorelli and Francesco di Giorgio Martini",
              "id": "Q54692597",
              "label_en": "Pala Bichi"
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
              "description_en": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
              "id": "P136",
              "label_en": "genre"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "maker of this creative work or other object (where no more specific property exists)",
              "id": "P170",
              "label_en": "creator"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "entity visually depicted in an image, literarily described in a work, or otherwise incorporated into an audiovisual or other medium; see also P921, 'main subject'",
              "id": "P180",
              "label_en": "depicts"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "material the subject or the object is made of or derived from (do not confuse with P10672 which is used for processes)",
              "id": "P186",
              "label_en": "made from material"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "art, museum, archival, or bibliographic collection of which the subject is part (item is in the collection of X)",
              "id": "P195",
              "label_en": "collection"
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
              "description_en": "vertical length of an entity",
              "id": "P2048",
              "label_en": "height"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "width of an object",
              "id": "P2049",
              "label_en": "width"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a physical object or a set of physical objects in a collection",
              "id": "P217",
              "label_en": "inventory number"
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
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "time when an entity begins to exist; for date of official opening use P1619",
              "id": "P571",
              "label_en": "inception"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "copyright status for intellectual creations like works of art, publications, software, etc.",
              "id": "P6216",
              "label_en": "copyright status"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "thematic group of an artist's works",
              "id": "Q15709879",
              "label_en": "artwork series"
            },
            {
              "description_en": "structure, typically with a roof and walls, standing more or less permanently in one place",
              "id": "Q41176",
              "label_en": "building"
            },
            {
              "description_en": "physical object made or shaped by humans",
              "id": "Q8205328",
              "label_en": "artificial physical object"
            },
            {
              "description_en": "aesthetic item or artistic creation",
              "id": "Q838948",
              "label_en": "work of art"
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
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "de"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
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
        "parameters": {
          "P2303": [
            {
              "description_en": "painting by David Teniers the Younger (Schleissheim Palace)",
              "id": "Q19960947",
              "label_en": "Archduke Leopold Wilhelm in his Gallery in Brussels (IV)"
            },
            {
              "description_en": "painting by David Teniers the Younger (Schleissheim Palace)",
              "id": "Q19960948",
              "label_en": "The Gallery of Archduke Leopold in Brussels (II)"
            },
            {
              "description_en": "painting by follower of Dieric Bouts",
              "id": "Q27518439",
              "label_en": "The Arrest of Christ with kiss of Judas and ear of Malchus"
            },
            {
              "description_en": "painting of a follower of Dieric Bouts",
              "id": "Q27519765",
              "label_en": "The Resurrection"
            },
            {
              "description_en": "gallery painting by Teniers",
              "id": "Q27919340",
              "label_en": "The Gallery of Archduke Leopold in Brussels (III)"
            },
            {
              "description_en": "painting by David Teniers the Younger in Schloss Schleißheim",
              "id": "Q27919851",
              "label_en": "The Gallery of Archduke Leopold in Brussels (I)"
            },
            {
              "description_en": "dismembered altarpiece by Luca Signorelli and Francesco di Giorgio Martini",
              "id": "Q54692597",
              "label_en": "Pala Bichi"
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
              "description_en": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
              "id": "P136",
              "label_en": "genre"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "maker of this creative work or other object (where no more specific property exists)",
              "id": "P170",
              "label_en": "creator"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "entity visually depicted in an image, literarily described in a work, or otherwise incorporated into an audiovisual or other medium; see also P921, 'main subject'",
              "id": "P180",
              "label_en": "depicts"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "material the subject or the object is made of or derived from (do not confuse with P10672 which is used for processes)",
              "id": "P186",
              "label_en": "made from material"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "art, museum, archival, or bibliographic collection of which the subject is part (item is in the collection of X)",
              "id": "P195",
              "label_en": "collection"
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
              "description_en": "vertical length of an entity",
              "id": "P2048",
              "label_en": "height"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "width of an object",
              "id": "P2049",
              "label_en": "width"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a physical object or a set of physical objects in a collection",
              "id": "P217",
              "label_en": "inventory number"
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
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "time when an entity begins to exist; for date of official opening use P1619",
              "id": "P571",
              "label_en": "inception"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "copyright status for intellectual creations like works of art, publications, software, etc.",
              "id": "P6216",
              "label_en": "copyright status"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "thematic group of an artist's works",
              "id": "Q15709879",
              "label_en": "artwork series"
            },
            {
              "description_en": "structure, typically with a roof and walls, standing more or less permanently in one place",
              "id": "Q41176",
              "label_en": "building"
            },
            {
              "description_en": "physical object made or shaped by humans",
              "id": "Q8205328",
              "label_en": "artificial physical object"
            },
            {
              "description_en": "aesthetic item or artistic creation",
              "id": "Q838948",
              "label_en": "work of art"
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
  "hash_after": "7000ebf81f03d2f0a6320e389cfb48adf76c1575",
  "hash_before": "b57bdb6c6241468d8e5dc6d0d7cd3d4e3cdc0bd1",
  "property_revision_id": 2333296438,
  "property_revision_prev": 2318183653,
  "qualifier_value_changes": [
    {
      "added_values": [
        "fr"
      ],
      "constraint_qid": "Q108139345",
      "qualifier_property": "P424",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P424",
          "values": [
            "de"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "label in language constraint: Wikimedia language code: de, fr",
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: \\d+; constraint status: mandatory constraint",
      "distinct-values constraint: exception to constraint: Archduke Leopold Wilhelm in his Gallery in Brussels (IV), The Gallery of Archduke Leopold in Brussels (II), The Arrest of Christ with kiss of Judas and ear of Malchus, The Resurrection, The Gallery of Archduke Leopold in Brussels (III), The Gallery of Archduke Leopold in Brussels (I), Pala Bichi",
      "item-requires-statement constraint: property: genre; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: creator; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: depicts; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: made from material; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: collection",
      "item-requires-statement constraint: property: height; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: width; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inventory number",
      "item-requires-statement constraint: property: location; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inception; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: copyright status; constraint status: suggestion constraint",
      "subject type constraint: class: artwork series, building, artificial physical object, work of art; relation: instance of; constraint status: mandatory constraint",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as reference"
    ],
    "before": [
      "label in language constraint: Wikimedia language code: de",
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: \\d+; constraint status: mandatory constraint",
      "distinct-values constraint: exception to constraint: Archduke Leopold Wilhelm in his Gallery in Brussels (IV), The Gallery of Archduke Leopold in Brussels (II), The Arrest of Christ with kiss of Judas and ear of Malchus, The Resurrection, The Gallery of Archduke Leopold in Brussels (III), The Gallery of Archduke Leopold in Brussels (I), Pala Bichi",
      "item-requires-statement constraint: property: genre; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: creator; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: depicts; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: made from material; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: collection",
      "item-requires-statement constraint: property: height; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: width; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inventory number",
      "item-requires-statement constraint: property: location; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inception; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: copyright status; constraint status: suggestion constraint",
      "subject type constraint: class: artwork series, building, artificial physical object, work of art; relation: instance of; constraint status: mandatory constraint",
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|136"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---

## 004. `reform_Q121727208_P1340_2445180072`

| Field | Value |
|---|---|
| qid | Q121727208 |
| property | P1340 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21510865 |
| group_key | TBOX::P1340::2445180072 |
| tbox_revision_key | TBOX::P1340::2445180072 |

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
| rationale | Type/value-type constraint classes and relations compared using P2308/P2309. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Trivialist",
  "kind": "T_BOX",
  "property_revision_id": 2445180072,
  "property_revision_prev": 2445179961
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-22T08:44:55",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1340",
  "report_revision_new": 2445421873,
  "report_revision_old": 2444860627,
  "report_violation_type": "Type Q|5, Q|95074, Q|146, Q|10832, Q|43577, Q|726, Q|1160573, Q|144, Q|39367, Q|3658341, Q|1114461, Q|584529, Q|729",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "fictional human or non-human character in a narrative work of art",
    "small domesticated carnivorous mammal",
    "anthropomorphic sexual device",
    "type of pet breed",
    "domesticated four-footed mammal from the equine family",
    "selectively bred form of the domesticated horse",
    "domesticated species of canid",
    "group of closely related and visibly similar domestic dogs",
    "fictional character appearing in written works",
    "fictional character in comics",
    "robot with its body shape built to resemble that of the human body",
    "kingdom of multicellular eukaryotic organisms"
  ],
  "report_violation_type_labels_en": [
    "human",
    "character",
    "cat",
    "sex doll",
    "cat breed",
    "horse",
    "horse breed",
    "dog",
    "dog breed",
    "literary character",
    "comics character",
    "humanoid robot",
    "Animalia"
  ],
  "report_violation_type_normalized": "Type Q|5, Q|95074, Q|146, Q|10832, Q|43577, Q|726, Q|1160573, Q|144, Q|39367, Q|3658341, Q|1114461, Q|584529, Q|729",
  "report_violation_type_qids": [
    "Q5",
    "Q95074",
    "Q146",
    "Q10832",
    "Q43577",
    "Q726",
    "Q1160573",
    "Q144",
    "Q39367",
    "Q3658341",
    "Q1114461",
    "Q584529",
    "Q729"
  ],
  "report_violation_type_raw": "Type Q|5, Q|95074, Q|146, Q|10832, Q|43577, Q|726, Q|1160573, Q|144, Q|39367, Q|3658341, Q|1114461, Q|584529, Q|729",
  "value": null,
  "value_current_2026": [
    "Q17126729"
  ],
  "value_current_2026_descriptions_en": [
    "eye color"
  ],
  "value_current_2026_labels_en": [
    "red"
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
    "description": "color of the irises of a person's eyes",
    "label": "eye color"
  },
  "qid": {
    "description": null,
    "label": "Varginha alien"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
      "constraint_qid": "Q21503250",
      "qualifiers": [
        {
          "property_id": "P2308",
          "values": [
            "Q1066288",
            "Q1114461",
            "Q11422",
            "Q1160573",
            "Q144",
            "Q146",
            "Q3658341",
            "Q39367",
            "Q43577",
            "Q5",
            "Q584529",
            "Q726",
            "Q729",
            "Q95074"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q30208840"
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
  "after_constraint_count": 7,
  "author": "Trivialist",
  "before_constraint_count": 7,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "small item resembling a person or animal, sometimes part of a larger work",
              "id": "Q1066288",
              "label_en": "figurine"
            },
            {
              "description_en": "fictional character in comics",
              "id": "Q1114461",
              "label_en": "comics character"
            },
            {
              "description_en": "object intended to be played with",
              "id": "Q11422",
              "label_en": "toy"
            },
            {
              "description_en": "selectively bred form of the domesticated horse",
              "id": "Q1160573",
              "label_en": "horse breed"
            },
            {
              "description_en": "domesticated species of canid",
              "id": "Q144",
              "label_en": "dog"
            },
            {
              "description_en": "small domesticated carnivorous mammal",
              "id": "Q146",
              "label_en": "cat"
            },
            {
              "description_en": "fictional character appearing in written works",
              "id": "Q3658341",
              "label_en": "literary character"
            },
            {
              "description_en": "group of closely related and visibly similar domestic dogs",
              "id": "Q39367",
              "label_en": "dog breed"
            },
            {
              "description_en": "type of pet breed",
              "id": "Q43577",
              "label_en": "cat breed"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            {
              "description_en": "robot with its body shape built to resemble that of the human body",
              "id": "Q584529",
              "label_en": "humanoid robot"
            },
            {
              "description_en": "domesticated four-footed mammal from the equine family",
              "id": "Q726",
              "label_en": "horse"
            },
            {
              "description_en": "kingdom of multicellular eukaryotic organisms",
              "id": "Q729",
              "label_en": "Animalia"
            },
            {
              "description_en": "fictional human or non-human character in a narrative work of art",
              "id": "Q95074",
              "label_en": "character"
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
              "description_en": "eye color",
              "id": "Q112135905",
              "label_en": "garnet"
            },
            {
              "description_en": "eye color",
              "id": "Q112259529",
              "label_en": "gold"
            },
            {
              "description_en": "eye color",
              "id": "Q120609071",
              "label_en": "copper"
            },
            {
              "description_en": "case in which the eye color of a character in a video game is determined by the player",
              "id": "Q123138274",
              "label_en": "eye color determined by the player"
            },
            {
              "description_en": "color shade of green eyes",
              "id": "Q131680960",
              "label_en": "emerald"
            },
            {
              "description_en": "eye color",
              "id": "Q17122705",
              "label_en": "brown"
            },
            {
              "description_en": "eye color",
              "id": "Q17122740",
              "label_en": "hazel"
            },
            {
              "description_en": "eye color",
              "id": "Q17122834",
              "label_en": "blue"
            },
            {
              "description_en": "eye color",
              "id": "Q17122854",
              "label_en": "green"
            },
            {
              "description_en": "eye color",
              "id": "Q17126729",
              "label_en": "red"
            },
            {
              "description_en": "eye color",
              "id": "Q17244465",
              "label_en": "black"
            },
            {
              "description_en": "eye color",
              "id": "Q17244894",
              "label_en": "dark brown"
            },
            {
              "description_en": "eye color",
              "id": "Q17245659",
              "label_en": "gray"
            },
            {
              "description_en": "eye color",
              "id": "Q17291407",
              "label_en": "amber"
            },
            {
              "description_en": "eye color",
              "id": "Q27777837",
              "label_en": "yellow"
            },
            {
              "description_en": "eye color",
              "id": "Q27839441",
              "label_en": "purple"
            },
            {
              "description_en": "eye color",
              "id": "Q3375649",
              "label_en": "blue-green"
            },
            {
              "description_en": "eye color",
              "id": "Q42845936",
              "label_en": "blue-gray"
            },
            {
              "description_en": "eye color",
              "id": "Q59318252",
              "label_en": "pink"
            },
            {
              "description_en": "eye color",
              "id": "Q59318527",
              "label_en": "orange"
            },
            {
              "description_en": "eye color",
              "id": "Q60172366",
              "label_en": "dark blue"
            },
            {
              "description_en": "eye color",
              "id": "Q61600018",
              "label_en": "light blue"
            },
            {
              "description_en": "eye color",
              "id": "Q62391724",
              "label_en": "white"
            },
            {
              "description_en": "eye colour",
              "id": "Q66821598",
              "label_en": "silver"
            },
            "... omitted 5 items"
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
          "P2306": [
            {
              "description_en": "image of relevant illustration of the subject; if available, also use more specific properties (sample: coat of arms image, locator map, flag image, signature image, logo image, collage image)",
              "id": "P18",
              "label_en": "image"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "polygenic phenotypic character",
              "id": "Q23786",
              "label_en": "eye color"
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
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "fictional character",
              "id": "Q7964329",
              "label_en": "Walter Blythe"
            }
          ],
          "P2305": [
            {
              "description_en": "primary color between purple and green in the spectrum",
              "id": "Q1088",
              "label_en": "blue"
            },
            {
              "description_en": "eye color",
              "id": "Q120609071",
              "label_en": "copper"
            },
            {
              "description_en": "brown color",
              "id": "Q15699769",
              "label_en": "dark brown"
            },
            {
              "description_en": "group of colors",
              "id": "Q1602687",
              "label_en": "light blue"
            },
            {
              "description_en": "color visible between blue and green; subtractive (CMY) primary color",
              "id": "Q180778",
              "label_en": "cyan"
            },
            {
              "description_en": "color",
              "id": "Q2040833",
              "label_en": "chestnut"
            },
            {
              "description_en": "dark shade of the color green",
              "id": "Q22963901",
              "label_en": "dark green"
            },
            {
              "description_en": "lightest color",
              "id": "Q23444",
              "label_en": "white"
            },
            {
              "description_en": "darkest color",
              "id": "Q23445",
              "label_en": "black"
            },
            {
              "description_en": "color",
              "id": "Q244822",
              "label_en": "garnet"
            },
            {
              "description_en": "light shade of purple",
              "id": "Q2468392",
              "label_en": "lavender"
            },
            {
              "description_en": "web color (grey ca. 17%)",
              "id": "Q24837023",
              "label_en": "light grey"
            },
            {
              "description_en": "achromatic color between black and grey (grey ca. 34%)",
              "id": "Q25614085",
              "label_en": "dark grey"
            },
            {
              "description_en": "additive primary color, visible between blue and yellow",
              "id": "Q3133",
              "label_en": "green"
            },
            {
              "description_en": "prismatic and primary color with longest visible wavelength in the electromagnetic spectrum",
              "id": "Q3142",
              "label_en": "red"
            },
            {
              "description_en": "metallic color tone resembling gray that is a representation of the color of polished silver",
              "id": "Q317802",
              "label_en": "silver"
            },
            {
              "description_en": "range of colors with the hues between blue and red",
              "id": "Q3257809",
              "label_en": "purple"
            },
            {
              "description_en": "color",
              "id": "Q3641131",
              "label_en": "dark blue"
            },
            {
              "description_en": "brown color",
              "id": "Q3985184",
              "label_en": "dark brown"
            },
            {
              "description_en": "intermediate color between black and white; for e.g. color of a cloud-covered sky, ash and lead",
              "id": "Q42519",
              "label_en": "grey"
            },
            {
              "description_en": "any of the colors between bluish red (purple) and red, of medium to high brightness and of low to moderate saturation",
              "id": "Q429220",
              "label_en": "pink"
            },
            {
              "description_en": "group of colours",
              "id": "Q4405716",
              "label_en": "light green"
            },
            {
              "description_en": "color",
              "id": "Q47071",
              "label_en": "brown"
            },
            {
              "description_en": "color",
              "id": "Q5223370",
              "label_en": "dark red"
            },
            "... omitted 3 items"
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
              "id": "Q54828449",
              "label_en": "as qualifier"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "fictional character in comics",
              "id": "Q1114461",
              "label_en": "comics character"
            },
            {
              "description_en": "object intended to be played with",
              "id": "Q11422",
              "label_en": "toy"
            },
            {
              "description_en": "selectively bred form of the domesticated horse",
              "id": "Q1160573",
              "label_en": "horse breed"
            },
            {
              "description_en": "domesticated species of canid",
              "id": "Q144",
              "label_en": "dog"
            },
            {
              "description_en": "small domesticated carnivorous mammal",
              "id": "Q146",
              "label_en": "cat"
            },
            {
              "description_en": "fictional character appearing in written works",
              "id": "Q3658341",
              "label_en": "literary character"
            },
            {
              "description_en": "group of closely related and visibly similar domestic dogs",
              "id": "Q39367",
              "label_en": "dog breed"
            },
            {
              "description_en": "type of pet breed",
              "id": "Q43577",
              "label_en": "cat breed"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            {
              "description_en": "robot with its body shape built to resemble that of the human body",
              "id": "Q584529",
              "label_en": "humanoid robot"
            },
            {
              "description_en": "domesticated four-footed mammal from the equine family",
              "id": "Q726",
              "label_en": "horse"
            },
            {
              "description_en": "kingdom of multicellular eukaryotic organisms",
              "id": "Q729",
              "label_en": "Animalia"
            },
            {
              "description_en": "fictional human or non-human character in a narrative work of art",
              "id": "Q95074",
              "label_en": "character"
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
              "description_en": "eye color",
              "id": "Q112135905",
              "label_en": "garnet"
            },
            {
              "description_en": "eye color",
              "id": "Q112259529",
              "label_en": "gold"
            },
            {
              "description_en": "eye color",
              "id": "Q120609071",
              "label_en": "copper"
            },
            {
              "description_en": "case in which the eye color of a character in a video game is determined by the player",
              "id": "Q123138274",
              "label_en": "eye color determined by the player"
            },
            {
              "description_en": "color shade of green eyes",
              "id": "Q131680960",
              "label_en": "emerald"
            },
            {
              "description_en": "eye color",
              "id": "Q17122705",
              "label_en": "brown"
            },
            {
              "description_en": "eye color",
              "id": "Q17122740",
              "label_en": "hazel"
            },
            {
              "description_en": "eye color",
              "id": "Q17122834",
              "label_en": "blue"
            },
            {
              "description_en": "eye color",
              "id": "Q17122854",
              "label_en": "green"
            },
            {
              "description_en": "eye color",
              "id": "Q17126729",
              "label_en": "red"
            },
            {
              "description_en": "eye color",
              "id": "Q17244465",
              "label_en": "black"
            },
            {
              "description_en": "eye color",
              "id": "Q17244894",
              "label_en": "dark brown"
            },
            {
              "description_en": "eye color",
              "id": "Q17245659",
              "label_en": "gray"
            },
            {
              "description_en": "eye color",
              "id": "Q17291407",
              "label_en": "amber"
            },
            {
              "description_en": "eye color",
              "id": "Q27777837",
              "label_en": "yellow"
            },
            {
              "description_en": "eye color",
              "id": "Q27839441",
              "label_en": "purple"
            },
            {
              "description_en": "eye color",
              "id": "Q3375649",
              "label_en": "blue-green"
            },
            {
              "description_en": "eye color",
              "id": "Q42845936",
              "label_en": "blue-gray"
            },
            {
              "description_en": "eye color",
              "id": "Q59318252",
              "label_en": "pink"
            },
            {
              "description_en": "eye color",
              "id": "Q59318527",
              "label_en": "orange"
            },
            {
              "description_en": "eye color",
              "id": "Q60172366",
              "label_en": "dark blue"
            },
            {
              "description_en": "eye color",
              "id": "Q61600018",
              "label_en": "light blue"
            },
            {
              "description_en": "eye color",
              "id": "Q62391724",
              "label_en": "white"
            },
            {
              "description_en": "eye colour",
              "id": "Q66821598",
              "label_en": "silver"
            },
            "... omitted 5 items"
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
          "P2306": [
            {
              "description_en": "image of relevant illustration of the subject; if available, also use more specific properties (sample: coat of arms image, locator map, flag image, signature image, logo image, collage image)",
              "id": "P18",
              "label_en": "image"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "polygenic phenotypic character",
              "id": "Q23786",
              "label_en": "eye color"
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
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "fictional character",
              "id": "Q7964329",
              "label_en": "Walter Blythe"
            }
          ],
          "P2305": [
            {
              "description_en": "primary color between purple and green in the spectrum",
              "id": "Q1088",
              "label_en": "blue"
            },
            {
              "description_en": "eye color",
              "id": "Q120609071",
              "label_en": "copper"
            },
            {
              "description_en": "brown color",
              "id": "Q15699769",
              "label_en": "dark brown"
            },
            {
              "description_en": "group of colors",
              "id": "Q1602687",
              "label_en": "light blue"
            },
            {
              "description_en": "color visible between blue and green; subtractive (CMY) primary color",
              "id": "Q180778",
              "label_en": "cyan"
            },
            {
              "description_en": "color",
              "id": "Q2040833",
              "label_en": "chestnut"
            },
            {
              "description_en": "dark shade of the color green",
              "id": "Q22963901",
              "label_en": "dark green"
            },
            {
              "description_en": "lightest color",
              "id": "Q23444",
              "label_en": "white"
            },
            {
              "description_en": "darkest color",
              "id": "Q23445",
              "label_en": "black"
            },
            {
              "description_en": "color",
              "id": "Q244822",
              "label_en": "garnet"
            },
            {
              "description_en": "light shade of purple",
              "id": "Q2468392",
              "label_en": "lavender"
            },
            {
              "description_en": "web color (grey ca. 17%)",
              "id": "Q24837023",
              "label_en": "light grey"
            },
            {
              "description_en": "achromatic color between black and grey (grey ca. 34%)",
              "id": "Q25614085",
              "label_en": "dark grey"
            },
            {
              "description_en": "additive primary color, visible between blue and yellow",
              "id": "Q3133",
              "label_en": "green"
            },
            {
              "description_en": "prismatic and primary color with longest visible wavelength in the electromagnetic spectrum",
              "id": "Q3142",
              "label_en": "red"
            },
            {
              "description_en": "metallic color tone resembling gray that is a representation of the color of polished silver",
              "id": "Q317802",
              "label_en": "silver"
            },
            {
              "description_en": "range of colors with the hues between blue and red",
              "id": "Q3257809",
              "label_en": "purple"
            },
            {
              "description_en": "color",
              "id": "Q3641131",
              "label_en": "dark blue"
            },
            {
              "description_en": "brown color",
              "id": "Q3985184",
              "label_en": "dark brown"
            },
            {
              "description_en": "intermediate color between black and white; for e.g. color of a cloud-covered sky, ash and lead",
              "id": "Q42519",
              "label_en": "grey"
            },
            {
              "description_en": "any of the colors between bluish red (purple) and red, of medium to high brightness and of low to moderate saturation",
              "id": "Q429220",
              "label_en": "pink"
            },
            {
              "description_en": "group of colours",
              "id": "Q4405716",
              "label_en": "light green"
            },
            {
              "description_en": "color",
              "id": "Q47071",
              "label_en": "brown"
            },
            {
              "description_en": "color",
              "id": "Q5223370",
              "label_en": "dark red"
            },
            "... omitted 3 items"
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
              "id": "Q54828449",
              "label_en": "as qualifier"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "1bf8eb100f990833313ef58451cec6e78c561abe",
  "hash_before": "cb371e31384dd73137dcb57a0cd20b6a6868b838",
  "property_revision_id": 2445180072,
  "property_revision_prev": 2445179961,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q1066288"
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
            "Q1114461",
            "Q11422",
            "Q1160573",
            "Q144",
            "Q146",
            "Q3658341",
            "Q39367",
            "Q43577",
            "Q5",
            "Q584529",
            "Q726",
            "Q729",
            "Q95074"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q30208840"
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
      "subject type constraint: class: figurine, comics character, toy, horse breed, dog, cat, literary character, dog breed, cat breed, human, humanoid robot, horse, Animalia, character; relation: instance or subclass of; constraint status: mandatory constraint",
      "one-of constraint: reason for deprecated rank: constraint provides suggestions for manual input; item of property constraint: garnet, gold, copper, eye color determined by the player, emerald, brown, hazel, blue, green, red, black, dark brown, gray, amber, yellow, purple, blue-green, blue-gray, pink, orange, dark blue, light blue, white, silver, light green, light brown, dark green, cyan, teal",
      "value-requires-statement constraint: property: image; constraint status: suggestion constraint",
      "value-type constraint: class: eye color; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы",
      "none-of constraint: exception to constraint: Walter Blythe; item of property constraint: blue, copper, dark brown, light blue, cyan, chestnut, dark green, white, black, garnet, lavender, light grey, dark grey, green, red, silver, purple, dark blue, dark brown, grey, pink, light green, brown, dark red, light red, light brown, yellow",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as qualifier"
    ],
    "before": [
      "subject type constraint: class: comics character, toy, horse breed, dog, cat, literary character, dog breed, cat breed, human, humanoid robot, horse, Animalia, character; relation: instance or subclass of; constraint status: mandatory constraint",
      "one-of constraint: reason for deprecated rank: constraint provides suggestions for manual input; item of property constraint: garnet, gold, copper, eye color determined by the player, emerald, brown, hazel, blue, green, red, black, dark brown, gray, amber, yellow, purple, blue-green, blue-gray, pink, orange, dark blue, light blue, white, silver, light green, light brown, dark green, cyan, teal",
      "value-requires-statement constraint: property: image; constraint status: suggestion constraint",
      "value-type constraint: class: eye color; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы",
      "none-of constraint: exception to constraint: Walter Blythe; item of property constraint: blue, copper, dark brown, light blue, cyan, chestnut, dark green, white, black, garnet, lavender, light grey, dark grey, green, red, silver, purple, dark blue, dark brown, grey, pink, light green, brown, dark red, light red, light brown, yellow",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as qualifier"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|5, Q|95074, Q|146, Q|10832, Q|43577, Q|726, Q|1160573, Q|144, Q|39367, Q|3658341, Q|1114461, Q|584529, Q|729"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint"
  },
  {
    "property_ids": [
      "P2308",
      "P2309"
    ],
    "result": "RELAXATION_SET_EXPANSION",
    "step": "set_semantics"
  }
]
```

---

## 005. `reform_Q123121331_P9058_2216480186`

| Field | Value |
|---|---|
| qid | Q123121331 |
| property | P9058 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | TBOX::P9058::2216480186 |
| tbox_revision_key | TBOX::P9058::2216480186 |

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
| rationale | Constraint qualifiers compared with generic set semantics. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Ayack",
  "kind": "T_BOX",
  "property_revision_id": 2216480186,
  "property_revision_prev": 2194709188
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-07-31T07:57:39",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P9058",
  "report_revision_new": 2217118138,
  "report_revision_old": 2216122682,
  "report_violation_type": "Item P|20",
  "report_violation_type_normalized": "Item P|20",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|20",
  "report_violation_types": [
    "Item P|20",
    "Item P|570"
  ],
  "value": null,
  "value_current_2026": [
    "h89v4gOZnBZ0"
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
    "description": "identifier for an entry in INSEE's 'Fichier des personnes décédées' (deaths since 1970)",
    "label": "Fichier des personnes décédées ID (matchID)"
  },
  "qid": {
    "description": null,
    "label": "Maurice Soubeyran"
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
    "label_en": "property scope constraint",
    "qid": "Q53869507"
  },
  {
    "label_en": "allowed-entity-types constraint",
    "qid": "Q52004125"
  },
  {
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
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
    "label_en": "label in language constraint",
    "qid": "Q108139345"
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P424",
          "values": [
            "fr",
            "mul"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 9,
  "author": "Ayack",
  "before_constraint_count": 9,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "fr"
            },
            {
              "value": "mul"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
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
              "value": "[A-Za-z0-9_-]{2,100}"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "most specific known (e.g. city instead of country, or hospital instead of city) death location of a person, animal or fictional character",
              "id": "P20",
              "label_en": "place of death"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "date on which the subject died",
              "id": "P570",
              "label_en": "date of death"
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
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "fr"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
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
              "value": "[A-Za-z0-9_-]{2,100}"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "most specific known (e.g. city instead of country, or hospital instead of city) death location of a person, animal or fictional character",
              "id": "P20",
              "label_en": "place of death"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "date on which the subject died",
              "id": "P570",
              "label_en": "date of death"
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
  "hash_after": "aadd4a6310aaab6fc1916dba0787f0fd614ebb9b",
  "hash_before": "7925ca3cd7a46cf7d54c3792deeec5fafacc0a22",
  "property_revision_id": 2216480186,
  "property_revision_prev": 2194709188,
  "qualifier_value_changes": [
    {
      "added_values": [
        "mul"
      ],
      "constraint_qid": "Q108139345",
      "qualifier_property": "P424",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P424",
          "values": [
            "fr"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "label in language constraint: Wikimedia language code: fr, mul",
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: [A-Za-z0-9_-]{2,100}",
      "distinct-values constraint: constraint status: mandatory constraint",
      "item-requires-statement constraint: property: place of death; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: date of death; constraint status: mandatory constraint",
      "subject type constraint: class: human; relation: instance of; constraint status: mandatory constraint",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "property scope constraint: property scope: as main value, as reference"
    ],
    "before": [
      "label in language constraint: Wikimedia language code: fr",
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: [A-Za-z0-9_-]{2,100}",
      "distinct-values constraint: constraint status: mandatory constraint",
      "item-requires-statement constraint: property: place of death; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: date of death; constraint status: mandatory constraint",
      "subject type constraint: class: human; relation: instance of; constraint status: mandatory constraint",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|20"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---

## 006. `reform_Q123916603_P403_2442802452`

| Field | Value |
|---|---|
| qid | Q123916603 |
| property | P403 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | TBOX::P403::2442802452 |
| tbox_revision_key | TBOX::P403::2442802452 |

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
| rationale | Type/value-type constraint classes and relations compared using P2308/P2309. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Vlk",
  "kind": "T_BOX",
  "property_revision_id": 2442802452,
  "property_revision_prev": 2421662999
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-19T10:56:29",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P403",
  "report_revision_new": 2444036897,
  "report_revision_old": 2443833509,
  "report_violation_type": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "report_violation_type_descriptions_en": [
    "any flowing body of water",
    "river which only exists in fiction",
    "river that only exists in myth",
    "terrestrial water source",
    "low area between hills, often with a river running through it",
    "large persistent body of ice",
    "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
    "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
    "any significant accumulation of water, generally on a planet's surface",
    "body of water without noticeable current",
    "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
    "waterbody only existing in fiction",
    "small stream, i.e. a small natural watercourse"
  ],
  "report_violation_type_labels_en": [
    "watercourse",
    "fictional river",
    "mythical river",
    "spring",
    "valley",
    "glacier",
    "lake system",
    "drainage system",
    "body of water",
    "still waters",
    "wadi",
    "fictional body of water",
    "brook"
  ],
  "report_violation_type_normalized": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "report_violation_type_qids": [
    "Q355304",
    "Q16338046",
    "Q24336031",
    "Q124714",
    "Q39816",
    "Q35666",
    "Q104347069",
    "Q285451",
    "Q15324",
    "Q337567",
    "Q187971",
    "Q16500104",
    "Q63565252"
  ],
  "report_violation_type_raw": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "value": null,
  "value_current_2026": [
    "Q98"
  ],
  "value_current_2026_descriptions_en": [
    "ocean between Asia, Australia and the Americas"
  ],
  "value_current_2026_labels_en": [
    "Pacific Ocean"
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
    "description": "the body of water to which the watercourse drains",
    "label": "mouth of the watercourse"
  },
  "qid": {
    "description": "Reserva natural de Chile",
    "label": "Desembocadura del río Biobío"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
            "Q104347069",
            "Q124714",
            "Q15324",
            "Q16338046",
            "Q16500104",
            "Q187971",
            "Q24336031",
            "Q285451",
            "Q337567",
            "Q355304",
            "Q35666",
            "Q39816",
            "Q63565252"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q21503252"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 6,
  "author": "Vlk",
  "before_constraint_count": 6,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
              "id": "Q104347069",
              "label_en": "lake system"
            },
            {
              "description_en": "terrestrial water source",
              "id": "Q124714",
              "label_en": "spring"
            },
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "river which only exists in fiction",
              "id": "Q16338046",
              "label_en": "fictional river"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "river that only exists in myth",
              "id": "Q24336031",
              "label_en": "mythical river"
            },
            {
              "description_en": "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
              "id": "Q285451",
              "label_en": "drainage system"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "any flowing body of water",
              "id": "Q355304",
              "label_en": "watercourse"
            },
            {
              "description_en": "large persistent body of ice",
              "id": "Q35666",
              "label_en": "glacier"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "small stream, i.e. a small natural watercourse",
              "id": "Q63565252",
              "label_en": "brook"
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
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "height of the item (geographical object) as measured relative to sea level",
              "id": "P2044",
              "label_en": "elevation above sea level"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            {
              "description_en": "specify if the stream confluence is a left bank or right bank tributary",
              "id": "P3871",
              "label_en": "tributary orientation"
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
              "description_en": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
              "id": "P613",
              "label_en": "OS grid reference"
            },
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "infrastructure that conveys sewage or surface runoff",
              "id": "Q156849",
              "label_en": "sewer network"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "meeting of two or more bodies of flowing water",
              "id": "Q723748",
              "label_en": "confluence"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "stream in Highland, Scotland, UK, tributary of the Allt Cuaich and of the Cuaich Aqueduct",
              "id": "Q112729719",
              "label_en": "Féith Chàm"
            },
            {
              "description_en": "canal section in Argyll and Bute, Scotland, UK, flows west into Loch Crinan at Crinan, and east into the canal section from Cairnbaan to Ardrishaig",
              "id": "Q56664718",
              "label_en": "Crinan Canal, section from Crinan to Cairnbaan"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
              "id": "Q104347069",
              "label_en": "lake system"
            },
            {
              "description_en": "terrestrial water source",
              "id": "Q124714",
              "label_en": "spring"
            },
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "river which only exists in fiction",
              "id": "Q16338046",
              "label_en": "fictional river"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "river that only exists in myth",
              "id": "Q24336031",
              "label_en": "mythical river"
            },
            {
              "description_en": "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
              "id": "Q285451",
              "label_en": "drainage system"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "any flowing body of water",
              "id": "Q355304",
              "label_en": "watercourse"
            },
            {
              "description_en": "large persistent body of ice",
              "id": "Q35666",
              "label_en": "glacier"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
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
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "height of the item (geographical object) as measured relative to sea level",
              "id": "P2044",
              "label_en": "elevation above sea level"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            {
              "description_en": "specify if the stream confluence is a left bank or right bank tributary",
              "id": "P3871",
              "label_en": "tributary orientation"
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
              "description_en": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
              "id": "P613",
              "label_en": "OS grid reference"
            },
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "infrastructure that conveys sewage or surface runoff",
              "id": "Q156849",
              "label_en": "sewer network"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "meeting of two or more bodies of flowing water",
              "id": "Q723748",
              "label_en": "confluence"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "stream in Highland, Scotland, UK, tributary of the Allt Cuaich and of the Cuaich Aqueduct",
              "id": "Q112729719",
              "label_en": "Féith Chàm"
            },
            {
              "description_en": "canal section in Argyll and Bute, Scotland, UK, flows west into Loch Crinan at Crinan, and east into the canal section from Cairnbaan to Ardrishaig",
              "id": "Q56664718",
              "label_en": "Crinan Canal, section from Crinan to Cairnbaan"
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
  "hash_after": "36188fe72951315c87372c2b0c88051dd7ea3646",
  "hash_before": "1eae5a89c1de92534f9fe810620210d2cac7c93f",
  "property_revision_id": 2442802452,
  "property_revision_prev": 2421662999,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q63565252"
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
            "Q104347069",
            "Q124714",
            "Q15324",
            "Q16338046",
            "Q16500104",
            "Q187971",
            "Q24336031",
            "Q285451",
            "Q337567",
            "Q355304",
            "Q35666",
            "Q39816"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q21503252"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "subject type constraint: class: lake system, spring, body of water, fictional river, fictional body of water, wadi, mythical river, drainage system, still waters, watercourse, glacier, valley, brook; relation: instance of",
      "allowed qualifiers constraint: property: located in the administrative territorial entity, end cause, subject named as, object named as, elevation above sea level, reason for deprecated rank, location, tributary orientation, start time, end time, OS grid reference, coordinate location, reason for preferred rank",
      "value-type constraint: class: body of water, sewer network, fictional body of water, wadi, still waters, valley, confluence; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "single-best-value constraint: exception to constraint: Féith Chàm, Crinan Canal, section from Crinan to Cairnbaan",
      "property scope constraint: property scope: as main value"
    ],
    "before": [
      "subject type constraint: class: lake system, spring, body of water, fictional river, fictional body of water, wadi, mythical river, drainage system, still waters, watercourse, glacier, valley; relation: instance of",
      "allowed qualifiers constraint: property: located in the administrative territorial entity, end cause, subject named as, object named as, elevation above sea level, reason for deprecated rank, location, tributary orientation, start time, end time, OS grid reference, coordinate location, reason for preferred rank",
      "value-type constraint: class: body of water, sewer network, fictional body of water, wadi, still waters, valley, confluence; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "single-best-value constraint: exception to constraint: Féith Chàm, Crinan Canal, section from Crinan to Cairnbaan",
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint"
  },
  {
    "property_ids": [
      "P2308",
      "P2309"
    ],
    "result": "RELAXATION_SET_EXPANSION",
    "step": "set_semantics"
  }
]
```

---

## 007. `reform_Q12471498_P1006_2407207080`

| Field | Value |
|---|---|
| qid | Q12471498 |
| property | P1006 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| group_key | TBOX::P1006::2407207080 |
| tbox_revision_key | TBOX::P1006::2407207080 |

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
| rationale | Constraint qualifiers compared with generic set semantics. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Bob08",
  "kind": "T_BOX",
  "property_revision_id": 2407207080,
  "property_revision_prev": 2376765283
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-09-19T22:59:13",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1006",
  "report_revision_new": 2407507647,
  "report_revision_old": 2407109254,
  "report_violation_type": "Label in nl language",
  "report_violation_type_normalized": "Label in nl language",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Label in nl language",
  "value": null,
  "value_current_2026": [
    "074637401"
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
    "description": "identifier for person names (not: works nor organisations) from the Dutch National Thesaurus for Author names (which also contains non-authors)",
    "label": "Nationale Thesaurus voor Auteursnamen ID"
  },
  "qid": {
    "description": "Indonesian Islamic religious leader (1926–2023)",
    "label": "Ali Yafie"
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
    "label_en": "subject type constraint",
    "qid": "Q21503250"
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
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P2316",
          "values": [
            "Q62026391"
          ]
        },
        {
          "property_id": "P424",
          "values": [
            "mul",
            "nl"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 8,
  "author": "Bob08",
  "before_constraint_count": 8,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ],
          "P424": [
            {
              "value": "mul"
            },
            {
              "value": "nl"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
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
              "value": "\\d{8}(\\d|X)"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "occurrence of a fact or object in space-time; instantiation of a property in an object",
              "id": "Q1190554",
              "label_en": "occurrence"
            },
            {
              "description_en": "human being that only exists in fictional works",
              "id": "Q15632617",
              "label_en": "fictional human"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "human (as opposed to supernatural) character in the Old Testament/Hebrew Bible or New Testament",
              "id": "Q20643955",
              "label_en": "human biblical figure"
            },
            {
              "description_en": "character who is hypothesized to exist, but where evidence is not conclusive",
              "id": "Q21070598",
              "label_en": "figure that may or may not be fictional"
            },
            {
              "description_en": "being that has certain capacities or attributes constituting personhood (for humans, use Q5 [human] with P31 [instance of])",
              "id": "Q215627",
              "label_en": "person"
            },
            {
              "description_en": "social entity established to meet needs or pursue goals",
              "id": "Q43229",
              "label_en": "organization"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            {
              "description_en": "fictitious name that a person or group assumes for a particular purpose, which differs from their original or true name (orthonym)",
              "id": "Q61002",
              "label_en": "pseudonym"
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
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ],
          "P424": [
            {
              "value": "nl"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
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
              "value": "\\d{8}(\\d|X)"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "occurrence of a fact or object in space-time; instantiation of a property in an object",
              "id": "Q1190554",
              "label_en": "occurrence"
            },
            {
              "description_en": "human being that only exists in fictional works",
              "id": "Q15632617",
              "label_en": "fictional human"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "human (as opposed to supernatural) character in the Old Testament/Hebrew Bible or New Testament",
              "id": "Q20643955",
              "label_en": "human biblical figure"
            },
            {
              "description_en": "character who is hypothesized to exist, but where evidence is not conclusive",
              "id": "Q21070598",
              "label_en": "figure that may or may not be fictional"
            },
            {
              "description_en": "being that has certain capacities or attributes constituting personhood (for humans, use Q5 [human] with P31 [instance of])",
              "id": "Q215627",
              "label_en": "person"
            },
            {
              "description_en": "social entity established to meet needs or pursue goals",
              "id": "Q43229",
              "label_en": "organization"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            {
              "description_en": "fictitious name that a person or group assumes for a particular purpose, which differs from their original or true name (orthonym)",
              "id": "Q61002",
              "label_en": "pseudonym"
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
  "hash_after": "ca030620ee7d5efdf9d65fc7e808ef7ff585b341",
  "hash_before": "e6f32c2b74e15e9b6a4698544e055002e406667b",
  "property_revision_id": 2407207080,
  "property_revision_prev": 2376765283,
  "qualifier_value_changes": [
    {
      "added_values": [
        "mul"
      ],
      "constraint_qid": "Q108139345",
      "qualifier_property": "P424",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P2316",
          "values": [
            "Q62026391"
          ]
        },
        {
          "property_id": "P424",
          "values": [
            "nl"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "label in language constraint: constraint status: suggestion constraint; Wikimedia language code: mul, nl",
      "single-value constraint: separator: subject named as",
      "format constraint: format as a regular expression: \\d{8}(\\d|X); constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "item-requires-statement constraint: property: VIAF cluster ID",
      "subject type constraint: class: occurrence, fictional human, group of humans, human biblical figure, figure that may or may not be fictional, person, organization, human, pseudonym; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as reference"
    ],
    "before": [
      "label in language constraint: constraint status: suggestion constraint; Wikimedia language code: nl",
      "single-value constraint: separator: subject named as",
      "format constraint: format as a regular expression: \\d{8}(\\d|X); constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "item-requires-statement constraint: property: VIAF cluster ID",
      "subject type constraint: class: occurrence, fictional human, group of humans, human biblical figure, figure that may or may not be fictional, person, organization, human, pseudonym; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Label in nl language"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---

## 008. `reform_Q1320666_P2517_2442912347`

| Field | Value |
|---|---|
| qid | Q1320666 |
| property | P2517 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | head |
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
| rationale | Constraint qualifiers compared with generic set semantics. |
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
  "report_violation_type": "Inverse",
  "report_violation_type_normalized": "Inverse",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Inverse",
  "value": null,
  "value_current_2026": [
    "Q111939707"
  ],
  "value_current_2026_descriptions_en": [
    "Wikimedia category"
  ],
  "value_current_2026_labels_en": [
    "Category:Recipients of the Frisch Medal"
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
    "description": "American economics award (1978–)",
    "label": "Frisch Medal"
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Inverse"
  },
  {
    "result": "Q19474404",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---

## 009. `reform_Q134738025_P403_2442802452`

| Field | Value |
|---|---|
| qid | Q134738025 |
| property | P403 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | TBOX::P403::2442802452 |
| tbox_revision_key | TBOX::P403::2442802452 |

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
| rationale | Type/value-type constraint classes and relations compared using P2308/P2309. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Vlk",
  "kind": "T_BOX",
  "property_revision_id": 2442802452,
  "property_revision_prev": 2421662999
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-21T10:49:32",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P403",
  "report_revision_new": 2444888525,
  "report_revision_old": 2444461949,
  "report_violation_type": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "report_violation_type_descriptions_en": [
    "any flowing body of water",
    "river which only exists in fiction",
    "river that only exists in myth",
    "terrestrial water source",
    "low area between hills, often with a river running through it",
    "large persistent body of ice",
    "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
    "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
    "any significant accumulation of water, generally on a planet's surface",
    "body of water without noticeable current",
    "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
    "waterbody only existing in fiction",
    "small stream, i.e. a small natural watercourse"
  ],
  "report_violation_type_labels_en": [
    "watercourse",
    "fictional river",
    "mythical river",
    "spring",
    "valley",
    "glacier",
    "lake system",
    "drainage system",
    "body of water",
    "still waters",
    "wadi",
    "fictional body of water",
    "brook"
  ],
  "report_violation_type_normalized": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "report_violation_type_qids": [
    "Q355304",
    "Q16338046",
    "Q24336031",
    "Q124714",
    "Q39816",
    "Q35666",
    "Q104347069",
    "Q285451",
    "Q15324",
    "Q337567",
    "Q187971",
    "Q16500104",
    "Q63565252"
  ],
  "report_violation_type_raw": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "value": null,
  "value_current_2026": [
    "Q1800895"
  ],
  "value_current_2026_descriptions_en": [
    "river in northern Maine, United States"
  ],
  "value_current_2026_labels_en": [
    "Allagash River"
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
    "description": "the body of water to which the watercourse drains",
    "label": "mouth of the watercourse"
  },
  "qid": {
    "description": "tributary of the Allagash River in Aroostook County, Maine, United States",
    "label": "Chemquasabamticook Stream"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
            "Q104347069",
            "Q124714",
            "Q15324",
            "Q16338046",
            "Q16500104",
            "Q187971",
            "Q24336031",
            "Q285451",
            "Q337567",
            "Q355304",
            "Q35666",
            "Q39816",
            "Q63565252"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q21503252"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 6,
  "author": "Vlk",
  "before_constraint_count": 6,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
              "id": "Q104347069",
              "label_en": "lake system"
            },
            {
              "description_en": "terrestrial water source",
              "id": "Q124714",
              "label_en": "spring"
            },
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "river which only exists in fiction",
              "id": "Q16338046",
              "label_en": "fictional river"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "river that only exists in myth",
              "id": "Q24336031",
              "label_en": "mythical river"
            },
            {
              "description_en": "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
              "id": "Q285451",
              "label_en": "drainage system"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "any flowing body of water",
              "id": "Q355304",
              "label_en": "watercourse"
            },
            {
              "description_en": "large persistent body of ice",
              "id": "Q35666",
              "label_en": "glacier"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "small stream, i.e. a small natural watercourse",
              "id": "Q63565252",
              "label_en": "brook"
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
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "height of the item (geographical object) as measured relative to sea level",
              "id": "P2044",
              "label_en": "elevation above sea level"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            {
              "description_en": "specify if the stream confluence is a left bank or right bank tributary",
              "id": "P3871",
              "label_en": "tributary orientation"
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
              "description_en": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
              "id": "P613",
              "label_en": "OS grid reference"
            },
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "infrastructure that conveys sewage or surface runoff",
              "id": "Q156849",
              "label_en": "sewer network"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "meeting of two or more bodies of flowing water",
              "id": "Q723748",
              "label_en": "confluence"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "stream in Highland, Scotland, UK, tributary of the Allt Cuaich and of the Cuaich Aqueduct",
              "id": "Q112729719",
              "label_en": "Féith Chàm"
            },
            {
              "description_en": "canal section in Argyll and Bute, Scotland, UK, flows west into Loch Crinan at Crinan, and east into the canal section from Cairnbaan to Ardrishaig",
              "id": "Q56664718",
              "label_en": "Crinan Canal, section from Crinan to Cairnbaan"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
              "id": "Q104347069",
              "label_en": "lake system"
            },
            {
              "description_en": "terrestrial water source",
              "id": "Q124714",
              "label_en": "spring"
            },
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "river which only exists in fiction",
              "id": "Q16338046",
              "label_en": "fictional river"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "river that only exists in myth",
              "id": "Q24336031",
              "label_en": "mythical river"
            },
            {
              "description_en": "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
              "id": "Q285451",
              "label_en": "drainage system"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "any flowing body of water",
              "id": "Q355304",
              "label_en": "watercourse"
            },
            {
              "description_en": "large persistent body of ice",
              "id": "Q35666",
              "label_en": "glacier"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
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
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "height of the item (geographical object) as measured relative to sea level",
              "id": "P2044",
              "label_en": "elevation above sea level"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            {
              "description_en": "specify if the stream confluence is a left bank or right bank tributary",
              "id": "P3871",
              "label_en": "tributary orientation"
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
              "description_en": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
              "id": "P613",
              "label_en": "OS grid reference"
            },
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "infrastructure that conveys sewage or surface runoff",
              "id": "Q156849",
              "label_en": "sewer network"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "meeting of two or more bodies of flowing water",
              "id": "Q723748",
              "label_en": "confluence"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "stream in Highland, Scotland, UK, tributary of the Allt Cuaich and of the Cuaich Aqueduct",
              "id": "Q112729719",
              "label_en": "Féith Chàm"
            },
            {
              "description_en": "canal section in Argyll and Bute, Scotland, UK, flows west into Loch Crinan at Crinan, and east into the canal section from Cairnbaan to Ardrishaig",
              "id": "Q56664718",
              "label_en": "Crinan Canal, section from Crinan to Cairnbaan"
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
  "hash_after": "36188fe72951315c87372c2b0c88051dd7ea3646",
  "hash_before": "1eae5a89c1de92534f9fe810620210d2cac7c93f",
  "property_revision_id": 2442802452,
  "property_revision_prev": 2421662999,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q63565252"
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
            "Q104347069",
            "Q124714",
            "Q15324",
            "Q16338046",
            "Q16500104",
            "Q187971",
            "Q24336031",
            "Q285451",
            "Q337567",
            "Q355304",
            "Q35666",
            "Q39816"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q21503252"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "subject type constraint: class: lake system, spring, body of water, fictional river, fictional body of water, wadi, mythical river, drainage system, still waters, watercourse, glacier, valley, brook; relation: instance of",
      "allowed qualifiers constraint: property: located in the administrative territorial entity, end cause, subject named as, object named as, elevation above sea level, reason for deprecated rank, location, tributary orientation, start time, end time, OS grid reference, coordinate location, reason for preferred rank",
      "value-type constraint: class: body of water, sewer network, fictional body of water, wadi, still waters, valley, confluence; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "single-best-value constraint: exception to constraint: Féith Chàm, Crinan Canal, section from Crinan to Cairnbaan",
      "property scope constraint: property scope: as main value"
    ],
    "before": [
      "subject type constraint: class: lake system, spring, body of water, fictional river, fictional body of water, wadi, mythical river, drainage system, still waters, watercourse, glacier, valley; relation: instance of",
      "allowed qualifiers constraint: property: located in the administrative territorial entity, end cause, subject named as, object named as, elevation above sea level, reason for deprecated rank, location, tributary orientation, start time, end time, OS grid reference, coordinate location, reason for preferred rank",
      "value-type constraint: class: body of water, sewer network, fictional body of water, wadi, still waters, valley, confluence; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "single-best-value constraint: exception to constraint: Féith Chàm, Crinan Canal, section from Crinan to Cairnbaan",
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint"
  },
  {
    "property_ids": [
      "P2308",
      "P2309"
    ],
    "result": "RELAXATION_SET_EXPANSION",
    "step": "set_semantics"
  }
]
```

---

## 010. `reform_Q1431507_P403_2442802452`

| Field | Value |
|---|---|
| qid | Q1431507 |
| property | P403 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | TBOX::P403::2442802452 |
| tbox_revision_key | TBOX::P403::2442802452 |

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
| rationale | Type/value-type constraint classes and relations compared using P2308/P2309. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Vlk",
  "kind": "T_BOX",
  "property_revision_id": 2442802452,
  "property_revision_prev": 2421662999
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-19T10:56:29",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P403",
  "report_revision_new": 2444036897,
  "report_revision_old": 2443833509,
  "report_violation_type": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "report_violation_type_descriptions_en": [
    "any flowing body of water",
    "river which only exists in fiction",
    "river that only exists in myth",
    "terrestrial water source",
    "low area between hills, often with a river running through it",
    "large persistent body of ice",
    "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
    "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
    "any significant accumulation of water, generally on a planet's surface",
    "body of water without noticeable current",
    "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
    "waterbody only existing in fiction",
    "small stream, i.e. a small natural watercourse"
  ],
  "report_violation_type_labels_en": [
    "watercourse",
    "fictional river",
    "mythical river",
    "spring",
    "valley",
    "glacier",
    "lake system",
    "drainage system",
    "body of water",
    "still waters",
    "wadi",
    "fictional body of water",
    "brook"
  ],
  "report_violation_type_normalized": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "report_violation_type_qids": [
    "Q355304",
    "Q16338046",
    "Q24336031",
    "Q124714",
    "Q39816",
    "Q35666",
    "Q104347069",
    "Q285451",
    "Q15324",
    "Q337567",
    "Q187971",
    "Q16500104",
    "Q63565252"
  ],
  "report_violation_type_raw": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "value": null,
  "value_current_2026": [
    "Q5484"
  ],
  "value_current_2026_descriptions_en": [
    "largest of the salt lake between Europe and Asia"
  ],
  "value_current_2026_labels_en": [
    "Caspian Sea"
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
    "description": "the body of water to which the watercourse drains",
    "label": "mouth of the watercourse"
  },
  "qid": {
    "description": "river delta",
    "label": "Volga Delta"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
            "Q104347069",
            "Q124714",
            "Q15324",
            "Q16338046",
            "Q16500104",
            "Q187971",
            "Q24336031",
            "Q285451",
            "Q337567",
            "Q355304",
            "Q35666",
            "Q39816",
            "Q63565252"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q21503252"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 6,
  "author": "Vlk",
  "before_constraint_count": 6,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
              "id": "Q104347069",
              "label_en": "lake system"
            },
            {
              "description_en": "terrestrial water source",
              "id": "Q124714",
              "label_en": "spring"
            },
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "river which only exists in fiction",
              "id": "Q16338046",
              "label_en": "fictional river"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "river that only exists in myth",
              "id": "Q24336031",
              "label_en": "mythical river"
            },
            {
              "description_en": "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
              "id": "Q285451",
              "label_en": "drainage system"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "any flowing body of water",
              "id": "Q355304",
              "label_en": "watercourse"
            },
            {
              "description_en": "large persistent body of ice",
              "id": "Q35666",
              "label_en": "glacier"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "small stream, i.e. a small natural watercourse",
              "id": "Q63565252",
              "label_en": "brook"
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
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "height of the item (geographical object) as measured relative to sea level",
              "id": "P2044",
              "label_en": "elevation above sea level"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            {
              "description_en": "specify if the stream confluence is a left bank or right bank tributary",
              "id": "P3871",
              "label_en": "tributary orientation"
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
              "description_en": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
              "id": "P613",
              "label_en": "OS grid reference"
            },
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "infrastructure that conveys sewage or surface runoff",
              "id": "Q156849",
              "label_en": "sewer network"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "meeting of two or more bodies of flowing water",
              "id": "Q723748",
              "label_en": "confluence"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "stream in Highland, Scotland, UK, tributary of the Allt Cuaich and of the Cuaich Aqueduct",
              "id": "Q112729719",
              "label_en": "Féith Chàm"
            },
            {
              "description_en": "canal section in Argyll and Bute, Scotland, UK, flows west into Loch Crinan at Crinan, and east into the canal section from Cairnbaan to Ardrishaig",
              "id": "Q56664718",
              "label_en": "Crinan Canal, section from Crinan to Cairnbaan"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
              "id": "Q104347069",
              "label_en": "lake system"
            },
            {
              "description_en": "terrestrial water source",
              "id": "Q124714",
              "label_en": "spring"
            },
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "river which only exists in fiction",
              "id": "Q16338046",
              "label_en": "fictional river"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "river that only exists in myth",
              "id": "Q24336031",
              "label_en": "mythical river"
            },
            {
              "description_en": "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
              "id": "Q285451",
              "label_en": "drainage system"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "any flowing body of water",
              "id": "Q355304",
              "label_en": "watercourse"
            },
            {
              "description_en": "large persistent body of ice",
              "id": "Q35666",
              "label_en": "glacier"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
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
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "height of the item (geographical object) as measured relative to sea level",
              "id": "P2044",
              "label_en": "elevation above sea level"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            {
              "description_en": "specify if the stream confluence is a left bank or right bank tributary",
              "id": "P3871",
              "label_en": "tributary orientation"
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
              "description_en": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
              "id": "P613",
              "label_en": "OS grid reference"
            },
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "infrastructure that conveys sewage or surface runoff",
              "id": "Q156849",
              "label_en": "sewer network"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "meeting of two or more bodies of flowing water",
              "id": "Q723748",
              "label_en": "confluence"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "stream in Highland, Scotland, UK, tributary of the Allt Cuaich and of the Cuaich Aqueduct",
              "id": "Q112729719",
              "label_en": "Féith Chàm"
            },
            {
              "description_en": "canal section in Argyll and Bute, Scotland, UK, flows west into Loch Crinan at Crinan, and east into the canal section from Cairnbaan to Ardrishaig",
              "id": "Q56664718",
              "label_en": "Crinan Canal, section from Crinan to Cairnbaan"
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
  "hash_after": "36188fe72951315c87372c2b0c88051dd7ea3646",
  "hash_before": "1eae5a89c1de92534f9fe810620210d2cac7c93f",
  "property_revision_id": 2442802452,
  "property_revision_prev": 2421662999,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q63565252"
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
            "Q104347069",
            "Q124714",
            "Q15324",
            "Q16338046",
            "Q16500104",
            "Q187971",
            "Q24336031",
            "Q285451",
            "Q337567",
            "Q355304",
            "Q35666",
            "Q39816"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q21503252"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "subject type constraint: class: lake system, spring, body of water, fictional river, fictional body of water, wadi, mythical river, drainage system, still waters, watercourse, glacier, valley, brook; relation: instance of",
      "allowed qualifiers constraint: property: located in the administrative territorial entity, end cause, subject named as, object named as, elevation above sea level, reason for deprecated rank, location, tributary orientation, start time, end time, OS grid reference, coordinate location, reason for preferred rank",
      "value-type constraint: class: body of water, sewer network, fictional body of water, wadi, still waters, valley, confluence; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "single-best-value constraint: exception to constraint: Féith Chàm, Crinan Canal, section from Crinan to Cairnbaan",
      "property scope constraint: property scope: as main value"
    ],
    "before": [
      "subject type constraint: class: lake system, spring, body of water, fictional river, fictional body of water, wadi, mythical river, drainage system, still waters, watercourse, glacier, valley; relation: instance of",
      "allowed qualifiers constraint: property: located in the administrative territorial entity, end cause, subject named as, object named as, elevation above sea level, reason for deprecated rank, location, tributary orientation, start time, end time, OS grid reference, coordinate location, reason for preferred rank",
      "value-type constraint: class: body of water, sewer network, fictional body of water, wadi, still waters, valley, confluence; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "single-best-value constraint: exception to constraint: Féith Chàm, Crinan Canal, section from Crinan to Cairnbaan",
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint"
  },
  {
    "property_ids": [
      "P2308",
      "P2309"
    ],
    "result": "RELAXATION_SET_EXPANSION",
    "step": "set_semantics"
  }
]
```

---

## 011. `reform_Q15712356_P1782_2250979466`

| Field | Value |
|---|---|
| qid | Q15712356 |
| property | P1782 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | TBOX::P1782::2250979466 |
| tbox_revision_key | TBOX::P1782::2250979466 |

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
| rationale | Constraint qualifiers compared with generic set semantics. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "RVA2869",
  "kind": "T_BOX",
  "property_revision_id": 2250979466,
  "property_revision_prev": 2250979461
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-09-23T07:58:17",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1782",
  "report_revision_new": 2251948633,
  "report_revision_old": 2251474443,
  "report_violation_type": "Item P|21",
  "report_violation_type_normalized": "Item P|21",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|21",
  "value": null,
  "value_current_2026": [
    "修园",
    "修園"
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
    "description": "name bestowed upon a person at adulthood in addition to one's given name, mostly in East Asia",
    "label": "courtesy name"
  },
  "qid": {
    "description": "Qing dynasty person CBDB = 81917",
    "label": "Chen Nianzu"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
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
    "label_en": "label in language constraint",
    "qid": "Q108139345"
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
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P424",
          "values": [
            "vi"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 10,
  "author": "RVA2869",
  "before_constraint_count": 9,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "ja"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "ko"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "vi"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "zh"
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
              "description_en": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
              "id": "P21",
              "label_en": "sex or gender"
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
              "description_en": "human being that only exists in fictional works",
              "id": "Q15632617",
              "label_en": "fictional human"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
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
              "description_en": "hanyu pinyin transliteration of a Mandarin Chinese text (usually to be used as a qualifier)",
              "id": "P1721",
              "label_en": "Hanyu Pinyin transliteration"
            },
            {
              "description_en": "the reading of a Japanese name in kana",
              "id": "P1814",
              "label_en": "name in kana"
            },
            {
              "description_en": "romanization of Korean developed by George M. McCune and Edwin O. Reischauer",
              "id": "P1942",
              "label_en": "McCune–Reischauer romanization"
            },
            {
              "description_en": "romanisation following the Revised Romanisation of the Korean language",
              "id": "P2001",
              "label_en": "Revised Romanization"
            },
            {
              "description_en": "romanized Japanese following the revised Hepburn romanization system",
              "id": "P2125",
              "label_en": "revised Hepburn romanization"
            },
            {
              "description_en": "alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface",
              "id": "P282",
              "label_en": "writing system"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "reading of Han character in Quốc Ngữ",
              "id": "P5625",
              "label_en": "Vietnamese reading"
            }
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
              "description_en": "alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface",
              "id": "P282",
              "label_en": "writing system"
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
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "ja"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "ko"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "zh"
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
              "description_en": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
              "id": "P21",
              "label_en": "sex or gender"
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
              "description_en": "human being that only exists in fictional works",
              "id": "Q15632617",
              "label_en": "fictional human"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
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
              "description_en": "hanyu pinyin transliteration of a Mandarin Chinese text (usually to be used as a qualifier)",
              "id": "P1721",
              "label_en": "Hanyu Pinyin transliteration"
            },
            {
              "description_en": "the reading of a Japanese name in kana",
              "id": "P1814",
              "label_en": "name in kana"
            },
            {
              "description_en": "romanization of Korean developed by George M. McCune and Edwin O. Reischauer",
              "id": "P1942",
              "label_en": "McCune–Reischauer romanization"
            },
            {
              "description_en": "romanisation following the Revised Romanisation of the Korean language",
              "id": "P2001",
              "label_en": "Revised Romanization"
            },
            {
              "description_en": "romanized Japanese following the revised Hepburn romanization system",
              "id": "P2125",
              "label_en": "revised Hepburn romanization"
            },
            {
              "description_en": "alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface",
              "id": "P282",
              "label_en": "writing system"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "reading of Han character in Quốc Ngữ",
              "id": "P5625",
              "label_en": "Vietnamese reading"
            }
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
              "description_en": "alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface",
              "id": "P282",
              "label_en": "writing system"
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
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "441c7855bef31cda6009fb200b499e96c6f260bb",
  "hash_before": "db66f155f284d28ced699111ae77f5acd63cae98",
  "property_revision_id": 2250979466,
  "property_revision_prev": 2250979461,
  "qualifier_value_changes": [
    {
      "added_values": [
        "vi"
      ],
      "constraint_qid": "Q108139345",
      "qualifier_property": "P424",
      "removed_values": [
        "zh"
      ],
      "same_qid_index": 2
    }
  ],
  "removed_constraint_entries": [],
  "rule_summaries_en": {
    "after": [
      "label in language constraint: Wikimedia language code: ja",
      "label in language constraint: Wikimedia language code: ko",
      "label in language constraint: Wikimedia language code: vi",
      "label in language constraint: Wikimedia language code: zh",
      "item-requires-statement constraint: property: sex or gender",
      "subject type constraint: class: fictional human, human; relation: instance of",
      "allowed qualifiers constraint: property: Hanyu Pinyin transliteration, name in kana, McCune–Reischauer romanization, Revised Romanization, revised Hepburn romanization, writing system, language of work or name, Vietnamese reading",
      "required qualifier constraint: property: writing system",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value"
    ],
    "before": [
      "label in language constraint: Wikimedia language code: ja",
      "label in language constraint: Wikimedia language code: ko",
      "label in language constraint: Wikimedia language code: zh",
      "item-requires-statement constraint: property: sex or gender",
      "subject type constraint: class: fictional human, human; relation: instance of",
      "allowed qualifiers constraint: property: Hanyu Pinyin transliteration, name in kana, McCune–Reischauer romanization, Revised Romanization, revised Hepburn romanization, writing system, language of work or name, Vietnamese reading",
      "required qualifier constraint: property: writing system",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|21"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---

## 012. `reform_Q16688035_P2517_2442912347`

| Field | Value |
|---|---|
| qid | Q16688035 |
| property | P2517 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
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
| rationale | Constraint qualifiers compared with generic set semantics. |
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
  "report_violation_type": "Type Q|216353, Q|618779, Q|38033430, Q|107655869, Q|107467117",
  "report_violation_type_descriptions_en": [
    "qualified name, rank, or other indication of a class or role given to or inherited by a person, often affixed to a person's name",
    "something given to a person or a group of people to recognize their merit or excellence",
    "class of award (order, medal, etc.)",
    "award group or series",
    "type of award classified by stylistic, thematic or technical criteria"
  ],
  "report_violation_type_labels_en": [
    "title",
    "award",
    "class of award",
    "group of awards",
    "type of award"
  ],
  "report_violation_type_normalized": "Type Q|216353, Q|618779, Q|38033430, Q|107655869, Q|107467117",
  "report_violation_type_qids": [
    "Q216353",
    "Q618779",
    "Q38033430",
    "Q107655869",
    "Q107467117"
  ],
  "report_violation_type_raw": "Type Q|216353, Q|618779, Q|38033430, Q|107655869, Q|107467117",
  "value": null,
  "value_current_2026": [
    "Q32298354"
  ],
  "value_current_2026_descriptions_en": [
    "Wikimedia category"
  ],
  "value_current_2026_labels_en": [
    "Категория:Лауреаты премии имени И. М. Сеченова"
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
    "description": null,
    "label": "Премия имени И. М. Сеченова"
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|216353, Q|618779, Q|38033430, Q|107655869, Q|107467117"
  },
  {
    "result": "Q19474404",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---

## 013. `reform_Q17243_P4969_2435927232`

| Field | Value |
|---|---|
| qid | Q17243 |
| property | P4969 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | TBOX::P4969::2435927232 |
| tbox_revision_key | TBOX::P4969::2435927232 |

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
| rationale | Constraint qualifiers compared with generic set semantics. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Trade",
  "kind": "T_BOX",
  "property_revision_id": 2435927232,
  "property_revision_prev": 2435926155
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-05T11:37:12",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4969",
  "report_revision_new": 2438318580,
  "report_revision_old": 2437937432,
  "report_violation_type": "Inverse",
  "report_violation_type_normalized": "Inverse",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Inverse",
  "value": null,
  "value_current_2026": [
    "Q1750894",
    "Q72345293",
    "Q4060295",
    "Q133696346"
  ],
  "value_current_2026_descriptions_en": [
    "video game series",
    "board game",
    "economic board game",
    "2014 boardgame"
  ],
  "value_current_2026_labels_en": [
    "Monopoly",
    "Banco Imobiliário",
    "Actioner",
    "ABBA Monopoly"
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
    "description": "new work of art (film, book, software, etc.) derived from major part of this work",
    "label": "derivative work"
  },
  "qid": {
    "description": "economics-themed board game",
    "label": "Monopoly"
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
    "label_en": "inverse constraint",
    "qid": "Q21510855"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502838",
      "qualifiers": [
        {
          "property_id": "P2305",
          "values": [
            "Q136747113",
            "Q136832029",
            "Q3331189",
            "Q57933693"
          ]
        },
        {
          "property_id": "P2306",
          "values": [
            "P31"
          ]
        },
        {
          "property_id": "P2316",
          "values": [
            "Q21502408"
          ]
        },
        {
          "property_id": "P6607",
          "values": [
            "works should have this property instead of editions, use in Q7725634@en"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 6,
  "author": "Trade",
  "before_constraint_count": 6,
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
              "description_en": "Japanese style of animation",
              "id": "Q1107",
              "label_en": "anime"
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
          ],
          "P9729": [
            {
              "description_en": "series of light novels published in Japan",
              "id": "Q104213567",
              "label_en": "light novel series"
            },
            {
              "description_en": "series of comics employing Japanese stylistic conventions that are that are formally identified together",
              "id": "Q21198342",
              "label_en": "manga series"
            },
            {
              "description_en": "Japanese animated television series",
              "id": "Q63952888",
              "label_en": "anime television series"
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
              "description_en": "edition of a light novel",
              "id": "Q136747113",
              "label_en": "light novel edition"
            },
            {
              "description_en": "edition of a manga",
              "id": "Q136832029",
              "label_en": "manga edition"
            },
            {
              "description_en": "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
              "id": "Q3331189",
              "label_en": "version, edition or translation"
            },
            {
              "description_en": "edition of a book",
              "id": "Q57933693",
              "label_en": "book edition"
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
          ],
          "P6607": [
            {
              "value": "works should have this property instead of editions, use in Q7725634@en"
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
              "description_en": "sequence of images that give the impression of movement, stored on film stock",
              "id": "Q11424",
              "label_en": "film"
            },
            {
              "description_en": "סוג קבוצת יצירות",
              "id": "Q116779428",
              "label_en": "group of works often treated as a singular work"
            },
            {
              "description_en": "specific weapon design, pattern, or version of which all examples are essentially identical",
              "id": "Q15142894",
              "label_en": "weapon model"
            },
            {
              "description_en": "anything created by humans (either material or mental)",
              "id": "Q16686448",
              "label_en": "artificial object"
            },
            {
              "description_en": "entity whose existence is possible, but not proven",
              "id": "Q18706315",
              "label_en": "hypothetical entity"
            },
            {
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
            },
            {
              "description_en": "production of the performing arts, consisting of a series of quasi-identical performances of the same performance work",
              "id": "Q43099500",
              "label_en": "performing arts production"
            },
            {
              "description_en": "recurring, self-sufficient plot or motif grouping, unit of classification in the Aarne–Thompson classification systems",
              "id": "Q47451145",
              "label_en": "tale type"
            },
            {
              "description_en": "group of related ammunition cartridges which share basic design elements",
              "id": "Q48708989",
              "label_en": "cartridge family"
            },
            {
              "description_en": "plant or grouping of plants selected for desirable characteristics",
              "id": "Q4886",
              "label_en": "cultivar"
            },
            {
              "description_en": "fictional human or non-human character in a narrative work of art",
              "id": "Q95074",
              "label_en": "character"
            }
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item has to refer back to this item with the given inverse property",
          "id": "Q21510855",
          "label_en": "inverse constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the work(s) or inputs used as the basis for subject item; for fictional analog use P1074",
              "id": "P144",
              "label_en": "based on"
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
              "id": "Q54828449",
              "label_en": "as qualifier"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "Japanese style of animation",
              "id": "Q1107",
              "label_en": "anime"
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
          ],
          "P9729": [
            {
              "description_en": "series of light novels published in Japan",
              "id": "Q104213567",
              "label_en": "light novel series"
            },
            {
              "description_en": "series of comics employing Japanese stylistic conventions that are that are formally identified together",
              "id": "Q21198342",
              "label_en": "manga series"
            },
            {
              "description_en": "Japanese animated television series",
              "id": "Q63952888",
              "label_en": "anime television series"
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
              "description_en": "edition of a manga",
              "id": "Q136832029",
              "label_en": "manga edition"
            },
            {
              "description_en": "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
              "id": "Q3331189",
              "label_en": "version, edition or translation"
            },
            {
              "description_en": "edition of a book",
              "id": "Q57933693",
              "label_en": "book edition"
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
          ],
          "P6607": [
            {
              "value": "works should have this property instead of editions, use in Q7725634@en"
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
              "description_en": "sequence of images that give the impression of movement, stored on film stock",
              "id": "Q11424",
              "label_en": "film"
            },
            {
              "description_en": "סוג קבוצת יצירות",
              "id": "Q116779428",
              "label_en": "group of works often treated as a singular work"
            },
            {
              "description_en": "specific weapon design, pattern, or version of which all examples are essentially identical",
              "id": "Q15142894",
              "label_en": "weapon model"
            },
            {
              "description_en": "anything created by humans (either material or mental)",
              "id": "Q16686448",
              "label_en": "artificial object"
            },
            {
              "description_en": "entity whose existence is possible, but not proven",
              "id": "Q18706315",
              "label_en": "hypothetical entity"
            },
            {
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
            },
            {
              "description_en": "production of the performing arts, consisting of a series of quasi-identical performances of the same performance work",
              "id": "Q43099500",
              "label_en": "performing arts production"
            },
            {
              "description_en": "recurring, self-sufficient plot or motif grouping, unit of classification in the Aarne–Thompson classification systems",
              "id": "Q47451145",
              "label_en": "tale type"
            },
            {
              "description_en": "group of related ammunition cartridges which share basic design elements",
              "id": "Q48708989",
              "label_en": "cartridge family"
            },
            {
              "description_en": "plant or grouping of plants selected for desirable characteristics",
              "id": "Q4886",
              "label_en": "cultivar"
            },
            {
              "description_en": "fictional human or non-human character in a narrative work of art",
              "id": "Q95074",
              "label_en": "character"
            }
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item has to refer back to this item with the given inverse property",
          "id": "Q21510855",
          "label_en": "inverse constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the work(s) or inputs used as the basis for subject item; for fictional analog use P1074",
              "id": "P144",
              "label_en": "based on"
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
              "id": "Q54828449",
              "label_en": "as qualifier"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "299ae3d11d3719afbe8480654eb40a568a11f3df",
  "hash_before": "068118b0b02e8854ee54a1f3bb75df62a2e858cc",
  "property_revision_id": 2435927232,
  "property_revision_prev": 2435926155,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q136747113"
      ],
      "constraint_qid": "Q21502838",
      "qualifier_property": "P2305",
      "removed_values": [],
      "same_qid_index": 1
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502838",
      "qualifiers": [
        {
          "property_id": "P2305",
          "values": [
            "Q136832029",
            "Q3331189",
            "Q57933693"
          ]
        },
        {
          "property_id": "P2306",
          "values": [
            "P31"
          ]
        },
        {
          "property_id": "P2316",
          "values": [
            "Q21502408"
          ]
        },
        {
          "property_id": "P6607",
          "values": [
            "works should have this property instead of editions, use in Q7725634@en"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "conflicts-with constraint: item of property constraint: anime, light novel, manga; property: instance of; replacement value: light novel series, manga series, anime television series",
      "conflicts-with constraint: item of property constraint: light novel edition, manga edition, version, edition or translation, book edition; property: instance of; constraint status: mandatory constraint; constraint clarification: works should have this property instead of editions, use in Q7725634@en",
      "subject type constraint: class: film, group of works often treated as a singular work, weapon model, artificial object, hypothetical entity, work, performing arts production, tale type, cartridge family, cultivar, character; relation: instance or subclass of",
      "inverse constraint: property: based on",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as qualifier"
    ],
    "before": [
      "conflicts-with constraint: item of property constraint: anime, light novel, manga; property: instance of; replacement value: light novel series, manga series, anime television series",
      "conflicts-with constraint: item of property constraint: manga edition, version, edition or translation, book edition; property: instance of; constraint status: mandatory constraint; constraint clarification: works should have this property instead of editions, use in Q7725634@en",
      "subject type constraint: class: film, group of works often treated as a singular work, weapon model, artificial object, hypothetical entity, work, performing arts production, tale type, cartridge family, cultivar, character; relation: instance or subclass of",
      "inverse constraint: property: based on",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as qualifier"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Inverse"
  },
  {
    "result": "Q21502838",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---

## 014. `reform_Q20689183_P1782_2250979466`

| Field | Value |
|---|---|
| qid | Q20689183 |
| property | P1782 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | TBOX::P1782::2250979466 |
| tbox_revision_key | TBOX::P1782::2250979466 |

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
| rationale | Constraint qualifiers compared with generic set semantics. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "RVA2869",
  "kind": "T_BOX",
  "property_revision_id": 2250979466,
  "property_revision_prev": 2250979461
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-09-23T07:58:17",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1782",
  "report_revision_new": 2251948633,
  "report_revision_old": 2251474443,
  "report_violation_type": "Label in ja language",
  "report_violation_type_normalized": "Label in ja language",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Label in ja language",
  "value": null,
  "value_current_2026": [
    "大聲"
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
    "description": "name bestowed upon a person at adulthood in addition to one's given name, mostly in East Asia",
    "label": "courtesy name"
  },
  "qid": {
    "description": "Ming dynasty person CBDB = 198664",
    "label": "Zhong Zhen"
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
    "label_en": "item-requires-statement constraint",
    "qid": "Q21503247"
  },
  {
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
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
    "label_en": "label in language constraint",
    "qid": "Q108139345"
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
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P424",
          "values": [
            "vi"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 10,
  "author": "RVA2869",
  "before_constraint_count": 9,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "ja"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "ko"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "vi"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "zh"
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
              "description_en": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
              "id": "P21",
              "label_en": "sex or gender"
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
              "description_en": "human being that only exists in fictional works",
              "id": "Q15632617",
              "label_en": "fictional human"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
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
              "description_en": "hanyu pinyin transliteration of a Mandarin Chinese text (usually to be used as a qualifier)",
              "id": "P1721",
              "label_en": "Hanyu Pinyin transliteration"
            },
            {
              "description_en": "the reading of a Japanese name in kana",
              "id": "P1814",
              "label_en": "name in kana"
            },
            {
              "description_en": "romanization of Korean developed by George M. McCune and Edwin O. Reischauer",
              "id": "P1942",
              "label_en": "McCune–Reischauer romanization"
            },
            {
              "description_en": "romanisation following the Revised Romanisation of the Korean language",
              "id": "P2001",
              "label_en": "Revised Romanization"
            },
            {
              "description_en": "romanized Japanese following the revised Hepburn romanization system",
              "id": "P2125",
              "label_en": "revised Hepburn romanization"
            },
            {
              "description_en": "alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface",
              "id": "P282",
              "label_en": "writing system"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "reading of Han character in Quốc Ngữ",
              "id": "P5625",
              "label_en": "Vietnamese reading"
            }
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
              "description_en": "alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface",
              "id": "P282",
              "label_en": "writing system"
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
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "ja"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "ko"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "zh"
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
              "description_en": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
              "id": "P21",
              "label_en": "sex or gender"
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
              "description_en": "human being that only exists in fictional works",
              "id": "Q15632617",
              "label_en": "fictional human"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
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
              "description_en": "hanyu pinyin transliteration of a Mandarin Chinese text (usually to be used as a qualifier)",
              "id": "P1721",
              "label_en": "Hanyu Pinyin transliteration"
            },
            {
              "description_en": "the reading of a Japanese name in kana",
              "id": "P1814",
              "label_en": "name in kana"
            },
            {
              "description_en": "romanization of Korean developed by George M. McCune and Edwin O. Reischauer",
              "id": "P1942",
              "label_en": "McCune–Reischauer romanization"
            },
            {
              "description_en": "romanisation following the Revised Romanisation of the Korean language",
              "id": "P2001",
              "label_en": "Revised Romanization"
            },
            {
              "description_en": "romanized Japanese following the revised Hepburn romanization system",
              "id": "P2125",
              "label_en": "revised Hepburn romanization"
            },
            {
              "description_en": "alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface",
              "id": "P282",
              "label_en": "writing system"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "reading of Han character in Quốc Ngữ",
              "id": "P5625",
              "label_en": "Vietnamese reading"
            }
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
              "description_en": "alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface",
              "id": "P282",
              "label_en": "writing system"
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
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "441c7855bef31cda6009fb200b499e96c6f260bb",
  "hash_before": "db66f155f284d28ced699111ae77f5acd63cae98",
  "property_revision_id": 2250979466,
  "property_revision_prev": 2250979461,
  "qualifier_value_changes": [
    {
      "added_values": [
        "vi"
      ],
      "constraint_qid": "Q108139345",
      "qualifier_property": "P424",
      "removed_values": [
        "zh"
      ],
      "same_qid_index": 2
    }
  ],
  "removed_constraint_entries": [],
  "rule_summaries_en": {
    "after": [
      "label in language constraint: Wikimedia language code: ja",
      "label in language constraint: Wikimedia language code: ko",
      "label in language constraint: Wikimedia language code: vi",
      "label in language constraint: Wikimedia language code: zh",
      "item-requires-statement constraint: property: sex or gender",
      "subject type constraint: class: fictional human, human; relation: instance of",
      "allowed qualifiers constraint: property: Hanyu Pinyin transliteration, name in kana, McCune–Reischauer romanization, Revised Romanization, revised Hepburn romanization, writing system, language of work or name, Vietnamese reading",
      "required qualifier constraint: property: writing system",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value"
    ],
    "before": [
      "label in language constraint: Wikimedia language code: ja",
      "label in language constraint: Wikimedia language code: ko",
      "label in language constraint: Wikimedia language code: zh",
      "item-requires-statement constraint: property: sex or gender",
      "subject type constraint: class: fictional human, human; relation: instance of",
      "allowed qualifiers constraint: property: Hanyu Pinyin transliteration, name in kana, McCune–Reischauer romanization, Revised Romanization, revised Hepburn romanization, writing system, language of work or name, Vietnamese reading",
      "required qualifier constraint: property: writing system",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Label in ja language"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---

## 015. `reform_Q23871475_P403_2442802452`

| Field | Value |
|---|---|
| qid | Q23871475 |
| property | P403 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | TBOX::P403::2442802452 |
| tbox_revision_key | TBOX::P403::2442802452 |

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
| rationale | Type/value-type constraint classes and relations compared using P2308/P2309. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Vlk",
  "kind": "T_BOX",
  "property_revision_id": 2442802452,
  "property_revision_prev": 2421662999
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-19T10:56:29",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P403",
  "report_revision_new": 2444036897,
  "report_revision_old": 2443833509,
  "report_violation_type": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "report_violation_type_descriptions_en": [
    "any flowing body of water",
    "river which only exists in fiction",
    "river that only exists in myth",
    "terrestrial water source",
    "low area between hills, often with a river running through it",
    "large persistent body of ice",
    "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
    "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
    "any significant accumulation of water, generally on a planet's surface",
    "body of water without noticeable current",
    "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
    "waterbody only existing in fiction",
    "small stream, i.e. a small natural watercourse"
  ],
  "report_violation_type_labels_en": [
    "watercourse",
    "fictional river",
    "mythical river",
    "spring",
    "valley",
    "glacier",
    "lake system",
    "drainage system",
    "body of water",
    "still waters",
    "wadi",
    "fictional body of water",
    "brook"
  ],
  "report_violation_type_normalized": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "report_violation_type_qids": [
    "Q355304",
    "Q16338046",
    "Q24336031",
    "Q124714",
    "Q39816",
    "Q35666",
    "Q104347069",
    "Q285451",
    "Q15324",
    "Q337567",
    "Q187971",
    "Q16500104",
    "Q63565252"
  ],
  "report_violation_type_raw": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "value": null,
  "value_current_2026": [
    "Q98"
  ],
  "value_current_2026_descriptions_en": [
    "ocean between Asia, Australia and the Americas"
  ],
  "value_current_2026_labels_en": [
    "Pacific Ocean"
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
    "description": "the body of water to which the watercourse drains",
    "label": "mouth of the watercourse"
  },
  "qid": {
    "description": "desembocadura de ríos en Ecuador",
    "label": "Boca del Río Santiago"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
            "Q104347069",
            "Q124714",
            "Q15324",
            "Q16338046",
            "Q16500104",
            "Q187971",
            "Q24336031",
            "Q285451",
            "Q337567",
            "Q355304",
            "Q35666",
            "Q39816",
            "Q63565252"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q21503252"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 6,
  "author": "Vlk",
  "before_constraint_count": 6,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
              "id": "Q104347069",
              "label_en": "lake system"
            },
            {
              "description_en": "terrestrial water source",
              "id": "Q124714",
              "label_en": "spring"
            },
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "river which only exists in fiction",
              "id": "Q16338046",
              "label_en": "fictional river"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "river that only exists in myth",
              "id": "Q24336031",
              "label_en": "mythical river"
            },
            {
              "description_en": "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
              "id": "Q285451",
              "label_en": "drainage system"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "any flowing body of water",
              "id": "Q355304",
              "label_en": "watercourse"
            },
            {
              "description_en": "large persistent body of ice",
              "id": "Q35666",
              "label_en": "glacier"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "small stream, i.e. a small natural watercourse",
              "id": "Q63565252",
              "label_en": "brook"
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
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "height of the item (geographical object) as measured relative to sea level",
              "id": "P2044",
              "label_en": "elevation above sea level"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            {
              "description_en": "specify if the stream confluence is a left bank or right bank tributary",
              "id": "P3871",
              "label_en": "tributary orientation"
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
              "description_en": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
              "id": "P613",
              "label_en": "OS grid reference"
            },
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "infrastructure that conveys sewage or surface runoff",
              "id": "Q156849",
              "label_en": "sewer network"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "meeting of two or more bodies of flowing water",
              "id": "Q723748",
              "label_en": "confluence"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "stream in Highland, Scotland, UK, tributary of the Allt Cuaich and of the Cuaich Aqueduct",
              "id": "Q112729719",
              "label_en": "Féith Chàm"
            },
            {
              "description_en": "canal section in Argyll and Bute, Scotland, UK, flows west into Loch Crinan at Crinan, and east into the canal section from Cairnbaan to Ardrishaig",
              "id": "Q56664718",
              "label_en": "Crinan Canal, section from Crinan to Cairnbaan"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
              "id": "Q104347069",
              "label_en": "lake system"
            },
            {
              "description_en": "terrestrial water source",
              "id": "Q124714",
              "label_en": "spring"
            },
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "river which only exists in fiction",
              "id": "Q16338046",
              "label_en": "fictional river"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "river that only exists in myth",
              "id": "Q24336031",
              "label_en": "mythical river"
            },
            {
              "description_en": "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
              "id": "Q285451",
              "label_en": "drainage system"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "any flowing body of water",
              "id": "Q355304",
              "label_en": "watercourse"
            },
            {
              "description_en": "large persistent body of ice",
              "id": "Q35666",
              "label_en": "glacier"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
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
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "height of the item (geographical object) as measured relative to sea level",
              "id": "P2044",
              "label_en": "elevation above sea level"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            {
              "description_en": "specify if the stream confluence is a left bank or right bank tributary",
              "id": "P3871",
              "label_en": "tributary orientation"
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
              "description_en": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
              "id": "P613",
              "label_en": "OS grid reference"
            },
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "infrastructure that conveys sewage or surface runoff",
              "id": "Q156849",
              "label_en": "sewer network"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "meeting of two or more bodies of flowing water",
              "id": "Q723748",
              "label_en": "confluence"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "stream in Highland, Scotland, UK, tributary of the Allt Cuaich and of the Cuaich Aqueduct",
              "id": "Q112729719",
              "label_en": "Féith Chàm"
            },
            {
              "description_en": "canal section in Argyll and Bute, Scotland, UK, flows west into Loch Crinan at Crinan, and east into the canal section from Cairnbaan to Ardrishaig",
              "id": "Q56664718",
              "label_en": "Crinan Canal, section from Crinan to Cairnbaan"
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
  "hash_after": "36188fe72951315c87372c2b0c88051dd7ea3646",
  "hash_before": "1eae5a89c1de92534f9fe810620210d2cac7c93f",
  "property_revision_id": 2442802452,
  "property_revision_prev": 2421662999,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q63565252"
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
            "Q104347069",
            "Q124714",
            "Q15324",
            "Q16338046",
            "Q16500104",
            "Q187971",
            "Q24336031",
            "Q285451",
            "Q337567",
            "Q355304",
            "Q35666",
            "Q39816"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q21503252"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "subject type constraint: class: lake system, spring, body of water, fictional river, fictional body of water, wadi, mythical river, drainage system, still waters, watercourse, glacier, valley, brook; relation: instance of",
      "allowed qualifiers constraint: property: located in the administrative territorial entity, end cause, subject named as, object named as, elevation above sea level, reason for deprecated rank, location, tributary orientation, start time, end time, OS grid reference, coordinate location, reason for preferred rank",
      "value-type constraint: class: body of water, sewer network, fictional body of water, wadi, still waters, valley, confluence; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "single-best-value constraint: exception to constraint: Féith Chàm, Crinan Canal, section from Crinan to Cairnbaan",
      "property scope constraint: property scope: as main value"
    ],
    "before": [
      "subject type constraint: class: lake system, spring, body of water, fictional river, fictional body of water, wadi, mythical river, drainage system, still waters, watercourse, glacier, valley; relation: instance of",
      "allowed qualifiers constraint: property: located in the administrative territorial entity, end cause, subject named as, object named as, elevation above sea level, reason for deprecated rank, location, tributary orientation, start time, end time, OS grid reference, coordinate location, reason for preferred rank",
      "value-type constraint: class: body of water, sewer network, fictional body of water, wadi, still waters, valley, confluence; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "single-best-value constraint: exception to constraint: Féith Chàm, Crinan Canal, section from Crinan to Cairnbaan",
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint"
  },
  {
    "property_ids": [
      "P2308",
      "P2309"
    ],
    "result": "RELAXATION_SET_EXPANSION",
    "step": "set_semantics"
  }
]
```

---

## 016. `reform_Q2823688_P4529_2172698999`

| Field | Value |
|---|---|
| qid | Q2823688 |
| property | P4529 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | TBOX::P4529::2172698999 |
| tbox_revision_key | TBOX::P4529::2172698999 |

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
| rationale | Constraint qualifiers compared with generic set semantics. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Kethyga",
  "kind": "T_BOX",
  "property_revision_id": 2172698999,
  "property_revision_prev": 2146311003
}
```

### Violation Context

```json
{
  "report_fix_date": "2024-06-06T07:48:42",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4529",
  "report_revision_new": 2173139515,
  "report_revision_old": 2172597099,
  "report_violation_type": "Label in zh language",
  "report_violation_type_normalized": "Label in zh language",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Label in zh language",
  "value": null,
  "value_current_2026": [
    "6061437"
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
    "description": "identifier for a film (movie) or TV series at the website Douban",
    "label": "Douban film ID"
  },
  "qid": {
    "description": "2012 film directed by Robert Heath",
    "label": "Truth or Dare"
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
  },
  {
    "label_en": "label in language constraint",
    "qid": "Q108139345"
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P2316",
          "values": [
            "Q62026391"
          ]
        },
        {
          "property_id": "P424",
          "values": [
            "zh",
            "zh-cn"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 8,
  "author": "Kethyga",
  "before_constraint_count": 8,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ],
          "P424": [
            {
              "value": "zh"
            },
            {
              "value": "zh-cn"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
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
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "season of a series (television, web, podcast)",
              "id": "P4908",
              "label_en": "season"
            },
            {
              "description_en": "time an entity begins to exist or a statement starts being valid",
              "id": "P580",
              "label_en": "start time"
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
              "value": "[1-9]\\d*|"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "sequence of images that give the impression of movement, stored on film stock",
              "id": "Q11424",
              "label_en": "film"
            },
            {
              "description_en": "method of creating moving pictures",
              "id": "Q11425",
              "label_en": "animation"
            },
            {
              "description_en": "episode of a web television series",
              "id": "Q1464125",
              "label_en": "web series episode"
            },
            {
              "description_en": "segment of audiovisual content intended for broadcast and streaming on television",
              "id": "Q15416",
              "label_en": "television program"
            },
            {
              "description_en": "creative work possessing both a sound and a visual component",
              "id": "Q2431196",
              "label_en": "audiovisual work"
            },
            {
              "description_en": "set of episodes produced for a television series",
              "id": "Q3464665",
              "label_en": "television series season"
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
              "description_en": "position of an item in its parent series (most frequently a 1-based index), generally to be used as a qualifier (different from \"rank\" defined as a class, and from \"ranking\" defined as a property for evaluating a quality)",
              "id": "P1545",
              "label_en": "series ordinal"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "season of a series (television, web, podcast)",
              "id": "P4908",
              "label_en": "season"
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
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ],
          "P424": [
            {
              "value": "zh"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
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
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "season of a series (television, web, podcast)",
              "id": "P4908",
              "label_en": "season"
            },
            {
              "description_en": "time an entity begins to exist or a statement starts being valid",
              "id": "P580",
              "label_en": "start time"
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
              "value": "[1-9]\\d*|"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "sequence of images that give the impression of movement, stored on film stock",
              "id": "Q11424",
              "label_en": "film"
            },
            {
              "description_en": "method of creating moving pictures",
              "id": "Q11425",
              "label_en": "animation"
            },
            {
              "description_en": "episode of a web television series",
              "id": "Q1464125",
              "label_en": "web series episode"
            },
            {
              "description_en": "segment of audiovisual content intended for broadcast and streaming on television",
              "id": "Q15416",
              "label_en": "television program"
            },
            {
              "description_en": "creative work possessing both a sound and a visual component",
              "id": "Q2431196",
              "label_en": "audiovisual work"
            },
            {
              "description_en": "set of episodes produced for a television series",
              "id": "Q3464665",
              "label_en": "television series season"
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
              "description_en": "position of an item in its parent series (most frequently a 1-based index), generally to be used as a qualifier (different from \"rank\" defined as a class, and from \"ranking\" defined as a property for evaluating a quality)",
              "id": "P1545",
              "label_en": "series ordinal"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
              "id": "P407",
              "label_en": "language of work or name"
            },
            {
              "description_en": "season of a series (television, web, podcast)",
              "id": "P4908",
              "label_en": "season"
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
  "hash_after": "a06f4e4b0c9b11a2ff0c21204efa3a4140691084",
  "hash_before": "7fdb89b49696dbda0be5e3ba1f0c873d49c9279e",
  "property_revision_id": 2172698999,
  "property_revision_prev": 2146311003,
  "qualifier_value_changes": [
    {
      "added_values": [
        "zh-cn"
      ],
      "constraint_qid": "Q108139345",
      "qualifier_property": "P424",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P2316",
          "values": [
            "Q62026391"
          ]
        },
        {
          "property_id": "P424",
          "values": [
            "zh"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "label in language constraint: constraint status: suggestion constraint; Wikimedia language code: zh, zh-cn",
      "single-value constraint: separator: series ordinal, subject named as, language of work or name, season, start time",
      "format constraint: format as a regular expression: [1-9]\\d*|; constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "subject type constraint: class: film, animation, web series episode, television program, audiovisual work, television series season; relation: instance of",
      "allowed qualifiers constraint: property: series ordinal, subject named as, language of work or name, season, start time, end time",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ],
    "before": [
      "label in language constraint: constraint status: suggestion constraint; Wikimedia language code: zh",
      "single-value constraint: separator: series ordinal, subject named as, language of work or name, season, start time",
      "format constraint: format as a regular expression: [1-9]\\d*|; constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "subject type constraint: class: film, animation, web series episode, television program, audiovisual work, television series season; relation: instance of",
      "allowed qualifiers constraint: property: series ordinal, subject named as, language of work or name, season, start time, end time",
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Label in zh language"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---

## 017. `reform_Q29117147_P481_2317303689`

| Field | Value |
|---|---|
| qid | Q29117147 |
| property | P481 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | TBOX::P481::2317303689 |
| tbox_revision_key | TBOX::P481::2317303689 |

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
| rationale | Constraint qualifiers compared with generic set semantics. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Bob08",
  "kind": "T_BOX",
  "property_revision_id": 2317303689,
  "property_revision_prev": 2126294359
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-02-27T13:51:16",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P481",
  "report_revision_new": 2317728623,
  "report_revision_old": 2317281348,
  "report_violation_type": "Item P|180",
  "report_violation_type_normalized": "Item P|180",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|180",
  "report_violation_types": [
    "Item P|180",
    "Item P|170",
    "Item P|217",
    "Item P|6216",
    "Item P|2049",
    "Item P|571",
    "Item P|2048",
    "Item P|195",
    "Type Q|8205328, Q|16887380",
    "Item P|136"
  ],
  "value": null,
  "value_current_2026": [
    "PM33000121"
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
    "description": "identifier in the Palissy database of moveable objects of French cultural heritage",
    "label": "Palissy ID"
  },
  "qid": {
    "description": "enluminure monument historique (PM33000121) située à Bordeaux (Gironde, France)",
    "label": "Joachim et Nabuchodonosor"
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
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P424",
          "values": [
            "fr",
            "mul"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 22,
  "author": "Bob08",
  "before_constraint_count": 22,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "fr"
            },
            {
              "value": "mul"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[PEI][M]\\d[0-9AB]\\d\\d\\d\\d\\d\\d"
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
        "parameters": {
          "P2303": [
            {
              "description_en": "fragment de peinture romane déposée",
              "id": "Q28319846",
              "label_en": "lion de saint Marc de l'église Saint-Sauveur de Casesnoves"
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
              "description_en": "number of people inhabiting the place; number of people of subject",
              "id": "P1082",
              "label_en": "population"
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
              "description_en": "status of an item that is designated as intangible heritage",
              "id": "P3259",
              "label_en": "intangible cultural heritage status"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "country in Western Europe and other continents (through its overseas territories in America, Africa and Oceania)",
              "id": "Q142",
              "label_en": "France"
            }
          ],
          "P2306": [
            {
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "French object classified as a Historical Monument by the French State",
              "id": "Q61058403",
              "label_en": "object classified as a historical monument"
            },
            {
              "description_en": "French object listed as a Historical Monument by the French State",
              "id": "Q61058419",
              "label_en": "object listed as historical monument"
            },
            {
              "description_en": "patrimoine inventorié français",
              "id": "Q86830939",
              "label_en": "objets non protégés de l'I.G.P.C."
            }
          ],
          "P2306": [
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
              "id": "P136",
              "label_en": "genre"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "maker of this creative work or other object (where no more specific property exists)",
              "id": "P170",
              "label_en": "creator"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "entity visually depicted in an image, literarily described in a work, or otherwise incorporated into an audiovisual or other medium; see also P921, 'main subject'",
              "id": "P180",
              "label_en": "depicts"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "material the subject or the object is made of or derived from (do not confuse with P10672 which is used for processes)",
              "id": "P186",
              "label_en": "made from material"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "art, museum, archival, or bibliographic collection of which the subject is part (item is in the collection of X)",
              "id": "P195",
              "label_en": "collection"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "vertical length of an entity",
              "id": "P2048",
              "label_en": "height"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "width of an object",
              "id": "P2049",
              "label_en": "width"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a physical object or a set of physical objects in a collection",
              "id": "P217",
              "label_en": "inventory number"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "time when an entity begins to exist; for date of official opening use P1619",
              "id": "P571",
              "label_en": "inception"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "copyright status for intellectual creations like works of art, publications, software, etc.",
              "id": "P6216",
              "label_en": "copyright status"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "well-defined, enumerable collection of discrete entities that form a collective whole",
              "id": "Q16887380",
              "label_en": "group"
            },
            {
              "description_en": "physical object made or shaped by humans",
              "id": "Q8205328",
              "label_en": "artificial physical object"
            }
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
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "fr"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[PEI][M]\\d[0-9AB]\\d\\d\\d\\d\\d\\d"
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
        "parameters": {
          "P2303": [
            {
              "description_en": "fragment de peinture romane déposée",
              "id": "Q28319846",
              "label_en": "lion de saint Marc de l'église Saint-Sauveur de Casesnoves"
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
              "description_en": "number of people inhabiting the place; number of people of subject",
              "id": "P1082",
              "label_en": "population"
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
              "description_en": "status of an item that is designated as intangible heritage",
              "id": "P3259",
              "label_en": "intangible cultural heritage status"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "country in Western Europe and other continents (through its overseas territories in America, Africa and Oceania)",
              "id": "Q142",
              "label_en": "France"
            }
          ],
          "P2306": [
            {
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "French object classified as a Historical Monument by the French State",
              "id": "Q61058403",
              "label_en": "object classified as a historical monument"
            },
            {
              "description_en": "French object listed as a Historical Monument by the French State",
              "id": "Q61058419",
              "label_en": "object listed as historical monument"
            },
            {
              "description_en": "patrimoine inventorié français",
              "id": "Q86830939",
              "label_en": "objets non protégés de l'I.G.P.C."
            }
          ],
          "P2306": [
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
              "id": "P136",
              "label_en": "genre"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "maker of this creative work or other object (where no more specific property exists)",
              "id": "P170",
              "label_en": "creator"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "entity visually depicted in an image, literarily described in a work, or otherwise incorporated into an audiovisual or other medium; see also P921, 'main subject'",
              "id": "P180",
              "label_en": "depicts"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "material the subject or the object is made of or derived from (do not confuse with P10672 which is used for processes)",
              "id": "P186",
              "label_en": "made from material"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "art, museum, archival, or bibliographic collection of which the subject is part (item is in the collection of X)",
              "id": "P195",
              "label_en": "collection"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "vertical length of an entity",
              "id": "P2048",
              "label_en": "height"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "width of an object",
              "id": "P2049",
              "label_en": "width"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a physical object or a set of physical objects in a collection",
              "id": "P217",
              "label_en": "inventory number"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "time when an entity begins to exist; for date of official opening use P1619",
              "id": "P571",
              "label_en": "inception"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "copyright status for intellectual creations like works of art, publications, software, etc.",
              "id": "P6216",
              "label_en": "copyright status"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "well-defined, enumerable collection of discrete entities that form a collective whole",
              "id": "Q16887380",
              "label_en": "group"
            },
            {
              "description_en": "physical object made or shaped by humans",
              "id": "Q8205328",
              "label_en": "artificial physical object"
            }
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
  "hash_after": "d27719909da0f0b271e813a62de14c170498c5b5",
  "hash_before": "a3523d7d6eee6283d4c053d612aaaf96b123f92f",
  "property_revision_id": 2317303689,
  "property_revision_prev": 2126294359,
  "qualifier_value_changes": [
    {
      "added_values": [
        "mul"
      ],
      "constraint_qid": "Q108139345",
      "qualifier_property": "P424",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P424",
          "values": [
            "fr"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "label in language constraint: Wikimedia language code: fr, mul",
      "single-value constraint: constraint status: suggestion constraint",
      "format constraint: format as a regular expression: [PEI][M]\\d[0-9AB]\\d\\d\\d\\d\\d\\d; constraint status: mandatory constraint",
      "distinct-values constraint: exception to constraint: lion de saint Marc de l'église Saint-Sauveur de Casesnoves",
      "conflicts-with constraint: property: population; constraint status: mandatory constraint",
      "conflicts-with constraint: property: intangible cultural heritage status; constraint status: mandatory constraint",
      "item-requires-statement constraint: item of property constraint: France; property: country; constraint status: mandatory constraint",
      "item-requires-statement constraint: item of property constraint: object classified as a historical monument, object listed as historical monument, objets non protégés de l'I.G.P.C.; property: heritage designation",
      "item-requires-statement constraint: property: genre; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: creator; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: depicts; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: made from material; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: collection; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: height; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: width; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inventory number; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: location; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inception; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: copyright status; constraint status: suggestion constraint",
      "subject type constraint: class: group, artificial physical object; relation: instance or subclass of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ],
    "before": [
      "label in language constraint: Wikimedia language code: fr",
      "single-value constraint: constraint status: suggestion constraint",
      "format constraint: format as a regular expression: [PEI][M]\\d[0-9AB]\\d\\d\\d\\d\\d\\d; constraint status: mandatory constraint",
      "distinct-values constraint: exception to constraint: lion de saint Marc de l'église Saint-Sauveur de Casesnoves",
      "conflicts-with constraint: property: population; constraint status: mandatory constraint",
      "conflicts-with constraint: property: intangible cultural heritage status; constraint status: mandatory constraint",
      "item-requires-statement constraint: item of property constraint: France; property: country; constraint status: mandatory constraint",
      "item-requires-statement constraint: item of property constraint: object classified as a historical monument, object listed as historical monument, objets non protégés de l'I.G.P.C.; property: heritage designation",
      "item-requires-statement constraint: property: genre; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: creator; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: depicts; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: made from material; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: collection; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: height; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: width; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inventory number; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: location; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inception; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: copyright status; constraint status: suggestion constraint",
      "subject type constraint: class: group, artificial physical object; relation: instance or subclass of",
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|180"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---

## 018. `reform_Q29119733_P481_2317303689`

| Field | Value |
|---|---|
| qid | Q29119733 |
| property | P481 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | TBOX::P481::2317303689 |
| tbox_revision_key | TBOX::P481::2317303689 |

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
| rationale | Constraint qualifiers compared with generic set semantics. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Bob08",
  "kind": "T_BOX",
  "property_revision_id": 2317303689,
  "property_revision_prev": 2126294359
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-02-27T13:51:16",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P481",
  "report_revision_new": 2317728623,
  "report_revision_old": 2317281348,
  "report_violation_type": "Item P|136",
  "report_violation_type_normalized": "Item P|136",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|136",
  "report_violation_types": [
    "Item P|136",
    "Item P|186",
    "Item P|1435 one of Q|61058403, Q|61058419, Q|86830939",
    "Type Q|8205328, Q|16887380"
  ],
  "value": null,
  "value_current_2026": [
    "PM64001546"
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
    "description": "identifier in the Palissy database of moveable objects of French cultural heritage",
    "label": "Palissy ID"
  },
  "qid": {
    "description": "objet monument historique (PM64001546) situé à Hasparren (Pyrénées-Atlantiques, France)",
    "label": "stèle tabulaire, croix et plate tombe"
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
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P424",
          "values": [
            "fr",
            "mul"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 22,
  "author": "Bob08",
  "before_constraint_count": 22,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "fr"
            },
            {
              "value": "mul"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[PEI][M]\\d[0-9AB]\\d\\d\\d\\d\\d\\d"
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
        "parameters": {
          "P2303": [
            {
              "description_en": "fragment de peinture romane déposée",
              "id": "Q28319846",
              "label_en": "lion de saint Marc de l'église Saint-Sauveur de Casesnoves"
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
              "description_en": "number of people inhabiting the place; number of people of subject",
              "id": "P1082",
              "label_en": "population"
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
              "description_en": "status of an item that is designated as intangible heritage",
              "id": "P3259",
              "label_en": "intangible cultural heritage status"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "country in Western Europe and other continents (through its overseas territories in America, Africa and Oceania)",
              "id": "Q142",
              "label_en": "France"
            }
          ],
          "P2306": [
            {
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "French object classified as a Historical Monument by the French State",
              "id": "Q61058403",
              "label_en": "object classified as a historical monument"
            },
            {
              "description_en": "French object listed as a Historical Monument by the French State",
              "id": "Q61058419",
              "label_en": "object listed as historical monument"
            },
            {
              "description_en": "patrimoine inventorié français",
              "id": "Q86830939",
              "label_en": "objets non protégés de l'I.G.P.C."
            }
          ],
          "P2306": [
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
              "id": "P136",
              "label_en": "genre"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "maker of this creative work or other object (where no more specific property exists)",
              "id": "P170",
              "label_en": "creator"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "entity visually depicted in an image, literarily described in a work, or otherwise incorporated into an audiovisual or other medium; see also P921, 'main subject'",
              "id": "P180",
              "label_en": "depicts"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "material the subject or the object is made of or derived from (do not confuse with P10672 which is used for processes)",
              "id": "P186",
              "label_en": "made from material"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "art, museum, archival, or bibliographic collection of which the subject is part (item is in the collection of X)",
              "id": "P195",
              "label_en": "collection"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "vertical length of an entity",
              "id": "P2048",
              "label_en": "height"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "width of an object",
              "id": "P2049",
              "label_en": "width"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a physical object or a set of physical objects in a collection",
              "id": "P217",
              "label_en": "inventory number"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "time when an entity begins to exist; for date of official opening use P1619",
              "id": "P571",
              "label_en": "inception"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "copyright status for intellectual creations like works of art, publications, software, etc.",
              "id": "P6216",
              "label_en": "copyright status"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "well-defined, enumerable collection of discrete entities that form a collective whole",
              "id": "Q16887380",
              "label_en": "group"
            },
            {
              "description_en": "physical object made or shaped by humans",
              "id": "Q8205328",
              "label_en": "artificial physical object"
            }
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
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "fr"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[PEI][M]\\d[0-9AB]\\d\\d\\d\\d\\d\\d"
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
        "parameters": {
          "P2303": [
            {
              "description_en": "fragment de peinture romane déposée",
              "id": "Q28319846",
              "label_en": "lion de saint Marc de l'église Saint-Sauveur de Casesnoves"
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
              "description_en": "number of people inhabiting the place; number of people of subject",
              "id": "P1082",
              "label_en": "population"
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
              "description_en": "status of an item that is designated as intangible heritage",
              "id": "P3259",
              "label_en": "intangible cultural heritage status"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "country in Western Europe and other continents (through its overseas territories in America, Africa and Oceania)",
              "id": "Q142",
              "label_en": "France"
            }
          ],
          "P2306": [
            {
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "French object classified as a Historical Monument by the French State",
              "id": "Q61058403",
              "label_en": "object classified as a historical monument"
            },
            {
              "description_en": "French object listed as a Historical Monument by the French State",
              "id": "Q61058419",
              "label_en": "object listed as historical monument"
            },
            {
              "description_en": "patrimoine inventorié français",
              "id": "Q86830939",
              "label_en": "objets non protégés de l'I.G.P.C."
            }
          ],
          "P2306": [
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
              "id": "P136",
              "label_en": "genre"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "maker of this creative work or other object (where no more specific property exists)",
              "id": "P170",
              "label_en": "creator"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "entity visually depicted in an image, literarily described in a work, or otherwise incorporated into an audiovisual or other medium; see also P921, 'main subject'",
              "id": "P180",
              "label_en": "depicts"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "material the subject or the object is made of or derived from (do not confuse with P10672 which is used for processes)",
              "id": "P186",
              "label_en": "made from material"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "art, museum, archival, or bibliographic collection of which the subject is part (item is in the collection of X)",
              "id": "P195",
              "label_en": "collection"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "vertical length of an entity",
              "id": "P2048",
              "label_en": "height"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "width of an object",
              "id": "P2049",
              "label_en": "width"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "identifier for a physical object or a set of physical objects in a collection",
              "id": "P217",
              "label_en": "inventory number"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "time when an entity begins to exist; for date of official opening use P1619",
              "id": "P571",
              "label_en": "inception"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "copyright status for intellectual creations like works of art, publications, software, etc.",
              "id": "P6216",
              "label_en": "copyright status"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "well-defined, enumerable collection of discrete entities that form a collective whole",
              "id": "Q16887380",
              "label_en": "group"
            },
            {
              "description_en": "physical object made or shaped by humans",
              "id": "Q8205328",
              "label_en": "artificial physical object"
            }
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
  "hash_after": "d27719909da0f0b271e813a62de14c170498c5b5",
  "hash_before": "a3523d7d6eee6283d4c053d612aaaf96b123f92f",
  "property_revision_id": 2317303689,
  "property_revision_prev": 2126294359,
  "qualifier_value_changes": [
    {
      "added_values": [
        "mul"
      ],
      "constraint_qid": "Q108139345",
      "qualifier_property": "P424",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q108139345",
      "qualifiers": [
        {
          "property_id": "P424",
          "values": [
            "fr"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "label in language constraint: Wikimedia language code: fr, mul",
      "single-value constraint: constraint status: suggestion constraint",
      "format constraint: format as a regular expression: [PEI][M]\\d[0-9AB]\\d\\d\\d\\d\\d\\d; constraint status: mandatory constraint",
      "distinct-values constraint: exception to constraint: lion de saint Marc de l'église Saint-Sauveur de Casesnoves",
      "conflicts-with constraint: property: population; constraint status: mandatory constraint",
      "conflicts-with constraint: property: intangible cultural heritage status; constraint status: mandatory constraint",
      "item-requires-statement constraint: item of property constraint: France; property: country; constraint status: mandatory constraint",
      "item-requires-statement constraint: item of property constraint: object classified as a historical monument, object listed as historical monument, objets non protégés de l'I.G.P.C.; property: heritage designation",
      "item-requires-statement constraint: property: genre; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: creator; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: depicts; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: made from material; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: collection; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: height; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: width; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inventory number; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: location; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inception; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: copyright status; constraint status: suggestion constraint",
      "subject type constraint: class: group, artificial physical object; relation: instance or subclass of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ],
    "before": [
      "label in language constraint: Wikimedia language code: fr",
      "single-value constraint: constraint status: suggestion constraint",
      "format constraint: format as a regular expression: [PEI][M]\\d[0-9AB]\\d\\d\\d\\d\\d\\d; constraint status: mandatory constraint",
      "distinct-values constraint: exception to constraint: lion de saint Marc de l'église Saint-Sauveur de Casesnoves",
      "conflicts-with constraint: property: population; constraint status: mandatory constraint",
      "conflicts-with constraint: property: intangible cultural heritage status; constraint status: mandatory constraint",
      "item-requires-statement constraint: item of property constraint: France; property: country; constraint status: mandatory constraint",
      "item-requires-statement constraint: item of property constraint: object classified as a historical monument, object listed as historical monument, objets non protégés de l'I.G.P.C.; property: heritage designation",
      "item-requires-statement constraint: property: genre; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: creator; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: depicts; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: made from material; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: collection; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: height; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: width; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inventory number; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: location; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inception; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: copyright status; constraint status: suggestion constraint",
      "subject type constraint: class: group, artificial physical object; relation: instance or subclass of",
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|136"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---

## 019. `reform_Q49005278_P403_2442802452`

| Field | Value |
|---|---|
| qid | Q49005278 |
| property | P403 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | TBOX::P403::2442802452 |
| tbox_revision_key | TBOX::P403::2442802452 |

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
| rationale | Type/value-type constraint classes and relations compared using P2308/P2309. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Vlk",
  "kind": "T_BOX",
  "property_revision_id": 2442802452,
  "property_revision_prev": 2421662999
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-19T10:56:29",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P403",
  "report_revision_new": 2444036897,
  "report_revision_old": 2443833509,
  "report_violation_type": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "report_violation_type_descriptions_en": [
    "any flowing body of water",
    "river which only exists in fiction",
    "river that only exists in myth",
    "terrestrial water source",
    "low area between hills, often with a river running through it",
    "large persistent body of ice",
    "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
    "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
    "any significant accumulation of water, generally on a planet's surface",
    "body of water without noticeable current",
    "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
    "waterbody only existing in fiction",
    "small stream, i.e. a small natural watercourse"
  ],
  "report_violation_type_labels_en": [
    "watercourse",
    "fictional river",
    "mythical river",
    "spring",
    "valley",
    "glacier",
    "lake system",
    "drainage system",
    "body of water",
    "still waters",
    "wadi",
    "fictional body of water",
    "brook"
  ],
  "report_violation_type_normalized": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "report_violation_type_qids": [
    "Q355304",
    "Q16338046",
    "Q24336031",
    "Q124714",
    "Q39816",
    "Q35666",
    "Q104347069",
    "Q285451",
    "Q15324",
    "Q337567",
    "Q187971",
    "Q16500104",
    "Q63565252"
  ],
  "report_violation_type_raw": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252",
  "value": null,
  "value_current_2026": [
    "Q6395627"
  ],
  "value_current_2026_descriptions_en": [
    "river in South Africa"
  ],
  "value_current_2026_labels_en": [
    "Keurbooms River"
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
    "description": "the body of water to which the watercourse drains",
    "label": "mouth of the watercourse"
  },
  "qid": {
    "description": null,
    "label": "Keurboomsriviermond"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
            "Q104347069",
            "Q124714",
            "Q15324",
            "Q16338046",
            "Q16500104",
            "Q187971",
            "Q24336031",
            "Q285451",
            "Q337567",
            "Q355304",
            "Q35666",
            "Q39816",
            "Q63565252"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q21503252"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 6,
  "author": "Vlk",
  "before_constraint_count": 6,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
              "id": "Q104347069",
              "label_en": "lake system"
            },
            {
              "description_en": "terrestrial water source",
              "id": "Q124714",
              "label_en": "spring"
            },
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "river which only exists in fiction",
              "id": "Q16338046",
              "label_en": "fictional river"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "river that only exists in myth",
              "id": "Q24336031",
              "label_en": "mythical river"
            },
            {
              "description_en": "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
              "id": "Q285451",
              "label_en": "drainage system"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "any flowing body of water",
              "id": "Q355304",
              "label_en": "watercourse"
            },
            {
              "description_en": "large persistent body of ice",
              "id": "Q35666",
              "label_en": "glacier"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "small stream, i.e. a small natural watercourse",
              "id": "Q63565252",
              "label_en": "brook"
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
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "height of the item (geographical object) as measured relative to sea level",
              "id": "P2044",
              "label_en": "elevation above sea level"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            {
              "description_en": "specify if the stream confluence is a left bank or right bank tributary",
              "id": "P3871",
              "label_en": "tributary orientation"
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
              "description_en": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
              "id": "P613",
              "label_en": "OS grid reference"
            },
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "infrastructure that conveys sewage or surface runoff",
              "id": "Q156849",
              "label_en": "sewer network"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "meeting of two or more bodies of flowing water",
              "id": "Q723748",
              "label_en": "confluence"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "stream in Highland, Scotland, UK, tributary of the Allt Cuaich and of the Cuaich Aqueduct",
              "id": "Q112729719",
              "label_en": "Féith Chàm"
            },
            {
              "description_en": "canal section in Argyll and Bute, Scotland, UK, flows west into Loch Crinan at Crinan, and east into the canal section from Cairnbaan to Ardrishaig",
              "id": "Q56664718",
              "label_en": "Crinan Canal, section from Crinan to Cairnbaan"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "part of river basin formed by lakes connected with short rivers, straits or narrows.; system of surface and ground waters flowing into a common terminus such as the sea, lake or aquifer",
              "id": "Q104347069",
              "label_en": "lake system"
            },
            {
              "description_en": "terrestrial water source",
              "id": "Q124714",
              "label_en": "spring"
            },
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "river which only exists in fiction",
              "id": "Q16338046",
              "label_en": "fictional river"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "river that only exists in myth",
              "id": "Q24336031",
              "label_en": "mythical river"
            },
            {
              "description_en": "pattern formed by the streams, rivers, and lakes in a particular drainage basin",
              "id": "Q285451",
              "label_en": "drainage system"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "any flowing body of water",
              "id": "Q355304",
              "label_en": "watercourse"
            },
            {
              "description_en": "large persistent body of ice",
              "id": "Q35666",
              "label_en": "glacier"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
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
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "qualifier used together with the end date qualifier (P582) to specify the reason for the end",
              "id": "P1534",
              "label_en": "end cause"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "height of the item (geographical object) as measured relative to sea level",
              "id": "P2044",
              "label_en": "elevation above sea level"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            {
              "description_en": "specify if the stream confluence is a left bank or right bank tributary",
              "id": "P3871",
              "label_en": "tributary orientation"
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
              "description_en": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
              "id": "P613",
              "label_en": "OS grid reference"
            },
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "id": "Q21510865",
          "label_en": "value-type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "any significant accumulation of water, generally on a planet's surface",
              "id": "Q15324",
              "label_en": "body of water"
            },
            {
              "description_en": "infrastructure that conveys sewage or surface runoff",
              "id": "Q156849",
              "label_en": "sewer network"
            },
            {
              "description_en": "waterbody only existing in fiction",
              "id": "Q16500104",
              "label_en": "fictional body of water"
            },
            {
              "description_en": "river valley, especially a dry (ephemeral) riverbed that contains water only during times of heavy rain",
              "id": "Q187971",
              "label_en": "wadi"
            },
            {
              "description_en": "body of water without noticeable current",
              "id": "Q337567",
              "label_en": "still waters"
            },
            {
              "description_en": "low area between hills, often with a river running through it",
              "id": "Q39816",
              "label_en": "valley"
            },
            {
              "description_en": "meeting of two or more bodies of flowing water",
              "id": "Q723748",
              "label_en": "confluence"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "stream in Highland, Scotland, UK, tributary of the Allt Cuaich and of the Cuaich Aqueduct",
              "id": "Q112729719",
              "label_en": "Féith Chàm"
            },
            {
              "description_en": "canal section in Argyll and Bute, Scotland, UK, flows west into Loch Crinan at Crinan, and east into the canal section from Cairnbaan to Ardrishaig",
              "id": "Q56664718",
              "label_en": "Crinan Canal, section from Crinan to Cairnbaan"
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
  "hash_after": "36188fe72951315c87372c2b0c88051dd7ea3646",
  "hash_before": "1eae5a89c1de92534f9fe810620210d2cac7c93f",
  "property_revision_id": 2442802452,
  "property_revision_prev": 2421662999,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q63565252"
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
            "Q104347069",
            "Q124714",
            "Q15324",
            "Q16338046",
            "Q16500104",
            "Q187971",
            "Q24336031",
            "Q285451",
            "Q337567",
            "Q355304",
            "Q35666",
            "Q39816"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q21503252"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "subject type constraint: class: lake system, spring, body of water, fictional river, fictional body of water, wadi, mythical river, drainage system, still waters, watercourse, glacier, valley, brook; relation: instance of",
      "allowed qualifiers constraint: property: located in the administrative territorial entity, end cause, subject named as, object named as, elevation above sea level, reason for deprecated rank, location, tributary orientation, start time, end time, OS grid reference, coordinate location, reason for preferred rank",
      "value-type constraint: class: body of water, sewer network, fictional body of water, wadi, still waters, valley, confluence; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "single-best-value constraint: exception to constraint: Féith Chàm, Crinan Canal, section from Crinan to Cairnbaan",
      "property scope constraint: property scope: as main value"
    ],
    "before": [
      "subject type constraint: class: lake system, spring, body of water, fictional river, fictional body of water, wadi, mythical river, drainage system, still waters, watercourse, glacier, valley; relation: instance of",
      "allowed qualifiers constraint: property: located in the administrative territorial entity, end cause, subject named as, object named as, elevation above sea level, reason for deprecated rank, location, tributary orientation, start time, end time, OS grid reference, coordinate location, reason for preferred rank",
      "value-type constraint: class: body of water, sewer network, fictional body of water, wadi, still waters, valley, confluence; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "single-best-value constraint: exception to constraint: Féith Chàm, Crinan Canal, section from Crinan to Cairnbaan",
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|355304, Q|16338046, Q|24336031, Q|124714, Q|39816, Q|35666, Q|104347069, Q|285451, Q|15324, Q|337567, Q|187971, Q|16500104, Q|63565252"
  },
  {
    "result": "Q21503250",
    "step": "target_constraint"
  },
  {
    "property_ids": [
      "P2308",
      "P2309"
    ],
    "result": "RELAXATION_SET_EXPANSION",
    "step": "set_semantics"
  }
]
```

---

## 020. `reform_Q53265697_P4969_2435927232`

| Field | Value |
|---|---|
| qid | Q53265697 |
| property | P4969 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / RELAXATION_SET_EXPANSION / high |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_relaxation_set_expansion |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | TBOX::P4969::2435927232 |
| tbox_revision_key | TBOX::P4969::2435927232 |

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
| rationale | Constraint qualifiers compared with generic set semantics. |
| local_match_kind |  |
| local_match_source |  |

### What Changed

```json
{
  "author": "Trade",
  "kind": "T_BOX",
  "property_revision_id": 2435927232,
  "property_revision_prev": 2435926155
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-06T06:28:43",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4969",
  "report_revision_new": 2438638194,
  "report_revision_old": 2438318580,
  "report_violation_type": "Type Q|386724, Q|4886, Q|43099500, Q|95074, Q|18706315, Q|16686448, Q|116779428, Q|11424, Q|48708989, Q|15142894, Q|47451145",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "plant or grouping of plants selected for desirable characteristics",
    "production of the performing arts, consisting of a series of quasi-identical performances of the same performance work",
    "fictional human or non-human character in a narrative work of art",
    "entity whose existence is possible, but not proven",
    "anything created by humans (either material or mental)",
    "סוג קבוצת יצירות",
    "sequence of images that give the impression of movement, stored on film stock",
    "group of related ammunition cartridges which share basic design elements",
    "specific weapon design, pattern, or version of which all examples are essentially identical",
    "recurring, self-sufficient plot or motif grouping, unit of classification in the Aarne–Thompson classification systems"
  ],
  "report_violation_type_labels_en": [
    "work",
    "cultivar",
    "performing arts production",
    "character",
    "hypothetical entity",
    "artificial object",
    "group of works often treated as a singular work",
    "film",
    "cartridge family",
    "weapon model",
    "tale type"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|4886, Q|43099500, Q|95074, Q|18706315, Q|16686448, Q|116779428, Q|11424, Q|48708989, Q|15142894, Q|47451145",
  "report_violation_type_qids": [
    "Q386724",
    "Q4886",
    "Q43099500",
    "Q95074",
    "Q18706315",
    "Q16686448",
    "Q116779428",
    "Q11424",
    "Q48708989",
    "Q15142894",
    "Q47451145"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|4886, Q|43099500, Q|95074, Q|18706315, Q|16686448, Q|116779428, Q|11424, Q|48708989, Q|15142894, Q|47451145",
  "value": null,
  "value_current_2026": [
    "Q169215"
  ],
  "value_current_2026_descriptions_en": [
    "national anthem of Slovenia (7th stanza)"
  ],
  "value_current_2026_labels_en": [
    "Zdravljica"
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
    "description": "new work of art (film, book, software, etc.) derived from major part of this work",
    "label": "derivative work"
  },
  "qid": {
    "description": "poem by France Prešeren",
    "label": "Zdravljica"
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
    "label_en": "inverse constraint",
    "qid": "Q21510855"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502838",
      "qualifiers": [
        {
          "property_id": "P2305",
          "values": [
            "Q136747113",
            "Q136832029",
            "Q3331189",
            "Q57933693"
          ]
        },
        {
          "property_id": "P2306",
          "values": [
            "P31"
          ]
        },
        {
          "property_id": "P2316",
          "values": [
            "Q21502408"
          ]
        },
        {
          "property_id": "P6607",
          "values": [
            "works should have this property instead of editions, use in Q7725634@en"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 6,
  "author": "Trade",
  "before_constraint_count": 6,
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
              "description_en": "Japanese style of animation",
              "id": "Q1107",
              "label_en": "anime"
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
          ],
          "P9729": [
            {
              "description_en": "series of light novels published in Japan",
              "id": "Q104213567",
              "label_en": "light novel series"
            },
            {
              "description_en": "series of comics employing Japanese stylistic conventions that are that are formally identified together",
              "id": "Q21198342",
              "label_en": "manga series"
            },
            {
              "description_en": "Japanese animated television series",
              "id": "Q63952888",
              "label_en": "anime television series"
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
              "description_en": "edition of a light novel",
              "id": "Q136747113",
              "label_en": "light novel edition"
            },
            {
              "description_en": "edition of a manga",
              "id": "Q136832029",
              "label_en": "manga edition"
            },
            {
              "description_en": "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
              "id": "Q3331189",
              "label_en": "version, edition or translation"
            },
            {
              "description_en": "edition of a book",
              "id": "Q57933693",
              "label_en": "book edition"
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
          ],
          "P6607": [
            {
              "value": "works should have this property instead of editions, use in Q7725634@en"
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
              "description_en": "sequence of images that give the impression of movement, stored on film stock",
              "id": "Q11424",
              "label_en": "film"
            },
            {
              "description_en": "סוג קבוצת יצירות",
              "id": "Q116779428",
              "label_en": "group of works often treated as a singular work"
            },
            {
              "description_en": "specific weapon design, pattern, or version of which all examples are essentially identical",
              "id": "Q15142894",
              "label_en": "weapon model"
            },
            {
              "description_en": "anything created by humans (either material or mental)",
              "id": "Q16686448",
              "label_en": "artificial object"
            },
            {
              "description_en": "entity whose existence is possible, but not proven",
              "id": "Q18706315",
              "label_en": "hypothetical entity"
            },
            {
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
            },
            {
              "description_en": "production of the performing arts, consisting of a series of quasi-identical performances of the same performance work",
              "id": "Q43099500",
              "label_en": "performing arts production"
            },
            {
              "description_en": "recurring, self-sufficient plot or motif grouping, unit of classification in the Aarne–Thompson classification systems",
              "id": "Q47451145",
              "label_en": "tale type"
            },
            {
              "description_en": "group of related ammunition cartridges which share basic design elements",
              "id": "Q48708989",
              "label_en": "cartridge family"
            },
            {
              "description_en": "plant or grouping of plants selected for desirable characteristics",
              "id": "Q4886",
              "label_en": "cultivar"
            },
            {
              "description_en": "fictional human or non-human character in a narrative work of art",
              "id": "Q95074",
              "label_en": "character"
            }
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item has to refer back to this item with the given inverse property",
          "id": "Q21510855",
          "label_en": "inverse constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the work(s) or inputs used as the basis for subject item; for fictional analog use P1074",
              "id": "P144",
              "label_en": "based on"
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
              "id": "Q54828449",
              "label_en": "as qualifier"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "Japanese style of animation",
              "id": "Q1107",
              "label_en": "anime"
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
          ],
          "P9729": [
            {
              "description_en": "series of light novels published in Japan",
              "id": "Q104213567",
              "label_en": "light novel series"
            },
            {
              "description_en": "series of comics employing Japanese stylistic conventions that are that are formally identified together",
              "id": "Q21198342",
              "label_en": "manga series"
            },
            {
              "description_en": "Japanese animated television series",
              "id": "Q63952888",
              "label_en": "anime television series"
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
              "description_en": "edition of a manga",
              "id": "Q136832029",
              "label_en": "manga edition"
            },
            {
              "description_en": "specific version of a work, resulting from its edition, adaptation, or translation; set of substantially similar copies of a work (use with P31 [\"instance of\"])",
              "id": "Q3331189",
              "label_en": "version, edition or translation"
            },
            {
              "description_en": "edition of a book",
              "id": "Q57933693",
              "label_en": "book edition"
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
          ],
          "P6607": [
            {
              "value": "works should have this property instead of editions, use in Q7725634@en"
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
              "description_en": "sequence of images that give the impression of movement, stored on film stock",
              "id": "Q11424",
              "label_en": "film"
            },
            {
              "description_en": "סוג קבוצת יצירות",
              "id": "Q116779428",
              "label_en": "group of works often treated as a singular work"
            },
            {
              "description_en": "specific weapon design, pattern, or version of which all examples are essentially identical",
              "id": "Q15142894",
              "label_en": "weapon model"
            },
            {
              "description_en": "anything created by humans (either material or mental)",
              "id": "Q16686448",
              "label_en": "artificial object"
            },
            {
              "description_en": "entity whose existence is possible, but not proven",
              "id": "Q18706315",
              "label_en": "hypothetical entity"
            },
            {
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
            },
            {
              "description_en": "production of the performing arts, consisting of a series of quasi-identical performances of the same performance work",
              "id": "Q43099500",
              "label_en": "performing arts production"
            },
            {
              "description_en": "recurring, self-sufficient plot or motif grouping, unit of classification in the Aarne–Thompson classification systems",
              "id": "Q47451145",
              "label_en": "tale type"
            },
            {
              "description_en": "group of related ammunition cartridges which share basic design elements",
              "id": "Q48708989",
              "label_en": "cartridge family"
            },
            {
              "description_en": "plant or grouping of plants selected for desirable characteristics",
              "id": "Q4886",
              "label_en": "cultivar"
            },
            {
              "description_en": "fictional human or non-human character in a narrative work of art",
              "id": "Q95074",
              "label_en": "character"
            }
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item has to refer back to this item with the given inverse property",
          "id": "Q21510855",
          "label_en": "inverse constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the work(s) or inputs used as the basis for subject item; for fictional analog use P1074",
              "id": "P144",
              "label_en": "based on"
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
              "id": "Q54828449",
              "label_en": "as qualifier"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "299ae3d11d3719afbe8480654eb40a568a11f3df",
  "hash_before": "068118b0b02e8854ee54a1f3bb75df62a2e858cc",
  "property_revision_id": 2435927232,
  "property_revision_prev": 2435926155,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q136747113"
      ],
      "constraint_qid": "Q21502838",
      "qualifier_property": "P2305",
      "removed_values": [],
      "same_qid_index": 1
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21502838",
      "qualifiers": [
        {
          "property_id": "P2305",
          "values": [
            "Q136832029",
            "Q3331189",
            "Q57933693"
          ]
        },
        {
          "property_id": "P2306",
          "values": [
            "P31"
          ]
        },
        {
          "property_id": "P2316",
          "values": [
            "Q21502408"
          ]
        },
        {
          "property_id": "P6607",
          "values": [
            "works should have this property instead of editions, use in Q7725634@en"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "conflicts-with constraint: item of property constraint: anime, light novel, manga; property: instance of; replacement value: light novel series, manga series, anime television series",
      "conflicts-with constraint: item of property constraint: light novel edition, manga edition, version, edition or translation, book edition; property: instance of; constraint status: mandatory constraint; constraint clarification: works should have this property instead of editions, use in Q7725634@en",
      "subject type constraint: class: film, group of works often treated as a singular work, weapon model, artificial object, hypothetical entity, work, performing arts production, tale type, cartridge family, cultivar, character; relation: instance or subclass of",
      "inverse constraint: property: based on",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as qualifier"
    ],
    "before": [
      "conflicts-with constraint: item of property constraint: anime, light novel, manga; property: instance of; replacement value: light novel series, manga series, anime television series",
      "conflicts-with constraint: item of property constraint: manga edition, version, edition or translation, book edition; property: instance of; constraint status: mandatory constraint; constraint clarification: works should have this property instead of editions, use in Q7725634@en",
      "subject type constraint: class: film, group of works often treated as a singular work, weapon model, artificial object, hypothetical entity, work, performing arts production, tale type, cartridge family, cultivar, character; relation: instance or subclass of",
      "inverse constraint: property: based on",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as qualifier"
    ]
  }
}
```

### Decision Trace

```json
[
  {
    "changed_constraint_qids": [],
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Type Q|386724, Q|4886, Q|43099500, Q|95074, Q|18706315, Q|16686448, Q|116779428, Q|11424, Q|48708989, Q|15142894, Q|47451145"
  },
  {
    "result": "Q21502838",
    "step": "target_constraint"
  },
  {
    "result": "RELAXATION_SET_EXPANSION",
    "step": "generic_set_semantics"
  }
]
```

---
