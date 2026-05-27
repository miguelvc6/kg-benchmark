# TBOX_SCHEMA_UPDATE

Cases: 20

Use this file for evidence review. Enter final annotations in the CSV copy, not here.

## 001. `reform_Q106814418_P1346_1577277439`

| Field | Value |
|---|---|
| qid | Q106814418 |
| property | P1346 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | TBOX::P1346::1577277439 |
| tbox_revision_key | TBOX::P1346::1577277439 |

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
  "author": "Lucio Luiz",
  "kind": "T_BOX",
  "property_revision_id": 1577277439,
  "property_revision_prev": 1575232987
}
```

### Violation Context

```json
{
  "report_fix_date": "2022-02-15T10:36:41",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1346",
  "report_revision_new": 1577696536,
  "report_revision_old": 1577200212,
  "report_violation_type": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
  "report_violation_type_descriptions_en": [
    "any single member of Homo sapiens, unique extant species of the genus Homo",
    "social entity established to meet needs or pursue goals",
    "community of people who share a common language, culture, ethnicity, descent, or history",
    "large landmass identified by convention",
    "domesticated four-footed mammal from the equine family",
    "publication type, serial publication that appears in a new edition on a regular schedule",
    "computer designed for playing chess",
    "place of any size, in which people permanently live",
    "territorial entity for administration purposes, with or without its own local government",
    "fictional human or non-human character in a narrative work of art",
    "medium for recording information (words or images) typically on bound pages or more abstractly in electronic or audio form",
    "creative work in which images and text convey information such as narratives",
    "intellectual or artistic creation"
  ],
  "report_violation_type_labels_en": [
    "human",
    "organization",
    "nation",
    "continent",
    "horse",
    "periodical",
    "chess computer",
    "human settlement",
    "administrative territorial entity",
    "character",
    "book",
    "comics",
    "work"
  ],
  "report_violation_type_normalized": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
  "report_violation_type_qids": [
    "Q5",
    "Q43229",
    "Q6266",
    "Q5107",
    "Q726",
    "Q1002697",
    "Q1364192",
    "Q486972",
    "Q56061",
    "Q95074",
    "Q571",
    "Q1004",
    "Q386724"
  ],
  "report_violation_type_raw": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724",
  "value": null,
  "value_current_2026": [
    "Q288333"
  ],
  "value_current_2026_descriptions_en": [
    "Italian diver"
  ],
  "value_current_2026_labels_en": [
    "Tania Cagnotto"
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
    "description": "winner of a competition or similar event, not to be used from the awardees record (instead use \"award received\" (P166), possibly qualified with \"for work\" (P1686)) nor for wars or battles",
    "label": "winner"
  },
  "qid": {
    "description": null,
    "label": "diving at the 2008 European Aquatics Championships – women's 10 metre platform"
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
    "label_en": "inverse constraint",
    "qid": "Q21510855"
  },
  {
    "label_en": "value-type constraint",
    "qid": "Q21510865"
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
      "constraint_qid": "Q21510865",
      "qualifiers": [
        {
          "property_id": "P2308",
          "values": [
            "Q1002697",
            "Q1004",
            "Q1364192",
            "Q1656682",
            "Q386724",
            "Q43229",
            "Q486972",
            "Q5",
            "Q5107",
            "Q56061",
            "Q571",
            "Q6266",
            "Q726",
            "Q95074"
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
  "after_constraint_count": 7,
  "author": "Lucio Luiz",
  "before_constraint_count": 7,
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
              "description_en": "men's 2 × 10 kilometre pursuit events at the Olympics",
              "id": "Q1004888",
              "label_en": "cross-country skiing at the 2002 Winter Olympics – men's 2 x 10 kilometre pursuit"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q1026922",
              "label_en": "gymnastics at the 1992 Summer Olympics – men's pommel horse"
            },
            {
              "description_en": "women's vault events at the Olympics",
              "id": "Q1026946",
              "label_en": "gymnastics at the 1992 Summer Olympics – women's vault"
            },
            {
              "description_en": "equestrian at the Olympics",
              "id": "Q1043979",
              "label_en": "equestrian at the 1900 Summer Olympics – high jump"
            },
            {
              "description_en": "artistic gymnastics event",
              "id": "Q1046746",
              "label_en": "gymnastics at the 1904 Summer Olympics – men's horizontal bar"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q1150111",
              "label_en": "gymnastics at the 1988 Summer Olympics – men's horizontal bar"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q1150216",
              "label_en": "gymnastics at the 1988 Summer Olympics – men's rings"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q1150230",
              "label_en": "gymnastics at the 1988 Summer Olympics – men's pommel horse"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q1252361",
              "label_en": "gymnastics at the 1984 Summer Olympics – men's rings"
            },
            {
              "description_en": "sportevenement op de Olympische Spelen",
              "id": "Q1253215",
              "label_en": "gymnastics at the 1984 Summer Olympics – women's balance beam"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q1253733",
              "label_en": "gymnastics at the 1984 Summer Olympics – men's pommel horse"
            },
            {
              "description_en": "sportevenement op de Olympische Spelen",
              "id": "Q1254154",
              "label_en": "gymnastics at the 1984 Summer Olympics – women's uneven bars"
            },
            {
              "description_en": "sportevenement op de Olympische Spelen",
              "id": "Q1389161",
              "label_en": "gymnastics at the 1980 Summer Olympics – women's floor"
            },
            {
              "description_en": "women's downhill events at the Olympics",
              "id": "Q15054994",
              "label_en": "alpine skiing at the 2014 Winter Olympics – women's downhill"
            },
            {
              "description_en": "Olympic gymnastics event",
              "id": "Q17631603",
              "label_en": "gymnastics at the 1948 Summer Olympics – men's pommel horse"
            },
            {
              "description_en": "women's solo events at the Olympics",
              "id": "Q20670790",
              "label_en": "synchronized swimming at the 1992 Summer Olympics – women's solo"
            },
            {
              "description_en": "women's 100 metre freestyle events at the Olympics",
              "id": "Q25932055",
              "label_en": "swimming at the 2016 Summer Olympics – women's 100 metre freestyle"
            },
            {
              "description_en": "Athletics at the Olympics",
              "id": "Q3369579",
              "label_en": "athletics at the 1908 Summer Olympics – men's pole vault"
            },
            {
              "description_en": "men's 50 metre freestyle events at the Olympics",
              "id": "Q3879604",
              "label_en": "swimming at the 2000 Summer Olympics – men's 50 metre freestyle"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q5624269",
              "label_en": "gymnastics at the 1956 Summer Olympics – men's vault"
            },
            {
              "description_en": "women's floor events at the Olympics",
              "id": "Q5624272",
              "label_en": "gymnastics at the 1956 Summer Olympics – women's floor"
            },
            {
              "description_en": "men's pommel horse events at the Olympics",
              "id": "Q5624282",
              "label_en": "gymnastics at the 1960 Summer Olympics – men's pommel horse"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q5624284",
              "label_en": "gymnastics at the 1960 Summer Olympics – men's vault"
            },
            {
              "description_en": "Olympic gymnastics event",
              "id": "Q5624313",
              "label_en": "gymnastics at the 1968 Summer Olympics – men's horizontal bar"
            },
            "... omitted 9 items"
          ],
          "P4155": [
            {
              "description_en": "qualifier of award received (P166) to specify the work that an award was given to the creator for",
              "id": "P1686",
              "label_en": "for work"
            },
            {
              "description_en": "qualifier to specify the item that this property is shared with",
              "id": "P1706",
              "label_en": "together with"
            },
            {
              "description_en": "official classification by a regulating body under which the subject (events, teams, participants, or equipment) qualifies for inclusion",
              "id": "P2094",
              "label_en": "competition class"
            },
            {
              "description_en": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
              "id": "P21",
              "label_en": "sex or gender"
            },
            {
              "description_en": "discipline an athlete competed in within a sport",
              "id": "P2416",
              "label_en": "sports discipline competed in"
            },
            {
              "description_en": "number of an edition (first, second, ... as 1, 2, ...) or event",
              "id": "P393",
              "label_en": "edition number"
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
              "description_en": "(Do not use. Find alternatives at WD:P642) qualifier stating that a statement applies within the scope of a particular item",
              "id": "P642",
              "label_en": "of (DEPRECATED)"
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
              "description_en": "coordinated military actions of a state or a non-state actor",
              "id": "Q645883",
              "label_en": "military operation"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "edition in a specific year of an annual music competition, song contest, etc.",
              "id": "Q106594095",
              "label_en": "annual music competition edition"
            },
            {
              "description_en": "event, during which one or more sporting events are held",
              "id": "Q13406554",
              "label_en": "sports competition"
            },
            {
              "description_en": "competition which only exists in a fictional universe",
              "id": "Q15707532",
              "label_en": "fictional competition"
            },
            {
              "description_en": "sports event scheduled to recur within a decided interval",
              "id": "Q18608583",
              "label_en": "recurring sporting event"
            },
            {
              "description_en": "set of episodes produced for a television series",
              "id": "Q3464665",
              "label_en": "television series season"
            },
            {
              "description_en": "rivalry where multiple parties strive for a goal which cannot be shared",
              "id": "Q476300",
              "label_en": "competition"
            },
            {
              "description_en": "video serial broadcast on the Internet",
              "id": "Q526877",
              "label_en": "web series"
            },
            {
              "description_en": "something given to a person or a group of people to recognize their merit or excellence",
              "id": "Q618779",
              "label_en": "award"
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
              "description_en": "competition or sports event won by the subject",
              "id": "P2522",
              "label_en": "competition won"
            }
          ],
          "P6607": [
            {
              "value": "inverse generally not used@en"
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
              "description_en": "publication type, serial publication that appears in a new edition on a regular schedule",
              "id": "Q1002697",
              "label_en": "periodical"
            },
            {
              "description_en": "creative work in which images and text convey information such as narratives",
              "id": "Q1004",
              "label_en": "comics"
            },
            {
              "description_en": "computer designed for playing chess",
              "id": "Q1364192",
              "label_en": "chess computer"
            },
            {
              "description_en": "temporary and scheduled happening, like a conference, festival, competition or similar",
              "id": "Q1656682",
              "label_en": "event"
            },
            {
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
            },
            {
              "description_en": "social entity established to meet needs or pursue goals",
              "id": "Q43229",
              "label_en": "organization"
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
              "description_en": "large landmass identified by convention",
              "id": "Q5107",
              "label_en": "continent"
            },
            {
              "description_en": "territorial entity for administration purposes, with or without its own local government",
              "id": "Q56061",
              "label_en": "administrative territorial entity"
            },
            {
              "description_en": "medium for recording information (words or images) typically on bound pages or more abstractly in electronic or audio form",
              "id": "Q571",
              "label_en": "book"
            },
            {
              "description_en": "community of people who share a common language, culture, ethnicity, descent, or history",
              "id": "Q6266",
              "label_en": "nation"
            },
            {
              "description_en": "domesticated four-footed mammal from the equine family",
              "id": "Q726",
              "label_en": "horse"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "id": "Q19474404",
          "label_en": "single-value constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "men's 2 × 10 kilometre pursuit events at the Olympics",
              "id": "Q1004888",
              "label_en": "cross-country skiing at the 2002 Winter Olympics – men's 2 x 10 kilometre pursuit"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q1026922",
              "label_en": "gymnastics at the 1992 Summer Olympics – men's pommel horse"
            },
            {
              "description_en": "women's vault events at the Olympics",
              "id": "Q1026946",
              "label_en": "gymnastics at the 1992 Summer Olympics – women's vault"
            },
            {
              "description_en": "equestrian at the Olympics",
              "id": "Q1043979",
              "label_en": "equestrian at the 1900 Summer Olympics – high jump"
            },
            {
              "description_en": "artistic gymnastics event",
              "id": "Q1046746",
              "label_en": "gymnastics at the 1904 Summer Olympics – men's horizontal bar"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q1150111",
              "label_en": "gymnastics at the 1988 Summer Olympics – men's horizontal bar"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q1150216",
              "label_en": "gymnastics at the 1988 Summer Olympics – men's rings"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q1150230",
              "label_en": "gymnastics at the 1988 Summer Olympics – men's pommel horse"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q1252361",
              "label_en": "gymnastics at the 1984 Summer Olympics – men's rings"
            },
            {
              "description_en": "sportevenement op de Olympische Spelen",
              "id": "Q1253215",
              "label_en": "gymnastics at the 1984 Summer Olympics – women's balance beam"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q1253733",
              "label_en": "gymnastics at the 1984 Summer Olympics – men's pommel horse"
            },
            {
              "description_en": "sportevenement op de Olympische Spelen",
              "id": "Q1254154",
              "label_en": "gymnastics at the 1984 Summer Olympics – women's uneven bars"
            },
            {
              "description_en": "sportevenement op de Olympische Spelen",
              "id": "Q1389161",
              "label_en": "gymnastics at the 1980 Summer Olympics – women's floor"
            },
            {
              "description_en": "women's downhill events at the Olympics",
              "id": "Q15054994",
              "label_en": "alpine skiing at the 2014 Winter Olympics – women's downhill"
            },
            {
              "description_en": "Olympic gymnastics event",
              "id": "Q17631603",
              "label_en": "gymnastics at the 1948 Summer Olympics – men's pommel horse"
            },
            {
              "description_en": "women's solo events at the Olympics",
              "id": "Q20670790",
              "label_en": "synchronized swimming at the 1992 Summer Olympics – women's solo"
            },
            {
              "description_en": "women's 100 metre freestyle events at the Olympics",
              "id": "Q25932055",
              "label_en": "swimming at the 2016 Summer Olympics – women's 100 metre freestyle"
            },
            {
              "description_en": "Athletics at the Olympics",
              "id": "Q3369579",
              "label_en": "athletics at the 1908 Summer Olympics – men's pole vault"
            },
            {
              "description_en": "men's 50 metre freestyle events at the Olympics",
              "id": "Q3879604",
              "label_en": "swimming at the 2000 Summer Olympics – men's 50 metre freestyle"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q5624269",
              "label_en": "gymnastics at the 1956 Summer Olympics – men's vault"
            },
            {
              "description_en": "women's floor events at the Olympics",
              "id": "Q5624272",
              "label_en": "gymnastics at the 1956 Summer Olympics – women's floor"
            },
            {
              "description_en": "men's pommel horse events at the Olympics",
              "id": "Q5624282",
              "label_en": "gymnastics at the 1960 Summer Olympics – men's pommel horse"
            },
            {
              "description_en": "olympic gymnastics event",
              "id": "Q5624284",
              "label_en": "gymnastics at the 1960 Summer Olympics – men's vault"
            },
            {
              "description_en": "Olympic gymnastics event",
              "id": "Q5624313",
              "label_en": "gymnastics at the 1968 Summer Olympics – men's horizontal bar"
            },
            "... omitted 9 items"
          ],
          "P4155": [
            {
              "description_en": "qualifier of award received (P166) to specify the work that an award was given to the creator for",
              "id": "P1686",
              "label_en": "for work"
            },
            {
              "description_en": "qualifier to specify the item that this property is shared with",
              "id": "P1706",
              "label_en": "together with"
            },
            {
              "description_en": "official classification by a regulating body under which the subject (events, teams, participants, or equipment) qualifies for inclusion",
              "id": "P2094",
              "label_en": "competition class"
            },
            {
              "description_en": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
              "id": "P21",
              "label_en": "sex or gender"
            },
            {
              "description_en": "discipline an athlete competed in within a sport",
              "id": "P2416",
              "label_en": "sports discipline competed in"
            },
            {
              "description_en": "number of an edition (first, second, ... as 1, 2, ...) or event",
              "id": "P393",
              "label_en": "edition number"
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
              "description_en": "(Do not use. Find alternatives at WD:P642) qualifier stating that a statement applies within the scope of a particular item",
              "id": "P642",
              "label_en": "of (DEPRECATED)"
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
              "description_en": "coordinated military actions of a state or a non-state actor",
              "id": "Q645883",
              "label_en": "military operation"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "edition in a specific year of an annual music competition, song contest, etc.",
              "id": "Q106594095",
              "label_en": "annual music competition edition"
            },
            {
              "description_en": "event, during which one or more sporting events are held",
              "id": "Q13406554",
              "label_en": "sports competition"
            },
            {
              "description_en": "competition which only exists in a fictional universe",
              "id": "Q15707532",
              "label_en": "fictional competition"
            },
            {
              "description_en": "sports event scheduled to recur within a decided interval",
              "id": "Q18608583",
              "label_en": "recurring sporting event"
            },
            {
              "description_en": "set of episodes produced for a television series",
              "id": "Q3464665",
              "label_en": "television series season"
            },
            {
              "description_en": "rivalry where multiple parties strive for a goal which cannot be shared",
              "id": "Q476300",
              "label_en": "competition"
            },
            {
              "description_en": "video serial broadcast on the Internet",
              "id": "Q526877",
              "label_en": "web series"
            },
            {
              "description_en": "something given to a person or a group of people to recognize their merit or excellence",
              "id": "Q618779",
              "label_en": "award"
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
              "description_en": "competition or sports event won by the subject",
              "id": "P2522",
              "label_en": "competition won"
            }
          ],
          "P6607": [
            {
              "value": "inverse generally not used@en"
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
              "description_en": "publication type, serial publication that appears in a new edition on a regular schedule",
              "id": "Q1002697",
              "label_en": "periodical"
            },
            {
              "description_en": "creative work in which images and text convey information such as narratives",
              "id": "Q1004",
              "label_en": "comics"
            },
            {
              "description_en": "computer designed for playing chess",
              "id": "Q1364192",
              "label_en": "chess computer"
            },
            {
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
            },
            {
              "description_en": "social entity established to meet needs or pursue goals",
              "id": "Q43229",
              "label_en": "organization"
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
              "description_en": "large landmass identified by convention",
              "id": "Q5107",
              "label_en": "continent"
            },
            {
              "description_en": "territorial entity for administration purposes, with or without its own local government",
              "id": "Q56061",
              "label_en": "administrative territorial entity"
            },
            {
              "description_en": "medium for recording information (words or images) typically on bound pages or more abstractly in electronic or audio form",
              "id": "Q571",
              "label_en": "book"
            },
            {
              "description_en": "community of people who share a common language, culture, ethnicity, descent, or history",
              "id": "Q6266",
              "label_en": "nation"
            },
            {
              "description_en": "domesticated four-footed mammal from the equine family",
              "id": "Q726",
              "label_en": "horse"
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
  "hash_after": "543e6e09dcb8cc7824fdf0db30d1c54aa8810398",
  "hash_before": "33ebba8f595b7d9221d8dad0f7114ae058a6f3d5",
  "property_revision_id": 1577277439,
  "property_revision_prev": 1575232987,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q1656682"
      ],
      "constraint_qid": "Q21510865",
      "qualifier_property": "P2308",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21510865",
      "qualifiers": [
        {
          "property_id": "P2308",
          "values": [
            "Q1002697",
            "Q1004",
            "Q1364192",
            "Q386724",
            "Q43229",
            "Q486972",
            "Q5",
            "Q5107",
            "Q56061",
            "Q571",
            "Q6266",
            "Q726",
            "Q95074"
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
      "single-value constraint: exception to constraint: cross-country skiing at the 2002 Winter Olympics – men's 2 x 10 kilometre pursuit, gymnastics at the 1992 Summer Olympics – men's pommel horse, gymnastics at the 1992 Summer Olympics – women's vault, equestrian at the 1900 Summer Olympics – high jump, gymnastics at the 1904 Summer Olympics – men's horizontal bar, gymnastics at the 1988 Summer Olympics – men's horizontal bar, gymnastics at the 1988 Summer Olympics –... [truncated 1737 chars]",
      "conflicts-with constraint: item of property constraint: military operation; property: instance of; constraint status: mandatory constraint",
      "subject type constraint: class: annual music competition edition, sports competition, fictional competition, recurring sporting event, television series season, competition, web series, award; relation: instance or subclass of",
      "inverse constraint: property: competition won; constraint clarification: inverse generally not used@en",
      "value-type constraint: class: periodical, comics, chess computer, event, work, organization, human settlement, human, continent, administrative territorial entity, book, nation, horse, character; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as qualifier"
    ],
    "before": [
      "single-value constraint: exception to constraint: cross-country skiing at the 2002 Winter Olympics – men's 2 x 10 kilometre pursuit, gymnastics at the 1992 Summer Olympics – men's pommel horse, gymnastics at the 1992 Summer Olympics – women's vault, equestrian at the 1900 Summer Olympics – high jump, gymnastics at the 1904 Summer Olympics – men's horizontal bar, gymnastics at the 1988 Summer Olympics – men's horizontal bar, gymnastics at the 1988 Summer Olympics –... [truncated 1737 chars]",
      "conflicts-with constraint: item of property constraint: military operation; property: instance of; constraint status: mandatory constraint",
      "subject type constraint: class: annual music competition edition, sports competition, fictional competition, recurring sporting event, television series season, competition, web series, award; relation: instance or subclass of",
      "inverse constraint: property: competition won; constraint clarification: inverse generally not used@en",
      "value-type constraint: class: periodical, comics, chess computer, work, organization, human settlement, human, continent, administrative territorial entity, book, nation, horse, character; relation: instance of",
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
    "violation_name": "Value type Q|5, Q|43229, Q|6266, Q|5107, Q|726, Q|1002697, Q|1364192, Q|486972, Q|56061, Q|95074, Q|571, Q|1004, Q|386724"
  },
  {
    "result": "Q19474404",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 002. `reform_Q10684978_P159_2438613236`

| Field | Value |
|---|---|
| qid | Q10684978 |
| property | P159 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q21503250 |
| group_key | TBOX::P159::2438613236 |
| tbox_revision_key | TBOX::P159::2438613236 |

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
  "author": "Arlo Barnes",
  "kind": "T_BOX",
  "property_revision_id": 2438613236,
  "property_revision_prev": 2436458585
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-09T13:19:18",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P159",
  "report_revision_new": 2440033122,
  "report_revision_old": 2439570804,
  "report_violation_type": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337",
  "report_violation_type_descriptions_en": [
    "social entity established to meet needs or pursue goals",
    "organization which only appears in works of fiction",
    "scheduled publication containing news of events, articles, features, editorials, and advertising; online, in print, or (usually) both",
    "territorial entity for administration purposes, with or without its own local government",
    "publication type, serial publication that appears in a new edition on a regular schedule",
    "business organization which only exists in fiction",
    "set of related web pages served from a single web domain",
    "legal entity representing an association of people, whether natural, legal or a mixture of both, with a specific objective",
    "organisational part of a government responsible for specific public services, such as health, judiciary, education, transportation, foreign affairs, etc",
    "a designated body with authority",
    "identification for a good or service",
    "transition of power from one president to another",
    "collaborative enterprise, frequently involving research or design, that is carefully planned to achieve a particular aim",
    "organized set of events or activities focused on a theme (cultural, religious or other) that recurs regularly (e.g. once a year) and lasts anywhere from several hours to weeks",
    "type of administrative division, in some countries",
    "series of operations undertaken to achieve a defined goal",
    "temporary and scheduled happening, like a conference, festival, competition or similar",
    "contest for a prize or award",
    "large area with high population density and infrastructure of built environment",
    "organization undertaking commercial, industrial, or professional activity",
    "entity that no longer operates or is terminated",
    "defunct, destroyed, demolished, or discontinued organization, establishment, group, etc.",
    "theatre organization that produces puppetry performances"
  ],
  "report_violation_type_labels_en": [
    "organization",
    "fictional organization",
    "newspaper",
    "administrative territorial entity",
    "periodical",
    "fictional company",
    "website",
    "company",
    "government agency",
    "governing body",
    "brand",
    "presidential transition",
    "project",
    "festival",
    "district",
    "campaign",
    "event",
    "competition",
    "urban area",
    "business",
    "former entity",
    "defunct organization",
    "puppetry company"
  ],
  "report_violation_type_normalized": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337",
  "report_violation_type_qids": [
    "Q43229",
    "Q14623646",
    "Q11032",
    "Q56061",
    "Q1002697",
    "Q5446565",
    "Q35127",
    "Q783794",
    "Q327333",
    "Q895526",
    "Q431289",
    "Q104921473",
    "Q170584",
    "Q132241",
    "Q149621",
    "Q6056746",
    "Q1656682",
    "Q841654",
    "Q702492",
    "Q4830453",
    "Q15893266",
    "Q55097243",
    "Q3982337"
  ],
  "report_violation_type_raw": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337",
  "value": null,
  "value_current_2026": [
    "Q113679"
  ],
  "value_current_2026_descriptions_en": [
    "municipality in Stockholm County, Sweden"
  ],
  "value_current_2026_labels_en": [
    "Danderyd Municipality"
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
    "description": "city or town where an organization's headquarters is or has been situated. Use P276 qualifier for specific building",
    "label": "headquarters location"
  },
  "qid": {
    "description": "specialidrottsförbund",
    "label": "Svenska Curlingförbundet"
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
            "P10046",
            "P10689",
            "P11693",
            "P12487",
            "P12506",
            "P1264",
            "P13096",
            "P131",
            "P1319",
            "P1326",
            "P1329",
            "P1480",
            "P1534",
            "P1545",
            "P1552",
            "P17",
            "P18",
            "P1810",
            "P1932",
            "P2241",
            "P2671",
            "P2699",
            "P276",
            "... omitted 34 items"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 7,
  "author": "Arlo Barnes",
  "before_constraint_count": 7,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
              "description_en": "publication type, serial publication that appears in a new edition on a regular schedule",
              "id": "Q1002697",
              "label_en": "periodical"
            },
            {
              "description_en": "transition of power from one president to another",
              "id": "Q104921473",
              "label_en": "presidential transition"
            },
            {
              "description_en": "scheduled publication containing news of events, articles, features, editorials, and advertising; online, in print, or (usually) both",
              "id": "Q11032",
              "label_en": "newspaper"
            },
            {
              "description_en": "organized set of events or activities focused on a theme (cultural, religious or other) that recurs regularly (e.g. once a year) and lasts anywhere from several hours to weeks",
              "id": "Q132241",
              "label_en": "festival"
            },
            {
              "description_en": "organization which only appears in works of fiction",
              "id": "Q14623646",
              "label_en": "fictional organization"
            },
            {
              "description_en": "type of administrative division, in some countries",
              "id": "Q149621",
              "label_en": "district"
            },
            {
              "description_en": "entity that no longer operates or is terminated",
              "id": "Q15893266",
              "label_en": "former entity"
            },
            {
              "description_en": "temporary and scheduled happening, like a conference, festival, competition or similar",
              "id": "Q1656682",
              "label_en": "event"
            },
            {
              "description_en": "collaborative enterprise, frequently involving research or design, that is carefully planned to achieve a particular aim",
              "id": "Q170584",
              "label_en": "project"
            },
            {
              "description_en": "organisational part of a government responsible for specific public services, such as health, judiciary, education, transportation, foreign affairs, etc",
              "id": "Q327333",
              "label_en": "government agency"
            },
            {
              "description_en": "set of related web pages served from a single web domain",
              "id": "Q35127",
              "label_en": "website"
            },
            {
              "description_en": "theatre organization that produces puppetry performances",
              "id": "Q3982337",
              "label_en": "puppetry company"
            },
            {
              "description_en": "identification for a good or service",
              "id": "Q431289",
              "label_en": "brand"
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
              "description_en": "business organization which only exists in fiction",
              "id": "Q5446565",
              "label_en": "fictional company"
            },
            {
              "description_en": "defunct, destroyed, demolished, or discontinued organization, establishment, group, etc.",
              "id": "Q55097243",
              "label_en": "defunct organization"
            },
            {
              "description_en": "territorial entity for administration purposes, with or without its own local government",
              "id": "Q56061",
              "label_en": "administrative territorial entity"
            },
            {
              "description_en": "series of operations undertaken to achieve a defined goal",
              "id": "Q6056746",
              "label_en": "campaign"
            },
            {
              "description_en": "large area with high population density and infrastructure of built environment",
              "id": "Q702492",
              "label_en": "urban area"
            },
            {
              "description_en": "legal entity representing an association of people, whether natural, legal or a mixture of both, with a specific objective",
              "id": "Q783794",
              "label_en": "company"
            },
            {
              "description_en": "contest for a prize or award",
              "id": "Q841654",
              "label_en": "competition"
            },
            {
              "description_en": "a designated body with authority",
              "id": "Q895526",
              "label_en": "governing body"
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
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "identifier of a place in Apple Maps (either legacy numeric AUID or hex “place-id”)",
              "id": "P10046",
              "label_en": "Apple Maps ID"
            },
            {
              "description_en": "identifier for a way in OpenStreetMap",
              "id": "P10689",
              "label_en": "OpenStreetMap way ID"
            },
            {
              "description_en": "ID of a node in OpenStreetMap for the item",
              "id": "P11693",
              "label_en": "OpenStreetMap node ID"
            },
            {
              "description_en": "identifier 2GIS of a place",
              "id": "P12487",
              "label_en": "2GIS place-ID"
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
              "description_en": "identifier for an organization listed in the Yandex Rubricator",
              "id": "P13096",
              "label_en": "Yandex Maps organization ID"
            },
            {
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
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
              "description_en": "telephone number in standard format (RFC3966), without “tel:” prefix",
              "id": "P1329",
              "label_en": "phone number"
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
              "description_en": "position of an item in its parent series (most frequently a 1-based index), generally to be used as a qualifier (different from \"rank\" defined as a class, and from \"ranking\" defined as a property for evaluating a quality)",
              "id": "P1545",
              "label_en": "series ordinal"
            },
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
            {
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
            },
            {
              "description_en": "image of relevant illustration of the subject; if available, also use more specific properties (sample: coat of arms image, locator map, flag image, signature image, logo image, collage image)",
              "id": "P18",
              "label_en": "image"
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
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "identifier for Google Knowledge Graph API, starting with \"/g/\". For IDs starting with \"/m/\", use Freebase ID (P646)",
              "id": "P2671",
              "label_en": "Google Knowledge Graph ID"
            },
            {
              "description_en": "location of a resource",
              "id": "P2699",
              "label_en": "URL"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            "... omitted 34 items"
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
              "description_en": "territorial entity of which the borders are determined by physiographic and human features",
              "id": "Q15642541",
              "label_en": "human-geographic territorial entity"
            },
            {
              "description_en": "point, line or area on or near Earth",
              "id": "Q2221906",
              "label_en": "geographic location"
            },
            {
              "description_en": "place that exists only in fiction and not in reality",
              "id": "Q3895768",
              "label_en": "fictional location"
            },
            {
              "description_en": "mobile machine used for transport, whether it has an engine or not, including wheeled and tracked vehicles, air-, water-, and space-craft",
              "id": "Q42889",
              "label_en": "vehicle"
            },
            {
              "description_en": "organization established by treaty between governments",
              "id": "Q484652",
              "label_en": "international organization"
            },
            {
              "description_en": "components of planets that can be geographically located",
              "id": "Q618123",
              "label_en": "geographical feature"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
              "description_en": "publication type, serial publication that appears in a new edition on a regular schedule",
              "id": "Q1002697",
              "label_en": "periodical"
            },
            {
              "description_en": "transition of power from one president to another",
              "id": "Q104921473",
              "label_en": "presidential transition"
            },
            {
              "description_en": "scheduled publication containing news of events, articles, features, editorials, and advertising; online, in print, or (usually) both",
              "id": "Q11032",
              "label_en": "newspaper"
            },
            {
              "description_en": "organized set of events or activities focused on a theme (cultural, religious or other) that recurs regularly (e.g. once a year) and lasts anywhere from several hours to weeks",
              "id": "Q132241",
              "label_en": "festival"
            },
            {
              "description_en": "organization which only appears in works of fiction",
              "id": "Q14623646",
              "label_en": "fictional organization"
            },
            {
              "description_en": "type of administrative division, in some countries",
              "id": "Q149621",
              "label_en": "district"
            },
            {
              "description_en": "entity that no longer operates or is terminated",
              "id": "Q15893266",
              "label_en": "former entity"
            },
            {
              "description_en": "temporary and scheduled happening, like a conference, festival, competition or similar",
              "id": "Q1656682",
              "label_en": "event"
            },
            {
              "description_en": "collaborative enterprise, frequently involving research or design, that is carefully planned to achieve a particular aim",
              "id": "Q170584",
              "label_en": "project"
            },
            {
              "description_en": "organisational part of a government responsible for specific public services, such as health, judiciary, education, transportation, foreign affairs, etc",
              "id": "Q327333",
              "label_en": "government agency"
            },
            {
              "description_en": "set of related web pages served from a single web domain",
              "id": "Q35127",
              "label_en": "website"
            },
            {
              "description_en": "theatre organization that produces puppetry performances",
              "id": "Q3982337",
              "label_en": "puppetry company"
            },
            {
              "description_en": "identification for a good or service",
              "id": "Q431289",
              "label_en": "brand"
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
              "description_en": "business organization which only exists in fiction",
              "id": "Q5446565",
              "label_en": "fictional company"
            },
            {
              "description_en": "defunct, destroyed, demolished, or discontinued organization, establishment, group, etc.",
              "id": "Q55097243",
              "label_en": "defunct organization"
            },
            {
              "description_en": "territorial entity for administration purposes, with or without its own local government",
              "id": "Q56061",
              "label_en": "administrative territorial entity"
            },
            {
              "description_en": "series of operations undertaken to achieve a defined goal",
              "id": "Q6056746",
              "label_en": "campaign"
            },
            {
              "description_en": "large area with high population density and infrastructure of built environment",
              "id": "Q702492",
              "label_en": "urban area"
            },
            {
              "description_en": "legal entity representing an association of people, whether natural, legal or a mixture of both, with a specific objective",
              "id": "Q783794",
              "label_en": "company"
            },
            {
              "description_en": "contest for a prize or award",
              "id": "Q841654",
              "label_en": "competition"
            },
            {
              "description_en": "a designated body with authority",
              "id": "Q895526",
              "label_en": "governing body"
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
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "identifier of a place in Apple Maps (either legacy numeric AUID or hex “place-id”)",
              "id": "P10046",
              "label_en": "Apple Maps ID"
            },
            {
              "description_en": "identifier for a way in OpenStreetMap",
              "id": "P10689",
              "label_en": "OpenStreetMap way ID"
            },
            {
              "description_en": "ID of a node in OpenStreetMap for the item",
              "id": "P11693",
              "label_en": "OpenStreetMap node ID"
            },
            {
              "description_en": "identifier 2GIS of a place",
              "id": "P12487",
              "label_en": "2GIS place-ID"
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
              "description_en": "identifier for an organization listed in the Yandex Rubricator",
              "id": "P13096",
              "label_en": "Yandex Maps organization ID"
            },
            {
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
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
              "description_en": "telephone number in standard format (RFC3966), without “tel:” prefix",
              "id": "P1329",
              "label_en": "phone number"
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
              "description_en": "position of an item in its parent series (most frequently a 1-based index), generally to be used as a qualifier (different from \"rank\" defined as a class, and from \"ranking\" defined as a property for evaluating a quality)",
              "id": "P1545",
              "label_en": "series ordinal"
            },
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
            {
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
            },
            {
              "description_en": "image of relevant illustration of the subject; if available, also use more specific properties (sample: coat of arms image, locator map, flag image, signature image, logo image, collage image)",
              "id": "P18",
              "label_en": "image"
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
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
            },
            {
              "description_en": "identifier for Google Knowledge Graph API, starting with \"/g/\". For IDs starting with \"/m/\", use Freebase ID (P646)",
              "id": "P2671",
              "label_en": "Google Knowledge Graph ID"
            },
            {
              "description_en": "location of a resource",
              "id": "P2699",
              "label_en": "URL"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            "... omitted 33 items"
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
              "description_en": "territorial entity of which the borders are determined by physiographic and human features",
              "id": "Q15642541",
              "label_en": "human-geographic territorial entity"
            },
            {
              "description_en": "point, line or area on or near Earth",
              "id": "Q2221906",
              "label_en": "geographic location"
            },
            {
              "description_en": "place that exists only in fiction and not in reality",
              "id": "Q3895768",
              "label_en": "fictional location"
            },
            {
              "description_en": "mobile machine used for transport, whether it has an engine or not, including wheeled and tracked vehicles, air-, water-, and space-craft",
              "id": "Q42889",
              "label_en": "vehicle"
            },
            {
              "description_en": "organization established by treaty between governments",
              "id": "Q484652",
              "label_en": "international organization"
            },
            {
              "description_en": "components of planets that can be geographically located",
              "id": "Q618123",
              "label_en": "geographical feature"
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
  "hash_after": "c9ee80954d6bab31b8d9166eba86ae29f8ec4a36",
  "hash_before": "299c5fe622eb5253084572fbe0e5a2bd6c3b94af",
  "property_revision_id": 2438613236,
  "property_revision_prev": 2436458585,
  "qualifier_value_changes": [
    {
      "added_values": [
        "P805"
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
            "P10046",
            "P10689",
            "P11693",
            "P12487",
            "P12506",
            "P1264",
            "P13096",
            "P131",
            "P1319",
            "P1326",
            "P1329",
            "P1480",
            "P1534",
            "P1545",
            "P1552",
            "P17",
            "P18",
            "P1810",
            "P1932",
            "P2241",
            "P2671",
            "P2699",
            "P276",
            "... omitted 33 items"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "item-requires-statement constraint: property: country",
      "subject type constraint: class: periodical, presidential transition, newspaper, festival, fictional organization, district, former entity, event, project, government agency, website, puppetry company, brand, organization, business, fictional company, defunct organization, administrative territorial entity, campaign, urban area, company, competition, governing body; relation: instance of",
      "allowed qualifiers constraint: property: applies to jurisdiction, Apple Maps ID, OpenStreetMap way ID, OpenStreetMap node ID, 2GIS place-ID, latest end date, valid in period, Yandex Maps organization ID, located in the administrative territorial entity, earliest date, latest date, phone number, sourcing circumstances, end cause, series ordinal, has characteristic, country, image, subject named as, object named as, reason for deprecated rank, Google Knowledge Graph... [truncated 702 chars]",
      "value-type constraint: class: human-geographic territorial entity, geographic location, fictional location, vehicle, international organization, geographical feature; relation: instance of",
      "contemporary constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "property scope constraint: property scope: as main value, as qualifier"
    ],
    "before": [
      "item-requires-statement constraint: property: country",
      "subject type constraint: class: periodical, presidential transition, newspaper, festival, fictional organization, district, former entity, event, project, government agency, website, puppetry company, brand, organization, business, fictional company, defunct organization, administrative territorial entity, campaign, urban area, company, competition, governing body; relation: instance of",
      "allowed qualifiers constraint: property: applies to jurisdiction, Apple Maps ID, OpenStreetMap way ID, OpenStreetMap node ID, 2GIS place-ID, latest end date, valid in period, Yandex Maps organization ID, located in the administrative territorial entity, earliest date, latest date, phone number, sourcing circumstances, end cause, series ordinal, has characteristic, country, image, subject named as, object named as, reason for deprecated rank, Google Knowledge Graph... [truncated 677 chars]",
      "value-type constraint: class: human-geographic territorial entity, geographic location, fictional location, vehicle, international organization, geographical feature; relation: instance of",
      "contemporary constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item; constraint status: mandatory constraint",
      "property scope constraint: property scope: as main value, as qualifier"
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
    "violation_name": "Type Q|43229, Q|14623646, Q|11032, Q|56061, Q|1002697, Q|5446565, Q|35127, Q|783794, Q|327333, Q|895526, Q|431289, Q|104921473, Q|170584, Q|132241, Q|149621, Q|6056746, Q|1656682, Q|841654, Q|702492, Q|4830453, Q|15893266, Q|55097243, Q|3982337"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 003. `reform_Q133308443_P535_2443240310`

| Field | Value |
|---|---|
| qid | Q133308443 |
| property | P535 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | TBOX::P535::2443240310 |
| tbox_revision_key | TBOX::P535::2443240310 |

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
  "author": "Jerimee",
  "kind": "T_BOX",
  "property_revision_id": 2443240310,
  "property_revision_prev": 2403307533
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-17T11:35:54",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P535",
  "report_revision_new": 2443385706,
  "report_revision_old": 2442968833,
  "report_violation_type": "Item P|21",
  "report_violation_type_normalized": "Item P|21",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|21",
  "value": null,
  "value_current_2026": [
    "89331656"
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
    "description": "identifier of an individual's burial place in the Find a Grave database",
    "label": "Find a Grave memorial ID"
  },
  "qid": {
    "description": "photographer, part of JAG",
    "label": "Gary Grether"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 11,
  "author": "Jerimee",
  "before_constraint_count": 12,
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
              "value": "en"
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
              "description_en": "location of grave, resting place, place of ash-scattering, etc. (e.g., town/city or cemetery) for a person or animal. There may be several places: e.g., re-burials, parts of body buried separately.",
              "id": "P119",
              "label_en": "place of burial"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
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
          "P2303": [
            {
              "description_en": "sculpture by Bertel Thorvaldsen in Lucerne (CH)",
              "id": "Q688214",
              "label_en": "Lion Monument Lucerne"
            }
          ],
          "P2306": [
            {
              "description_en": "location of grave, resting place, place of ash-scattering, etc. (e.g., town/city or cemetery) for a person or animal. There may be several places: e.g., re-burials, parts of body buried separately.",
              "id": "P119",
              "label_en": "place of burial"
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
          "P2303": [
            {
              "description_en": "sculpture by Bertel Thorvaldsen in Lucerne (CH)",
              "id": "Q688214",
              "label_en": "Lion Monument Lucerne"
            }
          ],
          "P2306": [
            {
              "description_en": "date on which the subject was born",
              "id": "P569",
              "label_en": "date of birth"
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
          "P2303": [
            {
              "description_en": "sculpture by Bertel Thorvaldsen in Lucerne (CH)",
              "id": "Q688214",
              "label_en": "Lion Monument Lucerne"
            }
          ],
          "P2306": [
            {
              "description_en": "date on which the subject died",
              "id": "P570",
              "label_en": "date of death"
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
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "being that has certain capacities or attributes constituting personhood (for humans, use Q5 [human] with P31 [instance of])",
              "id": "Q215627",
              "label_en": "person"
            },
            {
              "description_en": "twins physically joined in utero",
              "id": "Q216866",
              "label_en": "conjoined twins"
            },
            {
              "description_en": "species in the oceanic dolphin family",
              "id": "Q26843",
              "label_en": "Orcinus orca"
            },
            {
              "description_en": "genus of reptiles",
              "id": "Q288720",
              "label_en": "horned lizard"
            },
            {
              "description_en": "empty tomb or monument erected in honor of a person whose remains are lost or interred elsewhere",
              "id": "Q321053",
              "label_en": "cenotaph"
            },
            {
              "description_en": "genus of large African apes",
              "id": "Q36611",
              "label_en": "Gorilla"
            },
            {
              "description_en": "species of mammal; species of ape",
              "id": "Q4126704",
              "label_en": "chimpanzee"
            },
            {
              "description_en": "imposing structure created to commemorate a person or event, or used for that purpose",
              "id": "Q4989906",
              "label_en": "monument"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            {
              "description_en": "area or object, smaller than a monument, which serves as a focus for memory of something",
              "id": "Q5003624",
              "label_en": "memorial"
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
              "description_en": "domesticated omnivorous even-toed ungulate",
              "id": "Q787",
              "label_en": "pig"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "URL to the archived web page specified with URL property",
              "id": "P1065",
              "label_en": "archive URL"
            },
            {
              "description_en": "location of grave, resting place, place of ash-scattering, etc. (e.g., town/city or cemetery) for a person or animal. There may be several places: e.g., re-burials, parts of body buried separately.",
              "id": "P119",
              "label_en": "place of burial"
            },
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
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
              "description_en": "location of a resource",
              "id": "P2699",
              "label_en": "URL"
            },
            {
              "description_en": "date a document was archived",
              "id": "P2960",
              "label_en": "archive date"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "date a reference was modified, revised, or updated",
              "id": "P5017",
              "label_en": "last update"
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
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
          "description_en": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "id": "Q108139345",
          "label_en": "label in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "en"
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
              "description_en": "location of grave, resting place, place of ash-scattering, etc. (e.g., town/city or cemetery) for a person or animal. There may be several places: e.g., re-burials, parts of body buried separately.",
              "id": "P119",
              "label_en": "place of burial"
            },
            {
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
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
          "P2303": [
            {
              "description_en": "sculpture by Bertel Thorvaldsen in Lucerne (CH)",
              "id": "Q688214",
              "label_en": "Lion Monument Lucerne"
            }
          ],
          "P2306": [
            {
              "description_en": "location of grave, resting place, place of ash-scattering, etc. (e.g., town/city or cemetery) for a person or animal. There may be several places: e.g., re-burials, parts of body buried separately.",
              "id": "P119",
              "label_en": "place of burial"
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
          "P2303": [
            {
              "description_en": "sculpture by Bertel Thorvaldsen in Lucerne (CH)",
              "id": "Q688214",
              "label_en": "Lion Monument Lucerne"
            }
          ],
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "sculpture by Bertel Thorvaldsen in Lucerne (CH)",
              "id": "Q688214",
              "label_en": "Lion Monument Lucerne"
            }
          ],
          "P2306": [
            {
              "description_en": "date on which the subject was born",
              "id": "P569",
              "label_en": "date of birth"
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
          "P2303": [
            {
              "description_en": "sculpture by Bertel Thorvaldsen in Lucerne (CH)",
              "id": "Q688214",
              "label_en": "Lion Monument Lucerne"
            }
          ],
          "P2306": [
            {
              "description_en": "date on which the subject died",
              "id": "P570",
              "label_en": "date of death"
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
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "being that has certain capacities or attributes constituting personhood (for humans, use Q5 [human] with P31 [instance of])",
              "id": "Q215627",
              "label_en": "person"
            },
            {
              "description_en": "twins physically joined in utero",
              "id": "Q216866",
              "label_en": "conjoined twins"
            },
            {
              "description_en": "species in the oceanic dolphin family",
              "id": "Q26843",
              "label_en": "Orcinus orca"
            },
            {
              "description_en": "genus of reptiles",
              "id": "Q288720",
              "label_en": "horned lizard"
            },
            {
              "description_en": "empty tomb or monument erected in honor of a person whose remains are lost or interred elsewhere",
              "id": "Q321053",
              "label_en": "cenotaph"
            },
            {
              "description_en": "genus of large African apes",
              "id": "Q36611",
              "label_en": "Gorilla"
            },
            {
              "description_en": "species of mammal; species of ape",
              "id": "Q4126704",
              "label_en": "chimpanzee"
            },
            {
              "description_en": "imposing structure created to commemorate a person or event, or used for that purpose",
              "id": "Q4989906",
              "label_en": "monument"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            {
              "description_en": "area or object, smaller than a monument, which serves as a focus for memory of something",
              "id": "Q5003624",
              "label_en": "memorial"
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
              "description_en": "domesticated omnivorous even-toed ungulate",
              "id": "Q787",
              "label_en": "pig"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "URL to the archived web page specified with URL property",
              "id": "P1065",
              "label_en": "archive URL"
            },
            {
              "description_en": "location of grave, resting place, place of ash-scattering, etc. (e.g., town/city or cemetery) for a person or animal. There may be several places: e.g., re-burials, parts of body buried separately.",
              "id": "P119",
              "label_en": "place of burial"
            },
            {
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
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
              "description_en": "location of a resource",
              "id": "P2699",
              "label_en": "URL"
            },
            {
              "description_en": "date a document was archived",
              "id": "P2960",
              "label_en": "archive date"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "date a reference was modified, revised, or updated",
              "id": "P5017",
              "label_en": "last update"
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
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
  "hash_after": "739144e1e8759a54ba6450c00d75cd085d5983dd",
  "hash_before": "89eddd957695712cb532194c4dc94333b31f33af",
  "property_revision_id": 2443240310,
  "property_revision_prev": 2403307533,
  "qualifier_value_changes": [
    {
      "added_values": [
        "P569"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2306",
      "removed_values": [
        "P21"
      ],
      "same_qid_index": 1
    },
    {
      "added_values": [
        "P570"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2306",
      "removed_values": [
        "P569"
      ],
      "same_qid_index": 2
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21503247",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q688214"
          ]
        },
        {
          "property_id": "P2306",
          "values": [
            "P21"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "label in language constraint: Wikimedia language code: en",
      "single-value constraint: separator: place of burial, subject named as, applies to part",
      "format constraint: format as a regular expression: [1-9]\\d*; constraint status: mandatory constraint",
      "distinct-values constraint: constraint status: mandatory constraint",
      "item-requires-statement constraint: exception to constraint: Lion Monument Lucerne; property: place of burial; constraint status: suggestion constraint",
      "item-requires-statement constraint: exception to constraint: Lion Monument Lucerne; property: date of birth",
      "item-requires-statement constraint: exception to constraint: Lion Monument Lucerne; property: date of death",
      "subject type constraint: class: fictional human, group of humans, person, conjoined twins, Orcinus orca, horned lizard, cenotaph, Gorilla, chimpanzee, monument, human, memorial, horse, Animalia, pig; relation: instance of; constraint status: mandatory constraint",
      "allowed qualifiers constraint: property: archive URL, place of burial, has characteristic, subject named as, reason for deprecated rank, URL, archive date, object of statement has role, last update, applies to part, reason for preferred rank",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ],
    "before": [
      "label in language constraint: Wikimedia language code: en",
      "single-value constraint: separator: place of burial, subject named as, applies to part",
      "format constraint: format as a regular expression: [1-9]\\d*; constraint status: mandatory constraint",
      "distinct-values constraint: constraint status: mandatory constraint",
      "item-requires-statement constraint: exception to constraint: Lion Monument Lucerne; property: place of burial; constraint status: suggestion constraint",
      "item-requires-statement constraint: exception to constraint: Lion Monument Lucerne; property: sex or gender",
      "item-requires-statement constraint: exception to constraint: Lion Monument Lucerne; property: date of birth",
      "item-requires-statement constraint: exception to constraint: Lion Monument Lucerne; property: date of death",
      "subject type constraint: class: fictional human, group of humans, person, conjoined twins, Orcinus orca, horned lizard, cenotaph, Gorilla, chimpanzee, monument, human, memorial, horse, Animalia, pig; relation: instance of; constraint status: mandatory constraint",
      "allowed qualifiers constraint: property: archive URL, place of burial, has characteristic, subject named as, reason for deprecated rank, URL, archive date, object of statement has role, last update, applies to part, reason for preferred rank",
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
    "violation_name": "Item P|21"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 004. `reform_Q135514163_P5236_2387473038`

| Field | Value |
|---|---|
| qid | Q135514163 |
| property | P5236 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | TBOX::P5236::2387473038 |
| tbox_revision_key | TBOX::P5236::2387473038 |

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
  "author": "GZWDer",
  "kind": "T_BOX",
  "property_revision_id": 2387473038,
  "property_revision_prev": 2387464094
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-08-07T09:26:15",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P5236",
  "report_revision_new": 2388961222,
  "report_revision_old": 2388528441,
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
  "value": null,
  "value_current_2026": [
    "Q201"
  ],
  "value_current_2026_descriptions_en": [
    "natural number"
  ],
  "value_current_2026_labels_en": [
    "3"
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
    "description": "one of the prime numbers that can be multiplied to give this number",
    "label": "prime factor"
  },
  "qid": {
    "description": "natural number",
    "label": "21159"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 9,
  "author": "GZWDer",
  "before_constraint_count": 10,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "numerical value of a number, a mathematical constant, or a physical constant",
              "id": "P1181",
              "label_en": "numeric value"
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
              "description_en": "number of decimal digits of a natural number",
              "id": "P7316",
              "label_en": "number of decimal digits"
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
              "description_en": "integer greater than zero; natural number explicitly excluding zero",
              "id": "Q28920044",
              "label_en": "positive integer"
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
              "description_en": "number of instances of this subject in the universe of the subject (the actual number of instances in Wikidata may be lower or higher)",
              "id": "P1114",
              "label_en": "quantity"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
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
          "P2306": [
            {
              "description_en": "one of the prime numbers that can be multiplied to give this number",
              "id": "P5236",
              "label_en": "prime factor"
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
              "description_en": "positive integer with exactly two divisors, 1 and itself",
              "id": "Q49008",
              "label_en": "prime number"
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
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "natural number one",
              "id": "Q199",
              "label_en": "Акы"
            }
          ],
          "P6607": [
            {
              "value": "1（Q199）不是質數（Q49008）@zh-hant"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "numerical value of a number, a mathematical constant, or a physical constant",
              "id": "P1181",
              "label_en": "numeric value"
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
              "description_en": "number of decimal digits of a natural number",
              "id": "P7316",
              "label_en": "number of decimal digits"
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
              "description_en": "integer greater than zero; natural number explicitly excluding zero",
              "id": "Q28920044",
              "label_en": "positive integer"
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
              "description_en": "number of instances of this subject in the universe of the subject (the actual number of instances in Wikidata may be lower or higher)",
              "id": "P1114",
              "label_en": "quantity"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
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
          "P2306": [
            {
              "description_en": "one of the prime numbers that can be multiplied to give this number",
              "id": "P5236",
              "label_en": "prime factor"
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
              "description_en": "positive integer with exactly two divisors, 1 and itself",
              "id": "Q49008",
              "label_en": "prime number"
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
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "natural number",
              "id": "Q19108",
              "label_en": "bederatzi"
            },
            {
              "description_en": "natural number",
              "id": "Q202",
              "label_en": "4"
            },
            {
              "description_en": "natural number",
              "id": "Q23355",
              "label_en": "8"
            },
            {
              "description_en": "natural number",
              "id": "Q23488",
              "label_en": "6"
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
              "description_en": "natural number one",
              "id": "Q199",
              "label_en": "Акы"
            }
          ],
          "P6607": [
            {
              "value": "1（Q199）不是質數（Q49008）@zh-hant"
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
  "hash_after": "5010f0fa9be5d5a958f2c4a385c63467b7a0975f",
  "hash_before": "62f85bce25939cdc2321cd2a55f70c8ac56f2825",
  "property_revision_id": 2387473038,
  "property_revision_prev": 2387464094,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q199"
      ],
      "constraint_qid": "Q52558054",
      "qualifier_property": "P2305",
      "removed_values": [
        "Q19108",
        "Q202",
        "Q23355",
        "Q23488"
      ],
      "same_qid_index": 0
    },
    {
      "added_values": [
        "1（Q199）不是質數（Q49008）@zh-hant"
      ],
      "constraint_qid": "Q52558054",
      "qualifier_property": "P6607",
      "removed_values": [],
      "same_qid_index": 0
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
            "Q19108",
            "Q202",
            "Q23355",
            "Q23488"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "item-requires-statement constraint: property: numeric value",
      "item-requires-statement constraint: property: number of decimal digits",
      "subject type constraint: class: positive integer; relation: instance of",
      "allowed qualifiers constraint: property: quantity, reason for deprecated rank",
      "value-requires-statement constraint: property: prime factor",
      "value-type constraint: class: prime number; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "none-of constraint: item of property constraint: Акы; constraint clarification: 1（Q199）不是質數（Q49008）@zh-hant",
      "property scope constraint: property scope: as main value"
    ],
    "before": [
      "item-requires-statement constraint: property: numeric value",
      "item-requires-statement constraint: property: number of decimal digits",
      "subject type constraint: class: positive integer; relation: instance of",
      "allowed qualifiers constraint: property: quantity, reason for deprecated rank",
      "value-requires-statement constraint: property: prime factor",
      "value-type constraint: class: prime number; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "none-of constraint: item of property constraint: bederatzi, 4, 8, 6",
      "none-of constraint: item of property constraint: Акы; constraint clarification: 1（Q199）不是質數（Q49008）@zh-hant",
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
    "violation_name": "Value type Q|49008"
  },
  {
    "result": "Q21503247",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 005. `reform_Q13561329_P660_696618469`

| Field | Value |
|---|---|
| qid | Q13561329 |
| property | P660 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | TBOX::P660::696618469 |
| tbox_revision_key | TBOX::P660::696618469 |

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
  "author": "MisterSynergy",
  "kind": "T_BOX",
  "property_revision_id": 696618469,
  "property_revision_prev": 690339167
}
```

### Violation Context

```json
{
  "report_fix_date": "2018-06-19T21:59:23",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P660",
  "report_revision_new": 698776743,
  "report_revision_old": 698429836,
  "report_violation_type": "Value type Q|Q8047",
  "report_violation_type_descriptions_en": [
    "large biological molecule that acts as a catalyst"
  ],
  "report_violation_type_labels_en": [
    "enzyme"
  ],
  "report_violation_type_normalized": "Value type Q|Q8047",
  "report_violation_type_qids": [
    "Q8047"
  ],
  "report_violation_type_raw": "Value type Q|Q8047",
  "report_violation_types": [
    "Value type Q|Q8047",
    "Target required claim P|P591"
  ],
  "value": null,
  "value_current_2026": [
    "Q420032"
  ],
  "value_current_2026_descriptions_en": [
    "class of enzymes"
  ],
  "value_current_2026_labels_en": [
    "serine endopeptidase"
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
    "description": "the Enzyme Commission (EC)-based accepted name of any enzyme classifications of the protein or RNA molecule",
    "label": "EC enzyme classification"
  },
  "qid": {
    "description": "mammalian protein found in Homo sapiens",
    "label": "reelin"
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
            "Q54828448"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "polymer produced by a living organism",
              "id": "Q422649",
              "label_en": "biopolymer"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item should have a statement with a given property",
          "id": "Q21510864",
          "label_en": "value-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "classification scheme for enzymes",
              "id": "P591",
              "label_en": "EC enzyme number"
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
              "description_en": "large biological molecule that acts as a catalyst",
              "id": "Q8047",
              "label_en": "enzyme"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that a property can only be used as a property for values, not as a qualifier or reference",
          "id": "Q21528958",
          "label_en": "used for values only constraint"
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
              "description_en": "polymer produced by a living organism",
              "id": "Q422649",
              "label_en": "biopolymer"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced item should have a statement with a given property",
          "id": "Q21510864",
          "label_en": "value-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "classification scheme for enzymes",
              "id": "P591",
              "label_en": "EC enzyme number"
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
              "description_en": "large biological molecule that acts as a catalyst",
              "id": "Q8047",
              "label_en": "enzyme"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that a property can only be used as a property for values, not as a qualifier or reference",
          "id": "Q21528958",
          "label_en": "used for values only constraint"
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
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "4eec503d4342ad057650590b6c5f101e04c90af9",
  "hash_before": "39f837e11ab918a46639856f8a2ed7749cd7d854",
  "property_revision_id": 696618469,
  "property_revision_prev": 690339167,
  "qualifier_value_changes": [
    {
      "added_values": [],
      "constraint_qid": "Q53869507",
      "qualifier_property": "P4680",
      "removed_values": [
        "Q46466787"
      ],
      "same_qid_index": 0
    },
    {
      "added_values": [
        "Q54828448"
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
            "Q46466787"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "subject type constraint: class: biopolymer; relation: instance of",
      "value-requires-statement constraint: property: EC enzyme number",
      "value-type constraint: class: enzyme; relation: instance of",
      "used for values only constraint: no qualifiers recorded",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value"
    ],
    "before": [
      "subject type constraint: class: biopolymer; relation: instance of",
      "value-requires-statement constraint: property: EC enzyme number",
      "value-type constraint: class: enzyme; relation: instance of",
      "used for values only constraint: no qualifiers recorded",
      "property scope constraint: constraint status: mandatory constraint; constraint scope: constraint checked on main value"
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
    "violation_name": "Value type Q|Q8047"
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
    "result": "SCHEMA_UPDATE",
    "step": "set_semantics"
  }
]
```

---

## 006. `reform_Q136214804_P921_2447527451`

| Field | Value |
|---|---|
| qid | Q136214804 |
| property | P921 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q21503250 |
| group_key | TBOX::P921::2447527451 |
| tbox_revision_key | TBOX::P921::2447527451 |

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
  "author": "YotaMoteuchi",
  "kind": "T_BOX",
  "property_revision_id": 2447527451,
  "property_revision_prev": 2438998167
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T11:28:22",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
  "report_revision_new": 2447751779,
  "report_revision_old": 2447354743,
  "report_violation_type": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
  "report_violation_type_descriptions_en": [
    "intellectual or artistic creation",
    "any set of human beings",
    "reference in one place in a book to information at another place in the same work",
    "category of creative works based on stylistic, thematic or technical criteria",
    "temporary and scheduled happening, like a conference, festival, competition or similar",
    "program of study, or unit of teaching that typically lasts one academic term",
    "intangible asset consisting of ownership of ideas and processes",
    "theme or subject in a work of art",
    "a restaurant based around a concept or intellectual property",
    "non-repayable funds disbursed by one party to a recipient",
    "human subject research in medicine",
    "process that attempts to determine the facts of a crime and circumstances",
    "single content rating in a rating system",
    "transgression or alleged transgression resulting in public outrage",
    "disclosure of confidential or nonpublic information to unauthorized parties",
    "collection of materials with some unifying characteristic, housed in an archive",
    "set of purposely gathered physical or digital objects with some common characteristics",
    "experience of intense sexual arousal to atypical objects, situations, or individuals",
    "section of learning or teaching into which a wider learning content is divided",
    "process that has the aim of augmenting knowledge, resolving doubt, or solving a problem",
    "word or an unspaced phrase prefixed with the number sign, used to categorise a topic",
    "project of one or more scientists, or of an organization in a scientific field",
    "scientific procedure carried out to support, refute, or validate a hypothesis",
    "topic viewed from a historical point of view",
    "... omitted 62 items"
  ],
  "report_violation_type_labels_en": [
    "work",
    "group of humans",
    "cross-reference",
    "genre",
    "event",
    "course",
    "intellectual property",
    "artistic theme",
    "theme restaurant",
    "grant",
    "clinical trial",
    "criminal investigation",
    "content rating category",
    "scandal",
    "information leak",
    "archival collection",
    "collection",
    "paraphilia",
    "lesson",
    "inquiry",
    "#",
    "science project",
    "experiment",
    "aspect of history",
    "... omitted 62 items"
  ],
  "report_violation_type_normalized": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
  "report_violation_type_qids": [
    "Q386724",
    "Q16334295",
    "Q1302249",
    "Q483394",
    "Q1656682",
    "Q600134",
    "Q131257",
    "Q1406161",
    "Q676586",
    "Q230788",
    "Q30612",
    "Q1964968",
    "Q23649976",
    "Q192909",
    "Q2904148",
    "Q9388534",
    "Q2668072",
    "Q178059",
    "Q379833",
    "Q21004260",
    "Q278485",
    "Q1298668",
    "Q101965",
    "Q17524420",
    "... omitted 62 items"
  ],
  "report_violation_type_raw": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]",
  "value": null,
  "value_current_2026": [
    "Q17172850"
  ],
  "value_current_2026_descriptions_en": [
    "human voice as musical instrument"
  ],
  "value_current_2026_labels_en": [
    "voice"
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
    "description": "primary topic of a work or act of communication",
    "label": "main subject"
  },
  "qid": {
    "description": null,
    "label": "SOVT Exercises"
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
      "constraint_qid": "Q21503250",
      "qualifiers": [
        {
          "property_id": "P2308",
          "values": [
            "Q101965",
            "Q102345381",
            "Q10737",
            "Q108163",
            "Q11016",
            "Q110832782",
            "Q1151067",
            "Q11862829",
            "Q12139612",
            "Q124301146",
            "Q1298668",
            "Q1302249",
            "Q131257",
            "Q131714",
            "Q13406463",
            "Q134995",
            "Q1406161",
            "Q14204246",
            "Q151885",
            "Q16334295",
            "Q1656682",
            "Q16695773",
            "Q170584",
            "Q17524420",
            "... omitted 63 items"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q30208840"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 15,
  "author": "YotaMoteuchi",
  "before_constraint_count": 15,
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
          ],
          "P2306": [
            {
              "description_en": "method (or type) of distribution for the subject",
              "id": "P437",
              "label_en": "distribution format"
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
              "description_en": "Wikimedia template used for navigation",
              "id": "Q108094999",
              "label_en": "Wikimedia sidebar template"
            },
            {
              "description_en": "Wikimedia template used for navigation. Use with P31 'instance of' for navigational templates",
              "id": "Q11753321",
              "label_en": "Wikimedia navigational template"
            },
            {
              "description_en": "class of Wikimedia templates. Use with P31 'instance of' for infobox templates",
              "id": "Q19887878",
              "label_en": "Wikimedia infobox template"
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
              "value": "Use template has topic (P1423) instead@en"
            }
          ],
          "P6824": [
            {
              "description_en": "topic related to template",
              "id": "P1423",
              "label_en": "template has topic"
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
          ],
          "P6607": [
            {
              "value": "link the descriptive pages with ‘different from’ (P1889); if useful use the qualifier ‘object of statement has role’ (P3831); compare the item Abrasion (Q247701)@en"
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
          "P6607": [
            {
              "value": "Use category's main topic (P301) or a related property instead@en"
            }
          ],
          "P6824": [
            {
              "description_en": "primary topic of the subject Wikimedia category",
              "id": "P301",
              "label_en": "category's main topic"
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
              "description_en": "identifier of a version or edition, in the GoodReads.com website",
              "id": "P2969",
              "label_en": "Goodreads version/edition ID"
            }
          ],
          "P6607": [
            {
              "value": "This identifier should be used on items about book editions and not for the general works they are editions of@en"
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
              "description_en": "scientific procedure carried out to support, refute, or validate a hypothesis",
              "id": "Q101965",
              "label_en": "experiment"
            },
            {
              "description_en": "user account on a social network",
              "id": "Q102345381",
              "label_en": "social media account"
            },
            {
              "description_en": "intentional act of causing one's own death",
              "id": "Q10737",
              "label_en": "suicide"
            },
            {
              "description_en": "idea that is true or false and may be expressed in a declarative sentence",
              "id": "Q108163",
              "label_en": "proposition"
            },
            {
              "description_en": "making, modification, usage, and knowledge of tools, machines, techniques, crafts, systems, and methods of organization, or tools and techniques so created",
              "id": "Q11016",
              "label_en": "technology"
            },
            {
              "description_en": "part of a software application accessible, whether it be a data or a program",
              "id": "Q110832782",
              "label_en": "software resource"
            },
            {
              "description_en": "prescription, including laws, regulations, instructions, guidelines, and social conventions; determinate method for performing any operation",
              "id": "Q1151067",
              "label_en": "rule"
            },
            {
              "description_en": "academic field of study or profession",
              "id": "Q11862829",
              "label_en": "academic discipline"
            },
            {
              "description_en": "set of discrete items of information",
              "id": "Q12139612",
              "label_en": "list"
            },
            {
              "description_en": "work which only appears in works of fiction",
              "id": "Q124301146",
              "label_en": "fictional work"
            },
            {
              "description_en": "project of one or more scientists, or of an organization in a scientific field",
              "id": "Q1298668",
              "label_en": "science project"
            },
            {
              "description_en": "reference in one place in a book to information at another place in the same work",
              "id": "Q1302249",
              "label_en": "cross-reference"
            },
            {
              "description_en": "intangible asset consisting of ownership of ideas and processes",
              "id": "Q131257",
              "label_en": "intellectual property"
            },
            {
              "description_en": "Wikidata metaclass for concept in psychology, literature, philosophy",
              "id": "Q131714",
              "label_en": "archetype"
            },
            {
              "description_en": "page of a Wikimedia project with a list of something",
              "id": "Q13406463",
              "label_en": "Wikimedia list article"
            },
            {
              "description_en": "academic discipline that studies books",
              "id": "Q134995",
              "label_en": "bibliography"
            },
            {
              "description_en": "theme or subject in a work of art",
              "id": "Q1406161",
              "label_en": "artistic theme"
            },
            {
              "description_en": "page in the non-article namespace 4 on a Wikimedia project serving internal purposes",
              "id": "Q14204246",
              "label_en": "Wikimedia project page"
            },
            {
              "description_en": "semantic unit understood in different ways, e.g. as mental representation, ability or abstract object (philosophy)",
              "id": "Q151885",
              "label_en": "concept"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "temporary and scheduled happening, like a conference, festival, competition or similar",
              "id": "Q1656682",
              "label_en": "event"
            },
            {
              "description_en": "Wikimedia community project in which a group of contributors collaborates to improve a Wikimedia project on a specific topic",
              "id": "Q16695773",
              "label_en": "WikiProject"
            },
            {
              "description_en": "collaborative enterprise, frequently involving research or design, that is carefully planned to achieve a particular aim",
              "id": "Q170584",
              "label_en": "project"
            },
            {
              "description_en": "topic viewed from a historical point of view",
              "id": "Q17524420",
              "label_en": "aspect of history"
            },
            "... omitted 63 items"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2241": [
            {
              "description_en": "reason for deprecation of a Wikidata property constraint",
              "id": "Q99460987",
              "label_en": "constraint provides suggestions for manual input"
            }
          ],
          "P2306": [
            {
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "time period when a statement is valid",
              "id": "P1264",
              "label_en": "valid in period"
            },
            {
              "description_en": "subject or main statement value is a characteristic, quality, property, or state of this object (use a more specific property where possible)",
              "id": "P13044",
              "label_en": "characteristic of"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            {
              "description_en": "place where a statement is valid",
              "id": "P3005",
              "label_en": "valid in place"
            }
          ]
        },
        "rank": "deprecated",
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
          "P2241": [
            {
              "description_en": "reason for deprecated rank applicable to property constraints for which utility is not apparent, and for which no clarifying information has been given",
              "id": "Q112918944",
              "label_en": "purpose of property constraint unclear"
            }
          ],
          "P2305": [
            {
              "description_en": "genre of fiction",
              "id": "Q197949",
              "label_en": "post-apocalyptic fiction"
            },
            {
              "description_en": "subgenre of apocalyptic fiction",
              "id": "Q2633346",
              "label_en": "zombie apocalyptic fiction"
            },
            {
              "description_en": "genre of speculative fiction usually describing the end of human civilization",
              "id": "Q3919251",
              "label_en": "apocalyptic fiction"
            }
          ]
        },
        "rank": "deprecated",
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
              "description_en": "Japanese term and pornographic genre, typically referring to feminine characters with both female and male genitalia",
              "id": "Q1054122",
              "label_en": "futanari"
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
              "description_en": "artistic theme",
              "id": "Q112061883",
              "label_en": "Susanna and the Elders in art"
            },
            {
              "description_en": "Motif of the massacre of the Innocents from the Gospel of Matthew in the arts",
              "id": "Q15676570",
              "label_en": "Massacre of the Innocents"
            },
            {
              "description_en": "artistic theme",
              "id": "Q18809529",
              "label_en": "incredulity of Thomas"
            },
            {
              "description_en": "artistic theme",
              "id": "Q22818153",
              "label_en": "Presentation of Mary"
            }
          ],
          "P6607": [
            {
              "value": "Meta artistic theme items that shouldn't be used@en"
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
              "description_en": "American pornographic film studio",
              "id": "Q16632366",
              "label_en": "Afro-Centric Productions"
            }
          ],
          "P9729": [
            {
              "description_en": "worldview centered on the history of African civilization",
              "id": "Q1876790",
              "label_en": "Afrocentrism"
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
              "description_en": "manga marketed primarily to adolescent boys",
              "id": "Q231302",
              "label_en": "shōnen"
            },
            {
              "description_en": "manga marketed primarily to late adolescent young adults (18 yr and older)",
              "id": "Q237338",
              "label_en": "seinen"
            },
            {
              "description_en": "manga aimed at a teenage female readership",
              "id": "Q242492",
              "label_en": "shōjo"
            },
            {
              "description_en": "manga for children",
              "id": "Q478804",
              "label_en": "children's manga"
            },
            {
              "description_en": "manga marketed primarily to late adolescent girls and women",
              "id": "Q503106",
              "label_en": "josei"
            }
          ],
          "P6607": [
            {
              "value": "Use intended public (P2360) instead@en"
            }
          ],
          "P6824": [
            {
              "description_en": "intended audience or user of this work, product, object, or event",
              "id": "P2360",
              "label_en": "intended public"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
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
          ],
          "P2306": [
            {
              "description_en": "method (or type) of distribution for the subject",
              "id": "P437",
              "label_en": "distribution format"
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
              "description_en": "Wikimedia template used for navigation",
              "id": "Q108094999",
              "label_en": "Wikimedia sidebar template"
            },
            {
              "description_en": "Wikimedia template used for navigation. Use with P31 'instance of' for navigational templates",
              "id": "Q11753321",
              "label_en": "Wikimedia navigational template"
            },
            {
              "description_en": "class of Wikimedia templates. Use with P31 'instance of' for infobox templates",
              "id": "Q19887878",
              "label_en": "Wikimedia infobox template"
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
              "value": "Use template has topic (P1423) instead@en"
            }
          ],
          "P6824": [
            {
              "description_en": "topic related to template",
              "id": "P1423",
              "label_en": "template has topic"
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
          ],
          "P6607": [
            {
              "value": "link the descriptive pages with ‘different from’ (P1889); if useful use the qualifier ‘object of statement has role’ (P3831); compare the item Abrasion (Q247701)@en"
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
          "P6607": [
            {
              "value": "Use category's main topic (P301) or a related property instead@en"
            }
          ],
          "P6824": [
            {
              "description_en": "primary topic of the subject Wikimedia category",
              "id": "P301",
              "label_en": "category's main topic"
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
              "description_en": "identifier of a version or edition, in the GoodReads.com website",
              "id": "P2969",
              "label_en": "Goodreads version/edition ID"
            }
          ],
          "P6607": [
            {
              "value": "This identifier should be used on items about book editions and not for the general works they are editions of@en"
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
              "description_en": "scientific procedure carried out to support, refute, or validate a hypothesis",
              "id": "Q101965",
              "label_en": "experiment"
            },
            {
              "description_en": "user account on a social network",
              "id": "Q102345381",
              "label_en": "social media account"
            },
            {
              "description_en": "intentional act of causing one's own death",
              "id": "Q10737",
              "label_en": "suicide"
            },
            {
              "description_en": "idea that is true or false and may be expressed in a declarative sentence",
              "id": "Q108163",
              "label_en": "proposition"
            },
            {
              "description_en": "making, modification, usage, and knowledge of tools, machines, techniques, crafts, systems, and methods of organization, or tools and techniques so created",
              "id": "Q11016",
              "label_en": "technology"
            },
            {
              "description_en": "part of a software application accessible, whether it be a data or a program",
              "id": "Q110832782",
              "label_en": "software resource"
            },
            {
              "description_en": "prescription, including laws, regulations, instructions, guidelines, and social conventions; determinate method for performing any operation",
              "id": "Q1151067",
              "label_en": "rule"
            },
            {
              "description_en": "academic field of study or profession",
              "id": "Q11862829",
              "label_en": "academic discipline"
            },
            {
              "description_en": "set of discrete items of information",
              "id": "Q12139612",
              "label_en": "list"
            },
            {
              "description_en": "work which only appears in works of fiction",
              "id": "Q124301146",
              "label_en": "fictional work"
            },
            {
              "description_en": "project of one or more scientists, or of an organization in a scientific field",
              "id": "Q1298668",
              "label_en": "science project"
            },
            {
              "description_en": "reference in one place in a book to information at another place in the same work",
              "id": "Q1302249",
              "label_en": "cross-reference"
            },
            {
              "description_en": "intangible asset consisting of ownership of ideas and processes",
              "id": "Q131257",
              "label_en": "intellectual property"
            },
            {
              "description_en": "Wikidata metaclass for concept in psychology, literature, philosophy",
              "id": "Q131714",
              "label_en": "archetype"
            },
            {
              "description_en": "page of a Wikimedia project with a list of something",
              "id": "Q13406463",
              "label_en": "Wikimedia list article"
            },
            {
              "description_en": "academic discipline that studies books",
              "id": "Q134995",
              "label_en": "bibliography"
            },
            {
              "description_en": "theme or subject in a work of art",
              "id": "Q1406161",
              "label_en": "artistic theme"
            },
            {
              "description_en": "page in the non-article namespace 4 on a Wikimedia project serving internal purposes",
              "id": "Q14204246",
              "label_en": "Wikimedia project page"
            },
            {
              "description_en": "semantic unit understood in different ways, e.g. as mental representation, ability or abstract object (philosophy)",
              "id": "Q151885",
              "label_en": "concept"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "temporary and scheduled happening, like a conference, festival, competition or similar",
              "id": "Q1656682",
              "label_en": "event"
            },
            {
              "description_en": "Wikimedia community project in which a group of contributors collaborates to improve a Wikimedia project on a specific topic",
              "id": "Q16695773",
              "label_en": "WikiProject"
            },
            {
              "description_en": "collaborative enterprise, frequently involving research or design, that is carefully planned to achieve a particular aim",
              "id": "Q170584",
              "label_en": "project"
            },
            {
              "description_en": "topic viewed from a historical point of view",
              "id": "Q17524420",
              "label_en": "aspect of history"
            },
            "... omitted 62 items"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2241": [
            {
              "description_en": "reason for deprecation of a Wikidata property constraint",
              "id": "Q99460987",
              "label_en": "constraint provides suggestions for manual input"
            }
          ],
          "P2306": [
            {
              "description_en": "the item (institution, law, public office, public register, etc) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, etc)",
              "id": "P1001",
              "label_en": "applies to jurisdiction"
            },
            {
              "description_en": "time period when a statement is valid",
              "id": "P1264",
              "label_en": "valid in period"
            },
            {
              "description_en": "subject or main statement value is a characteristic, quality, property, or state of this object (use a more specific property where possible)",
              "id": "P13044",
              "label_en": "characteristic of"
            },
            {
              "description_en": "use as qualifier to indicate how the object's value was given in the source",
              "id": "P1932",
              "label_en": "object named as"
            },
            {
              "description_en": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
              "id": "P276",
              "label_en": "location"
            },
            {
              "description_en": "place where a statement is valid",
              "id": "P3005",
              "label_en": "valid in place"
            }
          ]
        },
        "rank": "deprecated",
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
          "P2241": [
            {
              "description_en": "reason for deprecated rank applicable to property constraints for which utility is not apparent, and for which no clarifying information has been given",
              "id": "Q112918944",
              "label_en": "purpose of property constraint unclear"
            }
          ],
          "P2305": [
            {
              "description_en": "genre of fiction",
              "id": "Q197949",
              "label_en": "post-apocalyptic fiction"
            },
            {
              "description_en": "subgenre of apocalyptic fiction",
              "id": "Q2633346",
              "label_en": "zombie apocalyptic fiction"
            },
            {
              "description_en": "genre of speculative fiction usually describing the end of human civilization",
              "id": "Q3919251",
              "label_en": "apocalyptic fiction"
            }
          ]
        },
        "rank": "deprecated",
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
              "description_en": "Japanese term and pornographic genre, typically referring to feminine characters with both female and male genitalia",
              "id": "Q1054122",
              "label_en": "futanari"
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
              "description_en": "artistic theme",
              "id": "Q112061883",
              "label_en": "Susanna and the Elders in art"
            },
            {
              "description_en": "Motif of the massacre of the Innocents from the Gospel of Matthew in the arts",
              "id": "Q15676570",
              "label_en": "Massacre of the Innocents"
            },
            {
              "description_en": "artistic theme",
              "id": "Q18809529",
              "label_en": "incredulity of Thomas"
            },
            {
              "description_en": "artistic theme",
              "id": "Q22818153",
              "label_en": "Presentation of Mary"
            }
          ],
          "P6607": [
            {
              "value": "Meta artistic theme items that shouldn't be used@en"
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
              "description_en": "American pornographic film studio",
              "id": "Q16632366",
              "label_en": "Afro-Centric Productions"
            }
          ],
          "P9729": [
            {
              "description_en": "worldview centered on the history of African civilization",
              "id": "Q1876790",
              "label_en": "Afrocentrism"
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
              "description_en": "manga marketed primarily to adolescent boys",
              "id": "Q231302",
              "label_en": "shōnen"
            },
            {
              "description_en": "manga marketed primarily to late adolescent young adults (18 yr and older)",
              "id": "Q237338",
              "label_en": "seinen"
            },
            {
              "description_en": "manga aimed at a teenage female readership",
              "id": "Q242492",
              "label_en": "shōjo"
            },
            {
              "description_en": "manga for children",
              "id": "Q478804",
              "label_en": "children's manga"
            },
            {
              "description_en": "manga marketed primarily to late adolescent girls and women",
              "id": "Q503106",
              "label_en": "josei"
            }
          ],
          "P6607": [
            {
              "value": "Use intended public (P2360) instead@en"
            }
          ],
          "P6824": [
            {
              "description_en": "intended audience or user of this work, product, object, or event",
              "id": "P2360",
              "label_en": "intended public"
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
  "hash_after": "d5ea1158d61a890f5b9b5973dbea33604cae9572",
  "hash_before": "27f65b7446fdc00965554e75fb3f409357651989",
  "property_revision_id": 2447527451,
  "property_revision_prev": 2438998167,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q3249551"
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
            "Q101965",
            "Q102345381",
            "Q10737",
            "Q108163",
            "Q11016",
            "Q110832782",
            "Q1151067",
            "Q11862829",
            "Q12139612",
            "Q124301146",
            "Q1298668",
            "Q1302249",
            "Q131257",
            "Q131714",
            "Q13406463",
            "Q134995",
            "Q1406161",
            "Q14204246",
            "Q151885",
            "Q16334295",
            "Q1656682",
            "Q16695773",
            "Q170584",
            "Q17524420",
            "... omitted 62 items"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q30208840"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "conflicts-with constraint: item of property constraint: bunkobon, shinsho, tête-bêche, Тыс, ebook, ring connection, pocket edition, paperback, hardcover, tankōbon, chapbook, leporello book, softcover; property: distribution format",
      "conflicts-with constraint: item of property constraint: Wikimedia sidebar template, Wikimedia navigational template, Wikimedia infobox template; property: instance of; constraint clarification: Use template has topic (P1423) instead@en; replacement property: template has topic",
      "conflicts-with constraint: item of property constraint: Wikimedia disambiguation page; property: instance of; constraint clarification: link the descriptive pages with ‘different from’ (P1889); if useful use the qualifier ‘object of statement has role’ (P3831); compare the item Abrasion (Q247701)@en",
      "conflicts-with constraint: item of property constraint: Wikimedia category; property: instance of; constraint clarification: Use category's main topic (P301) or a related property instead@en; replacement property: category's main topic",
      "conflicts-with constraint: item of property constraint: human; property: instance of; constraint status: mandatory constraint",
      "conflicts-with constraint: property: Goodreads version/edition ID; constraint clarification: This identifier should be used on items about book editions and not for the general works they are editions of@en",
      "subject type constraint: class: experiment, social media account, suicide, proposition, technology, software resource, rule, academic discipline, list, fictional work, science project, cross-reference, intellectual property, archetype, Wikimedia list article, bibliography, artistic theme, Wikimedia project page, concept, group of humans, event, WikiProject, project, aspect of history, YouTube channel, community, paraphilia, conflict, idiom, item of collection or e... [truncated 829 chars]",
      "allowed qualifiers constraint: reason for deprecated rank: constraint provides suggestions for manual input; property: applies to jurisdiction, valid in period, characteristic of, object named as, location, valid in place",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, Wikibase property, МэдыяІнфа Вікібазы; constraint status: mandatory constraint",
      "none-of constraint: reason for deprecated rank: purpose of property constraint unclear; item of property constraint: post-apocalyptic fiction, zombie apocalyptic fiction, apocalyptic fiction",
      "none-of constraint: item of property constraint: futanari",
      "none-of constraint: item of property constraint: Susanna and the Elders in art, Massacre of the Innocents, incredulity of Thomas, Presentation of Mary; constraint clarification: Meta artistic theme items that shouldn't be used@en",
      "none-of constraint: item of property constraint: Afro-Centric Productions; replacement value: Afrocentrism",
      "none-of constraint: item of property constraint: shōnen, seinen, shōjo, children's manga, josei; constraint clarification: Use intended public (P2360) instead@en; replacement property: intended public",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as qualifier, as reference"
    ],
    "before": [
      "conflicts-with constraint: item of property constraint: bunkobon, shinsho, tête-bêche, Тыс, ebook, ring connection, pocket edition, paperback, hardcover, tankōbon, chapbook, leporello book, softcover; property: distribution format",
      "conflicts-with constraint: item of property constraint: Wikimedia sidebar template, Wikimedia navigational template, Wikimedia infobox template; property: instance of; constraint clarification: Use template has topic (P1423) instead@en; replacement property: template has topic",
      "conflicts-with constraint: item of property constraint: Wikimedia disambiguation page; property: instance of; constraint clarification: link the descriptive pages with ‘different from’ (P1889); if useful use the qualifier ‘object of statement has role’ (P3831); compare the item Abrasion (Q247701)@en",
      "conflicts-with constraint: item of property constraint: Wikimedia category; property: instance of; constraint clarification: Use category's main topic (P301) or a related property instead@en; replacement property: category's main topic",
      "conflicts-with constraint: item of property constraint: human; property: instance of; constraint status: mandatory constraint",
      "conflicts-with constraint: property: Goodreads version/edition ID; constraint clarification: This identifier should be used on items about book editions and not for the general works they are editions of@en",
      "subject type constraint: class: experiment, social media account, suicide, proposition, technology, software resource, rule, academic discipline, list, fictional work, science project, cross-reference, intellectual property, archetype, Wikimedia list article, bibliography, artistic theme, Wikimedia project page, concept, group of humans, event, WikiProject, project, aspect of history, YouTube channel, community, paraphilia, conflict, idiom, item of collection or e... [truncated 820 chars]",
      "allowed qualifiers constraint: reason for deprecated rank: constraint provides suggestions for manual input; property: applies to jurisdiction, valid in period, characteristic of, object named as, location, valid in place",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, Wikibase property, МэдыяІнфа Вікібазы; constraint status: mandatory constraint",
      "none-of constraint: reason for deprecated rank: purpose of property constraint unclear; item of property constraint: post-apocalyptic fiction, zombie apocalyptic fiction, apocalyptic fiction",
      "none-of constraint: item of property constraint: futanari",
      "none-of constraint: item of property constraint: Susanna and the Elders in art, Massacre of the Innocents, incredulity of Thomas, Presentation of Mary; constraint clarification: Meta artistic theme items that shouldn't be used@en",
      "none-of constraint: item of property constraint: Afro-Centric Productions; replacement value: Afrocentrism",
      "none-of constraint: item of property constraint: shōnen, seinen, shōjo, children's manga, josei; constraint clarification: Use intended public (P2360) instead@en; replacement property: intended public",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as qualifier, as reference"
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
    "violation_name": "Type Q|386724, Q|16334295, Q|1302249, Q|483394, Q|1656682, Q|600134, Q|131257, Q|1406161, Q|676586, Q|230788, Q|30612, Q|1964968, Q|23649976, Q|192909, Q|2904148, Q|9388534, Q|2668072, Q|178059, Q|379833, Q|21004260, Q|278485, Q|1298668, Q|101965, Q|17524420, Q|305178, Q|170584, Q|4503831, Q|62090711, Q|4915012, Q|955824, Q|978, Q|134995, Q|362165, Q|3239681, Q|151885, Q|2416723, Q|18593264, Q|21484471, Q|3689704, Q|12139612, Q|13406463, Q|14204246, Q|3026787, Q|1... [truncated 445 chars]"
  },
  {
    "result": "Q21502838",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 007. `reform_Q15142894_P31_2440675178`

| Field | Value |
|---|---|
| qid | Q15142894 |
| property | P31 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
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
| rationale | Constraint qualifiers compared with generic set semantics. |
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
  "report_violation_type": "Values statistics",
  "report_violation_type_normalized": "Values statistics",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Values statistics",
  "value": null,
  "value_current_2026": [
    "Q24017414"
  ],
  "value_current_2026_descriptions_en": [
    "metaclass containing as instances all classes of individuals"
  ],
  "value_current_2026_labels_en": [
    "second-order class"
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
    "description": "specific weapon design, pattern, or version of which all examples are essentially identical",
    "label": "weapon model"
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Values statistics"
  },
  {
    "result": "Q21510851",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 008. `reform_Q16474964_P131_2437507566`

| Field | Value |
|---|---|
| qid | Q16474964 |
| property | P131 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q21502838 |
| group_key | TBOX::P131::2437507566 |
| tbox_revision_key | TBOX::P131::2437507566 |

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
  "author": "Necessarycoot72",
  "kind": "T_BOX",
  "property_revision_id": 2437507566,
  "property_revision_prev": 2437506232
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
  "value": null,
  "value_current_2026": [
    "Q141132",
    "Q12653219"
  ],
  "value_current_2026_descriptions_en": [
    "district municipality of Lithuania",
    null
  ],
  "value_current_2026_labels_en": [
    "Klaipėda District Municipality",
    "Dovilų valsčius"
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
    "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
    "label": "located in the administrative territorial entity"
  },
  "qid": {
    "description": null,
    "label": "Vaidaugų apylinkė"
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

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21510865",
      "qualifiers": [
        {
          "property_id": "P2308",
          "values": [
            "Q15617994",
            "Q19953632",
            "Q245016",
            "Q30070324",
            "Q56061",
            "Q64034456",
            "Q7275"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q30208840"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 18,
  "author": "Necessarycoot72",
  "before_constraint_count": 18,
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
              "description_en": "type of page in the Wikimedia system. Use with P31 'instance of' for template pages",
              "id": "Q11266439",
              "label_en": "Wikimedia template"
            },
            {
              "description_en": "use with 'instance of' (P31) for modules written in Lua, and working with the Scribunto extension for MediaWiki installed on a Wikimedia wiki",
              "id": "Q15184295",
              "label_en": "Wikimedia module"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
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
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
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
              "description_en": "qualifier: indicates the value at the next hierarchy level which is true for this item, when more than one is possible at the next level",
              "id": "P10229",
              "label_en": "next level in hierarchy"
            },
            {
              "description_en": "to be used as a qualifier, value must be between 0 and 1",
              "id": "P1107",
              "label_en": "proportion"
            },
            {
              "description_en": "latest date beyond which the statement could no longer be true",
              "id": "P12506",
              "label_en": "latest end date"
            },
            {
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "entity that disputes a given statement",
              "id": "P1310",
              "label_en": "statement disputed by"
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
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
            },
            {
              "description_en": "measured dimension of an object",
              "id": "P2043",
              "label_en": "length"
            },
            {
              "description_en": "area occupied by an object",
              "id": "P2046",
              "label_en": "area"
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
              "description_en": "code assigned by postal authorities for the subject area or building for the purpose of sorting and routing mail",
              "id": "P281",
              "label_en": "postal code"
            },
            {
              "description_en": "role held by the item the statement appears on (\"subject\") in the context of that statement. For the role of the statement object/value, use P3831 (\"object has role\"). For acting roles, use P453 (\"character role\"). For persons, use P39.",
              "id": "P2868",
              "label_en": "subject has role"
            },
            {
              "description_en": "entity that supports a given statement",
              "id": "P3680",
              "label_en": "statement supported by"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "number identifying a building in one cadastral area/village/neighborhood",
              "id": "P4856",
              "label_en": "conscription number"
            },
            {
              "description_en": "(qualifier only) the underlying circumstances of this statement",
              "id": "P5102",
              "label_en": "nature of statement"
            },
            "... omitted 16 items"
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
          "P2303": [
            {
              "description_en": "geomorphic features of the earth's surface that are located below the sea",
              "id": "Q55182671",
              "label_en": "undersea landform"
            }
          ],
          "P2306": [
            {
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
              "description_en": "type of administrative divisions, usually used in several countries",
              "id": "Q15617994",
              "label_en": "administrative territorial entity type"
            },
            {
              "description_en": "administrative division which is no longer in use",
              "id": "Q19953632",
              "label_en": "former administrative territorial entity"
            },
            {
              "description_en": "facility directly owned and operated by or for the military",
              "id": "Q245016",
              "label_en": "military base"
            },
            {
              "description_en": "county which only exists in a work of fiction",
              "id": "Q30070324",
              "label_en": "fictional county"
            },
            {
              "description_en": "territorial entity for administration purposes, with or without its own local government",
              "id": "Q56061",
              "label_en": "administrative territorial entity"
            },
            {
              "description_en": "administrative territorial entity which only exists in a work of fiction",
              "id": "Q64034456",
              "label_en": "fictional administrative territorial entity"
            },
            {
              "description_en": "organised community living under a system of government; either a sovereign state, constituent state, or federated state",
              "id": "Q7275",
              "label_en": "state"
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
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "multi-ethnic complex of territories in Western and Central Europe (800/962–1806)",
              "id": "Q12548",
              "label_en": "Holy Roman Empire"
            }
          ],
          "P6607": [
            {
              "value": "benutze stattdessen \"ist ein(e)\" (P31) mit Q26830017@de"
            },
            {
              "value": "use \"instance of\" (P31) with Q26830017 instead@en"
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
              "description_en": "Confederation of States in Germany from 1815 to 1866",
              "id": "Q151624",
              "label_en": "German Confederation"
            }
          ],
          "P6607": [
            {
              "value": "benutze stattdessen \"ist ein(e)\" (P31) mit Q113136497@de"
            },
            {
              "value": "use \"instance of\" (P31) with Q113136497 instead@en"
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
              "description_en": "confederation of client states of the First French Empire",
              "id": "Q154741",
              "label_en": "Confederation of the Rhine"
            }
          ],
          "P6607": [
            {
              "value": "benutze stattdessen \"ist ein(e)\" (P31) mit Q26879769@de"
            },
            {
              "value": "use \"instance of\" (P31) with Q26879769 instead@en"
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
              "description_en": "third planet from the Sun in the Solar System",
              "id": "Q2",
              "label_en": "Earth"
            },
            {
              "description_en": "type of a municipal formation in Russia",
              "id": "Q634099",
              "label_en": "rural settlement in Russia"
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
          "P2305": [
            {
              "description_en": "five-member executive committee of Revolutionary France (1795-1799)",
              "id": "Q219817",
              "label_en": "French Directory"
            },
            {
              "description_en": "single-chamber assembly in France from 21 September 1792 to 26 October 1795",
              "id": "Q219825",
              "label_en": "National Convention"
            },
            {
              "description_en": "former government of France",
              "id": "Q877619",
              "label_en": "French Consulate"
            }
          ],
          "P9729": [
            {
              "description_en": "republic governing France, 1792–1804",
              "id": "Q58296",
              "label_en": "French First Republic"
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
              "description_en": "Germany in the years 1918/1919–1933",
              "id": "Q41304",
              "label_en": "Weimar Republic"
            },
            {
              "description_en": "German nation-state in Central Europe from 1871 to 1918",
              "id": "Q43287",
              "label_en": "German Empire"
            },
            {
              "description_en": "term for Nazi Germany",
              "id": "Q518617",
              "label_en": "Third Reich"
            },
            {
              "description_en": "Germany from 1933 to 1945 while under control of the Nazi Party",
              "id": "Q7318",
              "label_en": "Nazi Germany"
            }
          ],
          "P6607": [
            {
              "value": "Use Q1206012 instead@en"
            }
          ],
          "P9729": [
            {
              "description_en": "official name for the German nation state from 1871 to 1945, and name of Germany until 1949",
              "id": "Q1206012",
              "label_en": "German Reich"
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
              "description_en": "Wikipedia overview article about the two historical German principalities of this name",
              "id": "Q650489",
              "label_en": "Margraviate of Baden"
            }
          ],
          "P9729": [
            {
              "description_en": "historical German principaility (1112-1535)",
              "id": "Q131545583",
              "label_en": "Margraviate of Baden"
            },
            {
              "description_en": "historical German principality (1771-1803)",
              "id": "Q20011161",
              "label_en": "Margraviate of Baden"
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
              "description_en": "Federal Republic of Germany in the period between its formation on 23 May 1949 and German reunification on 3 October 1990",
              "id": "Q713750",
              "label_en": "West Germany"
            }
          ],
          "P9729": [
            {
              "description_en": "country in Central Europe",
              "id": "Q183",
              "label_en": "Germany"
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
              "description_en": "type of page in the Wikimedia system. Use with P31 'instance of' for template pages",
              "id": "Q11266439",
              "label_en": "Wikimedia template"
            },
            {
              "description_en": "use with 'instance of' (P31) for modules written in Lua, and working with the Scribunto extension for MediaWiki installed on a Wikimedia wiki",
              "id": "Q15184295",
              "label_en": "Wikimedia module"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2305": [
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
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
              "description_en": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
              "id": "P31",
              "label_en": "instance of"
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
              "description_en": "qualifier: indicates the value at the next hierarchy level which is true for this item, when more than one is possible at the next level",
              "id": "P10229",
              "label_en": "next level in hierarchy"
            },
            {
              "description_en": "to be used as a qualifier, value must be between 0 and 1",
              "id": "P1107",
              "label_en": "proportion"
            },
            {
              "description_en": "latest date beyond which the statement could no longer be true",
              "id": "P12506",
              "label_en": "latest end date"
            },
            {
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "entity that disputes a given statement",
              "id": "P1310",
              "label_en": "statement disputed by"
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
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
            },
            {
              "description_en": "measured dimension of an object",
              "id": "P2043",
              "label_en": "length"
            },
            {
              "description_en": "area occupied by an object",
              "id": "P2046",
              "label_en": "area"
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
              "description_en": "code assigned by postal authorities for the subject area or building for the purpose of sorting and routing mail",
              "id": "P281",
              "label_en": "postal code"
            },
            {
              "description_en": "role held by the item the statement appears on (\"subject\") in the context of that statement. For the role of the statement object/value, use P3831 (\"object has role\"). For acting roles, use P453 (\"character role\"). For persons, use P39.",
              "id": "P2868",
              "label_en": "subject has role"
            },
            {
              "description_en": "entity that supports a given statement",
              "id": "P3680",
              "label_en": "statement supported by"
            },
            {
              "description_en": "(qualifier) role held by the predicate value (object) of a statement in the context of that statement; for the role of the item the statement appears on (subject), use P2868",
              "id": "P3831",
              "label_en": "object of statement has role"
            },
            {
              "description_en": "number identifying a building in one cadastral area/village/neighborhood",
              "id": "P4856",
              "label_en": "conscription number"
            },
            {
              "description_en": "(qualifier only) the underlying circumstances of this statement",
              "id": "P5102",
              "label_en": "nature of statement"
            },
            "... omitted 16 items"
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
          "P2303": [
            {
              "description_en": "geomorphic features of the earth's surface that are located below the sea",
              "id": "Q55182671",
              "label_en": "undersea landform"
            }
          ],
          "P2306": [
            {
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
              "description_en": "type of administrative divisions, usually used in several countries",
              "id": "Q15617994",
              "label_en": "administrative territorial entity type"
            },
            {
              "description_en": "administrative division which is no longer in use",
              "id": "Q19953632",
              "label_en": "former administrative territorial entity"
            },
            {
              "description_en": "facility directly owned and operated by or for the military",
              "id": "Q245016",
              "label_en": "military base"
            },
            {
              "description_en": "county which only exists in a work of fiction",
              "id": "Q30070324",
              "label_en": "fictional county"
            },
            {
              "description_en": "statistical concentration of population defined by the United States Census Bureau",
              "id": "Q498162",
              "label_en": "census-designated place in the United States"
            },
            {
              "description_en": "territorial entity for administration purposes, with or without its own local government",
              "id": "Q56061",
              "label_en": "administrative territorial entity"
            },
            {
              "description_en": "administrative territorial entity which only exists in a work of fiction",
              "id": "Q64034456",
              "label_en": "fictional administrative territorial entity"
            },
            {
              "description_en": "organised community living under a system of government; either a sovereign state, constituent state, or federated state",
              "id": "Q7275",
              "label_en": "state"
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
          "description_en": "constraint specifying values that should not be used for the given property",
          "id": "Q52558054",
          "label_en": "none-of constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "multi-ethnic complex of territories in Western and Central Europe (800/962–1806)",
              "id": "Q12548",
              "label_en": "Holy Roman Empire"
            }
          ],
          "P6607": [
            {
              "value": "benutze stattdessen \"ist ein(e)\" (P31) mit Q26830017@de"
            },
            {
              "value": "use \"instance of\" (P31) with Q26830017 instead@en"
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
              "description_en": "Confederation of States in Germany from 1815 to 1866",
              "id": "Q151624",
              "label_en": "German Confederation"
            }
          ],
          "P6607": [
            {
              "value": "benutze stattdessen \"ist ein(e)\" (P31) mit Q113136497@de"
            },
            {
              "value": "use \"instance of\" (P31) with Q113136497 instead@en"
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
              "description_en": "confederation of client states of the First French Empire",
              "id": "Q154741",
              "label_en": "Confederation of the Rhine"
            }
          ],
          "P6607": [
            {
              "value": "benutze stattdessen \"ist ein(e)\" (P31) mit Q26879769@de"
            },
            {
              "value": "use \"instance of\" (P31) with Q26879769 instead@en"
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
              "description_en": "third planet from the Sun in the Solar System",
              "id": "Q2",
              "label_en": "Earth"
            },
            {
              "description_en": "type of a municipal formation in Russia",
              "id": "Q634099",
              "label_en": "rural settlement in Russia"
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
          "P2305": [
            {
              "description_en": "five-member executive committee of Revolutionary France (1795-1799)",
              "id": "Q219817",
              "label_en": "French Directory"
            },
            {
              "description_en": "single-chamber assembly in France from 21 September 1792 to 26 October 1795",
              "id": "Q219825",
              "label_en": "National Convention"
            },
            {
              "description_en": "former government of France",
              "id": "Q877619",
              "label_en": "French Consulate"
            }
          ],
          "P9729": [
            {
              "description_en": "republic governing France, 1792–1804",
              "id": "Q58296",
              "label_en": "French First Republic"
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
              "description_en": "Germany in the years 1918/1919–1933",
              "id": "Q41304",
              "label_en": "Weimar Republic"
            },
            {
              "description_en": "German nation-state in Central Europe from 1871 to 1918",
              "id": "Q43287",
              "label_en": "German Empire"
            },
            {
              "description_en": "term for Nazi Germany",
              "id": "Q518617",
              "label_en": "Third Reich"
            },
            {
              "description_en": "Germany from 1933 to 1945 while under control of the Nazi Party",
              "id": "Q7318",
              "label_en": "Nazi Germany"
            }
          ],
          "P6607": [
            {
              "value": "Use Q1206012 instead@en"
            }
          ],
          "P9729": [
            {
              "description_en": "official name for the German nation state from 1871 to 1945, and name of Germany until 1949",
              "id": "Q1206012",
              "label_en": "German Reich"
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
              "description_en": "Wikipedia overview article about the two historical German principalities of this name",
              "id": "Q650489",
              "label_en": "Margraviate of Baden"
            }
          ],
          "P9729": [
            {
              "description_en": "historical German principaility (1112-1535)",
              "id": "Q131545583",
              "label_en": "Margraviate of Baden"
            },
            {
              "description_en": "historical German principality (1771-1803)",
              "id": "Q20011161",
              "label_en": "Margraviate of Baden"
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
              "description_en": "Federal Republic of Germany in the period between its formation on 23 May 1949 and German reunification on 3 October 1990",
              "id": "Q713750",
              "label_en": "West Germany"
            }
          ],
          "P9729": [
            {
              "description_en": "country in Central Europe",
              "id": "Q183",
              "label_en": "Germany"
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
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "5760e2f84413098a914c8a5b9269acb4fce68305",
  "hash_before": "69774ae093d067e99709c2489d12aec3785522ce",
  "property_revision_id": 2437507566,
  "property_revision_prev": 2437506232,
  "qualifier_value_changes": [
    {
      "added_values": [],
      "constraint_qid": "Q21510865",
      "qualifier_property": "P2308",
      "removed_values": [
        "Q498162"
      ],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21510865",
      "qualifiers": [
        {
          "property_id": "P2308",
          "values": [
            "Q15617994",
            "Q19953632",
            "Q245016",
            "Q30070324",
            "Q498162",
            "Q56061",
            "Q64034456",
            "Q7275"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q30208840"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "conflicts-with constraint: item of property constraint: Wikimedia template, Wikimedia module, Wikimedia category; property: instance of",
      "conflicts-with constraint: item of property constraint: Wikimedia disambiguation page; property: instance of; constraint status: mandatory constraint",
      "conflicts-with constraint: item of property constraint: human; property: instance of",
      "item-requires-statement constraint: property: country; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: instance of; constraint status: suggestion constraint",
      "allowed qualifiers constraint: property: applies to jurisdiction, excluding, including, criterion used, next level in hierarchy, proportion, latest end date, located in the administrative territorial entity, statement disputed by, earliest date, latest date, sourcing circumstances, end cause, country, length, area, reason for deprecated rank, location, postal code, subject has role, statement supported by, object of statement has role, conscription number, nature ... [truncated 305 chars]",
      "value-requires-statement constraint: exception to constraint: undersea landform; property: country",
      "value-type constraint: class: administrative territorial entity type, former administrative territorial entity, military base, fictional county, administrative territorial entity, fictional administrative territorial entity, state; relation: instance or subclass of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "none-of constraint: item of property constraint: Holy Roman Empire; constraint clarification: benutze stattdessen \"ist ein(e)\" (P31) mit Q26830017@de, use \"instance of\" (P31) with Q26830017 instead@en",
      "none-of constraint: item of property constraint: German Confederation; constraint clarification: benutze stattdessen \"ist ein(e)\" (P31) mit Q113136497@de, use \"instance of\" (P31) with Q113136497 instead@en",
      "none-of constraint: item of property constraint: Confederation of the Rhine; constraint clarification: benutze stattdessen \"ist ein(e)\" (P31) mit Q26879769@de, use \"instance of\" (P31) with Q26879769 instead@en",
      "none-of constraint: item of property constraint: Earth, rural settlement in Russia; constraint status: mandatory constraint",
      "none-of constraint: item of property constraint: French Directory, National Convention, French Consulate; replacement value: French First Republic",
      "none-of constraint: item of property constraint: Weimar Republic, German Empire, Third Reich, Nazi Germany; constraint clarification: Use Q1206012 instead@en; replacement value: German Reich",
      "none-of constraint: item of property constraint: Margraviate of Baden; replacement value: Margraviate of Baden, Margraviate of Baden",
      "none-of constraint: item of property constraint: West Germany; replacement value: Germany",
      "property scope constraint: property scope: as main value, as qualifier"
    ],
    "before": [
      "conflicts-with constraint: item of property constraint: Wikimedia template, Wikimedia module, Wikimedia category; property: instance of",
      "conflicts-with constraint: item of property constraint: Wikimedia disambiguation page; property: instance of; constraint status: mandatory constraint",
      "conflicts-with constraint: item of property constraint: human; property: instance of",
      "item-requires-statement constraint: property: country; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: instance of; constraint status: suggestion constraint",
      "allowed qualifiers constraint: property: applies to jurisdiction, excluding, including, criterion used, next level in hierarchy, proportion, latest end date, located in the administrative territorial entity, statement disputed by, earliest date, latest date, sourcing circumstances, end cause, country, length, area, reason for deprecated rank, location, postal code, subject has role, statement supported by, object of statement has role, conscription number, nature ... [truncated 305 chars]",
      "value-requires-statement constraint: exception to constraint: undersea landform; property: country",
      "value-type constraint: class: administrative territorial entity type, former administrative territorial entity, military base, fictional county, census-designated place in the United States, administrative territorial entity, fictional administrative territorial entity, state; relation: instance or subclass of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "none-of constraint: item of property constraint: Holy Roman Empire; constraint clarification: benutze stattdessen \"ist ein(e)\" (P31) mit Q26830017@de, use \"instance of\" (P31) with Q26830017 instead@en",
      "none-of constraint: item of property constraint: German Confederation; constraint clarification: benutze stattdessen \"ist ein(e)\" (P31) mit Q113136497@de, use \"instance of\" (P31) with Q113136497 instead@en",
      "none-of constraint: item of property constraint: Confederation of the Rhine; constraint clarification: benutze stattdessen \"ist ein(e)\" (P31) mit Q26879769@de, use \"instance of\" (P31) with Q26879769 instead@en",
      "none-of constraint: item of property constraint: Earth, rural settlement in Russia; constraint status: mandatory constraint",
      "none-of constraint: item of property constraint: French Directory, National Convention, French Consulate; replacement value: French First Republic",
      "none-of constraint: item of property constraint: Weimar Republic, German Empire, Third Reich, Nazi Germany; constraint clarification: Use Q1206012 instead@en; replacement value: German Reich",
      "none-of constraint: item of property constraint: Margraviate of Baden; replacement value: Margraviate of Baden, Margraviate of Baden",
      "none-of constraint: item of property constraint: West Germany; replacement value: Germany",
      "property scope constraint: property scope: as main value, as qualifier"
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
    "violation_name": "Target required claim P|17"
  },
  {
    "result": "Q21502838",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 009. `reform_Q165391_P1344_2444117279`

| Field | Value |
|---|---|
| qid | Q165391 |
| property | P1344 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| group_key | TBOX::P1344::2444117279 |
| tbox_revision_key | TBOX::P1344::2444117279 |

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
  "author": "Swpb",
  "kind": "T_BOX",
  "property_revision_id": 2444117279,
  "property_revision_prev": 2439333402
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-22T08:43:50",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P1344",
  "report_revision_new": 2445421520,
  "report_revision_old": 2444860386,
  "report_violation_type": "Value type Q|1190554, Q|386724, Q|43229, Q|101965, Q|14136353, Q|1914636, Q|1656682",
  "report_violation_type_descriptions_en": [
    "occurrence of a fact or object in space-time; instantiation of a property in an object",
    "intellectual or artistic creation",
    "social entity established to meet needs or pursue goals",
    "scientific procedure carried out to support, refute, or validate a hypothesis",
    "happening which occurs in a fiction",
    "series of actions done by an agent which results in an external change of state",
    "temporary and scheduled happening, like a conference, festival, competition or similar"
  ],
  "report_violation_type_labels_en": [
    "occurrence",
    "work",
    "organization",
    "experiment",
    "fictional occurrence",
    "activity",
    "event"
  ],
  "report_violation_type_normalized": "Value type Q|1190554, Q|386724, Q|43229, Q|101965, Q|14136353, Q|1914636, Q|1656682",
  "report_violation_type_qids": [
    "Q1190554",
    "Q386724",
    "Q43229",
    "Q101965",
    "Q14136353",
    "Q1914636",
    "Q1656682"
  ],
  "report_violation_type_raw": "Value type Q|1190554, Q|386724, Q|43229, Q|101965, Q|14136353, Q|1914636, Q|1656682",
  "value": null,
  "value_current_2026": [
    "Q9678"
  ],
  "value_current_2026_descriptions_en": [
    "22nd edition of Winter Olympics, in Sochi, Russia"
  ],
  "value_current_2026_labels_en": [
    "2014 Winter Olympics"
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
    "description": "event in which a person, organization or creative work was/is a participant; inverse of P710 or P1923",
    "label": "participant in"
  },
  "qid": {
    "description": "American former ski jumper (born 1984)",
    "label": "Lindsey Van"
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
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
      "constraint_qid": "Q21510851",
      "qualifiers": [
        {
          "property_id": "P2306",
          "values": [
            "P1001",
            "P1011",
            "P1013",
            "P1065",
            "P1111",
            "P1132",
            "P121",
            "P12506",
            "P1268",
            "P12912",
            "P131",
            "P1310",
            "P1317",
            "P1319",
            "P1326",
            "P1350",
            "P1351",
            "P1352",
            "P1355",
            "P1356",
            "P1357",
            "P1358",
            "P1359",
            "P1410",
            "... omitted 53 items"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 7,
  "author": "Swpb",
  "before_constraint_count": 7,
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
              "description_en": "type of page in the Wikimedia system. Use with P31 'instance of' for template pages",
              "id": "Q11266439",
              "label_en": "Wikimedia template"
            },
            {
              "description_en": "use with 'instance of' (P31) for modules written in Lua, and working with the Scribunto extension for MediaWiki installed on a Wikimedia wiki",
              "id": "Q15184295",
              "label_en": "Wikimedia module"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "individual team that plays sports",
              "id": "Q12973014",
              "label_en": "sports team"
            },
            {
              "description_en": "place, equipment, or service to support a specific function",
              "id": "Q13226383",
              "label_en": "facility"
            },
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
              "description_en": "craft designed for transportation on or through air, water, or space",
              "id": "Q16391167",
              "label_en": "vessel"
            },
            {
              "description_en": "artistic creation",
              "id": "Q17537576",
              "label_en": "creative work"
            },
            {
              "description_en": "being that has certain capacities or attributes constituting personhood (for humans, use Q5 [human] with P31 [instance of])",
              "id": "Q215627",
              "label_en": "person"
            },
            {
              "description_en": "singular named exemplar of an animal (e.g., the gorilla named Koko; the cat named Socks)",
              "id": "Q26401003",
              "label_en": "individual animal"
            },
            {
              "description_en": "participation of a nation at a competition",
              "id": "Q26887428",
              "label_en": "nation at competition"
            },
            {
              "description_en": "institution for the education of students by teachers",
              "id": "Q3914",
              "label_en": "school"
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
              "description_en": "robot with its body shape built to resemble that of the human body",
              "id": "Q584529",
              "label_en": "humanoid robot"
            },
            {
              "description_en": "general-purpose device for performing arithmetic or logical operations",
              "id": "Q68",
              "label_en": "computer"
            },
            {
              "description_en": "non-tangible executable component of a computer",
              "id": "Q7397",
              "label_en": "software"
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
              "description_en": "usually used as a qualifier",
              "id": "P1011",
              "label_en": "excluding"
            },
            {
              "description_en": "property by which a distinction or classification is made",
              "id": "P1013",
              "label_en": "criterion used"
            },
            {
              "description_en": "URL to the archived web page specified with URL property",
              "id": "P1065",
              "label_en": "archive URL"
            },
            {
              "description_en": "qualifier stating the number of votes for a candidate in an election",
              "id": "P1111",
              "label_en": "votes received"
            },
            {
              "description_en": "number of participants of an event, e.g. people or groups of people that take part in the event (NO units)",
              "id": "P1132",
              "label_en": "number of participants"
            },
            {
              "description_en": "equipment, installation or service operated by the subject",
              "id": "P121",
              "label_en": "item operated"
            },
            {
              "description_en": "latest date beyond which the statement could no longer be true",
              "id": "P12506",
              "label_en": "latest end date"
            },
            {
              "description_en": "organization, individual, or concept that an entity represents",
              "id": "P1268",
              "label_en": "represents"
            },
            {
              "description_en": "specific object to which an occurrence or class of occurrences applies",
              "id": "P12912",
              "label_en": "object of occurrence"
            },
            {
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "entity that disputes a given statement",
              "id": "P1310",
              "label_en": "statement disputed by"
            },
            {
              "description_en": "date when the person was known to be active or alive, when birth or death not documented",
              "id": "P1317",
              "label_en": "floruit"
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
              "description_en": "matches or games a player or a team played during an event. Also a total number of matches a player officially appeared in during the whole career",
              "id": "P1350",
              "label_en": "number of matches played/races/starts"
            },
            {
              "description_en": "goals / points scored in a match or an event used as qualifier to the participant. Use P1358 for league points.",
              "id": "P1351",
              "label_en": "number of points/goals/set scored"
            },
            {
              "description_en": "subject's ordinal position as qualitatively evaluated relative to other members of a group",
              "id": "P1352",
              "label_en": "ranking"
            },
            {
              "description_en": "number of sporting matches, games or events won",
              "id": "P1355",
              "label_en": "number of wins"
            },
            {
              "description_en": "number of sporting matches, games or events lost",
              "id": "P1356",
              "label_en": "number of losses"
            },
            {
              "description_en": "number of matches or games drawn or tied in a league or an event",
              "id": "P1357",
              "label_en": "number of draws/ties"
            },
            {
              "description_en": "number of points in a league table or decathlon. (Use P1351 for goals/points in a match)",
              "id": "P1358",
              "label_en": "points for"
            },
            {
              "description_en": "points conceded or goals against (use in league table items)",
              "id": "P1359",
              "label_en": "number of points/goals conceded"
            },
            {
              "description_en": "number of seats a political party, faction, or group has in a given assembly",
              "id": "P1410",
              "label_en": "number of seats in assembly"
            },
            "... omitted 53 items"
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
              "description_en": "scientific procedure carried out to support, refute, or validate a hypothesis",
              "id": "Q101965",
              "label_en": "experiment"
            },
            {
              "description_en": "occurrence of a fact or object in space-time; instantiation of a property in an object",
              "id": "Q1190554",
              "label_en": "occurrence"
            },
            {
              "description_en": "happening which occurs in a fiction",
              "id": "Q14136353",
              "label_en": "fictional occurrence"
            },
            {
              "description_en": "temporary and scheduled happening, like a conference, festival, competition or similar",
              "id": "Q1656682",
              "label_en": "event"
            },
            {
              "description_en": "series of actions done by an agent which results in an external change of state",
              "id": "Q1914636",
              "label_en": "activity"
            },
            {
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
            },
            {
              "description_en": "social entity established to meet needs or pursue goals",
              "id": "Q43229",
              "label_en": "organization"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the subject and the object have to coincide or coexist at some point of history",
          "id": "Q25796498",
          "label_en": "contemporary constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "English rock band (1968–1980)",
              "id": "Q2331",
              "label_en": "Led Zeppelin"
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
              "description_en": "type of page in the Wikimedia system. Use with P31 'instance of' for template pages",
              "id": "Q11266439",
              "label_en": "Wikimedia template"
            },
            {
              "description_en": "use with 'instance of' (P31) for modules written in Lua, and working with the Scribunto extension for MediaWiki installed on a Wikimedia wiki",
              "id": "Q15184295",
              "label_en": "Wikimedia module"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "id": "Q21503250",
          "label_en": "subject type constraint"
        },
        "parameters": {
          "P2308": [
            {
              "description_en": "individual team that plays sports",
              "id": "Q12973014",
              "label_en": "sports team"
            },
            {
              "description_en": "place, equipment, or service to support a specific function",
              "id": "Q13226383",
              "label_en": "facility"
            },
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
              "description_en": "craft designed for transportation on or through air, water, or space",
              "id": "Q16391167",
              "label_en": "vessel"
            },
            {
              "description_en": "artistic creation",
              "id": "Q17537576",
              "label_en": "creative work"
            },
            {
              "description_en": "being that has certain capacities or attributes constituting personhood (for humans, use Q5 [human] with P31 [instance of])",
              "id": "Q215627",
              "label_en": "person"
            },
            {
              "description_en": "singular named exemplar of an animal (e.g., the gorilla named Koko; the cat named Socks)",
              "id": "Q26401003",
              "label_en": "individual animal"
            },
            {
              "description_en": "participation of a nation at a competition",
              "id": "Q26887428",
              "label_en": "nation at competition"
            },
            {
              "description_en": "institution for the education of students by teachers",
              "id": "Q3914",
              "label_en": "school"
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
              "description_en": "robot with its body shape built to resemble that of the human body",
              "id": "Q584529",
              "label_en": "humanoid robot"
            },
            {
              "description_en": "general-purpose device for performing arithmetic or logical operations",
              "id": "Q68",
              "label_en": "computer"
            },
            {
              "description_en": "non-tangible executable component of a computer",
              "id": "Q7397",
              "label_en": "software"
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
              "description_en": "usually used as a qualifier",
              "id": "P1011",
              "label_en": "excluding"
            },
            {
              "description_en": "property by which a distinction or classification is made",
              "id": "P1013",
              "label_en": "criterion used"
            },
            {
              "description_en": "URL to the archived web page specified with URL property",
              "id": "P1065",
              "label_en": "archive URL"
            },
            {
              "description_en": "qualifier stating the number of votes for a candidate in an election",
              "id": "P1111",
              "label_en": "votes received"
            },
            {
              "description_en": "number of participants of an event, e.g. people or groups of people that take part in the event (NO units)",
              "id": "P1132",
              "label_en": "number of participants"
            },
            {
              "description_en": "latest date beyond which the statement could no longer be true",
              "id": "P12506",
              "label_en": "latest end date"
            },
            {
              "description_en": "organization, individual, or concept that an entity represents",
              "id": "P1268",
              "label_en": "represents"
            },
            {
              "description_en": "specific object to which an occurrence or class of occurrences applies",
              "id": "P12912",
              "label_en": "object of occurrence"
            },
            {
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
            },
            {
              "description_en": "entity that disputes a given statement",
              "id": "P1310",
              "label_en": "statement disputed by"
            },
            {
              "description_en": "date when the person was known to be active or alive, when birth or death not documented",
              "id": "P1317",
              "label_en": "floruit"
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
              "description_en": "matches or games a player or a team played during an event. Also a total number of matches a player officially appeared in during the whole career",
              "id": "P1350",
              "label_en": "number of matches played/races/starts"
            },
            {
              "description_en": "goals / points scored in a match or an event used as qualifier to the participant. Use P1358 for league points.",
              "id": "P1351",
              "label_en": "number of points/goals/set scored"
            },
            {
              "description_en": "subject's ordinal position as qualitatively evaluated relative to other members of a group",
              "id": "P1352",
              "label_en": "ranking"
            },
            {
              "description_en": "number of sporting matches, games or events won",
              "id": "P1355",
              "label_en": "number of wins"
            },
            {
              "description_en": "number of sporting matches, games or events lost",
              "id": "P1356",
              "label_en": "number of losses"
            },
            {
              "description_en": "number of matches or games drawn or tied in a league or an event",
              "id": "P1357",
              "label_en": "number of draws/ties"
            },
            {
              "description_en": "number of points in a league table or decathlon. (Use P1351 for goals/points in a match)",
              "id": "P1358",
              "label_en": "points for"
            },
            {
              "description_en": "points conceded or goals against (use in league table items)",
              "id": "P1359",
              "label_en": "number of points/goals conceded"
            },
            {
              "description_en": "number of seats a political party, faction, or group has in a given assembly",
              "id": "P1410",
              "label_en": "number of seats in assembly"
            },
            {
              "description_en": "organization that a person or organization is affiliated with (not necessarily member of or employed by)",
              "id": "P1416",
              "label_en": "affiliation"
            },
            "... omitted 52 items"
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
              "description_en": "scientific procedure carried out to support, refute, or validate a hypothesis",
              "id": "Q101965",
              "label_en": "experiment"
            },
            {
              "description_en": "occurrence of a fact or object in space-time; instantiation of a property in an object",
              "id": "Q1190554",
              "label_en": "occurrence"
            },
            {
              "description_en": "happening which occurs in a fiction",
              "id": "Q14136353",
              "label_en": "fictional occurrence"
            },
            {
              "description_en": "temporary and scheduled happening, like a conference, festival, competition or similar",
              "id": "Q1656682",
              "label_en": "event"
            },
            {
              "description_en": "series of actions done by an agent which results in an external change of state",
              "id": "Q1914636",
              "label_en": "activity"
            },
            {
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
            },
            {
              "description_en": "social entity established to meet needs or pursue goals",
              "id": "Q43229",
              "label_en": "organization"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the subject and the object have to coincide or coexist at some point of history",
          "id": "Q25796498",
          "label_en": "contemporary constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "English rock band (1968–1980)",
              "id": "Q2331",
              "label_en": "Led Zeppelin"
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
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "6545e4945ca6c09cf83514519b18146d39a5f7ce",
  "hash_before": "ca50d932ee6ddd7258eaedaed5c8392f6306bddd",
  "property_revision_id": 2444117279,
  "property_revision_prev": 2439333402,
  "qualifier_value_changes": [
    {
      "added_values": [
        "P121"
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
            "P1011",
            "P1013",
            "P1065",
            "P1111",
            "P1132",
            "P12506",
            "P1268",
            "P12912",
            "P131",
            "P1310",
            "P1317",
            "P1319",
            "P1326",
            "P1350",
            "P1351",
            "P1352",
            "P1355",
            "P1356",
            "P1357",
            "P1358",
            "P1359",
            "P1410",
            "P1416",
            "... omitted 52 items"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "conflicts-with constraint: item of property constraint: Wikimedia template, Wikimedia module, Wikimedia disambiguation page, Wikimedia category; property: instance of; constraint status: mandatory constraint",
      "subject type constraint: class: sports team, facility, group of fictional characters, group of humans, vessel, creative work, person, individual animal, nation at competition, school, organization, human, humanoid robot, computer, software, character; relation: instance or subclass of",
      "allowed qualifiers constraint: property: applies to jurisdiction, excluding, criterion used, archive URL, votes received, number of participants, item operated, latest end date, represents, object of occurrence, located in the administrative territorial entity, statement disputed by, floruit, earliest date, latest date, number of matches played/races/starts, number of points/goals/set scored, ranking, number of wins, number of losses, number of draws/ties, points ... [truncated 1015 chars]",
      "value-type constraint: class: experiment, occurrence, fictional occurrence, event, activity, work, organization; relation: instance or subclass of",
      "contemporary constraint: exception to constraint: Led Zeppelin",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы",
      "property scope constraint: property scope: as main value, as qualifier"
    ],
    "before": [
      "conflicts-with constraint: item of property constraint: Wikimedia template, Wikimedia module, Wikimedia disambiguation page, Wikimedia category; property: instance of; constraint status: mandatory constraint",
      "subject type constraint: class: sports team, facility, group of fictional characters, group of humans, vessel, creative work, person, individual animal, nation at competition, school, organization, human, humanoid robot, computer, software, character; relation: instance or subclass of",
      "allowed qualifiers constraint: property: applies to jurisdiction, excluding, criterion used, archive URL, votes received, number of participants, latest end date, represents, object of occurrence, located in the administrative territorial entity, statement disputed by, floruit, earliest date, latest date, number of matches played/races/starts, number of points/goals/set scored, ranking, number of wins, number of losses, number of draws/ties, points for, number of ... [truncated 1000 chars]",
      "value-type constraint: class: experiment, occurrence, fictional occurrence, event, activity, work, organization; relation: instance or subclass of",
      "contemporary constraint: exception to constraint: Led Zeppelin",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы",
      "property scope constraint: property scope: as main value, as qualifier"
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
    "violation_name": "Value type Q|1190554, Q|386724, Q|43229, Q|101965, Q|14136353, Q|1914636, Q|1656682"
  },
  {
    "result": "Q21502838",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 010. `reform_Q19660394_P347_2436835478`

| Field | Value |
|---|---|
| qid | Q19660394 |
| property | P347 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | TBOX::P347::2436835478 |
| tbox_revision_key | TBOX::P347::2436835478 |

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
  "author": "Nono314",
  "kind": "T_BOX",
  "property_revision_id": 2436835478,
  "property_revision_prev": 2430400733
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-08T10:11:43",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P347",
  "report_revision_new": 2439557139,
  "report_revision_old": 2439159573,
  "report_violation_type": "Item P|2049",
  "report_violation_type_normalized": "Item P|2049",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|2049",
  "report_violation_types": [
    "Item P|2049",
    "Item P|2048"
  ],
  "value": null,
  "value_current_2026": [
    "05620003668"
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
    "description": "identifier in the Joconde database of the French Ministry of Culture",
    "label": "Joconde work ID"
  },
  "qid": {
    "description": "painting by Canaletto (Musée des Augustins)",
    "label": "View of the Piazza San Marco, Venice"
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
    "label_en": "single-best-value constraint",
    "qid": "Q52060874"
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
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [],
  "after_constraint_count": 18,
  "author": "Nono314",
  "before_constraint_count": 19,
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
              "value": "[0-9A-Z][\\-0-9A-Za-z]{10}"
            }
          ],
          "P2303": [
            {
              "description_en": "céramique grecque conservée au musée Antoine-Vivenel",
              "id": "Q38620045",
              "label_en": "Coupe type B V 1090"
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
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "id": "Q21502404",
          "label_en": "format constraint"
        },
        "parameters": {
          "P1793": [
            {
              "value": "[0-9A-Z][\\-0-9A-Za-z]{10}"
            }
          ],
          "P2303": [
            {
              "description_en": "céramique grecque conservée au musée Antoine-Vivenel",
              "id": "Q38620045",
              "label_en": "Coupe type B V 1090"
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
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that this property generally contains a single “best” value per item, though other values may be included as long as the “best” value is marked with preferred rank",
          "id": "Q52060874",
          "label_en": "single-best-value constraint"
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
    ]
  },
  "hash_after": "397e7d698bdfbe2b67a6e8857db17e8e31033b49",
  "hash_before": "ccc33528971b9e2cc1eac655d494ed3d60e22d32",
  "property_revision_id": 2436835478,
  "property_revision_prev": 2430400733,
  "qualifier_value_changes": [
    {
      "added_values": [
        "P170"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2306",
      "removed_values": [
        "P17"
      ],
      "same_qid_index": 1
    },
    {
      "added_values": [
        "P180"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2306",
      "removed_values": [
        "P170"
      ],
      "same_qid_index": 2
    },
    {
      "added_values": [
        "P186"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2306",
      "removed_values": [
        "P180"
      ],
      "same_qid_index": 3
    },
    {
      "added_values": [
        "P195"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2306",
      "removed_values": [
        "P186"
      ],
      "same_qid_index": 4
    },
    {
      "added_values": [
        "Q21502408"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2316",
      "removed_values": [
        "Q62026391"
      ],
      "same_qid_index": 4
    },
    {
      "added_values": [
        "P2048"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2306",
      "removed_values": [
        "P195"
      ],
      "same_qid_index": 5
    },
    {
      "added_values": [
        "Q62026391"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2316",
      "removed_values": [
        "Q21502408"
      ],
      "same_qid_index": 5
    },
    {
      "added_values": [
        "P2049"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2306",
      "removed_values": [
        "P2048"
      ],
      "same_qid_index": 6
    },
    {
      "added_values": [
        "P217"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2306",
      "removed_values": [
        "P2049"
      ],
      "same_qid_index": 7
    },
    {
      "added_values": [],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2316",
      "removed_values": [
        "Q62026391"
      ],
      "same_qid_index": 7
    },
    {
      "added_values": [
        "P276"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2306",
      "removed_values": [
        "P217"
      ],
      "same_qid_index": 8
    },
    {
      "added_values": [
        "Q62026391"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2316",
      "removed_values": [],
      "same_qid_index": 8
    },
    {
      "added_values": [
        "P571"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2306",
      "removed_values": [
        "P276"
      ],
      "same_qid_index": 9
    },
    {
      "added_values": [
        "P6216"
      ],
      "constraint_qid": "Q21503247",
      "qualifier_property": "P2306",
      "removed_values": [
        "P571"
      ],
      "same_qid_index": 10
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q21503247",
      "qualifiers": [
        {
          "property_id": "P2306",
          "values": [
            "P17"
          ]
        },
        {
          "property_id": "P2316",
          "values": [
            "Q62026391"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "label in language constraint: Wikimedia language code: fr",
      "format constraint: format as a regular expression: [0-9A-Z][\\-0-9A-Za-z]{10}; exception to constraint: Coupe type B V 1090",
      "distinct-values constraint: no qualifiers recorded",
      "item-requires-statement constraint: property: genre; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: creator; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: depicts; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: made from material; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: collection; constraint status: mandatory constraint",
      "item-requires-statement constraint: property: height; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: width; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inventory number",
      "item-requires-statement constraint: property: location; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inception; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: copyright status; constraint status: suggestion constraint",
      "subject type constraint: class: artwork series, work; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "single-best-value constraint: no qualifiers recorded",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as reference"
    ],
    "before": [
      "label in language constraint: Wikimedia language code: fr",
      "format constraint: format as a regular expression: [0-9A-Z][\\-0-9A-Za-z]{10}; exception to constraint: Coupe type B V 1090",
      "distinct-values constraint: no qualifiers recorded",
      "item-requires-statement constraint: property: genre; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: country; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: creator; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: depicts; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: made from material; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: collection; constraint status: mandatory constraint",
      "item-requires-statement constraint: property: height; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: width; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inventory number",
      "item-requires-statement constraint: property: location; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: inception; constraint status: suggestion constraint",
      "item-requires-statement constraint: property: copyright status; constraint status: suggestion constraint",
      "subject type constraint: class: artwork series, work; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "single-best-value constraint: no qualifiers recorded",
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
    "violation_name": "Item P|2049"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 011. `reform_Q24030458_P373_2442604045`

| Field | Value |
|---|---|
| qid | Q24030458 |
| property | P373 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q21502838 |
| group_key | TBOX::P373::2442604045 |
| tbox_revision_key | TBOX::P373::2442604045 |

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
  "author": "Like the windows",
  "kind": "T_BOX",
  "property_revision_id": 2442604045,
  "property_revision_prev": 2442603848
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-16T11:51:46",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P373",
  "report_revision_new": 2442981743,
  "report_revision_old": 2442645840,
  "report_violation_type": "Commons link",
  "report_violation_type_normalized": "Commons link",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Commons link",
  "value": null,
  "value_current_2026": [
    "Kyrgyzstan in the 1970s"
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
    "description": "name of the Wikimedia Commons category containing files related to this item (without the prefix \"Category:\")",
    "label": "Commons category"
  },
  "qid": {
    "description": "Wikimedia category",
    "label": "Category:1970s in the Kirghiz Soviet Socialist Republic"
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
            "(?!:?Category:)[^{}\\[\\]<>]+"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 6,
  "author": "Like the windows",
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
          "P2303": [
            {
              "description_en": "family name",
              "id": "Q30091418",
              "label_en": "Hussein"
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
              "value": "(?!:?Category:)[^{}\\[\\]<>]+"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value must link to an existing Wikimedia Commons page",
          "id": "Q21510852",
          "label_en": "Commons link constraint"
        },
        "parameters": {
          "P2307": [
            {
              "value": "Category"
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
          "P2303": [
            {
              "description_en": "family name",
              "id": "Q30091418",
              "label_en": "Hussein"
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
              "value": "Category"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value must link to an existing Wikimedia Commons page",
          "id": "Q21510852",
          "label_en": "Commons link constraint"
        },
        "parameters": {
          "P2307": [
            {
              "value": "Category"
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
  "hash_after": "847cc606259b0433da724dc728d5a1a47ae1647c",
  "hash_before": "0b169af24e591bdd0ee72e9b6a399894100bd79e",
  "property_revision_id": 2442604045,
  "property_revision_prev": 2442603848,
  "qualifier_value_changes": [
    {
      "added_values": [
        "(?!:?Category:)[^{}\\[\\]<>]+"
      ],
      "constraint_qid": "Q21502404",
      "qualifier_property": "P1793",
      "removed_values": [
        "Category"
      ],
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
            "Category"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "single-value constraint: exception to constraint: Hussein",
      "format constraint: format as a regular expression: (?!:?Category:)[^{}\\[\\]<>]+",
      "conflicts-with constraint: item of property constraint: Wikimedia template; property: instance of",
      "Commons link constraint: Wikimedia Commons namespace: Category",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ],
    "before": [
      "single-value constraint: exception to constraint: Hussein",
      "format constraint: format as a regular expression: Category",
      "conflicts-with constraint: item of property constraint: Wikimedia template; property: instance of",
      "Commons link constraint: Wikimedia Commons namespace: Category",
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
    "violation_name": "Commons link"
  },
  {
    "result": "Q19474404",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 012. `reform_Q28606688_P910_2444366220`

| Field | Value |
|---|---|
| qid | Q28606688 |
| property | P910 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q52060874 |
| group_key | TBOX::P910::2444366220 |
| tbox_revision_key | TBOX::P910::2444366220 |

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
  "property_revision_id": 2444366220,
  "property_revision_prev": 2430910127
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-21T09:21:20",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P910",
  "report_revision_new": 2444868799,
  "report_revision_old": 2444440134,
  "report_violation_type": "Inverse",
  "report_violation_type_normalized": "Inverse",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Inverse",
  "value": null,
  "value_current_2026": [
    "Q9036179"
  ],
  "value_current_2026_descriptions_en": [
    "Wikimedia category"
  ],
  "value_current_2026_labels_en": [
    "Category:National parks of Oceania"
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
    "description": "main Wikimedia category",
    "label": "topic's main category"
  },
  "qid": {
    "description": null,
    "label": "national parks of Oceania"
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
    "label_en": "inverse constraint",
    "qid": "Q21510855"
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
      "constraint_qid": "Q52060874",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q198"
          ]
        },
        {
          "property_id": "P2316",
          "values": [
            "Q62026391"
          ]
        },
        {
          "property_id": "P4155",
          "values": [
            "P1545",
            "P4224",
            "P518"
          ]
        },
        {
          "property_id": "P6607",
          "values": [
            "separator used for gendered categories@en"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 8,
  "author": "Clemens Dulcis",
  "before_constraint_count": 8,
  "changed_constraint_types": [],
  "constraints_readable_en": {
    "after": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "textile floor covering",
              "id": "Q104602244",
              "label_en": "rug"
            },
            {
              "description_en": "college that is predominantly funded by public means",
              "id": "Q104694724",
              "label_en": "public college"
            },
            {
              "description_en": "Professional in the make-up arts for film, television and stage",
              "id": "Q110864961",
              "label_en": "Film- und Bühnencosmetologe"
            },
            {
              "description_en": "the supreme body of the regional organisation of the CPSU between regional party conferences",
              "id": "Q115324747",
              "label_en": "regional committee of the CPSU"
            },
            {
              "description_en": "individual whose gender identity (man, woman, other) or expression (masculine, feminine, other) is different from their sex (male, female) assigned at birth or gender binarism",
              "id": "Q11894636",
              "label_en": "gender minority"
            },
            {
              "description_en": "textile floor covering",
              "id": "Q163446",
              "label_en": "carpet"
            },
            {
              "description_en": "state, condition, or behavior in which a person's identity does not conform unambiguously to conventional notions of male or female sex",
              "id": "Q4135211",
              "label_en": "gender non-conformity"
            },
            {
              "description_en": "высший орган областной организации КПСС между областными партийными конференциями",
              "id": "Q4329294",
              "label_en": "regional committee of the CPSU"
            },
            {
              "description_en": "university predominantly funded by public means",
              "id": "Q875538",
              "label_en": "public university"
            },
            {
              "description_en": "person who applies make-up, style and applies special effects make-up, prosthetic make-up",
              "id": "Q935666",
              "label_en": "make-up artist"
            }
          ],
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "primary topic of the subject Wikimedia category",
              "id": "P301",
              "label_en": "category's main topic"
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
              "description_en": "primary topic of the subject Wikimedia category",
              "id": "P301",
              "label_en": "category's main topic"
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
          "P2303": [
            {
              "description_en": "organized and prolonged violent conflict between different nations, states, or different groups within a nation or state",
              "id": "Q198",
              "label_en": "war"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ],
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
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
            }
          ],
          "P6607": [
            {
              "value": "separator used for gendered categories@en"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "id": "Q21502410",
          "label_en": "distinct-values constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "textile floor covering",
              "id": "Q104602244",
              "label_en": "rug"
            },
            {
              "description_en": "college that is predominantly funded by public means",
              "id": "Q104694724",
              "label_en": "public college"
            },
            {
              "description_en": "Professional in the make-up arts for film, television and stage",
              "id": "Q110864961",
              "label_en": "Film- und Bühnencosmetologe"
            },
            {
              "description_en": "the supreme body of the regional organisation of the CPSU between regional party conferences",
              "id": "Q115324747",
              "label_en": "regional committee of the CPSU"
            },
            {
              "description_en": "individual whose gender identity (man, woman, other) or expression (masculine, feminine, other) is different from their sex (male, female) assigned at birth or gender binarism",
              "id": "Q11894636",
              "label_en": "gender minority"
            },
            {
              "description_en": "textile floor covering",
              "id": "Q163446",
              "label_en": "carpet"
            },
            {
              "description_en": "state, condition, or behavior in which a person's identity does not conform unambiguously to conventional notions of male or female sex",
              "id": "Q4135211",
              "label_en": "gender non-conformity"
            },
            {
              "description_en": "высший орган областной организации КПСС между областными партийными конференциями",
              "id": "Q4329294",
              "label_en": "regional committee of the CPSU"
            },
            {
              "description_en": "university predominantly funded by public means",
              "id": "Q875538",
              "label_en": "public university"
            },
            {
              "description_en": "person who applies make-up, style and applies special effects make-up, prosthetic make-up",
              "id": "Q935666",
              "label_en": "make-up artist"
            }
          ],
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "primary topic of the subject Wikimedia category",
              "id": "P301",
              "label_en": "category's main topic"
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
              "description_en": "primary topic of the subject Wikimedia category",
              "id": "P301",
              "label_en": "category's main topic"
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
          "P2303": [
            {
              "description_en": "organized and prolonged violent conflict between different nations, states, or different groups within a nation or state",
              "id": "Q198",
              "label_en": "war"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ],
          "P4155": [
            {
              "description_en": "category contains elements that are instances of this item",
              "id": "P4224",
              "label_en": "category contains"
            },
            {
              "description_en": "part, aspect, or form of the item to which the claim applies",
              "id": "P518",
              "label_en": "applies to part"
            }
          ],
          "P6607": [
            {
              "value": "separator used for gendered categories@en"
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
  "hash_after": "c04c8b7bba53756644b83d744bd63d65b3af1b1a",
  "hash_before": "243303b03a05ec9a4d9d29506693772706b73942",
  "property_revision_id": 2444366220,
  "property_revision_prev": 2430910127,
  "qualifier_value_changes": [
    {
      "added_values": [
        "P1545"
      ],
      "constraint_qid": "Q52060874",
      "qualifier_property": "P4155",
      "removed_values": [],
      "same_qid_index": 0
    }
  ],
  "removed_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q52060874",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q198"
          ]
        },
        {
          "property_id": "P2316",
          "values": [
            "Q62026391"
          ]
        },
        {
          "property_id": "P4155",
          "values": [
            "P4224",
            "P518"
          ]
        },
        {
          "property_id": "P6607",
          "values": [
            "separator used for gendered categories@en"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "distinct-values constraint: exception to constraint: rug, public college, Film- und Bühnencosmetologe, regional committee of the CPSU, gender minority, carpet, gender non-conformity, regional committee of the CPSU, public university, make-up artist; separator: together with",
      "conflicts-with constraint: item of property constraint: Wikimedia template, Wikimedia disambiguation page, Wikimedia category; property: instance of",
      "conflicts-with constraint: property: category's main topic",
      "inverse constraint: property: category's main topic",
      "value-type constraint: class: Wikimedia category; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "single-best-value constraint: exception to constraint: war; constraint status: suggestion constraint; separator: series ordinal, category contains, applies to part; constraint clarification: separator used for gendered categories@en",
      "property scope constraint: property scope: as main value"
    ],
    "before": [
      "distinct-values constraint: exception to constraint: rug, public college, Film- und Bühnencosmetologe, regional committee of the CPSU, gender minority, carpet, gender non-conformity, regional committee of the CPSU, public university, make-up artist; separator: together with",
      "conflicts-with constraint: item of property constraint: Wikimedia template, Wikimedia disambiguation page, Wikimedia category; property: instance of",
      "conflicts-with constraint: property: category's main topic",
      "inverse constraint: property: category's main topic",
      "value-type constraint: class: Wikimedia category; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "single-best-value constraint: exception to constraint: war; constraint status: suggestion constraint; separator: category contains, applies to part; constraint clarification: separator used for gendered categories@en",
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
    "result": "Q21502410",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 013. `reform_Q306233_P4654_696602063`

| Field | Value |
|---|---|
| qid | Q306233 |
| property | P4654 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | TBOX::P4654::696602063 |
| tbox_revision_key | TBOX::P4654::696602063 |

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
  "author": "MisterSynergy",
  "kind": "T_BOX",
  "property_revision_id": 696602063,
  "property_revision_prev": 690337647
}
```

### Violation Context

```json
{
  "report_fix_date": "2018-06-19T20:13:56",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P4654",
  "report_revision_new": 698738891,
  "report_revision_old": 698422158,
  "report_violation_type": "Type Q|Q174989",
  "report_violation_type_descriptions_en": [
    "concrete format or program for storing files and directories on a data storage device"
  ],
  "report_violation_type_labels_en": [
    "file system"
  ],
  "report_violation_type_normalized": "Type Q|Q174989",
  "report_violation_type_qids": [
    "Q174989"
  ],
  "report_violation_type_raw": "Type Q|Q174989",
  "report_violation_types": [
    "Type Q|Q174989",
    "Mandatory Qualifiers"
  ],
  "value": null,
  "value_current_2026": [
    "0x07",
    "EBD0A0A2-B9E5-4433-87C0-68B6B72699C7"
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
    "description": "string that contains the identifier of a partition type specified in a partition table",
    "label": "partition type identifier"
  },
  "qid": {
    "description": "non-journaled interoperable file system friendly for flash memory and allowing to overcome FAT32 limitations",
    "label": "exFAT"
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
    "label_en": "required qualifier constraint",
    "qid": "Q21510856"
  },
  {
    "label_en": "allowed qualifiers constraint",
    "qid": "Q21510851"
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
            "Q54828448"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 6,
  "author": "MisterSynergy",
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
              "description_en": "concrete format or program for storing files and directories on a data storage device",
              "id": "Q174989",
              "label_en": "file system"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "operating system (OS) on which a software works or the OS installed on hardware",
              "id": "P306",
              "label_en": "operating system"
            },
            {
              "description_en": "format according to which a code is to be interpreted (use only as a qualifier)",
              "id": "P3294",
              "label_en": "encoding"
            },
            {
              "description_en": "qualifier for \"partition identifier\" that determines what type of partition table is used with the identifier",
              "id": "P4653",
              "label_en": "partition table type"
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
              "description_en": "format according to which a code is to be interpreted (use only as a qualifier)",
              "id": "P3294",
              "label_en": "encoding"
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
              "description_en": "qualifier for \"partition identifier\" that determines what type of partition table is used with the identifier",
              "id": "P4653",
              "label_en": "partition table type"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property can only be used as a property for values, not as a qualifier or reference",
          "id": "Q21528958",
          "label_en": "used for values only constraint"
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
              "description_en": "concrete format or program for storing files and directories on a data storage device",
              "id": "Q174989",
              "label_en": "file system"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "operating system (OS) on which a software works or the OS installed on hardware",
              "id": "P306",
              "label_en": "operating system"
            },
            {
              "description_en": "format according to which a code is to be interpreted (use only as a qualifier)",
              "id": "P3294",
              "label_en": "encoding"
            },
            {
              "description_en": "qualifier for \"partition identifier\" that determines what type of partition table is used with the identifier",
              "id": "P4653",
              "label_en": "partition table type"
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
              "description_en": "format according to which a code is to be interpreted (use only as a qualifier)",
              "id": "P3294",
              "label_en": "encoding"
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
              "description_en": "qualifier for \"partition identifier\" that determines what type of partition table is used with the identifier",
              "id": "P4653",
              "label_en": "partition table type"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property can only be used as a property for values, not as a qualifier or reference",
          "id": "Q21528958",
          "label_en": "used for values only constraint"
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
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "0f25a001034b1c92a392b93b1b58a4e09b84ac14",
  "hash_before": "01050cc0dfe8913a0c1fca51cba0f1d6968d910e",
  "property_revision_id": 696602063,
  "property_revision_prev": 690337647,
  "qualifier_value_changes": [
    {
      "added_values": [],
      "constraint_qid": "Q53869507",
      "qualifier_property": "P4680",
      "removed_values": [
        "Q46466787"
      ],
      "same_qid_index": 0
    },
    {
      "added_values": [
        "Q54828448"
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
            "Q46466787"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "subject type constraint: class: file system; relation: subclass of",
      "allowed qualifiers constraint: property: operating system, encoding, partition table type",
      "required qualifier constraint: property: encoding",
      "required qualifier constraint: property: partition table type",
      "used for values only constraint: no qualifiers recorded",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value"
    ],
    "before": [
      "subject type constraint: class: file system; relation: subclass of",
      "allowed qualifiers constraint: property: operating system, encoding, partition table type",
      "required qualifier constraint: property: encoding",
      "required qualifier constraint: property: partition table type",
      "used for values only constraint: no qualifiers recorded",
      "property scope constraint: constraint status: mandatory constraint; constraint scope: constraint checked on main value"
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
    "violation_name": "Type Q|Q174989"
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
    "result": "SCHEMA_UPDATE",
    "step": "set_semantics"
  }
]
```

---

## 014. `reform_Q413940_P2231_696609154`

| Field | Value |
|---|---|
| qid | Q413940 |
| property | P2231 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q21514353 |
| group_key | TBOX::P2231::696609154 |
| tbox_revision_key | TBOX::P2231::696609154 |

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
  "author": "MisterSynergy",
  "kind": "T_BOX",
  "property_revision_id": 696609154,
  "property_revision_prev": 690333194
}
```

### Violation Context

```json
{
  "report_fix_date": "2018-06-19T21:06:13",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P2231",
  "report_revision_new": 698758341,
  "report_revision_old": 698394696,
  "report_violation_type": "Mandatory Qualifiers",
  "report_violation_type_normalized": "Mandatory Qualifiers",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Mandatory Qualifiers",
  "value": null,
  "value_current_2026": [
    "+10100 http://www.wikidata.org/entity/Q182429"
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
    "description": "detonating velocity is explosively measured",
    "label": "explosive velocity"
  },
  "qid": {
    "description": "chemical compound",
    "label": "octanitrocubane"
  }
}
```

### Constraint Types

```json
[
  {
    "label_en": "allowed units constraint",
    "qid": "Q21514353"
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
    "label_en": "citation-needed constraint",
    "qid": "Q54554025"
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
            "Q54828448"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the listed qualifier has to be used",
          "id": "Q21510856",
          "label_en": "required qualifier constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "density of a substance with phase of matter and temperature as qualifiers",
              "id": "P2054",
              "label_en": "density"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that only listed units may be used",
          "id": "Q21514353",
          "label_en": "allowed units constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "SI unit of speed and velocity",
              "id": "Q182429",
              "label_en": "metre per second"
            },
            {
              "description_en": "unit of speed",
              "id": "Q4220561",
              "label_en": "kilometre per second"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property can only be used as a property for values, not as a qualifier or reference",
          "id": "Q21528958",
          "label_en": "used for values only constraint"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that the listed qualifier has to be used",
          "id": "Q21510856",
          "label_en": "required qualifier constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "density of a substance with phase of matter and temperature as qualifiers",
              "id": "P2054",
              "label_en": "density"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that only listed units may be used",
          "id": "Q21514353",
          "label_en": "allowed units constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "SI unit of speed and velocity",
              "id": "Q182429",
              "label_en": "metre per second"
            },
            {
              "description_en": "unit of speed",
              "id": "Q4220561",
              "label_en": "kilometre per second"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that a property can only be used as a property for values, not as a qualifier or reference",
          "id": "Q21528958",
          "label_en": "used for values only constraint"
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
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "e039b273ce466ecd39b11bd71fe415a3a1707b3b",
  "hash_before": "bcb63409e6225d4c47fee3867be315a7406bb881",
  "property_revision_id": 696609154,
  "property_revision_prev": 690333194,
  "qualifier_value_changes": [
    {
      "added_values": [],
      "constraint_qid": "Q53869507",
      "qualifier_property": "P4680",
      "removed_values": [
        "Q46466787"
      ],
      "same_qid_index": 0
    },
    {
      "added_values": [
        "Q54828448"
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
            "Q46466787"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "required qualifier constraint: property: density",
      "allowed units constraint: item of property constraint: metre per second, kilometre per second",
      "used for values only constraint: no qualifiers recorded",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value"
    ],
    "before": [
      "required qualifier constraint: property: density",
      "allowed units constraint: item of property constraint: metre per second, kilometre per second",
      "used for values only constraint: no qualifiers recorded",
      "property scope constraint: constraint status: mandatory constraint; constraint scope: constraint checked on main value"
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
    "violation_name": "Mandatory Qualifiers"
  },
  {
    "result": "Q21510856",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 015. `reform_Q4157219_P856_2442691260`

| Field | Value |
|---|---|
| qid | Q4157219 |
| property | P856 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | tail |
| constraint_family | Q21502404 |
| group_key | TBOX::P856::2442691260 |
| tbox_revision_key | TBOX::P856::2442691260 |

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
  "author": "Legonin",
  "kind": "T_BOX",
  "property_revision_id": 2442691260,
  "property_revision_prev": 2442687184
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-18T17:47:44",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P856",
  "report_revision_new": 2443817852,
  "report_revision_old": 2443378012,
  "report_violation_type": "Item P|31",
  "report_violation_type_normalized": "Item P|31",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|31",
  "value": null,
  "value_current_2026": [
    "http://kostino.org"
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
    "label": "Деловой и культурный центр «Костино»"
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
      "constraint_qid": "Q21502410",
      "qualifiers": [
        {
          "property_id": "P2303",
          "values": [
            "Q104417848",
            "Q104533647",
            "Q105536068",
            "Q105550536",
            "Q106426688",
            "Q108001817",
            "Q111225168",
            "Q111982999",
            "Q112121329",
            "Q112126763",
            "Q113117660",
            "Q113133561",
            "Q113293536",
            "Q113297824",
            "Q113634129",
            "Q113634693",
            "Q113685547",
            "Q113685550",
            "Q114458607",
            "Q114882717",
            "Q115336397",
            "Q115384119",
            "Q115443051",
            "Q115553783",
            "... omitted 88 items"
          ]
        },
        {
          "property_id": "P4155",
          "values": [
            "P1476",
            "P1706",
            "P2868",
            "P407",
            "P518",
            "P958"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 12,
  "author": "Legonin",
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
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
            {
              "description_en": "for works, when the title is followed by a subtitle",
              "id": "P1680",
              "label_en": "subtitle"
            },
            {
              "description_en": "qualifier to specify the item that this property is shared with",
              "id": "P1706",
              "label_en": "together with"
            },
            "... omitted 43 items"
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
            "... omitted 86 items"
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
              "description_en": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
              "id": "P1552",
              "label_en": "has characteristic"
            },
            {
              "description_en": "for works, when the title is followed by a subtitle",
              "id": "P1680",
              "label_en": "subtitle"
            },
            {
              "description_en": "qualifier to specify the item that this property is shared with",
              "id": "P1706",
              "label_en": "together with"
            },
            "... omitted 43 items"
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
  "hash_after": "9cebba9c089543e647dd61637ec2b182fddf9841",
  "hash_before": "eb15cf149d21b5d909a97543682e7756361d93c0",
  "property_revision_id": 2442691260,
  "property_revision_prev": 2442687184,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q2994765",
        "Q76267992"
      ],
      "constraint_qid": "Q21502410",
      "qualifier_property": "P2303",
      "removed_values": [],
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
            "Q104417848",
            "Q104533647",
            "Q105536068",
            "Q105550536",
            "Q106426688",
            "Q108001817",
            "Q111225168",
            "Q111982999",
            "Q112121329",
            "Q112126763",
            "Q113117660",
            "Q113133561",
            "Q113293536",
            "Q113297824",
            "Q113634129",
            "Q113634693",
            "Q113685547",
            "Q113685550",
            "Q114458607",
            "Q114882717",
            "Q115336397",
            "Q115384119",
            "Q115443051",
            "Q115553783",
            "... omitted 86 items"
          ]
        },
        {
          "property_id": "P4155",
          "values": [
            "P1476",
            "P1706",
            "P2868",
            "P407",
            "P518",
            "P958"
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
      "allowed qualifiers constraint: property: applies to jurisdiction, web feed URL, archive URL, employer, publisher, latest end date, maintained by, owned by, characteristic of, earliest date, latest date, domain name, described by source, MCW-PL article ID, operator, form factor, named after, published in, title, sourcing circumstances, end cause, has characteristic, subtitle, together with, subject named as, object named as, reason for deprecated rank, intended pub... [truncated 707 chars]",
      "required qualifier constraint: property: language of work or name; constraint status: suggestion constraint",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "single-best-value constraint: separator: applies to jurisdiction, publisher, owned by, operator, title, subject named as, intended public, subject has role, object of statement has role, language of work or name, contains, applies to part, start time, end time, point in time",
      "property scope constraint: property scope: as main value; constraint clarification: For qualifiers use URL (P2699), for references use reference URL (P854)@en"
    ],
    "before": [
      "format constraint: format as a regular expression: (?i)((?!\\b(://web\\.archive\\.org/)).)*; exception to constraint: Wayback Machine; syntax clarification: Die Adresse der Webseite sollte nicht den Ausdruck 'web.archive.org' enthalten.@de, a string not including 'web.archive.org'@en; constraint clarification: Add archive link with 'archive URL' (P1065) and qualify former official website with 'end time' (P582) instead.@en, Ändere bitte die Aussage zur ehemaligen off... [truncated 184 chars]",
      "format constraint: format as a regular expression: (https?|ftps?)://\\S+",
      "format constraint: format as a regular expression: ^((?!google\\.com/search\\?).)*$; exception to constraint: AI Mode; syntax clarification: Die Adresse der Webseite sollte nicht den Ausdruck 'google.com/search' enthalten.@de, a string not including 'google.com/search'@en",
      "format constraint: format as a regular expression: ^(?!https?://www\\.$).+; syntax clarification: Die Adresse enthält keinen Domainnamen@de, empty value not including a domain name@en",
      "distinct-values constraint: exception to constraint: RDA value vocabularies, RDA/ONIX framework value vocabularies, AIATSIS Subject Thesaurus, AIATSIS Place Thesaurus, Washington State Parks website, ABBAsite.com, Gauche démocratique et sociale, Macaulay Library, Navile's University Library. Section of Chemistry \"Giacomo Ciamician\", Navile's University Library. Section of the Department of Pharmacy and Biotechnology, Lumi, Delaware Department of State, Rowland Ric... [truncated 2383 chars]",
      "conflicts-with constraint: exception to constraint: list of natural events named by the Japan Meteorological Agency; item of property constraint: Wikimedia template, Wikimedia disambiguation page, Wikimedia category; property: instance of",
      "item-requires-statement constraint: property: instance of",
      "allowed qualifiers constraint: property: applies to jurisdiction, web feed URL, archive URL, employer, publisher, latest end date, maintained by, owned by, characteristic of, earliest date, latest date, domain name, described by source, MCW-PL article ID, operator, form factor, named after, published in, title, sourcing circumstances, end cause, has characteristic, subtitle, together with, subject named as, object named as, reason for deprecated rank, intended pub... [truncated 707 chars]",
      "required qualifier constraint: property: language of work or name; constraint status: suggestion constraint",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "single-best-value constraint: separator: applies to jurisdiction, publisher, owned by, operator, title, subject named as, intended public, subject has role, object of statement has role, language of work or name, contains, applies to part, start time, end time, point in time",
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
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|31"
  },
  {
    "result": "Q21502404",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 016. `reform_Q43455083_P8988_2445674626`

| Field | Value |
|---|---|
| qid | Q43455083 |
| property | P8988 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q21502404 |
| group_key | TBOX::P8988::2445674626 |
| tbox_revision_key | TBOX::P8988::2445674626 |

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
  "author": "JAn Dudík",
  "kind": "T_BOX",
  "property_revision_id": 2445674626,
  "property_revision_prev": 2437926724
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-23T06:56:35",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P8988",
  "report_revision_new": 2445904045,
  "report_revision_old": 2445349914,
  "report_violation_type": "Item P|625",
  "report_violation_type_normalized": "Item P|625",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|625",
  "value": null,
  "value_current_2026": [
    "stre&id=99036"
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
    "description": "identifier for a point of interest in Mapy.com",
    "label": "Mapy.com ID"
  },
  "qid": {
    "description": "street in Kostelec nad Černými lesy, Czech Republic",
    "label": "Pražská"
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
          "property_id": "P2303",
          "values": [
            "Q104830959",
            "Q11087457",
            "Q11343687",
            "Q34069677",
            "Q41150",
            "Q466099",
            "Q498700",
            "Q557873",
            "Q94999656"
          ]
        },
        {
          "property_id": "P2308",
          "values": [
            "Q108696",
            "Q11875349",
            "Q1190554",
            "Q131734",
            "Q15642541",
            "Q178561",
            "Q3152824",
            "Q350268",
            "Q618123",
            "Q811534",
            "Q820477",
            "Q860861"
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
  "after_constraint_count": 11,
  "author": "JAn Dudík",
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
          "P2303": [
            {
              "description_en": "street in Brno",
              "id": "Q113988434",
              "label_en": "park Mariacela"
            },
            {
              "description_en": "tunnel in Brno",
              "id": "Q12031732",
              "label_en": "Královopolský tunel"
            },
            {
              "description_en": "obora s muflony, daňky a divokými prasaty",
              "id": "Q12041961",
              "label_en": "Obora Holedná"
            },
            {
              "description_en": "abandoned village in the Czech Republic",
              "id": "Q1360645",
              "label_en": "Rolava (Přebuz)"
            },
            {
              "description_en": "fountain in Brno",
              "id": "Q30306295",
              "label_en": "Fountain in Lužánky"
            },
            {
              "description_en": "street in Brno",
              "id": "Q44687987",
              "label_en": "sady Národního odboje"
            },
            {
              "description_en": "park in Brno, Czech Republic",
              "id": "Q75289136",
              "label_en": "park Danuše Muzikářové"
            },
            {
              "description_en": "forest park and gorge in Brno-Lesná",
              "id": "Q84043600",
              "label_en": "Čertova rokle"
            },
            {
              "description_en": "zrušená železniční stanice v Česku",
              "id": "Q97665609",
              "label_en": "Horní Benešov"
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
              "value": "(addr|area|base|cada|coun|dist|muni|quar|regi|stre|ward|osm|pubt|firm|traf)&id=[1-9][0-9]{0,9}"
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
          "P2305": [
            {
              "description_en": "page of a Wikimedia project with a list of something",
              "id": "Q13406463",
              "label_en": "Wikimedia list article"
            },
            {
              "description_en": "this item duplicates another item, it can be merged once the necessary merges are done in other Wikimedia projects",
              "id": "Q17362920",
              "label_en": "Wikimedia duplicated page"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
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
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
              "description_en": "fire in Brno",
              "id": "Q104830959",
              "label_en": "fire of a tree stump in Kohoutovice"
            },
            {
              "description_en": "a giant barrel in Mikulov, Czechia",
              "id": "Q11087457",
              "label_en": "Giant barrel in Mikulov"
            },
            {
              "description_en": "vycpané zvíře",
              "id": "Q11343687",
              "label_en": "Brno Dragon"
            },
            {
              "description_en": "graffiti v Brně",
              "id": "Q34069677",
              "label_en": "Pocta herci Jiřímu Pechovi a dalším jeho kolegům"
            },
            {
              "description_en": "Catholic pilgrimage to Santiago de Compostela, Spain",
              "id": "Q41150",
              "label_en": "Way of Saint James"
            },
            {
              "description_en": "annual cross-country skiing competition near Liberec, Czechia",
              "id": "Q466099",
              "label_en": "Jizerská padesátka"
            },
            {
              "description_en": "ethnic group of Central Europe",
              "id": "Q498700",
              "label_en": "Gorals"
            },
            {
              "description_en": "1945 offensive of the Soviet Army during World War II",
              "id": "Q557873",
              "label_en": "Moravia–Ostrava offensive"
            },
            {
              "description_en": "první dopravní nehoda se smrtelným zraněním ve střední Evropě",
              "id": "Q94999656",
              "label_en": "dopravní nehoda v Rychalticích"
            }
          ],
          "P2308": [
            {
              "description_en": "stud farm specialized in horse breeding",
              "id": "Q108696",
              "label_en": "horse stud farm"
            },
            {
              "description_en": "outdoor play space for children",
              "id": "Q11875349",
              "label_en": "playground"
            },
            {
              "description_en": "occurrence of a fact or object in space-time; instantiation of a property in an object",
              "id": "Q1190554",
              "label_en": "occurrence"
            },
            {
              "description_en": "business that makes and sells beer",
              "id": "Q131734",
              "label_en": "brewery"
            },
            {
              "description_en": "territorial entity of which the borders are determined by physiographic and human features",
              "id": "Q15642541",
              "label_en": "human-geographic territorial entity"
            },
            {
              "description_en": "part of a war which is well defined in duration, area and force commitment",
              "id": "Q178561",
              "label_en": "battle"
            },
            {
              "description_en": "organization that works for the preservation or promotion of culture",
              "id": "Q3152824",
              "label_en": "cultural institution"
            },
            {
              "description_en": "type of artwork",
              "id": "Q350268",
              "label_en": "plastic artwork"
            },
            {
              "description_en": "components of planets that can be geographically located",
              "id": "Q618123",
              "label_en": "geographical feature"
            },
            {
              "description_en": "tree which, because of its great age, size or condition, or historical connection, is of exceptional cultural, landscape or nature conservation value",
              "id": "Q811534",
              "label_en": "remarkable tree"
            },
            {
              "description_en": "place for the extraction of minerals",
              "id": "Q820477",
              "label_en": "mine"
            },
            {
              "description_en": "three-dimensional work of art",
              "id": "Q860861",
              "label_en": "sculpture"
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
              "description_en": "street in Brno",
              "id": "Q113988434",
              "label_en": "park Mariacela"
            },
            {
              "description_en": "tunnel in Brno",
              "id": "Q12031732",
              "label_en": "Královopolský tunel"
            },
            {
              "description_en": "obora s muflony, daňky a divokými prasaty",
              "id": "Q12041961",
              "label_en": "Obora Holedná"
            },
            {
              "description_en": "abandoned village in the Czech Republic",
              "id": "Q1360645",
              "label_en": "Rolava (Přebuz)"
            },
            {
              "description_en": "fountain in Brno",
              "id": "Q30306295",
              "label_en": "Fountain in Lužánky"
            },
            {
              "description_en": "street in Brno",
              "id": "Q44687987",
              "label_en": "sady Národního odboje"
            },
            {
              "description_en": "park in Brno, Czech Republic",
              "id": "Q75289136",
              "label_en": "park Danuše Muzikářové"
            },
            {
              "description_en": "forest park and gorge in Brno-Lesná",
              "id": "Q84043600",
              "label_en": "Čertova rokle"
            },
            {
              "description_en": "zrušená železniční stanice v Česku",
              "id": "Q97665609",
              "label_en": "Horní Benešov"
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
              "value": "(addr|area|base|cada|coun|dist|muni|quar|regi|stre|ward|osm|pubt|firm|traf)&id=[1-9][0-9]{0,9}"
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
          "P2305": [
            {
              "description_en": "page of a Wikimedia project with a list of something",
              "id": "Q13406463",
              "label_en": "Wikimedia list article"
            },
            {
              "description_en": "this item duplicates another item, it can be merged once the necessary merges are done in other Wikimedia projects",
              "id": "Q17362920",
              "label_en": "Wikimedia duplicated page"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
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
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "id": "Q21503247",
          "label_en": "item-requires-statement constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
              "description_en": "fire in Brno",
              "id": "Q104830959",
              "label_en": "fire of a tree stump in Kohoutovice"
            },
            {
              "description_en": "a giant barrel in Mikulov, Czechia",
              "id": "Q11087457",
              "label_en": "Giant barrel in Mikulov"
            },
            {
              "description_en": "vycpané zvíře",
              "id": "Q11343687",
              "label_en": "Brno Dragon"
            },
            {
              "description_en": "graffiti v Brně",
              "id": "Q34069677",
              "label_en": "Pocta herci Jiřímu Pechovi a dalším jeho kolegům"
            },
            {
              "description_en": "Catholic pilgrimage to Santiago de Compostela, Spain",
              "id": "Q41150",
              "label_en": "Way of Saint James"
            },
            {
              "description_en": "annual cross-country skiing competition near Liberec, Czechia",
              "id": "Q466099",
              "label_en": "Jizerská padesátka"
            },
            {
              "description_en": "ethnic group of Central Europe",
              "id": "Q498700",
              "label_en": "Gorals"
            },
            {
              "description_en": "1945 offensive of the Soviet Army during World War II",
              "id": "Q557873",
              "label_en": "Moravia–Ostrava offensive"
            },
            {
              "description_en": "první dopravní nehoda se smrtelným zraněním ve střední Evropě",
              "id": "Q94999656",
              "label_en": "dopravní nehoda v Rychalticích"
            }
          ],
          "P2308": [
            {
              "description_en": "stud farm specialized in horse breeding",
              "id": "Q108696",
              "label_en": "horse stud farm"
            },
            {
              "description_en": "outdoor play space for children",
              "id": "Q11875349",
              "label_en": "playground"
            },
            {
              "description_en": "occurrence of a fact or object in space-time; instantiation of a property in an object",
              "id": "Q1190554",
              "label_en": "occurrence"
            },
            {
              "description_en": "business that makes and sells beer",
              "id": "Q131734",
              "label_en": "brewery"
            },
            {
              "description_en": "territorial entity of which the borders are determined by physiographic and human features",
              "id": "Q15642541",
              "label_en": "human-geographic territorial entity"
            },
            {
              "description_en": "part of a war which is well defined in duration, area and force commitment",
              "id": "Q178561",
              "label_en": "battle"
            },
            {
              "description_en": "organization that works for the preservation or promotion of culture",
              "id": "Q3152824",
              "label_en": "cultural institution"
            },
            {
              "description_en": "components of planets that can be geographically located",
              "id": "Q618123",
              "label_en": "geographical feature"
            },
            {
              "description_en": "tree which, because of its great age, size or condition, or historical connection, is of exceptional cultural, landscape or nature conservation value",
              "id": "Q811534",
              "label_en": "remarkable tree"
            },
            {
              "description_en": "place for the extraction of minerals",
              "id": "Q820477",
              "label_en": "mine"
            },
            {
              "description_en": "three-dimensional work of art",
              "id": "Q860861",
              "label_en": "sculpture"
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
  "hash_after": "698c87a03258d2c8497f4f2fe89ac50608e68d09",
  "hash_before": "24d513db5ec3bd62001f537388c44a29b482fdd7",
  "property_revision_id": 2445674626,
  "property_revision_prev": 2437926724,
  "qualifier_value_changes": [
    {
      "added_values": [
        "Q350268"
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
          "property_id": "P2303",
          "values": [
            "Q104830959",
            "Q11087457",
            "Q11343687",
            "Q34069677",
            "Q41150",
            "Q466099",
            "Q498700",
            "Q557873",
            "Q94999656"
          ]
        },
        {
          "property_id": "P2308",
          "values": [
            "Q108696",
            "Q11875349",
            "Q1190554",
            "Q131734",
            "Q15642541",
            "Q178561",
            "Q3152824",
            "Q618123",
            "Q811534",
            "Q820477",
            "Q860861"
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
      "single-value constraint: exception to constraint: park Mariacela, Královopolský tunel, Obora Holedná, Rolava (Přebuz), Fountain in Lužánky, sady Národního odboje, park Danuše Muzikářové, Čertova rokle, Horní Benešov",
      "format constraint: format as a regular expression: (addr|area|base|cada|coun|dist|muni|quar|regi|stre|ward|osm|pubt|firm|traf)&id=[1-9][0-9]{0,9}",
      "distinct-values constraint: no qualifiers recorded",
      "conflicts-with constraint: item of property constraint: Wikimedia list article, Wikimedia duplicated page, Wikimedia disambiguation page, Wikimedia category; property: instance of; constraint status: mandatory constraint",
      "item-requires-statement constraint: property: located in the administrative territorial entity",
      "item-requires-statement constraint: property: country",
      "item-requires-statement constraint: property: instance of",
      "item-requires-statement constraint: property: coordinate location",
      "subject type constraint: exception to constraint: fire of a tree stump in Kohoutovice, Giant barrel in Mikulov, Brno Dragon, Pocta herci Jiřímu Pechovi a dalším jeho kolegům, Way of Saint James, Jizerská padesátka, Gorals, Moravia–Ostrava offensive, dopravní nehoda v Rychalticích; class: horse stud farm, playground, occurrence, brewery, human-geographic territorial entity, battle, cultural institution, plastic artwork, geographical feature, remarkable tree, mine, sculpture; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ],
    "before": [
      "single-value constraint: exception to constraint: park Mariacela, Královopolský tunel, Obora Holedná, Rolava (Přebuz), Fountain in Lužánky, sady Národního odboje, park Danuše Muzikářové, Čertova rokle, Horní Benešov",
      "format constraint: format as a regular expression: (addr|area|base|cada|coun|dist|muni|quar|regi|stre|ward|osm|pubt|firm|traf)&id=[1-9][0-9]{0,9}",
      "distinct-values constraint: no qualifiers recorded",
      "conflicts-with constraint: item of property constraint: Wikimedia list article, Wikimedia duplicated page, Wikimedia disambiguation page, Wikimedia category; property: instance of; constraint status: mandatory constraint",
      "item-requires-statement constraint: property: located in the administrative territorial entity",
      "item-requires-statement constraint: property: country",
      "item-requires-statement constraint: property: instance of",
      "item-requires-statement constraint: property: coordinate location",
      "subject type constraint: exception to constraint: fire of a tree stump in Kohoutovice, Giant barrel in Mikulov, Brno Dragon, Pocta herci Jiřímu Pechovi a dalším jeho kolegům, Way of Saint James, Jizerská padesátka, Gorals, Moravia–Ostrava offensive, dopravní nehoda v Rychalticích; class: horse stud farm, playground, occurrence, brewery, human-geographic territorial entity, battle, cultural institution, geographical feature, remarkable tree, mine, sculpture; relation: instance of",
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
    "violation_name": "Item P|625"
  },
  {
    "result": "Q19474404",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 017. `reform_Q7793203_P21_2439154480`

| Field | Value |
|---|---|
| qid | Q7793203 |
| property | P21 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q52004125 |
| group_key | TBOX::P21::2439154480 |
| tbox_revision_key | TBOX::P21::2439154480 |

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
  "property_revision_id": 2439154480,
  "property_revision_prev": 2439154461
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-09T16:46:38",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P21",
  "report_revision_new": 2440121961,
  "report_revision_old": 2439614652,
  "report_violation_type": "Conflicts with P|625",
  "report_violation_type_normalized": "Conflicts with P|625",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Conflicts with P|625",
  "value": null,
  "value_current_2026": [
    "Q6581097"
  ],
  "value_current_2026_descriptions_en": [
    "to be used in \"sex or gender\" (P21) to indicate that the human subject is a male or \"semantic gender\" (P10339) to indicate that a word refers to a male person"
  ],
  "value_current_2026_labels_en": [
    "male"
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
    "description": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
    "label": "sex or gender"
  },
  "qid": {
    "description": "Revolutionary War officer and politician",
    "label": "Thomas Polk"
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
    "label_en": "one-of constraint",
    "qid": "Q21510859"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
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
      "constraint_qid": "Q52558054",
      "qualifiers": [
        {
          "property_id": "P2305",
          "values": [
            "Q124726070"
          ]
        },
        {
          "property_id": "P9729",
          "values": [
            "Q189125"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 38,
  "author": "Trade",
  "before_constraint_count": 38,
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
          "P2241": [
            {
              "description_en": "reason for deprecation of a Wikidata property constraint",
              "id": "Q99460987",
              "label_en": "constraint provides suggestions for manual input"
            }
          ],
          "P2303": [
            {
              "description_en": "fictional government body and biomechatronic computer network from the Japanese anime television show Psycho-Pass",
              "id": "Q110541439",
              "label_en": "Sibyl System"
            },
            {
              "description_en": "either of two Canadian-made hitchhiking robots",
              "id": "Q17450611",
              "label_en": "hitchBOT"
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
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ]
        },
        "rank": "deprecated",
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
              "description_en": "rimu tree (Dacrydium cupressinum) in Ōtari-Wilton's Bush, Wellington City, New Zealand; Wellington's oldest and tallest tree",
              "id": "Q107394029",
              "label_en": "Moko"
            }
          ],
          "P2306": [
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
              "description_en": "fictional character appearing in a video game",
              "id": "Q1569167",
              "label_en": "video game character"
            },
            {
              "description_en": "unknown creator of a work (do not use as value of P50; use \"unknown value\" instead)",
              "id": "Q4233718",
              "label_en": "anonymous"
            },
            {
              "description_en": "fictional human or non-human character in a narrative work of art",
              "id": "Q95074",
              "label_en": "character"
            }
          ],
          "P2306": [
            {
              "description_en": "this item is a subclass (subset) of that item; ALL instances of this item are instances of that item; different from P31 (instance of), e.g.: volcano is a subclass of mountain; Everest is an instance of mountain",
              "id": "P279",
              "label_en": "subclass of"
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
          "P2303": [
            {
              "description_en": "well-preserved natural prehistoric mummy",
              "id": "Q171291",
              "label_en": "Ötzi"
            },
            {
              "description_en": "either of two Canadian-made hitchhiking robots",
              "id": "Q17450611",
              "label_en": "hitchBOT"
            },
            {
              "description_en": "operatic character in the opera The Nose by Dmitri Shostakovich; a Collegiate Assessor",
              "id": "Q55039999",
              "label_en": "Platon Kuzmich Koavalyov"
            },
            {
              "description_en": "operatic character in the opera Mauerschau",
              "id": "Q55052424",
              "label_en": "Penthesilea"
            },
            {
              "description_en": "fictional two-tailed fox familiar in Pepper&Carrot",
              "id": "Q75840084",
              "label_en": "Yuzu"
            }
          ],
          "P2306": [
            {
              "description_en": "part of this subject; inverse property of \"part of\" (P361). See also \"has parts of the class\" (P2670).",
              "id": "P527",
              "label_en": "has part(s)"
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
              "description_en": "part of a naming scheme for individuals, used in many cultures worldwide",
              "id": "Q101352",
              "label_en": "family name"
            },
            {
              "description_en": "unlawful killing of a human with malice aforethought",
              "id": "Q132821",
              "label_en": "murder"
            },
            {
              "description_en": "page of a Wikimedia project with a list of something",
              "id": "Q13406463",
              "label_en": "Wikimedia list article"
            },
            {
              "description_en": "list of films related by some criteria",
              "id": "Q1371849",
              "label_en": "filmography"
            },
            {
              "description_en": "study and cataloging of published sound recordings",
              "id": "Q273057",
              "label_en": "discography"
            },
            {
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
            },
            {
              "description_en": "permanent cessation of vital functions",
              "id": "Q4",
              "label_en": "death"
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
              "description_en": "narrative text, normally of a substantial length and in the form of prose describing a fictional and sequential story",
              "id": "Q8261",
              "label_en": "novel"
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
              "description_en": "two individuals who work together",
              "id": "Q10648343",
              "label_en": "duo"
            },
            {
              "description_en": "two comedians who perform together as a single act",
              "id": "Q1141470",
              "label_en": "double act"
            },
            {
              "description_en": "consists monarchs who are related to one another, as well as their non-reigning descendants and spouses",
              "id": "Q1156073",
              "label_en": "royal house"
            },
            {
              "description_en": "family part of the nobility of a region or country",
              "id": "Q13417114",
              "label_en": "noble family"
            },
            {
              "description_en": "two siblings that work together",
              "id": "Q14073567",
              "label_en": "sibling duo"
            },
            {
              "description_en": "two offspring produced in the same pregnancy (use with P31 on items for both twins - if known, item for \"identical twins\" or \"fraternal twins\" is preferred)",
              "id": "Q14756018",
              "label_en": "twins"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "sequence of rulers considered members of the same family",
              "id": "Q164950",
              "label_en": "dynasty"
            },
            {
              "description_en": "group of humans, they are siblings; e.g. Grimm brothers, Brontë sisters",
              "id": "Q16979650",
              "label_en": "sibling group"
            },
            {
              "description_en": "twins with the same genes, that grew from one egg that split into two",
              "id": "Q2301325",
              "label_en": "identical twins"
            },
            {
              "description_en": "group of three musicians",
              "id": "Q281643",
              "label_en": "musical trio"
            },
            {
              "description_en": "two people who are married to each other",
              "id": "Q3046146",
              "label_en": "married couple"
            },
            {
              "description_en": "one of two or more individuals having at least one parent in common",
              "id": "Q31184",
              "label_en": "sibling"
            },
            {
              "description_en": "twins formed by two separate eggs fertilised by two separate sperms",
              "id": "Q3418125",
              "label_en": "fraternal twins"
            },
            {
              "description_en": "group of people affiliated by consanguinity, law, affinity, or co-residence",
              "id": "Q8436",
              "label_en": "family"
            },
            {
              "description_en": "two or more humans who interact with one another",
              "id": "Q874405",
              "label_en": "social group"
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
              "description_en": "company that publishes video games",
              "id": "Q1137109",
              "label_en": "video game publisher"
            },
            {
              "description_en": "software development organization specializing in the creation of video games (for person use Q58287519)",
              "id": "Q210167",
              "label_en": "video game developer"
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
              "description_en": "list article about a set of items of a specific type that share the same (or similar) name",
              "id": "Q15623926",
              "label_en": "Wikimedia set index article"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "name typically used to differentiate people from the same family, clan, or other social group who have a common last name",
              "id": "Q202444",
              "label_en": "given name"
            },
            {
              "description_en": "group of people who perform instrumental and/or vocal music, with the ensemble typically known by a distinct name",
              "id": "Q2088357",
              "label_en": "musical ensemble"
            },
            {
              "description_en": "this item duplicates another item, the two can't be merged, as one Wikimedia project includes 2 pages, e.g. in different scripts or languages. Add properties other than P2959 (permanent duplicated item) and sitelinks for other wikis to the other item",
              "id": "Q21286738",
              "label_en": "Wikimedia permanent duplicate item"
            },
            {
              "description_en": "musical ensemble which performs music",
              "id": "Q215380",
              "label_en": "musical group"
            },
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            },
            {
              "description_en": "medium for recording information (words or images) typically on bound pages or more abstractly in electronic or audio form",
              "id": "Q571",
              "label_en": "book"
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
          "P2306": [
            {
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
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
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
              "description_en": "common element between all listed items",
              "id": "P360",
              "label_en": "is a list of"
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
              "description_en": "language in which a film or a performance work was originally created. Deprecated for written works and songs; use P407 (\"language of work or name\") instead.",
              "id": "P364",
              "label_en": "original language of film or TV show"
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
              "description_en": "main creator(s) of a written work (use on works, not humans); use P2093 (author name string) when Wikidata item is unknown or does not exist",
              "id": "P50",
              "label_en": "author"
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
              "description_en": "anthropomorphic sexual device",
              "id": "Q10832",
              "label_en": "sex doll"
            },
            {
              "description_en": "mechanical or virtual artificial agent carrying out physical activities, which can be guided by an external control device or the control may be embedded within",
              "id": "Q11012",
              "label_en": "robot"
            },
            {
              "description_en": "type of a doll or (action) figure that may appear in different variants",
              "id": "Q111282474",
              "label_en": "doll or action figure model"
            },
            {
              "description_en": "character known only from narrations (fictional or in a factual manner) without a proof of existence; includes fictional, mythical, legendary or religious characters and similar",
              "id": "Q115537581",
              "label_en": "imaginary character"
            },
            {
              "description_en": "teknonym in an Arabic name, the name of an adult derived from their eldest son",
              "id": "Q1285470",
              "label_en": "kunya"
            },
            {
              "description_en": "entity that has no physical realisation",
              "id": "Q15619164",
              "label_en": "abstract being"
            },
            {
              "description_en": "group of one or more fictional organism(s), which a (fictional) taxonomist adjudges to be a unit",
              "id": "Q15707583",
              "label_en": "fictional taxon"
            },
            {
              "description_en": "group of one or more organism(s), which a taxonomist adjudges to be a unit",
              "id": "Q16521",
              "label_en": "taxon"
            },
            {
              "description_en": "model of a character or a human being, often used as a toy for children or an artistic hobby for adults",
              "id": "Q168658",
              "label_en": "doll"
            },
            {
              "description_en": "Latin phrase; alternate self",
              "id": "Q201662",
              "label_en": "alter ego"
            },
            {
              "description_en": "individual who died before or during birth",
              "id": "Q2345820",
              "label_en": "stillborn child"
            },
            {
              "description_en": "organism not more specified in a work of fiction",
              "id": "Q2593744",
              "label_en": "fictional creature"
            },
            {
              "description_en": "singular named exemplar of an animal (e.g., the gorilla named Koko; the cat named Socks)",
              "id": "Q26401003",
              "label_en": "individual animal"
            },
            {
              "description_en": "prenatal organism between the embryonic state and birth",
              "id": "Q26513",
              "label_en": "human fetus"
            },
            {
              "description_en": "preserved remains or traces of organisms from a past geological age",
              "id": "Q40614",
              "label_en": "fossil"
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
            },
            {
              "description_en": "any individual living being or physical living system",
              "id": "Q7239",
              "label_en": "organism"
            },
            {
              "description_en": "kingdom of multicellular eukaryotic organisms",
              "id": "Q729",
              "label_en": "Animalia"
            },
            {
              "description_en": "a human-sounding voice generated by a computer",
              "id": "Q79600797",
              "label_en": "synthetic voice"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "work or narration for or in which this statement is true",
              "id": "P10663",
              "label_en": "applies to work"
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
              "description_en": "qualifier property to be used with statements having the object \"no value\", given to provide a reason for \"no value\"",
              "id": "P13589",
              "label_en": "‎reason for no value"
            },
            {
              "description_en": "qualification of the truth or accuracy of a source: circa (Q5727902), near (Q21818619), presumably (Q18122778), etc.",
              "id": "P1480",
              "label_en": "sourcing circumstances"
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
              "description_en": "entity that supports a given statement",
              "id": "P3680",
              "label_en": "statement supported by"
            },
            {
              "description_en": "how a value is determined, or the standard by which it is declared",
              "id": "P459",
              "label_en": "determination method or standard"
            },
            {
              "description_en": "(qualifier only) the underlying circumstances of this statement",
              "id": "P5102",
              "label_en": "nature of statement"
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
              "description_en": "time of the first public presentation of a subject by the creator, of information by the media",
              "id": "P6949",
              "label_en": "announcement date"
            },
            {
              "description_en": "qualifier to allow the reason to be indicated why a particular statement should be considered preferred",
              "id": "P7452",
              "label_en": "reason for preferred rank"
            },
            {
              "description_en": "reading this information or the content of the identifier might ruin the experience of receiving this work",
              "id": "P7528",
              "label_en": "statement or content of identifier is regarded as spoiler for"
            },
            {
              "description_en": "(qualifier) earliest date on which the statement could have begun to no longer be true",
              "id": "P8554",
              "label_en": "earliest end date"
            },
            {
              "description_en": "(qualifier) latest date on which the statement could have started to be true",
              "id": "P8555",
              "label_en": "latest start date"
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
          "P2305": [
            {
              "description_en": "woman assigned male at birth",
              "id": "Q1052281",
              "label_en": "trans woman"
            },
            {
              "description_en": "third gender in Niue culture",
              "id": "Q107427210",
              "label_en": "fakafifine"
            },
            {
              "description_en": "transgender identity of individuals who identify on the neutral side of the gender spectrum",
              "id": "Q107502361",
              "label_en": "transneutral"
            },
            {
              "description_en": "atypical congenital variations of sex characteristics",
              "id": "Q1097630",
              "label_en": "intersex"
            },
            {
              "description_en": "transgender women in Wallisian and Futunan culture",
              "id": "Q112597587",
              "label_en": "fakafafine"
            },
            {
              "description_en": "person born with any of several variations in sex characteristics including chromosomes, gonads, sex hormones or genitals that do not fit the typical definitions for male or female bodies",
              "id": "Q11287467",
              "label_en": "intersex person"
            },
            {
              "description_en": "datum representing when an individual has not publicly disclosed their gender identity or has expressed a wish not to disclose their gender identity, thereby making their gender identity unknown",
              "id": "Q113124952",
              "label_en": "undisclosed gender"
            },
            {
              "description_en": "case in which a fictional character's gender is not outright disclosed in a work and cannot be reasonably assumed",
              "id": "Q116254116",
              "label_en": "gender not disclosed in work"
            },
            {
              "description_en": "gender listing within the Pokémon franchise",
              "id": "Q116741172",
              "label_en": "gender unknown"
            },
            {
              "description_en": "intersex person who identifies as a man",
              "id": "Q121307094",
              "label_en": "intersex man"
            },
            {
              "description_en": "intersex person who identifies as a woman",
              "id": "Q121307100",
              "label_en": "intersex woman"
            },
            {
              "description_en": "gender identity where a person identifies partially as a man/boy or as otherwise being partly masculine in nature",
              "id": "Q121368243",
              "label_en": "demimasc"
            },
            {
              "description_en": "Japanese term referring to female characters with male genitalia either at birth or grownth through supernatural, magical or technological means",
              "id": "Q123479538",
              "label_en": "futanari"
            },
            {
              "description_en": "a term to describe an individual who considers the phenomenon of gender identity to be unknowable or indeterminable by its very nature",
              "id": "Q124637723",
              "label_en": "gender agnostic"
            },
            {
              "description_en": "non-binary gender identity",
              "id": "Q1289754",
              "label_en": "neutrois"
            },
            {
              "description_en": "range of gender identities that are not exclusively masculine or feminine and does not match with assigned sex",
              "id": "Q12964198",
              "label_en": "genderqueer"
            },
            {
              "description_en": "indigenous Australian transfeminine gender",
              "id": "Q130315001",
              "label_en": "sistergirl"
            },
            {
              "description_en": "indigenous australian transmasculine gender identity",
              "id": "Q130315012",
              "label_en": "brotherboy"
            },
            {
              "description_en": "individual who identifies with both female and non-binary gender identities",
              "id": "Q130477254",
              "label_en": "non-binary woman"
            },
            {
              "description_en": "individual who identifies with both male and non-binary gender identities",
              "id": "Q130477279",
              "label_en": "non-binary man"
            },
            {
              "description_en": "having both sexes, including naturally or intentionally",
              "id": "Q130899399",
              "label_en": "bisex"
            },
            {
              "description_en": "זהות מגדרית; דברים שמתאימים למין אחד",
              "id": "Q130964491",
              "label_en": "monogender"
            },
            {
              "description_en": "In Samoan culture, Samoan people who were assigned male at birth but behave in feminine ways and may consider themselves women",
              "id": "Q1399232",
              "label_en": "faʻafafine"
            },
            {
              "description_en": "man who was assigned male at birth and identifies as male",
              "id": "Q15145778",
              "label_en": "cisgender man"
            },
            "... omitted 47 items"
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
          "P2305": [
            {
              "description_en": "romantic and/or sexual attraction or behavior between people of different genders",
              "id": "Q1035954",
              "label_en": "heterosexuality"
            },
            {
              "description_en": "umbrella term for sexual orientations and identities where someone experiences attraction to more than one gender",
              "id": "Q106589379",
              "label_en": "plurisexuality"
            },
            {
              "description_en": "attraction only to those they are mentally connected to the person",
              "id": "Q108472356",
              "label_en": "noetisexuality"
            },
            {
              "description_en": "umbrella term for men loving men",
              "id": "Q115068942",
              "label_en": "Achillean"
            },
            {
              "description_en": "umbrella term for attractions and relationships involving at least one non-binary person",
              "id": "Q116280707",
              "label_en": "diamoric"
            },
            {
              "description_en": "non-binary people attracted to women",
              "id": "Q116286410",
              "label_en": "trixic"
            },
            {
              "description_en": "non-binary people attracted to men",
              "id": "Q116295197",
              "label_en": "toric"
            },
            {
              "description_en": "attraction to three genders or experiencing three sexualities",
              "id": "Q116820152",
              "label_en": "trisexuality"
            },
            {
              "description_en": "case in which the sexual orientation of a character in a video game is determined by the player",
              "id": "Q123138223",
              "label_en": "sexuality determined by the player"
            },
            {
              "description_en": "sexual, romantic, queer- & platonic, sensual and another attraction to women or femininity",
              "id": "Q1558475",
              "label_en": "gynephilia"
            },
            {
              "description_en": "changes in sexuality or sexual identity",
              "id": "Q19810527",
              "label_en": "sexual fluidity"
            },
            {
              "description_en": "sexual attraction based primarily on intellect",
              "id": "Q20011275",
              "label_en": "sapiosexuality"
            },
            {
              "description_en": "sexual attraction to multiple, but not all, genders",
              "id": "Q2094204",
              "label_en": "polysexuality"
            },
            {
              "description_en": "romantic attraction, sexual attraction, or sexual behavior between two men",
              "id": "Q2257941",
              "label_en": "male homosexuality"
            },
            {
              "description_en": "sexual attraction only to people with whom emotional bonds are formed",
              "id": "Q23912283",
              "label_en": "demisexuality"
            },
            {
              "description_en": "term used by individuals who do not wish to label their sexuality with more specific terms",
              "id": "Q25326668",
              "label_en": "unlabeled sexuality"
            },
            {
              "description_en": "umbrella term for feminine person loving feminine person",
              "id": "Q25447263",
              "label_en": "sapphism"
            },
            {
              "description_en": "person who is attracted to experiencing bisexuality",
              "id": "Q255155",
              "label_en": "bi-curious"
            },
            {
              "description_en": "sexual or romantic attraction to people regardless of gender",
              "id": "Q271534",
              "label_en": "pansexuality"
            },
            {
              "description_en": "sexual orientation other than heterosexual or straight",
              "id": "Q339014",
              "label_en": "non-heterosexuality"
            },
            {
              "description_en": "sexual orientation of people who reject, avoid or do not fit into any sexual orientation label",
              "id": "Q3626860",
              "label_en": "pomosexuality"
            },
            {
              "description_en": "sexual and/or romantic attraction to people of more than one gender",
              "id": "Q43200",
              "label_en": "bisexuality"
            },
            {
              "description_en": "sexual attraction to males, men or masculinity",
              "id": "Q43850015",
              "label_en": "androsexuality"
            },
            {
              "description_en": "sexual attraction to women, femininity or females",
              "id": "Q43850027",
              "label_en": "gynesexuality"
            },
            "... omitted 9 items"
          ],
          "P6824": [
            {
              "description_en": "the sexual orientation of the person relative to their declared gender — use ONLY IF they have stated it themselves, unambiguously, or it has been widely agreed upon by historians after their death",
              "id": "P91",
              "label_en": "sexual orientation"
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
              "description_en": "man who is sexually and/or romantically attracted to more than one gender",
              "id": "Q105079567",
              "label_en": "bisexual man"
            },
            {
              "description_en": "woman who is sexually and/or romantically attracted to more than one gender",
              "id": "Q105079916",
              "label_en": "bisexual woman"
            },
            {
              "description_en": "someone with a non-binary gender identity attracted to more than one gender",
              "id": "Q115870499",
              "label_en": "non-binary bisexual"
            },
            {
              "description_en": "person who is sexually and/or romantically attracted to persons with a gender or genders like theirs and those with a gender or genders unlike theirs",
              "id": "Q12905217",
              "label_en": "bisexual person"
            }
          ],
          "P6824": [
            {
              "description_en": "the sexual orientation of the person relative to their declared gender — use ONLY IF they have stated it themselves, unambiguously, or it has been widely agreed upon by historians after their death",
              "id": "P91",
              "label_en": "sexual orientation"
            }
          ],
          "P9729": [
            {
              "description_en": "sexual and/or romantic attraction to people of more than one gender",
              "id": "Q43200",
              "label_en": "bisexuality"
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
              "description_en": "man who is sexually and/or romantically attracted to more than one gender",
              "id": "Q105079567",
              "label_en": "bisexual man"
            },
            {
              "description_en": "male human that identifies as their sex assigned at birth and is sexually attracted to females",
              "id": "Q134897761",
              "label_en": "cisgender heterosexual man"
            },
            {
              "description_en": "young male human",
              "id": "Q3010",
              "label_en": "boy"
            },
            {
              "description_en": "male adult human",
              "id": "Q8441",
              "label_en": "man"
            },
            {
              "description_en": "demographic classification",
              "id": "Q853451",
              "label_en": "men who have sex with men"
            },
            {
              "description_en": "boy between birth-23 months",
              "id": "Q96780034",
              "label_en": "baby boy"
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
              "description_en": "woman who is sexually and/or romantically attracted to more than one gender",
              "id": "Q105079916",
              "label_en": "bisexual woman"
            },
            {
              "description_en": "female human that identifies as their sex assigned at birth and is sexually attracted to males",
              "id": "Q134897481",
              "label_en": "cisgender heterosexual woman"
            },
            {
              "description_en": "sexual identity-neutral term",
              "id": "Q210604",
              "label_en": "women who have sex with women"
            },
            {
              "description_en": "young female human",
              "id": "Q3031",
              "label_en": "girl"
            },
            {
              "description_en": "female adult human",
              "id": "Q467",
              "label_en": "woman"
            },
            {
              "description_en": "very young female human",
              "id": "Q97009651",
              "label_en": "baby girl"
            }
          ],
          "P9729": [
            {
              "description_en": "to be used in \"sex or gender\" (P21) to indicate that the human subject is a female or \"semantic gender\" (P10339) to indicate that a word refers to a female person",
              "id": "Q6581072",
              "label_en": "female"
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
              "description_en": "someone who is transgender",
              "id": "Q10701290",
              "label_en": "transgender person"
            },
            {
              "description_en": "a slang for a transgender person who is heterosexual",
              "id": "Q124726070",
              "label_en": "transhet"
            }
          ],
          "P9729": [
            {
              "description_en": "gender identity different to the gender assigned at birth",
              "id": "Q189125",
              "label_en": "transgender"
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
              "description_en": "male human that identifies as and is sexually attracted to their sex assigned at birth",
              "id": "Q107785560",
              "label_en": "cisgender gay male"
            },
            {
              "description_en": "female human that identifies as and is sexually attracted to their sex assigned at birth",
              "id": "Q124637919",
              "label_en": "cisgender lesbian"
            },
            {
              "description_en": "female human that identifies as their sex assigned at birth and is sexually attracted to males",
              "id": "Q134897481",
              "label_en": "cisgender heterosexual woman"
            },
            {
              "description_en": "male human that identifies as their sex assigned at birth and is sexually attracted to females",
              "id": "Q134897761",
              "label_en": "cisgender heterosexual man"
            }
          ],
          "P9729": [
            {
              "description_en": "correspondence between a person's gender identity and the sex assigned to them at birth",
              "id": "Q1093205",
              "label_en": "cisgender"
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
              "description_en": "Chinese given name (雄)",
              "id": "Q108598090",
              "label_en": "Xiong"
            },
            {
              "description_en": "CJK (hanzi/kanji/hanja) character",
              "id": "Q55814929",
              "label_en": "雄"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ],
          "P9729": [
            {
              "description_en": "organism of the male sex",
              "id": "Q44148",
              "label_en": "male organism"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      "... omitted 14 items"
    ],
    "before": [
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "id": "Q21502838",
          "label_en": "conflicts-with constraint"
        },
        "parameters": {
          "P2241": [
            {
              "description_en": "reason for deprecation of a Wikidata property constraint",
              "id": "Q99460987",
              "label_en": "constraint provides suggestions for manual input"
            }
          ],
          "P2303": [
            {
              "description_en": "fictional government body and biomechatronic computer network from the Japanese anime television show Psycho-Pass",
              "id": "Q110541439",
              "label_en": "Sibyl System"
            },
            {
              "description_en": "either of two Canadian-made hitchhiking robots",
              "id": "Q17450611",
              "label_en": "hitchBOT"
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
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint merely suggests additional improvements, and violations are not as severe as for regular or mandatory constraints",
              "id": "Q62026391",
              "label_en": "suggestion constraint"
            }
          ]
        },
        "rank": "deprecated",
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
              "description_en": "rimu tree (Dacrydium cupressinum) in Ōtari-Wilton's Bush, Wellington City, New Zealand; Wellington's oldest and tallest tree",
              "id": "Q107394029",
              "label_en": "Moko"
            }
          ],
          "P2306": [
            {
              "description_en": "geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported",
              "id": "P625",
              "label_en": "coordinate location"
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
              "description_en": "fictional character appearing in a video game",
              "id": "Q1569167",
              "label_en": "video game character"
            },
            {
              "description_en": "unknown creator of a work (do not use as value of P50; use \"unknown value\" instead)",
              "id": "Q4233718",
              "label_en": "anonymous"
            },
            {
              "description_en": "fictional human or non-human character in a narrative work of art",
              "id": "Q95074",
              "label_en": "character"
            }
          ],
          "P2306": [
            {
              "description_en": "this item is a subclass (subset) of that item; ALL instances of this item are instances of that item; different from P31 (instance of), e.g.: volcano is a subclass of mountain; Everest is an instance of mountain",
              "id": "P279",
              "label_en": "subclass of"
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
          "P2303": [
            {
              "description_en": "well-preserved natural prehistoric mummy",
              "id": "Q171291",
              "label_en": "Ötzi"
            },
            {
              "description_en": "either of two Canadian-made hitchhiking robots",
              "id": "Q17450611",
              "label_en": "hitchBOT"
            },
            {
              "description_en": "operatic character in the opera The Nose by Dmitri Shostakovich; a Collegiate Assessor",
              "id": "Q55039999",
              "label_en": "Platon Kuzmich Koavalyov"
            },
            {
              "description_en": "operatic character in the opera Mauerschau",
              "id": "Q55052424",
              "label_en": "Penthesilea"
            },
            {
              "description_en": "fictional two-tailed fox familiar in Pepper&Carrot",
              "id": "Q75840084",
              "label_en": "Yuzu"
            }
          ],
          "P2306": [
            {
              "description_en": "part of this subject; inverse property of \"part of\" (P361). See also \"has parts of the class\" (P2670).",
              "id": "P527",
              "label_en": "has part(s)"
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
              "description_en": "part of a naming scheme for individuals, used in many cultures worldwide",
              "id": "Q101352",
              "label_en": "family name"
            },
            {
              "description_en": "unlawful killing of a human with malice aforethought",
              "id": "Q132821",
              "label_en": "murder"
            },
            {
              "description_en": "page of a Wikimedia project with a list of something",
              "id": "Q13406463",
              "label_en": "Wikimedia list article"
            },
            {
              "description_en": "list of films related by some criteria",
              "id": "Q1371849",
              "label_en": "filmography"
            },
            {
              "description_en": "study and cataloging of published sound recordings",
              "id": "Q273057",
              "label_en": "discography"
            },
            {
              "description_en": "intellectual or artistic creation",
              "id": "Q386724",
              "label_en": "work"
            },
            {
              "description_en": "permanent cessation of vital functions",
              "id": "Q4",
              "label_en": "death"
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
              "description_en": "narrative text, normally of a substantial length and in the form of prose describing a fictional and sequential story",
              "id": "Q8261",
              "label_en": "novel"
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
              "description_en": "two individuals who work together",
              "id": "Q10648343",
              "label_en": "duo"
            },
            {
              "description_en": "two comedians who perform together as a single act",
              "id": "Q1141470",
              "label_en": "double act"
            },
            {
              "description_en": "consists monarchs who are related to one another, as well as their non-reigning descendants and spouses",
              "id": "Q1156073",
              "label_en": "royal house"
            },
            {
              "description_en": "family part of the nobility of a region or country",
              "id": "Q13417114",
              "label_en": "noble family"
            },
            {
              "description_en": "two siblings that work together",
              "id": "Q14073567",
              "label_en": "sibling duo"
            },
            {
              "description_en": "two offspring produced in the same pregnancy (use with P31 on items for both twins - if known, item for \"identical twins\" or \"fraternal twins\" is preferred)",
              "id": "Q14756018",
              "label_en": "twins"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "sequence of rulers considered members of the same family",
              "id": "Q164950",
              "label_en": "dynasty"
            },
            {
              "description_en": "group of humans, they are siblings; e.g. Grimm brothers, Brontë sisters",
              "id": "Q16979650",
              "label_en": "sibling group"
            },
            {
              "description_en": "twins with the same genes, that grew from one egg that split into two",
              "id": "Q2301325",
              "label_en": "identical twins"
            },
            {
              "description_en": "group of three musicians",
              "id": "Q281643",
              "label_en": "musical trio"
            },
            {
              "description_en": "two people who are married to each other",
              "id": "Q3046146",
              "label_en": "married couple"
            },
            {
              "description_en": "one of two or more individuals having at least one parent in common",
              "id": "Q31184",
              "label_en": "sibling"
            },
            {
              "description_en": "twins formed by two separate eggs fertilised by two separate sperms",
              "id": "Q3418125",
              "label_en": "fraternal twins"
            },
            {
              "description_en": "group of people affiliated by consanguinity, law, affinity, or co-residence",
              "id": "Q8436",
              "label_en": "family"
            },
            {
              "description_en": "two or more humans who interact with one another",
              "id": "Q874405",
              "label_en": "social group"
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
              "description_en": "company that publishes video games",
              "id": "Q1137109",
              "label_en": "video game publisher"
            },
            {
              "description_en": "software development organization specializing in the creation of video games (for person use Q58287519)",
              "id": "Q210167",
              "label_en": "video game developer"
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
              "description_en": "list article about a set of items of a specific type that share the same (or similar) name",
              "id": "Q15623926",
              "label_en": "Wikimedia set index article"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "name typically used to differentiate people from the same family, clan, or other social group who have a common last name",
              "id": "Q202444",
              "label_en": "given name"
            },
            {
              "description_en": "group of people who perform instrumental and/or vocal music, with the ensemble typically known by a distinct name",
              "id": "Q2088357",
              "label_en": "musical ensemble"
            },
            {
              "description_en": "this item duplicates another item, the two can't be merged, as one Wikimedia project includes 2 pages, e.g. in different scripts or languages. Add properties other than P2959 (permanent duplicated item) and sitelinks for other wikis to the other item",
              "id": "Q21286738",
              "label_en": "Wikimedia permanent duplicate item"
            },
            {
              "description_en": "musical ensemble which performs music",
              "id": "Q215380",
              "label_en": "musical group"
            },
            {
              "description_en": "type of wiki page usually in main namespace (article namespace, ns=0) containing links to articles with similar names, and very little details only, use with P31 \"instance of\"",
              "id": "Q4167410",
              "label_en": "Wikimedia disambiguation page"
            },
            {
              "description_en": "medium for recording information (words or images) typically on bound pages or more abstractly in electronic or audio form",
              "id": "Q571",
              "label_en": "book"
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
          "P2306": [
            {
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
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
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
              "description_en": "common element between all listed items",
              "id": "P360",
              "label_en": "is a list of"
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
              "description_en": "language in which a film or a performance work was originally created. Deprecated for written works and songs; use P407 (\"language of work or name\") instead.",
              "id": "P364",
              "label_en": "original language of film or TV show"
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
              "description_en": "main creator(s) of a written work (use on works, not humans); use P2093 (author name string) when Wikidata item is unknown or does not exist",
              "id": "P50",
              "label_en": "author"
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
              "description_en": "anthropomorphic sexual device",
              "id": "Q10832",
              "label_en": "sex doll"
            },
            {
              "description_en": "mechanical or virtual artificial agent carrying out physical activities, which can be guided by an external control device or the control may be embedded within",
              "id": "Q11012",
              "label_en": "robot"
            },
            {
              "description_en": "type of a doll or (action) figure that may appear in different variants",
              "id": "Q111282474",
              "label_en": "doll or action figure model"
            },
            {
              "description_en": "character known only from narrations (fictional or in a factual manner) without a proof of existence; includes fictional, mythical, legendary or religious characters and similar",
              "id": "Q115537581",
              "label_en": "imaginary character"
            },
            {
              "description_en": "teknonym in an Arabic name, the name of an adult derived from their eldest son",
              "id": "Q1285470",
              "label_en": "kunya"
            },
            {
              "description_en": "entity that has no physical realisation",
              "id": "Q15619164",
              "label_en": "abstract being"
            },
            {
              "description_en": "group of one or more fictional organism(s), which a (fictional) taxonomist adjudges to be a unit",
              "id": "Q15707583",
              "label_en": "fictional taxon"
            },
            {
              "description_en": "group of one or more organism(s), which a taxonomist adjudges to be a unit",
              "id": "Q16521",
              "label_en": "taxon"
            },
            {
              "description_en": "model of a character or a human being, often used as a toy for children or an artistic hobby for adults",
              "id": "Q168658",
              "label_en": "doll"
            },
            {
              "description_en": "Latin phrase; alternate self",
              "id": "Q201662",
              "label_en": "alter ego"
            },
            {
              "description_en": "individual who died before or during birth",
              "id": "Q2345820",
              "label_en": "stillborn child"
            },
            {
              "description_en": "organism not more specified in a work of fiction",
              "id": "Q2593744",
              "label_en": "fictional creature"
            },
            {
              "description_en": "singular named exemplar of an animal (e.g., the gorilla named Koko; the cat named Socks)",
              "id": "Q26401003",
              "label_en": "individual animal"
            },
            {
              "description_en": "prenatal organism between the embryonic state and birth",
              "id": "Q26513",
              "label_en": "human fetus"
            },
            {
              "description_en": "preserved remains or traces of organisms from a past geological age",
              "id": "Q40614",
              "label_en": "fossil"
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
            },
            {
              "description_en": "any individual living being or physical living system",
              "id": "Q7239",
              "label_en": "organism"
            },
            {
              "description_en": "kingdom of multicellular eukaryotic organisms",
              "id": "Q729",
              "label_en": "Animalia"
            },
            {
              "description_en": "a human-sounding voice generated by a computer",
              "id": "Q79600797",
              "label_en": "synthetic voice"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
            {
              "description_en": "work or narration for or in which this statement is true",
              "id": "P10663",
              "label_en": "applies to work"
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
              "description_en": "qualifier property to be used with statements having the object \"no value\", given to provide a reason for \"no value\"",
              "id": "P13589",
              "label_en": "‎reason for no value"
            },
            {
              "description_en": "qualification of the truth or accuracy of a source: circa (Q5727902), near (Q21818619), presumably (Q18122778), etc.",
              "id": "P1480",
              "label_en": "sourcing circumstances"
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
              "description_en": "entity that supports a given statement",
              "id": "P3680",
              "label_en": "statement supported by"
            },
            {
              "description_en": "how a value is determined, or the standard by which it is declared",
              "id": "P459",
              "label_en": "determination method or standard"
            },
            {
              "description_en": "(qualifier only) the underlying circumstances of this statement",
              "id": "P5102",
              "label_en": "nature of statement"
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
              "description_en": "time of the first public presentation of a subject by the creator, of information by the media",
              "id": "P6949",
              "label_en": "announcement date"
            },
            {
              "description_en": "qualifier to allow the reason to be indicated why a particular statement should be considered preferred",
              "id": "P7452",
              "label_en": "reason for preferred rank"
            },
            {
              "description_en": "reading this information or the content of the identifier might ruin the experience of receiving this work",
              "id": "P7528",
              "label_en": "statement or content of identifier is regarded as spoiler for"
            },
            {
              "description_en": "(qualifier) earliest date on which the statement could have begun to no longer be true",
              "id": "P8554",
              "label_en": "earliest end date"
            },
            {
              "description_en": "(qualifier) latest date on which the statement could have started to be true",
              "id": "P8555",
              "label_en": "latest start date"
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
          "P2305": [
            {
              "description_en": "woman assigned male at birth",
              "id": "Q1052281",
              "label_en": "trans woman"
            },
            {
              "description_en": "third gender in Niue culture",
              "id": "Q107427210",
              "label_en": "fakafifine"
            },
            {
              "description_en": "transgender identity of individuals who identify on the neutral side of the gender spectrum",
              "id": "Q107502361",
              "label_en": "transneutral"
            },
            {
              "description_en": "atypical congenital variations of sex characteristics",
              "id": "Q1097630",
              "label_en": "intersex"
            },
            {
              "description_en": "transgender women in Wallisian and Futunan culture",
              "id": "Q112597587",
              "label_en": "fakafafine"
            },
            {
              "description_en": "person born with any of several variations in sex characteristics including chromosomes, gonads, sex hormones or genitals that do not fit the typical definitions for male or female bodies",
              "id": "Q11287467",
              "label_en": "intersex person"
            },
            {
              "description_en": "datum representing when an individual has not publicly disclosed their gender identity or has expressed a wish not to disclose their gender identity, thereby making their gender identity unknown",
              "id": "Q113124952",
              "label_en": "undisclosed gender"
            },
            {
              "description_en": "case in which a fictional character's gender is not outright disclosed in a work and cannot be reasonably assumed",
              "id": "Q116254116",
              "label_en": "gender not disclosed in work"
            },
            {
              "description_en": "gender listing within the Pokémon franchise",
              "id": "Q116741172",
              "label_en": "gender unknown"
            },
            {
              "description_en": "intersex person who identifies as a man",
              "id": "Q121307094",
              "label_en": "intersex man"
            },
            {
              "description_en": "intersex person who identifies as a woman",
              "id": "Q121307100",
              "label_en": "intersex woman"
            },
            {
              "description_en": "gender identity where a person identifies partially as a man/boy or as otherwise being partly masculine in nature",
              "id": "Q121368243",
              "label_en": "demimasc"
            },
            {
              "description_en": "Japanese term referring to female characters with male genitalia either at birth or grownth through supernatural, magical or technological means",
              "id": "Q123479538",
              "label_en": "futanari"
            },
            {
              "description_en": "a term to describe an individual who considers the phenomenon of gender identity to be unknowable or indeterminable by its very nature",
              "id": "Q124637723",
              "label_en": "gender agnostic"
            },
            {
              "description_en": "non-binary gender identity",
              "id": "Q1289754",
              "label_en": "neutrois"
            },
            {
              "description_en": "range of gender identities that are not exclusively masculine or feminine and does not match with assigned sex",
              "id": "Q12964198",
              "label_en": "genderqueer"
            },
            {
              "description_en": "indigenous Australian transfeminine gender",
              "id": "Q130315001",
              "label_en": "sistergirl"
            },
            {
              "description_en": "indigenous australian transmasculine gender identity",
              "id": "Q130315012",
              "label_en": "brotherboy"
            },
            {
              "description_en": "individual who identifies with both female and non-binary gender identities",
              "id": "Q130477254",
              "label_en": "non-binary woman"
            },
            {
              "description_en": "individual who identifies with both male and non-binary gender identities",
              "id": "Q130477279",
              "label_en": "non-binary man"
            },
            {
              "description_en": "having both sexes, including naturally or intentionally",
              "id": "Q130899399",
              "label_en": "bisex"
            },
            {
              "description_en": "זהות מגדרית; דברים שמתאימים למין אחד",
              "id": "Q130964491",
              "label_en": "monogender"
            },
            {
              "description_en": "In Samoan culture, Samoan people who were assigned male at birth but behave in feminine ways and may consider themselves women",
              "id": "Q1399232",
              "label_en": "faʻafafine"
            },
            {
              "description_en": "man who was assigned male at birth and identifies as male",
              "id": "Q15145778",
              "label_en": "cisgender man"
            },
            "... omitted 47 items"
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
          "P2305": [
            {
              "description_en": "romantic and/or sexual attraction or behavior between people of different genders",
              "id": "Q1035954",
              "label_en": "heterosexuality"
            },
            {
              "description_en": "umbrella term for sexual orientations and identities where someone experiences attraction to more than one gender",
              "id": "Q106589379",
              "label_en": "plurisexuality"
            },
            {
              "description_en": "attraction only to those they are mentally connected to the person",
              "id": "Q108472356",
              "label_en": "noetisexuality"
            },
            {
              "description_en": "umbrella term for men loving men",
              "id": "Q115068942",
              "label_en": "Achillean"
            },
            {
              "description_en": "umbrella term for attractions and relationships involving at least one non-binary person",
              "id": "Q116280707",
              "label_en": "diamoric"
            },
            {
              "description_en": "non-binary people attracted to women",
              "id": "Q116286410",
              "label_en": "trixic"
            },
            {
              "description_en": "non-binary people attracted to men",
              "id": "Q116295197",
              "label_en": "toric"
            },
            {
              "description_en": "attraction to three genders or experiencing three sexualities",
              "id": "Q116820152",
              "label_en": "trisexuality"
            },
            {
              "description_en": "case in which the sexual orientation of a character in a video game is determined by the player",
              "id": "Q123138223",
              "label_en": "sexuality determined by the player"
            },
            {
              "description_en": "sexual, romantic, queer- & platonic, sensual and another attraction to women or femininity",
              "id": "Q1558475",
              "label_en": "gynephilia"
            },
            {
              "description_en": "changes in sexuality or sexual identity",
              "id": "Q19810527",
              "label_en": "sexual fluidity"
            },
            {
              "description_en": "sexual attraction based primarily on intellect",
              "id": "Q20011275",
              "label_en": "sapiosexuality"
            },
            {
              "description_en": "sexual attraction to multiple, but not all, genders",
              "id": "Q2094204",
              "label_en": "polysexuality"
            },
            {
              "description_en": "romantic attraction, sexual attraction, or sexual behavior between two men",
              "id": "Q2257941",
              "label_en": "male homosexuality"
            },
            {
              "description_en": "sexual attraction only to people with whom emotional bonds are formed",
              "id": "Q23912283",
              "label_en": "demisexuality"
            },
            {
              "description_en": "term used by individuals who do not wish to label their sexuality with more specific terms",
              "id": "Q25326668",
              "label_en": "unlabeled sexuality"
            },
            {
              "description_en": "umbrella term for feminine person loving feminine person",
              "id": "Q25447263",
              "label_en": "sapphism"
            },
            {
              "description_en": "person who is attracted to experiencing bisexuality",
              "id": "Q255155",
              "label_en": "bi-curious"
            },
            {
              "description_en": "sexual or romantic attraction to people regardless of gender",
              "id": "Q271534",
              "label_en": "pansexuality"
            },
            {
              "description_en": "sexual orientation other than heterosexual or straight",
              "id": "Q339014",
              "label_en": "non-heterosexuality"
            },
            {
              "description_en": "sexual orientation of people who reject, avoid or do not fit into any sexual orientation label",
              "id": "Q3626860",
              "label_en": "pomosexuality"
            },
            {
              "description_en": "sexual and/or romantic attraction to people of more than one gender",
              "id": "Q43200",
              "label_en": "bisexuality"
            },
            {
              "description_en": "sexual attraction to males, men or masculinity",
              "id": "Q43850015",
              "label_en": "androsexuality"
            },
            {
              "description_en": "sexual attraction to women, femininity or females",
              "id": "Q43850027",
              "label_en": "gynesexuality"
            },
            "... omitted 9 items"
          ],
          "P6824": [
            {
              "description_en": "the sexual orientation of the person relative to their declared gender — use ONLY IF they have stated it themselves, unambiguously, or it has been widely agreed upon by historians after their death",
              "id": "P91",
              "label_en": "sexual orientation"
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
              "description_en": "man who is sexually and/or romantically attracted to more than one gender",
              "id": "Q105079567",
              "label_en": "bisexual man"
            },
            {
              "description_en": "woman who is sexually and/or romantically attracted to more than one gender",
              "id": "Q105079916",
              "label_en": "bisexual woman"
            },
            {
              "description_en": "someone with a non-binary gender identity attracted to more than one gender",
              "id": "Q115870499",
              "label_en": "non-binary bisexual"
            },
            {
              "description_en": "person who is sexually and/or romantically attracted to persons with a gender or genders like theirs and those with a gender or genders unlike theirs",
              "id": "Q12905217",
              "label_en": "bisexual person"
            }
          ],
          "P6824": [
            {
              "description_en": "the sexual orientation of the person relative to their declared gender — use ONLY IF they have stated it themselves, unambiguously, or it has been widely agreed upon by historians after their death",
              "id": "P91",
              "label_en": "sexual orientation"
            }
          ],
          "P9729": [
            {
              "description_en": "sexual and/or romantic attraction to people of more than one gender",
              "id": "Q43200",
              "label_en": "bisexuality"
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
              "description_en": "man who is sexually and/or romantically attracted to more than one gender",
              "id": "Q105079567",
              "label_en": "bisexual man"
            },
            {
              "description_en": "male human that identifies as their sex assigned at birth and is sexually attracted to females",
              "id": "Q134897761",
              "label_en": "cisgender heterosexual man"
            },
            {
              "description_en": "young male human",
              "id": "Q3010",
              "label_en": "boy"
            },
            {
              "description_en": "male adult human",
              "id": "Q8441",
              "label_en": "man"
            },
            {
              "description_en": "demographic classification",
              "id": "Q853451",
              "label_en": "men who have sex with men"
            },
            {
              "description_en": "boy between birth-23 months",
              "id": "Q96780034",
              "label_en": "baby boy"
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
              "description_en": "woman who is sexually and/or romantically attracted to more than one gender",
              "id": "Q105079916",
              "label_en": "bisexual woman"
            },
            {
              "description_en": "female human that identifies as their sex assigned at birth and is sexually attracted to males",
              "id": "Q134897481",
              "label_en": "cisgender heterosexual woman"
            },
            {
              "description_en": "sexual identity-neutral term",
              "id": "Q210604",
              "label_en": "women who have sex with women"
            },
            {
              "description_en": "young female human",
              "id": "Q3031",
              "label_en": "girl"
            },
            {
              "description_en": "female adult human",
              "id": "Q467",
              "label_en": "woman"
            },
            {
              "description_en": "very young female human",
              "id": "Q97009651",
              "label_en": "baby girl"
            }
          ],
          "P9729": [
            {
              "description_en": "to be used in \"sex or gender\" (P21) to indicate that the human subject is a female or \"semantic gender\" (P10339) to indicate that a word refers to a female person",
              "id": "Q6581072",
              "label_en": "female"
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
              "description_en": "someone who is transgender",
              "id": "Q10701290",
              "label_en": "transgender person"
            },
            {
              "description_en": "a slang for a transgender person who is heterosexual",
              "id": "Q124726070",
              "label_en": "transhet"
            }
          ],
          "P9729": [
            {
              "description_en": "gender identity different to the gender assigned at birth",
              "id": "Q189125",
              "label_en": "transgender"
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
              "description_en": "male human that identifies as and is sexually attracted to their sex assigned at birth",
              "id": "Q107785560",
              "label_en": "cisgender gay male"
            },
            {
              "description_en": "female human that identifies as and is sexually attracted to their sex assigned at birth",
              "id": "Q124637919",
              "label_en": "cisgender lesbian"
            },
            {
              "description_en": "female human that identifies as their sex assigned at birth and is sexually attracted to males",
              "id": "Q134897481",
              "label_en": "cisgender heterosexual woman"
            },
            {
              "description_en": "male human that identifies as their sex assigned at birth and is sexually attracted to females",
              "id": "Q134897761",
              "label_en": "cisgender heterosexual man"
            }
          ],
          "P9729": [
            {
              "description_en": "correspondence between a person's gender identity and the sex assigned to them at birth",
              "id": "Q1093205",
              "label_en": "cisgender"
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
              "description_en": "Chinese given name (雄)",
              "id": "Q108598090",
              "label_en": "Xiong"
            },
            {
              "description_en": "CJK (hanzi/kanji/hanja) character",
              "id": "Q55814929",
              "label_en": "雄"
            }
          ],
          "P2316": [
            {
              "description_en": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
              "id": "Q21502408",
              "label_en": "mandatory constraint"
            }
          ],
          "P9729": [
            {
              "description_en": "organism of the male sex",
              "id": "Q44148",
              "label_en": "male organism"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      "... omitted 14 items"
    ]
  },
  "hash_after": "4153e44669496bf040a48eb9d9fecb724aee43d6",
  "hash_before": "7f4c42dc094e07e6c9eafdc0fbb8579598343f36",
  "property_revision_id": 2439154480,
  "property_revision_prev": 2439154461,
  "qualifier_value_changes": [
    {
      "added_values": [],
      "constraint_qid": "Q52558054",
      "qualifier_property": "P6824",
      "removed_values": [
        "P21"
      ],
      "same_qid_index": 14
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
            "Q124726070"
          ]
        },
        {
          "property_id": "P6824",
          "values": [
            "P21"
          ]
        },
        {
          "property_id": "P9729",
          "values": [
            "Q189125"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "conflicts-with constraint: reason for deprecated rank: constraint provides suggestions for manual input; exception to constraint: Sibyl System, hitchBOT; property: country; constraint status: suggestion constraint",
      "conflicts-with constraint: exception to constraint: Moko; property: coordinate location",
      "conflicts-with constraint: exception to constraint: video game character, anonymous, character; property: subclass of; constraint status: mandatory constraint",
      "conflicts-with constraint: exception to constraint: Ötzi, hitchBOT, Platon Kuzmich Koavalyov, Penthesilea, Yuzu; property: has part(s); constraint status: mandatory constraint",
      "conflicts-with constraint: item of property constraint: family name, murder, Wikimedia list article, filmography, discography, work, death, position, Wikimedia category, novel; property: instance of; constraint status: mandatory constraint",
      "conflicts-with constraint: item of property constraint: duo, double act, royal house, noble family, sibling duo, twins, group of humans, dynasty, sibling group, identical twins, musical trio, married couple, sibling, fraternal twins, family, social group; property: instance of",
      "conflicts-with constraint: item of property constraint: video game publisher, video game developer; property: instance of",
      "conflicts-with constraint: item of property constraint: Wikimedia set index article, group of humans, given name, musical ensemble, Wikimedia permanent duplicate item, musical group, Wikimedia disambiguation page, book; property: instance of",
      "conflicts-with constraint: property: located in the administrative territorial entity; constraint status: mandatory constraint",
      "conflicts-with constraint: property: country",
      "conflicts-with constraint: property: is a list of; constraint status: mandatory constraint",
      "conflicts-with constraint: property: original language of film or TV show; constraint status: mandatory constraint",
      "conflicts-with constraint: property: author; constraint status: mandatory constraint",
      "subject type constraint: class: sex doll, robot, doll or action figure model, imaginary character, kunya, abstract being, fictional taxon, taxon, doll, alter ego, stillborn child, fictional creature, individual animal, human fetus, fossil, human, pseudonym, organism, Animalia, synthetic voice; relation: instance of; constraint status: mandatory constraint",
      "allowed qualifiers constraint: property: applies to work, earliest date, latest date, ‎reason for no value, sourcing circumstances, object named as, reason for deprecated rank, statement supported by, determination method or standard, nature of statement, applies to part, start time, end time, announcement date, reason for preferred rank, statement or content of identifier is regarded as spoiler for, earliest end date, latest start date; constraint status: mandatory constraint",
      "one-of constraint: item of property constraint: trans woman, fakafifine, transneutral, intersex, fakafafine, intersex person, undisclosed gender, gender not disclosed in work, gender unknown, intersex man, intersex woman, demimasc, futanari, gender agnostic, neutrois, genderqueer, sistergirl, brotherboy, non-binary woman, non-binary man, bisex, monogender, faʻafafine, cisgender man, cisgender woman, castrated creature, hermaphroditism, travesti, eunuch, genderflui... [truncated 560 chars]",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы; constraint status: mandatory constraint",
      "none-of constraint: item of property constraint: heterosexuality, plurisexuality, noetisexuality, Achillean, diamoric, trixic, toric, trisexuality, sexuality determined by the player, gynephilia, sexual fluidity, sapiosexuality, polysexuality, male homosexuality, demisexuality, unlabeled sexuality, sapphism, bi-curious, pansexuality, non-heterosexuality, pomosexuality, bisexuality, androsexuality, gynesexuality, androphilia, queer, gay, homosexuality, lesbianism, ... [truncated 101 chars]",
      "none-of constraint: item of property constraint: bisexual man, bisexual woman, non-binary bisexual, bisexual person; replacement property: sexual orientation; replacement value: bisexuality",
      "none-of constraint: item of property constraint: bisexual man, cisgender heterosexual man, boy, man, men who have sex with men, baby boy; replacement value: male",
      "none-of constraint: item of property constraint: bisexual woman, cisgender heterosexual woman, women who have sex with women, girl, woman, baby girl; replacement value: female",
      "none-of constraint: item of property constraint: transgender person, transhet; replacement value: transgender",
      "none-of constraint: item of property constraint: cisgender gay male, cisgender lesbian, cisgender heterosexual woman, cisgender heterosexual man; replacement value: cisgender",
      "none-of constraint: item of property constraint: Xiong, 雄; constraint status: mandatory constraint; replacement value: male organism",
      "... omitted 14 items"
    ],
    "before": [
      "conflicts-with constraint: reason for deprecated rank: constraint provides suggestions for manual input; exception to constraint: Sibyl System, hitchBOT; property: country; constraint status: suggestion constraint",
      "conflicts-with constraint: exception to constraint: Moko; property: coordinate location",
      "conflicts-with constraint: exception to constraint: video game character, anonymous, character; property: subclass of; constraint status: mandatory constraint",
      "conflicts-with constraint: exception to constraint: Ötzi, hitchBOT, Platon Kuzmich Koavalyov, Penthesilea, Yuzu; property: has part(s); constraint status: mandatory constraint",
      "conflicts-with constraint: item of property constraint: family name, murder, Wikimedia list article, filmography, discography, work, death, position, Wikimedia category, novel; property: instance of; constraint status: mandatory constraint",
      "conflicts-with constraint: item of property constraint: duo, double act, royal house, noble family, sibling duo, twins, group of humans, dynasty, sibling group, identical twins, musical trio, married couple, sibling, fraternal twins, family, social group; property: instance of",
      "conflicts-with constraint: item of property constraint: video game publisher, video game developer; property: instance of",
      "conflicts-with constraint: item of property constraint: Wikimedia set index article, group of humans, given name, musical ensemble, Wikimedia permanent duplicate item, musical group, Wikimedia disambiguation page, book; property: instance of",
      "conflicts-with constraint: property: located in the administrative territorial entity; constraint status: mandatory constraint",
      "conflicts-with constraint: property: country",
      "conflicts-with constraint: property: is a list of; constraint status: mandatory constraint",
      "conflicts-with constraint: property: original language of film or TV show; constraint status: mandatory constraint",
      "conflicts-with constraint: property: author; constraint status: mandatory constraint",
      "subject type constraint: class: sex doll, robot, doll or action figure model, imaginary character, kunya, abstract being, fictional taxon, taxon, doll, alter ego, stillborn child, fictional creature, individual animal, human fetus, fossil, human, pseudonym, organism, Animalia, synthetic voice; relation: instance of; constraint status: mandatory constraint",
      "allowed qualifiers constraint: property: applies to work, earliest date, latest date, ‎reason for no value, sourcing circumstances, object named as, reason for deprecated rank, statement supported by, determination method or standard, nature of statement, applies to part, start time, end time, announcement date, reason for preferred rank, statement or content of identifier is regarded as spoiler for, earliest end date, latest start date; constraint status: mandatory constraint",
      "one-of constraint: item of property constraint: trans woman, fakafifine, transneutral, intersex, fakafafine, intersex person, undisclosed gender, gender not disclosed in work, gender unknown, intersex man, intersex woman, demimasc, futanari, gender agnostic, neutrois, genderqueer, sistergirl, brotherboy, non-binary woman, non-binary man, bisex, monogender, faʻafafine, cisgender man, cisgender woman, castrated creature, hermaphroditism, travesti, eunuch, genderflui... [truncated 560 chars]",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, МэдыяІнфа Вікібазы; constraint status: mandatory constraint",
      "none-of constraint: item of property constraint: heterosexuality, plurisexuality, noetisexuality, Achillean, diamoric, trixic, toric, trisexuality, sexuality determined by the player, gynephilia, sexual fluidity, sapiosexuality, polysexuality, male homosexuality, demisexuality, unlabeled sexuality, sapphism, bi-curious, pansexuality, non-heterosexuality, pomosexuality, bisexuality, androsexuality, gynesexuality, androphilia, queer, gay, homosexuality, lesbianism, ... [truncated 101 chars]",
      "none-of constraint: item of property constraint: bisexual man, bisexual woman, non-binary bisexual, bisexual person; replacement property: sexual orientation; replacement value: bisexuality",
      "none-of constraint: item of property constraint: bisexual man, cisgender heterosexual man, boy, man, men who have sex with men, baby boy; replacement value: male",
      "none-of constraint: item of property constraint: bisexual woman, cisgender heterosexual woman, women who have sex with women, girl, woman, baby girl; replacement value: female",
      "none-of constraint: item of property constraint: transgender person, transhet; replacement value: transgender",
      "none-of constraint: item of property constraint: cisgender gay male, cisgender lesbian, cisgender heterosexual woman, cisgender heterosexual man; replacement value: cisgender",
      "none-of constraint: item of property constraint: Xiong, 雄; constraint status: mandatory constraint; replacement value: male organism",
      "... omitted 14 items"
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
    "violation_name": "Conflicts with P|625"
  },
  {
    "result": "Q21502838",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 018. `reform_Q80476774_P12496_2441161838`

| Field | Value |
|---|---|
| qid | Q80476774 |
| property | P12496 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | mid |
| constraint_family | Q19474404 |
| group_key | TBOX::P12496::2441161838 |
| tbox_revision_key | TBOX::P12496::2441161838 |

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
  "author": "Tæppa",
  "kind": "T_BOX",
  "property_revision_id": 2441161838,
  "property_revision_prev": 2441161737
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-13T05:02:48",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P12496",
  "report_revision_new": 2441621369,
  "report_revision_old": 2440731542,
  "report_violation_type": "Item P|131",
  "report_violation_type_normalized": "Item P|131",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Item P|131",
  "value": null,
  "value_current_2026": [
    "NDE/187"
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
    "description": "identifier for a pub on WhatPub",
    "label": "WhatPub pub ID"
  },
  "qid": {
    "description": "pub in Instow, Devon, UK",
    "label": "The Wayfarer Inn"
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
    "label_en": "description in language constraint",
    "qid": "Q111204896"
  }
]
```

### T-box Constraint Diff

```json
{
  "added_constraint_entries": [
    {
      "_count": 1,
      "constraint_qid": "Q111204896",
      "qualifiers": [
        {
          "property_id": "P424",
          "values": [
            "en"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 12,
  "author": "Tæppa",
  "before_constraint_count": 11,
  "changed_constraint_types": [
    "Q111204896"
  ],
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
              "value": "en"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "constraint to ensure items using a property have description in the language. Use qualifier \"WMF language code\" (P424) to define language.",
          "id": "Q111204896",
          "label_en": "description in language constraint"
        },
        "parameters": {
          "P424": [
            {
              "value": "en"
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
              "value": "^[A-Z]{3}/[\\w-]+$"
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
              "description_en": "former pub in Killinghall, North Yorkshire, England",
              "id": "Q131312045",
              "label_en": "The Travellers Rest"
            },
            {
              "description_en": "pub in Killinghall, North Yorkshire, England",
              "id": "Q131312055",
              "label_en": "Curious Cow"
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
          "P2303": [
            {
              "description_en": "pub in Laugharne, Wales",
              "id": "Q80582145",
              "label_en": "The Under Milk Wood"
            },
            {
              "description_en": "pub in Cardiff, Wales",
              "id": "Q80584783",
              "label_en": "Chiquito"
            },
            {
              "description_en": "pub in Cardiff, Wales",
              "id": "Q80859081",
              "label_en": "Cafe Rouge"
            },
            {
              "description_en": "pub in Cardiff, Wales",
              "id": "Q80859116",
              "label_en": "Mimosa"
            }
          ],
          "P2306": [
            {
              "description_en": "Whatpub /CAMRA's database is a comprehensive repository of public houses in the UK with a focus on the ales that are traditional there",
              "id": "P13516",
              "label_en": "CAMRA pub ID"
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
              "description_en": "island country in north-west Europe",
              "id": "Q145",
              "label_en": "United Kingdom"
            },
            {
              "description_en": "one of the British Crown Dependencies in the Channel Islands",
              "id": "Q25230",
              "label_en": "Guernsey"
            },
            {
              "description_en": "British Crown dependency in the Channel Islands",
              "id": "Q785",
              "label_en": "Jersey"
            },
            {
              "description_en": "historic nation and a self-governing British Crown dependency",
              "id": "Q9676",
              "label_en": "Isle of Man"
            }
          ],
          "P2306": [
            {
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
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
              "description_en": "external identifier for pubs in the United Kingdom",
              "id": "P7411",
              "label_en": "Pubs Galore ID"
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
              "description_en": "business whose primary function is the serving of alcoholic beverages for consumption on the premises",
              "id": "Q5307737",
              "label_en": "alcohol drinking establishment"
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
              "value": "en"
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
              "value": "^[A-Z]{3}/[\\w-]+$"
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
              "description_en": "former pub in Killinghall, North Yorkshire, England",
              "id": "Q131312045",
              "label_en": "The Travellers Rest"
            },
            {
              "description_en": "pub in Killinghall, North Yorkshire, England",
              "id": "Q131312055",
              "label_en": "Curious Cow"
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
          "P2303": [
            {
              "description_en": "pub in Laugharne, Wales",
              "id": "Q80582145",
              "label_en": "The Under Milk Wood"
            },
            {
              "description_en": "pub in Cardiff, Wales",
              "id": "Q80584783",
              "label_en": "Chiquito"
            },
            {
              "description_en": "pub in Cardiff, Wales",
              "id": "Q80859081",
              "label_en": "Cafe Rouge"
            },
            {
              "description_en": "pub in Cardiff, Wales",
              "id": "Q80859116",
              "label_en": "Mimosa"
            }
          ],
          "P2306": [
            {
              "description_en": "Whatpub /CAMRA's database is a comprehensive repository of public houses in the UK with a focus on the ales that are traditional there",
              "id": "P13516",
              "label_en": "CAMRA pub ID"
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
              "description_en": "island country in north-west Europe",
              "id": "Q145",
              "label_en": "United Kingdom"
            },
            {
              "description_en": "one of the British Crown Dependencies in the Channel Islands",
              "id": "Q25230",
              "label_en": "Guernsey"
            },
            {
              "description_en": "British Crown dependency in the Channel Islands",
              "id": "Q785",
              "label_en": "Jersey"
            },
            {
              "description_en": "historic nation and a self-governing British Crown dependency",
              "id": "Q9676",
              "label_en": "Isle of Man"
            }
          ],
          "P2306": [
            {
              "description_en": "sovereign state that this item is in (not to be used for human beings)",
              "id": "P17",
              "label_en": "country"
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
              "description_en": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
              "id": "P131",
              "label_en": "located in the administrative territorial entity"
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
              "description_en": "external identifier for pubs in the United Kingdom",
              "id": "P7411",
              "label_en": "Pubs Galore ID"
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
              "description_en": "business whose primary function is the serving of alcoholic beverages for consumption on the premises",
              "id": "Q5307737",
              "label_en": "alcohol drinking establishment"
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
  "hash_after": "5bc450a4f62f6f37890025a7d03505b0b41277d9",
  "hash_before": "5902be22efa2d31f5021ae91d90fae0e2d437a82",
  "property_revision_id": 2441161838,
  "property_revision_prev": 2441161737,
  "qualifier_value_changes": [],
  "removed_constraint_entries": [],
  "rule_summaries_en": {
    "after": [
      "label in language constraint: Wikimedia language code: en",
      "description in language constraint: Wikimedia language code: en",
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: ^[A-Z]{3}/[\\w-]+$",
      "distinct-values constraint: exception to constraint: The Travellers Rest, Curious Cow",
      "item-requires-statement constraint: exception to constraint: The Under Milk Wood, Chiquito, Cafe Rouge, Mimosa; property: CAMRA pub ID",
      "item-requires-statement constraint: item of property constraint: United Kingdom, Guernsey, Jersey, Isle of Man; property: country",
      "item-requires-statement constraint: property: located in the administrative territorial entity",
      "item-requires-statement constraint: property: Pubs Galore ID; constraint status: suggestion constraint",
      "subject type constraint: class: alcohol drinking establishment; relation: instance of",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: property scope: as main value, as reference"
    ],
    "before": [
      "label in language constraint: Wikimedia language code: en",
      "single-value constraint: no qualifiers recorded",
      "format constraint: format as a regular expression: ^[A-Z]{3}/[\\w-]+$",
      "distinct-values constraint: exception to constraint: The Travellers Rest, Curious Cow",
      "item-requires-statement constraint: exception to constraint: The Under Milk Wood, Chiquito, Cafe Rouge, Mimosa; property: CAMRA pub ID",
      "item-requires-statement constraint: item of property constraint: United Kingdom, Guernsey, Jersey, Isle of Man; property: country",
      "item-requires-statement constraint: property: located in the administrative territorial entity",
      "item-requires-statement constraint: property: Pubs Galore ID; constraint status: suggestion constraint",
      "subject type constraint: class: alcohol drinking establishment; relation: instance of",
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
    "changed_constraint_qids": [
      "Q111204896"
    ],
    "mapped_constraint_qid": null,
    "result": true,
    "step": "causality_filter",
    "violation_name": "Item P|131"
  },
  {
    "result": "Q111204896",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---

## 019. `reform_Q854997_P140_1854912995`

| Field | Value |
|---|---|
| qid | Q854997 |
| property | P140 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q21503250 |
| group_key | TBOX::P140::1854912995 |
| tbox_revision_key | TBOX::P140::1854912995 |

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
  "author": "Mormegil",
  "kind": "T_BOX",
  "property_revision_id": 1854912995,
  "property_revision_prev": 1852401454
}
```

### Violation Context

```json
{
  "report_fix_date": "2023-03-21T21:01:45",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P140",
  "report_revision_new": 1857743994,
  "report_revision_old": 1856371769,
  "report_violation_type": "Values statistics",
  "report_violation_type_normalized": "Values statistics",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Values statistics",
  "value": null,
  "value_current_2026": [
    "Q748"
  ],
  "value_current_2026_descriptions_en": [
    "Indian religion"
  ],
  "value_current_2026_labels_en": [
    "Buddhism"
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
    "description": "religion of a person, organization or religious building, or associated with this subject",
    "label": "religion or worldview"
  },
  "qid": {
    "description": "fully ordained male Buddhist monastic",
    "label": "Buddhist monk"
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
            "Q1131696",
            "Q14897293",
            "Q16334295",
            "Q17524420",
            "Q178885",
            "Q189819",
            "Q192909",
            "Q208145",
            "Q21029893",
            "Q21070568",
            "Q2110808",
            "Q23847174",
            "Q24334685",
            "Q2627975",
            "Q27096235",
            "Q2728698",
            "Q3071477",
            "Q375011",
            "Q39614",
            "Q40953",
            "Q4164871",
            "Q43229",
            "Q47848",
            "Q5",
            "... omitted 6 items"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q30208840"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 10,
  "author": "Mormegil",
  "before_constraint_count": 10,
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
              "description_en": "ceremony or ritual of the passage which occurs when an individual leaves one group to enter another",
              "id": "Q1131696",
              "label_en": "rite of passage"
            },
            {
              "description_en": "entity that only exists in a work of fiction",
              "id": "Q14897293",
              "label_en": "fictional entity"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "topic viewed from a historical point of view",
              "id": "Q17524420",
              "label_en": "aspect of history"
            },
            {
              "description_en": "natural or supernatural god or goddess, divine being",
              "id": "Q178885",
              "label_en": "deity"
            },
            {
              "description_en": "activities performed according to a set sequence",
              "id": "Q189819",
              "label_en": "ritual"
            },
            {
              "description_en": "transgression or alleged transgression resulting in public outrage",
              "id": "Q192909",
              "label_en": "scandal"
            },
            {
              "description_en": "use of symbols, themes, and subject matter in the visual arts",
              "id": "Q208145",
              "label_en": "iconography"
            },
            {
              "description_en": "object used in a religion",
              "id": "Q21029893",
              "label_en": "religious object"
            },
            {
              "description_en": "human who is hypothesized to exist, but where evidence is not conclusive",
              "id": "Q21070568",
              "label_en": "human whose existence is disputed"
            },
            {
              "description_en": "behaviour motivated by religious belief",
              "id": "Q2110808",
              "label_en": "religious behaviour"
            },
            {
              "description_en": "abstract object associated with religion",
              "id": "Q23847174",
              "label_en": "religious concept"
            },
            {
              "description_en": "entity that only exists in myth, legends and folklore",
              "id": "Q24334685",
              "label_en": "mythical entity"
            },
            {
              "description_en": "event of ritual significance, performed on a special occasion",
              "id": "Q2627975",
              "label_en": "ceremony"
            },
            {
              "description_en": "non-natural geographic entities such as settlements, infrastructure, and excavations",
              "id": "Q27096235",
              "label_en": "artificial geographic entity"
            },
            {
              "description_en": "idea or tenet that is part of a faith; refers to attitudes towards mythological, supernatural, or spiritual aspects of a religion; is usually codified",
              "id": "Q2728698",
              "label_en": "religious belief"
            },
            {
              "description_en": "type of believer",
              "id": "Q3071477",
              "label_en": "religious adherent"
            },
            {
              "description_en": "time of special importance marked by adherents of some religion",
              "id": "Q375011",
              "label_en": "religious holiday"
            },
            {
              "description_en": "place of burial",
              "id": "Q39614",
              "label_en": "cemetery"
            },
            {
              "description_en": "invocation or act that seeks to activate a rapport with a deity",
              "id": "Q40953",
              "label_en": "prayer"
            },
            {
              "description_en": "social role with a set of powers and responsibilities within an organization",
              "id": "Q4164871",
              "label_en": "position"
            },
            {
              "description_en": "social entity established to meet needs or pursue goals",
              "id": "Q43229",
              "label_en": "organization"
            },
            {
              "description_en": "architectural practices used in places of worship",
              "id": "Q47848",
              "label_en": "sacred architecture"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            "... omitted 6 items"
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
              "description_en": "branch of Protestantism",
              "id": "Q101849",
              "label_en": "Reformed Christianity"
            },
            {
              "description_en": "branch of Protestant Christianity using presbyterian church government and originating in the British Isles",
              "id": "Q178169",
              "label_en": "Presbyterianism"
            },
            {
              "description_en": "Christian doctrine professed by the Catholic Church",
              "id": "Q1841",
              "label_en": "Catholicism"
            },
            {
              "description_en": "branch of Shi'a Islam",
              "id": "Q230386",
              "label_en": "Isma'ilism"
            },
            {
              "description_en": "division within Christianity, originating from the Reformation in the 16th century against the Catholic Church, that rejects the Catholic doctrines of papal supremacy and sacraments",
              "id": "Q23540",
              "label_en": "Protestantism"
            },
            {
              "description_en": "group of historically related denominations of Protestant Christianity",
              "id": "Q33203",
              "label_en": "Methodism"
            },
            {
              "description_en": "Christian denominational family",
              "id": "Q3333484",
              "label_en": "Eastern Orthodoxy"
            },
            {
              "description_en": "monotheistic Abrahamic religion founded by Muhammad",
              "id": "Q432",
              "label_en": "Islam"
            },
            {
              "description_en": "most populous Islamic denomination",
              "id": "Q483654",
              "label_en": "Sunni Islam"
            },
            {
              "description_en": "absence, indifference to, or rejection of religion",
              "id": "Q58721",
              "label_en": "irreligion"
            },
            {
              "description_en": "christian tradition developing out of the practices, liturgy and identity of the Church of England",
              "id": "Q6423963",
              "label_en": "Anglicanism"
            },
            {
              "description_en": "Indian religion",
              "id": "Q748",
              "label_en": "Buddhism"
            },
            {
              "description_en": "form of Protestantism commonly associated with the teachings of Martin Luther",
              "id": "Q75809",
              "label_en": "Lutheranism"
            },
            {
              "description_en": "religion widely practiced in the Indian subcontinent",
              "id": "Q9089",
              "label_en": "Hinduism"
            },
            {
              "description_en": "Indian religion",
              "id": "Q9232",
              "label_en": "Jainism"
            },
            {
              "description_en": "Abrahamic monotheistic ethnic religion of the Jews",
              "id": "Q9268",
              "label_en": "Judaism"
            },
            {
              "description_en": "monotheistic Indian religion",
              "id": "Q9316",
              "label_en": "Sikhism"
            },
            {
              "description_en": "evangelical Christian movement",
              "id": "Q93191",
              "label_en": "Baptists"
            },
            {
              "description_en": "second-most populous Islamic denomination",
              "id": "Q9585",
              "label_en": "Shia Islam"
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
              "description_en": "organization that supports the practice of a religion",
              "id": "Q1530022",
              "label_en": "religious organization"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "religion which only exists in a work of fiction",
              "id": "Q17364638",
              "label_en": "fictional religion"
            },
            {
              "description_en": "aggregate of patterned social arrangements in society",
              "id": "Q211606",
              "label_en": "social structure"
            },
            {
              "description_en": "type of identity create by a type of religious belief",
              "id": "Q4392985",
              "label_en": "religious identity"
            },
            {
              "description_en": "religion or world view of a person, organization or religious building",
              "id": "Q71966963",
              "label_en": "religion or world view"
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
              "description_en": "Wikibase entity type for lexicographic senses",
              "id": "Q54285715",
              "label_en": "Wikibase sense"
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
              "description_en": "Islamic school of Jurisprudence (Madhhab)",
              "id": "Q140592",
              "label_en": "Ẓāhirī"
            },
            {
              "description_en": "one of the four major schools of Sunni Islamic jurisprudence",
              "id": "Q228986",
              "label_en": "Hanafism"
            },
            {
              "description_en": "one of the schools or madhabs of Fiqh or religious law within Sunni Islam",
              "id": "Q233387",
              "label_en": "Hanbalism"
            },
            {
              "description_en": "school of Islam",
              "id": "Q243551",
              "label_en": "Ibadi Islam"
            },
            {
              "description_en": "Arab Muslim jurist of the Shafi'i school",
              "id": "Q335635",
              "label_en": "Al-Mawardi"
            },
            {
              "description_en": "one of four major schools of madhhab of Islamic jurisprudence within Sunni Islam",
              "id": "Q48221",
              "label_en": "Malikism"
            },
            {
              "description_en": "school of jurisprudence (fiqh) in Twelver and Ismaili Shia Islam",
              "id": "Q685567",
              "label_en": "Ja'fari School"
            },
            {
              "description_en": "school of Islamic jurisprudence",
              "id": "Q82245",
              "label_en": "Shafi'i"
            }
          ],
          "P6824": [
            {
              "description_en": "Islamic school of thought within Fiqh",
              "id": "P9929",
              "label_en": "madhhab"
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
              "description_en": "adherent of Sunni Islam",
              "id": "Q71986449",
              "label_en": "Sunni Muslim"
            }
          ],
          "P9729": [
            {
              "description_en": "most populous Islamic denomination",
              "id": "Q483654",
              "label_en": "Sunni Islam"
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
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: specifies that a property must have at least one reference",
          "id": "Q54554025",
          "label_en": "citation-needed constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "building for Christian worship",
              "id": "Q16970",
              "label_en": "church building"
            },
            {
              "description_en": "activity carried out by Christians to express or spread their faith",
              "id": "Q1729207",
              "label_en": "Christian ministry"
            },
            {
              "description_en": "place of worship for followers of Islam",
              "id": "Q32815",
              "label_en": "mosque"
            },
            {
              "description_en": "place of worship in Sikhism",
              "id": "Q337986",
              "label_en": "gurdwara"
            },
            {
              "description_en": "religious building at the center of a local organization where a Jewish (or rarely Samaritan) community gathers for prayer, education, social assistance or ceremonies",
              "id": "Q34627",
              "label_en": "synagogue"
            },
            {
              "description_en": "place of worship for Buddhists",
              "id": "Q5393308",
              "label_en": "Buddhist temple"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced lexeme should have a given lexical category",
          "id": "Q55819078",
          "label_en": "lexeme requires lexical category constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "noun (or noun phrase) that in its primary application refers to a unique entity or instance; the first letter is capitalized in difference to a common noun",
              "id": "Q147276",
              "label_en": "proper noun"
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
              "description_en": "ceremony or ritual of the passage which occurs when an individual leaves one group to enter another",
              "id": "Q1131696",
              "label_en": "rite of passage"
            },
            {
              "description_en": "entity that only exists in a work of fiction",
              "id": "Q14897293",
              "label_en": "fictional entity"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "topic viewed from a historical point of view",
              "id": "Q17524420",
              "label_en": "aspect of history"
            },
            {
              "description_en": "natural or supernatural god or goddess, divine being",
              "id": "Q178885",
              "label_en": "deity"
            },
            {
              "description_en": "activities performed according to a set sequence",
              "id": "Q189819",
              "label_en": "ritual"
            },
            {
              "description_en": "transgression or alleged transgression resulting in public outrage",
              "id": "Q192909",
              "label_en": "scandal"
            },
            {
              "description_en": "use of symbols, themes, and subject matter in the visual arts",
              "id": "Q208145",
              "label_en": "iconography"
            },
            {
              "description_en": "object used in a religion",
              "id": "Q21029893",
              "label_en": "religious object"
            },
            {
              "description_en": "human who is hypothesized to exist, but where evidence is not conclusive",
              "id": "Q21070568",
              "label_en": "human whose existence is disputed"
            },
            {
              "description_en": "behaviour motivated by religious belief",
              "id": "Q2110808",
              "label_en": "religious behaviour"
            },
            {
              "description_en": "abstract object associated with religion",
              "id": "Q23847174",
              "label_en": "religious concept"
            },
            {
              "description_en": "entity that only exists in myth, legends and folklore",
              "id": "Q24334685",
              "label_en": "mythical entity"
            },
            {
              "description_en": "event of ritual significance, performed on a special occasion",
              "id": "Q2627975",
              "label_en": "ceremony"
            },
            {
              "description_en": "non-natural geographic entities such as settlements, infrastructure, and excavations",
              "id": "Q27096235",
              "label_en": "artificial geographic entity"
            },
            {
              "description_en": "idea or tenet that is part of a faith; refers to attitudes towards mythological, supernatural, or spiritual aspects of a religion; is usually codified",
              "id": "Q2728698",
              "label_en": "religious belief"
            },
            {
              "description_en": "type of believer",
              "id": "Q3071477",
              "label_en": "religious adherent"
            },
            {
              "description_en": "time of special importance marked by adherents of some religion",
              "id": "Q375011",
              "label_en": "religious holiday"
            },
            {
              "description_en": "place of burial",
              "id": "Q39614",
              "label_en": "cemetery"
            },
            {
              "description_en": "invocation or act that seeks to activate a rapport with a deity",
              "id": "Q40953",
              "label_en": "prayer"
            },
            {
              "description_en": "social role with a set of powers and responsibilities within an organization",
              "id": "Q4164871",
              "label_en": "position"
            },
            {
              "description_en": "social entity established to meet needs or pursue goals",
              "id": "Q43229",
              "label_en": "organization"
            },
            {
              "description_en": "architectural practices used in places of worship",
              "id": "Q47848",
              "label_en": "sacred architecture"
            },
            {
              "description_en": "any single member of Homo sapiens, unique extant species of the genus Homo",
              "id": "Q5",
              "label_en": "human"
            },
            "... omitted 6 items"
          ],
          "P2309": [
            {
              "description_en": "relation of type or value type constraint",
              "id": "Q30208840",
              "label_en": "instance or subclass of"
            }
          ],
          "P4680": [
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
              "description_en": "branch of Protestantism",
              "id": "Q101849",
              "label_en": "Reformed Christianity"
            },
            {
              "description_en": "branch of Protestant Christianity using presbyterian church government and originating in the British Isles",
              "id": "Q178169",
              "label_en": "Presbyterianism"
            },
            {
              "description_en": "Christian doctrine professed by the Catholic Church",
              "id": "Q1841",
              "label_en": "Catholicism"
            },
            {
              "description_en": "branch of Shi'a Islam",
              "id": "Q230386",
              "label_en": "Isma'ilism"
            },
            {
              "description_en": "division within Christianity, originating from the Reformation in the 16th century against the Catholic Church, that rejects the Catholic doctrines of papal supremacy and sacraments",
              "id": "Q23540",
              "label_en": "Protestantism"
            },
            {
              "description_en": "group of historically related denominations of Protestant Christianity",
              "id": "Q33203",
              "label_en": "Methodism"
            },
            {
              "description_en": "Christian denominational family",
              "id": "Q3333484",
              "label_en": "Eastern Orthodoxy"
            },
            {
              "description_en": "monotheistic Abrahamic religion founded by Muhammad",
              "id": "Q432",
              "label_en": "Islam"
            },
            {
              "description_en": "most populous Islamic denomination",
              "id": "Q483654",
              "label_en": "Sunni Islam"
            },
            {
              "description_en": "absence, indifference to, or rejection of religion",
              "id": "Q58721",
              "label_en": "irreligion"
            },
            {
              "description_en": "christian tradition developing out of the practices, liturgy and identity of the Church of England",
              "id": "Q6423963",
              "label_en": "Anglicanism"
            },
            {
              "description_en": "Indian religion",
              "id": "Q748",
              "label_en": "Buddhism"
            },
            {
              "description_en": "form of Protestantism commonly associated with the teachings of Martin Luther",
              "id": "Q75809",
              "label_en": "Lutheranism"
            },
            {
              "description_en": "religion widely practiced in the Indian subcontinent",
              "id": "Q9089",
              "label_en": "Hinduism"
            },
            {
              "description_en": "Indian religion",
              "id": "Q9232",
              "label_en": "Jainism"
            },
            {
              "description_en": "Abrahamic monotheistic ethnic religion of the Jews",
              "id": "Q9268",
              "label_en": "Judaism"
            },
            {
              "description_en": "monotheistic Indian religion",
              "id": "Q9316",
              "label_en": "Sikhism"
            },
            {
              "description_en": "evangelical Christian movement",
              "id": "Q93191",
              "label_en": "Baptists"
            },
            {
              "description_en": "second-most populous Islamic denomination",
              "id": "Q9585",
              "label_en": "Shia Islam"
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
              "description_en": "organization that supports the practice of a religion",
              "id": "Q1530022",
              "label_en": "religious organization"
            },
            {
              "description_en": "any set of human beings",
              "id": "Q16334295",
              "label_en": "group of humans"
            },
            {
              "description_en": "religion which only exists in a work of fiction",
              "id": "Q17364638",
              "label_en": "fictional religion"
            },
            {
              "description_en": "aggregate of patterned social arrangements in society",
              "id": "Q211606",
              "label_en": "social structure"
            },
            {
              "description_en": "type of identity create by a type of religious belief",
              "id": "Q4392985",
              "label_en": "religious identity"
            },
            {
              "description_en": "religion or world view of a person, organization or religious building",
              "id": "Q71966963",
              "label_en": "religion or world view"
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
              "description_en": "Wikibase entity type for lexicographic senses",
              "id": "Q54285715",
              "label_en": "Wikibase sense"
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
              "description_en": "Islamic school of Jurisprudence (Madhhab)",
              "id": "Q140592",
              "label_en": "Ẓāhirī"
            },
            {
              "description_en": "one of the four major schools of Sunni Islamic jurisprudence",
              "id": "Q228986",
              "label_en": "Hanafism"
            },
            {
              "description_en": "one of the schools or madhabs of Fiqh or religious law within Sunni Islam",
              "id": "Q233387",
              "label_en": "Hanbalism"
            },
            {
              "description_en": "school of Islam",
              "id": "Q243551",
              "label_en": "Ibadi Islam"
            },
            {
              "description_en": "Arab Muslim jurist of the Shafi'i school",
              "id": "Q335635",
              "label_en": "Al-Mawardi"
            },
            {
              "description_en": "one of four major schools of madhhab of Islamic jurisprudence within Sunni Islam",
              "id": "Q48221",
              "label_en": "Malikism"
            },
            {
              "description_en": "school of jurisprudence (fiqh) in Twelver and Ismaili Shia Islam",
              "id": "Q685567",
              "label_en": "Ja'fari School"
            },
            {
              "description_en": "school of Islamic jurisprudence",
              "id": "Q82245",
              "label_en": "Shafi'i"
            }
          ],
          "P6824": [
            {
              "description_en": "Islamic school of thought within Fiqh",
              "id": "P9929",
              "label_en": "madhhab"
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
              "description_en": "adherent of Sunni Islam",
              "id": "Q71986449",
              "label_en": "Sunni Muslim"
            }
          ],
          "P9729": [
            {
              "description_en": "most populous Islamic denomination",
              "id": "Q483654",
              "label_en": "Sunni Islam"
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
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: specifies that a property must have at least one reference",
          "id": "Q54554025",
          "label_en": "citation-needed constraint"
        },
        "parameters": {
          "P2303": [
            {
              "description_en": "building for Christian worship",
              "id": "Q16970",
              "label_en": "church building"
            },
            {
              "description_en": "activity carried out by Christians to express or spread their faith",
              "id": "Q1729207",
              "label_en": "Christian ministry"
            },
            {
              "description_en": "place of worship for followers of Islam",
              "id": "Q32815",
              "label_en": "mosque"
            },
            {
              "description_en": "place of worship in Sikhism",
              "id": "Q337986",
              "label_en": "gurdwara"
            },
            {
              "description_en": "religious building at the center of a local organization where a Jewish (or rarely Samaritan) community gathers for prayer, education, social assistance or ceremonies",
              "id": "Q34627",
              "label_en": "synagogue"
            },
            {
              "description_en": "place of worship for Buddhists",
              "id": "Q5393308",
              "label_en": "Buddhist temple"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      },
      {
        "constraint_type": {
          "description_en": "type of constraint for Wikidata properties: used to specify that the referenced lexeme should have a given lexical category",
          "id": "Q55819078",
          "label_en": "lexeme requires lexical category constraint"
        },
        "parameters": {
          "P2305": [
            {
              "description_en": "noun (or noun phrase) that in its primary application refers to a unique entity or instance; the first letter is capitalized in difference to a common noun",
              "id": "Q147276",
              "label_en": "proper noun"
            }
          ]
        },
        "rank": "normal",
        "snaktype": "VALUE"
      }
    ]
  },
  "hash_after": "6a46f694bf5f39072848590d728fafac04322de9",
  "hash_before": "42cd235993d1b17882f095d2f662de95b68fc38e",
  "property_revision_id": 1854912995,
  "property_revision_prev": 1852401454,
  "qualifier_value_changes": [
    {
      "added_values": [],
      "constraint_qid": "Q21503250",
      "qualifier_property": "P4680",
      "removed_values": [
        "Q29934200"
      ],
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
            "Q1131696",
            "Q14897293",
            "Q16334295",
            "Q17524420",
            "Q178885",
            "Q189819",
            "Q192909",
            "Q208145",
            "Q21029893",
            "Q21070568",
            "Q2110808",
            "Q23847174",
            "Q24334685",
            "Q2627975",
            "Q27096235",
            "Q2728698",
            "Q3071477",
            "Q375011",
            "Q39614",
            "Q40953",
            "Q4164871",
            "Q43229",
            "Q47848",
            "Q5",
            "... omitted 6 items"
          ]
        },
        {
          "property_id": "P2309",
          "values": [
            "Q30208840"
          ]
        },
        {
          "property_id": "P4680",
          "values": [
            "Q29934200"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "subject type constraint: class: rite of passage, fictional entity, group of humans, aspect of history, deity, ritual, scandal, iconography, religious object, human whose existence is disputed, religious behaviour, religious concept, mythical entity, ceremony, artificial geographic entity, religious belief, religious adherent, religious holiday, cemetery, prayer, position, organization, sacred architecture, human, solar deity, religious symbol, rite, religious occu... [truncated 82 chars]",
      "one-of constraint: reason for deprecated rank: constraint provides suggestions for manual input; item of property constraint: Reformed Christianity, Presbyterianism, Catholicism, Isma'ilism, Protestantism, Methodism, Eastern Orthodoxy, Islam, Sunni Islam, irreligion, Anglicanism, Buddhism, Lutheranism, Hinduism, Jainism, Judaism, Sikhism, Baptists, Shia Islam",
      "value-type constraint: class: religious organization, group of humans, fictional religion, social structure, religious identity, religion or world view; relation: instance or subclass of",
      "contemporary constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, Wikibase sense",
      "none-of constraint: item of property constraint: Ẓāhirī, Hanafism, Hanbalism, Ibadi Islam, Al-Mawardi, Malikism, Ja'fari School, Shafi'i; replacement property: madhhab",
      "none-of constraint: item of property constraint: Sunni Muslim; replacement value: Sunni Islam",
      "property scope constraint: property scope: as main value, as qualifier",
      "citation-needed constraint: exception to constraint: church building, Christian ministry, mosque, gurdwara, synagogue, Buddhist temple",
      "lexeme requires lexical category constraint: item of property constraint: proper noun"
    ],
    "before": [
      "subject type constraint: class: rite of passage, fictional entity, group of humans, aspect of history, deity, ritual, scandal, iconography, religious object, human whose existence is disputed, religious behaviour, religious concept, mythical entity, ceremony, artificial geographic entity, religious belief, religious adherent, religious holiday, cemetery, prayer, position, organization, sacred architecture, human, solar deity, religious symbol, rite, religious occu... [truncated 115 chars]",
      "one-of constraint: reason for deprecated rank: constraint provides suggestions for manual input; item of property constraint: Reformed Christianity, Presbyterianism, Catholicism, Isma'ilism, Protestantism, Methodism, Eastern Orthodoxy, Islam, Sunni Islam, irreligion, Anglicanism, Buddhism, Lutheranism, Hinduism, Jainism, Judaism, Sikhism, Baptists, Shia Islam",
      "value-type constraint: class: religious organization, group of humans, fictional religion, social structure, religious identity, religion or world view; relation: instance or subclass of",
      "contemporary constraint: no qualifiers recorded",
      "allowed-entity-types constraint: item of property constraint: Wikibase item, Wikibase sense",
      "none-of constraint: item of property constraint: Ẓāhirī, Hanafism, Hanbalism, Ibadi Islam, Al-Mawardi, Malikism, Ja'fari School, Shafi'i; replacement property: madhhab",
      "none-of constraint: item of property constraint: Sunni Muslim; replacement value: Sunni Islam",
      "property scope constraint: property scope: as main value, as qualifier",
      "citation-needed constraint: exception to constraint: church building, Christian ministry, mosque, gurdwara, synagogue, Buddhist temple",
      "lexeme requires lexical category constraint: item of property constraint: proper noun"
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
    "violation_name": "Values statistics"
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
    "result": "SCHEMA_UPDATE",
    "step": "set_semantics"
  }
]
```

---

## 020. `reform_Q874154_P691_2445073477`

| Field | Value |
|---|---|
| qid | Q874154 |
| property | P691 |
| track | T_BOX |
| class / subtype / confidence | T_BOX / SCHEMA_UPDATE / medium |
| main_score / diagnostic_only | True / False |
| analysis_slice | main_tbox_schema_update |
| popularity_bucket | head |
| constraint_family | Q21502404 |
| group_key | TBOX::P691::2445073477 |
| tbox_revision_key | TBOX::P691::2445073477 |

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
  "author": "Mahir256",
  "kind": "T_BOX",
  "property_revision_id": 2445073477,
  "property_revision_prev": 2437675608
}
```

### Violation Context

```json
{
  "report_fix_date": "2025-12-27T11:31:49",
  "report_page_title": "Wikidata:Database reports/Constraint violations/P691",
  "report_revision_new": 2447752734,
  "report_revision_old": 2447355891,
  "report_violation_type": "Label in cs language",
  "report_violation_type_normalized": "Label in cs language",
  "report_violation_type_qids": [],
  "report_violation_type_raw": "Label in cs language",
  "value": null,
  "value_current_2026": [
    "pna2009514419"
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
    "description": "identifier in the Czech National Authority Database of National Library of the Czech Republic (NL CR)",
    "label": "NL CR AUT ID"
  },
  "qid": {
    "description": "ballet company",
    "label": "Hamburg Ballet"
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
    "label_en": "conflicts-with constraint",
    "qid": "Q21502838"
  },
  {
    "label_en": "single-value constraint",
    "qid": "Q19474404"
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
      "constraint_qid": "Q21510851",
      "qualifiers": [
        {
          "property_id": "P2306",
          "values": [
            "P1319",
            "P1326",
            "P1810",
            "P21",
            "P2241",
            "P2868",
            "P407",
            "P4900",
            "P4970",
            "P518",
            "P580",
            "P582",
            "P742",
            "P7452",
            "P8327",
            "P8554",
            "P8555",
            "P9570"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "after_constraint_count": 8,
  "author": "Mahir256",
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
              "value": "cs"
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
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
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
              "value": "[a-z]{2,4}[0-9]{2,14}"
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
              "description_en": "this item duplicates another item, it can be merged once the necessary merges are done in other Wikimedia projects",
              "id": "Q17362920",
              "label_en": "Wikimedia duplicated page"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
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
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
              "id": "P21",
              "label_en": "sex or gender"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
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
              "description_en": "qualifier to indicate a broader concept that the present item is part of, as mapped by an external source. The statement being qualified should be an exact match.",
              "id": "P4900",
              "label_en": "broader concept"
            },
            {
              "description_en": "qualifier for alternative name(s), given for a subject in a database entry, or preserved in references (even these are no longer the preferred name)",
              "id": "P4970",
              "label_en": "alternative name"
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
              "description_en": "alias used by someone (for nickname use P1449)",
              "id": "P742",
              "label_en": "pseudonym"
            },
            {
              "description_en": "qualifier to allow the reason to be indicated why a particular statement should be considered preferred",
              "id": "P7452",
              "label_en": "reason for preferred rank"
            },
            {
              "description_en": "(qualifier) this statement is deprecated as it is actually about another subject",
              "id": "P8327",
              "label_en": "intended subject of deprecated statement"
            },
            {
              "description_en": "(qualifier) earliest date on which the statement could have begun to no longer be true",
              "id": "P8554",
              "label_en": "earliest end date"
            },
            {
              "description_en": "(qualifier) latest date on which the statement could have started to be true",
              "id": "P8555",
              "label_en": "latest start date"
            },
            {
              "description_en": "note that describes what a term means in the context of a controlled vocabulary; use as a qualifier for identifier statements when a controlled vocabulary has a scope note",
              "id": "P9570",
              "label_en": "scope note"
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
              "value": "cs"
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
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
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
              "value": "[a-z]{2,4}[0-9]{2,14}"
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
              "description_en": "this item duplicates another item, it can be merged once the necessary merges are done in other Wikimedia projects",
              "id": "Q17362920",
              "label_en": "Wikimedia duplicated page"
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
          "description_en": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "id": "Q21510851",
          "label_en": "allowed qualifiers constraint"
        },
        "parameters": {
          "P2306": [
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
              "description_en": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
              "id": "P1810",
              "label_en": "subject named as"
            },
            {
              "description_en": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
              "id": "P21",
              "label_en": "sex or gender"
            },
            {
              "description_en": "qualifier to indicate why a particular statement should have deprecated rank",
              "id": "P2241",
              "label_en": "reason for deprecated rank"
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
              "description_en": "qualifier for alternative name(s), given for a subject in a database entry, or preserved in references (even these are no longer the preferred name)",
              "id": "P4970",
              "label_en": "alternative name"
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
              "description_en": "alias used by someone (for nickname use P1449)",
              "id": "P742",
              "label_en": "pseudonym"
            },
            {
              "description_en": "qualifier to allow the reason to be indicated why a particular statement should be considered preferred",
              "id": "P7452",
              "label_en": "reason for preferred rank"
            },
            {
              "description_en": "(qualifier) this statement is deprecated as it is actually about another subject",
              "id": "P8327",
              "label_en": "intended subject of deprecated statement"
            },
            {
              "description_en": "(qualifier) earliest date on which the statement could have begun to no longer be true",
              "id": "P8554",
              "label_en": "earliest end date"
            },
            {
              "description_en": "(qualifier) latest date on which the statement could have started to be true",
              "id": "P8555",
              "label_en": "latest start date"
            },
            {
              "description_en": "note that describes what a term means in the context of a controlled vocabulary; use as a qualifier for identifier statements when a controlled vocabulary has a scope note",
              "id": "P9570",
              "label_en": "scope note"
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
  "hash_after": "ea9a73a70384e60db1bf18f0aaef2889ee16b2b3",
  "hash_before": "ce2f07aaa93e3a38bd7317f8fc79ac642eefe1d3",
  "property_revision_id": 2445073477,
  "property_revision_prev": 2437675608,
  "qualifier_value_changes": [
    {
      "added_values": [
        "P4900"
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
            "P1319",
            "P1326",
            "P1810",
            "P21",
            "P2241",
            "P2868",
            "P407",
            "P4970",
            "P518",
            "P580",
            "P582",
            "P742",
            "P7452",
            "P8327",
            "P8554",
            "P8555",
            "P9570"
          ]
        }
      ],
      "rank": "normal",
      "snaktype": "VALUE"
    }
  ],
  "rule_summaries_en": {
    "after": [
      "label in language constraint: constraint status: suggestion constraint; Wikimedia language code: cs",
      "single-value constraint: separator: subject named as, reason for deprecated rank, subject has role, language of work or name, applies to part",
      "format constraint: format as a regular expression: [a-z]{2,4}[0-9]{2,14}; constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "conflicts-with constraint: item of property constraint: Wikimedia list article, Wikimedia duplicated page, Wikimedia disambiguation page, Wikimedia category; property: instance of; constraint status: mandatory constraint",
      "allowed qualifiers constraint: property: earliest date, latest date, subject named as, sex or gender, reason for deprecated rank, subject has role, language of work or name, broader concept, alternative name, applies to part, start time, end time, pseudonym, reason for preferred rank, intended subject of deprecated statement, earliest end date, latest start date, scope note",
      "allowed-entity-types constraint: item of property constraint: Wikibase item",
      "property scope constraint: constraint status: mandatory constraint; property scope: as main value, as reference"
    ],
    "before": [
      "label in language constraint: constraint status: suggestion constraint; Wikimedia language code: cs",
      "single-value constraint: separator: subject named as, reason for deprecated rank, subject has role, language of work or name, applies to part",
      "format constraint: format as a regular expression: [a-z]{2,4}[0-9]{2,14}; constraint status: mandatory constraint",
      "distinct-values constraint: no qualifiers recorded",
      "conflicts-with constraint: item of property constraint: Wikimedia list article, Wikimedia duplicated page, Wikimedia disambiguation page, Wikimedia category; property: instance of; constraint status: mandatory constraint",
      "allowed qualifiers constraint: property: earliest date, latest date, subject named as, sex or gender, reason for deprecated rank, subject has role, language of work or name, alternative name, applies to part, start time, end time, pseudonym, reason for preferred rank, intended subject of deprecated statement, earliest end date, latest start date, scope note",
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
    "violation_name": "Label in cs language"
  },
  {
    "result": "Q108139345",
    "step": "target_constraint"
  },
  {
    "result": "SCHEMA_UPDATE",
    "step": "generic_set_semantics"
  }
]
```

---
