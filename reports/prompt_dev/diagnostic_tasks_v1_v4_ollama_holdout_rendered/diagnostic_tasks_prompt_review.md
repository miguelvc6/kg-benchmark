# Diagnostic Tasks v1 Render Review

No model inference was run.

Rendered prompts: `768`

## a_box_value_extraction / case_000001

- Context: `logic_only`
- Track: `A_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: a_box_value_extraction

Extract only the candidate final A-box target values visible in the prompt evidence.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","visible_final_values":["..."],"visible_removed_values":["..."],"evidence_paths":["..."],"answerability":"visible|not_visible|deterministic_transform"}

Input case JSON:

{
  "id": "case_000001",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "Poet, Writer and Historian",
      "label": "Sriramoju Haragopal"
    }
  },
  "logic_context": {
    "constraints": [
      {
        "constraint_type": {
          "label": "property scope constraint",
          "qid": "Q53869507"
        },
        "qualifiers": [
          {
            "property_id": "P5314",
            "property_label": "property scope",
            "values": [
              {
                "label": "as main value",
                "qid": "Q54828448",
                "raw": "Q54828448"
              },
              {
                "label": "as qualifier",
                "qid": "Q54828449",
                "raw": "Q54828449"
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449)"
      },
      {
        "constraint_type": {
          "label": "allowed-entity-types constraint",
          "qid": "Q52004125"
        },
        "qualifiers": [
          {
            "property_id": "P2305",
            "property_label": "item of property constraint",
            "values": [
              {
                "label": "Wikibase item",
                "qid": "Q29934200",
                "raw": "Q29934200"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200)"
      }
    ],
    "property_id": "P131"
  },
  "property": "P131",
  "qid": "Q20563062",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
    "report_violation_type": "Conflicts with P|31",
    "report_violation_type_normalized": "Conflicts with P|31",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Conflicts with P|31",
    "value": [
      "Q677037",
      "Q59259864"
    ],
    "value_descriptions_en": [
      "state in southern India",
      "mandalam in Yadadri Bhuvanagiri district, Telangana"
    ],
    "value_labels_en": [
      "Telangana",
      "Alair Mandal"
    ]
  }
}
```

## a_box_operation_selection / case_000001

- Context: `logic_only`
- Track: `A_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: a_box_operation_selection

Choose the operation type supported by visible evidence, without choosing replacement values.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","operation":"SET|ADD|REMOVE|DELETE_ALL|ABSTAIN","preserve_values":["..."],"remove_values":["..."],"rationale":"..."}

Input case JSON:

{
  "id": "case_000001",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "Poet, Writer and Historian",
      "label": "Sriramoju Haragopal"
    }
  },
  "logic_context": {
    "constraints": [
      {
        "constraint_type": {
          "label": "property scope constraint",
          "qid": "Q53869507"
        },
        "qualifiers": [
          {
            "property_id": "P5314",
            "property_label": "property scope",
            "values": [
              {
                "label": "as main value",
                "qid": "Q54828448",
                "raw": "Q54828448"
              },
              {
                "label": "as qualifier",
                "qid": "Q54828449",
                "raw": "Q54828449"
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449)"
      },
      {
        "constraint_type": {
          "label": "allowed-entity-types constraint",
          "qid": "Q52004125"
        },
        "qualifiers": [
          {
            "property_id": "P2305",
            "property_label": "item of property constraint",
            "values": [
              {
                "label": "Wikibase item",
                "qid": "Q29934200",
                "raw": "Q29934200"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200)"
      }
    ],
    "property_id": "P131"
  },
  "property": "P131",
  "qid": "Q20563062",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
    "report_violation_type": "Conflicts with P|31",
    "report_violation_type_normalized": "Conflicts with P|31",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Conflicts with P|31",
    "value": [
      "Q677037",
      "Q59259864"
    ],
    "value_descriptions_en": [
      "state in southern India",
      "mandalam in Yadadri Bhuvanagiri district, Telangana"
    ],
    "value_labels_en": [
      "Telangana",
      "Alair Mandal"
    ]
  }
}
```

## a_box_answerability / case_000001

- Context: `logic_only`
- Track: `A_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: a_box_answerability

Decide whether an exact A-box repair is answerable from visible evidence.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","answerability":"exact_repair_visible|conservative_remove_only|insufficient_visible_evidence","missing_evidence":["..."],"rationale":"..."}

Input case JSON:

{
  "id": "case_000001",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "Poet, Writer and Historian",
      "label": "Sriramoju Haragopal"
    }
  },
  "logic_context": {
    "constraints": [
      {
        "constraint_type": {
          "label": "property scope constraint",
          "qid": "Q53869507"
        },
        "qualifiers": [
          {
            "property_id": "P5314",
            "property_label": "property scope",
            "values": [
              {
                "label": "as main value",
                "qid": "Q54828448",
                "raw": "Q54828448"
              },
              {
                "label": "as qualifier",
                "qid": "Q54828449",
                "raw": "Q54828449"
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449)"
      },
      {
        "constraint_type": {
          "label": "allowed-entity-types constraint",
          "qid": "Q52004125"
        },
        "qualifiers": [
          {
            "property_id": "P2305",
            "property_label": "item of property constraint",
            "values": [
              {
                "label": "Wikibase item",
                "qid": "Q29934200",
                "raw": "Q29934200"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200)"
      }
    ],
    "property_id": "P131"
  },
  "property": "P131",
  "qid": "Q20563062",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
    "report_violation_type": "Conflicts with P|31",
    "report_violation_type_normalized": "Conflicts with P|31",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Conflicts with P|31",
    "value": [
      "Q677037",
      "Q59259864"
    ],
    "value_descriptions_en": [
      "state in southern India",
      "mandalam in Yadadri Bhuvanagiri district, Telangana"
    ],
    "value_labels_en": [
      "Telangana",
      "Alair Mandal"
    ]
  }
}
```

## track_locus_contrast / case_000001

- Context: `logic_only`
- Track: `A_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: track_locus_contrast

Contrast A-box and T-box repair-locus evidence without proposing a repair.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","a_box_evidence":["..."],"t_box_evidence":["..."],"likely_locus":"A_BOX|T_BOX|AMBIGUOUS","rationale":"..."}

Input case JSON:

{
  "id": "case_000001",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "Poet, Writer and Historian",
      "label": "Sriramoju Haragopal"
    }
  },
  "logic_context": {
    "constraints": [
      {
        "constraint_type": {
          "label": "property scope constraint",
          "qid": "Q53869507"
        },
        "qualifiers": [
          {
            "property_id": "P5314",
            "property_label": "property scope",
            "values": [
              {
                "label": "as main value",
                "qid": "Q54828448",
                "raw": "Q54828448"
              },
              {
                "label": "as qualifier",
                "qid": "Q54828449",
                "raw": "Q54828449"
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449)"
      },
      {
        "constraint_type": {
          "label": "allowed-entity-types constraint",
          "qid": "Q52004125"
        },
        "qualifiers": [
          {
            "property_id": "P2305",
            "property_label": "item of property constraint",
            "values": [
              {
                "label": "Wikibase item",
                "qid": "Q29934200",
                "raw": "Q29934200"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200)"
      }
    ],
    "property_id": "P131"
  },
  "property": "P131",
  "qid": "Q20563062",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
    "report_violation_type": "Conflicts with P|31",
    "report_violation_type_normalized": "Conflicts with P|31",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Conflicts with P|31",
    "value": [
      "Q677037",
      "Q59259864"
    ],
    "value_descriptions_en": [
      "state in southern India",
      "mandalam in Yadadri Bhuvanagiri district, Telangana"
    ],
    "value_labels_en": [
      "Telangana",
      "Alair Mandal"
    ]
  }
}
```

## a_box_value_extraction / case_000001

- Context: `local_graph`
- Track: `A_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: a_box_value_extraction

Extract only the candidate final A-box target values visible in the prompt evidence.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","visible_final_values":["..."],"visible_removed_values":["..."],"evidence_paths":["..."],"answerability":"visible|not_visible|deterministic_transform"}

Input case JSON:

{
  "id": "case_000001",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "Poet, Writer and Historian",
      "label": "Sriramoju Haragopal"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "Poet, Writer and Historian",
      "label": "Sriramoju Haragopal",
      "properties": {
        "P131": [
          "Q677037",
          "Q59259864"
        ]
      },
      "qid": "Q20563062"
    },
    "L2_labels": {
      "entities": {
        "P131": {
          "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
          "label": "located in the administrative territorial entity"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q20563062": {
          "description": "Poet, Writer and Historian",
          "label": "Sriramoju Haragopal"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q52004125": {
          "description": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "label": "allowed-entity-types constraint"
        },
        "Q53869507": {
          "description": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "label": "property scope constraint"
        },
        "Q54828448": {
          "description": "property scope type",
          "label": "as main value"
        },
        "Q54828449": {
          "description": "property scope type",
          "label": "as qualifier"
        },
        "Q59259864": {
          "description": "mandalam in Yadadri Bhuvanagiri district, Telangana",
          "label": "Alair Mandal"
        },
        "Q677037": {
          "description": "state in southern India",
          "label": "Telangana"
        }
      }
    },
    "L3_neighborhood": {
      "outgoing_edges": []
    },
    "L4_constraints": {
      "constraints": [
        {
          "constraint_type": {
            "label": "property scope constraint",
            "qid": "Q53869507"
          },
          "qualifiers": [
            {
              "property_id": "P5314",
              "property_label": "property scope",
              "values": [
                {
                  "label": "as main value",
                  "qid": "Q54828448",
                  "raw": "Q54828448"
                },
                {
                  "label": "as qualifier",
                  "qid": "Q54828449",
                  "raw": "Q54828449"
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449)"
        },
        {
          "constraint_type": {
            "label": "allowed-entity-types constraint",
            "qid": "Q52004125"
          },
          "qualifiers": [
            {
              "property_id": "P2305",
              "property_label": "item of property constraint",
              "values": [
                {
                  "label": "Wikibase item",
                  "qid": "Q29934200",
                  "raw": "Q29934200"
                }
              ]
            }
          ],
          "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200)"
        }
      ],
      "property_id": "P131"
    }
  },
  "property": "P131",
  "qid": "Q20563062",
  "violation_context": {
    "report_page_title": "Wikidata:Database
```

## a_box_operation_selection / case_000001

- Context: `local_graph`
- Track: `A_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: a_box_operation_selection

Choose the operation type supported by visible evidence, without choosing replacement values.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","operation":"SET|ADD|REMOVE|DELETE_ALL|ABSTAIN","preserve_values":["..."],"remove_values":["..."],"rationale":"..."}

Input case JSON:

{
  "id": "case_000001",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "Poet, Writer and Historian",
      "label": "Sriramoju Haragopal"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "Poet, Writer and Historian",
      "label": "Sriramoju Haragopal",
      "properties": {
        "P131": [
          "Q677037",
          "Q59259864"
        ]
      },
      "qid": "Q20563062"
    },
    "L2_labels": {
      "entities": {
        "P131": {
          "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
          "label": "located in the administrative territorial entity"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q20563062": {
          "description": "Poet, Writer and Historian",
          "label": "Sriramoju Haragopal"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q52004125": {
          "description": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "label": "allowed-entity-types constraint"
        },
        "Q53869507": {
          "description": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "label": "property scope constraint"
        },
        "Q54828448": {
          "description": "property scope type",
          "label": "as main value"
        },
        "Q54828449": {
          "description": "property scope type",
          "label": "as qualifier"
        },
        "Q59259864": {
          "description": "mandalam in Yadadri Bhuvanagiri district, Telangana",
          "label": "Alair Mandal"
        },
        "Q677037": {
          "description": "state in southern India",
          "label": "Telangana"
        }
      }
    },
    "L3_neighborhood": {
      "outgoing_edges": []
    },
    "L4_constraints": {
      "constraints": [
        {
          "constraint_type": {
            "label": "property scope constraint",
            "qid": "Q53869507"
          },
          "qualifiers": [
            {
              "property_id": "P5314",
              "property_label": "property scope",
              "values": [
                {
                  "label": "as main value",
                  "qid": "Q54828448",
                  "raw": "Q54828448"
                },
                {
                  "label": "as qualifier",
                  "qid": "Q54828449",
                  "raw": "Q54828449"
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449)"
        },
        {
          "constraint_type": {
            "label": "allowed-entity-types constraint",
            "qid": "Q52004125"
          },
          "qualifiers": [
            {
              "property_id": "P2305",
              "property_label": "item of property constraint",
              "values": [
                {
                  "label": "Wikibase item",
                  "qid": "Q29934200",
                  "raw": "Q29934200"
                }
              ]
            }
          ],
          "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200)"
        }
      ],
      "property_id": "P131"
    }
  },
  "property": "P131",
  "qid": "Q20563062",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint vio
```

## a_box_answerability / case_000001

- Context: `local_graph`
- Track: `A_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: a_box_answerability

Decide whether an exact A-box repair is answerable from visible evidence.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","answerability":"exact_repair_visible|conservative_remove_only|insufficient_visible_evidence","missing_evidence":["..."],"rationale":"..."}

Input case JSON:

{
  "id": "case_000001",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "Poet, Writer and Historian",
      "label": "Sriramoju Haragopal"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "Poet, Writer and Historian",
      "label": "Sriramoju Haragopal",
      "properties": {
        "P131": [
          "Q677037",
          "Q59259864"
        ]
      },
      "qid": "Q20563062"
    },
    "L2_labels": {
      "entities": {
        "P131": {
          "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
          "label": "located in the administrative territorial entity"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q20563062": {
          "description": "Poet, Writer and Historian",
          "label": "Sriramoju Haragopal"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q52004125": {
          "description": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "label": "allowed-entity-types constraint"
        },
        "Q53869507": {
          "description": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "label": "property scope constraint"
        },
        "Q54828448": {
          "description": "property scope type",
          "label": "as main value"
        },
        "Q54828449": {
          "description": "property scope type",
          "label": "as qualifier"
        },
        "Q59259864": {
          "description": "mandalam in Yadadri Bhuvanagiri district, Telangana",
          "label": "Alair Mandal"
        },
        "Q677037": {
          "description": "state in southern India",
          "label": "Telangana"
        }
      }
    },
    "L3_neighborhood": {
      "outgoing_edges": []
    },
    "L4_constraints": {
      "constraints": [
        {
          "constraint_type": {
            "label": "property scope constraint",
            "qid": "Q53869507"
          },
          "qualifiers": [
            {
              "property_id": "P5314",
              "property_label": "property scope",
              "values": [
                {
                  "label": "as main value",
                  "qid": "Q54828448",
                  "raw": "Q54828448"
                },
                {
                  "label": "as qualifier",
                  "qid": "Q54828449",
                  "raw": "Q54828449"
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449)"
        },
        {
          "constraint_type": {
            "label": "allowed-entity-types constraint",
            "qid": "Q52004125"
          },
          "qualifiers": [
            {
              "property_id": "P2305",
              "property_label": "item of property constraint",
              "values": [
                {
                  "label": "Wikibase item",
                  "qid": "Q29934200",
                  "raw": "Q29934200"
                }
              ]
            }
          ],
          "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200)"
        }
      ],
      "property_id": "P131"
    }
  },
  "property": "P131",
  "qid": "Q20563062",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violat
```

## track_locus_contrast / case_000001

- Context: `local_graph`
- Track: `A_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: track_locus_contrast

Contrast A-box and T-box repair-locus evidence without proposing a repair.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","a_box_evidence":["..."],"t_box_evidence":["..."],"likely_locus":"A_BOX|T_BOX|AMBIGUOUS","rationale":"..."}

Input case JSON:

{
  "id": "case_000001",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "Poet, Writer and Historian",
      "label": "Sriramoju Haragopal"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "Poet, Writer and Historian",
      "label": "Sriramoju Haragopal",
      "properties": {
        "P131": [
          "Q677037",
          "Q59259864"
        ]
      },
      "qid": "Q20563062"
    },
    "L2_labels": {
      "entities": {
        "P131": {
          "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
          "label": "located in the administrative territorial entity"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q20563062": {
          "description": "Poet, Writer and Historian",
          "label": "Sriramoju Haragopal"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q52004125": {
          "description": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "label": "allowed-entity-types constraint"
        },
        "Q53869507": {
          "description": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "label": "property scope constraint"
        },
        "Q54828448": {
          "description": "property scope type",
          "label": "as main value"
        },
        "Q54828449": {
          "description": "property scope type",
          "label": "as qualifier"
        },
        "Q59259864": {
          "description": "mandalam in Yadadri Bhuvanagiri district, Telangana",
          "label": "Alair Mandal"
        },
        "Q677037": {
          "description": "state in southern India",
          "label": "Telangana"
        }
      }
    },
    "L3_neighborhood": {
      "outgoing_edges": []
    },
    "L4_constraints": {
      "constraints": [
        {
          "constraint_type": {
            "label": "property scope constraint",
            "qid": "Q53869507"
          },
          "qualifiers": [
            {
              "property_id": "P5314",
              "property_label": "property scope",
              "values": [
                {
                  "label": "as main value",
                  "qid": "Q54828448",
                  "raw": "Q54828448"
                },
                {
                  "label": "as qualifier",
                  "qid": "Q54828449",
                  "raw": "Q54828449"
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449)"
        },
        {
          "constraint_type": {
            "label": "allowed-entity-types constraint",
            "qid": "Q52004125"
          },
          "qualifiers": [
            {
              "property_id": "P2305",
              "property_label": "item of property constraint",
              "values": [
                {
                  "label": "Wikibase item",
                  "qid": "Q29934200",
                  "raw": "Q29934200"
                }
              ]
            }
          ],
          "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200)"
        }
      ],
      "property_id": "P131"
    }
  },
  "property": "P131",
  "qid": "Q20563062",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
    "report_violat
```

## t_box_constraint_family_selection / case_000002

- Context: `logic_only`
- Track: `T_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: t_box_constraint_family_selection

Select the changed/target constraint family supported by the visible T-box context.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","constraint_type_qid":"Q...|UNKNOWN","support":"visible_inventory|visible_pre_change_signature|not_visible","rationale":"..."}

Input case JSON:

{
  "id": "case_000002",
  "labels_en": {
    "property": {
      "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
      "label": "instance of"
    },
    "qid": {
      "description": "Manga di Gō Nagai",
      "label": "Cronache delle guerre demoniache"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q21510851",
        "label": "allowed qualifiers constraint"
      },
      {
        "constraint_qid": "Q52558054",
        "label": "none-of constraint"
      },
      {
        "constraint_qid": "Q53869507",
        "label": "property scope constraint"
      },
      {
        "constraint_qid": "Q25796498",
        "label": "contemporary constraint"
      },
      {
        "constraint_qid": "Q21510859",
        "label": "one-of constraint"
      },
      {
        "constraint_qid": "Q52004125",
        "label": "allowed-entity-types constraint"
      },
      {
        "constraint_qid": "Q21510864",
        "label": "value-requires-statement constraint"
      }
    ],
    "violation_context": {
      "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
      "report_violation_type": "None of",
      "report_violation_type_normalized": "None of",
      "report_violation_type_qids": [],
      "report_violation_type_raw": "None of"
    }
  },
  "property": "P31",
  "qid": "Q115933554",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
    "report_violation_type": "None of",
    "report_violation_type_normalized": "None of",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "None of"
  }
}
```

## t_box_action_selection / case_000002

- Context: `logic_only`
- Track: `T_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: t_box_action_selection

Choose the schema action supported by visible evidence only.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","action":"RELAXATION_SET_EXPANSION|RESTRICTION_SET_CONTRACTION|RELAXATION_RANGE_WIDENED|RESTRICTION_RANGE_NARROWED|SCHEMA_UPDATE|COINCIDENTAL_SCHEMA_CHANGE","direction_visible":true,"rationale":"..."}

Input case JSON:

{
  "id": "case_000002",
  "labels_en": {
    "property": {
      "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
      "label": "instance of"
    },
    "qid": {
      "description": "Manga di Gō Nagai",
      "label": "Cronache delle guerre demoniache"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q21510851",
        "label": "allowed qualifiers constraint"
      },
      {
        "constraint_qid": "Q52558054",
        "label": "none-of constraint"
      },
      {
        "constraint_qid": "Q53869507",
        "label": "property scope constraint"
      },
      {
        "constraint_qid": "Q25796498",
        "label": "contemporary constraint"
      },
      {
        "constraint_qid": "Q21510859",
        "label": "one-of constraint"
      },
      {
        "constraint_qid": "Q52004125",
        "label": "allowed-entity-types constraint"
      },
      {
        "constraint_qid": "Q21510864",
        "label": "value-requires-statement constraint"
      }
    ],
    "violation_context": {
      "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
      "report_violation_type": "None of",
      "report_violation_type_normalized": "None of",
      "report_violation_type_qids": [],
      "report_violation_type_raw": "None of"
    }
  },
  "property": "P31",
  "qid": "Q115933554",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
    "report_violation_type": "None of",
    "report_violation_type_normalized": "None of",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "None of"
  }
}
```

## t_box_signature_visibility / case_000002

- Context: `logic_only`
- Track: `T_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: t_box_signature_visibility

Decide whether exact signature_after values are visible or must be withheld.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","signature_after_visible":true,"visible_changed_values":["..."],"recommended_behavior":"exact_signature|schema_update_empty_signature","rationale":"..."}

Input case JSON:

{
  "id": "case_000002",
  "labels_en": {
    "property": {
      "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
      "label": "instance of"
    },
    "qid": {
      "description": "Manga di Gō Nagai",
      "label": "Cronache delle guerre demoniache"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q21510851",
        "label": "allowed qualifiers constraint"
      },
      {
        "constraint_qid": "Q52558054",
        "label": "none-of constraint"
      },
      {
        "constraint_qid": "Q53869507",
        "label": "property scope constraint"
      },
      {
        "constraint_qid": "Q25796498",
        "label": "contemporary constraint"
      },
      {
        "constraint_qid": "Q21510859",
        "label": "one-of constraint"
      },
      {
        "constraint_qid": "Q52004125",
        "label": "allowed-entity-types constraint"
      },
      {
        "constraint_qid": "Q21510864",
        "label": "value-requires-statement constraint"
      }
    ],
    "violation_context": {
      "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
      "report_violation_type": "None of",
      "report_violation_type_normalized": "None of",
      "report_violation_type_qids": [],
      "report_violation_type_raw": "None of"
    }
  },
  "property": "P31",
  "qid": "Q115933554",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
    "report_violation_type": "None of",
    "report_violation_type_normalized": "None of",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "None of"
  }
}
```

## track_locus_contrast / case_000002

- Context: `logic_only`
- Track: `T_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: track_locus_contrast

Contrast A-box and T-box repair-locus evidence without proposing a repair.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","a_box_evidence":["..."],"t_box_evidence":["..."],"likely_locus":"A_BOX|T_BOX|AMBIGUOUS","rationale":"..."}

Input case JSON:

{
  "id": "case_000002",
  "labels_en": {
    "property": {
      "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
      "label": "instance of"
    },
    "qid": {
      "description": "Manga di Gō Nagai",
      "label": "Cronache delle guerre demoniache"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q21510851",
        "label": "allowed qualifiers constraint"
      },
      {
        "constraint_qid": "Q52558054",
        "label": "none-of constraint"
      },
      {
        "constraint_qid": "Q53869507",
        "label": "property scope constraint"
      },
      {
        "constraint_qid": "Q25796498",
        "label": "contemporary constraint"
      },
      {
        "constraint_qid": "Q21510859",
        "label": "one-of constraint"
      },
      {
        "constraint_qid": "Q52004125",
        "label": "allowed-entity-types constraint"
      },
      {
        "constraint_qid": "Q21510864",
        "label": "value-requires-statement constraint"
      }
    ],
    "violation_context": {
      "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
      "report_violation_type": "None of",
      "report_violation_type_normalized": "None of",
      "report_violation_type_qids": [],
      "report_violation_type_raw": "None of"
    }
  },
  "property": "P31",
  "qid": "Q115933554",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
    "report_violation_type": "None of",
    "report_violation_type_normalized": "None of",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "None of"
  }
}
```

## t_box_constraint_family_selection / case_000002

- Context: `local_graph`
- Track: `T_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: t_box_constraint_family_selection

Select the changed/target constraint family supported by the visible T-box context.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","constraint_type_qid":"Q...|UNKNOWN","support":"visible_inventory|visible_pre_change_signature|not_visible","rationale":"..."}

Input case JSON:

{
  "id": "case_000002",
  "labels_en": {
    "property": {
      "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
      "label": "instance of"
    },
    "qid": {
      "description": "Manga di Gō Nagai",
      "label": "Cronache delle guerre demoniache"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "Manga di Gō Nagai",
      "label": "Cronache delle guerre demoniache",
      "qid": "Q115933554"
    },
    "L2_labels": {
      "entities": {
        "P31": {
          "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
          "label": "instance of"
        },
        "Q115933554": {
          "description": "Manga di Gō Nagai",
          "label": "Cronache delle guerre demoniache"
        },
        "Q21510851": {
          "description": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "label": "allowed qualifiers constraint"
        },
        "Q21510859": {
          "description": "type of constraint for Wikidata properties: used to specify that the value for this property has to be one of a given set of items",
          "label": "one-of constraint"
        },
        "Q21510864": {
          "description": "type of constraint for Wikidata properties: used to specify that the referenced item should have a statement with a given property",
          "label": "value-requires-statement constraint"
        },
        "Q25796498": {
          "description": "type of constraint for Wikidata properties: used to specify that the subject and the object have to coincide or coexist at some point of history",
          "label": "contemporary constraint"
        },
        "Q52004125": {
          "description": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "label": "allowed-entity-types constraint"
        },
        "Q52558054": {
          "description": "constraint specifying values that should not be used for the given property",
          "label": "none-of constraint"
        },
        "Q53869507": {
          "description": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "label": "property scope constraint"
        }
      }
    },
    "L3_neighborhood": {
      "outgoing_edges": []
    },
    "L4_constraints": {
      "constraint_family_inventory": [
        {
          "constraint_qid": "Q21510851",
          "label": "allowed qualifiers constraint"
        },
        {
          "constraint_qid": "Q52558054",
          "label": "none-of constraint"
        },
        {
          "constraint_qid": "Q53869507",
          "label": "property scope constraint"
        },
        {
          "constraint_qid": "Q25796498",
          "label": "contemporary constraint"
        },
        {
          "constraint_qid": "Q21510859",
          "label": "one-of constraint"
        },
        {
          "constraint_qid": "Q52004125",
          "label": "allowed-entity-types constraint"
        },
        {
          "constraint_qid": "Q21510864",
          "label": "value-requires-statement constraint"
        }
      ],
      "violation_context": {
        "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
        "report_violation_type": "None of",
        "report_violation_type_normalized": "None of",
        "report_violation_type_qids": [],
        "report_violation_type_raw": "None of"
      }
    }
  },
  "property": "P31",
  "qid": "Q115933554",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
    "report_violation_type": "None of",
    "report_violation_type_normalized": "None of",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "None of"
  }
}
```

## t_box_action_selection / case_000002

- Context: `local_graph`
- Track: `T_BOX`

```text
Prompt version: diagnostic_tasks_v1

Diagnostic task: t_box_action_selection

Choose the schema action supported by visible evidence only.

Return valid JSON only. Do not propose a full repair unless the diagnostic contract asks for it.

Output contract:

{"case_id":"<copy id>","action":"RELAXATION_SET_EXPANSION|RESTRICTION_SET_CONTRACTION|RELAXATION_RANGE_WIDENED|RESTRICTION_RANGE_NARROWED|SCHEMA_UPDATE|COINCIDENTAL_SCHEMA_CHANGE","direction_visible":true,"rationale":"..."}

Input case JSON:

{
  "id": "case_000002",
  "labels_en": {
    "property": {
      "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
      "label": "instance of"
    },
    "qid": {
      "description": "Manga di Gō Nagai",
      "label": "Cronache delle guerre demoniache"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "Manga di Gō Nagai",
      "label": "Cronache delle guerre demoniache",
      "qid": "Q115933554"
    },
    "L2_labels": {
      "entities": {
        "P31": {
          "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
          "label": "instance of"
        },
        "Q115933554": {
          "description": "Manga di Gō Nagai",
          "label": "Cronache delle guerre demoniache"
        },
        "Q21510851": {
          "description": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "label": "allowed qualifiers constraint"
        },
        "Q21510859": {
          "description": "type of constraint for Wikidata properties: used to specify that the value for this property has to be one of a given set of items",
          "label": "one-of constraint"
        },
        "Q21510864": {
          "description": "type of constraint for Wikidata properties: used to specify that the referenced item should have a statement with a given property",
          "label": "value-requires-statement constraint"
        },
        "Q25796498": {
          "description": "type of constraint for Wikidata properties: used to specify that the subject and the object have to coincide or coexist at some point of history",
          "label": "contemporary constraint"
        },
        "Q52004125": {
          "description": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "label": "allowed-entity-types constraint"
        },
        "Q52558054": {
          "description": "constraint specifying values that should not be used for the given property",
          "label": "none-of constraint"
        },
        "Q53869507": {
          "description": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "label": "property scope constraint"
        }
      }
    },
    "L3_neighborhood": {
      "outgoing_edges": []
    },
    "L4_constraints": {
      "constraint_family_inventory": [
        {
          "constraint_qid": "Q21510851",
          "label": "allowed qualifiers constraint"
        },
        {
          "constraint_qid": "Q52558054",
          "label": "none-of constraint"
        },
        {
          "constraint_qid": "Q53869507",
          "label": "property scope constraint"
        },
        {
          "constraint_qid": "Q25796498",
          "label": "contemporary constraint"
        },
        {
          "constraint_qid": "Q21510859",
          "label": "one-of constraint"
        },
        {
          "constraint_qid": "Q52004125",
          "label": "allowed-entity-types constraint"
        },
        {
          "constraint_qid": "Q21510864",
          "label": "value-requires-statement constraint"
        }
      ],
      "violation_context": {
        "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
        "report_violation_type": "None of",
        "report_violation_type_normalized": "None of",
        "report_violation_type_qids": [],
        "report_violation_type_raw": "None of"
      }
    }
  },
  "property": "P31",
  "qid": "Q115933554",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P31",
    "report_violation_type": "None of",
    "report_violation_type_normalized": "None of",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "None of"
  }
}
```
