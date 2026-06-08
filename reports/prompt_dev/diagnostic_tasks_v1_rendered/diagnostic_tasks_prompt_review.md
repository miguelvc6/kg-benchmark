# Diagnostic Tasks v1 Render Review

No model inference was run.

Rendered prompts: `768`

## a_box_value_extraction / case_000374

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
  "id": "case_000374",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "A caterer in Nigeria",
      "label": "Dorcas peter Atule Iyeoma"
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
  "qid": "Q136682731",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
    "report_violation_type": "Conflicts with P|31",
    "report_violation_type_normalized": "Conflicts with P|31",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Conflicts with P|31",
    "value": [
      "Q387745"
    ],
    "value_descriptions_en": [
      "state in Nigeria"
    ],
    "value_labels_en": [
      "Kogi State"
    ]
  }
}
```

## a_box_operation_selection / case_000374

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
  "id": "case_000374",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "A caterer in Nigeria",
      "label": "Dorcas peter Atule Iyeoma"
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
  "qid": "Q136682731",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
    "report_violation_type": "Conflicts with P|31",
    "report_violation_type_normalized": "Conflicts with P|31",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Conflicts with P|31",
    "value": [
      "Q387745"
    ],
    "value_descriptions_en": [
      "state in Nigeria"
    ],
    "value_labels_en": [
      "Kogi State"
    ]
  }
}
```

## a_box_answerability / case_000374

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
  "id": "case_000374",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "A caterer in Nigeria",
      "label": "Dorcas peter Atule Iyeoma"
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
  "qid": "Q136682731",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
    "report_violation_type": "Conflicts with P|31",
    "report_violation_type_normalized": "Conflicts with P|31",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Conflicts with P|31",
    "value": [
      "Q387745"
    ],
    "value_descriptions_en": [
      "state in Nigeria"
    ],
    "value_labels_en": [
      "Kogi State"
    ]
  }
}
```

## track_locus_contrast / case_000374

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
  "id": "case_000374",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "A caterer in Nigeria",
      "label": "Dorcas peter Atule Iyeoma"
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
  "qid": "Q136682731",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
    "report_violation_type": "Conflicts with P|31",
    "report_violation_type_normalized": "Conflicts with P|31",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Conflicts with P|31",
    "value": [
      "Q387745"
    ],
    "value_descriptions_en": [
      "state in Nigeria"
    ],
    "value_labels_en": [
      "Kogi State"
    ]
  }
}
```

## a_box_value_extraction / case_000374

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
  "id": "case_000374",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "A caterer in Nigeria",
      "label": "Dorcas peter Atule Iyeoma"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "A caterer in Nigeria",
      "label": "Dorcas peter Atule Iyeoma",
      "properties": {
        "P131": [
          "Q387745"
        ]
      },
      "qid": "Q136682731"
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
        "Q136682731": {
          "description": "A caterer in Nigeria",
          "label": "Dorcas peter Atule Iyeoma"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q387745": {
          "description": "state in Nigeria",
          "label": "Kogi State"
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
  "qid": "Q136682731",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
    "report_violation_type": "Conflicts with P|31",
    "report_violation_type_normalized": "Conflicts with P|31",
    "report_violation
```

## a_box_operation_selection / case_000374

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
  "id": "case_000374",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "A caterer in Nigeria",
      "label": "Dorcas peter Atule Iyeoma"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "A caterer in Nigeria",
      "label": "Dorcas peter Atule Iyeoma",
      "properties": {
        "P131": [
          "Q387745"
        ]
      },
      "qid": "Q136682731"
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
        "Q136682731": {
          "description": "A caterer in Nigeria",
          "label": "Dorcas peter Atule Iyeoma"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q387745": {
          "description": "state in Nigeria",
          "label": "Kogi State"
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
  "qid": "Q136682731",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
    "report_violation_type": "Conflicts with P|31",
    "report_violation_type_normalized": "Conflicts with P|31",
    "report_violation_type_qids": [],
    "r
```

## a_box_answerability / case_000374

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
  "id": "case_000374",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "A caterer in Nigeria",
      "label": "Dorcas peter Atule Iyeoma"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "A caterer in Nigeria",
      "label": "Dorcas peter Atule Iyeoma",
      "properties": {
        "P131": [
          "Q387745"
        ]
      },
      "qid": "Q136682731"
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
        "Q136682731": {
          "description": "A caterer in Nigeria",
          "label": "Dorcas peter Atule Iyeoma"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q387745": {
          "description": "state in Nigeria",
          "label": "Kogi State"
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
  "qid": "Q136682731",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
    "report_violation_type": "Conflicts with P|31",
    "report_violation_type_normalized": "Conflicts with P|31",
    "report_violation_type_qids": [],
    "repo
```

## track_locus_contrast / case_000374

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
  "id": "case_000374",
  "labels_en": {
    "property": {
      "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
      "label": "located in the administrative territorial entity"
    },
    "qid": {
      "description": "A caterer in Nigeria",
      "label": "Dorcas peter Atule Iyeoma"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "A caterer in Nigeria",
      "label": "Dorcas peter Atule Iyeoma",
      "properties": {
        "P131": [
          "Q387745"
        ]
      },
      "qid": "Q136682731"
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
        "Q136682731": {
          "description": "A caterer in Nigeria",
          "label": "Dorcas peter Atule Iyeoma"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q387745": {
          "description": "state in Nigeria",
          "label": "Kogi State"
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
  "qid": "Q136682731",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P131",
    "report_violation_type": "Conflicts with P|31",
    "report_violation_type_normalized": "Conflicts with P|31",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Confl
```

## t_box_constraint_family_selection / case_000093

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
  "id": "case_000093",
  "labels_en": {
    "property": {
      "description": "structure of a creative work",
      "label": "form of creative work"
    },
    "qid": {
      "description": "poem written by Bai Juyi",
      "label": "和錢員外禁中夙興見示"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q52004125",
        "label": "allowed-entity-types constraint"
      },
      {
        "constraint_qid": "Q53869507",
        "label": "property scope constraint"
      },
      {
        "constraint_qid": "Q21503250",
        "label": "subject type constraint"
      },
      {
        "constraint_qid": "Q21510865",
        "label": "value-type constraint"
      },
      {
        "constraint_qid": "Q52558054",
        "label": "none-of constraint"
      },
      {
        "constraint_qid": "Q21502838",
        "label": "conflicts-with constraint"
      }
    ],
    "temporal_policy": "compact_inventory_no_pre_change_signature",
    "violation_context": {
      "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
      "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
      "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
      "report_violation_type_qids": [
        "Q386724",
        "Q17489659",
        "Q15306849"
      ],
      "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849"
    }
  },
  "property": "P7937",
  "qid": "Q15937066",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
    "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
    "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
    "report_violation_type_qids": [
      "Q386724",
      "Q17489659",
      "Q15306849"
    ],
    "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849"
  }
}
```

## t_box_action_selection / case_000093

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
  "id": "case_000093",
  "labels_en": {
    "property": {
      "description": "structure of a creative work",
      "label": "form of creative work"
    },
    "qid": {
      "description": "poem written by Bai Juyi",
      "label": "和錢員外禁中夙興見示"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q52004125",
        "label": "allowed-entity-types constraint"
      },
      {
        "constraint_qid": "Q53869507",
        "label": "property scope constraint"
      },
      {
        "constraint_qid": "Q21503250",
        "label": "subject type constraint"
      },
      {
        "constraint_qid": "Q21510865",
        "label": "value-type constraint"
      },
      {
        "constraint_qid": "Q52558054",
        "label": "none-of constraint"
      },
      {
        "constraint_qid": "Q21502838",
        "label": "conflicts-with constraint"
      }
    ],
    "temporal_policy": "compact_inventory_no_pre_change_signature",
    "violation_context": {
      "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
      "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
      "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
      "report_violation_type_qids": [
        "Q386724",
        "Q17489659",
        "Q15306849"
      ],
      "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849"
    }
  },
  "property": "P7937",
  "qid": "Q15937066",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
    "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
    "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
    "report_violation_type_qids": [
      "Q386724",
      "Q17489659",
      "Q15306849"
    ],
    "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849"
  }
}
```

## t_box_signature_visibility / case_000093

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
  "id": "case_000093",
  "labels_en": {
    "property": {
      "description": "structure of a creative work",
      "label": "form of creative work"
    },
    "qid": {
      "description": "poem written by Bai Juyi",
      "label": "和錢員外禁中夙興見示"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q52004125",
        "label": "allowed-entity-types constraint"
      },
      {
        "constraint_qid": "Q53869507",
        "label": "property scope constraint"
      },
      {
        "constraint_qid": "Q21503250",
        "label": "subject type constraint"
      },
      {
        "constraint_qid": "Q21510865",
        "label": "value-type constraint"
      },
      {
        "constraint_qid": "Q52558054",
        "label": "none-of constraint"
      },
      {
        "constraint_qid": "Q21502838",
        "label": "conflicts-with constraint"
      }
    ],
    "temporal_policy": "compact_inventory_no_pre_change_signature",
    "violation_context": {
      "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
      "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
      "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
      "report_violation_type_qids": [
        "Q386724",
        "Q17489659",
        "Q15306849"
      ],
      "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849"
    }
  },
  "property": "P7937",
  "qid": "Q15937066",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
    "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
    "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
    "report_violation_type_qids": [
      "Q386724",
      "Q17489659",
      "Q15306849"
    ],
    "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849"
  }
}
```

## track_locus_contrast / case_000093

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
  "id": "case_000093",
  "labels_en": {
    "property": {
      "description": "structure of a creative work",
      "label": "form of creative work"
    },
    "qid": {
      "description": "poem written by Bai Juyi",
      "label": "和錢員外禁中夙興見示"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q52004125",
        "label": "allowed-entity-types constraint"
      },
      {
        "constraint_qid": "Q53869507",
        "label": "property scope constraint"
      },
      {
        "constraint_qid": "Q21503250",
        "label": "subject type constraint"
      },
      {
        "constraint_qid": "Q21510865",
        "label": "value-type constraint"
      },
      {
        "constraint_qid": "Q52558054",
        "label": "none-of constraint"
      },
      {
        "constraint_qid": "Q21502838",
        "label": "conflicts-with constraint"
      }
    ],
    "temporal_policy": "compact_inventory_no_pre_change_signature",
    "violation_context": {
      "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
      "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
      "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
      "report_violation_type_qids": [
        "Q386724",
        "Q17489659",
        "Q15306849"
      ],
      "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849"
    }
  },
  "property": "P7937",
  "qid": "Q15937066",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
    "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
    "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
    "report_violation_type_qids": [
      "Q386724",
      "Q17489659",
      "Q15306849"
    ],
    "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849"
  }
}
```

## t_box_constraint_family_selection / case_000093

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
  "id": "case_000093",
  "labels_en": {
    "property": {
      "description": "structure of a creative work",
      "label": "form of creative work"
    },
    "qid": {
      "description": "poem written by Bai Juyi",
      "label": "和錢員外禁中夙興見示"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "poem written by Bai Juyi",
      "label": "和錢員外禁中夙興見示",
      "qid": "Q15937066"
    },
    "L2_labels": {
      "entities": {
        "P7937": {
          "description": "structure of a creative work",
          "label": "form of creative work"
        },
        "Q15306849": {
          "description": "creative work which only appears in works of fiction",
          "label": "fictional creative work"
        },
        "Q15937066": {
          "description": "poem written by Bai Juyi",
          "label": "和錢員外禁中夙興見示"
        },
        "Q17489659": {
          "description": "any set of works",
          "label": "group of works"
        },
        "Q21502838": {
          "description": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "label": "conflicts-with constraint"
        },
        "Q21503250": {
          "description": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "label": "subject type constraint"
        },
        "Q21510865": {
          "description": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "label": "value-type constraint"
        },
        "Q386724": {
          "description": "intellectual or artistic creation",
          "label": "work"
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
          "constraint_qid": "Q52004125",
          "label": "allowed-entity-types constraint"
        },
        {
          "constraint_qid": "Q53869507",
          "label": "property scope constraint"
        },
        {
          "constraint_qid": "Q21503250",
          "label": "subject type constraint"
        },
        {
          "constraint_qid": "Q21510865",
          "label": "value-type constraint"
        },
        {
          "constraint_qid": "Q52558054",
          "label": "none-of constraint"
        },
        {
          "constraint_qid": "Q21502838",
          "label": "conflicts-with constraint"
        }
      ],
      "temporal_policy": "compact_inventory_no_pre_change_signature",
      "violation_context": {
        "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
        "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
        "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
        "report_violation_type_qids": [
          "Q386724",
          "Q17489659",
          "Q15306849"
        ],
        "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849"
      }
    }
  },
  "property": "P7937",
  "qid": "Q15937066",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
    "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
    "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
    "report_violation_type_qids": [
      "Q386724",
      "Q17489659",
      "Q15306849"
    ],
    "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849"
  }
}
```

## t_box_action_selection / case_000093

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
  "id": "case_000093",
  "labels_en": {
    "property": {
      "description": "structure of a creative work",
      "label": "form of creative work"
    },
    "qid": {
      "description": "poem written by Bai Juyi",
      "label": "和錢員外禁中夙興見示"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "poem written by Bai Juyi",
      "label": "和錢員外禁中夙興見示",
      "qid": "Q15937066"
    },
    "L2_labels": {
      "entities": {
        "P7937": {
          "description": "structure of a creative work",
          "label": "form of creative work"
        },
        "Q15306849": {
          "description": "creative work which only appears in works of fiction",
          "label": "fictional creative work"
        },
        "Q15937066": {
          "description": "poem written by Bai Juyi",
          "label": "和錢員外禁中夙興見示"
        },
        "Q17489659": {
          "description": "any set of works",
          "label": "group of works"
        },
        "Q21502838": {
          "description": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "label": "conflicts-with constraint"
        },
        "Q21503250": {
          "description": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "label": "subject type constraint"
        },
        "Q21510865": {
          "description": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "label": "value-type constraint"
        },
        "Q386724": {
          "description": "intellectual or artistic creation",
          "label": "work"
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
          "constraint_qid": "Q52004125",
          "label": "allowed-entity-types constraint"
        },
        {
          "constraint_qid": "Q53869507",
          "label": "property scope constraint"
        },
        {
          "constraint_qid": "Q21503250",
          "label": "subject type constraint"
        },
        {
          "constraint_qid": "Q21510865",
          "label": "value-type constraint"
        },
        {
          "constraint_qid": "Q52558054",
          "label": "none-of constraint"
        },
        {
          "constraint_qid": "Q21502838",
          "label": "conflicts-with constraint"
        }
      ],
      "temporal_policy": "compact_inventory_no_pre_change_signature",
      "violation_context": {
        "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
        "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
        "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
        "report_violation_type_qids": [
          "Q386724",
          "Q17489659",
          "Q15306849"
        ],
        "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849"
      }
    }
  },
  "property": "P7937",
  "qid": "Q15937066",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P7937",
    "report_violation_type": "Type Q|386724, Q|17489659, Q|15306849",
    "report_violation_type_normalized": "Type Q|386724, Q|17489659, Q|15306849",
    "report_violation_type_qids": [
      "Q386724",
      "Q17489659",
      "Q15306849"
    ],
    "report_violation_type_raw": "Type Q|386724, Q|17489659, Q|15306849"
  }
}
```
