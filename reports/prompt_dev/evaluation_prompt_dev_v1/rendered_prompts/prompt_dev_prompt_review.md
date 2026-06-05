# Prompt Development Review

No LLM inference was run for this artifact.

Rendered prompts: `96`

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / reform_Q100259828_P476_2417146821

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "reform_Q100259828_P476_2417146821",
  "labels_en": {
    "property": {
      "description": "identifier for European legal texts in EUR-Lex database",
      "label": "CELEX number"
    },
    "qid": {
      "description": "European Union Directive (EU) 2002/36",
      "label": "Commission Directive 2002/36/EC of 29 April 2002"
    }
  },
  "logic_context": {
    "constraints": [
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
              },
              {
                "label": "МэдыяІнфа Вікібазы",
                "qid": "Q59712033",
                "raw": "Q59712033"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033)"
      },
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
                "label": "as reference",
                "qid": "Q54828450",
                "raw": "Q54828450"
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448), as reference (Q54828450)"
      }
    ],
    "property_id": "P476"
  },
  "property": "P476",
  "qid": "Q100259828",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P476",
    "report_violation_type": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_normalized": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_qids": [
      "Q11122",
      "Q2334719",
      "Q131569",
      "Q3629172"
    ],
    "report_violation_type_raw": "Type Q|11122, Q|2334719, Q|131569, Q|3629172"
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / reform_Q100259828_P476_2417146821

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: hybrid_json_nl

Task: t_box_repair

Propose an executable T-box schema reform for the focus property. Use constraint-family QIDs from the supplied context as constraint_type_qid values. Do not copy ordinary entity/type QIDs into constraint-family fields.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly>",
  "target": {"pid": "P...", "constraint_type_qid": "Q..."},
  "proposal": {
    "action": "RELAXATION_RANGE_WIDENED"
      | "RESTRICTION_RANGE_NARROWED"
      | "RELAXATION_SET_EXPANSION"
      | "RESTRICTION_SET_CONTRACTION"
      | "SCHEMA_UPDATE"
      | "COINCIDENTAL_SCHEMA_CHANGE",
    "signature_after": [
      {
        "constraint_qid": "Q...",
        "snaktype": "VALUE" | "SOMEVALUE" | "NOVALUE",
        "rank": "normal" | "preferred" | "deprecated",
        "qualifiers": [{"property_id": "P...", "values": ["Q..." | "literal"]}]
      }
    ]
  },
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "HISTORY" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "reform_Q100259828_P476_2417146821",
  "labels_en": {
    "property": {
      "description": "identifier for European legal texts in EUR-Lex database",
      "label": "CELEX number"
    },
    "qid": {
      "description": "European Union Directive (EU) 2002/36",
      "label": "Commission Directive 2002/36/EC of 29 April 2002"
    }
  },
  "logic_context": {
    "constraints": [
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
              },
              {
                "label": "МэдыяІнфа Вікібазы",
                "qid": "Q59712033",
                "raw": "Q59712033"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033)"
      },
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
                "label": "as reference",
                "qid": "Q54828450",
                "raw": "Q54828450"
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448), as reference (Q54828450)"
      }
    ],
    "property_id": "P476"
  },
  "property": "P476",
  "qid": "Q100259828",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P476",
    "report_violation_type": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_normalized": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_qids": [
      "Q11122",
      "Q2334719",
      "Q131569",
      "Q3629172"
    ],
    "report_violation_type_raw": "Type Q|11122, Q|2334719, Q|131569, Q|3629172"
  }
}
```

## prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / reform_Q100259828_P476_2417146821

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "reform_Q100259828_P476_2417146821",
  "labels_en": {
    "property": {
      "description": "identifier for European legal texts in EUR-Lex database",
      "label": "CELEX number"
    },
    "qid": {
      "description": "European Union Directive (EU) 2002/36",
      "label": "Commission Directive 2002/36/EC of 29 April 2002"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "European Union Directive (EU) 2002/36",
      "label": "Commission Directive 2002/36/EC of 29 April 2002",
      "qid": "Q100259828",
      "sitelinks_count": 0
    },
    "L2_labels": {
      "entities": {
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P476": {
          "description": "identifier for European legal texts in EUR-Lex database",
          "label": "CELEX number"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q100259828": {
          "description": "European Union Directive (EU) 2002/36",
          "label": "Commission Directive 2002/36/EC of 29 April 2002"
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
        "Q54828450": {
          "description": "property scope type",
          "label": "as reference"
        },
        "Q59712033": {
          "description": "Wikibase entity type for Wikimedia Commons",
          "label": "МэдыяІнфа Вікібазы"
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
                },
                {
                  "label": "МэдыяІнфа Вікібазы",
                  "qid": "Q59712033",
                  "raw": "Q59712033"
                }
              ]
            }
          ],
          "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033)"
        },
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
                  "label": "as reference",
                  "qid": "Q54828450",
                  "raw": "Q54828450"
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448), as reference (Q54828450)"
        }
      ],
      "property_id": "P476"
    }
  },
  "property": "P476",
  "qid": "Q100259828",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P476",
    "report_violation_type": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_normalized": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_qids": [
      "Q11122",
      "Q2334719",
      "Q131569",
      "Q3629172"
    ],
    "report_violation_type_raw": "Type Q|11122, Q|2334719, Q|131569, Q|3629172"
  }
}
```

## prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / reform_Q100259828_P476_2417146821

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: hybrid_json_nl

Task: t_box_repair

Propose an executable T-box schema reform for the focus property. Use constraint-family QIDs from the supplied context as constraint_type_qid values. Do not copy ordinary entity/type QIDs into constraint-family fields.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly>",
  "target": {"pid": "P...", "constraint_type_qid": "Q..."},
  "proposal": {
    "action": "RELAXATION_RANGE_WIDENED"
      | "RESTRICTION_RANGE_NARROWED"
      | "RELAXATION_SET_EXPANSION"
      | "RESTRICTION_SET_CONTRACTION"
      | "SCHEMA_UPDATE"
      | "COINCIDENTAL_SCHEMA_CHANGE",
    "signature_after": [
      {
        "constraint_qid": "Q...",
        "snaktype": "VALUE" | "SOMEVALUE" | "NOVALUE",
        "rank": "normal" | "preferred" | "deprecated",
        "qualifiers": [{"property_id": "P...", "values": ["Q..." | "literal"]}]
      }
    ]
  },
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "HISTORY" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "reform_Q100259828_P476_2417146821",
  "labels_en": {
    "property": {
      "description": "identifier for European legal texts in EUR-Lex database",
      "label": "CELEX number"
    },
    "qid": {
      "description": "European Union Directive (EU) 2002/36",
      "label": "Commission Directive 2002/36/EC of 29 April 2002"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "European Union Directive (EU) 2002/36",
      "label": "Commission Directive 2002/36/EC of 29 April 2002",
      "qid": "Q100259828",
      "sitelinks_count": 0
    },
    "L2_labels": {
      "entities": {
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P476": {
          "description": "identifier for European legal texts in EUR-Lex database",
          "label": "CELEX number"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q100259828": {
          "description": "European Union Directive (EU) 2002/36",
          "label": "Commission Directive 2002/36/EC of 29 April 2002"
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
        "Q54828450": {
          "description": "property scope type",
          "label": "as reference"
        },
        "Q59712033": {
          "description": "Wikibase entity type for Wikimedia Commons",
          "label": "МэдыяІнфа Вікібазы"
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
                },
                {
                  "label": "МэдыяІнфа Вікібазы",
                  "qid": "Q59712033",
                  "raw": "Q59712033"
                }
              ]
            }
          ],
          "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033)"
        },
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
                  "label": "as reference",
                  "qid": "Q54828450",
                  "raw": "Q54828450"
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448), as reference (Q54828450)"
        }
      ],
      "property_id": "P476"
    }
  },
  "property": "P476",
  "qid": "Q100259828",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P476",
    "report_violation_type": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_normalized": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_qids": [
      "Q11122",
      "Q2334719",
      "Q131569",
      "Q3629172"
    ],
    "report_violation_type_raw": "Type Q|11122, Q|2334719, Q|131569, Q|3629172"
  }
}
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / reform_Q100263038_P476_2417146821

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "reform_Q100263038_P476_2417146821",
  "labels_en": {
    "property": {
      "description": "identifier for European legal texts in EUR-Lex database",
      "label": "CELEX number"
    },
    "qid": {
      "description": "European Union Directive (EU) 2013/29",
      "label": "Directive 2013/29/EU of the European Parliament and of the Council of 12 June 2013"
    }
  },
  "logic_context": {
    "constraints": [
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
              },
              {
                "label": "МэдыяІнфа Вікібазы",
                "qid": "Q59712033",
                "raw": "Q59712033"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033)"
      },
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
                "label": "as reference",
                "qid": "Q54828450",
                "raw": "Q54828450"
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448), as reference (Q54828450)"
      }
    ],
    "property_id": "P476"
  },
  "property": "P476",
  "qid": "Q100263038",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P476",
    "report_violation_type": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_normalized": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_qids": [
      "Q11122",
      "Q2334719",
      "Q131569",
      "Q3629172"
    ],
    "report_violation_type_raw": "Type Q|11122, Q|2334719, Q|131569, Q|3629172"
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / reform_Q100263038_P476_2417146821

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: hybrid_json_nl

Task: t_box_repair

Propose an executable T-box schema reform for the focus property. Use constraint-family QIDs from the supplied context as constraint_type_qid values. Do not copy ordinary entity/type QIDs into constraint-family fields.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly>",
  "target": {"pid": "P...", "constraint_type_qid": "Q..."},
  "proposal": {
    "action": "RELAXATION_RANGE_WIDENED"
      | "RESTRICTION_RANGE_NARROWED"
      | "RELAXATION_SET_EXPANSION"
      | "RESTRICTION_SET_CONTRACTION"
      | "SCHEMA_UPDATE"
      | "COINCIDENTAL_SCHEMA_CHANGE",
    "signature_after": [
      {
        "constraint_qid": "Q...",
        "snaktype": "VALUE" | "SOMEVALUE" | "NOVALUE",
        "rank": "normal" | "preferred" | "deprecated",
        "qualifiers": [{"property_id": "P...", "values": ["Q..." | "literal"]}]
      }
    ]
  },
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "HISTORY" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "reform_Q100263038_P476_2417146821",
  "labels_en": {
    "property": {
      "description": "identifier for European legal texts in EUR-Lex database",
      "label": "CELEX number"
    },
    "qid": {
      "description": "European Union Directive (EU) 2013/29",
      "label": "Directive 2013/29/EU of the European Parliament and of the Council of 12 June 2013"
    }
  },
  "logic_context": {
    "constraints": [
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
              },
              {
                "label": "МэдыяІнфа Вікібазы",
                "qid": "Q59712033",
                "raw": "Q59712033"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033)"
      },
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
                "label": "as reference",
                "qid": "Q54828450",
                "raw": "Q54828450"
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448), as reference (Q54828450)"
      }
    ],
    "property_id": "P476"
  },
  "property": "P476",
  "qid": "Q100263038",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P476",
    "report_violation_type": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_normalized": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_qids": [
      "Q11122",
      "Q2334719",
      "Q131569",
      "Q3629172"
    ],
    "report_violation_type_raw": "Type Q|11122, Q|2334719, Q|131569, Q|3629172"
  }
}
```

## prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / reform_Q100263038_P476_2417146821

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "reform_Q100263038_P476_2417146821",
  "labels_en": {
    "property": {
      "description": "identifier for European legal texts in EUR-Lex database",
      "label": "CELEX number"
    },
    "qid": {
      "description": "European Union Directive (EU) 2013/29",
      "label": "Directive 2013/29/EU of the European Parliament and of the Council of 12 June 2013"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "European Union Directive (EU) 2013/29",
      "label": "Directive 2013/29/EU of the European Parliament and of the Council of 12 June 2013",
      "qid": "Q100263038",
      "sitelinks_count": 0
    },
    "L2_labels": {
      "entities": {
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P476": {
          "description": "identifier for European legal texts in EUR-Lex database",
          "label": "CELEX number"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q100263038": {
          "description": "European Union Directive (EU) 2013/29",
          "label": "Directive 2013/29/EU of the European Parliament and of the Council of 12 June 2013"
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
        "Q54828450": {
          "description": "property scope type",
          "label": "as reference"
        },
        "Q59712033": {
          "description": "Wikibase entity type for Wikimedia Commons",
          "label": "МэдыяІнфа Вікібазы"
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
                },
                {
                  "label": "МэдыяІнфа Вікібазы",
                  "qid": "Q59712033",
                  "raw": "Q59712033"
                }
              ]
            }
          ],
          "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033)"
        },
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
                  "label": "as reference",
                  "qid": "Q54828450",
                  "raw": "Q54828450"
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448), as reference (Q54828450)"
        }
      ],
      "property_id": "P476"
    }
  },
  "property": "P476",
  "qid": "Q100263038",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P476",
    "report_violation_type": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_normalized": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_qids": [
      "Q11122",
      "Q2334719",
      "Q131569",
      "Q3629172"
    ],
    "report_violation_type_raw": "Type Q|11122, Q|2334719, Q|131569, Q|3629172"
  }
}
```

## prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / reform_Q100263038_P476_2417146821

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: hybrid_json_nl

Task: t_box_repair

Propose an executable T-box schema reform for the focus property. Use constraint-family QIDs from the supplied context as constraint_type_qid values. Do not copy ordinary entity/type QIDs into constraint-family fields.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly>",
  "target": {"pid": "P...", "constraint_type_qid": "Q..."},
  "proposal": {
    "action": "RELAXATION_RANGE_WIDENED"
      | "RESTRICTION_RANGE_NARROWED"
      | "RELAXATION_SET_EXPANSION"
      | "RESTRICTION_SET_CONTRACTION"
      | "SCHEMA_UPDATE"
      | "COINCIDENTAL_SCHEMA_CHANGE",
    "signature_after": [
      {
        "constraint_qid": "Q...",
        "snaktype": "VALUE" | "SOMEVALUE" | "NOVALUE",
        "rank": "normal" | "preferred" | "deprecated",
        "qualifiers": [{"property_id": "P...", "values": ["Q..." | "literal"]}]
      }
    ]
  },
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "HISTORY" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "reform_Q100263038_P476_2417146821",
  "labels_en": {
    "property": {
      "description": "identifier for European legal texts in EUR-Lex database",
      "label": "CELEX number"
    },
    "qid": {
      "description": "European Union Directive (EU) 2013/29",
      "label": "Directive 2013/29/EU of the European Parliament and of the Council of 12 June 2013"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "European Union Directive (EU) 2013/29",
      "label": "Directive 2013/29/EU of the European Parliament and of the Council of 12 June 2013",
      "qid": "Q100263038",
      "sitelinks_count": 0
    },
    "L2_labels": {
      "entities": {
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P476": {
          "description": "identifier for European legal texts in EUR-Lex database",
          "label": "CELEX number"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q100263038": {
          "description": "European Union Directive (EU) 2013/29",
          "label": "Directive 2013/29/EU of the European Parliament and of the Council of 12 June 2013"
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
        "Q54828450": {
          "description": "property scope type",
          "label": "as reference"
        },
        "Q59712033": {
          "description": "Wikibase entity type for Wikimedia Commons",
          "label": "МэдыяІнфа Вікібазы"
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
                },
                {
                  "label": "МэдыяІнфа Вікібазы",
                  "qid": "Q59712033",
                  "raw": "Q59712033"
                }
              ]
            }
          ],
          "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033)"
        },
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
                  "label": "as reference",
                  "qid": "Q54828450",
                  "raw": "Q54828450"
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448), as reference (Q54828450)"
        }
      ],
      "property_id": "P476"
    }
  },
  "property": "P476",
  "qid": "Q100263038",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P476",
    "report_violation_type": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_normalized": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_qids": [
      "Q11122",
      "Q2334719",
      "Q131569",
      "Q3629172"
    ],
    "report_violation_type_raw": "Type Q|11122, Q|2334719, Q|131569, Q|3629172"

```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / reform_Q100264032_P476_2417146821

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "reform_Q100264032_P476_2417146821",
  "labels_en": {
    "property": {
      "description": "identifier for European legal texts in EUR-Lex database",
      "label": "CELEX number"
    },
    "qid": {
      "description": "European Union Directive (EU) 2013/12",
      "label": "Council Directive 2013/12/EU of 13 May 2013"
    }
  },
  "logic_context": {
    "constraints": [
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
              },
              {
                "label": "МэдыяІнфа Вікібазы",
                "qid": "Q59712033",
                "raw": "Q59712033"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033)"
      },
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
                "label": "as reference",
                "qid": "Q54828450",
                "raw": "Q54828450"
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448), as reference (Q54828450)"
      }
    ],
    "property_id": "P476"
  },
  "property": "P476",
  "qid": "Q100264032",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P476",
    "report_violation_type": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_normalized": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_qids": [
      "Q11122",
      "Q2334719",
      "Q131569",
      "Q3629172"
    ],
    "report_violation_type_raw": "Type Q|11122, Q|2334719, Q|131569, Q|3629172"
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / reform_Q100264032_P476_2417146821

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: hybrid_json_nl

Task: t_box_repair

Propose an executable T-box schema reform for the focus property. Use constraint-family QIDs from the supplied context as constraint_type_qid values. Do not copy ordinary entity/type QIDs into constraint-family fields.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly>",
  "target": {"pid": "P...", "constraint_type_qid": "Q..."},
  "proposal": {
    "action": "RELAXATION_RANGE_WIDENED"
      | "RESTRICTION_RANGE_NARROWED"
      | "RELAXATION_SET_EXPANSION"
      | "RESTRICTION_SET_CONTRACTION"
      | "SCHEMA_UPDATE"
      | "COINCIDENTAL_SCHEMA_CHANGE",
    "signature_after": [
      {
        "constraint_qid": "Q...",
        "snaktype": "VALUE" | "SOMEVALUE" | "NOVALUE",
        "rank": "normal" | "preferred" | "deprecated",
        "qualifiers": [{"property_id": "P...", "values": ["Q..." | "literal"]}]
      }
    ]
  },
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "HISTORY" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "reform_Q100264032_P476_2417146821",
  "labels_en": {
    "property": {
      "description": "identifier for European legal texts in EUR-Lex database",
      "label": "CELEX number"
    },
    "qid": {
      "description": "European Union Directive (EU) 2013/12",
      "label": "Council Directive 2013/12/EU of 13 May 2013"
    }
  },
  "logic_context": {
    "constraints": [
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
              },
              {
                "label": "МэдыяІнфа Вікібазы",
                "qid": "Q59712033",
                "raw": "Q59712033"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033)"
      },
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
                "label": "as reference",
                "qid": "Q54828450",
                "raw": "Q54828450"
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448), as reference (Q54828450)"
      }
    ],
    "property_id": "P476"
  },
  "property": "P476",
  "qid": "Q100264032",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P476",
    "report_violation_type": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_normalized": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_qids": [
      "Q11122",
      "Q2334719",
      "Q131569",
      "Q3629172"
    ],
    "report_violation_type_raw": "Type Q|11122, Q|2334719, Q|131569, Q|3629172"
  }
}
```

## prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / reform_Q100264032_P476_2417146821

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "reform_Q100264032_P476_2417146821",
  "labels_en": {
    "property": {
      "description": "identifier for European legal texts in EUR-Lex database",
      "label": "CELEX number"
    },
    "qid": {
      "description": "European Union Directive (EU) 2013/12",
      "label": "Council Directive 2013/12/EU of 13 May 2013"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "European Union Directive (EU) 2013/12",
      "label": "Council Directive 2013/12/EU of 13 May 2013",
      "qid": "Q100264032",
      "sitelinks_count": 0
    },
    "L2_labels": {
      "entities": {
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P476": {
          "description": "identifier for European legal texts in EUR-Lex database",
          "label": "CELEX number"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q100264032": {
          "description": "European Union Directive (EU) 2013/12",
          "label": "Council Directive 2013/12/EU of 13 May 2013"
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
        "Q54828450": {
          "description": "property scope type",
          "label": "as reference"
        },
        "Q59712033": {
          "description": "Wikibase entity type for Wikimedia Commons",
          "label": "МэдыяІнфа Вікібазы"
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
                },
                {
                  "label": "МэдыяІнфа Вікібазы",
                  "qid": "Q59712033",
                  "raw": "Q59712033"
                }
              ]
            }
          ],
          "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033)"
        },
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
                  "label": "as reference",
                  "qid": "Q54828450",
                  "raw": "Q54828450"
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448), as reference (Q54828450)"
        }
      ],
      "property_id": "P476"
    }
  },
  "property": "P476",
  "qid": "Q100264032",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P476",
    "report_violation_type": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_normalized": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_qids": [
      "Q11122",
      "Q2334719",
      "Q131569",
      "Q3629172"
    ],
    "report_violation_type_raw": "Type Q|11122, Q|2334719, Q|131569, Q|3629172"
  }
}
```

## prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / reform_Q100264032_P476_2417146821

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: hybrid_json_nl

Task: t_box_repair

Propose an executable T-box schema reform for the focus property. Use constraint-family QIDs from the supplied context as constraint_type_qid values. Do not copy ordinary entity/type QIDs into constraint-family fields.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly>",
  "target": {"pid": "P...", "constraint_type_qid": "Q..."},
  "proposal": {
    "action": "RELAXATION_RANGE_WIDENED"
      | "RESTRICTION_RANGE_NARROWED"
      | "RELAXATION_SET_EXPANSION"
      | "RESTRICTION_SET_CONTRACTION"
      | "SCHEMA_UPDATE"
      | "COINCIDENTAL_SCHEMA_CHANGE",
    "signature_after": [
      {
        "constraint_qid": "Q...",
        "snaktype": "VALUE" | "SOMEVALUE" | "NOVALUE",
        "rank": "normal" | "preferred" | "deprecated",
        "qualifiers": [{"property_id": "P...", "values": ["Q..." | "literal"]}]
      }
    ]
  },
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "HISTORY" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "reform_Q100264032_P476_2417146821",
  "labels_en": {
    "property": {
      "description": "identifier for European legal texts in EUR-Lex database",
      "label": "CELEX number"
    },
    "qid": {
      "description": "European Union Directive (EU) 2013/12",
      "label": "Council Directive 2013/12/EU of 13 May 2013"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "European Union Directive (EU) 2013/12",
      "label": "Council Directive 2013/12/EU of 13 May 2013",
      "qid": "Q100264032",
      "sitelinks_count": 0
    },
    "L2_labels": {
      "entities": {
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P476": {
          "description": "identifier for European legal texts in EUR-Lex database",
          "label": "CELEX number"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q100264032": {
          "description": "European Union Directive (EU) 2013/12",
          "label": "Council Directive 2013/12/EU of 13 May 2013"
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
        "Q54828450": {
          "description": "property scope type",
          "label": "as reference"
        },
        "Q59712033": {
          "description": "Wikibase entity type for Wikimedia Commons",
          "label": "МэдыяІнфа Вікібазы"
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
                },
                {
                  "label": "МэдыяІнфа Вікібазы",
                  "qid": "Q59712033",
                  "raw": "Q59712033"
                }
              ]
            }
          ],
          "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033)"
        },
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
                  "label": "as reference",
                  "qid": "Q54828450",
                  "raw": "Q54828450"
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448), as reference (Q54828450)"
        }
      ],
      "property_id": "P476"
    }
  },
  "property": "P476",
  "qid": "Q100264032",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P476",
    "report_violation_type": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_normalized": "Type Q|11122, Q|2334719, Q|131569, Q|3629172",
    "report_violation_type_qids": [
      "Q11122",
      "Q2334719",
      "Q131569",
      "Q3629172"
    ],
    "report_violation_type_raw": "Type Q|11122, Q|2334719, Q|131569, Q|3629172"
  }
}
```
