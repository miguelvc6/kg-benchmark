# Prompt Development Review

No LLM inference was run for this artifact.

Rendered prompts: `384`

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

## prompt_dev_005_hybrid_json_nl_matched_2shot_logic_only_track_diagnosis_diagnosis_no_abstain / reform_Q100259828_P476_2417146821

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `matched_2shot`
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

Example 1 input:
{
  "id": "reform_Q2088074_P2605_2351781179",
  "labels_en": {
    "property": {
      "description": "identifier for a person in the Czecho-Slovak film database ČSFD",
      "label": "ČSFD person ID"
    },
    "qid": {
      "description": "Japanese filmmaker",
      "label": "Ryuhei Kitamura"
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
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200)"
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
    "property_id": "P2605"
  },
  "property": "P2605",
  "qid": "Q2088074",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P2605",
    "report_violation_type": "Label in cs language",
    "report_violation_type_normalized": "Label in cs language",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Label in cs language"
  }
}
Example 1 expected JSON output:
{
  "case_id": "reform_Q2088074_P2605_2351781179",
  "confidence": "high",
  "predicted_track": "T_BOX",
  "rationale": "The demonstration answer uses this dev example's historical repair locus."
}

Example 2 input:
{
  "id": "reform_Q4109733_P480_2387684055",
  "labels_en": {
    "property": {
      "description": "FilmAffinity identification number of a creative work",
      "label": "FilmAffinity film ID"
    },
    "qid": {
      "description": "1929 film by Lev Kuleshov",
      "label": "The Happy Canary"
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
            "property_id": "P2316",
            "property_label": "constraint status",
            "values": [
              {
                "label": "mandatory constraint",
                "qid": "Q21502408",
                "raw": "Q21502408"
              }
            ]
          },
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
        "rule_summary": "constraint status (P2316): mandatory constraint (Q21502408); property scope (P5314): as main value (Q54828448), as reference (Q54828450)"
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
    "property_id": "P480"
  },
  "property": "P480",
  "qid": "Q4109733",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P480",
    "report_violation_type": "Label in es language",
    "report_violation_type_normalized": "Label in es language",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Label in es language"
  }
}
Example 2 expected JSON output:
{
  "case_id": "reform_Q4109733_P480_2387684055",
  "confidence": "high",
  "predicted_track": "T_BOX",
  "rationale": "The demonstration answer uses this dev example's historical repair locus."
}

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

```

## prompt_dev_006_hybrid_json_nl_matched_2shot_logic_only_repair_proposal_oracle_no_abstain / reform_Q100259828_P476_2417146821

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `matched_2shot`
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

## prompt_dev_007_hybrid_json_nl_matched_2shot_local_graph_track_diagnosis_diagnosis_no_abstain / reform_Q100259828_P476_2417146821

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `matched_2shot`
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

Example 1 input:
{
  "id": "reform_Q2088074_P2605_2351781179",
  "labels_en": {
    "property": {
      "description": "identifier for a person in the Czecho-Slovak film database ČSFD",
      "label": "ČSFD person ID"
    },
    "qid": {
      "description": "Japanese filmmaker",
      "label": "Ryuhei Kitamura"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "Japanese filmmaker",
      "label": "Ryuhei Kitamura",
      "qid": "Q2088074",
      "sitelinks_count": 15
    },
    "L2_labels": {
      "entities": {
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P2605": {
          "description": "identifier for a person in the Czecho-Slovak film database ČSFD",
          "label": "ČSFD person ID"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q2088074": {
          "description": "Japanese filmmaker",
          "label": "Ryuhei Kitamura"
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
                }
              ]
            }
          ],
          "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200)"
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
      "property_id": "P2605"
    }
  },
  "property": "P2605",
  "qid": "Q2088074",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P2605",
    "report_violation_type": "Label in cs language",
    "report_violation_type_normalized": "Label in cs language",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Label in cs language"
  }
}
Example 1 expected JSON output:
{
  "case_id": "reform_Q2088074_P2605_2351781179",
  "confidence": "high",
  "predicted_track": "T_BOX",
  "rationale": "The demonstration answer uses this dev example's historical repair locus."
}

Example 2 input:
{
  "id": "reform_Q4109733_P480_2387684055",
  "labels_en": {
    "property": {
      "description": "FilmAffinity identification number of a creative work",
      "label": "FilmAffinity film ID"
    },
    "qid": {
      "description": "1929 film by Lev Kuleshov",
      "label": "The Happy Canary"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "1929 film by Lev Kuleshov",
      "label": "The Happy Canary",
      "qid": "Q4109733",
      "sitelinks_count": 10
    },
    "L2_labels": {
      "entities": {
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P2316": {
          "description": "qualifier to define a property constraint in combination with P2302. Use values \"mandatory constraint\" or \"suggestion constraint\"",
          "label": "constraint status"
        },
        "P480": {
          "description": "FilmAffinity identification number of a creative work",
          "label": "FilmAffinity film 
```

## prompt_dev_008_hybrid_json_nl_matched_2shot_local_graph_repair_proposal_oracle_no_abstain / reform_Q100259828_P476_2417146821

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `matched_2shot`
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

## prompt_dev_009_pure_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / reform_Q100259828_P476_2417146821

- Task: `track_diagnosis`
- Representation: `pure_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: pure_nl

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

Input case description:
Case id: reform_Q100259828_P476_2417146821
Focus entity: Q100259828 ({'label': 'Commission Directive 2002/36/EC of 29 April 2002', 'description': 'European Union Directive (EU) 2002/36'})
Target property: P476 ({'label': 'CELEX number', 'description': 'identifier for European legal texts in EUR-Lex database'})
Visible violation report context:
{
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
Visible rule and constraint context:
{
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
```

## prompt_dev_010_pure_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / reform_Q100259828_P476_2417146821

- Task: `t_box_repair`
- Representation: `pure_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: pure_nl

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

Input case description:
Case id: reform_Q100259828_P476_2417146821
Focus entity: Q100259828 ({'label': 'Commission Directive 2002/36/EC of 29 April 2002', 'description': 'European Union Directive (EU) 2002/36'})
Target property: P476 ({'label': 'CELEX number', 'description': 'identifier for European legal texts in EUR-Lex database'})
Visible violation report context:
{
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
Visible rule and constraint context:
{
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
```

## prompt_dev_011_pure_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / reform_Q100259828_P476_2417146821

- Task: `track_diagnosis`
- Representation: `pure_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: pure_nl

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

Input case description:
Case id: reform_Q100259828_P476_2417146821
Focus entity: Q100259828 ({'label': 'Commission Directive 2002/36/EC of 29 April 2002', 'description': 'European Union Directive (EU) 2002/36'})
Target property: P476 ({'label': 'CELEX number', 'description': 'identifier for European legal texts in EUR-Lex database'})
Visible violation report context:
{
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
Visible local graph and constraint context:
{
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
}
```

## prompt_dev_012_pure_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / reform_Q100259828_P476_2417146821

- Task: `t_box_repair`
- Representation: `pure_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences.
```

User prompt:
```text
Prompt version: prompt_dev_v1

Representation: pure_nl

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

Input case description:
Case id: reform_Q100259828_P476_2417146821
Focus entity: Q100259828 ({'label': 'Commission Directive 2002/36/EC of 29 April 2002', 'description': 'European Union Directive (EU) 2002/36'})
Target property: P476 ({'label': 'CELEX number', 'description': 'identifier for European legal texts in EUR-Lex database'})
Visible violation report context:
{
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
Visible local graph and constraint context:
{
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
}
```
