# Prompt Development Review

No LLM inference was run for this artifact.

Rendered prompts: `192`

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / case_000429

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v2

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely. A constraint report alone does not imply T_BOX; decide based on the likely repair locus.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Decision rule:
- Diagnose the likely repair locus, not the vocabulary of the report.
- A constraint report alone does not imply T_BOX. If the likely fix is to change, remove, or normalize the focus
  entity's claim value, choose A_BOX.
- Choose T_BOX only when the visible evidence supports changing the property constraint/schema itself.
- Choose AMBIGUOUS when both claim repair and schema reform remain plausible from the visible evidence.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000429",
  "labels_en": {
    "property": {
      "description": "(main or final) manufacturer or producer of this product",
      "label": "manufacturer"
    },
    "qid": {
      "description": "historical fort near Sann, Jamshoro District, Sindh, Pakistan",
      "label": "Ranikot Fort"
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
                "label": "as qualifier",
                "qid": "Q54828449",
                "raw": "Q54828449"
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449)"
      }
    ],
    "property_id": "P176"
  },
  "property": "P176",
  "qid": "Q2131024",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P176",
    "report_violation_type": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
    "report_violation_type_normalized": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
    "report_violation_type_qids": [
      "Q43229",
      "Q5",
      "Q95074",
      "Q14514600",
      "Q1294787",
      "Q28640",
      "Q729",
      "Q268592",
      "Q83405",
      "Q12737077",
      "Q656720",
      "Q16521"
    ],
    "report_violation_type_raw": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
    "value": [
      "Q7366810"
    ],
    "value_descriptions_en": [
      null
    ],
    "value_labels_en": [
      null
    ]
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000429

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v2

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Choose values only from visible target-value evidence. Preserve useful values when the evidence supports them; do not over-delete just to satisfy a rule.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "target": {"qid": "Q...", "pid": "P..."},
  "ops": [
    {
      "op": "SET" | "ADD" | "REMOVE" | "DELETE_ALL",
      "pid": "P...",
      "value": "Q..." | "<literal>" | 123,
      "rank": "normal" | "preferred" | "deprecated"
    }
  ],
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}
Value-source rules:
- Replacement values must come from visible old-value normalization, visible local evidence, retained values, or
  explicit prompt evidence for the target value.
- Do not use constraint-family QIDs, allowed-type QIDs, report type QIDs, or constraint class QIDs as replacement claim
  values unless that QID is explicitly visible as the target claim value evidence.
- Do not invent a new entity value. If no replacement value is visible, prefer REMOVE or DELETE_ALL only when the
  evidence supports an empty final target property.

Operation rubric:
- Use SET when the final target property should contain exactly one visible value.
- Use ADD only to add a visible missing value while preserving existing retained values.
- Use REMOVE only to remove a specific visible bad value while preserving other retained values.
- Use DELETE_ALL only when the final target property should be empty.
- Preserve retained values. Do not over-delete merely to satisfy a constraint.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000429",
  "labels_en": {
    "property": {
      "description": "(main or final) manufacturer or producer of this product",
      "label": "manufacturer"
    },
    "qid": {
      "description": "historical fort near Sann, Jamshoro District, Sindh, Pakistan",
      "label": "Ranikot Fort"
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
                "label": "as qualifier",
                "qid": "Q54828449",
                "raw": "Q54828449"
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449)"
      }
    ],
    "property_id": "P176"
  },
  "property": "P176",
  "qid": "Q2131024",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P176",
    "report_violation_type": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
    "report_violation_type_normalized": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
    "report_violation_type_qids": [
      "Q43229",
      "Q5",
      "Q95074",
      "Q14514600",
      "Q1294787",
      "Q28640",
      "Q729",
      "Q268592",
      "Q83405",
      "Q12737077",
      "Q656720",
      "Q16521"
    ],
    "report_violation_type_raw": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
    "value": [
      "Q7366810"
    ],
    "value_descriptions_en": [
      null
    ],
    "value_labels_en": [
      null
    ]
  }
}
```

## prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000429

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v2

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely. A constraint report alone does not imply T_BOX; decide based on the likely repair locus.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Decision rule:
- Diagnose the likely repair locus, not the vocabulary of the report.
- A constraint report alone does not imply T_BOX. If the likely fix is to change, remove, or normalize the focus
  entity's claim value, choose A_BOX.
- Choose T_BOX only when the visible evidence supports changing the property constraint/schema itself.
- Choose AMBIGUOUS when both claim repair and schema reform remain plausible from the visible evidence.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000429",
  "labels_en": {
    "property": {
      "description": "(main or final) manufacturer or producer of this product",
      "label": "manufacturer"
    },
    "qid": {
      "description": "historical fort near Sann, Jamshoro District, Sindh, Pakistan",
      "label": "Ranikot Fort"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "historical fort near Sann, Jamshoro District, Sindh, Pakistan",
      "label": "Ranikot Fort",
      "properties": {
        "P176": [
          "Q7366810"
        ]
      },
      "qid": "Q2131024"
    },
    "L2_labels": {
      "entities": {
        "P176": {
          "description": "(main or final) manufacturer or producer of this product",
          "label": "manufacturer"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q2131024": {
          "description": "historical fort near Sann, Jamshoro District, Sindh, Pakistan",
          "label": "Ranikot Fort"
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
                  "label": "as qualifier",
                  "qid": "Q54828449",
                  "raw": "Q54828449"
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449)"
        }
      ],
      "property_id": "P176"
    }
  },
  "property": "P176",
  "qid": "Q2131024",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P176",
    "report_violation_type": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
    "report_violation_type_normalized": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
    "report_violation_type_qids": [
      "Q43229",
      "Q5",
      "Q95074",
      "Q14514600",
      "Q1294787",
      "Q28640",
      "Q729",
      "Q268592",
      "Q83405",
      "Q12737077",
      "Q656720",
      "Q16521"
    ],
    "report_violation_type_raw": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28640, Q|729, Q|268592, Q|83405, Q|12737077, Q|656720, Q|16521",
    "value": [
      "Q7366810"
    ],
    "value_descriptions_en": [
      null
    ],
    "value_labels
```

## prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000429

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v2

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Choose values only from visible target-value evidence. Preserve useful values when the evidence supports them; do not over-delete just to satisfy a rule.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "target": {"qid": "Q...", "pid": "P..."},
  "ops": [
    {
      "op": "SET" | "ADD" | "REMOVE" | "DELETE_ALL",
      "pid": "P...",
      "value": "Q..." | "<literal>" | 123,
      "rank": "normal" | "preferred" | "deprecated"
    }
  ],
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}
Value-source rules:
- Replacement values must come from visible old-value normalization, visible local evidence, retained values, or
  explicit prompt evidence for the target value.
- Do not use constraint-family QIDs, allowed-type QIDs, report type QIDs, or constraint class QIDs as replacement claim
  values unless that QID is explicitly visible as the target claim value evidence.
- Do not invent a new entity value. If no replacement value is visible, prefer REMOVE or DELETE_ALL only when the
  evidence supports an empty final target property.

Operation rubric:
- Use SET when the final target property should contain exactly one visible value.
- Use ADD only to add a visible missing value while preserving existing retained values.
- Use REMOVE only to remove a specific visible bad value while preserving other retained values.
- Use DELETE_ALL only when the final target property should be empty.
- Preserve retained values. Do not over-delete merely to satisfy a constraint.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000429",
  "labels_en": {
    "property": {
      "description": "(main or final) manufacturer or producer of this product",
      "label": "manufacturer"
    },
    "qid": {
      "description": "historical fort near Sann, Jamshoro District, Sindh, Pakistan",
      "label": "Ranikot Fort"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "historical fort near Sann, Jamshoro District, Sindh, Pakistan",
      "label": "Ranikot Fort",
      "properties": {
        "P176": [
          "Q7366810"
        ]
      },
      "qid": "Q2131024"
    },
    "L2_labels": {
      "entities": {
        "P176": {
          "description": "(main or final) manufacturer or producer of this product",
          "label": "manufacturer"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q2131024": {
          "description": "historical fort near Sann, Jamshoro District, Sindh, Pakistan",
          "label": "Ranikot Fort"
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
                  "label": "as qualifier",
                  "qid": "Q54828449",
                  "raw": "Q54828449"
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449)"
        }
      ],
      "property_id": "P176"
    }
  },
  "property": "P176",
  "qid": "Q2131024",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P176",
    "report_violation_type": "Value type Q|43229, Q|5, Q|95074, Q|14514600, Q|1294787, Q|28
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / case_000138

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v2

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely. A constraint report alone does not imply T_BOX; decide based on the likely repair locus.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Decision rule:
- Diagnose the likely repair locus, not the vocabulary of the report.
- A constraint report alone does not imply T_BOX. If the likely fix is to change, remove, or normalize the focus
  entity's claim value, choose A_BOX.
- Choose T_BOX only when the visible evidence supports changing the property constraint/schema itself.
- Choose AMBIGUOUS when both claim repair and schema reform remain plausible from the visible evidence.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000138",
  "labels_en": {
    "property": {
      "description": "winner of a competition or similar event, not to be used from the awardees record (instead use \"award received\" (P166), possibly qualified with \"for work\" (P1686)) nor for wars or battles",
      "label": "winner"
    },
    "qid": {
      "description": "water polo league season",
      "label": "1928 Országos Bajnokság I"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q21503250",
        "label": "subject type constraint"
      },
      {
        "constraint_qid": "Q21502838",
        "label": "conflicts-with constraint"
      },
      {
        "constraint_qid": "Q21510855",
        "label": "inverse constraint"
      },
      {
        "constraint_qid": "Q21510865",
        "label": "value-type constraint"
      },
      {
        "constraint_qid": "Q19474404",
        "label": "single-value constraint"
      },
      {
        "constraint_qid": "Q52004125",
        "label": "allowed-entity-types constraint"
      },
      {
        "constraint_qid": "Q53869507",
        "label": "property scope constraint"
      },
      {
        "constraint_qid": "Q21510851",
        "label": "allowed qualifiers constraint"
      }
    ],
    "temporal_policy": "compact_inventory_no_pre_change_signature",
    "violation_context": {
      "report_page_title": "Wikidata:Database reports/Constraint violations/P1346",
      "report_violation_type": "Inverse",
      "report_violation_type_normalized": "Inverse",
      "report_violation_type_qids": [],
      "report_violation_type_raw": "Inverse"
    }
  },
  "property": "P1346",
  "qid": "Q254745",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P1346",
    "report_violation_type": "Inverse",
    "report_violation_type_normalized": "Inverse",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Inverse"
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000138

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v2

Representation: hybrid_json_nl

Task: t_box_repair

Propose an executable T-box schema reform for the focus property. Use constraint-family QIDs from the supplied context as constraint_type_qid values. Do not copy ordinary entity/type QIDs into constraint-family fields. Do not treat report_violation_type_qids as the repaired constraint signature unless the same QIDs are visible changed constraint values. Prefer SCHEMA_UPDATE with low confidence when exact direction or post-reform values are not visible.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
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
  "provenance": [{"kind": "KG" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}
Do not put report_violation_type_qids into signature_after unless those QIDs are visible semantic changed constraint
values in the supplied constraint context. If exact post-reform schema values are not visible, choose action
SCHEMA_UPDATE with low confidence rather than inventing an exact signature.

Action decision tree:
- RELAXATION_SET_EXPANSION: visible evidence shows the allowed set became larger.
- RESTRICTION_SET_CONTRACTION: visible evidence shows the allowed set became smaller.
- RELAXATION_RANGE_WIDENED: visible evidence shows numeric/date bounds became wider.
- RESTRICTION_RANGE_NARROWED: visible evidence shows numeric/date bounds became narrower.
- SCHEMA_UPDATE: the schema changed but exact direction or post-reform values are not visible.
- COINCIDENTAL_SCHEMA_CHANGE: a schema change is visible but the evidence does not support a causal repair for the
  reported violation.

Signature discipline:
- Do not invent a full signature_after.
- Do not copy violating item/type QIDs or report_violation_type_qids into signature_after unless they are visible
  changed constraint values.
- If the prompt exposes compact_inventory_no_pre_change_signature, exact post-reform values are not visible; prefer
  SCHEMA_UPDATE with low confidence and an empty signature_after unless a changed constraint value is explicitly shown.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000138",
  "labels_en": {
    "property": {
      "description": "winner of a competition or similar event, not to be used from the awardees record (instead use \"award received\" (P166), possibly qualified with \"for work\" (P1686)) nor for wars or battles",
      "label": "winner"
    },
    "qid": {
      "description": "water polo league season",
      "label": "1928 Országos Bajnokság I"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q21503250",
        "label": "subject type constraint"
      },
      {
        "constraint_qid": "Q21502838",
        "label": "conflicts-with constraint"
      },
      {
        "constraint_qid": "Q21510855",
        "label": "inverse constraint"
      },
      {
        "constraint_qid": "Q21510865",
        "label": "value-type constraint"
      },
      {
        "constraint_qid": "Q19474404",
        "label": "single-value constraint"
      },
      {
        "constraint_qid": "Q52004125",
        "label": "allowed-entity-types constraint"
      },
      {
        "constraint_qid": "Q53869507",
        "label": "property scope constraint"
      },
      {
        "constraint_qid": "Q21510851",
        "label": "allowed qualifiers constraint"
      }
    ],
    "temporal_policy": "compact_inventory_no_pre_change_signature",
    "violation_context": {
      "report_page_title": "Wikidata:Database reports/Constraint violations/P1346",
      "report_violation_type": "Inverse",
      "report_violation_type_normalized": "Inverse",
      "report_violation_type_qids": [],
      "report_violation_type_raw": "Inverse"
    }
  },
  "property": "P1346",
  "qid": "Q254745",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P1346",
    "report_violation_type": "Inverse",
    "report_violation_type_normalized": "Inverse",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Inverse"
  }
}
```

## prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000138

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v2

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely. A constraint report alone does not imply T_BOX; decide based on the likely repair locus.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Decision rule:
- Diagnose the likely repair locus, not the vocabulary of the report.
- A constraint report alone does not imply T_BOX. If the likely fix is to change, remove, or normalize the focus
  entity's claim value, choose A_BOX.
- Choose T_BOX only when the visible evidence supports changing the property constraint/schema itself.
- Choose AMBIGUOUS when both claim repair and schema reform remain plausible from the visible evidence.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000138",
  "labels_en": {
    "property": {
      "description": "winner of a competition or similar event, not to be used from the awardees record (instead use \"award received\" (P166), possibly qualified with \"for work\" (P1686)) nor for wars or battles",
      "label": "winner"
    },
    "qid": {
      "description": "water polo league season",
      "label": "1928 Országos Bajnokság I"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "water polo league season",
      "label": "1928 Országos Bajnokság I",
      "qid": "Q254745"
    },
    "L2_labels": {
      "entities": {
        "P1346": {
          "description": "winner of a competition or similar event, not to be used from the awardees record (instead use \"award received\" (P166), possibly qualified with \"for work\" (P1686)) nor for wars or battles",
          "label": "winner"
        },
        "Q19474404": {
          "description": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "label": "single-value constraint"
        },
        "Q21502838": {
          "description": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "label": "conflicts-with constraint"
        },
        "Q21503250": {
          "description": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "label": "subject type constraint"
        },
        "Q21510851": {
          "description": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "label": "allowed qualifiers constraint"
        },
        "Q21510855": {
          "description": "type of constraint for Wikidata properties: used to specify that the referenced item has to refer back to this item with the given inverse property",
          "label": "inverse constraint"
        },
        "Q21510865": {
          "description": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "label": "value-type constraint"
        },
        "Q254745": {
          "description": "water polo league season",
          "label": "1928 Országos Bajnokság I"
        },
        "Q52004125": {
          "description": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "label": "allowed-entity-types constraint"
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
          "constraint_qid": "Q21503250",
          "label": "subject type constraint"
        },
        {
          "constraint_qid": "Q21502838",
          "label": "conflicts-with constraint"
        },
        {
          "constraint_qid": "Q21510855",
          "label": "inverse constraint"
        },
        {
          "constraint_qid": "Q21510865",
          "label": "value-type constraint"
        },
        {
          "constraint_qid": "Q19474404",
          "label": "single-value constraint"
        },
        {
          "constraint_qid": "Q52004125",
          "label": "allowed-entity-types constraint"
        },
        {
          "constraint_qid": "Q53869507",
          "label": "property scope constraint"
        },
        {
          "constraint_qid": "Q21510851",
          "label": "allowed qualifiers constraint"
        }
      ],
      "temporal_policy": "compact_inventory_no_pre_change_signature",
      "violation_context": {
        "report_page_title": "Wikidata:Database reports/Constraint violations/P1346",
        "report_violation_type": "Inverse",
        "report_violation_type_normalized": "Inverse",
        "report_violation_type_qids": [],
        "report_violation_type_raw": "Inverse"
      }
    }
  },
  "property": "P1346",
  "qid": "Q254745",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P1346",
    "report_violation_type": "Inverse",
    "report_violation_type_normalized": "Inverse",
    "report_violation_type_
```

## prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000138

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v2

Representation: hybrid_json_nl

Task: t_box_repair

Propose an executable T-box schema reform for the focus property. Use constraint-family QIDs from the supplied context as constraint_type_qid values. Do not copy ordinary entity/type QIDs into constraint-family fields. Do not treat report_violation_type_qids as the repaired constraint signature unless the same QIDs are visible changed constraint values. Prefer SCHEMA_UPDATE with low confidence when exact direction or post-reform values are not visible.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
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
  "provenance": [{"kind": "KG" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}
Do not put report_violation_type_qids into signature_after unless those QIDs are visible semantic changed constraint
values in the supplied constraint context. If exact post-reform schema values are not visible, choose action
SCHEMA_UPDATE with low confidence rather than inventing an exact signature.

Action decision tree:
- RELAXATION_SET_EXPANSION: visible evidence shows the allowed set became larger.
- RESTRICTION_SET_CONTRACTION: visible evidence shows the allowed set became smaller.
- RELAXATION_RANGE_WIDENED: visible evidence shows numeric/date bounds became wider.
- RESTRICTION_RANGE_NARROWED: visible evidence shows numeric/date bounds became narrower.
- SCHEMA_UPDATE: the schema changed but exact direction or post-reform values are not visible.
- COINCIDENTAL_SCHEMA_CHANGE: a schema change is visible but the evidence does not support a causal repair for the
  reported violation.

Signature discipline:
- Do not invent a full signature_after.
- Do not copy violating item/type QIDs or report_violation_type_qids into signature_after unless they are visible
  changed constraint values.
- If the prompt exposes compact_inventory_no_pre_change_signature, exact post-reform values are not visible; prefer
  SCHEMA_UPDATE with low confidence and an empty signature_after unless a changed constraint value is explicitly shown.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000138",
  "labels_en": {
    "property": {
      "description": "winner of a competition or similar event, not to be used from the awardees record (instead use \"award received\" (P166), possibly qualified with \"for work\" (P1686)) nor for wars or battles",
      "label": "winner"
    },
    "qid": {
      "description": "water polo league season",
      "label": "1928 Országos Bajnokság I"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "water polo league season",
      "label": "1928 Országos Bajnokság I",
      "qid": "Q254745"
    },
    "L2_labels": {
      "entities": {
        "P1346": {
          "description": "winner of a competition or similar event, not to be used from the awardees record (instead use \"award received\" (P166), possibly qualified with \"for work\" (P1686)) nor for wars or battles",
          "label": "winner"
        },
        "Q19474404": {
          "description": "type of constraint for Wikidata properties: used to specify that this property generally contains a single value per item",
          "label": "single-value constraint"
        },
        "Q21502838": {
          "description": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "label": "conflicts-with constraint"
        },
        "Q21503250": {
          "description": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "label": "subject type constraint"
        },
        "Q21510851": {
          "description": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "label": "allowed qualifiers constraint"
        },
        "Q21510855": {
          "description": "type of constraint for Wikidata properties: used to specify that the referenced item has to refer back to this item with the given inverse property",
          "label": "inverse constraint"
        },
        "Q21510865": {
          "description": "type of constraint for Wikidata properties: used to specify that the value item should be a subclass or instance of a given type",
          "label": "value-type constraint"
        },
        "Q254745": {
          "description": "water polo league season",
          "label": "1928 Országos Bajnokság I"
        },
        "Q52004125": {
          "description": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "label": "allowed-entity-types constraint"
        },
        "Q53869507": {
          "description": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "label": "property scope constraint"
        }
      }
    },
    "L3_neighborhood": {
      "outg
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / case_000487

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v2

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely. A constraint report alone does not imply T_BOX; decide based on the likely repair locus.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Decision rule:
- Diagnose the likely repair locus, not the vocabulary of the report.
- A constraint report alone does not imply T_BOX. If the likely fix is to change, remove, or normalize the focus
  entity's claim value, choose A_BOX.
- Choose T_BOX only when the visible evidence supports changing the property constraint/schema itself.
- Choose AMBIGUOUS when both claim repair and schema reform remain plausible from the visible evidence.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000487",
  "labels_en": {
    "property": {
      "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
      "label": "SureChEMBL ID"
    },
    "qid": {
      "description": "antidepressant of the selective serotonin reuptake inhibitor (SSRI) class",
      "label": "escitalopram"
    }
  },
  "logic_context": {
    "constraints": [
      {
        "constraint_type": {
          "label": "format constraint",
          "qid": "Q21502404"
        },
        "qualifiers": [
          {
            "property_id": "P1793",
            "property_label": "format as a regular expression",
            "values": [
              {
                "raw": "\\d+"
              }
            ]
          },
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
          }
        ],
        "rule_summary": "format as a regular expression (P1793): \\d+; constraint status (P2316): mandatory constraint (Q21502408)"
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
    "property_id": "P2877"
  },
  "property": "P2877",
  "qid": "Q423757",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
    "report_violation_type": "Format",
    "report_violation_type_normalized": "Format",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Format",
    "value": [
      "SCHEMBL34948"
    ]
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000487

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v2

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Choose values only from visible target-value evidence. Preserve useful values when the evidence supports them; do not over-delete just to satisfy a rule.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "target": {"qid": "Q...", "pid": "P..."},
  "ops": [
    {
      "op": "SET" | "ADD" | "REMOVE" | "DELETE_ALL",
      "pid": "P...",
      "value": "Q..." | "<literal>" | 123,
      "rank": "normal" | "preferred" | "deprecated"
    }
  ],
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}
Value-source rules:
- Replacement values must come from visible old-value normalization, visible local evidence, retained values, or
  explicit prompt evidence for the target value.
- Do not use constraint-family QIDs, allowed-type QIDs, report type QIDs, or constraint class QIDs as replacement claim
  values unless that QID is explicitly visible as the target claim value evidence.
- Do not invent a new entity value. If no replacement value is visible, prefer REMOVE or DELETE_ALL only when the
  evidence supports an empty final target property.

Operation rubric:
- Use SET when the final target property should contain exactly one visible value.
- Use ADD only to add a visible missing value while preserving existing retained values.
- Use REMOVE only to remove a specific visible bad value while preserving other retained values.
- Use DELETE_ALL only when the final target property should be empty.
- Preserve retained values. Do not over-delete merely to satisfy a constraint.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000487",
  "labels_en": {
    "property": {
      "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
      "label": "SureChEMBL ID"
    },
    "qid": {
      "description": "antidepressant of the selective serotonin reuptake inhibitor (SSRI) class",
      "label": "escitalopram"
    }
  },
  "logic_context": {
    "constraints": [
      {
        "constraint_type": {
          "label": "format constraint",
          "qid": "Q21502404"
        },
        "qualifiers": [
          {
            "property_id": "P1793",
            "property_label": "format as a regular expression",
            "values": [
              {
                "raw": "\\d+"
              }
            ]
          },
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
          }
        ],
        "rule_summary": "format as a regular expression (P1793): \\d+; constraint status (P2316): mandatory constraint (Q21502408)"
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
    "property_id": "P2877"
  },
  "property": "P2877",
  "qid": "Q423757",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
    "report_violation_type": "Format",
    "report_violation_type_normalized": "Format",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Format",
    "value": [
      "SCHEMBL34948"
    ]
  }
}
```

## prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000487

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v2

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely. A constraint report alone does not imply T_BOX; decide based on the likely repair locus.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Decision rule:
- Diagnose the likely repair locus, not the vocabulary of the report.
- A constraint report alone does not imply T_BOX. If the likely fix is to change, remove, or normalize the focus
  entity's claim value, choose A_BOX.
- Choose T_BOX only when the visible evidence supports changing the property constraint/schema itself.
- Choose AMBIGUOUS when both claim repair and schema reform remain plausible from the visible evidence.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000487",
  "labels_en": {
    "property": {
      "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
      "label": "SureChEMBL ID"
    },
    "qid": {
      "description": "antidepressant of the selective serotonin reuptake inhibitor (SSRI) class",
      "label": "escitalopram"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "antidepressant of the selective serotonin reuptake inhibitor (SSRI) class",
      "label": "escitalopram",
      "properties": {
        "P2877": [
          "SCHEMBL34948"
        ]
      },
      "qid": "Q423757"
    },
    "L2_labels": {
      "entities": {
        "P1793": {
          "description": "regex describing an identifier or a Wikidata property. When using on property constraints, ensure syntax is a PCRE",
          "label": "format as a regular expression"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P2316": {
          "description": "qualifier to define a property constraint in combination with P2302. Use values \"mandatory constraint\" or \"suggestion constraint\"",
          "label": "constraint status"
        },
        "P2877": {
          "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
          "label": "SureChEMBL ID"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q21502404": {
          "description": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "label": "format constraint"
        },
        "Q21502408": {
          "description": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
          "label": "mandatory constraint"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q423757": {
          "description": "antidepressant of the selective serotonin reuptake inhibitor (SSRI) class",
          "label": "escitalopram"
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
            "label": "format constraint",
            "qid": "Q21502404"
          },
          "qualifiers": [
            {
              "property_id": "P1793",
              "property_label": "format as a regular expression",
              "values": [
                {
                  "raw": "\\d+"
                }
              ]
            },
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
            }
          ],
          "rule_summary": "format as a regular expression (P1793): \\d+; constraint status (P2316): mandatory constraint (Q21502408)"
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
                  "raw": "
```

## prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000487

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v2

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Choose values only from visible target-value evidence. Preserve useful values when the evidence supports them; do not over-delete just to satisfy a rule.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "target": {"qid": "Q...", "pid": "P..."},
  "ops": [
    {
      "op": "SET" | "ADD" | "REMOVE" | "DELETE_ALL",
      "pid": "P...",
      "value": "Q..." | "<literal>" | 123,
      "rank": "normal" | "preferred" | "deprecated"
    }
  ],
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}
Value-source rules:
- Replacement values must come from visible old-value normalization, visible local evidence, retained values, or
  explicit prompt evidence for the target value.
- Do not use constraint-family QIDs, allowed-type QIDs, report type QIDs, or constraint class QIDs as replacement claim
  values unless that QID is explicitly visible as the target claim value evidence.
- Do not invent a new entity value. If no replacement value is visible, prefer REMOVE or DELETE_ALL only when the
  evidence supports an empty final target property.

Operation rubric:
- Use SET when the final target property should contain exactly one visible value.
- Use ADD only to add a visible missing value while preserving existing retained values.
- Use REMOVE only to remove a specific visible bad value while preserving other retained values.
- Use DELETE_ALL only when the final target property should be empty.
- Preserve retained values. Do not over-delete merely to satisfy a constraint.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000487",
  "labels_en": {
    "property": {
      "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
      "label": "SureChEMBL ID"
    },
    "qid": {
      "description": "antidepressant of the selective serotonin reuptake inhibitor (SSRI) class",
      "label": "escitalopram"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "antidepressant of the selective serotonin reuptake inhibitor (SSRI) class",
      "label": "escitalopram",
      "properties": {
        "P2877": [
          "SCHEMBL34948"
        ]
      },
      "qid": "Q423757"
    },
    "L2_labels": {
      "entities": {
        "P1793": {
          "description": "regex describing an identifier or a Wikidata property. When using on property constraints, ensure syntax is a PCRE",
          "label": "format as a regular expression"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P2316": {
          "description": "qualifier to define a property constraint in combination with P2302. Use values \"mandatory constraint\" or \"suggestion constraint\"",
          "label": "constraint status"
        },
        "P2877": {
          "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
          "label": "SureChEMBL ID"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q21502404": {
          "description": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "label": "format constraint"
        },
        "Q21502408": {
          "description": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
          "label": "mandatory constraint"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q423757": {
          "description": "antidepressant of the selective serotonin reuptake inhibitor (SSRI) class",
          "label": "escitalopram"
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
            "label": "format constraint",
            "qid": "Q21502404"
          },
          "qualifiers": [
            {
              "property_id": "P1793",
              "property_label": "format as a regular expression",
              "values": [
                {
                  "raw": "\\d+"
                }
              ]
            },
            {
              "property_id": "P2316",
              "property_label": "constraint status",
              "values": [
                {
                  "label": "mandatory
```
