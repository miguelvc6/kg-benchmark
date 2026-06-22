# Prompt Development Review

No LLM inference was run for this artifact.

Rendered prompts: `512`

## Example Schema Summary

| Task | Example count | Example schemas |
| --- | ---: | --- |
| `a_box_repair` | 0 | n/a |

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000001

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Use only visible evidence and the output contract.

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
Field definitions:
- target.qid is the focus entity identifier from the input.
- target.pid is the target property identifier from the input.
- ops is the ordered set of claim edits to the target property on the focus entity.
- SET replaces the target property's value set with the supplied value.
- ADD adds the supplied value to the target property.
- REMOVE removes the supplied value from the target property.
- DELETE_ALL removes all values for the target property.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Replacement claim values must be ordinary claim values, not constraint-family identifiers, unless the prompt visibly
  presents that identifier as the claim value itself.
- Constraint-family QIDs, report-type QIDs, allowed-type QIDs, and ordinary entity/type QIDs have different roles; keep
  those roles distinct.
- Do not use hidden benchmark classes, subtypes, or historical labels.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000001",
  "labels_en": {
    "property": {
      "description": "mathematical formula representing a theorem or law",
      "label": "defining formula"
    },
    "qid": {
      "description": "graph whose edges are labelled with irreducible representations of a compact Lie group and whose vertices are associated with intertwiners of the edge representations adjacent to it",
      "label": "spin network"
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
    "property_id": "P2534"
  },
  "property": "P2534",
  "qid": "Q654081",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P2534",
    "report_violation_type": "Type Q|24034552, Q|33104303, Q|408891, Q|246672, Q|126473023, Q|126180647, Q|1140046",
    "report_violation_type_normalized": "Type Q|24034552, Q|33104303, Q|408891, Q|246672, Q|126473023, Q|126180647, Q|1140046",
    "report_violation_type_qids": [
      "Q24034552",
      "Q33104303",
      "Q408891",
      "Q246672",
      "Q126473023",
      "Q126180647",
      "Q1140046"
    ],
    "report_violation_type_raw": "Type Q|24034552, Q|33104303, Q|408891, Q|246672, Q|126473023, Q|126180647, Q|1140046",
    "value": [
      "M_\\odot = \\frac{4 \\pi^2 \\times (1\\,\\mathrm{AU})^3}{G \\times (1\\,\\mathrm{yr})^2}"
    ]
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000001

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Use only visible evidence and the output contract.

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
Field definitions:
- target.qid is the focus entity identifier from the input.
- target.pid is the target property identifier from the input.
- ops is the ordered set of claim edits to the target property on the focus entity.
- SET replaces the target property's value set with the supplied value.
- ADD adds the supplied value to the target property.
- REMOVE removes the supplied value from the target property.
- DELETE_ALL removes all values for the target property.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Replacement claim values must be ordinary claim values, not constraint-family identifiers, unless the prompt visibly
  presents that identifier as the claim value itself.
- Constraint-family QIDs, report-type QIDs, allowed-type QIDs, and ordinary entity/type QIDs have different roles; keep
  those roles distinct.
- Do not use hidden benchmark classes, subtypes, or historical labels.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000001",
  "labels_en": {
    "property": {
      "description": "mathematical formula representing a theorem or law",
      "label": "defining formula"
    },
    "qid": {
      "description": "graph whose edges are labelled with irreducible representations of a compact Lie group and whose vertices are associated with intertwiners of the edge representations adjacent to it",
      "label": "spin network"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "graph whose edges are labelled with irreducible representations of a compact Lie group and whose vertices are associated with intertwiners of the edge representations adjacent to it",
      "label": "spin network",
      "properties": {
        "P2534": [
          "M_\\odot = \\frac{4 \\pi^2 \\times (1\\,\\mathrm{AU})^3}{G \\times (1\\,\\mathrm{yr})^2}"
        ]
      },
      "qid": "Q654081"
    },
    "L2_labels": {
      "entities": {
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P2534": {
          "description": "mathematical formula representing a theorem or law",
          "label": "defining formula"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
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
        "Q654081": {
          "description": "graph whose edges are labelled with irreducible representations of a compact Lie group and whose vertices are associated with intertwiners of the edge representations adjacent to it",
          "label": "spin network"
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
        
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000002

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Use only visible evidence and the output contract.

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
Field definitions:
- target.qid is the focus entity identifier from the input.
- target.pid is the target property identifier from the input.
- ops is the ordered set of claim edits to the target property on the focus entity.
- SET replaces the target property's value set with the supplied value.
- ADD adds the supplied value to the target property.
- REMOVE removes the supplied value from the target property.
- DELETE_ALL removes all values for the target property.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Replacement claim values must be ordinary claim values, not constraint-family identifiers, unless the prompt visibly
  presents that identifier as the claim value itself.
- Constraint-family QIDs, report-type QIDs, allowed-type QIDs, and ordinary entity/type QIDs have different roles; keep
  those roles distinct.
- Do not use hidden benchmark classes, subtypes, or historical labels.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000002",
  "labels_en": {
    "property": {
      "description": "language(s) that a person or a people speaks, writes or signs, including the native language(s)",
      "label": "languages spoken, written or signed"
    },
    "qid": {
      "description": "Saudi royal",
      "label": "Faisal bin Salman bin Abdulaziz Al Saud"
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
              }
            ]
          },
          {
            "property_id": "P2303",
            "property_label": "exception to constraint",
            "values": [
              {
                "label": "speaker",
                "qid": "Q16657634",
                "raw": "Q16657634"
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448); exception to constraint (P2303): speaker (Q16657634)"
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
    "property_id": "P1412"
  },
  "property": "P1412",
  "qid": "Q5431091",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P1412",
    "report_violation_type": "Value type Q|17376908",
    "report_violation_type_normalized": "Value type Q|17376908",
    "report_violation_type_qids": [
      "Q17376908"
    ],
    "report_violation_type_raw": "Value type Q|17376908",
    "value": [
      "Q4783411"
    ],
    "value_descriptions_en": [
      "Edward Lane's Arabic–English dictionary"
    ],
    "value_labels_en": [
      "An Arabic-English Lexicon"
    ]
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000002

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Use only visible evidence and the output contract.

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
Field definitions:
- target.qid is the focus entity identifier from the input.
- target.pid is the target property identifier from the input.
- ops is the ordered set of claim edits to the target property on the focus entity.
- SET replaces the target property's value set with the supplied value.
- ADD adds the supplied value to the target property.
- REMOVE removes the supplied value from the target property.
- DELETE_ALL removes all values for the target property.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Replacement claim values must be ordinary claim values, not constraint-family identifiers, unless the prompt visibly
  presents that identifier as the claim value itself.
- Constraint-family QIDs, report-type QIDs, allowed-type QIDs, and ordinary entity/type QIDs have different roles; keep
  those roles distinct.
- Do not use hidden benchmark classes, subtypes, or historical labels.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000002",
  "labels_en": {
    "property": {
      "description": "language(s) that a person or a people speaks, writes or signs, including the native language(s)",
      "label": "languages spoken, written or signed"
    },
    "qid": {
      "description": "Saudi royal",
      "label": "Faisal bin Salman bin Abdulaziz Al Saud"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "Saudi royal",
      "label": "Faisal bin Salman bin Abdulaziz Al Saud",
      "properties": {
        "P1412": [
          "Q4783411"
        ]
      },
      "qid": "Q5431091"
    },
    "L2_labels": {
      "entities": {
        "P1412": {
          "description": "language(s) that a person or a people speaks, writes or signs, including the native language(s)",
          "label": "languages spoken, written or signed"
        },
        "P2303": {
          "description": "item that is an exception to the constraint, qualifier to define a property constraint in combination with P2302",
          "label": "exception to constraint"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q16657634": {
          "description": "person who speaks a language",
          "label": "speaker"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q4783411": {
          "description": "Edward Lane's Arabic–English dictionary",
          "label": "An Arabic-English Lexicon"
        },
        "Q52004125": {
          "description": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase MediaInfo",
          "label": "allowed-entity-types constraint"
        },
        "Q53869507": {
          "description": "constraint to define the scope of the property (as main property, as qualifier, as reference, or combination). Qualify with \"property scope\" (P5314)",
          "label": "property scope constraint"
        },
        "Q5431091": {
          "description": "Saudi royal",
          "label": "Faisal bin Salman bin Abdulaziz Al Saud"
        },
        "Q54828448": {
          "description": "property scope type",
          "label": "as main value"
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
                }
              ]
            },
            {
              "property_id": "P2303",
              "property_label": "exception to constraint",
              "values": [
                {
                  "label": "speaker",
                  "qid": "Q16657634",
                  "raw": "Q16657634"
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448); exception to constraint (P2303): speaker (Q16657634)"
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
 
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000003

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Use only visible evidence and the output contract.

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
Field definitions:
- target.qid is the focus entity identifier from the input.
- target.pid is the target property identifier from the input.
- ops is the ordered set of claim edits to the target property on the focus entity.
- SET replaces the target property's value set with the supplied value.
- ADD adds the supplied value to the target property.
- REMOVE removes the supplied value from the target property.
- DELETE_ALL removes all values for the target property.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Replacement claim values must be ordinary claim values, not constraint-family identifiers, unless the prompt visibly
  presents that identifier as the claim value itself.
- Constraint-family QIDs, report-type QIDs, allowed-type QIDs, and ordinary entity/type QIDs have different roles; keep
  those roles distinct.
- Do not use hidden benchmark classes, subtypes, or historical labels.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000003",
  "labels_en": {
    "property": {
      "description": "most specific known birth location of a person, animal or fictional character",
      "label": "place of birth"
    },
    "qid": {
      "description": "Swiss poet (1925-2021)",
      "label": "Philippe Jaccottet"
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
    "property_id": "P19"
  },
  "property": "P19",
  "qid": "Q123521",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P19",
    "report_violation_type": "Value type Q|2221906, Q|3895768, Q|27096213, Q|3238337, Q|6999, Q|11446, Q|16391167, Q|18670171, Q|811979, Q|115095765, Q|4130, Q|219858",
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
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000003

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Use only visible evidence and the output contract.

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
Field definitions:
- target.qid is the focus entity identifier from the input.
- target.pid is the target property identifier from the input.
- ops is the ordered set of claim edits to the target property on the focus entity.
- SET replaces the target property's value set with the supplied value.
- ADD adds the supplied value to the target property.
- REMOVE removes the supplied value from the target property.
- DELETE_ALL removes all values for the target property.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Replacement claim values must be ordinary claim values, not constraint-family identifiers, unless the prompt visibly
  presents that identifier as the claim value itself.
- Constraint-family QIDs, report-type QIDs, allowed-type QIDs, and ordinary entity/type QIDs have different roles; keep
  those roles distinct.
- Do not use hidden benchmark classes, subtypes, or historical labels.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000003",
  "labels_en": {
    "property": {
      "description": "most specific known birth location of a person, animal or fictional character",
      "label": "place of birth"
    },
    "qid": {
      "description": "Swiss poet (1925-2021)",
      "label": "Philippe Jaccottet"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "Swiss poet (1925-2021)",
      "label": "Philippe Jaccottet",
      "properties": {
        "P19": [
          "Q2708674"
        ]
      },
      "qid": "Q123521"
    },
    "L2_labels": {
      "entities": {
        "P19": {
          "description": "most specific known birth location of a person, animal or fictional character",
          "label": "place of birth"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q123521": {
          "description": "Swiss poet (1925-2021)",
          "label": "Philippe Jaccottet"
        },
        "Q2708674": {
          "description": "Dragon Ball character",
          "label": "Dr. Gero"
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
      "property_id": "P19"
    }
  },
  "property": "P19",
  "qid": "Q123521",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P19",
    "report_violation_type": "Value type Q|2221906, Q|3895768, Q|270
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000004

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Use only visible evidence and the output contract.

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
Field definitions:
- target.qid is the focus entity identifier from the input.
- target.pid is the target property identifier from the input.
- ops is the ordered set of claim edits to the target property on the focus entity.
- SET replaces the target property's value set with the supplied value.
- ADD adds the supplied value to the target property.
- REMOVE removes the supplied value from the target property.
- DELETE_ALL removes all values for the target property.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Replacement claim values must be ordinary claim values, not constraint-family identifiers, unless the prompt visibly
  presents that identifier as the claim value itself.
- Constraint-family QIDs, report-type QIDs, allowed-type QIDs, and ordinary entity/type QIDs have different roles; keep
  those roles distinct.
- Do not use hidden benchmark classes, subtypes, or historical labels.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000004",
  "labels_en": {
    "property": {
      "description": "sovereign state that this item is in (not to be used for human beings)",
      "label": "country"
    },
    "qid": {
      "description": null,
      "label": "Kirill Malofeyev"
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
              },
              {
                "label": "Wikibase property",
                "qid": "Q29934218",
                "raw": "Q29934218"
              },
              {
                "label": "Wikibase lexeme",
                "qid": "Q51885771",
                "raw": "Q51885771"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), Wikibase property (Q29934218), Wikibase lexeme (Q51885771)"
      }
    ],
    "property_id": "P17"
  },
  "property": "P17",
  "qid": "Q39082679",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P17",
    "report_violation_type": "Conflicts with P|31",
    "report_violation_type_normalized": "Conflicts with P|31",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Conflicts with P|31",
    "value": [
      "Q159"
    ],
    "value_descriptions_en": [
      "country in Eastern Europe and Northern Asia"
    ],
    "value_labels_en": [
      "Russia"
    ]
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000004

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Use only visible evidence and the output contract.

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
Field definitions:
- target.qid is the focus entity identifier from the input.
- target.pid is the target property identifier from the input.
- ops is the ordered set of claim edits to the target property on the focus entity.
- SET replaces the target property's value set with the supplied value.
- ADD adds the supplied value to the target property.
- REMOVE removes the supplied value from the target property.
- DELETE_ALL removes all values for the target property.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Replacement claim values must be ordinary claim values, not constraint-family identifiers, unless the prompt visibly
  presents that identifier as the claim value itself.
- Constraint-family QIDs, report-type QIDs, allowed-type QIDs, and ordinary entity/type QIDs have different roles; keep
  those roles distinct.
- Do not use hidden benchmark classes, subtypes, or historical labels.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000004",
  "labels_en": {
    "property": {
      "description": "sovereign state that this item is in (not to be used for human beings)",
      "label": "country"
    },
    "qid": {
      "description": null,
      "label": "Kirill Malofeyev"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "label": "Kirill Malofeyev",
      "properties": {
        "P17": [
          "Q159"
        ]
      },
      "qid": "Q39082679"
    },
    "L2_labels": {
      "entities": {
        "P17": {
          "description": "sovereign state that this item is in (not to be used for human beings)",
          "label": "country"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P27": {
          "description": "the object is a country that recognizes the subject as its citizen",
          "label": "country of citizenship"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q159": {
          "description": "country in Eastern Europe and Northern Asia",
          "label": "Russia"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q29934218": {
          "description": "entity type in Wikibase",
          "label": "Wikibase property"
        },
        "Q39082679": {
          "description": null,
          "label": "Kirill Malofeyev"
        },
        "Q51885771": {
          "description": "Wikibase entity type for lexemes",
          "label": "Wikibase lexeme"
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
      "outgoing_edges": [
        {
          "property_id": "P27",
          "target_description": "country in Eastern Europe and Northern Asia",
          "target_label": "Russia",
          "target_qid": "Q159"
        }
      ]
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
                  
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000005

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Use only visible evidence and the output contract.

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
Field definitions:
- target.qid is the focus entity identifier from the input.
- target.pid is the target property identifier from the input.
- ops is the ordered set of claim edits to the target property on the focus entity.
- SET replaces the target property's value set with the supplied value.
- ADD adds the supplied value to the target property.
- REMOVE removes the supplied value from the target property.
- DELETE_ALL removes all values for the target property.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Replacement claim values must be ordinary claim values, not constraint-family identifiers, unless the prompt visibly
  presents that identifier as the claim value itself.
- Constraint-family QIDs, report-type QIDs, allowed-type QIDs, and ordinary entity/type QIDs have different roles; keep
  those roles distinct.
- Do not use hidden benchmark classes, subtypes, or historical labels.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000005",
  "labels_en": {
    "property": {
      "description": "status of any use restrictions on the object, collection, or materials",
      "label": "use restriction status"
    },
    "qid": {
      "description": "series in the National Archives and Records Administration's holdings",
      "label": "Amy Kletnick's Files (NAID 7422239)"
    }
  },
  "logic_context": {
    "constraints": [
      {
        "constraint_type": {
          "label": "one-of constraint",
          "qid": "Q21510859"
        },
        "qualifiers": [
          {
            "property_id": "P2305",
            "property_label": "item of property constraint",
            "values": [
              {
                "label": "undetermined use restriction",
                "qid": "Q99868032",
                "raw": "Q99868032"
              },
              {
                "label": "possibly restricted use",
                "qid": "Q99867969",
                "raw": "Q99867969"
              },
              {
                "label": "unrestricted use",
                "qid": "Q99868068",
                "raw": "Q99868068"
              },
              {
                "label": "partly restricted use",
                "qid": "Q99867894",
                "raw": "Q99867894"
              },
              {
                "label": "use fully restricted",
                "qid": "Q99867853",
                "raw": "Q99867853"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): undetermined use restriction (Q99868032), possibly restricted use (Q99867969), unrestricted use (Q99868068), partly restricted use (Q99867894), use fully restricted (Q99867853)"
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
    "property_id": "P7261"
  },
  "property": "P7261",
  "qid": "Q63898616",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P7261",
    "report_violation_type": "One of",
    "report_violation_type_normalized": "One of",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "One of",
    "value": [
      "Q99868032"
    ],
    "value_descriptions_en": [
      "unknown if the archival materials have a use restriction"
    ],
    "value_labels_en": [
      "undetermined use restriction"
    ]
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000005

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Use only visible evidence and the output contract.

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
Field definitions:
- target.qid is the focus entity identifier from the input.
- target.pid is the target property identifier from the input.
- ops is the ordered set of claim edits to the target property on the focus entity.
- SET replaces the target property's value set with the supplied value.
- ADD adds the supplied value to the target property.
- REMOVE removes the supplied value from the target property.
- DELETE_ALL removes all values for the target property.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Replacement claim values must be ordinary claim values, not constraint-family identifiers, unless the prompt visibly
  presents that identifier as the claim value itself.
- Constraint-family QIDs, report-type QIDs, allowed-type QIDs, and ordinary entity/type QIDs have different roles; keep
  those roles distinct.
- Do not use hidden benchmark classes, subtypes, or historical labels.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000005",
  "labels_en": {
    "property": {
      "description": "status of any use restrictions on the object, collection, or materials",
      "label": "use restriction status"
    },
    "qid": {
      "description": "series in the National Archives and Records Administration's holdings",
      "label": "Amy Kletnick's Files (NAID 7422239)"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "series in the National Archives and Records Administration's holdings",
      "label": "Amy Kletnick's Files (NAID 7422239)",
      "properties": {
        "P7261": [
          "Q99868032"
        ]
      },
      "qid": "Q63898616"
    },
    "L2_labels": {
      "entities": {
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "P7261": {
          "description": "status of any use restrictions on the object, collection, or materials",
          "label": "use restriction status"
        },
        "Q21510859": {
          "description": "type of constraint for Wikidata properties: used to specify that the value for this property has to be one of a given set of items",
          "label": "one-of constraint"
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
        "Q63898616": {
          "description": "series in the National Archives and Records Administration's holdings",
          "label": "Amy Kletnick's Files (NAID 7422239)"
        },
        "Q99867853": {
          "description": "all the archival materials have either a copyright, donor, or other use restriction",
          "label": "use fully restricted"
        },
        "Q99867894": {
          "description": "some of the archival materials have a use restriction",
          "label": "partly restricted use"
        },
        "Q99867969": {
          "description": "archival materials may have a use restriction",
          "label": "possibly restricted use"
        },
        "Q99868032": {
          "description": "unknown if the archival materials have a use restriction",
          "label": "undetermined use restriction"
        },
        "Q99868068": {
          "description": "no copyright, donor, or other use restrictions on the archival materials",
          "label": "unrestricted use"
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
            "label": "one-of constraint",
            "qid": "Q21510859"
          },
          "qualifiers": [
            {
              "property_id": "P2305",
              "property_label": "item of property constraint",
              "values": [
                {
                  "label": "undetermined use restriction",
                  "qid": "Q99868032",
                  "raw": "Q99868032"
               
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000006

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Use only visible evidence and the output contract.

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
Field definitions:
- target.qid is the focus entity identifier from the input.
- target.pid is the target property identifier from the input.
- ops is the ordered set of claim edits to the target property on the focus entity.
- SET replaces the target property's value set with the supplied value.
- ADD adds the supplied value to the target property.
- REMOVE removes the supplied value from the target property.
- DELETE_ALL removes all values for the target property.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Replacement claim values must be ordinary claim values, not constraint-family identifiers, unless the prompt visibly
  presents that identifier as the claim value itself.
- Constraint-family QIDs, report-type QIDs, allowed-type QIDs, and ordinary entity/type QIDs have different roles; keep
  those roles distinct.
- Do not use hidden benchmark classes, subtypes, or historical labels.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000006",
  "labels_en": {
    "property": {
      "description": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
      "label": "location"
    },
    "qid": {
      "description": "Greek stage production of Shakespeare's play, 1995/96",
      "label": "King Lear"
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
      }
    ],
    "property_id": "P276"
  },
  "property": "P276",
  "qid": "Q99959551",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P276",
    "report_violation_type": "Value type Q|17334923, Q|618123, Q|82794, Q|3895768, Q|1656682, Q|47495022, Q|20203388, Q|4130, Q|6999, Q|190463, Q|2133296, Q|4936952, Q|988108, Q|11033, Q|2385804, Q|36133, Q|13226383, Q|2221906, Q|18670171, Q|3238337, Q|118547484, Q|204606, Q|9158768, Q|121141099, Q|41176, Q|7075, Q|166118, Q|43229, Q|853614, Q|13196193",
    "report_violation_type_normalized": "Value type Q|17334923, Q|618123, Q|82794, Q|3895768, Q|1656682, Q|47495022, Q|20203388, Q|4130, Q|6999, Q|190463, Q|2133296, Q|4936952, Q|988108, Q|11033, Q|2385804, Q|36133, Q|13226383, Q|2221906, Q|18670171, Q|3238337, Q|118547484, Q|204606, Q|9158768, Q|121141099, Q|41176, Q|7075, Q|166118, Q|43229, Q|853614, Q|13196193",
    "report_violation_type_qids": [
      "Q17334923",
      "Q618123",
      "Q82794",
      "Q3895768",
      "Q1656682",
      "Q47495022",
      "Q20203388",
      "Q4130",
      "Q6999",
      "Q190463",
      "Q2133296",
      "Q4936952",
      "Q988108",
      "Q11033",
      "Q2385804",
      "Q36133",
      "Q13226383",
      "Q2221906",
      "Q18670171",
      "Q3238337",
      "Q118547484",
      "Q204606",
      "Q9158768",
      "Q121141099",
      "Q41176",
      "Q7075",
      "Q166118",
      "Q43229",
      "Q853614",
      "Q13196193"
    ],
    "report_violation_type_raw": "Value type Q|17334923, Q|618123, Q|82794, Q|3895768, Q|1656682, Q|47495022, Q|20203388, Q|4130, Q|6999, Q|190463, Q|2133296, Q|4936952, Q|988108, Q|11033, Q|2385804, Q|36133, Q|13226383, Q|2221906, Q|18670171, Q|3238337, Q|118547484, Q|204606, Q|9158768, Q|121141099, Q|41176, Q|7075, Q|166118, Q|43229, Q|853614, Q|13196193",
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
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000006

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Use only visible evidence and the output contract.

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
Field definitions:
- target.qid is the focus entity identifier from the input.
- target.pid is the target property identifier from the input.
- ops is the ordered set of claim edits to the target property on the focus entity.
- SET replaces the target property's value set with the supplied value.
- ADD adds the supplied value to the target property.
- REMOVE removes the supplied value from the target property.
- DELETE_ALL removes all values for the target property.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Replacement claim values must be ordinary claim values, not constraint-family identifiers, unless the prompt visibly
  presents that identifier as the claim value itself.
- Constraint-family QIDs, report-type QIDs, allowed-type QIDs, and ordinary entity/type QIDs have different roles; keep
  those roles distinct.
- Do not use hidden benchmark classes, subtypes, or historical labels.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000006",
  "labels_en": {
    "property": {
      "description": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
      "label": "location"
    },
    "qid": {
      "description": "Greek stage production of Shakespeare's play, 1995/96",
      "label": "King Lear"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "Greek stage production of Shakespeare's play, 1995/96",
      "label": "King Lear",
      "properties": {
        "P276": [
          "Q12538685"
        ]
      },
      "qid": "Q99959551"
    },
    "L2_labels": {
      "entities": {
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P276": {
          "description": "location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object",
          "label": "location"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q12538685": {
          "description": "series of concerts by an artist or group of artists in different venues",
          "label": "concert tour"
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
        "Q59712033": {
          "description": "Wikibase entity type for Wikimedia Commons",
          "label": "МэдыяІнфа Вікібазы"
        },
        "Q99959551": {
          "description": "Greek stage production of Shakespeare's play, 1995/96",
          "label": "King Lear"
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
```
