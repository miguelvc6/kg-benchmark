# Prompt Development Review

No LLM inference was run for this artifact.

Rendered prompts: `512`

## Example Schema Summary

| Task | Example count | Example schemas |
| --- | ---: | --- |
| `a_box_repair` | 1536 | `a_box_v4_spec_only`: 1536 |

## prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain / case_000001

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
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

Example 1 input:
{
  "id": "example_a_000001",
  "labels_en": {
    "property": {
      "description": "primary topic of a work or act of communication",
      "label": "main subject"
    },
    "qid": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers"
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
              },
              {
                "label": "Wikibase property",
                "qid": "Q29934218",
                "raw": "Q29934218"
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
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033), Wikibase property (Q29934218); constraint status (P2316): mandatory constraint (Q21502408)"
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
              },
              {
                "label": "as reference",
                "qid": "Q54828450",
                "raw": "Q54828450"
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
        "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449), as reference (Q54828450); constraint status (P2316): mandatory constraint (Q21502408)"
      }
    ],
    "property_id": "P921"
  },
  "property": "P921",
  "qid": "Q95552174",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
    "report_violation_type": "Self link",
    "report_violation_type_normalized": "Self link",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Self link",
    "value": [
      "Q95552174"
    ],
    "value_descriptions_en": [
      "scientific article published on January 1, 1959"
    ],
    "value_labels_en": [
      "HOSPITAL volunteers"
    ]
  }
}
Example 1 expected JSON output:
{
  "case_id": "example_a_000001",
  "ops": [
    {
      "op": "DELETE_ALL",
      "pid": "P921"
    }
  ],
  "provenance": [
    {
      "kind": "OTHER",
      "snippet": "dev example visible repair target"
    }
  ],
  "rationale": "Demonstration answer reconstructed from the dev example's historical repaired value.",
  "target": {
    "pid": "P921",
    "qid": "Q95552174"
  },
  "uncertainty": {
    "confidence": 0.95,
    "notes": "Gold demonstration only; not a model output."
  }
}

Example 2 input:
{
  "id": "example_a_000002",
  "labels_en": {
    "property": {
      "description": 
```

## prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain / case_000001

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
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

Example 1 input:
{
  "id": "example_a_000001",
  "labels_en": {
    "property": {
      "description": "primary topic of a work or act of communication",
      "label": "main subject"
    },
    "qid": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers",
      "properties": {
        "P921": [
          "Q95552174"
        ]
      },
      "qid": "Q95552174"
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
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "P921": {
          "description": "primary topic of a work or act of communication",
          "label": "main subject"
        },
        "Q21502408": {
          "description": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
          "label": "mandatory constraint"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q29934218": {
          "description": "entity type in Wikibase",
          "label": "Wikibase property"
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
        "Q54828450": {
          "description": "property scope type",
          "label": "as reference"
        },
        "Q59712033": {
          "description": "Wikibase entity type for Wikimedia Commons",
          "label": "МэдыяІнфа Вікібазы"
        },
        "Q95552174": {
          "description": "scientific article published on January 1, 1959",
          "label": "HOSPITAL volunteers"
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
                },
                {
                  "label": "Wikibase property",
                  "qid": "Q29934218",
                  "raw": "Q29934218"
                }
              ]
            },
            {
              "property_id": "P2316",
              "prope
```

## prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain / case_000002

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
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

Example 1 input:
{
  "id": "example_a_000001",
  "labels_en": {
    "property": {
      "description": "primary topic of a work or act of communication",
      "label": "main subject"
    },
    "qid": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers"
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
              },
              {
                "label": "Wikibase property",
                "qid": "Q29934218",
                "raw": "Q29934218"
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
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033), Wikibase property (Q29934218); constraint status (P2316): mandatory constraint (Q21502408)"
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
              },
              {
                "label": "as reference",
                "qid": "Q54828450",
                "raw": "Q54828450"
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
        "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449), as reference (Q54828450); constraint status (P2316): mandatory constraint (Q21502408)"
      }
    ],
    "property_id": "P921"
  },
  "property": "P921",
  "qid": "Q95552174",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
    "report_violation_type": "Self link",
    "report_violation_type_normalized": "Self link",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Self link",
    "value": [
      "Q95552174"
    ],
    "value_descriptions_en": [
      "scientific article published on January 1, 1959"
    ],
    "value_labels_en": [
      "HOSPITAL volunteers"
    ]
  }
}
Example 1 expected JSON output:
{
  "case_id": "example_a_000001",
  "ops": [
    {
      "op": "DELETE_ALL",
      "pid": "P921"
    }
  ],
  "provenance": [
    {
      "kind": "OTHER",
      "snippet": "dev example visible repair target"
    }
  ],
  "rationale": "Demonstration answer reconstructed from the dev example's historical repaired value.",
  "target": {
    "pid": "P921",
    "qid": "Q95552174"
  },
  "uncertainty": {
    "confidence": 0.95,
    "notes": "Gold demonstration only; not a model output."
  }
}

Example 2 input:
{
  "id": "example_a_000002",
  "labels_en": {
    "property": {
      "description": 
```

## prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain / case_000002

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
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

Example 1 input:
{
  "id": "example_a_000001",
  "labels_en": {
    "property": {
      "description": "primary topic of a work or act of communication",
      "label": "main subject"
    },
    "qid": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers",
      "properties": {
        "P921": [
          "Q95552174"
        ]
      },
      "qid": "Q95552174"
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
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "P921": {
          "description": "primary topic of a work or act of communication",
          "label": "main subject"
        },
        "Q21502408": {
          "description": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
          "label": "mandatory constraint"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q29934218": {
          "description": "entity type in Wikibase",
          "label": "Wikibase property"
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
        "Q54828450": {
          "description": "property scope type",
          "label": "as reference"
        },
        "Q59712033": {
          "description": "Wikibase entity type for Wikimedia Commons",
          "label": "МэдыяІнфа Вікібазы"
        },
        "Q95552174": {
          "description": "scientific article published on January 1, 1959",
          "label": "HOSPITAL volunteers"
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
                },
                {
                  "label": "Wikibase property",
                  "qid": "Q29934218",
                  "raw": "Q29934218"
                }
              ]
            },
            {
              "property_id": "P2316",
              "prope
```

## prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain / case_000003

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
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

Example 1 input:
{
  "id": "example_a_000001",
  "labels_en": {
    "property": {
      "description": "primary topic of a work or act of communication",
      "label": "main subject"
    },
    "qid": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers"
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
              },
              {
                "label": "Wikibase property",
                "qid": "Q29934218",
                "raw": "Q29934218"
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
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033), Wikibase property (Q29934218); constraint status (P2316): mandatory constraint (Q21502408)"
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
              },
              {
                "label": "as reference",
                "qid": "Q54828450",
                "raw": "Q54828450"
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
        "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449), as reference (Q54828450); constraint status (P2316): mandatory constraint (Q21502408)"
      }
    ],
    "property_id": "P921"
  },
  "property": "P921",
  "qid": "Q95552174",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
    "report_violation_type": "Self link",
    "report_violation_type_normalized": "Self link",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Self link",
    "value": [
      "Q95552174"
    ],
    "value_descriptions_en": [
      "scientific article published on January 1, 1959"
    ],
    "value_labels_en": [
      "HOSPITAL volunteers"
    ]
  }
}
Example 1 expected JSON output:
{
  "case_id": "example_a_000001",
  "ops": [
    {
      "op": "DELETE_ALL",
      "pid": "P921"
    }
  ],
  "provenance": [
    {
      "kind": "OTHER",
      "snippet": "dev example visible repair target"
    }
  ],
  "rationale": "Demonstration answer reconstructed from the dev example's historical repaired value.",
  "target": {
    "pid": "P921",
    "qid": "Q95552174"
  },
  "uncertainty": {
    "confidence": 0.95,
    "notes": "Gold demonstration only; not a model output."
  }
}

Example 2 input:
{
  "id": "example_a_000002",
  "labels_en": {
    "property": {
      "description": 
```

## prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain / case_000003

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
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

Example 1 input:
{
  "id": "example_a_000001",
  "labels_en": {
    "property": {
      "description": "primary topic of a work or act of communication",
      "label": "main subject"
    },
    "qid": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers",
      "properties": {
        "P921": [
          "Q95552174"
        ]
      },
      "qid": "Q95552174"
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
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "P921": {
          "description": "primary topic of a work or act of communication",
          "label": "main subject"
        },
        "Q21502408": {
          "description": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
          "label": "mandatory constraint"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q29934218": {
          "description": "entity type in Wikibase",
          "label": "Wikibase property"
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
        "Q54828450": {
          "description": "property scope type",
          "label": "as reference"
        },
        "Q59712033": {
          "description": "Wikibase entity type for Wikimedia Commons",
          "label": "МэдыяІнфа Вікібазы"
        },
        "Q95552174": {
          "description": "scientific article published on January 1, 1959",
          "label": "HOSPITAL volunteers"
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
                },
                {
                  "label": "Wikibase property",
                  "qid": "Q29934218",
                  "raw": "Q29934218"
                }
              ]
            },
            {
              "property_id": "P2316",
              "prope
```

## prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain / case_000004

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
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

Example 1 input:
{
  "id": "example_a_000001",
  "labels_en": {
    "property": {
      "description": "primary topic of a work or act of communication",
      "label": "main subject"
    },
    "qid": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers"
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
              },
              {
                "label": "Wikibase property",
                "qid": "Q29934218",
                "raw": "Q29934218"
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
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033), Wikibase property (Q29934218); constraint status (P2316): mandatory constraint (Q21502408)"
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
              },
              {
                "label": "as reference",
                "qid": "Q54828450",
                "raw": "Q54828450"
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
        "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449), as reference (Q54828450); constraint status (P2316): mandatory constraint (Q21502408)"
      }
    ],
    "property_id": "P921"
  },
  "property": "P921",
  "qid": "Q95552174",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
    "report_violation_type": "Self link",
    "report_violation_type_normalized": "Self link",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Self link",
    "value": [
      "Q95552174"
    ],
    "value_descriptions_en": [
      "scientific article published on January 1, 1959"
    ],
    "value_labels_en": [
      "HOSPITAL volunteers"
    ]
  }
}
Example 1 expected JSON output:
{
  "case_id": "example_a_000001",
  "ops": [
    {
      "op": "DELETE_ALL",
      "pid": "P921"
    }
  ],
  "provenance": [
    {
      "kind": "OTHER",
      "snippet": "dev example visible repair target"
    }
  ],
  "rationale": "Demonstration answer reconstructed from the dev example's historical repaired value.",
  "target": {
    "pid": "P921",
    "qid": "Q95552174"
  },
  "uncertainty": {
    "confidence": 0.95,
    "notes": "Gold demonstration only; not a model output."
  }
}

Example 2 input:
{
  "id": "example_a_000002",
  "labels_en": {
    "property": {
      "description": 
```

## prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain / case_000004

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
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

Example 1 input:
{
  "id": "example_a_000001",
  "labels_en": {
    "property": {
      "description": "primary topic of a work or act of communication",
      "label": "main subject"
    },
    "qid": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers",
      "properties": {
        "P921": [
          "Q95552174"
        ]
      },
      "qid": "Q95552174"
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
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "P921": {
          "description": "primary topic of a work or act of communication",
          "label": "main subject"
        },
        "Q21502408": {
          "description": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
          "label": "mandatory constraint"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q29934218": {
          "description": "entity type in Wikibase",
          "label": "Wikibase property"
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
        "Q54828450": {
          "description": "property scope type",
          "label": "as reference"
        },
        "Q59712033": {
          "description": "Wikibase entity type for Wikimedia Commons",
          "label": "МэдыяІнфа Вікібазы"
        },
        "Q95552174": {
          "description": "scientific article published on January 1, 1959",
          "label": "HOSPITAL volunteers"
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
                },
                {
                  "label": "Wikibase property",
                  "qid": "Q29934218",
                  "raw": "Q29934218"
                }
              ]
            },
            {
              "property_id": "P2316",
              "prope
```

## prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain / case_000005

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
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

Example 1 input:
{
  "id": "example_a_000001",
  "labels_en": {
    "property": {
      "description": "primary topic of a work or act of communication",
      "label": "main subject"
    },
    "qid": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers"
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
              },
              {
                "label": "Wikibase property",
                "qid": "Q29934218",
                "raw": "Q29934218"
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
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033), Wikibase property (Q29934218); constraint status (P2316): mandatory constraint (Q21502408)"
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
              },
              {
                "label": "as reference",
                "qid": "Q54828450",
                "raw": "Q54828450"
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
        "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449), as reference (Q54828450); constraint status (P2316): mandatory constraint (Q21502408)"
      }
    ],
    "property_id": "P921"
  },
  "property": "P921",
  "qid": "Q95552174",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
    "report_violation_type": "Self link",
    "report_violation_type_normalized": "Self link",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Self link",
    "value": [
      "Q95552174"
    ],
    "value_descriptions_en": [
      "scientific article published on January 1, 1959"
    ],
    "value_labels_en": [
      "HOSPITAL volunteers"
    ]
  }
}
Example 1 expected JSON output:
{
  "case_id": "example_a_000001",
  "ops": [
    {
      "op": "DELETE_ALL",
      "pid": "P921"
    }
  ],
  "provenance": [
    {
      "kind": "OTHER",
      "snippet": "dev example visible repair target"
    }
  ],
  "rationale": "Demonstration answer reconstructed from the dev example's historical repaired value.",
  "target": {
    "pid": "P921",
    "qid": "Q95552174"
  },
  "uncertainty": {
    "confidence": 0.95,
    "notes": "Gold demonstration only; not a model output."
  }
}

Example 2 input:
{
  "id": "example_a_000002",
  "labels_en": {
    "property": {
      "description": 
```

## prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain / case_000005

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
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

Example 1 input:
{
  "id": "example_a_000001",
  "labels_en": {
    "property": {
      "description": "primary topic of a work or act of communication",
      "label": "main subject"
    },
    "qid": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers",
      "properties": {
        "P921": [
          "Q95552174"
        ]
      },
      "qid": "Q95552174"
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
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "P921": {
          "description": "primary topic of a work or act of communication",
          "label": "main subject"
        },
        "Q21502408": {
          "description": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
          "label": "mandatory constraint"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q29934218": {
          "description": "entity type in Wikibase",
          "label": "Wikibase property"
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
        "Q54828450": {
          "description": "property scope type",
          "label": "as reference"
        },
        "Q59712033": {
          "description": "Wikibase entity type for Wikimedia Commons",
          "label": "МэдыяІнфа Вікібазы"
        },
        "Q95552174": {
          "description": "scientific article published on January 1, 1959",
          "label": "HOSPITAL volunteers"
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
                },
                {
                  "label": "Wikibase property",
                  "qid": "Q29934218",
                  "raw": "Q29934218"
                }
              ]
            },
            {
              "property_id": "P2316",
              "prope
```

## prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain / case_000006

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
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

Example 1 input:
{
  "id": "example_a_000001",
  "labels_en": {
    "property": {
      "description": "primary topic of a work or act of communication",
      "label": "main subject"
    },
    "qid": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers"
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
              },
              {
                "label": "Wikibase property",
                "qid": "Q29934218",
                "raw": "Q29934218"
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
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033), Wikibase property (Q29934218); constraint status (P2316): mandatory constraint (Q21502408)"
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
              },
              {
                "label": "as reference",
                "qid": "Q54828450",
                "raw": "Q54828450"
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
        "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449), as reference (Q54828450); constraint status (P2316): mandatory constraint (Q21502408)"
      }
    ],
    "property_id": "P921"
  },
  "property": "P921",
  "qid": "Q95552174",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P921",
    "report_violation_type": "Self link",
    "report_violation_type_normalized": "Self link",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Self link",
    "value": [
      "Q95552174"
    ],
    "value_descriptions_en": [
      "scientific article published on January 1, 1959"
    ],
    "value_labels_en": [
      "HOSPITAL volunteers"
    ]
  }
}
Example 1 expected JSON output:
{
  "case_id": "example_a_000001",
  "ops": [
    {
      "op": "DELETE_ALL",
      "pid": "P921"
    }
  ],
  "provenance": [
    {
      "kind": "OTHER",
      "snippet": "dev example visible repair target"
    }
  ],
  "rationale": "Demonstration answer reconstructed from the dev example's historical repaired value.",
  "target": {
    "pid": "P921",
    "qid": "Q95552174"
  },
  "uncertainty": {
    "confidence": 0.95,
    "notes": "Gold demonstration only; not a model output."
  }
}

Example 2 input:
{
  "id": "example_a_000002",
  "labels_en": {
    "property": {
      "description": 
```

## prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain / case_000006

- Task: `a_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
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

Example 1 input:
{
  "id": "example_a_000001",
  "labels_en": {
    "property": {
      "description": "primary topic of a work or act of communication",
      "label": "main subject"
    },
    "qid": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "scientific article published on January 1, 1959",
      "label": "HOSPITAL volunteers",
      "properties": {
        "P921": [
          "Q95552174"
        ]
      },
      "qid": "Q95552174"
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
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "P921": {
          "description": "primary topic of a work or act of communication",
          "label": "main subject"
        },
        "Q21502408": {
          "description": "status of a Wikidata property constraint: indicates that the specified constraint applies to the subject property without exception and must not be violated",
          "label": "mandatory constraint"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q29934218": {
          "description": "entity type in Wikibase",
          "label": "Wikibase property"
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
        "Q54828450": {
          "description": "property scope type",
          "label": "as reference"
        },
        "Q59712033": {
          "description": "Wikibase entity type for Wikimedia Commons",
          "label": "МэдыяІнфа Вікібазы"
        },
        "Q95552174": {
          "description": "scientific article published on January 1, 1959",
          "label": "HOSPITAL volunteers"
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
                },
                {
                  "label": "Wikibase property",
                  "qid": "Q29934218",
                  "raw": "Q29934218"
                }
              ]
            },
            {
              "property_id": "P2316",
              "prope
```
