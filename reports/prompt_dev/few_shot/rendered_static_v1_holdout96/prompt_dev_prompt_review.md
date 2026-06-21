# Prompt Development Review

No LLM inference was run for this artifact.

Rendered prompts: `192`

## Example Schema Summary

| Task | Example count | Example schemas |
| --- | ---: | --- |
| `a_box_repair` | 286 | `a_box_v4_spec_only`: 286 |
| `t_box_repair` | 378 | `tbox_taxonomy_patch_v1`: 378 |

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

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v5_tbox_taxonomy_patch

Representation: hybrid_json_nl

Task: t_box_repair

Propose a T-box taxonomy patch for the focus property. Use only visible evidence, keep constraint-family identifiers distinct from ordinary item/type values, and report concrete value deltas only when they are visible.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "schema_decision": "CAUSAL_SCHEMA_REPAIR" | "NO_CAUSAL_SCHEMA_REPAIR" | "UNCLEAR_SCHEMA_EVIDENCE",
  "target": {
    "pid": "P...",
    "constraint_type_qid": "Q... or null only for UNCLEAR_SCHEMA_EVIDENCE when no visible constraint family is available"
  },
  "repairs": [
    {
      "repair_op": "CONSTRAINT_REMOVE"
        | "CONSTRAINT_DEPRECATE"
        | "CONSTRAINT_ADD"
        | "CONSTRAINT_TYPE_REPLACE"
        | "CONSTRAINT_QUALIFIER_ADD"
        | "CONSTRAINT_QUALIFIER_REMOVE"
        | "CONSTRAINT_QUALIFIER_REPLACE"
        | "CLASS_HIERARCHY_ADD"
        | "EXCEPTION_ADD"
        | "OTHER_TBOX_UPDATE",
      "taxonomy_code": "C_MINUS" | "C_D" | "C_PLUS" | "C_REPLACE" | "CQ_PLUS" | "CQ_MINUS"
        | "CQ_REPLACE" | "SUBCLASS_PLUS" | "E_PLUS" | "OTHER",
      "constraint_type_qid": "Q...",
      "qualifier_property_id": "P... or null",
      "added_values": ["Q..." | "P..." | "<literal>" | 123],
      "removed_values": ["Q..." | "P..." | "<literal>" | 123],
      "old_value": "Q... | P... | <literal> | 123 | null",
      "new_value": "Q... | P... | <literal> | 123 | null",
      "rank_after": "normal" | "preferred" | "deprecated" | null,
      "snaktype_after": "VALUE" | "SOMEVALUE" | "NOVALUE" | null,
      "evidence_level": "FAMILY_ONLY" | "OPERATION_VISIBLE" | "VALUE_DELTA_VISIBLE"
    }
  ],
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "OTHER", "node_id": "Q... or P... or null", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}
Field definitions:
- schema_decision states whether visible evidence supports a causal schema repair, no causal schema repair, or unclear
  schema evidence.
- target.pid is the focus property identifier from the input.
- target.constraint_type_qid is the visible constraint-family identifier being considered; use null only when
  schema_decision is UNCLEAR_SCHEMA_EVIDENCE and no visible constraint family is available.
- repairs is empty only for NO_CAUSAL_SCHEMA_REPAIR or UNCLEAR_SCHEMA_EVIDENCE.
- Use NO_CAUSAL_SCHEMA_REPAIR only when a visible constraint family can be named but visible evidence does not support a
  causal schema edit for it. If no constraint family can be named from visible evidence, use UNCLEAR_SCHEMA_EVIDENCE.
- repair_op is the schema-level operation.
- taxonomy_code is the code paired with repair_op.
- constraint_type_qid inside each repair is the edited constraint family.
- qualifier_property_id is the edited qualifier property, or null when not applicable.
- added_values and removed_values are concrete changed values only when visible; otherwise use empty lists.
- old_value and new_value summarize a replacement when visible; otherwise use null.
- rank_after and snaktype_after describe visible rank or snaktype after the update, or null.
- evidence_level is FAMILY_ONLY, OPERATION_VISIBLE, or VALUE_DELTA_VISIBLE depending on visible evidence.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Operation definitions:
- CONSTRAINT_REMOVE / C_MINUS removes a property-constraint statement or constraint family.
- CONSTRAINT_DEPRECATE / C_D deprecates or deactivates a constraint statement by rank or status.
- CONSTRAINT_ADD / C_PLUS adds a property-constraint statement or constraint family.
- CONSTRAINT_TYPE_REPLACE / C_REPLACE replaces one constraint family with another.
- CONSTRAINT_QUALIFIER_ADD / CQ_PLUS adds a qualifier value to a constraint definition.
- CONSTRAINT_QUALIFIER_REMOVE / CQ_MINUS removes a qualifier value from a constraint definition.
- CONSTRAINT_QUALIFIER_REPLACE / CQ_REPLACE replaces a qualifier value on the same qualifier property.
- CLASS_HIERARCHY_ADD / SUBCLASS_PLUS adds a subclass relation that resolves the violation through class hierarchy.
- EXCEPTION_ADD / E_PLUS adds an exception value to the constraint.
- OTHER_TBOX_UPDATE / OTHER is a schema-level update not covered by the listed operations.
Evidence boundary:
- Use only visible prompt evidence.
- Keep constraint-family QIDs separate from ordinary item/type values.
- Do not use an empty string or placeholder for target.constraint_type_qid.
- Do not copy report-violation QIDs into value deltas unless they are visibly presented as schema values.
- Use empty added_values and removed_values lists when concrete changed values are not visible.
- Do not construct a full post-repair signature_after.
- Do not use hidden benchmark metadata, hidden classes, hidden subtypes, historical labels, or raw benchmark prefixes.

Few-shot examples:

Example 1 input:
{
  "id": "example_t_000001",
  "labels_en": {
    "property": {
      "description": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
      "label": "OS grid reference"
    },
    "qid": {
      "description": "garden house in Boynton, East Riding of Yorkshire, England, UK",
      "label": "Garden House At Boynton Hall"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q21502404",
        "label": "format constraint"
      },
      {
        "constraint_qid": "Q52060874",
        "label": "single-best-value constraint"
      },
      {
        "constraint_qid": "Q21503247",
        "label": "item-requires-statement constraint"
      },
      {
        "constraint_qid": "Q21503250",
        "label": "subject type constraint"
      }
    ],
    "violation_context": {
    
```

## prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain / case_000002

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v5_tbox_taxonomy_patch

Representation: hybrid_json_nl

Task: t_box_repair

Propose a T-box taxonomy patch for the focus property. Use only visible evidence, keep constraint-family identifiers distinct from ordinary item/type values, and report concrete value deltas only when they are visible.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "schema_decision": "CAUSAL_SCHEMA_REPAIR" | "NO_CAUSAL_SCHEMA_REPAIR" | "UNCLEAR_SCHEMA_EVIDENCE",
  "target": {
    "pid": "P...",
    "constraint_type_qid": "Q... or null only for UNCLEAR_SCHEMA_EVIDENCE when no visible constraint family is available"
  },
  "repairs": [
    {
      "repair_op": "CONSTRAINT_REMOVE"
        | "CONSTRAINT_DEPRECATE"
        | "CONSTRAINT_ADD"
        | "CONSTRAINT_TYPE_REPLACE"
        | "CONSTRAINT_QUALIFIER_ADD"
        | "CONSTRAINT_QUALIFIER_REMOVE"
        | "CONSTRAINT_QUALIFIER_REPLACE"
        | "CLASS_HIERARCHY_ADD"
        | "EXCEPTION_ADD"
        | "OTHER_TBOX_UPDATE",
      "taxonomy_code": "C_MINUS" | "C_D" | "C_PLUS" | "C_REPLACE" | "CQ_PLUS" | "CQ_MINUS"
        | "CQ_REPLACE" | "SUBCLASS_PLUS" | "E_PLUS" | "OTHER",
      "constraint_type_qid": "Q...",
      "qualifier_property_id": "P... or null",
      "added_values": ["Q..." | "P..." | "<literal>" | 123],
      "removed_values": ["Q..." | "P..." | "<literal>" | 123],
      "old_value": "Q... | P... | <literal> | 123 | null",
      "new_value": "Q... | P... | <literal> | 123 | null",
      "rank_after": "normal" | "preferred" | "deprecated" | null,
      "snaktype_after": "VALUE" | "SOMEVALUE" | "NOVALUE" | null,
      "evidence_level": "FAMILY_ONLY" | "OPERATION_VISIBLE" | "VALUE_DELTA_VISIBLE"
    }
  ],
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "OTHER", "node_id": "Q... or P... or null", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}
Field definitions:
- schema_decision states whether visible evidence supports a causal schema repair, no causal schema repair, or unclear
  schema evidence.
- target.pid is the focus property identifier from the input.
- target.constraint_type_qid is the visible constraint-family identifier being considered; use null only when
  schema_decision is UNCLEAR_SCHEMA_EVIDENCE and no visible constraint family is available.
- repairs is empty only for NO_CAUSAL_SCHEMA_REPAIR or UNCLEAR_SCHEMA_EVIDENCE.
- Use NO_CAUSAL_SCHEMA_REPAIR only when a visible constraint family can be named but visible evidence does not support a
  causal schema edit for it. If no constraint family can be named from visible evidence, use UNCLEAR_SCHEMA_EVIDENCE.
- repair_op is the schema-level operation.
- taxonomy_code is the code paired with repair_op.
- constraint_type_qid inside each repair is the edited constraint family.
- qualifier_property_id is the edited qualifier property, or null when not applicable.
- added_values and removed_values are concrete changed values only when visible; otherwise use empty lists.
- old_value and new_value summarize a replacement when visible; otherwise use null.
- rank_after and snaktype_after describe visible rank or snaktype after the update, or null.
- evidence_level is FAMILY_ONLY, OPERATION_VISIBLE, or VALUE_DELTA_VISIBLE depending on visible evidence.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Operation definitions:
- CONSTRAINT_REMOVE / C_MINUS removes a property-constraint statement or constraint family.
- CONSTRAINT_DEPRECATE / C_D deprecates or deactivates a constraint statement by rank or status.
- CONSTRAINT_ADD / C_PLUS adds a property-constraint statement or constraint family.
- CONSTRAINT_TYPE_REPLACE / C_REPLACE replaces one constraint family with another.
- CONSTRAINT_QUALIFIER_ADD / CQ_PLUS adds a qualifier value to a constraint definition.
- CONSTRAINT_QUALIFIER_REMOVE / CQ_MINUS removes a qualifier value from a constraint definition.
- CONSTRAINT_QUALIFIER_REPLACE / CQ_REPLACE replaces a qualifier value on the same qualifier property.
- CLASS_HIERARCHY_ADD / SUBCLASS_PLUS adds a subclass relation that resolves the violation through class hierarchy.
- EXCEPTION_ADD / E_PLUS adds an exception value to the constraint.
- OTHER_TBOX_UPDATE / OTHER is a schema-level update not covered by the listed operations.
Evidence boundary:
- Use only visible prompt evidence.
- Keep constraint-family QIDs separate from ordinary item/type values.
- Do not use an empty string or placeholder for target.constraint_type_qid.
- Do not copy report-violation QIDs into value deltas unless they are visibly presented as schema values.
- Use empty added_values and removed_values lists when concrete changed values are not visible.
- Do not construct a full post-repair signature_after.
- Do not use hidden benchmark metadata, hidden classes, hidden subtypes, historical labels, or raw benchmark prefixes.

Few-shot examples:

Example 1 input:
{
  "id": "example_t_000001",
  "labels_en": {
    "property": {
      "description": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
      "label": "OS grid reference"
    },
    "qid": {
      "description": "garden house in Boynton, East Riding of Yorkshire, England, UK",
      "label": "Garden House At Boynton Hall"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "garden house in Boynton, East Riding of Yorkshire, England, UK",
      "label": "Garden House At Boynton Hall",
      "qid": "Q17555344"
    },
    "L2_labels": {
      "entities": {
        "P613": {
          "description": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
          "label": "OS grid reference"
        },
        "Q17555344": {
          "description": "garden 
```

## prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain / case_000003

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
- Context: `logic_only`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v5_tbox_taxonomy_patch

Representation: hybrid_json_nl

Task: t_box_repair

Propose a T-box taxonomy patch for the focus property. Use only visible evidence, keep constraint-family identifiers distinct from ordinary item/type values, and report concrete value deltas only when they are visible.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "schema_decision": "CAUSAL_SCHEMA_REPAIR" | "NO_CAUSAL_SCHEMA_REPAIR" | "UNCLEAR_SCHEMA_EVIDENCE",
  "target": {
    "pid": "P...",
    "constraint_type_qid": "Q... or null only for UNCLEAR_SCHEMA_EVIDENCE when no visible constraint family is available"
  },
  "repairs": [
    {
      "repair_op": "CONSTRAINT_REMOVE"
        | "CONSTRAINT_DEPRECATE"
        | "CONSTRAINT_ADD"
        | "CONSTRAINT_TYPE_REPLACE"
        | "CONSTRAINT_QUALIFIER_ADD"
        | "CONSTRAINT_QUALIFIER_REMOVE"
        | "CONSTRAINT_QUALIFIER_REPLACE"
        | "CLASS_HIERARCHY_ADD"
        | "EXCEPTION_ADD"
        | "OTHER_TBOX_UPDATE",
      "taxonomy_code": "C_MINUS" | "C_D" | "C_PLUS" | "C_REPLACE" | "CQ_PLUS" | "CQ_MINUS"
        | "CQ_REPLACE" | "SUBCLASS_PLUS" | "E_PLUS" | "OTHER",
      "constraint_type_qid": "Q...",
      "qualifier_property_id": "P... or null",
      "added_values": ["Q..." | "P..." | "<literal>" | 123],
      "removed_values": ["Q..." | "P..." | "<literal>" | 123],
      "old_value": "Q... | P... | <literal> | 123 | null",
      "new_value": "Q... | P... | <literal> | 123 | null",
      "rank_after": "normal" | "preferred" | "deprecated" | null,
      "snaktype_after": "VALUE" | "SOMEVALUE" | "NOVALUE" | null,
      "evidence_level": "FAMILY_ONLY" | "OPERATION_VISIBLE" | "VALUE_DELTA_VISIBLE"
    }
  ],
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "OTHER", "node_id": "Q... or P... or null", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}
Field definitions:
- schema_decision states whether visible evidence supports a causal schema repair, no causal schema repair, or unclear
  schema evidence.
- target.pid is the focus property identifier from the input.
- target.constraint_type_qid is the visible constraint-family identifier being considered; use null only when
  schema_decision is UNCLEAR_SCHEMA_EVIDENCE and no visible constraint family is available.
- repairs is empty only for NO_CAUSAL_SCHEMA_REPAIR or UNCLEAR_SCHEMA_EVIDENCE.
- Use NO_CAUSAL_SCHEMA_REPAIR only when a visible constraint family can be named but visible evidence does not support a
  causal schema edit for it. If no constraint family can be named from visible evidence, use UNCLEAR_SCHEMA_EVIDENCE.
- repair_op is the schema-level operation.
- taxonomy_code is the code paired with repair_op.
- constraint_type_qid inside each repair is the edited constraint family.
- qualifier_property_id is the edited qualifier property, or null when not applicable.
- added_values and removed_values are concrete changed values only when visible; otherwise use empty lists.
- old_value and new_value summarize a replacement when visible; otherwise use null.
- rank_after and snaktype_after describe visible rank or snaktype after the update, or null.
- evidence_level is FAMILY_ONLY, OPERATION_VISIBLE, or VALUE_DELTA_VISIBLE depending on visible evidence.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Operation definitions:
- CONSTRAINT_REMOVE / C_MINUS removes a property-constraint statement or constraint family.
- CONSTRAINT_DEPRECATE / C_D deprecates or deactivates a constraint statement by rank or status.
- CONSTRAINT_ADD / C_PLUS adds a property-constraint statement or constraint family.
- CONSTRAINT_TYPE_REPLACE / C_REPLACE replaces one constraint family with another.
- CONSTRAINT_QUALIFIER_ADD / CQ_PLUS adds a qualifier value to a constraint definition.
- CONSTRAINT_QUALIFIER_REMOVE / CQ_MINUS removes a qualifier value from a constraint definition.
- CONSTRAINT_QUALIFIER_REPLACE / CQ_REPLACE replaces a qualifier value on the same qualifier property.
- CLASS_HIERARCHY_ADD / SUBCLASS_PLUS adds a subclass relation that resolves the violation through class hierarchy.
- EXCEPTION_ADD / E_PLUS adds an exception value to the constraint.
- OTHER_TBOX_UPDATE / OTHER is a schema-level update not covered by the listed operations.
Evidence boundary:
- Use only visible prompt evidence.
- Keep constraint-family QIDs separate from ordinary item/type values.
- Do not use an empty string or placeholder for target.constraint_type_qid.
- Do not copy report-violation QIDs into value deltas unless they are visibly presented as schema values.
- Use empty added_values and removed_values lists when concrete changed values are not visible.
- Do not construct a full post-repair signature_after.
- Do not use hidden benchmark metadata, hidden classes, hidden subtypes, historical labels, or raw benchmark prefixes.

Few-shot examples:

Example 1 input:
{
  "id": "example_t_000001",
  "labels_en": {
    "property": {
      "description": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
      "label": "OS grid reference"
    },
    "qid": {
      "description": "garden house in Boynton, East Riding of Yorkshire, England, UK",
      "label": "Garden House At Boynton Hall"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q21502404",
        "label": "format constraint"
      },
      {
        "constraint_qid": "Q52060874",
        "label": "single-best-value constraint"
      },
      {
        "constraint_qid": "Q21503247",
        "label": "item-requires-statement constraint"
      },
      {
        "constraint_qid": "Q21503250",
        "label": "subject type constraint"
      }
    ],
    "violation_context": {
    
```

## prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain / case_000003

- Task: `t_box_repair`
- Representation: `hybrid_json_nl`
- Examples: `static_diverse_kshot`
- Context: `local_graph`

System prompt:
```text
You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences. Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON.
```

User prompt:
```text
Prompt version: prompt_dev_v5_tbox_taxonomy_patch

Representation: hybrid_json_nl

Task: t_box_repair

Propose a T-box taxonomy patch for the focus property. Use only visible evidence, keep constraint-family identifiers distinct from ordinary item/type values, and report concrete value deltas only when they are visible.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "schema_decision": "CAUSAL_SCHEMA_REPAIR" | "NO_CAUSAL_SCHEMA_REPAIR" | "UNCLEAR_SCHEMA_EVIDENCE",
  "target": {
    "pid": "P...",
    "constraint_type_qid": "Q... or null only for UNCLEAR_SCHEMA_EVIDENCE when no visible constraint family is available"
  },
  "repairs": [
    {
      "repair_op": "CONSTRAINT_REMOVE"
        | "CONSTRAINT_DEPRECATE"
        | "CONSTRAINT_ADD"
        | "CONSTRAINT_TYPE_REPLACE"
        | "CONSTRAINT_QUALIFIER_ADD"
        | "CONSTRAINT_QUALIFIER_REMOVE"
        | "CONSTRAINT_QUALIFIER_REPLACE"
        | "CLASS_HIERARCHY_ADD"
        | "EXCEPTION_ADD"
        | "OTHER_TBOX_UPDATE",
      "taxonomy_code": "C_MINUS" | "C_D" | "C_PLUS" | "C_REPLACE" | "CQ_PLUS" | "CQ_MINUS"
        | "CQ_REPLACE" | "SUBCLASS_PLUS" | "E_PLUS" | "OTHER",
      "constraint_type_qid": "Q...",
      "qualifier_property_id": "P... or null",
      "added_values": ["Q..." | "P..." | "<literal>" | 123],
      "removed_values": ["Q..." | "P..." | "<literal>" | 123],
      "old_value": "Q... | P... | <literal> | 123 | null",
      "new_value": "Q... | P... | <literal> | 123 | null",
      "rank_after": "normal" | "preferred" | "deprecated" | null,
      "snaktype_after": "VALUE" | "SOMEVALUE" | "NOVALUE" | null,
      "evidence_level": "FAMILY_ONLY" | "OPERATION_VISIBLE" | "VALUE_DELTA_VISIBLE"
    }
  ],
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "OTHER", "node_id": "Q... or P... or null", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}
Field definitions:
- schema_decision states whether visible evidence supports a causal schema repair, no causal schema repair, or unclear
  schema evidence.
- target.pid is the focus property identifier from the input.
- target.constraint_type_qid is the visible constraint-family identifier being considered; use null only when
  schema_decision is UNCLEAR_SCHEMA_EVIDENCE and no visible constraint family is available.
- repairs is empty only for NO_CAUSAL_SCHEMA_REPAIR or UNCLEAR_SCHEMA_EVIDENCE.
- Use NO_CAUSAL_SCHEMA_REPAIR only when a visible constraint family can be named but visible evidence does not support a
  causal schema edit for it. If no constraint family can be named from visible evidence, use UNCLEAR_SCHEMA_EVIDENCE.
- repair_op is the schema-level operation.
- taxonomy_code is the code paired with repair_op.
- constraint_type_qid inside each repair is the edited constraint family.
- qualifier_property_id is the edited qualifier property, or null when not applicable.
- added_values and removed_values are concrete changed values only when visible; otherwise use empty lists.
- old_value and new_value summarize a replacement when visible; otherwise use null.
- rank_after and snaktype_after describe visible rank or snaktype after the update, or null.
- evidence_level is FAMILY_ONLY, OPERATION_VISIBLE, or VALUE_DELTA_VISIBLE depending on visible evidence.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Operation definitions:
- CONSTRAINT_REMOVE / C_MINUS removes a property-constraint statement or constraint family.
- CONSTRAINT_DEPRECATE / C_D deprecates or deactivates a constraint statement by rank or status.
- CONSTRAINT_ADD / C_PLUS adds a property-constraint statement or constraint family.
- CONSTRAINT_TYPE_REPLACE / C_REPLACE replaces one constraint family with another.
- CONSTRAINT_QUALIFIER_ADD / CQ_PLUS adds a qualifier value to a constraint definition.
- CONSTRAINT_QUALIFIER_REMOVE / CQ_MINUS removes a qualifier value from a constraint definition.
- CONSTRAINT_QUALIFIER_REPLACE / CQ_REPLACE replaces a qualifier value on the same qualifier property.
- CLASS_HIERARCHY_ADD / SUBCLASS_PLUS adds a subclass relation that resolves the violation through class hierarchy.
- EXCEPTION_ADD / E_PLUS adds an exception value to the constraint.
- OTHER_TBOX_UPDATE / OTHER is a schema-level update not covered by the listed operations.
Evidence boundary:
- Use only visible prompt evidence.
- Keep constraint-family QIDs separate from ordinary item/type values.
- Do not use an empty string or placeholder for target.constraint_type_qid.
- Do not copy report-violation QIDs into value deltas unless they are visibly presented as schema values.
- Use empty added_values and removed_values lists when concrete changed values are not visible.
- Do not construct a full post-repair signature_after.
- Do not use hidden benchmark metadata, hidden classes, hidden subtypes, historical labels, or raw benchmark prefixes.

Few-shot examples:

Example 1 input:
{
  "id": "example_t_000001",
  "labels_en": {
    "property": {
      "description": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
      "label": "OS grid reference"
    },
    "qid": {
      "description": "garden house in Boynton, East Riding of Yorkshire, England, UK",
      "label": "Garden House At Boynton Hall"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "garden house in Boynton, East Riding of Yorkshire, England, UK",
      "label": "Garden House At Boynton Hall",
      "qid": "Q17555344"
    },
    "L2_labels": {
      "entities": {
        "P613": {
          "description": "grid location reference from the Ordnance Survey National Grid reference system used in Great Britain",
          "label": "OS grid reference"
        },
        "Q17555344": {
          "description": "garden 
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
