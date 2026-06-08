# Prompt Development Review

No LLM inference was run for this artifact.

Rendered prompts: `384`

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / case_000374

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
Prompt version: prompt_dev_v3

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely. A constraint report alone does not imply T_BOX, but property-level schema-change evidence does. Decide based on likely repair locus.

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
- Choose T_BOX when the visible evidence points to a property-level rule change, such as changed constraint families,
  schema-change context, or a report that is better resolved by editing the constraint than by editing one entity.
- Choose AMBIGUOUS when both claim repair and schema reform remain plausible from the visible evidence.
- Do not use AMBIGUOUS merely because the case is hard; use it only when the visible evidence supports neither repair
  locus clearly.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

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

## prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000374

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
Prompt version: prompt_dev_v3

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Choose values only from visible target-value evidence. Preserve useful values when the evidence supports them; use targeted REMOVE instead of DELETE_ALL when only one visible value is bad.

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
- Do not invent a new entity value. If no replacement value is visible, remove only the specific visible bad value; do
  not delete retained values.

Operation rubric:
- Use SET when the final target property should contain exactly one visible value.
- Use ADD only to add a visible missing value while preserving existing retained values.
- Use REMOVE to remove a specific visible bad value while preserving all other retained values.
- Use DELETE_ALL only when the prompt evidence shows every current target value should be removed and no retained value
  remains.
- If evidence is insufficient for a replacement value, a targeted REMOVE is safer than SET to a constraint/type QID or
  DELETE_ALL.
- Preserve retained values. Do not over-delete merely to satisfy a constraint.
- For TypeC or unknown/insufficient-evidence cases, avoid hallucinated replacements; make the smallest visible repair
  and report low confidence.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

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

## prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000374

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
Prompt version: prompt_dev_v3

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely. A constraint report alone does not imply T_BOX, but property-level schema-change evidence does. Decide based on likely repair locus.

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
- Choose T_BOX when the visible evidence points to a property-level rule change, such as changed constraint families,
  schema-change context, or a report that is better resolved by editing the constraint than by editing one entity.
- Choose AMBIGUOUS when both claim repair and schema reform remain plausible from the visible evidence.
- Do not use AMBIGUOUS merely because the case is hard; use it only when the visible evidence supports neither repair
  locus clearly.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

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
    "report_violation_type"
```

## prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000374

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
Prompt version: prompt_dev_v3

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Choose values only from visible target-value evidence. Preserve useful values when the evidence supports them; use targeted REMOVE instead of DELETE_ALL when only one visible value is bad.

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
- Do not invent a new entity value. If no replacement value is visible, remove only the specific visible bad value; do
  not delete retained values.

Operation rubric:
- Use SET when the final target property should contain exactly one visible value.
- Use ADD only to add a visible missing value while preserving existing retained values.
- Use REMOVE to remove a specific visible bad value while preserving all other retained values.
- Use DELETE_ALL only when the prompt evidence shows every current target value should be removed and no retained value
  remains.
- If evidence is insufficient for a replacement value, a targeted REMOVE is safer than SET to a constraint/type QID or
  DELETE_ALL.
- Preserve retained values. Do not over-delete merely to satisfy a constraint.
- For TypeC or unknown/insufficient-evidence cases, avoid hallucinated replacements; make the smallest visible repair
  and report low confidence.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

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
            "l
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / case_000093

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
Prompt version: prompt_dev_v3

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely. A constraint report alone does not imply T_BOX, but property-level schema-change evidence does. Decide based on likely repair locus.

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
- Choose T_BOX when the visible evidence points to a property-level rule change, such as changed constraint families,
  schema-change context, or a report that is better resolved by editing the constraint than by editing one entity.
- Choose AMBIGUOUS when both claim repair and schema reform remain plausible from the visible evidence.
- Do not use AMBIGUOUS merely because the case is hard; use it only when the visible evidence supports neither repair
  locus clearly.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

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

## prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000093

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
Prompt version: prompt_dev_v3

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

## prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000093

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
Prompt version: prompt_dev_v3

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely. A constraint report alone does not imply T_BOX, but property-level schema-change evidence does. Decide based on likely repair locus.

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
- Choose T_BOX when the visible evidence points to a property-level rule change, such as changed constraint families,
  schema-change context, or a report that is better resolved by editing the constraint than by editing one entity.
- Choose AMBIGUOUS when both claim repair and schema reform remain plausible from the visible evidence.
- Do not use AMBIGUOUS merely because the case is hard; use it only when the visible evidence supports neither repair
  locus clearly.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

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

## prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000093

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
Prompt version: prompt_dev_v3

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
          "label": "value-type c
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / case_000479

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
Prompt version: prompt_dev_v3

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely. A constraint report alone does not imply T_BOX, but property-level schema-change evidence does. Decide based on likely repair locus.

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
- Choose T_BOX when the visible evidence points to a property-level rule change, such as changed constraint families,
  schema-change context, or a report that is better resolved by editing the constraint than by editing one entity.
- Choose AMBIGUOUS when both claim repair and schema reform remain plausible from the visible evidence.
- Do not use AMBIGUOUS merely because the case is hard; use it only when the visible evidence supports neither repair
  locus clearly.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000479",
  "labels_en": {
    "property": {
      "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
      "label": "SureChEMBL ID"
    },
    "qid": {
      "description": "chemical compound",
      "label": "suprofen"
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
  "qid": "Q3978097",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
    "report_violation_type": "Format",
    "report_violation_type_normalized": "Format",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Format",
    "value": [
      "SCHEMBL23792"
    ]
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000479

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
Prompt version: prompt_dev_v3

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Choose values only from visible target-value evidence. Preserve useful values when the evidence supports them; use targeted REMOVE instead of DELETE_ALL when only one visible value is bad.

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
- Do not invent a new entity value. If no replacement value is visible, remove only the specific visible bad value; do
  not delete retained values.

Operation rubric:
- Use SET when the final target property should contain exactly one visible value.
- Use ADD only to add a visible missing value while preserving existing retained values.
- Use REMOVE to remove a specific visible bad value while preserving all other retained values.
- Use DELETE_ALL only when the prompt evidence shows every current target value should be removed and no retained value
  remains.
- If evidence is insufficient for a replacement value, a targeted REMOVE is safer than SET to a constraint/type QID or
  DELETE_ALL.
- Preserve retained values. Do not over-delete merely to satisfy a constraint.
- For TypeC or unknown/insufficient-evidence cases, avoid hallucinated replacements; make the smallest visible repair
  and report low confidence.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000479",
  "labels_en": {
    "property": {
      "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
      "label": "SureChEMBL ID"
    },
    "qid": {
      "description": "chemical compound",
      "label": "suprofen"
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
  "qid": "Q3978097",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
    "report_violation_type": "Format",
    "report_violation_type_normalized": "Format",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Format",
    "value": [
      "SCHEMBL23792"
    ]
  }
}
```

## prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000479

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
Prompt version: prompt_dev_v3

Representation: hybrid_json_nl

Task: track_diagnosis

Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support choosing safely. A constraint report alone does not imply T_BOX, but property-level schema-change evidence does. Decide based on likely repair locus.

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
- Choose T_BOX when the visible evidence points to a property-level rule change, such as changed constraint families,
  schema-change context, or a report that is better resolved by editing the constraint than by editing one entity.
- Choose AMBIGUOUS when both claim repair and schema reform remain plausible from the visible evidence.
- Do not use AMBIGUOUS merely because the case is hard; use it only when the visible evidence supports neither repair
  locus clearly.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000479",
  "labels_en": {
    "property": {
      "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
      "label": "SureChEMBL ID"
    },
    "qid": {
      "description": "chemical compound",
      "label": "suprofen"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "chemical compound",
      "label": "suprofen",
      "properties": {
        "P2877": [
          "SCHEMBL23792"
        ]
      },
      "qid": "Q3978097"
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
        "Q3978097": {
          "description": "chemical compound",
          "label": "suprofen"
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
              "values
```

## prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000479

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
Prompt version: prompt_dev_v3

Representation: hybrid_json_nl

Task: a_box_repair

Propose an executable A-box repair transaction for the focus entity and target property. Choose values only from visible target-value evidence. Preserve useful values when the evidence supports them; use targeted REMOVE instead of DELETE_ALL when only one visible value is bad.

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
- Do not invent a new entity value. If no replacement value is visible, remove only the specific visible bad value; do
  not delete retained values.

Operation rubric:
- Use SET when the final target property should contain exactly one visible value.
- Use ADD only to add a visible missing value while preserving existing retained values.
- Use REMOVE to remove a specific visible bad value while preserving all other retained values.
- Use DELETE_ALL only when the prompt evidence shows every current target value should be removed and no retained value
  remains.
- If evidence is insufficient for a replacement value, a targeted REMOVE is safer than SET to a constraint/type QID or
  DELETE_ALL.
- Preserve retained values. Do not over-delete merely to satisfy a constraint.
- For TypeC or unknown/insufficient-evidence cases, avoid hallucinated replacements; make the smallest visible repair
  and report low confidence.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000479",
  "labels_en": {
    "property": {
      "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
      "label": "SureChEMBL ID"
    },
    "qid": {
      "description": "chemical compound",
      "label": "suprofen"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "chemical compound",
      "label": "suprofen",
      "properties": {
        "P2877": [
          "SCHEMBL23792"
        ]
      },
      "qid": "Q3978097"
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
        "Q3978097": {
          "description": "chemical compound",
          "label": "suprofen"
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
      
```
