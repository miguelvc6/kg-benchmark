# Prompt Development Review

No LLM inference was run for this artifact.

Rendered prompts: `192`

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / case_000001

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
Prompt version: prompt_dev_diag_v1_locus_spec

Representation: hybrid_json_nl

Task: track_diagnosis

Diagnose the repair locus using only visible evidence. A_BOX means the repair belongs on the focus entity claim. T_BOX means the repair belongs on the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support a safe routing choice.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the most likely repair locus is the focus entity's claim for the target property.
- T_BOX means the most likely repair locus is the target property's constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to choose between those repair loci.
Repair-locus semantics:
- Decide where the repair should be applied, not which vocabulary appears in the violation report.
- A constraint report can be resolved by an A_BOX claim edit or a T_BOX schema edit; choose the locus supported by the
  visible evidence.
- Choose AMBIGUOUS only when the visible evidence leaves both repair loci plausible or neither repair locus supported.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
- Do not use case-id prefixes, raw benchmark identifiers, or provenance outside the prompt.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

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

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000001

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
Prompt version: prompt_dev_diag_v1_locus_spec

Representation: hybrid_json_nl

Task: track_diagnosis

Diagnose the repair locus using only visible evidence. A_BOX means the repair belongs on the focus entity claim. T_BOX means the repair belongs on the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support a safe routing choice.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the most likely repair locus is the focus entity's claim for the target property.
- T_BOX means the most likely repair locus is the target property's constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to choose between those repair loci.
Repair-locus semantics:
- Decide where the repair should be applied, not which vocabulary appears in the violation report.
- A constraint report can be resolved by an A_BOX claim edit or a T_BOX schema edit; choose the locus supported by the
  visible evidence.
- Choose AMBIGUOUS only when the visible evidence leaves both repair loci plausible or neither repair locus supported.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
- Do not use case-id prefixes, raw benchmark identifiers, or provenance outside the prompt.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

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
          "rule_summary": "item of property constraint (P2305): Wikibase i
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / case_000002

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
Prompt version: prompt_dev_diag_v1_locus_spec

Representation: hybrid_json_nl

Task: track_diagnosis

Diagnose the repair locus using only visible evidence. A_BOX means the repair belongs on the focus entity claim. T_BOX means the repair belongs on the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support a safe routing choice.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the most likely repair locus is the focus entity's claim for the target property.
- T_BOX means the most likely repair locus is the target property's constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to choose between those repair loci.
Repair-locus semantics:
- Decide where the repair should be applied, not which vocabulary appears in the violation report.
- A constraint report can be resolved by an A_BOX claim edit or a T_BOX schema edit; choose the locus supported by the
  visible evidence.
- Choose AMBIGUOUS only when the visible evidence leaves both repair loci plausible or neither repair locus supported.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
- Do not use case-id prefixes, raw benchmark identifiers, or provenance outside the prompt.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

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

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000002

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
Prompt version: prompt_dev_diag_v1_locus_spec

Representation: hybrid_json_nl

Task: track_diagnosis

Diagnose the repair locus using only visible evidence. A_BOX means the repair belongs on the focus entity claim. T_BOX means the repair belongs on the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support a safe routing choice.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the most likely repair locus is the focus entity's claim for the target property.
- T_BOX means the most likely repair locus is the target property's constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to choose between those repair loci.
Repair-locus semantics:
- Decide where the repair should be applied, not which vocabulary appears in the violation report.
- A constraint report can be resolved by an A_BOX claim edit or a T_BOX schema edit; choose the locus supported by the
  visible evidence.
- Choose AMBIGUOUS only when the visible evidence leaves both repair loci plausible or neither repair locus supported.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
- Do not use case-id prefixes, raw benchmark identifiers, or provenance outside the prompt.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

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

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / case_000003

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
Prompt version: prompt_dev_diag_v1_locus_spec

Representation: hybrid_json_nl

Task: track_diagnosis

Diagnose the repair locus using only visible evidence. A_BOX means the repair belongs on the focus entity claim. T_BOX means the repair belongs on the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support a safe routing choice.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the most likely repair locus is the focus entity's claim for the target property.
- T_BOX means the most likely repair locus is the target property's constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to choose between those repair loci.
Repair-locus semantics:
- Decide where the repair should be applied, not which vocabulary appears in the violation report.
- A constraint report can be resolved by an A_BOX claim edit or a T_BOX schema edit; choose the locus supported by the
  visible evidence.
- Choose AMBIGUOUS only when the visible evidence leaves both repair loci plausible or neither repair locus supported.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
- Do not use case-id prefixes, raw benchmark identifiers, or provenance outside the prompt.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000003",
  "labels_en": {
    "property": {
      "description": "general link between a disease and the causal genetic entity, if the detailed mechanism is unknown/unavailable",
      "label": "genetic association"
    },
    "qid": {
      "description": "protein-coding gene in the species Homo sapiens",
      "label": "ABCC9"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q21503250",
        "label": "subject type constraint"
      },
      {
        "constraint_qid": "Q53869507",
        "label": "property scope constraint"
      }
    ],
    "violation_context": {
      "report_page_title": "Wikidata:Database reports/Constraint violations/P2293",
      "report_violation_type": "Symmetric",
      "report_violation_type_normalized": "Symmetric",
      "report_violation_type_qids": [],
      "report_violation_type_raw": "Symmetric"
    }
  },
  "property": "P2293",
  "qid": "Q18034993",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P2293",
    "report_violation_type": "Symmetric",
    "report_violation_type_normalized": "Symmetric",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Symmetric"
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000003

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
Prompt version: prompt_dev_diag_v1_locus_spec

Representation: hybrid_json_nl

Task: track_diagnosis

Diagnose the repair locus using only visible evidence. A_BOX means the repair belongs on the focus entity claim. T_BOX means the repair belongs on the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support a safe routing choice.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the most likely repair locus is the focus entity's claim for the target property.
- T_BOX means the most likely repair locus is the target property's constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to choose between those repair loci.
Repair-locus semantics:
- Decide where the repair should be applied, not which vocabulary appears in the violation report.
- A constraint report can be resolved by an A_BOX claim edit or a T_BOX schema edit; choose the locus supported by the
  visible evidence.
- Choose AMBIGUOUS only when the visible evidence leaves both repair loci plausible or neither repair locus supported.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
- Do not use case-id prefixes, raw benchmark identifiers, or provenance outside the prompt.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000003",
  "labels_en": {
    "property": {
      "description": "general link between a disease and the causal genetic entity, if the detailed mechanism is unknown/unavailable",
      "label": "genetic association"
    },
    "qid": {
      "description": "protein-coding gene in the species Homo sapiens",
      "label": "ABCC9"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "protein-coding gene in the species Homo sapiens",
      "label": "ABCC9",
      "qid": "Q18034993"
    },
    "L2_labels": {
      "entities": {
        "P2293": {
          "description": "general link between a disease and the causal genetic entity, if the detailed mechanism is unknown/unavailable",
          "label": "genetic association"
        },
        "Q18034993": {
          "description": "protein-coding gene in the species Homo sapiens",
          "label": "ABCC9"
        },
        "Q21503250": {
          "description": "type of constraint for Wikidata properties: used to specify that the item described by such properties should be a subclass or instance of a given type",
          "label": "subject type constraint"
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
          "constraint_qid": "Q53869507",
          "label": "property scope constraint"
        }
      ],
      "violation_context": {
        "report_page_title": "Wikidata:Database reports/Constraint violations/P2293",
        "report_violation_type": "Symmetric",
        "report_violation_type_normalized": "Symmetric",
        "report_violation_type_qids": [],
        "report_violation_type_raw": "Symmetric"
      }
    }
  },
  "property": "P2293",
  "qid": "Q18034993",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P2293",
    "report_violation_type": "Symmetric",
    "report_violation_type_normalized": "Symmetric",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Symmetric"
  }
}
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / case_000004

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
Prompt version: prompt_dev_diag_v1_locus_spec

Representation: hybrid_json_nl

Task: track_diagnosis

Diagnose the repair locus using only visible evidence. A_BOX means the repair belongs on the focus entity claim. T_BOX means the repair belongs on the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support a safe routing choice.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the most likely repair locus is the focus entity's claim for the target property.
- T_BOX means the most likely repair locus is the target property's constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to choose between those repair loci.
Repair-locus semantics:
- Decide where the repair should be applied, not which vocabulary appears in the violation report.
- A constraint report can be resolved by an A_BOX claim edit or a T_BOX schema edit; choose the locus supported by the
  visible evidence.
- Choose AMBIGUOUS only when the visible evidence leaves both repair loci plausible or neither repair locus supported.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
- Do not use case-id prefixes, raw benchmark identifiers, or provenance outside the prompt.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000004",
  "labels_en": {
    "property": {
      "description": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
      "label": "title"
    },
    "qid": {
      "description": "daily newspaper in Atlanta, Georgia",
      "label": "The Atlanta Journal-Constitution"
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
          },
          {
            "property_id": "P4680",
            "property_label": "constraint scope",
            "values": [
              {
                "label": "constraint checked on main value",
                "qid": "Q46466787",
                "raw": "Q46466787"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), МэдыяІнфа Вікібазы (Q59712033); constraint scope (P4680): constraint checked on main value (Q46466787)"
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
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448), as qualifier (Q54828449), as reference (Q54828450)"
      }
    ],
    "property_id": "P1476"
  },
  "property": "P1476",
  "qid": "Q3092348",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P1476",
    "report_violation_type": "Single value",
    "report_violation_type_normalized": "Single value",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Single value",
    "value": [
      "The Atlanta Journal-Constitution@en",
      "Atlanta Journal-Constitution@en"
    ]
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000004

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
Prompt version: prompt_dev_diag_v1_locus_spec

Representation: hybrid_json_nl

Task: track_diagnosis

Diagnose the repair locus using only visible evidence. A_BOX means the repair belongs on the focus entity claim. T_BOX means the repair belongs on the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support a safe routing choice.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the most likely repair locus is the focus entity's claim for the target property.
- T_BOX means the most likely repair locus is the target property's constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to choose between those repair loci.
Repair-locus semantics:
- Decide where the repair should be applied, not which vocabulary appears in the violation report.
- A constraint report can be resolved by an A_BOX claim edit or a T_BOX schema edit; choose the locus supported by the
  visible evidence.
- Choose AMBIGUOUS only when the visible evidence leaves both repair loci plausible or neither repair locus supported.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
- Do not use case-id prefixes, raw benchmark identifiers, or provenance outside the prompt.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000004",
  "labels_en": {
    "property": {
      "description": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
      "label": "title"
    },
    "qid": {
      "description": "daily newspaper in Atlanta, Georgia",
      "label": "The Atlanta Journal-Constitution"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "daily newspaper in Atlanta, Georgia",
      "label": "The Atlanta Journal-Constitution",
      "properties": {
        "P1476": [
          "The Atlanta Journal-Constitution@en",
          "Atlanta Journal-Constitution@en"
        ]
      },
      "qid": "Q3092348"
    },
    "L2_labels": {
      "entities": {
        "P1476": {
          "description": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
          "label": "title"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P4680": {
          "description": "defines the scope where a constraint is checked – can specify the usage scope (main value of a statement, on qualifiers, or on references) and the datatype (Wikibase item, property, lexeme, etc.)",
          "label": "constraint scope"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q3092348": {
          "description": "daily newspaper in Atlanta, Georgia",
          "label": "The Atlanta Journal-Constitution"
        },
        "Q46466787": {
          "description": "scope for constraints that should be checked on the main value of a statement",
          "label": "constraint checked on main value"
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
            },
            {
              "property_id": "P4680",
              "property_label": "constraint scope",
              "values": [
                {
                  "label": "constraint checked on main value",
                  "qid": "Q46466787",
                  "raw": "Q46466787"
                }
              ]
            }
          ],
          "rule_summary": "item of property constraint (P2305): Wikibase it
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / case_000005

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
Prompt version: prompt_dev_diag_v1_locus_spec

Representation: hybrid_json_nl

Task: track_diagnosis

Diagnose the repair locus using only visible evidence. A_BOX means the repair belongs on the focus entity claim. T_BOX means the repair belongs on the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support a safe routing choice.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the most likely repair locus is the focus entity's claim for the target property.
- T_BOX means the most likely repair locus is the target property's constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to choose between those repair loci.
Repair-locus semantics:
- Decide where the repair should be applied, not which vocabulary appears in the violation report.
- A constraint report can be resolved by an A_BOX claim edit or a T_BOX schema edit; choose the locus supported by the
  visible evidence.
- Choose AMBIGUOUS only when the visible evidence leaves both repair loci plausible or neither repair locus supported.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
- Do not use case-id prefixes, raw benchmark identifiers, or provenance outside the prompt.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000005",
  "labels_en": {
    "property": {
      "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
      "label": "SureChEMBL ID"
    },
    "qid": {
      "description": "chemical compound",
      "label": "isoxaben"
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
  "qid": "Q2677285",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P2877",
    "report_violation_type": "Format",
    "report_violation_type_normalized": "Format",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Format",
    "value": [
      "SCHEMBL54432"
    ]
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000005

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
Prompt version: prompt_dev_diag_v1_locus_spec

Representation: hybrid_json_nl

Task: track_diagnosis

Diagnose the repair locus using only visible evidence. A_BOX means the repair belongs on the focus entity claim. T_BOX means the repair belongs on the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support a safe routing choice.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the most likely repair locus is the focus entity's claim for the target property.
- T_BOX means the most likely repair locus is the target property's constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to choose between those repair loci.
Repair-locus semantics:
- Decide where the repair should be applied, not which vocabulary appears in the violation report.
- A constraint report can be resolved by an A_BOX claim edit or a T_BOX schema edit; choose the locus supported by the
  visible evidence.
- Choose AMBIGUOUS only when the visible evidence leaves both repair loci plausible or neither repair locus supported.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
- Do not use case-id prefixes, raw benchmark identifiers, or provenance outside the prompt.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000005",
  "labels_en": {
    "property": {
      "description": "the chemical compound identifier for the EBI SureChEMBL 'chemical compounds in patents' database",
      "label": "SureChEMBL ID"
    },
    "qid": {
      "description": "chemical compound",
      "label": "isoxaben"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "chemical compound",
      "label": "isoxaben",
      "properties": {
        "P2877": [
          "SCHEMBL54432"
        ]
      },
      "qid": "Q2677285"
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
        "Q2677285": {
          "description": "chemical compound",
          "label": "isoxaben"
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
      
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_track_diagnosis_diagnosis_no_abstain / case_000006

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
Prompt version: prompt_dev_diag_v1_locus_spec

Representation: hybrid_json_nl

Task: track_diagnosis

Diagnose the repair locus using only visible evidence. A_BOX means the repair belongs on the focus entity claim. T_BOX means the repair belongs on the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support a safe routing choice.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the most likely repair locus is the focus entity's claim for the target property.
- T_BOX means the most likely repair locus is the target property's constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to choose between those repair loci.
Repair-locus semantics:
- Decide where the repair should be applied, not which vocabulary appears in the violation report.
- A constraint report can be resolved by an A_BOX claim edit or a T_BOX schema edit; choose the locus supported by the
  visible evidence.
- Choose AMBIGUOUS only when the visible evidence leaves both repair loci plausible or neither repair locus supported.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
- Do not use case-id prefixes, raw benchmark identifiers, or provenance outside the prompt.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000006",
  "labels_en": {
    "property": {
      "description": "identifier of legislation on the irishstatutebook.ie website",
      "label": "Irish Statute Book ID"
    },
    "qid": {
      "description": "Irish Statutory Instrument S.I. No. 289/2015",
      "label": "European Union (Bank Recovery and Resolution) Regulations 2015"
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
                "raw": "[12]\\d{3}/(act/\\d{1,2}/enacted|si/\\d{1,3}/made)"
              }
            ]
          }
        ],
        "rule_summary": "format as a regular expression (P1793): [12]\\d{3}/(act/\\d{1,2}/enacted|si/\\d{1,3}/made)"
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
              }
            ]
          }
        ],
        "rule_summary": "property scope (P5314): as main value (Q54828448)"
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
    "property_id": "P8726"
  },
  "property": "P8726",
  "qid": "Q100541019",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P8726",
    "report_violation_type": "Format",
    "report_violation_type_normalized": "Format",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Format",
    "value": [
      "MISSING"
    ]
  }
}
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000006

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
Prompt version: prompt_dev_diag_v1_locus_spec

Representation: hybrid_json_nl

Task: track_diagnosis

Diagnose the repair locus using only visible evidence. A_BOX means the repair belongs on the focus entity claim. T_BOX means the repair belongs on the property constraint or schema rule. AMBIGUOUS means the visible evidence does not support a safe routing choice.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the most likely repair locus is the focus entity's claim for the target property.
- T_BOX means the most likely repair locus is the target property's constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to choose between those repair loci.
Repair-locus semantics:
- Decide where the repair should be applied, not which vocabulary appears in the violation report.
- A constraint report can be resolved by an A_BOX claim edit or a T_BOX schema edit; choose the locus supported by the
  visible evidence.
- Choose AMBIGUOUS only when the visible evidence leaves both repair loci plausible or neither repair locus supported.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
- Do not use case-id prefixes, raw benchmark identifiers, or provenance outside the prompt.

Few-shot examples:

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000006",
  "labels_en": {
    "property": {
      "description": "identifier of legislation on the irishstatutebook.ie website",
      "label": "Irish Statute Book ID"
    },
    "qid": {
      "description": "Irish Statutory Instrument S.I. No. 289/2015",
      "label": "European Union (Bank Recovery and Resolution) Regulations 2015"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "Irish Statutory Instrument S.I. No. 289/2015",
      "label": "European Union (Bank Recovery and Resolution) Regulations 2015",
      "qid": "Q100541019"
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
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "P8726": {
          "description": "identifier of legislation on the irishstatutebook.ie website",
          "label": "Irish Statute Book ID"
        },
        "Q100541019": {
          "description": "Irish Statutory Instrument S.I. No. 289/2015",
          "label": "European Union (Bank Recovery and Resolution) Regulations 2015"
        },
        "Q21502404": {
          "description": "type of constraint for Wikidata properties: used to specify that the value for this property has to correspond to a given pattern",
          "label": "format constraint"
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
                  "raw": "[12]\\d{3}/(act/\\d{1,2}/enacted|si/\\d{1,3}/made)"
                }
              ]
            }
          ],
          "rule_summary": "format as a regular expression (P1793): [12]\\d{3}/(act/\\d{1,2}/enacted|si/\\d{1,3}/made)"
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
                }
              ]
            }
          ],
          "rule_summary": "property scope (P5314): as main value (Q54828448)"
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
```
