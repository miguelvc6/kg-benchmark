# Prompt Development Review

No LLM inference was run for this artifact.

Rendered prompts: `384`

## Example Schema Summary

| Task | Example count | Example schemas |
| --- | ---: | --- |
| `a_box_repair` | 0 | n/a |
| `t_box_repair` | 0 | n/a |
| `track_diagnosis` | 0 | n/a |

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: track_diagnosis

Classify the repair locus using only visible evidence. A_BOX edits a focus-entity claim. T_BOX edits a property constraint or schema rule. AMBIGUOUS means the visible evidence is not enough to choose between those repair loci.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the repair edits a claim on the focus entity.
- T_BOX means the repair edits a property constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to determine whether the repair locus is the focus entity claim
  or the property/schema rule.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.

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

## prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000001

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

## prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000001

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: track_diagnosis

Classify the repair locus using only visible evidence. A_BOX edits a focus-entity claim. T_BOX edits a property constraint or schema rule. AMBIGUOUS means the visible evidence is not enough to choose between those repair loci.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the repair edits a claim on the focus entity.
- T_BOX means the repair edits a property constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to determine whether the repair locus is the focus entity claim
  or the property/schema rule.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.

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
        "P106": [
          "Q36180"
        ],
        "P131": [
          "Q677037",
          "Q59259864"
        ],
        "P18": [
          "Sriramoju Haragopal.jpg"
        ],
        "P21": [
          "Q6581097"
        ],
        "P27": [
          "Q668"
        ],
        "P31": [
          "Q5"
        ],
        "P373": [
          "Sriramoju Haragopal"
        ],
        "P569": [
          "+1957-03-25T00:00:00Z"
        ],
        "P6886": [
          "Q8097"
        ]
      },
      "qid": "Q20563062"
    },
    "L2_labels": {
      "entities": {
        "P106": {
          "description": "occupation of a person. See also \"field of work\" (Property:P101), \"position held\" (Property:P39). Not for groups of people. There, use \"field of work\" (Property:P101), \"industry\" (Property:P452), \"members have occupation\" (Property:P3989).",
          "label": "occupation"
        },
        "P131": {
          "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
          "label": "located in the administrative territorial entity"
        },
        "P21": {
          "description": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
          "label": "sex or gender"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P27": {
          "description": "the object is a country that recognizes the subject as its citizen",
          "label": "country of citizenship"
        },
        "P31": {
          "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
          "label": "instance of"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "P6886": {
          "description": "language in which the writer has written their work",
          "label": "writing language"
        },
        "Q20563062": {
          "description": "Poet, Writer and Historian",
          "label": "Sriramoju Haragopal"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q36180": {
          "description": "person who uses written words to communicate ideas and to produce literary works",
          "label": "writer"
        },
        "Q5": {
          "description": "any single member of Homo sapiens, unique extant species of the genus Homo",
          "label": "human"
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
        "Q6581097": {
          "description": "to be used in \"sex or gender\" (P21) to indicate that the human subject
```

## prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000001

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
        "P106": [
          "Q36180"
        ],
        "P131": [
          "Q677037",
          "Q59259864"
        ],
        "P18": [
          "Sriramoju Haragopal.jpg"
        ],
        "P21": [
          "Q6581097"
        ],
        "P27": [
          "Q668"
        ],
        "P31": [
          "Q5"
        ],
        "P373": [
          "Sriramoju Haragopal"
        ],
        "P569": [
          "+1957-03-25T00:00:00Z"
        ],
        "P6886": [
          "Q8097"
        ]
      },
      "qid": "Q20563062"
    },
    "L2_labels": {
      "entities": {
        "P106": {
          "description": "occupation of a person. See also \"field of work\" (Property:P101), \"position held\" (Property:P39). Not for groups of people. There, use \"field of work\" (Property:P101), \"industry\" (Property:P452), \"members have occupation\" (Property:P3989).",
          "label": "occupation"
        },
        "P131": {
          "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
          "label": "located in the administrative territorial entity"
        },
        "P21": {
          "description": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
          "label": "sex or gender"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P27": {
          "description": "the object is a country that recognizes the subject as its citizen",
          "label": "country of citizenship"
        },
        "P31": {
          "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
          "label": "instance of"
        },
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "P6886": {
          "description": "language in which the writer has written their work",
          "label": "writing language"
        },
        "Q20563062": {
          "description": "Poet, Writer and Historian",
          "label": "Sriramoju Haragopal"
        },
        "Q29934200": {
          "description": "entity type for Wikibase items",
          "label": "Wikibase item"
        },
        "Q36180": {
          "description": "person who uses written words to communicate ideas and to produce literary works",
          "label": "writer"
        },
        "Q5": {
          "description": "any single member of Homo sapiens, unique extant species of the genus Homo",
          "label": "human"
        },
        "Q52004125": {
          "description": "type of constraint for Wikidata properties: used to specify that a property may only be used on a certain listed entity type: Wikibase item, Wikibase property, lexeme, form, sense, Wikibase Medi
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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: track_diagnosis

Classify the repair locus using only visible evidence. A_BOX edits a focus-entity claim. T_BOX edits a property constraint or schema rule. AMBIGUOUS means the visible evidence is not enough to choose between those repair loci.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the repair edits a claim on the focus entity.
- T_BOX means the repair edits a property constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to determine whether the repair locus is the focus entity claim
  or the property/schema rule.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.

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

## prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000002

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: t_box_repair

Propose an executable T-box schema reform for the focus property. Use only visible evidence and keep constraint-family identifiers distinct from ordinary entity/type values.

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
Field definitions:
- target.pid is the focus property identifier from the input.
- target.constraint_type_qid is the constraint-family identifier being edited.
- proposal.action must be one of the listed enum values.
- proposal.signature_after is the proposed post-repair constraint signature when visible evidence supports specifying
  one.
- signature_after[*].constraint_qid is a constraint-family QID, not an ordinary item/type value.
- signature_after[*].qualifiers[*].values are qualifier values inside the constraint signature.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Keep constraint-family QIDs separate from ordinary entity/type QIDs and qualifier values.
- Do not copy report_violation_type_qids into target.constraint_type_qid or signature_after unless the prompt visibly
  presents the same QID in that schema role.
- Do not use hidden benchmark classes, subtypes, or historical labels.

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

## prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000002

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: track_diagnosis

Classify the repair locus using only visible evidence. A_BOX edits a focus-entity claim. T_BOX edits a property constraint or schema rule. AMBIGUOUS means the visible evidence is not enough to choose between those repair loci.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the repair edits a claim on the focus entity.
- T_BOX means the repair edits a property constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to determine whether the repair locus is the focus entity claim
  or the property/schema rule.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.

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
      "properties": {
        "P123": [
          "Q726081"
        ],
        "P1433": [
          "Q845016"
        ],
        "P1476": [
          "永井豪SF怪奇傑作選 邪神戦記@ja"
        ],
        "P2360": [
          "Q237338"
        ],
        "P2635": [
          "+1 http://www.wikidata.org/entity/Q88392887"
        ],
        "P407": [
          "Q5287"
        ],
        "P495": [
          "Q17"
        ],
        "P50": [
          "Q551359"
        ],
        "P577": [
          "+2013-01-04T00:00:00Z"
        ],
        "P5849": [
          "40778"
        ]
      },
      "qid": "Q115933554"
    },
    "L2_labels": {
      "entities": {
        "P123": {
          "description": "organization or person responsible for publishing a work, such as a book, periodical, printed music, podcast, game or software",
          "label": "publisher"
        },
        "P1433": {
          "description": "larger work that a given work was published in, like a journal, a website, a collection, a book or a music album",
          "label": "published in"
        },
        "P2360": {
          "description": "intended audience or user of this work, product, object, or event",
          "label": "intended public"
        },
        "P31": {
          "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
          "label": "instance of"
        },
        "P407": {
          "description": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
          "label": "language of work or name"
        },
        "P495": {
          "description": "country of origin of this item (creative work, food, phrase, product, etc.)",
          "label": "country of origin"
        },
        "P50": {
          "description": "main creator(s) of a written work (use on works, not humans); use P2093 (author name string) when Wikidata item is unknown or does not exist",
          "label": "author"
        },
        "Q115933554": {
          "description": "Manga di Gō Nagai",
          "label": "Cronache delle guerre demoniache"
        },
        "Q17": {
          "description": "island country in East Asia",
          "label": "Japan"
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
        "Q237338": {
          "description": "manga marketed primarily to late adolescent young adults (18 yr and older)",
          "label": "seinen"
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
        "Q5287": {
          "description": "language spoken in East Asia",
          "label": "Japanese"
        },
        "Q53869507": {
          "description": "constraint to define the scope of the property (as main property, as qualifier, as 
```

## prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000002

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: t_box_repair

Propose an executable T-box schema reform for the focus property. Use only visible evidence and keep constraint-family identifiers distinct from ordinary entity/type values.

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
Field definitions:
- target.pid is the focus property identifier from the input.
- target.constraint_type_qid is the constraint-family identifier being edited.
- proposal.action must be one of the listed enum values.
- proposal.signature_after is the proposed post-repair constraint signature when visible evidence supports specifying
  one.
- signature_after[*].constraint_qid is a constraint-family QID, not an ordinary item/type value.
- signature_after[*].qualifiers[*].values are qualifier values inside the constraint signature.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Keep constraint-family QIDs separate from ordinary entity/type QIDs and qualifier values.
- Do not copy report_violation_type_qids into target.constraint_type_qid or signature_after unless the prompt visibly
  presents the same QID in that schema role.
- Do not use hidden benchmark classes, subtypes, or historical labels.

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
      "properties": {
        "P123": [
          "Q726081"
        ],
        "P1433": [
          "Q845016"
        ],
        "P1476": [
          "永井豪SF怪奇傑作選 邪神戦記@ja"
        ],
        "P2360": [
          "Q237338"
        ],
        "P2635": [
          "+1 http://www.wikidata.org/entity/Q88392887"
        ],
        "P407": [
          "Q5287"
        ],
        "P495": [
          "Q17"
        ],
        "P50": [
          "Q551359"
        ],
        "P577": [
          "+2013-01-04T00:00:00Z"
        ],
        "P5849": [
          "40778"
        ]
      },
      "qid": "Q115933554"
    },
    "L2_labels": {
      "entities": {
        "P123": {
          "description": "organization or person responsible for publishing a work, such as a book, periodical, printed music, podcast, game or software",
          "label": "publisher"
        },
        "P1433": {
          "description": "larger work that a given work was published in, like a journal, a website, a collection, a book or a music album",
          "label": "published in"
        },
        "P2360": {
          "description": "intended audience or user of this work, product, object, or event",
          "label": "intended public"
        },
        "P31": {
          "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
          "label": "instance of"
        },
        "P407": {
          "description": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
          "label": "language of work or name"
        },
        "P495": {
          "description": "country of origin of this item (creative work, food, phrase, product, etc.)",
          "label": "country of origin"
        },
        "P50": {
          "description": "main creator(s) of a written work (use on works, not humans); use P2093 (author name string) when Wikidata item is unknown or does not exist",
          "label": "author"
        },
        "Q115933554": {
          "description": "Manga di Gō Nagai",
          "label": "Cronache delle guerre demoniache"
        },
        "Q17": {
          "description": "island country in East Asia",
          "label": "Japan"
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
          "label": "value-requir
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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: track_diagnosis

Classify the repair locus using only visible evidence. A_BOX edits a focus-entity claim. T_BOX edits a property constraint or schema rule. AMBIGUOUS means the visible evidence is not enough to choose between those repair loci.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the repair edits a claim on the focus entity.
- T_BOX means the repair edits a property constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to determine whether the repair locus is the focus entity claim
  or the property/schema rule.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.

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

## prompt_dev_002_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000003

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: t_box_repair

Propose an executable T-box schema reform for the focus property. Use only visible evidence and keep constraint-family identifiers distinct from ordinary entity/type values.

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
Field definitions:
- target.pid is the focus property identifier from the input.
- target.constraint_type_qid is the constraint-family identifier being edited.
- proposal.action must be one of the listed enum values.
- proposal.signature_after is the proposed post-repair constraint signature when visible evidence supports specifying
  one.
- signature_after[*].constraint_qid is a constraint-family QID, not an ordinary item/type value.
- signature_after[*].qualifiers[*].values are qualifier values inside the constraint signature.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Keep constraint-family QIDs separate from ordinary entity/type QIDs and qualifier values.
- Do not copy report_violation_type_qids into target.constraint_type_qid or signature_after unless the prompt visibly
  presents the same QID in that schema role.
- Do not use hidden benchmark classes, subtypes, or historical labels.

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

## prompt_dev_003_hybrid_json_nl_zero_shot_local_graph_track_diagnosis_diagnosis_no_abstain / case_000003

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: track_diagnosis

Classify the repair locus using only visible evidence. A_BOX edits a focus-entity claim. T_BOX edits a property constraint or schema rule. AMBIGUOUS means the visible evidence is not enough to choose between those repair loci.

Output contract:

Return exactly one JSON object:
{
  "case_id": "<copy input id exactly; this is the neutral prompt-visible id>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
Definitions:
- A_BOX means the repair edits a claim on the focus entity.
- T_BOX means the repair edits a property constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to determine whether the repair locus is the focus entity claim
  or the property/schema rule.
Evidence boundary:
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.

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
      "properties": {
        "P1057": [
          "Q847102"
        ],
        "P2548": [
          "Q22809711"
        ],
        "P279": [
          "Q20747295"
        ],
        "P2888": [
          "http://identifiers.org/ncbigene/10060"
        ],
        "P2892": [
          "C1412083"
        ],
        "P31": [
          "Q7187"
        ],
        "P351": [
          "10060"
        ],
        "P353": [
          "ABCC9"
        ],
        "P354": [
          "60"
        ],
        "P4196": [
          "12p12.1"
        ],
        "P492": [
          "601439"
        ],
        "P5572": [
          "Q943203",
          "Q66514508",
          "Q223172",
          "Q876089",
          "Q66502809",
          "Q6493472",
          "Q383249",
          "Q66592424",
          "Q116529247",
          "Q66508550"
        ],
        "P593": [
          "56521"
        ],
        "P594": [
          "ENSG00000069431"
        ],
        "P6366": [
          "2778528692"
        ],
        "P639": [
          "XM_011520545",
          "NM_005691",
          "NM_020297",
          "NM_020298",
          "XM_005253288",
          "XM_005253289",
          "XM_005253290",
          "NM_001377273",
          "NM_001377274"
        ],
        "P644": [
          "21950335",
          "21797389"
        ],
        "P645": [
          "22094336",
          "21942543"
        ],
        "P646": [
          "/m/02kfsxd"
        ],
        "P684": [
          "Q18254413",
          "Q24396222",
          "Q29778313",
          "Q29733788"
        ],
        "P688": [
          "Q21111140",
          "Q21137270",
          "Q21138267"
        ],
        "P703": [
          "Q15978631"
        ],
        "P704": [
          "ENST00000621589",
          "ENST00000261200",
          "ENST00000261201",
          "ENST00000326684",
          "ENST00000538350",
          "ENST00000544039",
          "ENST00000636888",
          "ENST00000682068",
          "ENST00000682789",
          "ENST00000682426",
          "ENST00000683695",
          "ENST00000683811"
        ]
      },
      "qid": "Q18034993"
    },
    "L2_labels": {
      "entities": {
        "P1057": {
          "description": "chromosome on which an entity is localized",
          "label": "chromosome"
        },
        "P2293": {
          "description": "general link between a disease and the causal genetic entity, if the detailed mechanism is unknown/unavailable",
          "label": "genetic association"
        },
        "P2548": {
          "description": "orientation of gene on double stranded DNA molecule",
          "label": "strand orientation"
        },
        "P279": {
          "description": "this item is a subclass (subset) of that item; ALL instances of this item are instances of that item; different from P31 (instance of), e.g.: volcano is a subclass of mountain; Everest is an instance of mountain",
          "label": "subclass of"
        },
        "P31": {
          "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
          "label": "instance of"
        },
        "P5572": {
          "description": "gene or protein is expressed during a specific condition/cell cycle/process/form",
          "label": "expressed in"
        },
        "P684": {
          "description": "orthologous gene in another species (use with 'species' qualifier)",
          "label": "ortholog"
        },
        "P688": {
          "description": "the product of a gene (protein or RNA)",
          "label": "encodes"
        },
        "P703": {
          "description": "the taxon in which the item can be found",
          "label": "found in taxon"
        },
        "Q116529247": {
          "description": "cell type",
          "label": "buccal mucosa cell"
        },
        "Q15978631": {
          "description": "species of mammal",
          "label": "Homo sapiens"
        },
        "Q18034993": {
          "description": "protein-coding gene in the species Homo sapiens",
          "label": "ABCC9"
        },
        "Q18254413": {
          "description": "protein-coding gene in the species Mus musculus",
          "label": "Abcc9"
        },
        "Q20747295": {
          "description": "type of a gene
```

## prompt_dev_004_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000003

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
Prompt version: prompt_dev_v4_spec_only

Representation: hybrid_json_nl

Task: t_box_repair

Propose an executable T-box schema reform for the focus property. Use only visible evidence and keep constraint-family identifiers distinct from ordinary entity/type values.

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
Field definitions:
- target.pid is the focus property identifier from the input.
- target.constraint_type_qid is the constraint-family identifier being edited.
- proposal.action must be one of the listed enum values.
- proposal.signature_after is the proposed post-repair constraint signature when visible evidence supports specifying
  one.
- signature_after[*].constraint_qid is a constraint-family QID, not an ordinary item/type value.
- signature_after[*].qualifiers[*].values are qualifier values inside the constraint signature.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
Evidence boundary:
- Use only visible prompt evidence.
- Keep constraint-family QIDs separate from ordinary entity/type QIDs and qualifier values.
- Do not copy report_violation_type_qids into target.constraint_type_qid or signature_after unless the prompt visibly
  presents the same QID in that schema role.
- Do not use hidden benchmark classes, subtypes, or historical labels.

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
      "properties": {
        "P1057": [
          "Q847102"
        ],
        "P2548": [
          "Q22809711"
        ],
        "P279": [
          "Q20747295"
        ],
        "P2888": [
          "http://identifiers.org/ncbigene/10060"
        ],
        "P2892": [
          "C1412083"
        ],
        "P31": [
          "Q7187"
        ],
        "P351": [
          "10060"
        ],
        "P353": [
          "ABCC9"
        ],
        "P354": [
          "60"
        ],
        "P4196": [
          "12p12.1"
        ],
        "P492": [
          "601439"
        ],
        "P5572": [
          "Q943203",
          "Q66514508",
          "Q223172",
          "Q876089",
          "Q66502809",
          "Q6493472",
          "Q383249",
          "Q66592424",
          "Q116529247",
          "Q66508550"
        ],
        "P593": [
          "56521"
        ],
        "P594": [
          "ENSG00000069431"
        ],
        "P6366": [
          "2778528692"
        ],
        "P639": [
          "XM_011520545",
          "NM_005691",
          "NM_020297",
          "NM_020298",
          "XM_005253288",
          "XM_005253289",
          "XM_005253290",
          "NM_001377273",
          "NM_001377274"
        ],
        "P644": [
          "21950335",
          "21797389"
        ],
        "P645": [
          "22094336",
          "21942543"
        ],
        "P646": [
          "/m/02kfsxd"
        ],
        "P684": [
          "Q18254413",
          "Q24396222",
          "Q29778313",
          "Q29733788"
        ],
        "P688": [
          "Q21111140",
          "Q21137270",
          "Q21138267"
        ],
        "P703": [
          "Q15978631"
        ],
        "P704": [
          "ENST00000621589",
          "ENST00000261200",
          "ENST00000261201",
          "ENST00000326684",
          "ENST00000538350",
          "ENST00000544039",
          "ENST00000636888",
          "ENST00000682068",
          "ENST00000682789",
          "ENST00000682426",
          "ENST00000683695",
          "ENST00000683811"
        ]
      },
      "qid": "Q18034993"
    },
    "L2_labels": {
      "entities": {
        "P1057": {
          "description": "chromosome on which an entity is localized",
          "label": "chromosome"
        },
        "P2293": {
          "description": "general link between a disease and the causal genetic entity, if the detailed mechanism is unknown/unavailable",
          "label": "genetic association"
        },
        "P2548": {
          "description": "orientation of gene on double stranded DNA molecule",
          "label": "strand orientation"
        },
        "P279": {
          "description": "this item is a subclass (subset) of that item; ALL instances of this item are instances of that item; different from P31 (instance of), e.g.: volcano is a subclass of mountain; Everest is an instance of mountain",
          "label": "subclass of"
        },
        "P31": {
          "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
 
```
