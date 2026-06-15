# Prompt Development Review

No LLM inference was run for this artifact.

Rendered prompts: `288`

## prompt_dev_001_hybrid_json_nl_zero_shot_diagnosis_minimal_track_diagnosis_diagnosis_no_abstain / case_000001

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `diagnosis_minimal`

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

## prompt_dev_002_hybrid_json_nl_zero_shot_diagnosis_logic_neutral_track_diagnosis_diagnosis_no_abstain / case_000001

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `diagnosis_logic_neutral`

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
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q21502838",
        "label": "conflicts-with constraint"
      },
      {
        "constraint_qid": "Q21503247",
        "label": "item-requires-statement constraint"
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
        "constraint_qid": "Q21510851",
        "label": "allowed qualifiers constraint"
      },
      {
        "constraint_qid": "Q21510864",
        "label": "value-requires-statement constraint"
      },
      {
        "constraint_qid": "Q53869507",
        "label": "property scope constraint"
      },
      {
        "constraint_qid": "Q52004125",
        "label": "allowed-entity-types constraint"
      }
    ],
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
    ]
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

## prompt_dev_003_hybrid_json_nl_zero_shot_diagnosis_local_neutral_track_diagnosis_diagnosis_no_abstain / case_000001

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `diagnosis_local_neutral`

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
        "P106": [
          "Q36180"
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
        "Q21502838": {
          "description": "type of constraint for Wikidata properties: used to specify that an item must not have a given statement",
          "label": "conflicts-with constraint"
        },
        "Q21503247": {
          "description": "type of constraint for Wikidata properties: used to specify that an item with this property should also have another given property",
          "label": "item-requires-statement constraint"
        },
        "Q21510851": {
          "description": "type of constraint for Wikidata properties: used to specify that only the listed qualifiers should be used. \"Novalue\" disallows any qualifier",
          "label": "allowed qualifiers constraint"
        },
        "Q21510864": {
          "description": "type of constraint for Wikidata properties: used to specify that the referenced item should have a statement with a given property",
          "label": "value-requires-statement constraint"
        },
        "Q21510865": {
          "description": "type of constraint for Wikidata properties: used to specify t
```

## prompt_dev_001_hybrid_json_nl_zero_shot_diagnosis_minimal_track_diagnosis_diagnosis_no_abstain / case_000002

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `diagnosis_minimal`

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

## prompt_dev_002_hybrid_json_nl_zero_shot_diagnosis_logic_neutral_track_diagnosis_diagnosis_no_abstain / case_000002

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `diagnosis_logic_neutral`

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
    "constraints": [
      {
        "constraint_type": {
          "label": "none-of constraint",
          "qid": "Q52558054"
        },
        "qualifiers": [
          {
            "property_id": "P2305",
            "property_label": "item of property constraint",
            "values": [
              {
                "label": "art",
                "qid": "Q735",
                "raw": "Q735"
              }
            ]
          },
          {
            "property_id": "P9729",
            "property_label": "replacement value",
            "values": [
              {
                "label": "work of art",
                "qid": "Q838948",
                "raw": "Q838948"
              }
            ]
          }
        ],
        "rule_summary": "item of property constraint (P2305): art (Q735); replacement value (P9729): work of art (Q838948)"
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
              },
              {
                "label": "МэдыяІнфа Вікібазы",
                "qid": "Q59712033",
                "raw": "Q59712033"
              },
              {
                "label": "Wikibase form",
                "qid": "Q54285143",
                "raw": "Q54285143"
              },
              {
                "label": "Wikibase sense",
                "qid": "Q54285715",
                "raw": "Q54285715"
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
        "rule_summary": "item of property constraint (P2305): Wikibase item (Q29934200), Wikibase property (Q29934218), Wikibase lexeme (Q51885771), МэдыяІнфа Вікібазы (Q59712033), Wikibase form (Q54285143), Wikibase sense (Q54285715); co
```

## prompt_dev_003_hybrid_json_nl_zero_shot_diagnosis_local_neutral_track_diagnosis_diagnosis_no_abstain / case_000002

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `diagnosis_local_neutral`

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
        "P101": {
          "description": "specialization of a person or organization; see P106 for the occupation",
          "label": "field of work"
        },
        "P106": {
          "description": "occupation of a person. See also \"field of work\" (Property:P101), \"position held\" (Property:P39). Not for groups of people. There, use \"field of work\" (Property:P101), \"industry\" (Property:P452), \"members have occupation\" (Property:P3989).",
          "label": "occupation"
        },
        "P123": {
          "description": "organization or person responsible for publishing a work, such as a book, periodical, printed music, podcast, game or software",
          "label": "publisher"
        },
        "P136": {
          "description": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
          "label": "genre"
        },
        "P1376": {
          "description": "country, state, department, canton or other administrative division of which the municipality is the governmental seat",
          "label": "capital of"
        },
        "P140": {
          "description": "religion of a person, organization or religious building, or associated with this subject",
          "label": "religion or worldview"
        },
        "P1433": {
          "description": "larger work that a given work was published in, like a journal, a website, a collection, a book or a music album",
          "label": "published in"
        },
        "P1435": {
          "description": "heritage designation of a cultural or natural site",
          "label": "heritage designation"
        },
        "P144": {
          "description": "the work(s) or inputs used as the basis for subject item; for fictional analog use P1074",
          "label": "based on"
        },
        "P1454": {
          "description": "legal form of an entity",
          "label": "legal form"
        },
        "P1552": {
          "description": "inherent or distinguishing quality or feature of the entity. Use a more specific property when possible",
          "label": "has characteristic"
        },
        "P2079": {
          "description": "method, process or technique used to grow, cook, weave, build, assemble, manufacture the item",
          "label": "fabrication method"
        },
        "P21": {
          "description": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
          "label": "sex or gender"
        },
        "P2303": {
          "description": "item that is an exception to the constraint, qualifier to define a property constraint in combination with P2302",
          "label": "exception to constraint"
        },
        "P2305": {
          "description": "qualifier to define a property constraint in combination with \"property constraint\" (P2302)",
          "label": "item of property constraint"
        },
        "P2316": {
          "description": "qualifier 
```

## prompt_dev_001_hybrid_json_nl_zero_shot_diagnosis_minimal_track_diagnosis_diagnosis_no_abstain / case_000003

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `diagnosis_minimal`

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

## prompt_dev_002_hybrid_json_nl_zero_shot_diagnosis_logic_neutral_track_diagnosis_diagnosis_no_abstain / case_000003

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `diagnosis_logic_neutral`

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
              }
            ]
          }
        ],
        "rule_summary": "constraint status (P2316): mandatory constraint (Q21502408); property scope (P5314): as main value (Q54828448)"
      }
    ]
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

## prompt_dev_003_hybrid_json_nl_zero_shot_diagnosis_local_neutral_track_diagnosis_diagnosis_no_abstain / case_000003

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `diagnosis_local_neutral`

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
        "P2316": {
          "description": "qualifier to define a property constraint in combination with P2302. Use values \"mandatory constraint\" or \"suggestion constraint\"",
          "label": "constraint status"
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
        "P5314": {
          "description": "constraint system qualifier to define the scope of a property",
          "label": "property scope"
        },
        "P5572": {
          "description": "gene or protein is expressed during a specific condition/cell cycle/process/form",
          "label": "expressed in"
        },
        "P684": {
          "description": "orthologous gene in anothe
```

## prompt_dev_001_hybrid_json_nl_zero_shot_diagnosis_minimal_track_diagnosis_diagnosis_no_abstain / case_000004

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `diagnosis_minimal`

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

## prompt_dev_002_hybrid_json_nl_zero_shot_diagnosis_logic_neutral_track_diagnosis_diagnosis_no_abstain / case_000004

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `diagnosis_logic_neutral`

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
        "constraint_qid": "Q21502404",
        "label": "format constraint"
      },
      {
        "constraint_qid": "Q21503250",
        "label": "subject type constraint"
      },
      {
        "constraint_qid": "Q21510851",
        "label": "allowed qualifiers constraint"
      },
      {
        "constraint_qid": "Q21502838",
        "label": "conflicts-with constraint"
      },
      {
        "constraint_qid": "Q52060874",
        "label": "single-best-value constraint"
      }
    ],
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
    ]
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

## prompt_dev_003_hybrid_json_nl_zero_shot_diagnosis_local_neutral_track_diagnosis_diagnosis_no_abstain / case_000004

- Task: `track_diagnosis`
- Representation: `hybrid_json_nl`
- Examples: `zero_shot`
- Context: `diagnosis_local_neutral`

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
        "P10006": [
          "atlanta-journal-constitution"
        ],
        "P1235": [
          "43870"
        ],
        "P12361": [
          "ajc.com"
        ],
        "P127": [
          "Q1138249"
        ],
        "P131": [
          "Q23556"
        ],
        "P13337": [
          "ajc.com"
        ],
        "P1343": [
          "Q99202247"
        ],
        "P1365": [
          "Q135213096",
          "Q112117141"
        ],
        "P154": [
          "The Atlanta Journal-Constitution logo.svg"
        ],
        "P159": [
          "Q1208436"
        ],
        "P166": [
          "Q7184075"
        ],
        "P17": [
          "Q30"
        ],
        "P1813": [
          "AJC@en"
        ],
        "P2002": [
          "ajc"
        ],
        "P2003": [
          "allthingsajc"
        ],
        "P2013": [
          "ajc"
        ],
        "P2088": [
          "cox-media-group-ecb9"
        ],
        "P2267": [
          "atlanta-journal-constitution"
        ],
        "P236": [
          "1539-7459",
          "2690-8093"
        ],
        "P2390": [
          "Atlanta_Journal-Constitution"
        ],
        "P291": [
          "Q23556"
        ],
        "P31": [
          "Q1110794"
        ],
        "P3417": [
          "The-Atlanta-Journal-Constitution"
        ],
        "P373": [
          "Atlanta Journal-Constitution"
        ],
        "P3912": [
          "Q665319"
        ],
        "P407": [
          "Q1860"
        ],
        "P463": [
          "Q117804348"
        ],
        "P4903": [
          "arts-culture/atlanta-journal-constitution"
        ],
        "P495": [
          "Q30"
        ],
        "P5454": [
          "710"
        ],
        "P571": [
          "+1868-00-00T00:00:00Z"
        ],
        "P6136": [
          "GA_AJC"
        ],
        "P646": [
          "/m/02xt43"
        ],
        "P7259": [
          "406",
          "2213"
        ],
        "P7363": [
          "1539-7459"
        ],
        "P8507": [
          "atlanta"
        ],
        "P856": [
          "https://www.ajc.com/",
          "http://infoweb.newsbank.com/?db=AJBK",
          "http://bibpurl.oclc.org/web/597",
          "http://www.accessatlanta.com/ajc/",
          "http://gateway.proquest.com/openurl?url_ver=Z39.88-2004&res_dat=xri:pqd&rft_val_fmt=info:ofi/fmt:kev:mtx:journal&rft_dat=xri:pqd:PMID=49509"
        ],
        "P8687": [
          "+1060826 1",
          "+1045696 1",
          "+1064864 1",
          "+1086625 1"
        ],
        "P9035": [
          "ajc"
        ],
        "P9852": [
          "atlanta-journal-constitution"
        ],
        "P9922": [
          "ajcnews"
        ]
      },
      "qid": "Q3092348"
    },
    "L2_labels": {
      "entities": {
        "P127": {
          "description": "owner of the subject",
          "label": "owned by"
        },
        "P131": {
          "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity",
          "label": "located in the administrative territorial entity"
        },
        "P1343": {
          "description": "work where this item is described, in statistical contexts, a methodological note describing the data",
          "label": "described by source"
        },
        "P1365": {
          "description": "person, state or item replaced. Use \"structure replaces\" (P1398) for structures. Use \"follows\" (P155) if the previous item was not replaced or predecessor and successor are identical",
          "label": "replaces"
        },
        "P1476": {
          "description": "published name of 
```
