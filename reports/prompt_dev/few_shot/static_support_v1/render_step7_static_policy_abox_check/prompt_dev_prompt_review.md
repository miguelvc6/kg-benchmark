# Prompt Development Review

No LLM inference was run for this artifact.

Rendered prompts: `2`

## prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_track_diagnosis_diagnosis_no_abstain / case_000001

- Task: `track_diagnosis`
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

Example 1 input:
{
  "id": "example_d_000001",
  "labels_en": {
    "property": {
      "description": "numerical identifier for a plant name in the International Plant Names Index",
      "label": "IPNI plant ID"
    },
    "qid": {
      "description": "species of plant",
      "label": "Orostachys japonica"
    }
  },
  "logic_context": {
    "constraints": [
      {
        "constraint_type": {
          "label": "single-value constraint",
          "qid": "Q19474404"
        },
        "qualifiers": [
          {
            "property_id": "P2303",
            "property_label": "exception to constraint",
            "values": [
              {
                "label": "Scutellaria versicolor",
                "qid": "Q123983269",
                "raw": "Q123983269"
              }
            ]
          }
        ],
        "rule_summary": "exception to constraint (P2303): Scutellaria versicolor (Q123983269)"
      },
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
    "property_id": "P961"
  },
  "property": "P961",
  "qid": "Q15484095",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P961",
    "report_violation_type": "Single value",
    "report_violation_type_normalized": "Single value",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Single value",
    "value": [
      "274621-1",
      "274615-1"
    ]
  }
}
Example 1 expected JSON output:
{
  "case_id": "example_d_000001",
  "confidence": "high",
  "predicted_track": "A_BOX",
  "rationale": "The demonstration answer uses this dev example's historical repair locus."
}

Example 2 input:
{
  "id": "example_d_000002",
  "labels_en": {
    "property": {
      "description": "identifier assigned by the National Library of Latvia",
      "label": "National Library of Latvia ID"
    },
    "qid": {
      "description": "American novelist and poet (1943–2022)",
      "label": "Peter Straub"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q21502404",
        "label": "format constraint"
      },
      {
        "constraint_qid": "Q21502410",
        "label": "distinct-values constraint"
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
        "constraint_qid": "Q108139345",
        "label": "label in language constraint"
      }
    ],
    "violation_context": {
      "report_page_title": "Wikidata:Database reports/Constraint violations/P1368",
      "report_violation_type": "Label in lv language",
      "report_violation_type_normalized": "Label in lv language",
      "report_violation_type_qids": [],
      "report_violation_type_raw": "Label in lv language"
    }
  },
  "property": "P1368",
  "qid": "Q364189",
  "violation_context": {
    "report_page_title": "Wikidata:Database reports/Constraint violations/P1368",
    "report_violation_type": "Label in lv language",
    "report_violation_type_normalized": "Label in lv language",
    "report_violation_type_qids": [],
    "report_violation_type_raw": "Label in lv languag
```

## prompt_dev_002_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain / case_000001

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
