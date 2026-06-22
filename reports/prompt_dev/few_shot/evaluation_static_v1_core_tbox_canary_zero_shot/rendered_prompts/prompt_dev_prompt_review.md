# Prompt Development Review

No LLM inference was run for this artifact.

Rendered prompts: `256`

## Example Schema Summary

| Task | Example count | Example schemas |
| --- | ---: | --- |
| `t_box_repair` | 0 | n/a |

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000001

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

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000001",
  "labels_en": {
    "property": {
      "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
      "label": "instance of"
    },
    "qid": {
      "description": null,
      "label": "1999 Prague Marathon"
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
   
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000001

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

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000001",
  "labels_en": {
    "property": {
      "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
      "label": "instance of"
    },
    "qid": {
      "description": null,
      "label": "1999 Prague Marathon"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "label": "1999 Prague Marathon",
      "qid": "Q2094536"
    },
    "L2_labels": {
      "entities": {
        "P31": {
          "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain",
          "label": "instance of"
        },
        "Q2094536": {
          "description": null,
          "lab
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000002

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

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000002",
  "labels_en": {
    "property": {
      "description": "country of origin of this item (creative work, food, phrase, product, etc.)",
      "label": "country of origin"
    },
    "qid": {
      "description": "novel by J. Gregory Keyes",
      "label": "Babylon 5: Final Reckoning – The Fate of Bester"
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
        "constraint_qid": "Q21510865",
        "label": "value-type constraint"
      },
      {
        "constraint_qid": "Q21502838",
        "label": "conflicts-with constraint"
      },
      {
        "constraint_qid":
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000002

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

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000002",
  "labels_en": {
    "property": {
      "description": "country of origin of this item (creative work, food, phrase, product, etc.)",
      "label": "country of origin"
    },
    "qid": {
      "description": "novel by J. Gregory Keyes",
      "label": "Babylon 5: Final Reckoning – The Fate of Bester"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "novel by J. Gregory Keyes",
      "label": "Babylon 5: Final Reckoning – The Fate of Bester",
      "qid": "Q4332652"
    },
    "L2_labels": {
      "entities": {
        "P495": {
          "description": "country of origin of this item (creative work, food, phrase, product, etc.)",
          "label": "country of origin"
        },
        "Q21502838": {
          "description": "type of constraint for Wikidata properties: used
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000003

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

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000003",
  "labels_en": {
    "property": {
      "description": "alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface",
      "label": "writing system"
    },
    "qid": {
      "description": "male given name (حسان)",
      "label": "Ḥassān"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q21510865",
        "label": "value-type constraint"
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
        "constraint_qid": "Q53869507",
        "label": "property scope constraint"
      },
      {
        "constraint_qid": "Q52558054",
       
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000003

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

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000003",
  "labels_en": {
    "property": {
      "description": "alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface",
      "label": "writing system"
    },
    "qid": {
      "description": "male given name (حسان)",
      "label": "Ḥassān"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "male given name (حسان)",
      "label": "Ḥassān",
      "qid": "Q30314187"
    },
    "L2_labels": {
      "entities": {
        "P282": {
          "description": "alphabet, character set or other system of writing used by a language, word, or text, supported by a typeface",
          "label": "writing system"
        },
        "Q21510859": {
          "description": "type of constraint for Wikidata properties: used to specify that the valu
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000004

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

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000004",
  "labels_en": {
    "property": {
      "description": "platform for which a work was developed or released, or the specific platform version of a software product",
      "label": "platform"
    },
    "qid": {
      "description": "2000 video game",
      "label": "Dracula 2: The Last Sanctuary"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q21510865",
        "label": "value-type constraint"
      },
      {
        "constraint_qid": "Q21503250",
        "label": "subject type constraint"
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
        "constraint_qid": "Q5255
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000004

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

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000004",
  "labels_en": {
    "property": {
      "description": "platform for which a work was developed or released, or the specific platform version of a software product",
      "label": "platform"
    },
    "qid": {
      "description": "2000 video game",
      "label": "Dracula 2: The Last Sanctuary"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "description": "2000 video game",
      "label": "Dracula 2: The Last Sanctuary",
      "qid": "Q3714890"
    },
    "L2_labels": {
      "entities": {
        "P400": {
          "description": "platform for which a work was developed or released, or the specific platform version of a software product",
          "label": "platform"
        },
        "Q21502838": {
          "description": "type of constraint for Wikidata properties: used to specif
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000005

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

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000005",
  "labels_en": {
    "property": {
      "description": "identifier of a person in the Léonore database",
      "label": "Léonore ID"
    },
    "qid": {
      "description": null,
      "label": "Georges Petitmengin"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q19474404",
        "label": "single-value constraint"
      },
      {
        "constraint_qid": "Q21502410",
        "label": "distinct-values constraint"
      },
      {
        "constraint_qid": "Q21502404",
        "label": "format constraint"
      },
      {
        "constraint_qid": "Q21503247",
        "label": "item-requires-statement constraint"
      },
      {
        "constraint_qid": "Q52004125",
        "label": "allowed-entity-types constraint"
      },
      {
       
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000005

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

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000005",
  "labels_en": {
    "property": {
      "description": "identifier of a person in the Léonore database",
      "label": "Léonore ID"
    },
    "qid": {
      "description": null,
      "label": "Georges Petitmengin"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "label": "Georges Petitmengin",
      "qid": "Q112939573"
    },
    "L2_labels": {
      "entities": {
        "P640": {
          "description": "identifier of a person in the Léonore database",
          "label": "Léonore ID"
        },
        "Q108139345": {
          "description": "constraint to ensure items using a property have label in the language (Use qualifier \"Wikimedia language code\" (P424) to define language)",
          "label": "label in language constraint"
        },
        "Q112939573": {
          "descrip
```

## prompt_dev_001_hybrid_json_nl_zero_shot_logic_only_repair_proposal_oracle_no_abstain / case_000006

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

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000006",
  "labels_en": {
    "property": {
      "description": "main Wikimedia category",
      "label": "topic's main category"
    },
    "qid": {
      "description": null,
      "label": "1644 год в России"
    }
  },
  "logic_context": {
    "constraint_family_inventory": [
      {
        "constraint_qid": "Q52060874",
        "label": "single-best-value constraint"
      },
      {
        "constraint_qid": "Q21502410",
        "label": "distinct-values constraint"
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
        "constraint_qid": "Q21502838",
        "label": "conflicts-with constraint"
      },
      {
        "constraint_qid": "Q538695
```

## prompt_dev_002_hybrid_json_nl_zero_shot_local_graph_repair_proposal_oracle_no_abstain / case_000006

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

No examples are provided. Solve the task zero-shot.

Input case JSON:
{
  "id": "case_000006",
  "labels_en": {
    "property": {
      "description": "main Wikimedia category",
      "label": "topic's main category"
    },
    "qid": {
      "description": null,
      "label": "1644 год в России"
    }
  },
  "local_context": {
    "L1_ego_node": {
      "label": "1644 год в России",
      "qid": "Q19907031"
    },
    "L2_labels": {
      "entities": {
        "P910": {
          "description": "main Wikimedia category",
          "label": "topic's main category"
        },
        "Q19907031": {
          "description": null,
          "label": "1644 год в России"
        },
        "Q21502410": {
          "description": "type of constraint for Wikidata properties: used to specify that the value for this property is likely to be different from all other items",
          "label": "distinct-v
```
