# Prompt Development Review

No LLM inference was run for this artifact.

Rendered prompts: `256`

## Example Schema Summary

| Task | Example count | Example schemas |
| --- | ---: | --- |
| `t_box_repair` | 1024 | `tbox_taxonomy_patch_v1`: 1024 |

## prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain / case_000001

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

## prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain / case_000001

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

## prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain / case_000004

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

## prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain / case_000005

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

## prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain / case_000005

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

## prompt_dev_001_hybrid_json_nl_static_diverse_kshot_logic_only_repair_proposal_oracle_no_abstain / case_000006

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

## prompt_dev_002_hybrid_json_nl_static_diverse_kshot_local_graph_repair_proposal_oracle_no_abstain / case_000006

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
