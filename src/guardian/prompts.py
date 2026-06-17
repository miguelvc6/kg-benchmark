from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    description: str
    system_prompt: str
    user_prompt_template: str = "{payload_json}"
    response_format: dict[str, Any] = field(default_factory=lambda: {"type": "json_object"})

    def render(self, payload: Any) -> str:
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        return self.user_prompt_template.replace("{payload_json}", payload_json)

    def response_format_copy(self) -> dict[str, Any]:
        return dict(self.response_format)


PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "reasoning_floor_a_box_zero_shot": PromptTemplate(
        name="reasoning_floor_a_box_zero_shot",
        description="Zero-shot proposal prompt for A-box repair cases in the reasoning floor.",
        system_prompt=(
            "Produce one A-box repair proposal in the benchmark's canonical JSON shape. Return JSON only. "
            "Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON."
        ),
        user_prompt_template="""Return exactly one JSON object with this shape:
{
  "case_id": "<copy input id exactly>",
  "target": {"qid": "Q...", "pid": "P..."},
  "ops": [
    {
      "op": "SET" | "ADD" | "REMOVE" | "DELETE_ALL",
      "pid": "P...",
      "value": "Q..." | "<literal>" | 123,
      "rank": "normal" | "preferred" | "deprecated"
    }
  ]
}

Also include: "rationale", "provenance", and "uncertainty".
"provenance" must be an array of objects such as:
[{"kind":"KG","node_id":"<visible node id>","snippet":"<visible evidence>"}]
"uncertainty" must be an object such as:
{"confidence": 0.0, "notes": "<short uncertainty note>"}

Rules:
- Copy "case_id" exactly from the input case.
- Copy the focus entity/property into target.qid and target.pid.
- A_BOX means this proposal edits claim values on the focus entity, not the property constraint/schema rule.
- Use only the contract fields above. Do not wrap the answer in keys like "proposal_id", "repair_id",
  "summary", "actions", "patch", "current_state", or "proposal".
- target.qid is the focus entity identifier from the input.
- target.pid is the target property identifier from the input.
- ops is the ordered set of claim edits to the target property on the focus entity.
- SET replaces the target property's value set with the supplied value.
- ADD adds the supplied value to the target property.
- REMOVE removes the supplied value from the target property.
- DELETE_ALL removes all values for the target property.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
- Use only visible prompt evidence.
- Replacement claim values must be ordinary claim values, not constraint-family identifiers, unless the prompt visibly
  presents that identifier as the claim value itself.
- Constraint-family QIDs, report-type QIDs, allowed-type QIDs, and ordinary entity/type QIDs have different roles; keep
  those roles distinct.
- Do not use hidden benchmark classes, subtypes, or historical labels.
- Output valid JSON only. No markdown. No code fences.

Input case:
{payload_json}
""",
    ),
    "reasoning_floor_t_box_zero_shot": PromptTemplate(
        name="reasoning_floor_t_box_zero_shot",
        description="Zero-shot proposal prompt for T-box reform cases in the reasoning floor.",
        system_prompt=(
            "Produce one T-box reform proposal in the benchmark's canonical JSON shape. Return JSON only. "
            "Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON."
        ),
        user_prompt_template="""Return exactly one JSON object with this shape:
{
  "case_id": "<copy input id exactly>",
  "target": {
    "pid": "P...",
    "constraint_type_qid": "Q... or null only for UNCLEAR_SCHEMA_EVIDENCE when no visible constraint family is available"
  },
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
        "snaktype": "VALUE",
        "rank": "normal",
        "qualifiers": [
          {
            "property_id": "P2305",
            "values": ["Q...", "Q..."]
          }
        ]
      }
    ]
  }
}

Also include: "rationale", "provenance", and "uncertainty".
"provenance" must be an array of objects such as:
[{"kind":"KG","node_id":"<constraint family qid from input>","snippet":"current constraint family"}]
"uncertainty" must be an object such as:
{"confidence": 0.0, "notes": "<short uncertainty note>"}

Rules:
- Copy "case_id" exactly from the input case.
- Copy the focus property into target.pid.
- T_BOX means this proposal edits a property constraint or schema rule, not the violating claim on the focus entity.
- target.constraint_type_qid and every proposal.signature_after[*].constraint_qid must be constraint-family QIDs
  from the supplied constraint context, not ordinary entity/type/item QIDs.
- Keep the target constraint family separate from qualifier values. Qualifier values are the item/type/range/pattern
  values inside the constraint, not the constraint family itself.
- Use only the contract fields above. Do not wrap the answer in keys like "proposal_id", "summary", "changes",
  "recommended_changes", or "proposed_changes".
- proposal.action must be one of the listed enum values.
- target.pid is the focus property identifier from the input.
- target.constraint_type_qid is the constraint-family identifier being edited.
- proposal.signature_after is the proposed post-repair constraint signature when visible evidence supports specifying
  one.
- signature_after[*].constraint_qid is a constraint-family QID, not an ordinary item/type value.
- signature_after[*].qualifiers[*].values are qualifier values inside the constraint signature.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
- Use only visible prompt evidence.
- Keep constraint-family QIDs separate from ordinary entity/type QIDs and qualifier values.
- Do not copy report_violation_type_qids into target.constraint_type_qid or signature_after unless the prompt visibly
  presents the same QID in that schema role.
- Do not use hidden benchmark classes, subtypes, or historical labels.
- Output valid JSON only. No markdown. No code fences.

Input case:
{payload_json}
""",
    ),
    "reasoning_floor_t_box_taxonomy_patch_zero_shot": PromptTemplate(
        name="reasoning_floor_t_box_taxonomy_patch_zero_shot",
        description="Zero-shot taxonomy-patch prompt for T-box repair cases in the reasoning floor.",
        system_prompt=(
            "Produce one T-box taxonomy patch in the benchmark's canonical JSON shape. Return JSON only. "
            "Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON."
        ),
        user_prompt_template="""Return exactly one JSON object with this shape:
{
  "case_id": "<copy input id exactly>",
  "schema_decision": "CAUSAL_SCHEMA_REPAIR" | "NO_CAUSAL_SCHEMA_REPAIR" | "UNCLEAR_SCHEMA_EVIDENCE",
  "target": {"pid": "P...", "constraint_type_qid": "Q..."},
  "repairs": [
    {
      "repair_op": "CONSTRAINT_REMOVE" | "CONSTRAINT_DEPRECATE" | "CONSTRAINT_ADD"
        | "CONSTRAINT_TYPE_REPLACE" | "CONSTRAINT_QUALIFIER_ADD" | "CONSTRAINT_QUALIFIER_REMOVE"
        | "CONSTRAINT_QUALIFIER_REPLACE" | "CLASS_HIERARCHY_ADD" | "EXCEPTION_ADD" | "OTHER_TBOX_UPDATE",
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

Rules:
- Copy "case_id" exactly from the input case.
- schema_decision must be one of the listed enum values.
- target.pid is the focus property identifier from the input.
- target.constraint_type_qid is the visible constraint-family QID being considered; use null only when
  schema_decision is UNCLEAR_SCHEMA_EVIDENCE and no visible constraint family is available.
- repairs[*].constraint_type_qid values are constraint-family QIDs.
- repairs may be empty only for NO_CAUSAL_SCHEMA_REPAIR or UNCLEAR_SCHEMA_EVIDENCE.
- Use NO_CAUSAL_SCHEMA_REPAIR only when a visible constraint family can be named but visible evidence does not support a
  causal schema edit for it. If no constraint family can be named from visible evidence, use UNCLEAR_SCHEMA_EVIDENCE.
- taxonomy_code must match repair_op.
- qualifier_property_id is the edited qualifier property or null.
- added_values and removed_values are concrete changed values only when visible; otherwise use empty lists.
- old_value and new_value summarize a replacement when visible; otherwise use null.
- rank_after and snaktype_after describe visible post-update rank or snaktype, or null.
- evidence_level is FAMILY_ONLY, OPERATION_VISIBLE, or VALUE_DELTA_VISIBLE.
- provenance cites visible prompt evidence used for the proposal.
- uncertainty records confidence and important visible-evidence limits.
- Use only visible prompt evidence.
- Keep constraint-family QIDs separate from ordinary item/type values.
- Do not use an empty string or placeholder for target.constraint_type_qid.
- Do not copy report-violation QIDs into value deltas unless visibly presented as schema values.
- Do not construct a full post-repair signature_after.
- Do not use hidden benchmark metadata, hidden classes, hidden subtypes, historical labels, or raw benchmark prefixes.
- Output valid JSON only. No markdown. No code fences.

Input case:
{payload_json}
""",
    ),
    "reasoning_floor_track_diagnosis_zero_shot": PromptTemplate(
        name="reasoning_floor_track_diagnosis_zero_shot",
        description="Zero-shot diagnosis prompt that classifies a case into A_BOX, T_BOX, or AMBIGUOUS.",
        system_prompt=(
            "You are performing a zero-shot benchmark diagnosis task. "
            "Decide whether the historical case should be treated as A_BOX, T_BOX, or AMBIGUOUS. "
            "Return JSON only with case_id, predicted_track, optional confidence, and optional rationale. "
            "Do not include <think> tags, chain-of-thought, markdown, or text before/after JSON."
        ),
        user_prompt_template="""Return exactly one JSON object that follows this contract:
{
  "case_id": "<copy input id exactly>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<optional short explanation>"
}

Rules:
- Copy "case_id" exactly from the input case.
- predicted_track must be exactly one of A_BOX, T_BOX, or AMBIGUOUS.
- A_BOX means the most likely repair locus is the focus entity's claim for the target property.
- T_BOX means the most likely repair locus is the target property's constraint or schema rule.
- AMBIGUOUS means the visible evidence is insufficient to choose between those repair loci.
- Decide where the repair should be applied, not which vocabulary appears in the violation report.
- A constraint report can be resolved by an A_BOX claim edit or a T_BOX schema edit; choose the locus supported by the
  visible evidence.
- Choose AMBIGUOUS only when the visible evidence leaves both repair loci plausible or neither repair locus supported.
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
- Do not use case-id prefixes, raw benchmark identifiers, or provenance outside the prompt.
- If you include confidence, prefer a string such as "high" or "0.90".
- Output valid JSON only. No markdown. No code fences.

Input case:
{payload_json}
""",
    ),
}


def get_prompt_template(name: str) -> PromptTemplate:
    try:
        return PROMPT_TEMPLATES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(f"Unknown prompt template {name!r}. Available templates: {available}") from exc
