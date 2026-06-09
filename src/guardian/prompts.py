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
[{"kind":"KG","node_id":"Q5","snippet":"historical target"}]
"uncertainty" must be an object such as:
{"confidence": 0.15, "notes": "Low confidence because only local evidence is visible."}

Rules:
- Copy "case_id" exactly from the input case.
- Copy the focus entity/property into target.qid and target.pid.
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

Example:
{
  "case_id": "case_000001",
  "target": {"qid": "Q1", "pid": "P31"},
  "ops": [
    {"op": "SET", "pid": "P31", "value": "Q5"}
  ],
  "rationale": "Replace the invalid type with the historical repaired value.",
  "provenance": [{"kind": "KG", "node_id": "Q5", "snippet": "visible target evidence"}],
  "uncertainty": {"confidence": 0.15, "notes": "The local constraint context is sparse."}
}

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
{"confidence": 0.20, "notes": "The exact historical signature may have alternative equivalent forms."}

Rules:
- Copy "case_id" exactly from the input case.
- Copy the focus property into target.pid.
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

Template example 1 (placeholders, not literal ids):
{
  "case_id": "case_000001",
  "target": {"pid": "P_TARGET", "constraint_type_qid": "<constraint family qid from input>"},
  "proposal": {
    "action": "RELAXATION_SET_EXPANSION",
    "signature_after": [
      {
        "constraint_qid": "<same constraint family qid>",
        "snaktype": "VALUE",
        "rank": "normal",
        "qualifiers": [
          {
            "property_id": "P2305",
            "values": ["<item qid a>", "<item qid b>"]
          }
        ]
      }
    ]
  },
  "rationale": "Expand the allowed set by editing the constraint family, not by changing the violating claim directly.",
  "provenance": [
    {"kind": "KG", "node_id": "<constraint family qid from input>", "snippet": "visible constraint family"}
  ],
  "uncertainty": {"confidence": 0.20, "notes": "The exact historical signature may have equivalent reorderings."}
}

Template example 2 (placeholders, not literal ids):
{
  "case_id": "case_000001",
  "target": {"pid": "P_TARGET", "constraint_type_qid": "<constraint family qid from input>"},
  "proposal": {
    "action": "RESTRICTION_SET_CONTRACTION",
    "signature_after": [
      {
        "constraint_qid": "<same constraint family qid>",
        "snaktype": "VALUE",
        "rank": "normal",
        "qualifiers": [
          {
            "property_id": "P2305",
            "values": ["<allowed type qid>"]
          }
        ]
      }
    ]
  },
  "rationale": "Narrow the allowed types encoded by the constraint itself.",
  "provenance": [
    {"kind": "KG", "node_id": "<constraint family qid from input>", "snippet": "constraint family in context"}
  ],
  "uncertainty": {"confidence": 0.35, "notes": "Another schema-level reform family may also be plausible."}
}

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
- A_BOX means the fix should change the claim on the focus entity.
- T_BOX means the fix should reform the property constraint or schema rule itself.
- AMBIGUOUS means the visible evidence is insufficient to choose safely between claim repair and schema reform.
- Use only visible prompt evidence.
- Do not infer hidden benchmark classes, subtypes, or historical labels.
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
