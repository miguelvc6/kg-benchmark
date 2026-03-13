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
            "You are producing a zero-shot A-box repair proposal for a benchmark case. "
            "Return JSON only."
        ),
        user_prompt_template="""Return exactly one JSON object that follows this contract.

Required shape:
{
  "case_id": "<copy input id exactly>",
  "target": {
    "qid": "Q...",
    "pid": "P..."
  },
  "ops": [
    {
      "op": "SET" | "ADD" | "REMOVE" | "DELETE_ALL",
      "pid": "P...",
      "value": "Q..." | "<literal>" | 123,
      "rank": "normal" | "preferred" | "deprecated"
    }
  ]
}

Optional top-level fields:
- "rationale"
- "provenance"
- "metadata"

Rules:
- Use only the contract fields above. Do not wrap the answer in keys like "proposal_id", "repair_id", "summary", "actions", "patch", "current_state", or "proposal".
- Copy "case_id" exactly from the input case.
- Copy the focus entity/property into target.qid and target.pid.
- If the final property value should be empty, use REMOVE or DELETE_ALL.
- If the final property value should be a single value, prefer SET.
- Output valid JSON only. No markdown. No code fences.

Canonical example:
{
  "case_id": "repair_case",
  "target": {
    "qid": "Q1",
    "pid": "P31"
  },
  "ops": [
    {
      "op": "SET",
      "pid": "P31",
      "value": "Q5"
    }
  ],
  "rationale": "Replace the invalid type with the historically repaired value."
}

Input case:
{payload_json}
""",
    ),
    "reasoning_floor_t_box_zero_shot": PromptTemplate(
        name="reasoning_floor_t_box_zero_shot",
        description="Zero-shot proposal prompt for T-box reform cases in the reasoning floor.",
        system_prompt=(
            "You are producing a zero-shot T-box reform proposal for a benchmark case. "
            "Return JSON only."
        ),
        user_prompt_template="""Return exactly one JSON object that follows this contract.

Required shape:
{
  "case_id": "<copy input id exactly>",
  "target": {
    "pid": "P...",
    "constraint_type_qid": "Q..."
  },
  "proposal": {
    "action": "RELAXATION_RANGE_WIDENED" | "RESTRICTION_RANGE_NARROWED" | "RELAXATION_SET_EXPANSION" | "RESTRICTION_SET_CONTRACTION" | "SCHEMA_UPDATE" | "COINCIDENTAL_SCHEMA_CHANGE",
    "signature_after": [
      {
        "constraint_qid": "Q...",
        "snaktype": "VALUE",
        "rank": "normal",
        "qualifiers": [
          {
            "property_id": "P2305",
            "values": ["Q5", "Q43229"]
          }
        ]
      }
    ]
  }
}

Optional top-level fields:
- "rationale"
- "provenance"
- "metadata"

Rules:
- Use only the contract fields above. Do not wrap the answer in keys like "proposal_id", "summary", "changes", "recommended_changes", or "proposed_changes".
- Copy "case_id" exactly from the input case.
- Copy the focus property into target.pid.
- Use the historically relevant constraint type in target.constraint_type_qid.
- proposal.action must be one of the listed enum values.
- Output valid JSON only. No markdown. No code fences.

Canonical example:
{
  "case_id": "reform_case",
  "target": {
    "pid": "P31",
    "constraint_type_qid": "Q21510859"
  },
  "proposal": {
    "action": "RELAXATION_SET_EXPANSION",
    "signature_after": [
      {
        "constraint_qid": "Q21510859",
        "snaktype": "VALUE",
        "rank": "normal",
        "qualifiers": [
          {
            "property_id": "P2305",
            "values": ["Q5", "Q43229"]
          }
        ]
      }
    ]
  },
  "rationale": "Expand the allowed set to include the historically repaired classes."
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
