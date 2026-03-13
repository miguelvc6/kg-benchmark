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
        return self.user_prompt_template.format(payload_json=payload_json)

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
    ),
    "reasoning_floor_t_box_zero_shot": PromptTemplate(
        name="reasoning_floor_t_box_zero_shot",
        description="Zero-shot proposal prompt for T-box reform cases in the reasoning floor.",
        system_prompt=(
            "You are producing a zero-shot T-box reform proposal for a benchmark case. "
            "Return JSON only."
        ),
    ),
    "reasoning_floor_track_diagnosis_zero_shot": PromptTemplate(
        name="reasoning_floor_track_diagnosis_zero_shot",
        description="Zero-shot diagnosis prompt that classifies a case into A_BOX, T_BOX, or AMBIGUOUS.",
        system_prompt=(
            "You are performing a zero-shot benchmark diagnosis task. "
            "Decide whether the historical case should be treated as A_BOX, T_BOX, or AMBIGUOUS. "
            "Return JSON only with case_id, predicted_track, optional confidence, and optional rationale. "
        ),
    ),
}


def get_prompt_template(name: str) -> PromptTemplate:
    try:
        return PROMPT_TEMPLATES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(f"Unknown prompt template {name!r}. Available templates: {available}") from exc
