from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    description: str
    system_prompt: str
    user_prompt_template: str
    response_format: dict[str, Any] = field(default_factory=lambda: {"type": "json_object"})

    def render(self, payload: Any) -> str:
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        return self.user_prompt_template.replace("{payload_json}", payload_json)

    def response_format_copy(self) -> dict[str, Any]:
        return dict(self.response_format)


def _load_paper_prompt(filename: str) -> tuple[str, str]:
    path = Path(__file__).resolve().parents[2] / "paper" / "prompts" / filename
    text = path.read_text(encoding="utf-8")
    system_marker = "[system]\n"
    user_marker = "\n[user]\n"
    if not text.startswith(system_marker) or user_marker not in text:
        raise ValueError(f"Malformed paper prompt source: {path}")
    system, user = text[len(system_marker) :].split(user_marker, 1)
    if "{payload_json}" not in user:
        raise ValueError(f"Paper prompt has no payload placeholder: {path}")
    return system.strip(), user.rstrip()


def _template(name: str, description: str, filename: str) -> PromptTemplate:
    system, user = _load_paper_prompt(filename)
    return PromptTemplate(
        name=name,
        description=description,
        system_prompt=system,
        user_prompt_template=user,
    )


PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "reasoning_floor_a_box_zero_shot": _template(
        "reasoning_floor_a_box_zero_shot",
        "Paper A-box repair prompt.",
        "abox-repair.txt",
    ),
    "reasoning_floor_t_box_taxonomy_patch_zero_shot": _template(
        "reasoning_floor_t_box_taxonomy_patch_zero_shot",
        "Paper T-box taxonomy-patch prompt.",
        "tbox-taxonomy-patch.txt",
    ),
    "reasoning_floor_track_diagnosis_zero_shot": _template(
        "reasoning_floor_track_diagnosis_zero_shot",
        "Paper repair-locus diagnosis prompt.",
        "track-diagnosis.txt",
    ),
}


def get_prompt_template(name: str) -> PromptTemplate:
    try:
        return PROMPT_TEMPLATES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(f"Unknown prompt template {name!r}. Available templates: {available}") from exc
