from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

PROMPT_DEV_VERSION = "prompt_dev_v1"

REPRESENTATIONS = (
    "hybrid_json_nl",
    "pure_nl",
    "compact_table",
    "turtle",
)

TASKS = (
    "track_diagnosis",
    "a_box_repair",
    "t_box_repair",
)


@dataclass(frozen=True)
class RenderedPrompt:
    prompt_name: str
    system_prompt: str
    user_prompt: str
    response_format: dict[str, Any]


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _flatten_pairs(value: Any, *, prefix: str = "case", limit: int = 140) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    def visit(item: Any, path: str) -> None:
        if len(pairs) >= limit:
            return
        if isinstance(item, dict):
            for key in sorted(item):
                visit(item[key], f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item[:20]):
                visit(child, f"{path}[{index}]")
            if len(item) > 20:
                pairs.append((f"{path}.truncated_count", str(len(item) - 20)))
        else:
            pairs.append((path, json.dumps(item, ensure_ascii=False)))

    visit(value, prefix)
    return pairs


def _pure_nl_payload(payload: dict[str, Any]) -> str:
    labels = payload.get("labels_en") if isinstance(payload.get("labels_en"), dict) else {}
    violation = payload.get("violation_context") if isinstance(payload.get("violation_context"), dict) else {}
    qid_label = labels.get("qid") or labels.get("qid_label_en") or "no label"
    property_label = labels.get("property") or labels.get("property_label_en") or "no label"
    lines = [
        f"Case id: {payload.get('id')}",
        f"Focus entity: {payload.get('qid')} ({qid_label})",
        f"Target property: {payload.get('property')} ({property_label})",
        "Visible violation report context:",
        _json_block(violation),
    ]
    if "logic_context" in payload:
        lines.extend(["Visible rule and constraint context:", _json_block(payload["logic_context"])])
    if "local_context" in payload:
        lines.extend(["Visible local graph and constraint context:", _json_block(payload["local_context"])])
    return "\n".join(lines)


def _compact_table_payload(payload: dict[str, Any]) -> str:
    pairs = _flatten_pairs(payload)
    width = max((len(key) for key, _ in pairs), default=3)
    return "\n".join(f"{key.ljust(width)} | {value}" for key, value in pairs)


def _turtle_value(value: str) -> str:
    if value.startswith(("Q", "P")):
        return f"wd:{value}"
    return json.dumps(value, ensure_ascii=False)


def _turtle_payload(payload: dict[str, Any]) -> str:
    subject = payload.get("qid") if isinstance(payload.get("qid"), str) else "CASE"
    case_id = payload.get("id")
    target_pid = payload.get("property")
    triples = [
        "@prefix wd: <http://www.wikidata.org/entity/> .",
        "@prefix wdt: <http://www.wikidata.org/prop/direct/> .",
        "@prefix kb: <https://kg-benchmark.local/> .",
        "",
        f"wd:{subject} kb:caseId {json.dumps(case_id, ensure_ascii=False)} .",
        f"wd:{subject} kb:targetProperty {_turtle_value(str(target_pid))} .",
    ]
    for key, value in _flatten_pairs(payload.get("violation_context", {}), prefix="violation", limit=40):
        predicate = key.replace(".", "_").replace("[", "_").replace("]", "")
        triples.append(f"wd:{subject} kb:{predicate} {_turtle_value(value)} .")
    if "logic_context" in payload:
        triples.append("")
        triples.append("# Pruned constraint context")
        for key, value in _flatten_pairs(payload["logic_context"], prefix="logic", limit=70):
            triples.append(
                f"wd:{subject} kb:{key.replace('.', '_').replace('[', '_').replace(']', '')} {_turtle_value(value)} ."
            )
    if "local_context" in payload:
        triples.append("")
        triples.append("# Pruned local graph context")
        for key, value in _flatten_pairs(payload["local_context"], prefix="local", limit=90):
            triples.append(
                f"wd:{subject} kb:{key.replace('.', '_').replace('[', '_').replace(']', '')} {_turtle_value(value)} ."
            )
    return "\n".join(triples)


def render_payload(payload: dict[str, Any], representation: str) -> str:
    if representation == "hybrid_json_nl":
        return _json_block(payload)
    if representation == "pure_nl":
        return _pure_nl_payload(payload)
    if representation == "compact_table":
        return _compact_table_payload(payload)
    if representation == "turtle":
        return _turtle_payload(payload)
    raise ValueError(f"Unsupported Phase F representation: {representation}")


def _examples_block(examples: list[dict[str, Any]], representation: str) -> str:
    if not examples:
        return "No examples are provided. Solve the task zero-shot."
    blocks: list[str] = []
    for index, example in enumerate(examples, start=1):
        input_payload = example.get("input_payload") if isinstance(example.get("input_payload"), dict) else {}
        output_payload = example.get("output_payload") if isinstance(example.get("output_payload"), dict) else {}
        blocks.append(
            "\n".join(
                [
                    f"Example {index} input:",
                    render_payload(input_payload, representation),
                    f"Example {index} expected JSON output:",
                    _json_block(output_payload),
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_input_section(payload: dict[str, Any], representation: str) -> str:
    labels = {
        "hybrid_json_nl": "Input case JSON",
        "pure_nl": "Input case description",
        "compact_table": "Input case compact table",
        "turtle": "Input case Turtle-like triples",
    }
    return f"{labels[representation]}:\n{render_payload(payload, representation)}"


TRACK_DIAGNOSIS_CONTRACT = """Return exactly one JSON object:
{
  "case_id": "<copy input id exactly>",
  "predicted_track": "A_BOX" | "T_BOX" | "AMBIGUOUS",
  "confidence": "low" | "medium" | "high" | "0.0-1.0 as a string",
  "rationale": "<short evidence-based explanation>"
}
"""

A_BOX_REPAIR_CONTRACT = """Return exactly one JSON object:
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
  ],
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "HISTORY" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}
"""

T_BOX_REPAIR_CONTRACT = """Return exactly one JSON object:
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
        "snaktype": "VALUE" | "SOMEVALUE" | "NOVALUE",
        "rank": "normal" | "preferred" | "deprecated",
        "qualifiers": [{"property_id": "P...", "values": ["Q..." | "literal"]}]
      }
    ]
  },
  "rationale": "<short evidence-based explanation>",
  "provenance": [{"kind": "KG" | "HISTORY" | "OTHER", "node_id": "Q...", "snippet": "<visible evidence>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}
"""

ABSTENTION_CONTRACT = """If the visible evidence is insufficient, return exactly:
{
  "case_id": "<copy input id exactly>",
  "abstain": true,
  "reason": "insufficient_visible_evidence" | "ambiguous_repair_locus" | "unsupported_schema_reform",
  "rationale": "<short explanation>",
  "provenance": [{"kind": "OTHER", "snippet": "<what evidence was missing>"}],
  "uncertainty": {"confidence": 0.0, "notes": "<short uncertainty note>"}
}
Only abstain when a concrete repair would require evidence not present in the prompt.
"""


def render_prompt_dev_prompt(
    *,
    task: str,
    representation: str,
    case_payload: dict[str, Any],
    examples: list[dict[str, Any]] | None = None,
    include_abstention: bool = False,
) -> RenderedPrompt:
    if task not in TASKS:
        raise ValueError(f"Unsupported Phase F task: {task}")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unsupported Phase F representation: {representation}")

    examples = examples or []
    prompt_name = f"{PROMPT_DEV_VERSION}_{task}_{representation}"
    system_prompt = (
        "You are evaluating knowledge-graph repair capability under a controlled benchmark prompt. "
        "Use only the evidence in the prompt. Return valid JSON only; no markdown and no code fences."
    )
    if task == "track_diagnosis":
        task_instruction = (
            "Decide whether the visible historical repair case should be treated as A_BOX, T_BOX, or AMBIGUOUS. "
            "A_BOX edits the focus entity claim. T_BOX edits the property constraint or schema rule. "
            "AMBIGUOUS means the visible evidence does not support choosing safely."
        )
        contract = TRACK_DIAGNOSIS_CONTRACT
    elif task == "a_box_repair":
        task_instruction = (
            "Propose an executable A-box repair transaction for the focus entity and target property. "
            "Preserve useful values when the evidence supports them; do not over-delete just to satisfy a rule."
        )
        contract = A_BOX_REPAIR_CONTRACT
    else:
        task_instruction = (
            "Propose an executable T-box schema reform for the focus property. "
            "Use constraint-family QIDs from the supplied context as constraint_type_qid values. "
            "Do not copy ordinary entity/type QIDs into constraint-family fields."
        )
        contract = T_BOX_REPAIR_CONTRACT

    if include_abstention and task != "track_diagnosis":
        contract = f"{contract}\nOptional abstention contract:\n{ABSTENTION_CONTRACT}"

    user_prompt = "\n\n".join(
        [
            f"Prompt version: {PROMPT_DEV_VERSION}",
            f"Representation: {representation}",
            f"Task: {task}",
            task_instruction,
            "Output contract:",
            contract.strip(),
            "Few-shot examples:",
            _examples_block(examples, representation),
            _format_input_section(case_payload, representation),
        ]
    )
    return RenderedPrompt(
        prompt_name=prompt_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format={"type": "json_object"},
    )
