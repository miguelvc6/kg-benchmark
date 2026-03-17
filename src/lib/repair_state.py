from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def comparable_atom(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("qid", "id", "raw", "value"):
            if key in value:
                return comparable_atom(value[key])
    return str(value)


def normalize_value_list(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [comparable_atom(item) for item in items if item not in (None, "MISSING")]


def pre_repair_target_raw_value(record: dict[str, Any]) -> tuple[Any, str]:
    repair_target = record.get("repair_target", {})
    if isinstance(repair_target, dict) and "old_value" in repair_target:
        return repair_target.get("old_value"), "repair_target.old_value"
    violation_context = record.get("violation_context", {})
    if isinstance(violation_context, dict) and "value" in violation_context:
        return violation_context.get("value"), "violation_context.value"
    return None, "missing"


@dataclass(frozen=True)
class PreRepairTargetState:
    values: list[str]
    labels_en: list[Any]
    descriptions_en: list[Any]
    source: str

    def entity_label_entries(self) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}
        for index, value in enumerate(self.values):
            if not isinstance(value, str) or not value.startswith(("Q", "P")):
                continue
            label = self.labels_en[index] if index < len(self.labels_en) else None
            description = self.descriptions_en[index] if index < len(self.descriptions_en) else None
            payload = {
                key: item
                for key, item in {
                    "label": label,
                    "description": description,
                }.items()
                if item is not None
            }
            if payload:
                entries[value] = payload
        return entries


def pre_repair_target_state(record: dict[str, Any]) -> PreRepairTargetState:
    raw_value, source = pre_repair_target_raw_value(record)
    values = normalize_value_list(raw_value)

    labels_en: list[Any] = []
    descriptions_en: list[Any] = []
    repair_target = record.get("repair_target", {})
    violation_context = record.get("violation_context", {})

    if source == "repair_target.old_value" and isinstance(repair_target, dict):
        raw_labels = repair_target.get("old_value_labels_en")
        raw_descriptions = repair_target.get("old_value_descriptions_en")
        labels_en = raw_labels if isinstance(raw_labels, list) else []
        descriptions_en = raw_descriptions if isinstance(raw_descriptions, list) else []
    elif source == "violation_context.value" and isinstance(violation_context, dict):
        raw_labels = violation_context.get("value_labels_en")
        raw_descriptions = violation_context.get("value_descriptions_en")
        labels_en = raw_labels if isinstance(raw_labels, list) else []
        descriptions_en = raw_descriptions if isinstance(raw_descriptions, list) else []

    return PreRepairTargetState(
        values=values,
        labels_en=labels_en,
        descriptions_en=descriptions_en,
        source=source,
    )


def reconstruct_properties_with_pre_repair_target(
    record: dict[str, Any],
    properties: Any,
) -> dict[str, list[str]]:
    reconstructed = dict(properties) if isinstance(properties, dict) else {}
    target_pid = record.get("property")
    state = pre_repair_target_state(record)
    if isinstance(target_pid, str):
        if state.values:
            reconstructed[target_pid] = list(state.values)
        else:
            reconstructed.pop(target_pid, None)
    return reconstructed
