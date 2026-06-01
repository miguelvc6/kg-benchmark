"""Helpers for deriving normalized A-box value-delta summaries."""

from __future__ import annotations

from collections import Counter
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
        for key in ("qid", "id", "raw", "value", "time", "amount", "text"):
            if key in value:
                return comparable_atom(value[key])
    return str(value)


def normalize_value_list(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [comparable_atom(item) for item in items if item not in (None, "MISSING")]


def raw_value_list(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [comparable_atom(item) for item in items if item is not None]


@dataclass(frozen=True)
class ValueChangeSummary:
    old_values_raw: list[str]
    new_values_raw: list[str]
    old_values: list[str]
    new_values: list[str]
    old_unique: list[str]
    new_unique: list[str]
    old_counts: dict[str, int]
    new_counts: dict[str, int]
    retained_unique_values: list[str]
    added_unique_values: list[str]
    removed_unique_values: list[str]
    normalized_unique_values_unchanged: bool
    exact_value_lists_unchanged: bool
    value_multiplicity_changes: dict[str, dict[str, int]]
    semantic_action: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "old_values_raw": self.old_values_raw,
            "new_values_raw": self.new_values_raw,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "old_unique": self.old_unique,
            "new_unique": self.new_unique,
            "old_counts": self.old_counts,
            "new_counts": self.new_counts,
            "retained_unique_values": self.retained_unique_values,
            "added_unique_values": self.added_unique_values,
            "removed_unique_values": self.removed_unique_values,
            "normalized_unique_values_unchanged": self.normalized_unique_values_unchanged,
            "exact_value_lists_unchanged": self.exact_value_lists_unchanged,
            "value_multiplicity_changes": self.value_multiplicity_changes,
            "semantic_action": self.semantic_action,
        }


def derive_value_change_summary(record: dict[str, Any]) -> ValueChangeSummary:
    repair_target = record.get("repair_target", {})
    if not isinstance(repair_target, dict):
        repair_target = {}
    old_raw = raw_value_list(repair_target.get("old_value"))
    new_raw = raw_value_list(repair_target.get("new_value"))
    if not old_raw and repair_target.get("action") == "DELETE":
        old_raw = raw_value_list(repair_target.get("value"))
    if not new_raw and repair_target.get("action") in {"UPDATE", "CREATE"}:
        new_raw = raw_value_list(repair_target.get("value"))

    old_values = [value for value in old_raw if value != "MISSING"]
    new_values = [value for value in new_raw if value != "MISSING"]
    old_counter = Counter(old_values)
    new_counter = Counter(new_values)
    old_set = set(old_counter)
    new_set = set(new_counter)
    retained = sorted(old_set & new_set)
    added = sorted(new_set - old_set)
    removed = sorted(old_set - new_set)
    multiplicity_changes = {
        value: {"old": old_counter.get(value, 0), "new": new_counter.get(value, 0)}
        for value in sorted(old_set | new_set)
        if old_counter.get(value, 0) != new_counter.get(value, 0)
    }
    same_unique = old_set == new_set
    exact_same = old_values == new_values

    action = repair_target.get("action")
    if not old_values and new_values:
        semantic_action = "CREATE_FROM_MISSING"
    elif old_values and not new_values:
        semantic_action = "DELETE_TO_MISSING"
    elif same_unique and exact_same:
        semantic_action = "NO_CHANGE_OR_REORDER_ONLY"
    elif same_unique:
        decreases = any(change["new"] < change["old"] for change in multiplicity_changes.values())
        increases = any(change["new"] > change["old"] for change in multiplicity_changes.values())
        if decreases and not increases:
            semantic_action = "MULTIPLICITY_DECREASE_SAME_UNIQUE"
        elif increases and not decreases:
            semantic_action = "MULTIPLICITY_INCREASE_SAME_UNIQUE"
        else:
            semantic_action = "MULTIPLICITY_CHANGE_SAME_UNIQUE"
    elif old_set and new_set and new_set < old_set:
        semantic_action = "DELETE_SUBSET"
    elif old_set and new_set and old_set < new_set:
        semantic_action = "ADD_SUPERSET"
    elif len(old_set) == 1 and len(new_set) == 1 and old_set != new_set:
        semantic_action = "REPLACE_1_TO_1"
    elif action == "DELETE":
        semantic_action = "DELETE_TO_MISSING"
    elif action == "CREATE":
        semantic_action = "CREATE_FROM_MISSING"
    else:
        semantic_action = "MIXED_UPDATE"

    return ValueChangeSummary(
        old_values_raw=old_raw,
        new_values_raw=new_raw,
        old_values=old_values,
        new_values=new_values,
        old_unique=sorted(old_set),
        new_unique=sorted(new_set),
        old_counts=dict(sorted(old_counter.items())),
        new_counts=dict(sorted(new_counter.items())),
        retained_unique_values=retained,
        added_unique_values=added,
        removed_unique_values=removed,
        normalized_unique_values_unchanged=same_unique,
        exact_value_lists_unchanged=exact_same,
        value_multiplicity_changes=multiplicity_changes,
        semantic_action=semantic_action,
    )


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
