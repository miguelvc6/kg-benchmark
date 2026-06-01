#!/usr/bin/env python3
"""
classifier.py — WikidataRepairEval 1.0
Phase 1 / Stage 4: Taxonomy of Information Necessity (Type A/B/C)

Reads:
  - data/02_wikidata_repairs.json  (or .jsonl)
  - data/03_world_state.json       (dict keyed by repair id)
  - (optional) data/00_entity_popularity.json

Writes:
  - data/04_classified_benchmark.jsonl          (LEAN, references world_state)
  - data/04_classified_benchmark_full.jsonl     (FULL, embeds world_state; optional)
  - reports/classifier_stats.json
"""

import argparse
import datetime as _dt
import json
import logging
import re
import sqlite3
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import ijson
from tqdm import tqdm

from lib.utils import (
    _json_default,
    count_repairs,
    is_pid,
    is_qid,
    iter_jsonl,
    iter_repairs,
    normalize_text,
    read_json,
    safe_get,
    utc_now_iso,
)
from lib.repair_state import pre_repair_target_raw_value
from lib.repair_state import derive_value_change_summary

DATE_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
WIKIDATA_DATE_RE = re.compile(r"^[+-]\d{4,}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z)?$")

# Default paths
DEFAULT_REPAIRS_PATH = "02_wikidata_repairs.json"
DEFAULT_WORLD_STATE_PATH = "03_world_state.json"
DEFAULT_POPULARITY_PATH = "00_entity_popularity.json"
DEFAULT_OUT_PATH = "04_classified_benchmark.jsonl"
DEFAULT_OUT_FULL_PATH = "04_classified_benchmark_full.jsonl"
DEFAULT_STATS_PATH = "reports/classifier_stats.json"

# Constraint QID families (observed in outputs)
FORMAT_QIDS = {"Q21502404"}  # format constraint
SINGLE_VALUE_QIDS = {"Q19474404"}
DISTINCT_VALUE_QIDS = {"Q21502410"}
ONE_OF_QIDS = {"Q21510859", "Q21502402"}  # one-of constraint (observed + legacy)
RANGE_QIDS = {"Q21510860"}  # range constraint
TYPE_QIDS = {"Q21503250"}  # subject type constraint
VALUE_TYPE_QIDS = {"Q21510865"}  # value-type constraint
SELF_LINK_QIDS: set[str] = set()
SYMMETRIC_QIDS = {"Q21510862"}
NONE_OF_QIDS = {"Q52558054"}
INVERSE_QIDS = {"Q21510855"}
MANDATORY_QUALIFIER_QIDS = {"Q21510856"}
ALLOWED_QUALIFIER_QIDS = {"Q21510851"}
REQUIRED_STATEMENT_QIDS = {"Q21503247"}
VALUE_REQUIRES_STATEMENT_QIDS = {"Q21510864"}
ALLOWED_ENTITY_TYPES_QIDS = {"Q52004125"}
PROPERTY_SCOPE_QIDS = {"Q53869507"}
CONFLICTS_WITH_QIDS = {"Q21502838"}
LABEL_IN_LANGUAGE_QIDS = {"Q108139345"}

ONE_OF_VALUE_QUALIFIER = "P2305"
TYPE_VALUE_QUALIFIERS = ("P2308", "P2309")
NUMERIC_MIN_QUALIFIER = "P2313"
NUMERIC_MAX_QUALIFIER = "P2312"
DATE_MIN_QUALIFIER = "P2310"
DATE_MAX_QUALIFIER = "P2311"
FORMAT_PATTERN_QUALIFIER = "P1793"
LANGUAGE_CODE_QUALIFIER = "P424"
CONSTRAINT_STATUS_QUALIFIER = "P2316"
PROPERTY_PARAMETER_QUALIFIER = "P2306"
SINGLE_VALUE_SEPARATOR_QUALIFIER = "P4155"
PROPERTY_SCOPE_QUALIFIER = "P4680"
TBOX_METADATA_QUALIFIER_PROPERTIES = {
    CONSTRAINT_STATUS_QUALIFIER,
}
CURRENT_VALUE_TRUTH_SOURCES = {
    "persistence_check.current_value_2026",
    "violation_context.value_current_2026",
    "persistence_check.current_value_2025",
    "violation_context.value_current_2025",
}

VIOLATION_TO_CONSTRAINT_MAP = {
    "Single value": "Q19474404",
    "Unique value": "Q21502410",
    "Distinct values": "Q21502410",
    "Format": "Q21502404",
    "One of": "Q21510859",
    "Inverse": "Q21510855",
    "Type": "Q21503250",
    "Value type": "Q21510865",
    "Symmetric": "Q21510862",
    "None of": "Q52558054",
    "Conflicts with": "Q21502838",
    "Mandatory Qualifiers": "Q21510856",
    "Mandatory qualifier": "Q21510856",
    "Allowed qualifiers": "Q21510851",
    "Item": "Q21503247",
    "Target required claim": "Q21510864",
    "Value requires statement": "Q21510864",
    "Entity types": "Q52004125",
    "Property scope": "Q53869507",
    "Scope": "Q53869507",
    "Label in language": "Q108139345",
    "Description in language": "Q108139345",
    "Range": "Q21510860",
    "Diff within range": "Q21510861",
    "Quantity": "Q21510857",
}

CONSTRAINT_LABELS = {
    "Q19474404": "single-value constraint",
    "Q21502410": "distinct-values constraint",
    "Q21502404": "format constraint",
    "Q21510859": "one-of constraint",
    "Q21502402": "one-of constraint",
    "Q21510855": "inverse constraint",
    "Q21503250": "type constraint",
    "Q21510865": "value-type constraint",
    "Q21510862": "symmetric constraint",
    "Q52558054": "none-of constraint",
    "Q21502838": "conflicts-with constraint",
    "Q21510856": "mandatory qualifier constraint",
    "Q21510851": "allowed qualifiers constraint",
    "Q21503247": "item requires statement constraint",
    "Q21510864": "value requires statement constraint",
    "Q52004125": "allowed entity types constraint",
    "Q53869507": "property scope constraint",
    "Q108139345": "label-in-language constraint",
    "Q21510860": "range constraint",
    "Q21510861": "diff within range constraint",
    "Q21510857": "quantity constraint",
}


def configure_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def extract_constraint_types(world_state_entry: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    """
    From world_state.L4_constraints.constraints[*].constraint_type -> [{'qid','label_en'}...]
    De-duplicates by qid while preserving first label.
    """
    seen = set()
    out: List[Dict[str, Optional[str]]] = []
    constraints = safe_get(world_state_entry, "L4_constraints", "constraints", default=[])
    if not isinstance(constraints, list):
        return out

    for c in constraints:
        ctype = c.get("constraint_type", {}) if isinstance(c, dict) else {}
        qid = ctype.get("qid")
        label = ctype.get("label")
        if isinstance(qid, str) and qid and qid not in seen:
            seen.add(qid)
            out.append({"qid": qid, "label_en": label if isinstance(label, str) else None})
    return out


def extract_constraint_entries(world_state_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    constraints = safe_get(world_state_entry, "L4_constraints", "constraints", default=[])
    if not isinstance(constraints, list):
        return []
    return [c for c in constraints if isinstance(c, dict)]


def constraint_kind(ctype: Dict[str, Any]) -> Optional[str]:
    qid = ctype.get("qid") if isinstance(ctype, dict) else None
    label = ctype.get("label") if isinstance(ctype, dict) else None
    if qid in SINGLE_VALUE_QIDS:
        return "single_value"
    if qid in DISTINCT_VALUE_QIDS:
        return "distinct_values"
    if qid in FORMAT_QIDS:
        return "format"
    if qid in ONE_OF_QIDS:
        return "one_of"
    if qid in RANGE_QIDS:
        return "range"
    if qid in TYPE_QIDS:
        return "type"
    if qid in VALUE_TYPE_QIDS:
        return "value_type"
    if qid in SELF_LINK_QIDS:
        return "self_link"
    if qid in SYMMETRIC_QIDS:
        return "symmetric"
    if qid in NONE_OF_QIDS:
        return "none_of"
    if qid in INVERSE_QIDS:
        return "inverse"
    if qid in CONFLICTS_WITH_QIDS:
        return "conflicts_with"
    if qid in MANDATORY_QUALIFIER_QIDS:
        return "mandatory_qualifier"
    if qid in ALLOWED_QUALIFIER_QIDS:
        return "allowed_qualifier"
    if qid in REQUIRED_STATEMENT_QIDS:
        return "required_statement"
    if qid in VALUE_REQUIRES_STATEMENT_QIDS:
        return "value_required_statement"
    if qid in ALLOWED_ENTITY_TYPES_QIDS:
        return "allowed_entity_types"
    if qid in PROPERTY_SCOPE_QIDS:
        return "property_scope"
    if qid in LABEL_IN_LANGUAGE_QIDS:
        return "label_in_language"
    if isinstance(label, str):
        ln = normalize_text(label)
        if ln in {"single-value constraint", "single value constraint", "single value"}:
            return "single_value"
        if ln in {"distinct-values constraint", "distinct values constraint", "unique value constraint", "unique value"}:
            return "distinct_values"
        if ln in {"format constraint", "format"}:
            return "format"
        if ln in {"one of constraint", "one-of constraint", "one of", "one-of"}:
            return "one_of"
        if ln in {"range constraint", "range"}:
            return "range"
        if ln in {"type constraint", "type"}:
            return "type"
        if ln in {"value type constraint", "value-type constraint", "value type", "value-type"}:
            return "value_type"
        if ln in {"self-link constraint", "self link constraint", "self link"}:
            return "self_link"
        if ln in {"symmetric constraint", "symmetric"}:
            return "symmetric"
        if ln in {"none of constraint", "none-of constraint", "none of", "none-of"}:
            return "none_of"
        if ln in {"inverse constraint", "inverse"}:
            return "inverse"
        if ln in {"conflicts-with constraint", "conflicts with constraint", "conflicts with"}:
            return "conflicts_with"
        if ln in {"mandatory qualifier constraint", "mandatory qualifiers constraint", "mandatory qualifiers"}:
            return "mandatory_qualifier"
        if ln in {"allowed qualifiers constraint", "allowed qualifier constraint", "allowed qualifiers"}:
            return "allowed_qualifier"
        if ln in {"item requires statement constraint", "required statement constraint"}:
            return "required_statement"
        if ln in {"value requires statement constraint", "value requires statement", "target required claim"}:
            return "value_required_statement"
        if ln in {"allowed entity types constraint", "entity types constraint", "entity types"}:
            return "allowed_entity_types"
        if ln in {"property scope constraint", "property scope"}:
            return "property_scope"
        if ln in {"label-in-language constraint", "label in language constraint", "label in language", "description in language"}:
            return "label_in_language"
    return None


def is_causal_repair(violation_name: Optional[str], changed_constraint_qids: Iterable[str]) -> bool:
    if not isinstance(violation_name, str) or not violation_name.strip():
        return False
    if changed_constraint_qids is None:
        changed_constraint_qids = []
    target_qid = _map_tbox_violation(violation_name).get("mapped_violation_constraint_qid")
    if not target_qid:
        return False
    changed_set = {qid for qid in changed_constraint_qids if isinstance(qid, str)}
    return target_qid in changed_set


def _ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                return loaded
        except Exception:
            return []
    return []


def _load_signature_list(delta: Dict[str, Any], primary_key: str, fallback_key: str) -> List[Dict[str, Any]]:
    raw = delta.get(primary_key)
    if isinstance(raw, list):
        return [entry for entry in raw if isinstance(entry, dict)]
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                return [entry for entry in loaded if isinstance(entry, dict)]
        except Exception:
            return []
    fallback = delta.get(fallback_key)
    if isinstance(fallback, list):
        return [entry for entry in fallback if isinstance(entry, dict)]
    if isinstance(fallback, str):
        try:
            loaded = json.loads(fallback)
            if isinstance(loaded, list):
                return [entry for entry in loaded if isinstance(entry, dict)]
        except Exception:
            return []
    return []


def _collect_qualifiers_for_qid(signature: List[Dict[str, Any]], qid: str) -> List[Dict[str, Any]]:
    qualifiers: List[Dict[str, Any]] = []
    for entry in signature:
        if not isinstance(entry, dict):
            continue
        if entry.get("constraint_qid") != qid:
            continue
        q_list = entry.get("qualifiers")
        if isinstance(q_list, list):
            qualifiers.extend([q for q in q_list if isinstance(q, dict)])
    return qualifiers


def _collect_qualifier_values(
    qualifiers: List[Dict[str, Any]],
    property_id: Optional[str] = None,
    property_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    out: List[str] = []
    property_id_set = set(property_ids) if property_ids is not None else None
    for q in qualifiers:
        if not isinstance(q, dict):
            continue
        q_pid = q.get("property_id")
        if property_id and q_pid != property_id:
            continue
        if property_id_set is not None and q_pid not in property_id_set:
            continue
        values = q.get("values")
        if values is None:
            continue
        for tok in flatten_truth(values):
            if tok is None:
                continue
            out.append(str(tok))
    return out


def _extract_numeric_bound(
    qualifiers: List[Dict[str, Any]],
    property_id: str,
    prefer: str,
) -> Optional[Decimal]:
    values = _collect_qualifier_values(qualifiers, property_id=property_id)
    nums = [n for n in (_parse_numeric_token(v) for v in values) if n is not None]
    if not nums:
        return None
    return min(nums) if prefer == "min" else max(nums)


def _extract_date_bound(qualifiers: List[Dict[str, Any]], property_id: str, prefer: str) -> Optional[str]:
    values = _collect_qualifier_values(qualifiers, property_id=property_id)
    dates = [d for d in (_parse_date_boundary(v) for v in values) if d is not None]
    if not dates:
        return None
    return min(dates) if prefer == "min" else max(dates)


def analyze_range_change(old_qualifiers: List[Dict[str, Any]], new_qualifiers: List[Dict[str, Any]]) -> str:
    old_min = _extract_numeric_bound(old_qualifiers, NUMERIC_MIN_QUALIFIER, "min")
    old_max = _extract_numeric_bound(old_qualifiers, NUMERIC_MAX_QUALIFIER, "max")
    new_min = _extract_numeric_bound(new_qualifiers, NUMERIC_MIN_QUALIFIER, "min")
    new_max = _extract_numeric_bound(new_qualifiers, NUMERIC_MAX_QUALIFIER, "max")
    old_date_min = _extract_date_bound(old_qualifiers, DATE_MIN_QUALIFIER, "min")
    old_date_max = _extract_date_bound(old_qualifiers, DATE_MAX_QUALIFIER, "max")
    new_date_min = _extract_date_bound(new_qualifiers, DATE_MIN_QUALIFIER, "min")
    new_date_max = _extract_date_bound(new_qualifiers, DATE_MAX_QUALIFIER, "max")

    widened = False
    narrowed = False
    if old_min is not None and new_min is not None:
        if new_min < old_min:
            widened = True
        elif new_min > old_min:
            narrowed = True
    if old_max is not None and new_max is not None:
        if new_max > old_max:
            widened = True
        elif new_max < old_max:
            narrowed = True
    if old_date_min is not None and new_date_min is not None:
        if new_date_min < old_date_min:
            widened = True
        elif new_date_min > old_date_min:
            narrowed = True
    if old_date_max is not None and new_date_max is not None:
        if new_date_max > old_date_max:
            widened = True
        elif new_date_max < old_date_max:
            narrowed = True

    if widened:
        return "RELAXATION_RANGE_WIDENED"
    if narrowed:
        return "RESTRICTION_RANGE_NARROWED"
    return "SCHEMA_UPDATE"


def _compare_sets(old_values: set, new_values: set) -> str:
    if not old_values or not new_values:
        return "SCHEMA_UPDATE"
    if old_values < new_values:
        return "RELAXATION_SET_EXPANSION"
    if new_values < old_values:
        return "RESTRICTION_SET_CONTRACTION"
    return "SCHEMA_UPDATE"


def _analyze_set_change(
    old_qualifiers: List[Dict[str, Any]],
    new_qualifiers: List[Dict[str, Any]],
    property_id: Optional[str] = None,
    property_ids: Optional[Iterable[str]] = None,
) -> str:
    old_values = set(_collect_qualifier_values(old_qualifiers, property_id=property_id, property_ids=property_ids))
    new_values = set(_collect_qualifier_values(new_qualifiers, property_id=property_id, property_ids=property_ids))
    return _compare_sets(old_values, new_values)


def _analyze_generic_set_change(
    old_qualifiers: List[Dict[str, Any]],
    new_qualifiers: List[Dict[str, Any]],
) -> str:
    old_values = set(_collect_qualifier_values(old_qualifiers))
    new_values = set(_collect_qualifier_values(new_qualifiers))
    return _compare_sets(old_values, new_values)


def _constraint_label(qid: Optional[str], signature: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    if isinstance(signature, list) and isinstance(qid, str):
        for entry in signature:
            if not isinstance(entry, dict) or entry.get("constraint_qid") != qid:
                continue
            label = entry.get("constraint_label") or entry.get("label")
            if isinstance(label, str) and label:
                return label
    if isinstance(qid, str):
        return CONSTRAINT_LABELS.get(qid)
    return None


def _preview_list(values: Iterable[Any], limit: int = 25) -> List[Any]:
    items = list(values) if not isinstance(values, list) else values
    if len(items) <= limit:
        return items
    return items[:limit] + [f"...(+{len(items) - limit})"]


def _constraint_family(qid: Optional[str], label: Optional[str] = None) -> str:
    kind = constraint_kind({"qid": qid, "label": label})
    return kind or "unknown"


def _constraint_set_semantics(qid: Optional[str], label: Optional[str] = None) -> str:
    family = _constraint_family(qid, label)
    if family in {
        "one_of",
        "type",
        "value_type",
        "allowed_qualifier",
        "allowed_entity_types",
        "property_scope",
        "single_value",
        "distinct_values",
        "label_in_language",
        "symmetric",
    }:
        return "allowed"
    if family in {"none_of", "conflicts_with"}:
        return "forbidden"
    if family in {"mandatory_qualifier", "required_statement", "value_required_statement"}:
        return "required"
    return "unknown"


def _constraint_value_property_ids(qid: Optional[str], label: Optional[str] = None) -> Optional[Iterable[str]]:
    family = _constraint_family(qid, label)
    if family == "one_of":
        return {ONE_OF_VALUE_QUALIFIER}
    if family in {"type", "value_type"}:
        return set(TYPE_VALUE_QUALIFIERS)
    return None


def tbox_relevant_qualifier_properties_for_family(family_or_constraint_qid: Optional[str]) -> Optional[set[str]]:
    if not isinstance(family_or_constraint_qid, str) or not family_or_constraint_qid:
        return None
    family = (
        _constraint_family(family_or_constraint_qid, _constraint_label(family_or_constraint_qid))
        if family_or_constraint_qid.startswith("Q")
        else family_or_constraint_qid
    )
    relevant: dict[str, set[str]] = {
        "format": {FORMAT_PATTERN_QUALIFIER},
        "one_of": {ONE_OF_VALUE_QUALIFIER},
        "none_of": {ONE_OF_VALUE_QUALIFIER},
        "type": set(TYPE_VALUE_QUALIFIERS),
        "value_type": set(TYPE_VALUE_QUALIFIERS),
        "allowed_entity_types": set(TYPE_VALUE_QUALIFIERS),
        "label_in_language": {LANGUAGE_CODE_QUALIFIER},
        "description_in_language": {LANGUAGE_CODE_QUALIFIER},
        "allowed_qualifier": {PROPERTY_PARAMETER_QUALIFIER},
        "mandatory_qualifier": {PROPERTY_PARAMETER_QUALIFIER},
        "required_statement": {PROPERTY_PARAMETER_QUALIFIER, ONE_OF_VALUE_QUALIFIER},
        "value_required_statement": {PROPERTY_PARAMETER_QUALIFIER, ONE_OF_VALUE_QUALIFIER},
        "inverse": {PROPERTY_PARAMETER_QUALIFIER},
        "conflicts_with": {PROPERTY_PARAMETER_QUALIFIER, ONE_OF_VALUE_QUALIFIER},
        "property_scope": {PROPERTY_SCOPE_QUALIFIER},
        "range": {DATE_MIN_QUALIFIER, DATE_MAX_QUALIFIER, NUMERIC_MIN_QUALIFIER, NUMERIC_MAX_QUALIFIER},
        "single_value": {SINGLE_VALUE_SEPARATOR_QUALIFIER},
        "distinct_values": {PROPERTY_PARAMETER_QUALIFIER},
        "symmetric": {PROPERTY_PARAMETER_QUALIFIER},
    }
    return relevant.get(family)


def is_tbox_metadata_qualifier_property(pid: str) -> bool:
    return isinstance(pid, str) and pid in TBOX_METADATA_QUALIFIER_PROPERTIES


def _values_from_change_rows(rows: List[Dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    added: list[str] = []
    removed: list[str] = []
    props: list[str] = []
    for row in rows:
        prop = row.get("qualifier_property")
        if isinstance(prop, str) and prop not in props:
            props.append(prop)
        added.extend(str(value) for value in row.get("added_values", []))
        removed.extend(str(value) for value in row.get("removed_values", []))
    return sorted(set(added)), sorted(set(removed)), sorted(props)


def filter_tbox_semantic_qualifier_changes(
    changes: List[Dict[str, Any]],
    target_family_or_qid: Optional[str],
    *,
    target_constraint_qid: Optional[str] = None,
) -> Dict[str, Any]:
    relevant = tbox_relevant_qualifier_properties_for_family(target_family_or_qid)
    family = (
        _constraint_family(target_family_or_qid, _constraint_label(target_family_or_qid))
        if isinstance(target_family_or_qid, str) and target_family_or_qid.startswith("Q")
        else target_family_or_qid or "unknown"
    )
    semantic_rows: List[Dict[str, Any]] = []
    ignored_rows: List[Dict[str, Any]] = []
    for row in changes:
        row_qid = row.get("constraint_qid")
        if target_constraint_qid and row_qid != target_constraint_qid:
            continue
        prop = row.get("qualifier_property")
        prop_s = str(prop) if prop is not None else ""
        if relevant is not None and prop_s in relevant and not is_tbox_metadata_qualifier_property(prop_s):
            semantic_rows.append(row)
        else:
            ignored_rows.append(row)
    semantic_added, semantic_removed, semantic_props = _values_from_change_rows(semantic_rows)
    ignored_added, ignored_removed, ignored_props = _values_from_change_rows(ignored_rows)
    if relevant is None:
        reason = "unknown_family_no_semantic_qualifier_filter"
    elif ignored_rows and semantic_rows:
        reason = "family_relevant_qualifiers_kept_metadata_or_irrelevant_qualifiers_ignored"
    elif ignored_rows:
        reason = "all_changed_qualifiers_are_metadata_or_irrelevant_for_family"
    else:
        reason = "all_changed_qualifiers_are_semantic_for_family"
    return {
        "target_family": family,
        "relevant_qualifier_properties": sorted(relevant) if relevant is not None else [],
        "semantic_changed_qualifier_properties": semantic_props,
        "ignored_changed_qualifier_properties": ignored_props,
        "semantic_added_values": semantic_added,
        "semantic_removed_values": semantic_removed,
        "ignored_added_values": ignored_added,
        "ignored_removed_values": ignored_removed,
        "semantic_added_value_count": len(semantic_added),
        "semantic_removed_value_count": len(semantic_removed),
        "ignored_value_count": len(ignored_added) + len(ignored_removed),
        "qualifier_filter_reason": reason,
        "semantic_change_rows": semantic_rows,
        "ignored_change_rows": ignored_rows,
    }


def _qualifier_value_counter(
    constraints: List[Dict[str, Any]],
    qid: str,
    property_ids: Optional[Iterable[str]] = None,
) -> Counter[str]:
    qualifiers = _collect_qualifiers_for_qid(constraints, qid)
    return Counter(_collect_qualifier_values(qualifiers, property_ids=property_ids))


def _qualifier_value_changes_for_qid(
    before: List[Dict[str, Any]],
    after: List[Dict[str, Any]],
    qid: str,
    property_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    old_counter = _qualifier_value_counter(before, qid, property_ids)
    new_counter = _qualifier_value_counter(after, qid, property_ids)
    before_props: Dict[str, Counter[str]] = {}
    after_props: Dict[str, Counter[str]] = {}
    for target, bucket in ((before, before_props), (after, after_props)):
        for qualifier in _collect_qualifiers_for_qid(target, qid):
            property_id = str(qualifier.get("property_id") or "UNKNOWN")
            bucket.setdefault(property_id, Counter()).update(_collect_qualifier_values([qualifier], property_ids=property_ids))
    changed_props = sorted(
        prop
        for prop in set(before_props) | set(after_props)
        if (before_props.get(prop, Counter()) - after_props.get(prop, Counter()))
        or (after_props.get(prop, Counter()) - before_props.get(prop, Counter()))
    )
    return {
        "added_values": sorted((new_counter - old_counter).elements()),
        "removed_values": sorted((old_counter - new_counter).elements()),
        "old_values": sorted(old_counter.elements()),
        "new_values": sorted(new_counter.elements()),
        "changed_qualifier_properties": changed_props,
    }


def _qid_values_from_entries(entries: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        qid = entry.get("constraint_qid")
        if isinstance(qid, str) and qid not in out:
            out.append(qid)
    return out


def _signature_entry_diff(before: List[Dict[str, Any]], after: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    before_qids = Counter(entry.get("constraint_qid") for entry in before if isinstance(entry, dict) and isinstance(entry.get("constraint_qid"), str))
    after_qids = Counter(entry.get("constraint_qid") for entry in after if isinstance(entry, dict) and isinstance(entry.get("constraint_qid"), str))
    changed = sorted(set((after_qids - before_qids).keys()) | set((before_qids - after_qids).keys()))
    return changed, list(changed)


def _qualifier_value_change_rows(before: List[Dict[str, Any]], after: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def counters_by_qid_and_property(entries: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Counter[str]]:
        result: Dict[Tuple[str, str], Counter[str]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("constraint_qid"), str):
                continue
            qid = entry["constraint_qid"]
            qualifiers = entry.get("qualifiers")
            if not isinstance(qualifiers, list):
                continue
            for qualifier in qualifiers:
                if not isinstance(qualifier, dict):
                    continue
                property_id = str(qualifier.get("property_id") or "UNKNOWN")
                result.setdefault((qid, property_id), Counter()).update(_collect_qualifier_values([qualifier]))
        return result

    before_by_key = counters_by_qid_and_property(before)
    after_by_key = counters_by_qid_and_property(after)
    rows: List[Dict[str, Any]] = []
    for qid, property_id in sorted(set(before_by_key) | set(after_by_key)):
        old_counter = before_by_key.get((qid, property_id), Counter())
        new_counter = after_by_key.get((qid, property_id), Counter())
        added = sorted((new_counter - old_counter).elements())
        removed = sorted((old_counter - new_counter).elements())
        if added or removed:
            rows.append(
                {
                    "constraint_qid": qid,
                    "qualifier_property": property_id,
                    "added_values": added,
                    "removed_values": removed,
                }
            )
    return rows


def _extract_report_tokens_from_text(texts: str) -> Dict[str, List[str]]:
    texts = texts or ""
    qids = sorted(set(re.findall(r"\bQ\|?(\d+)\b", texts, flags=re.IGNORECASE)))
    pids = sorted(set(re.findall(r"\bP\|?(\d+)\b", texts, flags=re.IGNORECASE)))
    langs = sorted(set(re.findall(r"\b(?:label|description)\s+in\s+([a-z][a-z0-9-]*)\s+language\b", texts, flags=re.IGNORECASE)))
    scopes = sorted(set(re.findall(r"\b(?:as|scope)\s+(?:a\s+)?([a-z][a-z _-]{2,30})\b", texts, flags=re.IGNORECASE)))
    return {
        "qids": [f"Q{qid}" for qid in qids],
        "pids": [f"P{pid}" for pid in pids],
        "langs": [lang.lower() for lang in langs],
        "scope_values": [normalize_text(scope) for scope in scopes],
    }


def _extract_report_tokens(repair_event: Dict[str, Any]) -> Dict[str, List[str]]:
    texts = " ".join(_violation_report_texts(repair_event))
    vc = repair_event.get("violation_context", {})
    for value in flatten_truth(vc if isinstance(vc, dict) else {}):
        texts += f" {value}"
    return _extract_report_tokens_from_text(texts)


def _map_tbox_violation(violation_name: Optional[str]) -> Dict[str, Any]:
    normalized = normalize_text(violation_name or "")
    if not normalized:
        return {
            "mapped_violation_constraint_qid": None,
            "mapped_violation_constraint_label": None,
            "mapped_violation_family": "unknown",
            "mapped_violation_confidence": "none",
            "mapped_violation_reason": "missing_violation_name",
        }
    mapping = {normalize_text(k): v for k, v in VIOLATION_TO_CONSTRAINT_MAP.items()}
    qid = None
    reason = "exact_violation_type_mapping"
    confidence = "high"
    if normalized in mapping:
        qid = mapping[normalized]
    elif normalized.startswith("value type"):
        qid = "Q21510865"
        reason = "value_type_prefix_mapping"
        confidence = "medium"
    elif normalized.startswith("type q") or normalized == "type":
        qid = "Q21503250"
        reason = "type_prefix_mapping"
        confidence = "medium"
    elif normalized.startswith("item p"):
        qid = "Q21503247"
        reason = "item_requires_statement_prefix_mapping"
        confidence = "medium"
    elif normalized.startswith("conflicts with"):
        qid = "Q21502838"
        reason = "conflicts_with_prefix_mapping"
        confidence = "medium"
    elif normalized.startswith("target required claim") or normalized.startswith("value requires statement"):
        qid = "Q21510864"
        reason = "value_requires_statement_prefix_mapping"
        confidence = "medium"
    elif normalized.startswith("label in"):
        qid = "Q108139345"
        reason = "label_language_report_mapping"
        confidence = "medium"
    elif normalized.startswith("description in"):
        qid = "Q108139345"
        reason = "description_language_report_mapping"
        confidence = "medium"
    elif normalized.startswith("mandatory qualifier") or normalized.startswith("mandatory qualifiers"):
        qid = "Q21510856"
        reason = "mandatory_qualifier_prefix_mapping"
        confidence = "medium"
    elif normalized.startswith("allowed qualifier") or normalized.startswith("allowed qualifiers"):
        qid = "Q21510851"
        reason = "allowed_qualifier_prefix_mapping"
        confidence = "medium"
    elif normalized.startswith("scope") or normalized.startswith("property scope"):
        qid = "Q53869507"
        reason = "property_scope_prefix_mapping"
        confidence = "medium"
    elif normalized.startswith("symmetric"):
        qid = "Q21510862"
        reason = "symmetric_prefix_mapping"
        confidence = "medium"
    if qid:
        return {
            "mapped_violation_constraint_qid": qid,
            "mapped_violation_constraint_label": _constraint_label(qid),
            "mapped_violation_family": _constraint_family(qid, _constraint_label(qid)),
            "mapped_violation_confidence": confidence,
            "mapped_violation_reason": reason,
        }
    return {
        "mapped_violation_constraint_qid": None,
        "mapped_violation_constraint_label": None,
        "mapped_violation_family": "unknown",
        "mapped_violation_confidence": "none",
        "mapped_violation_reason": "unmapped_violation_type",
    }


def _overlap_detail(changes: List[Dict[str, Any]], report_tokens: Dict[str, List[str]]) -> Dict[str, List[str]]:
    changed_values: set[str] = set()
    for row in changes:
        changed_values.update(str(value) for value in row.get("added_values", []))
        changed_values.update(str(value) for value in row.get("removed_values", []))
    qids = sorted(set(report_tokens.get("qids", [])) & changed_values)
    pids = sorted(set(report_tokens.get("pids", [])) & changed_values)
    langs = sorted(set(report_tokens.get("langs", [])) & {normalize_text(value) for value in changed_values})
    scopes = sorted(set(report_tokens.get("scope_values", [])) & {normalize_text(value) for value in changed_values})
    return {
        "value_overlap_with_report_qids": qids,
        "property_overlap_with_report_pids": pids,
        "language_overlap_with_report_langs": langs,
        "scope_overlap_with_report_values": scopes,
    }


def compatible_overlap_for_mapped_family(
    mapped_family: str,
    report_tokens: Dict[str, List[str]],
    semantic_change_summary: Dict[str, Any],
    raw_overlap: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    semantic_values = set(map(str, semantic_change_summary.get("semantic_added_values", []))) | set(
        map(str, semantic_change_summary.get("semantic_removed_values", []))
    )
    semantic_values_normalized = {normalize_text(value) for value in semantic_values}
    semantic_props = set(semantic_change_summary.get("semantic_changed_qualifier_properties", []))
    compatible = {
        "value_overlap_with_report_qids": [],
        "property_overlap_with_report_pids": [],
        "language_overlap_with_report_langs": [],
        "scope_overlap_with_report_values": [],
        "compatible_overlap_used": False,
        "compatible_overlap_reason": "no_type_compatible_overlap",
        "incompatible_overlap_ignored": {},
    }
    qids = sorted(set(report_tokens.get("qids", [])) & semantic_values)
    pids = sorted(set(report_tokens.get("pids", [])) & semantic_values)
    langs = sorted(set(report_tokens.get("langs", [])) & semantic_values_normalized)
    scopes = sorted(set(report_tokens.get("scope_values", [])) & semantic_values_normalized)

    if mapped_family == "format":
        if FORMAT_PATTERN_QUALIFIER in semantic_props:
            compatible["compatible_overlap_used"] = True
            compatible["compatible_overlap_reason"] = "format_regex_qualifier_changed"
    elif mapped_family in {"label_in_language", "description_in_language"}:
        compatible["language_overlap_with_report_langs"] = langs
    elif mapped_family in {"type", "value_type", "allowed_entity_types", "one_of", "none_of"}:
        compatible["value_overlap_with_report_qids"] = qids
    elif mapped_family in {"allowed_qualifier", "mandatory_qualifier", "required_statement", "value_required_statement", "inverse"}:
        compatible["property_overlap_with_report_pids"] = pids
    elif mapped_family == "conflicts_with":
        compatible["property_overlap_with_report_pids"] = pids
        compatible["value_overlap_with_report_qids"] = qids
    elif mapped_family == "property_scope":
        compatible["scope_overlap_with_report_values"] = scopes
    elif mapped_family == "range":
        if semantic_props & {DATE_MIN_QUALIFIER, DATE_MAX_QUALIFIER, NUMERIC_MIN_QUALIFIER, NUMERIC_MAX_QUALIFIER}:
            compatible["compatible_overlap_used"] = True
            compatible["compatible_overlap_reason"] = "range_bound_qualifier_changed"
    elif mapped_family in {"single_value", "distinct_values", "symmetric"}:
        if semantic_props or semantic_values:
            compatible["compatible_overlap_used"] = True
            compatible["compatible_overlap_reason"] = "family_specific_semantic_qualifier_changed"

    if _has_any_overlap(compatible):
        compatible["compatible_overlap_used"] = True
        compatible["compatible_overlap_reason"] = f"{mapped_family}_compatible_report_argument_overlap"
    if raw_overlap:
        ignored = {
            key: sorted(set(raw_overlap.get(key, [])) - set(compatible.get(key, [])))
            for key in (
                "value_overlap_with_report_qids",
                "property_overlap_with_report_pids",
                "language_overlap_with_report_langs",
                "scope_overlap_with_report_values",
            )
            if set(raw_overlap.get(key, [])) - set(compatible.get(key, []))
        }
        compatible["incompatible_overlap_ignored"] = ignored
    return compatible


def _family_has_concrete_report_tokens(family: str, report_tokens: Dict[str, List[str]]) -> bool:
    if family in {"type", "value_type", "allowed_entity_types", "one_of", "none_of"}:
        return bool(report_tokens.get("qids"))
    if family in {
        "conflicts_with",
        "inverse",
        "allowed_qualifier",
        "mandatory_qualifier",
        "required_statement",
        "value_required_statement",
    }:
        return bool(report_tokens.get("pids"))
    if family in {"label_in_language", "description_in_language"}:
        return bool(report_tokens.get("langs"))
    if family == "property_scope":
        return bool(report_tokens.get("scope_values"))
    return False


def report_has_concrete_arguments(
    candidate_violation_name: Optional[str],
    report_tokens: Dict[str, List[str]],
    violation_context: Optional[Dict[str, Any]] = None,
) -> bool:
    text = normalize_text(candidate_violation_name or "")
    if report_tokens.get("qids") or report_tokens.get("pids") or report_tokens.get("langs") or report_tokens.get("scope_values"):
        return True
    if isinstance(violation_context, dict):
        qids = violation_context.get("report_violation_type_qids")
        if isinstance(qids, list) and qids:
            return True
    return bool(
        re.search(r"\b(?:type|value type)\s+q\|?\d+\b", text)
        or re.search(r"\b(?:item|target required claim|conflicts with)\s+p\|?\d+\b", text)
        or re.search(r"\b(?:label|description)\s+in\s+[a-z][a-z0-9-]*\s+language\b", text)
    )


def _compatible_overlap_for_family(
    family: str,
    overlap: Dict[str, List[str]],
    report_tokens: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    compatible = {
        "value_overlap_with_report_qids": [],
        "property_overlap_with_report_pids": [],
        "language_overlap_with_report_langs": [],
        "scope_overlap_with_report_values": [],
    }
    if family in {"type", "value_type", "allowed_entity_types", "one_of", "none_of"}:
        compatible["value_overlap_with_report_qids"] = list(overlap.get("value_overlap_with_report_qids", []))
    elif family in {"conflicts_with"}:
        compatible["property_overlap_with_report_pids"] = list(overlap.get("property_overlap_with_report_pids", []))
        compatible["value_overlap_with_report_qids"] = list(overlap.get("value_overlap_with_report_qids", []))
    elif family in {
        "inverse",
        "allowed_qualifier",
        "mandatory_qualifier",
        "required_statement",
        "value_required_statement",
    }:
        compatible["property_overlap_with_report_pids"] = list(overlap.get("property_overlap_with_report_pids", []))
    elif family in {"label_in_language", "description_in_language"}:
        compatible["language_overlap_with_report_langs"] = list(overlap.get("language_overlap_with_report_langs", []))
    elif family == "property_scope":
        compatible["scope_overlap_with_report_values"] = list(overlap.get("scope_overlap_with_report_values", []))
    elif family in {"format", "single_value", "distinct_values", "symmetric", "range"}:
        # These families normally carry causality by exact constraint-family match.
        return compatible
    return compatible


def _has_any_overlap(overlap: Dict[str, List[str]]) -> bool:
    return any(bool(overlap.get(key)) for key in (
        "value_overlap_with_report_qids",
        "property_overlap_with_report_pids",
        "language_overlap_with_report_langs",
        "scope_overlap_with_report_values",
    ))


def _families_related_for_tbox_target(mapped_family: str, changed_family: str, compatible_overlap: Dict[str, Any]) -> bool:
    if not compatible_overlap.get("compatible_overlap_used"):
        return False
    related = {
        "type": {"allowed_entity_types"},
        "value_type": {"allowed_entity_types"},
        "allowed_entity_types": {"type", "value_type"},
        "required_statement": {"value_required_statement"},
        "value_required_statement": {"required_statement"},
        "label_in_language": {"description_in_language"},
        "description_in_language": {"label_in_language"},
    }
    return changed_family in related.get(mapped_family, set())


def _analysis_slice_precise_for_tbox(step: Dict[str, Any]) -> str:
    precise = step.get("directional_subtype_precise")
    if isinstance(precise, str) and precise:
        return f"main_tbox_{precise.lower()}"
    subtype = step.get("result")
    if subtype == "UNKNOWN_TBOX_CAUSALITY":
        return "diagnostic_tbox_unknown_causality"
    if subtype == "COINCIDENTAL_SCHEMA_CHANGE":
        return "diagnostic_tbox_coincidental"
    return f"main_tbox_{str(subtype).lower()}" if subtype else ""


DIRECTIONAL_TBOX_PUBLIC_SUBTYPES = {"RELAXATION_SET_EXPANSION", "RESTRICTION_SET_CONTRACTION"}


def _active_and_potential_tbox_direction_details(
    result: str,
    polarity_detail: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if result in DIRECTIONAL_TBOX_PUBLIC_SUBTYPES:
        active = {
            "set_semantics": polarity_detail.get("set_semantics"),
            "set_operation": polarity_detail.get("set_operation"),
            "polarity": polarity_detail.get("polarity"),
            "polarity_basis": polarity_detail.get("polarity_basis"),
            "directional_subtype_basis": polarity_detail.get("directional_subtype_basis"),
            "directional_subtype_precise": polarity_detail.get("directional_subtype_precise"),
        }
        active["analysis_slice_precise"] = _analysis_slice_precise_for_tbox({"result": result, **active})
        return active, {}

    active = {
        "set_semantics": polarity_detail.get("set_semantics"),
        "set_operation": polarity_detail.get("set_operation"),
        "polarity": "unknown",
        "polarity_basis": "not active because final T-box subtype is non-directional",
        "directional_subtype_basis": None,
        "directional_subtype_precise": None,
        "analysis_slice_precise": _analysis_slice_precise_for_tbox({"result": result}),
    }
    potential = {
        "potential_set_semantics": polarity_detail.get("set_semantics"),
        "potential_set_operation": polarity_detail.get("set_operation"),
        "potential_polarity": polarity_detail.get("polarity"),
        "potential_polarity_basis": polarity_detail.get("polarity_basis"),
        "potential_directional_subtype_basis": polarity_detail.get("directional_subtype_basis"),
        "potential_directional_subtype_precise": polarity_detail.get("directional_subtype_precise"),
    }
    return active, potential


def _polarity_from_delta(
    *,
    qid: str,
    label: Optional[str],
    added_values: List[str],
    removed_values: List[str],
) -> Dict[str, Any]:
    set_semantics = _constraint_set_semantics(qid, label)
    added = bool(added_values)
    removed = bool(removed_values)
    set_operation = "mixed" if added and removed else "expansion" if added else "contraction" if removed else "unchanged"
    if added and removed:
        return {
            "subtype": "SCHEMA_UPDATE",
            "polarity": "unknown",
            "set_semantics": set_semantics,
            "set_operation": set_operation,
            "polarity_basis": "both_added_and_removed_values",
            "directional_subtype_basis": "mixed qualifier-value change",
            "directional_subtype_precise": None,
        }
    if set_semantics == "allowed":
        if added:
            return {
                "subtype": "RELAXATION_SET_EXPANSION",
                "polarity": "relaxation",
                "set_semantics": set_semantics,
                "set_operation": set_operation,
                "polarity_basis": "allowed set gained values",
                "directional_subtype_basis": "allowed set expansion",
                "directional_subtype_precise": "RELAXATION_ALLOWED_SET_EXPANSION",
            }
        if removed:
            return {
                "subtype": "RESTRICTION_SET_CONTRACTION",
                "polarity": "restriction",
                "set_semantics": set_semantics,
                "set_operation": set_operation,
                "polarity_basis": "allowed set lost values",
                "directional_subtype_basis": "allowed set contraction",
                "directional_subtype_precise": "RESTRICTION_ALLOWED_SET_CONTRACTION",
            }
    if set_semantics == "forbidden":
        if added:
            return {
                "subtype": "RESTRICTION_SET_CONTRACTION",
                "polarity": "restriction",
                "set_semantics": set_semantics,
                "set_operation": set_operation,
                "polarity_basis": "forbidden set gained prohibited values",
                "directional_subtype_basis": "forbidden set expansion",
                "directional_subtype_precise": "RESTRICTION_FORBIDDEN_SET_EXPANSION",
            }
        if removed:
            return {
                "subtype": "RELAXATION_SET_EXPANSION",
                "polarity": "relaxation",
                "set_semantics": set_semantics,
                "set_operation": set_operation,
                "polarity_basis": "forbidden set lost prohibited values",
                "directional_subtype_basis": "forbidden set contraction",
                "directional_subtype_precise": "RELAXATION_FORBIDDEN_SET_CONTRACTION",
            }
    if set_semantics == "required":
        if added:
            return {
                "subtype": "RESTRICTION_SET_CONTRACTION",
                "polarity": "restriction",
                "set_semantics": set_semantics,
                "set_operation": set_operation,
                "polarity_basis": "required set gained required values",
                "directional_subtype_basis": "required set expansion",
                "directional_subtype_precise": "RESTRICTION_REQUIRED_SET_EXPANSION",
            }
        if removed:
            return {
                "subtype": "RELAXATION_SET_EXPANSION",
                "polarity": "relaxation",
                "set_semantics": set_semantics,
                "set_operation": set_operation,
                "polarity_basis": "required set lost required values",
                "directional_subtype_basis": "required set contraction",
                "directional_subtype_precise": "RELAXATION_REQUIRED_SET_CONTRACTION",
            }
    return {
        "subtype": "SCHEMA_UPDATE",
        "polarity": "unknown",
        "set_semantics": set_semantics,
        "set_operation": set_operation,
        "polarity_basis": "unknown constraint-family polarity",
        "directional_subtype_basis": "causal family match without interpretable polarity",
        "directional_subtype_precise": None,
    }


class ConstraintDiffer:
    def __init__(self, repair_event: Dict[str, Any], constraint_delta: Optional[Dict[str, Any]]):
        self.repair_event = repair_event
        self.delta = constraint_delta if isinstance(constraint_delta, dict) else {}
        changed_from_delta = [
            qid for qid in _ensure_list(self.delta.get("changed_constraint_types")) if isinstance(qid, str)
        ]
        self.signature_before = _load_signature_list(self.delta, "signature_before", "old_constraints")
        self.signature_after = _load_signature_list(self.delta, "signature_after", "new_constraints")
        changed_from_entries, _ = _signature_entry_diff(self.signature_before, self.signature_after)
        self.qualifier_value_changes = _qualifier_value_change_rows(self.signature_before, self.signature_after)
        changed_from_qualifiers = [
            row["constraint_qid"]
            for row in self.qualifier_value_changes
            if isinstance(row.get("constraint_qid"), str)
        ]
        self.changed_constraint_qids_from_entries = sorted(set(changed_from_delta) | set(changed_from_entries))
        self.changed_constraint_qids_from_qualifier_changes = sorted(set(changed_from_qualifiers))
        self.changed_constraint_qids_all = sorted(
            set(self.changed_constraint_qids_from_entries) | set(self.changed_constraint_qids_from_qualifier_changes)
        )
        self.changed_constraint_qids = self.changed_constraint_qids_all

    def _violation_names(self) -> List[str]:
        vc = self.repair_event.get("violation_context", {})
        if not isinstance(vc, dict):
            return []
        names: List[str] = []
        for key in (
            "report_violation_type_normalized",
            "report_violation_type",
            "report_violation_type_raw",
            "report_violation_types",
            "violation_name",
            "message",
        ):
            raw = vc.get(key)
            if isinstance(raw, str) and raw.strip():
                names.append(raw.strip())
            elif isinstance(raw, list):
                names.extend([item.strip() for item in raw if isinstance(item, str) and item.strip()])
        seen: set[str] = set()
        deduped: List[str] = []
        for name in names:
            key = normalize_text(name)
            if key and key not in seen:
                seen.add(key)
                deduped.append(name)
        return deduped

    def _candidate_constraint_qids(self) -> List[str]:
        qids = []
        for entry in self.signature_before + self.signature_after:
            if not isinstance(entry, dict):
                continue
            qid = entry.get("constraint_qid")
            if isinstance(qid, str) and qid not in qids:
                qids.append(qid)
        return qids

    def _select_target_qid(
        self,
        mapping: Dict[str, Any],
        compatible_overlap: Dict[str, Any],
    ) -> Tuple[Optional[str], str, str, bool, bool]:
        mapped_qid = mapping.get("mapped_violation_constraint_qid")
        mapped_family = mapping.get("mapped_violation_family", "unknown")
        changed = set(self.changed_constraint_qids_all)
        if isinstance(mapped_qid, str):
            if mapped_qid in changed:
                return mapped_qid, "mapped_violation_constraint_changed", "high", True, False
        if compatible_overlap.get("compatible_overlap_used"):
            for row in self.qualifier_value_changes:
                qid = row.get("constraint_qid")
                if isinstance(qid, str) and qid in changed:
                    changed_family = _constraint_family(qid, _constraint_label(qid, self.signature_before + self.signature_after))
                    if changed_family == mapped_family:
                        return qid, "changed_constraint_family_has_compatible_report_overlap", "medium", True, False
                    if _families_related_for_tbox_target(mapped_family, changed_family, compatible_overlap):
                        return qid, "changed_related_constraint_has_report_overlap", "medium", True, True
        if mapped_family not in {"unknown", "label_in_language", "description_in_language"}:
            for qid in self.changed_constraint_qids_all:
                changed_family = _constraint_family(qid, _constraint_label(qid, self.signature_before + self.signature_after))
                if changed_family == mapped_family:
                    return qid, "changed_constraint_family_matches_mapped_violation", "medium", True, False
                if _families_related_for_tbox_target(mapped_family, changed_family, compatible_overlap):
                    return qid, "changed_related_constraint_family_matches_mapped_violation", "medium", True, True
        return None, "no_matching_changed_constraint", "low", False, False

    def _semantic_summary_for_family(self, mapped_family: str) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        for row in self.qualifier_value_changes:
            qid = row.get("constraint_qid")
            if not isinstance(qid, str):
                continue
            family = _constraint_family(qid, _constraint_label(qid, self.signature_before + self.signature_after))
            if family == mapped_family:
                rows.append(row)
            elif mapped_family in {"type", "value_type"} and family == "allowed_entity_types":
                rows.append(row)
            elif mapped_family in {"required_statement", "value_required_statement"} and family in {"required_statement", "value_required_statement"}:
                rows.append(row)
            elif mapped_family in {"label_in_language", "description_in_language"} and family in {"label_in_language", "description_in_language"}:
                rows.append(row)
        if not rows:
            rows = self.qualifier_value_changes
        return filter_tbox_semantic_qualifier_changes(rows, mapped_family)

    def _candidate_mapping(self, name: str) -> Dict[str, Any]:
        mapping = _map_tbox_violation(name)
        report_tokens = _extract_report_tokens_from_text(name)
        vc = self.repair_event.get("violation_context", {})
        if isinstance(vc, dict):
            qids = vc.get("report_violation_type_qids")
            if isinstance(qids, list):
                report_tokens["qids"] = sorted(set(report_tokens.get("qids", [])) | {str(qid) for qid in qids if is_qid(str(qid))})
        semantic_summary = self._semantic_summary_for_family(mapping["mapped_violation_family"])
        overlap = _overlap_detail(self.qualifier_value_changes, report_tokens)
        compatible_overlap = compatible_overlap_for_mapped_family(
            mapping["mapped_violation_family"],
            report_tokens,
            semantic_summary,
            raw_overlap=overlap,
        )
        mapped_qid = mapping["mapped_violation_constraint_qid"]
        exact_family = isinstance(mapped_qid, str) and mapped_qid in self.changed_constraint_qids_all
        compatible = bool(compatible_overlap.get("compatible_overlap_used"))
        has_concrete = report_has_concrete_arguments(
            name,
            report_tokens,
            self.repair_event.get("violation_context") if isinstance(self.repair_event.get("violation_context"), dict) else None,
        )
        if exact_family and compatible:
            score = 40
            level = "exact_constraint_and_value_match"
        elif exact_family and not has_concrete:
            score = 30
            level = "exact_constraint_family_only"
        elif exact_family:
            score = 25
            level = "exact_constraint_family_only"
        elif compatible:
            score = 20
            level = "value_or_property_overlap_only"
        elif mapping["mapped_violation_family"] == "unknown":
            score = 0
            level = "unmapped_violation"
        else:
            score = 5
            level = "constraint_family_mismatch"
        return {
            "violation_name": name,
            **mapping,
            "report_tokens": report_tokens,
            "overlap": overlap,
            "compatible_overlap": compatible_overlap,
            "semantic_change_summary": semantic_summary,
            "has_concrete_report_tokens": has_concrete,
            "candidate_score": score,
            "candidate_causality_match_level": level,
        }

    def _select_violation_candidate(self) -> Dict[str, Any]:
        names = self._violation_names()
        if not names:
            names = [""]
        candidates = [self._candidate_mapping(name) for name in names]
        candidates.sort(
            key=lambda item: (
                item["candidate_score"],
                1 if item["compatible_overlap"].get("compatible_overlap_used") else 0,
                len(item["compatible_overlap"].get("value_overlap_with_report_qids", []))
                + len(item["compatible_overlap"].get("property_overlap_with_report_pids", []))
                + len(item["compatible_overlap"].get("language_overlap_with_report_langs", []))
                + len(item["compatible_overlap"].get("scope_overlap_with_report_values", [])),
            ),
            reverse=True,
        )
        selected = dict(candidates[0])
        selected["candidate_violation_names"] = names
        selected["candidate_violation_mappings_preview"] = _preview_list(
            [
                {
                    "violation_name": candidate["violation_name"],
                    "mapped_violation_constraint_qid": candidate["mapped_violation_constraint_qid"],
                    "mapped_violation_family": candidate["mapped_violation_family"],
                    "candidate_causality_match_level": candidate["candidate_causality_match_level"],
                    "candidate_score": candidate["candidate_score"],
                }
                for candidate in candidates
            ],
            limit=12,
        )
        return selected

    def classify_change(self) -> Tuple[str, List[Dict[str, Any]], str]:
        trace: List[Dict[str, Any]] = []
        selected = self._select_violation_candidate()
        violation_name = selected["violation_name"]
        mapping = {key: selected[key] for key in (
            "mapped_violation_constraint_qid",
            "mapped_violation_constraint_label",
            "mapped_violation_family",
            "mapped_violation_confidence",
            "mapped_violation_reason",
        )}
        mapped_qid = mapping["mapped_violation_constraint_qid"]
        report_tokens = selected["report_tokens"]
        overlap = selected["overlap"]
        compatible_overlap = selected["compatible_overlap"]
        target_qid, target_reason, target_confidence, target_is_changed, target_is_related = self._select_target_qid(
            mapping,
            compatible_overlap,
        )
        target_label = _constraint_label(target_qid, self.signature_before + self.signature_after)
        exact_family = isinstance(mapped_qid, str) and mapped_qid in self.changed_constraint_qids_all
        compatible = bool(compatible_overlap.get("compatible_overlap_used"))
        has_concrete = selected["has_concrete_report_tokens"]
        if exact_family and compatible:
            causality_level = "exact_constraint_and_value_match"
        elif exact_family and has_concrete:
            causality_level = "exact_constraint_family_only_no_compatible_overlap"
        elif exact_family:
            causality_level = "exact_constraint_family_only"
        elif target_qid and compatible:
            causality_level = "related_constraint_with_value_overlap" if target_is_related else "value_or_property_overlap_only"
        elif mapping["mapped_violation_family"] == "unknown":
            causality_level = "unmapped_violation"
        elif target_qid is None:
            causality_level = "constraint_family_mismatch"
        else:
            causality_level = "unknown"

        causal = causality_level in {
            "exact_constraint_and_value_match",
            "related_constraint_with_value_overlap",
            "exact_constraint_family_only",
            "exact_constraint_family_only_no_compatible_overlap",
            "value_or_property_overlap_only",
        }
        value_specific_without_overlap = exact_family and has_concrete and not compatible
        directional_allowed = causal and not value_specific_without_overlap
        causality_detail = {
            "mapped_report_constraint_qid": mapping.get("mapped_violation_constraint_qid"),
            "mapped_report_constraint_label": mapping.get("mapped_violation_constraint_label"),
            "mapped_report_family": mapping.get("mapped_violation_family"),
            **mapping,
            "selected_violation_name": violation_name,
            "candidate_violation_names": selected["candidate_violation_names"],
            "candidate_violation_mappings_preview": selected["candidate_violation_mappings_preview"],
            "violation_name": violation_name,
            "changed_constraint_qids_from_entries": _preview_list(self.changed_constraint_qids_from_entries),
            "changed_constraint_qids_from_qualifier_changes": _preview_list(self.changed_constraint_qids_from_qualifier_changes),
            "changed_constraint_qids_all": _preview_list(self.changed_constraint_qids_all),
            "target_constraint_qid": target_qid,
            "target_constraint_label": target_label,
            "target_constraint_selection_reason": target_reason,
            "target_constraint_selection_confidence": target_confidence,
            "target_constraint_is_changed": target_is_changed,
            "target_constraint_is_related_family": target_is_related,
            "causality_match_level": causality_level,
            "causality_match_reason": (
                "mapped constraint family changed, but no compatible changed value/property/language overlaps the concrete report"
                if value_specific_without_overlap
                else
                "mapped constraint family and compatible changed values support the violation report"
                if causal
                else "changed constraints do not establish a causal link to the reported violation"
            ),
            "value_specific_without_overlap": value_specific_without_overlap,
            **overlap,
            "compatible_value_overlap_with_report_qids": compatible_overlap.get("value_overlap_with_report_qids", []),
            "compatible_property_overlap_with_report_pids": compatible_overlap.get("property_overlap_with_report_pids", []),
            "compatible_language_overlap_with_report_langs": compatible_overlap.get("language_overlap_with_report_langs", []),
            "compatible_scope_overlap_with_report_values": compatible_overlap.get("scope_overlap_with_report_values", []),
            "compatible_overlap_used": compatible_overlap.get("compatible_overlap_used", False),
            "compatible_overlap_reason": compatible_overlap.get("compatible_overlap_reason", ""),
            "incompatible_overlap_ignored": compatible_overlap.get("incompatible_overlap_ignored", {}),
        }
        trace.append(
            {
                "step": "causality_filter",
                "result": causal,
                "violation_name": violation_name,
                "mapped_violation_constraint_qid": mapped_qid,
                "causality_match_level": causality_level,
            }
        )
        trace.append(
            {
                "step": "target_constraint",
                "result": target_qid,
                "target_constraint_selection_reason": target_reason,
                "target_constraint_selection_confidence": target_confidence,
            }
        )
        if not causal:
            subtype = "UNKNOWN_TBOX_CAUSALITY" if causality_level in {"unmapped_violation", "unknown"} else "COINCIDENTAL_SCHEMA_CHANGE"
            trace.append({"step": "tbox_causality", "result": subtype, **causality_detail})
            return (
                subtype,
                trace,
                "The property revision changed constraints, but the changed constraint family or changed qualifier values do not establish a causal link to the reported violation type.",
            )
        if not target_qid:
            trace.append({"step": "tbox_causality", "result": "UNKNOWN_TBOX_CAUSALITY", **causality_detail})
            return "UNKNOWN_TBOX_CAUSALITY", trace, "No changed target constraint could be selected for the reported violation."

        target_family = _constraint_family(target_qid, target_label)
        target_change_rows = [
            row for row in self.qualifier_value_changes if row.get("constraint_qid") == target_qid
        ]
        semantic_summary = filter_tbox_semantic_qualifier_changes(
            target_change_rows,
            target_qid,
            target_constraint_qid=target_qid,
        )
        causality_detail.update(
            {
                key: value
                for key, value in semantic_summary.items()
                if key not in {"semantic_change_rows", "ignored_change_rows"}
            }
        )
        if (
            value_specific_without_overlap
            and semantic_summary.get("semantic_added_value_count", 0) == 0
            and semantic_summary.get("semantic_removed_value_count", 0) == 0
        ):
            trace.append({"step": "tbox_causality", "result": "UNKNOWN_TBOX_CAUSALITY", **causality_detail})
            return (
                "UNKNOWN_TBOX_CAUSALITY",
                trace,
                "Mapped constraint family changed, but concrete report arguments do not overlap any semantic qualifier change.",
            )

        old_qualifiers = _collect_qualifiers_for_qid(self.signature_before, target_qid)
        new_qualifiers = _collect_qualifiers_for_qid(self.signature_after, target_qid)

        if target_qid in RANGE_QIDS:
            result = analyze_range_change(old_qualifiers, new_qualifiers)
            if value_specific_without_overlap:
                result = "SCHEMA_UPDATE"
            trace.append({"step": "range_semantics", "result": result})
            trace.append(
                {
                    "step": "tbox_causality",
                    "result": result,
                    "set_semantics": "range",
                    "polarity": "relaxation" if result == "RELAXATION_RANGE_WIDENED" else "restriction" if result == "RESTRICTION_RANGE_NARROWED" else "unknown",
                    "polarity_basis": "numeric/date range bounds",
                    "directional_subtype_basis": result.lower(),
                    **causality_detail,
                }
            )
            rationale = "Range constraint qualifiers compared using numeric and date bounds."
            return result, trace, rationale

        polarity_detail = _polarity_from_delta(
            qid=target_qid,
            label=target_label,
            added_values=semantic_summary["semantic_added_values"],
            removed_values=semantic_summary["semantic_removed_values"],
        )
        semantic_has_values = bool(semantic_summary["semantic_added_values"] or semantic_summary["semantic_removed_values"])
        result = polarity_detail["subtype"] if directional_allowed and semantic_has_values else "SCHEMA_UPDATE"
        active_direction_detail, potential_direction_detail = _active_and_potential_tbox_direction_details(
            result,
            polarity_detail,
        )
        trace.append(
            {
                "step": "set_semantics",
                "result": result,
                "property_ids": semantic_summary.get("semantic_changed_qualifier_properties"),
                "added_value_count": len(semantic_summary["semantic_added_values"]),
                "removed_value_count": len(semantic_summary["semantic_removed_values"]),
                **active_direction_detail,
                **potential_direction_detail,
            }
        )
        trace.append(
            {
                "step": "tbox_causality",
                "result": result,
                "added_values": _preview_list(semantic_summary["semantic_added_values"]),
                "removed_values": _preview_list(semantic_summary["semantic_removed_values"]),
                "added_value_count": len(semantic_summary["semantic_added_values"]),
                "removed_value_count": len(semantic_summary["semantic_removed_values"]),
                "changed_qualifier_properties": semantic_summary.get("semantic_changed_qualifier_properties", []),
                **active_direction_detail,
                **potential_direction_detail,
                **causality_detail,
            }
        )
        rationale = (
            "Constraint family matched the violation and qualifier-value polarity was interpreted for the target constraint."
            if result != "SCHEMA_UPDATE"
            else "Constraint family matched the violation, but polarity could not be interpreted directionally."
        )
        return result, trace, rationale


def _compact_tbox_diff_summary(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    step = next((item for item in reversed(trace) if item.get("step") == "tbox_causality"), {})
    if not isinstance(step, dict):
        return {}
    return {
        "lean_stage4_pruned_full_signatures": True,
        "source": "classification.decision_trace.tbox_causality",
        "selected_violation_name": step.get("selected_violation_name"),
        "target_constraint_qid": step.get("target_constraint_qid"),
        "target_constraint_label": step.get("target_constraint_label"),
        "target_constraint_selection_reason": step.get("target_constraint_selection_reason"),
        "mapped_report_constraint_qid": step.get("mapped_report_constraint_qid"),
        "mapped_report_constraint_label": step.get("mapped_report_constraint_label"),
        "mapped_report_family": step.get("mapped_report_family"),
        "target_constraint_is_changed": step.get("target_constraint_is_changed"),
        "target_constraint_is_related_family": step.get("target_constraint_is_related_family"),
        "changed_constraint_qids_from_entries": step.get("changed_constraint_qids_from_entries"),
        "changed_constraint_qids_from_qualifier_changes": step.get("changed_constraint_qids_from_qualifier_changes"),
        "changed_constraint_qids_all": step.get("changed_constraint_qids_all"),
        "changed_qualifier_properties": step.get("changed_qualifier_properties"),
        "added_values": step.get("added_values"),
        "removed_values": step.get("removed_values"),
        "added_value_count": step.get("added_value_count"),
        "removed_value_count": step.get("removed_value_count"),
        "set_semantics": step.get("set_semantics"),
        "set_operation": step.get("set_operation"),
        "polarity": step.get("polarity"),
        "polarity_basis": step.get("polarity_basis"),
        "directional_subtype_precise": step.get("directional_subtype_precise"),
        "analysis_slice_precise": step.get("analysis_slice_precise"),
        "potential_set_semantics": step.get("potential_set_semantics"),
        "potential_set_operation": step.get("potential_set_operation"),
        "potential_polarity": step.get("potential_polarity"),
        "potential_polarity_basis": step.get("potential_polarity_basis"),
        "potential_directional_subtype_basis": step.get("potential_directional_subtype_basis"),
        "potential_directional_subtype_precise": step.get("potential_directional_subtype_precise"),
        "semantic_changed_qualifier_properties": step.get("semantic_changed_qualifier_properties"),
        "ignored_changed_qualifier_properties": step.get("ignored_changed_qualifier_properties"),
        "semantic_added_values": step.get("semantic_added_values"),
        "semantic_removed_values": step.get("semantic_removed_values"),
        "ignored_added_values": step.get("ignored_added_values"),
        "ignored_removed_values": step.get("ignored_removed_values"),
        "ignored_value_count": step.get("ignored_value_count"),
        "qualifier_filter_reason": step.get("qualifier_filter_reason"),
        "compatible_overlap_used": step.get("compatible_overlap_used"),
        "compatible_overlap_reason": step.get("compatible_overlap_reason"),
        "incompatible_overlap_ignored": step.get("incompatible_overlap_ignored"),
        "value_specific_without_overlap": step.get("value_specific_without_overlap"),
    }


def _parse_numeric_token(token: str) -> Optional[Decimal]:
    if not isinstance(token, str):
        return None
    token = token.strip()
    if not token:
        return None
    if DATE_ISO_RE.match(token.lstrip("+")):
        return None

    # Handle Wikidata time strings like "+2025-01-01T00:00:00Z"
    if token.startswith("+") or "T" in token:
        clean_ts = token.lstrip("+").rstrip("Z")
        try:
            dt = _dt.datetime.fromisoformat(clean_ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_dt.UTC)
            return Decimal(str(dt.timestamp()))
        except ValueError:
            pass

    # Handle quantities with units like "10 km"
    match = re.match(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", token)
    if match:
        try:
            return Decimal(match.group(0))
        except Exception:
            return None

    return None


def _parse_date_boundary(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    m = re.search(r"[+-]?\d{4,}-\d{2}-\d{2}", raw)
    if not m:
        return None
    out = m.group(0)
    if out.startswith("+"):
        out = out[1:]
    return out


def _extract_allowed_values(constraint: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    qualifiers = constraint.get("qualifiers")
    if not isinstance(qualifiers, list):
        return out
    for q in qualifiers:
        if not isinstance(q, dict):
            continue
        pid = q.get("property_id")
        if pid != ONE_OF_VALUE_QUALIFIER:
            continue
        values = q.get("values")
        if not isinstance(values, list):
            continue
        for v in values:
            if isinstance(v, dict):
                if isinstance(v.get("qid"), str):
                    out.append(v["qid"])
                elif isinstance(v.get("raw"), str):
                    out.append(v["raw"])
            elif isinstance(v, str):
                out.append(v)
    # de-dup
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _extract_range_bounds(constraint: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], List[str]]:
    numeric_mins: List[str] = []
    numeric_maxs: List[str] = []
    date_mins: List[str] = []
    date_maxs: List[str] = []
    qualifiers = constraint.get("qualifiers")
    if not isinstance(qualifiers, list):
        return numeric_mins, numeric_maxs, date_mins, date_maxs
    for q in qualifiers:
        if not isinstance(q, dict):
            continue
        pid = q.get("property_id")
        if pid not in {NUMERIC_MIN_QUALIFIER, NUMERIC_MAX_QUALIFIER, DATE_MIN_QUALIFIER, DATE_MAX_QUALIFIER}:
            continue
        values = q.get("values")
        if not isinstance(values, list):
            continue
        for v in values:
            raw = None
            if isinstance(v, dict):
                raw = v.get("raw")
            elif isinstance(v, str):
                raw = v
            if not isinstance(raw, str):
                continue
            if pid == NUMERIC_MIN_QUALIFIER:
                numeric_mins.append(raw)
            elif pid == NUMERIC_MAX_QUALIFIER:
                numeric_maxs.append(raw)
            elif pid == DATE_MIN_QUALIFIER:
                date_mins.append(raw)
            else:
                date_maxs.append(raw)
    return numeric_mins, numeric_maxs, date_mins, date_maxs


def flatten_truth(value: Any) -> List[str]:
    """
    Flattens various shapes into a list of atomic string tokens.
    We keep strings as-is; numbers become strings; dicts -> try common keys.
    Order is deterministic.
    """
    out: List[str] = []

    def rec(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            out.append(v)
            return
        if isinstance(v, (int, float)):
            out.append(str(v))
            return
        if isinstance(v, list):
            for x in v:
                rec(x)
            return
        if isinstance(v, dict):
            # common shapes in your pipeline: {"qid": "..."} or {"raw": "..."}
            if "qid" in v and isinstance(v["qid"], str):
                out.append(v["qid"])
                return
            if "id" in v and isinstance(v["id"], str):
                out.append(v["id"])
                return
            if "entity_id" in v and isinstance(v["entity_id"], str):
                out.append(v["entity_id"])
                return
            if "raw" in v and isinstance(v["raw"], str):
                out.append(v["raw"])
                return
            if "value" in v:
                if isinstance(v["value"], (str, int, float)):
                    out.append(str(v["value"]))
                    return
                rec(v["value"])
                return
            # otherwise, recurse values in key order for determinism
            for k in sorted(v.keys()):
                rec(v[k])
            return

        # unknown type: stringified
        out.append(str(v))

    rec(value)
    # de-dup while keeping order
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def get_truth_info(repair_event: Dict[str, Any]) -> Tuple[List[str], str, bool]:
    """
    Returns (truth_tokens, truth_source, truth_applicable).
    For A-BOX UPDATE/CREATE: truth is the historical post-repair value.
    Current snapshot fields are returned only as a diagnostic fallback source so
    the classifier can quarantine them instead of using them as ordinary truth.
    For DELETE and T-BOX: truth is not applicable.
    """
    rt = repair_event.get("repair_target", {})
    if not isinstance(rt, dict):
        return [], "missing_unexpected", True

    kind = rt.get("kind")
    if kind != "A_BOX":
        return [], "none_expected", False

    action = rt.get("action")
    if action == "DELETE":
        return [], "none_expected", False

    historical_candidates = [
        ("repair_target.new_value", rt.get("new_value")),
        ("repair_target.value", rt.get("value")),
    ]
    for src, v in historical_candidates:
        toks = flatten_truth(v)
        if toks:
            return toks, src, True

    diagnostic_candidates = [
        ("persistence_check.current_value_2026", safe_get(repair_event, "persistence_check", "current_value_2026")),
        ("violation_context.value_current_2026", safe_get(repair_event, "violation_context", "value_current_2026")),
        ("persistence_check.current_value_2025", safe_get(repair_event, "persistence_check", "current_value_2025")),
        ("violation_context.value_current_2025", safe_get(repair_event, "violation_context", "value_current_2025")),
    ]
    for src, v in diagnostic_candidates:
        toks = flatten_truth(v)
        if toks:
            return toks, src, True
    return [], "missing_unexpected", True


def _pre_repair_value(repair_event: Dict[str, Any]) -> Tuple[Any, str]:
    return pre_repair_target_raw_value(repair_event)


def _label_entry_for_id(world_state_entry: Dict[str, Any], entity_id: str) -> Dict[str, Any]:
    entities = safe_get(world_state_entry, "L2_labels", "entities", default={})
    if isinstance(entities, dict):
        entry = entities.get(entity_id)
        if isinstance(entry, dict):
            return entry
    return {}


def local_context_buckets(
    repair_event: Dict[str, Any],
    world_state_entry: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds separate buckets for local matching provenance. Synthetic pre-repair target-property
    values are intentionally kept separate from independent focus text.
    """
    ids_neighbors: set = set()
    ids_focus_prerepair: set = set()
    ids_focus_non_target: set = set()
    locally_referenced_ids: set = set()
    ids_focus_qid: set = set()
    text_focus_label: List[str] = []
    text_focus_description: List[str] = []
    text_focus_alias: List[str] = []
    text_focus_prerepair_literals: List[str] = []
    text_neighbor_label: List[str] = []
    text_neighbor_description: List[str] = []
    text_focus_properties: List[str] = []
    text_focus_property_entries: List[Dict[str, Any]] = []
    text_l2_referenced: List[str] = []
    synth_info = {"used_pre_repair_value": False, "pre_repair_source": "missing", "tokens": []}
    target_pid = repair_event.get("property") if is_pid(repair_event.get("property")) else None

    # Focus node text
    q_label = repair_event.get("qid_label_en") or safe_get(world_state_entry, "L1_ego_node", "label")
    q_desc = repair_event.get("qid_description_en") or safe_get(world_state_entry, "L1_ego_node", "description")
    focus_qid = repair_event.get("qid") or safe_get(world_state_entry, "L1_ego_node", "qid")
    if is_qid(focus_qid):
        ids_focus_qid.add(focus_qid)
    if isinstance(q_label, str) and q_label.strip():
        text_focus_label.append(q_label)
    if isinstance(q_desc, str) and q_desc.strip():
        text_focus_description.append(q_desc)
    aliases = safe_get(world_state_entry, "L1_ego_node", "aliases", default=[])
    if isinstance(aliases, list):
        text_focus_alias.extend([alias for alias in aliases if isinstance(alias, str) and alias.strip()])

    # Focus-node non-target properties. Target property values are intentionally excluded because
    # the frozen world state may contain the historical post-repair answer.
    l1_properties = safe_get(world_state_entry, "L1_ego_node", "properties", default={})
    if isinstance(l1_properties, dict):
        for pid, values in l1_properties.items():
            if target_pid and pid == target_pid:
                continue
            for tok in flatten_truth(values):
                if is_qid(tok) or is_pid(tok):
                    ids_focus_non_target.add(tok)
                    locally_referenced_ids.add(tok)
                elif tok not in (None, ""):
                    tok_s = str(tok)
                    text_focus_properties.append(tok_s)
                    text_focus_property_entries.append(
                        {
                            "text": tok_s,
                            "supporting_property_id": pid,
                            "supporting_property_label": None,
                            "supporting_value": tok_s,
                        }
                    )

    # Neighborhood
    edges = safe_get(world_state_entry, "L3_neighborhood", "outgoing_edges", default=[])
    if isinstance(edges, list):
        for e in edges:
            if not isinstance(e, dict):
                continue
            pid = e.get("property_id")
            if target_pid and pid == target_pid:
                # Skip post-repair values for the target property; add synthetic pre-repair values instead.
                continue
            tq = e.get("target_qid")
            if is_qid(tq):
                ids_neighbors.add(tq)
                locally_referenced_ids.add(tq)
            tl = e.get("target_label")
            td = e.get("target_description")
            if isinstance(tl, str) and tl.strip():
                text_neighbor_label.append(tl)
            if isinstance(td, str) and td.strip():
                text_neighbor_description.append(td)

    # Synthetic pre-repair values for the target property (A-Box only)
    rt = repair_event.get("repair_target", {}) if isinstance(repair_event.get("repair_target"), dict) else {}
    if rt.get("kind") == "A_BOX" and rt.get("action") != "DELETE":
        pre_val, pre_src = _pre_repair_value(repair_event)
        pre_tokens = flatten_truth(pre_val)
        if pre_tokens:
            synth_info["used_pre_repair_value"] = True
            synth_info["pre_repair_source"] = pre_src
            synth_info["tokens"] = pre_tokens
            for tok in pre_tokens:
                if is_qid(tok):
                    ids_focus_prerepair.add(tok)
                    locally_referenced_ids.add(tok)
                else:
                    text_focus_prerepair_literals.append(str(tok))
        else:
            synth_info["pre_repair_source"] = pre_src
    else:
        synth_info["pre_repair_source"] = "not_applicable"

    for entity_id in sorted(locally_referenced_ids):
        entry = _label_entry_for_id(world_state_entry, entity_id)
        for key in ("label", "description", "label_en", "description_en"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                text_l2_referenced.append(value)

    buckets = {
        "ids_neighbors": ids_neighbors,
        "ids_focus_qid": ids_focus_qid,
        "ids_focus_prerepair": ids_focus_prerepair,
        "ids_focus_non_target": ids_focus_non_target,
        "text_focus_label_fields": text_focus_label,
        "text_focus_description_fields": text_focus_description,
        "text_focus_alias_fields": text_focus_alias,
        "text_focus_prerepair_literal_fields": text_focus_prerepair_literals,
        "text_neighbor_label_fields": text_neighbor_label,
        "text_neighbor_description_fields": text_neighbor_description,
        "text_focus_property_fields": text_focus_properties,
        "text_focus_property_entries": text_focus_property_entries,
        "text_l2_referenced_fields": text_l2_referenced,
        # Legacy aggregate fields remain for older tests/callers.
        "text_focus": normalize_text(" \n ".join(text_focus_label + text_focus_description + text_focus_alias + text_focus_properties)),
        "text_neighbors": normalize_text(" \n ".join(text_neighbor_label + text_neighbor_description + text_l2_referenced)),
    }
    return buckets, synth_info


def report_type_normalized(repair_event: Dict[str, Any]) -> str:
    vc = repair_event.get("violation_context", {})
    if not isinstance(vc, dict):
        return ""
    t = vc.get("report_violation_type_normalized") or vc.get("report_violation_type") or ""
    if not isinstance(t, str):
        return ""
    return normalize_text(t)


def _violation_report_texts(repair_event: Dict[str, Any]) -> List[str]:
    vc = repair_event.get("violation_context", {})
    if not isinstance(vc, dict):
        return []
    values: List[str] = []
    for key in (
        "report_violation_type_normalized",
        "report_violation_type",
        "report_violation_types",
        "violation_name",
        "message",
    ):
        raw = vc.get(key)
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, list):
            values.extend([item for item in raw if isinstance(item, str)])
    return values


def _is_format_report(repair_event: Dict[str, Any]) -> bool:
    return report_type_normalized(repair_event) == "format"


def _is_self_link_report(repair_event: Dict[str, Any]) -> bool:
    return report_type_normalized(repair_event) == "self link"


def _is_selection_report(repair_event: Dict[str, Any]) -> bool:
    return report_type_normalized(repair_event) in {"single value", "unique value"}


def _is_single_value_report(repair_event: Dict[str, Any]) -> bool:
    return any(normalize_text(text) in {"single value", "single-value", "single value constraint"} for text in _violation_report_texts(repair_event))


def _is_set_membership_report(repair_event: Dict[str, Any]) -> bool:
    return report_type_normalized(repair_event) in {"one of", "none of"}


def _is_cardinality_or_duplicate_report(repair_event: Dict[str, Any], constraint_types: List[Dict[str, Optional[str]]]) -> bool:
    del constraint_types
    report = report_type_normalized(repair_event)
    if report in {
        "single value",
        "single-value",
        "single value constraint",
        "unique value",
        "unique-value",
        "unique value constraint",
        "distinct values",
        "distinct-values",
        "distinct values constraint",
        "duplicate value",
        "duplicate values",
    }:
        return True
    return any(term in report for term in ("duplicate", "cardinality"))


def _target_required_claim_pid(repair_event: Dict[str, Any]) -> Optional[str]:
    vc = repair_event.get("violation_context", {})
    if not isinstance(vc, dict):
        return None
    candidates = [
        vc.get("report_violation_type_normalized"),
        vc.get("report_violation_type"),
        vc.get("violation_name"),
        vc.get("message"),
    ]
    for value in candidates:
        if not isinstance(value, str):
            continue
        if "target required claim" not in normalize_text(value):
            continue
        match = re.search(r"\bP\|?(\d+)\b", value, flags=re.IGNORECASE)
        if match:
            return f"P{match.group(1)}"
        return "UNKNOWN"
    return None


def _classification_target_tokens(value_change: Any) -> Dict[str, Any]:
    action = value_change.semantic_action
    if action in {"CREATE_FROM_MISSING", "ADD_SUPERSET"}:
        return {
            "role": "added",
            "tokens": list(value_change.added_unique_values or value_change.new_unique),
            "reason": "created or added values are the changed repair target",
        }
    if action == "DELETE_SUBSET":
        return {
            "role": "removed",
            "tokens": list(value_change.removed_unique_values),
            "reason": "subset deletion is explained by removed values, not retained values",
        }
    if action == "DELETE_TO_MISSING":
        return {
            "role": "removed",
            "tokens": list(value_change.removed_unique_values or value_change.old_unique),
            "reason": "delete-to-missing is explained by deleted old values",
        }
    if action == "REPLACE_1_TO_1":
        return {
            "role": "replacement_new",
            "tokens": list(value_change.new_unique),
            "reason": "one-to-one replacement is classified from the replacement relation",
        }
    if action == "MIXED_UPDATE":
        if len(value_change.added_unique_values) == 1 and len(value_change.removed_unique_values) == 1:
            return {
                "role": "changed_pair",
                "tokens": list(value_change.removed_unique_values + value_change.added_unique_values),
                "old_changed_value": value_change.removed_unique_values[0],
                "new_changed_value": value_change.added_unique_values[0],
                "retained_values": list(value_change.retained_unique_values),
                "reason": "mixed update classification uses the deterministic changed pair while ignoring retained values",
            }
        return {
            "role": "changed",
            "tokens": list(value_change.added_unique_values + value_change.removed_unique_values),
            "reason": "mixed update classification uses added and removed changed values only",
        }
    if action.startswith("MULTIPLICITY_"):
        return {
            "role": "multiplicity",
            "tokens": list(value_change.old_unique or value_change.new_unique),
            "reason": "unique values are unchanged; only multiplicity changed",
        }
    return {"role": "none", "tokens": [], "reason": "no changed semantic value tokens"}


def _independent_local_text_fields(
    repair_event: Dict[str, Any],
    world_state_entry: Dict[str, Any],
) -> List[Tuple[str, str]]:
    buckets, _ = local_context_buckets(repair_event, world_state_entry)
    fields: List[Tuple[str, str]] = []
    source_keys = [
        ("FOCUS_LABEL", "text_focus_label_fields"),
        ("FOCUS_DESCRIPTION", "text_focus_description_fields"),
        ("FOCUS_ALIAS", "text_focus_alias_fields"),
        ("FOCUS_NON_TARGET_PROPERTY_TEXT", "text_focus_property_fields"),
        ("NEIGHBOR_LABEL", "text_neighbor_label_fields"),
        ("NEIGHBOR_DESCRIPTION", "text_neighbor_description_fields"),
    ]
    for source, key in source_keys:
        value = buckets.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    fields.append((source, item))
    return fields


def _p8726_local_text_derived_detail(
    repair_event: Dict[str, Any],
    world_state_entry: Dict[str, Any],
    value_change: Any,
) -> Optional[Dict[str, Any]]:
    if repair_event.get("property") != "P8726":
        return None
    target = _classification_target_tokens(value_change)
    target_values = [str(token) for token in target.get("tokens", []) if token not in (None, "")]
    if len(target_values) != 1:
        return None
    target_value = target_values[0]
    match = re.fullmatch(r"(?P<year>\d{4})/si/(?P<number>\d+)/made", target_value, flags=re.IGNORECASE)
    if not match:
        return None
    year = match.group("year")
    number = str(int(match.group("number")))
    pattern = re.compile(
        rf"(?:\bS\.?\s*I\.?\s*(?:No\.?)?\s*{re.escape(number)}\s*/\s*{re.escape(year)}\b|"
        rf"\bStatutory\s+Instrument\b.{0,80}\b{re.escape(number)}\s*/\s*{re.escape(year)}\b)",
        flags=re.IGNORECASE,
    )
    for source, text in _independent_local_text_fields(repair_event, world_state_entry):
        m = pattern.search(text)
        if not m:
            continue
        return {
            "source": source,
            "raw_matched_text": m.group(0),
            "source_text": text,
            "extracted_year": year,
            "extracted_number": number,
            "derived_target": f"{year}/si/{number}/made",
            "actual_target": target_value,
            "independent_of_target_property": True,
            "derivation_rule": "p8726_statutory_instrument_id",
            "classification_target_role": target.get("role"),
        }
    return None


def match_truth_locally(truth_tokens: List[str], buckets: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns:
      - matched: bool
      - evidence: dict with counts and examples
    Matching policy:
      - QIDs/PIDs: exact id match only in locally referenced id buckets
      - ISO dates: exact field or token-boundary match at full-date precision
      - short literals (<4 chars): exact normalized field equality only
      - longer literals: exact field equality or token-boundary match
    """
    if not truth_tokens:
        return False, {"matched": False, "needed": 0, "found": 0, "matches": []}

    needed = 0
    found = 0
    matches: List[Dict[str, Any]] = []
    used_literal_substring = False
    sources_used: set = set()

    id_sources = [
        ("FOCUS_QID", buckets.get("ids_focus_qid", set()), True),
        ("FOCUS_PREREPAIR_TARGET_PROPERTY_QID", buckets.get("ids_focus_prerepair", set()), False),
        ("FOCUS_NON_TARGET_PROPERTY", buckets.get("ids_focus_non_target", set()), True),
        ("NEIGHBOR_ID", buckets.get("ids_neighbors", set()), True),
    ]

    def fields_for(source_key: str, legacy_key: str = "") -> List[Tuple[str, Dict[str, Any]]]:
        fields = buckets.get(source_key)
        if isinstance(fields, list):
            return [(str(field), {}) for field in fields if str(field).strip()]
        if not legacy_key:
            return []
        legacy = buckets.get(legacy_key, "")
        if isinstance(legacy, str) and legacy.strip():
            return [(legacy, {})]
        return []

    def focus_property_fields() -> List[Tuple[str, Dict[str, Any]]]:
        entries = buckets.get("text_focus_property_entries")
        if isinstance(entries, list):
            out: List[Tuple[str, Dict[str, Any]]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                text = str(entry.get("text") or "")
                if not text.strip():
                    continue
                out.append(
                    (
                        text,
                        {
                            "supporting_property_id": entry.get("supporting_property_id"),
                            "supporting_property_label": entry.get("supporting_property_label"),
                            "supporting_value": entry.get("supporting_value"),
                        },
                    )
                )
            return out
        return fields_for("text_focus_property_fields")

    text_sources = [
        ("FOCUS_LABEL", fields_for("text_focus_label_fields", "text_focus"), True),
        ("FOCUS_DESCRIPTION", fields_for("text_focus_description_fields"), True),
        ("FOCUS_ALIAS", fields_for("text_focus_alias_fields"), True),
        ("FOCUS_NON_TARGET_PROPERTY_TEXT", focus_property_fields(), True),
        ("FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL", fields_for("text_focus_prerepair_literal_fields"), False),
        ("NEIGHBOR_LABEL", fields_for("text_neighbor_label_fields", "text_neighbors"), True),
        ("NEIGHBOR_DESCRIPTION", fields_for("text_neighbor_description_fields"), True),
        ("NEIGHBOR_PROPERTY_TEXT", fields_for("text_l2_referenced_fields"), True),
    ]

    def boundary_match(token_norm: str, field_norm: str) -> bool:
        if not token_norm:
            return False
        return re.search(rf"(?<![0-9a-z]){re.escape(token_norm)}(?![0-9a-z])", field_norm) is not None

    def match_text_token(token: str, *, is_date: bool) -> Optional[Dict[str, Any]]:
        tok_n = normalize_text(token)
        if not tok_n:
            return None
        for source, fields, independent in text_sources:
            for raw_field, metadata in fields:
                field_n = normalize_text(raw_field)
                if not field_n:
                    continue
                if field_n == tok_n:
                    kind = "date_exact" if is_date else ("literal_exact_raw" if raw_field.strip() == token else "literal_normalized_exact")
                    match = {
                        "token": token,
                        "kind": kind,
                        "source": source,
                        "independent_of_target_property": independent,
                        "raw_match_text": raw_field,
                        "normalized_match_text": field_n,
                    }
                    match.update({key: value for key, value in metadata.items() if value not in (None, "")})
                    return match
                if is_date and boundary_match(tok_n, field_n):
                    match = {
                        "token": token,
                        "kind": "date_boundary",
                        "source": source,
                        "independent_of_target_property": independent,
                        "raw_match_text": raw_field,
                        "normalized_match_text": field_n,
                    }
                    match.update({key: value for key, value in metadata.items() if value not in (None, "")})
                    return match
                if not is_date and len(tok_n) >= 4 and boundary_match(tok_n, field_n):
                    match = {
                        "token": token,
                        "kind": "literal_boundary",
                        "source": source,
                        "independent_of_target_property": independent,
                        "raw_match_text": raw_field,
                        "normalized_match_text": field_n,
                    }
                    match.update({key: value for key, value in metadata.items() if value not in (None, "")})
                    return match
        return None

    for tok in truth_tokens:
        if tok is None:
            continue
        if isinstance(tok, str):
            tok_s = tok.strip()
        else:
            tok_s = str(tok).strip()

        if not tok_s:
            continue

        needed += 1

        if is_qid(tok_s) or is_pid(tok_s):
            for source, values, independent in id_sources:
                if tok_s in values:
                    found += 1
                    matches.append(
                        {
                            "token": tok_s,
                            "kind": "id_exact",
                            "source": source,
                            "independent_of_target_property": independent,
                        }
                    )
                    sources_used.add(source)
                    break
            continue

        text_match = match_text_token(tok_s, is_date=bool(DATE_ISO_RE.match(tok_s) or WIKIDATA_DATE_RE.match(tok_s)))
        if text_match:
            found += 1
            if text_match["kind"] in {"literal_boundary", "date_boundary"}:
                used_literal_substring = True
            matches.append(text_match)
            sources_used.add(text_match["source"])

    matched = needed > 0 and found == needed
    evidence = {
        "matched": matched,
        "needed": needed,
        "found": found,
        "matches": matches,
        "used_literal_substring": used_literal_substring,
        "sources_used": sorted(sources_used),
        "local_ids_count": sum(len(values) for _, values, _ in id_sources),
        "independent_match_count": sum(
            1 for match in matches if isinstance(match, dict) and match.get("independent_of_target_property")
        ),
    }
    return matched, evidence


def local_match_subtype(matches: List[Dict[str, Any]]) -> Optional[str]:
    sources = {m.get("source") for m in matches if isinstance(m, dict)}
    sources.discard(None)
    if not sources:
        return None
    if len(sources) > 1:
        return "LOCAL_MIXED"
    src = next(iter(sources))
    if src == "FOCUS_QID":
        return "LOCAL_FOCUS_QID"
    if src == "NEIGHBOR_ID":
        return "LOCAL_NEIGHBOR_IDS"
    if src == "FOCUS_PREREPAIR_TARGET_PROPERTY_QID":
        return "LOCAL_FOCUS_PREREPAIR_PROPERTY"
    if src == "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL":
        return "LOCAL_TEXT"
    if src == "FOCUS_NON_TARGET_PROPERTY":
        return "LOCAL_FOCUS_NON_TARGET_PROPERTY"
    if src == "FOCUS_NON_TARGET_PROPERTY_TEXT":
        return "LOCAL_FOCUS_NON_TARGET_PROPERTY"
    if src == "NEIGHBOR_PROPERTY_TEXT":
        return "LOCAL_L2_REFERENCED_TEXT"
    if src in {"FOCUS_LABEL", "FOCUS_DESCRIPTION", "FOCUS_ALIAS", "NEIGHBOR_LABEL", "NEIGHBOR_DESCRIPTION"}:
        return "LOCAL_TEXT_CONFIRMED"
    return "LOCAL_MIXED"


def evidence_has_independent_support(evidence: Dict[str, Any]) -> bool:
    matches = evidence.get("matches")
    return isinstance(matches, list) and any(
        isinstance(match, dict) and bool(match.get("independent_of_target_property")) for match in matches
    )


def evidence_only_prerepair_target_property(evidence: Dict[str, Any]) -> bool:
    matches = evidence.get("matches")
    if not isinstance(matches, list) or not matches:
        return False
    return all(
        isinstance(match, dict)
        and match.get("source") in {"FOCUS_PREREPAIR_TARGET_PROPERTY_QID", "FOCUS_PREREPAIR_TARGET_PROPERTY_LITERAL"}
        for match in matches
    )


def _collapse_internal_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _is_simple_format_normalization(old_value: str, new_value: str) -> Tuple[bool, str]:
    old_s = str(old_value)
    new_s = str(new_value)
    if old_s == new_s:
        return False, "unchanged"
    if old_s.strip() == new_s:
        return True, "strip_whitespace"
    if _collapse_internal_whitespace(old_s) == new_s:
        return True, "collapse_whitespace"
    if old_s.strip().rstrip(".,;:") == new_s:
        return True, "strip_trailing_punctuation"
    if old_s.strip().rstrip("/\\") == new_s:
        return True, "strip_trailing_slash"
    if new_s.strip().rstrip("/\\") == old_s.strip():
        return True, "append_trailing_slash"
    if old_s.strip().startswith("Category:") and old_s.strip()[len("Category:") :] == new_s:
        return True, "strip_category_prefix"
    url_match = re.match(r"^https?://[^/]+/(?:[^?#]*/)?([^/?#]+)/?[/]?(?:[?#].*)?$", old_s.strip(), flags=re.I)
    if url_match and url_match.group(1) == new_s:
        return True, "extract_url_slug"
    if re.fullmatch(r"SCHEMBL\d+", old_s.strip(), flags=re.IGNORECASE) and new_s == re.sub(
        r"(?i)^SCHEMBL", "", old_s.strip()
    ):
        return True, "strip_schembl_prefix"
    if re.fullmatch(r"[A-Za-z]+[0-9]+", old_s.strip()) and re.fullmatch(r"[0-9]+", new_s):
        prefix_stripped = re.sub(r"^[A-Za-z]+", "", old_s.strip())
        if prefix_stripped == new_s and len(new_s) >= 3:
            return True, "strip_alpha_prefix"
    if old_s.strip().lower() == new_s.lower() and old_s.strip() != new_s:
        return True, "normalize_case"
    old_date = _parse_date_boundary(old_s)
    if old_date and old_date == new_s:
        return True, "normalize_date_literal"
    return False, "non_deterministic_format_update"


def _format_pair_detail(old_value: str, new_value: str, regexes: List[str]) -> Tuple[str, Dict[str, Any]]:
    if is_qid(old_value) or is_qid(new_value) or is_pid(old_value) or is_pid(new_value):
        return "not_format_literals", {
            "reason": "format_normalization_requires_literals",
            "old_value": old_value,
            "new_value": new_value,
        }
    simple, kind = _is_simple_format_normalization(old_value, new_value)
    old_pass = _passes_any_regex(old_value, regexes)
    new_pass = _passes_any_regex(new_value, regexes)
    detail = {
        "normalization_kind": kind,
        "normalization_rule": kind,
        "old_value": old_value,
        "new_value": new_value,
        "old_changed": old_value,
        "new_changed": new_value,
        "regexes_present": bool(regexes),
        "old_pass_regex": old_pass,
        "new_pass_regex": new_pass,
    }
    if regexes and old_pass is True and new_pass is False:
        return "bad_target", {**detail, "reason": "old_value_passes_regex_new_value_fails"}
    if regexes and old_pass is False and new_pass is True and simple:
        return "format_normalization", detail
    if simple and (not regexes or new_pass is not False):
        return "format_normalization", detail
    return "not_deterministic", detail


def _format_normalization_detail(
    repair_event: Dict[str, Any],
    truth_tokens: List[str],
    truth_source: str,
) -> Tuple[bool, Dict[str, Any]]:
    pre_val, pre_src = _pre_repair_value(repair_event)
    pre_tokens = flatten_truth(pre_val)
    if len(pre_tokens) != 1 or len(truth_tokens) != 1:
        return False, {
            "pre_repair_source": pre_src,
            "truth_source": truth_source,
            "reason": "format_repair_requires_single_old_and_new_value",
        }
    simple, kind = _is_simple_format_normalization(pre_tokens[0], truth_tokens[0])
    return simple, {
        "pre_repair_source": pre_src,
        "truth_source": truth_source,
        "normalization_kind": kind,
        "old_value": pre_tokens[0],
        "new_value": truth_tokens[0],
    }


def _format_normalization_from_delta(value_change: Any, world_state_entry: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    regexes = _format_regexes(world_state_entry)
    if value_change.semantic_action == "REPLACE_1_TO_1":
        if len(value_change.old_unique) != 1 or len(value_change.new_unique) != 1:
            return "not_deterministic", {"reason": "format_repair_requires_single_old_and_new_value"}
        old_value = value_change.old_unique[0]
        new_value = value_change.new_unique[0]
        return _format_pair_detail(old_value, new_value, regexes)
    if value_change.semantic_action == "MIXED_UPDATE":
        if len(value_change.added_unique_values) == 1 and len(value_change.removed_unique_values) == 1:
            old_value = value_change.removed_unique_values[0]
            new_value = value_change.added_unique_values[0]
            status, detail = _format_pair_detail(old_value, new_value, regexes)
            return status, {**detail, "mixed_update": True, "retained_values": value_change.retained_unique_values}
    return "not_deterministic", {"reason": "not_supported_format_delta", "semantic_action": value_change.semantic_action}


def rule_deterministic_classification(
    repair_event: Dict[str, Any],
    world_state_entry: Dict[str, Any],
    truth_tokens: List[str],
    truth_source: str,
) -> Tuple[bool, str, str, Dict[str, Any]]:
    """
    Returns (is_deterministic, subtype, confidence, detail)
    subtype values: FORMAT, RANGE, ONE_OF
    """
    constraints = extract_constraint_entries(world_state_entry)
    if not constraints:
        # fallback: format report type is still rule-driven
        if report_type_normalized(repair_event) == "format":
            simple, detail = _format_normalization_detail(repair_event, truth_tokens, truth_source)
            if simple:
                detail["signal"] = "report_type"
                return True, "FORMAT", "high", detail
            return False, "", "", {"signal": "report_type", **detail}
        return False, "", "", {"signal": "none"}

    pre_val, _ = _pre_repair_value(repair_event)
    pre_tokens = flatten_truth(pre_val)

    for c in constraints:
        ctype = c.get("constraint_type", {}) if isinstance(c, dict) else {}
        kind = constraint_kind(ctype)
        if kind == "format":
            simple, detail = _format_normalization_detail(repair_event, truth_tokens, truth_source)
            if simple:
                return True, "FORMAT", "high", {"signal": "L4_constraints", "constraint_type": ctype, **detail}
            continue

        if kind == "one_of":
            allowed = _extract_allowed_values(c)
            if len(allowed) == 1:
                allowed_val = allowed[0]
                if not truth_tokens:
                    return (
                        True,
                        "ONE_OF",
                        "medium",
                        {
                            "signal": "L4_constraints",
                            "constraint_type": ctype,
                            "allowed_count": 1,
                            "allowed_value": allowed_val,
                            "truth_source": truth_source,
                        },
                    )
                if any(tok == allowed_val for tok in truth_tokens):
                    return (
                        True,
                        "ONE_OF",
                        "high",
                        {
                            "signal": "L4_constraints",
                            "constraint_type": ctype,
                            "allowed_count": 1,
                            "allowed_value": allowed_val,
                        },
                    )
            continue

        if kind == "range":
            if len(truth_tokens) != 1:
                continue
            tok = truth_tokens[0]
            numeric_min_raws, numeric_max_raws, date_min_raws, date_max_raws = _extract_range_bounds(c)
            min_dates = [d for d in (_parse_date_boundary(x) for x in date_min_raws) if d]
            max_dates = [d for d in (_parse_date_boundary(x) for x in date_max_raws) if d]
            min_nums = [n for n in (_parse_numeric_token(x) for x in numeric_min_raws) if n is not None]
            max_nums = [n for n in (_parse_numeric_token(x) for x in numeric_max_raws) if n is not None]

            if DATE_ISO_RE.match(str(tok)):
                if tok in min_dates or tok in max_dates:
                    return True, "RANGE", "high", {"signal": "L4_constraints", "constraint_type": ctype}
            num_tok = _parse_numeric_token(str(tok))
            if num_tok is not None:
                if num_tok in min_nums or num_tok in max_nums:
                    return True, "RANGE", "high", {"signal": "L4_constraints", "constraint_type": ctype}
            rule_summary = c.get("rule_summary") if isinstance(c, dict) else None
            if isinstance(rule_summary, str) and truth_tokens and str(tok) in rule_summary:
                if pre_tokens:
                    return (
                        True,
                        "RANGE",
                        "medium",
                        {
                            "signal": "rule_summary",
                            "constraint_type": ctype,
                            "truth_source": truth_source,
                        },
                    )
            continue

    return False, "", "", {"signal": "L4_constraints", "constraint_type": None}


def _has_constraint_kind(constraint_types: List[Dict[str, Optional[str]]], kinds: Iterable[str]) -> bool:
    wanted = set(kinds)
    for ctype in constraint_types:
        if constraint_kind({"qid": ctype.get("qid"), "label": ctype.get("label_en")}) in wanted:
            return True
    return False


def _format_constraint_entries(world_state_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        entry
        for entry in extract_constraint_entries(world_state_entry)
        if constraint_kind(entry.get("constraint_type", {}) if isinstance(entry, dict) else {}) == "format"
    ]


def _format_regexes(world_state_entry: Dict[str, Any]) -> List[str]:
    regexes: List[str] = []
    for entry in _format_constraint_entries(world_state_entry):
        qualifiers = entry.get("qualifiers")
        if not isinstance(qualifiers, list):
            continue
        for qualifier in qualifiers:
            if not isinstance(qualifier, dict) or qualifier.get("property_id") != "P1793":
                continue
            values = qualifier.get("values")
            if isinstance(values, list):
                for value in values:
                    regexes.extend([token for token in flatten_truth(value) if token not in (None, "")])
            elif values not in (None, ""):
                regexes.extend([token for token in flatten_truth(values) if token not in (None, "")])
    return regexes


def _passes_any_regex(value: str, regexes: List[str]) -> Optional[bool]:
    if not regexes:
        return None
    for pattern in regexes:
        try:
            if re.fullmatch(pattern, value):
                return True
        except re.error:
            continue
    return False


def _violation_report_values(repair_event: Dict[str, Any]) -> List[str]:
    vc = repair_event.get("violation_context", {})
    if not isinstance(vc, dict):
        return []
    out: List[str] = []
    for key in ("value", "report_value", "violation_value"):
        out.extend(flatten_truth(vc.get(key)))
    seen = set()
    uniq: List[str] = []
    for value in out:
        if value not in seen:
            seen.add(value)
            uniq.append(value)
    return uniq


def _format_value_pruning_detail(
    repair_event: Dict[str, Any],
    world_state_entry: Dict[str, Any],
    value_change: Any,
) -> Tuple[bool, str, Dict[str, Any]]:
    if value_change.semantic_action != "DELETE_SUBSET" or not _is_format_report(repair_event):
        return False, "", {"reason": "not_format_delete_subset"}
    removed = list(value_change.removed_unique_values)
    retained = list(value_change.retained_unique_values)
    report_values = set(_violation_report_values(repair_event))
    regexes = _format_regexes(world_state_entry)
    removed_reported = bool(removed) and set(removed).issubset(report_values)
    removed_fail_regex = all(_passes_any_regex(value, regexes) is False for value in removed) if regexes else False
    retained_pass_regex = all(_passes_any_regex(value, regexes) is True for value in retained) if regexes else None
    if regexes and retained_pass_regex is not True:
        ok = False
        reason = "retained_values_do_not_all_pass_format_regex"
    elif removed_reported:
        ok = True
        reason = "removed_values_reported_as_offending_values"
    elif removed_fail_regex and retained_pass_regex is True:
        ok = True
        reason = "removed_values_fail_and_retained_values_pass_format_regex"
    else:
        ok = False
        reason = "removed_values_not_verified_as_format_failures"
    confidence = "high" if removed_fail_regex and retained_pass_regex is True else "medium"
    return ok, confidence, {
        "reason": reason,
        "removed_values": removed,
        "retained_values": retained,
        "report_values": sorted(report_values),
        "regexes_present": bool(regexes),
        "removed_reported": removed_reported,
        "removed_fail_regex": removed_fail_regex,
        "retained_pass_regex": retained_pass_regex,
    }


def _set_membership_rejection_detail(
    repair_event: Dict[str, Any],
    world_state_entry: Dict[str, Any],
    value_change: Any,
) -> Tuple[bool, Dict[str, Any]]:
    if not _is_set_membership_report(repair_event) or value_change.semantic_action not in {"DELETE_SUBSET", "DELETE_TO_MISSING"}:
        return False, {"reason": "not_set_membership_delete"}
    report = report_type_normalized(repair_event)
    removed = set(value_change.removed_unique_values or value_change.old_unique)
    allowed_or_forbidden: set[str] = set()
    for entry in extract_constraint_entries(world_state_entry):
        ctype = entry.get("constraint_type", {}) if isinstance(entry, dict) else {}
        kind = constraint_kind(ctype)
        qid = ctype.get("qid") if isinstance(ctype, dict) else None
        if report == "one of" and kind == "one_of":
            allowed_or_forbidden.update(_extract_allowed_values(entry))
        elif report == "none of" and qid in NONE_OF_QIDS:
            allowed_or_forbidden.update(_extract_allowed_values(entry))
    if not removed or not allowed_or_forbidden:
        return False, {"removed_values": sorted(removed), "known_set": sorted(allowed_or_forbidden)}
    if report == "one of":
        ok = removed.isdisjoint(allowed_or_forbidden)
    else:
        ok = removed.issubset(allowed_or_forbidden)
    return ok, {"report_type": report, "removed_values": sorted(removed), "known_set": sorted(allowed_or_forbidden)}


def classify_delete_action(
    repair_event: Dict[str, Any],
    constraint_types: List[Dict[str, Optional[str]]],
    value_change: Any = None,
    world_state_entry: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, str, Dict[str, Any]]:
    report_type = report_type_normalized(repair_event)
    removed_values = []
    if value_change is not None:
        removed_values = list(value_change.removed_unique_values or value_change.old_unique)

    if _is_self_link_report(repair_event) and repair_event.get("qid") in removed_values:
        return (
            "SELF_LINK_REJECTION",
            "high",
            "A-Box DELETE removes the focus entity from a self-link violation.",
            {"report_type": report_type, "delete_reason": "self_link", "removed_values": removed_values},
        )

    if report_type == "format":
        return (
            "REJECTION_FORMAT_INVALID",
            "high",
            "A-Box DELETE removes a value reported as a format violation.",
            {"report_type": report_type, "delete_reason": "format_invalid_report"},
        )

    if world_state_entry is not None:
        ok, detail = _set_membership_rejection_detail(repair_event, world_state_entry, value_change)
        if ok:
            return (
                "SET_MEMBERSHIP_REJECTION",
                "medium",
                "A-Box DELETE removes a value proven invalid by one-of/none-of set membership.",
                {"report_type": report_type, "delete_reason": "set_membership", **detail},
            )

    rule_invalid_reports = {"range", "type", "value type", "diff within range", "quantity"}
    if report_type in rule_invalid_reports:
        return (
            "REJECTION_RULE_INVALID",
            "medium",
            "A-Box DELETE removes a value identified as invalid by a supported rule family.",
            {"report_type": report_type, "delete_reason": "rule_invalid"},
        )

    if report_type in {"single value", "unique value"}:
        return (
            "DELETE_AMBIGUOUS",
            "low",
            "A-Box DELETE under a single/unique-value conflict is not automatically logical because value selection may require evidence.",
            {"report_type": report_type, "delete_reason": "selection_conflict"},
        )

    return (
        "DELETE_AMBIGUOUS",
        "low",
        "A-Box DELETE lacks enough rule/local evidence to classify the deletion target confidently.",
        {"report_type": report_type, "delete_reason": "insufficient_delete_evidence"},
    )


def _first_trace_detail(trace: List[Dict[str, Any]], key: str) -> Any:
    for step in trace:
        if not isinstance(step, dict):
            continue
        detail = step.get("detail")
        if isinstance(detail, dict) and key in detail:
            return detail.get(key)
        if key in step:
            return step.get(key)
    return None


def _constraint_label_from_types(qid: Optional[str], ctypes: List[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(qid, str):
        return None
    for ctype in ctypes:
        if isinstance(ctype, dict) and ctype.get("qid") == qid:
            label = ctype.get("label_en") or ctype.get("label")
            if isinstance(label, str):
                return label
    return _constraint_label(qid)


def _rule_metadata_for_classification(
    cls: str,
    subtype: str,
    trace: List[Dict[str, Any]],
    ctypes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    family = "unknown"
    subfamily = subtype.lower() if isinstance(subtype, str) else "unknown"
    qid: Optional[str] = None
    source = "classifier_rule"
    if subtype in {"FORMAT_NORMALIZATION", "FORMAT_VALUE_PRUNING", "REJECTION_FORMAT_INVALID"}:
        family = "format"
        qid = "Q21502404"
        subfamily = (
            str(_first_trace_detail(trace, "normalization_rule") or _first_trace_detail(trace, "normalization_kind") or subtype.lower())
            if subtype == "FORMAT_NORMALIZATION"
            else subtype.lower()
        )
    elif subtype == "SET_MEMBERSHIP_REJECTION":
        family = "set_membership"
        report = str(_first_trace_detail(trace, "report_type") or "")
        subfamily = normalize_text(report).replace(" ", "_") if report else "set_membership"
        qid = "Q52558054" if report == "none of" else "Q21510859"
    elif subtype == "SELF_LINK_REJECTION":
        family = "self_link_report"
        subfamily = "self_link_rejection"
        qid = None
        source = "violation_report"
    elif subtype == "TARGET_REQUIRED_CLAIM":
        family = "target_required_claim"
        qid = "Q21510864"
    elif subtype == "MULTIPLICITY_NORMALIZATION":
        family = "multiplicity"
        qid = "Q19474404"
    elif cls == "TypeB" and subtype in {"LOCAL_TEXT_CONFIRMED", "LOCAL_SELECTION_CONFIRMED", "LOCAL_FOCUS_QID", "LOCAL_TEXT_DERIVED"}:
        family = "local_evidence"
        subfamily = {
            "LOCAL_TEXT_CONFIRMED": "local_text_raw",
            "LOCAL_SELECTION_CONFIRMED": "local_selection_confirmed",
            "LOCAL_FOCUS_QID": "focus_qid",
            "LOCAL_TEXT_DERIVED": "local_text_derived",
        }.get(subtype, subtype.lower())
        source = "local_context"
    elif cls == "TypeC" and subtype == "EXTERNAL_BY_ELIMINATION":
        family = "negative_rule_and_local_scan"
        subfamily = "external_by_elimination"
    elif cls == "TypeC" and isinstance(subtype, str) and subtype.startswith("UNKNOWN_"):
        family = "diagnostic_unknown"
        subfamily = subtype.lower()
    elif cls == "T_BOX":
        family = "tbox_schema_causality"
        subfamily = subtype.lower()
        qid = _first_trace_detail(trace, "target_constraint_qid") or _first_trace_detail(trace, "mapped_violation_constraint_qid")
        source = str(_first_trace_detail(trace, "target_constraint_selection_reason") or "tbox_causality_filter")
    elif cls == "TypeA":
        family = "rule_or_logical"
        subfamily = subtype.lower()
    return {
        "decision_constraint_type_qid": qid,
        "decision_constraint_type_label": _constraint_label_from_types(qid, ctypes),
        "decision_constraint_source": source,
        "classification_rule_family": family,
        "classification_rule_subfamily": subfamily,
    }


def local_context_is_sparse(buckets: Dict[str, Any]) -> bool:
    text_field_count = 0
    for key in (
        "text_focus_label_fields",
        "text_focus_description_fields",
        "text_focus_alias_fields",
        "text_focus_prerepair_literal_fields",
        "text_neighbor_label_fields",
        "text_neighbor_description_fields",
        "text_focus_property_fields",
        "text_l2_referenced_fields",
    ):
        value = buckets.get(key)
        if isinstance(value, list):
            text_field_count += len([field for field in value if str(field).strip()])
    id_count = 0
    for key in ("ids_neighbors", "ids_focus_qid", "ids_focus_prerepair", "ids_focus_non_target"):
        value = buckets.get(key)
        if isinstance(value, set):
            id_count += len(value)
    return id_count == 0 and text_field_count <= 1


def classify_one(
    repair_event: Dict[str, Any],
    world_state_entry: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
    """
    Returns:
      - classification block (per proposed schema)
      - error string (if classification couldn't be computed)
      - diagnostics dict for stats accounting
    """

    # Default classification for failure modes
    def make(
        cls: str, subtype: str, conf: str, trace: List[Dict[str, Any]], rationale: str, ctypes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        classification = {
            "class": cls,
            "subtype": subtype,
            "confidence": conf,
            "decision_trace": trace,
            "rationale": rationale,
            "constraint_types": ctypes,
        }
        classification.update(_rule_metadata_for_classification(cls, subtype, trace, ctypes))
        return classification

    truth_tokens, truth_source, truth_applicable = get_truth_info(repair_event)
    value_change = derive_value_change_summary(repair_event)
    target_tokens = _classification_target_tokens(value_change)
    diagnostics = {
        "truth_applicable": truth_applicable,
        "truth_tokens": truth_tokens,
        "truth_source": truth_source,
        "value_change_summary": value_change.as_dict(),
        "classification_target_tokens": target_tokens,
    }

    if not isinstance(world_state_entry, dict):
        # no context => we cannot safely make a negative local-evidence claim.
        trace = [
            {"step": "is_delete", "result": None},
            {"step": "rule_deterministic", "result": None},
            {"step": "local_availability", "result": "missing_world_state"},
            {"step": "fallback_external", "result": True},
            {"step": "branch", "result": "missing_world_state"},
        ]
        classification = make(
            "TypeC",
            "UNKNOWN_MISSING_WORLD_STATE",
            "low",
            trace,
            "Missing world-state entry; cannot claim externality from absent local context.",
            [],
        )
        classification["local_subtype"] = None
        classification["diagnostics"] = diagnostics
        return (
            classification,
            "missing_world_state",
            {"missing_truth_tokens": truth_applicable and not truth_tokens, "missing_old_value": False},
        )

    constraint_types = extract_constraint_types(world_state_entry)

    track = repair_event.get("track")
    rt = repair_event.get("repair_target", {}) if isinstance(repair_event.get("repair_target"), dict) else {}

    # T-BOX: taxonomy does not strictly apply (different task: rule drift detection).
    if track == "T_BOX" or rt.get("kind") == "T_BOX":
        differ = ConstraintDiffer(repair_event, rt.get("constraint_delta"))
        schema_subtype, trace, rationale = differ.classify_change()
        conf = "medium"
        if schema_subtype in {
            "RELAXATION_RANGE_WIDENED",
            "RESTRICTION_RANGE_NARROWED",
            "RELAXATION_SET_EXPANSION",
            "RESTRICTION_SET_CONTRACTION",
        }:
            conf = "high"
        if schema_subtype in {"COINCIDENTAL_SCHEMA_CHANGE", "UNKNOWN_TBOX_CAUSALITY"}:
            conf = "low"
        classification = make(
            "T_BOX",
            schema_subtype,
            conf,
            trace,
            rationale,
            constraint_types,
        )
        classification["local_subtype"] = None
        classification["analysis_slice_precise"] = _first_trace_detail(trace, "analysis_slice_precise") or ""
        classification["diagnostics"] = diagnostics
        classification["diagnostics"]["tbox_diff_summary"] = _compact_tbox_diff_summary(trace)
        return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    # A-BOX delete => Type A rejection
    action = rt.get("action")
    if action == "DELETE":
        subtype, conf, rationale, detail = classify_delete_action(
            repair_event,
            constraint_types,
            value_change=value_change,
            world_state_entry=world_state_entry,
        )
        trace = [
            {"step": "is_delete", "result": True},
            {"step": "delete_classification", "result": subtype, "detail": detail},
            {"step": "rule_deterministic", "result": None},
            {"step": "local_availability", "result": False},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "delete_refined"},
        ]
        classification = make(
            "TypeA",
            subtype,
            conf,
            trace,
            rationale,
            constraint_types,
        )
        classification["local_subtype"] = None
        classification["diagnostics"] = diagnostics
        return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    if value_change.semantic_action == "MULTIPLICITY_DECREASE_SAME_UNIQUE":
        is_cardinality = _is_cardinality_or_duplicate_report(repair_event, constraint_types)
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
            {"step": "rule_deterministic", "result": is_cardinality, "kind": "MULTIPLICITY"},
            {"step": "local_availability", "result": None},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "multiplicity_normalization" if is_cardinality else "unknown_multiplicity_artifact"},
        ]
        if not is_cardinality:
            classification = make(
                "TypeC",
                "UNKNOWN_MULTIPLICITY_ARTIFACT",
                "low",
                trace,
                "Unique values are unchanged and multiplicity decreases, but the violation report is not cardinality or duplicate related.",
                constraint_types,
            )
            classification["local_subtype"] = None
            classification["diagnostics"] = diagnostics
            return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}
        classification = make(
            "TypeA",
            "MULTIPLICITY_NORMALIZATION",
            "high",
            trace,
            "Unique values are unchanged and duplicate multiplicity decreases.",
            constraint_types,
        )
        classification["local_subtype"] = None
        classification["diagnostics"] = diagnostics
        return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    if value_change.semantic_action in {"MULTIPLICITY_INCREASE_SAME_UNIQUE", "MULTIPLICITY_CHANGE_SAME_UNIQUE", "NO_CHANGE_OR_REORDER_ONLY"}:
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
            {"step": "rule_deterministic", "result": False},
            {"step": "local_availability", "result": None},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "unknown_multiplicity_artifact"},
        ]
        classification = make(
            "TypeC",
            "UNKNOWN_MULTIPLICITY_ARTIFACT",
            "low",
            trace,
            "Unique values are unchanged; the repair appears to be a multiplicity or reconstruction artifact.",
            constraint_types,
        )
        classification["local_subtype"] = None
        classification["diagnostics"] = diagnostics
        return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    target_required_pid = _target_required_claim_pid(repair_event)
    if (
        target_required_pid
        and value_change.semantic_action == "CREATE_FROM_MISSING"
        and len(value_change.new_unique) == 1
        and value_change.new_unique[0] == repair_event.get("qid")
    ):
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
            {
                "step": "rule_deterministic",
                "result": True,
                "kind": "TARGET_REQUIRED_CLAIM",
                "detail": {"target_required_property": target_required_pid},
            },
            {"step": "local_availability", "result": None},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "target_required_claim"},
        ]
        classification = make(
            "TypeA",
            "TARGET_REQUIRED_CLAIM",
            "high" if target_required_pid != "UNKNOWN" else "medium",
            trace,
            "Target-required-claim violation deterministically requires the focus entity as the target value.",
            constraint_types,
        )
        classification["local_subtype"] = None
        classification["diagnostics"] = diagnostics
        return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    if _is_self_link_report(repair_event) and repair_event.get("qid") in value_change.added_unique_values:
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
            {"step": "rule_deterministic", "result": False, "kind": "SELF_LINK_BAD_TARGET"},
            {"step": "local_availability", "result": None},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "unknown_bad_target_or_context"},
        ]
        classification = make(
            "TypeC",
            "UNKNOWN_BAD_TARGET_OR_CONTEXT",
            "low",
            trace,
            "Repair adds the focus entity under a self-link violation; this is likely a bad target or insufficient context, not local evidence.",
            constraint_types,
        )
        classification["local_subtype"] = None
        classification["diagnostics"] = diagnostics
        return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    if value_change.semantic_action == "DELETE_SUBSET" and _is_self_link_report(repair_event) and repair_event.get("qid") in value_change.removed_unique_values:
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
            {"step": "rule_deterministic", "result": True, "kind": "SELF_LINK"},
            {"step": "local_availability", "result": None},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "self_link_rejection"},
        ]
        classification = make(
            "TypeA",
            "SELF_LINK_REJECTION",
            "high",
            trace,
            "Subset repair removes the focus entity from a self-link violation.",
            constraint_types,
        )
        classification["local_subtype"] = None
        classification["diagnostics"] = diagnostics
        return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    if value_change.semantic_action == "DELETE_SUBSET":
        ok, fmt_conf, fmt_detail = _format_value_pruning_detail(repair_event, world_state_entry, value_change)
        if ok:
            trace = [
                {"step": "is_delete", "result": False},
                {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
                {"step": "rule_deterministic", "result": True, "kind": "FORMAT_VALUE_PRUNING", "detail": fmt_detail},
                {"step": "local_availability", "result": None},
                {"step": "fallback_external", "result": False},
                {"step": "branch", "result": "format_value_pruning"},
            ]
            classification = make(
                "TypeA",
                "FORMAT_VALUE_PRUNING",
                fmt_conf,
                trace,
                "Subset repair removes the value indicated by a format violation.",
                constraint_types,
            )
            classification["local_subtype"] = None
            classification["diagnostics"] = diagnostics
            return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}
        if _is_format_report(repair_event) and fmt_detail.get("reason") == "retained_values_do_not_all_pass_format_regex":
            trace = [
                {"step": "is_delete", "result": False},
                {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
                {"step": "rule_deterministic", "result": False, "kind": "FORMAT_VALUE_PRUNING", "detail": fmt_detail},
                {"step": "local_availability", "result": None},
                {"step": "fallback_external", "result": False},
                {"step": "branch", "result": "unknown_format_pruning_retained_unverified"},
            ]
            classification = make(
                "TypeC",
                "UNKNOWN_FORMAT_PRUNING_RETAINED_UNVERIFIED",
                "low",
                trace,
                "Format subset pruning removed invalid-looking values, but retained values were not verified against the format regex.",
                constraint_types,
            )
            classification["local_subtype"] = None
            classification["diagnostics"] = diagnostics
            return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}
        ok_set, set_detail = _set_membership_rejection_detail(repair_event, world_state_entry, value_change)
        if ok_set:
            trace = [
                {"step": "is_delete", "result": False},
                {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
                {"step": "rule_deterministic", "result": True, "kind": "SET_MEMBERSHIP_REJECTION", "detail": set_detail},
                {"step": "local_availability", "result": None},
                {"step": "fallback_external", "result": False},
                {"step": "branch", "result": "set_membership_rejection"},
            ]
            classification = make(
                "TypeA",
                "SET_MEMBERSHIP_REJECTION",
                "medium",
                trace,
                "Subset repair removes a value proven invalid by parsed one-of/none-of constraints.",
                constraint_types,
            )
            classification["local_subtype"] = None
            classification["diagnostics"] = diagnostics
            return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    fmt_status, fmt_detail = _format_normalization_from_delta(value_change, world_state_entry)
    strong_nonreport_format_normalization = (
        fmt_status == "format_normalization"
        and fmt_detail.get("normalization_rule")
        in {"strip_schembl_prefix", "strip_category_prefix", "extract_url_slug"}
    )
    if _is_format_report(repair_event) or strong_nonreport_format_normalization:
        if fmt_status == "format_normalization":
            trace = [
                {"step": "is_delete", "result": False},
                {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
                {"step": "rule_deterministic", "result": True, "kind": "FORMAT_NORMALIZATION", "detail": fmt_detail},
                {"step": "local_availability", "result": None},
                {"step": "fallback_external", "result": False},
                {"step": "branch", "result": "format_normalization"},
            ]
            classification = make(
                "TypeA",
                "FORMAT_NORMALIZATION",
                "high" if _is_format_report(repair_event) else "medium",
                trace,
                "One-to-one literal update is a deterministic format normalization.",
                constraint_types,
            )
            classification["local_subtype"] = None
            classification["diagnostics"] = diagnostics
            return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}
        if fmt_status == "bad_target":
            trace = [
                {"step": "is_delete", "result": False},
                {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
                {"step": "rule_deterministic", "result": False, "kind": "FORMAT_NORMALIZATION", "detail": fmt_detail},
                {"step": "local_availability", "result": None},
                {"step": "fallback_external", "result": False},
                {"step": "branch", "result": "unknown_bad_target_or_context"},
            ]
            classification = make(
                "TypeC",
                "UNKNOWN_BAD_TARGET_OR_CONTEXT",
                "low",
                trace,
                "Format update moves from a regex-valid value to a regex-invalid value; target or context is suspect.",
                constraint_types,
            )
            classification["local_subtype"] = None
            classification["diagnostics"] = diagnostics
            return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    if (
        _is_single_value_report(repair_event)
        and value_change.semantic_action in {"CREATE_FROM_MISSING", "ADD_SUPERSET", "MIXED_UPDATE"}
        and len(value_change.new_unique) > 1
    ):
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
            {
                "step": "rule_deterministic",
                "result": False,
                "kind": "SINGLE_VALUE_MULTIPLE_NEW_VALUES",
                "detail": {"report_type": report_type_normalized(repair_event), "new_unique": value_change.new_unique},
            },
            {"step": "local_availability", "result": None},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "single_value_report_multiple_new_values"},
        ]
        classification = make(
            "TypeC",
            "UNKNOWN_BAD_TARGET_OR_CONTEXT",
            "low",
            trace,
            "Single-value violation is followed by multiple created/added target values; treated as report-context mismatch or bad target rather than clean external evidence.",
            constraint_types,
        )
        classification["local_subtype"] = None
        classification["diagnostics"] = diagnostics
        return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    classification_tokens = [str(token) for token in target_tokens.get("tokens", []) if token not in (None, "")]
    if not classification_tokens and value_change.semantic_action not in {"DELETE_TO_MISSING", "DELETE_SUBSET"}:
        classification_tokens = truth_tokens
    missing_truth = len(truth_tokens) == 0 and truth_applicable

    if truth_source in CURRENT_VALUE_TRUTH_SOURCES:
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "truth_source", "result": truth_source},
            {"step": "rule_deterministic", "result": None},
            {"step": "local_availability", "result": None},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "current_value_truth_fallback"},
        ]
        classification = make(
            "TypeC",
            "UNKNOWN_CURRENT_VALUE_FALLBACK",
            "low",
            trace,
            "Historical repair truth is missing; only current snapshot value is available, so the case is quarantined.",
            constraint_types,
        )
        classification["local_subtype"] = None
        classification["diagnostics"] = diagnostics
        return (
            classification,
            None,
            {"missing_truth_tokens": False, "missing_old_value": False, "current_value_truth_fallback": True},
        )

    if truth_applicable and missing_truth:
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "truth_source", "result": truth_source},
            {"step": "rule_deterministic", "result": None},
            {"step": "local_availability", "result": None},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "missing_truth"},
        ]
        classification = make(
            "TypeC",
            "UNKNOWN_MISSING_TRUTH",
            "low",
            trace,
            "No usable historical repair truth tokens are available for classification.",
            constraint_types,
        )
        classification["local_subtype"] = None
        classification["diagnostics"] = diagnostics
        return (
            classification,
            None,
            {"missing_truth_tokens": True, "missing_old_value": False},
        )

    # Rule-deterministic check (precedes local)
    det, det_kind, det_conf, det_detail = rule_deterministic_classification(
        repair_event, world_state_entry, classification_tokens, target_tokens.get("role", truth_source)
    )
    if det:
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "rule_deterministic", "result": True, "detail": det_detail, "kind": det_kind},
            {"step": "local_availability", "result": None},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "rule_deterministic"},
        ]
        classification = make(
            "TypeA",
            "FORMAT_NORMALIZATION" if det_kind == "FORMAT" else "LOGICAL",
            det_conf,
            trace,
            f"Rule-deterministic {det_kind.lower()} constraint fix.",
            constraint_types,
        )
        classification["local_subtype"] = None
        classification["diagnostics"] = diagnostics
        return (
            classification,
            None,
            {"missing_truth_tokens": truth_applicable and missing_truth, "missing_old_value": False},
        )

    derived_detail = _p8726_local_text_derived_detail(repair_event, world_state_entry, value_change)
    if derived_detail:
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
            {"step": "rule_deterministic", "result": False, "detail": det_detail},
            {
                "step": "local_text_derived",
                "result": True,
                "detail": derived_detail,
                "independent_of_target_property": True,
            },
            {"step": "local_availability", "result": True, "evidence": {"matches": [derived_detail]}},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "local_text_derived"},
        ]
        classification = make(
            "TypeB",
            "LOCAL_TEXT_DERIVED",
            "high" if derived_detail.get("raw_matched_text") else "medium",
            trace,
            "Target literal derived from independent local text by deterministic property-specific transformation.",
            constraint_types,
        )
        classification["local_subtype"] = "LOCAL_TEXT_DERIVED"
        classification["diagnostics"] = diagnostics
        return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    buckets, synth_info = local_context_buckets(repair_event, world_state_entry)

    if value_change.semantic_action == "DELETE_SUBSET":
        retained_tokens = list(value_change.retained_unique_values)
        retained_matched, retained_evidence = match_truth_locally(retained_tokens, buckets)
        if _is_selection_report(repair_event) and retained_matched and evidence_has_independent_support(retained_evidence):
            trace = [
                {"step": "is_delete", "result": False},
                {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
                {"step": "rule_deterministic", "result": False, "detail": det_detail},
                {"step": "local_availability", "result": True, "evidence": retained_evidence, "synthetic": synth_info},
                {"step": "fallback_external", "result": False},
                {"step": "branch", "result": "local_selection_confirmed"},
            ]
            classification = make(
                "TypeB",
                "LOCAL_SELECTION_CONFIRMED",
                "medium",
                trace,
                "Retained value in a subset repair has independent local support; pre-repair target values alone are not counted.",
                constraint_types,
            )
            classification["local_subtype"] = "LOCAL_SELECTION_CONFIRMED"
            classification["diagnostics"] = diagnostics
            return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}
        if retained_matched and evidence_only_prerepair_target_property(retained_evidence):
            branch = "unknown_selection_ambiguous"
            trace = [
                {"step": "is_delete", "result": False},
                {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
                {"step": "rule_deterministic", "result": False, "detail": det_detail},
                {"step": "local_availability", "result": False, "evidence": retained_evidence, "synthetic": synth_info},
                {"step": "fallback_external", "result": False},
                {"step": "branch", "result": branch},
            ]
            classification = make(
                "TypeC",
                "UNKNOWN_SELECTION_AMBIGUOUS",
                "low",
                trace,
                "Subset repair only shows retained values in the pre-repair target property; this is not independent local grounding.",
                constraint_types,
            )
            classification["local_subtype"] = None
            classification["diagnostics"] = diagnostics
            return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    matched, evidence = match_truth_locally(classification_tokens, buckets)
    local_subtype = local_match_subtype(evidence.get("matches", []))

    if matched and evidence_only_prerepair_target_property(evidence):
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
            {"step": "rule_deterministic", "result": False, "detail": det_detail},
            {"step": "local_availability", "result": False, "evidence": evidence, "synthetic": synth_info},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "pre_repair_target_only_not_local"},
        ]
        classification = make(
            "TypeC",
            "UNKNOWN_SELECTION_AMBIGUOUS" if value_change.semantic_action in {"DELETE_SUBSET", "MIXED_UPDATE"} else "UNKNOWN_INCOMPLETE_LOCAL_CONTEXT",
            "low",
            trace,
            "Only synthetic pre-repair target-property values matched; this is not independent local evidence.",
            constraint_types,
        )
        classification["local_subtype"] = None
        classification["diagnostics"] = diagnostics
        return classification, None, {"missing_truth_tokens": False, "missing_old_value": synth_info.get("pre_repair_source") == "missing"}

    if matched:
        conf = "high"
        if evidence.get("used_literal_substring"):
            conf = "medium"
        if isinstance(evidence.get("needed"), int) and evidence["needed"] > 3:
            conf = "medium"
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "value_delta", "result": value_change.semantic_action, "detail": value_change.as_dict()},
            {"step": "rule_deterministic", "result": False, "detail": det_detail},
            {"step": "local_availability", "result": True, "evidence": evidence, "synthetic": synth_info},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "local_match"},
        ]
        if local_subtype == "LOCAL_FOCUS_QID" and repair_event.get("qid") in classification_tokens:
            trace[-1] = {"step": "branch", "result": "unknown_focus_qid_domain_reasoning"}
            classification = make(
                "TypeC",
                "UNKNOWN_FOCUS_QID_DOMAIN_REASONING",
                "low",
                trace,
                "Focus QID is locally available, but local identity alone does not justify asserting this property value; correctness requires a domain/property-specific rule or external/domain reasoning.",
                constraint_types,
            )
            classification["local_subtype"] = None
            classification["diagnostics"] = diagnostics
            return classification, None, {"missing_truth_tokens": False, "missing_old_value": synth_info.get("pre_repair_source") == "missing"}
        if local_subtype == "LOCAL_FOCUS_QID":
            rationale = "Repair target matched the focus entity id."
        elif local_subtype == "LOCAL_FOCUS_PREREPAIR_PROPERTY":
            rationale = "Truth tokens matched synthetic pre-repair target property values."
        elif local_subtype == "LOCAL_FOCUS_NON_TARGET_PROPERTY":
            rationale = "Truth tokens matched non-target focus-node property values."
        elif local_subtype == "LOCAL_NEIGHBOR_IDS":
            rationale = "Truth tokens matched neighbor identifiers."
        elif local_subtype == "LOCAL_L2_REFERENCED_TEXT":
            rationale = "Truth tokens matched labels or descriptions for locally referenced ids."
        elif local_subtype in {"LOCAL_TEXT", "LOCAL_TEXT_CONFIRMED"}:
            rationale = "Truth tokens matched independent local text context."
        else:
            rationale = "Truth tokens matched multiple local sources."
        classification = make(
            "TypeB",
            local_subtype or "LOCAL_MIXED",
            conf,
            trace,
            rationale,
            constraint_types,
        )
        classification["local_subtype"] = local_subtype
        classification["diagnostics"] = diagnostics
        return (
            classification,
            None,
            {
                "missing_truth_tokens": truth_applicable and missing_truth,
                "missing_old_value": synth_info.get("pre_repair_source") == "missing",
            },
        )

    # Default: external by elimination or unknown if local context is too sparse for a negative claim.
    sparse_local_context = local_context_is_sparse(buckets)
    fallback_subtype = "UNKNOWN_INCOMPLETE_LOCAL_CONTEXT" if sparse_local_context else "EXTERNAL_BY_ELIMINATION"
    conf = "low" if sparse_local_context else "medium"
    trace = [
        {"step": "is_delete", "result": False},
        {"step": "rule_deterministic", "result": False, "detail": det_detail},
        {"step": "local_availability", "result": False, "evidence": evidence, "synthetic": synth_info},
        {"step": "fallback_external", "result": True},
        {"step": "branch", "result": "incomplete_local_context" if sparse_local_context else "external_by_elimination"},
    ]
    rationale = (
        "Local context is too sparse to claim externality."
        if sparse_local_context
        else "Historical truth exists, but supported rule and local-evidence checks did not find it; externality is by elimination."
    )
    classification = make(
        "TypeC",
        fallback_subtype,
        conf,
        trace,
        rationale,
        constraint_types,
    )
    classification["local_subtype"] = None
    classification["diagnostics"] = diagnostics
    return (
        classification,
        None,
        {
            "missing_truth_tokens": truth_applicable and missing_truth,
            "missing_old_value": synth_info.get("pre_repair_source") == "missing",
        },
    )


def build_labels_en(repair_event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "qid": {
            "label": repair_event.get("qid_label_en"),
            "description": repair_event.get("qid_description_en"),
        },
        "property": {
            "label": repair_event.get("property_label_en"),
            "description": repair_event.get("property_description_en"),
        },
    }


def lean_repair_target(repair_target: Any) -> Any:
    if not isinstance(repair_target, dict):
        return repair_target
    if repair_target.get("kind") != "T_BOX":
        return repair_target
    out = {key: value for key, value in repair_target.items() if key != "constraint_delta"}
    delta = repair_target.get("constraint_delta")
    if isinstance(delta, dict):
        keep_keys = {
            "revision_id",
            "property_revision_id",
            "property_revision_new",
            "property_revision_prev",
            "hash_before",
            "hash_after",
            "changed_constraint_types",
            "added_constraint_types",
            "removed_constraint_types",
        }
        out["constraint_delta"] = {key: delta.get(key) for key in keep_keys if key in delta}
        out["constraint_delta"]["lean_stage4_note"] = (
            "Full T-box signatures are omitted from lean Stage 4; use data/02 and data/03 "
            "or classification.tbox_causality trace fields for detailed inspection."
        )
    return out


def ensure_popularity(
    repair_event: Dict[str, Any],
    popularity_by_qid: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if isinstance(repair_event.get("popularity"), dict):
        return repair_event["popularity"]
    qid = repair_event.get("qid")
    if popularity_by_qid and isinstance(qid, str) and qid in popularity_by_qid:
        return popularity_by_qid[qid]
    return None


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")


class WorldStateStore:
    """Disk-backed lookup for world_state entries keyed by repair id."""

    def __init__(self, world_state_path: Path, log: logging.Logger):
        self.world_state_path = Path(world_state_path)
        self.db_path = self.world_state_path.with_suffix(self.world_state_path.suffix + ".sqlite")
        self.log = log
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> "WorldStateStore":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self._ensure_schema()
        if self._needs_rebuild():
            self._rebuild()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _ensure_schema(self) -> None:
        assert self.conn is not None
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS world_state (
              id TEXT PRIMARY KEY,
              payload TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def _meta_get(self, key: str) -> Optional[str]:
        assert self.conn is not None
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        return row[0]

    def _meta_set(self, key: str, value: str) -> None:
        assert self.conn is not None
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _source_signature(self) -> str:
        stat = self.world_state_path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def _needs_rebuild(self) -> bool:
        if not self.world_state_path.exists():
            raise FileNotFoundError(f"World state file not found: {self.world_state_path}")
        sig = self._source_signature()
        cached_sig = self._meta_get("source_signature")
        cached_count = self._meta_get("entry_count")
        if cached_sig != sig:
            return True
        if cached_count is None:
            return True
        return False

    def _rebuild(self) -> None:
        assert self.conn is not None
        self.log.info("Building world-state index: source=%s db=%s", self.world_state_path, self.db_path)
        start = time.monotonic()
        self.conn.execute("DELETE FROM world_state")
        self.conn.execute("DELETE FROM meta")
        self.conn.commit()

        batch: List[Tuple[str, str]] = []
        inserted = 0
        last_heartbeat = start
        batch_size = 1000
        heartbeat_seconds = 30

        with open(self.world_state_path, "rb") as fh:
            for rid, payload in ijson.kvitems(fh, ""):
                if not isinstance(rid, str):
                    continue
                batch.append((rid, json.dumps(payload, ensure_ascii=False, default=_json_default)))
                if len(batch) >= batch_size:
                    self.conn.executemany(
                        "INSERT INTO world_state(id, payload) VALUES (?, ?)",
                        batch,
                    )
                    inserted += len(batch)
                    batch.clear()
                    now = time.monotonic()
                    if now - last_heartbeat >= heartbeat_seconds:
                        elapsed = now - start
                        rate = inserted / elapsed if elapsed > 0 else 0.0
                        self.log.info(
                            "World-state index heartbeat: inserted=%s elapsed=%.0fs rate=%.1f rows/s",
                            f"{inserted:,}",
                            elapsed,
                            rate,
                        )
                        last_heartbeat = now

        if batch:
            self.conn.executemany("INSERT INTO world_state(id, payload) VALUES (?, ?)", batch)
            inserted += len(batch)
            batch.clear()

        self._meta_set("source_signature", self._source_signature())
        self._meta_set("entry_count", str(inserted))
        self.conn.commit()
        elapsed = time.monotonic() - start
        self.log.info("World-state index ready: entries=%s elapsed=%.0fs", f"{inserted:,}", elapsed)

    def __len__(self) -> int:
        count = self._meta_get("entry_count")
        if count is None:
            return 0
        try:
            return int(count)
        except Exception:
            return 0

    def get(self, rid: str) -> Optional[Dict[str, Any]]:
        assert self.conn is not None
        row = self.conn.execute("SELECT payload FROM world_state WHERE id = ?", (rid,)).fetchone()
        if not row:
            return None
        payload = row[0]
        if not isinstance(payload, str):
            return None
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
        return None


def _run_self_tests() -> None:
    # Synthetic context should not treat target property new_value as local evidence.
    repair_event = {
        "id": "R1",
        "qid": "Q10",
        "property": "P1",
        "track": "A_BOX",
        "repair_target": {"kind": "A_BOX", "action": "UPDATE", "new_value": "Q99", "old_value": "Q1"},
        "violation_context": {},
    }
    world_state_entry = {
        "L1_ego_node": {"label": "X", "description": "Y"},
        "L3_neighborhood": {
            "outgoing_edges": [
                {"property_id": "P1", "target_qid": "Q99", "target_label": "New Value"},
                {"property_id": "P2", "target_qid": "Q2", "target_label": "Other"},
            ]
        },
    }
    buckets, synth_info = local_context_buckets(repair_event, world_state_entry)
    assert "Q99" not in buckets["ids_neighbors"], "new_value leaked into neighbor ids"
    assert "Q99" not in buckets["ids_focus_prerepair"], "new_value leaked into prerepair ids"
    assert "Q1" in buckets["ids_focus_prerepair"], "old_value missing from synthetic prerepair ids"
    matched, _ = match_truth_locally(["Q99"], buckets)
    assert not matched, "new_value should not match local context via target property"
    assert synth_info["used_pre_repair_value"], "synthetic pre-repair value not used"

    # ISO date precision: full date must match exactly.
    local_text = normalize_text("born 1999-01-02")
    matched, _ = match_truth_locally(
        ["1999-01-02"],
        {"ids_neighbors": set(), "ids_focus_prerepair": set(), "text_focus": local_text, "text_neighbors": ""},
    )
    assert matched, "ISO date should match exact full date"
    matched, _ = match_truth_locally(
        ["1999-01-03"],
        {"ids_neighbors": set(), "ids_focus_prerepair": set(), "text_focus": local_text, "text_neighbors": ""},
    )
    assert not matched, "ISO date should not match a different date"

    # Range widening should be detected via numeric bounds.
    range_before = [{"property_id": "P2313", "values": ["1900"]}, {"property_id": "P2312", "values": ["2000"]}]
    range_after = [{"property_id": "P2313", "values": ["1850"]}, {"property_id": "P2312", "values": ["2000"]}]
    assert analyze_range_change(range_before, range_after) == "RELAXATION_RANGE_WIDENED", (
        "Range widening not detected"
    )

    # Causality filter should require the mapped constraint QID.
    assert is_causal_repair("Unique value", ["Q21502410"]) is True, "Causality filter rejected valid mapping"
    assert is_causal_repair("Unique value", ["Q21502404"]) is False, "Causality filter allowed mismatched mapping"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="Use data_sample/ inputs/outputs instead of data/")
    ap.add_argument("--self-test", action="store_true", help="Run minimal self-tests and exit")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    ap.add_argument("--quiet", action="store_true", help="Show only warnings and errors")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    ap.add_argument("--repairs-path", type=Path, help="Override repairs input path.")
    ap.add_argument("--world-state-path", type=Path, help="Override world-state input path.")
    ap.add_argument("--popularity-path", type=Path, help="Override popularity input path.")
    ap.add_argument("--out-path", type=Path, help="Override lean classified JSONL output path.")
    ap.add_argument("--out-full-path", type=Path, help="Override full classified JSONL output path.")
    ap.add_argument("--no-full-output", action="store_true", help="Do not write the full embedded world-state output.")
    ap.add_argument("--stats-path", type=Path, help="Override classifier stats output path.")
    args = ap.parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)
    log = logging.getLogger("classifier")

    if args.self_test:
        log.info("Running self-tests")
        _run_self_tests()
        log.info("Self-tests passed")
        return 0

    # Set up paths
    FOLDER_PATH = Path("data/")
    if args.sample:
        FOLDER_PATH = Path("data_sample/")
    repairs_path = args.repairs_path or (FOLDER_PATH / DEFAULT_REPAIRS_PATH)
    world_state_path = args.world_state_path or (FOLDER_PATH / DEFAULT_WORLD_STATE_PATH)
    popularity_path = args.popularity_path or (FOLDER_PATH / DEFAULT_POPULARITY_PATH)
    out_path = args.out_path or (FOLDER_PATH / DEFAULT_OUT_PATH)
    out_full_path = args.out_full_path or (FOLDER_PATH / DEFAULT_OUT_FULL_PATH if DEFAULT_OUT_FULL_PATH else None)
    if args.no_full_output:
        out_full_path = None
    stats_path = args.stats_path or Path(DEFAULT_STATS_PATH)
    use_progress = (not args.no_progress) and sys.stderr.isatty()
    log.info("Starting classifier run")
    log.info("Inputs: repairs=%s world_state=%s", repairs_path, world_state_path)
    if popularity_path:
        log.info("Optional popularity input=%s", popularity_path)
    log.info("Outputs: lean=%s full=%s stats=%s", out_path, out_full_path, stats_path)

    world_state_store = WorldStateStore(world_state_path, log)
    world_state_store.open()
    try:
        log.info("Loaded world state entries: %d", len(world_state_store))

        popularity_by_qid: Optional[Dict[str, Any]] = None
        if popularity_path and popularity_path.exists():
            try:
                popularity_by_qid = read_json(popularity_path)
                if not isinstance(popularity_by_qid, dict):
                    log.warning("Popularity file is not a dict: %s", popularity_path)
                    popularity_by_qid = None
            except Exception:
                log.warning("Failed to read popularity file: %s", popularity_path)
                popularity_by_qid = None
        elif popularity_path:
            log.info("Popularity file not found (continuing without it): %s", popularity_path)
        if popularity_by_qid is not None:
            log.info("Loaded popularity entries: %d", len(popularity_by_qid))

        stats = {
            "build_utc": utc_now_iso(),
            "inputs": {
                "repairs": str(repairs_path),
                "world_state": str(world_state_path),
                "popularity": str(popularity_path) if popularity_by_qid is not None else None,
            },
            "counts": Counter(),
            "errors": Counter(),
            "decision_trace": Counter(),
        }

        def iter_outputs() -> Iterator[Dict[str, Any]]:
            repairs_iter: Iterable[Dict[str, Any]] = iter_repairs(repairs_path)
            total_repairs = count_repairs(repairs_path) if use_progress else None
            if total_repairs is None:
                log.info("Repairs to process: progress disabled; skipped input pre-count")
            else:
                log.info("Repairs to process: %d", total_repairs)
            repairs_iter = tqdm(
                repairs_iter,
                desc="Classifying",
                unit="records",
                total=total_repairs,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:,}/{total:,}{unit} [{elapsed}<{remaining}, {rate_fmt}]",
                disable=not use_progress,
            )
            for repair_event in repairs_iter:
                if not isinstance(repair_event, dict):
                    continue

                rid = repair_event.get("id")
                if not isinstance(rid, str) or not rid:
                    stats["errors"]["missing_id"] += 1
                    continue

                ws_entry = world_state_store.get(rid)
                if ws_entry is None:
                    stats["errors"]["missing_world_state"] += 1
                    classification, err, diag = classify_one(repair_event, None)
                else:
                    classification, err, diag = classify_one(repair_event, ws_entry)

                if err:
                    stats["errors"][err] += 1
                if diag.get("missing_truth_tokens"):
                    stats["errors"]["missing_truth_tokens"] += 1
                if diag.get("missing_old_value"):
                    stats["errors"]["missing_old_value"] += 1

                pop = ensure_popularity(repair_event, popularity_by_qid)
                if pop is None:
                    stats["errors"]["missing_popularity"] += 1
                    # Keep going; popularity can be optional in early iterations

                out = {
                    "id": rid,
                    "qid": repair_event.get("qid"),
                    "property": repair_event.get("property"),
                    "track": repair_event.get("track"),
                    "information_type": repair_event.get("information_type", None),
                    "labels_en": build_labels_en(repair_event),
                    "violation_context": repair_event.get("violation_context"),
                    "repair_target": lean_repair_target(repair_event.get("repair_target")),
                    "persistence_check": repair_event.get("persistence_check"),
                    "popularity": pop,
                    "context_ref": {
                        "world_state_id": rid,
                        "world_state_path": str(world_state_path),
                    },
                    "classification": classification,
                    "build": {
                        "classifier_version": "0.1.0",
                        "built_at_utc": stats["build_utc"],
                    },
                }

                # Counters
                c = classification.get("class")
                s = classification.get("subtype")
                conf = classification.get("confidence")
                stats["counts"][f"class:{c}"] += 1
                stats["counts"][f"subtype:{s}"] += 1
                stats["counts"][f"confidence:{conf}"] += 1
                stats["counts"][f"track:{out.get('track')}"] += 1
                local_subtype = classification.get("local_subtype")
                if isinstance(local_subtype, str) and local_subtype:
                    stats["counts"][f"local_subtype:{local_subtype}"] += 1
                diag_block = classification.get("diagnostics", {})
                if isinstance(diag_block, dict):
                    ts = diag_block.get("truth_source")
                    if isinstance(ts, str):
                        stats["counts"][f"truth_source:{ts}"] += 1
                        if ts in CURRENT_VALUE_TRUTH_SOURCES:
                            stats["counts"]["truth_source:current_value_fallback_total"] += 1
                for step in classification.get("decision_trace", []):
                    if not isinstance(step, dict):
                        continue
                    if step.get("step") == "branch":
                        stats["decision_trace"][str(step.get("result"))] += 1

                stats["counts"]["output_records"] += 1
                yield out

        # Write lean output (streaming)
        log.info("Writing lean output")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            for rec in iter_outputs():
                fh.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
        log.info("Lean output complete: %s", out_path)

        # Write optional full output (second pass; acceptable for first draft)
        if out_full_path:
            log.info("Writing full output")

            def iter_full() -> Iterator[Dict[str, Any]]:
                full_iter: Iterable[Dict[str, Any]] = iter_jsonl(out_path)
                full_iter = tqdm(
                    full_iter,
                    desc="Embedding world state",
                    unit="records",
                    total=stats["counts"].get("output_records"),
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:,}/{total:,}{unit} [{elapsed}<{remaining}, {rate_fmt}]",
                    disable=not use_progress,
                )
                for rec in full_iter:
                    rid = rec.get("id")
                    ws_entry = world_state_store.get(rid)
                    rec2 = dict(rec)
                    rec2["world_state"] = ws_entry
                    yield rec2

            out_full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_full_path, "w", encoding="utf-8") as fh:
                for rec in iter_full():
                    fh.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
            log.info("Full output complete: %s", out_full_path)

        # Write stats
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_out = dict(stats)
        stats_out["counts"] = dict(stats["counts"])
        stats_out["errors"] = dict(stats["errors"])
        stats_out["decision_trace"] = dict(stats["decision_trace"])
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats_out, fh, ensure_ascii=False, indent=2, default=_json_default)
        log.info("Stats written: %s", stats_path)
        log.info(
            "Completed classifier run: output_records=%d errors=%d",
            stats["counts"].get("output_records", 0),
            sum(stats["errors"].values()),
        )
    finally:
        world_state_store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
