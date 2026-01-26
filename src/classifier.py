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
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

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

DATE_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Default paths
DEFAULT_REPAIRS_PATH = "02_wikidata_repairs.json"
DEFAULT_WORLD_STATE_PATH = "03_world_state.json"
DEFAULT_POPULARITY_PATH = "00_entity_popularity.json"
DEFAULT_OUT_PATH = "04_classified_benchmark.jsonl"
DEFAULT_OUT_FULL_PATH = "04_classified_benchmark_full.jsonl"
DEFAULT_STATS_PATH = "reports/classifier_stats.json"

# Constraint QID families (observed in outputs)
FORMAT_QIDS = {"Q21502404"}  # format constraint
ONE_OF_QIDS = {"Q21510859", "Q21502402"}  # one-of constraint (observed + legacy)
RANGE_QIDS = {"Q21510860"}  # range constraint
TYPE_QIDS = {"Q21503250"}  # type constraint

VIOLATION_TO_CONSTRAINT_MAP = {
    "Single value": "Q19474404",
    "Unique value": "Q21502410",
    "Format": "Q21502404",
    "One of": "Q21510859",
    "Type": "Q21503250",
    "Value type": "Q21510865",
    "Range": "Q21510860",
    "Diff within range": "Q21510861",
    "Quantity": "Q21510857",
}


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
    if qid in FORMAT_QIDS:
        return "format"
    if qid in ONE_OF_QIDS:
        return "one_of"
    if qid in RANGE_QIDS:
        return "range"
    if isinstance(label, str):
        ln = normalize_text(label)
        if ln in {"format constraint", "format"}:
            return "format"
        if ln in {"one of constraint", "one-of constraint", "one of", "one-of"}:
            return "one_of"
        if ln in {"range constraint", "range"}:
            return "range"
    return None


def is_causal_repair(violation_name: Optional[str], changed_constraint_qids: Iterable[str]) -> bool:
    if not isinstance(violation_name, str) or not violation_name.strip():
        return True
    if changed_constraint_qids is None:
        changed_constraint_qids = []
    normalized = normalize_text(violation_name)
    mapping = {normalize_text(k): v for k, v in VIOLATION_TO_CONSTRAINT_MAP.items()}
    target_qid = mapping.get(normalized)
    if not target_qid:
        return True
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


def _collect_qualifier_values(qualifiers: List[Dict[str, Any]], property_id: Optional[str] = None) -> List[str]:
    out: List[str] = []
    for q in qualifiers:
        if not isinstance(q, dict):
            continue
        if property_id and q.get("property_id") != property_id:
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


def analyze_range_change(old_qualifiers: List[Dict[str, Any]], new_qualifiers: List[Dict[str, Any]]) -> str:
    old_min = _extract_numeric_bound(old_qualifiers, "P2313", "min")
    old_max = _extract_numeric_bound(old_qualifiers, "P2312", "max")
    new_min = _extract_numeric_bound(new_qualifiers, "P2313", "min")
    new_max = _extract_numeric_bound(new_qualifiers, "P2312", "max")

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
    property_id: str,
) -> str:
    old_values = set(_collect_qualifier_values(old_qualifiers, property_id=property_id))
    new_values = set(_collect_qualifier_values(new_qualifiers, property_id=property_id))
    return _compare_sets(old_values, new_values)


def _analyze_generic_set_change(
    old_qualifiers: List[Dict[str, Any]],
    new_qualifiers: List[Dict[str, Any]],
) -> str:
    old_values = set(_collect_qualifier_values(old_qualifiers))
    new_values = set(_collect_qualifier_values(new_qualifiers))
    return _compare_sets(old_values, new_values)


class ConstraintDiffer:
    def __init__(self, repair_event: Dict[str, Any], constraint_delta: Optional[Dict[str, Any]]):
        self.repair_event = repair_event
        self.delta = constraint_delta if isinstance(constraint_delta, dict) else {}
        self.changed_constraint_qids = [
            qid for qid in _ensure_list(self.delta.get("changed_constraint_types")) if isinstance(qid, str)
        ]
        self.signature_before = _load_signature_list(self.delta, "signature_before", "old_constraints")
        self.signature_after = _load_signature_list(self.delta, "signature_after", "new_constraints")

    def _violation_name(self) -> Optional[str]:
        vc = self.repair_event.get("violation_context", {})
        if not isinstance(vc, dict):
            return None
        name = vc.get("report_violation_type_normalized") or vc.get("report_violation_type")
        return name if isinstance(name, str) else None

    def _candidate_constraint_qids(self) -> List[str]:
        qids = []
        for entry in self.signature_before + self.signature_after:
            if not isinstance(entry, dict):
                continue
            qid = entry.get("constraint_qid")
            if isinstance(qid, str) and qid not in qids:
                qids.append(qid)
        return qids

    def _select_target_qid(self, mapped_qid: Optional[str]) -> Optional[str]:
        if isinstance(mapped_qid, str):
            candidates = set(self.changed_constraint_qids) | set(self._candidate_constraint_qids())
            if mapped_qid in candidates:
                return mapped_qid
        if self.changed_constraint_qids:
            return self.changed_constraint_qids[0]
        candidates = self._candidate_constraint_qids()
        return candidates[0] if candidates else None

    def classify_change(self) -> Tuple[str, List[Dict[str, Any]], str]:
        trace: List[Dict[str, Any]] = []
        violation_name = self._violation_name()
        mapping = {normalize_text(k): v for k, v in VIOLATION_TO_CONSTRAINT_MAP.items()}
        mapped_qid = mapping.get(normalize_text(violation_name)) if isinstance(violation_name, str) else None
        causal = is_causal_repair(violation_name, self.changed_constraint_qids)
        trace.append(
            {
                "step": "causality_filter",
                "result": causal,
                "violation_name": violation_name,
                "mapped_constraint_qid": mapped_qid,
                "changed_constraint_qids": self.changed_constraint_qids,
            }
        )
        if not causal:
            return (
                "COINCIDENTAL_SCHEMA_CHANGE",
                trace,
                "Violation type did not map to the changed constraint types; treated as coincidental schema change.",
            )

        target_qid = self._select_target_qid(mapped_qid)
        trace.append({"step": "target_constraint", "result": target_qid})
        if not target_qid:
            return "SCHEMA_UPDATE", trace, "No constraint type selected; defaulted to schema update."

        old_qualifiers = _collect_qualifiers_for_qid(self.signature_before, target_qid)
        new_qualifiers = _collect_qualifiers_for_qid(self.signature_after, target_qid)

        if target_qid in RANGE_QIDS:
            result = analyze_range_change(old_qualifiers, new_qualifiers)
            trace.append({"step": "range_semantics", "result": result})
            rationale = "Range constraint qualifiers compared using numeric bounds."
            return result, trace, rationale

        if target_qid in ONE_OF_QIDS or target_qid in TYPE_QIDS:
            result = _analyze_set_change(old_qualifiers, new_qualifiers, property_id="P2305")
            trace.append({"step": "set_semantics", "result": result, "property_id": "P2305"})
            rationale = "Constraint values compared as a set for one-of/type constraints."
            return result, trace, rationale

        result = _analyze_generic_set_change(old_qualifiers, new_qualifiers)
        trace.append({"step": "generic_set_semantics", "result": result})
        rationale = "Constraint qualifiers compared with generic set semantics."
        return result, trace, rationale


def _parse_numeric_token(token: str) -> Optional[Decimal]:
    if not isinstance(token, str):
        return None
    token = token.strip()
    if not token:
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
        if pid != "P2305":
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


def _extract_range_bounds(constraint: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    mins: List[str] = []
    maxs: List[str] = []
    qualifiers = constraint.get("qualifiers")
    if not isinstance(qualifiers, list):
        return mins, maxs
    for q in qualifiers:
        if not isinstance(q, dict):
            continue
        pid = q.get("property_id")
        if pid not in {"P2310", "P2311"}:
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
            if pid == "P2310":
                mins.append(raw)
            else:
                maxs.append(raw)
    return mins, maxs


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
    For A-BOX UPDATE/CREATE: truth is the post-repair value with fallbacks.
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

    candidates = [
        ("repair_target.new_value", rt.get("new_value")),
        ("repair_target.value", rt.get("value")),
        ("persistence_check.current_value_2026", safe_get(repair_event, "persistence_check", "current_value_2026")),
        ("violation_context.value_current_2026", safe_get(repair_event, "violation_context", "value_current_2026")),
    ]
    for src, v in candidates:
        toks = flatten_truth(v)
        if toks:
            return toks, src, True
    return [], "missing_unexpected", True


def _pre_repair_value(repair_event: Dict[str, Any]) -> Tuple[Any, str]:
    rt = repair_event.get("repair_target", {})
    if isinstance(rt, dict) and "old_value" in rt:
        return rt.get("old_value"), "repair_target.old_value"
    vc_val = safe_get(repair_event, "violation_context", "value")
    if vc_val is not None:
        return vc_val, "violation_context.value"
    return None, "missing"


def local_context_buckets(
    repair_event: Dict[str, Any],
    world_state_entry: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds separate buckets for local matching provenance:
      - ids_neighbors: neighbor QIDs
      - ids_focus_prerepair: pre-repair target-property QIDs on focus node
      - text_focus: focus node label/description + pre-repair literals
      - text_neighbors: neighbor labels/descriptions
    """
    ids_neighbors: set = set()
    ids_focus_prerepair: set = set()
    text_focus: List[str] = []
    text_neighbors: List[str] = []
    synth_info = {"used_pre_repair_value": False, "pre_repair_source": "missing", "tokens": []}

    # Focus node text
    q_label = repair_event.get("qid_label_en") or safe_get(world_state_entry, "L1_ego_node", "label")
    q_desc = repair_event.get("qid_description_en") or safe_get(world_state_entry, "L1_ego_node", "description")
    for x in (q_label, q_desc):
        if isinstance(x, str) and x.strip():
            text_focus.append(x)

    # Neighborhood
    edges = safe_get(world_state_entry, "L3_neighborhood", "outgoing_edges", default=[])
    target_pid = repair_event.get("property") if is_pid(repair_event.get("property")) else None
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
            tl = e.get("target_label")
            td = e.get("target_description")
            if isinstance(tl, str) and tl.strip():
                text_neighbors.append(tl)
            if isinstance(td, str) and td.strip():
                text_neighbors.append(td)

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
                else:
                    text_focus.append(str(tok))
        else:
            synth_info["pre_repair_source"] = pre_src
    else:
        synth_info["pre_repair_source"] = "not_applicable"

    buckets = {
        "ids_neighbors": ids_neighbors,
        "ids_focus_prerepair": ids_focus_prerepair,
        "text_focus": normalize_text(" \n ".join(text_focus)),
        "text_neighbors": normalize_text(" \n ".join(text_neighbors)),
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


def match_truth_locally(truth_tokens: List[str], buckets: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns:
      - matched: bool
      - evidence: dict with counts and examples
    Matching policy:
      - QIDs: must appear in ids_neighbors or ids_focus_prerepair
      - ISO dates: exact substring on normalized text
      - Other literals: exact substring on normalized text
    """
    if not truth_tokens:
        return False, {"matched": False, "needed": 0, "found": 0, "matches": []}

    needed = 0
    found = 0
    matches: List[Dict[str, Any]] = []
    used_literal_substring = False
    sources_used: set = set()

    ids_neighbors = buckets.get("ids_neighbors", set())
    ids_focus_prerepair = buckets.get("ids_focus_prerepair", set())
    text_focus = buckets.get("text_focus", "")
    text_neighbors = buckets.get("text_neighbors", "")

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

        if is_qid(tok_s):
            if tok_s in ids_focus_prerepair:
                found += 1
                matches.append({"token": tok_s, "kind": "qid", "source": "FOCUS_PREREPAIR_PROPERTY"})
                sources_used.add("FOCUS_PREREPAIR_PROPERTY")
            elif tok_s in ids_neighbors:
                found += 1
                matches.append({"token": tok_s, "kind": "qid", "source": "NEIGHBOR_IDS"})
                sources_used.add("NEIGHBOR_IDS")
            continue

        # literal/date matching on normalized text
        if DATE_ISO_RE.match(tok_s):
            tok_n = normalize_text(tok_s)
            if tok_n in text_focus:
                found += 1
                matches.append({"token": tok_s, "kind": "date", "source": "FOCUS_TEXT"})
                sources_used.add("FOCUS_TEXT")
            elif tok_n in text_neighbors:
                found += 1
                matches.append({"token": tok_s, "kind": "date", "source": "NEIGHBOR_TEXT"})
                sources_used.add("NEIGHBOR_TEXT")
            continue

        tok_n = normalize_text(tok_s)
        if tok_n and tok_n in text_focus:
            found += 1
            used_literal_substring = True
            matches.append({"token": tok_s, "kind": "literal", "source": "FOCUS_TEXT"})
            sources_used.add("FOCUS_TEXT")
        elif tok_n and tok_n in text_neighbors:
            found += 1
            used_literal_substring = True
            matches.append({"token": tok_s, "kind": "literal", "source": "NEIGHBOR_TEXT"})
            sources_used.add("NEIGHBOR_TEXT")

    matched = needed > 0 and found == needed
    evidence = {
        "matched": matched,
        "needed": needed,
        "found": found,
        "matches": matches,
        "used_literal_substring": used_literal_substring,
        "sources_used": sorted(sources_used),
        "local_ids_count": len(ids_neighbors) + len(ids_focus_prerepair),
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
    if src == "NEIGHBOR_IDS":
        return "LOCAL_NEIGHBOR_IDS"
    if src == "FOCUS_PREREPAIR_PROPERTY":
        return "LOCAL_FOCUS_PREREPAIR_PROPERTY"
    if src in {"FOCUS_TEXT", "NEIGHBOR_TEXT"}:
        return "LOCAL_TEXT"
    return "LOCAL_MIXED"


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
            return True, "FORMAT", "high", {"signal": "report_type"}
        return False, "", "", {"signal": "none"}

    pre_val, _ = _pre_repair_value(repair_event)
    pre_tokens = flatten_truth(pre_val)

    for c in constraints:
        ctype = c.get("constraint_type", {}) if isinstance(c, dict) else {}
        kind = constraint_kind(ctype)
        if kind == "format":
            return True, "FORMAT", "high", {"signal": "L4_constraints", "constraint_type": ctype}

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
            min_raws, max_raws = _extract_range_bounds(c)
            min_dates = [d for d in (_parse_date_boundary(x) for x in min_raws) if d]
            max_dates = [d for d in (_parse_date_boundary(x) for x in max_raws) if d]
            min_nums = [n for n in (_parse_numeric_token(x) for x in min_raws) if n is not None]
            max_nums = [n for n in (_parse_numeric_token(x) for x in max_raws) if n is not None]

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
        return {
            "class": cls,
            "subtype": subtype,
            "confidence": conf,
            "decision_trace": trace,
            "rationale": rationale,
            "constraint_types": ctypes,
        }

    truth_tokens, truth_source, truth_applicable = get_truth_info(repair_event)
    diagnostics = {
        "truth_applicable": truth_applicable,
        "truth_tokens": truth_tokens,
        "truth_source": truth_source,
    }

    if not isinstance(world_state_entry, dict):
        # no context => we cannot safely do local check; treat as external with low confidence
        trace = [
            {"step": "is_delete", "result": None},
            {"step": "rule_deterministic", "result": None},
            {"step": "local_availability", "result": "missing_world_state"},
            {"step": "fallback_external", "result": True},
            {"step": "branch", "result": "missing_world_state"},
        ]
        classification = make(
            "TypeC", "EXTERNAL", "low", trace, "Missing world_state entry; defaulted to external.", []
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
        if schema_subtype == "COINCIDENTAL_SCHEMA_CHANGE":
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
        classification["diagnostics"] = diagnostics
        return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    # A-BOX delete => Type A rejection
    action = rt.get("action")
    if action == "DELETE":
        trace = [
            {"step": "is_delete", "result": True},
            {"step": "rule_deterministic", "result": None},
            {"step": "local_availability", "result": False},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "delete_rejection"},
        ]
        classification = make(
            "TypeA",
            "REJECTION",
            "high",
            trace,
            "A-Box DELETE action treated as logical rejection/cleaning.",
            constraint_types,
        )
        classification["local_subtype"] = None
        classification["diagnostics"] = diagnostics
        return classification, None, {"missing_truth_tokens": False, "missing_old_value": False}

    missing_truth = len(truth_tokens) == 0

    # Rule-deterministic check (precedes local)
    det, det_kind, det_conf, det_detail = rule_deterministic_classification(
        repair_event, world_state_entry, truth_tokens, truth_source
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
            "LOGICAL",
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

    buckets, synth_info = local_context_buckets(repair_event, world_state_entry)
    matched, evidence = match_truth_locally(truth_tokens, buckets)
    local_subtype = local_match_subtype(evidence.get("matches", []))

    if matched:
        conf = "high"
        if evidence.get("used_literal_substring"):
            conf = "medium"
        if isinstance(evidence.get("needed"), int) and evidence["needed"] > 3:
            conf = "medium"
        trace = [
            {"step": "is_delete", "result": False},
            {"step": "rule_deterministic", "result": False, "detail": det_detail},
            {"step": "local_availability", "result": True, "evidence": evidence, "synthetic": synth_info},
            {"step": "fallback_external", "result": False},
            {"step": "branch", "result": "local_match"},
        ]
        if local_subtype == "LOCAL_FOCUS_PREREPAIR_PROPERTY":
            rationale = "Truth tokens matched synthetic pre-repair target property values."
        elif local_subtype == "LOCAL_NEIGHBOR_IDS":
            rationale = "Truth tokens matched neighbor identifiers."
        elif local_subtype == "LOCAL_TEXT":
            rationale = "Truth tokens matched local text context."
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

    # Default: external
    conf = "medium"
    if truth_applicable and missing_truth:
        conf = "low"
    trace = [
        {"step": "is_delete", "result": False},
        {"step": "rule_deterministic", "result": False, "detail": det_detail},
        {"step": "local_availability", "result": False, "evidence": evidence, "synthetic": synth_info},
        {"step": "fallback_external", "result": True},
        {"step": "branch", "result": "external_fallback"},
    ]
    classification = make(
        "TypeC",
        "EXTERNAL",
        conf,
        trace,
        "Truth not found locally and rule is not deterministic; treated as external.",
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
    args = ap.parse_args()

    if args.self_test:
        _run_self_tests()
        return 0

    # Set up paths
    FOLDER_PATH = Path("data/")
    if args.sample:
        FOLDER_PATH = Path("data_sample/")
    repairs_path = FOLDER_PATH / DEFAULT_REPAIRS_PATH
    world_state_path = FOLDER_PATH / DEFAULT_WORLD_STATE_PATH
    popularity_path = FOLDER_PATH / DEFAULT_POPULARITY_PATH
    out_path = FOLDER_PATH / DEFAULT_OUT_PATH
    out_full_path = FOLDER_PATH / DEFAULT_OUT_FULL_PATH if DEFAULT_OUT_FULL_PATH else None
    stats_path = Path(DEFAULT_STATS_PATH)

    # Load world state (keyed by repair id)
    world_state = read_json(world_state_path)
    if not isinstance(world_state, dict):
        raise ValueError(f"Expected dict in {world_state_path}, got {type(world_state)}")

    popularity_by_qid: Optional[Dict[str, Any]] = None
    if popularity_path and popularity_path.exists():
        try:
            popularity_by_qid = read_json(popularity_path)
            if not isinstance(popularity_by_qid, dict):
                popularity_by_qid = None
        except Exception:
            popularity_by_qid = None

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
        total_repairs = count_repairs(repairs_path)
        repairs_iter = tqdm(
            repairs_iter,
            desc="Classifying",
            unit="records",
            total=total_repairs,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:,}/{total:,}{unit} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for repair_event in repairs_iter:
            if not isinstance(repair_event, dict):
                continue

            rid = repair_event.get("id")
            if not isinstance(rid, str) or not rid:
                stats["errors"]["missing_id"] += 1
                continue

            ws_entry = world_state.get(rid)
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
                "repair_target": repair_event.get("repair_target"),
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
            stats["counts"][f"class:{c}"] += 1
            stats["counts"][f"subtype:{s}"] += 1
            stats["counts"][f"track:{out.get('track')}"] += 1
            diag_block = classification.get("diagnostics", {})
            if isinstance(diag_block, dict):
                ts = diag_block.get("truth_source")
                if isinstance(ts, str):
                    stats["counts"][f"truth_source:{ts}"] += 1
            for step in classification.get("decision_trace", []):
                if not isinstance(step, dict):
                    continue
                if step.get("step") == "branch":
                    stats["decision_trace"][str(step.get("result"))] += 1

            yield out

    # Write lean output (streaming)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in iter_outputs():
            fh.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")

    # Write optional full output (second pass; acceptable for first draft)
    if out_full_path:

        def iter_full() -> Iterator[Dict[str, Any]]:
            for rec in iter_jsonl(out_path):
                rid = rec.get("id")
                ws_entry = world_state.get(rid)
                rec2 = dict(rec)
                rec2["world_state"] = ws_entry
                yield rec2

        out_full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_full_path, "w", encoding="utf-8") as fh:
            for rec in iter_full():
                fh.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")

    # Write stats
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_out = dict(stats)
    stats_out["counts"] = dict(stats["counts"])
    stats_out["errors"] = dict(stats["errors"])
    stats_out["decision_trace"] = dict(stats["decision_trace"])
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats_out, fh, ensure_ascii=False, indent=2, default=_json_default)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
